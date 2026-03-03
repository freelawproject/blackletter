"""Scan a PDF with YOLO and produce annotated page images."""

from __future__ import annotations

import io
import logging
import re
import subprocess
import tempfile
from pathlib import Path

import cv2
import fitz
import numpy as np
from PIL import Image
from ultralytics import YOLO

from blackletter.models import BBox, Detection, Document, Label, Page

logger = logging.getLogger(__name__)

DPI = 200
CONFIDENCE_THRESHOLD = 0.20

# Per-label minimum confidence for drawing/output (overrides default).
LABEL_CONFIDENCE: dict[Label, float] = {
    Label.KEY_ICON: 0.50,
    Label.CASE_CAPTION: 0.50,
    Label.PAGE_HEADER: 0.50,
    Label.PAGE_NUMBER: 0.75,
    Label.HEADNOTE: 0.50,
    Label.HEADNOTE_BRACKET: 0.50,
    Label.BACKGROUND: 0.50,
}

# Colors per label (RGB)
LABEL_COLORS: dict[Label, tuple[int, int, int]] = {
    Label.KEY_ICON: (231, 76, 60),
    Label.DIVIDER: (149, 165, 166),
    Label.PAGE_HEADER: (230, 126, 34),
    Label.CASE_CAPTION: (52, 152, 219),
    Label.FOOTNOTES: (26, 188, 156),
    Label.HEADNOTE_BRACKET: (155, 89, 182),
    Label.CASE_METADATA: (46, 204, 113),
    Label.CASE_SEQUENCE: (243, 156, 18),
    Label.PAGE_NUMBER: (96, 125, 139),
    Label.STATE_ABBREVIATION: (0, 188, 212),
    Label.IMAGE: (255, 87, 34),
    Label.HEADNOTE: (142, 68, 173),
    Label.BACKGROUND: (189, 195, 199),
}

# Max allowed deviation from median key-icon size (as a fraction).
_KEY_ICON_SIZE_TOLERANCE = 0.40


def _filter_key_icons_by_size(
    detections: list[Detection],
    tolerance: float = _KEY_ICON_SIZE_TOLERANCE,
) -> list[Detection]:
    """Keep only KEY_ICON detections whose size is close to the median.

    Key icons are all the same symbol, so valid detections should have
    very similar bounding-box dimensions.  We compute the median width
    and height from high-confidence detections (>= 0.75) and reject any
    detection whose width or height deviates by more than *tolerance*
    from that median.
    """
    from statistics import median

    key_dets = [d for d in detections if d.label == Label.KEY_ICON]
    if len(key_dets) < 2:
        return detections

    # Use high-confidence detections to establish expected size
    high_conf = [d for d in key_dets if d.confidence >= 0.75]
    reference = high_conf if len(high_conf) >= 2 else key_dets

    med_w = median(d.bbox.width for d in reference)
    med_h = median(d.bbox.height for d in reference)

    def size_ok(d: Detection) -> bool:
        if d.label != Label.KEY_ICON:
            return True
        if d.confidence >= 0.75:
            return True  # already trusted
        return (
            abs(d.bbox.width - med_w) / med_w <= tolerance
            and abs(d.bbox.height - med_h) / med_h <= tolerance
        )

    kept = [d for d in detections if size_ok(d)]
    removed = len(detections) - len(kept)
    if removed:
        logger.info("Removed %d KEY_ICON detection(s) with outlier size", removed)
    return kept


def detect_columns(
    img_bgr: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """Detect left/right column boundaries from image using projection analysis.

    Ported from blackletter's detect_columns_from_image.

    Returns (left_x1, left_x2, right_x1, right_x2, center_X) in pixels.
    Raises ValueError on failure.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Ignore header/footer regions
    y1 = int(h * 0.06)
    y2 = int(h * 0.94)
    roi = gray[y1:y2, :]

    # Binarize; ink=1
    _, bw = cv2.threshold(
        roi,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    ink = (bw == 0).astype(np.uint8)

    # Light horizontal dilation to stabilize projection
    k = max(1, int(w * 0.002))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * k + 1, 1))
    ink = cv2.dilate(ink, kernel, iterations=1)

    proj = ink.sum(axis=0).astype(np.float32)

    # Smooth projection
    win = max(5, int(w * 0.02))
    if win % 2 == 0:
        win += 1
    proj_smooth = cv2.GaussianBlur(
        proj.reshape(1, -1),
        (win, 1),
        0,
    ).ravel()

    # Find lowest-ink valley near center
    center = w // 2
    band = int(w * 0.25)
    lo = max(0, center - band)
    hi = min(w, center + band)

    valley_center = lo + int(np.argmin(proj_smooth[lo:hi]))

    # Expand valley using threshold
    med = float(np.median(proj_smooth))
    thr = max(1.0, 0.15 * med)

    vx1 = valley_center
    while vx1 > 0 and proj_smooth[vx1] <= thr:
        vx1 -= 1
    vx2 = valley_center
    while vx2 < w - 1 and proj_smooth[vx2] <= thr:
        vx2 += 1

    # Ensure minimum gutter width
    min_vw = max(3, int(w * 0.01))
    if (vx2 - vx1) < min_vw:
        half = min_vw // 2
        vx1 = max(0, valley_center - half)
        vx2 = min(w, valley_center + half)

    center_x = (vx1 + vx2) // 2

    # Determine where text exists on each side
    text_thr = max(1.0, 0.20 * float(np.mean(proj_smooth)))

    def first_last_above(arr: np.ndarray, offset: int = 0):
        idx = np.where(arr > text_thr)[0]
        if idx.size == 0:
            return None
        return int(idx[0] + offset), int(idx[-1] + offset)

    left_bounds = first_last_above(proj_smooth[:center_x], 0)
    right_bounds = first_last_above(proj_smooth[center_x:], center_x)

    if left_bounds is None or right_bounds is None:
        raise ValueError("Column detection found empty side")

    left_x1, left_x2 = left_bounds
    right_x1, right_x2 = right_bounds

    if not (0 <= left_x1 < left_x2 < center_x < right_x1 < right_x2 <= w):
        raise ValueError(
            f"Column bounds invalid: {(left_x1, left_x2, center_x, right_x1, right_x2)}"
        )

    return (
        float(left_x1),
        float(left_x2),
        float(right_x1),
        float(right_x2),
        float(center_x),
    )


def _ocr_region(fitz_page, rect: fitz.Rect, psm: int = 7) -> str:
    """Run tesseract OCR on a page region, digits only.

    Renders at 4x zoom with binarization for cleaner input.
    """
    mat = fitz.Matrix(4, 4)
    pix = fitz_page.get_pixmap(matrix=mat, clip=rect)

    # Convert to numpy for preprocessing
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    elif pix.n == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Binarize with Otsu to clean up noise
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        cv2.imwrite(tmp.name, bw)
        result = subprocess.run(
            [
                "tesseract",
                tmp.name,
                "stdout",
                "-c",
                "tessedit_char_whitelist=0123456789",
                "--psm",
                str(psm),
            ],
            capture_output=True,
            text=True,
        )
        Path(tmp.name).unlink(missing_ok=True)
    return result.stdout.strip()


def _plausible(n: int, hint: int, max_digits: int = 4) -> bool:
    """Check if a page number is plausible given the hint."""
    if len(str(n)) > max_digits:
        return False
    if n == 0:
        return False
    if hint > 0 and abs(n - hint) > 20:
        return False
    return True


def _extract_page_number(fitz_page, rect: fitz.Rect, hint: int = 0) -> int | None:
    """Extract a page number from a PDF region using text + OCR.

    Tightens the rect to actual text first, then tries the embedded PDF
    text layer and multiple tesseract PSM modes.  Filters out implausible
    results (too many digits, too far from expected position).

    Returns the integer page number, or None if extraction fails.
    """
    # Tighten rect to actual text to avoid grabbing adjacent characters
    tight = _tighten_to_text(fitz_page, rect, padding=2.0)
    if tight is not None:
        rect = tight

    # Determine max expected digits from the hint
    max_digits = max(4, len(str(hint)) + 1) if hint > 0 else 5

    # Method 1: PDF text layer — fast, try this first
    pdf_text = fitz_page.get_text("text", clip=rect).strip()
    pdf_digits = re.sub(r"\D", "", pdf_text)

    # If PDF text gives a plausible number, use it without OCR
    if pdf_digits:
        n = int(pdf_digits)
        if _plausible(n, hint, max_digits):
            return n

    # Method 2: tesseract OCR — only if text layer failed
    # Try modes in order; stop as soon as we get a plausible result
    ocr_results: list[int] = []
    for psm in (8, 7, 13):
        digits = _ocr_region(fitz_page, rect, psm=psm)
        if digits:
            n = int(digits)
            if _plausible(n, hint, max_digits):
                return n
            ocr_results.append(n)

    # Fallback: take any result without plausibility filter
    candidates = []
    if pdf_digits:
        candidates.append(int(pdf_digits))
    candidates.extend(ocr_results)
    if candidates:
        if hint > 0:
            return min(candidates, key=lambda n: abs(n - hint))
        return min(candidates, key=lambda n: len(str(n)))
    return None


def validate_page_numbers(document: Document) -> list[str]:
    """Check page number sequence for gaps and mismatches.

    Returns a list of warning strings (empty if everything looks good).
    """
    warnings: list[str] = []

    numbered = [(p.index, p.page_number) for p in document.pages if p.page_number is not None]

    if not numbered:
        warnings.append("No page numbers detected in document")
        return warnings

    # Check for pages with no detected number
    for p in document.pages:
        if p.page_number is None:
            warnings.append(f"Page index {p.index}: no page number detected")

    # Check sequential consistency — look for gaps
    for i in range(1, len(numbered)):
        prev_idx, prev_num = numbered[i - 1]
        curr_idx, curr_num = numbered[i]
        expected = prev_num + (curr_idx - prev_idx)
        if curr_num != expected:
            warnings.append(
                f"Page index {curr_idx}: expected page {expected}, "
                f"got {curr_num} (gap or misnumber after index {prev_idx}={prev_num})"
            )

    return warnings


def _fix_rotated_pages(pdf_path: Path, dest_dir: Path, stem: str) -> Path:
    """Rasterize rotated pages so every page has rotation=0.

    Pages without rotation are copied as-is.  Rotated pages are rendered
    to a JPEG pixmap (which applies the rotation) and inserted as an image.
    Returns the original path unchanged if no pages are rotated.
    """
    src = fitz.open(pdf_path)
    if not any(src[i].rotation != 0 for i in range(len(src))):
        src.close()
        return pdf_path

    n_fixed = 0
    dst = fitz.open()
    for page in src:
        if page.rotation == 0:
            dst.insert_pdf(src, from_page=page.number, to_page=page.number)
        else:
            pix = page.get_pixmap(dpi=300)
            jpg = pix.tobytes("jpeg", jpg_quality=85)
            new_page = dst.new_page(width=page.rect.width, height=page.rect.height)
            new_page.insert_image(new_page.rect, stream=jpg)
            n_fixed += 1

    out_path = dest_dir / f"{stem}_fixed.pdf"
    dst.save(str(out_path), garbage=4, deflate=True)
    dst.close()
    src.close()
    print(f"  Fixed {n_fixed} rotated page(s)")
    return out_path


def scan(
    pdf_path: Path,
    model: YOLO,
    confidence: float = CONFIDENCE_THRESHOLD,
    first_page: int = 1,
    output_dir: Path | None = None,
    shrink: bool = False,
    optimize: int = 1,
    output_name: str | None = None,
) -> Document:
    """Scan a PDF and return a Document with all detections.

    If the PDF has no text layer, automatically runs OCR first to add
    one (required for accurate page number extraction and redaction
    tightening).

    If shrink=True, downsample images to ~148 KB/page even if the PDF
    already has a text layer.
    """
    from blackletter.ocr import needs_ocr, ocr_pdf, _downsample_pdf

    dest_dir = output_dir if output_dir else Path(tempfile.gettempdir())
    dest_dir.mkdir(parents=True, exist_ok=True)

    out_stem = output_name if output_name else pdf_path.stem

    # Fix rotated pages first so all downstream coordinates are consistent.
    actual_path = _fix_rotated_pages(pdf_path, dest_dir, out_stem)

    if needs_ocr(actual_path):
        print("  PDF has no text layer — running OCR...")
        ocr_out = dest_dir / f"{out_stem}.pdf"
        ocr_pdf(actual_path, ocr_out, optimize=optimize)
        orig_mb = actual_path.stat().st_size / (1024 * 1024)
        actual_path = ocr_out
        final_mb = ocr_out.stat().st_size / (1024 * 1024)
        print(f"  OCR complete: {orig_mb:.1f} MB -> {final_mb:.1f} MB ({ocr_out.name})")
    elif shrink:
        print("  Downsampling images...")
        shrunk = dest_dir / f"{out_stem}.pdf"
        _downsample_pdf(actual_path, shrunk)
        orig_mb = actual_path.stat().st_size / (1024 * 1024)
        actual_path = shrunk
        final_mb = shrunk.stat().st_size / (1024 * 1024)
        print(f"  Downsampled: {orig_mb:.1f} MB -> {final_mb:.1f} MB ({shrunk.name})")
    else:
        mb = actual_path.stat().st_size / (1024 * 1024)
        print(f"  Source PDF: {mb:.1f} MB (no shrink/OCR needed)")

    doc = Document(pdf_path=actual_path, first_page=first_page, ocr_applied=actual_path != pdf_path)
    pdf = fitz.open(actual_path)
    total_pages = len(pdf)

    BATCH_SIZE = 16
    mat = fitz.Matrix(DPI / 72, DPI / 72)

    def _render_batch(start: int, end: int):
        """Render pages and run column detection for a batch."""
        imgs: list[Image.Image] = []
        meta: list[tuple] = []
        for page_idx in range(start, end):
            fitz_page = pdf[page_idx]
            pix = fitz_page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            imgs.append(img)

            img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            try:
                lx1, lx2, rx1, rx2, mid = detect_columns(img_bgr)
            except (ValueError, Exception):
                lx1, lx2, rx1, rx2, mid = 0, 0, 0, 0, 0
            meta.append((page_idx, fitz_page, pix.width, pix.height, (lx1, lx2, rx1, rx2, mid)))
        return imgs, meta

    def _process_results(batch_results, batch_meta):
        """Convert YOLO results into Page objects."""
        for j, (page_idx, fitz_page, pw, ph, cols) in enumerate(batch_meta):
            lx1, lx2, rx1, rx2, mid = cols
            page = Page(
                index=page_idx,
                pdf_width=fitz_page.rect.width,
                pdf_height=fitz_page.rect.height,
                img_width=pw,
                img_height=ph,
                col_left_x1=lx1,
                col_left_x2=lx2,
                col_right_x1=rx1,
                col_right_x2=rx2,
                midpoint=mid,
            )

            for box in batch_results[j].boxes:
                class_id = int(box.cls[0].item())
                page.detections.append(
                    Detection(
                        bbox=BBox.from_xyxy(box.xyxy[0].tolist()),
                        label=Label(class_id),
                        confidence=float(box.conf[0].item()),
                        page_index=page_idx,
                    )
                )

            pn_dets = [d for d in page.detections if d.label == Label.PAGE_NUMBER]
            if pn_dets:
                best_pn = max(pn_dets, key=lambda d: d.confidence)
                b = best_pn.bbox.to_pdf(page.scale_x, page.scale_y)
                rect = fitz.Rect(b.x1, b.y1, b.x2, b.y2)
                page.page_number = _extract_page_number(fitz_page, rect, hint=page_idx + first_page)

            doc.pages.append(page)

    from concurrent.futures import ThreadPoolExecutor

    print(f"  Detecting on {total_pages} pages...")

    # Pipeline: render next batch while YOLO runs on current batch
    batches = [(s, min(s + BATCH_SIZE, total_pages)) for s in range(0, total_pages, BATCH_SIZE)]

    with ThreadPoolExecutor(max_workers=1) as render_pool:
        # Kick off first render
        next_future = render_pool.submit(_render_batch, *batches[0])

        for i, (batch_start, batch_end) in enumerate(batches):
            # Wait for this batch's render to complete
            batch_imgs, batch_meta = next_future.result()

            # Start rendering next batch while we run YOLO
            if i + 1 < len(batches):
                next_future = render_pool.submit(_render_batch, *batches[i + 1])

            # YOLO inference on current batch
            batch_results = model(batch_imgs, conf=confidence, verbose=False)
            _process_results(batch_results, batch_meta)

            if batch_end % 50 < BATCH_SIZE or batch_end == total_pages:
                print(f"    Page {batch_end}/{total_pages}", flush=True)

    # Sequential correction pass: fix misreads using neighbor context
    _correct_page_numbers(doc.pages)

    return doc


def _correct_page_numbers(pages: list[Page]) -> None:
    """Fix page number misreads using sequential context.

    Makes multiple passes.  On each pass, for every page, looks for the
    nearest neighbors (up to 10 pages away) that agree with each other
    sequentially.  If they do, interpolates the expected value.

    Repeats until no more corrections are made.
    """
    n = len(pages)
    if n < 3:
        return

    max_look = 10  # how far to search for a good neighbor

    for _pass in range(5):  # up to 5 correction passes
        corrections = 0
        for i in range(n):
            # Find nearest valid neighbor before
            prev_num = None
            prev_dist = 0
            for j in range(i - 1, max(i - max_look - 1, -1), -1):
                if pages[j].page_number is not None:
                    prev_num = pages[j].page_number
                    prev_dist = i - j
                    break

            # Find nearest valid neighbor after
            next_num = None
            next_dist = 0
            for j in range(i + 1, min(i + max_look + 1, n)):
                if pages[j].page_number is not None:
                    next_num = pages[j].page_number
                    next_dist = j - i
                    break

            if prev_num is None or next_num is None:
                continue

            # Check if neighbors are consistent with each other
            total_dist = prev_dist + next_dist
            if next_num - prev_num != total_dist:
                continue

            expected = prev_num + prev_dist
            if pages[i].page_number != expected:
                pages[i].page_number = expected
                corrections += 1

        if corrections == 0:
            break


def draw_detections(
    pdf_path: Path,
    document: Document,
    output_path: Path,
    labels: set[Label] | None = None,
) -> Path:
    """Draw bounding boxes on the PDF and write an annotated copy.

    Uses PyMuPDF to draw directly on the PDF pages — no rasterization needed.

    Args:
        pdf_path: Path to the source PDF.
        document: Scanned Document with detections.
        output_path: Path for the output PDF.
        labels: If provided, only draw these labels. None = draw all.

    Returns:
        Path to the written PDF.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = fitz.open(pdf_path)

    for page in document.pages:
        fitz_page = pdf[page.index]
        shape = fitz_page.new_shape()

        dets = page.detections
        if labels is not None:
            dets = [d for d in dets if d.label in labels]
        dets = _filter_dets(dets)

        _draw_boxes(shape, dets, page, fitz_page)
        shape.commit()

    pdf.save(output_path)
    pdf.close()
    return output_path


def _pair_opinions(
    document: Document,
) -> list[tuple[Detection, Detection]]:
    """Pair each CASE_CAPTION with the next KEY_ICON in reading order.

    Key icons are the reliable anchor.  After each key icon, the first
    caption that follows starts the next opinion.  A BACKGROUND detection
    can also signal the end of a multi-page caption (the opinion text
    begins after the background region).

    Only the first caption after a key icon is used; subsequent captions
    before the next key icon are ignored to avoid false-positive splits.
    """
    captions = [
        d
        for d in document.by_label(Label.CASE_CAPTION)
        if d.confidence >= LABEL_CONFIDENCE.get(Label.CASE_CAPTION, CONFIDENCE_THRESHOLD)
    ]
    all_keys = [
        d
        for d in document.by_label(Label.KEY_ICON)
        if d.confidence >= LABEL_CONFIDENCE.get(Label.KEY_ICON, CONFIDENCE_THRESHOLD)
    ]
    keys = [d for d in _filter_key_icons_by_size(all_keys) if d.label == Label.KEY_ICON]

    if not document.pages:
        return []

    mid = document.pages[0].midpoint
    markers = [(d, "C") for d in captions] + [(d, "K") for d in keys]
    markers.sort(key=lambda m: m[0].sort_key(mid))

    opinions: list[tuple[Detection, Detection]] = []
    pending_caption = None
    for det, kind in markers:
        if kind == "C":
            # Only accept the first caption after a key (or at the start)
            if pending_caption is None:
                pending_caption = det
        elif kind == "K" and pending_caption is not None:
            opinions.append((pending_caption, det))
            pending_caption = None

    return opinions


def _tighten_to_text(
    fitz_page,
    rect: fitz.Rect,
    padding: float = 2.0,
    skip: bool = False,
) -> fitz.Rect | None:
    """Shrink a rect to tightly fit the PDF text within it.

    Uses word-level extraction for precise bounds (important for small
    elements like brackets, page numbers, state abbreviations).

    Returns None if no text is found inside the rect, or if skip=True
    (used when text layer is from our own OCR and positions are imprecise).
    """
    if skip:
        return None
    words = fitz_page.get_text("words", clip=rect)
    # words = [(x0, y0, x1, y1, word, block_no, line_no, word_no), ...]
    words = [w for w in words if w[4].strip()]
    if not words:
        return None
    return fitz.Rect(
        min(w[0] for w in words) - padding,
        min(w[1] for w in words) - padding,
        max(w[2] for w in words) + padding,
        max(w[3] for w in words) + padding,
    )


def _filter_dets(dets: list[Detection]) -> list[Detection]:
    """Apply per-label confidence thresholds and resolve overlaps.

    - Drops detections below their label's confidence threshold.
    - When STATE_ABBREVIATION overlaps a PAGE_NUMBER, trims the SA box
      so page numbers are never covered by a state-abbreviation redaction.
    """
    filtered = [
        d for d in dets if d.confidence >= LABEL_CONFIDENCE.get(d.label, CONFIDENCE_THRESHOLD)
    ]

    # Resolve PAGE_NUMBER / STATE_ABBREVIATION overlaps: trim SA
    pn_boxes = [d.bbox for d in filtered if d.label == Label.PAGE_NUMBER]
    result = []
    for d in filtered:
        if d.label == Label.STATE_ABBREVIATION and pn_boxes:
            box = d.bbox
            for pn in pn_boxes:
                if box.iou(pn) > 0:
                    # Trim SA to not overlap PN
                    if box.center_x < pn.center_x:
                        box = BBox(box.x1, box.y1, min(box.x2, pn.x1), box.y2)
                    else:
                        box = BBox(max(box.x1, pn.x2), box.y1, box.x2, box.y2)
            if box.width > 5:
                result.append(
                    Detection(
                        bbox=box,
                        label=d.label,
                        confidence=d.confidence,
                        page_index=d.page_index,
                    )
                )
        else:
            result.append(d)
    return result


def _draw_boxes(
    shape,
    dets: list[Detection],
    page: Page,
    fitz_page=None,
) -> None:
    """Draw labeled bounding boxes onto a fitz shape.

    If fitz_page is provided, boxes are tightened to the actual text.
    PAGE_NUMBER / STATE_ABBREVIATION overlaps are resolved after tightening
    so page numbers are never covered.
    """
    # First pass: compute all rects (tightened where possible)
    items: list[tuple[Detection, fitz.Rect]] = []
    for det in dets:
        b = det.bbox.to_pdf(page.scale_x, page.scale_y)
        rect = fitz.Rect(b.x1, b.y1, b.x2, b.y2)
        if fitz_page is not None and det.label != Label.KEY_ICON:
            tight = _tighten_to_text(fitz_page, rect)
            if tight is not None:
                rect = tight
        items.append((det, rect))

    # Resolve overlaps: trim STATE_ABBREVIATION away from PAGE_NUMBER
    pn_rects = [r for d, r in items if d.label == Label.PAGE_NUMBER]
    final: list[tuple[Detection, fitz.Rect]] = []
    for det, rect in items:
        if det.label == Label.STATE_ABBREVIATION and pn_rects:
            for pn_r in pn_rects:
                if rect.intersects(pn_r):
                    if rect.x0 < pn_r.x0:
                        rect = fitz.Rect(rect.x0, rect.y0, min(rect.x1, pn_r.x0), rect.y1)
                    else:
                        rect = fitz.Rect(max(rect.x0, pn_r.x1), rect.y0, rect.x1, rect.y1)
            if rect.width < 3:
                continue
        final.append((det, rect))

    # Draw
    for det, rect in final:
        rgb = LABEL_COLORS.get(det.label, (255, 255, 255))
        color = (rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)
        shape.draw_rect(rect)
        shape.finish(color=color, width=1.5)

        text = f"{det.label.name} {det.confidence:.0%}"
        text_point = fitz.Point(rect.x0, rect.y0 - 2)
        shape.insert_text(text_point, text, fontsize=7, color=color)


def _mask_rect(shape, rect: fitz.Rect) -> None:
    """Draw a filled white rectangle to mask content."""
    shape.draw_rect(rect)
    shape.finish(color=(1, 1, 1), fill=(1, 1, 1), width=0)


_MARGIN_LABELS = frozenset(
    {
        Label.PAGE_HEADER,
        Label.PAGE_NUMBER,
        Label.STATE_ABBREVIATION,
    }
)


def _margin_bounds(page: Page) -> tuple[float, float]:
    """Return (header_bottom, footer_top) in PDF points.

    header_bottom: below PAGE_HEADER, PAGE_NUMBER, STATE_ABBREVIATION
    footer_top: above footnotes and any bottom-of-page margin elements

    Returns (0, page_height) if none are found.
    """
    sx, sy = page.scale_x, page.scale_y
    page_h = page.pdf_height
    mid_y = page_h / 2

    top_bottom = 0.0
    bot_top = page_h

    for det in page.detections:
        b = det.bbox.to_pdf(sx, sy)

        if det.label in _MARGIN_LABELS:
            if b.center_y < mid_y:
                top_bottom = max(top_bottom, b.y2)
            else:
                bot_top = min(bot_top, b.y1)

        if det.label == Label.FOOTNOTES:
            bot_top = min(bot_top, b.y1)

    return top_bottom, bot_top


def _text_bottom(fitz_page, clip: fitz.Rect) -> float:
    """Return the y-coordinate of the bottom of the last text in clip."""
    blocks = fitz_page.get_text("blocks", clip=clip)
    if not blocks:
        return clip.y0
    # blocks are (x0, y0, x1, y1, text, block_no, block_type)
    # type 0 = text
    text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]
    if not text_blocks:
        return clip.y0
    return max(b[3] for b in text_blocks)


def _text_x_bounds(fitz_page, clip: fitz.Rect, padding: float = 2.0) -> tuple[float, float]:
    """Return (left, right) x-coordinates of the text extent within clip."""
    blocks = fitz_page.get_text("blocks", clip=clip)
    if not blocks:
        return clip.x0, clip.x1
    text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]
    if not text_blocks:
        return clip.x0, clip.x1
    return (
        min(b[0] for b in text_blocks) - padding,
        max(b[2] for b in text_blocks) + padding,
    )


def _outside_opinion_rects(
    page: Page,
    page_width: float,
    caption: Detection,
    key: Detection,
    is_first: bool,
    is_last: bool,
) -> list[fitz.Rect]:
    """Return rects for content outside the opinion span on this page."""
    sx = page.scale_x
    w = page_width
    mid_pdf = page.midpoint * sx
    header_bottom, footer_top = _margin_bounds(page)
    rects: list[fitz.Rect] = []

    if is_first:
        cap_box = caption.bbox.to_pdf(page.scale_x, page.scale_y)
        cap_col = "LEFT" if cap_box.center_x < mid_pdf else "RIGHT"
        if cap_col == "LEFT":
            rects.append(fitz.Rect(0, header_bottom, mid_pdf, cap_box.y1))
        else:
            rects.append(fitz.Rect(0, header_bottom, mid_pdf, footer_top))
            rects.append(fitz.Rect(mid_pdf, header_bottom, w, cap_box.y1))

    if is_last:
        key_box = key.bbox.to_pdf(page.scale_x, page.scale_y)
        key_col = "LEFT" if key_box.center_x < mid_pdf else "RIGHT"
        if key_col == "RIGHT":
            rects.append(fitz.Rect(mid_pdf, key_box.y2, w, footer_top))
        else:
            rects.append(fitz.Rect(0, key_box.y2, mid_pdf, footer_top))
            rects.append(fitz.Rect(mid_pdf, header_bottom, w, footer_top))

    return [r for r in rects if r.y0 < r.y1 and r.x0 < r.x1]


def _mask_outside_opinion(
    shape,
    fitz_page,
    page: Page,
    caption: Detection,
    key: Detection,
    is_first: bool,
    is_last: bool,
) -> None:
    """Draw white masks for content outside the opinion span."""
    for rect in _outside_opinion_rects(
        page,
        fitz_page.rect.width,
        caption,
        key,
        is_first,
        is_last,
    ):
        _mask_rect(shape, rect)


def _find_redaction_end(
    opinion_dets: list[Detection],
    caption: Detection,
    key: Detection,
    mid: float,
) -> Detection | None:
    """Find the end of the headnote zone.

    Looks for the first DIVIDER or CASE_METADATA after the caption.
    Returns None if neither is found (caller should try fallback).
    """
    cap_sk = caption.sort_key(mid)
    after_caption = [d for d in opinion_dets if d.sort_key(mid) > cap_sk]

    for d in after_caption:
        if d.label == Label.DIVIDER:
            return d
    for d in after_caption:
        if d.label == Label.CASE_METADATA:
            return d

    return None


def _headnote_fallback_rects(
    opinion_dets: list[Detection],
    caption: Detection,
    pages_by_index: dict[int, Page],
    mid: float,
    min_confidence: float = 0.75,
) -> list[tuple[int, fitz.Rect]]:
    """Fallback redaction when no DIVIDER/CASE_METADATA exists.

    Redacts from the caption down to the end of the last column that
    has HEADNOTE detections >= min_confidence.
    """
    cap_sk = caption.sort_key(mid)
    after_caption = [d for d in opinion_dets if d.sort_key(mid) > cap_sk]

    headnotes = [
        d for d in after_caption if d.label == Label.HEADNOTE and d.confidence >= min_confidence
    ]
    if not headnotes:
        return []

    cap_page = pages_by_index[caption.page_index]
    cap_box = caption.bbox.to_pdf(cap_page.scale_x, cap_page.scale_y)

    # Group headnotes by (page_index, column)
    from collections import defaultdict

    col_headnotes: dict[tuple[int, int], list[Detection]] = defaultdict(list)
    for hn in headnotes:
        page = pages_by_index[hn.page_index]
        col = 0 if hn.bbox.center_x < page.midpoint else 1
        col_headnotes[(hn.page_index, col)].append(hn)

    # Find the last column in reading order that has headnotes
    last_page = max(k[0] for k in col_headnotes)
    last_cols = [k[1] for k in col_headnotes if k[0] == last_page]
    last_col = max(last_cols)

    rects: list[tuple[int, fitz.Rect]] = []
    cap_col = 0 if cap_box.center_x < (cap_page.midpoint * cap_page.scale_x) else 1

    for page_idx in range(caption.page_index, last_page + 1):
        page = pages_by_index[page_idx]
        sx = page.scale_x
        header_bottom, footer_top = _margin_bounds(page)

        col_bounds = [
            (page.col_left_x1 * sx, page.col_left_x2 * sx),
            (page.col_right_x1 * sx, page.col_right_x2 * sx),
        ]

        is_start = page_idx == caption.page_index
        is_end = page_idx == last_page

        for col in range(2):
            if is_start and col < cap_col:
                continue
            if is_end and col > last_col:
                continue

            # Only redact columns that actually have headnotes
            col_hns = col_headnotes.get((page_idx, col))
            if not col_hns and is_end:
                continue

            cx1, cx2 = col_bounds[col]
            y_top = header_bottom
            y_bot = footer_top

            if is_start and col == cap_col:
                y_top = cap_box.y2

            # All columns from caption to the last column with headnotes
            # get full-column redaction (y_bot stays at footer_top)

            y_top = max(y_top, header_bottom)
            y_bot = min(y_bot, footer_top)

            if y_top < y_bot:
                rects.append((page_idx, fitz.Rect(cx1, y_top, cx2, y_bot)))

    return rects


def _redaction_rects(
    caption: Detection,
    end_marker: Detection,
    pages_by_index: dict[int, Page],
) -> list[tuple[int, fitz.Rect]]:
    """Generate column-aware redaction rectangles from caption to end marker.

    Fills every column in reading order between the caption and end marker.

    Returns a list of (source_page_index, rect_in_pdf_points).
    """
    cap_page = pages_by_index[caption.page_index]
    end_page = pages_by_index[end_marker.page_index]
    cap_box = caption.bbox.to_pdf(cap_page.scale_x, cap_page.scale_y)
    end_box = end_marker.bbox.to_pdf(end_page.scale_x, end_page.scale_y)

    rects: list[tuple[int, fitz.Rect]] = []

    for page_idx in range(caption.page_index, end_marker.page_index + 1):
        page = pages_by_index[page_idx]
        sx = page.scale_x
        mid_pdf = page.midpoint * sx
        header_bottom, footer_top = _margin_bounds(page)

        col_bounds = [
            (page.col_left_x1 * sx, page.col_left_x2 * sx),
            (page.col_right_x1 * sx, page.col_right_x2 * sx),
        ]

        cap_col = 0 if cap_box.center_x < mid_pdf else 1
        end_col = 0 if end_box.center_x < mid_pdf else 1
        is_start = page_idx == caption.page_index
        is_end = page_idx == end_marker.page_index

        for col in range(2):
            cx1, cx2 = col_bounds[col]
            y_top = header_bottom
            y_bot = footer_top

            if is_start:
                if col < cap_col:
                    continue
                if col == cap_col:
                    y_top = cap_box.y2

            if is_end:
                if col > end_col:
                    continue
                if col == end_col:
                    y_bot = end_box.y1

            y_top = max(y_top, header_bottom)
            y_bot = min(y_bot, footer_top)

            if y_top < y_bot:
                rects.append((page_idx, fitz.Rect(cx1, y_top, cx2, y_bot)))

    return rects


_HEADNOTE_LABELS = frozenset({Label.HEADNOTE, Label.HEADNOTE_BRACKET})

REDACTION_COLOR = (1, 0, 0)  # red for debug
REDACTION_OPACITY = 0.25


_REDACT_WHITE = frozenset(
    {
        Label.STATE_ABBREVIATION,
        Label.PAGE_HEADER,
    }
)
_REDACT_BLACK = frozenset(
    {
        Label.DIVIDER,
        Label.HEADNOTE_BRACKET,
    }
)


def _extract_opinion_footnotes(
    src_pdf,
    footnote_dets: list[Detection],
    pages_by_index: dict[int, Page],
    output_path: Path,
    skip_tighten: bool = False,
) -> Path | None:
    """Extract footnote regions for one opinion into a PDF.

    If a footnote spans both columns, the left half is placed above
    the right half so footnotes read top-to-bottom in order.

    Returns the output path, or None if no footnotes.
    """
    if not footnote_dets:
        return None

    # Deduplicate: on each page, if a wide (both-column) detection exists,
    # drop any per-column detections that overlap it.
    deduped: list[Detection] = []
    by_page: dict[int, list[Detection]] = {}
    for d in footnote_dets:
        by_page.setdefault(d.page_index, []).append(d)
    for page_idx in sorted(by_page):
        dets = sorted(by_page[page_idx], key=lambda d: -d.confidence)
        keep: list[Detection] = []
        for d in dets:
            if any(d.bbox.iou(k.bbox) > 0.3 for k in keep):
                continue
            keep.append(d)
        deduped.extend(sorted(keep, key=lambda d: d.bbox.y1))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_pdf = fitz.open()

    page_w = 612.0  # letter
    page_h = 792.0
    margin = 36.0
    usable_w = page_w - 2 * margin
    gap = 8.0

    out_page = out_pdf.new_page(width=page_w, height=page_h)
    y_cursor = margin

    for det in deduped:
        page = pages_by_index[det.page_index]
        sx, sy = page.scale_x, page.scale_y
        b = det.bbox.to_pdf(sx, sy)
        full_clip = fitz.Rect(b.x1, b.y1, b.x2, b.y2)

        if full_clip.is_empty:
            continue

        mid_pdf = page.midpoint * sx

        # Does this footnote span both columns?
        spans_both = full_clip.x0 < mid_pdf and full_clip.x1 > mid_pdf

        if spans_both:
            # Split into left column then right column
            clips = [
                fitz.Rect(full_clip.x0, full_clip.y0, mid_pdf, full_clip.y1),
                fitz.Rect(mid_pdf, full_clip.y0, full_clip.x1, full_clip.y1),
            ]
        else:
            clips = [full_clip]

        for clip in clips:
            src_page = src_pdf[det.page_index]
            tight = _tighten_to_text(src_page, clip, padding=4.0, skip=skip_tighten)
            if tight is not None:
                clip = tight
            if clip.is_empty or clip.width < 2 or clip.height < 2:
                continue

            # Copy page, redact outside clip, place on output
            tmp = fitz.open()
            tmp.insert_pdf(
                src_pdf,
                from_page=det.page_index,
                to_page=det.page_index,
            )
            tp = tmp[0]
            pw, ph = tp.rect.width, tp.rect.height
            if clip.y0 > 0:
                tp.add_redact_annot(fitz.Rect(0, 0, pw, clip.y0), fill=(1, 1, 1))
            if clip.y1 < ph:
                tp.add_redact_annot(fitz.Rect(0, clip.y1, pw, ph), fill=(1, 1, 1))
            if clip.x0 > 0:
                tp.add_redact_annot(fitz.Rect(0, clip.y0, clip.x0, clip.y1), fill=(1, 1, 1))
            if clip.x1 < pw:
                tp.add_redact_annot(fitz.Rect(clip.x1, clip.y0, pw, clip.y1), fill=(1, 1, 1))
            tp.apply_redactions()

            scale = min(1.0, usable_w / clip.width)
            dest_w = clip.width * scale
            dest_h = clip.height * scale

            if y_cursor + dest_h > page_h - margin and y_cursor > margin + 1:
                out_page = out_pdf.new_page(width=page_w, height=page_h)
                y_cursor = margin

            dest_rect = fitz.Rect(
                margin,
                y_cursor,
                margin + dest_w,
                y_cursor + dest_h,
            )
            out_page.show_pdf_page(dest_rect, tmp, 0, clip=clip)
            tmp.close()
            y_cursor += dest_h + gap

    if len(out_pdf) == 0:
        out_pdf.close()
        return None

    out_pdf.save(output_path)
    out_pdf.close()
    return output_path


def recompress_images(pdf: fitz.Document, quality: int = 80) -> None:
    """Re-encode PNG images as grayscale JPEG after redactions.

    apply_redactions() converts JPEG images to PNG, inflating file size.
    This converts them back to grayscale JPEG using fitz.Pixmap for speed.
    """
    seen: set[int] = set()
    for i in range(len(pdf)):
        for img_info in pdf[i].get_images(full=True):
            xref = img_info[0]
            if xref in seen:
                continue
            seen.add(xref)
            if "/DCTDecode" in pdf.xref_object(xref):
                continue
            pix = fitz.Pixmap(pdf, xref)
            if pix.n > 1:  # convert RGB/RGBA to grayscale
                pix = fitz.Pixmap(fitz.csGRAY, pix)
            pil_img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=quality)
            pdf.update_stream(xref, buf.getvalue(), compress=False)
            pdf.xref_set_key(xref, "Filter", "/DCTDecode")
            pdf.xref_set_key(xref, "DecodeParms", "null")
            pdf.xref_set_key(xref, "ColorSpace", "/DeviceGray")


def split_opinions(
    pdf_path: Path,
    document: Document,
    output_dir: Path,
    first_page: int = 1,
    draw_labels: set[Label] | None = None,
    mask: bool = False,
    debug_redactions: bool = False,
    redact: bool = False,
    redact_mode: str | None = None,
    extract_footnotes: bool = False,
) -> list[Path]:
    """Split a PDF into individual opinion files based on caption→key pairs.

    Modes:
        debug (draw_labels + debug_redactions): bounding boxes and red overlays
        redact: actual PDF redactions — white for outside-opinion content,
                black for headnote zones, white for headers/state abbrevs/dividers

    redact_mode overrides the redact boolean when set:
        None: use the redact boolean (backwards compat)
        "unredacted": no redactions
        "redacted": headnote blackout + per-detection redactions, NO outside whiteout
        "masked": outside-opinion whiteout only, no headnote/per-detection redactions

    Files are named {page_number:04d}-{seq:02d}.pdf.
    """
    # Resolve mode
    if redact_mode == "unredacted":
        redact = False
    elif redact_mode in ("redacted", "masked"):
        redact = True
    output_dir.mkdir(parents=True, exist_ok=True)
    opinions = _pair_opinions(document)

    if not opinions:
        return []

    mid = document.pages[0].midpoint
    pages_by_index = {p.index: p for p in document.pages}

    pdf = fitz.open(pdf_path)
    written: list[Path] = []

    # Build filename prefix from reporter/volume if available
    prefix = ""
    if document.reporter and document.volume:
        prefix = f"{document.reporter}.{document.volume}."

    # First pass: compute base names and detect gaps within each opinion
    base_names: list[str] = []
    for caption, key in opinions:
        cap_page = pages_by_index[caption.page_index]
        key_page = pages_by_index[key.page_index]
        first_num = cap_page.page_number
        last_num = key_page.page_number

        # Check for gaps and suspicious page numbers
        has_gap = False
        needs_verify = False
        if first_num is not None and last_num is not None:
            expected_pages = last_num - first_num + 1
            actual_pages = key.page_index - caption.page_index + 1
            if actual_pages < expected_pages:
                has_gap = True
                print(
                    f"GAP in opinion {first_num}-{last_num}: expected "
                    f"{expected_pages} pages, have {actual_pages} "
                    f"({expected_pages - actual_pages} pages missing from source)"
                )
            # Check for non-sequential page numbers (possible OCR misreads)
            nums = []
            for idx in range(caption.page_index, key.page_index + 1):
                p = pages_by_index[idx]
                if p.page_number is not None:
                    nums.append((idx, p.page_number))
            for j in range(1, len(nums)):
                prev_idx, prev_num = nums[j - 1]
                curr_idx, curr_num = nums[j]
                expected = prev_num + (curr_idx - prev_idx)
                if curr_num != expected and not has_gap:
                    needs_verify = True
                    print(
                        f"VERIFY opinion {first_num}-{last_num}: page index "
                        f"{curr_idx} reads as {curr_num}, expected {expected} "
                        f"(possible OCR misread)"
                    )

        if first_num is not None and last_num is not None:
            base = f"{prefix}{first_num:04d}-{last_num:04d}"
            if has_gap:
                base += "_GAP"
            elif needs_verify:
                base += "_VERIFY"
            base_names.append(base)
        else:
            fb_first = caption.page_index + first_page
            fb_last = key.page_index + first_page
            base_names.append(f"{prefix}UNVERIFIED_{fb_first:04d}-{fb_last:04d}")

    # Check for gaps between consecutive opinions
    for i in range(1, len(opinions)):
        _, prev_key = opinions[i - 1]
        curr_caption, _ = opinions[i]
        prev_key_page = pages_by_index[prev_key.page_index]
        curr_cap_page = pages_by_index[curr_caption.page_index]
        prev_last = prev_key_page.page_number
        curr_first = curr_cap_page.page_number
        if prev_last is not None and curr_first is not None:
            # Pages between opinions: prev_key page to curr_caption page
            pdf_pages_between = curr_caption.page_index - prev_key.page_index
            page_nums_between = curr_first - prev_last
            if page_nums_between > pdf_pages_between:
                missing = page_nums_between - pdf_pages_between
                print(
                    f"GAP between opinions: {prev_last} -> {curr_first} "
                    f"({missing} pages missing from source)"
                )
                # Flag both adjacent opinions
                if "_GAP" not in base_names[i - 1]:
                    base_names[i - 1] += "_GAP"
                if "_GAP" not in base_names[i]:
                    base_names[i] += "_GAP"

    # Count occurrences to know which names need sequence numbers
    from collections import Counter

    name_counts = Counter(base_names)
    name_seq: dict[str, int] = {}

    for i, (caption, key) in enumerate(opinions):
        base = base_names[i]
        if name_counts[base] > 1:
            seq = name_seq.get(base, 0) + 1
            name_seq[base] = seq
            name = f"{base}-{seq:02d}.pdf"
        else:
            name = f"{base}.pdf"

        if "UNVERIFIED" in name:
            print(
                f"UNVERIFIED: opinion at index {caption.page_index}-"
                f"{key.page_index}, using fallback: {name}"
            )
        out_path = output_dir / name

        out_pdf = fitz.open()
        out_pdf.insert_pdf(pdf, from_page=caption.page_index, to_page=key.page_index)

        cap_key = caption.sort_key(mid)
        key_key = key.sort_key(mid)

        # Gather all detections in this opinion's span
        opinion_dets = []
        for src_idx in range(caption.page_index, key.page_index + 1):
            for d in pages_by_index[src_idx].detections:
                sk = d.sort_key(mid)
                if cap_key <= sk <= key_key:
                    opinion_dets.append(d)
        opinion_dets.sort(key=lambda d: d.sort_key(mid))

        # Compute headnote-zone rects (caption to divider/metadata)
        headnote_rects: list[tuple[int, fitz.Rect]] = []
        if debug_redactions or redact:
            end_marker = _find_redaction_end(
                opinion_dets,
                caption,
                key,
                mid,
            )
            if end_marker is not None:
                headnote_rects = _redaction_rects(
                    caption,
                    end_marker,
                    pages_by_index,
                )
            else:
                headnote_rects = _headnote_fallback_rects(
                    opinion_dets,
                    caption,
                    pages_by_index,
                    mid,
                )

        for local_idx, src_idx in enumerate(range(caption.page_index, key.page_index + 1)):
            page = pages_by_index[src_idx]
            fitz_page = out_pdf[local_idx]
            is_first = src_idx == caption.page_index
            is_last = src_idx == key.page_index

            if redact:
                # ── Actual redaction mode ──────────────────────────
                sx, sy = page.scale_x, page.scale_y

                # Collect page number rects — these must never be redacted
                pn_rects = []
                for d in page.detections:
                    if d.label == Label.PAGE_NUMBER:
                        b = d.bbox.to_pdf(sx, sy)
                        r = fitz.Rect(b.x1, b.y1, b.x2, b.y2)
                        tight = _tighten_to_text(fitz_page, r, skip=document.ocr_applied)
                        pn_rects.append(tight if tight is not None else r)

                def add_safe(rect: fitz.Rect, fill) -> None:
                    """Add redaction annotation, clipping around page numbers."""
                    if rect.is_empty:
                        return
                    for pn in pn_rects:
                        if rect.intersects(pn):
                            # Split: add parts above/below/left/right of PN
                            # Top slice
                            if rect.y0 < pn.y0:
                                add_safe(
                                    fitz.Rect(
                                        rect.x0,
                                        rect.y0,
                                        rect.x1,
                                        pn.y0,
                                    ),
                                    fill,
                                )
                            # Bottom slice
                            if rect.y1 > pn.y1:
                                add_safe(
                                    fitz.Rect(
                                        rect.x0,
                                        pn.y1,
                                        rect.x1,
                                        rect.y1,
                                    ),
                                    fill,
                                )
                            # Left slice (middle band only)
                            top = max(rect.y0, pn.y0)
                            bot = min(rect.y1, pn.y1)
                            if rect.x0 < pn.x0 and top < bot:
                                add_safe(
                                    fitz.Rect(
                                        rect.x0,
                                        top,
                                        pn.x0,
                                        bot,
                                    ),
                                    fill,
                                )
                            # Right slice (middle band only)
                            if rect.x1 > pn.x1 and top < bot:
                                add_safe(
                                    fitz.Rect(
                                        pn.x1,
                                        top,
                                        rect.x1,
                                        bot,
                                    ),
                                    fill,
                                )
                            return
                    fitz_page.add_redact_annot(rect, fill=fill)

                # White redact: content outside opinion (redacted + masked)
                if is_first or is_last:
                    for rect in _outside_opinion_rects(
                        page,
                        fitz_page.rect.width,
                        caption,
                        key,
                        is_first,
                        is_last,
                    ):
                        add_safe(rect, (1, 1, 1))

                # Black redact: headnote zone (clipped to actual text)
                header_bottom, footer_top = _margin_bounds(page)
                for rect_page_idx, rect in headnote_rects:
                    if rect_page_idx == src_idx:
                        if not document.ocr_applied:
                            tight = _tighten_to_text(fitz_page, rect)
                            if tight is not None:
                                rect = tight
                            text_bot = _text_bottom(fitz_page, rect)
                            text_left, text_right = _text_x_bounds(fitz_page, rect)
                            rect = fitz.Rect(
                                max(rect.x0, text_left),
                                max(rect.y0, header_bottom),
                                min(rect.x1, text_right),
                                min(rect.y1, footer_top, text_bot),
                            )
                        else:
                            # No reliable text layer — just clip to margins
                            rect = fitz.Rect(
                                rect.x0,
                                max(rect.y0, header_bottom),
                                rect.x1,
                                min(rect.y1, footer_top),
                            )
                        if rect.y0 < rect.y1 and rect.x0 < rect.x1:
                            add_safe(rect, (0, 0, 0))

                # Per-detection redactions (skip black outside opinion bounds)
                redact_labels = _REDACT_WHITE | _REDACT_BLACK
                for d in page.detections:
                    if d.label not in redact_labels:
                        continue
                    if d.confidence < LABEL_CONFIDENCE.get(
                        d.label,
                        CONFIDENCE_THRESHOLD,
                    ):
                        continue
                    # Skip detections outside this opinion's content
                    # (but always redact margin elements like headers)
                    if d.label not in _MARGIN_LABELS:
                        sk = d.sort_key(mid)
                        if sk < cap_key or sk > key_key:
                            continue
                    b = d.bbox.to_pdf(sx, sy)
                    rect = fitz.Rect(b.x1, b.y1, b.x2, b.y2)
                    tight = _tighten_to_text(fitz_page, rect, skip=document.ocr_applied)
                    if tight is not None:
                        rect = tight
                    fill = (0, 0, 0) if d.label in _REDACT_BLACK else (1, 1, 1)
                    add_safe(rect, fill)

                # KEY_ICON redactions: white before caption, black at end
                for d in page.detections:
                    if d.label != Label.KEY_ICON:
                        continue
                    if d.confidence < LABEL_CONFIDENCE.get(
                        d.label,
                        CONFIDENCE_THRESHOLD,
                    ):
                        continue
                    b = d.bbox.to_pdf(sx, sy)
                    rect = fitz.Rect(b.x1, b.y1, b.x2, b.y2)
                    sk = d.sort_key(mid)
                    if sk < cap_key or sk > key_key:
                        # Outside opinion bounds — always white
                        add_safe(rect, (1, 1, 1))
                    else:
                        # Within opinion (including own key) — black
                        add_safe(rect, (0, 0, 0))

                fitz_page.apply_redactions()

                # Draw bounding boxes for requested labels on top of redacted page
                if draw_labels:
                    shape = fitz_page.new_shape()
                    dets = _filter_dets([d for d in page.detections if d.label in draw_labels])
                    _draw_boxes(shape, dets, page, fitz_page)
                    shape.commit()

            else:
                # ── Debug mode ────────────────────────────────────
                shape = fitz_page.new_shape()

                if mask:
                    _mask_outside_opinion(
                        shape,
                        fitz_page,
                        page,
                        caption,
                        key,
                        is_first,
                        is_last,
                    )

                # Debug headnote zone overlays
                header_bottom, footer_top = _margin_bounds(page)
                for rect_page_idx, rect in headnote_rects:
                    if rect_page_idx == src_idx:
                        tight = _tighten_to_text(fitz_page, rect, skip=document.ocr_applied)
                        if tight is not None:
                            rect = tight
                        rect = fitz.Rect(
                            rect.x0,
                            max(rect.y0, header_bottom),
                            rect.x1,
                            min(rect.y1, footer_top),
                        )
                        if rect.y0 >= rect.y1:
                            continue
                        shape.draw_rect(rect)
                        shape.finish(
                            color=REDACTION_COLOR,
                            fill=REDACTION_COLOR,
                            fill_opacity=REDACTION_OPACITY,
                            width=1,
                        )

                if draw_labels is not None:
                    dets = _filter_dets(
                        [
                            d
                            for d in page.detections
                            if d.label in draw_labels
                            and (d.label in _MARGIN_LABELS or cap_key <= d.sort_key(mid) <= key_key)
                        ]
                    )
                    _draw_boxes(shape, dets, page, fitz_page)

                shape.commit()

        # For masked mode, remove pages that are fully headnotes
        if redact_mode == "masked" and headnote_rects:
            pages_to_delete: list[int] = []
            for local_idx, src_idx in enumerate(range(caption.page_index, key.page_index + 1)):
                page = pages_by_index[src_idx]
                header_bottom, footer_top = _margin_bounds(page)
                sx = page.scale_x
                mid_pdf = page.midpoint * sx
                content_height = footer_top - header_bottom
                if content_height <= 0:
                    continue

                # Collect headnote rects for this page, split by column
                left_coverage = 0.0
                right_coverage = 0.0
                for rect_page_idx, rect in headnote_rects:
                    if rect_page_idx != src_idx:
                        continue
                    rect_height = min(rect.y1, footer_top) - max(rect.y0, header_bottom)
                    if rect_height <= 0:
                        continue
                    if rect.x0 + rect.width / 2 < mid_pdf:
                        left_coverage += rect_height / content_height
                    else:
                        right_coverage += rect_height / content_height

                # If both columns are >=95% covered by headnotes, skip page
                if left_coverage >= 0.95 and right_coverage >= 0.95:
                    pages_to_delete.append(local_idx)

            if pages_to_delete:
                out_pdf.delete_pages(pages_to_delete)

        if redact:
            recompress_images(out_pdf)
        out_pdf.save(out_path, garbage=4, deflate=True)
        out_pdf.close()
        written.append(out_path)

        # Optionally extract footnotes for this opinion
        if extract_footnotes:
            fn_dets = [
                d
                for src_idx in range(caption.page_index, key.page_index + 1)
                for d in pages_by_index[src_idx].detections
                if d.label == Label.FOOTNOTES
            ]
            if fn_dets:
                fn_base = name.replace(".pdf", "")
                fn_name = f"{fn_base}-footnotes.pdf"
                _extract_opinion_footnotes(
                    pdf,
                    fn_dets,
                    pages_by_index,
                    output_dir / fn_name,
                    skip_tighten=document.ocr_applied,
                )

    pdf.close()
    return written
