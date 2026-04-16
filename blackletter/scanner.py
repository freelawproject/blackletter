"""Scan a PDF with YOLO and produce annotated page images."""

from __future__ import annotations

import io
import logging
import re
import subprocess
import tempfile
from collections.abc import Callable, Iterator
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
    Label.SYLLABUS: (241, 196, 15),
    Label.EDITORIAL: (44, 62, 80),
    Label.JUDGES: (127, 140, 141),
    Label.TEXT_COLUMN: (174, 214, 241),
    Label.DOCKET: (82, 190, 128),
    Label.DATE: (245, 176, 65),
    Label.COURT: (72, 201, 176),
    Label.CITATION: (205, 97, 85),
}

# Max allowed deviation from median key-icon size (as a fraction).
_KEY_ICON_SIZE_TOLERANCE = 0.40
_KEY_ICON_MIN_RATIO = 1.5  # must be wider than tall
_KEY_ICON_MAX_RATIO = 4.0  # not absurdly wide (e.g. a full-width divider)

# Pixel tolerance for fuzzy exclusion matching (handles YOLO non-determinism on re-scan).
_EXCLUSION_TOLERANCE = 10


def _check_excluded(
    d: "Detection",
    excluded: "set[tuple[int, int, int, int]] | None",
    tolerance: int = _EXCLUSION_TOLERANCE,
) -> bool:
    """Return True if detection *d* matches any entry in *excluded*.

    Matches on (page_index, label_id) exactly, then checks bbox origin
    within +/-tolerance pixels to handle minor YOLO coordinate variation
    between scans of the same PDF.

    :param d: Detection to check.
    :param excluded: Set of (page_index, label_id, bbox_x1, bbox_y1)
        tuples to suppress.
    :param tolerance: Maximum pixel distance for fuzzy bbox matching.
    :returns: ``True`` if the detection is excluded.
    """
    if not excluded:
        return False
    pi = d.page_index
    lid = int(d.label)
    bx = round(d.bbox.x1)
    by = round(d.bbox.y1)
    if (pi, lid, bx, by) in excluded:
        return True
    return any(
        ex_pi == pi
        and ex_lid == lid
        and abs(bx - ex_bx) <= tolerance
        and abs(by - ex_by) <= tolerance
        for ex_pi, ex_lid, ex_bx, ex_by in excluded
    )


def _filter_key_icons_by_size(
    detections: list[Detection],
    tolerance: float = _KEY_ICON_SIZE_TOLERANCE,
) -> list[Detection]:
    """Keep only KEY_ICON detections whose size is close to the median.

    Key icons are all the same symbol, so valid detections should have
    very similar bounding-box dimensions. We compute the median width
    and height from high-confidence detections (>= 0.75) and reject any
    detection whose width or height deviates by more than *tolerance*
    from that median.

    :param detections: Full list of detections (all labels).
    :param tolerance: Max allowed deviation from median as a fraction.
    :returns: Filtered detection list.
    """
    from statistics import median

    key_dets = [d for d in detections if d.label == Label.KEY_ICON]

    # Always apply ratio filter, even for single detections
    def ratio_ok(d: Detection) -> bool:
        ratio = d.bbox.width / d.bbox.height if d.bbox.height > 0 else 0
        return _KEY_ICON_MIN_RATIO <= ratio <= _KEY_ICON_MAX_RATIO

    if len(key_dets) < 2:
        kept = [d for d in detections if d.label != Label.KEY_ICON or ratio_ok(d)]
        removed = len(detections) - len(kept)
        if removed:
            logger.info("Removed %d KEY_ICON detection(s) with bad ratio", removed)
        return kept

    # Use high-confidence detections to establish expected size
    high_conf = [d for d in key_dets if d.confidence >= 0.90]
    reference = high_conf if len(high_conf) >= 2 else key_dets

    med_w = median(d.bbox.width for d in reference)
    med_h = median(d.bbox.height for d in reference)

    def size_ok(d: Detection) -> bool:
        if d.label != Label.KEY_ICON:
            return True
        if not ratio_ok(d):
            return False
        if d.confidence >= 0.90:
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

    :param img_bgr: BGR image as a numpy array.
    :returns: Tuple of (left_x1, left_x2, right_x1, right_x2, center_x)
        in pixels.
    :raises ValueError: If column detection fails.
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


def _safe_detect_columns(
    img_rgb: Image.Image,
) -> tuple[float, float, float, float, float]:
    """Convert RGB PIL image to BGR and run :func:`detect_columns`.

    Returns ``(0, 0, 0, 0, 0)`` if detection fails, so callers can keep
    processing pages without a hard failure.

    :param img_rgb: Page image as a PIL ``Image`` in RGB.
    :returns: Column-bound tuple matching :func:`detect_columns`.
    """
    img_bgr = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
    try:
        return detect_columns(img_bgr)
    except Exception:
        return 0.0, 0.0, 0.0, 0.0, 0.0


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
                "tessedit_char_whitelist=0123456789-,",
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


def _parse_page_range(text: str) -> tuple[int, ...] | None:
    """Try to parse a page range like '31-32', '31,32', '31, 32' from text.

    Returns a tuple of (start, end) if a range is found, (single,) if just
    one number, or None if nothing plausible.
    """
    text = text.strip()
    # Match patterns like "31-32", "31–32", "31,32", "31, 32"
    m = re.match(r"(\d+)\s*[-–,]\s*(\d+)$", text)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if b > a and b - a <= 2:  # reasonable range (1-2 page span)
            return (a, b)
    # Single number
    digits = re.sub(r"\D", "", text)
    if digits:
        return (int(digits),)
    return None


def _extract_page_number(
    fitz_page, rect: fitz.Rect, hint: int = 0
) -> tuple[int, int | None] | None:
    """Extract a page number (or range) from a PDF region using text + OCR.

    Tightens the rect to actual text first, then tries the embedded PDF
    text layer and multiple tesseract PSM modes.  Filters out implausible
    results (too many digits, too far from expected position).

    Returns (page_number, page_number_end) where page_number_end is set
    only if a range was detected, or None if extraction fails.
    """
    # Tighten rect to actual text to avoid grabbing adjacent characters
    tight = _tighten_to_text(fitz_page, rect, padding=2.0)
    if tight is not None:
        rect = tight

    # Determine max expected digits from the hint
    max_digits = max(4, len(str(hint)) + 1) if hint > 0 else 5

    def _range_if_plausible(text: str) -> tuple[int, int] | None:
        """Return a plausible (a, b) range parsed from ``text``, else None."""
        parsed = _parse_page_range(text)
        if parsed and len(parsed) == 2:
            a, b = parsed
            if _plausible(a, hint, max_digits) or _plausible(b, hint, max_digits):
                return (a, b)
        return None

    # Method 1: PDF text layer, fast, try this first
    pdf_text = fitz_page.get_text("text", clip=rect).strip()

    # Check for page range first
    rng = _range_if_plausible(pdf_text)
    if rng is not None:
        return rng

    pdf_digits = re.sub(r"\D", "", pdf_text)

    # If PDF text gives a plausible number, use it without OCR
    if pdf_digits:
        n = int(pdf_digits)
        if _plausible(n, hint, max_digits):
            return (n, None)

    # Method 2: tesseract OCR, only if text layer failed
    # Try modes in order; stop as soon as we get a plausible result
    ocr_results: list[int] = []
    for psm in (8, 7, 13):
        raw = _ocr_region(fitz_page, rect, psm=psm)
        if raw:
            # Check for range in OCR output too
            rng = _range_if_plausible(raw)
            if rng is not None:
                return rng
            digits = re.sub(r"\D", "", raw)
            if digits:
                n = int(digits)
                if _plausible(n, hint, max_digits):
                    return (n, None)
                ocr_results.append(n)

    # Fallback: take any result without plausibility filter
    candidates = []
    if pdf_digits:
        candidates.append(int(pdf_digits))
    candidates.extend(ocr_results)
    if candidates:
        if hint > 0:
            n = min(candidates, key=lambda n: abs(n - hint))
        else:
            n = min(candidates, key=lambda n: len(str(n)))
        return (n, None)
    return None


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


_SCAN_WORKER_SCRIPT = """
import os, sys, json
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
import torch
torch.set_num_threads(2)

import fitz, cv2, numpy as np
from PIL import Image
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from blackletter.scanner import _safe_detect_columns, _extract_page_number, DPI
from blackletter.models import Label

pdf_path, model_path = sys.argv[1], sys.argv[2]
page_start, page_end = int(sys.argv[3]), int(sys.argv[4])
confidence, first_page = float(sys.argv[5]), int(sys.argv[6])
output_path = sys.argv[7]

model = YOLO(model_path)
pdf = fitz.open(pdf_path)
mat = fitz.Matrix(DPI / 72, DPI / 72)
BATCH = 4
pages_data = []

for bs in range(page_start, page_end, BATCH):
    be = min(bs + BATCH, page_end)
    imgs, meta = [], []
    for pi in range(bs, be):
        fp = pdf[pi]
        pix = fp.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        imgs.append(img)
        lx1, lx2, rx1, rx2, mid = _safe_detect_columns(img)
        meta.append((pi, fp.rect.width, fp.rect.height, pix.width, pix.height, (lx1, lx2, rx1, rx2, mid)))

    results = model(imgs, conf=confidence, verbose=False, device=None)
    for j, (pi, pdf_w, pdf_h, pw, ph, cols) in enumerate(meta):
        dets = []
        for box in results[j].boxes:
            dets.append({"bbox": box.xyxy[0].tolist(), "label": int(box.cls[0].item()), "conf": float(box.conf[0].item())})

        fp = pdf[pi]
        sx, sy = pdf_w / pw, pdf_h / ph
        pn, pn_end = None, None
        pn_dets = [d for d in dets if d["label"] == int(Label.PAGE_NUMBER)
                   and (d["bbox"][3]-d["bbox"][1]) >= 40 and (d["bbox"][2]-d["bbox"][0]) >= 40
                   and d["bbox"][1] >= 5 and d["bbox"][0] >= 5]
        pn_dets.sort(key=lambda d: d["conf"], reverse=True)
        for pd in pn_dets:
            bx1, by1, bx2, by2 = pd["bbox"]
            rect = fitz.Rect(bx1*sx, by1*sy, bx2*sx, by2*sy)
            r = _extract_page_number(fp, rect, hint=pi + first_page)
            if r is not None:
                pn, pn_end = r
                break

        pages_data.append({"index": pi, "pdf_width": pdf_w, "pdf_height": pdf_h,
                           "img_width": pw, "img_height": ph, "cols": cols,
                           "detections": dets, "page_number": pn, "page_number_end": pn_end})

    print(f"    Worker pages {bs}-{be}/{page_end}", flush=True)

pdf.close()
with open(output_path, "w") as f:
    json.dump(pages_data, f)
"""


def scan(
    pdf_path: Path,
    model: YOLO,
    confidence: float = CONFIDENCE_THRESHOLD,
    first_page: int = 1,
    output_dir: Path | None = None,
    shrink: bool = False,
    skip_ocr: bool = False,
    optimize: int = 1,
    output_name: str | None = None,
    device: str | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> Document:
    """Scan a PDF and return a Document with all detections.

    If the PDF has no text layer, automatically runs OCR first to add
    one (required for accurate page number extraction and redaction
    tightening).

    :param pdf_path: Path to the source PDF.
    :param model: Loaded YOLO model instance.
    :param confidence: Minimum confidence threshold for detections.
    :param first_page: First page number of the volume.
    :param output_dir: Directory for intermediate and output files.
    :param shrink: Downsample images to ~148 KB/page even if the PDF
        already has a text layer.
    :param skip_ocr: Never run OCR regardless of text layer presence.
    :param optimize: OCR optimization level (passed to ocrmypdf).
    :param output_name: Custom stem for output filenames.
    :param device: YOLO inference device (e.g. ``"cpu"``, ``"cuda:0"``).
    :param progress_callback: Optional callable(current, total, message).
    :returns: Document with all detections populated.
    """
    from blackletter.ocr import needs_ocr, ocr_pdf, _downsample_pdf

    dest_dir = output_dir if output_dir else Path(tempfile.gettempdir())
    dest_dir.mkdir(parents=True, exist_ok=True)

    out_stem = output_name if output_name else pdf_path.stem

    # Fix rotated pages first so all downstream coordinates are consistent.
    actual_path = _fix_rotated_pages(pdf_path, dest_dir, out_stem)

    if not skip_ocr and needs_ocr(actual_path):
        print("  PDF has no text layer, running OCR...")
        ocr_out = dest_dir / f"{out_stem}.pdf"
        ocr_pdf(actual_path, ocr_out, optimize=optimize)
        orig_mb = actual_path.stat().st_size / (1024 * 1024)
        actual_path = ocr_out
        final_mb = ocr_out.stat().st_size / (1024 * 1024)
        print(f"  OCR complete: {orig_mb:.1f} MB -> {final_mb:.1f} MB ({ocr_out.name})")
    elif shrink:
        # Skip downsample for bitonal (1-bit) PDFs, already compact
        _chk = fitz.open(str(actual_path))
        _imgs = _chk[0].get_images(full=True) if _chk.page_count else []
        _is_bitonal = bool(_imgs and _imgs[0][4] == 1)
        _chk.close()
        if _is_bitonal:
            mb = actual_path.stat().st_size / (1024 * 1024)
            print(f"  Source PDF: {mb:.1f} MB (bitonal, skipping downsample)")
            # Copy to output dir so output is self-contained
            if output_dir and actual_path.parent != dest_dir:
                import shutil

                dest_copy = dest_dir / f"{out_stem}.pdf"
                if not dest_copy.exists():
                    shutil.copy2(str(actual_path), str(dest_copy))
                    print(f"  Copied to {dest_copy.name}")
                actual_path = dest_copy
        else:
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
        # Copy to output dir so output is self-contained
        if output_dir and actual_path.parent != dest_dir:
            import shutil

            dest_copy = dest_dir / f"{out_stem}.pdf"
            if not dest_copy.exists():
                shutil.copy2(str(actual_path), str(dest_copy))
                print(f"  Copied to {dest_copy.name}")
            actual_path = dest_copy

    doc = Document(pdf_path=actual_path, first_page=first_page, ocr_applied=actual_path != pdf_path)
    pdf = fitz.open(actual_path)
    total_pages = len(pdf)

    import multiprocessing

    n_workers = max(1, multiprocessing.cpu_count() // 2)
    # Use parallel scan for large documents
    use_parallel = total_pages >= 40 and n_workers > 1

    print(
        f"  Detecting on {total_pages} pages ({n_workers} workers)..."
        if use_parallel
        else f"  Detecting on {total_pages} pages...",
        flush=True,
    )

    if use_parallel:
        import json as _json
        import subprocess as _sp

        model_path = (
            Path(model.ckpt_path) if hasattr(model, "ckpt_path") else Path(str(model.model))
        )
        from blackletter.tasks import split_page_ranges

        chunks = split_page_ranges(total_pages, n_workers)

        # Write worker script to temp file
        worker_script = Path(tempfile.gettempdir()) / "_bl_scan_worker.py"
        worker_script.write_text(_SCAN_WORKER_SCRIPT)

        # Launch subprocesses
        import sys as _sys

        sub_procs = []
        output_files = []
        for wid, (s, e) in enumerate(chunks):
            out_file = Path(tempfile.gettempdir()) / f"_bl_scan_{wid}.json"
            output_files.append(out_file)
            p = _sp.Popen(
                [
                    _sys.executable,
                    str(worker_script),
                    str(actual_path),
                    str(model_path),
                    str(s),
                    str(e),
                    str(confidence),
                    str(first_page),
                    str(out_file),
                ],
                stdout=_sp.PIPE,
                stderr=_sp.STDOUT,
                cwd=str(Path(__file__).resolve().parent.parent),
            )
            sub_procs.append((wid, p, s, e))

        # Stream output in real-time and wait for completion
        import select

        completed = set()
        fds = {p.stdout.fileno(): (wid, p, s, e) for wid, p, s, e in sub_procs}
        while len(completed) < len(sub_procs):
            readable, _, _ = select.select(list(fds.keys()), [], [], 0.5)
            for fd in readable:
                wid, p, s, e = fds[fd]
                line = p.stdout.readline()
                if line:
                    text = line.decode(errors="replace").strip()
                    if text:
                        print(text, flush=True)
            for wid, p, s, e in sub_procs:
                if wid in completed:
                    continue
                ret = p.poll()
                if ret is not None:
                    completed.add(wid)
                    for line in p.stdout:
                        text = line.decode(errors="replace").strip()
                        if text:
                            print(text, flush=True)
                    if ret != 0:
                        raise RuntimeError(f"YOLO worker {wid} failed (pages {s}-{e})")
                    done_pages = sum(e2 - s2 for w2, _, s2, e2 in sub_procs if w2 in completed)
                    print(
                        f"    Worker {wid + 1}/{len(chunks)} done ({done_pages}/{total_pages} pages)",
                        flush=True,
                    )
                    if progress_callback:
                        progress_callback(
                            done_pages, total_pages, f"Scanning {done_pages}/{total_pages} pages"
                        )

        # Assemble Page objects from JSON output files
        for wid, out_file in enumerate(output_files):
            chunk_data = _json.loads(out_file.read_text())
            out_file.unlink(missing_ok=True)
            for pd in chunk_data:
                lx1, lx2, rx1, rx2, mid = pd["cols"]
                page = Page(
                    index=pd["index"],
                    pdf_width=pd["pdf_width"],
                    pdf_height=pd["pdf_height"],
                    img_width=pd["img_width"],
                    img_height=pd["img_height"],
                    col_left_x1=lx1,
                    col_left_x2=lx2,
                    col_right_x1=rx1,
                    col_right_x2=rx2,
                    midpoint=mid,
                )
                for det in pd["detections"]:
                    page.detections.append(
                        Detection(
                            bbox=BBox.from_xyxy(det["bbox"]),
                            label=Label(det["label"]),
                            confidence=det["conf"],
                            page_index=pd["index"],
                        )
                    )
                page.page_number = pd["page_number"]
                page.page_number_end = pd["page_number_end"]
                doc.pages.append(page)

        worker_script.unlink(missing_ok=True)
        # Sort by page index
        doc.pages.sort(key=lambda p: p.index)

    else:
        # Single-process path for small documents
        BATCH_SIZE = 4
        mat = fitz.Matrix(DPI / 72, DPI / 72)

        def _render_batch(start: int, end: int):
            imgs: list[Image.Image] = []
            meta: list[tuple] = []
            for page_idx in range(start, end):
                fitz_page = pdf[page_idx]
                pix = fitz_page.get_pixmap(matrix=mat)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                imgs.append(img)
                lx1, lx2, rx1, rx2, mid = _safe_detect_columns(img)
                meta.append((page_idx, fitz_page, pix.width, pix.height, (lx1, lx2, rx1, rx2, mid)))
            return imgs, meta

        def _process_results(batch_results, batch_meta):
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
                pn_dets = [
                    d
                    for d in pn_dets
                    if d.bbox.height >= 40
                    and d.bbox.width >= 40
                    and d.bbox.y1 >= 5
                    and d.bbox.x1 >= 5
                ]
                pn_dets.sort(key=lambda d: d.confidence, reverse=True)
                for pn_det in pn_dets:
                    b = pn_det.bbox.to_pdf(page.scale_x, page.scale_y)
                    rect = fitz.Rect(b.x1, b.y1, b.x2, b.y2)
                    result = _extract_page_number(fitz_page, rect, hint=page_idx + first_page)
                    if result is not None:
                        page.page_number, page.page_number_end = result
                doc.pages.append(page)

        from concurrent.futures import ThreadPoolExecutor

        batches = [(s, min(s + BATCH_SIZE, total_pages)) for s in range(0, total_pages, BATCH_SIZE)]
        with ThreadPoolExecutor(max_workers=1) as render_pool:
            next_future = render_pool.submit(_render_batch, *batches[0])
            for i, (batch_start, batch_end) in enumerate(batches):
                batch_imgs, batch_meta = next_future.result()
                if i + 1 < len(batches):
                    next_future = render_pool.submit(_render_batch, *batches[i + 1])
                batch_results = model(batch_imgs, conf=confidence, verbose=False, device=device)
                _process_results(batch_results, batch_meta)
                print(f"    Page {batch_end}/{total_pages}", flush=True)
                if progress_callback:
                    progress_callback(
                        batch_end, total_pages, f"Scanning page {batch_end}/{total_pages}"
                    )

    # Sequential correction pass: fix misreads using neighbor context
    _correct_page_numbers(doc.pages)

    return doc


def _nearest_numbered_neighbor(
    pages: list[Page],
    i: int,
    step: int,
    max_look: int,
) -> tuple[int | None, int]:
    """Scan up to ``max_look`` pages in direction ``step`` from index ``i``.

    :param pages: Pages to scan over.
    :param i: Index to start from (exclusive).
    :param step: ``-1`` to scan before, ``+1`` to scan after.
    :param max_look: Maximum distance to search.
    :returns: ``(page_number, distance)`` of the nearest page with a
        non-null ``page_number``, or ``(None, 0)`` if none is found.
    """
    n = len(pages)
    for dist in range(1, max_look + 1):
        j = i + step * dist
        if not 0 <= j < n:
            return None, 0
        if pages[j].page_number is not None:
            return pages[j].page_number, dist
    return None, 0


def _correct_page_numbers(pages: list[Page]) -> None:
    """Fix page number misreads using sequential context.

    Makes multiple passes.  On each pass, for every page, looks for the
    nearest neighbors (up to 10 pages away) that agree with each other
    sequentially.  If they do, interpolates the expected value.

    Pages with ranges are left alone if either number fits the sequence.

    Repeats until no more corrections are made.
    """
    n = len(pages)
    if n < 3:
        return

    max_look = 10  # how far to search for a good neighbor

    for _pass in range(5):  # up to 5 correction passes
        corrections = 0
        for i in range(n):
            prev_num, prev_dist = _nearest_numbered_neighbor(pages, i, -1, max_look)
            next_num, next_dist = _nearest_numbered_neighbor(pages, i, +1, max_look)

            if prev_num is None or next_num is None:
                continue

            # Check if neighbors are consistent with each other
            total_dist = prev_dist + next_dist
            if next_num - prev_num != total_dist:
                continue

            expected = prev_num + prev_dist

            # If page has a range and either number matches, leave it alone
            if pages[i].page_number_end is not None:
                if pages[i].page_number == expected or pages[i].page_number_end == expected:
                    continue

            if pages[i].page_number != expected:
                pages[i].page_number = expected
                pages[i].page_number_end = None  # clear range if correcting
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

    Uses PyMuPDF to draw directly on the PDF pages (no rasterization).

    :param pdf_path: Path to the source PDF.
    :param document: Scanned Document with detections.
    :param output_path: Path for the output PDF.
    :param labels: If provided, only draw these labels. ``None`` draws
        all.
    :returns: Path to the written PDF.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = fitz.open(pdf_path)

    for page in document.pages:
        fitz_page = pdf[page.index]
        shape = fitz_page.new_shape()

        dets = page.detections
        dets = _filter_key_icons_by_size(dets)
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
    excluded: set[tuple[int, int, int, int]] | None = None,
) -> list[tuple[Detection, Detection]]:
    """Pair each CASE_CAPTION with the next KEY_ICON in reading order.

    Key icons are the reliable anchor. After each key icon, the first
    caption that follows starts the next opinion. Only the first caption
    after a key icon is used; subsequent captions before the next key
    icon are ignored to avoid false-positive splits.

    :param document: Document with pages and detections.
    :param excluded: Optional set of (page_index, label_id,
        round(bbox_x1), round(bbox_y1)) tuples to suppress specific
        detections.
    :returns: List of (caption, key) detection pairs.
    """

    def _is_excluded(d):
        return _check_excluded(d, excluded)

    if not document.pages:
        return []

    caption_thresh = LABEL_CONFIDENCE.get(Label.CASE_CAPTION, CONFIDENCE_THRESHOLD)
    key_thresh = LABEL_CONFIDENCE.get(Label.KEY_ICON, CONFIDENCE_THRESHOLD)

    captions: list[Detection] = []
    all_keys: list[Detection] = []
    for page in document.pages:
        for d in page.detections:
            if _is_excluded(d):
                continue
            if d.label == Label.CASE_CAPTION and d.confidence >= caption_thresh:
                captions.append(d)
            elif d.label == Label.KEY_ICON and d.confidence >= key_thresh:
                all_keys.append(d)

    keys = [d for d in _filter_key_icons_by_size(all_keys) if d.label == Label.KEY_ICON]

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


_WORD_CACHE_ATTR = "_blackletter_word_cache"


def _get_page_words(fitz_page) -> list[tuple]:
    """Return all non-empty words on the page, caching the result.

    The cache is stored on the parent fitz.Document keyed by page number.
    It is populated lazily on first call and remains valid until
    apply_redactions() modifies the page content.
    """
    doc = fitz_page.parent
    cache = getattr(doc, _WORD_CACHE_ATTR, None)
    if cache is None:
        cache = {}
        setattr(doc, _WORD_CACHE_ATTR, cache)
    page_num = fitz_page.number
    if page_num not in cache:
        words = fitz_page.get_text("words")
        cache[page_num] = [w for w in words if w[4].strip()]
    return cache[page_num]


def _words_in_rect(fitz_page, rect: fitz.Rect) -> list[tuple]:
    """Return cached words whose bounding box intersects the given rect."""
    all_words = _get_page_words(fitz_page)
    rx0, ry0, rx1, ry1 = rect.x0, rect.y0, rect.x1, rect.y1
    return [w for w in all_words if w[0] < rx1 and w[2] > rx0 and w[1] < ry1 and w[3] > ry0]


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
    # words = [(x0, y0, x1, y1, word, block_no, line_no, word_no), ...]
    words = _words_in_rect(fitz_page, rect)
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
    words = _words_in_rect(fitz_page, clip)
    if not words:
        return clip.y0
    return max(w[3] for w in words)


def _text_x_bounds(fitz_page, clip: fitz.Rect, padding: float = 2.0) -> tuple[float, float]:
    """Return (left, right) x-coordinates of the text extent within clip."""
    words = _words_in_rect(fitz_page, clip)
    if not words:
        return clip.x0, clip.x1
    return (
        min(w[0] for w in words) - padding,
        max(w[2] for w in words) + padding,
    )


def _column_bounds_pdf(page: Page) -> tuple[float, float, float, float, float] | None:
    """Return (left_x0, left_x1, right_x0, right_x1, col_boundary) in PDF points.

    Uses TEXT_COLUMN detections (the same source of truth used for
    headnote redaction snapping.  Returns None if no TEXT_COLUMN
    detections are available (caller should skip masking for that page).

    col_boundary is the midpoint of the gap between columns) used to
    decide which column a detection belongs to.
    """
    sx, sy = page.scale_x, page.scale_y
    tc_dets = [d for d in page.detections if d.label == Label.TEXT_COLUMN]

    if len(tc_dets) < 2:
        return None

    tc_sorted = sorted(tc_dets, key=lambda d: d.bbox.center_x)
    left = tc_sorted[0].bbox.to_pdf(sx, sy)
    right = tc_sorted[-1].bbox.to_pdf(sx, sy)
    col_boundary = (left.x2 + right.x1) / 2
    return left.x1, left.x2, right.x1, right.x2, col_boundary


def _outside_opinion_rects(
    page: Page,
    page_width: float,
    caption: Detection,
    key: Detection,
    is_first: bool,
    is_last: bool,
) -> list[fitz.Rect]:
    """Return rects for content outside the opinion span on this page.

    Uses TEXT_COLUMN detection boundaries so masks align with actual
    column edges rather than an arbitrary midpoint.

    Reading order: left column top-to-bottom, then right column top-to-bottom.

    First page masks everything before the caption in reading order.
    Last page masks everything after the key in reading order.
    """
    sx, sy = page.scale_x, page.scale_y
    header_bottom, footer_top = _margin_bounds(page)
    bounds = _column_bounds_pdf(page)
    if bounds is None:
        return []
    left_x0, left_x1, right_x0, right_x1, col_boundary = bounds
    rects: list[fitz.Rect] = []

    if is_first:
        cap_box = caption.bbox.to_pdf(sx, sy)
        cap_col = "LEFT" if cap_box.center_x < col_boundary else "RIGHT"
        if cap_col == "LEFT":
            # Mask left column above caption only
            rects.append(fitz.Rect(left_x0, header_bottom, left_x1, cap_box.y1))
        else:
            # Mask entire left column + right column above caption
            rects.append(fitz.Rect(left_x0, header_bottom, left_x1, footer_top))
            rects.append(fitz.Rect(right_x0, header_bottom, right_x1, cap_box.y1))

    if is_last:
        key_box = key.bbox.to_pdf(sx, sy)
        key_col = "LEFT" if key_box.center_x < col_boundary else "RIGHT"
        if key_col == "RIGHT":
            # Mask right column below key only
            rects.append(fitz.Rect(right_x0, key_box.y2, right_x1, footer_top))
        else:
            # Mask left column below key + entire right column
            rects.append(fitz.Rect(left_x0, key_box.y2, left_x1, footer_top))
            rects.append(fitz.Rect(right_x0, header_bottom, right_x1, footer_top))

    return [r for r in rects if r.y0 < r.y1 and r.x0 < r.x1]


def _build_opinions_data(
    opinions: list[tuple[Detection, Detection]],
    pages_by_index: dict[int, Page],
    src_pdf: fitz.Document,
) -> list[dict]:
    """Serialise paired opinions to the opinions.json dict shape.

    For each ``(caption, key)`` pair this computes the per-page
    ``_outside_opinion_rects`` and assembles the metadata dict written
    to ``opinions.json``.

    :param opinions: Paired ``(caption, key)`` detections from
        :func:`_pair_opinions`.
    :param pages_by_index: Mapping of page index to :class:`Page`.
    :param src_pdf: Open source PDF used to look up per-page widths.
    :returns: List of opinion dicts ready for JSON serialisation.
    """
    opinions_data: list[dict] = []
    for caption, key in opinions:
        outside_rects: list[dict] = []
        for pi in range(caption.page_index, key.page_index + 1):
            page = pages_by_index.get(pi)
            if page is None:
                continue
            is_first = pi == caption.page_index
            is_last = pi == key.page_index
            pw = src_pdf[pi].rect.width
            for rect in _outside_opinion_rects(page, pw, caption, key, is_first, is_last):
                outside_rects.append(
                    {
                        "page_index": pi,
                        "x0": round(rect.x0, 1),
                        "y0": round(rect.y0, 1),
                        "x1": round(rect.x1, 1),
                        "y1": round(rect.y1, 1),
                    }
                )
        has_image = any(
            d.label == Label.IMAGE
            for pi2 in range(caption.page_index, key.page_index + 1)
            if pi2 in pages_by_index
            for d in pages_by_index[pi2].detections
        )
        opinions_data.append(
            {
                "caption_page": caption.page_index,
                "caption_label": caption.label.name,
                "caption_bbox": [
                    round(caption.bbox.x1, 1),
                    round(caption.bbox.y1, 1),
                    round(caption.bbox.x2, 1),
                    round(caption.bbox.y2, 1),
                ],
                "key_page": key.page_index,
                "key_label": key.label.name,
                "key_bbox": [
                    round(key.bbox.x1, 1),
                    round(key.bbox.y1, 1),
                    round(key.bbox.x2, 1),
                    round(key.bbox.y2, 1),
                ],
                "page_count": key.page_index - caption.page_index + 1,
                "outside_rects": outside_rects,
                "has_image": has_image,
            }
        )
    return opinions_data


def _make_add_safe(
    fitz_page,
    pn_rects: list[fitz.Rect],
    collector: list[tuple[fitz.Rect, tuple]] | None = None,
) -> Callable[[fitz.Rect, tuple], None]:
    """Build a recursive rect-adder that clips around page-number boxes.

    The returned closure adds ``(rect, fill)`` as a PyMuPDF redaction
    annotation unless the rect intersects any page-number rect, in which
    case it splits the rect into up-to-four non-overlapping slices and
    recurses.

    :param fitz_page: Page to attach redaction annotations to.
    :param pn_rects: Page-number rectangles to preserve (never redact).
    :param collector: If provided, each ``(rect, fill)`` pair that ends
        up annotated is also appended here (used for bitonal-safe
        redaction where the image pixels must be overwritten directly).
    :returns: The ``add_safe`` closure.
    """

    def add_safe(rect: fitz.Rect, fill) -> None:
        if rect.is_empty:
            return
        for pn in pn_rects:
            if rect.intersects(pn):
                if rect.y0 < pn.y0:
                    add_safe(fitz.Rect(rect.x0, rect.y0, rect.x1, pn.y0), fill)
                if rect.y1 > pn.y1:
                    add_safe(fitz.Rect(rect.x0, pn.y1, rect.x1, rect.y1), fill)
                top = max(rect.y0, pn.y0)
                bot = min(rect.y1, pn.y1)
                if rect.x0 < pn.x0 and top < bot:
                    add_safe(fitz.Rect(rect.x0, top, pn.x0, bot), fill)
                if rect.x1 > pn.x1 and top < bot:
                    add_safe(fitz.Rect(pn.x1, top, rect.x1, bot), fill)
                return
        if collector is not None:
            collector.append((fitz.Rect(rect), fill))
        fitz_page.add_redact_annot(rect, fill=fill)

    return add_safe


_CS_INSET = 3.0
_CS_MIN_PX = 40
_CS_MAX_PX = 60


def _iter_case_sequence_rects(
    page: Page,
    sx: float,
    sy: float,
    caption_rects: list[fitz.Rect],
    excluded: set | None,
) -> Iterator[fitz.Rect]:
    """Yield PDF-point redaction rects for every CASE_SEQUENCE detection.

    Applies the 40-60 px clamp, 3 pt inset, and caption-overlap clipping.
    Skips excluded detections and rects that end up empty.

    :param page: Page whose detections are scanned.
    :param sx: Horizontal image-to-PDF scale factor.
    :param sy: Vertical image-to-PDF scale factor.
    :param caption_rects: PDF-point CASE_CAPTION rects on this page; any
        CASE_SEQUENCE rect that overlaps a caption is trimmed to the
        area above it or dropped entirely.
    :param excluded: Set of excluded detection-bbox tuples, or ``None``.
    """
    from blackletter.models import BBox

    for d in page.detections:
        if d.label != Label.CASE_SEQUENCE:
            continue
        if _check_excluded(d, excluded):
            continue
        bx0, by0 = d.bbox.x1, d.bbox.y1
        bx1, by1 = d.bbox.x2, d.bbox.y2
        cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
        pw = max(_CS_MIN_PX, min(bx1 - bx0, _CS_MAX_PX))
        ph = max(_CS_MIN_PX, min(by1 - by0, _CS_MAX_PX))
        bx0, bx1 = cx - pw / 2, cx + pw / 2
        by0, by1 = cy - ph / 2, cy + ph / 2
        b = BBox(bx0, by0, bx1, by1).to_pdf(sx, sy)
        rect = fitz.Rect(
            b.x1 + _CS_INSET,
            b.y1 + _CS_INSET,
            b.x2 - _CS_INSET,
            b.y2 - _CS_INSET,
        )
        for cap_r in caption_rects:
            if rect.intersects(cap_r):
                if rect.y0 < cap_r.y0:
                    rect = fitz.Rect(rect.x0, rect.y0, rect.x1, cap_r.y0)
                else:
                    rect = fitz.Rect(0, 0, 0, 0)
        if rect.is_empty or rect.y0 >= rect.y1 or rect.x0 >= rect.x1:
            continue
        yield rect


def _clip_headnote_rect(
    fitz_page,
    rect: fitz.Rect,
    header_bottom: float,
    footer_top: float,
    ocr_applied: bool,
) -> fitz.Rect | None:
    """Tighten a headnote rect to page text and clamp it to margin bounds.

    When ``ocr_applied`` is True the rect is only clamped to
    ``(header_bottom, footer_top)``; when False it is further
    ``_tighten_to_text``-adjusted and clipped to text bounds on all four
    sides. Returns ``None`` when the resulting rect has no area.

    :param fitz_page: PyMuPDF page used to measure text bounds.
    :param rect: Input rect in PDF points.
    :param header_bottom: Upper Y cut-off (from :func:`_margin_bounds`).
    :param footer_top: Lower Y cut-off (from :func:`_margin_bounds`).
    :param ocr_applied: Whether the PDF has a reliable text layer.
    :returns: The clipped rect, or ``None`` if degenerate.
    """
    if not ocr_applied:
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
        rect = fitz.Rect(
            rect.x0,
            max(rect.y0, header_bottom),
            rect.x1,
            min(rect.y1, footer_top),
        )
    if rect.y0 < rect.y1 and rect.x0 < rect.x1:
        return rect
    return None


def _write_detections_sidecar(document: Document, output_dir: Path) -> int:
    """Serialise all detections to ``output_dir/detections.json``.

    :param document: Source document whose pages/detections are dumped.
    :param output_dir: Directory to write the sidecar into; created if
        necessary.
    :returns: Number of detection rows written.
    """
    import json as _json

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for page in document.pages:
        for det in page.detections:
            rows.append(
                {
                    "page_index": det.page_index,
                    "label": det.label.name,
                    "label_id": int(det.label),
                    "confidence": round(det.confidence, 3),
                    "bbox": [
                        round(det.bbox.x1, 1),
                        round(det.bbox.y1, 1),
                        round(det.bbox.x2, 1),
                        round(det.bbox.y2, 1),
                    ],
                    "page_number": page.page_number,
                    "img_width": page.img_width,
                    "img_height": page.img_height,
                }
            )
    (output_dir / "detections.json").write_text(_json.dumps(rows))
    return len(rows)


def _write_pages_meta_sidecar(document: Document, output_dir: Path) -> None:
    """Serialise column bounds and midpoint to ``output_dir/pages_meta.json``.

    :param document: Source document.
    :param output_dir: Directory to write the sidecar into; created if
        necessary.
    """
    import json as _json

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta: dict[int, dict] = {}
    for page in document.pages:
        meta[page.index] = {
            "col_left_x1": page.col_left_x1,
            "col_left_x2": page.col_left_x2,
            "col_right_x1": page.col_right_x1,
            "col_right_x2": page.col_right_x2,
            "midpoint": page.midpoint,
        }
    (output_dir / "pages_meta.json").write_text(_json.dumps(meta))


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
    reporter: str | None = None,
) -> Detection | None:
    """Find the end of the headnote zone.

    Looks for the first DIVIDER or CASE_METADATA after the caption.
    Returns None if neither is found (caller should try fallback).
    """
    cap_sk = caption.sort_key(mid)
    after_caption = [d for d in opinion_dets if d.sort_key(mid) > cap_sk]

    # For Supreme Court opinions, prefer SYLLABUS as the boundary
    if reporter and reporter.lower().replace("-", "") == "sct":
        for d in after_caption:
            if d.label == Label.SYLLABUS:
                return d

    for d in after_caption:
        if d.label == Label.DIVIDER:
            return d

    # Only use CASE_METADATA as a fallback if there are actual headnotes
    # between the caption and the metadata.  A false CASE_METADATA with
    # no headnotes would trigger spurious redactions.
    has_headnotes = any(d.label in (Label.HEADNOTE, Label.BACKGROUND) for d in after_caption)
    if has_headnotes:
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


def _find_redaction_start(
    opinion_dets: list[Detection],
    caption: Detection,
    mid: float,
) -> Detection:
    """Find the start of the headnote zone.

    If a BACKGROUND detection exists after the caption (from the large model),
    use its top edge as the redaction start. Otherwise fall back to the caption.
    """
    cap_sk = caption.sort_key(mid)
    for d in opinion_dets:
        if d.label == Label.BACKGROUND and d.sort_key(mid) > cap_sk:
            return d
    return caption


def _redaction_rects(
    caption: Detection,
    end_marker: Detection,
    pages_by_index: dict[int, Page],
    start_marker: Detection | None = None,
) -> list[tuple[int, fitz.Rect]]:
    """Generate column-aware redaction rectangles from start to end marker.

    If start_marker is provided (e.g. a BACKGROUND detection), redaction
    begins at start_marker.y1 instead of caption.y2. Otherwise falls back
    to starting below the caption.

    Returns a list of (source_page_index, rect_in_pdf_points).
    """
    start = start_marker or caption
    start_page = pages_by_index[start.page_index]
    end_page = pages_by_index[end_marker.page_index]
    start_box = start.bbox.to_pdf(start_page.scale_x, start_page.scale_y)
    end_box = end_marker.bbox.to_pdf(end_page.scale_x, end_page.scale_y)

    rects: list[tuple[int, fitz.Rect]] = []

    for page_idx in range(start.page_index, end_marker.page_index + 1):
        page = pages_by_index[page_idx]
        sx = page.scale_x
        mid_pdf = page.midpoint * sx
        header_bottom, footer_top = _margin_bounds(page)

        col_bounds = [
            (page.col_left_x1 * sx, page.col_left_x2 * sx),
            (page.col_right_x1 * sx, page.col_right_x2 * sx),
        ]

        start_col = 0 if start_box.center_x < mid_pdf else 1
        end_col = 0 if end_box.center_x < mid_pdf else 1
        is_start = page_idx == start.page_index
        is_end = page_idx == end_marker.page_index

        for col in range(2):
            cx1, cx2 = col_bounds[col]
            y_top = header_bottom
            y_bot = footer_top

            if is_start:
                if col < start_col:
                    continue
                if col == start_col:
                    # If using a BACKGROUND marker, start at its top edge;
                    # if falling back to caption, start below it
                    y_top = start_box.y1 if start_marker else start_box.y2

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
        Label.EDITORIAL,
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
    excluded: set[tuple[int, int, int, int]] | None = None,
) -> list[Path]:
    """Split a PDF into individual opinion files based on caption→key pairs.

    Modes:
        debug (draw_labels + debug_redactions): bounding boxes and red overlays
        redact: actual PDF redactions, white for outside-opinion content,
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
    opinions = _pair_opinions(document, excluded=excluded)

    if not opinions:
        return []

    mid = document.pages[0].midpoint
    pages_by_index = {p.index: p for p in document.pages}

    pdf = fitz.open(pdf_path)
    written: list[Path] = []

    # Each opinion runs from its caption page to its key page.
    page_ranges: list[tuple[int, int]] = []
    for idx, (caption, key) in enumerate(opinions):
        page_ranges.append((caption.page_index, key.page_index))

    # Build filename prefix from reporter/volume if available
    prefix = ""
    if document.reporter and document.volume:
        prefix = f"{document.reporter}.{document.volume}."

    # Compute base names for each opinion using full page range
    from collections import Counter

    base_names: list[str] = []
    for idx, (caption, _key) in enumerate(opinions):
        start_idx, end_idx = page_ranges[idx]
        start_page = pages_by_index.get(start_idx)
        end_page = pages_by_index.get(end_idx)

        # Opinion starting on a range page → use end number
        if start_page and start_page.page_number_end:
            first_num = start_page.page_number_end
        elif start_page and start_page.page_number:
            first_num = start_page.page_number
        else:
            first_num = start_idx + first_page

        # Opinion ending on a range page → use first number
        if end_page and end_page.page_number_end:
            last_num = end_page.page_number
        elif end_page and end_page.page_number:
            last_num = end_page.page_number
        else:
            last_num = end_idx + first_page

        base_names.append(f"{prefix}{first_num:04d}-{last_num:04d}")

    name_counts = Counter(base_names)
    name_seq: dict[str, int] = {}

    for i, (caption, key) in enumerate(opinions):
        start_idx, end_idx = page_ranges[i]
        base = base_names[i]
        if name_counts[base] > 1:
            seq = name_seq.get(base, 0) + 1
            name_seq[base] = seq
            name = f"{base}-{seq}.pdf"
        else:
            name = f"{base}.pdf"

        out_path = output_dir / name

        out_pdf = fitz.open()
        out_pdf.insert_pdf(pdf, from_page=start_idx, to_page=end_idx)

        # Detect bitonal for safe redaction
        _s_imgs = out_pdf[0].get_images(full=True) if out_pdf.page_count else []
        _is_bitonal = bool(_s_imgs and _s_imgs[0][4] == 1)

        cap_key = caption.sort_key(mid)
        key_key = key.sort_key(mid)

        # Gather all detections in this opinion's span
        opinion_dets = []
        for src_idx in range(start_idx, end_idx + 1):
            if src_idx not in pages_by_index:
                continue
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
                reporter=document.reporter,
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

            # Refine block rects to line-level with docTR
            # Keep block-level rects for page-deletion coverage check (line-level
            # rects have gaps between lines and won't sum to 95% coverage).
            block_headnote_rects = headnote_rects
            if headnote_rects:
                from blackletter.refine import refine_headnote_rects

                headnote_rects = refine_headnote_rects(pdf, headnote_rects, pages_by_index)

        for local_idx, src_idx in enumerate(range(start_idx, end_idx + 1)):
            if src_idx not in pages_by_index:
                continue
            page = pages_by_index[src_idx]
            fitz_page = out_pdf[local_idx]
            is_first = src_idx == start_idx
            is_last = src_idx == end_idx

            if redact:
                # ── Actual redaction mode ──────────────────────────
                sx, sy = page.scale_x, page.scale_y

                # Collect page number rects (these must never be redacted)
                pn_rects = []
                for d in page.detections:
                    if d.label == Label.PAGE_NUMBER:
                        b = d.bbox.to_pdf(sx, sy)
                        r = fitz.Rect(b.x1, b.y1, b.x2, b.y2)
                        tight = _tighten_to_text(fitz_page, r, skip=False)
                        pn_rects.append(tight if tight is not None else r)

                _page_redact_rects: list[tuple[fitz.Rect, tuple]] = []
                add_safe = _make_add_safe(fitz_page, pn_rects, collector=_page_redact_rects)

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
                    if rect_page_idx != src_idx:
                        continue
                    clipped = _clip_headnote_rect(
                        fitz_page, rect, header_bottom, footer_top, document.ocr_applied
                    )
                    if clipped is not None:
                        add_safe(clipped, (0, 0, 0))

                # Per-detection redactions (skip black outside opinion bounds)
                redact_labels = _REDACT_WHITE | _REDACT_BLACK
                for d in page.detections:
                    if d.label not in redact_labels:
                        continue
                    if _check_excluded(d, excluded):
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
                    tight = _tighten_to_text(fitz_page, rect, skip=False)
                    if tight is not None:
                        rect = tight
                    fill = (0, 0, 0) if d.label in _REDACT_BLACK else (1, 1, 1)
                    add_safe(rect, fill)

                # KEY_ICON redactions: white before caption, black at end
                # Only redact icons that pass the ratio filter (same as pairing)
                valid_key_icons = {
                    id(d)
                    for d in _filter_key_icons_by_size(
                        [x for x in page.detections if x.label == Label.KEY_ICON]
                    )
                }
                for d in page.detections:
                    if d.label != Label.KEY_ICON:
                        continue
                    if id(d) not in valid_key_icons:
                        continue
                    if _check_excluded(d, excluded):
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
                        # Outside opinion bounds, always white
                        add_safe(rect, (1, 1, 1))
                    else:
                        # Within opinion (including own key), black
                        add_safe(rect, (0, 0, 0))

                if _is_bitonal and _page_redact_rects:
                    from blackletter.process import _redact_bitonal_image

                    _redact_bitonal_image(fitz_page, out_pdf, _page_redact_rects)
                    fitz_page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
                else:
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
                        tight = _tighten_to_text(fitz_page, rect, skip=False)
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
        # Use block-level rects (pre-refinement) for coverage. Line-level rects
        # have gaps between lines and won't reliably sum to >=95%.
        if redact_mode == "masked" and block_headnote_rects:
            pages_to_delete: list[int] = []
            for local_idx, src_idx in enumerate(range(start_idx, end_idx + 1)):
                if src_idx not in pages_by_index:
                    continue
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
                for rect_page_idx, rect in block_headnote_rects:
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

        out_pdf.save(out_path, garbage=4, deflate=True)
        out_pdf.close()
        if redact and out_path.stat().st_size > 50 * 1024 * 1024:
            reopen = fitz.open(str(out_path))
            sample_imgs = reopen[0].get_images(full=True) if reopen.page_count else []
            is_bitonal = bool(sample_imgs and sample_imgs[0][4] == 1)
            if is_bitonal:
                reopen.close()
            else:
                print(
                    f"    Compressing {out_path.name} ({out_path.stat().st_size // (1024 * 1024)} MB)...",
                    flush=True,
                )
                recompress_images(reopen)
                data = reopen.tobytes(garbage=4, deflate=True)
                reopen.close()
                out_path.write_bytes(data)
        print(f"    Split {i + 1}/{len(opinions)}", flush=True)
        written.append(out_path)

        # Optionally extract footnotes for this opinion
        if extract_footnotes:
            fn_dets = [
                d
                for src_idx in range(start_idx, end_idx + 1)
                if src_idx in pages_by_index
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
