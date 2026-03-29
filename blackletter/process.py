"""Unified process command: scan, verify, and split in one pass."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import fitz
from ultralytics import YOLO

from blackletter.models import BBox, Detection, Label
from blackletter.refine import refine_headnote_rects
from blackletter.scanner import (
    scan,
    split_opinions,
    recompress_images,
    _pair_opinions,
    _check_excluded,
    _outside_opinion_rects,
    _margin_bounds,
    _find_redaction_end,
    _headnote_fallback_rects,
    _redaction_rects,
    _find_redaction_start,
    _tighten_to_text,
    _text_bottom,
    _text_x_bounds,
    _REDACT_WHITE,
    _REDACT_BLACK,
    _MARGIN_LABELS,
    _filter_key_icons_by_size,
    LABEL_CONFIDENCE,
    CONFIDENCE_THRESHOLD,
)

logger = logging.getLogger("blackletter")

DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "small.pt"


def _build_output_dir(args: argparse.Namespace) -> Path:
    """Build output dir: base / reporter / volume / first_page."""
    base = args.output
    if args.reporter:
        base = base / args.reporter
    if args.volume:
        base = base / str(args.volume)
    base = base / str(args.first_page)
    return base


def _build_masked_opinions(
    masked_paths: list[Path],
    opinions: list,
    document,
    output_dir: Path,
    excluded: set[tuple[int, int, int, int]] | None = None,
) -> list[Path]:
    """Consolidate same-page masked opinions into single PDF pages.

    For opinions that all start and end on the same source page, produces
    a single 1-page PDF extracted from the source with:
    - All same-page opinions visible
    - Content from multi-page opinions (that start on this page but
      continue to the next) whited out
    - Headnote/per-detection redactions applied
    """
    pages_by_index = {p.index: p for p in document.pages}
    mid = document.pages[0].midpoint

    # Group consecutive single-page opinions on the same source page
    groups: list[list[int]] = []
    i = 0
    while i < len(opinions):
        caption, key = opinions[i]
        if caption.page_index == key.page_index:
            group = [i]
            j = i + 1
            while j < len(opinions):
                next_cap, next_key = opinions[j]
                if (
                    next_cap.page_index == next_key.page_index
                    and next_cap.page_index == caption.page_index
                ):
                    group.append(j)
                    j += 1
                else:
                    break
            groups.append(group)
            i = j
        else:
            groups.append([i])
            i += 1

    prefix = ""
    if document.reporter and document.volume:
        prefix = f"{document.reporter}.{document.volume}."

    src_pdf = fitz.open(str(document.pdf_path))
    final_paths: list[Path] = []

    for group in groups:
        if len(group) == 1:
            final_paths.append(masked_paths[group[0]])
            continue

        # Multiple same-page opinions — build a single-page PDF
        first_cap, _ = opinions[group[0]]
        _, last_key = opinions[group[-1]]
        src_page_idx = first_cap.page_index
        page = pages_by_index[src_page_idx]
        first_num = page.page_number

        if first_num is not None:
            name = f"{prefix}{first_num:04d}-{first_num:04d}.pdf"
        else:
            fb = src_page_idx + document.first_page
            name = f"{prefix}{fb:04d}-{fb:04d}.pdf"

        out_path = output_dir / name

        # Extract source page
        out_pdf = fitz.open()
        out_pdf.insert_pdf(src_pdf, from_page=src_page_idx, to_page=src_page_idx)
        fitz_page = out_pdf[0]

        sx, sy = page.scale_x, page.scale_y

        # Collect page number rects — never redact these
        pn_rects = []
        for d in page.detections:
            if d.label == Label.PAGE_NUMBER:
                b = d.bbox.to_pdf(sx, sy)
                r = fitz.Rect(b.x1, b.y1, b.x2, b.y2)
                tight = _tighten_to_text(fitz_page, r, skip=False)
                pn_rects.append(tight if tight is not None else r)

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
            fitz_page.add_redact_annot(rect, fill=fill)

        # Outside-opinion whiteout: treat the group as one mega-opinion
        # from first caption to last key
        for rect in _outside_opinion_rects(
            page,
            fitz_page.rect.width,
            first_cap,
            last_key,
            is_first=True,
            is_last=True,
        ):
            add_safe(rect, (1, 1, 1))

        # Headnote blackout for each opinion in the group
        cap_key = first_cap.sort_key(mid)
        key_key = last_key.sort_key(mid)
        for idx in group:
            cap, key = opinions[idx]
            opinion_dets = []
            for d in page.detections:
                sk = d.sort_key(mid)
                if cap.sort_key(mid) <= sk <= key.sort_key(mid):
                    opinion_dets.append(d)
            opinion_dets.sort(key=lambda d: d.sort_key(mid))

            end_marker = _find_redaction_end(
                opinion_dets, cap, key, mid, reporter=document.reporter
            )
            if end_marker is not None:
                start = _find_redaction_start(opinion_dets, cap, mid)
                headnote_rects = _redaction_rects(
                    cap,
                    end_marker,
                    pages_by_index,
                    start_marker=start if start is not cap else None,
                )
            else:
                headnote_rects = _headnote_fallback_rects(opinion_dets, cap, pages_by_index, mid)
            if headnote_rects:
                header_bottom, footer_top = _margin_bounds(page)
                for rect_page_idx, rect in headnote_rects:
                    if rect_page_idx == src_page_idx:
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
                            rect = fitz.Rect(
                                rect.x0,
                                max(rect.y0, header_bottom),
                                rect.x1,
                                min(rect.y1, footer_top),
                            )
                        if rect.y0 < rect.y1 and rect.x0 < rect.x1:
                            add_safe(rect, (0, 0, 0))

        # Per-detection redactions (within the mega-span)
        # STATE_ABBREVIATION lives above the first caption so treat it like
        # a margin label — always redact regardless of sort-key position.
        _always_redact = _MARGIN_LABELS | {Label.STATE_ABBREVIATION}
        redact_labels = _REDACT_WHITE | _REDACT_BLACK
        for d in page.detections:
            if d.label not in redact_labels:
                continue
            if _check_excluded(d, excluded):
                continue
            if d.confidence < LABEL_CONFIDENCE.get(d.label, CONFIDENCE_THRESHOLD):
                continue
            if d.label not in _always_redact:
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

        # Key icon redactions (ratio-filtered only)
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
            if d.confidence < LABEL_CONFIDENCE.get(d.label, CONFIDENCE_THRESHOLD):
                continue
            b = d.bbox.to_pdf(sx, sy)
            rect = fitz.Rect(b.x1, b.y1, b.x2, b.y2)
            sk = d.sort_key(mid)
            if sk < cap_key or sk > key_key:
                add_safe(rect, (1, 1, 1))
            else:
                add_safe(rect, (0, 0, 0))

        # CASE_SEQUENCE: black redaction, clamped to 40-60px, never overlapping captions
        _caption_rects = []
        for d in page.detections:
            if d.label == Label.CASE_CAPTION:
                b = d.bbox.to_pdf(sx, sy)
                _caption_rects.append(fitz.Rect(b.x1, b.y1, b.x2, b.y2))
        _CS_INSET = 3.0
        _CS_MIN_PX, _CS_MAX_PX = 40, 60
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
            from blackletter.models import BBox as _BBox

            b = _BBox(bx0, by0, bx1, by1).to_pdf(sx, sy)
            rect = fitz.Rect(
                b.x1 + _CS_INSET,
                b.y1 + _CS_INSET,
                b.x2 - _CS_INSET,
                b.y2 - _CS_INSET,
            )
            for cap_r in _caption_rects:
                if rect.intersects(cap_r):
                    if rect.y0 < cap_r.y0:
                        rect = fitz.Rect(rect.x0, rect.y0, rect.x1, cap_r.y0)
                    else:
                        rect = fitz.Rect(0, 0, 0, 0)
            if rect.is_empty or rect.y0 >= rect.y1 or rect.x0 >= rect.x1:
                continue
            add_safe(rect, (0, 0, 0))

        fitz_page.apply_redactions()
        recompress_images(out_pdf)
        out_pdf.save(str(out_path), garbage=4, deflate=True)
        out_pdf.close()

        # Remove the individual files
        for idx in group:
            p = masked_paths[idx]
            if p.exists() and p != out_path:
                p.unlink()

        final_paths.append(out_path)

    src_pdf.close()
    return final_paths


def _apply_margin_rects(pdf_path: Path, margin_rects_path: Path) -> None:
    """Apply stored margin_rects.json as white redactions to the PDF in-place."""
    import json as _json

    margin_data = _json.loads(margin_rects_path.read_text())
    doc = fitz.open(str(pdf_path))
    _sample_imgs = doc[0].get_images(full=True) if doc.page_count else []
    is_bitonal = bool(_sample_imgs and _sample_imgs[0][4] == 1)
    for entry in margin_data:
        pi = entry["page_index"]
        if pi >= doc.page_count:
            continue
        page = doc[pi]
        white = (1, 1, 1)
        page_margin_rects = []
        for r in entry.get("rects", []):
            rect = fitz.Rect(r["x0"], r["y0"], r["x1"], r["y1"])
            page.add_redact_annot(rect, fill=white)
            page_margin_rects.append((rect, white))
        if is_bitonal and page_margin_rects:
            _redact_bitonal_image(page, doc, page_margin_rects)
            page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
        else:
            page.apply_redactions()
    doc.save(str(pdf_path), incremental=True, encryption=fitz.PDF_ENCRYPT_KEEP)
    doc.close()


def _redact_bitonal_image(fitz_page, fitz_doc, redaction_rects):
    """Apply redaction rectangles directly to bitonal image pixels.

    Replaces the actual pixel data so content is truly removed (real redaction),
    then re-embeds as a 1-bit FlateDecode image.
    """
    import io
    import zlib
    import numpy as np
    from PIL import Image, ImageDraw

    imgs = fitz_page.get_images(full=True)
    if not imgs:
        return

    xref = imgs[0][0]
    img_info = fitz_doc.extract_image(xref)
    pil_img = Image.open(io.BytesIO(img_info["image"]))

    gray = pil_img.convert("L")
    draw = ImageDraw.Draw(gray)

    # Scale: image pixels / PDF points
    sx = gray.width / fitz_page.rect.width
    sy = gray.height / fitz_page.rect.height

    for rect, fill in redaction_rects:
        x0 = int(rect.x0 * sx)
        y0 = int(rect.y0 * sy)
        x1 = int(rect.x1 * sx)
        y1 = int(rect.y1 * sy)
        color = 0 if fill[0] == 0 else 255
        draw.rectangle([x0, y0, x1, y1], fill=color)

    # Pack to 1-bit and compress with Flate
    arr = np.array(gray)
    bits = (arr > 128).astype(np.uint8)
    packed = np.packbits(bits, axis=1)
    compressed = zlib.compress(packed.tobytes())

    fitz_doc.update_stream(xref, compressed, compress=False)
    fitz_doc.xref_set_key(xref, "Filter", "/FlateDecode")
    fitz_doc.xref_set_key(xref, "DecodeParms", "null")
    fitz_doc.xref_set_key(xref, "Width", str(gray.width))
    fitz_doc.xref_set_key(xref, "Height", str(gray.height))
    fitz_doc.xref_set_key(xref, "BitsPerComponent", "1")
    fitz_doc.xref_set_key(xref, "ColorSpace", "/DeviceGray")


def compute_redaction_rects(
    document,
    opinions: list[tuple],
    excluded: set[tuple[int, int, int, int]] | None = None,
    approved: set[tuple[int, int, int, int]] | None = None,
    skip_doctr: bool = False,
) -> list[dict]:
    """Compute redaction rects per page, with text tightening.

    Opens the PDF to tighten rects to actual text boundaries — matches
    what _build_full_redacted would produce. approved detections bypass
    the confidence threshold. Returns list of
    {page_index, rects: [{x0, y0, x1, y1, fill: "black"|"white", type: str}]}.
    """
    pages_by_index = {p.index: p for p in document.pages}
    mid = document.pages[0].midpoint

    src_pdf = fitz.open(str(document.pdf_path))

    # Pre-compute headnote rects
    all_headnote_rects = []
    for caption, key in opinions:
        opinion_dets = []
        for p in document.pages:
            for d in p.detections:
                sk = d.sort_key(mid)
                if caption.sort_key(mid) <= sk <= key.sort_key(mid):
                    opinion_dets.append(d)
        opinion_dets.sort(key=lambda d: d.sort_key(mid))
        end_marker = _find_redaction_end(
            opinion_dets, caption, key, mid, reporter=document.reporter
        )
        if end_marker is not None:
            start = _find_redaction_start(opinion_dets, caption, mid)
            all_headnote_rects.extend(
                _redaction_rects(
                    caption,
                    end_marker,
                    pages_by_index,
                    start_marker=start if start is not caption else None,
                )
            )
        else:
            all_headnote_rects.extend(
                _headnote_fallback_rects(opinion_dets, caption, pages_by_index, mid)
            )

    # Refine headnote rects to line-level with docTR
    if all_headnote_rects and not skip_doctr:
        all_headnote_rects = refine_headnote_rects(src_pdf, all_headnote_rects, pages_by_index)

    result = {}

    def _add_px(page_idx, x0, y0, x1, y1, fill, rtype):
        """Add rect in image pixel coordinates."""
        if x0 >= x1 or y0 >= y1:
            return
        if page_idx not in result:
            result[page_idx] = []
        result[page_idx].append(
            {
                "x0": round(x0, 1),
                "y0": round(y0, 1),
                "x1": round(x1, 1),
                "y1": round(y1, 1),
                "fill": fill,
                "type": rtype,
            }
        )

    for page in document.pages:
        src_idx = page.index
        fitz_page = src_pdf[src_idx]
        sx, sy = page.scale_x, page.scale_y

        # Headnote zones — compute in image pixel coords
        hb, ft = _margin_bounds(page)

        # Use right-column HEADNOTE detections to establish the inner column
        # boundary: the leftmost x0 of any right-column HEADNOTE is where the
        # right column starts. The left column ends 10px before that.
        mid_px = page.midpoint
        # Right-column HEADNOTEs: center in right half, left edge at least 75% across
        right_hn = [
            d
            for d in page.detections
            if d.label == Label.HEADNOTE and d.bbox.center_x > mid_px and d.bbox.x1 >= mid_px * 0.75
        ]
        right_col_inner_pdf = min(d.bbox.x1 for d in right_hn) * sx if right_hn else None

        for rect_page_idx, rect in all_headnote_rects:
            if rect_page_idx == src_idx:
                # Determine column from the raw rect center BEFORE tightening
                mid_pdf = mid_px * sx
                raw_center_x = (rect.x0 + rect.x1) / 2
                is_left_col = raw_center_x < mid_pdf
                # Tighten to text in PDF points first
                tight = _tighten_to_text(fitz_page, rect)
                if tight is not None:
                    rect = tight
                text_bot = _text_bottom(fitz_page, rect)
                text_left, text_right = _text_x_bounds(fitz_page, rect)
                if right_col_inner_pdf is not None:
                    if is_left_col:
                        new_x0 = text_left
                        new_x1 = min(text_right, right_col_inner_pdf)
                    else:
                        new_x0 = max(text_left, right_col_inner_pdf)
                        new_x1 = text_right
                else:
                    # No right-column headnotes — keep original column x bounds
                    new_x0 = rect.x0
                    new_x1 = rect.x1
                rect = fitz.Rect(
                    new_x0,
                    max(rect.y0, hb),
                    new_x1,
                    min(rect.y1, ft, text_bot),
                )
                if rect.is_empty or rect.x0 >= rect.x1 or rect.y0 >= rect.y1:
                    continue
                # Convert to image pixels
                rx0 = rect.x0 / sx
                ry0 = rect.y0 / sy
                rx1 = rect.x1 / sx
                ry1 = rect.y1 / sy
                _add_px(src_idx, rx0, ry0, rx1, ry1, "black", "headnote")

        # Per-detection redactions — output in image pixel coords
        for d in page.detections:
            if d.label not in (_REDACT_WHITE | _REDACT_BLACK):
                continue
            if _check_excluded(d, excluded):
                continue
            _is_approved = approved and _check_excluded(d, approved)
            if not _is_approved and d.confidence < LABEL_CONFIDENCE.get(
                d.label, CONFIDENCE_THRESHOLD
            ):
                continue
            _skip_tighten = d.label in (Label.HEADNOTE_BRACKET, Label.STATE_ABBREVIATION)
            if _skip_tighten:
                # Use raw YOLO bbox (image pixels)
                fill = "black" if d.label in _REDACT_BLACK else "white"
                _add_px(src_idx, d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2, fill, d.label.name)
            else:
                # Tighten in PDF coords, convert back to pixels
                b = d.bbox.to_pdf(sx, sy)
                rect = fitz.Rect(b.x1, b.y1, b.x2, b.y2)
                tight = _tighten_to_text(fitz_page, rect, skip=False)
                if tight is not None:
                    rect = tight
                fill = "black" if d.label in _REDACT_BLACK else "white"
                _add_px(
                    src_idx,
                    rect.x0 / sx,
                    rect.y0 / sy,
                    rect.x1 / sx,
                    rect.y1 / sy,
                    fill,
                    d.label.name,
                )

        # CASE_SEQUENCE
        caption_rects = []
        for d in page.detections:
            if d.label == Label.CASE_CAPTION:
                b = d.bbox.to_pdf(sx, sy)
                caption_rects.append(fitz.Rect(b.x1, b.y1, b.x2, b.y2))

        _CS_INSET = 3.0
        _CS_MIN_PX = 40  # min width/height in image pixels
        _CS_MAX_PX = 60  # max width/height in image pixels
        for d in page.detections:
            if d.label != Label.CASE_SEQUENCE:
                continue
            if _check_excluded(d, excluded):
                continue
            # Clamp raw YOLO bbox to min/max size in pixels, then convert
            bx0, by0 = d.bbox.x1, d.bbox.y1
            bx1, by1 = d.bbox.x2, d.bbox.y2
            cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
            pw = max(_CS_MIN_PX, min(bx1 - bx0, _CS_MAX_PX))
            ph = max(_CS_MIN_PX, min(by1 - by0, _CS_MAX_PX))
            bx0, bx1 = cx - pw / 2, cx + pw / 2
            by0, by1 = cy - ph / 2, cy + ph / 2
            b = BBox(bx0, by0, bx1, by1).to_pdf(sx, sy)
            rect = fitz.Rect(b.x1, b.y1, b.x2, b.y2)
            # Small inset
            rect = fitz.Rect(
                rect.x0 + _CS_INSET,
                rect.y0 + _CS_INSET,
                rect.x1 - _CS_INSET,
                rect.y1 - _CS_INSET,
            )
            if rect.is_empty or rect.y0 >= rect.y1 or rect.x0 >= rect.x1:
                continue
            _add_px(
                src_idx,
                rect.x0 / sx,
                rect.y0 / sy,
                rect.x1 / sx,
                rect.y1 / sy,
                "black",
                "CASE_SEQUENCE",
            )

        # Key icons — use raw YOLO bbox (image pixels)
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
            if d.confidence < LABEL_CONFIDENCE.get(d.label, CONFIDENCE_THRESHOLD):
                continue
            _add_px(src_idx, d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2, "black", "KEY_ICON")

    src_pdf.close()
    return [{"page_index": pi, "rects": rects} for pi, rects in sorted(result.items())]


def _split_from_redacted(
    redacted_pdf_path: Path,
    document,
    opinions: list[tuple],
    output_dir: Path,
    first_page: int = 1,
) -> list[Path]:
    """Split individual opinion PDFs from an already-redacted full PDF.

    Extracts pages for each opinion and applies outside-opinion whiteout.
    This preserves manual redaction adjustments from redaction_rects.json.

    Each opinion spans from its caption page to the page before the next
    opinion starts (or end of document for the last opinion).

    Filenames use actual page numbers from OCR.  When a page shows a range
    (e.g. "677-685"): opinion starting on that page uses the end number,
    opinion ending on that page uses the first number.
    """
    pages_by_index = {p.index: p for p in document.pages}
    redacted = fitz.open(str(redacted_pdf_path))

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    total = len(opinions)

    # Each opinion runs from its caption page to its key page.
    page_ranges: list[tuple[int, int]] = []
    for idx, (caption, key) in enumerate(opinions):
        page_ranges.append((caption.page_index, key.page_index))

    # Pre-compute base names using actual page numbers
    from collections import Counter as _Counter

    _reporter = document.reporter or ""
    _volume = document.volume or ""
    _bases: list[str] = []
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

        _bases.append(f"{_reporter}.{_volume}.{first_num:04d}-{last_num:04d}")

    _name_counts = _Counter(_bases)
    _name_seq: dict[str, int] = {}

    for idx, (caption, key) in enumerate(opinions):
        start_idx, end_idx = page_ranges[idx]
        out_pdf = fitz.open()
        out_pdf.insert_pdf(redacted, from_page=start_idx, to_page=end_idx)

        # Apply outside-opinion whiteout on first/last pages
        for local_idx, src_idx in enumerate(range(start_idx, end_idx + 1)):
            if src_idx not in pages_by_index:
                continue
            page = pages_by_index[src_idx]
            fitz_page = out_pdf[local_idx]
            is_first = src_idx == start_idx
            is_last = src_idx == end_idx

            if is_first or is_last:
                for rect in _outside_opinion_rects(
                    page, fitz_page.rect.width, caption, key, is_first, is_last
                ):
                    fitz_page.add_redact_annot(rect, fill=(1, 1, 1))

                # Bitonal-safe redaction
                _sample = fitz_page.get_images(full=True) if out_pdf.page_count else []
                _is_bit = bool(_sample and _sample[0][4] == 1)
                if _is_bit:
                    white_rects = [
                        (rect, (1, 1, 1))
                        for rect in _outside_opinion_rects(
                            page, fitz_page.rect.width, caption, key, is_first, is_last
                        )
                    ]
                    if white_rects:
                        _redact_bitonal_image(fitz_page, out_pdf, white_rects)
                        fitz_page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
                    else:
                        fitz_page.apply_redactions()
                else:
                    fitz_page.apply_redactions()

        # Build filename — use -1/-2/-3 suffix when multiple opinions share the same page range
        base = _bases[idx]
        if _name_counts[base] > 1:
            _name_seq[base] = _name_seq.get(base, 0) + 1
            name = f"{base}-{_name_seq[base]}.pdf"
        else:
            name = f"{base}.pdf"
        out_path = output_dir / name

        out_pdf.save(str(out_path), garbage=4, deflate=True)
        out_pdf.close()
        paths.append(out_path)

        done = idx + 1
        if done % 20 == 0 or done == total:
            print(f"    Split {done}/{total}", flush=True)

    redacted.close()
    print(f"  Wrote {len(paths)} redacted PDFs", flush=True)
    return paths


def _build_redacted_from_rects(
    document,
    rects_path: Path,
    output_path: Path,
) -> Path:
    """Build a redacted PDF using precomputed redaction_rects.json.

    Uses the exact rects from the review UI (including manual adjustments)
    instead of recomputing from detections.
    """
    import json as _json

    rects_data = _json.loads(rects_path.read_text())
    rects_by_page = {}
    for entry in rects_data:
        rects_by_page[entry["page_index"]] = entry["rects"]

    src_pdf = fitz.open(str(document.pdf_path))
    out_pdf = fitz.open()
    out_pdf.insert_pdf(src_pdf)

    # Detect bitonal
    _sample_imgs = out_pdf[0].get_images(full=True) if out_pdf.page_count else []
    is_bitonal = bool(_sample_imgs and _sample_imgs[0][4] == 1)

    total_pages = len(document.pages)
    pages_by_idx = {p.index: p for p in document.pages}
    for page in document.pages:
        src_idx = page.index
        fitz_page = out_pdf[src_idx]
        page_rects = rects_by_page.get(src_idx, [])

        # Convert image pixel coords → PDF points
        p = pages_by_idx.get(src_idx)
        if p and p.img_width > 1:
            to_pdf_x = p.pdf_width / p.img_width
            to_pdf_y = p.pdf_height / p.img_height
        else:
            to_pdf_x = to_pdf_y = 1.0

        page_redact_rects = []
        for r in page_rects:
            rect = fitz.Rect(
                r["x0"] * to_pdf_x, r["y0"] * to_pdf_y, r["x1"] * to_pdf_x, r["y1"] * to_pdf_y
            )
            if rect.is_empty or rect.y0 >= rect.y1 or rect.x0 >= rect.x1:
                continue
            fill_tuple = (0, 0, 0) if r.get("fill") == "black" else (1, 1, 1)
            page_redact_rects.append((fitz.Rect(rect), fill_tuple))
            fitz_page.add_redact_annot(rect, fill=fill_tuple)

        if is_bitonal and page_redact_rects:
            _redact_bitonal_image(fitz_page, out_pdf, page_redact_rects)
            fitz_page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
        else:
            fitz_page.apply_redactions()

        pg = page.index + 1
        if pg % 20 == 0 or pg == total_pages:
            print(f"    Redacted {pg}/{total_pages} pages", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_pdf.save(str(output_path), garbage=4, deflate=True)
    out_pdf.close()
    src_pdf.close()

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved {output_path.name} ({size_mb:.1f} MB)")
    return output_path


def _build_full_redacted(
    document,
    opinions: list[tuple],
    output_path: Path,
    excluded: set[tuple[int, int, int, int]] | None = None,
) -> Path:
    """Build a single fully-redacted copy of the entire document.

    Applies headnote blackout, per-detection redactions, and key icon
    blackout across every page.  No masking / outside-opinion whiteout.
    """
    pages_by_index = {p.index: p for p in document.pages}
    mid = document.pages[0].midpoint

    src_pdf = fitz.open(str(document.pdf_path))
    out_pdf = fitz.open()
    out_pdf.insert_pdf(src_pdf)

    # Detect bitonal — need special redaction path to avoid CCITT corruption
    _sample_imgs = out_pdf[0].get_images(full=True) if out_pdf.page_count else []
    is_bitonal = bool(_sample_imgs and _sample_imgs[0][4] == 1)

    # Pre-compute headnote rects for every opinion
    all_headnote_rects: list[tuple[int, fitz.Rect]] = []
    for caption, key in opinions:
        page = pages_by_index[caption.page_index]
        opinion_dets = []
        for p in document.pages:
            for d in p.detections:
                sk = d.sort_key(mid)
                if caption.sort_key(mid) <= sk <= key.sort_key(mid):
                    opinion_dets.append(d)
        opinion_dets.sort(key=lambda d: d.sort_key(mid))
        end_marker = _find_redaction_end(
            opinion_dets, caption, key, mid, reporter=document.reporter
        )
        if end_marker is not None:
            start = _find_redaction_start(opinion_dets, caption, mid)
            all_headnote_rects.extend(
                _redaction_rects(
                    caption,
                    end_marker,
                    pages_by_index,
                    start_marker=start if start is not caption else None,
                )
            )
        else:
            all_headnote_rects.extend(
                _headnote_fallback_rects(opinion_dets, caption, pages_by_index, mid)
            )

    # Refine block rects to line-level with docTR
    all_headnote_rects = refine_headnote_rects(src_pdf, all_headnote_rects, pages_by_index)

    # Process each page
    total_pages = len(document.pages)
    for page in document.pages:
        src_idx = page.index
        fitz_page = out_pdf[src_idx]
        sx, sy = page.scale_x, page.scale_y

        # Collect page number rects — never redact these
        pn_rects = []
        for d in page.detections:
            if d.label == Label.PAGE_NUMBER:
                b = d.bbox.to_pdf(sx, sy)
                r = fitz.Rect(b.x1, b.y1, b.x2, b.y2)
                tight = _tighten_to_text(fitz_page, r, skip=False)
                pn_rects.append(tight if tight is not None else r)

        # Collect caption rects — CASE_SEQUENCE must not overlap these
        caption_rects = []
        for d in page.detections:
            if d.label == Label.CASE_CAPTION:
                b = d.bbox.to_pdf(sx, sy)
                caption_rects.append(fitz.Rect(b.x1, b.y1, b.x2, b.y2))

        page_redact_rects = []

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
            page_redact_rects.append((fitz.Rect(rect), fill))
            fitz_page.add_redact_annot(rect, fill=fill)

        # Headnote blackout
        header_bottom, footer_top = _margin_bounds(page)
        for rect_page_idx, rect in all_headnote_rects:
            if rect_page_idx == src_idx:
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
                if rect.y0 < rect.y1 and rect.x0 < rect.x1:
                    add_safe(rect, (0, 0, 0))

        # Per-detection redactions
        for d in page.detections:
            if d.label not in (_REDACT_WHITE | _REDACT_BLACK):
                continue
            if _check_excluded(d, excluded):
                continue
            if d.confidence < LABEL_CONFIDENCE.get(d.label, CONFIDENCE_THRESHOLD):
                continue
            b = d.bbox.to_pdf(sx, sy)
            rect = fitz.Rect(b.x1, b.y1, b.x2, b.y2)
            _skip_tighten = d.label in (Label.HEADNOTE_BRACKET, Label.STATE_ABBREVIATION)
            if not _skip_tighten:
                tight = _tighten_to_text(fitz_page, rect, skip=False)
                if tight is not None:
                    rect = tight
            fill = (0, 0, 0) if d.label in _REDACT_BLACK else (1, 1, 1)
            add_safe(rect, fill)

        # CASE_SEQUENCE: redact black, but never overlap CASE_CAPTION
        _CS_INSET = 3.0
        _CS_MIN_PX = 40
        _CS_MAX_PX = 60
        for d in page.detections:
            if d.label != Label.CASE_SEQUENCE:
                continue
            if _check_excluded(d, excluded):
                continue
            # Clamp to min/max pixels, then convert to PDF points
            bx0, by0 = d.bbox.x1, d.bbox.y1
            bx1, by1 = d.bbox.x2, d.bbox.y2
            cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
            pw = max(_CS_MIN_PX, min(bx1 - bx0, _CS_MAX_PX))
            ph = max(_CS_MIN_PX, min(by1 - by0, _CS_MAX_PX))
            bx0, bx1 = cx - pw / 2, cx + pw / 2
            by0, by1 = cy - ph / 2, cy + ph / 2
            from blackletter.models import BBox

            b = BBox(bx0, by0, bx1, by1).to_pdf(sx, sy)
            rect = fitz.Rect(b.x1, b.y1, b.x2, b.y2)
            rect = fitz.Rect(
                rect.x0 + _CS_INSET,
                rect.y0 + _CS_INSET,
                rect.x1 - _CS_INSET,
                rect.y1 - _CS_INSET,
            )
            # Clip away any overlap with caption rects
            for cap_r in caption_rects:
                if rect.intersects(cap_r):
                    if rect.y0 < cap_r.y0:
                        # Sequence box hangs into the caption from above — trim its bottom
                        rect = fitz.Rect(rect.x0, rect.y0, rect.x1, cap_r.y0)
                    else:
                        # Sequence box starts inside the caption — skip entirely
                        rect = fitz.Rect(0, 0, 0, 0)
            if rect.is_empty or rect.y0 >= rect.y1 or rect.x0 >= rect.x1:
                continue
            add_safe(rect, (0, 0, 0))

        # Key icon redactions (always black, ratio-filtered only)
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
            if d.confidence < LABEL_CONFIDENCE.get(d.label, CONFIDENCE_THRESHOLD):
                continue
            b = d.bbox.to_pdf(sx, sy)
            rect = fitz.Rect(b.x1, b.y1, b.x2, b.y2)
            add_safe(rect, (0, 0, 0))

        if is_bitonal and page_redact_rects:
            _redact_bitonal_image(fitz_page, out_pdf, page_redact_rects)
            fitz_page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
        else:
            fitz_page.apply_redactions()
        pg = page.index + 1
        if pg % 20 == 0 or pg == total_pages:
            print(f"    Redacted {pg}/{total_pages} pages", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_pdf.save(str(output_path), garbage=4, deflate=True)
    out_pdf.close()
    src_pdf.close()
    size_mb = output_path.stat().st_size / (1024 * 1024)
    if size_mb > 50:
        # Skip recompression if already bitonal (1-bit images can't be compressed further)
        reopen = fitz.open(str(output_path))
        sample_imgs = reopen[0].get_images(full=True) if reopen.page_count else []
        is_bitonal = bool(sample_imgs and sample_imgs[0][4] == 1)
        if is_bitonal:
            reopen.close()
            print("  Skipping recompression (bitonal)")
        else:
            print(f"  Recompressing images ({size_mb:.1f} MB > 50 MB limit)...")
            recompress_images(reopen)
            data = reopen.tobytes(garbage=4, deflate=True)
            reopen.close()
            output_path.write_bytes(data)
            size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved {output_path.name} ({size_mb:.1f} MB)")
    return output_path


def _extract_images(document, output_dir: Path) -> int:
    """Extract detected IMAGE regions from the PDF as PNG files.

    Returns the number of images extracted.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    src_pdf = fitz.open(str(document.pdf_path))
    # 300 DPI: PDF points are 72 per inch, so scale factor = 300/72
    mat = fitz.Matrix(300 / 72, 300 / 72)
    count = 0

    for page in document.pages:
        sx, sy = page.scale_x, page.scale_y
        fitz_page = src_pdf[page.index]

        # Collect IMAGE detections above confidence threshold
        images = [
            d
            for d in page.detections
            if d.label == Label.IMAGE
            and d.confidence >= LABEL_CONFIDENCE.get(Label.IMAGE, CONFIDENCE_THRESHOLD)
        ]
        if not images:
            continue

        # Sort by reading order (top-to-bottom, left-to-right)
        images.sort(key=lambda d: d.sort_key(page.midpoint))

        for seq, det in enumerate(images, start=1):
            b = det.bbox.to_pdf(sx, sy)
            clip = fitz.Rect(b.x1, b.y1, b.x2, b.y2)
            pix = fitz_page.get_pixmap(matrix=mat, clip=clip)

            # Use detected page number, fall back to index + first_page
            page_num = page.page_number
            if page_num is None:
                page_num = page.index + document.first_page

            fname = f"{page_num:04d}-{seq:03d}.png"
            pix.save(str(output_dir / fname))
            count += 1

    src_pdf.close()
    return count


def _infer_from_filename(pdf_path: Path) -> dict:
    """Try to infer reporter, volume, first_page, last_page from filename.

    Expects patterns like:
        sct.143.1.888.pdf
        f3d.100.1.1200.pdf
        reporter.volume.first.last.pdf
    """
    stem = pdf_path.stem
    # Strip known suffixes like .redacted, .ocr, _dupes, _bitonal, etc.
    import re as _re

    stem = _re.sub(r"[_-](redacted|ocr|compressed|dupes|bitonal|copy|fixed)\b.*", "", stem)
    for suffix in (".redacted", ".ocr", ".compressed"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]

    parts = stem.split(".")
    if len(parts) >= 4:
        try:
            first = int(parts[-2])
            last = int(parts[-1])
            volume = parts[-3]
            reporter = ".".join(parts[:-3])
            if reporter and volume and first >= 1 and last >= first:
                return {
                    "reporter": reporter,
                    "volume": volume,
                    "first_page": first,
                    "last_page": last,
                }
        except ValueError:
            pass
    return {}


def cmd_process(args: argparse.Namespace) -> None:
    """Scan, verify, and split into opinion PDFs in one pass."""

    # Infer missing args from filename
    inferred = _infer_from_filename(args.pdf)
    if inferred:
        if not args.reporter and "reporter" in inferred:
            args.reporter = inferred["reporter"]
            print(f"Inferred reporter: {args.reporter}")
        if not args.volume and "volume" in inferred:
            args.volume = inferred["volume"]
            print(f"Inferred volume: {args.volume}")
        if args.first_page == 1 and "first_page" in inferred:
            args.first_page = inferred["first_page"]
            print(f"Inferred first page: {args.first_page}")

    # Require reporter and volume
    if not args.reporter:
        print(
            "Error: --reporter is required (or use a filename like reporter.volume.first.last.pdf)"
        )
        sys.exit(1)
    if not args.volume:
        print("Error: --volume is required (or use a filename like reporter.volume.first.last.pdf)")
        sys.exit(1)

    import time as _time

    _t_total = _time.time()

    model = YOLO(str(args.model))

    base_dir = _build_output_dir(args)
    base_dir.mkdir(parents=True, exist_ok=True)

    # ── Bitonal conversion ──
    if getattr(args, "bitonal", False):
        # Skip if already bitonal (BitsPerComponent == 1)
        doc_check = fitz.open(str(args.pdf))
        sample_imgs = doc_check[0].get_images(full=True) if doc_check.page_count else []
        is_bitonal = bool(sample_imgs and sample_imgs[0][4] == 1)
        doc_check.close()
        if is_bitonal:
            print("Source PDF is already bitonal, skipping conversion.", flush=True)
        else:
            from blackletter.ocr import bitonal_convert

            _t0 = _time.time()
            print("\n── Bitonal conversion ──", flush=True)
            bitonal_path = base_dir / f"{args.pdf.stem}_bitonal.pdf"
            bitonal_convert(args.pdf, bitonal_path)
            args.pdf = bitonal_path
            print(f"  Bitonal done ({_time.time() - _t0:.0f}s)", flush=True)

    unredacted_dir = base_dir / "unredacted"
    redacted_dir = base_dir / "redacted"
    masked_dir = base_dir / "masked"

    # ── Scan ──
    shrink = not getattr(args, "no_shrink", False)

    # Build output name: reporter.volume.firstpage.lastpage
    page_count = len(fitz.open(str(args.pdf)))
    # Prefer the last_page from filename parsing over computing from page count
    last_page = inferred.get("last_page") or (args.first_page + page_count - 1)
    parts = []
    if args.reporter:
        parts.append(args.reporter)
    if args.volume:
        parts.append(str(args.volume))
    parts.append(str(args.first_page))
    parts.append(str(last_page))
    scan_name = ".".join(parts)

    src_mb = args.pdf.stat().st_size / (1024 * 1024)
    _t0 = _time.time()
    print("\n── Scan (OCR + YOLO) ──", flush=True)
    print(f"Scanning {args.pdf.name} ({page_count} pages, {src_mb:.1f} MB)...", flush=True)
    device = "cpu" if getattr(args, "cpu", False) else None
    cb = getattr(args, "progress_callback", None)
    document = scan(
        args.pdf,
        model,
        first_page=args.first_page,
        output_dir=base_dir,
        shrink=shrink,
        optimize=getattr(args, "optimize", 1),
        output_name=scan_name,
        device=device,
        progress_callback=cb,
    )
    print(f"  Scan done ({_time.time() - _t0:.0f}s)", flush=True)
    if args.reporter:
        document.reporter = args.reporter
    if args.volume:
        document.volume = args.volume

    # ── Save detections for review UI ──
    import json as _json
    import time as _time

    detections_path = base_dir / "detections.json"
    detections_data = []
    for _page in document.pages:
        for _det in _page.detections:
            detections_data.append(
                {
                    "page_index": _det.page_index,
                    "label": _det.label.name,
                    "label_id": int(_det.label),
                    "confidence": round(_det.confidence, 3),
                    "bbox": [
                        round(_det.bbox.x1, 1),
                        round(_det.bbox.y1, 1),
                        round(_det.bbox.x2, 1),
                        round(_det.bbox.y2, 1),
                    ],
                    "page_number": _page.page_number,
                    "img_width": _page.img_width,
                    "img_height": _page.img_height,
                }
            )
    with open(detections_path, "w") as _f:
        _json.dump(detections_data, _f)
    print(f"  Saved {len(detections_data)} detections to {detections_path.name}")

    # Save page metadata (column bounds, midpoint) for re-pair
    pages_meta = {}
    for _page in document.pages:
        pages_meta[_page.index] = {
            "col_left_x1": _page.col_left_x1,
            "col_left_x2": _page.col_left_x2,
            "col_right_x1": _page.col_right_x1,
            "col_right_x2": _page.col_right_x2,
            "midpoint": _page.midpoint,
        }
    with open(base_dir / "pages_meta.json", "w") as _f:
        _json.dump(pages_meta, _f)

    # ── Auto-fill missing STATE_ABBREVIATION with large model ──
    _pages_by_idx = {p.index: p for p in document.pages}
    sa_pages = {
        d.page_index
        for p in document.pages
        for d in p.detections
        if d.label == Label.STATE_ABBREVIATION
    }
    all_pages = {p.index for p in document.pages}
    missing_sa = sorted(all_pages - sa_pages)
    # Auto-fill if flag is set, or if SA exists on enough pages to be a pattern
    _has_sa_flag = getattr(args, "has_state_abbrev", None)
    if _has_sa_flag is False:
        missing_sa = []  # explicitly disabled
    if missing_sa and (_has_sa_flag or (sa_pages and len(sa_pages) > len(all_pages) * 0.1)):
        print(
            f"\n  Auto-fill: {len(sa_pages)} pages have STATE_ABBREVIATION, "
            f"{len(missing_sa)} missing — scanning with large model...",
            flush=True,
        )
        _t0 = _time.time()
        large_model_path = Path(__file__).resolve().parent / "models" / "large.pt"
        if large_model_path.exists():
            from PIL import Image as _PILImage
            from blackletter.scanner import DPI

            large_model = YOLO(str(large_model_path))
            mat = fitz.Matrix(DPI / 72, DPI / 72)
            pdf_file = fitz.open(str(document.pdf_path))
            added = 0
            for pi in missing_sa:
                pix = pdf_file[pi].get_pixmap(matrix=mat)
                img = _PILImage.frombytes("RGB", (pix.width, pix.height), pix.samples)
                results = large_model([img], conf=0.50, verbose=False)
                for box in results[0].boxes:
                    cls = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    if cls == int(Label.STATE_ABBREVIATION) and conf >= 0.50:
                        bbox = box.xyxy[0].tolist()
                        det = Detection(
                            bbox=BBox.from_xyxy(bbox),
                            label=Label.STATE_ABBREVIATION,
                            confidence=float(box.conf[0].item()),
                            page_index=pi,
                        )
                        _pages_by_idx[pi].detections.append(det)
                        detections_data.append(
                            {
                                "page_index": pi,
                                "label": "STATE_ABBREVIATION",
                                "label_id": int(Label.STATE_ABBREVIATION),
                                "confidence": round(float(box.conf[0].item()), 3),
                                "bbox": [round(v, 1) for v in bbox],
                                "page_number": _pages_by_idx[pi].page_number,
                                "img_width": pix.width,
                                "img_height": pix.height,
                            }
                        )
                        added += 1
            pdf_file.close()
            if added:
                with open(detections_path, "w") as _f:
                    _json.dump(detections_data, _f)
            print(
                f"  Auto-fill: added {added} STATE_ABBREVIATION ({_time.time() - _t0:.0f}s)",
                flush=True,
            )

    # ── Opinion pairing ──
    excluded = getattr(args, "excluded", None)
    _t0 = _time.time()
    print("\nPairing opinions...", flush=True)
    opinions = _pair_opinions(document, excluded=excluded)
    print(f"  Found {len(opinions)} opinions ({_time.time() - _t0:.0f}s)", flush=True)

    pages_by_index = {p.index: p for p in document.pages}
    _src_pdf = fitz.open(str(document.pdf_path))
    opinions_data = []
    for caption, key in opinions:
        # Compute outside-opinion rects for each page in this opinion
        outside_rects = []
        for pi in range(caption.page_index, key.page_index + 1):
            page = pages_by_index[pi]
            is_first = pi == caption.page_index
            is_last = pi == key.page_index
            pw = _src_pdf[pi].rect.width
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
                "has_image": any(
                    d.label == Label.IMAGE
                    for pi2 in range(caption.page_index, key.page_index + 1)
                    for d in pages_by_index[pi2].detections
                ),
            }
        )
    _src_pdf.close()

    # ── Precompute redaction rects (with tightening) for review ──
    _t0 = _time.time()
    print("\nComputing redaction rects...", flush=True)
    redaction_rects = compute_redaction_rects(document, opinions, excluded=excluded)
    rects_path = base_dir / "redaction_rects.json"
    with open(rects_path, "w") as _f:
        _json.dump(redaction_rects, _f)
    total_rects = sum(len(r["rects"]) for r in redaction_rects)
    print(f"  Saved {total_rects} redaction rects ({_time.time() - _t0:.0f}s)", flush=True)

    # ── Precompute margin rects for review ──
    _t0 = _time.time()
    print("\nComputing margin rects...", flush=True)
    from blackletter.margins import compute_margin_rects

    margin_rects = compute_margin_rects(document.pdf_path)
    margins_path = base_dir / "margin_rects.json"
    with open(margins_path, "w") as _f:
        _json.dump(margin_rects, _f)
    print(f"  Saved margin rects ({_time.time() - _t0:.0f}s)", flush=True)

    # ── Check if detect-only mode ──
    if getattr(args, "detect_only", False):
        print(f"\n── Phase 1 complete: {_time.time() - _t_total:.0f}s ──", flush=True)
        if cb:
            cb(0, 0, "Ready for review")
        return

    # ── Clean margins ──
    from blackletter.margins import clean_margins

    if cb:
        cb(0, 0, "Cleaning margins...")
    _t0 = _time.time()
    print("\nCleaning margins...", flush=True)
    clean_margins(document.pdf_path)
    print(f"  Margins cleaned ({_time.time() - _t0:.0f}s)", flush=True)

    # ── Extract images ──
    _t0 = _time.time()
    images_dir = base_dir / "images"
    n_images = _extract_images(document, images_dir)
    if n_images:
        print(f"\nExtracted {n_images} images ({_time.time() - _t0:.0f}s)", flush=True)

    if cb:
        cb(0, 0, f"Found {len(opinions)} opinions — building redacted PDF...")

    # ── Full redacted (single PDF) ──
    prefix = ""
    if args.reporter:
        prefix = f"{args.reporter}."
    if args.volume:
        prefix += f"{args.volume}."
    full_redacted_name = f"{scan_name}.redacted.pdf"
    full_redacted_path = base_dir / full_redacted_name
    _t0 = _time.time()
    print("\nBuilding full redacted PDF...", flush=True)
    _build_full_redacted(document, opinions, full_redacted_path, excluded=excluded)
    print(f"  Full redacted done ({_time.time() - _t0:.0f}s)", flush=True)

    # ── Unredacted ──
    if args.unredacted:
        if cb:
            cb(0, 0, "Splitting unredacted opinions...")
        _t0 = _time.time()
        print(f"\nSplitting unredacted into {unredacted_dir}...", flush=True)
        unredacted_paths = split_opinions(
            document.pdf_path,
            document,
            unredacted_dir,
            first_page=args.first_page,
            redact_mode="unredacted",
            extract_footnotes=args.footnotes,
            excluded=excluded,
        )
        print(
            f"  Wrote {len(unredacted_paths)} unredacted PDFs ({_time.time() - _t0:.0f}s)",
            flush=True,
        )

    # ── Redacted ──
    if cb:
        cb(0, 0, "Splitting redacted opinions...")
    _t0 = _time.time()
    print(f"\nSplitting redacted into {redacted_dir}...", flush=True)
    redacted_paths = split_opinions(
        document.pdf_path,
        document,
        redacted_dir,
        first_page=args.first_page,
        redact_mode="redacted",
        extract_footnotes=args.footnotes,
        excluded=excluded,
    )
    print(f"  Wrote {len(redacted_paths)} redacted PDFs ({_time.time() - _t0:.0f}s)", flush=True)

    # ── Masked (for LLM) ──
    if cb:
        cb(0, 0, "Splitting masked opinions...")
    _t0 = _time.time()
    print(f"\nSplitting masked into {masked_dir}...", flush=True)
    masked_paths = split_opinions(
        document.pdf_path,
        document,
        masked_dir,
        first_page=args.first_page,
        redact_mode="masked",
        excluded=excluded,
    )
    # Consolidate same-page opinions
    final_masked = _build_masked_opinions(
        masked_paths,
        opinions,
        document,
        masked_dir,
    )
    print(
        f"  Wrote {len(final_masked)} masked PDFs ({len(masked_paths)} opinions consolidated) ({_time.time() - _t0:.0f}s)",
        flush=True,
    )
    print(f"\n── Total processing time: {_time.time() - _t_total:.0f}s ──", flush=True)
    if cb:
        cb(0, 0, "Done")


def reprocess_section(
    ocr_pdf_path: str | Path,
    output_dir: str | Path,
    page_start: int,
    page_end: int,
    *,
    first_page: int = 1,
    reporter: str | None = None,
    volume: str | None = None,
    model: str | Path | None = None,
    excluded: set[tuple[int, int, int, int]] | None = None,
    injected: list[dict] | None = None,
    redactions: list[dict] | None = None,
    progress_callback=None,
) -> dict:
    """Reprocess a section of an already-OCR'd PDF.

    Extracts pages page_start..page_end (logical page numbers), runs YOLO
    detection, pairs opinions (with exclusions), and splits into PDFs.

    Args:
        ocr_pdf_path: Path to the OCR'd (non-redacted) PDF.
        output_dir: Directory to write new opinion PDFs.
        page_start: First logical page number of the section.
        page_end: Last logical page number (inclusive).
        first_page: Logical page number of the first page in the full PDF.
        reporter: Reporter abbreviation for filenames.
        volume: Volume number for filenames.
        model: Path to YOLO model. Defaults to bundled model.
        excluded: Detection exclusions for _pair_opinions().
        injected: Manual detections to inject before pairing. Each dict has
            page_index (absolute), label_id, bbox [x1,y1,x2,y2], img_width,
            img_height. confidence defaults to 1.0.
        redactions: Redactions to burn into the section before YOLO. Each dict
            has page_number (1-based absolute), x, y, width, height, fill
            ('black'|'white'). Applied to the temp section extract only —
            the original OCR PDF is not modified.
        progress_callback: Optional callable(current, total, message).

    Returns:
        Dict with keys: opinions (list of {filename, first_page, last_page}),
        redacted_paths, unredacted_paths.
    """
    import tempfile

    ocr_pdf_path = Path(ocr_pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if model is None:
        model = DEFAULT_MODEL

    # Convert logical page numbers to 0-based PDF indices
    pdf_start = page_start - first_page
    pdf_end = page_end - first_page

    # Extract section
    src_pdf = fitz.open(str(ocr_pdf_path))
    section_path = Path(tempfile.mktemp(suffix=".pdf"))
    section_pdf = fitz.open()
    section_pdf.insert_pdf(src_pdf, from_page=pdf_start, to_page=pdf_end)
    section_pdf.save(str(section_path))
    section_pdf.close()
    src_pdf.close()

    print(f"Reprocessing section: pages {page_start}-{page_end} ({pdf_end - pdf_start + 1} pages)")

    # Apply drawn redactions to section extract before YOLO (non-destructive to OCR PDF)
    if redactions:
        sec = fitz.open(str(section_path))
        applied = 0
        for r in redactions:
            # Convert absolute page_number to local section index
            local_idx = r["page_number"] - first_page - pdf_start
            if 0 <= local_idx < sec.page_count:
                page = sec[local_idx]
                rect = fitz.Rect(r["x"], r["y"], r["x"] + r["width"], r["y"] + r["height"])
                fill_color = (1, 1, 1) if r.get("fill") == "white" else (0, 0, 0)
                page.add_redact_annot(rect, fill=fill_color)
                page.apply_redactions()
                applied += 1
        if applied:
            sec.save(str(section_path), garbage=4, deflate=True)
            print(f"  Applied {applied} redaction(s) to section extract")
        sec.close()

    # Scan section (no OCR needed — already has text layer)
    yolo_model = YOLO(str(model))
    cb = progress_callback
    document = scan(
        section_path,
        yolo_model,
        first_page=page_start,
        output_dir=output_dir,
        shrink=False,
        skip_ocr=True,
        progress_callback=cb,
    )
    if reporter:
        document.reporter = reporter
    if volume:
        document.volume = volume

    # Inject manual detections (remapped to section-local page indices)
    if injected:
        from blackletter.models import BBox, Detection, Label as _InjLabel

        injected_count = 0
        for det_dict in injected:
            abs_pi = det_dict.get("page_index", 0)
            local_pi = abs_pi - pdf_start
            if local_pi < 0 or local_pi > pdf_end - pdf_start:
                continue
            # Find the page object
            page = next((p for p in document.pages if p.index == local_pi), None)
            if page is None:
                continue
            try:
                label = _InjLabel(det_dict["label_id"])
            except (KeyError, ValueError):
                continue
            bbox_raw = det_dict.get("bbox", [0, 0, 1, 1])
            det = Detection(
                bbox=BBox(bbox_raw[0], bbox_raw[1], bbox_raw[2], bbox_raw[3]),
                label=label,
                confidence=float(det_dict.get("confidence", 1.0)),
                page_index=local_pi,
            )
            page.detections.append(det)
            injected_count += 1
        print(f"  Injected {injected_count} manual detections")

    # Pair opinions with exclusions
    # Remap exclusions: page_index in section = original_page_index - pdf_start
    section_excluded = None
    if excluded:
        section_excluded = set()
        for pi, label_id, bx, by in excluded:
            remapped = pi - pdf_start
            if 0 <= remapped <= pdf_end - pdf_start:
                section_excluded.add((remapped, label_id, bx, by))
        print(
            f"  Section exclusions ({len(section_excluded)} of {len(excluded)} total): {section_excluded}"
        )

    from blackletter.models import Label as _Label

    key_dets = [d for p in document.pages for d in p.detections if d.label == _Label.KEY_ICON]
    print(
        f"  KEY_ICONs in section: {[(d.page_index, round(d.bbox.x1), round(d.bbox.y1)) for d in key_dets]}"
    )

    opinions = _pair_opinions(document, excluded=section_excluded)
    print(f"  Found {len(opinions)} opinions in section")

    # Split
    redacted_dir = output_dir / "redacted"
    unredacted_dir = output_dir / "unredacted"

    redacted_paths = split_opinions(
        document.pdf_path,
        document,
        redacted_dir,
        first_page=page_start,
        redact_mode="redacted",
        excluded=section_excluded,
    )
    print(f"  Wrote {len(redacted_paths)} redacted PDFs")

    unredacted_paths = split_opinions(
        document.pdf_path,
        document,
        unredacted_dir,
        first_page=page_start,
        redact_mode="unredacted",
        excluded=section_excluded,
    )
    print(f"  Wrote {len(unredacted_paths)} unredacted PDFs")

    # Clean up temp file
    section_path.unlink(missing_ok=True)

    # Build opinion list
    result_opinions = []
    for f in sorted(redacted_dir.glob("*.pdf")):
        parts = f.stem.rsplit(".", 1)
        page_range = parts[-1] if len(parts) > 1 else f.stem
        range_parts = page_range.split("-")
        try:
            result_opinions.append(
                {
                    "filename": f.name,
                    "first_page": int(range_parts[0]),
                    "last_page": int(range_parts[1]),
                }
            )
        except (ValueError, IndexError):
            result_opinions.append({"filename": f.name, "first_page": 0, "last_page": 0})

    return {
        "opinions": result_opinions,
        "redacted_paths": [str(p) for p in redacted_paths],
        "unredacted_paths": [str(p) for p in unredacted_paths],
    }


def rebuild_full_redacted_from_detections(
    detections_json_path: str | Path,
    ocr_pdf_path: str | Path,
    output_path: str | Path,
    *,
    reporter: str | None = None,
    volume: str | None = None,
    first_page: int = 1,
    excluded: set[tuple[int, int, int, int]] | None = None,
) -> Path:
    """Rebuild the full redacted PDF from saved detections.json without re-running YOLO.

    Reconstructs the Document object from detections.json, re-pairs opinions
    (with optional exclusions), and calls _build_full_redacted.
    """
    import json as _json
    from blackletter.models import BBox, Detection, Document, Label, Page

    detections_json_path = Path(detections_json_path)
    ocr_pdf_path = Path(ocr_pdf_path)
    output_path = Path(output_path)

    raw = _json.loads(detections_json_path.read_text())

    # Get PDF page dimensions
    src_pdf = fitz.open(str(ocr_pdf_path))
    page_dims: dict[int, tuple[float, float]] = {}
    for i in range(len(src_pdf)):
        r = src_pdf.load_page(i).rect
        page_dims[i] = (r.width, r.height)
    src_pdf.close()

    # Group detections by page_index
    pages_data: dict[int, dict] = {}
    for entry in raw:
        pi = entry["page_index"]
        if pi not in pages_data:
            pages_data[pi] = {
                "page_number": entry.get("page_number"),
                "img_width": entry.get("img_width", 1),
                "img_height": entry.get("img_height", 1),
                "detections": [],
            }
        pages_data[pi]["detections"].append(entry)

    pages = []
    for pi in sorted(pages_data.keys()):
        pd = pages_data[pi]
        pdf_w, pdf_h = page_dims.get(pi, (612.0, 792.0))
        page = Page(
            index=pi,
            pdf_width=pdf_w,
            pdf_height=pdf_h,
            img_width=pd["img_width"],
            img_height=pd["img_height"],
            page_number=pd["page_number"],
        )
        for d in pd["detections"]:
            bbox = d["bbox"]
            page.detections.append(
                Detection(
                    bbox=BBox(bbox[0], bbox[1], bbox[2], bbox[3]),
                    label=Label(d["label_id"]),
                    confidence=d["confidence"],
                    page_index=pi,
                )
            )
        pages.append(page)

    document = Document(
        pdf_path=ocr_pdf_path,
        pages=pages,
        reporter=reporter,
        volume=volume,
        first_page=first_page,
        ocr_applied=True,
    )

    opinions = _pair_opinions(document, excluded=excluded)
    print(f"  Rebuilt {len(opinions)} opinions from detections")
    _build_full_redacted(document, opinions, output_path, excluded=excluded)
    return output_path


def _headnote_local_pages_to_delete(
    caption,
    key,
    pages_by_index: dict,
    mid: float,
    reporter: str | None = None,
) -> list[int]:
    """Return local page indices (0-based within the masked PDF) to delete for one opinion.

    Local index 0 = caption page, 1 = next page, etc.  Pages strictly between
    the caption page and the headnote-end page are fully headnote and can be removed.
    Returns an empty list if nothing should be deleted.
    """
    if caption.page_index == key.page_index:
        return []

    cap_key = caption.sort_key(mid)
    key_key = key.sort_key(mid)
    opinion_dets = []
    for src_idx in range(caption.page_index, key.page_index + 1):
        for d in pages_by_index[src_idx].detections:
            sk = d.sort_key(mid)
            if cap_key <= sk <= key_key:
                opinion_dets.append(d)
    opinion_dets.sort(key=lambda d: d.sort_key(mid))

    end_marker = _find_redaction_end(opinion_dets, caption, key, mid, reporter=reporter)
    if end_marker is not None:
        first_hn = caption.page_index
        last_hn = end_marker.page_index
    else:
        fallback_rects = _headnote_fallback_rects(opinion_dets, caption, pages_by_index, mid)
        if not fallback_rects:
            return []
        first_hn = caption.page_index
        last_hn = max(r[0] for r in fallback_rects)

    if last_hn - first_hn < 2:
        return []

    return [
        local_idx
        for local_idx, src_idx in enumerate(range(caption.page_index, key.page_index + 1))
        if first_hn < src_idx < last_hn
    ]


def _delete_headnote_pages(
    masked_paths: list[Path],
    opinions: list[tuple],
    document,
    rects_path: Path,
) -> None:
    """Remove fully-headnote middle pages from masked opinion PDFs."""
    pages_by_index = {p.index: p for p in document.pages}
    mid = document.pages[0].midpoint

    for path, (caption, key) in zip(masked_paths, opinions):
        pages_to_delete = _headnote_local_pages_to_delete(
            caption, key, pages_by_index, mid, reporter=document.reporter
        )
        if pages_to_delete and path.exists():
            tmp = path.with_suffix(".tmp.pdf")
            doc = fitz.open(str(path))
            doc.delete_pages(pages_to_delete)
            doc.save(str(tmp), garbage=4, deflate=True)
            doc.close()
            tmp.replace(path)


def generate_files(
    ocr_pdf: str | Path,
    output: str | Path,
    *,
    reporter: str | None = None,
    volume: str | None = None,
    first_page: int = 1,
    footnotes: bool = False,
    unredacted: bool = False,
    excluded: set[tuple[int, int, int, int]] | None = None,
) -> Path:
    """Phase 2: Generate redacted/split files from existing detections.

    Reads detections.json from the output dir, re-pairs opinions, then
    runs clean margins, build redacted PDF, and split opinions.
    """
    import json as _json
    import time as _time
    from blackletter.models import BBox, Detection, Document, Label, Page

    _t_total = _time.time()
    ocr_pdf = Path(ocr_pdf)
    output = Path(output)

    # Find detections.json — check output dir, parent dirs, and subdirs
    det_path = None
    for candidate in [
        output / "detections.json",
        output.parent / "detections.json",
        output.parent.parent / "detections.json",
    ]:
        if candidate.exists():
            det_path = candidate
            break
    if not det_path:
        # Search subdirectories
        for f in output.rglob("detections.json"):
            det_path = f
            output = f.parent  # update output to the dir containing detections
            break
    if not det_path:
        raise FileNotFoundError(f"detections.json not found in {output}")

    raw = _json.loads(det_path.read_text())
    print(f"Loaded {len(raw)} detections from {det_path.name}", flush=True)

    # Reconstruct Document from detections
    src_pdf = fitz.open(str(ocr_pdf))
    page_dims = {}
    for i in range(len(src_pdf)):
        r = src_pdf.load_page(i).rect
        page_dims[i] = (r.width, r.height)
    src_pdf.close()

    pages_data: dict[int, dict] = {}
    for entry in raw:
        pi = entry["page_index"]
        if pi not in pages_data:
            pages_data[pi] = {
                "page_number": entry.get("page_number"),
                "page_number_end": entry.get("page_number_end"),
                "img_width": entry.get("img_width", 1),
                "img_height": entry.get("img_height", 1),
                "detections": [],
            }
        pages_data[pi]["detections"].append(entry)

    # Load page metadata (column bounds) if available
    _pages_meta = {}
    for _candidate in [output / "pages_meta.json", output.parent / "pages_meta.json"]:
        if _candidate.exists():
            _pages_meta = _json.loads(_candidate.read_text())
            break

    pages = []
    for pi in sorted(pages_data.keys()):
        pd = pages_data[pi]
        pdf_w, pdf_h = page_dims.get(pi, (612.0, 792.0))
        meta = _pages_meta.get(str(pi), _pages_meta.get(pi, {}))
        page = Page(
            index=pi,
            pdf_width=pdf_w,
            pdf_height=pdf_h,
            img_width=pd["img_width"],
            img_height=pd["img_height"],
            page_number=pd["page_number"],
            page_number_end=pd.get("page_number_end"),
            col_left_x1=meta.get("col_left_x1", 0),
            col_left_x2=meta.get("col_left_x2", 0),
            col_right_x1=meta.get("col_right_x1", 0),
            col_right_x2=meta.get("col_right_x2", 0),
            midpoint=meta.get("midpoint", 0),
        )
        for d in pd["detections"]:
            bbox_raw = d.get("bbox", [0, 0, 1, 1])
            page.detections.append(
                Detection(
                    bbox=BBox(x1=bbox_raw[0], y1=bbox_raw[1], x2=bbox_raw[2], y2=bbox_raw[3]),
                    label=Label(d["label_id"]),
                    confidence=d["confidence"],
                    page_index=pi,
                )
            )
        # Fill in missing page numbers from first_page offset
        if page.page_number is None:
            page.page_number = pi + first_page
        pages.append(page)

    document = Document(
        pdf_path=ocr_pdf,
        pages=pages,
        reporter=reporter,
        volume=volume,
        first_page=first_page,
        ocr_applied=True,
    )

    # Pair opinions
    _t0 = _time.time()
    print("\nPairing opinions...", flush=True)
    opinions = _pair_opinions(document, excluded=excluded)
    print(f"  Found {len(opinions)} opinions ({_time.time() - _t0:.0f}s)", flush=True)

    # Split unredacted opinions from the clean OCR PDF BEFORE any margin modification
    unredacted_dir = output / "unredacted"
    if unredacted:
        _t0 = _time.time()
        print(f"\nSplitting unredacted into {unredacted_dir}...", flush=True)
        split_opinions(
            document.pdf_path,
            document,
            unredacted_dir,
            first_page=first_page,
            redact_mode="unredacted",
            extract_footnotes=footnotes,
            excluded=excluded,
        )
        print(f"  Unredacted done ({_time.time() - _t0:.0f}s)", flush=True)

    # Apply margins — use stored margin_rects.json if available, else compute fresh
    margin_rects_path = output / "margin_rects.json"
    _t0 = _time.time()
    if margin_rects_path.exists():
        print("\nApplying stored margin rects...", flush=True)
        _apply_margin_rects(document.pdf_path, margin_rects_path)
        print(f"  Margins applied ({_time.time() - _t0:.0f}s)", flush=True)
    else:
        from blackletter.margins import clean_margins

        print("\nCleaning margins...", flush=True)
        clean_margins(document.pdf_path)
        print(f"  Margins cleaned ({_time.time() - _t0:.0f}s)", flush=True)

    # Extract images
    _t0 = _time.time()
    images_dir = output / "images"
    n_images = _extract_images(document, images_dir)
    if n_images:
        print(f"\nExtracted {n_images} images ({_time.time() - _t0:.0f}s)", flush=True)

    # Build scan_name using actual page numbers from OCR
    parts = []
    if reporter:
        parts.append(reporter)
    if volume:
        parts.append(str(volume))
    # Use actual page numbers: first page's number and last page's number
    actual_first = pages[0].page_number if pages and pages[0].page_number else first_page
    last_pg = pages[-1] if pages else None
    actual_last = (
        (last_pg.page_number_end or last_pg.page_number)
        if last_pg and last_pg.page_number
        else first_page + len(pages) - 1
    )
    parts.append(str(actual_first))
    parts.append(str(actual_last))
    scan_name = ".".join(parts)

    # Build full redacted PDF — use precomputed rects if available, else compute them now
    full_redacted_path = output / f"{scan_name}.redacted.pdf"
    _t0 = _time.time()
    rects_path = output / "redaction_rects.json"
    if not rects_path.exists():
        print("\nComputing redaction rects...", flush=True)
        rects = compute_redaction_rects(document, opinions, skip_doctr=True)
        rects_path.write_text(_json.dumps(rects))
    if rects_path.exists():
        print("\nBuilding full redacted PDF from precomputed rects...", flush=True)
        _build_redacted_from_rects(document, rects_path, full_redacted_path)
    else:
        print("\nBuilding full redacted PDF...", flush=True)
        _build_full_redacted(document, opinions, full_redacted_path, excluded=excluded)
    print(f"  Full redacted done ({_time.time() - _t0:.0f}s)", flush=True)

    # Split redacted and masked opinions
    redacted_dir = output / "redacted"
    masked_dir = output / "masked"

    _t0 = _time.time()
    print(f"\nSplitting redacted into {redacted_dir}...", flush=True)
    if rects_path.exists():
        # Split from the already-redacted PDF so manual adjustments are preserved
        _split_from_redacted(full_redacted_path, document, opinions, redacted_dir, first_page)
    else:
        split_opinions(
            document.pdf_path,
            document,
            redacted_dir,
            first_page=first_page,
            redact_mode="redacted",
            extract_footnotes=footnotes,
            excluded=excluded,
        )
    print(f"  Redacted done ({_time.time() - _t0:.0f}s)", flush=True)

    _t0 = _time.time()
    print(f"\nSplitting masked into {masked_dir}...", flush=True)
    if rects_path.exists():
        # Split masked from the already-redacted PDF too
        masked_paths = _split_from_redacted(
            full_redacted_path, document, opinions, masked_dir, first_page
        )
        _delete_headnote_pages(masked_paths, opinions, document, rects_path)
    else:
        masked_paths = split_opinions(
            document.pdf_path,
            document,
            masked_dir,
            first_page=first_page,
            redact_mode="masked",
            excluded=excluded,
        )
    _build_masked_opinions(masked_paths, opinions, document, masked_dir)
    print(f"  Masked done ({_time.time() - _t0:.0f}s)", flush=True)

    print(f"\n── Generate complete: {_time.time() - _t_total:.0f}s ──", flush=True)
    return output


def process(
    pdf: str | Path,
    output: str | Path,
    *,
    reporter: str | None = None,
    volume: str | None = None,
    first_page: int = 1,
    model: str | Path | None = None,
    large: bool = False,
    footnotes: bool = False,
    unredacted: bool = False,
    no_shrink: bool = False,
    optimize: int = 1,
    progress_callback=None,
    excluded: set[tuple[int, int, int, int]] | None = None,
    bitonal: bool = False,
    detect_only: bool = False,
    has_state_abbrev: bool | None = None,
) -> Path:
    """Process a legal PDF: scan, verify, split, and redact.

    Args:
        pdf: Path to the source PDF.
        output: Base output directory.
        reporter: Reporter abbreviation (e.g. "f3d", "a3d").
        volume: Volume number.
        first_page: Page number of the first page in the PDF.
        model: Path to YOLO model weights. Defaults to bundled model.
        large: Use the large model (analyze.pt) with more detection classes.
        footnotes: Extract footnotes into separate PDFs.
        unredacted: Also generate unredacted opinion PDFs.
        no_shrink: Skip downsampling images.
        optimize: ocrmypdf optimization level (0-3).
        progress_callback: Optional callable(current, total, message) for progress.
        detect_only: If True, stop after detection + pairing (Phase 1).

    Returns:
        Path to the output directory containing all results.
    """
    if model is None:
        if large:
            from blackletter.analyze import DEFAULT_ANALYZE_MODEL

            model = DEFAULT_ANALYZE_MODEL
        else:
            model = DEFAULT_MODEL
    args = argparse.Namespace(
        pdf=Path(pdf),
        output=Path(output),
        reporter=reporter,
        volume=volume,
        first_page=first_page,
        model=Path(model),
        footnotes=footnotes,
        unredacted=unredacted,
        no_shrink=no_shrink,
        optimize=optimize,
        progress_callback=progress_callback,
        excluded=excluded,
        bitonal=bitonal,
        detect_only=detect_only,
        has_state_abbrev=has_state_abbrev,
    )
    cmd_process(args)
    return _build_output_dir(args)


def build_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Register the process subcommand on an existing subparser group."""
    p = sub.add_parser(
        "process",
        help="Scan, verify, and split into opinion PDFs in one pass",
    )
    p.add_argument("pdf", type=Path, help="Path to the source PDF")
    p.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"Path to the YOLO model weights (.pt) (default: {DEFAULT_MODEL.name})",
    )
    p.add_argument(
        "--medium",
        action="store_true",
        help="Use the medium model (medium.pt, run_57) with 17 detection classes",
    )
    p.add_argument(
        "--large",
        action="store_true",
        help="Use the large model (analyze.pt, run_59) with 21 detection classes",
    )
    p.add_argument(
        "--reporter",
        type=str,
        default=None,
        help="Reporter abbreviation (e.g. f3d, a3d)",
    )
    p.add_argument(
        "--volume",
        type=str,
        default=None,
        help="Volume number",
    )
    p.add_argument(
        "--first-page",
        type=int,
        default=1,
        help="Page number of the first page in the PDF (default: 1)",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Base output directory (organized as <reporter>/<volume>/<first-page>)",
    )
    p.add_argument(
        "--footnotes",
        action="store_true",
        help="Extract footnotes into separate PDFs",
    )
    p.add_argument(
        "--unredacted",
        action="store_true",
        help="Skip generating unredacted opinion PDFs",
    )
    p.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU-only inference (no GPU)",
    )
    p.add_argument(
        "--no-shrink",
        action="store_true",
        help="Skip downsampling (default: shrink to ~148 KB/page)",
    )
    p.add_argument(
        "--bitonal",
        action="store_true",
        help="Convert to bitonal (1-bit B&W) before processing for speed/size",
    )
    p.add_argument(
        "--detect-only",
        action="store_true",
        help="Stop after detection + pairing (Phase 1) — no file generation",
    )
    p.add_argument(
        "--optimize",
        type=int,
        default=1,
        choices=[0, 1, 2, 3],
        help="ocrmypdf optimization level (0=none, 1=lossless, 2=lossy, 3=aggressive)",
    )
    return p
