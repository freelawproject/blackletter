"""Clean scan artifacts from page margins using text-layer boundaries."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import fitz

logger = logging.getLogger(__name__)

# Buffer in PDF points (72 pts = 1 inch)
DEFAULT_BUFFER = 5.0  # ~1.8mm

# Only clean margins if text spans at least this fraction of the page width.
# Pages with images/appendices typically have narrow text spans and are skipped.
MIN_TEXT_WIDTH_FRACTION = 0.40


def _text_bounds(
    fitz_page: fitz.Page, page_width: float
) -> tuple[float, float, float, float] | None:
    """Find the bounding box of all text on a page.

    Returns (left, top, right, bottom) in PDF points, or None if the text
    doesn't span enough of the page to justify margin cleanup.
    """
    blocks = fitz_page.get_text("blocks")
    text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]
    if not text_blocks:
        return None

    left = min(b[0] for b in text_blocks)
    top = min(b[1] for b in text_blocks)
    right = max(b[2] for b in text_blocks)
    bottom = max(b[3] for b in text_blocks)

    # Skip pages where text is too narrow — likely an appendix or image page
    if (right - left) < page_width * MIN_TEXT_WIDTH_FRACTION:
        return None

    return left, top, right, bottom


def compute_margin_rects(
    pdf_path: Path,
    buffer: float = DEFAULT_BUFFER,
) -> list[dict]:
    """Compute margin rects for each page without applying them.

    Returns list of {page_index, rects: [{x0, y0, x1, y1}]}.
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(str(pdf_path))
    result = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        pw = page.rect.width
        ph = page.rect.height
        bounds = _text_bounds(page, pw)
        if bounds is None:
            result.append({"page_index": page_idx, "rects": []})
            continue

        left, top, right, bottom = bounds
        safe_left = max(0, left - buffer)
        safe_top = max(0, top - buffer)
        safe_right = min(pw, right + buffer)
        safe_bottom = min(ph, bottom + buffer)

        rects = []
        if safe_left > 1:
            rects.append({"x0": 0, "y0": 0, "x1": round(safe_left, 1), "y1": round(ph, 1)})
        if pw - safe_right > 1:
            rects.append(
                {"x0": round(safe_right, 1), "y0": 0, "x1": round(pw, 1), "y1": round(ph, 1)}
            )
        if safe_top > 1:
            rects.append(
                {
                    "x0": round(safe_left, 1),
                    "y0": 0,
                    "x1": round(safe_right, 1),
                    "y1": round(safe_top, 1),
                }
            )
        if ph - safe_bottom > 1:
            rects.append(
                {
                    "x0": round(safe_left, 1),
                    "y0": round(safe_bottom, 1),
                    "x1": round(safe_right, 1),
                    "y1": round(ph, 1),
                }
            )
        result.append({"page_index": page_idx, "rects": rects})

    doc.close()
    return result


def clean_margins(
    pdf_path: Path,
    buffer: float = DEFAULT_BUFFER,
    output_path: Path | None = None,
) -> Path:
    """White out margins beyond the text content area on every page.

    Uses the embedded text layer to find content boundaries, then
    applies white redactions to the four margin strips (left, right,
    top, bottom) with a safety buffer around the content.

    Modifies the PDF in-place if output_path is None.

    Returns the output path.
    """
    pdf_path = Path(pdf_path)
    if output_path is None:
        output_path = pdf_path

    doc = fitz.open(str(pdf_path))
    cleaned = 0

    # Detect bitonal — apply_redactions corrupts CCITT G4 streams
    _sample_imgs = doc[0].get_images(full=True) if doc.page_count else []
    is_bitonal = bool(_sample_imgs and _sample_imgs[0][4] == 1)

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        pw = page.rect.width
        ph = page.rect.height

        bounds = _text_bounds(page, pw)
        if bounds is None:
            continue

        left, top, right, bottom = bounds

        # Apply buffer (shrink the content area we protect)
        safe_left = left - buffer
        safe_top = top - buffer
        safe_right = right + buffer
        safe_bottom = bottom + buffer

        # Clamp to page bounds
        safe_left = max(0, safe_left)
        safe_top = max(0, safe_top)
        safe_right = min(pw, safe_right)
        safe_bottom = min(ph, safe_bottom)

        # White out the four margin strips
        white = (1, 1, 1)
        margin_rects = []

        # Left margin
        if safe_left > 1:
            r = fitz.Rect(0, 0, safe_left, ph)
            margin_rects.append((r, white))
            page.add_redact_annot(r, fill=white)

        # Right margin
        if pw - safe_right > 1:
            r = fitz.Rect(safe_right, 0, pw, ph)
            margin_rects.append((r, white))
            page.add_redact_annot(r, fill=white)

        # Top margin (between left and right content edges)
        if safe_top > 1:
            r = fitz.Rect(safe_left, 0, safe_right, safe_top)
            margin_rects.append((r, white))
            page.add_redact_annot(r, fill=white)

        # Bottom margin (between left and right content edges)
        if ph - safe_bottom > 1:
            r = fitz.Rect(safe_left, safe_bottom, safe_right, ph)
            margin_rects.append((r, white))
            page.add_redact_annot(r, fill=white)

        if is_bitonal and margin_rects:
            from blackletter.process import _redact_bitonal_image

            _redact_bitonal_image(page, doc, margin_rects)
            page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
        else:
            page.apply_redactions()
        cleaned += 1

    if not is_bitonal:
        # Recompress images — apply_redactions converts JPEGs to PNG, inflating size
        from blackletter.scanner import recompress_images

        recompress_images(doc, quality=65)

    total = len(doc)

    if output_path == pdf_path:
        # Can't save over the source directly — use temp file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False, dir=pdf_path.parent) as tmp:
            tmp_path = Path(tmp.name)
        doc.save(str(tmp_path), garbage=4, deflate=True)
        doc.close()
        tmp_path.replace(pdf_path)
    else:
        doc.save(str(output_path), garbage=4, deflate=True)
        doc.close()

    logger.info("Margin cleanup: %d/%d pages cleaned", cleaned, total)
    return output_path
