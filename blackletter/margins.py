"""Clean scan artifacts from page margins.

Each page's printed-content box is measured, and the strips outside it are
whited out. The box comes from whichever signal the page actually has:

1. the text layer, when the page has one (:func:`blackletter.api.ocr` ran
   over the file, or it was born digital);
2. otherwise the rendered ink, i.e. the tightest box containing the dark
   pixels that make up the printed text.

The ink path exists because the text layer is optional and expensive: on a
bitonal scan that never went through ocrmypdf, ``get_text("blocks")``
returns nothing for every page and margin cleanup used to silently do
nothing at all. The measurement lives in :mod:`blackletter.ink`, which also
has to ignore scanner artifacts along the page edges (the very thing margin
cleanup exists to remove); see ``ink.content_box`` for how, and for the
tunable thresholds. It errs toward a larger content box, i.e. narrower
margin strips and less cleanup, never toward covering printed text.

Ink alone is not enough, though: it is the union of *everything* dark, so
a single bleed-through mark in a corner drags the content box out to the
page edge and the strips on that side shrink or vanish. Detections are the
second signal, and a sturdier one because they describe content rather
than marks: ``TEXT_COLUMN`` boxes bound the printed text horizontally, and
the header row (``PAGE_HEADER`` / ``PAGE_NUMBER`` / ``STATE_ABBREVIATION``)
bounds it vertically. The two estimates are intersected, so a bound is only
tightened when both signals support it.

The strips are laid out so the header row is never at risk: full-width
strips above and below the text body, and side strips that span the body
rows only. A page number sitting outside the column band therefore survives
by construction rather than by luck. The union of the four strips is the
same region either way, so this is a reshaping, not extra coverage.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Sequence
from pathlib import Path

import fitz

from blackletter.ink import content_box
from blackletter.models import Label, Page

logger = logging.getLogger(__name__)

# Buffer in PDF points (72 pts = 1 inch)
DEFAULT_BUFFER = 5.0  # ~1.8mm

# Only clean margins if the content spans at least this fraction of the
# page width. Pages with images/appendices typically have narrow content
# spans and are skipped.
MIN_TEXT_WIDTH_FRACTION = 0.40

# Detections whose bboxes bound the printed text horizontally.
COLUMN_LABELS = frozenset({Label.TEXT_COLUMN})

# Detections that make up the header row, which bounds the text vertically
# and must never be covered by a side strip.
HEADER_LABELS = frozenset({Label.PAGE_HEADER, Label.PAGE_NUMBER, Label.STATE_ABBREVIATION})

# A "header" detection lying entirely within this many points of the top
# edge is bleed-through from the facing page, not this page's header. Those
# are exactly what a top strip is for, so they must not define the top
# bound. Real headers on letter-size reporter pages sit around 38-55 pt.
EDGE_BLEED_PT = 20.0

# ...and one below this fraction of the page height is not part of the
# header row either. Some reporters print the page number at the foot of
# the page, and a footer must not be allowed to define the top bound: that
# would put a full-width top strip over the whole body of the page.
HEADER_MAX_FRACTION = 0.25


def _text_bounds(
    fitz_page: fitz.Page, page_width: float
) -> tuple[float, float, float, float] | None:
    """Find the bounding box of all text and images on a page.

    :param fitz_page: A PyMuPDF page object.
    :param page_width: Width of the page in PDF points.
    :returns: ``(left, top, right, bottom)`` in PDF points, or ``None``
        if the text doesn't span enough of the page to justify margin
        cleanup.
    """
    blocks = fitz_page.get_text("blocks")
    text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]
    if not text_blocks:
        return None

    left = min(b[0] for b in text_blocks)
    top = min(b[1] for b in text_blocks)
    right = max(b[2] for b in text_blocks)
    bottom = max(b[3] for b in text_blocks)

    # Skip pages where text is too narrow, likely an appendix or image page
    if (right - left) < page_width * MIN_TEXT_WIDTH_FRACTION:
        return None

    # Extend bounds to include image blocks (e.g. key icons at page bottom).
    # Only extend vertically; images outside the text column are margin artifacts
    img_blocks = [b for b in blocks if b[6] == 1]
    for b in img_blocks:
        # Only consider images that overlap the text column horizontally
        if b[2] > left and b[0] < right:
            top = min(top, b[1])
            bottom = max(bottom, b[3])

    return left, top, right, bottom


def _content_bounds(
    fitz_page: fitz.Page, page_width: float
) -> tuple[float, float, float, float] | None:
    """Find the page's content box from text if it has any, else from ink.

    :param fitz_page: A PyMuPDF page object.
    :param page_width: Width of the page in PDF points.
    :returns: ``(left, top, right, bottom)`` in PDF points, or ``None``
        when neither signal gives a box wide enough to trust.
    """
    return _text_bounds(fitz_page, page_width) or content_box(fitz_page)


def _detection_bounds(page: Page) -> tuple[float | None, float | None, float | None]:
    """Content bounds a page's detections support, in PDF points.

    The horizontal band spans the text columns *and* the header row. A page
    number can sit outside the columns, and the side strips reach up into
    the header row, so leaving the header out of the band would let a strip
    cover it.

    :param page: The page whose detections to read.
    :returns: ``(band_left, band_right, header_top)``, each None when the
        page has no detection to derive it from.
    """
    band_left = band_right = header_top = None
    sx, sy = page.scale_x, page.scale_y
    header_limit = page.pdf_height * HEADER_MAX_FRACTION
    for d in page.detections:
        is_header = d.label in HEADER_LABELS
        if is_header and (d.bbox.y2 * sy <= EDGE_BLEED_PT or d.bbox.y1 * sy >= header_limit):
            # Bleed-through from the facing page, or a footer: neither
            # defines a bound, and covering the bleed is the whole point.
            continue
        if is_header or d.label in COLUMN_LABELS:
            left, right = d.bbox.x1 * sx, d.bbox.x2 * sx
            band_left = left if band_left is None else min(band_left, left)
            band_right = right if band_right is None else max(band_right, right)
        if is_header:
            top = d.bbox.y1 * sy
            header_top = top if header_top is None else min(header_top, top)
    return band_left, band_right, header_top


def _tighten_bounds(
    bounds: tuple[float, float, float, float],
    page: Page,
) -> tuple[float, float, float, float]:
    """Intersect measured content bounds with what detections support.

    Ink is the union of every mark on the page, so it only ever errs
    outward; detections describe content, so they only err inward. Taking
    the tighter of the two per side means a bound moves in only when both
    signals agree there is nothing there.

    Falls back to ``bounds`` if the result would be degenerate (a bogus
    ``TEXT_COLUMN`` box should not be able to collapse the content box).

    :param bounds: ``(left, top, right, bottom)`` from text or ink.
    :param page: The page whose detections to read.
    :returns: The tightened ``(left, top, right, bottom)``.
    """
    left, top, right, bottom = bounds
    band_left, band_right, header_top = _detection_bounds(page)
    if band_left is not None:
        left = max(left, band_left)
    if band_right is not None:
        right = min(right, band_right)
    if header_top is not None:
        top = max(top, header_top)
    if right - left < page.pdf_width * MIN_TEXT_WIDTH_FRACTION or bottom <= top:
        return bounds
    return left, top, right, bottom


def _rects_for_bounds(
    bounds: tuple[float, float, float, float],
    page_width: float,
    page_height: float,
    buffer: float,
) -> list[dict]:
    """Build the margin strips around a content box.

    Full-width strips above and below the content, then side strips that
    span only the rows between them. Keeping the side strips out of the
    header and footer rows is what lets the x-bounds be tightened to the
    text columns without ever reaching a page number in a corner; the
    corners themselves are still covered, by the full-width strips.

    :param bounds: ``(left, top, right, bottom)`` content box.
    :param page_width: Page width in PDF points.
    :param page_height: Page height in PDF points.
    :param buffer: Safety buffer in PDF points around the content.
    :returns: List of ``{x0, y0, x1, y1}`` rect dicts, ordered left, right,
        top, bottom.
    """
    left, top, right, bottom = bounds
    safe_left = max(0, left - buffer)
    safe_top = max(0, top - buffer)
    safe_right = min(page_width, right + buffer)
    safe_bottom = min(page_height, bottom + buffer)

    rects: list[dict] = []
    if safe_left > 1:
        rects.append(
            {
                "x0": 0,
                "y0": round(safe_top, 1),
                "x1": round(safe_left, 1),
                "y1": round(safe_bottom, 1),
            }
        )
    if page_width - safe_right > 1:
        rects.append(
            {
                "x0": round(safe_right, 1),
                "y0": round(safe_top, 1),
                "x1": round(page_width, 1),
                "y1": round(safe_bottom, 1),
            }
        )
    if safe_top > 1:
        rects.append({"x0": 0, "y0": 0, "x1": round(page_width, 1), "y1": round(safe_top, 1)})
    if page_height - safe_bottom > 1:
        rects.append(
            {
                "x0": 0,
                "y0": round(safe_bottom, 1),
                "x1": round(page_width, 1),
                "y1": round(page_height, 1),
            }
        )
    return rects


# Detections too noisy at a page edge to push a margin strip back. They are
# the ones that bound the content box in the first place (see
# ``_detection_bounds``), and a bleed-through blob labelled PAGE_NUMBER is
# exactly what a strip is meant to cover.
NO_PUSHBACK_LABELS = frozenset({Label.PAGE_NUMBER, Label.PAGE_HEADER, Label.STATE_ABBREVIATION})


def _shrink_rects_for_detections(page: Page, rects: list[dict]) -> None:
    """Pull back any strip that would cover a detection.

    The bounds tightening in :func:`_tighten_bounds` positions the strips
    from the column band and the header row, which says where the *text*
    is. It says nothing about a key icon at the foot of a page, a caption
    that reaches into a margin, or an image that bleeds outward, so a strip
    can still land on one. Each strip is anchored to a page edge, and which
    edge tells us which of its own edges to pull back.

    Rects are modified in place, in PDF points.

    :param page: The page whose detections to respect.
    :param rects: That page's margin strips.
    """
    boxes = [
        d.bbox.to_pdf(page.scale_x, page.scale_y)
        for d in page.detections
        if d.label not in NO_PUSHBACK_LABELS
    ]
    if not boxes:
        return
    pdf_w, pdf_h = page.pdf_width, page.pdf_height
    for rect in rects:
        full_width = rect["x0"] <= 1 and rect["x1"] >= pdf_w - 1
        for box in boxes:
            if not (
                box.x1 < rect["x1"]
                and box.x2 > rect["x0"]
                and box.y1 < rect["y1"]
                and box.y2 > rect["y0"]
            ):
                continue
            if full_width and rect["y0"] <= 1:
                rect["y1"] = min(rect["y1"], box.y1)
            elif full_width and rect["y1"] >= pdf_h - 1:
                rect["y0"] = max(rect["y0"], box.y2)
            elif rect["x0"] <= 1:
                rect["x1"] = min(rect["x1"], box.x1)
            elif rect["x1"] >= pdf_w - 1:
                rect["x0"] = max(rect["x0"], box.x2)


def compute_margin_rects(
    pdf_path: Path,
    buffer: float = DEFAULT_BUFFER,
    pages: Sequence[Page] | None = None,
) -> list[dict]:
    """Compute margin rects for each page without applying them.

    Pages whose content box cannot be established get an empty rect list,
    meaning "leave this page alone".

    :param pdf_path: Path to the input PDF file.
    :param buffer: Safety buffer in PDF points around the content area.
    :param pages: Optional detected pages, used to tighten the bounds.
        Without them the bounds come from the page's text or marks alone,
        which is also what happens on a page carrying no ``TEXT_COLUMN``
        detection.
    :returns: List of dicts with ``page_index``, ``rects``, ``page_width``
        and ``page_height`` keys, where each rect is a dict with ``x0``,
        ``y0``, ``x1``, ``y1`` in PDF points.
    """
    pdf_path = Path(pdf_path)
    by_index = {p.index: p for p in pages or []}
    result = []

    with fitz.open(str(pdf_path)) as doc:
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            pw = page.rect.width
            ph = page.rect.height
            entry = {
                "page_index": page_idx,
                "rects": [],
                # Consumers that adjust these rects need the page size and
                # cannot always infer it from the rects themselves.
                "page_width": round(pw, 1),
                "page_height": round(ph, 1),
            }
            bounds = _content_bounds(page, pw)
            if bounds is None:
                result.append(entry)
                continue

            detected = by_index.get(page_idx)
            if detected is not None:
                bounds = _tighten_bounds(bounds, detected)
            entry["rects"] = _rects_for_bounds(bounds, pw, ph, buffer)
            if detected is not None:
                _shrink_rects_for_detections(detected, entry["rects"])
            result.append(entry)

    return result


def clean_margins(
    pdf_path: Path,
    buffer: float = DEFAULT_BUFFER,
    output_path: Path | None = None,
    pages: Sequence[Page] | None = None,
) -> Path:
    """White out margins beyond the content area on every page.

    Finds each page's content boundaries (see :func:`compute_margin_rects`),
    then applies white redactions to the margin strips around them.

    :param pdf_path: Path to the input PDF file.
    :param buffer: Safety buffer in PDF points around the content area.
    :param output_path: Where to write the cleaned PDF. If ``None``,
        modifies the PDF in-place.
    :param pages: Optional detected pages, used to tighten the bounds.
    :returns: The output path.
    """
    pdf_path = Path(pdf_path)
    if output_path is None:
        output_path = pdf_path

    margins_by_page = {
        entry["page_index"]: entry["rects"]
        for entry in compute_margin_rects(pdf_path, buffer=buffer, pages=pages)
    }
    cleaned = 0

    with fitz.open(str(pdf_path)) as doc:
        # Detect bitonal. apply_redactions corrupts CCITT G4 streams
        _sample_imgs = doc[0].get_images(full=True) if doc.page_count else []
        is_bitonal = bool(_sample_imgs and _sample_imgs[0][4] == 1)

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            white = (1, 1, 1)
            margin_rects = [
                (fitz.Rect(r["x0"], r["y0"], r["x1"], r["y1"]), white)
                for r in margins_by_page.get(page_idx, [])
            ]
            if not margin_rects:
                continue

            for rect, color in margin_rects:
                page.add_redact_annot(rect, fill=color)

            if is_bitonal:
                from blackletter.process import _redact_bitonal_image

                _redact_bitonal_image(page, doc, margin_rects)
                page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
            else:
                page.apply_redactions()
            # Overdraw with fill-only rects to cover 1pt stroke from apply_redactions
            for rect, color in margin_rects:
                page.draw_rect(rect, fill=color, color=None, width=0)
            cleaned += 1

        if not is_bitonal:
            # Recompress images. apply_redactions converts JPEGs to PNG, inflating size
            from blackletter.scanner import recompress_images

            recompress_images(doc, quality=65)

        total = len(doc)

        if output_path == pdf_path:
            # Can't save over the source directly, use temp file
            with tempfile.NamedTemporaryFile(
                suffix=".pdf", delete=False, dir=pdf_path.parent
            ) as tmp:
                tmp_path = Path(tmp.name)
            doc.save(str(tmp_path), garbage=4, deflate=True)
        else:
            doc.save(str(output_path), garbage=4, deflate=True)
            tmp_path = None

    if tmp_path is not None:
        tmp_path.replace(pdf_path)

    logger.info("Margin cleanup: %d/%d pages cleaned", cleaned, total)
    return output_path
