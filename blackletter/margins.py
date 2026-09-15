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

Detections bound the text, but they do not say where the text stops when
the ink on the far side of the band reads as text itself. A long blot down
the outer edge of a leaf does: its pixel columns carry a middling fraction
of dark rows, exactly what printed type looks like, so the band tightening
refuses to give it up and the strip on that side stops short of it. The
third signal is the caller's own ``Page.text_box``: a caller that runs an
OCR pass or a layout model knows where the text is far better than the
ink does, and the content box is intersected with it, last. That box can
only make the content box smaller, it is never allowed to cut inside a
header-row or ``TEXT_COLUMN`` detection (some reporters print the page
number at the foot, and no reader describes it), and a box that would keep
less than ``MARGIN_MIN_KEEP_RATIO`` of the tightened box is refused, so a
partial read never puts a strip through the type. Every refusal keeps the
box as the first two signals left it, so a page with a text box never
answers looser than the same page without one, and a page whose caller
has no box answers as it did before.

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

from blackletter.ink import content_box, page_mask
from blackletter.models import BBox, Detection, Label, Page

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

# The least of the content box (as tightened by the band and the header
# row) that a caller's ``Page.text_box`` may keep for the box to be fitted
# to it. A partial read (cells that describe a fifth of the page) is the
# one failure the fit cannot see from the outside, and this floor is the
# only guard in front of it. Measured on scan 1828 of the scanning app
# (143 S.Ct., 888 pages): the pages a correct fit had to refuse kept 18-23%
# of the box, the accepted pages kept 44% at the minimum, 70.6% at the 1st
# percentile and 92.1% at the median. 0.30 sits in the middle of that gap.
MARGIN_MIN_KEEP_RATIO = 0.30

# Detections whose extent a fitted content box may never cut inside: the
# header row, because some reporters print the page number at the foot and
# no reader describes it, and the text columns, because they are where the
# text is by the one signal that describes content rather than marks.
HOLD_LABELS = HEADER_LABELS | COLUMN_LABELS


def _is_edge_bleed(page: Page, d: Detection) -> bool:
    """Is a header-family detection bleed-through at the top edge?

    A ``PAGE_NUMBER`` box wholly within ``EDGE_BLEED_PT`` of the top edge
    is the facing page's number showing through, and a strip is meant to
    cover it, so it may neither define nor hold a bound. There is no
    matching band at the foot: ``EDGE_BLEED_PT`` was measured against
    running heads only, and a folio printed close to the foot is content
    that a wrongly exempted detection would let a strip cover.

    :param page: The page the detection is on, for the scale.
    :param d: The detection.
    :returns: True when the box is inside the top band.
    """
    return d.bbox.y2 * page.scale_y <= EDGE_BLEED_PT


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
    # Images are opt-in: get_text("blocks") omits them under its default
    # flags, so the extension below never saw one and a key icon at the foot
    # of a page fell outside the content box it is supposed to widen.
    blocks = fitz_page.get_text("blocks", flags=fitz.TEXTFLAGS_BLOCKS | fitz.TEXT_PRESERVE_IMAGES)
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

    Text wins where a page has any, including a text layer this library's
    own OCR produced. Its block bounds run large, so the strips come out
    shyer than the ink would place them, which is the safe direction; the
    redaction geometry distrusts those same word positions because there
    the error runs the other way.

    :param fitz_page: A PyMuPDF page object.
    :param page_width: Width of the page in PDF points.
    :returns: ``(left, top, right, bottom)`` in PDF points, or ``None``
        when neither signal gives a box wide enough to trust.
    """
    return _text_bounds(fitz_page, page_width) or content_box(fitz_page)


def _detection_bounds(page: Page) -> tuple[float | None, float | None, float | None]:
    """Content bounds a page's detections support, in PDF points.

    The horizontal band spans the text columns *and* the header family,
    wherever it sits on the page. A page number can sit outside the
    columns, and the side strips reach the header row and the foot, so
    leaving one out of the band would let a strip cover it: a corner folio
    is small enough that the ink check reads it as a speck and lets the
    band past it. Only a header-family detection above
    ``HEADER_MAX_FRACTION`` defines the header top, though, since a footer
    there would put a full-width strip over the body.

    :param page: The page whose detections to read.
    :returns: ``(band_left, band_right, header_top)``, each None when the
        page has no detection to derive it from.
    """
    band_left = band_right = header_top = None
    sx, sy = page.scale_x, page.scale_y
    header_limit = page.pdf_height * HEADER_MAX_FRACTION
    for d in page.detections:
        is_header = d.label in HEADER_LABELS
        if is_header and _is_edge_bleed(page, d):
            # Bleed-through from the facing page defines nothing, and
            # covering it is the whole point.
            continue
        if is_header or d.label in COLUMN_LABELS:
            left, right = d.bbox.x1 * sx, d.bbox.x2 * sx
            band_left = left if band_left is None else min(band_left, left)
            band_right = right if band_right is None else max(band_right, right)
        if is_header and d.bbox.y1 * sy < header_limit:
            top = d.bbox.y1 * sy
            header_top = top if header_top is None else min(header_top, top)
    return band_left, band_right, header_top


# What the ink outside the detection band has to look like before a side
# bound may be tightened past it. A scanner artifact is either near-solid in
# its own columns (a platen line, a fold, a gutter shadow) or barely there (a
# speck of dust, a smudge). Printed text is neither: its columns carry a
# middling fraction of dark rows. These two thresholds bracket that gap.
ARTIFACT_MIN_ROW_FRACTION = 0.60
SPECK_MAX_ROW_FRACTION = 0.03


def _ink_is_artifact_like(
    fitz_page: fitz.Page,
    x0: float,
    x1: float,
    top: float,
    bottom: float,
) -> bool:
    """Is the ink in a vertical slice safe to white out?

    Asked of the ink a detection band would give up. Density per pixel
    column is what separates the cases, and extent is not: a platen line
    down the edge of a page runs the full height and must be covered, while
    the tail of a table row runs a dozen rows and must not be.

    :param fitz_page: The page to measure.
    :param x0: Left edge of the slice, in PDF points.
    :param x1: Right edge of the slice, in PDF points.
    :param top: Top of the region of interest, in PDF points.
    :param bottom: Bottom of the region of interest, in PDF points.
    :returns: True when every inked column in the slice is either near-solid
        or negligible, and so when the slice holds nothing that reads as
        text. True for an empty slice, which gives up nothing.
    """
    if x1 - x0 <= 0 or bottom - top <= 0:
        return True
    mask, sx, sy = page_mask(fitz_page)
    height, width = mask.shape
    c0 = max(0, min(int(x0 / sx), width))
    c1 = max(c0, min(int(round(x1 / sx)), width))
    r0 = max(0, min(int(top / sy), height))
    r1 = max(r0, min(int(round(bottom / sy)), height))
    window = mask[r0:r1, c0:c1]
    if not window.size or window.shape[0] == 0:
        return True
    per_column = window.sum(axis=0) / window.shape[0]
    inked = per_column[per_column > 0]
    if not inked.size:
        return True
    text_like = (inked > SPECK_MAX_ROW_FRACTION) & (inked < ARTIFACT_MIN_ROW_FRACTION)
    return not bool(text_like.any())


def _fit_to_text_box(
    bounds: tuple[float, float, float, float],
    page: Page,
    buffer: float,
) -> tuple[float, float, float, float]:
    """Intersect content bounds with the caller's own text box, if any.

    The text box arrives in the page's pixels and is padded by ``buffer``
    on every side, the same slack the strips leave around the ink. It may
    only make the box smaller: a side moves in when the text box says the
    text stops short of the ink, and never out. Whatever refuses the fit
    returns ``bounds`` as given, so a page with a text box never answers
    looser than the same page without one.

    The box is never allowed to cut inside a header-row or text-column
    detection (see ``HOLD_LABELS``). Some reporters print the page number
    at the foot, and no reader describes it, so each such detection holds
    the four sides of the box off itself, apart from bleed-through at the
    top edge (see :func:`_is_edge_bleed`). The side strips span the header
    row vertically, so the horizontal hold is what keeps a strip off a
    corner number the caller's reader missed. Only a detection that reaches
    into ``bounds`` can hold: one wholly outside it is under a strip either
    way, and clamping to it would undo the fit while protecting nothing. A
    hold is applied to a box that is already sound, never to rescue a
    degenerate one.

    A ``TEXT_COLUMN`` the ink snap grew onto a blot abutting the type holds
    the fit at the blot, and :func:`_shrink_rects_for_detections` would pin
    the strip there regardless: a strip never covers a detection. The
    measured effect of the fit on a real volume was taken with that
    pull-back in place.

    The result is refused when the text box is malformed or outside the
    page's pixel frame, when the fitted box is degenerate or narrower than
    ``MIN_TEXT_WIDTH_FRACTION`` of the page, or when it keeps less than
    ``MARGIN_MIN_KEEP_RATIO`` of ``bounds`` (see that constant for the
    measurement behind the floor): that is what a partial read of the page
    looks like, and today's answer is the floor.

    :param bounds: ``(left, top, right, bottom)`` as tightened so far.
    :param page: The page carrying ``text_box`` and the detections.
    :param buffer: Slack in PDF points to leave around the text box.
    :returns: The fitted ``(left, top, right, bottom)``, or ``bounds``.
    """
    if page.text_box is None:
        return bounds
    try:
        text = BBox(*(float(v) for v in page.text_box))
    except (TypeError, ValueError):
        # A caller's bug on one page must not cost the volume its margins.
        logger.warning("Page %d: text box %r is malformed; ignoring it", page.index, page.text_box)
        return bounds
    # A box beyond the page's pixels was measured in some other frame (a
    # different render resolution); only the too-large case is visible.
    slack_x, slack_y = page.img_width * 0.01, page.img_height * 0.01
    if (
        text.x2 <= text.x1
        or text.y2 <= text.y1
        or text.x1 < -slack_x
        or text.y1 < -slack_y
        or text.x2 > page.img_width + slack_x
        or text.y2 > page.img_height + slack_y
    ):
        logger.debug(
            "Page %d: text box %s is inverted or outside the %dx%d page frame; ignoring it",
            page.index,
            page.text_box,
            page.img_width,
            page.img_height,
        )
        return bounds

    left, top, right, bottom = bounds
    sx, sy = page.scale_x, page.scale_y
    tb = text.to_pdf(sx, sy)
    new_left = max(left, tb.x1 - buffer)
    new_top = max(top, tb.y1 - buffer)
    new_right = min(right, tb.x2 + buffer)
    new_bottom = min(bottom, tb.y2 + buffer)
    if new_right <= new_left or new_bottom <= new_top:
        logger.debug("Page %d: text box lies outside the content box; ignoring it", page.index)
        return bounds

    for d in page.detections:
        if d.label not in HOLD_LABELS:
            continue
        if d.label in HEADER_LABELS and _is_edge_bleed(page, d):
            continue
        box = d.bbox.to_pdf(sx, sy)
        if box.x2 - box.x1 <= 1 or box.y2 - box.y1 <= 1:
            # A placeholder, not a measurement.
            continue
        if not (box.x1 < right and box.x2 > left and box.y1 < bottom and box.y2 > top):
            continue
        held = (
            min(new_left, max(left, box.x1)),
            min(new_top, max(top, box.y1)),
            max(new_right, min(right, box.x2)),
            max(new_bottom, min(bottom, box.y2)),
        )
        if held != (new_left, new_top, new_right, new_bottom):
            logger.debug(
                "Page %d: %s detection holds the text box from %s to %s",
                page.index,
                d.label.name,
                (new_left, new_top, new_right, new_bottom),
                held,
            )
            new_left, new_top, new_right, new_bottom = held

    if new_right - new_left < page.pdf_width * MIN_TEXT_WIDTH_FRACTION:
        logger.debug(
            "Page %d: text box is narrower than %.0f%% of the page; keeping the measured box",
            page.index,
            MIN_TEXT_WIDTH_FRACTION * 100,
        )
        return bounds
    kept = (new_right - new_left) * (new_bottom - new_top) / ((right - left) * (bottom - top))
    if kept < MARGIN_MIN_KEEP_RATIO:
        logger.debug(
            "Page %d: text box keeps %.0f%% of the tightened content box, below the %.0f%% "
            "floor; keeping the measured box",
            page.index,
            kept * 100,
            MARGIN_MIN_KEEP_RATIO * 100,
        )
        return bounds
    return new_left, new_top, new_right, new_bottom


def _tighten_bounds(
    bounds: tuple[float, float, float, float],
    page: Page,
    fitz_page: fitz.Page | None,
    buffer: float,
) -> tuple[float, float, float, float]:
    """Intersect measured content bounds with what detections support.

    Ink is the union of every mark on the page, so it only ever errs
    outward; detections describe content, so they only err inward. Taking
    the tighter of the two per side means a bound moves in only when both
    signals agree there is nothing there.

    "Both signals agree" has to be checked rather than assumed. A page whose
    second column went undetected has a band narrower than its own text, and
    tightening to it puts a strip through the type: seen on real pages, where
    the running head was then the widest horizontal detection. So a side is
    tightened only when the ink it would give up reads as an artifact rather
    than as text (see :func:`_ink_is_artifact_like`).

    A degenerate band result falls back to ``bounds`` (a bogus
    ``TEXT_COLUMN`` box should not be able to collapse the content box).

    The caller's own ``text_box``, when the page carries one, is applied
    last, to whichever box survived (see :func:`_fit_to_text_box`). It is
    the signal that reaches ink the band cannot: a blot down the page edge
    reads as text to the check above, and only a measurement of where the
    text actually is can give it up.

    :param bounds: ``(left, top, right, bottom)`` from text or ink.
    :param page: The page whose detections and text box to read.
    :param fitz_page: The PDF page, for the ink check. Without it the side
        bounds are left alone, since the check cannot be made.
    :param buffer: Slack in PDF points to leave around the text box.
    :returns: The tightened ``(left, top, right, bottom)``.
    """
    left, top, right, bottom = bounds
    band_left, band_right, header_top = _detection_bounds(page)

    def givable(x0: float, x1: float) -> bool:
        """Is the ink between two x positions safe to hand to a strip?"""
        if fitz_page is None:
            return False
        return _ink_is_artifact_like(fitz_page, x0, x1, top, bottom)

    if band_left is not None and band_left > left and givable(left, band_left):
        left = band_left
    if band_right is not None and band_right < right and givable(band_right, right):
        right = band_right
    # The top bound is not gated the same way: the ink above a header row is
    # bleed-through by construction (see EDGE_BLEED_PT, HEADER_MAX_FRACTION),
    # which is exactly what a top strip is for.
    if header_top is not None:
        top = max(top, header_top)
    tightened = (left, top, right, bottom)
    if right - left < page.pdf_width * MIN_TEXT_WIDTH_FRACTION or bottom <= top:
        tightened = bounds
    return _fit_to_text_box(tightened, page, buffer)


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
        # A 1x1 box is a placeholder for a missing measurement, not content
        # at the page corner, and it would collapse the top and left strips.
        if d.label not in NO_PUSHBACK_LABELS and d.bbox.width > 1 and d.bbox.height > 1
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


def _page_size_agrees(page: Page, width: float, height: float) -> bool:
    """Does a detected page describe the PDF page it is paired with?

    :param page: The detected page, carrying the size detection ran against.
    :param width: The PDF page's width in points.
    :param height: The PDF page's height in points.
    :returns: True when the two agree to within a point.
    """
    return abs(page.pdf_width - width) <= 1.0 and abs(page.pdf_height - height) <= 1.0


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
    :param pages: Optional detected pages, used to tighten the bounds and
        to pull a strip back off anything real it would cover. Without them
        the bounds come from the page's text or marks alone, which is also
        what happens on a page carrying no ``TEXT_COLUMN`` detection. The
        caller owns which detections are in each page: pass the ones a
        reviewer has kept, not everything the model proposed. A page that
        also carries a ``text_box`` (the caller's own measurement of its
        printed text, in the page's pixels) has its content box fitted to
        it, smaller only, never across a header-row or ``TEXT_COLUMN``
        detection, and never below ``MARGIN_MIN_KEEP_RATIO`` of the box;
        a page with none answers as it would without.
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
            if detected is not None and not _page_size_agrees(detected, pw, ph):
                # Every detection-derived bound is in the caller's frame. If
                # that frame is not this page's, a strip computed from it
                # lands somewhere arbitrary, so use the marks alone.
                logger.warning(
                    "Page %d: detections describe a %.0fx%.0f page but the PDF "
                    "page is %.0fx%.0f; ignoring them and its text box for margins",
                    page_idx,
                    detected.pdf_width,
                    detected.pdf_height,
                    pw,
                    ph,
                )
                detected = None
            if detected is not None:
                bounds = _tighten_bounds(bounds, detected, page, buffer)
            entry["rects"] = _rects_for_bounds(bounds, pw, ph, buffer)
            if detected is not None:
                _shrink_rects_for_detections(detected, entry["rects"])
                # The shrink can collapse a strip it pulled back, and a
                # zero-width rect is no use to a consumer drawing overlays.
                entry["rects"] = [
                    r for r in entry["rects"] if r["x1"] - r["x0"] > 1 and r["y1"] - r["y0"] > 1
                ]
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
            try:
                doc.save(str(tmp_path), garbage=4, deflate=True)
            except Exception:
                # Otherwise a failed save leaves the temp beside the file it
                # was meant to replace.
                tmp_path.unlink(missing_ok=True)
                raise
        else:
            doc.save(str(output_path), garbage=4, deflate=True)
            tmp_path = None

    if tmp_path is not None:
        tmp_path.replace(pdf_path)

    logger.info("Margin cleanup: %d/%d pages cleaned", cleaned, total)
    return output_path
