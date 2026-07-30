"""Measure where the marks are on a page, without a text layer.

Most of the geometry in this library asks ``page.get_text("words")`` or
``page.get_text("blocks")`` where the printed text sits. That only works
on a PDF that carries a text layer, which for a scanned volume means
``api.ocr()`` (ocrmypdf/Tesseract) has run over the whole file first. That
pass is expensive enough that callers want to skip it, and on a bitonal
PDF without it the text-bound helpers return nothing, which silently
collapses headnote redaction rects and disables margin cleanup.

A scanned page is black marks on white, so the same question can be
answered from the pixels: render the page, threshold it, and project the
dark pixels onto each axis. Nothing here reads text, so it works on any
PDF, with or without a text layer.

Used by :mod:`blackletter.margins` (the page content box) and by
:mod:`blackletter.process` (redaction rect bounds).
"""

from __future__ import annotations

import math
from pathlib import Path

import fitz
import numpy as np

# Render resolution, deliberately half the 200 dpi the bitonal images
# these measurements are applied to are produced at.
#
# Matching the image's own resolution was tried and measured worse: at 200
# dpi the projection resolves the real gaps between characters and words,
# so a growing edge stops at the first one and rects come out smaller --
# uncovered ink around headnote rects went from 8.6 to 21.8 pt^2 over 40
# pages of a real volume, and the tighter content box let a margin strip
# reach a STATE_ABBREVIATION detection. Downsampling averages 2x2 blocks
# of the bitonal, so ink blooms by up to one pixel; that bloom errs toward
# larger content boxes, further growth and shyer margins, which is the
# safe direction for every consumer here.
DPI = 100

# Grayscale value below which a pixel counts as a mark (0 = black).
DARK_LEVEL = 200

# Cache attribute set on the parent fitz.Document. Holds one page's mask:
# callers walk pages in order, so a single slot avoids re-rendering a page
# for each rect on it without holding a mask per page (a 600-page book at
# ~0.9 MB per mask would not fit).
_CACHE_ATTR = "_blackletter_ink_cache"


def page_mask(fitz_page: fitz.Page, dpi: int = DPI) -> tuple[np.ndarray, float, float]:
    """Return a page's ink mask and its pixel-to-point scale factors.

    :param fitz_page: The PyMuPDF page to render.
    :param dpi: Render resolution.
    :return: ``(mask, sx, sy)`` where ``mask[y, x]`` is True for dark
        pixels, and ``sx``/``sy`` convert pixels to PDF points.
    """
    doc = fitz_page.parent
    cached = getattr(doc, _CACHE_ATTR, None)
    if cached is not None and cached[0] == fitz_page.number and cached[1] == dpi:
        return cached[2], cached[3], cached[4]

    pix = fitz_page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
    gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)[
        :, : pix.width
    ]
    mask = gray < DARK_LEVEL
    sx = fitz_page.rect.width / pix.width if pix.width else 1.0
    sy = fitz_page.rect.height / pix.height if pix.height else 1.0
    setattr(doc, _CACHE_ATTR, (fitz_page.number, dpi, mask, sx, sy))
    return mask, sx, sy


def content_span(
    profile: np.ndarray, min_count: int, max_count: int, max_gap: int
) -> tuple[int, int] | None:
    """Find the largest run of content indices in an ink profile.

    :param profile: Per-row (or per-column) count of dark pixels.
    :param min_count: Counts below this are blank (speckle, dust).
    :param max_count: Counts above this are a solid bar, not text.
    :param max_gap: Blank runs no longer than this are bridged.
    :return: Inclusive ``(start, end)`` index pair, or None if the
        profile holds no content at all.
    """
    active = (profile >= min_count) & (profile <= max_count)
    if not active.any():
        return None

    indices = np.flatnonzero(active)
    # Split where consecutive content indices are more than max_gap
    # apart, then keep the widest resulting run.
    breaks = np.flatnonzero(np.diff(indices) > max_gap + 1)
    starts = np.concatenate(([indices[0]], indices[breaks + 1]))
    ends = np.concatenate((indices[breaks], [indices[-1]]))
    widest = int(np.argmax(ends - starts))
    return int(starts[widest]), int(ends[widest])


# Only treat a page's ink as a content box when it spans at least this
# fraction of the page width. Narrower means an image page or appendix,
# where the bounds are not a reliable boundary. Matches
# ``margins.MIN_TEXT_WIDTH_FRACTION``.
MIN_CONTENT_WIDTH_FRACTION = 0.40

# A row/column needs at least this fraction of dark pixels to count as
# content (drops speckle and dust) and at most the max (above which it is
# a solid bar: platen edge, gutter shadow, fold).
CONTENT_MIN_FRACTION = 0.004
CONTENT_MAX_FRACTION = 0.60

# Runs of content separated by less than this fraction of the page
# dimension are one block, so inter-line and header-to-body white space
# does not split a page into pieces.
CONTENT_GAP_FRACTION = 0.06

# Thresholds for measuring ink inside a single rect. Looser than the page
# thresholds: the region is already known to hold text, so only isolated
# speckle (min) and near-solid rules (max) are excluded.
BBOX_MIN_FRACTION = 0.004
BBOX_MAX_FRACTION = 0.9

_CONTENT_BOX_ATTR = "_blackletter_content_box_cache"


def content_box(fitz_page: fitz.Page) -> tuple[float, float, float, float] | None:
    """Return the page's printed-content box, measured from ink.

    Scanner artifacts along the page edges are the thing this has to
    exclude, since they are what margin cleanup exists to remove. Two
    rules do it: a row or column that is *mostly* dark is a platen bar or
    gutter shadow rather than a line of text, and content is the largest
    run of inked rows (columns) after bridging small gaps, so an artifact
    separated from the text block by white space falls outside the box.

    Columns are measured over the content rows only. A full-width bar at
    the top of the page puts ink in every column, which would otherwise
    stretch the box across the whole page even though the vertical span
    already excluded the bar.

    Cached per page on the parent document.

    :param fitz_page: The page to measure.
    :return: ``(left, top, right, bottom)`` in PDF points, or None when
        the page has no usable ink or the ink is too narrow to trust.
    """
    doc = fitz_page.parent
    cached = getattr(doc, _CONTENT_BOX_ATTR, None)
    if cached is not None and cached[0] == fitz_page.number:
        return cached[1]

    box = _measure_content_box(fitz_page)
    setattr(doc, _CONTENT_BOX_ATTR, (fitz_page.number, box))
    return box


def _measure_content_box(
    fitz_page: fitz.Page,
) -> tuple[float, float, float, float] | None:
    """Uncached body of :func:`content_box`."""
    ink, sx, sy = page_mask(fitz_page)
    if not ink.size:
        return None
    img_height, img_width = ink.shape

    vertical = content_span(
        ink.sum(axis=1),
        min_count=max(2, int(img_width * CONTENT_MIN_FRACTION)),
        max_count=int(img_width * CONTENT_MAX_FRACTION),
        max_gap=int(img_height * CONTENT_GAP_FRACTION),
    )
    if vertical is None:
        return None

    band = ink[vertical[0] : vertical[1] + 1, :]
    band_height = band.shape[0]
    horizontal = content_span(
        band.sum(axis=0),
        min_count=max(2, int(band_height * CONTENT_MIN_FRACTION)),
        max_count=int(band_height * CONTENT_MAX_FRACTION),
        max_gap=int(img_width * CONTENT_GAP_FRACTION),
    )
    if horizontal is None:
        return None

    left = horizontal[0] * sx
    right = (horizontal[1] + 1) * sx
    top = vertical[0] * sy
    bottom = (vertical[1] + 1) * sy
    if (right - left) < fitz_page.rect.width * MIN_CONTENT_WIDTH_FRACTION:
        return None
    return left, top, right, bottom


def content_clip(fitz_page: fitz.Page, clip: fitz.Rect) -> fitz.Rect | None:
    """Restrict a region to the page's printed-content box.

    Everything measured from ink goes through here, so ink that is not
    printed text (the platen band along a page edge, which sits tens of
    points below the last line) cannot stretch a measurement to the page
    edge.

    :param fitz_page: The page whose content box to use.
    :param clip: Region of interest, in PDF points.
    :return: The intersection, or None when the region lies wholly outside
        the content box (in which case the ink cannot say anything about
        it and callers should not shrink anything).
    """
    box = content_box(fitz_page)
    if box is None:
        return fitz.Rect(clip)
    region = fitz.Rect(
        max(clip.x0, box[0]),
        max(clip.y0, box[1]),
        min(clip.x1, box[2]),
        min(clip.y1, box[3]),
    )
    if region.x1 <= region.x0 or region.y1 <= region.y0:
        return None
    return region


def ink_bbox(
    fitz_page: fitz.Page,
    clip: fitz.Rect,
) -> tuple[float, float, float, float] | None:
    """Return the tightest box around the ink inside ``clip``.

    The ink equivalent of "where are the words in this rect", measured
    within :func:`content_clip`.

    :param fitz_page: The page to measure.
    :param clip: Region of interest, in PDF points.
    :return: ``(x0, y0, x1, y1)`` in PDF points, or None when the region
        holds no ink (or lies outside the content box).
    """
    region = content_clip(fitz_page, clip)
    if region is None:
        return None

    mask, sx, sy = page_mask(fitz_page)
    height, width = mask.shape
    x0, y0, x1, y1 = region.x0, region.y0, region.x1, region.y1

    c0 = max(0, min(int(x0 / sx), width))
    c1 = max(c0, min(int(round(x1 / sx)), width))
    r0 = max(0, min(int(y0 / sy), height))
    r1 = max(r0, min(int(round(y1 / sy)), height))
    window = mask[r0:r1, c0:c1]
    if not window.size:
        return None
    win_height, win_width = window.shape

    row_counts = window.sum(axis=1)
    rows = np.flatnonzero(
        (row_counts >= max(2, int(win_width * BBOX_MIN_FRACTION)))
        & (row_counts <= win_width * BBOX_MAX_FRACTION)
    )
    if not rows.size:
        return None
    band = window[rows[0] : rows[-1] + 1, :]
    col_counts = band.sum(axis=0)
    cols = np.flatnonzero(
        (col_counts >= max(1, int(win_height * BBOX_MIN_FRACTION)))
        & (col_counts <= band.shape[0] * BBOX_MAX_FRACTION)
    )
    if not cols.size:
        return None
    return (
        (c0 + int(cols[0])) * sx,
        (r0 + int(rows[0])) * sy,
        (c0 + int(cols[-1]) + 1) * sx,
        (r0 + int(rows[-1]) + 1) * sy,
    )


# How far a rect may grow outward over ink that continues past its edge,
# in PDF points. Horizontally this has to clear a clipped first or last
# character (and column bounds can be off by ~10 pt); vertically a
# descender or a single cut line is enough.
GROW_MARGIN_X = 20.0
GROW_MARGIN_Y = 8.0

# Growing edges cross no white space at all: the 100 dpi downsample already
# blurs the gaps between characters shut, so bridging is unnecessary, and
# blank runs inside a column (median 1.1 pt, max 6.5 pt on a real volume)
# overlap the gutters between columns (0.0-7.2 pt) so completely that any
# bridge wide enough to be useful also crosses a narrow gutter. Tried at
# 1 pt: it ate the single blank pixel that keeps a rect out of the next
# column.


def _outside_before(edge: float, scale: float) -> int:
    """Last pixel line lying wholly *before* a rect's low edge.

    Pixel line ``c`` covers ``[c * scale, (c + 1) * scale)``, so the last
    line entirely before ``edge`` is ``floor(edge / scale) - 1``. Using the
    line that straddles the edge instead would read the rect's own ink and
    creep outward on every call; using one further out would skip the line
    that holds a one-pixel gutter, and the rect would grow across it into
    the neighbouring column.

    :param edge: The rect's low edge (x0 or y0), in PDF points.
    :param scale: Points per pixel line along that axis.
    :return: The pixel index.
    """
    return int(math.floor(edge / scale)) - 1


def _outside_after(edge: float, scale: float) -> int:
    """First pixel line lying wholly *after* a rect's high edge.

    :param edge: The rect's high edge (x1 or y1), in PDF points.
    :param scale: Points per pixel line along that axis.
    :return: The pixel index.
    """
    return int(math.ceil(edge / scale))


def _grow_edge(
    counts: np.ndarray,
    start: int,
    step: int,
    max_steps: int,
    min_count: int,
) -> int:
    """Walk outward from ``start`` while lines keep carrying ink.

    :param counts: Per-line ink counts (rows or columns).
    :param start: First line lying wholly outside the rect.
    :param step: ``-1`` to walk up/left, ``+1`` to walk down/right.
    :param max_steps: Cap on how far to walk.
    :param min_count: Ink a line needs to count as a continuation.
    :return: Number of lines to grow by.
    """
    grown = 0
    idx = start
    while grown < max_steps and 0 <= idx < counts.size:
        if counts[idx] < min_count:
            break
        grown += 1
        idx += step
    return grown


def grow_to_ink(
    fitz_page: fitz.Page,
    rect: fitz.Rect,
    margin_x: float = GROW_MARGIN_X,
    margin_y: float = GROW_MARGIN_Y,
) -> fitz.Rect:
    """Expand a rect over ink that runs past its edges.

    A redaction rect derived from detection geometry can cut through a
    glyph: the first character of every line when a ``TEXT_COLUMN`` box
    starts a few points inside the printed text, or the tail of the line a
    block boundary lands on. The text-layer code path grew rects for free,
    because ``_words_in_rect`` returns every word *overlapping* a rect and
    the resulting bounds could fall outside it. Ink measured strictly
    inside a rect has no such slack, so it is added back here.

    Growth stops at the first blank row/column, so it takes in the rest of
    a clipped character or line and nothing more, and never leaves the
    page's content box (which keeps it out of the platen bands along the
    page edges).

    :param fitz_page: The page to measure.
    :param rect: Rect to grow, in PDF points.
    :param margin_x: Maximum growth per side, horizontally.
    :param margin_y: Maximum growth per side, vertically.
    :return: The grown rect, or ``rect`` unchanged when there is nothing
        to grow onto.
    """
    region = content_clip(fitz_page, rect)
    if region is None:
        return fitz.Rect(rect)

    mask, sx, sy = page_mask(fitz_page)
    height, width = mask.shape
    box = content_box(fitz_page)
    lim_x0, lim_y0, lim_x1, lim_y1 = box if box else (0.0, 0.0, width * sx, height * sy)

    c0 = max(0, min(int(region.x0 / sx), width - 1))
    c1 = max(c0 + 1, min(int(round(region.x1 / sx)), width))
    r0 = max(0, min(int(region.y0 / sy), height - 1))
    r1 = max(r0 + 1, min(int(round(region.y1 / sy)), height))

    # Columns first, over the rect's own rows, then rows over the widened
    # span, so a clipped character contributes to both.
    col_counts = mask[r0:r1, :].sum(axis=0)
    min_cols = max(1, int((r1 - r0) * 0.01))
    # Each walk starts at the first pixel column lying wholly outside the
    # rect: see _outside_before for why neither neighbour of that column
    # will do.
    out_left = _outside_before(region.x0, sx)
    out_right = _outside_after(region.x1, sx)
    left = _grow_edge(
        col_counts,
        out_left,
        -1,
        min(int(margin_x / sx), max(0, out_left - int(lim_x0 / sx) + 1)),
        min_cols,
    )
    right = _grow_edge(
        col_counts,
        out_right,
        1,
        min(int(margin_x / sx), max(0, int(lim_x1 / sx) - out_right + 1)),
        min_cols,
    )
    c0 = min(c0, out_left - left + 1)
    c1 = max(c1, out_right + right)

    row_counts = mask[:, c0:c1].sum(axis=1)
    min_rows = max(1, int((c1 - c0) * 0.01))
    out_top = _outside_before(region.y0, sy)
    out_bottom = _outside_after(region.y1, sy)
    top = _grow_edge(
        row_counts,
        out_top,
        -1,
        min(int(margin_y / sy), max(0, out_top - int(lim_y0 / sy) + 1)),
        min_rows,
    )
    bottom = _grow_edge(
        row_counts,
        out_bottom,
        1,
        min(int(margin_y / sy), max(0, int(lim_y1 / sy) - out_bottom + 1)),
        min_rows,
    )

    return fitz.Rect(
        min(rect.x0, (out_left - left + 1) * sx),
        min(rect.y0, (out_top - top + 1) * sy),
        max(rect.x1, (out_right + right) * sx),
        max(rect.y1, (out_bottom + bottom) * sy),
    )


def has_text_layer(pdf_path: str | Path, sample_pages: int = 5) -> bool:
    """Report whether a PDF carries a usable text layer.

    Samples the first few pages rather than the whole document: a PDF
    either went through :func:`blackletter.api.ocr` and has a text layer
    on every page, or it is a bitonal scan and has one on none of them.

    :param pdf_path: PDF to inspect.
    :param sample_pages: How many leading pages to sample.
    :return: True if any sampled page yields text.
    """
    try:
        with fitz.open(str(pdf_path)) as doc:
            for page_idx in range(min(sample_pages, doc.page_count)):
                if doc[page_idx].get_text("text").strip():
                    return True
    except Exception:
        return False
    return False
