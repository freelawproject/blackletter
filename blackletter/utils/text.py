"""Text extraction and redaction utilities."""

import logging
from typing import List, Dict, Tuple
from blackletter.config import RedactionConfig

import fitz

logger = logging.getLogger(__name__)

config = RedactionConfig()


def redact_text_lines_in_window(
    page_pl,
    page_fitz,
    win_pdf: Tuple[float, float, float, float],
    pad: float = 1.5,
    y_tol: float = 3.0,
    merge_gap: float = 2.5,
    min_h: float = 6.0,
):
    """Redact text lines within a window using pdfplumber.

    Args:
        page_pl: pdfplumber Page object
        page_fitz: fitz Page object (same page)
        win_pdf: (x0, y0, x1, y1) in PDF points
        pad: Padding around detected text
        y_tol: Y-tolerance for grouping words into lines
        merge_gap: Maximum gap for merging rectangles
        min_h: Minimum height for redaction
    """
    x0, y0, x1, y1 = win_pdf
    region = page_pl.crop((x0, y0, x1, y1))

    # Extract words from window
    words = region.extract_words(
        x_tolerance=1,
        y_tolerance=2,
        use_text_flow=False,
        keep_blank_chars=False,
    )

    if not words:
        return

    # Group words into lines
    line_groups = _cluster_words_into_lines(words, y_tol=y_tol)

    # Build redaction rectangles
    rects = []
    for line in line_groups:
        lx0 = min(w["x0"] for w in line) - pad
        ly0 = min(w["top"] for w in line) - pad
        lx1 = max(w["x1"] for w in line) + pad
        ly1 = max(w["bottom"] for w in line) + pad

        # Clamp to window bounds
        lx0 = max(x0, lx0)
        ly0 = max(y0, ly0)
        lx1 = min(x1, lx1)
        ly1 = min(y1, ly1)

        if (ly1 - ly0) >= min_h and (lx1 > lx0):
            rects.append((lx0, ly0, lx1, ly1))

    # Merge nearby rectangles
    rects = _merge_close_rects(rects, gap_tol=merge_gap)

    # Apply redactions
    for rx0, ry0, rx1, ry1 in rects:
        page_fitz.add_redact_annot(fitz.Rect(rx0, ry0, rx1, ry1), fill=config.redaction_fill)


def _cluster_words_into_lines(words: List[Dict], y_tol: float = 3.0) -> List[List[Dict]]:
    """Group words into horizontal lines using top coordinate.

    Args:
        words: List of word dicts from pdfplumber
        y_tol: Maximum vertical distance to group as same line

    Returns:
        List of word groups (each group is a line)
    """
    if not words:
        return []

    words = sorted(words, key=lambda w: (w["top"], w["x0"]))
    lines = []
    cur = [words[0]]
    cur_top = words[0]["top"]

    for w in words[1:]:
        if abs(w["top"] - cur_top) <= y_tol:
            cur.append(w)
        else:
            lines.append(cur)
            cur = [w]
            cur_top = w["top"]

    lines.append(cur)
    return lines


def _merge_close_rects(rects: List[Tuple], gap_tol: float = 2.5) -> List[Tuple]:
    """Merge vertically-close rectangles to reduce redaction count.

    Args:
        rects: List of (x0, y0, x1, y1) tuples
        gap_tol: Maximum vertical gap to merge rectangles

    Returns:
        Merged list of rectangles
    """
    if not rects:
        return rects

    rects = sorted(rects, key=lambda r: (r[1], r[0]))
    merged = [rects[0]]

    for r in rects[1:]:
        x0, y0, x1, y1 = r
        mx0, my0, mx1, my1 = merged[-1]

        # If rectangles almost touch vertically, merge them
        if y0 <= my1 + gap_tol:
            merged[-1] = (min(mx0, x0), my0, max(mx1, x1), max(my1, y1))
        else:
            merged.append(r)

    return merged
