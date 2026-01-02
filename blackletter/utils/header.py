"""Header detection and processing utilities."""

import logging
from typing import Optional, Tuple, List, Dict

logger = logging.getLogger(__name__)


class HeaderProcessor:
    """Processes and detects header regions in PDFs."""

    @staticmethod
    def extract_header_words(page_pl, top_pts: float, gap_pts: float, y_tol: float) -> List[Dict]:
        """Extract and group words from header region.

        Args:
            page_pl: pdfplumber page object
            top_pts: How far down from top to search
            gap_pts: Maximum horizontal gap between chars to group as word
            y_tol: Y-tolerance for grouping chars into same line

        Returns:
            List of word dictionaries with position info
        """
        chars = [c for c in page_pl.chars if c.get("top", 1e9) < top_pts and c.get("text")]
        if not chars:
            return []

        chars.sort(key=lambda c: (c["top"], c["x0"]))

        # Group chars into lines by y-position
        lines = []
        for c in chars:
            if not lines:
                lines.append([c])
                continue

            ymid = (c["top"] + c["bottom"]) / 2
            last = lines[-1][0]
            last_ymid = (last["top"] + last["bottom"]) / 2

            if abs(ymid - last_ymid) <= y_tol:
                lines[-1].append(c)
            else:
                lines.append([c])

        # Group chars within lines into words
        word_objs = []
        for line in lines:
            line.sort(key=lambda c: c["x0"])
            cur = [line[0]]

            for c in line[1:]:
                prev = cur[-1]
                if (c["x0"] - prev["x1"]) > gap_pts:
                    word_objs.append(cur)
                    cur = [c]
                else:
                    cur.append(c)
            word_objs.append(cur)

        # Build word dictionaries
        words = []
        for wchars in word_objs:
            text = "".join(ch["text"] for ch in wchars).strip()
            if not text:
                continue

            words.append(
                {
                    "text": text,
                    "x0": min(ch["x0"] for ch in wchars),
                    "x1": max(ch["x1"] for ch in wchars),
                    "top": min(ch["top"] for ch in wchars),
                    "bottom": max(ch["bottom"] for ch in wchars),
                }
            )

        return sorted(words, key=lambda w: (w["top"], w["x0"]))

    @staticmethod
    def redaction_bbox_for_header(
        page_pl,
        top_pts: float = 40.0,
        gap_pts: float = 2.0,
        y_tol: float = 3.0,
        margin_pts: float = 120.0,
        pad_x: float = 2.0,
        pad_y: float = 1.0,
    ) -> Optional[Tuple[float, float, float, float]]:
        """Calculate header redaction bbox, preserving page numbers at margins.

        Detects header content and returns a redaction box, but preserves
        page numbers that appear near the left/right margins.

        Args:
            page_pl: pdfplumber page object
            top_pts: How far down to search for header content
            gap_pts: Maximum gap between characters
            y_tol: Y-tolerance for grouping text
            margin_pts: Distance from edge where page numbers are expected
            pad_x: Horizontal padding around detected text
            pad_y: Vertical padding around detected text

        Returns:
            (x0, y0, x1, y1) redaction box in points, or None
        """
        words = HeaderProcessor.extract_header_words(page_pl, top_pts, gap_pts, y_tol)
        if not words:
            return None

        # Find numeric word closest to margin (likely page number)
        best = None
        for i, w in enumerate(words):
            if not w["text"].isdigit():
                continue

            d_left = w["x0"]
            d_right = page_pl.width - w["x1"]
            edge_dist = min(d_left, d_right)

            if edge_dist <= margin_pts:
                cand = (edge_dist, i)
                if best is None or cand < best:
                    best = cand

        if best is None:
            return (31.0, 28, 361, 41)

        keep_i = best[1]
        others = [w for i, w in enumerate(words) if i != keep_i]

        if not others:
            return None

        # Union bbox of non-page-number words
        x0 = min(w["x0"] for w in others) - pad_x
        y0 = min(w["top"] for w in others) - pad_y
        x1 = max(w["x1"] for w in others) + pad_x
        y1 = max(w["bottom"] for w in others) + pad_y

        # Clamp to page bounds
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(page_pl.width, x1)
        y1 = min(page_pl.height, y1)

        if x1 <= x0 or y1 <= y0:
            return None

        return (x0, y0, x1, y1)
