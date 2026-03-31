"""Tests for _outside_opinion_rects masking logic.

Verifies that whiteout masks align with actual column boundaries from
TEXT_COLUMN detections and do not bleed into adjacent columns.
"""

from blackletter.models import BBox, Detection, Label, Page
from blackletter.scanner import _outside_opinion_rects, _column_bounds_pdf


def _make_page(
    img_width=1207,
    img_height=1938,
    pdf_width=612.0,
    pdf_height=792.0,
    left_tc=(78, 164, 597, 1811),
    right_tc=(620, 168, 1138, 1803),
    extra_dets=None,
):
    """Build a Page with TEXT_COLUMN detections and optional extras."""
    dets = []
    if left_tc:
        dets.append(
            Detection(
                bbox=BBox(*left_tc),
                label=Label.TEXT_COLUMN,
                confidence=0.95,
                page_index=0,
            )
        )
    if right_tc:
        dets.append(
            Detection(
                bbox=BBox(*right_tc),
                label=Label.TEXT_COLUMN,
                confidence=0.95,
                page_index=0,
            )
        )
    if extra_dets:
        dets.extend(extra_dets)
    return Page(
        index=0,
        pdf_width=pdf_width,
        pdf_height=pdf_height,
        img_width=img_width,
        img_height=img_height,
        detections=dets,
    )


def _make_det(x0, y0, x1, y1, label=Label.CASE_CAPTION, page_index=0):
    return Detection(
        bbox=BBox(x0, y0, x1, y1),
        label=label,
        confidence=0.95,
        page_index=page_index,
    )


# ── _column_bounds_pdf ──────────────────────────────────────────────────


class TestColumnBoundsPdf:
    def test_uses_text_column_detections(self):
        page = _make_page()
        sx = page.scale_x  # 612 / 1207 ≈ 0.507

        lx0, lx1, rx0, rx1, boundary = _column_bounds_pdf(page)

        # Left column: 78→597 in pixels → PDF
        assert abs(lx0 - 78 * sx) < 0.1
        assert abs(lx1 - 597 * sx) < 0.1
        # Right column: 620→1138 in pixels → PDF
        assert abs(rx0 - 620 * sx) < 0.1
        assert abs(rx1 - 1138 * sx) < 0.1
        # Boundary should be between left_x1 and right_x0
        assert lx1 < boundary < rx0

    def test_returns_none_without_text_columns(self):
        page = _make_page(left_tc=None, right_tc=None)
        result = _column_bounds_pdf(page)
        assert result is None


# ── _outside_opinion_rects ──────────────────────────────────────────────


class TestOutsideOpinionRects:
    """Test masking rects for partial pages at opinion boundaries."""

    def _page_and_pdf_width(self):
        page = _make_page()
        return page, page.pdf_width

    def test_first_page_caption_left_masks_left_col_only(self):
        """Caption in LEFT column: mask left column above caption."""
        page, w = self._page_and_pdf_width()
        sx, sy = page.scale_x, page.scale_y

        # Caption at left column, partway down
        caption = _make_det(100, 500, 580, 800)
        key = _make_det(100, 800, 580, 850, label=Label.KEY_ICON)

        rects = _outside_opinion_rects(page, w, caption, key, is_first=True, is_last=False)

        assert len(rects) == 1
        r = rects[0]
        # Should cover left column width (from TEXT_COLUMN bounds)
        left_x1_pdf = 597 * sx
        assert abs(r.x1 - left_x1_pdf) < 1.0, (
            f"mask x1={r.x1:.1f} should be near left col edge {left_x1_pdf:.1f}"
        )
        # Should NOT extend into right column
        right_x0_pdf = 620 * sx
        assert r.x1 < right_x0_pdf, "mask should not reach into right column"
        # Bottom should be at caption top
        cap_y1_pdf = 500 * sy
        assert abs(r.y1 - cap_y1_pdf) < 1.0

    def test_first_page_caption_right_masks_entire_left_and_right_above(self):
        """Caption in RIGHT column: mask entire left col + right above caption."""
        page, w = self._page_and_pdf_width()
        sx, sy = page.scale_x, page.scale_y

        caption = _make_det(650, 500, 1100, 800)
        key = _make_det(650, 800, 1100, 850, label=Label.KEY_ICON)

        rects = _outside_opinion_rects(page, w, caption, key, is_first=True, is_last=False)

        assert len(rects) == 2
        # Sort by x0 to identify left and right rects
        rects.sort(key=lambda r: r.x0)
        left_rect, right_rect = rects

        # Left rect should cover left column
        left_x0_pdf = 78 * sx
        left_x1_pdf = 597 * sx
        assert abs(left_rect.x0 - left_x0_pdf) < 1.0
        assert abs(left_rect.x1 - left_x1_pdf) < 1.0

        # Right rect should cover right column above caption
        right_x0_pdf = 620 * sx
        right_x1_pdf = 1138 * sx
        assert abs(right_rect.x0 - right_x0_pdf) < 1.0
        assert abs(right_rect.x1 - right_x1_pdf) < 1.0
        cap_y1_pdf = 500 * sy
        assert abs(right_rect.y1 - cap_y1_pdf) < 1.0

    def test_last_page_key_right_masks_right_col_below(self):
        """Key in RIGHT column: mask right column below key."""
        page, w = self._page_and_pdf_width()
        sx, sy = page.scale_x, page.scale_y

        caption = _make_det(100, 100, 580, 300)
        key = _make_det(650, 300, 1100, 400, label=Label.KEY_ICON)

        rects = _outside_opinion_rects(page, w, caption, key, is_first=False, is_last=True)

        assert len(rects) == 1
        r = rects[0]
        # Should cover right column width
        right_x0_pdf = 620 * sx
        right_x1_pdf = 1138 * sx
        assert abs(r.x0 - right_x0_pdf) < 1.0
        assert abs(r.x1 - right_x1_pdf) < 1.0
        # Top should be at key bottom
        key_y2_pdf = 400 * sy
        assert abs(r.y0 - key_y2_pdf) < 1.0

    def test_last_page_key_left_masks_left_below_and_entire_right(self):
        """Key in LEFT column: mask left below key + entire right column."""
        page, w = self._page_and_pdf_width()
        sx, sy = page.scale_x, page.scale_y

        caption = _make_det(100, 100, 580, 300)
        key = _make_det(100, 300, 580, 400, label=Label.KEY_ICON)

        rects = _outside_opinion_rects(page, w, caption, key, is_first=False, is_last=True)

        assert len(rects) == 2
        rects.sort(key=lambda r: r.x0)
        left_rect, right_rect = rects

        # Left: below key
        left_x0_pdf = 78 * sx
        left_x1_pdf = 597 * sx
        assert abs(left_rect.x0 - left_x0_pdf) < 1.0
        assert abs(left_rect.x1 - left_x1_pdf) < 1.0
        key_y2_pdf = 400 * sy
        assert abs(left_rect.y0 - key_y2_pdf) < 1.0

        # Right: entire column
        right_x0_pdf = 620 * sx
        right_x1_pdf = 1138 * sx
        assert abs(right_rect.x0 - right_x0_pdf) < 1.0
        assert abs(right_rect.x1 - right_x1_pdf) < 1.0

    def test_masks_never_extend_into_other_column(self):
        """No mask rect should overlap with the other column's text area."""
        page, w = self._page_and_pdf_width()
        sx = page.scale_x
        left_x1_pdf = 597 * sx
        right_x0_pdf = 620 * sx

        # Test all four scenarios
        cases = [
            # (caption, key, is_first, is_last)
            (
                _make_det(100, 500, 580, 800),
                _make_det(100, 800, 580, 850, label=Label.KEY_ICON),
                True,
                False,
            ),
            (
                _make_det(650, 500, 1100, 800),
                _make_det(650, 800, 1100, 850, label=Label.KEY_ICON),
                True,
                False,
            ),
            (
                _make_det(100, 100, 580, 300),
                _make_det(650, 300, 1100, 400, label=Label.KEY_ICON),
                False,
                True,
            ),
            (
                _make_det(100, 100, 580, 300),
                _make_det(100, 300, 580, 400, label=Label.KEY_ICON),
                False,
                True,
            ),
        ]
        for caption, key, is_first, is_last in cases:
            rects = _outside_opinion_rects(page, w, caption, key, is_first, is_last)
            for r in rects:
                # Each rect should be entirely within one column
                is_left = r.x1 <= right_x0_pdf + 0.5
                is_right = r.x0 >= left_x1_pdf - 0.5
                assert is_left or is_right, (
                    f"Rect ({r.x0:.1f}, {r.y0:.1f}, {r.x1:.1f}, {r.y1:.1f}) "
                    f"spans both columns (left ends {left_x1_pdf:.1f}, right starts {right_x0_pdf:.1f})"
                )

    def test_single_page_opinion_masks_above_and_below(self):
        """Opinion on single page: mask above caption + below key."""
        page, w = self._page_and_pdf_width()

        caption = _make_det(100, 400, 580, 700)
        key = _make_det(650, 1200, 1100, 1300, label=Label.KEY_ICON)

        rects = _outside_opinion_rects(page, w, caption, key, is_first=True, is_last=True)

        # Should have rects for both first and last
        assert len(rects) >= 2

    def test_no_text_columns_returns_empty(self):
        """Without TEXT_COLUMN detections, skip masking (no rects)."""
        page = _make_page(left_tc=None, right_tc=None)
        w = page.pdf_width

        caption = _make_det(100, 500, 580, 800)
        key = _make_det(100, 800, 580, 850, label=Label.KEY_ICON)

        rects = _outside_opinion_rects(page, w, caption, key, is_first=True, is_last=False)
        assert rects == []

    def test_middle_page_produces_no_rects(self):
        """Middle pages (not first, not last) need no outside-opinion masks."""
        page, w = self._page_and_pdf_width()
        caption = _make_det(100, 100, 580, 300)
        key = _make_det(650, 300, 1100, 400, label=Label.KEY_ICON)

        rects = _outside_opinion_rects(page, w, caption, key, is_first=False, is_last=False)
        assert rects == []
