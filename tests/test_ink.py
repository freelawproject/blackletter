"""Tests for ``blackletter.ink``.

These measurements stand in for the PDF text layer, which a caller that
skips ``api.ocr`` never has, so the cases that matter are the ones where
ink and text disagree: scanner artifacts along the page edges, which carry
ink but are not printed text.
"""

from __future__ import annotations

import fitz
import pytest

from blackletter.ink import content_box, grow_to_ink, has_text_layer, ink_bbox, page_mask
from tests.pdf_fixtures import (
    BOTTOM_BAR,
    COLUMN_LEFT,
    COLUMN_RIGHT,
    CONTENT,
    HEADER_LINE_Y,
    PAGE_H,
    PAGE_W,
    write_bitonal_page,
    write_text_page,
    write_two_column_page,
)

# The measured box lands within a few points of CONTENT: glyphs start
# below the box top, and the last line ends above its bottom.
SLACK = 6.0


class TestContentBox:
    @staticmethod
    def _box(pdf_path):
        with fitz.open(str(pdf_path)) as doc:
            return content_box(doc[0])

    def test_finds_the_text_block(self, tmp_path):
        pdf = tmp_path / "clean.pdf"
        write_bitonal_page(pdf)
        left, top, right, bottom = self._box(pdf)
        assert left == pytest.approx(CONTENT.x0, abs=SLACK)
        assert top == pytest.approx(CONTENT.y0, abs=SLACK)
        assert right == pytest.approx(CONTENT.x1, abs=SLACK)
        assert bottom == pytest.approx(CONTENT.y1, abs=SLACK)

    def test_excludes_edge_bars(self, tmp_path):
        """Platen bands at the page edges are not printed content."""
        pdf = tmp_path / "bars.pdf"
        write_bitonal_page(pdf, top_bar=True, bottom_bar=True)
        _left, top, _right, bottom = self._box(pdf)
        assert top > 12, "top bar counted as content"
        assert bottom < BOTTOM_BAR.y0, "bottom bar counted as content"

    def test_none_for_blank_page(self, tmp_path):
        pdf = tmp_path / "blank.pdf"
        with fitz.open() as doc:
            doc.new_page(width=PAGE_W, height=PAGE_H)
            doc.save(str(pdf))
        assert self._box(pdf) is None

    def test_none_for_narrow_content(self, tmp_path):
        pdf = tmp_path / "narrow.pdf"
        with fitz.open() as doc:
            page = doc.new_page(width=PAGE_W, height=PAGE_H)
            page.insert_text((280, 400), "12", fontsize=9)
            doc.save(str(pdf))
        assert self._box(pdf) is None

    def test_cached_per_page(self, tmp_path):
        pdf = tmp_path / "clean.pdf"
        write_bitonal_page(pdf)
        with fitz.open(str(pdf)) as doc:
            first = content_box(doc[0])
            with_cache = content_box(doc[0])
        assert first is with_cache


class TestInkBbox:
    def test_measures_the_text_in_a_clip(self, tmp_path):
        pdf = tmp_path / "clean.pdf"
        write_bitonal_page(pdf)
        with fitz.open(str(pdf)) as doc:
            box = ink_bbox(doc[0], fitz.Rect(0, 0, PAGE_W, PAGE_H))
        assert box is not None
        left, top, right, bottom = box
        assert left == pytest.approx(CONTENT.x0, abs=SLACK)
        assert top == pytest.approx(CONTENT.y0, abs=SLACK)
        assert right == pytest.approx(CONTENT.x1, abs=SLACK)
        assert bottom == pytest.approx(CONTENT.y1, abs=SLACK)

    def test_ignores_bottom_edge_artifact(self, tmp_path):
        """The regression that emptied the headnote rects.

        A headnote block's rect runs to the bottom of its column. Measured
        naively, a platen band along the page edge becomes the "last line"
        and the rect stretches to the page edge; clipping to the content
        box keeps it at the real last line.
        """
        pdf = tmp_path / "bottom_bar.pdf"
        write_bitonal_page(pdf, bottom_bar=True)
        with fitz.open(str(pdf)) as doc:
            box = ink_bbox(doc[0], fitz.Rect(72, 400, 540, PAGE_H))
        assert box is not None
        assert box[3] < BOTTOM_BAR.y0, "measured to the edge band"
        assert box[3] == pytest.approx(CONTENT.y1, abs=SLACK)

    def test_none_for_empty_region(self, tmp_path):
        pdf = tmp_path / "clean.pdf"
        write_bitonal_page(pdf)
        with fitz.open(str(pdf)) as doc:
            # Between the content box top and the page edge: no ink.
            box = ink_bbox(doc[0], fitz.Rect(0, 0, PAGE_W, 40))
        assert box is None

    def test_reads_a_page_once(self, tmp_path):
        pdf = tmp_path / "clean.pdf"
        write_bitonal_page(pdf)
        with fitz.open(str(pdf)) as doc:
            page = doc[0]
            mask, _sx, _sy = page_mask(page)
            ink_bbox(page, fitz.Rect(72, 100, 540, 400))
            again, _sx, _sy = page_mask(page)
        assert mask is again


class TestHasTextLayer:
    def test_true_for_text_pdf(self, tmp_path):
        pdf = tmp_path / "text.pdf"
        write_text_page(pdf)
        assert has_text_layer(pdf) is True

    def test_false_for_bitonal_pdf(self, tmp_path):
        pdf = tmp_path / "bitonal.pdf"
        write_bitonal_page(pdf)
        assert has_text_layer(pdf) is False

    def test_false_for_unreadable_path(self, tmp_path):
        assert has_text_layer(tmp_path / "nope.pdf") is False


class TestGrowToInk:
    """Tests for ``grow_to_ink``.

    Redaction rects come from detection geometry, which can sit a few
    points inside the printed text and clip the first character of every
    line or the tail of a line. The text-layer code path absorbed that,
    because word boxes overlapping a rect pulled its bounds outward; ink
    measured inside a rect has to grow back out explicitly.
    """

    @pytest.fixture
    def pdf(self, tmp_path):
        path = tmp_path / "bitonal.pdf"
        write_bitonal_page(path, header_line=True, bottom_bar=True)
        return path

    @staticmethod
    def _grow(pdf, rect):
        with fitz.open(str(pdf)) as doc:
            return grow_to_ink(doc[0], rect)

    def test_grows_over_a_clipped_first_character(self, pdf):
        """A rect starting inside the text column reaches back out to it."""
        clipped = fitz.Rect(CONTENT.x0 + 10, 300, CONTENT.x1, 400)
        grown = self._grow(pdf, clipped)
        assert grown.x0 == pytest.approx(CONTENT.x0, abs=4.0)
        assert grown.x1 <= CONTENT.x1 + 4.0

    def test_grows_over_a_clipped_line(self, pdf):
        """A rect whose bottom cuts a line takes in the rest of it."""
        # 300 pt is mid-block, so the edge lands inside a line of text.
        cut = fitz.Rect(CONTENT.x0, 200, CONTENT.x1, 300)
        grown = self._grow(pdf, cut)
        assert grown.y1 > 300
        assert grown.y1 < 300 + 12, "grew past the cut line"

    def test_stops_at_white_space(self, pdf):
        """Growth does not jump the blank gap up to the running head."""
        block = fitz.Rect(CONTENT.x0, CONTENT.y0 + 6, CONTENT.x1, 400)
        grown = self._grow(pdf, block)
        assert grown.y0 > HEADER_LINE_Y + 4, "swallowed the header"

    def test_stays_off_the_platen_band(self, pdf):
        """A rect at the foot of the page does not grow onto the edge band."""
        tail = fitz.Rect(CONTENT.x0, 600, CONTENT.x1, CONTENT.y1)
        grown = self._grow(pdf, tail)
        assert grown.y1 < BOTTOM_BAR.y0

    def test_never_shrinks(self, pdf):
        """Growth only ever adds coverage."""
        rect = fitz.Rect(CONTENT.x0 + 20, 250, CONTENT.x1 - 20, 350)
        grown = self._grow(pdf, rect)
        assert grown.x0 <= rect.x0
        assert grown.y0 <= rect.y0
        assert grown.x1 >= rect.x1
        assert grown.y1 >= rect.y1

    def test_unmeasurable_rect_is_returned_unchanged(self, pdf):
        outside = fitz.Rect(0, PAGE_H - 20, 30, PAGE_H)
        assert tuple(self._grow(pdf, outside)) == tuple(outside)


class TestGrowAcrossGutter:
    """Growth must not cross a hairline gutter into the next column.

    Regression: the outward walk started one pixel column *past* the rect
    edge, stepping over the single blank column that separates tightly set
    columns. A rect then grew across the gutter and swallowed its
    neighbour's text, which on a real page meant a ``TEXT_COLUMN`` box
    widening by 20 pt into the other column.
    """

    @pytest.fixture
    def pdf(self, tmp_path):
        path = tmp_path / "two_col.pdf"
        write_two_column_page(path)
        return path

    def test_right_column_does_not_grow_into_the_left(self, pdf):
        with fitz.open(str(pdf)) as doc:
            grown = grow_to_ink(doc[0], fitz.Rect(COLUMN_RIGHT), margin_y=0.0)
        assert grown.x0 >= COLUMN_LEFT.x1 - 1, f"grew back over the gutter to {grown.x0}"

    def test_left_column_does_not_grow_into_the_right(self, pdf):
        with fitz.open(str(pdf)) as doc:
            grown = grow_to_ink(doc[0], fitz.Rect(COLUMN_LEFT), margin_y=0.0)
        assert grown.x1 <= COLUMN_RIGHT.x0 + 1, f"grew forward over the gutter to {grown.x1}"

    def test_a_narrow_box_still_reaches_its_own_text(self, pdf):
        """The fix must not stop legitimate growth."""
        inset = fitz.Rect(COLUMN_RIGHT.x0 + 6, 200, COLUMN_RIGHT.x1 - 6, 400)
        with fitz.open(str(pdf)) as doc:
            grown = grow_to_ink(doc[0], inset, margin_y=0.0)
        assert grown.x0 == pytest.approx(COLUMN_RIGHT.x0, abs=2.0)
        assert grown.x1 == pytest.approx(COLUMN_RIGHT.x1, abs=2.0)
