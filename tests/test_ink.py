"""Tests for ``blackletter.ink``.

These measurements stand in for the PDF text layer, which a caller that
skips ``api.ocr`` never has, so the cases that matter are the ones where
ink and text disagree: scanner artifacts along the page edges, which carry
ink but are not printed text.
"""

from __future__ import annotations

import fitz
import pytest

from blackletter.ink import (
    content_box,
    grow_to_ink,
    has_text_layer,
    ink_bbox,
    invalidate,
    page_mask,
)
from tests.pdf_fixtures import (
    BOTTOM_BAR,
    FONT_SIZE,
    LINE_LEADING,
    COLUMN_LEFT,
    COLUMN_RIGHT,
    CONTENT,
    HEADER_LINE_Y,
    PAGE_H,
    PAGE_W,
    TOUCHING_BAR,
    rasterize,
    write_bitonal_page,
    write_multi_page,
    write_text_page,
    write_two_column_page,
)
from tests.pdf_fixtures import _line_for_width

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

    def test_a_solid_bar_next_to_the_text_is_still_excluded(self, tmp_path):
        """Isolates the mostly-dark rule from the gap rule.

        A band far from the text is excluded by either rule, so neither is
        pinned. This one sits inside bridging distance, close enough that
        the gap rule takes it for part of the text block, leaving only
        ``CONTENT_MAX_FRACTION`` to keep it out.
        """
        pdf = tmp_path / "near_bar.pdf"
        src = tmp_path / "near_bar.text.pdf"
        write_text_page(src)
        with fitz.open(str(src)) as doc:
            page = doc[0]
            page.draw_rect(
                fitz.Rect(0, CONTENT.y0 - 20, PAGE_W, CONTENT.y0 - 12),
                fill=(0, 0, 0),
                width=0,
            )
            doc.save(str(tmp_path / "near_bar.src.pdf"))
        rasterize(tmp_path / "near_bar.src.pdf", pdf)
        _left, top, _right, _bottom = self._box(pdf)
        # The box may start at the bar's lower edge, since the white gap
        # below it is bridged, but must not take in the bar's own rows.
        assert top >= CONTENT.y0 - 12 - 1, "a full-width bar was taken for text"

    def test_none_for_blank_page(self, tmp_path):
        pdf = tmp_path / "blank.pdf"
        with fitz.open() as doc:
            doc.new_page(width=PAGE_W, height=PAGE_H)
            doc.save(str(pdf))
        assert self._box(pdf) is None

    def test_the_widest_run_wins_when_a_page_has_two(self, tmp_path):
        """Content is the largest block of inked rows, not the first.

        A page whose front matter or an artifact band sits above the body
        has two candidate runs; taking the first would put the content box
        around the wrong one.
        """
        pdf = tmp_path / "two_runs.pdf"
        src = tmp_path / "two_runs.src.pdf"
        line = _line_for_width(CONTENT.width)
        with fitz.open() as doc:
            page = doc.new_page(width=PAGE_W, height=PAGE_H)
            # A short block near the top, well clear of the body below.
            for y in (60.0, 72.0):
                page.insert_text((CONTENT.x0, y), line, fontsize=FONT_SIZE)
            y = 300.0
            while y <= 700:
                page.insert_text((CONTENT.x0, y), line, fontsize=FONT_SIZE)
                y += LINE_LEADING
            doc.save(str(src))
        rasterize(src, pdf)
        box = self._box(pdf)
        assert box is not None
        assert box[1] > 200, f"the box began at the short upper block ({box[1]})"

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

    def test_each_page_gets_its_own_measurement(self, tmp_path):
        """The cache holds one page, so it must check which page it holds.

        Every other fixture here is a single page, which cannot tell a
        working key from a missing one. These three pages disagree: only the
        first has a measurable content box.
        """
        pdf = tmp_path / "multi.pdf"
        write_multi_page(pdf, ["body", "narrow", "blank"], tmp_dir=tmp_path)
        with fitz.open(str(pdf)) as doc:
            assert content_box(doc[0]) is not None
            assert content_box(doc[1]) is None
            assert content_box(doc[2]) is None

    def test_the_answer_does_not_depend_on_visit_order(self, tmp_path):
        pdf = tmp_path / "multi.pdf"
        write_multi_page(pdf, ["narrow", "body"], tmp_dir=tmp_path)
        with fitz.open(str(pdf)) as doc:
            forward = [content_box(doc[i]) for i in (0, 1)]
        with fitz.open(str(pdf)) as doc:
            backward = [content_box(doc[i]) for i in (1, 0)][::-1]
        assert forward == backward
        assert forward[0] is None and forward[1] is not None

    def test_the_mask_is_per_page_too(self, tmp_path):
        pdf = tmp_path / "multi.pdf"
        write_multi_page(pdf, ["body", "blank"], tmp_dir=tmp_path)
        with fitz.open(str(pdf)) as doc:
            body, _sx, _sy = page_mask(doc[0])
            blank, _sx, _sy = page_mask(doc[1])
            assert body.any(), "the body page measured as blank"
            assert not blank.any(), "the blank page got the body page's mask"

    def test_invalidate_drops_a_stale_measurement(self, tmp_path):
        """A page whose pixels change must be re-measurable."""
        pdf = tmp_path / "clean.pdf"
        write_bitonal_page(pdf)
        with fitz.open(str(pdf)) as doc:
            page = doc[0]
            assert content_box(page) is not None
            page.add_redact_annot(page.rect, fill=(1, 1, 1))
            page.apply_redactions()
            assert content_box(page) is not None, "expected the stale answer"
            invalidate(page)
            assert content_box(page) is None, "re-measured a blank page as content"


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

    def test_pages_beyond_the_first_are_sampled(self, tmp_path):
        """A volume whose front matter is blank still has a text layer."""
        pdf = tmp_path / "late_text.pdf"
        src = tmp_path / "late_text.src.pdf"
        with fitz.open() as doc:
            doc.new_page(width=PAGE_W, height=PAGE_H)
            doc.new_page(width=PAGE_W, height=PAGE_H)
            doc[1].insert_text((72, 200), "words on the second page", fontsize=9)
            doc.save(str(src))
        src.replace(pdf)
        assert has_text_layer(pdf) is True
        assert has_text_layer(pdf, sample_pages=1) is False


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

    def test_stays_inside_the_content_box_where_ink_crosses_it(self, tmp_path):
        """Growth stops at the content box, not merely at its own margin.

        The band here touches the first line of text, so the ink runs
        continuously from inside the content box into an artifact the box
        excludes. Nothing else can stop the walk: there is no white space to
        break on, and the margin is set well past the band.
        """
        pdf = tmp_path / "touching.pdf"
        write_bitonal_page(pdf, touching_bar=True, tmp_dir=tmp_path)
        with fitz.open(str(pdf)) as doc:
            page = doc[0]
            box = content_box(page)
            assert box is not None
            assert box[1] > TOUCHING_BAR.y1, "the band was taken for content"
            rect = fitz.Rect(CONTENT.x0, box[1] + 2, CONTENT.x1, 300)
            grown = grow_to_ink(page, rect, margin_y=60.0)
        assert grown.y0 == pytest.approx(box[1], abs=1.0), "did not stop at the box"
        assert grown.y0 > TOUCHING_BAR.y1, "grew onto the band"

    def test_growth_never_pushes_an_edge_outside_the_content_box(self, pdf):
        """Each edge is either untouched or inside the box, for any rect."""
        with fitz.open(str(pdf)) as doc:
            page = doc[0]
            box = content_box(page)
            assert box is not None
            for rect in (
                fitz.Rect(CONTENT.x0, 600, CONTENT.x1, BOTTOM_BAR.y0 - 1),
                fitz.Rect(CONTENT.x0, CONTENT.y0 + 4, CONTENT.x1, 300),
                fitz.Rect(CONTENT.x0 + 5, 200, CONTENT.x1 - 5, 400),
                fitz.Rect(0, 0, PAGE_W, PAGE_H),
            ):
                grown = grow_to_ink(page, rect)
                for edge, low, limit in (
                    (grown.x0, rect.x0, box[0]),
                    (grown.y0, rect.y0, box[1]),
                ):
                    assert edge == low or edge >= limit - 1e-6, rect
                for edge, high, limit in (
                    (grown.x1, rect.x1, box[2]),
                    (grown.y1, rect.y1, box[3]),
                ):
                    assert edge == high or edge <= limit + 1e-6, rect

    def test_an_edge_that_never_finds_blank_space_is_refused(self, tmp_path):
        """Growth that consumes its whole margin has learned nothing.

        A rect floating inside a solid band has ink for the full margin in
        both directions, so each walk runs to its limit. Moving the edge by
        exactly the margin would be arbitrary, and on a page whose gutter is
        thinner than a measuring pixel it is how a mask reaches the facing
        column. Vertically the band does have gaps between its lines, so
        that axis is disabled here to isolate the rule.
        """
        pdf = tmp_path / "band.pdf"
        write_two_column_page(pdf, tmp_dir=tmp_path)
        inside = fitz.Rect(COLUMN_LEFT.x0 + 60, 200, COLUMN_LEFT.x1 - 60, 400)
        with fitz.open(str(pdf)) as doc:
            grown = grow_to_ink(doc[0], inside, margin_y=0.0)
        assert (grown.x0, grown.x1) == (inside.x0, inside.x1), (
            "moved by the margin instead of refusing"
        )

    def test_a_rect_with_nothing_to_grow_onto_is_returned_exactly(self, pdf):
        """No growth means no movement, not a snap to the pixel grid."""
        blank = fitz.Rect(200.0, 70.3, 300.0, 90.7)
        grown = self._grow(pdf, blank)
        assert tuple(grown) == tuple(blank)

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

    The assertions here are tight to the column edge on purpose. With a
    tolerance of a point, reverting the off-by-one still passes: the
    fixture's 1.5 pt gutter spans two measuring pixels, so a walk that
    starts one column too far out overshoots by less than the tolerance and
    the regression goes unnoticed.
    """

    @pytest.fixture
    def pdf(self, tmp_path):
        path = tmp_path / "two_col.pdf"
        write_two_column_page(path)
        return path

    def test_right_column_does_not_grow_into_the_left(self, pdf):
        with fitz.open(str(pdf)) as doc:
            grown = grow_to_ink(doc[0], fitz.Rect(COLUMN_RIGHT), margin_y=0.0)
        assert grown.x0 >= COLUMN_RIGHT.x0 - 0.01, f"grew back over the gutter to {grown.x0}"

    def test_left_column_does_not_grow_into_the_right(self, pdf):
        with fitz.open(str(pdf)) as doc:
            grown = grow_to_ink(doc[0], fitz.Rect(COLUMN_LEFT), margin_y=0.0)
        assert grown.x1 <= COLUMN_LEFT.x1 + 0.01, f"grew forward over the gutter to {grown.x1}"

    def test_repeated_growth_does_not_creep(self, pdf):
        """The off-by-one showed up as drift, a fraction of a point per call.

        A single application can look harmless; the rect is regrown on every
        recompute, so an edge that moves each time eventually arrives in the
        next column.
        """
        rect = fitz.Rect(COLUMN_LEFT)
        with fitz.open(str(pdf)) as doc:
            for _ in range(5):
                rect = grow_to_ink(doc[0], rect, margin_y=0.0)
        assert rect.x1 <= COLUMN_LEFT.x1 + 0.01, f"crept to {rect.x1} over five calls"

    def test_a_narrow_box_still_reaches_its_own_text(self, pdf):
        """The fix must not stop legitimate growth."""
        inset = fitz.Rect(COLUMN_RIGHT.x0 + 6, 200, COLUMN_RIGHT.x1 - 6, 400)
        with fitz.open(str(pdf)) as doc:
            grown = grow_to_ink(doc[0], inset, margin_y=0.0)
        assert grown.x0 == pytest.approx(COLUMN_RIGHT.x0, abs=2.0)
        assert grown.x1 == pytest.approx(COLUMN_RIGHT.x1, abs=2.0)
