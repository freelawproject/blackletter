"""Tests for the detection-driven passes over headnote and margin rects.

These finish a rect once its geometry is known: snap its side edges to the
column box, cut it at the headnotes inside it, and pull a margin strip back
off anything real it would cover. All three used to live in the scanning
portal, post-processing this library's own output.
"""

from __future__ import annotations

import pytest

from blackletter.margins import _shrink_rects_for_detections
from blackletter.models import Label
import fitz

from blackletter.process import (
    _drop_degenerate,
    _drop_rects_off_the_columns,
    _grow_headnote_rects,
    _snap_headnote_x_to_columns,
    _split_headnote_rects_at_headnotes,
)
from tests.pdf_fixtures import (
    PAGE_H,
    PAGE_W,
    detected_page,
    detection,
)

# Image pixel geometry for a two-column page, at 1 px per point so the
# fixtures read as points (see ``detected_page``).
LEFT_COL = (72.0, 300.0)
RIGHT_COL = (320.0, 540.0)


def headnote(x0, y0, x1, y1, **extra):
    """A headnote rect as ``compute_redaction_rects`` emits it."""
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "fill": "black", "type": "headnote", **extra}


def columns():
    return [
        detection(Label.TEXT_COLUMN, LEFT_COL[0], 100, LEFT_COL[1], 700),
        detection(Label.TEXT_COLUMN, RIGHT_COL[0], 100, RIGHT_COL[1], 700),
    ]


class TestSnapHeadnoteXToColumns:
    def test_edges_move_to_the_column_box(self):
        page = detected_page(columns())
        rects = [headnote(80, 150, 290, 400)]
        _snap_headnote_x_to_columns(page, rects)
        assert (rects[0]["x0"], rects[0]["x1"]) == LEFT_COL

    def test_each_rect_snaps_to_its_own_side(self):
        page = detected_page(columns())
        rects = [headnote(80, 150, 290, 400), headnote(330, 150, 530, 400)]
        _snap_headnote_x_to_columns(page, rects)
        assert (rects[0]["x0"], rects[0]["x1"]) == LEFT_COL
        assert (rects[1]["x0"], rects[1]["x1"]) == RIGHT_COL

    def test_a_rect_never_crosses_the_midpoint(self):
        """The regression this guard exists for.

        With only the right column detected, the nearest column box to a
        left-hand rect is the right one, and snapping to it would move the
        rect to the other side of the page.
        """
        page = detected_page([columns()[1]])
        rects = [headnote(80, 150, 290, 400)]
        _snap_headnote_x_to_columns(page, rects)
        assert (rects[0]["x0"], rects[0]["x1"]) == (80, 290), "snapped across the page"

    def test_other_rect_types_are_untouched(self):
        page = detected_page(columns())
        rects = [{**headnote(80, 150, 290, 400), "type": "KEY_ICON"}]
        _snap_headnote_x_to_columns(page, rects)
        assert (rects[0]["x0"], rects[0]["x1"]) == (80, 290)

    def test_a_box_spanning_both_columns_is_refused(self):
        """The worst case found: it blacks out the facing column's text.

        A merged column detection, or one a reviewer widened by hand, is
        about twice a headnote rect's width. Snapping to it stretches every
        headnote blackout across the court's own opinion text.
        """
        # Centre kept on the rect's own side of the midpoint, so the
        # same-side rule cannot reject it first and the width rule is what
        # is under test.
        merged = detection(Label.TEXT_COLUMN, LEFT_COL[0], 100, 400.0, 700)
        page = detected_page([merged])
        rect = headnote(80, 150, 290, 400)
        assert merged.bbox.center_x < page.img_width / 2, "not on the rect's side"
        assert merged.bbox.width > (rect["x1"] - rect["x0"]) * 1.5, "not wide enough to reject"
        rects = [rect]
        _snap_headnote_x_to_columns(page, rects)
        assert (rects[0]["x0"], rects[0]["x1"]) == (80, 290), "stretched over the neighbour"

    def test_a_degenerate_box_is_refused(self):
        """The mirror image: it collapses the rect and leaves headnotes visible."""
        for x1 in (186.0, 188.0):  # zero width, then two points
            page = detected_page([detection(Label.TEXT_COLUMN, 186.0, 100, x1, 700)])
            rects = [headnote(80, 150, 290, 400)]
            _snap_headnote_x_to_columns(page, rects)
            assert (rects[0]["x0"], rects[0]["x1"]) == (80, 290), f"collapsed to {x1}"

    def test_a_degenerate_rect_is_dropped_before_returning(self):
        """Defence in depth for the passes above.

        ``_add_px`` rejects a zero-area rect on the way into
        ``compute_redaction_rects``, and these passes run after it, so the
        same check runs on the way out. Tested directly: with the guards in
        place nothing here can produce such a rect any more, which is the
        point, and the filter stays because a future pass might.
        """
        rects = [
            headnote(80, 150, 290, 400),
            headnote(186, 150, 186, 400),
            headnote(80, 150, 290, 150),
        ]
        assert _drop_degenerate(rects) == [rects[0]]

    def test_a_column_a_little_out_is_still_used(self):
        """The guard must not block the correction it exists to allow."""
        page = detected_page([detection(Label.TEXT_COLUMN, 72.0, 100, 300.0, 700)])
        rects = [headnote(80, 150, 290, 400)]
        _snap_headnote_x_to_columns(page, rects)
        assert (rects[0]["x0"], rects[0]["x1"]) == (72.0, 300.0)

    def test_no_columns_is_a_no_op(self):
        page = detected_page([detection(Label.PAGE_HEADER, 72, 40, 540, 52)])
        rects = [headnote(80, 150, 290, 400)]
        _snap_headnote_x_to_columns(page, rects)
        assert (rects[0]["x0"], rects[0]["x1"]) == (80, 290)


class TestSplitHeadnoteRects:
    """A block rect runs to the end of its column and can span several
    numbered headnotes with body text between them."""

    def test_splits_at_each_detection(self):
        page = detected_page(
            [
                detection(Label.HEADNOTE, 80, 300, 290, 340),
                detection(Label.HEADNOTE, 80, 500, 290, 540),
            ]
        )
        out = _split_headnote_rects_at_headnotes(page, [headnote(80, 150, 290, 700)])
        assert len(out) == 3
        assert [(r["y0"], r["y1"]) for r in out] == [
            (150, 297.0),
            (303.0, 497.0),
            (503.0, 700),
        ]

    def test_pieces_keep_the_x_bounds_and_type(self):
        page = detected_page([detection(Label.HEADNOTE, 80, 300, 290, 340)])
        out = _split_headnote_rects_at_headnotes(page, [headnote(80, 150, 290, 700)])
        assert all(r["x0"] == 80 and r["x1"] == 290 for r in out)
        assert all(r["type"] == "headnote" and r["fill"] == "black" for r in out)

    def test_detections_in_the_other_column_do_not_cut(self):
        page = detected_page([detection(Label.HEADNOTE, 330, 300, 530, 340)])
        rects = [headnote(80, 150, 290, 700)]
        assert _split_headnote_rects_at_headnotes(page, rects) == rects

    def test_detections_flush_with_the_rect_edges_do_not_cut(self):
        """Otherwise every rect gains a zero-height sliver at its own top."""
        page = detected_page([detection(Label.HEADNOTE, 80, 152, 290, 200)])
        rects = [headnote(80, 150, 290, 700)]
        assert _split_headnote_rects_at_headnotes(page, rects) == rects

    def test_overlapping_detections_merge_into_one_cut(self):
        page = detected_page(
            [
                detection(Label.HEADNOTE, 80, 300, 290, 400),
                detection(Label.HEADNOTE, 80, 350, 290, 450),
            ]
        )
        out = _split_headnote_rects_at_headnotes(page, [headnote(80, 150, 290, 700)])
        assert [(r["y0"], r["y1"]) for r in out] == [(150, 297.0), (303.0, 700)]

    def test_other_rect_types_pass_through_in_order(self):
        page = detected_page([detection(Label.HEADNOTE, 80, 300, 290, 340)])
        other = {**headnote(80, 60, 290, 80), "type": "PAGE_HEADER"}
        out = _split_headnote_rects_at_headnotes(page, [other, headnote(80, 150, 290, 700)])
        assert out[0] is other
        assert len(out) == 3


class TestShrinkMarginsForDetections:
    """Bounds tightening positions the strips from the text band, which says
    nothing about a key icon at the foot of a page or an image bleeding
    outward, so a strip can still land on one."""

    @staticmethod
    def _strips():
        """Left, right, top and bottom strips around a body box."""
        return [
            {"x0": 0, "y0": 40, "x1": 60, "y1": 720},
            {"x0": 550, "y0": 40, "x1": PAGE_W, "y1": 720},
            {"x0": 0, "y0": 0, "x1": PAGE_W, "y1": 40},
            {"x0": 0, "y0": 720, "x1": PAGE_W, "y1": PAGE_H},
        ]

    def test_side_strip_pulls_back_from_a_key_icon(self):
        page = detected_page([detection(Label.KEY_ICON, 40, 300, 70, 330)])
        strips = self._strips()
        _shrink_rects_for_detections(page, strips)
        assert strips[0]["x1"] == 40, "left strip still covers the icon"
        assert strips[1]["x0"] == 550, "right strip moved for no reason"

    def test_right_strip_pulls_back_from_the_right(self):
        page = detected_page([detection(Label.IMAGE, 540, 300, 580, 330)])
        strips = self._strips()
        _shrink_rects_for_detections(page, strips)
        assert strips[1]["x0"] == 580

    def test_full_width_strips_pull_back_vertically(self):
        page = detected_page(
            [
                detection(Label.CASE_CAPTION, 100, 20, 400, 60),
                detection(Label.KEY_ICON, 100, 700, 130, 740),
            ]
        )
        strips = self._strips()
        _shrink_rects_for_detections(page, strips)
        assert strips[2]["y1"] == 20, "top strip covers the caption"
        assert strips[3]["y0"] == 740, "bottom strip covers the key icon"

    def test_edge_labels_do_not_push_strips_back(self):
        """A bleed-through blob labelled PAGE_NUMBER is what a strip is for."""
        page = detected_page(
            [
                detection(Label.PAGE_NUMBER, 500, 2, 560, 16),
                detection(Label.PAGE_HEADER, 100, 4, 400, 18),
                detection(Label.STATE_ABBREVIATION, 20, 300, 50, 330),
            ]
        )
        strips = self._strips()
        before = [dict(s) for s in strips]
        _shrink_rects_for_detections(page, strips)
        assert strips == before

    def test_detections_clear_of_the_strips_change_nothing(self):
        page = detected_page([detection(Label.HEADNOTE, 100, 200, 400, 300)])
        strips = self._strips()
        before = [dict(s) for s in strips]
        _shrink_rects_for_detections(page, strips)
        assert strips == before

    def test_page_with_no_detections_is_a_no_op(self):
        page = detected_page([])
        strips = self._strips()
        before = [dict(s) for s in strips]
        _shrink_rects_for_detections(page, strips)
        assert strips == before

    def test_the_worst_intruder_wins(self):
        """Two detections in one strip: the strip clears both."""
        page = detected_page(
            [
                detection(Label.KEY_ICON, 45, 300, 70, 330),
                detection(Label.IMAGE, 20, 500, 55, 530),
            ]
        )
        strips = self._strips()
        _shrink_rects_for_detections(page, strips)
        assert strips[0]["x1"] == pytest.approx(20)


class TestGrowHeadnoteRects:
    """The pass that puts back what the snap takes away.

    The snap replaces a headnote rect's side edges with its column box, on
    any page. If growth then fails to run, or converts coordinates wrongly,
    every headnote rect is left as narrow as the detector drew it, and a
    column box a few points inside the text clips a character on every line.
    """

    @pytest.fixture
    def columns_page(self, hairline_pdf):
        """A hairline-gutter page and a Page describing its two columns."""
        pdf, left, right = hairline_pdf
        page = detected_page(
            [
                detection(Label.TEXT_COLUMN, left.x0, left.y0, left.x1, left.y1),
                detection(Label.TEXT_COLUMN, right.x0, right.y0, right.x1, right.y1),
            ]
        )
        return pdf, page, left, right

    def test_a_narrowed_rect_is_grown_back_onto_its_text(self, columns_page):
        pdf, page, left, _right = columns_page
        rects = [headnote(left.x0 + 6, 200, left.x1 - 6, 400)]
        with fitz.open(str(pdf)) as doc:
            _grow_headnote_rects(doc[0], page, rects, ocr_applied=True)
        assert rects[0]["x0"] < left.x0 + 6, "the left edge was not grown"
        assert rects[0]["x1"] > left.x1 - 6, "the right edge was not grown"

    def test_growth_stays_out_of_the_facing_column(self, columns_page):
        """A gutter thinner than a measuring pixel is not a barrier by itself.

        The walk starts past the single pixel column holding the gutter, so
        without the cap a short rect meets the facing column's type and
        keeps going.
        """
        pdf, page, left, right = columns_page
        for height in (6, 12, 24, 60):
            rects = [headnote(left.x0, 200, left.x1, 200 + height)]
            with fitz.open(str(pdf)) as doc:
                _grow_headnote_rects(doc[0], page, rects, ocr_applied=True)
            assert rects[0]["x1"] <= right.x0, f"crossed the gutter at height {height}"

    def test_the_scale_factors_are_not_swapped(self, hairline_pdf):
        """Rects are in image pixels and growth measures points.

        The two axes are given different scales on purpose: with the same
        factor on both, swapping them is a no-op and the test proves
        nothing. A swap then moves an edge by a factor rather than a hair,
        so the grown rect lands nowhere near its own text.
        """
        pdf, left, _right = hairline_pdf
        page = detected_page([detection(Label.TEXT_COLUMN, left.x0, left.y0, left.x1, left.y1)])
        page.img_width, page.img_height = int(PAGE_W * 2), int(PAGE_H * 3)
        assert page.scale_x != page.scale_y, "a swap would be undetectable"
        # The same rect, expressed in that page's pixels.
        rects = [headnote((left.x0 + 6) * 2, 200 * 3, (left.x1 - 6) * 2, 400 * 3)]
        with fitz.open(str(pdf)) as doc:
            _grow_headnote_rects(doc[0], page, rects, ocr_applied=True)
        assert rects[0]["x0"] == pytest.approx(left.x0 * 2, abs=6.0)
        assert rects[0]["x1"] == pytest.approx(left.x1 * 2, abs=6.0)

    def test_other_rect_types_are_left_alone(self, columns_page):
        pdf, page, left, _right = columns_page
        other = {**headnote(left.x0 + 6, 200, left.x1 - 6, 400), "type": "KEY_ICON"}
        rects = [other]
        before = dict(other)
        with fitz.open(str(pdf)) as doc:
            _grow_headnote_rects(doc[0], page, rects, ocr_applied=True)
        assert rects[0] == before


class TestClipHeadnoteRectGutter:
    """``_clip_headnote_rect`` grows too, and must respect the same bound.

    Its growth was the one site the gutter cap missed. The path is live
    through ``rebuild_full_redacted_from_detections`` and ``split_opinions``,
    both of which set ``ocr_applied``, which is what turns the growth on.
    """

    def test_growth_stays_out_of_the_facing_column(self, hairline_pdf):
        from blackletter.scanner import _clip_headnote_rect

        pdf, left, right = hairline_pdf
        page = detected_page(
            [
                detection(Label.TEXT_COLUMN, left.x0, left.y0, left.x1, left.y1),
                detection(Label.TEXT_COLUMN, right.x0, right.y0, right.x1, right.y1),
            ]
        )
        rect = fitz.Rect(left.x0, 200, left.x1, 206)
        with fitz.open(str(pdf)) as doc:
            clipped = _clip_headnote_rect(doc[0], rect, 50.0, PAGE_H, ocr_applied=True, page=page)
        assert clipped is not None
        assert clipped.x1 <= right.x0, f"grew into the facing column, to {clipped.x1}"

    def test_without_a_page_it_still_returns_a_rect(self, hairline_pdf):
        """Callers that have no detected page keep the older behaviour."""
        from blackletter.scanner import _clip_headnote_rect

        pdf, left, _right = hairline_pdf
        rect = fitz.Rect(left.x0, 200, left.x1, 260)
        with fitz.open(str(pdf)) as doc:
            clipped = _clip_headnote_rect(doc[0], rect, 50.0, PAGE_H, ocr_applied=True)
        assert clipped is not None


class TestDropRectsOffTheColumns:
    """A headnote rect that touches no column is not covering headnote text.

    Real case, from a reporter volume: on a band whose columns are blank,
    the tightening finds the only ink there is, a speck out in the margin,
    and the rect collapses onto it. The column snap will not rescue it,
    because by then the rect is a few points wide and the column box is not
    a plausible width for it. What ships is a small black mark printed over
    a speck, in a band that needed no covering at all.
    """

    def test_a_rect_collapsed_onto_a_margin_speck_is_dropped(self):
        page = detected_page(columns())
        speck = headnote(40, 150, 52, 160)
        assert _drop_rects_off_the_columns(page, [speck]) == []

    def test_a_rect_on_a_column_is_kept(self):
        page = detected_page(columns())
        rects = [headnote(*LEFT_COL[:1], 150, LEFT_COL[1], 400)]
        assert _drop_rects_off_the_columns(page, rects) == rects

    def test_a_rect_overlapping_a_column_at_all_is_kept(self):
        """Partial overlap still covers column text, so it stays."""
        page = detected_page(columns())
        rects = [headnote(40, 150, LEFT_COL[0] + 5, 400)]
        assert _drop_rects_off_the_columns(page, rects) == rects

    def test_a_rect_in_the_gutter_is_dropped(self):
        page = detected_page(columns())
        gutter = headnote(LEFT_COL[1] + 2, 150, RIGHT_COL[0] - 2, 400)
        assert _drop_rects_off_the_columns(page, [gutter]) == []

    def test_a_rect_above_the_columns_is_dropped(self):
        """Vertical position counts too: the header band is not a column."""
        page = detected_page(columns())
        header = headnote(LEFT_COL[0], 20, LEFT_COL[1], 60)
        assert _drop_rects_off_the_columns(page, [header]) == []

    def test_other_rect_types_are_untouched(self):
        """Only headnote rects are judged this way.

        A page number or a key icon whiteout sits outside the columns by
        definition, and covering it is the whole point.
        """
        page = detected_page(columns())
        icon = {**headnote(40, 150, 52, 160), "type": "KEY_ICON"}
        assert _drop_rects_off_the_columns(page, [icon]) == [icon]

    def test_a_page_with_no_columns_keeps_everything(self):
        """With nothing to be outside of, there is nothing to judge."""
        page = detected_page([])
        rects = [headnote(40, 150, 52, 160)]
        assert _drop_rects_off_the_columns(page, rects) == rects
