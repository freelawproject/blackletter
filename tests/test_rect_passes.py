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
from blackletter.process import (
    _snap_headnote_x_to_columns,
    _split_headnote_rects_at_headnotes,
)
from tests.pdf_fixtures import PAGE_H, PAGE_W, detected_page, detection

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
