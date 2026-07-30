"""Tests for ``snap_text_columns_to_ink``.

YOLO's ``TEXT_COLUMN`` boxes land a little inside the printed text. Three
consumers read them (headnote rect x-bounds, the margin text band, and the
outside-opinion masks), so the boxes are corrected once, at the source.
"""

from __future__ import annotations

import fitz
import pytest

from blackletter.models import Label
from blackletter.scanner import snap_text_columns_to_ink
from tests.pdf_fixtures import (
    COLUMN_LEFT,
    COLUMN_RIGHT,
    detected_page,
    detection,
    write_two_column_page,
)

# How far inside its text band each synthetic column box starts, in points.
INSET = 6.0


def _columns_of(page):
    """The page's TEXT_COLUMN detections, left to right."""
    return sorted(
        (d for d in page.detections if d.label == Label.TEXT_COLUMN),
        key=lambda d: d.bbox.x1,
    )


@pytest.fixture
def two_column(tmp_path):
    path = tmp_path / "two_col.pdf"
    write_two_column_page(path)
    return path


@pytest.fixture
def inset_page():
    """Column detections sitting ``INSET`` points inside their text."""
    return detected_page(
        [
            detection(
                Label.TEXT_COLUMN,
                COLUMN_LEFT.x0 + INSET,
                COLUMN_LEFT.y0,
                COLUMN_LEFT.x1 - INSET,
                COLUMN_LEFT.y1,
            ),
            detection(
                Label.TEXT_COLUMN,
                COLUMN_RIGHT.x0 + INSET,
                COLUMN_RIGHT.y0,
                COLUMN_RIGHT.x1 - INSET,
                COLUMN_RIGHT.y1,
            ),
        ]
    )


class TestSnapTextColumnsToInk:
    def test_widens_boxes_onto_their_text(self, two_column, inset_page):
        with fitz.open(str(two_column)) as doc:
            changed = snap_text_columns_to_ink(doc[0], inset_page)
        assert changed == 2
        left, right = _columns_of(inset_page)
        assert left.bbox.x1 == pytest.approx(COLUMN_LEFT.x0, abs=2.0)
        assert left.bbox.x2 == pytest.approx(COLUMN_LEFT.x1, abs=2.0)
        assert right.bbox.x1 == pytest.approx(COLUMN_RIGHT.x0, abs=2.0)
        assert right.bbox.x2 == pytest.approx(COLUMN_RIGHT.x1, abs=2.0)

    def test_neither_box_crosses_the_gutter(self, two_column, inset_page):
        """The gutter here is 1.5 pt: one blank pixel at 100 dpi."""
        with fitz.open(str(two_column)) as doc:
            snap_text_columns_to_ink(doc[0], inset_page)
        left, right = _columns_of(inset_page)
        assert left.bbox.x2 <= COLUMN_RIGHT.x0 + 1
        assert right.bbox.x1 >= COLUMN_LEFT.x1 - 1

    def test_y_bounds_and_labels_are_untouched(self, two_column, inset_page):
        before = [(d.bbox.y1, d.bbox.y2, d.label, d.confidence) for d in _columns_of(inset_page)]
        with fitz.open(str(two_column)) as doc:
            snap_text_columns_to_ink(doc[0], inset_page)
        after = [(d.bbox.y1, d.bbox.y2, d.label, d.confidence) for d in _columns_of(inset_page)]
        assert before == after

    def test_converges(self, two_column, inset_page):
        """Running it twice must not keep sliding the edges outward."""
        with fitz.open(str(two_column)) as doc:
            snap_text_columns_to_ink(doc[0], inset_page)
        first = [(d.bbox.x1, d.bbox.x2) for d in _columns_of(inset_page)]
        with fitz.open(str(two_column)) as doc:
            again = snap_text_columns_to_ink(doc[0], inset_page)
        assert again == 0
        assert [(d.bbox.x1, d.bbox.x2) for d in _columns_of(inset_page)] == first

    def test_refuses_an_edge_that_never_found_the_end_of_the_ink(self, two_column):
        """A box floating inside a wider text band must stay where it is.

        Both edges run the whole way to the growth limit, which means the
        measurement never found where the text stops: a full-width table
        under a one-column detection does this on a real page. Sliding the
        box out by the limit instead would move it again on every run.
        """
        floating = COLUMN_LEFT.x0 + 60, COLUMN_LEFT.x1 - 60
        page = detected_page(
            [detection(Label.TEXT_COLUMN, floating[0], COLUMN_LEFT.y0, floating[1], COLUMN_LEFT.y1)]
        )
        with fitz.open(str(two_column)) as doc:
            changed = snap_text_columns_to_ink(doc[0], page)
        assert changed == 0
        (column,) = _columns_of(page)
        assert (column.bbox.x1, column.bbox.x2) == floating

    def test_page_without_column_detections_is_a_no_op(self, two_column):
        page = detected_page([detection(Label.PAGE_HEADER, 72, 40, 540, 52)])
        with fitz.open(str(two_column)) as doc:
            assert snap_text_columns_to_ink(doc[0], page) == 0
