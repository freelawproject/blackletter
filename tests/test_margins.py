"""Tests for ``blackletter.margins``.

Margin rects have to be measurable from a text-less bitonal PDF, since the
text layer is optional. These tests build small synthetic PDFs (a text
page, the same page rasterized to 1-bit, and variants carrying scanner
artifacts) and check the content box the margin rects leave uncovered
against the page's actual ink, measured independently with a plain
dark-pixel scan.
"""

from __future__ import annotations

from pathlib import Path

import fitz
import numpy as np
import pytest

from blackletter.margins import compute_margin_rects
from blackletter.models import Label
from tests.pdf_fixtures import (
    BLEED_MARK,
    BOTTOM_BAR,
    CONTENT,
    CORNER_NUMBER_X,
    HEADER_LINE_Y,
    IMAGE_BLOCK,
    PAGE_H,
    PAGE_W,
    STRAY_MARK,
    TOP_BAR,
    detected_page,
    detection,
    rasterize,
    write_bitonal_page,
    write_text_page,
)

BUFFER = 5.0


def _ink_bbox(pdf_path: Path) -> tuple[float, float, float, float]:
    """Measure where the marks are on page 0, in PDF points.

    Deliberately naive (any dark pixel counts) so it is independent of the
    heuristics under test. Used as the reference box.

    :param pdf_path: PDF to measure.
    :return: ``(left, top, right, bottom)`` in PDF points.
    """
    with fitz.open(str(pdf_path)) as doc:
        page = doc[0]
        pix = page.get_pixmap(dpi=200, colorspace=fitz.csGRAY)
        gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)[
            :, : pix.width
        ]
        dark = gray < 200
        rows = np.flatnonzero(dark.any(axis=1))
        cols = np.flatnonzero(dark.any(axis=0))
        sx = page.rect.width / pix.width
        sy = page.rect.height / pix.height
    return cols[0] * sx, rows[0] * sy, (cols[-1] + 1) * sx, (rows[-1] + 1) * sy


def _uncovered_box(rects: list[dict]) -> tuple[float, float, float, float]:
    """Derive the region a page's margin rects leave uncovered.

    Works off a coverage grid rather than the rects' shapes, so it does not
    care how the strips are cut up.

    :param rects: The ``rects`` list for one page.
    :return: ``(left, top, right, bottom)`` in PDF points.
    """
    cell = 1.0
    cols, rows = int(PAGE_W / cell), int(PAGE_H / cell)
    covered = np.zeros((rows, cols), dtype=bool)
    for r in rects:
        covered[
            max(0, int(r["y0"] / cell)) : int(r["y1"] / cell),
            max(0, int(r["x0"] / cell)) : int(r["x1"] / cell),
        ] = True
    free_rows = np.flatnonzero(~covered.all(axis=1))
    free_cols = np.flatnonzero(~covered.all(axis=0))
    if not free_rows.size or not free_cols.size:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        free_cols[0] * cell,
        free_rows[0] * cell,
        (free_cols[-1] + 1) * cell,
        (free_rows[-1] + 1) * cell,
    )


def _rects_for(result: list[dict], page_index: int = 0) -> list[dict]:
    """Pull one page's rects out of a ``compute_margin_rects`` result."""
    return next(e for e in result if e["page_index"] == page_index)["rects"]


def _columns() -> list:
    """TEXT_COLUMN detections spanning the body text."""
    mid = (CONTENT.x0 + CONTENT.x1) / 2
    return [
        detection(Label.TEXT_COLUMN, CONTENT.x0, CONTENT.y0, mid - 6, CONTENT.y1),
        detection(Label.TEXT_COLUMN, mid + 6, CONTENT.y0, CONTENT.x1, CONTENT.y1),
    ]


def _header():
    """The header-row detection a real page carries."""
    return detection(
        Label.PAGE_HEADER, CONTENT.x0, HEADER_LINE_Y - 8, CONTENT.x1, HEADER_LINE_Y + 2
    )


class TestComputeMarginRects:
    """Content box detection with and without a text layer."""

    # The measured box may sit a few points outside the ink (the safety
    # buffer, glyph bounds, ink rounding at 100 dpi) but must never cut
    # into it.
    SLACK = 16.0

    def assert_box_matches_ink(self, rects, ink_box):
        """Assert the uncovered box contains the ink and stays tight."""
        left, top, right, bottom = _uncovered_box(rects)
        ink_left, ink_top, ink_right, ink_bottom = ink_box
        assert left <= ink_left, "left margin covers content"
        assert top <= ink_top, "top margin covers content"
        assert right >= ink_right, "right margin covers content"
        assert bottom >= ink_bottom, "bottom margin covers content"
        assert left > ink_left - self.SLACK
        assert top > ink_top - self.SLACK
        assert right < ink_right + self.SLACK
        assert bottom < ink_bottom + self.SLACK

    def test_text_layer_page(self, tmp_path):
        """A page with text is measured from its text blocks."""
        pdf = tmp_path / "text.pdf"
        write_text_page(pdf)
        result = compute_margin_rects(pdf)
        ink_box = _ink_bbox(pdf)

        assert len(result) == 1
        assert result[0]["page_index"] == 0
        assert _rects_for(result), "no margin rects computed"
        self.assert_box_matches_ink(_rects_for(result), ink_box)

    def test_textless_page_measured_from_ink(self, tmp_path):
        """A bitonal page has no text, so ink defines the content box."""
        src = tmp_path / "text.pdf"
        pdf = tmp_path / "bitonal.pdf"
        write_text_page(src)
        rasterize(src, pdf)
        with fitz.open(str(pdf)) as doc:
            assert doc[0].get_text("text").strip() == ""
        result = compute_margin_rects(pdf)
        ink_box = _ink_bbox(pdf)

        assert _rects_for(result), "no margin rects computed"
        self.assert_box_matches_ink(_rects_for(result), ink_box)

    def test_scanner_edge_bar_is_excluded(self, tmp_path):
        """A solid bar at the page edge must not become content."""
        clean_src = tmp_path / "clean.pdf"
        bar_src = tmp_path / "bar.pdf"
        pdf = tmp_path / "bitonal.pdf"
        write_text_page(clean_src)
        write_text_page(bar_src, top_bar=True)
        rasterize(bar_src, pdf)
        result = compute_margin_rects(pdf)
        # Reference is the page without the artifact: the bar must not move
        # the content box.
        ink_box = _ink_bbox(clean_src)

        rects = _rects_for(result)
        assert rects, "no margin rects computed"
        _left, top, _right, _bottom = _uncovered_box(rects)
        # The bar sits at y=4..12, so the top margin rect must reach past
        # it for cleanup to cover it.
        assert top > 12
        self.assert_box_matches_ink(rects, ink_box)

    def test_blank_page_gets_no_rects(self, tmp_path):
        """Nothing to measure means leave the page alone."""
        pdf = tmp_path / "blank.pdf"
        with fitz.open() as doc:
            doc.new_page(width=PAGE_W, height=PAGE_H)
            doc.save(str(pdf))
        result = compute_margin_rects(pdf)

        assert _rects_for(result) == []
        assert result[0]["page_width"] == PAGE_W
        assert result[0]["page_height"] == PAGE_H

    def test_narrow_content_is_skipped(self, tmp_path):
        """Narrow content (image page, appendix) is not a margin boundary."""
        src = tmp_path / "narrow.pdf"
        with fitz.open() as doc:
            page = doc.new_page(width=PAGE_W, height=PAGE_H)
            page.insert_text((280, 400), "12", fontsize=9)
            doc.save(str(src))
        # Both paths must skip it: with a text layer and without.
        assert _rects_for(compute_margin_rects(src)) == []
        raster = tmp_path / "bitonal.pdf"
        rasterize(src, raster)
        assert _rects_for(compute_margin_rects(raster)) == []

    def test_an_image_below_the_text_extends_the_content_box(self, tmp_path):
        """A key icon at the foot of a page is content, not an artifact."""
        plain = tmp_path / "plain.pdf"
        with_icon = tmp_path / "icon.pdf"
        write_text_page(plain)
        write_text_page(with_icon, image_block=True)
        plain_box = _uncovered_box(_rects_for(compute_margin_rects(plain)))
        icon_box = _uncovered_box(_rects_for(compute_margin_rects(with_icon)))
        assert icon_box[3] > plain_box[3] + 10, "the image did not move the bottom"
        assert icon_box[3] >= IMAGE_BLOCK.y1, "the bottom strip covers the image"


class TestDetectionTightenedMargins:
    """Margins tightened with detection geometry.

    Ink is the union of every mark on a page, so one speck out in a margin
    pushes the content box to it and the strip on that side shrinks away.
    ``TEXT_COLUMN`` boxes bound the printed text instead, and the header row
    bounds it vertically, so the two estimates get intersected.
    """

    def test_column_band_tightens_the_side_strips(self, tmp_path):
        """A speck in the margin no longer decides where a strip stops."""
        pdf = tmp_path / "stray.pdf"
        write_bitonal_page(pdf, stray_mark=True)

        loose = _uncovered_box(_rects_for(compute_margin_rects(pdf)))
        assert loose[0] <= STRAY_MARK.x0, "speck should widen the content box"

        tight = _uncovered_box(
            _rects_for(compute_margin_rects(pdf, pages=[detected_page(_columns())]))
        )
        assert tight[0] == pytest.approx(CONTENT.x0 - BUFFER, abs=2.0)
        assert tight[0] > STRAY_MARK.x1, "speck left unmasked"

    def test_header_detection_restores_the_top_strip(self, tmp_path):
        """Bleed-through at the very top edge must not suppress the strip."""
        pdf = tmp_path / "bleed.pdf"
        write_bitonal_page(pdf, header_line=True, bleed_mark=True)

        without = _rects_for(compute_margin_rects(pdf))
        assert not [r for r in without if r["y0"] <= 1 and r["x1"] - r["x0"] > 400], (
            "expected the bleed mark to suppress the top strip"
        )

        dets = [
            *_columns(),
            _header(),
            # ...and the bleed itself, which YOLO also labels; being inside
            # the top edge band, it must not define the top bound.
            detection(
                Label.PAGE_NUMBER, BLEED_MARK.x0, BLEED_MARK.y0, BLEED_MARK.x1, BLEED_MARK.y1
            ),
        ]
        rects = _rects_for(compute_margin_rects(pdf, pages=[detected_page(dets)]))
        top = [r for r in rects if r["y0"] <= 1 and r["x1"] - r["x0"] > 400]
        assert top, "no top strip"
        assert top[0]["y1"] > BLEED_MARK.y1, "bleed left unmasked"
        assert top[0]["y1"] < HEADER_LINE_Y - 8, "top strip hits header"

    def test_page_number_outside_the_band_survives(self, tmp_path):
        """A page number printed outside the columns keeps its whitespace.

        The band spans the header row as well as the text columns, so a
        strip tightened to the columns cannot reach a number in the corner.
        """
        pdf = tmp_path / "corner.pdf"
        write_bitonal_page(pdf, header_line=True, corner_number=True)
        page_no = detection(
            Label.PAGE_NUMBER,
            CORNER_NUMBER_X - 2,
            HEADER_LINE_Y - 9,
            CORNER_NUMBER_X + 12,
            HEADER_LINE_Y + 2,
        )
        rects = _rects_for(compute_margin_rects(pdf, pages=[detected_page([*_columns(), page_no])]))
        box = page_no.bbox
        for r in rects:
            overlap_x = min(box.x2, r["x1"]) - max(box.x1, r["x0"])
            overlap_y = min(box.y2, r["y1"]) - max(box.y1, r["y0"])
            assert not (overlap_x > 1 and overlap_y > 1), f"page number covered by margin {r}"
        # ...and a strip is still produced on that side, it just stops short
        # of the number instead of running into it.
        assert [r for r in rects if r["x0"] <= 1 and r["x1"] > 1]

    def test_footer_page_number_does_not_define_the_top(self, tmp_path):
        """A page number below the header row is not the header row.

        Some reporters print it at the foot. Letting one set the top bound
        would put a full-width strip over the body of the page.

        The detection sits a third of the way down rather than at the very
        foot, deliberately: a bound below the measured ink bottom makes the
        tightened box degenerate, and ``_tighten_bounds`` then discards it
        wholesale, so the guard under test is never consulted and the test
        passes either way.
        """
        pdf = tmp_path / "footer.pdf"
        write_bitonal_page(pdf, header_line=True)
        below_header = detection(Label.PAGE_NUMBER, 300, 240, 320, 256)
        assert below_header.bbox.y1 > PAGE_H * 0.25, "not below the header limit"
        assert below_header.bbox.y2 < CONTENT.y1, "must stay above the ink bottom"
        rects = _rects_for(
            compute_margin_rects(pdf, pages=[detected_page([*_columns(), below_header])])
        )
        _left, top, _right, _bottom = _uncovered_box(rects)
        assert top < CONTENT.y0, "a page number below the header defined the top"

    def test_degenerate_band_is_ignored(self, tmp_path):
        """A bogus TEXT_COLUMN box cannot collapse the content box."""
        pdf = tmp_path / "narrow_band.pdf"
        write_bitonal_page(pdf)
        bogus = [detection(Label.TEXT_COLUMN, 300, 300, 320, 320)]
        assert _rects_for(compute_margin_rects(pdf, pages=[detected_page(bogus)])) == _rects_for(
            compute_margin_rects(pdf)
        )

    def test_margins_never_cover_the_text(self, tmp_path):
        """The acceptance property, with every signal in play."""
        pdf = tmp_path / "all.pdf"
        write_bitonal_page(
            pdf,
            header_line=True,
            bleed_mark=True,
            stray_mark=True,
            top_bar=True,
            bottom_bar=True,
        )
        rects = _rects_for(
            compute_margin_rects(pdf, pages=[detected_page([*_columns(), _header()])])
        )
        left, top, right, bottom = _uncovered_box(rects)
        assert left <= CONTENT.x0
        assert top <= CONTENT.y0
        assert right >= CONTENT.x1
        assert bottom >= CONTENT.y1
        # ...while the edge artifacts are all masked.
        assert left > STRAY_MARK.x1
        assert top > max(TOP_BAR.y1, BLEED_MARK.y1)
        assert bottom < BOTTOM_BAR.y0
