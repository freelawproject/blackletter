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

from blackletter.margins import EDGE_BLEED_PT, MARGIN_MIN_KEEP_RATIO, compute_margin_rects
from blackletter.models import BBox, Detection, Label, Page
from tests.pdf_fixtures import (
    BLEED_MARK,
    BOTTOM_BAR,
    CONTENT,
    CORNER_NUMBER_X,
    EDGE_BLOT,
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


def _box(rect: fitz.Rect) -> tuple[float, float, float, float]:
    """A ``fitz.Rect`` as the tuple ``Page.text_box`` takes."""
    return (rect.x0, rect.y0, rect.x1, rect.y1)


def _fitted_page(detections: list, text_box: tuple | None) -> Page:
    """A :func:`detected_page` that also carries the caller's text box.

    The fixture page's image dimensions equal its size in points, so the
    text box reads as points too.
    """
    page = detected_page(detections)
    page.text_box = text_box
    return page


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

    def test_a_band_narrower_than_the_text_does_not_win(self, tmp_path):
        """Found on real pages whose second column went undetected.

        The band is then narrower than the page's own text, and tightening
        to it puts a side strip through the type. Ink that reads as text is
        not the band's to give away, however confident the detections look.
        """
        pdf = tmp_path / "one_column_detected.pdf"
        write_bitonal_page(pdf)
        # Only the left half of the text is claimed by a column box.
        mid = (CONTENT.x0 + CONTENT.x1) / 2
        partial = [detection(Label.TEXT_COLUMN, CONTENT.x0, CONTENT.y0, mid, CONTENT.y1)]
        rects = _rects_for(compute_margin_rects(pdf, pages=[detected_page(partial)]))
        _left, _top, right, _bottom = _uncovered_box(rects)
        assert right >= CONTENT.x1 - 2, "a strip was placed inside the text"

    def test_an_artifact_outside_the_band_is_still_covered(self, tmp_path):
        """The other direction: tightening must still do its job.

        A speck out in the margin widened the ink box, and the strip on that
        side should reach past it rather than stopping short.
        """
        pdf = tmp_path / "stray.pdf"
        write_bitonal_page(pdf, stray_mark=True)
        tight = _uncovered_box(
            _rects_for(compute_margin_rects(pdf, pages=[detected_page(_columns())]))
        )
        assert tight[0] > STRAY_MARK.x1, "the speck was left uncovered"

    def test_detections_from_a_different_page_size_are_ignored(self, tmp_path):
        """A caller's frame that is not this page's would place strips anywhere."""
        pdf = tmp_path / "mismatch.pdf"
        write_bitonal_page(pdf, stray_mark=True)
        page = detected_page(_columns())
        page.pdf_width, page.pdf_height = PAGE_W / 2, PAGE_H / 2
        # The caller's text box arrives in the same frame, so it goes too.
        page.text_box = _box(CONTENT)
        with_bad = _rects_for(compute_margin_rects(pdf, pages=[page]))
        without = _rects_for(compute_margin_rects(pdf))
        assert with_bad == without, "trusted detections in the wrong frame"

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


class TestTextBoxFittedMargins:
    """Margins fitted to the caller's own text box (scanning #323).

    A long blot down the outer edge of a leaf reads as text to the band
    tightening: its pixel columns are about half dark, neither a speck nor
    a bar. Ink takes it into the content box and the band cannot give it
    up, so the strip on that side stops short of it. The caller knows
    where the text is (an OCR pass, a layout model) and says so on
    ``Page.text_box``; the content box is fitted to it, smaller only.
    """

    # The text box is padded by the buffer and the strips leave the same
    # buffer again, so a fitted side sits two buffers off the text.
    SLACK = 2 * BUFFER

    def test_the_blot_defeats_the_band_tightening(self, tmp_path):
        """The premise: today the band cannot give a text-like blot up."""
        pdf = tmp_path / "blot.pdf"
        write_bitonal_page(pdf, edge_blot=True)
        rects = _rects_for(compute_margin_rects(pdf, pages=[detected_page(_columns())]))
        _left, top, right, bottom = _uncovered_box(rects)
        assert right >= EDGE_BLOT.x1, "the band gave up ink that reads as text"
        # ...and the box runs the blot's height, so the top and bottom
        # strips have all but vanished with it (2 pt for ink rounding).
        assert top <= EDGE_BLOT.y0 + 2 and bottom >= EDGE_BLOT.y1 - 2, "blot not in the ink box"

    def test_a_text_box_inside_the_ink_box_places_the_strips(self, tmp_path):
        """With the caller's box the strips land on the text, not the blot."""
        pdf = tmp_path / "blot.pdf"
        clean = tmp_path / "clean.pdf"
        write_bitonal_page(pdf, edge_blot=True)
        write_text_page(clean)
        page = _fitted_page(_columns(), _box(CONTENT))
        rects = _rects_for(compute_margin_rects(pdf, pages=[page]))
        left, top, right, bottom = _uncovered_box(rects)
        assert right == pytest.approx(CONTENT.x1 + self.SLACK, abs=2.0)
        assert bottom == pytest.approx(CONTENT.y1 + self.SLACK, abs=2.0)
        assert top == pytest.approx(CONTENT.y0 - self.SLACK, abs=2.0)
        assert right < EDGE_BLOT.x0, "blot left uncovered"
        assert len(rects) == 4, "a fitted page gets all four strips"
        # ...and the text itself is untouched.
        ink_left, ink_top, ink_right, ink_bottom = _ink_bbox(clean)
        assert left <= ink_left and top <= ink_top
        assert right >= ink_right and bottom >= ink_bottom

    def test_the_fit_needs_no_detections(self, tmp_path):
        """A page with no TEXT_COLUMN still takes the caller's box."""
        pdf = tmp_path / "stray.pdf"
        write_bitonal_page(pdf, stray_mark=True)
        loose = _uncovered_box(_rects_for(compute_margin_rects(pdf, pages=[detected_page([])])))
        assert loose[0] <= STRAY_MARK.x0, "speck should widen the content box"
        fitted = _uncovered_box(
            _rects_for(compute_margin_rects(pdf, pages=[_fitted_page([], _box(CONTENT))]))
        )
        assert fitted[0] == pytest.approx(CONTENT.x0 - self.SLACK, abs=2.0)
        assert fitted[0] > STRAY_MARK.x1, "speck left unmasked"

    def test_a_text_box_larger_than_the_ink_box_moves_nothing(self, tmp_path):
        """Smaller only. None answers exactly as today."""
        pdf = tmp_path / "stray.pdf"
        write_bitonal_page(pdf, stray_mark=True)
        baseline = _rects_for(compute_margin_rects(pdf, pages=[detected_page(_columns())]))
        whole_page = _fitted_page(_columns(), (0.0, 0.0, PAGE_W, PAGE_H))
        assert _rects_for(compute_margin_rects(pdf, pages=[whole_page])) == baseline
        assert detected_page(_columns()).text_box is None
        none = _fitted_page(_columns(), None)
        assert _rects_for(compute_margin_rects(pdf, pages=[none])) == baseline

    def test_a_text_box_below_the_keep_floor_is_refused(self, tmp_path):
        """A partial read describes a fifth of the page; today's answer is the floor."""
        pdf = tmp_path / "blot.pdf"
        write_bitonal_page(pdf, edge_blot=True)
        baseline = _rects_for(compute_margin_rects(pdf, pages=[detected_page(_columns())]))
        # Full width, so only the area floor can refuse it, not the width check.
        partial = fitz.Rect(CONTENT.x0, CONTENT.y0, CONTENT.x1, CONTENT.y0 + 120)
        ink_area = (EDGE_BLOT.x1 - CONTENT.x0) * (EDGE_BLOT.y1 - EDGE_BLOT.y0)
        assert partial.get_area() / ink_area < MARGIN_MIN_KEEP_RATIO
        refused = _fitted_page(_columns(), _box(partial))
        assert _rects_for(compute_margin_rects(pdf, pages=[refused])) == baseline

    def test_the_box_is_held_off_a_page_number_at_the_foot(self, tmp_path):
        """Some reporters print the number at the foot, and no reader describes it."""
        pdf = tmp_path / "plain.pdf"
        write_bitonal_page(pdf)
        footer = detection(Label.PAGE_NUMBER, 300, 650, 320, 665)
        assert footer.bbox.y1 > PAGE_H * 0.25, "must not be read as the header row"
        short = fitz.Rect(CONTENT.x0, CONTENT.y0, CONTENT.x1, 600)
        # No TEXT_COLUMN here: a strip is pulled back off any real detection
        # it covers, and a column box reaching the ink bottom would hide
        # whether the hold did its job.
        page = _fitted_page([footer], _box(short))
        rects = _rects_for(compute_margin_rects(pdf, pages=[page]))
        _left, _top, _right, bottom = _uncovered_box(rects)
        assert bottom >= footer.bbox.y2, "the bottom strip cut into the page number"
        assert bottom < CONTENT.y1, "the fit was refused rather than held"

    def test_the_box_is_held_off_a_corner_page_number(self, tmp_path):
        """The side strips span the header row, so the hold is horizontal too."""
        pdf = tmp_path / "corner.pdf"
        write_bitonal_page(pdf, header_line=True, corner_number=True)
        page_no = detection(
            Label.PAGE_NUMBER,
            CORNER_NUMBER_X - 2,
            HEADER_LINE_Y - 9,
            CORNER_NUMBER_X + 12,
            HEADER_LINE_Y + 2,
        )
        # The caller's reader missed the number: its box spans the body only.
        page = _fitted_page([*_columns(), _header(), page_no], _box(CONTENT))
        rects = _rects_for(compute_margin_rects(pdf, pages=[page]))
        box = page_no.bbox
        for r in rects:
            overlap_x = min(box.x2, r["x1"]) - max(box.x1, r["x0"])
            overlap_y = min(box.y2, r["y1"]) - max(box.y1, r["y0"])
            assert not (overlap_x > 1 and overlap_y > 1), f"page number covered by margin {r}"
        assert [r for r in rects if r["x0"] <= 1 and r["x1"] > 1], "no left strip"

    def test_a_header_detection_in_an_edge_band_does_not_hold_the_box(self, tmp_path):
        """Bleed-through labelled PAGE_NUMBER is what a strip is for."""
        pdf = tmp_path / "bleed.pdf"
        write_bitonal_page(pdf, bleed_mark=True, edge_blot=True)
        top_bleed = detection(
            Label.PAGE_NUMBER, BLEED_MARK.x0, BLEED_MARK.y0, BLEED_MARK.x1, BLEED_MARK.y1
        )
        assert top_bleed.bbox.y2 <= EDGE_BLEED_PT
        bottom_bleed = detection(Label.PAGE_NUMBER, 300, PAGE_H - 15, 320, PAGE_H - 5)
        assert bottom_bleed.bbox.y1 >= PAGE_H - EDGE_BLEED_PT
        page = _fitted_page([*_columns(), top_bleed, bottom_bleed], _box(CONTENT))
        rects = _rects_for(compute_margin_rects(pdf, pages=[page]))
        _left, top, _right, bottom = _uncovered_box(rects)
        assert top == pytest.approx(CONTENT.y0 - self.SLACK, abs=2.0)
        assert bottom == pytest.approx(CONTENT.y1 + self.SLACK, abs=2.0)
        assert top > BLEED_MARK.y1, "bleed left unmasked"

    def test_the_text_box_is_read_in_the_page_pixels(self, tmp_path):
        """The caller's frame is the detection render, not points."""
        pdf = tmp_path / "blot.pdf"
        write_bitonal_page(pdf, edge_blot=True)
        in_points = _fitted_page(_columns(), _box(CONTENT))
        expected = _rects_for(compute_margin_rects(pdf, pages=[in_points]))

        k = 2.0
        scaled = Page(
            index=0,
            pdf_width=PAGE_W,
            pdf_height=PAGE_H,
            img_width=int(PAGE_W * k),
            img_height=int(PAGE_H * k),
            detections=[
                Detection(
                    bbox=BBox(d.bbox.x1 * k, d.bbox.y1 * k, d.bbox.x2 * k, d.bbox.y2 * k),
                    label=d.label,
                    confidence=d.confidence,
                    page_index=0,
                )
                for d in _columns()
            ],
            text_box=(CONTENT.x0 * k, CONTENT.y0 * k, CONTENT.x1 * k, CONTENT.y1 * k),
        )
        assert _rects_for(compute_margin_rects(pdf, pages=[scaled])) == expected
