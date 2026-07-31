"""Tests for the steps a review UI needs the library to expose separately.

``validate`` runs OCR, analysis and issue building in one call, and
``generate`` wants a single combined payload. Both are right for a CLI and
wrong for an application driving a review loop, which has to rebuild
findings after a human edits a page number, and has to assemble the payload
from the library's own outputs. Those consumers were reimplementing all of
it; these are the seams that let them stop.
"""

from __future__ import annotations

import pytest

from blackletter.api import build_redactions
from blackletter.models import Label
from blackletter.process import page_body_covered
from blackletter.validate import build_analysis, build_issues, parse_expected_range
from tests.pdf_fixtures import PAGE_H, PAGE_W, detected_page, detection


def result(pdf_page: int, detected: str, **extra) -> dict:
    """One per-page OCR result, as analyze_pdf reports them.

    ``type`` defaults to ``"single"`` because analyze_pdf always sets it and
    ``build_issues`` reads it directly; pass ``type="range"`` for a page
    printed with a span like "677-685".
    """
    return {"pdf_page": pdf_page, "detected": detected, "type": "single", **extra}


class TestBuildAnalysis:
    def test_clean_sequence_has_nothing_to_report(self):
        results = [result(i, str(100 + i - 1)) for i in range(1, 6)]
        analysis = build_analysis(results, 100, 104)
        assert analysis["seq_issues"] == []
        assert analysis["duplicates"] == {}
        assert analysis["missing_pages"] == []
        assert analysis["all_nums"] == [100, 101, 102, 103, 104]

    def test_duplicate_and_backward_and_gap(self):
        results = [
            result(1, "100"),
            result(2, "100"),  # duplicate
            result(3, "99"),  # backward
            result(4, "110"),  # gap
        ]
        analysis = build_analysis(results)
        kinds = [issue[0] for issue in analysis["seq_issues"]]
        assert kinds == ["DUPLICATE", "BACKWARD", "GAP"]
        gap = next(i for i in analysis["seq_issues"] if i[0] == "GAP")
        assert gap[5] == list(range(100, 110)), "gap should name the missing numbers"

    def test_every_copy_of_a_repeated_number_is_recorded(self):
        results = [result(1, "100"), result(2, "100"), result(3, "100")]
        analysis = build_analysis(results)
        assert analysis["duplicates"] == {100: [1, 2, 3]}

    def test_a_printed_range_covers_its_pages(self):
        """A page printed "677-685" accounts for all of them, not just two."""
        results = [result(1, "676"), result(2, "677-685", type="range"), result(3, "686")]
        analysis = build_analysis(results, 676, 686)
        assert analysis["missing_pages"] == []
        assert len(analysis["ranges_found"]) == 1

    def test_a_range_breaks_the_sequence_chain(self):
        """The number after a range must not be compared to the one before."""
        results = [result(1, "100"), result(2, "101-105", type="range"), result(3, "106")]
        analysis = build_analysis(results)
        assert analysis["seq_issues"] == []

    def test_missing_pages_within_the_expected_range(self):
        results = [result(1, "100"), result(2, "103")]
        analysis = build_analysis(results, 100, 103)
        assert analysis["missing_pages"] == [101, 102]

    def test_the_expected_range_extends_beyond_what_was_detected(self):
        """The known-range branch, which the case above cannot distinguish.

        With results 100 and 103, the range branch and the fallback both
        report [101, 102]. Here the volume is known to run past the last
        detected page, so only the range branch reports the tail.
        """
        results = [result(1, "101"), result(2, "102")]
        assert build_analysis(results, 100, 105)["missing_pages"] == [100, 103, 104, 105]
        assert build_analysis(results)["missing_pages"] == []

    def test_a_two_page_step_is_tolerated(self):
        """The gap rule fires above 2, so one skipped leaf is not an issue."""
        assert build_analysis([result(1, "100"), result(2, "102")])["seq_issues"] == []
        gaps = build_analysis([result(1, "100"), result(2, "103")])["seq_issues"]
        assert [g[0] for g in gaps] == ["GAP"]

    def test_undetected_pages_are_collected(self):
        results = [result(1, "100"), result(2, ""), result(3, "102")]
        analysis = build_analysis(results)
        assert [r["pdf_page"] for r in analysis["not_detected"]] == [2]

    def test_out_of_range_pages_do_not_anchor_the_sequence(self):
        """A page number far outside the volume is a misread, not a jump."""
        results = [result(1, "100"), result(2, "9999"), result(3, "101")]
        analysis = build_analysis(results, 100, 110)
        assert [r["pdf_page"] for r in analysis["out_of_range"]] == [2]
        assert analysis["seq_issues"] == [], "the misread anchored the chain"

    def test_feeds_build_issues(self):
        """The whole point: the two halves still fit together."""
        results = [result(1, "100"), result(2, "103")]
        analysis = build_analysis(results, 100, 103)
        issues = build_issues(analysis, pdf_page_count=2, exp_start=100, exp_end=103)
        assert {i["check_name"] for i in issues["issues"]} == {"missing_page"}
        assert issues["missing_pages"] == [101, 102]
        assert issues["page_map"]

    def test_public_names_are_the_private_ones(self):
        from blackletter import validate as v

        assert v._build_issues is build_issues
        assert v._parse_expected_range is parse_expected_range


class TestBuildRedactions:
    @staticmethod
    def _page(index: int = 0):
        # img dims twice the page size, so scale is 0.5 and the conversion
        # is visible in the output.
        page = detected_page([detection(Label.TEXT_COLUMN, 72, 100, 540, 700)], page_index=index)
        page.img_width = int(PAGE_W * 2)
        page.img_height = int(PAGE_H * 2)
        return page

    def test_redaction_rects_convert_to_points(self):
        payload = build_redactions(
            [self._page()],
            redaction_rects=[
                {
                    "page_index": 0,
                    "rects": [
                        {
                            "x0": 100,
                            "y0": 200,
                            "x1": 300,
                            "y1": 400,
                            "fill": "black",
                            "type": "headnote",
                        }
                    ],
                }
            ],
            margin_rects=[],
            opinions=[],
        )
        (rect,) = payload["pages"]["0"]
        assert (rect["x0"], rect["y0"], rect["x1"], rect["y1"]) == (50.0, 100.0, 150.0, 200.0)
        assert rect["fill"] == "black"
        assert rect["type"] == "headnote"

    def test_margin_rects_are_already_points(self):
        payload = build_redactions(
            [self._page()],
            redaction_rects=[],
            margin_rects=[{"page_index": 0, "rects": [{"x0": 0, "y0": 0, "x1": PAGE_W, "y1": 40}]}],
            opinions=[],
        )
        (rect,) = payload["pages"]["0"]
        assert (rect["x0"], rect["y1"]) == (0, 40)
        assert rect["fill"] == "white"
        assert rect["type"] == "margin"

    def test_margins_come_before_redactions_on_a_page(self):
        """generate paints in order, and a black rect must survive a margin."""
        payload = build_redactions(
            [self._page()],
            redaction_rects=[
                {
                    "page_index": 0,
                    "rects": [
                        {
                            "x0": 100,
                            "y0": 200,
                            "x1": 300,
                            "y1": 400,
                            "fill": "black",
                            "type": "headnote",
                        }
                    ],
                }
            ],
            margin_rects=[{"page_index": 0, "rects": [{"x0": 0, "y0": 0, "x1": PAGE_W, "y1": 40}]}],
            opinions=[],
        )
        assert [r["type"] for r in payload["pages"]["0"]] == ["margin", "headnote"]

    def test_degenerate_rects_are_dropped(self):
        payload = build_redactions(
            [self._page()],
            redaction_rects=[
                {
                    "page_index": 0,
                    "rects": [
                        {
                            "x0": 100,
                            "y0": 200,
                            "x1": 100,
                            "y1": 400,
                            "fill": "black",
                            "type": "headnote",
                        },
                        {
                            "x0": 100,
                            "y0": 400,
                            "x1": 300,
                            "y1": 200,
                            "fill": "black",
                            "type": "headnote",
                        },
                    ],
                }
            ],
            margin_rects=[],
            opinions=[],
        )
        assert payload["pages"]["0"] == []

    def test_filenames_use_the_printed_page_numbers(self):
        opinions = [{"first_page_number": 1, "last_page_number": 27}, {"first_page_number": 28}]
        payload = build_redactions([self._page()], [], [], opinions, reporter="a3d", volume="222")
        assert [op["filename"] for op in payload["opinions"]] == [
            "a3d.222.0001-0027.pdf",
            "a3d.222.0028-0028.pdf",
        ]

    def test_no_reporter_means_no_filename(self):
        opinions = [{"first_page_number": 1, "last_page_number": 27}]
        payload = build_redactions([self._page()], [], [], opinions)
        assert "filename" not in payload["opinions"][0]

    def test_a_page_with_no_rects_at_all_is_absent(self):
        """Pinned because a consumer iterating pages needs to know."""
        payload = build_redactions([self._page()], [], [], [])
        assert payload["pages"] == {}

    def test_page_keys_are_strings_in_order(self):
        pages = [self._page(index) for index in (0, 1, 2)]
        rects = [
            {"page_index": 2, "rects": []},
            {"page_index": 0, "rects": []},
            {"page_index": 1, "rects": []},
        ]
        payload = build_redactions(pages, rects, [], [])
        assert list(payload["pages"]) == ["0", "1", "2"]

    def test_rects_for_an_unknown_page_are_an_error(self):
        """Silence here would mean pixel coordinates read as points.

        The scale factors come from the page's own image dimensions, so a
        page the caller did not pass has nothing to convert its rects by,
        and guessing would be wrong by whatever the render resolution was.
        """
        rects = [
            {
                "page_index": 7,
                "rects": [
                    {"x0": 10, "y0": 20, "x1": 30, "y1": 40, "fill": "black", "type": "headnote"}
                ],
            }
        ]
        with pytest.raises(KeyError, match="page 7"):
            build_redactions([self._page()], rects, [], [])

    def test_page_without_image_dimensions_is_left_in_points(self):
        """A page with no detections has no pixel geometry to scale by."""
        page = detected_page([])
        page.img_width = 1
        page.img_height = 1
        payload = build_redactions(
            [page],
            [
                {
                    "page_index": 0,
                    "rects": [
                        {
                            "x0": 10,
                            "y0": 20,
                            "x1": 30,
                            "y1": 40,
                            "fill": "black",
                            "type": "headnote",
                        }
                    ],
                }
            ],
            [],
            [],
        )
        (rect,) = payload["pages"]["0"]
        assert (rect["x0"], rect["y1"]) == (10, 40)


class TestPageBodyCovered:
    """Replaces "extract the text and see if there is any", which reports
    every page empty once the pipeline stops embedding a text layer."""

    def test_a_fully_covered_body_reads_as_covered(self):
        rects = [{"x0": 0, "y0": 0, "x1": PAGE_W, "y1": PAGE_H}]
        assert page_body_covered(rects, PAGE_W, PAGE_H) is True

    def test_a_live_page_does_not(self):
        rects = [{"x0": 0, "y0": 0, "x1": PAGE_W, "y1": 100}]
        assert page_body_covered(rects, PAGE_W, PAGE_H) is False

    def test_no_rects_is_not_covered(self):
        assert page_body_covered([], PAGE_W, PAGE_H) is False

    def test_the_header_band_is_ignored(self):
        """The printed page number lives there and is never redacted.

        A rect covering everything below y=60 is 92% of the page, so with a
        small band it clears the default threshold either way. The band has
        to be large enough that counting it would drop the fraction below
        the threshold, or this proves nothing.
        """
        rects = [{"x0": 0, "y0": 100, "x1": PAGE_W, "y1": PAGE_H}]
        assert (PAGE_H - 100) / PAGE_H < 0.9, "the band cannot affect the outcome"
        assert page_body_covered(rects, PAGE_W, PAGE_H, header_height=100.0) is True
        assert page_body_covered(rects, PAGE_W, PAGE_H, header_height=0.0) is False

    def test_the_defaults_are_what_they_claim(self):
        """Pin the default band and threshold, which nothing else does."""
        rects = [{"x0": 0, "y0": 60, "x1": PAGE_W, "y1": PAGE_H * 0.85}]
        assert page_body_covered(rects, PAGE_W, PAGE_H) is False
        assert page_body_covered(rects, PAGE_W, PAGE_H, min_coverage=0.8) is True

    def test_several_rects_add_up(self):
        rects = [
            {"x0": 0, "y0": 0, "x1": PAGE_W / 2, "y1": PAGE_H},
            {"x0": PAGE_W / 2, "y0": 0, "x1": PAGE_W, "y1": PAGE_H},
        ]
        assert page_body_covered(rects, PAGE_W, PAGE_H) is True

    def test_overlapping_rects_are_not_double_counted(self):
        rects = [
            {"x0": 0, "y0": 0, "x1": PAGE_W, "y1": PAGE_H / 2},
            {"x0": 0, "y0": 0, "x1": PAGE_W, "y1": PAGE_H / 2},
        ]
        assert page_body_covered(rects, PAGE_W, PAGE_H) is False

    def test_coverage_threshold_is_respected(self):
        rects = [{"x0": 0, "y0": 60, "x1": PAGE_W, "y1": PAGE_H * 0.75}]
        assert page_body_covered(rects, PAGE_W, PAGE_H, min_coverage=0.9) is False
        assert page_body_covered(rects, PAGE_W, PAGE_H, min_coverage=0.5) is True

    def test_a_degenerate_page_is_not_covered(self):
        assert page_body_covered([{"x0": 0, "y0": 0, "x1": 1, "y1": 1}], 0, 0) is False
        assert page_body_covered([{"x0": 0, "y0": 0, "x1": 1, "y1": 1}], PAGE_W, 10) is False

    def test_rects_reaching_past_the_page_are_clamped(self):
        rects = [{"x0": -500, "y0": -500, "x1": PAGE_W + 500, "y1": PAGE_H + 500}]
        assert page_body_covered(rects, PAGE_W, PAGE_H) is True

    def test_margin_strips_count_toward_coverage(self):
        """Every rect removes content, not just the headnote blackouts."""
        rects = [
            {"x0": 0, "y0": 0, "x1": 60, "y1": PAGE_H},
            {"x0": 60, "y0": 0, "x1": PAGE_W, "y1": PAGE_H},
        ]
        assert page_body_covered(rects, PAGE_W, PAGE_H) is True


@pytest.mark.parametrize(
    "name,expected",
    [
        ("a3d.222.1.30.pdf", (1, 30)),
        ("f3d.952.100.200.pdf", (100, 200)),
    ],
)
def test_parse_expected_range_is_public(tmp_path, name, expected):
    pdf = tmp_path / name
    pdf.write_bytes(b"%PDF-1.4\n")
    assert parse_expected_range(pdf) == expected


class TestProcessExposesTheFlags:
    """``process()`` is the documented Python entry point.

    The README tells a Python caller to pass ``--text-layer`` for searchable
    output, and CHANGES tells one who wants the old pre-pass back to pass
    ``ocr=True``. Both go through ``cmd_process``, which reads them off an
    ``argparse.Namespace`` that ``process()`` builds by hand, so a flag left
    out of that namespace silently does nothing.
    """

    def test_both_flags_reach_the_namespace(self, monkeypatch):
        import argparse

        import blackletter.process as bl

        seen: dict[str, argparse.Namespace] = {}
        monkeypatch.setattr(bl, "cmd_process", lambda args: seen.setdefault("args", args))
        monkeypatch.setattr(bl, "_build_output_dir", lambda args: args.output)

        bl.process("in.pdf", "out", text_layer=True, ocr=True)

        assert seen["args"].text_layer is True
        assert seen["args"].ocr is True

    def test_they_default_to_off(self, monkeypatch):
        import argparse

        import blackletter.process as bl

        seen: dict[str, argparse.Namespace] = {}
        monkeypatch.setattr(bl, "cmd_process", lambda args: seen.setdefault("args", args))
        monkeypatch.setattr(bl, "_build_output_dir", lambda args: args.output)

        bl.process("in.pdf", "out")

        assert seen["args"].text_layer is False
        assert seen["args"].ocr is False
