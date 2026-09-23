"""Regression tests for page_map duplicate detection in validate._build_issues.

Guards against unnumbered front matter (cover, table of contents) "stealing"
logical page numbers and flagging the real, numbered pages as duplicates
(#52; surfaced in the scanning portal, freelawproject/scanning#100).

Also covers #55: every copy of a repeated page number is flagged
``duplicate`` (matching the duplicate_page issue's page list), while
missing-page placeholders still anchor on the first copy.

Also covers #83: a missing-page placeholder anchors only on pages that
carry a real printed number, so unnumbered front matter, whose logical
number is its PDF page number, does not pull a gap near the start of a
volume into the front matter.
"""

from blackletter.validate import _build_issues, build_analysis, build_issues


def _analysis(results, duplicates=None, seen_nums=None):
    """Build a minimal analysis dict for _build_issues from OCR results.

    :param results: List of per-page OCR result dicts.
    :param duplicates: Optional coverage-duplicates mapping.
    :param seen_nums: Optional detected-number to pdf-pages mapping.
    :returns: Analysis dict accepted by ``_build_issues``.
    :rtype: dict
    """
    return {
        "results": results,
        "seq_issues": [],
        "duplicates": duplicates or {},
        "seen_nums": seen_nums or {},
        "all_nums": sorted(seen_nums or {}),
        "missing_pages": [],
        "ranges_found": [],
        "not_detected": [r for r in results if not r["detected"]],
        "out_of_range": [],
    }


class TestPageMapDuplicates:
    def test_front_matter_does_not_flag_real_pages_as_duplicate(self):
        # PDF pages 1-3 are unnumbered front matter; pages 4-6 are the real
        # opinion pages numbered 1-3. The front matter must not steal 1-3.
        results = [
            {"pdf_page": 1, "detected": None, "type": None},
            {"pdf_page": 2, "detected": None, "type": None},
            {"pdf_page": 3, "detected": None, "type": None},
            {"pdf_page": 4, "detected": "1", "type": "single"},
            {"pdf_page": 5, "detected": "2", "type": "single"},
            {"pdf_page": 6, "detected": "3", "type": "single"},
        ]
        analysis = _analysis(results, seen_nums={1: [4], 2: [5], 3: [6]})

        page_map = _build_issues(analysis, 6, 1, 3)["page_map"]

        flagged = [e["pdf_index"] for e in page_map if e.get("duplicate")]
        assert flagged == []

    def test_all_copies_of_a_duplicate_are_flagged(self):
        # Two pages both genuinely read "2"; every copy is flagged (not just
        # the later one), matching the duplicate_page issue's page list (#55).
        results = [
            {"pdf_page": 1, "detected": None, "type": None},
            {"pdf_page": 2, "detected": "1", "type": "single"},
            {"pdf_page": 3, "detected": "2", "type": "single"},
            {"pdf_page": 4, "detected": "2", "type": "single"},
            {"pdf_page": 5, "detected": "3", "type": "single"},
        ]
        analysis = _analysis(
            results,
            duplicates={2: [3, 4]},
            seen_nums={1: [2], 2: [3, 4], 3: [5]},
        )

        page_map = _build_issues(analysis, 5, 1, 3)["page_map"]

        flagged = [e["pdf_index"] for e in page_map if e.get("duplicate")]
        assert flagged == [2, 3]  # both "2" pages: pdf_page 3 and 4

    def test_missing_page_anchors_before_first_duplicate_copy(self):
        # Page 3 is missing and page 4 is duplicated. The missing-3 placeholder
        # must still land before the FIRST copy of "4" (the reliable anchor),
        # even though every copy of "4" is now flagged duplicate (#55).
        results = [
            {"pdf_page": 1, "detected": "1", "type": "single"},
            {"pdf_page": 2, "detected": "2", "type": "single"},
            {"pdf_page": 3, "detected": "4", "type": "single"},
            {"pdf_page": 4, "detected": "4", "type": "single"},
        ]
        analysis = _analysis(
            results,
            duplicates={4: [3, 4]},
            seen_nums={1: [1], 2: [2], 4: [3, 4]},
        )
        analysis["missing_pages"] = [3]

        page_map = _build_issues(analysis, 4, 1, 4)["page_map"]

        flagged = [e["pdf_index"] for e in page_map if e.get("duplicate")]
        assert flagged == [2, 3]  # both "4" copies flagged

        # The missing-3 placeholder sits before the first "4" (pdf_index 2).
        missing_pos = next(
            i for i, e in enumerate(page_map) if e["type"] == "missing" and e["logical_number"] == 3
        )
        first_four_pos = next(i for i, e in enumerate(page_map) if e.get("pdf_index") == 2)
        assert missing_pos == first_four_pos - 1


def _page(pdf_page, detected=""):
    """Build one OCR result dict, unnumbered when ``detected`` is empty.

    :param pdf_page: 1-based PDF page number.
    :param detected: Printed page number read on the page, or ``""``.
    :returns: OCR result dict accepted by ``build_analysis``.
    :rtype: dict
    """
    return {"pdf_page": pdf_page, "detected": detected, "type": "single"}


def _layout(page_map):
    """Reduce a page_map to its order: PDF indices and missing numbers.

    :param page_map: The ``page_map`` returned by ``build_issues``.
    :returns: ``("pdf", pdf_index)`` or ``("missing", number)`` per entry.
    :rtype: list[tuple[str, int]]
    """
    return [
        ("missing", e["logical_number"]) if e["type"] == "missing" else ("pdf", e["pdf_index"])
        for e in page_map
    ]


class TestMissingPlaceholderPlacement:
    def test_gap_after_long_front_matter_is_placed_at_the_gap(self):
        # Scan 3156 of the scanning app: 13 unnumbered pages, then 1-9 on PDF
        # pages 14-22 and 12-20 from PDF page 23. The placeholders for 10 and
        # 11 used to land after PDF pages 10 and 11, in the front matter.
        results = [_page(p) for p in range(1, 14)]
        numbers = list(range(1, 10)) + list(range(12, 21))
        results += [_page(14 + i, str(n)) for i, n in enumerate(numbers)]

        result = build_issues(build_analysis(results), len(results))
        page_map = result["page_map"]

        assert result["missing_pages"] == [10, 11]
        placed = [
            (e["logical_number"], page_map[i - 1].get("pdf_index"))
            for i, e in enumerate(page_map)
            if e["type"] == "missing"
        ]
        assert placed == [(10, 21), (11, None)]
        after = next(i for i, e in enumerate(page_map) if e["type"] == "missing") + 2
        assert page_map[after]["pdf_index"] == 22  # the page that carries 12

    def test_unnumbered_page_inside_the_gap_stays_before_the_placeholders(self):
        # 8, 9, unnumbered, 12: the unnumbered page's fallback number (its
        # PDF page, 3) is above neither gap number, but a larger PDF page
        # would be; either way the placeholders go right before 12.
        for leading in (0, 20):
            results = [_page(p) for p in range(1, leading + 1)]
            results += [
                _page(leading + 1, "8"),
                _page(leading + 2, "9"),
                _page(leading + 3),
                _page(leading + 4, "12"),
            ]
            page_map = build_issues(build_analysis(results), len(results))["page_map"]

            assert _layout(page_map)[leading:] == [
                ("pdf", leading),
                ("pdf", leading + 1),
                ("pdf", leading + 2),
                ("missing", 10),
                ("missing", 11),
                ("pdf", leading + 3),
            ]

    def test_volume_without_front_matter_is_unchanged(self):
        results = [_page(i, str(n)) for i, n in enumerate((1, 2, 4, 5), start=1)]

        page_map = build_issues(build_analysis(results), len(results))["page_map"]

        assert page_map == [
            {"type": "pdf_page", "pdf_index": 0, "logical_number": 1},
            {"type": "pdf_page", "pdf_index": 1, "logical_number": 2},
            {"type": "missing", "logical_number": 3},
            {"type": "pdf_page", "pdf_index": 2, "logical_number": 4},
            {"type": "pdf_page", "pdf_index": 3, "logical_number": 5},
        ]

    def test_numbers_missing_before_the_first_printed_number_follow_front_matter(self):
        # Expected pages 1-5, but the first printed number read is 3: the
        # placeholders for 1 and 2 go after the front matter, before 3.
        results = [_page(1), _page(2), _page(3), _page(4, "3"), _page(5, "4"), _page(6, "5")]

        page_map = build_issues(build_analysis(results, 1, 5), len(results), 1, 5)["page_map"]

        assert _layout(page_map) == [
            ("pdf", 0),
            ("pdf", 1),
            ("pdf", 2),
            ("missing", 1),
            ("missing", 2),
            ("pdf", 3),
            ("pdf", 4),
            ("pdf", 5),
        ]

    def test_out_of_range_reading_does_not_anchor_a_placeholder(self):
        # A stray "900" read on PDF page 6, inside the gap, is out of range.
        # Its fallback number (6) is above both gap numbers, so under the old
        # rule it anchored the placeholders in the middle of the gap's
        # unnumbered pages; they belong before the page that carries 5.
        results = [
            _page(1, "1"),
            _page(2, "2"),
            _page(3),
            _page(4),
            _page(5),
            _page(6, "900"),
            _page(7, "5"),
        ]

        page_map = build_issues(build_analysis(results, 1, 5), len(results), 1, 5)["page_map"]

        assert _layout(page_map) == [
            ("pdf", 0),
            ("pdf", 1),
            ("pdf", 2),
            ("pdf", 3),
            ("pdf", 4),
            ("pdf", 5),
            ("missing", 3),
            ("missing", 4),
            ("pdf", 6),
        ]

    def test_gap_past_the_last_printed_number_precedes_an_unnumbered_tail(self):
        # Printed 1-5, then unnumbered appendix pages, and the volume should
        # end at 7. No page printed above 6 exists, so the placeholders go
        # right after the page that carries 5, not after the whole appendix.
        results = [_page(p, str(p)) for p in range(1, 6)]
        results += [_page(p) for p in range(6, 51)]

        page_map = build_issues(build_analysis(results, 1, 7), len(results), 1, 7)["page_map"]

        layout = _layout(page_map)
        assert layout[4:7] == [("pdf", 4), ("missing", 6), ("missing", 7)]
        assert layout[7:] == [("pdf", i) for i in range(5, 50)]

    def test_range_page_anchors_by_its_first_number(self):
        # Printed 1, 2, then a page labelled "4-6", then 7: page 3 is missing
        # and goes before the range page, also behind front matter.
        for leading in (0, 5):
            results = [_page(p) for p in range(1, leading + 1)]
            results += [
                _page(leading + 1, "1"),
                _page(leading + 2, "2"),
                {"pdf_page": leading + 3, "detected": "4-6", "type": "range"},
                _page(leading + 4, "7"),
            ]
            page_map = build_issues(build_analysis(results), len(results))["page_map"]

            assert _layout(page_map)[leading:] == [
                ("pdf", leading),
                ("pdf", leading + 1),
                ("missing", 3),
                ("pdf", leading + 2),
                ("pdf", leading + 3),
            ]

    def test_range_page_anchors_a_gap_past_it_by_its_last_number(self):
        # Printed 1, 2, then "3-5" and an unnumbered tail; the volume should
        # end at 6. The placeholder follows the range page.
        results = [
            _page(1, "1"),
            _page(2, "2"),
            {"pdf_page": 3, "detected": "3-5", "type": "range"},
            _page(4),
            _page(5),
        ]

        page_map = build_issues(build_analysis(results, 1, 6), len(results), 1, 6)["page_map"]

        assert _layout(page_map) == [
            ("pdf", 0),
            ("pdf", 1),
            ("pdf", 2),
            ("missing", 6),
            ("pdf", 3),
            ("pdf", 4),
        ]

    def test_number_without_a_type_is_a_printed_number(self):
        # ``build_analysis`` counts a number whose ``type`` is not set; the
        # page map treats it as printed too, so it anchors the gap before it.
        results = [
            _page(1, "1"),
            _page(2, "2"),
            {"pdf_page": 3, "detected": "4", "type": None},
            _page(4, "5"),
        ]

        page_map = build_issues(build_analysis(results), len(results))["page_map"]

        assert _layout(page_map) == [
            ("pdf", 0),
            ("pdf", 1),
            ("missing", 3),
            ("pdf", 2),
            ("pdf", 3),
        ]
        assert page_map[3]["logical_number"] == 4

    def test_gap_past_a_duplicated_last_number_follows_the_last_copy(self):
        # Printed 1, 2, 3, 3 and the volume should end at 5: nothing is
        # printed above the gap, so 4 and 5 follow the second "3", not sit
        # between the two copies.
        results = [_page(1, "1"), _page(2, "2"), _page(3, "3"), _page(4, "3")]

        page_map = build_issues(build_analysis(results, 1, 5), len(results), 1, 5)["page_map"]

        assert _layout(page_map) == [
            ("pdf", 0),
            ("pdf", 1),
            ("pdf", 2),
            ("pdf", 3),
            ("missing", 4),
            ("missing", 5),
        ]

    def test_gap_past_a_duplicate_before_an_unnumbered_tail_follows_the_last_copy(self):
        results = [_page(1, "1"), _page(2, "2"), _page(3, "2"), _page(4), _page(5)]

        page_map = build_issues(build_analysis(results, 1, 3), len(results), 1, 3)["page_map"]

        assert _layout(page_map) == [
            ("pdf", 0),
            ("pdf", 1),
            ("pdf", 2),
            ("missing", 3),
            ("pdf", 3),
            ("pdf", 4),
        ]

    def test_inverted_range_label_does_not_anchor(self):
        # "6-4" is a misread: it covers no page, so 3-6 are all missing and
        # all sit together before 7, after the misread page.
        results = [
            _page(1, "1"),
            _page(2, "2"),
            {"pdf_page": 3, "detected": "6-4", "type": "range"},
            _page(4, "7"),
        ]

        page_map = build_issues(build_analysis(results), len(results))["page_map"]

        assert _layout(page_map) == [
            ("pdf", 0),
            ("pdf", 1),
            ("pdf", 2),
            ("missing", 3),
            ("missing", 4),
            ("missing", 5),
            ("missing", 6),
            ("pdf", 3),
        ]
