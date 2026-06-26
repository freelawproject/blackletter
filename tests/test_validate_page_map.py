"""Regression tests for page_map duplicate detection in validate._build_issues.

Guards against unnumbered front matter (cover, table of contents) "stealing"
logical page numbers and flagging the real, numbered pages as duplicates
(#52; surfaced in the scanning portal, freelawproject/scanning#100).

Also covers #55: every copy of a repeated page number is flagged
``duplicate`` (matching the duplicate_page issue's page list), while
missing-page placeholders still anchor on the first copy.
"""

from blackletter.validate import _build_issues


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
