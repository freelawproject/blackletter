"""Regression tests for page_map duplicate detection in validate._build_issues.

Guards issue #90: unnumbered front matter (cover, table of contents) must not
"steal" logical page numbers and flag the real, numbered pages as duplicates.
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

    def test_genuine_duplicate_is_still_flagged(self):
        # Two pages both genuinely read "2" -> the second is a real duplicate.
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
        assert flagged == [3]  # only the second "2", pdf_page 4 (index 3)
