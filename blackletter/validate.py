"""Validate a PDF's page integrity: detect missing, duplicate, and misnumbered pages.

Wraps analyze.py's raw OCR results with issue detection, auto-correction,
gap collapsing, and page map construction for viewer display.
"""

from __future__ import annotations

import itertools
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import fitz

from .analyze import analyze_pdf
from .process import _infer_from_filename

RANGE_RE = re.compile(r"^(\d{1,4})\s*[–\-]\s*(\d{1,4})$")


def _parse_expected_range(pdf_path: str | Path) -> tuple[int | None, int | None]:
    """Extract expected first/last page numbers from filename.

    :param pdf_path: Path to the PDF file.
    :returns: Tuple of (first_page, last_page), either may be ``None``.
    """
    info = _infer_from_filename(Path(pdf_path))
    if info:
        return info.get("first_page"), info.get("last_page")
    return None, None


def _split_in_out_of_range(
    results: list[dict],
    exp_start: int | None,
    exp_end: int | None,
) -> tuple[list[dict], dict[int, list[int]]]:
    """Separate OCR results into out-of-range and in-range buckets.

    :param results: List of per-page OCR result dicts.
    :param exp_start: Expected first page number, or ``None``.
    :param exp_end: Expected last page number, or ``None``.
    :returns: Tuple of (out_of_range_results, seen_nums_dict).
    """
    out_of_range: list[dict] = []
    seen_nums: dict[int, list[int]] = {}
    for r in results:
        if not r["detected"] or r.get("type") == "range":
            continue
        try:
            num = int(r["detected"])
        except ValueError:
            continue
        if num < 1:
            out_of_range.append(r)
        elif exp_start is not None and exp_end is not None:
            if num < exp_start - 5 or num > exp_end + 5:
                out_of_range.append(r)
            else:
                seen_nums.setdefault(num, []).append(r["pdf_page"])
        else:
            seen_nums.setdefault(num, []).append(r["pdf_page"])
    return out_of_range, seen_nums


def _auto_correct(
    ocr_results: list[dict],
    out_of_range: list[dict],
    seen_nums: dict[int, list[int]],
) -> tuple[list[dict], list[tuple[int, str, str]]]:
    """Attempt to fix out-of-range readings via neighbor interpolation.

    If a consistent offset is detected (e.g. all bad readings are +800 from
    expected), apply the correction and return updated results.

    :param ocr_results: Full list of per-page OCR result dicts.
    :param out_of_range: Subset of results whose detected numbers fall
        outside the expected range.
    :param seen_nums: Mapping of detected page number to list of PDF pages
        where it was seen.
    :returns: Tuple of (updated_ocr_results, list_of_corrections) where
        each correction is (pdf_page, old_value, new_value).
    """
    corrections: list[tuple[int, str, str]] = []
    if not out_of_range or not seen_nums:
        return ocr_results, corrections

    in_range_by_page = {p: num for num, pages in seen_nums.items() for p in pages}
    in_range_sorted = sorted(in_range_by_page.items())

    offsets: dict[int, tuple[int, int, int]] = {}
    for r in out_of_range:
        p, detected = r["pdf_page"], int(r["detected"])
        before = [(pp, n) for pp, n in in_range_sorted if pp < p]
        after = [(pp, n) for pp, n in in_range_sorted if pp > p]
        if before and after:
            pp_b, n_b = before[-1]
            pp_a, n_a = after[0]
            expected = round(n_b + (n_a - n_b) / max(pp_a - pp_b, 1) * (p - pp_b))
        elif before:
            pp_b, n_b = before[-1]
            expected = n_b + (p - pp_b)
        elif after:
            pp_a, n_a = after[0]
            expected = n_a - (pp_a - p)
        else:
            continue
        offsets[p] = (detected, expected, expected - detected)

    if not offsets:
        return ocr_results, corrections

    offset_vals = [v[2] for v in offsets.values()]
    modal_offset, modal_count = Counter(offset_vals).most_common(1)[0]
    if modal_count < len(offset_vals) * 0.5 or modal_offset == 0:
        return ocr_results, corrections

    to_fix = {p for p, (d, e, o) in offsets.items() if o == modal_offset}
    new_results = []
    for r in ocr_results:
        if r["pdf_page"] in to_fix and r.get("detected"):
            old_val = r["detected"]
            r = dict(r)
            r["detected"] = str(int(old_val) + modal_offset)
            corrections.append((r["pdf_page"], old_val, r["detected"]))
        new_results.append(r)
    return new_results, corrections


def _build_issues(
    analysis: dict,
    pdf_page_count: int,
    exp_start: int | None = None,
    exp_end: int | None = None,
) -> dict:
    """Convert raw analysis into issues, page_map, and missing_pages.

    This is the core validation logic. Sequence issues become structured
    issue dicts, large gaps are collapsed into warnings, and a page_map
    is built for viewer display.

    :param analysis: Dict of filtered OCR analysis data (results,
        seq_issues, duplicates, missing_pages, etc.).
    :param pdf_page_count: Total number of pages in the PDF.
    :param exp_start: Expected first page number, or ``None``.
    :param exp_end: Expected last page number, or ``None``.
    :returns: Dict with keys ``page_count``, ``page_map``,
        ``missing_pages``, ``issues``, and ``ocr_results``.
    """
    issues: list[dict] = []

    all_nums = analysis.get("all_nums", [])

    # --- Mislabeled document detection ---
    if exp_start is not None and exp_end is not None and all_nums:
        detected_max = max(all_nums)
        expected_count = exp_end - exp_start + 1
        if detected_max < exp_end - max(10, expected_count * 0.1):
            issues.append(
                {
                    "page_number": None,
                    "check_name": "mislabeled_document",
                    "severity": "warning",
                    "message": (
                        f"Filename suggests pages {exp_start}–{exp_end} but page numbers "
                        f"only go up to {detected_max}. Document may be mislabeled or "
                        f"missing a large section."
                    ),
                }
            )

    # --- Sequence issues ---
    for seq in analysis["seq_issues"]:
        issue_type = seq[0]
        if issue_type == "GAP":
            _type, pdf_page, num, prev_pdf, prev_num, gap = seq
            if len(gap) > 6:
                issues.append(
                    {
                        "page_number": prev_num,
                        "check_name": "large_gap",
                        "severity": "warning",
                        "message": (
                            f"Large jump from page {prev_num} to {num} ({len(gap)} pages) "
                            f", likely an OCR misread. Correct the page number and "
                            f"recalculate."
                        ),
                    }
                )
            else:
                for gap_num in gap:
                    if gap_num < 1:
                        continue
                    issues.append(
                        {
                            "page_number": gap_num,
                            "check_name": "missing_page",
                            "severity": "error",
                            "message": (
                                f"Page {gap_num} appears missing "
                                f"(gap between {prev_num} and {num})."
                            ),
                        }
                    )
        elif issue_type == "DUPLICATE":
            _type, pdf_page, num, prev_pdf, prev_num = seq
            issues.append(
                {
                    "page_number": num,
                    "check_name": "duplicate_page",
                    "severity": "warning",
                    "message": (
                        f"Page number {num} appears on both PDF page {prev_pdf} and {pdf_page}."
                    ),
                }
            )
        elif issue_type == "BACKWARD":
            _type, pdf_page, num, prev_pdf, prev_num = seq
            issues.append(
                {
                    "page_number": num,
                    "check_name": "backward_page",
                    "severity": "warning",
                    "message": (
                        f"Page {num} (PDF page {pdf_page}) goes backward "
                        f"from {prev_num} (PDF page {prev_pdf})."
                    ),
                }
            )

    # --- Out-of-range readings ---
    for r in analysis.get("out_of_range", []):
        issues.append(
            {
                "page_number": r["pdf_page"],
                "check_name": "suspicious_reading",
                "severity": "warning",
                "message": (
                    f"PDF page {r['pdf_page']} detected as '{r['detected']}' which is "
                    f"outside the expected page range, likely a stray number."
                ),
            }
        )

    # --- Undetected pages ---
    for r in analysis["not_detected"]:
        issues.append(
            {
                "page_number": r["pdf_page"],
                "check_name": "no_page_number",
                "severity": "info",
                "message": f"No page number detected on PDF page {r['pdf_page']}.",
            }
        )

    # --- Duplicate page numbers (from coverage, not just sequence) ---
    for num, pdf_pages in analysis["duplicates"].items():
        already = [i["page_number"] for i in issues if i["check_name"] == "duplicate_page"]
        if num not in already:
            issues.append(
                {
                    "page_number": num,
                    "check_name": "duplicate_page",
                    "severity": "warning",
                    "message": f"Page number {num} found on PDF pages {pdf_pages}.",
                }
            )

    # --- Range coverage ---
    range_covered: set[int] = set()
    for r in analysis.get("ranges_found", []):
        text = r["detected"].replace("\u2013", "-")
        m = RANGE_RE.match(text)
        if m:
            rs, re_ = int(m.group(1)), int(m.group(2))
            for pg in range(rs, re_ + 1):
                range_covered.add(pg)
            issues.append(
                {
                    "page_number": rs,
                    "check_name": "page_range",
                    "severity": "warning",
                    "message": (
                        f"PDF page {r['pdf_page']} covers page range "
                        f"{r['detected']}. Verify this is expected."
                    ),
                }
            )

    # --- Missing pages (collapse large runs >6 into warnings) ---
    actually_missing = [p for p in analysis["missing_pages"] if p not in range_covered and p >= 1]
    already_flagged = {i["page_number"] for i in issues if i["check_name"] == "missing_page"}

    large_gap_pages: set[int] = set()
    if actually_missing:
        for _, g in itertools.groupby(enumerate(actually_missing), lambda x: x[0] - x[1]):
            run = [v for _, v in g]
            if len(run) > 6:
                large_gap_pages.update(run)
                if run[0] not in already_flagged:
                    issues.append(
                        {
                            "page_number": run[0],
                            "check_name": "large_gap",
                            "severity": "warning",
                            "message": (
                                f"Large gap: pages {run[0]}–{run[-1]} ({len(run)} pages) "
                                f"not found, likely an OCR misread rather than genuinely "
                                f"missing pages."
                            ),
                        }
                    )
            else:
                for gap_num in run:
                    if gap_num not in already_flagged:
                        issues.append(
                            {
                                "page_number": gap_num,
                                "check_name": "missing_page",
                                "severity": "error",
                                "message": f"Page {gap_num} is missing from the document.",
                            }
                        )

    actually_missing = [p for p in actually_missing if p not in large_gap_pages]

    # --- Build page map ---
    page_map: list[dict] = []
    out_of_range_pages = {r["pdf_page"] for r in analysis.get("out_of_range", [])}
    seen_logical: dict[int, int] = {}

    for r in analysis["results"]:
        pdf_idx = r["pdf_page"] - 1
        if r["pdf_page"] in out_of_range_pages:
            logical = r["pdf_page"]
        elif r["detected"] and r["type"] == "single":
            try:
                logical = int(r["detected"])
            except ValueError:
                logical = r["pdf_page"]
        elif r["detected"] and r["type"] == "range":
            logical = r["pdf_page"]
            page_map.append(
                {
                    "type": "pdf_page",
                    "pdf_index": pdf_idx,
                    "logical_number": logical,
                    "range_label": r["detected"],
                }
            )
            continue
        else:
            logical = r["pdf_page"]

        is_dupe = logical in seen_logical
        entry: dict = {
            "type": "pdf_page",
            "pdf_index": pdf_idx,
            "logical_number": logical,
        }
        if is_dupe:
            entry["duplicate"] = True
        seen_logical[logical] = pdf_idx
        page_map.append(entry)

    # Insert missing page placeholders
    if actually_missing and all_nums:
        inserts = []
        for gap_num in actually_missing:
            insert_pos = len(page_map)
            for i, entry in enumerate(page_map):
                if entry["logical_number"] > gap_num and not entry.get("duplicate"):
                    insert_pos = i
                    break
            inserts.append((insert_pos, gap_num))

        for pos, gap_num in reversed(inserts):
            page_map.insert(pos, {"type": "missing", "logical_number": gap_num})

    return {
        "page_count": pdf_page_count,
        "page_map": page_map,
        "missing_pages": actually_missing,
        "issues": issues,
        "ocr_results": analysis["results"],
    }


def _structural_checks(file_path: str | Path) -> list[dict]:
    """Quick PyMuPDF checks for blank pages and orientation changes.

    :param file_path: Path to the PDF file.
    :returns: List of issue dicts for any blank pages or orientation changes.
    """
    doc = fitz.open(str(file_path))
    issues: list[dict] = []
    first_is_landscape = None

    for i in range(len(doc)):
        page = doc.load_page(i)
        display_num = i + 1
        text = page.get_text("text").strip()
        images = page.get_images(full=True)

        if not text and not images:
            issues.append(
                {
                    "page_number": display_num,
                    "check_name": "blank_page",
                    "severity": "error",
                    "message": f"Page {display_num} appears blank (no text or images).",
                }
            )

        rect = page.rect
        is_landscape = rect.width > rect.height
        if first_is_landscape is None:
            first_is_landscape = is_landscape
        elif is_landscape != first_is_landscape:
            orientation = "landscape" if is_landscape else "portrait"
            issues.append(
                {
                    "page_number": display_num,
                    "check_name": "orientation",
                    "severity": "info",
                    "message": f"Page {display_num} is {orientation}, differs from page 1.",
                }
            )

    doc.close()
    return issues


def validate(
    pdf_path: str | Path,
    *,
    exp_start: int | None = None,
    exp_end: int | None = None,
    auto_correct: bool = True,
    progress_callback: Callable[[int, int, str], None] | None = None,
    model: str | Path | None = None,
    num_workers: int | None = None,
) -> dict:
    """Validate a PDF's page integrity.

    Runs OCR-based page number detection, then analyzes the sequence for
    missing pages, duplicates, gaps, and numbering errors. Optionally
    auto-corrects consistent OCR misreadings.

    :param pdf_path: Path to the PDF file.
    :param exp_start: Expected first page number (inferred from filename
        if omitted).
    :param exp_end: Expected last page number (inferred from filename
        if omitted).
    :param auto_correct: Attempt to fix consistent out-of-range OCR
        readings.
    :param progress_callback: Optional callable(current, total, message).
    :param model: Path to YOLO model for page number detection.
    :param num_workers: Number of parallel workers for OCR.
    :returns: Dict with keys ``page_count``, ``page_map``,
        ``missing_pages``, ``issues``, ``ocr_results``, and
        ``auto_corrections``.
    """
    pdf_path = str(pdf_path)

    # Get page count
    doc = fitz.open(pdf_path)
    pdf_page_count = len(doc)
    doc.close()

    # Infer expected range from filename if not provided
    if exp_start is None:
        exp_start, exp_end = _parse_expected_range(pdf_path)

    # Run OCR analysis
    analysis = analyze_pdf(
        pdf_path,
        exp_start=exp_start,
        exp_end=exp_end,
        progress_callback=progress_callback,
        model=model,
        num_workers=num_workers,
    )

    ocr_results = analysis["results"]

    # Split into in-range and out-of-range
    out_of_range, seen_nums = _split_in_out_of_range(ocr_results, exp_start, exp_end)

    # Auto-correct if enabled
    corrections: list[tuple[int, str, str]] = []
    if auto_correct:
        ocr_results, corrections = _auto_correct(ocr_results, out_of_range, seen_nums)
        if corrections:
            # Re-split after corrections
            out_of_range, seen_nums = _split_in_out_of_range(ocr_results, exp_start, exp_end)

    # Rebuild analysis with filtered data
    out_of_range_pages = {r["pdf_page"] for r in out_of_range}
    all_nums = sorted(seen_nums.keys())
    duplicates = {k: v for k, v in seen_nums.items() if len(v) > 1}

    # Sequence analysis (skip out-of-range pages)
    prev_num = prev_pdf = None
    seq_issues: list[tuple] = []
    for r in ocr_results:
        if not r["detected"] or r.get("type") == "range":
            prev_num = None
            continue
        try:
            num = int(r["detected"])
        except ValueError:
            continue
        if r["pdf_page"] in out_of_range_pages:
            continue
        if prev_num is not None:
            diff = num - prev_num
            if diff == 0:
                seq_issues.append(("DUPLICATE", r["pdf_page"], num, prev_pdf, prev_num))
            elif diff < 0:
                seq_issues.append(("BACKWARD", r["pdf_page"], num, prev_pdf, prev_num))
            elif diff > 2:
                seq_issues.append(
                    (
                        "GAP",
                        r["pdf_page"],
                        num,
                        prev_pdf,
                        prev_num,
                        list(range(prev_num + 1, num)),
                    )
                )
        prev_num = num
        prev_pdf = r["pdf_page"]

    # Range coverage
    ranges_found = [r for r in ocr_results if r.get("type") == "range"]
    range_pages: set[int] = set()
    for r in ranges_found:
        m = RANGE_RE.match(r["detected"].replace("\u2013", "-"))
        if m:
            for pg in range(int(m.group(1)), int(m.group(2)) + 1):
                range_pages.add(pg)

    # Missing pages
    if exp_start is not None and exp_end is not None and all_nums:
        missing_pages = sorted(
            (set(range(exp_start, exp_end + 1)) - set(all_nums)) - range_pages - {0}
        )
    elif all_nums:
        missing_pages = sorted(
            (set(range(all_nums[0], all_nums[-1] + 1)) - set(all_nums)) - range_pages - {0}
        )
    else:
        missing_pages = []

    filtered_analysis = {
        "results": ocr_results,
        "seq_issues": seq_issues,
        "duplicates": duplicates,
        "seen_nums": seen_nums,
        "all_nums": all_nums,
        "missing_pages": missing_pages,
        "ranges_found": ranges_found,
        "not_detected": [r for r in ocr_results if not r["detected"]],
        "out_of_range": out_of_range,
    }

    result = _build_issues(
        filtered_analysis,
        pdf_page_count,
        exp_start=exp_start,
        exp_end=exp_end,
    )

    # Add auto-correction warnings
    for pdf_page, old_val, new_val in corrections:
        result["issues"].append(
            {
                "page_number": pdf_page,
                "check_name": "auto_corrected",
                "severity": "warning",
                "message": (
                    f"PDF page {pdf_page}: OCR read '{old_val}', auto-corrected "
                    f"to '{new_val}' based on surrounding page numbers. Verify "
                    f"this is correct."
                ),
            }
        )

    # Structural checks
    structural = _structural_checks(pdf_path)
    result["issues"] = structural + result["issues"]
    result["auto_corrections"] = corrections

    return result
