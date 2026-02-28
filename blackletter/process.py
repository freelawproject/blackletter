"""Unified process command: scan, verify, and split in one pass."""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

import fitz
from ultralytics import YOLO

from blackletter.models import Label
from blackletter.scanner import (
    scan,
    split_opinions,
    validate_page_numbers,
    recompress_images,
    _pair_opinions,
    _outside_opinion_rects,
    _margin_bounds,
    _find_redaction_end,
    _headnote_fallback_rects,
    _redaction_rects,
    _tighten_to_text,
    _text_bottom,
    _text_x_bounds,
    _REDACT_WHITE,
    _REDACT_BLACK,
    _MARGIN_LABELS,
    LABEL_CONFIDENCE,
    CONFIDENCE_THRESHOLD,
)

logger = logging.getLogger("blackletter")

DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "run_9.pt"


def _build_output_dir(args: argparse.Namespace) -> Path:
    """Build output dir: base / reporter / volume / first_page."""
    base = args.output
    if args.reporter:
        base = base / args.reporter
    if args.volume:
        base = base / str(args.volume)
    base = base / str(args.first_page)
    return base


def _build_verify_report(document, opinions) -> list[str]:
    """Build verify report lines from a scanned document."""
    lines: list[str] = []

    # Detection stats (filtered by confidence threshold)
    counts: Counter[Label] = Counter()
    for p in document.pages:
        for d in p.detections:
            if d.confidence >= LABEL_CONFIDENCE.get(d.label, CONFIDENCE_THRESHOLD):
                counts[d.label] += 1

    lines.append(f"{len(document.pages)} pages scanned")
    lines.append("")
    lines.append("Detection counts:")
    for label in sorted(counts, key=lambda lab: lab.value):
        lines.append(f"  {label.name:25s}: {counts[label]}")

    # Page number stats
    numbered = [p for p in document.pages if p.page_number is not None]
    lines.append("")
    lines.append(f"Page numbers: {len(numbered)}/{len(document.pages)} detected")
    if numbered:
        lines.append(f"Range: {numbered[0].page_number} - {numbered[-1].page_number}")

    # Validation
    warnings = validate_page_numbers(document)
    if warnings:
        lines.append("")
        lines.append(f"{len(warnings)} warnings:")
        for w in warnings:
            lines.append(f"  {w}")
    else:
        lines.append("")
        lines.append("No page number warnings")

    # Page number listing
    lines.append("")
    lines.append("Page numbers:")
    for p in document.pages:
        pn = p.page_number
        if pn is not None:
            lines.append(f"  PDF page {p.index + 1:3d} -> {pn}")
        else:
            lines.append(f"  PDF page {p.index + 1:3d} -> ???")

    # Opinion pairing preview
    captions = [
        d
        for d in document.by_label(Label.CASE_CAPTION)
        if d.confidence >= LABEL_CONFIDENCE.get(Label.CASE_CAPTION, CONFIDENCE_THRESHOLD)
    ]
    keys = [
        d
        for d in document.by_label(Label.KEY_ICON)
        if d.confidence >= LABEL_CONFIDENCE.get(Label.KEY_ICON, CONFIDENCE_THRESHOLD)
    ]
    lines.append("")
    lines.append(
        f"Opinion pairing: {len(captions)} captions, {len(keys)} key icons -> {len(opinions)} pairs"
    )

    # Per-opinion details
    if opinions:
        pages_by_index = {p.index: p for p in document.pages}

        prefix = ""
        if document.reporter and document.volume:
            prefix = f"{document.reporter}.{document.volume}."

        lines.append("")
        lines.append("Opinion details:")

        for i, (caption, key) in enumerate(opinions):
            cap_page = pages_by_index[caption.page_index]
            key_page = pages_by_index[key.page_index]
            first_num = cap_page.page_number
            last_num = key_page.page_number

            # Build the filename this opinion would get
            if first_num is not None and last_num is not None:
                fname = f"{prefix}{first_num:04d}-{last_num:04d}.pdf"
            else:
                fb_first = caption.page_index + (document.first_page or 1)
                fb_last = key.page_index + (document.first_page or 1)
                fname = f"{prefix}UNVERIFIED_{fb_first:04d}-{fb_last:04d}.pdf"

            # Header line for this opinion
            pdf_first = caption.page_index + 1  # 1-based PDF page
            pdf_last = key.page_index + 1
            if first_num is not None and last_num is not None:
                lines.append(
                    f"  Opinion {i + 1}: PDF pages {pdf_first}-{pdf_last}"
                    f" -> pages {first_num}-{last_num}  [{fname}]"
                )
            else:
                lines.append(f"  Opinion {i + 1}: PDF pages {pdf_first}-{pdf_last}  [{fname}]")

            # List each page in the span with its detected number
            for idx in range(caption.page_index, key.page_index + 1):
                p = pages_by_index[idx]
                pn = p.page_number

                # Compute expected page number based on first page in opinion
                if first_num is not None:
                    expected = first_num + (idx - caption.page_index)
                else:
                    expected = None

                if pn is not None:
                    line = f"    PDF page {idx + 1:3d} -> {pn}"
                    if expected is not None and pn != expected:
                        line += f"  WARNING: expected {expected}, likely OCR misread"
                    lines.append(line)
                else:
                    lines.append(f"    PDF page {idx + 1:3d} -> ???")

            # Gap to next opinion
            if i + 1 < len(opinions):
                next_caption, _ = opinions[i + 1]
                next_cap_page = pages_by_index[next_caption.page_index]
                next_first = next_cap_page.page_number

                pdf_gap = next_caption.page_index - key.page_index
                if last_num is not None and next_first is not None:
                    page_gap = next_first - last_num
                    if page_gap > pdf_gap:
                        missing = page_gap - pdf_gap
                        lines.append(
                            f"    GAP: missing {missing} source page(s)"
                            f" between pages {last_num} and {next_first}"
                        )

            lines.append("")  # blank line between opinions

    return lines


def _build_masked_opinions(
    masked_paths: list[Path],
    opinions: list,
    document,
    output_dir: Path,
) -> list[Path]:
    """Consolidate same-page masked opinions into single PDF pages.

    For opinions that all start and end on the same source page, produces
    a single 1-page PDF extracted from the source with:
    - All same-page opinions visible
    - Content from multi-page opinions (that start on this page but
      continue to the next) whited out
    - Headnote/per-detection redactions applied
    """
    pages_by_index = {p.index: p for p in document.pages}
    mid = document.pages[0].midpoint

    # Group consecutive single-page opinions on the same source page
    groups: list[list[int]] = []
    i = 0
    while i < len(opinions):
        caption, key = opinions[i]
        if caption.page_index == key.page_index:
            group = [i]
            j = i + 1
            while j < len(opinions):
                next_cap, next_key = opinions[j]
                if (
                    next_cap.page_index == next_key.page_index
                    and next_cap.page_index == caption.page_index
                ):
                    group.append(j)
                    j += 1
                else:
                    break
            groups.append(group)
            i = j
        else:
            groups.append([i])
            i += 1

    prefix = ""
    if document.reporter and document.volume:
        prefix = f"{document.reporter}.{document.volume}."

    src_pdf = fitz.open(str(document.pdf_path))
    final_paths: list[Path] = []

    for group in groups:
        if len(group) == 1:
            final_paths.append(masked_paths[group[0]])
            continue

        # Multiple same-page opinions — build a single-page PDF
        first_cap, _ = opinions[group[0]]
        _, last_key = opinions[group[-1]]
        src_page_idx = first_cap.page_index
        page = pages_by_index[src_page_idx]
        first_num = page.page_number

        if first_num is not None:
            name = f"{prefix}{first_num:04d}-{first_num:04d}.pdf"
        else:
            fb = src_page_idx + document.first_page
            name = f"{prefix}{fb:04d}-{fb:04d}.pdf"

        out_path = output_dir / name

        # Extract source page
        out_pdf = fitz.open()
        out_pdf.insert_pdf(src_pdf, from_page=src_page_idx, to_page=src_page_idx)
        fitz_page = out_pdf[0]

        sx, sy = page.scale_x, page.scale_y

        # Collect page number rects — never redact these
        pn_rects = []
        for d in page.detections:
            if d.label == Label.PAGE_NUMBER:
                b = d.bbox.to_pdf(sx, sy)
                r = fitz.Rect(b.x1, b.y1, b.x2, b.y2)
                tight = _tighten_to_text(fitz_page, r, skip=document.ocr_applied)
                pn_rects.append(tight if tight is not None else r)

        def add_safe(rect: fitz.Rect, fill) -> None:
            if rect.is_empty:
                return
            for pn in pn_rects:
                if rect.intersects(pn):
                    if rect.y0 < pn.y0:
                        add_safe(fitz.Rect(rect.x0, rect.y0, rect.x1, pn.y0), fill)
                    if rect.y1 > pn.y1:
                        add_safe(fitz.Rect(rect.x0, pn.y1, rect.x1, rect.y1), fill)
                    top = max(rect.y0, pn.y0)
                    bot = min(rect.y1, pn.y1)
                    if rect.x0 < pn.x0 and top < bot:
                        add_safe(fitz.Rect(rect.x0, top, pn.x0, bot), fill)
                    if rect.x1 > pn.x1 and top < bot:
                        add_safe(fitz.Rect(pn.x1, top, rect.x1, bot), fill)
                    return
            fitz_page.add_redact_annot(rect, fill=fill)

        # Outside-opinion whiteout: treat the group as one mega-opinion
        # from first caption to last key
        for rect in _outside_opinion_rects(
            page,
            fitz_page.rect.width,
            first_cap,
            last_key,
            is_first=True,
            is_last=True,
        ):
            add_safe(rect, (1, 1, 1))

        # Headnote blackout for each opinion in the group
        cap_key = first_cap.sort_key(mid)
        key_key = last_key.sort_key(mid)
        for idx in group:
            cap, key = opinions[idx]
            opinion_dets = []
            for d in page.detections:
                sk = d.sort_key(mid)
                if cap.sort_key(mid) <= sk <= key.sort_key(mid):
                    opinion_dets.append(d)
            opinion_dets.sort(key=lambda d: d.sort_key(mid))

            end_marker = _find_redaction_end(opinion_dets, cap, key, mid)
            if end_marker is not None:
                headnote_rects = _redaction_rects(cap, end_marker, pages_by_index)
            else:
                headnote_rects = _headnote_fallback_rects(opinion_dets, cap, pages_by_index, mid)
            if headnote_rects:
                header_bottom, footer_top = _margin_bounds(page)
                for rect_page_idx, rect in headnote_rects:
                    if rect_page_idx == src_page_idx:
                        if not document.ocr_applied:
                            tight = _tighten_to_text(fitz_page, rect)
                            if tight is not None:
                                rect = tight
                            text_bot = _text_bottom(fitz_page, rect)
                            text_left, text_right = _text_x_bounds(fitz_page, rect)
                            rect = fitz.Rect(
                                max(rect.x0, text_left),
                                max(rect.y0, header_bottom),
                                min(rect.x1, text_right),
                                min(rect.y1, footer_top, text_bot),
                            )
                        else:
                            rect = fitz.Rect(
                                rect.x0,
                                max(rect.y0, header_bottom),
                                rect.x1,
                                min(rect.y1, footer_top),
                            )
                        if rect.y0 < rect.y1 and rect.x0 < rect.x1:
                            add_safe(rect, (0, 0, 0))

        # Per-detection redactions (within the mega-span)
        redact_labels = _REDACT_WHITE | _REDACT_BLACK
        for d in page.detections:
            if d.label not in redact_labels:
                continue
            if d.confidence < LABEL_CONFIDENCE.get(d.label, CONFIDENCE_THRESHOLD):
                continue
            if d.label not in _MARGIN_LABELS:
                sk = d.sort_key(mid)
                if sk < cap_key or sk > key_key:
                    continue
            b = d.bbox.to_pdf(sx, sy)
            rect = fitz.Rect(b.x1, b.y1, b.x2, b.y2)
            tight = _tighten_to_text(fitz_page, rect, skip=document.ocr_applied)
            if tight is not None:
                rect = tight
            fill = (0, 0, 0) if d.label in _REDACT_BLACK else (1, 1, 1)
            add_safe(rect, fill)

        # Key icon redactions
        for d in page.detections:
            if d.label != Label.KEY_ICON:
                continue
            if d.confidence < LABEL_CONFIDENCE.get(d.label, CONFIDENCE_THRESHOLD):
                continue
            b = d.bbox.to_pdf(sx, sy)
            rect = fitz.Rect(b.x1, b.y1, b.x2, b.y2)
            sk = d.sort_key(mid)
            if sk < cap_key or sk > key_key:
                add_safe(rect, (1, 1, 1))
            else:
                add_safe(rect, (0, 0, 0))

        fitz_page.apply_redactions()
        recompress_images(out_pdf)
        out_pdf.save(str(out_path), garbage=4, deflate=True)
        out_pdf.close()

        # Remove the individual files
        for idx in group:
            p = masked_paths[idx]
            if p.exists() and p != out_path:
                p.unlink()

        final_paths.append(out_path)

    src_pdf.close()
    return final_paths


def _build_full_redacted(
    document,
    opinions: list[tuple],
    output_path: Path,
) -> Path:
    """Build a single fully-redacted copy of the entire document.

    Applies headnote blackout, per-detection redactions, and key icon
    blackout across every page.  No masking / outside-opinion whiteout.
    """
    pages_by_index = {p.index: p for p in document.pages}
    mid = document.pages[0].midpoint

    src_pdf = fitz.open(str(document.pdf_path))
    out_pdf = fitz.open()
    out_pdf.insert_pdf(src_pdf)

    # Pre-compute headnote rects for every opinion
    all_headnote_rects: list[tuple[int, fitz.Rect]] = []
    for caption, key in opinions:
        page = pages_by_index[caption.page_index]
        opinion_dets = []
        for p in document.pages:
            for d in p.detections:
                sk = d.sort_key(mid)
                if caption.sort_key(mid) <= sk <= key.sort_key(mid):
                    opinion_dets.append(d)
        opinion_dets.sort(key=lambda d: d.sort_key(mid))
        end_marker = _find_redaction_end(opinion_dets, caption, key, mid)
        if end_marker is not None:
            all_headnote_rects.extend(_redaction_rects(caption, end_marker, pages_by_index))
        else:
            all_headnote_rects.extend(
                _headnote_fallback_rects(opinion_dets, caption, pages_by_index, mid)
            )

    # Process each page
    total_pages = len(document.pages)
    for page in document.pages:
        src_idx = page.index
        fitz_page = out_pdf[src_idx]
        sx, sy = page.scale_x, page.scale_y

        # Collect page number rects — never redact these
        pn_rects = []
        for d in page.detections:
            if d.label == Label.PAGE_NUMBER:
                b = d.bbox.to_pdf(sx, sy)
                r = fitz.Rect(b.x1, b.y1, b.x2, b.y2)
                tight = _tighten_to_text(fitz_page, r, skip=document.ocr_applied)
                pn_rects.append(tight if tight is not None else r)

        def add_safe(rect: fitz.Rect, fill) -> None:
            if rect.is_empty:
                return
            for pn in pn_rects:
                if rect.intersects(pn):
                    if rect.y0 < pn.y0:
                        add_safe(fitz.Rect(rect.x0, rect.y0, rect.x1, pn.y0), fill)
                    if rect.y1 > pn.y1:
                        add_safe(fitz.Rect(rect.x0, pn.y1, rect.x1, rect.y1), fill)
                    top = max(rect.y0, pn.y0)
                    bot = min(rect.y1, pn.y1)
                    if rect.x0 < pn.x0 and top < bot:
                        add_safe(fitz.Rect(rect.x0, top, pn.x0, bot), fill)
                    if rect.x1 > pn.x1 and top < bot:
                        add_safe(fitz.Rect(pn.x1, top, rect.x1, bot), fill)
                    return
            fitz_page.add_redact_annot(rect, fill=fill)

        # Headnote blackout
        header_bottom, footer_top = _margin_bounds(page)
        for rect_page_idx, rect in all_headnote_rects:
            if rect_page_idx == src_idx:
                if not document.ocr_applied:
                    tight = _tighten_to_text(fitz_page, rect)
                    if tight is not None:
                        rect = tight
                    text_bot = _text_bottom(fitz_page, rect)
                    text_left, text_right = _text_x_bounds(fitz_page, rect)
                    rect = fitz.Rect(
                        max(rect.x0, text_left),
                        max(rect.y0, header_bottom),
                        min(rect.x1, text_right),
                        min(rect.y1, footer_top, text_bot),
                    )
                else:
                    rect = fitz.Rect(
                        rect.x0,
                        max(rect.y0, header_bottom),
                        rect.x1,
                        min(rect.y1, footer_top),
                    )
                if rect.y0 < rect.y1 and rect.x0 < rect.x1:
                    add_safe(rect, (0, 0, 0))

        # Per-detection redactions
        for d in page.detections:
            if d.label not in (_REDACT_WHITE | _REDACT_BLACK):
                continue
            if d.confidence < LABEL_CONFIDENCE.get(d.label, CONFIDENCE_THRESHOLD):
                continue
            b = d.bbox.to_pdf(sx, sy)
            rect = fitz.Rect(b.x1, b.y1, b.x2, b.y2)
            tight = _tighten_to_text(fitz_page, rect, skip=document.ocr_applied)
            if tight is not None:
                rect = tight
            fill = (0, 0, 0) if d.label in _REDACT_BLACK else (1, 1, 1)
            add_safe(rect, fill)

        # Key icon redactions (always black)
        for d in page.detections:
            if d.label != Label.KEY_ICON:
                continue
            if d.confidence < LABEL_CONFIDENCE.get(d.label, CONFIDENCE_THRESHOLD):
                continue
            b = d.bbox.to_pdf(sx, sy)
            rect = fitz.Rect(b.x1, b.y1, b.x2, b.y2)
            add_safe(rect, (0, 0, 0))

        fitz_page.apply_redactions()
        pg = page.index + 1
        if pg % 50 == 0 or pg == total_pages:
            print(f"    Redacted {pg}/{total_pages} pages", flush=True)

    print("  Recompressing images (JPEG)...")
    recompress_images(out_pdf)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_pdf.save(str(output_path), garbage=4, deflate=True)
    final_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved {output_path.name} ({final_mb:.1f} MB)")
    out_pdf.close()
    src_pdf.close()
    return output_path


def _extract_images(document, output_dir: Path) -> int:
    """Extract detected IMAGE regions from the PDF as PNG files.

    Returns the number of images extracted.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    src_pdf = fitz.open(str(document.pdf_path))
    # 300 DPI: PDF points are 72 per inch, so scale factor = 300/72
    mat = fitz.Matrix(300 / 72, 300 / 72)
    count = 0

    for page in document.pages:
        sx, sy = page.scale_x, page.scale_y
        fitz_page = src_pdf[page.index]

        # Collect IMAGE detections above confidence threshold
        images = [
            d
            for d in page.detections
            if d.label == Label.IMAGE
            and d.confidence >= LABEL_CONFIDENCE.get(Label.IMAGE, CONFIDENCE_THRESHOLD)
        ]
        if not images:
            continue

        # Sort by reading order (top-to-bottom, left-to-right)
        images.sort(key=lambda d: d.sort_key(page.midpoint))

        for seq, det in enumerate(images, start=1):
            b = det.bbox.to_pdf(sx, sy)
            clip = fitz.Rect(b.x1, b.y1, b.x2, b.y2)
            pix = fitz_page.get_pixmap(matrix=mat, clip=clip)

            # Use detected page number, fall back to index + first_page
            page_num = page.page_number
            if page_num is None:
                page_num = page.index + document.first_page

            fname = f"{page_num:04d}-{seq:03d}.png"
            pix.save(str(output_dir / fname))
            count += 1

    src_pdf.close()
    return count


def cmd_process(args: argparse.Namespace) -> None:
    """Scan, verify, and split into opinion PDFs in one pass."""
    model = YOLO(str(args.model))

    base_dir = _build_output_dir(args)
    base_dir.mkdir(parents=True, exist_ok=True)

    unredacted_dir = base_dir / "unredacted"
    redacted_dir = base_dir / "redacted"
    masked_dir = base_dir / "masked"

    # ── Scan ──
    shrink = not getattr(args, "no_shrink", False)

    # Build output name: reporter.volume.firstpage.lastpage
    page_count = len(fitz.open(str(args.pdf)))
    last_page = args.first_page + page_count - 1
    parts = []
    if args.reporter:
        parts.append(args.reporter)
    if args.volume:
        parts.append(str(args.volume))
    parts.append(str(args.first_page))
    parts.append(str(last_page))
    scan_name = ".".join(parts)

    src_mb = args.pdf.stat().st_size / (1024 * 1024)
    print(f"Scanning {args.pdf.name} ({page_count} pages, {src_mb:.1f} MB)...")
    document = scan(
        args.pdf,
        model,
        first_page=args.first_page,
        output_dir=base_dir,
        shrink=shrink,
        optimize=getattr(args, "optimize", 1),
        output_name=scan_name,
    )
    if args.reporter:
        document.reporter = args.reporter
    if args.volume:
        document.volume = args.volume

    # ── Extract images ──
    images_dir = base_dir / "images"
    n_images = _extract_images(document, images_dir)
    if n_images:
        print(f"\nExtracted {n_images} images to {images_dir}")

    # ── Verify report ──
    opinions = _pair_opinions(document)
    report_lines = _build_verify_report(document, opinions)

    for line in report_lines:
        print(line)

    report_path = base_dir / "verify.txt"
    report_path.write_text("\n".join(report_lines) + "\n")
    print(f"\nVerify report saved to {report_path}")

    # ── Full redacted (single PDF) ──
    prefix = ""
    if args.reporter:
        prefix = f"{args.reporter}."
    if args.volume:
        prefix += f"{args.volume}."
    full_redacted_name = f"{prefix}redacted.pdf" if prefix else "redacted.pdf"
    full_redacted_path = base_dir / full_redacted_name
    print("\nBuilding full redacted PDF...")
    _build_full_redacted(document, opinions, full_redacted_path)

    # ── Unredacted ──
    if not args.no_unredacted:
        print(f"\nSplitting unredacted into {unredacted_dir}...")
        unredacted_paths = split_opinions(
            document.pdf_path,
            document,
            unredacted_dir,
            first_page=args.first_page,
            redact_mode="unredacted",
            extract_footnotes=args.footnotes,
        )
        print(f"  Wrote {len(unredacted_paths)} unredacted PDFs")
    else:
        print("\nSkipping unredacted (--no-unredacted)")

    # ── Redacted ──
    print(f"\nSplitting redacted into {redacted_dir}...")
    redacted_paths = split_opinions(
        document.pdf_path,
        document,
        redacted_dir,
        first_page=args.first_page,
        redact_mode="redacted",
        extract_footnotes=args.footnotes,
    )
    print(f"  Wrote {len(redacted_paths)} redacted PDFs")

    # ── Masked (for LLM) ──
    print(f"\nSplitting masked into {masked_dir}...")
    masked_paths = split_opinions(
        document.pdf_path,
        document,
        masked_dir,
        first_page=args.first_page,
        redact_mode="masked",
    )
    # Consolidate same-page opinions
    final_masked = _build_masked_opinions(
        masked_paths,
        opinions,
        document,
        masked_dir,
    )
    print(f"  Wrote {len(final_masked)} masked PDFs ({len(masked_paths)} opinions consolidated)")


def process(
    pdf: str | Path,
    output: str | Path,
    *,
    reporter: str | None = None,
    volume: str | None = None,
    first_page: int = 1,
    model: str | Path | None = None,
    footnotes: bool = False,
    no_unredacted: bool = False,
    no_shrink: bool = False,
    optimize: int = 1,
) -> Path:
    """Process a legal PDF: scan, verify, split, and redact.

    Args:
        pdf: Path to the source PDF.
        output: Base output directory.
        reporter: Reporter abbreviation (e.g. "f3d", "a3d").
        volume: Volume number.
        first_page: Page number of the first page in the PDF.
        model: Path to YOLO model weights. Defaults to bundled model.
        footnotes: Extract footnotes into separate PDFs.
        no_unredacted: Skip generating unredacted opinion PDFs.
        no_shrink: Skip downsampling images.
        optimize: ocrmypdf optimization level (0-3).

    Returns:
        Path to the output directory containing all results.
    """
    if model is None:
        model = DEFAULT_MODEL
    args = argparse.Namespace(
        pdf=Path(pdf),
        output=Path(output),
        reporter=reporter,
        volume=volume,
        first_page=first_page,
        model=Path(model),
        footnotes=footnotes,
        no_unredacted=no_unredacted,
        no_shrink=no_shrink,
        optimize=optimize,
    )
    cmd_process(args)
    return _build_output_dir(args)


def build_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Register the process subcommand on an existing subparser group."""
    p = sub.add_parser(
        "process",
        help="Scan, verify, and split into opinion PDFs in one pass",
    )
    p.add_argument("pdf", type=Path, help="Path to the source PDF")
    p.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"Path to the YOLO model weights (.pt) (default: {DEFAULT_MODEL.name})",
    )
    p.add_argument(
        "--reporter",
        type=str,
        default=None,
        help="Reporter abbreviation (e.g. f3d, a3d)",
    )
    p.add_argument(
        "--volume",
        type=str,
        default=None,
        help="Volume number",
    )
    p.add_argument(
        "--first-page",
        type=int,
        default=1,
        help="Page number of the first page in the PDF (default: 1)",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Base output directory (organized as <reporter>/<volume>/<first-page>)",
    )
    p.add_argument(
        "--footnotes",
        action="store_true",
        help="Extract footnotes into separate PDFs",
    )
    p.add_argument(
        "--no-unredacted",
        action="store_true",
        help="Skip generating unredacted opinion PDFs",
    )
    p.add_argument(
        "--no-shrink",
        action="store_true",
        help="Skip downsampling (default: shrink to ~148 KB/page)",
    )
    p.add_argument(
        "--optimize",
        type=int,
        default=1,
        choices=[0, 1, 2, 3],
        help="ocrmypdf optimization level (0=none, 1=lossless, 2=lossy, 3=aggressive)",
    )
    return p
