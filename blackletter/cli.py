"""Command-line interface for blackletter."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ultralytics import YOLO

from blackletter.models import Label
from blackletter.scanner import (
    scan,
    draw_detections,
)

logger = logging.getLogger("blackletter")


DEFAULT_MODEL = Path(__file__).resolve().parent / "weights" / "small.pt"
MEDIUM_MODEL = Path(__file__).resolve().parent / "weights" / "medium.pt"
LARGE_MODEL = Path(__file__).resolve().parent / "weights" / "large.pt"

# Hugging Face repo for models not bundled in the package
_HF_SOURCES = {
    LARGE_MODEL: ("flooie/blackletter-large", "large.pt"),
}


def _ensure_model(path: Path) -> None:
    """Download model from Hugging Face if not present locally."""
    if path.exists():
        return
    if path not in _HF_SOURCES:
        print(f"Model not found: {path}")
        sys.exit(1)
    repo_id, filename = _HF_SOURCES[path]
    print(f"Model not found locally — downloading {filename} from {repo_id}...")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface_hub is required to download models. Run: pip install huggingface_hub")
        sys.exit(1)
    hf_hub_download(repo_id=repo_id, filename=filename, local_dir=path.parent)
    print(f"Downloaded to {path}")


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by all subcommands."""
    parser.add_argument("pdf", type=Path, help="Path to the source PDF")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"Path to the YOLO model weights (.pt) (default: {DEFAULT_MODEL.name})",
    )
    parser.add_argument(
        "--medium",
        action="store_true",
        help="Use the medium model (medium.pt, run_57) with 17 detection classes",
    )
    parser.add_argument(
        "--large",
        action="store_true",
        help="Use the large model (analyze.pt, run_59) with 21 detection classes",
    )
    parser.add_argument(
        "--reporter",
        type=str,
        default=None,
        help="Reporter abbreviation (e.g. f3d, a3d)",
    )
    parser.add_argument(
        "--volume",
        type=str,
        default=None,
        help="Volume number",
    )
    parser.add_argument(
        "--first-page",
        type=int,
        default=1,
        help="Page number of the first page in the PDF (default: 1)",
    )


def cmd_validate(args: argparse.Namespace) -> None:
    """Validate page number sequence and report issues."""
    from blackletter.analyze import DEFAULT_ANALYZE_MODEL
    from blackletter.validate import validate

    # Ensure the analyze model is present (downloads from HF if missing)
    _ensure_model(args.model or DEFAULT_ANALYZE_MODEL)

    def progress(current, total, message):
        print(f"\r  {message}", end="", flush=True)

    print(f"Validating {args.pdf.name}...")
    result = validate(
        args.pdf,
        exp_start=args.first_page,
        exp_end=args.last_page,
        auto_correct=not args.no_auto_correct,
        progress_callback=progress,
        model=args.model,
    )
    print()  # newline after progress

    if args.json_output:
        import json

        print(json.dumps(result, indent=2, default=str))
        return

    # Human-readable report
    print(f"\n{result['page_count']} pages in PDF")

    if result.get("auto_corrections"):
        print(f"\nAuto-corrections ({len(result['auto_corrections'])}):")
        for pdf_page, old_val, new_val in result["auto_corrections"]:
            print(f"  PDF page {pdf_page}: {old_val} → {new_val}")

    errors = [i for i in result["issues"] if i["severity"] == "error"]
    warnings = [i for i in result["issues"] if i["severity"] == "warning"]
    infos = [i for i in result["issues"] if i["severity"] == "info"]

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for i in errors:
            page = f"p.{i['page_number']}" if i["page_number"] else "doc"
            print(f"  [{page}] {i['message']}")

    if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for i in warnings:
            page = f"p.{i['page_number']}" if i["page_number"] else "doc"
            print(f"  [{page}] {i['message']}")

    if infos:
        print(f"\nInfo ({len(infos)}):")
        for i in infos[:10]:
            page = f"p.{i['page_number']}" if i["page_number"] else "doc"
            print(f"  [{page}] {i['message']}")
        if len(infos) > 10:
            print(f"  ... and {len(infos) - 10} more")

    if result["missing_pages"]:
        print(f"\nMissing pages ({len(result['missing_pages'])}):")
        # Group into ranges for compact display
        groups = []
        if result["missing_pages"]:
            start = end = result["missing_pages"][0]
            for n in result["missing_pages"][1:]:
                if n == end + 1:
                    end = n
                else:
                    groups.append((start, end))
                    start = end = n
            groups.append((start, end))
        for s, e in groups:
            if s == e:
                print(f"  #{s}")
            else:
                print(f"  #{s}-{e} ({e - s + 1} pages)")

    if not errors and not warnings:
        print("\nAll clear — no issues found.")


def cmd_draw(args: argparse.Namespace) -> None:
    """Scan and draw bounding boxes on the full PDF."""
    model = YOLO(str(args.model))

    print(f"Scanning {args.pdf}...")
    document = scan(args.pdf, model, first_page=args.first_page)

    # Parse label names
    if args.labels:
        label_set = set()
        for name in args.labels:
            try:
                label_set.add(Label[name.upper()])
            except KeyError:
                print(f"Unknown label: {name}")
                print(f"Available: {', '.join(lab.name for lab in Label)}")
                sys.exit(1)
    else:
        label_set = None  # draw all

    output_path = args.output
    print(f"Drawing boxes to {output_path}...")
    draw_detections(document.pdf_path, document, output_path, labels=label_set)
    print("Done")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="blackletter",
        description="Process scanned legal PDFs with YOLO detection",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── process ──
    from blackletter.process import build_parser as _build_process_parser

    _build_process_parser(sub)

    # ── validate ──
    p_validate = sub.add_parser(
        "validate",
        help="Check page number sequence for missing, duplicate, or misnumbered pages",
    )
    p_validate.add_argument("pdf", type=Path, help="Path to the PDF to validate")
    p_validate.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to YOLO model for page number detection (default: bundled analyze.pt)",
    )
    p_validate.add_argument(
        "--first-page",
        type=int,
        default=None,
        help="Expected first page number (inferred from filename if omitted)",
    )
    p_validate.add_argument(
        "--last-page",
        type=int,
        default=None,
        help="Expected last page number (inferred from filename if omitted)",
    )
    p_validate.add_argument(
        "--no-auto-correct",
        action="store_true",
        help="Disable automatic correction of consistent OCR misreadings",
    )
    p_validate.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output raw JSON instead of human-readable report",
    )

    # ── draw ──
    p_draw = sub.add_parser(
        "draw",
        help="Draw bounding boxes on the full PDF for review",
    )
    _add_common_args(p_draw)
    p_draw.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output PDF path",
    )
    p_draw.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Labels to draw (e.g. CASE_CAPTION KEY_ICON BACKGROUND). Default: all",
    )

    args = parser.parse_args()

    # Resolve --medium / --large flags to the appropriate model path
    if getattr(args, "large", False) and args.model == DEFAULT_MODEL:
        args.model = LARGE_MODEL
    elif getattr(args, "medium", False) and args.model == DEFAULT_MODEL:
        args.model = MEDIUM_MODEL

    # Download model from Hugging Face if not present locally
    if hasattr(args, "model") and args.model is not None:
        _ensure_model(args.model)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    if args.command == "process":
        from blackletter.process import cmd_process

        cmd_process(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "draw":
        cmd_draw(args)


if __name__ == "__main__":
    main()
