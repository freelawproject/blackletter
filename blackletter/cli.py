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


DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "run_9.pt"


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by all subcommands."""
    parser.add_argument("pdf", type=Path, help="Path to the source PDF")
    parser.add_argument(
        "--model", type=Path, default=DEFAULT_MODEL,
        help=f"Path to the YOLO model weights (.pt) (default: {DEFAULT_MODEL.name})",
    )
    parser.add_argument(
        "--reporter", type=str, default=None,
        help="Reporter abbreviation (e.g. f3d, a3d)",
    )
    parser.add_argument(
        "--volume", type=str, default=None,
        help="Volume number",
    )
    parser.add_argument(
        "--first-page", type=int, default=1,
        help="Page number of the first page in the PDF (default: 1)",
    )


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
                print(f"Available: {', '.join(l.name for l in Label)}")
                sys.exit(1)
    else:
        label_set = None  # draw all

    output_path = args.output
    print(f"Drawing boxes to {output_path}...")
    draw_detections(args.pdf, document, output_path, labels=label_set)
    print("Done")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="blackletter",
        description="Process scanned legal PDFs with YOLO detection",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── process ──
    from blackletter.process import build_parser as _build_process_parser
    _build_process_parser(sub)

    # ── draw ──
    p_draw = sub.add_parser(
        "draw",
        help="Draw bounding boxes on the full PDF for review",
    )
    _add_common_args(p_draw)
    p_draw.add_argument(
        "--output", "-o", type=Path, required=True,
        help="Output PDF path",
    )
    p_draw.add_argument(
        "--labels", nargs="+", default=None,
        help="Labels to draw (e.g. CASE_CAPTION KEY_ICON BACKGROUND). Default: all",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    if args.command == "process":
        from blackletter.process import cmd_process
        cmd_process(args)
    elif args.command == "draw":
        cmd_draw(args)


if __name__ == "__main__":
    main()
