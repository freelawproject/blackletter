"""Command-line interface for blackletter."""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Blackletter: Remove copyrighted material from legal PDFs"
    )

    parser.add_argument("pdf", type=Path, help="Path to PDF file")

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output folder (default: pdf_parent/redactions)",
    )

    # To be used for naming conventions
    parser.add_argument(
        "-v",
        "--volume",
        type=str,
        default=None,
        help="The volume to redact useful for naming in the future",
    )

    parser.add_argument(
        "-r", "--reporter", type=str, default=None, help="The reporter to extract out."
    )

    parser.add_argument(
        "-p", "--page", type=int, default=1, help="First page number for case naming (default: 1)"
    )

    parser.add_argument(
        "-m", "--model", type=str, default="best.pt", help="Path to YOLO model (default: best.pt)"
    )

    parser.add_argument(
        "-c", "--confidence", type=float, default=0.20, help="Confidence threshold (default: 0.20)"
    )

    parser.add_argument(
        "-d", "--dpi", type=int, default=200, help="DPI for PDF rendering (default: 200)"
    )

    parser.add_argument("--redact", action="store_true", help="Only redact PDF without masking")

    parser.add_argument(
        "--mask",
        action="store_true",
        help="Only mask opinions (extract to file without redacting PDF)",
    )

    parser.add_argument(
        "--reduce",
        action="store_true",
        help="Remove pages that are fully redacted from mask",
    )

    parser.add_argument(
        "--combine",
        action="store_true",
        help="Combine consecutive short opinions into grouped PDFs",
    )

    parser.add_argument(
        "--combine-threshold",
        type=int,
        default=2,
        help="Max page span for 'short' opinion (default: 2)",
    )

    args = parser.parse_args()

    if not args.pdf.exists():
        logger.error(f"PDF not found: {args.pdf}")
        return 1

    try:
        from blackletter import BlackletterPipeline
        from blackletter.config import RedactionConfig

        config = RedactionConfig(
            MODEL_PATH=args.model,
            confidence_threshold=args.confidence,
            dpi=args.dpi,
            combine_short_opinions=args.combine,
            short_opinion_threshold=args.combine_threshold,
        )

        pipeline = BlackletterPipeline(config)
        redacted_pdf, redacted_opinions, masked_opinions = pipeline.process(
            args.pdf,
            args.output,
            args.page,
            mask=args.mask,
            redact=args.redact,
            reduce=args.reduce,
        )

        logger.info(f"✓ Redacted PDF: {redacted_pdf}")
        logger.info(f"✓ Opinions extracted to: {masked_opinions}")
        logger.info(f"✓ Opinions extracted to: {redacted_opinions}")
        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())

__all__ = ["main"]
