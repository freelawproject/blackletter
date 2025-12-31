"""Blackletter: Remove copyrighted material from case law PDFs.

A reference to blackletter law, this module removes proprietary annotations
from legal documents—specifically Westlaw key citations, headnotes, and other
copyrighted judicial materials—while preserving the authentic opinion text.
"""

import logging
from pathlib import Path
from typing import Tuple

from ultralytics import YOLO

from blackletter.config import RedactionConfig
from blackletter.core.scanner import PDFScanner
from blackletter.core.planner import OpinionPlanner
from blackletter.core.redactor import PDFRedactor
from blackletter.core.extractor import OpinionExtractor

__version__ = "0.1.0"
__author__ = "Your Name"

logger = logging.getLogger(__name__)


class BlackletterPipeline:
    """Complete redaction pipeline: scan -> plan -> execute -> extract."""

    def __init__(self, config: RedactionConfig = None):
        self.config = config or RedactionConfig()
        logger.info(f"Using model: {self.config.MODEL_PATH}")
        self.model = YOLO(self.config.MODEL_PATH)

    def process(
        self, pdf_path: Path, output_folder: Path = None, first_page: int = 1
    ) -> Tuple[Path, Path, Path]:
        """Execute complete redaction pipeline.

        Args:
            pdf_path: Path to input PDF
            output_folder: Where to save redacted PDFs (default: pdf_path.parent/redactions)
            first_page: First page number for case naming

        Returns:
            (redacted_pdf_path, redacted_opinions_dir, masked_opinions_dir)
        """
        pdf_path = Path(pdf_path)
        if output_folder is None:
            output_folder = pdf_path.parent / "redactions"

        logger.info(f"Processing: {pdf_path}")

        # Phase 1: Scan
        scanner = PDFScanner(self.config, self.model)
        global_objects, page_dimensions, page_columns_px = scanner.scan(pdf_path)

        # Phase 2: Plan
        planner = OpinionPlanner(self.config)
        (
            redaction_instructions,
            opinion_spans,
            page_headers,
            page_footers,
        ) = planner.plan(global_objects, first_page)

        # Phase 3: Execute
        redactor = PDFRedactor(self.config)
        redacted_pdf = redactor.redact(
            pdf_path,
            redaction_instructions,
            global_objects,
            page_dimensions,
            page_columns_px,
            output_folder,
        )

        # Post-processing: Extract opinions
        extractor = OpinionExtractor(self.config)

        redacted_opinions_dir = extractor.split_opinions(
            src_pdf_path=redacted_pdf,
            opinion_spans=opinion_spans,
        )

        masked_opinions_dir = extractor.split_and_mask_opinions(
            src_pdf_path=redacted_pdf,
            opinion_spans=opinion_spans,
            page_columns_px=page_columns_px,
            page_headers=page_headers,
            page_footers=page_footers,
            redaction_instructions=redaction_instructions,
        )

        logger.info("Pipeline completed successfully")
        return redacted_pdf, redacted_opinions_dir, masked_opinions_dir


# Convenience functions
def redact_pdf(pdf_path: Path, output_folder: Path = None) -> Tuple[Path, Path, Path]:
    """Simple interface: redact a PDF with default configuration."""
    pipeline = BlackletterPipeline()
    return pipeline.process(pdf_path, output_folder)