"""Blackletter: Remove copyrighted material from case law PDFs.

A reference to blackletter law, this module removes proprietary annotations
from legal documents—specifically key citations, headnotes, and other
copyrighted non-judicial materials—while preserving the authentic opinion text.
"""

import logging
from pathlib import Path
from typing import Tuple

from ultralytics import YOLO

from blackletter.config import RedactionConfig
from blackletter.core.scanner import PDFScanner, Document
from blackletter.core.planner import OpinionPlanner
from blackletter.core.redactor import PDFRedactor
from blackletter.core.extractor import OpinionExtractor
from blackletter.core.advance_sheet import scan_splitter

__version__ = "0.0.1"

logger = logging.getLogger(__name__)


class BlackletterPipeline:
    """Complete redaction pipeline: scan -> plan -> execute -> extract."""

    def __init__(self, config: RedactionConfig = None):
        self.config = config or RedactionConfig()
        logger.info(f"Using model: {self.config.MODEL_PATH}")
        self.model = YOLO(self.config.MODEL_PATH)

    def process(
        self,
        pdf_path: Path,
        output_folder: Path = None,
        first_page: int = 1,
        mask: bool = False,
        redact: bool = False,
        reduce: bool = False,
    ) -> Tuple[Path, Path, Path]:
        """Execute complete redaction pipeline.

        Args:
            pdf_path: Path to input PDF
            output_folder: Where to save redacted PDFs (default: pdf_path.parent/redactions)
            first_page: First page number for case naming

        Returns:
            (redacted_pdf_path, redacted_opinions_dir, masked_opinions_dir)
        """
        document = Document(pages=[], first_page=first_page, pdf_path=pdf_path)

        if output_folder is None:
            output_folder = pdf_path.parent / "redactions"

        logger.info(f"Processing: {pdf_path}")

        # Phase 1: Scan
        scanner = PDFScanner(self.config, self.model)
        document = scanner.scan(document)

        # Phase 2: Plan
        planner = OpinionPlanner(self.config)
        document = planner.plan(document)

        # Phase 3: Execute
        redactor = PDFRedactor(self.config)
        redactor.redact(document, output_folder)

        redacted_opinions_dir, masked_opinions_dir = None, None
        # Post-processing: Extract opinions
        extractor = OpinionExtractor(self.config)
        if redact == True:
            redacted_opinions_dir = extractor.split_opinions(
                document=document, combine_short=self.config.combine_short_opinions
            )
        if mask == True:
            masked_opinions_dir = extractor.split_and_mask_opinions(
                document=document,
                reduce=reduce,
                combine_short=self.config.combine_short_opinions,
            )

        logger.info("Pipeline completed successfully")
        return Path(document.redacted_pdf_path), redacted_opinions_dir, masked_opinions_dir


# Convenience functions
def redact_pdf(pdf_path: Path, output_folder: Path = None) -> Tuple[Path, Path, Path]:
    """Simple interface: redact a PDF with default configuration."""
    pipeline = BlackletterPipeline()
    return pipeline.process(pdf_path, output_folder)
