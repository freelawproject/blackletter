"""Blackletter: Remove copyrighted material from case law PDFs.

A reference to blackletter law, this module removes proprietary annotations
from legal documents—specifically key citations, headnotes, and other
copyrighted non-judicial materials—while preserving the authentic opinion text.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from pathlib import Path
from typing import Tuple

if TYPE_CHECKING:
    from blackletter.config import RedactionConfig

__version__ = "0.0.1"

logger = logging.getLogger(__name__)


class BlackletterPipeline:
    """Complete redaction pipeline: scan -> plan -> execute -> extract."""

    def __init__(self, config: RedactionConfig | None = None):
        from ultralytics import YOLO
        from blackletter.config import RedactionConfig

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
        combine_short: bool = False,
    ) -> Document:
        """Execute complete redaction pipeline.

        :param document: The document to scan.

        :param pdf_path: Path to input PDF
        :param output_folder: Where to save redacted PDFs
        :param first_page: First page number for case naming
        :param mask: Whether to generate masked versions of opinions
        :param redact: Whether to perform redaction on detected elements
        :param reduce: Whether to remove fully redacted pages
        :param combine_short: Whether to combine opinions into a single page

        :return Document: The docuemnt object
        """
        from blackletter.core.scanner import PDFScanner, Document
        from blackletter.core.planner import OpinionPlanner
        from blackletter.core.redactor import PDFRedactor
        from blackletter.core.extractor import OpinionExtractor

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
            extractor.split_opinions(
                document=document,
                combine_short=combine_short,
            )
        if mask == True:
            extractor.split_and_mask_opinions(
                document=document,
                reduce=reduce,
                combine_short=combine_short,
            )

        logger.info("Pipeline completed successfully")
        return document


# Convenience functions
def redact_pdf(
    pdf_path: Path,
    output_folder: Path = None,
) -> Document:
    """Simple interface: redact a PDF with default configuration."""
    pipeline = BlackletterPipeline()
    return pipeline.process(pdf_path, output_folder)
