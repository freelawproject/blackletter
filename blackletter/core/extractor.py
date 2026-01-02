"""Phase 4: Extract and mask opinions into separate PDFs."""

import logging
from pathlib import Path

import fitz

from blackletter import Document
from blackletter.config import RedactionConfig

logger = logging.getLogger(__name__)


class OpinionExtractor:
    """Extracts opinions into separate PDFs with optional masking."""

    def __init__(self, config: RedactionConfig):
        self.config = config

    def split_opinions(
        self,
        document: Document,
    ) -> Path:
        """Extract opinions into separate PDFs WITHOUT masking.

        Creates redacted/ directory with individual opinions.
        All pages from start to end included, no masking applied.

        Args:
            src_pdf_path: Path to the redacted source PDF
            opinion_spans: List of opinion spans from planner

        Returns:
            Path to the redacted opinions directory
        """

        src_pdf_path = Path(document.redacted_pdf_path)
        redacted_dir = src_pdf_path.parent / "redacted"
        redacted_dir.mkdir(parents=True, exist_ok=True)

        src = fitz.open(src_pdf_path)

        logger.info(f"Extracting {len(document.opinions)} opinions to redacted/")
        for op in document.opinions:
            start_pg_idx = int(op.caption.page_index)
            end_pg_idx = int(op.key.page_index)
            redacted_fp = redacted_dir / f"{op.case_name}.pdf"

            # Create PDF with only this opinion's pages (no masking)
            doc_out = fitz.open()
            doc_out.insert_pdf(src, from_page=start_pg_idx, to_page=end_pg_idx)

            doc_out.save(str(redacted_fp), garbage=4, deflate=True)
            doc_out.close()

            logger.info(f"Extracted to redacted/: {op.case_name}")

        src.close()
        logger.info(f"Saved {len(document.opinions)} opinions to {redacted_dir}")
        return redacted_dir

    def split_and_mask_opinions(
        self,
        document: Document,
        reduce: bool = True,
    ) -> Path:
        """Extract opinions into separate PDFs with masking.

        Args:
            src_pdf_path: Path to the redacted source PDF
            opinion_spans: List of opinion spans from planner
            page_columns_px: Column boundaries per page
            page_headers: Header y-coordinates per page
            page_footers: Footer y-coordinates per page

        Returns:
            Path to the masked opinions directory
        """

        src_pdf_path = Path(document.redacted_pdf_path)
        masked_dir = src_pdf_path.parent / "masked"
        masked_dir.mkdir(parents=True, exist_ok=True)

        src = fitz.open(src_pdf_path)
        scale = 72 / self.config.dpi

        logger.info(f"Extracting {len(document.opinions)} opinions")

        for op in document.opinions:
            start_pg_idx = op.caption.page_index
            end_pg_idx = op.key.page_index
            case_name = op.case_name
            masked_fp = masked_dir / f"{case_name}.pdf"

            # Create PDF with only this opinion's pages
            doc_out = fitz.open()
            doc_out.insert_pdf(src, from_page=start_pg_idx, to_page=end_pg_idx)

            # Apply masking to hide non-opinion content
            self._apply_opinion_masking(
                doc_out,
                start_pg_idx,
                end_pg_idx,
                op.caption.coords[1],
                op.key.coords[3],
                op.caption.col,
                op.key.col,
                scale,
                document,
            )
            if reduce == True:
                # Remove pages that are fully redacted to shrink file size
                filler_pages = document.get_filler_pages()
                for pg_idx in sorted(filler_pages, reverse=True):
                    if start_pg_idx <= pg_idx <= end_pg_idx:
                        local_idx = pg_idx - start_pg_idx
                        doc_out.delete_page(local_idx)

            doc_out.save(str(masked_fp), garbage=4, deflate=True)
            doc_out.close()

            logger.info(f"Extracted: {case_name}")

        src.close()
        logger.info(f"Saved {len(document.opinions)} opinions to {masked_dir}")
        return masked_dir

    def _apply_opinion_masking(
        self,
        doc_out,
        start_pg_idx: int,
        end_pg_idx: int,
        start_y_px: float,
        end_y_px: float,
        start_col: str,
        end_col: str,
        scale: float,
        document: Document,
    ):
        """Apply masking to hide non-opinion content on extracted pages.

        Masks everything outside the opinion span using white rectangles.
        """
        p_start = doc_out[0]
        p_end = doc_out[-1]

        def safe_redact(page, rect):
            """Safely add redaction if rect is valid."""
            if rect.y1 > rect.y0 and rect.x1 > rect.x0:
                page.add_redact_annot(rect, fill=self.config.mask_color)

        # === START PAGE ===
        W_start, H_start = p_start.rect.width, p_start.rect.height
        split_start = document.pages[start_pg_idx].midpoint * scale
        sy_pt = start_y_px * scale
        hy_start = (document.pages[start_pg_idx].header_bottom * scale) + 2
        if document.pages[start_pg_idx].footer_top:
            fy_start = (document.pages[start_pg_idx].footer_top * scale) + 2
        else:
            fy_start = H_start

        if start_col == "LEFT":
            # Mask right column and everything above opinion on left
            safe_redact(p_start, fitz.Rect(0, hy_start, split_start, min(sy_pt, fy_start)))
        elif start_col == "RIGHT":
            # Mask left column and everything above opinion on right
            safe_redact(p_start, fitz.Rect(0, hy_start, split_start, fy_start))
            safe_redact(p_start, fitz.Rect(split_start, hy_start, W_start, min(sy_pt, fy_start)))

        # === END PAGE ===
        W_end, H_end = p_end.rect.width, p_end.rect.height
        # split_end = get_split_pt(end_pg_idx, W_end)
        split_end = document.pages[end_pg_idx].midpoint * scale

        ey_pt = end_y_px * scale
        hy_end = (document.pages[end_pg_idx].header_bottom * scale) + 2

        if document.pages[end_pg_idx].footer_top:
            fy_end = (document.pages[end_pg_idx].footer_top * scale) + 2
        else:
            fy_end = H_end

        if end_col == "LEFT":
            # Mask right column and everything below opinion on left
            safe_redact(p_end, fitz.Rect(0, ey_pt, split_end, fy_end))
            safe_redact(p_end, fitz.Rect(split_end, hy_end, W_end, fy_end))
        elif end_col == "RIGHT":
            # Mask everything below opinion on right
            safe_redact(p_end, fitz.Rect(split_end, ey_pt, W_end, fy_end))

        # Apply all redactions
        for page in doc_out:
            page.apply_redactions()
