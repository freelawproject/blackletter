"""Phase 4: Extract and mask opinions into separate PDFs."""

import logging
from pathlib import Path
from typing import Dict, List, Set

import fitz

from blackletter.config import RedactionConfig

logger = logging.getLogger(__name__)


class OpinionExtractor:
    """Extracts opinions into separate PDFs with optional masking."""

    def __init__(self, config: RedactionConfig):
        self.config = config

    def get_filler_pages(self, redaction_instructions: List[Dict]) -> Set[int]:
        """Identify pages between caption and line/headmatter (filler pages)."""
        filler_pages = set()

        for instr in redaction_instructions:
            start_pg = instr["start"]["page_index"]
            end_pg = instr["end"]["page_index"]

            # Pages strictly between caption and line/headmatter are filler
            for pg in range(start_pg + 1, end_pg):
                filler_pages.add(pg)

        return filler_pages

    def split_opinions(
            self,
            src_pdf_path: Path,
            opinion_spans: List[Dict],
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
        src_pdf_path = Path(src_pdf_path)
        redacted_dir = src_pdf_path.parent / "redacted"
        redacted_dir.mkdir(parents=True, exist_ok=True)

        src = fitz.open(src_pdf_path)

        logger.info(f"Extracting {len(opinion_spans)} opinions to redacted/")

        for item in opinion_spans:
            start_pg_idx = int(item["start"]["page_index"])
            end_pg_idx = int(item["end"]["page_index"])
            case_name = item.get("case_name", f"opinion_{item['n']:03d}")

            redacted_fp = redacted_dir / f"{case_name}.pdf"

            # Create PDF with only this opinion's pages (no masking)
            doc_out = fitz.open()
            doc_out.insert_pdf(src, from_page=start_pg_idx, to_page=end_pg_idx)

            doc_out.save(str(redacted_fp), garbage=4, deflate=True)
            doc_out.close()

            logger.info(f"Extracted to redacted/: {case_name}")

        src.close()
        logger.info(f"Saved {len(opinion_spans)} opinions to {redacted_dir}")
        return redacted_dir


    def split_and_mask_opinions(
            self,
            src_pdf_path: Path,
            opinion_spans: List[Dict],
            page_columns_px: Dict,
            page_headers: Dict,
            page_footers: Dict,
            redaction_instructions: List[Dict],
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

        src_pdf_path = Path(src_pdf_path)
        masked_dir = src_pdf_path.parent / "masked"
        masked_dir.mkdir(parents=True, exist_ok=True)

        src = fitz.open(src_pdf_path)
        scale = 72 / self.config.dpi

        logger.info(f"Extracting {len(opinion_spans)} opinions")

        for item in opinion_spans:
            start_pg_idx = int(item["start"]["page_index"])
            end_pg_idx = int(item["end"]["page_index"])
            case_name = item.get("case_name", f"opinion_{item['n']:03d}")

            masked_fp = masked_dir / f"{case_name}.pdf"

            # Create PDF with only this opinion's pages
            doc_out = fitz.open()
            doc_out.insert_pdf(src, from_page=start_pg_idx, to_page=end_pg_idx)

            # Apply masking to hide non-opinion content
            self._apply_opinion_masking(
                doc_out,
                start_pg_idx,
                end_pg_idx,
                item["start"]["coords"][1],  # start_y_px
                item["end"]["coords"][3],  # end_y_px
                item["start"]["col"],
                item["end"]["col"],
                page_columns_px,
                page_headers,
                page_footers,
                scale,
            )
            filler_pages = self.get_filler_pages(redaction_instructions)
            for pg_idx in sorted(filler_pages, reverse=True):
                if start_pg_idx <= pg_idx <= end_pg_idx:
                    # Convert global page index to local index in doc_out
                    local_idx = pg_idx - start_pg_idx
                    doc_out.delete_page(local_idx)

            doc_out.save(str(masked_fp), garbage=4, deflate=True)
            doc_out.close()

            logger.info(f"Extracted: {case_name}")

        src.close()
        logger.info(f"Saved {len(opinion_spans)} opinions to {masked_dir}")
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
            page_columns_px: Dict,
            page_headers: Dict,
            page_footers: Dict,
            scale: float,
    ):
        """Apply masking to hide non-opinion content on extracted pages.

        Masks everything outside the opinion span using white rectangles.
        """
        p_start = doc_out[0]
        p_end = doc_out[-1]

        def get_footer_pt(pg_idx: int, page_height_pt: float) -> float:
            """Get footer y-coordinate in points."""
            y_px = page_footers.get(pg_idx)
            return min(page_height_pt,
                       float(y_px) * scale) if y_px else page_height_pt

        def get_header_pt(pg_idx: int) -> float:
            """Get header y-coordinate in points."""
            y_px = page_headers.get(pg_idx, 0)
            return max(0.0, float(y_px) * scale)

        def safe_redact(page, rect):
            """Safely add redaction if rect is valid."""
            if rect.y1 > rect.y0 and rect.x1 > rect.x0:
                page.add_redact_annot(rect, fill=self.config.mask_color)

        def get_split_pt(pg_idx: int, page_width_pt: float) -> float:
            """Get column split point in points."""
            if pg_idx in page_columns_px:
                _, _, _, _, split_x_px = page_columns_px[pg_idx]
                return split_x_px * scale
            return page_width_pt / 2

        # === START PAGE ===
        W_start, H_start = p_start.rect.width, p_start.rect.height
        split_start = get_split_pt(start_pg_idx, W_start)
        sy_pt = start_y_px * scale
        hy_start = get_header_pt(start_pg_idx) + 2
        fy_start = get_footer_pt(start_pg_idx, H_start)

        if start_col == "LEFT":
            # Mask right column and everything above opinion on left
            safe_redact(
                p_start,
                fitz.Rect(0, hy_start, split_start, min(sy_pt, fy_start))
            )
        elif start_col == "RIGHT":
            # Mask left column and everything above opinion on right
            safe_redact(p_start, fitz.Rect(0, hy_start, split_start, fy_start))
            safe_redact(
                p_start,
                fitz.Rect(split_start, hy_start, W_start, min(sy_pt, fy_start))
            )

        # === END PAGE ===
        W_end, H_end = p_end.rect.width, p_end.rect.height
        split_end = get_split_pt(end_pg_idx, W_end)
        ey_pt = end_y_px * scale
        hy_end = get_header_pt(end_pg_idx) + 2
        fy_end = get_footer_pt(end_pg_idx, H_end)

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