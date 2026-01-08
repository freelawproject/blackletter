"""Phase 4: Extract and mask opinions into separate PDFs."""

import logging
from pathlib import Path
from typing import List

import fitz

from blackletter.config import RedactionConfig
from blackletter.core.scanner import Opinion, Document

logger = logging.getLogger(__name__)


class OpinionExtractor:
    """Extracts opinions into separate PDFs with optional masking."""

    def __init__(self, config: RedactionConfig):
        self.config = config

    def _group_opinions(self, opinions: List[Opinion]) -> List[List[Opinion]]:
        """Group consecutive short opinions using page-based logic.

        Algorithm:
        1. Start with a short opinion, track the furthest page it reaches
        2. Keep adding short opinions as long as they start on or before the furthest page reached
        3. Each added opinion may extend the furthest page reached
        4. Stop when we hit a long opinion or an opinion starting beyond our reach

        :param opinions: List of opinions (already sorted by page/column)
        :return: List of opinion groups
        """
        if not opinions:
            return []

        groups = []
        i = 0
        threshold = self.config.short_opinion_threshold

        while i < len(opinions):
            opinion = opinions[i]
            page_span = opinion.key.page_index - opinion.caption.page_index
            is_short = page_span <= threshold

            if not is_short:
                # Long opinion - its own group
                groups.append([opinion])
                i += 1
                continue

            # Start grouping short opinions
            group = [opinion]
            max_page_reached = opinion.key.page_index
            i += 1

            # Keep adding short opinions while they start on or before max_page_reached
            while i < len(opinions):
                next_op = opinions[i]
                next_span = next_op.key.page_index - next_op.caption.page_index
                next_is_short = next_span <= threshold

                if not next_is_short:
                    # Hit long opinion, stop
                    break

                if next_op.caption.page_index <= max_page_reached:
                    # This opinion starts on a page we've reached
                    group.append(next_op)
                    max_page_reached = max(max_page_reached, next_op.key.page_index)
                    i += 1
                else:
                    # This opinion starts on a page beyond what we've reached
                    break

            groups.append(group)

        return groups

    def split_opinions(
        self,
        document: Document,
        combine_short: bool = False,
    ) -> Path:
        """Extract opinions into separate PDFs WITHOUT masking.

        Creates redacted/ directory with individual opinions.
        All pages from start to end included, no masking applied.

        :param document: document object containing PDF and opinion instructions
        :param combine_short: whether to combine short opinions into groups
        :return: path to the redacted opinions directory
        """
        src_pdf_path = Path(document.redacted_pdf_path)
        redacted_dir = src_pdf_path.parent / "redacted"
        redacted_dir.mkdir(parents=True, exist_ok=True)

        src = fitz.open(src_pdf_path)

        # Group opinions if combine_short is enabled
        if combine_short:
            opinion_groups = self._group_opinions(document.opinions)
        else:
            opinion_groups = [[op] for op in document.opinions]

        logger.info(f"Extracting {len(opinion_groups)} opinion file(s) to redacted/")

        for group in opinion_groups:
            # Determine file name and page range
            if len(group) == 1:
                case_name = group[0].case_name
            else:
                case_name = f"{group[0].case_name}_to_{group[-1].case_name}"

            start_pg_idx = int(group[0].caption.page_index)
            end_pg_idx = int(group[-1].key.page_index)
            redacted_fp = redacted_dir / f"{case_name}.pdf"

            # Create PDF with only this opinion's pages (no masking)
            doc_out = fitz.open()
            doc_out.insert_pdf(src, from_page=start_pg_idx, to_page=end_pg_idx)

            doc_out.save(str(redacted_fp), garbage=4, deflate=True)
            doc_out.close()

            logger.info(f"Extracted to redacted/: {case_name}")

        src.close()
        logger.info(f"Saved {len(opinion_groups)} opinion file(s) to {redacted_dir}")
        return redacted_dir

    def split_and_mask_opinions(
        self,
        document: Document,
        reduce: bool = True,
        combine_short: bool = False,
    ) -> Path:
        """Extract opinions into separate PDFs with masking.

        :param document: document object containing PDF and opinion instructions
        :param reduce: whether to reduce file size after masking
        :param combine_short: whether to combine short opinions into groups
        :return: path to the masked opinions directory
        """
        src_pdf_path = Path(document.redacted_pdf_path)
        masked_dir = src_pdf_path.parent / "masked"
        masked_dir.mkdir(parents=True, exist_ok=True)

        src = fitz.open(src_pdf_path)

        # Group opinions if combine_short is enabled
        if combine_short:
            opinion_groups = self._group_opinions(document.opinions)
        else:
            opinion_groups = [[op] for op in document.opinions]

        logger.info(f"Extracting {len(opinion_groups)} opinion file(s)")

        for group in opinion_groups:
            if len(group) == 1:
                case_name = group[0].case_name
            else:
                case_name = f"{group[0].case_name}_to_{group[-1].case_name}"

            start_pg_idx = group[0].caption.page_index
            end_pg_idx = group[-1].key.page_index
            masked_fp = masked_dir / f"{case_name}.pdf"

            # Create PDF with only this opinion's pages
            doc_out = fitz.open()
            doc_out.insert_pdf(src, from_page=start_pg_idx, to_page=end_pg_idx)

            # Apply masking
            if len(group) == 1:
                self._apply_opinion_masking(doc_out, group[0], document)
            else:
                self._apply_group_masking(doc_out, group, document)

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
        logger.info(f"Saved {len(opinion_groups)} opinion file(s) to {masked_dir}")
        return masked_dir

    def _apply_opinion_masking(
        self,
        doc_out,
        opinion: Opinion,
        document: Document,
    ) -> None:
        """Apply masking to hide non-opinion content on extracted pages.

        Masks everything outside the opinion span using white rectangles.

        :param doc_out: output document object to apply masking to
        :param opinion: Opinion object
        :param document: document object with page layout information

        :return: None
        """
        scale = 72 / self.config.dpi
        p_start = doc_out[0]
        p_end = doc_out[-1]
        start_pg_idx = opinion.caption.page_index
        end_pg_idx = opinion.key.page_index
        start_y_px = opinion.caption.coords[1]
        end_y_px = opinion.key.coords[3]
        start_col = opinion.caption.col
        end_col = opinion.key.col

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

    def _apply_group_masking(
        self,
        doc_out,
        opinion_group: List[Opinion],
        document: Document,
    ) -> None:
        """Apply masking for a group of opinions.

        Masks content before first caption and after last key.
        Content between opinions in group remains visible.

        :param doc_out: output document
        :param opinion_group: list of opinions in the group
        :param document: document with layout info
        :return: None
        """
        scale = 72 / self.config.dpi

        first_op = opinion_group[0]
        last_op = opinion_group[-1]

        start_pg_idx = first_op.caption.page_index
        end_pg_idx = last_op.key.page_index
        start_y_px = first_op.caption.coords[1]
        end_y_px = last_op.key.coords[3]
        start_col = first_op.caption.col
        end_col = last_op.key.col

        p_start = doc_out[0]
        p_end = doc_out[-1]

        def safe_redact(page, rect):
            if rect.y1 > rect.y0 and rect.x1 > rect.x0:
                page.add_redact_annot(rect, fill=self.config.mask_color)

        # === MASK START PAGE ===
        W_start, H_start = p_start.rect.width, p_start.rect.height
        split_start = document.pages[start_pg_idx].midpoint * scale
        sy_pt = start_y_px * scale
        hy_start = (document.pages[start_pg_idx].header_bottom * scale) + 2
        if document.pages[start_pg_idx].footer_top:
            fy_start = (document.pages[start_pg_idx].footer_top * scale) + 2
        else:
            fy_start = H_start

        if start_col == "LEFT":
            safe_redact(p_start, fitz.Rect(0, hy_start, split_start, min(sy_pt, fy_start)))
        elif start_col == "RIGHT":
            safe_redact(p_start, fitz.Rect(0, hy_start, split_start, fy_start))
            safe_redact(p_start, fitz.Rect(split_start, hy_start, W_start, min(sy_pt, fy_start)))

        # === MASK END PAGE ===
        W_end, H_end = p_end.rect.width, p_end.rect.height
        split_end = document.pages[end_pg_idx].midpoint * scale
        ey_pt = end_y_px * scale
        hy_end = (document.pages[end_pg_idx].header_bottom * scale) + 2
        if document.pages[end_pg_idx].footer_top:
            fy_end = (document.pages[end_pg_idx].footer_top * scale) + 2
        else:
            fy_end = H_end

        if end_col == "LEFT":
            safe_redact(p_end, fitz.Rect(0, ey_pt, split_end, fy_end))
            safe_redact(p_end, fitz.Rect(split_end, hy_end, W_end, fy_end))
        elif end_col == "RIGHT":
            safe_redact(p_end, fitz.Rect(split_end, ey_pt, W_end, fy_end))

        # Apply all redactions
        for page in doc_out:
            page.apply_redactions()
