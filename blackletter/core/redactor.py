"""Phase 3: Apply redactions to PDF."""

import logging
from pathlib import Path
from typing import Tuple

import fitz
import pdfplumber

from blackletter import Document
from blackletter.config import RedactionConfig
from blackletter.core.scanner import PageContext, Opinion
from blackletter.utils.text import redact_text_lines_in_window

logger = logging.getLogger(__name__)


class TextRedactor:
    """Handles text-level redactions within windows."""

    def __init__(self, config: RedactionConfig):
        self.config = config

    def redact_text_window(self, page_pl, page_fitz, win_pdf: Tuple[float, float, float, float]):
        """Redact text lines within a window using pdfplumber."""
        redact_text_lines_in_window(
            page_pl=page_pl,
            page_fitz=page_fitz,
            win_pdf=win_pdf,
            pad=self.config.text_pad,
            y_tol=self.config.y_tolerance,
            merge_gap=self.config.merge_gap,
        )


class PDFRedactor:
    """Applies all redactions to PDF in Phase 3."""

    def __init__(self, config: RedactionConfig):
        self.config = config
        self.text_redactor = TextRedactor(config)

    def redact(
        self,
        document: Document,
        output_folder: Path,
    ) -> Path:
        """Apply all redactions to PDF.

        Args:
            pdf_path: Input PDF path
            redaction_instructions: List of start->end redaction pairs
            output_folder: Where to save redacted PDF

        Returns:
            Path to redacted PDF
        """
        logger.info("Starting PHASE 3: Applying redactions")

        pdf_path = Path(document.pdf_path)
        doc = fitz.open(pdf_path)

        with pdfplumber.open(pdf_path) as pdf_read:
            for page in document.pages:
                page_fitz = doc[page.index]
                # page_pl = page.plumber_page
                page_pl = pdf_read.pages[page.index]

                self._apply_body_redactions(
                    page_fitz,
                    page_pl,
                    # redaction_instructions,
                    page,
                    document.opinions,
                )

                self._apply_object_redactions(page_fitz, page_pl, page)

                page_fitz.apply_redactions()

        output_path = output_folder / f"{pdf_path.stem}_redacted.pdf"
        output_folder.mkdir(exist_ok=True, parents=True)
        doc.save(output_path, garbage=4, deflate=True)
        doc.close()

        logger.info(f"Saved redacted PDF to: {output_path}")
        document.redacted_pdf_path = output_path.resolve()

    def _apply_instruction(
        self,
        page_fitz,
        page_pl,
        instr: Opinion,
        page: PageContext,
    ):
        """Apply a single redaction instruction."""
        start = instr.caption
        end = instr.line or instr.headmatter

        if end is None:
            return

        page_idx = page.index
        LEFT_X1 = page.column_left_x1
        LEFT_X2 = page.column_left_x2
        RIGHT_X1 = page.column_right_x1
        RIGHT_X2 = page.column_right_x2
        scale_x = page.pdf_pg_width / page.img_width
        scale_y = page.pdf_pg_height / page.img_height

        ceiling_y = page.header_bottom
        limit_bottom_left = page.footer_top or page.img_height - 60
        limit_bottom_right = page.footer_top or page.img_height - 60

        def do_column_box(y_top_px: int, y_bottom_px: int, is_left: bool):
            """Redact a column from y_top to y_bottom."""
            y_top_px = int(max(ceiling_y, y_top_px))
            max_y_px = limit_bottom_left if is_left else limit_bottom_right
            y_bottom_px = int(min(max_y_px, y_bottom_px))

            if y_bottom_px <= y_top_px:
                return

            xx1_px = LEFT_X1 if is_left else RIGHT_X1
            xx2_px = LEFT_X2 if is_left else RIGHT_X2

            x0_pdf = xx1_px * scale_x
            x1_pdf = xx2_px * scale_x
            y0_pdf = y_top_px * scale_y
            y1_pdf = y_bottom_px * scale_y

            self.text_redactor.redact_text_window(
                page_pl=page_pl,
                page_fitz=page_fitz,
                win_pdf=(x0_pdf, y0_pdf, x1_pdf, y1_pdf),
            )

        s_col = start.col == "LEFT"
        e_col = end.col == "LEFT"

        # Case 1: Start & End on same page
        if start.page_index == page_idx and end.page_index == page_idx:
            sy = start.coords[3] + self.config.start_offset
            ey = end.coords[1] + self.config.end_offset
            if s_col == e_col:
                do_column_box(sy, ey, s_col)
            else:
                do_column_box(sy, 9999, True)
                do_column_box(0, ey, False)

        # Case 2: Start on previous page, end on this page
        elif start.page_index < page_idx and end.page_index == page_idx:
            ey = end.coords[1] + self.config.end_offset
            if e_col:
                do_column_box(0, ey, True)
            else:
                do_column_box(0, 9999, True)
                do_column_box(0, ey, False)

        # Case 3: Start on this page, end on future page
        elif start.page_index == page_idx and end.page_index > page_idx:
            sy = start.coords[3] + self.config.start_offset
            if s_col:
                do_column_box(sy, 9999, True)
                do_column_box(0, 9999, False)
            else:
                do_column_box(sy, 9999, False)

        # Case 4: Middle page (between start and end)
        elif start.page_index < page_idx and end.page_index > page_idx:
            do_column_box(0, 9999, True)
            do_column_box(0, 9999, False)

    def _apply_body_redactions(
        self,
        page_fitz,
        page_pl,
        page: PageContext,
        opinions: list[Opinion],
    ):
        """Apply body text redactions for a page."""

        for opinion in opinions:
            self._apply_instruction(
                page_fitz,
                page_pl,
                opinion,
                page,
            )

    def _apply_object_redactions(
        self,
        page_fitz,
        page_pl,
        page: PageContext,
    ):
        """Apply redactions for specific detected objects."""
        from blackletter.utils.header import HeaderProcessor

        header_coord = None

        objs_on_page = page.page_objects
        scale_x = page.pdf_pg_width / page.img_width
        scale_y = page.pdf_pg_height / page.img_height

        # Redact specific object types
        for o in objs_on_page:
            label = o.label

            if label in ["line", "Key", "brackets", "order"]:
                c = [int(x) for x in o.coords]
                self._add_redaction_box(page_fitz, c[0], c[1], c[2], c[3], scale_x, scale_y)

            if label == "header":
                header_coord = [int(x) for x in o.coords]

        # Header redaction with special processing
        hdr = HeaderProcessor.redaction_bbox_for_header(
            page_pl,
            top_pts=self.config.header_top_pts,
            gap_pts=self.config.header_gap_pts,
            y_tol=self.config.header_y_tol,
            margin_pts=self.config.header_margin_pts,
            pad_x=self.config.header_pad_x,
            pad_y=self.config.header_pad_y,
        )

        if hdr is not None:
            x0, y0, x1, y1 = hdr
            ry2 = header_coord[3] * scale_y if header_coord else y1
            page_fitz.add_redact_annot(
                fitz.Rect(x0, y0, x1, max(y1, ry2)),
                fill=self.config.redaction_fill,
            )
        elif header_coord:
            self._add_redaction_box(
                page_fitz,
                header_coord[0],
                header_coord[1],
                header_coord[2],
                header_coord[3],
                scale_x,
                scale_y,
            )

    # @staticmethod
    def _add_redaction_box(
        self, page_fitz, x1: int, y1: int, x2: int, y2: int, scale_x: float, scale_y: float
    ):
        """Add a redaction box to a page."""
        if y2 <= y1 or x2 <= x1:
            return

        rx1, ry1 = x1 * scale_x, y1 * scale_y
        rx2, ry2 = x2 * scale_x, y2 * scale_y

        page_fitz.add_redact_annot(fitz.Rect(rx1, ry1, rx2, ry2), fill=self.config.redaction_fill)
