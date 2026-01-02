"""Phase 1: PDF scanning and object detection using YOLO."""

import logging

from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pdfplumber
from ultralytics import YOLO

from blackletter.config import RedactionConfig
from blackletter.utils import processing

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """A single YOLO detection."""

    coords: List[float]
    confidence: float
    label: str
    col: Optional[str] = ""
    page_index: int = 0


@dataclass
class Opinion:
    """An opinion identified in the document."""

    caption: Detection
    key: Detection = None
    line: Optional[Detection] = None
    headmatter: Optional[Detection] = None
    case_name: str = ""


@dataclass
class PageContext:
    """A single YOLO detection."""

    plumber_page: object
    img: np.ndarray
    index: int
    pdf_pg_width: float
    pdf_pg_height: float
    img_width: int
    img_height: int
    column_left_x1: float = 0.0
    column_left_x2: float = 0.0
    column_right_x1: float = 0.0
    column_right_x2: float = 0.0
    midpoint: float = 0.0
    page_objects: List[Detection] = field(default_factory=list)
    header_bottom: Optional[float] = None  # y2 of header, if present
    footer_top: Optional[float] = None  # y1 of footnotes, if present

    def extract_bounds(self):
        """Extract header and footer bounds from page_objects."""
        for obj in self.page_objects:
            if obj.label == "header":
                self.header_bottom = obj.coords[3]
            elif obj.label == "footnotes":
                if self.footer_top is None:
                    self.footer_top = obj.coords[1]
                else:
                    self.footer_top = min(self.footer_top, obj.coords[1])

    @property
    def page_dimensions(self):
        return (self.pdf_pg_width, self.pdf_pg_height, self.img_width, self.img_height)

    @property
    def columns(self):
        return (
            self.column_left_x1,
            self.column_left_x2,
            self.column_right_x1,
            self.column_right_x2,
            self.midpoint,
        )


@dataclass
class Document:
    pages: list[PageContext] = field(default_factory=list)
    opinions: list[Opinion] = field(default_factory=list)
    pdf_path: Path = None
    redacted_pdf_path: Path = None
    volume: Optional[str] = None
    reporter: Optional[str] = None
    first_page: Optional[int] = 1

    def sort_all_objects(self):
        """Sort page_objects on each page by column and y position."""
        for page in self.pages:
            page.page_objects = sorted(page.page_objects, key=lambda o: (o.col, o.coords[1]))
            page.extract_bounds()

    def add_opinion(self, start: Detection, end: Detection, midpoint: Optional[Detection] = None):
        """Add an opinion to the document."""
        self.opinions.append(Opinion(start, end, midpoint))

    def assign_case_names(self):
        """Assign case names to all opinions, sorted by page and column."""
        if not self.opinions:
            return

        COL_ORDER = {"LEFT": 0, "RIGHT": 1}

        def sort_key(opinion):
            start = opinion.caption
            page = start.page_index
            col = COL_ORDER.get(start.col, 99)
            y1 = start.coords[1]
            return (page, col, y1)

        self.opinions.sort(key=sort_key)

        page_counter = {}
        for opinion in self.opinions:
            first_page = opinion.caption.page_index + self.first_page
            counter = page_counter.get(first_page, 0) + 1
            page_counter[first_page] = counter
            opinion.case_name = f"{first_page:04d}-{counter:02d}"

    def get_filler_pages(self) -> set[int]:
        """Identify pages between caption and line/headmatter (filler pages)."""
        filler_pages = set()

        for opinion in self.opinions:
            end = opinion.line or opinion.headmatter
            if end is None:
                continue

            start_pg = opinion.caption.page_index
            end_pg = end.page_index

            # Pages strictly between caption and line/headmatter are filler
            for pg in range(start_pg + 1, end_pg):
                filler_pages.add(pg)

        return filler_pages


class PDFScanner:
    """Scans PDF pages and detects objects using YOLO."""

    TARGET_LABELS = {
        "caption",
        "line",
        "headmatter",
        "Key",
        "brackets",
        "order",
        "header",
        "footnotes",
    }

    def __init__(self, config: RedactionConfig, model: YOLO = None):
        self.config = config
        self.model = model or YOLO(config.MODEL_PATH)

    def scan(self, document: Document) -> Document:
        """Scan all pages and detect objects.

        :param document: The document to scan.

        :return: A document with detected objects.
        """
        logger.info("Starting PHASE 1: Scanning all pages")

        with pdfplumber.open(document.pdf_path) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                if page_idx > 25:
                    break
                logger.info(f"Scanning page {page_idx + 1}/{len(pdf.pages)}")

                pil_img = page.to_image(resolution=self.config.dpi).original
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                h_img, w_img = img.shape[:2]
                x1, x2, x3, x4, midpoint = self._detect_columns(page, w_img, page.width)

                page_context = PageContext(
                    plumber_page=page,
                    img=img,
                    index=page_idx,
                    pdf_pg_width=page.width,
                    pdf_pg_height=page.height,
                    img_width=w_img,
                    img_height=h_img,
                    column_left_x1=x1,
                    column_left_x2=x2,
                    column_right_x1=x3,
                    column_right_x2=x4,
                    midpoint=midpoint,
                )

                page_context = self._detect_objects(page_context)
                document.pages.append(page_context)

        # logger.info(f"Detected {len(global_objects)} total objects")
        logger.info(f"Detected XXXX total objects")

        return document

    def _detect_columns(self, page, w_img: int, pdf_width: float) -> Tuple[int, int, int, int, int]:
        """Detect left/right column boundaries (pixel x coords)."""
        try:
            pil_img = page.to_image(resolution=self.config.dpi).original
            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            return processing.detect_columns_from_image(img_bgr)

        except Exception as e:
            logger.warning(f"Image-based column detection failed; using 50/50 fallback: {e}")
            return processing.fallback_column_detection(w_img)

    def _detect_objects(
        self,
        page_context: PageContext,
    ) -> PageContext:
        """Detect objects on a single page using YOLO."""
        results = self.model(
            page_context.img, conf=self.config.low_confidence_threshold, verbose=False
        )
        columns = (
            page_context.column_left_x1,
            page_context.column_left_x2,
            page_context.column_right_x1,
            page_context.column_right_x2,
            page_context.midpoint,
        )
        for r in results:
            for box in r.boxes:
                obj = Detection(
                    coords=box.xyxy[0].tolist(),
                    confidence=float(box.conf[0].item()),
                    label=self.model.names[int(box.cls[0].item())],
                    page_index=page_context.index,
                )

                if not self._passes_confidence_filters(obj):
                    continue

                if obj.label == "brackets":
                    result = processing.process_brackets(
                        page=page_context.plumber_page,
                        bracket=obj,
                        page_context=page_context,
                    )
                    if result is None:
                        continue

                    coords, col = result
                    obj.coords = coords
                    obj.col = col

                else:
                    if obj.label not in self.TARGET_LABELS:
                        continue

                    obj.col = processing.column_for_coords(obj.coords, columns[-1])
                page_context.page_objects.append(obj)

        return page_context

    def _passes_confidence_filters(self, obj: Detection) -> bool:
        """Check if detection passes confidence thresholds."""
        return obj.confidence >= self.config.confidence_threshold
