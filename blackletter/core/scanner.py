"""Phase 1: PDF scanning and object detection using YOLO."""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pdfplumber
from ultralytics import YOLO

from blackletter.config import RedactionConfig
from blackletter.utils import processing

logger = logging.getLogger(__name__)


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

    def scan(self, pdf_path: Path) -> Tuple[List[Dict], Dict, Dict]:
        """Scan all pages and detect objects.

        Returns:
            - global_objects: List of detected objects across all pages
            - page_dimensions: Dict mapping page_idx to (pdf_w, pdf_h, img_w, img_h)
            - page_columns_px: Dict mapping page_idx to column boundaries in pixels
        """
        logger.info("Starting PHASE 1: Scanning all pages")

        global_objects = []
        page_dimensions = {}
        page_columns_px = {}

        with pdfplumber.open(pdf_path) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                logger.info(f"Scanning page {page_idx + 1}/{len(pdf.pages)}")

                pil_img = page.to_image(resolution=self.config.dpi).original
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                h_img, w_img = img.shape[:2]

                page_dimensions[page_idx] = (
                page.width, page.height, w_img, h_img)
                page_columns_px[page_idx] = self._detect_columns(
                    page, w_img, page.width
                )

                page_objects = self._detect_objects(
                    page,
                    img,
                    page_idx,
                    page.width,
                    page.height,
                    w_img,
                    h_img,
                    page_columns_px[page_idx],
                )
                global_objects.extend(page_objects)

        logger.info(f"Detected {len(global_objects)} total objects")
        return global_objects, page_dimensions, page_columns_px

    def _detect_columns(
            self, page, w_img: int, pdf_width: float
    ) -> Tuple[int, int, int, int, int]:
        """Detect left/right column boundaries (pixel x coords)."""
        try:
            pil_img = page.to_image(resolution=self.config.dpi).original
            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            return processing.detect_columns_from_image(img_bgr)

        except Exception as e:
            logger.warning(
                f"Image-based column detection failed; using 50/50 fallback: {e}"
            )
            return processing.fallback_column_detection(w_img)

    def _detect_objects(
            self,
            page,
            img,
            page_idx: int,
            pdf_w: float,
            pdf_h: float,
            img_w: int,
            img_h: int,
            columns: Tuple,
    ) -> List[Dict]:
        """Detect objects on a single page using YOLO."""
        # LEFT_X1, LEFT_X2, RIGHT_X1, RIGHT_X2, split_x = columns
        results = self.model(
            img, conf=self.config.low_confidence_threshold, verbose=False
        )

        page_objects = []
        for r in results:
            for box in r.boxes:
                coords = box.xyxy[0].tolist()
                conf = float(box.conf[0].item())
                label = self.model.names[int(box.cls[0].item())]

                if not self._passes_confidence_filters(label, conf):
                    continue

                if label == "brackets":
                    result = processing.process_brackets(
                        page=page,
                        img=img,
                        coords=coords,
                        conf=conf,
                        pdf_w=pdf_w,
                        pdf_h=pdf_h,
                        img_w=img_w,
                        img_h=img_h,
                        split_x=split_x,
                        page_brackets=[],
                        LEFT_X1=LEFT_X1,
                        LEFT_X2=LEFT_X2,
                        RIGHT_X1=RIGHT_X1,
                        RIGHT_X2=RIGHT_X2,
                    )
                    if result is None:
                        continue
                    coords, col = result
                else:
                    if label not in self.TARGET_LABELS:
                        continue
                    col = processing.column_for_coords(coords, split_x)

                page_objects.append(
                    {
                        "page_index": page_idx,
                        "label": label,
                        "coords": coords,
                        "col": col,
                        "y1": coords[1],
                        "y2": coords[3],
                    }
                )

        return page_objects

    def _passes_confidence_filters(self, label: str, conf: float) -> bool:
        """Check if detection passes confidence thresholds."""
        return conf >= self.config.confidence_threshold