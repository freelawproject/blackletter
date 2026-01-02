"""Image processing for bbox tightening and detection."""

from typing import List, Optional, Tuple, Callable

import cv2
import numpy as np


class ImageProcessor:
    """Image processing for bbox tightening and detection."""

    @staticmethod
    def tighten_bbox_px(
        img_bgr: np.ndarray,
        coords: List[float],
        kind: str = "key",
        expand: int = 12,
        pad: int = 2,
        min_area: int = 40,
        min_rel_w: float = 0.45,
        min_rel_h: float = 0.45,
    ) -> List[float]:
        """Tighten bounding box using image content analysis.

        Args:
            img_bgr: Input image (BGR)
            coords: [x1, y1, x2, y2] in pixels
            kind: Detection type ("key", "textline", "hline", "edges")
            expand: How much to expand ROI around bbox
            pad: Padding around detected content
            min_area: Minimum contour area to consider
            min_rel_w: Minimum relative width (shrink guard)
            min_rel_h: Minimum relative height (shrink guard)

        Returns:
            Tightened [x1, y1, x2, y2] coordinates
        """
        H, W = img_bgr.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in coords]
        orig_w = max(1, x2 - x1)
        orig_h = max(1, y2 - y1)

        # Expand ROI
        ex1, ey1, ex2, ey2 = _clip_rect(x1 - expand, y1 - expand, x2 + expand, y2 + expand, W, H)
        if ex1 is None:
            return coords

        roi = img_bgr[ey1:ey2, ex1:ex2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        rr = None

        # Choose detection method
        if kind in ("key", "textline", "hline"):
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if th.mean() > 127:
                th = cv2.bitwise_not(th)

            if kind == "key":
                k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
                th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=1)
                rr = ImageProcessor._finish_from_mask(th, pick="largest", min_area=min_area)

            elif kind == "textline":
                k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)
                rr = ImageProcessor._finish_from_mask(th, pick="union", min_area=min_area)

            elif kind == "hline":
                kw = max(25, int(th.shape[1] * 0.25))
                hk = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))
                only_h = cv2.morphologyEx(th, cv2.MORPH_OPEN, hk, iterations=1)
                dk = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
                only_h = cv2.dilate(only_h, dk, iterations=1)
                rr = ImageProcessor._finish_from_mask(
                    only_h, pick="best_aspect", aspect_pref=lambda a: a, min_area=min_area
                )

        elif kind == "edges":
            edges = cv2.Canny(gray, 60, 180)
            dk = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.dilate(edges, dk, iterations=2)
            rr = ImageProcessor._finish_from_mask(edges, pick="union", min_area=min_area)

        if rr is None:
            return coords

        rx, ry, rw, rh = rr
        nx1 = ex1 + rx - pad
        ny1 = ey1 + ry - pad
        nx2 = ex1 + rx + rw + pad
        ny2 = ey1 + ry + rh + pad

        clipped = _clip_rect(nx1, ny1, nx2, ny2, W, H)
        if clipped is None:
            return coords

        nx1, ny1, nx2, ny2 = clipped
        new_w = nx2 - nx1
        new_h = ny2 - ny1

        # Shrink guard: if result is too small, revert to original
        if new_w < orig_w * min_rel_w or new_h < orig_h * min_rel_h:
            return coords

        return [float(nx1), float(ny1), float(nx2), float(ny2)]

    @staticmethod
    def _finish_from_mask(
        mask: np.ndarray,
        pick: str = "union",
        aspect_pref: Optional[Callable] = None,
        min_area: int = 40,
    ) -> Optional[Tuple[int, int, int, int]]:
        """Find bounding box from binary mask.

        Args:
            mask: Binary mask image
            pick: Selection strategy ("largest", "union", "best_aspect")
            aspect_pref: Function to score aspect ratio (if pick="best_aspect")
            min_area: Minimum contour area

        Returns:
            (x, y, width, height) or None
        """
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None

        boxes = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            rx, ry, rw, rh = cv2.boundingRect(c)
            boxes.append((rx, ry, rw, rh, area))

        if not boxes:
            return None

        if pick == "largest":
            return max(boxes, key=lambda b: b[4])[:4]

        if pick == "best_aspect" and aspect_pref is not None:
            best = None
            best_score = -1e9
            for rx, ry, rw, rh, area in boxes:
                aspect = rw / max(1, rh)
                score = aspect_pref(aspect) + 0.001 * area
                if score > best_score:
                    best_score = score
                    best = (rx, ry, rw, rh)
            return best

        # Union: merge all boxes
        rx0 = min(b[0] for b in boxes)
        ry0 = min(b[1] for b in boxes)
        rx1 = max(b[0] + b[2] for b in boxes)
        ry1 = max(b[1] + b[3] for b in boxes)
        return (rx0, ry0, rx1 - rx0, ry1 - ry0)


def _clip_rect(
    x1: float, y1: float, x2: float, y2: float, W: int, H: int
) -> Optional[Tuple[int, int, int, int]]:
    """Clip rectangle to image bounds."""
    x1 = max(0, min(W, int(x1)))
    x2 = max(0, min(W, int(x2)))
    y1 = max(0, min(H, int(y1)))
    y2 = max(0, min(H, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2
