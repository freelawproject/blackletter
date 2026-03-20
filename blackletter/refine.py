"""Refine block-level redaction rects to line-level using docTR word detection."""

from __future__ import annotations

import logging

import fitz
import numpy as np

logger = logging.getLogger(__name__)

DPI = 200
_SCALE = DPI / 72  # PDF points → pixels


def _get_doctr_model():
    """Lazy-load docTR detection model (singleton)."""
    if not hasattr(_get_doctr_model, "_model"):
        from doctr.models import detection_predictor

        _get_doctr_model._model = detection_predictor(arch="db_resnet50", pretrained=True)
    return _get_doctr_model._model


def _refine_single_rect(
    det_model,
    page_img: np.ndarray,
    rect_px: tuple[float, float, float, float],
    y_merge: int = 10,
    min_gap: int = 10,
) -> list[tuple[float, float, float, float]]:
    """Run docTR on a single pixel-coord rect crop and return line-level rects."""
    rx1, ry1, rx2, ry2 = int(rect_px[0]), int(rect_px[1]), int(rect_px[2]), int(rect_px[3])

    if (rx2 - rx1) < 10 or (ry2 - ry1) < 10:
        return [rect_px]

    crop = page_img[ry1:ry2, rx1:rx2]
    if crop.size == 0:
        return [rect_px]

    ch, cw = crop.shape[:2]
    det_result = det_model([crop])[0]
    words_arr = det_result.get("words", np.empty((0, 5)))
    if len(words_arr) == 0:
        return [rect_px]

    # Convert relative coords to pixel coords in crop
    word_boxes = [(w[0] * cw, w[1] * ch, w[2] * cw, w[3] * ch) for w in words_arr]

    if not word_boxes:
        return [rect_px]

    # Group into lines by y-center
    used = [False] * len(word_boxes)
    lines = []
    for i in range(len(word_boxes)):
        if used[i]:
            continue
        yc_i = (word_boxes[i][1] + word_boxes[i][3]) / 2
        group = [word_boxes[i]]
        used[i] = True
        for j in range(i + 1, len(word_boxes)):
            if used[j]:
                continue
            yc_j = (word_boxes[j][1] + word_boxes[j][3]) / 2
            if abs(yc_i - yc_j) < y_merge:
                group.append(word_boxes[j])
                used[j] = True
        ly1 = min(w[1] for w in group)
        ly2 = max(w[3] for w in group)
        lines.append([ly1, ly2])
    lines.sort(key=lambda r: r[0])

    # Merge lines with small gaps
    merged = [list(lines[0])]
    for ln in lines[1:]:
        prev = merged[-1]
        if ln[0] - prev[1] < min_gap:
            prev[1] = max(prev[1], ln[1])
        else:
            merged.append(list(ln))

    # Convert crop-relative y back to page coords, keep full rect width
    return [(rx1, ry1 + m[0], rx2, ry1 + m[1]) for m in merged]


def refine_headnote_rects(
    pdf_path: str | fitz.Document,
    headnote_rects: list[tuple[int, fitz.Rect]],
    pages_by_index: dict,
) -> list[tuple[int, fitz.Rect]]:
    """Refine block-level headnote rects to line-level using docTR.

    Args:
        pdf_path: Path to PDF or open fitz.Document.
        headnote_rects: List of (page_index, fitz.Rect) in PDF points.
        pages_by_index: Page lookup dict for scale info.

    Returns:
        List of (page_index, fitz.Rect) with tighter line-level rects.
    """
    if not headnote_rects:
        return headnote_rects

    det_model = _get_doctr_model()

    # Group rects by page
    by_page: dict[int, list[fitz.Rect]] = {}
    for page_idx, rect in headnote_rects:
        by_page.setdefault(page_idx, []).append(rect)

    # Render needed pages
    if isinstance(pdf_path, fitz.Document):
        pdf = pdf_path
        own_pdf = False
    else:
        pdf = fitz.open(str(pdf_path))
        own_pdf = True

    mat = fitz.Matrix(_SCALE, _SCALE)
    result: list[tuple[int, fitz.Rect]] = []

    total_pages = len(by_page)
    done_pages = 0
    print(f"    docTR: refining {len(headnote_rects)} rects on {total_pages} pages...", flush=True)
    for page_idx, rects in by_page.items():
        done_pages += 1
        if done_pages % 10 == 0 or done_pages == total_pages:
            print(f"    docTR: {done_pages}/{total_pages} pages", flush=True)
        fitz_page = pdf[page_idx]
        pix = fitz_page.get_pixmap(matrix=mat)
        page_img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)

        for rect in rects:
            # PDF points → pixel coords
            px_rect = (
                rect.x0 * _SCALE,
                rect.y0 * _SCALE,
                rect.x1 * _SCALE,
                rect.y1 * _SCALE,
            )

            refined_px = _refine_single_rect(det_model, page_img, px_rect)

            # Pixel coords → PDF points
            for rpx in refined_px:
                pdf_rect = fitz.Rect(
                    rpx[0] / _SCALE,
                    rpx[1] / _SCALE,
                    rpx[2] / _SCALE,
                    rpx[3] / _SCALE,
                )
                if pdf_rect.y0 < pdf_rect.y1 and pdf_rect.x0 < pdf_rect.x1:
                    result.append((page_idx, pdf_rect))

    if own_pdf:
        pdf.close()

    logger.info(
        "docTR refinement: %d block rects → %d line rects",
        len(headnote_rects),
        len(result),
    )
    return result


class PageRefiner:
    """Caches rendered page images so multiple rects on the same page share one render.

    Usage::

        refiner = PageRefiner(pdf)
        refined_rects = refiner.refine(page_idx, rect)  # returns list[fitz.Rect]
    """

    def __init__(self, pdf: fitz.Document):
        self._pdf = pdf
        self._det_model = _get_doctr_model()
        self._mat = fitz.Matrix(_SCALE, _SCALE)
        self._cache: dict[int, np.ndarray] = {}

    def _get_page_img(self, page_idx: int) -> np.ndarray:
        if page_idx not in self._cache:
            pix = self._pdf[page_idx].get_pixmap(matrix=self._mat)
            self._cache[page_idx] = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, 3
            )
        return self._cache[page_idx]

    def refine(self, page_idx: int, rect: fitz.Rect) -> list[fitz.Rect]:
        """Refine a single PDF-point rect into line-level rects using docTR."""
        page_img = self._get_page_img(page_idx)
        px_rect = (
            rect.x0 * _SCALE,
            rect.y0 * _SCALE,
            rect.x1 * _SCALE,
            rect.y1 * _SCALE,
        )
        refined_px = _refine_single_rect(self._det_model, page_img, px_rect)
        result = []
        for rpx in refined_px:
            pdf_rect = fitz.Rect(
                rpx[0] / _SCALE,
                rpx[1] / _SCALE,
                rpx[2] / _SCALE,
                rpx[3] / _SCALE,
            )
            if pdf_rect.y0 < pdf_rect.y1 and pdf_rect.x0 < pdf_rect.x1:
                result.append(pdf_rect)
        return result if result else [rect]

    def clear_cache(self):
        """Free cached page images."""
        self._cache.clear()
