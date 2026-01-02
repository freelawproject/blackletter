"""Image processing and coordinate utilities."""

import logging
from typing import List, Tuple, Optional

import cv2
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from blackletter.core.scanner import Detection, PageContext


logger = logging.getLogger(__name__)


def detect_columns_from_image(img_bgr: np.ndarray) -> Tuple[int, int, int, int, int]:
    """Detect left/right column boundaries from image using projection analysis.

    Returns:
        (LEFT_X1, LEFT_X2, RIGHT_X1, RIGHT_X2, center_X)
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Ignore header/footer regions
    crop_top_frac = 0.06
    crop_bot_frac = 0.06
    y1 = int(h * crop_top_frac)
    y2 = int(h * (1.0 - crop_bot_frac))
    roi = gray[y1:y2, :]

    # Binarize; ink=1
    _, bw = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = (bw == 0).astype(np.uint8)

    # Light dilation horizontally to stabilize projection
    k = max(1, int(w * 0.002))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * k + 1, 1))
    ink = cv2.dilate(ink, kernel, iterations=1)

    proj = ink.sum(axis=0).astype(np.float32)

    # Smooth projection
    win = max(5, int(w * 0.02))
    if win % 2 == 0:
        win += 1
    proj_smooth = cv2.GaussianBlur(proj.reshape(1, -1), (win, 1), 0).ravel()

    # Find lowest-ink valley near center
    center = w // 2
    band = int(w * 0.25)  # +/- 25% around center
    lo = max(0, center - band)
    hi = min(w, center + band)

    valley_center = lo + int(np.argmin(proj_smooth[lo:hi]))

    med = float(np.median(proj_smooth))
    thr = max(1.0, 0.15 * med)

    vx1 = valley_center
    while vx1 > 0 and proj_smooth[vx1] <= thr:
        vx1 -= 1
    vx2 = valley_center
    while vx2 < w - 1 and proj_smooth[vx2] <= thr:
        vx2 += 1

    # Ensure minimum gutter width
    min_vw = max(3, int(w * 0.01))
    if (vx2 - vx1) < min_vw:
        half = min_vw // 2
        vx1 = max(0, valley_center - half)
        vx2 = min(w, valley_center + half)

    center_X = (vx1 + vx2) // 2

    # Determine where text exists on each side
    text_thr = max(1.0, 0.20 * float(np.mean(proj_smooth)))

    def first_last_above(arr: np.ndarray, offset: int = 0):
        idx = np.where(arr > text_thr)[0]
        if idx.size == 0:
            return None
        return int(idx[0] + offset), int(idx[-1] + offset)

    left_bounds = first_last_above(proj_smooth[:center_X], 0)
    right_bounds = first_last_above(proj_smooth[center_X:], center_X)

    if left_bounds is None or right_bounds is None:
        raise ValueError("Image-based column detection found empty side")

    left_x1, left_x2 = left_bounds
    right_x1, right_x2 = right_bounds

    if not (0 <= left_x1 < left_x2 < center_X < right_x1 < right_x2 <= w):
        raise ValueError(
            f"Image-based bounds look wrong: {(left_x1, left_x2, center_X, right_x1, right_x2)}"
        )

    return (left_x1, left_x2, right_x1, right_x2, center_X)


def fallback_column_detection(w_img: int) -> Tuple[int, int, int, int, int]:
    """Fallback 50/50 column split when image detection fails."""
    side_pad = max(20, int(w_img * 0.03))
    gutter = max(10, int(w_img * 0.02))
    center_X = w_img // 2
    left_x1 = side_pad
    left_x2 = max(left_x1 + 10, center_X - gutter)
    right_x1 = min(w_img - side_pad - 10, center_X + gutter)
    right_x2 = w_img - side_pad
    return (left_x1, left_x2, right_x1, right_x2, center_X)


def column_for_coords(coords: List[float], center_X: int) -> str:
    """Determine if coordinates are in LEFT or RIGHT column.

    Args:
        coords: [x1, y1, x2, y2]
        center_X: x-coordinate of column split

    Returns:
        "LEFT" or "RIGHT"
    """
    x1, _, x2, _ = coords
    center_x = (x1 + x2) / 2
    return "LEFT" if center_x < center_X else "RIGHT"


def process_brackets(
    page,
    bracket: "Detection",
    page_context: "PageContext",
) -> Optional[tuple[list[float], str]]:
    """Process bracket detections and assign to columns.

    Brackets are editorial marks (e.g., [sic], [note]) that should be
    attributed to a specific column.

    Args:
        page: pdfplumber page object
        img: cv2 image
        coords: [x1, y1, x2, y2] in image pixels
        conf: confidence score
        page_dimension:
        columns:

    Returns:
        (coords, col) tuple or None if bracket should be filtered
    """

    page_dimension = page_context.page_dimensions
    columns = page_context.columns

    # Extract text from bracket region
    pdf_bbox = _yolo_to_pdf_bbox(bracket.coords, *page_dimension)
    box_text = _extract_bracket_text(page, pdf_bbox)

    # Validate text content
    if not _passes_bracket_text_filters(box_text, int(bracket.confidence * 1000)):
        return None

    # Determine column
    col = column_for_coords(bracket.coords, columns[-1])

    # Clamp bracket to column bounds
    clamped = _clamp_bracket_to_column(bracket.coords, col, columns)

    if clamped is None:
        return None

    # Tighten bbox using image content
    from blackletter.utils.image import ImageProcessor

    tightened = ImageProcessor.tighten_bbox_px(
        page_context.img,
        clamped,
        kind="textline",
        expand=0,
        pad=0,
        min_area=10,
        min_rel_w=0.45,
        min_rel_h=0.15,
    )

    return (tightened, col)


# --- Bracket text extraction helpers ---

BRACKET_CHARS = set("[]0123456789")
BRACKETS_ONLY = set("[]")


def _extract_bracket_text(page, pdf_bbox: Tuple[float, float, float, float]) -> str:
    """Extract text from a bracket region in PDF coordinates."""
    cropped = page.crop(pdf_bbox, strict=False)
    try:
        return (
            cropped.extract_text(
                x_tolerance=1,
                y_tolerance=2,
                layout=False,
            )
            or ""
        )
    except Exception:
        return ""


def _passes_bracket_text_filters(text: str, fc: int) -> bool:
    """Validate bracket text content and confidence.

    Args:
        text: Extracted text from bracket
        fc: Confidence * 1000 (integer confidence score)
    """
    if not text:
        return False

    # Must contain at least one bracket or digit
    if not (set(text) & BRACKET_CHARS):
        return False

    # Quick rejects
    if text[0] in ("¶", "$"):
        return False
    if len(text) < 3 or len(text) > 10:
        return False
    if len(text) > 1 and text[1] == "¶":
        return False

    # If no brackets, require higher confidence
    if not (set(text) & BRACKETS_ONLY) and fc < 150:
        return False

    if fc < 25:
        return False

    return True


def _yolo_to_pdf_bbox(
    coords: List[float], pdf_w: float, pdf_h: float, img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    """Convert YOLO coords (pixels) to PDF coords."""
    scale_x = pdf_w / img_w
    scale_y = pdf_h / img_h
    x1, y1, x2, y2 = coords
    return (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)


def _clamp_bracket_to_column(
    coords: List[float],
    col: str,
    columns: tuple,
) -> Optional[List[float]]:
    """Clamp bracket coordinates to its column boundaries."""
    x1, y1, x2, y2 = coords
    LEFT_X1, LEFT_X2, RIGHT_X1, RIGHT_X2, _ = columns

    col_x1 = LEFT_X1 if col == "LEFT" else RIGHT_X1
    col_x2 = LEFT_X2 if col == "LEFT" else RIGHT_X2

    x1 = max(x1, col_x1 + 5)
    x2 = min(x2, col_x2 - 2)

    # If we collapsed the box, signal drop
    if x2 <= x1 + 2:
        return None

    return [x1, y1, x2, y2]
