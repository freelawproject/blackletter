"""Blackletter-native PaddleOCR engine for ocrmypdf.

An ocrmypdf ``OcrEngine`` backed by PaddleOCR, selectable via
``engine="paddle"`` on blackletter's OCR API. It emits hOCR that ocrmypdf
turns into the invisible text layer, so it is driven with
``pdf_renderer="hocr"``.

Design notes (why this exists rather than a third-party plugin):
- Reuses the SAME PP-OCRv5 *server* model configuration as
  ``blackletter.analyze`` (the page-number reader). A worker image that
  bakes those models for analyze reuses them here with no extra runtime
  download.
- The PaddleOCR instance is built once per process and cached, so a
  multi-thousand-page document does not re-instantiate the model per page.
- ``enable_mkldnn=False`` mirrors ``analyze`` and avoids paddlepaddle's
  PIR + oneDNN crash on the CPU path. GPU is used automatically when
  paddlepaddle is CUDA-enabled (no explicit device kwarg, matching
  ``analyze``).
"""

from __future__ import annotations

import html
from pathlib import Path

from ocrmypdf import hookimpl
from ocrmypdf.pluginspec import OcrEngine, OrientationConfidence

# Module name to pass to ocrmypdf's ``plugins=[...]`` argument.
PLUGIN = "blackletter.paddle_ocr"

# Built once per process and reused across pages (see module docstring).
_PADDLE = None


def _get_paddle():
    """Return a cached PaddleOCR instance (PP-OCRv5 server models).

    Matches ``blackletter.analyze``'s configuration (server detector +
    server recognizer), so the baked weights are shared, line-box quality
    for the downstream redaction-rect tightening is preserved, and text
    accuracy is as high as PP-OCRv5 offers. The recognizer was benchmarked
    against the lighter mobile model, which did not speed up the
    (CPU-bound) pipeline, so the accurate server model is kept.
    ``enable_mkldnn=False`` mirrors analyze and avoids paddle's PIR + oneDNN
    crash on the CPU path; the GPU is used automatically when paddlepaddle
    is CUDA-enabled.

    :returns: A configured ``PaddleOCR`` instance.
    """
    global _PADDLE
    if _PADDLE is None:
        import time

        from paddleocr import PaddleOCR

        print("  paddle: building PaddleOCR (PP-OCRv5 server)...", flush=True)
        t0 = time.time()
        _PADDLE = PaddleOCR(
            text_detection_model_name="PP-OCRv5_server_det",
            text_recognition_model_name="PP-OCRv5_server_rec",
            use_textline_orientation=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            enable_mkldnn=False,  # avoids paddle PIR+oneDNN crash on CPU
        )
        print(f"  paddle: model ready ({time.time() - t0:.1f}s)", flush=True)
    return _PADDLE


def _bbox_from_poly(poly) -> tuple[int, int, int, int]:
    """Return the axis-aligned ``(x0, y0, x1, y1)`` of a 4-point polygon."""
    xs = [int(p[0]) for p in poly]
    ys = [int(p[1]) for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def _recognize(image_path: Path) -> tuple[int, int, list[tuple[int, int, int, int, str, float]]]:
    """Run PaddleOCR on a page image and return ``(width, height, lines)``.

    Each line is ``(x0, y0, x1, y1, text, conf)`` in pixel coordinates,
    where ``conf`` is a 0-100 confidence.

    :param image_path: Path to the page image to OCR.
    :returns: Image width, height, and the recognized line boxes.
    """
    import time

    import numpy as np
    from PIL import Image

    with Image.open(image_path) as im:
        pil = im.convert("RGB")
        width, height = pil.size
        # ascontiguousarray: PaddleOCR/paddle want a contiguous BGR buffer;
        # the [::-1] channel flip alone yields a negative-stride view.
        arr = np.ascontiguousarray(np.array(pil)[:, :, ::-1])  # RGB -> BGR

    paddle = _get_paddle()
    print(f"  paddle: recognizing {image_path.name} ({width}x{height})...", flush=True)
    t0 = time.time()
    results = paddle.predict(arr)
    dt = time.time() - t0
    lines: list[tuple[int, int, int, int, str, float]] = []
    if not results:
        print(f"  paddle: {image_path.name} -> 0 lines in {dt:.1f}s", flush=True)
        return width, height, lines

    r = results[0]
    texts = r.get("rec_texts") or []
    scores = r.get("rec_scores") or []
    polys = r.get("rec_polys")
    if polys is None:
        polys = r.get("dt_polys")
    boxes = r.get("rec_boxes")

    for i, text in enumerate(texts):
        if not text or not text.strip():
            continue
        if polys is not None and i < len(polys):
            box = _bbox_from_poly(polys[i])
        elif boxes is not None and i < len(boxes):
            b = boxes[i]
            box = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
        else:
            continue
        conf = float(scores[i]) * 100.0 if i < len(scores) else 0.0
        lines.append((box[0], box[1], box[2], box[3], text.strip(), conf))
    print(f"  paddle: {image_path.name} -> {len(lines)} lines in {dt:.1f}s", flush=True)
    return width, height, lines


def _reading_order(
    lines: list[tuple[int, int, int, int, str, float]], width: int, height: int
) -> list[tuple[int, int, int, int, str, float]]:
    """Sort recognized lines into human reading order (column-aware).

    PaddleOCR's OCR pipeline detects and recognizes text lines but does no
    layout analysis, so a two-column page comes back roughly row-major. Left
    in that order the invisible text layer reads (and selects) straight
    across both columns. This applies a recursive XY-cut: within each region
    it first looks for a clean vertical gap (a column gutter) and splits
    left/right, otherwise a clean horizontal gap (splitting stacked blocks
    such as a running header above the body), recursing until no gap remains
    and the leftover lines are read top-to-bottom. Trying the vertical cut
    first yields column-major order: the whole left column, then the whole
    right column. A full-width line (header/footer) blocks the vertical cut,
    so it is peeled off by a horizontal cut and read in its own band.

    :param lines: List of ``(x0, y0, x1, y1, text, conf)`` tuples.
    :param width: Page image width in pixels.
    :param height: Page image height in pixels.
    :returns: The same tuples reordered into reading order.
    """
    h_gap = max(6, int(height * 0.006))  # min block/line separation

    def vsplit(group):
        # Split into two columns at a vertical gutter: the x in the central
        # band crossed by (almost) no line box. Column gutters in dense text
        # are narrow, so a fixed gap threshold is unreliable; scanning for
        # minimum coverage finds the gutter wherever it sits.
        n = len(group)
        if n < 6:
            return None
        x_lo = min(ln[0] for ln in group)
        x_hi = max(ln[2] for ln in group)
        if x_hi - x_lo < width * 0.4:  # too narrow to hold two columns
            return None
        lo = int(x_lo + 0.25 * (x_hi - x_lo))
        hi = int(x_lo + 0.75 * (x_hi - x_lo))
        step = max(1, (hi - lo) // 60)
        best_x, best_cov = None, n + 1
        for x in range(lo, hi + 1, step):
            cov = sum(1 for ln in group if ln[0] <= x <= ln[2])
            if cov < best_cov:
                best_cov, best_x = cov, x
        # Accept only a near-clean gutter (a few full-width headers may cross)
        # that divides the lines into two substantial columns.
        if best_x is None or best_cov > max(1, int(0.03 * n)):
            return None
        left = [ln for ln in group if (ln[0] + ln[2]) / 2 < best_x]
        right = [ln for ln in group if (ln[0] + ln[2]) / 2 >= best_x]
        if min(len(left), len(right)) < max(3, int(0.15 * n)):
            return None
        return left, right

    def hsplit(group):
        # Split stacked blocks at the widest clean horizontal gap (e.g. a
        # running header above the body, or a paragraph break).
        ordered = sorted(group, key=lambda ln: ln[1])
        run_end = ordered[0][3]
        best_gap, best_at = 0, None
        for k in range(1, len(ordered)):
            gap = ordered[k][1] - run_end
            if gap > best_gap:
                best_gap, best_at = gap, k
            run_end = max(run_end, ordered[k][3])
        if best_at is not None and best_gap >= h_gap:
            return ordered[:best_at], ordered[best_at:]
        return None

    def cut(group, depth):
        if len(group) <= 1 or depth > 40:
            return sorted(group, key=lambda ln: (ln[1], ln[0]))
        parts = vsplit(group) or hsplit(group)  # columns before stacked blocks
        if parts is None:
            return sorted(group, key=lambda ln: (ln[1], ln[0]))
        first, second = parts
        return cut(first, depth + 1) + cut(second, depth + 1)

    return cut(list(lines), 0)


def _group_paragraphs(lines):
    """Group reading-ordered lines into paragraphs at column/region breaks.

    Within a column the top edge advances downward; a line whose top jumps
    back above the previous line marks the start of a new column or region,
    so a fresh paragraph begins there. Keeps the text layer's structure close
    to Tesseract's (one block per column) for well-behaved selection.

    :param lines: Reading-ordered ``(x0, y0, x1, y1, text, conf)`` tuples.
    :returns: List of paragraphs, each a list of line tuples.
    """
    paragraphs: list[list] = []
    current: list = []
    prev_top = None
    for ln in lines:
        if prev_top is not None and ln[1] < prev_top - 5:
            paragraphs.append(current)
            current = []
        current.append(ln)
        prev_top = ln[1]
    if current:
        paragraphs.append(current)
    return paragraphs


def _build_hocr(image_path, lines, width: int, height: int) -> str:
    """Build a line-level hOCR document from recognized lines.

    Lines are first put into reading order and grouped into per-column
    paragraphs so the resulting text layer selects column-by-column rather
    than straight across a multi-column page.

    :param image_path: Source image path (recorded in the page title).
    :param lines: List of ``(x0, y0, x1, y1, text, conf)`` tuples.
    :param width: Image width in pixels.
    :param height: Image height in pixels.
    :returns: The hOCR document as a string.
    """
    name = html.escape(str(image_path))
    out = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" '
        '"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">',
        '<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">',
        "<head><title></title>",
        '<meta http-equiv="Content-Type" content="text/html;charset=utf-8" />',
        '<meta name="ocr-system" content="paddleocr" />',
        '<meta name="ocr-capabilities" content="ocr_page ocr_carea ocr_par ocr_line ocrx_word" />',
        "</head><body>",
        f"<div class='ocr_page' id='page_1' title='image \"{name}\"; "
        f"bbox 0 0 {width} {height}; ppageno 0'>",
        f"<div class='ocr_carea' id='block_1' title='bbox 0 0 {width} {height}'>",
    ]
    line_no = 0
    for par_no, paragraph in enumerate(_group_paragraphs(_reading_order(lines, width, height)), 1):
        px0 = min(ln[0] for ln in paragraph)
        py0 = min(ln[1] for ln in paragraph)
        px1 = max(ln[2] for ln in paragraph)
        py1 = max(ln[3] for ln in paragraph)
        out.append(
            f"<p class='ocr_par' id='par_{par_no}' lang='eng' title='bbox {px0} {py0} {px1} {py1}'>"
        )
        for x0, y0, x1, y1, text, conf in paragraph:
            line_no += 1
            esc = html.escape(text)
            c = int(conf)
            out.append(
                f"<span class='ocr_line' id='line_{line_no}' "
                f"title='bbox {x0} {y0} {x1} {y1}; baseline 0 0'>"
            )
            out.append(
                f"<span class='ocrx_word' id='word_{line_no}_1' "
                f"title='bbox {x0} {y0} {x1} {y1}; x_wconf {c}'>{esc}</span>"
            )
            out.append("</span>")
        out.append("</p>")
    out += ["</div>", "</div></body></html>"]
    return "\n".join(out)


class PaddleOcrEngine(OcrEngine):
    """ocrmypdf OCR engine backed by PaddleOCR (hOCR renderer only)."""

    @staticmethod
    def version() -> str:
        """Return the PaddleOCR version string."""
        import paddleocr

        return getattr(paddleocr, "__version__", "unknown")

    @staticmethod
    def creator_tag(options) -> str:
        """Return the PDF creator tag for this engine."""
        return f"PaddleOCR {PaddleOcrEngine.version()}"

    def __str__(self) -> str:
        """Return a human-readable engine name."""
        return f"PaddleOCR {self.version()}"

    @staticmethod
    def languages(options) -> set[str]:
        """Return the set of supported language codes."""
        return {"eng"}

    @staticmethod
    def get_orientation(input_file, options) -> OrientationConfidence:
        """Return page orientation (disabled, so always no rotation)."""
        return OrientationConfidence(angle=0, confidence=0.0)

    @staticmethod
    def generate_hocr(input_file, output_hocr, output_text, options) -> None:
        """Run PaddleOCR and write the hOCR file plus a text sidecar."""
        width, height, lines = _recognize(Path(input_file))
        Path(output_hocr).write_text(
            _build_hocr(input_file, lines, width, height), encoding="utf-8"
        )
        text = "\n".join(line[4] for line in lines)
        Path(output_text).write_text(text + "\n", encoding="utf-8")

    @staticmethod
    def generate_pdf(input_file, output_pdf, output_text, options) -> None:
        """Not supported; this engine only emits hOCR."""
        raise NotImplementedError(
            "PaddleOcrEngine supports only the hOCR renderer; "
            "run ocrmypdf with pdf_renderer='hocr'."
        )


@hookimpl
def get_ocr_engine(options=None) -> OcrEngine:
    """ocrmypdf hook: provide the PaddleOCR engine.

    :param options: ocrmypdf options (unused).
    :returns: A ``PaddleOcrEngine`` instance.
    """
    return PaddleOcrEngine()
