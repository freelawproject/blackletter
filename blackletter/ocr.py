"""OCR preprocessing for image-only PDFs.

Detects whether a PDF has an embedded text layer. If not, downsamples
page images to a target size (~148 KB/page) and runs ocrmypdf to add
a perfectly aligned invisible text layer.

Two-step pipeline:
  1. Extract + downsample/compress each page image to roughly the
     target size (single resize + encode per image)
  2. Run ocrmypdf on the downsampled PDF so the OCR text layer
     coordinates match the final images exactly

Adapted from scanning-utils/process_scan.py.
"""

from __future__ import annotations

import contextlib
import io
import logging
import tempfile
from collections.abc import Callable
from pathlib import Path

import fitz
from PIL import Image

# Suppress noisy PIL EXIF/tag debug output
logging.getLogger("PIL").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Target ~148KB per page by default
DEFAULT_TARGET_KB = 148
# Don't go below this DPI — text becomes unreadable
MIN_DPI = 100

_NOISY_OCR_LOGGERS = ("pikepdf", "fontTools", "fontTools.subset", "fontTools.ttLib", "ocrmypdf")


def _silence_ocr_loggers() -> None:
    """Raise noisy ocrmypdf/pikepdf/fontTools loggers to ERROR level."""
    for name in _NOISY_OCR_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)
    for name in list(logging.root.manager.loggerDict):
        if name.startswith("ocrmypdf"):
            logging.getLogger(name).setLevel(logging.ERROR)


@contextlib.contextmanager
def _icc_disabled():
    """Disable MuPDF ICC colour management for the duration of the block.

    Bitonal rendering rasterises straight to greyscale and immediately
    thresholds to one bit, so colour management cannot meaningfully change
    the result, and it costs roughly 30% of render time.

    MuPDF enables ICC by default and exposes no getter for the current
    state, so that default is what gets restored on the way out.
    """
    fitz.TOOLS.set_icc(False)
    try:
        yield
    finally:
        fitz.TOOLS.set_icc(True)


def _render_bitonal_page(
    src_page: fitz.Page,
    out_doc: fitz.Document,
    dpi: int,
    threshold: int,
) -> None:
    """Render ``src_page`` as a 1-bit image and append it to ``out_doc``.

    The TIFF handed to ``insert_image`` is deliberately uncompressed.
    MuPDF decodes whatever it is given and re-encodes the embedded image
    as ``/FlateDecode``, so compressing here (this used to emit CCITT G4)
    costs time and produces a byte-identical PDF.

    Callers that convert more than a page or two should wrap the loop in
    :func:`_icc_disabled`.

    :param src_page: Source page to rasterize.
    :param out_doc: Destination ``fitz.Document`` the rendered page is
        appended to (via ``new_page`` + ``insert_image``).
    :param dpi: Render DPI.
    :param threshold: Grayscale threshold (0-255) for binarisation.
    """
    import numpy as np

    new_page = out_doc.new_page(width=src_page.rect.width, height=src_page.rect.height)
    pix = src_page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
    gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    # packbits writes the 1-bit rows PIL's "1" mode expects (each row padded
    # to a byte boundary) in one pass, rather than materialising a full
    # uint8 image and letting PIL threshold it a second time.
    packed = np.packbits(gray > threshold, axis=1)
    bw = Image.frombytes("1", (pix.width, pix.height), packed.tobytes())
    buf = io.BytesIO()
    bw.save(buf, format="TIFF")
    new_page.insert_image(new_page.rect, stream=buf.getvalue())


def needs_ocr(pdf_path: Path, sample_pages: int = 3) -> bool:
    """Check if a PDF needs OCR (has no usable text layer).

    Samples a few pages and counts extracted words.  If very few words
    are found, the PDF is likely image-only.
    """
    doc = fitz.open(pdf_path)
    pages_to_check = min(sample_pages, len(doc))
    total_words = 0

    for i in range(pages_to_check):
        words = doc[i].get_text("words")
        total_words += len(words)

    doc.close()
    return total_words < 10


def _extract_page_image(
    doc: fitz.Document,
    page_num: int,
) -> tuple[Image.Image, float]:
    """Extract the image from a page, return PIL Image and effective DPI."""
    page = doc[page_num]
    images = page.get_images(full=True)
    if not images:
        raise ValueError(f"Page {page_num + 1} has no images")

    xref = images[0][0]
    img_data = doc.extract_image(xref)
    pil_img = Image.open(io.BytesIO(img_data["image"]))

    page_width_inches = page.rect.width / 72.0
    dpi = pil_img.width / page_width_inches
    return pil_img, dpi


COMPRESS_QUALITY = 65


def _compress_image(
    img: Image.Image,
    current_dpi: float,
    target_kb: int = DEFAULT_TARGET_KB,
    min_dpi: int = MIN_DPI,
) -> tuple[bytes, int, float]:
    """Compress an image to roughly target_kb.

    Estimates the right DPI from pixel-count ratio, resizes once,
    encodes once at a fixed quality. Fast single-pass approach.

    Returns (jpeg_bytes, quality_used, final_dpi).
    """
    if img.mode != "L":
        img = img.convert("L")

    # Estimate target DPI: encode once to calibrate, then scale
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=COMPRESS_QUALITY)
    current_kb = buf.tell() / 1024

    if current_kb <= target_kb:
        return buf.getvalue(), COMPRESS_QUALITY, current_dpi

    # size ~ pixels, pixels ~ dpi^2
    ratio = (target_kb / current_kb) ** 0.5
    target_dpi = max(min_dpi, current_dpi * ratio)

    scale = target_dpi / current_dpi
    new_size = (int(img.width * scale), int(img.height * scale))
    resized = img.resize(new_size, Image.LANCZOS)

    buf = io.BytesIO()
    resized.save(buf, format="JPEG", quality=COMPRESS_QUALITY)
    return buf.getvalue(), COMPRESS_QUALITY, target_dpi


def _downsample_pdf(
    input_path: Path,
    output_path: Path,
    target_kb: int = DEFAULT_TARGET_KB,
) -> None:
    """Downsample all images in a PDF to fit under target_kb per page.

    Replaces image streams in-place so the text layer, annotations,
    and other page content are preserved.
    """
    doc = fitz.open(str(input_path))
    total = doc.page_count
    seen: set[int] = set()

    # Collect work items: extract images that need compression
    work_items: list[tuple[int, Image.Image, float]] = []
    for i in range(total):
        page = doc[i]
        images = page.get_images(full=True)
        if not images:
            continue

        xref = images[0][0]
        if xref in seen:
            continue
        seen.add(xref)

        if "/DCTDecode" in doc.xref_object(xref):
            raw = doc.xref_stream_raw(xref)
            if len(raw) <= target_kb * 1024:
                continue

        try:
            img, dpi = _extract_page_image(doc, i)
        except ValueError:
            continue

        work_items.append((xref, img, dpi))

    if not work_items:
        doc.save(str(output_path), garbage=4, deflate=True)
        doc.close()
        return

    n_items = len(work_items)
    print(f"    Compressing {n_items} images...", flush=True)

    results = []
    for i, (xref, img, dpi) in enumerate(work_items):
        jpeg_bytes, quality, final_dpi = _compress_image(img, dpi, target_kb)
        out_img = Image.open(io.BytesIO(jpeg_bytes))
        results.append((xref, jpeg_bytes, out_img.width, out_img.height))
        done = i + 1
        if done % 10 == 0 or done == n_items:
            print(f"    Compressed {done}/{n_items} images", flush=True)

    # Apply results back to the document (must be sequential)
    for xref, jpeg_bytes, width, height in results:
        doc.update_stream(xref, jpeg_bytes, compress=False)
        doc.xref_set_key(xref, "Filter", "/DCTDecode")
        doc.xref_set_key(xref, "DecodeParms", "null")
        doc.xref_set_key(xref, "Width", str(width))
        doc.xref_set_key(xref, "Height", str(height))
        doc.xref_set_key(xref, "ColorSpace", "/DeviceGray")
        doc.xref_set_key(xref, "BitsPerComponent", "8")

    doc.save(str(output_path), garbage=4, deflate=True)
    doc.close()


def ocr_pdf(
    input_path: Path,
    output_path: Path | None = None,
    target_kb: int = DEFAULT_TARGET_KB,
    language: str = "eng",
    optimize: int = 1,
) -> Path:
    """Downsample and OCR an image-only PDF.

    1. Downsample page images to ~target_kb per page
    2. Run ocrmypdf to add an aligned invisible text layer

    The OCR runs on the final downsampled images so text coordinates
    match exactly.

    Args:
        input_path: Path to input PDF.
        output_path: Path for output PDF. Defaults to {stem}_ocr.pdf.
        target_kb: Target size per page in KB.
        language: Tesseract language code.

    Returns:
        Path to the OCR'd PDF.
    """

    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_ocr.pdf"
    output_path = Path(output_path)

    print(f"  OCR: processing {input_path.name} (target {target_kb} KB/page)")

    # Check if source is already bitonal — skip downsample if so
    _chk = fitz.open(str(input_path))
    _imgs = _chk[0].get_images(full=True) if _chk.page_count else []
    _is_bitonal = bool(_imgs and _imgs[0][4] == 1)
    _chk.close()

    if _is_bitonal:
        print("  OCR: bitonal source, skipping downsample")
        tmp_path = input_path
    else:
        # Step 1: Downsample
        print("  OCR step 1: Downsampling images...")
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        _downsample_pdf(input_path, tmp_path, target_kb)
        ds_mb = tmp_path.stat().st_size / (1024 * 1024)
        print(f"  OCR step 1 done: downsampled to {ds_mb:.1f} MB")

    # Step 2: OCR
    src_doc = fitz.open(str(tmp_path))
    n_pages = len(src_doc)
    src_doc.close()

    import multiprocessing
    import subprocess
    import sys

    n_workers = max(1, multiprocessing.cpu_count() // 2)

    if n_pages >= 20 and n_workers > 1:
        # ── Parallel OCR: split into chunks, OCR each, merge ──
        print(
            f"  OCR step 2: Adding text layer — {n_pages} pages ({n_workers} workers)...",
            flush=True,
        )

        chunk_size = (n_pages + n_workers - 1) // n_workers
        chunk_inputs = []
        chunk_outputs = []

        # Split PDF into chunks
        src_doc = fitz.open(str(tmp_path))
        for i in range(n_workers):
            start = i * chunk_size
            end = min(start + chunk_size, n_pages)
            if start >= end:
                break
            chunk_in = Path(tempfile.gettempdir()) / f"_ocr_chunk_{i}_in.pdf"
            chunk_out = Path(tempfile.gettempdir()) / f"_ocr_chunk_{i}_out.pdf"
            chunk_doc = fitz.open()
            chunk_doc.insert_pdf(src_doc, from_page=start, to_page=end - 1)
            chunk_doc.save(str(chunk_in), garbage=4, deflate=True)
            chunk_doc.close()
            chunk_inputs.append(chunk_in)
            chunk_outputs.append(chunk_out)
        src_doc.close()

        # Launch OCR subprocesses
        procs = []
        for i, (c_in, c_out) in enumerate(zip(chunk_inputs, chunk_outputs)):
            script = f"""
import os, logging
for n in ("pikepdf","fontTools","fontTools.subset","fontTools.ttLib","ocrmypdf"):
    logging.getLogger(n).setLevel(logging.ERROR)
for _n in list(logging.root.manager.loggerDict):
    if _n.startswith("ocrmypdf"):
        logging.getLogger(_n).setLevel(logging.ERROR)
import ocrmypdf
from ocrmypdf import hookimpl
from ocrmypdf._plugin_manager import get_plugin_manager

class _Progress:
    def __init__(self, *, total=None, desc=None, unit=None, disable=False, **kw):
        self._total = total
        self._unit = unit
        self._current = 0
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def update(self, n=1, *, completed=None):
        if self._unit != "page": return
        self._current += n
        if self._current % 10 == 0 or self._current == self._total:
            print(f"  OCR worker {i + 1}: {{self._current}}/{{self._total}}", flush=True)

class _Plugin:
    @hookimpl
    def get_progressbar_class(self):
        return _Progress

pm = get_plugin_manager()
pm._pm.register(_Plugin())
ocrmypdf.ocr(
    "{c_in}", "{c_out}",
    pdf_renderer="auto", optimize={optimize},
    output_type="pdf", language=["{language}"],
    tesseract_timeout=120, progress_bar=True,
    plugin_manager=pm,
)
print("DONE", flush=True)
"""
            p = subprocess.Popen(
                [sys.executable, "-c", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            procs.append((i, p))

        # Stream output and wait for all to finish
        import select

        completed = set()
        fds = {p.stdout.fileno(): (i, p) for i, p in procs}
        while len(completed) < len(procs):
            # Read available output from any worker
            readable, _, _ = select.select(list(fds.keys()), [], [], 0.5)
            for fd in readable:
                i, p = fds[fd]
                line = p.stdout.readline()
                if line:
                    text = line.decode(errors="replace").strip()
                    if text:
                        print(text, flush=True)
            # Check for finished workers
            for i, p in procs:
                if i in completed:
                    continue
                ret = p.poll()
                if ret is not None:
                    completed.add(i)
                    # Drain remaining output
                    for line in p.stdout:
                        text = line.decode(errors="replace").strip()
                        if text:
                            print(text, flush=True)
                    start = i * chunk_size
                    end = min(start + chunk_size, n_pages)
                    if ret != 0:
                        raise RuntimeError(f"OCR worker {i} failed (pages {start}-{end})")
                    print(
                        f"  OCR: worker {i + 1}/{len(procs)} done (pages {start + 1}-{end})",
                        flush=True,
                    )

        # Merge chunks back into one PDF
        print("  OCR: merging chunks...", flush=True)
        from blackletter.tasks import merge_pdfs

        merge_pdfs(chunk_outputs, output_path)

        # Cleanup
        for f in chunk_inputs + chunk_outputs:
            f.unlink(missing_ok=True)
    else:
        # ── Single-process OCR for small documents ──
        print(f"  OCR step 2: Adding text layer (tesseract) — {n_pages} pages...", flush=True)
        _silence_ocr_loggers()

        import ocrmypdf as _ocrmypdf
        from ocrmypdf import hookimpl as _hookimpl
        from ocrmypdf._plugin_manager import get_plugin_manager as _get_pm

        class _PrintProgress:
            def __init__(self, *, total=None, desc=None, unit=None, disable=False, **kw):
                self._total = total
                self._desc = desc
                self._unit = unit
                self._disable = disable
                self._current = 0

            def __enter__(self):
                if not self._disable and self._unit == "page":
                    print(f"  OCR: {self._desc or 'processing'} (0/{self._total})", flush=True)
                return self

            def __exit__(self, *args):
                return False

            def update(self, n=1, *, completed=None):
                if self._disable or self._unit != "page":
                    return
                self._current += n
                total = self._total or "?"
                if isinstance(total, int) and self._current % 4 == 0 or self._current == total:
                    print(
                        f"  OCR: {self._desc or 'processing'} ({self._current}/{total})", flush=True
                    )

        class _ProgressPlugin:
            @_hookimpl
            def get_progressbar_class(self):
                return _PrintProgress

        pm = _get_pm()
        pm._pm.register(_ProgressPlugin())
        _ocrmypdf.ocr(
            str(tmp_path),
            str(output_path),
            pdf_renderer="auto",
            optimize=optimize,
            output_type="pdf",
            language=[language],
            tesseract_timeout=120,
            progress_bar=True,
            plugin_manager=pm,
        )

    if tmp_path != input_path:
        tmp_path.unlink(missing_ok=True)

    final_mb = output_path.stat().st_size / (1024 * 1024)
    orig_mb = input_path.stat().st_size / (1024 * 1024)
    print(f"  OCR complete: {orig_mb:.1f} MB -> {final_mb:.1f} MB")
    return output_path


def run_bitonal(
    src_path: str | Path,
    dst_path: str | Path,
    dpi: int = 200,
    threshold: int = 160,
    workers: int = 1,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> int:
    """Convert every page of ``src_path`` to 1-bit and write ``dst_path``.

    The shared implementation behind :func:`blackletter.api.bitonal` and
    :func:`bitonal_convert`.

    With ``workers=1`` the conversion runs in the calling process and
    progress is reported per page. With ``workers>1`` the pages are split
    into one contiguous range per worker, converted in separate processes,
    and merged back in page order; progress is then reported as each range
    completes, so it advances in jumps rather than smoothly.

    Rasterisation is memory-bandwidth bound, so throughput flattens out
    well before one worker per core. Callers are expected to choose a
    worker count that suits the machine they are on: the default of 1
    keeps this a plain sequential call unless asked otherwise.

    :param src_path: PDF to convert.
    :param dst_path: Where the converted PDF is written.
    :param dpi: Render DPI.
    :param threshold: Grayscale threshold (0-255) for binarisation.
    :param workers: Number of worker processes. Values below 2 run inline.
        Above that the pool uses the "spawn" start method, which re-imports
        the caller's ``__main__`` in each child, so a script calling this at
        module scope needs the usual ``if __name__ == "__main__":`` guard.
    :param progress_callback: Optional callable(current, total, message).
    :returns: Number of pages converted.
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)

    # One open serves the page count, the guard, and the sequential render.
    # The parallel path cannot share it: workers are separate processes and
    # get the path instead.
    with fitz.open(str(src_path)) as src:
        total = src.page_count

        if total == 0:
            # Defensive: MuPDF currently refuses to *open* a page-less PDF,
            # so this is unreachable through a file path. It stays because
            # the alternative is "cannot save with zero pages" from deep
            # inside the save, which names neither the file nor the step
            # that failed.
            raise ValueError(f"cannot convert {src_path.name}: it has no pages")

        if workers < 2:
            with _icc_disabled(), fitz.open() as out:
                for i in range(total):
                    _render_bitonal_page(src[i], out, dpi, threshold)
                    if progress_callback and ((i + 1) % 10 == 0 or i == total - 1):
                        progress_callback(i + 1, total, f"Bitonal: {i + 1}/{total} pages")
                out.save(str(dst_path), garbage=4, deflate=True)
            return total

    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from blackletter.tasks import bitonal_chunk, merge_pdfs, split_page_ranges

    # "spawn" is pinned rather than left to the platform default, which is
    # "fork" on Linux but "spawn" on macOS and unavailable on Windows. Pinning
    # it is what makes this behave the same everywhere, and it is what the
    # subprocess-per-chunk implementation this replaced already did.
    #
    # "fork" would also be the riskier of the two. It has bitten this codebase
    # before (see the oneDNN/MKL note in analyze.py), and the main consumer
    # calls this from inside a Django worker, where forking hands every child
    # a copy of that process's database connections and client state. Workers
    # need nothing from the parent but their arguments.
    context = multiprocessing.get_context("spawn")

    ranges = split_page_ranges(total, workers)
    with tempfile.TemporaryDirectory(prefix="bl_bitonal_") as tmp_dir:
        chunk_paths = [str(Path(tmp_dir) / f"chunk_{i:04d}.pdf") for i in range(len(ranges))]
        done = 0
        with ProcessPoolExecutor(max_workers=len(ranges), mp_context=context) as pool:
            futures = {
                pool.submit(bitonal_chunk, str(src_path), chunk, start, end, dpi, threshold): (
                    start,
                    end,
                )
                for chunk, (start, end) in zip(chunk_paths, ranges, strict=True)
            }
            for future in as_completed(futures):
                future.result()  # re-raise whatever the worker hit
                start, end = futures[future]
                done += end - start
                if progress_callback:
                    progress_callback(done, total, f"Bitonal: {done}/{total} pages")
        # Merge in page order, which is not the order they finished in.
        merge_pdfs(chunk_paths, str(dst_path))
    return total


def bitonal_convert(
    src_path: Path,
    dst_path: Path,
    dpi: int = 200,
    threshold: int = 160,
    workers: int = 1,
) -> None:
    """Convert a PDF to bitonal (1-bit black/white), printing progress.

    Renders each page to grayscale at ``dpi`` and thresholds it. See
    :func:`run_bitonal` for the ``workers`` semantics.
    """
    orig_mb = src_path.stat().st_size / (1024 * 1024)
    suffix = f" ({workers} workers)" if workers > 1 else ""
    with fitz.open(str(src_path)) as src:
        total = src.page_count
    print(f"  Bitonal: converting {total} pages at {dpi} DPI{suffix}...", flush=True)

    def _report(current: int, page_total: int, _message: str) -> None:
        if workers > 1 or current % 20 == 0 or current == page_total:
            print(f"  Bitonal: {current}/{page_total}", flush=True)

    run_bitonal(
        src_path,
        dst_path,
        dpi=dpi,
        threshold=threshold,
        workers=workers,
        progress_callback=_report,
    )

    final_mb = dst_path.stat().st_size / (1024 * 1024)
    print(f"  Bitonal: {orig_mb:.1f} MB -> {final_mb:.1f} MB")
