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

import io
import logging
import tempfile
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
        if done % 50 == 0 or done == n_items:
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
    import ocrmypdf

    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_ocr.pdf"
    output_path = Path(output_path)

    print(f"  OCR: processing {input_path.name} (target {target_kb} KB/page)")

    # Step 1: Downsample
    print("  OCR step 1: Downsampling images...")
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    _downsample_pdf(input_path, tmp_path, target_kb)
    ds_mb = tmp_path.stat().st_size / (1024 * 1024)
    print(f"  OCR step 1 done: downsampled to {ds_mb:.1f} MB")

    # Step 2: OCR
    print("  OCR step 2: Adding text layer (tesseract)...")
    # Suppress verbose ocrmypdf/pikepdf/fontTools debug noise
    for name in (
        "ocrmypdf",
        "pikepdf",
        "ocrmypdf._exec",
        "fontTools",
        "fontTools.subset",
        "fontTools.ttLib",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)
    ocrmypdf.ocr(
        str(tmp_path),
        str(output_path),
        pdf_renderer="auto",
        optimize=optimize,
        output_type="pdf",
        language=[language],
        tesseract_timeout=120,
        progress_bar=True,
    )

    tmp_path.unlink(missing_ok=True)

    final_mb = output_path.stat().st_size / (1024 * 1024)
    orig_mb = input_path.stat().st_size / (1024 * 1024)
    print(f"  OCR complete: {orig_mb:.1f} MB -> {final_mb:.1f} MB")
    return output_path
