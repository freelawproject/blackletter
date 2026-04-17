"""Chunk-able task functions for celery integration.

Each function processes a specific page range or step of the pipeline.
No internal threading or subprocess parallelism — the caller (celery)
handles distribution across workers.

Usage from celery:
    # Fan out YOLO across 5 workers
    from celery import chord, group
    chunks = split_page_ranges(1293, 5)
    chord(
        group(yolo_scan_chunk.s(pdf, s, e, model) for s, e in chunks),
        merge_detections.s(output_dir)
    ).apply_async()
"""

from __future__ import annotations

import json
from pathlib import Path

import fitz


def split_page_ranges(total_pages: int, n_chunks: int) -> list[tuple[int, int]]:
    """Split a page count into N roughly equal ranges.

    Returns list of (start, end) tuples (0-indexed, end exclusive).
    """
    chunk_size = (total_pages + n_chunks - 1) // n_chunks
    ranges = []
    for i in range(n_chunks):
        s = i * chunk_size
        e = min(s + chunk_size, total_pages)
        if s < e:
            ranges.append((s, e))
    return ranges


def bitonal_chunk(
    src_path: str | Path,
    dst_path: str | Path,
    page_start: int,
    page_end: int,
    dpi: int = 200,
    threshold: int = 160,
) -> str:
    """Convert a page range to bitonal CCITT G4.

    Args:
        src_path: Input PDF path.
        dst_path: Output PDF path for this chunk.
        page_start: First page index (0-based).
        page_end: Last page index (exclusive).
        dpi: Render DPI.
        threshold: Binarization threshold.

    Returns:
        Path to the output chunk PDF.
    """
    from blackletter.ocr import _render_bitonal_page

    src = fitz.open(str(src_path))
    out = fitz.open()
    for i in range(page_start, page_end):
        _render_bitonal_page(src[i], out, dpi, threshold)
    out.save(str(dst_path), garbage=4, deflate=True)
    out.close()
    src.close()
    return str(dst_path)


def merge_pdfs(chunk_paths: list[str | Path], output_path: str | Path) -> str:
    """Merge multiple PDF chunks into one.

    Args:
        chunk_paths: List of PDF file paths to merge in order.
        output_path: Output merged PDF path.

    Returns:
        Path to merged PDF.
    """
    merged = fitz.open()
    for p in chunk_paths:
        chunk = fitz.open(str(p))
        merged.insert_pdf(chunk)
        chunk.close()
    merged.save(str(output_path), garbage=4, deflate=True)
    merged.close()
    return str(output_path)


def ocr_chunk(
    src_path: str | Path,
    dst_path: str | Path,
    page_start: int,
    page_end: int,
    language: str = "eng",
    optimize: int = 1,
) -> str:
    """OCR a page range.

    Extracts pages from src, runs ocrmypdf, saves result.

    Args:
        src_path: Input PDF path.
        dst_path: Output PDF path for this chunk.
        page_start: First page index (0-based).
        page_end: Last page index (exclusive).
        language: Tesseract language code.
        optimize: ocrmypdf optimization level.

    Returns:
        Path to OCR'd chunk PDF.
    """
    import tempfile

    from blackletter.ocr import _silence_ocr_loggers

    # Extract pages
    src = fitz.open(str(src_path))
    chunk = fitz.open()
    chunk.insert_pdf(src, from_page=page_start, to_page=page_end - 1)
    chunk_in = Path(tempfile.gettempdir()) / f"_ocr_in_{page_start}_{page_end}.pdf"
    chunk.save(str(chunk_in), garbage=4, deflate=True)
    chunk.close()
    src.close()

    _silence_ocr_loggers()

    import ocrmypdf

    ocrmypdf.ocr(
        str(chunk_in),
        str(dst_path),
        pdf_renderer="auto",
        optimize=optimize,
        output_type="pdf",
        language=[language],
        tesseract_timeout=120,
        progress_bar=False,
    )

    chunk_in.unlink(missing_ok=True)
    return str(dst_path)


def yolo_scan_chunk(
    pdf_path: str | Path,
    page_start: int,
    page_end: int,
    model_path: str | Path,
    confidence: float = 0.20,
    first_page: int = 1,
) -> list[dict]:
    """Run YOLO detection on a page range.

    Returns serializable list of page data dicts with detections.
    No subprocess spawning — runs in the calling process.

    Args:
        pdf_path: Path to PDF.
        page_start: First page index (0-based).
        page_end: Last page index (exclusive).
        model_path: Path to YOLO model weights.
        confidence: Detection confidence threshold.
        first_page: Logical page number of the first page.

    Returns:
        List of page data dicts, each with:
            index, pdf_width, pdf_height, img_width, img_height,
            cols, detections, page_number, page_number_end
    """
    from PIL import Image
    from ultralytics import YOLO

    from blackletter.models import Label
    from blackletter.scanner import _safe_detect_columns, _extract_page_number, DPI, YOLO_BATCH

    model = YOLO(str(model_path))
    pdf = fitz.open(str(pdf_path))
    mat = fitz.Matrix(DPI / 72, DPI / 72)
    pages_data = []

    for bs in range(page_start, page_end, YOLO_BATCH):
        be = min(bs + YOLO_BATCH, page_end)
        imgs = []
        meta = []
        for pi in range(bs, be):
            fp = pdf[pi]
            pix = fp.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            imgs.append(img)
            cols = _safe_detect_columns(img)
            meta.append(
                (
                    pi,
                    fp.rect.width,
                    fp.rect.height,
                    pix.width,
                    pix.height,
                    cols,
                )
            )

        results = model(imgs, conf=confidence, verbose=False, device=None)

        for j, (pi, pdf_w, pdf_h, pw, ph, cols) in enumerate(meta):
            dets = []
            for box in results[j].boxes:
                dets.append(
                    {
                        "bbox": box.xyxy[0].tolist(),
                        "label": int(box.cls[0].item()),
                        "conf": float(box.conf[0].item()),
                    }
                )

            # Extract page number
            fp = pdf[pi]
            sx, sy = pdf_w / pw, pdf_h / ph
            pn, pn_end = None, None
            pn_dets = [
                d
                for d in dets
                if d["label"] == int(Label.PAGE_NUMBER)
                and (d["bbox"][3] - d["bbox"][1]) >= 40
                and (d["bbox"][2] - d["bbox"][0]) >= 40
                and d["bbox"][1] >= 5
                and d["bbox"][0] >= 5
            ]
            pn_dets.sort(key=lambda d: d["conf"], reverse=True)
            for pd in pn_dets:
                bx1, by1, bx2, by2 = pd["bbox"]
                rect = fitz.Rect(bx1 * sx, by1 * sy, bx2 * sx, by2 * sy)
                r = _extract_page_number(fp, rect, hint=pi + first_page)
                if r is not None:
                    pn, pn_end = r
                    break

            pages_data.append(
                {
                    "index": pi,
                    "pdf_width": pdf_w,
                    "pdf_height": pdf_h,
                    "img_width": pw,
                    "img_height": ph,
                    "cols": cols,
                    "detections": dets,
                    "page_number": pn,
                    "page_number_end": pn_end,
                }
            )

    pdf.close()
    return pages_data


def merge_detections(
    chunks: list[list[dict]],
    pdf_path: str | Path,
    output_dir: str | Path,
    first_page: int = 1,
    reporter: str = "",
    volume: str = "",
) -> dict:
    """Merge YOLO detection chunks into a Document and save outputs.

    Args:
        chunks: List of page data lists from yolo_scan_chunk.
        pdf_path: Path to the source PDF.
        output_dir: Where to save detections.json, pages_meta.json.
        first_page: Logical first page number.
        reporter: Reporter abbreviation.
        volume: Volume number.

    Returns:
        Dict with keys: detections_count, pages_count.
    """
    from blackletter.models import BBox, Detection, Document, Label, Page
    from blackletter.scanner import _correct_page_numbers

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Flatten chunks and build Document
    all_pages = []
    for chunk in chunks:
        all_pages.extend(chunk)
    all_pages.sort(key=lambda p: p["index"])

    pages = []
    for pd in all_pages:
        lx1, lx2, rx1, rx2, mid = pd["cols"]
        page = Page(
            index=pd["index"],
            pdf_width=pd["pdf_width"],
            pdf_height=pd["pdf_height"],
            img_width=pd["img_width"],
            img_height=pd["img_height"],
            col_left_x1=lx1,
            col_left_x2=lx2,
            col_right_x1=rx1,
            col_right_x2=rx2,
            midpoint=mid,
            page_number=pd["page_number"],
            page_number_end=pd["page_number_end"],
        )
        for d in pd["detections"]:
            page.detections.append(
                Detection(
                    bbox=BBox.from_xyxy(d["bbox"]),
                    label=Label(d["label"]),
                    confidence=d["conf"],
                    page_index=pd["index"],
                )
            )
        pages.append(page)

    # Correct page numbers
    _correct_page_numbers(pages)

    doc = Document(
        pdf_path=Path(pdf_path),
        pages=pages,
        first_page=first_page,
        reporter=reporter,
        volume=volume,
        ocr_applied=True,
    )

    from blackletter.scanner import _write_detections_sidecar, _write_pages_meta_sidecar

    detections_count = _write_detections_sidecar(doc, output_dir)
    _write_pages_meta_sidecar(doc, output_dir)

    return {
        "detections_count": detections_count,
        "pages_count": len(pages),
    }


def pair_and_compute_rects(
    pdf_path: str | Path,
    output_dir: str | Path,
    first_page: int = 1,
    reporter: str = "",
    volume: str = "",
    excluded: set | None = None,
    approved: set | None = None,
) -> dict:
    """Pair opinions and compute redaction/margin rects.

    Reads detections.json + pages_meta.json from output_dir,
    runs pairing, computes rects with docTR, saves results.

    Returns:
        Dict with opinions_count, rects_count.
    """
    from blackletter.models import Detection, Document, Page
    from blackletter.scanner import (
        _pair_opinions,
        _build_opinions_data,
        _group_detections_by_page,
    )
    from blackletter.process import compute_redaction_rects
    from blackletter.margins import compute_margin_rects

    output_dir = Path(output_dir)
    pdf_path = Path(pdf_path)

    raw = json.loads((output_dir / "detections.json").read_text())
    pages_meta = (
        json.loads((output_dir / "pages_meta.json").read_text())
        if (output_dir / "pages_meta.json").exists()
        else {}
    )

    # Reconstruct Document
    src_pdf = fitz.open(str(pdf_path))
    page_dims = {
        i: (src_pdf.load_page(i).rect.width, src_pdf.load_page(i).rect.height)
        for i in range(len(src_pdf))
    }
    src_pdf.close()

    pages_data = _group_detections_by_page(raw)

    pages = []
    for pi in sorted(pages_data.keys()):
        pd = pages_data[pi]
        w, h = page_dims.get(pi, (612, 792))
        meta = pages_meta.get(str(pi), pages_meta.get(pi, {}))
        page = Page(
            index=pi,
            pdf_width=w,
            pdf_height=h,
            img_width=pd["img_width"],
            img_height=pd["img_height"],
            page_number=pd["page_number"],
            col_left_x1=meta.get("col_left_x1", 0),
            col_left_x2=meta.get("col_left_x2", 0),
            col_right_x1=meta.get("col_right_x1", 0),
            col_right_x2=meta.get("col_right_x2", 0),
            midpoint=meta.get("midpoint", 0),
        )
        for d in pd["detections"]:
            page.detections.append(Detection.from_raw_dict(d, pi, bbox_default=[0, 0, 1, 1]))
        pages.append(page)

    document = Document(
        pdf_path=pdf_path,
        pages=pages,
        reporter=reporter,
        volume=volume,
        first_page=first_page,
        ocr_applied=True,
    )

    # Pair
    opinions = _pair_opinions(document, excluded=excluded)

    # Save opinions.json with outside_rects
    pages_by_index = {p.index: p for p in document.pages}
    _src_pdf = fitz.open(str(pdf_path))
    opinions_data = _build_opinions_data(opinions, pages_by_index, _src_pdf)
    _src_pdf.close()
    with open(output_dir / "opinions.json", "w") as f:
        json.dump(opinions_data, f)

    # Compute redaction rects
    rects = compute_redaction_rects(document, opinions, excluded=excluded, approved=approved)
    with open(output_dir / "redaction_rects.json", "w") as f:
        json.dump(rects, f)

    # Compute margin rects
    margin_rects = compute_margin_rects(pdf_path)
    with open(output_dir / "margin_rects.json", "w") as f:
        json.dump(margin_rects, f)

    return {
        "opinions_count": len(opinions),
        "rects_count": sum(len(r["rects"]) for r in rects),
    }
