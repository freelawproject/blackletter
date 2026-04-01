"""Clean step-by-step API for the blackletter pipeline.

Each function is one discrete step. Call only what you need.
All functions work with file paths and return file paths or data.

Usage:
    from blackletter.api import bitonal, ocr, detect, pair, compute_rects, build_redacted, split_opinions

    bitonal_pdf = bitonal(source_pdf, output_dir)
    ocr_pdf = ocr(bitonal_pdf, output_dir)
    detections = detect(bitonal_pdf, output_dir, models=["medium", "large"])
    opinions = pair(detections, ocr_pdf, reporter="a3d", volume="333", first_page=1)
    rects = compute_rects(opinions, detections, ocr_pdf)
    redacted_pdf = build_redacted(ocr_pdf, rects, output_dir)
    opinion_files = split_opinions(ocr_pdf, opinions, rects, output_dir)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import fitz


def bitonal(
    pdf_path: str | Path,
    output_dir: str | Path,
    dpi: int = 200,
    threshold: int = 160,
    progress_callback=None,
) -> Path:
    """Convert a PDF to bitonal (CCITT G4 TIFF images).

    Args:
        progress_callback: Optional callable(current, total, message)

    Returns path to the bitonal PDF.
    """
    import io
    import numpy as np
    from PIL import Image

    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "bitonal.pdf"

    src = fitz.open(str(pdf_path))
    out = fitz.open()
    total = src.page_count

    for i in range(total):
        page = src[i]
        new_page = out.new_page(width=page.rect.width, height=page.rect.height)
        pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
        gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
        bw = Image.fromarray((gray > threshold).astype(np.uint8) * 255).convert("1")
        buf = io.BytesIO()
        bw.save(buf, format="TIFF", compression="group4")
        new_page.insert_image(new_page.rect, stream=buf.getvalue())
        if progress_callback and ((i + 1) % 10 == 0 or i == total - 1):
            progress_callback(i + 1, total, f"Bitonal: {i + 1}/{total} pages")
        if (i + 1) % 50 == 0 or i == total - 1:
            print(f"  Bitonal {i + 1}/{total}", flush=True)

    out.save(str(output_path), garbage=4, deflate=True)
    out.close()
    src.close()
    print(f"  Saved bitonal.pdf ({output_path.stat().st_size / 1024 / 1024:.1f} MB)", flush=True)
    return output_path


def ocr(
    pdf_path: str | Path,
    output_dir: str | Path,
    reporter: str = "",
    volume: str = "",
    first_page: int = 1,
    language: str = "eng",
) -> Path:
    """OCR a PDF (add text layer via ocrmypdf/Tesseract).

    Returns path to the OCR'd PDF.
    """
    import logging

    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    # Build scan name
    total_pages = fitz.open(str(pdf_path)).page_count
    last_page = first_page + total_pages - 1
    parts = [p for p in [reporter, str(volume), str(first_page), str(last_page)] if p]
    scan_name = ".".join(parts) if parts else pdf_path.stem
    output_path = output_dir / f"{scan_name}.pdf"

    for name in ("pikepdf", "fontTools", "fontTools.subset", "fontTools.ttLib", "ocrmypdf"):
        logging.getLogger(name).setLevel(logging.ERROR)
    for _n in list(logging.root.manager.loggerDict):
        if _n.startswith("ocrmypdf"):
            logging.getLogger(_n).setLevel(logging.ERROR)

    import ocrmypdf

    print(f"  OCR {total_pages} pages...", flush=True)
    t0 = time.time()
    ocrmypdf.ocr(
        str(pdf_path),
        str(output_path),
        pdf_renderer="auto",
        optimize=1,
        output_type="pdf",
        language=[language],
        tesseract_timeout=120,
        progress_bar=False,
    )
    print(f"  OCR done ({time.time() - t0:.0f}s)", flush=True)
    return output_path


def detect(
    pdf_path: str | Path,
    output_dir: str | Path,
    models: list[str] | None = None,
    confidence: float = 0.20,
) -> list[dict]:
    """Run YOLO detection on all pages with one or more models.

    Returns merged detection list. Also saves detections.json.
    """
    from PIL import Image
    from ultralytics import YOLO
    from blackletter.models import Label

    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    models_dir = Path(__file__).parent / "weights"

    if models is None:
        models = ["small", "medium", "large"]

    model_map = {"small": "small.pt", "medium": "medium.pt", "large": "large.pt"}

    pdf = fitz.open(str(pdf_path))
    DPI = 200
    mat = fitz.Matrix(DPI / 72, DPI / 72)
    total = pdf.page_count

    all_raw = []
    for model_name in models:
        model_file = models_dir / model_map.get(model_name, f"{model_name}.pt")
        if not model_file.exists():
            print(f"  Model {model_file} not found, skipping", flush=True)
            continue

        model = YOLO(str(model_file))
        print(f"  Detecting with {model_name}...", flush=True)
        t0 = time.time()

        BATCH = 4
        for bs in range(0, total, BATCH):
            be = min(bs + BATCH, total)
            imgs = []
            metas = []
            for i in range(bs, be):
                pix = pdf[i].get_pixmap(matrix=mat)
                imgs.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
                metas.append({"index": i, "img_width": pix.width, "img_height": pix.height})

            results = model(imgs, conf=confidence, verbose=False)
            for j, res in enumerate(results):
                pm = metas[j]
                for box in res.boxes:
                    class_id = int(box.cls[0].item())
                    try:
                        label_name = Label(class_id).name
                    except ValueError:
                        continue
                    all_raw.append(
                        {
                            "page_index": pm["index"],
                            "label": label_name,
                            "label_id": class_id,
                            "confidence": round(float(box.conf[0].item()), 3),
                            "bbox": [round(v, 1) for v in box.xyxy[0].tolist()],
                            "img_width": pm["img_width"],
                            "img_height": pm["img_height"],
                            "model": model_name,
                        }
                    )
            if (bs + BATCH) % 100 == 0 or bs + BATCH >= total:
                print(f"    {min(bs + BATCH, total)}/{total} pages", flush=True)

        print(f"    {model_name} done ({time.time() - t0:.0f}s)", flush=True)

    pdf.close()

    # Merge across models with label-specific strategies:
    #   CASE_CAPTION, KEY_ICON: medium model only (large hallucinates)
    #   CASE_SEQUENCE: all models, overlaps keep smallest box
    #   Everything else: all models, overlaps keep highest confidence
    MEDIUM_ONLY = {"CASE_CAPTION", "KEY_ICON"}
    SMALLEST_BOX = {"CASE_SEQUENCE"}

    # Filter: for medium-only labels, keep only medium model detections
    filtered = [d for d in all_raw if d["label"] not in MEDIUM_ONLY or d["model"] == "medium"]

    def _bbox_area(bbox):
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

    def _iou(a, b):
        """Intersection over union of two [x0, y0, x1, y1] boxes."""
        ix0 = max(a[0], b[0])
        iy0 = max(a[1], b[1])
        ix1 = min(a[2], b[2])
        iy1 = min(a[3], b[3])
        if ix1 <= ix0 or iy1 <= iy0:
            return 0.0
        inter = (ix1 - ix0) * (iy1 - iy0)
        union = _bbox_area(a) + _bbox_area(b) - inter
        return inter / union if union > 0 else 0.0

    def _contains(outer, inner):
        """True if outer bbox fully contains inner bbox."""
        return (
            outer[0] <= inner[0]
            and outer[1] <= inner[1]
            and outer[2] >= inner[2]
            and outer[3] >= inner[3]
        )

    filtered.sort(key=lambda d: (d["page_index"], d["label_id"], d["bbox"][1]))
    merged = []
    used = set()
    for i, d in enumerate(filtered):
        if i in used:
            continue
        used.add(i)
        found_by = [{"model": d["model"], "confidence": d["confidence"]}]
        best = d
        for j in range(i + 1, len(filtered)):
            if j in used:
                continue
            od = filtered[j]
            if od["page_index"] != d["page_index"]:
                break
            if od["label_id"] != d["label_id"]:
                continue
            # Match if boxes overlap significantly or one contains the other
            overlap = _iou(d["bbox"], od["bbox"]) > 0.3
            contained = _contains(d["bbox"], od["bbox"]) or _contains(od["bbox"], d["bbox"])
            if overlap or contained:
                used.add(j)
                found_by.append({"model": od["model"], "confidence": od["confidence"]})
                if d["label"] in SMALLEST_BOX:
                    if _bbox_area(od["bbox"]) < _bbox_area(best["bbox"]):
                        best = od
                else:
                    if od["confidence"] > best["confidence"]:
                        best = od
        det = dict(best)
        det["found_by"] = found_by
        det["model_count"] = len(found_by)
        det.pop("model", None)
        merged.append(det)

    # Save
    det_path = output_dir / "detections.json"
    det_path.write_text(json.dumps(merged))
    print(f"  {len(merged)} detections ({len(all_raw)} raw from {len(models)} models)", flush=True)
    return merged


def pair(
    detections: list[dict] | str | Path,
    pdf_path: str | Path,
    reporter: str = "",
    volume: str = "",
    first_page: int = 1,
    excluded: set | None = None,
) -> list[dict]:
    """Pair opinions from detections. Returns opinions list. Saves opinions.json.

    Can accept detections as a list or path to detections.json.
    """
    from blackletter.models import BBox, Detection as BLDetection, Document, Label, Page
    from blackletter.scanner import _pair_opinions, _outside_opinion_rects

    pdf_path = Path(pdf_path)

    # Load detections
    if isinstance(detections, (str, Path)):
        raw = json.loads(Path(detections).read_text())
    else:
        raw = detections

    # Build Document
    src_pdf = fitz.open(str(pdf_path))
    pages_data = {}
    for entry in raw:
        pi = entry["page_index"]
        if pi not in pages_data:
            pages_data[pi] = {
                "page_number": entry.get("page_number"),
                "page_number_end": entry.get("page_number_end"),
                "img_width": entry.get("img_width", 1),
                "img_height": entry.get("img_height", 1),
                "detections": [],
            }
        pages_data[pi]["detections"].append(entry)

    pages = []
    for pi in sorted(pages_data.keys()):
        pd = pages_data[pi]
        if pi < src_pdf.page_count:
            pw, ph = src_pdf[pi].rect.width, src_pdf[pi].rect.height
        else:
            pw, ph = 612.0, 792.0
        page = Page(
            index=pi,
            pdf_width=pw,
            pdf_height=ph,
            img_width=pd["img_width"],
            img_height=pd["img_height"],
            page_number=pd["page_number"],
            page_number_end=pd.get("page_number_end"),
        )
        for d in pd["detections"]:
            b = d.get("bbox", [0, 0, 1, 1])
            page.detections.append(
                BLDetection(
                    bbox=BBox(x1=b[0], y1=b[1], x2=b[2], y2=b[3]),
                    label=Label(d["label_id"]),
                    confidence=d["confidence"],
                    page_index=pi,
                )
            )
        if page.page_number is None:
            page.page_number = pi + first_page
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
    t0 = time.time()
    opinions = _pair_opinions(document, excluded=excluded)
    print(f"  Paired {len(opinions)} opinions ({time.time() - t0:.0f}s)", flush=True)

    # Each opinion runs from its caption page to its key page.
    page_ranges: list[tuple[int, int]] = []
    for idx, (caption, key) in enumerate(opinions):
        page_ranges.append((caption.page_index, key.page_index))

    # Save opinions.json
    pages_by_index = {p.index: p for p in document.pages}
    opinions_data = []
    for idx, (caption, key) in enumerate(opinions):
        start_idx, end_idx = page_ranges[idx]
        outside_rects = []
        for pi in range(start_idx, end_idx + 1):
            pg = pages_by_index.get(pi)
            if not pg:
                continue
            is_first = pi == start_idx
            is_last = pi == end_idx
            pw = src_pdf[pi].rect.width if pi < src_pdf.page_count else 612.0
            for rect in _outside_opinion_rects(pg, pw, caption, key, is_first, is_last):
                outside_rects.append(
                    {
                        "page_index": pi,
                        "x0": round(rect.x0, 1),
                        "y0": round(rect.y0, 1),
                        "x1": round(rect.x1, 1),
                        "y1": round(rect.y1, 1),
                    }
                )
        has_image = any(
            d.label == Label.IMAGE
            for pi2 in range(start_idx, end_idx + 1)
            if pi2 in pages_by_index
            for d in pages_by_index[pi2].detections
        )

        # For naming: opinion starting on a range page uses end number,
        # ending on a range page uses first number
        start_page = pages_by_index.get(start_idx)
        end_page = pages_by_index.get(end_idx)
        if start_page and start_page.page_number_end:
            first_num = start_page.page_number_end
        elif start_page and start_page.page_number:
            first_num = start_page.page_number
        else:
            first_num = start_idx + first_page
        if end_page and end_page.page_number_end:
            last_num = end_page.page_number
        elif end_page and end_page.page_number:
            last_num = end_page.page_number
        else:
            last_num = end_idx + first_page

        opinions_data.append(
            {
                "caption_page": caption.page_index,
                "caption_bbox": [
                    round(caption.bbox.x1, 1),
                    round(caption.bbox.y1, 1),
                    round(caption.bbox.x2, 1),
                    round(caption.bbox.y2, 1),
                ],
                "key_page": key.page_index,
                "key_bbox": [
                    round(key.bbox.x1, 1),
                    round(key.bbox.y1, 1),
                    round(key.bbox.x2, 1),
                    round(key.bbox.y2, 1),
                ],
                "end_page": end_idx,
                "page_count": end_idx - start_idx + 1,
                "first_page_number": first_num,
                "last_page_number": last_num,
                "outside_rects": outside_rects,
                "has_image": has_image,
            }
        )

    src_pdf.close()

    # # Save
    # output_dir = pdf_path.parent
    # (output_dir / "opinions.json").write_text(json.dumps(opinions_data))

    return opinions_data


def compute_rects(
    pdf_path: str | Path,
    output_dir: str | Path,
    excluded: set | None = None,
    approved: set | None = None,
    skip_doctr: bool = False,
) -> list[dict]:
    """Compute redaction rects from detections + opinions. Saves redaction_rects.json.

    Reads detections.json and opinions.json from output_dir.
    """
    from blackletter.tasks import pair_and_compute_rects as _pair_compute

    pdf_path = str(pdf_path)
    output_dir = Path(output_dir)

    # Read reporter/volume/first_page from opinions or infer
    det_path = output_dir / "detections.json"
    if not det_path.exists():
        raise FileNotFoundError(f"No detections.json in {output_dir}")

    t0 = time.time()
    result = _pair_compute(
        pdf_path,
        str(output_dir),
        excluded=excluded,
        approved=approved,
    )
    print(f"  Computed {result['rects_count']} rects ({time.time() - t0:.0f}s)", flush=True)

    rects = json.loads((output_dir / "redaction_rects.json").read_text())
    return rects


def build_redacted(
    pdf_path: str | Path,
    output_dir: str | Path,
    rects: list[dict] | str | Path | None = None,
) -> Path:
    """Build the full redacted PDF from precomputed rects.

    Returns path to the redacted PDF.
    """
    from blackletter.process import _build_redacted_from_rects
    from blackletter.models import Document, Page

    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    # Load rects
    rects_path = output_dir / "redaction_rects.json"
    if rects is not None:
        if isinstance(rects, (str, Path)):
            rects_path = Path(rects)

    # Build minimal Document for the function
    src_pdf = fitz.open(str(pdf_path))
    det_data = (
        json.loads((output_dir / "detections.json").read_text())
        if (output_dir / "detections.json").exists()
        else []
    )

    pages_data = {}
    for entry in det_data:
        pi = entry["page_index"]
        if pi not in pages_data:
            pages_data[pi] = {
                "img_width": entry.get("img_width", 1),
                "img_height": entry.get("img_height", 1),
            }

    pages = []
    for i in range(src_pdf.page_count):
        pd = pages_data.get(i, {"img_width": 1700, "img_height": 2200})
        pages.append(
            Page(
                index=i,
                pdf_width=src_pdf[i].rect.width,
                pdf_height=src_pdf[i].rect.height,
                img_width=pd["img_width"],
                img_height=pd["img_height"],
            )
        )
    src_pdf.close()

    document = Document(pdf_path=pdf_path, pages=pages, ocr_applied=True)

    # Build scan name
    stem = pdf_path.stem
    output_path = output_dir / f"{stem}.redacted.pdf"

    t0 = time.time()
    print("  Building redacted PDF...", flush=True)
    _build_redacted_from_rects(document, rects_path, output_path)
    print(f"  Redacted done ({time.time() - t0:.0f}s)", flush=True)
    return output_path


def split_opinions(
    pdf_path: str | Path,
    output_dir: str | Path,
    unredacted: bool = True,
    opinions: list[dict] | None = None,
) -> dict:
    """Split the redacted PDF into individual opinion PDFs.

    Creates redacted/, unredacted/, and masked/ subdirectories.
    Returns dict with counts.

    opinions: precomputed opinions list (from api.pair()). If omitted,
              falls back to reading opinions.json from output_dir.
    """
    from blackletter.process import _split_from_redacted, _build_masked_opinions

    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    if opinions is not None:
        opinions_data = opinions
    else:
        opinions_data = json.loads((output_dir / "opinions.json").read_text())

    # Find the redacted PDF
    redacted_pdfs = list(output_dir.glob("*.redacted.pdf"))
    if not redacted_pdfs:
        raise FileNotFoundError("No redacted PDF found")
    redacted_pdf = redacted_pdfs[0]

    t0 = time.time()

    # Split redacted
    redacted_dir = output_dir / "redacted"
    redacted_dir.mkdir(exist_ok=True)
    _split_from_redacted(str(redacted_pdf), opinions_data, str(redacted_dir))
    redacted_files = sorted(redacted_dir.glob("*.pdf"))
    print(f"  Split {len(redacted_files)} redacted ({time.time() - t0:.0f}s)", flush=True)

    # Split unredacted
    if unredacted:
        t0 = time.time()
        unredacted_dir = output_dir / "unredacted"
        unredacted_dir.mkdir(exist_ok=True)
        _split_from_redacted(str(pdf_path), opinions_data, str(unredacted_dir))
        print(f"  Split unredacted ({time.time() - t0:.0f}s)", flush=True)

    # Build masked
    t0 = time.time()
    masked_dir = output_dir / "masked"
    masked_dir.mkdir(exist_ok=True)
    _build_masked_opinions(str(redacted_pdf), opinions_data, str(masked_dir))
    print(f"  Split masked ({time.time() - t0:.0f}s)", flush=True)

    return {
        "redacted": len(redacted_files),
        "unredacted": len(list((output_dir / "unredacted").glob("*.pdf"))) if unredacted else 0,
        "masked": len(list(masked_dir.glob("*.pdf"))),
    }


def generate(
    pdf_path: str | Path,
    redactions: str | Path | dict,
    output_dir: str | Path,
    unredacted: bool = False,
    progress_callback=None,
) -> dict:
    """Generate all output PDFs from a source PDF and a single redactions.json.

    redactions.json contains:
      - "opinions": list of opinion dicts with outside_rects, page ranges, filenames
      - "pages": dict of page_index → list of rects (margins + redaction rects, all PDF points)

    Builds in one pass per page — no layering.

    Args:
        pdf_path: Path to the source (OCR'd) PDF.
        redactions: Path to redactions.json, or the parsed dict.
        output_dir: Base output directory.
        unredacted: Also generate unredacted opinion PDFs.
        progress_callback: Optional callable(current, total, message).

    Returns dict with keys: redacted_dir, masked_dir, full_redacted, opinion_count.
    """
    # Labels to white-out in masked output instead of black
    WHITE_IN_MASKED = {"PAGE_HEADER", "STATE_ABBREVIATION"}

    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(redactions, (str, Path)):
        data = json.loads(Path(redactions).read_text())
    else:
        data = redactions

    opinions = data["opinions"]
    pages_rects = data["pages"]

    src = fitz.open(str(pdf_path))

    def _apply_page(fitz_page, src_idx, opinion, mode):
        """Apply all rects for one page in one pass.

        mode: 'full'     — all rects as specified, no outside-opinion whiteout
              'redacted' — all rects as specified + outside-opinion whiteout
              'masked'   — WHITE_IN_MASKED → white + outside-opinion whiteout
        """
        # Page rects (margins + redactions) — all PDF points
        for r in pages_rects.get(str(src_idx), []):
            rect = fitz.Rect(r["x0"], r["y0"], r["x1"], r["y1"])
            if rect.is_empty or rect.y0 >= rect.y1 or rect.x0 >= rect.x1:
                continue
            if r["type"] == "margin":
                fill = (1, 1, 1)

            elif mode == "masked" and r["type"] in WHITE_IN_MASKED:
                fill = (1, 1, 1)
            elif mode != "masked" and r["type"] in WHITE_IN_MASKED:
                fill = (0, 0, 0)

            else:
                fill = (0, 0, 0) if r["fill"] == "black" else (1, 1, 1)
            fitz_page.add_redact_annot(rect, fill=fill)

        # Outside-opinion whiteout (skip for full redacted)
        if opinion is not None and mode != "full":
            for orect in opinion.get("outside_rects", []):
                if orect["page_index"] != src_idx:
                    continue
                rect = fitz.Rect(orect["x0"], orect["y0"] + 3, orect["x1"], orect["y1"])
                if not rect.is_empty:
                    fitz_page.add_redact_annot(rect, fill=(1, 1, 1))

        fitz_page.apply_redactions()

    # ── Full redacted PDF ──
    t0 = time.time()
    full_out = fitz.open()
    full_out.insert_pdf(src)
    for page_idx in range(full_out.page_count):
        _apply_page(full_out[page_idx], page_idx, None, "full")
        if progress_callback and ((page_idx + 1) % 20 == 0 or page_idx == full_out.page_count - 1):
            progress_callback(page_idx + 1, full_out.page_count, "Redacting pages...")

    full_name = f"{pdf_path.stem}.redacted.pdf"
    full_path = output_dir / full_name
    full_out.save(str(full_path), garbage=4, deflate=True)
    full_out.close()
    print(
        f"  Full redacted: {full_path.name} ({full_path.stat().st_size / 1024 / 1024:.1f} MB, {time.time() - t0:.0f}s)",
        flush=True,
    )

    # ── Split opinions ──
    redacted_dir = output_dir / "redacted"
    masked_dir = output_dir / "masked"
    redacted_dir.mkdir(exist_ok=True)
    masked_dir.mkdir(exist_ok=True)

    if unredacted:
        unredacted_dir = output_dir / "unredacted"
        unredacted_dir.mkdir(exist_ok=True)

    t0 = time.time()

    # ── Redacted + unredacted: one PDF per opinion ──
    for i, op in enumerate(opinions):
        start_idx = op["caption_page"]
        end_idx = op["end_page"]
        filename = op.get("filename", f"{start_idx:04d}-{end_idx:04d}.pdf")

        out = fitz.open()
        out.insert_pdf(src, from_page=start_idx, to_page=end_idx)
        for local_idx, src_idx in enumerate(range(start_idx, end_idx + 1)):
            _apply_page(out[local_idx], src_idx, op, "redacted")
        out.save(str(redacted_dir / filename), garbage=4, deflate=True)
        out.close()

        if unredacted:
            out = fitz.open()
            out.insert_pdf(src, from_page=start_idx, to_page=end_idx)
            for local_idx, src_idx in enumerate(range(start_idx, end_idx + 1)):
                for orect in op.get("outside_rects", []):
                    if orect["page_index"] != src_idx:
                        continue
                    rect = fitz.Rect(orect["x0"], orect["y0"], orect["x1"], orect["y1"])
                    if not rect.is_empty:
                        out[local_idx].add_redact_annot(rect, fill=(1, 1, 1))
                out[local_idx].apply_redactions()
            out.save(str(unredacted_dir / filename), garbage=4, deflate=True)
            out.close()

        done = i + 1
        if done % 20 == 0 or done == len(opinions):
            print(f"    Redacted {done}/{len(opinions)}", flush=True)

    # ── Masked: group same-page opinions together ──
    # Build groups of consecutive single-page opinions on the same source page
    groups: list[list[int]] = []
    i = 0
    while i < len(opinions):
        op = opinions[i]
        if op["caption_page"] == op["end_page"]:
            group = [i]
            j = i + 1
            while j < len(opinions):
                next_op = opinions[j]
                if (
                    next_op["caption_page"] == next_op["end_page"]
                    and next_op["caption_page"] == op["caption_page"]
                ):
                    group.append(j)
                    j += 1
                else:
                    break
            groups.append(group)
            i = j
        else:
            groups.append([i])
            i += 1

    for group in groups:
        if len(group) == 1:
            # Single opinion (single-page or multi-page) — build normally
            op = opinions[group[0]]
            start_idx = op["caption_page"]
            end_idx = op["end_page"]
            filename = op.get("filename", f"{start_idx:04d}-{end_idx:04d}.pdf")

            out = fitz.open()
            out.insert_pdf(src, from_page=start_idx, to_page=end_idx)
            for local_idx, src_idx in enumerate(range(start_idx, end_idx + 1)):
                _apply_page(out[local_idx], src_idx, op, "masked")
            out.save(str(masked_dir / filename), garbage=4, deflate=True)
            out.close()
        else:
            # Multiple same-page opinions — consolidate into one PDF
            # Only white out areas that ALL opinions consider "outside"
            first_op = opinions[group[0]]
            src_page_idx = first_op["caption_page"]

            # Intersect outside_rects across all opinions in the group:
            # start with the first opinion's rects, then intersect with each
            # subsequent opinion's rects. Only areas outside ALL opinions remain.
            result_rects = [
                fitz.Rect(r["x0"], r["y0"], r["x1"], r["y1"])
                for r in first_op.get("outside_rects", [])
                if r["page_index"] == src_page_idx
            ]
            for idx in group[1:]:
                other_rects = [
                    fitz.Rect(r["x0"], r["y0"], r["x1"], r["y1"])
                    for r in opinions[idx].get("outside_rects", [])
                    if r["page_index"] == src_page_idx
                ]
                new_result = []
                for r in result_rects:
                    for o in other_rects:
                        intersection = r & o
                        if (
                            not intersection.is_empty
                            and intersection.x0 < intersection.x1
                            and intersection.y0 < intersection.y1
                        ):
                            new_result.append(intersection)
                result_rects = new_result

            # Filename: strip the -1/-2/-3 suffix from first opinion's name
            base_name = first_op.get("filename", f"{src_page_idx:04d}-{src_page_idx:04d}.pdf")
            # Remove trailing -N before .pdf (e.g. "a3d.214.0001-0001-1.pdf" → "a3d.214.0001-0001.pdf")
            import re

            base_name = re.sub(r"-\d+\.pdf$", ".pdf", base_name)

            out = fitz.open()
            out.insert_pdf(src, from_page=src_page_idx, to_page=src_page_idx)
            fitz_page = out[0]

            # Apply page rects (margins + redactions) in masked mode
            for r in pages_rects.get(str(src_page_idx), []):
                rect = fitz.Rect(r["x0"], r["y0"], r["x1"], r["y1"])
                if rect.is_empty or rect.y0 >= rect.y1 or rect.x0 >= rect.x1:
                    continue
                if r["type"] in WHITE_IN_MASKED:
                    fill = (1, 1, 1)
                else:
                    fill = (0, 0, 0) if r["fill"] == "black" else (1, 1, 1)
                fitz_page.add_redact_annot(rect, fill=fill)

            # Apply intersected outside rects
            for rect in result_rects:
                if not rect.is_empty:
                    fitz_page.add_redact_annot(rect, fill=(1, 1, 1))

            fitz_page.apply_redactions()
            out.save(str(masked_dir / base_name), garbage=4, deflate=True)
            out.close()

    masked_count = len(list(masked_dir.glob("*.pdf")))
    print(f"    Masked {masked_count} files ({len(opinions)} opinions)", flush=True)

    src.close()
    print(f"  Split complete ({time.time() - t0:.0f}s)", flush=True)

    return {
        "full_redacted": full_path,
        "redacted_dir": redacted_dir,
        "masked_dir": masked_dir,
        "opinion_count": len(opinions),
    }


def margins(
    pdf_path: str | Path,
    output_dir: str | Path,
) -> list[dict]:
    """Compute margin rects. Saves margin_rects.json.

    Returns the margin rects data.
    """
    from blackletter.margins import compute_margin_rects

    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    margin_path = output_dir / "margin_rects.json"
    if margin_path.exists():
        print("  Margins already computed, skipping.", flush=True)
        return json.loads(margin_path.read_text())

    t0 = time.time()
    print("  Computing margins...", flush=True)
    rects = compute_margin_rects(pdf_path)
    margin_path.write_text(json.dumps(rects))
    print(f"  Margins done ({time.time() - t0:.0f}s)", flush=True)
    return rects
