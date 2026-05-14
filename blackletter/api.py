"""Clean step-by-step API for the blackletter pipeline.

Each function is one discrete step. Call only what you need.
All functions work with file paths and return file paths or data.

Usage:
    from blackletter.api import ensure_weights, bitonal, ocr, detect, pair, compute_rects, build_redacted, split_opinions

    ensure_weights(["large"])  # download large.pt from HF if absent
    bitonal_pdf = bitonal(source_pdf, output_dir)
    ocr_pdf = ocr(bitonal_pdf, output_dir)
    detections = detect(bitonal_pdf, output_dir, models=["medium", "large"])
    opinions = pair(detections, ocr_pdf, reporter="a3d", volume="333", first_page=1)
    rects = compute_rects(ocr_pdf, output_dir)
    redacted_pdf = build_redacted(ocr_pdf, output_dir)
    opinion_files = split_opinions(ocr_pdf, output_dir, opinions=opinions)
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import fitz


# Hugging Face sources for weights not bundled in the package.
# ``small`` and ``medium`` ship inside ``blackletter/weights/`` via
# ``package-data`` in ``pyproject.toml``; ``large`` is too big for
# PyPI and is downloaded on demand.
_HF_WEIGHTS: dict[str, tuple[str, str]] = {
    "large": ("flooie/blackletter-large", "large.pt"),
}


def ensure_weights(models: list[str] | None = None) -> dict[str, Path]:
    """Ensure named YOLO weights exist under ``blackletter/weights/``.

    For bundled weights (``small``, ``medium``), this simply resolves
    the path. For weights sourced from Hugging Face (currently only
    ``large``), downloads them to the package weights directory if
    they are not already present. Safe to call repeatedly; a noop when
    every requested weight is already on disk.

    Call this before :func:`detect` if you want to guarantee that a
    weight is available rather than relying on :func:`detect`'s
    silent-skip behaviour for missing weights.

    :param models: Model size names to ensure (e.g. ``["large"]``).
        Defaults to all three: ``small``, ``medium``, ``large``.
    :returns: Mapping from model name to its resolved path on disk.
    :rtype: dict[str, Path]
    :raises RuntimeError: If ``huggingface_hub`` is not installed but
        a download is required. Install with
        ``pip install blackletter[analyze]``.
    :raises FileNotFoundError: If a requested weight is missing from
        the installation and has no Hugging Face source.
    """
    weights_dir = Path(__file__).resolve().parent / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    if models is None:
        models = ["small", "medium", "large"]

    resolved: dict[str, Path] = {}
    for name in models:
        path = weights_dir / f"{name}.pt"
        if path.is_file():
            resolved[name] = path
            continue

        source = _HF_WEIGHTS.get(name)
        if source is None:
            raise FileNotFoundError(f"Weight {path} is missing and has no Hugging Face source.")

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError(
                f"huggingface_hub is required to download {name}.pt. "
                "Install with `pip install blackletter[analyze]`."
            ) from exc

        repo_id, filename = source
        print(f"  Downloading {filename} from {repo_id}...", flush=True)
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(weights_dir),
        )
        resolved[name] = Path(downloaded)

    return resolved


def bitonal(
    pdf_path: str | Path,
    output_dir: str | Path,
    dpi: int = 200,
    threshold: int = 160,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> Path:
    """Convert a PDF to bitonal (CCITT G4 TIFF images).

    :param pdf_path: Path to the source PDF.
    :param output_dir: Directory to write bitonal.pdf into.
    :param dpi: Rendering resolution for rasterisation.
    :param threshold: Grayscale threshold (0-255) for binarisation.
    :param progress_callback: Optional callable(current, total, message)
        invoked during processing.
    :returns: Path to the bitonal PDF.
    """
    from blackletter.ocr import _render_bitonal_page

    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "bitonal.pdf"

    src = fitz.open(str(pdf_path))
    out = fitz.open()
    total = src.page_count

    for i in range(total):
        _render_bitonal_page(src[i], out, dpi, threshold)
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

    :param pdf_path: Path to the source PDF.
    :param output_dir: Directory to write the OCR'd PDF into.
    :param reporter: Reporter abbreviation for the output filename.
    :param volume: Volume number for the output filename.
    :param first_page: First page number (used to build the output filename).
    :param language: Tesseract language code.
    :returns: Path to the OCR'd PDF.
    """
    from blackletter.ocr import _silence_ocr_loggers

    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    # Build scan name
    with fitz.open(str(pdf_path)) as src:
        total_pages = src.page_count
    last_page = first_page + total_pages - 1
    parts = [p for p in [reporter, str(volume), str(first_page), str(last_page)] if p]
    scan_name = ".".join(parts) if parts else pdf_path.stem
    output_path = output_dir / f"{scan_name}.pdf"

    _silence_ocr_loggers()

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

    :param pdf_path: Path to the bitonal PDF.
    :param output_dir: Directory to write detections.json into.
    :param models: Model size names to run (e.g. ``["medium", "large"]``).
        Defaults to all three: small, medium, large.
    :param confidence: Minimum confidence threshold for detections.
    :returns: Merged detection list. Also saves detections.json to
        *output_dir*.
    """
    from PIL import Image
    from ultralytics import YOLO
    from blackletter.models import Label

    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    if models is None:
        models = ["small", "medium", "large"]

    resolved = ensure_weights(models)

    from blackletter.scanner import DPI, YOLO_BATCH

    pdf = fitz.open(str(pdf_path))
    mat = fitz.Matrix(DPI / 72, DPI / 72)
    total = pdf.page_count

    all_raw = []
    for model_name in models:
        model_file = resolved[model_name]
        model = YOLO(str(model_file))
        print(f"  Detecting with {model_name}...", flush=True)
        t0 = time.time()

        for bs in range(0, total, YOLO_BATCH):
            be = min(bs + YOLO_BATCH, total)
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
            if (bs + YOLO_BATCH) % 100 == 0 or bs + YOLO_BATCH >= total:
                print(f"    {min(bs + YOLO_BATCH, total)}/{total} pages", flush=True)

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
        """Compute intersection-over-union of two ``[x0, y0, x1, y1]`` boxes.

        :param a: First bounding box.
        :param b: Second bounding box.
        :returns: IoU value in ``[0.0, 1.0]``.
        """
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
        """Check whether *outer* bbox fully contains *inner* bbox.

        :param outer: Outer ``[x0, y0, x1, y1]`` bounding box.
        :param inner: Inner ``[x0, y0, x1, y1]`` bounding box.
        :returns: ``True`` if *outer* contains *inner*.
        """
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
    """Pair opinions from detections.

    :param detections: Detection list or path to detections.json.
    :param pdf_path: Path to the OCR'd PDF.
    :param reporter: Reporter abbreviation (e.g. ``"a3d"``).
    :param volume: Volume number (e.g. ``"333"``).
    :param first_page: First page number of the volume.
    :param excluded: Set of page indices to exclude from pairing.
    :returns: List of opinion dicts with page ranges, bboxes, and
        outside_rects.
    """
    from blackletter.models import Detection as BLDetection, Document, Page
    from blackletter.scanner import _pair_opinions

    pdf_path = Path(pdf_path)

    # Load detections
    if isinstance(detections, (str, Path)):
        raw = json.loads(Path(detections).read_text())
    else:
        raw = detections

    # Build Document
    from blackletter.scanner import _group_detections_by_page

    src_pdf = fitz.open(str(pdf_path))
    pages_data = _group_detections_by_page(raw, include_page_number_end=True)

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
            page.detections.append(BLDetection.from_raw_dict(d, pi, bbox_default=[0, 0, 1, 1]))
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
    from blackletter.scanner import _build_opinions_data, _opinion_page_bounds

    pages_by_index = {p.index: p for p in document.pages}
    opinions_data = _build_opinions_data(opinions, pages_by_index, src_pdf)

    # Augment each entry with filename-inference fields unique to this API
    for idx, entry in enumerate(opinions_data):
        start_idx, end_idx = page_ranges[idx]
        first_num, last_num = _opinion_page_bounds(
            pages_by_index.get(start_idx),
            pages_by_index.get(end_idx),
            start_idx,
            end_idx,
            first_page,
        )
        entry["end_page"] = end_idx
        entry["first_page_number"] = first_num
        entry["last_page_number"] = last_num

    src_pdf.close()

    return opinions_data


def compute_rects(
    pdf_path: str | Path,
    output_dir: str | Path,
    excluded: set | None = None,
    approved: set | None = None,
    skip_doctr: bool = False,
) -> list[dict]:
    """Compute redaction rects from detections + opinions.

    Reads detections.json from *output_dir*, pairs opinions, and writes
    redaction_rects.json back into *output_dir*.

    :param pdf_path: Path to the OCR'd PDF.
    :param output_dir: Directory containing detections.json.
    :param excluded: Set of page indices to exclude from pairing.
    :param approved: Set of page indices pre-approved for redaction.
    :param skip_doctr: Unused, kept for backwards compatibility.
    :returns: List of redaction rect dicts.
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
    rects: str | Path | None = None,
) -> Path:
    """Build the full redacted PDF from precomputed rects.

    :param pdf_path: Path to the OCR'd PDF.
    :param output_dir: Directory containing detections.json and where the
        redacted PDF will be written.
    :param rects: Path to a redaction_rects.json file. If ``None``, uses
        ``output_dir / "redaction_rects.json"``.
    :returns: Path to the redacted PDF.
    """
    from blackletter.process import _build_redacted_from_rects
    from blackletter.models import Document, Page

    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    # Load rects
    rects_path = Path(rects) if rects is not None else output_dir / "redaction_rects.json"

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

    Creates redacted/ and (optionally) unredacted/ subdirectories.

    :param pdf_path: Path to the OCR'd (unredacted) PDF.
    :param output_dir: Directory containing the redacted PDF and
        opinions.json. Output subdirectories are created here.
    :param unredacted: Whether to also generate unredacted opinion PDFs.
    :param opinions: Precomputed opinions list (from :func:`pair`). If
        ``None``, reads opinions.json from *output_dir*.
    :returns: Dict with ``redacted`` and ``unredacted`` counts.
    """
    from blackletter.process import _split_from_redacted

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

    return {
        "redacted": len(redacted_files),
        "unredacted": len(list((output_dir / "unredacted").glob("*.pdf"))) if unredacted else 0,
    }


def generate(
    pdf_path: str | Path,
    redactions: str | Path | dict,
    output_dir: str | Path,
    reporter: str = "",
    volume: str = "",
    unredacted: bool = False,
    llm: bool = False,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict:
    """Generate all output PDFs from a source PDF and a redactions payload.

    The *redactions* payload (file or dict) contains:

    - ``"opinions"``: list of opinion dicts with outside_rects, page
      ranges, and filenames.
    - ``"pages"``: dict mapping page_index to a list of rects (margins
      + redaction rects, all in PDF points).

    Builds in one pass per page (no layering).

    :param pdf_path: Path to the source (OCR'd) PDF.
    :param redactions: Path to redactions.json, or the parsed dict.
    :param output_dir: Base output directory.
    :param reporter: Reporter abbreviation for filenames (e.g.
        ``"a3d"``).
    :param volume: Volume number for filenames (e.g. ``"214"``).
    :param unredacted: Also generate unredacted opinion PDFs.
    :param llm: Also generate per-page LLM PDFs with invisible
        ``<--CASEEND-->`` stamps on Key-icon locations.
    :param progress_callback: Optional callable(current, total, message).
    :returns: Dict with keys ``redacted_dir``, ``full_redacted``, and
        ``opinion_count``. Includes ``llm_dir`` when *llm* is True.
    """
    import re

    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(redactions, (str, Path)):
        data = json.loads(Path(redactions).read_text())
    else:
        data = redactions

    opinions = data["opinions"]
    pages_rects = data["pages"]

    # Build prefix from reporter/volume
    prefix = ""
    if reporter:
        prefix += f"{reporter}."
    if volume:
        prefix += f"{volume}."

    def _opinion_filename(op):
        """Build filename from opinion page numbers.

        :param op: Opinion dict with ``first_page_number`` and
            ``last_page_number`` keys.
        :returns: Filename string (e.g. ``"a3d.333.0001-0010.pdf"``).
        """
        first = op.get("first_page_number")
        last = op.get("last_page_number")
        if first is not None and last is not None:
            return f"{prefix}{first:04d}-{last:04d}.pdf"
        # Fall back to existing filename
        return op.get("filename", f"{op['caption_page']:04d}-{op['end_page']:04d}.pdf")

    # Detect duplicate filenames and add -1/-2/-3 suffixes
    raw_names = [_opinion_filename(op) for op in opinions]
    name_counts = Counter(raw_names)
    name_seq: dict[str, int] = {}
    filenames = []
    for name in raw_names:
        if name_counts[name] > 1:
            name_seq[name] = name_seq.get(name, 0) + 1
            filenames.append(re.sub(r"\.pdf$", f"-{name_seq[name]}.pdf", name))
        else:
            filenames.append(name)

    src = fitz.open(str(pdf_path))

    def _apply_page(fitz_page, src_idx, opinion, mode):
        """Apply all rects for one page in one pass.

        :param fitz_page: The ``fitz.Page`` to apply redactions to.
        :param src_idx: Source page index in the original PDF.
        :param opinion: Opinion dict (or ``None`` for full-redacted mode).
        :param mode: One of ``"full"`` (all rects, no outside-opinion
            whiteout) or ``"redacted"`` (all rects + outside-opinion
            whiteout).
        """
        # Page rects (margins + redactions), all PDF points
        for r in pages_rects.get(str(src_idx), []):
            rect = fitz.Rect(r["x0"], r["y0"], r["x1"], r["y1"])
            if rect.is_empty or rect.y0 >= rect.y1 or rect.x0 >= rect.x1:
                continue
            if r["type"] == "margin":
                fill = (1, 1, 1)
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

    # Name: reporter.volume.first_page.last_page.redacted.pdf
    first_pn = opinions[0].get("first_page_number", 1)
    last_pn = opinions[-1].get("last_page_number", first_pn)
    full_name = f"{prefix}{first_pn}.{last_pn}.redacted.pdf"
    full_path = output_dir / full_name
    full_out.save(str(full_path), garbage=4, deflate=True)
    full_out.close()
    print(
        f"  Full redacted: {full_path.name} ({full_path.stat().st_size / 1024 / 1024:.1f} MB, {time.time() - t0:.0f}s)",
        flush=True,
    )

    # ── Split opinions ──
    redacted_dir = output_dir / "redacted"
    redacted_dir.mkdir(exist_ok=True)

    if unredacted:
        unredacted_dir = output_dir / "unredacted"
        unredacted_dir.mkdir(exist_ok=True)

    if llm:
        llm_dir = output_dir / "llm"
        llm_dir.mkdir(exist_ok=True)

    t0 = time.time()

    # ── Redacted + unredacted: one PDF per opinion ──
    for i, op in enumerate(opinions):
        start_idx = op["caption_page"]
        end_idx = op["end_page"]
        filename = filenames[i]

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

    # ── LLM per-page split with CASEEND stamps on Key icons (opt-in) ──
    if llm:
        from blackletter.process import _split_llm_pages

        t_llm = time.time()
        key_by_page: dict[int, list[fitz.Rect]] = {}
        for pi_str, rs in pages_rects.items():
            rects = [
                fitz.Rect(r["x0"], r["y0"], r["x1"], r["y1"])
                for r in rs
                if r.get("type") == "KEY_ICON"
            ]
            if rects:
                key_by_page[int(pi_str)] = rects
        total = _split_llm_pages(full_path, key_by_page, llm_dir)
        print(f"    LLM {total} pages ({time.time() - t_llm:.0f}s)", flush=True)

    src.close()
    print(f"  Split complete ({time.time() - t0:.0f}s)", flush=True)

    result = {
        "full_redacted": full_path,
        "redacted_dir": redacted_dir,
        "opinion_count": len(opinions),
    }
    if llm:
        result["llm_dir"] = llm_dir
    return result
