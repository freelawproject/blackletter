"""Clean step-by-step API for the blackletter pipeline.

Each function is one discrete step. Call only what you need.
All functions work with file paths and return file paths or data.

Usage:
    from blackletter.api import ensure_weights, bitonal, detect, pair, compute_rects, build_redacted, split_opinions, add_text_layer

    ensure_weights(["large"])  # download large.pt from HF if absent
    bitonal_pdf = bitonal(source_pdf, output_dir)
    detections = detect(bitonal_pdf, output_dir, models=["medium", "large"])
    opinions = pair(detections, bitonal_pdf, reporter="a3d", volume="333", first_page=1)
    rects = compute_rects(bitonal_pdf, output_dir)
    redacted_pdf = build_redacted(bitonal_pdf, output_dir)
    opinion_files = split_opinions(bitonal_pdf, output_dir, opinions=opinions)
    add_text_layer(opinion_files)  # optional, and only worth doing last

No step needs a text layer: the geometry measures the page's ink. ``ocr``
remains available as a pre-pass over a whole source document, but if what
you want is searchable *output*, use ``add_text_layer`` on the files you
are delivering instead. It runs after redaction, so no time is spent
OCRing content that is about to be blacked out.
"""

from __future__ import annotations

import json
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path

import fitz

from blackletter.bl_warm import iter_label_rows


# Hugging Face sources for the YOLO weights. No weights are bundled in
# the package (to keep it small); all are downloaded on demand to
# ``blackletter/weights/``.
# Each weight is pinned to the commit sha it was uploaded at. Pinning
# (instead of tracking ``main``) protects against a compromised repo
# serving different binaries: torch weight files are pickles and can
# execute code on load. Bump a sha deliberately whenever new weights
# are uploaded. bl_warm ships in the same consolidated weights repo as
# the legacy trio, as an ADDITIONAL file, pinned at the commit that
# added it.
_HF_WEIGHTS: dict[str, tuple[str, str, str]] = {
    "small": (
        "freelawproject/blackletter-weights",
        "small.pt",
        "3808b7ef889420cf145e26483106d04ca4de811d",
    ),
    "medium": (
        "freelawproject/blackletter-weights",
        "medium.pt",
        "3808b7ef889420cf145e26483106d04ca4de811d",
    ),
    "large": (
        "freelawproject/blackletter-weights",
        "large.pt",
        "3808b7ef889420cf145e26483106d04ca4de811d",
    ),
    "bl_warm": (
        "freelawproject/blackletter-weights",
        "bl_warm.pt",
        "ee34c6c625dc2d2f49d389946922a5e3861af098",
    ),
}


def ensure_weights(models: list[str] | None = None) -> dict[str, Path]:
    """Ensure named YOLO weights exist under ``blackletter/weights/``.

    Weights already on disk simply resolve to their path; missing ones
    are downloaded from Hugging Face into the package weights
    directory. Safe to call repeatedly; a noop when every requested
    weight is already on disk.

    Call this before :func:`detect` if you want to guarantee that a
    weight is available rather than relying on :func:`detect`'s
    silent-skip behaviour for missing weights.

    :param models: Model size names to ensure (e.g. ``["large"]``).
        Defaults to all three: ``small``, ``medium``, ``large``.
    :returns: Mapping from model name to its resolved path on disk.
    :rtype: dict[str, Path]
    :raises RuntimeError: If ``huggingface_hub`` is not installed but
        a download is required. Install with
        ``pip install blackletter[detect]``.
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
                "Install with `pip install blackletter[detect]`."
            ) from exc

        repo_id, filename, revision = source
        print(f"  Downloading {filename} from {repo_id}...", flush=True)
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
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

    with fitz.open(str(pdf_path)) as src, fitz.open() as out:
        total = src.page_count

        for i in range(total):
            _render_bitonal_page(src[i], out, dpi, threshold)
            if progress_callback and ((i + 1) % 10 == 0 or i == total - 1):
                progress_callback(i + 1, total, f"Bitonal: {i + 1}/{total} pages")
            if (i + 1) % 50 == 0 or i == total - 1:
                print(f"  Bitonal {i + 1}/{total}", flush=True)

        out.save(str(output_path), garbage=4, deflate=True)
    print(f"  Saved bitonal.pdf ({output_path.stat().st_size / 1024 / 1024:.1f} MB)", flush=True)
    return output_path


def ocr(
    pdf_path: str | Path,
    output_dir: str | Path,
    reporter: str = "",
    volume: str = "",
    first_page: int = 1,
    language: str = "eng",
    progress_callback: Callable[[int, int], None] | None = None,
) -> Path:
    """OCR a PDF (add text layer via ocrmypdf/Tesseract).

    :param pdf_path: Path to the source PDF.
    :param output_dir: Directory to write the OCR'd PDF into.
    :param reporter: Reporter abbreviation for the output filename.
    :param volume: Volume number for the output filename.
    :param first_page: First page number (used to build the output filename).
    :param language: Tesseract language code.
    :param progress_callback: Optional callable invoked with
        ``(pages_done, total_pages)`` as Tesseract completes pages. When
        omitted the OCR runs with no progress bar (the default).
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

    # ocrmypdf only exposes per-page progress through its progress-bar
    # plugin hook, so a callback requires enabling the bar and routing
    # its updates to the caller.
    plugin_manager = None
    if progress_callback is not None:
        from ocrmypdf import hookimpl
        from ocrmypdf._plugin_manager import get_plugin_manager

        class _CallbackProgressBar:
            def __init__(self, *, total=None, desc=None, unit=None, **kwargs):
                self._total = total or total_pages
                self._desc = desc
                self._current = 0

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def update(self, n=1, *, completed=None):
                # ocrmypdf builds a fresh progress bar per phase. Several
                # use unit="page" (the pdfinfo "Scanning contents" scan
                # that runs *before* OCR, and the OCR pass itself), so
                # matching on unit alone would report the pre-OCR scan
                # filling to 100% and then reset to 0% for OCR. Track only
                # the "OCR" bar to keep the reported count monotonic.
                if self._desc != "OCR":
                    return
                # The OCR pass calls update(0.5) twice per page
                # (ocrmypdf/_pipelines/ocr.py), so accumulate and report a
                # clean integer page count clamped to the total.
                self._current += n
                done = min(int(self._current), self._total)
                progress_callback(done, self._total)

        class _CallbackProgressPlugin:
            @hookimpl
            def get_progressbar_class(self):
                return _CallbackProgressBar

        plugin_manager = get_plugin_manager()
        plugin_manager._pm.register(_CallbackProgressPlugin())

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
        progress_bar=progress_callback is not None,
        plugin_manager=plugin_manager,
    )
    print(f"  OCR done ({time.time() - t0:.0f}s)", flush=True)
    return output_path


def _text_layer_jobs(total: int, jobs: int | None) -> int:
    """How many files :func:`add_text_layer` should process at once.

    :param total: Number of files that need a text layer.
    :param jobs: Caller's request, or None to decide from the CPU count.
    :returns: A worker count of at least 1 and never more than ``total``.
    """
    import multiprocessing

    if jobs is None:
        jobs = multiprocessing.cpu_count() // 2
    return max(1, min(jobs, total))


def _inner_jobs(workers: int) -> int | None:
    """Cores to give each ocrmypdf run when ``workers`` run side by side.

    Splitting the machine between the workers keeps it busy in both
    regimes without oversubscribing it. A volume of short opinions is
    dominated by ocrmypdf's fixed startup cost, so the win comes from
    running many at once; a handful of long PDFs has little startup cost to
    amortise and wants ocrmypdf's own page-level parallelism instead.

    :param workers: How many files are being processed at once.
    :returns: ocrmypdf's ``jobs`` value, or None to leave its default
        (which is every core) alone for a single worker.
    """
    import multiprocessing

    if workers <= 1:
        return None
    return max(1, multiprocessing.cpu_count() // workers)


def _ocr_in_place(pdf: Path, language: str, optimize: int, jobs: int | None) -> None:
    """OCR one PDF, replacing it only once ocrmypdf has succeeded.

    The temporary file is created in the target's own directory so the
    move is atomic, and removed on failure so a crashed run never leaves
    a stray file beside a deliverable.

    :param pdf: The PDF to give a text layer, modified in place.
    :param language: Tesseract language code.
    :param optimize: ocrmypdf optimization level (0-3).
    :param jobs: ocrmypdf's internal worker count, or None for its default.
    """
    from blackletter.ocr import _silence_ocr_loggers

    _silence_ocr_loggers()

    import ocrmypdf

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False, dir=pdf.parent) as tmp:
        tmp_path = Path(tmp.name)
    extra = {} if jobs is None else {"jobs": jobs}
    try:
        ocrmypdf.ocr(
            str(pdf),
            str(tmp_path),
            pdf_renderer="auto",
            optimize=optimize,
            output_type="pdf",
            language=[language],
            # Leave pages that already have text alone rather than
            # failing on them or rasterising them away.
            skip_text=True,
            tesseract_timeout=120,
            progress_bar=False,
            **extra,
        )
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    tmp_path.replace(pdf)


def add_text_layer(
    paths: str | Path | Iterable[str | Path],
    language: str = "eng",
    optimize: int = 1,
    skip_existing: bool = True,
    jobs: int | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> list[Path]:
    """Add a searchable text layer to PDFs that already exist, in place.

    This is the step to reach for when what you want is searchable
    *output*. :func:`ocr` is a pre-pass over a whole source document,
    named from reporter, volume and page numbers, and run before anything
    is redacted; this runs over files that are already finished, so no CPU
    is spent OCRing content that is about to be blacked out and no text
    layer is left for ``apply_redactions`` to scrub.

    Nothing calls this implicitly. A caller that wants searchable
    deliverables asks for them, typically over the full redacted PDF and
    the per-opinion PDFs in ``redacted/``, and typically not over
    ``unredacted/`` (a searchable copy of the copyrighted text is the
    opposite of the point).

    Work is spread across files rather than within them, because that is
    where the time goes: each ocrmypdf run carries several seconds of fixed
    cost, which dominates for the short PDFs a volume splits into, and
    ocrmypdf parallelises poorly over a handful of pages. Each file is
    OCR'd to a temporary file beside it and moved into place only on
    success, so a failure leaves the original untouched, and pages that
    already carry text are left alone, so running this twice is safe.

    :param paths: A PDF, a directory of PDFs (non-recursive), or an
        iterable of either.
    :param language: Tesseract language code.
    :param optimize: ocrmypdf optimization level (0-3).
    :param skip_existing: Skip files that already have a text layer
        throughout. Pass False to OCR every file's text-less pages anyway.
    :param jobs: How many files to process at once. Defaults to half the
        CPU count, capped at the number of files. Pass 1 to run in the
        calling process, which is also what happens for a single file.
    :param progress_callback: Optional callable invoked with
        ``(files_done, total_files, message)``.
    :returns: The files a text layer was run over, sorted by path. Files
        skipped as already searchable are not included; with
        ``skip_existing`` False a file whose every page already had text is
        still listed, since the pass ran even though it added nothing.
    """
    from blackletter.ocr import needs_ocr

    pdfs = _collect_pdfs(paths)
    if skip_existing:
        pdfs = [p for p in pdfs if needs_ocr(p)]
    total = len(pdfs)
    if not total:
        print("  Text layer: nothing to do", flush=True)
        return []

    jobs = _text_layer_jobs(total, jobs)
    inner_jobs = _inner_jobs(jobs)

    written: list[Path] = []
    done = 0
    t0 = time.time()

    def _report(pdf: Path) -> None:
        nonlocal done
        done += 1
        written.append(pdf)
        if progress_callback:
            progress_callback(done, total, f"Text layer: {pdf.name}")
        if done % 10 == 0 or done == total:
            print(f"  Text layer {done}/{total}", flush=True)

    if jobs == 1:
        for pdf in pdfs:
            _ocr_in_place(pdf, language, optimize, inner_jobs)
            _report(pdf)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        print(f"  Text layer: {total} PDFs across {jobs} workers", flush=True)
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {
                pool.submit(_ocr_in_place, pdf, language, optimize, inner_jobs): pdf for pdf in pdfs
            }
            for future in as_completed(futures):
                future.result()
                _report(futures[future])

    print(
        f"  Text layer added to {len(written)}/{total} PDFs ({time.time() - t0:.0f}s)",
        flush=True,
    )
    # Workers finish out of order, so sort rather than return whichever
    # order the pool happened to complete in.
    return sorted(written)


def _collect_pdfs(paths: str | Path | Iterable[str | Path]) -> list[Path]:
    """Expand a path, a directory, or an iterable of them into PDF files.

    :param paths: A PDF, a directory of PDFs, or an iterable of either.
    :returns: Existing ``.pdf`` files, directories expanded and sorted.
    :raises FileNotFoundError: If a named path does not exist.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    out: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            out.extend(
                sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")
            )
        elif path.is_file():
            out.append(path)
        else:
            raise FileNotFoundError(path)
    # A caller naming both a file and its directory should not have it
    # processed twice, which with a worker pool means two workers OCRing
    # the same file at once.
    return list(dict.fromkeys(p.resolve() for p in out))


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

    mat = fitz.Matrix(DPI / 72, DPI / 72)

    all_raw = []
    with fitz.open(str(pdf_path)) as pdf:
        total = pdf.page_count
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
                    for class_id, conf_v, xyxy in iter_label_rows(res):
                        try:
                            label_name = Label(class_id).name
                        except ValueError:
                            continue
                        all_raw.append(
                            {
                                "page_index": pm["index"],
                                "label": label_name,
                                "label_id": class_id,
                                "confidence": round(conf_v, 3),
                                "bbox": [round(v, 1) for v in xyxy],
                                "img_width": pm["img_width"],
                                "img_height": pm["img_height"],
                                "model": model_name,
                            }
                        )
                if (bs + YOLO_BATCH) % 100 == 0 or bs + YOLO_BATCH >= total:
                    print(f"    {min(bs + YOLO_BATCH, total)}/{total} pages", flush=True)

            print(f"    {model_name} done ({time.time() - t0:.0f}s)", flush=True)

    # Merge across models with label-specific strategies:
    #   CASE_CAPTION, KEY_ICON: trusted models only (large hallucinates;
    #     of the legacy trio only medium is trusted — bl_warm is trusted
    #     as a single model)
    #   CASE_SEQUENCE: all models, overlaps keep smallest box
    #   Everything else: all models, overlaps keep highest confidence
    MEDIUM_ONLY = {"CASE_CAPTION", "KEY_ICON"}
    TRUSTED_CAPTION_MODELS = {"medium", "bl_warm"}
    SMALLEST_BOX = {"CASE_SEQUENCE"}

    filtered = [
        d
        for d in all_raw
        if d["label"] not in MEDIUM_ONLY
        or d["model"] in TRUSTED_CAPTION_MODELS
        or d["model"].startswith("bl_warm")
    ]

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
    :param pdf_path: Path to the PDF to work from.
    :param reporter: Reporter abbreviation (e.g. ``"a3d"``).
    :param volume: Volume number (e.g. ``"333"``).
    :param first_page: First page number of the volume.
    :param excluded: Set of page indices to exclude from pairing.
    :returns: List of opinion dicts with page ranges, bboxes, and
        outside_rects.
    """
    from blackletter.models import Detection as BLDetection, Document, Page
    from blackletter.scanner import _pair_opinions, snap_document_columns

    pdf_path = Path(pdf_path)

    # Load detections
    if isinstance(detections, (str, Path)):
        raw = json.loads(Path(detections).read_text())
    else:
        raw = detections

    # Build Document
    from blackletter.scanner import _group_detections_by_page

    pages_data = _group_detections_by_page(raw, include_page_number_end=True)

    with fitz.open(str(pdf_path)) as src_pdf:
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
        snap_document_columns(document)

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

    :param pdf_path: Path to the PDF to work from.
    :param output_dir: Directory containing detections.json.
    :param excluded: Set of page indices to exclude from pairing.
    :param approved: Set of page indices pre-approved for redaction.
    :param skip_doctr: Skip the docTR headnote-refinement pass. Set True on an
        install without the ``refine`` extra to avoid importing ``doctr``.
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
        skip_doctr=skip_doctr,
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

    :param pdf_path: Path to the PDF to work from.
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
    with fitz.open(str(pdf_path)) as src_pdf:
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

    from blackletter.scanner import snap_document_columns

    document = Document(pdf_path=pdf_path, pages=pages, ocr_applied=True)
    snap_document_columns(document)

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


def build_redactions(
    pages: Iterable,
    redaction_rects: list[dict],
    margin_rects: list[dict],
    opinions: list[dict],
    reporter: str = "",
    volume: str = "",
) -> dict:
    """Combine this library's own outputs into :func:`generate`'s input.

    ``compute_rects`` returns image-pixel rects, ``compute_margin_rects``
    returns PDF points, and ``generate`` wants one payload in points with
    every rect labelled. Without this, each consumer converts and merges
    them itself, and gets to rediscover that the scale factors come from
    the page's own image dimensions rather than a global assumption.

    :param pages: The detected pages, for their dimensions and scale.
    :param redaction_rects: ``[{"page_index", "rects"}]`` in image pixels,
        each rect carrying ``fill`` and ``type``.
    :param margin_rects: ``[{"page_index", "rects"}]`` in PDF points.
    :param opinions: Opinion dicts, as ``pair`` produces them. Given a
        reporter and volume, each gains a ``filename``. Modified in place
        and returned inside the payload, rather than copied.
    :param reporter: Reporter abbreviation for the output filenames.
    :param volume: Volume number for the output filenames.
    :returns: ``{"opinions": [...], "pages": {"<index>": [rect, ...]}}``,
        all coordinates in PDF points. Pages absent from both rect lists
        are absent from ``pages``.
    :raises KeyError: If ``redaction_rects`` names a page that ``pages``
        does not, since its pixel rects could then only be guessed at.
    """
    by_index = {p.index: p for p in pages}

    prefix = f"{reporter}.{volume}" if reporter and volume else ""
    if prefix:
        # Two opinions can share a page range, and ``generate`` suffixes the
        # duplicates -1, -2. Doing the same here means the names in this
        # payload are the names that end up on disk, which is the only
        # reason to carry them.
        stems = [
            f"{prefix}.{op.get('first_page_number', 0):04d}-"
            f"{op.get('last_page_number', op.get('first_page_number', 0)):04d}"
            for op in opinions
        ]
        counts = Counter(stems)
        seen: dict[str, int] = {}
        for op, stem in zip(opinions, stems, strict=True):
            if counts[stem] > 1:
                seen[stem] = seen.get(stem, 0) + 1
                op["filename"] = f"{stem}-{seen[stem]}.pdf"
            else:
                op["filename"] = f"{stem}.pdf"

    combined: dict[int, list[dict]] = {}

    for entry in margin_rects:
        page_rects = combined.setdefault(entry["page_index"], [])
        for r in entry.get("rects", []):
            page_rects.append(
                {
                    "x0": round(r["x0"], 1),
                    "y0": round(r["y0"], 1),
                    "x1": round(r["x1"], 1),
                    "y1": round(r["y1"], 1),
                    "fill": "white",
                    "type": "margin",
                }
            )

    for entry in redaction_rects:
        page_index = entry["page_index"]
        page_rects = combined.setdefault(page_index, [])
        page = by_index.get(page_index)
        if page is None:
            raise KeyError(
                f"redaction_rects names page {page_index}, which is not in `pages`. "
                "Its rects are in image pixels and there is nothing to scale them by."
            )
        # A page with no detections has no image dimensions to scale by, and
        # its rects are already in points.
        to_x = page.scale_x if page.img_width > 1 else 1.0
        to_y = page.scale_y if page.img_height > 1 else 1.0
        for r in entry.get("rects", []):
            x0, y0 = r["x0"] * to_x, r["y0"] * to_y
            x1, y1 = r["x1"] * to_x, r["y1"] * to_y
            if x0 >= x1 or y0 >= y1:
                continue
            page_rects.append(
                {
                    "x0": round(x0, 1),
                    "y0": round(y0, 1),
                    "x1": round(x1, 1),
                    "y1": round(y1, 1),
                    "fill": r["fill"],
                    "type": r["type"],
                }
            )

    return {
        "opinions": opinions,
        "pages": {str(k): v for k, v in sorted(combined.items())},
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

    :param pdf_path: Path to the source PDF.
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

    with fitz.open(str(pdf_path)) as src:

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
            applied: list[tuple[fitz.Rect, tuple]] = []
            for r in pages_rects.get(str(src_idx), []):
                rect = fitz.Rect(r["x0"], r["y0"], r["x1"], r["y1"])
                if rect.is_empty or rect.y0 >= rect.y1 or rect.x0 >= rect.x1:
                    continue
                if r["type"] == "margin":
                    fill = (1, 1, 1)
                else:
                    fill = (0, 0, 0) if r["fill"] == "black" else (1, 1, 1)
                fitz_page.add_redact_annot(rect, fill=fill)
                applied.append((rect, fill))

            # Outside-opinion whiteout (skip for full redacted)
            if opinion is not None and mode != "full":
                for orect in opinion.get("outside_rects", []):
                    if orect["page_index"] != src_idx:
                        continue
                    rect = fitz.Rect(orect["x0"], orect["y0"] + 3, orect["x1"], orect["y1"])
                    if not rect.is_empty:
                        fitz_page.add_redact_annot(rect, fill=(1, 1, 1))
                        applied.append((rect, (1, 1, 1)))

            fitz_page.apply_redactions()

            # PyMuPDF has painted each redaction as a fill *and* a 1pt
            # stroke straddling its edge, strokes last, which left a hairline
            # wherever a black rect met a white one; that was visible in real
            # deliverables. It does not reproduce on 1.26.7, so treat this as
            # insurance rather than a fix: repainting fill-only in the same
            # order costs one op per rect and covers the case if a future
            # version strokes again. The CLI path has always done it.
            for rect, fill in applied:
                fitz_page.draw_rect(rect, fill=fill, color=None, width=0)

        # ── Full redacted PDF ──
        t0 = time.time()
        # Name: reporter.volume.first_page.last_page.redacted.pdf
        first_pn = opinions[0].get("first_page_number", 1)
        last_pn = opinions[-1].get("last_page_number", first_pn)
        full_name = f"{prefix}{first_pn}.{last_pn}.redacted.pdf"
        full_path = output_dir / full_name
        with fitz.open() as full_out:
            full_out.insert_pdf(src)
            for page_idx in range(full_out.page_count):
                _apply_page(full_out[page_idx], page_idx, None, "full")
                if progress_callback and (
                    (page_idx + 1) % 20 == 0 or page_idx == full_out.page_count - 1
                ):
                    progress_callback(page_idx + 1, full_out.page_count, "Redacting pages...")
            full_out.save(str(full_path), garbage=4, deflate=True)
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

            with fitz.open() as out:
                out.insert_pdf(src, from_page=start_idx, to_page=end_idx)
                for local_idx, src_idx in enumerate(range(start_idx, end_idx + 1)):
                    _apply_page(out[local_idx], src_idx, op, "redacted")
                out.save(str(redacted_dir / filename), garbage=4, deflate=True)

            if unredacted:
                with fitz.open() as out:
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

    print(f"  Split complete ({time.time() - t0:.0f}s)", flush=True)

    result = {
        "full_redacted": full_path,
        "redacted_dir": redacted_dir,
        "opinion_count": len(opinions),
    }
    if llm:
        result["llm_dir"] = llm_dir
    return result
