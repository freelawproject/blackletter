"""Analyze a PDF's page number sequence for QA/verification."""

from __future__ import annotations

import os
import re
from multiprocessing import Pool, cpu_count
from pathlib import Path

from PIL import Image

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

DEFAULT_ANALYZE_MODEL = Path(__file__).resolve().parent / "models" / "large.pt"

CROP_FRAC = 0.07
MIN_SCORE = 0.80
PAGE_HEADER_CLASS = 2
PAGE_NUMBER_CLASS = 8

RANGE_RE = re.compile(r"^(\d{1,4})\s*[–\-]\s*(\d{1,4})$")


def _page_num_base(text: str) -> int | None:
    m = re.match(r"^(\d+)", str(text))
    return int(m.group(1)) if m else None


def _classify_text(stripped: str) -> tuple[str, str] | None:
    if RANGE_RE.match(stripped):
        return stripped, "range"
    if stripped.isdigit() and len(stripped) <= 4:
        return stripped, "single"
    cleaned = stripped.strip(".,;:!?()[]{}'\"")
    if RANGE_RE.match(cleaned):
        return cleaned, "range"
    if cleaned.isdigit() and len(cleaned) <= 4:
        return cleaned, "single"
    m = re.search(r"\b(\d{1,4})\s*$", stripped)
    if m:
        return m.group(1), "single"
    return None


def _scan_crop(
    ocr,
    img_bytes: bytes,
    zone_name: str,
    page_idx: int,
    x_offset: int = 0,
    y_offset: int = 0,
) -> tuple[list, list]:
    tmp_path = f"/tmp/pn_{os.getpid()}_{page_idx}_{zone_name}.png"
    with open(tmp_path, "wb") as f:
        f.write(img_bytes)
    hits, near_misses = [], []
    if ocr is None:
        try:
            import pytesseract
            from PIL import Image as _I

            pil_img = _I.open(tmp_path)
            tess_text = pytesseract.image_to_string(pil_img, config="--psm 6").strip()
            for line in tess_text.splitlines():
                classified = _classify_text(line.strip())
                if classified:
                    clean_text, page_type = classified
                    hits.append(
                        {
                            "text": clean_text,
                            "score": 0.5,
                            "zone": zone_name,
                            "type": page_type,
                            "poly": None,
                        }
                    )
        except Exception:
            pass
        return hits, near_misses
    result = ocr.predict(tmp_path)
    if not result or not result[0]:
        return hits, near_misses
    r = result[0]
    texts = r.get("rec_texts", []) if isinstance(r, dict) else getattr(r, "rec_texts", [])
    scores = r.get("rec_scores", []) if isinstance(r, dict) else getattr(r, "rec_scores", [])
    polys = r.get("dt_polys", []) if isinstance(r, dict) else getattr(r, "dt_polys", [])
    for text, score, poly in zip(texts, scores, polys):
        stripped = text.strip()
        classified = _classify_text(stripped)
        poly_pts = [[p[0] + x_offset, p[1] + y_offset] for p in poly]
        if classified:
            clean_text, page_type = classified
            hits.append(
                {
                    "text": clean_text,
                    "score": score,
                    "zone": zone_name,
                    "poly": poly_pts,
                    "type": page_type,
                }
            )
        elif re.search(r"\d{1,4}", stripped):
            near_misses.append(
                {"text": stripped, "score": score, "zone": zone_name, "poly": poly_pts}
            )
    return hits, near_misses


def _ocr_crop_multi(
    ocr,
    glm_processor,
    glm_model,
    pil_crop,
    zone_name: str,
    page_idx: int,
    exp_start: int | None = None,
    exp_end: int | None = None,
) -> dict | None:
    margin = 20

    def is_valid(text, typ):
        if typ == "range":
            return True
        if typ == "single" and exp_start is not None:
            try:
                num = int(text)
                return (exp_start - margin) <= num <= (exp_end + margin)
            except ValueError:
                return False
        return typ == "single"

    tmp_path = f"/tmp/pn_{os.getpid()}_{page_idx}_{zone_name}.png"
    pil_crop.save(tmp_path)

    # 1) PaddleOCR
    if ocr is None:
        result = None
    else:
        result = ocr.predict(tmp_path)
    if result and result[0]:
        r = result[0]
        texts = r.get("rec_texts", []) if isinstance(r, dict) else getattr(r, "rec_texts", [])
        scores = r.get("rec_scores", []) if isinstance(r, dict) else getattr(r, "rec_scores", [])
        for text, score in zip(texts, scores):
            classified = _classify_text(text.strip())
            if classified:
                clean_text, page_type = classified
                if is_valid(clean_text, page_type):
                    return {
                        "text": clean_text,
                        "score": score,
                        "zone": zone_name,
                        "type": page_type,
                        "ocr": "paddle",
                    }

    # 2) Tesseract
    try:
        import pytesseract

        tess_text = pytesseract.image_to_string(pil_crop, config="--psm 7 digits").strip()
        if tess_text:
            classified = _classify_text(tess_text)
            if classified:
                clean_text, page_type = classified
                if is_valid(clean_text, page_type):
                    return {
                        "text": clean_text,
                        "score": 0.5,
                        "zone": zone_name,
                        "type": page_type,
                        "ocr": "tesseract",
                    }
    except Exception:
        pass

    # 3) GLM-OCR — lazy-load only when paddle + tesseract both fail
    if glm_processor is None and glm_model is None:
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor

            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            device_map = "auto" if torch.cuda.is_available() else "cpu"
            glm_processor = AutoProcessor.from_pretrained("zai-org/GLM-OCR")
            glm_model = AutoModelForImageTextToText.from_pretrained(
                "zai-org/GLM-OCR", torch_dtype=dtype, device_map=device_map
            )
            # Cache on the function for reuse
            _process_page._glm_processor = glm_processor
            _process_page._glm_model = glm_model
            print(f"  GLM-OCR loaded (page {page_idx})", flush=True)
        except Exception:
            pass
    if glm_processor is not None and glm_model is not None:
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": tmp_path},
                        {
                            "type": "text",
                            "text": "What number is shown in this image? Reply with just the number.",
                        },
                    ],
                }
            ]
            inputs = glm_processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(glm_model.device)
            inputs.pop("token_type_ids", None)
            generated_ids = glm_model.generate(**inputs, max_new_tokens=32)
            glm_text = glm_processor.decode(
                generated_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            ).strip()
            if glm_text:
                classified = _classify_text(glm_text)
                if classified:
                    clean_text, page_type = classified
                    if is_valid(clean_text, page_type):
                        return {
                            "text": clean_text,
                            "score": 0.5,
                            "zone": zone_name,
                            "type": page_type,
                            "ocr": "glm",
                        }
        except Exception:
            pass

    return None


def _pick_best(
    candidates: list,
    img_w: int,
    img_h: int,
    exp_start: int | None = None,
    exp_end: int | None = None,
) -> dict | None:
    if not candidates:
        return None

    positioned = []
    for c in candidates:
        poly = c["poly"]
        if not poly:
            positioned.append(c)
            continue
        y_center = sum(p[1] for p in poly) / len(poly)
        x_center = sum(p[0] for p in poly) / len(poly)
        if y_center <= img_h * 0.15 and (x_center <= 0.20 * img_w or x_center >= 0.80 * img_w):
            positioned.append(c)
    if not positioned:
        return None
    pool = positioned

    if exp_start is not None and exp_end is not None:
        margin = 20
        in_range = []
        for c in pool:
            if c["type"] == "range":
                m = RANGE_RE.match(c["text"].replace("–", "-"))
                if m:
                    rs, re_ = int(m.group(1)), int(m.group(2))
                    if rs >= exp_start - margin and re_ <= exp_end + margin:
                        in_range.append(c)
            elif c["type"] == "single":
                try:
                    num = int(c["text"])
                    if exp_start - margin <= num <= exp_end + margin:
                        in_range.append(c)
                except ValueError:
                    pass
        if in_range:
            pool = in_range

    good = [c for c in pool if c["score"] >= MIN_SCORE]
    if not good:
        good = pool
    return max(good, key=lambda c: c["score"])


def _process_page(args: tuple) -> dict:
    """Process a single page. Runs in a worker process."""
    page_idx, pdf_path, exp_start, exp_end, model_path = args

    if not hasattr(_process_page, "_ocr"):
        try:
            from paddleocr import PaddleOCR

            _process_page._ocr = PaddleOCR(
                text_detection_model_name="PP-OCRv5_server_det",
                text_recognition_model_name="PP-OCRv5_server_rec",
                use_textline_orientation=False,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                enable_mkldnn=False,  # x86 fix
            )
        except Exception:
            _process_page._ocr = None
    if not hasattr(_process_page, "_yolo"):
        from ultralytics import YOLO

        model_path = Path(model_path)
        if not model_path.exists():
            from huggingface_hub import hf_hub_download

            model_path.parent.mkdir(parents=True, exist_ok=True)
            downloaded = hf_hub_download(
                repo_id="flooie/blackletter-large",
                filename=model_path.name,
                local_dir=model_path.parent,
            )
            model_path = Path(downloaded)
        _process_page._yolo = YOLO(str(model_path))
    if not hasattr(_process_page, "_glm"):
        # Lazy-load GLM-OCR: don't load the model upfront — only when needed
        _process_page._glm_processor = None
        _process_page._glm_model = None
        _process_page._glm = True
    import fitz as _fitz

    if not hasattr(_process_page, "_pdf") or getattr(_process_page, "_pdf_path", None) != pdf_path:
        if hasattr(_process_page, "_pdf"):
            _process_page._pdf.close()
        _process_page._pdf = _fitz.open(pdf_path)
        _process_page._pdf_path = pdf_path

    import io

    ocr = _process_page._ocr
    yolo = _process_page._yolo
    glm_proc = _process_page._glm_processor
    glm_model = _process_page._glm_model
    pdf = _process_page._pdf

    page = pdf[page_idx]
    pix = page.get_pixmap(dpi=200)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    img_w, img_h = img.size

    def img_to_bytes(pil_img):
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return buf.getvalue()

    page_num = None

    # Step 1: YOLO detection
    det = yolo(img, conf=0.30, verbose=False)
    boxes = det[0].boxes

    pn_boxes = []
    best_header = None
    for box in boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        if cls_id == PAGE_HEADER_CLASS:
            if best_header is None or conf > best_header[4]:
                best_header = (x1, y1, x2, y2, conf)
        elif cls_id == PAGE_NUMBER_CLASS:
            if y1 < 5 or y2 > img_h - 5:
                continue
            pn_boxes.append((x1, y1, x2, y2, conf))

    # 1a) YOLO page number box
    yolo_result = None
    pn_boxes.sort(key=lambda b: b[4], reverse=True)
    for x1, y1, x2, y2, conf in pn_boxes:
        pad = 10
        crop = img.crop(
            (
                max(0, int(x1) - pad),
                max(0, int(y1) - pad),
                min(img_w, int(x2) + pad),
                min(img_h, int(y2) + pad),
            )
        )
        if crop.size[0] < 30 or crop.size[1] < 30:
            continue
        yolo_result = _ocr_crop_multi(
            ocr, glm_proc, glm_model, crop, "yolo-pn", page_idx, exp_start, exp_end
        )
        if yolo_result:
            break

    # 1b) YOLO header — crop sides
    if not yolo_result and best_header:
        hx1, hy1, hx2, hy2, _ = best_header
        pad = 10
        top = max(0, int(hy1) - pad)
        bot = min(img_h, int(hy2) + pad)
        for side_crop, sname in [
            (img.crop((0, top, max(1, int(hx1)), bot)), "hdr-L"),
            (img.crop((min(img_w - 1, int(hx2)), top, img_w, bot)), "hdr-R"),
        ]:
            if side_crop.size[0] < 10:
                continue
            yolo_result = _ocr_crop_multi(
                ocr, None, None, side_crop, sname, page_idx, exp_start, exp_end
            )
            if yolo_result:
                break

    # Check plausibility
    yolo_plausible = False
    if yolo_result and yolo_result.get("type") == "single" and exp_start is not None:
        try:
            num = int(yolo_result["text"])
            expected_num = exp_start + page_idx
            if abs(num - expected_num) <= 10:
                yolo_plausible = True
        except ValueError:
            pass
    if (
        yolo_result
        and yolo_result.get("ocr") == "paddle"
        and yolo_result.get("score", 0) >= MIN_SCORE
    ):
        yolo_plausible = True

    if yolo_result and yolo_plausible:
        page_num = yolo_result
    else:
        # Step 2: Corner crops
        crop_h = int(img_h * CROP_FRAC)
        half_w = img_w // 2
        candidates = []
        for crop, cname, x_off in [
            (img.crop((0, 0, half_w, crop_h)), "corner-L", 0),
            (img.crop((half_w, 0, img_w, crop_h)), "corner-R", half_w),
        ]:
            hits, _ = _scan_crop(ocr, img_to_bytes(crop), cname, page_idx, x_offset=x_off)
            candidates.extend(hits)
        corner_result = _pick_best(candidates, img_w, img_h, exp_start, exp_end)

        if not corner_result or corner_result["score"] < MIN_SCORE:
            top_crop = img.crop((0, 0, img_w, crop_h))
            hits, _ = _scan_crop(ocr, img_to_bytes(top_crop), "top", page_idx)
            candidates.extend(hits)
            corner_result = _pick_best(candidates, img_w, img_h, exp_start, exp_end)

        if corner_result:
            page_num = corner_result
        elif yolo_result and yolo_result.get("score", 0) >= MIN_SCORE:
            page_num = yolo_result
        else:
            page_num = None

    # Capture all YOLO detections (not just page numbers)
    all_detections = []
    for box in det[0].boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        bx1, by1, bx2, by2 = box.xyxy[0].tolist()
        all_detections.append(
            {
                "label_id": cls_id,
                "confidence": round(conf, 3),
                "bbox": [round(bx1, 1), round(by1, 1), round(bx2, 1), round(by2, 1)],
            }
        )

    return {
        "pdf_page": page_idx + 1,
        "detected": page_num["text"] if page_num else None,
        "zone": page_num.get("zone") if page_num else None,
        "type": page_num.get("type") if page_num else None,
        "score": page_num.get("score") if page_num else None,
        "ocr": page_num.get("ocr", "paddle") if page_num else None,
        "detections": all_detections,
        "img_width": img_w,
        "img_height": img_h,
    }


def analyze_pdf(
    pdf_path: str | Path,
    exp_start: int | None = None,
    exp_end: int | None = None,
    max_pages: int = 9999,
    num_workers: int | None = None,
    model: str | Path | None = None,
    progress_callback=None,
) -> dict:
    """Analyze a PDF's page number sequence for QA/verification.

    Uses YOLO to locate page number regions, PaddleOCR to read them,
    then checks the sequence for gaps, duplicates, and coverage against
    an expected range.

    Args:
        pdf_path: Path to the PDF to analyze.
        exp_start: Expected first page number (optional; inferred from filename if omitted).
        exp_end: Expected last page number (optional).
        max_pages: Maximum number of pages to process.
        num_workers: Number of parallel workers (default: min(4, cpu_count)).
        model: Path to YOLO model. Defaults to blackletter's bundled analyze.pt.
        progress_callback: Optional callable(current, total, message) called after each page.

    Returns:
        Dict with keys:
            total_pages, results, seq_issues, duplicates, seen_nums,
            all_nums, missing_pages, ranges_found, not_detected
    """
    import fitz as _fitz

    pdf_path = str(pdf_path)
    if model is None:
        model = DEFAULT_ANALYZE_MODEL

    # Auto-download model from Hugging Face if not present
    model = Path(model)
    if not model.exists():
        try:
            from huggingface_hub import hf_hub_download

            print(f"  Downloading {model.name} from Hugging Face...", flush=True)
            downloaded = hf_hub_download(
                repo_id="flooie/blackletter-large",
                filename=model.name,
                local_dir=model.parent,
            )
            model = Path(downloaded)
        except Exception as e:
            raise FileNotFoundError(
                f"Model not found at {model} and could not be downloaded: {e}"
            ) from e

    if num_workers is None:
        num_workers = min(4, cpu_count())

    pdf = _fitz.open(pdf_path)
    total = min(max_pages, len(pdf))
    pdf.close()

    tasks = [(i, pdf_path, exp_start, exp_end, model) for i in range(total)]

    if progress_callback:
        progress_callback(0, total, "Starting OCR...")

    results = []
    if num_workers == 1:
        # Skip multiprocessing.Pool when single-threaded. fork() duplicates
        # PyTorch's C++ runtime (oneDNN/MKL), and when PaddlePaddle then
        # initializes its own oneDNN in the child, the two conflict,
        # causing segfaults or deadlocks.
        for task in tasks:
            r = _process_page(task)
            results.append(r)
            if progress_callback:
                detected = r["detected"] or "none"
                progress_callback(len(results), total, f"Page {r['pdf_page']}/{total}: {detected}")
    else:
        with Pool(num_workers) as pool:
            for r in pool.imap(_process_page, tasks):
                results.append(r)
                if progress_callback:
                    detected = r["detected"] or "none"
                    progress_callback(len(results), total, f"Page {r['pdf_page']}/{total}: {detected}")

    # Build mapping: page number → PDF page(s)
    seen_nums: dict[int, list[int]] = {}
    for r in results:
        if not r["detected"] or r["type"] == "range":
            continue
        try:
            num = int(r["detected"])
        except ValueError:
            continue
        seen_nums.setdefault(num, []).append(r["pdf_page"])

    # Sequence analysis
    prev_num = None
    prev_pdf = None
    seq_issues = []
    for r in results:
        if not r["detected"] or r["type"] == "range":
            prev_num = None
            continue
        try:
            num = int(r["detected"])
        except ValueError:
            continue
        if prev_num is not None:
            diff = num - prev_num
            if diff == 0:
                seq_issues.append(("DUPLICATE", r["pdf_page"], num, prev_pdf, prev_num))
            elif diff < 0:
                seq_issues.append(("BACKWARD", r["pdf_page"], num, prev_pdf, prev_num))
            elif diff > 2:
                gap = list(range(prev_num + 1, num))
                seq_issues.append(("GAP", r["pdf_page"], num, prev_pdf, prev_num, gap))
        prev_num = num
        prev_pdf = r["pdf_page"]

    duplicates = {k: v for k, v in seen_nums.items() if len(v) > 1}
    all_nums = sorted(seen_nums.keys())
    ranges_found = [r for r in results if r["type"] == "range"]
    missing_pages: list[int] = []

    if exp_start is not None and exp_end is not None:
        range_pages: set[int] = set()
        for r in ranges_found:
            m = RANGE_RE.match(r["detected"].replace("\u2013", "-"))
            if m:
                rs, re_ = int(m.group(1)), int(m.group(2))
                for pg in range(rs, re_ + 1):
                    range_pages.add(pg)
        expected_set = set(range(exp_start, exp_end + 1))
        actual_set = set(all_nums) | range_pages
        missing_pages = sorted(expected_set - actual_set)
    elif all_nums:
        expected = set(range(all_nums[0], all_nums[-1] + 1))
        missing_pages = sorted(expected - set(all_nums))

    return {
        "total_pages": total,
        "results": results,
        "seq_issues": seq_issues,
        "duplicates": duplicates,
        "seen_nums": seen_nums,
        "all_nums": all_nums,
        "missing_pages": missing_pages,
        "ranges_found": ranges_found,
        "not_detected": [r for r in results if not r["detected"]],
    }
