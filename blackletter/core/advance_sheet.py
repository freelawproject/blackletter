"""Advance sheet splitter module for blackletter.

Splits advance sheets (legal reporter volumes) into individual opinions
using YOLO detection and Gemini metadata extraction.
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import fitz
import numpy as np
import pdfplumber
from google import genai
from google.genai import types
from ultralytics import YOLO

logger = logging.getLogger(__name__)


# ================= CONFIG =================
@dataclass(frozen=True)
class AdvanceSheetConfig:
    """Configuration for advance sheet processing."""

    base_dir: Path = Path(__file__).parent.parent
    model_path: Path = Path(__file__).parent.parent / "models" / "best.pt"
    input_dir: Path = Path(__file__).parent.parent / "scans"
    output_dir: Path = Path(__file__).parent.parent / "output"
    prompt_path: Path = Path(__file__).parent.parent / "prompts" / "advance_sheet.txt"

    dpi: int = 200
    conf_default: float = 0.25

    toc_min_conf: float = 0.80
    header_min_conf: float = 0.15

    # How many consecutive pages can miss a detection and still be part of the run?
    max_missing_gap: int = 1

    gemini_model: str = "gemini-2.5-flash"


def build_config(
    base_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> AdvanceSheetConfig:
    """Build advance sheet config.

    :param base_dir: base directory (default: parent of this file)
    :param output_dir: output directory for extracted PDFs (default: base_dir/output)
    :return: AdvanceSheetConfig instance
    """
    if base_dir is None:
        base_dir = Path(__file__).parent.parent

    if output_dir is None:
        output_dir = base_dir / "output"

    cfg = AdvanceSheetConfig(
        base_dir=base_dir,
        output_dir=output_dir,
    )
    return cfg


# ================= UTIL =================
def convert_page_to_cv2(page, dpi: int) -> np.ndarray:
    """Convert pdfplumber page to OpenCV format.

    :param page: pdfplumber page object
    :param dpi: DPI for rendering
    :return: BGR image array
    """
    pil_img = page.to_image(resolution=dpi).original.convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def find_longest_run(flags: List[bool], max_missing: int) -> Optional[Tuple[int, int]]:
    """Find longest inclusive run of True values, bridging gaps up to max_missing False pages.

    :param flags: list of boolean flags
    :param max_missing: maximum consecutive False values to bridge
    :return: (start_idx, end_idx) or None
    """
    hit_indices = [i for i, v in enumerate(flags) if v]
    if not hit_indices:
        return None

    max_step = max_missing + 1
    best_start = best_end = hit_indices[0]
    curr_start = curr_end = hit_indices[0]

    for p in hit_indices[1:]:
        if p - curr_end <= max_step:
            curr_end = p
        else:
            if (curr_end - curr_start) > (best_end - best_start):
                best_start, best_end = curr_start, curr_end
            curr_start = curr_end = p

    if (curr_end - curr_start) > (best_end - best_start):
        best_start, best_end = curr_start, curr_end

    return best_start, best_end


def extract_pdf_span(src_path: Path, start_idx: int, end_idx: int, out_fp: Path) -> None:
    """Extract inclusive [start_idx, end_idx] pages from src_path into out_fp.

    :param src_path: source PDF path
    :param start_idx: starting page index (inclusive)
    :param end_idx: ending page index (inclusive)
    :param out_fp: output file path
    :return: None
    """
    out_fp.parent.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(src_path)
    start_idx = max(0, start_idx)
    end_idx = min(len(doc) - 1, end_idx)

    out_doc = fitz.open()
    out_doc.insert_pdf(doc, from_page=start_idx, to_page=end_idx)
    out_doc.save(out_fp, garbage=4, deflate=True)

    out_doc.close()
    doc.close()


# ================= PHASE 0: GEMINI METADATA =================
class AdvanceSheetExtractor:
    """Extract metadata ranges from advance sheet using Gemini."""

    def __init__(self, config: AdvanceSheetConfig):
        self.config = config

    def extract_ranges(self, target_file: Path) -> List[Dict]:
        """Extract opinion ranges from first page using Gemini.

        :param target_file: path to advance sheet PDF
        :return: list of metadata dictionaries
        """
        api_key = os.environ.get("LLM_API_KEY")
        if not api_key:
            raise RuntimeError("LLM_API_KEY is missing from environment")

        if not self.config.prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {self.config.prompt_path}")

        client = genai.Client(api_key=api_key)
        system_prompt = self.config.prompt_path.read_text(encoding="utf-8")

        with pdfplumber.open(target_file) as pdf:
            page_one = pdf.pages[0]
            img = convert_page_to_cv2(page_one, dpi=self.config.dpi)
            ok, buf = cv2.imencode(".png", img)
            if not ok:
                raise RuntimeError("cv2.imencode failed")
            img_bytes = buf.tobytes()

        resp = client.models.generate_content(
            model=self.config.gemini_model,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=system_prompt),
                        types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                    ],
                )
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0,
            ),
        )

        data = json.loads(resp.text)
        if not isinstance(data, list):
            raise ValueError("Expected Gemini to return a JSON list of metadata items")
        return data


# ================= PHASE 1: SCANNING =================
class SectionScanner:
    """Detects TOC/header pages and returns the longest runs (spans)."""

    def __init__(self, config: AdvanceSheetConfig, model: YOLO):
        self.config = config
        self.model = model

    def scan(
        self, pdf_path: Path
    ) -> Tuple[List[bool], List[bool], Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        """Scan PDF for TOC and header sections.

        :param pdf_path: path to PDF to scan
        :return: (toc_flags, header_flags, toc_span, header_span)
        """
        toc_flags: List[bool] = []
        header_flags: List[bool] = []

        with pdfplumber.open(pdf_path) as pdf:
            total = len(pdf.pages)
            logger.info("Scanning %d pages for toc/header…", total)

            for i, page in enumerate(pdf.pages):
                img = convert_page_to_cv2(page, dpi=self.config.dpi)
                results = self.model(img, conf=self.config.conf_default, verbose=False)

                has_toc = False
                has_header = False

                for r in results:
                    for box in r.boxes:
                        conf = float(box.conf[0].item())
                        cls_id = int(box.cls[0].item())
                        label = self.model.names[cls_id]

                        if label == "toc" and conf >= self.config.toc_min_conf:
                            has_toc = True
                        elif label == "header" and conf >= self.config.header_min_conf:
                            has_header = True

                toc_flags.append(has_toc)
                header_flags.append(has_header)

                if (i + 1) % 10 == 0 or (i + 1) == total:
                    logger.info("  scanned %d/%d", i + 1, total)

        toc_span = find_longest_run(toc_flags, max_missing=self.config.max_missing_gap)
        header_span = find_longest_run(header_flags, max_missing=self.config.max_missing_gap)

        logger.info("toc_span=%s header_span=%s", toc_span, header_span)
        return toc_flags, header_flags, toc_span, header_span


# ================= PHASE 2: PLANNING =================
class AdvanceSheetPlanner:
    """Turns metadata + header_span into extraction jobs."""

    def __init__(self, config: AdvanceSheetConfig):
        self.config = config

    def plan_jobs(
        self,
        metadata: List[Dict],
        header_span: Optional[Tuple[int, int]],
        toc_span: Optional[Tuple[int, int]],
    ) -> List[Dict]:
        """Plan extraction jobs from metadata and detected spans.

        :param metadata: list of metadata dictionaries from Gemini
        :param header_span: detected header span (start, end)
        :param toc_span: detected TOC span (start, end)
        :return: list of extraction jobs
        """
        if not header_span:
            return []

        metadata = sorted(metadata, key=lambda x: x.get("id", 0))

        s_header, _ = header_span
        jobs: List[Dict] = []

        count = 0
        for item in metadata:
            start = int(item["pages"]["start"])
            end = int(item["pages"]["end"])
            volume = item["volume"]
            reporter = item["reporter"]

            addon = end - start
            start_page = s_header + count
            end_page = start_page + addon

            jobs.append(
                {
                    "volume": volume,
                    "reporter": str(reporter).lower(),
                    "start_page_key": start,
                    "pdf_start_idx": start_page,
                    "pdf_end_idx": end_page,
                    "toc_span": toc_span,
                }
            )

            count += addon
            count += 1

        return jobs


# ================= PHASE 3: EXECUTION =================
class PDFExtractor:
    """Extract PDFs based on jobs."""

    def __init__(self, config: AdvanceSheetConfig):
        self.config = config

    def execute(self, src_pdf: Path, jobs: List[Dict]) -> List[Dict]:
        """Execute extraction jobs and return list of output file paths.

        :param src_pdf: source PDF path
        :param jobs: list of extraction jobs
        :return: list of result dictionaries with file paths
        """
        results = []

        for job in jobs:
            volume = job["volume"]
            reporter = job["reporter"]
            start_key = job["start_page_key"]

            out_dir = self.config.output_dir / reporter / str(volume) / str(start_key)
            out_dir.mkdir(parents=True, exist_ok=True)

            opinion_fp = out_dir / "opinion.pdf"
            extract_pdf_span(src_pdf, job["pdf_start_idx"], job["pdf_end_idx"], opinion_fp)
            logger.info("Saved opinion: %s", opinion_fp)

            job_result = {
                "volume": volume,
                "reporter": reporter,
                "start_page": start_key,
                "opinion_pdf": opinion_fp,
                "toc_pdf": None,
            }

            toc_span = job.get("toc_span")
            if toc_span:
                toc_fp = out_dir / "toc.pdf"
                extract_pdf_span(src_pdf, toc_span[0], toc_span[1], toc_fp)
                logger.info("Saved toc: %s", toc_fp)
                job_result["toc_pdf"] = toc_fp

            results.append(job_result)

        return results


# ================= MAIN =================
def scan_splitter(
    target_file: Path,
    model: YOLO,
    output_dir: Path | str,
    base_dir: Optional[Path] = None,
    metadata: Optional[List[Dict]] = None,
) -> List[Dict]:
    """Execute complete advance sheet splitter pipeline.

    :param target_file: path to advance sheet PDF
    :param model: loaded YOLO model instance
    :param output_dir: output directory for extracted PDFs
    :param base_dir: base directory for config (default: parent of this file)

    :return: list of extracted file results
    """
    if type(output_dir) == str:
        output_dir = Path(output_dir)
    cfg = build_config(base_dir=base_dir, output_dir=output_dir)

    if not cfg.model_path.exists():
        raise FileNotFoundError(f"YOLO model not found: {cfg.model_path}")

    if not target_file.exists():
        raise FileNotFoundError(f"Input PDF not found: {target_file}")

    # this is optional i guess.  but we probably want it

    # Phase 0: Extract metadata
    extractor = AdvanceSheetExtractor(cfg)
    if metadata == None:
        metadata = extractor.extract_ranges(target_file)

    logger.info("Gemini returned %d metadata items", len(metadata))

    # Phase 1: Scan for sections
    scanner = SectionScanner(cfg, model)
    _, _, toc_span, header_span = scanner.scan(target_file)

    if not header_span:
        logger.warning("No OPINION section found (header_span missing).")
        return []

    # Phase 2: Plan jobs
    planner = AdvanceSheetPlanner(cfg)
    jobs = planner.plan_jobs(metadata=metadata, header_span=header_span, toc_span=toc_span)
    logger.info("Planned %d extraction jobs", len(jobs))

    # Phase 3: Execute extraction
    pdfx = PDFExtractor(cfg)
    results = pdfx.execute(target_file, jobs)

    logger.info("Done. Extracted %d files", len(results))
    return results
