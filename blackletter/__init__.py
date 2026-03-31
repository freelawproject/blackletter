from blackletter.process import process, generate_files
from blackletter.validate import validate
from blackletter.tasks import (
    split_page_ranges,
    bitonal_chunk,
    ocr_chunk,
    yolo_scan_chunk,
    merge_pdfs,
    merge_detections,
    pair_and_compute_rects,
)

__all__ = [
    "process",
    "generate_files",
    "validate",
    # Chunk-able task functions for celery
    "split_page_ranges",
    "bitonal_chunk",
    "ocr_chunk",
    "yolo_scan_chunk",
    "merge_pdfs",
    "merge_detections",
    "pair_and_compute_rects",
]
