"""Blackletter package.

Public API is exposed lazily via PEP 562 ``__getattr__`` so that
``import blackletter`` (or ``import blackletter.models``) does not
transitively load ``blackletter.process`` and pull in
``ultralytics``/``torch``. Consumers that don't need YOLO (the
scanning daemon's idle loop, or any code path that only touches
``blackletter.api`` / ``blackletter.models``) avoid hundreds of MB of
GPU library state until something actually calls into a YOLO-using
path.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from blackletter.process import generate_files, process
    from blackletter.tasks import (
        bitonal_chunk,
        merge_detections,
        merge_pdfs,
        ocr_chunk,
        pair_and_compute_rects,
        split_page_ranges,
        yolo_scan_chunk,
    )
    from blackletter.validate import validate

__all__ = [
    "process",
    "generate_files",
    "validate",
    "split_page_ranges",
    "bitonal_chunk",
    "ocr_chunk",
    "yolo_scan_chunk",
    "merge_pdfs",
    "merge_detections",
    "pair_and_compute_rects",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "process": ("blackletter.process", "process"),
    "generate_files": ("blackletter.process", "generate_files"),
    "validate": ("blackletter.validate", "validate"),
    "split_page_ranges": ("blackletter.tasks", "split_page_ranges"),
    "bitonal_chunk": ("blackletter.tasks", "bitonal_chunk"),
    "ocr_chunk": ("blackletter.tasks", "ocr_chunk"),
    "yolo_scan_chunk": ("blackletter.tasks", "yolo_scan_chunk"),
    "merge_pdfs": ("blackletter.tasks", "merge_pdfs"),
    "merge_detections": ("blackletter.tasks", "merge_detections"),
    "pair_and_compute_rects": ("blackletter.tasks", "pair_and_compute_rects"),
}


def __getattr__(name: str) -> Any:
    """Resolve public exports on first access.

    :param name: Attribute being accessed on the package.
    :returns: The resolved object, cached in ``globals()`` for
        subsequent accesses.
    :rtype: Any
    :raises AttributeError: When ``name`` isn't part of the public API.
    """
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'blackletter' has no attribute {name!r}")
    module_name, attr = target
    module = importlib.import_module(module_name)
    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose lazy exports to ``dir(blackletter)`` and tooling.

    :returns: Sorted list of public names plus already-resolved globals.
    :rtype: list[str]
    """
    return sorted(set(__all__) | set(globals().keys()))
