"""Adapter for the bl-warm replacement model.

bl-warm is a single YOLOv8m checkpoint (warm-started from ``large.pt``,
trained on hand-reviewed unredacted golden pages) intended to replace
the small/medium/large trio. It emits 18 classes under its own names
and ids, so this module translates its raw output into the
:class:`~blackletter.models.Label` taxonomy at detection-build time:

- name-based id mapping (bl-warm ids do not line up with ``Label`` ids)
- ``keycite`` -> ``HEADNOTE``: bl-warm boxes the West key-number token
  rather than the whole headnote line; consumers use its center/column
  membership, which the token preserves
- ``body`` -> two ``TEXT_COLUMN`` boxes split at the body box's center
  line, since bl-warm draws one box around the whole text body while
  blackletter expects one box per column (``snap_text_columns_to_ink``
  then corrects the synthetic edges against the page ink)
- ``heading`` and ``blockquote`` have no ``Label`` equivalent and are
  dropped

Classes blackletter's models emit that bl-warm does not: the
consumer-less ``JUDGES``/``DOCKET``/``DATE``/``COURT``/``CITATION``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from blackletter.models import Label

_NAME_TO_LABEL: dict[str, Label] = {
    "key_icon": Label.KEY_ICON,
    "divider": Label.DIVIDER,
    "page_header": Label.PAGE_HEADER,
    "caption": Label.CASE_CAPTION,
    "footnote_block": Label.FOOTNOTES,
    "headnote_bracket": Label.HEADNOTE_BRACKET,
    "case_metadata": Label.CASE_METADATA,
    "case_sequence": Label.CASE_SEQUENCE,
    "page_number": Label.PAGE_NUMBER,
    "state_abbreviation": Label.STATE_ABBREVIATION,
    "keycite": Label.HEADNOTE,
    "background": Label.BACKGROUND,
    "syllabus": Label.SYLLABUS,
    "editorial": Label.EDITORIAL,
    "body": Label.TEXT_COLUMN,
    # 18-class model (2026-08-10): image feeds _extract_images /
    # has_image / _stamp_original_images; heading and blockquote have
    # no Label consumer and fall through the None filter below
    "image": Label.IMAGE,
}


def is_bl_warm(names: dict[int, str]) -> bool:
    """True when a loaded model's ``names`` dict is bl-warm's class set.

    :param names: The ``names`` mapping from an ultralytics model/result.
    :returns: Whether detections need translating through this adapter.
    """
    vals = set(names.values())
    return "keycite" in vals and "body" in vals


def iter_label_rows(result: Any) -> Iterator[tuple[int, float, list[float]]]:
    """Yield ``(label_id, confidence, xyxy)`` rows in blackletter's
    taxonomy from one ultralytics result.

    For the original models this is a passthrough of the raw boxes.
    For bl-warm, names are mapped through :data:`_NAME_TO_LABEL`
    (classes without a Label are dropped) and each ``body`` box
    becomes two ``TEXT_COLUMN`` rows split at the body's vertical
    center line. Duplicate overlapping detections are left to
    :func:`blackletter.api.detect`'s same-label merge, shared with
    the original models.

    :param result: One ultralytics ``Results`` object.
    :returns: Iterator of ``(int(Label), confidence, [x1, y1, x2, y2])``.
    """
    names = getattr(result, "names", None)
    if not names or not is_bl_warm(names):
        for box in result.boxes:
            yield (int(box.cls[0].item()), float(box.conf[0].item()), box.xyxy[0].tolist())
        return
    for box in result.boxes:
        label = _NAME_TO_LABEL.get(names[int(box.cls[0].item())])
        if label is None:
            continue
        conf = float(box.conf[0].item())
        xyxy = box.xyxy[0].tolist()
        if label is Label.TEXT_COLUMN:
            cx = (xyxy[0] + xyxy[2]) / 2
            yield int(label), conf, [xyxy[0], xyxy[1], cx, xyxy[3]]
            yield int(label), conf, [cx, xyxy[1], xyxy[2], xyxy[3]]
        else:
            yield int(label), conf, xyxy
