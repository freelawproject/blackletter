"""Tests for the bl-warm adapter in :mod:`blackletter.bl_warm`.

bl-warm names its classes differently from the ``Label`` taxonomy and
emits two classes, ``heading`` and ``blockquote``, that the legacy trio
never had. The adapter used to drop those silently; these tests pin the
translation of every bl-warm class so a missing mapping fails loudly.
"""

from __future__ import annotations

from blackletter.bl_warm import _NAME_TO_LABEL, is_bl_warm, iter_label_rows
from blackletter.models import Label

# The 18 class names bl-warm emits, as ordered in its ``names`` dict.
BL_WARM_NAMES: dict[int, str] = {
    0: "key_icon",
    1: "divider",
    2: "page_header",
    3: "caption",
    4: "footnote_block",
    5: "headnote_bracket",
    6: "case_metadata",
    7: "case_sequence",
    8: "page_number",
    9: "state_abbreviation",
    10: "keycite",
    11: "background",
    12: "syllabus",
    13: "editorial",
    14: "body",
    15: "image",
    16: "heading",
    17: "blockquote",
}


class _Scalar:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


class _List:
    def __init__(self, values):
        self._values = list(values)

    def tolist(self):
        return self._values


class _Box:
    """One YOLO box, quacking like ultralytics' result rows."""

    def __init__(self, cls_id: int, bbox, conf: float = 0.9):
        self.cls = [_Scalar(cls_id)]
        self.conf = [_Scalar(conf)]
        self.xyxy = [_List(bbox)]


class _Result:
    def __init__(self, boxes, names=BL_WARM_NAMES):
        self.boxes = boxes
        self.names = names


def _id_of(name: str) -> int:
    return next(k for k, v in BL_WARM_NAMES.items() if v == name)


class TestNameMapping:
    def test_every_bl_warm_class_has_a_label(self):
        missing = set(BL_WARM_NAMES.values()) - set(_NAME_TO_LABEL)
        assert not missing, f"bl-warm classes with no Label mapping: {sorted(missing)}"

    def test_mapping_has_no_stale_names(self):
        stale = set(_NAME_TO_LABEL) - set(BL_WARM_NAMES.values())
        assert not stale, f"mapped names bl-warm does not emit: {sorted(stale)}"

    def test_heading_and_blockquote_map_to_their_own_labels(self):
        assert _NAME_TO_LABEL["heading"] is Label.HEADING
        assert _NAME_TO_LABEL["blockquote"] is Label.BLOCKQUOTE

    def test_heading_and_blockquote_are_opinion_text(self):
        """Neither is West's work product, and neither bounds an opinion."""
        for label in (Label.HEADING, Label.BLOCKQUOTE):
            assert not label.is_copyrighted
            assert not label.is_structural


class TestIterLabelRows:
    def test_is_bl_warm_recognises_the_class_set(self):
        assert is_bl_warm(BL_WARM_NAMES)
        assert not is_bl_warm({0: "Key Icon", 1: "Divider"})

    def test_heading_and_blockquote_are_no_longer_dropped(self):
        result = _Result(
            [
                _Box(_id_of("heading"), [10, 20, 110, 40], conf=0.8),
                _Box(_id_of("blockquote"), [10, 60, 110, 160], conf=0.7),
            ]
        )
        rows = list(iter_label_rows(result))
        assert rows == [
            (int(Label.HEADING), 0.8, [10, 20, 110, 40]),
            (int(Label.BLOCKQUOTE), 0.7, [10, 60, 110, 160]),
        ]

    def test_every_class_yields_at_least_one_row(self):
        boxes = [_Box(cls_id, [0, 0, 100, 100]) for cls_id in BL_WARM_NAMES]
        rows = list(iter_label_rows(_Result(boxes)))
        # 17 one-to-one classes plus body, which splits into two columns.
        assert len(rows) == len(BL_WARM_NAMES) + 1
        assert {Label(label_id) for label_id, _, _ in rows} >= {
            Label.HEADING,
            Label.BLOCKQUOTE,
            Label.TEXT_COLUMN,
            Label.HEADNOTE,
        }

    def test_unknown_future_class_is_still_dropped(self):
        names = {**BL_WARM_NAMES, 18: "sidebar"}
        result = _Result([_Box(18, [0, 0, 10, 10])], names=names)
        assert list(iter_label_rows(result)) == []

    def test_body_splits_at_center(self):
        result = _Result([_Box(_id_of("body"), [100, 0, 300, 500], conf=0.95)])
        rows = list(iter_label_rows(result))
        assert rows == [
            (int(Label.TEXT_COLUMN), 0.95, [100, 0, 200.0, 500]),
            (int(Label.TEXT_COLUMN), 0.95, [200.0, 0, 300, 500]),
        ]

    def test_legacy_model_is_a_passthrough(self):
        result = _Result([_Box(int(Label.KEY_ICON), [1, 2, 3, 4], conf=0.5)], names=None)
        assert list(iter_label_rows(result)) == [(int(Label.KEY_ICON), 0.5, [1, 2, 3, 4])]
