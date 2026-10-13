"""The pairing on the key before (``scanner._pair_on_keys_before``).

An opinion starts at a caption box and ends at a key icon, and a key
the gate pairing leaves unmatched is paired on the caption box in the
caption's place after the key before it, at any confidence; with no
box there, the opinion before runs on to it, and the miss is shown.
"""

from __future__ import annotations

from pathlib import Path

from blackletter.models import CaptionMarker, Document, Label
from blackletter.scanner import _pair_opinions, key_icons
from tests.pdf_fixtures import PAGE_W, detected_page, detection

LEFT = 100.0
RIGHT = PAGE_W / 2 + 60.0


def key(x0, y0, page_index=0, confidence=0.95):
    return detection(Label.KEY_ICON, x0, y0, x0 + 60, y0 + 20, page_index, confidence)


def caption(x0, y0, page_index=0, confidence=0.9):
    return detection(Label.CASE_CAPTION, x0, y0, x0 + 200, y0 + 80, page_index, confidence)


def synopsis(x0, y0, page_index=0):
    return detection(Label.BACKGROUND, x0, y0, x0 + 200, y0 + 40, page_index, 0.9)


def document(*pages):
    return Document(pdf_path=Path("volume.pdf"), pages=list(pages), bl_warm=True)


def spans(pairs):
    return [((c.page_index, c.bbox.y1), (k.page_index, k.bbox.y1)) for c, k in pairs]


class TestTheGatePairingIsUnchanged:
    def test_a_confident_caption_pairs_at_the_box(self):
        c1, k1 = caption(LEFT, 100), key(LEFT, 400)
        c2, k2 = caption(LEFT, 450), key(LEFT, 700)
        pairs = _pair_opinions(document(detected_page([c1, k1, c2, k2])))
        assert pairs == [(c1, k1), (c2, k2)]


class TestAKeyWithACaptionUnderTheGate:
    def test_it_pairs_on_a_marker_under_the_key_before(self):
        c1, k1 = caption(LEFT, 100), key(LEFT, 400)
        faint = caption(LEFT, 450, confidence=0.25)
        k2 = key(LEFT, 700)
        pairs = _pair_opinions(document(detected_page([c1, k1, faint, k2])))
        assert [(type(c), k) for c, k in pairs] == [(type(c1), k1), (CaptionMarker, k2)]
        marker = pairs[1][0]
        assert marker.box is faint
        # One pixel tall, right under the key before, in its column.
        assert (marker.bbox.x1, marker.bbox.x2) == (k1.bbox.x1, k1.bbox.x2)
        assert (marker.bbox.y1, marker.bbox.y2) == (k1.bbox.y2 + 1, k1.bbox.y2 + 2)
        assert marker.page_index == k1.page_index

    def test_a_caption_past_the_synopsis_is_a_box_over_the_body(self):
        c1, k1 = caption(LEFT, 100), key(LEFT, 400)
        head = synopsis(LEFT, 450)
        late = caption(LEFT, 520, confidence=0.25)
        k2 = key(LEFT, 700)
        pairs = _pair_opinions(document(detected_page([c1, k1, head, late, k2])))
        # No caption in the place: the first opinion runs on to k2.
        assert pairs == [(c1, k2)]


class TestTheColumnVouchesForAFaintCaption:
    def test_a_faint_box_below_the_column_is_no_caption(self):
        # The footnote at the foot of the left column, called a caption
        # at 0.38: outside the column box, so the opinion before runs on.
        column = detection(Label.TEXT_COLUMN, 80, 60, 300, 600)
        c1, k1 = caption(LEFT, 100), key(LEFT, 400)
        footnote = caption(LEFT, 620, confidence=0.38)
        k2 = key(LEFT, 700)
        pairs = _pair_opinions(document(detected_page([column, c1, k1, footnote, k2])))
        assert pairs == [(c1, k2)]

    def test_a_faint_box_inside_the_column_pairs(self):
        column = detection(Label.TEXT_COLUMN, 80, 60, 300, 760)
        c1, k1 = caption(LEFT, 100), key(LEFT, 400)
        faint = caption(LEFT, 450, confidence=0.38)
        k2 = key(LEFT, 700)
        pairs = _pair_opinions(document(detected_page([column, c1, k1, faint, k2])))
        assert [type(c) for c, _ in pairs] == [type(c1), CaptionMarker]
        assert pairs[1][0].box is faint

    def test_a_page_with_no_column_box_keeps_every_box(self):
        c1, k1 = caption(LEFT, 100), key(LEFT, 400)
        faint = caption(LEFT, 620, confidence=0.38)
        k2 = key(LEFT, 700)
        pairs = _pair_opinions(document(detected_page([c1, k1, faint, k2])))
        assert pairs[1][0].box is faint


class TestAKeyWithNoCaption:
    def test_the_opinion_before_runs_on_to_it(self):
        c1, k1, k2 = caption(LEFT, 100), key(LEFT, 400), key(LEFT, 700)
        pairs = _pair_opinions(document(detected_page([c1, k1, k2])))
        assert pairs == [(c1, k2)]

    def test_a_key_before_any_opinion_closes_nothing(self):
        k0 = key(LEFT, 100)
        c1, k1 = caption(LEFT, 200), key(LEFT, 500)
        pairs = _pair_opinions(document(detected_page([k0, c1, k1])))
        assert pairs == [(c1, k1)]


class TestTheFirstKey:
    def test_it_pairs_on_a_faint_caption_at_the_box(self):
        faint = caption(LEFT, 100, confidence=0.3)
        k1 = key(LEFT, 400)
        pairs = _pair_opinions(document(detected_page([faint, k1])))
        assert pairs == [(faint, k1)]

    def test_it_stays_unmatched_with_no_box(self):
        k1 = key(LEFT, 400)
        c2, k2 = caption(LEFT, 450), key(LEFT, 700)
        pairs = _pair_opinions(document(detected_page([k1, c2, k2])))
        assert pairs == [(c2, k2)]


class TestTwoBoxesOverOneIcon:
    def test_the_second_is_skipped(self):
        c1, k1 = caption(LEFT, 100), key(LEFT, 400)
        twin = key(LEFT, 405)
        c2, k2 = caption(LEFT, 450), key(LEFT, 700)
        pairs = _pair_opinions(document(detected_page([c1, k1, twin, c2, k2])))
        assert pairs == [(c1, k1), (c2, k2)]


class TestAcrossColumnsAndPages:
    def test_the_caption_in_the_facing_column_pairs_the_next_key(self):
        c1, k1 = caption(LEFT, 100), key(LEFT, 400)
        faint = caption(RIGHT, 120, confidence=0.3)
        k2 = key(RIGHT, 600)
        pairs = _pair_opinions(document(detected_page([c1, k1, faint, k2])))
        assert spans(pairs) == [((0, 100), (0, 400)), ((0, k1.bbox.y2 + 1), (0, 600))]
        assert pairs[1][0].box is faint

    def test_the_caption_on_the_next_page_pairs_the_next_key(self):
        c1, k1 = caption(LEFT, 100, 0), key(LEFT, 400, 0)
        faint = caption(LEFT, 100, 1, confidence=0.3)
        k2 = key(LEFT, 500, 1)
        pages = [detected_page([c1, k1], 0), detected_page([faint, k2], 1)]
        pairs = _pair_opinions(document(*pages))
        assert spans(pairs) == [((0, 100), (0, 400)), ((0, k1.bbox.y2 + 1), (1, 500))]


class TestKeyIcons:
    def test_the_gate_and_the_size_filter_are_one_rule(self):
        real = [key(LEFT, 300), key(LEFT, 600)]
        under_gate = key(LEFT, 100, confidence=0.1)
        sliver = detection(Label.KEY_ICON, LEFT, 2, LEFT + 45, 9, 0, 0.5)
        other = caption(LEFT, 150)
        assert key_icons(real + [under_gate, sliver, other], bl_warm=True) == real
