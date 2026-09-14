"""A DIVIDER redaction has to cover the whole printed rule.

The rect used to be tightened to the ink inside it, and ``ink.ink_bbox``
refuses ink that is too solid: a rule is 100% dark in every pixel column
it occupies, over the ``BBOX_MAX_FRACTION`` ceiling that keeps the platen
bar and the gutter shadow out of a headnote measurement. Two answers came
out of that, and only sub-pixel luck at the ends of the rule decided
which: no column survived and the raw box stayed, or the one half-covered
end column survived and became a 0.7 pt redaction over a 62 pt rule.

The guard is right for a headnote rect and wrong for a box whose content
*is* a rule, so ``DIVIDER`` joins the labels that are never tightened
rather than the ceiling moving (#75).
"""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from blackletter.models import Document, Label
from blackletter.process import compute_redaction_rects
from blackletter.scanner import _NO_TIGHTEN
from tests.pdf_fixtures import PAGE_H, PAGE_W, detected_page, detection


# The real detection from scan 2845, page_index 100: bbox [438.2, 754.5,
# 611.0, 794.5] in the 200 dpi pixels of a 1700 px page, confidence 0.876.
# 62.2 x 14.4 pt once scaled.
_SCALE = PAGE_W / 1700.0
BOX = fitz.Rect(438.2 * _SCALE, 754.5 * _SCALE, 611.0 * _SCALE, 794.5 * _SCALE)

# Two rule widths inside that box. Both under-covered before the fix, but
# by different routes: 30 pt left one half-covered end column standing and
# collapsed the redaction onto it, 40 pt left no column at all and kept
# the raw box by accident. Pinning both says the fix covers the rule and
# does not disturb the case that happened to work.
COLLAPSING_RULE = 30.0
SURVIVING_RULE = 40.0

RULE_HEIGHT = 1.4


def _write_divider_page(path: Path, rule_width: float) -> None:
    """Write a page of body text with a centered rule inside :data:`BOX`.

    The divider band itself is left clear, as it is on a printed page.
    """
    with fitz.open() as doc:
        page = doc.new_page(width=PAGE_W, height=PAGE_H)
        for i in range(50):
            y = 100 + i * 12
            if BOX.y0 - 6 < y < BOX.y1 + 6:
                continue
            page.insert_text((80, y), "the quick brown fox jumps over the lazy dog" * 2, fontsize=9)
        cx = (BOX.x0 + BOX.x1) / 2
        cy = (BOX.y0 + BOX.y1) / 2
        page.draw_rect(
            fitz.Rect(
                cx - rule_width / 2,
                cy - RULE_HEIGHT / 2,
                cx + rule_width / 2,
                cy + RULE_HEIGHT / 2,
            ),
            color=(0, 0, 0),
            fill=(0, 0, 0),
        )
        doc.save(str(path))


def _divider_rect(tmp_path: Path, rule_width: float, label: Label = Label.DIVIDER) -> dict:
    """Compute the redaction rects for one detection over that rule.

    ``opinions=[]`` leaves the headnote pre-pass with nothing to do, so
    only the per-detection loop runs, and ``skip_doctr`` keeps the
    line-level refinement (and its torch stack) out of the test.
    ``ocr_applied=True`` is what a ``Document`` rebuilt from a detections
    sidecar carries, and it is what sends the measurement to the ink.
    """
    pdf = tmp_path / "vol.pdf"
    _write_divider_page(pdf, rule_width)
    page = detected_page([detection(label, BOX.x0, BOX.y0, BOX.x1, BOX.y1, confidence=0.876)])
    document = Document(pdf_path=pdf, pages=[page], ocr_applied=True)

    pages = compute_redaction_rects(document, [], skip_doctr=True)
    assert len(pages) == 1, "the detection should produce exactly one page of rects"
    rects = pages[0]["rects"]
    assert len(rects) == 1, "the detection should produce exactly one rect"
    return rects[0]


class TestDividerCoversTheRule:
    def test_the_raw_box_survives_the_collapsing_alignment(self, tmp_path):
        """The bug: this rect was 4.72 pt wide over a 62 pt rule."""
        rect = _divider_rect(tmp_path, COLLAPSING_RULE)

        assert rect["type"] == "DIVIDER"
        assert rect["fill"] == "black"
        assert rect["x0"] == pytest.approx(BOX.x0, abs=0.1)
        assert rect["x1"] == pytest.approx(BOX.x1, abs=0.1)

    def test_the_rule_is_covered_end_to_end(self, tmp_path):
        """What the deliverable actually needs, stated without the box."""
        rect = _divider_rect(tmp_path, COLLAPSING_RULE)
        cx = (BOX.x0 + BOX.x1) / 2

        assert rect["x0"] <= cx - COLLAPSING_RULE / 2
        assert rect["x1"] >= cx + COLLAPSING_RULE / 2

    def test_the_alignment_that_already_worked_is_unchanged(self, tmp_path):
        """No column survived here, so the raw box stayed by accident."""
        rect = _divider_rect(tmp_path, SURVIVING_RULE)

        assert rect["x0"] == pytest.approx(BOX.x0, abs=0.1)
        assert rect["x1"] == pytest.approx(BOX.x1, abs=0.1)


class TestTighteningIsOnlyOffForTheseLabels:
    def test_the_divider_joined_the_untightened_labels(self):
        assert Label.DIVIDER in _NO_TIGHTEN

    def test_the_labels_that_were_already_there_stayed(self):
        assert Label.HEADNOTE_BRACKET in _NO_TIGHTEN
        assert Label.STATE_ABBREVIATION in _NO_TIGHTEN

    def test_a_text_label_over_the_same_box_is_still_tightened(self, tmp_path):
        """EDITORIAL is redacted black like a divider, but holds text."""
        rect = _divider_rect(tmp_path, COLLAPSING_RULE, label=Label.EDITORIAL)

        assert Label.EDITORIAL not in _NO_TIGHTEN
        assert rect["type"] == "EDITORIAL"
        assert rect["x1"] - rect["x0"] < BOX.width
