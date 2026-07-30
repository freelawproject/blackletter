"""Redaction geometry on pages whose words are missing or unreliable.

Two failure modes, both fixed by measuring ink instead of words:

- A PDF that never went through :func:`blackletter.api.ocr` has no words at
  all. ``_text_bottom`` used to report the *top* of its clip there, which
  collapsed every headnote rect to zero height and dropped it, so a bitonal
  PDF came out with no headnote redactions whatsoever.
- A PDF whose text layer came from our own OCR (``Document.ocr_applied``)
  has words in roughly, but not exactly, the right places. Trusting them
  moves a rect edge to wherever the OCR thought a word was.
"""

from __future__ import annotations

import fitz
import pytest

from blackletter.scanner import (
    _clip_headnote_rect,
    _text_bottom,
    _text_x_bounds,
    _tighten_to_text,
)
from tests.pdf_fixtures import (
    CONTENT,
    FONT_SIZE,
    LINE_LEADING,
    PAGE_H,
    PAGE_W,
    _line_for_width,
    write_bitonal_page,
    write_text_page,
)

# Rows the visible body lines occupy on the mixed page below.
INK_TOP, INK_BOTTOM = 200.0, 300.0

# Baseline of that page's invisible text, far below the visible lines. It is
# in the text layer and nowhere in the pixels, standing in for an OCR word
# dropped in the wrong place.
GHOST_Y = 600.0


def _mixed_page(path):
    """Write a page whose word positions disagree with its marks.

    Visible body lines between :data:`INK_TOP` and :data:`INK_BOTTOM`, plus
    invisible text at :data:`GHOST_Y` that only the text layer knows about.

    :param path: Where to write the PDF.
    """
    line = _line_for_width(CONTENT.width)
    with fitz.open() as doc:
        page = doc.new_page(width=PAGE_W, height=PAGE_H)
        y = INK_TOP
        while y <= INK_BOTTOM:
            page.insert_text((CONTENT.x0, y), line, fontsize=FONT_SIZE)
            y += LINE_LEADING
        page.insert_text((CONTENT.x0, GHOST_Y), "ghost words", fontsize=FONT_SIZE, render_mode=3)
        doc.save(str(path))


@pytest.fixture
def bitonal(tmp_path):
    """A text-less page: a running head, a body block, a platen band."""
    path = tmp_path / "bitonal.pdf"
    write_bitonal_page(path, header_line=True, bottom_bar=True)
    return path


@pytest.fixture
def mixed(tmp_path):
    """A page whose words sit far below its marks."""
    path = tmp_path / "mixed.pdf"
    _mixed_page(path)
    return path


class TestTextBottomWithoutWords:
    """``_text_bottom`` on a PDF with no text layer."""

    def test_returns_the_last_line_not_the_clip_top(self, bitonal):
        """The bug: a rect clamped to ``clip.y0`` has no height and is lost."""
        clip = fitz.Rect(CONTENT.x0, 150, CONTENT.x1, PAGE_H)
        with fitz.open(str(bitonal)) as doc:
            bottom = _text_bottom(doc[0], clip)
        assert bottom > clip.y0
        assert bottom == pytest.approx(CONTENT.y1, abs=6.0)

    def test_blank_region_still_reports_the_clip_top(self, bitonal):
        """A region that really is blank must still collapse its rect.

        Otherwise the rect survives and paints a black box over white space.
        The gap between the running head and the body is blank but sits
        inside the page's content box, so the ink can say so.
        """
        clip = fitz.Rect(CONTENT.x0, 64, CONTENT.x1, 96)
        with fitz.open(str(bitonal)) as doc:
            assert _text_bottom(doc[0], clip) == clip.y0

    def test_unmeasurable_region_clamps_nothing(self, bitonal):
        """Outside the content box the ink knows nothing, so do not guess."""
        clip = fitz.Rect(0, PAGE_H - 20, 30, PAGE_H)
        with fitz.open(str(bitonal)) as doc:
            assert _text_bottom(doc[0], clip) == clip.y1


class TestClipHeadnoteRectWithoutWords:
    """``_clip_headnote_rect`` on a PDF with no text layer."""

    def test_rect_survives_and_hugs_the_text(self, bitonal):
        rect = fitz.Rect(CONTENT.x0, 150, CONTENT.x1, PAGE_H - 20)
        with fitz.open(str(bitonal)) as doc:
            clipped = _clip_headnote_rect(
                doc[0], rect, header_bottom=100.0, footer_top=PAGE_H, ocr_applied=False
            )
        assert clipped is not None, "headnote rect dropped on a text-less page"
        assert clipped.height > 0
        # Stops at the last line rather than running onto the platen band.
        assert clipped.y1 == pytest.approx(CONTENT.y1, abs=8.0)

    def test_x_bounds_are_left_alone(self, bitonal):
        """Narrowing a headnote rect horizontally is the unsafe direction."""
        rect = fitz.Rect(CONTENT.x0, 150, CONTENT.x1, 400)
        with fitz.open(str(bitonal)) as doc:
            clipped = _clip_headnote_rect(
                doc[0], rect, header_bottom=100.0, footer_top=PAGE_H, ocr_applied=False
            )
        assert clipped.x0 <= rect.x0 + 3
        assert clipped.x1 >= rect.x1 - 3


class TestOcrAppliedPrefersInk:
    """``ocr_applied`` selects ink over the word boxes it distrusts."""

    def test_text_bottom_follows_the_words_by_default(self, mixed):
        clip = fitz.Rect(CONTENT.x0, 150, CONTENT.x1, PAGE_H - 50)
        with fitz.open(str(mixed)) as doc:
            assert _text_bottom(doc[0], clip) == pytest.approx(GHOST_Y, abs=6.0)

    def test_text_bottom_follows_the_ink_when_ocr_applied(self, mixed):
        clip = fitz.Rect(CONTENT.x0, 150, CONTENT.x1, PAGE_H - 50)
        with fitz.open(str(mixed)) as doc:
            bottom = _text_bottom(doc[0], clip, ocr_applied=True)
        assert bottom == pytest.approx(INK_BOTTOM, abs=6.0)

    def test_tighten_follows_the_ink_when_ocr_applied(self, mixed):
        rect = fitz.Rect(CONTENT.x0 - 20, 150, CONTENT.x1 + 20, PAGE_H - 50)
        with fitz.open(str(mixed)) as doc:
            tight = _tighten_to_text(doc[0], rect, ocr_applied=True)
        assert tight is not None
        assert tight.y1 == pytest.approx(INK_BOTTOM, abs=8.0)
        assert tight.y0 == pytest.approx(INK_TOP - FONT_SIZE, abs=10.0)

    def test_headnote_rect_stops_at_the_ink(self, mixed):
        """The rect no longer stretches down to a stray OCR word."""
        rect = fitz.Rect(CONTENT.x0, 150, CONTENT.x1, PAGE_H - 50)
        with fitz.open(str(mixed)) as doc:
            page = doc[0]
            with_words = _clip_headnote_rect(
                page, rect, header_bottom=100.0, footer_top=PAGE_H, ocr_applied=False
            )
        with fitz.open(str(mixed)) as doc:
            with_ink = _clip_headnote_rect(
                doc[0], rect, header_bottom=100.0, footer_top=PAGE_H, ocr_applied=True
            )
        assert with_words.y1 > GHOST_Y - 6
        assert with_ink.y1 < INK_BOTTOM + 8


class TestTextPagesAreUnchanged:
    """A trustworthy text layer is still measured from its words."""

    def test_bounds_come_from_the_text_layer(self, tmp_path):
        pdf = tmp_path / "text.pdf"
        write_text_page(pdf)
        clip = fitz.Rect(0, 0, PAGE_W, PAGE_H)
        with fitz.open(str(pdf)) as doc:
            page = doc[0]
            words = page.get_text("words")
            assert words
            bottom = _text_bottom(page, clip)
            left, right = _text_x_bounds(page, clip)
        assert bottom == pytest.approx(max(w[3] for w in words), abs=0.01)
        assert left == pytest.approx(min(w[0] for w in words) - 2.0, abs=0.01)
        assert right == pytest.approx(max(w[2] for w in words) + 2.0, abs=0.01)
