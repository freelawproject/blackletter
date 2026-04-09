"""Tests for _tighten_to_text, _text_bottom, and _text_x_bounds.

These tests pin down the current behaviour so it stays invariant when
we introduce per-page word caching.
"""

import fitz

from blackletter.scanner import _tighten_to_text, _text_bottom, _text_x_bounds


# ── Helpers ─────────────────────────────────────────────────────────────


def _make_page(texts: list[tuple[float, float, str]]) -> fitz.Page:
    """Create a single-page PDF with text inserted at (x, y) positions.

    Returns the first (and only) page of the document.
    """
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    for x, y, text in texts:
        page.insert_text((x, y), text, fontsize=12)
    return page


# ── _tighten_to_text ───────────────────────────────────────────────────


class TestTightenToText:
    def test_shrinks_rect_to_text_bounds(self):
        page = _make_page([(100, 200, "Hello World")])
        # Use a large rect that fully encloses the text
        big_rect = fitz.Rect(50, 150, 400, 300)
        result = _tighten_to_text(page, big_rect, padding=2.0)

        assert result is not None
        # Tightened rect should be smaller than the input
        assert result.x0 > big_rect.x0
        assert result.y0 > big_rect.y0
        assert result.x1 < big_rect.x1
        assert result.y1 < big_rect.y1

    def test_returns_none_when_no_text_in_rect(self):
        page = _make_page([(100, 200, "Hello")])
        # Rect that doesn't overlap the text
        empty_rect = fitz.Rect(400, 400, 600, 600)
        result = _tighten_to_text(page, empty_rect)

        assert result is None

    def test_returns_none_when_skip_is_true(self):
        page = _make_page([(100, 200, "Hello")])
        rect = fitz.Rect(50, 150, 400, 300)
        result = _tighten_to_text(page, rect, skip=True)

        assert result is None

    def test_padding_is_applied(self):
        page = _make_page([(100, 200, "Hello")])
        rect = fitz.Rect(50, 150, 400, 300)

        tight = _tighten_to_text(page, rect, padding=0.0)
        padded = _tighten_to_text(page, rect, padding=5.0)

        assert tight is not None
        assert padded is not None
        assert padded.x0 == tight.x0 - 5.0
        assert padded.y0 == tight.y0 - 5.0
        assert padded.x1 == tight.x1 + 5.0
        assert padded.y1 == tight.y1 + 5.0

    def test_multiple_calls_same_page_same_result(self):
        """Calling _tighten_to_text multiple times on the same page
        should return identical results (invariant for caching)."""
        page = _make_page([(100, 200, "Hello"), (100, 400, "World")])
        rect = fitz.Rect(50, 150, 400, 500)

        r1 = _tighten_to_text(page, rect)
        r2 = _tighten_to_text(page, rect)

        assert r1 is not None
        assert r1 == r2

    def test_different_clips_isolate_text(self):
        """Two non-overlapping rects should tighten to their own text."""
        page = _make_page([(100, 100, "Top"), (100, 700, "Bottom")])
        top_rect = fitz.Rect(50, 50, 400, 200)
        bot_rect = fitz.Rect(50, 600, 400, 780)

        top_result = _tighten_to_text(page, top_rect)
        bot_result = _tighten_to_text(page, bot_rect)

        assert top_result is not None
        assert bot_result is not None
        # They should not overlap
        assert top_result.y1 < bot_result.y0

    def test_empty_page(self):
        page = _make_page([])
        rect = fitz.Rect(50, 50, 400, 400)
        result = _tighten_to_text(page, rect)

        assert result is None


# ── _text_bottom ───────────────────────────────────────────────────────


class TestTextBottom:
    def test_returns_bottom_of_last_text(self):
        page = _make_page([(100, 200, "Line one"), (100, 400, "Line two")])
        clip = fitz.Rect(50, 50, 400, 500)
        bottom = _text_bottom(page, clip)

        # Should be at or below the second line's y position
        assert bottom > 350

    def test_returns_clip_y0_when_no_text(self):
        page = _make_page([(100, 200, "Hello")])
        clip = fitz.Rect(400, 400, 600, 600)  # no text here
        bottom = _text_bottom(page, clip)

        assert bottom == clip.y0

    def test_clip_restricts_to_region(self):
        page = _make_page([(100, 100, "Top"), (100, 700, "Bottom")])
        # Clip only the top region
        clip = fitz.Rect(50, 50, 400, 200)
        bottom = _text_bottom(page, clip)

        # Should reflect only the top text, not the bottom one
        assert bottom < 300

    def test_empty_page(self):
        page = _make_page([])
        clip = fitz.Rect(50, 50, 400, 400)
        bottom = _text_bottom(page, clip)

        assert bottom == clip.y0


# ── _text_x_bounds ─────────────────────────────────────────────────────


class TestTextXBounds:
    def test_returns_text_extent(self):
        page = _make_page([(100, 200, "Hello")])
        clip = fitz.Rect(50, 150, 400, 300)
        left, right = _text_x_bounds(page, clip, padding=2.0)

        # Bounds should be tighter than the clip
        assert left > clip.x0
        assert right < clip.x1

    def test_returns_clip_when_no_text(self):
        page = _make_page([(100, 200, "Hello")])
        clip = fitz.Rect(400, 400, 600, 600)
        left, right = _text_x_bounds(page, clip)

        assert left == clip.x0
        assert right == clip.x1

    def test_padding_is_applied(self):
        page = _make_page([(100, 200, "Hello")])
        clip = fitz.Rect(50, 150, 400, 300)

        l0, r0 = _text_x_bounds(page, clip, padding=0.0)
        l5, r5 = _text_x_bounds(page, clip, padding=5.0)

        assert l5 == l0 - 5.0
        assert r5 == r0 + 5.0

    def test_empty_page(self):
        page = _make_page([])
        clip = fitz.Rect(50, 50, 400, 400)
        left, right = _text_x_bounds(page, clip)

        assert left == clip.x0
        assert right == clip.x1
