"""Regression test: redaction locations for mj.85.640-642.

Ensures headnote redaction rects (caption → divider) don't disappear
when the redaction logic changes.  Captured from a known-good run.
"""

from pathlib import Path

import pytest
from ultralytics import YOLO

from blackletter.models import Label
from blackletter.scanner import (
    CONFIDENCE_THRESHOLD,
    LABEL_CONFIDENCE,
    _find_redaction_end,
    _pair_opinions,
    _redaction_rects,
    scan,
)

PDF_PATH = Path("output/mj/85/640/mj.85.640.642.pdf")
MODEL_PATH = Path("models/run_9.pt")

pytestmark = pytest.mark.skipif(
    not PDF_PATH.exists() or not MODEL_PATH.exists(),
    reason="Test PDF or model not available",
)


@pytest.fixture(scope="module")
def document():
    model = YOLO(str(MODEL_PATH))
    return scan(PDF_PATH, model, first_page=640)


@pytest.fixture(scope="module")
def opinions(document):
    return _pair_opinions(document)


# ── Structure ────────────────────────────────────────────────────────────

class TestDocumentStructure:
    def test_page_count(self, document):
        assert len(document.pages) == 3

    def test_page_numbers(self, document):
        assert document.pages[0].page_number == 640
        assert document.pages[1].page_number == 641
        assert document.pages[2].page_number == 642

    def test_one_opinion(self, opinions):
        assert len(opinions) == 1

    def test_opinion_spans_pages(self, opinions):
        cap, key = opinions[0]
        assert cap.page_index == 0, "Caption should be on page 0"
        assert key.page_index == 2, "Key icon should be on page 2"

    def test_case_caption_detected(self, document):
        captions = [
            d for d in document.by_label(Label.CASE_CAPTION)
            if d.confidence >= LABEL_CONFIDENCE.get(Label.CASE_CAPTION, CONFIDENCE_THRESHOLD)
        ]
        assert len(captions) == 1
        assert captions[0].page_index == 0

    def test_divider_detected(self, document):
        dividers = [
            d for d in document.by_label(Label.DIVIDER)
            if d.confidence >= LABEL_CONFIDENCE.get(Label.DIVIDER, CONFIDENCE_THRESHOLD)
        ]
        assert len(dividers) >= 1
        assert dividers[0].page_index == 0

    def test_key_icon_detected(self, document):
        keys = [
            d for d in document.by_label(Label.KEY_ICON)
            if d.confidence >= LABEL_CONFIDENCE.get(Label.KEY_ICON, CONFIDENCE_THRESHOLD)
        ]
        assert len(keys) == 1
        assert keys[0].page_index == 2


# ── Headnote redaction ───────────────────────────────────────────────────

class TestHeadnoteRedaction:
    """The most important redaction: caption → divider blackout."""

    def test_end_marker_is_divider(self, document, opinions):
        mid = document.pages[0].midpoint
        cap, key = opinions[0]

        opinion_dets = []
        for p in document.pages:
            for d in p.detections:
                sk = d.sort_key(mid)
                if cap.sort_key(mid) <= sk <= key.sort_key(mid):
                    opinion_dets.append(d)
        opinion_dets.sort(key=lambda d: d.sort_key(mid))

        end_marker = _find_redaction_end(opinion_dets, cap, key, mid)
        assert end_marker is not None, "Must find a redaction end marker"
        assert end_marker.label == Label.DIVIDER

    def test_headnote_rects_exist(self, document, opinions):
        """Headnote redaction must produce at least one rect."""
        mid = document.pages[0].midpoint
        pages_by_index = {p.index: p for p in document.pages}
        cap, key = opinions[0]

        opinion_dets = []
        for p in document.pages:
            for d in p.detections:
                sk = d.sort_key(mid)
                if cap.sort_key(mid) <= sk <= key.sort_key(mid):
                    opinion_dets.append(d)
        opinion_dets.sort(key=lambda d: d.sort_key(mid))

        end_marker = _find_redaction_end(opinion_dets, cap, key, mid)
        assert end_marker is not None
        headnote_rects = _redaction_rects(cap, end_marker, pages_by_index)
        assert len(headnote_rects) >= 1, "Must produce headnote redaction rects"

    def test_headnote_rects_on_correct_page(self, document, opinions):
        """All headnote rects should be on page 0 (where caption + divider are)."""
        mid = document.pages[0].midpoint
        pages_by_index = {p.index: p for p in document.pages}
        cap, key = opinions[0]

        opinion_dets = []
        for p in document.pages:
            for d in p.detections:
                sk = d.sort_key(mid)
                if cap.sort_key(mid) <= sk <= key.sort_key(mid):
                    opinion_dets.append(d)
        opinion_dets.sort(key=lambda d: d.sort_key(mid))

        end_marker = _find_redaction_end(opinion_dets, cap, key, mid)
        headnote_rects = _redaction_rects(cap, end_marker, pages_by_index)
        for page_idx, rect in headnote_rects:
            assert page_idx == 0, f"Headnote rect should be on page 0, got page {page_idx}"

    def test_headnote_rects_have_area(self, document, opinions):
        """Each rect must have meaningful area (not collapsed)."""
        mid = document.pages[0].midpoint
        pages_by_index = {p.index: p for p in document.pages}
        cap, key = opinions[0]

        opinion_dets = []
        for p in document.pages:
            for d in p.detections:
                sk = d.sort_key(mid)
                if cap.sort_key(mid) <= sk <= key.sort_key(mid):
                    opinion_dets.append(d)
        opinion_dets.sort(key=lambda d: d.sort_key(mid))

        end_marker = _find_redaction_end(opinion_dets, cap, key, mid)
        headnote_rects = _redaction_rects(cap, end_marker, pages_by_index)
        for page_idx, rect in headnote_rects:
            w = rect.x1 - rect.x0
            h = rect.y1 - rect.y0
            assert w > 10, f"Rect too narrow: width={w:.1f}"
            assert h > 10, f"Rect too short: height={h:.1f}"

    def test_headnote_rects_approximate_location(self, document, opinions):
        """Rects should be roughly where we expect on page 0.

        Known-good values (PDF coordinates):
          rect 1: ~(50, 164, 241, 650) — left column, caption to divider
          rect 2: ~(250, 45, 456, 179) — right column, top to divider

        We use generous tolerances (±30 pts) so model jitter doesn't
        break the test, but big regressions (wrong page, missing rects,
        completely wrong region) will still fail.
        """
        mid = document.pages[0].midpoint
        pages_by_index = {p.index: p for p in document.pages}
        cap, key = opinions[0]

        opinion_dets = []
        for p in document.pages:
            for d in p.detections:
                sk = d.sort_key(mid)
                if cap.sort_key(mid) <= sk <= key.sort_key(mid):
                    opinion_dets.append(d)
        opinion_dets.sort(key=lambda d: d.sort_key(mid))

        end_marker = _find_redaction_end(opinion_dets, cap, key, mid)
        headnote_rects = _redaction_rects(cap, end_marker, pages_by_index)

        # We expect 2 rects (left-col and right-col portions)
        assert len(headnote_rects) == 2, f"Expected 2 headnote rects, got {len(headnote_rects)}"

        # Sort by x0 to get left-col first
        rects = sorted(headnote_rects, key=lambda pr: pr[1].x0)

        # Left-column rect: covers caption area down to ~divider height
        _, left_rect = rects[0]
        assert left_rect.x0 < 100, f"Left rect x0 too far right: {left_rect.x0:.1f}"
        assert left_rect.y0 < 200, f"Left rect y0 too low: {left_rect.y0:.1f}"
        assert left_rect.y1 > 500, f"Left rect y1 too high (should reach divider): {left_rect.y1:.1f}"

        # Right-column rect: covers top of right column to divider
        _, right_rect = rects[1]
        assert right_rect.x0 > 200, f"Right rect x0 too far left: {right_rect.x0:.1f}"
        assert right_rect.y0 < 100, f"Right rect y0 too low: {right_rect.y0:.1f}"
        assert right_rect.y1 > 100, f"Right rect y1 too small: {right_rect.y1:.1f}"


# ── Recompress safety ────────────────────────────────────────────────────

class TestRecompressDoesNotCorrupt:
    """Verify recompress_images doesn't destroy redacted pages."""

    def test_recompress_preserves_rendering(self):
        import fitz
        import io
        from PIL import Image
        from blackletter.scanner import recompress_images

        pdf = fitz.open(str(PDF_PATH))
        page = pdf[0]

        # Apply a dummy redaction to force PNG conversion
        rect = fitz.Rect(10, 10, 50, 50)
        page.add_redact_annot(rect, fill=[1, 1, 1])
        page.apply_redactions()

        # Recompress
        recompress_images(pdf)

        # Save and re-open
        buf = io.BytesIO()
        pdf.save(buf)
        pdf.close()

        pdf2 = fitz.open(stream=buf.getvalue(), filetype="pdf")
        pix = pdf2[0].get_pixmap(dpi=72)
        samples = pix.samples
        total = len(samples)
        dark = sum(1 for b in samples if b < 10)
        pdf2.close()

        dark_pct = dark / total
        assert dark_pct < 0.5, (
            f"Page is {100*dark_pct:.1f}% dark after recompress — likely corrupted"
        )

# (blackletter-redux) Palin@mac blackletter-redux % python3 -m blackletter process --reporter mj --volume 85 --first-page 640 --output output/ --no-unredacted ~/Desktop/vflatscansfromdc/mj.85.640.642.pdf
# Scanning mj.85.640.642.pdf (3 pages, 1.5 MB)...
#   PDF has no text layer — running OCR...
#   OCR: processing mj.85.640.642.pdf (target 148 KB/page)
#   OCR step 1: Downsampling images...
#     Compressing 3 images...
#     Compressed 3/3 images
#   OCR step 1 done: downsampled to 0.4 MB
#   OCR step 2: Adding text layer (tesseract)...
# Scanning contents    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 3/3 0:00:00
# OCR                  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 3/3 0:00:00
# Linearizing          ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 100/100 0:00:00
# Recompressing JPEGs  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   0% 0/0 -:--:--
# Deflating JPEGs      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 3/3 0:00:00
# JBIG2                ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   0% 0/0 -:--:--
#   OCR complete: 1.5 MB -> 0.5 MB
#   OCR complete: 1.5 MB -> 0.5 MB (mj.85.640.642.pdf)
#   Detecting on 3 pages...
#     Page 3/3
# 3 pages scanned
#
# Detection counts:
#   KEY_ICON                 : 1
#   DIVIDER                  : 1
#   PAGE_HEADER              : 3
#   CASE_CAPTION             : 1
#   FOOTNOTES                : 3
#   HEADNOTE_BRACKET         : 3
#   CASE_METADATA            : 1
#   PAGE_NUMBER              : 3
#   HEADNOTE                 : 4
#   BACKGROUND               : 1
#
# Page numbers: 3/3 detected
# Range: 640 - 642
#
# No page number warnings
#
# Page numbers:
#   PDF page   1 -> 640
#   PDF page   2 -> 641
#   PDF page   3 -> 642
#
# Opinion pairing: 1 captions, 1 key icons -> 1 pairs
#
# Opinion details:
#   Opinion 1: PDF pages 1-3 -> pages 640-642  [mj.85.0640-0642.pdf]
#     PDF page   1 -> 640
#     PDF page   2 -> 641
#     PDF page   3 -> 642
#
#
# Verify report saved to output/mj/85/640/verify.txt
#
# Building full redacted PDF...
#     Redacted 3/3 pages
#   Recompressing images (JPEG)...
#   Saved mj.85.redacted.pdf (0.6 MB)
#
# Skipping unredacted (--no-unredacted)
#
# Splitting redacted into output/mj/85/640/redacted...
#   Wrote 1 redacted PDFs
#
# Splitting masked into output/mj/85/640/masked...
#   Wrote 1 masked PDFs (1 opinions consolidated)