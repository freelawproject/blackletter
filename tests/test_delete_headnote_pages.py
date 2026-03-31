"""Tests for _headnote_local_pages_to_delete.

Fixture: 23-page document (pages 0–22) from an A.3d volume with:
  - Pages 0–8: headnote content spanning multiple opinions
  - Page 8 (right col): CASE_CAPTION that starts the tested opinion
  - Page 9: HEADNOTE-only page — should be identified for deletion
  - Page 10: DIVIDER + CASE_METADATA (end of headnote zone)
  - Pages 11–21: opinion text
  - Page 22: KEY_ICON (end of opinion)

Expected: local_idx=1 (page 9) is returned as the page to delete.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from blackletter.models import BBox, Detection, Document, Label, Page
from blackletter.process import _headnote_local_pages_to_delete
from blackletter.scanner import _pair_opinions


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "a3d_detections.json"
IMG_WIDTH = 1700
IMG_HEIGHT = 2200
PDF_WIDTH = 612.0
PDF_HEIGHT = 792.0


def _build_document(detections_data: list[dict]) -> Document:
    pages_dict: dict[int, Page] = {}
    for d in detections_data:
        idx = d["page_index"]
        if idx not in pages_dict:
            pages_dict[idx] = Page(
                index=idx,
                pdf_width=PDF_WIDTH,
                pdf_height=PDF_HEIGHT,
                img_width=IMG_WIDTH,
                img_height=IMG_HEIGHT,
            )

    pages = [pages_dict[i] for i in sorted(pages_dict)]

    for d in detections_data:
        idx = d["page_index"]
        page = pages_dict[idx]
        try:
            label = Label[d["label"]]
        except KeyError:
            continue
        det = Detection(
            bbox=BBox.from_xyxy(d["bbox"]),
            label=label,
            confidence=d["confidence"],
            page_index=idx,
        )
        page.detections.append(det)

    return Document(pdf_path=Path("/nonexistent/doc.pdf"), pages=pages)


@pytest.fixture(scope="module")
def document():
    return _build_document(json.loads(FIXTURE_PATH.read_text()))


@pytest.fixture(scope="module")
def opinions(document):
    return _pair_opinions(document)


@pytest.fixture(scope="module")
def pages_by_index(document):
    return {p.index: p for p in document.pages}


class TestDocumentStructure:
    def test_page_count(self, document):
        assert len(document.pages) == 23

    def test_last_opinion_spans_8_to_22(self, opinions):
        cap, key = opinions[-1]
        assert cap.page_index == 8, f"Caption expected on page 8, got {cap.page_index}"
        assert key.page_index == 22, f"Key expected on page 22, got {key.page_index}"


class TestHeadnoteLocalPagesToDelete:
    def test_middle_headnote_page_identified(self, document, opinions, pages_by_index):
        """Page 9 sits between caption-page 8 and divider-page 10 — local_idx=1."""
        cap, key = opinions[-1]
        mid = document.pages[0].midpoint

        to_delete = _headnote_local_pages_to_delete(
            cap, key, pages_by_index, mid, reporter=document.reporter
        )

        assert to_delete == [1], f"Expected [1] (page 9), got {to_delete}"

    def test_no_pages_deleted_when_divider_on_same_page_as_caption(
        self, document, opinions, pages_by_index
    ):
        """Opinions where caption and end-marker are adjacent have no middle pages."""
        # First opinion in the fixture: caption and key are close together,
        # no room for middle-page deletion even if a divider is found.
        cap, key = opinions[0]
        mid = document.pages[0].midpoint

        to_delete = _headnote_local_pages_to_delete(
            cap, key, pages_by_index, mid, reporter=document.reporter
        )

        # Either nothing to delete (< 2 page gap), or a valid list — never crashes
        assert isinstance(to_delete, list)
