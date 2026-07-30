"""Tests that ``scan()`` actually performs the steps it is supposed to.

``scan()`` has two detection paths, single-process and a worker pool for
documents of 40 pages or more, and they assemble their ``Page`` objects
independently. A step wired into one of them silently does nothing on the
other, which is how the column snap came to run only on documents too small
for any real volume. These tests drive ``scan()`` with a stub model so the
wiring is checked rather than assumed.

The parallel path loads the model from a file inside a subprocess, so it
cannot be driven from here. That is precisely why the snap belongs outside
both paths, where one test covers it either way.
"""

from __future__ import annotations

import fitz
import pytest

from blackletter.models import Label
from blackletter.scanner import scan
from tests.pdf_fixtures import COLUMN_LEFT, COLUMN_RIGHT, write_two_column_page


class _Box:
    """One YOLO box, quacking like ultralytics' result rows."""

    def __init__(self, label: Label, bbox, conf: float = 0.95):
        self.cls = [_Scalar(int(label))]
        self.conf = [_Scalar(conf)]
        self.xyxy = [_List(bbox)]


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


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


class _StubModel:
    """Returns the same two column boxes for every page it is given.

    Coordinates are in the 200 dpi image space ``scan()`` renders at, hence
    the scale factor: the fixtures are described in PDF points.
    """

    scale = 200 / 72

    def __init__(self, inset: float):
        self.inset = inset
        self.calls = 0

    def __call__(self, imgs, **kwargs):
        self.calls += len(imgs)
        results = []
        for _img in imgs:
            boxes = [
                _Box(
                    Label.TEXT_COLUMN,
                    [
                        (COLUMN_LEFT.x0 + self.inset) * self.scale,
                        COLUMN_LEFT.y0 * self.scale,
                        (COLUMN_LEFT.x1 - self.inset) * self.scale,
                        COLUMN_LEFT.y1 * self.scale,
                    ],
                ),
                _Box(
                    Label.TEXT_COLUMN,
                    [
                        (COLUMN_RIGHT.x0 + self.inset) * self.scale,
                        COLUMN_RIGHT.y0 * self.scale,
                        (COLUMN_RIGHT.x1 - self.inset) * self.scale,
                        COLUMN_RIGHT.y1 * self.scale,
                    ],
                ),
            ]
            results.append(_Result(boxes))
        return results


@pytest.fixture
def two_column_pdf(tmp_path):
    """A one-page, text-less PDF with two hard-edged text bands."""
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    path = tmp_path / "vol.pdf"
    write_two_column_page(path, tmp_dir=scratch)
    return path


@pytest.fixture(autouse=True)
def _single_process(monkeypatch):
    """Keep scan() out of the worker pool, which needs a model file."""
    import multiprocessing

    monkeypatch.setattr(multiprocessing, "cpu_count", lambda: 1)


def _columns_of(page):
    return sorted(
        (d for d in page.detections if d.label == Label.TEXT_COLUMN),
        key=lambda d: d.bbox.x1,
    )


class TestScanSnapsColumns:
    def test_column_boxes_are_corrected_against_the_ink(self, two_column_pdf, tmp_path):
        """The inset boxes the model returns should come back on the text."""
        model = _StubModel(inset=6.0)
        document = scan(two_column_pdf, model, output_dir=tmp_path / "out")

        assert model.calls == 1, "the stub model was not driven"
        (page,) = document.pages
        left, right = _columns_of(page)
        scale = page.scale_x  # image pixels to points
        assert left.bbox.x1 * scale == pytest.approx(COLUMN_LEFT.x0, abs=2.0)
        assert left.bbox.x2 * scale == pytest.approx(COLUMN_LEFT.x1, abs=2.0)
        assert right.bbox.x1 * scale == pytest.approx(COLUMN_RIGHT.x0, abs=2.0)
        assert right.bbox.x2 * scale == pytest.approx(COLUMN_RIGHT.x1, abs=2.0)

    def test_boxes_already_on_the_text_are_left_alone(self, two_column_pdf, tmp_path):
        model = _StubModel(inset=0.0)
        document = scan(two_column_pdf, model, output_dir=tmp_path / "out")
        (page,) = document.pages
        left, right = _columns_of(page)
        scale = page.scale_x
        assert left.bbox.x1 * scale == pytest.approx(COLUMN_LEFT.x0, abs=2.0)
        assert right.bbox.x2 * scale == pytest.approx(COLUMN_RIGHT.x1, abs=2.0)

    def test_no_column_ever_crosses_the_gutter(self, two_column_pdf, tmp_path):
        """The gutter here is 1.5 pt, one blank pixel at the measuring dpi."""
        model = _StubModel(inset=6.0)
        document = scan(two_column_pdf, model, output_dir=tmp_path / "out")
        (page,) = document.pages
        left, right = _columns_of(page)
        scale = page.scale_x
        assert left.bbox.x2 * scale <= COLUMN_RIGHT.x0 + 1
        assert right.bbox.x1 * scale >= COLUMN_LEFT.x1 - 1


class TestScanDoesNotOcr:
    def test_no_text_layer_is_added_by_default(self, two_column_pdf, tmp_path):
        """The pass is opt-in now: scan() must not reach for ocrmypdf."""
        model = _StubModel(inset=0.0)
        document = scan(two_column_pdf, model, output_dir=tmp_path / "out")
        with fitz.open(str(document.pdf_path)) as doc:
            assert doc[0].get_text("text").strip() == ""

    def test_ocr_is_not_attempted_even_when_asked_without_the_tool(
        self, two_column_pdf, tmp_path, monkeypatch
    ):
        """``ocr=True`` is the only route to the pre-pass, and it is not taken."""
        calls = []
        import blackletter.ocr as bl_ocr

        monkeypatch.setattr(bl_ocr, "ocr_pdf", lambda *a, **k: calls.append(a))
        model = _StubModel(inset=0.0)
        scan(two_column_pdf, model, output_dir=tmp_path / "out")
        assert calls == [], "scan() ran the OCR pre-pass without being asked"
