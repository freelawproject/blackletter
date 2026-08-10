"""Tests for bitonal conversion, sequential and parallel.

The parallel path splits the document into one contiguous page range per
worker, converts each in its own process, and merges the chunks back. The
merge has to restore page order, which is *not* the order the workers
finish in, so most of what is worth asserting here is that ``workers=N``
is indistinguishable from ``workers=1`` no matter how the ranges land.

The pages carry a marker whose vertical position encodes the page index,
so a reordered merge is detectable from the raster alone.
"""

from __future__ import annotations

from pathlib import Path

import fitz
import numpy as np
import pytest

from blackletter.api import bitonal
from blackletter.ocr import _icc_disabled, run_bitonal

PAGE_W, PAGE_H = 612.0, 792.0

# Marker band, in PDF points. Page i puts it at MARKER_Y0 + i * MARKER_STEP,
# far enough apart that the raster row it lands on identifies the page.
MARKER_X0, MARKER_X1 = 100.0, 500.0
MARKER_Y0 = 80.0
MARKER_STEP = 40.0
MARKER_H = 20.0


def write_numbered_pages(path: Path, pages: int) -> None:
    """Write a PDF whose pages each carry a marker at a distinct height."""
    with fitz.open() as doc:
        for i in range(pages):
            page = doc.new_page(width=PAGE_W, height=PAGE_H)
            top = MARKER_Y0 + i * MARKER_STEP
            page.draw_rect(
                fitz.Rect(MARKER_X0, top, MARKER_X1, top + MARKER_H),
                fill=(0, 0, 0),
                width=0,
            )
        doc.save(str(path))


def marker_rows(pdf: Path) -> list[int]:
    """Return the first inked raster row of each page, top to bottom."""
    rows = []
    with fitz.open(str(pdf)) as doc:
        for page in doc:
            pix = page.get_pixmap(dpi=72, colorspace=fitz.csGRAY)
            gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
            inked = np.flatnonzero((gray < 128).any(axis=1))
            rows.append(int(inked[0]) if inked.size else -1)
    return rows


def page_rasters(pdf: Path) -> list[np.ndarray]:
    """Rasterise every page to a comparable greyscale array."""
    out = []
    with fitz.open(str(pdf)) as doc:
        for page in doc:
            pix = page.get_pixmap(dpi=72, colorspace=fitz.csGRAY)
            out.append(
                np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).copy()
            )
    return out


@pytest.fixture(scope="module")
def source_pdf(tmp_path_factory):
    """A 12-page source, enough to split unevenly across several workers."""
    path = tmp_path_factory.mktemp("bitonal") / "source.pdf"
    write_numbered_pages(path, 12)
    return path


class TestRunBitonal:
    def test_converts_every_page_to_one_bit(self, source_pdf, tmp_path):
        dst = tmp_path / "out.pdf"
        assert run_bitonal(source_pdf, dst) == 12

        with fitz.open(str(dst)) as doc:
            assert doc.page_count == 12
            for page in doc:
                images = page.get_images(full=True)
                assert len(images) == 1
                assert images[0][4] == 1, "bits per component should be 1"

    def test_preserves_page_geometry(self, source_pdf, tmp_path):
        dst = tmp_path / "out.pdf"
        run_bitonal(source_pdf, dst)

        with fitz.open(str(dst)) as doc:
            for page in doc:
                assert page.rect.width == pytest.approx(PAGE_W)
                assert page.rect.height == pytest.approx(PAGE_H)

    def test_threshold_sets_how_much_ink_survives(self, source_pdf, tmp_path):
        """Pixels are kept black when grey <= threshold, so ink grows with it."""
        counts = []
        for threshold in (0, 160, 255):
            dst = tmp_path / f"t{threshold}.pdf"
            run_bitonal(source_pdf, dst, threshold=threshold)
            counts.append(sum(int((p < 128).sum()) for p in page_rasters(dst)))

        # 0 keeps only pure black (the markers), 255 blackens the whole page.
        assert counts[0] < counts[1] < counts[2]
        assert counts[0] > 0, "the pure-black markers should survive"
        assert all((p < 128).all() for p in page_rasters(tmp_path / "t255.pdf"))


class TestParallelMatchesSequential:
    @pytest.mark.parametrize("workers", [2, 3, 5, 8])
    def test_page_order_survives_the_merge(self, source_pdf, tmp_path, workers):
        dst = tmp_path / f"out_{workers}.pdf"
        run_bitonal(source_pdf, dst, workers=workers)

        rows = marker_rows(dst)
        assert len(rows) == 12
        assert rows == sorted(rows), "pages came back out of order"
        assert len(set(rows)) == 12, "a page was duplicated or dropped"

    @pytest.mark.parametrize("workers", [2, 3, 5])
    def test_output_is_identical_to_sequential(self, source_pdf, tmp_path, workers):
        sequential = tmp_path / "seq.pdf"
        parallel = tmp_path / f"par_{workers}.pdf"
        run_bitonal(source_pdf, sequential)
        run_bitonal(source_pdf, parallel, workers=workers)

        for i, (a, b) in enumerate(zip(page_rasters(sequential), page_rasters(parallel))):
            assert np.array_equal(a, b), f"page {i} differs"

    def test_start_method_is_pinned_to_spawn(self, source_pdf, tmp_path, monkeypatch):
        """The pool must not fall back to the platform default.

        "fork" is that default on Linux, is unavailable on Windows, and is
        the method this codebase has hit runtime problems with before. A
        change that drops ``mp_context`` would otherwise be invisible until
        it reached a machine that behaves differently.
        """
        import concurrent.futures

        captured = {}
        real_executor = concurrent.futures.ProcessPoolExecutor

        class RecordingExecutor(real_executor):
            def __init__(self, *args, **kwargs):
                captured["mp_context"] = kwargs.get("mp_context")
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", RecordingExecutor)
        run_bitonal(source_pdf, tmp_path / "out.pdf", workers=2)

        context = captured["mp_context"]
        assert context is not None, "pool fell back to the platform default"
        assert context.get_start_method() == "spawn"

    def test_more_workers_than_pages_is_harmless(self, tmp_path):
        src = tmp_path / "two.pdf"
        write_numbered_pages(src, 2)
        dst = tmp_path / "out.pdf"

        assert run_bitonal(src, dst, workers=16) == 2
        with fitz.open(str(dst)) as doc:
            assert doc.page_count == 2


class TestProgressCallback:
    def test_sequential_reports_each_page_and_finishes_at_total(self, source_pdf, tmp_path):
        seen = []
        run_bitonal(
            source_pdf,
            tmp_path / "out.pdf",
            progress_callback=lambda c, t, m: seen.append((c, t)),
        )

        assert seen[-1] == (12, 12)
        assert [c for c, _ in seen] == sorted(c for c, _ in seen)
        assert all(t == 12 for _, t in seen)

    def test_parallel_accounts_for_every_page(self, source_pdf, tmp_path):
        seen = []
        run_bitonal(
            source_pdf,
            tmp_path / "out.pdf",
            workers=3,
            progress_callback=lambda c, t, m: seen.append((c, t)),
        )

        assert seen[-1] == (12, 12)
        assert [c for c, _ in seen] == sorted(c for c, _ in seen)


class TestIccHandling:
    def test_state_is_restored_after_conversion(self, source_pdf, tmp_path):
        """The context manager must not leave ICC off for later renders."""
        with _icc_disabled():
            pass
        baseline = page_rasters(source_pdf)

        run_bitonal(source_pdf, tmp_path / "out.pdf")

        assert np.array_equal(baseline[0], page_rasters(source_pdf)[0])

    def test_restores_on_exception(self):
        with pytest.raises(RuntimeError), _icc_disabled():
            raise RuntimeError("boom")
        # Reaching here without the flag stuck off is the assertion; a
        # leaked "off" would change every subsequent colour render.
        fitz.TOOLS.set_icc(True)


class TestApiBitonal:
    def test_writes_bitonal_pdf_into_output_dir(self, source_pdf, tmp_path):
        out = bitonal(source_pdf, tmp_path)

        assert out == tmp_path / "bitonal.pdf"
        assert out.exists()
        with fitz.open(str(out)) as doc:
            assert doc.page_count == 12

    def test_creates_missing_output_dir(self, source_pdf, tmp_path):
        target = tmp_path / "nested" / "dir"
        out = bitonal(source_pdf, target, workers=2)

        assert out.exists()
        assert out.parent == target

    def test_workers_reaches_the_conversion(self, source_pdf, tmp_path):
        sequential = bitonal(source_pdf, tmp_path / "a")
        parallel = bitonal(source_pdf, tmp_path / "b", workers=3)

        for a, b in zip(page_rasters(sequential), page_rasters(parallel)):
            assert np.array_equal(a, b)

    def test_progress_callback_is_forwarded(self, source_pdf, tmp_path):
        seen = []
        bitonal(
            source_pdf,
            tmp_path,
            progress_callback=lambda c, t, m: seen.append((c, t, m)),
        )

        assert seen[-1][0] == 12
        assert seen[-1][1] == 12
        assert "Bitonal" in seen[-1][2]
