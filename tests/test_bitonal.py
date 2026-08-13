"""Tests for bitonal conversion.

Conversion runs page by page in the calling process, and a caller that
wants it parallelised fans out over ``tasks.bitonal_chunk`` itself. What
is worth asserting here is that the renderer both paths share produces
1-bit pages, in order, with the geometry and ink the threshold implies.

The pages carry a marker whose vertical position encodes the page index,
so a page that came back out of order is detectable from the raster alone.
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


# A colour fill rasterises to a different grey depending on whether MuPDF's
# colour management is on, which is the only handle these tests have on a
# flag PyMuPDF exposes no getter for.
COLOUR_FILL = (0.2, 0.6, 0.9)
ICC_ON_GREY = 145
ICC_OFF_GREY = 130


def write_colour_page(path: Path) -> None:
    """Write a one-page PDF filled with :data:`COLOUR_FILL`."""
    with fitz.open() as doc:
        page = doc.new_page(width=PAGE_W, height=PAGE_H)
        page.draw_rect(page.rect, fill=COLOUR_FILL, width=0)
        doc.save(str(path))


def grey_of(pdf: Path) -> int:
    """Rasterise page 1 to greyscale and return its centre pixel."""
    with fitz.open(str(pdf)) as doc:
        pix = doc[0].get_pixmap(dpi=72, colorspace=fitz.csGRAY)
        gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
        return int(gray[pix.height // 2, pix.width // 2])


def convert_in_chunks(src: Path, dst: Path, n_chunks: int, tmp_dir: Path) -> None:
    """Convert ``src`` a range at a time and merge, as a caller would."""
    from blackletter.tasks import bitonal_chunk, merge_pdfs, split_page_ranges

    with fitz.open(str(src)) as doc:
        total = doc.page_count

    chunks = []
    for i, (start, end) in enumerate(split_page_ranges(total, n_chunks)):
        chunk = tmp_dir / f"chunk_{dst.stem}_{i:04d}.pdf"
        bitonal_chunk(str(src), str(chunk), start, end)
        chunks.append(chunk)
    merge_pdfs(chunks, dst)


@pytest.fixture(scope="module")
def source_pdf(tmp_path_factory):
    """A 12-page source, enough to split unevenly into several ranges."""
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


class TestChunkedMatchesWholeDocument:
    """The contract a caller fanning out over ``bitonal_chunk`` relies on.

    Parallelism lives above this library, so what has to hold is that
    converting a document in ranges and merging the chunks back gives the
    same pages, in the same order, as converting it in one call.
    """

    @pytest.mark.parametrize("n_chunks", [2, 3, 5, 8])
    def test_page_order_survives_the_merge(self, source_pdf, tmp_path, n_chunks):
        dst = tmp_path / f"out_{n_chunks}.pdf"
        convert_in_chunks(source_pdf, dst, n_chunks, tmp_path)

        rows = marker_rows(dst)
        assert len(rows) == 12
        assert rows == sorted(rows), "pages came back out of order"
        assert len(set(rows)) == 12, "a page was duplicated or dropped"

    @pytest.mark.parametrize("n_chunks", [2, 3, 5])
    def test_output_is_identical_to_one_call(self, source_pdf, tmp_path, n_chunks):
        whole = tmp_path / "whole.pdf"
        chunked = tmp_path / f"chunked_{n_chunks}.pdf"
        run_bitonal(source_pdf, whole)
        convert_in_chunks(source_pdf, chunked, n_chunks, tmp_path)

        for i, (a, b) in enumerate(zip(page_rasters(whole), page_rasters(chunked))):
            assert np.array_equal(a, b), f"page {i} differs"

    def test_chunk_disables_icc_itself(self, tmp_path):
        """ICC is process-global, so a chunk running in its own process
        cannot inherit the setting from whoever fanned it out.

        Thresholded at ``ICC_OFF_GREY < t < ICC_ON_GREY``, so the fill
        comes out black only if the chunk turned colour management off.
        """
        from blackletter.tasks import bitonal_chunk

        src = tmp_path / "colour.pdf"
        write_colour_page(src)
        chunk = tmp_path / "chunk.pdf"

        fitz.TOOLS.set_icc(True)
        bitonal_chunk(str(src), str(chunk), 0, 1, threshold=137)

        assert (page_rasters(chunk)[0] < 128).any(), "the fill rendered under ICC"

    def test_more_chunks_than_pages_is_harmless(self, tmp_path):
        src = tmp_path / "two.pdf"
        write_numbered_pages(src, 2)
        dst = tmp_path / "out.pdf"

        convert_in_chunks(src, dst, 16, tmp_path)
        with fitz.open(str(dst)) as doc:
            assert doc.page_count == 2


class TestProgressCallback:
    def test_reports_each_page_and_finishes_at_total(self, source_pdf, tmp_path):
        seen = []
        run_bitonal(
            source_pdf,
            tmp_path / "out.pdf",
            progress_callback=lambda c, t, m: seen.append((c, t)),
        )

        assert seen[-1] == (12, 12)
        assert [c for c, _ in seen] == sorted(c for c, _ in seen)
        assert all(t == 12 for _, t in seen)


class TestIccHandling:
    """ICC is global to the process, so the flag has to come back on.

    None of this can be checked against the black-on-white fixture above,
    which rasterises identically either way — a test written on it would
    pass with the restore deleted. These use a colour fill instead, the
    one thing in this file whose rendering the flag actually moves.
    """

    @pytest.fixture(autouse=True)
    def _icc_on(self):
        """Leave the flag on for the next test whatever this one does."""
        fitz.TOOLS.set_icc(True)
        yield
        fitz.TOOLS.set_icc(True)

    def test_the_probe_can_see_the_flag(self, tmp_path):
        """Guards the rest of the class: without this they go blind."""
        probe = tmp_path / "probe.pdf"
        write_colour_page(probe)

        assert grey_of(probe) == ICC_ON_GREY
        with _icc_disabled():
            assert grey_of(probe) == ICC_OFF_GREY

    def test_state_is_restored_after_conversion(self, source_pdf, tmp_path):
        probe = tmp_path / "probe.pdf"
        write_colour_page(probe)

        run_bitonal(source_pdf, tmp_path / "out.pdf")

        assert grey_of(probe) == ICC_ON_GREY, "conversion left colour management off"

    def test_restores_on_exception(self, tmp_path):
        probe = tmp_path / "probe.pdf"
        write_colour_page(probe)

        with pytest.raises(RuntimeError), _icc_disabled():
            raise RuntimeError("boom")

        assert grey_of(probe) == ICC_ON_GREY, "the flag stuck off"


class TestApiBitonal:
    def test_writes_bitonal_pdf_into_output_dir(self, source_pdf, tmp_path):
        out = bitonal(source_pdf, tmp_path)

        assert out == tmp_path / "bitonal.pdf"
        assert out.exists()
        with fitz.open(str(out)) as doc:
            assert doc.page_count == 12

    def test_creates_missing_output_dir(self, source_pdf, tmp_path):
        target = tmp_path / "nested" / "dir"
        out = bitonal(source_pdf, target)

        assert out.exists()
        assert out.parent == target

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
