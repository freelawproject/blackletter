"""Tests for ``api.add_text_layer``.

The point of this entry point is that it runs over files that are already
finished, so what matters is which files it picks up, that it leaves the
originals intact when a file already has text or when ocrmypdf fails, and
that it replaces each file in place rather than writing a copy somewhere.

ocrmypdf itself is replaced with a stub: it is the one part of this that is
not ours, and running it for real would make these tests minutes long. The
stub lives in this process, so these tests pass ``jobs=1`` to stay out of
the worker pool; the pool's own arithmetic is covered separately.
"""

from __future__ import annotations

import inspect
import sys
from types import SimpleNamespace

import fitz
import pytest

from blackletter.api import _collect_pdfs, _inner_jobs, _text_layer_jobs, add_text_layer
from tests.pdf_fixtures import write_bitonal_page, write_text_page

STUB_MARK = b"stubbed-ocr-output"


@pytest.fixture
def scratch(tmp_path):
    """Somewhere for the fixtures' intermediate text PDFs to live.

    ``write_bitonal_page`` rasterizes a text PDF it writes next to its
    target, and these tests assert on exactly which files exist in a
    directory, so the intermediates have to go elsewhere.
    """
    path = tmp_path / "scratch"
    path.mkdir()
    return path


@pytest.fixture
def fake_ocrmypdf(monkeypatch):
    """Stand in for ocrmypdf, recording calls and writing a marked file."""
    calls = []

    # Captured before the stub replaces the module: real ocrmypdf takes
    # **kwargs and silently ignores names it does not know, so a misspelled
    # option would pass unnoticed, and its own signature is the only guard.
    import ocrmypdf

    allowed = set(inspect.signature(ocrmypdf.ocr).parameters) - {"kwargs"}

    def _ocr(src, dst, **kwargs):
        unknown = set(kwargs) - allowed
        assert not unknown, f"ocrmypdf does not take {sorted(unknown)}"
        calls.append({"src": src, "dst": dst, **kwargs})
        with fitz.open(src) as doc:
            doc.save(dst)
        with open(dst, "ab") as fh:
            fh.write(b"%%" + STUB_MARK)

    monkeypatch.setitem(sys.modules, "ocrmypdf", SimpleNamespace(ocr=_ocr))
    return calls


def _is_stubbed(path) -> bool:
    """Whether the stub rewrote this file in place."""
    return STUB_MARK in path.read_bytes()


class TestCollectPdfs:
    def test_single_file(self, tmp_path, scratch):
        pdf = tmp_path / "one.pdf"
        write_bitonal_page(pdf, tmp_dir=scratch)
        assert _collect_pdfs(pdf) == [pdf]

    def test_directory_is_expanded_and_sorted(self, tmp_path, scratch):
        opinions = tmp_path / "redacted"
        opinions.mkdir()
        for name in ("b.pdf", "a.pdf", "c.pdf"):
            write_bitonal_page(opinions / name, tmp_dir=scratch)
        (opinions / "notes.txt").write_text("not a pdf")
        nested = opinions / "deeper"
        nested.mkdir()
        write_bitonal_page(nested / "d.pdf", tmp_dir=scratch)

        found = _collect_pdfs(opinions)
        # Sorted, non-recursive, and only PDFs.
        assert [p.name for p in found] == ["a.pdf", "b.pdf", "c.pdf"]

    def test_mixed_iterable(self, tmp_path, scratch):
        loose = tmp_path / "full.redacted.pdf"
        write_bitonal_page(loose, tmp_dir=scratch)
        opinions = tmp_path / "redacted"
        opinions.mkdir()
        write_bitonal_page(opinions / "op.pdf", tmp_dir=scratch)

        found = _collect_pdfs([loose, opinions])
        assert loose in found
        assert opinions / "op.pdf" in found

    def test_a_file_and_its_own_directory_yield_it_once(self, tmp_path, scratch):
        """The case dedup exists for: two workers would OCR it at once."""
        write_bitonal_page(tmp_path / "a.pdf", tmp_dir=scratch)
        found = _collect_pdfs([tmp_path / "a.pdf", tmp_path])
        assert [p.name for p in found] == ["a.pdf"]

    def test_an_uppercase_suffix_is_a_pdf_too(self, tmp_path, scratch):
        write_bitonal_page(tmp_path / "b.PDF", tmp_dir=scratch)
        assert [p.name for p in _collect_pdfs(tmp_path)] == ["b.PDF"]

    def test_missing_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _collect_pdfs(tmp_path / "nope.pdf")


class TestAddTextLayer:
    def test_replaces_each_file_in_place(self, tmp_path, scratch, fake_ocrmypdf):
        pdf = tmp_path / "op.pdf"
        write_bitonal_page(pdf, tmp_dir=scratch)
        before = pdf.read_bytes()

        written = add_text_layer(pdf, jobs=1)

        assert written == [pdf]
        assert pdf.read_bytes() != before
        assert _is_stubbed(pdf)
        # No stray output beside it, and no leftover temp file.
        assert [p.name for p in tmp_path.glob("*.pdf")] == ["op.pdf"]

    def test_skips_files_that_already_have_text(self, tmp_path, fake_ocrmypdf):
        searchable = tmp_path / "searchable.pdf"
        write_text_page(searchable)
        before = searchable.read_bytes()

        assert add_text_layer(searchable, jobs=1) == []
        assert not fake_ocrmypdf, "ran ocrmypdf on an already searchable file"
        assert searchable.read_bytes() == before

    def test_skip_existing_false_processes_anyway(self, tmp_path, fake_ocrmypdf):
        searchable = tmp_path / "searchable.pdf"
        write_text_page(searchable)

        assert add_text_layer(searchable, skip_existing=False, jobs=1) == [searchable]
        assert len(fake_ocrmypdf) == 1

    def test_leaves_pages_that_already_have_text_alone(self, tmp_path, scratch, fake_ocrmypdf):
        """A mixed file must not have its text pages rasterised away."""
        pdf = tmp_path / "op.pdf"
        write_bitonal_page(pdf, tmp_dir=scratch)
        add_text_layer(pdf, jobs=1)
        assert fake_ocrmypdf[0]["skip_text"] is True

    def test_original_survives_a_failure(self, tmp_path, scratch, monkeypatch):
        pdf = tmp_path / "op.pdf"
        write_bitonal_page(pdf, tmp_dir=scratch)
        before = pdf.read_bytes()

        def _boom(src, dst, **kwargs):
            raise RuntimeError("tesseract exploded")

        monkeypatch.setitem(sys.modules, "ocrmypdf", SimpleNamespace(ocr=_boom))

        with pytest.raises(RuntimeError):
            add_text_layer(pdf, jobs=1)

        assert pdf.read_bytes() == before
        # The temp file is created in the target's own directory, so it has
        # to be cleaned up or it ships alongside the deliverable.
        assert sorted(p.name for p in tmp_path.iterdir()) == ["op.pdf", "scratch"]

    def test_the_returned_list_is_sorted(self, tmp_path, scratch, fake_ocrmypdf):
        """Order must not depend on which worker finished first.

        With a pool the completion order is arbitrary, so the serial and
        parallel paths would otherwise return different lists for the same
        input.
        """
        names = ["c.pdf", "a.pdf", "b.pdf"]
        for name in names:
            write_bitonal_page(tmp_path / name, tmp_dir=scratch)
        written = add_text_layer([tmp_path / n for n in names], jobs=1)
        assert [p.name for p in written] == ["a.pdf", "b.pdf", "c.pdf"]

    def test_walks_a_directory_of_opinions(self, tmp_path, scratch, fake_ocrmypdf):
        opinions = tmp_path / "redacted"
        opinions.mkdir()
        names = ["a3d.222.0001-0027.pdf", "a3d.222.0028-0031.pdf"]
        for name in names:
            write_bitonal_page(opinions / name, tmp_dir=scratch)

        written = add_text_layer(opinions, jobs=1)

        for name in names:
            assert opinions / name in written
            assert _is_stubbed(opinions / name)

    def test_progress_counts_only_the_files_that_need_work(self, tmp_path, scratch, fake_ocrmypdf):
        """An already searchable file is not work, so it is not in the total."""
        first = tmp_path / "a.pdf"
        second = tmp_path / "b.pdf"
        write_bitonal_page(first, tmp_dir=scratch)
        write_bitonal_page(second, tmp_dir=scratch)
        searchable = tmp_path / "searchable.pdf"
        write_text_page(searchable)
        events = []

        written = add_text_layer(
            [first, second, searchable],
            jobs=1,
            progress_callback=lambda done, total, msg: events.append((done, total, msg)),
        )

        assert written == [first, second]
        assert [(d, t) for d, t, _m in events] == [(1, 2), (2, 2)]
        assert all("Text layer" in msg for _d, _t, msg in events)

    def test_empty_input_is_a_no_op(self, tmp_path, fake_ocrmypdf):
        empty = tmp_path / "redacted"
        empty.mkdir()
        assert add_text_layer(empty, jobs=1) == []
        assert not fake_ocrmypdf

    def test_language_and_optimize_reach_ocrmypdf(self, tmp_path, scratch, fake_ocrmypdf):
        pdf = tmp_path / "op.pdf"
        write_bitonal_page(pdf, tmp_dir=scratch)
        add_text_layer(pdf, language="deu", optimize=3, jobs=1)
        assert fake_ocrmypdf[0]["language"] == ["deu"]
        assert fake_ocrmypdf[0]["optimize"] == 3

    def test_every_argument_is_pinned(self, tmp_path, scratch, fake_ocrmypdf):
        """The whole call, so a changed default cannot slip through.

        Values are checked against ocrmypdf's own options model, which does
        validate them, unlike the stub.
        """
        pdf = tmp_path / "op.pdf"
        write_bitonal_page(pdf, tmp_dir=scratch)
        add_text_layer(pdf, jobs=1)
        call = dict(fake_ocrmypdf[0])
        assert call.pop("src") == str(pdf)
        assert call.pop("dst").startswith(str(tmp_path))
        assert call == {
            "pdf_renderer": "auto",
            "optimize": 1,
            "output_type": "pdf",
            "language": ["eng"],
            "skip_text": True,
            "tesseract_timeout": 120,
            "progress_bar": False,
        }

    def test_a_misnamed_option_would_be_caught(self, tmp_path, scratch, fake_ocrmypdf):
        """Proof that the stub's signature check does something.

        ``skiptext`` is a plausible typo for ``skip_text``, and the real
        library accepts it silently, so nothing but this catches it.
        """
        pdf = tmp_path / "op.pdf"
        write_bitonal_page(pdf, tmp_dir=scratch)
        with pytest.raises(AssertionError, match="does not take"):
            sys.modules["ocrmypdf"].ocr(str(pdf), str(pdf), skiptext=True)


class TestWorkerCount:
    """``jobs`` arithmetic, which decides whether a pool is used at all."""

    def test_never_more_workers_than_files(self):
        assert _text_layer_jobs(total=2, jobs=16) == 2

    def test_single_file_runs_in_process(self):
        assert _text_layer_jobs(total=1, jobs=None) == 1

    def test_explicit_one_is_respected(self):
        assert _text_layer_jobs(total=100, jobs=1) == 1

    def test_zero_and_negative_are_floored(self):
        assert _text_layer_jobs(total=10, jobs=0) == 1
        assert _text_layer_jobs(total=10, jobs=-4) == 1

    def test_default_scales_with_the_machine(self, monkeypatch):
        import multiprocessing

        monkeypatch.setattr(multiprocessing, "cpu_count", lambda: 8)
        assert _text_layer_jobs(total=100, jobs=None) == 4
        # A single-core machine still gets a worker.
        monkeypatch.setattr(multiprocessing, "cpu_count", lambda: 1)
        assert _text_layer_jobs(total=100, jobs=None) == 1


class TestInnerJobs:
    """Cores handed to each ocrmypdf run when several run side by side.

    Both regimes were measured on real opinions. Forcing one core per
    worker made a page-heavy batch 1.6x slower than running the files one
    at a time, because ocrmypdf's own page-level parallelism was doing the
    work; splitting the machine between workers instead is faster than
    sequential in both shapes.
    """

    def test_single_worker_keeps_ocrmypdf_defaults(self):
        assert _inner_jobs(1) is None
        assert _inner_jobs(0) is None

    def test_cores_are_split_between_workers(self, monkeypatch):
        import multiprocessing

        monkeypatch.setattr(multiprocessing, "cpu_count", lambda: 16)
        assert _inner_jobs(4) == 4
        assert _inner_jobs(8) == 2

    def test_never_drops_below_one(self, monkeypatch):
        import multiprocessing

        monkeypatch.setattr(multiprocessing, "cpu_count", lambda: 4)
        assert _inner_jobs(32) == 1
