"""The options a caller drives ``generate`` with, per opinion.

The scanning portal calls ``generate`` once per opinion over a small
source PDF holding that opinion's pages alone, once per daemon tick. That
call wants the opinion file and nothing else: no second pass over the
whole source, the file under its own name, the path handed back rather
than re-derived, one bad opinion costing one opinion, and a photograph
put on the page before the paint. Everything here is one of those.
"""

from __future__ import annotations

import logging

import fitz
import numpy as np
import pytest

from blackletter.api import generate
from tests.pdf_fixtures import write_multi_page

DPI = 150


def dark_fraction(page, rect: fitz.Rect) -> float:
    """Fraction of dark pixels inside a region of a rendered page."""
    pix = page.get_pixmap(dpi=DPI, colorspace=fitz.csGRAY, clip=rect)
    gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)[
        :, : pix.width
    ]
    return float((gray < 128).mean()) if gray.size else 0.0


@pytest.fixture
def source(tmp_path):
    """A four-page source, the shape of one opinion's shard."""
    path = tmp_path / "small.pdf"
    write_multi_page(path, ["body", "body", "body", "body"], tmp_dir=tmp_path)
    return path


def opinion(caption: int, end: int, **extra) -> dict:
    """One opinion dict over *caption*..*end*, with no rects outside it."""
    return {"caption_page": caption, "end_page": end, "outside_rects": [], **extra}


def payload(opinions: list[dict], pages: dict | None = None, images: dict | None = None) -> dict:
    """A ``generate`` payload; ``pages`` is required even when empty."""
    data = {"opinions": opinions, "pages": pages or {}}
    if images is not None:
        data["images"] = images
    return data


class TestSkippingTheFullPDF:
    """``full_redacted=False``: the opinion files alone.

    Over a volume the full pass is one ``apply_redactions`` over ~1300
    image pages; over the portal's shard it is a second copy of a
    four-page file, deleted after the call.
    """

    def test_no_full_file_is_written_and_the_result_says_so(self, source, tmp_path):
        out = tmp_path / "out"
        result = generate(
            source, payload([opinion(0, 1, filename="op.pdf")]), out, full_redacted=False
        )

        assert result["full_redacted"] is None
        assert list(out.glob("*.redacted.pdf")) == []
        assert (out / "redacted" / "op.pdf").is_file()

    def test_the_full_file_is_still_the_default(self, source, tmp_path):
        out = tmp_path / "out"
        result = generate(
            source,
            payload([opinion(0, 1, first_page_number=1, last_page_number=2)]),
            out,
            reporter="a3d",
            volume="222",
        )

        assert result["full_redacted"] == out / "a3d.222.1.2.redacted.pdf"
        assert result["full_redacted"].is_file()

    def test_an_empty_opinion_list_raises_nothing(self, source, tmp_path):
        """The name of the full file was the one read of ``opinions[0]``."""
        result = generate(source, payload([]), tmp_path / "out", full_redacted=False)

        assert result["files"] == []
        assert result["opinion_count"] == 0
        assert result["failed"] == []

    def test_llm_without_the_full_file_is_refused(self, source, tmp_path):
        """``_split_llm_pages`` slices the full document; there is none."""
        with pytest.raises(ValueError, match="llm=True requires full_redacted=True"):
            generate(
                source,
                payload([opinion(0, 1)]),
                tmp_path / "out",
                full_redacted=False,
                llm=True,
            )


class TestTheCallersFileNameWins:
    """A printed page range in the name goes stale when a boundary moves.

    A consumer that stores these files under an internal name needs that
    to be a contract, not the fallback it reached by leaving the printed
    numbers off the dict.
    """

    def test_a_filename_beats_the_printed_numbers(self, source, tmp_path):
        out = tmp_path / "out"
        result = generate(
            source,
            payload(
                [
                    opinion(
                        0,
                        1,
                        first_page_number=41,
                        last_page_number=42,
                        filename="internal-7.pdf",
                    )
                ]
            ),
            out,
            reporter="a3d",
            volume="222",
            full_redacted=False,
        )

        assert (out / "redacted" / "internal-7.pdf").is_file()
        assert list((out / "redacted").iterdir()) == [out / "redacted" / "internal-7.pdf"]
        assert result["files"] == [out / "redacted" / "internal-7.pdf"]

    def test_a_dict_with_no_filename_is_still_named_from_its_numbers(self, source, tmp_path):
        out = tmp_path / "out"
        generate(
            source,
            payload([opinion(0, 1, first_page_number=41, last_page_number=42)]),
            out,
            reporter="a3d",
            volume="222",
            full_redacted=False,
        )

        assert (out / "redacted" / "a3d.222.0041-0042.pdf").is_file()

    def test_duplicates_are_still_suffixed(self, source, tmp_path):
        """Two opinions can share a name, from either rule."""
        out = tmp_path / "out"
        result = generate(
            source,
            payload([opinion(0, 1, filename="same.pdf"), opinion(2, 3, filename="same.pdf")]),
            out,
            full_redacted=False,
        )

        assert result["files"] == [
            out / "redacted" / "same-1.pdf",
            out / "redacted" / "same-2.pdf",
        ]
        assert all(p.is_file() for p in result["files"])

    def test_duplicates_are_suffixed_whatever_the_extension(self, source, tmp_path):
        """The old split matched a lowercase ``.pdf`` and nothing else.

        With the caller's name as the contract, a pair it spells
        ``.PDF`` would have come back as two paths to one file, the
        second having overwritten the first.
        """
        out = tmp_path / "out"
        result = generate(
            source,
            payload([opinion(0, 1, filename="same.PDF"), opinion(2, 3, filename="same.PDF")]),
            out,
            full_redacted=False,
        )

        assert result["files"] == [
            out / "redacted" / "same-1.PDF",
            out / "redacted" / "same-2.PDF",
        ]
        assert len({p.read_bytes() for p in result["files"]}) == 2, "one file, written twice"

    def test_a_name_with_a_path_fails_only_that_opinion(self, source, tmp_path):
        """A name is a file name. Joined verbatim it would escape.

        The cleanup below deletes what a failed opinion left behind, so
        an escaping name is also a delete outside ``output_dir``.
        """
        out = tmp_path / "out"
        bystander = tmp_path / "bystander.pdf"
        bystander.write_bytes(b"%PDF-1.7 not ours\n")
        result = generate(
            source,
            payload(
                [
                    opinion(0, 0, filename="../bystander.pdf"),
                    opinion(1, 1, filename="fine.pdf"),
                ]
            ),
            out,
            full_redacted=False,
        )

        assert bystander.read_bytes() == b"%PDF-1.7 not ours\n", "wrote or deleted outside"
        assert result["files"] == [None, out / "redacted" / "fine.pdf"]
        assert "bare file name" in result["failed"][0]["error"]


class TestTheResultNamesEachFile:
    """A caller should not have to re-derive the name it just passed in."""

    def test_files_are_in_input_order(self, source, tmp_path):
        out = tmp_path / "out"
        ops = [
            opinion(0, 0, filename="c.pdf"),
            opinion(1, 2, filename="a.pdf"),
            opinion(3, 3, filename="b.pdf"),
        ]
        result = generate(source, payload(ops), out, full_redacted=False)

        assert result["files"] == [
            out / "redacted" / "c.pdf",
            out / "redacted" / "a.pdf",
            out / "redacted" / "b.pdf",
        ]
        assert [p.name for p in result["files"]] != sorted(p.name for p in result["files"])

    def test_the_unredacted_twin_shares_the_name(self, source, tmp_path):
        out = tmp_path / "out"
        result = generate(
            source,
            payload([opinion(0, 1, filename="op.pdf")]),
            out,
            unredacted=True,
            full_redacted=False,
        )

        assert result["files"] == [out / "redacted" / "op.pdf"]
        assert (out / "unredacted" / "op.pdf").is_file()


class TestOneBadOpinionFailsOneOpinion:
    """A fault used to end the call with the earlier files on disk.

    The caller could not tell which opinions were written from which were
    never reached, and the half-written one looked like a deliverable.
    """

    @staticmethod
    def _three(source, tmp_path, **kwargs):
        """Three opinions; the middle one runs past the last page."""
        out = tmp_path / "out"
        ops = [
            opinion(0, 0, filename="first.pdf"),
            opinion(1, 99, filename="broken.pdf"),
            opinion(2, 3, filename="third.pdf"),
        ]
        return out, generate(source, payload(ops), out, full_redacted=False, **kwargs)

    def test_the_other_files_are_written(self, source, tmp_path):
        out, _ = self._three(source, tmp_path)

        assert (out / "redacted" / "first.pdf").is_file()
        assert (out / "redacted" / "third.pdf").is_file()

    def test_the_fault_is_named_and_the_slot_is_empty(self, source, tmp_path):
        out, result = self._three(source, tmp_path)

        assert result["files"][1] is None
        assert result["files"][0] == out / "redacted" / "first.pdf"
        assert result["files"][2] == out / "redacted" / "third.pdf"
        assert [f["index"] for f in result["failed"]] == [1]
        assert result["failed"][0]["error"]

    def test_no_partial_file_is_left_behind(self, source, tmp_path):
        out, _ = self._three(source, tmp_path)

        assert not (out / "redacted" / "broken.pdf").exists()

    def test_an_unredacted_file_of_an_earlier_run_survives(self, source, tmp_path):
        """The cleanup removes what *this* call wrote, not what it found."""
        out = tmp_path / "out"
        earlier = out / "unredacted"
        earlier.mkdir(parents=True)
        (earlier / "broken.pdf").write_bytes(b"%PDF-1.7 earlier run\n")

        self._three(source, tmp_path)

        assert (earlier / "broken.pdf").read_bytes() == b"%PDF-1.7 earlier run\n"

    def test_a_dict_with_a_null_page_range_fails_only_that_opinion(self, source, tmp_path):
        """The derived name formats the range, before the loop."""
        out = tmp_path / "out"
        result = generate(
            source,
            payload(
                [
                    {"caption_page": None, "end_page": None, "outside_rects": []},
                    opinion(0, 0, filename="ok.pdf"),
                ]
            ),
            out,
            full_redacted=False,
        )

        assert result["files"][1] == out / "redacted" / "ok.pdf"
        assert result["failed"][0]["index"] == 0

    def test_opinion_count_still_counts_the_inputs(self, source, tmp_path):
        _, result = self._three(source, tmp_path)

        assert result["opinion_count"] == 3

    def test_a_clean_run_reports_no_failures(self, source, tmp_path):
        result = generate(
            source,
            payload([opinion(0, 1, filename="op.pdf")]),
            tmp_path / "out",
            full_redacted=False,
        )

        assert result["failed"] == []

    def test_a_dict_with_no_page_range_fails_only_that_opinion(self, source, tmp_path):
        """The range is read inside the loop, so a malformed dict is one row."""
        out = tmp_path / "out"
        result = generate(
            source,
            payload([{"filename": "nothing.pdf"}, opinion(0, 0, filename="ok.pdf")]),
            out,
            full_redacted=False,
        )

        assert result["files"] == [None, out / "redacted" / "ok.pdf"]
        assert result["failed"][0]["index"] == 0

    def test_a_payload_missing_pages_is_a_fault_of_the_whole_call(self, source, tmp_path):
        with pytest.raises(ValueError, match="no 'pages'"):
            generate(source, {"opinions": []}, tmp_path / "out", full_redacted=False)

    def test_a_payload_missing_opinions_is_a_fault_of_the_whole_call(self, source, tmp_path):
        with pytest.raises(ValueError, match="no 'opinions'"):
            generate(source, {"pages": {}}, tmp_path / "out", full_redacted=False)


class TestThePictures:
    """A bitonal conversion destroys a photograph.

    The caller holds the higher-quality copy, so it supplies the bytes and
    ``generate`` places them -- before the paint, so a rect covering part
    of a picture still blacks it out.
    """

    IMAGE_RECT = {"x0": 200.0, "y0": 200.0, "x1": 400.0, "y1": 400.0}

    #: The fixture's pages are rasterized scans, so each one already holds
    #: one image before anything is inserted.
    BASE_IMAGES = 1

    @staticmethod
    def _png(color: tuple[int, int, int]) -> bytes:
        """A small solid-colour PNG."""
        pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 60, 60), False)
        pix.set_rect(pix.irect, color)
        return pix.tobytes("png")

    def test_the_picture_lands_on_the_page(self, source, tmp_path):
        out = tmp_path / "out"
        asked: list[tuple[int, fitz.Rect]] = []

        def image_for(page_index, rect):
            asked.append((page_index, rect))
            return self._png((200, 30, 30))

        generate(
            source,
            payload(
                [opinion(1, 2, filename="op.pdf")],
                images={"1": [self.IMAGE_RECT]},
            ),
            out,
            full_redacted=False,
            image_for=image_for,
        )

        assert asked == [(1, fitz.Rect(200, 200, 400, 400))], "the source page index is passed"
        with fitz.open(str(out / "redacted" / "op.pdf")) as doc:
            assert len(doc[0].get_images()) == self.BASE_IMAGES + 1, "no picture was inserted"
            assert len(doc[1].get_images()) == self.BASE_IMAGES, "a page with no entry was stamped"
            # ...and it is on the page, not merely in its resources.
            covered = dark_fraction(doc[0], fitz.Rect(210, 210, 390, 390))
        assert covered > 0.95, f"the picture is not on the page ({covered:.3f} dark)"

    def test_the_paint_runs_after_the_picture(self, source, tmp_path):
        """A black rect over half the picture still reads black."""
        out = tmp_path / "out"
        rects = {
            "1": [
                {
                    "x0": 200.0,
                    "y0": 200.0,
                    "x1": 300.0,
                    "y1": 400.0,
                    "fill": "black",
                    "type": "HEADNOTE",
                }
            ]
        }
        generate(
            source,
            payload(
                [opinion(1, 1, filename="op.pdf")],
                pages=rects,
                images={"1": [self.IMAGE_RECT]},
            ),
            out,
            full_redacted=False,
            image_for=lambda page_index, rect: self._png((255, 255, 255)),
        )

        with fitz.open(str(out / "redacted" / "op.pdf")) as doc:
            painted = dark_fraction(doc[0], fitz.Rect(210, 210, 290, 390))
        assert painted > 0.95, f"the picture survived the paint ({painted:.3f} dark)"

    def test_none_leaves_the_rect_alone(self, source, tmp_path):
        out = tmp_path / "out"
        generate(
            source,
            payload([opinion(0, 0, filename="op.pdf")], images={"0": [self.IMAGE_RECT]}),
            out,
            full_redacted=False,
            image_for=lambda page_index, rect: None,
        )

        with fitz.open(str(out / "redacted" / "op.pdf")) as doc:
            assert len(doc[0].get_images()) == self.BASE_IMAGES

    def test_the_images_entry_is_ignored_without_a_callable(self, source, tmp_path):
        out = tmp_path / "out"
        generate(
            source,
            payload([opinion(0, 0, filename="op.pdf")], images={"0": [self.IMAGE_RECT]}),
            out,
            full_redacted=False,
        )

        with fitz.open(str(out / "redacted" / "op.pdf")) as doc:
            assert len(doc[0].get_images()) == self.BASE_IMAGES

    def test_a_raising_callable_fails_only_that_opinion(self, source, tmp_path):
        out = tmp_path / "out"

        def image_for(page_index, rect):
            raise RuntimeError("the shard is gone")

        result = generate(
            source,
            payload(
                [opinion(0, 0, filename="ok.pdf"), opinion(1, 1, filename="bad.pdf")],
                images={"1": [self.IMAGE_RECT]},
            ),
            out,
            full_redacted=False,
            image_for=image_for,
        )

        assert result["files"][0] == out / "redacted" / "ok.pdf"
        assert result["files"][1] is None
        assert "the shard is gone" in result["failed"][0]["error"]


class TestProgressReachesTheOpinions:
    """It used to fire only inside the full pass, which a caller may skip."""

    def test_the_callback_fires_once_per_opinion(self, source, tmp_path):
        seen: list[tuple[int, int, str]] = []
        generate(
            source,
            payload([opinion(0, 0, filename="a.pdf"), opinion(1, 1, filename="b.pdf")]),
            tmp_path / "out",
            full_redacted=False,
            progress_callback=lambda current, total, message: seen.append(
                (current, total, message)
            ),
        )

        assert seen == [(1, 2, "Writing opinion PDFs..."), (2, 2, "Writing opinion PDFs...")]


class TestWhatTheLogSays:
    """The daemon's only account of a call it did not watch.

    A failed opinion does not raise, so the ERROR record is the only place
    its traceback exists.
    """

    def test_a_run_logs_its_start_and_its_end(self, source, tmp_path, caplog):
        with caplog.at_level(logging.INFO, logger="blackletter.api"):
            generate(
                source,
                payload([opinion(0, 1, filename="op.pdf")]),
                tmp_path / "out",
                full_redacted=False,
            )

        messages = [r.getMessage() for r in caplog.records]
        assert any("start - 4 page(s), 1 opinion(s)" in m for m in messages), messages
        assert any("done - 1 written, 0 failed" in m for m in messages), messages
        assert all(m.startswith("generate small.pdf:") for m in messages), messages

    def test_a_failed_opinion_logs_an_error_with_its_traceback(self, source, tmp_path, caplog):
        with caplog.at_level(logging.INFO, logger="blackletter.api"):
            generate(
                source,
                payload([opinion(1, 99, filename="broken.pdf")]),
                tmp_path / "out",
                full_redacted=False,
            )

        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1
        assert "opinion 0 (broken.pdf)" in errors[0].getMessage()
        assert errors[0].exc_info is not None, "no traceback: it exists nowhere else"
        assert any("done - 0 written, 1 failed" in r.getMessage() for r in caplog.records)
