"""Tests that measure the pixels of what actually gets written.

Everything else here checks geometry: rect coordinates, bounds, decisions.
These two call the functions that *apply* that geometry to a PDF and then
render the result, because both have failure modes no coordinate assertion
can see. ``clean_margins`` handles bitonal images through a separate code
path (``apply_redactions`` corrupts CCITT G4 streams), and the hairline
``apply_redactions`` leaves at a rect boundary is a stroke that no rect in
the payload describes.
"""

from __future__ import annotations

import fitz
import numpy as np
import pytest

from blackletter.api import build_redactions, generate
from blackletter.margins import clean_margins
from blackletter.models import Label
from tests.pdf_fixtures import (
    BOTTOM_BAR,
    CONTENT,
    TOP_BAR,
    detected_page,
    detection,
    write_bitonal_page,
)

DPI = 150


def dark_fraction(page, rect: fitz.Rect) -> float:
    """Fraction of dark pixels inside a region of a rendered page."""
    pix = page.get_pixmap(dpi=DPI, colorspace=fitz.csGRAY, clip=rect)
    gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)[
        :, : pix.width
    ]
    return float((gray < 128).mean()) if gray.size else 0.0


class TestCleanMarginsOnBitonal:
    """The bitonal path: a 1-bit CCITT image, redacted through pixels.

    ``apply_redactions`` corrupts those streams, so ``clean_margins``
    rewrites the image data instead. Nothing else exercises that, and a
    coordinate test cannot tell a whited-out artifact from an untouched one.
    """

    @pytest.fixture
    def bitonal(self, tmp_path):
        path = tmp_path / "vol.pdf"
        write_bitonal_page(path, top_bar=True, bottom_bar=True, header_line=True, tmp_dir=tmp_path)
        return path

    def test_edge_artifacts_are_erased_and_the_text_survives(self, bitonal, tmp_path):
        out = tmp_path / "cleaned.pdf"
        with fitz.open(str(bitonal)) as doc:
            before_top = dark_fraction(doc[0], TOP_BAR)
            before_bottom = dark_fraction(doc[0], BOTTOM_BAR)
            before_body = dark_fraction(doc[0], CONTENT)
        assert before_top > 0.5, "fixture has no top bar to remove"
        assert before_bottom > 0.5, "fixture has no bottom bar to remove"

        pages = [
            detected_page(
                [detection(Label.TEXT_COLUMN, CONTENT.x0, CONTENT.y0, CONTENT.x1, CONTENT.y1)]
            )
        ]
        clean_margins(bitonal, output_path=out, pages=pages)

        with fitz.open(str(out)) as doc:
            after_top = dark_fraction(doc[0], TOP_BAR)
            after_bottom = dark_fraction(doc[0], BOTTOM_BAR)
            after_body = dark_fraction(doc[0], CONTENT)
        assert after_top < 0.01, f"top bar survived ({after_top:.3f} dark)"
        assert after_bottom < 0.01, f"bottom bar survived ({after_bottom:.3f} dark)"
        assert after_body == pytest.approx(before_body, rel=0.05), "body text was damaged"

    def test_the_image_is_still_a_readable_bitonal_pdf(self, bitonal, tmp_path):
        """Guards the CCITT-safe path: a corrupt stream renders as nothing."""
        out = tmp_path / "cleaned.pdf"
        clean_margins(bitonal, output_path=out)
        with fitz.open(str(out)) as doc:
            page = doc[0]
            assert page.get_images(full=True), "the page lost its image"
            assert dark_fraction(page, CONTENT) > 0.01, "the page rendered blank"

    def test_in_place_leaves_no_temp_file_behind(self, bitonal, tmp_path):
        before = {p.name for p in tmp_path.iterdir()}
        clean_margins(bitonal)
        assert {p.name for p in tmp_path.iterdir()} == before


class TestGenerateSeamsAreClean:
    """No dark line where a black rect meets a white one.

    PyMuPDF has painted redactions as a fill plus a 1pt stroke straddling
    the edge, strokes last, which left a hairline at such a seam in real
    deliverables. It does not reproduce on 1.26.7: removing ``generate``'s
    fill-only overdraw leaves this test green, so what is pinned here is the
    invariant, not the overdraw. Worth keeping either way, since a rendered
    seam is the only place the defect was ever visible.
    """

    @pytest.fixture
    def volume(self, tmp_path):
        path = tmp_path / "vol.pdf"
        write_bitonal_page(path, tmp_dir=tmp_path)
        return path

    @staticmethod
    def _payload(seam_x: float):
        """A black rect meeting a white one at ``seam_x``."""
        page = detected_page([])
        opinions = [
            {
                "caption_page": 0,
                "key_page": 0,
                "end_page": 0,
                "first_page_number": 1,
                "last_page_number": 1,
                "outside_rects": [],
            }
        ]
        rects = [
            {
                "page_index": 0,
                "rects": [
                    {
                        "x0": CONTENT.x0,
                        "y0": 200,
                        "x1": seam_x,
                        "y1": 400,
                        "fill": "black",
                        "type": "headnote",
                    },
                    {
                        "x0": seam_x,
                        "y0": 200,
                        "x1": CONTENT.x1,
                        "y1": 400,
                        "fill": "white",
                        "type": "PAGE_HEADER",
                    },
                ],
            }
        ]
        return build_redactions([page], rects, [], opinions, reporter="a3d", volume="222")

    def test_no_dark_line_survives_on_the_white_side_of_a_seam(self, volume, tmp_path):
        seam_x = 300.0
        out = tmp_path / "out"
        generate(volume, self._payload(seam_x), out, reporter="a3d", volume="222")

        full = next(out.glob("*.redacted.pdf"))
        with fitz.open(str(full)) as doc:
            page = doc[0]
            # A 2pt strip just inside the white rect, where the black rect's
            # stroke would land.
            strip = fitz.Rect(seam_x + 0.2, 220, seam_x + 2.2, 380)
            leaked = dark_fraction(page, strip)
            # ...and the black rect is still black, so this is not passing
            # because nothing was painted at all.
            filled = dark_fraction(page, fitz.Rect(CONTENT.x0 + 5, 220, seam_x - 5, 380))
        assert leaked < 0.02, f"a hairline survived at the seam ({leaked:.3f} dark)"
        assert filled > 0.95, "the black rect was not painted"
