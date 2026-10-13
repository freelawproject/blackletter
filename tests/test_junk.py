"""The boxes the model drew on no print (``blackletter.junk``).

A scanner's blot along a page edge, an image box over a caption the
model boxed already, and the mirrored key of the leaf's other side are
dropped before the pairing, the rects and the margins read the page,
in one pass (``drop_junk``). A box a person drew is never judged.
"""

from __future__ import annotations

from pathlib import Path

from blackletter import junk
from blackletter.models import Document, Label
from tests.pdf_fixtures import PAGE_H, PAGE_W, detected_page, detection

# A real key of this volume: 60 by 20 points, ratio 3, confident.
KEY = (60.0, 20.0)


def key(x0, y0, page_index=0, width=KEY[0], height=KEY[1], confidence=0.95):
    return detection(Label.KEY_ICON, x0, y0, x0 + width, y0 + height, page_index, confidence)


def document(*pages):
    return Document(pdf_path=Path("volume.pdf"), pages=list(pages), bl_warm=True)


def reference_keys(page_index=0):
    """Two confident keys in the text, the median the band rule reads."""
    return [key(100, 300, page_index), key(400, 500, page_index)]


class TestEdgeBands:
    def test_a_narrow_key_in_the_bottom_band_is_the_blot(self):
        blot = key(100, PAGE_H - 30, width=40, confidence=0.6)
        page = detected_page(reference_keys() + [blot])
        assert junk.band_junk(document(page)) == [blot]

    def test_a_sliver_ending_in_the_top_band_is_the_blot(self):
        # A third of a key's height, at the very top: the facing page's ghost.
        ghost = key(100, 2, width=45, height=7, confidence=0.5)
        page = detected_page(reference_keys() + [ghost])
        assert junk.band_junk(document(page)) == [ghost]

    def test_a_real_key_in_a_band_stays(self):
        # Full width and key-shaped: a tightly cropped volume can put a
        # real key in a band, and the rule needs both signs to drop one.
        low = key(100, PAGE_H - 30, confidence=0.6)
        page = detected_page(reference_keys() + [low])
        assert junk.band_junk(document(page)) == []

    def test_a_headnote_in_the_bottom_band_is_the_blot(self):
        blot = detection(Label.HEADNOTE, 100, PAGE_H - 20, 300, PAGE_H - 2, confidence=0.7)
        page = detected_page(reference_keys() + [blot])
        assert junk.band_junk(document(page)) == [blot]

    def test_a_key_in_the_text_is_not_read(self):
        page = detected_page(reference_keys())
        assert junk.band_junk(document(page)) == []

    def test_a_margin_label_is_never_read(self):
        number = detection(Label.PAGE_NUMBER, 100, PAGE_H - 20, 130, PAGE_H - 5)
        page = detected_page(reference_keys() + [number])
        assert junk.band_junk(document(page)) == []

    def test_a_persons_key_in_a_band_is_never_judged(self):
        drawn = key(100, PAGE_H - 30, width=40, confidence=1.0)
        page = detected_page(reference_keys() + [drawn])
        assert junk.band_junk(document(page)) == []


class TestTheCut:
    def test_a_column_running_into_the_bottom_band_is_cut_at_the_band_top(self):
        column = detection(Label.TEXT_COLUMN, 50, 100, 300, PAGE_H - 5, confidence=0.9)
        page = detected_page([column])
        cut = junk.cut_band_boxes(document(page))
        assert [(before, after.bbox.y2) for before, after in cut] == [(column, junk.band_top(page))]
        assert page.detections[0] is cut[0][1]
        assert page.detections[0].bbox.y1 == 100

    def test_a_box_that_ends_above_the_band_is_not_cut(self):
        column = detection(Label.TEXT_COLUMN, 50, 100, 300, 600)
        page = detected_page([column])
        assert junk.cut_band_boxes(document(page)) == []
        assert page.detections[0] is column

    def test_a_persons_box_is_not_cut(self):
        drawn = detection(Label.FOOTNOTES, 50, 600, 300, PAGE_H - 5, confidence=1.0)
        page = detected_page([drawn])
        assert junk.cut_band_boxes(document(page)) == []


class TestImageTwins:
    def test_an_image_over_a_caption_is_the_caption_boxed_twice(self):
        caption = detection(Label.CASE_CAPTION, 100, 100, 300, 200, confidence=0.45)
        image = detection(Label.IMAGE, 102, 101, 298, 199, confidence=0.8)
        page = detected_page([caption, image])
        assert junk.twin_images(document(page)) == [image]

    def test_an_image_inside_a_column_box_is_a_picture(self):
        column = detection(Label.TEXT_COLUMN, 50, 100, 300, 700)
        image = detection(Label.IMAGE, 60, 300, 290, 500)
        page = detected_page([column, image])
        assert junk.twin_images(document(page)) == []

    def test_an_image_beside_a_caption_is_a_picture(self):
        caption = detection(Label.CASE_CAPTION, 100, 100, 300, 200)
        image = detection(Label.IMAGE, 100, 400, 300, 600)
        page = detected_page([caption, image])
        assert junk.twin_images(document(page)) == []

    def test_a_persons_image_is_never_judged(self):
        caption = detection(Label.CASE_CAPTION, 100, 100, 300, 200)
        drawn = detection(Label.IMAGE, 100, 100, 300, 200, confidence=1.0)
        page = detected_page([caption, drawn])
        assert junk.twin_images(document(page)) == []


class TestShowThrough:
    def test_a_mirrored_key_on_a_page_with_no_column_is_show_through(self):
        real = key(100, 300, page_index=0)
        column = detection(Label.TEXT_COLUMN, 50, 100, 300, 700, page_index=0)
        ghost = key(PAGE_W - 160, 300, page_index=1, confidence=0.64)
        pages = [detected_page([real, column], 0), detected_page([ghost], 1)]
        assert junk.mirror_keys(document(*pages)) == [ghost]

    def test_a_mirrored_key_on_a_page_the_model_read_is_a_key(self):
        # Facing pages of a volume of orders line their keys up.
        left = key(100, 300, page_index=0)
        right = key(PAGE_W - 160, 300, page_index=1)
        columns = [detection(Label.TEXT_COLUMN, 50, 100, 300, 700, page_index=i) for i in (0, 1)]
        pages = [detected_page([left, columns[0]], 0), detected_page([right, columns[1]], 1)]
        assert junk.mirror_keys(document(*pages)) == []

    def test_a_key_with_no_mirror_stays(self):
        real = key(100, 300, page_index=0)
        lone = key(100, 500, page_index=1)
        pages = [detected_page([real], 0), detected_page([lone], 1)]
        assert junk.mirror_keys(document(*pages)) == []


class TestDropJunk:
    def test_the_pass_drops_and_cuts_and_reports(self):
        blot = key(100, PAGE_H - 30, width=40, confidence=0.6)
        column = detection(Label.TEXT_COLUMN, 50, 100, 300, PAGE_H - 5, confidence=0.9)
        caption = detection(Label.CASE_CAPTION, 100, 100, 300, 200)
        twin = detection(Label.IMAGE, 100, 100, 300, 200, confidence=0.8)
        page = detected_page(reference_keys() + [blot, column, caption, twin])
        report = junk.drop_junk(document(page))
        assert report.dropped == (blot, twin)
        assert [before for before, _ in report.cut] == [column]
        assert blot not in page.detections and twin not in page.detections
        assert caption in page.detections
        assert page.detections[-2].bbox.y2 == junk.band_top(page)

    def test_a_clean_document_is_left_alone(self):
        column = detection(Label.TEXT_COLUMN, 50, 100, 300, 600)
        page = detected_page(reference_keys() + [column])
        before = list(page.detections)
        report = junk.drop_junk(document(page))
        assert report == junk.JunkReport()
        assert page.detections == before
