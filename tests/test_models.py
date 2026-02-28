from pathlib import Path

from blackletter.models import BBox, Detection, Document, Label, Page


# ── Label ────────────────────────────────────────────────────────────────

class TestLabel:
    def test_int_values_match_yolo_class_ids(self):
        assert Label.KEY_ICON == 0
        assert Label.HEADNOTE == 11
        assert len(Label) == 13

    def test_copyrighted_labels(self):
        assert Label.KEY_ICON.is_copyrighted
        assert Label.PAGE_HEADER.is_copyrighted
        assert Label.HEADNOTE_BRACKET.is_copyrighted
        assert Label.HEADNOTE.is_copyrighted

    def test_non_copyrighted_labels(self):
        for label in Label:
            if label not in (Label.KEY_ICON, Label.PAGE_HEADER,
                             Label.HEADNOTE_BRACKET, Label.HEADNOTE):
                assert not label.is_copyrighted, f"{label.name} should not be copyrighted"

    def test_structural_labels(self):
        assert Label.CASE_CAPTION.is_structural
        assert Label.DIVIDER.is_structural
        assert Label.PAGE_NUMBER.is_structural
        assert Label.STATE_ABBREVIATION.is_structural
        assert not Label.KEY_ICON.is_structural
        assert not Label.HEADNOTE.is_structural

    def test_from_int(self):
        """Can construct from YOLO integer class ID."""
        assert Label(3) is Label.CASE_CAPTION
        assert Label(0) is Label.KEY_ICON


# ── BBox ─────────────────────────────────────────────────────────────────

class TestBBox:
    def test_from_xyxy(self):
        box = BBox.from_xyxy([10.0, 20.0, 110.0, 70.0])
        assert box.x1 == 10.0
        assert box.y1 == 20.0
        assert box.x2 == 110.0
        assert box.y2 == 70.0

    def test_dimensions(self):
        box = BBox(0, 0, 100, 50)
        assert box.width == 100
        assert box.height == 50
        assert box.area == 5000

    def test_center(self):
        box = BBox(10, 20, 110, 70)
        assert box.center_x == 60.0
        assert box.center_y == 45.0

    def test_to_pdf(self):
        box = BBox(100, 200, 300, 400)
        pdf_box = box.to_pdf(scale_x=0.5, scale_y=0.5)
        assert pdf_box.x1 == 50.0
        assert pdf_box.y1 == 100.0
        assert pdf_box.x2 == 150.0
        assert pdf_box.y2 == 200.0

    def test_iou_identical(self):
        box = BBox(0, 0, 100, 100)
        assert box.iou(box) == 1.0

    def test_iou_no_overlap(self):
        a = BBox(0, 0, 50, 50)
        b = BBox(100, 100, 200, 200)
        assert a.iou(b) == 0.0

    def test_iou_partial_overlap(self):
        a = BBox(0, 0, 100, 100)
        b = BBox(50, 50, 150, 150)
        # intersection = 50*50 = 2500
        # union = 10000 + 10000 - 2500 = 17500
        assert abs(a.iou(b) - 2500 / 17500) < 1e-9

    def test_contains(self):
        outer = BBox(0, 0, 200, 200)
        inner = BBox(10, 10, 50, 50)
        assert outer.contains(inner)
        assert not inner.contains(outer)

    def test_contains_partial(self):
        a = BBox(0, 0, 100, 100)
        b = BBox(50, 50, 120, 120)  # 50*50=2500 overlap, b area=4900
        assert not a.contains(b, threshold=0.8)
        assert a.contains(b, threshold=0.5)

    def test_frozen(self):
        box = BBox(0, 0, 100, 100)
        try:
            box.x1 = 50  # type: ignore
            assert False, "Should not allow mutation"
        except AttributeError:
            pass


# ── Detection ────────────────────────────────────────────────────────────

def _det(label=Label.CASE_CAPTION, x1=100, y1=200, x2=400, y2=250,
         conf=0.9, page=0):
    return Detection(
        bbox=BBox(x1, y1, x2, y2),
        label=label,
        confidence=conf,
        page_index=page,
    )


class TestDetection:
    def test_is_copyrighted(self):
        assert _det(Label.KEY_ICON).is_copyrighted
        assert not _det(Label.CASE_CAPTION).is_copyrighted

    def test_column_left(self):
        d = _det(x1=10, x2=100)  # center_x = 55
        assert d.column(midpoint=300) == "LEFT"

    def test_column_right(self):
        d = _det(x1=400, x2=600)  # center_x = 500
        assert d.column(midpoint=300) == "RIGHT"

    def test_sort_key_orders_by_page_col_y(self):
        d1 = _det(page=0, x1=10, x2=100, y1=50, y2=80)   # p0, LEFT, y=50
        d2 = _det(page=0, x1=400, x2=600, y1=10, y2=40)  # p0, RIGHT, y=10
        d3 = _det(page=1, x1=10, x2=100, y1=10, y2=40)   # p1, LEFT, y=10
        mid = 300.0
        assert d1.sort_key(mid) < d2.sort_key(mid) < d3.sort_key(mid)

    def test_frozen(self):
        d = _det()
        try:
            d.label = Label.DIVIDER  # type: ignore
            assert False, "Should not allow mutation"
        except AttributeError:
            pass


# ── Page ─────────────────────────────────────────────────────────────────

def _page(index=0, detections=None):
    return Page(
        index=index,
        pdf_width=612.0,  # standard letter
        pdf_height=792.0,
        img_width=1700,   # ~200 dpi
        img_height=2200,
        detections=detections or [],
    )


class TestPage:
    def test_scale_factors(self):
        p = _page()
        assert abs(p.scale_x - 612.0 / 1700) < 1e-9
        assert abs(p.scale_y - 792.0 / 2200) < 1e-9

    def test_midpoint(self):
        p = _page()
        assert p.midpoint == 850.0

    def test_by_label(self):
        dets = [
            _det(Label.CASE_CAPTION),
            _det(Label.KEY_ICON),
            _det(Label.CASE_CAPTION),
            _det(Label.HEADNOTE),
        ]
        p = _page(detections=dets)
        assert len(p.by_label(Label.CASE_CAPTION)) == 2
        assert len(p.by_label(Label.KEY_ICON, Label.HEADNOTE)) == 2

    def test_copyrighted(self):
        dets = [
            _det(Label.KEY_ICON),
            _det(Label.CASE_CAPTION),
            _det(Label.HEADNOTE),
        ]
        p = _page(detections=dets)
        assert len(p.copyrighted) == 2

    def test_reading_order(self):
        left_low = _det(x1=10, x2=100, y1=500, y2=550)
        left_high = _det(x1=10, x2=100, y1=50, y2=80)
        right = _det(x1=1000, x2=1200, y1=50, y2=80)
        p = _page(detections=[left_low, right, left_high])
        ordered = p.in_reading_order()
        assert ordered[0] is left_high  # LEFT, y=50
        assert ordered[1] is left_low   # LEFT, y=500
        assert ordered[2] is right      # RIGHT, y=50


# ── Document ─────────────────────────────────────────────────────────────

class TestDocument:
    def test_captions(self):
        p0 = _page(index=0, detections=[
            _det(Label.CASE_CAPTION, x1=10, x2=100, y1=200, y2=250, page=0),
            _det(Label.KEY_ICON, page=0),
        ])
        p1 = _page(index=1, detections=[
            _det(Label.CASE_CAPTION, x1=10, x2=100, y1=100, y2=150, page=1),
        ])
        doc = Document(pdf_path=Path("test.pdf"), pages=[p0, p1])
        assert len(doc.captions) == 2
        assert doc.captions[0].page_index == 0
        assert doc.captions[1].page_index == 1

    def test_copyrighted(self):
        p = _page(detections=[
            _det(Label.KEY_ICON),
            _det(Label.PAGE_HEADER),
            _det(Label.CASE_CAPTION),
            _det(Label.FOOTNOTES),
        ])
        doc = Document(pdf_path=Path("test.pdf"), pages=[p])
        assert len(doc.copyrighted) == 2

    def test_by_label(self):
        p = _page(detections=[
            _det(Label.PAGE_NUMBER, x1=10, x2=50, y1=10, y2=30),
            _det(Label.STATE_ABBREVIATION, x1=1000, x2=1100, y1=10, y2=30),
            _det(Label.CASE_CAPTION, x1=10, x2=100, y1=200, y2=250),
        ])
        doc = Document(pdf_path=Path("test.pdf"), pages=[p])
        assert len(doc.by_label(Label.PAGE_NUMBER)) == 1
        assert len(doc.by_label(Label.PAGE_NUMBER, Label.STATE_ABBREVIATION)) == 2

    def test_empty_document(self):
        doc = Document(pdf_path=Path("test.pdf"))
        assert doc.captions == []
        assert doc.copyrighted == []
        assert doc.by_label(Label.KEY_ICON) == []
