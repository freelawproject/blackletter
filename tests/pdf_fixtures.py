"""Builders for small synthetic PDFs used by the geometry tests.

The PDFs this library is handed come in two flavors and the geometry code
has to work with both: a page with a real text layer (``api.ocr`` ran over
it), and a ``bitonal``-style page that is a single 1-bit image with no text
at all. These helpers produce both, plus the scanner artifacts (a solid
band along a page edge, bleed-through from the facing page) that the ink
measurements have to ignore.
"""

from __future__ import annotations

import io
from pathlib import Path

import fitz
from PIL import Image

from blackletter.models import BBox, Detection, Label, Page

PAGE_W, PAGE_H = 612.0, 792.0

# Box the synthetic body text is laid out in, in PDF points.
CONTENT = fitz.Rect(72, 100, 540, 700)

BODY_SENTENCE = (
    "The judgment of the district court is affirmed in part and reversed "
    "in part, and the cause is remanded for further proceedings. "
)

# Body text is drawn line by line rather than with ``insert_textbox``,
# which inserts nothing at all when the text overflows its rect. Lines are
# laid out on this leading so they fill CONTENT top to bottom, and each is
# cut to CONTENT's width, so a clip over any part of the box lands on text.
LINE_LEADING = 12.0
FONT_SIZE = 9.0

# A single line of text above the body block, like a running head. The gap
# between it and the body is blank but sits *inside* the page's content
# box, which is where empty redaction rects used to come from.
HEADER_LINE_Y = 58.0
HEADER_TEXT = "FRATERNAL ORDER OF POLICE v. BOARD OF GOVERNORS"

# A solid black band along the top edge, like a platen shadow.
TOP_BAR = fitz.Rect(0, 4, PAGE_W, 12)

# A small blob at the top-right corner, like a page number bleeding through
# from the facing page. Close enough to the header that ink alone treats it
# as content, which is what suppressed the top margin strip.
BLEED_MARK = fitz.Rect(470, 2, 530, 14)

# A speck in the left margin, close enough to the text that the ink content
# box takes it in (further out and ``ink.content_box`` discards it as its
# own low-mass run), which is how a real gutter smudge widens the box and
# shrinks the strip on that side.
STRAY_MARK = fitz.Rect(45, 400, 52, 407)

# A page number printed in the outer corner, outside the text columns --
# real content that a tightened side strip must not reach.
CORNER_NUMBER_X = 40.0

# ...and along the bottom edge, where it sits well below the last line of
# text (the case that stretched ink measurements to the page edge).
BOTTOM_BAR = fitz.Rect(120, PAGE_H - 10, 500, PAGE_H - 4)

# A solid band that overlaps the top of the first line of text, like a
# gutter shadow reaching into the type. The overlap is the point: a band
# merely adjacent to the text still leaves a row or two of white before the
# glyph tops, and growth stops there whether or not anything clamps it. The content
# box excludes it (a mostly-dark row is not text) while the ink runs
# continuously across that boundary, so growth can only be stopped by the
# clamp rather than by white space.
TOUCHING_BAR = fitz.Rect(0, CONTENT.y0 - 8, PAGE_W, CONTENT.y0 + 6)

# An image inside the text column but below the last line, like a key icon
# at the foot of a page. margins._text_bounds extends the content box
# vertically to take these in.
IMAGE_BLOCK = fitz.Rect(200, CONTENT.y1 + 12, 240, CONTENT.y1 + 52)


def _line_for_width(width: float) -> str:
    """Return a body-text line that just fits ``width`` at FONT_SIZE.

    :param width: Target line width in PDF points.
    :return: The line text.
    """
    text = BODY_SENTENCE
    while fitz.get_text_length(text, fontsize=FONT_SIZE) < width:
        text += BODY_SENTENCE
    while fitz.get_text_length(text, fontsize=FONT_SIZE) > width:
        text = text[:-1]
    return text


def write_text_page(
    path: Path,
    top_bar: bool = False,
    bottom_bar: bool = False,
    header_line: bool = False,
    bleed_mark: bool = False,
    stray_mark: bool = False,
    corner_number: bool = False,
    image_block: bool = False,
    touching_bar: bool = False,
) -> None:
    """Write a one-page PDF with real text filling :data:`CONTENT`.

    :param path: Where to write the PDF.
    :param top_bar: Paint :data:`TOP_BAR`.
    :param bottom_bar: Paint :data:`BOTTOM_BAR`.
    :param header_line: Also draw :data:`HEADER_TEXT` above the body, so
        the page has a blank gap inside its content box.
    :param bleed_mark: Paint :data:`BLEED_MARK`.
    :param stray_mark: Paint :data:`STRAY_MARK`.
    :param corner_number: Print a page number at :data:`CORNER_NUMBER_X`,
        outside the text columns.
    :param image_block: Place an image below the text block, inside the
        text column, like a key icon at the foot of a page. Only the text
        layer sees it as an image block.
    :param touching_bar: Paint :data:`TOUCHING_BAR`.
    """
    line = _line_for_width(CONTENT.width)
    with fitz.open() as doc:
        page = doc.new_page(width=PAGE_W, height=PAGE_H)
        if header_line:
            page.insert_text((CONTENT.x0, HEADER_LINE_Y), HEADER_TEXT, fontsize=FONT_SIZE)
        if corner_number:
            page.insert_text((CORNER_NUMBER_X, HEADER_LINE_Y), "12", fontsize=FONT_SIZE)
        y = CONTENT.y0 + FONT_SIZE
        while y <= CONTENT.y1:
            page.insert_text((CONTENT.x0, y), line, fontsize=FONT_SIZE)
            y += LINE_LEADING
        if image_block:
            img = Image.new("L", (40, 40), color=0)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            page.insert_image(IMAGE_BLOCK, stream=buf.getvalue())
        for draw, rect in (
            (top_bar, TOP_BAR),
            (bottom_bar, BOTTOM_BAR),
            (bleed_mark, BLEED_MARK),
            (stray_mark, STRAY_MARK),
            (touching_bar, TOUCHING_BAR),
        ):
            if draw:
                page.draw_rect(rect, fill=(0, 0, 0), width=0)
        doc.save(str(path))


def rasterize(src: Path, dst: Path) -> None:
    """Rebuild a PDF as one 1-bit image per page, like a bitonal scan.

    The result has no text layer, so anything measuring the page has to
    read its pixels.

    :param src: Source PDF to rasterize.
    :param dst: Where to write the rasterized PDF.
    """
    with fitz.open(str(src)) as doc, fitz.open() as out:
        for page_idx in range(len(doc)):
            src_page = doc[page_idx]
            pix = src_page.get_pixmap(dpi=200, colorspace=fitz.csGRAY)
            img = Image.frombytes("L", (pix.width, pix.height), pix.samples).convert("1")
            buf = io.BytesIO()
            img.save(buf, format="TIFF", compression="group4")
            page = out.new_page(width=src_page.rect.width, height=src_page.rect.height)
            page.insert_image(page.rect, stream=buf.getvalue())
        out.save(str(dst))


def write_bitonal_page(
    path: Path,
    top_bar: bool = False,
    bottom_bar: bool = False,
    header_line: bool = False,
    bleed_mark: bool = False,
    stray_mark: bool = False,
    corner_number: bool = False,
    touching_bar: bool = False,
    tmp_dir: Path | None = None,
) -> None:
    """Write a text-less, 1-bit version of :func:`write_text_page`.

    :param path: Where to write the PDF.
    :param top_bar: Paint :data:`TOP_BAR` before rasterizing.
    :param bottom_bar: Paint :data:`BOTTOM_BAR` before rasterizing.
    :param header_line: Draw :data:`HEADER_TEXT` above the body.
    :param bleed_mark: Paint :data:`BLEED_MARK` before rasterizing.
    :param stray_mark: Paint :data:`STRAY_MARK` before rasterizing.
    :param corner_number: Print a page number outside the text columns.
    :param touching_bar: Paint :data:`TOUCHING_BAR` before rasterizing.
    :param tmp_dir: Directory for the intermediate text PDF. Defaults to
        ``path``'s parent.
    """
    tmp_dir = tmp_dir or path.parent
    src = tmp_dir / f"{path.stem}.text.pdf"
    write_text_page(
        src,
        top_bar=top_bar,
        bottom_bar=bottom_bar,
        header_line=header_line,
        bleed_mark=bleed_mark,
        stray_mark=stray_mark,
        corner_number=corner_number,
        touching_bar=touching_bar,
    )
    rasterize(src, path)


def write_multi_page(
    path: Path,
    kinds: list[str],
    tmp_dir: Path | None = None,
) -> None:
    """Write a text-less PDF whose pages differ from one another.

    Single-page fixtures cannot catch a per-page cache that returns the
    wrong page's measurement, so this builds pages that disagree: a
    ``"body"`` page has a full text block, ``"narrow"`` has only a couple
    of characters (so its content box is unmeasurable), and ``"blank"`` has
    nothing at all.

    :param path: Where to write the PDF.
    :param kinds: One entry per page: ``"body"``, ``"narrow"`` or
        ``"blank"``.
    :param tmp_dir: Directory for the intermediate text PDF. Defaults to
        ``path``'s parent.
    """
    tmp_dir = tmp_dir or path.parent
    src = tmp_dir / f"{path.stem}.text.pdf"
    line = _line_for_width(CONTENT.width)
    with fitz.open() as doc:
        for kind in kinds:
            page = doc.new_page(width=PAGE_W, height=PAGE_H)
            if kind == "body":
                y = CONTENT.y0 + FONT_SIZE
                while y <= CONTENT.y1:
                    page.insert_text((CONTENT.x0, y), line, fontsize=FONT_SIZE)
                    y += LINE_LEADING
            elif kind == "narrow":
                page.insert_text((280, 400), "12", fontsize=FONT_SIZE)
            elif kind != "blank":
                raise ValueError(f"unknown page kind: {kind}")
        doc.save(str(src))
    rasterize(src, path)


def detection(
    label: Label,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    page_index: int = 0,
    confidence: float = 0.9,
) -> Detection:
    """Build a :class:`~blackletter.models.Detection`.

    Meant to be handed to :func:`detected_page`, whose image dimensions
    equal the page size in points, so bbox values in tests read as points.

    :param label: Detection label.
    :param x0: Left edge.
    :param y0: Top edge.
    :param x1: Right edge.
    :param y1: Bottom edge.
    :param page_index: 0-based page index.
    :param confidence: Detection confidence.
    :return: The detection.
    """
    return Detection(
        bbox=BBox(x1=x0, y1=y0, x2=x1, y2=y1),
        label=label,
        confidence=confidence,
        page_index=page_index,
    )


def detected_page(detections: list[Detection], page_index: int = 0) -> Page:
    """Wrap detections in a :class:`~blackletter.models.Page`.

    Image dimensions are set to the page size in points, so ``scale_x`` and
    ``scale_y`` are 1 and bbox values read as PDF points.

    :param detections: Detections found on the page.
    :param page_index: 0-based page index.
    :return: The page.
    """
    return Page(
        index=page_index,
        pdf_width=PAGE_W,
        pdf_height=PAGE_H,
        img_width=int(PAGE_W),
        img_height=int(PAGE_H),
        detections=list(detections),
    )


# Two text bands with a hairline gutter between them, like a tightly set
# reporter page. A gutter this narrow is what let an ink measurement walk
# straight across it into the neighbouring column.
COLUMN_GAP = 1.5
COLUMN_LEFT = fitz.Rect(72, 100, 300, 700)
COLUMN_RIGHT = fitz.Rect(COLUMN_LEFT.x1 + COLUMN_GAP, 100, 540, 700)


def write_two_column_text_page(path: Path, tmp_dir: Path | None = None) -> None:
    """Write a text-less, 1-bit page with two columns of *typeset* lines.

    The companion to :func:`write_two_column_page`, which draws each row as
    a solid filled rect. Solid rows are 76% dark, above
    ``ink.CONTENT_MAX_FRACTION``, so every content row there reads as a
    solid bar and the page's content box comes out artifact-driven. Real
    text is sparse enough to measure, and each row here carries a different
    slice of the sentence so inter-character gaps do not line up into
    full-height white channels the way they do in :func:`write_text_page`.

    :param path: Where to write the PDF.
    :param tmp_dir: Directory for the intermediate PDF. Defaults to
        ``path``'s parent.
    """
    tmp_dir = tmp_dir or path.parent
    src = tmp_dir / f"{path.stem}.text.pdf"
    sentence = BODY_SENTENCE * 4
    with fitz.open() as doc:
        page = doc.new_page(width=PAGE_W, height=PAGE_H)
        for band in (COLUMN_LEFT, COLUMN_RIGHT):
            y = band.y0 + FONT_SIZE
            row = 0
            while y <= band.y1:
                # A different starting offset per row de-aligns the gaps.
                start = (row * 7) % (len(sentence) // 2)
                text = sentence[start:]
                while fitz.get_text_length(text, fontsize=FONT_SIZE) > band.width:
                    text = text[:-1]
                page.insert_text((band.x0, y), text, fontsize=FONT_SIZE)
                y += LINE_LEADING
                row += 1
        doc.save(str(src))
    rasterize(src, path)


def write_hairline_column_page(
    path: Path, gutter: float = 0.3, tmp_dir: Path | None = None
) -> tuple[fitz.Rect, fitz.Rect]:
    """Two columns of real type separated by a sub-pixel gutter.

    At 100 dpi a pixel spans 0.72 pt, so a gutter narrower than that falls
    inside a single pixel column. A growth walk starts past that column (see
    ``ink._outside_after``) and the next ink it meets belongs to the facing
    column. Type rather than solid bars, because with solid bars the walk
    runs to its margin and is refused, which hides the case.

    :param path: Where to write the PDF.
    :param gutter: Width of the gutter, in points.
    :param tmp_dir: Directory for the intermediate PDF. Defaults to
        ``path``'s parent.
    :returns: The two column bands, in PDF points.
    """
    tmp_dir = tmp_dir or path.parent
    left = fitz.Rect(110.6, 100, 311.8, 700)
    right = fitz.Rect(311.8 + gutter, 100, 527.3, 700)
    src = tmp_dir / f"{path.stem}.text.pdf"
    sentence = BODY_SENTENCE * 4
    with fitz.open() as doc:
        page = doc.new_page(width=PAGE_W, height=PAGE_H)
        for band in (left, right):
            y, row = band.y0 + FONT_SIZE, 0
            while y <= band.y1:
                text = sentence[(row * 11) % 80 :]
                while fitz.get_text_length(text, fontsize=FONT_SIZE) > band.width:
                    text = text[:-1]
                page.insert_text((band.x0, y), text, fontsize=FONT_SIZE)
                y += LINE_LEADING
                row += 1
        doc.save(str(src))
    rasterize(src, path)
    return left, right


def write_two_column_page(path: Path, tmp_dir: Path | None = None) -> None:
    """Write a text-less, 1-bit page holding two columns of text lines.

    Lines are drawn as thin filled rects rather than glyphs so the bands
    have exact, predictable edges.

    :param path: Where to write the PDF.
    :param tmp_dir: Directory for the intermediate PDF. Defaults to
        ``path``'s parent.
    """
    tmp_dir = tmp_dir or path.parent
    src = tmp_dir / f"{path.stem}.text.pdf"
    with fitz.open() as doc:
        page = doc.new_page(width=PAGE_W, height=PAGE_H)
        for band in (COLUMN_LEFT, COLUMN_RIGHT):
            y = band.y0
            while y < band.y1:
                page.draw_rect(
                    fitz.Rect(band.x0, y, band.x1, y + 4),
                    color=None,
                    fill=(0, 0, 0),
                    width=0,
                )
                y += LINE_LEADING
        doc.save(str(src))
    rasterize(src, path)
