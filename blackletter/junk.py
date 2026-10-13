"""Boxes the model drew on no print, dropped before anything reads them.

Three rules over a :class:`~blackletter.models.Document`, run once by
:func:`drop_junk` as soon as the document is built, so the pairing, the
redaction rects, the margin strips and the outside masks all stop
seeing a junk box at once, and a caller that draws the document draws
what the geometry read.

**The edge bands.** A scanner leaves a blot along the bottom edge of
some pages (a binding shadow, a page tab), and the model draws a box on
it: a ``KEY_ICON`` on 71 pages of one volume (scan 3692 of the scanning
app, 763 F. Appx), a ``HEADNOTE`` on 17 more. Every such box costs the
page its bottom margin strip, every such key is a key no opinion ends
at, and one of them, wide enough to pass the size filter, got paired
and stole the caption of the real key on its page. The top edge gets
the same blot, the ghost of the facing page's icon: a ``KEY_ICON``
sliver at the very top of five pages of 275 A.3d, a third the height
of a key, that cost each page its top strip and, read as a key inside
an opinion, named a caption nobody missed.

The rule reads the width and the place, nothing else, and it is one
rule at either edge. A real key or headnote heading of a reporter page
never starts in the bottom :data:`BAND` of the page (the text of that
volume ends at 0.93 of the height, the lowest real key top is at 0.88)
and never ends in the top :data:`BAND` (the running head is there, and
a key closes an opinion below it); the blot starts at 0.96 or lower,
and the ghost ends at 0.02. A real key is never narrower than 0.91 of
the median width of the volume's confident keys; the blot is never
wider than 0.83 of it, the ghost than 0.79. A key must fail both the
width and the shape, because a tightly cropped volume could put a real
key in a band. A margin label is never read: a reporter with its
number at the head or the foot would lose it.

A box that starts in the text and runs down onto the blot (a
``FOOTNOTES`` or a ``TEXT_COLUMN`` box on 56 pages of that volume,
ending anywhere from the band top to the page edge) is cut at the band
top (:func:`cut_band_boxes`): the box is real, its foot is not, and a
box that reaches the edge costs the page its bottom strip too.

**The image twin.** The model has no label for a table or a bordered
notice, and it draws an ``IMAGE`` box over some of them; it also draws
one over a case caption it boxed as a caption already (three pages of
that volume, at IoU 0.96 and up). An ``IMAGE`` box that covers a box
of any text label but ``TEXT_COLUMN`` (the column box holds everything
on its side) at :data:`IMAGE_TWIN_IOU` or more is a second reading of
the same print, not a picture (:func:`twin_images`): a consumer that
pastes the original scan of a picture over a bitonal page would paste
a photograph of the text.

**The show-through.** A thin or washed-out page shows the print of the
other side of its leaf, mirrored about the page's vertical axis, and
the model boxes a mirrored key icon as a key: two on page 792 of 459
P.3d, at 0.64 and at 0.96, each the mirror of a key on page 791 at the
same height. Each read as a key no caption precedes. Two signs, both
required (:func:`mirror_keys`): the box is the horizontal mirror of a
key on the page before or after it (:data:`MIRROR_IOU` after mirroring
about the page middle), and its page has no ``TEXT_COLUMN`` box,
because a page whose own print the model read does not show the other
side's. The mirror alone names real keys: in a volume of orders the
keys of facing pages line up, 1,206 such pairs over fourteen volumes,
and none of them on a page with no column box but those two.

**A person's box is never judged.** A detection at
:data:`HUMAN_CONFIDENCE` is one a curator drew or approved, which the
model never scores; it is neither dropped nor cut.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import median

from blackletter.models import BBox, Detection, Document, Label, Page
from blackletter.scanner import _KEY_ICON_MAX_RATIO, _KEY_ICON_MIN_RATIO

logger = logging.getLogger(__name__)

#: The share of the page height, from either edge, that holds no key
#: icon or headnote heading on a reporter page.
BAND = 0.06

#: A key narrower than this share of the median width of the volume's
#: confident keys is not a key.
KEY_WIDTH_FLOOR = 0.85

#: The confidence from which a key is one of the reference population
#: for the median key width, the population
#: ``scanner._filter_key_icons_by_size`` measures its median on.
CONFIDENT_KEY = 0.90

#: The labels the blot takes.
BAND_LABELS = frozenset({Label.KEY_ICON, Label.HEADNOTE})

#: An ``IMAGE`` box that covers a text box at this IoU or more is that
#: text, boxed twice.
IMAGE_TWIN_IOU = 0.5

#: The labels an ``IMAGE`` box is never a twin of: itself, and the
#: column box that holds everything on its side.
NOT_A_TWIN = frozenset({Label.IMAGE, Label.TEXT_COLUMN})

#: A key whose box, mirrored about the page's vertical axis, lands on a
#: key of the page before or after it at this IoU or more is the
#: show-through of that key, on a page with no column box.
MIRROR_IOU = 0.5

#: The confidence of a box a person drew or approved. The model never
#: scores a box this high, so such a box is never judged here.
HUMAN_CONFIDENCE = 1.0


@dataclass(frozen=True)
class JunkReport:
    """What :func:`drop_junk` did to a document.

    ``dropped`` holds the detections taken out of their pages, and
    ``cut`` one ``(before, after)`` pair per box cut at the band top,
    ``after`` being the detection that now stands in the page, so a
    caller that keys anything by the detection objects can re-key.
    """

    dropped: tuple[Detection, ...] = ()
    cut: tuple[tuple[Detection, Detection], ...] = ()


def is_human(detection: Detection) -> bool:
    """Return whether a box is a person's word, never the model's."""
    return detection.confidence >= HUMAN_CONFIDENCE


def band_top(page: Page) -> float:
    """Return the pixel row where the bottom band of ``page`` begins."""
    return page.img_height * (1.0 - BAND)


def band_bottom(page: Page) -> float:
    """Return the pixel row where the top band of ``page`` ends."""
    return page.img_height * BAND


def in_a_band(detection: Detection, page: Page) -> bool:
    """Return whether a box starts in the bottom band or ends in the top one."""
    if page.img_height <= 0:
        return False
    return detection.bbox.y1 >= band_top(page) or detection.bbox.y2 <= band_bottom(page)


def _key_shaped(detection: Detection) -> bool:
    """Return whether the box is inside the key ratio window."""
    height = detection.bbox.height
    ratio = detection.bbox.width / height if height > 0 else 0.0
    return _KEY_ICON_MIN_RATIO <= ratio <= _KEY_ICON_MAX_RATIO


def _all(document: Document):
    for page in document.pages:
        for detection in page.detections:
            yield page, detection


def median_key_width(document: Document) -> float | None:
    """Return the median width of the volume's keys.

    The confident keys when there are at least two of them, else every
    key, the population ``scanner._filter_key_icons_by_size`` measures
    on; None when there is no key at all.

    :param document: The document.
    :returns: The median width in pixels, or None.
    :rtype: float | None
    """
    keys = [d for _, d in _all(document) if d.label == Label.KEY_ICON]
    if not keys:
        return None
    confident = [d for d in keys if d.confidence >= CONFIDENT_KEY]
    reference = confident if len(confident) >= 2 else keys
    return float(median(d.bbox.width for d in reference))


def band_junk(document: Document) -> list[Detection]:
    """Return the model boxes the edge bands say are the blot.

    :param document: The document.
    :returns: The detections to drop, in page order.
    :rtype: list[Detection]
    """
    med = median_key_width(document)
    out: list[Detection] = []
    for page, d in _all(document):
        if d.label not in BAND_LABELS or is_human(d):
            continue
        if not in_a_band(d, page):
            continue
        if d.label == Label.KEY_ICON:
            narrow = med is not None and d.bbox.width < KEY_WIDTH_FLOOR * med
            if not narrow and _key_shaped(d):
                continue
        out.append(d)
    return out


def twin_images(document: Document) -> list[Detection]:
    """Return the model ``IMAGE`` boxes that are a text box's twin.

    :param document: The document.
    :returns: The detections to drop, in page order.
    :rtype: list[Detection]
    """
    out: list[Detection] = []
    for page in document.pages:
        texts = [d for d in page.detections if d.label not in NOT_A_TWIN]
        for d in page.detections:
            if d.label != Label.IMAGE or is_human(d):
                continue
            if any(d.bbox.iou(other.bbox) >= IMAGE_TWIN_IOU for other in texts):
                out.append(d)
    return out


def _mirror(detection: Detection, page: Page) -> BBox:
    """Return the box mirrored about the vertical axis of its page."""
    return BBox(
        x1=page.img_width - detection.bbox.x2,
        y1=detection.bbox.y1,
        x2=page.img_width - detection.bbox.x1,
        y2=detection.bbox.y2,
    )


def mirror_keys(document: Document) -> list[Detection]:
    """Return the model keys that are the show-through of the leaf's other side.

    :param document: The document.
    :returns: The detections to drop, in page order.
    :rtype: list[Detection]
    """
    keys: dict[int, list[Detection]] = {}
    columns: set[int] = set()
    pages: dict[int, Page] = {}
    for page, d in _all(document):
        pages[page.index] = page
        if d.label == Label.TEXT_COLUMN:
            columns.add(page.index)
        elif d.label == Label.KEY_ICON:
            keys.setdefault(page.index, []).append(d)
    out: list[Detection] = []
    for index in sorted(keys):
        if index in columns:
            continue
        neighbours = keys.get(index - 1, []) + keys.get(index + 1, [])
        page = pages[index]
        if not neighbours or page.img_width <= 0:
            continue
        for d in keys[index]:
            if is_human(d):
                continue
            mirror = _mirror(d, page)
            if any(mirror.iou(other.bbox) >= MIRROR_IOU for other in neighbours):
                out.append(d)
    return out


def junk(document: Document) -> list[Detection]:
    """Return every box the three rules drop, in page order, each once.

    :param document: The document.
    :returns: The detections.
    :rtype: list[Detection]
    """
    found = {id(d): d for d in band_junk(document)}
    found.update((id(d), d) for d in twin_images(document))
    found.update((id(d), d) for d in mirror_keys(document))
    return [d for _, d in _all(document) if id(d) in found]


def cut_band_boxes(document: Document) -> list[tuple[Detection, Detection]]:
    """Cut every model box that runs from the text into the bottom band.

    A box whose top is above the band and whose foot is inside it gets
    the band top for its foot: the detection is replaced in its page by
    one with the new box, in the same place in the list. A box that
    starts in the band is :func:`junk`'s question.

    :param document: The document, changed in place.
    :returns: ``(before, after)`` per box cut, in page order.
    :rtype: list[tuple[Detection, Detection]]
    """
    cut: list[tuple[Detection, Detection]] = []
    for page in document.pages:
        if page.img_height <= 0:
            continue
        top = band_top(page)
        for i, d in enumerate(page.detections):
            if is_human(d) or not (d.bbox.y1 < top < d.bbox.y2):
                continue
            after = replace(d, bbox=replace(d.bbox, y2=top))
            page.detections[i] = after
            cut.append((d, after))
    return cut


def drop_junk(document: Document) -> JunkReport:
    """Take the junk out of a document and cut the boxes on the blot.

    **The one pass**, run where a document is built and before anything
    reads it: :func:`junk` leaves the pages, then
    :func:`cut_band_boxes` runs over what is left.

    :param document: The document, changed in place.
    :returns: What was dropped and what was cut.
    :rtype: JunkReport
    """
    dropped = junk(document)
    if dropped:
        gone = {id(d) for d in dropped}
        for page in document.pages:
            page.detections[:] = [d for d in page.detections if id(d) not in gone]
    cut = cut_band_boxes(document)
    if dropped or cut:
        logger.info(
            "%s: %d junk box(es) dropped (an edge band, an image twin, a "
            "show-through), %d box(es) cut at the band top",
            Path(document.pdf_path).name if document.pdf_path else "document",
            len(dropped),
            len(cut),
        )
    return JunkReport(dropped=tuple(dropped), cut=tuple(cut))
