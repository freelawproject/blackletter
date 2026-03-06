from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path


class Label(IntEnum):
    """YOLO detection classes matching the trained model."""

    KEY_ICON = 0
    DIVIDER = 1
    PAGE_HEADER = 2
    CASE_CAPTION = 3
    FOOTNOTES = 4
    HEADNOTE_BRACKET = 5
    CASE_METADATA = 6
    CASE_SEQUENCE = 7
    PAGE_NUMBER = 8
    STATE_ABBREVIATION = 9
    IMAGE = 10
    HEADNOTE = 11
    BACKGROUND = 12
    SYLLABUS = 13
    EDGES = 14

    @property
    def is_copyrighted(self) -> bool:
        return self in _COPYRIGHTED

    @property
    def is_structural(self) -> bool:
        """Labels that define page/opinion structure rather than content."""
        return self in _STRUCTURAL


_COPYRIGHTED = frozenset(
    {
        Label.KEY_ICON,
        Label.PAGE_HEADER,
        Label.HEADNOTE_BRACKET,
        Label.HEADNOTE,
    }
)

_STRUCTURAL = frozenset(
    {
        Label.CASE_CAPTION,
        Label.DIVIDER,
        Label.CASE_METADATA,
        Label.CASE_SEQUENCE,
        Label.PAGE_NUMBER,
        Label.STATE_ABBREVIATION,
    }
)


@dataclass(frozen=True, slots=True)
class BBox:
    """Bounding box in pixel coordinates (xyxy format)."""

    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def from_xyxy(cls, coords: list[float]) -> BBox:
        """Create from YOLO xyxy output: [x1, y1, x2, y2]."""
        return cls(coords[0], coords[1], coords[2], coords[3])

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_pdf(self, scale_x: float, scale_y: float) -> BBox:
        """Convert pixel coordinates to PDF points."""
        return BBox(
            x1=self.x1 * scale_x,
            y1=self.y1 * scale_y,
            x2=self.x2 * scale_x,
            y2=self.y2 * scale_y,
        )

    def iou(self, other: BBox) -> float:
        """Intersection over union with another box."""
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def contains(self, other: BBox, threshold: float = 0.8) -> bool:
        """True if `other` is mostly inside this box."""
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        return (inter / other.area) >= threshold if other.area > 0 else False


@dataclass(frozen=True, slots=True)
class Detection:
    """A single YOLO detection on a page."""

    bbox: BBox
    label: Label
    confidence: float
    page_index: int

    @property
    def is_copyrighted(self) -> bool:
        return self.label.is_copyrighted

    def column(self, midpoint: float) -> str:
        """Which column this detection falls in relative to a midpoint (pixels)."""
        return "LEFT" if self.bbox.center_x < midpoint else "RIGHT"

    def sort_key(self, midpoint: float) -> tuple[int, int, float]:
        """Key for reading order: page, then column (L=0, R=1), then y position."""
        col_ord = 0 if self.bbox.center_x < midpoint else 1
        return (self.page_index, col_ord, self.bbox.y1)


@dataclass
class Page:
    """A page in the document with its dimensions and detections."""

    index: int
    pdf_width: float
    pdf_height: float
    img_width: int
    img_height: int
    detections: list[Detection] = field(default_factory=list)
    page_number: int | None = None
    col_left_x1: float = 0.0
    col_left_x2: float = 0.0
    col_right_x1: float = 0.0
    col_right_x2: float = 0.0
    midpoint: float = 0.0

    def __post_init__(self):
        if self.midpoint == 0.0:
            self._set_fallback_columns()

    def _set_fallback_columns(self) -> None:
        """50/50 column split with padding."""
        w = self.img_width
        side_pad = max(20, int(w * 0.03))
        gutter = max(10, int(w * 0.02))
        self.midpoint = w / 2
        self.col_left_x1 = float(side_pad)
        self.col_left_x2 = float(max(side_pad + 10, w // 2 - gutter))
        self.col_right_x1 = float(min(w - side_pad - 10, w // 2 + gutter))
        self.col_right_x2 = float(w - side_pad)

    @property
    def scale_x(self) -> float:
        """Pixels to PDF points, horizontal."""
        return self.pdf_width / self.img_width

    @property
    def scale_y(self) -> float:
        """Pixels to PDF points, vertical."""
        return self.pdf_height / self.img_height

    def by_label(self, *labels: Label) -> list[Detection]:
        """Filter detections by one or more labels."""
        label_set = set(labels)
        return [d for d in self.detections if d.label in label_set]

    @property
    def copyrighted(self) -> list[Detection]:
        return [d for d in self.detections if d.is_copyrighted]

    def in_reading_order(self) -> list[Detection]:
        """Detections sorted by column then vertical position."""
        return sorted(self.detections, key=lambda d: d.sort_key(self.midpoint))


@dataclass
class Document:
    """A PDF document with pages and detections."""

    pdf_path: Path
    pages: list[Page] = field(default_factory=list)
    volume: str | None = None
    reporter: str | None = None
    first_page: int = 1
    ocr_applied: bool = False

    def by_label(self, *labels: Label) -> list[Detection]:
        """All detections matching the given labels, in reading order."""
        hits = [d for p in self.pages for d in p.by_label(*labels)]
        if self.pages:
            mid = self.pages[0].midpoint
            hits.sort(key=lambda d: d.sort_key(mid))
        return hits

    @property
    def captions(self) -> list[Detection]:
        """All case captions in reading order."""
        return self.by_label(Label.CASE_CAPTION)

    @property
    def copyrighted(self) -> list[Detection]:
        """All copyrighted detections across all pages."""
        return [d for p in self.pages for d in p.copyrighted]
