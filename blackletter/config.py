"""Configuration management for blackletter redaction pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass
class RedactionConfig:
    """Configuration for PDF redaction pipeline."""

    MODEL_PATH: str = None  # Will be resolved in __post_init__

    def __post_init__(self):
        """Resolve model path after initialization."""
        root = Path(__file__).parent.parent
        self.MODEL_PATH: str = str(root / "models" / "best.pt")

    # Image processing
    dpi: int = 200
    confidence_threshold: float = 0.20
    low_confidence_threshold: float = 0.005

    # Redaction positioning
    start_offset: int = 0
    end_offset: int = 0

    # Column and margin detection
    top_margin: int = 72  # 1 inch in points
    bottom_margin: int = 0
    min_redaction_height: int = 18  # PDF points
    left_margin: int = 40
    right_margin: int = 120

    # Redaction styling
    redaction_fill: Tuple[float, float, float] = (0.2, 0.2, 0.2)  # dark gray
    mask_color: Tuple[float, float, float] = (1, 1, 1)  # white for opinion masking

    # Text redaction parameters
    text_pad: float = 1.5
    y_tolerance: float = 3.0
    merge_gap: float = 2.5
    min_redaction_height_px: float = 6.0

    # Word extraction
    word_x_tolerance: int = 1
    word_y_tolerance: int = 2

    # Header detection
    header_top_pts: float = 40.0
    header_gap_pts: float = 2.0
    header_margin_pts: float = 120.0
    header_pad_x: float = 2.0
    header_pad_y: float = 1.0
    header_y_tol: float = 3.0
