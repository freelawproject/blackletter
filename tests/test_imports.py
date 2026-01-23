"""Test that all modules import correctly."""

import unittest


def test_config_imports():
    """Test config module imports."""
    from blackletter.config import RedactionConfig

    config = RedactionConfig()
    assert config.dpi == 200
    assert config.confidence_threshold == 0.20
    assert config.MODEL_PATH == "best.pt"


def test_core_imports():
    """Test that all core modules can be imported."""
    from blackletter.core.scanner import PDFScanner
    from blackletter.core.planner import OpinionPlanner
    from blackletter.core.redactor import PDFRedactor
    from blackletter.core.extractor import OpinionExtractor

    assert PDFScanner is not None
    assert OpinionPlanner is not None
    assert PDFRedactor is not None
    assert OpinionExtractor is not None


def test_utils_imports():
    """Test that all utils can be imported."""
    from blackletter.utils import (
        HeaderProcessor,
        ImageProcessor,
        process_brackets,
        column_for_coords,
        detect_columns_from_image,
        fallback_column_detection,
        redact_text_lines_in_window,
    )

    assert HeaderProcessor is not None
    assert ImageProcessor is not None
    assert callable(process_brackets)
    assert callable(column_for_coords)
    assert callable(detect_columns_from_image)
    assert callable(fallback_column_detection)
    assert callable(redact_text_lines_in_window)


def test_pipeline_imports():
    """Test main pipeline import."""
    from blackletter import BlackletterPipeline, redact_pdf

    assert BlackletterPipeline is not None
    assert callable(redact_pdf)


def test_cli_imports():
    """Test CLI module imports."""
    from blackletter.cli import main

    assert callable(main)


def test_config_customization():
    """Test that config can be customized."""
    from blackletter.config import RedactionConfig

    config = RedactionConfig(dpi=150, confidence_threshold=0.25, MODEL_PATH="custom.pt")

    assert config.dpi == 150
    assert config.confidence_threshold == 0.25
    assert config.MODEL_PATH == "custom.pt"


def test_column_for_coords():
    """Test column detection helper."""
    from blackletter.utils.processing import column_for_coords

    # Left column
    col = column_for_coords([10, 10, 50, 50], center_X=100)
    assert col == "LEFT"

    # Right column
    col = column_for_coords([150, 10, 190, 50], center_X=100)
    assert col == "RIGHT"


if __name__ == "__main__":
    unittest.main()
