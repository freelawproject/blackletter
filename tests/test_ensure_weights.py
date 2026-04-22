"""Tests for ``blackletter.api.ensure_weights``.

``small`` and ``medium`` are bundled in the package, so we exercise
them directly. ``large`` is normally pulled from Hugging Face, so the
download is mocked to avoid network access.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from blackletter import api


PACKAGE_WEIGHTS_DIR = Path(api.__file__).resolve().parent / "weights"


# ── Bundled weights (small, medium) ──────────────────────────────────────


class TestBundledWeights:
    def test_small_and_medium_resolve_to_bundled_files(self):
        """``small`` and ``medium`` ship in-package and just resolve."""
        resolved = api.ensure_weights(["small", "medium"])

        assert set(resolved) == {"small", "medium"}
        assert resolved["small"] == PACKAGE_WEIGHTS_DIR / "small.pt"
        assert resolved["medium"] == PACKAGE_WEIGHTS_DIR / "medium.pt"
        assert resolved["small"].is_file()
        assert resolved["medium"].is_file()

    def test_does_not_touch_huggingface_hub_for_bundled(self):
        """No download attempt when bundled weights exist."""
        fake_hf = MagicMock()
        fake_hf.hf_hub_download.side_effect = AssertionError(
            "should not be called for bundled weights"
        )
        with patch.dict(sys.modules, {"huggingface_hub": fake_hf}):
            api.ensure_weights(["small", "medium"])

        fake_hf.hf_hub_download.assert_not_called()


# ── Large weight (mocked) ────────────────────────────────────────────────


class TestLargeWeightMocked:
    def test_large_noop_when_already_present(self, tmp_path, monkeypatch):
        """If ``large.pt`` exists locally, no download is attempted."""
        fake_pkg = tmp_path / "blackletter"
        fake_pkg.mkdir()
        (fake_pkg / "api.py").touch()
        (fake_pkg / "weights").mkdir()
        (fake_pkg / "weights" / "large.pt").write_bytes(b"stub")
        monkeypatch.setattr(api, "__file__", str(fake_pkg / "api.py"))

        fake_hf = MagicMock()
        fake_hf.hf_hub_download.side_effect = AssertionError("should not download when file exists")
        with patch.dict(sys.modules, {"huggingface_hub": fake_hf}):
            resolved = api.ensure_weights(["large"])

        assert resolved["large"] == fake_pkg / "weights" / "large.pt"
        fake_hf.hf_hub_download.assert_not_called()

    def test_large_downloads_when_missing(self, tmp_path, monkeypatch):
        """Missing ``large.pt`` triggers a HF download call."""
        fake_pkg = tmp_path / "blackletter"
        fake_pkg.mkdir()
        (fake_pkg / "api.py").touch()
        weights_dir = fake_pkg / "weights"
        # Intentionally do NOT create the weights dir or large.pt.
        monkeypatch.setattr(api, "__file__", str(fake_pkg / "api.py"))

        downloaded_path = weights_dir / "large.pt"

        def fake_download(*, repo_id, filename, local_dir):
            assert repo_id == "flooie/blackletter-large"
            Path(local_dir).mkdir(parents=True, exist_ok=True)
            target = Path(local_dir) / filename
            target.write_bytes(b"downloaded-stub")
            return str(target)

        fake_hf = MagicMock()
        fake_hf.hf_hub_download.side_effect = fake_download
        with patch.dict(sys.modules, {"huggingface_hub": fake_hf}):
            resolved = api.ensure_weights(["large"])

        fake_hf.hf_hub_download.assert_called_once_with(
            repo_id="flooie/blackletter-large",
            filename="large.pt",
            local_dir=str(weights_dir),
        )
        assert resolved["large"] == downloaded_path
        assert downloaded_path.is_file()


# ── Error cases ──────────────────────────────────────────────────────────


class TestErrors:
    def test_unknown_weight_without_hf_source_raises(self, tmp_path, monkeypatch):
        """A missing weight with no HF mapping raises ``FileNotFoundError``."""
        fake_pkg = tmp_path / "blackletter"
        fake_pkg.mkdir()
        (fake_pkg / "api.py").touch()
        monkeypatch.setattr(api, "__file__", str(fake_pkg / "api.py"))

        with pytest.raises(FileNotFoundError, match="no Hugging Face source"):
            api.ensure_weights(["nonexistent"])


# ── detect() integration ─────────────────────────────────────────────────


class TestDetectIntegration:
    def test_detect_raises_for_unknown_model_instead_of_skipping(self, tmp_path):
        """``detect`` should raise for unknown model names, not silently skip.

        ``ensure_weights`` runs up front in ``detect``, so the PDF is
        never opened and we can pass a non-existent path.
        """
        with pytest.raises(FileNotFoundError, match="no Hugging Face source"):
            api.detect(
                pdf_path=tmp_path / "fake.pdf",
                output_dir=tmp_path,
                models=["nonexistent"],
            )
