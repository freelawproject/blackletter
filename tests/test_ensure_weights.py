"""Tests for ``blackletter.api.ensure_weights``.

All weights (``small``, ``medium``, ``large``) are pulled from Hugging
Face on demand, so downloads are mocked to avoid network access.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from blackletter import api


HF_REPO = "freelawproject/blackletter-weights"


def _fake_package(tmp_path, monkeypatch) -> Path:
    """Point ``api`` at a throwaway package dir and return it."""
    fake_pkg = tmp_path / "blackletter"
    fake_pkg.mkdir()
    (fake_pkg / "api.py").touch()
    monkeypatch.setattr(api, "__file__", str(fake_pkg / "api.py"))
    return fake_pkg


# ── Weights already on disk ──────────────────────────────────────────────


class TestPresentWeights:
    def test_present_weights_resolve_without_download(self, tmp_path, monkeypatch):
        """Weights already on disk just resolve; no HF call."""
        fake_pkg = _fake_package(tmp_path, monkeypatch)
        weights_dir = fake_pkg / "weights"
        weights_dir.mkdir()
        for name in ("small", "medium", "large"):
            (weights_dir / f"{name}.pt").write_bytes(b"stub")

        fake_hf = MagicMock()
        fake_hf.hf_hub_download.side_effect = AssertionError(
            "should not be called when weights exist"
        )
        with patch.dict(sys.modules, {"huggingface_hub": fake_hf}):
            resolved = api.ensure_weights()

        assert set(resolved) == {"small", "medium", "large"}
        for name in ("small", "medium", "large"):
            assert resolved[name] == weights_dir / f"{name}.pt"
        fake_hf.hf_hub_download.assert_not_called()


# ── Missing weights (mocked download) ────────────────────────────────────


class TestMissingWeightsDownload:
    @pytest.mark.parametrize("name", ["small", "medium", "large"])
    def test_missing_weight_downloads_from_hf(self, tmp_path, monkeypatch, name):
        """A missing weight triggers a HF download from the weights repo."""
        fake_pkg = _fake_package(tmp_path, monkeypatch)
        weights_dir = fake_pkg / "weights"

        def fake_download(*, repo_id, filename, revision, local_dir):
            assert repo_id == HF_REPO
            Path(local_dir).mkdir(parents=True, exist_ok=True)
            target = Path(local_dir) / filename
            target.write_bytes(b"downloaded-stub")
            return str(target)

        fake_hf = MagicMock()
        fake_hf.hf_hub_download.side_effect = fake_download
        with patch.dict(sys.modules, {"huggingface_hub": fake_hf}):
            resolved = api.ensure_weights([name])

        fake_hf.hf_hub_download.assert_called_once_with(
            repo_id=HF_REPO,
            filename=f"{name}.pt",
            revision=api._HF_WEIGHTS[name][2],
            local_dir=str(weights_dir),
        )
        assert resolved[name] == weights_dir / f"{name}.pt"
        assert resolved[name].is_file()


# ── Error cases ──────────────────────────────────────────────────────────


class TestErrors:
    def test_unknown_weight_without_hf_source_raises(self, tmp_path, monkeypatch):
        """A missing weight with no HF mapping raises ``FileNotFoundError``."""
        _fake_package(tmp_path, monkeypatch)

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
