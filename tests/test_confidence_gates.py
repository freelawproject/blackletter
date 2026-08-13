"""Tests that a model family only ever gets its own confidence gates.

bl-warm and the legacy small/medium/large trio score the same labels
differently, so the gates measured against one family's distribution are
wrong for the other. EDITORIAL is the case that matters: bl-warm needs a
0.50 floor to drop its one harmful false positive, while the legacy models
have no entry at all and redact everything down to CONFIDENCE_THRESHOLD.
Applying bl-warm's floor to a legacy run would leave copyrighted editorial
content unredacted in the 0.20-0.50 band, so these tests pin the gates to
the family that produced the detection, and pin the provenance that carries
that family across a sidecar round-trip.
"""

from __future__ import annotations

import json

import pytest

from blackletter.bl_warm import rows_are_bl_warm
from blackletter.models import Document, Label
from blackletter.scanner import (
    CONFIDENCE_THRESHOLD,
    _filter_dets,
    _write_detections_sidecar,
    label_confidence,
)
from tests.pdf_fixtures import detected_page, detection


class TestLegacyGatesAreUnchanged:
    """What `blackletter process` used before bl-warm existed."""

    def test_editorial_falls_through_to_the_default(self):
        assert label_confidence(Label.EDITORIAL) == CONFIDENCE_THRESHOLD

    def test_page_header_and_headnote_bracket_stay_at_a_half(self):
        assert label_confidence(Label.PAGE_HEADER) == 0.50
        assert label_confidence(Label.HEADNOTE_BRACKET) == 0.50

    def test_a_low_confidence_editorial_is_still_redacted(self):
        """The under-redaction this scoping exists to prevent."""
        dets = [detection(Label.EDITORIAL, 100, 100, 300, 200, confidence=0.30)]
        assert _filter_dets(dets) == dets


class TestBlWarmGatesApplyToBlWarmOnly:
    def test_editorial_needs_a_half(self):
        assert label_confidence(Label.EDITORIAL, bl_warm=True) == 0.50

    def test_page_header_and_headnote_bracket_drop_to_three_tenths(self):
        assert label_confidence(Label.PAGE_HEADER, bl_warm=True) == 0.30
        assert label_confidence(Label.HEADNOTE_BRACKET, bl_warm=True) == 0.30

    def test_labels_without_an_override_are_shared(self):
        for label in (Label.KEY_ICON, Label.CASE_CAPTION, Label.PAGE_NUMBER):
            assert label_confidence(label, bl_warm=True) == label_confidence(label)

    def test_a_low_confidence_editorial_is_dropped(self):
        dets = [detection(Label.EDITORIAL, 100, 100, 300, 200, confidence=0.30)]
        assert _filter_dets(dets, bl_warm=True) == []

    def test_a_low_confidence_page_header_is_kept(self):
        dets = [detection(Label.PAGE_HEADER, 100, 40, 300, 60, confidence=0.35)]
        assert _filter_dets(dets, bl_warm=True) == dets
        assert _filter_dets(dets) == []


class TestProvenanceSurvivesTheSidecar:
    """A Document rebuilt from detections.json has to know its family."""

    def _document(self, tmp_path, bl_warm: bool) -> Document:
        page = detected_page([detection(Label.EDITORIAL, 100, 100, 300, 200)])
        return Document(pdf_path=tmp_path / "vol.pdf", pages=[page], bl_warm=bl_warm)

    def _rows(self, tmp_path, bl_warm: bool) -> list[dict]:
        _write_detections_sidecar(self._document(tmp_path, bl_warm), tmp_path)
        return json.loads((tmp_path / "detections.json").read_text())

    def test_bl_warm_rows_are_stamped(self, tmp_path):
        assert rows_are_bl_warm(self._rows(tmp_path, bl_warm=True))

    def test_legacy_rows_carry_no_model_key(self, tmp_path):
        rows = self._rows(tmp_path, bl_warm=False)
        assert all("model" not in row for row in rows)
        assert not rows_are_bl_warm(rows)


class TestRowsAreBlWarm:
    def test_api_detect_rows_are_read_from_found_by(self):
        rows = [{"found_by": [{"model": "bl_warm", "confidence": 0.9}]}]
        assert rows_are_bl_warm(rows)

    def test_a_mixed_run_keeps_the_legacy_gates(self):
        rows = [
            {"found_by": [{"model": "bl_warm", "confidence": 0.9}]},
            {"found_by": [{"model": "medium", "confidence": 0.9}]},
        ]
        assert not rows_are_bl_warm(rows)

    def test_rows_without_provenance_keep_the_legacy_gates(self):
        assert not rows_are_bl_warm([{"label": "EDITORIAL"}])
        assert not rows_are_bl_warm([])


class TestBitonalIsRefusedForBlWarm:
    """bl-warm has to see the original render, not the 1-bit copy.

    ``--bitonal`` reassigns the source PDF before the scan, so the pair
    would otherwise hand bl-warm exactly the input its region classes
    collapse on, with nothing in the output saying so.
    """

    def _args(self, tmp_path, **overrides):
        import argparse

        weights = tmp_path / "bl_warm.pt"
        weights.touch()
        return argparse.Namespace(
            pdf=tmp_path / "vol.pdf",
            model=weights,
            reporter="f3d",
            volume="100",
            first_page=1,
            bitonal=True,
            **overrides,
        )

    def _stub_yolo(self, monkeypatch, names):
        import ultralytics

        class _Model:
            def __init__(self, _path):
                self.names = names

        monkeypatch.setattr(ultralytics, "YOLO", _Model)

    def test_bl_warm_with_bitonal_exits(self, tmp_path, monkeypatch, capsys):
        from blackletter.process import cmd_process

        self._stub_yolo(monkeypatch, {0: "keycite", 1: "body"})
        with pytest.raises(SystemExit) as exc:
            cmd_process(self._args(tmp_path))
        assert exc.value.code == 1
        assert "--bitonal cannot be combined with bl-warm" in capsys.readouterr().err

    def test_a_legacy_model_with_bitonal_is_left_alone(self, tmp_path, monkeypatch):
        """It gets past the guard; the run then fails on the missing PDF."""
        from blackletter.process import cmd_process

        self._stub_yolo(monkeypatch, {0: "key_icon", 1: "page_header"})
        with pytest.raises(Exception) as exc:
            cmd_process(self._args(tmp_path))
        assert not isinstance(exc.value, SystemExit)
