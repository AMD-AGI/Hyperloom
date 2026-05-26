"""Targeted unit tests for ``inference_optimizer.breakdown.exporter`` helpers.

The end-to-end ``build`` flow has integration coverage via
``test_breakdown_smoke`` and friends; this module pins the small
internal helpers (``_load_state`` / ``_load_manifest`` / ``_coverage_label``)
plus the malformed-JSON branches that integration tests skip.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_optimizer.breakdown import exporter as ex


# ---------------------------------------------------------------------------
# _coverage_label
# ---------------------------------------------------------------------------

class TestCoverageLabel:
    @pytest.mark.parametrize(
        "state, manifest, expected",
        [
            (True, True, "full"),
            (True, False, "partial"),
            (False, True, "partial"),
            (False, False, "shell_only"),
        ],
    )
    def test_combinations(self, state, manifest, expected):
        assert ex._coverage_label(state, manifest) == expected


# ---------------------------------------------------------------------------
# _load_state
# ---------------------------------------------------------------------------

class TestLoadState:
    def test_missing_returns_empty_and_warns(self, tmp_path):
        warnings: list[str] = []
        state, present = ex._load_state(tmp_path, warnings)
        assert state == {}
        assert present is False
        assert any("state.json missing" in w for w in warnings)

    def test_parses_valid_state(self, tmp_path):
        (tmp_path / "state.json").write_text(json.dumps({"k": 1}))
        warnings: list[str] = []
        state, present = ex._load_state(tmp_path, warnings)
        assert state == {"k": 1}
        assert present is True
        assert warnings == []

    def test_malformed_state_returns_empty_with_warn(self, tmp_path):
        (tmp_path / "state.json").write_text("{not valid json")
        warnings: list[str] = []
        state, present = ex._load_state(tmp_path, warnings)
        assert state == {}
        assert present is False
        assert any("failed to parse state.json" in w for w in warnings)


# ---------------------------------------------------------------------------
# _load_manifest
# ---------------------------------------------------------------------------

class TestLoadManifest:
    def test_missing_returns_empty_and_warns(self, tmp_path):
        warnings: list[str] = []
        manifest, present = ex._load_manifest(tmp_path, warnings)
        assert manifest == {}
        assert present is False
        assert any("manifest.json missing" in w for w in warnings)

    def test_parses_valid_manifest(self, tmp_path):
        (tmp_path / "manifest.json").write_text(json.dumps({"a": 1}))
        warnings: list[str] = []
        manifest, present = ex._load_manifest(tmp_path, warnings)
        assert manifest == {"a": 1}
        assert present is True

    def test_malformed_manifest_returns_empty_with_warn(self, tmp_path):
        (tmp_path / "manifest.json").write_text("{broken")
        warnings: list[str] = []
        manifest, present = ex._load_manifest(tmp_path, warnings)
        assert manifest == {}
        assert present is False
        assert any("failed to parse manifest.json" in w for w in warnings)


# ---------------------------------------------------------------------------
# build() integration with malformed inputs
# ---------------------------------------------------------------------------

class TestBuildShellOnly:
    def test_shell_only_emits_consolidated_warning(self, tmp_path):
        out = ex.build(tmp_path)
        warnings = out["warnings"]
        # The dedicated coverage warning replaces both "missing" warnings.
        assert any("shell_only" in w for w in warnings)
        # Both missing-file warnings should NOT also leak through.
        assert not any("manifest.json missing" in w for w in warnings)
