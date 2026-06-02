"""Targeted unit tests for ``inference_optimizer.breakdown.exporter`` helpers.

The end-to-end ``build`` flow has integration coverage via
``test_breakdown_smoke`` and friends; this module pins the small
internal helpers (``_load_state`` / ``_load_manifest``) plus the
malformed-JSON branches that integration tests skip.
"""

from __future__ import annotations

import json

from inference_optimizer.breakdown import exporter as ex


# ---------------------------------------------------------------------------
# _load_state
# ---------------------------------------------------------------------------

class TestLoadState:
    def test_missing_returns_empty_and_warns(self, tmp_path):
        warnings: list[str] = []
        state = ex._load_state(tmp_path, warnings)
        assert state == {}
        assert any("state.json missing" in w for w in warnings)

    def test_parses_valid_state(self, tmp_path):
        (tmp_path / "state.json").write_text(json.dumps({"k": 1}))
        warnings: list[str] = []
        state = ex._load_state(tmp_path, warnings)
        assert state == {"k": 1}
        assert warnings == []

    def test_malformed_state_returns_empty_with_warn(self, tmp_path):
        (tmp_path / "state.json").write_text("{not valid json")
        warnings: list[str] = []
        state = ex._load_state(tmp_path, warnings)
        assert state == {}
        assert any("failed to parse state.json" in w for w in warnings)


# ---------------------------------------------------------------------------
# _load_manifest
# ---------------------------------------------------------------------------

class TestLoadManifest:
    def test_missing_returns_empty_and_warns(self, tmp_path):
        warnings: list[str] = []
        manifest = ex._load_manifest(tmp_path, warnings)
        assert manifest == {}
        assert any("manifest.json missing" in w for w in warnings)

    def test_parses_valid_manifest(self, tmp_path):
        (tmp_path / "manifest.json").write_text(json.dumps({"a": 1}))
        warnings: list[str] = []
        manifest = ex._load_manifest(tmp_path, warnings)
        assert manifest == {"a": 1}

    def test_malformed_manifest_returns_empty_with_warn(self, tmp_path):
        (tmp_path / "manifest.json").write_text("{broken")
        warnings: list[str] = []
        manifest = ex._load_manifest(tmp_path, warnings)
        assert manifest == {}
        assert any("failed to parse manifest.json" in w for w in warnings)
