# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Targeted unit tests for ``inference_optimizer.breakdown.exporter`` helpers."""

from __future__ import annotations

import json

from inference_optimizer.breakdown import collectors as col
from inference_optimizer.breakdown import exporter as ex


# _load_state

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


# _load_manifest

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


# collect_specialist_runs — round_id coercion (fail-soft)

class TestCoerceRoundId:
    def test_numeric_string_becomes_int(self):
        assert col._coerce_round_id("3") == 3

    def test_int_passthrough(self):
        assert col._coerce_round_id(7) == 7

    def test_explore_label_kept_as_string(self):
        assert col._coerce_round_id("explore-001") == "explore-001"

    def test_task_id_hash_kept_as_string(self):
        # The bug repro: a task-id hash must NOT raise on int() cast.
        assert col._coerce_round_id(
            "607ba5c978a147d2a2b2ef8132fe2730"
        ) == "607ba5c978a147d2a2b2ef8132fe2730"

    def test_none_and_empty_collapse_to_zero(self):
        assert col._coerce_round_id(None) == 0
        assert col._coerce_round_id("") == 0


class TestCollectSpecialistRuns:
    def test_hash_round_id_does_not_crash_and_is_preserved(self, tmp_path):
        """Regression: a task-id-hash ``round_id`` used to raise ``ValueError`` and drop the whole specialist_runs section."""
        warnings: list[str] = []
        state = {
            "specialist_rounds": [
                {
                    "round_id": "607ba5c978a147d2a2b2ef8132fe2730",
                    "domains": ["serving_specialist"],
                    "proposals_total": 4,
                    "proposals_kept": 1,
                },
                {
                    "round_id": "2",
                    "domains": ["kernel_specialist"],
                    "proposals_total": 2,
                },
            ],
        }
        out = col.collect_specialist_runs(tmp_path, state, warnings)
        assert len(out) == 2
        assert out[0]["round_id"] == "607ba5c978a147d2a2b2ef8132fe2730"
        assert out[1]["round_id"] == 2
        assert out[0]["proposals_kept"] == 1
        # No ValueError surfaced into warnings for the hash round_id.
        assert not any("invalid literal" in w for w in warnings)

    def test_singular_domain_task_id_and_confidence_fallbacks(self, tmp_path):
        """Rounds with singular ``domain`` / ``task_id`` + bare ``confidence`` must be surfaced rather than emitting empty fields."""
        runs_dir = tmp_path / "runs" / "specialist" / "abc123"
        runs_dir.mkdir(parents=True)
        (runs_dir / "specialist_done.json").write_text(
            json.dumps({"domain": "comm_specialist", "proposal_set": []})
        )
        state = {
            "specialist_rounds": [
                {
                    "round_id": "abc123",
                    "task_id": "abc123",
                    "domain": "comm_specialist",
                    "tags": ["communication"],
                    "confidence": 0.55,
                    "proposals_total": 3,
                },
            ],
        }
        warnings: list[str] = []
        out = col.collect_specialist_runs(tmp_path, state, warnings)
        assert len(out) == 1
        row = out[0]
        assert row["domains"] == ["comm_specialist"]
        assert row["confidence_avg"] == 0.55
        assert len(row["transcripts"]) == 1
        assert row["transcripts"][0]["task_id"] == "abc123"
        assert row["transcripts"][0]["domain"] == "communication"
