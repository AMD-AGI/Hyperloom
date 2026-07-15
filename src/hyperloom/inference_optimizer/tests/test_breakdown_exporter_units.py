# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Targeted unit tests for ``hyperloom.inference_optimizer.breakdown.exporter`` helpers."""

from __future__ import annotations

import json

from hyperloom.inference_optimizer.breakdown import collectors as col
from hyperloom.inference_optimizer.breakdown import exporter as ex


# _load_session_json for state.json


class TestLoadState:
    def test_missing_returns_empty_and_warns(self, tmp_path):
        warnings: list[str] = []
        state = ex._load_session_json(tmp_path / "state.json", "state.json", warnings)
        assert state == {}
        assert any("state.json missing" in w for w in warnings)

    def test_parses_valid_state(self, tmp_path):
        (tmp_path / "state.json").write_text(json.dumps({"k": 1}))
        warnings: list[str] = []
        state = ex._load_session_json(tmp_path / "state.json", "state.json", warnings)
        assert state == {"k": 1}
        assert warnings == []

    def test_malformed_state_returns_empty_with_warn(self, tmp_path):
        (tmp_path / "state.json").write_text("{not valid json")
        warnings: list[str] = []
        state = ex._load_session_json(tmp_path / "state.json", "state.json", warnings)
        assert state == {}
        assert any("failed to parse state.json" in w for w in warnings)


# _load_session_json for manifest.json


class TestLoadManifest:
    def test_missing_returns_empty_and_warns(self, tmp_path):
        warnings: list[str] = []
        manifest = ex._load_session_json(tmp_path / "manifest.json", "manifest.json", warnings)
        assert manifest == {}
        assert any("manifest.json missing" in w for w in warnings)

    def test_parses_valid_manifest(self, tmp_path):
        (tmp_path / "manifest.json").write_text(json.dumps({"a": 1}))
        warnings: list[str] = []
        manifest = ex._load_session_json(tmp_path / "manifest.json", "manifest.json", warnings)
        assert manifest == {"a": 1}

    def test_malformed_manifest_returns_empty_with_warn(self, tmp_path):
        (tmp_path / "manifest.json").write_text("{broken")
        warnings: list[str] = []
        manifest = ex._load_session_json(tmp_path / "manifest.json", "manifest.json", warnings)
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
        # A task-id hash must NOT raise on int() cast.
        assert col._coerce_round_id("607ba5c978a147d2a2b2ef8132fe2730") == "607ba5c978a147d2a2b2ef8132fe2730"

    def test_none_and_empty_collapse_to_zero(self):
        assert col._coerce_round_id(None) == 0
        assert col._coerce_round_id("") == 0


class TestCollectSpecialistRuns:
    def test_hash_round_id_does_not_crash_and_is_preserved(self, tmp_path):
        """A task-id-hash ``round_id`` must not raise ``ValueError`` and drop the specialist_runs section."""
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
        assert not any("invalid literal" in w for w in warnings)

    def test_singular_domain_task_id_and_confidence_fallbacks(self, tmp_path):
        """Rounds with singular ``domain`` / ``task_id`` + bare ``confidence`` must be surfaced."""
        runs_dir = tmp_path / "runs" / "specialist" / "abc123"
        runs_dir.mkdir(parents=True)
        (runs_dir / "specialist_done.json").write_text(json.dumps({"domain": "comm_specialist", "proposal_set": []}))
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


class TestCollectGemmTuning:
    def test_empty_when_no_attempts(self):
        assert col.collect_gemm_tuning({}) == {}

    def test_adopted_run_takes_kept_gain_and_engine(self):
        state = {
            "baseline_tput": 1000.0,
            "tp": 1,
            "conc": 64,
            "isl": 128,
            "osl": 128,
            "precision": "fp8",
            "framework": "sglang",
            "gpu_type": "mi355x",
            "cumulative_gain_validated_stack_len": 1,
            "gemm_tuning_attempts": [
                {
                    "engine": "geak",
                    "status": "ok",
                    "decision": "KEEP",
                    "source": "kernel_entry_auto",
                    "best_speedup": 1.28,
                    "tuned_file": "/w/a8w8.csv",
                    "final_report_path": "/w/final_report.json",
                    "workspace": "/w",
                    "ts": "2026-06-19T00:00:00Z",
                    "summary": {"best_conc": 64},
                },
                {
                    "engine": "geak",
                    "status": "complete",
                    "decision": "REVERT",
                    "best_speedup": 0.98,
                    "tuned_file": "/w2/a8w8.csv",
                    "workspace": "/w2",
                    "ts": "2026-06-19T00:05:00Z",
                },
            ],
            "optimization_stack": [
                {
                    "action": "gemm_tuning",
                    "engine": "geak",
                    "tuned_file": "/w/a8w8.csv",
                    "gain_pct": 28.0,
                    "tput": 1280.0,
                },
            ],
        }
        out = col.collect_gemm_tuning(state)
        assert len(out["runs"]) == 2
        kept, reverted = out["runs"]
        assert kept["engine"] == "geak"
        assert kept["adopted"] is True
        assert kept["gain_pct"] == 28.0
        assert kept["tuned_tput"] == 1280.0
        assert kept["conc"] == 64
        assert kept["summary"] == {"best_conc": 64}
        assert reverted["adopted"] is False
        assert reverted["decision"] == "REVERT"
        assert out["adopted_engine"] == "geak"
        assert out["adopted_tuned_file"] == "/w/a8w8.csv"
        assert out["total_gain_pct"] == 28.0

    def test_engine_defaults_to_geak_and_falls_back_to_last(self):
        state = {
            "last_gemm_tuning": {
                "status": "ok",
                "decision": "KEEP",
                "best_speedup": 1.1,
                "tuned_file": "/w/a8w8.csv",
            },
        }
        out = col.collect_gemm_tuning(state)
        assert len(out["runs"]) == 1
        assert out["runs"][0]["engine"] == "geak"

    def test_skip_reason_surfaced_in_run(self):
        state = {
            "baseline_tput": 1000.0,
            "precision": "fp8",
            "framework": "sglang",
            "gemm_tuning_attempts": [
                {
                    "backend": "forge",
                    "status": "skipped",
                    "decision": "REVERT",
                    "skip_reason": "a8w8_blockscale: needs --untuned-csv",
                    "tuners_skipped": [
                        {"tuner": "a8w8_blockscale", "skip_reason": "needs --untuned-csv"}
                    ],
                    "ts": "2026-06-21T00:00:00Z",
                },
            ],
        }
        out = col.collect_gemm_tuning(state)
        run = out["runs"][0]
        assert run["skip_reason"] == "a8w8_blockscale: needs --untuned-csv"
        assert run["tuners_skipped"][0]["tuner"] == "a8w8_blockscale"

    def test_forge_backend_not_mislabeled_as_geak(self):
        # Forge records the tuner under ``backend`` and leaves ``engine`` unset;
        # the collector must surface ``forge`` rather than defaulting to geak.
        state = {
            "baseline_tput": 1000.0,
            "precision": "bf16",
            "framework": "sglang",
            "cumulative_gain_validated_stack_len": 1,
            "gemm_tuning_attempts": [
                {
                    "backend": "forge",
                    "status": "complete",
                    "decision": "KEEP",
                    "best_speedup": 1.2,
                    "tuned_file": "AITER_CONFIG_GEMM_BF16",
                    "ts": "2026-06-21T00:00:00Z",
                },
            ],
            "optimization_stack": [
                {
                    "action": "gemm_tuning",
                    "backend": "forge",
                    "tuned_file": "AITER_CONFIG_GEMM_BF16",
                    "gain_pct": 5.5,
                    "tput": 1055.0,
                },
            ],
        }
        out = col.collect_gemm_tuning(state)
        assert out["runs"][0]["engine"] == "forge"
        assert out["adopted_engine"] == "forge"

    def test_forge_engine_attributed_not_defaulted_to_geak(self):
        state = {
            "baseline_tput": 1000.0,
            "gemm_tuning_attempts": [
                {
                    "engine": "forge",
                    "status": "ok",
                    "decision": "KEEP",
                    "best_speedup": 1.1,
                },
            ],
        }
        out = col.collect_gemm_tuning(state)
        assert len(out["runs"]) == 1
        assert out["runs"][0]["engine"] == "forge"


# _collect_recovery + collect_session.recovery


class TestCollectRecovery:
    def test_clean_run_reports_not_recovered(self):
        rec = col._collect_recovery({})
        assert rec["recovered"] is False
        assert rec["crash_count"] == 0
        assert rec["crash_timestamps"] == []
        assert rec["last_tick_exception"] is None
        assert rec["steward_infra_failures_total"] == 0

    def test_crash_and_resume_signals_surface(self):
        # crashed + steward-continued + stack awaiting post-resume revalidation;
        # all must show, and epoch crash timestamps render as ISO UTC.
        state = {
            "crash_count": 2,
            "crash_timestamps": [1782800000.0, 1782803600.5],
            "degraded_mode": True,
            "steward_continuation_used": True,
            "resume_pending_revalidation": True,
            "steward_infra_failures_by_round": {"12": 1, "13": 2},
            "last_tick_exception": {
                "tick": 12,
                "ts": "2026-06-29T22:25:00Z",
                "stage": "kernel_opt",
                "agent": "kernel_agent",
                "type": "TimeoutError",
                "message": "x" * 900,
                "traceback": "y" * 5000,
            },
        }
        rec = col._collect_recovery(state)
        assert rec["recovered"] is True
        assert rec["crash_count"] == 2
        assert len(rec["crash_timestamps"]) == 2
        assert all(ts.endswith("+00:00") for ts in rec["crash_timestamps"])
        assert rec["degraded_mode"] is True
        assert rec["steward_continuation_used"] is True
        assert rec["resume_pending_revalidation"] is True
        assert rec["steward_infra_failures_total"] == 3
        assert rec["steward_infra_failures_by_round"] == {"12": 1, "13": 2}
        # Compact exception header; traceback dropped, message capped.
        assert rec["last_tick_exception"]["type"] == "TimeoutError"
        assert rec["last_tick_exception"]["tick"] == 12
        assert "traceback" not in rec["last_tick_exception"]
        assert len(rec["last_tick_exception"]["message"]) == 500

    def test_malformed_signals_are_skipped_not_raised(self):
        rec = col._collect_recovery(
            {
                "crash_count": "not-int",
                "crash_timestamps": ["bad", None, 1782800000.0],
                "steward_infra_failures_by_round": {"r": "x", "s": 4},
                "last_tick_exception": "not-a-dict",
            }
        )
        assert rec["crash_count"] == 0
        assert len(rec["crash_timestamps"]) == 1
        assert rec["steward_infra_failures_total"] == 4
        assert rec["last_tick_exception"] is None

    def test_collect_session_embeds_recovery_block(self, tmp_path):
        warnings: list[str] = []
        state = {"session_id": "s1", "crash_count": 1, "steward_continuation_used": True}
        out = col.collect_session(tmp_path, state, {}, warnings)
        assert "recovery" in out
        assert out["recovery"]["recovered"] is True
        assert out["recovery"]["crash_count"] == 1
