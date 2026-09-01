"""Coverage for the additive SBD V6 Recipe KB timeline events."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.breakdown.collectors.v6 import collect_v6_timeline
from hyperloom.inference_optimizer.breakdown.kb_timeline import (
    collect_kb_events,
    collect_kb_write_back_event,
    collect_warm_replay_event,
    collect_warm_start_event,
)


CID = "inference:qwen3-8b:mi355x:sglang:qwen3:qwen3forcausallm:0.5.17:bf16"
DONOR_CID = "inference:qwen3-32b:mi355x:sglang:qwen3:qwen3forcausallm:0.5.16:bf16"


def _scope_state(**overrides) -> dict:
    state = {
        "kernel_optimizer": "geak",
        "tp": 1,
        "conc": 64,
        "isl": 8192,
        "osl": 1024,
    }
    state.update(overrides)
    return state


def _matched_state(*, tier: str = "exact", **overrides) -> dict:
    state = _scope_state(
        warm_start_ts="2026-08-20T07:25:00+00:00",
        warm_start_lessons=[{"a": 1}, {"b": 2}],
        warm_start_pitfalls=[{"c": 3}],
        warm_start_recipe={
            "tier": tier,
            "confidence": 1.0 if tier == "exact" else 0.72,
            "recipe": {
                "canonical_id": CID,
                "best_throughput": 3239.9,
                "validated_gain_pct": 36.16,
                "replayable": True,
                "replay_material_available": True,
                "view_source": "current",
                "workload_shape": {"tp": 1, "conc": 64, "isl": 8192, "osl": 1024},
                "remote_session_id": "20260818T063226Z",
                "sessions": [{"session_id": "20260818T063226Z", "gain_pct": 36.23}],
                "provenance": {"session_id": "20260818T063226Z"},
            },
        },
        warm_start_context={
            "status": "hit",
            "match": {"tier": tier, "confidence": 1.0, "source": "kb-store", "canonical_id": CID},
            "recommended_replay": {"expected_gain_pct": 36.23},
        },
    )
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# warm_start
# ---------------------------------------------------------------------------


def test_warm_start_absent_emits_no_event():
    """A session that never reached T0 must not get an empty stage."""
    assert collect_warm_start_event(_scope_state()) is None


def test_warm_start_exact_hit_reports_identity_gain_and_origin():
    event = collect_warm_start_event(_matched_state())
    assert event["type"] == "warm_start"
    assert event["status"] == "matched"
    assert event["start_time"] == event["end_time"] == "2026-08-20T07:25:00+00:00"

    matched = event["ext"]["matched"]
    assert matched["match_type"] == "exact"
    assert matched["tier"] == "exact"
    assert matched["confidence"] == pytest.approx(1.0)
    assert matched["source"] == "kb-store"
    assert matched["optimized_throughput"] == pytest.approx(3239.9)
    assert matched["validated_gain_pct"] == pytest.approx(36.16)
    assert matched["expected_gain_pct"] == pytest.approx(36.23)
    assert matched["replay_material_available"] is True
    assert matched["origin"] == {"session_id": "20260818T063226Z", "gain_pct": pytest.approx(36.23)}
    assert matched["experience"] == {"lessons_count": 2, "pitfalls_count": 1}
    assert event["ext"]["requested"]["scope"] == {
        "kernel_optimizer": "geak",
        "tp": 1,
        "conc": 64,
        "isl": 8192,
        "osl": 1024,
    }


def test_warm_start_degraded_tier_is_not_reported_as_exact():
    """A hit that relaxed framework version must stay distinguishable."""
    event = collect_warm_start_event(_matched_state(tier="compatible_framework_version"))
    matched = event["ext"]["matched"]
    assert matched["match_type"] == "degraded"
    assert matched["tier"] == "compatible_framework_version"
    assert matched["confidence"] == pytest.approx(0.72)


def test_warm_start_requested_id_prefers_recorded_close_identity():
    """CLOSE derives the session's own identity; a degraded hit must not stand in for it."""
    state = _matched_state(
        tier="same_gpu_isa",
        recipe_finalize_outcome={"canonical_id": "inference:asked-for:mi355x:sglang:a:b:0.5.17:bf16"},
    )
    event = collect_warm_start_event(state)
    assert event["ext"]["requested"]["canonical_id"] == "inference:asked-for:mi355x:sglang:a:b:0.5.17:bf16"
    assert event["ext"]["matched"]["canonical_id"] == CID


def test_warm_start_degraded_hit_without_close_leaves_requested_blank():
    """Guessing the asked-for identity from a relaxed match would be wrong."""
    event = collect_warm_start_event(_matched_state(tier="same_arch_class"))
    assert event["ext"]["requested"]["canonical_id"] == ""


def test_warm_start_miss_carries_no_matched_block():
    state = _scope_state(
        warm_start_ts="2026-08-20T07:25:00+00:00",
        warm_start_context={"status": "miss", "match": {"tier": "miss", "confidence": 0.0}},
    )
    event = collect_warm_start_event(state)
    assert event["status"] == "not_matched"
    assert "matched" not in event["ext"]
    assert event["ext"]["match_status"] == "miss"


def test_warm_start_seed_only_keeps_its_raw_status():
    """``seed_only`` is a hit that could not be executed, not a plain miss."""
    state = _scope_state(
        warm_start_ts="2026-08-20T07:25:00+00:00",
        warm_start_context={"status": "seed_only", "match": {"tier": "exact"}},
    )
    event = collect_warm_start_event(state)
    assert event["status"] == "not_matched"
    assert event["ext"]["match_status"] == "seed_only"


def test_warm_start_error_status_maps_to_failed():
    state = _scope_state(
        warm_start_ts="2026-08-20T07:25:00+00:00",
        warm_start_context={"status": "error"},
    )
    assert collect_warm_start_event(state)["status"] == "failed"


def _write_recipe_audit(session_dir: Path, rows: list[dict]) -> None:
    from hyperloom.inference_optimizer.session.session_paths import recipe_snapshot_audit_jsonl

    audit = recipe_snapshot_audit_jsonl(session_dir)
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_warm_start_reads_attribution_migrated_from_recipe_audit(tmp_path: Path):
    """The retired ``kb_provenance.recipe_snapshot_reads`` now rides warm_start.ext."""
    _write_recipe_audit(
        tmp_path,
        [
            {
                "method": "get_recipe",
                "remote": "kb-store",
                "resolution": "remote",
                "hit": True,
                "result": {"sources": ["kb-store", "recipe_kb"], "best_config_source": "kb-store"},
            },
            {
                "method": "get_recipe",
                "remote": "recipe_kb",
                "resolution": "local",
                "hit": False,
                "result": {"sources": ["recipe_kb"]},
            },
        ],
    )
    events = collect_kb_events(tmp_path, _matched_state(), [])
    warm_start = next(event for event in events if event["type"] == "warm_start")
    reads = warm_start["ext"]["reads"]
    assert reads["count"] == 2
    assert reads["hits"] == 1
    assert reads["by_remote"] == {"kb-store": 1, "recipe_kb": 1}
    assert reads["by_resolution"] == {"remote": 1, "local": 1}
    assert reads["by_source"] == {"kb-store": 1, "recipe_kb": 2}
    assert reads["best_config_by_source"] == {"kb-store": 1}
    assert len(reads["tail"]) == 2


def test_warm_start_reads_absent_when_no_audit(tmp_path: Path):
    events = collect_kb_events(tmp_path, _matched_state(), [])
    warm_start = next(event for event in events if event["type"] == "warm_start")
    assert "reads" not in warm_start["ext"]


# ---------------------------------------------------------------------------
# warm_replay
# ---------------------------------------------------------------------------


def test_warm_replay_absent_emits_no_event():
    assert collect_warm_replay_event(_scope_state()) is None


def test_warm_replay_disabled_by_flag_records_reason_without_config():
    """``--no-warm-replay`` is an operator decision, not a KB failure."""
    state = _scope_state(
        warm_start_ts="2026-08-20T07:25:00+00:00",
        warm_replay_attempted=True,
        warm_replay_outcome={"status": "skipped", "reason": "disabled_by_flag"},
    )
    event = collect_warm_replay_event(state)
    assert event["status"] == "skipped"
    assert event["ext"]["result_type"] == "disabled_by_flag"
    assert event["ext"]["raw_reason"] == "disabled_by_flag"
    assert "applied" not in event["ext"]
    # Skips are decided inline, before a replay task exists.
    assert event["start_time"] == "2026-08-20T07:25:00+00:00"


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("no_warm_start_recipe", "no_warm_start_recipe"),
        ("best_config_empty", "best_config_empty"),
        ("confidence_below_threshold (0.50 < 0.70)", "confidence_below_threshold"),
        ("remote_recipe_view_not_replayable", "recipe_not_replayable"),
        ("current_recipe_sdk_read_failed:ValueError:boom", "recipe_read_failed"),
        ("active_framework_root_missing", "framework_root_missing"),
        ("workload_config_incompatible:ValueError:context_length=6144", "workload_config_incompatible"),
        ("no_matching_root", "patch_root_unresolved"),
        ("something nobody has seen", "skipped_other"),
    ],
)
def test_warm_replay_skip_reasons_map_to_stable_codes(reason, expected):
    state = _scope_state(warm_replay_outcome={"status": "skipped", "reason": reason})
    assert collect_warm_replay_event(state)["ext"]["result_type"] == expected


def test_warm_replay_reproduced_records_the_merged_running_config():
    """The columns are measured together, so one merged configuration is reported."""
    state = _scope_state(
        baseline_tput=2379.4,
        warm_replay_outcome={
            "status": "reproduced",
            "enqueued_at": "2026-08-20T08:00:00+00:00",
            "settled_at": "2026-08-20T08:40:00+00:00",
            "warm_recipe_tier": "exact",
            "warm_recipe_conf": 1.0,
            "config_source": CID,
            "expected_gain_pct": 36.23,
            "keep_threshold_pct": 1.0,
            "actual_gain_pct": 36.055,
            "throughput_after": 3237.2,
            "eval_ran": True,
            "baseline_accuracy": 0.9363,
            "replay_accuracy": 0.9361,
            "replayed_patch_refs": ["patch/overlays/000000/00-fix.patch"],
            "kernel": {"status": "kept", "total": 2, "kept": 2, "reverted": 0},
            "active_framework_root": "/sglang",
        },
        optimization_stack=[
            {
                "action": "replay_warm_recipe",
                "candidate_extra_server_args": "--enable-aiter",
                "candidate_extra_envs": {"SGLANG_USE_AITER": "1"},
                "kernel_replay": {"validation": "combined_recipe_kernel", "count": 2, "columns": ["fusion"]},
            }
        ],
    )
    event = collect_warm_replay_event(state)
    assert event["status"] == "reproduced"
    assert "result_type" not in event["ext"]
    assert event["start_time"] == "2026-08-20T08:00:00+00:00"
    assert event["end_time"] == "2026-08-20T08:40:00+00:00"

    applied = event["ext"]["applied"]
    assert applied["config"] == {
        "extra_server_args": "--enable-aiter",
        "extra_envs": {"SGLANG_USE_AITER": "1"},
    }
    assert applied["patch"] == ["patch/overlays/000000/00-fix.patch"]
    assert applied["kernel"]["kept"] == 2
    assert applied["kernel"]["columns"] == ["fusion"]

    assert event["ext"]["before_tput"] == pytest.approx(2379.4)
    assert event["ext"]["gain_pct"] == pytest.approx(36.055)
    assert event["ext"]["accuracy"]["passed"] is True


def test_warm_replay_drift_reports_reason_and_withholds_config():
    """A rejected replay already rolled its material back; only the reason is real."""
    state = _scope_state(
        baseline_tput=2379.4,
        warm_replay_outcome={
            "status": "drift",
            "reason": "measured +0.20% below keep threshold +1.00%",
            "actual_gain_pct": 0.2,
            "throughput_after": 2384.2,
            "keep_threshold_pct": 1.0,
        },
        optimization_stack=[
            {"action": "replay_warm_recipe", "candidate_extra_server_args": "--enable-aiter"},
        ],
    )
    event = collect_warm_replay_event(state)
    assert event["status"] == "drift"
    assert event["ext"]["result_type"] == "below_keep_threshold"
    assert "applied" not in event["ext"]
    # The measurement itself still has to be readable.
    assert event["ext"]["gain_pct"] == pytest.approx(0.2)
    assert event["ext"]["after_tput"] == pytest.approx(2384.2)


def test_warm_replay_positive_gain_under_historical_bar_is_called_out():
    state = _scope_state(
        warm_replay_outcome={
            "status": "drift",
            "reason": "measured +12.00% below keep threshold +20.00%",
            "below_historical_reproduce_pct": True,
            "historical_reproduce_bar_pct": 28.98,
        },
    )
    event = collect_warm_replay_event(state)
    assert event["ext"]["result_type"] == "below_historical_reproduce_bar"
    assert event["ext"]["below_historical_reproduce"] is True


def test_warm_replay_accuracy_rejection_keeps_its_measurement():
    """A gate rejection must stay distinguishable from a replay that never measured."""
    state = _scope_state(
        warm_replay_outcome={
            "status": "accuracy_failed",
            "reason": "accuracy regression on the replayed config (baseline 0.9363, replay 0.6001)",
            "actual_gain_pct": 41.2,
            "throughput_after": 3359.0,
            "eval_ran": True,
            "baseline_accuracy": 0.9363,
            "replay_accuracy": 0.6001,
        },
    )
    event = collect_warm_replay_event(state)
    assert event["status"] == "failed"
    assert event["ext"]["result_type"] == "accuracy_failed"
    assert event["ext"]["gain_pct"] == pytest.approx(41.2)
    assert event["ext"]["accuracy"]["passed"] is False
    assert "applied" not in event["ext"]


def test_warm_replay_rollback_failure_surfaces_the_errors():
    state = _scope_state(
        warm_replay_outcome={
            "status": "rollback_failed",
            "reason": "combined warm replay rollback failed",
            "rollback": {"ok": False, "errors": ["git checkout failed"]},
        },
    )
    event = collect_warm_replay_event(state)
    assert event["status"] == "failed"
    assert event["ext"]["result_type"] == "rollback_failed"
    assert event["ext"]["rollback"] == {"ok": False, "errors": ["git checkout failed"]}


def test_warm_replay_borrowed_donor_is_attributed():
    state = _scope_state(
        warm_replay_outcome={
            "status": "skipped",
            "reason": "best_config_empty",
            "config_donor_tier": "same_arch_class",
            "donor_canonical_id": DONOR_CID,
            "donor_model": "qwen3-32b",
            "donor_session_id": "20260812T101010Z",
            "donor_gain_pct": 18.4,
        },
    )
    donor = collect_warm_replay_event(state)["ext"]["donor"]
    assert donor["canonical_id"] == DONOR_CID
    assert donor["session_id"] == "20260812T101010Z"
    assert donor["gain_pct"] == pytest.approx(18.4)


def test_warm_replay_before_tput_falls_back_to_the_measured_pair():
    """A session with no recorded baseline still reports a usable anchor."""
    state = _scope_state(
        warm_replay_outcome={"status": "reproduced", "actual_gain_pct": 100.0, "throughput_after": 200.0},
    )
    assert collect_warm_replay_event(state)["ext"]["before_tput"] == pytest.approx(100.0)


def test_warm_replay_reproduced_without_params_is_named():
    state = _scope_state(
        warm_replay_outcome={
            "status": "reproduced_but_no_params",
            "reason": "task.params missing extra_server_args/extra_envs and no warm patch was applied",
        },
    )
    event = collect_warm_replay_event(state)
    assert event["status"] == "reproduced"
    assert event["ext"]["result_type"] == "reproduced_without_params"


# ---------------------------------------------------------------------------
# kb_write_back
# ---------------------------------------------------------------------------


def test_kb_write_back_absent_emits_no_event(tmp_path: Path):
    assert collect_kb_write_back_event(tmp_path, _scope_state(), []) is None


def test_kb_write_back_written_reports_identity_and_throughput(tmp_path: Path):
    state = _scope_state(
        current_best={"tput": 3239.9},
        cumulative_gain_validated=36.16,
        recipe_finalize_status="written",
        recipe_finalize_attempts=1,
        recipe_finalize_outcome={
            "status": "written",
            "reason": "",
            "backend": "kb-store",
            "canonical_id": CID,
            "session_id": "20260820T072455Z",
            "source": "close",
            "updated_at": "2026-08-20T17:34:00+00:00",
        },
    )
    warnings: list[str] = []
    event = collect_kb_write_back_event(tmp_path, state, warnings)
    assert event["status"] == "written"
    assert "result_type" not in event["ext"]
    assert event["ext"]["backend"] == "kb-store"
    assert event["ext"]["canonical_id"] == CID
    assert event["ext"]["optimized_throughput"] == pytest.approx(3239.9)
    assert event["ext"]["validated_gain_pct"] == pytest.approx(36.16)
    assert event["ext"]["scope"]["kernel_optimizer"] == "geak"
    assert event["ext"]["queue"] == {"pending_lines": 0, "flushed_bookmarks": 0, "dead_letter_lines": 0}
    assert warnings == []


def test_kb_write_back_counts_the_local_queue_depth(tmp_path: Path):
    queue_dir = tmp_path / "runtime" / "recipe_kb"
    queue_dir.mkdir(parents=True)
    (queue_dir / ".kb_pending.ndjson").write_text('{"a":1}\n\n{"a":2}\n', encoding="utf-8")
    (queue_dir / ".kb_dead_letter.ndjson").write_text('{"b":1}\n', encoding="utf-8")
    state = _scope_state(recipe_finalize_status="written", recipe_finalize_outcome={"status": "written"})
    event = collect_kb_write_back_event(tmp_path, state, [])
    assert event["ext"]["queue"] == {"pending_lines": 2, "flushed_bookmarks": 0, "dead_letter_lines": 1}


@pytest.mark.parametrize(
    ("status", "reason", "expected_status", "expected_code"),
    [
        ("skipped", "not_better_than_champion", "skipped", "not_better_than_champion"),
        ("skipped", "no_new_keep_or_pure_warm_replay", "skipped", "no_new_keep_or_pure_warm_replay"),
        ("skipped", "nonfinite_optimized_throughput", "skipped", "invalid_throughput"),
        ("skipped", "missing_optimized_throughput", "skipped", "missing_throughput"),
        ("skipped", "invalid_recipe_scope", "skipped", "invalid_scope"),
        ("skipped", "empty_replay_material", "skipped", "empty_replay_material"),
        ("skipped", "degraded_kb", "skipped", "kb_disabled"),
        ("disabled", "KB_STORE_URL/TOKEN not configured", "skipped", "kb_disabled"),
        ("skipped", "agentx", "skipped", "agentx_blocked"),
        ("failed", "configuration:KeyError", "failed", "configuration_failed"),
        ("failed", "RemoteRecipeValidationError", "failed", "bundle_build_failed"),
        ("failed", "KBStoreError", "failed", "transport_failed"),
    ],
)
def test_kb_write_back_reasons_map_to_stable_codes(
    tmp_path: Path,
    status,
    reason,
    expected_status,
    expected_code,
):
    state = _scope_state(
        recipe_finalize_status=status,
        recipe_finalize_outcome={"status": status, "reason": reason},
    )
    event = collect_kb_write_back_event(tmp_path, state, [])
    assert event["status"] == expected_status
    assert event["ext"]["result_type"] == expected_code
    assert event["ext"]["raw_reason"] == reason


def test_kb_write_back_champion_not_promoted_stays_written(tmp_path: Path):
    """The knowledge landed; only the Champion pointer lost a race."""
    state = _scope_state(
        recipe_finalize_status="written",
        recipe_finalize_outcome={"status": "written", "reason": "champion_not_promoted"},
    )
    event = collect_kb_write_back_event(tmp_path, state, [])
    assert event["status"] == "written"
    assert event["ext"]["result_type"] == "champion_not_promoted"


def test_kb_write_back_pending_marker_is_not_reported_as_written(tmp_path: Path):
    """A run that died mid-publish never had its write confirmed."""
    state = _scope_state(recipe_finalize_status="pending", recipe_finalize_outcome={"status": "pending"})
    assert collect_kb_write_back_event(tmp_path, state, [])["status"] == "failed"


# ---------------------------------------------------------------------------
# timeline integration
# ---------------------------------------------------------------------------


def test_kb_events_join_the_timeline_in_execution_order(tmp_path: Path):
    state = _matched_state(
        warm_replay_outcome={
            "status": "reproduced",
            "enqueued_at": "2026-08-20T08:00:00+00:00",
            "settled_at": "2026-08-20T08:40:00+00:00",
        },
        recipe_finalize_status="written",
        recipe_finalize_outcome={
            "status": "written",
            "canonical_id": CID,
            "updated_at": "2026-08-20T17:34:00+00:00",
        },
    )
    timeline = collect_v6_timeline(tmp_path, [], state=state, recorded_operations=[])
    assert [event["type"] for event in timeline] == ["warm_start", "warm_replay", "kb_write_back"]


def test_timeline_stays_empty_for_a_session_that_touched_no_kb(tmp_path: Path):
    assert collect_v6_timeline(tmp_path, [], state={}, recorded_operations=[]) == []
