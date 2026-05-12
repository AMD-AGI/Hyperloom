"""Phase 3 — `validate_stack` action end-to-end.

Covers four layers:

* **combine_optimization_stack** — pure function that merges
  args/envs from a list of stack entries. Filtered by
  ``include_actions`` / ``exclude_variants``; honours the
  ``candidate_extra_sglang_args`` precedence used by Coordinator.

* **ValidateStackExecutor** — re-uses BaselineExecutor's subprocess
  machinery; on a successful Magpie run, surfaces
  ``validated_stack_len`` / ``applied_args`` / ``applied_envs`` /
  ``applied_entries`` so the Coordinator can promote the result.

* **Coordinator promotion** — ``_is_promotable_result`` accepts a
  validate_stack result, ``_promote_to_shared_state`` writes the
  three ``cumulative_gain_validated*`` core fields atomically.

* **report.py** — final.md / final.json show both gains and surface
  the staleness warning when the stack changed since the last
  validation.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from inference_optimizer.orchestrator.action_executors.baseline import (
    _default_baseline_config,
)
from inference_optimizer.orchestrator.action_executors.validate_stack import (
    ValidateStackExecutor,
    combine_optimization_stack,
)
from inference_optimizer.orchestrator.action_executors.report import (
    _build_summary_dict, _format_md,
)
from inference_optimizer.orchestrator.backends import (
    MockBackend, ScriptedPlan,
)
from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.orchestrator.intent_parser import Intent, IntentType
from inference_optimizer.orchestrator.resource_lock import (
    ResourceLockManager, SqliteLeaseBackend,
)
from inference_optimizer.orchestrator.shared_state import SharedState
from inference_optimizer.orchestrator.sub_agent_runner import SubAgentRunner
from inference_optimizer.orchestrator.task_registry import TaskRegistry
from inference_optimizer.paths import make_session_dir
from inference_optimizer.storage import SqliteConnection


# ===========================================================================
# fixtures
# ===========================================================================
@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("INFERENCE_OPTIMIZER_SESSION_DIR", str(tmp_path))
    return make_session_dir()


def _silent() -> Intent:
    return Intent(
        type=IntentType.SEND_MESSAGE,
        payload={"topic": "heartbeat", "body_md": "ok"},
    )


def _backends() -> dict[str, object]:
    silent = ScriptedPlan(turns=[], default_intent=_silent())
    return {n: MockBackend(silent, name=n)
            for n in ("orchestration", "kernel", "critic", "robustness")}


# ===========================================================================
# combine_optimization_stack — pure helper
# ===========================================================================
def test_combine_empty_stack_yields_empty_args_and_envs():
    args, envs, applied = combine_optimization_stack([])
    assert args == ""
    assert envs == {}
    assert applied == []


def test_combine_concatenates_args_in_order():
    stack = [
        {
            "action": "backends", "variant_name": "aiter",
            "candidate_extra_sglang_args": "--attention-backend aiter",
            "extra_envs": {"AITER_USE_OOB": "0"},
        },
        {
            "action": "params", "variant_name": "graphs",
            "candidate_extra_sglang_args": "--cuda-graph-max-bs 256",
            "extra_envs": {"NCCL_DEBUG": "WARN"},
        },
    ]
    args, envs, applied = combine_optimization_stack(stack)
    # Args are concatenated in order. We don't normalise — sglang argv
    # parsing wins on conflicts already.
    assert args == "--attention-backend aiter --cuda-graph-max-bs 256"
    assert envs == {"AITER_USE_OOB": "0", "NCCL_DEBUG": "WARN"}
    assert [a["variant_name"] for a in applied] == ["aiter", "graphs"]


def test_combine_later_envs_override_earlier():
    stack = [
        {"action": "a", "variant_name": "v1", "extra_envs": {"X": "1", "Y": "a"}},
        {"action": "b", "variant_name": "v2", "extra_envs": {"X": "2", "Z": "z"}},
    ]
    _args, envs, _applied = combine_optimization_stack(stack)
    assert envs == {"X": "2", "Y": "a", "Z": "z"}


def test_combine_prefers_candidate_args_over_full_args():
    """When both candidate_extra_sglang_args and extra_sglang_args are set,
    the candidate (per-round delta) wins to avoid double-counting args
    that were already in the previous current_best."""
    stack = [{
        "action": "params", "variant_name": "vx",
        "candidate_extra_sglang_args": "--cuda-graph-max-bs 256",
        "extra_sglang_args": "--attention-backend aiter --cuda-graph-max-bs 256",
        "extra_envs": {},
    }]
    args, _envs, _applied = combine_optimization_stack(stack)
    assert args == "--cuda-graph-max-bs 256"


def test_combine_falls_back_to_extra_sglang_args_when_no_candidate():
    """Stack entries that came from `seed_stack_from_current_best` only
    have extra_sglang_args; the helper must still find them."""
    stack = [{
        "action": "backends", "variant_name": "legacy",
        "extra_sglang_args": "--attention-backend aiter",
        "extra_envs": {},
    }]
    args, _envs, _applied = combine_optimization_stack(stack)
    assert args == "--attention-backend aiter"


def test_combine_include_actions_filter():
    stack = [
        {"action": "backends", "variant_name": "v1",
         "candidate_extra_sglang_args": "--attention-backend aiter",
         "extra_envs": {}},
        {"action": "params", "variant_name": "v2",
         "candidate_extra_sglang_args": "--cuda-graph-max-bs 256",
         "extra_envs": {}},
        {"action": "integrate", "variant_name": "kernel-fix",
         "candidate_extra_sglang_args": "",
         "extra_envs": {"AITER_X": "1"}},
    ]
    args, envs, applied = combine_optimization_stack(
        stack, include_actions=["backends", "integrate"],
    )
    assert "aiter" in args
    assert "cuda-graph-max-bs" not in args
    assert envs == {"AITER_X": "1"}
    assert [a["action"] for a in applied] == ["backends", "integrate"]


def test_combine_exclude_variants_filter():
    stack = [
        {"action": "backends", "variant_name": "good",
         "candidate_extra_sglang_args": "--good", "extra_envs": {}},
        {"action": "backends", "variant_name": "bad",
         "candidate_extra_sglang_args": "--bad", "extra_envs": {}},
    ]
    args, _envs, applied = combine_optimization_stack(
        stack, exclude_variants=["bad"],
    )
    assert args == "--good"
    assert [a["variant_name"] for a in applied] == ["good"]


def test_combine_skips_non_dict_entries():
    """Defensive: state.json could in principle hold a stray non-dict."""
    stack = [
        None,
        "scrap",
        {"action": "backends", "variant_name": "ok",
         "candidate_extra_sglang_args": "--ok", "extra_envs": {}},
    ]
    args, _envs, applied = combine_optimization_stack(stack)  # type: ignore[arg-type]
    assert args == "--ok"
    assert len(applied) == 1


# ===========================================================================
# ValidateStackExecutor — end-to-end (subprocess mocked)
# ===========================================================================
def _write_fake_magpie_workspace(tmp_path: Path, *, output_throughput: float) -> Path:
    """Create a benchmark_*/benchmark_report.json under tmp_path."""
    workspace = tmp_path / "benchmark_sglang_20260501_001122"
    workspace.mkdir(parents=True)
    (workspace / "benchmark_report.json").write_text(json.dumps({
        "success": True,
        "framework": "sglang",
        "model": "/wekafs/models/Qwen-Qwen3-8B",
        "throughput": {
            "request_throughput": 3.2,
            "output_throughput": output_throughput,
            "total_token_throughput": output_throughput * 2,
            "completed_requests": 80,
            "duration_seconds": 60.0,
        },
        "latency": {"ttft": {"mean_ms": 140}, "e2el": {"mean_ms": 2500}},
    }))
    return workspace


@pytest.mark.asyncio
async def test_validate_stack_executor_combines_stack_and_returns_validated_fields(
    tmp_path,
):
    """End-to-end: state.json lists 2 KEEPs, executor merges them and
    surfaces validated_stack_len/applied_args/applied_envs/applied_entries
    on top of a successful baseline result."""
    db = SqliteConnection(tmp_path / "validate.db")
    locks = ResourceLockManager(SqliteLeaseBackend(db))
    tr = TaskRegistry(db)
    sub = SubAgentRunner(locks, tr)

    # Persist a SharedState with a non-trivial optimization_stack.
    state = SharedState(
        baseline_tput=1000.0,
        optimization_stack=[
            {
                "action": "backends", "variant_name": "aiter",
                "candidate_extra_sglang_args": "--attention-backend aiter",
                "extra_envs": {"AITER_USE_OOB": "0"},
            },
            {
                "action": "params", "variant_name": "graphs",
                "candidate_extra_sglang_args": "--cuda-graph-max-bs 256",
                "extra_envs": {"NCCL_DEBUG": "WARN"},
            },
        ],
    )
    state.save(tmp_path)

    # Mock Magpie subprocess: write workspace + return rc=0
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    workspace = _write_fake_magpie_workspace(output_dir, output_throughput=1100.0)
    fake_completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="", stderr="",
    )

    task = await tr.create(
        kind="validate_stack",
        params={
            "output_dir": str(output_dir),
            "config_path": str(_default_baseline_config()),
        },
        idempotency_key="validate-1",
    )
    sub.register_executor("validate_stack", ValidateStackExecutor(session_dir=tmp_path))

    with patch("subprocess.run", return_value=fake_completed):
        res = await sub.run_task(task)

    assert res.state == "succeeded"
    assert res.result["status"] == "succeeded"
    assert res.result["output_throughput"] == 1100.0
    # Phase 3-specific fields
    assert res.result["validated_stack_len"] == 2
    assert "--attention-backend aiter" in res.result["applied_args"]
    assert "--cuda-graph-max-bs 256" in res.result["applied_args"]
    assert res.result["applied_envs"]["AITER_USE_OOB"] == "0"
    assert res.result["applied_envs"]["NCCL_DEBUG"] == "WARN"
    assert [e["variant_name"] for e in res.result["applied_entries"]] == [
        "aiter", "graphs",
    ]
    db.close()


@pytest.mark.asyncio
async def test_validate_stack_executor_warns_when_stack_empty(tmp_path):
    """Empty optimization_stack -> warning logged but executor still runs
    a clean baseline (validated_stack_len=0)."""
    db = SqliteConnection(tmp_path / "validate.db")
    locks = ResourceLockManager(SqliteLeaseBackend(db))
    tr = TaskRegistry(db)
    sub = SubAgentRunner(locks, tr)
    SharedState(baseline_tput=1000.0, optimization_stack=[]).save(tmp_path)

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    _write_fake_magpie_workspace(output_dir, output_throughput=1000.5)
    fake_completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="", stderr="",
    )

    task = await tr.create(
        kind="validate_stack",
        params={"output_dir": str(output_dir),
                "config_path": str(_default_baseline_config())},
        idempotency_key="validate-empty",
    )
    sub.register_executor(
        "validate_stack", ValidateStackExecutor(session_dir=tmp_path),
    )
    with patch("subprocess.run", return_value=fake_completed):
        res = await sub.run_task(task)

    assert res.state == "succeeded"
    assert res.result["validated_stack_len"] == 0
    assert res.result["applied_args"] == ""
    assert res.result["applied_envs"] == {}
    db.close()


@pytest.mark.asyncio
async def test_validate_stack_executor_accepts_explicit_stack_param(tmp_path):
    """Passing ``stack`` via task.params bypasses SharedState."""
    db = SqliteConnection(tmp_path / "validate.db")
    locks = ResourceLockManager(SqliteLeaseBackend(db))
    tr = TaskRegistry(db)
    sub = SubAgentRunner(locks, tr)

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    _write_fake_magpie_workspace(output_dir, output_throughput=1234.5)
    fake_completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="", stderr="",
    )

    task = await tr.create(
        kind="validate_stack",
        params={
            "output_dir": str(output_dir),
            "config_path": str(_default_baseline_config()),
            "stack": [{
                "action": "params", "variant_name": "explicit",
                "candidate_extra_sglang_args": "--mem-fraction-static 0.85",
                "extra_envs": {},
            }],
        },
        idempotency_key="validate-explicit",
    )
    sub.register_executor(
        "validate_stack", ValidateStackExecutor(session_dir=tmp_path),
    )
    with patch("subprocess.run", return_value=fake_completed):
        res = await sub.run_task(task)

    assert res.state == "succeeded"
    assert res.result["validated_stack_len"] == 1
    assert "--mem-fraction-static 0.85" in res.result["applied_args"]
    db.close()


# ===========================================================================
# Coordinator promotion
# ===========================================================================
@pytest.mark.asyncio
async def test_coordinator_promotes_validate_stack_result(session_dir):
    coord = Coordinator(session_dir, backends=_backends())
    coord.shared_state.baseline_tput = 1000.0
    coord.shared_state.optimization_stack = [
        {"action": "backends", "variant_name": "aiter",
         "candidate_extra_sglang_args": "--x", "extra_envs": {}},
    ]
    payload = {
        "status": "succeeded",
        "output_throughput": 1100.0,
        "completed_requests": 80,
        "validated_stack_len": 1,
        "applied_args": "--x",
        "applied_envs": {},
        "applied_entries": [],
        "workspace": "/tmp/validate",
    }
    assert coord._is_promotable_result("validate_stack", payload)

    await coord._promote_to_shared_state("validate_stack", payload)

    assert coord.shared_state.cumulative_gain_validated == pytest.approx(10.0)
    assert coord.shared_state.cumulative_gain_validated_stack_len == 1
    assert coord.shared_state.cumulative_gain_validated_ts != ""
    # Validate must NOT touch current_best / cumulative_gain (it's a
    # measurement, not a new modification).
    assert coord.shared_state.cumulative_gain == 0.0
    # And the validate_stack TODO clears once the lengths line up.
    assert not coord.shared_state.optimization_stack_has_unvalidated_keeps()


@pytest.mark.asyncio
async def test_coordinator_pins_validation_to_executor_reported_length(session_dir):
    """If a new KEEP sneaks in between executor read and Coordinator
    write, we must record the length the executor actually re-bench'd
    against — not the current length."""
    coord = Coordinator(session_dir, backends=_backends())
    coord.shared_state.baseline_tput = 1000.0
    coord.shared_state.optimization_stack = [
        {"action": "backends", "variant_name": "v1",
         "candidate_extra_sglang_args": "--a", "extra_envs": {}},
        {"action": "params", "variant_name": "v2",
         "candidate_extra_sglang_args": "--b", "extra_envs": {}},
    ]
    payload = {
        "status": "succeeded",
        "output_throughput": 1050.0,
        "completed_requests": 80,
        "validated_stack_len": 1,  # executor only saw 1 entry
    }
    await coord._promote_to_shared_state("validate_stack", payload)

    assert coord.shared_state.cumulative_gain_validated_stack_len == 1
    # Stack now has 2 entries vs validated_at_len=1 → still unvalidated.
    assert coord.shared_state.optimization_stack_has_unvalidated_keeps()


@pytest.mark.asyncio
async def test_coordinator_skips_validate_stack_promotion_on_failure(session_dir):
    coord = Coordinator(session_dir, backends=_backends())
    coord.shared_state.baseline_tput = 1000.0
    coord.shared_state.optimization_stack = [
        {"action": "backends", "variant_name": "v1",
         "candidate_extra_sglang_args": "--x", "extra_envs": {}},
    ]
    payload = {
        "status": "failed",
        "output_throughput": 0.0,
        "error_class": "subprocess_nonzero",
    }
    assert not coord._is_promotable_result("validate_stack", payload)
    # Even if the bug let it through, _promote_to_shared_state must not
    # touch the validated fields when output_throughput is non-positive.
    await coord._promote_to_shared_state("validate_stack", payload)
    assert coord.shared_state.cumulative_gain_validated == 0.0
    assert coord.shared_state.cumulative_gain_validated_ts == ""


# ===========================================================================
# report.py — both gains visible
# ===========================================================================
def test_report_summary_includes_both_gains():
    state = SharedState(
        session_id="test-sess",
        model_name="m1", model_path="/m1", model_class="dense",
        stop_reason="goal_reached",
        baseline_tput=1000.0,
        cumulative_gain=12.0,
        cumulative_gain_validated=8.5,
        cumulative_gain_validated_ts="2026-05-11T12:00:00+00:00",
        cumulative_gain_validated_stack_len=2,
        optimization_stack=[{"a": 1}, {"a": 2}],
        max_minutes=120,
    )
    summary = _build_summary_dict(state, ev_counts={}, highlights=[])
    assert summary["cumulative_gain"] == 12.0
    assert summary["cumulative_gain_validated"] == 8.5
    assert summary["cumulative_gain_validated_ts"] == "2026-05-11T12:00:00+00:00"
    assert summary["cumulative_gain_validated_stack_len"] == 2
    assert summary["optimization_stack_len"] == 2

    md = _format_md(summary)
    # Both numbers visible
    assert "12.00%" in md
    assert "8.50%" in md
    assert "per-round sum" in md
    assert "validated_at_stack_len=2" in md
    # Stack length matches validation length → no staleness warning
    assert "stack changed since validation" not in md


def test_report_md_flags_stale_validation():
    state = SharedState(
        session_id="test-sess",
        baseline_tput=1000.0,
        cumulative_gain=12.0,
        cumulative_gain_validated=8.5,
        cumulative_gain_validated_ts="2026-05-11T12:00:00+00:00",
        cumulative_gain_validated_stack_len=1,
        optimization_stack=[{"a": 1}, {"a": 2}, {"a": 3}],
        max_minutes=120,
    )
    summary = _build_summary_dict(state, ev_counts={}, highlights=[])
    md = _format_md(summary)
    assert "stack changed since validation" in md


def test_report_md_warns_when_never_validated():
    state = SharedState(
        session_id="test-sess",
        baseline_tput=1000.0,
        cumulative_gain=12.0,
        cumulative_gain_validated=0.0,
        cumulative_gain_validated_ts="",
        cumulative_gain_validated_stack_len=0,
        optimization_stack=[{"a": 1}],
        max_minutes=120,
    )
    summary = _build_summary_dict(state, ev_counts={}, highlights=[])
    md = _format_md(summary)
    assert "never validated" in md
    assert "no `validate_stack` action ran" in md
