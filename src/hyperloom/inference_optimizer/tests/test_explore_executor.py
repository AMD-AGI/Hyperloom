# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""ExploreExecutor and explore_search ledger tests."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from hyperloom.orchestrator.actions.executors import (
    ExploreExecutor,
)
from hyperloom.orchestrator.actions.executors._canonical_fingerprint import (
    canonical_fingerprint,
)
from hyperloom.orchestrator.actions.executors._grid_runner import (
    GridVariant,
    apply_compatibility_filter,
)
from hyperloom.orchestrator.actions.executors._subprocess_kill import (
    ORCHESTRATOR_CANCELLED_RETURNCODE,
    SESSION_TIME_EXHAUSTED_RETURNCODE,
)
from hyperloom.orchestrator.actions.stop_attribution import (
    ORCHESTRATOR_CANCELLED_CLASS,
    SESSION_TIME_EXHAUSTED_CLASS,
)
from hyperloom.orchestrator.actions.executors._accuracy_gate import (
    _RUN_EVAL_FALSE_VALUES as _RUN_EVAL_FALSE,
)
from hyperloom.orchestrator.actions.executors.explore import (
    _atom_default_grid,
    _default_grid_for_framework,
)
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.bus.resource_lock import (
    ResourceLockManager,
    SqliteLeaseBackend,
)
from hyperloom.orchestrator.loop.sub_agent_runner import SubAgentRunner
from hyperloom.orchestrator.state.task_registry import TaskRegistry
from hyperloom.orchestrator.bus.storage import SqliteConnection


@pytest.fixture(autouse=True)
def _isolate_leak_root(tmp_path_factory, monkeypatch):
    sandbox = tmp_path_factory.mktemp("isolated_leak_root_m3")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_LEAK_ROOTS", str(sandbox))


def _force_cold_decision(monkeypatch) -> None:
    """Make server_lifecycle reuse ineligible, one of the two warm-decision preconditions."""
    monkeypatch.setattr(
        "hyperloom.orchestrator.actions.executors.explore.resolve_lifecycle_params",
        lambda _config_path: {
            "eligible": False,
            "framework": "sglang",
            "port": 30000,
            "reason": "test: server_lifecycle reuse disabled",
        },
    )


def _write_baseline_yaml(path: Path) -> None:
    cfg = {
        "benchmark": {
            "framework": "sglang",
            "model": "/path/models/Qwen-Qwen3-8B",
            "precision": "bf16",
            "run_mode": "local",
            "envs": {"TP": 1, "CONC": 8, "ISL": 256, "OSL": 256},
            "benchmark_script": "sglang_mi300x.sh",
            "timeout_seconds": 600,
            "profiler": {
                "torch_profiler": {"enabled": False},
                "system_profiler": {"enabled": False},
                "tracelens": {"enabled": False},
            },
            "gpu_selection": {"auto": False},
        },
    }
    with path.open("w") as f:
        yaml.safe_dump(cfg, f)


def _fake_workspace(slot: Path, *, tput: float = 800.0, perf_axes: dict[str, float] | None = None) -> Path:
    workspace = slot / "benchmark_sglang_20260519_001122"
    workspace.mkdir(parents=True)
    (workspace / "benchmark_report.json").write_text(
        json.dumps(
            {
                "success": True,
                "framework": "sglang",
                "model": "/path/models/Qwen-Qwen3-8B",
                "throughput": {
                    "request_throughput": tput / 256,
                    "output_throughput": tput,
                    "total_token_throughput": (
                        tput * 2 if perf_axes is None else perf_axes.get("total_token_throughput")
                    ),
                    "completed_requests": 80,
                    "duration_seconds": 25.0,
                },
                "latency": {
                    "ttft": {"mean_ms": 140.0, "p99_ms": 160.0},
                    "e2el": {"mean_ms": 2500.0, "p99_ms": 2560.0},
                },
            }
        )
    )
    if perf_axes is not None:
        (workspace / "inferencex_result.json").write_text(json.dumps({"output_throughput": tput, **perf_axes}))
    return workspace


@pytest.fixture
def sub_agent_runner(tmp_path):
    db = SqliteConnection(tmp_path / "db.db")
    locks = ResourceLockManager(SqliteLeaseBackend(db))
    tr = TaskRegistry(db)
    runner = SubAgentRunner(locks, tr)
    yield runner, tr, tmp_path
    db.close()


def test_canonical_fingerprint_collapses_renames():
    """Two variants with identical content collapse to the same fingerprint."""
    fp_a = canonical_fingerprint("--max-num-seqs 256", {})
    fp_b = canonical_fingerprint("--max-num-seqs 256", {})
    assert fp_a == fp_b

    # Argument order doesn't matter (sorted token tuple).
    fp_c = canonical_fingerprint("--max-num-seqs 256 --block-size 128", {})
    fp_d = canonical_fingerprint("--block-size 128 --max-num-seqs 256", {})
    assert fp_c == fp_d


def test_canonical_fingerprint_distinguishes_envs():
    fp_args = canonical_fingerprint("--max-num-seqs 256", {})
    fp_args_envs = canonical_fingerprint(
        "--max-num-seqs 256",
        {"VLLM_ROCM_USE_AITER": "1"},
    )
    assert fp_args != fp_args_envs


def test_record_explore_accepted_dedup_by_fingerprint():
    state = SharedState()
    variant = {
        "name": "vllm_kv_fp8",
        "extra_server_args": "--kv-cache-dtype fp8",
        "extra_envs": {},
        "output_throughput": 1500.0,
        "gain_pct": 4.0,
        "provenance": "llm_direct",
    }
    state.record_explore_accepted(variant)
    state.record_explore_accepted(variant)
    assert len(state.explore_search["accepted"]) == 1
    assert len(state.explore_search["winners_history"]) == 2


def test_apply_explore_search_update_preserves_accepted():
    state = SharedState()
    state.record_explore_accepted(
        {
            "name": "a",
            "extra_server_args": "--flag-a",
            "fingerprint": "aa" * 8,
            "gain_pct": 3.0,
        }
    )
    # Update arrives with tested/rejected but NOT accepted; the bucket survives the merge.
    state.apply_explore_search_update(
        {
            "schema_version": 1,
            "tested": {
                "bb" * 8: {
                    "name": "b",
                    "extra_server_args": "--flag-b",
                    "extra_envs": {},
                    "outcome": "REVERT",
                }
            },
            "rejected": [
                {
                    "fingerprint": "bb" * 8,
                    "name": "b",
                    "reason": "not_keep",
                }
            ],
            "name_index": {"b": "bb" * 8},
            "cursor": 1,
            "last_round": {"round_id": "explore-001"},
        }
    )
    assert len(state.explore_search["accepted"]) == 1
    assert state.explore_search["accepted"][0]["name"] == "a"
    assert "bb" * 8 in state.explore_search["tested"]


def _round_update(round_no: int, fingerprint: str, name: str) -> dict:
    """One round's worth of executor ledger writes."""
    return {
        "schema_version": 1,
        "tested": {fingerprint: {"name": name, "outcome": "REVERT"}},
        "rejected": [{"fingerprint": fingerprint, "name": name, "reason": "not_keep"}],
        "name_index": {name: fingerprint},
        "last_round": {"round_id": f"explore-{round_no:03d}"},
    }


def test_apply_explore_search_update_accumulates_across_rounds():
    """The executor reports one round; the ledger has to remember the ones before it."""
    state = SharedState()
    state.apply_explore_search_update(_round_update(1, "aa" * 8, "a"))
    state.apply_explore_search_update(_round_update(2, "bb" * 8, "b"))

    search = state.explore_search
    assert set(search["tested"]) == {"aa" * 8, "bb" * 8}
    assert {r["fingerprint"] for r in search["rejected"]} == {"aa" * 8, "bb" * 8}
    assert search["name_index"] == {"a": "aa" * 8, "b": "bb" * 8}


def test_apply_explore_search_update_remeasured_fingerprint_replaces_its_row():
    state = SharedState()
    state.apply_explore_search_update(_round_update(1, "aa" * 8, "a"))
    second = _round_update(2, "aa" * 8, "a")
    second["tested"]["aa" * 8]["outcome"] = "KEEP"
    state.apply_explore_search_update(second)

    assert state.explore_search["tested"]["aa" * 8]["outcome"] == "KEEP"
    assert len(state.explore_search["rejected"]) == 1


def test_apply_explore_search_update_advances_the_cursor_per_round():
    """The cursor is the round ordinal: round ids and idempotency keys derive from it."""
    state = SharedState()
    state.apply_explore_search_update(_round_update(1, "aa" * 8, "a"))
    assert state.explore_search["cursor"] == 1
    # A round that benched three variants still advances the ordinal by one.
    third = _round_update(2, "bb" * 8, "b")
    third["tested"].update({"cc" * 8: {"name": "c"}, "dd" * 8: {"name": "d"}})
    state.apply_explore_search_update(third)
    assert state.explore_search["cursor"] == 2


def test_warm_history_rejected_rows_survive_the_first_round():
    """Warm-start pre-fills ``rejected`` so the dedup gate denies re-tests; a round must not wipe it."""
    state = SharedState()
    state.explore_search = {
        "rejected": [{"fingerprint": "ee" * 8, "name": "warm", "reason": "warm_recipe_what_failed"}]
    }
    state.apply_explore_search_update(_round_update(1, "aa" * 8, "a"))
    assert {r["fingerprint"] for r in state.explore_search["rejected"]} == {"ee" * 8, "aa" * 8}


@pytest.mark.asyncio
async def test_explore_executor_keeps_and_reverts_per_variant(sub_agent_runner, tmp_path):
    sub, tr, _ = sub_agent_runner
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    output_dir = tmp_path / "explore-out"

    call_counter = {"i": 0}

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        slug = slot.parent.name + "/" + slot.name
        if "v00_v_keep" in slug:
            tput = 840.0  # +5% vs base 800
        elif "v01_v_revert" in slug:
            tput = 800.4  # +0.05% — below 1.0% threshold
        else:
            tput = 800.0
        _fake_workspace(slot, tput=tput)
        call_counter["i"] += 1
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="ok",
            stderr="",
        )

    grid = [
        {
            "name": "v_keep",
            "extra_args": "--keep-flag",
            "extra_envs": {},
            "provenance": "default_grid",
        },
        {
            "name": "v_revert",
            "extra_args": "--revert-flag",
            "extra_envs": {},
            "provenance": "llm_direct",
        },
    ]
    task = await tr.create(
        kind="explore",
        params={
            "config_path": str(base),
            "output_dir": str(output_dir),
            "base_tput": 800.0,
            "grid": grid,
        },
        idempotency_key="ex-1",
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        res = await sub.run_task(task)

    assert res.state == "succeeded"
    out = res.result
    assert out["status"] == "succeeded"
    assert {w["name"] for w in out["winners"]} == {"v_keep"}
    assert {lr["name"] for lr in out["losers"]} == {"v_revert"}
    ledger = out["explore_search_update"]
    assert set(ledger["tested"].keys()) == {
        canonical_fingerprint("--keep-flag", {}),
        canonical_fingerprint("--revert-flag", {}),
    }
    outcomes = {v["name"]: v["outcome"] for v in ledger["tested"].values()}
    assert outcomes["v_keep"] == "KEEP"
    assert outcomes["v_revert"] == "REVERT"
    assert out["best_variant"]["name"] == "v_keep"
    assert out["best_gain_pct"] >= 4.0
    rejected_provenance = {r["provenance"] for r in ledger["rejected"]}
    assert rejected_provenance == {"llm_direct"}


@pytest.mark.asyncio
async def test_actual_explore_axis_rejection_cannot_be_revived_by_geak_fallback(
    sub_agent_runner, tmp_path, monkeypatch
):
    from hyperloom.inference_optimizer.tests.test_geak_gain_alignment import _coord, _ok_result

    _force_cold_decision(monkeypatch)
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    monkeypatch.delenv("HYPERLOOM_PERF_METRIC", raising=False)
    sub, tr, _ = sub_agent_runner
    coord = _coord(tmp_path, baseline=100.0, best_tput=110.0)
    state = coord.shared_state
    state.framework = "sglang"
    state.benchmark_mode = "agentx"
    state.baseline_perf = {"total_throughput": 1000.0, "e2e_norm_intvty_p90": 100.0}
    state.current_best.update(state.baseline_perf)
    state.geak_result = _ok_result(final=150.0)
    sub.shared_state = state
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    fingerprint = canonical_fingerprint("--test-flag", {})
    task = await tr.create(
        kind="explore",
        params={
            "config_path": str(base),
            "output_dir": str(tmp_path / "axis-rejection"),
            "base_tput": 110.0,
            "grid": [{"name": "candidate", "extra_args": "--test-flag"}],
            "source": "resume_stack_revalidate",
            "geak_fallback": True,
            "expected_cfg_hash": fingerprint,
        },
        idempotency_key="geak-axis-rejection",
    )
    state.geak_pending = {"status": "awaiting_rebench", "revalidation_task_id": task.task_id}
    state.resume_pending_revalidation = True
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))

    def fake_measure(cmd, *args, **kwargs):
        slot = Path(cmd[cmd.index("--output-dir") + 1])
        _fake_workspace(slot, tput=132.0, perf_axes={"total_token_throughput": 900.0, "e2e_norm_intvty_p90": 50.0})
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    async def must_not_replay(**kwargs):
        pytest.fail("native rejection must settle the candidate before any favorable fallback can run")

    coord._validate_geak_via_geak_harness = must_not_replay
    with patch("hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill", side_effect=fake_measure):
        produced = (await sub.run_task(task)).result
    rejection = produced["per_variant_outcomes"][0]
    assert rejection["reason"].startswith("both_axes_regressed")
    assert any(gate["gate"] == "graded_axes" and gate["passed"] is False for gate in rejection["gates"])
    await coord._promote_to_shared_state("explore", produced, task=task)
    assert state.current_best["tput"] == 110.0
    assert state.geak_result["revalidation_status"] == "no_promote"
    assert state.geak_result["revalidation_error"] == rejection["reason"]
    assert not state.optimization_stack


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_from", ["candidate", "current_best", "baseline"])
@pytest.mark.parametrize("missing_axis", ["total", "intvty"])
@pytest.mark.parametrize("output", [20000.0, 180.0])
async def test_explore_missing_axes_fails_closed(
    sub_agent_runner, tmp_path, monkeypatch, missing_from, missing_axis, output
):
    """Incomplete AgentX evidence fails closed instead of KEEPing on output throughput."""
    _force_cold_decision(monkeypatch)
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    monkeypatch.delenv("HYPERLOOM_PERF_METRIC", raising=False)
    sub, tr, _ = sub_agent_runner
    state = SharedState(framework="sglang", benchmark_mode="agentx")
    state.baseline_tput = 200.0
    state.baseline_perf = {
        "output_throughput": 200.0,
        "total_token_throughput": 20000.0,
        "e2e_norm_intvty_p90": 300.0,
    }
    base_tput = state.baseline_tput
    if missing_from == "current_best":
        state.current_best = {
            "action": "explore",
            "tput": 250.0,
            "total_token_throughput": 25000.0,
            "e2e_norm_intvty_p90": 300.0,
        }
        base_tput = 250.0
    candidate_axes = {
        "input_throughput": 20000.0,
        "total_token_throughput": 40000.0,
        "e2e_norm_intvty_p90": 300.0,
    }
    incomplete = {
        "candidate": candidate_axes,
        "current_best": state.current_best,
        "baseline": state.baseline_perf,
    }[missing_from]
    missing_keys = (
        ("input_throughput", "total_token_throughput") if missing_axis == "total" else ("e2e_norm_intvty_p90",)
    )
    for key in missing_keys:
        incomplete.pop(key, None)
    baseline_before = dict(state.baseline_perf)
    best_before = dict(state.current_best)
    sub.shared_state = state
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)

    def _fake_run(cmd, *args, **kwargs):
        slot = Path(cmd[cmd.index("--output-dir") + 1])
        _fake_workspace(slot, tput=output, perf_axes=candidate_axes)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    task = await tr.create(
        kind="explore",
        params={
            "config_path": str(base),
            "output_dir": str(tmp_path / "explore-missing-axes"),
            "base_tput": base_tput,
            "grid": [{"name": "v_incomplete", "extra_args": "--incomplete-flag"}],
        },
        idempotency_key="ex-missing-axes",
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        res = await sub.run_task(task)

    out = res.result
    tested = out["explore_search_update"]["tested"][canonical_fingerprint("--incomplete-flag", {})]
    assert tested["status"] == "succeeded"
    assert tested["outcome"] == "FAILED"
    assert tested["graded_objective"] == "output_throughput"
    assert tested["tput"] == output
    assert tested["base_tput"] == base_tput
    assert tested["gain_pct"] is None
    assert out["winners"] == []
    assert out["best_variant"] is None
    assert out["output_throughput"] is None
    assert out["running_base_tput"] == base_tput
    assert any(gate["gate"] == "graded_axes" and gate["passed"] is False for gate in tested["gates"])
    assert state.baseline_perf == baseline_before
    assert state.current_best == best_before


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_axis", ["total", "intvty"])
@pytest.mark.parametrize("incomplete_output", [170.0, 20000.0])
@pytest.mark.parametrize(
    "next_intvty,next_total,intvty_outcome",
    [(363.0, 23000.0, "KEEP"), (313.0, 20000.0, "REVERT"), (335.0, 22000.0, "RECORDED")],
)
async def test_explore_missing_axes_preserves_running_grading_anchor(
    sub_agent_runner, tmp_path, monkeypatch, missing_axis, incomplete_output, next_intvty, next_total, intvty_outcome
):
    """An incomparable variant fails closed and leaves the intvty anchor on the last KEEP."""
    _force_cold_decision(monkeypatch)
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    monkeypatch.delenv("HYPERLOOM_PERF_METRIC", raising=False)
    sub, tr, _ = sub_agent_runner
    state = SharedState(framework="sglang", benchmark_mode="agentx")
    state.baseline_tput = 200.0
    state.baseline_perf = {
        "output_throughput": 200.0,
        "total_token_throughput": 20000.0,
        "e2e_norm_intvty_p90": 300.0,
    }
    sub.shared_state = state
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    observed_args: dict[str, str] = {}

    def _fake_run(cmd, *args, **kwargs):
        slot = Path(cmd[cmd.index("--output-dir") + 1])
        name = slot.parent.name
        config = yaml.safe_load(Path(cmd[cmd.index("--benchmark-config") + 1]).read_text())
        observed_args[name] = config["benchmark"]["envs"]["EXTRA_SGLANG_ARGS"]
        output, total, intvty = {
            "v00_v_good": (180.0, 22000.0, 330.0),
            "v01_v_incomplete": (incomplete_output, 40000.0, 360.0),
            "v02_v_next": (160.0, next_total, next_intvty),
        }[name]
        axes = {
            "input_throughput": total - output,
            "total_token_throughput": total,
            "e2e_norm_intvty_p90": intvty,
        }
        if name == "v01_v_incomplete":
            for key in (
                ("input_throughput", "total_token_throughput") if missing_axis == "total" else ("e2e_norm_intvty_p90",)
            ):
                axes.pop(key)
        _fake_workspace(slot, tput=output, perf_axes=axes)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    task = await tr.create(
        kind="explore",
        params={
            "config_path": str(base),
            "output_dir": str(tmp_path / "explore-anchor-sequence"),
            "base_tput": 200.0,
            "grid": [
                {"name": "v_good", "extra_args": "--good-flag"},
                {"name": "v_incomplete", "extra_args": "--incomplete-flag"},
                {"name": "v_next", "extra_args": "--next-flag"},
            ],
        },
        idempotency_key="ex-anchor-sequence",
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        res = await sub.run_task(task)

    out = res.result
    tested = {row["name"]: row for row in out["explore_search_update"]["tested"].values()}
    assert out["status"] == "succeeded"
    assert len(tested) == 3
    assert tested["v_good"]["outcome"] == "KEEP"
    assert tested["v_good"]["graded_objective"] == "e2e_norm_intvty_p90"
    assert tested["v_good"]["gain_pct"] == pytest.approx(10.0)
    assert tested["v_good"]["tput"] == 180.0
    assert tested["v_incomplete"]["status"] == "succeeded"
    assert tested["v_incomplete"]["outcome"] == "FAILED"
    assert tested["v_incomplete"]["graded_objective"] == "output_throughput"
    assert tested["v_incomplete"]["base_tput"] == 180.0
    assert tested["v_next"]["base_tput"] == 180.0
    assert tested["v_next"]["graded_objective"] == "e2e_norm_intvty_p90"
    if intvty_outcome == "REVERT":
        assert tested["v_next"]["gain_pct"] is None
    else:
        assert tested["v_next"]["gain_pct"] == pytest.approx((next_intvty / 330.0 - 1.0) * 100.0)
    assert tested["v_next"]["outcome"] == intvty_outcome
    assert "--good-flag" in observed_args["v02_v_next"]
    assert "--incomplete-flag" not in observed_args["v02_v_next"]
    assert "--next-flag" in observed_args["v02_v_next"]
    expected_winners = ["v_good"] + (["v_next"] if intvty_outcome == "KEEP" else [])
    assert [row["name"] for row in out["winners"]] == expected_winners
    assert [row["variant_name"] for row in out["explore_search_update"]["winners_history"]] == expected_winners
    assert out["running_base_tput"] == (160.0 if intvty_outcome == "KEEP" else 180.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["explicit-output", "synthetic-legacy"])
@pytest.mark.parametrize("output,expected_outcome", [(220.0, "KEEP"), (180.0, "REVERT")])
async def test_explore_required_axes_output_modes_keep_legacy_behavior(
    sub_agent_runner, tmp_path, monkeypatch, mode, output, expected_outcome
):
    """Missing total/interactivity stays irrelevant when output grading is requested."""
    _force_cold_decision(monkeypatch)
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    monkeypatch.delenv("HYPERLOOM_PERF_METRIC", raising=False)
    sub, tr, _ = sub_agent_runner
    state = SharedState(framework="sglang", benchmark_mode="agentx" if mode == "explicit-output" else "")
    state.baseline_tput = 200.0
    if mode == "explicit-output":
        monkeypatch.setenv("HYPERLOOM_PERF_METRIC", "output_throughput")
    sub.shared_state = state
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)

    def _fake_run(cmd, *args, **kwargs):
        slot = Path(cmd[cmd.index("--output-dir") + 1])
        _fake_workspace(slot, tput=output, perf_axes={})
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    task = await tr.create(
        kind="explore",
        params={
            "config_path": str(base),
            "output_dir": str(tmp_path / "explore-output-mode"),
            "base_tput": 200.0,
            "grid": [{"name": "v_output", "extra_args": "--output-flag"}],
        },
        idempotency_key="ex-output-mode",
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        res = await sub.run_task(task)

    out = res.result
    tested = out["explore_search_update"]["tested"][canonical_fingerprint("--output-flag", {})]
    assert out["status"] == "succeeded"
    assert tested["status"] == "succeeded"
    assert tested["outcome"] == expected_outcome
    assert tested["graded_objective"] == "output_throughput"
    if expected_outcome == "KEEP":
        assert tested["gain_pct"] == pytest.approx((output / 200.0 - 1.0) * 100.0)
    else:
        assert tested["gain_pct"] is None
        assert out["losers"][0]["reason"] == "gain_below_threshold"
    assert bool(out["winners"]) is (expected_outcome == "KEEP")
    assert out["running_base_tput"] == (output if expected_outcome == "KEEP" else 200.0)


@pytest.mark.asyncio
async def test_explore_serving_no_eval_reverts_without_stopping(sub_agent_runner, tmp_path):
    """A high-risk serving variant that clears throughput but yields no accuracy verdict used to skip the gate (throughput-only fallback)."""
    sub, tr, _ = sub_agent_runner
    state = SharedState()
    state.baseline_tput = 800.0
    state.baseline_accuracy = 0.80
    sub.shared_state = state

    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    output_dir = tmp_path / "explore-acc-revert"

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_workspace(slot, tput=840.0)  # +5% vs base 800 (clears throughput)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    grid = [
        {
            # High-risk (precision) flag -> serving accuracy gate applies.
            "name": "v_risky",
            "extra_args": "--kv-cache-dtype fp8",
            "extra_envs": {},
            "provenance": "llm_direct",
        }
    ]
    task = await tr.create(
        kind="explore",
        params={
            "config_path": str(base),
            "output_dir": str(output_dir),
            "base_tput": 800.0,
            "accuracy_baseline": 0.80,
            "grid": grid,
        },
        idempotency_key="ex-acc-revert",
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        res = await sub.run_task(task)

    out = res.result
    assert out["status"] == "succeeded"
    fp = canonical_fingerprint("--kv-cache-dtype fp8", {})
    tested = out["explore_search_update"]["tested"][fp]
    assert tested["outcome"] == "REVERT"
    reasons = {lr["name"]: lr.get("reason") for lr in out["losers"]}
    assert reasons.get("v_risky") == "accuracy_unavailable"
    # Post-baseline accuracy failure reverts the variant but never halts the run.
    assert state.stop_reason == ""
    # The arc is carried gate by gate: it cleared the gain bar and died on
    # accuracy. The outcome alone cannot say which of the two ended it.
    row = next(v for v in out["per_variant_outcomes"] if v["variant_name"] == "v_risky")
    assert [(g["gate"], g["passed"]) for g in row["gates"]] == [
        ("keep_threshold", True),
        ("accuracy", False),
    ]
    assert row["gates"][1]["reason"] == "accuracy_unavailable"
    # Nothing was adopted, so nothing stands behind an adoption.
    assert row["validation_basis"] == ""


@pytest.mark.asyncio
async def test_explore_gates_a_variant_no_flag_catalogue_would_have_caught(sub_agent_runner, tmp_path):
    """The gate no longer asks which knobs look risky."""
    sub, tr, _ = sub_agent_runner
    state = SharedState()
    state.baseline_tput = 800.0
    state.baseline_accuracy = 0.80
    sub.shared_state = state

    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_workspace(slot, tput=840.0)  # +5% vs base 800 (clears throughput)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    extra_args = '--online_quant_config {"global_quant_config":"ptpc_fp8"}'
    task = await tr.create(
        kind="explore",
        params={
            "config_path": str(base),
            "output_dir": str(tmp_path / "explore-uncatalogued"),
            "base_tput": 800.0,
            "accuracy_baseline": 0.80,
            "grid": [
                {
                    "name": "v_quant",
                    "extra_args": extra_args,
                    "extra_envs": {},
                    "provenance": "llm_direct",
                }
            ],
        },
        idempotency_key="ex-uncatalogued-acc",
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        res = await sub.run_task(task)

    tested = res.result["explore_search_update"]["tested"][canonical_fingerprint(extra_args, {})]
    assert tested["outcome"] == "REVERT"
    reasons = {lr["name"]: lr.get("reason") for lr in res.result["losers"]}
    assert reasons.get("v_quant") == "accuracy_unavailable"
    assert state.stop_reason == ""


@pytest.mark.asyncio
async def test_explore_accuracy_gate_falls_back_to_shared_state(sub_agent_runner, tmp_path):
    sub, tr, _ = sub_agent_runner
    state = SharedState()
    state.baseline_tput = 800.0
    state.baseline_accuracy = 0.80
    sub.shared_state = state

    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_workspace(slot, tput=840.0)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    task = await tr.create(
        kind="explore",
        params={
            "config_path": str(base),
            "output_dir": str(tmp_path / "explore-shared-acc"),
            "base_tput": 800.0,
            "grid": [
                {
                    "name": "v_risky",
                    "extra_args": "--attention-backend ROCM_AITER_FA",
                    "extra_envs": {"VLLM_ROCM_USE_AITER": "1"},
                    "provenance": "llm_direct",
                }
            ],
        },
        idempotency_key="ex-shared-acc",
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        res = await sub.run_task(task)

    fp = canonical_fingerprint(
        "--attention-backend ROCM_AITER_FA",
        {"VLLM_ROCM_USE_AITER": "1"},
    )
    tested = res.result["explore_search_update"]["tested"][fp]
    assert tested["outcome"] == "REVERT"
    reasons = {lr["name"]: lr.get("reason") for lr in res.result["losers"]}
    assert reasons.get("v_risky") == "accuracy_unavailable"


@pytest.mark.asyncio
async def test_explore_executor_keep_persists_effective_removal_stack(sub_agent_runner, tmp_path):
    sub, tr, _ = sub_agent_runner
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_workspace(slot, tput=900.0)
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="ok",
            stderr="",
        )

    task = await tr.create(
        kind="explore",
        params={
            "config_path": str(base),
            "output_dir": str(tmp_path / "explore-out"),
            "base_tput": 800.0,
            "base_extra_args": "--bad-base 1 --keep-base 2",
            "grid": [
                {
                    "name": "remove_bad_base",
                    "extra_args": "--variant 4",
                    "remove_args": ["--bad-base"],
                    "provenance": "llm_direct",
                }
            ],
        },
        idempotency_key="ex-remove-keep",
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        res = await sub.run_task(task)

    winner = res.result["winners"][0]
    assert winner["remove_args"] == ["--bad-base"]
    assert "--bad-base" not in winner["extra_server_args"]
    assert "--keep-base 2" in winner["extra_server_args"]
    assert "--variant 4" in winner["extra_server_args"]
    assert res.result["best_variant"]["extra_server_args"] == winner["extra_server_args"]


@pytest.mark.asyncio
async def test_explore_executor_recovers_base_tput_from_shared_state(
    sub_agent_runner,
    tmp_path,
):
    """Regression: when params omits ``base_tput``, the executor recovers it from SharedState (else real wins are discarded)."""
    sub, tr, _ = sub_agent_runner
    state = SharedState()
    state.baseline_tput = 800.0
    sub.shared_state = state

    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    output_dir = tmp_path / "explore-base-tput-recovery"

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_workspace(slot, tput=840.0)  # +5% vs recovered baseline 800
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="ok",
            stderr="",
        )

    grid = [
        {
            "name": "v_keep",
            "extra_args": "--keep-flag",
            "extra_envs": {},
            "provenance": "default_grid",
        }
    ]
    task = await tr.create(
        kind="explore",
        params={
            "config_path": str(base),
            "output_dir": str(output_dir),
            # base_tput intentionally omitted to exercise SharedState recovery.
            "grid": grid,
        },
        idempotency_key="ex-base-tput-recovery",
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        res = await sub.run_task(task)

    out = res.result
    assert out["status"] == "succeeded"
    assert {w["name"] for w in out["winners"]} == {"v_keep"}
    fp = canonical_fingerprint("--keep-flag", {})
    tested = out["explore_search_update"]["tested"][fp]
    assert tested["outcome"] == "KEEP"
    assert tested["base_tput"] == 800.0


@pytest.mark.asyncio
async def test_per_variant_rows_carry_the_verdicts_and_the_stack(
    sub_agent_runner,
    tmp_path,
):
    """The round's own verdicts travel out with its numbers.

    Write-back records these on the framework timeline verbatim, because it
    cannot honestly rebuild them: an outcome of ``REVERT`` does not name the
    gate that ended the arc, and the stack a variant launched on has already
    moved on to whatever KEEP'd after it.
    """
    sub, tr, _ = sub_agent_runner
    state = SharedState()
    state.baseline_tput = 800.0
    sub.shared_state = state

    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    output_dir = tmp_path / "explore-verdict-carry"

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        _fake_workspace(Path(cmd[out_idx + 1]), tput=840.0)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    task = await tr.create(
        kind="explore",
        params={
            "config_path": str(base),
            "output_dir": str(output_dir),
            "base_tput": 800.0,
            "grid": [
                {
                    "name": "v_keep",
                    "extra_args": "--keep-flag",
                    "extra_envs": {},
                    "provenance": "llm_direct",
                }
            ],
        },
        idempotency_key="ex-verdict-carry",
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        res = await sub.run_task(task)

    row = next(v for v in res.result["per_variant_outcomes"] if v["variant_name"] == "v_keep")
    assert row["outcome"] == "KEEP"
    # The gain bar ruled, and says what it ruled against.
    keep_gate = next(g for g in row["gates"] if g["gate"] == "keep_threshold")
    assert keep_gate["passed"] is True
    assert keep_gate["observed"] == pytest.approx(5.0, abs=0.01)
    # This session has no baseline accuracy, so nothing gated the change and
    # the KEEP rests on throughput alone. The accuracy gate is absent rather
    # than reported as having passed.
    assert [g["gate"] for g in row["gates"]] == ["keep_threshold"]
    assert row["validation_basis"] == "keep_verdict_unscored"
    # The stack it launched on: the anchor plus a base config still empty,
    # since nothing has KEPT before it.
    assert row["measured_against"]["throughput"] == 800.0
    assert row["measured_against"]["extra_server_args"] == ""


@pytest.mark.asyncio
async def test_explore_executor_prefers_current_best_over_baseline_for_recovery(
    sub_agent_runner,
    tmp_path,
):
    """SharedState recovery prefers ``current_best.tput`` over ``baseline_tput``, so a +5%-vs-baseline variant REVERTs against the best."""
    sub, tr, _ = sub_agent_runner
    state = SharedState()
    state.baseline_tput = 800.0
    state.current_best = {"action": "explore", "tput": 900.0}
    sub.shared_state = state

    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    output_dir = tmp_path / "explore-cb-recovery"

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_workspace(slot, tput=840.0)  # +5% vs baseline, -6.7% vs best
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="ok",
            stderr="",
        )

    grid = [
        {
            "name": "v_below_best",
            "extra_args": "--below-best-flag",
            "extra_envs": {},
            "provenance": "default_grid",
        }
    ]
    task = await tr.create(
        kind="explore",
        params={
            "config_path": str(base),
            "output_dir": str(output_dir),
            "grid": grid,
        },
        idempotency_key="ex-cb-recovery",
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        res = await sub.run_task(task)

    out = res.result
    assert out["status"] == "succeeded"
    assert out["winners"] == []
    fp = canonical_fingerprint("--below-best-flag", {})
    tested = out["explore_search_update"]["tested"][fp]
    assert tested["base_tput"] == 900.0
    assert tested["outcome"] == "REVERT"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source", "expected_base_tput", "expected_outcome", "has_winner"),
    [
        (None, 2358.80, "REVERT", False),
        ("resume_stack_revalidate", 2192.52, "KEEP", True),
    ],
)
async def test_explore_executor_supersedes_stale_params_base_tput(
    sub_agent_runner,
    tmp_path,
    source,
    expected_base_tput,
    expected_outcome,
    has_winner,
):
    """Use the live anchor except when revalidating the complete stack."""
    sub, tr, _ = sub_agent_runner
    state = SharedState()
    state.baseline_tput = 2195.86
    state.current_best = {"action": "replay_warm_recipe", "tput": 2358.80}
    sub.shared_state = state

    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    output_dir = tmp_path / "explore-stale-anchor"

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_workspace(slot, tput=2355.46)
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="ok",
            stderr="",
        )

    grid = [
        {
            "name": "minimax-fused-swiglu+moe-combine",
            "extra_args": "--fused-flag",
            "extra_envs": {},
            "provenance": "specialist:research_scout",
        }
    ]
    task = await tr.create(
        kind="explore",
        params={
            "config_path": str(base),
            "output_dir": str(output_dir),
            # Snapshotted when the task was queued, before the warm replay landed.
            "base_tput": 2192.52,
            "grid": grid,
            **({"source": source} if source else {}),
        },
        idempotency_key="ex-stale-anchor",
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        res = await sub.run_task(task)

    out = res.result
    assert out["status"] == "succeeded"
    assert bool(out["winners"]) is has_winner
    fp = canonical_fingerprint("--fused-flag", {})
    tested = out["explore_search_update"]["tested"][fp]
    assert tested["base_tput"] == expected_base_tput
    assert tested["outcome"] == expected_outcome


@pytest.mark.asyncio
async def test_explore_executor_takes_live_base_args_with_the_live_anchor(
    sub_agent_runner,
    tmp_path,
):
    """Superseding a stale ``base_tput`` also re-reads the args it was measured on."""
    sub, tr, _ = sub_agent_runner
    state = SharedState()
    state.baseline_tput = 800.0
    state.current_best = {
        "action": "explore",
        "tput": 1000.0,
        "extra_server_args": "--live-layer 1",
        "extra_envs": {"LIVE_ENV": "1"},
    }
    sub.shared_state = state

    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        _fake_workspace(Path(cmd[out_idx + 1]), tput=1100.0)  # +10% vs the live 1000
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    task = await tr.create(
        kind="explore",
        params={
            "config_path": str(base),
            "output_dir": str(tmp_path / "explore-live-base"),
            # Snapshotted together at dispatch, before the newer layer landed.
            "base_tput": 800.0,
            "base_extra_args": "--stale-layer 1",
            "grid": [
                {
                    "name": "on_live_stack",
                    "extra_args": "--variant 2",
                    "extra_envs": {},
                    "provenance": "llm_direct",
                }
            ],
        },
        idempotency_key="ex-live-base-args",
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        res = await sub.run_task(task)

    out = res.result
    fp = canonical_fingerprint("--variant 2", {})
    assert out["explore_search_update"]["tested"][fp]["base_tput"] == 1000.0
    winner = out["winners"][0]
    assert "--live-layer 1" in winner["extra_server_args"]
    assert "--variant 2" in winner["extra_server_args"]
    assert "--stale-layer" not in winner["extra_server_args"]
    assert winner["extra_envs"]["LIVE_ENV"] == "1"


@pytest.mark.asyncio
async def test_explore_executor_historical_fingerprint_reruns(sub_agent_runner, tmp_path, monkeypatch):
    """A variant in explore_search.tested runs again; only same-grid exact duplicates are collapsed."""
    _force_cold_decision(monkeypatch)
    sub, tr, _ = sub_agent_runner
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    output_dir = tmp_path / "explore-dedup"

    bench_calls: list[str] = []

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        bench_calls.append(slot.name)
        _fake_workspace(slot, tput=900.0)
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="ok",
            stderr="",
        )

    fp_dup = canonical_fingerprint("--dup-flag", {})
    grid = [
        # Historical REVERT in ledger — must now execute (no cross-round block).
        {"name": "v_prior_revert", "extra_args": "--dup-flag", "extra_envs": {}, "provenance": "llm_direct"},
        # New variant — runs.
        {"name": "v_fresh", "extra_args": "--fresh-flag", "extra_envs": {}, "provenance": "llm_direct"},
        # Exact same-grid duplicate of v_prior_revert — collapsed to round_dup.
        {"name": "v_same_again", "extra_args": "--dup-flag", "extra_envs": {}, "provenance": "llm_direct"},
    ]
    task = await tr.create(
        kind="explore",
        params={
            "config_path": str(base),
            "output_dir": str(output_dir),
            "base_tput": 800.0,
            "grid": grid,
            "explore_search": {
                "tested": {
                    fp_dup: {
                        "fingerprint": fp_dup,
                        "name": "previous_run_name",
                        "extra_server_args": "--dup-flag",
                        "extra_envs": {},
                        "outcome": "REVERT",
                    }
                },
                "rejected": [],
                "accepted": [],
                "name_index": {},
            },
        },
        idempotency_key="ex-dedup",
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        res = await sub.run_task(task)

    out = res.result
    # v_same_again is a same-grid round_dup; v_prior_revert runs (no cross-round block).
    assert {d["name"] for d in out["skipped_dup"]} == {"v_same_again"}
    assert {d["reason"] for d in out["skipped_dup"]} == {"round_dup"}
    # Both historical and fresh variants execute (two bench calls, not one).
    assert len(bench_calls) == 2
    # Historical REVERT fingerprint was not blocked — it ran and can win.
    assert "v_prior_revert" in {w["name"] for w in out["winners"]}


@pytest.mark.asyncio
async def test_explore_executor_defaults_to_warm_decision_matching_hot_baseline(
    sub_agent_runner,
    tmp_path,
    monkeypatch,
):
    """Default EXPLORE measures hot decisions, matching default hot baseline."""
    sub, tr, _ = sub_agent_runner
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    output_dir = tmp_path / "explore-warm"

    bench_calls: list[str] = []

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        bench_calls.append(str(slot))
        _fake_workspace(slot, tput=920.0)  # +15% vs 800 — KEEP and stable
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    task = await tr.create(
        kind="explore",
        params={
            "config_path": str(base),
            "output_dir": str(output_dir),
            "base_tput": 800.0,
            "grid": [
                {
                    "name": "warm_keep",
                    "extra_args": "--warm-flag",
                    "extra_envs": {},
                    "provenance": "llm_direct",
                }
            ],
            "baseline_runtime_sec": 10.0,
            "baseline_warm_runtime_sec": 5.0,
        },
        idempotency_key="ex-warm",
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        res = await sub.run_task(task)

    out = res.result
    # warmup (discarded) + decision == 2 Magpie runs.
    assert len(bench_calls) == 2, bench_calls
    assert sum("warmup_round" in c for c in bench_calls) == 1
    assert {w["name"] for w in out["winners"]} == {"warm_keep"}


def _run_eval_of(cmd: list[str]) -> str:
    """Read RUN_EVAL out of the materialized YAML a Magpie call was handed."""
    cfg_idx = cmd.index("--benchmark-config")
    with Path(cmd[cfg_idx + 1]).open() as f:
        cfg = yaml.safe_load(f)
    return str(cfg["benchmark"]["envs"].get("RUN_EVAL", "")).strip().lower()


@pytest.mark.asyncio
async def test_explore_decision_round_skips_eval_warmup_keeps_it(
    sub_agent_runner,
    tmp_path,
    monkeypatch,
):
    """The overtime deadline is anchored on a throughput-only baseline, so the rounds it gates must measure throughput only."""
    sub, tr, _ = sub_agent_runner
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    output_dir = tmp_path / "explore-noeval"

    seen: list[tuple[str, str]] = []

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        seen.append((str(slot), _run_eval_of(cmd)))
        _fake_workspace(slot, tput=920.0)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    task = await tr.create(
        kind="explore",
        params={
            "config_path": str(base),
            "output_dir": str(output_dir),
            "base_tput": 800.0,
            "grid": [
                {
                    "name": "noeval_keep",
                    "extra_args": "--warm-flag",
                    "extra_envs": {},
                    "provenance": "llm_direct",
                }
            ],
            "baseline_runtime_sec": 10.0,
            "baseline_warm_runtime_sec": 5.0,
        },
        idempotency_key="ex-noeval",
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        await sub.run_task(task)

    warmup = [ev for slot, ev in seen if "warmup_round" in slot]
    decision = [ev for slot, ev in seen if "warmup_round" not in slot]
    assert warmup and warmup[0] not in _RUN_EVAL_FALSE
    assert decision and all(ev in _RUN_EVAL_FALSE for ev in decision)


@pytest.mark.asyncio
async def test_explore_no_eval_disables_magpie_warmup_and_decision(
    sub_agent_runner,
    tmp_path,
):
    """Session ``--no-eval`` turns Magpie RUN_EVAL off for every explore round."""
    sub, tr, _ = sub_agent_runner
    state = SharedState()
    state.eval_disabled = True
    sub.shared_state = state
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    output_dir = tmp_path / "explore-session-noeval"

    seen: list[tuple[str, str]] = []

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        seen.append((str(slot), _run_eval_of(cmd)))
        _fake_workspace(slot, tput=920.0)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    task = await tr.create(
        kind="explore",
        params={
            "config_path": str(base),
            "output_dir": str(output_dir),
            "base_tput": 800.0,
            "grid": [
                {
                    "name": "session_noeval",
                    "extra_args": "--warm-flag",
                    "extra_envs": {},
                    "provenance": "llm_direct",
                }
            ],
            "baseline_runtime_sec": 10.0,
            "baseline_warm_runtime_sec": 5.0,
        },
        idempotency_key="ex-session-noeval",
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        await sub.run_task(task)

    assert seen
    assert all(ev in _RUN_EVAL_FALSE for _slot, ev in seen)
    base_yaml = yaml.safe_load((output_dir / "explore_base.with_envs.yaml").read_text())
    assert str(base_yaml["benchmark"]["envs"].get("RUN_EVAL", "")).strip().lower() in _RUN_EVAL_FALSE


@pytest.mark.asyncio
async def test_explore_cold_decision_keeps_eval(
    sub_agent_runner,
    tmp_path,
    monkeypatch,
):
    """Without server_lifecycle reuse there is no warmup round whose eval the decision round could fall back on, so it must run its own accuracy gate."""
    _force_cold_decision(monkeypatch)
    sub, tr, _ = sub_agent_runner
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    output_dir = tmp_path / "explore-coldeval"

    seen: list[tuple[str, str]] = []

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        seen.append((str(slot), _run_eval_of(cmd)))
        _fake_workspace(slot, tput=920.0)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    task = await tr.create(
        kind="explore",
        params={
            "config_path": str(base),
            "output_dir": str(output_dir),
            "base_tput": 800.0,
            "grid": [
                {
                    "name": "cold_keep",
                    "extra_args": "--cold-flag",
                    "extra_envs": {},
                    "provenance": "llm_direct",
                }
            ],
            "baseline_runtime_sec": 10.0,
        },
        idempotency_key="ex-coldeval",
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        await sub.run_task(task)

    assert not [slot for slot, _ in seen if "warmup_round" in slot]
    decision = [ev for _slot, ev in seen]
    assert decision and all(ev not in _RUN_EVAL_FALSE for ev in decision)


@pytest.mark.asyncio
async def test_explore_decision_stays_cold_when_the_session_skips_the_double_run(
    sub_agent_runner,
    tmp_path,
):
    """A cold ``baseline_tput`` must be graded cold even when lifecycle reuse is available."""
    sub, tr, _ = sub_agent_runner
    state = SharedState()
    state.baseline_tput = 800.0
    state.baseline_double_run = False
    sub.shared_state = state

    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    seen: list[str] = []

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        seen.append(str(slot))
        _fake_workspace(slot, tput=920.0)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    task = await tr.create(
        kind="explore",
        params={
            "config_path": str(base),
            "output_dir": str(tmp_path / "explore-singleround"),
            "base_tput": 800.0,
            "grid": [{"name": "v", "extra_args": "--flag", "extra_envs": {}, "provenance": "llm_direct"}],
        },
        idempotency_key="ex-no-double-run",
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        await sub.run_task(task)

    assert not [slot for slot in seen if "warmup_round" in slot]


@pytest.mark.asyncio
async def test_explore_executor_warm_decision_warmup_failure_marks_failed(
    sub_agent_runner,
    tmp_path,
    monkeypatch,
):
    """A failed warmup round records the variant FAILED(reason=warmup_failed), no decision run."""
    sub, tr, _ = sub_agent_runner
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    output_dir = tmp_path / "explore-warmfail"

    calls: list[str] = []

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        calls.append(str(slot))
        # Warmup boot fails (nonzero) — no workspace written.
        return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="boom")

    task = await tr.create(
        kind="explore",
        params={
            "config_path": str(base),
            "output_dir": str(output_dir),
            "base_tput": 800.0,
            "grid": [
                {
                    "name": "warmfail",
                    "extra_args": "--warmfail-flag",
                    "extra_envs": {},
                    "provenance": "llm_direct",
                }
            ],
        },
        idempotency_key="ex-warmfail",
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        res = await sub.run_task(task)

    out = res.result
    assert len(calls) == 1, calls
    assert out["winners"] == []
    fp = canonical_fingerprint("--warmfail-flag", {})
    te = out["explore_search_update"]["tested"][fp]
    assert te["outcome"] == "FAILED"
    assert te["reason"] == "warmup_failed"
    assert te["stage"] == "warmup"
    assert "workspace" in te
    pvo = [v for v in out["per_variant_outcomes"] if v["outcome"] == "FAILED"]
    assert pvo, "expected FAILED entry in per_variant_outcomes"
    assert pvo[0]["stage"] == "warmup"
    assert "failure_id" in pvo[0]
    assert pvo[0]["failure_id"].startswith("fail.")


@pytest.mark.asyncio
async def test_explore_executor_overtime_disabled_when_ratio_zero(
    sub_agent_runner,
    tmp_path,
):
    """ratio<=0 disables the gate; executor must NOT pass ``soft_deadline_sec``."""
    sub, tr, _ = sub_agent_runner
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    output_dir = tmp_path / "explore-no-overtime"

    received_deadlines: list[float | None] = []

    def _fake_kill(cmd, *args, **kwargs):
        received_deadlines.append(kwargs.get("silence_timeout_sec"))
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_workspace(slot, tput=820.0)  # +2.5% KEEP
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="ok",
            stderr="",
        )

    task = await tr.create(
        kind="explore",
        params={
            "config_path": str(base),
            "output_dir": str(output_dir),
            "base_tput": 800.0,
            "grid": [
                {
                    "name": "fast_variant",
                    "extra_args": "--fast-flag",
                    "extra_envs": {},
                    "provenance": "default_grid",
                }
            ],
            "baseline_runtime_sec": 10.0,
        },
        idempotency_key="ex-overtime-off",
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_kill,
    ):
        res = await sub.run_task(task)

    assert received_deadlines, "no Magpie calls were made"
    assert all(d == 600 for d in received_deadlines)
    out = res.result
    assert out["status"] == "succeeded"


@pytest.mark.asyncio
async def test_explore_variant_cap_is_clamped_to_the_session_budget(
    sub_agent_runner,
    tmp_path,
    monkeypatch,
):
    """A granted cap never exceeds what is left of the session."""
    _force_cold_decision(monkeypatch)
    sub, tr, _ = sub_agent_runner
    state = SharedState()
    state.baseline_tput = 800.0
    state.max_minutes = 3.0
    sub.shared_state = state
    # Read before the run: the budget only shrinks from here, so a cap granted later can only be smaller than what
    # this allows.
    usable_sec = state.session_budget_usable_sec()

    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    granted: list[int] = []

    def _fake_run(cmd, *args, **kwargs):
        # Only benchmark rounds carry --output-dir; the interpreter probe does not, and it is module-memoized, so
        # counting it would make this order-dependent.
        if "--output-dir" not in cmd:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")
        granted.append(int(kwargs["timeout"]))
        slot = Path(cmd[cmd.index("--output-dir") + 1])
        _fake_workspace(slot, tput=840.0)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    task = await tr.create(
        kind="explore",
        params={
            "config_path": str(base),
            "output_dir": str(tmp_path / "explore-clamp"),
            "base_tput": 800.0,
            "grid": [
                {
                    "name": "v_fits",
                    "extra_args": "--max-num-seqs 256",
                    "extra_envs": {},
                    "provenance": "default_grid",
                }
            ],
            "baseline_runtime_sec": 20.0,
        },
        idempotency_key="ex-budget-clamp",
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        res = await sub.run_task(task)

    assert res.result["status"] == "succeeded"
    assert granted, f"the variant should have been admitted (20s expected, ~{usable_sec:.0f}s left)"
    assert all(t == 7800 for t in granted)


@pytest.mark.asyncio
async def test_explore_skips_a_variant_the_budget_cannot_fit(
    sub_agent_runner,
    tmp_path,
    monkeypatch,
):
    """Admission is judged on the expected runtime, and refused when it does not fit."""
    _force_cold_decision(monkeypatch)
    sub, tr, _ = sub_agent_runner
    state = SharedState()
    state.baseline_tput = 800.0
    state.max_minutes = 3.0  # ~60s usable
    sub.shared_state = state

    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    granted: list[int] = []

    def _fake_run(cmd, *args, **kwargs):
        if "--output-dir" not in cmd:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")
        granted.append(int(kwargs["timeout"]))
        slot = Path(cmd[cmd.index("--output-dir") + 1])
        _fake_workspace(slot, tput=840.0)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    task = await tr.create(
        kind="explore",
        params={
            "config_path": str(base),
            "output_dir": str(tmp_path / "explore-nofit"),
            "base_tput": 800.0,
            "grid": [
                {
                    "name": "v_too_long",
                    "extra_args": "--max-num-seqs 256",
                    "extra_envs": {},
                    "provenance": "default_grid",
                }
            ],
            "baseline_runtime_sec": 600.0,
        },
        idempotency_key="ex-budget-nofit",
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        res = await sub.run_task(task)

    assert granted == [], "a variant needing 600s must not start with ~60s left"
    # Measuring nothing because the budget ran out is not the same as variants failing, and it must not be reported as
    # a bare, unattributed failure.
    assert res.result["error_class"] == "session_time_exhausted"
    assert res.result["session_budget_untested"] == 1
    # Untested variants stay out of the ledger so a resume can retry them.
    assert res.result["losers"] == []
    assert res.result["explore_search_update"]["tested"] == {}


@pytest.mark.asyncio
async def test_explore_leaves_a_variant_the_run_reaped_out_of_the_ledger(
    sub_agent_runner,
    tmp_path,
    monkeypatch,
):
    """The common case: the budget expires while a variant is running, not before it."""
    _force_cold_decision(monkeypatch)
    sub, tr, _ = sub_agent_runner
    state = SharedState()
    state.baseline_tput = 800.0
    state.max_minutes = 600.0  # admits both variants; the reap comes mid-round
    state.baseline_runtime_sec = 20.0
    sub.shared_state = state

    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    ran: list[str] = []

    def _fake_run(cmd, *args, **kwargs):
        if "--output-dir" not in cmd:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")
        slot = Path(cmd[cmd.index("--output-dir") + 1])
        ran.append(slot.name)
        if "v_reaped" not in str(slot):
            _fake_workspace(slot, tput=840.0)
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")
        # The session deadline elapsed while this round was running, so the reaper tore the tree down and named the
        # cause.
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=SESSION_TIME_EXHAUSTED_RETURNCODE,
            stdout="",
            stderr="reaped",
        )

    task = await tr.create(
        kind="explore",
        params={
            "config_path": str(base),
            "output_dir": str(tmp_path / "explore-reaped"),
            "base_tput": 800.0,
            "grid": [
                {
                    "name": "v_measured",
                    "extra_args": "--max-num-seqs 256",
                    "extra_envs": {},
                    "provenance": "default_grid",
                },
                {
                    "name": "v_reaped",
                    "extra_args": "--max-num-seqs 512",
                    "extra_envs": {},
                    "provenance": "default_grid",
                },
            ],
            "baseline_runtime_sec": 20.0,
        },
        idempotency_key="ex-budget-reaped",
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        res = await sub.run_task(task)

    tested = res.result["explore_search_update"]["tested"]
    # A variant the run reaped was never measured; recording it keeps a resume from ever retrying it, and teaches the
    # KB a clock's verdict.
    assert [t["name"] for t in tested.values()] == ["v_measured"]
    assert [lr["name"] for lr in res.result["losers"]] == []
    assert res.result["session_budget_untested"] == 1


@pytest.mark.asyncio
async def test_explore_leaves_a_variant_out_when_the_run_reaped_its_grid_warmup(
    sub_agent_runner,
    tmp_path,
    monkeypatch,
):
    """The stop has to survive ``run_grid``'s own discarded warmup round."""
    _force_cold_decision(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RUN_GRID_WARMUP", "1")
    monkeypatch.setattr(
        "hyperloom.orchestrator.actions.executors._server_lifecycle.resolve_lifecycle_params",
        lambda _config_path: {
            "eligible": True,
            "framework": "sglang",
            "port": 30000,
            "reason": "",
        },
    )
    sub, tr, _ = sub_agent_runner
    state = SharedState()
    state.baseline_tput = 800.0
    state.max_minutes = 600.0
    state.baseline_runtime_sec = 20.0
    sub.shared_state = state

    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    ran: list[str] = []

    def _fake_run(cmd, *args, **kwargs):
        if "--output-dir" not in cmd:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")
        slot = Path(cmd[cmd.index("--output-dir") + 1])
        ran.append(slot.name)
        if slot.name == "warmup_round":
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=SESSION_TIME_EXHAUSTED_RETURNCODE,
                stdout="",
                stderr="reaped",
            )
        _fake_workspace(slot, tput=840.0)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    task = await tr.create(
        kind="explore",
        params={
            "config_path": str(base),
            "output_dir": str(tmp_path / "explore-warmup-reaped"),
            "base_tput": 800.0,
            "grid": [
                {
                    "name": "v_warmup_reaped",
                    "extra_args": "--max-num-seqs 256",
                    "extra_envs": {},
                    "provenance": "default_grid",
                }
            ],
            "baseline_runtime_sec": 20.0,
        },
        idempotency_key="ex-budget-warmup-reaped",
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        res = await sub.run_task(task)

    assert ran == ["warmup_round"], f"the decision round ran after the warmup was reaped: {ran}"
    assert res.result["explore_search_update"]["tested"] == {}
    assert res.result["losers"] == []
    assert res.result["error_class"] == SESSION_TIME_EXHAUSTED_CLASS
    assert res.result["session_budget_untested"] == 1


@pytest.mark.asyncio
async def test_explore_attributes_a_round_the_run_reaped_before_anything_measured(
    sub_agent_runner,
    tmp_path,
    monkeypatch,
):
    """With nothing measured the round is ``failed``, and must say who stopped it."""
    _force_cold_decision(monkeypatch)
    sub, tr, _ = sub_agent_runner
    state = SharedState()
    state.baseline_tput = 800.0
    state.max_minutes = 600.0
    state.baseline_runtime_sec = 20.0
    sub.shared_state = state

    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)

    def _fake_run(cmd, *args, **kwargs):
        if "--output-dir" not in cmd:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=ORCHESTRATOR_CANCELLED_RETURNCODE,
            stdout="",
            stderr="cancelled",
        )

    task = await tr.create(
        kind="explore",
        params={
            "config_path": str(base),
            "output_dir": str(tmp_path / "explore-cancelled"),
            "base_tput": 800.0,
            "grid": [
                {
                    "name": "v_cancelled",
                    "extra_args": "--max-num-seqs 256",
                    "extra_envs": {},
                    "provenance": "default_grid",
                }
            ],
            "baseline_runtime_sec": 20.0,
        },
        idempotency_key="ex-budget-cancelled",
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        res = await sub.run_task(task)

    assert res.result["status"] == "failed"
    assert res.result["error_class"] == ORCHESTRATOR_CANCELLED_CLASS
    assert res.result["explore_search_update"]["tested"] == {}


@pytest.mark.asyncio
async def test_explore_executor_empty_grid_returns_failed(sub_agent_runner, tmp_path):
    sub, tr, _ = sub_agent_runner
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    task = await tr.create(
        kind="explore",
        params={
            "config_path": str(base),
            "base_tput": 800.0,
            "grid": [],
        },
        idempotency_key="ex-empty",
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    res = await sub.run_task(task)
    assert res.result["status"] == "failed"
    assert res.result["error_class"] == "empty_grid"


def _names(variants):
    return [v.name for v in variants]


def test_atom_default_grid_mla_moe_model_emits_all_gated_variants():
    """MoE + MLA + MTP-capable model class (``moe_mla``) unlocks the full atom seed grid (>= 5 variants)."""
    grid = _atom_default_grid(
        model_class="moe_mla",
        conc=64,
        isl=1024,
        osl=1024,
    )
    names = _names(grid)
    assert len(grid) >= 5, f"too few variants: {names}"
    assert "atom_level_2" in names
    assert "atom_level_3" not in names
    assert "atom_prefix_cache" in names
    assert "atom_ep" in names, "MoE branch missing for moe_mla"
    assert "atom_dp_attn" in names, "MLA branch missing for moe_mla"
    assert "atom_mtp_3" in names, "MTP branch missing for moe_mla"
    assert "atom_mtp_1" in names
    assert "atom_cudagraph_bracket" in names


def test_atom_default_grid_dense_model_omits_moe_mla_mtp():
    """Dense model class must NOT emit MoE / MLA / MTP variants (would fail flag-compat or crash atom)."""
    grid = _atom_default_grid(
        model_class="dense",
        conc=8,
        isl=512,
        osl=512,
    )
    names = _names(grid)
    assert "atom_ep" not in names
    assert "atom_dp_attn" not in names
    assert "atom_mtp_3" not in names
    assert "atom_mtp_1" not in names
    assert "atom_level_2" in names
    assert "atom_level_3" not in names
    assert "atom_prefix_cache" in names


def test_atom_default_grid_fp8_model_emits_kv_fp8():
    grid = _atom_default_grid(
        model_class="moe_fp8",
        conc=16,
        isl=512,
        osl=512,
    )
    names = _names(grid)
    assert "atom_kv_fp8" in names
    assert "atom_ep" in names


def test_atom_default_grid_non_fp8_omits_kv_fp8():
    grid = _atom_default_grid(model_class="dense", conc=8)
    names = _names(grid)
    assert "atom_kv_fp8" not in names


def test_atom_default_grid_variants_have_unique_names():
    """Names must be unique within each model class's grid (the ledger keys by name per round)."""
    for mc in ("dense", "moe", "moe_mla", "moe_mla_nsa", "moe_fp8", ""):
        grid = _atom_default_grid(model_class=mc, conc=32)
        names = _names(grid)
        assert len(names) == len(set(names)), f"duplicate names in atom grid for model_class={mc!r}: {names}"


def test_atom_default_grid_names_use_atom_prefix():
    grid = _atom_default_grid(model_class="moe_mla", conc=64)
    for v in grid:
        assert v.name.startswith("atom_"), f"variant name does not start with 'atom_': {v.name!r}"


def test_atom_default_grid_variants_carry_default_grid_provenance():
    grid = _atom_default_grid(model_class="moe_mla", conc=64)
    for v in grid:
        assert getattr(v, "provenance", None) == "default_grid", (
            f"variant {v.name!r} provenance not set to default_grid"
        )


def test_atom_default_grid_conc_zero_omits_cudagraph_bracket():
    """When CONC is unavailable, the bracket variant is skipped."""
    grid = _atom_default_grid(model_class="dense", conc=0)
    assert "atom_cudagraph_bracket" not in _names(grid)


def test_default_grid_for_framework_atom_returns_seeded_grid():
    grid = _default_grid_for_framework(
        "atom",
        model_class="moe_mla",
        conc=64,
        isl=1024,
        osl=1024,
    )
    assert grid, "atom framework should produce a non-empty default grid"
    assert any(v.name == "atom_level_2" for v in grid)


@pytest.mark.parametrize("framework", ["sglang", "vllm", "", "unknown"])
def test_default_grid_for_framework_non_atom_returns_empty(framework):
    """Sglang/vllm rely on LLM-emitted variants; unknown frameworks also return ``[]`` (no crash)."""
    grid = _default_grid_for_framework(
        framework,
        model_class="moe_mla",
        conc=64,
    )
    assert grid == []


def _write_atom_baseline_yaml(path: Path) -> None:
    """Atom-flavoured base YAML for gap-G1 cold-start wiring tests."""
    cfg = {
        "benchmark": {
            "framework": "atom",
            "model": "/path/models/Qwen-Qwen3-32B",
            "precision": "fp8",
            "run_mode": "local",
            "envs": {"TP": 4, "CONC": 64, "ISL": 1024, "OSL": 1024},
            "benchmark_script": "atom_mi355x.sh",
            "timeout_seconds": 600,
            "profiler": {
                "torch_profiler": {"enabled": False},
                "system_profiler": {"enabled": False},
                "tracelens": {"enabled": False},
            },
            "gpu_selection": {"auto": False},
        },
    }
    with path.open("w") as f:
        yaml.safe_dump(cfg, f)


@pytest.mark.asyncio
async def test_explore_executor_atom_empty_grid_seeds_default_grid(
    sub_agent_runner,
    tmp_path,
    monkeypatch,
):
    """An empty grid on an atom session falls through to ``_default_grid_for_framework('atom', ...)`` rather than failing."""
    sub, tr, _ = sub_agent_runner
    base = tmp_path / "base_atom.yaml"
    _write_atom_baseline_yaml(base)

    # Sandbox MODEL_PATH so compatibility_filter doesn't auto-drop MoE/MLA variants.
    monkeypatch.setenv("MODEL_PATH", "/path/models/Qwen-Qwen3-32B")
    monkeypatch.setenv("FRAMEWORK", "atom")

    received_grid: list[list[str]] = []

    from hyperloom.orchestrator.actions.executors import (
        explore as explore_mod,
    )

    async def _capture_run_grid(**kwargs):
        received_grid.append([v.name for v in (kwargs.get("grid") or [])])
        return []

    monkeypatch.setattr(explore_mod, "run_grid", _capture_run_grid)

    task = await tr.create(
        kind="explore",
        params={
            "config_path": str(base),
            "base_tput": 800.0,
            "grid": [],
            "model_class": "moe_mla",
        },
        idempotency_key="ex-atom-empty-seeded",
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))

    await sub.run_task(task)

    assert received_grid, "run_grid was not invoked by the seed path"
    flat_names = [n for sub in received_grid for n in sub]
    assert all(n.startswith("atom_") for n in flat_names), (
        f"non-atom variants reached run_grid via the seed: {flat_names!r}"
    )
    assert "atom_level_2" in flat_names
    assert "atom_ep" in flat_names
    assert "atom_dp_attn" in flat_names


@pytest.mark.asyncio
async def test_explore_executor_sglang_empty_grid_still_fails_with_empty_grid(
    sub_agent_runner,
    tmp_path,
):
    """Inverse of the atom test: an empty grid on sglang/vllm still returns ``error_class='empty_grid'``."""
    sub, tr, _ = sub_agent_runner
    base = tmp_path / "base_sglang.yaml"
    _write_baseline_yaml(base)
    task = await tr.create(
        kind="explore",
        params={
            "config_path": str(base),
            "base_tput": 800.0,
            "grid": [],
        },
        idempotency_key="ex-sglang-empty-still-fails",
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    res = await sub.run_task(task)
    assert res.result["status"] == "failed"
    assert res.result["error_class"] == "empty_grid"


@pytest.mark.asyncio
async def test_explore_rejects_unsafe_aiter_unified_attn_before_benchmark(
    sub_agent_runner,
    tmp_path,
):
    sub, tr, _ = sub_agent_runner
    model = tmp_path / "Qwen3-14B-FP8"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3ForCausalLM"],
                "model_type": "qwen3",
                "torch_dtype": "bfloat16",
                "head_dim": 128,
                "num_attention_heads": 40,
                "num_key_value_heads": 8,
                "quantization_config": {"quant_method": "fp8"},
            }
        ),
        encoding="utf-8",
    )
    base = tmp_path / "base_sglang.yaml"
    _write_baseline_yaml(base)
    config = yaml.safe_load(base.read_text(encoding="utf-8"))
    config["benchmark"]["model"] = str(model)
    config["benchmark"]["precision"] = "fp8"
    config["benchmark"]["envs"]["EXTRA_SGLANG_ARGS"] = "--page-size 1 --kv-cache-dtype auto"
    base.write_text(yaml.safe_dump(config), encoding="utf-8")

    state = SharedState()
    state.model_path = str(model)
    state.model_name = "Qwen3-14B-FP8"
    state.model_type = "qwen3"
    state.gpu_type = "mi355x"
    state.baseline_double_run = False
    state.stack_fingerprint_meta = {
        "sglang": "0.5.20.dev20260920+gc610c40399",
        "aiter": "4ad99832823dde2315b361cbd3b54b1c5c12acd5",
        "rocm": "10.0.0",
    }
    sub.shared_state = state

    output_dir = tmp_path / "explore-unified-attn-filter"
    benchmarked: list[str] = []

    def _fake_run(cmd, *args, **kwargs):
        slot = Path(cmd[cmd.index("--output-dir") + 1])
        benchmarked.append(slot.relative_to(output_dir).as_posix())
        _fake_workspace(slot, tput=800.0)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    task = await tr.create(
        kind="explore",
        params={
            "config_path": str(base),
            "output_dir": str(output_dir),
            "base_tput": 800.0,
            "grid": [
                {
                    "name": "unified",
                    "extra_envs": {
                        "SGLANG_USE_AITER": "1",
                        "SGLANG_USE_AITER_UNIFIED_ATTN": "1",
                    },
                },
                {
                    "name": "unified_combo",
                    "extra_args": "--chunked-prefill-size 32768",
                    "extra_envs": {
                        "SGLANG_USE_AITER": "1",
                        "SGLANG_USE_AITER_UNIFIED_ATTN": "1",
                        "SGLANG_USE_AITER_FP8_PER_TOKEN": "1",
                    },
                },
                {"name": "master", "extra_envs": {"SGLANG_USE_AITER": "1"}},
                {
                    "name": "per_token",
                    "extra_envs": {
                        "SGLANG_USE_AITER": "1",
                        "SGLANG_USE_AITER_FP8_PER_TOKEN": "1",
                    },
                },
                {
                    "name": "unified_page16",
                    "extra_args": "--page-size 16",
                    "extra_envs": {
                        "SGLANG_USE_AITER": "1",
                        "SGLANG_USE_AITER_UNIFIED_ATTN": "1",
                    },
                },
            ],
        },
        idempotency_key="ex-unified-attn-filter",
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        result = await sub.run_task(task)

    assert benchmarked == [
        "v00_master/variant_00_master",
        "v01_per_token/variant_00_per_token",
        "v02_unified_page16/variant_00_unified_page16",
    ]
    assert result.result["skipped_dup"] == [
        {
            "name": "unified",
            "reason": "compatibility_filter",
            "detail": "SGLANG_USE_AITER_UNIFIED_ATTN=1 is unsafe on the exact ROCm 10 Qwen3-14B-FP8 stack",
        },
        {
            "name": "unified_combo",
            "reason": "compatibility_filter",
            "detail": "SGLANG_USE_AITER_UNIFIED_ATTN=1 is unsafe on the exact ROCm 10 Qwen3-14B-FP8 stack",
        },
    ]
    other_stack = dict(state.stack_fingerprint_meta)
    other_stack["aiter"] = "newer-aiter"
    other_variant = GridVariant(
        "other_stack_unified",
        extra_envs={"SGLANG_USE_AITER_UNIFIED_ATTN": "1"},
    )
    kept, dropped = apply_compatibility_filter(
        [other_variant],
        framework="sglang",
        model_path=str(model),
        gpu_type="mi355x",
        stack_fingerprint=other_stack,
        base_server_args="--page-size 1 --kv-cache-dtype auto",
    )
    assert ([variant.name for variant in kept], dropped) == (["other_stack_unified"], [])


def test_unified_attn_filter_matches_aiter_dist_version_short_sha(tmp_path):
    """The aiter fingerprint degrades to a git-describe dist version when ``AITER_COMMIT`` is unset; the same source
    tree must still be recognised, otherwise the filter fails open and the unsafe lever reaches the benchmark.
    """
    model = tmp_path / "Qwen3-14B-FP8"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3ForCausalLM"],
                "model_type": "qwen3",
                "torch_dtype": "bfloat16",
                "head_dim": 128,
                "num_attention_heads": 40,
                "num_key_value_heads": 8,
                "quantization_config": {"quant_method": "fp8"},
            }
        ),
        encoding="utf-8",
    )
    kept, dropped = apply_compatibility_filter(
        [GridVariant("unified", extra_envs={"SGLANG_USE_AITER_UNIFIED_ATTN": "1"})],
        framework="sglang",
        model_path=str(model),
        gpu_type="mi355x",
        stack_fingerprint={
            "sglang": "0.5.20.dev20260920+gc610c40399",
            # The dist version install_baremetal.sh falls back to; embeds the short sha of the pinned commit.
            "aiter": "0.1.21.dev48+g4ad998328.d20260920",
            "rocm": "10.0.0",
        },
        base_server_args="--page-size 1 --kv-cache-dtype auto",
    )
    assert ([variant.name for variant in kept], dropped) == (
        [],
        [
            {
                "name": "unified",
                "source": "compatibility_filter",
                "reason": "SGLANG_USE_AITER_UNIFIED_ATTN=1 is unsafe on the exact ROCm 10 Qwen3-14B-FP8 stack",
            }
        ],
    )


def test_atom_default_grid_survives_compatibility_filter_without_help_probe(
    monkeypatch,
):
    """When the atom help-text probe is unavailable, ``apply_compatibility_filter`` drops no seed variant."""
    # The filter resolves this name in its own module, so patching the re-export on _grid_runner would leave the real
    # ten-second probe running.
    from hyperloom.orchestrator.actions.executors import _grid_variant_filter

    monkeypatch.setattr(_grid_variant_filter, "_probe_server_help_text", lambda fw: "")

    grid = _atom_default_grid(model_class="moe_mla", conc=64)
    kept, dropped = apply_compatibility_filter(grid, framework="atom", model_path="")
    assert kept == grid, f"compatibility filter dropped seed variants when help-text probe is empty; dropped={dropped}"


def test_grid_variants_from_payload_coerces_list_extra_args():
    """A JSON-list ``extra_args`` must be space-joined into a shell-arg string, not stringified into a Python repr."""
    from hyperloom.orchestrator.actions.executors.explore import (
        _grid_variants_from_payload,
    )

    payload = [
        {"name": "list_args", "extra_args": ["--max-num-batched-tokens", "32768"]},
        {"name": "str_args", "extra_args": "--block-size 64"},
        {"name": "tuple_server_args", "extra_server_args": ("--distributed-executor-backend", "mp")},
    ]
    by_name = {v.name: v for v in _grid_variants_from_payload(payload)}

    assert by_name["list_args"].extra_server_args == "--max-num-batched-tokens 32768"
    assert "[" not in by_name["list_args"].extra_server_args
    assert by_name["str_args"].extra_server_args == "--block-size 64"
    assert by_name["tuple_server_args"].extra_server_args == "--distributed-executor-backend mp"


def test_grid_variants_from_payload_carries_removal_controls():
    from hyperloom.orchestrator.actions.executors.explore import (
        _grid_variants_from_payload,
    )

    payload = [
        {
            "name": "without_cache",
            "remove_args": "--enable-prefix-caching",
            "unset_envs": ["SGLANG_ENABLE_FOO"],
            "args_mode": "replace",
            "extra_args": "--max-num-seqs 256",
        }
    ]

    variant = _grid_variants_from_payload(payload)[0]
    assert variant.remove_args == ["--enable-prefix-caching"]
    assert variant.unset_envs == ["SGLANG_ENABLE_FOO"]
    assert variant.args_mode == "replace"
    assert variant.extra_server_args == "--max-num-seqs 256"


def test_on_disk_stderr_tail_reads_benchmark_stderr_log(tmp_path):
    from hyperloom.orchestrator.actions.executors._grid_runner import (
        _on_disk_stderr_tail,
        _report_errors_summary,
    )

    (tmp_path / "benchmark_stderr.log").write_text("bench_fps.py: error: unrecognized arguments: --use_cache teacache")
    tail = _on_disk_stderr_tail(tmp_path)
    assert "unrecognized arguments" in tail
    # Empty dir → empty string (caller keeps its original blank error).
    assert _on_disk_stderr_tail(tmp_path / "nope") == ""
    assert _report_errors_summary(None) == ""
    assert _report_errors_summary({"errors": []}) == ""
    assert (
        _report_errors_summary({"errors": ["scriptable benchmark script not found for custom_mi355x.sh"]})
        == "scriptable benchmark script not found for custom_mi355x.sh"
    )


@pytest.mark.asyncio
async def test_explore_executor_historical_failed_and_accepted_rerun(sub_agent_runner, tmp_path, monkeypatch):
    """FAILED and accepted historical fingerprints may rerun; rejected contains latest attempt."""
    _force_cold_decision(monkeypatch)
    sub, tr, _ = sub_agent_runner
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    output_dir = tmp_path / "explore-rerun"

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_workspace(slot, tput=900.0)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    fp_failed = canonical_fingerprint("--prev-failed", {})
    fp_accepted = canonical_fingerprint("--prev-kept", {})
    grid = [
        {"name": "rerun_failed", "extra_args": "--prev-failed", "extra_envs": {}, "provenance": "llm_direct"},
        {"name": "rerun_kept", "extra_args": "--prev-kept", "extra_envs": {}, "provenance": "llm_direct"},
    ]
    task = await tr.create(
        kind="explore",
        params={
            "config_path": str(base),
            "output_dir": str(output_dir),
            "base_tput": 800.0,
            "grid": grid,
            "explore_search": {
                "tested": {
                    fp_failed: {
                        "fingerprint": fp_failed,
                        "name": "rerun_failed",
                        "extra_server_args": "--prev-failed",
                        "extra_envs": {},
                        "outcome": "FAILED",
                        "gain_pct": None,
                    },
                },
                "rejected": [],
                "accepted": [
                    {
                        "fingerprint": fp_accepted,
                        "name": "rerun_kept",
                        "extra_server_args": "--prev-kept",
                        "extra_envs": {},
                        "outcome": "KEEP",
                        "gain_pct": 5.0,
                    }
                ],
                "name_index": {},
            },
        },
        idempotency_key="ex-rerun-all",
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        res = await sub.run_task(task)

    out = res.result
    # Neither historical FAILED nor accepted fingerprint is blocked.
    assert not any(d["reason"] == "ledger_dup" for d in out["skipped_dup"])
    # Both ran (first one wins; second measures same tput as updated base so it won't KEEP).
    tested = out["explore_search_update"]["tested"]
    assert fp_failed in tested
    # The latest result for fp_failed overwrites the FAILED entry.
    assert tested[fp_failed]["outcome"] in ("KEEP", "REVERT", "FAILED", "KILLED_OVERTIME")


# ───────────────────────────────────────────────────────────────────────────── Post-run orphan reap
# (AMD-AGI/Hyperloom#1354) ─────────────────────────────────────────────────────────────────────────────


def _missing_config_ctx(tmp_path: Path) -> SimpleNamespace:
    """A ctx that makes __call__ take its earliest ``return`` (missing_config), exercising the wrapper without needing
    to drive a full benchmark round.
    """
    return SimpleNamespace(
        task=SimpleNamespace(
            task_id="t-explore-reap",
            params={"config_path": str(tmp_path / "does_not_exist.yaml")},
        ),
        extra={},
    )


@pytest.mark.asyncio
async def test_explore_call_reaps_stale_servers_even_on_early_return(tmp_path, monkeypatch):
    """__call__ must reap any lingering server after _run_explore returns, even on its earliest failure path
    (missing_config) -- not just after a full benchmark round.
    """
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    executor = ExploreExecutor(session_dir=tmp_path)
    ctx = _missing_config_ctx(tmp_path)

    kill_calls = {"n": 0}

    def fake_kill():
        kill_calls["n"] += 1

    with patch(
        "hyperloom.orchestrator.actions.executors.explore._kill_stale_servers",
        side_effect=fake_kill,
    ):
        result = await executor(ctx)

    assert result["status"] == "failed"
    assert result["error_class"] == "missing_config"
    assert kill_calls["n"] == 1


@pytest.mark.asyncio
async def test_explore_call_skips_reap_under_pytest(tmp_path):
    """Direct guard: the reap must NOT fire while ``PYTEST_CURRENT_TEST`` is set (pytest always sets it for a running test), mirroring the guard on the per-launch preclean in ``_grid_runner.py``."""
    executor = ExploreExecutor(session_dir=tmp_path)
    ctx = _missing_config_ctx(tmp_path)

    kill_calls = {"n": 0}

    def fake_kill():
        kill_calls["n"] += 1

    with patch(
        "hyperloom.orchestrator.actions.executors.explore._kill_stale_servers",
        side_effect=fake_kill,
    ):
        result = await executor(ctx)

    assert result["status"] == "failed"
    assert kill_calls["n"] == 0, "must be a no-op while PYTEST_CURRENT_TEST is set"
