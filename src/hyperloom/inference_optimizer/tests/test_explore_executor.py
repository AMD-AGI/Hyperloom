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
from hyperloom.orchestrator.actions.executors.explore import (
    DEFAULT_EXPLORE_TIMEOUT_CEILING_SEC,
    DEFAULT_EXPLORE_TIMEOUT_FLOOR_SEC,
    _compute_explore_variant_timeout,
)
from hyperloom.orchestrator.actions.executors._canonical_fingerprint import (
    canonical_fingerprint,
)
from hyperloom.orchestrator.actions.executors._grid_runner import (
    _SESSION_KILL_GRACE_SEC,
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
from hyperloom.inference_optimizer.breakdown.collectors import (
    collect_capability_summary,
)


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


def _fake_workspace(slot: Path, *, tput: float = 800.0) -> Path:
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
                    "total_token_throughput": tput * 2,
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


def test_compute_explore_variant_timeout_floor_when_no_baseline():
    """No baseline yet (cold start / failed baseline) → floor."""
    assert _compute_explore_variant_timeout(0.0, 1.10) == DEFAULT_EXPLORE_TIMEOUT_FLOOR_SEC
    assert _compute_explore_variant_timeout(-1.0, 1.10) == DEFAULT_EXPLORE_TIMEOUT_FLOOR_SEC
    assert _compute_explore_variant_timeout(0.0, 1.10, floor_sec=1800) == 1800


def test_compute_explore_variant_timeout_scales_with_baseline():
    """Hard cap auto-scales above the soft kill (kill_ratio + safety_margin)."""
    derived = _compute_explore_variant_timeout(4140.0, 1.10)
    assert derived == 6624

    derived_small = _compute_explore_variant_timeout(300.0, 1.10)
    assert derived_small == DEFAULT_EXPLORE_TIMEOUT_FLOOR_SEC


def test_compute_explore_variant_timeout_ceiling_caps_runaway():
    """Pathological baseline value can't push the cap past the ceiling."""
    at_ceiling = _compute_explore_variant_timeout(9000.0, 1.10)
    assert at_ceiling == DEFAULT_EXPLORE_TIMEOUT_CEILING_SEC

    over = _compute_explore_variant_timeout(20000.0, 1.10)
    assert over == DEFAULT_EXPLORE_TIMEOUT_CEILING_SEC


def test_compute_explore_variant_timeout_kill_ratio_below_one_clamps():
    """A non-positive / sub-1 kill_ratio still gives a sensible cap (clamped to max(1.0, kill_ratio))."""
    derived = _compute_explore_variant_timeout(4140.0, 0.0)
    assert derived == int(4140.0 * 1.5)


def test_compute_explore_variant_timeout_safety_margin_override():
    """Operator can shrink/expand the safety margin (e.g. for torch.compile AOTI cold-start tax)."""
    generous = _compute_explore_variant_timeout(4140.0, 1.10, safety_margin=1.0)
    assert generous == 8694

    tight = _compute_explore_variant_timeout(4140.0, 1.10, safety_margin=0.0)
    assert tight == 4554


@pytest.mark.asyncio
async def test_explore_executor_auto_derives_variant_timeout(
    sub_agent_runner,
    tmp_path,
):
    """No ``variant_timeout_sec`` + injected baseline/kill_ratio → executor auto-derives the cap."""
    sub, tr, _ = sub_agent_runner
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    output_dir = tmp_path / "explore-derive"

    captured_timeouts: list[int] = []

    async def _spy_run_grid(*args, **kwargs):
        captured_timeouts.append(int(kwargs.get("variant_timeout_sec")))
        from hyperloom.orchestrator.actions.executors._grid_runner import (
            VariantResult,
        )

        slot = Path(kwargs["output_root"])
        slot.mkdir(parents=True, exist_ok=True)
        ws = _fake_workspace(slot, tput=805.0)
        return [
            VariantResult(
                name="v_smoke",
                extra_server_args="--smoke",
                extra_envs={},
                status="succeeded",
                output_throughput=805.0,
                workspace=str(ws),
            )
        ]

    grid = [
        {
            "name": "v_smoke",
            "extra_args": "--smoke",
            "extra_envs": {},
            "provenance": "default_grid",
        }
    ]
    task = await tr.create(
        kind="explore",
        params={
            "config_path": str(base),
            "output_dir": str(output_dir),
            "base_tput": 800.0,
            "grid": grid,
            "baseline_runtime_sec": 4140.0,
            "explore_overtime_kill_ratio": 1.10,
        },
        idempotency_key="ex-derive",
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors.explore.run_grid",
        side_effect=_spy_run_grid,
    ):
        res = await sub.run_task(task)

    assert res.state == "succeeded"
    assert captured_timeouts, "run_grid was not invoked"
    assert captured_timeouts[0] == 6624


@pytest.mark.asyncio
async def test_explore_executor_safety_margin_param_overrides_default(
    sub_agent_runner,
    tmp_path,
):
    """``params['variant_timeout_safety_margin']`` adjusts auto-derived headroom absent an explicit timeout."""
    sub, tr, _ = sub_agent_runner
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    output_dir = tmp_path / "explore-margin"

    captured_timeouts: list[int] = []

    async def _spy_run_grid(*args, **kwargs):
        captured_timeouts.append(int(kwargs.get("variant_timeout_sec")))
        from hyperloom.orchestrator.actions.executors._grid_runner import (
            VariantResult,
        )

        slot = Path(kwargs["output_root"])
        slot.mkdir(parents=True, exist_ok=True)
        ws = _fake_workspace(slot, tput=805.0)
        return [
            VariantResult(
                name="v_smoke",
                extra_server_args="--smoke",
                extra_envs={},
                status="succeeded",
                output_throughput=805.0,
                workspace=str(ws),
            )
        ]

    grid = [
        {
            "name": "v_smoke",
            "extra_args": "--smoke",
            "extra_envs": {},
            "provenance": "default_grid",
        }
    ]
    task = await tr.create(
        kind="explore",
        params={
            "config_path": str(base),
            "output_dir": str(output_dir),
            "base_tput": 800.0,
            "grid": grid,
            "baseline_runtime_sec": 4140.0,
            "explore_overtime_kill_ratio": 1.10,
            "variant_timeout_safety_margin": 1.0,
        },
        idempotency_key="ex-margin",
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors.explore.run_grid",
        side_effect=_spy_run_grid,
    ):
        res = await sub.run_task(task)

    assert res.state == "succeeded"
    assert captured_timeouts and captured_timeouts[0] == 8694


@pytest.mark.asyncio
async def test_explore_executor_explicit_variant_timeout_wins(
    sub_agent_runner,
    tmp_path,
):
    """Operator-pinned ``variant_timeout_sec`` takes precedence over auto-derive."""
    sub, tr, _ = sub_agent_runner
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    output_dir = tmp_path / "explore-pinned"

    captured_timeouts: list[int] = []

    async def _spy_run_grid(*args, **kwargs):
        captured_timeouts.append(int(kwargs.get("variant_timeout_sec")))
        from hyperloom.orchestrator.actions.executors._grid_runner import (
            VariantResult,
        )

        slot = Path(kwargs["output_root"])
        slot.mkdir(parents=True, exist_ok=True)
        ws = _fake_workspace(slot, tput=805.0)
        return [
            VariantResult(
                name="v_smoke",
                extra_server_args="--smoke",
                extra_envs={},
                status="succeeded",
                output_throughput=805.0,
                workspace=str(ws),
            )
        ]

    grid = [
        {
            "name": "v_smoke",
            "extra_args": "--smoke",
            "extra_envs": {},
            "provenance": "default_grid",
        }
    ]
    task = await tr.create(
        kind="explore",
        params={
            "config_path": str(base),
            "output_dir": str(output_dir),
            "base_tput": 800.0,
            "grid": grid,
            "variant_timeout_sec": 9000,
            "baseline_runtime_sec": 4140.0,
            "explore_overtime_kill_ratio": 1.10,
        },
        idempotency_key="ex-pin",
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors.explore.run_grid",
        side_effect=_spy_run_grid,
    ):
        res = await sub.run_task(task)

    assert res.state == "succeeded"
    assert captured_timeouts and captured_timeouts[0] == 9000


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
            "variant_timeout_sec": 10,
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
async def test_explore_serving_no_eval_reverts_without_stopping(sub_agent_runner, tmp_path):
    """A high-risk serving variant that clears throughput but yields no accuracy
    verdict used to skip the gate (throughput-only fallback). That fallback is
    removed: the variant REVERTs (the change likely broke the eval path), but
    the run does NOT stop -- only a broken baseline halts the run."""
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
            "variant_timeout_sec": 10,
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


@pytest.mark.asyncio
async def test_explore_gates_a_variant_no_flag_catalogue_would_have_caught(sub_agent_runner, tmp_path):
    """The gate no longer asks which knobs look risky.

    ``--online_quant_config`` changes numeric precision directly, and no entry of
    the deleted high-risk catalogue matched it, so a variant carrying it cleared
    on throughput alone with its measured accuracy discarded. With a baseline on
    the state it is now gated like any other variant, and no eval verdict is a
    REVERT rather than a KEEP.
    """
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
            "variant_timeout_sec": 10,
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
            "variant_timeout_sec": 10,
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
            "variant_timeout_sec": 10,
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
            "variant_timeout_sec": 10,
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
            "variant_timeout_sec": 10,
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
            "variant_timeout_sec": 10,
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
            "variant_timeout_sec": 10,
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
            "variant_timeout_sec": 10,
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
            "variant_timeout_sec": 30,
            "baseline_runtime_sec": 10.0,
            "baseline_warm_runtime_sec": 5.0,
            "explore_overtime_kill_ratio": 1.20,
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
    """The overtime deadline is anchored on a throughput-only baseline, so the
    rounds it gates must measure throughput only. The warmup round is ungated and
    remains the accuracy source."""
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
            "variant_timeout_sec": 30,
            "baseline_runtime_sec": 10.0,
            "baseline_warm_runtime_sec": 5.0,
            "explore_overtime_kill_ratio": 1.20,
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
async def test_explore_cold_decision_keeps_eval(
    sub_agent_runner,
    tmp_path,
    monkeypatch,
):
    """Without server_lifecycle reuse there is no warmup round whose eval the
    decision round could fall back on, so it must run its own accuracy gate."""
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
            "variant_timeout_sec": 30,
            "baseline_runtime_sec": 10.0,
            "explore_overtime_kill_ratio": 1.20,
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
    """A cold ``baseline_tput`` must be graded cold even when lifecycle reuse is available.

    The baseline gates its cold+hot double run on ``baseline_double_run`` as well
    as lifecycle eligibility, so warm-decision has to honour both or a hot
    candidate is scored against a cold anchor.
    """
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
            "variant_timeout_sec": 30,
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
            "variant_timeout_sec": 30,
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
async def test_explore_executor_killed_overtime_no_tput_no_keep(
    sub_agent_runner,
    tmp_path,
    monkeypatch,
):
    """A fired soft deadline records KILLED_OVERTIME (no tput, no KEEP/REVERT, stack unchanged)."""
    _force_cold_decision(monkeypatch)
    sub, tr, _ = sub_agent_runner
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    output_dir = tmp_path / "explore-overtime"

    from hyperloom.orchestrator.actions.executors._subprocess_kill import (
        OVERTIME_KILL_RETURNCODE,
    )

    def _fake_kill(cmd, *args, **kwargs):
        assert kwargs.get("soft_deadline_sec") == pytest.approx(11.0)
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=OVERTIME_KILL_RETURNCODE,
            stdout="",
            stderr="",
        )

    grid = [
        {
            "name": "slow_variant",
            "extra_args": "--slow-flag",
            "extra_envs": {},
            "provenance": "default_grid",
        }
    ]
    task = await tr.create(
        kind="explore",
        params={
            "config_path": str(base),
            "output_dir": str(output_dir),
            "base_tput": 800.0,
            "grid": grid,
            "variant_timeout_sec": 60,
            "baseline_runtime_sec": 10.0,
            "explore_overtime_kill_ratio": 1.10,
        },
        idempotency_key="ex-overtime",
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_kill,
    ):
        res = await sub.run_task(task)

    out = res.result
    assert out["status"] == "succeeded"
    assert out["winners"] == []
    assert len(out["losers"]) == 1
    loser = out["losers"][0]
    assert loser["name"] == "slow_variant"
    assert loser["reason"] == "killed_overtime"
    assert loser["tput"] is None
    assert loser["gain_pct"] is None
    assert loser["runtime_sec"] is not None
    assert loser["wall_clock_ratio_vs_baseline"] is not None
    fp = canonical_fingerprint("--slow-flag", {})
    ledger = out["explore_search_update"]
    te = ledger["tested"][fp]
    assert te["outcome"] == "KILLED_OVERTIME"
    assert te["tput"] is None
    assert te["gain_pct"] is None
    assert te["runtime_sec"] is not None
    assert te["wall_clock_ratio_vs_baseline"] is not None
    assert te["baseline_runtime_sec"] == pytest.approx(10.0)
    assert te["overtime_kill_ratio"] == pytest.approx(1.10)
    rejected_reasons = {r["reason"] for r in ledger["rejected"]}
    assert "killed_overtime" in rejected_reasons
    outcomes = {row["variant_name"]: row["outcome"] for row in out["per_variant_outcomes"]}
    assert outcomes["slow_variant"] == "KILLED_OVERTIME"
    assert fp in out["explore_search_update"]["last_round"]["killed_overtime"]
    # KILLED_OVERTIME rows must carry the decision stage and a distinct error_class.
    killed_pvo = [v for v in out["per_variant_outcomes"] if v["outcome"] == "KILLED_OVERTIME"]
    assert killed_pvo
    assert killed_pvo[0]["stage"] == "decision"
    assert "failure_id" in killed_pvo[0]
    assert killed_pvo[0]["failure_id"].startswith("fail.")
    assert te["stage"] == "decision"
    assert te["error_class"] == "killed_overtime"


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
        received_deadlines.append(kwargs.get("soft_deadline_sec"))
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
            "variant_timeout_sec": 60,
            "baseline_runtime_sec": 10.0,
            "explore_overtime_kill_ratio": 0.0,
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
    assert all(d is None for d in received_deadlines)
    out = res.result
    assert out["status"] == "succeeded"


@pytest.mark.asyncio
async def test_explore_variant_cap_is_clamped_to_the_session_budget(
    sub_agent_runner,
    tmp_path,
    monkeypatch,
):
    """A granted cap never exceeds what is left of the session.

    explore derives the cap from the measured baseline (up to 4h) and never
    consulted the budget, so a 3h session could hand a single variant more time
    than the whole run was given.
    """
    _force_cold_decision(monkeypatch)
    sub, tr, _ = sub_agent_runner
    state = SharedState()
    state.baseline_tput = 800.0
    state.max_minutes = 3.0
    sub.shared_state = state
    # Read before the run: the budget only shrinks from here, so a cap granted
    # later can only be smaller than what this allows.
    usable_sec = state.session_budget_usable_sec()

    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    granted: list[int] = []

    def _fake_run(cmd, *args, **kwargs):
        # Only benchmark rounds carry --output-dir; the interpreter probe does not,
        # and it is module-memoized, so counting it would make this order-dependent.
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
            "variant_timeout_sec": 3600,
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
    # The hard cap is allowed to sit a grace window past the deadline so the
    # in-process session watchdog reaps the tree first and the kill is attributed
    # to the budget rather than to a slow variant.
    assert all(t <= usable_sec + _SESSION_KILL_GRACE_SEC for t in granted), (
        f"caps must be clamped to the ~{usable_sec:.0f}s budget, got {granted}"
    )
    assert all(t < 3600 for t in granted), f"the declared 3600s cap must not survive the budget, got {granted}"


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
            "variant_timeout_sec": 3600,
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
    # Measuring nothing because the budget ran out is not the same as variants
    # failing, and it must not be reported as a bare, unattributed failure.
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
    """The common case: the budget expires while a variant is running, not before it.

    ``run_grid`` records such a variant as ``skipped`` because nothing was
    measured. Explore has to consume that distinction: a variant written into
    the KB-facing ``tested`` ledger as ``FAILED`` is one a resume will skip
    forever, and one the KB learns is a bad idea, on the evidence of a clock.
    """
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
        # The session deadline elapsed while this round was running, so the
        # reaper tore the tree down and named the cause.
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
            "variant_timeout_sec": 3600,
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
    # A variant the run reaped was never measured; recording it keeps a resume
    # from ever retrying it, and teaches the KB a clock's verdict.
    assert [t["name"] for t in tested.values()] == ["v_measured"]
    assert [lr["name"] for lr in res.result["losers"]] == []
    assert res.result["session_budget_untested"] == 1


@pytest.mark.asyncio
async def test_explore_leaves_a_variant_out_when_the_run_reaped_its_grid_warmup(
    sub_agent_runner,
    tmp_path,
    monkeypatch,
):
    """The stop has to survive ``run_grid``'s own discarded warmup round.

    With server-lifecycle reuse ineligible, explore's decision round is a plain
    ``run_grid`` call, and ``run_grid`` runs its own warmup pass before it (on by
    default outside pytest). Explore reads the stop off the result's
    ``error_class``, so a warmup reap graded as ``warmup_round_failed`` reaches
    this ledger as a measured verdict about the variant.
    """
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
            "variant_timeout_sec": 3600,
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
            "variant_timeout_sec": 3600,
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


def test_atom_default_grid_survives_compatibility_filter_without_help_probe(
    monkeypatch,
):
    """When the atom help-text probe is unavailable, ``apply_compatibility_filter`` drops no seed variant."""
    # The filter resolves this name in its own module, so patching the
    # re-export on _grid_runner would leave the real ten-second probe running.
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
            "variant_timeout_sec": 10,
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


# ─────────────────────────────────────────────────────────────────────────────
# Post-run orphan reap (AMD-AGI/Hyperloom#1354)
# ─────────────────────────────────────────────────────────────────────────────


def _missing_config_ctx(tmp_path: Path) -> SimpleNamespace:
    """A ctx that makes __call__ take its earliest ``return`` (missing_config),
    exercising the wrapper without needing to drive a full benchmark round."""
    return SimpleNamespace(
        task=SimpleNamespace(
            task_id="t-explore-reap",
            params={"config_path": str(tmp_path / "does_not_exist.yaml")},
        ),
        extra={},
    )


@pytest.mark.asyncio
async def test_explore_call_reaps_stale_servers_even_on_early_return(tmp_path, monkeypatch):
    """__call__ must reap any lingering server after _run_explore returns,
    even on its earliest failure path (missing_config) -- not just after a
    full benchmark round. A magpie_timeout that fires before a
    server_lifecycle variant's pidfile is ever written leaves nothing for
    that pidfile-based teardown to find, orphaning its server
    (AMD-AGI/Hyperloom#1354)."""
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
    """Direct guard: the reap must NOT fire while ``PYTEST_CURRENT_TEST`` is
    set (pytest always sets it for a running test), mirroring the guard on
    the per-launch preclean in ``_grid_runner.py``."""
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
