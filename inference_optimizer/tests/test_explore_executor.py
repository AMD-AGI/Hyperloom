"""v0.8 M3 — ExploreExecutor + explore_search ledger tests.

Mirror of ``test_p2_3_param_executors.py`` for the merged ``explore``
action (KB_design §3.4 §M3). Covers:

* canonical_fingerprint behaviour (rename-resistant dedup).
* SharedState legacy → explore_search migration.
* ExploreExecutor happy path (KEEP + REVERT + dedup + stack rebench).
* Stack-rebench eviction (KEEP_UNSTABLE).
* breakdown.capability_summary surfaces the new explore row alongside
  the legacy backends/params/validate_stack alias rows.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from inference_optimizer.orchestrator.action_executors import (
    ExploreExecutor,
)
from inference_optimizer.orchestrator.action_executors.explore import (
    DEFAULT_EXPLORE_TIMEOUT_CEILING_SEC,
    DEFAULT_EXPLORE_TIMEOUT_FLOOR_SEC,
    _compute_explore_variant_timeout,
)
from inference_optimizer.orchestrator.action_executors._canonical_fingerprint import (
    canonical_fingerprint,
)
from inference_optimizer.orchestrator.action_executors._grid_runner import (
    apply_compatibility_filter,
    variant_fingerprint,
)
from inference_optimizer.orchestrator.action_executors.explore import (
    _atom_default_grid,
    _default_grid_for_framework,
)
from inference_optimizer.orchestrator.shared_state import SharedState
from inference_optimizer.orchestrator.resource_lock import (
    ResourceLockManager, SqliteLeaseBackend,
)
from inference_optimizer.orchestrator.sub_agent_runner import SubAgentRunner
from inference_optimizer.orchestrator.task_registry import TaskRegistry
from inference_optimizer.storage import SqliteConnection
from inference_optimizer.breakdown.collectors import (
    collect_capability_summary,
)


@pytest.fixture(autouse=True)
def _isolate_leak_root(tmp_path_factory, monkeypatch):
    sandbox = tmp_path_factory.mktemp("isolated_leak_root_m3")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_LEAK_ROOTS", str(sandbox))


def _write_baseline_yaml(path: Path) -> None:
    cfg = {
        "benchmark": {
            "framework": "sglang",
            "model": "/wekafs/models/Qwen-Qwen3-8B",
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
    (workspace / "benchmark_report.json").write_text(json.dumps({
        "success": True,
        "framework": "sglang",
        "model": "/wekafs/models/Qwen-Qwen3-8B",
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
    }))
    return workspace


@pytest.fixture
def sub_agent_runner(tmp_path):
    db = SqliteConnection(tmp_path / "db.db")
    locks = ResourceLockManager(SqliteLeaseBackend(db))
    tr = TaskRegistry(db)
    runner = SubAgentRunner(locks, tr)
    yield runner, tr, tmp_path
    db.close()


# ===========================================================================
# canonical_fingerprint — content addressing, rename-resistant
# ===========================================================================
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
        "--max-num-seqs 256", {"VLLM_ROCM_USE_AITER": "1"},
    )
    assert fp_args != fp_args_envs


def test_canonical_fingerprint_matches_variant_fingerprint():
    """Identity with v0.6 ``variant_fingerprint`` (lossless migration)."""
    args = "--attention-backend aiter"
    envs = {"VLLM_ROCM_USE_AITER": "1"}
    assert canonical_fingerprint(args, envs) == variant_fingerprint(args, envs)


# ===========================================================================
# SharedState — record_explore_accepted / apply_explore_search_update
# ===========================================================================
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
    state.record_explore_accepted(variant)  # second promote with same content
    assert len(state.explore_search["accepted"]) == 1
    # winners_history grows on each call (history vs accepted bucket).
    assert len(state.explore_search["winners_history"]) == 2


def test_apply_explore_search_update_preserves_accepted():
    state = SharedState()
    state.record_explore_accepted({
        "name": "a",
        "extra_server_args": "--flag-a",
        "fingerprint": "aa" * 8,
        "gain_pct": 3.0,
    })
    # Executor update arrives with tested/rejected but NOT accepted —
    # the bucket should survive the merge.
    state.apply_explore_search_update({
        "schema_version": 1,
        "tested": {"bb" * 8: {
            "name": "b", "extra_server_args": "--flag-b",
            "extra_envs": {}, "outcome": "REVERT",
        }},
        "rejected": [{
            "fingerprint": "bb" * 8,
            "name": "b", "reason": "not_keep",
        }],
        "name_index": {"b": "bb" * 8},
        "cursor": 1,
        "last_round": {"round_id": "explore-001"},
    })
    assert len(state.explore_search["accepted"]) == 1
    assert state.explore_search["accepted"][0]["name"] == "a"
    assert "bb" * 8 in state.explore_search["tested"]


# ===========================================================================
# _compute_explore_variant_timeout — auto-derive helper
# ===========================================================================
def test_compute_explore_variant_timeout_floor_when_no_baseline():
    """No baseline yet (cold start / failed baseline) → floor."""
    assert _compute_explore_variant_timeout(0.0, 1.10) == DEFAULT_EXPLORE_TIMEOUT_FLOOR_SEC
    assert _compute_explore_variant_timeout(-1.0, 1.10) == DEFAULT_EXPLORE_TIMEOUT_FLOOR_SEC
    assert (
        _compute_explore_variant_timeout(0.0, 1.10, floor_sec=1800) == 1800
    )


def test_compute_explore_variant_timeout_scales_with_baseline():
    """Hard cap auto-scales above the soft kill (kill_ratio + safety_margin)."""
    # Qwen3-32B TP=1 BF16 example: 4140 s baseline × (1.10 + 0.5) = 6624 s.
    derived = _compute_explore_variant_timeout(4140.0, 1.10)
    assert derived == 6624

    # 7B TP=1 example: 300 s baseline × 1.6 = 480 s → floored to 2400.
    derived_small = _compute_explore_variant_timeout(300.0, 1.10)
    assert derived_small == DEFAULT_EXPLORE_TIMEOUT_FLOOR_SEC


def test_compute_explore_variant_timeout_ceiling_caps_runaway():
    """Pathological baseline value can't push the cap past the ceiling."""
    # Hypothetical 2.5 h baseline × 1.6 = 14400 s → exactly at ceiling.
    at_ceiling = _compute_explore_variant_timeout(9000.0, 1.10)
    assert at_ceiling == DEFAULT_EXPLORE_TIMEOUT_CEILING_SEC

    # Above ceiling clamps.
    over = _compute_explore_variant_timeout(20000.0, 1.10)
    assert over == DEFAULT_EXPLORE_TIMEOUT_CEILING_SEC


def test_compute_explore_variant_timeout_kill_ratio_below_one_clamps():
    """A non-positive / sub-1 kill_ratio still gives a sensible cap.

    The hard cap must always sit ABOVE the soft kill threshold; clamping
    kill_ratio to ``max(1.0, kill_ratio)`` preserves that invariant when
    callers disable the soft kill (kill_ratio=0).
    """
    # kill_ratio=0 (gate disabled) → effective_kill_ratio=1.0, derived = 1.5 × baseline.
    derived = _compute_explore_variant_timeout(4140.0, 0.0)
    assert derived == int(4140.0 * 1.5)


def test_compute_explore_variant_timeout_safety_margin_override():
    """Operator can shrink/expand the safety margin (e.g. for known
    cold-start tax such as torch.compile AOTI)."""
    # safety_margin=1.0 (very generous) → 4140 × 2.10 = 8694 s.
    generous = _compute_explore_variant_timeout(4140.0, 1.10, safety_margin=1.0)
    assert generous == 8694

    # safety_margin=0.0 (no headroom) → 4140 × 1.10 = 4554 s, equal to soft kill.
    tight = _compute_explore_variant_timeout(4140.0, 1.10, safety_margin=0.0)
    assert tight == 4554


# ===========================================================================
# ExploreExecutor wires the auto-derived timeout through run_grid
# ===========================================================================
@pytest.mark.asyncio
async def test_explore_executor_auto_derives_variant_timeout(
    sub_agent_runner, tmp_path,
):
    """No ``variant_timeout_sec`` in params + Coordinator-injected
    ``baseline_runtime_sec`` and ``explore_overtime_kill_ratio`` → executor
    auto-derives the hard cap and forwards it to run_grid."""
    sub, tr, _ = sub_agent_runner
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    output_dir = tmp_path / "explore-derive"

    captured_timeouts: list[int] = []

    async def _spy_run_grid(*args, **kwargs):
        captured_timeouts.append(int(kwargs.get("variant_timeout_sec")))
        # Return one fake successful result so the executor proceeds without
        # crashing on the empty-results path. Reuse the existing fake
        # workspace helper to populate the slot.
        from inference_optimizer.orchestrator.action_executors._grid_runner import (
            VariantResult,
        )
        slot = Path(kwargs["output_root"])
        slot.mkdir(parents=True, exist_ok=True)
        ws = _fake_workspace(slot, tput=805.0)
        return [VariantResult(
            name="v_smoke",
            extra_server_args="--smoke",
            extra_envs={},
            status="succeeded",
            output_throughput=805.0,
            workspace=str(ws),
        )]

    grid = [{
        "name": "v_smoke",
        "extra_args": "--smoke",
        "extra_envs": {},
        "provenance": "default_grid",
    }]
    task = await tr.create(
        kind="explore",
        params={
            "config_path": str(base),
            "output_dir":  str(output_dir),
            "base_tput":   800.0,
            "grid":        grid,
            # NO variant_timeout_sec → auto-derive should fire.
            "baseline_runtime_sec": 4140.0,
            "explore_overtime_kill_ratio": 1.10,
        },
        idempotency_key="ex-derive",
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "inference_optimizer.orchestrator.action_executors.explore.run_grid",
        side_effect=_spy_run_grid,
    ):
        res = await sub.run_task(task)

    assert res.state == "succeeded"
    # Auto-derived: 4140 × (1.10 + 0.5) = 6624.
    assert captured_timeouts, "run_grid was not invoked"
    assert captured_timeouts[0] == 6624


@pytest.mark.asyncio
async def test_explore_executor_safety_margin_param_overrides_default(
    sub_agent_runner, tmp_path,
):
    """``params['variant_timeout_safety_margin']`` widens (or shrinks) the
    auto-derived headroom when there is no explicit ``variant_timeout_sec``.
    """
    sub, tr, _ = sub_agent_runner
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    output_dir = tmp_path / "explore-margin"

    captured_timeouts: list[int] = []

    async def _spy_run_grid(*args, **kwargs):
        captured_timeouts.append(int(kwargs.get("variant_timeout_sec")))
        from inference_optimizer.orchestrator.action_executors._grid_runner import (
            VariantResult,
        )
        slot = Path(kwargs["output_root"])
        slot.mkdir(parents=True, exist_ok=True)
        ws = _fake_workspace(slot, tput=805.0)
        return [VariantResult(
            name="v_smoke",
            extra_server_args="--smoke",
            extra_envs={},
            status="succeeded",
            output_throughput=805.0,
            workspace=str(ws),
        )]

    grid = [{
        "name": "v_smoke",
        "extra_args": "--smoke",
        "extra_envs": {},
        "provenance": "default_grid",
    }]
    task = await tr.create(
        kind="explore",
        params={
            "config_path": str(base),
            "output_dir":  str(output_dir),
            "base_tput":   800.0,
            "grid":        grid,
            "baseline_runtime_sec": 4140.0,
            "explore_overtime_kill_ratio": 1.10,
            # Generous headroom (1.0) → 4140 * 2.10 = 8694 s.
            "variant_timeout_safety_margin": 1.0,
        },
        idempotency_key="ex-margin",
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "inference_optimizer.orchestrator.action_executors.explore.run_grid",
        side_effect=_spy_run_grid,
    ):
        res = await sub.run_task(task)

    assert res.state == "succeeded"
    assert captured_timeouts and captured_timeouts[0] == 8694


@pytest.mark.asyncio
async def test_explore_executor_roofline_hard_gate_drops_saturated_variants(
    sub_agent_runner, tmp_path,
):
    """``roofline_hard_gate=True`` + a saturation snapshot drops variants
    whose flags target only saturated directions; uncategorized / multi-
    direction variants survive."""
    sub, tr, _ = sub_agent_runner
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    output_dir = tmp_path / "explore-roofline-gate"

    benched_variants: list[str] = []

    async def _spy_run_grid(*args, **kwargs):
        from inference_optimizer.orchestrator.action_executors._grid_runner import (
            VariantResult,
        )
        grid = list(kwargs.get("grid") or [])
        # The executor calls run_grid once per variant (single-element grid).
        assert len(grid) == 1
        gv = grid[0]
        benched_variants.append(gv.name)
        slot = Path(kwargs["output_root"])
        slot.mkdir(parents=True, exist_ok=True)
        ws = _fake_workspace(slot, tput=805.0)
        return [VariantResult(
            name=gv.name,
            extra_server_args=gv.extra_server_args,
            extra_envs=dict(gv.extra_envs),
            status="succeeded",
            output_throughput=805.0,
            workspace=str(ws),
        )]

    grid = [
        {
            "name": "host_only",
            "extra_args": "--num-continuous-decode-steps 4",
            "extra_envs": {},
            "provenance": "default_grid",
        },
        {
            "name": "memory_only",
            "extra_args": "--max-running-requests 256",
            "extra_envs": {},
            "provenance": "default_grid",
        },
        {
            "name": "uncategorized",
            "extra_args": "--brand-new-flag",
            "extra_envs": {},
            "provenance": "default_grid",
        },
    ]
    task = await tr.create(
        kind="explore",
        params={
            "config_path": str(base),
            "output_dir":  str(output_dir),
            "base_tput":   800.0,
            "grid":        grid,
            # Roofline says memory is at 92 % (saturated), host_overhead
            # is at 18 % (plenty of headroom). The gate should drop
            # `memory_only` and keep `host_only` + `uncategorized`.
            "roofline_hard_gate": True,
            "roofline_saturation_snapshot": {
                "compute": 25.0, "memory": 92.0,
                "host_overhead": 18.0, "comm": 0.0,
            },
        },
        idempotency_key="ex-roofline-gate",
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "inference_optimizer.orchestrator.action_executors.explore.run_grid",
        side_effect=_spy_run_grid,
    ):
        res = await sub.run_task(task)

    assert res.state == "succeeded"
    out = res.result
    # The inlined stack_rebench path re-benches a KEEP'd variant under
    # ``<name>__stack_rebench`` — ignore that suffix when asserting which
    # base variants the gate let through.
    base_benched = {n.split("__")[0] for n in benched_variants}
    assert base_benched == {"host_only", "uncategorized"}
    skip_reasons = {
        s.get("name"): s.get("reason") for s in out.get("skipped_dup", [])
    }
    assert skip_reasons.get("memory_only") == "roofline_saturated"


@pytest.mark.asyncio
async def test_explore_executor_roofline_gate_disabled_by_default(
    sub_agent_runner, tmp_path,
):
    """Without ``roofline_hard_gate=True``, the soft advisory path is the
    only one in play and every variant runs (legacy behaviour)."""
    sub, tr, _ = sub_agent_runner
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    output_dir = tmp_path / "explore-roofline-default"

    benched_variants: list[str] = []

    async def _spy_run_grid(*args, **kwargs):
        from inference_optimizer.orchestrator.action_executors._grid_runner import (
            VariantResult,
        )
        gv = list(kwargs.get("grid") or [])[0]
        benched_variants.append(gv.name)
        slot = Path(kwargs["output_root"])
        slot.mkdir(parents=True, exist_ok=True)
        ws = _fake_workspace(slot, tput=805.0)
        return [VariantResult(
            name=gv.name,
            extra_server_args=gv.extra_server_args,
            extra_envs=dict(gv.extra_envs),
            status="succeeded",
            output_throughput=805.0,
            workspace=str(ws),
        )]

    grid = [{
        "name": "memory_only",
        "extra_args": "--max-running-requests 256",
        "extra_envs": {},
        "provenance": "default_grid",
    }]
    task = await tr.create(
        kind="explore",
        params={
            "config_path": str(base),
            "output_dir":  str(output_dir),
            "base_tput":   800.0,
            "grid":        grid,
            # snapshot present, but flag is OFF (or omitted).
            "roofline_saturation_snapshot": {
                "compute": 5.0, "memory": 99.0,
                "host_overhead": 5.0, "comm": 0.0,
            },
        },
        idempotency_key="ex-roofline-off",
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "inference_optimizer.orchestrator.action_executors.explore.run_grid",
        side_effect=_spy_run_grid,
    ):
        res = await sub.run_task(task)

    assert res.state == "succeeded"
    base_benched = {n.split("__")[0] for n in benched_variants}
    assert base_benched == {"memory_only"}


@pytest.mark.asyncio
async def test_explore_executor_explicit_variant_timeout_wins(
    sub_agent_runner, tmp_path,
):
    """Operator-pinned ``variant_timeout_sec`` in params takes precedence
    over the auto-derive even when baseline_runtime_sec is set."""
    sub, tr, _ = sub_agent_runner
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    output_dir = tmp_path / "explore-pinned"

    captured_timeouts: list[int] = []

    async def _spy_run_grid(*args, **kwargs):
        captured_timeouts.append(int(kwargs.get("variant_timeout_sec")))
        from inference_optimizer.orchestrator.action_executors._grid_runner import (
            VariantResult,
        )
        slot = Path(kwargs["output_root"])
        slot.mkdir(parents=True, exist_ok=True)
        ws = _fake_workspace(slot, tput=805.0)
        return [VariantResult(
            name="v_smoke",
            extra_server_args="--smoke",
            extra_envs={},
            status="succeeded",
            output_throughput=805.0,
            workspace=str(ws),
        )]

    grid = [{
        "name": "v_smoke",
        "extra_args": "--smoke",
        "extra_envs": {},
        "provenance": "default_grid",
    }]
    task = await tr.create(
        kind="explore",
        params={
            "config_path": str(base),
            "output_dir":  str(output_dir),
            "base_tput":   800.0,
            "grid":        grid,
            "variant_timeout_sec": 9000,    # explicit pin
            "baseline_runtime_sec": 4140.0,
            "explore_overtime_kill_ratio": 1.10,
        },
        idempotency_key="ex-pin",
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "inference_optimizer.orchestrator.action_executors.explore.run_grid",
        side_effect=_spy_run_grid,
    ):
        res = await sub.run_task(task)

    assert res.state == "succeeded"
    assert captured_timeouts and captured_timeouts[0] == 9000


# ===========================================================================
# ExploreExecutor — happy path (KEEP + REVERT + dedup)
# ===========================================================================
@pytest.mark.asyncio
async def test_explore_executor_keeps_and_reverts_per_variant(sub_agent_runner, tmp_path):
    sub, tr, _ = sub_agent_runner
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    output_dir = tmp_path / "explore-out"

    # Tput map: v_keep returns +5%, v_revert returns +0.05%, v_fail
    # returns +3% (above threshold, but accuracy-gated route).
    call_counter = {"i": 0}

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        # The stack-rebench path also runs through here; the slot's
        # parent dir name carries the variant slug so we can branch.
        slug = slot.parent.name + "/" + slot.name
        if "v00_v_keep" in slug:
            tput = 840.0   # +5% vs base 800
        elif "v01_v_revert" in slug:
            tput = 800.4   # +0.05% — below 1.0% threshold
        elif "stack_rebench" in slug:
            tput = 845.0   # stable for the KEEP'd variant
        else:
            tput = 800.0
        _fake_workspace(slot, tput=tput)
        call_counter["i"] += 1
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout="ok", stderr="",
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
            "output_dir":  str(output_dir),
            "base_tput":   800.0,
            "grid":        grid,
            "variant_timeout_sec": 10,
        },
        idempotency_key="ex-1",
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "inference_optimizer.orchestrator.action_executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        res = await sub.run_task(task)

    assert res.state == "succeeded"
    out = res.result
    assert out["status"] == "succeeded"
    assert {w["name"] for w in out["winners"]} == {"v_keep"}
    assert {lr["name"] for lr in out["losers"]} == {"v_revert"}
    assert out["keep_unstable_in_stack"] == []
    # explore_search_update populated with three buckets.
    ledger = out["explore_search_update"]
    assert set(ledger["tested"].keys()) == {
        canonical_fingerprint("--keep-flag", {}),
        canonical_fingerprint("--revert-flag", {}),
    }
    outcomes = {v["name"]: v["outcome"] for v in ledger["tested"].values()}
    assert outcomes["v_keep"] == "KEEP"
    assert outcomes["v_revert"] == "REVERT"
    # Best variant + best gain reflect the v_keep result (post-rebench).
    assert out["best_variant"]["name"] == "v_keep"
    assert out["best_gain_pct"] >= 4.0
    # Provenance from input is preserved on the ledger entries.
    rejected_provenance = {r["provenance"] for r in ledger["rejected"]}
    assert rejected_provenance == {"llm_direct"}


@pytest.mark.asyncio
async def test_explore_executor_dedups_against_ledger(sub_agent_runner, tmp_path):
    """A variant whose fingerprint already lives in explore_search.tested is
    not benchmarked again — it lands in ``skipped_dup`` instead."""
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
            args=cmd, returncode=0, stdout="ok", stderr="",
        )

    fp_dup = canonical_fingerprint("--dup-flag", {})
    grid = [
        # Already in ledger.tested -> SKIPPED
        {"name": "v_dup", "extra_args": "--dup-flag", "extra_envs": {}, "provenance": "llm_direct"},
        # New -> runs
        {"name": "v_fresh", "extra_args": "--fresh-flag", "extra_envs": {}, "provenance": "llm_direct"},
    ]
    task = await tr.create(
        kind="explore",
        params={
            "config_path": str(base),
            "output_dir":  str(output_dir),
            "base_tput":   800.0,
            "grid":        grid,
            "explore_search": {
                "tested": {fp_dup: {
                    "fingerprint": fp_dup,
                    "name": "previous_run_name",
                    "extra_server_args": "--dup-flag",
                    "extra_envs": {},
                    "outcome": "REVERT",
                }},
                "rejected": [],
                "accepted": [],
                "name_index": {},
            },
            "variant_timeout_sec": 10,
            # disable stack rebench so we can count the bench calls
            # against just the single-variant runs.
            "enable_stack_rebench": False,
        },
        idempotency_key="ex-dedup",
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "inference_optimizer.orchestrator.action_executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        res = await sub.run_task(task)

    out = res.result
    assert {d["name"] for d in out["skipped_dup"]} == {"v_dup"}
    # Only one Magpie subprocess fired — for v_fresh.
    assert len(bench_calls) == 1
    # v_fresh KEEP'd (900 vs 800 = +12.5%).
    assert {w["name"] for w in out["winners"]} == {"v_fresh"}


@pytest.mark.asyncio
async def test_explore_executor_stack_rebench_evicts_unstable_keep(
    sub_agent_runner, tmp_path,
):
    """KB_design §3.4 §4.4: if the inlined stack rebench can't beat
    ``base_tput * (1 + stack_stable_threshold_pct/100)``, the just-KEEP'd
    variant is evicted (KEEP_UNSTABLE) and demoted to REVERT.
    """
    sub, tr, _ = sub_agent_runner
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    output_dir = tmp_path / "explore-unstable"

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        slug = slot.parent.name + "/" + slot.name
        # Single-variant run looks great:
        if "v00_unstable" in slug and "stack_rebench" not in slug:
            tput = 850.0    # +6.25%
        elif "stack_rebench" in slug:
            tput = 802.0    # +0.25% — below the 0.5% stable floor
        else:
            tput = 800.0
        _fake_workspace(slot, tput=tput)
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout="ok", stderr="",
        )

    task = await tr.create(
        kind="explore",
        params={
            "config_path": str(base),
            "output_dir":  str(output_dir),
            "base_tput":   800.0,
            "grid": [{
                "name": "unstable",
                "extra_args": "--unstable-flag",
                "extra_envs": {},
                "provenance": "llm_direct",
            }],
            "variant_timeout_sec": 10,
            "stack_stable_threshold_pct": 0.5,
        },
        idempotency_key="ex-unstable",
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "inference_optimizer.orchestrator.action_executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        res = await sub.run_task(task)

    out = res.result
    assert {k["name"] for k in out["keep_unstable_in_stack"]} == {"unstable"}
    assert out["winners"] == []
    # Ledger reflects the eviction.
    ledger = out["explore_search_update"]
    fp = canonical_fingerprint("--unstable-flag", {})
    assert ledger["tested"][fp]["outcome"] == "KEEP_UNSTABLE"
    rejected_reasons = {r["reason"] for r in ledger["rejected"]}
    assert "stack_unstable" in rejected_reasons


def test_default_keep_and_stack_stable_thresholds():
    """Pin the per-variant KEEP threshold and the stack-rebench
    stability floor. The KEEP gate is 1.0 %; the rebench floor sits
    below it (0.5 %) so a genuine win that loses a little headroom when
    layered onto the cumulative stack is not immediately evicted."""
    from inference_optimizer.orchestrator.action_executors.explore import (
        DEFAULT_KEEP_THRESHOLD_PCT,
        DEFAULT_STACK_STABLE_PCT,
    )
    assert DEFAULT_KEEP_THRESHOLD_PCT == 1.0
    assert DEFAULT_STACK_STABLE_PCT == 0.5
    # The rebench floor must not exceed the single-variant KEEP gate.
    assert DEFAULT_STACK_STABLE_PCT <= DEFAULT_KEEP_THRESHOLD_PCT


def test_stack_stable_floor_arithmetic_at_new_default():
    """Integration-pin: a rebench at +0.6 % vs baseline sits above
    ``base * (1 + 0.5/100)`` and would NOT trigger KEEP_UNSTABLE
    eviction, while a +0.3 % rebench falls below the 0.5 % floor and is
    evicted. Guards the stability-floor arithmetic against drift."""
    from inference_optimizer.orchestrator.action_executors.explore import (
        DEFAULT_STACK_STABLE_PCT,
    )
    base = 4438.83
    stable_floor = base * (1.0 + DEFAULT_STACK_STABLE_PCT / 100.0)
    assert base * 1.006 > stable_floor, (
        f"DEFAULT_STACK_STABLE_PCT={DEFAULT_STACK_STABLE_PCT}: +0.6% "
        f"rebench should clear floor={stable_floor:.2f}"
    )
    assert base * 1.003 < stable_floor, (
        f"DEFAULT_STACK_STABLE_PCT={DEFAULT_STACK_STABLE_PCT}: +0.3% "
        f"rebench should fall below floor={stable_floor:.2f}"
    )


@pytest.mark.asyncio
async def test_explore_executor_killed_overtime_no_tput_no_keep(
    sub_agent_runner, tmp_path,
):
    """Fix E (Q3c): when the per-variant soft deadline fires, the
    variant is recorded with outcome=KILLED_OVERTIME, no tput, no
    KEEP/REVERT branch, and the running stack does NOT advance.
    """
    sub, tr, _ = sub_agent_runner
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    output_dir = tmp_path / "explore-overtime"

    # Patch run_with_session_kill (where the soft_deadline lives) to
    # return the OVERTIME_KILL_RETURNCODE sentinel directly. This
    # bypasses the actual sleep / poll loop so the test is fast and
    # deterministic. We patch it at the _grid_runner module symbol
    # because that's where _run_magpie imports it from.
    from inference_optimizer.orchestrator.action_executors._subprocess_kill import (
        OVERTIME_KILL_RETURNCODE,
    )

    def _fake_kill(cmd, *args, **kwargs):
        # Make sure the executor passed our deadline through.
        assert kwargs.get("soft_deadline_sec") == pytest.approx(11.0)
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=OVERTIME_KILL_RETURNCODE,
            stdout="",
            stderr="",
        )

    grid = [{
        "name": "slow_variant",
        "extra_args": "--slow-flag",
        "extra_envs": {},
        "provenance": "default_grid",
    }]
    task = await tr.create(
        kind="explore",
        params={
            "config_path": str(base),
            "output_dir":  str(output_dir),
            "base_tput":   800.0,
            "grid":        grid,
            "variant_timeout_sec": 60,
            "baseline_runtime_sec": 10.0,
            "explore_overtime_kill_ratio": 1.10,
        },
        idempotency_key="ex-overtime",
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "inference_optimizer.orchestrator.action_executors._grid_runner.run_with_session_kill",
        side_effect=_fake_kill,
    ):
        res = await sub.run_task(task)

    out = res.result
    # Status is succeeded — KILLED_OVERTIME counts as a useful signal
    # the LLM needs to see, not a hard failure of the explore task.
    assert out["status"] == "succeeded"
    # No winners; no KEEP_UNSTABLE; one loser tagged killed_overtime.
    assert out["winners"] == []
    assert out["keep_unstable_in_stack"] == []
    assert len(out["losers"]) == 1
    loser = out["losers"][0]
    assert loser["name"] == "slow_variant"
    assert loser["reason"] == "killed_overtime"
    assert loser["tput"] is None
    assert loser["gain_pct"] is None
    assert loser["runtime_sec"] is not None
    assert loser["wall_clock_ratio_vs_baseline"] is not None
    # Ledger row carries the diagnostic fields.
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
    # Rejected ledger picked the variant up so a re-proposal hits the
    # ledger.dedup path immediately.
    rejected_reasons = {r["reason"] for r in ledger["rejected"]}
    assert "killed_overtime" in rejected_reasons
    # per_variant_outcomes surfaces the KILLED_OVERTIME row alongside
    # KEEP / REVERT / FAILED / KEEP_UNSTABLE so Cortex KB sees it too.
    outcomes = {row["variant_name"]: row["outcome"]
                for row in out["per_variant_outcomes"]}
    assert outcomes["slow_variant"] == "KILLED_OVERTIME"
    # last_round summary surfaces the kill set so the prompt can see
    # how many variants were reaped this round at a glance.
    assert fp in out["explore_search_update"]["last_round"]["killed_overtime"]


@pytest.mark.asyncio
async def test_explore_executor_overtime_disabled_when_ratio_zero(
    sub_agent_runner, tmp_path,
):
    """Fix E (Q5): ratio=0 (or any non-positive) disables the gate; the
    legacy ``variant_timeout_sec`` hard cap is the only gate. The
    ExploreExecutor must NOT pass ``soft_deadline_sec`` in that case so
    ``run_with_session_kill``'s legacy timeout semantics stay intact.
    """
    sub, tr, _ = sub_agent_runner
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    output_dir = tmp_path / "explore-no-overtime"

    received_deadlines: list[float | None] = []

    def _fake_kill(cmd, *args, **kwargs):
        received_deadlines.append(kwargs.get("soft_deadline_sec"))
        # Simulate a successful run so we can verify the legacy path
        # still completes happily.
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_workspace(slot, tput=820.0)   # +2.5% KEEP
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout="ok", stderr="",
        )

    task = await tr.create(
        kind="explore",
        params={
            "config_path": str(base),
            "output_dir":  str(output_dir),
            "base_tput":   800.0,
            "grid": [{
                "name": "fast_variant",
                "extra_args": "--fast-flag",
                "extra_envs": {},
                "provenance": "default_grid",
            }],
            "variant_timeout_sec": 60,
            "baseline_runtime_sec": 10.0,
            "explore_overtime_kill_ratio": 0.0,   # disabled
        },
        idempotency_key="ex-overtime-off",
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "inference_optimizer.orchestrator.action_executors._grid_runner.run_with_session_kill",
        side_effect=_fake_kill,
    ):
        res = await sub.run_task(task)

    # Every Magpie call must have received ``soft_deadline_sec=None``.
    assert received_deadlines, "no Magpie calls were made"
    assert all(d is None for d in received_deadlines)
    out = res.result
    assert out["status"] == "succeeded"


@pytest.mark.asyncio
async def test_explore_executor_empty_grid_returns_failed(sub_agent_runner, tmp_path):
    sub, tr, _ = sub_agent_runner
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    task = await tr.create(
        kind="explore",
        params={
            "config_path": str(base),
            "base_tput":   800.0,
            "grid":        [],
        },
        idempotency_key="ex-empty",
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    res = await sub.run_task(task)
    # Even with empty grid, the executor returns a "failed" result
    # (NOT a runtime crash). KB_design §3.4 §7 boundary condition.
    assert res.result["status"] == "failed"
    assert res.result["error_class"] == "empty_grid"


# ===========================================================================
# breakdown.capability_summary — explore row + legacy compat rows
# ===========================================================================
def test_capability_summary_has_explore_row_with_legacy_aliases():
    state = {
        "explore_search": {
            "tested": {"aabbccdd11223344": {"name": "v1", "outcome": "KEEP"}},
            "accepted": [{"name": "v1", "gain_pct": 4.2,
                           "fingerprint": "aabbccdd11223344"}],
            "rejected": [{"name": "v2", "reason": "stack_unstable",
                           "fingerprint": "deadbeefdeadbeef"}],
            "winners_history": [{"round_id": "explore-001"}],
        },
        "optimization_stack": [
            {"action": "explore", "variant_name": "v1"},
        ],
    }
    cap = collect_capability_summary(state, [], [], [])
    # Primary row.
    assert "explore" in cap
    assert cap["explore"]["status"] == "kept"
    assert cap["explore"]["best_gain_pct"] == 4.2
    assert cap["explore"]["keep_unstable_count"] == 1
    assert cap["explore"]["winners_history"] == 1
    # Legacy compat rows stay emitted so archived (pre-merge) sessions
    # still render; on a current session they are ``not_attempted``.
    assert "backends" in cap
    assert "params" in cap
    assert "validate_stack" in cap
    assert cap["backends"]["status"] == "not_attempted"
    assert cap["params"]["status"] == "not_attempted"


# ===========================================================================
# _atom_default_grid + framework dispatch
# ===========================================================================
def _names(variants):
    return [v.name for v in variants]


def test_atom_default_grid_mla_moe_model_emits_all_gated_variants():
    """MoE + MLA + MTP-capable model class (e.g. ``moe_mla``) unlocks
    the full atom seed grid. Acceptance gate: >= 5 variants; each
    gated branch present.
    """
    grid = _atom_default_grid(
        model_class="moe_mla", conc=64, isl=1024, osl=1024,
    )
    names = _names(grid)
    assert len(grid) >= 5, f"too few variants: {names}"
    # Base coverage.
    assert "atom_level_2" in names
    # ``atom_level_3`` not emitted: atom_mi*x.sh injects ``--level 3``
    # as the bare baseline, so adding a level-3 variant here would A/B
    # against itself.
    assert "atom_level_3" not in names
    assert "atom_prefix_cache" in names
    # Model-class-gated branches.
    assert "atom_ep" in names, "MoE branch missing for moe_mla"
    assert "atom_dp_attn" in names, "MLA branch missing for moe_mla"
    assert "atom_mtp_3" in names, "MTP branch missing for moe_mla"
    assert "atom_mtp_1" in names
    # CONC bracket variant emitted because conc > 0.
    assert "atom_cudagraph_bracket" in names


def test_atom_default_grid_dense_model_omits_moe_mla_mtp():
    """Dense model class must NOT emit MoE / MLA / MTP variants —
    those flags would either fail flag-compat or crash atom on
    startup.
    """
    grid = _atom_default_grid(
        model_class="dense", conc=8, isl=512, osl=512,
    )
    names = _names(grid)
    assert "atom_ep" not in names
    assert "atom_dp_attn" not in names
    assert "atom_mtp_3" not in names
    assert "atom_mtp_1" not in names
    # Basic variants still present.
    assert "atom_level_2" in names
    # ``atom_level_3`` not emitted; see the rationale comment in the
    # default-grid test above.
    assert "atom_level_3" not in names
    assert "atom_prefix_cache" in names


def test_atom_default_grid_fp8_model_emits_kv_fp8():
    grid = _atom_default_grid(
        model_class="moe_fp8", conc=16, isl=512, osl=512,
    )
    names = _names(grid)
    assert "atom_kv_fp8" in names
    # FP8 + MoE gate the EP variant on too.
    assert "atom_ep" in names


def test_atom_default_grid_non_fp8_omits_kv_fp8():
    grid = _atom_default_grid(model_class="dense", conc=8)
    names = _names(grid)
    assert "atom_kv_fp8" not in names


def test_atom_default_grid_variants_have_unique_names():
    """Round-trip every supported model class through the grid and
    assert names are unique within each grid (the EXPLORE ledger keys
    by name within a single round).
    """
    for mc in ("dense", "moe", "moe_mla", "moe_mla_nsa", "moe_fp8", ""):
        grid = _atom_default_grid(model_class=mc, conc=32)
        names = _names(grid)
        assert len(names) == len(set(names)), (
            f"duplicate names in atom grid for model_class={mc!r}: {names}"
        )


def test_atom_default_grid_names_use_atom_prefix():
    grid = _atom_default_grid(model_class="moe_mla", conc=64)
    for v in grid:
        assert v.name.startswith("atom_"), (
            f"variant name does not start with 'atom_': {v.name!r}"
        )


def test_atom_default_grid_variants_carry_default_grid_provenance():
    grid = _atom_default_grid(model_class="moe_mla", conc=64)
    for v in grid:
        # ``provenance`` is stashed onto the dataclass instance.
        assert getattr(v, "provenance", None) == "default_grid", (
            f"variant {v.name!r} provenance not set to default_grid"
        )


def test_atom_default_grid_conc_zero_omits_cudagraph_bracket():
    """When the Coordinator can't supply CONC, the bracket variant is
    skipped (would otherwise emit an empty / nonsensical list).
    """
    grid = _atom_default_grid(model_class="dense", conc=0)
    assert "atom_cudagraph_bracket" not in _names(grid)


def test_default_grid_for_framework_atom_returns_seeded_grid():
    grid = _default_grid_for_framework(
        "atom", model_class="moe_mla", conc=64, isl=1024, osl=1024,
    )
    assert grid, "atom framework should produce a non-empty default grid"
    assert any(v.name == "atom_level_2" for v in grid)


@pytest.mark.parametrize("framework", ["sglang", "vllm", "", "unknown"])
def test_default_grid_for_framework_non_atom_returns_empty(framework):
    """Sglang / vllm rely on LLM-emitted variants (no programmatic
    seed exists today). Unknown frameworks must also return ``[]``
    rather than crash.
    """
    grid = _default_grid_for_framework(
        framework, model_class="moe_mla", conc=64,
    )
    assert grid == []


def _write_atom_baseline_yaml(path: Path) -> None:
    """Atom-flavoured base YAML for gap-G1 cold-start wiring tests."""
    cfg = {
        "benchmark": {
            "framework": "atom",
            "model": "/wekafs/models/Qwen-Qwen3-32B",
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
    sub_agent_runner, tmp_path, monkeypatch,
):
    """When the orchestration LLM emits an empty grid on an atom
    session, ExploreExecutor must NOT return
    ``error_class='empty_grid'`` — it must fall through to
    ``_default_grid_for_framework('atom', ...)`` and run those
    variants.

    Bench is mocked so the test asserts *only* the wiring (empty grid
    → seed grid loaded, executor proceeds to grid runner), not the
    seed's per-variant outcome which is gated by the Magpie / atom
    server.
    """
    sub, tr, _ = sub_agent_runner
    base = tmp_path / "base_atom.yaml"
    _write_atom_baseline_yaml(base)

    # Sandbox MODEL_PATH so compatibility_filter doesn't auto-drop
    # MoE/MLA variants.
    monkeypatch.setenv("MODEL_PATH", "/wekafs/models/Qwen-Qwen3-32B")
    monkeypatch.setenv("FRAMEWORK", "atom")

    received_grid: list[list[str]] = []

    # Monkeypatch the ``run_grid`` symbol bound INSIDE the explore
    # module (the executor imports it at module-load time, so
    # patching ``_grid_runner.run_grid`` would miss the call site).
    from inference_optimizer.orchestrator.action_executors import (
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
            "base_tput":   800.0,
            "grid":        [],
            "model_class": "moe_mla",
        },
        idempotency_key="ex-atom-empty-seeded",
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))

    await sub.run_task(task)

    # ``run_grid`` is called once per variant (variants stream one at
    # a time through the executor). Each invocation should carry
    # exactly one atom_ seed variant.
    assert received_grid, "run_grid was not invoked by the seed path"
    flat_names = [n for sub in received_grid for n in sub]
    assert all(n.startswith("atom_") for n in flat_names), (
        f"non-atom variants reached run_grid via the seed: {flat_names!r}"
    )
    # The full seed for moe_mla + CONC>0 has at least 5 variants; the
    # executor may have stopped earlier (every per-variant call
    # returned []), so we assert lower-bound coverage.
    assert "atom_level_2" in flat_names
    assert "atom_ep" in flat_names
    assert "atom_dp_attn" in flat_names


@pytest.mark.asyncio
async def test_explore_executor_sglang_empty_grid_still_fails_with_empty_grid(
    sub_agent_runner, tmp_path,
):
    """Inverse of the atom test: sglang / vllm have NO programmatic
    seed today (intentional — see ``_default_grid_for_framework``);
    an empty grid on those frameworks must continue to return
    ``error_class='empty_grid'`` so the orchestration LLM is forced
    to emit variants. Regression guard against accidentally seeding
    sglang from the atom branch.
    """
    sub, tr, _ = sub_agent_runner
    base = tmp_path / "base_sglang.yaml"
    _write_baseline_yaml(base)
    task = await tr.create(
        kind="explore",
        params={
            "config_path": str(base),
            "base_tput":   800.0,
            "grid":        [],
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
    """When the atom help-text probe is unavailable (e.g. test sandbox
    with no atom installed), ``apply_compatibility_filter`` MUST NOT
    drop any atom seed variant — it treats absent help text as
    "defer to graceful runtime failure".

    Guarantees the seed grid is never silently emptied by the
    compatibility filter on test boxes.
    """
    monkeypatch.delenv("MODEL_PATH", raising=False)
    monkeypatch.delenv("FRAMEWORK", raising=False)
    # Force the help-text probe to return empty (simulates atom not
    # importable). The filter must not drop any of the seed variants.
    from inference_optimizer.orchestrator.action_executors import (
        _grid_runner,
    )
    monkeypatch.setattr(
        _grid_runner, "_probe_server_help_text", lambda fw: "",
    )

    grid = _atom_default_grid(model_class="moe_mla", conc=64)
    kept, dropped = apply_compatibility_filter(grid)
    assert kept == grid, (
        f"compatibility filter dropped seed variants when help-text "
        f"probe is empty; dropped={dropped}"
    )
