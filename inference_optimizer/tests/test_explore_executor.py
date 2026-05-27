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
    variant_fingerprint,
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
# SharedState — explore_search legacy migration
# ===========================================================================
def test_explore_search_migration_unions_legacy_ledgers():
    fp_attn = variant_fingerprint("--attention-backend aiter", {})
    fp_max_seqs = variant_fingerprint("--max-num-seqs 512", {})
    fp_triton = variant_fingerprint("--decode-attention-backend triton", {})
    raw = {
        "backends_search": {
            "schema_version": 1,
            "tested": {
                fp_attn: {
                    "name": "attn_aiter",
                    "extra_sglang_args": "--attention-backend aiter",
                    "extra_envs": {},
                    "status": "succeeded",
                    "tput": 1500.0, "gain_pct": 5.0,
                    "round_id": "backends-001",
                },
                fp_triton: {
                    "name": "decode_triton",
                    "extra_sglang_args": "--decode-attention-backend triton",
                    "extra_envs": {},
                    "status": "succeeded",
                    "tput": 750.0, "gain_pct": -25.0,
                    "round_id": "backends-001",
                },
            },
            "accepted": [{
                "name": "attn_aiter",
                "extra_sglang_args": "--attention-backend aiter",
                "extra_envs": {},
                "fingerprint": fp_attn,
                "gain_pct": 5.0,
            }],
            "rejected": [{
                "name": "decode_triton",
                "extra_sglang_args": "--decode-attention-backend triton",
                "extra_envs": {},
                "fingerprint": fp_triton,
                "reason": "not_keep", "gain_pct": -25.0,
            }],
            "name_index": {}, "cursor": 2,
        },
        "params_search": {
            "schema_version": 2,
            "tested": {
                fp_max_seqs: {
                    "name": "max_seqs_512",
                    "extra_sglang_args": "--max-num-seqs 512",
                    "extra_envs": {},
                    "gain_pct": 2.5,
                    "round_id": "params-001",
                },
            },
            "accepted": [{
                "name": "max_seqs_512",
                "extra_sglang_args": "--max-num-seqs 512",
                "extra_envs": {},
                "fingerprint": fp_max_seqs,
                "gain_pct": 2.5,
            }],
            "rejected": [], "name_index": {}, "cursor": 1,
        },
        "backend_winners_history": [{
            "round_id": "backends-001",
            "variant_name": "attn_aiter",
            "gain_pct": 5.0,
            "extra_sglang_args": "--attention-backend aiter",
            "extra_envs": {},
            "ts": "2026-05-01T00:00:30",
        }],
        "params_winner_history": [{
            "round_id": "params-001",
            "variant_name": "max_seqs_512",
            "gain_pct": 2.5, "tput": 1450.0,
            "ts": "2026-05-01T00:02:00",
        }],
        "synergy_attempted": ["attn_aiter+max_seqs_512"],
    }

    state = SharedState.from_dict(raw)
    es = state.explore_search

    # All three fingerprints made it across (no name collisions, no
    # accidental dedup).
    assert set(es["tested"].keys()) == {fp_attn, fp_triton, fp_max_seqs}
    # Accepted bucket is the union (attn_aiter + max_seqs_512).
    assert {a["name"] for a in es["accepted"]} == {"attn_aiter", "max_seqs_512"}
    # Rejected fingerprints don't appear in accepted (so a re-promote
    # wouldn't be double-counted).
    rejected_fps = {r["fingerprint"] for r in es["rejected"]}
    accepted_fps = {a["fingerprint"] for a in es["accepted"]}
    assert not (rejected_fps & accepted_fps)
    # winners_history merges both lists, in stable order.
    assert len(es["winners_history"]) == 2
    # Synergy combo string was normalized to a sorted list.
    assert es["synergy_attempted"] == [["attn_aiter", "max_seqs_512"]]
    # provenance is stamped on every row so the ledger origin is clear.
    for row in es["tested"].values():
        assert row["provenance"].startswith("legacy:")


def test_explore_search_migration_is_idempotent():
    """Re-loading the same raw state.json must not double-count entries."""
    fp_attn = variant_fingerprint("--attention-backend aiter", {})
    raw = {
        "backends_search": {
            "schema_version": 1,
            "tested": {fp_attn: {
                "name": "attn_aiter",
                "extra_sglang_args": "--attention-backend aiter",
                "extra_envs": {},
                "status": "succeeded",
                "tput": 1500.0, "gain_pct": 5.0,
            }},
            "accepted": [],
            "rejected": [],
            "name_index": {}, "cursor": 1,
        },
    }
    state_a = SharedState.from_dict(raw)
    # Round-trip through to_dict / from_dict to mimic resume.
    state_b = SharedState.from_dict(state_a.to_dict())
    assert state_a.explore_search["tested"] == state_b.explore_search["tested"]
    assert state_a.explore_search["winners_history"] == state_b.explore_search["winners_history"]


# ===========================================================================
# SharedState — record_explore_accepted / apply_explore_search_update
# ===========================================================================
def test_record_explore_accepted_dedup_by_fingerprint():
    state = SharedState()
    variant = {
        "name": "vllm_kv_fp8",
        "extra_sglang_args": "--kv-cache-dtype fp8",
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
        "extra_sglang_args": "--flag-a",
        "fingerprint": "aa" * 8,
        "gain_pct": 3.0,
    })
    # Executor update arrives with tested/rejected but NOT accepted —
    # the bucket should survive the merge.
    state.apply_explore_search_update({
        "schema_version": 1,
        "tested": {"bb" * 8: {
            "name": "b", "extra_sglang_args": "--flag-b",
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
            extra_sglang_args="--smoke",
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
            extra_sglang_args="--smoke",
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
            tput = 800.4   # +0.05% — below 0.2% threshold
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
                    "extra_sglang_args": "--dup-flag",
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


def test_default_stack_stable_pct_lowered_to_noise_band():
    """Fix B unit-pin: the module default ``DEFAULT_STACK_STABLE_PCT``
    must equal :data:`DEFAULT_KEEP_THRESHOLD_PCT` so the stack-rebench
    gate stays inside the ±1 % inter-run noise band (the 0.5 % floor
    used to silently downgrade real +0.3 % wins on MI300X)."""
    from inference_optimizer.orchestrator.action_executors.explore import (
        DEFAULT_KEEP_THRESHOLD_PCT,
        DEFAULT_STACK_STABLE_PCT,
    )
    # Pin: defaults must match — both gates speak the same noise floor.
    assert DEFAULT_STACK_STABLE_PCT == DEFAULT_KEEP_THRESHOLD_PCT == 0.2


def test_stack_stable_floor_arithmetic_at_new_default():
    """Fix B integration-pin: a rebench at +0.3 % vs baseline now sits
    above ``base * (1 + 0.2/100)`` and would NOT trigger KEEP_UNSTABLE
    eviction. Under the legacy 0.5 % default the same rebench was
    below the floor — this test guards against any future revert."""
    from inference_optimizer.orchestrator.action_executors.explore import (
        DEFAULT_STACK_STABLE_PCT,
    )
    base = 4438.83   # mirrors the real-run baseline_tput in the bug report
    rebench_tput = base * 1.003   # +0.3 %, just inside the noise band
    stable_floor = base * (1.0 + DEFAULT_STACK_STABLE_PCT / 100.0)
    assert rebench_tput > stable_floor, (
        f"DEFAULT_STACK_STABLE_PCT={DEFAULT_STACK_STABLE_PCT} would "
        f"still reject +0.3% rebenches; floor={stable_floor:.2f}, "
        f"rebench={rebench_tput:.2f}"
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
# breakdown.capability_summary — explore row + legacy aliases
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
    # Legacy aliases still emitted (KB_design §3.12 §4.2).
    assert "backends" in cap
    assert "params" in cap
    assert "validate_stack" in cap
