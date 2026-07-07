# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the FRAMEWORK config-exploration capability (Route B).

FRAMEWORK replicates EXPLORE's config search by, within its phase:
1. dispatching a config-generation specialist (LLM proposes a variant grid),
2. harvesting its ``proposal_set`` into a grid,
3. benchmarking it via the ExploreExecutor (KEEP/REVERT/rebench/ledger),
4. iterating over rounds until no new candidates or the round cap.

Coverage:
* ``_build_framework_config_grid`` / ``_framework_config_explore_params`` /
  ``_run_framework_config_exploration`` — grid + explore-task plumbing.
* ``_framework_config_new_variants`` — tested-ledger de-dup.
* ``_framework_config_grid_from_proposals`` / ``_ingest_framework_config_generation``
  — proposal_set -> grid harvest.
* ``_maybe_hold_for_framework_config_lane`` — the default-OFF subphase state
  machine ('' -> generating -> running -> done).

Methods are invoked unbound on a light fake ``self`` (mirroring
``test_mn_auto_materialize``); real helpers are bound with ``types.MethodType``.
"""

from __future__ import annotations

import asyncio
import types
from pathlib import Path
from types import SimpleNamespace

from inference_optimizer.orchestrator.action_executors._canonical_fingerprint import (
    canonical_fingerprint,
)
from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.orchestrator.shared_state import SharedState


class _FakeTasks:
    def __init__(self):
        self.calls = []

    async def create_or_return_existing(self, *, kind, params, idempotency_key, **kwargs):
        self.calls.append(
            {"kind": kind, "params": params, "idempotency_key": idempotency_key}
        )
        return SimpleNamespace(task_id="framework-config-explore-1"), False


def _async_return(value):
    async def _fn(*a, **k):
        return value

    return _fn


def _fake_self(**state_overrides):
    state = SimpleNamespace(
        framework="sglang",
        model_class="",
        baseline_config_path="/cfg.yaml",
        current_best={"extra_server_args": "--base-arg 1"},
        baseline_tput=123.0,
        last_baseline={"benchmark_script": "bench.sh"},
        framework_config_exploration_enabled=False,
        framework_config_exploration_results=[],
        framework_config_lane_state="",
        framework_config_lane_round=0,
        framework_config_pending_grid=[],
        explore_search={},
        save=lambda *a, **k: None,
    )
    for k, v in state_overrides.items():
        setattr(state, k, v)
    s = SimpleNamespace(
        _FRAMEWORK_CONFIG_GRID_CAP=Coordinator._FRAMEWORK_CONFIG_GRID_CAP,
        _FRAMEWORK_CONFIG_MAX_ROUNDS=Coordinator._FRAMEWORK_CONFIG_MAX_ROUNDS,
        shared_state=state,
        tasks=_FakeTasks(),
        session_dir=Path("/tmp"),
        _inject_explore_runtime_params=lambda params: None,
        _cycle_idem_suffix=lambda: "",
        # Safe async defaults; individual tests override to exercise branches.
        _framework_config_generation_inflight=_async_return(False),
        _framework_config_exploration_inflight=_async_return(False),
        _dispatch_framework_config_generation_specialist=_async_return("gen-default"),
        _run_framework_config_exploration=_async_return("run-default"),
    )
    for name in (
        "_build_framework_config_grid",
        "_framework_config_explore_params",
        "_framework_config_new_variants",
        "_finish_framework_config_lane",
        "_framework_config_max_rounds",
        "_framework_config_grid_from_proposals",
        "_ingest_framework_config_generation",
        "_start_framework_config_generation",
    ):
        setattr(s, name, types.MethodType(getattr(Coordinator, name), s))
    return s


def _build(self_obj, **kwargs):
    return Coordinator._build_framework_config_grid(self_obj, **kwargs)


def _hold(self_obj):
    return asyncio.run(Coordinator._maybe_hold_for_framework_config_lane(self_obj))


# --------------------------------------------------------------------------
# SharedState defaults
# --------------------------------------------------------------------------
def test_shared_state_defaults_off():
    ss = SharedState()
    assert ss.framework_config_exploration_enabled is False
    assert ss.framework_config_lane_state == ""
    assert ss.framework_config_lane_round == 0
    assert ss.framework_config_pending_grid == []


def test_shared_state_flag_roundtrips_through_serialization():
    ss = SharedState()
    ss.framework_config_exploration_enabled = True
    restored = SharedState.from_dict(ss.to_dict())
    assert restored.framework_config_exploration_enabled is True


# --------------------------------------------------------------------------
# _build_framework_config_grid
# --------------------------------------------------------------------------
def test_build_grid_explicit_stamps_provenance_dedups_and_drops_empty():
    s = _fake_self()
    grid = _build(
        s,
        explicit_grid=[
            {"name": "arg-variant", "extra_args": "--enable-foo", "note": "r1"},
            {"name": "env-variant", "extra_envs": {"MORI_DISPATCH": "2"}},
            {"name": "arg-variant", "extra_args": "--dup"},  # duplicate name -> dropped
            {"name": "empty", "note": "no args/envs"},  # dropped
            "not-a-dict",  # skipped
        ],
    )
    assert [g["name"] for g in grid] == ["arg-variant", "env-variant"]
    assert all(g["provenance"] == "framework_agent:config" for g in grid)


def test_build_grid_capped_at_grid_cap():
    s = _fake_self()
    explicit = [{"name": f"v{i}", "extra_args": f"--flag {i}"} for i in range(20)]
    assert len(_build(s, explicit_grid=explicit)) == Coordinator._FRAMEWORK_CONFIG_GRID_CAP


def test_build_grid_sglang_without_explicit_is_empty():
    assert _build(_fake_self(framework="sglang")) == []


# --------------------------------------------------------------------------
# _framework_config_explore_params
# --------------------------------------------------------------------------
def test_explore_params_marks_source_and_threads_baseline():
    s = _fake_self()
    grid = [{"name": "v", "extra_args": "--x", "extra_envs": {}, "provenance": "framework_agent:config"}]
    params = Coordinator._framework_config_explore_params(s, grid, reason="unit")
    assert params["source"] == "framework_config_exploration"
    assert params["grid"] == grid
    assert params["base_extra_args"] == "--base-arg 1"
    assert params["base_tput"] == 123.0
    assert params["benchmark_script"] == "bench.sh"


# --------------------------------------------------------------------------
# _run_framework_config_exploration
# --------------------------------------------------------------------------
def test_run_enqueues_explore_task_with_marker():
    s = _fake_self()
    task_id = asyncio.run(
        Coordinator._run_framework_config_exploration(
            s, explicit_grid=[{"name": "cfg-a", "extra_args": "--enable-foo"}], reason="r1"
        )
    )
    assert task_id == "framework-config-explore-1"
    call = s.tasks.calls[0]
    assert call["kind"] == "explore"
    assert call["params"]["source"] == "framework_config_exploration"
    assert all(g["provenance"] == "framework_agent:config" for g in call["params"]["grid"])


def test_run_skips_when_grid_empty():
    s = _fake_self(framework="sglang")
    assert asyncio.run(Coordinator._run_framework_config_exploration(s, explicit_grid=[])) == ""
    assert s.tasks.calls == []


# --------------------------------------------------------------------------
# _framework_config_new_variants
# --------------------------------------------------------------------------
def test_new_variants_filters_already_tested():
    fp = canonical_fingerprint("--x", {})
    s = _fake_self(explore_search={"tested": {fp: {"outcome": "REVERT"}}})
    grid = [
        {"name": "tested", "extra_args": "--x", "extra_envs": {}},
        {"name": "fresh", "extra_args": "--y", "extra_envs": {}},
    ]
    assert [g["name"] for g in s._framework_config_new_variants(grid)] == ["fresh"]


# --------------------------------------------------------------------------
# proposal_set -> grid harvest
# --------------------------------------------------------------------------
def test_grid_from_proposals_keeps_applicable_and_stamps_provenance():
    s = _fake_self()
    out = s._framework_config_grid_from_proposals(
        [
            {"name": "a", "extra_args": "--foo", "reason": "r"},
            {"extra_envs": {"E": "1"}},  # unnamed -> fallback name
            {"name": "empty", "reason": "no args/envs"},  # dropped
            "nope",  # skipped
        ]
    )
    assert [g["name"] for g in out] == ["a", "framework-config-1"]
    assert all(g["provenance"] == "framework_agent:config" for g in out)


def test_ingest_harvests_proposal_set_into_pending_grid():
    s = _fake_self()
    task = SimpleNamespace(task_id="gen-1", params={"framework_config_generation": True})
    s._ingest_framework_config_generation(
        task=task, done_payload={"proposal_set": [{"name": "a", "extra_args": "--foo"}]}
    )
    assert [g["name"] for g in s.shared_state.framework_config_pending_grid] == ["a"]


def test_ingest_noop_without_marker():
    s = _fake_self(framework_config_pending_grid=[{"name": "keep"}])
    task = SimpleNamespace(task_id="x", params={})  # no marker
    s._ingest_framework_config_generation(
        task=task, done_payload={"proposal_set": [{"name": "a", "extra_args": "--foo"}]}
    )
    assert s.shared_state.framework_config_pending_grid == [{"name": "keep"}]


# --------------------------------------------------------------------------
# _maybe_hold_for_framework_config_lane (subphase state machine)
# --------------------------------------------------------------------------
def test_hold_disabled_is_strict_noop():
    s = _fake_self(framework_config_exploration_enabled=False)
    s._dispatch_framework_config_generation_specialist = _async_return("must-not-run")
    assert _hold(s) is False
    assert s.shared_state.framework_config_lane_state == ""


def test_hold_start_dispatches_generation():
    s = _fake_self(framework_config_exploration_enabled=True)
    s._dispatch_framework_config_generation_specialist = _async_return("gen-1")
    assert _hold(s) is True
    assert s.shared_state.framework_config_lane_state == "generating"


def test_hold_generating_holds_while_specialist_inflight():
    s = _fake_self(
        framework_config_exploration_enabled=True,
        framework_config_lane_state="generating",
    )
    s._framework_config_generation_inflight = _async_return(True)

    async def _no_run(**kwargs):
        raise AssertionError("must not run explore while generation is in flight")

    s._run_framework_config_exploration = _no_run
    assert _hold(s) is True


def test_hold_generating_runs_explore_from_pending():
    s = _fake_self(
        framework_config_exploration_enabled=True,
        framework_config_lane_state="generating",
        framework_config_pending_grid=[{"name": "v", "extra_args": "--x", "extra_envs": {}}],
    )
    s._framework_config_generation_inflight = _async_return(False)
    s._run_framework_config_exploration = _async_return("run-1")
    assert _hold(s) is True
    assert s.shared_state.framework_config_lane_state == "running"
    assert s.shared_state.framework_config_lane_round == 1
    assert s.shared_state.framework_config_pending_grid == []


def test_hold_generating_finishes_when_no_new_candidates():
    fp = canonical_fingerprint("--x", {})
    s = _fake_self(
        framework_config_exploration_enabled=True,
        framework_config_lane_state="generating",
        framework_config_pending_grid=[{"name": "v", "extra_args": "--x", "extra_envs": {}}],
        explore_search={"tested": {fp: {"outcome": "REVERT"}}},
    )
    s._framework_config_generation_inflight = _async_return(False)
    assert _hold(s) is False
    assert s.shared_state.framework_config_lane_state == "done"


def test_hold_running_holds_while_explore_inflight():
    s = _fake_self(
        framework_config_exploration_enabled=True,
        framework_config_lane_state="running",
        framework_config_lane_round=1,
    )
    s._framework_config_exploration_inflight = _async_return(True)
    assert _hold(s) is True


def test_hold_running_starts_next_round():
    s = _fake_self(
        framework_config_exploration_enabled=True,
        framework_config_lane_state="running",
        framework_config_lane_round=1,
    )
    s._framework_config_exploration_inflight = _async_return(False)
    s._dispatch_framework_config_generation_specialist = _async_return("gen-2")
    assert _hold(s) is True
    assert s.shared_state.framework_config_lane_state == "generating"


def test_hold_running_finishes_at_max_rounds():
    s = _fake_self(
        framework_config_exploration_enabled=True,
        framework_config_lane_state="running",
        framework_config_lane_round=Coordinator._FRAMEWORK_CONFIG_MAX_ROUNDS,
    )
    s._framework_config_exploration_inflight = _async_return(False)
    assert _hold(s) is False
    assert s.shared_state.framework_config_lane_state == "done"


def test_hold_noop_when_lane_done():
    s = _fake_self(
        framework_config_exploration_enabled=True,
        framework_config_lane_state="done",
    )
    assert _hold(s) is False


def test_start_generation_falls_back_to_default_grid_when_dispatch_fails():
    s = _fake_self(framework_config_exploration_enabled=True)
    s._dispatch_framework_config_generation_specialist = _async_return("")  # generation unavailable
    s._build_framework_config_grid = lambda **k: [
        {"name": "seed", "extra_args": "--seed", "extra_envs": {}, "provenance": "framework_agent:config"}
    ]
    s._run_framework_config_exploration = _async_return("run-fallback")
    held = asyncio.run(Coordinator._start_framework_config_generation(s, round_no=0))
    assert held is True
    assert s.shared_state.framework_config_lane_state == "running"
    assert s.shared_state.framework_config_lane_round == 1


def test_run_idempotency_key_is_round_unique():
    # Regression: a shared key would collapse rounds 2..N onto round 1's task
    # (create_or_return_existing dedups by idempotency_key regardless of state).
    s1 = _fake_self()
    asyncio.run(
        Coordinator._run_framework_config_exploration(
            s1, explicit_grid=[{"name": "v", "extra_args": "--x"}], round_no=1
        )
    )
    s2 = _fake_self()
    asyncio.run(
        Coordinator._run_framework_config_exploration(
            s2, explicit_grid=[{"name": "v", "extra_args": "--x"}], round_no=2
        )
    )
    k1 = s1.tasks.calls[0]["idempotency_key"]
    k2 = s2.tasks.calls[0]["idempotency_key"]
    assert k1 != k2
    assert "round1" in k1 and "round2" in k2


# --------------------------------------------------------------------------
# Wiring: advance-hook trigger predicate + mn-explore collision guard
# --------------------------------------------------------------------------
def test_lane_should_engage_only_when_enabled_and_leaving_framework():
    # disabled -> never engage (regardless of phase / next_phase)
    s = _fake_self(framework_config_exploration_enabled=False, phase="FRAMEWORK_AGENT")
    assert Coordinator._framework_config_lane_should_engage(s, ("EXPLORE", "r", {})) is False

    # enabled + in FRAMEWORK + leaving FRAMEWORK -> engage
    s = _fake_self(framework_config_exploration_enabled=True, phase="FRAMEWORK_AGENT")
    assert Coordinator._framework_config_lane_should_engage(s, ("EXPLORE", "r", {})) is True

    # enabled but next_phase None -> no engage
    assert Coordinator._framework_config_lane_should_engage(s, None) is False

    # enabled but staying in FRAMEWORK -> no engage
    assert Coordinator._framework_config_lane_should_engage(s, ("FRAMEWORK_AGENT", "r", {})) is False

    # enabled but not currently in FRAMEWORK -> no engage
    s2 = _fake_self(framework_config_exploration_enabled=True, phase="EXPLORE")
    assert Coordinator._framework_config_lane_should_engage(s2, ("KERNEL_AGENT", "r", {})) is False


def test_mn_explore_skips_framework_config_generation_specialist(monkeypatch):
    # The mn-explore bridge must skip config-generation specialists so their
    # proposal_set is not double-consumed (owned by the config subphase).
    from inference_optimizer.orchestrator.action_executors import _multi_node_env as mne

    monkeypatch.setattr(mne, "is_multi_node", lambda: True)
    s = SimpleNamespace(
        _MN_AUTO_EXPLORE_GRID_CAP=Coordinator._MN_AUTO_EXPLORE_GRID_CAP,
        shared_state=SimpleNamespace(
            baseline_config_path="", current_best={}, baseline_tput=0.0, last_baseline={}
        ),
        tasks=_FakeTasks(),
    )
    task = SimpleNamespace(task_id="gen-1", params={"framework_config_generation": True})
    asyncio.run(
        Coordinator._maybe_materialize_mn_explore(
            s, task=task, domain="serving_specialist", proposals=[{"name": "v", "extra_args": "--x"}]
        )
    )
    assert s.tasks.calls == []  # guard skipped the generation specialist
