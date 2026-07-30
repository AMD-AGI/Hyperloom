# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the FRAMEWORK_AGENT local-exploration arm.

Covers:
- the phase-budget re-split (FRAMEWORK 20 / EXPLORE 35 / KERNEL 35, sum == 1.0);
- the selectable phase-audit LLM policy (off / on / auto) + uncertainty gate;
- the candidate-free local-exploration pseudo-candidate + id sequencing;
- the discovery-exhaustion pivot to a local-exploration specialist (instead of
  flipping ``framework_agent_phase_done``);
- the resident arm offering the pseudo-candidate to the ranker.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hyperloom.orchestrator.framework import client as _fa_client
from hyperloom.orchestrator.phases import machine_state as _phase_state
from hyperloom.orchestrator.phases.framework import FrameworkPhase


# --------------------------------------------------------------------------- #
# 1. Budget re-split
# --------------------------------------------------------------------------- #
def test_phase_budget_resplit_values_and_sum():
    """FRAMEWORK 20 / EXPLORE 35 / KERNEL 35 / SWEEP 5; the split still sums to 1.0."""
    b = _phase_state.DEFAULT_PHASE_BUDGET_PCT
    assert b[_phase_state.PHASE_FRAMEWORK_AGENT] == pytest.approx(0.20)
    assert b[_phase_state.PHASE_EXPLORE] == pytest.approx(0.35)
    # KERNEL was raised 0.28 -> 0.35 (funded by SWEEP 0.12 -> 0.05) because
    # MI355X GEMM tuning pays a one-time ~20min JIT before any benchmarking.
    assert b[_phase_state.PHASE_KERNEL_AGENT] == pytest.approx(0.35)
    assert b[_phase_state.PHASE_SWEEP] == pytest.approx(0.05)
    assert sum(b.values()) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# 2. Audit LLM policy
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "env_val,expected",
    [
        ("", "auto"),
        ("auto", "auto"),
        ("on", "on"),
        ("1", "on"),
        ("true", "on"),
        ("always", "on"),
        ("off", "off"),
        ("0", "off"),
        ("false", "off"),
        ("never", "off"),
        ("garbage", "auto"),
    ],
)
def test_audit_use_llm_mode(monkeypatch: pytest.MonkeyPatch, env_val: str, expected: str):
    if env_val:
        monkeypatch.setenv("INFERENCE_OPTIMIZER_FRAMEWORK_AUDIT_USE_LLM", env_val)
    else:
        monkeypatch.delenv("INFERENCE_OPTIMIZER_FRAMEWORK_AUDIT_USE_LLM", raising=False)
    assert FrameworkPhase._framework_audit_use_llm_mode() == expected


@pytest.mark.parametrize(
    "audit,uncertain",
    [
        (None, True),
        ({}, True),
        ({"semantic_status": "unknown", "confidence": 0.99}, True),
        ({"semantic_status": "", "confidence": 0.99}, True),
        ({"semantic_status": "already_equivalent", "confidence": 0.2}, True),
        ({"semantic_status": "already_equivalent", "confidence": 0.9}, False),
        ({"semantic_status": "not_present", "confidence": 0.6}, False),
        ({"semantic_status": "not_present", "confidence": "bad"}, True),
    ],
)
def test_audit_verdict_uncertain(audit: dict[str, Any] | None, uncertain: bool):
    assert FrameworkPhase._framework_audit_verdict_uncertain(audit) is uncertain


# --------------------------------------------------------------------------- #
# Shared stub for the arm behavior
# --------------------------------------------------------------------------- #
class _State:
    def __init__(self, *, authoring: bool, local_explore: bool) -> None:
        self.phase = "FRAMEWORK_AGENT"
        self.framework = "sglang"
        self.gpu_type = "MI300X"
        self.model_class = "dense"
        self.precision = "fp8"
        self.model = "test-model"
        self.last_profile_kernel_breakdown = None
        self.framework_agent_authoring_enabled = authoring
        self.framework_local_explore_enabled = local_explore
        self.framework_agent_phase_done = False
        self.framework_agent_discover_failures = 0
        self.framework_agent_empty_discoveries = 0
        self.framework_agent_batches: list[dict[str, Any]] = []
        self.framework_agent_phase_progress: list[dict[str, Any]] = []
        self.framework_agent_critic_decisions: list[dict[str, Any]] = []
        self.framework_agent_specialist_candidate_map: dict[str, str] = {}
        self.phase_history: list[dict[str, Any]] = []
        self.gaps: list[dict[str, Any]] = []
        self._saves = 0

    def save(self, _session_dir: Path) -> None:
        self._saves += 1


class _Tasks:
    def __init__(self) -> None:
        self._queued: list[Any] = []
        self._running: list[Any] = []
        self.created: list[dict[str, Any]] = []

    async def queued(self) -> list[Any]:
        return list(self._queued)

    async def running(self) -> list[Any]:
        return list(self._running)

    async def create_or_return_existing(self, **kwargs: Any) -> tuple[Any, bool]:
        self.created.append(kwargs)
        task = SimpleNamespace(
            kind=kwargs.get("kind"),
            task_id=f"t-{len(self.created)}",
            params=kwargs.get("params") or {},
            state="queued",
        )
        self._queued.append(task)
        return task, False


class _Stub:
    """Binds the FrameworkPhase local-explore surface; overrides GPU/warm/lanes."""

    _LOCAL_EXPLORE_KIND = FrameworkPhase._LOCAL_EXPLORE_KIND
    _framework_candidate_key = staticmethod(FrameworkPhase._framework_candidate_key)
    _framework_local_explore_arm_enabled = FrameworkPhase._framework_local_explore_arm_enabled
    _compose_framework_local_explore_gap = FrameworkPhase._compose_framework_local_explore_gap
    _next_local_explore_candidate_id = FrameworkPhase._next_local_explore_candidate_id
    _make_local_explore_pseudo_candidate = FrameworkPhase._make_local_explore_pseudo_candidate
    _maybe_dispatch_local_explore = FrameworkPhase._maybe_dispatch_local_explore
    _enqueue_framework_agent_local_explore_specialist = (
        FrameworkPhase._enqueue_framework_agent_local_explore_specialist
    )
    _framework_processed_candidate_keys = FrameworkPhase._framework_processed_candidate_keys
    _unprocessed_framework_agent_candidates = FrameworkPhase._unprocessed_framework_agent_candidates
    _select_best_framework_agent_candidate = FrameworkPhase._select_best_framework_agent_candidate

    def __init__(self, tmp_path: Path, *, authoring: bool = True, local_explore: bool = True) -> None:
        self.session_dir = tmp_path
        self.shared_state = _State(authoring=authoring, local_explore=local_explore)
        self.state = self.shared_state
        self.tasks = _Tasks()

    # Overrides to keep the specialist dispatch hermetic (no GPU pool / warmup).
    def _framework_gpu_params(self) -> dict[str, Any]:
        return {}

    async def _warm_specialist_params(self, _params: dict[str, Any]) -> None:
        return None

    def _framework_authoring_lanes_ttl(self, _params: dict[str, Any], *, base_ttl_sec: int):
        return ["research_lane"], int(base_ttl_sec)

    def _build_framework_working_memory(self) -> dict[str, Any]:
        return {}

    @staticmethod
    def _render_framework_memory_for_prompt(_memory: dict[str, Any] | None) -> str:
        return ""

    async def _framework_agent_authoring_inflight(self) -> bool:
        return False


# --------------------------------------------------------------------------- #
# 3. Pseudo-candidate + id sequencing
# --------------------------------------------------------------------------- #
def test_pseudo_candidate_none_when_arm_disabled(tmp_path: Path):
    disabled_auth = _Stub(tmp_path, authoring=False, local_explore=True)
    assert disabled_auth._make_local_explore_pseudo_candidate() is None
    disabled_arm = _Stub(tmp_path, authoring=True, local_explore=False)
    assert disabled_arm._make_local_explore_pseudo_candidate() is None


def test_pseudo_candidate_shape_when_enabled(tmp_path: Path):
    stub = _Stub(tmp_path, authoring=True, local_explore=True)
    pseudo = stub._make_local_explore_pseudo_candidate()
    assert pseudo is not None
    assert pseudo["kind"] == FrameworkPhase._LOCAL_EXPLORE_KIND
    assert pseudo["candidate_id"] == "local_explore:0"
    assert pseudo["framework"] == "sglang"
    # Gap composed from workload taxonomy (fp8 / dense / attention keywords).
    assert isinstance(pseudo["gap_keywords"], list)
    assert "sglang" in pseudo["gap_keywords"]


def test_next_local_explore_id_increments_with_progress(tmp_path: Path):
    stub = _Stub(tmp_path, authoring=True, local_explore=True)
    assert stub._next_local_explore_candidate_id() == "local_explore:0"
    stub.shared_state.framework_agent_phase_progress.append(
        {"candidate_id": "local_explore:0", "status": "author_empty", "kept": False}
    )
    assert stub._next_local_explore_candidate_id() == "local_explore:1"
    # Non-local rows do not advance the sequence.
    stub.shared_state.framework_agent_phase_progress.append(
        {"candidate_id": "https://example.com/pr/9", "status": "reverted", "kept": False}
    )
    assert stub._next_local_explore_candidate_id() == "local_explore:1"


# --------------------------------------------------------------------------- #
# 4. Dispatch behavior
# --------------------------------------------------------------------------- #
def test_maybe_dispatch_local_explore_disabled_is_noop(tmp_path: Path):
    stub = _Stub(tmp_path, authoring=False, local_explore=True)
    dispatched = asyncio.run(stub._maybe_dispatch_local_explore(reason="discover_exhausted"))
    assert dispatched is False
    assert stub.tasks.created == []


def test_maybe_dispatch_local_explore_enabled_creates_specialist(tmp_path: Path):
    stub = _Stub(tmp_path, authoring=True, local_explore=True)
    dispatched = asyncio.run(stub._maybe_dispatch_local_explore(reason="discover_exhausted"))
    assert dispatched is True
    assert len(stub.tasks.created) == 1
    created = stub.tasks.created[0]
    assert created["kind"] == "specialist"
    params = created["params"]
    assert params["framework_agent_authoring"] is True
    assert params["framework_local_explore"] is True
    assert params["framework_agent_candidate_id"] == "local_explore:0"
    assert params["domain"] == "serving_specialist"
    # No upstream PR lead: the mandate is authored from live source.
    assert "LOCAL-EXPLORATION" in params["notes"]
    assert "installed package source" in params["notes"]
    # WebSearch/WebFetch available so the specialist can compare upstream latest.
    assert "WebSearch" in created["allowed_tools"]
    assert "WebFetch" in created["allowed_tools"]
    # Idempotency keyed on the candidate id.
    assert created["idempotency_key"] == "framework_agent_local_explore:local_explore:0"
    # The specialist->candidate provenance map is recorded.
    assert stub.shared_state.framework_agent_specialist_candidate_map == {"t-1": "local_explore:0"}


# --------------------------------------------------------------------------- #
# 5. Resident arm offers the pseudo-candidate to the ranker
# --------------------------------------------------------------------------- #
def test_select_best_offers_local_explore_to_ranker(tmp_path: Path):
    stub = _Stub(tmp_path, authoring=True, local_explore=True)
    stub.shared_state.framework_agent_batches = [
        {
            "batch_id": "b",
            "candidates": [
                {"candidate_id": "pr1", "pr_url": "https://example.com/pr/1", "repo": "a/b", "ref": "x"},
            ],
        }
    ]
    seen: dict[str, Any] = {}

    async def _rank(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
        seen["ids"] = [c.get("candidate_id") for c in candidates]
        # Pick the local-explore arm.
        for c in candidates:
            if str(c.get("kind") or "") == FrameworkPhase._LOCAL_EXPLORE_KIND:
                return c
        return None

    stub._rank_framework_agent_candidates_llm = _rank  # type: ignore[assignment]
    chosen = asyncio.run(stub._select_best_framework_agent_candidate())
    # Ranker saw both the real PR and the local-explore pseudo-candidate.
    assert "pr1" in seen["ids"]
    assert "local_explore:0" in seen["ids"]
    assert chosen is not None and chosen["kind"] == FrameworkPhase._LOCAL_EXPLORE_KIND


def test_select_best_no_pseudo_when_batch_empty(tmp_path: Path):
    """No PR batch -> return None (the discovery trigger), never a bare pseudo."""
    stub = _Stub(tmp_path, authoring=True, local_explore=True)
    chosen = asyncio.run(stub._select_best_framework_agent_candidate())
    assert chosen is None


# --------------------------------------------------------------------------- #
# 6. Pump pivot: discovery exhaustion dispatches local-explore, not phase_done
# --------------------------------------------------------------------------- #
class _PumpStub(_Stub):
    """Adds the pump surface with discovery/enablement/audit shimmed out."""

    _pump_framework_agent_phase = FrameworkPhase._pump_framework_agent_phase
    _record_framework_agent_phase_done = FrameworkPhase._record_framework_agent_phase_done

    async def _maybe_enqueue_enablement_specialist(self) -> str:
        return ""

    async def _discover_next_framework_batch(self) -> bool:
        return False


def test_pump_pivots_to_local_explore_on_discovery_failure(tmp_path: Path):
    stub = _PumpStub(tmp_path, authoring=True, local_explore=True)
    asyncio.run(stub._pump_framework_agent_phase())
    # The phase did NOT give up; a local-exploration specialist was dispatched.
    assert stub.shared_state.framework_agent_phase_done is False
    assert len(stub.tasks.created) == 1
    assert stub.tasks.created[0]["params"]["framework_local_explore"] is True
    assert (
        stub.tasks.created[0]["params"]["framework_agent_candidate_id"] == "local_explore:0"
    )


def test_pump_falls_back_to_exit_when_arm_disabled(tmp_path: Path):
    stub = _PumpStub(tmp_path, authoring=True, local_explore=False)
    # First (limit-1) empty discoveries retry; the limit-th marks the phase done.
    for _ in range(_fa_client.DISCOVER_FAILURE_RETRY_LIMIT):
        asyncio.run(stub._pump_framework_agent_phase())
    assert stub.tasks.created == []
    assert stub.shared_state.framework_agent_phase_done is True


def test_pump_routes_ranked_local_explore_candidate(tmp_path: Path):
    """A resident-arm pick routes to local-explore dispatch, skipping PR audit."""
    stub = _PumpStub(tmp_path, authoring=True, local_explore=True)
    stub.shared_state.framework_agent_batches = [
        {
            "batch_id": "b",
            "candidates": [
                {"candidate_id": "pr1", "pr_url": "https://example.com/pr/1", "repo": "a/b", "ref": "x"},
            ],
        }
    ]
    audited: dict[str, int] = {"n": 0}

    async def _rank(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
        for c in candidates:
            if str(c.get("kind") or "") == FrameworkPhase._LOCAL_EXPLORE_KIND:
                return c
        return None

    async def _audit(_candidate: dict[str, Any]) -> dict[str, Any]:
        audited["n"] += 1
        return {"recommended_next_step": ""}

    stub._rank_framework_agent_candidates_llm = _rank  # type: ignore[assignment]
    stub._audit_framework_agent_candidate = _audit  # type: ignore[assignment]

    asyncio.run(stub._pump_framework_agent_phase())

    assert audited["n"] == 0, "local-explore arm must not run the PR semantic audit"
    assert len(stub.tasks.created) == 1
    assert stub.tasks.created[0]["params"]["framework_local_explore"] is True


def test_forward_enablement_carriers_eval_origin():
    from hyperloom.orchestrator.phases.explore import _forward_enablement_carriers

    src = {
        "enablement_origin": "eval",
        "enablement_accuracy_floor": 0.4,
        "enablement_probe_config_path": "/runs/baseline/materialized.yaml",
        "enablement_eval_contract_fingerprint": "fp1",
    }
    dst: dict[str, Any] = {}
    _forward_enablement_carriers(src, dst)
    assert dst["enablement_origin"] == "eval"
    assert dst["enablement_accuracy_floor"] == 0.4
    assert dst["enablement_probe_config_path"] == "/runs/baseline/materialized.yaml"
    # The eval-contract fingerprint is no longer forwarded: nothing downstream
    # reads it. Correctness is judged from the candidate's own measurement.
    assert "enablement_eval_contract_fingerprint" not in dst
    # Benches against the original workload config, not the shipped default.
    assert dst["config_path"] == "/runs/baseline/materialized.yaml"


def test_forward_enablement_carriers_boot_origin_noop():
    from hyperloom.orchestrator.phases.explore import _forward_enablement_carriers

    dst: dict[str, Any] = {}
    _forward_enablement_carriers({}, dst)
    assert dst == {}
    # An existing config_path is not overwritten for boot-origin.
    dst2 = {"config_path": "/keep.yaml"}
    _forward_enablement_carriers({"enablement_origin": ""}, dst2)
    assert dst2 == {"config_path": "/keep.yaml"}
