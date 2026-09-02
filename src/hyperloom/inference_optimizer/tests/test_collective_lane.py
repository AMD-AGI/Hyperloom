# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Test Collective candidate selection, KERNEL gating, and state."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.kernel import request_handlers as krh
from hyperloom.orchestrator.kernel.request_handlers import (
    KERNEL_REQUEST_HANDLERS,
    _collective_budget,
    select_collective_candidate,
)
from hyperloom.orchestrator.kernel._kernel_decisions import (
    untried_hot_reusable_kernels,
)
from hyperloom.orchestrator.state.shared_state import SharedState


def _state_with_remaining(minutes):
    return SimpleNamespace(remaining_minutes=lambda: minutes)


class TestCollectiveBudget:
    """Collective campaigns must fit preparation and finalization."""

    def test_budget_follows_the_session_minus_a_reserve(self):
        hours, timeout = _collective_budget(_state_with_remaining(600.0), None, 0)
        assert hours == 8.33
        assert timeout == 33288

    def test_short_session_skips_when_one_hour_cannot_fit(self):
        """KernelForge rejects a campaign shorter than one hour."""
        hours, timeout = _collective_budget(_state_with_remaining(90.0), None, 14400)
        assert hours is None
        assert timeout == 0

    def test_explicit_request_wins(self):
        hours, timeout = _collective_budget(_state_with_remaining(600.0), 2.0, 14400)
        assert hours == 2.0
        assert timeout == 10500

    def test_explicit_request_is_clamped_by_remaining_session(self):
        hours, timeout = _collective_budget(_state_with_remaining(180.0), 3.0, 0)
        assert hours == 1.33
        assert timeout == 8088

    def test_explicit_timeout_skips_when_minimum_campaign_cannot_fit(self):
        """Preparation plus finalization leave under an hour of campaign."""
        hours, timeout = _collective_budget(_state_with_remaining(600.0), 2.0, 6600)
        assert hours is None
        assert timeout == 0

    def test_unbounded_session_uses_an_explicit_wall_window(self):
        assert _collective_budget(_state_with_remaining(None), None, 14400) == (3.08, 14388)

    def test_unbounded_session_without_limits_defers_to_forge(self):
        assert _collective_budget(_state_with_remaining(None), None, 0) == (None, 14400)

    def test_budget_below_the_reserve_skips(self):
        assert _collective_budget(_state_with_remaining(30.0), None, 14400) == (None, 0)

    def test_state_without_the_hook_is_tolerated(self):
        assert _collective_budget(SimpleNamespace(), None, 14400) == (3.08, 14388)


from hyperloom.orchestrator.phases.kernel import KernelPhase


def _collective_entry(**extra) -> dict:
    entry = {
        "kernel_id": "k007",
        "name": "hipLaunchKernel->_ZN5aiter18all_reduce... (Synthetic Op)",
        "gpu_pct": 4.5,
        "reusable_native_kernel": True,
        "source_file": "/sgl-workspace/aiter/csrc/include/custom_all_reduce.cuh",
        "source_function": "all_reduce_cross_device",
        "input_shapes": [{"shape": "(4096, 7168)"}],
        "input_dtypes": ["bf16"],
        "kernel_contract": {"kind": "collective", "collective_op": "all_reduce", "world_size": 8},
    }
    entry.update(extra)
    return entry


def _projection(entry: dict) -> dict:
    """Build the routing fields retained in shared state."""
    keep = (
        "kernel_id",
        "name",
        "gpu_pct",
        "source_file",
        "reusable_native_kernel",
        "kernel_contract",
        "is_multigpu",
        "candidate_source",
    )
    return {k: entry[k] for k in keep if k in entry}


def _state_from_disk(tmp_path, *entries) -> SimpleNamespace:
    """State shaped like a real session: projection in memory, full rows on disk."""
    path = tmp_path / "kernel_candidates.json"
    path.write_text(json.dumps({"hot_kernels": list(entries)}), encoding="utf-8")
    return SimpleNamespace(
        last_trace_analyze={
            "hot_kernels_top15": [_projection(e) for e in entries],
            "candidates_path": str(path),
        }
    )


# --- Candidate selection ------------------------------------------------------


def test_selects_the_hottest_collective(tmp_path):
    hot = _collective_entry(kernel_id="k002", gpu_pct=9.1)
    picked = select_collective_candidate(_state_from_disk(tmp_path, _collective_entry(), hot))
    assert picked["kernel_id"] == "k002"


def test_prefers_source_resolved_nccl_summary_over_wrapper(tmp_path):
    wrapper = _collective_entry(kernel_id="k002", gpu_pct=20.0)
    resolved = _collective_entry(
        kernel_id="k009",
        gpu_pct=4.0,
        candidate_source="nccl_summary",
    )
    picked = select_collective_candidate(_state_from_disk(tmp_path, wrapper, resolved))
    assert picked["kernel_id"] == "k009"


def test_skips_non_collective_kernels(tmp_path):
    gemm = _collective_entry(kernel_id="k001", gpu_pct=40.0, kernel_contract={"kind": "gemm"})
    picked = select_collective_candidate(_state_from_disk(tmp_path, gemm, _collective_entry()))
    assert picked["kernel_id"] == "k007"


def test_skips_non_reusable_candidates(tmp_path):
    """nccl/rccl reach here already marked non-reusable; they must stay out."""
    vendor = _collective_entry(kernel_id="k003", gpu_pct=30.0, reusable_native_kernel=False)
    assert select_collective_candidate(_state_from_disk(tmp_path, vendor)) is None


def test_skips_candidates_without_source(tmp_path):
    state = _state_from_disk(tmp_path, _collective_entry(source_file=""))
    assert select_collective_candidate(state) is None


def test_invalid_candidate_does_not_hide_a_valid_candidate(tmp_path):
    """One incomplete summary row must not poison the candidate pool."""
    invalid = _collective_entry(
        kernel_id="k001",
        candidate_source="nccl_summary",
        input_shapes=[],
        input_dtypes=[],
    )
    valid = _collective_entry(kernel_id="k002")

    picked = select_collective_candidate(_state_from_disk(tmp_path, invalid, valid))

    assert picked["kernel_id"] == "k002"


def test_returns_none_without_analysis():
    assert select_collective_candidate(SimpleNamespace(last_trace_analyze=None)) is None
    assert select_collective_candidate(SimpleNamespace()) is None


# --- The enriched rows live on disk, not in shared state ----------------------


def test_reads_the_enriched_rows_from_candidates_path(tmp_path):
    """Selection reads complete driver inputs from the artifact."""
    state = _state_from_disk(tmp_path, _collective_entry())
    picked = select_collective_candidate(state)
    assert picked is not None
    assert picked["kernel_id"] == "k007"
    assert all("kernel_contract" in row for row in state.last_trace_analyze["hot_kernels_top15"])


def test_still_ranks_by_gpu_pct_when_reading_from_disk(tmp_path):
    hot = _collective_entry(kernel_id="k002", gpu_pct=9.1)
    picked = select_collective_candidate(_state_from_disk(tmp_path, _collective_entry(), hot))
    assert picked["kernel_id"] == "k002"


def test_trace_projection_preserves_collective_queue_isolation(tmp_path):
    """Persisted routing fields must keep collectives out of kernel_opt."""
    state = SharedState.load_or_init(tmp_path)
    state.record_trace_analyze(
        {"trace_input": "/trace"},
        {
            "status": "ok",
            "hot_kernels": [
                _collective_entry(
                    is_multigpu=True,
                    candidate_source="nccl_summary",
                ),
            ],
            "trace_health_warnings": [],
        },
    )

    projected = state.last_trace_analyze["hot_kernels_top15"][0]
    assert projected["kernel_contract"]["kind"] == "collective"
    assert projected["is_multigpu"] is True
    assert untried_hot_reusable_kernels(state) == []
    # The prompt offers this list as the kernel_opt target set, so a collective
    # left in it would dispatch an empty batch on every pick.
    assert state.last_trace_analyze["reusable_native_kernel_ids"] == []
    codes = {w.get("code") for w in state.last_trace_analyze["trace_health_warnings"]}
    assert "collective_lane_withheld_kernels" in codes


@pytest.mark.parametrize(
    "entry,reason",
    [
        (
            {
                "kernel_contract": {"kind": "collective", "collective_op": "reduce"},
                "is_multigpu": True,
                "candidate_source": "nccl_summary",
            },
            "a single-GPU reduction the lane cannot measure",
        ),
        (
            {
                "kernel_contract": {"kind": "collective", "collective_op": "broadcast"},
                "is_multigpu": True,
                "candidate_source": "nccl_summary",
            },
            "a primitive outside the lane's supported set",
        ),
        (
            {
                "kernel_contract": {"kind": "collective", "collective_op": "all_reduce"},
                "is_multigpu": True,
                "candidate_source": "",
            },
            "a name/path heuristic match the lane never injected",
        ),
    ],
)
def test_a_kernel_the_collective_lane_cannot_admit_stays_routable(tmp_path, entry, reason):
    """The contract's ``collective`` kind is a heuristic, not lane ownership.

    It also fires on a plain ``block_reduce`` and on any source under ``dist/``.
    The lane is opt-in and admits none of these, so withholding them from
    kernel_opt would leave them with no lane at all.
    """
    state = SharedState.load_or_init(tmp_path)
    state.record_trace_analyze(
        {"trace_input": "/trace"},
        {
            "status": "ok",
            "hot_kernels": [_collective_entry(**entry)],
            "trace_health_warnings": [],
        },
    )

    assert state.last_trace_analyze["reusable_native_kernel_ids"] == ["k007"], reason


def test_multigpu_hint_without_collective_contract_stays_routable(tmp_path):
    """A coarse multi-GPU hint must not suppress a dense candidate."""
    state = SharedState.load_or_init(tmp_path)
    state.record_trace_analyze(
        {"trace_input": "/trace"},
        {
            "status": "ok",
            "hot_kernels": [
                _collective_entry(
                    is_multigpu=True,
                    kernel_contract={"kind": "gemm"},
                ),
            ],
            "trace_health_warnings": [],
        },
    )

    assert untried_hot_reusable_kernels(
        state,
        min_gpu_pct=0.0,
        top_n=10,
    ) == ["k007"]
    assert state.last_trace_analyze["reusable_native_kernel_ids"] == ["k007"]


def test_collective_replay_preserves_completed_integration(tmp_path):
    """Replaying a campaign must not reopen its completed integration."""
    state = SharedState.load_or_init(tmp_path)
    campaign = {
        "collective_attempt_id": "attempt-1",
        "integration_id": "integration-1",
        "status": "ok",
        "decision": "KEEP",
        "engine": "forge_collective",
        "kept": True,
        "requires_e2e_validation": True,
    }
    state.record_collective(campaign, tmp_path)
    state.record_collective_integration(
        {
            "status": "ok",
            "decision": "KEEP",
            "integration_status": "complete",
            "integration_recovery_action": "",
        },
        tmp_path,
        integration_id="integration-1",
    )

    state.record_collective(campaign, tmp_path)

    assert state.last_collective["patch_cleanup_status"] == "complete"
    assert state.last_collective["integration_decision"] == "KEEP"


# --- record_collective_integration ---------------------------------------------


def _seeded_collective_state(tmp_path):
    """A state whose last_collective/collective_attempts hold one open KEEP campaign."""
    state = SharedState.load_or_init(tmp_path)
    campaign = {
        "collective_attempt_id": "attempt-1",
        "integration_id": "integration-1",
        "status": "ok",
        "decision": "KEEP",
        "engine": "forge_collective",
        "kept": True,
        "requires_e2e_validation": True,
    }
    state.record_collective(campaign, tmp_path)
    return state


def test_record_collective_integration_updates_last_collective_and_history(tmp_path):
    """The verdict must land on both the live pointer and its campaign row."""
    state = _seeded_collective_state(tmp_path)

    state.record_collective_integration(
        {
            "status": "ok",
            "decision": "KEEP",
            "integration_status": "complete",
            "integration_recovery_action": "",
            "gain_pct": 12.5,
            "base_tput": 100.0,
            "new_tput": 112.5,
        },
        tmp_path,
        integration_id="integration-1",
    )

    assert state.last_collective["patch_cleanup_status"] == "complete"
    assert state.last_collective["integration_decision"] == "KEEP"
    assert state.last_collective["integration_gain_pct"] == 12.5

    history_entry = next(
        item
        for item in state.collective_attempts
        if item["collective_attempt_id"] == "attempt-1"
    )
    assert history_entry["patch_cleanup_status"] == "complete"
    assert history_entry["integration_gain_pct"] == 12.5


@pytest.mark.parametrize(
    "payload,exception,match",
    [
        ("not-a-dict", TypeError, "mapping"),
        (
            {"decision": "KEEP", "integration_status": "complete"},
            ValueError,
            "missing integration_id",
        ),
        (
            {
                "decision": "MAYBE",
                "integration_status": "complete",
                "integration_id": "integration-1",
            },
            ValueError,
            "Invalid collective integration decision",
        ),
        (
            {
                "decision": "KEEP",
                "integration_status": "pending",
                "integration_id": "integration-1",
            },
            ValueError,
            "patch_cleanup_status must be complete",
        ),
        (
            {
                "decision": "KEEP",
                "integration_status": "complete",
                "integration_id": "integration-1",
                "integration_recovery_action": "finalize",
            },
            ValueError,
            "cannot require recovery",
        ),
        (
            {
                "decision": "KEEP",
                "integration_status": "recovery_required",
                "integration_id": "integration-1",
                "integration_recovery_action": "retry",
            },
            ValueError,
            "recovery action must be finalize or revert",
        ),
        (
            {
                "decision": "KEEP",
                "integration_status": "complete",
                "integration_id": "integration-1",
                "gain_pct": float("nan"),
            },
            ValueError,
            "invalid gain_pct",
        ),
        (
            {
                "decision": "KEEP",
                "integration_status": "complete",
                "integration_id": "integration-99",
            },
            ValueError,
            "does not match last_collective",
        ),
    ],
    ids=[
        "non_mapping_result",
        "missing_integration_id",
        "invalid_decision",
        "invalid_integration_status",
        "completed_with_recovery_action",
        "unrecognized_recovery_action",
        "non_numeric_gain_pct",
        "integration_id_mismatch",
    ],
)
def test_record_collective_integration_rejects_invalid_input(tmp_path, payload, exception, match):
    """Each malformed verdict is rejected before any campaign state is touched."""
    state = _seeded_collective_state(tmp_path)
    with pytest.raises(exception, match=match):
        state.record_collective_integration(payload, tmp_path)


@pytest.mark.parametrize(
    "mutate_history",
    [
        lambda attempts: [],
        lambda attempts: attempts + [dict(attempts[0])],
    ],
    ids=["no_matching_campaign", "ambiguous_campaign"],
)
def test_record_collective_integration_rejects_bad_attempt_history(tmp_path, mutate_history):
    """A history the campaign never wrote to, or one that is ambiguous, must not be guessed at."""
    state = _seeded_collective_state(tmp_path)
    state.collective_attempts = mutate_history(state.collective_attempts)
    with pytest.raises(ValueError, match="exactly one campaign"):
        state.record_collective_integration(
            {
                "decision": "KEEP",
                "integration_status": "complete",
                "integration_id": "integration-1",
            },
            tmp_path,
        )


def test_record_collective_integration_rolls_back_when_save_fails(tmp_path, monkeypatch):
    """A failed persist must restore in-memory collective state and leave state.json untouched."""
    state = _seeded_collective_state(tmp_path)
    before_last = dict(state.last_collective)
    before_history = [dict(item) for item in state.collective_attempts]
    disk_before = (tmp_path / "state.json").read_text(encoding="utf-8")

    def _boom(_session_dir):
        raise OSError("disk full")

    monkeypatch.setattr(state, "save", _boom)

    with pytest.raises(OSError, match="disk full"):
        state.record_collective_integration(
            {
                "decision": "KEEP",
                "integration_status": "complete",
                "integration_id": "integration-1",
            },
            tmp_path,
        )

    assert state.last_collective == before_last
    assert state.collective_attempts == before_history
    assert (tmp_path / "state.json").read_text(encoding="utf-8") == disk_before


def test_rejects_a_missing_candidate_artifact(tmp_path):
    state = SimpleNamespace(
        last_trace_analyze={
            "hot_kernels_top15": [_collective_entry()],
            "candidates_path": str(tmp_path / "gone.json"),
        }
    )
    with pytest.raises(ValueError, match="invalid collective candidate artifact"):
        select_collective_candidate(state)


def test_rejects_an_unreadable_candidate_artifact(tmp_path):
    bad = tmp_path / "kernel_candidates.json"
    bad.write_text("{not json", encoding="utf-8")
    state = SimpleNamespace(
        last_trace_analyze={"hot_kernels_top15": [_collective_entry()], "candidates_path": str(bad)}
    )
    with pytest.raises(ValueError, match="invalid collective candidate artifact"):
        select_collective_candidate(state)


# --- KERNEL-entry gate --------------------------------------------------------


def _gate(*, tp=8, comm_pct=5.0, last_collective=None, skip_env=None, monkeypatch=None, analysis=True):
    if monkeypatch is not None:
        monkeypatch.setenv("HYPERLOOM_SKIP_COLLECTIVE", skip_env or "")
    state = SimpleNamespace(
        tp=tp,
        last_collective=last_collective or {},
        current_comm_pct=lambda: comm_pct,
        last_trace_analyze={"hot_kernels_top15": []} if analysis else {},
    )
    fake = SimpleNamespace(
        shared_state=state,
        COLLECTIVE_COMM_PCT_FLOOR=KernelPhase.COLLECTIVE_COMM_PCT_FLOOR,
        COLLECTIVE_CANDIDATE_GPU_PCT_FLOOR=KernelPhase.COLLECTIVE_CANDIDATE_GPU_PCT_FLOOR,
    )
    return KernelPhase._collective_required_before_kernel_opt(fake)


def test_gate_opens_for_multi_gpu_with_exposed_comm():
    assert _gate(tp=8, comm_pct=5.0) is True


@pytest.mark.parametrize("tp", [0, 1])
def test_gate_closed_below_tp2(tp):
    """A single rank issues no collective at all."""
    assert _gate(tp=tp) is False


def test_gate_closed_when_comm_is_overlapped():
    """Communication hidden behind compute is not worth a tuning round."""
    assert _gate(comm_pct=0.2) is False


def test_gate_closed_without_a_roofline_snapshot():
    """No roofline comm bucket and no resolvable candidate to fall back on."""
    assert _gate(comm_pct=None) is False


def test_gate_falls_back_to_the_candidate_share_without_a_roofline(monkeypatch):
    """The roofline comm bucket needs a TraceLens extension a public checkout lacks.

    Without a fallback the whole lane would disappear behind a log line on any
    such checkout, so the hottest resolved collective's own GPU share stands in.
    """
    monkeypatch.setattr(
        krh,
        "select_collective_candidate",
        lambda _state: {"kernel_id": "k007", "gpu_pct": 6.881},
    )

    assert _gate(comm_pct=None) is True


def test_candidate_fallback_still_respects_the_floor(monkeypatch):
    """The fallback substitutes the share, it does not bypass the gate."""
    monkeypatch.setattr(
        krh,
        "select_collective_candidate",
        lambda _state: {"kernel_id": "k007", "gpu_pct": 0.2},
    )

    assert _gate(comm_pct=None) is False


def test_the_fallback_share_is_judged_on_its_own_floor(monkeypatch):
    """The two shares are not the same measurement.

    The roofline value is the exposed part of all communication; the fallback is
    one kernel's whole GPU time, which a compute overlap can hide entirely. A
    share that clears the roofline floor must not therefore clear the fallback's.
    """
    between = (KernelPhase.COLLECTIVE_COMM_PCT_FLOOR + KernelPhase.COLLECTIVE_CANDIDATE_GPU_PCT_FLOOR) / 2
    monkeypatch.setattr(
        krh,
        "select_collective_candidate",
        lambda _state: {"kernel_id": "k007", "gpu_pct": between},
    )

    assert _gate(comm_pct=between) is True
    assert _gate(comm_pct=None) is False


def test_candidate_fallback_survives_an_unreadable_artifact(monkeypatch):
    """A broken candidates artifact closes the gate instead of raising."""

    def _raise(_state):
        raise ValueError("candidate artifact has no valid hot_kernels list")

    monkeypatch.setattr(krh, "select_collective_candidate", _raise)

    assert _gate(comm_pct=None) is False


@pytest.mark.parametrize("status", ["ok", "complete", "kept"])
def test_gate_is_idempotent_after_a_completed_run(status):
    assert _gate(last_collective={"status": status}) is False


def test_gate_reopens_after_a_failed_run():
    assert _gate(last_collective={"status": "failed"}) is True


# --- A skip is scoped to its analysis, not to the session --------------------


def _gate_with_analysis(candidates_path: str, last_collective: dict) -> bool:
    state = SimpleNamespace(
        tp=8,
        last_collective=last_collective,
        current_comm_pct=lambda: 5.0,
        last_trace_analyze={"hot_kernels_top15": [], "candidates_path": candidates_path},
    )
    fake = SimpleNamespace(
        shared_state=state,
        COLLECTIVE_COMM_PCT_FLOOR=KernelPhase.COLLECTIVE_COMM_PCT_FLOOR,
        COLLECTIVE_CANDIDATE_GPU_PCT_FLOOR=KernelPhase.COLLECTIVE_CANDIDATE_GPU_PCT_FLOOR,
    )
    return KernelPhase._collective_required_before_kernel_opt(fake)


def test_skip_is_terminal_for_the_analysis_that_produced_it():
    """Re-deciding the same analysis on every KERNEL re-entry is noise."""
    assert (
        _gate_with_analysis(
            "/run/a/kernel_candidates.json", {"status": "skipped", "analysis_key": "/run/a/kernel_candidates.json"}
        )
        is False
    )


def test_skip_does_not_block_a_later_analysis():
    """Nothing clears last_collective, so an unscoped skip would lock the lane
    out for the whole session even after a new trace exposes a collective."""
    assert (
        _gate_with_analysis(
            "/run/b/kernel_candidates.json", {"status": "skipped", "analysis_key": "/run/a/kernel_candidates.json"}
        )
        is True
    )


def test_skip_without_an_analysis_key_does_not_block():
    assert _gate_with_analysis("/run/a/kernel_candidates.json", {"status": "skipped"}) is True


def test_gate_closed_before_any_trace_analysis():
    """Candidate selection reads the analysis, so a skip recorded before one
    exists would wrongly become terminal."""
    assert _gate(analysis=False) is False


def test_gate_respects_the_kill_switch(monkeypatch):
    assert _gate(skip_env="1", monkeypatch=monkeypatch) is False


@pytest.mark.asyncio
async def test_collective_only_mode_never_falls_through_to_kernel_opt(tmp_path, monkeypatch):
    """A directed Collective session must wind down even when its gate is closed."""
    from hyperloom.orchestrator.phases import machine_state

    class _State:
        """Minimal state sink for the Collective-only wind-down hint."""

        def __init__(self):
            self.hint = ""

        def set_pending_escalate_hint(self, hint):
            """Record the requested phase escalation."""
            self.hint = hint

        def save(self, _session_dir):
            """Accept the durable-save call made by the phase."""

    state = _State()
    fake = SimpleNamespace(
        shared_state=state,
        session_dir=tmp_path,
        _collective_required_before_kernel_opt=lambda: False,
        _collective_only_mode=lambda: True,
    )
    monkeypatch.setenv("HYPERLOOM_COLLECTIVE_ONLY", "1")

    await KernelPhase._maybe_run_collective_before_kernel_opt(fake)

    assert state.hint == machine_state.ESCALATE_HINT_SKIP_TO_SWEEP


# --- Registration -------------------------------------------------------------


def test_handler_is_registered():
    assert "run_collective" in KERNEL_REQUEST_HANDLERS


def test_lane_is_not_exposed_to_the_llm():
    """Deterministic gate => Coordinator-only, same posture as fusion."""
    from hyperloom.inference_optimizer.protocol.action_surfaces import (
        FULL_ENABLED_ACTIONS,
        KERNEL_AGENT_OWNED_ACTIONS,
    )
    from hyperloom.orchestrator.phases.machine_state import (
        PHASE_ALLOWED_ACTIONS,
        PHASE_KERNEL_AGENT,
    )

    assert "run_collective" not in KERNEL_AGENT_OWNED_ACTIONS
    assert "run_collective" not in FULL_ENABLED_ACTIONS
    assert "run_collective" not in PHASE_ALLOWED_ACTIONS[PHASE_KERNEL_AGENT]


def test_policy_gate_rejects_an_llm_issued_lane_request():
    """Absence from the allowlists is not a gate; PolicyGate must deny outright.

    A registered handler with no owning action set would otherwise skip every
    phase check, letting the LLM bypass the comm_pct gate, ``record_collective``
    accounting, and ``_integrate_collective``.
    """
    from hyperloom.orchestrator.policy.gate import PolicyDenied, PolicyGate
    from hyperloom.orchestrator.roles.agent_role import default_role_registry

    registry = default_role_registry()
    gate = PolicyGate(role_registry=registry, session_dir=None)

    for kind in ("run_collective", "run_fusion"):
        with pytest.raises(PolicyDenied) as exc:
            gate._validate_request(
                registry["orchestration"],
                {"target_agent": "kernel_agent", "kind": kind},
            )
        assert exc.value.rule == "phase_incompatible"
        assert kind in str(exc.value)


def test_resume_compat_old_integration_status_accepted(tmp_path):
    """A state.json written before the patch_cleanup_status migration must load cleanly.

    The validator and classifiers fall back to the legacy 'integration_status'
    field so a --resume of a session that was mid-collective-integration when
    the binary was updated does not raise.
    """
    from hyperloom.orchestrator.state.shared_state import SharedState
    from hyperloom.orchestrator.kernel.attempt_summary import (
        _classify_collective_attempt,
        CATEGORY_INTEGRATED,
        CATEGORY_KEEP_PENDING,
    )
    from hyperloom.orchestrator.phases.machine_state import collective_integration_pending
    from types import SimpleNamespace

    state = SharedState.load_or_init(tmp_path)
    campaign = {
        "collective_attempt_id": "attempt-legacy",
        "integration_id": "integ-legacy",
        "status": "ok",
        "decision": "KEEP",
        "engine": "forge_collective",
        "kept": True,
        "requires_e2e_validation": True,
    }
    state.record_collective(campaign, tmp_path)

    # Simulate a state.json written by old code: integration_status instead of
    # patch_cleanup_status.
    old_style_result = {
        "status": "ok",
        "decision": "KEEP",
        "integration_status": "complete",  # old field name
        "integration_recovery_action": "",
    }
    # Should not raise despite using the old field.
    state.record_collective_integration(old_style_result, tmp_path, integration_id="integ-legacy")

    # Reads are also backward-compat.
    last = state.last_collective
    assert last.get("patch_cleanup_status") == "complete" or last.get("integration_status") == "complete"
    assert last["integration_decision"] == "KEEP"

    # Classifiers handle old records.
    old_record = {"integration_decision": "KEEP", "integration_status": "complete"}
    assert _classify_collective_attempt(old_record) == CATEGORY_INTEGRATED

    old_pending = {"integration_decision": "KEEP", "integration_status": "pending"}
    assert _classify_collective_attempt(old_pending) == CATEGORY_KEEP_PENDING

    # collective_integration_pending uses fallback too.
    fake_state = SimpleNamespace(
        last_collective={
            "kept": True,
            "requires_e2e_validation": True,
            "integration_status": "pending",  # old field, no patch_cleanup_status
        }
    )
    assert collective_integration_pending(fake_state) is True


def _keep_campaign(attempt_id: str, integration_id: str) -> dict:
    return {
        "collective_attempt_id": attempt_id,
        "integration_id": integration_id,
        "status": "ok",
        "decision": "KEEP",
        "engine": "forge_collective",
        "kept": True,
        "requires_e2e_validation": True,
    }


def test_record_collective_rolls_back_when_save_fails(tmp_path, monkeypatch):
    """A failed persist must restore in-memory collective state and leave state.json untouched."""
    state = SharedState.load_or_init(tmp_path)
    state.record_collective(_keep_campaign("attempt-1", "integration-1"), tmp_path)
    before_last = dict(state.last_collective)
    before_attempts = [dict(item) for item in state.collective_attempts]
    disk_before = (tmp_path / "state.json").read_text(encoding="utf-8")

    def _boom(_session_dir):
        raise OSError("disk full")

    monkeypatch.setattr(state, "save", _boom)
    with pytest.raises(OSError, match="disk full"):
        state.record_collective(_keep_campaign("attempt-2", "integration-2"), tmp_path)

    assert state.last_collective == before_last
    assert state.collective_attempts == before_attempts
    assert (tmp_path / "state.json").read_text(encoding="utf-8") == disk_before


def test_record_collective_integration_rolls_back_when_save_fails(tmp_path, monkeypatch):
    """A failed persist must restore last_collective/collective_attempts and leave state.json untouched."""
    state = SharedState.load_or_init(tmp_path)
    state.record_collective(_keep_campaign("attempt-1", "integration-1"), tmp_path)
    before_last = dict(state.last_collective)
    before_attempts = [dict(item) for item in state.collective_attempts]
    disk_before = (tmp_path / "state.json").read_text(encoding="utf-8")

    def _boom(_session_dir):
        raise OSError("disk full")

    monkeypatch.setattr(state, "save", _boom)
    with pytest.raises(OSError, match="disk full"):
        state.record_collective_integration(
            {
                "status": "ok",
                "decision": "KEEP",
                "patch_cleanup_status": "complete",
                "patch_cleanup_action": "",
            },
            tmp_path,
            integration_id="integration-1",
        )

    assert state.last_collective == before_last
    assert state.collective_attempts == before_attempts
    assert (tmp_path / "state.json").read_text(encoding="utf-8") == disk_before
