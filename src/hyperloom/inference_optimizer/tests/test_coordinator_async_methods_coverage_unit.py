# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coverage for Coordinator async/stateful methods invoked directly against a real (mock-backed) Coordinator: SharedState promotion across task kinds, prompt composition per agent, advisory blocks, research-scout harvest, the orchestration checkpoint guard, strategy-change escalation, specialist autosubmit routing and warm-up, and per-task/per-variant fact journaling."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from hyperloom.common.deadline import Deadline
from hyperloom.orchestrator.roles import (
    Backend,
    MockBackend,
    ScriptedPlan,
)
from hyperloom.inference_optimizer.session.optimization_journal import Verdict
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.state.objective import TargetGainObjective, TimeOnlyObjective
from hyperloom.orchestrator.state.task_registry import Task
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType


def _heartbeat() -> Intent:
    return Intent(type=IntentType.SEND_MESSAGE, payload={"topic": "heartbeat", "body_md": "ok"})


def _silent_plan() -> ScriptedPlan:
    return ScriptedPlan(turns=[], default_intent=_heartbeat())


def _build_backends() -> dict[str, Backend]:
    return {name: MockBackend(_silent_plan(), name=name) for name in ("orchestration", "critic")}


@pytest.fixture
def coord(session_dir) -> Coordinator:
    return Coordinator(session_dir, backends=_build_backends())


# -- promote_to_shared_state ----------------------------------------------
@pytest.mark.asyncio
async def test_promote_baseline_sets_anchor_and_current_best(coord: Coordinator) -> None:
    coord.shared_state.auto_roofline_pending_task_id = "pending-x"
    coord.shared_state.baseline_failure_streak = 2
    coord.shared_state.baseline_arg_error_streak = 1
    await coord.writeback.promote_to_shared_state(
        "baseline",
        {
            "output_throughput": 1000.0,
            "warmup_round_tput": 900.0,
            "accuracy": 0.95,
            "subprocess_runtime_sec": 120.0,
            "ttft_mean_ms": 100.0,
            "e2el_mean_ms": 2000.0,
            "tpot_mean_ms": 10.0,
            "workspace": "/tmp/ws",
        },
    )
    assert coord.shared_state.baseline_tput == 1000.0
    assert coord.shared_state.baseline_failure_streak == 0
    # The arg-error streak is not a boot-failure streak and a promotion does not
    # clear it; nothing counts into it again once an anchor exists.
    assert coord.shared_state.baseline_arg_error_streak == 1
    assert coord.shared_state.current_best["action"] == "baseline"
    assert coord.shared_state.current_best["tput"] == 1000.0
    assert coord.shared_state.current_best["cold_tput"] == 900.0


@pytest.mark.asyncio
async def test_promote_single_round_baseline_clears_stale_warm_runtime(coord: Coordinator) -> None:
    coord.shared_state.auto_roofline_pending_task_id = "pending-x"
    coord.shared_state.baseline_warm_runtime_sec = 7.5

    await coord.writeback.promote_to_shared_state(
        "baseline",
        {
            "output_throughput": 1000.0,
            "subprocess_runtime_sec": 120.0,
            "workspace": "/tmp/ws",
        },
    )

    assert coord.shared_state.baseline_runtime_sec == 120.0
    assert coord.shared_state.baseline_warm_runtime_sec == 0.0


@pytest.mark.asyncio
async def test_promote_baseline_carries_the_boot_and_benchmark_split(coord: Coordinator) -> None:
    """The two figures that let later work be priced on what it will spend."""
    await coord.writeback.promote_to_shared_state(
        "baseline",
        {
            "output_throughput": 1000.0,
            "subprocess_runtime_sec": 900.0,
            "post_ready_runtime_sec": 550.0,
            "workspace": "/tmp/ws",
        },
    )

    assert coord.shared_state.baseline_runtime_sec == 900.0
    assert coord.shared_state.baseline_post_ready_runtime_sec == 550.0


@pytest.mark.asyncio
async def test_promote_baseline_clears_a_split_a_later_round_did_not_report(
    coord: Coordinator,
) -> None:
    """A stale split would be subtracted from a fresh total and called the boot."""
    coord.shared_state.baseline_post_ready_runtime_sec = 550.0

    await coord.writeback.promote_to_shared_state(
        "baseline",
        {
            "output_throughput": 1000.0,
            "subprocess_runtime_sec": 900.0,
            "workspace": "/tmp/ws",
        },
    )

    assert coord.shared_state.baseline_post_ready_runtime_sec == 0.0


@pytest.mark.asyncio
async def test_promote_baseline_carries_a_dropped_hot_pass_to_the_session(
    coord: Coordinator,
) -> None:
    """The marker drives a session-level decision, so it has to reach the session."""
    await coord.writeback.promote_to_shared_state(
        "baseline",
        {
            "output_throughput": 1000.0,
            "subprocess_runtime_sec": 900.0,
            "measure_round_dropped": {"reason": "measure_round_reaped_by_the_run"},
            "workspace": "/tmp/ws",
        },
    )
    assert coord.shared_state.baseline_measure_round_dropped is True

    await coord.writeback.promote_to_shared_state(
        "baseline",
        {
            "output_throughput": 1200.0,
            "subprocess_runtime_sec": 900.0,
            "measure_round_runtime_sec": 400.0,
            "workspace": "/tmp/ws",
        },
    )
    assert coord.shared_state.baseline_measure_round_dropped is False


class TestAHotPassCorrectsAColdAnchor:
    """The escape from the marker, without which PRELUDE cannot finish."""

    @pytest.mark.asyncio
    async def test_a_lower_hot_figure_replaces_a_marked_cold_one(self, coord: Coordinator) -> None:
        coord.shared_state.baseline_tput = 1000.0
        coord.shared_state.baseline_measure_round_dropped = True

        await coord.writeback.promote_to_shared_state(
            "baseline",
            {
                "output_throughput": 980.0,
                "subprocess_runtime_sec": 900.0,
                "measure_round_runtime_sec": 400.0,
                "workspace": "/tmp/ws",
            },
        )

        assert coord.shared_state.baseline_tput == 980.0
        assert coord.shared_state.baseline_measure_round_dropped is False
        assert coord.shared_state.baseline_warm_runtime_sec == 400.0

    @pytest.mark.asyncio
    async def test_a_lower_cold_figure_does_not_replace_a_marked_cold_one(
        self,
        coord: Coordinator,
    ) -> None:
        """Only a hot pass corrects the anchor; another cold one is just noisier."""
        coord.shared_state.baseline_tput = 1000.0
        coord.shared_state.baseline_measure_round_dropped = True

        await coord.writeback.promote_to_shared_state(
            "baseline",
            {
                "output_throughput": 980.0,
                "subprocess_runtime_sec": 900.0,
                "workspace": "/tmp/ws",
            },
        )

        assert coord.shared_state.baseline_tput == 1000.0
        assert coord.shared_state.baseline_measure_round_dropped is True

    @pytest.mark.asyncio
    async def test_a_lower_hot_figure_still_loses_to_a_hot_anchor(self, coord: Coordinator) -> None:
        """With no marker there is nothing to correct, so drift is refused again."""
        coord.shared_state.baseline_tput = 1000.0
        coord.shared_state.baseline_measure_round_dropped = False

        await coord.writeback.promote_to_shared_state(
            "baseline",
            {
                "output_throughput": 980.0,
                "subprocess_runtime_sec": 900.0,
                "measure_round_runtime_sec": 400.0,
                "workspace": "/tmp/ws",
            },
        )

        assert coord.shared_state.baseline_tput == 1000.0


@pytest.mark.asyncio
async def test_unpromotable_baseline_fast_arg_errors_stop_after_two(
    coord: Coordinator,
) -> None:
    task = Task(
        task_id="baseline-fast-arg",
        kind="baseline",
        state="running",
        params={"config_path": "baseline.yaml"},
        idempotency_key="baseline-fast-arg",
    )
    result = {
        "status": "failed",
        "error_class": "fast_exit_arg_error",
        "error": "ValueError: Unknown attention backend: ROCM_FLASH",
    }

    await coord.writeback.handle_unpromotable_result(task, result)
    assert coord.shared_state.baseline_arg_error_streak == 1
    assert coord.shared_state.stop_reason != "baseline_arg_error"

    await coord.writeback.handle_unpromotable_result(task, result)
    assert coord.shared_state.baseline_arg_error_streak == 2
    assert coord.shared_state.baseline_failure_streak == 0
    assert coord.shared_state.stop_reason == "baseline_arg_error"


@pytest.mark.asyncio
async def test_unpromotable_baseline_agentx_preflight_stops_immediately(
    coord: Coordinator,
) -> None:
    """A missing AgentX client is a supply gap, not a code gap."""
    from hyperloom.orchestrator.actions.executors._subprocess_kill import (
        AGENTX_PREFLIGHT_ERROR_CLASS,
    )
    from hyperloom.inference_optimizer.breakdown.stop_reasons import AGENTX_PREFLIGHT_STOP_REASON

    task = Task(
        task_id="baseline-agentx-preflight",
        kind="baseline",
        state="running",
        params={"config_path": "baseline.yaml"},
        idempotency_key="baseline-agentx-preflight",
    )
    result = {
        "status": "failed",
        "error_class": AGENTX_PREFLIGHT_ERROR_CLASS,
        "error": "AgentX preflight failed: HYPERLOOM_AGENTX is on but aiperf was not found.",
    }

    await coord.writeback.handle_unpromotable_result(task, result)

    assert coord.shared_state.stop_reason == AGENTX_PREFLIGHT_STOP_REASON
    # No launch log is stashed: the FRAMEWORK pump reads a non-blank log as "there is something here to author
    # against", and there is not.
    assert not (coord.shared_state.enablement.launch_log or "").strip()
    # The slow-baseline retry budget is untouched -- retrying cannot help.
    assert coord.shared_state.baseline_failure_streak == 0


@pytest.mark.asyncio
async def test_unpromotable_baseline_mixed_classes_stop_after_three_total(
    coord: Coordinator,
) -> None:
    """Mixed subprocess_nonzero + fast_exit_arg_error failures must still fast-fail once 3 total baseline failures accrue, even though neither per-class streak reaches its own threshold."""

    def _task() -> Task:
        return Task(
            task_id="bl-mixed",
            kind="baseline",
            state="running",
            params={"config_path": "baseline.yaml"},
            idempotency_key="bl-mixed",
        )

    coord.shared_state.enablement_mode = "off"
    # Two different outputs: the identical-failure rule must not be what stops the third call.
    subproc_a = _nonzero_result("RuntimeError: HIP out of memory")
    subproc_b = _nonzero_result("ConnectionError: port refused")
    argerr = {"status": "failed", "error_class": "fast_exit_arg_error", "error": "bad arg"}

    await coord.writeback.handle_unpromotable_result(_task(), subproc_a)
    await coord.writeback.handle_unpromotable_result(_task(), argerr)
    assert coord.shared_state.stop_reason not in (
        "baseline_failed",
        "baseline_arg_error",
    )
    await coord.writeback.handle_unpromotable_result(_task(), subproc_b)
    assert coord.shared_state.baseline_failure_streak == 2
    assert coord.shared_state.baseline_total_failures == 3
    assert coord.shared_state.stop_reason == "baseline_failed"


@pytest.mark.asyncio
async def test_promote_profile_succeeded_records_trace(coord: Coordinator) -> None:
    coord.shared_state.baseline_tput = 800.0
    await coord.writeback.promote_to_shared_state(
        "profile",
        {
            "status": "succeeded",
            "main_trace_path": "/tmp/trace.json",
            "output_throughput": 820.0,
        },
    )
    assert coord.shared_state.last_profile_status == "succeeded"
    assert coord.shared_state.last_profile_trace == "/tmp/trace.json"


@pytest.mark.asyncio
async def test_promote_profile_failed_clears_trace(coord: Coordinator) -> None:
    coord.shared_state.last_profile_trace = "/tmp/old.trace.json"
    coord.shared_state.last_profile_args = "--old-backend"
    coord.shared_state.last_profile_workload = {"framework": "vllm"}
    await coord.writeback.promote_to_shared_state(
        "profile",
        {
            "status": "failed",
            "error_class": "no_trace_files",
        },
    )
    assert coord.shared_state.last_profile_status == "failed"
    assert coord.shared_state.last_profile_trace == ""
    assert coord.shared_state.last_profile_args == ""
    assert coord.shared_state.last_profile_workload == {}


@pytest.mark.asyncio
async def test_promote_profile_does_not_reuse_unready_merged_trace(coord: Coordinator) -> None:
    coord.shared_state.last_profile_trace = "/tmp/old.trace.json"
    await coord.writeback.promote_to_shared_state(
        "profile",
        {
            "status": "failed",
            "error_class": "primary_rank_trace_missing",
            "trace_input_ready": False,
            "trace_files": ["/tmp/merged.trace.json.gz"],
        },
    )
    assert coord.shared_state.last_profile_status == "failed"
    assert coord.shared_state.last_profile_trace == ""


@pytest.mark.asyncio
async def test_promote_roofline_succeeded_and_skipped_and_failed(coord: Coordinator) -> None:
    coord.shared_state.baseline_tput = 800.0
    await coord.writeback.promote_to_shared_state("roofline", {"status": "succeeded"})
    await coord.writeback.promote_to_shared_state("roofline", {"status": "skipped"})
    await coord.writeback.promote_to_shared_state(
        "roofline",
        {
            "status": "failed",
            "error_class": "boom",
            "phase": "trace",
        },
    )
    assert getattr(coord.shared_state, "roofline_failure_streak", 0) >= 1


@pytest.mark.asyncio
async def test_promote_explore_with_winner(coord: Coordinator) -> None:
    coord.shared_state.baseline_tput = 800.0
    await coord.writeback.promote_to_shared_state(
        "explore",
        {
            "winners": [{"name": "v0", "fingerprint": "fp-v0", "extra_server_args": "--tp 1", "tput": 900.0}],
            "best_variant": {"name": "v0", "extra_server_args": "--tp 1", "tput": 900.0},
            "output_throughput": 900.0,
            "round_id": "r1",
            "losers": [],
            "skipped_dup": [],
        },
    )
    assert coord.shared_state.current_best.get("tput") == 900.0


@pytest.mark.asyncio
async def test_compose_prompt_orchestration_with_time_budget(coord: Coordinator) -> None:
    coord._run_started_monotonic = time.monotonic() - 60.0
    coord._run_deadline = Deadline.after(600.0)
    coord.shared_state.max_minutes = 60
    out = (await coord.conversation.compose_prompt("orchestration")).text
    assert "SESSION_DIR=" in out
    assert "Mission progress" in out
    assert "Time budget" in out


@pytest.mark.asyncio
async def test_compose_prompt_orchestration_deadline_imminent_warning(coord: Coordinator) -> None:
    coord._run_started_monotonic = time.monotonic() - 60.0
    coord._run_deadline = Deadline.after(60.0)
    coord.shared_state.max_minutes = 60
    coord.shared_state.closing_phase = False
    out = (await coord.conversation.compose_prompt("orchestration")).text
    assert "< 5 min remaining" in out


@pytest.mark.asyncio
async def test_compose_prompt_kernel(coord: Coordinator) -> None:
    coord._run_started_monotonic = time.monotonic() - 60.0
    coord._run_deadline = Deadline.after(600.0)
    coord.shared_state.max_minutes = 60
    out_k = (await coord.conversation.compose_prompt("kernel_agent")).text
    assert "SESSION_DIR=" in out_k


#: ``(heading, owner, producer)`` for every advisory the orchestration prompt carries, in the order it carries
#: them. A heading of ``None`` is a block the prompt emits without one. Section order is prompt content: the
#: model reads these in this sequence, so a reordering is a change to what it was asked. The session-stable
#: advisories lead, in the cached context; the plateau advisory moves with the session, so it is rendered per
#: tick and lands last.
_ORCHESTRATION_ADVISORIES = (
    ("=== Warm start (Recipe KB T0) ===", "shared_state", "to_warm_start_summary"),
    ("=== Current gaps ===", "shared_state", "to_gaps_summary"),
    (None, "conversation", "_specialist_findings_block"),
    ("=== External target gap (advisory) ===", "conversation", "target_gap_advisory_block"),
    ("=== Specialist proposal scores (advisory) ===", "shared_state", "to_proposal_scores_summary"),
    ("=== Priors-match (advisory ordering) ===", "conversation", "_priors_match_advisory_block"),
    ("=== Intervention mix (telemetry) ===", "shared_state", "to_intervention_mix_summary"),
    ("=== Bottleneck redirect (advisory) ===", "conversation", "_bottleneck_redirect_advisory_block"),
    ("=== Acceptance threshold (advisory) ===", "conversation", "_acceptance_threshold_advisory_block"),
    ("=== Discarded escalation hint (advisory) ===", "conversation", "_discarded_escalate_hint_advisory_block"),
    ("=== Plateau advisory ===", "conversation", "plateau_advisory_block"),
)


def _stub_advisories(coord: Coordinator, monkeypatch) -> list[str]:
    """Give every advisory a unique non-empty block; return the markers in prompt order."""
    owners = {"shared_state": coord.shared_state, "conversation": coord.conversation}
    markers = []
    for _heading, owner, producer in _ORCHESTRATION_ADVISORIES:
        marker = f"<{producer}>"
        monkeypatch.setattr(owners[owner], producer, lambda marker=marker: marker)
        markers.append(marker)
    return markers


@pytest.mark.asyncio
async def test_compose_prompt_keeps_the_orchestration_advisory_order(coord: Coordinator, monkeypatch) -> None:
    markers = _stub_advisories(coord, monkeypatch)

    prompt = (await coord.conversation.compose_prompt("orchestration")).text

    for (heading, _owner, _producer), marker in zip(_ORCHESTRATION_ADVISORIES, markers):
        assert (marker if heading is None else f"{heading}\n{marker}") in prompt
    positions = [prompt.index(marker) for marker in markers]
    assert positions == sorted(positions)


@pytest.mark.asyncio
async def test_compose_prompt_withholds_every_advisory_from_a_reviewer(coord: Coordinator, monkeypatch) -> None:
    markers = _stub_advisories(coord, monkeypatch)

    prompt = (await coord.conversation.compose_prompt("critic")).text

    assert [marker for marker in markers if marker in prompt] == []


@pytest.mark.asyncio
async def test_compose_prompt_drops_the_heading_of_an_advisory_that_produced_nothing(
    coord: Coordinator,
    monkeypatch,
) -> None:
    _stub_advisories(coord, monkeypatch)
    monkeypatch.setattr(coord.conversation, "plateau_advisory_block", lambda: "")

    prompt = (await coord.conversation.compose_prompt("orchestration")).text

    assert "=== Plateau advisory ===" not in prompt
    # The trajectory digest is shown only on a plateau, so it goes with it.
    assert "=== Trajectory review (advisory) ===" not in prompt
    assert "=== Bottleneck redirect (advisory) ===" in prompt


@pytest.mark.asyncio
async def test_compose_prompt_puts_the_trajectory_review_under_the_plateau_advisory(
    coord: Coordinator,
    monkeypatch,
) -> None:
    from hyperloom.orchestrator.knowledge import trajectory_reviewer

    _stub_advisories(coord, monkeypatch)
    monkeypatch.setattr(trajectory_reviewer, "build_trajectory_digest", lambda *_a, **_k: "<trajectory>")

    prompt = (await coord.conversation.compose_prompt("orchestration")).text

    assert (
        "=== Plateau advisory ===\n<plateau_advisory_block>\n=== Trajectory review (advisory) ===\n<trajectory>"
    ) in prompt


def _unreadable(message: str):
    def _fail(*_a, **_k):
        raise RuntimeError(message)

    return _fail


def _error_messages(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]


@pytest.mark.asyncio
async def test_trajectory_reviewer_task_is_seeded_without_a_digest_when_the_digest_fails(
    coord: Coordinator, monkeypatch, caplog
) -> None:
    from hyperloom.orchestrator.knowledge import trajectory_reviewer

    created: list[dict] = []

    async def _create_or_return_existing(spec):
        created.append(spec.params)
        return SimpleNamespace(task_id="t-1"), False

    async def _warm(_params):
        return None

    def _undecodable_journal(*_a, **_k):
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "journal unreadable")

    monkeypatch.setattr(trajectory_reviewer, "build_trajectory_digest", _undecodable_journal)
    monkeypatch.setattr(coord.conversation, "plateau_advisory_block", lambda: "plateau")
    monkeypatch.setattr(coord.specialist_dispatch, "warm_specialist_params", _warm)
    monkeypatch.setattr(coord.gpu_lanes, "framework_authoring_lanes_ttl", lambda params, *, base_ttl_sec: ([], 60))
    monkeypatch.setattr(coord.tasks, "create_or_return_existing", _create_or_return_existing)

    with caplog.at_level("ERROR"):
        await coord.phase_internal.maybe_enqueue_trajectory_reviewer()

    (params,) = created
    assert params["gap_symptom"].endswith("exhausted ones).\n(no digest)")
    assert _error_messages(caplog) == ["Coordinator: trajectory review digest failed"]


def test_primary_gap_lookup_failure_propagates(coord: Coordinator, monkeypatch) -> None:
    from hyperloom.inference_optimizer.baseline_comparison import research_hints as rh

    monkeypatch.setattr(rh, "load_competitor_target", _unreadable("target unreadable"))
    coord.shared_state.target_advisory_enabled = True

    with pytest.raises(RuntimeError, match="target unreadable"):
        coord.conversation._current_primary_gap()


@pytest.mark.parametrize("failing", ["load_hints", "load_competitor_target"])
def test_priors_match_advisory_failure_propagates(coord: Coordinator, monkeypatch, failing: str) -> None:
    from hyperloom.inference_optimizer.baseline_comparison import research_hints as rh

    monkeypatch.setattr(coord.conversation, "_recent_proposed_variants", lambda: [{"name": "v"}])
    monkeypatch.setattr(rh, "load_hints", lambda _session_dir: {})
    monkeypatch.setattr(rh, "load_competitor_target", lambda _session_dir: None)
    monkeypatch.setattr(rh, failing, _unreadable("hints unreadable"))
    coord.shared_state.target_advisory_enabled = True

    with pytest.raises(RuntimeError, match="hints unreadable"):
        coord.conversation._priors_match_advisory_block()


# -- advisory blocks -------------------------------------------------------
def test_advisory_blocks_disabled_return_empty(coord: Coordinator) -> None:
    coord.shared_state.target_advisory_enabled = False
    assert coord.conversation.target_gap_advisory_block() == ""
    assert coord.conversation._current_primary_gap() is None


def test_plateau_advisory_block_no_signal(coord: Coordinator) -> None:
    assert isinstance(coord.conversation.plateau_advisory_block(), str)


def test_priors_match_advisory_block_no_variants(coord: Coordinator) -> None:
    assert coord.conversation._priors_match_advisory_block() == ""


# -- _harvest_specialist_findings -----------------------------------------------


@pytest.mark.asyncio
async def test_harvest_specialist_findings_does_not_persist_llm_competitor_target(coord: Coordinator) -> None:
    """LLM-authored competitor numbers must never be persisted as a consumable competitor target."""
    from hyperloom.inference_optimizer.session import session_paths
    from hyperloom.inference_optimizer.baseline_comparison import research_hints

    await coord.specialist_dispatch._harvest_specialist_findings(
        {
            "new_findings": [{"what": "try mtp", "source": "https://pr/1"}],
            "competitor_target": {
                "gpu": "b200",
                "model": "m",
                "per_conc": [
                    {"conc": 64, "tput_per_gpu": 999999.0, "source": "some blog"},
                ],
            },
        }
    )

    assert not session_paths.competitor_target_json(coord.session_dir).exists()
    assert research_hints.load_competitor_target(coord.session_dir) is None


# -- _handle_escalate_strategy_change --------------------------------------
def _escalate(hint: str) -> Intent:
    return Intent(
        type=IntentType.ESCALATE_STRATEGY_CHANGE,
        payload={"summary": "s", "next_action_hint": hint},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("hint", ["bogus", "skip_to_close"])
async def test_escalate_hint_outside_the_vocab_is_denied(coord: Coordinator, hint: str) -> None:
    await coord.router._handle_escalate_strategy_change("orchestration", _escalate(hint))
    assert coord.shared_state.pending_escalate_hint == ""
    assert coord.shared_state.policy_denial_history[-1]["rule"] == "escalate_hint_not_in_vocab"


@pytest.mark.asyncio
async def test_escalate_cannot_lengthen_a_phase_budget(coord: Coordinator) -> None:
    """Phase budgets are operator configuration; no escalate hint raises one."""
    before = dict(coord.shared_state.phase_budget_pct)

    await coord.router._handle_escalate_strategy_change("orchestration", _escalate("extend_explore_budget"))

    assert coord.shared_state.phase_budget_pct == before
    assert coord.shared_state.pending_escalate_hint == ""


@pytest.mark.asyncio
async def test_escalate_skip_to_kernel_deferred(coord: Coordinator) -> None:
    coord.shared_state.phase = "FRAMEWORK_AGENT"
    await coord.router._handle_escalate_strategy_change(
        "orchestration",
        _escalate("skip_to_kernel"),
    )
    assert coord.shared_state.pending_escalate_hint == "skip_to_kernel"


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["PRELUDE", "ENABLEMENT", "KERNEL_AGENT", "SWEEP"])
async def test_escalate_skip_to_kernel_outside_framework_agent_is_dropped(coord: Coordinator, phase: str) -> None:
    """Only the FRAMEWORK_AGENT exit reads skip_to_kernel; left pending elsewhere it would end that phase on arrival."""
    coord.shared_state.phase = phase
    await coord.router._handle_escalate_strategy_change(
        "orchestration",
        _escalate("skip_to_kernel"),
    )
    assert coord.shared_state.pending_escalate_hint == ""


# -- maybe_autosubmit_specialist_patches ----------------------------------
@pytest.mark.asyncio
async def test_autosubmit_skipped_when_no_patches(coord: Coordinator) -> None:
    from hyperloom.orchestrator.state.task_registry import Task

    task = Task(task_id="spec-1", kind="specialist", state="running", params={}, idempotency_key="k1")
    await coord.phase_framework.maybe_autosubmit_specialist_patches(
        task=task,
        done_payload={"patches_written": []},
    )


@pytest.mark.asyncio
async def test_autosubmit_skipped_when_files_missing(coord: Coordinator) -> None:
    from hyperloom.orchestrator.state.task_registry import Task

    task = Task(task_id="spec-2", kind="specialist", state="running", params={}, idempotency_key="k2")
    await coord.phase_framework.maybe_autosubmit_specialist_patches(
        task=task,
        done_payload={"patches_written": ["ghost.py"]},
    )


@pytest.mark.asyncio
async def test_autosubmit_creates_proposal_for_real_file(coord: Coordinator) -> None:
    from hyperloom.orchestrator.state.task_registry import Task
    from hyperloom.inference_optimizer.session.session_paths import runs_dir

    sid = "spec-3"
    spec_root = runs_dir(coord.session_dir, "specialist", sid)
    wt = spec_root / "worktree"
    wt.mkdir(parents=True, exist_ok=True)
    (wt / "kernel.py").write_text("# patched\n", encoding="utf-8")
    coord.shared_state.phase = "KERNEL_AGENT"
    task = Task(
        task_id=sid,
        kind="specialist",
        state="running",
        params={
            "domain": "serving_specialist",
            "gap_canonical_id": "gap.framework.fp8",
            "gap_layer": "framework",
            "framework": "other-framework",
            "framework_agent_authoring": True,
        },
        idempotency_key="k3",
    )
    n_before = len(coord.state.pending_proposals)
    await coord.phase_framework.maybe_autosubmit_specialist_patches(
        task=task,
        done_payload={
            "patches_written": ["kernel.py"],
            "proposal_set": [{"name": "fuse-moe"}],
        },
    )
    assert len(coord.state.pending_proposals) == n_before + 1
    pending = list(coord.state.pending_proposals.values())[-1]
    params = pending.payload["params"]
    assert params["source_phase"] == "FRAMEWORK_AGENT"
    assert params["domain"] == "serving_specialist"
    assert params["provenance"] == "specialist:serving_specialist"
    assert params["gap_canonical_id"] == "gap.framework.fp8"
    assert "framework" not in params


@pytest.mark.asyncio
async def test_autosubmit_creates_proposal_for_artifacts_only(coord: Coordinator) -> None:
    """A specialist with NO source patch but a non-diff tuned artifact (``artifacts_written`` with a real file in its worktree) is a routable deliverable: autosubmit must create an integrate_patch proposal so the artifact-install channel runs."""
    from hyperloom.orchestrator.state.task_registry import Task
    from hyperloom.inference_optimizer.session.session_paths import runs_dir

    sid = "spec-art-route"
    art_dir = runs_dir(coord.session_dir, "specialist", sid) / "worktree" / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)
    (art_dir / "tuned_fmoe.csv").write_text("cu_num,token\n304,16\n", encoding="utf-8")
    task = Task(task_id=sid, kind="specialist", state="running", params={}, idempotency_key="ka1")
    n_before = len(coord.state.pending_proposals)
    await coord.phase_framework.maybe_autosubmit_specialist_patches(
        task=task,
        done_payload={
            "patches_written": [],
            "proposal_set": [],
            "artifacts_written": [
                {
                    "source": "artifacts/tuned_fmoe.csv",
                    "target": "configs/model_configs/qwen3_tuned_fmoe.csv",
                    "kind": "aiter_tuned_fmoe_csv",
                }
            ],
        },
    )
    assert len(coord.state.pending_proposals) == n_before + 1


@pytest.mark.asyncio
async def test_autosubmit_skipped_when_artifact_source_outside_sandbox(coord: Coordinator, tmp_path) -> None:
    """An ``artifacts_written`` entry whose ``source`` is an ABSOLUTE path OUTSIDE the specialist sandbox must NOT be routable: integrate_patch would reject it as ``source_outside_workspace``, so autosubmit must not create a proposal for it."""
    from hyperloom.orchestrator.state.task_registry import Task

    outside = tmp_path / "outside.csv"
    outside.write_text("x", encoding="utf-8")
    task = Task(
        task_id="spec-art-outside",
        kind="specialist",
        state="running",
        params={},
        idempotency_key="ka2",
    )
    n_before = len(coord.state.pending_proposals)
    await coord.phase_framework.maybe_autosubmit_specialist_patches(
        task=task,
        done_payload={
            "patches_written": [],
            "proposal_set": [],
            "artifacts_written": [
                {
                    "source": str(outside),
                    "target": "configs/model_configs/x.csv",
                    "kind": "k",
                }
            ],
        },
    )
    assert len(coord.state.pending_proposals) == n_before


@pytest.mark.asyncio
async def test_autosubmit_skipped_when_artifact_source_relative_escapes_sandbox(
    coord: Coordinator,
) -> None:
    """A RELATIVE artifact ``source`` that escapes the specialist sandbox via ``..`` must NOT be routable, even though it resolves to a real file: integrate_patch rejects it as ``source_outside_workspace``, so autosubmit must not route it."""
    import os

    from hyperloom.orchestrator.state.task_registry import Task
    from hyperloom.inference_optimizer.session.session_paths import runs_dir

    sid = "spec-art-escape"
    worktree = runs_dir(coord.session_dir, "specialist", sid) / "worktree"
    worktree.mkdir(parents=True, exist_ok=True)
    outside = coord.session_dir / "escape.csv"
    outside.write_text("x", encoding="utf-8")
    rel_escape = os.path.relpath(outside, worktree)
    task = Task(task_id=sid, kind="specialist", state="running", params={}, idempotency_key="ka3")
    n_before = len(coord.state.pending_proposals)
    await coord.phase_framework.maybe_autosubmit_specialist_patches(
        task=task,
        done_payload={
            "patches_written": [],
            "proposal_set": [],
            "artifacts_written": [{"source": rel_escape, "target": "configs/model_configs/x.csv", "kind": "k"}],
        },
    )
    assert len(coord.state.pending_proposals) == n_before


@pytest.mark.asyncio
async def test_autosubmit_routes_relative_source_in_workspace_parent(coord: Coordinator) -> None:
    """A relative artifact ``source`` that climbs out of ``worktree`` via ``..`` but lands INSIDE the workspace is still contained, so it MUST remain routable: the sandbox check must not reject a legitimate ``../file`` source that resolves within an allowed base."""
    from hyperloom.orchestrator.state.task_registry import Task
    from hyperloom.inference_optimizer.session.session_paths import runs_dir

    sid = "spec-art-parent"
    spec_root = runs_dir(coord.session_dir, "specialist", sid)
    (spec_root / "worktree").mkdir(parents=True, exist_ok=True)
    (spec_root / "tuned.csv").write_text("cu_num\n304\n", encoding="utf-8")
    task = Task(task_id=sid, kind="specialist", state="running", params={}, idempotency_key="ka4")
    n_before = len(coord.state.pending_proposals)
    await coord.phase_framework.maybe_autosubmit_specialist_patches(
        task=task,
        done_payload={
            "patches_written": [],
            "proposal_set": [],
            "artifacts_written": [{"source": "../tuned.csv", "target": "configs/model_configs/x.csv", "kind": "k"}],
        },
    )
    assert len(coord.state.pending_proposals) == n_before + 1


# -- _record_fact_per_task -------------------------------------------------
def test_record_fact_per_task_keep_and_revert(coord: Coordinator) -> None:
    from hyperloom.orchestrator.state.task_registry import Task

    task = Task(task_id="t-fact", kind="explore", state="succeeded", params={}, idempotency_key="kf")
    coord.recipe_journal._record_fact_per_task(
        task=task,
        source_session_id="sess-a",
        result_dict={"gain_pct": 5.0, "output_throughput": 900.0},
        verdict=Verdict.ADOPTED,
    )
    coord.recipe_journal._record_fact_per_task(
        task=task,
        source_session_id="sess-a",
        result_dict={"error_class": "boom", "reason": "bad"},
        verdict=Verdict.FAILED,
    )
    outcomes = [entry.outcome for entry in coord.recipe_journal.ensure_journal().entries[-2:]]
    assert outcomes == ["KEEP", "no_promote"]


def test_record_fact_reverted_integrate_patch_journals_revert(coord: Coordinator) -> None:
    """A reverted integrate_patch settles REVERTED and journals REVERT with the REAL measured delta (from delta_pct)."""
    from hyperloom.inference_optimizer.session.optimization_journal import (
        OUTCOME_REVERT,
    )
    from hyperloom.orchestrator.state.task_registry import Task

    task = Task(
        task_id="t-revert-fake-keep",
        kind="integrate_patch",
        state="succeeded",
        params={},
        idempotency_key="t-revert-fake-keep",
    )
    coord.recipe_journal._record_fact_per_task(
        task=task,
        source_session_id="sess-a",
        # tput == baseline → delta_pct ~0, executor returns "reverted", promotable.
        result_dict={
            "status": "reverted",
            "delta_pct": -0.44,
            "output_throughput": 0.440529,
            "reason": "throughput delta -0.44% < keep_threshold 1.00%",
        },
        verdict=Verdict.REVERTED,
    )
    entry = coord.recipe_journal.ensure_journal().entries[-1]
    assert entry.outcome == OUTCOME_REVERT
    assert entry.gain_pct == -0.44
    assert entry.reason and "keep_threshold" in entry.reason


def test_record_fact_kept_integrate_patch_journals_keep(coord: Coordinator) -> None:
    from hyperloom.inference_optimizer.session.optimization_journal import OUTCOME_KEEP
    from hyperloom.orchestrator.state.task_registry import Task

    task = Task(
        task_id="t-real-keep",
        kind="integrate_patch",
        state="succeeded",
        params={},
        idempotency_key="t-real-keep",
    )
    coord.recipe_journal._record_fact_per_task(
        task=task,
        source_session_id="sess-a",
        result_dict={"status": "kept", "delta_pct": 6.2, "output_throughput": 1100.0},
        verdict=Verdict.ADOPTED,
    )
    entry = coord.recipe_journal.ensure_journal().entries[-1]
    assert entry.outcome == OUTCOME_KEEP
    assert entry.gain_pct == 6.2


def test_is_promotable_result_unchanged_for_reverted_integrate_patch(coord: Coordinator) -> None:
    """A reverted integrate_patch stays promotable so it still runs the pending_integrate cleanup in promote_to_shared_state."""
    assert coord.writeback.is_promotable_result("integrate_patch", {"status": "reverted"}) is True
    assert coord.writeback.is_promotable_result("integrate_patch", {"status": "failed"}) is False


# -- compose_prompt additional branches -----------------------------------
@pytest.mark.asyncio
async def test_compose_prompt_renders_the_gap_to_the_gain_objective(coord: Coordinator) -> None:
    coord._current_objective = TargetGainObjective(target_gain_pct=20.0)
    coord.shared_state.cumulative_gain_validated = 5.0
    text = (await coord.conversation.compose_prompt("orchestration")).text
    assert "target_gap_pct=15.00" in text


@pytest.mark.asyncio
async def test_compose_prompt_time_only_objective_leaves_no_gap(coord: Coordinator) -> None:
    coord._current_objective = TimeOnlyObjective()
    coord.shared_state.cumulative_gain_validated = 5.0
    text = (await coord.conversation.compose_prompt("orchestration")).text
    assert "target_gap_pct=0.00" in text


# -- _context_analysis_reader ----------------------------------------------
def test_context_analysis_reader(coord: Coordinator) -> None:
    out = coord.conversation._context_analysis_reader()
    assert isinstance(out, str)


def test_context_analysis_reader_fallback_path(coord: Coordinator, tmp_path) -> None:
    md = tmp_path / "analysis.md"
    md.write_text("# roofline\n", encoding="utf-8")
    coord.shared_state.last_trace_analyze = {"analysis_md_path": str(md)}
    coord.shared_state.analysis_md = ""
    out = coord.conversation._context_analysis_reader()
    assert isinstance(out, str)


# -- advisory blocks enabled paths -----------------------------------------
def test_target_gap_advisory_enabled(coord: Coordinator, monkeypatch) -> None:
    from hyperloom.inference_optimizer.baseline_comparison import research_hints as rh

    monkeypatch.setattr(rh, "load_competitor_target", lambda _sd: {"name": "comp"})
    monkeypatch.setattr(rh, "gap_analysis", lambda *a, **k: {"primary_gap": "throughput"})
    monkeypatch.setattr(rh, "full_gap_summary", lambda g: "GAP-SUMMARY")
    coord.shared_state.target_advisory_enabled = True
    coord.shared_state.current_best = {"tput": 1000.0, "tpot_mean_ms": 5.0}
    coord.shared_state.tp = 1
    coord.shared_state.conc = 64
    assert coord.conversation.target_gap_advisory_block() == "GAP-SUMMARY"
    assert coord.conversation._current_primary_gap() == "throughput"


def test_agentx_advisory_and_primary_gap_share_accepted_state(coord: Coordinator, monkeypatch) -> None:
    from hyperloom.inference_optimizer.baseline_comparison import research_hints as rh

    target = {
        "benchmark_mode": "agentx",
        "throughput_basis": "total_token_throughput_per_gpu",
        "per_conc": [
            {"conc": 4, "tput_per_gpu": 800.0, "e2e_norm_intvty_p90": 20.0, "benchmark_id": "2", "source": "api"}
        ],
    }
    monkeypatch.setattr(rh, "load_competitor_target", lambda _sd: target)
    state = coord.shared_state
    state.benchmark_mode = "agentx"
    state.current_best = {"total_throughput": 800.0, "e2e_norm_intvty_p90": 5.0}
    state.tp = 2
    state.conc = 4
    text = coord.conversation.target_gap_advisory_block()
    assert "total throughput/GPU gap vs target: +50.0%" in text
    assert "E2E normalized interactivity P90 gap vs target: +75.0%" in text
    assert coord.conversation._current_primary_gap() == "latency"


def test_target_gap_advisory_no_target(coord: Coordinator, monkeypatch) -> None:
    from hyperloom.inference_optimizer.baseline_comparison import research_hints as rh

    monkeypatch.setattr(rh, "load_competitor_target", lambda _sd: None)
    coord.shared_state.target_advisory_enabled = True
    assert coord.conversation.target_gap_advisory_block() == ""
    assert coord.conversation._current_primary_gap() is None


# -- promote_warm_replay --------------------------------------------------
def test_promote_warm_replay_non_dict(coord: Coordinator) -> None:
    coord.phase_prelude.promote_warm_replay("nope")  # type: ignore[arg-type]
    assert coord.shared_state.warm_replay_outcome["status"] == "failed"


def test_promote_warm_replay_failed_status(coord: Coordinator) -> None:
    coord.phase_prelude.promote_warm_replay({"status": "failed", "error_class": "x", "error": "boom"})
    assert coord.shared_state.warm_replay_outcome["status"] == "failed"


def test_promote_warm_replay_invalid_tput(coord: Coordinator) -> None:
    coord.shared_state.baseline_tput = 0.0
    coord.phase_prelude.promote_warm_replay({"status": "succeeded", "output_throughput": 0.0})
    assert coord.shared_state.warm_replay_outcome["status"] == "failed"


def test_promote_warm_replay_reproduced_no_params(coord: Coordinator) -> None:
    coord.shared_state.baseline_tput = 800.0
    coord.phase_prelude.promote_warm_replay(
        {"status": "succeeded", "output_throughput": 900.0},
        task=None,
    )
    assert coord.shared_state.warm_replay_outcome["status"] == "reproduced_but_no_params"


def test_promote_warm_replay_reproduced_with_params(coord: Coordinator) -> None:
    from hyperloom.orchestrator.state.task_registry import Task

    coord.shared_state.baseline_tput = 800.0
    task = Task(
        task_id="warm-1",
        kind="replay_warm_recipe",
        state="running",
        params={"extra_envs": {"A": "1"}, "baseline_tput_anchor": 800.0},
        idempotency_key="kw",
    )
    coord.phase_prelude.promote_warm_replay(
        {"status": "succeeded", "output_throughput": 900.0},
        task=task,
    )
    assert coord.shared_state.warm_replay_outcome["status"] == "reproduced"


# -- maybe_auto_retry_specialist ------------------------------------------
def _spec_task(**params):
    from hyperloom.orchestrator.state.task_registry import Task

    return Task(task_id="spec-r", kind="specialist", state="running", params=params, idempotency_key="spec-r-key")


def _result(state="failed", result=None, error=None):
    from hyperloom.orchestrator.loop.sub_agent_runner import SubAgentResult

    return SubAgentResult(task_id="spec-r", state=state, result=result or {}, error=error)


@pytest.mark.asyncio
async def test_auto_retry_disabled_by_env(coord: Coordinator, monkeypatch) -> None:
    monkeypatch.setenv("INFERENCE_OPTIMIZER_SPECIALIST_AUTO_RETRY", "0")
    assert await coord.specialist_dispatch.maybe_auto_retry_specialist(_spec_task(), _result()) is False


@pytest.mark.asyncio
async def test_auto_retry_not_eligible(coord: Coordinator, monkeypatch) -> None:
    monkeypatch.setenv("INFERENCE_OPTIMIZER_SPECIALIST_AUTO_RETRY", "1")
    res = _result(result={"runner_status": "empty_synthesised"}, error="no_output")
    assert await coord.specialist_dispatch.maybe_auto_retry_specialist(_spec_task(), res) is False


@pytest.mark.asyncio
async def test_auto_retry_schedules_on_transient_failure(coord: Coordinator, monkeypatch) -> None:
    monkeypatch.setenv("INFERENCE_OPTIMIZER_SPECIALIST_AUTO_RETRY", "1")
    res = _result(result={"runner_status": "stale"}, error="timeout waiting")
    scheduled = await coord.specialist_dispatch.maybe_auto_retry_specialist(_spec_task(), res)
    assert scheduled is True


@pytest.mark.asyncio
async def test_auto_retry_caps_attempts(coord: Coordinator, monkeypatch) -> None:
    monkeypatch.setenv("INFERENCE_OPTIMIZER_SPECIALIST_AUTO_RETRY", "1")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_SPECIALIST_AUTO_RETRY_MAX", "1")
    res = _result(result={"runner_status": "stale"}, error="timeout")
    task = _spec_task(_auto_retry_attempt=1)
    assert await coord.specialist_dispatch.maybe_auto_retry_specialist(task, res) is False


# -- fan_out_specialist_wave (invalid entries) ---------------------------
@pytest.mark.asyncio
async def test_fan_out_wave_rejects_invalid_entries(coord: Coordinator, monkeypatch) -> None:
    called = []
    monkeypatch.setattr(
        coord.router,
        "handle_delegate",
        lambda *a, **k: called.append(a),
    )
    intent = Intent(type=IntentType.DELEGATE, payload={"idempotency_key": "w", "action_name": "specialist"})
    from hyperloom.orchestrator.policy.gate import PolicyDenied

    with pytest.raises(PolicyDenied):
        await coord.specialist_dispatch.fan_out_specialist_wave(
            "orchestration",
            intent,
            {"tasks": ["not-a-dict", {}, {"task_description": "   "}]},
        )
    assert called == []


# -- warm_specialist_params -----------------------------------------------
@pytest.mark.asyncio
async def test_warm_specialist_params_always_writes_the_unconditional_keys(coord: Coordinator) -> None:
    """Three keys are written whatever the session holds, so a specialist can read them without a presence test."""
    coord.shared_state.gpu_type = "mi300x"
    params: dict = {"domain": "kernel_agent"}

    await coord.specialist_dispatch.warm_specialist_params(params)

    assert params["gpu_type"] == "mi300x"
    assert params["kb_subgraph"] == {}
    assert "keep_threshold_pct" in params

    empty: dict = {"domain": "kernel_agent"}
    coord.shared_state.gpu_type = ""

    await coord.specialist_dispatch.warm_specialist_params(empty)

    assert empty["gpu_type"] == ""


#: ``(state field, param key, value to set, value expected on the params)`` for every field whose value
#: SharedState hands a specialist. ``grading`` is the one whose key is not its own name.
_STATE_EXPORTS = (
    ("warm_start_recipe", "warm_start_recipe", {"backend": "aiter"}, {"backend": "aiter"}),
    ("warm_start_pitfalls", "warm_start_pitfalls", ["oom"], ["oom"]),
    ("warm_start_lessons", "warm_start_lessons", ["raise tp"], ["raise tp"]),
    ("framework", "framework", "  sglang  ", "sglang"),
    ("tp", "tp", 8, 8),
    ("precision", "precision", "fp8", "fp8"),
    ("conc", "conc", 64, 64),
    ("isl", "isl", 1024, 1024),
    ("osl", "osl", 128, 128),
    ("max_model_len", "max_model_len", 8192, 8192),
    ("benchmark_mode", "benchmark_mode", "agentx", "agentx"),
    ("agentx_corpus_shape", "agentx_corpus_shape", {"isl": 1024}, {"isl": 1024}),
    ("grading", "agentx_grading", {"objective": "intvty_p90"}, {"objective": "intvty_p90"}),
    ("agentx_backend", "agentx_backend", "vllm", "vllm"),
    ("baseline_tput", "baseline_tput", 900.0, 900.0),
    ("cumulative_gain_validated", "cumulative_gain_validated", -3.5, -3.5),
)


@pytest.mark.asyncio
async def test_warm_specialist_params_exports_every_state_field_it_carries(coord: Coordinator) -> None:
    for field, _key, value, _expected in _STATE_EXPORTS:
        setattr(coord.shared_state, field, value)
    params: dict = {"domain": "kernel_agent"}

    await coord.specialist_dispatch.warm_specialist_params(params)

    assert {key: params.get(key) for _f, key, _v, _e in _STATE_EXPORTS} == {
        key: expected for _f, key, _v, expected in _STATE_EXPORTS
    }


@pytest.mark.asyncio
async def test_warm_specialist_params_omits_a_field_state_left_empty(coord: Coordinator) -> None:
    """An absent key is how a specialist is told the session has no value, so an empty one must not be exported."""
    params: dict = {"domain": "kernel_agent"}

    await coord.specialist_dispatch.warm_specialist_params(params)

    assert [key for _f, key, _v, _e in _STATE_EXPORTS if key in params] == []


@pytest.mark.asyncio
async def test_warm_specialist_params_never_overwrites_a_key_the_caller_set(coord: Coordinator) -> None:
    for field, _key, value, _expected in _STATE_EXPORTS:
        setattr(coord.shared_state, field, value)
    params: dict = {"domain": "kernel_agent"}
    params.update({key: "caller" for _f, key, _v, _e in _STATE_EXPORTS})

    await coord.specialist_dispatch.warm_specialist_params(params)

    assert {key: params[key] for _f, key, _v, _e in _STATE_EXPORTS} == {
        key: "caller" for _f, key, _v, _e in _STATE_EXPORTS
    }


# -- finalize_recipe_and_journal ------------------------------------
def test_recipe_kb_finalize_recipe_and_journal_no_kb(coord: Coordinator) -> None:
    coord.shared_state.current_best = {"tput": 950.0}
    coord.shared_state.cumulative_gain_validated = 12.5
    coord.recipe_journal.finalize_recipe_and_journal()


# -- _record_fact_per_variant ----------------------------------------------
def test_record_fact_per_variant_keep_revert_skip(coord: Coordinator) -> None:
    from hyperloom.orchestrator.state.task_registry import Task

    task = Task(task_id="t-var", kind="explore", state="succeeded", params={}, idempotency_key="kv")
    # SKIPPED_DEDUP -> early return (no journal row)
    coord.recipe_journal._record_fact_per_variant(
        task=task,
        source_session_id="s",
        variant_outcome={"outcome": "SKIPPED_DEDUP", "variant_name": "v0"},
        adopted=False,
    )
    coord.recipe_journal._record_fact_per_variant(
        task=task,
        source_session_id="s",
        variant_outcome={
            "outcome": "KEEP",
            "variant_name": "v1",
            "metrics": {"gain_pct": 4.0, "output_throughput": 900.0},
            "variant": {"name": "v1"},
        },
        adopted=True,
    )
    coord.recipe_journal._record_fact_per_variant(
        task=task,
        source_session_id="s",
        variant_outcome={
            "outcome": "REVERT",
            "variant_name": "v2",
            "error_class": "regressed",
            "reason": "slower",
            "metrics": {},
        },
        adopted=False,
    )
    coord.recipe_journal._record_fact_per_variant(
        task=task,
        source_session_id="s",
        variant_outcome={
            "outcome": "KEEP",
            "variant_name": "v3",
            "metrics": {"gain_pct": 2.0, "output_throughput": 880.0},
            "variant": {"name": "v3"},
        },
        adopted=False,
    )
    by_name = {entry.variant_name: entry.outcome for entry in coord.recipe_journal.ensure_journal().entries}
    # An executor KEEP whose lift did not land adopted nothing.
    assert by_name == {"v1": "KEEP", "v2": "REVERT", "v3": "no_promote"}


def _nonzero_result(output: str) -> dict:
    """A ``subprocess_nonzero`` baseline result shaped the way the baseline executor builds one."""
    from hyperloom.orchestrator.actions.executors.baseline import nonzero_output_signature

    return {
        "status": "failed",
        "error_class": "subprocess_nonzero",
        "error": output[-2000:],
        "output_signature": nonzero_output_signature("subprocess_nonzero", "", output),
    }


def _baseline_task(task_id: str) -> Task:
    return Task(
        task_id=task_id,
        kind="baseline",
        state="running",
        params={"config_path": "baseline.yaml"},
        idempotency_key=task_id,
    )


@pytest.mark.asyncio
async def test_unpromotable_baseline_same_failure_twice_stops_at_two(coord: Coordinator) -> None:
    """Two subprocess_nonzero baselines that printed the same thing (modulo numbers) end the run at two."""
    coord.shared_state.enablement_mode = "off"
    first = _nonzero_result("RuntimeError: worker 1234 died at t=17")
    second = _nonzero_result("RuntimeError: worker 98 died at t=4")

    await coord.writeback.handle_unpromotable_result(_baseline_task("bl-same-1"), first)
    assert coord.shared_state.stop_reason != "baseline_failed"
    await coord.writeback.handle_unpromotable_result(_baseline_task("bl-same-2"), second)
    assert coord.shared_state.baseline_failure_streak == 2
    assert coord.shared_state.stop_reason == "baseline_failed"


@pytest.mark.asyncio
async def test_unpromotable_baseline_different_failures_keep_three_strikes(coord: Coordinator) -> None:
    """Two different failures are not evidence of a third: the three-strike budget still applies."""
    coord.shared_state.enablement_mode = "off"
    a = _nonzero_result("RuntimeError: HIP out of memory")
    b = _nonzero_result("ConnectionError: port refused")

    await coord.writeback.handle_unpromotable_result(_baseline_task("bl-diff-1"), a)
    await coord.writeback.handle_unpromotable_result(_baseline_task("bl-diff-2"), b)
    assert coord.shared_state.baseline_failure_streak == 2
    assert coord.shared_state.stop_reason != "baseline_failed"
    await coord.writeback.handle_unpromotable_result(_baseline_task("bl-diff-3"), a)
    assert coord.shared_state.stop_reason == "baseline_failed"


@pytest.mark.asyncio
async def test_unpromotable_baseline_repeat_of_another_class_keeps_three_strikes(coord: Coordinator) -> None:
    """Only subprocess_nonzero carries the process's own output; a fixed sentence repeating proves nothing."""
    result = {"status": "failed", "error_class": "no_report", "error": "benchmark_report.json missing under /x"}

    await coord.writeback.handle_unpromotable_result(_baseline_task("bl-nr-1"), result)
    await coord.writeback.handle_unpromotable_result(_baseline_task("bl-nr-2"), result)
    assert coord.shared_state.baseline_failure_streak == 2
    assert coord.shared_state.stop_reason != "baseline_failed"


@pytest.mark.asyncio
async def test_unpromotable_baseline_infeasible_ray_cluster_stops_on_the_first(coord: Coordinator) -> None:
    """A Ray cluster that cannot place the round fails the same way every time, and no patch changes that."""
    from hyperloom.orchestrator.actions.executors._ray_serving import RAY_INFEASIBLE_MARKER

    result = {
        "status": "failed",
        "error_class": "subprocess_nonzero",
        "error": f"ray_ensure_error: {RAY_INFEASIBLE_MARKER}: existing Ray head has no serving_slot resource",
    }

    await coord.writeback.handle_unpromotable_result(_baseline_task("bl-ray-1"), result)

    assert coord.shared_state.stop_reason == "baseline_failed"
    # Not handed to the enablement lane as something to author a patch against.
    assert not (coord.shared_state.enablement.launch_log or "").strip()


@pytest.mark.asyncio
async def test_unpromotable_baseline_infeasible_ray_cluster_stops_in_enablement_too(coord: Coordinator) -> None:
    from hyperloom.orchestrator.actions.executors._ray_serving import RAY_INFEASIBLE_MARKER
    from hyperloom.orchestrator.phases.machine_state import PHASE_ENABLEMENT

    coord.shared_state.phase = PHASE_ENABLEMENT
    result = {
        "status": "failed",
        "error_class": "subprocess_nonzero",
        "error": f"ray_ensure_error: {RAY_INFEASIBLE_MARKER}: cluster has 0.0 GPU(s), 1 requested",
    }

    await coord.writeback.handle_unpromotable_result(_baseline_task("bl-ray-en"), result)

    assert coord.shared_state.stop_reason == "baseline_failed"


@pytest.mark.asyncio
async def test_unpromotable_baseline_same_failure_in_enablement_does_not_stop_early(coord: Coordinator) -> None:
    """An ordinary repeated boot failure is enablement's input; the early stop does not pre-empt the lane."""
    from hyperloom.orchestrator.phases.machine_state import PHASE_ENABLEMENT

    coord.shared_state.phase = PHASE_ENABLEMENT
    result = _nonzero_result("ImportError: no module named foo")

    await coord.writeback.handle_unpromotable_result(_baseline_task("bl-en-1"), result)
    await coord.writeback.handle_unpromotable_result(_baseline_task("bl-en-2"), result)
    assert coord.shared_state.stop_reason != "baseline_failed"


@pytest.mark.asyncio
async def test_unpromotable_baseline_same_failure_in_prelude_with_enablement_admitted_does_not_stop_early(
    coord: Coordinator,
) -> None:
    """A retry that settles before the PRELUDE->ENABLEMENT transition is still enablement's input."""
    from hyperloom.orchestrator.phases.machine_state import PHASE_PRELUDE

    coord.shared_state.phase = PHASE_PRELUDE
    result = _nonzero_result("ImportError: no module named foo")

    await coord.writeback.handle_unpromotable_result(_baseline_task("bl-pre-1"), result)
    await coord.writeback.handle_unpromotable_result(_baseline_task("bl-pre-2"), result)
    assert coord.shared_state.stop_reason != "baseline_failed"


_SHARED_TAIL = (
    "Traceback (most recent call last):\n"
    + "  File engine.py, in run_engine_core\n    engine.step()\n" * 60
    + "RuntimeError: engine core died unexpectedly\n"
)


async def _executor_nonzero_result(
    tmp_path, monkeypatch, name: str, stderr: str, *, workspace: bool, stdout: str = ""
) -> dict:
    """Run one baseline round through the executor's real result builder, with only Magpie's process faked."""
    import types

    from hyperloom.orchestrator.actions.executors import baseline as bl
    from hyperloom.orchestrator.loop.sub_agent_runner import RunnerContext

    session = tmp_path / "session"
    session.mkdir(exist_ok=True)
    out = tmp_path / name
    out.mkdir()
    run_workspace = out / "benchmark_vllm_1" if workspace else None
    if run_workspace is not None:
        run_workspace.mkdir()
    monkeypatch.setattr(bl, "harvest_leaked_artifacts", lambda *a, **k: [])
    monkeypatch.setattr(bl, "snapshot_workspaces", lambda *a, **k: set())
    monkeypatch.setattr(bl, "select_run_workspace", lambda *a, **k: run_workspace)
    monkeypatch.setattr(
        bl, "run_with_session_kill", lambda cmd, **kw: types.SimpleNamespace(returncode=1, stdout=stdout, stderr=stderr)
    )
    cfg = session / "bench.yaml"
    cfg.write_text("benchmark:\n  framework: vllm\n  model: /tmp/m\n", encoding="utf-8")
    ex = bl.BaselineExecutor(session_dir=session, magpie_python="/usr/bin/python3")
    return await ex._run_single_benchmark(
        bl.BenchmarkAttempt(
            timeout_sec=30,
            override_result_dir=None,
            resolved_model="/tmp/m",
            materialized_config_path=cfg,
            inferencex_path="",
            effective_extra_server_args="",
            params={"framework": "vllm"},
            ctx=RunnerContext(task=_baseline_task(name), lease=None, extra={}),
        ),
        config_path=cfg,
        output_dir=out,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("workspace", [False, True], ids=["no_workspace", "workspace_without_report"])
async def test_baseline_executor_signs_the_whole_output_not_the_error_tail(
    coord: Coordinator, tmp_path, monkeypatch, workspace: bool
) -> None:
    """Two causes that end in the same 2000+ characters share an ``error`` but not a signature, so they keep three strikes."""
    oom = await _executor_nonzero_result(
        tmp_path, monkeypatch, "oom", "HIP out of memory on device 3\n" + _SHARED_TAIL, workspace=workspace
    )
    nccl = await _executor_nonzero_result(
        tmp_path, monkeypatch, "nccl", "NCCL watchdog timeout after 600s\n" + _SHARED_TAIL, workspace=workspace
    )

    assert oom["error_class"] == nccl["error_class"] == "subprocess_nonzero"
    # The executor's own ``error`` is the 2000-character tail, identical for both causes.
    assert oom["error"] == nccl["error"]
    assert oom["output_signature"] and nccl["output_signature"]
    assert oom["output_signature"] != nccl["output_signature"]

    await coord.writeback.handle_unpromotable_result(_baseline_task("bl-tail-1"), oom)
    await coord.writeback.handle_unpromotable_result(_baseline_task("bl-tail-2"), nccl)
    assert coord.shared_state.baseline_failure_streak == 2
    assert coord.shared_state.stop_reason != "baseline_failed"


@pytest.mark.asyncio
async def test_baseline_executor_same_output_modulo_numbers_stops_at_two(
    coord: Coordinator, tmp_path, monkeypatch
) -> None:
    first = await _executor_nonzero_result(
        tmp_path, monkeypatch, "r1", "worker pid 4121 died at 0x7f3a2c\n" + _SHARED_TAIL, workspace=False
    )
    second = await _executor_nonzero_result(
        tmp_path, monkeypatch, "r2", "worker pid 977 died at 0x7f99e0\n" + _SHARED_TAIL, workspace=False
    )

    assert first["output_signature"] == second["output_signature"] != ""

    coord.shared_state.enablement_mode = "off"
    await coord.writeback.handle_unpromotable_result(_baseline_task("bl-same-1"), first)
    assert coord.shared_state.stop_reason != "baseline_failed"
    await coord.writeback.handle_unpromotable_result(_baseline_task("bl-same-2"), second)
    assert coord.shared_state.stop_reason == "baseline_failed"


@pytest.mark.asyncio
async def test_baseline_executor_signs_stdout_as_well_as_stderr(coord: Coordinator, tmp_path, monkeypatch) -> None:
    """The same generic stderr over different causes on stdout is two failures, not one."""
    stderr = "RuntimeError: engine died\n"
    oom = await _executor_nonzero_result(
        tmp_path, monkeypatch, "oom", stderr, stdout="HIP out of memory on device 3\n", workspace=False
    )
    nccl = await _executor_nonzero_result(
        tmp_path, monkeypatch, "nccl", stderr, stdout="NCCL watchdog timeout after 600s\n", workspace=False
    )

    assert oom["error"] == nccl["error"]
    assert oom["output_signature"] != nccl["output_signature"]

    await coord.writeback.handle_unpromotable_result(_baseline_task("bl-out-1"), oom)
    await coord.writeback.handle_unpromotable_result(_baseline_task("bl-out-2"), nccl)
    assert coord.shared_state.stop_reason != "baseline_failed"
