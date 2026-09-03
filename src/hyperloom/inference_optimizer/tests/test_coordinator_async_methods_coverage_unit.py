# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coverage for Coordinator async/stateful methods invoked directly against a
real (mock-backed) Coordinator: SharedState promotion across task kinds, prompt
composition per agent, advisory blocks, research-scout harvest, the
orchestration checkpoint guard, strategy-change escalation, specialist
autosubmit routing and warm-up, and per-task/per-variant fact journaling."""

from __future__ import annotations

import time

import pytest

from hyperloom.orchestrator.roles.mcp_context_tools import CONTEXT_TOOL_NAMES
from hyperloom.orchestrator.roles import (
    Backend,
    MockBackend,
    ScriptedPlan,
)
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.state.objective import TargetGainObjective, TimeOnlyObjective
from hyperloom.orchestrator.state.task_registry import Task
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType


def _heartbeat() -> Intent:
    return Intent(type=IntentType.SEND_MESSAGE, payload={"topic": "heartbeat", "body_md": "ok"})


def _silent_plan() -> ScriptedPlan:
    return ScriptedPlan(turns=[], default_intent=_heartbeat())


def _build_backends() -> dict[str, Backend]:
    return {name: MockBackend(_silent_plan(), name=name) for name in ("orchestration", "critic", "robustness")}


@pytest.fixture
def coord(session_dir) -> Coordinator:
    return Coordinator(session_dir, backends=_build_backends())


# -- _promote_to_shared_state ----------------------------------------------
@pytest.mark.asyncio
async def test_promote_baseline_sets_anchor_and_current_best(coord: Coordinator) -> None:
    coord.shared_state.auto_roofline_pending_task_id = "pending-x"
    coord.shared_state.baseline_failure_streak = 2
    coord.shared_state.baseline_arg_error_streak = 1
    await coord._promote_to_shared_state(
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
    assert coord.shared_state.baseline_arg_error_streak == 0
    assert coord.shared_state.current_best["action"] == "baseline"
    assert coord.shared_state.current_best["tput"] == 1000.0
    assert coord.shared_state.current_best["cold_tput"] == 900.0


@pytest.mark.asyncio
async def test_promote_single_round_baseline_clears_stale_warm_runtime(coord: Coordinator) -> None:
    coord.shared_state.auto_roofline_pending_task_id = "pending-x"
    coord.shared_state.baseline_warm_runtime_sec = 7.5

    await coord._promote_to_shared_state(
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
    """The two figures that let later work be priced on what it will spend.

    The whole round and the part of it that ran after the server was ready; the
    difference between them is what booting this workload costs, and every
    variant boots again.
    """
    await coord._promote_to_shared_state(
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

    await coord._promote_to_shared_state(
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
    """The marker drives a session-level decision, so it has to reach the session.

    PRELUDE routes to CLOSE on it rather than optimizing against a denominator
    that was never the baseline, and it is cleared by the next baseline that does
    land a hot figure -- otherwise a session resumed with a fresh clock stays
    condemned by the earlier leg's shortfall.
    """
    await coord._promote_to_shared_state(
        "baseline",
        {
            "output_throughput": 1000.0,
            "subprocess_runtime_sec": 900.0,
            "measure_round_dropped": {"reason": "measure_round_reaped_by_the_run"},
            "workspace": "/tmp/ws",
        },
    )
    assert coord.shared_state.baseline_measure_round_dropped is True

    await coord._promote_to_shared_state(
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
    """The escape from the marker, without which PRELUDE cannot finish.

    A cold anchor holds the phase open until a hot pass replaces it. The rule
    that keeps a later, lower re-baseline from displacing the anchor would reject
    that replacement whenever the cold figure reads higher -- which it does
    whenever the "cold" pass was not really cold, its weights already in page
    cache and its kernels already compiled by an earlier run. The session would
    then re-measure whole baseline rounds until the clock killed it, each one
    landing the very measurement that was supposed to release it.
    """

    @pytest.mark.asyncio
    async def test_a_lower_hot_figure_replaces_a_marked_cold_one(self, coord: Coordinator) -> None:
        coord.shared_state.baseline_tput = 1000.0
        coord.shared_state.baseline_measure_round_dropped = True

        await coord._promote_to_shared_state(
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
        """Only a hot pass corrects the anchor; another cold one is just noisier.

        Two cold figures are comparable to each other, so the ordinary rule
        applies and the better one stands. Nothing has been corrected, so the
        marker stays and the phase stays open.
        """
        coord.shared_state.baseline_tput = 1000.0
        coord.shared_state.baseline_measure_round_dropped = True

        await coord._promote_to_shared_state(
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

        await coord._promote_to_shared_state(
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
async def test_promote_baseline_non_dict_is_noop(coord: Coordinator) -> None:
    await coord._promote_to_shared_state("baseline", "not-a-dict")  # type: ignore[arg-type]


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

    await coord._handle_unpromotable_result(task, result)
    assert coord.shared_state.baseline_arg_error_streak == 1
    assert coord.shared_state.stop_reason != "baseline_arg_error"

    await coord._handle_unpromotable_result(task, result)
    assert coord.shared_state.baseline_arg_error_streak == 2
    assert coord.shared_state.baseline_failure_streak == 0
    assert coord.shared_state.stop_reason == "baseline_arg_error"


@pytest.mark.asyncio
async def test_unpromotable_baseline_mixed_classes_stop_after_three_total(
    coord: Coordinator,
) -> None:
    """Mixed subprocess_nonzero + fast_exit_arg_error failures must still
    fast-fail once 3 total baseline failures accrue, even though neither
    per-class streak reaches its own threshold."""

    def _task() -> Task:
        return Task(
            task_id="bl-mixed",
            kind="baseline",
            state="running",
            params={"config_path": "baseline.yaml"},
            idempotency_key="bl-mixed",
        )

    subproc = {"status": "failed", "error_class": "subprocess_nonzero", "error": "boom"}
    argerr = {"status": "failed", "error_class": "fast_exit_arg_error", "error": "bad arg"}

    await coord._handle_unpromotable_result(_task(), subproc)
    await coord._handle_unpromotable_result(_task(), argerr)
    assert coord.shared_state.stop_reason not in (
        "baseline_failed",
        "baseline_arg_error",
    )
    await coord._handle_unpromotable_result(_task(), subproc)
    assert coord.shared_state.baseline_failure_streak == 2
    assert coord.shared_state.baseline_total_failures == 3
    assert coord.shared_state.stop_reason == "baseline_failed"


@pytest.mark.asyncio
async def test_promote_profile_succeeded_records_trace(coord: Coordinator) -> None:
    coord.shared_state.baseline_tput = 800.0
    await coord._promote_to_shared_state(
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
    await coord._promote_to_shared_state(
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
    await coord._promote_to_shared_state(
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
    await coord._promote_to_shared_state("roofline", {"status": "succeeded"})
    await coord._promote_to_shared_state("roofline", {"status": "skipped"})
    await coord._promote_to_shared_state(
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
    await coord._promote_to_shared_state(
        "explore",
        {
            "winners": [{"name": "v0", "extra_server_args": "--tp 1", "tput": 900.0}],
            "best_variant": {"name": "v0", "extra_server_args": "--tp 1", "tput": 900.0},
            "output_throughput": 900.0,
            "round_id": "r1",
            "losers": [],
            "skipped_dup": [],
        },
    )
    assert coord.shared_state.current_best.get("tput") == 900.0


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_compose_prompt_orchestration_with_time_budget(coord: Coordinator) -> None:
    coord._run_started_monotonic = time.monotonic() - 60.0
    coord._run_deadline = time.monotonic() + 600.0
    coord.shared_state.max_minutes = 60
    out = await coord._compose_prompt("orchestration")
    assert "SESSION_DIR=" in out
    assert "Mission progress" in out
    assert "Time budget" in out


@pytest.mark.asyncio
async def test_compose_prompt_orchestration_deadline_imminent_warning(coord: Coordinator) -> None:
    coord._run_started_monotonic = time.monotonic() - 60.0
    coord._run_deadline = time.monotonic() + 60.0
    coord.shared_state.max_minutes = 60
    coord.shared_state.closing_phase = False
    out = await coord._compose_prompt("orchestration")
    assert "< 5 min remaining" in out


@pytest.mark.asyncio
async def test_compose_prompt_robustness_and_kernel(coord: Coordinator) -> None:
    coord._run_started_monotonic = time.monotonic() - 60.0
    coord._run_deadline = time.monotonic() + 600.0
    coord.shared_state.max_minutes = 60
    out_rob = await coord._compose_prompt("robustness")
    out_k = await coord._compose_prompt("kernel_agent")
    assert "SESSION_DIR=" in out_rob
    assert "SESSION_DIR=" in out_k


# -- advisory blocks -------------------------------------------------------
def test_advisory_blocks_disabled_return_empty(coord: Coordinator) -> None:
    coord.shared_state.target_advisory_enabled = False
    assert coord._target_gap_advisory_block() == ""
    assert coord._current_primary_gap() is None


def test_plateau_advisory_block_no_signal(coord: Coordinator) -> None:
    assert isinstance(coord._plateau_advisory_block(), str)


def test_priors_match_advisory_block_no_variants(coord: Coordinator) -> None:
    assert coord._priors_match_advisory_block() == ""


# -- _harvest_research_scout -----------------------------------------------
@pytest.mark.asyncio
async def test_harvest_research_scout_empty_and_populated(
    coord: Coordinator,
    monkeypatch,
) -> None:
    from hyperloom.orchestrator.knowledge import research_hints

    events: list[str] = []

    async def checkpoint(**kwargs):
        assert kwargs["force"] is True
        events.append("checkpoint")
        return False

    def reset():
        events.append("reset")
        coord._orchestration_seeded = False

    monkeypatch.setattr(coord, "_maybe_checkpoint_orchestration", checkpoint)
    monkeypatch.setattr(coord, "_reset_orchestration_conversation", reset)

    await coord._harvest_research_scout({})
    coord._orchestration_seeded = True
    await coord._harvest_research_scout(
        {
            "new_findings": [
                {
                    "what": "enable aiter",
                    "source": "https://example.test/aiter",
                    "domain_tags": ["serving"],
                }
            ],
            "proposal_set": [
                {
                    "name": "aiter",
                    "extra_envs": {"VLLM_ROCM_USE_AITER": "1"},
                    "source_evidence": ["https://example.test/aiter"],
                }
            ],
        }
    )
    assert research_hints.load_hints(coord.session_dir)[0]["what"] == "enable aiter"
    assert "https://example.test/aiter" in coord.shared_state.research_scout_seen_pr_ids
    assert coord._orchestration_seeded is False
    assert events == ["checkpoint", "reset", "checkpoint", "reset"]


@pytest.mark.asyncio
async def test_harvest_research_scout_does_not_persist_llm_competitor_target(coord: Coordinator) -> None:
    """LLM-authored competitor numbers must never be persisted as a consumable
    competitor target.

    Previously the scout could emit ``competitor_target`` numbers that were
    written to ``competitor_target.json`` and then consumed by the advisory
    gap block, masquerading as InferenceX-measured data. The scout is now a
    text-hints-only collector, so no competitor target must be persisted.
    """
    from hyperloom.inference_optimizer.session import session_paths
    from hyperloom.orchestrator.knowledge import research_hints

    await coord._harvest_research_scout(
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


# -- _maybe_checkpoint_orchestration ---------------------------------------
@pytest.mark.asyncio
async def test_maybe_checkpoint_orchestration_non_conversational(coord: Coordinator) -> None:
    took = await coord._maybe_checkpoint_orchestration(tick=1, phase_changed=False)
    assert took is False


# -- _handle_escalate_strategy_change --------------------------------------
def _escalate(hint: str) -> Intent:
    return Intent(
        type=IntentType.ESCALATE_STRATEGY_CHANGE,
        payload={"summary": "s", "next_action_hint": hint},
    )


@pytest.mark.asyncio
async def test_escalate_invalid_hint_broadcasts_only(coord: Coordinator) -> None:
    await coord._handle_escalate_strategy_change("orchestration", _escalate("bogus"))
    assert coord.shared_state.last_consumed_escalate_hint != "bogus"


@pytest.mark.asyncio
async def test_escalate_extend_explore_budget(coord: Coordinator) -> None:
    from hyperloom.orchestrator.phases.machine_state import (
        ESCALATE_HINT_EXTEND_EXPLORE_BUDGET,
    )

    await coord._handle_escalate_strategy_change(
        "orchestration",
        _escalate(ESCALATE_HINT_EXTEND_EXPLORE_BUDGET),
    )
    assert coord.shared_state.last_consumed_escalate_hint == ESCALATE_HINT_EXTEND_EXPLORE_BUDGET


@pytest.mark.asyncio
async def test_escalate_extend_kernel_budget(coord: Coordinator) -> None:
    from hyperloom.orchestrator.phases.machine_state import (
        ESCALATE_HINT_EXTEND_KERNEL_BUDGET,
    )

    await coord._handle_escalate_strategy_change(
        "orchestration",
        _escalate(ESCALATE_HINT_EXTEND_KERNEL_BUDGET),
    )
    assert coord.shared_state.last_consumed_escalate_hint == ESCALATE_HINT_EXTEND_KERNEL_BUDGET


@pytest.mark.asyncio
async def test_escalate_skip_to_kernel_deferred(coord: Coordinator) -> None:
    await coord._handle_escalate_strategy_change(
        "orchestration",
        _escalate("skip_to_kernel"),
    )
    assert coord.shared_state.pending_escalate_hint == "skip_to_kernel"


@pytest.mark.asyncio
async def test_escalate_skip_to_close_suppressed_pre_enablement(coord: Coordinator) -> None:
    """Q2: skip_to_close is dropped while a not-yet-enabled run is still enabling."""
    coord.shared_state.phase = "PRELUDE"
    coord.shared_state.baseline_tput = 0.0
    coord.shared_state.enablement.succeeded = False
    await coord._handle_escalate_strategy_change(
        "orchestration",
        _escalate("skip_to_close"),
    )
    assert coord.shared_state.pending_escalate_hint != "skip_to_close"


@pytest.mark.asyncio
async def test_escalate_skip_to_close_allowed_after_enablement(coord: Coordinator) -> None:
    """skip_to_close is honored once a baseline exists (guard no longer active)."""
    coord.shared_state.phase = "FRAMEWORK_AGENT"
    coord.shared_state.baseline_tput = 1234.0
    coord.shared_state.enablement.succeeded = True
    await coord._handle_escalate_strategy_change(
        "orchestration",
        _escalate("skip_to_close"),
    )
    assert coord.shared_state.pending_escalate_hint == "skip_to_close"


# -- _maybe_autosubmit_specialist_patches ----------------------------------
@pytest.mark.asyncio
async def test_autosubmit_skipped_when_no_patches(coord: Coordinator) -> None:
    from hyperloom.orchestrator.state.task_registry import Task

    task = Task(task_id="spec-1", kind="specialist", state="running", params={}, idempotency_key="k1")
    await coord._maybe_autosubmit_specialist_patches(
        task=task,
        done_payload={"patches_written": []},
    )


@pytest.mark.asyncio
async def test_autosubmit_skipped_when_files_missing(coord: Coordinator) -> None:
    from hyperloom.orchestrator.state.task_registry import Task

    task = Task(task_id="spec-2", kind="specialist", state="running", params={}, idempotency_key="k2")
    await coord._maybe_autosubmit_specialist_patches(
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
    await coord._maybe_autosubmit_specialist_patches(
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
    """A specialist with NO source patch but a non-diff tuned artifact
    (``artifacts_written`` with a real file in its worktree) is a routable
    deliverable: autosubmit must create an integrate_patch proposal so the
    artifact-install channel runs."""
    from hyperloom.orchestrator.state.task_registry import Task
    from hyperloom.inference_optimizer.session.session_paths import runs_dir

    sid = "spec-art-route"
    art_dir = runs_dir(coord.session_dir, "specialist", sid) / "worktree" / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)
    (art_dir / "tuned_fmoe.csv").write_text("cu_num,token\n304,16\n", encoding="utf-8")
    task = Task(task_id=sid, kind="specialist", state="running", params={}, idempotency_key="ka1")
    n_before = len(coord.state.pending_proposals)
    await coord._maybe_autosubmit_specialist_patches(
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
    """An ``artifacts_written`` entry whose ``source`` is an ABSOLUTE path
    OUTSIDE the specialist sandbox must NOT be routable: integrate_patch would
    reject it as ``source_outside_workspace``, so autosubmit must not create a
    proposal for it."""
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
    await coord._maybe_autosubmit_specialist_patches(
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
    """A RELATIVE artifact ``source`` that escapes the specialist sandbox via
    ``..`` must NOT be routable, even though it resolves to a real file:
    integrate_patch rejects it as ``source_outside_workspace``, so autosubmit
    must not route it."""
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
    await coord._maybe_autosubmit_specialist_patches(
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
    """A relative artifact ``source`` that climbs out of ``worktree`` via ``..``
    but lands INSIDE the workspace is still contained, so it MUST remain
    routable: the sandbox check must not reject a legitimate ``../file`` source
    that resolves within an allowed base."""
    from hyperloom.orchestrator.state.task_registry import Task
    from hyperloom.inference_optimizer.session.session_paths import runs_dir

    sid = "spec-art-parent"
    spec_root = runs_dir(coord.session_dir, "specialist", sid)
    (spec_root / "worktree").mkdir(parents=True, exist_ok=True)
    (spec_root / "tuned.csv").write_text("cu_num\n304\n", encoding="utf-8")
    task = Task(task_id=sid, kind="specialist", state="running", params={}, idempotency_key="ka4")
    n_before = len(coord.state.pending_proposals)
    await coord._maybe_autosubmit_specialist_patches(
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
    coord._record_fact_per_task(
        task=task,
        source_session_id="sess-a",
        result_dict={"gain_pct": 5.0, "output_throughput": 900.0},
        kept=True,
    )
    coord._record_fact_per_task(
        task=task,
        source_session_id="sess-a",
        result_dict={"error_class": "boom", "reason": "bad"},
        kept=False,
    )


def test_record_fact_reverted_integrate_patch_journals_revert(coord: Coordinator) -> None:
    """A reverted integrate_patch reaches the fact hook with kept=True
    (``status != failed`` is promotable), yet the journal must record REVERT
    with the REAL measured delta (from delta_pct)."""
    from hyperloom.orchestrator.state.optimization_journal import (
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
    coord._record_fact_per_task(
        task=task,
        source_session_id="sess-a",
        # tput == baseline → delta_pct ~0, executor returns "reverted", promotable.
        result_dict={
            "status": "reverted",
            "delta_pct": -0.44,
            "output_throughput": 0.440529,
            "reason": "throughput delta -0.44% < keep_threshold 1.00%",
        },
        kept=True,
    )
    entry = coord._ensure_journal().entries[-1]
    assert entry.outcome == OUTCOME_REVERT
    assert entry.gain_pct == -0.44
    assert entry.reason and "keep_threshold" in entry.reason


def test_record_fact_kept_integrate_patch_journals_keep(coord: Coordinator) -> None:
    from hyperloom.orchestrator.state.optimization_journal import OUTCOME_KEEP
    from hyperloom.orchestrator.state.task_registry import Task

    task = Task(
        task_id="t-real-keep",
        kind="integrate_patch",
        state="succeeded",
        params={},
        idempotency_key="t-real-keep",
    )
    coord._record_fact_per_task(
        task=task,
        source_session_id="sess-a",
        result_dict={"status": "kept", "delta_pct": 6.2, "output_throughput": 1100.0},
        kept=True,
    )
    entry = coord._ensure_journal().entries[-1]
    assert entry.outcome == OUTCOME_KEEP
    assert entry.gain_pct == 6.2


def test_is_promotable_result_unchanged_for_reverted_integrate_patch(coord: Coordinator) -> None:
    """A reverted integrate_patch stays promotable so it still runs the
    pending_integrate cleanup in _promote_to_shared_state."""
    assert coord._is_promotable_result("integrate_patch", {"status": "reverted"}) is True
    assert coord._is_promotable_result("integrate_patch", {"status": "failed"}) is False


# -- _compose_prompt additional branches -----------------------------------
@pytest.mark.asyncio
async def test_compose_prompt_orchestration_gain_objective(coord: Coordinator) -> None:
    coord._current_objective = TargetGainObjective(target_gain_pct=20.0)
    coord.shared_state.cumulative_gain_validated = 5.0
    await coord._compose_prompt("orchestration")
    assert coord.shared_state.target_gap_pct == pytest.approx(15.0)


@pytest.mark.asyncio
async def test_compose_prompt_renders_the_gap_it_just_computed(coord: Coordinator) -> None:
    """The first SEED must carry the live gap, not the value left from a prior tick.

    The stale value has to be absent as well as the live one present: the bug was
    a shared-state dump assembled before the recompute, which renders both.
    """
    coord._current_objective = TargetGainObjective(target_gain_pct=20.0)
    coord.shared_state.cumulative_gain_validated = 5.0
    text = await coord._compose_prompt("orchestration")
    assert "target_gap_pct=15.00" in text
    assert "target_gap_pct=0.00" not in text


@pytest.mark.asyncio
async def test_compose_prompt_time_only_objective_leaves_no_gap(coord: Coordinator) -> None:
    coord._current_objective = TimeOnlyObjective()
    coord.shared_state.cumulative_gain_validated = 5.0
    await coord._compose_prompt("orchestration")
    assert coord.shared_state.target_gap_pct == 0.0


def _pin_conversational_with_context_tools(coord: Coordinator, monkeypatch) -> None:
    """Pin the delta path AND the mounted pull tools the banner advertises."""
    monkeypatch.setattr(coord.conversation, "_orchestration_conversational", lambda: True)
    monkeypatch.setattr(coord.conversation, "_orchestration_context_tools_mounted", lambda: True)
    coord._orchestration_seeded = True


@pytest.mark.asyncio
async def test_delta_banner_names_every_registered_context_tool(coord: Coordinator, monkeypatch) -> None:
    _pin_conversational_with_context_tools(coord, monkeypatch)
    text = await coord._compose_prompt("orchestration")
    banner_start = text.find("=== Context (pull on demand) ===")
    assert banner_start != -1, "DELTA banner missing"
    banner = text[banner_start:]
    for tool in CONTEXT_TOOL_NAMES:
        assert tool in banner, f"{tool!r} not in DELTA banner"


@pytest.mark.asyncio
async def test_compose_prompt_conversational_delta(coord: Coordinator, monkeypatch) -> None:
    _pin_conversational_with_context_tools(coord, monkeypatch)
    out = await coord._compose_prompt("orchestration")
    assert "Context (pull on demand)" in out


@pytest.mark.asyncio
async def test_compose_prompt_delta_without_context_tools_names_none(coord: Coordinator, monkeypatch) -> None:
    """A backend with no pull tools must not be told to call them."""
    monkeypatch.setattr(coord.conversation, "_orchestration_conversational", lambda: True)
    coord._orchestration_seeded = True
    out = await coord._compose_prompt("orchestration")
    assert "=== Context (delta turn) ===" in out
    for tool in CONTEXT_TOOL_NAMES:
        assert tool not in out


@pytest.mark.asyncio
async def test_compose_prompt_conversational_seed_memory(coord: Coordinator, monkeypatch) -> None:
    monkeypatch.setattr(coord.conversation, "_orchestration_conversational", lambda: True)
    coord._orchestration_seeded = False
    coord._orchestration_seed_memory = "=== recovered memory ==="
    out = await coord._compose_prompt("orchestration")
    assert "recovered memory" in out


@pytest.mark.asyncio
async def test_compose_prompt_robustness_high_no_progress(coord: Coordinator, monkeypatch) -> None:
    monkeypatch.setattr(
        coord.conversation,
        "_conversation_progress_signal",
        lambda: {
            "ticks_without_progress": 9,
            "threshold": 5,
            "severity": "high",
            "last_progress_tick": 1,
        },
    )
    out = await coord._compose_prompt("robustness")
    assert "no observable progress" in out


# -- _context_analysis_reader ----------------------------------------------
def test_context_analysis_reader(coord: Coordinator) -> None:
    out = coord._context_analysis_reader()
    assert isinstance(out, str)


def test_context_analysis_reader_fallback_path(coord: Coordinator, tmp_path) -> None:
    md = tmp_path / "analysis.md"
    md.write_text("# roofline\n", encoding="utf-8")
    coord.shared_state.last_trace_analyze = {"analysis_md_path": str(md)}
    coord.shared_state.analysis_md = ""
    out = coord._context_analysis_reader()
    assert isinstance(out, str)


# -- advisory blocks enabled paths -----------------------------------------
def test_target_gap_advisory_enabled(coord: Coordinator, monkeypatch) -> None:
    from hyperloom.orchestrator.knowledge import research_hints as rh

    monkeypatch.setattr(rh, "load_competitor_target", lambda _sd: {"name": "comp"})
    monkeypatch.setattr(rh, "gap_analysis", lambda *a, **k: {"primary_gap": "throughput"})
    monkeypatch.setattr(rh, "full_gap_summary", lambda g: "GAP-SUMMARY")
    coord.shared_state.target_advisory_enabled = True
    coord.shared_state.current_best = {"tput": 1000.0, "tpot_mean_ms": 5.0}
    coord.shared_state.tp = 1
    coord.shared_state.conc = 64
    assert coord._target_gap_advisory_block() == "GAP-SUMMARY"
    assert coord._current_primary_gap() == "throughput"


def test_target_gap_advisory_no_target(coord: Coordinator, monkeypatch) -> None:
    from hyperloom.orchestrator.knowledge import research_hints as rh

    monkeypatch.setattr(rh, "load_competitor_target", lambda _sd: None)
    coord.shared_state.target_advisory_enabled = True
    assert coord._target_gap_advisory_block() == ""
    assert coord._current_primary_gap() is None


# -- _promote_warm_replay --------------------------------------------------
def test_promote_warm_replay_non_dict(coord: Coordinator) -> None:
    coord._promote_warm_replay("nope")  # type: ignore[arg-type]
    assert coord.shared_state.warm_replay_outcome["status"] == "failed"


def test_promote_warm_replay_failed_status(coord: Coordinator) -> None:
    coord._promote_warm_replay({"status": "failed", "error_class": "x", "error": "boom"})
    assert coord.shared_state.warm_replay_outcome["status"] == "failed"


def test_promote_warm_replay_invalid_tput(coord: Coordinator) -> None:
    coord.shared_state.baseline_tput = 0.0
    coord._promote_warm_replay({"status": "succeeded", "output_throughput": 0.0})
    assert coord.shared_state.warm_replay_outcome["status"] == "failed"


def test_promote_warm_replay_reproduced_no_params(coord: Coordinator) -> None:
    coord.shared_state.baseline_tput = 800.0
    coord._promote_warm_replay(
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
    coord._promote_warm_replay(
        {"status": "succeeded", "output_throughput": 900.0},
        task=task,
    )
    assert coord.shared_state.warm_replay_outcome["status"] == "reproduced"


# -- _maybe_auto_retry_specialist ------------------------------------------
def _spec_task(**params):
    from hyperloom.orchestrator.state.task_registry import Task

    return Task(task_id="spec-r", kind="specialist", state="running", params=params, idempotency_key="spec-r-key")


def _result(state="failed", result=None, error=None):
    from hyperloom.orchestrator.loop.sub_agent_runner import SubAgentResult

    return SubAgentResult(task_id="spec-r", state=state, result=result or {}, error=error)


@pytest.mark.asyncio
async def test_auto_retry_disabled_by_env(coord: Coordinator, monkeypatch) -> None:
    monkeypatch.setenv("INFERENCE_OPTIMIZER_SPECIALIST_AUTO_RETRY", "0")
    assert await coord._maybe_auto_retry_specialist(_spec_task(), _result()) is False


@pytest.mark.asyncio
async def test_auto_retry_not_eligible(coord: Coordinator, monkeypatch) -> None:
    monkeypatch.setenv("INFERENCE_OPTIMIZER_SPECIALIST_AUTO_RETRY", "1")
    res = _result(result={"runner_status": "empty_synthesised"}, error="no_output")
    assert await coord._maybe_auto_retry_specialist(_spec_task(), res) is False


@pytest.mark.asyncio
async def test_auto_retry_schedules_on_transient_failure(coord: Coordinator, monkeypatch) -> None:
    monkeypatch.setenv("INFERENCE_OPTIMIZER_SPECIALIST_AUTO_RETRY", "1")
    res = _result(result={"runner_status": "stale"}, error="timeout waiting")
    scheduled = await coord._maybe_auto_retry_specialist(_spec_task(), res)
    assert scheduled is True


@pytest.mark.asyncio
async def test_auto_retry_caps_attempts(coord: Coordinator, monkeypatch) -> None:
    monkeypatch.setenv("INFERENCE_OPTIMIZER_SPECIALIST_AUTO_RETRY", "1")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_SPECIALIST_AUTO_RETRY_MAX", "1")
    res = _result(result={"runner_status": "stale"}, error="timeout")
    task = _spec_task(_auto_retry_attempt=1)
    assert await coord._maybe_auto_retry_specialist(task, res) is False


# -- _fan_out_specialist_wave (invalid entries) ---------------------------
@pytest.mark.asyncio
async def test_fan_out_wave_rejects_invalid_entries(coord: Coordinator, monkeypatch) -> None:
    called = []
    monkeypatch.setattr(
        coord,
        "_handle_delegate",
        lambda *a, **k: called.append(a),
    )
    intent = Intent(type=IntentType.DELEGATE, payload={"idempotency_key": "w", "action_name": "specialist"})
    from hyperloom.orchestrator.policy.gate import PolicyDenied

    with pytest.raises(PolicyDenied):
        await coord._fan_out_specialist_wave(
            "orchestration",
            intent,
            {"tasks": ["not-a-dict", {}, {"task_description": "   "}]},
        )
    assert called == []


# -- _warm_specialist_params -----------------------------------------------
@pytest.mark.asyncio
async def test_warm_specialist_params_fills_defaults(coord: Coordinator) -> None:
    coord.shared_state.gpu_type = "mi300x"
    coord.shared_state.tp = 1
    coord.shared_state.conc = 64
    coord.shared_state.isl = 256
    coord.shared_state.osl = 256
    params: dict = {"domain": "kernel_agent"}
    await coord._warm_specialist_params(params)
    assert params["gpu_type"] == "mi300x"
    assert params["tp"] == 1
    assert "kb_subgraph" in params


# -- finalize_recipe_and_journal ------------------------------------
def test_recipe_kb_finalize_recipe_and_journal_no_kb(coord: Coordinator) -> None:
    coord.shared_state.current_best = {"tput": 950.0}
    coord.shared_state.cumulative_gain_validated = 12.5
    coord.finalize_recipe_and_journal()


# -- _record_fact_per_variant ----------------------------------------------
def test_record_fact_per_variant_keep_revert_skip(coord: Coordinator) -> None:
    from hyperloom.orchestrator.state.task_registry import Task

    task = Task(task_id="t-var", kind="explore", state="succeeded", params={}, idempotency_key="kv")
    # SKIPPED_DEDUP -> early return (no journal row)
    coord._record_fact_per_variant(
        task=task,
        source_session_id="s",
        variant_outcome={"outcome": "SKIPPED_DEDUP", "variant_name": "v0"},
    )
    coord._record_fact_per_variant(
        task=task,
        source_session_id="s",
        variant_outcome={
            "outcome": "KEEP",
            "variant_name": "v1",
            "metrics": {"gain_pct": 4.0, "output_throughput": 900.0},
            "variant": {"name": "v1"},
        },
    )
    coord._record_fact_per_variant(
        task=task,
        source_session_id="s",
        variant_outcome={
            "outcome": "REVERT",
            "variant_name": "v2",
            "error_class": "regressed",
            "reason": "slower",
            "metrics": {},
        },
    )
