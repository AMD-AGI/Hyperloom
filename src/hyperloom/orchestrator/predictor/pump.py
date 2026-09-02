# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Drive the predictor at FRAMEWORK entry and after each accepted step.

The predictor answers once per decision point, so a single call would only ever
use the empty-stack part of what it knows. A KEEP changes ``current_best``, the
stack and the cumulative gain -- a new decision point, and one it is equally
equipped to answer. The pump therefore re-fires as the stack deepens, forming a
greedy chain that stops on its own when a step fails to land.

Termination is a property of the idempotency key rather than a counter. The key
carries the macro-cycle and the stack depth, and the task registry returns an
existing row for a known key whatever its state. No KEEP means the depth is
unchanged, the key repeats, nothing is enqueued, and the chain ends. A durable
``coordinator.db`` extends that across a resume.

Nothing here evaluates a proposal. Config answers become ordinary ``explore``
variants and patch answers become free-form specialist mandates, so both are
measured by the machinery that already grades ``default_grid`` and
``llm_direct``.
"""

from __future__ import annotations

import logging
from typing import Any

from hyperloom.common.prompt_safety import flatten_for_prompt
from hyperloom.orchestrator.phases.machine_state import (
    PHASE_FRAMEWORK_AGENT,
    phase_budget_remaining_seconds,
    phase_elapsed_seconds,
)
from hyperloom.orchestrator.predictor import config as predictor_config
from hyperloom.orchestrator.predictor.client import Prediction, predict
from hyperloom.orchestrator.predictor.payload import build_request

log = logging.getLogger(__name__)

#: Audit label on everything the predictor produced. Not a scheduling gate:
#: ``proposer_for()`` passes an unknown provenance through unchanged, so this
#: reaches the stack, the journal and the breakdown without a closed set.
PROVENANCE = "primatune"

#: ``params["source"]`` for the enqueued explore task. Distinct from
#: ``resume_stack_revalidate``, which the explore executor special-cases to skip
#: anchor rebinding.
TASK_SOURCE = "coordinator_internal_primatune"

#: Cap on a mandate handed to a specialist. The prompt builder allows more; this
#: is model-authored text entering another model's prompt, so it stays short.
MAX_MANDATE_CHARS = 4000


def _chain_depth(state: Any) -> int:
    """Steps the chain has taken in the current macro-cycle."""
    if int(getattr(state, "predictor_chain_cycle", -1)) != int(getattr(state, "macro_cycle", 0) or 0):
        return 0
    return int(getattr(state, "predictor_chain_steps", 0) or 0)


def _note_step(state: Any) -> None:
    """Record one chain step against the current macro-cycle."""
    cycle = int(getattr(state, "macro_cycle", 0) or 0)
    if int(getattr(state, "predictor_chain_cycle", -1)) != cycle:
        state.predictor_chain_cycle = cycle
        state.predictor_chain_steps = 0
    state.predictor_chain_steps = int(getattr(state, "predictor_chain_steps", 0) or 0) + 1


def _budget_spent_pct(state: Any) -> float | None:
    """Share of this phase entry's budget already spent, or ``None`` if unlimited.

    ``phase_budget_remaining_seconds`` charges ``phase_elapsed_seconds`` against
    the entry's allotment, so the two together recover the allotment itself.
    """
    remaining = phase_budget_remaining_seconds(state)
    if remaining is None:
        return None
    elapsed = phase_elapsed_seconds(state)
    total = elapsed + remaining
    if total <= 0:
        return 100.0
    return 100.0 * elapsed / total


def _declined(reason: str) -> None:
    log.debug("predictor_pump: standing down (%s)", reason)


def _gate(phase: Any, conf: predictor_config.PredictorConfig) -> bool:
    """Whether to spend a request at this decision point."""
    state = phase.shared_state

    if not conf.enabled:
        _declined("disabled")
        return False

    # The tick calls the FRAMEWORK pump unconditionally and lets it return
    # early, so the phase check belongs here rather than at the call site.
    if str(getattr(state, "phase", "") or "").strip().upper() != PHASE_FRAMEWORK_AGENT:
        _declined("not in FRAMEWORK_AGENT")
        return False

    framework = getattr(state, "framework", "")
    if not conf.supports(framework):
        # Flag catalogues exist for sglang and vllm only, so the consumer cannot
        # validate an answer for anything else. Declining beats sending a
        # request whose reply could not be trusted.
        _declined(f"framework {framework!r} has no flag catalogue")
        return False

    depth = _chain_depth(state)
    if depth >= conf.max_chain:
        _declined(f"chain cap reached ({depth}/{conf.max_chain})")
        return False

    spent = _budget_spent_pct(state)
    if spent is not None and spent > conf.budget_pct:
        # Budgets are cumulative across entries, so a cycle_reloop can land in a
        # phase with nothing left while the chain re-fires on the new
        # macro-cycle. The idempotency key cannot see that.
        _declined(f"phase {spent:.0f}% spent, past the {conf.budget_pct:.0f}% the chain may use")
        return False

    return True


def _grid_entry(answer: Prediction, *, cycle: int, depth: int) -> dict[str, Any] | None:
    """Turn a config answer into one explore variant, or ``None`` when it has none."""
    if not answer.has_config:
        return None
    extra_args = " ".join(
        flag if value is True else f"{flag} {value}" for flag, value in answer.server_args.items()
    ).strip()
    return {
        "name": f"primatune-c{cycle}-s{depth}",
        "extra_args": extra_args,
        "extra_envs": dict(answer.envs),
        "provenance": PROVENANCE,
        "note": "first-pass tuning prediction",
    }


async def _enqueue_config(phase: Any, answer: Prediction, *, cycle: int, depth: int) -> bool:
    """Enqueue the config channel as an explore task. Returns whether it landed."""
    entry = _grid_entry(answer, cycle=cycle, depth=depth)
    if entry is None:
        return False

    from hyperloom.orchestrator.state.shared_state import inject_stack_base_params

    state = phase.shared_state
    params: dict[str, Any] = {
        "source": TASK_SOURCE,
        "reason": f"primatune:cycle{cycle}:step{depth}",
        "grid": [entry],
        # Read by the breakdown recorder: per-variant provenance does not reach
        # it, so the round carries the proposer when the whole grid agrees.
        "provenance": PROVENANCE,
    }
    if getattr(state, "baseline_config_path", ""):
        params["config_path"] = state.baseline_config_path
    # Without the anchor the variant is graded against the bare baseline rather
    # than current_best, which is what it will actually be launched on top of.
    inject_stack_base_params(params, state, anchor=True)

    last_baseline = getattr(state, "last_baseline", None)
    if isinstance(last_baseline, dict):
        script = str(last_baseline.get("benchmark_script") or "").strip()
        if script:
            params["benchmark_script"] = script

    lanes, ttl = phase._registry_lanes_ttl("explore")
    task, existing = await phase.tasks.create_or_return_existing(
        kind="explore",
        params=params,
        idempotency_key=f"primatune-c{cycle}-s{depth}",
        requires_lanes=lanes,
        lease_ttl_sec=ttl,
    )
    log.info(
        "predictor_pump: explore task_id=%s cycle=%d depth=%d existing=%s args=%r envs=%r",
        task.task_id,
        cycle,
        depth,
        existing,
        entry["extra_args"],
        entry["extra_envs"],
    )
    return not existing


async def _dispatch_patch(phase: Any, answer: Prediction, *, cycle: int, depth: int) -> bool:
    """Hand a prose source change to a free-form specialist. Returns whether it landed."""
    if not answer.has_source_change:
        return False

    from hyperloom.inference_optimizer.breakdown.agent_ownership import LEVER_SOURCE_PATCH

    state = phase.shared_state
    # The prompt builder interpolates this into a one-line markdown quote
    # (``> {desc}``) without sanitising it. Flattening is what stops a newline
    # from leaving the quote and forging a section header; it also defangs code
    # fences and angle brackets on the way through.
    mandate = flatten_for_prompt(answer.source_change)[:MAX_MANDATE_CHARS]
    params: dict[str, Any] = {
        "scope": "freeform",
        "task_description": mandate,
        # A free-form specialist resolves its mode before a domain is assigned,
        # so FREEFORM_DOMAIN.default_mode never applies and the default is
        # research: no worktree, no patch instruction, no patches_written.
        "mode": "patch",
        "lane": "cpu",
        "source_phase": "FRAMEWORK_AGENT",
        "source": TASK_SOURCE,
        # Deliberately no "domain": _forward_integrate_source rewrites
        # provenance to specialist:<domain> when one is present, which would
        # erase the label this whole exercise exists to measure.
        "provenance": PROVENANCE,
        "lever_kind": LEVER_SOURCE_PATCH,
        "framework": str(getattr(state, "framework", "") or "").strip().lower(),
    }
    lanes, ttl = phase._registry_lanes_ttl("specialist")
    task, existing = await phase.tasks.create_or_return_existing(
        kind="specialist",
        params=params,
        idempotency_key=f"primatune-patch-c{cycle}-s{depth}",
        requires_lanes=lanes,
        side_effects=["writes_results", "writes_patches"],
        lease_ttl_sec=ttl,
    )
    log.info(
        "predictor_pump: freeform specialist task_id=%s cycle=%d depth=%d existing=%s mandate=%r",
        task.task_id,
        cycle,
        depth,
        existing,
        mandate[:120],
    )
    return not existing


def _log_shadow(answer: Prediction, *, cycle: int, depth: int, session_id: str) -> None:
    """Record what would have been enqueued, at zero GPU cost."""
    log.info(
        "predictor_shadow: session=%s cycle=%d depth=%d parsed=%s "
        "server_args=%r envs=%r source_change=%r prompt_chars=%s dropped=%r",
        session_id,
        cycle,
        depth,
        answer.parsed,
        answer.server_args,
        answer.envs,
        answer.source_change[:200],
        answer.meta.get("prompt_chars"),
        answer.meta.get("dropped_flags"),
    )


async def pump(phase: Any, *, caller: str) -> None:
    """Consult the predictor once, and act on the answer.

    Safe to call on every tick: the gate and the idempotency key make repeat
    calls at an unchanged decision point free. Never raises into the tick loop.

    Args:
        phase (Any): The ``FrameworkPhase`` collaborator.
        caller (str): Label for the log ("entry" / "tick").
    """
    try:
        conf = predictor_config.load()
        if not _gate(phase, conf):
            return

        state = phase.shared_state
        cycle = int(getattr(state, "macro_cycle", 0) or 0)
        depth = len(getattr(state, "optimization_stack", None) or [])
        session_id = str(getattr(state, "session_id", "") or "")

        request = build_request(state, session_id=session_id, phase_label=conf.phase_label)
        answer = predict(request, endpoint=conf.endpoint, timeout_sec=conf.timeout_sec)

        if not conf.enqueues:
            _log_shadow(answer, cycle=cycle, depth=depth, session_id=session_id)
            return

        if not answer.parsed or answer.is_empty:
            log.info("predictor_pump: no action from %s (cycle=%d depth=%d)", caller, cycle, depth)
            return

        landed = await _enqueue_config(phase, answer, cycle=cycle, depth=depth)
        landed = await _dispatch_patch(phase, answer, cycle=cycle, depth=depth) or landed
        if landed:
            _note_step(state)
    except Exception:  # noqa: BLE001 — advisory work must never fail a session
        log.exception("predictor_pump (%s) failed", caller)
