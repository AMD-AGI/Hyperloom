# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Drive the predictor at FRAMEWORK entry, after each accepted step, and after
each fresh roofline.

The predictor answers once per decision point, so a single call would only ever
use the empty-stack part of what it knows. A KEEP changes ``current_best``, the
stack and the cumulative gain -- a new decision point, and one it is equally
equipped to answer. The pump therefore re-fires as the stack deepens, forming a
greedy chain.

Why this runs ahead of the specialists
--------------------------------------
The predictor is a local model: one request is seconds of GPU on its own host
and no API spend at all. An LLM specialist is the opposite -- measured over a
real session, specialists were 83 of 87 LLM calls and 97% of the output tokens.
So the loop asks the free proposer first and only falls back to the paid ones
once the free one has stopped landing KEEPs; see
``predictor_holds_specialists`` in ``phases/framework.py``.

Termination
-----------
``predictor_chain_steps`` counts *consecutive* attempts that have not produced a
KEEP, within one macro-cycle. It is bumped when a round is enqueued and reset to
zero when one of that round's variants lands, so it measures a losing streak
rather than total work: a chain that keeps winning is never cut off, and one
that stops winning hands over after ``max_chain`` attempts (hardcoded to 1:
one sample batch is measured in full; a KEEP still resets the streak so a win
can deepen the stack and earn a second HTTP).

The count is also the attempt number in the idempotency key. After a KEEP the
depth changes and the key is new; without the attempt in the key a later
macro-cycle at the same depth would collide with the first round's row.

Evidence freshness
------------------
A KEEP big enough to cross the roofline watermark leaves a re-profile in flight.
The pump stands down until it lands rather than asking again over evidence it
already knows is stale. A KEEP too small to trigger one is not waited for --
there would be nothing to wait for, and the chain would stall forever.

Nothing here evaluates a proposal. Config answers become ordinary ``explore``
variants and patch answers become free-form specialist mandates, so both are
measured by the machinery that already grades ``default_grid`` and
``llm_direct``.
"""

from __future__ import annotations

import logging
from typing import Any

from hyperloom.common.prompt_safety import flatten_for_prompt
from hyperloom.orchestrator.actions.executors._canonical_fingerprint import (
    canonical_fingerprint,
)
from hyperloom.orchestrator.actions.executors._proposal_identity import (
    effective_fingerprint,
)
from hyperloom.orchestrator.phases.machine_state import PHASE_FRAMEWORK_AGENT
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


def attempts_without_keep(state: Any) -> int:
    """Consecutive predictor rounds in this macro-cycle that landed no KEEP.

    A counter from an earlier macro-cycle does not apply: ``cycle_reloop``
    re-enters the phase against a different stack, which is a decision point the
    predictor has not answered yet.

    Args:
        state (Any): The ``SharedState``.

    Returns:
        int: The losing streak, zero when the recorded cycle is not the current
            one.
    """
    if int(getattr(state, "predictor_chain_cycle", -1)) != int(getattr(state, "macro_cycle", 0) or 0):
        return 0
    return int(getattr(state, "predictor_chain_steps", 0) or 0)


def note_attempt(state: Any) -> None:
    """Count one attempt against the streak, re-basing on a new macro-cycle."""
    cycle = int(getattr(state, "macro_cycle", 0) or 0)
    if int(getattr(state, "predictor_chain_cycle", -1)) != cycle:
        state.predictor_chain_cycle = cycle
        state.predictor_chain_steps = 0
    state.predictor_chain_steps = int(getattr(state, "predictor_chain_steps", 0) or 0) + 1


def note_keep(state: Any) -> None:
    """Clear the streak because a predictor variant landed.

    Called from writeback rather than here: whether a round produced a KEEP is
    only known once it has been benchmarked, which is long after the pump
    returned.

    Args:
        state (Any): The ``SharedState``.
    """
    state.predictor_chain_cycle = int(getattr(state, "macro_cycle", 0) or 0)
    state.predictor_chain_steps = 0


def predictor_holds_specialists(state: Any) -> bool:
    """Whether the free proposer still owns this phase, so paid ones stand down.

    Read by the FRAMEWORK phase before it spends an LLM specialist and by the
    PolicyGate before it admits a free-form delegate or an orchestration
    explore. Specialists were 97% of a real session's LLM output tokens, so
    deferring them until the predictor has stopped landing KEEPs is where the
    saving comes from.

    The streak is bumped when a round is *enqueued*, not when it finishes.
    With ``max_chain=1`` that would release the hold the moment the grid
    lands on the benchmark lane -- which is when orchestration explore and
    candidate_discovery most want the same GPUs. So an in-flight round
    (``predictor_round_task_id``) keeps the hold on until it is graded.
    ``_release_finished_round`` clears the marker; a KEEP resets the streak
    and the hold stays on for the next HTTP.

    Returns ``False`` whenever the predictor could not answer anyway -- no
    endpoint, shadow mode, an unsupported framework. Getting that wrong would
    suppress every proposer at once and leave the phase with nothing to
    benchmark.

    Args:
        state (Any): The ``SharedState``.

    Returns:
        bool: True while specialists should be held back.
    """
    conf = predictor_config.load()
    if not conf.enqueues:
        return False
    if not conf.supports(getattr(state, "framework", "")):
        return False
    if attempts_without_keep(state) < conf.max_chain:
        return True
    return bool(str(getattr(state, "predictor_round_task_id", "") or "").strip())


def _declined(reason: str) -> None:
    log.debug("predictor_pump: standing down (%s)", reason)


async def _release_finished_round(phase: Any) -> None:
    """Drop an in-flight marker naming a round that already finished.

    ``predictor_round_task_id`` is what stops a second round from being
    enqueued while the first is still on the benchmark lane -- without it the
    attempt number bumps on dispatch, the key changes, and the next tick buys a
    duplicate round at full GPU cost.

    A task deduplicated into an already-finished attempt never reports back, so
    the marker it left would gate the chain permanently, and it is persisted
    state: a resumed session would inherit a gate nothing could open. That is
    not hypothetical -- the roofline watermark shipped with exactly this bug and
    needed the same release. Checking the registry here is what makes it
    self-healing.

    Args:
        phase (Any): The collaborator exposing ``shared_state`` and ``tasks``.
    """
    from hyperloom.orchestrator.state.task_registry import TERMINAL_STATES

    state = phase.shared_state
    pending = str(getattr(state, "predictor_round_task_id", "") or "").strip()
    if not pending:
        return
    try:
        task = await phase.tasks.get(pending)
    except Exception:  # noqa: BLE001 — a missing row is itself finished
        task = None
    if task is not None and str(getattr(task, "state", "")) not in TERMINAL_STATES:
        return
    state.predictor_round_task_id = ""
    log.info("predictor_pump: released the round gate held by finished task=%s", pending)


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

    pending_round = str(getattr(state, "predictor_round_task_id", "") or "").strip()
    if pending_round:
        # One round at a time. The attempt number bumps on dispatch, so without
        # this the next tick would see a different key and buy a second round
        # while the first is still on the benchmark lane.
        _declined(f"round {pending_round} still being measured")
        return False

    streak = attempts_without_keep(state)
    if streak >= conf.max_chain:
        _declined(f"{streak} attempts without a KEEP, at the cap of {conf.max_chain}")
        return False

    pending_roofline = str(getattr(state, "auto_roofline_pending_task_id", "") or "").strip()
    if pending_roofline:
        # Asking now would answer over the snapshot the last KEEP already
        # invalidated. Only an in-flight re-profile is waited for: the watermark
        # needs a 10% step, so a smaller KEEP triggers none and there would be
        # nothing to wait for -- waiting anyway would stall the chain for good.
        _declined(f"roofline {pending_roofline} in flight, waiting for fresh evidence")
        return False

    return True


#: Env prefixes that belong to the other serving stack. Repair already drops
#: illegal *flags*; it does not drop envs, so a vLLM round used to inherit
#: ``SGLANG_*`` from an SGLang-heavy sampler.
_FOREIGN_ENV_PREFIXES: dict[str, tuple[str, ...]] = {
    "vllm": ("SGLANG_",),
    "sglang": ("VLLM_",),
}


def _envs_for_framework(envs: dict[str, Any] | None, framework: str) -> dict[str, str]:
    """Keep same-framework and shared envs; drop the other stack's prefixes."""
    raw = dict(envs or {})
    prefixes = _FOREIGN_ENV_PREFIXES.get(str(framework or "").strip().lower(), ())
    if not prefixes:
        return {str(k): str(v) for k, v in raw.items()}
    kept: dict[str, str] = {}
    dropped: list[str] = []
    for key, value in raw.items():
        name = str(key)
        if any(name.upper().startswith(prefix) for prefix in prefixes):
            dropped.append(name)
            continue
        kept[name] = str(value)
    if dropped:
        log.info(
            "predictor_pump: dropped cross-framework envs %s (framework=%s)",
            dropped,
            framework,
        )
    return kept


def _stack_base(state: Any | None) -> tuple[str, dict[str, str]]:
    """Current champion extras the next explore variant will launch on top of."""
    if state is None:
        return "", {}
    best = getattr(state, "current_best", None) or {}
    if not isinstance(best, dict):
        return "", {}
    args = str(best.get("effective_extra_server_args") or best.get("extra_server_args") or "").strip()
    raw = best.get("extra_envs") or {}
    envs = {str(k): str(v) for k, v in dict(raw).items()} if isinstance(raw, dict) else {}
    return args, envs


def _merge_launch(base_args: str, extra_args: str, base_envs: dict[str, str], extra_envs: dict[str, str]) -> tuple[str, dict[str, str]]:
    """Stack ∪ proposal, last-wins on flags via ``canonical_fingerprint`` pairing."""
    merged_args = f"{base_args} {extra_args}".strip()
    merged_envs = dict(base_envs)
    merged_envs.update(extra_envs)
    return merged_args, merged_envs


def _tested_maps(state: Any | None) -> tuple[set[str], set[str]]:
    """Delta fingerprints (tested keys) and launched-recipe fingerprints."""
    if state is None:
        return set(), set()
    search = getattr(state, "explore_search", None) or {}
    tested = search.get("tested") if isinstance(search, dict) else None
    if not isinstance(tested, dict):
        return set(), set()
    delta_keys = {str(key) for key in tested}
    launch_fps: set[str] = set()
    for row in tested.values():
        if not isinstance(row, dict):
            continue
        evidence = row.get("launch_evidence")
        flags = ""
        if isinstance(evidence, dict):
            flags = str(evidence.get("requested_server_flags") or "")
        if not flags.strip():
            flags = str(row.get("extra_server_args") or row.get("extra_args") or "")
        envs = row.get("extra_envs") or {}
        if not isinstance(envs, dict):
            envs = {}
        launch_fps.add(canonical_fingerprint(flags, envs))
    return delta_keys, launch_fps


def _skip_reason(
    extra_args: str,
    extra_envs: dict[str, str],
    *,
    seen_delta: set[str],
    base_args: str,
    base_envs: dict[str, str],
    tested_delta: set[str],
    tested_launch: set[str],
) -> str | None:
    """Why this proposal should not be benched, or ``None`` to keep it."""
    delta_fp = effective_fingerprint(extra_args, extra_envs)
    merged_args, merged_envs = _merge_launch(base_args, extra_args, base_envs, extra_envs)
    launch_fp = canonical_fingerprint(merged_args, merged_envs)
    base_fp = canonical_fingerprint(base_args, base_envs)
    if delta_fp in seen_delta:
        return "batch_dup"
    if launch_fp == base_fp:
        return "already_on_stack"
    if delta_fp in tested_delta:
        return "already_tested_delta"
    if launch_fp in tested_launch:
        return "already_tested_launch"
    return None


def _grid_entries(
    answer: Prediction,
    *,
    cycle: int,
    depth: int,
    attempt: int,
    framework: str = "",
    state: Any | None = None,
) -> list[dict[str, Any]]:
    """Turn every distinct config proposal into one explore variant.

    All of them go into one grid rather than one task each, which is what keeps
    the idempotency key -- and with it the chain's termination rule -- a
    property of the decision point rather than of how many samples the service
    happened to return.

    A grid is not a set of alternatives. The explore executor grades variants in
    order and folds each KEEP onto the stack before grading the next, so N
    entries are a greedy N-deep stacking attempt within one round. Order
    therefore matters, and it is the order the service sampled in: the same
    AITER backend switch measured -1.17% on a bare baseline and +2.68% stacked
    on fp8 KV cache in this fleet.

    Historical ``explore_search.tested`` *is* an eligibility gate here. Explore
    itself only collapses duplicates inside one submitted grid; a second HTTP
    that re-proposes a delta already measured -- or a new delta whose launched
    recipe matches one already measured -- would otherwise buy another Magpie
    round. ``already_tested_launch`` is the case that caught DPO round-2
    ``--quantization fp8`` on an ``fp8_e4m3`` champion after round 1 had already
    launched that combination.

    Cross-framework envs (``SGLANG_*`` on vLLM, ``VLLM_*`` on SGLang) are
    dropped here. The consumer's ``repair()`` already strips illegal flags; it
    does not strip envs, and a foreign env would ride into the launch.

    Args:
        answer (Prediction): The predictor's answer.
        cycle (int): Macro-cycle, for the variant name.
        depth (int): Stack depth, for the variant name.
        attempt (int): Attempt at this depth, for the variant name.
        framework (str): Session framework; used to drop the other stack's envs.
        state (Any | None): SharedState, for stack extras and ``explore_search``.

    Returns:
        list[dict[str, Any]]: One grid entry per configuration proposal that
            is not a known duplicate, empty when the answer has none left.
    """
    base_args, base_envs = _stack_base(state)
    tested_delta, tested_launch = _tested_maps(state)
    seen_delta: set[str] = set()
    entries: list[dict[str, Any]] = []
    for index, action in enumerate(answer.config_actions):
        extra_args = " ".join(
            flag if value is True else f"{flag} {value}" for flag, value in action.server_args.items()
        ).strip()
        extra_envs = _envs_for_framework(action.envs, framework)
        reason = _skip_reason(
            extra_args,
            extra_envs,
            seen_delta=seen_delta,
            base_args=base_args,
            base_envs=base_envs,
            tested_delta=tested_delta,
            tested_launch=tested_launch,
        )
        delta_fp = effective_fingerprint(extra_args, extra_envs)
        if reason is not None:
            log.info(
                "predictor_pump: skip reason=%s fp=%s args=%r envs=%r",
                reason,
                delta_fp,
                extra_args,
                extra_envs,
            )
            continue
        seen_delta.add(delta_fp)
        entries.append(
            {
                "name": f"primatune-c{cycle}-s{depth}-a{attempt}-{index}",
                "extra_args": extra_args,
                "extra_envs": extra_envs,
                "provenance": PROVENANCE,
                "note": "first-pass tuning prediction",
            }
        )
    return entries


async def _enqueue_config(
    phase: Any, answer: Prediction, *, cycle: int, depth: int, attempt: int
) -> str:
    """Enqueue the config channel as an explore task.

    Returns:
        str: ``"new"`` when a fresh explore task was created, ``"existing"``
            when the idempotency key hit an already-queued round, ``"empty"``
            when every config proposal was absent or skipped.
    """
    state = phase.shared_state
    entries = _grid_entries(
        answer,
        cycle=cycle,
        depth=depth,
        attempt=attempt,
        framework=str(getattr(state, "framework", "") or ""),
        state=state,
    )
    if not entries:
        return "empty"

    from hyperloom.orchestrator.state.shared_state import inject_stack_base_params

    params: dict[str, Any] = {
        "source": TASK_SOURCE,
        "reason": f"primatune:cycle{cycle}:step{depth}:attempt{attempt}",
        "grid": entries,
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
        idempotency_key=f"primatune-c{cycle}-s{depth}-a{attempt}",
        requires_lanes=lanes,
        lease_ttl_sec=ttl,
    )
    if not existing:
        # Claimed before the log line so a crash between the two cannot leave a
        # round running with nothing pointing at it.
        state.predictor_round_task_id = str(task.task_id)
    log.info(
        "predictor_pump: explore task_id=%s cycle=%d depth=%d attempt=%d existing=%s "
        "variants=%d/%d %r",
        task.task_id,
        cycle,
        depth,
        attempt,
        existing,
        len(entries),
        len(answer.config_actions),
        [(e["extra_args"], e["extra_envs"]) for e in entries],
    )
    return "existing" if existing else "new"


async def _dispatch_patch(
    phase: Any, answer: Prediction, *, cycle: int, depth: int, attempt: int
) -> bool:
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
        idempotency_key=f"primatune-patch-c{cycle}-s{depth}-a{attempt}",
        requires_lanes=lanes,
        side_effects=["writes_results", "writes_patches"],
        lease_ttl_sec=ttl,
    )
    log.info(
        "predictor_pump: freeform specialist task_id=%s cycle=%d depth=%d attempt=%d "
        "existing=%s mandate=%r",
        task.task_id,
        cycle,
        depth,
        attempt,
        existing,
        mandate[:120],
    )
    return not existing


def _log_shadow(answer: Prediction, *, cycle: int, depth: int, session_id: str) -> None:
    """Record what would have been enqueued, at zero GPU cost.

    Every proposal is logged, not just the one that would have been benchmarked:
    shadow mode exists to measure what the predictor nominates, and with
    sampling on, the spread is the measurement. A run that only recorded the
    head would have shown the flag that mattered on none of its lines while the
    model was proposing it in one sample out of four.
    """
    log.info(
        "predictor_shadow: session=%s cycle=%d depth=%d parsed=%s actions=%d "
        "samples=%s prompt_chars=%s dropped=%r",
        session_id,
        cycle,
        depth,
        answer.parsed,
        len(answer.actions),
        answer.meta.get("samples"),
        answer.meta.get("prompt_chars"),
        answer.meta.get("dropped_flags"),
    )
    for index, action in enumerate(answer.actions):
        log.info(
            "predictor_shadow:   [%d/%d] server_args=%r envs=%r source_change=%r",
            index,
            len(answer.actions),
            action.server_args,
            action.envs,
            action.source_change[:200],
        )


async def pump(phase: Any, *, caller: str) -> None:
    """Consult the predictor once, and act on the answer.

    Safe to call on every tick and from writeback: the gate and the idempotency
    key make repeat calls at an unchanged decision point free. Never raises into
    the tick loop.

    Args:
        phase (Any): The ``FrameworkPhase`` collaborator, or anything else
            exposing ``shared_state``, ``tasks`` and ``_registry_lanes_ttl``.
        caller (str): Label for the log ("entry" / "tick" / "keep" / "roofline").
    """
    try:
        conf = predictor_config.load()
        await _release_finished_round(phase)
        if not _gate(phase, conf):
            return

        state = phase.shared_state
        cycle = int(getattr(state, "macro_cycle", 0) or 0)
        depth = len(getattr(state, "optimization_stack", None) or [])
        attempt = attempts_without_keep(state)
        session_id = str(getattr(state, "session_id", "") or "")

        request = build_request(state, session_id=session_id, phase_label=conf.phase_label)
        answer = predict(request, endpoint=conf.endpoint, timeout_sec=conf.timeout_sec)

        if not conf.enqueues:
            _log_shadow(answer, cycle=cycle, depth=depth, session_id=session_id)
            return

        if not answer.parsed or answer.is_empty:
            # An answer with nothing in it is a spent attempt like any other.
            # Without counting it a predictor that always declines would hold
            # the specialists back for the whole phase.
            note_attempt(state)
            log.info(
                "predictor_pump: no action from %s (cycle=%d depth=%d attempt=%d)",
                caller,
                cycle,
                depth,
                attempt,
            )
            return

        config_status = await _enqueue_config(
            phase, answer, cycle=cycle, depth=depth, attempt=attempt
        )
        patch_landed = await _dispatch_patch(
            phase, answer, cycle=cycle, depth=depth, attempt=attempt
        )
        if config_status == "new" or patch_landed:
            # Counted on dispatch, not on the result: the round takes tens of
            # minutes to grade, and an unincremented counter would leave the key
            # unchanged and the streak unable to advance if the task failed
            # outright. ``note_keep`` clears it from writeback when a variant
            # lands, which is what makes this a losing streak.
            note_attempt(state)
        elif config_status == "empty" and answer.config_actions:
            # Every config proposal was a known duplicate. Count it so
            # max_chain still hands the phase to specialists instead of
            # re-POSTing forever.
            note_attempt(state)
            log.info(
                "predictor_pump: all config proposals skipped (%s cycle=%d depth=%d attempt=%d)",
                caller,
                cycle,
                depth,
                attempt,
            )
    except Exception:  # noqa: BLE001 — advisory work must never fail a session
        log.exception("predictor_pump (%s) failed", caller)
