# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Compose the Orchestration agent's system prompt from typed inputs.

Wraps the ``orchestration.md`` rules fragment with generated sections
(mission, session context, pipeline/budget, phase contract, action catalogue,
decision framework, cycle directive, optional kernel-opt reference, rules).
Deterministic for given inputs; the only IO is reading the rules fragment.

Sections are scoped by the ``phase`` argument: a module whose behaviour the
phase cannot reach is omitted, so the agent is never handed a payload contract
PolicyGate would deny. A blank phase renders every module.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path

from hyperloom.inference_optimizer.protocol.action_surfaces import (
    ActionMetadata,
    COORDINATOR_INTERNAL_ACTIONS,
    FULL_ENABLED_ACTIONS,
    KERNEL_ACTION_REQUEST_KINDS,
    KERNEL_AGENT_OWNED_ACTIONS,
    NO_KERNEL_AGENT_ENABLED_ACTIONS,
)
from . import read_rules_fragment as _read_rules_fragment
from .transport import TRANSPORTS, TRANSPORT_STRUCTURED_OUTPUT, TRANSPORT_TOOLS


# Phase ordering for the catalogue; unknown phases appended at the end.
_PHASE_ORDER: tuple[str, ...] = (
    "prep",
    "measure",
    "analysis",
    "explore",
    "deep",
    "validate",
    "finalize",
    "support",
)
_PHASE_HEADERS: dict[str, str] = {
    "prep": "Prep — initialise session metadata. Always finishes first.",
    "measure": "Measure — establish baseline_tput. Gate for everything else.",
    "analysis": "Analysis — read-only; produces traces / candidate kernels.",
    "explore": "Explore — propose modifications; one round produces a candidate, not yet validated.",
    "deep": "Deep — kernel_agent-owned. Emit via REQUEST(target_agent='kernel_agent', kind=...).",
    "validate": "Validate — apply the accumulated optimization_stack and re-bench to get an honest cumulative gain.",
    "finalize": "Finalize — write the final report.",
    "support": "Support — invoke only when triggered (plateau / crash / re-exploration).",
}


# Phases each scoped prompt module belongs to.
_KERNEL_REQUEST_PHASES: frozenset[str] = frozenset({"KERNEL_AGENT"})
_EXPLORE_GRID_PHASES: frozenset[str] = frozenset({"FRAMEWORK_AGENT"})
_BASELINE_RECOVERY_PHASES: frozenset[str] = frozenset({"PRELUDE"})

# ``<!-- phase: A, B -->`` scopes the ``### `` heading that follows it.
_PHASE_TAG_RE = re.compile(r"^<!--\s*phase:\s*(?P<phases>[A-Za-z_,\s]+?)\s*-->$")

# ``<!-- transport: tools -->`` scopes the ``### `` heading that follows it,
# exactly as the phase tag does, and the two compose.
_TRANSPORT_TAG_RE = re.compile(r"^<!--\s*transport:\s*(?P<transports>[a-z_,\s]+?)\s*-->$")


def _renders_in(phase: str, phases: frozenset[str]) -> bool:
    """Whether a phase-scoped module renders for ``phase``.

    Args:
        phase (str): Normalised current phase; ``""`` disables scoping.
        phases (frozenset[str]): Phases the module belongs to.

    Returns:
        bool: ``True`` when the module should render.
    """
    return not phase or phase in phases


def _section_mission() -> list[str]:
    """Build the MISSION section lines.

    Returns:
        list[str]: Markdown lines describing the Orchestration agent's
        cumulative-gain objective and per-tick decision question.
    """
    return [
        "## 1. MISSION",
        "",
        "You are the Orchestration agent of an autonomous inference-optimization loop.",
        "Your single most important goal is to maximise the run's **cumulative_gain_validated**",
        "(percent over baseline_tput) within the wall-clock budget.",
        "",
        "Every tick, ask yourself:",
        '  "Given current SharedState, remaining time, and the action catalogue below,',
        '   which next action gives the highest expected_gain / cost_minutes?"',
        "",
        'An optimization is only "real" once it has been validated as part of the',
        "full optimization_stack. ``explore`` measures each KEEP on that stack, so",
        "cumulative_gain_validated advances automatically — drive the loop until",
        "``explore`` has produced at least one KEEP.",
    ]


def _section_session_context(
    *,
    framework: str,
    kernel_enabled: bool,
    objective_kind: str,
    objective_value: float | str | None,
    max_minutes: int,
    framework_agent_phase_enabled: bool = True,
    framework_source_roots: tuple[str, ...] | None = None,
) -> list[str]:
    """Build the SESSION CONTEXT section lines.

    Args:
        framework (str): The framework name shown verbatim.
        kernel_enabled (bool): Whether kernel_agent-owned actions are enabled.
        framework_agent_phase_enabled (bool): Whether the FRAMEWORK_AGENT phase
            is enabled.
        objective_kind (str): The objective kind (e.g. ``time_only``,
            ``gain_pct``).
        objective_value (float | str | None): Optional objective target value
            rendered alongside the kind.
        max_minutes (int): Wall-clock budget for the run, in minutes.
        framework_source_roots (tuple[str, ...] | None): Optional framework
            source roots; a PolicyGate-default note is shown when empty.

    Returns:
        list[str]: Markdown lines describing static session context and phase
        awareness.
    """
    obj = f"{objective_kind}"
    if objective_value not in (None, ""):
        obj = f"{objective_kind}={objective_value}"
    roots = framework_source_roots or ()
    roots_line = ", ".join(roots) if roots else "(defaults from PolicyGate)"
    return [
        "## 2. SESSION CONTEXT",
        "",
        f"- framework        : {framework}",
        f"- kernel_enabled   : {'true' if kernel_enabled else 'false'}",
        f"- optimize_enabled : {'true' if framework_agent_phase_enabled else 'false'}",
        f"- objective        : {obj}",
        f"- max_minutes      : {max_minutes}",
        f"- framework_source_roots: {roots_line}",
        "",
        "Per-tick dynamic context (Phase, Mission progress, Time budget,",
        "Shared session state, KB hints, inbox tail) is appended below the",
        "system prompt every tick by the Coordinator.",
        "The Time-budget block carries `remaining=X.Xmin`.",
        "See PHASE CONTRACT below for the phase chain, per-phase allowed",
        "actions, and phase-transition rules.",
    ]


def _section_phase_semantics(
    *,
    kernel_enabled: bool,
    framework_agent_phase_enabled: bool = True,
) -> list[str]:
    """Render the per-phase LLM-proposable action contract (current phase
    injected dynamically by the Coordinator).

    Phases switched off by ``--no-framework-agent`` / ``--no-kernel`` keep
    their row in the chain but are annotated
    ``(DISABLED: --no-xxx — phase skipped)`` so Orchestration plans against the
    phases the run will actually enter.

    Args:
        kernel_enabled: Whether kernel_agent-owned actions are enabled.
        framework_agent_phase_enabled: Whether the FRAMEWORK_AGENT phase is enabled.

    Returns:
        Markdown lines for the phase-contract section.
    """
    from ..phases.machine_state import render_phase_proposable_bullets

    # phase name -> the flag that disabled it (None => always enabled).
    disabled_suffix: dict[str, str] = {}
    if not framework_agent_phase_enabled:
        disabled_suffix["FRAMEWORK_AGENT"] = "--no-framework-agent"
    if not kernel_enabled:
        disabled_suffix["KERNEL_AGENT"] = "--no-kernel"

    lines: list[str] = [
        "## 3a. PHASE CONTRACT",
        "",
        "The Coordinator runs the optimization as a linear pipeline.",
        "Each tick it injects a `=== Phase ===` block with the current",
        "phase. Per-phase proposable action sets (PolicyGate R1 enforces these):",
        "",
    ]
    if disabled_suffix:
        skipped = ", ".join(f"{ph} ({flag})" for ph, flag in disabled_suffix.items())
        lines.append(f"Phases SKIPPED this run (never entered): {skipped}.")
        lines.append("")
    lines.extend(
        render_phase_proposable_bullets(
            disabled_suffix=disabled_suffix,
        )
    )
    lines.extend(
        [
            "",
            f"{', '.join(sorted(COORDINATOR_INTERNAL_ACTIONS))} are never in the",
            "sets above: the Coordinator auto-manages them and PolicyGate denies",
            "any attempt to propose them. Denial of any action",
            "lands in your inbox as a `policy_denied` event.",
            "",
            "Phase transitions are Coordinator-owned. The hard advance gates",
            "are: `baseline_tput > 0` exits PRELUDE; the per-phase budget cap",
            "or a terminal stop_reason exits FRAMEWORK_AGENT / KERNEL_AGENT /",
            "SWEEP; the wall-clock deadline (closing phase) routes to CLOSE.",
            "You may also emit `escalate_strategy_change{next_action_hint=",
            "'skip_to_kernel' | 'skip_to_sweep' | 'skip_to_close'}` directly",
            "(no longer robustness-only) when you judge the current phase",
            "exhausted; the Coordinator validates the hint vocab and routes",
            "the transition on the next tick.",
            "EXCEPTION — normal SWEEP convergence: do NOT emit `skip_to_close`",
            "once the sweep has completed (sweep_done). The",
            "Coordinator exits SWEEP → CLOSE on its own with an honest terminal",
            "stop_reason (`sweep_done` / `global_converged`). `skip_to_close`",
            "is reserved for genuine early abandonment (e.g. infra is dead and",
            "the sweep cannot run at all) — it stamps `robustness_escalated`,",
            "so emitting it on a normal finish mislabels the run.",
        ]
    )
    return lines


def _filter_actions(
    registry: Mapping[str, ActionMetadata],
    enabled: Iterable[str],
) -> list[ActionMetadata]:
    """Resolve enabled action names to their catalogue metadata.

    Args:
        registry (Mapping[str, ActionMetadata]): The action catalogue to look up.
        enabled (Iterable[str]): Enabled action names, drawn from the closed
            :data:`FULL_ENABLED_ACTIONS` set.

    Returns:
        list[ActionMetadata]: Metadata for each enabled action, in the input
        order.
    """
    enabled_set: list[str] = list(enabled)
    out: list[ActionMetadata] = []
    for name in enabled_set:
        meta = registry.get(name)
        assert meta is not None
        out.append(meta)
    return out


def _resolve_prompt_prelude(
    action_registry: Mapping[str, ActionMetadata],
    enabled_actions: Iterable[str],
    framework: str,
    kernel_enabled: bool | None,
    rules_fragment_path: Path | None,
) -> tuple[list[ActionMetadata], bool, str, str]:
    """Resolve the shared prelude for the orchestration / critic prompt builders.

    Args:
        action_registry (Mapping[str, ActionMetadata]): The action catalogue.
        enabled_actions (Iterable[str]): Action names enabled for this run.
        framework (str): The framework name; normalised to lower-case (default
            ``sglang``).
        kernel_enabled (bool | None): Explicit override; ``None`` derives from
            whether any KERNEL_OWNED action is enabled.
        rules_fragment_path (Path | None): Path to the rules fragment.

    Returns:
        tuple[list[ActionMetadata], bool, str, str]: ``(actions, kernel_enabled,
        framework_norm, rules_md)``.
    """
    actions = _filter_actions(action_registry, enabled_actions)
    if kernel_enabled is None:
        kernel_enabled = any(a.name in KERNEL_AGENT_OWNED_ACTIONS for a in actions)
    framework_norm = (framework or "sglang").strip().lower() or "sglang"
    rules_md = _read_rules_fragment(rules_fragment_path)
    return actions, kernel_enabled, framework_norm, rules_md


def join_sections(sections: list[list[str]]) -> str:
    """Join prompt sections into the final prompt string (shared epilogue).

    Args:
        sections (list[list[str]]): Per-section line lists.

    Returns:
        str: The sections joined (lines by ``\\n``, sections by blank line),
        right-stripped with a trailing newline.
    """
    return "\n\n".join("\n".join(s) for s in sections).rstrip() + "\n"


def _phase_eta_summary(actions: list[ActionMetadata]) -> list[tuple[str, float, list[str]]]:
    """Group actions by phase in _PHASE_ORDER; return (phase, eta_min_sum, names).

    Args:
        actions: The enabled actions to group by pipeline phase.

    Returns:
        A list of ``(phase, eta_min_sum, names)`` tuples ordered by
        ``_PHASE_ORDER`` with unknown phases appended last.
    """
    bucket: dict[str, list[ActionMetadata]] = {}
    for a in actions:
        bucket.setdefault(a.pipeline_phase, []).append(a)
    ordered: list[tuple[str, float, list[str]]] = []
    seen: set[str] = set()
    for phase in _PHASE_ORDER:
        if phase not in bucket:
            continue
        items = bucket[phase]
        eta = sum(max(0.0, a.typical_runtime_min) for a in items)
        ordered.append((phase, eta, [a.name for a in items]))
        seen.add(phase)
    for phase, items in bucket.items():
        if phase in seen:
            continue
        eta = sum(max(0.0, a.typical_runtime_min) for a in items)
        ordered.append((phase, eta, [a.name for a in items]))
    return ordered


def _section_pipeline_and_budget(
    actions: list[ActionMetadata],
    *,
    max_minutes: int,
) -> list[str]:
    """Build the PIPELINE & TIME BUDGET section lines.

    Args:
        actions (list[ActionMetadata]): The enabled actions, summarised by
            phase ETA.
        max_minutes (int): Wall-clock budget for the run, compared against the
            summed phase ETAs.

    Returns:
        list[str]: Markdown lines describing per-phase ETAs and budget guidance.
    """
    lines: list[str] = [
        "## 3. PIPELINE & TIME BUDGET",
        "",
        "Run roughly in phase order; you may revisit a phase, but never skip prep / measure.",
        "Per-phase typical wall-clock (sum of typical_runtime_min over enabled actions):",
        "",
    ]
    eta_total = 0.0
    for phase, eta, names in _phase_eta_summary(actions):
        header = _PHASE_HEADERS.get(phase, phase)
        eta_total += eta
        joined = ", ".join(names) or "(none enabled)"
        lines.append(f"- **{phase}** (~{eta:.0f} min) — {header}")
        lines.append(f"    actions: {joined}")
    lines.extend(
        [
            "",
            f"Sum of typical phase ETAs: ~{eta_total:.0f} min vs max_minutes={max_minutes}.",
            "If sum >> budget, prefer high-gain/low-cost actions and skip optional",
            "phases (analysis / support). If sum << budget, do an extra explore round",
            "before report.",
        ]
    )
    return lines


def _format_gain_pair(meta: ActionMetadata) -> str:
    """Format an action's expected-gain range as a short string.

    Args:
        meta (ActionMetadata): The action whose ``expected_gain_pct`` range to
            format.

    Returns:
        str: ``"0%"`` when the range is zero, otherwise ``"lo-hi%"``.
    """
    lo, hi = meta.expected_gain_pct
    if lo == 0.0 and hi == 0.0:
        return "0%"
    return f"{lo:.0f}-{hi:.0f}%"


def _llm_selectable_domains() -> str:
    """The domains Orchestration may name, pipe-separated, from the registry."""
    from ..specialists.domains import SPECIALIST_DOMAINS

    return "|".join(d.key for d in SPECIALIST_DOMAINS if d.llm_selectable)


def _format_emit_hint(meta: ActionMetadata) -> str:
    """Build the per-action ``EMIT:`` hint showing the correct transport.

    Kernel-owned actions render a ``REQUEST{...}`` template; ``specialist`` /
    ``integrate_patch`` render their closed ``delegate`` payload contracts;
    ``report`` renders a fixed zero-gain propose_action; everything else
    renders a ``propose_action`` template.

    Args:
        meta (ActionMetadata): The action to build an emit hint for.

    Returns:
        str: The emit-hint string for the catalogue entry.
    """
    if meta.name in KERNEL_AGENT_OWNED_ACTIONS:
        kind_hint = KERNEL_ACTION_REQUEST_KINDS[meta.name]
        return f"REQUEST{{target_agent='kernel_agent', kind='{kind_hint}', params={{...}}}}"
    if meta.name == "report":
        return "propose_action{action_name='report', predicted_gain_pct=0.0}"
    if meta.name == "specialist":
        return (
            "delegate{action_name='specialist', params={"
            f"domain=<one of {_llm_selectable_domains()}>, "
            "gap_canonical_id=<stable gap id>, "
            "gap_symptom?=<str>, gap_layer?=<str>, "
            "gap_evidence?={profile_trace:..., ...}, "
            "max_turns?=<int<=1000 or 0=unbounded>}}"
        )
    if meta.name == "integrate_patch":
        return (
            "delegate{action_name='integrate_patch', params={"
            "specialist_task_id=<completed specialist task_id>, "
            "patches?=[<patch paths from specialist_done>], "
            "config_changes?={ENV_VAR: value}, "
            "keep_threshold_pct?=<session-cycle default>, "
            "accuracy_baseline?=<float>}}"
        )
    return f"propose_action{{action_name='{meta.name}', predicted_gain_pct=<your estimate>}}"


def _format_grid_injection_hint(name: str) -> str | None:
    """Return a per-action one-liner showing how to override grid, or None.

    Args:
        name: The action name to render a grid-injection hint for.

    Returns:
        The grid-injection hint string for ``explore``, or ``None`` for any
        other action.
    """
    if name == "explore":
        return (
            "GRID INPUT (REQUIRED): emit "
            "`delegate{action_name='explore', params={grid: [{name, "
            "extra_args, extra_envs, remove_args?, unset_envs?, "
            "args_mode?: 'append'|'replace', provenance, kb_evidence?, "
            "pr_evidence?, source_evidence?}, ...], "
            "base_extra_args?, base_tput?, accuracy_baseline?, "
            "keep_threshold_pct?: <session-cycle default>}}`. "
            "Variants run serially; a KEEP is graded on its decision "
            "round. Variant identity is content-based (args+envs+"
            "remove_args+unset_envs+args_mode); only exact duplicates within "
            "the same submitted grid are collapsed, so any prior fingerprint "
            "may be re-proposed. "
            "Use remove_args/unset_envs to ablate harmful base flags; "
            "args_mode='replace' to drop inherited server args. "
            "provenance values: 'llm_direct', 'default_grid', "
            "'specialist:<domain-or-tag>' (audit/advisory, not a gate). "
            "SIZE: target 4 variants, hard maximum 6. Variants run serially "
            "on a single benchmark lane at ~13min each, so a 4-variant round "
            "is about an hour of GPU. Submit a 5th or 6th only when it still "
            "beats the median of the four you already have; a grid the round "
            "cannot finish is truncated from the end, dropping whatever you "
            "ranked last rather than whatever is worth least."
        )
    return None


def _section_action_catalogue(actions: list[ActionMetadata], *, phase: str = "") -> list[str]:
    """Build the ACTIONS YOU MAY USE catalogue section, grouped by phase.

    Every enabled action keeps its description and cost/gain/risk line in every
    phase, so a ``skip_to_*`` decision can still compare what later phases do.
    Only the payload contracts (``EMIT:`` template and grid schema) are scoped.

    Args:
        actions (list[ActionMetadata]): The actions enabled for this run.
        phase (str): Normalised current pipeline phase; ``""`` renders every
            payload contract.

    Returns:
        list[str]: Markdown lines for the action catalogue.
    """
    from ..phases.machine_state import llm_proposable_actions_for

    proposable = frozenset(llm_proposable_actions_for(phase)) if phase else frozenset()
    lines: list[str] = [
        "## 4. ACTIONS YOU MAY USE",
        "",
        "Catalogue is filtered to the actions enabled for this run. Each entry",
        "carries: phase / typical wall-clock / expected gain range / accuracy_risk /",
        "crash_risk / one-line description / how to emit it in its own phase.",
        "",
    ]
    by_phase = _phase_eta_summary(actions)
    name_to_meta = {a.name: a for a in actions}
    for action_phase, _eta, names in by_phase:
        lines.append(f"### {action_phase}")
        lines.append("")
        for name in names:
            meta = name_to_meta[name]
            tag_parts: list[str] = []
            if name in KERNEL_AGENT_OWNED_ACTIONS:
                tag_parts.append("KERNEL_AGENT-OWNED")
            tag = (" (" + ", ".join(tag_parts) + ")") if tag_parts else ""
            lines.append(f"- **{name}**{tag} — {meta.description}")
            lines.append(
                f"    cost ~{meta.typical_runtime_min:.0f}min  "
                f"gain {_format_gain_pair(meta)}  "
                f"acc_risk={meta.accuracy_risk:.2f}  "
                f"crash_risk={meta.crash_risk:.2f}  "
                f"family={meta.family}"
            )
            if phase and name not in proposable:
                lines.append(f"    (not proposable in {phase} — see PHASE CONTRACT for its phase)")
                continue
            lines.append(f"    EMIT: {_format_emit_hint(meta)}")
            grid_hint = _format_grid_injection_hint(name)
            if grid_hint:
                lines.append(f"    {grid_hint}")
        lines.append("")
    return lines


def _section_decision_framework(*, kernel_enabled: bool, phase: str = "", transport: str = "") -> list[str]:
    """Build the DECISION FRAMEWORK section lines.

    Covers the per-tick selection order, FAILURE RECOVERY, and the
    Config-arm IDEA GENERATION block.

    Args:
        kernel_enabled (bool): Whether kernel_agent-owned actions are enabled for this
            run.
        phase (str): Normalised current pipeline phase; ``""`` renders every
            phase-scoped block.
        transport (str): One of :data:`TRANSPORTS`; scopes the tool references
            inside FAILURE RECOVERY.

    Returns:
        list[str]: Markdown lines for the decision framework.
    """
    lines = [
        "## 5. DECISION FRAMEWORK (heuristics + facts — the next action is your call)",
        "",
        "These are reference heuristics and objective facts, not a forced",
        "sequence. Read the dynamic SharedState section and decide:",
        "",
        "1. **Stop**: if `stop_reason` is set OR `cumulative_gain_validated >= target_gain_pct`,",
        "   propose `report` once (if not already done) then heartbeat 'goal-reached'.",
        "2. **Measure**: if `baseline_tput == 0`, propose `baseline`. Wait for",
        "   delegated_result; do NOT re-baseline on a positive result with warnings.",
        "3. **Stack-aware grids**: route every grid attempt through",
        "   ``delegate{action_name='explore', params={grid: [...] }}``;",
        "   there is no standalone validation step (see Hard rules).",
    ]
    lines.append(
        "4. **Analysis is auto-managed**. Roofline/profile is enqueued by "
        "the Coordinator at PRELUDE and at every +10% validated-gain "
        "watermark crossing. Never propose ``profile`` or ``roofline`` "
        "(see Hard rules). While a refresh is in flight, specialist / "
        "explore / kernel_agent-owned dispatches are deferred until "
        "``analysis.md`` / ``last_profile_trace`` refreshes.",
    )
    lines.extend(
        [
            "5. **Phase-aware action selection**. There is no system-side",
            "   priority list. Pick the next action by reading FACTS in this order:",
            "",
            "   a. **Phase + allowed actions** (the `=== Phase ===` /",
            "      `=== Phase-allowed actions ===` blocks). PolicyGate denies",
            "      anything outside the allowed set.",
            "   b. **Current gaps** (the `=== Current gaps ===` block, sourced",
            "      from `SharedState.gaps[]`). Each row shows canonical_id /",
            "      layer / severity / symptom / attempts count + last attempt.",
            "      The LLM picks the next gap to tackle based on layer (which",
            "      routes the specialist domain), severity (high vs medium),",
            "      and whether the attempts history shows the gap is still",
            "      worth pushing on. When the section is missing it means",
            "      baseline hasn't completed yet — fall back to",
            "      `last_action_failures` + `explore_search.winners_history`.",
            "   c. **KB sub-graphs + warm-start recipe** when present —",
            "      cross-session priors carry " + "*qualitative* hints (what worked / what failed last time).",
            "   d. **`=== Untested proposals (current cycle) ===`** — the",
            "      executable specialist proposals this cycle that no explore",
            "      round has benched, ranked by gap severity and truncated to",
            "      a count the block states. This is the grid's primary",
            "      source; an entry marked ATOMIC goes in verbatim.",
            "   e. **Ordering facts**: baseline runs before anything else",
            "      (invariant). ``analysis.md`` / ``last_profile_trace`` arrive",
            "      automatically from the Coordinator-owned analysis task at",
            "      PRELUDE and at every +10% watermark crossing — you do not",
            "      need a manually-proposed profile before ``kernel_opt``.",
            "6. **Phase budget awareness**. The `=== Phase ===` block's",
            "   ``budget`` line carries ``remaining_sec`` against the phase's",
            "   ``pct`` share; as it falls, prefer lower-cost / known-good",
            "   actions (explore over kernel_opt).",
            "   The Plateau advisory block is informational only for KERNEL. In",
            "   OPTIMIZE it reports each arm separately: BOTH arms dry advances",
            "   to KERNEL_AGENT (``reason=optimize_no_more_leverage``) at the",
            "   next phase-compute, while one arm dry means work the other.",
            "   When you judge the current phase exhausted,",
            "   emit ``escalate_strategy_change{next_action_hint=",
            "   'skip_to_kernel' | 'skip_to_sweep' | 'skip_to_close'}`` (see",
            "   PHASE CONTRACT for the skip_to_close exception).",
            "",
            "If you cannot move forward, emit",
            "`send_message{topic='heartbeat', body_md='blocked: <reason>'}` and let",
            "Robustness escalate. NEVER stay silent.",
        ]
    )
    lines.extend(_failure_recovery_lines(phase=phase, transport=transport))
    if _renders_in(phase, _EXPLORE_GRID_PHASES):
        lines.extend(_idea_generation_lines())
    return lines


def _failure_recovery_lines(*, phase: str, transport: str = "") -> list[str]:
    """Build the FAILURE RECOVERY block.

    Always-on trigger and F3/F4 rules are inlined; the detailed diagnostic
    surfaces, fingerprint semantics, and worked examples live in the
    ``failure_recovery`` reference document, which only a transport with the
    ``read_reference`` tool can pull.

    Args:
        phase (str): Normalised current pipeline phase.
        transport (str): One of :data:`TRANSPORTS`; a transport without tools
            is not pointed at the reference document.

    Returns:
        list[str]: Markdown lines for the failure-recovery block.
    """
    baseline_scoped = _renders_in(phase, _BASELINE_RECOVERY_PHASES)
    detail = (
        "/ `last_action_failures` for the error detail."
        if transport == TRANSPORT_STRUCTURED_OUTPUT
        else (
            "/ `last_action_failures` for the error detail. Full diagnostic surfaces,\n"
            "fingerprint semantics, and examples: ``read_reference('failure_recovery')``."
        )
    )
    lines = [
        "",
        "### FAILURE RECOVERY (apply BEFORE re-proposing an action that just failed)",
        "",
        "When the inbox carries a fresh `delegated_result{state!='succeeded'}`",
        "or `last_action_failures[-1].action == <X>`, do NOT re-propose the same",
        "action with the same params. Consult `last_<action>` / `<action>_attempts`",
        detail,
        "",
        "Rules (apply in order):",
        "",
    ]
    if baseline_scoped:
        lines.extend(
            [
                "* **RULE F1** (PRELUDE) — same baseline fingerprint twice failed →"
                " change at least one of the eight fingerprint fields.",
                "* **RULE F2** (PRELUDE) — `error_class='no_report'` + no"
                " `rescued_from_leaked_path:*` → redirect RESULT_DIR or set"
                " INFERENCE_OPTIMIZER_RESCUE_PATHS.",
            ]
        )
    lines.extend(
        [
            "* **RULE F3** — repeated `error_class='subprocess_nonzero'` on `baseline`"
            " → stop retrying baseline; heartbeat 'blocked: …' and let Robustness"
            " intervene. Explore variants may be re-proposed; read the failure log first.",
            "* **RULE F4** — `policy_denial_streak` is information only."
            " Change something substantive; re-emitting the identical intent wastes a tick.",
        ]
    )
    return lines


def _idea_generation_lines() -> list[str]:
    """Build the config-arm IDEA GENERATION block.

    Returns:
        list[str]: Markdown lines describing how to compose the next
        ``explore`` grid.
    """
    return [
        "",
        "### IDEA GENERATION (apply after EVERY explore round)",
        "",
        "Compose the next `explore` grid from `explore_search` (winners +",
        "rejected, each with `±x.xx% gain_pct`) and `discovered_flags`:",
        "",
        "1. **Sibling values** — if `--max-num-seqs 256` won, try 128 / 512;",
        "   sweep a winning boolean's related `*_AITER_*` family.",
        "2. **Synergy** — combine last round's winners via",
        "   `synergy_mode='auto'` (deduped against `synergy_attempted`).",
        "3. **Re-examine rejects** — per `explore_search.rejected` variant, judge",
        "   whether the failure is stale, fixable, or ruled out; re-propose the",
        "   same config to revalidate or change the value (a `-2%` reject is a",
        "   dead flag; `-0.3%` may clear the bar once patched).",
        "4. **Mine flags** — when winners are empty, pull untested boolean",
        "   toggles from `discovered_flags.<framework>.backend_flags`.",
        "5. **Ablate harmful base config** — when a user/base flag or env may",
        "   be slowing the workload, emit a variant with `remove_args` and/or",
        "   `unset_envs` instead of only adding more knobs.",
        "",
        "Variant identity is content-based (args+envs+remove_args+",
        "unset_envs+args_mode); only exact same-grid duplicates are collapsed.",
        "`extra_server_args` is framework-neutral (routed to EXTRA_SGLANG_ARGS",
        "/ EXTRA_VLLM_ARGS / EXTRA_ATOM_ARGS by `--framework`).",
        "",
        "Draw first from `=== Untested proposals (current cycle) ===`; the",
        "five moves above are for topping the grid up to its target of 4",
        "(hard maximum 6) once the queue is drained of anything worth running.",
        "",
        "An explore round that produces zero new ideas is a bug — heartbeat",
        "with body_md='idea-pipeline-empty' so Robustness can intervene.",
    ]


_KERNEL_OPT_PIPELINE_BODY: str = """\
## 6. KERNEL-OPT REQUEST REFERENCE (payload templates — NOT a forced ordering)

The four kernel_agent-owned actions are picked per the DECISION FRAMEWORK
(phase allowed-set + gaps + KB priors); there is no system-side
priority ranking. Pick the next one by reading these facts in order:
a `state.gaps[]` `layer='kernel_agent'` gap with attempts left →
`last_kernel_opt` (KEEP→integrate next; PARTIAL→retry at most
`_DEFAULT_KERNEL_OPT_MAX_PARTIAL` then rejected; REVERT→rejected) →
skip ids in `rejected_kernel_ids` → recover from `last_action_failures`.
A KERNEL_AGENT plateau signal (3 REVERTs across distinct kernels, or low
recent KEEP gain) is rendered as advisory; KERNEL_AGENT → SWEEP advance is
driven by the phase budget, an `escalate_strategy_change` hint, or a
terminal stop_reason. Read the advisory and emit `skip_to_sweep` if
you want to wind down sooner.

### `trace_analyze` — must precede every `run_optimization`

  request{target_agent: 'kernel_agent', kind: 'trace_analyze',
          params: {trace_input: <verbatim last_profile_trace>, top_k: 10}}

  Skip if `last_trace_analyze.trace_input` already equals
  `last_profile_trace` (cached). Explore/sweep/report are NEVER gated on it.

### `gemm_tuning` — `run_gemm_tuning`

  request{target_agent: 'kernel_agent', kind: 'run_gemm_tuning', params={}}

  Current GEAK owns the KERNEL phase by default and decides GEMM applicability
  internally. Only use this legacy request in explicit per-kernel forge mode
  (`KERNEL_OPT_BACKEND_ORDER=forge`).

### `kernel_opt` — payload for `run_optimization`

Pick the next id from `last_trace_analyze.reusable_native_kernel_ids`
(NEVER from raw `hot_kernels_top15` — vendor binaries reject as
`non_reusable_kernel`); if that list is empty, don't propose kernel_opt.

The `kernel_id` MUST be one of those ids copied verbatim (e.g. `k001`).
NEVER invent an id and NEVER pass an operator name (e.g. `aten::mm`,
`aiter.silu_and_mul`) or any token from `analysis_md` — operator names
are non-unique (several kernels share `aten::mm`) and are rejected.
`skipped_kernels_top` lists operators TraceLens detected but cannot
rewrite (each with a `skip_reason`); they are off-limits, not targets.

  request{target_agent: 'kernel_agent', kind: 'run_optimization',
          params: {kernel_id: <picked kernel_id>,
                   source_file: <hot_kernels[i].source_file>,
                   candidates_path: <trace_analyze_done.candidates_path>}}

  Budget policy: DO NOT add a `budget_minutes` field. The Coordinator owns
  the per-optimization wall clock and applies the same value it uses for its
  own dispatch; naming one here pins the backend to this template's number
  instead, which is how a raised operator budget got silently discarded.

  Backend policy: DO NOT add a `backends` field. Current GEAK owns the
  KERNEL phase by default. Forge per-kernel mode is available only when the
  operator set exactly `KERNEL_OPT_BACKEND_ORDER=forge`.
  Read `kernel_opt_task_attempts` +
  `pending_keep_kernels` to
  see what's still queueable; the batch handler filters
  rejected/in-flight/exhausted candidates.

### `integrate` — forced immediately after a KEEP

On `run_optimization_done` with `decision='KEEP'`, integrate is the only
allowed action until the patch lands on `optimization_stack`:

  request{target_agent: 'kernel_agent', kind: 'integrate',
          params: {kernel_id, patch_path, target_file, base_tput,
                   extra_server_args, config_path}}

  Omit `base_tput` / `patch_path` / `source_file` and the Coordinator
  fills them from `current_best.tput` and the per-kernel
  `kernel_opt_task_attempts` ledger (this is what drains a multi-KEEP queue).
  PARTIAL / REVERT → do NOT integrate; pick the next action normally.

  **Multi-KEEP queue:** `pending_keep_kernels` (sorted strongest-first)
  lists queued KEEPs; integrate `[0]` each tick. Do NOT propose `report`
  while it is non-empty, nor while `untried_hot_reusable_kernels`
  (reusable hot kernels with zero attempts and `gpu_pct >= 5%`, the
  default that `HYPERLOOM_KERNEL_OPT_MIN_GPU_PCT` overrides) remain —
  drain them with `run_optimization{candidates_path: <from
  last_trace_analyze>}` (the batch handler fans out automatically).

### KERNEL TARGETING

Only rewrite reusable native sources in the trace. NEVER optimize
`/tmp/torchinductor*` / `triton_poi_*` / `triton_red_*` runtime-generated
kernels — they're tied to one compile cache and not reusable."""


def _filter_rules_fragment(rules_md: str, *, phase: str = "", transport: str = "") -> str:
    """Drop rules-fragment ``### `` blocks that this run cannot reach.

    A ``<!-- phase: A, B -->`` or ``<!-- transport: tools -->`` comment scopes
    the ``### `` heading that follows it, up to the next ``### `` / ``## ``
    heading; both may precede the same heading and both must then match.
    Untagged blocks always render, so a section added without a tag stays
    always-on. The tag comments and the fragment's leading maintainer
    blockquote are never emitted.

    Args:
        rules_md (str): Raw rules-fragment markdown.
        phase (str): Normalised current pipeline phase; ``""`` keeps everything.
        transport (str): One of :data:`TRANSPORTS`; ``""`` keeps everything.

    Returns:
        str: The fragment with unreachable blocks removed.
    """
    kept: list[str] = []
    pending_phases: frozenset[str] | None = None
    pending_transports: frozenset[str] | None = None
    active_phases: frozenset[str] | None = None
    active_transports: frozenset[str] | None = None
    in_header = True
    for line in rules_md.splitlines():
        if in_header:
            if not line.strip() or line.lstrip().startswith(">"):
                continue
            in_header = False
        stripped = line.strip()
        phase_tag = _PHASE_TAG_RE.match(stripped)
        if phase_tag is not None:
            pending_phases = frozenset(p.strip().upper() for p in phase_tag.group("phases").split(",") if p.strip())
            continue
        transport_tag = _TRANSPORT_TAG_RE.match(stripped)
        if transport_tag is not None:
            pending_transports = frozenset(t.strip() for t in transport_tag.group("transports").split(",") if t.strip())
            continue
        if line.startswith("### "):
            active_phases, pending_phases = pending_phases, None
            active_transports, pending_transports = pending_transports, None
        elif line.startswith("## "):
            active_phases = active_transports = None
            pending_phases = pending_transports = None
        if active_phases is not None and not _renders_in(phase, active_phases):
            continue
        if active_transports is not None and transport and transport not in active_transports:
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def _section_rules(rules_md: str, *, phase: str = "", transport: str = "") -> list[str]:
    """Build the RULES & OUTPUT PROTOCOL section wrapping the rules fragment.

    Args:
        rules_md (str): The raw rules-fragment markdown; a placeholder is used
            when empty.
        phase (str): Normalised current pipeline phase; scopes the fragment's
            phase-tagged blocks.
        transport (str): One of :data:`TRANSPORTS`; scopes the fragment's
            transport-tagged blocks.

    Returns:
        list[str]: Markdown lines for the RULES & OUTPUT PROTOCOL section.
    """
    body = _filter_rules_fragment(rules_md, phase=phase, transport=transport) or (
        "(orchestration.md rules fragment not found — Coordinator will still enforce PolicyGate hard rules at runtime.)"
    )
    return ["## 7. RULES & OUTPUT PROTOCOL", "", body]


def _section_cycle_directive(*, macro_cycle: int = 0, cycle_directive: str = "") -> list[str]:
    """Build the CYCLE DIRECTIVE section.

    When ``cycle_directive`` is non-empty it carries an LLM-authored focus
    mandate for this macro-cycle (see ``orchestration_memory.next_cycle_directive``).
    Otherwise the standing breadth→depth arc is used as the default.

    Args:
        macro_cycle: Current macro-cycle counter; shown verbatim.
        cycle_directive: Optional LLM-authored focus text for this cycle.

    Returns:
        list[str]: Markdown lines for the section.
    """
    lines = [
        "## CYCLE DIRECTIVE (advisory — this macro-cycle's focus)",
        "",
        f"macro_cycle={int(macro_cycle)}. Live cycle number is in the"
        " ``cycle`` line of the per-tick ``=== Phase ===`` block.",
        "The machinery already (a) decays the KEEP threshold each cycle and"
        " (b) amplifies specialist wall budgets — plan with that arc.",
        "",
    ]
    if cycle_directive and cycle_directive.strip():
        lines.append("Focus for this cycle (LLM-authored at prior cycle boundary):")
        lines.append(cycle_directive.strip())
    else:
        lines.extend(
            [
                "Default arc (no per-cycle directive yet):",
                "- Early cycles (≈0-2): cast WIDE — many cheap config/env levers and"
                "  several specialists in parallel to map the space fast.",
                "- Later cycles: FEWER, DEEPER, longer-running specialist tasks —"
                "  spend the amplified budget on autotune / kernel / profiling-driven"
                "  work that needs a long measure→edit→measure loop.",
            ]
        )
    return lines


_WHEN_TAG_RE = re.compile(r"^<!--\s*when:\s*(?P<when>.+?)\s*-->$")


def _section_reference_index(*, references_dir: Path, phase: str = "") -> list[str]:
    """Build ``## 8.`` from the reference docs that apply to *phase*.

    Args:
        references_dir: Directory containing the reference markdown files.
        phase: Normalised current pipeline phase; ``""`` includes all entries.

    Returns:
        Markdown lines, or ``[]`` when the directory is absent or empty.
    """
    if not references_dir.is_dir():
        return []
    entries: list[tuple[str, str]] = []
    for path in sorted(references_dir.glob("*.md")):
        when_text = ""
        file_phases: frozenset[str] = frozenset()
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith(">"):
                continue
            when_m = _WHEN_TAG_RE.match(stripped)
            if when_m:
                when_text = when_m.group("when")
                continue
            phase_m = _PHASE_TAG_RE.match(stripped)
            if phase_m:
                file_phases = frozenset(p.strip().upper() for p in phase_m.group("phases").split(",") if p.strip())
                continue
            break
        if file_phases and not _renders_in(phase, file_phases):
            continue
        entries.append((path.stem, when_text or "see document"))
    if not entries:
        return []
    lines: list[str] = [
        "## 8. ON-DEMAND REFERENCE INDEX",
        "",
        "Use ``read_reference(name='<name>')`` to pull the full document.",
        "",
    ]
    for stem, when in entries:
        lines.append(f"- **{stem}** — {when}")
    return lines


def build_orchestration_prompt(
    *,
    action_registry: Mapping[str, ActionMetadata],
    enabled_actions: Iterable[str],
    framework: str = "sglang",
    kernel_enabled: bool | None = None,
    framework_agent_phase_enabled: bool = True,
    objective_kind: str = "time_only",
    objective_value: float | str | None = None,
    max_minutes: int = 0,
    macro_cycle: int = 0,
    cycle_directive: str = "",
    phase: str = "",
    transport: str = TRANSPORT_TOOLS,
    rules_fragment_path: Path | None = None,
    framework_source_roots: tuple[str, ...] | None = None,
    references_dir: Path | None = None,
) -> str:
    """Compose the Orchestration system prompt (deterministic for given inputs).

    Args:
        action_registry: the ``ACTION_CATALOGUE`` mapping.
        enabled_actions: enabled action names; final ordering is by
            pipeline_phase.
        framework: ``sglang`` / ``vllm`` — printed in SESSION CONTEXT.
        kernel_enabled: explicit override; ``None`` derives from KERNEL_OWNED
            actions.
            skipped; the prompt annotates it as DISABLED so Orchestration's plan
            matches the real phase chain.
        framework_agent_phase_enabled: when False (``--no-framework-agent``) the
            FRAMEWORK_AGENT phase is skipped; annotated DISABLED in the prompt.
        objective_kind: :mod:`objective` kind string, printed verbatim.
        objective_value: :mod:`objective` target value, printed verbatim.
        max_minutes: wall-clock budget for the run.
        macro_cycle: current macro-cycle counter; shown in the CYCLE DIRECTIVE
            section.
        cycle_directive: optional LLM-authored focus text for this cycle
            (from ``orchestration_memory.next_cycle_directive``); empty string
            renders the standing breadth→depth default.
        phase: current pipeline phase; omits the modules whose behaviour it
            cannot reach. Empty renders every module. The Coordinator rebuilds
            the prompt at each phase seam.
        transport: one of :data:`TRANSPORTS`, taken from the backend that will
            actually run the role. Omits the modules describing a tool surface
            that transport does not mount.
        rules_fragment_path: path to ``orchestration.md``; placeholder if
            unreadable.
        framework_source_roots: optional framework source roots passed through
            to the session-context section.
        references_dir: directory of on-demand reference documents; defaults
            to ``asset_prompt_references_dir()`` when ``None``.

    Returns:
        The composed Orchestration system prompt text.

    Raises:
        ValueError: If ``transport`` is neither empty nor one of
            :data:`TRANSPORTS`. Raised rather than tolerated because the caller
            builds this at start-up, and an unknown transport silently strips
            the Output protocol instead of failing.
    """
    # Checked here rather than tolerated downstream: a transport nobody declares
    # matches no `<!-- transport: ... -->` block, and both Output protocol blocks
    # are scoped by one, so an unknown value renders a prompt that never tells
    # the model how to answer. Nothing later in the pipeline can tell that apart
    # from a fragment that simply had nothing to say.
    if transport and transport not in TRANSPORTS:
        raise ValueError(f"unknown prompt transport {transport!r}; expected one of {', '.join(sorted(TRANSPORTS))}")
    actions, kernel_enabled, framework_norm, rules_md = _resolve_prompt_prelude(
        action_registry,
        enabled_actions,
        framework,
        kernel_enabled,
        rules_fragment_path,
    )
    phase_norm = (phase or "").strip().upper()

    if references_dir is None:
        from hyperloom.inference_optimizer.session.paths import asset_prompt_references_dir

        references_dir = asset_prompt_references_dir()

    sections: list[list[str]] = [
        _section_mission(),
        _section_session_context(
            framework=framework_norm,
            kernel_enabled=kernel_enabled,
            objective_kind=objective_kind,
            objective_value=objective_value,
            max_minutes=max_minutes,
            framework_agent_phase_enabled=framework_agent_phase_enabled,
            framework_source_roots=framework_source_roots,
        ),
        _section_pipeline_and_budget(actions, max_minutes=max_minutes),
        _section_phase_semantics(
            kernel_enabled=kernel_enabled,
            framework_agent_phase_enabled=framework_agent_phase_enabled,
        ),
        _section_action_catalogue(actions, phase=phase_norm),
        _section_decision_framework(kernel_enabled=kernel_enabled, phase=phase_norm, transport=transport),
        _section_cycle_directive(macro_cycle=macro_cycle, cycle_directive=cycle_directive),
    ]
    if (
        kernel_enabled
        and any(a.name == "kernel_opt" for a in actions)
        and _renders_in(phase_norm, _KERNEL_REQUEST_PHASES)
    ):
        sections.append(_KERNEL_OPT_PIPELINE_BODY.splitlines())
    # The reference index is an index of documents ``read_reference`` pulls;
    # without that tool it is a list the model cannot act on.
    if transport != TRANSPORT_STRUCTURED_OUTPUT:
        ref_index = _section_reference_index(references_dir=references_dir, phase=phase_norm)
        if ref_index:
            sections.append(ref_index)
    sections.append(_section_rules(rules_md, phase=phase_norm, transport=transport))

    return join_sections(sections)


def default_enabled_actions(
    *,
    no_kernel: bool,
    no_optimize: bool = False,
) -> tuple[str, ...]:
    """Return the canonical enabled-action set used by the CLI.

    Filters :data:`FULL_ENABLED_ACTIONS` per flag so the flags compose: a
    ``--no-kernel --no-framework-agent`` run drops both kernel_agent-owned
    names and the ``explore`` grid-runner.

    Args:
        no_kernel (bool): When ``True``, drop the kernel-only actions (keep the
            intersection with :data:`NO_KERNEL_AGENT_ENABLED_ACTIONS`).
        no_optimize (bool): When ``True``, drop the ``explore`` grid-runner
            action: the phase that dispatches it is skipped.

    Returns:
        tuple[str, ...]: The filtered enabled-action set, preserving
        :data:`FULL_ENABLED_ACTIONS` ordering.
    """
    actions = list(FULL_ENABLED_ACTIONS)
    if no_kernel:
        actions = [a for a in actions if a in NO_KERNEL_AGENT_ENABLED_ACTIONS]
    if no_optimize:
        actions = [a for a in actions if a != "explore"]
    return tuple(actions)


__all__ = [
    "FULL_ENABLED_ACTIONS",
    "KERNEL_AGENT_OWNED_ACTIONS",
    "NO_KERNEL_AGENT_ENABLED_ACTIONS",
    "TRANSPORTS",
    "TRANSPORT_STRUCTURED_OUTPUT",
    "TRANSPORT_TOOLS",
    "build_orchestration_prompt",
    "default_enabled_actions",
]
