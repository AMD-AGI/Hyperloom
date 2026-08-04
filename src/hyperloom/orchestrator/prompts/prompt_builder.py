# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Compose the Orchestration agent's system prompt from typed inputs.

Wraps the ``orchestration.md`` rules fragment with generated sections
(mission, session context, pipeline/budget, phase contract, action catalogue,
decision framework, cycle directive, optional kernel-opt reference, rules).
Deterministic for given inputs; the only IO is reading the rules fragment.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from ..actions.registry import (
    ActionMetadata,
    ActionRegistry,
    VALID_PIPELINE_PHASES,
)
from hyperloom.inference_optimizer.protocol.action_surfaces import (
    FULL_ENABLED_ACTIONS,
    GRID_INJECTABLE_ACTIONS,
    KERNEL_AGENT_OWNED_ACTIONS,
    NO_KERNEL_AGENT_ENABLED_ACTIONS,
)
from . import read_rules_fragment as _read_rules_fragment


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
        "Your single most important goal is to maximise the run's **cumulative_gain**",
        "(percent over baseline_tput) within the wall-clock budget.",
        "",
        "Every tick, ask yourself:",
        '  "Given current SharedState, remaining time, and the action catalogue below,',
        '   which next action gives the highest expected_gain / cost_minutes?"',
        "",
        'An optimization is only "real" once it has been validated as part of the',
        "full optimization_stack. ``explore`` inlines a per-KEEP stack rebench, so",
        "the validated cumulative gain advances automatically — sums of per-round",
        "gains still do NOT compose linearly, so drive the loop until ``explore``",
        "has produced at least one KEEP that survived the stack rebench.",
    ]


def _section_session_context(
    *,
    framework: str,
    kernel_enabled: bool,
    objective_kind: str,
    objective_value: float | str | None,
    max_minutes: int,
    explore_enabled: bool = True,
    framework_agent_phase_enabled: bool = True,
    framework_source_roots: tuple[str, ...] | None = None,
) -> list[str]:
    """Build the SESSION CONTEXT section lines.

    Args:
        framework (str): The framework name shown verbatim.
        kernel_enabled (bool): Whether kernel_agent-owned actions are enabled.
        explore_enabled (bool): Whether the EXPLORE phase is enabled.
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
        f"- explore_enabled  : {'true' if explore_enabled else 'false'}",
        f"- framework_agent_phase_enabled : {'true' if framework_agent_phase_enabled else 'false'}",
        f"- objective        : {obj}",
        f"- max_minutes      : {max_minutes}",
        f"- framework_source_roots: {roots_line}",
        "",
        "Per-tick dynamic context (Phase, Mission progress, Time budget,",
        "Shared session state, KB hints, inbox tail) is appended below the",
        "system prompt every tick by the Coordinator.",
        "The Time-budget block carries `remaining=X.Xmin`.",
        "See PHASE CONTRACT below for the 6-phase chain, per-phase allowed",
        "actions, and phase-transition rules.",
    ]


def _section_phase_semantics(
    *,
    kernel_enabled: bool,
    explore_enabled: bool = True,
    framework_agent_phase_enabled: bool = True,
) -> list[str]:
    """Render the per-phase LLM-proposable action contract (current phase
    injected dynamically by the Coordinator).

    Phases switched off by ``--no-explore`` / ``--no-kernel`` /
    ``--no-framework-agent`` keep their row in the 6-phase chain but are annotated
    ``(DISABLED: --no-xxx — phase skipped)`` so Orchestration plans against the
    phases the run will actually enter.

    Args:
        kernel_enabled: Whether kernel_agent-owned actions are enabled.
        explore_enabled: Whether the EXPLORE phase is enabled.
        framework_agent_phase_enabled: Whether the FRAMEWORK_AGENT phase is enabled.

    Returns:
        Markdown lines for the phase-contract section.
    """
    from ..phases.machine_state import render_phase_proposable_bullets

    # phase name -> the flag that disabled it (None => always enabled).
    disabled_suffix: dict[str, str] = {}
    if not framework_agent_phase_enabled:
        disabled_suffix["FRAMEWORK_AGENT"] = "--no-framework-agent"
    if not explore_enabled:
        disabled_suffix["EXPLORE"] = "--no-explore"
    if not kernel_enabled:
        disabled_suffix["KERNEL_AGENT"] = "--no-kernel"

    lines: list[str] = [
        "## 3a. PHASE CONTRACT",
        "",
        "The Coordinator runs the optimization in a 6-phase linear pipeline.",
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
            "roofline, profile, replay_warm_recipe and framework are never",
            "in the sets above: the Coordinator auto-manages them and PolicyGate",
            "denies any attempt to propose them. Denial of any action lands in",
            "your inbox as a `policy_denied` event.",
            "",
            "Phase transitions are Coordinator-owned. The hard advance gates",
            "are: `baseline_tput > 0` exits PRELUDE; IR-6 force-exit, the per-",
            "phase budget cap, or a terminal stop_reason exit EXPLORE / KERNEL_AGENT",
            "/ SWEEP; the wall-clock deadline (closing phase) routes to CLOSE.",
            "You may also emit `escalate_strategy_change{next_action_hint=",
            "'skip_to_kernel' | 'skip_to_sweep' | 'skip_to_close'}` directly",
            "(no longer robustness-only) when you judge the current phase",
            "exhausted; the Coordinator validates the hint vocab and routes",
            "the transition on the next tick.",
            "EXCEPTION — normal SWEEP convergence: do NOT emit `skip_to_close`",
            "once the sweep has completed (sweep_done / conc_sweep_done). The",
            "Coordinator exits SWEEP → CLOSE on its own with an honest terminal",
            "stop_reason (`sweep_done` / `global_converged`). `skip_to_close`",
            "is reserved for genuine early abandonment (e.g. infra is dead and",
            "the sweep cannot run at all) — it stamps `robustness_escalated`,",
            "so emitting it on a normal finish mislabels the run.",
        ]
    )
    return lines


def _filter_actions(
    registry: ActionRegistry,
    enabled: Iterable[str],
) -> list[ActionMetadata]:
    """Resolve enabled action names to their registry metadata.

    Args:
        registry (ActionRegistry): The loaded action registry to look up.
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
    action_registry: ActionRegistry,
    enabled_actions: Iterable[str],
    framework: str,
    kernel_enabled: bool | None,
    rules_fragment_path: Path | None,
) -> tuple[list[ActionMetadata], bool, str, str]:
    """Resolve the shared prelude for the orchestration / critic prompt builders.

    Args:
        action_registry (ActionRegistry): The loaded action registry.
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
        if meta.name == "kernel_opt":
            kind_hint = "run_optimization"
        elif meta.name == "gemm_tuning":
            kind_hint = "run_gemm_tuning"
        elif meta.name == "integrate":
            kind_hint = "integrate"
        else:
            kind_hint = meta.name
        return f"REQUEST{{target_agent='kernel_agent', kind='{kind_hint}', params={{...}}}}"
    if meta.name == "report":
        return "propose_action{action_name='report', predicted_gain_pct=0.0}"
    if meta.name == "specialist":
        return (
            "delegate{action_name='specialist', params={"
            "domain=<one of serving_specialist|kernel_switch_specialist|"
            "comm_specialist|compiler_specialist|system_specialist|"
            "pr_intel_specialist|research_scout_specialist|"
            "static_recon_specialist>, "
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
            "keep_threshold_pct?=1.0, "
            "accuracy_baseline?=<float>}}"
        )
    return f"propose_action{{action_name='{meta.name}', predicted_gain_pct=<your estimate>}}"


def _format_grid_injection_hint(name: str) -> str | None:
    """Return a per-action one-liner showing how to override grid, or None.

    Args:
        name: The action name to render a grid-injection hint for.

    Returns:
        The grid-injection hint string for ``explore`` / ``sweep``, or ``None``
        for any other action.
    """
    if name == "explore":
        return (
            "GRID INPUT (REQUIRED): emit "
            "`delegate{action_name='explore', params={grid: [{name, "
            "extra_args, extra_envs, remove_args?, unset_envs?, "
            "args_mode?: 'append'|'replace', provenance, kb_evidence?, "
            "pr_evidence?, source_evidence?}, ...], "
            "base_extra_args?, base_tput?, accuracy_baseline?, "
            "keep_threshold_pct?: 1.0, stack_stable_threshold_pct?: 0.5}}`. "
            "Variants run serially; each KEEP triggers an inlined stack "
            "rebench. Variant identity is content-based (args+envs+"
            "remove_args+unset_envs+args_mode) — rename alone does NOT "
            "bypass dedup. Use remove_args/unset_envs to ablate harmful "
            "base flags; args_mode='replace' to drop inherited server args. "
            "provenance values: 'llm_direct', 'default_grid', "
            "'specialist:<domain-or-tag>' (audit/advisory, not a gate)."
        )
    if name == "sweep":
        return (
            "GRID OVERRIDE: emit `delegate{action_name='sweep', params={grid: "
            "[{conc, isl, osl}, ...]}}` (or params.conc_values / "
            "params.isl_osl_values) to override the workload frontier."
        )
    return None


def _section_action_catalogue(actions: list[ActionMetadata]) -> list[str]:
    """Build the ACTIONS YOU MAY USE catalogue section, grouped by phase.

    Each entry lists cost, gain, risks, family, the emit hint, and any
    grid-injection hint.

    Args:
        actions (list[ActionMetadata]): The actions enabled for this run.

    Returns:
        list[str]: Markdown lines for the action catalogue.
    """
    lines: list[str] = [
        "## 4. ACTIONS YOU MAY USE",
        "",
        "Catalogue is filtered to the actions enabled for this run. Each entry",
        "carries: phase / typical wall-clock / expected gain range / accuracy_risk /",
        "crash_risk / one-line description / how to emit.",
        "",
    ]
    by_phase = _phase_eta_summary(actions)
    name_to_meta = {a.name: a for a in actions}
    for phase, _eta, names in by_phase:
        lines.append(f"### {phase}")
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
            lines.append(f"    EMIT: {_format_emit_hint(meta)}")
            grid_hint = _format_grid_injection_hint(name)
            if grid_hint:
                lines.append(f"    {grid_hint}")
        lines.append("")
    return lines


def _section_decision_framework(*, kernel_enabled: bool) -> list[str]:
    """Build the DECISION FRAMEWORK section lines.

    Covers the per-tick selection order, failure-recovery rules (F1–F4), and
    idea-generation guidance.

    Args:
        kernel_enabled (bool): Whether kernel_agent-owned actions are enabled for this
            run.

    Returns:
        list[str]: Markdown lines for the decision framework.
    """
    lines = [
        "## 5. DECISION FRAMEWORK (heuristics + facts — the next action is your call)",
        "",
        "These are reference heuristics and objective facts, not a forced",
        "sequence. Read the dynamic SharedState section and decide:",
        "",
        "1. **Stop**: if `stop_reason` is set OR `cumulative_gain >= target_gain_pct`,",
        "   propose `report` once (if not already done) then heartbeat 'goal-reached'.",
        "2. **Measure**: if `baseline_tput == 0`, propose `baseline`. Wait for",
        "   delegated_result; do NOT re-baseline on a positive result with warnings.",
        "3. **Inlined stack-rebench**: route every grid attempt through",
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
            "   d. **specialist proposal_set** — when an explore round just",
            "      finished, the proposal_set drives the next `explore` grid.",
            "   e. **Ordering facts**: baseline runs before anything else",
            "      (invariant). ``analysis.md`` / ``last_profile_trace`` arrive",
            "      automatically from the Coordinator-owned analysis task at",
            "      PRELUDE and at every +10% watermark crossing — you do not",
            "      need a manually-proposed profile before ``kernel_opt``.",
            "6. **Phase budget awareness**. The `=== Phase ===` block carries",
            "   ``phase_budget_remaining_pct``. As that number falls below 0.2,",
            "   prefer lower-cost / known-good actions (explore over kernel_opt).",
            "   The Plateau advisory block is informational only for KERNEL /",
            "   FRAMEWORK plateaus (they never auto-advance the phase); a detected",
            "   EXPLORE plateau, by contrast, deterministically advances",
            "   EXPLORE -> KERNEL_AGENT (``reason=explore_no_more_leverage``) at the",
            "   next phase-compute. When you judge the current phase exhausted,",
            "   emit ``escalate_strategy_change{next_action_hint=",
            "   'skip_to_kernel' | 'skip_to_sweep' | 'skip_to_close'}`` (see",
            "   PHASE CONTRACT for the skip_to_close exception).",
            "",
            "If you cannot move forward, emit",
            "`send_message{topic='heartbeat', body_md='blocked: <reason>'}` and let",
            "Robustness escalate. NEVER stay silent.",
            "",
            "### FAILURE RECOVERY (apply BEFORE re-proposing an action that just failed)",
            "",
            "When the inbox carries a fresh `delegated_result{state!='succeeded'}`",
            "or `last_action_failures[-1].action == <X>`, do NOT re-propose the same",
            "action with the same params. Coordinator stamps an audit trail on every",
            "attempt; consult these three SharedState surfaces in order:",
            "",
            "1. **`last_<action>`** (snapshot of the latest attempt) and",
            "   **`<action>_attempts`** (capped per-action history, newest last).",
            "   Each entry carries `status` / `decision` / `error_class` /",
            "   `error_excerpt` / `workspace` / `raw_result_path` /",
            "   `extras.fingerprint`. The fingerprint is the canonical hash of",
            "   the eight params fields that determine baseline behavior:",
            "   `benchmark_script` / `result_dir` / `extra_server_args` /",
            "   `extra_envs` / `model_path` / `gpu_type` / `config_path` /",
            "   `disable_run_eval`. Those eight are the WHOLE fingerprint —",
            "   any other key you add to `params` (a tag, a note, a counter) is",
            "   invisible to it and does NOT make the proposal distinct.",
            "   Coordinator keys task idempotency on that same fingerprint, so a",
            "   proposal matching a queued or running task is dropped before it",
            "   ever reaches the executor.",
            "   A baseline may also come back `status='succeeded'` with",
            "   `decision='no_promote'` and `extras.anchor_kept_tput`: it ran fine",
            "   but measured below the established anchor, so the anchor was kept.",
            "   That is a completed measurement, NOT a failure — do not retry it.",
            "2. **`last_action_failures`** (global rolling log capped at the last",
            "   10 unpromotable results across ALL kinds, including kernel_agent-owned).",
            "   Use this when the per-action history doesn't carry the kind you",
            "   need (e.g. `integrate` failure visible here but `<action>_attempts`",
            "   only covers the six explore/validate kinds).",
            "3. **`baseline_failure_streak`** (consecutive failed baselines).",
            "   Once this hits 3, Coordinator sets `stop_reason='baseline_failed'`",
            "   and the run terminates; you must recover BEFORE the third failure.",
            "",
            "Three decision rules apply in order:",
            "",
            "* **RULE F1 — same fingerprint, twice failed → change at least one",
            "  of the eight fingerprint fields.** Because you run as a persistent",
            "  conversation you remember the attempts you already made this",
            "  session — do not re-propose a baseline whose params fingerprint",
            "  matches a recent failure; it will fail the same way. Changing a",
            "  field OUTSIDE the eight does not count and does not get you a new",
            "  attempt: the proposal is dropped as a duplicate and you have burned",
            "  a tick. Bump at least one of: `params.benchmark_script`",
            "  (a sanitized `*.sh` file name that MUST match THIS run's framework —",
            "  e.g. for a vllm run pin `vllm_mi300x.sh`, never `sglang_*`; a",
            "  cross-framework script boots the wrong engine and is rejected),",
            "  `params.result_dir`, `params.extra_server_args`, or",
            "  `params.extra_envs`.",
            "* **RULE F2 — `error_class='no_report'` + no `rescued_from_leaked_path:*`",
            "  warning ⇒ leak salvage missed.** The script wrote results outside the",
            "  workspace and outside the configured leak destinations. Override",
            "  `params.benchmark_script` to a script that respects `$RESULT_DIR`",
            "  (Coordinator already exports `RESULT_DIR=<workspace>` by default), or",
            "  set `$INFERENCE_OPTIMIZER_RESCUE_PATHS` via `update_state` so the next",
            "  attempt salvages the leak.",
            "* **RULE F3 — `error_class='subprocess_nonzero'` with same fingerprint",
            "  ⇒ stop retrying.** Heartbeat with `body_md='blocked: subprocess",
            "  repeatedly nonzero <action>'` and let Robustness intervene. Do NOT",
            "  switch action families just to dodge the failure; Robustness'",
            "  escalation policy needs the heartbeat to fire its RCA.",
            "* **RULE F4 — `policy_denial_streak` is a pure fact, not a lock.** The",
            "  `why_denied` context tool (and the `Recent policy denials` block on a",
            "  seed turn) shows repeated (action, rule) collisions. The system no",
            "  longer reacts to the streak — there is NO auto-prune at streak≥5 and",
            "  NO `policy_loop` stop at streak≥10; the run continues until the",
            "  wall-clock deadline or another stop_reason fires. So the streak is",
            "  purely a signal for YOU: the same params keep colliding with the same",
            "  invariant, so change something substantive — a new `params.grid`",
            "  variant, a different `benchmark_script`, or a sibling action family.",
            "  Re-emitting the identical denied intent just wastes a tick.",
            "",
            "Example (baseline failed twice with `error_class='no_report'`):",
            "",
            "    # Prefer RESULT_DIR rescue first; only override benchmark_script",
            "    # when you have verified a same-framework script-specific bug.",
            "    propose_action{action_name='baseline',",
            "        params={result_dir: '<session_dir>/runs/baseline/<task>/leak'},",
            "        predicted_gain_pct: 0,",
            "        notes: 'recover from no_report streak by redirecting RESULT_DIR",
            "                to the observed leak location'}",
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
            "3. **Retry rejects** — for each `explore_search.rejected` variant,",
            "   change the value or pair it with a winner (a `-2%` reject is a",
            "   dead flag; `-0.3%` just needs a different value).",
            "4. **Mine flags** — when winners are empty, pull untested boolean",
            "   toggles from `discovered_flags.<framework>.backend_flags`.",
            "5. **Ablate harmful base config** — when a user/base flag or env may",
            "   be slowing the workload, emit a variant with `remove_args` and/or",
            "   `unset_envs` instead of only adding more knobs.",
            "",
            "Variant identity is content-based (args+envs+remove_args+",
            "unset_envs+args_mode) — rename alone does NOT bypass dedup.",
            "`extra_server_args` is framework-neutral (routed to EXTRA_SGLANG_ARGS",
            "/ EXTRA_VLLM_ARGS / EXTRA_ATOM_ARGS by `--framework`).",
            "",
            "An explore round that produces zero new ideas is a bug — heartbeat",
            "with body_md='idea-pipeline-empty' so Robustness can intervene.",
        ]
    )
    return lines


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
                   candidates_path: <trace_analyze_done.candidates_path>,
                   budget_minutes: 60}}

  Backend policy: DO NOT add a `backends` field. Current GEAK owns the
  KERNEL phase by default. Forge per-kernel mode is available only when the
  operator set exactly `KERNEL_OPT_BACKEND_ORDER=forge`.
  Read `kernel_opt_attempts` +
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
  `kernel_opt_attempts` ledger (this is what drains a multi-KEEP queue).
  PARTIAL / REVERT → do NOT integrate; pick the next action normally.

  **Multi-KEEP queue:** `pending_keep_kernels` (sorted strongest-first)
  lists queued KEEPs; integrate `[0]` each tick. Do NOT propose `report`
  while it is non-empty, nor while `untried_hot_reusable_kernels`
  (reusable hot kernels with zero attempts and `gpu_pct >= 10%`, the
  default that `HYPERLOOM_KERNEL_OPT_MIN_GPU_PCT` overrides) remain —
  drain them with `run_optimization{candidates_path: <from
  last_trace_analyze>}` (the batch handler fans out automatically).

### KERNEL TARGETING

Only rewrite reusable native sources in the trace. NEVER optimize
`/tmp/torchinductor*` / `triton_poi_*` / `triton_red_*` runtime-generated
kernels — they're tied to one compile cache and not reusable."""


def _section_rules(rules_md: str) -> list[str]:
    """Build the RULES & OUTPUT PROTOCOL section wrapping the rules fragment.

    Args:
        rules_md (str): The raw rules-fragment markdown; a placeholder is used
            when empty.

    Returns:
        list[str]: Markdown lines for the RULES & OUTPUT PROTOCOL section.
    """
    body = rules_md.strip() or (
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
        lines.extend([
            "Default arc (no per-cycle directive yet):",
            "- Early cycles (≈0-2): cast WIDE — many cheap config/env levers and"
            "  several specialists in parallel to map the space fast.",
            "- Later cycles: FEWER, DEEPER, longer-running specialist tasks —"
            "  spend the amplified budget on autotune / kernel / profiling-driven"
            "  work that needs a long measure→edit→measure loop.",
        ])
    return lines


def build_orchestration_prompt(
    *,
    action_registry: ActionRegistry,
    enabled_actions: Iterable[str],
    framework: str = "sglang",
    kernel_enabled: bool | None = None,
    explore_enabled: bool = True,
    framework_agent_phase_enabled: bool = True,
    objective_kind: str = "time_only",
    objective_value: float | str | None = None,
    max_minutes: int = 0,
    macro_cycle: int = 0,
    cycle_directive: str = "",
    rules_fragment_path: Path | None = None,
    framework_source_roots: tuple[str, ...] | None = None,
) -> str:
    """Compose the Orchestration system prompt (deterministic for given inputs).

    Args:
        action_registry: pre-loaded ``ActionRegistry`` (caller calls
            ``.load()``).
        enabled_actions: enabled action names; final ordering is by
            pipeline_phase.
        framework: ``sglang`` / ``vllm`` — printed in SESSION CONTEXT.
        kernel_enabled: explicit override; ``None`` derives from KERNEL_OWNED
            actions.
        explore_enabled: when False (``--no-explore``) the EXPLORE phase is
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
        rules_fragment_path: path to ``orchestration.md``; placeholder if
            unreadable.
        framework_source_roots: optional framework source roots passed through
            to the session-context section.

    Returns:
        The composed Orchestration system prompt text.
    """
    actions, kernel_enabled, framework_norm, rules_md = _resolve_prompt_prelude(
        action_registry,
        enabled_actions,
        framework,
        kernel_enabled,
        rules_fragment_path,
    )

    sections: list[list[str]] = [
        _section_mission(),
        _section_session_context(
            framework=framework_norm,
            kernel_enabled=kernel_enabled,
            objective_kind=objective_kind,
            objective_value=objective_value,
            max_minutes=max_minutes,
            explore_enabled=explore_enabled,
            framework_agent_phase_enabled=framework_agent_phase_enabled,
            framework_source_roots=framework_source_roots,
        ),
        _section_pipeline_and_budget(actions, max_minutes=max_minutes),
        _section_phase_semantics(
            kernel_enabled=kernel_enabled,
            explore_enabled=explore_enabled,
            framework_agent_phase_enabled=framework_agent_phase_enabled,
        ),
        _section_action_catalogue(actions),
        _section_decision_framework(kernel_enabled=kernel_enabled),
        _section_cycle_directive(macro_cycle=macro_cycle, cycle_directive=cycle_directive),
    ]
    if kernel_enabled and any(a.name == "kernel_opt" for a in actions):
        sections.append(_KERNEL_OPT_PIPELINE_BODY.splitlines())
    sections.append(_section_rules(rules_md))

    return join_sections(sections)


def default_enabled_actions(
    *,
    no_kernel: bool,
    no_explore: bool = False,
) -> tuple[str, ...]:
    """Return the canonical enabled-action set used by the CLI.

    Filters :data:`FULL_ENABLED_ACTIONS` per flag so the flags compose: a
    ``--no-kernel --no-explore`` run drops both kernel_agent-owned names and the
    ``explore`` grid-runner. ``--no-framework-agent`` is intentionally absent — the
    ``framework`` action is Coordinator-internal and never appears in the
    catalogue, so it has nothing to trim.

    Args:
        no_kernel (bool): When ``True``, drop the kernel-only actions (keep the
            intersection with :data:`NO_KERNEL_AGENT_ENABLED_ACTIONS`).
        no_explore (bool): When ``True``, drop the ``explore`` grid-runner
            action (EXPLORE phase is skipped).

    Returns:
        tuple[str, ...]: The filtered enabled-action set, preserving
        :data:`FULL_ENABLED_ACTIONS` ordering.
    """
    actions = list(FULL_ENABLED_ACTIONS)
    if no_kernel:
        actions = [a for a in actions if a in NO_KERNEL_AGENT_ENABLED_ACTIONS]
    if no_explore:
        actions = [a for a in actions if a != "explore"]
    return tuple(actions)


__all__ = [
    "FULL_ENABLED_ACTIONS",
    "GRID_INJECTABLE_ACTIONS",
    "KERNEL_AGENT_OWNED_ACTIONS",
    "NO_KERNEL_AGENT_ENABLED_ACTIONS",
    "VALID_PIPELINE_PHASES",
    "build_orchestration_prompt",
    "default_enabled_actions",
]
