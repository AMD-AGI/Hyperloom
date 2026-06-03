"""Compose the Orchestration agent's system prompt from typed inputs.

Replaces the previous hand-maintained pair ``orchestration.md`` /
``orchestration.no_kernel.md``:

* ``orchestration.md`` is now only the *rules fragment* (HARD RULES,
  SESSION_DIR contract, output protocol) — content that doesn't depend
  on which actions are enabled.

* This module wraps that fragment with:
    1. MISSION                           (always)
    2. SESSION CONTEXT                   (objective / framework / max_minutes)
    3. PIPELINE & TIME BUDGET            (phase ordering + per-phase ETA)
    4. ACTIONS YOU MAY USE               (filtered by enabled_actions)
    5. DECISION FRAMEWORK                (always — short, generic)
    6. KERNEL-OPT REQUEST REFERENCE      (only when "kernel_opt" is enabled)
    7. RULES FRAGMENT                    (orchestration.md verbatim)

Output is deterministic given the same inputs (sorted action listing,
fixed section order). The CLI snapshots the output to
``agents/orchestration/system_prompt.snapshot.md`` for resume / audit.

The builder is a pure function: no IO besides reading the rules
fragment, no env access, no logging side effects. This makes it
trivially testable and easy to use as a one-shot CLI introspection
tool (see ``scripts/print_orchestration_prompt.py`` if added later).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from ..action_registry import (
    ActionMetadata,
    ActionRegistry,
    VALID_PIPELINE_PHASES,
)


# ---------------------------------------------------------------------------
# Default enabled-action sets (mirrors `cli._register_executors`)
# ---------------------------------------------------------------------------
FULL_ENABLED_ACTIONS: tuple[str, ...] = (
    # prep
    "target_analysis", "baseline",
    # analysis — roofline is Coordinator-auto-enqueued (PRELUDE + watermark);
    # not LLM-proposable. deep_kernel_analysis stays as a kernel-owned REQUEST.
    "roofline", "deep_kernel_analysis",
    # explore — the single grid-runner entry (per-KEEP stack rebench
    # inlined).
    "explore",
    # PR-A1 (Arbor-into-Hyperloom): ``specialist`` is the LLM sub-agent
    # dispatch surface; ``integrate_patch`` is the orchestrator-side
    # apply+restart+gate that consumes specialist worktree patches.
    # Both live under pipeline_phase=explore in the registry.
    "specialist",
    # Supplementary cross-domain ReAct sub-agent channel; EXPLORE-only,
    # round-cap 1, sits next to specialist in the catalogue.
    "dynamic_action",
    "integrate_patch",
    "sweep",
    # deep — kernel-owned, emitted via REQUEST{target_agent='kernel', kind=...}
    "kernel_opt", "integrate", "operator_tuning", "vendor_kernel_config",
    "gemm_tuning",
    # finalize
    "report",
    # support — ``recover`` frees leaked VRAM and (when
    # ``HYPERLOOM_RECOVER_ALLOW_GPU_RESET=1``) attempts
    # ``rocm-smi --gpureset``. The replacement path for diagnostic work
    # is a specialist sub-agent.
    "recover",
)

NO_KERNEL_ENABLED_ACTIONS: tuple[str, ...] = (
    # prep
    "target_analysis", "baseline",
    # explore (no profile — it only feeds kernel-opt). Carries an
    # inlined per-KEEP stack rebench.
    "explore",
    # PR-A1 (Arbor-into-Hyperloom): specialist + integrate_patch are
    # always-on; they are EXPLORE-phase actions and unrelated to kernel mode.
    "specialist",
    "dynamic_action",
    "integrate_patch",
    "sweep",
    # finalize
    "report",
    # support — recover is needed even without kernel-opt because GPU
    # leaks from baseline / explore / sweep can still hang the session;
    # the executor itself is kernel-agnostic.
    "recover",
)

# Actions that the Kernel agent owns end-to-end (Plan A). Orchestration MUST
# emit `request{target_agent='kernel', kind=...}` for these instead of
# `delegate{action_name=...}`. We highlight the difference in the catalogue
# section so the LLM picks the right transport.
KERNEL_OWNED_ACTIONS: frozenset[str] = frozenset({
    "kernel_opt", "integrate", "deep_kernel_analysis",
    "operator_tuning", "vendor_kernel_config", "gemm_tuning",
})

# Actions that accept LLM-injected grid candidates via ``params.grid``.
# The catalogue section appends a grid-override hint for these so the
# LLM knows it can expand the search space beyond the shipped defaults.
GRID_INJECTABLE_ACTIONS: frozenset[str] = frozenset({
    "explore", "sweep",
})

# Phase ordering for the catalogue section. Any action whose pipeline_phase
# is not in this tuple is appended at the end (defensive; current registry
# fully covers the set).
_PHASE_ORDER: tuple[str, ...] = (
    "prep", "measure", "analysis", "explore", "deep", "validate",
    "finalize", "support",
)
_PHASE_HEADERS: dict[str, str] = {
    "prep":     "Prep — initialise session metadata. Always finishes first.",
    "measure":  "Measure — establish baseline_tput. Gate for everything else.",
    "analysis": "Analysis — read-only; produces traces / candidate kernels.",
    "explore":  "Explore — propose modifications; one round produces a candidate, not yet validated.",
    "deep":     "Deep — kernel-owned. Emit via REQUEST(target_agent='kernel', kind=...).",
    "validate": "Validate — apply the accumulated optimization_stack and re-bench to get an honest cumulative gain.",
    "finalize": "Finalize — write the final report.",
    "support":  "Support — invoke only when triggered (plateau / crash / re-exploration).",
}


# ---------------------------------------------------------------------------
# Section builders (each returns a list[str] of lines, no trailing blank)
# ---------------------------------------------------------------------------
def _section_mission() -> list[str]:
    return [
        "## 1. MISSION",
        "",
        "You are the Orchestration agent of an autonomous inference-optimization loop.",
        "Your single most important goal is to maximise the run's **cumulative_gain**",
        "(percent over baseline_tput) within the wall-clock budget.",
        "",
        "Every tick, ask yourself:",
        "  \"Given current SharedState, remaining time, and the action catalogue below,",
        "   which next action gives the highest expected_gain / cost_minutes?\"",
        "",
        "An optimization is only \"real\" once it has been validated as part of the",
        "full optimization_stack. v0.8 M3 inlines that validation into every",
        "``explore`` KEEP (per-KEEP stack rebench), so the validated cumulative",
        "gain advances automatically — sums of per-round gains still do NOT",
        "compose linearly, so drive the loop until ``explore`` has produced",
        "at least one KEEP that survived the stack rebench.",
    ]


def _section_session_context(
    *,
    framework: str,
    kernel_enabled: bool,
    objective_kind: str,
    objective_value: float | str | None,
    max_minutes: int,
    framework_source_roots: tuple[str, ...] | None = None,
) -> list[str]:
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
        f"- objective        : {obj}",
        f"- max_minutes      : {max_minutes}",
        f"- framework_source_roots: {roots_line}",
        "",
        "Per-tick dynamic context (Mission progress, Time budget, Shared",
        "session state, Coordinator checklist, KB hints, inbox tail) is",
        "appended below the system prompt every tick by the Coordinator.",
        "The Time-budget block carries `remaining=X.Xmin`. When `remaining` is",
        "smaller than `report.typical_runtime_min` * 3 you SHOULD propose",
        "`report` as the next action — the Coordinator will also auto-flush a",
        "deterministic report at the deadline, but proposing it earlier",
        "captures any LLM narrative you want surfaced.",
        "",
        "**Phase awareness (v0.8 §3.2 / §3.3)**: every tick also brings a",
        "`=== Phase ===` block carrying the current phase, elapsed seconds",
        "in that phase, and budget cap. The `=== Phase-allowed actions ===`",
        "block lists the exact set of actions you may propose this tick;",
        "anything outside that set returns `policy_denied` with rule",
        "`phase_incompatible`. The 6-phase chain is:",
        "  PRELUDE → FRAMEWORK_PR → EXPLORE → KERNEL → SWEEP → CLOSE",
        "(FRAMEWORK_PR is skipped under ``--no-framework``.)",
        "Transitions are Coordinator-owned (you cannot write phase).",
    ]


# ---------------------------------------------------------------------------
# phase semantics injected into the static system prompt
# ---------------------------------------------------------------------------
def _section_phase_semantics(*, kernel_enabled: bool) -> list[str]:
    """Render the per-phase allowed-action contract.

    The actual *current* phase is injected dynamically by
    ``Coordinator._compose_prompt``; this section explains what each
    phase **means** so the LLM has a stable mental model independent
    of the runtime state.
    """
    # Lazy import: phase_state imports only stdlib so this is safe at
    # module-import time, but keeping it local makes the lazy
    # dependency explicit (and lets tests stub PHASE_ALLOWED_ACTIONS
    # without rewriting this file).
    from ..phase_state import PHASE_ALLOWED_ACTIONS, PHASE_NAMES

    lines: list[str] = [
        "## 3a. PHASE CONTRACT (v0.8 §3.2 / §3.3)",
        "",
        "The Coordinator runs the optimization in a 6-phase linear pipeline",
        "(FRAMEWORK_PR collapses out with `--no-framework`, leaving 5).",
        "Each tick it injects a `=== Phase ===` block with the current",
        "phase. Per-phase action allowlists (PolicyGate R1 enforces these):",
        "",
    ]
    for phase in PHASE_NAMES:
        allowed = sorted(PHASE_ALLOWED_ACTIONS.get(phase, frozenset()))
        if not kernel_enabled and phase == "KERNEL":
            # No-kernel run will not enter KERNEL phase; render but flag.
            lines.append(
                f"- **{phase}**: {', '.join(allowed)} (skipped in --no-kernel runs)"
            )
        else:
            lines.append(f"- **{phase}**: {', '.join(allowed)}")
    lines.extend([
        "",
        "Phase transitions are Coordinator-owned and based on machine-",
        "judgeable signals: `baseline_tput > 0` exits PRELUDE; plateau",
        "judges or budget caps exit EXPLORE / KERNEL / SWEEP; the wall-",
        "clock deadline (closing phase) exits to CLOSE. You influence",
        "transitions indirectly — by driving the current phase's signals",
        "in the right direction — never by writing `phase` directly.",
    ])
    return lines


def _filter_actions(
    registry: ActionRegistry,
    enabled: Iterable[str],
) -> list[ActionMetadata]:
    enabled_set: list[str] = list(enabled)
    out: list[ActionMetadata] = []
    for name in enabled_set:
        meta = registry.get(name)
        if meta is None:  # silently skip; caller already validated
            continue
        out.append(meta)
    return out


def _phase_eta_summary(actions: list[ActionMetadata]) -> list[tuple[str, float, list[str]]]:
    """Group actions by phase preserving _PHASE_ORDER; return (phase, eta_min_sum, names)."""
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
    # Defensive: any unknown phases at the end (should never happen for the
    # shipped registry, but keeps the builder robust against future actions).
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
    lines.extend([
        "",
        f"Sum of typical phase ETAs: ~{eta_total:.0f} min vs max_minutes={max_minutes}.",
        "If sum >> budget, prefer high-gain/low-cost actions and skip optional",
        "phases (analysis / support). If sum << budget, do an extra explore round",
        "before report.",
        "",
        "``explore`` runs its per-KEEP stack rebench inline — there is no",
        "standalone validation step. Route every grid attempt through",
        "``delegate{action_name='explore', params={grid: ...}}``.",
        "",
        "At the wall-clock deadline the Coordinator auto-enqueues a deterministic",
        "`report` (no LLM) during closing phase — do not waste ticks re-proposing",
        "it unless you want an earlier narrative version before time runs out.",
    ])
    return lines


def _format_gain_pair(meta: ActionMetadata) -> str:
    lo, hi = meta.expected_gain_pct
    if lo == 0.0 and hi == 0.0:
        return "0%"
    return f"{lo:.0f}-{hi:.0f}%"


def _format_emit_hint(meta: ActionMetadata) -> str:
    if meta.name in KERNEL_OWNED_ACTIONS:
        if meta.name == "kernel_opt":
            kind_hint = "run_optimization"
        elif meta.name == "gemm_tuning":
            kind_hint = "run_gemm_tuning"
        elif meta.name == "integrate":
            kind_hint = "integrate"
        else:
            kind_hint = meta.name
        return (
            f"REQUEST{{target_agent='kernel', kind='{kind_hint}', params={{...}}}}"
        )
    if meta.name == "report":
        return "propose_action{action_name='report', predicted_gain_pct=0.0}"
    # PR-A1 (Arbor-into-Hyperloom): ``specialist`` is a synthetic LLM
    # sub-agent dispatch (no propose_action wrapper — go straight to
    # delegate with the per-payload contract enforced by PolicyGate's
    # ``specialist_dispatch_source``). ``integrate_patch`` is the
    # serving-lane-locked follow-up that consumes a specialist's worktree
    # patches; expose it as a direct delegate too so the LLM does not
    # waste a tick proposing it first.
    if meta.name == "specialist":
        return (
            "delegate{action_name='specialist', params={"
            "domain=<one of serving_specialist|kernel_switch_specialist|"
            "comm_specialist|compiler_specialist|system_specialist|"
            "pr_intel_specialist>, "
            "gap_canonical_id=<stable gap id>, "
            "gap_symptom?=<str>, gap_layer?=<str>, "
            "gap_evidence?={profile_trace:..., ...}, "
            "max_turns?=<int<=16>}}"
        )
    if meta.name == "integrate_patch":
        return (
            "delegate{action_name='integrate_patch', params={"
            "specialist_task_id=<completed specialist task_id>, "
            "patches?=[<patch paths from specialist_done>], "
            "config_changes?={ENV_VAR: value}, "
            "keep_threshold_pct?=1.0, "
            "accuracy_baseline?={task: {metric: score}}}}"
        )
    # Mirrors the specialist emit hint shape: payload field table +
    # key constraints + PolicyGate reason-code surface for self-correction.
    if meta.name == "dynamic_action":
        return (
            "delegate{action_name='dynamic_action', params={"
            "motivation_gap_text=<why no single specialist can cover>, "
            "scope_domains=[<>=2 specialist domain keys>], "
            "side_effects_declared=[<framework_source|...>], "
            "budget_hint?=<low|medium|high>}}. "
            "Constraints: scope_domains length >= 2; "
            "side_effects_declared must not include any kernel-owned "
            "action / metric / accuracy_gate / server lifecycle; "
            "at most 1 dispatch per EXPLORE round. "
            "PolicyGate reason codes on denial: "
            "dynamic_phase_violation / dynamic_source_violation / "
            "dynamic_payload_schema / dynamic_scope_too_narrow / "
            "dynamic_scope_unknown_domain / "
            "dynamic_side_effects_red_line / "
            "dynamic_kernel_only_disallowed / "
            "dynamic_round_cap_exhausted."
        )
    return (
        f"propose_action{{action_name='{meta.name}', "
        f"predicted_gain_pct=<your estimate>}}"
    )


def _format_grid_injection_hint(name: str) -> str | None:
    """Return a per-action one-liner showing the LLM how to override grid."""
    if name == "explore":
        return (
            "GRID INPUT (v0.8 M3, REQUIRED): emit "
            "`delegate{action_name='explore', params={grid: [{name, "
            "extra_args, extra_envs, provenance, kb_evidence?, "
            "pr_evidence?, source_evidence?}, ...], "
            "base_extra_args?, base_tput?, accuracy_baseline?, "
            "keep_threshold_pct?: 1.0, stack_stable_threshold_pct?: 0.5}}`. "
            "Variants run serially; each KEEP triggers an inlined "
            "stack rebench. "
            "**Provenance is audit/advisory, not a hard gate:** "
            "accepted values include provenance='llm_direct' "
            "(Orchestration-authored), provenance='default_grid', "
            "provenance='specialist:<domain-or-tag>', and "
            "provenance='dynamic'. Prefer specialist / research-hint "
            "variants when they exist, but use llm_direct for uncovered "
            "gaps or cold-start hypotheses. **Remaining caps:** up to "
            "research_lane_capacity variants (clamped to the 2 x visible "
            "GPU count ceiling) in the grid may carry "
            "provenance='specialist:*' (rule "
            "'explore_specialist_grid_max_one'); at most one "
            "provenance='dynamic' variant may run per explore round. The "
            "executor dedups against SharedState.explore_search by "
            "canonical_fingerprint, so a rename of an already-tested "
            "(args, envs) collapses to the same row."
        )
    if name == "sweep":
        return (
            "GRID OVERRIDE: emit `delegate{action_name='sweep', params={grid: "
            "[{conc, isl, osl}, ...]}}` (or params.conc_values / "
            "params.isl_osl_values) to override the workload frontier."
        )
    return None


def _section_action_catalogue(actions: list[ActionMetadata]) -> list[str]:
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
            if name in KERNEL_OWNED_ACTIONS:
                tag_parts.append("KERNEL-OWNED")
            tag = (" (" + ", ".join(tag_parts) + ")") if tag_parts else ""
            lines.append(
                f"- **{name}**{tag} — {meta.description}"
            )
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
    lines = [
        "## 5. DECISION FRAMEWORK (apply EVERY tick BEFORE emitting)",
        "",
        "Read the dynamic SharedState section and apply, in order:",
        "",
        "1. **Stop**: if `stop_reason` is set OR `cumulative_gain >= target_gain_pct`,",
        "   propose `report` once (if not already done) then heartbeat 'goal-reached'.",
        "2. **Measure**: if `baseline_tput == 0`, propose `baseline`. Wait for",
        "   delegated_result; do NOT re-baseline on a positive result with warnings.",
        "3. **Inlined stack-rebench**: the ``explore`` action runs its own",
        "   per-KEEP stack rebench, so there is no standalone validation step",
        "   to schedule. Route every grid attempt through",
        "   ``delegate{action_name='explore', params={grid: [...] }}``.",
    ]
    lines.append(
        "4. **Analysis is auto-managed**. Roofline (or profile under "
        "``--no-enable-roofline``) is enqueued by the Coordinator at "
        "PRELUDE and at every +10% validated-gain watermark crossing. "
        "Do not propose ``profile`` or ``roofline`` — PolicyGate denies "
        "both with ``rule='analysis_action_not_llm_proposable'``. While "
        "the analysis task is in flight, ``specialist`` / ``explore`` / "
        "kernel-owned dispatches are deferred until ``analysis.md`` / "
        "``last_profile_trace`` refreshes.",
    )
    lines.extend([
        "5. **Phase-aware action selection**. v0.8",
        "   retired the legacy ``Action scores`` block. There is no system-side",
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
        "      cross-session priors carry "
        "*qualitative* hints (what worked / what failed last time).",
        "   d. **specialist proposal_set** (M5+) — when an explore round just",
        "      finished, the proposal_set drives the next `explore` grid.",
        "   e. **Mandatory MUST-FIRST rules**: baseline before anything else.",
        "      ``analysis.md`` / ``last_profile_trace`` arrive automatically",
        "      from the Coordinator-owned analysis task at PRELUDE and at",
        "      every +10% watermark crossing — do not gate ``kernel_opt`` on",
        "      a manually-proposed profile.",
        "6. **Phase budget awareness**. The `=== Phase ===` block carries",
        "   ``phase_budget_remaining_pct``. As that number falls below 0.2,",
        "   prefer lower-cost / known-good actions (explore over kernel_opt).",
        "   Plateau judgments fire when the system thinks you're stalled;",
        "   they auto-advance the phase. Don't fight them — drive the",
        "   current phase's signal in the right direction or emit",
        "   ``escalate_strategy_change{hint='skip_to_kernel'}`` via",
        "   robustness if you know it's time to move on.",
        "7. **Sweep / report tail**: when EXPLORE plateau fires, the",
        "   Coordinator routes EXPLORE → KERNEL (kernel_enabled) or →",
        "   SWEEP (--no-kernel). When SWEEP completes, propose `report`",
        "   for the LLM narrative (Coordinator also auto-enqueues one at",
        "   the deadline).",
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
        "   `disable_run_eval`.",
        "2. **`last_action_failures`** (global rolling log capped at the last",
        "   10 unpromotable results across ALL kinds, including kernel-owned).",
        "   Use this when the per-action history doesn't carry the kind you",
        "   need (e.g. `integrate` failure visible here but `<action>_attempts`",
        "   only covers the six explore/validate kinds).",
        "3. **`baseline_failure_streak`** (consecutive failed baselines).",
        "   Once this hits 3, Coordinator sets `stop_reason='baseline_failed'`",
        "   and the run terminates; you must recover BEFORE the third failure.",
        "",
        "Three decision rules apply in order:",
        "",
        "* **RULE F1 — same fingerprint, twice failed → change at least one knob.**",
        "  If the last two `<action>_attempts` entries share an `extras.fingerprint`",
        "  with the proposal you're about to emit, PolicyGate WILL deny it with",
        "  `rule='baseline_self_loop'`. Bump at least one of: `params.benchmark_script`",
        "  (a sanitized `*.sh` file name — Magpie's `dsr1_fp8_mi300x.sh` hardcodes",
        "  `--result-dir /workspace/`, but `sglang_mi300x.sh` respects",
        "  `$RESULT_DIR`), `params.result_dir`, `params.extra_server_args`, or",
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
        "* **RULE F4 — `policy_denial_streak` is a fact, not a lock.** When the",
        "  per-tick `Recent policy denials` block shows `streak≥2` for an",
        "  (action, rule) pair, the SAME params will keep colliding.",
        "  v0.8 §3.9 retired the legacy ``locked_reason`` mirror, but the",
        "  underlying anti-loop is unchanged: at streak≥5 the family is",
        "  auto-pruned; at streak≥10 the run stops with",
        "  ``stop_reason='policy_loop'``. Recover by omitting",
        "  ``idempotency_key`` (Coordinator derives a fresh tick+content",
        "  fingerprint) AND changing at least one substantive param —",
        "  a new ``params.grid`` variant, different ``benchmark_script``,",
        "  or a sibling action family.",
        "",
        "Example (baseline failed twice with `error_class='no_report'`):",
        "",
        "    propose_action{action_name='baseline',",
        "        params={benchmark_script: 'sglang_mi300x.sh',",
        "                result_dir: '<session_dir>/runs/baseline/<task>/leak'},",
        "        predicted_gain_pct: 0,",
        "        notes: 'recover from no_report streak — sglang_mi300x.sh honors",
        "                $RESULT_DIR; the default model script hardcodes /workspace/'}",
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
        "",
        "Variant identity is content-based: the executor fingerprints",
        "`(sorted extra_server_args, sorted extra_envs)`, so renaming a",
        "tested variant does NOT bypass dedup — change the actual args/envs.",
        "`extra_server_args` is framework-neutral (routed to EXTRA_SGLANG_ARGS",
        "/ EXTRA_VLLM_ARGS / EXTRA_ATOM_ARGS by `--framework`).",
        "",
        "An explore round that produces zero new ideas is a bug — heartbeat",
        "with body_md='idea-pipeline-empty' so Robustness can intervene.",
    ])
    return lines


_KERNEL_OPT_PIPELINE_BODY: str = """\
## 6. KERNEL-OPT REQUEST REFERENCE (payload templates — NOT a forced ordering)

The four kernel-owned actions are picked per the DECISION FRAMEWORK
(phase allowed-set + gaps + KB priors); there is no system-side
priority ranking (§3.9 Inv-9.1). Pick the next one by reading these facts in order:
a `state.gaps[]` `layer='kernel'` gap with attempts left →
`last_kernel_opt` (KEEP→integrate next; PARTIAL→retry at most
`_DEFAULT_KERNEL_OPT_MAX_PARTIAL` then rejected; REVERT→rejected) →
skip ids in `rejected_kernel_ids` → recover from `last_action_failures`.
When `plateau_kernel` fires (3 REVERTs across distinct kernels) the
Coordinator auto-advances KERNEL → SWEEP — stop and let it.

### `trace_analyze` — must precede every `run_optimization`

  request{target_agent: 'kernel', kind: 'trace_analyze',
          params: {trace_input: <verbatim last_profile_trace>, top_k: 10}}

  Skip if `last_trace_analyze.trace_input` already equals
  `last_profile_trace` (cached). Explore/sweep/report are NEVER gated on it.

### `gemm_tuning` — `run_gemm_tuning` (FP8 SGLang only)

  request{target_agent: 'kernel', kind: 'run_gemm_tuning', params={}}

  Only when `precision='fp8'` SGLang and `last_gemm_tuning` is empty.

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

  request{target_agent: 'kernel', kind: 'run_optimization',
          params: {kernel_id: <picked kernel_id>,
                   source_file: <hot_kernels[i].source_file>,
                   candidates_path: <trace_analyze_done.candidates_path>,
                   budget_minutes: 60}}

  Backend auto-pick: DO NOT add a `backends` field. The kernel-agent's
  `choose_backends()` auto-picks per kernel (hip_cpp+bench→GEAK;
  triton+bench→GEAK then Claude/Codex; python/unknown→Claude/Codex).
  Pinning a backend forces every kernel through it even where GEAK is
  the only one that can KEEP — the exact #144 last comment Layer 2
  regression. Read `kernel_opt_attempts` + `pending_keep_kernels` to
  see what's still queueable; the batch handler filters
  rejected/in-flight/exhausted candidates.

### `integrate` — forced immediately after a KEEP

On `run_optimization_done` with `decision='KEEP'`, integrate is the only
allowed action until the patch lands on `optimization_stack`:

  request{target_agent: 'kernel', kind: 'integrate',
          params: {kernel_id, patch_path, target_file, base_tput,
                   extra_server_args, config_path}}

  Omit `base_tput` / `patch_path` / `source_file` and the Coordinator
  fills them from `current_best.tput` and the per-kernel
  `kernel_opt_attempts` ledger (this is what drains a multi-KEEP queue).
  PARTIAL / REVERT → do NOT integrate; pick the next action normally.

  **Multi-KEEP queue:** `pending_keep_kernels` (sorted strongest-first)
  lists queued KEEPs; integrate `[0]` each tick. Do NOT propose `report`
  while it is non-empty, nor while `untried_hot_reusable_kernels`
  (reusable hot kernels with `gpu_pct >= 3%` and zero attempts) remain —
  drain them with `run_optimization{candidates_path: <from
  last_trace_analyze>}` (the batch handler fans out automatically).

### KERNEL TARGETING

Only rewrite reusable native sources in the trace. NEVER optimize
`/tmp/torchinductor*` / `triton_poi_*` / `triton_red_*` runtime-generated
kernels — they're tied to one compile cache and not reusable."""


def _read_rules_fragment(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# Entry declaration for the supplementary cross-domain ReAct channel.
# Renders only when ``dynamic_action`` is in the enabled action set.
# No triggering heuristics, examples, or fallback guidance — those
# would shift the channel from supplementary to default.
# ---------------------------------------------------------------------------
def _section_dynamic_action(actions: list[ActionMetadata]) -> list[str] | None:
    if not any(a.name == "dynamic_action" for a in actions):
        return None
    return [
        "## 6b. DYNAMIC ACTION (supplementary EXPLORE channel)",
        "",
        "If you believe a single cross-domain patch combination exists",
        "that no specialist could surface within its own-domain prompt",
        "boundary, you MAY dispatch one `dynamic_action` in the EXPLORE",
        "phase via `delegate{action_name='dynamic_action', params={...}}`.",
        "",
        "`dynamic_action` is a **supplementary** channel, not the default.",
        "Specialists remain the primary EXPLORE entry. At most ONE",
        "`dynamic_action` dispatch is allowed per EXPLORE round.",
        "",
        "After an `explore` KEEP, you MAY pair that winning direction",
        "with an adjacent domain via one `dynamic_action` deep-dive —",
        "an encouragement, not a requirement; skip it when no cross-domain",
        "combination is apparent.",
        "",
        "Payload contract is closed; see the EMIT line on the action",
        "catalogue entry above for the field table + PolicyGate denial",
        "reason codes. The `=== Dynamic Action History ===` block (when",
        "non-empty) lists the most recent outcomes of your own dispatches.",
    ]


def _section_rules(rules_md: str) -> list[str]:
    body = rules_md.strip() or (
        "(orchestration.md rules fragment not found — Coordinator will still"
        " enforce PolicyGate hard rules at runtime.)"
    )
    return ["## 7. RULES & OUTPUT PROTOCOL", "", body]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_orchestration_prompt(
    *,
    action_registry: ActionRegistry,
    enabled_actions: Iterable[str],
    framework: str = "sglang",
    kernel_enabled: bool | None = None,
    objective_kind: str = "time_only",
    objective_value: float | str | None = None,
    max_minutes: int = 0,
    rules_fragment_path: Path | None = None,
    framework_source_roots: tuple[str, ...] | None = None,
) -> str:
    """Compose the Orchestration system prompt.

    Parameters
    ----------
    action_registry: pre-loaded ``ActionRegistry`` (caller is responsible
        for ``.load()``).
    enabled_actions: names that this run's CLI registered executors for
        (or that the Kernel agent will service via REQUEST). Order is
        preserved only for missing-action filtering; final ordering is
        by pipeline_phase.
    framework: ``sglang`` / ``vllm`` — printed in the SESSION CONTEXT
        section.
    kernel_enabled: explicit override; when ``None``, derived from whether
        any KERNEL_OWNED action is in ``enabled_actions``.
    objective_kind / objective_value: matches :mod:`objective` strings;
        printed verbatim (e.g. ``gain_pct=10`` or ``tput=4500``).
    max_minutes: wall-clock budget for the run.
    rules_fragment_path: path to ``orchestration.md`` (the rules-only
        fragment). When ``None`` or unreadable, a placeholder is emitted.

    Returns
    -------
    str: the full system prompt, deterministic for given inputs.
    """
    actions = _filter_actions(action_registry, enabled_actions)
    if kernel_enabled is None:
        kernel_enabled = any(
            a.name in KERNEL_OWNED_ACTIONS for a in actions
        )
    framework_norm = (framework or "sglang").strip().lower() or "sglang"

    rules_md = _read_rules_fragment(rules_fragment_path)

    sections: list[list[str]] = [
        _section_mission(),
        _section_session_context(
            framework=framework_norm,
            kernel_enabled=kernel_enabled,
            objective_kind=objective_kind,
            objective_value=objective_value,
            max_minutes=max_minutes,
            framework_source_roots=framework_source_roots,
        ),
        _section_pipeline_and_budget(actions, max_minutes=max_minutes),
        # phase contract sits between the legacy
        # PIPELINE & TIME BUDGET (§3, action-runtime view) and the
        # ACTIONS catalogue (§4) so the LLM sees the *policy* layer
        # before the *catalogue*.
        _section_phase_semantics(kernel_enabled=kernel_enabled),
        _section_action_catalogue(actions),
        _section_decision_framework(kernel_enabled=kernel_enabled),
    ]
    if kernel_enabled and any(a.name == "kernel_opt" for a in actions):
        sections.append(_KERNEL_OPT_PIPELINE_BODY.splitlines())
    # Dynamic action declaration sits after the decision framework so
    # the LLM sees the supplementary-channel framing after the primary
    # EXPLORE action catalogue.
    dyn_section = _section_dynamic_action(actions)
    if dyn_section is not None:
        sections.append(dyn_section)
    sections.append(_section_rules(rules_md))

    # Join sections with a blank line between each; ensure single trailing
    # newline so snapshot files don't accumulate trailing whitespace diffs.
    parts: list[str] = []
    for sect in sections:
        parts.append("\n".join(sect))
    return "\n\n".join(parts).rstrip() + "\n"


def default_enabled_actions(*, no_kernel: bool) -> tuple[str, ...]:
    """Return the canonical enabled-action set used by the CLI."""
    return NO_KERNEL_ENABLED_ACTIONS if no_kernel else FULL_ENABLED_ACTIONS


__all__ = [
    "FULL_ENABLED_ACTIONS",
    "GRID_INJECTABLE_ACTIONS",
    "KERNEL_OWNED_ACTIONS",
    "NO_KERNEL_ENABLED_ACTIONS",
    "VALID_PIPELINE_PHASES",
    "build_orchestration_prompt",
    "default_enabled_actions",
]
