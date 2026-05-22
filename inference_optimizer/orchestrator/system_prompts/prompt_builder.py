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
    # analysis — `roofline` is the composite action (profile + trace_analyze
    # in one shot), replacing the standalone `profile` + `pmc_roofline`
    # actions per roofline-v2 D1/N2. The latter two executors remain
    # registered for stale-state.json resume compatibility (cli.py
    # _REAL_EXECUTORS_KERNEL_ONLY) and to let RooflineExecutor invoke
    # `profile` internally, but they MUST NOT be proposed directly by
    # the LLM — surface only `roofline` in the catalogue + critic
    # approve list + scoring priors so the orchestration loop has a
    # single canonical entry point.
    "roofline", "deep_kernel_analysis",
    # explore
    "backends", "params", "sweep",
    # deep — kernel-owned, emitted via REQUEST{target_agent='kernel', kind=...}
    "kernel_opt", "integrate", "operator_tuning", "vendor_kernel_config",
    # validate (Phase 3 — closes the loop on accumulated KEEPs)
    "validate_stack",
    # finalize
    "report",
    # support
    #
    # ``recover`` was re-enabled in 2026-05 alongside the robustness-agent
    # ``gpu_memory_leaked`` signal. A real executor (see
    # ``orchestrator/action_executors/recover.py``) now frees leaked
    # VRAM and, when ``HYPERLOOM_RECOVER_ALLOW_GPU_RESET=1``, attempts
    # ``rocm-smi --gpureset``. The Critic's default approve whitelist
    # already includes ``recover``.
    #
    # ``comm_optimization`` / ``compiler_tuning`` / ``dream`` /
    # ``re_explore`` remain disabled here — their executors are still
    # ``_noop_prep`` stubs and would just produce silent "succeeded"
    # spins. Re-add when real executors land (see remain_todo.md
    # sections C, I, M).
    "recover",
)

NO_KERNEL_ENABLED_ACTIONS: tuple[str, ...] = (
    # prep
    "target_analysis", "baseline",
    # analysis — roofline is still useful in no-kernel mode for the
    # snapshot it provides (cheap actions consume it via discovered_flags)
    "roofline",
    # explore
    "backends", "params", "sweep",
    # validate (still useful — bench the stacked backends/params)
    "validate_stack",
    # finalize
    "report",
    # support — recover is needed even without kernel-opt because GPU
    # leaks from baseline / backends / params / sweep can still hang the
    # session; the executor itself is kernel-agnostic.
    "recover",
)

# Actions that the Kernel agent owns end-to-end (Plan A). Orchestration MUST
# emit `request{target_agent='kernel', kind=...}` for these instead of
# `delegate{action_name=...}`. We highlight the difference in the catalogue
# section so the LLM picks the right transport.
KERNEL_OWNED_ACTIONS: frozenset[str] = frozenset({
    "kernel_opt", "integrate", "deep_kernel_analysis",
    "operator_tuning", "vendor_kernel_config",
})

# Actions that accept LLM-injected grid candidates via ``params.grid``
# (see backends.py / params.py / sweep.py for the schema). The catalogue
# section appends a grid-override hint for these so the LLM knows it can
# expand the search space beyond the shipped DEFAULT_*_GRID — this is the
# T1/T2 "search-space expansion" hook (see SKILL.md). Without this, the
# LLM only sees the action name and may never realize it can synthesize
# new variants from the discovered_flags surface.
GRID_INJECTABLE_ACTIONS: frozenset[str] = frozenset({
    "backends", "params", "sweep",
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
        "full optimization_stack via the `validate_stack` action — sums of per-round",
        "gains do NOT compose linearly. Drive the loop until you have at least one",
        "validated cumulative_gain to report.",
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
    ]


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
        "before validate_stack + report.",
        "",
        "MANDATORY rule: after any backends / params / kernel round produces a",
        "KEEP'd entry in optimization_stack, you MUST run `validate_stack` before",
        "the next explore round or before `report`. The Coordinator surfaces a",
        "TODO in the per-tick checklist when this trigger is active.",
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
        elif meta.name == "integrate":
            kind_hint = "integrate"
        else:
            kind_hint = meta.name
        return (
            f"REQUEST{{target_agent='kernel', kind='{kind_hint}', params={{...}}}}"
        )
    if meta.name == "report":
        return "propose_action{action_name='report', predicted_gain_pct=0.0}"
    return (
        f"propose_action{{action_name='{meta.name}', "
        f"predicted_gain_pct=<your estimate>}}"
    )


def _format_grid_injection_hint(name: str) -> str | None:
    """Return a per-action one-liner showing the LLM how to override grid."""
    if name == "backends":
        return (
            "TWO GRID-CONTROL SURFACES (pick the one that fits your "
            "intent):\n"
            "  (A) SUBSET (N20-A, recommended for roofline-driven runs): "
            "emit `delegate{action_name='backends', params={variants: "
            "['attn_aiter','sched_lpm', ...]}}` — names must come from "
            "the registered DEFAULT_BACKENDS_GRID listed in the "
            "BACKENDS GRID CATALOGUE block below. Use this when the "
            "roofline analysis.md points at specific kernel categories "
            "(e.g. attention-heavy -> only try `attn_*` variants; "
            "AllReduce-heavy -> only `custom_ar`). Cheaper + safer "
            "than full grid; no flag hallucination risk.\n"
            "  (B) FULL CUSTOM (T1/T2): emit "
            "`delegate{action_name='backends', params={grid: [{name, "
            "extra_sglang_args, extra_envs, note}, ...], "
            "synergy_groups?: [[name1,name2], ...] | "
            "synergy_mode?: 'auto'}}` to add candidates beyond the "
            "shipped DEFAULT_BACKENDS_GRID. Use this when "
            "SharedState.discovered_flags surfaces a flag the registered "
            "grid doesn't carry yet, or when you want to combine winners "
            "from prior rounds (backend_winners_history).\n"
            "If both `variants` and `grid` are passed, `grid` wins "
            "(full control mode)."
        )
    if name == "params":
        return (
            "TWO GRID-CONTROL SURFACES (pick the one that fits your "
            "intent):\n"
            "  (A) SUBSET (N20-A, recommended for roofline-driven runs): "
            "emit `delegate{action_name='params', params={variants: "
            "['cuda_graph_max_bs_64','mem_fraction_0_90', ...]}}` — "
            "names must come from the registered DEFAULT_PARAMS_GRID "
            "listed in the PARAMS GRID CATALOGUE block below. Use this "
            "when the roofline analysis.md points at specific bottlenecks "
            "(e.g. cuda-graph misses high -> try the cuda_graph_max_bs_* "
            "family; KV-cache pressure -> mem_fraction_* + chunked_prefill_*; "
            "queue depth growing -> max_running_requests_* + scheduling). "
            "Safer + cheaper than full grid; no flag-value hallucination "
            "risk (the values come from the registered grid).\n"
            "  (B) FULL CUSTOM (T1/T2): emit "
            "`delegate{action_name='params', params={grid: [{name, "
            "extra_sglang_args, extra_envs, note}, ...]}}` to synthesize "
            "value-filled parameter candidates beyond the registered "
            "grid (e.g. when SharedState.discovered_flags surfaces a new "
            "param flag and you want to pick a specific value). "
            "SharedState.discovered_flags lists the param flag namespace "
            "(--max-num-seqs, --cuda-graph-max-bs, etc.) you can fill in.\n"
            "If both `variants` and `grid` are passed, `grid` wins "
            "(full control mode)."
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
            tag = " (KERNEL-OWNED)" if name in KERNEL_OWNED_ACTIONS else ""
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
        "3. **Mandatory validation**: if the per-tick checklist contains a",
        "   `validate_stack required` TODO, propose `validate_stack` immediately —",
        "   no other action is allowed until cumulative_gain_validated is updated.",
    ]
    if kernel_enabled:
        lines.extend([
            "4. **Profile**: if `last_profile_trace == ''`, propose `profile`. If the",
            "   profile is fresh (matches `last_profile_args`) reuse it; do not re-profile.",
        ])
    else:
        lines.append(
            "4. (No-kernel run.) Skip profile — the Kernel agent is disabled in this run.",
        )
    lines.extend([
        "5. **Consult the `Action scores` block** (`=== Action scores (top 12 by",
        "   eff_score, tick=N) ===`). The Coordinator pre-sorts the action catalogue",
        "   by `eff_score` (descending). Pick the **highest-`eff_score`** row whose",
        "   `applicable_when` is satisfied and which has no `[cooldown N]` or",
        "   `[locked: ...]` tag. Top-1 is the default; you MAY skip down to a",
        "   lower row when you have a one-line justification in the proposal",
        "   `notes`, but NEVER propose a row that is on cooldown or locked.",
        "6. **Trust the scoreboard, don't hand-roll priorities**. Coordinator-side",
        "   updates already apply marathon-style rules to `score_mult`: KEEP decays",
        "   the same action (diminishing returns), DISCARD dampens it by 30%,",
        "   aging + UCB bonuses revive starved actions. Do NOT alternate",
        "   `backends`/`params` by hand — cooldown on KEEP enforces alternation;",
        "   do not invent your own ordering on top of the scoreboard.",
        "7. **Sweep / report**: when every explore-family row sits at low",
        "   `eff_score` (or is locked), propose `sweep` once to validate gains",
        "   across (CONC, ISL, OSL), then `report`.",
        "",
        "### How scoring updates",
        "",
        "* Every completed action sets a per-action cooldown so the next tick",
        "  prefers a different family (anti-loop).",
        "* KEEP: `score_mult *= max(0.5, 1 - 0.1*gain_pct)` — strong wins decay",
        "  faster than marginal ones.",
        "* DISCARD: `score_mult *= 0.7` — repeated DISCARDs push the row out of",
        "  the top quickly.",
        "* Aging + UCB bonuses bubble under-sampled rows up so deep / support",
        "  actions (operator_tuning / compiler_tuning / dream) eventually get a",
        "  turn even when their base score is low.",
        "* `[locked: grid_exhausted]` means the search grid is depleted; do not",
        "  propose the action until the Coordinator re-unlocks it (e.g. after",
        "  `re_explore` refreshes the candidate set).",
        "",
        "If you cannot move forward (everything cooldown'd or locked / new failures),",
        "emit `send_message{topic='heartbeat', body_md='blocked: <reason>'}` and let",
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
        "   `benchmark_script` / `result_dir` / `extra_sglang_args` /",
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
        "  `$RESULT_DIR`), `params.result_dir`, `params.extra_sglang_args`, or",
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
        "* **RULE F4 — `policy_denial_streak` ≥ 2 locks the action.** When the",
        "  per-tick `Policy denials` block shows `streak≥2` for an action, the",
        "  Coordinator tags it `[locked: policy_loop:<rule>]`. Re-trying the",
        "  SAME params will keep colliding. Omit `idempotency_key` (Coordinator",
        "  derives a fresh tick+content fingerprint) AND change at least one",
        "  substantive param — e.g. a new `params.grid` variant, different",
        "  `benchmark_script`, or switch to a sibling action family. At streak≥5",
        "  the family is auto-pruned; at streak≥10 the run stops with",
        "  `stop_reason='policy_loop'`.",
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
        "### IDEA GENERATION (T2 / marathon IR-26 — apply after EVERY explore round)",
        "",
        "When a `backends` / `params` / `sweep` round completes, use",
        "`SharedState.backend_winners_history` and `SharedState.discovered_flags`",
        "to compose the next round's grid before re-proposing the same action.",
        "Walk through these stages in order; push each new idea via",
        "`params.grid` (and optionally `params.synergy_groups` /",
        "`params.synergy_mode='auto'`) on the next `delegate`:",
        "",
        "**Variant identity is content-based.** The executor hashes",
        "`(sorted(extra_sglang_args tokens), sorted(extra_envs pairs))` and",
        "indexes `SharedState.{backends_search,params_search}.tested` by that",
        "fingerprint. Renaming an already-tested variant (e.g. `attn_aiter`",
        "→ `attn_aiter_v2`) does NOT bypass dedup — your grid entry will be",
        "dropped before launch. To re-test, change the actual `extra_sglang_args`",
        "or `extra_envs`. Both `backends_search` and `params_search` are the",
        "authoritative dedup ledgers and filter BOTH default grids and",
        "LLM-supplied `params.grid` uniformly.",
        "",
        "**Use the numeric `gain_pct` on every row.** The `*_search` and",
        "`backend_winners_history` blocks now render `±x.xx%` per variant.",
        "Read them as ranking signal: a `-2%` reject means the flag itself is",
        "bad on this workload (don't retry the same value), `-0.3%` is",
        "'try a different value', and a sub-threshold `+0.4%` reject is a",
        "candidate for a synergy combo with another winner.",
        "",
        "1. **Sub-actions** — sibling flags. If `--max-num-seqs 256` won, also",
        "   try `--max-num-seqs 128` / `512` / `1024`; if `VLLM_ROCM_USE_AITER=1`",
        "   won, sweep the related `VLLM_ROCM_USE_AITER_*` boolean family.",
        "2. **Follow-ons (success → deepen)** — new combos of last round's",
        "   winners (use `synergy_mode='auto'` or list the group explicitly in",
        "   `synergy_groups`); the executor de-duplicates against",
        "   `SharedState.synergy_attempted` so you can re-emit safely.",
        "3. **Retry-with-alternate-strategy (failure)** — for each variant",
        "   listed in `params_search.rejected`, propose the same flag with a",
        "   different value or pair it with a complementary winner.",
        "4. **Synthetic fallback** — when `backend_winners_history` is empty for",
        "   the last 2 rounds, mine `discovered_flags.<framework>.backend_flags`",
        "   for boolean toggles not yet in any GridVariant and propose them.",
        "5. **Self-reflection** — if stages 1-4 produced fewer than 2 new",
        "   variants, consult the dynamic KB hints for cross-model lessons",
        "   that fit the current model_class / framework, and add them.",
        "",
        "An explore round that produces zero new ideas is a bug — heartbeat",
        "with body_md='idea-pipeline-empty' so Robustness can intervene.",
        "",
        "### Roofline analysis (composite action)",
        "",
        "Propose `roofline` whenever you need a fresh TraceLens snapshot:",
        "right after baseline (snapshot #1), then between every interleaved",
        "round of cheap exploration (backends + params) so kernel_opt sees",
        "the post-exploration hot-kernel distribution rather than the",
        "baseline one. The executor internally runs `profile` + `trace_analyze`",
        "in one shot; do NOT propose `profile` or the legacy `pmc_roofline`",
        "directly — they are kept registered for back-compat only and the",
        "PolicyGate hard-blocks direct profile proposals (N9, see",
        "design/roofline-v2.md §6.5/§6.5.1 for the full enforcement chain).",
        "",
        "    delegate{action_name='roofline',",
        "        params={notes: 'baseline snapshot' | 'post-backends-N snapshot' | ...},",
        "        predicted_gain_pct: 0,",
        "        notes: 'TraceLens analysis.md will appear in shared_state.last_trace_analyze'}",
    ])
    return lines


_KERNEL_OPT_PIPELINE_BODY: str = """\
## 6. KERNEL-OPT REQUEST REFERENCE (payload templates — NOT a forced ordering)

The three kernel-owned actions (`trace_analyze`, `kernel_opt`,
`integrate`) are scored on the `Action scores` board like every other
action. Pick them by `eff_score` per the DECISION FRAMEWORK; the blocks
below are only **payload templates** describing how to build the REQUEST
once you have selected the action. The Coordinator hard-gates the
obvious prerequisites (TODO 3/5 surfaces as guidance after a fresh
`profile` until `trace_analyze` populates the cache so TraceLens writes
`analysis.md`; TODO 4/5 fires after a `kernel_opt` KEEP forces
`integrate`). Explore actions like `params` / `backends` / `sweep` are
NEVER gated on `trace_analyze` at the action layer (only
`run_optimization` REQUESTs are gated, by
`_sequence_denial_for_request`). Everything else flows through the
scoreboard.

### `trace_analyze` — payload (Coordinator gates `run_optimization` until cache is fresh)

  request{target_agent: 'kernel', kind: 'trace_analyze',
          params: {trace_input: <verbatim last_profile_trace>, top_k: 10}}

  STRICT: if `last_trace_analyze.trace_input` already equals
  `last_profile_trace`, the candidate list is cached — do NOT re-emit.
  `trace_analyze` must precede every `run_optimization` request — the
  Coordinator denies kernel_opt requests with `trace_analyze must run
  first` when the cache is stale, but `params` / `backends` / `sweep` /
  `report` are NEVER gated on it. Re-emit only after a fresh `profile`
  action invalidates the cache.  TODO 3/5 surfaces in
  `_required_next_step` as guidance whenever `last_profile_trace` is
  fresh but the cache is still stale — emit the REQUEST yourself
  before kernel_opt / integrate cycles.

### `kernel_opt` — payload for `run_optimization`

When the scoreboard surfaces `kernel_opt`, pick the next reusable
native kernel from `last_trace_analyze.reusable_native_kernel_ids`,
in order, skipping any kernel_id already present in
`last_kernel_opt.kernel_id`.

HARD RULES (applied at REQUEST build time, NOT at action-selection time):
  - kernel_id MUST appear in `reusable_native_kernel_ids`. Never pick
    from raw `hot_kernels_top15` if the entry is missing — top hot
    kernels are often Tensile / CK / vendor binaries and will be
    rejected with `non_reusable_kernel`.
  - If `reusable_native_kernel_ids` is empty, do NOT propose
    `kernel_opt` (its `applicable_when` is implicitly violated).
    Heartbeat instead and consider re-profiling.

  request{target_agent: 'kernel', kind: 'run_optimization',
          params: {kernel_id: <picked kernel_id>,
                   source_file: <hot_kernels[i].source_file>,
                   candidates_path: <trace_analyze_done.candidates_path>,
                   budget_minutes: 60}}

  HARD RULE — backend selection: DO NOT add a `backends` field unless the
  operator has explicitly asked you to pin a specific backend. The
  kernel-agent's `choose_backends()` auto-picks per kernel from
  `(source_type, benchmark_available)`:
    - hip_cpp + bench → GEAK
    - triton + bench → GEAK, then Claude/Codex as fallback
    - python / unknown → Claude, Codex
  Hard-coding `backends: 'claude'` here forces every kernel through Claude
  even on hip_cpp+bench candidates that GEAK can rewrite, which is the
  exact regression that closed #144's last comment Layer 2. Omit the field
  and let auto-pick fire.

### `integrate` — payload (TODO 4/5 forces this immediately after a KEEP)

When `run_optimization_done` arrives with `result.proposal.decision='KEEP'`,
the Coordinator's TODO 4/5 makes `integrate` the only allowed action
until the patch lands on `optimization_stack`. Payload:

  request{target_agent: 'kernel', kind: 'integrate',
          params: {kernel_id, patch_path, target_file, base_tput,
                   extra_sglang_args, config_path}}

If you omit `base_tput`, the Coordinator auto-fills it from
`current_best.tput` so chained integrates (multi-KEEP drain, see below)
do not need you to track the running baseline manually. Explicit operator
override still wins.

If you omit `patch_path` / `source_file`, the Coordinator resolves them
from `last_kernel_opt.best_artifact_path` first, then from
`kernel_opt_attempts[<kernel_id>].last_artifact_path` (the per-kernel
ledger). This second fallback is what makes multi-KEEP queue drain
work: queued KEEPs whose kernel_id != `last_kernel_opt.kernel_id` still
resolve to a real patch.

If `result.proposal.decision` is `PARTIAL` or `REVERT`, the patch is
rejected — do NOT integrate. The Coordinator unlocks immediately; consult
the scoreboard for the next action like normal. A second `kernel_opt`
round on the next reusable kernel_id often surfaces as top-1 (because
the previous KEEP/REVERT decayed only that kernel_id's branch), but is
not required — the scoreboard decides.

#### Multi-KEEP integrate queue (PR-B)

A single `run_optimization` batch may produce KEEPs for multiple
kernels at once. The Coordinator streams each sub-result into
SharedState the instant it lands (not after gather wait-all), and
maintains a queue keyed off `kernel_opt_attempts`. Read these state
fields to drive the drain:

  * `pending_keep_kernels`         — list[str] of queued KEEP
    `kernel_id`s, sorted strongest-first by micro_speedup. The TODO 4/5
    integrate gate stays open as long as this list is non-empty, so
    DO NOT propose `report` while pending KEEPs remain.
  * `has_keep_pending_integrate`   — bool mirror, convenient short-circuit.

For each tick where `has_keep_pending_integrate=true`:
  1. Pick `pending_keep_kernels[0]` (highest micro) as the next
     `integrate` target.
  2. Emit the `integrate` request; the Coordinator fills in
     `patch_path` / `source_file` / `base_tput` automatically.
  3. After the result lands, the queue either advances to the next
     pending KEEP or drains to empty — `validate_stack` (TODO 5/5)
     fires once the integrate stack has new unvalidated entries.

Same-source-file collision: `apply_kernel_patch` is a whole-file
overwrite, so if two KEEPs target the same `source_file`, the
queue collapses to the strongest one and silently drops the rest
(no manual conflict handling required).

After every successful `integrate` (KEEP), the Coordinator records a
new entry on `optimization_stack` and the TODO 5/5 `validate_stack`
gate fires; obey it before resuming any explore / deep round.

### KERNEL TARGETING (native vs torch.compile)

`trace_analyze` profiles the *final* serving mode (with or without
torch.compile / CUDAGraph), but kernel-opt may only rewrite reusable
native sources that still appear in that trace. NEVER optimize
`/tmp/torchinductor*`, Inductor cache, or `triton_poi_*` /
`triton_red_*` runtime-generated kernels — they're tied to one compile
cache and the patch is not reusable. If compile-on leaves no high-share
reusable native kernels, stop kernel-opt and continue with framework /
params / compile configuration tuning instead."""


def _read_rules_fragment(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _section_params_grid_catalogue(*, framework: str) -> list[str]:
    """N20-A: render the registered params grid as a catalogue so the
    LLM can name a SUBSET when proposing the `params` action.

    Mirror of _section_backends_grid_catalogue. params grid is bigger
    (~28 SGLang variants vs ~10 backends) so the LLM benefit from
    subset selection is larger here — running all 28 is ~4-5h on
    R1 / 30-45min on Qwen3-32B, vs ~30-60min if the LLM picks 3-5
    relevant ones based on the roofline analysis.
    """
    from inference_optimizer.orchestrator.action_executors.params import (
        DEFAULT_PARAMS_GRID,
        DEFAULT_VLLM_PARAMS_GRID,
    )

    fw = (framework or "sglang").strip().lower()
    grid = DEFAULT_VLLM_PARAMS_GRID if "vllm" in fw else DEFAULT_PARAMS_GRID
    fw_label = "vLLM" if "vllm" in fw else "SGLang"

    HINT_BY_NOTE: dict[str, str] = {
        "cuda_graph":  "high cuda-graph miss rate / decode under-utilised",
        "decode_steps":"long decode chains (raise step batching)",
        "memory":      "KV-cache pressure / OOM near-misses",
        "scheduling":  "queue depth growing / request batching",
        "prefill":     "long prompts / prefill-throughput bound",
        "cache":       "radix-cache hit rate low / hurts MoE throughput",
        "tokenizer":   "tokenizer CPU bound (rare; high-concurrency only)",
        "streaming":   "client streaming-interval pressure",
        "overlap":     "decode/prefill overlap stalls (CPU sync hot)",
        "attention":   "attention backend swap (advanced; for hot attn)",
        "indexer":     "MLA indexer kernel hot",
        "comm":        "NCCL/RCCL tuning (rare; comm-bound only)",
    }

    lines: list[str] = [
        f"## PARAMS GRID CATALOGUE ({fw_label})",
        "",
        "Registered variants you may name in `params.variants=[...]` "
        "when proposing `params`. The executor will run only the ones "
        "you list. Pick variants whose trigger hint matches the "
        "dominant pattern in the roofline analysis.md.",
        "",
        f"{'name':32s}  {'flag(s) / env(s)':46s}  trigger hint",
        f"{'-' * 32}  {'-' * 46}  {'-' * 60}",
    ]
    for v in grid:
        flag = (v.extra_sglang_args or "").strip()
        if v.extra_envs:
            env_repr = " ".join(
                f"{k}={vv}" for k, vv in sorted(v.extra_envs.items())
            )
            flag = (
                f"{flag} {env_repr}" if flag else f"env: {env_repr}"
            )
        if len(flag) > 46:
            flag = flag[:43] + "..."
        hint = HINT_BY_NOTE.get(v.note or "", v.note or "(generic)")
        lines.append(f"{v.name:32s}  {flag:46s}  {hint}")

    lines.extend([
        "",
        "Example (roofline shows cuda-graph miss 30% + KV pressure):",
        "  delegate{action_name='params', params={variants: "
        "['cuda_graph_max_bs_64','cuda_graph_max_bs_32',"
        "'mem_fraction_0_90']}, predicted_gain_pct: 1.0}",
        "",
        "Same advice as backends — name ALL variants whose trigger hint "
        "matches the dominant pattern(s) in analysis.md (no upper cap; "
        "the executor has no max_candidates_per_round limit by default). "
        "When analysis.md flags multiple bottleneck categories (e.g. "
        "host-bound + KV-pressure), include every relevant variant "
        "across all flagged categories. Skipping `torch_compile_on` "
        "when analysis explicitly mentions `torch.compile` is the "
        "kind of miss this catalogue is designed to prevent. Better "
        "to over-include 8-10 relevant variants than miss a category "
        "the analysis flagged; only skip variants whose trigger hint "
        "is clearly orthogonal to what analysis.md surfaces.",
        "",
        "### N22 ENFORCED MAPPINGS (analysis.md keyword -> required variants)",
        "",
        "The PolicyGate runs a non-blocking advisory check on every "
        "params proposal: if analysis.md mentions a keyword below and "
        "your `variants` list omits the implied variant, an advisory "
        "lands in SharedState.last_proposal_advice (you'll see it on "
        "the NEXT tick). The propose still goes through — but you should "
        "self-correct on the follow-up cheap-action propose by adding "
        "the missing variant. The map below is the canonical contract;",
        "consult `_analysis_keyword_map.py` for the full machine-readable "
        "version (~50 keys covering compile, cuda-graph, KV cache, attn, "
        "moe, scheduling, host-bound, fusion, radix cache).",
        "",
        "  analysis.md mentions ...     -> include variant(s)",
        "  -----------------------       -------------------",
        "  'torch.compile'              -> torch_compile_on",
        "  'cuda graph(s)'              -> cuda_graph_max_bs_8/16/32/64",
        "  'kv cache' / 'kv-cache'      -> mem_fraction_0_85/0_90/0_80",
        "  'decode bound' / 'long decode' -> decode_steps_8/16/32",
        "  'long prompt(s)' / 'prefill' -> chunked_prefill_32k/64k/128k,",
        "                                  max_prefill_tokens_32k/64k",
        "  'queue depth' / 'scheduling' / 'concurrency' / 'batching'",
        "                               -> max_running_requests_128/256,",
        "                                  sched_lpm, sched_dfs",
        "  'host-bound' / 'host-side'   -> torch_compile_on,",
        "                                  cuda_graph_max_bs_32/64,",
        "                                  decode_steps_16/32",
        "  'gpu idle' / 'underutilization' -> torch_compile_on,",
        "                                  cuda_graph_max_bs_64,",
        "                                  decode_steps_32",
        "  'overlap' / 'multi-stream'   -> sglang_multi_stream_overlap",
        "  'kernel fusion'              -> enable_fused_moe, enable_mixed",
        "  'radix cache'                -> disable_radix_cache",
        "  'MoE' / 'expert routing'     -> moe_aiter, enable_fused_moe",
        "  'attention backend'          -> attn_aiter, attn_triton,",
        "                                  decode_aiter",
        "  'AllReduce' / 'NCCL' / 'RCCL'-> custom_ar",
        "",
        "(Mapping is case-insensitive substring match. False positives "
        "are tolerable — the LLM may trim genuinely-orthogonal ones; "
        "false negatives — i.e. silently skipping a clearly-implied "
        "variant — is what this catalogue is here to prevent.)",
    ])
    return lines


def _section_backends_grid_catalogue(*, framework: str) -> list[str]:
    """N20-A: render the registered backends grid as a catalogue so the
    LLM can name a SUBSET when proposing the `backends` action.

    Each row shows: variant name, the flag(s) it sets, a one-line
    trigger hint ("when roofline shows X is hot, try me"). The names
    are stable identifiers the executor matches against in
    DEFAULT_BACKENDS_GRID / DEFAULT_VLLM_BACKENDS_GRID.

    Framework-aware: SGLang grid for sglang, vLLM grid for vllm. We
    keep the full registered set visible (no Tier 1 filter at prompt
    time) — the LLM decides which subset to try based on hot-kernel
    analysis, not based on a pre-baked heuristic.
    """
    # Lazy import to avoid a circular dep (action_executors -> prompt_builder
    # is fine, prompt_builder -> action_executors at module load is not).
    from inference_optimizer.orchestrator.action_executors.backends import (
        DEFAULT_BACKENDS_GRID,
        DEFAULT_VLLM_BACKENDS_GRID,
    )

    fw = (framework or "sglang").strip().lower()
    grid = DEFAULT_VLLM_BACKENDS_GRID if "vllm" in fw else DEFAULT_BACKENDS_GRID
    fw_label = "vLLM" if "vllm" in fw else "SGLang"

    # Per-variant trigger hint. The LLM reads roofline analysis.md to
    # find which kernel category dominates GPU time; then maps that to
    # a variant whose `note` tag matches. We keep this map small +
    # explicit (no auto-generation) so the prompt stays deterministic.
    HINT_BY_NOTE: dict[str, str] = {
        "tier1_attention":     "attention-heavy traces (FlashAttn/AITER bound)",
        "tier1_decode_attn":   "decode-bound traces (long decode chains)",
        "tier2_schedule":      "queue-bound traces (high prefill queue depth)",
        "tier2_overlap":       "overlap conflicts (decode stalls on prefill)",
        "tier3_fusion":        "fused-MoE or chunked-prefill candidates",
        "tier4_moe":           "MoE-dominant traces (expert dispatch hot)",
        "tier5_comm":          "AllReduce-heavy traces (TP comm bound)",
        "kv_cache":            "KV-cache memory bound traces",
        "memory":              "GPU-memory ceiling pressure",
        "scheduling":          "request-batching bound traces",
        "compile":             "graph-capture / compile bound",
        "compile_off":         "compile causing regressions (eager-only test)",
        "cuda_graph":          "CUDA-graph capture bound",
        "rocm_aiter":          "AITER toggle (rocm general)",
        "rocm_aiter_linear":   "linear ops AITER bound",
        "rocm_aiter_rmsnorm":  "rmsnorm-hot traces",
        "rocm_aiter_fp8bmm":   "FP8 BMM hot in attention",
        "rocm_fp4":            "FP4 GEMM workloads",
        "rocm_rope":           "RoPE-bound traces (long context)",
        "rocm_collectives":    "AllReduce / collectives bound (ROCm path)",
        "rocm_buffer":         "buffer-op corruption suspected (regression test)",
        "rocm_scratch":        "scratch-reclaim caused stalls",
        "rocm_kv_layout":      "KV-layout shuffle experiment",
        "attention_backend":   "attention backend swap (vLLM ROCM_AITER_FA)",
        "cache":               "prefix-cache off (some MoE workloads)",
        "cache_mla":           "block-size 1 (MLA-specific cache)",
        "prefill":             "prefill-throughput bound (long prompts)",
    }

    lines: list[str] = [
        f"## BACKENDS GRID CATALOGUE ({fw_label})",
        "",
        "Registered variants you may name in "
        "`params.variants=[...]` when proposing `backends`. The "
        "executor will run only the ones you list, in order. Pick "
        "variants whose trigger hint matches the dominant pattern "
        "in the roofline analysis.md (e.g. if analysis says "
        "attention is 40% of GPU time, name only `attn_*` variants).",
        "",
        f"{'name':28s}  {'flag(s) / env(s)':50s}  trigger hint",
        f"{'-' * 28}  {'-' * 50}  {'-' * 60}",
    ]
    for v in grid:
        # Truncate long flag strings so the catalogue stays readable.
        flag = (v.extra_sglang_args or "").strip()
        if v.extra_envs:
            env_repr = " ".join(
                f"{k}={v}" for k, v in sorted(v.extra_envs.items())
            )
            flag = (
                f"{flag} {env_repr}" if flag else f"env: {env_repr}"
            )
        if len(flag) > 50:
            flag = flag[:47] + "..."
        hint = HINT_BY_NOTE.get(v.note or "", v.note or "(generic)")
        lines.append(f"{v.name:28s}  {flag:50s}  {hint}")

    lines.extend([
        "",
        "Example (roofline shows attention 38% + AllReduce 22%):",
        "  delegate{action_name='backends', params={variants: "
        "['attn_aiter','attn_triton','custom_ar']}, "
        "predicted_gain_pct: 1.5}",
        "",
        "When in doubt, name ALL variants whose trigger hint matches "
        "the dominant pattern(s) in analysis.md (no upper cap; the "
        "executor has no max_candidates_per_round limit by default). "
        "When analysis.md flags multiple bottleneck categories include "
        "every relevant variant across all flagged categories. Only "
        "skip variants whose trigger hint is clearly orthogonal to "
        "what analysis.md surfaces.",
        "",
        "### N22 ENFORCED MAPPINGS (analysis.md keyword -> required variants)",
        "",
        "Same PolicyGate advisory applies to `backends` proposals. The",
        "machine-readable map lives in `_analysis_keyword_map.py`; the "
        "backends-relevant subset:",
        "",
        "  analysis.md mentions ...     -> include variant(s)",
        "  -----------------------       -------------------",
        "  'attention backend' /        -> attn_aiter, attn_triton,",
        "  'flash attention' / 'aiter'     decode_aiter",
        "  'decode bound' / 'long decode' -> decode_aiter",
        "  'MoE' / 'mixture of experts' /-> moe_aiter, enable_fused_moe",
        "  'expert routing/dispatch'",
        "  'kernel fusion' / 'fusion'   -> enable_fused_moe, enable_mixed",
        "  'scheduling' / 'schedule'    -> sched_lpm, sched_dfs",
        "  'overlap' / 'multi-stream'   -> sglang_multi_stream_overlap",
        "                                  (params-side, but listed for",
        "                                   completeness across catalogues)",
        "  'AllReduce' / 'NCCL' / 'RCCL' -> custom_ar",
        "  'collective communication'   -> custom_ar",
        "",
        "(Same false-positive-tolerable, false-negative-blocking contract",
        "as the params catalogue.)",
    ])
    return lines


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
        _section_action_catalogue(actions),
        _section_decision_framework(kernel_enabled=kernel_enabled),
    ]
    # N20-A: render the registered backends grid catalogue so the LLM
    # can name a SUBSET (params.variants=['name', ...]) when proposing
    # the `backends` action. Only emit when backends is in this run's
    # enabled-action set; otherwise the catalogue is dead weight in
    # the prompt.
    if any(a.name == "backends" for a in actions):
        sections.append(
            _section_backends_grid_catalogue(framework=framework_norm),
        )
    # N20-A params: same conditional render, separate catalogue. The
    # params grid is ~3x larger than backends so subset selection
    # has larger leverage here.
    if any(a.name == "params" for a in actions):
        sections.append(
            _section_params_grid_catalogue(framework=framework_norm),
        )
    if kernel_enabled and any(a.name == "kernel_opt" for a in actions):
        sections.append(_KERNEL_OPT_PIPELINE_BODY.splitlines())
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
