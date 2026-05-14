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
    6. KERNEL-OPT PIPELINE               (only when "kernel_opt" is enabled)
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
    "setup", "classify", "target_analysis", "baseline",
    # analysis
    "profile", "pmc_roofline", "deep_kernel_analysis",
    # explore
    "backends", "params", "sweep",
    # deep — kernel-owned, emitted via REQUEST{target_agent='kernel', kind=...}
    "kernel_opt", "integrate", "operator_tuning", "vendor_kernel_config",
    # validate (Phase 3 — closes the loop on accumulated KEEPs)
    "validate_stack",
    # finalize
    "report",
    # support phase intentionally removed — comm_optimization /
    # compiler_tuning / dream / re_explore / recover are all _noop_prep
    # (cli._NOOP_KINDS_COMMON) and not in the Critic's default approve
    # whitelist; surfacing them only produced silent "succeeded" spins
    # and rejected proposals. Re-add when real executors land (see
    # remain_todo.md sections C, I, M).
)

NO_KERNEL_ENABLED_ACTIONS: tuple[str, ...] = (
    # prep
    "setup", "classify", "target_analysis", "baseline",
    # explore (no profile — it only feeds kernel-opt)
    "backends", "params", "sweep",
    # validate (still useful — bench the stacked backends/params)
    "validate_stack",
    # finalize
    "report",
    # support phase (dream / re_explore) removed for the same reason
    # as FULL_ENABLED_ACTIONS.
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
) -> list[str]:
    obj = f"{objective_kind}"
    if objective_value not in (None, ""):
        obj = f"{objective_kind}={objective_value}"
    return [
        "## 2. SESSION CONTEXT",
        "",
        f"- framework        : {framework}",
        f"- kernel_enabled   : {'true' if kernel_enabled else 'false'}",
        f"- objective        : {obj}",
        f"- max_minutes      : {max_minutes}",
        "",
        "Per-tick dynamic context (Mission progress, Time budget, Shared",
        "session state, Coordinator checklist, KB hints, inbox tail) is",
        "appended below the system prompt every tick by the Coordinator.",
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
            kind_hint = "select_kernels  -> run_optimization"
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
            "GRID OVERRIDE (T1/T2): emit "
            "`delegate{action_name='backends', params={grid: [{name, "
            "extra_sglang_args, extra_envs, note}, ...], "
            "synergy_groups?: [[name1,name2], ...] | synergy_mode?: 'auto'}}` "
            "to add candidates beyond the shipped DEFAULT_BACKENDS_GRID. "
            "See SharedState.discovered_flags for the live framework's "
            "full flag namespace and SharedState.backend_winners_history "
            "for prior-round winners worth combining."
        )
    if name == "params":
        return (
            "GRID OVERRIDE (T1/T2): emit "
            "`delegate{action_name='params', params={grid: [{name, "
            "extra_sglang_args, extra_envs, note}, ...]}}` to add value-"
            "filled parameter candidates. SharedState.discovered_flags "
            "lists the param flag namespace (--max-num-seqs, "
            "--cuda-graph-max-bs, etc.) you can fill in."
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
    ])
    return lines


_KERNEL_OPT_PIPELINE_BODY: str = """\
## 6. KERNEL-OPT PIPELINE (sequential, no backtracking)

Step **K1** (skip when cached). Emit:
    request{target_agent: 'kernel', kind: 'select_kernels',
            params: {trace_input: <verbatim last_profile_trace>, top_k: 10}}

  STRICT: if `last_select_kernels.trace_input` already equals
  `last_profile_trace`, the candidate list is cached — skip K1 and go to K2
  using `last_select_kernels.candidates_path` and the kernel_id list under
  `last_select_kernels.top5`. Re-emit `select_kernels` ONLY after a fresh
  `profile` action.

Step **K2**. Pick the next reusable native kernel from
`last_select_kernels.reusable_native_kernel_ids`, in order, skipping any
kernel_id already present in `last_kernel_opt.kernel_id`. HARD RULES:
  - kernel_id MUST appear in `reusable_native_kernel_ids`. Never pick from
    raw `hot_kernels_top15` if the entry is missing — top hot kernels are
    often Tensile / CK / vendor binaries and will be rejected with
    `non_reusable_kernel`.
  - If `reusable_native_kernel_ids` is empty, do NOT keep emitting
    run_optimization. Heartbeat instead and consider re-profiling.
Then emit:
    request{target_agent: 'kernel', kind: 'run_optimization',
            params: {kernel_id: <picked kernel_id>,
                     source_file: <hot_kernels[i].source_file>,
                     candidates_path: <select_kernels_done.candidates_path>,
                     backends: 'claude',
                     budget_minutes: 60}}

Step **K3**. When `run_optimization_done` arrives, look at
`result.proposal.decision`:
  - **KEEP**     -> emit request{kind: 'integrate', params: {kernel_id, patch_path,
                                target_file, base_tput, extra_sglang_args, config_path}}
  - **PARTIAL/REVERT** -> don't integrate; pick the NEXT hot kernel (skip
                          kernels with kernel_id == last_kernel_opt.kernel_id)
                          and re-issue step K2 with that one.

After every successful `integrate` (KEEP), the Coordinator records a new
entry on `optimization_stack`. The next decision framework tick will trigger
the validate_stack TODO; obey it before resuming kernel-opt rounds.

### KERNEL TARGETING (native vs torch.compile)

`select_kernels` profiles the *final* serving mode (with or without
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
        ),
        _section_pipeline_and_budget(actions, max_minutes=max_minutes),
        _section_action_catalogue(actions),
        _section_decision_framework(kernel_enabled=kernel_enabled),
    ]
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
