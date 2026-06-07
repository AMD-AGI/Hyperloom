"""SharedState — DESIGN v0.6 §8.3 / §17.2.

Persistent session-level state that all reactors read (via prompt injection)
and that PolicyGate uses to enforce CORE_STATE_FIELDS guards.

Backed by JSON at ``$SESSION_DIR/state.json``. The file write is atomic
(``tmp`` + ``os.replace``) so concurrent readers never see a partial blob.
The Coordinator is the **only** writer; LLM agents go through
``UPDATE_STATE`` intents which the Coordinator validates + persists.

v0.6 fields:

    session_id          str   — set by Coordinator at session creation
    model_name          str   — e.g. "meta-llama/Llama-3.1-8B-Instruct"
    model_path          str   — local NFS path to weights
    model_class         str   — set by `classify` action
    target_summary      str   — set by `target_analysis` action
    baseline_tput       float — tok/s/GPU after `baseline` action
    baseline_accuracy   float — GSM8K score after `baseline`
    current_best        dict  — {action: str, tput: float, accuracy: float}
    cumulative_gain     float — % over baseline
    stop_reason         str   — set when graceful stop fires (§9)
    current_action      str   — what's running right now (set by Orchestration)
    crash_count         int   — incremented by Robustness on real failures
    pruned_families     list[str]  — set by Robustness via PRUNE_BRANCH
    start_ts            str   — ISO timestamp
    max_minutes         int   — wall-clock budget (0 = unlimited)
    last_profile_trace  str   — set by Coordinator when `profile` returns a
                                trace path; consumed by Orch to populate
                                `trace_analyze` REQUEST `trace_input` param
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .scoring import (
    ActionScore,
    rank_top_k as _rank_top_k,
    target_gap_multiplier as _target_gap_multiplier,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


# Default partial-attempt cap for run_optimization. kernel_opt is an
# expensive action (60–120 min p75); 2 attempts already burns 2–4 h of
# budget on a single kernel, so retiring the kernel after the second
# PARTIAL is the right balance between giving the LLM a second swing and
# bailing out before a deterministic dead-end (auth-loop / unsupported
# backend) consumes the whole run. Override via the matching env var
# named in ``record_kernel_opt`` (1 disables the second-chance entirely).
_DEFAULT_KERNEL_OPT_MAX_PARTIAL = 2
# PR-C: a completed backend ladder (GEAK -> Claude -> Codex) that did
# not produce a KEEP is the operator's definition of "this kernel
# cannot be optimized". One such failure retires the kernel for the
# rest of the session; the LLM does not get to re-dispatch the same
# kernel via a fresh ``run_optimization`` request. Override via
# ``INFERENCE_OPTIMIZER_KERNEL_OPT_MAX_FAILURES`` (>=1).
_DEFAULT_KERNEL_OPT_MAX_FAILURES = 1
# PR-C: hot-kernel report gate threshold. Reusable hot kernels with
# ``gpu_pct`` >= this value MUST get at least one kernel_opt attempt
# (or be rejected after one) before ``report`` is allowed to fire.
_DEFAULT_HOT_KERNEL_MIN_GPU_PCT = 3.0
# PR-C: cap how many hot kernels the report-gate will demand. Even on
# a noisy trace with 15 reusable rows, only the top-N by gpu_pct are
# enforced -- avoids the LLM stalling on dozens of tiny kernels.
_DEFAULT_HOT_KERNEL_GATE_TOP_N = 5
# PART C (DEFER_TO_E2E): the integrate E2E budget per kernel. Mirrors
# ``record_kernel_integrate_result``'s ``max_attempts`` default so a
# DEFER_TO_E2E candidate is routed to integrate only while the same E2E
# budget allows; once exhausted it drops out of the integrate queue and
# falls back to its non-deferred disposition.
_DEFAULT_MAX_E2E_ATTEMPTS = 3

# Per-action audit history cap. ``<action>_attempts`` lists keep the most
# recent N entries (full audit trail — both successes and failures) so
# the Orchestration prompt has a bounded but informative view of each
# action's history. 20 is large enough to span a few rounds of a grid
# action without unbounded growth.
_DEFAULT_ATTEMPTS_HISTORY = 20

# Global ``last_action_failures`` rolling-log cap. 10 entries covers the
# typical "what blew up in the last few ticks" view without bloating the
# prompt; older failures stay in the event log but drop from the prompt.
_DEFAULT_LAST_FAILURES = 10

# Set of action kinds that participate in the kernel-equivalent per-action
# audit trail (Plan: SharedState audit-trail). Kernel-owned actions are
# intentionally excluded — they already have richer dedicated structures
# (last_kernel_opt / kernel_opt_attempts / kernel_integrate_attempts /
# rejected_kernel_*). Membership is consulted by Coordinator and renderer
# helpers so adding a new audit action is a one-line change.
_AUDIT_ACTIONS: frozenset[str] = frozenset({
    "baseline", "profile", "backends", "params", "sweep", "validate_stack",
    # Roofline-v2 N10: roofline composite action audit so each
    # invocation lands in `roofline_attempts` (counted by N7
    # verify/audit scripts as roofline_action_count). See
    # coordinator.py _AUDIT_ACTIONS for the matching counterpart.
    "roofline",
})

# Mapping from audit-action name to (result-dict key, prompt-display label).
# ``key`` is what we read out of the executor result dict; ``label`` is the
# ``key_metric_kind`` written into each attempt entry so prompt readers
# know how to interpret the number (e.g. ``output_throughput`` vs raw
# ``gain_pct`` vs ``validated_gain_pct``).
_KEY_METRIC_MAP: dict[str, tuple[str, str]] = {
    "baseline":       ("output_throughput", "output_throughput"),
    "profile":        ("output_throughput", "output_throughput"),
    # N10: roofline composite action — `snapshot_id` is the natural
    # progress metric (each successful run produces a fresh snapshot).
    "roofline":       ("snapshot_id",       "snapshot_id"),
    "backends":       ("gain_pct",          "gain_pct"),
    "params":         ("gain_pct",          "gain_pct"),
    "sweep":          ("output_throughput", "output_throughput"),
    "validate_stack": ("gain_pct",          "validated_gain_pct"),
}


@dataclass
class SharedState:
    session_id: str = ""
    # Primus-Claw session UUID (when Hyperloom runs inside the claw
    # sandbox); empty when running standalone. Wired into the manifest +
    # session_breakdown.json so downstream dashboards can join Hyperloom
    # sessions back to claw sessions without an env-var lookup.
    claw_session_id: str = ""
    # Primus-Claw sandbox user id (string). Same provenance as
    # ``claw_session_id``; empty when running standalone.
    sandbox_user_id: str = ""
    model_name: str = ""
    model_path: str = ""
    model_class: str = ""
    framework: str = ""
    gpu_type: str = ""
    kernel_enabled: bool = True
    # Framework-agent (fa) bandit-arm toggle. When False the
    # ``framework_pr`` arm is unregistered + stripped from the orchestration
    # prompt so the bandit can never pull it. Default True (fa is on by
    # default; mirrors ``kernel_enabled``). Older state.json files lacking
    # this field decode as True via the dataclass default — see
    # ``SharedState.load``.
    framework_enabled: bool = True
    target_summary: str = ""
    baseline_tput: float = 0.0
    baseline_accuracy: float = 0.0
    baseline_failure_streak: int = 0
    # N27 (May 2026): consecutive roofline-action failures since the
    # last successful roofline. Bumped by Coordinator's
    # ``_promote_to_shared_state`` (roofline branch) whenever the
    # executor returns ``status=failed`` (covers profile sub-step
    # failures, trace_analyze sub-step failures, and N26 auto-retry
    # exhausted). Reset to 0 the moment a roofline succeeds (i.e.
    # ``last_trace_analyze.analysis_md_text`` gets re-populated by
    # the C1 recorder).
    #
    # Read by ``_sequence_denial_for_action``: once this reaches
    # ``INFERENCE_OPTIMIZER_ROOFLINE_FAILURE_FALLBACK_THRESHOLD``
    # (default 2), the roofline-required gate on
    # ``backends`` / ``params`` / ``comm_optimization`` downgrades
    # from PolicyDenied to PASS-with-advisory. That lets the LLM
    # fall back to the pre-roofline default-grid behaviour rather
    # than hard-looping on roofline forever (the empirical case is
    # rocprofiler-sdk corner cases or splitter chunk-quality
    # failures that can't be auto-recovered by N26).
    #
    # IMPORTANT: streak only bumps on the OUTER roofline action
    # status -- N26 inner retry attempts are NOT counted here
    # because they live inside a single roofline action's
    # execution. Two outer roofline failures imply at least
    # 3-4 trace_analyze attempts already happened (each outer call
    # tried mixed -> retry to alternate mode via N26).
    roofline_failure_streak: int = 0
    # N33 (May 2026): consecutive coordinator ticks in which nothing
    # happened that would advance the run -- i.e. no queued tasks, no
    # running tasks, no pending proposals (so no LLM proposal landed
    # this tick either) and ``current_action`` is empty. The Coordinator
    # tick loop bumps this when its end-of-tick snapshot matches the
    # "idle" definition above; any change (new proposal, new task,
    # stack growth) resets it back to 0.
    #
    # Read by the tick loop after the wall-clock-deadline check: once
    # this exceeds ``INFERENCE_OPTIMIZER_IDLE_CLOSE_TICKS`` (default
    # 120 -- ~10 min at the 5s sleep used in prod), the loop calls
    # ``_enter_closing_phase`` to enqueue the final report instead of
    # idling until the wall-clock deadline. This is the canonical
    # remedy for "LLM has gone silent / refuses to propose anything
    # actionable" cases that would otherwise burn the entire budget
    # before the operator gets a report. Reset to 0 the moment we
    # actually enter closing (the same field is the trigger, so we
    # don't want the early-close to fire twice).
    consecutive_silent_ticks: int = 0
    # Path to the YAML the baseline executor materialized with the operator's
    # workload envs (CONC/ISL/OSL/TP/MAX_MODEL_LEN/PRECISION/RUN_EVAL/...).
    # Coordinator injects this into params/backends/sweep tasks as
    # ``task.params["config_path"]`` so downstream variants inherit the same
    # workload contract baseline ran. Empty before the first baseline result;
    # downstream executors fall back to materializing the shipped YAML
    # against current process env when this is empty.
    baseline_config_path: str = ""
    current_best: dict[str, Any] = field(default_factory=dict)
    # Full accepted configuration stack across action families. Each entry
    # records the incremental candidate that was accepted; current_best keeps
    # the materialized full args/env for execution.
    optimization_stack: list[dict[str, Any]] = field(default_factory=list)
    # Parallel to ``optimization_stack``: per-entry incremental gain in
    # percent (current_best vs. baseline at the moment that stack entry
    # was promoted). Index ``i`` here aligns with index ``i`` in
    # ``optimization_stack``. session_breakdown's capability_summary uses
    # this to attribute "how much of the validated cumulative gain came
    # from this action / capability" without re-walking the event log.
    # Coordinator appends to this list at the same time it appends to
    # ``optimization_stack`` (see ``_lift_to_current_best``); missing
    # entries (e.g. on resumed sessions) are treated as ``None``.
    gain_per_stack_entry: list[float | None] = field(default_factory=list)
    cumulative_gain: float = 0.0
    # Cumulative gain measured by the `validate_stack` action — i.e. by
    # actually re-baselining a fresh server with EVERY KEEP'd entry of
    # ``optimization_stack`` applied. The plain ``cumulative_gain`` field
    # only sums per-round gains (which do not compose linearly), so the
    # validated number is what the final report quotes. Stays 0.0 until the
    # first successful validate_stack run.
    cumulative_gain_validated: float = 0.0
    cumulative_gain_validated_ts: str = ""
    # Length of ``optimization_stack`` at the time of the last successful
    # validate_stack run; used by the Coordinator to decide whether the
    # current stack still matches the validated number, or whether a
    # re-validation is required after new KEEPs landed.
    cumulative_gain_validated_stack_len: int = 0
    stop_reason: str = ""
    # Closing phase — set when the wall-clock deadline fires. While True,
    # Coordinator skips reactor passes and only pumps the dispatcher to
    # drain a Coordinator-enqueued ``report`` task. Cleared on resume.
    closing_phase: bool = False
    closing_started_unix: float = 0.0
    closing_report_task_id: str = ""
    current_action: str = ""
    crash_count: int = 0
    pruned_families: list[str] = field(default_factory=list)
    start_ts: str = field(default_factory=_now_iso)
    max_minutes: int = 0
    last_profile_trace: str = ""
    # ``succeeded`` / ``failed`` for the most recent profile attempt. When
    # ``failed`` (e.g. ``no_trace_files``), Orchestration may re-run profile
    # even though ``last_profile_trace`` is non-empty from a prior bad run.
    last_profile_status: str = ""
    # Rolling log of PolicyGate denials (newest last, cap 50).
    policy_denial_history: list[dict[str, Any]] = field(default_factory=list)
    # Per-(action_name, rule) consecutive denial counter.
    policy_denial_streak: dict[str, int] = field(default_factory=dict)
    # Set when AST flag discovery cannot locate framework source files.
    discovered_flags_error: str = ""
    # Server EXTRA_SGLANG_ARGS in effect when last_profile_trace was captured.
    # Orchestration uses this to decide whether re-profiling would change the
    # hot-kernel distribution; identical args means the same trace.
    last_profile_args: str = ""
    last_profile_pmc_summary: str = ""
    last_profile_roofline: str = ""
    last_profile_kernel_breakdown: str = ""
    # Cached result of the most recent `trace_analyze` request keyed by
    # `trace_input`. Coordinator short-circuits subsequent identical requests
    # so Orchestration does not waste budget re-analysing the same trace.
    last_trace_analyze: dict[str, Any] = field(default_factory=dict)
    # N31 (May 2026): freeze the FIRST successful roofline snapshot
    # ("baseline" snapshot) so the final `report` can render a
    # before/after Roofline Comparison section. The Coordinator writes
    # this field on the first promotion of `roofline` and never updates
    # it after -- subsequent rooflines (N31 gate exception, or future
    # snapshot refreshes) only update ``last_trace_analyze``. Carries
    # minimal fields needed by ``report.py``'s
    # ``## Roofline Comparison`` section:
    # * ``roofline_snapshot_id`` -- always 1 (the first)
    # * ``analysis_md_path``     -- path to the on-disk analysis.md
    # * ``trace_input``          -- which profile trace produced it
    # * ``ts``                   -- when it was promoted
    # We deliberately do NOT freeze ``analysis_md_text`` here -- the
    # path is enough (report extracts the Executive Summary at render
    # time) and the text can be hundreds of KB.
    last_trace_analyze_baseline: dict[str, Any] = field(default_factory=dict)
    # Roofline-v2 N19c (May 2026): "gain-driven kernel_opt unlock" +
    # "flags-conditional roofline" replace the N14 counter-driven
    # (>=2/>=2/>=3) hard requirements. Two new fields drive the gates.
    #
    # `last_cheap_delta_gain` — delta_gain_pct of the last completed
    # backends/params attempt vs SharedState.current_best (NOT vs
    # baseline_tput; we care about marginal improvement from this round
    # alone, not the cumulative gain). The Coordinator writes this in
    # `_promote_to_shared_state` for task_kind in ("backends","params")
    # using the same `gain_vs_cb` calculation it already does for
    # promotion. None / 0.0 means "this round found no improvement",
    # which is the signal N19 reads to unlock kernel_opt (cheap
    # exhausted) and N21 reads to deny redundant roofline re-runs.
    #
    # `discovered_flags_at_last_snapshot` — frozen snapshot of
    # `discovered_flags` at the moment a roofline action completes.
    # The Coordinator writes this in `_promote_to_shared_state` for
    # task_kind == "roofline" by deep-copying the current
    # `discovered_flags`. N21 compares against the current
    # `discovered_flags` to detect "flags unchanged since snapshot",
    # in which case re-running roofline would produce a byte-equivalent
    # trace (sglang launch args identical).
    last_cheap_delta_gain: float | None = None
    discovered_flags_at_last_snapshot: dict[str, Any] = field(default_factory=dict)
    # Roofline-v2 N22: advisory messages from the PolicyGate's keyword-
    # implied variant check. Each entry is a self-contained block of
    # operator-facing text describing what the LLM missed and which
    # analysis.md keyword triggered the advisory. Capped FIFO so a
    # long-running session doesn't grow this list unbounded; the
    # rendered orchestration prompt shows the most recent N (see
    # prompt_builder._section_session_context's last_proposal_advice
    # block). Empty = no outstanding advisories.
    last_proposal_advice: list[str] = field(default_factory=list)
    # Most recent workload sweep; used to reason about gains beyond the
    # smoke workload (CONC/ISL/OSL frontier).
    last_sweep: dict[str, Any] = field(default_factory=dict)
    # Kernel-opt response tracking — Coordinator records the most recent
    # `run_optimization_done` so Orch sees what's been tried and doesn't
    # re-dispatch the same kernel_id every tick.
    last_kernel_opt: dict[str, Any] = field(default_factory=dict)
    # ---------------------------------------------------------------
    # Per-action audit (kernel parity for non-kernel actions). Each
    # ``last_<action>`` mirrors :attr:`last_kernel_opt`: a single snapshot
    # dict of the most recent attempt (success or failure). The matching
    # ``<action>_attempts`` is a flat capped list (newest last) with one
    # entry per attempt — the uniform schema is documented on
    # :meth:`record_action_attempt`. ``last_sweep`` already exists above
    # and acts as sweep's snapshot; ``sweep_attempts`` is added here for
    # symmetry.
    last_baseline: dict[str, Any] = field(default_factory=dict)
    last_profile: dict[str, Any] = field(default_factory=dict)
    last_backends: dict[str, Any] = field(default_factory=dict)
    last_params: dict[str, Any] = field(default_factory=dict)
    last_validate_stack: dict[str, Any] = field(default_factory=dict)
    # Roofline-v2 N10: composite roofline action audit snapshot +
    # rolling history. Mirrors the v0 per-action audit pattern (one
    # dict snapshot for "what was the most recent run", one capped
    # list for "what was the per-tick history"). Counted by N7's
    # verify_roofline_v2 / audit_roofline_decisions scripts.
    last_roofline: dict[str, Any] = field(default_factory=dict)
    baseline_attempts: list[dict[str, Any]] = field(default_factory=list)
    profile_attempts: list[dict[str, Any]] = field(default_factory=list)
    backends_attempts: list[dict[str, Any]] = field(default_factory=list)
    params_attempts: list[dict[str, Any]] = field(default_factory=list)
    sweep_attempts: list[dict[str, Any]] = field(default_factory=list)
    validate_stack_attempts: list[dict[str, Any]] = field(default_factory=list)
    roofline_attempts: list[dict[str, Any]] = field(default_factory=list)
    # Global rolling log of unpromotable task results, capped at
    # ``_DEFAULT_LAST_FAILURES``. Carries the rich failure context
    # (error_class / error_excerpt / stderr_tail / workspace /
    # raw_result_path / reported_success) so Orchestration sees enough
    # context to self-correct even when the inbox has rotated past the
    # original ``delegated_result`` event. Populated by
    # :meth:`Coordinator._handle_unpromotable_result`. Covers every task
    # kind (including kernel-owned actions), not just the audit set.
    last_action_failures: list[dict[str, Any]] = field(default_factory=list)
    # Per-kernel run_optimization attempt history keyed by kernel_id.
    # Each entry: {"attempts": int, "partial_count": int, "last_decision": str,
    #              "last_ts": str, "history": [{"decision","ts"}...max 10],
    #              "rejected_reason": str (only when retired)}.
    # `record_kernel_opt` retires kernels whose run_optimization keeps
    # returning PARTIAL (no measurable speedup) — the prior policy only
    # retired on REVERT, so a kernel stuck in PARTIAL/PARTIAL/... burned
    # the whole wall-clock budget on the same dead-end (e.g. the r24
    # custom_allreduce loop with inner GEAK 401-retry that prompted this
    # field). Threshold defaults to 2 PARTIAL outcomes; override via
    # ``INFERENCE_OPTIMIZER_KERNEL_OPT_MAX_PARTIAL``.
    kernel_opt_attempts: dict[str, Any] = field(default_factory=dict)
    # Cross-round params/backends/sweep aggregation. Each entry is
    # {action, variant_name, tput, gain_pct, ts}; we cap the list at 10
    # rows so the prompt summary stays bounded. Used by
    # `_promote_to_shared_state` to detect a "consistent winner that's
    # below the 1-shot threshold but consistent across rounds" pattern
    # the resume5 9h run hit (best variant +0.5–0.8% across 38 rounds,
    # but never promoted because each single run sat under the 1.0% bar).
    params_winner_history: list[dict[str, Any]] = field(default_factory=list)
    # How many CONSECUTIVE grid-runner (params/backends/sweep) tasks
    # finished without producing a new current_best. Robustness uses
    # this to nudge Orch off the params plateau. Reset to 0 whenever
    # current_best advances.
    params_no_promote_streak: int = 0
    # Persistent params DFS state. ParamsExecutor owns the search mechanics,
    # Coordinator is still the only writer to state.json.
    params_search: dict[str, Any] = field(default_factory=dict)
    # Persistent backends DFS state — same schema as ``params_search``
    # (``schema_version`` / ``accepted`` / ``rejected`` / ``tested`` /
    # ``name_index`` / ``cursor`` / ``last_round``). Owned by
    # BackendsExecutor; Coordinator merges via
    # :meth:`apply_backends_search_update` after each round and appends to
    # ``accepted`` on promote (see :meth:`record_backends_accepted`).
    #
    # ``tested`` is keyed by **content fingerprint** (see
    # :func:`variant_fingerprint`) so two variants with identical
    # ``extra_sglang_args`` + ``extra_envs`` under different names collapse
    # to the same row. ``name_index`` is a name → fingerprint map used by
    # the executor's pre-filter to also reject explicit renames that the
    # LLM might submit in a fresh ``params.grid``.
    backends_search: dict[str, Any] = field(default_factory=dict)
    # E2E integrate bookkeeping keyed by kernel_id + patch_path + args. This
    # prevents Orchestration from spending hours re-validating the same patch
    # after repeated NEEDS_REVIEW/REVERT outcomes.
    kernel_integrate_attempts: dict[str, Any] = field(default_factory=dict)
    rejected_kernel_patches: list[dict[str, Any]] = field(default_factory=list)
    # Kernel ids with no remaining automated path. This is fed by
    # run_optimization REVERTs and exhausted integrate attempts.
    rejected_kernel_ids: list[str] = field(default_factory=list)

    # T1+T2 (search-space expansion) — see SKILL.md "Search-space expansion".
    # Populated once per session by BackendsExecutor / ParamsExecutor on the
    # first run after they AST-parse the live framework's server_args.py.
    # Schema: {framework: {"backend_flags": [...], "param_flags": [...],
    #                       "ts": iso, "source_path": str}}.
    # The Orchestration prompt surfaces this so the LLM knows the full
    # framework-version-correct flag namespace it can synthesize variants
    # from (instead of being limited to the shipped DEFAULT_*_GRID).
    discovered_flags: dict[str, Any] = field(default_factory=dict)
    # Rolling per-action winners log used for IR-26 dynamic idea generation.
    # Each entry: {action, round_id, base_tput, winners: [{name, tput,
    # gain_pct, extra_sglang_args, extra_envs}], best: {...}, ts}.
    # Capped at 20 rows to keep prompt context bounded.
    backend_winners_history: list[dict[str, Any]] = field(default_factory=list)
    # Set of synergy combo keys ("name1+name2+...") that have already been
    # tested this session, so the IR-26 re-explore loop doesn't re-run the
    # same combination after each new round of explore. Populated by
    # BackendsExecutor when phase-2 combos run.
    synergy_attempted: list[str] = field(default_factory=list)

    # ---------------------------------------------------------------
    # Action scoring (see orchestrator/scoring.py + plan
    # action-scoring-in-shared-state). Coordinator seeds ``action_scores``
    # once at session start from ActionRegistry + marathon priors and
    # mutates it after every task completion. Each value is the raw dict
    # returned by ``ActionScore.to_dict()`` so JSON serialization is
    # transparent. Use :meth:`get_action_score` / :meth:`put_action_score`
    # to round-trip via the typed dataclass.
    action_scores: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Monotonic Coordinator tick counter. Drives cooldown + aging math in
    # scoring.py. Bumped once per Coordinator.run() / Coordinator.tick(n)
    # iteration.
    tick: int = 0
    # Remaining gain-pct target gap (0.0 means "no target"). Coordinator
    # refreshes this each prompt build when the run objective is
    # ``gain_pct=N``. Drives ``scoring.target_gap_multiplier``.
    target_gap_pct: float = 0.0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    @classmethod
    def state_path(cls, session_dir: Path) -> Path:
        return Path(session_dir) / "state.json"

    @classmethod
    def load_or_init(cls, session_dir: Path) -> "SharedState":
        """Load existing ``state.json`` or return a fresh blank instance."""
        path = cls.state_path(session_dir)
        if not path.exists():
            return cls()
        with path.open(encoding="utf-8") as f:
            raw = json.load(f)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SharedState":
        # Filter to known fields so older / newer state.json shapes don't
        # crash. Unknown keys are dropped; missing keys fall back to defaults.
        known = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in raw.items() if k in known}
        # Defensive: ``action_scores`` is supposed to be a dict-of-dict. If a
        # corrupted state.json carries a non-dict entry, drop it instead of
        # failing the whole load — the missing rows will be re-seeded on the
        # next Coordinator.start().
        if "action_scores" in filtered:
            scores = filtered["action_scores"]
            if isinstance(scores, dict):
                filtered["action_scores"] = {
                    str(k): v for k, v in scores.items() if isinstance(v, dict)
                }
            else:
                filtered["action_scores"] = {}
        # Phase 7 of the dedup-by-fingerprint plan: migrate any v1
        # ``params_search`` ledger (where ``tested`` was keyed by display
        # name) to schema v2 (keyed by content fingerprint). Backends has
        # no pre-fingerprint persisted data so it only needs default-key
        # normalization. We do this here — at the load boundary — so the
        # executor and Coordinator paths can assume the v2 schema and
        # never need a fallback branch.
        filtered["params_search"] = cls._migrate_search_ledger(
            filtered.get("params_search"), schema_target=2,
        )
        filtered["backends_search"] = cls._migrate_search_ledger(
            filtered.get("backends_search"), schema_target=1,
        )
        return cls(**filtered)

    @staticmethod
    def _migrate_search_ledger(
        ledger: Any, *, schema_target: int,
    ) -> dict[str, Any]:
        """Normalize an *_search ledger to the fingerprint-keyed schema.

        Idempotent: already-migrated ledgers are returned with only the
        defensive defaults filled in. A legacy v1 ledger whose ``tested``
        is keyed by variant name gets re-keyed by content fingerprint
        re-computed from the stored ``extra_sglang_args`` / ``extra_envs``;
        the original name is preserved inside each entry and surfaced
        through ``name_index`` so display lookups remain stable.
        """
        if not isinstance(ledger, dict) or not ledger:
            return {}
        from .action_executors._grid_runner import variant_fingerprint
        out: dict[str, Any] = dict(ledger)
        out.setdefault("schema_version", schema_target)
        out.setdefault("accepted", [])
        out.setdefault("rejected", [])
        out.setdefault("tested", {})
        out.setdefault("name_index", {})
        out.setdefault("cursor", 0)
        tested = out.get("tested") or {}
        if not isinstance(tested, dict):
            tested = {}
        # A fingerprint key is a 16-char lowercase hex string; anything
        # else is treated as a legacy display-name key.
        def _looks_like_fingerprint(key: str) -> bool:
            return (
                isinstance(key, str)
                and len(key) == 16
                and all(c in "0123456789abcdef" for c in key)
            )
        migrated: dict[str, Any] = {}
        name_index = dict(out.get("name_index") or {})
        for key, entry in tested.items():
            if not isinstance(entry, dict):
                continue
            if _looks_like_fingerprint(str(key)):
                # Already fingerprint-keyed; just ensure name_index is in
                # sync so display-name lookups also work on resume.
                fp = str(key)
                entry.setdefault("fingerprint", fp)
                migrated[fp] = entry
                nm = entry.get("name")
                if nm:
                    name_index[str(nm)] = fp
                continue
            # Legacy: key was a display name. Re-derive fingerprint from
            # stored args/envs. Older entries nested the executor's
            # full ``result`` dict under ``result``; check both.
            nested = entry.get("result") if isinstance(entry.get("result"), dict) else {}
            args = str(
                entry.get("extra_sglang_args")
                or nested.get("extra_sglang_args") or ""
            )
            envs = dict(
                entry.get("extra_envs")
                or nested.get("extra_envs") or {}
            )
            fp = variant_fingerprint(args, envs)
            new_entry = dict(entry)
            new_entry.setdefault("name", str(key))
            new_entry.setdefault("extra_sglang_args", args)
            new_entry.setdefault("extra_envs", envs)
            new_entry["fingerprint"] = fp
            migrated[fp] = new_entry
            name_index[str(key)] = fp
        out["tested"] = migrated
        out["name_index"] = name_index
        # Stamp fingerprints onto accepted/rejected too, so the executor's
        # fast-path dedup sets fill cleanly on the first resume round.
        for bucket in ("accepted", "rejected"):
            rebuilt: list[dict[str, Any]] = []
            for v in out.get(bucket) or []:
                if not isinstance(v, dict):
                    continue
                v = dict(v)
                if not v.get("fingerprint"):
                    v["fingerprint"] = variant_fingerprint(
                        str(v.get("extra_sglang_args") or ""),
                        dict(v.get("extra_envs") or {}),
                    )
                rebuilt.append(v)
                if v.get("name") and v.get("fingerprint"):
                    name_index[str(v["name"])] = str(v["fingerprint"])
            out[bucket] = rebuilt
        out["name_index"] = name_index
        # Bump the schema marker so callers can short-circuit re-migration.
        out["schema_version"] = max(int(out.get("schema_version") or 0), schema_target)
        return out

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, session_dir: Path) -> None:
        """Atomically write state.json (tmp + os.replace)."""
        path = self.state_path(session_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".state-", suffix=".json", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2, sort_keys=True)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------
    # Mutators (used by the Coordinator only — LLM agents go via intents)
    # ------------------------------------------------------------------
    def add_pruned_family(self, family: str) -> bool:
        """Idempotent add. Returns True iff the family was newly added."""
        if family in self.pruned_families:
            return False
        self.pruned_families.append(family)
        return True

    def is_pruned(self, family: str) -> bool:
        return family in self.pruned_families

    def prune_family(self, family: str) -> bool:
        """Alias for :meth:`add_pruned_family` (policy-loop stop-loss)."""
        return self.add_pruned_family(family)

    _POLICY_DENIAL_HISTORY_CAP = 50

    def record_policy_denial(
        self,
        *,
        action_name: str,
        rule: str,
        hint: str,
        intent_type: str,
        tick: int,
        intent_payload: dict[str, Any] | None = None,
    ) -> int:
        """Append a denial row and bump per-(action,rule) streak."""
        key = f"{action_name or '*'}:{rule}"
        streak = int(self.policy_denial_streak.get(key, 0)) + 1
        self.policy_denial_streak[key] = streak
        entry = {
            "tick": int(tick),
            "action_name": action_name or "",
            "rule": rule,
            "hint": hint or "",
            "intent_type": intent_type,
            "streak": streak,
            "ts": _now_iso(),
        }
        if intent_payload:
            entry["intent_payload_keys"] = sorted(intent_payload.keys())
        history = list(self.policy_denial_history or [])
        history.append(entry)
        if len(history) > self._POLICY_DENIAL_HISTORY_CAP:
            history = history[-self._POLICY_DENIAL_HISTORY_CAP :]
        self.policy_denial_history = history
        return streak

    def reset_policy_denial_streak(self, action_name: str) -> None:
        if not action_name:
            return
        prefix = f"{action_name}:"
        self.policy_denial_streak = {
            k: v
            for k, v in (self.policy_denial_streak or {}).items()
            if not k.startswith(prefix)
        }

    def to_policy_denial_summary(self, *, top_k: int = 6) -> str:
        if not self.policy_denial_history:
            return ""
        rows = list(self.policy_denial_history)[-top_k:]
        lines = [
            "=== Recent policy denials "
            f"(newest last, total={len(self.policy_denial_history)}) ==="
        ]
        for r in rows:
            lines.append(
                f"  tick={r.get('tick')} action={r.get('action_name')!r} "
                f"rule={r.get('rule')!r} streak={r.get('streak')} "
                f"hint={str(r.get('hint') or '')[:140]!r}"
            )
        return "\n".join(lines)

    def all_top_actions_policy_locked(self, registry: Any, *, top_k: int = 12) -> bool:
        """True when every visible top-K action row carries a policy_loop lock."""
        if not self.action_scores or registry is None:
            return False
        target_mult = _target_gap_multiplier(
            target_gap_pct=float(self.target_gap_pct or 0.0),
            cumulative_gain=float(self.cumulative_gain or 0.0),
        )
        rows = _rank_top_k(
            self.action_scores,
            registry,
            tick=int(self.tick or 0),
            target_gap_mult=target_mult,
            k=int(top_k),
            shared_state=self,  # N30: enable cheap-exhausted deep boost
        )
        if not rows:
            return False
        positive = [(n, eff, a) for n, eff, a in rows if eff > 0]
        if not positive:
            return False
        return all(
            str(a.locked_reason or "").startswith("policy_loop:")
            for _, _, a in positive
        )

    def increment_crash_count(self, by: int = 1) -> int:
        self.crash_count += by
        return self.crash_count

    def apply_changes(self, changes: dict[str, Any], *, allow_core: bool) -> dict[str, Any]:
        """Merge a non-empty changes dict into this state.

        ``allow_core=True`` is reserved for Coordinator-internal callers that
        update fields in :data:`policy.CORE_STATE_FIELDS` (current_best,
        baseline_tput, etc.). LLM-driven UPDATE_STATE intents pass
        ``allow_core=False`` and PolicyGate already filtered them upstream
        — this method does *not* re-validate the role/source allowlist.

        Returns the dict of fields that were actually written (may be a
        subset of input if unknown keys are passed).
        """
        if not changes:
            return {}
        applied: dict[str, Any] = {}
        for key, value in changes.items():
            if key not in self.__dataclass_fields__:
                continue
            setattr(self, key, value)
            applied[key] = value
        return applied

    def _format_last_kernel_opt(self) -> str:
        """Single-line repr of last kernel-opt outcome for prompt injection."""
        if not self.last_kernel_opt:
            return "(none)"
        ko = self.last_kernel_opt
        kid = str(ko.get("kernel_id") or "")
        attempts_entry = self.kernel_opt_attempts.get(kid) or {}
        history_tag = ""
        if attempts_entry:
            history_tag = (
                f" history=attempts={attempts_entry.get('attempts', 0)}"
                f"/partial={attempts_entry.get('partial_count', 0)}"
            )
            rej_reason = attempts_entry.get("rejected_reason")
            if rej_reason:
                history_tag += f"/retired={rej_reason}"
        return (
            f"kernel_id={kid or '?'} "
            f"decision={ko.get('decision','?')} "
            f"speedup={ko.get('micro_speedup','?')}"
            f"{history_tag}"
        )

    def _resolve_kernel_patch_identity(
        self, payload: dict[str, Any] | None,
    ) -> tuple[str, str, str, str]:
        payload = payload or {}
        kernel_id = str(payload.get("kernel_id") or "")
        patch_path = str(
            payload.get("patch_path")
            or payload.get("best_artifact_path")
            or ""
        )
        if (
            not patch_path
            and kernel_id
            and str((self.last_kernel_opt or {}).get("kernel_id") or "") == kernel_id
        ):
            patch_path = str(
                (self.last_kernel_opt or {}).get("best_artifact_path")
                or (self.last_kernel_opt or {}).get("patch_path")
                or ""
            )
        target_file = str(
            payload.get("target_file")
            or payload.get("source_file")
            or ""
        )
        extra_args = str(payload.get("extra_sglang_args") or "").strip()
        return kernel_id, patch_path, target_file, extra_args

    def kernel_patch_key(self, payload: dict[str, Any] | None) -> str:
        kernel_id, patch_path, _target_file, extra_args = (
            self._resolve_kernel_patch_identity(payload)
        )
        if not kernel_id or not patch_path:
            return ""
        return "|".join([kernel_id, patch_path, extra_args])

    def find_rejected_kernel_patch(
        self,
        payload: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        key = self.kernel_patch_key(payload)
        if not key:
            return None
        for entry in self.rejected_kernel_patches:
            if isinstance(entry, dict) and entry.get("key") == key:
                return entry
        return None

    def record_kernel_integrate_result(
        self,
        result: dict[str, Any],
        *,
        max_attempts: int = 3,
        keep_threshold_pct: float = 1.0,
    ) -> dict[str, Any] | None:
        """Persist one integrate E2E result and reject exhausted patch attempts."""
        if not isinstance(result, dict):
            return None
        key = self.kernel_patch_key(result)
        if not key:
            return None
        kernel_id, patch_path, target_file, extra_args = (
            self._resolve_kernel_patch_identity(result)
        )
        entry = dict(self.kernel_integrate_attempts.get(key) or {})
        attempts = list(entry.get("attempts") or [])
        attempt = {
            "decision": result.get("decision"),
            "status": result.get("status"),
            "new_tput": result.get("new_tput"),
            "gain_pct": result.get("gain_pct"),
            "workspace": result.get("workspace"),
            "report_path": result.get("report_path"),
            "ts": _now_iso(),
        }
        attempts.append(attempt)
        best_gain = max(
            (
                float(a.get("gain_pct"))
                for a in attempts
                if isinstance(a, dict) and isinstance(a.get("gain_pct"), (int, float))
            ),
            default=0.0,
        )
        entry.update({
            "key": key,
            "kernel_id": kernel_id,
            "patch_path": patch_path,
            "target_file": target_file,
            "extra_sglang_args": extra_args,
            "attempts": attempts,
            "attempt_count": len(attempts),
            "best_gain_pct": best_gain,
            "last_decision": result.get("decision"),
            "last_status": result.get("status"),
            "updated_at": _now_iso(),
        })
        self.kernel_integrate_attempts[key] = entry

        if result.get("decision") == "KEEP":
            return entry

        should_reject = (
            result.get("decision") == "REVERT"
            or len(attempts) >= max_attempts
        )
        if not should_reject:
            return entry

        reason = (
            "revert_decision"
            if result.get("decision") == "REVERT"
            else f"max_e2e_attempts_{max_attempts}_without_keep"
        )
        rejected = {
            "key": key,
            "kernel_id": kernel_id,
            "patch_path": patch_path,
            "target_file": target_file,
            "extra_sglang_args": extra_args,
            "attempt_count": len(attempts),
            "best_gain_pct": best_gain,
            "keep_threshold_pct": keep_threshold_pct,
            "last_decision": result.get("decision"),
            "reason": reason,
            "ts": _now_iso(),
        }
        self.rejected_kernel_patches = [
            r for r in self.rejected_kernel_patches
            if not (isinstance(r, dict) and r.get("key") == key)
        ]
        self.rejected_kernel_patches.append(rejected)
        if kernel_id and kernel_id not in self.rejected_kernel_ids:
            self.rejected_kernel_ids.append(kernel_id)
        entry["rejected"] = rejected
        self.kernel_integrate_attempts[key] = entry
        return entry

    def record_kernel_opt(self, result: dict[str, Any]) -> None:
        """Capture the result returned by kernel_optimization_handler so the
        next Orch turn knows what's already been tried (and the outcome).

        Two overwrite invariants protect the multi-KEEP integrate queue:

        * **Empty kernel_id is a no-op.** A metadata-less failure result
          (e.g. ``{"status":"failed", "error":"TimeoutExpired"}`` wrapped
          by the Coordinator's batch handler exception path) must NOT
          clobber a previously-recorded KEEP. Otherwise streaming
          ``record_partial`` writes from a batch's fast KEEPs would be
          erased the moment one slow sibling times out, and Orchestration
          would never see ``last_kernel_opt.decision == "KEEP"`` -- which
          is the only gate that opens TODO 4/5 integrate.

        * **Non-KEEP cannot overwrite a pending KEEP.** During a
          ``_run_optimization_batch`` fan-out, sub-results land in
          completion order, not micro_speedup order. A late-arriving
          REVERT / PARTIAL must not displace an earlier KEEP that hasn't
          been integrated yet. We keep the strongest unresolved KEEP in
          ``last_kernel_opt`` and let the integrate queue
          (``next_pending_keep_kernel_id``) drain via the per-kernel
          ``kernel_opt_attempts`` ledger.

        Retires ``kernel_id`` into ``rejected_kernel_ids`` when the same
        kernel has accumulated >= ``max_partial`` PARTIAL outcomes (no
        measurable speedup), not just on REVERT. Without this guard, an
        inner-tool auth failure that surfaces as PARTIAL keeps the kernel
        in ``applicable_kernel_set`` forever and Orch re-dispatches it
        every tick (the r24 custom_allreduce dead-end). Threshold defaults
        to 2 attempts; override via
        ``INFERENCE_OPTIMIZER_KERNEL_OPT_MAX_PARTIAL`` (>=1).
        """
        if not isinstance(result, dict):
            return
        kernel_id = str(result.get("kernel_id") or "")
        if not kernel_id:
            # Metadata-less failure (batch handler exception, programmatic
            # rejection, ...). Preserve last_kernel_opt + attempts so a
            # prior streaming-record KEEP survives.
            return

        verification = result.get("verification") or {}
        proposal = result.get("proposal") or {}
        decision = str(proposal.get("decision", "")).upper()
        micro_speedup = verification.get("micro_speedup", 0.0)
        try:
            micro_float = float(micro_speedup)
        except (TypeError, ValueError):
            micro_float = 0.0
        best_artifact_path = str(verification.get("best_artifact_path", "") or "")
        source_file = str(
            result.get("source_file")
            or (result.get("candidate") or {}).get("source_file")
            or ""
        )
        status = str(result.get("status") or "").lower()
        err_class = str(result.get("error_class") or "")
        # PR-C: An "infra failure" is a backend ladder that finished
        # WITHOUT delivering any verdict at all -- subprocess timeout,
        # batch handler exception, GEAK/OOB rc!=0, missing inputs, etc.
        # These are distinct from REVERT (which has its own
        # ``should_reject`` rule) and PARTIAL (which has its own
        # ``max_partial`` streak gate); we don't double-count them.
        # The new ``max_failures = 1`` rule covers ONLY this pure
        # infra-failure case: Qwen3-30B-A3B-Base 164405Z burned 8h
        # because GEAK -> Claude -> Codex timed out repeatedly on the
        # same k002/k004 and each ladder showed up as status=failed
        # with empty decision -- silently bumping attempts but never
        # tripping a retire gate.
        is_infra_failure = (
            decision == ""
            and (
                status in {"failed", "error", "timeout"}
                or err_class in {
                    "subtask_exception",
                    "handler_exception",
                    "subprocess_timeout",
                    "kernel_agent_root_missing",
                    "missing_integration_inputs",
                }
            )
        )
        ts = _now_iso()

        entry = dict(self.kernel_opt_attempts.get(kernel_id) or {})
        history = list(entry.get("history") or [])
        history.append({
            "decision": decision, "micro": micro_float,
            "status": status, "ts": ts,
        })
        history = history[-10:]
        entry["attempts"] = int(entry.get("attempts", 0)) + 1
        if decision == "PARTIAL":
            entry["partial_count"] = int(entry.get("partial_count", 0)) + 1
        elif decision in {"KEEP", "DEFER_TO_E2E"}:
            # A KEEP -- or a DEFER_TO_E2E (correctness passed, high-impact,
            # routed to integrate so the E2E gain_pct decides) -- resets the
            # streaks so a promising kernel is not auto-retired on stale
            # PARTIAL/failure history.
            entry["partial_count"] = 0
            entry["failure_count"] = 0
        if is_infra_failure:
            entry["failure_count"] = int(entry.get("failure_count", 0)) + 1
        entry["last_decision"] = decision
        entry["last_status"] = status
        entry["last_micro_speedup"] = micro_float
        entry["last_artifact_path"] = best_artifact_path
        entry["last_source_file"] = source_file
        entry["last_ts"] = ts
        entry["history"] = history
        # PART C: surface the DEFER_TO_E2E disposition on the ledger entry so
        # state inspectors / prompt rendering can see a candidate is awaiting
        # an authoritative E2E verdict (and what it falls back to).
        if decision == "DEFER_TO_E2E":
            entry["deferred_to_e2e"] = True
            entry["defer_fallback_decision"] = str(proposal.get("fallback_decision", ""))

        # Overwrite policy for last_kernel_opt:
        #   * KEEP / DEFER_TO_E2E always win (both route to integrate).
        #   * A non-promoting result only writes when there's no pending
        #     KEEP / DEFER_TO_E2E to protect (PART C: a late REVERT/PARTIAL
        #     sibling must not clobber a pending deferred candidate either).
        prev = self.last_kernel_opt or {}
        prev_decision = str(prev.get("decision", "")).upper()
        prev_kid = str(prev.get("kernel_id", ""))
        integrated_ids = self._kernel_ids_in_optimization_stack()
        prev_pending = (
            prev_decision in {"KEEP", "DEFER_TO_E2E"}
            and bool(prev_kid)
            and prev_kid not in (self.rejected_kernel_ids or [])
            and prev_kid not in integrated_ids
        )
        if decision in {"KEEP", "DEFER_TO_E2E"} or not prev_pending:
            self.last_kernel_opt = {
                "kernel_id": kernel_id,
                "decision": decision,
                "reasons": proposal.get("reasons", []),
                "micro_speedup": micro_float,
                "compile_passed": verification.get("compile_passed"),
                "correctness_passed": verification.get("correctness_passed"),
                "best_artifact_path": best_artifact_path,
                "source_file": source_file,
                "ts": ts,
            }

        max_partial = _DEFAULT_KERNEL_OPT_MAX_PARTIAL
        env_v = os.environ.get("INFERENCE_OPTIMIZER_KERNEL_OPT_MAX_PARTIAL")
        if env_v:
            try:
                max_partial = max(1, int(env_v))
            except (TypeError, ValueError):
                # Invalid env override; keep _DEFAULT_KERNEL_OPT_MAX_PARTIAL
                # (already assigned above) instead of failing.
                pass

        # PR-C: max_failures defaults to 1 -- a single completed backend
        # ladder (GEAK -> Claude -> Codex) that did not produce a KEEP
        # is the operator's definition of "this kernel cannot be
        # optimized; do not retry". Operators that want to give a
        # flaky GEAK/OOB a second chance can bump via env.
        max_failures = _DEFAULT_KERNEL_OPT_MAX_FAILURES
        env_f = os.environ.get("INFERENCE_OPTIMIZER_KERNEL_OPT_MAX_FAILURES")
        if env_f:
            try:
                max_failures = max(1, int(env_f))
            except (TypeError, ValueError):
                pass

        should_reject = (
            decision == "REVERT"
            or int(entry.get("partial_count", 0)) >= max_partial
            or int(entry.get("failure_count", 0)) >= max_failures
        )
        if should_reject:
            if kernel_id not in self.rejected_kernel_ids:
                self.rejected_kernel_ids.append(kernel_id)
            entry["rejected_reason"] = (
                "revert_decision"
                if decision == "REVERT"
                else (
                    f"max_partial_attempts_{max_partial}_without_keep"
                    if int(entry.get("partial_count", 0)) >= max_partial
                    else f"max_failures_{max_failures}_without_keep"
                )
            )

        self.kernel_opt_attempts[kernel_id] = entry

    # ------------------------------------------------------------------
    # Multi-KEEP integrate queue helpers (PR-B follow-up).
    # ------------------------------------------------------------------
    def _kernel_ids_in_optimization_stack(self) -> set[str]:
        """kernel_ids already absorbed into optimization_stack as integrate entries."""
        return {
            str(e.get("kernel_id"))
            for e in (self.optimization_stack or [])
            if isinstance(e, dict)
            and e.get("action") == "integrate"
            and e.get("kernel_id")
        }

    def _source_files_in_optimization_stack(self) -> set[str]:
        """source_file paths already touched by an integrate entry on the stack.

        Used to enforce "same source_file, only the strongest KEEP gets
        integrated" -- ``apply_kernel_patch`` is a ``shutil.copy2`` whole-
        file overwrite, so two patches on the same file would silently
        clobber each other.
        """
        sources: set[str] = set()
        for e in (self.optimization_stack or []):
            if not isinstance(e, dict) or e.get("action") != "integrate":
                continue
            src = str(e.get("target_file") or e.get("source_file") or "")
            if src:
                sources.add(src)
        return sources

    def _integrate_attempts_for_kernel(self, kernel_id: str) -> int:
        """Count integrate (E2E) attempts already recorded for ``kernel_id``.

        The ``kernel_integrate_attempts`` ledger is keyed by
        ``kernel_id|patch_path|extra_args`` (the patch identity), so a kernel
        can have several patch keys; we take the max attempt_count across the
        keys that carry this ``kernel_id``. Used to bound the PART C
        DEFER_TO_E2E routing by the existing max_e2e_attempts budget.
        """
        kid = str(kernel_id or "")
        if not kid:
            return 0
        best = 0
        for entry in (self.kernel_integrate_attempts or {}).values():
            if not isinstance(entry, dict):
                continue
            if str(entry.get("kernel_id") or "") != kid:
                continue
            count = entry.get("attempt_count")
            if count is None:
                count = len(entry.get("attempts") or [])
            try:
                best = max(best, int(count))
            except (TypeError, ValueError):
                continue
        return best

    def _defer_e2e_budget_exhausted(
        self, kernel_id: str, *, max_attempts: int | None = None,
    ) -> bool:
        """PART C: True iff a DEFER_TO_E2E candidate has already spent its
        integrate (E2E) budget.

        Bounds the defer by the SAME ledger that ``record_kernel_integrate_
        result`` enforces (default 3 attempts; override via
        ``HYPERLOOM_MAX_E2E_ATTEMPTS``) so we never blow up expensive E2E runs;
        once exhausted the candidate drops out of the integrate queue and falls
        back to its non-deferred disposition.
        """
        if max_attempts is None:
            try:
                max_attempts = int(os.environ.get(
                    "HYPERLOOM_MAX_E2E_ATTEMPTS", _DEFAULT_MAX_E2E_ATTEMPTS,
                ))
            except (TypeError, ValueError):
                max_attempts = _DEFAULT_MAX_E2E_ATTEMPTS
        max_attempts = max(1, int(max_attempts))
        return self._integrate_attempts_for_kernel(kernel_id) >= max_attempts

    def next_pending_keep_kernel_id(self) -> str:
        """Return next KEEP kernel_id awaiting integrate, or "" if drained.

        Ordering:
          1. Exclude already-integrated (in optimization_stack), rejected
             kernels, and any KEEP whose source_file is already covered
             by an earlier integrate (same-file conflict guard).
          2. Among the remaining KEEPs, pick the highest
             ``last_micro_speedup`` so the strongest lever lands first.

        Consumers: Coordinator's ``_kernel_opt_keep_pending`` for the
        TODO 4/5 integrate gate, plus the prompt_builder rendering used
        by Orchestration / robustness to see how many KEEPs are queued.
        """
        integrated_ids = self._kernel_ids_in_optimization_stack()
        integrated_sources = self._source_files_in_optimization_stack()
        rejected = set(self.rejected_kernel_ids or [])

        best_kid = ""
        best_micro = float("-inf")
        for kid, entry in (self.kernel_opt_attempts or {}).items():
            if not isinstance(entry, dict):
                continue
            # PART C: KEEP *and* DEFER_TO_E2E are routed to integrate. A
            # DEFER_TO_E2E candidate is bounded by the E2E budget -- once it
            # has used its max_e2e_attempts integrate runs without a KEEP it
            # drops out of the queue (the E2E gain_pct was authoritative) and
            # falls back to its non-deferred disposition.
            last_decision = str(entry.get("last_decision", "")).upper()
            if last_decision not in ("KEEP", "DEFER_TO_E2E"):
                continue
            if last_decision == "DEFER_TO_E2E" and self._defer_e2e_budget_exhausted(kid):
                continue
            if kid in integrated_ids or kid in rejected:
                continue
            src = str(entry.get("last_source_file") or "")
            if src and src in integrated_sources:
                # Same-file conflict: a stronger KEEP on this file was
                # already integrated (or this KEEP came after one did).
                continue
            try:
                micro = float(entry.get("last_micro_speedup") or 0.0)
            except (TypeError, ValueError):
                micro = 0.0
            if micro > best_micro:
                best_micro = micro
                best_kid = kid
        return best_kid

    def pending_keep_kernel_ids(self) -> list[str]:
        """All KEEP kernel_ids awaiting integrate, sorted strongest-first.

        Used by the Orchestration prompt so the LLM sees how many
        integrate cycles are queued and does NOT propose ``report``
        before draining them.
        """
        integrated_ids = self._kernel_ids_in_optimization_stack()
        integrated_sources = self._source_files_in_optimization_stack()
        rejected = set(self.rejected_kernel_ids or [])
        # Track which source_files we've already counted so the queue
        # mirrors ``next_pending_keep_kernel_id``'s same-file conflict
        # guard (only the strongest KEEP per source_file is queueable).
        claimed_sources: set[str] = set()
        ranked: list[tuple[float, str, str]] = []
        for kid, entry in (self.kernel_opt_attempts or {}).items():
            if not isinstance(entry, dict):
                continue
            # PART C: include DEFER_TO_E2E alongside KEEP (both route to
            # integrate), bounded by the per-kernel E2E budget.
            last_decision = str(entry.get("last_decision", "")).upper()
            if last_decision not in ("KEEP", "DEFER_TO_E2E"):
                continue
            if last_decision == "DEFER_TO_E2E" and self._defer_e2e_budget_exhausted(kid):
                continue
            if kid in integrated_ids or kid in rejected:
                continue
            src = str(entry.get("last_source_file") or "")
            if src and src in integrated_sources:
                continue
            try:
                micro = float(entry.get("last_micro_speedup") or 0.0)
            except (TypeError, ValueError):
                micro = 0.0
            ranked.append((micro, kid, src))
        ranked.sort(key=lambda x: x[0], reverse=True)
        result: list[str] = []
        for _micro, kid, src in ranked:
            if src and src in claimed_sources:
                continue
            if src:
                claimed_sources.add(src)
            result.append(kid)
        return result

    @property
    def has_keep_pending_integrate(self) -> bool:
        return bool(self.next_pending_keep_kernel_id())

    @property
    def kernel_opt_attempts_count(self) -> int:
        return len(self.kernel_opt_attempts or {})

    # ------------------------------------------------------------------
    # PR-C: hot-kernel report gate. Drives Coordinator's "report cannot
    # fire until every meaningful hot reusable kernel has been
    # attempted" guard, so the LLM cannot prematurely emit `report`
    # (Qwen3-30B-A3B-Base 164910Z bug: tick=8 -> report_emitted with
    # k001=24% / k002=37% / k004=9.7% untouched).
    # ------------------------------------------------------------------
    def untried_hot_reusable_kernels(
        self,
        *,
        min_gpu_pct: float | None = None,
        top_n: int | None = None,
    ) -> list[str]:
        """Hot kernels still owing at least one ``kernel_opt`` attempt.

        A kernel qualifies when it satisfies ALL of:
          * ``reusable_native_kernel is True`` in ``last_trace_analyze``
          * ``gpu_pct >= min_gpu_pct`` (default 3.0; override via
            ``HYPERLOOM_KERNEL_OPT_MIN_GPU_PCT``)
          * no member of its ``task_group`` is already on
            ``optimization_stack`` (i.e. nothing from this AST function
            has been integrated)
          * no member of its ``task_group`` appears in
            ``rejected_kernel_ids`` (the whole group is "live")
          * no member of its ``task_group`` has any
            ``kernel_opt_attempts`` entry (zero attempts so far)
          * its ``source_file`` has not been touched by any prior
            ``integrate`` on the stack (same-file conflict guard)

        The candidate set is capped to ``top_n`` (default 5; override
        via ``HYPERLOOM_KERNEL_OPT_GATE_TOP_N``) by ``gpu_pct`` desc so
        the gate cannot demand 15 tiny rmsnorms.

        Each task_group contributes at most one kernel_id (the highest
        ``gpu_pct`` member that satisfies the filters), matching what
        ``_batch_kernel_candidates`` would actually dispatch.
        """
        info = self.last_trace_analyze or {}
        hot = info.get("hot_kernels_top15") or info.get("hot_kernels") or []
        task_groups = info.get("task_groups") or []
        if not isinstance(hot, list):
            return []

        if min_gpu_pct is None:
            try:
                min_gpu_pct = float(os.environ.get(
                    "HYPERLOOM_KERNEL_OPT_MIN_GPU_PCT",
                    _DEFAULT_HOT_KERNEL_MIN_GPU_PCT,
                ))
            except (TypeError, ValueError):
                min_gpu_pct = _DEFAULT_HOT_KERNEL_MIN_GPU_PCT
        if top_n is None:
            try:
                top_n = int(os.environ.get(
                    "HYPERLOOM_KERNEL_OPT_GATE_TOP_N",
                    _DEFAULT_HOT_KERNEL_GATE_TOP_N,
                ))
            except (TypeError, ValueError):
                top_n = _DEFAULT_HOT_KERNEL_GATE_TOP_N
        top_n = max(1, int(top_n))

        kid_to_group: dict[str, list[str]] = {}
        for g in task_groups:
            if not isinstance(g, dict):
                continue
            members = [str(m) for m in (g.get("kernel_ids") or []) if m]
            for m in members:
                kid_to_group[m] = members

        integrated_ids = self._kernel_ids_in_optimization_stack()
        integrated_sources = self._source_files_in_optimization_stack()
        rejected = set(self.rejected_kernel_ids or [])
        attempts = self.kernel_opt_attempts or {}

        # Collect every reusable hot row above threshold, then sort by
        # gpu_pct desc so dedup picks the STRONGEST member of each
        # task_group (otherwise prompt order would arbitrarily make the
        # 24% kernel represent a group whose 37% sibling exists -- bad
        # signal for the LLM).
        rows: list[tuple[float, str, str, list[str]]] = []
        for k in hot:
            if not isinstance(k, dict):
                continue
            if k.get("reusable_native_kernel") is not True:
                continue
            try:
                gpu_pct = float(k.get("gpu_pct") or 0.0)
            except (TypeError, ValueError):
                gpu_pct = 0.0
            if gpu_pct < min_gpu_pct:
                continue
            kid = str(k.get("kernel_id") or "")
            if not kid:
                continue
            src = str(k.get("source_file") or "")
            members = sorted(kid_to_group.get(kid, [kid]))
            rows.append((gpu_pct, kid, src, members))
        rows.sort(key=lambda x: x[0], reverse=True)

        ranked: list[tuple[float, str, str, list[str]]] = []
        seen_groups: set[tuple[str, ...]] = set()
        for row in rows:
            group_key = tuple(row[3])
            if group_key in seen_groups:
                continue
            seen_groups.add(group_key)
            ranked.append(row)
        ranked = ranked[:top_n]

        untried: list[str] = []
        for _pct, kid, src, members in ranked:
            # Whole task_group rejected -> skip
            if members and all(m in rejected for m in members):
                continue
            # Any member already integrated -> skip
            if any(m in integrated_ids for m in members):
                continue
            # Source already covered by a stack integrate -> skip
            if src and src in integrated_sources:
                continue
            # Any member has been tried -> skip
            if any(int((attempts.get(m) or {}).get("attempts", 0)) > 0
                   for m in members):
                continue
            untried.append(kid)
        return untried

    # ------------------------------------------------------------------
    # Per-action audit (kernel parity for non-kernel actions)
    # ------------------------------------------------------------------
    @staticmethod
    def _truncate_excerpt(value: Any, *, limit: int = 800) -> str | None:
        """Coerce ``value`` to str and trim to ``limit`` characters.

        Returns None for falsy inputs so the prompt renderer can show
        ``err=(none)`` instead of a quoted empty string.
        """
        if value is None:
            return None
        text = str(value)
        if not text:
            return None
        if len(text) <= limit:
            return text
        return text[:limit]

    @staticmethod
    def _stderr_tail(value: Any, *, limit: int = 1000) -> str | None:
        """Pull the last ``limit`` characters from a subprocess error blob.

        Distinct helper from :meth:`_truncate_excerpt` because subprocess
        stderr usually has the actionable signal at the *end* (traceback,
        last log line), while a free-form error message is informative
        from the start.
        """
        if value is None:
            return None
        text = str(value)
        if not text:
            return None
        return text[-limit:] if len(text) > limit else text

    def record_action_attempt(
        self,
        action: str,
        *,
        task_id: str,
        status: str,
        decision: str,
        result: dict[str, Any] | None,
        extras: dict[str, Any] | None = None,
        max_history: int = _DEFAULT_ATTEMPTS_HISTORY,
    ) -> dict[str, Any] | None:
        """Append one attempt to ``<action>_attempts`` and refresh ``last_<action>``.

        Uniform entry schema (see audit-trail plan):

            {ts, task_id, status, decision, key_metric, key_metric_kind,
             workspace, error_class, error_excerpt, raw_result_path,
             reported_success, extras}

        ``status`` is ``"succeeded"`` or ``"failed"``; ``decision`` is the
        Coordinator's interpretation of what it did with the result
        (``"promoted"`` / ``"discarded"`` / ``"salvaged"`` /
        ``"no_promote"`` / ``"error"``). ``extras`` is appended verbatim
        for action-specific context (round_id, trace_path, etc.).

        Returns the new entry dict (so callers can attach it to the event
        log if useful), or ``None`` when ``action`` is not in the audit
        set — kernel-owned actions go through their own bespoke recorders
        and intentionally skip this surface.

        Pure persistence helper: does NOT call :meth:`save`. The
        Coordinator batches a single ``save()`` per dispatcher pass.
        """
        if action not in _AUDIT_ACTIONS:
            return None
        attempts_attr = f"{action}_attempts"
        last_attr = f"last_{action}"
        if not hasattr(self, attempts_attr) or not hasattr(self, last_attr):
            return None
        result = result or {}
        metric_key, metric_kind = _KEY_METRIC_MAP.get(
            action, ("output_throughput", "output_throughput"),
        )
        raw_metric = result.get(metric_key)
        try:
            key_metric: float | None = (
                float(raw_metric) if isinstance(raw_metric, (int, float))
                else None
            )
        except (TypeError, ValueError):
            key_metric = None
        entry: dict[str, Any] = {
            "ts": _now_iso(),
            "task_id": str(task_id or ""),
            "status": str(status or ""),
            "decision": str(decision or ""),
            "key_metric": key_metric,
            "key_metric_kind": metric_kind,
            "workspace": (
                str(result.get("workspace"))
                if result.get("workspace") else None
            ),
            "error_class": (
                str(result.get("error_class"))
                if result.get("error_class") else None
            ),
            "error_excerpt": self._truncate_excerpt(result.get("error")),
            "raw_result_path": (
                str(result.get("raw_result_path"))
                if result.get("raw_result_path") else None
            ),
            "reported_success": result.get("reported_success"),
            "extras": dict(extras or {}),
        }
        history: list[dict[str, Any]] = list(getattr(self, attempts_attr) or [])
        history.append(entry)
        if len(history) > max_history:
            history = history[-max_history:]
        setattr(self, attempts_attr, history)
        setattr(self, last_attr, dict(entry))
        return entry

    def record_action_failure(
        self,
        *,
        action: str,
        task_id: str,
        result: dict[str, Any] | None,
        max_history: int = _DEFAULT_LAST_FAILURES,
    ) -> dict[str, Any]:
        """Append one rich failure record to :attr:`last_action_failures`.

        Carries the failure context Orchestration needs to self-correct
        even after the inbox has rotated past the matching
        ``delegated_result`` event:

        * ``error_class``        — short tag from the executor (e.g.
          ``"no_report"`` / ``"invalid_measurement"`` /
          ``"subprocess_nonzero"`` / ``"timeout"``).
        * ``error_excerpt``      — first 800 chars of ``result['error']``.
        * ``stderr_tail``        — last 1000 chars of ``result['error']``
          when ``error_class`` looks like a subprocess failure
          (``subprocess_nonzero`` / ``timeout``).
        * ``workspace``          — per-task workspace path the executor
          materialized (the place the operator would dig into next).
        * ``raw_result_path``    — set by
          :func:`extract_benchmark_measurement` when it salvaged a raw
          inferencex_result.json (so Orchestration can see *where* the
          rescue came from).
        * ``reported_success``   — what the wrapper claimed
          (``benchmark_report.json:success``).

        Unlike :meth:`record_action_attempt` this is invoked for **every**
        unpromotable task kind — kernel-owned actions land here too so
        the global failure tail is comprehensive.
        """
        result = result or {}
        error_class = result.get("error_class")
        error_class_str = str(error_class) if error_class else None
        entry: dict[str, Any] = {
            "ts": _now_iso(),
            "action": str(action or ""),
            "task_id": str(task_id or ""),
            "error_class": error_class_str,
            "error_excerpt": self._truncate_excerpt(result.get("error")),
            "stderr_tail": (
                self._stderr_tail(result.get("error"))
                if error_class_str in {"subprocess_nonzero", "timeout"}
                else None
            ),
            "workspace": (
                str(result.get("workspace"))
                if result.get("workspace") else None
            ),
            "raw_result_path": (
                str(result.get("raw_result_path"))
                if result.get("raw_result_path") else None
            ),
            "reported_success": result.get("reported_success"),
        }
        history = list(self.last_action_failures or [])
        history.append(entry)
        if len(history) > max_history:
            history = history[-max_history:]
        self.last_action_failures = history
        return entry

    def record_trace_analyze(self, payload: dict[str, Any],
                              result: dict[str, Any]) -> None:
        """Cache the latest trace_analyze output keyed by trace_input.

        We persist a wider window than the prompt-visible top5 so that when
        the very top GPU consumers are vendor-binary kernels (Tensile / CK)
        Orchestration still sees lower-ranked but **reusable native** entries
        (e.g. AITER RMSNorm) and can dispatch ``run_optimization`` against
        them instead of looping on rejected ones.

        We also persist ``trace_health_warnings`` so Orchestration's prompt
        surfaces the structured routing signals produced by the TraceLens
        analyzer (Hyperloom v0.4 finishing-touches T3 / T4):

        * ``high_gpu_idle_pct`` — Executive Summary's ``Idle %`` exceeded
          the gate threshold; per Report_Interfacing.docx §2 (idle-gate
          sanity check) the LLM should pivot to parameter optimization
          rather than kernel rewriting in this regime.
        * ``tracelens_analysis_failed`` — the TraceLens subprocess crashed
          permanently (perf-CLI missing, ``analysis.md`` not produced,
          timeout, …); Coordinator already demoted this to ``status=ok``
          + empty ``hot_kernels`` at the handler boundary, but the LLM
          still needs to *see* the failure so it picks parameter
          optimization explicitly instead of inferring "TraceLens is
          still running" from the empty list.

        Without this pass-through the warnings produced upstream are
        dropped at the SharedState boundary and Orchestration cannot
        ground its routing decisions on them.
        """
        if not isinstance(result, dict):
            return
        trace_input = (
            (payload or {}).get("trace_input")
            or (payload or {}).get("trace_dir")
            or ""
        )
        candidates_path = result.get("candidates_path") or ""
        if not candidates_path:
            artifacts = result.get("artifact_paths") or {}
            if isinstance(artifacts, dict):
                candidates_path = artifacts.get("kernel_candidates", "") or ""
        hot = result.get("hot_kernels") or []
        summary: list[dict[str, Any]] = []
        reusable_ids: list[str] = []
        for entry in hot[:15] if isinstance(hot, list) else []:
            if not isinstance(entry, dict):
                continue
            kid = entry.get("kernel_id")
            reusable = bool(entry.get("reusable_native_kernel"))
            summary.append({
                "kernel_id": kid,
                "name": entry.get("name"),
                "gpu_pct": entry.get("gpu_pct"),
                "bottleneck": entry.get("bottleneck"),
                "arithmetic_intensity": entry.get("arithmetic_intensity"),
                "source_file": entry.get("source_file"),
                "reusable_native_kernel": reusable,
                "recommended_backends": entry.get("recommended_backends") or [],
                "recommended_actions": entry.get("recommended_actions") or [],
            })
            if reusable and kid:
                reusable_ids.append(str(kid))

        # T3 / T4: keep the structured warning list verbatim — handler
        # already shaped each entry into the documented form (code,
        # severity, message, plus code-specific extras like idle_pct or
        # returncode). We filter to ``dict`` to be defensive against a
        # buggy tool emitting non-dict junk, but otherwise pass through
        # untouched so the LLM sees the full diagnostic.
        raw_warnings = result.get("trace_health_warnings") or []
        warnings_cleaned: list[dict[str, Any]] = []
        if isinstance(raw_warnings, list):
            for entry in raw_warnings:
                if isinstance(entry, dict) and entry.get("code"):
                    warnings_cleaned.append(dict(entry))

        # Roofline-v2 C1: surface the TraceLens analysis.md full text on
        # SharedState so the downstream ``roofline`` action (added in C4)
        # can read the per-category bottleneck breakdown without re-hitting
        # the filesystem on every tick, and so Orchestration can ground
        # PRUNE_BRANCH / propose_action decisions on the actual report.
        #
        # Snapshot accounting (snapshot_id + baseline_gain_at_snapshot)
        # lets the prompt show "report taken at gain=X%" and lets the
        # re-profile guidance trigger when ``cumulative_gain_validated``
        # has moved by ≥3% since the snapshot was taken. The counter
        # lives inside ``last_trace_analyze`` (not as a new top-level
        # field) so we do not widen the SharedState surface — every
        # call reads the previous snapshot_id and bumps it by one.
        prev_snapshot_id = 0
        if isinstance(self.last_trace_analyze, dict):
            prev_raw = self.last_trace_analyze.get("roofline_snapshot_id")
            if isinstance(prev_raw, int):
                prev_snapshot_id = prev_raw
        snapshot_id = prev_snapshot_id + 1

        analysis_md_path = result.get("trace_report_path") or ""
        analysis_md_text = ""
        if analysis_md_path:
            try:
                # Decision A3: no truncation — typical analysis.md is
                # 10-20 KB, worst-case (long-ISL Case A-D) ~200 KB; even
                # 100+ ticks against a 200 KB cached report stays well
                # within the 200K-token Orchestration context budget once
                # ``_format_roofline_decision`` (C5) replaces verbatim
                # injection with the structured roofline-action output.
                analysis_md_text = Path(analysis_md_path).read_text(
                    encoding="utf-8", errors="replace",
                )
            except (OSError, ValueError):
                # Degrade silently — empty text signals "no report
                # available" to downstream prompt rendering, and the
                # ``analysis_md_path`` field still lets a future
                # read_artifact intent fetch it on demand.
                analysis_md_text = ""

        # PR-G: propagate ``task_groups`` from the handler result so the
        # multi-KEEP integrate queue's source-of-truth lookups stay in
        # sync with what ``_batch_kernel_candidates`` dispatched.
        # Without this, ``untried_hot_reusable_kernels()`` and
        # ``next_pending_keep_kernel_id()`` see ``task_groups=[]`` and
        # fall through to per-kernel logic -- e.g. for
        # ``primary=k004 kids=[k003,k004]`` they treat k003 as an
        # untried independent kernel even though dispatching k004
        # already covered the same AST function. The LLM then proposes
        # a second ``run_optimization`` batch for k001/k003/..., wastes
        # GEAK->Claude->Codex wall-clock on kernels that share patches
        # with what the prior batch already produced
        # (Qwen3-30B-A3B-Base session 20260523T035235Z saw this with
        # k001/k003/k005 spinning up after k002/k004/k009 retired).
        task_groups = result.get("task_groups") or []
        if not isinstance(task_groups, list):
            task_groups = []
        self.last_trace_analyze = {
            "trace_input": str(trace_input),
            "candidates_path": str(candidates_path),
            "hot_kernels_top15": summary,
            "task_groups": task_groups,
            "reusable_native_kernel_ids": reusable_ids,
            "trace_health_warnings": warnings_cleaned,
            "analysis_md_path": str(analysis_md_path),
            "analysis_md_text": analysis_md_text,
            "roofline_snapshot_id": snapshot_id,
            "roofline_baseline_gain_at_snapshot": float(
                self.cumulative_gain_validated,
            ),
            "ts": _now_iso(),
        }

    def record_sweep(self, result: dict[str, Any]) -> None:
        if not isinstance(result, dict):
            return
        grid = result.get("sweep_grid") or []
        best = None
        if isinstance(grid, list):
            best = max(
                (
                    e for e in grid
                    if isinstance(e, dict)
                    and e.get("status") == "succeeded"
                    and isinstance(e.get("output_throughput"), (int, float))
                ),
                default=None,
                key=lambda e: e.get("output_throughput") or 0.0,
            )
        self.last_sweep = {
            "ts": _now_iso(),
            "grid_size": result.get("grid_size", len(grid) if isinstance(grid, list) else 0),
            "best_overall": best or {},
            "best_for_each_conc": result.get("best_for_each_conc") or {},
            "pareto_front": result.get("pareto_front") or [],
            "workspace": result.get("workspace", ""),
        }

    def push_params_winner(
        self,
        *,
        action: str,
        variant_name: str,
        tput: float,
        gain_pct: float,
        extra_sglang_args: str | None = None,
        extra_envs: dict[str, Any] | None = None,
        max_history: int = 10,
    ) -> None:
        """Append one round's winner to the rolling history buffer.

        ``extra_sglang_args`` + ``extra_envs`` (when provided) are folded
        into the row as ``fingerprint`` so the cross-round
        :meth:`consistent_winner` detector and the IR-26 idea generator
        see content identity, not just the display name. Old callers
        passing only ``variant_name`` still work (fingerprint = empty).
        """
        from .action_executors._grid_runner import variant_fingerprint
        fp = (
            variant_fingerprint(extra_sglang_args, extra_envs)
            if (extra_sglang_args is not None or extra_envs is not None)
            else ""
        )
        self.params_winner_history.append({
            "action": action,
            "variant_name": variant_name,
            "tput": float(tput) if tput is not None else 0.0,
            "gain_pct": float(gain_pct) if gain_pct is not None else 0.0,
            "fingerprint": fp,
            "ts": _now_iso(),
        })
        if len(self.params_winner_history) > max_history:
            self.params_winner_history = self.params_winner_history[-max_history:]

    def consistent_winner(self, *, lookback: int = 3,
                          min_appearances: int = 2,
                          min_avg_gain_pct: float = 0.3) -> dict[str, Any] | None:
        """Detect a variant_name that consistently wins across recent rounds.

        Returns the winning variant's most-recent record (so callers can
        promote it) or ``None`` if no variant qualifies.
        """
        if len(self.params_winner_history) < min_appearances:
            return None
        recent = self.params_winner_history[-lookback:]
        from collections import Counter
        counts = Counter(w["variant_name"] for w in recent if w.get("variant_name"))
        for name, n in counts.most_common():
            if n < min_appearances:
                continue
            picks = [w for w in recent if w.get("variant_name") == name]
            avg_gain = sum(w["gain_pct"] for w in picks) / len(picks)
            if avg_gain >= min_avg_gain_pct:
                # Return the most-recent record for this winner so caller
                # can lift its tput / extra_sglang_args into current_best.
                return picks[-1]
        return None

    def apply_params_search_update(self, update: dict[str, Any]) -> None:
        """Merge a ParamsExecutor search update into persistent state."""
        if not isinstance(update, dict):
            return
        self.params_search = dict(update)

    def apply_backends_search_update(self, update: dict[str, Any]) -> None:
        """Merge a BackendsExecutor search update into persistent state.

        Mirror of :meth:`apply_params_search_update`. Coordinator calls
        this once per backends round from
        :meth:`Coordinator._promote_to_shared_state`. ``accepted`` writes
        are NOT performed here — the executor only reports
        ``tested`` / ``rejected`` / ``last_round`` increments;
        :meth:`record_backends_accepted` is the single writer for
        ``accepted`` (called by Coordinator on promote).
        """
        if not isinstance(update, dict):
            return
        # Preserve any ``accepted`` we already promoted: the executor's
        # update only touches tested/rejected/last_round; overwriting
        # ``accepted`` from a fresh round would lose history.
        prior_accepted = list(
            (self.backends_search or {}).get("accepted") or []
        )
        merged = dict(update)
        if "accepted" not in update or not update.get("accepted"):
            merged["accepted"] = prior_accepted
        self.backends_search = merged

    def record_backends_accepted(self, variant: dict[str, Any]) -> None:
        """Append one promoted variant to ``backends_search.accepted``.

        Called by Coordinator after a backends winner is lifted to
        ``current_best``. Dedupes by ``fingerprint`` (computed on the fly
        if absent) so repeated promotes of the same content don't bloat
        the list. Also removes a matching entry from ``rejected`` so a
        previously-rejected variant that later won doesn't appear in
        both buckets.
        """
        if not isinstance(variant, dict) or not variant:
            return
        from .action_executors._grid_runner import variant_fingerprint
        args = str(
            variant.get("candidate_extra_sglang_args")
            or variant.get("extra_sglang_args") or ""
        )
        envs = dict(variant.get("extra_envs") or {})
        fp = str(variant.get("fingerprint") or variant_fingerprint(args, envs))
        entry = {
            "name": str(variant.get("name") or ""),
            "extra_sglang_args": args,
            "extra_envs": envs,
            "note": str(variant.get("note") or ""),
            "fingerprint": fp,
            "tput": variant.get("output_throughput") or variant.get("tput"),
            "gain_pct": variant.get("gain_pct"),
        }
        search = dict(self.backends_search or {})
        search.setdefault("schema_version", 1)
        accepted = list(search.get("accepted") or [])
        accepted = [
            v for v in accepted
            if not (isinstance(v, dict) and v.get("fingerprint") == fp)
        ]
        accepted.append(entry)
        search["accepted"] = accepted
        rejected = [
            v for v in (search.get("rejected") or [])
            if not (isinstance(v, dict) and v.get("fingerprint") == fp)
        ]
        search["rejected"] = rejected
        name_index = dict(search.get("name_index") or {})
        if entry["name"]:
            name_index[entry["name"]] = fp
        search["name_index"] = name_index
        self.backends_search = search

    # ------------------------------------------------------------------
    # T1/T2 — search-space expansion bookkeeping
    # ------------------------------------------------------------------
    def record_discovered_flags(
        self,
        *,
        framework: str,
        backend_flags: list[str] | None = None,
        param_flags: list[str] | None = None,
        source_path: str = "",
    ) -> None:
        """Persist the AST-discovered flag list for a framework.

        Called by BackendsExecutor / ParamsExecutor when they first run
        ``discover_*_flags()`` on a fresh session. The Orchestration prompt
        surfaces the union so the LLM can synthesize new GridVariant
        candidates that the shipped DEFAULT_*_GRID may not cover.

        Idempotent: re-recording overwrites the per-framework entry but
        leaves other frameworks untouched.
        """
        fw = (framework or "").strip().lower() or "unknown"
        entry = dict(self.discovered_flags.get(fw) or {})
        if backend_flags is not None:
            entry["backend_flags"] = sorted(set(str(f) for f in backend_flags))
        if param_flags is not None:
            entry["param_flags"] = sorted(set(str(f) for f in param_flags))
        if source_path:
            entry["source_path"] = str(source_path)
        entry["ts"] = _now_iso()
        self.discovered_flags[fw] = entry

    def push_backend_winners_round(
        self,
        *,
        action: str,
        base_tput: float,
        base_extra_args: str,
        winners: list[dict[str, Any]],
        best: dict[str, Any] | None,
        max_history: int = 20,
    ) -> None:
        """Append one explore round's winners (≥+1% over base) to history.

        IR-26 (dynamic idea generation) reads this so the LLM, before
        proposing the next backends/params round, can compose new combos /
        retries / sibling-flag variants from what previously won. Marathon
        equivalent: orchestrator pane's per-tick "follow-on actions"
        synthesis (marathon/skills/SKILL.md §"Dynamic Idea Generation").
        """
        from .action_executors._grid_runner import variant_fingerprint
        round_id = f"{action}-{len(self.backend_winners_history) + 1:03d}"

        def _stamped(entry: dict[str, Any]) -> dict[str, Any]:
            args = str(
                entry.get("candidate_extra_sglang_args")
                or entry.get("extra_sglang_args") or ""
            )
            envs = dict(entry.get("extra_envs") or {})
            return {
                "name": str(entry.get("name", "")),
                "tput": entry.get("output_throughput") or entry.get("tput"),
                "gain_pct": entry.get("gain_pct"),
                "extra_sglang_args": args,
                "extra_envs": envs,
                "note": str(entry.get("note") or ""),
                "fingerprint": (
                    str(entry.get("fingerprint"))
                    if entry.get("fingerprint")
                    else variant_fingerprint(args, envs)
                ),
            }

        entry = {
            "action": str(action),
            "round_id": round_id,
            "base_tput": float(base_tput) if base_tput is not None else 0.0,
            "base_extra_args": str(base_extra_args or ""),
            "winners": [
                _stamped(w) for w in (winners or []) if isinstance(w, dict)
            ],
            "best": (
                {
                    **_stamped(best),
                }
                if isinstance(best, dict) else None
            ),
            "ts": _now_iso(),
        }
        self.backend_winners_history.append(entry)
        if len(self.backend_winners_history) > max_history:
            self.backend_winners_history = (
                self.backend_winners_history[-max_history:]
            )

    def mark_synergy_attempted(self, combo_names: list[str]) -> None:
        """Record one synergy combo as already tested.

        ``combo_names`` is a list of GridVariant.name members ordered by
        the synergy group; the canonical key is ``"+".join(sorted(names))``
        so the same set isn't double-counted regardless of input order.
        """
        if not combo_names:
            return
        key = "+".join(sorted(str(n) for n in combo_names if n))
        if not key:
            return
        if key in self.synergy_attempted:
            return
        self.synergy_attempted.append(key)
        if len(self.synergy_attempted) > 100:
            self.synergy_attempted = self.synergy_attempted[-100:]

    def is_synergy_attempted(self, combo_names: list[str]) -> bool:
        if not combo_names:
            return False
        key = "+".join(sorted(str(n) for n in combo_names if n))
        return bool(key) and key in self.synergy_attempted

    # ------------------------------------------------------------------
    # Action scoring (see orchestrator/scoring.py)
    # ------------------------------------------------------------------
    def get_action_score(self, name: str) -> ActionScore | None:
        raw = self.action_scores.get(name)
        if not isinstance(raw, dict):
            return None
        return ActionScore.from_dict(raw)

    def put_action_score(self, name: str, score: ActionScore) -> None:
        """Persist an ``ActionScore`` instance back into the raw dict map."""
        if not name:
            return
        self.action_scores[name] = score.to_dict()

    def all_action_scores(self) -> dict[str, ActionScore]:
        out: dict[str, ActionScore] = {}
        for name, raw in self.action_scores.items():
            if isinstance(raw, dict):
                out[name] = ActionScore.from_dict(raw)
        return out

    def increment_tick(self) -> int:
        """Bump the Coordinator tick counter and return the new value."""
        self.tick = int(self.tick or 0) + 1
        return self.tick

    def to_action_scores_summary(
        self,
        *,
        registry: Any,
        top_k: int = 12,
    ) -> str:
        """Render the per-tick `Action scores` block consumed by the
        Orchestration prompt.

        The block is a header + one row per action (sorted by eff_score desc),
        followed by a single ``locked: ...`` summary row listing any
        cooldown / locked rows present in the registry but pushed below
        positive scores. The renderer is deliberately compact — the LLM only
        needs name + eff_score + a few diagnostics to pick a next action.

        ``registry`` is an ``ActionRegistry`` (kept untyped here to avoid a
        circular import: shared_state already imports scoring which itself
        imports ActionRegistry / ActionMetadata).
        """
        if not self.action_scores:
            return (
                f"=== Action scores (top 0 by eff_score, tick={self.tick}) ===\n"
                "(no scores seeded)"
            )
        target_mult = _target_gap_multiplier(
            target_gap_pct=float(self.target_gap_pct or 0.0),
            cumulative_gain=float(self.cumulative_gain or 0.0),
        )
        rows = _rank_top_k(
            self.action_scores,
            registry,
            tick=int(self.tick or 0),
            target_gap_mult=target_mult,
            k=int(top_k),
            shared_state=self,  # N30: enable cheap-exhausted deep boost
        )
        # B' (May 2026, follow-up to N30): per-row actionability tag
        # so the LLM can tell at a glance whether the top eff_score
        # row is something it can ``propose_action`` (cheap / analysis
        # actions) or needs to emit as a ``request`` (kernel-owned
        # deep actions like deep_kernel_analysis / operator_tuning /
        # kernel_opt / integrate / vendor_kernel_config). Pre-B' the
        # ranking just listed names, so when N30's boost surfaced two
        # kernel-owned rows above kernel_opt (rank 1 + 2 unreachable
        # via propose), the LLM tended to skip all the way down to a
        # familiar shallow row like params (rank 6) instead of
        # picking kernel_opt (rank 3, the top REQUEST-actionable
        # deep choice).
        from .policy import KERNEL_OWNED_ACTIONS as _KERNEL_OWNED
        # Kernel-owned actions get an explicit "REQUEST kind=..." tag
        # so the LLM sees the propose path next to the score.
        _REQUEST_KIND_BY_ACTION = {
            "kernel_opt": "run_optimization",
            "integrate": "integrate",
            "deep_kernel_analysis": "deep_kernel_analysis",
            "operator_tuning": "operator_tuning",
            "vendor_kernel_config": "vendor_kernel_config",
        }
        lines: list[str] = [
            f"=== Action scores (top {len(rows)} by eff_score, tick={self.tick}) ==="
        ]
        # B': surface the cheap-exhausted state as a header note so
        # the LLM understands why the deep rows just jumped 6+ points
        # in eff_score this tick. Pure informational -- no MUST/SHALL
        # prescriptions; the scoreboard already encodes the priority.
        try:
            from .scoring import _is_cheap_exhausted as _scoring_cheap_exhausted
            if _scoring_cheap_exhausted(self):
                lines.append(
                    "  (cheap_exploration_exhausted=True; N30 boost "
                    "applied to deep-family rows so deep actions reflect "
                    "their post-cheap priority)"
                )
        except ImportError:
            pass
        locked_rows: list[tuple[str, str]] = []
        # B': remember the highest-eff_score actionable row so we can
        # mark it with a trailing "← top actionable" hint. We compute
        # this in a first pass so the hint is correct even when the
        # raw top rows are kernel-owned (LLM must REQUEST those, not
        # propose them).
        def _is_actionable(name: str, locked: str | None, cd: int) -> bool:
            if locked or cd > 0:
                return False
            # All actions are actionable from LLM's point of view --
            # kernel-owned via REQUEST, others via propose_action.
            # "Top actionable" means top by eff_score among unlocked,
            # uncooldown'd rows.
            return True
        top_actionable_name: str | None = None
        top_actionable_eff = float("-inf")
        for name, eff, a in rows:
            cd_remaining = max(0, int(a.cooldown_until_tick) - int(self.tick or 0))
            if eff < 0:
                continue  # _LOCKED_SCORE
            if not _is_actionable(name, a.locked_reason or None, cd_remaining):
                continue
            if eff > top_actionable_eff:
                top_actionable_eff = eff
                top_actionable_name = name
        for name, eff, a in rows:
            cd_remaining = max(0, int(a.cooldown_until_tick) - int(self.tick or 0))
            age = (
                (int(self.tick or 0) - int(a.last_run_tick))
                if int(a.last_run_tick) >= 0
                else int(self.tick or 0) + 1
            )
            tag = ""
            if a.locked_reason:
                tag = f"   [locked: {a.locked_reason}]"
                locked_rows.append((name, a.locked_reason))
            elif cd_remaining > 0:
                tag = f"   [cooldown {cd_remaining}]"
            else:
                # B' actionability tag (only on unlocked / unconcooldown'd
                # rows; locked / cooldown'd take precedence above).
                if name in _KERNEL_OWNED:
                    kind = _REQUEST_KIND_BY_ACTION.get(name, name)
                    tag = f"   [REQUEST: kernel-owned, kind={kind}]"
                else:
                    tag = "   [propose_action]"
            # B': append "← top actionable" to the highest-scoring
            # unlocked row so LLM doesn't have to scan two columns to
            # find what it should propose this tick.
            if name == top_actionable_name and eff > 0:
                tag = tag + "  ← top actionable"
            eff_display = "  N/A" if eff < 0 else f"{eff:.2f}"
            lines.append(
                f"  eff={eff_display:>5} base={a.base_score:.2f} "
                f"mult={a.score_mult:.2f} "
                f"runs={a.runs} keeps={a.keeps} disc={a.discards} "
                f"cd={cd_remaining} age={age}   {name}{tag}"
            )
        if locked_rows:
            lines.append(
                "locked: "
                + ", ".join(f"{n}({r})" for n, r in sorted(locked_rows))
            )
        return "\n".join(lines)

    def append_stack_gain_entry(
        self,
        *,
        action: str,
        variant_name: str | None,
        new_tput: float,
        extra_sglang_args: str = "",
        ts: str | None = None,
    ) -> float | None:
        """N32b: Mirror an ``optimization_stack`` append into
        ``gain_per_stack_entry`` so the two lists stay index-aligned.

        Computes ``(new_tput - baseline_tput) / baseline_tput * 100``
        and appends. Returns the computed gain_pct (None when
        baseline_tput is 0 or new_tput is non-positive). Callers don't
        usually consume the return value -- the list is the persisted
        side-effect.

        This method was a missing piece between main's
        ``_lift_to_current_best`` (which calls it) and SharedState
        (which previously only carried the ``gain_per_stack_entry``
        field without an explicit mutator). Without this method, every
        params/backends/integrate KEEP that promotes a winner crashes
        with ``AttributeError``. Filling it in is a no-design-change
        fix: the math + list append were already implied by the
        adjacent comments on the call sites.
        """
        try:
            base = float(self.baseline_tput or 0.0)
        except (TypeError, ValueError):
            base = 0.0
        try:
            tput = float(new_tput or 0.0)
        except (TypeError, ValueError):
            tput = 0.0
        gain_pct: float | None
        if base > 0 and tput > 0:
            gain_pct = (tput - base) / base * 100.0
        else:
            gain_pct = None
        self.gain_per_stack_entry.append(gain_pct)
        return gain_pct

    def seed_stack_from_current_best(self) -> None:
        """Backfill stack for old sessions that only had current_best."""
        if self.optimization_stack or not isinstance(self.current_best, dict):
            return
        variant = self.current_best.get("variant_name")
        extra_args = self.current_best.get("extra_sglang_args")
        if not variant and not extra_args:
            return
        self.optimization_stack = [{
            "action": self.current_best.get("action", "unknown"),
            "variant_name": variant or "legacy_current_best",
            "extra_sglang_args": extra_args or "",
            "extra_envs": dict(self.current_best.get("extra_envs") or {}),
            "tput": self.current_best.get("tput"),
            "workspace": self.current_best.get("workspace"),
            "source": "seeded_from_current_best",
        }]
        # Keep ``gain_per_stack_entry`` aligned with ``optimization_stack``
        # (None == we don't know the per-entry gain for seeded entries).
        if len(self.gain_per_stack_entry) < len(self.optimization_stack):
            self.gain_per_stack_entry.extend(
                [None] * (len(self.optimization_stack) - len(self.gain_per_stack_entry))
            )

    # ------------------------------------------------------------------
    # Time-budget helpers (Phase 2 — consumed by Coordinator._compose_prompt)
    # ------------------------------------------------------------------
    def elapsed_minutes(self, *, now: datetime | None = None) -> float:
        """Wall-clock minutes since ``start_ts``.

        Returns 0.0 when ``start_ts`` is empty / unparseable so callers can
        treat the value as "no time consumed yet" without a try/except.
        """
        if not self.start_ts:
            return 0.0
        try:
            start = datetime.fromisoformat(self.start_ts)
        except ValueError:
            return 0.0
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        now_dt = now or datetime.now(timezone.utc)
        if now_dt.tzinfo is None:
            now_dt = now_dt.replace(tzinfo=timezone.utc)
        delta = (now_dt - start).total_seconds() / 60.0
        return max(0.0, delta)

    def remaining_minutes(self, *, now: datetime | None = None) -> float | None:
        """Minutes left in the wall-clock budget.

        Returns ``None`` when ``max_minutes`` is 0 / unset — i.e. the run
        has no upper bound. Otherwise the result is clamped at 0 so the
        prompt never advertises negative time.
        """
        if not self.max_minutes:
            return None
        return max(0.0, float(self.max_minutes) - self.elapsed_minutes(now=now))

    def optimization_stack_has_unvalidated_keeps(self) -> bool:
        """True iff a new KEEP has landed since the last validate_stack.

        Used by Coordinator to surface the ``validate_stack required`` TODO
        in the per-tick checklist. The check is purely on stack *length*:
        every successful validate_stack records ``cumulative_gain_validated_stack_len``,
        so a longer stack means at least one new KEEP came in.
        """
        return len(self.optimization_stack) > int(self.cumulative_gain_validated_stack_len)

    def to_mission_summary(self, *, now: datetime | None = None) -> str:
        """Mission-progress block printed at the very top of every tick.

        Distinct from :meth:`to_prompt_summary` because we want the LLM to
        see the *outcome-shaped* state (raw gain, validated gain, time
        spent vs budget, validated-stack staleness) before drowning in
        verbose execution detail.
        """
        elapsed = self.elapsed_minutes(now=now)
        remaining = self.remaining_minutes(now=now)
        budget_line = (
            f"time      : elapsed={elapsed:.1f}min "
            f"remaining={remaining:.1f}min "
            f"budget={self.max_minutes}min"
        ) if remaining is not None else (
            f"time      : elapsed={elapsed:.1f}min budget=unlimited"
        )
        validated_age = ""
        if self.cumulative_gain_validated_ts:
            validated_age = f" (ts={self.cumulative_gain_validated_ts})"
        unvalidated = self.optimization_stack_has_unvalidated_keeps()
        unvalidated_tag = (
            " ⚠ stack changed since last validate_stack — RUN validate_stack"
            if unvalidated else ""
        )
        return (
            f"baseline  : {self.baseline_tput} tok/s/GPU\n"
            f"current   : {self._format_current_best_for_mission()}\n"
            f"gain      : per-round-sum={self.cumulative_gain:.2f}% "
            f"validated={self.cumulative_gain_validated:.2f}%{validated_age}\n"
            f"stack     : {len(self.optimization_stack)} entries "
            f"(validated_at_len={self.cumulative_gain_validated_stack_len})"
            f"{unvalidated_tag}\n"
            f"{budget_line}"
        )

    def _format_current_best_for_mission(self) -> str:
        if not isinstance(self.current_best, dict) or not self.current_best:
            return "(none)"
        return (
            f"action={self.current_best.get('action','?')} "
            f"tput={self.current_best.get('tput','?')} "
            f"variant={self.current_best.get('variant_name','?')}"
        )

    def to_prompt_summary(self) -> str:
        """Compact, human-readable snapshot for prompt injection (DESIGN §8.3)."""
        lines = [
            f"session_id={self.session_id or '(unset)'}",
            f"model={self.model_name or '(unset)'}  class={self.model_class or '(unset)'}",
            f"baseline_tput={self.baseline_tput}  baseline_acc={self.baseline_accuracy}",
            f"baseline_failure_streak={self.baseline_failure_streak}",
            f"current_best={self.current_best or '(none)'}",
            f"optimization_stack={self._format_optimization_stack()}",
            f"cumulative_gain={self.cumulative_gain}%",
            (
                f"cumulative_gain_validated={self.cumulative_gain_validated}% "
                f"(stack_len_at_validation={self.cumulative_gain_validated_stack_len}, "
                f"ts={self.cumulative_gain_validated_ts or '(never)'})"
            ),
            f"last_sweep={self._format_last_sweep()}",
            f"current_action={self.current_action or '(idle)'}",
            f"crash_count={self.crash_count}",
            f"pruned_families={self.pruned_families or '(none)'}",
            f"last_profile_trace={self.last_profile_trace or '(none)'}",
            f"last_profile_status={self.last_profile_status or '(none)'}",
            f"last_profile_args='{self.last_profile_args}'",
            f"discovered_flags_error={self.discovered_flags_error or '(none)'}",
            f"last_profile_roofline={self.last_profile_roofline or '(none)'}",
            f"last_profile_kernel_breakdown={self.last_profile_kernel_breakdown or '(none)'}",
            f"last_trace_analyze={self._format_last_trace_analyze()}",
            f"analysis_md={self._format_analysis_md_full()}",
            f"params_no_promote_streak={self.params_no_promote_streak}",
            f"params_search={self._format_params_search()}",
            f"backends_search={self._format_backends_search()}",
            f"discovered_flags={self._format_discovered_flags()}",
            f"backend_winners_history={self._format_backend_winners_history()}",
            f"synergy_attempted={len(self.synergy_attempted)} combos",
            f"last_kernel_opt={self._format_last_kernel_opt()}",
            # PR-B multi-KEEP integrate queue: surfaces the full set of
            # pending KEEPs the integrate gate will drain (strongest-
            # first), plus the per-kernel attempts count Fix-2 reads to
            # silence ``no_levers_found`` false positives while batch
            # kernel_opt is in flight.
            (
                "pending_keep_kernels="
                f"{self.pending_keep_kernel_ids() or '(none)'}"
            ),
            (
                "has_keep_pending_integrate="
                f"{'true' if self.has_keep_pending_integrate else 'false'}"
            ),
            f"kernel_opt_attempts_count={self.kernel_opt_attempts_count}",
            f"rejected_kernel_patches={self._format_rejected_kernel_patches()}",
            f"rejected_kernel_ids={self.rejected_kernel_ids or '(none)'}",
            f"last_baseline={self._format_attempt(self.last_baseline)}",
            f"last_profile={self._format_attempt(self.last_profile)}",
            f"last_backends={self._format_attempt(self.last_backends)}",
            f"last_params={self._format_attempt(self.last_params)}",
            f"last_validate_stack={self._format_attempt(self.last_validate_stack)}",
            f"attempts_history={self._format_attempts_history()}",
            f"last_action_failures={self._format_last_action_failures()}",
            f"last_proposal_advice={self._format_last_proposal_advice()}",
            f"tick={int(self.tick or 0)}  "
            f"target_gap_pct={float(self.target_gap_pct or 0.0):.2f}",
            f"stop_reason={self.stop_reason or '(none)'}",
            f"closing_phase={self.closing_phase}  "
            f"closing_started_unix={self.closing_started_unix or 0.0}  "
            f"closing_report_task_id={self.closing_report_task_id or '(none)'}",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Audit-trail renderers (kernel-parity per-action attempts + global
    # failure log). Compact one-liners so the prompt stays readable.
    # ------------------------------------------------------------------
    def _format_last_proposal_advice(self) -> str:
        """N22: render the FIFO of keyword-implied variant advisories
        so the orchestration LLM sees them on the next tick and can
        extend its variants list. Empty list -> '(none)' so the prompt
        stays consistent across ticks (LLM doesn't have to guess
        whether the field even exists). Each entry is a multi-line
        block; we join with a blank line for readability and prepend
        the index so the LLM can reference them by number."""
        advisories = self.last_proposal_advice or []
        if not advisories:
            return "(none)"
        lines = [""]
        for i, msg in enumerate(advisories, 1):
            lines.append(f"--- advisory #{i} ---")
            lines.append(str(msg))
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _format_attempt(entry: dict[str, Any] | None) -> str:
        """Render one ``last_<action>`` snapshot or attempts[-1] entry."""
        if not isinstance(entry, dict) or not entry:
            return "(none)"
        metric = entry.get("key_metric")
        metric_kind = entry.get("key_metric_kind") or "metric"
        metric_str = (
            f"{metric_kind}={metric:.2f}"
            if isinstance(metric, (int, float)) else f"{metric_kind}=N/A"
        )
        err = entry.get("error_class") or "-"
        ws = entry.get("workspace") or "-"
        return (
            f"status={entry.get('status','?')} "
            f"decision={entry.get('decision','?')} "
            f"{metric_str} err={err} ws={ws} "
            f"task_id={entry.get('task_id','?')} ts={entry.get('ts','?')}"
        )

    def _format_attempts_history(self) -> str:
        """One-line summary across the 6 audit actions.

        Format: ``baseline:total(s<successes>,f<failures>) ...``. Helps
        the LLM gauge per-action reliability without flooding the prompt
        with up to 6x20 individual rows.
        """
        parts: list[str] = []
        for action in sorted(_AUDIT_ACTIONS):
            attempts_attr = f"{action}_attempts"
            history = getattr(self, attempts_attr, None) or []
            if not history:
                continue
            total = len(history)
            succ = sum(
                1 for e in history
                if isinstance(e, dict) and e.get("status") == "succeeded"
            )
            fail = sum(
                1 for e in history
                if isinstance(e, dict) and e.get("status") == "failed"
            )
            parts.append(f"{action}:{total}(s{succ},f{fail})")
        return " ".join(parts) if parts else "(no attempts recorded)"

    def _format_last_action_failures(self) -> str:
        """Render up to the 3 most-recent global failures.

        ``last_action_failures`` is the rich-context companion to
        ``crash_count`` / ``baseline_failure_streak``. We deliberately
        truncate to 3 rows in the prompt so the LLM sees what blew up
        most recently without re-reading 10 rows of stale subprocess
        tails. The full list is still on disk in ``state.json``.
        """
        if not self.last_action_failures:
            return "(none)"
        rows: list[str] = []
        for entry in self.last_action_failures[-3:]:
            if not isinstance(entry, dict):
                continue
            action = entry.get("action") or "?"
            error_class = entry.get("error_class") or "?"
            ts = entry.get("ts") or "?"
            excerpt = entry.get("error_excerpt") or ""
            ws = entry.get("workspace") or "-"
            excerpt_short = excerpt.splitlines()[0][:200] if excerpt else ""
            rows.append(
                f"[{action}/{error_class}@{ts}] err=\"{excerpt_short}\" ws={ws}"
            )
        suffix = (
            f" [+{len(self.last_action_failures) - 3} earlier]"
            if len(self.last_action_failures) > 3 else ""
        )
        return " | ".join(rows) + suffix if rows else "(none)"

    def _format_rejected_kernel_patches(self) -> str:
        if not self.rejected_kernel_patches:
            return "(none)"
        return [
            (
                f"{r.get('kernel_id','?')}: attempts={r.get('attempt_count','?')} "
                f"best_gain={r.get('best_gain_pct','?')} reason={r.get('reason','?')}"
            )
            for r in self.rejected_kernel_patches[-5:]
            if isinstance(r, dict)
        ] or "(none)"

    def _format_discovered_flags(self) -> str:
        """Roofline-v2 N4: layered Z-scheme rendering of discovered flags.

        Before N4 this method emitted only the per-framework count
        summary (``sglang:backend=42/param=58, ...``). The 58 real
        flag names were never visible to the main LLM, which left
        flag selection to either prior knowledge (hallucination risk)
        or implicit pattern-matching against ``params_search.tested``
        fingerprints. Both routes are documented failure modes in
        design §6.1.

        N4 renders every discovered flag grouped by
        ``<framework>.<action_kind>`` with a per-flag ``tested``
        status tag derived from cross-referencing
        ``params_search.tested`` / ``backends_search.tested``:

        * ``[untested]`` — no fingerprint in tested[] contains this flag
        * ``[tested: ±X.XX%]`` — exactly one variant tried, surface its gain
        * ``[tested N vars, best ±X.XX%]`` — multiple variants tried

        The status tags let the main LLM pick "untested + matches
        bottleneck" flags directly (per §8.7 orchestration prompt
        guidance) without needing to reverse-engineer the tested
        ledger fingerprints.

        Empty input degrades to the v0 placeholder message; missing
        framework entries / non-dict children skip silently so a
        malformed dict cannot crash the renderer.
        """
        if not self.discovered_flags:
            return "(none — first backends/params round will populate)"
        out_lines: list[str] = []
        for fw, entry in sorted(self.discovered_flags.items()):
            if not isinstance(entry, dict):
                continue
            for action_kind, key in (
                ("backends", "backend_flags"),
                ("params", "param_flags"),
            ):
                flags = entry.get(key) or []
                if not isinstance(flags, (list, tuple)) or not flags:
                    continue
                out_lines.append(
                    f"  {fw}.{action_kind} ({len(flags)} flags):"
                )
                for flag in sorted(str(f) for f in flags):
                    tag = self._tested_tag_for_flag(flag, action_kind)
                    out_lines.append(f"    {flag:42s} {tag}")
        if not out_lines:
            return "(none)"
        return "\n" + "\n".join(out_lines)

    def _tested_tag_for_flag(self, flag: str, action_kind: str) -> str:
        """Cross-reference ``params_search.tested`` / ``backends_search.tested``
        for variants whose extra_sglang_args contains ``flag`` and surface
        the gain summary.

        ``flag`` is a CLI-style string like ``--enable-two-batch-overlap``.
        Matching is substring containment in ``extra_sglang_args`` —
        accurate enough to catch the common case (single-flag variant)
        and the multi-flag case (synergy combo includes this flag).
        """
        if action_kind == "params":
            search = self.params_search or {}
        elif action_kind == "backends":
            search = self.backends_search or {}
        else:
            return "[untested]"
        tested = search.get("tested") if isinstance(search, dict) else None
        if not isinstance(tested, dict) or not tested:
            return "[untested]"
        matched_gains: list[float] = []
        for snap in tested.values():
            if not isinstance(snap, dict):
                continue
            args = str(snap.get("extra_sglang_args") or "")
            if not args or flag not in args:
                continue
            gain = snap.get("gain_pct")
            if isinstance(gain, (int, float)):
                matched_gains.append(float(gain))
        if not matched_gains:
            return "[untested]"
        n = len(matched_gains)
        best = max(matched_gains)
        if n == 1:
            return f"[tested: {best:+.2f}%]"
        return f"[tested {n} vars, best {best:+.2f}%]"

    @staticmethod
    def _format_variant_line(entry: dict[str, Any]) -> str:
        """One-line render of a search variant for prompt blocks.

        Format: ``{name:28s} {±gain%:>9} (tput=...) <flags> <K=V envs>``.
        Used by :meth:`_format_backend_winners_history` and
        :meth:`_format_search_state` so the LLM sees the same numerical
        signal everywhere the explore ledger surfaces.
        """
        name = str(entry.get("name") or "?")
        gain = entry.get("gain_pct")
        tput = entry.get("tput") or entry.get("output_throughput")
        gain_s = (
            f"{gain:+.2f}%" if isinstance(gain, (int, float)) else " no_meas"
        )
        tput_s = (
            f" (tput={tput:.1f})"
            if isinstance(tput, (int, float)) and tput > 0
            else ""
        )
        args = (
            str(entry.get("extra_sglang_args") or "").strip()
            or "(no-flag)"
        )
        envs = entry.get("extra_envs") or {}
        envs_s = (
            " " + " ".join(f"{k}={v}" for k, v in sorted(envs.items()))
            if envs else ""
        )
        return f"{name:28s} {gain_s:>9}{tput_s}  {args}{envs_s}"

    @staticmethod
    def _enrich_with_tested_gain(
        entry: dict[str, Any], tested: dict[str, Any],
    ) -> dict[str, Any]:
        """Backfill ``gain_pct``/``tput`` from the matching ``tested[fp]``.

        ``params_search.accepted`` is built from ``_variant_to_dict()`` and
        does NOT persist ``gain_pct``; the matching ``tested[fingerprint]``
        entry does. ``backends_search.accepted`` already stamps gain on
        promote (see :meth:`record_backends_accepted`), so this is a no-op
        there. Pulling the value across at render time keeps the renderer
        symmetric and avoids a second writer for accepted entries.
        """
        if (
            entry.get("gain_pct") is not None
            and entry.get("tput") is not None
        ):
            return entry
        fp = str(entry.get("fingerprint") or "")
        snap = tested.get(fp) if fp else None
        if not isinstance(snap, dict):
            return entry
        out = dict(entry)
        if out.get("gain_pct") is None:
            out["gain_pct"] = snap.get("gain_pct")
        if out.get("tput") is None:
            result = (
                snap.get("result")
                if isinstance(snap.get("result"), dict) else {}
            )
            out["tput"] = (
                snap.get("tput") or (result or {}).get("output_throughput")
            )
        return out

    def _format_backend_winners_history(self) -> str:
        """Multi-line render of the explore-round winners history.

        Surfaces per-winner ``gain_pct`` + ``tput`` + resolved flags/envs
        for the last 5 rounds so the IDEA GENERATION step (SKILL.md) has
        real signal to rank retries / synergy combos by, not just names.
        Older rounds collapse to a ``[+N earlier rounds elided]`` line.
        """
        if not self.backend_winners_history:
            return "(no explore rounds completed)"
        last = self.backend_winners_history[-5:]
        out: list[str] = [""]
        for r in last:
            if not isinstance(r, dict):
                continue
            best = r.get("best") if isinstance(r.get("best"), dict) else None
            best_gain = best.get("gain_pct") if best else None
            gain_tag = (
                f" {best_gain:+.2f}%"
                if isinstance(best_gain, (int, float)) else ""
            )
            base = float(r.get("base_tput", 0.0) or 0.0)
            out.append(
                f"    {r.get('round_id','?')} ({r.get('action','?')}): "
                f"base_tput={base:.1f}  "
                f"best={(best.get('name') if best else '(none)')}{gain_tag}"
            )
            winners = [
                w for w in (r.get("winners") or []) if isinstance(w, dict)
            ]
            if not winners:
                out.append("      (no winners this round)")
                continue
            for w in winners:
                out.append("      • " + SharedState._format_variant_line(w))
        if len(self.backend_winners_history) > 5:
            out.append(
                f"    [+{len(self.backend_winners_history) - 5} "
                f"earlier rounds elided]"
            )
        return "\n".join(out)

    def _format_params_search(self) -> str:
        return self._format_search_state(self.params_search)

    def _format_backends_search(self) -> str:
        return self._format_search_state(self.backends_search)

    @staticmethod
    def _format_search_state(search: dict[str, Any] | None) -> str:
        """Multi-line render of a ``*_search`` dedup ledger.

        Each accepted/rejected entry surfaces its real ``gain_pct`` so the
        LLM ranks retries by observed impact instead of guessing from
        names alone. Counts go on the head line; bodies show only the
        most recent 5 entries per bucket (older rows are still in the
        ledger; only the prompt body is truncated).
        """
        if not search:
            return "(none)"
        accepted = list(search.get("accepted") or [])
        rejected = list(search.get("rejected") or [])
        tested = search.get("tested") or {}
        cursor = search.get("cursor", 0)
        out: list[str] = [
            "",
            f"    cursor={cursor}  accepted={len(accepted)}  "
            f"rejected={len(rejected)}  tested={len(tested)}",
        ]
        if accepted:
            out.append("    accepted:")
            for entry in accepted[-5:]:
                if not isinstance(entry, dict):
                    continue
                out.append("      • " + SharedState._format_variant_line(
                    SharedState._enrich_with_tested_gain(entry, tested)
                ))
        if rejected:
            out.append("    rejected (last 5):")
            for entry in rejected[-5:]:
                if not isinstance(entry, dict):
                    continue
                out.append("      • " + SharedState._format_variant_line(
                    entry
                ))
        return "\n".join(out)

    def _format_optimization_stack(self) -> str:
        if not self.optimization_stack:
            return "(none)"
        parts = []
        for entry in self.optimization_stack:
            if not isinstance(entry, dict):
                continue
            parts.append(
                f"{entry.get('action','?')}:{entry.get('variant_name','?')}"
            )
        return parts or "(none)"

    def _format_last_trace_analyze(self) -> str:
        if not self.last_trace_analyze:
            return "(none)"
        ids = [
            str(e.get("kernel_id"))
            for e in self.last_trace_analyze.get("hot_kernels_top15", [])
            if isinstance(e, dict) and e.get("kernel_id")
        ]
        reusable = list(
            self.last_trace_analyze.get("reusable_native_kernel_ids", [])
        )
        base = (
            f"trace={self.last_trace_analyze.get('trace_input','?')} "
            f"candidates_path={self.last_trace_analyze.get('candidates_path','?')} "
            f"top={ids or []} reusable_native={reusable or []}"
        )
        # T3 / T4 finishing-touches: when TraceLens emitted a routing
        # signal (high GPU idle → prefer params; permanent failure →
        # don't keep waiting on kernel candidates), surface it inline
        # so the Orchestration LLM grounds the next ACTION on this
        # signal rather than re-trying TraceLens or guessing why
        # ``top=[]``. We render compactly:
        #   warnings=[high_gpu_idle_pct(idle=35.0%,threshold=20.0%); …]
        # and omit the suffix entirely in the steady-state (no
        # warnings) so existing prompt-format-stable tests don't see
        # gratuitous additions.
        warnings = self.last_trace_analyze.get("trace_health_warnings") or []
        if not warnings:
            return base
        rendered: list[str] = []
        for w in warnings:
            if not isinstance(w, dict):
                continue
            code = str(w.get("code") or "unknown")
            extras: list[str] = []
            if "idle_pct" in w and "threshold_pct" in w:
                extras.append(f"idle={w['idle_pct']}%")
                extras.append(f"threshold={w['threshold_pct']}%")
            if "returncode" in w:
                extras.append(f"rc={w['returncode']}")
            if extras:
                rendered.append(f"{code}({','.join(extras)})")
            else:
                rendered.append(code)
        return f"{base} warnings=[{'; '.join(rendered)}]"

    @staticmethod
    def _strip_base64_data_urls(text: str) -> str:
        """Roofline-v2 N11: replace `data:image/...;base64,...` blobs.

        GPU-empirical context (DeepSeek-R1 session 16:06-02:00 of
        2026-05-19): TraceLens analysis.md embeds a "Performance
        Improvement" chart as a base64 PNG data URL inside a markdown
        image (`![alt](data:image/png;base64,iVBOR...)`). For the R1
        report the single line clocks in at 184530 chars / 184.5 KB
        — i.e. 92% of the 200 KB analysis.md is base64 noise. The
        main Orchestration LLM cannot see PNG pixels through base64,
        so injecting it wholesale dilutes the prompt and obscures
        the high-signal 🔴 P1 / 🟢 P1 recommendation sections (~16 KB
        of pure text) that drive the entire roofline-v2 decision
        loop.

        This helper replaces every `data:image/...;base64,<payload>`
        URL inside a markdown image with a short placeholder that
        preserves the alt-text for context but drops the payload.
        It is intentionally **not** TraceLens-specific — any
        future report format that uses base64 data URLs gets the
        same treatment automatically.

        The strip is read-only relative to the source analysis.md
        file on disk; only the in-memory string injected into the
        LLM prompt is modified. The on-disk file remains intact for
        operator inspection.
        Implementation lives in ``inference_optimizer.tracelens_md``.
        """
        from inference_optimizer.tracelens_md import strip_base64_data_urls

        return strip_base64_data_urls(text)

    def _format_analysis_md_full(self) -> str:
        """Roofline-v2 N5: inject TraceLens analysis.md verbatim into
        the main Orchestration prompt.

        This is the "directly give the report to the orchestrator"
        contract the TraceLens team mandated and v1 violated by
        introducing a sub-agent interpretation layer (see design §6.1).
        The full report is read as-is — no truncation, no
        reformatting, no markdown wrapper — so the main LLM sees the
        same Executive Summary / Top Operations / Recommendations
        sections a human engineer would consume.

        Snapshot-stable: the underlying ``analysis_md_text`` only
        changes when a new ``roofline`` action completes
        (``record_trace_analyze`` overwrites the cache). Within one
        snapshot every tick produces identical SECTION-B prompt
        content, which is what lets Claude Code automatic prompt
        caching (N6) cache this section verbatim across ticks. See
        design §5.1.

        Render modes:

        * **No cache** (`last_trace_analyze` empty or
          ``analysis_md_text`` missing) — emit a one-line hint asking
          the LLM to propose `roofline`. Bookends absent so the
          surrounding prompt summary stays compact when there's no
          report to render.
        * **Cache populated** — emit the full report between explicit
          `=== TraceLens Analysis ... ===` bookends so the LLM can
          syntactically distinguish report-content from surrounding
          SharedState dump lines.

        The header line includes ``snapshot=N`` and
        ``gain_at_snapshot=X.XX%`` so the LLM can detect "report is
        stale; gain has moved by ≥3% since snapshot" without parsing
        the body. The re-profile guidance in orchestration.md
        references these two fields directly.
        """
        cached = self.last_trace_analyze or {}
        md_text = cached.get("analysis_md_text") or ""
        if not md_text:
            # N27: when the coordinator's roofline-required gate has
            # already downgraded to fallback (roofline failed >= N
            # consecutive times and the gate stamped
            # ``fallback_mode_active=True`` on the otherwise-empty
            # last_trace_analyze), tell the LLM it's in degraded
            # mode and grid-tuning actions are unlocked WITHOUT
            # analysis.md guidance. The LLM should still re-propose
            # `roofline` when it suspects the upstream issue cleared
            # (success resets the streak), but in the meantime
            # backends/params/comm_optimization will pass-through and
            # use the executor's default full grid (i.e. pre-roofline
            # behaviour).
            if cached.get("fallback_mode_active"):
                streak = cached.get("fallback_after_failures", "?")
                threshold = cached.get("fallback_threshold", "?")
                return (
                    f"(no TraceLens snapshot yet — N27 FALLBACK MODE: "
                    f"roofline failed {streak} consecutive times "
                    f"(>= threshold {threshold}); "
                    "`backends` / `params` / `comm_optimization` are "
                    "UNLOCKED and will run the executor's full default "
                    "grid without analysis.md-driven N20-A subset / N22 "
                    "advisory. Re-propose `roofline` whenever you think "
                    "the upstream profile/trace issue may have cleared — "
                    "success resets the streak and restores roofline-"
                    "driven variant selection.)"
                )
            return (
                "(no TraceLens snapshot yet — propose `roofline` to "
                "produce one; roofline is a composite action that "
                "runs profile + trace_analyze atomically)"
            )
        # N11: strip base64 image payloads before injection so the
        # prompt is not 92% noise (see _strip_base64_data_urls).
        md_text = self._strip_base64_data_urls(md_text)
        snap = cached.get("roofline_snapshot_id", "?")
        gain = cached.get("roofline_baseline_gain_at_snapshot", 0.0)
        try:
            gain_str = f"{float(gain):.2f}"
        except (TypeError, ValueError):
            gain_str = "?"
        return (
            f"\n=== TraceLens Analysis (snapshot #{snap}, "
            f"gain at snapshot = {gain_str}%) ===\n"
            f"{md_text}\n"
            f"=== End TraceLens Analysis ===\n"
        )

    def _format_last_sweep(self) -> str:
        if not self.last_sweep:
            return "(none)"
        best = self.last_sweep.get("best_overall") or {}
        if not best:
            return f"grid_size={self.last_sweep.get('grid_size', 0)} best=(none)"
        return (
            f"grid_size={self.last_sweep.get('grid_size', 0)} "
            f"best={best.get('name','?')} "
            f"tput={best.get('output_throughput','?')} "
            f"conc={best.get('conc','?')} isl={best.get('isl','?')} osl={best.get('osl','?')}"
        )


__all__ = ["SharedState"]
