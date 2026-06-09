"""SharedState

Persistent session-level state that all reactors read (via prompt injection)
and that PolicyGate uses to enforce CORE_STATE_FIELDS guards.

Backed by JSON at ``$SESSION_DIR/state.json``. The file write is atomic
(``tmp`` + ``os.replace``) so concurrent readers never see a partial blob.
The Coordinator is the **only** writer; LLM agents go through
``UPDATE_STATE`` intents which the Coordinator validates + persists.

fields:

    session_id          str   — set by Coordinator at session creation
    model_name          str   — e.g. "meta-llama/Llama-3.1-8B-Instruct"
    model_path          str   — local NFS path to weights
    model_class         str   — categorical key supplied via --model-class
    model_arch          dict  — advisory architecture profile (hybrid
                                structured + free-text notes) loaded from
                                the launcher's ``$USER_DATA_PATH/model_arch.json``;
                                prompt-context only, no deterministic gating
    model_architectures list  — config.json ``architectures``; stamped into
                                the recipe-snapshot ``extras`` as a KB tag
    model_type          str   — config.json ``model_type``; stamped into
                                the recipe-snapshot ``extras`` as a KB tag
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
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# the ``orchestrator.scoring`` module was retired; the
# v0.6 ``ActionScore`` / ``rank_top_k`` / ``target_gap_multiplier``
# imports below are gone. The LLM now decides by reading facts
# (phase / gaps / KB), not by consuming a system-side priority
# ranking. See ``Coordinator._compose_prompt`` for the replacement
# fact set.


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


_RAY_TRANSIENT_ERROR_CLASSES = frozenset({
    "ray_transient",
    "raylet_died",
    "localrayletdiederror",
    "ray_submission_failed",
})

_RAY_TRANSIENT_MARKERS = (
    "localrayletdiederror",
    "raylet died",
    "ray submission failed",
    "failed to start ray",
    "ray start failed",
    "global_state_accessor",
    "too many open files",
    "modulenotfounderror: no module named 'geak_submit'",
    "no module named 'geak_submit'",
)


def _kernel_opt_result_text_blobs(result: dict[str, Any]) -> str:
    """Collect free-text fields from a kernel_opt result for Ray matching."""
    parts: list[str] = [
        str(result.get("error") or ""),
        str(result.get("stderr_tail") or ""),
        str(result.get("error_class") or ""),
    ]
    proposal = result.get("proposal") or {}
    if isinstance(proposal, dict):
        parts.extend(str(x) for x in (proposal.get("reasons") or []))
    verification = result.get("verification") or {}
    if isinstance(verification, dict):
        parts.append(str(verification.get("artifact_error") or ""))
    for key in ("attempts", "backend_fallback_attempts"):
        for att in result.get(key) or []:
            if not isinstance(att, dict):
                continue
            parts.append(str(att.get("stdout_tail") or ""))
            parts.append(str(att.get("stderr_tail") or ""))
            parts.append(str(att.get("error") or ""))
    return "\n".join(parts).lower()


def is_ray_transient_kernel_opt_failure(result: dict[str, Any]) -> bool:
    """True when a kernel_opt failure was caused by Ray/raylet instability."""
    if not isinstance(result, dict):
        return False
    err_class = str(result.get("error_class") or "").strip().lower()
    if err_class in _RAY_TRANSIENT_ERROR_CLASSES:
        return True
    blob = _kernel_opt_result_text_blobs(result)
    if not blob.strip():
        return False
    if any(marker in blob for marker in _RAY_TRANSIENT_MARKERS):
        return True
    if "connectionerror" in blob and "ray" in blob:
        return True
    return False


def _default_kernel_opt_max_ray_retries() -> int:
    env_v = os.environ.get("INFERENCE_OPTIMIZER_KERNEL_OPT_MAX_RAY_RETRIES")
    if env_v:
        try:
            return max(1, int(env_v))
        except (TypeError, ValueError):
            pass
    return _DEFAULT_KERNEL_OPT_MAX_RAY_RETRIES


# Ordered (key, label) projection for the advisory ``model_arch`` profile.
# ``notes`` is rendered last as a free-text trailer. Keys absent / empty /
# ``None`` are dropped so the rendered block stays compact and only
# surfaces fields the launcher actually populated.
_MODEL_ARCH_STRUCTURED_FIELDS: tuple[tuple[str, str], ...] = (
    ("decoder_type", "decoder"),
    ("attention", "attention"),
    ("layer_mix", "layers"),
    ("kv_cache_per_token", "kv/token"),
    ("active_params", "params"),
    ("num_experts", "experts"),
    ("experts_per_tok", "experts/tok"),
    ("mtp", "mtp"),
    ("swa_window", "swa_window"),
    ("norm", "norm"),
)


def render_model_arch_compact(arch: dict | None) -> str:
    """Render the advisory ``model_arch`` profile as a single compact line.

    Returns ``""`` when ``arch`` is empty / not a dict so callers can omit
    the prompt block entirely. Structured fields render as
    ``decoder=Sparse MoE; attention=MLA; ...`` followed by an optional
    ``notes=<free text>`` trailer. Empty / ``None`` field values are
    dropped. This is the single source of truth shared by the
    orchestration prompt summary and the specialist ``arch_notes`` carrier.
    """
    if not isinstance(arch, dict) or not arch:
        return ""
    parts: list[str] = []
    for key, label in _MODEL_ARCH_STRUCTURED_FIELDS:
        val = arch.get(key)
        if val is None or val == "":
            continue
        parts.append(f"{label}={val}")
    notes = str(arch.get("notes") or "").strip()
    if notes:
        parts.append(f"notes={notes}")
    return "; ".join(parts)


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
# Ray/raylet death is transient infra — retry after bootstrap instead of
# permanently retiring hot kernels on the first LocalRayletDiedError.
_DEFAULT_KERNEL_OPT_MAX_RAY_RETRIES = 3

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

# phase_history cap. There are only 6 phases in the line
# (PRELUDE/FRAMEWORK_PR/EXPLORE/KERNEL/SWEEP/CLOSE) so 100 rows is wildly generous;
# the only realistic path to hitting it is repeated escalate/recover
# loops, which we'd want surfaced as a warning anyway. Cap is enforced
# in :meth:`SharedState.record_phase_transition`.
_PHASE_HISTORY_CAP = 100

# roofline_snapshots history cap. PRELUDE bootstrap writes #1; every
# +10% watermark crossing writes a refresh. Even a pathological run
# with 50 watermark crossings would stay well under this cap; the
# limit only exists to prevent unbounded state.json growth if the
# refresh policy ever loosens. Cap is enforced in
# :meth:`SharedState.record_trace_analyze`.
_ROOFLINE_SNAPSHOTS_CAP = 50

# gap ledger caps. ``_GAPS_MAX_ENTRIES`` bounds the
# total list so a pathological session can't blow up state.json;
# ``_GAPS_ATTEMPTS_HISTORY`` bounds the per-gap ``attempts`` list so even
# a hot gap with 200 KEEP/REVERT events stays cheap to render. Both caps
# are enforced inside :meth:`SharedState.upsert_gap`.
_GAPS_MAX_ENTRIES = 50
_GAPS_ATTEMPTS_HISTORY = 20

# Set of action kinds that participate in the kernel-equivalent per-action
# audit trail (Plan: SharedState audit-trail). Kernel-owned actions are
# intentionally excluded — they already have richer dedicated structures
# (last_kernel_opt / kernel_opt_attempts / kernel_integrate_attempts /
# rejected_kernel_*). Membership is consulted by Coordinator and renderer
# helpers so adding a new audit action is a one-line change.
_AUDIT_ACTIONS: frozenset[str] = frozenset({
    "baseline", "profile", "sweep", "explore",
    # F1-3 (Roofline-v2 / ) + N10: the composite
    # ``roofline`` action runs profile + trace_analyze atomically.
    # Audit each attempt so the prompt's RECENT ACTION ATTEMPTS block
    # surfaces the snapshot id + analysis_md_path the executor produced
    # (or the failure phase / error_class on the failure path).
    "roofline",
})

# Mapping from audit-action name to (result-dict key, prompt-display label).
# ``key`` is what we read out of the executor result dict; ``label`` is the
# ``key_metric_kind`` written into each attempt entry so prompt readers
# know how to interpret the number (e.g. ``output_throughput`` vs raw
# ``gain_pct`` vs ``validated_gain_pct``).
_KEY_METRIC_MAP: dict[str, tuple[str, str]] = {
    "baseline": ("output_throughput", "output_throughput"),
    "profile":  ("output_throughput", "output_throughput"),
    "sweep":    ("output_throughput", "output_throughput"),
    "explore":  ("best_gain_pct",     "gain_pct"),
    # F1-3 + N10: ``roofline`` is an analysis composite, not a
    # benchmark — its key metric is the monotonic snapshot id, not a
    # throughput number.
    "roofline": ("snapshot_id",       "snapshot_id"),
}


#: top-level state.json schema version. v0.6 did
#: not write a value; ``from_dict`` treats an absent key as
#: ``schema_version=1`` and runs the §3.10 §5.2 migration step,
#: bumping to ``LATEST_STATE_SCHEMA_VERSION`` on the first save.
LATEST_STATE_SCHEMA_VERSION: int = 2


# The payload-surface field was renamed from ``extra_sglang_args``
# (sglang-era name) to ``extra_server_args`` (framework-neutral).
# The on-disk state.json may carry the legacy key in any of several
# deeply-nested ledgers, so a one-shot walk-and-rewrite on load is
# the cleanest migration — the next save then emits canonical only
# and a re-load is a no-op.
_PHASE4_LEGACY_KEY_RENAMES: dict[str, str] = {
    "extra_sglang_args":           "extra_server_args",
    "candidate_extra_sglang_args": "candidate_extra_server_args",
}


def _migrate_legacy_extra_sglang_args_keys(obj: Any) -> int:
    """Recursively rewrite legacy ``extra_sglang_args`` field names in-place.

    Walks every dict and list reachable from ``obj`` and renames keys
    listed in :data:`_PHASE4_LEGACY_KEY_RENAMES` to their canonical
    counterparts. When both names are present on the same dict the
    canonical value is kept (writer migration is considered done as
    soon as the new key appears); otherwise the legacy value is
    copied over.

    Returns the total number of keys rewritten so the caller can log
    a single info-level migration event.
    """
    migrated = 0
    if isinstance(obj, dict):
        for legacy_key, canonical_key in _PHASE4_LEGACY_KEY_RENAMES.items():
            if legacy_key in obj:
                if canonical_key not in obj:
                    obj[canonical_key] = obj.pop(legacy_key)
                else:
                    del obj[legacy_key]
                migrated += 1
        for v in obj.values():
            migrated += _migrate_legacy_extra_sglang_args_keys(v)
    elif isinstance(obj, list):
        for item in obj:
            migrated += _migrate_legacy_extra_sglang_args_keys(item)
    return migrated


@dataclass
class TraceAnalyzeSnapshot:
    """Reference shape for ``SharedState.last_trace_analyze``.

    placeholder added by ``F0_pre_merge.MD`` §9.
    F1 will populate the canonical 11-field dict via
    ``SharedState.record_trace_analyze`` (ported from main); this dataclass
    serves as a typed reader on the consumer side via :meth:`from_dict`.

    The on-disk shape stays a plain ``dict`` so the legacy ``state.json``
    round-trip (Inv-10.1 fact-layer compatibility) keeps working.
    """

    trace_input: str = ""
    candidates_path: str = ""
    kernel_roofline_path: str = ""
    hot_kernels_top15: list[dict[str, Any]] = field(default_factory=list)
    kernel_roofline_top15: list[dict[str, Any]] = field(default_factory=list)
    task_groups: list[dict[str, Any]] = field(default_factory=list)
    reusable_native_kernel_ids: list[str] = field(default_factory=list)
    trace_health_warnings: list[dict[str, Any]] = field(default_factory=list)
    analysis_md_path: str = ""
    analysis_md_text: str = ""
    roofline_snapshot_id: int = 0
    roofline_baseline_gain_at_snapshot: float = 0.0
    ts: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "TraceAnalyzeSnapshot":
        d = d or {}
        return cls(
            trace_input=str(d.get("trace_input") or ""),
            candidates_path=str(d.get("candidates_path") or ""),
            kernel_roofline_path=str(d.get("kernel_roofline_path") or ""),
            hot_kernels_top15=list(d.get("hot_kernels_top15") or []),
            kernel_roofline_top15=list(d.get("kernel_roofline_top15") or []),
            task_groups=list(d.get("task_groups") or []),
            reusable_native_kernel_ids=list(
                d.get("reusable_native_kernel_ids") or []
            ),
            trace_health_warnings=list(d.get("trace_health_warnings") or []),
            analysis_md_path=str(d.get("analysis_md_path") or ""),
            analysis_md_text=str(d.get("analysis_md_text") or ""),
            roofline_snapshot_id=int(d.get("roofline_snapshot_id") or 0),
            roofline_baseline_gain_at_snapshot=float(
                d.get("roofline_baseline_gain_at_snapshot") or 0.0
            ),
            ts=str(d.get("ts") or ""),
        )


@dataclass
class SharedState:
    # versioned state.json schema. Bumped by the
    # migration step in :meth:`from_dict` whenever a payload is
    # loaded. Fresh sessions are born at the latest version.
    schema_version: int = LATEST_STATE_SCHEMA_VERSION
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
    # Advisory architecture profile (hybrid: a few structured fields +
    # free-text ``notes``). Produced pre-launch by the SKILL launcher
    # (LLM Architecture Gallery lookup, else a lightweight classify) and
    # loaded from ``$USER_DATA_PATH/model_arch.json``. Prompt-context only
    # — no deterministic gating consumes it (atom seed grid / compat
    # filter / framework gap token / recipe key stay on ``model_class``).
    # Empty dict means "no profile available"; renderers omit the block.
    model_arch: dict = field(default_factory=dict)
    # KB tags lifted verbatim from the model weights' ``config.json``
    # (``architectures`` list + ``model_type`` string). Populated at
    # fresh-launch by ``cli._load_model_config_tags``; resume rehydrates
    # the persisted values from state.json. Stamped into the recipe-snapshot
    # ``extras`` on every KB write (T0 anchor + KEEP/REVERT/CLOSE amend) so a
    # fine-tuned model carries the same architecture identity as the base
    # model it derives from. Empty (``[]`` / ``""``) means "config.json
    # absent or unreadable".
    model_architectures: list[str] = field(default_factory=list)
    model_type: str = ""
    framework: str = ""
    gpu_type: str = ""
    # Workload metadata mirrored from manifest.json at session start
    #. Used by:
    #   * Coordinator._warm_specialist_params to populate
    #     specialist task params so SpecialistPromptInputs renders
    #     real hardware context in "## 2. HARDWARE CONTEXT" instead
    #     of the dataclass defaults (which silently render TP=1 and
    #     make comm_specialist self-veto with "TP=1 → cross-GPU
    #     collectives non-actionable" even on TP=8 sessions).
    #   * the per-tick orchestration prompt's SESSION CONTEXT block.
    # Populated by ``cli._init_fresh_session`` from CLI flags / env;
    # resumed sessions read them back from state.json verbatim and
    # cli.py re-exports the env vars (TP/CONC/ISL/OSL/MAX_MODEL_LEN/
    # PRECISION) so downstream executors see the same values the
    # original run used. Zero / empty means "unspecified".
    tp: int = 0
    # Expert-parallel size for MoE inference. Mirror of the ``EP`` env
    # var (cli writes both at boot). Stored on SharedState so resume in
    # a fresh shell that hasn't ``export EP`` still recovers the
    # original value — without this, KB warm-start queries would lose
    # the EP filter on resume and the recipe anchor's ``ep`` tag would
    # become unreliable. Multi-node lifecycle scripts still read the
    # env var; this field is the resume-safe authority.
    ep: int = 0
    precision: str = ""
    # Recipe-snapshot v2 canonical id is a five-tuple
    # ``model + hardware + framework + framework_version + precision``;
    # ``framework_version`` is the only one not derivable from existing
    # SharedState fields, so it lives here. Populated by ``cli`` from
    # ``--framework-version`` (operator override) or, when omitted,
    # auto-detected via :func:`recipe_snapshot_constants.detect_framework_version`
    # (best-effort: imports the framework's top-level package and reads
    # ``__version__``). Mirrored to env ``FRAMEWORK_VERSION`` on resume so
    # downstream executors see the same value the original run used.
    # Empty string means "unknown" — the recipe canonical_id falls back
    # to ``unknown_version`` and KB lookups for this dimension lose
    # specificity (rows still write under that slug).
    framework_version: str = ""
    conc: int = 0
    isl: int = 0
    osl: int = 0
    max_model_len: int = 0
    kernel_enabled: bool = True
    # When False (CLI ``--no-explore``) the EXPLORE phase is skipped:
    # PRELUDE / FRAMEWORK_PR route straight to KERNEL (or SWEEP when
    # kernel is also disabled). Mirrors the ``kernel_enabled`` /
    # ``framework_phase_enabled`` opt-out pattern.
    explore_enabled: bool = True
    # After deterministic FP8 GEMM tuning succeeds, continue into source-level
    # kernel_opt by default. Operators can disable this for a GEMM-only run.
    continue_kernel_after_gemm: bool = True
    # SWEEP-phase post-sweep concurrency sweep. On by default;
    # operator opts out via ``--no-enable-conc-sweep`` /
    # ``INFERENCE_OPTIMIZER_ENABLE_CONC_SWEEP=0``. When True, the
    # Coordinator auto-enqueues a ``conc_sweep`` task right after the
    # SWEEP-entry sweep task lands (see ``_on_sweep_task_completed``);
    # the executor runs baseline + current_best across the
    # ``conc_sweep_concs`` ladder bounded by
    # ``conc_sweep_total_budget_sec`` total wall-clock.
    conc_sweep_enabled: bool = True
    # CONC ladder used by the conc_sweep action. Default mirrors
    # ``orchestrator.conc_sweep.DEFAULT_CONCS``. Empty list short-
    # circuits the executor with skip_reason=empty_conc_list.
    conc_sweep_concs: list[int] = field(
        default_factory=lambda: [1, 2, 4, 8, 16, 32, 64, 128],
    )
    # Total wall-clock budget (seconds) for the whole conc_sweep
    # action. Default 9000 (~2.5h). 0 disables the gate so the
    # executor runs every variant until the per-variant timeout
    # alone caps it.
    conc_sweep_total_budget_sec: int = 9000
    # Per-variant Magpie subprocess timeout (seconds). Default 1800
    # (~30 min). Effective timeout is clamped to the remaining total
    # budget so the last variant cannot overshoot the deadline.
    conc_sweep_variant_timeout_sec: int = 1800
    target_summary: str = ""
    baseline_tput: float = 0.0
    baseline_accuracy: float = 0.0
    baseline_failure_streak: int = 0
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
    # idling until the wall-clock deadline. Reset to 0 the moment we
    # actually enter closing.
    consecutive_silent_ticks: int = 0
    # Path to the YAML the baseline executor materialized with the operator's
    # workload envs (CONC/ISL/OSL/TP/MAX_MODEL_LEN/PRECISION/RUN_EVAL/...).
    # Coordinator injects this into params/backends/sweep tasks as
    # ``task.params["config_path"]`` so downstream variants inherit the same
    # workload contract baseline ran. Empty before the first baseline result;
    # downstream executors fall back to materializing the shipped YAML
    # against current process env when this is empty.
    baseline_config_path: str = ""
    # GAP 5 (KB tag completeness): runtime component versions populated
    # by cli at boot from manifest / stack_fingerprint. Mirror of the
    # equivalent fields written to ``recipe.attrs`` by the T0 backfill,
    # exposed here so ``_collect_workload_tags`` can stamp them onto
    # every lesson / pitfall write WITHOUT having to re-parse manifest
    # at fact-write time.
    #
    # Shape: ``{"sglang": "0.5.11", "vllm": "0.19.0", "rocm": "6.2.0",
    #           "aiter": "abc123", "image_digest": "sha256:..."}``.
    # All keys optional; empty / "unknown" values are stripped before
    # writing onto the recipe so KB attrs stays compact. Resume-safe
    # because the field is JSON-serialised into state.json.
    stack_fingerprint_meta: dict = field(default_factory=dict)
    # GAP 5: extra workload-shape fields parsed from the materialized
    # baseline YAML — ``max_running_requests`` / ``max_num_seqs`` /
    # ``chunked_prefill_enabled`` / ``enable_torch_compile`` /
    # ``quant_scheme`` / ``workload_mode``. These are *not* in the
    # canonical id (would explode the recipe space) but they are
    # crucial filters for the warm-start ladder and the lesson reader
    # so future sessions can pick a closer prior.
    #
    # Populated by ``BaselineExecutor._promote_to_shared_state`` after
    # the baseline YAML is materialized. Empty dict before first
    # baseline result; downstream consumers tolerate missing keys.
    baseline_workload_extra: dict = field(default_factory=dict)
    # GAP 1 (warm-recipe replay) — one-shot guard so the PRELUDE
    # auto-enqueue path doesn't fire twice for the same session
    # (resume safety). The Coordinator flips this to True at the
    # moment a ``replay_warm_recipe`` task is created; the field
    # survives resume via state.json so a robustness restart cannot
    # accidentally double-spend the replay budget. ``False`` is the
    # default for fresh sessions.
    warm_replay_attempted: bool = False
    # GAP 1 supporting field — one-shot guard for
    # ``_inject_warm_recipe_history_into_ledger``. Decoupled from
    # ``warm_replay_attempted`` because the history injection is
    # independent of whether the operator enabled warm replay:
    # ``--no-warm-replay`` users still benefit from "don't retry
    # known-failed variants". Resume-safe: persists into state.json
    # so a robustness restart cannot double-inject the same rows.
    warm_history_injected: bool = False
    # GAP 1 — structured outcome of the warm-replay attempt so the
    # report / prompt can render "we tried the KB best_config and
    # got +X% (vs the recipe's claim of +Y%)". Shape::
    #
    #   {
    #     "status":            "reproduced" | "drift" | "failed" | "skipped",
    #     "expected_gain_pct": 25.0,
    #     "actual_gain_pct":   23.5,
    #     "warm_recipe_tier":  "exact",
    #     "warm_recipe_conf":  1.0,
    #     "replay_task_id":    "task-uuid",
    #     "reason":            "..."        # only on failed / skipped / drift
    #   }
    #
    # Empty dict before the replay completes or when ``--no-warm-replay``
    # is in effect. Persisted into state.json so resume + breakdown
    # collectors see it.
    warm_replay_outcome: dict = field(default_factory=dict)
    # Wall-clock seconds the baseline Magpie subprocess took to
    # finish (success path only). Populated by Coordinator from the
    # baseline executor's ``subprocess_runtime_sec`` and read by the
    # ExploreExecutor to derive a per-variant overtime-kill deadline
    # (``baseline_runtime_sec * SharedState.explore_overtime_kill_ratio``).
    # Zero = "unset" → overtime kill is a no-op (variant uses the
    # legacy timeout). Resume restores the value verbatim.
    baseline_runtime_sec: float = 0.0
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
    # Cumulative gain measured by re-baselining a fresh server with
    # EVERY KEEP'd entry of ``optimization_stack`` applied end-to-end.
    # The plain ``cumulative_gain`` field only sums per-round gains
    # (which do not compose linearly), so the validated number is
    # what the final report quotes. the
    # rebench runs inline inside the merged ``explore`` action's
    # per-KEEP loop; the standalone v0.6 ``validate_stack`` action
    # is denied by PolicyGate. Stays 0.0 until the first KEEP
    # cleared its inline rebench.
    cumulative_gain_validated: float = 0.0
    cumulative_gain_validated_ts: str = ""
    # Length of ``optimization_stack`` at the time of the last successful
    # inline stack rebench; used by the Coordinator to decide whether
    # the current stack still matches the validated number, or whether
    # the TODO 4 stack-rebench guard should fire after new KEEPs landed.
    cumulative_gain_validated_stack_len: int = 0
    # Tput (tok/s/GPU) measured at the most recent successful roofline
    # task; serves as the watermark for the gain-driven roofline refresh
    # policy. Coordinator enqueues a fresh roofline task whenever
    # ``baseline_tput * (1 + cumulative_gain_validated/100) /
    # last_roofline_tput >= 1.10`` (a 10% step over the last
    # measurement, compound). Initialised from the PRELUDE roofline.
    last_roofline_tput: float = 0.0
    stop_reason: str = ""
    # Closing phase — set when the wall-clock deadline fires. While True,
    # Coordinator skips reactor passes and only pumps the dispatcher to
    # drain a Coordinator-enqueued ``report`` task. Cleared on resume.
    closing_phase: bool = False
    closing_started_unix: float = 0.0
    closing_report_task_id: str = ""
    # set to True at the END of the
    # CLOSE phase 5-step sequencer (after step 5). cli.finally reads
    # this to short-circuit its emergency ``session_breakdown.json``
    # write so the sequencer's artifact isn't overwritten by a
    # duplicate post-stop() pass. Resume clears it back to False so
    # a subsequent resumed CLOSE can re-run the sequence (idempotent
    # by design — every step uses fixed idempotency keys / set-once
    # writes).
    close_sequence_done: bool = False
    # Auto-roofline gate (EXPLORE-entry): set to the task_id of the
    # Coordinator-enqueued ``roofline`` task while it is pending. The
    # field is read by ``_handle_delegate`` to block first-round
    # specialist dispatches until the snapshot lands (so specialists
    # have ``analysis.md`` ground truth to reason against), and is
    # cleared by ``_promote_to_shared_state`` when that roofline task
    # reaches a terminal state. Always defaults to "" on a fresh
    # session; resume readers tolerate the field being absent.
    auto_roofline_pending_task_id: str = ""
    current_action: str = ""
    crash_count: int = 0
    # Last Coordinator-side exception caught by the tick-loop resilience
    # guard. Coordinator-only; gives postmortems a traceback without relying
    # on harness stdout.
    last_tick_exception: dict[str, Any] = field(default_factory=dict)
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

    # ------------------------------------------------------------------
    # Roofline-v2 trace-analyze cache (canonical, post-M4 rename).
    #
    # ``last_trace_analyze`` is the canonical 11-field dict written by
    # :meth:`record_trace_analyze` after a successful ``trace_analyze``
    # sub-step (typically driven by ``RooflineExecutor``). Coordinator
    # short-circuits subsequent identical ``trace_analyze`` requests off
    # this cache so Orchestration does not waste budget re-analysing the
    # same trace.
    #
    # ``roofline_snapshot_id`` mirrors ``last_trace_analyze['roofline_snapshot_id']``
    # at the top level for fast PolicyGate / Coordinator access (avoids the
    # nested-dict lookup on hot paths).
    #
    # Pre-M4 ``last_select_kernels`` field was removed in this branch
    # (commit "drop select_kernels alias…") — all readers must use
    # ``last_trace_analyze``. Resume of a stale state.json that still
    # carries ``last_select_kernels`` will silently drop the extra key
    # via :meth:`_apply_loaded_state`.
    # ------------------------------------------------------------------
    last_trace_analyze: dict[str, Any] = field(default_factory=dict)
    roofline_snapshot_id: int = 0
    # Append-only history of compact roofline snapshots used by
    # ``report.py`` to render the ``## Roofline Comparison`` section.
    # PR #321 retired the legacy ``last_trace_analyze_baseline``
    # baseline-freeze field; the snapshot history preserves the first
    # (baseline) snapshot across watermark-driven refreshes of
    # ``last_trace_analyze`` so a real before/after comparison stays
    # available even after multiple +10% refreshes overwrite the
    # latest-snapshot cache.
    #
    # Each entry matches the shape returned by
    # :func:`orchestrator.roofline_snapshot.build_roofline_snapshot`
    # (snapshot_id / ts / compute_pct / idle_pct / comm_pct /
    # top_bottleneck / top_kernel) plus ``analysis_md_path`` and
    # ``trace_input`` for downstream re-extraction. Capped at
    # ``MAX_ROOFLINE_SNAPSHOTS`` to bound on-disk state.json size.
    roofline_snapshots: list[dict[str, Any]] = field(default_factory=list)
    # N27 — outer roofline failure counter. Bumped by
    # ``Coordinator._promote_to_shared_state`` on every failed
    # ``roofline`` task and reset to 0 on the next successful one.
    # The per-phase fallback in commit 6078012 ("per-phase fallback
    # when auto-roofline fails") already handles the EXPLORE-degraded
    # / KERNEL-fall-back-to-profile behaviour main's N27 fallback
    # threshold targets; the counter exists here for prompt-side
    # visibility (the LLM sees how many outer attempts failed).
    roofline_failure_streak: int = 0

    # ------------------------------------------------------------------
    # Feature toggles (mirrored from ``cli.py`` flags at session start).
    # ------------------------------------------------------------------
    # FRAMEWORK_PR phase toggle. When True (the default) the Coordinator
    # routes PRELUDE → FRAMEWORK_PR → EXPLORE; the FRAMEWORK_PR phase
    # batches ``fa phase-discover`` candidates, runs each through the
    # standard Critic-gated ``integrate_patch``-style benchmark, and
    # KEEPs winners to ``optimization_stack``. Operators opt out via
    # ``--no-framework`` (PRELUDE → EXPLORE directly, ``prelude_done``
    # reason preserved). Replaces the v0.8 ``framework_agent_enabled``
    # serving-sub-kind toggle; PolicyGate's
    # ``framework_pr_action_not_llm_proposable`` rule keeps the LLM
    # from proposing the action itself.
    framework_phase_enabled: bool = True
    # FRAMEWORK_PR phase progress tracker. One entry per candidate
    # benchmark, written by the FrameworkPrExecutor: ``{candidate_id,
    # pr_url, batch_id, status, pre_tput, post_tput, gain_pct, kept,
    # ts}``. Used by the breakdown collector + the phase exit logic's
    # plateau judgment (lookback over the per-batch max gain).
    framework_pr_phase_progress: list[dict[str, Any]] = field(
        default_factory=list,
    )
    # One row per ``fa phase-discover`` batch: ``{batch_id, ts,
    # candidate_count, max_gain_pct_observed_in_batch}``. Read by
    # ``exit_normal_framework_pr`` for the plateau gate (default: 3
    # consecutive batches with max gain < 1% → exit).
    framework_pr_batches: list[dict[str, Any]] = field(
        default_factory=list,
    )
    # Coordinator sets this to True when the FRAMEWORK_PR loop has no
    # more candidates to run (``fa phase-discover`` returned 0 or every
    # candidate in the latest batch has been tried). ``compute_next_phase``
    # consults this for the ``framework_pr_phase_done`` exit reason.
    framework_pr_phase_done: bool = False
    # Consecutive ``fa phase-discover`` failures (timeout / non-zero
    # exit / parse error). The Coordinator's discover loop bumps this
    # on each failure and resets to 0 on a successful batch. Phase is
    # only marked done after ``DISCOVER_FAILURE_RETRY_LIMIT`` (default
    # 3) consecutive failures, so a transient network blip or a slow
    # PR scan no longer collapses the whole FRAMEWORK_PR phase
    # silently.
    framework_pr_discover_failures: int = 0
    # Per-repo candidate cap passed to ``fa phase-discover`` during the
    # FRAMEWORK_PR phase. 0 / unset → the Coordinator falls back to
    # ``DEFAULT_FRAMEWORK_PR_MAX_CANDIDATES``. Bumping this makes each
    # batch probe deeper (more PRs per repo per batch). Round-trips via
    # the dataclass asdict/from_dict path like every other field.
    framework_pr_max_candidates: int = 0
    # FRAMEWORK_PR Critic-gate decisions, one row per reviewed candidate:
    # ``{candidate_id, batch_id, verdict, rationale, ts}``. The
    # Coordinator's pump calls the Critic backend before each
    # ``_enqueue_framework_pr_task``; ``approve`` proceeds, ``reject``
    # records a ``critic_denied`` progress row instead. The cache lets a
    # resume avoid double-calling the Critic on candidates the prior run
    # already reviewed.
    framework_pr_critic_decisions: list[dict[str, Any]] = field(
        default_factory=list,
    )
    # When True (the default) the Coordinator's auto-managed analysis
    # action — at PRELUDE bootstrap and on every +10% watermark
    # crossing — is ``roofline`` (composite: profile + trace_analyze +
    # analysis.md snapshot). When False the same trigger paths enqueue
    # plain ``profile`` instead (no trace_analyze, no analysis.md);
    # behaviour is otherwise identical (same idempotency keys, same
    # pending-task gate, same watermark anchor update). Both kinds
    # remain Coordinator-internal and are denied by PolicyGate when
    # proposed by the LLM (``analysis_action_not_llm_proposable``).
    enable_roofline: bool = True
    # Per-variant overtime kill multiplier for ExploreExecutor: when
    # > 0 AND ``baseline_runtime_sec`` > 0, single-variant Magpie runs
    # in the explore loop are killed once their wall-clock exceeds
    # ``baseline_runtime_sec * explore_overtime_kill_ratio``. The
    # variant is recorded with ``outcome='KILLED_OVERTIME'`` +
    # ``runtime_sec`` + ``wall_clock_ratio_vs_baseline`` (no tput) so
    # the orchestration LLM can distinguish "ran too slow → early
    # kill" from "benchmark crashed / timed out at the hard cap".
    # Mirrored from ``--explore-overtime-kill-ratio`` at session
    # start. The stack-rebench step does NOT use this gate — only the
    # initial single-variant run (kernel rationale: a stack already
    # carrying multiple winners can legitimately take longer than the
    # bare baseline, and rebench is small enough that the hard
    # ``variant_timeout_sec`` is a sufficient backstop). Default 1.10
    # (kill at +10 % over baseline wall-clock).
    explore_overtime_kill_ratio: float = 1.10
    # Per-variant hard timeout override for ExploreExecutor. Mirrored from
    # ``--explore-variant-timeout-sec`` (CLI) /
    # ``INFERENCE_OPTIMIZER_EXPLORE_VARIANT_TIMEOUT_SEC`` (env) at session
    # start. ``0`` means "auto-derive from baseline_runtime_sec * (kill_ratio
    # + safety_margin)" — see ``explore._compute_explore_variant_timeout``.
    # An explicit positive value pins the cap (e.g. CI smoke runs that want
    # a tight bound regardless of baseline). The Coordinator injects this
    # into every explore task's params so the executor's call site honors
    # it without each LLM proposal having to re-emit it.
    explore_variant_timeout_sec_override: int = 0
    # Headroom on top of the soft kill ratio when the executor auto-derives
    # the per-variant hard cap: ``timeout = baseline_runtime_sec *
    # (kill_ratio + safety_margin)``. Default 0.5 (≈ 50 % of baseline as
    # buffer for one-off variant cold starts: torch.compile AOTI compile,
    # fresh aiter shapes, spec-decoding draft load). Mirrored from
    # ``--explore-variant-timeout-safety-margin`` (CLI) /
    # ``INFERENCE_OPTIMIZER_EXPLORE_VARIANT_TIMEOUT_SAFETY_MARGIN`` (env).
    # Has no effect when ``explore_variant_timeout_sec_override > 0``
    # (operator-pinned timeout bypasses the auto-derive).
    explore_variant_timeout_safety_margin: float = 0.5
    # Opt-in hard filter: when True AND the latest
    # ``roofline_saturation_history`` snapshot has at least one direction
    # crossing the saturation threshold, ExploreExecutor drops every
    # variant whose flags target ONLY saturated directions before running
    # the grid. Variants targeting at least one non-saturated direction
    # (or that don't match the categorization table) are kept. The
    # existing ``roofline_saturation_advisory`` (soft prompt hint) stays
    # unchanged. Mirrored from ``--explore-roofline-hard-gate`` (CLI).
    explore_roofline_hard_gate: bool = False
    # Most recent workload sweep; used to reason about gains beyond the
    # smoke workload (CONC/ISL/OSL frontier).
    last_sweep: dict[str, Any] = field(default_factory=dict)
    # Mirrors last_sweep for the conc_sweep post-hook so SWEEP→CLOSE can
    # exit on conc_sweep completion (not only sweep completion). Carries
    # status / skip_reason / summary so phase_state.exit_normal_sweep
    # treats succeeded / partial / completed / skipped as terminal.
    last_conc_sweep: dict[str, Any] = field(default_factory=dict)
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
    # GEAK FP8 GEMM tuning snapshot. This is kernel-owned but not a
    # source-level kernel rewrite: it produces an aiter A8W8 block-scale
    # tuned CSV and (for SGLang) a dispatch patch before kernel_opt runs.
    last_gemm_tuning: dict[str, Any] = field(default_factory=dict)
    # merged explore action snapshot. Same
    # schema as the other ``last_<action>`` mirrors.
    last_explore: dict[str, Any] = field(default_factory=dict)
    # Roofline-v2 N10: composite roofline action audit snapshot +
    # rolling history. Mirrors the v0 per-action audit pattern (one
    # dict snapshot for "what was the most recent run", one capped
    # list for "what was the per-tick history"). Counted by N7's
    # verify_roofline_v2 / audit_roofline_decisions scripts.
    last_roofline: dict[str, Any] = field(default_factory=dict)
    baseline_attempts: list[dict[str, Any]] = field(default_factory=list)
    profile_attempts: list[dict[str, Any]] = field(default_factory=list)
    gemm_tuning_attempts: list[dict[str, Any]] = field(default_factory=list)
    sweep_attempts: list[dict[str, Any]] = field(default_factory=list)
    # explore audit log. Capped per _DEFAULT_ATTEMPTS_HISTORY.
    explore_attempts: list[dict[str, Any]] = field(default_factory=list)
    # Roofline-v2 N10: per-tick roofline audit log (capped). Records the
    # snapshot_id / analysis_md_path each successful invocation produced.
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
    # unified explore ledger (KB_design §3.4 Inv-4.1 "single
    # ledger"). Persistent DFS state for the merged ``explore``
    # action. ``tested`` is keyed by canonical_fingerprint (content-based,
    # see ``action_executors._canonical_fingerprint``), same hashing as
    # ``variant_fingerprint`` so the ledgers migrate losslessly.
    #
    # Schema (M3, may grow in M5/M6 with specialist provenance):
    #
    #   {
    #     "schema_version": 1,
    #     "tested": {
    #       fingerprint: {name, extra_server_args, extra_envs, outcome,
    #                     round_id, ts, gain_pct, tput, provenance,
    #                     workload_signature}
    #     },
    #     "accepted": [
    #       {fingerprint, name, extra_server_args, extra_envs,
    #        gain_pct, stack_index, accepted_at_round, ts}
    #     ],
    #     "rejected": [
    #       {fingerprint, name, extra_server_args, extra_envs,
    #        reason, gain_pct, tput, round_id, ts}
    #     ],
    #     "winners_history": [
    #       {round_id, variant_name, fingerprint, gain_pct,
    #        extra_args, extra_envs}
    #     ],
    #     "discovered_flags": [
    #       {flag, source, first_seen_round}
    #     ],
    #     "synergy_attempted": [["name1", "name2"], ...],
    #     "domains_round_summary": [...],   # M5/M6 fills, M3 leaves []
    #     "name_index": {name: fingerprint},
    #     "cursor": int,
    #     "last_round": {...},
    #   }
    #
    # ``accepted`` outcomes include the inlined stack-rebench result —
    # entries that landed in optimization_stack and survived the
    # subsequent stack rebench. Items the rebench evicted live in
    # ``rejected`` with ``reason='stack_unstable'``.
    explore_search: dict[str, Any] = field(default_factory=dict)
    # specialist sub-agent rolling state (KB_design §3.5 +
    # §3.10 §4.1). Each entry summarises one EXPLORE round of specialist
    # dispatch (M5: 1 entry per round; M6 grows to N when 6 domains run
    # concurrently). Schema (per round):
    #
    #   {
    #     "round_id": str,                 # explore-round-N
    #     "dispatched_at": iso,
    #     "completed_at": iso,
    #     "domains": [domain_key, ...],
    #     "parallelism": int,              # actual concurrent specialists
    #     "tasks": [{task_id, domain, status, gap_canonical_id,
    #                turns_used, workspace, transcript_path, done_path,
    #                proposals_count, confidence}],
    #     "proposals_total": int,
    #     "proposals_kept": int,           # filled by Coordinator after
    #                                      # explore round finishes
    #     "proposals_rejected": int,
    #     "proposals_skipped": int,
    #     "confidence_avg": float | None,
    #     "domain_breakdown": {domain_key: {...}},
    #     "notes": [str, ...],
    #   }
    specialist_rounds: list[dict[str, Any]] = field(default_factory=list)
    # Per-domain "empty proposal_set" streak. Reset on a non-empty
    # specialist_done; Robustness reads this to escalate when a
    # specialist domain consistently fails to produce ideas
    #.
    specialist_domain_empty_streak: dict[str, int] = field(default_factory=dict)
    # IR-7 — session_steward_specialist (honest self-stop). The
    # Coordinator dispatches this domain on EXPLORE plateau and routes
    # the recommendation:
    #   * ``continue_explore``    — inject ``next_gap_canonical_id`` into
    #                               gaps[], reset plateau counters once;
    #                               sets ``steward_continuation_used``.
    #   * ``advance_to_kernel``   — set pending_escalate_hint='skip_to_kernel'.
    #   * ``stop_session``        — set_stop_reason('no_more_leverage').
    last_remaining_gaps_assessment: dict[str, Any] = field(default_factory=dict)
    remaining_gaps_assessments: list[dict[str, Any]] = field(default_factory=list)
    steward_continuation_used: bool = False
    # Per EXPLORE-round count of steward subprocess/transport failures
    # (used to mint retry idempotency keys; not a plateau signal).
    steward_infra_failures_by_round: dict[str, int] = field(
        default_factory=dict,
    )
    # last specialist task snapshot (parity with other
    # ``last_<action>`` mirrors; useful for the orchestration prompt
    # to surface "last round's specialist outcome").
    last_specialist: dict[str, Any] = field(default_factory=dict)
    # PR-A7 (Arbor-into-Hyperloom) — per-specialist patch verdict
    # ledger keyed by specialist task_id. The Critic role reviews a
    # specialist's worktree patches before ``integrate_patch`` runs;
    # values are entries from REVIEW_VERDICTS
    # (``approve`` / ``reject`` / ``needs_review`` / ``advise`` /
    # ``redirect``). PolicyGate's
    # ``integrate_patch_requires_critic_verdict`` rule denies an
    # integrate_patch delegate whose specialist_task_id is missing
    # from this map OR carries a ``reject`` verdict — so a hostile
    # / regressive patch never reaches the serving GPU. ``approve`` /
    # ``advise`` allow integrate; ``needs_review`` / ``redirect``
    # require explicit operator override via
    # ``params.bypass_critic=True``.
    specialist_patch_verdicts: dict[str, str] = field(default_factory=dict)
    # PR-A8 (Arbor-into-Hyperloom) — intervention-mix ledger.
    #
    # Each entry: ``{change_type, action, task_id, ts, delta_pct}``.
    # ``change_type`` ∈ {"config", "code_patch"}:
    #   * "config"     — env-var / CLI-flag tweaks via the merged
    #                    ``explore`` action (Arbor's "config agent").
    #   * "code_patch" — specialist-authored source patch promoted via
    #                    ``integrate_patch`` (Arbor's "code agent").
    # The Robustness role consumes this ledger to detect consecutive
    # config-only rounds and recommend escalating to a patch-authoring
    # specialist next round.
    intervention_mix: list[dict[str, Any]] = field(default_factory=list)
    # PR-A8 — counts the *current* run of contiguous KEEPs whose
    # change_type is ``config``. Resets to 0 every time a ``code_patch``
    # KEEP lands. Robustness reads this via the per-tick prompt.
    consecutive_config_only_rounds: int = 0
    # Research scout bookkeeping.
    #   * ``research_scout_enabled`` — master switch (``--no-research-scout``
    #     turns the whole feature off).
    #   * ``research_scout_runs`` — number of scout dispatches so far;
    #     feeds exploration-depth tracking.
    #   * ``research_scout_seen_pr_ids`` — PR ids already surfaced by the
    #     scout or the FRAMEWORK_PR phase, shared so the two never re-mine
    #     the same PR.
    research_scout_enabled: bool = True
    research_scout_interval: int = 3
    # Master switch for the advisory "External target gap" prompt block
    # (``--no-target-advisory`` turns it off). Advisory only — never gates
    # Objective / scoring.
    target_advisory_enabled: bool = True
    # Master switch for sedimenting KEEP/REVERT provenance (research-hint
    # source + measured gain) into the persistent recipe. When off the
    # recipe stays purely ephemeral (existing behaviour).
    recipe_sediment_enabled: bool = True
    research_scout_runs: int = 0
    research_scout_seen_pr_ids: list[str] = field(default_factory=list)
    # Round id at which the scout was last dispatched, so the K-round
    # re-dispatch fires at most once per qualifying round.
    research_scout_last_round: int = -1
    # Exploration-depth counters consumed by the depth gate. Counters are
    # deterministic (config/code patches, reverts); the id sets are
    # supplied by specialist research evidence. ``enabled`` mirrors the
    # ``--depth-gate`` switch. Stored as a dict so old state.json shapes
    # round-trip with a defaulted sub-structure.
    depth_tracker: dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "prs_fetched": [],
        "pr_diffs_read": [],
        "nvidia_refs_compared": [],
        "code_patches_attempted": 0,
        "config_changes_attempted": 0,
        "consecutive_reverts": 0,
    })
    # Optional CLI overrides for the depth-gate thresholds; unset keys
    # fall back to the phase_state defaults.
    depth_gate_thresholds: dict[str, int] = field(default_factory=dict)
    # PR-A8 — total specialist dispatches in the current EXPLORE entry.
    # Reset on phase transition into a fresh EXPLORE. Robustness's
    # storm detector fires when this crosses the configured cap
    # without a single non-empty proposal_set in the same window.
    explore_specialist_dispatched_count: int = 0
    # Aggregate view of dynamic_action dispatches keyed by ``dyn_id``;
    # Coordinator-only writer (``CORE_STATE_FIELDS`` blocks LLM
    # ``UPDATE_STATE``). Each value is the lifecycle summary dict
    # (status / last_outcome / cumulative_gain / …).
    dynamic_actions: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Per-EXPLORE-round counter of *successful* dynamic_action
    # dispatches (PolicyGate denials do not increment). Reset on
    # every fresh EXPLORE entry; read by
    # ``PolicyGate._validate_dynamic_action_dispatch`` for the
    # ``MAX_DYNAMIC_PER_ROUND`` cap; Coordinator is the sole writer.
    dynamic_action_round_count: int = 0
    # research_lane capacity locked at session start
    #. M5 default is 1 (single-specialist series);
    # M6 raises to 6 (concurrent). PolicyGate denies mid-session
    # mutation because it's listed in CORE_STATE_FIELDS.
    research_lane_capacity: int = 1
    # Separate GPU pool capacity for specialists that explicitly request
    # ``needs_gpu=true``. Zero disables GPU specialists. Locked at session
    # start so LLM roles cannot inflate GPU access mid-run.
    gpu_specialist_capacity: int = 0
    # escalate_strategy_change carry-over field
    #. The Coordinator's
    # ``_handle_escalate_strategy_change`` writes the validated
    # ``next_action_hint`` here so the phase_state machine can see it
    # on the next ``compute_next_phase`` pass; the field is cleared
    # by the Coordinator once the hint has been acted on (so the
    # phase machine doesn't re-trigger on the same hint).
    pending_escalate_hint: str = ""
    # last cleared escalate hint (audit only). Useful for
    # the breakdown to surface "we honored a llm_escalation here".
    last_consumed_escalate_hint: str = ""
    last_consumed_escalate_hint_ts: str = ""
    # per-phase plateau threshold overrides locked at
    # session start (CLI flags, KB_design §3.13 M7 §4). Empty dict
    # means "use library defaults"; phase_state reads these fields
    # for the dispatcher's phase-decision call.
    plateau_overrides: dict[str, Any] = field(default_factory=dict)
    # E2E integrate bookkeeping keyed by kernel_id + patch_path + args. This
    # prevents Orchestration from spending hours re-validating the same patch
    # after repeated NEEDS_REVIEW/REVERT outcomes.
    kernel_integrate_attempts: dict[str, Any] = field(default_factory=dict)
    rejected_kernel_patches: list[dict[str, Any]] = field(default_factory=list)
    # Kernel ids with no remaining automated path. This is fed by
    # run_optimization REVERTs and exhausted integrate attempts.
    rejected_kernel_ids: list[str] = field(default_factory=list)

    # Search-space expansion ledger surfaced in the Orchestration
    # prompt so the LLM sees the live framework's full flag namespace
    # rather than the shipped seed grid. Schema:
    # ``{framework: {"backend_flags": [...], "param_flags": [...],
    # "ts": iso, "source_path": str}}``.
    discovered_flags: dict[str, Any] = field(default_factory=dict)
    # Rolling per-action winners log (cap 20) used by IR-26 idea
    # generation. Schema: ``{action, round_id, base_tput,
    # winners: [{name, tput, gain_pct, extra_server_args, extra_envs}],
    # best: {...}, ts}``.
    backend_winners_history: list[dict[str, Any]] = field(default_factory=list)
    # Synergy combo keys (``"name1+name2+..."``) already tested this
    # session — prevents re-running the same combination after each
    # new explore round.
    synergy_attempted: list[str] = field(default_factory=list)

    # ---------------------------------------------------------------
    # the legacy ``action_scores`` decision system was retired.
    # The Coordinator no longer maintains a per-action numeric
    # priority; instead the Orchestration prompt surfaces facts
    # (phase / gaps / KB sub-graphs / specialist proposal_set) and
    # the LLM decides. Inv-9.1 forbids any system-side priority
    # value. Legacy fields:
    #
    # * ``action_scores``         — dropped from the dataclass; resume
    #                                migration logs + discards.
    # * ``params_no_promote_streak`` — kept as a *read-only* hint for
    #                                  M2 / legacy resume paths so
    #                                  ``phase_state.exit_normal_explore``
    #                                  fallback proxy keeps working;
    #                                  not written by Coordinator.
    #
    # Monotonic Coordinator tick counter. Bumped once per
    # ``Coordinator.run()`` / ``Coordinator.tick(n)`` iteration; kept
    # so plateau / phase budget math has a stable monotonic anchor.
    tick: int = 0
    # Remaining gain-pct target gap (0.0 means "no target"). Coordinator
    # refreshes this when the run objective is ``gain_pct=N``. Kept
    # because the prompt builder + breakdown rely on it for the
    # "Mission progress" line (NOT a priority — pure fact).
    target_gap_pct: float = 0.0

    # ------------------------------------------------------------------
    # Phase state machine fields
    # ------------------------------------------------------------------
    # ``phase`` is the run-level pipeline phase (PRELUDE / FRAMEWORK_PR /
    # EXPLORE / KERNEL / SWEEP / CLOSE). Coordinator is the only writer
    # (PolicyGate adds it to CORE_STATE_FIELDS); LLM agents can read
    # via prompt injection but cannot update_state. Empty string
    # signals "phase machine not yet initialised" — Coordinator
    # initialises on construction. legacy resume infers a value via
    # :func:`phase_state.infer_phase_from_state`.
    phase: str = ""
    # ISO UTC timestamp the current phase was entered. Used by
    # observability (breakdown.phase_segments) and the budget judge.
    phase_started_ts: str = ""
    # Monotonic-ish unix epoch matching ``phase_started_ts`` so the
    # phase budget judge can compute elapsed seconds without
    # re-parsing ISO strings every tick.
    phase_started_unix: float = 0.0
    # Append-only log of phase transitions; each row is built by
    # :func:`phase_state.make_history_row` and conforms to KB_design
    # §3.2 §6 (reason must be in ``PHASE_EXIT_REASONS``).  Capped at
    # ``_PHASE_HISTORY_CAP`` so a runaway transition never bloats
    # state.json (unlikely — at most ~6 phases in the chain — but defensive).
    phase_history: list[dict[str, Any]] = field(default_factory=list)
    # Wall-clock budget percentages per phase.
    # Coordinator populates from CLI flags / defaults at construction
    # time; persisted so resume picks up the exact split the original
    # run used. Empty dict means "use library defaults".
    phase_budget_pct: dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Cortex KB integration fields
    # ------------------------------------------------------------------
    # Coordinator-only writers. Listed in PolicyGate.CORE_STATE_FIELDS so
    # any LLM ``update_state`` intent that touches these is denied
    # (Inv-1 single writer). LLM consumers read them indirectly via
    # prompt injection.
    #
    # ``cortex_session_id`` is the hyperloom-local session identifier
    # carried into KB fact-write attrs (``source_session_id``) for
    # cross-session traceability. It is **not** a KB-side session id
    # (the KB session begin/commit protocol was retired); it now
    # defaults to ``session_dir.name`` when T0 mints it.
    cortex_session_id: str = ""
    # Retired: the KB ``session commit`` protocol was removed alongside
    # T2/T3. The field is kept (always ``{}``) for state.json resume
    # back-compat — the next ``state.save`` writes an empty dict and
    # the breakdown collector tolerates a missing summary. The
    # ``breakdown.kb_provenance.commit`` section is now derived from
    # ``drain_pending`` results instead.
    cortex_session_summary: dict[str, Any] = field(default_factory=dict)
    # T0 snapshot of ``find-recipe`` raw output (CLI ``--format text``,
    # one entry per recipe row). v0.8 M1 only **records** this — it is
    # not yet injected into the orchestration prompt; that happens in M5
    # specialist assembly. Kept as ``dict`` (parsed) so M5 can read
    # without re-parsing. Empty dict on first-ever session for a
    # (workload, hw) pair.
    warm_start_recipe: dict[str, Any] = field(default_factory=dict)
    # T0 snapshot of ``pitfalls`` output (negative priors from prior
    # REVERT / crash / OOM decisions on this (model, hardware),
    # optionally filtered by framework). List of KB point dicts
    # (each with ``{canonical_id, kind, attrs, confidence, ...}``),
    # mirroring ``warm_start_lessons``. Consumed by the specialist
    # prompt's "§ 5c. KNOWN PITFALLS" section.
    #
    # Schema change history: pre-fix this field held
    # ``[{"raw": <json_string>}]`` because the broken
    # ``traps(symptom=...)`` reader returned an opaque JSON blob.
    # Resume from such a snapshot is tolerated (the prompt section
    # filters out rows without ``attrs.description``).
    warm_start_pitfalls: list[dict[str, Any]] = field(default_factory=list)
    # T0 snapshot of ``lessons`` output (positive priors from prior
    # KEEPs on this (model, hardware), optionally filtered by
    # framework). Symmetric with ``warm_start_pitfalls``; consumed
    # by the specialist prompt's "§ 5b. RELATED LESSONS" section.
    # Empty when Cortex was bypassed (``--degraded-kb``) or T0
    # failed.
    warm_start_lessons: list[dict[str, Any]] = field(default_factory=list)
    # Iso UTC timestamp of the T0 snapshot. Empty when Cortex was
    # bypassed (``--degraded-kb``) or T0 failed.
    warm_start_ts: str = ""

    # ------------------------------------------------------------------
    # structured gaps ledger (KB_design §3.3 /
    # §3.5 / §3.9 §6). Replaces the proxy block (which derived
    # decision input from ``last_action_failures`` +
    # ``explore_search.winners_history``) with a structured, dedup'd
    # list of unresolved bottlenecks. Coordinator is the sole writer
    # (:meth:`Coordinator._refresh_gaps`); LLM agents read via
    # prompt injection. Listed in :data:`policy.CORE_STATE_FIELDS` so
    # any LLM ``update_state{changes={gaps: ...}}`` intent is denied.
    #
    # Refresh entry points:
    #   1. baseline completion           — initial gap extraction
    #   2. EXPLORE round KEEP/REVERT     — append to gap.attempts
    #   3. Cortex traverse(issue_node)   — merge cross-session priors
    #   4. specialist_done bookkeeping   — gap.attempts ← specialist
    #
    # Schema (per entry, KB_design §3.3 §4.2 / §3.5 §11):
    #
    #   {
    #     "canonical_id": str,        # issue_node canonical from Cortex
    #     "symptom":      str,        # human-readable: "MoE comm overhead"
    #     "layer":        str,        # comm / kernel / framework / param
    #     "severity":     str,        # high / medium / low
    #     "domain_hint":  str,        # which specialist domain best fits
    #     "source":       str,        # baseline / attempts / cortex
    #     "first_seen_ts":     iso,
    #     "last_updated_ts":   iso,
    #     "attempts": [
    #       {action, variant_name, outcome, gain_pct?, ts},
    #       ...
    #     ],
    #   }
    #
    # ``attempts`` is capped at the most recent 20 per gap; the whole
    # list is capped at :data:`_GAPS_MAX_ENTRIES` so a long session
    # can't blow up state.json. Dedup is keyed by ``canonical_id``.
    gaps: list[dict[str, Any]] = field(default_factory=list)

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
        # unified migration entry point. A v0.6 state.json
        # has no top-level ``schema_version`` field; treat the absence as
        # ``schema_version=1`` and run the §3.10 §5.2 field-mapping step.
        # The function is idempotent (Inv-10.3): re-loading a v0.8
        # state.json (``schema_version == LATEST_STATE_SCHEMA_VERSION``)
        # short-circuits the migration logging without touching the
        # fact-layer payload (Inv-10.1).
        incoming_version = int(raw.get("schema_version") or 1)
        needs_migration = incoming_version < LATEST_STATE_SCHEMA_VERSION
        migration_events: list[str] = []

        # Migrate the ``extra_sglang_args`` -> ``extra_server_args`` rename
        # (same for ``candidate_extra_sglang_args``). The on-disk shape
        # stays a plain dict, so the legacy key may appear in any of the
        # many nested ledgers (winners, baseline_artifacts, action_attempts,
        # explore_search, etc.). Walk the entire payload once at load time
        # and rewrite the legacy keys in place — the next save will then
        # emit canonical only and a future load of the same file is a no-op.
        legacy_migrations = _migrate_legacy_extra_sglang_args_keys(raw)
        if legacy_migrations:
            migration_events.append(
                f"extra_server_args rename: migrated {legacy_migrations} legacy "
                f"extra_sglang_args / candidate_extra_sglang_args key(s) "
                f"to extra_server_args / candidate_extra_server_args"
            )

        # Filter to known fields so older / newer state.json shapes don't
        # crash. Unknown keys are dropped; missing keys fall back to defaults.
        known = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in raw.items() if k in known}
        # drop the legacy scoreboard fields from the loaded
        # dict. The dataclass no longer carries
        # ``action_scores`` so it would be filtered out anyway; we
        # also strip ``params_no_promote_streak`` /
        # ``score_violation`` / ``cooldown_until_tick`` / ``streak_*``
        # / ``locked_reason`` family explicitly so the ``warn`` mode
        # gets a usable count. ``params_no_promote_streak`` is kept
        # as a read-only fallback for legacy M2 plateau proxy when
        # ``explore_search`` is empty; everything else is dropped.
        _legacy_drop_fields = (
            "action_scores",
            "score_violation",
            "cooldown_until_tick",
            "locked_reason",
            "ucb_bonus",
            "aging_bonus",
            "score_mult",
            "effective_score",
            "last_action_score_snapshot",
            # Removed in this branch: M4 renamed select_kernels →
            # trace_analyze; the legacy mirror field was dropped so all
            # readers use ``last_trace_analyze``. Resume of an older
            # state.json silently discards this slot.
            "last_select_kernels",
        )
        legacy_seen: list[tuple[str, int]] = []
        for legacy in _legacy_drop_fields:
            if legacy in raw:
                payload = raw.get(legacy)
                size = (
                    len(payload) if isinstance(payload, (dict, list, str))
                    else 1
                )
                legacy_seen.append((legacy, int(size)))
            filtered.pop(legacy, None)
        if legacy_seen:
            mode = os.environ.get(
                "INFERENCE_OPTIMIZER_LEGACY_ACTION_SCORES", "drop",
            ).strip().lower()
            import logging as _logging
            log = _logging.getLogger(__name__)
            summary = ", ".join(f"{k}={n}" for k, n in legacy_seen)
            migration_events.append(
                f"§3.9 dropped scoreboard fields ({summary})"
            )
            if mode == "warn":
                log.warning(
                    "v0.8 §3.9: dropped legacy scoreboard fields from "
                    "state.json (%s). set "
                    "--legacy-action-scores=drop to silence this.",
                    summary,
                )
            else:
                log.info(
                    "v0.8 §3.9: dropped legacy scoreboard fields from "
                    "state.json (%s).", summary,
                )
        # Normalize the unified ``explore_search`` ledger at the load
        # boundary so the executor and Coordinator paths can assume the
        # shaped schema and never need a fallback branch. Winners /
        # synergy history are folded in from the live history fields.
        filtered["explore_search"] = cls._build_explore_search(
            existing=filtered.get("explore_search"),
            backend_winners_history=filtered.get("backend_winners_history"),
            params_winner_history=filtered.get("params_winner_history"),
            synergy_attempted=filtered.get("synergy_attempted"),
        )

        # fact-layer integrity check. ``strict``
        # (the default) aborts when a fact-layer key was present in a
        # *non-empty* legacy state.json but couldn't be loaded into
        # the dataclass (caller dropped it / type mismatch). ``lenient``
        # downgrades to WARNING and continues. Fresh sessions
        # (``raw == {}``) skip the check entirely. Inv-10.1.
        if needs_migration and raw:
            mode = os.environ.get(
                "INFERENCE_OPTIMIZER_MIGRATION_MODE", "strict",
            ).strip().lower()
            fact_layer_keys = (
                "baseline_tput", "baseline_accuracy", "current_best",
                "cumulative_gain", "cumulative_gain_validated",
                "optimization_stack",
                # IR-7 additions (steward); safe to lose on v0.6 → v0.8
                # migration since the LLM treats missing assessment as
                # "no priors".
                "last_remaining_gaps_assessment",
                "remaining_gaps_assessments",
            )
            missing: list[str] = []
            for key in fact_layer_keys:
                if key in raw and key not in filtered:
                    missing.append(key)
            if missing:
                import logging as _logging
                log = _logging.getLogger(__name__)
                fmt = (
                    "v0.8 §3.10: fact-layer field(s) %s present in "
                    "state.json but not loaded into SharedState "
                    "(Inv-10.1 violation)."
                )
                if mode == "lenient":
                    log.warning(
                        fmt + " --migration-mode=lenient → continuing.",
                        ", ".join(missing),
                    )
                else:
                    log.error(fmt, ", ".join(missing))
                    raise ValueError(
                        f"v0.8 §3.10 strict migration failed: fact-layer "
                        f"field(s) {missing!r} lost. Re-run with "
                        f"--migration-mode=lenient to continue."
                    )

        # bump schema_version once migrations finish.
        # Idempotent: a payload already at the latest version
        # short-circuits the helper at the top of this function.
        filtered["schema_version"] = LATEST_STATE_SCHEMA_VERSION

        # operator-visible migration log. ``strict``
        # (default) is silent on info-level events but surfaces fatal
        # fact-layer errors elsewhere (e.g. CLI bootstrap); ``lenient``
        # downgrades any fact-layer error logged downstream. Here we
        # only emit the migration summary so resume traces are
        # self-describing.
        if needs_migration:
            import logging as _logging
            log = _logging.getLogger(__name__)
            event_str = "; ".join(migration_events) or "(no field changes)"
            log.info(
                "v0.8 §3.10: state.json migrated v%d → v%d. Events: %s",
                incoming_version, LATEST_STATE_SCHEMA_VERSION, event_str,
            )

        return cls(**filtered)

    @staticmethod
    def _build_explore_search(
        *,
        existing: Any,
        backend_winners_history: Any,
        params_winner_history: Any,
        synergy_attempted: Any,
    ) -> dict[str, Any]:
        """Shape the unified ``explore_search`` ledger at load time.

        The ExploreExecutor owns ``explore_search`` at runtime; this
        load-boundary normalizer only fills the defensive defaults and
        folds the live ``*_winner_history`` / ``synergy_attempted``
        history fields into ``winners_history`` / ``synergy_attempted``
        so a resume preserves the cross-round aggregation the prompt
        summary and plateau proxy read.
        """
        from .action_executors._grid_runner import variant_fingerprint as _fp

        existing = existing if isinstance(existing, dict) else {}
        out: dict[str, Any] = dict(existing)
        out.setdefault("schema_version", 1)
        out.setdefault("tested", {})
        out.setdefault("accepted", [])
        out.setdefault("rejected", [])
        out.setdefault("discovered_flags", [])
        out.setdefault("domains_round_summary", [])
        out.setdefault("name_index", {})
        out.setdefault("cursor", len(out.get("tested") or {}))
        out.setdefault("last_round", {})

        # winners_history: fold the live history fields + preserve any
        # prior explore-side rows, then sort by (round_id, ts).
        wh: list[dict[str, Any]] = []
        for source_list in (
            backend_winners_history,
            params_winner_history,
            existing.get("winners_history") or [],
        ):
            if not isinstance(source_list, list):
                continue
            for entry in source_list:
                if not isinstance(entry, dict):
                    continue
                fp_val = entry.get("fingerprint") or _fp(
                    str(entry.get("extra_server_args") or ""),
                    dict(entry.get("extra_envs") or {}),
                )
                wh.append({
                    "round_id": str(entry.get("round_id") or ""),
                    "variant_name": str(entry.get("variant_name")
                                          or entry.get("name") or ""),
                    "fingerprint": str(fp_val),
                    "gain_pct": entry.get("gain_pct"),
                    "extra_args": str(entry.get("extra_args")
                                       or entry.get("extra_server_args") or ""),
                    "extra_envs": dict(entry.get("extra_envs") or {}),
                    "provenance": str(entry.get("provenance") or ""),
                    "ts": str(entry.get("ts") or ""),
                })
        wh.sort(key=lambda r: (str(r.get("round_id") or ""), str(r.get("ts") or "")))
        out["winners_history"] = wh

        # synergy_attempted: fold the live field + preserve any
        # ExploreExecutor-side additions, deduped.
        sa_set: set[tuple[str, ...]] = set()

        def _normalize_combo(c: Any) -> tuple[str, ...] | None:
            if isinstance(c, list):
                items = tuple(sorted(str(x) for x in c if isinstance(x, str)))
                return items if items else None
            if isinstance(c, str) and c.strip():
                parts = tuple(sorted(p for p in c.split("+") if p))
                return parts if parts else None
            return None

        for source in (synergy_attempted, existing.get("synergy_attempted") or []):
            if not isinstance(source, list):
                continue
            for c in source:
                norm = _normalize_combo(c)
                if norm:
                    sa_set.add(norm)
        out["synergy_attempted"] = [list(c) for c in sorted(sa_set)]
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

    # ------------------------------------------------------------------
    # stop_reason ENUM validator
    # ------------------------------------------------------------------
    def set_stop_reason(
        self,
        value: str,
        *,
        strict: bool | None = None,
    ) -> str:
        """Validated writer for :attr:`stop_reason`.

        Inv-8.3 (stop_reason vocab closed): anything outside the
        ``STOP_REASON_VOCAB`` in ``phase_state`` is mapped to
        ``"unknown"`` in lenient mode (default) and emits a warning
        so the breakdown can surface the migration. ``strict=True``
        raises ``ValueError`` instead — used by the test path that
        wants fail-fast behaviour.

        ``strict`` defaults to the
        ``INFERENCE_OPTIMIZER_STRICT_STOP_REASON`` env var (``"1"`` /
        ``"true"`` enables strict). The CLI flips the env on for
        production runs once the vocab has been dogfooded.

        Returns the value actually written (normalised + clipped to
        the vocab).
        """
        from .phase_state import STOP_REASON_VOCAB, is_valid_stop_reason
        text = str(value or "").strip()
        if not text:
            self.stop_reason = ""
            return ""
        if is_valid_stop_reason(text):
            self.stop_reason = text
            return text
        if strict is None:
            strict_env = os.environ.get(
                "INFERENCE_OPTIMIZER_STRICT_STOP_REASON", "",
            ).strip().lower()
            strict = strict_env in ("1", "true", "yes")
        if strict:
            raise ValueError(
                f"stop_reason={text!r} not in STOP_REASON_VOCAB "
                f"({sorted(STOP_REASON_VOCAB)!r})"
            )
        # Lenient: map to "unknown" and surface a warning. Callers can
        # still observe the original via the warnings log.
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "stop_reason=%r not in STOP_REASON_VOCAB; mapped to 'unknown' "
            ". Set "
            "INFERENCE_OPTIMIZER_STRICT_STOP_REASON=1 to fail-fast.",
            text,
        )
        self.stop_reason = "unknown"
        return "unknown"

    # ------------------------------------------------------------------
    # IR-7 — session steward assessment writer
    # ------------------------------------------------------------------
    _STEWARD_ASSESSMENT_HISTORY_CAP = 10

    def record_steward_assessment(
        self,
        *,
        recommendation: str,
        next_gap_canonical_id: str,
        remaining_potential_pct_estimate: float,
        rationale: str,
        task_id: str,
        round_at_assessment: int,
        source_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Stash a session_steward_specialist verdict on SharedState.

        Returns the row that landed in ``last_remaining_gaps_assessment``
        so the Coordinator can route on it immediately without
        re-reading state. History capped at
        :attr:`_STEWARD_ASSESSMENT_HISTORY_CAP` (drops oldest).
        """
        row = {
            "ts": _now_iso(),
            "recommendation": str(recommendation),
            "next_gap_canonical_id": str(next_gap_canonical_id or ""),
            "remaining_potential_pct_estimate": float(
                remaining_potential_pct_estimate or 0.0
            ),
            "rationale": str(rationale or "")[:2000],
            "task_id": str(task_id or ""),
            "round_at_assessment": int(round_at_assessment or 0),
        }
        if source_payload is not None:
            # Preserve the full payload (truncated to a manageable size)
            # so the breakdown collector can dig deeper if needed.
            row["source_payload_keys"] = sorted(source_payload.keys())
        self.last_remaining_gaps_assessment = row
        history = list(self.remaining_gaps_assessments or [])
        history.append(row)
        if len(history) > self._STEWARD_ASSESSMENT_HISTORY_CAP:
            history = history[-self._STEWARD_ASSESSMENT_HISTORY_CAP:]
        self.remaining_gaps_assessments = history
        return row

    # ------------------------------------------------------------------
    # escalate hint plumbing
    # ------------------------------------------------------------------
    def set_pending_escalate_hint(self, hint: str) -> str:
        """Stash the LLM-supplied hint for the next phase compute pass.

        Returns the value actually written. Unknown hints are
        silently dropped (Inv-8.2 — phase decisions only react to a
        closed vocabulary; arbitrary robustness payloads should not
        steer the state machine).
        """
        from .phase_state import is_valid_escalate_hint
        text = str(hint or "").strip()
        if text and not is_valid_escalate_hint(text):
            return ""
        self.pending_escalate_hint = text
        return text

    def consume_pending_escalate_hint(self) -> str:
        """Pop the pending hint, recording the consumption in audit fields.

        The phase machine calls this *after* it acted on the hint so
        the next tick doesn't re-trigger the same transition. Returns
        the hint that was cleared (empty string when nothing was
        pending).
        """
        hint = (self.pending_escalate_hint or "").strip()
        if not hint:
            return ""
        self.pending_escalate_hint = ""
        self.last_consumed_escalate_hint = hint
        self.last_consumed_escalate_hint_ts = _now_iso()
        return hint

    # ------------------------------------------------------------------
    # phase machine writer (Coordinator-only, Inv-1 + Inv-8.1)
    # ------------------------------------------------------------------
    def record_phase_transition(
        self,
        *,
        to_phase: str,
        reason: str,
        evidence: dict[str, Any] | None = None,
        ts: str | None = None,
        ts_unix: float | None = None,
    ) -> dict[str, Any]:
        """Append a phase_history row and atomically update ``phase`` fields.

        The Coordinator calls this from ``_advance_phase_if_needed`` at
        the end of each tick (also once during ``__init__`` for the
        initial "phase_entered" row). LLM roles cannot reach this code
        path — PolicyGate adds ``phase`` / ``phase_history`` to
        :data:`policy.CORE_STATE_FIELDS`, so any ``update_state`` intent
        touching them is rejected first.

        Returns the inserted row (so callers don't have to re-read the
        list to log it).
        """
        from datetime import datetime as _dt, timezone as _tz
        import time as _time
        # Lazy import to avoid an import-time cycle with the orchestrator
        # package (phase_state itself imports nothing from SharedState).
        from .phase_state import make_history_row

        now_ts = ts or _dt.now(_tz.utc).isoformat(timespec="seconds")
        now_unix = float(ts_unix if ts_unix is not None else _time.time())
        row = make_history_row(
            from_phase=self.phase or "",
            to_phase=to_phase,
            reason=reason,
            evidence=evidence,
            ts=now_ts,
            ts_unix=now_unix,
        )
        history = list(self.phase_history or [])
        history.append(row)
        if len(history) > _PHASE_HISTORY_CAP:
            history = history[-_PHASE_HISTORY_CAP:]
        self.phase_history = history
        self.phase = row["to_phase"]
        self.phase_started_ts = now_ts
        self.phase_started_unix = now_unix
        return row

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

    def increment_crash_count(self, by: int = 1) -> int:
        self.crash_count += by
        return self.crash_count

    def record_tick_exception(
        self,
        *,
        tick: int,
        stage: str,
        exc_type: str,
        message: str,
        traceback_text: str,
        agent: str = "",
    ) -> dict[str, Any]:
        """Persist a compact Coordinator exception summary for postmortems."""
        entry = {
            "tick": int(tick or 0),
            "ts": _now_iso(),
            "stage": str(stage or ""),
            "agent": str(agent or ""),
            "type": str(exc_type or ""),
            "message": str(message or "")[:1000],
            "traceback": str(traceback_text or "")[:12000],
        }
        self.last_tick_exception = entry
        return entry

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
        # ``payload`` is an external envelope (LLM intent or sub-agent
        # kernel_opt result); route through the compat helper so a legacy
        # ``extra_sglang_args`` key still resolves with a single
        # DeprecationWarning logged via stacklevel=3.
        from ..compat.payload_aliases import read_extra_server_args
        extra_args = read_extra_server_args(payload).strip()
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
            "extra_server_args": extra_args,
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
            "extra_server_args": extra_args,
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
        ray_transient = is_ray_transient_kernel_opt_failure(result)
        # PR-C: An "infra failure" is a backend ladder that finished
        # WITHOUT delivering any verdict at all -- subprocess timeout,
        # batch handler exception, GEAK/OOB rc!=0, missing inputs, etc.
        # These are distinct from REVERT (which has its own
        # ``should_reject`` rule) and PARTIAL (which has its own
        # ``max_partial`` streak gate); we don't double-count them.
        # Ray/raylet death is NOT a permanent infra failure — it is retried
        # after Ray bootstrap (see ``clear_ray_transient_kernel_rejections``).
        is_infra_failure = (
            not ray_transient
            and decision == ""
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
            "ray_transient": ray_transient,
        })
        history = history[-10:]
        if not ray_transient:
            entry["attempts"] = int(entry.get("attempts", 0)) + 1
        per_source = dict(entry.get("attempts_per_source") or {})
        src_key = source_file or ""
        if not ray_transient:
            per_source[src_key] = int(per_source.get(src_key, 0)) + 1
            entry["attempts_per_source"] = per_source
        elif per_source:
            entry["attempts_per_source"] = per_source
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
        if ray_transient:
            entry["ray_transient_failures"] = (
                int(entry.get("ray_transient_failures", 0)) + 1
            )
            entry["last_ray_transient"] = True
            entry["failure_count"] = 0
            if kernel_id in self.rejected_kernel_ids:
                self.rejected_kernel_ids = [
                    k for k in (self.rejected_kernel_ids or []) if k != kernel_id
                ]
            entry.pop("rejected_reason", None)
        else:
            entry["last_ray_transient"] = False
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

        max_ray_retries = _default_kernel_opt_max_ray_retries()
        ray_retries_exhausted = (
            int(entry.get("ray_transient_failures", 0)) >= max_ray_retries
        )
        # Dispatch-only REVERT (micro=0, no artifact) is infra — never terminal
        # even when ray retry budget is exhausted (run13 k001 x3).
        ray_dispatch_only = (
            decision == "REVERT"
            and ray_transient
            and micro_float <= 0.0
            and not best_artifact_path
        )
        ray_only_revert = (
            decision == "REVERT"
            and ray_transient
            and (not ray_retries_exhausted or ray_dispatch_only)
        )

        should_reject = (
            (decision == "REVERT" and not ray_only_revert)
            or int(entry.get("partial_count", 0)) >= max_partial
            or (
                int(entry.get("failure_count", 0)) >= max_failures
                and not ray_transient
            )
            or (
                ray_transient
                and ray_retries_exhausted
                and decision == ""
                and status in {"failed", "error", "timeout"}
            )
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

    def record_gemm_tuning(self, result: dict[str, Any]) -> None:
        """Capture the GEAK GEMM tuning result for sequencing and prompts."""
        if not isinstance(result, dict):
            result = {"status": "failed", "error": "non-dict gemm tuning result"}
        entry = dict(result)
        entry.setdefault("ts", _now_iso())
        self.last_gemm_tuning = entry
        attempts = list(self.gemm_tuning_attempts or [])
        attempts.append(entry)
        self.gemm_tuning_attempts = attempts[-_DEFAULT_ATTEMPTS_HISTORY:]

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

    def kernel_integrate_keep_satisfied(self) -> bool:
        """True when at least one kernel integrate KEEP is on record.

        Satisfied when an integrate entry landed on ``optimization_stack``
        or ``kernel_integrate_attempts`` records a KEEP decision.
        """
        if self._kernel_ids_in_optimization_stack():
            return True
        for entry in (self.kernel_integrate_attempts or {}).values():
            if not isinstance(entry, dict):
                continue
            if str(entry.get("last_decision") or "").upper() == "KEEP":
                return True
        return False

    def kernel_integrate_validated_gain_pct(self) -> float:
        """Sum validated gain from integrate entries on ``optimization_stack``."""
        total = 0.0
        for entry in (self.optimization_stack or []):
            if not isinstance(entry, dict):
                continue
            if str(entry.get("action") or "") != "integrate":
                continue
            try:
                total += float(entry.get("gain_pct") or 0.0)
            except (TypeError, ValueError):
                continue
        return total

    def kernel_opt_has_keep_decision(self) -> bool:
        """True when any ``kernel_opt_attempts`` row ended KEEP/DEFER_TO_E2E."""
        for entry in (self.kernel_opt_attempts or {}).values():
            if not isinstance(entry, dict):
                continue
            decision = str(entry.get("last_decision") or "").upper()
            if decision in ("KEEP", "DEFER_TO_E2E"):
                return True
        return False

    def _kernel_entry_ray_transient_only(self, kernel_id: str) -> bool:
        """True when ``kernel_id`` was retired only due to Ray instability."""
        entry = (self.kernel_opt_attempts or {}).get(kernel_id) or {}
        if not isinstance(entry, dict):
            return False
        if entry.get("last_ray_transient"):
            return True
        reason = str(entry.get("rejected_reason") or "")
        if reason.startswith("max_failures_") and int(entry.get("ray_transient_failures") or 0) > 0:
            return True
        if reason == "revert_decision" and int(entry.get("ray_transient_failures") or 0) > 0:
            return True
        return False

    def clear_ray_transient_kernel_rejections(self) -> list[str]:
        """Unprune kernels rejected only for Ray when Ray is healthy again."""
        cleared: list[str] = []
        for kid in list(self.rejected_kernel_ids or []):
            if not self._kernel_entry_ray_transient_only(kid):
                continue
            self.rejected_kernel_ids = [
                k for k in (self.rejected_kernel_ids or []) if k != kid
            ]
            entry = dict(self.kernel_opt_attempts.get(kid) or {})
            entry["failure_count"] = 0
            entry["ray_transient_failures"] = 0
            entry["last_ray_transient"] = False
            entry.pop("rejected_reason", None)
            self.kernel_opt_attempts[kid] = entry
            cleared.append(kid)
        return cleared

    def reset_ray_transient_kernel_counters(self) -> list[str]:
        """Zero ray-transient streaks after a healthy Ray bootstrap."""
        reset: list[str] = []
        for kid, raw in list((self.kernel_opt_attempts or {}).items()):
            if not isinstance(raw, dict):
                continue
            if int(raw.get("ray_transient_failures") or 0) <= 0:
                continue
            entry = dict(raw)
            entry["ray_transient_failures"] = 0
            entry["last_ray_transient"] = False
            entry.pop("rejected_reason", None)
            self.kernel_opt_attempts[kid] = entry
            if kid in (self.rejected_kernel_ids or []):
                self.rejected_kernel_ids = [
                    k for k in (self.rejected_kernel_ids or []) if k != kid
                ]
            reset.append(kid)
        return reset

    def kernel_opt_owes_ray_recovery(self) -> bool:
        """True when any hot kernel has pending Ray recovery retries."""
        max_ray = _default_kernel_opt_max_ray_retries()
        for entry in (self.kernel_opt_attempts or {}).values():
            if not isinstance(entry, dict):
                continue
            if int(entry.get("ray_transient_failures") or 0) >= max_ray:
                continue
            if entry.get("last_ray_transient") or int(
                entry.get("ray_transient_failures") or 0
            ) > 0:
                return True
        return self.kernel_phase_blocked_on_ray_unresolved(ray_healthy=False)

    def kernel_phase_blocked_on_ray_unresolved(self, *, ray_healthy: bool) -> bool:
        """Hold KERNEL→SWEEP when hot kernels owe Ray recovery retries.

        Returns True when a top hot reusable kernel was retired only due to
        Ray/raylet failure and either Ray retries remain or Ray is still
        unhealthy after bootstrap.
        """
        if self.kernel_integrate_keep_satisfied():
            return False
        max_ray = _default_kernel_opt_max_ray_retries()
        info = self.last_trace_analyze or {}
        hot = info.get("hot_kernels_top15") or info.get("hot_kernels") or []
        if not isinstance(hot, list):
            return False
        rejected = set(self.rejected_kernel_ids or [])
        for row in hot:
            if not isinstance(row, dict):
                continue
            if row.get("reusable_native_kernel") is not True:
                continue
            kid = str(row.get("kernel_id") or "")
            if not kid or kid not in rejected:
                continue
            if not self._kernel_entry_ray_transient_only(kid):
                continue
            entry = (self.kernel_opt_attempts or {}).get(kid) or {}
            ray_count = int(entry.get("ray_transient_failures") or 0)
            if ray_count < max_ray:
                return True
            if not ray_healthy:
                return True
        return False

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

    def record_trace_analyze(
        self,
        payload: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        """write the canonical 11-field ``last_trace_analyze`` dict.

        Called by :class:`RooflineExecutor` (F1-2) after a successful
        ``trace_analyze`` sub-step, and by the inline programmatic
        handler path in :meth:`Coordinator._handle_request` when an LLM
        emits a ``trace_analyze`` request directly. Single canonical
        writer for this cache — the M4 legacy ``record_select_kernels``
        twin was removed in this branch.

        On every successful call, ``roofline_snapshot_id`` is read from the
        previous ``last_trace_analyze`` and incremented by one — giving a
        monotonic counter the orchestration prompt + the watermark-driven
        freshness gate (:meth:`Coordinator._needs_roofline_for_watermark`)
        both rely on. ``roofline_baseline_gain_at_snapshot`` captures
        ``cumulative_gain_validated`` at write time so the prompt can
        surface the gain delta since the report was taken.

        A compact copy is also appended to
        :attr:`roofline_snapshots` (PR #321 retired the legacy
        ``last_trace_analyze_baseline`` field; the history list is the
        new source for ``report.py``'s ``## Roofline Comparison``
        before/after view).

        ``analysis_md_text`` is read verbatim from
        ``result['trace_report_path']``; OSErrors degrade silently to
        empty text (the ``analysis_md_path`` field still lets a future
        ``read_artifact`` intent re-fetch it on demand).
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
        kernel_roofline_path = result.get("kernel_roofline_path") or ""
        if not kernel_roofline_path:
            artifacts = result.get("artifact_paths") or {}
            if isinstance(artifacts, dict):
                kernel_roofline_path = artifacts.get("kernel_roofline", "") or ""
        hot = result.get("hot_kernels") or []
        summary: list[dict[str, Any]] = []
        kernel_roofline: list[dict[str, Any]] = []
        reusable_ids: list[str] = []
        for entry in hot[:15] if isinstance(hot, list) else []:
            if not isinstance(entry, dict):
                continue
            kid = entry.get("kernel_id")
            reusable = bool(entry.get("reusable_native_kernel"))
            arithmetic_intensity = entry.get("arithmetic_intensity")
            if arithmetic_intensity is None:
                arithmetic_intensity = entry.get("flops_per_byte")
            efficiency_percent = entry.get("efficiency_percent")
            if efficiency_percent is None:
                efficiency_percent = entry.get("efficiency_pct")
            summary_entry = {
                "kernel_id": kid,
                "name": entry.get("name"),
                # TraceLens derive_kernel_category bucket (MoE / LayerNorm /
                # GEMM / ...). Without this passthrough downstream consumers
                # (kernel_attempt_summary by_kernel rows) get an empty string.
                "kernel_category": entry.get("kernel_category") or "",
                "gpu_pct": entry.get("gpu_pct"),
                "bottleneck": entry.get("bottleneck"),
                "bound_type": entry.get("bound_type"),
                "arithmetic_intensity": arithmetic_intensity,
                "flops_per_byte": entry.get("flops_per_byte"),
                "efficiency_percent": efficiency_percent,
                "compute_utilization_pct": entry.get("compute_utilization_pct"),
                "bandwidth_utilization_pct": entry.get("bandwidth_utilization_pct"),
                "suggestion": entry.get("suggestion") or "",
                "roofline_name": entry.get("roofline_name"),
                "source_file": entry.get("source_file"),
                "reusable_native_kernel": reusable,
                "recommended_backends": entry.get("recommended_backends") or [],
                "recommended_actions": entry.get("recommended_actions") or [],
            }
            summary.append(summary_entry)
            if any(
                summary_entry.get(key) not in (None, "", [])
                for key in (
                    "bound_type",
                    "arithmetic_intensity",
                    "flops_per_byte",
                    "efficiency_percent",
                    "compute_utilization_pct",
                    "bandwidth_utilization_pct",
                    "suggestion",
                    "roofline_name",
                )
            ):
                kernel_roofline.append(dict(summary_entry))
            if reusable and kid:
                reusable_ids.append(str(kid))

        # Project the skipped (non-routable) candidates so the prompt can
        # show the LLM that these operators were *seen* but cannot be
        # optimized (e.g. "source file not resolved"). Without this the
        # structured block renders an empty candidate list whenever
        # hot_kernels is empty, leaving analysis.md's operator names as the
        # only kernel identifiers in the prompt -- which the LLM then echoes
        # as a hallucinated kernel_id (operator names are also non-unique,
        # e.g. several k00x all named ``aten::mm``). Surfacing (id, name,
        # reason) lets the LLM avoid re-requesting them by id.
        skipped = result.get("skipped_kernels") or []
        skipped_summary: list[dict[str, Any]] = []
        if isinstance(skipped, list):
            skipped_sorted = sorted(
                (e for e in skipped if isinstance(e, dict)),
                key=lambda e: float(e.get("gpu_pct") or 0.0),
                reverse=True,
            )
            for entry in skipped_sorted[:15]:
                skipped_summary.append({
                    "kernel_id": entry.get("kernel_id"),
                    "name": entry.get("name"),
                    "skip_reason": entry.get("skip_reason") or "",
                    "gpu_pct": entry.get("gpu_pct"),
                })

        raw_warnings = result.get("trace_health_warnings") or []
        warnings_cleaned: list[dict[str, Any]] = []
        if isinstance(raw_warnings, list):
            for entry in raw_warnings:
                if isinstance(entry, dict) and entry.get("code"):
                    warnings_cleaned.append(dict(entry))

        # Monotonic snapshot counter: read previous value + 1.
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
                # Stored verbatim, but the prompt path strips embedded
                # base64 data-URLs before injection (see
                # ``_format_analysis_md_full`` /
                # ``strip_base64_data_urls``). A raw analysis.md can be
                # ~120K tokens of base64 images; post-strip it is ~6K
                # tokens of actual report text. NEVER inject this field
                # raw — always go through the strip helper.
                analysis_md_text = Path(analysis_md_path).read_text(
                    encoding="utf-8", errors="replace",
                )
            except (OSError, ValueError):
                analysis_md_text = ""

        task_groups = result.get("task_groups") or []
        if not isinstance(task_groups, list):
            task_groups = []

        ts_iso = _now_iso()
        self.last_trace_analyze = {
            "trace_input": str(trace_input),
            "candidates_path": str(candidates_path),
            "kernel_roofline_path": str(kernel_roofline_path),
            "hot_kernels_top15": summary,
            "kernel_roofline_top15": kernel_roofline,
            "skipped_kernels_top": skipped_summary,
            "task_groups": task_groups,
            "reusable_native_kernel_ids": reusable_ids,
            "trace_health_warnings": warnings_cleaned,
            "analysis_md_path": str(analysis_md_path),
            "analysis_md_text": analysis_md_text,
            "roofline_snapshot_id": snapshot_id,
            "roofline_baseline_gain_at_snapshot": float(
                self.cumulative_gain_validated,
            ),
            "ts": ts_iso,
        }
        # Mirror the snapshot id at the top level so PolicyGate /
        # Coordinator can read it without the nested-dict lookup.
        self.roofline_snapshot_id = snapshot_id

        # Append a compact history entry for the report-side Roofline
        # Comparison renderer. Parses analysis.md once at write time so
        # the renderer never has to (and the snapshot survives even if
        # the on-disk file is later deleted). Best-effort: parsing
        # errors degrade silently to None fields.
        try:
            from .roofline_snapshot import build_roofline_snapshot
            # Stamp decode-roofline ceiling + measured throughput so
            # the dashboard can surface "% within roofline" without
            # re-reading model files. Ceiling is a session-level
            # constant (hardware + model + isl/osl don't change),
            # achieved tput is current_best.tput if optimization has
            # produced a winner, else baseline_tput.
            from .roofline_ceiling import (
                RooflineBreakdown,
                compute_roofline_breakdown_from_state,
            )
            # Two-sided roofline (T_mem + T_cmp + min) — see formula
            # change in roofline_ceiling.py. ``peak_tput`` continues to
            # equal ``min(mem, cmp)`` so the existing dashboard
            # ``theoretical_peak_tok_per_sec`` field stays meaningful.
            breakdown = RooflineBreakdown(0.0, 0.0, 0.0, "unknown")
            try:
                breakdown = compute_roofline_breakdown_from_state(self)
            except Exception:  # noqa: BLE001 — ceiling is best-effort
                pass
            peak_tput = float(breakdown.peak_tok_per_sec or 0.0)
            achieved_tput = 0.0
            cb = self.current_best if isinstance(self.current_best, dict) else {}
            cb_tput = cb.get("tput")
            if isinstance(cb_tput, (int, float)) and cb_tput > 0:
                achieved_tput = float(cb_tput)
            elif isinstance(self.baseline_tput, (int, float)) and self.baseline_tput > 0:
                achieved_tput = float(self.baseline_tput)
            history_entry = build_roofline_snapshot(
                snapshot_id=snapshot_id,
                ts=ts_iso,
                analysis_md_path=str(analysis_md_path),
                theoretical_peak_tok_per_sec=peak_tput,
                achieved_tok_per_sec=achieved_tput,
                mem_ceiling_tok_per_sec=float(breakdown.mem_tok_per_sec or 0.0),
                cmp_ceiling_tok_per_sec=float(breakdown.cmp_tok_per_sec or 0.0),
                bound_kind=breakdown.bound_kind,
            )
            history_entry["trace_input"] = str(trace_input)
            history_entry["analysis_md_path"] = str(analysis_md_path)
            # 9fe4609 sidecar artifact pointer: dashboards read this
            # path to surface per-kernel roofline data from the
            # tracelens-written ``reports/kernel_roofline.json``.
            history_entry["kernel_roofline_path"] = str(kernel_roofline_path)
            if not isinstance(self.roofline_snapshots, list):
                self.roofline_snapshots = []
            self.roofline_snapshots.append(history_entry)
            if len(self.roofline_snapshots) > _ROOFLINE_SNAPSHOTS_CAP:
                # Drop oldest non-baseline entries; always keep
                # snapshot #1 so the report's "baseline" anchor never
                # rotates away.
                base = self.roofline_snapshots[0]
                tail = self.roofline_snapshots[-(_ROOFLINE_SNAPSHOTS_CAP - 1):]
                self.roofline_snapshots = [base, *tail]
        except Exception:  # noqa: BLE001 — never block record on render concerns
            pass

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

    def record_conc_sweep(self, result: dict[str, Any]) -> None:
        """Record conc_sweep task completion (mirrors record_sweep).

        Bug #12 fix: ``phase_state.exit_normal_sweep`` previously only
        looked at ``last_sweep.status`` to fire ``sweep_done``. When the
        SWEEP phase reaches its terminal state via conc_sweep alone (e.g.
        sweep was singleton-blocked or completed in a prior session and
        only conc_sweep runs in this resume), the phase had no exit
        signal and idled until ``sweep_budget_exhausted``. Writing
        ``last_conc_sweep.status`` lets ``exit_normal_sweep`` return
        ``conc_sweep_done`` and the SWEEP→CLOSE transition fires
        immediately after conc_sweep settles.
        """
        if not isinstance(result, dict):
            return
        self.last_conc_sweep = {
            "ts":               _now_iso(),
            "status":           str(result.get("status") or "succeeded"),
            "skip_reason":      str(result.get("skip_reason") or ""),
            "was_skipped":      bool(result.get("was_skipped", False)),
            "budget_exhausted": bool(result.get("budget_exhausted", False)),
            "summary":          dict(result.get("summary") or {}),
            "workspace":        str(result.get("workspace") or ""),
        }

    # ------------------------------------------------------------------
    # specialist round bookkeeping
    # ------------------------------------------------------------------
    def record_specialist_round(self, entry: dict[str, Any]) -> None:
        """Append one round summary to ``specialist_rounds``.

        Coordinator calls this once all dispatched specialists in a
        round have terminated (either via specialist_done or empty
        synthesise). Idempotent on ``round_id``: a re-record with the
        same id overwrites the latest entry rather than duplicating.
        """
        if not isinstance(entry, dict) or not entry:
            return
        round_id = str(entry.get("round_id") or "").strip()
        if not round_id:
            self.specialist_rounds.append(dict(entry))
            return
        existing = self.specialist_rounds
        for i, prev in enumerate(existing):
            if isinstance(prev, dict) and str(prev.get("round_id") or "") == round_id:
                existing[i] = dict(entry)
                return
        existing.append(dict(entry))

    def bump_specialist_domain_empty_streak(
        self, domain: str, *, empty: bool,
    ) -> int:
        """Increment / reset the per-domain empty-proposal streak.

        Returns the new streak value (0 when reset). The streak
        threshold for escalation is read by Robustness from
        ``KB_design §3.9`` and is not encoded here.
        """
        d = str(domain or "").strip() or "unknown"
        if empty:
            self.specialist_domain_empty_streak[d] = int(
                self.specialist_domain_empty_streak.get(d, 0) or 0
            ) + 1
        else:
            self.specialist_domain_empty_streak[d] = 0
        return self.specialist_domain_empty_streak[d]

    # ------------------------------------------------------------------
    # gaps ledger helpers
    # ------------------------------------------------------------------
    def find_gap(self, canonical_id: str) -> dict[str, Any] | None:
        """Return the gap entry matching ``canonical_id`` (or ``None``).

        Coordinator's :meth:`_warm_specialist_params` calls this to
        attach ``gap_symptom`` / ``gap_layer`` / ``gap_attempts`` onto
        a specialist task before dispatch.
        """
        if not canonical_id:
            return None
        cid = str(canonical_id)
        for gap in self.gaps:
            if isinstance(gap, dict) and str(gap.get("canonical_id") or "") == cid:
                return gap
        return None

    def upsert_gap(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Insert or update one gap row, keyed by ``canonical_id``.

        Merges ``attempts`` (existing + incoming, capped at the most
        recent :data:`_GAPS_ATTEMPTS_HISTORY` rows by ``ts``). Updates
        ``last_updated_ts`` to ``now``; preserves the original
        ``first_seen_ts`` when present.

        Coordinator-only writer (Inv-1 single-writer + PolicyGate
        ``CORE_STATE_FIELDS`` lock). Returns the merged entry so the
        caller can keep working with the canonical row.
        """
        if not isinstance(entry, dict):
            return {}
        cid = str(entry.get("canonical_id") or "").strip()
        if not cid:
            return {}
        now = _now_iso()
        existing = self.find_gap(cid)
        if existing is None:
            merged: dict[str, Any] = {
                "canonical_id":    cid,
                "symptom":         str(entry.get("symptom") or ""),
                "layer":           str(entry.get("layer") or ""),
                "severity":        str(entry.get("severity") or "medium"),
                "domain_hint":     str(entry.get("domain_hint") or ""),
                "source":          str(entry.get("source") or ""),
                # Origin reference (e.g. a research-hint PR/blog URL) that
                # outlives the session so KEEP/REVERT outcomes can be
                # sedimented into the recipe with provenance. Optional;
                # absent on gaps without an external origin.
                "provenance":      str(entry.get("provenance") or ""),
                "first_seen_ts":   str(entry.get("first_seen_ts") or now),
                "last_updated_ts": now,
                "attempts":        list(entry.get("attempts") or []),
            }
            # Trim attempts to the cap to be safe.
            if len(merged["attempts"]) > _GAPS_ATTEMPTS_HISTORY:
                merged["attempts"] = merged["attempts"][-_GAPS_ATTEMPTS_HISTORY:]
            self.gaps.append(merged)
        else:
            # Field-wise merge: incoming non-empty values win except
            # for ``first_seen_ts`` (preserve oldest).
            for key in ("symptom", "layer", "severity", "domain_hint",
                        "source", "provenance"):
                incoming = entry.get(key)
                if incoming:
                    existing[key] = str(incoming)
            existing.setdefault("first_seen_ts", str(entry.get("first_seen_ts") or now))
            existing["last_updated_ts"] = now
            incoming_attempts = list(entry.get("attempts") or [])
            if incoming_attempts:
                merged_attempts = list(existing.get("attempts") or []) + incoming_attempts
                # Capped tail; the cap dominates ordering — callers
                # supply newest-last lists which is the convention used
                # everywhere else (e.g. ``last_action_failures``).
                if len(merged_attempts) > _GAPS_ATTEMPTS_HISTORY:
                    merged_attempts = merged_attempts[-_GAPS_ATTEMPTS_HISTORY:]
                existing["attempts"] = merged_attempts
            merged = existing
        # Enforce the global cap. We trim oldest (by last_updated_ts when
        # available, falling back to insertion order). Trimming runs after
        # the upsert so the just-touched gap is always retained.
        if len(self.gaps) > _GAPS_MAX_ENTRIES:
            keep_cid = cid
            others = [g for g in self.gaps if g is not merged]

            def _sort_key(g: dict[str, Any]) -> str:
                return str(g.get("last_updated_ts") or g.get("first_seen_ts") or "")

            others.sort(key=_sort_key)
            keep_count = _GAPS_MAX_ENTRIES - 1
            others = others[-keep_count:] if keep_count > 0 else []
            self.gaps = others + [merged]
            del keep_cid  # silence linters when the local isn't used elsewhere
        return merged

    def append_gap_attempt(
        self,
        canonical_id: str,
        attempt: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Append one attempt row to an existing gap.

        Returns the updated gap entry or ``None`` when the gap is not
        known yet (caller may decide to ``upsert_gap`` with a freshly
        synthesised symptom in that case).
        """
        gap = self.find_gap(canonical_id)
        if gap is None:
            return None
        attempts = list(gap.get("attempts") or [])
        attempts.append(dict(attempt) | {"ts": str(attempt.get("ts") or _now_iso())})
        if len(attempts) > _GAPS_ATTEMPTS_HISTORY:
            attempts = attempts[-_GAPS_ATTEMPTS_HISTORY:]
        gap["attempts"] = attempts
        gap["last_updated_ts"] = _now_iso()
        return gap

    def replace_gaps(self, entries: list[dict[str, Any]]) -> None:
        """Bulk-replace ``gaps`` with a fresh dedup'd list.

        Used by :meth:`Coordinator._refresh_gaps` when the rebuild
        wants to discard stale rows wholesale (e.g. after a Cortex
        traverse returned a new canonical set). Idempotent.
        """
        if not isinstance(entries, list):
            return
        dedup: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            cid = str(entry.get("canonical_id") or "").strip()
            if not cid:
                continue
            if cid not in dedup:
                order.append(cid)
            dedup[cid] = dict(entry)
        # Apply the per-entry cap on attempts.
        new_list: list[dict[str, Any]] = []
        for cid in order:
            row = dedup[cid]
            attempts = list(row.get("attempts") or [])
            if len(attempts) > _GAPS_ATTEMPTS_HISTORY:
                attempts = attempts[-_GAPS_ATTEMPTS_HISTORY:]
            row["attempts"] = attempts
            new_list.append(row)
        if len(new_list) > _GAPS_MAX_ENTRIES:
            new_list = new_list[-_GAPS_MAX_ENTRIES:]
        self.gaps = new_list

    def record_intervention(
        self,
        *,
        change_type: str,
        action: str,
        task_id: str = "",
        delta_pct: float | None = None,
    ) -> None:
        """PR-A8 (Arbor-into-Hyperloom): append one entry to the
        intervention-mix ledger and update ``consecutive_config_only_rounds``.

        ``change_type`` is normalised to lowercase. Unknown values
        ("kernel", "noop", ...) are still appended (the ledger is
        descriptive); the consecutive-config counter only advances on
        explicit ``"config"`` values and resets on ``"code_patch"``.
        ``"code_patch_attempt"`` satisfies exploration-depth accounting
        without counting as a kept code patch.
        """
        ct = str(change_type or "").strip().lower()
        entry = {
            "change_type": ct,
            "action": str(action or ""),
            "task_id": str(task_id or ""),
            "delta_pct": delta_pct,
            "ts": _now_iso(),
        }
        self.intervention_mix.append(entry)
        if ct == "config":
            self.consecutive_config_only_rounds = (
                int(self.consecutive_config_only_rounds or 0) + 1
            )
        elif ct == "code_patch":
            self.consecutive_config_only_rounds = 0
        self.bump_depth_patch(ct)

    def get_intervention_mix(self, *, recent_window: int = 5) -> dict[str, Any]:
        """Summarise the intervention-mix ledger as derived counts.

        Returns config vs code_patch totals over the whole ledger and
        the most recent ``recent_window`` entries, plus
        ``consecutive_config_only`` (length of the trailing config-only
        run) and ``config_heavy`` (config has dominated and no code
        patch has landed). Read-only; never raises.

        Entries with an unknown ``change_type`` (e.g. analysis-only
        rounds) are ignored for the config/code tallies but still break
        the trailing config-only run.
        """
        ledger = [e for e in (self.intervention_mix or []) if isinstance(e, dict)]

        def _ct(entry: dict[str, Any]) -> str:
            return str(entry.get("change_type") or "").strip().lower()

        total_config = sum(1 for e in ledger if _ct(e) == "config")
        total_code_patch = sum(1 for e in ledger if _ct(e) == "code_patch")
        total_code_patch_attempt = sum(
            1 for e in ledger
            if _ct(e) in ("code_patch", "code_patch_attempt")
        )
        window = ledger[-recent_window:] if recent_window > 0 else ledger
        recent_config = sum(1 for e in window if _ct(e) == "config")
        recent_code_patch = sum(1 for e in window if _ct(e) == "code_patch")
        recent_code_patch_attempt = sum(
            1 for e in window
            if _ct(e) in ("code_patch", "code_patch_attempt")
        )

        consecutive_config_only = 0
        for e in reversed(ledger):
            ct = _ct(e)
            if ct == "config":
                consecutive_config_only += 1
            else:
                break

        return {
            "total_config": total_config,
            "total_code_patch": total_code_patch,
            "total_code_patch_attempt": total_code_patch_attempt,
            "recent_config": recent_config,
            "recent_code_patch": recent_code_patch,
            "recent_code_patch_attempt": recent_code_patch_attempt,
            "consecutive_config_only": consecutive_config_only,
            "config_heavy": total_config >= recent_window and total_code_patch == 0,
        }

    def bump_specialist_dispatched(self, n: int = 1) -> int:
        """PR-A8: increment the per-EXPLORE specialist dispatch counter.

        Returns the post-increment value so callers can act on it
        inline (e.g. a robustness storm threshold).
        """
        self.explore_specialist_dispatched_count = (
            int(self.explore_specialist_dispatched_count or 0) + int(n)
        )
        return self.explore_specialist_dispatched_count

    def reset_specialist_dispatched(self) -> None:
        """PR-A8: zero the per-EXPLORE specialist dispatch counter.

        Called by Coordinator on phase transition into a fresh
        EXPLORE entry.
        """
        self.explore_specialist_dispatched_count = 0

    def bump_research_scout_runs(self, n: int = 1) -> int:
        """Increment the research-scout dispatch counter; return new total."""
        self.research_scout_runs = int(self.research_scout_runs or 0) + int(n)
        return self.research_scout_runs

    def register_seen_pr_ids(self, pr_ids: Any) -> int:
        """Add PR ids to the shared seen-set (scout + FRAMEWORK_PR dedup).

        Ids are normalised to strings; duplicates and blanks are ignored.
        Returns the number of newly-added ids.
        """
        seen = set(self.research_scout_seen_pr_ids or [])
        added = 0
        for raw in pr_ids or []:
            pid = str(raw or "").strip()
            if not pid or pid in seen:
                continue
            seen.add(pid)
            self.research_scout_seen_pr_ids.append(pid)
            added += 1
        return added

    def has_seen_pr_id(self, pr_id: Any) -> bool:
        """True iff ``pr_id`` was already surfaced by scout / FRAMEWORK_PR."""
        pid = str(pr_id or "").strip()
        return bool(pid) and pid in set(self.research_scout_seen_pr_ids or [])

    # ------------------------------------------------------------------
    # Exploration-depth tracker
    # ------------------------------------------------------------------
    def _depth(self) -> dict[str, Any]:
        """Return the depth_tracker sub-dict, repairing a missing / stale
        shape so old state.json loads degrade to defaults."""
        dt = self.depth_tracker
        if not isinstance(dt, dict):
            dt = {}
            self.depth_tracker = dt
        dt.setdefault("enabled", True)
        for key in ("prs_fetched", "pr_diffs_read", "nvidia_refs_compared"):
            if not isinstance(dt.get(key), list):
                dt[key] = []
        for key in ("code_patches_attempted", "config_changes_attempted",
                    "consecutive_reverts"):
            try:
                dt[key] = int(dt.get(key) or 0)
            except (TypeError, ValueError):
                dt[key] = 0
        return dt

    def set_depth_gate_enabled(self, enabled: bool) -> None:
        self._depth()["enabled"] = bool(enabled)

    def depth_gate_enabled(self) -> bool:
        return bool(self._depth().get("enabled", True))

    def _depth_register_ids(self, key: str, ids: Any) -> int:
        dt = self._depth()
        seen = set(dt.get(key) or [])
        added = 0
        for raw in ids or []:
            val = str(raw or "").strip()
            if not val or val in seen:
                continue
            seen.add(val)
            dt[key].append(val)
            added += 1
        return added

    def register_research_evidence(
        self,
        *,
        prs_fetched: Any = None,
        pr_diffs_read: Any = None,
        nvidia_refs_compared: Any = None,
    ) -> dict[str, int]:
        """De-dup research evidence into the depth tracker; return per-key
        added counts. Used when aggregating ``specialist_done.research``."""
        return {
            "prs_fetched": self._depth_register_ids("prs_fetched", prs_fetched),
            "pr_diffs_read": self._depth_register_ids(
                "pr_diffs_read", pr_diffs_read,
            ),
            "nvidia_refs_compared": self._depth_register_ids(
                "nvidia_refs_compared", nvidia_refs_compared,
            ),
        }

    def bump_depth_patch(self, change_type: str) -> None:
        """Advance the code/config patch-attempt counters from a KEEP."""
        dt = self._depth()
        ct = str(change_type or "").strip().lower()
        if ct in ("code_patch", "code_patch_attempt"):
            dt["code_patches_attempted"] = int(
                dt.get("code_patches_attempted") or 0
            ) + 1
        elif ct == "config":
            dt["config_changes_attempted"] = int(
                dt.get("config_changes_attempted") or 0
            ) + 1

    def note_explore_outcome(self, *, promoted: bool) -> int:
        """Update ``consecutive_reverts`` from an explore-round outcome.

        A promoted KEEP resets the run to 0; a no-promote round increments
        it. Returns the post-update value.
        """
        dt = self._depth()
        if promoted:
            dt["consecutive_reverts"] = 0
        else:
            dt["consecutive_reverts"] = int(
                dt.get("consecutive_reverts") or 0
            ) + 1
        return int(dt["consecutive_reverts"])

    def depth_snapshot(self) -> dict[str, Any]:
        """Read-only normalised view of the depth tracker for the gate /
        prompt (sets rendered as counts + the underlying id lists)."""
        dt = self._depth()
        return {
            "enabled": bool(dt.get("enabled", True)),
            "research_scout_runs": int(self.research_scout_runs or 0),
            "prs_fetched": list(dt.get("prs_fetched") or []),
            "pr_diffs_read": list(dt.get("pr_diffs_read") or []),
            "nvidia_refs_compared": list(dt.get("nvidia_refs_compared") or []),
            "code_patches_attempted": int(dt.get("code_patches_attempted") or 0),
            "config_changes_attempted": int(
                dt.get("config_changes_attempted") or 0
            ),
            "consecutive_reverts": int(dt.get("consecutive_reverts") or 0),
        }

    def to_intervention_mix_summary(self) -> str:
        """PR-A8 / D3 (Arbor-into-Hyperloom): render the config-vs-code_patch
        intervention ledger for the Orchestration per-tick prompt.

        Returns ``""`` when the ledger is empty (nothing recorded yet — no
        escalation possible). Otherwise returns a one-line counts summary,
        plus an ``ESCALATION`` directive when the session has been
        config-only for too long, nudging Orchestration to dispatch a
        code-patch ``serving_specialist`` next (Arbor's "do not settle for
        config-only" rule). ``record_intervention`` maintains the ledger;
        this is its sole consumer.

        Thresholds mirror Arbor's ``get_intervention_mix`` heuristics:
        escalate when ``consecutive_config_only_rounds >= 2`` OR the ledger
        is config-heavy (>= 5 config keeps) with zero code_patch keeps.
        """
        mix = self.intervention_mix or []
        if not mix:
            return ""
        escalate_at = 2
        config_heavy_at = 5
        n_config = sum(
            1 for m in mix if (m or {}).get("change_type") == "config"
        )
        n_patch = sum(
            1 for m in mix if (m or {}).get("change_type") == "code_patch"
        )
        n_patch_attempt = sum(
            1 for m in mix
            if (m or {}).get("change_type") in (
                "code_patch", "code_patch_attempt",
            )
        )
        # B2: config explore rounds that produced measurements but KEPT
        # nothing (all REVERT / KEEP_UNSTABLE) are recorded as
        # ``config_attempt``. They count toward the escalation signal so
        # repeated fruitless config tuning escalates to a code-patch
        # ``integrate_patch`` even when the MI300X noise floor prevents
        # any config KEEP (the failure mode that left this loop spinning).
        n_config_attempt = sum(
            1 for m in mix if (m or {}).get("change_type") == "config_attempt"
        )
        consec = int(self.consecutive_config_only_rounds or 0)
        config_pressure = n_config + n_config_attempt
        lines = [
            f"config_keeps={n_config} config_attempts={n_config_attempt} "
            f"code_patch_keeps={n_patch} code_patch_attempts={n_patch_attempt} "
            f"consecutive_config_only_rounds={consec}"
        ]
        if (
            consec >= escalate_at
            or (n_config >= config_heavy_at and n_patch == 0)
            or (config_pressure >= escalate_at and n_patch == 0)
        ):
            lines.append(
                "ESCALATION: config-only for too long. Config tuning has a "
                "low ceiling — your NEXT EXPLORE dispatch SHOULD delegate a "
                "`serving_specialist` to author a framework SOURCE patch "
                "(scheduler / kv_cache / chunked-prefill), integrated via "
                "`integrate_patch`, rather than another config-only "
                "`params` / `explore` round."
            )
        return "\n".join(lines)

    def record_dynamic_action_dispatch(
        self, dyn_id: str, summary: dict[str, Any],
    ) -> int:
        """Register a freshly-dispatched ``dynamic_action`` + bump the
        per-round counter atomically.

        Coordinator-only writer (CORE_STATE_FIELDS protects it from
        LLM ``UPDATE_STATE``). Idempotent on ``dyn_id``: a re-record
        overwrites the summary but does not double-bump the counter.
        Returns the post-increment round counter value.
        """
        key = str(dyn_id or "").strip()
        if not key:
            return self.dynamic_action_round_count
        was_new = key not in self.dynamic_actions
        self.dynamic_actions[key] = dict(summary or {})
        if was_new:
            self.dynamic_action_round_count = int(
                self.dynamic_action_round_count or 0
            ) + 1
        return self.dynamic_action_round_count

    def reset_dynamic_action_round_count(self) -> None:
        """Clear the per-EXPLORE round cap counter; the cumulative
        ``dynamic_actions`` ledger is preserved across rounds."""
        self.dynamic_action_round_count = 0

    def record_dynamic_action_outcome(
        self,
        dyn_id: str,
        *,
        status: str,
        last_outcome: str | None = None,
        cumulative_gain: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Transition-validated update of the summary row keyed by
        ``dyn_id``.

        Coordinator-only writer (``CORE_STATE_FIELDS`` guards it from
        LLM ``UPDATE_STATE``). Only keys passed in are touched;
        ``last_outcome`` defaults to the prompt-friendly label for
        ``status``; ``last_updated_at`` is always refreshed. Illegal
        transitions are logged and skipped so a buggy hook cannot
        corrupt the audit trail.
        """
        # Local import avoids the cycle with
        # :mod:`dynamic_action_proposal`.
        from .dynamic_action_proposal import (
            DynamicActionStatus,
            LAST_OUTCOME_BY_STATUS,
            can_transition,
        )

        key = str(dyn_id or "").strip()
        if not key:
            return
        import logging as _logging
        _log = _logging.getLogger(__name__)
        try:
            target_status = DynamicActionStatus(
                str(status or "").strip(),
            )
        except ValueError:
            _log.warning(
                "record_dynamic_action_outcome: unknown status=%r for "
                "dyn_id=%s; dropping write",
                status, key,
            )
            return
        existing = dict(self.dynamic_actions.get(key) or {})
        current_status_raw = existing.get("status")
        current_status: DynamicActionStatus | None
        if current_status_raw:
            try:
                current_status = DynamicActionStatus(str(current_status_raw))
            except ValueError:
                current_status = None
        else:
            current_status = None
        if not can_transition(current_status, target_status):
            _log.warning(
                "record_dynamic_action_outcome: illegal transition "
                "%s → %s for dyn_id=%s; preserving prior state",
                current_status_raw or "(none)", target_status.value, key,
            )
            return
        existing.setdefault("dyn_id", key)
        existing["status"] = target_status.value
        existing["last_outcome"] = str(
            last_outcome
            if last_outcome is not None
            else LAST_OUTCOME_BY_STATUS.get(target_status, target_status.value.lower()),
        )
        if cumulative_gain is not None:
            existing["cumulative_gain"] = float(cumulative_gain)
        if extra:
            for k, v in extra.items():
                existing[k] = v
        existing["last_updated_at"] = _now_iso()
        self.dynamic_actions[key] = existing

    def to_dynamic_actions_prompt_section(
        self,
        *,
        max_entries: int = 5,
        title: str = "Dynamic Action History",
    ) -> str:
        """Compact ``=== <title> ===`` block for orchestration prompt
        injection.

        Renders the most recent ``max_entries`` summaries sorted by
        ``last_updated_at`` descending (stable tiebreak on
        ``dyn_id``). Older rows are collapsed into an elision marker
        pointing at the on-disk artefact dir. Returns the empty
        string when no rows exist so callers can skip the section.
        """
        summaries = self.dynamic_actions or {}
        if not summaries:
            return ""
        from .dynamic_action_proposal import (
            DynamicActionStatus,
            LAST_OUTCOME_BY_STATUS,
        )

        ordered = sorted(
            summaries.items(),
            key=lambda kv: (
                str(kv[1].get("last_updated_at") or ""),
                str(kv[0]),
            ),
            reverse=True,
        )
        recent = ordered[: max(0, int(max_entries))]
        older = ordered[max(0, int(max_entries)):]
        lines: list[str] = [f"=== {title} ==="]
        for _dyn_id, summary in recent:
            lines.append(self._format_dynamic_action_summary_row(
                summary,
                _last_outcome_lookup=LAST_OUTCOME_BY_STATUS,
                _status_enum=DynamicActionStatus,
            ))
        if older:
            lines.append(
                f"... ({len(older)} more older entries; full list in "
                f"$SESSION_DIR/agents/orchestration/dynamic_actions/)"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_dynamic_action_summary_row(
        summary: dict[str, Any],
        *,
        _last_outcome_lookup: dict | None = None,
        _status_enum: Any | None = None,
    ) -> str:
        """Render one compact summary row (~50 tokens):

            - <dyn_id> [<STATUS>, gain=<delta>%] scope=[d1,d2]
              motivation: "<motivation_gap_short>"
              artifact: <artifact_path>
        """
        dyn_id = str(summary.get("dyn_id") or "(unknown)")
        status = str(summary.get("status") or "(unknown)")
        scope = list(summary.get("scope_domains") or ())
        motivation = str(summary.get("motivation_gap_short") or "").strip()
        if not motivation:
            motivation = "(no motivation summary)"
        gain = summary.get("cumulative_gain")
        gain_text = (
            f"gain={gain:+.2f}%" if isinstance(gain, (int, float)) else "gain=n/a"
        )
        artifact = str(summary.get("artifact_path") or "(missing)")
        last_outcome = str(summary.get("last_outcome") or "")
        last_outcome_suffix = (
            f" outcome={last_outcome}" if last_outcome else ""
        )
        head = (
            f"- {dyn_id} [{status}, {gain_text}{last_outcome_suffix}] "
            f"scope={scope!r}"
        )
        body = f'  motivation: "{motivation}"'
        tail = f"  artifact: {artifact}"
        return "\n".join((head, body, tail))

    def record_specialist_patch_verdict(
        self, specialist_task_id: str, verdict: str,
    ) -> None:
        """PR-A7 (Arbor-into-Hyperloom): record the Critic's verdict on a
        specialist's worktree patches.

        Idempotent: a later verdict overwrites an earlier one (the
        Critic may produce a revised verdict after a needs_review
        round-trip). Empty / falsy ``verdict`` clears the entry; this
        is how an operator forces a re-review by deleting the prior
        decision.
        """
        sid = str(specialist_task_id or "").strip()
        if not sid:
            return
        v = str(verdict or "").strip().lower()
        if not v:
            self.specialist_patch_verdicts.pop(sid, None)
            return
        self.specialist_patch_verdicts[sid] = v

    def get_specialist_patch_verdict(
        self, specialist_task_id: str,
    ) -> str:
        """PR-A7: return the recorded patch verdict, or empty string
        when no critic decision is on record yet."""
        sid = str(specialist_task_id or "").strip()
        if not sid:
            return ""
        return self.specialist_patch_verdicts.get(sid, "") or ""

    def update_last_specialist(self, snapshot: dict[str, Any]) -> None:
        """Snapshot the most recent specialist task (parity with last_*)."""
        if isinstance(snapshot, dict):
            self.last_specialist = dict(snapshot)

    def apply_explore_search_update(self, update: dict[str, Any]) -> None:
        """Merge an ExploreExecutor search update into persistent state.

        v0.8 M3. The executor returns a ledger
        increment containing ``tested`` (full fingerprint-keyed map),
        ``rejected`` (list), ``winners_history`` increment,
        ``synergy_attempted`` increment, ``last_round``, etc. The
        executor never writes ``accepted`` directly —
        :meth:`record_explore_accepted` is the single writer for that
        bucket (Coordinator calls it on promote).
        """
        if not isinstance(update, dict):
            return
        prior = self.explore_search if isinstance(self.explore_search, dict) else {}
        merged = dict(prior)
        merged["schema_version"] = int(update.get("schema_version") or 1)
        merged["tested"] = dict(update.get("tested") or prior.get("tested") or {})
        merged["rejected"] = list(update.get("rejected") or prior.get("rejected") or [])
        merged["name_index"] = dict(
            update.get("name_index") or prior.get("name_index") or {}
        )
        merged["cursor"] = int(update.get("cursor") or len(merged["tested"]))
        merged["last_round"] = dict(update.get("last_round") or {})
        # Append-only history fields — merge instead of overwrite.
        wh = list(prior.get("winners_history") or [])
        for entry in update.get("winners_history") or []:
            if isinstance(entry, dict):
                wh.append(dict(entry))
        merged["winners_history"] = wh
        sa: set[tuple[str, ...]] = set()
        for src in (prior.get("synergy_attempted"), update.get("synergy_attempted")):
            for c in src or []:
                if isinstance(c, list):
                    items = tuple(sorted(str(x) for x in c if isinstance(x, str)))
                    if items:
                        sa.add(items)
                elif isinstance(c, str) and c:
                    items = tuple(sorted(c.split("+")))
                    if items:
                        sa.add(items)
        merged["synergy_attempted"] = [list(c) for c in sorted(sa)]
        merged["discovered_flags"] = list(
            update.get("discovered_flags") or prior.get("discovered_flags") or []
        )
        merged["domains_round_summary"] = list(
            update.get("domains_round_summary")
            or prior.get("domains_round_summary") or []
        )
        # Preserve any accepted bucket from prior runs; executor never
        # writes accepted directly (record_explore_accepted does).
        merged["accepted"] = list(prior.get("accepted") or [])
        # Drop the merged_from_legacy_sig sentinel on each update so a
        # subsequent SharedState load re-runs the legacy union (defensive
        # against an interleaved v0.6 fallback session writing into the
        # old ledgers between save/load cycles).
        merged.pop("merged_from_legacy_sig", None)
        self.explore_search = merged

    def record_explore_accepted(self, variant: dict[str, Any]) -> None:
        """Append one promoted variant to ``explore_search.accepted``.

        Called by Coordinator after an explore winner survives both the
        per-variant KEEP gate and the inlined stack rebench. Dedupes by
        ``fingerprint`` so repeated promotes of the same content don't
        bloat the list; also removes a matching entry from
        ``rejected`` so a previously-rejected variant that later
        promotes doesn't appear in both buckets.
        """
        if not isinstance(variant, dict) or not variant:
            return
        from .action_executors._canonical_fingerprint import canonical_fingerprint
        args = str(
            variant.get("candidate_extra_server_args")
            or variant.get("extra_server_args") or ""
        )
        envs = dict(variant.get("extra_envs") or {})
        fp = str(variant.get("fingerprint") or canonical_fingerprint(args, envs))
        entry = {
            "fingerprint": fp,
            "name": str(variant.get("name") or ""),
            "extra_server_args": args,
            "extra_envs": envs,
            "note": str(variant.get("note") or ""),
            "tput": variant.get("output_throughput") or variant.get("tput"),
            "gain_pct": variant.get("gain_pct"),
            "stack_index": variant.get("stack_index"),
            "accepted_at_round": str(variant.get("accepted_at_round") or ""),
            "ts": str(variant.get("ts") or _now_iso()),
            "provenance": str(variant.get("provenance") or "llm_direct"),
        }
        search = dict(self.explore_search or {})
        search.setdefault("schema_version", 1)
        accepted = [
            v for v in (search.get("accepted") or [])
            if not (isinstance(v, dict) and v.get("fingerprint") == fp)
        ]
        accepted.append(entry)
        search["accepted"] = accepted
        search["rejected"] = [
            v for v in (search.get("rejected") or [])
            if not (isinstance(v, dict) and v.get("fingerprint") == fp)
        ]
        name_index = dict(search.get("name_index") or {})
        if entry["name"]:
            name_index[entry["name"]] = fp
        search["name_index"] = name_index
        # Append a winners_history row so plateau judges can read this
        # without crawling the optimization_stack.
        wh = list(search.get("winners_history") or [])
        wh.append({
            "round_id": entry["accepted_at_round"],
            "variant_name": entry["name"],
            "fingerprint": fp,
            "gain_pct": entry["gain_pct"],
            "extra_args": args,
            "extra_envs": envs,
            "provenance": entry["provenance"],
            "ts": entry["ts"],
        })
        search["winners_history"] = wh
        self.explore_search = search

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

        Called by ExploreExecutor (and historically by the legacy
        BackendsExecutor / ParamsExecutor) when they first run
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
                entry.get("candidate_extra_server_args")
                or entry.get("extra_server_args") or ""
            )
            envs = dict(entry.get("extra_envs") or {})
            return {
                "name": str(entry.get("name", "")),
                "tput": entry.get("output_throughput") or entry.get("tput"),
                "gain_pct": entry.get("gain_pct"),
                "extra_server_args": args,
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

    # ------------------------------------------------------------------
    # scoring helpers removed
    # ------------------------------------------------------------------
    # The v0.6 ``get_action_score`` / ``put_action_score`` /
    # ``all_action_scores`` / ``to_action_scores_summary`` API has
    # been retired. The LLM no longer consumes a system-side priority;
    # decisions are based on facts (phase / gaps / KB / specialist
    # rounds). ``increment_tick`` stays — it's a pure monotonic
    # counter used by plateau / phase budget math.
    def increment_tick(self) -> int:
        """Bump the Coordinator tick counter and return the new value."""
        self.tick = int(self.tick or 0) + 1
        return self.tick

    # Note: main commit 8e69732 also ports ``to_action_scores_summary``
    # — a render helper for the legacy ``Action scores`` prompt block.
    # KB_design §3.9 retired the scoreboard on this branch (see
    # the retired-features list §4), so the helper
    # has no live consumer and is intentionally omitted.

    def append_stack_gain_entry(
        self,
        *,
        action: str,
        variant_name: str | None,
        new_tput: float,
        extra_server_args: str = "",
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
        extra_args = self.current_best.get("extra_server_args")
        if not variant and not extra_args:
            return
        self.optimization_stack = [{
            "action": self.current_best.get("action", "unknown"),
            "variant_name": variant or "legacy_current_best",
            "extra_server_args": extra_args or "",
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
    # Time-budget helpers (consumed by Coordinator._compose_prompt)
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
        """True iff a new KEEP has landed since the last inline stack rebench.

        Used by Coordinator to surface the TODO 4 ``stack rebench
        required`` guard in the per-tick checklist. The check is purely
        on stack *length*: every successful inline rebench (v0.8 M3
        explore per-KEEP loop) records
        ``cumulative_gain_validated_stack_len``, so a longer stack means
        at least one new KEEP (e.g. from ``integrate``) came in without
        an end-to-end revalidation.
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
            " ⚠ stack changed since last rebench — RUN `explore` "
            "(per-KEEP stack rebench is inlined)"
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

    def to_phase_status_summary(
        self,
        *,
        budget_pct: dict[str, float] | None = None,
        now_unix: float | None = None,
    ) -> str:
        """Render the per-tick ``=== Phase ===`` block (v0.8 §3.3).

        The Coordinator pipes this into every agent's per-tick prompt.
        Output stays compact (≤ 5 lines) because every reactor reads
        it; we keep the human-readable form here and let
        :func:`phase_state.phase_budget_remaining_seconds` carry the
        budget math.

        When in EXPLORE, an extra ``force_exit`` line surfaces how much
        runway is left before the HARD force-exit gate (IR-6) fires.
        """
        from .phase_state import (
            DEFAULT_EXPLORE_FORCE_EXIT_BUDGET_PCT,
            DEFAULT_EXPLORE_FORCE_EXIT_HOURS_REMAINING,
            PHASE_EXPLORE,
            allowed_actions_for,
            normalize_budget_pct,
            phase_budget_remaining_seconds,
            phase_elapsed_seconds,
            session_remaining_seconds,
        )

        phase = (self.phase or "").strip().upper() or "UNSET"
        elapsed = int(phase_elapsed_seconds(self, now_unix=now_unix))
        budget = normalize_budget_pct(budget_pct or self.phase_budget_pct)
        budget_pct_for_phase = budget.get(phase, 0.0)
        remaining = phase_budget_remaining_seconds(
            self, budget_pct=budget, now_unix=now_unix,
        )
        budget_line: str
        if remaining is None:
            budget_line = (
                f"budget    : pct={budget_pct_for_phase:.2f} (unlimited run; "
                f"no per-phase cap)"
            )
        else:
            budget_line = (
                f"budget    : pct={budget_pct_for_phase:.2f} "
                f"elapsed_sec={elapsed} remaining_sec={int(remaining)}"
            )
        allowed = allowed_actions_for(phase)
        allowed_line = (
            f"allowed   : {', '.join(allowed) if allowed else '(none)'}"
        )
        lines = [
            f"phase     : {phase}",
            f"entered   : {self.phase_started_ts or '(unset)'}",
            budget_line,
            allowed_line,
        ]
        # EXPLORE-only: show distance to HARD force-exit (IR-6) so the
        # LLM has a deterministic countdown alongside the soft budget.
        if phase == PHASE_EXPLORE:
            overrides = self.plateau_overrides or {}
            hours_thresh = float(overrides.get(
                "force_exit_hours_remaining",
                DEFAULT_EXPLORE_FORCE_EXIT_HOURS_REMAINING,
            ))
            pct_thresh = float(overrides.get(
                "force_exit_budget_pct",
                DEFAULT_EXPLORE_FORCE_EXIT_BUDGET_PCT,
            ))
            session_remaining = session_remaining_seconds(
                self, now_unix=now_unix,
            )
            session_buffer = (
                int(session_remaining - hours_thresh * 3600.0)
                if session_remaining is not None else None
            )
            if remaining is not None and budget_pct_for_phase > 0:
                mm = float(self.max_minutes or 0)
                phase_total_sec = mm * 60.0 * budget_pct_for_phase
                phase_remaining_pct = (
                    remaining / phase_total_sec if phase_total_sec > 0 else 0.0
                )
            else:
                phase_remaining_pct = None
            force_line = (
                f"force_exit: hours_thresh={hours_thresh:.1f}h "
                f"pct_thresh={pct_thresh:.2f}"
            )
            if session_buffer is not None:
                force_line += f" session_buffer_sec={session_buffer}"
            if phase_remaining_pct is not None:
                force_line += (
                    f" phase_remaining_pct={phase_remaining_pct:.3f}"
                )
            lines.append(force_line)
        return "\n".join(lines)

    def to_phase_budget_telemetry(
        self,
        *,
        budget_pct: dict[str, float] | None = None,
        now_unix: float | None = None,
    ) -> str:
        """Render the per-phase budget telemetry block for Robustness.

        Lists each phase that appears in ``phase_history`` plus the
        current phase. Each line shows ``phase: elapsed=Xs cap=Ys (Z%)``
        so Robustness can spot a phase that's blown past its budget.
        """
        from .phase_state import (
            DEFAULT_PHASE_BUDGET_PCT,
            PHASE_NAMES,
            normalize_budget_pct,
            phase_elapsed_seconds,
        )

        budget = normalize_budget_pct(budget_pct or self.phase_budget_pct)
        # Aggregate elapsed per phase using phase_history.
        elapsed_per_phase: dict[str, float] = {}
        history = self.phase_history or []
        for idx, row in enumerate(history):
            if not isinstance(row, dict):
                continue
            phase = str(row.get("to_phase") or "").upper()
            entered = float(row.get("ts_unix") or 0.0)
            if not phase or entered <= 0:
                continue
            if idx + 1 < len(history) and isinstance(history[idx + 1], dict):
                exited = float(history[idx + 1].get("ts_unix") or entered)
            else:
                # Currently-active segment — measure to now.
                elapsed_now = phase_elapsed_seconds(self, now_unix=now_unix)
                exited = entered + elapsed_now
            elapsed_per_phase[phase] = (
                elapsed_per_phase.get(phase, 0.0) + max(0.0, exited - entered)
            )
        if not elapsed_per_phase:
            return "(no phase history yet)"
        mm = float(self.max_minutes or 0.0)
        total_budget_sec = mm * 60.0
        lines: list[str] = []
        # Stable order — iterate ``PHASE_NAMES`` so any phase added to
        # the chain (e.g. FRAMEWORK_PR) renders automatically.
        for phase in PHASE_NAMES:
            if phase not in elapsed_per_phase:
                continue
            elapsed = elapsed_per_phase[phase]
            pct = budget.get(phase, DEFAULT_PHASE_BUDGET_PCT.get(phase, 0.0))
            cap_sec = total_budget_sec * pct if total_budget_sec > 0 else 0.0
            used_pct = (elapsed / cap_sec * 100.0) if cap_sec > 0 else 0.0
            cap_line = f"cap={int(cap_sec)}s" if cap_sec > 0 else "cap=unlimited"
            lines.append(
                f"  {phase}: elapsed={int(elapsed)}s {cap_line} used={used_pct:.0f}%"
            )
        return "\n".join(lines) or "(no phase history yet)"

    def to_warm_start_summary(self, *, max_lines: int = 12) -> str:
        """Render T0 warm-start snapshot for Orchestration prompt injection.

        Produces a compact block fed into ``Coordinator._compose_prompt``'s
        ``=== Warm start ===`` section (v0.8 §3.3 §4.1). Returns empty
        string when both ``warm_start_recipe`` and ``warm_start_pitfalls``
        are absent (M1 ``--degraded-kb`` mode / first-ever session for a
        (workload, hw) pair).

        Lines are capped so a recipe blob never bloats the prompt; the
        full JSON snapshot lives at
        ``runtime/cortex/.kb_warm.json`` / ``.kb_pitfalls.json`` for any
        agent willing to Read it directly.
        """
        recipe = self.warm_start_recipe or {}
        pitfalls = self.warm_start_pitfalls or []
        if not recipe and not pitfalls:
            return ""
        out: list[str] = []
        workload = str(recipe.get("workload") or "") if isinstance(recipe, dict) else ""
        hw = str(recipe.get("hw") or "") if isinstance(recipe, dict) else ""
        if workload or hw:
            out.append(f"recipe: workload={workload or '?'} hw={hw or '?'}")
        raw = str(recipe.get("raw") or "") if isinstance(recipe, dict) else ""
        # Trim recipe raw text — at most 5 lines, 240 chars each.
        if raw.strip():
            kept = 0
            for line in raw.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                out.append(f"  · {stripped[:240]}")
                kept += 1
                if kept >= 5:
                    break
            if kept == 0:
                out.append("  · (recipe present but text was empty)")
        else:
            out.append("  · (no recipe text — first session for this workload/hw)")
        if pitfalls:
            out.append(f"pitfalls ({len(pitfalls)}):")
            for entry in pitfalls[:5]:
                if not isinstance(entry, dict):
                    continue
                snippet = str(entry.get("raw") or entry.get("symptom") or "")
                if not snippet.strip():
                    continue
                first_line = snippet.splitlines()[0].strip()
                out.append(f"  · {first_line[:240]}")
        if max_lines and len(out) > max_lines:
            out = out[:max_lines]
            out.append(f"  · (truncated to {max_lines} lines; "
                       f"see runtime/cortex/.kb_warm.json for full snapshot)")
        return "\n".join(out)

    def to_gaps_summary(self, *, max_entries: int = 10) -> str:
        """Render :attr:`gaps` for Orchestration / specialist prompt injection.

        KB_design §3.3 / §3.5: surfaces the structured gap list that
        replaced the legacy ``last_action_failures`` proxy. Returns an
        empty string when ``gaps`` is empty so callers can skip the
        whole section header on cold-start sessions.

        Format (per row):

            - <canonical_id> [<layer>/<severity>] <symptom>
                attempts=N last=<action:outcome>

        Capped at ``max_entries`` newest rows (sorted by
        ``last_updated_ts``) to keep the prompt section bounded —
        long sessions occasionally accumulate dozens of gaps; the
        full list still lives in ``state.json`` for resume.
        """
        if not self.gaps:
            return ""
        # Newest first by last_updated_ts (falling back to first_seen_ts
        # / insertion order to keep deterministic for tests).
        ordered = list(self.gaps)
        ordered.sort(
            key=lambda g: str(
                g.get("last_updated_ts") or g.get("first_seen_ts") or "",
            ),
            reverse=True,
        )
        rows: list[str] = []
        for gap in ordered[:max_entries]:
            if not isinstance(gap, dict):
                continue
            cid = str(gap.get("canonical_id") or "?")
            layer = str(gap.get("layer") or "?")
            severity = str(gap.get("severity") or "?")
            symptom = str(gap.get("symptom") or "").replace("\n", " ").strip()
            if len(symptom) > 200:
                symptom = symptom[:197] + "..."
            attempts = gap.get("attempts") or []
            attempt_n = len(attempts) if isinstance(attempts, list) else 0
            last_tag = ""
            if isinstance(attempts, list) and attempts:
                last = attempts[-1]
                if isinstance(last, dict):
                    last_tag = (
                        f" last={last.get('action','?')}:"
                        f"{last.get('outcome','?')}"
                    )
            rows.append(
                f"  - {cid} [{layer}/{severity}] {symptom}\n"
                f"      attempts={attempt_n}{last_tag}"
            )
        if len(ordered) > max_entries:
            rows.append(
                f"  · (+{len(ordered) - max_entries} older gaps elided; "
                f"see state.json `gaps[]`)"
            )
        return "\n".join(rows)

    def to_proposal_scores_summary(self, *, max_rounds: int = 2) -> str:
        """Render advisory multi-model proposal scores for Orchestration.

        Reads the ``ensemble_scores`` blob attached to the most recent
        :attr:`specialist_rounds` entries by the ProposalScorer (see
        ``orchestrator/proposal_scorer.py``). Each line shows a variant's
        per-rater ``score ("reason")`` side-by-side so Orchestration can
        see agreement / disagreement. There is deliberately NO mean and
        NO sorting — these are one reference among many (parallel to
        gaps / KB / analysis.md), not a ranking directive (Inv-9.1: no
        system-side scoreboard).

        The rater identities are **anonymized** (``rater_1``, ``rater_2``,
        …) so Orchestration weighs the scores on their merits and reasons
        alone, free of any per-model prior / brand bias. The mapping is
        *stable within a render* (the same model maps to the same
        ``rater_N`` across every round shown) so cross-round agreement is
        still legible, but the real model slugs never reach the prompt —
        they stay in ``ensemble_scores`` (state.json) for debug / audit.

        Returns ``""`` when no recent round carries scores so the caller
        skips the whole section header.

        Format::

            round=<round_id> domain=<domain>
              - <variant_name>: rater_1=8.0 ("..."), rater_2=6.5 ("...")
        """
        rounds = [
            r for r in (self.specialist_rounds or [])
            if isinstance(r, dict)
            and isinstance(r.get("ensemble_scores"), dict)
            and (r["ensemble_scores"].get("models") or {})
        ]
        if not rounds:
            return ""
        shown = rounds[-max_rounds:]
        # Stable, anonymized rater labels: collect every real model slug
        # across the rounds being shown, sort for determinism, and map
        # each to ``rater_N``. The same model gets the same label across
        # rounds (cross-round agreement stays legible) while the real
        # slug never reaches the prompt — Orchestration must weigh scores
        # on merit, not on which brand emitted them (avoids model bias).
        all_slugs: set[str] = set()
        for r in shown:
            models = r["ensemble_scores"].get("models") or {}
            all_slugs.update(str(s) for s in models.keys())
            errs = r["ensemble_scores"].get("errors") or {}
            all_slugs.update(str(s) for s in errs.keys())
        rater_label = {
            slug: f"rater_{i}"
            for i, slug in enumerate(sorted(all_slugs), start=1)
        }
        rows: list[str] = [
            "(Advisory only — one reference among many, NOT a ranking "
            "directive. Scores are 0-10 likelihood-of-throughput-gain "
            "priors from independent anonymized raters; weigh on merit "
            "alongside gaps / KB / analysis.md.)",
        ]
        for r in shown:
            ens = r["ensemble_scores"]
            models = ens.get("models") or {}
            scale = str(ens.get("scale") or "0-10")
            round_id = str(r.get("round_id") or "?")
            domain = str(r.get("domain") or "?")
            rows.append(f"round={round_id} domain={domain} scale={scale}")
            # Collect every variant name seen across models, preserving
            # the proposal_set order when available.
            ordered_names: list[str] = []
            seen: set[str] = set()
            for variant in (r.get("proposal_set") or []):
                if isinstance(variant, dict):
                    nm = str(variant.get("name") or "")
                    if nm and nm not in seen:
                        ordered_names.append(nm)
                        seen.add(nm)
            for per_model in models.values():
                if isinstance(per_model, dict):
                    for nm in per_model:
                        if nm not in seen:
                            ordered_names.append(nm)
                            seen.add(nm)
            # Render raters in stable label order (rater_1, rater_2, …)
            # so a given column means the same model across every round
            # without ever printing the model slug.
            ordered_slugs = sorted(
                (s for s in models if s in rater_label),
                key=lambda s: rater_label[s],
            )
            for nm in ordered_names:
                parts: list[str] = []
                for model_slug in ordered_slugs:
                    per_model = models.get(model_slug)
                    if not isinstance(per_model, dict):
                        continue
                    label = rater_label[model_slug]
                    cell = per_model.get(nm)
                    if isinstance(cell, dict) and cell.get("score") is not None:
                        reason = str(cell.get("reason") or "").replace("\n", " ")
                        if len(reason) > 80:
                            reason = reason[:77] + "..."
                        parts.append(
                            f"{label}={float(cell['score']):.1f} "
                            f"(\"{reason}\")"
                        )
                    else:
                        parts.append(f"{label}=n/a")
                rows.append(f"  - {nm}: " + ", ".join(parts))
            errors = ens.get("errors") or {}
            if errors:
                err_labels = ", ".join(
                    sorted(
                        rater_label.get(str(s), "rater_?")
                        for s in errors
                    )
                )
                rows.append(f"  · raters unavailable this round: {err_labels}")
        return "\n".join(rows)

    def to_prompt_summary(self) -> str:
        """Compact, human-readable snapshot for prompt injection (DESIGN §8.3)."""
        lines = [
            f"session_id={self.session_id or '(unset)'}",
            f"model={self.model_name or '(unset)'}  class={self.model_class or '(unset)'}",
        ]
        # Advisory architecture profile (launcher-supplied). Prompt-context
        # only; the live TraceLens ``analysis_md`` snapshot below remains the
        # ground truth for bottleneck classification. Omit entirely when no
        # profile was loaded so non-arch sessions render exactly as before.
        _arch_line = render_model_arch_compact(self.model_arch)
        if _arch_line:
            lines.append(
                f"model_arch(advisory; subordinate to TraceLens analysis_md)={_arch_line}"
            )
        lines += [
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
            # Canonical post-M4 cache key; legacy ``last_select_kernels``
            # was removed in this branch (callers must use this field).
            f"last_trace_analyze={self._format_last_trace_analyze()}",
            # Full TraceLens ``analysis.md`` (snapshot id + gain in the
            # bookend header) so the orchestration LLM grounds
            # propose_action decisions in the actual report.
            f"analysis_md={self._format_analysis_md_full()}",
            # the streak counter is a *fact* the LLM may
            # read (KEEP/REVERT counts are explicitly allowed per
            # Inv-9.1); only system-side *priorities* (action_scores)
            # were removed. The plateau judges also consume this on
            # legacy resume sessions when ``explore_search`` is empty.
            f"params_no_promote_streak={self.params_no_promote_streak}",
            f"explore_search={self._format_explore_search()}",
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
            f"last_gemm_tuning={self._format_attempt(self.last_gemm_tuning)}",
            f"last_explore={self._format_attempt(self.last_explore)}",
            f"last_sweep={self._format_attempt(self.last_sweep)}",
            f"attempts_history={self._format_attempts_history()}",
            f"last_action_failures={self._format_last_action_failures()}",
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
        if not self.discovered_flags:
            return "(none — first backends/params round will populate)"
        parts: list[str] = []
        for fw, entry in sorted(self.discovered_flags.items()):
            if not isinstance(entry, dict):
                continue
            n_b = len(entry.get("backend_flags") or [])
            n_p = len(entry.get("param_flags") or [])
            parts.append(f"{fw}:backend={n_b}/param={n_p}")
        return ", ".join(parts) or "(none)"

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
            str(entry.get("extra_server_args") or "").strip()
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

        Some ``explore_search.accepted`` entries do NOT persist
        ``gain_pct``; the matching ``tested[fingerprint]`` entry does.
        Pulling the value across at render time keeps the renderer
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

    def _format_explore_search(self) -> str:
        return self._format_search_state(self.explore_search)

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

    @staticmethod
    def _strip_base64_data_urls(text: str) -> str:
        """Drop base64 image payloads before prompt injection.

        TraceLens ``analysis.md`` embeds charts as
        ``![alt](data:image/png;base64,...)`` data URLs that can balloon
        the section to hundreds of KB of opaque noise. Stripping is
        in-memory only; the on-disk file stays intact for operators.

        Cherry-picked from main alongside F1-2 (RooflineExecutor); the
        helper module ``inference_optimizer/tracelens_md.py`` lands in
        the same commit.
        """
        if not text:
            return text or ""
        from inference_optimizer.tracelens_md import strip_base64_data_urls
        return strip_base64_data_urls(text)

    def _format_analysis_md_full(self) -> str:
        """F1-4 (Roofline-v2 N5): inject TraceLens analysis.md verbatim.

        Roofline composite design §6.1 mandates that analysis.md is
        handed to the orchestration LLM verbatim — no truncation, no
        sub-agent interpretation layer. The full report renders between
        ``=== TraceLens Analysis ... ===`` bookends so the LLM can
        syntactically distinguish report content from surrounding
        SharedState dump lines.

        Header carries ``snapshot=N`` + ``gain at snapshot=X.XX%`` so
        the LLM (and the orchestration.md re-profile guidance landed in
        F1-5) can detect "report is stale, gain has moved by ≥3% since
        snapshot" without parsing the body.

        Render modes:
          * cache empty / ``analysis_md_text`` missing → one-line hint
            asking the LLM to propose ``roofline``.
          * cache populated → full report between bookends.

        Cherry-picked from /wekafs/zgong/Hyperloom main @ c6f0a71
        ``shared_state.py:2772`` ; the N27 fallback-mode branch is
        omitted because PolicyGate fallback (F3) is not yet on this
        branch — the simpler default-hint message stands in until F3.
        """
        cached = self.last_trace_analyze or {}
        md_text = cached.get("analysis_md_text") or ""
        if not md_text:
            return (
                "(no TraceLens snapshot yet — analysis is auto-enqueued "
                "by the Coordinator at the end of PRELUDE and on every "
                "+10% validated-gain crossing; wait for the pending "
                "task to land, or continue with specialist / explore "
                "work that does not need analysis.md. PolicyGate denies "
                "any LLM-emitted propose_action/delegate against "
                "`roofline` or `profile` with rule "
                "`analysis_action_not_llm_proposable`.)"
            )
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

    def _format_last_trace_analyze(self) -> str:
        # Canonical post-M4 cache key. Legacy ``last_select_kernels``
        # field was removed in this branch — see SharedState dataclass
        # docstring.
        return self._format_trace_analyze_blob(self.last_trace_analyze)

    def _format_trace_analyze_blob(self, blob: dict[str, Any] | None) -> str:
        if not blob:
            return "(none)"
        ids = [
            str(e.get("kernel_id"))
            for e in blob.get("hot_kernels_top15", [])
            if isinstance(e, dict) and e.get("kernel_id")
        ]
        reusable = list(blob.get("reusable_native_kernel_ids", []))
        base = (
            f"trace={blob.get('trace_input','?')} "
            f"candidates_path={blob.get('candidates_path','?')} "
            f"top={ids or []} reusable_native={reusable or []}"
        )
        # When there are no routable candidates, surface the skipped
        # operators (id:name:reason) so the LLM sees they were detected but
        # cannot be rewritten -- rather than an empty list that pushes it to
        # echo analysis.md operator names as a (non-unique, invalid)
        # kernel_id. Suppressed when candidates exist to keep the
        # steady-state prompt format stable.
        skipped_suffix = ""
        if not ids:
            sk = blob.get("skipped_kernels_top") or []
            rendered_sk = [
                f"{s.get('kernel_id')}:{s.get('name')}:{s.get('skip_reason') or '?'}"
                for s in sk
                if isinstance(s, dict) and s.get("kernel_id")
            ]
            if rendered_sk:
                skipped_suffix = (
                    f" skipped_kernels_top=[{'; '.join(rendered_sk)}]"
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
        warnings = blob.get("trace_health_warnings") or []
        if not warnings:
            return base + skipped_suffix
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
        return f"{base}{skipped_suffix} warnings=[{'; '.join(rendered)}]"

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


__all__ = ["SharedState", "render_model_arch_compact"]
