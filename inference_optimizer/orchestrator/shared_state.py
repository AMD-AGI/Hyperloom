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
                                `select_kernels` REQUEST `trace_input` param
"""

from __future__ import annotations

import json
import os
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
})

# Mapping from audit-action name to (result-dict key, prompt-display label).
# ``key`` is what we read out of the executor result dict; ``label`` is the
# ``key_metric_kind`` written into each attempt entry so prompt readers
# know how to interpret the number (e.g. ``output_throughput`` vs raw
# ``gain_pct`` vs ``validated_gain_pct``).
_KEY_METRIC_MAP: dict[str, tuple[str, str]] = {
    "baseline":       ("output_throughput", "output_throughput"),
    "profile":        ("output_throughput", "output_throughput"),
    "backends":       ("gain_pct",          "gain_pct"),
    "params":         ("gain_pct",          "gain_pct"),
    "sweep":          ("output_throughput", "output_throughput"),
    "validate_stack": ("gain_pct",          "validated_gain_pct"),
}


@dataclass
class SharedState:
    session_id: str = ""
    model_name: str = ""
    model_path: str = ""
    model_class: str = ""
    framework: str = ""
    gpu_type: str = ""
    kernel_enabled: bool = True
    target_summary: str = ""
    baseline_tput: float = 0.0
    baseline_accuracy: float = 0.0
    baseline_failure_streak: int = 0
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
    current_action: str = ""
    crash_count: int = 0
    pruned_families: list[str] = field(default_factory=list)
    start_ts: str = field(default_factory=_now_iso)
    max_minutes: int = 0
    last_profile_trace: str = ""
    # Server EXTRA_SGLANG_ARGS in effect when last_profile_trace was captured.
    # Orchestration uses this to decide whether re-profiling would change the
    # hot-kernel distribution; identical args means the same trace.
    last_profile_args: str = ""
    last_profile_pmc_summary: str = ""
    last_profile_roofline: str = ""
    last_profile_kernel_breakdown: str = ""
    # Cached result of the most recent `select_kernels` request keyed by
    # `trace_input`. Coordinator short-circuits subsequent identical requests
    # so Orchestration does not waste budget re-analysing the same trace.
    last_select_kernels: dict[str, Any] = field(default_factory=dict)
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
    baseline_attempts: list[dict[str, Any]] = field(default_factory=list)
    profile_attempts: list[dict[str, Any]] = field(default_factory=list)
    backends_attempts: list[dict[str, Any]] = field(default_factory=list)
    params_attempts: list[dict[str, Any]] = field(default_factory=list)
    sweep_attempts: list[dict[str, Any]] = field(default_factory=list)
    validate_stack_attempts: list[dict[str, Any]] = field(default_factory=list)
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
        verification = result.get("verification") or {}
        proposal = result.get("proposal") or {}
        decision = str(proposal.get("decision", ""))
        self.last_kernel_opt = {
            "kernel_id": result.get("kernel_id", ""),
            "decision": decision,
            "reasons": proposal.get("reasons", []),
            "micro_speedup": verification.get("micro_speedup", 0.0),
            "compile_passed": verification.get("compile_passed"),
            "correctness_passed": verification.get("correctness_passed"),
            "best_artifact_path": verification.get("best_artifact_path", ""),
            "ts": _now_iso(),
        }
        kernel_id = str(self.last_kernel_opt.get("kernel_id") or "")
        if not kernel_id:
            return

        entry = dict(self.kernel_opt_attempts.get(kernel_id) or {})
        history = list(entry.get("history") or [])
        history.append({"decision": decision, "ts": self.last_kernel_opt["ts"]})
        history = history[-10:]
        entry["attempts"] = int(entry.get("attempts", 0)) + 1
        if decision == "PARTIAL":
            entry["partial_count"] = int(entry.get("partial_count", 0)) + 1
        elif decision == "KEEP":
            # A successful attempt resets the partial streak; a future
            # regression should not be auto-retired on stale history.
            entry["partial_count"] = 0
        entry["last_decision"] = decision
        entry["last_ts"] = self.last_kernel_opt["ts"]
        entry["history"] = history

        max_partial = _DEFAULT_KERNEL_OPT_MAX_PARTIAL
        env_v = os.environ.get("INFERENCE_OPTIMIZER_KERNEL_OPT_MAX_PARTIAL")
        if env_v:
            try:
                max_partial = max(1, int(env_v))
            except (TypeError, ValueError):
                # Invalid env override; keep _DEFAULT_KERNEL_OPT_MAX_PARTIAL
                # (already assigned above) instead of failing.
                pass

        should_reject = (
            decision == "REVERT"
            or int(entry.get("partial_count", 0)) >= max_partial
        )
        if should_reject:
            if kernel_id not in self.rejected_kernel_ids:
                self.rejected_kernel_ids.append(kernel_id)
            entry["rejected_reason"] = (
                "revert_decision"
                if decision == "REVERT"
                else f"max_partial_attempts_{max_partial}_without_keep"
            )

        self.kernel_opt_attempts[kernel_id] = entry

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

    def record_select_kernels(self, payload: dict[str, Any],
                              result: dict[str, Any]) -> None:
        """Cache the latest select_kernels output keyed by trace_input.

        We persist a wider window than the prompt-visible top5 so that when
        the very top GPU consumers are vendor-binary kernels (Tensile / CK)
        Orchestration still sees lower-ranked but **reusable native** entries
        (e.g. AITER RMSNorm) and can dispatch ``run_optimization`` against
        them instead of looping on rejected ones.
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
        self.last_select_kernels = {
            "trace_input": str(trace_input),
            "candidates_path": str(candidates_path),
            "hot_kernels_top15": summary,
            "reusable_native_kernel_ids": reusable_ids,
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
        )
        lines: list[str] = [
            f"=== Action scores (top {len(rows)} by eff_score, tick={self.tick}) ==="
        ]
        locked_rows: list[tuple[str, str]] = []
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
            f"last_profile_args='{self.last_profile_args}'",
            f"last_profile_roofline={self.last_profile_roofline or '(none)'}",
            f"last_profile_kernel_breakdown={self.last_profile_kernel_breakdown or '(none)'}",
            f"last_select_kernels={self._format_last_select_kernels()}",
            f"params_no_promote_streak={self.params_no_promote_streak}",
            f"params_search={self._format_params_search()}",
            f"backends_search={self._format_backends_search()}",
            f"discovered_flags={self._format_discovered_flags()}",
            f"backend_winners_history={self._format_backend_winners_history()}",
            f"synergy_attempted={len(self.synergy_attempted)} combos",
            f"last_kernel_opt={self._format_last_kernel_opt()}",
            f"rejected_kernel_patches={self._format_rejected_kernel_patches()}",
            f"rejected_kernel_ids={self.rejected_kernel_ids or '(none)'}",
            f"last_baseline={self._format_attempt(self.last_baseline)}",
            f"last_profile={self._format_attempt(self.last_profile)}",
            f"last_backends={self._format_attempt(self.last_backends)}",
            f"last_params={self._format_attempt(self.last_params)}",
            f"last_validate_stack={self._format_attempt(self.last_validate_stack)}",
            f"attempts_history={self._format_attempts_history()}",
            f"last_action_failures={self._format_last_action_failures()}",
            f"tick={int(self.tick or 0)}  "
            f"target_gap_pct={float(self.target_gap_pct or 0.0):.2f}",
            f"stop_reason={self.stop_reason or '(none)'}",
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

    def _format_backend_winners_history(self) -> str:
        if not self.backend_winners_history:
            return "(no explore rounds completed)"
        last = self.backend_winners_history[-3:]
        parts: list[str] = []
        for r in last:
            if not isinstance(r, dict):
                continue
            wn = [w.get("name") for w in (r.get("winners") or [])
                  if isinstance(w, dict)]
            best = r.get("best") or {}
            best_name = (
                best.get("name") if isinstance(best, dict) else None
            )
            parts.append(
                f"{r.get('round_id','?')}({r.get('action','?')}): "
                f"winners={wn or []} best={best_name or '(none)'}"
            )
        suffix = (
            f" [+{len(self.backend_winners_history) - 3} earlier rounds]"
            if len(self.backend_winners_history) > 3 else ""
        )
        return " | ".join(parts) + suffix

    def _format_params_search(self) -> str:
        return self._format_search_state(self.params_search)

    def _format_backends_search(self) -> str:
        return self._format_search_state(self.backends_search)

    @staticmethod
    def _format_search_state(search: dict[str, Any] | None) -> str:
        """Compact one-liner for a *_search dedup ledger (prompt-friendly)."""
        if not search:
            return "(none)"
        accepted = search.get("accepted") or []
        rejected = search.get("rejected") or []
        tested = search.get("tested") or {}
        cursor = search.get("cursor", 0)
        acc_names = [
            str(v.get("name", "")) for v in accepted
            if isinstance(v, dict) and v.get("name")
        ]
        return (
            f"accepted={acc_names or []} rejected={len(rejected)} "
            f"tested={len(tested)} cursor={cursor}"
        )

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

    def _format_last_select_kernels(self) -> str:
        if not self.last_select_kernels:
            return "(none)"
        ids = [
            str(e.get("kernel_id"))
            for e in self.last_select_kernels.get("hot_kernels_top15", [])
            if isinstance(e, dict) and e.get("kernel_id")
        ]
        reusable = list(
            self.last_select_kernels.get("reusable_native_kernel_ids", [])
        )
        return (
            f"trace={self.last_select_kernels.get('trace_input','?')} "
            f"candidates_path={self.last_select_kernels.get('candidates_path','?')} "
            f"top={ids or []} reusable_native={reusable or []}"
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
