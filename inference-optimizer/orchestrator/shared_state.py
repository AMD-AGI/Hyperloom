"""SharedState — DESIGN v0.6 §8.3 / §17.2.

Persistent session-level state that all reactors read (via prompt injection)
and that PolicyGate uses to enforce CORE_STATE_FIELDS guards.

Backed by JSON at ``$SESSION_DIR/state.json``. The file write is atomic
(``tmp`` + ``os.replace``) so concurrent readers never see a partial blob.
The Conductor is the **only** writer; LLM agents go through
``UPDATE_STATE`` intents which the Conductor validates + persists.

v0.6 fields:

    session_id          str   — set by Conductor at session creation
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
    last_profile_trace  str   — set by Conductor when `profile` returns a
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


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


@dataclass
class SharedState:
    session_id: str = ""
    model_name: str = ""
    model_path: str = ""
    model_class: str = ""
    target_summary: str = ""
    baseline_tput: float = 0.0
    baseline_accuracy: float = 0.0
    current_best: dict[str, Any] = field(default_factory=dict)
    cumulative_gain: float = 0.0
    stop_reason: str = ""
    current_action: str = ""
    crash_count: int = 0
    pruned_families: list[str] = field(default_factory=list)
    start_ts: str = field(default_factory=_now_iso)
    max_minutes: int = 0
    last_profile_trace: str = ""
    # Kernel-opt response tracking — Conductor records the most recent
    # `run_optimization_done` so Orch sees what's been tried and doesn't
    # re-dispatch the same kernel_id every tick.
    last_kernel_opt: dict[str, Any] = field(default_factory=dict)
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
    # Conductor is still the only writer to state.json.
    params_search: dict[str, Any] = field(default_factory=dict)

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
        return cls(**{k: v for k, v in raw.items() if k in known})

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
    # Mutators (used by the Conductor only — LLM agents go via intents)
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

        ``allow_core=True`` is reserved for Conductor-internal callers that
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
        return (
            f"kernel_id={ko.get('kernel_id','?')} "
            f"decision={ko.get('decision','?')} "
            f"speedup={ko.get('micro_speedup','?')}"
        )

    def record_kernel_opt(self, result: dict[str, Any]) -> None:
        """Capture the result returned by kernel_optimization_handler so the
        next Orch turn knows what's already been tried (and the outcome)."""
        if not isinstance(result, dict):
            return
        verification = result.get("verification") or {}
        proposal = result.get("proposal") or {}
        self.last_kernel_opt = {
            "kernel_id": result.get("kernel_id", ""),
            "decision": proposal.get("decision", ""),
            "reasons": proposal.get("reasons", []),
            "micro_speedup": verification.get("micro_speedup", 0.0),
            "compile_passed": verification.get("compile_passed"),
            "correctness_passed": verification.get("correctness_passed"),
            "best_artifact_path": verification.get("best_artifact_path", ""),
            "ts": _now_iso(),
        }

    def push_params_winner(self, *, action: str, variant_name: str,
                           tput: float, gain_pct: float, max_history: int = 10) -> None:
        """Append one round's winner to the rolling history buffer."""
        self.params_winner_history.append({
            "action": action,
            "variant_name": variant_name,
            "tput": float(tput) if tput is not None else 0.0,
            "gain_pct": float(gain_pct) if gain_pct is not None else 0.0,
            "ts": _now_iso(),
        })
        # Trim oldest entries if we're past max_history.
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

    def to_prompt_summary(self) -> str:
        """Compact, human-readable snapshot for prompt injection (DESIGN §8.3)."""
        lines = [
            f"session_id={self.session_id or '(unset)'}",
            f"model={self.model_name or '(unset)'}  class={self.model_class or '(unset)'}",
            f"baseline_tput={self.baseline_tput}  baseline_acc={self.baseline_accuracy}",
            f"current_best={self.current_best or '(none)'}",
            f"cumulative_gain={self.cumulative_gain}%",
            f"current_action={self.current_action or '(idle)'}",
            f"crash_count={self.crash_count}",
            f"pruned_families={self.pruned_families or '(none)'}",
            f"last_profile_trace={self.last_profile_trace or '(none)'}",
            f"params_no_promote_streak={self.params_no_promote_streak}",
            f"params_search={self._format_params_search()}",
            f"last_kernel_opt={self._format_last_kernel_opt()}",
            f"stop_reason={self.stop_reason or '(none)'}",
        ]
        return "\n".join(lines)

    def _format_params_search(self) -> str:
        if not self.params_search:
            return "(none)"
        accepted = self.params_search.get("accepted") or []
        rejected = self.params_search.get("rejected") or []
        tested = self.params_search.get("tested") or {}
        cursor = self.params_search.get("cursor", 0)
        acc_names = [
            str(v.get("name", "")) for v in accepted
            if isinstance(v, dict) and v.get("name")
        ]
        return (
            f"accepted={acc_names or []} rejected={len(rejected)} "
            f"tested={len(tested)} cursor={cursor}"
        )


__all__ = ["SharedState"]
