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
            f"stop_reason={self.stop_reason or '(none)'}",
        ]
        return "\n".join(lines)


__all__ = ["SharedState"]
