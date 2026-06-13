# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Per-session optimization journal — structured JSON record of every KEEP / REVERT / no_promote decision.

Lives at ``<session_dir>/reports/optimization_journal.json``; rewritten
incrementally (atomic tmp + ``os.replace``) so a mid-session crash leaves a usable artifact.
:meth:`Journal.append_entry` dedups for resume safety.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


log = logging.getLogger(__name__)


# Stable filename so dashboards / report scripts can hard-code it.
JOURNAL_FILENAME: str = "optimization_journal.json"

# Outcome literals — keep stable (consumed by render scripts + KB fact-write hooks).
OUTCOME_KEEP:        str = "KEEP"
OUTCOME_REVERT:      str = "REVERT"
OUTCOME_NO_PROMOTE:  str = "no_promote"

# Change-kind vocabulary — coarse dashboard grouping. Extend by appending; don't reuse old strings.
KIND_BACKEND:      str = "backend"      # --attention-backend, kv_cache_dtype, ...
KIND_PARAM:        str = "param"        # --max-num-batched-tokens, --gpu-memory-utilization, ...
KIND_ENV:          str = "env"          # ROCm / vLLM env vars
KIND_KERNEL_FILE:  str = "kernel_file"  # kernel-opt patch on a specific file
KIND_INTEGRATE:    str = "integrate"    # framework PR / patch integration
KIND_BASELINE:     str = "baseline"
KIND_PROFILE:      str = "profile"
KIND_OTHER:        str = "other"


def _optional_int(value: Any) -> int | None:
    """Coerce a value to int, or ``None`` on absence / bad type.

    Used to load the optional ``tick`` field tolerantly: an old journal
    written before D1 has no ``tick`` key, and a corrupted value must not
    crash :meth:`JournalEntry.from_dict` (the journal is a best-effort
    audit artifact loaded on resume).
    """
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass
class JournalEntry:
    """One KEEP / REVERT / no_promote decision (``None`` distinguishes "not measured" from "measured zero")."""

    phase:             str
    iter:              int
    kind:              str
    change:            str
    outcome:           str
    gain_pct:          float | None = None
    throughput_after:  float | None = None
    error_class:       str | None = None
    reason:            str | None = None
    task_id:           str = ""
    variant_name:      str = ""
    ts:                str = ""
    # Proposer attribution (who proposed this change). ``provenance`` is the raw
    # explore label (``llm_direct`` / ``default_grid`` / ``specialist:<domain>``);
    # ``scope`` is the orthogonal specialist dial (domain / domains / freeform);
    # ``fingerprint`` is the variant join key into ``explore_search``. All empty
    # on non-explore rows and on legacy journals (stripped by ``to_dict``).
    provenance:        str = ""
    scope:             str = ""
    fingerprint:       str = ""
    # Per-variant measurement detail beyond the headline gain/throughput
    # (runtime_sec / wall_clock_ratio_vs_baseline / stack_rebench_tput /
    # estimated_output_throughput). Empty dict stripped by ``to_dict``.
    metrics:           dict[str, Any] = field(default_factory=dict)
    # Full-trace D1: orchestrator tick at the moment of decision. Lets the
    # decision-trace collector join this KEEP/REVERT row to the LLM calls
    # recorded for the same tick. Defaults to ``None`` (not 0) so older
    # journals — and call sites that don't know the tick — are stripped by
    # ``to_dict`` and remain indistinguishable from "tick unknown" rather
    # than masquerading as the pre-first-increment tick 0.
    tick:              int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Strip ``None`` values so the file stays compact and JSON-diffable.

        Returns:
            dict[str, Any]: The entry as a dict with ``None``/empty fields
                removed.
        """
        raw = dataclasses.asdict(self)
        # Strip None, empty strings, and empty containers ({} / []) so the file
        # stays compact and byte-diffable (an unset ``metrics`` dict vanishes).
        return {
            k: v for k, v in raw.items()
            if v is not None and v != "" and v != {} and v != []
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> JournalEntry:
        """Reconstruct a :class:`JournalEntry` from a plain dict.

        Args:
            d (dict[str, Any]): The serialised entry.

        Returns:
            JournalEntry: The reconstructed entry with coerced field types.
        """
        return cls(
            phase=str(d.get("phase", "")),
            iter=int(d.get("iter", 0)),
            kind=str(d.get("kind", "")),
            change=str(d.get("change", "")),
            outcome=str(d.get("outcome", "")),
            gain_pct=d.get("gain_pct"),
            throughput_after=d.get("throughput_after"),
            error_class=d.get("error_class"),
            reason=d.get("reason"),
            task_id=str(d.get("task_id", "")),
            variant_name=str(d.get("variant_name", "")),
            ts=str(d.get("ts", "")),
            provenance=str(d.get("provenance", "")),
            scope=str(d.get("scope", "")),
            fingerprint=str(d.get("fingerprint", "")),
            metrics=dict(d.get("metrics") or {}),
            tick=_optional_int(d.get("tick")),
        )

    def dedupe_key(self) -> tuple[str, int, str, str, str, str, str]:
        """Dedup tuple for resume replay (includes variant_name + task_id so same-tick siblings don't collide)."""
        return (
            self.phase, self.iter, self.kind, self.change,
            self.outcome, self.variant_name, self.task_id,
        )


@dataclass
class Journal:
    """In-memory representation of the journal file (mutations write through to disk before returning)."""

    session_id:           str
    model:                str
    hardware:             str
    framework:            str = ""
    baseline_throughput:  float = 0.0
    final_throughput:     float | None = None
    total_gain_pct:       float | None = None
    entries:              list[JournalEntry] = field(default_factory=list)
    path:                 Path = field(default_factory=Path)

    # Construction
    @classmethod
    def load_or_create(
        cls,
        session_dir: Path,
        *,
        session_id: str,
        model: str,
        hardware: str,
        framework: str = "",
        baseline_throughput: float = 0.0,
    ) -> Journal:
        """Return the existing journal if on disk, else mint a new one (on-disk header fields win only when the caller leaves them empty)."""
        path = cls._journal_path(session_dir)
        if path.exists():
            try:
                blob = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                log.warning(
                    "optimization_journal: failed to parse %s (%s); "
                    "recreating fresh", path, exc,
                )
                blob = {}
        else:
            blob = {}

        entries_raw = blob.get("entries") or []
        entries = [
            JournalEntry.from_dict(e) for e in entries_raw if isinstance(e, dict)
        ]
        journal = cls(
            session_id=str(blob.get("session_id") or session_id),
            model=str(blob.get("model") or model),
            hardware=str(blob.get("hardware") or hardware),
            framework=str(blob.get("framework") or framework),
            baseline_throughput=float(
                blob.get("baseline_throughput") or baseline_throughput
            ),
            final_throughput=blob.get("final_throughput"),
            total_gain_pct=blob.get("total_gain_pct"),
            entries=entries,
            path=path,
        )
        return journal

    @staticmethod
    def _journal_path(session_dir: Path) -> Path:
        """Resolve (and create) the journal file path under a session dir.

        Args:
            session_dir (Path): The session directory root.

        Returns:
            Path: The path to the journal file inside ``reports/``.
        """
        reports = Path(session_dir) / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        return reports / JOURNAL_FILENAME

    # Mutation
    def append_entry(self, entry: JournalEntry) -> bool:
        """Append a decision row and flush; ``False`` on duplicate dedupe_key (resume replay safety)."""
        if not entry.ts:
            entry.ts = _now_iso()
        key = entry.dedupe_key()
        for existing in self.entries:
            if existing.dedupe_key() == key:
                return False
        self.entries.append(entry)
        self._flush()
        return True

    def finalize(
        self,
        *,
        final_throughput: float | None = None,
        total_gain_pct: float | None = None,
    ) -> None:
        """Update top-level summary fields and flush (called once at CLOSE; partial finalize allowed)."""
        if final_throughput is not None:
            self.final_throughput = float(final_throughput)
        if total_gain_pct is not None:
            self.total_gain_pct = float(total_gain_pct)
        self._flush()

    def update_baseline(self, baseline_throughput: float) -> None:
        """Late-binding setter for the baseline measurement (no-op on non-positive, to avoid erasing a real value with a stale 0)."""
        if baseline_throughput and baseline_throughput > 0:
            self.baseline_throughput = float(baseline_throughput)
            self._flush()

    # Persistence
    def _flush(self) -> None:
        """Atomic write (tmp + os.replace); best-effort — IOError logged and swallowed (forensic aid, not a correctness invariant)."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = self.to_dict()
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(payload, indent=2, sort_keys=False) + "\n",
                encoding="utf-8",
            )
            os.replace(tmp, self.path)
        except OSError as exc:
            log.warning("optimization_journal flush failed (%s): %s", self.path, exc)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the whole journal (header + entries) to a dict.

        Returns:
            dict[str, Any]: The journal as a JSON-serialisable dict.
        """
        out: dict[str, Any] = {
            "session_id":          self.session_id,
            "model":               self.model,
            "hardware":            self.hardware,
            "framework":           self.framework,
            "baseline_throughput": self.baseline_throughput,
            "final_throughput":    self.final_throughput,
            "total_gain_pct":      self.total_gain_pct,
            "entries":             [e.to_dict() for e in self.entries],
        }
        return out


# helpers
def _now_iso() -> str:
    """ISO-8601 UTC timestamp (seconds precision) used for entry ``ts``.

    Returns:
        str: The current UTC timestamp with a ``Z`` suffix.
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z",
    )


def _variant_args(variant: dict[str, Any]) -> str:
    """Read a variant's server-arg string, canonical (``extra_server_args``) first with a legacy ``extra_sglang_args`` fallback."""
    return str(
        variant.get("extra_server_args")
        or variant.get("extra_sglang_args")
        or ""
    )


def classify_change_kind(task_kind: str, variant: dict[str, Any] | None = None) -> str:
    """Map a task / variant to a ``KIND_*`` value (priority: env-only > kernel_file > integrate > backend > param)."""
    kind = (task_kind or "").lower()
    if kind in ("kernel_opt", "deep_kernel_analysis", "operator_tuning"):
        return KIND_KERNEL_FILE
    if kind == "integrate":
        return KIND_INTEGRATE
    if kind == "baseline":
        return KIND_BASELINE
    if kind == "profile":
        return KIND_PROFILE
    if isinstance(variant, dict):
        args = _variant_args(variant)
        if variant.get("extra_envs") and not args:
            return KIND_ENV
        if "--attention-backend" in args or "kv-cache-dtype" in args:
            return KIND_BACKEND
        if args:
            return KIND_PARAM
    return KIND_OTHER


# operation_kind: a single stable, filterable label for "what this step did".
# Reuses the change-kind vocabulary but renames the two kernel kinds to the
# action names dashboards/traces filter on, and falls back to the raw action
# for non-explore steps (baseline / profile / roofline / sweep / framework_pr).
_OP_KIND_RENAME: dict[str, str] = {
    KIND_KERNEL_FILE: "kernel_opt",
    KIND_INTEGRATE:   "kernel_integrate",
}


def operation_kind_for(action: str, kind: str = "") -> str:
    """Map an (action, change-kind) pair to a stable ``operation_kind`` label.

    Examples: explore ``backend`` / ``param`` / ``env``; kernel ``kernel_opt`` /
    ``kernel_integrate``; ``baseline`` / ``profile`` / ``roofline`` / ``sweep``.
    Prefers the fine change-kind, falling back to the action when the kind is
    absent or ``other``.
    """
    k = (kind or "").lower()
    if k and k != KIND_OTHER:
        return _OP_KIND_RENAME.get(k, k)
    a = (action or "").lower()
    if a in ("kernel_opt", "deep_kernel_analysis", "operator_tuning"):
        return "kernel_opt"
    if a == "integrate":
        return "kernel_integrate"
    return a or KIND_OTHER


def proposer_for(provenance: str) -> str:
    """Map an explore ``provenance`` label to a stable proposer/component name.

    ``specialist:<domain>`` is kept verbatim (so a trace can filter on the exact
    specialist); ``llm_direct`` / ``legacy:*`` / empty collapse to
    ``orchestration``; ``default_grid`` becomes ``grid``.
    """
    p = (provenance or "").strip()
    if not p or p == "llm_direct" or p.startswith("legacy:"):
        return "orchestration"
    if p == "default_grid":
        return "grid"
    return p


def summarize_change(
    task_kind: str,
    variant: dict[str, Any] | None = None,
    result_dict: dict[str, Any] | None = None,
) -> str:
    """Human-readable one-line description used as the ``change`` field (falls back to task kind)."""
    if isinstance(variant, dict):
        name = str(variant.get("name") or "").strip()
        args = _variant_args(variant).strip()
        envs = variant.get("extra_envs") or {}
        if envs and isinstance(envs, dict):
            env_str = " ".join(f"{k}={v}" for k, v in envs.items())
            if args:
                return f"{args} | env: {env_str}"
            return f"env: {env_str}"
        if args:
            return args
        if name:
            return name
    if isinstance(result_dict, dict):
        for key in ("kernel_id", "patch_path", "pr_url"):
            v = result_dict.get(key)
            if v:
                return f"{task_kind}: {v}"
    return task_kind or "(unknown)"


__all__ = [
    "JOURNAL_FILENAME",
    "Journal",
    "JournalEntry",
    "KIND_BACKEND",
    "KIND_BASELINE",
    "KIND_ENV",
    "KIND_INTEGRATE",
    "KIND_KERNEL_FILE",
    "KIND_OTHER",
    "KIND_PARAM",
    "KIND_PROFILE",
    "OUTCOME_KEEP",
    "OUTCOME_NO_PROMOTE",
    "OUTCOME_REVERT",
    "classify_change_kind",
    "summarize_change",
]
