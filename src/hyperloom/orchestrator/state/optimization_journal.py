# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Per-session optimization journal — structured JSON record of every KEEP / REVERT / no_promote / skipped decision.

Lives at ``<session_dir>/reports/optimization_journal.json``; rewritten
incrementally (atomic tmp + ``os.replace``) so a mid-session crash leaves a usable artifact.
:meth:`Journal.append_entry` dedups for resume safety.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hyperloom.common.io import atomic_write_text
from hyperloom.common.jsonio import read_json
from hyperloom.common.timeutil import now_iso


log = logging.getLogger(__name__)


# Stable filename so dashboards / report scripts can hard-code it.
JOURNAL_FILENAME: str = "optimization_journal.json"

# Outcome literals consumed by render scripts + KB fact-write hooks.
OUTCOME_KEEP: str = "KEEP"
OUTCOME_REVERT: str = "REVERT"
OUTCOME_NO_PROMOTE: str = "no_promote"
# A step that declined to run. Distinct from ``no_promote``, which readers take
# as "we tried this and it did not pay" and harvest as a direction to stop
# exploring (see ``knowledge.trajectory_reviewer``); a step that never ran is
# evidence of nothing.
OUTCOME_SKIP: str = "skipped"

# Task kinds whose result carries an authoritative per-status verdict the
# journal outcome must follow rather than the coarse dispatcher ``promotable``
# flag (a ``reverted`` patch is promotable yet was rolled back).
_STATUS_DRIVEN_JOURNAL_KINDS: frozenset[str] = frozenset({"integrate_patch"})

# Task kinds whose result can legitimately declare ``was_skipped``. Scoped so
# reusing that key elsewhere cannot silently demote a kept patch.
_SKIPPABLE_JOURNAL_KINDS: frozenset[str] = frozenset({"conc_sweep"})

# The only status meaning the change was adopted into current_best.
_JOURNAL_KEEP_STATUSES: frozenset[str] = frozenset({"kept"})

# Stamped on the result when the anchor gate refused an executor-granted KEEP.
PROMOTION_REFUSED_KEY: str = "promotion_refused"

# Statuses meaning a real change was tested/applied then rolled back or rejected
# on measured grounds → REVERT. Everything else is ``no_promote``.
_JOURNAL_REVERT_STATUSES: frozenset[str] = frozenset({"reverted", "accuracy_unavailable_reject", "regression"})

# Change-kind vocabulary for coarse dashboard grouping.
KIND_BACKEND: str = "backend"  # --attention-backend, kv_cache_dtype, ...
KIND_PARAM: str = "param"  # --max-num-batched-tokens, --gpu-memory-utilization, ...
KIND_ENV: str = "env"  # ROCm / vLLM env vars
KIND_KERNEL_FILE: str = "kernel_file"  # kernel-opt patch on a specific file
KIND_INTEGRATE: str = "integrate"  # framework / patch integration
KIND_BASELINE: str = "baseline"
KIND_PROFILE: str = "profile"
KIND_GEMM_TUNING: str = "gemm_tuning"  # adopted GEMM-tuning run (GEAK / forge)
KIND_OTHER: str = "other"


def _optional_int(value: Any) -> int | None:
    """Coerce a value to int, or ``None`` on absence / bad type.

    Used to load the optional ``tick`` field tolerantly so a journal missing
    or carrying a corrupted value does not crash :meth:`JournalEntry.from_dict`.

    Args:
        value: The value to coerce to an int.

    Returns:
        The coerced int, or ``None`` when the value is absent or not
        convertible.
    """
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass
class JournalEntry:
    """One KEEP / REVERT / no_promote / skipped decision (``None`` distinguishes "not measured" from "measured zero")."""

    phase: str
    iter: int
    kind: str
    change: str
    outcome: str
    gain_pct: float | None = None
    throughput_after: float | None = None
    error_class: str | None = None
    reason: str | None = None
    # Predicted (pre-measurement) gain when the proposer supplied one; ``None``
    # (stripped by ``to_dict``) when no prediction was available.
    predicted_gain_pct: float | None = None
    task_id: str = ""
    variant_name: str = ""
    ts: str = ""
    # Which lever this decision moved (config / source_patch / upstream_pr /
    # enablement). ``phase`` above says when the decision was taken, which stops
    # being evidence of ownership once two lanes share a phase; this says what
    # was changed, which does not.
    lever_kind: str = ""
    # Proposer attribution: ``provenance`` is the raw explore label, ``scope``
    # the specialist dial, ``fingerprint`` the join key into ``explore_search``.
    provenance: str = ""
    scope: str = ""
    fingerprint: str = ""
    # Per-variant measurement detail beyond the headline gain/throughput.
    metrics: dict[str, Any] = field(default_factory=dict)
    # Orchestrator tick at the moment of decision; joins this row to LLM calls
    # for the same tick. ``None`` (not 0) so unknown ticks are stripped.
    tick: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Strip ``None`` values so the file stays compact and JSON-diffable.

        Returns:
            dict[str, Any]: The entry as a dict with ``None``/empty fields
                removed.
        """
        raw = dataclasses.asdict(self)
        # Strip None, empty strings, and empty containers to stay compact.
        return {k: v for k, v in raw.items() if v is not None and v != "" and v != {} and v != []}

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
            predicted_gain_pct=d.get("predicted_gain_pct"),
            task_id=str(d.get("task_id", "")),
            variant_name=str(d.get("variant_name", "")),
            ts=str(d.get("ts", "")),
            provenance=str(d.get("provenance", "")),
            scope=str(d.get("scope", "")),
            fingerprint=str(d.get("fingerprint", "")),
            metrics=dict(d.get("metrics") or {}),
            tick=_optional_int(d.get("tick")),
            lever_kind=str(d.get("lever_kind", "")),
        )

    def dedupe_key(self) -> tuple[str, int, str, str, str, str, str]:
        """Dedup tuple for resume replay (includes variant_name + task_id so same-tick siblings don't collide).

        Returns:
            A tuple of ``(phase, iter, kind, change, outcome, variant_name,
            task_id)`` uniquely identifying this decision for dedup.
        """
        return (
            self.phase,
            self.iter,
            self.kind,
            self.change,
            self.outcome,
            self.variant_name,
            self.task_id,
        )


@dataclass
class Journal:
    """In-memory representation of the journal file (mutations write through to disk before returning)."""

    session_id: str
    model: str
    hardware: str
    framework: str = ""
    baseline_throughput: float = 0.0
    final_throughput: float | None = None
    total_gain_pct: float | None = None
    entries: list[JournalEntry] = field(default_factory=list)
    path: Path = field(default_factory=Path)

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
        """Return the existing journal if on disk, else mint a new one (on-disk header fields win only when the caller leaves them empty).

        Args:
            session_dir: The session directory whose ``reports/`` holds the
                journal file.
            session_id: Session identifier used when minting a new journal.
            model: Model identifier used when minting a new journal.
            hardware: Hardware identifier used when minting a new journal.
            framework: Optional framework identifier for a new journal.
            baseline_throughput: Optional baseline throughput for a new journal.

        Returns:
            The loaded or newly created ``Journal`` instance.
        """
        path = cls._journal_path(session_dir)
        blob: dict[str, Any] = {}
        if path.exists():

            def _warn(exc: BaseException) -> None:
                log.warning("optimization_journal: failed to parse %s (%s); recreating fresh", path, exc)

            blob = read_json(path, default={}, require_dict=True, on_error=_warn)

        entries_raw = blob.get("entries") or []
        entries = [JournalEntry.from_dict(e) for e in entries_raw if isinstance(e, dict)]
        journal = cls(
            session_id=str(blob.get("session_id") or session_id),
            model=str(blob.get("model") or model),
            hardware=str(blob.get("hardware") or hardware),
            framework=str(blob.get("framework") or framework),
            baseline_throughput=float(blob.get("baseline_throughput") or baseline_throughput),
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
        """Append a decision row and flush; ``False`` on duplicate dedupe_key (resume replay safety).

        Args:
            entry: The decision row to append; its ``ts`` is stamped when empty.

        Returns:
            ``True`` when the entry was appended and flushed, ``False`` when a
            row with the same dedupe key already exists.
        """
        if not entry.ts:
            entry.ts = now_iso("seconds", z_suffix=True)
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
        """Update top-level summary fields and flush (called once at CLOSE; partial finalize allowed).

        Args:
            final_throughput: Final measured throughput; left unchanged when
                ``None``.
            total_gain_pct: Total gain percentage over baseline; left unchanged
                when ``None``.
        """
        if final_throughput is not None:
            self.final_throughput = float(final_throughput)
        if total_gain_pct is not None:
            self.total_gain_pct = float(total_gain_pct)
        self._flush()

    def update_baseline(self, baseline_throughput: float) -> None:
        """Late-binding setter for the baseline measurement (no-op on non-positive, to avoid erasing a real value with a stale 0).

        Args:
            baseline_throughput: The measured baseline throughput; ignored when
                non-positive.
        """
        if baseline_throughput and baseline_throughput > 0:
            self.baseline_throughput = float(baseline_throughput)
            self._flush()

    # Persistence
    def _flush(self) -> None:
        """Atomic write (tmp + os.replace); best-effort — IOError logged and swallowed (forensic aid, not a correctness invariant)."""
        try:
            atomic_write_text(
                self.path,
                json.dumps(self.to_dict(), indent=2, sort_keys=False) + "\n",
                make_parents=True,
            )
        except OSError as exc:
            log.warning("optimization_journal flush failed (%s): %s", self.path, exc)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the whole journal (header + entries) to a dict.

        Returns:
            dict[str, Any]: The journal as a JSON-serialisable dict.
        """
        out: dict[str, Any] = {
            "session_id": self.session_id,
            "model": self.model,
            "hardware": self.hardware,
            "framework": self.framework,
            "baseline_throughput": self.baseline_throughput,
            "final_throughput": self.final_throughput,
            "total_gain_pct": self.total_gain_pct,
            "entries": [e.to_dict() for e in self.entries],
        }
        return out


# helpers
def _variant_args(variant: dict[str, Any]) -> str:
    """Read a variant's canonical server-arg string.

    Args:
        variant: The variant dict to read server args from.

    Returns:
        The server-arg string, or an empty string when the key is absent.
    """
    return str(variant.get("extra_server_args") or "")


def derive_journal_outcome(
    task_kind: str,
    result_dict: dict[str, Any] | None,
    *,
    promotable: bool,
) -> str:
    """Derive the journal ``outcome`` for a settled per-task result.

    For ``integrate_patch`` the outcome follows the executor's authoritative
    per-status verdict:

    - ``status == "kept"`` → ``OUTCOME_KEEP``, unless the promote path stamped
      :data:`PROMOTION_REFUSED_KEY` because the anchor gate declined to lift it
    - ``status in {reverted, accuracy_unavailable_reject, regression}`` →
      ``OUTCOME_REVERT``
    - any other status → ``OUTCOME_NO_PROMOTE``

    For every other task kind the binary behaviour applies (``promotable`` →
    KEEP, else REVERT).

    A result from a skippable kind declaring ``was_skipped`` outranks both
    rules. A skip succeeds by design -- a concurrency sweep with no
    optimization to compare has nothing to do and says so -- but a success that
    changed nothing is neither a KEEP nor a measured dead end.

    Args:
        task_kind: The settled task's kind.
        result_dict: The task result dict (``status`` drives the patch kinds).
        promotable: The dispatcher's promotable flag (``_is_promotable_result``).

    Returns:
        One of :data:`OUTCOME_KEEP` / :data:`OUTCOME_REVERT` /
        :data:`OUTCOME_NO_PROMOTE` / :data:`OUTCOME_SKIP`.
    """
    result = result_dict or {}
    kind = (task_kind or "").lower()
    if kind in _SKIPPABLE_JOURNAL_KINDS and result.get("was_skipped"):
        return OUTCOME_SKIP
    if kind in _STATUS_DRIVEN_JOURNAL_KINDS:
        status = str(result.get("status") or "").strip().lower()
        if status in _JOURNAL_KEEP_STATUSES:
            if result.get(PROMOTION_REFUSED_KEY):
                return OUTCOME_NO_PROMOTE
            return OUTCOME_KEEP
        if status in _JOURNAL_REVERT_STATUSES:
            return OUTCOME_REVERT
        return OUTCOME_NO_PROMOTE
    return OUTCOME_KEEP if promotable else OUTCOME_REVERT


def classify_change_kind(task_kind: str, variant: dict[str, Any] | None = None) -> str:
    """Map a task / variant to a ``KIND_*`` value (priority: env-only > kernel_file > integrate > backend > param).

    Args:
        task_kind: The task kind to classify.
        variant: Optional variant dict inspected for server args and envs when
            the task kind alone is not decisive.

    Returns:
        The matching ``KIND_*`` constant.
    """
    kind = (task_kind or "").lower()
    if kind == "kernel_opt":
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


# operation_kind: stable filterable label for "what this step did"; renames the
# two kernel kinds to the action names dashboards/traces filter on.
_OP_KIND_RENAME: dict[str, str] = {
    KIND_KERNEL_FILE: "kernel_opt",
    KIND_INTEGRATE: "kernel_integrate",
}


def operation_kind_for(action: str, kind: str = "") -> str:
    """Map an (action, change-kind) pair to a stable ``operation_kind`` label.

    Examples: explore ``backend`` / ``param`` / ``env``; kernel ``kernel_opt`` /
    ``kernel_integrate``; ``baseline`` / ``profile`` / ``roofline`` / ``sweep``.
    Prefers the fine change-kind, falling back to the action when the kind is
    absent or ``other``.

    Args:
        action: The action name, used as the fallback label.
        kind: The fine change-kind; preferred when present and not ``other``.

    Returns:
        The stable ``operation_kind`` label.
    """
    k = (kind or "").lower()
    if k and k != KIND_OTHER:
        return _OP_KIND_RENAME.get(k, k)
    a = (action or "").lower()
    if a == "kernel_opt":
        return "kernel_opt"
    if a == "integrate":
        return "kernel_integrate"
    return a or KIND_OTHER


def proposer_for(provenance: str) -> str:
    """Map an explore ``provenance`` label to a stable proposer/component name.

    ``specialist:<domain>`` is kept verbatim (so a trace can filter on the exact
    specialist); ``llm_direct`` / ``legacy:*`` / empty collapse to
    ``orchestration``; ``default_grid`` becomes ``grid``.

    Args:
        provenance: The explore provenance label to map.

    Returns:
        The stable proposer/component name.
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
    """Human-readable one-line description used as the ``change`` field (falls back to task kind).

    Args:
        task_kind: The task kind, used as a fallback label.
        variant: Optional variant dict whose server args, envs, or name supply
            the description.
        result_dict: Optional result dict mined for identifiers (kernel id,
            patch path, PR url) when no variant detail is available.

    Returns:
        A one-line human-readable description of the change.
    """
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
    "KIND_GEMM_TUNING",
    "KIND_INTEGRATE",
    "KIND_KERNEL_FILE",
    "KIND_OTHER",
    "KIND_PARAM",
    "KIND_PROFILE",
    "OUTCOME_KEEP",
    "OUTCOME_NO_PROMOTE",
    "OUTCOME_REVERT",
    "OUTCOME_SKIP",
    "PROMOTION_REFUSED_KEY",
    "classify_change_kind",
    "derive_journal_outcome",
    "summarize_change",
]
