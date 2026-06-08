# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Per-session optimization journal — structured JSON record of every
KEEP / REVERT / no_promote decision taken during an
``inference_optimizer optimize`` run.

The journal lives at ``<session_dir>/reports/optimization_journal.json``
and is rewritten *incrementally* after each decision (atomic tmp +
``os.replace``) so a crash mid-session still leaves a usable artifact.

Shape (stable contract; downstream dashboards may depend on these
field names):

.. code-block:: json

    {
      "session_id": "...",
      "model": "DeepSeek-R1",
      "hardware": "MI300X",
      "framework": "sglang",
      "baseline_throughput": 603.6,
      "final_throughput": 875.0,
      "total_gain_pct": 44.9,
      "entries": [
        {
          "phase": "EXPLORE",
          "iter": 1,
          "kind": "backend",
          "change": "--attention-backend ROCM_AITER_UNIFIED_ATTN",
          "outcome": "KEEP",
          "gain_pct": 12.3,
          "throughput_after": 678.0,
          "ts": "2026-05-26T08:48:00Z"
        },
        {
          "phase": "EXPLORE",
          "iter": 4,
          "kind": "env",
          "change": "VLLM_ROCM_USE_AITER_FP4BMM=1",
          "outcome": "REVERT",
          "error_class": "crash",
          "reason": "gfx942 不支持",
          "ts": "..."
        }
      ]
    }

Design rationale:

* **JSON over markdown** — structured so a downstream tool can diff /
  filter / aggregate across sessions. Render-to-markdown is a separate
  concern (a future ``render_journal.py`` script).
* **Append-only entries** — every decision lands as one entry; finalize
  only mutates the top-level summary fields.
* **No KB dependency** — works even with ``--degraded-kb`` /
  ``--no-fact-writes``; the journal is local-only so operators always
  get a session report.
* **Idempotency** — :meth:`Journal.append_entry` dedupes by
  ``(phase, iter, kind, change, outcome)`` so a resume that replays the
  event log doesn't double-write entries.
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

# Outcome literals — keep stable; consumed by render scripts and KB
# fact-write hooks (KEEP → propose_lesson; REVERT → maybe propose_pitfall).
OUTCOME_KEEP:        str = "KEEP"
OUTCOME_REVERT:      str = "REVERT"
OUTCOME_NO_PROMOTE:  str = "no_promote"

# Change-kind vocabulary — coarse classification so a downstream
# dashboard can group entries without parsing free-text. Extend by
# appending; don't reuse old strings.
KIND_BACKEND:      str = "backend"      # --attention-backend, kv_cache_dtype, ...
KIND_PARAM:        str = "param"        # --max-num-batched-tokens, --gpu-memory-utilization, ...
KIND_ENV:          str = "env"          # ROCm / vLLM env vars
KIND_KERNEL_FILE:  str = "kernel_file"  # kernel-opt patch on a specific file
KIND_INTEGRATE:    str = "integrate"    # framework PR / patch integration
KIND_BASELINE:     str = "baseline"
KIND_PROFILE:      str = "profile"
KIND_OTHER:        str = "other"


@dataclass
class JournalEntry:
    """One row in the journal — exactly one KEEP / REVERT / no_promote
    decision recorded by the Coordinator.

    All fields are JSON-serialisable; ``None`` is preserved so a
    consumer can distinguish "not measured" from "measured zero".
    """

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

    def to_dict(self) -> dict[str, Any]:
        """Strip ``None`` values so the file stays compact and JSON-diffable.

        Returns:
            dict[str, Any]: The entry as a dict with ``None``/empty fields
                removed.
        """
        raw = dataclasses.asdict(self)
        return {k: v for k, v in raw.items() if v is not None and v != ""}

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
        )

    def dedupe_key(self) -> tuple[str, int, str, str, str, str, str]:
        """Tuple used by :meth:`Journal.append_entry` to skip duplicates
        on resume replay.

        Both ``variant_name`` and ``task_id`` participate in the key:

        * ``variant_name`` — an explore round emits multiple entries per
          ``(phase, iter)`` that differ only by which variant was tried.
        * ``task_id`` — two independent non-explore tasks scheduled in
          the same tick (e.g. two baseline / profile / kernel_opt runs)
          legitimately collide on ``(phase, iter, kind, change, outcome)``
          when :func:`summarize_change` falls back to the task kind
          string; without ``task_id`` in the key the second entry would
          be silently dropped as a "resume replay".

        Returns:
            tuple[str, int, str, str, str, str, str]: The dedupe key
                ``(phase, iter, kind, change, outcome, variant_name, task_id)``.
        """
        return (
            self.phase, self.iter, self.kind, self.change,
            self.outcome, self.variant_name, self.task_id,
        )


@dataclass
class Journal:
    """In-memory representation of the journal file.

    Construct via :meth:`load_or_create`; mutate via
    :meth:`append_entry` / :meth:`finalize`; both methods write through
    to disk before returning so a crash never loses state.
    """

    session_id:           str
    model:                str
    hardware:             str
    framework:            str = ""
    baseline_throughput:  float = 0.0
    final_throughput:     float | None = None
    total_gain_pct:       float | None = None
    entries:              list[JournalEntry] = field(default_factory=list)
    path:                 Path = field(default_factory=Path)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
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
        """Return the existing journal if one is on disk, else mint a new
        one.

        Header fields (``session_id`` / ``model`` / ``hardware`` /
        ``framework`` / ``baseline_throughput``) are updated from the
        on-disk file's defaults *only* when the caller leaves them empty,
        so a resume call that doesn't know the baseline yet won't blow
        away an earlier write that captured it.

        Args:
            session_dir (Path): The session directory root.
            session_id (str): Session identifier.
            model (str): Model name.
            hardware (str): Hardware label.
            framework (str): Framework name. Defaults to ``""``.
            baseline_throughput (float): Baseline throughput. Defaults to
                ``0.0``.

        Returns:
            Journal: The loaded or newly created journal.
        """
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

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    def append_entry(self, entry: JournalEntry) -> bool:
        """Append a decision row and flush to disk.

        Returns ``True`` when the entry was new (and the file was
        rewritten), ``False`` when a duplicate dedupe_key was found
        (resume replay safety).

        Args:
            entry (JournalEntry): The decision row to append; its ``ts`` is
                populated when empty.

        Returns:
            bool: ``True`` when the entry was new and persisted, ``False`` on a
                duplicate dedupe key.
        """
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
        """Update top-level summary fields and flush.

        Intended to be called once at CLOSE (T4). Both arguments are
        optional so a partial finalize (e.g. only ``final_throughput``
        known) is allowed.

        Args:
            final_throughput (float | None): Final measured throughput.
            total_gain_pct (float | None): Total gain percentage vs baseline.
        """
        if final_throughput is not None:
            self.final_throughput = float(final_throughput)
        if total_gain_pct is not None:
            self.total_gain_pct = float(total_gain_pct)
        self._flush()

    def update_baseline(self, baseline_throughput: float) -> None:
        """Late-binding setter for the baseline measurement.

        T0 typically runs before the baseline action completes, so the
        constructor receives ``0.0`` and we backfill once the baseline
        executor finishes. No-op when the new value is non-positive
        (avoids erasing a real measurement with a stale 0).

        Args:
            baseline_throughput (float): The measured baseline throughput;
                ignored when non-positive.
        """
        if baseline_throughput and baseline_throughput > 0:
            self.baseline_throughput = float(baseline_throughput)
            self._flush()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _flush(self) -> None:
        """Atomic write of the whole journal to disk.

        Uses ``tmp + os.replace`` so a reader never observes a partially
        written file. Best-effort: an IOError is logged at warning
        level and swallowed — the journal is a forensic aid, not a
        correctness invariant, and the coordinator must not abort
        a session because of a disk hiccup.
        """
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


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _now_iso() -> str:
    """ISO-8601 UTC timestamp (seconds precision) used for entry ``ts``.

    Returns:
        str: The current UTC timestamp with a ``Z`` suffix.
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z",
    )


def _variant_args(variant: dict[str, Any]) -> str:
    """Read a variant's server-arg string, canonical-first.

    The field was renamed ``extra_sglang_args`` -> ``extra_server_args``
    (framework-neutral; see compat.payload_aliases). Explore / stack
    ledger entries now carry the canonical key, so reading only the
    legacy name would silently classify every param/backend variant as
    ``KIND_OTHER`` and emit an empty journal ``change`` string. Read
    canonical first, keep a read-only legacy fallback for any
    pre-rename caller.

    Args:
        variant (dict[str, Any]): The variant descriptor.

    Returns:
        str: The server-arg string, or ``""`` when none is present.
    """
    return str(
        variant.get("extra_server_args")
        or variant.get("extra_sglang_args")
        or ""
    )


def classify_change_kind(task_kind: str, variant: dict[str, Any] | None = None) -> str:
    """Map a task / variant to one of the ``KIND_*`` vocab values.

    Coarse heuristic; the inputs are noisy (variants can mix backend +
    param + env in one cell) so we pick the most prominent dimension
    in the order: env-only > kernel_file > integrate > backend > param.

    Args:
        task_kind (str): The task kind/action name.
        variant (dict[str, Any] | None): Optional explore variant descriptor.

    Returns:
        str: One of the ``KIND_*`` vocabulary values.
    """
    kind = (task_kind or "").lower()
    if kind in ("kernel_opt", "deep_kernel_analysis", "operator_tuning"):
        return KIND_KERNEL_FILE
    if kind == "integrate":
        return KIND_INTEGRATE
    if kind == "baseline":
        return KIND_BASELINE
    if kind == "profile":
        return KIND_PROFILE
    # explore variants — peek inside to refine.
    if isinstance(variant, dict):
        args = _variant_args(variant)
        if variant.get("extra_envs") and not args:
            return KIND_ENV
        if "--attention-backend" in args or "kv-cache-dtype" in args:
            return KIND_BACKEND
        if args:
            return KIND_PARAM
    return KIND_OTHER


def summarize_change(
    task_kind: str,
    variant: dict[str, Any] | None = None,
    result_dict: dict[str, Any] | None = None,
) -> str:
    """Human-readable one-line description of what was changed.

    Caller-friendly summary used as the ``change`` field. Falls back to
    the task kind when nothing more informative is available.

    Args:
        task_kind (str): The kind of task being summarized; used as the
            fallback description.
        variant (dict[str, Any] | None): Optional variant dict whose name,
            args, and ``extra_envs`` are preferred for the summary.
        result_dict (dict[str, Any] | None): Optional result dict scanned for
            ``kernel_id`` / ``patch_path`` / ``pr_url`` when no variant info is
            available.

    Returns:
        str: A one-line human-readable summary of the change.
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
