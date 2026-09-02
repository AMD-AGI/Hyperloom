# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Write-side of the breakdown recorder.

A :class:`Recorder` lets the code that *produces* a fact record it at author
time, into a per-session spool directory, instead of having the exporter
re-walk heterogeneous artifacts later. Each producer owns its own files:

* :meth:`Recorder.record_singleton` — one final dict per section; the owner
  overwrites its own stable file (safe: single writer of that file).
* :meth:`Recorder.record_item` — one fragment per event; uniquely named so
  concurrent producers never collide. Pass ``key`` for an idempotent
  (overwrite-on-rewrite) item that survives resume without duplicating.
* :meth:`Recorder.record_upsert_singleton` / :meth:`Recorder.record_upsert_item`
  — the same, but merged into the prior fragment payload instead of replacing
  it, for producers that emit a fact in several partial updates.

Every write lands atomically (tmp + ``os.replace``) and filenames are unique
per (section, producer), so :meth:`Recorder.record_item` and
:meth:`Recorder.record_singleton` are safe across processes and on network
filesystems (no shared-append dependency). The ``record_upsert_*`` methods are
NOT: they read the current fragment, merge, and rewrite it under an in-process
lock only, so two processes upserting the same (section, producer, key) can
lose one side of the merge. Keep every upsert for a given fragment in one
process -- today the coordinator is the only writer, and
``test_breakdown_recorder_no_subprocess_writers`` keeps it that way.

Every write funnels through :meth:`Recorder._write`, which is where the write
trace is emitted; ``HYPERLOOM_BREAKDOWN_TRACE=1`` turns it on (see
:mod:`.trace`).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Literal, Mapping

from hyperloom.common.io import atomic_write_text
from hyperloom.common.timeutil import now_iso

from .trace import trace_enabled, trace_write

SectionShape = Literal["item", "singleton"]

# Per-section wire-shape registry for the breakdown recorder.
#
# Each ``session_breakdown.json`` section has exactly one owning producer, so
# there is never cross-producer write contention. A section is one of:
#
# * ``singleton`` — one final dict; the owner rewrites its own file on update
#   (last write by ``ts`` wins at assembly time).
# * ``item`` — an event stream assembled into a list. The v4 entity streams
#   (phase_transitions, subjects, operations, measurements, adoptions,
#   artifacts, trace_events) are ordered by ``ts`` then ``seq`` and deep-merged
#   by stable entity id, so repeated partial updates for one id collapse into a
#   single record; every other item section stays append-only and is
#   concatenated in ``seq`` then ``ts`` order.
#
# Derived sections (see ``DERIVED_SECTIONS``) are NOT written by producers
# during the run; they are computed at finalize from in-memory ``SharedState``
# (the Coordinator owns every input), so they never appear as fragments.
#
# Producer-written sections and their fragment shape. Payloads match the
# corresponding ``schema.py`` TypedDict so assembly is structure-preserving.
SECTION_SHAPES: dict[str, SectionShape] = {
    # Session Breakdown v4 canonical author-time streams. These names are
    # intentionally separate from the legacy v2/v3 sections so the live v4
    # builder can consume a closed set of SDK-authored facts.
    "run_snapshot": "singleton",
    "phase_transitions": "item",
    "subjects": "item",
    "operations": "item",
    "measurements": "item",
    "adoptions": "item",
    "artifacts": "item",
    "trace_events": "item",
    "session": "singleton",
    "workload": "singleton",
    "baseline": "singleton",
    "final": "singleton",
    "phase_timeline": "item",
    "geak_invocations": "item",
    "forge_invocations": "item",
    "kernel_lifecycle": "singleton",
    "explore_search": "singleton",
    "sweep": "singleton",
    "critic_robustness": "singleton",
    # Author-time item substreams composed into the ``critic_robustness``
    # singleton at assembly (recorded per-iteration so the backend's workdir
    # pruning never erases history).
    "critic_iterations": "item",
    "robustness_signals": "item",
    "telemetry": "singleton",
    "kb_provenance": "singleton",
    "specialist_runs": "item",
    "optimization_stack": "item",
    "kernel_roofline": "singleton",
    "kernel_optimization_summary": "singleton",
    "conc_sweep_summary": "singleton",
    "roofline": "item",
    "roofline_progress": "singleton",
    # Kernel-major lifecycle substreams. Recorded by their respective owners at
    # author time and folded into the ``kernel_journey`` view at assembly (same
    # compose-on-read pattern as ``critic_robustness``); none of these leak into
    # the breakdown envelope on their own.
    "kernel_discovery": "item",  # one per hot-kernel discovery run (tracelens/roofline)
    "kernel_dispatch": "item",  # one per kernel: dispatched? which backends?
    "kernel_backend_result": "item",  # one per backend attempt
    "kernel_e2e": "item",  # one per kernel: e2e integrate gain
    # Authoritative external-tool versions (geak/tracelens/claude/codex/...),
    # one item per tool (idempotent by tool name); folded into the top-level
    # ``versions`` map at assembly.
    "versions": "item",
    # SBD v6 KERNEL substreams. One ``kernel_event`` fragment per event holds
    # its mapping-shaped blocks; every row-shaped fact owns a section of its
    # own, one fragment per row keyed by its real id, because ``_merge_lists``
    # only merges nested rows in place by ``_ENTITY_ID_FIELDS`` and appends
    # anything else -- a partial update to a row nested in the event-level
    # fragment would silently duplicate it. Assembled into one ``kernel``
    # timeline event when the phase is left.
    "kernel_event": "item",  # one per KERNEL phase entry, keyed by event id
    "kernel_lane_run": "item",  # one per forge candidate, ``lane`` discriminates
    "kernel_rebench_attempt": "item",  # one per re-measurement, forge and geak alike
    "kernel_trace_analyze": "item",  # one per analysis the phase requested for itself
    "kernel_geak_attempt": "item",  # one per kernel geak considered
    "kernel_geak_discovery": "item",  # one per geak hot-kernel discovery run
    "kernel_geak_acceptance": "item",  # one per kernel or env selection geak accepted
    # SBD v6 roofline substreams. Written by the same executor whether it was
    # dispatched on its own or called inline by a phase that owns an event, and
    # the only difference between the two is the event id on the rows: the
    # sections are the same either way, and which wire position they assemble
    # into is the assembler's to decide.
    "roofline_event": "item",  # one per roofline event, holding its timeline sequence
    "roofline_action": "item",  # one per roofline action, keyed by its task id
    "roofline_profile_run": "item",  # one per profile attempt within an action
    "roofline_analysis_run": "item",  # one per trace-analysis attempt within an action
}

# Sections computed at finalize from in-memory state, never written as
# fragments.
DERIVED_SECTIONS: frozenset[str] = frozenset(
    {
        "capability_summary",
        "attribution",
        "phase_segments",
        "source_files",
    }
)


def section_shape(section: str) -> SectionShape | None:
    """Return the declared shape for ``section`` (``None`` if unregistered).

    Args:
        section: The breakdown section name to look up.

    Returns:
        The declared section shape (``"item"`` / ``"singleton"``), or ``None``
        when the section is not registered.
    """
    return SECTION_SHAPES.get(section)


log = logging.getLogger(__name__)

_SANITIZE = re.compile(r"[^A-Za-z0-9._-]+")
_ENTITY_ID_FIELDS = (
    "attempt_id",
    "substep_id",
    "gate_id",
    "decision_id",
    "relation_id",
    "measurement_id",
    "artifact_id",
    "adoption_id",
    "subject_id",
    "operation_id",
)


def _slug(value: str) -> str:
    """Filesystem-safe token; empty input collapses to ``unknown``.

    Args:
        value: The raw string to sanitise into a filesystem-safe token.

    Returns:
        The sanitised token, or ``"unknown"`` when the input is empty.
    """
    s = _SANITIZE.sub("-", str(value or "").strip())
    return s.strip("-.") or "unknown"


def _merge_mappings(
    current: Mapping[str, Any],
    update: Mapping[str, Any],
) -> dict[str, Any]:
    """Recursively merge a partial entity update into its current payload."""
    merged = dict(current)
    for key, value in update.items():
        previous = merged.get(key)
        if isinstance(previous, Mapping) and isinstance(value, Mapping):
            merged[key] = _merge_mappings(previous, value)
        elif isinstance(previous, list) and isinstance(value, list):
            merged[key] = _merge_lists(previous, value)
        else:
            merged[key] = value
    return merged


def _merge_lists(current: list[Any], update: list[Any]) -> list[Any]:
    """Merge stable nested entities while retaining unrelated list entries."""
    merged = list(current)
    indexes: dict[tuple[str, str], int] = {}
    for index, value in enumerate(merged):
        if not isinstance(value, Mapping):
            continue
        identity = next(
            ((field, str(value[field])) for field in _ENTITY_ID_FIELDS if value.get(field)),
            None,
        )
        if identity:
            indexes[identity] = index
    for value in update:
        identity = (
            next(
                ((field, str(value[field])) for field in _ENTITY_ID_FIELDS if value.get(field)),
                None,
            )
            if isinstance(value, Mapping)
            else None
        )
        index = indexes.get(identity) if identity else None
        if index is not None and isinstance(merged[index], Mapping):
            merged[index] = _merge_mappings(merged[index], value)
        elif value not in merged:
            merged.append(dict(value) if isinstance(value, Mapping) else value)
            if identity:
                indexes[identity] = len(merged) - 1
    return merged


class Recorder:
    """Per-(session, producer) writer of breakdown record fragments."""

    def __init__(self, parts_dir: Path | str, *, producer: str) -> None:
        """Initialize a recorder writing into ``parts_dir`` for ``producer``.

        Args:
            parts_dir (Path | str): the spool directory fragments are written
                into.
            producer (str): the producer label owning the written fragments
                (sanitized into a filesystem-safe slug).
        """
        self._dir = Path(parts_dir)
        self._producer = _slug(producer)
        self._seq = 0
        self._lock = threading.RLock()

    @property
    def producer(self) -> str:
        """Return the sanitized producer slug owning this recorder's fragments.

        Returns:
            The sanitized producer slug.
        """
        return self._producer

    @property
    def parts_dir(self) -> Path:
        """Return the spool directory fragments are written into.

        Returns:
            The spool directory path.
        """
        return self._dir

    def _next_seq(self) -> int:
        """Return the next monotonically increasing per-recorder sequence number.

        Returns:
            int: the next sequence number (thread-safe).
        """
        with self._lock:
            self._seq += 1
            return self._seq

    def record_singleton(
        self,
        section: str,
        payload: Mapping[str, Any],
    ) -> Path:
        """Write/overwrite this producer's single final blob for ``section``.

        Args:
            section: The breakdown section name (must be declared
                ``singleton``-shaped).
            payload: The final payload mapping for the section.

        Returns:
            The path of the written singleton fragment.
        """
        self._check_shape(section, "singleton")
        filename = f"{_slug(section)}__{self._producer}.json"
        return self._write(section, "singleton", payload, filename=filename)

    def record_upsert_singleton(
        self,
        section: str,
        payload: Mapping[str, Any],
    ) -> Path:
        """Merge and atomically rewrite this producer's singleton fragment."""
        self._check_shape(section, "singleton")
        filename = f"{_slug(section)}__{self._producer}.json"
        target = self._dir / filename
        with self._lock:
            previous: Mapping[str, Any] | None = None
            try:
                current = json.loads(target.read_text(encoding="utf-8"))
                current_payload = current.get("payload") if isinstance(current, dict) else None
                if isinstance(current_payload, Mapping):
                    previous = current_payload
                    merged = _merge_mappings(current_payload, payload)
                else:
                    merged = dict(payload)
            except (OSError, ValueError, TypeError):
                merged = dict(payload)
            return self._write(
                section,
                "singleton",
                merged,
                filename=filename,
                operation="upsert",
                previous=previous,
            )

    def record_item(
        self,
        section: str,
        payload: Mapping[str, Any],
        *,
        key: str | None = None,
    ) -> Path:
        """Append one event fragment to the ``section`` stream.

        ``key`` (optional): a stable per-item identity. When given the fragment
        filename is derived from it, so re-recording the same key overwrites
        rather than duplicates (idempotent across retries / resume).

        Args:
            section: The breakdown section name (must be declared
                ``item``-shaped).
            payload: The event fragment payload mapping.
            key: Optional stable per-item identity for idempotent rewrites;
                when omitted a pid/sequence-unique filename is used.

        Returns:
            The path of the written item fragment.
        """
        self._check_shape(section, "item")
        seq: int | None = None
        if key:
            filename = self._stable_item_filename(section, key)
        else:
            # One number serves both the filename and the envelope: someone
            # reading ``seq=N`` in a trace line must be able to find the file
            # that write produced.
            seq = self._next_seq()
            filename = f"{_slug(section)}__{self._producer}__{os.getpid()}-{seq:06d}.json"
        return self._write(section, "item", payload, filename=filename, seq=seq)

    def _stable_item_filename(self, section: str, key: str) -> str:
        """Name the fragment file that holds ``key``'s item in ``section``.

        ``_slug`` folds every character outside ``[A-Za-z0-9._-]`` to a dash, so
        ``a/b``, ``a:b`` and ``a b`` all name the same file and the last writer
        silently wins. A digest of the untouched key keeps the readable part
        readable while making the name injective.

        Fragments written before this digest existed keep their old name: a
        resumed session must go on updating the file it already wrote, not
        start a second one for the same key. That reuse is only safe when the
        key survived sanitizing untouched -- otherwise the legacy file could
        belong to any of the keys that fold onto that name, and adopting it
        would merge two entities, which is the bug the digest exists to stop.

        Args:
            section (str): The breakdown section name.
            key (str): The caller's stable item identity, unsanitized.

        Returns:
            str: The fragment filename to write within the spool directory.
        """
        slug = _slug(key)
        prefix = f"{_slug(section)}__{self._producer}__{slug}"
        digest = hashlib.sha256(key.encode("utf-8", errors="replace")).hexdigest()[:8]
        filename = f"{prefix}-{digest}.json"
        legacy = self._dir / f"{prefix}.json"
        if slug != key:
            if legacy.exists():
                log.warning(
                    "breakdown recorder: not reusing %s for key %r -- the name is "
                    "ambiguous after sanitizing; writing %s instead. The legacy "
                    "fragment stays on disk and may hold a different key.",
                    legacy.name,
                    key,
                    filename,
                )
            return filename
        if not (self._dir / filename).exists() and legacy.exists():
            return legacy.name
        return filename

    def record_upsert_item(
        self,
        section: str,
        payload: Mapping[str, Any],
        *,
        key: str,
    ) -> Path:
        """Merge and atomically rewrite one stable item fragment.

        This is the write-side primitive used by v4 entity helpers. Repeated
        updates from the same producer preserve fields omitted by later partial
        updates while retaining one stable fragment file.
        """
        self._check_shape(section, "item")
        if not key:
            raise ValueError("upsert key must be non-empty")
        filename = self._stable_item_filename(section, key)
        target = self._dir / filename
        merged: dict[str, Any] = {}
        with self._lock:
            previous: Mapping[str, Any] | None = None
            try:
                current = json.loads(target.read_text(encoding="utf-8"))
                current_payload = current.get("payload") if isinstance(current, dict) else None
                if isinstance(current_payload, Mapping):
                    previous = current_payload
                    merged = _merge_mappings(current_payload, payload)
                else:
                    merged = dict(payload)
            except (OSError, ValueError, TypeError):
                merged = dict(payload)
            return self._write(
                section,
                "item",
                merged,
                filename=filename,
                operation="upsert",
                previous=previous,
            )

    @staticmethod
    def _check_shape(section: str, kind: str) -> None:
        """Validate that ``section`` is used with its declared shape.

        Args:
            section: The breakdown section name being written.
            kind: The shape being used (``"singleton"`` or ``"item"``).

        Raises:
            ValueError: If ``section`` is declared with a different shape.
        """
        declared = SECTION_SHAPES.get(section)
        if declared is not None and declared != kind:
            raise ValueError(f"section {section!r} is declared {declared!r}, not {kind!r}")

    def _write(
        self,
        section: str,
        kind: str,
        payload: Mapping[str, Any],
        *,
        filename: str,
        operation: str = "write",
        previous: Mapping[str, Any] | None = None,
        seq: int | None = None,
    ) -> Path:
        """Atomically write one fragment record to ``filename`` in the spool dir.

        Wraps ``payload`` in the fragment envelope (section / kind / seq / ts /
        producer) and writes it via a temp file plus ``os.replace`` so readers
        never observe a partial write.

        Every write in this class funnels through here, so this is also where
        the write trace is emitted (see :mod:`.trace`); it costs one level check
        when switched off.

        Args:
            section (str): the breakdown section name.
            kind (str): the fragment kind (``singleton`` or ``item``).
            payload (Mapping[str, Any]): the record payload.
            filename (str): the destination filename within the spool directory.
            operation (str): ``write`` when the fragment is replaced wholesale,
                ``upsert`` when it is merged into what was already there.
            previous (Mapping[str, Any] | None): the payload that was already on
                disk, so the trace can report what this write changed. ``None``
                when there was nothing to merge into.
            seq (int | None): a sequence number already drawn by the caller,
                for callers that also spend it on the filename. ``None`` draws
                a fresh one.

        Returns:
            Path: the path of the written fragment.

        Raises:
            Exception: re-raised if writing or replacing the file fails (the
                temp file is removed first).
        """
        record = {
            "section": section,
            "kind": kind,
            "seq": self._next_seq() if seq is None else seq,
            "ts": now_iso(timespec="microseconds"),
            "producer": self._producer,
            "payload": dict(payload) if isinstance(payload, Mapping) else payload,
        }
        data = json.dumps(record, ensure_ascii=False, sort_keys=True, default=str)
        target = self._dir / filename
        # Whether the fragment already existed is only knowable before the
        # write, and it is the difference between recording a new fact and
        # replacing one, so it is resolved here rather than after.
        traced = trace_enabled()
        existed = target.exists() if traced else False
        try:
            atomic_write_text(target, data, make_parents=True)
        except BaseException as exc:
            if traced:
                self._trace(record, target, payload, operation, previous, existed, len(data), exc)
            raise
        if traced:
            self._trace(record, target, payload, operation, previous, existed, len(data), None)
        return target

    def _trace(
        self,
        record: Mapping[str, Any],
        target: Path,
        payload: Mapping[str, Any],
        operation: str,
        previous: Mapping[str, Any] | None,
        existed: bool,
        size: int,
        error: BaseException | None,
    ) -> None:
        """Emit one write-trace line for a fragment this recorder just wrote."""
        trace_write(
            section=str(record.get("section") or ""),
            kind=str(record.get("kind") or ""),
            operation=operation,
            target=target,
            payload=payload if isinstance(payload, Mapping) else {},
            producer=self._producer,
            seq=int(record.get("seq") or 0),
            ts=str(record.get("ts") or ""),
            size=size,
            existed=existed,
            previous=previous,
            error=error,
        )


_RECORDERS: dict[tuple[str, str], Recorder] = {}
_RECORDERS_LOCK = threading.Lock()


def get_recorder(*, producer: str) -> Recorder:
    """Return the process-cached :class:`Recorder` for the bound session.

    The entry point for recording: a call site needs to know what it is
    recording and nothing else. Which session it lands in was decided once, at
    startup, by :func:`~...session.session_binding.bind_session`.

    Args:
        producer: The producer name owning the written fragments.

    Returns:
        The process-cached :class:`Recorder` for the bound session and this
        producer.

    Raises:
        SessionNotBoundError: If no session is bound -- either startup never
            bound one, or this is running in a subprocess, where writing
            fragments loses writes and is forbidden outright.
    """
    from ...session.session_binding import bound_session

    return recorder_for(bound_session(), producer=producer)


def recorder_for(session_dir: Path | str, *, producer: str) -> Recorder:
    """Return a process-cached :class:`Recorder` for an explicit session.

    Exists only for the v4 ``instrument`` helpers, whose whole API already
    takes ``session_dir`` as its first parameter. It goes away with them: no
    new caller should appear here, and every v6 entry point uses
    :func:`get_recorder` instead so the session path stays in one place.

    Args:
        session_dir: The session directory whose breakdown parts dir backs the
            recorder.
        producer: The producer name owning the written fragments.

    Returns:
        The process-cached :class:`Recorder` for the
        ``(session_dir, producer)`` pair.
    """
    from ...session.session_paths import breakdown_parts_dir  # local: avoid import cycle

    pd = breakdown_parts_dir(Path(session_dir))
    cache_key = (str(pd), _slug(producer))
    with _RECORDERS_LOCK:
        rec = _RECORDERS.get(cache_key)
        if rec is None:
            rec = Recorder(pd, producer=producer)
            _RECORDERS[cache_key] = rec
        return rec


__all__ = [
    "DERIVED_SECTIONS",
    "SECTION_SHAPES",
    "Recorder",
    "SectionShape",
    "get_recorder",
    "recorder_for",
    "section_shape",
]
