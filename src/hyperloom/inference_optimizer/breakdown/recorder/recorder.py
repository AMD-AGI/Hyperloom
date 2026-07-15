# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Write-side of the breakdown recorder.

A :class:`Recorder` lets the code that *produces* a fact record it at author
time, into a per-session spool directory, instead of having the exporter
re-walk heterogeneous artifacts later. Each producer owns its own files:

* :meth:`Recorder.record_singleton` — one final dict per section; the owner
  overwrites its own stable file (safe: single writer of that file).
* :meth:`Recorder.record_item` — one fragment per event; uniquely named so
  concurrent producers never collide. Pass ``key`` for an idempotent
  (overwrite-on-rewrite) item that survives resume without duplicating.

Writes are atomic (tmp + ``os.replace``) and filenames are unique per
(section, producer), so this is safe across processes and on network
filesystems (no shared-append dependency).
"""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Literal, Mapping

from hyperloom.common.io import atomic_write_text
from hyperloom.common.timeutil import now_iso

SectionShape = Literal["item", "singleton"]

# Per-section wire-shape registry for the breakdown recorder.
#
# Each ``session_breakdown.json`` section has exactly one owning producer, so
# there is never cross-producer write contention. A section is one of:
#
# * ``singleton`` — one final dict; the owner rewrites its own file on update
#   (last write by ``ts`` wins at assembly time).
# * ``item`` — an append-only event stream concatenated into a list (ordered by
#   ``seq`` then ``ts``) at assembly time.
#
# Derived sections (see ``DERIVED_SECTIONS``) are NOT written by producers
# during the run; they are computed at finalize from in-memory ``SharedState``
# (the Coordinator owns every input), so they never appear as fragments.
#
# Producer-written sections and their fragment shape. Payloads match the
# corresponding ``schema.py`` TypedDict so assembly is structure-preserving.
SECTION_SHAPES: dict[str, SectionShape] = {
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
}

# Sections computed at finalize from in-memory state, never written as
# fragments. Listed so the assembler can distinguish "expected absent" from
# "missing producer".
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


_SANITIZE = re.compile(r"[^A-Za-z0-9._-]+")


def _slug(value: str) -> str:
    """Filesystem-safe token; empty input collapses to ``unknown``.

    Args:
        value: The raw string to sanitise into a filesystem-safe token.

    Returns:
        The sanitised token, or ``"unknown"`` when the input is empty.
    """
    s = _SANITIZE.sub("-", str(value or "").strip())
    return s.strip("-.") or "unknown"


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
        self._lock = threading.Lock()

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
        if key:
            filename = f"{_slug(section)}__{self._producer}__{_slug(key)}.json"
        else:
            seq = self._next_seq()
            filename = f"{_slug(section)}__{self._producer}__{os.getpid()}-{seq:06d}.json"
        return self._write(section, "item", payload, filename=filename)

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
    ) -> Path:
        """Atomically write one fragment record to ``filename`` in the spool dir.

        Wraps ``payload`` in the fragment envelope (section / kind / seq / ts /
        producer) and writes it via a temp file plus ``os.replace`` so readers
        never observe a partial write.

        Args:
            section (str): the breakdown section name.
            kind (str): the fragment kind (``singleton`` or ``item``).
            payload (Mapping[str, Any]): the record payload.
            filename (str): the destination filename within the spool directory.

        Returns:
            Path: the path of the written fragment.

        Raises:
            Exception: re-raised if writing or replacing the file fails (the
                temp file is removed first).
        """
        record = {
            "section": section,
            "kind": kind,
            "seq": self._next_seq(),
            "ts": now_iso(timespec="microseconds"),
            "producer": self._producer,
            "payload": dict(payload) if isinstance(payload, Mapping) else payload,
        }
        data = json.dumps(record, ensure_ascii=False, sort_keys=True, default=str)
        target = self._dir / filename
        atomic_write_text(target, data, make_parents=True)
        return target


_RECORDERS: dict[tuple[str, str], Recorder] = {}
_RECORDERS_LOCK = threading.Lock()


def get_recorder(session_dir: Path | str, *, producer: str) -> Recorder:
    """Return a process-cached :class:`Recorder` for ``(session_dir, producer)``.

    Lets deep call sites obtain the writer without threading it through every
    function signature.

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
    "section_shape",
]
