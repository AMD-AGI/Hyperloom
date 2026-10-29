# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Write-side of the breakdown recorder."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Mapping

from hyperloom_common.io import WriteOptions, atomic_write_text
from hyperloom_common.timeutil import now_iso

from .sections import EVENT_SECTIONS, slug
from .trace import WriteTrace, trace_enabled, trace_write

SectionShape = Literal["item", "singleton"]

# Per-section wire-shape registry. Each section has exactly one owning
# producer, so there is never cross-producer write contention. A ``singleton``
# is one final dict the owner rewrites in place (last write by ``ts`` wins at
# assembly); an ``item`` section is an append-only stream assembled into a list
# in ``seq`` then ``ts`` order. Every row-shaped fact owns a section of its own,
# one fragment per row keyed by its real id, so a partial update cannot
# silently duplicate a row. Payloads match the corresponding ``schema.py``
# TypedDict, so assembly is structure-preserving.
SECTION_SHAPES: dict[str, SectionShape] = {
    "session": "singleton",
    "metadata": "singleton",
    "outcome": "singleton",
    # A section of its own rather than a second producer on ``metadata``,
    # because assembly keeps only the newest singleton per producer and the
    # Coordinator's -- reissued on every state save -- would always win.
    "versions": "item",
    # Keyed by content rather than by iteration number, which a resume reuses
    # after workdir pruning.
    "critic_iteration": "item",
    **dict.fromkeys(EVENT_SECTIONS, "item"),
    # ``close_step`` is composed into ``close.steps`` at assembly. The verdict
    # is recorded by the sequencer's last act rather than inferred at export
    # from which steps are present, because ``session_breakdown`` is itself a
    # step in the middle of the sequence.
    "close": "singleton",
    "close_step": "item",
    # Composed into ``close.kb_write_back``. Part of the close-out rather than
    # a timeline event of its own because the session attempts it
    # unconditionally, so its absence is meaningful. Attempts are keyed by
    # number because the publication is retried, and each is opened before the
    # write, so an attempt with no close died mid-publish.
    "close_write_back": "singleton",
    "close_write_back_attempt": "item",
}


log = logging.getLogger(__name__)

_ENTITY_ID_FIELDS = ("attempt_id", "gate_id", "decision_id")


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


@dataclass(frozen=True)
class FragmentSlot:
    """Where one fragment lives: its section, shape and file name."""

    section: str
    kind: str
    filename: str


class Recorder:
    """Per-(session, producer) writer of breakdown record fragments."""

    def __init__(self, parts_dir: Path | str, *, producer: str) -> None:
        """Initialize a recorder writing into ``parts_dir`` for ``producer``."""
        self._dir = Path(parts_dir)
        self._producer = slug(producer)
        self._seq = 0
        self._lock = threading.RLock()

    def _next_seq(self) -> int:
        """Return the next monotonically increasing per-recorder sequence number."""
        with self._lock:
            self._seq += 1
            return self._seq

    def _park_spool_failure(self, section: str, error: BaseException) -> None:
        """Park a spool/binding failure. Projection bugs are not passed here."""
        from .recorder_warnings import note_failure

        note_failure(section=section, error=error, producer=self._producer)

    def record_singleton(
        self,
        section: str,
        payload: Mapping[str, Any],
    ) -> Path | None:
        """Write/overwrite this producer's single final blob for ``section``,
        which must be declared ``singleton``-shaped.

        Spool and binding failures are parked and return ``None``. Anything
        else raises: this is the write-side owner of :data:`RECORDING_ERRORS`.
        """
        from .recorder_warnings import RECORDING_ERRORS

        try:
            self._check_shape(section, "singleton")
            filename = f"{slug(section)}__{self._producer}.json"
            return self._write(FragmentSlot(section, "singleton", filename), payload)
        except RECORDING_ERRORS as exc:
            self._park_spool_failure(section, exc)
            return None

    def record_upsert_singleton(
        self,
        section: str,
        payload: Mapping[str, Any],
    ) -> Path | None:
        """Merge and atomically rewrite this producer's singleton fragment."""
        from .recorder_warnings import RECORDING_ERRORS

        try:
            self._check_shape(section, "singleton")
            return self._upsert(FragmentSlot(section, "singleton", f"{slug(section)}__{self._producer}.json"), payload)
        except RECORDING_ERRORS as exc:
            self._park_spool_failure(section, exc)
            return None

    def _upsert(self, slot: FragmentSlot, payload: Mapping[str, Any]) -> Path:
        """Deep-merge ``payload`` into the fragment at ``slot`` and rewrite it."""
        with self._lock:
            previous: Mapping[str, Any] | None = None
            try:
                current = json.loads((self._dir / slot.filename).read_text(encoding="utf-8"))
                current_payload = current.get("payload") if isinstance(current, dict) else None
                if isinstance(current_payload, Mapping):
                    previous = current_payload
                    merged = _merge_mappings(current_payload, payload)
                else:
                    merged = dict(payload)
            except (OSError, ValueError, TypeError):
                merged = dict(payload)
            return self._write(slot, merged, operation="upsert", previous=previous)

    def record_item(
        self,
        section: str,
        payload: Mapping[str, Any],
        *,
        key: str | None = None,
    ) -> Path | None:
        """Append one event fragment to the ``item``-shaped ``section`` stream.

        ``key`` is a stable per-item identity; when given, the fragment
        filename is derived from it, so re-recording the same key overwrites
        rather than duplicates and the write is idempotent across retries and
        resume. Without one, a pid/sequence-unique filename is used.
        """
        from .recorder_warnings import RECORDING_ERRORS

        try:
            self._check_shape(section, "item")
            seq: int | None = None
            if key:
                filename = self._stable_item_filename(section, key)
            else:
                # One number serves both filename and envelope, so ``seq=N`` in a trace line locates the file that write produced.
                seq = self._next_seq()
                filename = f"{slug(section)}__{self._producer}__{os.getpid()}-{seq:06d}.json"
            return self._write(FragmentSlot(section, "item", filename), payload, seq=seq)
        except RECORDING_ERRORS as exc:
            self._park_spool_failure(section, exc)
            return None

    def _stable_item_filename(self, section: str, key: str) -> str:
        """Name the fragment file that holds ``key``'s item in ``section``.

        ``slug`` folds every character outside ``[A-Za-z0-9._-]`` to a dash,
        so ``a/b``, ``a:b`` and ``a b`` would all name one file; a digest of
        the untouched key makes the name injective again. Pre-digest fragments
        keep their old name so a resumed session goes on updating the file it
        already wrote, but only when the key survived sanitizing untouched --
        otherwise the legacy file could belong to any key that folds onto it.
        """
        key_slug = slug(key)
        prefix = f"{slug(section)}__{self._producer}__{key_slug}"
        digest = hashlib.sha256(key.encode("utf-8", errors="replace")).hexdigest()[:8]
        filename = f"{prefix}-{digest}.json"
        if len(filename.encode("utf-8")) > 180:
            short_digest = hashlib.sha256(key.encode("utf-8", errors="replace")).hexdigest()[:16]
            return f"{slug(section)}__{self._producer}__id-{short_digest}.json"
        legacy = self._dir / f"{prefix}.json"
        if key_slug != key:
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
    ) -> Path | None:
        """Merge and atomically rewrite one stable item fragment."""
        from .recorder_warnings import RECORDING_ERRORS

        try:
            self._check_shape(section, "item")
            if not key:
                raise ValueError("upsert key must be non-empty")
            return self._upsert(FragmentSlot(section, "item", self._stable_item_filename(section, key)), payload)
        except RECORDING_ERRORS as exc:
            self._park_spool_failure(section, exc)
            return None

    def item_fragment_exists(self, section: str, *, key: str) -> bool:
        """Whether an item fragment under ``key`` has already been written.

        For a late verdict merging onto a row an earlier stage recorded. The
        merge is keyed by the row's identity *and* its event, and an upsert
        onto an absent key does not fail but mints a row holding the verdict
        and nothing else. Asking first records nothing instead, which is what
        a fact with no row to belong to should do.
        """
        if not key:
            return False
        return (self._dir / self._stable_item_filename(section, key)).exists()

    @staticmethod
    def _check_shape(section: str, kind: str) -> None:
        """Validate that ``section`` is used with its declared shape."""
        declared = SECTION_SHAPES.get(section)
        if declared is not None and declared != kind:
            raise ValueError(f"section {section!r} is declared {declared!r}, not {kind!r}")

    def _write(
        self,
        slot: FragmentSlot,
        payload: Mapping[str, Any],
        *,
        operation: str = "write",
        previous: Mapping[str, Any] | None = None,
        seq: int | None = None,
    ) -> Path:
        """Atomically write one fragment record to ``slot`` in the spool dir.

        Wraps ``payload`` in the fragment envelope (section / kind / seq / ts /
        producer) and writes it via a temp file plus ``os.replace``, so readers
        never observe a partial write. Every write in this class funnels
        through here, so this is also where the write trace is emitted (see
        :mod:`.trace`); it costs one level check when switched off.

        ``operation`` is ``write`` when the fragment is replaced wholesale and
        ``upsert`` when it is merged into what was there; ``previous`` is that
        prior payload, so the trace can report what changed. ``seq`` is for a
        caller that already drew one to spend on the filename. A write failure
        is re-raised after the temp file is removed.
        """
        record = {
            "section": slot.section,
            "kind": slot.kind,
            "seq": self._next_seq() if seq is None else seq,
            "ts": now_iso(timespec="microseconds"),
            "producer": self._producer,
            "payload": dict(payload) if isinstance(payload, Mapping) else payload,
        }
        data = json.dumps(record, ensure_ascii=False, sort_keys=True, default=str)
        target = self._dir / slot.filename
        # Only knowable before the write, and it is the difference between recording a new fact and replacing one.
        trace = (
            WriteTrace(
                section=slot.section,
                kind=slot.kind,
                operation=operation,
                target=target,
                payload=payload if isinstance(payload, Mapping) else {},
                producer=self._producer,
                seq=record["seq"],
                ts=record["ts"],
                size=len(data),
                existed=target.exists(),
                previous=previous,
            )
            if trace_enabled()
            else None
        )
        try:
            atomic_write_text(target, data, options=WriteOptions(make_parents=True))
        except BaseException as exc:
            if trace is not None:
                trace_write(replace(trace, error=exc))
            raise
        if trace is not None:
            trace_write(trace)
        return target


_RECORDERS: dict[tuple[str, str], Recorder] = {}
_RECORDERS_LOCK = threading.Lock()


def get_recorder(*, producer: str) -> Recorder:
    """Return the process-cached :class:`Recorder` for the bound session.

    The entry point for recording: a call site needs to know what it is
    recording and nothing else, the session having been decided once at startup
    by :func:`~...session.session_binding.bind_session`. Raises
    :exc:`SessionNotBoundError` when nothing is bound, which also covers a
    subprocess, where writing fragments loses writes and is forbidden.
    """
    from ...session.session_binding import bound_session

    return recorder_for(bound_session(), producer=producer)


def recorder_for(session_dir: Path | str, *, producer: str) -> Recorder:
    """Return a process-cached :class:`Recorder` for an explicit session."""
    from ...session.session_paths import breakdown_parts_dir  # local: avoid import cycle

    pd = breakdown_parts_dir(Path(session_dir))
    cache_key = (str(pd), slug(producer))
    with _RECORDERS_LOCK:
        rec = _RECORDERS.get(cache_key)
        if rec is None:
            rec = Recorder(pd, producer=producer)
            _RECORDERS[cache_key] = rec
        return rec


__all__ = [
    "SECTION_SHAPES",
    "Recorder",
    "SectionShape",
    "get_recorder",
    "recorder_for",
]
