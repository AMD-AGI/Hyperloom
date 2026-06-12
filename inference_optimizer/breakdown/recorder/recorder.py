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
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .sections import SECTION_SHAPES

_SANITIZE = re.compile(r"[^A-Za-z0-9._-]+")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _slug(value: str) -> str:
    """Filesystem-safe token; empty input collapses to ``unknown``."""
    s = _SANITIZE.sub("-", str(value or "").strip())
    return s.strip("-.") or "unknown"


class Recorder:
    """Per-(session, producer) writer of breakdown record fragments."""

    def __init__(self, parts_dir: Path | str, *, producer: str) -> None:
        self._dir = Path(parts_dir)
        self._producer = _slug(producer)
        self._seq = 0
        self._lock = threading.Lock()

    @property
    def producer(self) -> str:
        return self._producer

    @property
    def parts_dir(self) -> Path:
        return self._dir

    def _next_seq(self) -> int:
        with self._lock:
            self._seq += 1
            return self._seq

    def record_singleton(
        self,
        section: str,
        payload: Mapping[str, Any],
    ) -> Path:
        """Write/overwrite this producer's single final blob for ``section``."""
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
        """
        self._check_shape(section, "item")
        if key:
            filename = f"{_slug(section)}__{self._producer}__{_slug(key)}.json"
        else:
            seq = self._next_seq()
            filename = (
                f"{_slug(section)}__{self._producer}__"
                f"{os.getpid()}-{seq:06d}.json"
            )
        return self._write(section, "item", payload, filename=filename)

    @staticmethod
    def _check_shape(section: str, kind: str) -> None:
        declared = SECTION_SHAPES.get(section)
        if declared is not None and declared != kind:
            raise ValueError(
                f"section {section!r} is declared {declared!r}, "
                f"not {kind!r}"
            )

    def _write(
        self,
        section: str,
        kind: str,
        payload: Mapping[str, Any],
        *,
        filename: str,
    ) -> Path:
        self._dir.mkdir(parents=True, exist_ok=True)
        record = {
            "section":  section,
            "kind":     kind,
            "seq":      self._next_seq(),
            "ts":       _now_iso(),
            "producer": self._producer,
            "payload":  dict(payload) if isinstance(payload, Mapping) else payload,
        }
        data = json.dumps(record, ensure_ascii=False, sort_keys=True, default=str)
        target = self._dir / filename
        fd, tmp = tempfile.mkstemp(
            prefix=f".{filename}.", suffix=".tmp", dir=str(self._dir),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(data)
            os.replace(tmp, target)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return target


_RECORDERS: dict[tuple[str, str], Recorder] = {}
_RECORDERS_LOCK = threading.Lock()


def get_recorder(session_dir: Path | str, *, producer: str) -> Recorder:
    """Return a process-cached :class:`Recorder` for ``(session_dir, producer)``.

    Lets deep call sites obtain the writer without threading it through every
    function signature.
    """
    from ...session_paths import breakdown_parts_dir  # local: avoid import cycle

    pd = breakdown_parts_dir(Path(session_dir))
    cache_key = (str(pd), _slug(producer))
    with _RECORDERS_LOCK:
        rec = _RECORDERS.get(cache_key)
        if rec is None:
            rec = Recorder(pd, producer=producer)
            _RECORDERS[cache_key] = rec
        return rec


__all__ = ["Recorder", "get_recorder"]
