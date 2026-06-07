# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Cross-tick state persistence for stateful subsystems.

The robustness-agent M1 transport spawns a fresh ``python -m
robustness_agent.runtime.cli tick`` subprocess per Coordinator tick.
That means **any in-memory state accumulated by a detector / ladder /
throttle is lost between ticks** — fatal for any rule that depends on
"N consecutive ticks", a rolling window, or a cooldown timestamp.

:class:`DetectorStateStore` is the disk-backed unification layer.
One JSON file per session lives at::

    <session_dir>/agents/robustness/detector_state.json

with a flat namespaced layout::

    {
      "gpu_leak":      {"consecutive_hits": 2},
      "ray_pending":   {"consecutive_hits": 1},
      "aiter_jit":     {"last_so_count": 173, "last_build_count": 4, "build_count_streak_ticks": 0},
      "progress":      {"gain_history": [0.0, 0.1, 0.2, ...]},
      "preflight":     {"fired_fingerprint": [...], "fired_mtime": 1700000000.0},
      "external_deps": {"tracelens_cli_fired": true},
      "action_ladder": {"last_emitted": {"key|tuple": 17, ...}},
      "rca_throttle":  {"last_called_unix": {"key|tuple": 1700000003.5, ...}}
    }

Owners hold a thin :class:`DetectorStateView` handle that exposes
``load() / save(dict)`` against their own slot — they don't see other
namespaces. The store is responsible for atomic flush via
``tmpfile + os.replace``; views just mutate the shared in-memory dict.

The reactor calls :meth:`DetectorStateStore.flush_atomic` at the end
of every successful tick (via ``asyncio.to_thread`` so the tick budget
isn't blocked on the fsync).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any


log = logging.getLogger(__name__)


# Default filename. The path lives under ``<session_dir>/<subdir>/``
# so a single ``rm -r agents/robustness`` resets all robustness disk
# artefacts together (findings + state + reports).
_STATE_FILENAME: str = "detector_state.json"

# Default subdir under ``session_dir``. Keep aligned with
# ``FindingSinkConfig.subdir.parent`` so the disk layout is uniform.
_DEFAULT_SUBDIR: str = "agents/robustness"


class DetectorStateStore:
    """JSON-backed namespaced key-value store.

    All writes go to an in-memory ``dict[str, dict[str, Any]]`` and are
    materialised by :meth:`flush_atomic` (called once per tick by the
    reactor). Reads always hit the in-memory copy — load happens once
    in the constructor.
    """

    def __init__(
        self,
        *,
        session_dir: Path,
        subdir: str = _DEFAULT_SUBDIR,
        filename: str = _STATE_FILENAME,
    ) -> None:
        self._dir = Path(session_dir) / subdir
        self._path = self._dir / filename
        self._data: dict[str, dict[str, Any]] = {}
        self._dirty: bool = False
        self._load()

    @property
    def file_path(self) -> Path:
        return self._path

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------
    def _load(self) -> None:
        if not self._path.is_file():
            return
        try:
            raw = self._path.read_text(encoding="utf-8")
        except OSError as exc:
            log.warning(
                "state_store: cannot read %s: %s — starting empty",
                self._path, exc,
            )
            return
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            log.warning(
                "state_store: malformed JSON at %s: %s — starting empty",
                self._path, exc,
            )
            return
        if not isinstance(parsed, dict):
            log.warning(
                "state_store: top-level JSON at %s is not an object — "
                "starting empty",
                self._path,
            )
            return
        # Per-slot normalisation — drop non-dict values so consumers
        # always see ``dict[str, Any]`` on read.
        for key, value in parsed.items():
            if isinstance(value, dict):
                self._data[str(key)] = value

    def flush_atomic(self) -> None:
        """Atomically write the current in-memory state to disk.

        Uses ``tmpfile + os.replace`` so concurrent readers (e.g. an
        operator running ``finalize`` while the reactor is mid-tick)
        never see a partially-written file.
        """
        if not self._dirty:
            return
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            log.warning(
                "state_store: cannot create %s: %s", self._dir, exc,
            )
            return
        try:
            # NamedTemporaryFile in the same dir guarantees os.replace
            # is atomic (same filesystem).
            tmp = tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=str(self._dir),
                prefix=".detector_state.",
                suffix=".tmp",
                delete=False,
            )
            try:
                json.dump(self._data, tmp, indent=2, sort_keys=True)
                tmp.write("\n")
                tmp.flush()
                os.fsync(tmp.fileno())
            finally:
                tmp.close()
            os.replace(tmp.name, self._path)
            self._dirty = False
        except (OSError, TypeError, ValueError) as exc:
            log.warning(
                "state_store: flush_atomic to %s failed: %s",
                self._path, exc,
            )

    # ------------------------------------------------------------------
    # slot API
    # ------------------------------------------------------------------
    def load_slot(self, name: str) -> dict[str, Any]:
        """Return a copy of the slot's content (empty dict if absent)."""
        return dict(self._data.get(name) or {})

    def save_slot(self, name: str, payload: dict[str, Any]) -> None:
        """Replace the slot's content (does not flush to disk)."""
        if not isinstance(payload, dict):
            raise TypeError(
                f"save_slot payload must be a dict, got {type(payload).__name__}"
            )
        self._data[name] = dict(payload)
        self._dirty = True

    def view(self, name: str) -> "DetectorStateView":
        """Return a per-slot handle for a detector / ladder / throttle."""
        return DetectorStateView(store=self, slot=name)

    # ------------------------------------------------------------------
    # introspection (tests / operators)
    # ------------------------------------------------------------------
    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {k: dict(v) for k, v in self._data.items()}


class DetectorStateView:
    """Per-slot accessor passed to a single detector / ladder / throttle.

    Owners don't know (and shouldn't care) what other slots exist; they
    just call :meth:`load` / :meth:`save` against their own namespace.
    Construction with ``store=None`` (e.g. legacy tests) keeps the API
    surface usable but in-memory only.
    """

    __slots__ = ("_store", "_slot")

    def __init__(
        self,
        *,
        store: DetectorStateStore | None,
        slot: str,
    ) -> None:
        self._store = store
        self._slot = slot

    @property
    def slot_name(self) -> str:
        return self._slot

    @property
    def is_persistent(self) -> bool:
        return self._store is not None

    def load(self) -> dict[str, Any]:
        if self._store is None:
            return {}
        return self._store.load_slot(self._slot)

    def save(self, payload: dict[str, Any]) -> None:
        if self._store is None:
            return
        self._store.save_slot(self._slot, payload)


__all__ = [
    "DetectorStateStore",
    "DetectorStateView",
]
