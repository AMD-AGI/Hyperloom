# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Cross-tick state persistence for stateful subsystems."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from hyperloom.common.io import atomic_write_json


log = logging.getLogger(__name__)


_STATE_FILENAME: str = "detector_state.json"

# Keep aligned with ``FindingSinkConfig.subdir.parent`` so the disk layout is uniform.
_DEFAULT_SUBDIR: str = "agents/robustness"


class DetectorStateStore:
    """JSON-backed namespaced key-value store."""

    def __init__(
        self,
        *,
        session_dir: Path,
        subdir: str = _DEFAULT_SUBDIR,
        filename: str = _STATE_FILENAME,
    ) -> None:
        """Initialise the store and eagerly load any existing state."""
        self._dir = Path(session_dir) / subdir
        self._path = self._dir / filename
        self._data: dict[str, dict[str, Any]] = {}
        self._dirty: bool = False
        self._load()

    @property
    def file_path(self) -> Path:
        """On-disk path of the backing JSON state file."""
        return self._path

    # I/O
    def _load(self) -> None:
        """Load and normalise the on-disk state into memory."""
        if not self._path.is_file():
            return
        try:
            raw = self._path.read_text(encoding="utf-8")
        except OSError as exc:
            log.warning(
                "state_store: cannot read %s: %s — starting empty",
                self._path,
                exc,
            )
            return
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            log.warning(
                "state_store: malformed JSON at %s: %s — starting empty",
                self._path,
                exc,
            )
            return
        if not isinstance(parsed, dict):
            log.warning(
                "state_store: top-level JSON at %s is not an object — starting empty",
                self._path,
            )
            return
        # Drop non-dict slot values so consumers always see ``dict[str, Any]``.
        for key, value in parsed.items():
            if isinstance(value, dict):
                self._data[str(key)] = value

    def flush_atomic(self) -> None:
        """Atomically write the current in-memory state to disk."""
        if not self._dirty:
            return
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            log.warning(
                "state_store: cannot create %s: %s",
                self._dir,
                exc,
            )
            return
        try:
            atomic_write_json(
                self._path,
                self._data,
                indent=2,
                sort_keys=True,
                trailing_newline=True,
                make_parents=False,
                fsync=True,
            )
            self._dirty = False
        except (OSError, TypeError, ValueError) as exc:
            log.warning(
                "state_store: flush_atomic to %s failed: %s",
                self._path,
                exc,
            )

    # slot API
    def load_slot(self, name: str) -> dict[str, Any]:
        """Return a copy of the slot's content (empty dict if absent)."""
        return dict(self._data.get(name) or {})

    def save_slot(self, name: str, payload: dict[str, Any]) -> None:
        """Replace the slot's content (does not flush to disk)."""
        if not isinstance(payload, dict):
            raise TypeError(f"save_slot payload must be a dict, got {type(payload).__name__}")
        self._data[name] = dict(payload)
        self._dirty = True

    def view(self, name: str) -> "DetectorStateView":
        """Return a per-slot handle for a detector / ladder / throttle."""
        return DetectorStateView(store=self, slot=name)

    # introspection (tests / operators)
    def snapshot(self) -> dict[str, dict[str, Any]]:
        """Return a deep-ish copy of all slots for tests / operators."""
        return {k: dict(v) for k, v in self._data.items()}


class DetectorStateView:
    """Per-slot accessor passed to a single detector / ladder / throttle."""

    __slots__ = ("_store", "_slot")

    def __init__(
        self,
        *,
        store: DetectorStateStore,
        slot: str,
    ) -> None:
        """Bind a view to one slot of a store."""
        self._store = store
        self._slot = slot

    def load(self) -> dict[str, Any]:
        """Load this view's slot content."""
        return self._store.load_slot(self._slot)

    def save(self, payload: dict[str, Any]) -> None:
        """Save content into this view's slot."""
        self._store.save_slot(self._slot, payload)


__all__ = [
    "DetectorStateStore",
    "DetectorStateView",
]
