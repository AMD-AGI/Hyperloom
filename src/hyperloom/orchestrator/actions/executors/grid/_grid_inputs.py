# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What an executor resolves before it runs a grid: its session dir, the session's bounds, safe overrides."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


from ....phases import machine_state as _phase_state


def _resolve_session_dir() -> Path:
    """Resolve the active session_dir for executors that need an output root."""
    from hyperloom.inference_optimizer.session.paths import session_dir as _sd

    return _sd()


class SessionDirField:
    """An executor's ``session_dir``, resolved on read rather than at construction.

    The executors here are module-level singletons built at import, which is
    before the CLI pins the session. Resolving in ``__init__`` freezes them on
    the workspace root that concurrent sessions share.
    """

    def __set_name__(self, owner: type, name: str) -> None:
        self._slot = f"_{name}"

    def __get__(self, obj: Any, _objtype: type | None = None) -> Any:
        if obj is None:
            return self
        explicit = getattr(obj, self._slot, None)
        return explicit if explicit is not None else _resolve_session_dir()

    def __set__(self, obj: Any, value: Path | str | None) -> None:
        setattr(obj, self._slot, Path(value) if value else None)


# Sanitization for the LLM-supplied ``benchmark_script`` override: reject path separators / shell metacharacters,
# raising ``ValueError`` instead of running an unsafe subprocess.
_SCRIPT_NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]+\.sh$")


def sanitize_script_name(value: Any) -> str | None:
    """Return ``value`` if it's a safe benchmark script file name."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if not _SCRIPT_NAME_RE.match(text):
        raise ValueError(
            f"benchmark_script={text!r} rejected: must be a bare *.sh "
            "file name (no path separators, no shell metacharacters)"
        )
    return text


def session_grid_bounds(shared_state: Any) -> tuple[float | None, float | None]:
    """Resolve ``(session_deadline_sec, variant_expected_sec)`` for a :func:`run_grid` call."""
    if shared_state is None:
        return (None, None)
    deadline_fn = getattr(shared_state, "grid_session_deadline_sec", None)
    deadline = deadline_fn() if callable(deadline_fn) else None
    variant_sec = _phase_state.one_more_measurement_sec(shared_state)
    if variant_sec is None:
        variant_sec = _phase_state.measured_seconds(shared_state, "baseline_runtime_sec")
    return (deadline, variant_sec)
