# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Resource samples kept on disk, outside the session, so a restarted Inspector can backfill its chart.

The session directory stays read-only: :func:`history_path` refuses any location inside it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

_ENCODING = "utf-8"


def default_history_dir() -> Path:
    """``$XDG_STATE_HOME/hyperloom-inspector``, or ``~/.local/state/hyperloom-inspector``."""
    base = os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state"
    return Path(base) / "hyperloom-inspector"


def history_path(session_dir: Path, history_dir: Path) -> Path:
    """One file per session: its name plus a hash of its absolute path, so equal names don't collide.

    Raises ``ValueError`` when *history_dir* is inside *session_dir*.
    """
    session = Path(session_dir).expanduser().resolve()
    target = Path(history_dir).expanduser().resolve()
    if target == session or session in target.parents:
        raise ValueError(f"history dir {target} is inside the session {session}; the session stays read-only")
    digest = hashlib.sha256(str(session).encode()).hexdigest()[:10]
    return target / f"{session.name}-{digest}.jsonl"


class MetricsHistory:
    """A JSONL ring of ``{"t", ...sample}`` lines.

    Appends one line per sample and rewrites the file with the newest *keep* lines once it holds twice that many.
    A write error disables it for the rest of the run instead of failing the watcher.
    """

    def __init__(self, path: Path, keep: int = 4000) -> None:
        self.path: Path | None = Path(path)
        self.keep = keep
        self._lines = 0

    def load(self) -> list[dict]:
        """The newest *keep* samples, oldest first; a torn last line is skipped."""
        if self.path is None:
            return []
        try:
            lines = self.path.read_text(encoding=_ENCODING, errors="replace").splitlines()
        except OSError:
            return []
        self._lines = len(lines)
        out = []
        for line in lines[-self.keep :]:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if isinstance(rec, dict) and isinstance(rec.get("t"), (int, float)):
                out.append(rec)
        return out

    def append(self, rec: dict) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding=_ENCODING) as fh:
                fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
            self._lines += 1
            if self._lines >= 2 * self.keep:
                self._rotate()
        except OSError as exc:
            log.warning("inspector: resource history disabled, cannot write %s: %s", self.path, exc)
            self.path = None

    def _rotate(self) -> None:
        assert self.path is not None  # nosec B101 - append() checked it
        lines = self.path.read_text(encoding=_ENCODING, errors="replace").splitlines()[-self.keep :]
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text("".join(line + "\n" for line in lines), encoding=_ENCODING)
        os.replace(tmp, self.path)
        self._lines = len(lines)
