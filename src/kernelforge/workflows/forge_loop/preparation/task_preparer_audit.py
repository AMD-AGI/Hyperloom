# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The audit trail one task preparation leaves behind."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path


class PrepareAudit:
    """The audit trail one preparation writes, and the driver digest it compares attempts against.

    Every artifact is best-effort: a snapshot that cannot be written never
    affects whether the driver is valid.
    """

    def __init__(self, audit_dir: Path | None, driver_path: Path) -> None:
        self._audit_dir = audit_dir
        self._driver_path = driver_path

    def text(self, relative: str, body: str) -> None:
        """Write one audit artifact."""
        if self._audit_dir is None:
            return
        try:
            path = self._audit_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body, encoding="utf-8")
        except OSError:
            # Audit artifacts are best-effort and never affect driver validity.
            pass

    def json(self, relative: str, payload: dict) -> None:
        """Write one audit artifact as JSON; metadata that will not serialize is dropped, not raised."""
        try:
            self.text(relative, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
        except (TypeError, ValueError):
            # Non-serializable audit metadata must not block task preparation.
            pass

    def driver(self, relative: str) -> None:
        """Snapshot the driver as it stands now."""
        if self._audit_dir is None or not self._driver_path.is_file():
            return
        try:
            path = self._audit_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self._driver_path, path)
            # copy2 carries the SOURCE mtime across, so every snapshot in the audit trail claimed the driver's own
            # mtime rather than when it was captured — reconstructing a timeline from this directory (the obvious
            # thing to do when a prep fails) then yields wildly wrong durations.
            os.utime(path, None)
        except OSError:
            # A missing audit copy is non-fatal; the staged driver remains the authoritative preparation artifact.
            pass

    def driver_digest(self) -> str:
        """The driver's current digest, or "" when it cannot be read."""
        try:
            return hashlib.sha256(self._driver_path.read_bytes()).hexdigest()
        except OSError:
            return ""

    def driver_edited(self, digest_before: str) -> bool:
        """Whether the driver changed since that digest."""
        return self.driver_digest() != digest_before


__all__ = ["PrepareAudit"]
