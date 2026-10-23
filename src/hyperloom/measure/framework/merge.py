# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The directory aiter writes one test's merged tuned tables to, and the hook that makes it write there.

aiter writes the merged view of a variable that lists several tables to ``/tmp/aiter_configs``, which no variable moves
and local sandboxes share. ``aiter_merge_hook`` (copied as ``sitecustomize.py``, first on the server's ``PYTHONPATH``)
rebinds that one path to a directory of the test's own; this module makes the directory and the hook, and afterwards
reads the notes each process of the server left to say whether the hook did its work. A hook that cannot be installed
or did not run never refuses a test: the result says the merged view was shared (or that it could not be confirmed).
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom.measure import aiter_merge_hook
from hyperloom.measure.framework.base import file_sha256

__all__ = ["MergeView"]

_HOOK_FILE = "sitecustomize.py"
_PYTHONPATH = "PYTHONPATH"
_UTF8 = "utf-8"
PRIVATE, SHARED, UNCONFIRMED = "private", "shared", "unconfirmed"


@dataclass(frozen=True)
class MergeView:
    """The test's merge directory and the directory its hook is in; both below the test's private directory."""

    directory: Path
    hook: Path

    @classmethod
    def make(cls, private: Path) -> MergeView:
        """Make both directories and put the hook in ``hook``.

        Raises:
            OSError: When either cannot be made.
        """
        view = cls(private / "merged", private / "hook")
        view.directory.mkdir(parents=True)
        view.hook.mkdir(parents=True)
        shutil.copyfile(Path(aiter_merge_hook.__file__), view.hook / _HOOK_FILE)
        return view

    def environment(self, pythonpath: str) -> dict[str, str]:
        """The variables the server needs: the merge directory, and the hook ahead of ``pythonpath``."""
        parts = [str(self.hook), *pythonpath.split(os.pathsep)]
        return {
            aiter_merge_hook.MERGE_DIR_ENV: str(self.directory),
            _PYTHONPATH: os.pathsep.join(dict.fromkeys(part for part in parts if part)),
        }

    def report(self) -> dict[str, Any]:
        """What the test's processes did with the hook, and the merged tables aiter wrote below the directory."""
        notes = [_note(path) for path in sorted((self.hook / aiter_merge_hook.NOTES_DIR).glob("*.json"))]
        failed = [
            str(note.get("reason") or "the hook left a process undone") for note in notes if not note.get("patched")
        ]
        if failed:
            isolation, reason = SHARED, failed[0]
        elif not (self.hook / aiter_merge_hook.STARTED_NOTE).is_file():
            isolation, reason = UNCONFIRMED, "no process of the test ran the hook"
        else:
            isolation, reason = PRIVATE, None
        failed_chain = self.hook / aiter_merge_hook.CHAIN_FAILED_NOTE
        return {
            "merge_isolation": isolation,
            "merge_reason": reason,
            "merge_chain_failure": _note(failed_chain).get("reason") if failed_chain.is_file() else None,
            "merge_dir": str(self.directory),
            "merge_patched_processes": len(notes) - len(failed),
            "merge_files": {path.name: file_sha256(path) for path in sorted(self.directory.glob("*.csv"))},
        }


def _note(path: Path) -> dict[str, Any]:
    try:
        return dict(json.loads(path.read_text(encoding=_UTF8)))
    except (OSError, ValueError):
        return {"patched": False, "reason": f"the note {path.name} cannot be read"}
