# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A sandbox whose files are a directory, for the tests of what the host reads from and writes into a sandbox."""

from __future__ import annotations

from pathlib import Path

from hyperloom.sandbox.socket import DirEntry, SandboxGone


class DirSandbox:
    """A sandbox whose files are a directory: the file API reads through links, the listing does not. With
    ``fail_writes`` every write but the one of a ``DONE`` marker fails as if the sandbox were gone."""

    sandbox_id = "sbx"

    def __init__(self, root: Path, *, fail_writes: bool = False) -> None:
        self.root, self.fail_writes, self.writes = root, fail_writes, []

    def _at(self, path: str) -> Path:
        return self.root / path.lstrip("/")

    def write_file(self, path: str, data: bytes) -> None:
        self.writes.append(path)
        if self.fail_writes and not path.endswith("DONE"):
            raise SandboxGone("gone")
        self._at(path).parent.mkdir(parents=True, exist_ok=True)
        self._at(path).write_bytes(data)

    def read_file(self, path: str) -> bytes:
        return self._at(path).read_bytes()

    def copy_out(self, path: str, dest: Path) -> int:
        data = self._at(path).read_bytes()
        dest.write_bytes(data)
        return len(data)

    def list_entries(self, path: str) -> list[DirEntry]:
        found = []
        for child in sorted(self._at(path).iterdir()):
            kind = "link" if child.is_symlink() else "dir" if child.is_dir() else "file"
            found.append(DirEntry(child.name, kind, child.lstat().st_size))
        return found
