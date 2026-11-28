# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A sandbox whose files are a directory, for the tests of what the host reads from and writes into a sandbox."""

from __future__ import annotations

from pathlib import Path

from hyperloom.sandbox.socket import DirEntry, SandboxDenied, SandboxGone

#: The largest single write the hosted platform's sandbox file API accepts.
PLATFORM_WRITE_LIMIT = 32 * 1000**2


class DirSandbox:
    """A sandbox whose files are a directory: the file API reads through links, the listing does not. With
    ``fail_writes`` every write but the one of a ``DONE`` marker fails as if the sandbox were gone; a write larger than
    ``write_limit`` is refused as the platform refuses it (413)."""

    sandbox_id = "sbx"

    def __init__(self, root: Path, *, fail_writes: bool = False, write_limit: int | None = None) -> None:
        self.root, self.fail_writes, self.write_limit, self.writes = root, fail_writes, write_limit, []

    def _at(self, path: str) -> Path:
        return self.root / path.lstrip("/")

    def write_file(self, path: str, data: bytes) -> None:
        self.writes.append(path)
        if self.fail_writes and not path.endswith("DONE"):
            raise SandboxGone("gone")
        if self.write_limit is not None and len(data) > self.write_limit:
            raise SandboxDenied("413: file exceeds 32MB")
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
