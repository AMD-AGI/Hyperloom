# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""List the files a test left in the output directory, with their sizes and hashes (design 4.8.3 step 6)."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from hyperloom.measure import fsafe
from hyperloom.measure.farm import Farm
from hyperloom.measure.manifest import RESERVED_NAMES, ManifestFile

__all__ = [
    "FARM_INVENTORY_FILE",
    "OutputRejected",
    "drop_agentx_runtime",
    "farm_inventory",
    "inventory",
    "write_farm_inventory",
]


_UNSAFE = "unsafe_output"
#: Where Magpie installs the replay client of an AgentX round, inside the round's workspace: a Python environment
#: with links and the wheels it downloaded. It is the tool's installation, not a result of the test.
AGENTX_RUNTIME_DIR = ".agentx-runtime"


class OutputRejected(Exception):
    """The output cannot be listed: it holds a link or special file, or it is larger than the sheet allows."""

    def __init__(self, error_class: str, message: str) -> None:
        super().__init__(message)
        self.error_class = error_class


def drop_agentx_runtime(files_dir: Path) -> None:
    """Remove the replay client's installation from every AgentX workspace below ``files_dir``.

    A link in its place, or an output directory that is itself a link, is left for :func:`inventory` to refuse.
    """
    if files_dir.is_symlink():
        return
    for current, directories, _ in os.walk(files_dir):
        if AGENTX_RUNTIME_DIR in directories:
            directories.remove(AGENTX_RUNTIME_DIR)
            installation = Path(current, AGENTX_RUNTIME_DIR)
            if not installation.is_symlink():
                shutil.rmtree(installation)


def inventory(files_dir: Path, *, max_total_bytes: int) -> tuple[ManifestFile, ...]:
    """Every regular file below ``files_dir``, hashed through the descriptor that was checked.

    Raises:
        OutputRejected: ``unsafe_output`` for a symlink or special file anywhere below, ``output_too_large`` when the
            files add up to more than ``max_total_bytes``.
    """
    try:
        root_fd = fsafe.open_root(files_dir)
    except (fsafe.UnsafeEntry, fsafe.MissingEntry) as exc:
        raise OutputRejected(_UNSAFE, f"the output directory is not usable: {exc}") from exc
    entries: list[ManifestFile] = []
    total = 0
    try:
        for relative in fsafe.walk_regular_files(root_fd):
            if relative in RESERVED_NAMES:
                raise OutputRejected(_UNSAFE, f"{relative} is a name the bundle owns")
            fd = fsafe.open_beneath(root_fd, relative)
            try:
                sha256, size = fsafe.sha256_and_size(fd)
            finally:
                os.close(fd)
            total += size
            if total > max_total_bytes:
                raise OutputRejected("output_too_large", f"the output exceeds {max_total_bytes} bytes")
            entries.append(ManifestFile(relative, size, sha256))
    except fsafe.UnsafeEntry as exc:
        raise OutputRejected(_UNSAFE, str(exc)) from exc
    finally:
        os.close(root_fd)
    return tuple(entries)


FARM_INVENTORY_FILE = "farm_inventory.json"
_FARM_INVENTORY_SCHEMA = "hyperloom.farm-inventory/v1"


def farm_inventory(farm: Farm | None) -> dict:
    """What the farm rebuilt, by root id: the sha256 of each upserted file as it is on disk, and the deleted paths.

    Paths are relative to the package root. The hashes are taken from the files in the farm, not from the sheet, so
    they say what the server could import. Without a farm the roots are empty: the test had no patch.
    """
    roots = {}
    for root in farm.roots if farm else ():
        base = farm.site / root.import_name
        roots[root.id] = {
            "import_name": root.import_name,
            "upserts": {
                entry.path: hashlib.sha256((base / entry.path).read_bytes()).hexdigest()
                for entry in root.entries
                if entry.op == "upsert"
            },
            "deletes": sorted(entry.path for entry in root.entries if entry.op == "delete"),
        }
    return {"schema": _FARM_INVENTORY_SCHEMA, "roots": roots}


def write_farm_inventory(files_dir: Path, farm: Farm | None) -> None:
    """Write ``farm_inventory.json`` into the output files, so the manifest lists it and the signature covers it."""
    files_dir.mkdir(parents=True, exist_ok=True)
    (files_dir / FARM_INVENTORY_FILE).write_text(json.dumps(farm_inventory(farm), sort_keys=True), encoding="utf-8")
