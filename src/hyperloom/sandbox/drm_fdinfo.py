# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the processes a reader can see hold on the GPUs' render nodes, read from their open descriptors.

The kernel's list of the GPUs' processes (:mod:`hyperloom.sandbox.card_occupancy`) covers the whole machine, and a reader in
a pid namespace of its own cannot tell which of its entries are the processes it sees. What it can read is what those
processes themselves report: every descriptor of a render node has an fdinfo file with the memory the open DRM file holds
(the rule pack's ``drm_fdinfo``). Set against the kernel's total for a card, the difference is memory some process the
reader does not see holds.

A reading is either complete or refused: a process whose descriptors cannot be listed (another account's, where this one is
not root) could hold anything, so a sum over the rest says nothing, and :class:`FdinfoUnreadable` says why.
"""

from __future__ import annotations

import os
from collections.abc import Collection, Mapping
from pathlib import Path

from hyperloom.common.outside_facts import (
    DRM_CLIENT_ID_KEY,
    DRM_FD_DIR,
    DRM_FDINFO_DIR,
    DRM_RENDER_NODE_PREFIX,
    DRM_VRAM_KEY,
)

__all__ = ["FdinfoUnreadable", "vram_by_render_minor"]

#: What a process that is gone while its files are read raises: it holds nothing now.
_GONE = (FileNotFoundError, ProcessLookupError, NotADirectoryError)
#: The units fdinfo prints a size in; a bare number is bytes (it only ever reads 0).
_UNITS = {"": 1, "B": 1, "KiB": 1 << 10, "MiB": 1 << 20, "GiB": 1 << 30}
_LOSSY_TEXT = {"encoding": "utf-8", "errors": "replace"}


class FdinfoUnreadable(Exception):
    """The processes' descriptors cannot be read completely; the message says why."""


def vram_by_render_minor(procfs: Path, minors: Collection[int]) -> dict[int, int]:
    """The bytes of video memory the processes in ``procfs`` hold on each render node in ``minors``.

    An open DRM file is counted once however many descriptors, of however many processes, share it.

    Raises:
        FdinfoUnreadable: When the processes cannot be listed or the descriptors of one cannot be read.
    """
    clients: dict[tuple[int, str], int] = {}
    for process in _processes(procfs):
        _collect(process, frozenset(minors), clients)
    totals = dict.fromkeys(minors, 0)
    for (minor, _), vram in clients.items():
        totals[minor] += vram
    return totals


def _processes(procfs: Path) -> list[Path]:
    try:
        return [entry for entry in procfs.iterdir() if entry.name.isdigit()]
    except OSError as exc:
        raise FdinfoUnreadable(f"the processes of {procfs} cannot be listed ({type(exc).__name__})") from exc


def _collect(process: Path, minors: frozenset[int], clients: dict[tuple[int, str], int]) -> None:
    try:
        descriptors = [entry.name for entry in os.scandir(process / DRM_FD_DIR)]
    except _GONE:
        return
    except OSError as exc:
        raise FdinfoUnreadable(
            f"{process / DRM_FD_DIR} cannot be listed: {exc.strerror or type(exc).__name__}"
        ) from exc
    for descriptor in descriptors:
        minor = _render_minor(process / DRM_FD_DIR / descriptor)
        if minor not in minors:
            continue
        info = _fdinfo(process / DRM_FDINFO_DIR / descriptor)
        if info is not None:
            client = info.get(DRM_CLIENT_ID_KEY) or f"{process.name}/{descriptor}"
            clients[(minor, client)] = _bytes(info.get(DRM_VRAM_KEY, "0"), process)


def _render_minor(link: Path) -> int | None:
    """The minor of the render node a descriptor is open on, or ``None`` when it is open on anything else."""
    try:
        target = os.readlink(link)
    except OSError:
        return None
    name = target.rsplit("/", 1)[-1]
    suffix = name[len(DRM_RENDER_NODE_PREFIX) :]
    return int(suffix) if name.startswith(DRM_RENDER_NODE_PREFIX) and suffix.isdigit() else None


def _fdinfo(path: Path) -> Mapping[str, str] | None:
    try:
        text = path.read_text(**_LOSSY_TEXT)
    except _GONE:
        return None
    except OSError as exc:
        raise FdinfoUnreadable(f"{path} cannot be read: {exc.strerror or type(exc).__name__}") from exc
    pairs = (line.partition(":") for line in text.splitlines())
    return {key.strip(): value.strip() for key, _, value in pairs}


def _bytes(text: str, process: Path) -> int:
    number, _, unit = text.partition(" ")
    try:
        return int(number) * _UNITS[unit.strip()]
    except (ValueError, KeyError) as exc:
        raise FdinfoUnreadable(f"{process.name}: {DRM_VRAM_KEY} reads {text!r}, not a size it knows") from exc
