# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which user and group ids this process's user namespace maps, read from the kernel.

A process can become, or give a file to, only an id its user namespace maps. ``/proc/self/uid_map`` and
``/proc/self/gid_map`` hold one ``inside outside count`` line per mapped range: the ids ``inside`` to
``inside + count - 1``. A container whose namespace maps one id (its root) has no other: ``setuid``, ``setgroups`` or
``chown`` to any other id fails there with ``EINVAL``, and under a fakeroot ``LD_PRELOAD`` it appears to succeed while
the process stays root. The calls cannot tell the two apart; the maps can, as they are read with plain reads, which such
a preload does not fake.

A map that cannot be read or parsed answers nothing, so :func:`unmapped_ids` reports it as the problem: a caller that
refuses on a problem refuses then too.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

__all__ = ["GID_MAP", "UID_MAP", "unmapped_ids"]

UID_MAP = Path("/proc/self/uid_map")
GID_MAP = Path("/proc/self/gid_map")
#: How many characters of a map, or of a line that is not one, a message quotes.
_QUOTED = 200


class _Unreadable(Exception):
    """A map that cannot be read, or is not ``inside outside count`` lines of decimal ids; the message names the file."""


@dataclass(frozen=True)
class _IdMap:
    ranges: tuple[range, ...]
    text: str

    def maps(self, wanted: int) -> bool:
        return any(wanted in mapped for mapped in self.ranges)


def _read(path: os.PathLike[str] | str) -> _IdMap:
    try:
        text = Path(path).read_text(encoding="ascii")
    except (OSError, UnicodeDecodeError) as exc:
        raise _Unreadable(f"cannot read {path}: {type(exc).__name__}: {exc}") from None
    ranges, lines = [], []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) != 3 or not all(field.isdecimal() for field in fields):
            raise _Unreadable(f"cannot parse {path}: {line.strip()[:_QUOTED]!r} is not 'inside outside count'")
        inside, count = int(fields[0]), int(fields[2])
        ranges.append(range(inside, inside + count))
        lines.append(" ".join(fields))
    return _IdMap(tuple(ranges), "; ".join(lines)[:_QUOTED] or "empty")


def unmapped_ids(
    uids: Iterable[int] = (),
    gids: Iterable[int] = (),
    *,
    uid_map: os.PathLike[str] | str = UID_MAP,
    gid_map: os.PathLike[str] | str = GID_MAP,
) -> str | None:
    """``None`` when this process's user namespace maps every uid of ``uids`` (in ``uid_map``) and every gid of ``gids``
    (in ``gid_map``); otherwise the ids it does not map, with the content of the map each was looked up in, or why a
    map could not be read or parsed (which is never taken as mapped)."""
    missing: list[str] = []
    quoted: list[str] = []
    for kind, ids, path in (("uid", uids, uid_map), ("gid", gids, gid_map)):
        wanted = list(dict.fromkeys(ids))
        if not wanted:
            continue
        try:
            found = _read(path)
        except _Unreadable as exc:
            return str(exc)
        absent = [f"{kind} {each}" for each in wanted if not found.maps(each)]
        if absent:
            missing.extend(absent)
            quoted.append(f"{path}: {found.text!r}")
    if not missing:
        return None
    verb = "is" if len(missing) == 1 else "are"
    return f"{', '.join(missing)} {verb} not mapped in this process's user namespace ({'; '.join(quoted)})"
