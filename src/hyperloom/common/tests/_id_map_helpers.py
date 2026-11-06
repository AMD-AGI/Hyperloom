# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Stand-in user namespace maps for the tests of what reads them (:mod:`hyperloom.common.id_map`)."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.common import id_map

#: The map of a host's initial user namespace, as the kernel writes it: every id is mapped.
HOST_MAP = "         0          0 4294967295\n"
#: A user namespace that maps one id, its root, onto the account that started the container.
ONE_ID_MAP = "         0       1234          1\n"


def use_id_maps(
    monkeypatch: pytest.MonkeyPatch, where: Path, uid_map: str | None, gid_map: str | None = None
) -> tuple[Path, Path]:
    """Write a uid map and a gid map under ``where`` (the gid map is the uid map unless given; ``None`` writes no file)
    and have :func:`~hyperloom.common.id_map.unmapped_ids` read them in place of this process's own; returns their
    paths."""
    paths = (where / "uid_map", where / "gid_map")
    for path, text in zip(paths, (uid_map, uid_map if gid_map is None else gid_map), strict=True):
        if text is not None:
            path.write_text(text, encoding="ascii")
    monkeypatch.setattr(id_map, "UID_MAP", paths[0])
    monkeypatch.setattr(id_map, "GID_MAP", paths[1])
    return paths
