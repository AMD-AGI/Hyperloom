# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A JIT directory of the test's own for an aiter that the compiled-code cache gives none.

aiter installs a module it builds into ``get_user_jit_dir()``: ``AITER_JIT_DIR`` when it is set, else the ``jit``
directory of the package it was imported from. A test whose server imports the image's aiter (no farm rebuilt it) and
gets no JIT directory from the cache (none is configured, or the aiter kinds are off) would install what it builds into
the image, where every later test on the node loads it. Such a server is told ``AITER_JIT_DIR`` of a directory of the
test's own, holding links to the image's prebuilt modules (as the cache's directory does), so it builds only what the
image lacks and what the test's tuned tables make stale.

The directory only names a location: the cache's key leaves ``AITER_JIT_DIR`` out (``measure.jit.key``), so no key moves.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import yaml

from hyperloom.measure.farm import Farm, FarmError, locate_package, server_environment, server_python
from hyperloom.measure.framework.table_modules import stale_table_modules
from hyperloom.measure.jit.key import AITER_PACKAGE
from hyperloom.measure.jit.modules import config_path, prebuilt_names, read_config, touched_entries
from hyperloom.measure.jit.session import JitCacheSession, link_prebuilt
from hyperloom.measure.jit_kinds import AITER_JIT
from hyperloom.measure.socket import TestCall

__all__ = ["OWN_AITER_JIT_DIR", "own_aiter_jit"]

#: The directory below the test's work directory. Not below the cache's private root, which seeding empties.
OWN_AITER_JIT_DIR = "aiter_jit"
_TABLE_PREFIX = "AITER_CONFIG_"
_UTF8 = "utf-8"


def own_aiter_jit(
    call: TestCall, farm: Farm | None, session: JitCacheSession | None, environ: Mapping[str, str], directory: Path
) -> dict[str, str]:
    """``{AITER_JIT_DIR: directory}`` with the image's prebuilt modules linked into ``directory``, for a server that
    would otherwise build into the image; empty when the cache's session gives aiter its directory, the farm rebuilds
    aiter (its ``jit`` is the farm's own: the links to the image's modules its tuned tables make stale are taken out of
    it, so the server builds them from the farm's tables), something already sets the variable, or the server's
    interpreter has no aiter.

    ``environ`` is what the server inherits, before the sheet's ``call.env_overrides`` and ``call.unset_envs``.
    """
    variable = AITER_JIT.env_var
    if session is not None and variable in session.environment():
        return {}
    configs = [yaml.safe_load(Path(point).read_text(encoding=_UTF8)) or {} for point in call.points]
    inherited = {name: value for name, value in environ.items() if name not in call.unset_envs}
    servers = [
        {**inherited, **call.env_overrides, **{str(k): str(v) for k, v in server_environment(config).items()}}
        for config in configs
    ]
    if any((server.get(variable) or "").strip() for server in servers):
        return {}
    try:
        image = locate_package(AITER_PACKAGE, server_python(configs[0], default="python3"), directory.parent)
    except FarmError:
        return {}
    if farm is not None and AITER_PACKAGE in farm.packages:
        _unlink_stale(farm.site / AITER_PACKAGE, image, servers)
        return {}
    directory.mkdir(parents=True, exist_ok=True)
    link_prebuilt(image / "jit", directory, _stale(image, servers))
    return {variable: str(directory)}


def _stale(image: Path, servers: list[dict[str, str]]) -> frozenset[str]:
    """The prebuilt modules the tables the servers name make stale; none when no table is named or the module config
    cannot be read (the image's modules are then what the server would have loaded anyway)."""
    if not any(
        name.startswith(_TABLE_PREFIX) and value.strip() for server in servers for name, value in server.items()
    ):
        return frozenset()
    try:
        config = read_config(config_path(image))
    except (OSError, ValueError):
        return frozenset()
    stale = stale_table_modules(config, image, image, servers)
    return touched_entries(stale, config, prebuilt_names(image)).names if stale else frozenset()


def _unlink_stale(package: Path, image: Path, servers: list[dict[str, str]]) -> None:
    """Take out of the farm's ``jit`` (links to the image's modules) the modules the farm's tuned tables, as the servers
    resolve them, make stale: aiter builds a module only when its ``.so`` is missing, and the image's holds only the
    kernels the image's tables name."""
    try:
        config = read_config(config_path(package))
    except (OSError, ValueError):
        return
    stale = stale_table_modules(config, image, package, servers)
    if not stale:
        return
    for name in touched_entries(stale, config, prebuilt_names(image)).names:
        link = package / "jit" / f"{name}.so"
        if link.is_symlink():
            link.unlink()
