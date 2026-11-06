# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The aiter modules hyperloom's prewarm built into an image's ``aiter/jit``, and the servers that may load them.

The prewarm (:mod:`hyperloom.measure.runner.native.prewarm`) builds a module once, on CPU, for the card's ``gfx`` and CU
count, under the environment a test's server has when nothing of the test sets a build variable, and lists it in the
manifest beside it (:data:`~hyperloom.orchestrator.actions.executors._aiter_jit.AITER_PREWARM_MANIFEST`, whose format the
launch preflight owns) with the target and the build's variables (:func:`build_record`): each variable the module's
build reads, and the value of each that was set.

aiter builds a module only when its ``.so`` is missing, so a server given the image's module runs it whatever its own
environment says. A server that would have built another one -- for another target, or under another value of one of
those variables -- must build its own: :func:`unlike_prewarmed` names the listed modules it must not be given, and a
listed module whose entry records no variables (the target is all an older prewarm wrote) is one of them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from hyperloom.common.outside_facts import AITER_META_DIR_ENV
from hyperloom.measure.jit.key import plain_build_names, without_card_arch
from hyperloom.orchestrator.actions.executors._aiter_jit import read_prewarm_manifest

__all__ = ["CU_NUM", "build_record", "unlike_prewarmed"]

#: What aiter's build reads the card's CU count from (``aiter/jit/utils/build_targets.py``).
CU_NUM = "CU_NUM"
#: What the CPU build sets itself: the target (an entry's ``gfx``) and the meta directory it lays.
_SET_BY_BUILD = frozenset({"GPU_ARCHS", AITER_META_DIR_ENV})
_READS, _VALUES = "build_variables", "build_environment"
_SO = ".so"


def build_record(names: Iterable[str], environ: Mapping[str, str]) -> dict[str, Any]:
    """What an entry records of a build under ``environ`` whose module reads the variables ``names``: those taken by
    their value (:func:`~hyperloom.measure.jit.key.plain_build_names`), and the value of each that was set, less the
    target (``GPU_ARCHS``, and ``CU_NUM``: the entry's ``gfx`` and ``cu_num``) and the meta directory the build lays."""
    reads = plain_build_names(names)
    values = {name: environ[name] for name in sorted(reads - _SET_BY_BUILD - {CU_NUM}) if name in environ}
    return {_READS: sorted(reads), _VALUES: values}


def unlike_prewarmed(jit_dir: Path, servers: Iterable[Mapping[str, str]], gfx_arch: str | None) -> frozenset[str]:
    """The modules the manifest in ``jit_dir`` lists, and ``jit_dir`` holds, that a server of ``servers`` would build
    otherwise: for another architecture than the card's ``gfx_arch`` (or when that is not known), or under a value of a
    variable their build reads other than the recorded one -- a ``CU_NUM`` other than the entry's count, a ``GPU_ARCHS``
    naming another target than the card, any other variable set otherwise -- and those whose entry records none."""
    servers = list(servers)
    return frozenset(
        name
        for name, entry in read_prewarm_manifest(jit_dir).items()
        if (Path(jit_dir) / f"{name}{_SO}").is_file() and _unlike(entry, servers, gfx_arch)
    )


def _unlike(entry: Mapping[str, Any], servers: Sequence[Mapping[str, str]], gfx_arch: str | None) -> bool:
    reads, recorded = entry.get(_READS), entry.get(_VALUES)
    if (
        gfx_arch is None
        or entry.get("gfx") != gfx_arch
        or not isinstance(reads, list)
        or not isinstance(recorded, dict)
    ):
        return True
    return any(_as_built(server, reads, gfx_arch, entry.get("cu_num")) != recorded for server in servers)


def _as_built(server: Mapping[str, str], reads: Sequence[str], gfx_arch: str, cu_num: object) -> dict[str, str]:
    """The server's values of the variables ``reads`` as the entry records them: ``GPU_ARCHS`` naming the card, and
    ``CU_NUM`` the entry's count, name the target the entry states."""
    values = without_card_arch({name: server[name] for name in reads if name in server}, gfx_arch)
    if cu_num is not None and values.get(CU_NUM, "").strip() == str(cu_num):
        del values[CU_NUM]
    return values
