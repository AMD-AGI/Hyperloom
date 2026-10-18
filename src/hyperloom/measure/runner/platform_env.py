# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The platform's own environment variables, which a Pod sets and an image does not (design 4.8.1).

A Pod's value overrides the image's setting and the platform cannot delete a variable its template carries, so the
runner does both: it removes the job-topology variables once it has read them, and brings the RCCL and HSA variables to
the values the instruction sheet carries.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping

from hyperloom.measure.spec import PLATFORM_ENV_NAMES

__all__ = ["align_platform_env", "take_topology"]

_TOPOLOGY_NAMES = frozenset({"MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK"})
_TOPOLOGY_PREFIX = "PET_"


def take_topology(environ: MutableMapping[str, str]) -> dict[str, str]:
    """Remove the job-topology variables from ``environ`` and return them, so no later process sees them."""
    taken = {
        name: value for name, value in environ.items() if name in _TOPOLOGY_NAMES or name.startswith(_TOPOLOGY_PREFIX)
    }
    for name in taken:
        del environ[name]
    return taken


def align_platform_env(environ: MutableMapping[str, str], wanted: Mapping[str, str]) -> tuple[str, ...]:
    """Make the platform variables in ``environ`` equal ``wanted``: set those it has, delete those it lacks.

    Returns the names that were changed or removed, sorted, for the result manifest.

    Raises:
        ValueError: ``wanted`` names a variable that is not a platform variable.
    """
    unknown = sorted(set(wanted) - set(PLATFORM_ENV_NAMES))
    if unknown:
        raise ValueError(f"not platform variables: {unknown}")
    changed = []
    for name in PLATFORM_ENV_NAMES:
        want, have = wanted.get(name), environ.get(name)
        if want == have:
            continue
        changed.append(name)
        if want is None:
            del environ[name]
        else:
            environ[name] = want
    return tuple(sorted(changed))
