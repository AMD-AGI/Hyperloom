# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A recipe delta in one line: what a test changed, for briefs and tool replies; and whether two recipes differ at all.

A brief that only names a direction (target, mechanism, layer) leaves a specialist to guess what was actually tried;
the line names the arguments and environment set or removed, the components moved or patched, and the kernels added
or dropped, in the order :func:`hyperloom.contracts.recipe.apply` applies them.

:func:`same_recipe` is the one rule for two recipes being the same, which a merge writes as a new commit: a replay
asks it of a champion and a head, FINAL of two candidates.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import PurePosixPath

from hyperloom.contracts.recipe import (
    ArgsMode,
    ComponentChange,
    ConfigDelta,
    Recipe,
    RecipeDelta,
    delta_between,
    diff,
)

NO_CHANGE = "no change"
LIST = ", "
PARTS = "; "
#: Characters of an upstream commit a line keeps, as ``git log --oneline`` does.
SHORT_REF = 8


def _names(paths: tuple[str, ...]) -> str:
    return LIST.join(PurePosixPath(path).name for path in paths)


def _config(config: ConfigDelta) -> list[str]:
    parts = []
    if config.args_mode is ArgsMode.REPLACE:
        parts.append(f"replace args with {config.server_args or 'none'}")
    elif config.server_args:
        parts.append(f"add args {config.server_args}")
    if config.remove_args:
        parts.append(f"remove args {' '.join(config.remove_args)}")
    if config.launch_remove_args is not None:
        names = " ".join(config.launch_remove_args)
        parts.append(f"resolved launch removes only {names}" if names else "resolved launch removes nothing")
    if config.envs:
        parts.append(f"set env {LIST.join(f'{name}={value}' for name, value in sorted(config.envs.items()))}")
    if config.unset_envs:
        parts.append(f"unset env {LIST.join(config.unset_envs)}")
    return parts


def _component(change: ComponentChange) -> str:
    pieces = []
    if change.upstream_ref is not None:
        verb = "new at" if change.upstream_repo is not None else "move to"
        pieces.append(f"{verb} {change.upstream_ref[:SHORT_REF]}")
    if change.add_patches:
        pieces.append(f"add {_names(tuple(patch.path for patch in change.add_patches))}")
    if change.remove_patches:
        pieces.append(f"drop {_names(change.remove_patches)}")
    return f"{change.component}: {LIST.join(pieces) or 'unchanged'}"


def describe_change(delta: RecipeDelta) -> str:
    """What ``delta`` changes, in one line; :data:`NO_CHANGE` for an empty delta, such as a baseline retest's."""

    parts = [
        *_config(delta.config),
        *map(_component, delta.components),
        *(f"drop kernel {PurePosixPath(ref).name}" for ref in delta.remove_kernels),
        *(f"add {artifact.kind.value} kernel for {artifact.target}" for artifact in delta.add_kernels),
    ]
    return PARTS.join(parts) or NO_CHANGE


def same_recipe(first: Recipe, second: Recipe) -> bool:
    """Whether ``first`` is ``second`` already: no component or kernel artifact :func:`delta_between` would change, and
    the same server arguments and environment, flag by flag whatever their order.

    Raises ``RecipeError`` for a component the two take from different upstream repositories, as ``delta_between``
    does.
    """

    change = diff(first, second)
    rest = replace(delta_between(first, second), config=ConfigDelta())
    return not (change.args or change.envs) and rest == RecipeDelta()


__all__ = ["NO_CHANGE", "describe_change", "same_recipe"]
