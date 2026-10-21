# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What an autonomous branch submits for each change its agent keeps.

The agent edits a workspace of its own, a checkout of the session's source base at the root, and keeps the server
arguments and environment it runs with in a file the manifest names. A keep is read as a :class:`KeptChange`: the
workspace's patch against the root and the agent's whole config, both cumulative. :func:`import_delta` turns it into
the delta against the branch head, which holds only what MB committed: the head's own import is replaced by the new
cumulative patch, so a keep the gate rejected is not on the head and comes back inside the next one.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field

from hyperloom.contracts.recipe import (
    ComponentChange,
    ConfigDelta,
    LaunchConfig,
    PatchRef,
    Recipe,
    RecipeDelta,
)
from hyperloom_multibranch.specialist.agent_manifest import ConfigSource, read_field

#: Stores a component's patch text and returns its reference, as ``ExperimentRepo.store_patch`` does.
PatchStore = Callable[[str, str], Awaitable[PatchRef]]


class KeptChangeError(ValueError):
    """A keep whose config the manifest's config source cannot read."""


@dataclass(frozen=True)
class KeptChange:
    """The agent's state at its ``seq``-th keep: ``source_patch`` against the root and the whole ``server_args`` and
    ``envs`` it runs with; ``summary`` is the agent's own words for it."""

    seq: int
    source_patch: str
    server_args: str
    envs: Mapping[str, str] = field(default_factory=dict)
    summary: str = ""


def read_kept_change(seq: int, source_patch: str, state: object, source: ConfigSource, summary: str) -> KeptChange:
    """The keep ``seq`` of an agent whose workspace differs from the root by ``source_patch`` and whose config file,
    which ``source`` describes, parsed to ``state``."""

    try:
        args, envs = read_field(state, source.server_args), read_field(state, source.envs)
    except (KeyError, IndexError, ValueError, TypeError) as exc:
        raise KeptChangeError(f"{source.path} holds no {exc}") from exc
    if not isinstance(args, str) or not isinstance(envs, Mapping):
        raise KeptChangeError(f"{source.path}: {source.server_args} is not a string or {source.envs} not an object")
    return KeptChange(seq, source_patch, args, {str(name): str(value) for name, value in envs.items()}, summary)


async def import_delta(
    kept: KeptChange, *, root: Recipe, head: Recipe, component: str, store: PatchStore
) -> RecipeDelta | None:
    """The delta that makes ``head`` the agent's state at ``kept`` on the framework ``component``; ``None`` when the
    head already is that state."""

    change = await _source_change(
        kept.source_patch, _patches(root, component), _patches(head, component), component, store
    )
    config = _config_change(head.launch, kept)
    if change is None and config == ConfigDelta():
        return None
    return RecipeDelta(config=config, components=() if change is None else (change,))


def _patches(recipe: Recipe, component: str) -> tuple[PatchRef, ...]:
    return next((item.patches for item in recipe.components if item.component == component), ())


async def _source_change(
    text: str, root: tuple[PatchRef, ...], head: tuple[PatchRef, ...], component: str, store: PatchStore
) -> ComponentChange | None:
    imported = tuple(ref for ref in head if ref not in root)
    wanted = await store(component, text) if text else None
    if (wanted,) == imported or (wanted is None and not imported):
        return None
    return ComponentChange(
        component,
        remove_patches=tuple(ref.path for ref in imported),
        add_patches=() if wanted is None else (wanted,),
    )


def _config_change(launch: LaunchConfig, kept: KeptChange) -> ConfigDelta:
    replaced = kept.server_args != launch.server_args
    dropped = _flags(launch.server_args) - _flags(kept.server_args) if replaced else set()
    envs = {name: value for name, value in kept.envs.items() if launch.envs.get(name) != value}
    unset = tuple(sorted(name for name in launch.envs if name not in kept.envs))
    return ConfigDelta(
        server_args=kept.server_args if replaced else "",
        remove_args=tuple(sorted(dropped)),
        envs=envs,
        unset_envs=unset,
    )


def _flags(args: str) -> set[str]:
    return {token.split("=", 1)[0] for token in args.split() if token.startswith("--")}


__all__ = ["KeptChange", "KeptChangeError", "PatchStore", "import_delta", "read_kept_change"]
