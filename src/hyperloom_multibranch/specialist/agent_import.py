# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What an autonomous branch submits for each change its agent keeps.

The agent edits a workspace of its own, a checkout of the session's source base at the root, and keeps the server
arguments and environment it runs with in a file the manifest names. A keep is read as a :class:`KeptChange`: the
workspace's patch against the root and the agent's whole config, both cumulative. :func:`import_delta` turns it into
the delta against the branch head, which holds only what MB committed: the head's own import is replaced by the new
cumulative patch, so a keep the gate rejected is not on the head and comes back inside the next one. A kernel table the
agent loads from a file of its workspace goes in as a kernel artifact, the text the file held at the keep.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

from hyperloom.common.outside_facts import TUNABLEOP_TABLE_ENV
from hyperloom.contracts.recipe import (
    ComponentChange,
    ConfigDelta,
    KernelArtifact,
    LaunchConfig,
    PatchRef,
    Recipe,
    RecipeDelta,
)
from hyperloom.measure.framework.tables import AITER_VARIABLE_PREFIX
from hyperloom_multibranch.specialist.agent_manifest import ConfigSource, read_field
from hyperloom_multibranch.specialist.worktree import Upstream

#: Stores a component's patch text and returns its reference, as ``ExperimentRepo.store_patch`` does.
PatchStore = Callable[[str, str], Awaitable[PatchRef]]
#: Keeps the kernel table text a variable loads and returns its artifact.
TableStore = Callable[[str, str], Awaitable[KernelArtifact]]


@dataclass(frozen=True)
class KeepStores:
    """Where an import keeps what a keep carries: its patches and its kernel tables."""

    patch: PatchStore
    table: TableStore


class KeptChangeError(ValueError):
    """A keep whose config the manifest's config source cannot read."""


@dataclass(frozen=True)
class KeptChange:
    """The agent's state at its ``seq``-th keep: ``source_patch`` against the root and the whole ``server_args`` and
    ``envs`` it runs with; ``summary`` is the agent's own words for it; ``tables``, the text at the keep of each kernel
    table file of its workspace a variable of ``envs`` names, by that path."""

    seq: int
    source_patch: str
    server_args: str
    envs: Mapping[str, str] = field(default_factory=dict)
    summary: str = ""
    tables: Mapping[str, str] = field(default_factory=dict)


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


def with_tables(kept: KeptChange, owned: Path) -> KeptChange:
    """``kept`` with the text of each kernel table its variables load from a file under ``owned``, read now: the agent
    may rewrite the file after the keep, and a path into the branch's workspace loads nowhere else.

    Raises :class:`KeptChangeError` for such a file that cannot be read.
    """

    tables: dict[str, str] = {}
    for name, value in kept.envs.items():
        path = Path(value)
        owned_file = path.is_absolute() and path.resolve().is_relative_to(owned.resolve()) and path.is_file()
        if not (_loads_table(name) and owned_file):
            continue
        try:
            tables[value] = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise KeptChangeError(f"the table {value} that {name} loads cannot be read: {exc}") from exc
    return replace(kept, tables=tables)


async def import_delta(
    kept: KeptChange, *, root: Recipe, head: Recipe, component: str, stores: KeepStores
) -> RecipeDelta | None:
    """The delta that makes ``head`` the agent's state at ``kept`` on the framework ``component``; ``None`` when the
    head already is that state. Each table of :attr:`KeptChange.tables` goes in as the artifact ``stores`` keeps it
    as, which sets its variable, so the variable is not imported as well."""

    artifacts = {
        name: await stores.table(name, kept.tables[value]) for name, value in kept.envs.items() if value in kept.tables
    }
    change = await _source_change(
        kept.source_patch, _patches(root, component), _patches(head, component), component, stores.patch
    )
    envs = {name: value for name, value in kept.envs.items() if name not in artifacts}
    config = _config_change(head.launch, replace(kept, envs=envs))
    add, remove = _kernel_change(root.kernels, head.kernels, tuple(artifacts.values()))
    if change is None and config == ConfigDelta() and not add and not remove:
        return None
    return RecipeDelta(
        config=config, components=() if change is None else (change,), add_kernels=add, remove_kernels=remove
    )


def added_at(delta: RecipeDelta, head: Recipe, upstream: Upstream) -> RecipeDelta:
    """``delta`` with each component the ``head`` does not have yet added at ``upstream``, the repository and commit
    its installed package was built from, which the recipe needs before it takes a patch of it."""

    present = {item.component for item in head.components}
    changes = tuple(
        change
        if change.component in present
        else replace(change, upstream_repo=upstream.repo, upstream_ref=upstream.commit)
        for change in delta.components
    )
    return replace(delta, components=changes)


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


def _kernel_change(
    root: tuple[KernelArtifact, ...], head: tuple[KernelArtifact, ...], wanted: tuple[KernelArtifact, ...]
) -> tuple[tuple[KernelArtifact, ...], tuple[str, ...]]:
    imported = [artifact for artifact in head if artifact not in root]
    add = tuple(artifact for artifact in wanted if artifact not in head)
    return add, tuple(artifact.ref for artifact in imported if artifact not in wanted)


def _loads_table(name: str) -> bool:
    return name.startswith(AITER_VARIABLE_PREFIX) or name == TUNABLEOP_TABLE_ENV


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


__all__ = [
    "KeepStores",
    "KeptChange",
    "KeptChangeError",
    "PatchStore",
    "TableStore",
    "added_at",
    "import_delta",
    "read_kept_change",
    "with_tables",
]
