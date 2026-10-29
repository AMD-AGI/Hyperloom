# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What an autonomous branch submits for each change its agent keeps.

The agent edits a workspace of its own, a checkout of the session's source base at the root, and keeps the server
arguments and environment it runs with in a file the manifest names. A keep is read as a :class:`KeptChange`: the
workspace's patch against the root and the agent's whole config, both cumulative. :func:`import_delta` turns it into
the delta against the branch head, which holds only what MB committed. Its source goes in as one more patch on the
head's, the increment from the state the head's imports make to the keep's, so a commit carries only its own change to
a merge; a keep the gate rejected is not on the head and comes back inside the next increment. A kernel table the agent
loads from a file of its workspace goes in as a kernel artifact, the text the file held at the keep.

An agent run as a component writes each keep whole instead, as a snapshot directory (:func:`read_snapshot`, the layout
of Arbor's component mode): ``keep.json`` names the keep before it and the commits it was made on and made, its
``change.patch`` takes the one to the other, ``config.json`` is its whole launch and ``tables/<VAR>/`` the kernel table
each variable loads, with the digest ``keep.json`` lists (a table too large to copy is listed with no file). Its source
is the chain of its patches from the root, so a snapshot made on a commit that no snapshot of the chain made, or on a
root other than the one MB gave, is refused; one made on the root after another, its ``notes`` saying the earlier commit
is gone, starts the chain again with ``served.patch``, its whole tree against the root. What the agent's tree links to
outside itself (``outside.json``) is not read: AITER builds a JIT module it lacks at its first call.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath

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
#: The patch the given patches make, applied in order to the root.
Applied = Callable[[Sequence[str]], str]
#: The file of a component's outbox naming its newest snapshot, and the files of one snapshot.
LATEST_FILE = "LATEST"
KEEP_FILE = "keep.json"
CHANGE_FILE = "change.patch"
SERVED_FILE = "served.patch"
LAUNCH_FILE = "config.json"
TABLES_DIR = "tables"
#: The patch from the state the head's imported patches make, in order, to the state a keep's cumulative patch makes,
#: both over the root; empty when they are the same.
SourceIncrement = Callable[[tuple[PatchRef, ...], str], Awaitable[str]]


@dataclass(frozen=True)
class KeepStores:
    """Where an import keeps what a keep carries, its patches and its kernel tables, and how it reads the keep's
    source as an increment over the head's imports."""

    patch: PatchStore
    table: TableStore
    increment: SourceIncrement


class KeptChangeError(ValueError):
    """A keep whose config the manifest's config source cannot read."""


@dataclass(frozen=True)
class KeptChange:
    """The agent's state at its ``seq``-th keep: ``source_patch`` against the root and the whole ``server_args`` and
    ``envs`` it runs with; ``summary`` is the agent's own words for it; ``tables``, the text at the keep of each kernel
    table file of its workspace a variable of ``envs`` names, by that path; ``uncopied``, the variables whose table the
    agent did not copy into its snapshot, which then load what the variable names."""

    seq: int
    source_patch: str
    server_args: str
    envs: Mapping[str, str] = field(default_factory=dict)
    summary: str = ""
    tables: Mapping[str, str] = field(default_factory=dict)
    uncopied: tuple[str, ...] = ()


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


def read_json(path: Path) -> object:
    """The JSON ``path`` holds; raise :class:`KeptChangeError` when it cannot be read."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise KeptChangeError(f"{path.name} cannot be read: {error}") from error


def latest_snapshot(outbox: Path) -> int | None:
    """The keep the ``LATEST`` file of ``outbox`` names, ``None`` before the first; raise ``OSError`` when it cannot be
    read and ``ValueError`` when it names no keep."""

    try:
        named = (outbox / LATEST_FILE).read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    return int(named.strip())


@dataclass(frozen=True)
class _Snapshot:
    """One snapshot's ``keep.json``; ``rebased`` when its change is against the root, though a keep came before it."""

    number: int
    previous: int | None
    parent: str
    commit: str
    summary: str
    tables: Mapping[str, object]
    rebased: bool


def read_snapshot(outbox: Path, number: int, root: str, applied: Applied) -> KeptChange:
    """The keep ``number`` of the snapshots in ``outbox``: its source is what the patches of the chain of snapshots it
    was made on make of ``root`` (``applied``), its launch and kernel tables its own.

    Raises :class:`KeptChangeError` for a snapshot that cannot be read, or whose chain does not reach ``root``.
    """

    chain = _chain(outbox, number, root)
    patches = [_patch(outbox / _directory(link.number), SERVED_FILE if link.rebased else CHANGE_FILE) for link in chain]
    directory = outbox / _directory(number)
    launch = read_json(directory / LAUNCH_FILE)
    args = launch.get("server_flags") if isinstance(launch, Mapping) else None
    envs = launch.get("server_env") if isinstance(launch, Mapping) else None
    if not isinstance(args, str) or not isinstance(envs, Mapping):
        raise KeptChangeError(f"{_directory(number)}/{LAUNCH_FILE} holds no server_flags string or server_env object")
    named = {str(name): str(value) for name, value in envs.items()}
    tables, uncopied = _snapshot_tables(directory / TABLES_DIR, chain[-1].tables, named)
    return KeptChange(number, applied(patches), args, named, chain[-1].summary, tables, uncopied)


def _chain(outbox: Path, number: int, root: str) -> list[_Snapshot]:
    """The snapshots from the first, or the last one made on ``root``, to ``number``, each made on the commit the one
    its ``previous`` names made."""

    chain = [_snapshot(outbox, number, root)]
    while (later := chain[-1]).previous is not None and not later.rebased:
        if later.previous >= later.number:
            raise KeptChangeError(f"{_refused(later)} it names {later.previous} as the keep before it")
        try:
            earlier = _snapshot(outbox, later.previous, root)
        except KeptChangeError as error:
            raise KeptChangeError(f"{_refused(later)} the snapshot it was made on is unread: {error}") from error
        if later.parent != earlier.commit:
            made = f"{_directory(earlier.number)}'s commit {earlier.commit}"
            raise KeptChangeError(f"{_refused(later)} it was made on {later.parent}, not on {made}")
        chain.append(earlier)
    if chain[-1].parent != root:
        raise KeptChangeError(f"{_refused(chain[-1])} it was made on {chain[-1].parent}, not on the root {root}")
    return chain[::-1]


def _snapshot(outbox: Path, number: int, root: str) -> _Snapshot:
    keep = read_json(outbox / _directory(number) / KEEP_FILE)
    if not isinstance(keep, Mapping):
        raise KeptChangeError(f"{_directory(number)}/{KEEP_FILE} is no object")
    previous, parent, commit, tables = keep.get("previous"), keep.get("parent"), keep.get("commit"), keep.get("tables")
    if keep.get("number") != number or not (previous is None or isinstance(previous, int)):
        raise KeptChangeError(f"{_directory(number)}/{KEEP_FILE} numbers it {keep.get('number')!r}, after {previous!r}")
    if not isinstance(parent, str) or not isinstance(commit, str):
        raise KeptChangeError(f"{_directory(number)}/{KEEP_FILE} names no parent and commit")
    if not isinstance(tables, Mapping):
        raise KeptChangeError(f"{_directory(number)}/{KEEP_FILE} lists no tables")
    # Arbor makes a keep on the root when the commit of the one before is gone from its workspace, and says why.
    notes = keep.get("notes")
    rebased = previous is not None and parent == root and isinstance(notes, str) and bool(notes.strip())
    snapshot = _Snapshot(number, previous, parent, commit, _said(keep), tables, rebased)
    if keep.get("root") != root:
        raise KeptChangeError(f"{_refused(snapshot)} it was made against the root {keep.get('root')}, not {root}")
    return snapshot


def _said(keep: Mapping[str, object]) -> str:
    """The agent's own words for a keep: its change, else its intent."""
    return next((said for said in (keep.get("change"), keep.get("intent")) if isinstance(said, str) and said), "")


def _patch(directory: Path, name: str) -> str:
    try:
        return (directory / name).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise KeptChangeError(f"{directory.name}/{name} cannot be read: {error}") from error


def _snapshot_tables(
    tables: Path, listed: Mapping[str, object], envs: Mapping[str, str]
) -> tuple[dict[str, str], tuple[str, ...]]:
    """The text of the kernel table each variable of ``envs`` that loads one has in ``tables``, by the variable's value,
    each checked against the digest ``listed`` gives it; and the variables listed with no file, which were not copied.
    """

    found: dict[str, str] = {}
    uncopied: list[str] = []
    for name, value in envs.items():
        entry = listed.get(name)
        if not (_loads_table(name) and isinstance(entry, Mapping)):
            continue
        file, digest = entry.get("file"), entry.get("sha256")
        if file is None:
            uncopied.append(name)
            continue
        if not isinstance(file, str) or not isinstance(digest, str):
            raise KeptChangeError(f"{KEEP_FILE} lists the table {name} loads with no file name and sha256")
        try:
            data = (tables / name / PurePosixPath(file).name).read_bytes()
        except OSError as error:
            raise KeptChangeError(f"the table {name} loads cannot be read: {error}") from error
        if hashlib.sha256(data).hexdigest() != digest:
            raise KeptChangeError(f"the table {name} loads does not match its sha256 {digest}")
        try:
            found[value] = data.decode("utf-8")
        except UnicodeDecodeError as error:
            raise KeptChangeError(f"the table {name} loads is no text: {error}") from error
    return found, tuple(uncopied)


def _directory(number: int) -> str:
    return f"{number:04d}"


def _refused(snapshot: _Snapshot) -> str:
    return f"snapshot {_directory(snapshot.number)} is refused:"


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
    imported = tuple(ref for ref in _patches(head, component) if ref not in _patches(root, component))
    change = await _source_change(await stores.increment(imported, kept.source_patch), component, stores.patch)
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


async def _source_change(increment: str, component: str, store: PatchStore) -> ComponentChange | None:
    if not increment:
        return None
    return ComponentChange(component, add_patches=(await store(component, increment),))


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
    "Applied",
    "CHANGE_FILE",
    "KEEP_FILE",
    "LATEST_FILE",
    "LAUNCH_FILE",
    "SERVED_FILE",
    "TABLES_DIR",
    "KeepStores",
    "KeptChange",
    "KeptChangeError",
    "PatchStore",
    "SourceIncrement",
    "TableStore",
    "added_at",
    "import_delta",
    "latest_snapshot",
    "read_json",
    "read_kept_change",
    "read_snapshot",
    "with_tables",
]
