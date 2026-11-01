# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What an autonomous branch submits for each change its agent keeps.

The agent edits a workspace of its own, a checkout of the session's source base at the root, and keeps the server
arguments and environment it runs with in a file the manifest names. A keep is read as a :class:`KeptChange`: the
workspace's patch against the root and the agent's whole config, both cumulative. :func:`import_delta` turns it into
the delta against the branch head, which holds only what MB committed. Its source goes in as one more patch on the
head's, the increment from the state the head's imports make to the keep's, so a commit carries only its own change to
a merge; a keep the gate rejected is not on the head and comes back inside the next increment. A kernel table the agent
loads from files of its own goes in as a kernel artifact, the text the files held at the keep: an aiter variable's
value may list the image's default tables before them, which the artifact's rows are read over at a test as they were
in the agent's run, so it carries only the agent's rows.

An agent run as a component writes each keep whole instead, as a snapshot directory (:func:`read_snapshot`, the layout
of Arbor's component mode): ``keep.json`` names the keep before it and the commits it was made on and made, its
``change.patch`` takes the one to the other, ``config.json`` is its whole launch, of which only what differs from the
launch it was started with goes in, and ``tables/<VAR>/`` the kernel table each variable loads, with the digest
``keep.json`` lists (a table too large to copy is listed with no file); for an aiter value that lists several files,
each of them, of which the image's default tables, MB's private copies of them included, carry no rows. Its source is
the chain of its patches from the root, so a snapshot made on a commit that no snapshot of the chain made, or on a
root other than the one MB gave, is refused; one made on the root after another, its ``notes`` saying the earlier commit
is gone, starts the chain again with ``served.patch``, its whole tree against the root. What the agent's tree links to
outside itself (``outside.json``) is not read: AITER builds a JIT module it lacks at its first call.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath

from hyperloom.common.outside_facts import SGLANG_IMAGE_AITER_CONFIGS, TUNABLEOP_TABLE_ENV
from hyperloom.contracts.recipe import (
    ComponentChange,
    ConfigDelta,
    KernelArtifact,
    LaunchConfig,
    PatchRef,
    Recipe,
    RecipeDelta,
)
from hyperloom.measure.framework.rows import DEFAULT_TABLE_PLACEHOLDER
from hyperloom.measure.framework.tables import AITER_VARIABLE_PREFIX
from hyperloom.measure.surface import measurement_envs
from hyperloom_multibranch.specialist.agent_manifest import ConfigSource, read_field
from hyperloom_multibranch.specialist.worktree import Upstream
from hyperloom_multibranch.toolbox.kernel_artifacts import image_aiter_configs

#: Stores a component's patch text and returns its reference, as ``ExperimentRepo.store_patch`` does.
PatchStore = Callable[[str, str], Awaitable[PatchRef]]
#: Keeps the kernel table text a variable loads and returns its artifact, rows read over the image's default table when
#: the flag is set.
TableStore = Callable[[str, str, bool], Awaitable[KernelArtifact]]
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


def _image_tables() -> tuple[PurePosixPath, ...]:
    """Where the image keeps aiter's default tables: the installed aiter's (a vLLM image's site-packages) and an SGLang
    image's, either of which a keep's aiter variable may name."""

    installed = PurePosixPath(image_aiter_configs().as_posix())
    return tuple(dict.fromkeys((installed, PurePosixPath(SGLANG_IMAGE_AITER_CONFIGS))))


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
    ``envs`` it runs with; ``summary`` is the agent's own words for it; ``tables``, the text at the keep of the kernel
    table files of its own a variable of ``envs`` lists, one table by the variable's value; ``uncopied``, the variables
    whose table the agent did not copy into its snapshot, which then load what the variable names; ``dropped``, the
    variables the agent runs with that the measurement sets itself (workload, evaluation, request), left out of
    ``envs``."""

    seq: int
    source_patch: str
    server_args: str
    envs: Mapping[str, str] = field(default_factory=dict)
    summary: str = ""
    tables: Mapping[str, str] = field(default_factory=dict)
    uncopied: tuple[str, ...] = ()
    dropped: Mapping[str, str] = field(default_factory=dict)


def read_kept_change(
    seq: int, source_patch: str, state: object, source: ConfigSource, summary: str, start: LaunchConfig
) -> KeptChange:
    """The keep ``seq`` of an agent whose workspace differs from the root by ``source_patch``, whose config file, which
    ``source`` describes, parsed to ``state``, and which was started with ``start``."""

    try:
        args, envs = read_field(state, source.server_args), read_field(state, source.envs)
    except (KeyError, IndexError, ValueError, TypeError) as exc:
        raise KeptChangeError(f"{source.path} holds no {exc}") from exc
    if not isinstance(args, str) or not isinstance(envs, (Mapping, str)):
        raise KeptChangeError(
            f"{source.path}: {source.server_args} is not a string or {source.envs} not an environment"
        )
    named = (
        _assignments(envs, source) if isinstance(envs, str) else {str(name): str(value) for name, value in envs.items()}
    )
    if source.adds:
        args, named = _over(start, ("", {}), args, named)
    named, dropped = _without_measurement(named)
    return KeptChange(seq, source_patch, args, named, summary, dropped=dropped)


def _assignments(words: str, source: ConfigSource) -> dict[str, str]:
    try:
        split = [word.partition("=") for word in shlex.split(words)]
    except ValueError as exc:
        raise KeptChangeError(f"{source.path}: {source.envs} is not NAME=VALUE words: {exc}") from exc
    if any(not name or not equals for name, equals, _ in split):
        raise KeptChangeError(f"{source.path}: {source.envs} is not NAME=VALUE words: {words!r}")
    return {name: value for name, _, value in split}


#: Server arguments and environment as an agent's config holds them, before they are a :class:`LaunchConfig`.
_Launch = tuple[str, Mapping[str, str]]


def _over(base: LaunchConfig, since: _Launch, args: str, envs: Mapping[str, str]) -> _Launch:
    """``base`` with what ``args`` and ``envs`` change from ``since``: each flag set otherwise set as ``args`` sets it,
    each flag and variable of ``since`` they no longer have dropped, each variable set otherwise set."""

    groups, before = _by_flag(args), _by_flag(since[0])
    changed = {name for name, words in groups.items() if before.get(name) != words}
    gone = changed | (before.keys() - groups.keys())
    kept = [words for name, words in _flag_groups(base.server_args) if name not in gone]
    added = [words for name, words in _flag_groups(args) if name in changed]
    unset = since[1].keys() - envs.keys()
    named = {name: value for name, value in base.envs.items() if name not in unset}
    named.update({name: value for name, value in envs.items() if since[1].get(name) != value})
    return " ".join(word for words in [*kept, *added] for word in words), named


def _by_flag(args: str) -> dict[str, list[list[str]]]:
    groups: dict[str, list[list[str]]] = {}
    for name, words in _flag_groups(args):
        groups.setdefault(name, []).append(words)
    return groups


def _without_measurement(envs: Mapping[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    """``envs`` without the variables the measurement sets itself, which a test refuses in its launch, and those."""

    owned = set(measurement_envs(envs))
    return {name: value for name, value in envs.items() if name not in owned}, {name: envs[name] for name in owned}


def _flag_groups(args: str) -> list[tuple[str, list[str]]]:
    """``args`` as each flag with the words up to the next, words before the first flag as one group of no name."""

    groups: list[tuple[str, list[str]]] = []
    for word in args.split():
        if word.startswith("--"):
            groups.append((word.split("=", 1)[0], [word]))
        elif groups:
            groups[-1][1].append(word)
        else:
            groups.append(("", [word]))
    return groups


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


@dataclass(frozen=True)
class ComponentStart:
    """What MB started a component agent with: ``launch``, the start commit's; ``config``, the config MB wrote, which
    also carries what the agent's own harness reads; ``copies``, MB's private copies of the image's default tables the
    agent's aiter variables list, which read as those tables."""

    launch: LaunchConfig
    config: Mapping[str, object]
    copies: Collection[str]


def read_snapshot(outbox: Path, number: int, root: str, applied: Applied, start: ComponentStart) -> KeptChange:
    """The keep ``number`` of the snapshots in ``outbox``: its source is what the patches of the chain of snapshots it
    was made on make of ``root`` (``applied``), its kernel tables its own, and its launch ``start.launch`` with what
    the snapshot's launch changes from the one ``start.config`` starts the agent with (from ``start.launch`` when that
    config names none).

    Raises :class:`KeptChangeError` for a snapshot that cannot be read, or whose chain does not reach ``root``.
    """

    chain = _chain(outbox, number, root)
    patches = [_patch(outbox / _directory(link.number), SERVED_FILE if link.rebased else CHANGE_FILE) for link in chain]
    directory = outbox / _directory(number)
    launch = _component_launch(read_json(directory / LAUNCH_FILE))
    if launch is None:
        raise KeptChangeError(f"{_directory(number)}/{LAUNCH_FILE} holds no server_flags string or server_env object")
    named, dropped = _without_measurement(launch[1])
    base = start.launch
    args, named = _over(base, _component_launch(start.config) or (base.server_args, base.envs), launch[0], named)
    tables, uncopied = _snapshot_tables(directory / TABLES_DIR, chain[-1].tables, named, start.copies)
    return KeptChange(number, applied(patches), args, named, chain[-1].summary, tables, uncopied, dropped)


def _component_launch(config: object) -> _Launch | None:
    """The ``server_flags`` and ``server_env`` of a config of Arbor's, ``None`` when it holds no such pair."""

    args = config.get("server_flags") if isinstance(config, Mapping) else None
    envs = config.get("server_env") if isinstance(config, Mapping) else None
    if not isinstance(args, str) or not isinstance(envs, Mapping):
        return None
    return args, {str(name): str(value) for name, value in envs.items()}


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
    tables: Path, listed: Mapping[str, object], envs: Mapping[str, str], copies: Collection[str]
) -> tuple[dict[str, str], tuple[str, ...]]:
    """The kernel tables of ``envs`` (:func:`_kernel_tables`) the snapshot copied to ``tables``, each file checked
    against the digest ``listed`` gives it; and the variables listed with a file not copied. An entry lists one file,
    or under ``files`` each file of a value that lists several, the image's default tables and ``copies`` among them,
    which carry no rows."""

    def copied(name: str, value: str) -> list[str] | None:
        entry = listed.get(name)
        if not isinstance(entry, Mapping) or _uncopied(entry, name, value, copies):
            return None
        if "files" not in entry:
            return [_copied_text(tables / name, name, entry)]
        return _copied_parts(tables / name, name, value, entry["files"], copies)

    uncopied = tuple(
        name
        for name, value in envs.items()
        if _loads_table(name)
        and isinstance(entry := listed.get(name), Mapping)
        and _uncopied(entry, name, value, copies)
    )
    return _kernel_tables(envs, copied), uncopied


def _copied_parts(directory: Path, name: str, value: str, files: object, copies: Collection[str]) -> list[str]:
    """The texts of the copies ``files`` lists of the parts of ``value`` that carry rows, in its order; the image's
    tables are not read."""

    if not isinstance(files, list) or not all(isinstance(item, Mapping) for item in files):
        raise KeptChangeError(f"{KEEP_FILE} lists the files of the table {name} loads as no list of objects")
    by_path = {item.get("path"): item for item in files}
    texts = []
    for part in _own_parts(name, value, copies):
        if part not in by_path:
            raise KeptChangeError(f"{KEEP_FILE} lists no copy of {part}, which {name} loads")
        texts.append(_copied_text(directory, name, by_path[part]))
    return texts


def _uncopied(entry: Mapping[str, object], name: str, value: str, copies: Collection[str]) -> bool:
    """Whether the table entry of ``keep.json`` for ``name`` set to ``value`` lists a file that carries rows and was
    not copied."""

    files = entry.get("files")
    if not isinstance(files, list):
        return entry.get("file") is None
    return any(
        isinstance(item, Mapping) and item.get("file") is None and item.get("path") in _own_parts(name, value, copies)
        for item in files
    )


def _copied_text(directory: Path, name: str, entry: Mapping[str, object]) -> str:
    """The text of the copy under ``directory`` that ``entry`` names, checked against its sha256."""

    file, digest = entry.get("file"), entry.get("sha256")
    if not isinstance(file, str) or not isinstance(digest, str):
        raise KeptChangeError(f"{KEEP_FILE} lists the table {name} loads with no file name and sha256")
    try:
        data = (directory / PurePosixPath(file).name).read_bytes()
    except OSError as error:
        raise KeptChangeError(f"the table {name} loads cannot be read: {error}") from error
    if hashlib.sha256(data).hexdigest() != digest:
        raise KeptChangeError(f"the table {name} loads does not match its sha256 {digest}")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise KeptChangeError(f"the table {name} loads is no text: {error}") from error


def _directory(number: int) -> str:
    return f"{number:04d}"


def _refused(snapshot: _Snapshot) -> str:
    return f"snapshot {_directory(snapshot.number)} is refused:"


def with_tables(kept: KeptChange, owned: Sequence[Path]) -> KeptChange:
    """``kept`` with the text of the kernel tables each of its variables loads from files under ``owned``, read now:
    the agent may rewrite a file after the keep, and a path into the branch's directories loads nowhere else. An aiter
    variable may list the image's default tables too, and several files of its own, which make one table.

    Raises :class:`KeptChangeError` for such a file that cannot be read, or files of one variable under other headers.
    """

    def read(name: str, value: str) -> list[str] | None:
        paths = _own_tables(name, value, owned)
        if paths is None:
            return None
        texts = []
        for path in paths:
            try:
                texts.append(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError) as exc:
                raise KeptChangeError(f"the table {path} that {name} loads cannot be read: {exc}") from exc
        return texts

    return replace(kept, tables=_kernel_tables(kept.envs, read))


def _kernel_tables(envs: Mapping[str, str], own_tables: Callable[[str, str], Sequence[str] | None]) -> dict[str, str]:
    """The kernel table of each variable of ``envs`` that loads one, by the variable's value: the rows of the texts of
    the agent's own tables ``own_tables`` gives for it, as one table. A variable it gives none for stays as the agent
    set it; the rest go in as artifacts (:func:`import_delta`), an aiter one as rows read over the image's default table
    in place of the default paths its value lists."""

    tables: dict[str, str] = {}
    for name, value in envs.items():
        texts = own_tables(name, value) if _loads_table(name) else None
        if texts:
            tables[value] = _joined(texts, name)
    return tables


def _own_tables(name: str, value: str, owned: Sequence[Path]) -> list[Path] | None:
    """The files under ``owned`` that ``value`` lists, ``None`` unless it lists one or more and, for an aiter variable,
    nothing else but the image's default tables."""

    paths = [Path(part) for part in _own_parts(name, value)]
    if not paths or not all(_owned(path, owned) for path in paths):
        return None
    return paths


def _own_parts(name: str, value: str, copies: Collection[str] = ()) -> list[str]:
    """The files ``value`` lists that carry rows of a table: for an aiter variable, which reads several as one table,
    those other than the image's default tables (:func:`_image_table`)."""

    if not _rows_only(name):
        return [value]
    return [part for part in value.split(os.pathsep) if part and not _image_table(part, copies)]


def _image_table(part: str, copies: Collection[str] = ()) -> bool:
    """Whether ``part`` of an aiter value is one of the image's default tables, ``copies`` being MB's private copies
    of them."""
    return (
        part in copies
        or part == DEFAULT_TABLE_PLACEHOLDER
        or any(PurePosixPath(os.path.normpath(part)).is_relative_to(root) for root in _image_tables())
    )


def _owned(path: Path, owned: Sequence[Path]) -> bool:
    return (
        path.is_absolute() and path.is_file() and any(path.resolve().is_relative_to(root.resolve()) for root in owned)
    )


def _joined(texts: Sequence[str], name: str) -> str:
    """One table of the rows of ``texts``, each row once; aiter refuses a shape two of the files set, as it would have
    in the agent's run."""

    header = {text.splitlines()[0] if text.strip() else "" for text in texts}
    if len(header) != 1:
        raise KeptChangeError(f"the tables {name} lists have other headers: {', '.join(sorted(header))}")
    rows = dict.fromkeys(line for text in texts for line in text.splitlines()[1:] if line.strip())
    return "".join(f"{line}\n" for line in (*header, *rows))


async def import_delta(
    kept: KeptChange, *, root: Recipe, head: Recipe, component: str, stores: KeepStores
) -> RecipeDelta | None:
    """The delta that makes ``head`` the agent's state at ``kept`` on the framework ``component``; ``None`` when the
    head already is that state. Each table of :attr:`KeptChange.tables` goes in as the artifact ``stores`` keeps it
    as, which sets its variable, so the variable is not imported as well; an aiter variable's table is rows read over
    the image's default table."""

    artifacts = {
        name: await stores.table(name, kept.tables[value], _rows_only(name))
        for name, value in kept.envs.items()
        if value in kept.tables
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
    return _rows_only(name) or name == TUNABLEOP_TABLE_ENV


def _rows_only(name: str) -> bool:
    return name.startswith(AITER_VARIABLE_PREFIX)


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
    "ComponentStart",
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
