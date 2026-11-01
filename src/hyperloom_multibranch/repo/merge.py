# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A candidate commit's change carried onto another commit's recipe, for the integrator's merge.

The change is what the candidate's recipe changed from the merge base's: the nearest commit both histories hold, so a
candidate whose parents were never merged brings their change with its own. A server argument or environment variable
both sides changed takes the candidate's value and is listed; a kernel slot both sides filled is decided by the two tables
when one holds every row of the other, since a target loads one table and the wider one serves every shape the
narrower did, and otherwise takes a new table of both tables' rows. A shape both set to other rows takes the row with
the lower time the tuner measured (``us``); where a row has none, the row of the side that moved it from the base's
table, the head's where both did. A union of only the head's rows keeps the head's table.
A component travels only for the patches it gains or loses, with its upstream commit where
those patches create a component the other side lacks: every component is built from the installed package of the
session's fixed base image, so a recipe's upstream pin is provenance no merge can act on, and a component the
candidate records without patching it changes nothing a test would build. The merged patch stacks are applied to the
installed sources (the package, and the source trees beside it such as aiter's csrc) as the toolbox applies them when
it builds a test's environment, so a conflict shows before the merge test is paid for. A patch the candidate brings
that does not apply there is rebased by a three-way merge of the
files it touches. Where a line is changed by both sides, a head patch the candidate's stack does not hold is replaced
by the candidate's patch when that patch rewrites every line the head patch changes (see ``_replacement``); otherwise
the candidate's patch is split by file: the files that merge three ways are carried rebased, and only the rest stays a
conflict for the integrator's resolver (see ``_split``).
"""

from __future__ import annotations

import csv
import io
import math
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path

from hyperloom.contracts.recipe import (
    REMOVED,
    ComponentChange,
    ComponentPatches,
    ConfigDelta,
    KernelArtifact,
    KernelKind,
    LaunchConfig,
    PatchRef,
    Recipe,
    RecipeDelta,
    RecipeDiff,
    RecipeError,
    ValueChange,
    apply,
    diff,
)
from hyperloom.measure.framework.rows import cell
from hyperloom.measure.compile import KernelLoadError, PatchApplyError, kernel_load_env, kernel_rows_only
from hyperloom_multibranch.contracts.repo import PatchConflict, RebasedPatch, ReplacedPatch, RepoMerge, UnitedTable
from hyperloom_multibranch.repo.rebase import Files, merge_files, patch_between
from hyperloom_multibranch.toolbox.kernel_artifacts import (
    KernelRowsConflict,
    KernelRowsError,
    mint_kernel_artifact,
    union_rows,
)

#: Applies a component's patch stack to its installed sources, and returns every file a patch of it touches and each of
#: the given paths as the stack leaves them, by path in the upstream repository; raises ``PatchApplyError`` when a patch
#: does not apply.
StackFiles = Callable[[ComponentPatches, Collection[str]], Files]
#: Reads a kernel table's bytes, for the two that fill one slot.
TableBytes = Callable[[KernelArtifact], bytes]
#: Keeps a patch file of a component, as the experiment repository's ``store_patch`` does.
PatchStore = Callable[[str, str], PatchRef]
#: Keeps a kernel table's bytes in the session's blob store, as the toolbox's ``put_kernel_artifact`` does.
TableStore = Callable[[bytes], str]


@dataclass(frozen=True)
class Stored:
    """What a merge consults beyond the three recipes: the installed packages a patch stack has to apply to, the bytes
    behind a kernel artifact, and the stores a rebased patch and a united table are kept in. ``aiter_configs`` is the
    aiter install a union reads a table's shape key from, the image's when ``None``. ``protected`` are the paths of
    head patches a merge never takes out to replace them: those the integrator's foundation fixes contributed."""

    stack_files: StackFiles
    kernel_table: TableBytes
    store_patch: PatchStore
    store_table: TableStore
    aiter_configs: Path | None = None
    protected: frozenset[str] = frozenset()


@dataclass(frozen=True)
class _Kernels:
    """The kernel artifacts a merge carries, and the slots among them it fills with a union of both sides' tables."""

    added: tuple[KernelArtifact, ...]
    united: tuple[UnitedTable, ...]
    displaced: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Settled:
    """Where the patches of merged stacks that do not apply went: left out, rebased, put in for the head patches they
    replace, or ``held`` by the stack already. ``parts`` are the conflicts split off a candidate's patch, which no
    stack holds."""

    conflicts: tuple[PatchConflict, ...] = ()
    rebased: tuple[RebasedPatch, ...] = ()
    held: frozenset[str] = field(default_factory=frozenset)
    replaced: tuple[ReplacedPatch, ...] = ()
    parts: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class _Sides:
    """The files a candidate's patch touches, three ways: ``original`` under the candidate's stack before it,
    ``changed`` with it, and ``current`` under the stack it now lands on, which also holds every file that stack
    touches."""

    original: Files
    changed: Files
    current: Files


def carry(onto: str, base: Recipe, candidate: Recipe, ours: Recipe, stored: Stored) -> RepoMerge:
    """``candidate``'s change from ``base``, the recipe of its merge base with ``onto``, carried onto ``ours``, the
    recipe of ``onto``."""
    change = diff(base, candidate)
    if change.base_image_changed:
        raise RecipeError("the candidate changes the base image, which a session keeps")
    kernels = _added_kernels(base, candidate, ours, stored)
    delta = RecipeDelta(
        config=_config(change, ours.launch),
        components=_components(base, candidate, ours),
        add_kernels=kernels.added,
        remove_kernels=(*_removed_kernels(base, candidate, ours), *kernels.displaced),
    )
    merged = apply(ours, delta)
    brought = {patch.path for touched in delta.components for patch in touched.add_patches}
    settled = _joined(
        _settle(merged.component(name), candidate.component(name), brought, stored)
        for name in (touched.component for touched in delta.components)
    )
    return RepoMerge(
        onto,
        _without(delta, settled),
        _config_conflicts(change, diff(base, ours)),
        settled.conflicts,
        settled.rebased,
        kernels.united,
        replaced_patches=settled.replaced,
    )


def _config(change: RecipeDiff, ours: LaunchConfig) -> ConfigDelta:
    """The candidate's launch change, onto ``ours``. One that removes anything gives the launch's removals whole:
    ``ours``'s, with the options the candidate removed from the resolved launch (:data:`REMOVED`) and without those it
    stopped removing, so an argument it dropped is never taken for one of them."""

    values = [f"{arg.name} {arg.after}".strip() for arg in change.args if arg.after not in (None, REMOVED)]
    resolved = {arg.name: arg.after == REMOVED for arg in change.args if REMOVED in (arg.before, arg.after)}
    dropped = tuple(arg.name for arg in change.args if arg.after in (None, REMOVED) and arg.before != REMOVED)
    added = {name for name, removes in resolved.items() if removes}
    removals = (set(ours.remove_args) | added) - (set(resolved) - added)
    return ConfigDelta(
        server_args=" ".join(values),
        remove_args=dropped,
        envs={env.name: env.after for env in change.envs if env.after is not None},
        unset_envs=tuple(env.name for env in change.envs if env.after is None),
        launch_remove_args=tuple(sorted(removals)) if dropped or resolved else None,
    )


def _config_conflicts(change: RecipeDiff, onto_change: RecipeDiff) -> tuple[ValueChange, ...]:
    ours = {setting.name: setting.after for setting in (*onto_change.args, *onto_change.envs)}
    return tuple(
        ValueChange(setting.name, ours[setting.name], setting.after)
        for setting in (*change.args, *change.envs)
        if setting.name in ours and ours[setting.name] != setting.after
    )


def _components(base: Recipe, candidate: Recipe, ours: Recipe) -> tuple[ComponentChange, ...]:
    names = sorted({component.component for component in (*base.components, *candidate.components)})
    changes = (_component_change(name, base, candidate, ours) for name in names)
    return tuple(change for change in changes if change is not None)


def _component_change(name: str, base: Recipe, candidate: Recipe, ours: Recipe) -> ComponentChange | None:
    """The patches the candidate added to ``name`` or dropped from it, with its upstream pin where they create a
    component ``ours`` lacks."""
    before, after, current = base.component(name), candidate.component(name), ours.component(name)
    old, new = (before.patches if before else ()), (after.patches if after else ())
    have = current.patches if current else ()
    added = tuple(patch for patch in new if patch not in old and patch not in have)
    removed = tuple(patch.path for patch in old if patch not in new and patch in have)
    if not (added or removed):
        return None
    if current is None and after is not None:
        return ComponentChange(name, after.upstream_ref, after.upstream_repo, added)
    return ComponentChange(name, add_patches=added, remove_patches=removed)


def _added_kernels(base: Recipe, candidate: Recipe, ours: Recipe, stored: Stored) -> _Kernels:
    """Each artifact the candidate added, in the slot it fills: it lands where no other side moved the slot, the wider
    of two tables when one holds every row of the other, and a union of both tables' rows otherwise. The head's table
    on the variable the artifact loads through fills its slot too, whatever target it serves: the service reads one
    table per variable, so the two are decided the same way and the head's is ``displaced`` when it loses."""
    carried, united, displaced = [], [], []
    for artifact in (artifact for artifact in candidate.kernels if artifact not in base.kernels):
        current = kernel_occupant(ours, artifact)
        if current is None or current in (kernel_occupant(base, artifact), artifact):
            carried.append(artifact)
            continue
        wider = wider_table(artifact, current, stored.kernel_table)
        if wider is None:
            union = _united(artifact, current, stored, kernel_occupant(base, artifact))
            united.append(union)
            wider = union.table
        if wider.slot != current.slot:
            displaced.append(current.ref)
        carried.append(wider)
    return _Kernels(
        tuple(artifact for artifact in carried if artifact not in ours.kernels),
        tuple(union for union in united if union.table not in ours.kernels or union.kept_head_shapes),
        tuple(displaced),
    )


def _removed_kernels(base: Recipe, candidate: Recipe, ours: Recipe) -> tuple[str, ...]:
    kept = {artifact.slot for artifact in candidate.kernels}
    dropped = [artifact for artifact in base.kernels if artifact.slot not in kept]
    for artifact in dropped:
        current = _in_slot(ours, artifact.slot)
        if current is not None and current != artifact:
            raise _both_moved(artifact.slot)
    return tuple(artifact.ref for artifact in dropped if artifact in ours.kernels)


def _in_slot(recipe: Recipe, slot: tuple[KernelKind, str]) -> KernelArtifact | None:
    return next((artifact for artifact in recipe.kernels if artifact.slot == slot), None)


def kernel_occupant(recipe: Recipe, artifact: KernelArtifact) -> KernelArtifact | None:
    """The artifact of ``recipe`` that ``artifact`` would replace or meet: the one in its slot, else the one loading
    through the same variable, which no launch can set twice."""

    in_slot = _in_slot(recipe, artifact.slot)
    env = _load_env(artifact)
    if in_slot is not None or env is None:
        return in_slot
    return next((current for current in recipe.kernels if _load_env(current) == env), None)


def _load_env(artifact: KernelArtifact) -> str | None:
    try:
        return kernel_load_env(artifact)
    except KernelLoadError:
        return None


def wider_table(wanted: KernelArtifact, current: KernelArtifact, table: TableBytes) -> KernelArtifact | None:
    """Whichever of two artifacts filling one slot serves every shape the other does: the one whose table holds every
    row of the other's under the same header, or ``None`` when neither does and only a union of their rows would serve
    both. The narrower table is dropped whole rather than merged into the wider, so the slot keeps a table some branch
    tuned and measured as it stands.

    An artifact put as rows over the image's default table is an override of the shapes it names, not a table of the
    shapes it serves, so it is never the narrower of a pair and never compared.
    """

    if kernel_rows_only(wanted) or kernel_rows_only(current):
        return None
    theirs, mine = table(wanted), table(current)
    if _holds(theirs, mine):
        return wanted
    if _holds(mine, theirs):
        return current
    return None


def stack_kernel_slot(wanted: KernelArtifact, current: KernelArtifact, stored: Stored) -> KernelArtifact | None:
    """The artifact a replay of ``wanted`` puts in the slot ``current`` fills, decided as a merge decides it: the wider
    of the two tables, else the union of their rows with the faster row for a shape both tuned, and ``current``
    itself where that union is its rows; ``None`` where a merge would refuse the pair, so the replay swaps ``current``
    out and says so. A replay has no base table, so a shape both set to other rows keeps ``current``'s where either
    row has no measured time."""

    wider = wider_table(wanted, current, stored.kernel_table)
    if wider is not None:
        return wider
    try:
        return _united(wanted, current, stored).table
    except RecipeError:
        return None


def _united(
    theirs: KernelArtifact, ours: KernelArtifact, stored: Stored, base: KernelArtifact | None = None
) -> UnitedTable:
    """The rows of both tables as one table of their slot, kept in the blob store; ``ours`` itself when the union is
    its rows, since a reordered copy would change the recipe and cost a merge test for no change. A shape both set to
    other rows is decided against ``base``, the table the merge base had in the slot (``_union_three_way``).
    Rows put over the image's default table read the same over a whole table, so a rows-only set and a whole table
    make a whole table. Refused when the two do not load through one variable or are not one kind of table."""

    try:
        env, other = kernel_load_env(theirs), kernel_load_env(ours)
        rows_only = kernel_rows_only(theirs) and kernel_rows_only(ours)
    except KernelLoadError as error:
        raise _both_moved(theirs.slot, str(error)) from error
    if env != other:
        raise _both_moved(theirs.slot, f"one table loads through {env}, the other through {other}")
    tables = stored.kernel_table(theirs), stored.kernel_table(ours)
    before = None if base is None else stored.kernel_table(base)
    try:
        unite = partial(union_rows, env=env, rows_only=(rows_only, rows_only), configs=stored.aiter_configs)
        data, kept = _union_three_way(*tables, before, unite)
    except KernelRowsError as error:
        raise _both_moved(theirs.slot, str(error)) from error
    if rows_only == kernel_rows_only(ours) and _rows(data) == _rows(tables[1]):
        return UnitedTable(theirs, ours, ours, kept)
    stored.store_table(data)
    return UnitedTable(theirs, ours, mint_kernel_artifact(ours.kind, ours.target, env, data, rows_only=rows_only), kept)


def _union_three_way(
    theirs: bytes, ours: bytes, base: bytes | None, unite: Callable[[bytes, bytes], bytes]
) -> tuple[bytes, tuple[Mapping[str, str], ...]]:
    """``unite`` (``union_rows``) of both tables, each shape both set to other rows left out of the side that loses it
    first, and the shapes whose head's row won though the candidate also moved it from ``base``.

    The row with the lower measured time wins (:func:`_faster`). Where a row has none, the candidate's row wins where
    the head's is still ``base``'s: the head never retuned that shape, so its row is stale beside the candidate's
    measured one. Elsewhere the head's row wins, measured on the head already. Without a base table every such shape
    keeps the head's row.
    """
    try:
        return unite(theirs, ours), ()
    except KernelRowsConflict as conflict:
        shapes = conflict.shapes
    theirs_win, ours_win, kept = [], [], []
    for shape in shapes:
        candidate, head = _row(theirs, shape), _row(ours, shape)
        before = None if base is None else _row(base, shape)
        if _theirs_wins(candidate, head, before):
            theirs_win.append(shape)
            continue
        ours_win.append(shape)
        if candidate != before:
            kept.append(shape)
    return unite(_without_shapes(theirs, ours_win), _without_shapes(ours, theirs_win)), tuple(kept)


_Row = frozenset[tuple[str, str]]
#: The column the aiter tuner writes a row's measured time to, in microseconds.
_TIME_COLUMN = "us"


def _theirs_wins(theirs: _Row | None, ours: _Row | None, base: _Row | None) -> bool:
    faster = _faster(theirs, ours)
    return ours == base if faster is None else faster


def _faster(theirs: _Row | None, ours: _Row | None) -> bool | None:
    """Whether ``theirs`` has the lower measured time; ``None`` when they tie or either has none. A time of 0 is none:
    a row the tuner wrote without measuring it has 0 for its time, flops and bandwidth."""

    times = [_time(row) for row in (theirs, ours)]
    if None in times or times[0] == times[1]:
        return None
    return times[0] < times[1]


def _time(row: _Row | None) -> float | None:
    value = None if row is None else dict(row).get(_TIME_COLUMN)
    if value is None:
        return None
    try:
        time = float(value)
    except ValueError:
        return None
    return time if math.isfinite(time) and time > 0 else None


def _records(table: bytes) -> tuple[list[str], list[list[str]]]:
    header, *rows = [record for record in csv.reader(io.StringIO(table.decode(), newline="")) if record]
    return header, rows


def _rows(table: bytes) -> frozenset[frozenset[tuple[str, str]]]:
    """``table``'s rows as aiter reads them: by column name, cells compared as ``cell`` compares them."""
    header, rows = _records(table)
    return frozenset(frozenset(zip(header, map(cell, row), strict=False)) for row in rows)


def _row(table: bytes, shape: Mapping[str, str]) -> frozenset[tuple[str, str]] | None:
    """``table``'s row of ``shape`` as ``_rows`` reads it, ``None`` where it has none."""
    return next((row for row in _rows(table) if shape.items() <= row), None)


def _without_shapes(table: bytes, shapes: Sequence[Mapping[str, str]]) -> bytes:
    """``table`` without its rows of ``shapes``, whose key cells are compared as aiter reads them."""
    if not shapes:
        return table
    header, rows = _records(table)
    place = {name: index for index, name in enumerate(header)}

    def named(row: Sequence[str]) -> bool:
        return any(all(cell(row[place[name]]) == value for name, value in shape.items()) for shape in shapes)

    out = io.StringIO()
    csv.writer(out, lineterminator="\r\n" if b"\r\n" in table else "\n").writerows(
        [header, *(row for row in rows if not named(row))]
    )
    return out.getvalue().encode()


def _holds(wider: bytes, narrower: bytes) -> bool:
    """Whether ``wider`` is a table under the same header as ``narrower`` with every row of it."""

    header, rows = _table(wider)
    other_header, other_rows = _table(narrower)
    return bool(header) and header == other_header and rows >= other_rows


def _table(data: bytes) -> tuple[bytes, frozenset[bytes]]:
    """A table's header and the set of its rows; the header is empty for bytes that are no table at all."""

    header, *body = data.splitlines() or [b""]
    return header.strip(), frozenset(line for line in body if line.strip())


def _both_moved(slot: tuple[KernelKind, str], reason: str = "") -> RecipeError:
    kind, target = slot
    return RecipeError(f"both sides replaced the {kind.value} kernel for {target}" + (f": {reason}" if reason else ""))


def _settle(
    component: ComponentPatches | None, theirs: ComponentPatches | None, brought: Collection[str], stored: Stored
) -> _Settled:
    """Each patch of ``component`` that does not apply after the ones before it that do: rebased where the candidate
    ``brought`` it and no line overlaps, put in for the head patches it replaces where it supersedes them, held where
    the stack before it already makes its change, split where only some of its files overlap, a conflict otherwise."""
    if component is None or not component.patches or _apply_error(component, stored.stack_files) is None:
        return _Settled()
    heads = frozenset(
        patch
        for patch in component.patches
        if patch.path not in brought
        and patch.path not in stored.protected
        and (theirs is None or patch not in theirs.patches)
    )
    settling = _Settling(component, theirs, frozenset(brought), heads, stored)
    for patch in component.patches:
        settling.add(patch)
    return settling.settled()


@dataclass
class _Settling:
    """``_settle`` of one component's stack under way: ``applied`` are the patches that apply so far, in order."""

    component: ComponentPatches
    theirs: ComponentPatches | None
    brought: frozenset[str]
    heads: frozenset[PatchRef]
    stored: Stored
    applied: list[PatchRef] = field(default_factory=list)
    conflicts: list[PatchConflict] = field(default_factory=list)
    rebased: list[RebasedPatch] = field(default_factory=list)
    replaced: list[ReplacedPatch] = field(default_factory=list)
    held: set[str] = field(default_factory=set)
    parts: set[str] = field(default_factory=set)

    def add(self, patch: PatchRef) -> None:
        name = self.component.component
        stack = replace(self.component, patches=tuple(self.applied))
        error = self._error(patch)
        if error is None:
            self.applied.append(patch)
            return
        theirs = self.theirs
        sides = _sides(stack, theirs, patch, self.stored) if theirs is not None and patch.path in self.brought else None
        text = None if sides is None else _rebased_text(stack, sides)
        if sides is not None and text is None:
            self._overlapping(stack, sides, patch, error)
        elif text is None:
            self.conflicts.append(PatchConflict(name, patch.path, error))
        elif text:
            self.applied.append(self.stored.store_patch(name, text))
            self.rebased.append(RebasedPatch(name, patch.path, self.applied[-1]))
        else:
            self.held.add(patch.path)

    def _overlapping(self, stack: ComponentPatches, sides: _Sides, patch: PatchRef, error: str) -> None:
        """A candidate's ``patch`` some line of which the stack also changed: put in for the head patches it replaces,
        else split so only the files that overlap stay a conflict, else a conflict whole."""
        name = self.component.component
        kept = _replacement(stack, sides, patch, self.heads, self.stored)
        if kept is not None:
            removed = tuple(head.path for head in self.applied if head not in kept)
            self.applied = [*kept, patch]
            self.replaced.append(ReplacedPatch(name, removed, patch))
            return
        split = _split(name, sides, self.stored)
        if split is None:
            self.conflicts.append(PatchConflict(name, patch.path, error))
            return
        if split.carried is None:
            self.held.add(patch.path)
        else:
            self.applied.append(split.carried)
            self.rebased.append(RebasedPatch(name, patch.path, split.carried))
        self.conflicts.append(PatchConflict(name, split.rest.path, self._error(split.rest) or error))
        self.parts.add(split.rest.path)

    def _error(self, patch: PatchRef) -> str | None:
        return _apply_error(replace(self.component, patches=(*self.applied, patch)), self.stored.stack_files)

    def settled(self) -> _Settled:
        return _Settled(
            tuple(self.conflicts),
            tuple(self.rebased),
            frozenset(self.held),
            tuple(self.replaced),
            frozenset(self.parts),
        )


def _sides(stack: ComponentPatches, theirs: ComponentPatches, patch: PatchRef, stored: Stored) -> _Sides | None:
    """The files ``patch`` of the candidate's stack ``theirs`` touches, before and after it there and under ``stack``;
    ``None`` when a stack does not apply."""
    index = theirs.patches.index(patch)
    try:
        changed = stored.stack_files(replace(theirs, patches=theirs.patches[: index + 1]), ())
        original = stored.stack_files(replace(theirs, patches=theirs.patches[:index]), changed)
        current = stored.stack_files(stack, changed)
    except PatchApplyError:
        return None
    return _Sides(original, changed, current)


def _rebased_text(stack: ComponentPatches, sides: _Sides) -> str | None:
    """The patch's change merged three ways into the files ``stack`` leaves, as a patch over ``stack``: empty when
    ``stack`` already makes the change, ``None`` when a line is changed by both."""
    merged = merge_files(sides.original, sides.current, sides.changed)
    return None if merged is None else patch_between(sides.current, merged)


@dataclass(frozen=True)
class _Split:
    """A candidate's patch split by file: ``carried`` is the change to the files that merge three ways, rebased onto the
    stack (``None`` where the stack already makes it), and ``rest`` the change to the others as the candidate wrote
    it."""

    carried: PatchRef | None
    rest: PatchRef


def _split(component: str, sides: _Sides, stored: Stored) -> _Split | None:
    """A patch some of whose files overlap the stack, split by file so only those files go to the resolver; ``None``
    when every file the patch changes overlaps.

    A patch is resolved by one model call that rewrites all of it, so a large one is cut off by the gateway or by the
    output cap even when only a few of its files conflict (c7eddf22's 3b275cb2: 9 files, 3 conflicting, 35.5 KB).
    """
    touched = [path for path, content in sides.changed.items() if content != sides.original[path]]
    merged, overlapping = dict(sides.current), []
    for path in touched:
        files = merge_files(sides.original, sides.current, {path: sides.changed[path]})
        if files is None:
            overlapping.append(path)
        else:
            merged[path] = files[path]
    if len(overlapping) == len(touched):
        return None
    text = patch_between(sides.current, merged)
    rest = patch_between(
        {path: sides.original[path] for path in overlapping}, {path: sides.changed[path] for path in overlapping}
    )
    return _Split(stored.store_patch(component, text) if text else None, stored.store_patch(component, rest))


def _replacement(
    stack: ComponentPatches, sides: _Sides, patch: PatchRef, heads: Collection[PatchRef], stored: Stored
) -> tuple[PatchRef, ...] | None:
    """What is left of ``stack`` once ``patch`` replaces one of its ``heads`` (head patches the candidate's stack does
    not hold and no foundation fix contributed), ``None`` when it replaces none.

    Two branches that wrote the same lines independently cannot be merged by text, but one version can be picked by
    measurement: the head patch X is taken out with every later patch that applies only on top of it, and ``patch``
    goes in as it is. That is a replacement only when nothing of X survives on its own: every file must equal, byte for
    byte, the three-way merge that takes the candidate's side of every overlap. Otherwise both changes are needed
    together, and the patch stays a conflict for the integrator's resolver. A dependent the candidate brought or a
    foundation fix contributed is never taken out.
    """
    wanted = merge_files(sides.original, sides.current, sides.changed, theirs_wins=True)
    if wanted is None:
        return None
    for head in (head for head in reversed(stack.patches) if head in heads):
        kept = _without_dependents(stack, head, stored)
        if any(gone not in heads for gone in stack.patches if gone not in kept):
            continue
        try:
            files = stored.stack_files(replace(stack, patches=(*kept, patch)), wanted)
        except PatchApplyError:
            continue
        if all(files.get(path) == content for path, content in wanted.items()):
            return kept
    return None


def _without_dependents(stack: ComponentPatches, head: PatchRef, stored: Stored) -> tuple[PatchRef, ...]:
    """``stack`` without ``head`` and without each later patch that no longer applies once it is gone."""
    index = stack.patches.index(head)
    kept = list(stack.patches[:index])
    for later in stack.patches[index + 1 :]:
        if _apply_error(replace(stack, patches=(*kept, later)), stored.stack_files) is None:
            kept.append(later)
    return tuple(kept)


def _apply_error(component: ComponentPatches, stack_files: StackFiles) -> str | None:
    try:
        stack_files(component, ())
    except PatchApplyError as error:
        return str(error)
    return None


def _joined(settled: Iterable[_Settled]) -> _Settled:
    parts = list(settled)
    return _Settled(
        tuple(conflict for part in parts for conflict in part.conflicts),
        tuple(rebased for part in parts for rebased in part.rebased),
        frozenset().union(*(part.held for part in parts)),
        tuple(replaced for part in parts for replaced in part.replaced),
        frozenset().union(*(part.parts for part in parts)),
    )


def _without(delta: RecipeDelta, settled: _Settled) -> RecipeDelta:
    """``delta`` with every conflicting patch left out, not added when the candidate brings it and dropped otherwise;
    a rebased patch in place of the candidate's, none for one the stack already holds, and the head patches a
    candidate's patch replaces dropped. A conflict split off a candidate's patch is in no stack to drop."""
    if not (settled.conflicts or settled.rebased or settled.held or settled.replaced):
        return delta
    changes = []
    for change in delta.components:
        failed = {conflict.path for conflict in settled.conflicts if conflict.component == change.component}
        failed |= {path for swap in settled.replaced if swap.component == change.component for path in swap.removed}
        moved = {rebased.path: rebased.patch for rebased in settled.rebased if rebased.component == change.component}
        brought = {patch.path for patch in change.add_patches}
        kept = (patch for patch in change.add_patches if patch.path not in failed | settled.held)
        changes.append(
            replace(
                change,
                add_patches=tuple(moved.get(patch.path, patch) for patch in kept),
                remove_patches=(*change.remove_patches, *sorted(failed - brought - settled.parts)),
            )
        )
    return replace(delta, components=tuple(changes))


__all__ = [
    "PatchStore",
    "StackFiles",
    "Stored",
    "TableBytes",
    "TableStore",
    "carry",
    "kernel_occupant",
    "stack_kernel_slot",
    "wider_table",
]
