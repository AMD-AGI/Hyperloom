# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A candidate commit's own change carried onto another commit's recipe, for the integrator's merge.

The change is what the candidate's recipe changed from its parent's. A server argument or environment variable both
sides changed takes the candidate's value and is listed; a kernel slot both sides filled is decided by the two tables
when one holds every row of the other, since a target loads one table and the wider one serves every shape the
narrower did, and refused when neither does. A component travels only for the patches it gains or loses, with its
upstream commit where
those patches create a component the other side lacks: every component is built from the installed package of the
session's fixed base image, so a recipe's upstream pin is provenance no merge can act on, and a component the
candidate records without patching it changes nothing a test would build. The merged patch stacks are applied to the
installed package as the toolbox applies them when it builds a test's environment, so a conflict shows before the
merge test is paid for.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from hyperloom.contracts.recipe import (
    ComponentChange,
    ComponentPatches,
    ConfigDelta,
    KernelArtifact,
    KernelKind,
    PatchRef,
    Recipe,
    RecipeDelta,
    RecipeDiff,
    RecipeError,
    ValueChange,
    apply,
    diff,
)
from hyperloom.measure.compile import PatchApplyError, kernel_rows_only
from hyperloom_multibranch.contracts.repo import PatchConflict, RepoMerge

#: Applies a component's patch stack to the installed package; raises ``PatchApplyError`` when one does not apply.
PatchCheck = Callable[[ComponentPatches], None]
#: Reads a kernel table's bytes, for the two that fill one slot.
TableBytes = Callable[[KernelArtifact], bytes]


@dataclass(frozen=True)
class Stored:
    """What a merge consults beyond the three recipes: the installed packages a patch stack has to apply to, and the
    bytes behind a kernel artifact."""

    apply_patches: PatchCheck
    kernel_table: TableBytes


def carry(onto: str, base: Recipe, candidate: Recipe, ours: Recipe, stored: Stored) -> RepoMerge:
    """``candidate``'s change from ``base``, its parent's recipe, carried onto ``ours``, the recipe of ``onto``."""
    change = diff(base, candidate)
    if change.base_image_changed:
        raise RecipeError("the candidate changes the base image, which a session keeps")
    delta = RecipeDelta(
        config=_config(change),
        components=_components(base, candidate, ours),
        add_kernels=_added_kernels(base, candidate, ours, stored.kernel_table),
        remove_kernels=_removed_kernels(base, candidate, ours),
    )
    merged = apply(ours, delta)
    conflicts = [
        conflict
        for touched in delta.components
        for conflict in _stack_conflicts(merged.component(touched.component), stored.apply_patches)
    ]
    return RepoMerge(onto, _without(delta, conflicts), _config_conflicts(change, diff(base, ours)), tuple(conflicts))


def _config(change: RecipeDiff) -> ConfigDelta:
    return ConfigDelta(
        server_args=" ".join(f"{arg.name} {arg.after}".strip() for arg in change.args if arg.after is not None),
        remove_args=tuple(arg.name for arg in change.args if arg.after is None),
        envs={env.name: env.after for env in change.envs if env.after is not None},
        unset_envs=tuple(env.name for env in change.envs if env.after is None),
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


def _added_kernels(base: Recipe, candidate: Recipe, ours: Recipe, table: TableBytes) -> tuple[KernelArtifact, ...]:
    added = [artifact for artifact in candidate.kernels if artifact not in base.kernels]
    carried = [
        artifact
        for artifact in added
        if _takes_slot(artifact, _in_slot(base, artifact.slot), _in_slot(ours, artifact.slot), table)
    ]
    return tuple(artifact for artifact in carried if artifact not in ours.kernels)


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


def _takes_slot(
    wanted: KernelArtifact, was: KernelArtifact | None, current: KernelArtifact | None, table: TableBytes
) -> bool:
    """Whether ``wanted`` lands in the slot ``current`` holds: it does when no other side moved the slot, and when its
    table is the wider of the two. Two tables neither of which holds the other are refused, as are unlike headers.
    """

    if current is None or current in (was, wanted):
        return True
    wider = wider_table(wanted, current, table)
    if wider is None:
        raise _both_moved(wanted.slot)
    return wider == wanted


def _holds(wider: bytes, narrower: bytes) -> bool:
    """Whether ``wider`` is a table under the same header as ``narrower`` with every row of it."""

    header, rows = _table(wider)
    other_header, other_rows = _table(narrower)
    return bool(header) and header == other_header and rows >= other_rows


def _table(data: bytes) -> tuple[bytes, frozenset[bytes]]:
    """A table's header and the set of its rows; the header is empty for bytes that are no table at all."""

    header, *body = data.splitlines() or [b""]
    return header.strip(), frozenset(line for line in body if line.strip())


def _both_moved(slot: tuple[KernelKind, str]) -> RecipeError:
    kind, target = slot
    return RecipeError(f"both sides replaced the {kind.value} kernel for {target}")


def _stack_conflicts(component: ComponentPatches | None, check: PatchCheck) -> list[PatchConflict]:
    """The patches of ``component`` that do not apply after the ones before them that do."""
    if component is None or not component.patches or _applies(component, check):
        return []
    applied: list[PatchRef] = []
    conflicts = []
    for patch in component.patches:
        try:
            check(replace(component, patches=(*applied, patch)))
        except PatchApplyError as error:
            conflicts.append(PatchConflict(component.component, patch.path, str(error)))
            continue
        applied.append(patch)
    return conflicts


def _applies(component: ComponentPatches, check: PatchCheck) -> bool:
    try:
        check(component)
    except PatchApplyError:
        return False
    return True


def _without(delta: RecipeDelta, conflicts: Sequence[PatchConflict]) -> RecipeDelta:
    """``delta`` with every conflicting patch left out: not added when the candidate brings it, dropped otherwise."""
    if not conflicts:
        return delta
    changes = []
    for change in delta.components:
        failed = {conflict.path for conflict in conflicts if conflict.component == change.component}
        brought = {patch.path for patch in change.add_patches}
        changes.append(
            replace(
                change,
                add_patches=tuple(patch for patch in change.add_patches if patch.path not in failed),
                remove_patches=(*change.remove_patches, *sorted(failed - brought)),
            )
        )
    return replace(delta, components=tuple(changes))


__all__ = ["PatchCheck", "Stored", "TableBytes", "carry", "wider_table"]
