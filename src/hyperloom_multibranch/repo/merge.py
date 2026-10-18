# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A candidate commit's own change carried onto another commit's recipe, for the integrator's merge.

The change is what the candidate's recipe changed from its parent's. A server argument or environment variable both
sides changed takes the candidate's value and is listed; a component pin or a kernel slot both sides moved has no
value to prefer, so the merge is refused. The merged patch stacks are applied to the installed package as the toolbox
applies them when it builds a test's environment, so a conflict shows before the merge test is paid for.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace

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
from hyperloom.measure.compile import PatchApplyError
from hyperloom_multibranch.contracts.repo import PatchConflict, RepoMerge

#: Applies a component's patch stack to the installed package; raises ``PatchApplyError`` when one does not apply.
PatchCheck = Callable[[ComponentPatches], None]


def carry(onto: str, base: Recipe, candidate: Recipe, ours: Recipe, check: PatchCheck) -> RepoMerge:
    """``candidate``'s change from ``base``, its parent's recipe, carried onto ``ours``, the recipe of ``onto``."""
    change = diff(base, candidate)
    if change.base_image_changed:
        raise RecipeError("the candidate changes the base image, which a session keeps")
    delta = RecipeDelta(
        config=_config(change),
        components=_components(base, candidate, ours),
        add_kernels=_added_kernels(base, candidate, ours),
        remove_kernels=_removed_kernels(base, candidate, ours),
    )
    merged = apply(ours, delta)
    conflicts = [
        conflict
        for touched in delta.components
        for conflict in _stack_conflicts(merged.component(touched.component), check)
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
    before, after, current = base.component(name), candidate.component(name), ours.component(name)
    old, new = (before.patches if before else ()), (after.patches if after else ())
    have = current.patches if current else ()
    added = tuple(patch for patch in new if patch not in old and patch not in have)
    removed = tuple(patch.path for patch in old if patch not in new and patch in have)
    pin = _pin(name, before, after, current)
    if not (added or removed or pin):
        return None
    if current is None and after is not None:
        return ComponentChange(name, after.upstream_ref, after.upstream_repo, added)
    return ComponentChange(name, pin, add_patches=added, remove_patches=removed)


def _pin(
    name: str, before: ComponentPatches | None, after: ComponentPatches | None, current: ComponentPatches | None
) -> str | None:
    """The upstream commit the candidate moved ``name`` to, if it moved it."""
    was = before.upstream_ref if before else None
    if after is None or after.upstream_ref == was:
        return None
    if current is not None and current.upstream_ref not in (was, after.upstream_ref):
        raise RecipeError(
            f"both sides moved {name}: onto to {current.upstream_ref}, the candidate to {after.upstream_ref}"
        )
    return after.upstream_ref


def _added_kernels(base: Recipe, candidate: Recipe, ours: Recipe) -> tuple[KernelArtifact, ...]:
    added = [artifact for artifact in candidate.kernels if artifact not in base.kernels]
    for artifact in added:
        _refuse_if_both_moved(artifact.slot, _in_slot(base, artifact.slot), _in_slot(ours, artifact.slot), artifact)
    return tuple(artifact for artifact in added if artifact not in ours.kernels)


def _removed_kernels(base: Recipe, candidate: Recipe, ours: Recipe) -> tuple[str, ...]:
    kept = {artifact.slot for artifact in candidate.kernels}
    dropped = [artifact for artifact in base.kernels if artifact.slot not in kept]
    for artifact in dropped:
        _refuse_if_both_moved(artifact.slot, artifact, _in_slot(ours, artifact.slot), None)
    return tuple(artifact.ref for artifact in dropped if artifact in ours.kernels)


def _in_slot(recipe: Recipe, slot: tuple[KernelKind, str]) -> KernelArtifact | None:
    return next((artifact for artifact in recipe.kernels if artifact.slot == slot), None)


def _refuse_if_both_moved(
    slot: tuple[KernelKind, str],
    was: KernelArtifact | None,
    current: KernelArtifact | None,
    wanted: KernelArtifact | None,
) -> None:
    if current is not None and current not in (was, wanted):
        kind, target = slot
        raise RecipeError(f"both sides replaced the {kind.value} kernel for {target}")


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


__all__ = ["PatchCheck", "carry"]
