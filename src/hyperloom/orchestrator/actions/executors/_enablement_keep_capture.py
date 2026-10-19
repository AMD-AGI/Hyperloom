# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What an accepted enablement round records about the stack it leaves behind.

Every value here is read at the KEEP: the roots the resolvers bound, the
pre-mutation ``base_sha`` captured before the apply, the byte-exact content of
each declared target, and the versions observed through the interpreter the
accepted bench launched.

The declared set is the whole accepted stack, every earlier round's patches and
artifacts included, and not this round's diff. A KEEP is committed and never
re-applied, so a round that captures only what it wrote snapshots the last
round's files while the recipe still carries a patch step per earlier round -- a
consumer replaying that rebuilds a tree missing every round but the last, and
the decision certified it. Inherited identity wins over this round's reading of
it for the same reason: ``base_sha`` has to keep naming the tree the first
accepted round applied to, because this round's HEAD already contains its
predecessors.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, NamedTuple

from hyperloom.inference_optimizer import framework_paths

from ...enablement.recipe import keep_probe, keep_records
from ...enablement.recipe.projections import project_launch_evidence
from . import _patch_snapshot
from ._integrate_attempt import IntegrateAttempt
from ._nogit_patch import _is_git_tree

log = logging.getLogger(__name__)


class _StackTargets(NamedTuple):
    """The files the accepted stack declares, addressed by the tree and by the patch that declares them."""

    by_root: dict[str, dict[str, str]]
    by_patch: dict[str, dict[str, str]]


class _RootIdentity(NamedTuple):
    """Which trees the accepted stack contributes to, and the commit each one is bound to."""

    contributions: dict[str, Any]
    base_sha_by_root: dict[str, str]
    records: list[dict[str, Any]]


def _bind_roots(attempt: IntegrateAttempt, patch_roots: dict[str, str], artifacts: Any) -> _RootIdentity:
    """Resolve every tree the accepted stack writes to and the base commit its capture is proven against.

    Only what was captured before this round touched each tree, and only where no
    earlier round already named the tree the stack applies to. A root left with no
    sha either way keeps none, which the decision refuses rather than answering
    with a HEAD that has moved since.
    """
    enablement = attempt.shared_state.enablement
    contributions = keep_records.collect_contributions(
        framework_root=str(attempt.framework_root or ""),
        patch_roots=patch_roots,
        artifacts=artifacts,
    )
    git_roots = [root for root in contributions if root and _is_git_tree(Path(root))]
    inherited_sha = enablement.inherited_base_shas()
    captured = attempt.base_sha_by_root
    base_sha_by_root = {root: inherited_sha.get(root, "") or captured.get(root, "") for root in git_roots}
    return _RootIdentity(
        contributions=contributions,
        base_sha_by_root=base_sha_by_root,
        records=keep_records.build_root_records(
            contributions=contributions,
            base_sha_by_root=base_sha_by_root,
            git_roots=git_roots,
            session_framework_root=framework_paths.resolve_session_framework_root(),
        ),
    )


def _patch_stack_by_root(enablement: Any, patch_roots: dict[str, str], applied: list[Path]) -> dict[str, list[Path]]:
    """Group the accepted patch stack by the tree each patch applies to, in apply order.

    Apply order, not dict order: ``patch_roots`` is keyed by patch path and its
    iteration order follows how the mapping was assembled, while the operation a
    later patch declares must override an earlier one's for the same file.

    ``EnablementRound.accepted_patch_roots`` binds only patches that ARE in the
    accepted stack, so the prefix normally adds nothing. It stays because the
    durable mapping outlives the round that wrote it: an entry for a patch no
    longer in ``kept_patches`` would otherwise contribute a root record with no
    declared target, and be refused as an uncaptured root rather than classified.
    First, because a position it does not have cannot override one that is known.
    """
    accepted = enablement.accepted_stack(applied)
    ordered = [patch for patch in patch_roots if patch not in set(accepted)] + accepted
    by_root: dict[str, list[Path]] = {}
    for patch_path in dict.fromkeys(ordered):
        patch_root = str(patch_roots.get(patch_path) or "")
        if patch_root:
            by_root.setdefault(patch_root, []).append(Path(patch_path))
    return by_root


def _declared_by_patch(patch_root: str, patches: list[Path], *, base_sha: str) -> dict[str, dict[str, str]]:
    """The file operations each patch declares against ``patch_root``, per patch.

    Proven, not believed: the whole stack is replayed from its base in an isolated
    tree and the result compared against what is about to be captured. Per root and
    over the ORDERED stack, because a patch's preimage is the tree its predecessors
    left behind, not this root's base and not the final tree.
    """
    replayed = _patch_snapshot.replayed_stack_ops(Path(patch_root), patches, base_sha=base_sha)
    if replayed is not None:
        return replayed
    if base_sha:
        log.warning(
            "integrate_patch: enablement KEEP cannot replay the stack on %s from %s; "
            "its targets are left undeclared and the recipe is refused",
            patch_root,
            base_sha,
        )
        return {}
    # No base commit: a framework installed from a wheel, which is the ordinary
    # production shape. There is no "check out the base and apply" to prove, but
    # that is not how such a root is replayed -- it is replayed by OVERLAYING the
    # captured files, which is exactly what building an image from this recipe
    # does. Declaring nothing here removes those files from the capture and with
    # them the only replay path that applies, leaving the recipe unusable in the
    # case it is most needed for.
    #
    # What is NOT claimed is that the patch was proven applied: there is no
    # preimage to prove it against. Do not read an empty base_sha as the thing
    # that withholds that claim -- the judge's base_sha refusal is gated on
    # ``is_git``, so it never fires for a wheel root and this path can and does
    # reach ``sufficient``. The protection that does apply is the op map itself:
    # ``overlay_inventory_without_base`` returns None for anything it cannot
    # PROVE, and undeclared targets are what the judge blocks on. The targets are
    # what the overlay must contain, not evidence that the patch applied cleanly.
    overlaid: dict[str, dict[str, str]] = {}
    for patch in patches:
        ops = _patch_snapshot.overlay_inventory_without_base(Path(patch_root), patch)
        if ops is None:
            log.warning(
                "integrate_patch: enablement KEEP cannot resolve %s against the "
                "base-less root %s; its targets are left undeclared",
                patch,
                patch_root,
            )
            continue
        overlaid[str(patch)] = ops
    return overlaid


def _stack_targets(
    stack_by_root: dict[str, list[Path]],
    *,
    base_sha_by_root: dict[str, str],
    framework_root: str,
    artifacts: Any,
) -> _StackTargets:
    """Classify the whole accepted stack's files once per root.

    A rel path is only meaningful against the tree its patch was bound to, and
    reading them all against this round's root would name files that root does not
    have. The per-patch half exists because the decision cross-checks each patch
    step against the snapshot of the files that step declares, and it cannot read
    the diffs itself.
    """
    by_root: dict[str, dict[str, str]] = {}
    by_patch: dict[str, dict[str, str]] = {}
    for patch_root, patches in stack_by_root.items():
        declared = _declared_by_patch(patch_root, patches, base_sha=str(base_sha_by_root.get(patch_root) or ""))
        for patch_path, ops in declared.items():
            if not ops:
                continue
            by_patch[patch_path] = ops
            by_root.setdefault(patch_root, {}).update(ops)
    for declared_root, ops in keep_records.declared_targets(
        framework_root=framework_root,
        upserted=(),
        deleted=(),
        artifacts=artifacts,
    ).items():
        by_root.setdefault(declared_root, {}).update(ops)
    return _StackTargets(by_root=by_root, by_patch=by_patch)


def capture_keep_records(
    attempt: IntegrateAttempt,
    *,
    params: dict[str, Any],
    bench_result: dict[str, Any],
    session_dir: Path,
) -> dict[str, Any]:
    """Capture the per-root identity, payload and assertions of this enablement KEEP."""
    root = str(attempt.framework_root or "")
    enablement = attempt.shared_state.enablement
    patch_roots = enablement.accepted_patch_roots(
        done_payload=attempt.done_payload,
        applied=attempt.applied,
        framework_root=root,
    )
    # Read from the durable stack: the round lifecycle does not dispatch the base
    # set with the round.
    stack_artifacts = keep_records.accepted_stack_artifacts(
        inherited=enablement.kept_artifacts,
        applied=attempt.applied_artifacts,
    )
    identity = _bind_roots(attempt, patch_roots, stack_artifacts)
    targets = _stack_targets(
        _patch_stack_by_root(enablement, patch_roots, attempt.applied),
        base_sha_by_root=identity.base_sha_by_root,
        framework_root=root,
        artifacts=stack_artifacts,
    )
    snapshots = keep_records.capture_root_snapshots(
        records=identity.records,
        targets=targets.by_root,
        dest_root=session_dir / "optimization_stack" / "enablement",
        session_dir=session_dir,
    )
    materialized_config = str(bench_result.get("materialized_config") or "")
    framework = keep_probe.graded_framework(params, materialized_config)
    closure, assertions = keep_probe.probe_keep_environment(
        params,
        keep_probe.KeepEvidence(
            build_manifest=enablement.build_manifest,
            specialist_task_id=attempt.specialist_task_id,
            provision_result=attempt.provision_result,
        ),
        framework=framework,
        materialized_config=materialized_config,
    )
    launch_evidence, argv_refused = project_launch_evidence(bench_result.get("launch_evidence"))
    return {
        "enablement_roots": identity.records,
        "enablement_patch_roots": patch_roots,
        # Per root, and through the same inherited-then-captured resolution every
        # other root gets. Preferring the persisted scalar outright hands this
        # round's root the sha an earlier round read off a *different* tree, and
        # the snapshot is then captured against a base commit that tree never had.
        "enablement_base_sha": str(identity.base_sha_by_root.get(root) or ""),
        "enablement_source_snapshots": snapshots,
        "enablement_patch_targets": targets.by_patch,
        "enablement_accepted_stack_targets": {
            str(record["id"]): dict(targets.by_root.get(str(record["path"])) or {}) for record in identity.records
        },
        "enablement_launch_evidence": launch_evidence or {},
        "enablement_launch_argv_refused": argv_refused,
        "enablement_environment_closure": closure,
        "enablement_installed_versions_at_keep": assertions,
        "enablement_build_extensions_not_carried": keep_records.build_extensions_not_carried(
            enablement,
            attempt.framework_root,
            specialist_task_id=attempt.specialist_task_id,
        ),
        "enablement_levers_without_readers": keep_records.levers_without_readers(
            enablement,
            attempt.framework_root,
            framework=framework,
            effective_config=bench_result.get("effective_config"),
        ),
    }
