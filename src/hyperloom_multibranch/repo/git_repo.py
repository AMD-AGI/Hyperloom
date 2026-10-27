# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The experiment repository as a bare git repository, one per session; every commit holds one recipe document.

The patch and kernel files a recipe names are kept in the session's blob store under their sha256, where the toolbox
reads them when it builds a test's environment.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from hyperloom.contracts.recipe import (
    PATCH_ROOT,
    ComponentPatches,
    KernelArtifact,
    PatchRef,
    Recipe,
    RecipeDelta,
    RecipeError,
    apply,
    with_unexpressed,
)
from hyperloom.measure.compile import (
    COMPONENT_LAYOUTS,
    BlobError,
    BlobStore,
    ComponentLayout,
    PackageLocator,
    PatchApplyError,
    UnsupportedComponentError,
    locate_installed_package,
)
from hyperloom_multibranch.contracts.notebook.vocab import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.repo import BranchMoved, BundleMerge, RepoMerge
from hyperloom_multibranch.contracts.sources import NativeTree, SourceRoot, source_roots
from hyperloom_multibranch.repo.merge import Stored, carry, stack_kernel_slot
from hyperloom_multibranch.repo.rebase import Files
from hyperloom_multibranch.repo.stack import ScratchBlobs, stack_files

RECIPE_FILE = "recipe.json"
_UPDATE_REF = "update-ref"
_NO_COMMIT = "0" * 40
# A re-rooted branch's old head stays under this prefix: its commits are still merge candidates, which a repack must
# keep.
_REROOTED = "refs/rerooted"
# The notebook knows a state only by its full commit id; git would also resolve a branch name or a short id.
_COMMIT_ID = re.compile(r"[0-9a-f]{40}")
# The path ``_put_patch`` gives a patch it keeps, which names the blob it is kept in.
_STORED_PATCH = re.compile(rf"{re.escape(PATCH_ROOT)}/[^/]+/(?P<sha256>[0-9a-f]{{64}})\.patch")
# Plumbing on a local bare repository takes milliseconds; this only bounds a wedged filesystem.
GIT_TIMEOUT_S = 60.0
# Session commits carry this identity whoever runs the session, so the operator's git config never decides it.
_IDENTITY = {
    "GIT_AUTHOR_NAME": "hyperloom-multibranch",
    "GIT_AUTHOR_EMAIL": "multibranch@hyperloom.invalid",
    "GIT_COMMITTER_NAME": "hyperloom-multibranch",
    "GIT_COMMITTER_EMAIL": "multibranch@hyperloom.invalid",
}


class GitError(RuntimeError):
    """A git command failed for a reason the repository contract does not name."""


@dataclass(frozen=True)
class _Completed:
    code: int | None
    out: str
    err: str


def _ref(branch: str) -> str:
    return f"refs/heads/{branch}"


def _patch_paths(recipe: Recipe) -> set[str]:
    return {patch.path for component in recipe.components for patch in component.patches}


class GitExperimentRepo:
    """Writes commits with plumbing and moves a branch by compare-and-swap on its ref.

    No working tree or index is involved, so writers on different branches never contend, and two writers racing on one
    branch leave exactly one commit on it. ``aiter_configs`` is the aiter install a merge that unites two kernel tables
    reads their shape key from, the image's when ``None``. ``native`` is each component's native tree, where the
    host builds native code: a patch to it applies there.
    """

    def __init__(
        self,
        path: Path,
        blobs: BlobStore,
        locator: PackageLocator = locate_installed_package,
        aiter_configs: Path | None = None,
        native: Mapping[str, NativeTree] | None = None,
    ) -> None:
        self._path = path.resolve()
        self._blobs = blobs
        self._locator = locator
        self._aiter_configs = aiter_configs
        self._native = dict(native or {})
        # An inherited GIT_* (GIT_DIR inside a hook or `git rebase --exec`) would aim every command at another
        # repository, so none reaches git and GIT_DIR names this one.
        inherited = {key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")}
        self._env = {**inherited, **_IDENTITY, "GIT_DIR": str(self._path)}

    async def init_root(self, recipe: Recipe) -> str:
        await asyncio.to_thread(self._path.mkdir, parents=True)
        await self._git("init", "--bare", "--quiet", f"--initial-branch={INTEGRATION_BRANCH}")
        root = await self._write(recipe, (), "baseline")
        await self._git(_UPDATE_REF, _ref(INTEGRATION_BRANCH), root, _NO_COMMIT)
        return root

    async def read_recipe(self, commit: str) -> Recipe:
        resolved = await self._commit_of(commit)
        return Recipe.from_document(json.loads(await self._git("show", f"{resolved}:{RECIPE_FILE}")))

    async def create_branch(self, branch: str, at_commit: str) -> None:
        resolved = await self._commit_of(at_commit)
        created = await self._run(_UPDATE_REF, _ref(branch), resolved, _NO_COMMIT)
        if created.code == 0:
            return
        if await self._resolve(_ref(branch)) is not None:
            raise ValueError(f"branch {branch!r} already exists")
        raise GitError(f"cannot create branch {branch!r}: {created.err}")

    async def head(self, branch: str) -> str:
        found = await self._resolve(_ref(branch))
        if found is None:
            raise KeyError(branch)
        return found

    async def commit(self, branch: str, parent: str, recipe: Recipe, message: str) -> str:
        return await self._land(branch, parent, recipe, (parent,), message)

    async def merge(self, candidate: str, onto: str, protected: frozenset[str] = frozenset()) -> RepoMerge:
        resolved, target = await self._commit_of(candidate), await self._commit_of(onto)
        base = await self._merge_base(resolved, target)
        if base == resolved:
            return RepoMerge(onto, RecipeDelta())
        recipes = [await self.read_recipe(commit) for commit in (base, resolved, target)]
        stored = replace(self._stored(), protected=await self._contributed(target, protected))
        merge = await asyncio.to_thread(carry, onto, *recipes, stored)
        carried = await self._git("rev-list", "--topo-order", "--reverse", resolved, f"^{target}")
        return replace(merge, carried=tuple(carried.split()))

    async def merge_many(
        self, candidates: Sequence[str], onto: str, protected: frozenset[str] = frozenset()
    ) -> BundleMerge:
        target = await self._commit_of(onto)
        head = await self.read_recipe(target)
        stored = replace(self._stored(), protected=await self._contributed(target, protected))
        bundle, steps, members, left_out = head, [], [], []
        for candidate in candidates:
            step = await self._carried_onto(candidate, target, bundle, stored)
            if step is None:
                left_out.append(candidate)
                continue
            bundle = apply(bundle, step.delta)
            steps.append(step)
            members.append(candidate)
            # A later member may not take out an earlier one's patch, as no merge takes out a foundation fix's.
            stored = replace(stored, protected=stored.protected | _patch_paths(bundle) - _patch_paths(head))
        whole = await asyncio.to_thread(carry, onto, head, bundle, head, stored)
        carried = tuple(dict.fromkeys(commit for step in steps for commit in step.carried))
        merge = RepoMerge(
            onto,
            whole.delta,
            tuple(conflict for step in steps for conflict in step.config_conflicts),
            rebased_patches=tuple(patch for step in steps for patch in step.rebased_patches),
            united_tables=tuple(table for step in steps for table in step.united_tables),
            replaced_patches=tuple(patch for step in steps for patch in step.replaced_patches),
            carried=carried,
        )
        return BundleMerge(merge, tuple(members), tuple(left_out))

    async def _carried_onto(self, candidate: str, onto: str, ours: Recipe, stored: Stored) -> RepoMerge | None:
        """``candidate``'s change carried onto ``ours``, the recipe ``onto`` becomes with the bundle so far; ``None``
        when it is on ``onto`` already, does not combine or leaves a patch that does not apply."""
        resolved = await self._commit_of(candidate)
        try:
            base = await self._merge_base(resolved, onto)
            if base == resolved:
                return None
            recipes = [await self.read_recipe(commit) for commit in (base, resolved)]
            step = await asyncio.to_thread(carry, onto, *recipes, ours, stored)
        except RecipeError:
            return None
        if not step.clean:
            return None
        carried = await self._git("rev-list", "--topo-order", "--reverse", resolved, f"^{onto}")
        return replace(step, carried=tuple(carried.split()))

    async def _merge_base(self, candidate: str, onto: str) -> str:
        """The one nearest common ancestor of ``candidate`` and ``onto``; ``RecipeError`` when there are several, since
        the change carried from each differs."""

        bases = (await self._git("merge-base", "--all", candidate, onto)).split()
        if len(bases) > 1:
            raise RecipeError(f"{candidate} and {onto} have several merge bases: {', '.join(bases)}")
        return bases[0]

    async def _contributed(self, onto: str, merged: frozenset[str]) -> frozenset[str]:
        """The paths of the patches the merges of ``merged`` added to ``onto``'s first-parent chain: each merge commit's
        later parents are the commits it merged, and the patches its recipe added over its first parent's are theirs.

        The chain is read from the repository, but ``merged`` is the session's: a head carried into a later session
        protects nothing until that set is derived from the merge log.
        """
        if not merged:
            return frozenset()
        chain = (await self._git("rev-list", "--first-parent", "--parents", await self._commit_of(onto))).splitlines()
        paths: set[str] = set()
        for commit, first, *others in (line.split() for line in chain if len(line.split()) > 2):
            if not merged.isdisjoint(others):
                before, after = await self.read_recipe(first), await self.read_recipe(commit)
                paths |= _patch_paths(after) - _patch_paths(before)
        return frozenset(paths)

    def stack_kernel_slot(self, wanted: KernelArtifact, current: KernelArtifact) -> KernelArtifact | None:
        """What a replay of ``wanted`` puts in the slot ``current`` fills, read and kept in the stores :meth:`merge`
        uses (:func:`~hyperloom_multibranch.repo.merge.stack_kernel_slot`)."""

        return stack_kernel_slot(wanted, current, self._stored())

    def _stored(self) -> Stored:
        return Stored(self._stack_files, self._table_bytes, self._put_patch, self._blobs.put, self._aiter_configs)

    async def commit_integration(self, onto: str, delta: RecipeDelta, merged_from: Sequence[str]) -> str:
        recipe = apply(await self.read_recipe(onto), delta)
        merged = [await self._commit_of(commit) for commit in merged_from]
        for commit in merged:
            recipe = with_unexpressed(recipe, (await self.read_recipe(commit)).unexpressed)
        return await self._land(INTEGRATION_BRANCH, onto, recipe, (onto, *merged), f"merge {', '.join(merged)}")

    async def rewind(self, branch: str, commit: str, parent: str) -> None:
        moved = await self._run(_UPDATE_REF, _ref(branch), await self._commit_of(parent), commit)
        if moved.code != 0:
            raise BranchMoved(f"{branch} is no longer at {commit}: {moved.err}")

    async def reroot(self, branch: str, head: str, onto: str) -> None:
        resolved, target = await self._commit_of(head), await self._commit_of(onto)
        if (await self._run("merge-base", "--is-ancestor", target, resolved)).code == 0:
            raise ValueError(f"{branch} at {resolved} holds {target} already")
        await self._git(_UPDATE_REF, f"{_REROOTED}/{branch}/{resolved}", resolved)
        moved = await self._run(_UPDATE_REF, _ref(branch), target, resolved)
        if moved.code != 0:
            raise BranchMoved(f"{branch} is no longer at {resolved}: {moved.err}")

    async def store_patch(self, component: str, text: str) -> PatchRef:
        return await asyncio.to_thread(self._put_patch, component, text)

    async def apply_error(self, stack: ComponentPatches, text: str) -> str | None:
        return await asyncio.to_thread(self._apply_error, stack, text)

    async def patch_text(self, commit: str, path: str) -> str:
        """Also the text of a patch :meth:`store_patch` kept under ``path`` that the recipe does not name: a patch
        conflict a merge split off a candidate's patch is in no recipe until the resolver rewrites it."""
        recipe = await self.read_recipe(commit)
        for patch in (patch for component in recipe.components for patch in component.patches):
            if patch.path == path:
                return (await asyncio.to_thread(self._blobs.get, patch.sha256)).decode()
        stored = _STORED_PATCH.fullmatch(path)
        if stored is not None:
            try:
                return (await asyncio.to_thread(self._blobs.get, stored["sha256"])).decode()
            except BlobError:
                pass
        raise KeyError(f"the recipe of {commit} names no patch {path}")

    async def checkout(self, commit: str, destination: Path) -> None:
        recipe = await self.read_recipe(commit)
        files = [(patch.path, patch.sha256) for component in recipe.components for patch in component.patches]
        files += [(kernel.ref, kernel.sha256) for kernel in recipe.kernels]
        await asyncio.to_thread(self._write_files, files, destination)

    def _table_bytes(self, artifact: KernelArtifact) -> bytes:
        return self._blobs.get(artifact.sha256)

    def _put_patch(self, component: str, text: str) -> PatchRef:
        sha256 = self._blobs.put(text.encode())
        return PatchRef(f"{PATCH_ROOT}/{component}/{sha256}.patch", sha256)

    def _apply_error(self, stack: ComponentPatches, text: str) -> str | None:
        scratch = ScratchBlobs(self._blobs)
        sha256 = scratch.put(text.encode())
        patch = PatchRef(f"{PATCH_ROOT}/{stack.component}/{sha256}.patch", sha256)
        try:
            stack_files(replace(stack, patches=(*stack.patches, patch)), self._roots(stack), scratch, ())
        except PatchApplyError as error:
            return str(error)
        return None

    @staticmethod
    def _layout(component: ComponentPatches) -> ComponentLayout:
        layout = COMPONENT_LAYOUTS.get(component.component)
        if layout is None:
            raise UnsupportedComponentError(f"{component.component} has no package layout; its patches cannot apply")
        return layout

    def _stack_files(self, component: ComponentPatches, paths: Collection[str]) -> Files:
        return stack_files(component, self._roots(component), self._blobs, paths)

    def _roots(self, component: ComponentPatches) -> tuple[SourceRoot, ...]:
        """The trees ``component``'s patches apply to: its installed package, the meta trees the specialist's source
        base lays out beside it, and its native tree."""
        layout = self._layout(component)
        return source_roots(layout, self._locator(layout).path, self._native.get(component.component))

    def _write_files(self, files: Sequence[tuple[str, str]], destination: Path) -> None:
        for path, sha256 in files:
            target = destination / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(self._blobs.get(sha256))

    async def _land(self, branch: str, parent: str, recipe: Recipe, parents: tuple[str, ...], message: str) -> str:
        current = await self.head(branch)
        if current != parent:
            raise BranchMoved(f"{branch} is at {current}, not {parent}")
        made = await self._write(recipe, parents, message)
        moved = await self._run(_UPDATE_REF, _ref(branch), made, parent)
        if moved.code == 0:
            return made
        if await self.head(branch) != parent:
            raise BranchMoved(f"{branch} moved off {parent} while {made} was written")
        raise GitError(f"cannot move {branch} to {made}: {moved.err}")

    async def _write(self, recipe: Recipe, parents: tuple[str, ...], message: str) -> str:
        document = json.dumps(recipe.to_document(), indent=2, sort_keys=True) + "\n"
        blob = await self._git("hash-object", "-w", "--stdin", stdin=document.encode())
        tree = await self._git("mktree", stdin=f"100644 blob {blob}\t{RECIPE_FILE}\n".encode())
        parent_args = [arg for parent in parents for arg in ("-p", parent)]
        return await self._git("commit-tree", "--no-gpg-sign", *parent_args, "-m", message, tree)

    async def _commit_of(self, commit: str) -> str:
        resolved = await self._resolve(f"{commit}^{{commit}}") if _COMMIT_ID.fullmatch(commit) else None
        if resolved is None:
            raise KeyError(commit)
        return resolved

    async def _resolve(self, name: str) -> str | None:
        resolved = await self._run("rev-parse", "--verify", "--quiet", name)
        return resolved.out if resolved.code == 0 else None

    async def _git(self, *args: str, stdin: bytes | None = None) -> str:
        completed = await self._run(*args, stdin=stdin)
        if completed.code != 0:
            raise GitError(f"git {args[0]} exited {completed.code}: {completed.err}")
        return completed.out

    async def _run(self, *args: str, stdin: bytes | None = None) -> _Completed:
        process = await asyncio.create_subprocess_exec(
            "git",
            *args,
            cwd=self._path,
            env=self._env,
            stdin=asyncio.subprocess.DEVNULL if stdin is None else asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(process.communicate(stdin), GIT_TIMEOUT_S)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise GitError(f"git {args[0]} did not finish in {GIT_TIMEOUT_S:g}s") from None
        return _Completed(process.returncode, out.decode().strip(), err.decode(errors="replace").strip())


__all__ = ["GIT_TIMEOUT_S", "RECIPE_FILE", "GitError", "GitExperimentRepo"]
