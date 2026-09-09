# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pinned Git worktrees for controller-owned operator campaigns."""

from __future__ import annotations

import contextlib
import hashlib
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

from kernelforge.kernel_rewrite_controller.contracts import KernelRewriteTask
from kernelforge.kernel_rewrite_controller.paths import ControllerLayout
from kernelforge.llm.git import GitError, git
from kernelforge.loop.editable_repo import (
    RepoLock,
    acquire_repo_lock,
    needs_inplace,
    release_repo_lock,
)


#: Directory ``forge-loop`` writes its campaign state, JIT caches and iteration
#: archive into, relative to the workspace it optimizes.
FORGE_LOOP_OUTPUT_DIRNAME = "forge_experiments"

#: Prefix of the branch one campaign commits onto. Shared with the sweep that
#: reclaims a repository from a run the host killed before it could restore.
CAMPAIGN_BRANCH_PREFIX = "forge/controller/"

#: Tells a recorded object id apart from a recorded branch name, so HEAD is
#: put back the way it was found rather than always as one or the other.
_COMMIT_LIKE = re.compile(r"^[0-9a-f]{40}$|^[0-9a-f]{64}$")


class WorktreeError(RuntimeError):
    """An operator worktree could not be created or validated."""


@dataclass(frozen=True)
class OperatorWorktree:
    """One task's workspace pinned to the shared base commit.

    ``inplace`` distinguishes the two shapes this can take. An ordinary task
    gets a private checkout. A task whose repository answers ``import`` from an
    editable-install finder cannot: that finder is pinned to the live directory
    and PYTHONPATH cannot outrank it, so a private checkout would be edited and
    never loaded, and the campaign would measure the unmodified original. Such a
    task borrows the live repository instead, which is why it also carries the
    lock that makes the borrow exclusive.
    """

    repo_root: Path
    workspace: Path
    branch: str
    base_commit: str
    kernel_path: Path
    source_files: tuple[Path, ...]
    inplace: bool = False
    lock: RepoLock | None = None
    #: Where HEAD pointed before the campaign: a branch name, or an object id
    #: when the repository was detached. Restored without touching the tree.
    origin_ref: str = ""
    #: Untracked paths the repository already held when it was borrowed. What a
    #: campaign leaves behind cannot be told from what its owner keeps by name
    #: -- a rewrite is free to add a header or a config beside the kernel -- so
    #: the only honest test is whether the path was there before.
    baseline_untracked: frozenset[str] = frozenset()
    #: Where the campaign's own bookkeeping is moved on the way out, so a run
    #: that published nothing can still be read afterwards. It cannot stay in
    #: the repository: every in-place task in one repository is handed the same
    #: experiments directory, so the next would inherit this one's state.
    archive_dir: Path | None = None


def _git_toplevel(repo_root: Path) -> Path:
    try:
        result = git("rev-parse", "--show-toplevel", cwd=repo_root)
    except GitError as error:
        raise WorktreeError(f"repo_root is not a Git checkout: {repo_root}: {error}") from error
    return Path(result.stdout.strip()).resolve()


def _require_commit(repo_root: Path, commit: str) -> None:
    result = git("cat-file", "-e", f"{commit}^{{commit}}", cwd=repo_root, check=False)
    if result.returncode != 0:
        raise WorktreeError(f"base commit does not exist in {repo_root}: {commit}")


def _branch_name(operator_id: str) -> str:
    digest = hashlib.sha256(operator_id.encode("utf-8")).hexdigest()[:16]
    return f"{CAMPAIGN_BRANCH_PREFIX}{digest}-{uuid.uuid4().hex[:8]}"


def _head_ref(repo_root: Path) -> str:
    """Name the ref HEAD points at, or its object id when detached."""
    branch = git("rev-parse", "--abbrev-ref", "HEAD", cwd=repo_root).stdout.strip()
    if branch and branch != "HEAD":
        return branch
    return git("rev-parse", "HEAD", cwd=repo_root).stdout.strip().lower()


def untracked_paths(repo_root: Path) -> frozenset[str]:
    """Repository-relative untracked paths, as the restore will see them.

    ``--exclude-standard`` on purpose: the ignored set is runtime output -- JIT
    caches, bytecode -- that no campaign owns and no restore should remove. Both
    the inventory and the comparison against it use these same flags, or the two
    would not be answering the same question.
    """
    listed = git("ls-files", "--others", "--exclude-standard", "-z", cwd=repo_root, check=False)
    return frozenset(path for path in (listed.stdout or "").split("\0") if path)


def reclaim_campaign_branch(
    repo_root: Path,
    base_commit: str,
    *,
    baseline_untracked: frozenset[str] = frozenset(),
) -> str:
    """Return a repository a campaign never handed back, and name the branch.

    Reached from two directions -- the next borrow finding the repository still
    on a campaign branch, and Hyperloom finding it there after the host killed
    the controller -- so it lives here rather than once per caller. Returns the
    branch it reclaimed, or ``""`` when there was nothing to reclaim.

    ``checkout --force`` restores tracked content but leaves whatever the
    campaign committed and the switch untracked, so the inventory is what
    finally removes it. Without one, nothing here can tell those files from the
    operator's own and they are left alone.
    """
    branch = git("rev-parse", "--abbrev-ref", "HEAD", cwd=repo_root, check=False).stdout.strip()
    if not branch.startswith(CAMPAIGN_BRANCH_PREFIX):
        return ""
    git("checkout", "--force", base_commit, cwd=repo_root)
    git("branch", "-D", branch, cwd=repo_root, check=False)
    remove_foreign_untracked(repo_root, baseline_untracked)
    return branch


def _require_tree_at(repo_root: Path, base_commit: str) -> None:
    """Refuse to borrow a repository that is not the base commit it claims.

    Borrowing a tree that already differs would fold whoever else's edit into
    this campaign's patch and revert it on the way out. The host commits the
    pre-campaign state before the controller starts, so the honest answer when
    this fails is to skip the operator rather than to guess whose change it is.
    """
    dirty = git("diff", "--quiet", base_commit, cwd=repo_root, check=False)
    if dirty.returncode != 0:
        changed = git("diff", "--name-only", base_commit, cwd=repo_root, check=False).stdout.strip()
        raise WorktreeError(
            f"{repo_root} carries uncommitted changes against base commit {base_commit} and cannot be "
            f"borrowed for an in-place campaign: {changed.replace(chr(10), ', ')}. "
            "The KERNEL entry seals the serving tree into this commit before the controller starts; "
            "a repository that is still dirty here is one whose seal did not take."
        )


def remove_foreign_untracked(repo_root: Path, baseline_untracked: frozenset[str]) -> None:
    """Delete untracked paths the repository did not hold before the campaign.

    Ownership is decided by the inventory, not by name. A rewrite may add a
    header, a config or a generated kernel beside the one it was pointed at, and
    ``is_producer_owned_path`` knows nothing of those -- it only recognises forge
    bookkeeping. Anything untracked and absent from the inventory is this
    campaign's, and anything in it is the operator's and stays.

    Never ``git clean``: that is exactly the call that cannot make this
    distinction.
    """
    for relative in sorted(untracked_paths(repo_root) - baseline_untracked, reverse=True):
        target = (repo_root / relative).resolve()
        if not target.is_relative_to(repo_root):
            continue
        with contextlib.suppress(OSError):
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
            else:
                target.unlink()
        parent = target.parent
        while parent != repo_root and parent.is_relative_to(repo_root):
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent
    # Gitignored by its own ``.gitignore``, so the listing above never sees it.
    output_root = repo_root / FORGE_LOOP_OUTPUT_DIRNAME
    if output_root.is_dir():
        shutil.rmtree(output_root, ignore_errors=True)


def _archive_campaign_output(worktree: OperatorWorktree) -> None:
    """Move the campaign's bookkeeping out of the repository, keeping it.

    It cannot stay: every in-place task in one repository is handed the same
    experiments directory, so leaving this one's state would have the next
    campaign resume it. It is worth keeping: when a run publishes no patch, this
    is the only account of what the loop actually did.
    """
    source = worktree.repo_root / FORGE_LOOP_OUTPUT_DIRNAME
    destination = worktree.archive_dir
    if destination is None or not source.is_dir():
        return
    with contextlib.suppress(OSError, shutil.Error):
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / FORGE_LOOP_OUTPUT_DIRNAME
        shutil.rmtree(target, ignore_errors=True)
        shutil.move(str(source), str(target))


def _remove_partial_worktree(repo_root: Path, workspace: Path, branch: str) -> None:
    with contextlib.suppress(Exception):
        git("worktree", "remove", "--force", str(workspace), cwd=repo_root, check=False)
    shutil.rmtree(workspace, ignore_errors=True)
    with contextlib.suppress(Exception):
        git("branch", "-D", branch, cwd=repo_root, check=False)


def _ignore_forge_loop_output(workspace: Path) -> None:
    """Hide forge-loop's own output directory from Git inside one worktree.

    forge-loop optimizes a Git workspace while writing its campaign state and
    JIT caches into that same workspace, and its workspace guard rejects any
    untracked path the caller did not declare. It therefore requires the
    workspace to ignore that directory -- the packaged examples satisfy this by
    writing a ``.gitignore`` themselves. A framework repository never does, so
    without this the first JIT compile inside an operator worktree fails the
    iteration for infrastructure output rather than for anything the agent did.

    The rule is a ``.gitignore`` inside the directory rather than
    ``$GIT_DIR/info/exclude``: Git resolves ``info/`` against the common
    directory, so an exclude file would be ignored for this worktree and instead
    leak into the shared repository. Matching ``*`` also covers the file itself,
    which keeps it untracked and therefore out of the exported patch.
    """
    output_root = workspace / FORGE_LOOP_OUTPUT_DIRNAME
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / ".gitignore").write_text("*\n", encoding="utf-8")


def operator_workspace(task: KernelRewriteTask, layout: ControllerLayout) -> Path:
    """Where this task's forge-loop works, private checkout or live repository.

    Asked in one place because the answer has to match on both sides: the
    dispatch that creates the workspace and the recovery that reads a result out
    of it would otherwise look in different directories for an in-place task.
    """
    repo_root = task.repo_root.resolve()
    if needs_inplace(str(repo_root)):
        return repo_root
    return layout.workspace_dir(task.operator_id)


def _borrow_live_repository(task: KernelRewriteTask, layout: ControllerLayout) -> OperatorWorktree:
    """Take the live repository for one campaign, exclusively and reversibly."""
    repo_root = task.repo_root.resolve()
    lock = acquire_repo_lock(str(repo_root))
    if lock is None:
        raise WorktreeError(
            f"another in-place campaign already holds {repo_root}; "
            "an editable-install repository can only be borrowed by one at a time"
        )
    try:
        reclaim_campaign_branch(repo_root, task.base_commit)
        _require_tree_at(repo_root, task.base_commit)
        origin_ref = _head_ref(repo_root)
        # Taken before the branch is cut, so it describes the repository as its
        # owner left it and not as the campaign will.
        baseline_untracked = untracked_paths(repo_root)
        branch = _branch_name(task.operator_id)
        git("branch", "-D", branch, cwd=repo_root, check=False)
        git("checkout", "-b", branch, task.base_commit, cwd=repo_root)
        kernel_path, source_files = _validate_declared_sources(repo_root, task)
        _ignore_forge_loop_output(repo_root)
        return OperatorWorktree(
            repo_root=repo_root,
            workspace=repo_root,
            branch=branch,
            base_commit=task.base_commit,
            kernel_path=kernel_path,
            source_files=source_files,
            inplace=True,
            lock=lock,
            origin_ref=origin_ref,
            baseline_untracked=baseline_untracked,
            archive_dir=layout.workspace_dir(task.operator_id),
        )
    except Exception:
        release_repo_lock(lock)
        raise


def _validate_declared_sources(workspace: Path, task: KernelRewriteTask) -> tuple[Path, tuple[Path, ...]]:
    kernel_path = (workspace / task.kernel_path).resolve()
    if not kernel_path.is_relative_to(workspace) or not kernel_path.is_file():
        raise WorktreeError(f"kernel path is not a file in the base commit: {task.kernel_path}")
    source_files = tuple((workspace / relative).resolve() for relative in task.source_files)
    for source_file in source_files:
        if not source_file.is_relative_to(workspace) or not source_file.is_file():
            raise WorktreeError(f"source file is not a file in the base commit: {source_file}")
    return kernel_path, source_files


def release_operator_worktree(worktree: OperatorWorktree | None) -> None:
    """Hand a borrowed repository back at the base commit it was taken at.

    Safe to call once the patch has been exported, which is the whole reason a
    campaign's leavings are disposable: a best commit that passed correctness
    and the microbenchmark is already a published patch by the time this runs,
    so the tree it was built in carries nothing that is not saved elsewhere.

    Best-effort per step. A repository left half-restored is worse than one
    restored past a failing step, and the lock must come off either way or the
    next campaign on this repository cannot start at all.
    """
    if worktree is None or not worktree.inplace:
        return
    repo_root = worktree.repo_root
    try:
        _archive_campaign_output(worktree)
        # Tracked content first. A path the campaign created is not in the base
        # commit, so this cannot restore it -- it is still carried by the branch
        # HEAD is on, and only becomes visible as untracked once HEAD and the
        # index have moved below.
        changed = git("diff", "--name-only", worktree.base_commit, cwd=repo_root, check=False)
        for relative in (changed.stdout or "").splitlines():
            if relative.strip():
                git("checkout", worktree.base_commit, "--", relative.strip(), cwd=repo_root, check=False)
        # HEAD by ref and the index by commit, neither touching the tree.
        if worktree.origin_ref and not _COMMIT_LIKE.fullmatch(worktree.origin_ref):
            git("symbolic-ref", "HEAD", f"refs/heads/{worktree.origin_ref}", cwd=repo_root, check=False)
        elif worktree.origin_ref:
            git("update-ref", "--no-deref", "HEAD", worktree.origin_ref, cwd=repo_root, check=False)
        git("reset", "--quiet", worktree.base_commit, "--", ".", cwd=repo_root, check=False)
        # Last, because until the index matches the base commit a file the
        # campaign committed does not read as untracked and this cannot see it.
        remove_foreign_untracked(repo_root, worktree.baseline_untracked)
        git("branch", "-D", worktree.branch, cwd=repo_root, check=False)
    finally:
        release_repo_lock(worktree.lock)


def create_operator_worktree(
    task: KernelRewriteTask,
    layout: ControllerLayout,
) -> OperatorWorktree:
    """Create one fresh branch/worktree from the task's pinned base commit."""
    repo_root = task.repo_root.resolve()
    if _git_toplevel(repo_root) != repo_root:
        raise WorktreeError(f"repo_root must be the Git top-level directory: {repo_root}")
    _require_commit(repo_root, task.base_commit)
    if needs_inplace(str(repo_root)):
        return _borrow_live_repository(task, layout)

    workspace = layout.workspace_dir(task.operator_id)
    if workspace.exists():
        raise WorktreeError(f"operator workspace already exists and cannot be resumed: {workspace}")
    workspace.parent.mkdir(parents=True, exist_ok=True)
    branch = _branch_name(task.operator_id)
    try:
        git("worktree", "prune", cwd=repo_root, check=False)
        git(
            "worktree",
            "add",
            "-b",
            branch,
            str(workspace),
            task.base_commit,
            cwd=repo_root,
        )
        actual_head = git("rev-parse", "HEAD", cwd=workspace).stdout.strip().lower()
        if actual_head != task.base_commit:
            raise WorktreeError(f"worktree HEAD mismatch: created {actual_head}, expected {task.base_commit}")
        kernel_path, source_files = _validate_declared_sources(workspace, task)
        _ignore_forge_loop_output(workspace)
        return OperatorWorktree(
            repo_root=repo_root,
            workspace=workspace,
            branch=branch,
            base_commit=task.base_commit,
            kernel_path=kernel_path,
            source_files=source_files,
        )
    except Exception:
        _remove_partial_worktree(repo_root, workspace, branch)
        raise


def changed_files_from_base(
    worktree: OperatorWorktree,
    *,
    best_commit: str,
) -> tuple[str, ...]:
    """List the repo-relative paths one KEEP changes against the controller base.

    Read from Git rather than from the forge-loop manifest. The manifest is the
    optimizer's own account of what it edited; this is what the published patch
    will actually apply, and only the second one bounds what integration commits.
    The task's ``source_files`` do not bound it either -- forge-loop treats them
    as orientation, not as an edit allowlist -- so without this the scope of a
    published patch is not recorded anywhere its consumer can check.
    """
    best = str(best_commit or "").strip().lower()
    if not best:
        return ()
    output = git(
        # Without this Git renders a path holding any non-ASCII byte as a quoted,
        # escaped string, and the consumer stages what the name says.
        "-c",
        "core.quotePath=false",
        "diff",
        "--name-only",
        f"{worktree.base_commit}..{best}",
        cwd=worktree.workspace,
    ).stdout
    return tuple(line.strip() for line in output.splitlines() if line.strip())


def export_patch_from_base(
    worktree: OperatorWorktree,
    *,
    best_commit: str,
) -> str:
    """Export the full binary diff from the controller base to a Forge KEEP."""
    best = str(best_commit or "").strip().lower()
    if not best:
        raise WorktreeError("forge-loop returned no best commit")
    _require_commit(worktree.workspace, best)
    ancestor = git(
        "merge-base",
        "--is-ancestor",
        worktree.base_commit,
        best,
        cwd=worktree.workspace,
        check=False,
    )
    if ancestor.returncode != 0:
        raise WorktreeError(f"best commit {best} is not based on controller base {worktree.base_commit}")
    return str(
        git(
            "diff",
            "--binary",
            f"{worktree.base_commit}..{best}",
            cwd=worktree.workspace,
        ).stdout
    )


__all__ = [
    "CAMPAIGN_BRANCH_PREFIX",
    "FORGE_LOOP_OUTPUT_DIRNAME",
    "OperatorWorktree",
    "WorktreeError",
    "changed_files_from_base",
    "create_operator_worktree",
    "export_patch_from_base",
    "operator_workspace",
    "reclaim_campaign_branch",
    "release_operator_worktree",
    "remove_foreign_untracked",
    "untracked_paths",
]
