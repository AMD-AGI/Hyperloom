# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Give the rewrite campaign a base commit that is what the server is running.

Hyperloom dirties the framework trees by design: the TraceLens and ck-blockscale
instrumentation is patched in place, fusion writes its module beside the model it
edits, and every lane's KEEP sits uncommitted until the session ends. So by the
time the controller starts, "the code being served" and "the code at HEAD" are
different things, and a campaign has to choose which one it means.

It means the former. A rewrite is measured against the running server, and the
patch it produces has to apply to the tree that server was built from. Sealing
those changes into a commit makes that tree nameable: the campaign's base commit,
the diff's starting point, and the state a borrowed repository is handed back at
all become the same object id.

Only tracked changes are sealed. Untracked files -- tuned GEMM tables, JIT
caches, an operator's own notes -- are not part of any patch and have no business
in a commit this session created.

Because it commits, sealing is also where a campaign a previous session never
handed back has to be returned. Nothing else can: both reclaims run after the
campaign, in the process that died holding it.
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .kernel_evidence import campaign_repositories
from kernelforge.kernel_rewrite_controller.worktree import (
    CAMPAIGN_BRANCH_PREFIX,
    reclaim_campaign_branch,
    untracked_paths,
)
from kernelforge.loop.editable_repo import acquire_repo_lock, release_repo_lock


@dataclass(frozen=True)
class RepoBaseline:
    """What one repository looked like when the campaign was given it.

    The commit alone is not enough to hand a repository back. A campaign can
    commit a file the base does not carry, and switching away from its branch
    turns that file untracked rather than removing it -- so without knowing
    which untracked paths were already there, nothing can tell the campaign's
    leavings from the operator's own, and the residue this whole path exists to
    stop accumulates again.
    """

    commit: str
    untracked: frozenset[str] = frozenset()


log = logging.getLogger(__name__)

_GIT_TIMEOUT_SEC = 120

#: A commit nobody authored needs an author anyway, and the container's Git has
#: no global identity to fall back on.
_COMMIT_IDENTITY = (
    "-c",
    "user.name=hyperloom",
    "-c",
    "user.email=hyperloom@localhost",
)


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT_SEC,
        check=check,
    )


def session_branch_name(session_id: str, macro_cycle: int) -> str:
    """Name the branch one session's sealed baselines live on."""
    safe = "".join(character if character.isalnum() or character in "-_." else "-" for character in str(session_id))
    return f"hyperloom/{safe or 'session'}-c{max(0, int(macro_cycle))}"


def _seal_one(repo: Path, branch: str) -> RepoBaseline:
    """Commit this repository's tracked changes and describe what it then was."""
    head = _git(repo, "rev-parse", "HEAD").stdout.strip().lower()
    if not _git(repo, "status", "--porcelain", "--untracked-files=no").stdout.strip():
        return RepoBaseline(commit=head, untracked=untracked_paths(repo))

    current = _git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    if current != branch:
        # ``-B`` rather than ``-b``: a second entry into KERNEL within one
        # session finds the branch already there, and it should continue from
        # where the tree is now rather than refuse or rewind to an older seal.
        _git(repo, "checkout", "-B", branch)
    _git(repo, "add", "--update")
    _git(
        repo,
        *_COMMIT_IDENTITY,
        "commit",
        "--message",
        f"hyperloom: seal the serving tree for {branch}",
    )
    sealed = _git(repo, "rev-parse", "HEAD").stdout.strip().lower()
    # Named in the log because the repository is left on this branch: the seal
    # has no inverse, and an operator reading their own checkout afterwards
    # should be able to find out from here what moved it.
    log.info("sealed %s at %s (was %s) and left it on %s", repo, sealed[:12], head[:12], branch)
    return RepoBaseline(commit=sealed, untracked=untracked_paths(repo))


def _abandoned_campaign_branch(repo: Path) -> str:
    """Name the campaign branch this repository is still sitting on, if any."""
    current = _git(repo, "rev-parse", "--abbrev-ref", "HEAD", check=False).stdout.strip()
    return current if current.startswith(CAMPAIGN_BRANCH_PREFIX) else ""


def _seal_repository(repo: Path, branch: str) -> RepoBaseline | None:
    """Seal one repository, first putting back a campaign nothing else will.

    Sealing commits whatever the tree holds, so a repository left on a campaign
    branch has to be returned before this runs or the commit is that campaign's
    rewrite -- unvalidated, unattributed, and from then on the baseline every
    measurement in this session and every session after it is taken against.
    The reclaim that runs when the controller exits cannot cover this: it is in
    the process the host killed. This is the other end of that window, and the
    last moment before the tree stops being answerable.

    ``None`` when the repository must not be sealed at all. Its operators are
    skipped, which is what a missing baseline already means, and the alternative
    is committing a rewrite nobody measured.
    """
    abandoned = _abandoned_campaign_branch(repo)
    if not abandoned:
        return _seal_one(repo, branch)
    # Only now, and only for this: an ordinary tree seals without the lock, and
    # a tree on a campaign branch may be one a live campaign is still writing.
    lock = acquire_repo_lock(str(repo))
    if lock is None:
        log.warning(
            "not sealing %s: it is on campaign branch %s and its lock could not be taken, "
            "so either a live campaign is writing this tree or the lock file is unreachable; "
            "in-place operators here will be skipped rather than measured against a rewrite "
            "that may still be in progress",
            repo,
            abandoned,
        )
        return None
    try:
        reclaimed = reclaim_campaign_branch(repo)
        if not reclaimed:
            log.warning(
                "not sealing %s: it is on campaign branch %s and no record of what it held "
                "beforehand survived, so nothing can say which commit to return it to. "
                "Recover what that branch is worth by hand; sealing it would make its "
                "rewrite this session's baseline",
                repo,
                abandoned,
            )
            return None
        log.warning(
            "reclaimed %s from campaign branch %s before sealing it: a previous session was "
            "killed outright and never gave the repository back",
            repo,
            reclaimed,
        )
        return _seal_one(repo, branch)
    finally:
        release_repo_lock(lock)


def seal_campaign_baseline(
    state: object,
    *,
    session_id: str,
    macro_cycle: int,
) -> dict[str, RepoBaseline]:
    """Seal every configured source repository and return what each was pinned to.

    Best-effort per repository. One tree that cannot be sealed -- no Git, no
    write permission -- costs that repository's operators, which the controller
    refuses individually; it must not cost the whole KERNEL phase.
    """
    branch = session_branch_name(session_id, macro_cycle)
    baselines: dict[str, RepoBaseline] = {}
    for repo in campaign_repositories(state):
        try:
            baseline = _seal_repository(repo, branch)
        except (OSError, subprocess.SubprocessError) as error:
            log.warning(
                "could not seal the serving tree in %s: %s; in-place operators in this "
                "repository will be skipped for having no base commit to borrow at",
                repo,
                error,
            )
            continue
        if baseline is not None:
            baselines[str(repo)] = baseline
    return baselines


def reclaim_campaign_repositories(baselines: Mapping[str, RepoBaseline]) -> dict[str, str]:
    """Put back a repository the controller was killed before it could return.

    A hard timeout kills the process tree, so the controller's own restore never
    runs and the repository is left sitting on a campaign branch. Integration
    then reads a HEAD that is not the base commit its publications name and
    refuses every patch -- which is exactly the case incremental publication
    exists to survive, so reclaiming here is what keeps a killed campaign's
    validated work landable.

    Returns the repositories that were actually reclaimed.
    """
    reclaimed: dict[str, str] = {}
    for raw_repo, baseline in baselines.items():
        repo = Path(raw_repo)
        try:
            # One implementation, reached from both directions: here after a kill,
            # and from the next borrow finding the repository still on a branch.
            branch = reclaim_campaign_branch(
                repo,
                baseline.commit,
                baseline_untracked=baseline.untracked,
            )
        except (OSError, subprocess.SubprocessError) as error:
            log.warning("could not reclaim %s after the controller exited: %s", repo, error)
            continue
        if branch:
            reclaimed[str(repo)] = branch
            log.warning("reclaimed %s from abandoned campaign branch %s", repo, branch)
    return reclaimed


__all__ = [
    "RepoBaseline",
    "reclaim_campaign_repositories",
    "seal_campaign_baseline",
    "session_branch_name",
]
