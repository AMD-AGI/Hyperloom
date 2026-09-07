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
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from hyperloom.orchestrator.kernel.forge_handoff import source_repository_roots

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


def _seal_one(repo: Path, branch: str) -> str:
    """Commit this repository's tracked changes and return the commit to build on."""
    head = _git(repo, "rev-parse", "HEAD").stdout.strip().lower()
    if not _git(repo, "status", "--porcelain", "--untracked-files=no").stdout.strip():
        return head

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
    log.info("sealed %s at %s (was %s) on %s", repo, sealed[:12], head[:12], branch)
    return sealed


def seal_campaign_baseline(
    state: object,
    *,
    session_id: str,
    macro_cycle: int,
) -> dict[str, str]:
    """Seal every configured source repository and return what each was pinned to.

    Best-effort per repository. One tree that cannot be sealed -- no Git, no
    write permission -- costs that repository's operators, which the controller
    refuses individually; it must not cost the whole KERNEL phase.
    """
    branch = session_branch_name(session_id, macro_cycle)
    pins: dict[str, str] = {}
    for repo in source_repository_roots(state):
        try:
            pins[str(repo)] = _seal_one(repo, branch)
        except (OSError, subprocess.SubprocessError) as error:
            log.warning("could not seal the serving tree in %s: %s", repo, error)
    return pins


__all__ = [
    "seal_campaign_baseline",
    "session_branch_name",
]
