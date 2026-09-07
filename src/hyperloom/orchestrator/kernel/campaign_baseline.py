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

import importlib.util
import logging
import os
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from kernelforge.kernel_rewrite_controller.worktree import (
    CAMPAIGN_BRANCH_PREFIX,
    FORGE_LOOP_OUTPUT_DIRNAME,
)

log = logging.getLogger(__name__)

_GIT_TIMEOUT_SEC = 120

#: Packages that can hold a rewritable kernel and are installed from source.
#: Probed by name rather than read from configuration: a container serving one
#: framework has that one importable and the others absent, so the interpreter
#: already knows the answer, while the configured roots are routinely unset --
#: in the GLM-5.2 session all three sources were empty and the handoff reported
#: no source repository at all.
_FRAMEWORK_PACKAGES: tuple[str, ...] = ("vllm", "sglang", "aiter")
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


def _configured_repository_roots(state: Any) -> set[Path]:
    """Resolve explicitly configured source paths to distinct Git repository roots."""
    raw_paths = [
        getattr(state, "framework_repo_path", ""),
        os.environ.get("FRAMEWORK_REPO_PATH", ""),
    ]
    raw_paths.extend(
        value for value in os.environ.get("INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS", "").split(os.pathsep) if value
    )
    roots: set[Path] = set()
    for raw in raw_paths:
        text = str(raw or "").strip()
        if not text:
            continue
        path = Path(text).expanduser().resolve(strict=False)
        if path.is_file():
            path = path.parent
        for candidate in (path, *path.parents):
            if (candidate / ".git").exists():
                roots.add(candidate)
                break
    return roots


def _package_repository(name: str) -> Path | None:
    """Return the Git top level one framework package is imported from.

    ``None`` when the package is absent, or present as a wheel. A wheel carries
    no source to rewrite, so it is not a repository any campaign can name.
    """
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin:
        return None
    package_dir = Path(spec.origin).resolve().parent
    completed = _git(package_dir, "rev-parse", "--show-toplevel", check=False)
    if completed.returncode != 0:
        return None
    return Path(completed.stdout.strip()).resolve()


def campaign_repositories(state: Any) -> tuple[Path, ...]:
    """Every repository a rewrite could name, from the runtime and the config.

    The runtime is the authority and the configuration is an addition, not the
    other way round: an operator who points at a fourth checkout should be
    honoured, but nobody should have to configure the framework they are
    already serving.
    """
    roots = _configured_repository_roots(state)
    for name in _FRAMEWORK_PACKAGES:
        repo = _package_repository(name)
        if repo is not None:
            roots.add(repo)
    return tuple(sorted(roots, key=str))


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
    for repo in campaign_repositories(state):
        try:
            pins[str(repo)] = _seal_one(repo, branch)
        except (OSError, subprocess.SubprocessError) as error:
            log.warning("could not seal the serving tree in %s: %s", repo, error)
    return pins


def reclaim_campaign_repositories(pins: Mapping[str, str]) -> dict[str, str]:
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
    for raw_repo, base_commit in pins.items():
        repo = Path(raw_repo)
        try:
            branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD", check=False).stdout.strip()
            if not branch.startswith(CAMPAIGN_BRANCH_PREFIX):
                continue
            # ``--force`` because the campaign's own edits are still in the tree
            # and every one of them is either already an exported patch or of no
            # further use.
            _git(repo, "checkout", "--force", base_commit)
            _git(repo, "branch", "-D", branch, check=False)
            shutil.rmtree(repo / FORGE_LOOP_OUTPUT_DIRNAME, ignore_errors=True)
            reclaimed[str(repo)] = branch
            log.warning("reclaimed %s from abandoned campaign branch %s", repo, branch)
        except (OSError, subprocess.SubprocessError) as error:
            log.warning("could not reclaim %s after the controller exited: %s", repo, error)
    return reclaimed


__all__ = [
    "campaign_repositories",
    "reclaim_campaign_repositories",
    "seal_campaign_baseline",
    "session_branch_name",
]
