# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a local specialist launch decides before it spawns: GPU environment and the child's user."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from hyperloom.common.nofollow import read_bytes_nofollow, write_text_nofollow
from hyperloom.common.run_as import (
    RunAs,
    RunAsError,
    missing_launch_capabilities,
    set_group_shared_umask,
    specialist_run_as,
)
from hyperloom.common.visible_devices import GPU_MASK_ENV_NAMES

from .agent_cli_config import SpecialistAgentUnavailableError

_GPU_IDS_ENV = "INFERENCE_OPTIMIZER_SPECIALIST_GPU_IDS"
_AITER_JIT_ENV = "INFERENCE_OPTIMIZER_AITER_JIT_DIR"


def apply_gpu_env(env: dict[str, str], workspace: Path, gpu_ids: Sequence[int], gpu_lease: object | None) -> None:
    """Set the GPU and compiler-cache environment of a specialist, in place.

    Args:
        env: The child environment.
        workspace: The task workspace, home of the private compiler caches of a CPU specialist.
        gpu_ids: Cards pinned for the task (the logical view under Ray's mask when ``gpu_lease`` is set).
        gpu_lease: The Ray GPU lease when the specialist runs in a Ray actor, else ``None``.
    """
    if gpu_lease is not None:
        # Ray-managed GPU execution: Ray sets *_VISIBLE_DEVICES in
        # the actor's worker; never let the caller env pin them (that would
        # override Ray's card assignment). ``gpu_ids`` here is the logical
        # 0..N-1 view the specialist sees under Ray's mask — kept only as the
        # informational count env for specialist tooling.
        for var in GPU_MASK_ENV_NAMES:
            env.pop(var, None)
        if gpu_ids:
            env[_GPU_IDS_ENV] = ",".join(str(g) for g in gpu_ids)
    elif gpu_ids:
        visible = ",".join(str(g) for g in gpu_ids)
        env["HIP_VISIBLE_DEVICES"] = visible
        env["CUDA_VISIBLE_DEVICES"] = visible
        env["ROCR_VISIBLE_DEVICES"] = visible
        env[_GPU_IDS_ENV] = visible
    else:
        # CPU specialists: hide all GPUs and use workspace-local compiler caches.
        for var in GPU_MASK_ENV_NAMES:
            env[var] = ""
        cache_root = workspace / ".cache"
        aiter_jit = str(cache_root / "aiter_jit")
        env["TRITON_CACHE_DIR"] = str(cache_root / "triton")
        env["TORCHINDUCTOR_CACHE_DIR"] = str(cache_root / "torchinductor")
        env["AITER_JIT_DIR"] = aiter_jit
        env[_AITER_JIT_ENV] = aiter_jit


def resolve_launch_user(gpu_lease: object | None) -> RunAs | None:
    """The account to run a local specialist as, or ``None`` for no change.

    Naming one also sets the app's own umask (``set_group_shared_umask``).

    Raises:
        RunAsError: The configured user is unusable, or the specialist runs in
            a Ray actor, which cannot be started as another user, or the app
            lacks CAP_SETUID, CAP_SETGID or CAP_KILL (a child it cannot signal
            is a child it cannot stop).
    """
    run_as = specialist_run_as()
    if run_as is not None and gpu_lease is not None:
        raise RunAsError("the Ray GPU actor path cannot run the specialist as another user")
    missing = missing_launch_capabilities() if run_as is not None else []
    if missing:
        raise RunAsError(f"running a specialist as another user needs root or {', '.join(missing)}")
    if run_as is not None:
        set_group_shared_umask()
    return run_as


def prepare_demote(
    run_as: RunAs | None,
    workspace: Path,
    worktree: Path | None,
    env: dict[str, str],
    private_files: Sequence[Path],
) -> dict[str, Any]:
    """Make the specialist's tree writable by ``run_as`` and return its ``Popen`` kwargs.

    ``None`` (no ``HYPERLOOM_SPECIALIST_USER``) returns no kwargs and touches
    nothing. The workspace is shared with the sticky bit so the child cannot
    delete the app's own files; ``private_files`` (the prompt and the log) are
    set back to 0600. A worktree outside the workspace is shared as well.
    """
    if run_as is None:
        return {}
    run_as.share_tree(workspace, sticky=True)
    for private in private_files:
        if private.exists():
            private.chmod(0o600)
    if worktree is not None and workspace not in worktree.parents:
        run_as.share_tree(worktree)
    if worktree is not None:
        run_as.share_worktree_git(worktree)
    run_as.apply_env(env, safe_dirs=[worktree] if worktree is not None else ())
    return run_as.popen_kwargs()


def stage_mcp_config(run_as: RunAs | None, workspace: Path, source: str | None) -> str | None:
    """A copy of the ``--mcp-config`` file in the task workspace, when the child cannot read the original.

    The original sits in the app's runtime directory at 0600. The copy is made
    before :func:`prepare_demote` shares the workspace, so it is readable by the
    account; ``None`` (use the configured path) when nothing is demoted or no
    config is set.
    """
    if run_as is None or not source:
        return None
    try:
        text = read_bytes_nofollow(source).decode("utf-8")
        return str(write_text_nofollow(workspace, ("specialist_mcp.json",), text, mode=0o600))
    except (OSError, UnicodeDecodeError) as exc:
        raise SpecialistAgentUnavailableError(
            f"cannot stage the MCP config {source} for the specialist: {exc}"
        ) from exc
