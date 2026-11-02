# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a specialist launch decides before it spawns: the child's environment, GPU visibility and user."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from hyperloom_common.fs_utils import read_bytes_nofollow, write_text_nofollow
from hyperloom.common.env_safety import (
    RunAs,
    RunAsError,
    missing_launch_capabilities,
    specialist_run_as,
)
from hyperloom.common.proctree import REAP_MARK_ENV
from hyperloom.common.visible_devices import GPU_MASK_ENV_NAMES
from hyperloom_common.llm.llm_attribution import inject_env as inject_attribution_env

from .agent_cli_config import SpecialistAgentUnavailableError, _build_specialist_env

_GPU_IDS_ENV = "INFERENCE_OPTIMIZER_SPECIALIST_GPU_IDS"
_AITER_JIT_ENV = "INFERENCE_OPTIMIZER_AITER_JIT_DIR"
_MCP_FLAG = "--mcp-config"


@dataclass(frozen=True)
class SpecialistLaunch:
    """A resolved launch: argv, child env, demote user, backend, and the task's prompt and log files."""

    cmd: list[str]
    env: dict[str, str]
    run_as: RunAs | None
    backend: str
    prompt_file: Path
    process_log: Path


def compose_launch_env(task_id: str) -> tuple[dict[str, str], str]:
    """Compose the child env for one launch; return it with the launch's reap mark."""
    # Compose a minimal env. Provider credentials are inherited by default
    # for compatibility with deployments that authenticate the agent CLI
    # via env; set HYPERLOOM_SPECIALIST_INHERIT_SECRET_ENV=0 to disable.
    env = _build_specialist_env()
    # Keep the spawned CLI off its non-essential traffic and auto-updater.
    from hyperloom_common.llm.llm_stability_env import apply_llm_stability_env

    apply_llm_stability_env(env)
    # Unique to this launch, so every process the CLI starts can be found after it detaches.
    reap_mark = uuid.uuid4().hex
    env[REAP_MARK_ENV] = reap_mark
    # The child spends against the gateway, so tag it or its spend lands
    # under no component at all. The task tag is what attributes the
    # requests of a child that dies before its result row -- the only
    # place such a child's token usage survives is the gateway's log.
    inject_attribution_env(env, component="specialist", operation="run_agent", task=task_id)
    return env, reap_mark


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

    Reads the environment and the capability set only; it changes nothing.

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
        if private.is_file() and not private.is_symlink():
            private.chmod(0o600)
    if worktree is not None and workspace not in worktree.parents:
        run_as.share_tree(worktree)
    if worktree is not None:
        run_as.share_worktree_git(worktree)
    run_as.apply_env(env, safe_dirs=[worktree] if worktree is not None else ())
    return run_as.popen_kwargs()


def stage_mcp_config(run_as: RunAs | None, workspace: Path, cmd: list[str]) -> None:
    """Point ``--mcp-config`` in ``cmd`` at a copy the demoted child can read, in place.

    The configured file sits in the app's runtime directory at 0600. The copy
    goes in the task workspace, and is made before :func:`prepare_demote` shares
    that workspace, so the account can read it. Nothing changes when nothing is
    demoted or ``cmd`` carries no ``--mcp-config``.
    """
    if run_as is None or _MCP_FLAG not in cmd:
        return
    at = cmd.index(_MCP_FLAG) + 1
    try:
        text = read_bytes_nofollow(cmd[at]).decode("utf-8")
        cmd[at] = str(write_text_nofollow(workspace, ("specialist_mcp.json",), text, mode=0o600))
    except (OSError, UnicodeDecodeError) as exc:
        raise SpecialistAgentUnavailableError(
            f"cannot stage the MCP config {cmd[at]} for the specialist: {exc}"
        ) from exc
