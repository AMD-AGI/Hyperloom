# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The spawn request and reaped result a specialist subprocess dispatch exchanges."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hyperloom.common.deadline import Deadline
from hyperloom.inference_optimizer.trace.agent_trace import AgentTrace


#: How long the reaper waits on a specialist nobody bounded, matching the
#: largest deadline the dispatcher hands down.
UNBOUNDED_REAP_CAP_SEC: float = 4 * 60 * 60.0


@dataclass
class SpecialistSubprocessResult:
    """Outcome of one specialist subprocess invocation.

    The SpecialistRunner translates this into its own
    :class:`SpecialistRunResult`.
    """

    done_payload: dict[str, Any] | None = None
    """Parsed ``specialist_done.json`` content, or None when the file never
    appeared (the runner then falls back to ``build_empty_specialist_done``)."""

    exit_code: int | None = None
    """Subprocess exit code (None when killed before exit)."""

    elapsed_seconds: float = 0.0

    timed_out: bool = False
    """True when the dispatcher killed the subprocess at its deadline."""

    stale_heartbeat: bool = False
    """True when the heartbeat went stale and the dispatcher killed
    the subprocess."""

    process_log_path: str = ""
    patches: list[str] = field(default_factory=list)
    """Full filesystem paths of patch files discovered under
    ``<worktree>/patches/`` and ``<workspace>/patches/`` (absolute, since both
    roots are session-absolute). Consumers read and sandbox-check these
    directly — do not join them onto a base."""

    patch_roots: dict[str, str] = field(default_factory=dict)
    """Apply root per patch path, for patches harvested from a worktree.

    Empty for patches merely found on disk, whose target tree is not knowable at
    collection time and must therefore still be resolved from their contents."""

    trace: AgentTrace = field(default_factory=AgentTrace)
    """What the agent CLI's own log recorded: token usage (re-entering the
    unified ledger the production specialist's spend), the reply text, the
    intel/tool calls and the per-turn usage. Empty when the log was missing or
    truncated."""

    worktree_refused: str = ""
    """Why nothing was collected from the worktree, ``error`` saying it too: its path no longer named the directory
    the specialist was given (:class:`~.worktrees.HeldWorktree`), so no patch, scanned or claimed, is taken from the
    run. Empty when it did, or there was no worktree."""

    error: str = ""

    @property
    def patch_claims_ignored(self) -> bool:
        """Whether the patches the specialist's done payload claims are not read: a harvested diff is the whole round,
        so a claim only restates it, and a refused worktree gives no patch at all."""
        return bool(self.patch_roots or self.worktree_refused)


@dataclass(frozen=True)
class SpecialistSpawn:
    """One specialist subprocess to spawn and reap.

    Attributes:
        task_id: Task identifier used for logging / workspace layout.
        workspace: ``runs/specialist/<task_id>/`` -- where prompt.md,
            process.log, heartbeat.json, and specialist_done.json live.
        worktree: Per-task git worktree (None when worktree setup failed; the
            dispatcher still spawns the agent but it has no write-isolated
            tree, only ``--add-dir <workspace>``).
        worktree_base: Base checkout the worktree was branched off; the clean
            base patch harvesting diffs against.
        system_prompt: System prompt assembled by
            :func:`specialist_prompt_builder.build_specialist_prompts`.
        user_prompt: User prompt from the same builder.
        disallowed_tools: Tool names to deny in the agent CLI subprocess.
            Passed as ``--disallowedTools`` to the Claude CLI; has no Codex or
            dsh equivalent (Codex containment uses ``--sandbox``).
        gpu_ids: GPU ids to expose to the subprocess.
        deadline: Absolute instant the reaper kills at. ``None`` means the
            caller set no bound and the reaper falls back to
            :data:`UNBOUNDED_REAP_CAP_SEC`. A deadline already in the past
            kills on the first poll.
        progress_cb: Optional async callback invoked with each new partial
            checkpoint the specialist writes while it is still alive.
            Exceptions from it never affect the run.
        readable: Paths besides its toolchain the specialist reads (a local
            model): bound read-only when it runs in a bubblewrap sandbox.
        worktree_admin: The admin directory of ``worktree`` as its repository
            records it (:mod:`.worktrees`): the app's git over the worktree goes
            through it alone (``None``: no git, only ``patches/`` is scanned).
        low_tier: Runs on the dispatcher config's ``low_model`` rather than its
            ``model`` when one is set (:attr:`.profile.SpecialistProfile.low_tier`).
    """

    task_id: str
    workspace: Path
    worktree: Path | None
    worktree_base: Path | None
    system_prompt: str
    user_prompt: str
    disallowed_tools: frozenset[str] | set[str] | tuple[str, ...] = frozenset()
    gpu_ids: tuple[int, ...] = ()
    deadline: Deadline | None = None
    progress_cb: Any = None
    readable: tuple[Path, ...] = ()
    worktree_admin: Path | None = None
    low_tier: bool = False
