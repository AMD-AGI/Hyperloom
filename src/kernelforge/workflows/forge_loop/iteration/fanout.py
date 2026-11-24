# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A round's lanes: one workspace copy each, sessions run side by side on one queued device."""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from kernelforge.utils import device_queue
from kernelforge.utils.git import git, git_async
from kernelforge.utils.import_root import import_name_of, link_beside, linked_beside
from kernelforge.utils.process_reaping import (
    ReapReport,
    install_child_subreaper,
    reap_processes_under,
)
from kernelforge.workflows.forge_loop.campaign.experiments_layout import ExperimentsLayout
from kernelforge.workflows.forge_loop.roles.implementer_method import MAX_COMMAND_MINUTES

log = logging.getLogger(__name__)

DEVICE_QUEUE_SUFFIX = ".forge-device-queue"
LANE_ROOT_PREFIX = "forge-lanes-"
_GIT_PATH = "--git-path"


def campaign_device_queue_path(workspace: str | Path) -> Path:
    """The queue every lane of a campaign waits in for the device, beside the workspace so no lane copy carries it."""
    workspace = Path(workspace).expanduser().resolve()
    return workspace.parent / f"{workspace.name}{DEVICE_QUEUE_SUFFIX}"


@dataclass(frozen=True)
class LaneAssignment:
    """One lane of a round: its number, and the focus the round's Analysis gave it ("" when it gave none)."""

    lane_id: str
    focus: str = ""


@dataclass(frozen=True)
class LaneSeat:
    """Where one lane works: its workspace copy, and the runner and queue through which it reaches the device."""

    assignment: LaneAssignment
    lane_dir: Path
    device_runner: Path
    queue_root: Path
    usage_log: Path


@dataclass
class LaneResult:
    """What one lane's session left: its diff against the round's base, and what the session handed back."""

    assignment: LaneAssignment
    lane_dir: Path
    diff: str = ""
    rationale: str = ""
    # The session's hand-back: its plan line, end reason, gate verdict and measurement, and the callable that resumes
    # it for its record.
    sink: dict = field(default_factory=dict)
    error: str = ""
    # What this lane's teardown found. It is about the device every lane and the canonical measurement share, so a
    # lane that failed for a reason of its own still reports it.
    reaped: ReapReport | None = None

    @property
    def lane_id(self) -> str:
        return self.assignment.lane_id

    @property
    def changed(self) -> bool:
        """Whether the lane left a candidate the round may use."""
        return bool(self.diff.strip()) and not self.error and self.sink.get("integrity_violation") is not True

    @property
    def verified(self) -> bool:
        """Whether the lane's own stop check found its candidate correct and measured it."""
        return self.changed and self.sink.get("gate_passed") is True and self.score is not None

    @property
    def score(self) -> float | None:
        """The mean case speedup the lane's stop check measured, when it measured one."""
        value = self.sink.get("mean_case_speedup")
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    @property
    def contended(self) -> bool:
        """Whether this lane's teardown left the device unsafe to measure on."""
        return self.reaped is not None and self.reaped.contended


async def _run(*command: str) -> str:
    """Run one filesystem command, failing with its own words."""
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"`{' '.join(command[:2])}` failed: {stderr.decode(errors='replace')[-400:]}")
    return stdout.decode(errors="replace")


async def _tree_bytes(tree: Path) -> int:
    """Disk footprint of one tree."""
    fields = (await _run("du", "-sB1", str(tree))).split(maxsplit=1)
    if not fields or not fields[0].isdigit():
        raise RuntimeError(f"could not size {tree}: unreadable du output")
    return int(fields[0])


async def _require_room(*, source: Path, skip: Path, parent: Path, lane_count: int) -> None:
    """Refuse the round unless every lane copy fits where it is going."""
    needed = await _tree_bytes(source)
    if skip.exists():
        needed = max(0, needed - await _tree_bytes(skip))
    needed *= lane_count
    available = shutil.disk_usage(parent).free
    if available < needed:
        raise RuntimeError(
            f"not enough room for {lane_count} lane copies of {source} under {parent}: {needed} B needed "
            f"({needed / 1024**3:.1f} GiB), {available} B available ({available / 1024**3:.1f} GiB)"
        )


async def _copy_tree(source: Path, destination: Path, skip: Path) -> None:
    """Clone a workspace, untracked build state included, leaving out ``skip``.

    ``skip`` is the Analysis evidence: it is the bulk of a long campaign's workspace, a session reads it by the
    absolute path its prompt names, and no lane writes it.
    """
    whole = []
    for entry in source.iterdir():
        if entry == skip:
            continue
        if entry in skip.parents:
            inner = destination / entry.name
            inner.mkdir()
            await _copy_tree(entry, inner, skip)
        else:
            whole.append(str(entry))
    if whole:
        await _run("cp", "-a", *whole, str(destination))


async def _git(*args: str, cwd: Path) -> str:
    """Run one git command in a lane copy, failing loudly with git's own words."""
    return (await git_async(*args, cwd=cwd)).stdout.strip()


async def _isolate_lane_repository(lane_dir: Path) -> None:
    """Give a lane copy its own git index when the workspace is not a plain repo."""
    marker = lane_dir / ".git"
    if not marker.is_file():
        return
    head = await _git("rev-parse", "HEAD", cwd=lane_dir)
    # A detached HEAD is a normal answer here and reported as a non-zero exit.
    branch = (await git_async("symbolic-ref", "--quiet", "HEAD", cwd=lane_dir, check=False)).stdout.strip()
    objects = Path(await _git("rev-parse", _GIT_PATH, "objects", cwd=lane_dir))
    if not objects.is_absolute():
        objects = lane_dir / objects
    if not objects.is_dir():
        raise RuntimeError(f"lane isolation found no object store for {lane_dir}: {objects}")
    marker.unlink()
    await _git("init", "--quiet", cwd=lane_dir)
    alternates = lane_dir / ".git" / "objects" / "info" / "alternates"
    alternates.parent.mkdir(parents=True, exist_ok=True)
    alternates.write_text(f"{objects}\n")
    if branch:
        await _git("symbolic-ref", "HEAD", branch, cwd=lane_dir)
    else:
        await _git("update-ref", "--no-deref", "HEAD", head, cwd=lane_dir)
    # Rebuild the lane's index from the commit the canonical workspace is on, leaving the copied working tree exactly
    # as cp left it.
    await _git("reset", "--quiet", "--mixed", head, cwd=lane_dir)


async def _install_device_runner(seat: LaneSeat) -> None:
    """Put the device runner and its configuration in one lane, invisible to the lane's git."""
    lane_dir = seat.lane_dir
    shutil.copyfile(device_queue.__file__, seat.device_runner)
    device_queue.write_runner_config(
        lane_dir / device_queue.RUNNER_CONFIG_NAME,
        device_queue.runner_config(
            queue_root=seat.queue_root,
            usage_log=seat.usage_log,
            lane=seat.assignment.lane_id,
            hold_limit_sec=MAX_COMMAND_MINUTES * 60,
            environment=device_queue.visible_device_environment(),
        ),
    )
    exclude = Path(await _git("rev-parse", _GIT_PATH, "info/exclude", cwd=lane_dir))
    if not exclude.is_absolute():
        exclude = lane_dir / exclude
    exclude.parent.mkdir(parents=True, exist_ok=True)
    with exclude.open("a", encoding="utf-8") as handle:
        handle.write(f"/{device_queue.RUNNER_NAME}\n/{device_queue.RUNNER_CONFIG_NAME}\n")


def lane_diff(lane_dir: Path, new_paths: Sequence[str] = ()) -> str:
    """The lane's edits against the round's base: tracked changes, and the new files in ``new_paths``.

    The new files are added to a scratch copy of the index, so the lane's own index is left as the session had it.
    """
    if not new_paths:
        return git("diff", "HEAD", "--", ".", cwd=lane_dir).stdout
    index = Path(git("rev-parse", _GIT_PATH, "index", cwd=lane_dir).stdout.strip())
    if not index.is_absolute():
        index = lane_dir / index
    with tempfile.TemporaryDirectory(prefix="forge-lane-index-") as scratch:
        scratch_index = Path(scratch) / "index"
        shutil.copy2(index, scratch_index)
        env = {"GIT_INDEX_FILE": str(scratch_index)}
        git("add", "--", *new_paths, cwd=lane_dir, env=env)
        return git("diff", "HEAD", "--", ".", cwd=lane_dir, env=env).stdout


class LaneRound:
    """The workspace copies of one round, kept until the round's sessions have written their records."""

    def __init__(self, *, workspace_dir: str, parent_dir: str, lanes: Sequence[LaneAssignment]) -> None:
        self.source = Path(workspace_dir).resolve()
        self.parent = Path(parent_dir).resolve()
        self.lanes = list(lanes)
        self.root: Path | None = None
        self.seats: list[LaneSeat] = []

    async def open(self) -> list[LaneSeat]:
        """Copy the workspace once per lane and give each copy its device runner."""
        # Before the first lane process exists: it is what makes a lane's orphaned benchmark still identifiable as
        # this campaign's when its shell is gone.
        install_child_subreaper()
        layout = ExperimentsLayout.of(self.source)
        await _require_room(source=self.source, skip=layout.analysis, parent=self.parent, lane_count=len(self.lanes))
        self.root = Path(tempfile.mkdtemp(prefix=LANE_ROOT_PREFIX, dir=str(self.parent)))
        # A workspace imported by its directory's name is copied under that name, into a directory laid out like the
        # one that holds it, so that what a lane imports by that name is the lane.
        import_name = import_name_of(self.source)
        queue_root = campaign_device_queue_path(self.source)
        self.seats = [
            LaneSeat(
                assignment=lane,
                lane_dir=(lane_dir := self.root / lane.lane_id / import_name),
                device_runner=lane_dir / device_queue.RUNNER_NAME,
                queue_root=queue_root,
                usage_log=layout.device_usage,
            )
            for lane in self.lanes
        ]
        for seat in self.seats:
            seat.lane_dir.mkdir(parents=True)
        await asyncio.gather(*(self._prepare(seat, layout.analysis) for seat in self.seats))
        if import_name:
            for seat in self.seats:
                link_beside(seat.lane_dir.parent, linked_beside(self.source.parent))
        return self.seats

    async def _prepare(self, seat: LaneSeat, skip: Path) -> None:
        await _copy_tree(self.source, seat.lane_dir, skip)
        await _isolate_lane_repository(seat.lane_dir)
        await _install_device_runner(seat)

    async def run(self, session: Callable[[LaneSeat], Awaitable[tuple[str, dict]]]) -> list[LaneResult]:
        """Run every lane's session concurrently; each result carries what the session and its teardown left."""
        return list(await asyncio.gather(*(self._run_lane(seat, session) for seat in self.seats)))

    async def _run_lane(self, seat: LaneSeat, session) -> LaneResult:
        result = LaneResult(assignment=seat.assignment, lane_dir=seat.lane_dir)
        try:
            try:
                result.rationale, result.sink = await session(seat)
            finally:
                # A session that raised is exactly the lane most likely to have left something running.
                result.reaped = await reap_processes_under(
                    seat.lane_dir, description=f"left running in lane {seat.lane_dir}"
                )
            if result.reaped.contended:
                raise RuntimeError(
                    f"lane workspace could not be cleared, so its candidate cannot be trusted: "
                    f"{result.reaped.describe()}"
                )
        except Exception as error:  # isolation boundary: one lane of a round
            log.warning("lane %s lost: %s", seat.assignment.lane_id, error, exc_info=True)
            result.error = f"{type(error).__name__}: {error}"
        return result

    def close(self) -> None:
        """Remove the round's copies."""
        if self.root is not None:
            shutil.rmtree(self.root, ignore_errors=True)
            self.root = None
