# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run several Implementer lanes side by side, one isolated workspace each."""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from kernelforge.llm.git import git, git_async

from kernelforge.llm.process_reaping import (
    ReapReport,
    install_child_subreaper,
    reap_processes_under,
)


log = logging.getLogger(__name__)

SERIALIZED_DRIVER_NAME = "forge_lane_driver.py"
DEVICE_LOCK_SENTINEL = ".forge-device-bench.lock"


def campaign_device_lock_path(workspace: str | Path) -> Path:
    """The sentinel the fan-out lanes and the analysis-phase probes lock."""
    workspace = Path(workspace).expanduser().resolve()
    return workspace.parent / f"{workspace.name}{DEVICE_LOCK_SENTINEL}"


@dataclass(frozen=True)
class LanePlan:
    """One lane's assignment for this round."""

    lane_id: str
    plan: str


@dataclass(frozen=True)
class LaneResult:
    """What one lane produced, measured later and separately."""

    lane_id: str
    plan: str
    diff: str = ""
    error: str = ""
    # What this lane's teardown found, carried structured rather than folded
    # into ``error``. The two say different things to different readers: the
    # error is why THIS lane produced nothing, which costs the round one
    # candidate, while the report is about the device every lane and the
    # canonical measurement share, which costs the round its measurement. A
    # lane that failed for a reason of its own still has to report the second,
    # so it cannot be encoded in the first.
    reaped: ReapReport | None = None

    @property
    def produced_candidate(self) -> bool:
        return bool(self.diff.strip()) and not self.error

    @property
    def contended(self) -> bool:
        """Whether this lane's teardown left the device unsafe to measure on."""
        return self.reaped is not None and self.reaped.contended


_SERIALIZED_DRIVER = '''\
"""Run this lane's measurement driver while holding the shared device lock.

Generated for one fan-out round by kernelforge.loop.fanout. Every lane of the
round has one of these and they all lock the same campaign-wide sentinel -- the
one an analysis-phase probe locks too -- so the lanes think in parallel (the
long pole, measured in hours) while every run that touches the device queues
behind the others on it, measured in minutes.

This file only queues the run: it passes every argument to the driver unchanged,
writes nothing of its own to stdout, and returns the driver's exit status. Read
{driver_name} itself for what is measured and how.
"""

import fcntl
import subprocess
import sys

DRIVER = {driver!r}
SENTINEL = {sentinel!r}


def main() -> int:
    # One flock-able sentinel, held exclusively for the whole driver run. flock
    # is what reaches across processes: the agent CLI, the shell it ran this
    # from and this wrapper are all separate processes, so an in-process lock
    # would serialize nothing. The driver runs as a child in this process group
    # rather than in a session of its own, so a group-scoped teardown -- how a
    # shell enforces a command timeout -- reaches the driver as well, instead of
    # leaving it running once the kernel has dropped this process's lock.
    with open(SENTINEL, "r+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            # Say so rather than looking hung: the wait is another lane's whole
            # benchmark and there is no output until it finishes.
            sys.stderr.write(
                "[device] another lane holds the device; waiting for it\\n"
            )
            sys.stderr.flush()
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            completed = subprocess.run([sys.executable, DRIVER, *sys.argv[1:]])
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
'''


class DeviceBenchmarkLock:
    """The lock a lane's driver run has to take before it touches the device."""

    def __init__(self, sentinel: str | Path) -> None:
        self.sentinel = Path(sentinel)
        self.sentinel.touch(exist_ok=True)

    async def install(self, *, lane_dir: Path, driver: Path) -> Path:
        """Write one lane's serialized driver, invisible to the lane's git."""
        wrapper = lane_dir / SERIALIZED_DRIVER_NAME
        wrapper.write_text(
            _SERIALIZED_DRIVER.format(
                driver=str(driver),
                driver_name=driver.name,
                sentinel=str(self.sentinel),
            )
        )
        exclude = Path(await _git("rev-parse", "--git-path", "info/exclude", cwd=lane_dir))
        if not exclude.is_absolute():
            exclude = lane_dir / exclude
        exclude.parent.mkdir(parents=True, exist_ok=True)
        with exclude.open("a", encoding="utf-8") as handle:
            handle.write(f"/{SERIALIZED_DRIVER_NAME}\n")
        return wrapper


async def _tree_bytes(source: Path) -> int:
    """Disk footprint of the tree every lane is given a full copy of."""
    process = await asyncio.create_subprocess_exec(
        "du",
        "-sB1",
        str(source),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"could not size workspace {source}: {stderr.decode(errors='replace')[-400:]}")
    fields = stdout.decode(errors="replace").split(maxsplit=1)
    if not fields or not fields[0].isdigit():
        raise RuntimeError(
            f"could not size workspace {source}: unreadable du output {stdout.decode(errors='replace')[:200]!r}"
        )
    return int(fields[0])


def _available_bytes(directory: Path) -> int:
    """Free space on the filesystem that will hold the lane copies."""
    return shutil.disk_usage(directory).free


async def _require_room(*, source: Path, parent: Path, lane_count: int) -> None:
    """Refuse the round unless every lane copy fits where it is going."""
    needed = await _tree_bytes(source) * lane_count
    available = _available_bytes(parent)
    if available < needed:
        raise RuntimeError(
            f"not enough room for {lane_count} lane copies of {source} under "
            f"{parent}: {needed} B needed "
            f"({needed / 1024**3:.1f} GiB), {available} B available "
            f"({available / 1024**3:.1f} GiB)"
        )


async def _copy_workspace(source: Path, destination: Path) -> None:
    """Clone a workspace including the untracked build state a bench needs."""
    process = await asyncio.create_subprocess_exec(
        "cp",
        "-a",
        f"{source}/.",
        str(destination),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    _stdout, stderr = await process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"could not clone workspace into {destination}: {stderr.decode(errors='replace')[-400:]}")


async def _git(*args: str, cwd: Path) -> str:
    """Run one git command in a lane copy, failing loudly with git's own words."""
    return (await git_async(*args, cwd=cwd)).stdout.strip()


async def _head_branch(lane_dir: Path) -> str:
    """The branch HEAD names, or empty when HEAD is detached."""
    # A detached HEAD is a normal answer here and reported as a non-zero exit.
    completed = await git_async("symbolic-ref", "--quiet", "HEAD", cwd=lane_dir, check=False)
    return completed.stdout.strip()


async def _isolate_lane_repository(lane_dir: Path) -> None:
    """Give a lane copy its own git index when the workspace is not a plain repo."""
    marker = lane_dir / ".git"
    if not marker.is_file():
        return
    head = await _git("rev-parse", "HEAD", cwd=lane_dir)
    branch = await _head_branch(lane_dir)
    objects = Path(await _git("rev-parse", "--git-path", "objects", cwd=lane_dir))
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


async def _clone_lane(source: Path, lane_dir: Path) -> None:
    """Give one lane a workspace copy it can edit without touching another."""
    await _copy_workspace(source, lane_dir)
    await _isolate_lane_repository(lane_dir)


def _lane_driver(*, source: Path, driver: str) -> Path:
    """The driver's path within a workspace, checked for a lane to be given it."""
    relative = Path(driver)
    resolved = (source / relative).resolve()
    if relative.is_absolute() or source not in resolved.parents or not resolved.is_file():
        raise RuntimeError(
            f"lanes cannot be given their own copy of the driver {driver!r}: it "
            f"is not a file inside the campaign workspace {source}"
        )
    return relative


async def _reap_lane_processes(lane_dir: Path) -> ReapReport:
    """Kill whatever is still running inside a lane whose session has ended."""
    return await reap_processes_under(lane_dir, description=f"left running in lane {lane_dir}")


def _tracked_diff(lane_dir: Path) -> str:
    """The lane's staged and unstaged edits, in the form the archive stores."""
    return git("diff", "HEAD", "--", ".", cwd=lane_dir).stdout


async def run_lanes(
    *,
    workspace_dir: str,
    lanes: Sequence[LanePlan],
    session: Callable[[LanePlan, Path, Path], Awaitable[None]],
    parent_dir: str,
    driver: str,
) -> list[LaneResult]:
    """Run every lane's session concurrently and return each lane's own diff."""
    if not lanes:
        return []
    # Before the first lane process exists: it is what makes a lane's orphaned benchmark still identifiable as this
    # campaign's when its shell is gone.
    install_child_subreaper()
    source = Path(workspace_dir).resolve()
    parent = Path(parent_dir).resolve()
    driver_relative = _lane_driver(source=source, driver=driver)
    await _require_room(source=source, parent=parent, lane_count=len(lanes))
    root = Path(tempfile.mkdtemp(prefix="forge-lanes-", dir=str(parent)))
    try:
        lane_dirs = [root / lane.lane_id for lane in lanes]
        for lane_dir in lane_dirs:
            lane_dir.mkdir(parents=True)
        await asyncio.gather(*(_clone_lane(source, lane_dir) for lane_dir in lane_dirs))
        # One sentinel for the whole campaign rather than for this round, so a lane queues behind an analysis-phase
        # probe as well as behind its siblings.
        lock = DeviceBenchmarkLock(campaign_device_lock_path(source))
        serialized_drivers = [
            await lock.install(lane_dir=lane_dir, driver=lane_dir / driver_relative) for lane_dir in lane_dirs
        ]

        async def _one(
            lane: LanePlan,
            lane_dir: Path,
            serialized_driver: Path,
        ) -> LaneResult:
            # Assigned before the guard and returned on both paths: the report is the round's, not this lane's, so it
            # has to survive a lane that failed for a reason of its own -- and a session that raised is exactly the
            # lane most likely to have left something running.
            reaped: ReapReport | None = None
            try:
                try:
                    await session(lane, lane_dir, serialized_driver)
                finally:
                    reaped = await _reap_lane_processes(lane_dir)
                # Outside the guard above on purpose: a lane whose session already failed has to report why it failed,
                # not what its teardown found afterwards.
                if reaped.contended:
                    raise RuntimeError(
                        f"lane workspace could not be cleared, so its candidate cannot be trusted: {reaped.describe()}"
                    )
                # Reading the diff belongs inside the same guard: a lane whose result cannot be read is lost for a
                # different reason, but it is just as lost, and reporting it as an empty diff would file a session
                # that cost hours as a deliberate no-op.
                diff = _tracked_diff(lane_dir)
            except Exception as error:  # noqa: BLE001 - reported as a lost lane
                return LaneResult(
                    lane_id=lane.lane_id,
                    plan=lane.plan,
                    error=f"{type(error).__name__}: {error}",
                    reaped=reaped,
                )
            return LaneResult(
                lane_id=lane.lane_id,
                plan=lane.plan,
                diff=diff,
                reaped=reaped,
            )

        return list(
            await asyncio.gather(
                *(
                    _one(lane, lane_dir, serialized_driver)
                    for lane, lane_dir, serialized_driver in zip(lanes, lane_dirs, serialized_drivers)
                )
            )
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)
