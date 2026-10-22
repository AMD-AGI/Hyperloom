# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A branch driven by a third-party agent that optimizes end to end by its own logic: an autonomous branch.

The preflight makes the agent ready (:mod:`agent_onboarding`) and keeps its manifest in the branch's session
directory, so a run after a crash resumes it without onboarding again. A run then watches the agent's keep signal and
reads each keep whole (:mod:`agent_import`) the moment it shows, since the agent goes on to its next candidate in the
same workspace. Keeps are submitted as claimless explore tests against the branch head, one at a time: a keep that
lands while an import is measured or judged waits, and a later keep replaces it, since each is cumulative. What the run
knows of the keeps is kept beside the manifest, so a keep waiting at a crash is submitted by the run after it. The
branch's judge judges the imports as it judges any branch's tests; nothing of its verdicts reaches the agent. An agent
that exits 0 finishes the run once what it kept last settled; one that exits otherwise is a crash the runtime restarts
it from. A session never ends idle: what the agent keeps, or does not, is its own business, bounded by the session's
stops.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from hyperloom.contracts.measurement import Purpose, TestKind, TestRequest
from hyperloom.contracts.recipe import LaunchConfig, RecipeDelta, RecipeError
from hyperloom_multibranch.contracts.branch_log import BranchLogEntry, BranchLogType
from hyperloom_multibranch.contracts.notebook.records import TestRecord
from hyperloom_multibranch.contracts.repo import ExperimentRepo
from hyperloom_multibranch.contracts.toolbox import Toolbox
from hyperloom_multibranch.specialist.agent_import import KeptChange, KeptChangeError, import_delta, read_kept_change
from hyperloom_multibranch.specialist.agent_manifest import AgentManifest, read_field
from hyperloom_multibranch.specialist.agent_onboarding import (
    AgentProcess,
    ENCODING,
    SOURCE_DIR,
    AgentSpec,
    OnboardingSetup,
    agent_env,
    make_ready,
    tail,
)
from hyperloom_multibranch.specialist.agent_session import BranchPlace
from hyperloom_multibranch.specialist.judging import NOTICE
from hyperloom_multibranch.specialist.runtime import PreflightFailed, SessionCrashed, SessionFinished, SessionStart
from hyperloom_multibranch.specialist.tools import sync_to_head
from hyperloom_multibranch.specialist.worktree import WorktreeError

log = logging.getLogger(__name__)

MANIFEST_FILE = "agent-manifest.json"
#: What a run knows of the agent's keeps (:class:`KeepWatch`), so the run after a crash goes on from it.
WATCH_FILE = "agent-watch.json"
AGENT_DIR = "agent"
#: How long a measured import its judge has not judged yet counts as being judged. The judge polls every few seconds
#: and follows a gain with its retest and then its commit; an import submitted meanwhile would be on a stale head.
SETTLE_SECONDS = 120.0


@dataclass(frozen=True)
class AutonomousSetup:
    """What an autonomous branch runs on: how its agent is made ready, the spec it is given (its root's server
    arguments and environment and its workspace are filled in at the preflight), what imports are submitted through
    and read against, and ``component``, the framework the agent edits."""

    onboarding: OnboardingSetup
    spec: AgentSpec
    toolbox: Toolbox
    repo: ExperimentRepo
    component: str
    poll_seconds: float = 5.0
    settle_seconds: float = SETTLE_SECONDS
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    clock: Callable[[], float] = time.time


@dataclass
class KeepWatch:
    """What a run knows of the agent's keeps: ``root``, the workspace's git commit every keep is read against (the
    agent may commit on top of it); ``seq``, the keeps seen; ``mark``, what the keep file last said of them; and
    ``pending``, the latest keep read and not submitted yet."""

    root: str
    seq: int = 0
    mark: int | str | None = None
    pending: KeptChange | None = None

    @classmethod
    def load(cls, path: Path) -> KeepWatch:
        document = json.loads(path.read_text(encoding=ENCODING))
        pending = document.pop("pending")
        return cls(**document, pending=None if pending is None else KeptChange(**pending))

    def save(self, path: Path) -> None:
        pending = None if self.pending is None else asdict(self.pending)
        document = {"root": self.root, "seq": self.seq, "mark": self.mark, "pending": pending}
        path.write_text(json.dumps(document), encoding=ENCODING)


class AutonomousSession:
    """One run of the agent of the branch ``start`` names, its source in ``start.task.agent``."""

    def __init__(self, start: SessionStart, setup: AutonomousSetup, place: BranchPlace) -> None:
        self._start = start
        self._setup = setup
        self._place = place
        self._agent_dir = place.workspace / AGENT_DIR
        self._manifest: AgentManifest | None = None
        self._process: AgentProcess | None = None
        self._root = LaunchConfig()
        self._stopped = asyncio.Event()

    async def preflight(self) -> None:
        await sync_to_head(self._setup.repo, self._start.branch, self._place.worktree)
        root = self._place.worktree.commit()
        launch = (await self._setup.repo.read_recipe(self._start.start_commit)).launch
        workspace = str(self._place.worktree.path)
        spec = replace(self._setup.spec, server_args=launch.server_args, envs=dict(launch.envs), workspace=workspace)
        if not spec.cards:
            raise PreflightFailed(["no GPU is left for the agent: every card the agents may lease is taken"])
        ready = await make_ready(self._start.task.agent, spec, self._agent_dir, self._setup.onboarding)
        self._manifest, self._process = ready.manifest, ready.process
        (self._place.workspace / MANIFEST_FILE).write_text(json.dumps(ready.manifest.to_document()), encoding=ENCODING)
        KeepWatch(root, mark=self._mark(ready.manifest)).save(self._place.workspace / WATCH_FILE)
        await self._note(f"the agent {ready.manifest.source} at {ready.manifest.commit} is running")

    async def run(self) -> None:
        manifest, process = await self._running()
        self._root = (await self._setup.repo.read_recipe(self._start.start_commit)).launch
        watch = KeepWatch.load(self._place.workspace / WATCH_FILE)
        while not self._stopped.is_set():
            ended = process.returncode is not None
            await self._observe(manifest, watch)
            if ended:
                await self._ended(manifest, process, watch)
            await self._submit(manifest, watch)
            await self._setup.sleep(self._setup.poll_seconds)

    async def set_k(self, k: int) -> None:
        """An autonomous branch measures one import at a time, whatever K the branch has."""

    async def stop(self) -> None:
        self._stopped.set()
        if self._process is not None:
            await self._process.kill()

    async def _running(self) -> tuple[AgentManifest, AgentProcess]:
        """The agent preflight started, or, in a run after a crash, the agent resumed by the stored manifest."""

        if self._manifest is not None and self._process is not None:
            return self._manifest, self._process
        document = json.loads((self._place.workspace / MANIFEST_FILE).read_text(encoding=ENCODING))
        manifest = AgentManifest.read(document)
        command = manifest.resume or manifest.launch
        sandbox = self._setup.onboarding.sandbox
        process = await sandbox.start(command, self._agent_dir / SOURCE_DIR, agent_env(self._agent_dir))
        self._manifest, self._process = manifest, process
        await self._note(f"the agent resumed with `{command}`")
        return manifest, process

    async def _ended(self, manifest: AgentManifest, process: AgentProcess, watch: KeepWatch) -> None:
        """End the run of an agent that exited: a crash when it failed, else once what it kept last settled."""

        if process.returncode != 0:
            raise SessionCrashed(f"the agent exited {process.returncode}: {tail(await process.output())}")
        if manifest.keep is None:
            await self._take(manifest, watch, watch.seq + 1)
        while not self._stopped.is_set() and (watch.pending is not None or await self._busy()):
            await self._submit(manifest, watch)
            await self._setup.sleep(self._setup.poll_seconds)
        await self._note("the agent finished its run")
        raise SessionFinished("the agent finished its run")

    async def _observe(self, manifest: AgentManifest, watch: KeepWatch) -> None:
        """Read the agent's state the moment its keep file shows a new keep: by the time an import can go, the agent
        may be measuring its next candidate in the same workspace."""

        mark = self._mark(manifest)
        if mark is None or mark == watch.mark:
            return
        seq = mark if isinstance(mark, int) else watch.seq + 1
        watch.mark = mark
        if seq > watch.seq:
            await self._take(manifest, watch, seq)
        watch.save(self._place.workspace / WATCH_FILE)

    async def _take(self, manifest: AgentManifest, watch: KeepWatch, seq: int) -> None:
        watch.seq = seq
        try:
            watch.pending = await asyncio.to_thread(self._read, manifest, watch.root, seq)
        except (KeptChangeError, WorktreeError) as error:
            await self._note(f"keep {seq} cannot be read and is skipped: {error}")
        watch.save(self._place.workspace / WATCH_FILE)

    async def _submit(self, manifest: AgentManifest, watch: KeepWatch) -> None:
        """Submit the pending keep once nothing of the branch is measured or being judged, against the head then."""

        kept = watch.pending
        if kept is None or await self._busy():
            return
        repo, branch = self._setup.repo, self._start.branch
        head = await repo.head(branch)
        try:
            delta = await import_delta(
                kept,
                root=await repo.read_recipe(self._start.start_commit),
                head=await repo.read_recipe(head),
                component=self._setup.component,
                store=repo.store_patch,
            )
        except RecipeError as error:
            delta = None
            await self._note(f"keep {kept.seq} is no recipe change and is skipped: {error}")
        else:
            await self._submitted(kept, head, delta)
        watch.pending = None
        watch.save(self._place.workspace / WATCH_FILE)

    async def _submitted(self, kept: KeptChange, head: str, delta: RecipeDelta | None) -> None:
        if delta is None:
            await self._note(f"keep {kept.seq} changes nothing on the branch head")
            return
        request = TestRequest(TestKind.TEST, Purpose.EXPLORE, self._start.branch, head, delta)
        request_id = await self._setup.toolbox.submit_test(request)
        await self._note(f"keep {kept.seq} imported as {request_id}: {kept.summary or 'no summary'}")

    def _mark(self, manifest: AgentManifest) -> int | str | None:
        """What the agent's keep file says of its keeps now; ``None`` when it has none or cannot be read mid-write."""

        if manifest.keep is None:
            return None
        path = self._agent_dir / SOURCE_DIR / manifest.keep.path
        try:
            return manifest.keep.mark(path.read_bytes())
        except (OSError, ValueError, KeyError, IndexError) as error:
            log.debug("the keep file %s is not readable yet: %s", path, error)
            return None

    def _read(self, manifest: AgentManifest, root: str, seq: int) -> KeptChange:
        """The agent's state now; an agent that keeps no config of its own runs with the root's."""

        patch = without_file(self._place.worktree.diff_from(root, self._setup.component), manifest)
        source, summary = manifest.config_source, self._summary(manifest)
        if source is None:
            return KeptChange(seq, patch, self._root.server_args, dict(self._root.envs), summary)
        return read_kept_change(seq, patch, _json(self._place.worktree.path / source.path), source, summary)

    def _summary(self, manifest: AgentManifest) -> str:
        if manifest.keep is None or not manifest.summary_field:
            return ""
        try:
            state = _json(self._agent_dir / SOURCE_DIR / manifest.keep.path)
            return str(read_field(state, manifest.summary_field))
        except (KeptChangeError, KeyError, IndexError, ValueError):
            return ""

    async def _busy(self) -> bool:
        """Whether a test of the branch is in flight, or measured and not judged yet within the settle time."""

        snapshot = await self._place.notebook.snapshot()
        settled_before = self._setup.clock() - self._setup.settle_seconds
        return any(
            record.request.branch == self._start.branch and (record.in_flight or _being_judged(record, settled_before))
            for record in snapshot.records
        )

    async def _note(self, content: str) -> None:
        entry = BranchLogEntry(self._start.branch, BranchLogType.RESULT, f"{NOTICE} {content}")
        await self._place.notebook.append_branch_log(entry)


def _being_judged(record: TestRecord, settled_before: float) -> bool:
    measured_at = record.measured_at if record.measured_at is not None else record.requested_at
    return record.result is not None and record.judgement is None and measured_at > settled_before


def without_file(patch: str, manifest: AgentManifest) -> str:
    """``patch`` without the sections of the file the agent keeps its config in, which the config delta carries."""

    if manifest.config_source is None:
        return patch
    path = manifest.config_source.path
    sections = patch.split("diff --git ")
    kept = [section for section in sections[1:] if section.split("\n", 1)[0] != f"a/{path} b/{path}"]
    return "".join([sections[0], *(f"diff --git {section}" for section in kept)])


def _json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding=ENCODING))
    except (OSError, ValueError) as error:
        raise KeptChangeError(f"{path.name} cannot be read: {error}") from error


__all__ = [
    "AGENT_DIR",
    "MANIFEST_FILE",
    "SETTLE_SECONDS",
    "WATCH_FILE",
    "AutonomousSession",
    "AutonomousSetup",
    "KeepWatch",
    "without_file",
]
