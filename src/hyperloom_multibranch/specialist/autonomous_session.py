# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A branch driven by a third-party agent that optimizes end to end by its own logic: an autonomous branch.

The preflight makes the agent ready (:mod:`agent_onboarding`) and keeps its manifest in the branch's session
directory, so a run after a crash resumes it without onboarding again. A run then watches the agent's keep signal.
Each keep is read whole (:mod:`agent_import`) and submitted as one claimless explore test against the branch head,
one at a time: a keep that lands while an import is measured waits, and a later keep replaces it, since each is
cumulative. The branch's judge judges the imports as it judges any branch's tests; nothing of its verdicts reaches the
agent. An agent that exits 0 ends the run; one that exits otherwise is a crash the runtime restarts it from. A session
never ends idle: what the agent keeps, or does not, is its own business, bounded by the session's stops.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from hyperloom.contracts.measurement import Purpose, TestKind, TestRequest
from hyperloom_multibranch.contracts.branch_log import BranchLogEntry, BranchLogType
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
    make_ready,
    tail,
)
from hyperloom_multibranch.specialist.agent_session import BranchPlace
from hyperloom_multibranch.specialist.judging import NOTICE
from hyperloom_multibranch.specialist.runtime import SessionCrashed, SessionStart
from hyperloom_multibranch.specialist.tools import sync_to_head

log = logging.getLogger(__name__)

MANIFEST_FILE = "agent-manifest.json"
AGENT_DIR = "agent"


@dataclass(frozen=True)
class AutonomousSetup:
    """What every autonomous branch of a session shares: how agents are made ready, the spec a branch's agent is given
    (from the branch's agent workspace), what imports are submitted through and read against, and ``component``, the
    framework the agent edits."""

    onboarding: OnboardingSetup
    spec: Callable[[Path], AgentSpec]
    toolbox: Toolbox
    repo: ExperimentRepo
    component: str
    poll_seconds: float = 5.0
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep


class AutonomousSession:
    """One run of the agent of the branch ``start`` names, its source in ``start.task.agent``."""

    def __init__(self, start: SessionStart, setup: AutonomousSetup, place: BranchPlace) -> None:
        self._start = start
        self._setup = setup
        self._place = place
        self._agent_dir = place.workspace / AGENT_DIR
        self._manifest: AgentManifest | None = None
        self._process: AgentProcess | None = None
        self._stopped = asyncio.Event()

    async def preflight(self) -> None:
        await sync_to_head(self._setup.repo, self._start.branch, self._place.worktree)
        spec = self._setup.spec(self._place.worktree.path)
        ready = await make_ready(self._start.task.agent, spec, self._agent_dir, self._setup.onboarding)
        self._manifest, self._process = ready.manifest, ready.process
        (self._place.workspace / MANIFEST_FILE).write_text(json.dumps(ready.manifest.to_document()), encoding=ENCODING)
        await self._note(f"the agent {ready.manifest.source} at {ready.manifest.commit} is running")

    async def run(self) -> None:
        manifest, process = await self._running()
        seen = self._keeps(manifest)
        pending = False
        while not self._stopped.is_set():
            if process.returncode is not None:
                await self._ended(manifest, process, pending or manifest.keep is None)
                return
            count = self._keeps(manifest)
            pending, seen = pending or count > seen, max(seen, count)
            if pending and not await self._in_flight():
                pending = not await self._import(manifest, seen)
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
        process = await self._setup.onboarding.sandbox.start(command, self._agent_dir / SOURCE_DIR, {})
        self._manifest, self._process = manifest, process
        await self._note(f"the agent resumed with `{command}`")
        return manifest, process

    async def _ended(self, manifest: AgentManifest, process: AgentProcess, collect: bool) -> None:
        if process.returncode != 0:
            raise SessionCrashed(f"the agent exited {process.returncode}: {tail(await process.output())}")
        if collect:
            while await self._in_flight():
                await self._setup.sleep(self._setup.poll_seconds)
            await self._import(manifest, self._keeps(manifest))
        await self._note("the agent finished its run")

    def _keeps(self, manifest: AgentManifest) -> int:
        """The keeps the agent's keep file records now; the last count when it cannot be read mid-write."""

        if manifest.keep is None:
            return 0
        path = self._agent_dir / SOURCE_DIR / manifest.keep.path
        try:
            return manifest.keep.count(json.loads(path.read_text(encoding=ENCODING)))
        except (OSError, ValueError, KeyError, IndexError) as error:
            log.debug("the keep file %s is not readable yet: %s", path, error)
            return 0

    async def _import(self, manifest: AgentManifest, seq: int) -> bool:
        """Submit the agent's state now against the branch head; true once submitted or found already on the head."""

        try:
            kept = await asyncio.to_thread(self._read, manifest, seq)
        except KeptChangeError as error:
            await self._note(f"keep {seq} was not imported: {error}")
            return False
        repo, branch = self._setup.repo, self._start.branch
        head = await repo.head(branch)
        delta = await import_delta(
            kept,
            root=await repo.read_recipe(self._start.start_commit),
            head=await repo.read_recipe(head),
            component=self._setup.component,
            store=repo.store_patch,
        )
        if delta is None:
            await self._note(f"keep {seq} changes nothing on the branch head")
            return True
        request = TestRequest(TestKind.TEST, Purpose.EXPLORE, branch, head, delta)
        request_id = await self._setup.toolbox.submit_test(request)
        await self._note(f"keep {seq} imported as {request_id}: {kept.summary or 'no summary'}")
        return True

    def _read(self, manifest: AgentManifest, seq: int) -> KeptChange:
        """The agent's state now; an agent that keeps no config of its own runs with the root's."""

        patch = without_file(self._place.worktree.diff(self._setup.component), manifest)
        source, summary = manifest.config_source, self._summary(manifest)
        if source is None:
            spec = self._setup.spec(self._place.worktree.path)
            return KeptChange(seq, patch, spec.server_args, dict(spec.envs), summary)
        return read_kept_change(seq, patch, _json(self._place.worktree.path / source.path), source, summary)

    def _summary(self, manifest: AgentManifest) -> str:
        if manifest.keep is None or not manifest.summary_field:
            return ""
        try:
            state = _json(self._agent_dir / SOURCE_DIR / manifest.keep.path)
            return str(read_field(state, manifest.summary_field))
        except (KeptChangeError, KeyError, IndexError, ValueError):
            return ""

    async def _in_flight(self) -> bool:
        snapshot = await self._place.notebook.snapshot()
        return any(record.request.branch == self._start.branch and record.in_flight for record in snapshot.records)

    async def _note(self, content: str) -> None:
        entry = BranchLogEntry(self._start.branch, BranchLogType.RESULT, f"{NOTICE} {content}")
        await self._place.notebook.append_branch_log(entry)


def without_file(patch: str, manifest: AgentManifest) -> str:
    """``patch`` without the sections of the file the agent keeps its config in, which the config delta carries."""

    if manifest.config_source is None:
        return patch
    sections = patch.split("diff --git ")
    kept = [section for section in sections[1:] if manifest.config_source.path not in section.split("\n", 1)[0]]
    return "".join([sections[0], *(f"diff --git {section}" for section in kept)])


def _json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding=ENCODING))
    except (OSError, ValueError) as error:
        raise KeptChangeError(f"{path.name} cannot be read: {error}") from error


__all__ = ["AGENT_DIR", "MANIFEST_FILE", "AutonomousSession", "AutonomousSetup", "without_file"]
