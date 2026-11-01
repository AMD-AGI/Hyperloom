# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A branch driven by a third-party agent that optimizes end to end by its own logic: an autonomous branch.

The preflight leases the branch's sandbox, its own cards, from the toolbox, makes the agent ready in it
(:mod:`agent_onboarding`) and keeps the lease and the manifest in the branch's session directory, so a run after a crash
resumes the agent in the same sandbox without onboarding again, or, when that sandbox ended (a hosted one is gone once
its lifetime is over), in the sandbox of a new lease with the agent's install run there again. The agent starts with the
launch of the branch's start commit, its kernel tables laid out under the agent's directory as a test loads them
(:mod:`start_tables`), and a keep that leaves them as given changes no table. The preflight records
the manifest in the blackboard for CLOSE's archive, and the lease goes back when the branch stops. A lease asks for the
time left until the branch's deadline. A run then
watches the agent's keep signal and reads each keep whole (:mod:`agent_import`) the moment it shows, since the agent
goes on to its next candidate in the same workspace. Keeps are submitted as claimless explore tests against the branch
head, one at a time: a keep that lands while an import is measured or judged waits, and a later keep replaces it, since
each is cumulative. An agent that signals no keep has its workspace taken as its keep once, early enough before EXPLORE
ends for the session's queued tests and then its own to be measured and judged in EXPLORE (:func:`collection_margin`),
and again when it exits 0. What the run knows of the keeps is kept beside the manifest, so a keep waiting at a crash is
submitted by the run after it. The agent's own LLM spend is read from its Claude CLI's transcripts (:mod:`agent_spend`)
every :data:`SPEND_SECONDS` and once more when the run ends, and recorded as the branch's. At each keep, the findings
the agent wrote down in its ledger since the last one become the branch's learnings, a finding it revised superseding
what it said before, so other branches read them; what it wrote after its last keep is learnt when the branch stops. The
branch's judge judges the imports as it judges any branch's tests; nothing of its verdicts reaches the agent, but
what the integration head holds does, through the notes (:mod:`agent_knowledge`) rewritten whenever the head moves,
and the agent is given until its last import can still be judged in EXPLORE as its time budget. An agent that writes
a snapshot of each keep (Arbor's component mode) runs to that time as its absolute deadline, against the commit its
workspace starts on, and has its newest snapshot read whole, chained back to that commit, its launch only as it
differs from the one MB started it with (:func:`read_snapshot`); a variable the measurement sets itself is left out of
every keep and noted once; why it ended, from its exit file, goes into the branch log; one that refuses its config
(exit 2) ends the branch with why, from its refusal file, and is never started again. An agent
that exits 0 finishes the run once what it kept last settled; one that exits otherwise is a crash the runtime restarts
it from. A session never ends idle: what the agent keeps, or does not, is its own business, bounded by the session's
stops. A watch beside the run's poll makes the run a lost session when the running agent writes nothing (no output,
transcript or file), or the poll does not come round, for :data:`STALL_SECONDS`: the runtime then kills the agent and resumes it, and gives the branch
up after a few such losses in a row, so a hung agent does not hold its cards and its branch's slot.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import asdict, dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from hyperloom.contracts.measurement import Purpose, TestKind, TestRequest
from hyperloom.contracts.recipe import KernelArtifact, KernelKind, LaunchConfig, PatchRef, RecipeDelta, RecipeError
from hyperloom.measure.compile import COMPONENT_LAYOUTS, kernel_load_env, locate_installed_package
from hyperloom.sandbox import SandboxGone
from hyperloom_multibranch.contracts.branch_log import BranchLogEntry, BranchLogType
from hyperloom_multibranch.contracts.blackboard.records import Learning
from hyperloom_multibranch.contracts.blackboard.records import TestRecord
from hyperloom_multibranch.contracts.blackboard.vocab import LearningKind, LeaseKind
from hyperloom_multibranch.contracts.repo import ExperimentRepo
from hyperloom_multibranch.contracts.toolbox import Backpressure, BranchLease, LeaseCharge, RequestRejected
from hyperloom_multibranch.specialist.agent_import import (
    ComponentStart,
    KeptChange,
    KeptChangeError,
    added_at,
    KeepStores,
    import_delta,
    latest_snapshot,
    read_json,
    read_snapshot,
    with_tables,
    read_kept_change,
)
from hyperloom_multibranch.blackboard.local.blackboard import LEARNING_MAX_CHARS
from hyperloom_multibranch.specialist.agent_knowledge import NOTES_FILE, IntegrationNotes
from hyperloom_multibranch.specialist.agent_manifest import WORKSPACE, AgentManifest, LedgerSource, read_field
from hyperloom_multibranch.specialist.agent_onboarding import (
    AgentProcess,
    AgentSandbox,
    ENCODING,
    SOURCE_DIR,
    AgentSpec,
    AttemptFailed,
    OnboardingSetup,
    agent_env,
    install,
    make_ready,
    tail,
)
from hyperloom_multibranch.specialist.agent_session import BranchPlace
from hyperloom_multibranch.specialist.agent_spend import AgentSpend, newest_write
from hyperloom_multibranch.specialist.arbor_adapter import refused_reasons
from hyperloom_multibranch.specialist.judging import NOTICE
from hyperloom_multibranch.specialist.runtime import (
    PreflightFailed,
    SessionCrashed,
    SessionFinished,
    SessionRefused,
    SessionStart,
)
from hyperloom_multibranch.specialist.start_tables import (
    StartTable,
    StartTableError,
    from_start_tables,
    lay_start_tables,
    load_start_tables,
    save_start_tables,
)
from hyperloom_multibranch.specialist.tools import sync_to_head
from hyperloom_multibranch.specialist.worktree import Upstream, WorktreeError, installed_upstream
from hyperloom_multibranch.toolbox.kernel_artifacts import image_aiter_configs

if TYPE_CHECKING:
    from hyperloom_multibranch.toolbox.toolbox import RealToolbox

log = logging.getLogger(__name__)

MANIFEST_FILE = "agent-manifest.json"
LEASE_FILE = "agent-lease.json"
#: How often the agent's LLM spend is read and recorded while it runs: one record per CLI session a minute, not a poll.
SPEND_SECONDS = 60.0
#: The least time before the EXPLORE end an agent that signals no keep has its workspace collected, for a session that
#: has measured nothing yet to time a test by: an import's test and its confirm, about ten minutes each on an InferenceX
#: workload, after whatever test runs then (see :func:`collection_margin`).
CUTOFF_SECONDS = 40 * 60.0
SPEND_FILE = "agent-spend.json"
#: How long the running agent may write nothing (output, transcript or file), and the run's poll take to come round,
#: before the run is lost. A Claude CLI ends a tool call or a model request within ten minutes and writes its transcript.
STALL_SECONDS = 1800.0
#: What a run knows of the agent's keeps (:class:`KeepWatch`), so the run after a crash goes on from it.
WATCH_FILE = "agent-watch.json"
AGENT_DIR = "agent"
#: Below the agent's directory, where the start commit's kernel tables are laid out; what each variable is given for
#: them is kept beside the manifest.
START_TABLES_DIR = "start-tables"
START_TABLES_FILE = "agent-start-tables.json"
#: How long a measured import its judge has not judged yet counts as being judged. The judge polls every few seconds
#: and follows a gain with its retest and then its commit; an import submitted meanwhile would be on a stale head.
SETTLE_SECONDS = 120.0
#: The toolbox's refusal of a delta it cannot compile, among them a launch the measurement's launcher overrides.
LAUNCH_REFUSED = "delta_not_applicable"
#: The exit code of an agent run as a component that refused its config, saying why on stderr and in its refusal file.
REFUSED_CONFIG = 2
#: The deadline of an agent run as a component when EXPLORE has none, which it refuses to run without; MB stops it
#: long before.
NO_DEADLINE_SECONDS = 365 * 24 * 3600.0
#: The toolbox's refusal of a kernel table put with no target for a variable no table was put for yet.
TARGET_REQUIRED = "target_required"


def installed_component_upstream(component: str) -> Upstream | None:
    """The repository and commit the image's installed ``component`` was built from, ``None`` when it names none."""
    return installed_upstream(locate_installed_package(COMPONENT_LAYOUTS[component]))


@dataclass(frozen=True)
class AutonomousSetup:
    """What an autonomous branch runs on: how its agent is made ready in a sandbox, the sandbox a lease of the toolbox's
    cards gives, the spec it is given (its cards, the server arguments and environment of the commit it starts on and
    its workspace are filled in at the preflight), the toolbox the sandbox is leased from and imports are submitted to,
    the repository they are read against, ``component``, the framework the agent edits, ``spend``, the agent's own LLM
    spend for the directories it works under, read with the file of what was read, and ``aiter_configs``, the image's
    aiter tables the start commit's rows-only tables are laid over."""

    onboarding: Callable[[AgentSandbox], OnboardingSetup]
    sandbox: Callable[[BranchLease], AgentSandbox]
    spec: AgentSpec
    toolbox: RealToolbox
    repo: ExperimentRepo
    component: str
    spend: Callable[[tuple[Path, ...], Path], AgentSpend]
    poll_seconds: float = 5.0
    cutoff_seconds: float = CUTOFF_SECONDS
    upstream: Callable[[str], Upstream | None] = installed_component_upstream
    settle_seconds: float = SETTLE_SECONDS
    stall_seconds: float = STALL_SECONDS
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    clock: Callable[[], float] = time.time
    aiter_configs: Path = field(default_factory=image_aiter_configs)


@dataclass
class KeepWatch:
    """What a run knows of the agent's keeps: ``root``, the workspace's git commit every keep is read against (the agent
    may commit on top of it); ``seq``, the keeps seen; ``mark``, what the keep file last said of them; ``base``, the
    keeps seen before the agent last started its count over; ``pending``, the latest keep read and not submitted yet;
    ``collected``, whether the workspace of an agent that signals no keep was taken ahead of the EXPLORE end; and
    what of the agent's launch the toolbox refused, which every later keep that still runs with it is imported
    without: ``refused_envs``, each variable with the value refused (``None``: the variable left unset), and
    ``refused_args``, server arguments refused as a whole; ``learnt``, for each finding of the agent's ledger the branch
    learnt, the learning it became and a digest of what it said."""

    root: str
    seq: int = 0
    mark: int | str | None = None
    base: int = 0
    pending: KeptChange | None = None
    collected: bool = False
    refused_envs: dict[str, str | None] = field(default_factory=dict)
    refused_args: str | None = None
    learnt: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> KeepWatch:
        document = json.loads(path.read_text(encoding=ENCODING))
        pending = document.pop("pending")
        return cls(**document, pending=None if pending is None else KeptChange(**pending))

    def save(self, path: Path) -> None:
        pending = None if self.pending is None else asdict(self.pending)
        document = {
            "root": self.root,
            "seq": self.seq,
            "mark": self.mark,
            "base": self.base,
            "pending": pending,
            "collected": self.collected,
            "refused_envs": self.refused_envs,
            "refused_args": self.refused_args,
            "learnt": self.learnt,
        }
        path.write_text(json.dumps(document), encoding=ENCODING)


class AutonomousSession:
    """One run of the agent of the branch ``start`` names, its source in ``start.task.agent``."""

    def __init__(self, start: SessionStart, setup: AutonomousSetup, place: BranchPlace) -> None:
        self._start = start
        self._setup = setup
        self._place = place
        self._agent_dir = place.workspace / AGENT_DIR
        self._notes = IntegrationNotes(setup.repo, place.blackboard, self._agent_dir / NOTES_FILE)
        self._manifest: AgentManifest | None = None
        self._process: AgentProcess | None = None
        self._sandbox: AgentSandbox | None = None
        self._root = LaunchConfig()
        self._root_commit = ""
        self._start_tables: dict[str, StartTable] = {}
        self._unread = ""
        self._spend_unread = False
        self._spent_at = -math.inf
        self._polled_at = self._heard_at = 0.0
        self._heard = ""
        self._stopped = asyncio.Event()

    async def preflight(self) -> None:
        await sync_to_head(self._setup.repo, self._start.branch, self._place.worktree)
        root = self._place.worktree.commit()
        recipe = await self._setup.repo.read_recipe(self._start.start_commit)
        tables = await self._lay_start_tables(recipe.kernels)
        await self._notes.refresh()
        snapshot = await self._place.blackboard.snapshot()
        baseline = snapshot.baseline
        seconds = self._run_seconds()
        spec = replace(
            self._setup.spec,
            server_args=recipe.launch.server_args,
            envs={**recipe.launch.envs, **{name: table.value for name, table in tables.items()}},
            workspace=str(self._place.worktree.path),
            seconds=seconds,
            baseline=None if baseline is None else baseline.values.get(self._setup.spec.metric),
            knowledge=str(self._notes.path),
            deadline=self._setup.clock() + (seconds or NO_DEADLINE_SECONDS),
            root=root,
            root_shares=None if baseline is None else snapshot.op_shares.get(baseline.root_commit),
        )
        lease = await self._lease(spec.tp)
        spec = replace(spec, cards=lease.gpu_ids)
        self._sandbox = self._setup.sandbox(lease)
        ready = await make_ready(self._start.task.agent, spec, self._agent_dir, self._setup.onboarding(self._sandbox))
        self._manifest, self._process = ready.manifest, ready.process
        document = ready.manifest.to_document()
        (self._place.workspace / MANIFEST_FILE).write_text(json.dumps(document), encoding=ENCODING)
        await self._place.blackboard.record_agent_manifest(self._start.branch, document)
        # Every snapshot of the agent's session is one of this branch's keeps, the first among them possibly written
        # while the preflight waited for the agent to start.
        mark = None if ready.manifest.snapshots is not None else self._mark(ready.manifest)
        KeepWatch(root, mark=mark).save(self._place.workspace / WATCH_FILE)
        await self._note(f"the agent {ready.manifest.source} at {ready.manifest.commit} is running")

    async def _lay_start_tables(self, kernels: tuple[KernelArtifact, ...]) -> dict[str, StartTable]:
        directory, configs = self._agent_dir / START_TABLES_DIR, self._setup.aiter_configs
        try:
            tables = await lay_start_tables(self._setup.repo, self._start.start_commit, kernels, directory, configs)
        except StartTableError as error:
            raise PreflightFailed([str(error)]) from error
        self._place.workspace.mkdir(parents=True, exist_ok=True)
        save_start_tables(tables, self._place.workspace / START_TABLES_FILE)
        return tables

    def _run_seconds(self) -> float:
        """How long the agent may run: until the import of what it keeps last can still be measured and judged in
        EXPLORE, or, when the branch starts later than that, until EXPLORE ends; 0, no end, with no deadline."""

        deadline = self._start.task.deadline
        if not deadline:
            return 0.0
        left = deadline - self._setup.clock()
        return left - self._setup.cutoff_seconds if left > self._setup.cutoff_seconds else max(left, 1.0)

    async def run(self) -> None:
        manifest, process, sandbox = await self._running()
        self._unread = ""
        self._root = (await self._setup.repo.read_recipe(self._start.start_commit)).launch
        watch = KeepWatch.load(self._place.workspace / WATCH_FILE)
        self._root_commit = watch.root
        self._start_tables = load_start_tables(self._place.workspace / START_TABLES_FILE)
        spend = self._setup.spend((self._place.workspace.parent, self._held().root), self._place.workspace / SPEND_FILE)
        self._polled_at = self._heard_at = self._setup.clock()
        poll = asyncio.ensure_future(self._poll(manifest, process, sandbox, watch, spend))
        stall = asyncio.ensure_future(self._stall(process, partial(self._written_at, spend)))
        stalled = False
        try:
            done, _ = await asyncio.wait((poll, stall), return_when=asyncio.FIRST_COMPLETED)
            stalled = poll not in done
            (stall if stalled else poll).result()
        finally:
            poll.cancel()
            stall.cancel()
            await asyncio.gather(poll, stall, return_exceptions=True)
            # The poll may be stuck on the spend read itself; the next run reads what this one did not.
            if not stalled:
                await self._record_spend(spend, final=True)

    async def _poll(
        self, manifest: AgentManifest, process: AgentProcess, sandbox: AgentSandbox, watch: KeepWatch, spend: AgentSpend
    ) -> None:
        while not self._stopped.is_set():
            ended = process.returncode is not None
            await self._listen(process)
            await self._observe(manifest, watch)
            await self._notes.refresh()
            if said := sandbox.watched():
                await self._note(said)
            await self._collect_before_the_end(manifest, watch)
            await self._record_spend(spend, final=False)
            if ended:
                await self._ended(manifest, process, watch)
            await self._submit(manifest, watch)
            await self._pause()

    async def _pause(self) -> None:
        self._polled_at = self._setup.clock()
        await self._setup.sleep(self._setup.poll_seconds)

    async def _listen(self, process: AgentProcess) -> None:
        output = await process.output()
        if output != self._heard:
            self._heard, self._heard_at = output, self._setup.clock()

    async def _stall(self, process: AgentProcess, written_at: Callable[[], float | None]) -> None:
        """Raise a lost :class:`SessionCrashed` once the poll has not come round, or the running agent has written
        nothing, for :attr:`AutonomousSetup.stall_seconds`. Writing is its console, its CLI's transcripts or its
        files: an agent may print nothing for hours while it works (GEAK captures its workflow's output), so the
        transcripts and files are read once the console has been quiet that long."""

        bound = self._setup.stall_seconds
        while True:
            await self._setup.sleep(self._setup.poll_seconds)
            now = self._setup.clock()
            if now - self._polled_at > bound:
                raise SessionCrashed(f"the run has not polled the agent for {bound:g} s", lost=True)
            if process.returncode is None and now - self._heard_at > bound:
                written = await asyncio.to_thread(written_at)
                if written is not None and now - written <= bound:
                    self._heard_at = max(self._heard_at, written)
                    continue
                raise SessionCrashed(
                    f"the agent wrote no output, transcript or file for {bound:g} s: {tail(self._heard)}", lost=True
                )

    def _written_at(self, spend: AgentSpend) -> float | None:
        """When the agent last wrote a transcript or a file under its checkout or its workspace."""
        files = (
            path for root in (self._agent_dir, self._place.worktree.path) for path in root.rglob("*") if path.is_file()
        )
        return max((at for at in (spend.written_at(), newest_write(files)) if at is not None), default=None)

    async def _collect_before_the_end(self, manifest: AgentManifest, watch: KeepWatch) -> None:
        """Take, once, the workspace of an agent that signals no keep as its keep, :func:`collection_margin` before the
        EXPLORE end, so its work is measured and judged before EXPLORE ends."""

        deadline = self._start.task.deadline
        if manifest.signals_keeps or watch.collected or not deadline:
            return
        records = (await self._place.blackboard.snapshot()).records
        if self._setup.clock() < deadline - collection_margin(records, self._setup.cutoff_seconds):
            return
        watch.collected = True
        await self._note("EXPLORE ends soon: the agent's workspace is taken as its keep now")
        await self._take(manifest, watch, watch.seq + 1)

    async def _record_spend(self, spend: AgentSpend, *, final: bool) -> None:
        """Record the agent's new LLM spend, :data:`SPEND_SECONDS` after the last record or when the run ends; a read
        that fails is noted once and tried again at the next poll."""

        now = self._setup.clock()
        if not final and now - self._spent_at < SPEND_SECONDS:
            return
        self._spent_at = now
        try:
            await spend.record()
        except (OSError, ValueError, KeyError) as error:
            if not self._spend_unread:
                await self._note(f"the agent's LLM spend cannot be read, and is not recorded until it can: {error}")
            self._spend_unread = True
        else:
            self._spend_unread = False

    async def set_k(self, k: int) -> None:
        """An autonomous branch measures one import at a time, whatever K the branch has."""

    async def release(self) -> None:
        """Give the branch's sandbox lease back, when its preflight took one."""

        if (self._place.workspace / LEASE_FILE).exists():
            await self._setup.toolbox.release_branch_sandbox(self._held().lease_id)

    async def stop(self) -> None:
        """Stop the agent, then learn what its ledger gained since the last keep, which no later keep imports."""

        self._stopped.set()
        running = self._process is not None and self._process.returncode is None
        if self._process is not None:
            await self._process.kill()
        if running and self._manifest is not None and self._manifest.snapshots is not None:
            await self._note(f"the agent stopped: {self._exit_reason(self._manifest)}")
        watch_file = self._place.workspace / WATCH_FILE
        if self._manifest is not None and watch_file.exists():
            watch = KeepWatch.load(watch_file)
            await self._learn(self._manifest.ledger, watch)
            watch.save(watch_file)

    async def _running(self) -> tuple[AgentManifest, AgentProcess, AgentSandbox]:
        """The agent preflight started, or, in a run after a crash, the agent resumed by the stored manifest, and the
        sandbox it runs in."""

        if self._manifest is not None and self._process is not None and self._sandbox is not None:
            return self._manifest, self._process, self._sandbox
        document = json.loads((self._place.workspace / MANIFEST_FILE).read_text(encoding=ENCODING))
        manifest = AgentManifest.read(document)
        command = manifest.resume or manifest.launch
        checkout, env = self._agent_dir / SOURCE_DIR, agent_env(self._agent_dir)
        try:
            sandbox = self._setup.sandbox(self._held())
            process = await sandbox.start(command, checkout, env)
        except SandboxGone as gone:
            sandbox = await self._leased_again(manifest, gone)
            process = await sandbox.start(command, checkout, env)
        self._manifest, self._process, self._sandbox = manifest, process, sandbox
        await self._note(f"the agent resumed with `{command}`")
        return manifest, process, sandbox

    async def _leased_again(self, manifest: AgentManifest, gone: SandboxGone) -> AgentSandbox:
        """A sandbox of a new lease in place of the one that ended, with the agent's install run in it again; what the
        agent wrote is on this host and stays."""

        await self._note(f"the agent's sandbox ended ({gone}): the agent is installed in a new one and resumed")
        await self.release()
        sandbox = self._setup.sandbox(await self._lease(self._setup.spec.tp))
        checkout = self._agent_dir / SOURCE_DIR
        try:
            await install(manifest, self._setup.onboarding(sandbox), checkout, agent_env(self._agent_dir))
        except AttemptFailed as failed:
            raise SessionCrashed(f"the agent's install in its new sandbox failed: {failed}") from failed
        return sandbox

    async def _lease(self, cards: int) -> BranchLease:
        """Lease ``cards`` for the branch's sandbox, for as long as is left until the branch's deadline, and keep the
        lease for :meth:`release` and a later run."""

        # The lease is only given back through its file, so the file's directory must exist before the lease does.
        self._place.workspace.mkdir(parents=True, exist_ok=True)
        deadline = self._start.task.deadline
        try:
            lease = await self._setup.toolbox.lease_branch_sandbox(
                self._start.branch,
                cards,
                charge=LeaseCharge(LeaseKind.AUTONOMOUS, self._start.branch),
                lifetime_sec=max(deadline - self._setup.clock(), 1.0) if deadline else None,
            )
        except RequestRejected as error:
            raise PreflightFailed([f"the toolbox leased no cards for the agent's sandbox: {error}"]) from error
        document = {
            "lease_id": lease.lease_id,
            "branch_id": lease.branch_id,
            "gpu_ids": list(lease.gpu_ids),
            "root": str(lease.root),
            "env": dict(lease.env),
            "remaining_cards": lease.remaining_cards,
        }
        (self._place.workspace / LEASE_FILE).write_text(json.dumps(document), encoding=ENCODING)
        return lease

    def _held(self) -> BranchLease:
        document = json.loads((self._place.workspace / LEASE_FILE).read_text(encoding=ENCODING))
        return BranchLease(**{**document, "gpu_ids": tuple(document["gpu_ids"]), "root": Path(document["root"])})

    async def _ended(self, manifest: AgentManifest, process: AgentProcess, watch: KeepWatch) -> None:
        """End the run of an agent that exited: for good when it refused its config, a crash when it failed, else once
        what it kept last settled."""

        if process.returncode == REFUSED_CONFIG and manifest.snapshots is not None:
            why = refused_reasons(self._agent_dir / SOURCE_DIR / manifest.snapshots_refused)
            refused = f"the agent refused its config: {why or tail(await process.output())}"
            await self._note(refused)
            raise SessionRefused(refused)
        if process.returncode != 0:
            raise SessionCrashed(f"the agent exited {process.returncode}: {tail(await process.output())}")
        said = self._exit_reason(manifest) if manifest.snapshots is not None else ""
        if not manifest.signals_keeps:
            await self._take(manifest, watch, watch.seq + 1)
        await self._learn(manifest.ledger, watch)
        while not self._stopped.is_set() and (watch.pending is not None or await self._busy()):
            await self._submit(manifest, watch)
            await self._pause()
        finished = f"the agent finished its run: {said}" if said else "the agent finished its run"
        await self._note(finished)
        raise SessionFinished(finished)

    def _exit_reason(self, manifest: AgentManifest) -> str:
        """Why the agent run as a component says it ended, as the exit file it writes before an exit 0 has it."""

        path = self._agent_dir / SOURCE_DIR / manifest.snapshots_exit
        if not path.exists():
            return "it wrote no exit reason"
        try:
            document = read_json(path)
        except KeptChangeError as error:
            return f"its exit reason cannot be read: {error}"
        reason = document.get("reason") if isinstance(document, dict) else None
        return reason if isinstance(reason, str) and reason else f"its exit file names no reason: {document!r}"

    async def _observe(self, manifest: AgentManifest, watch: KeepWatch) -> None:
        """Read the agent's state the moment its keep file shows a new keep: by the time an import can go, the agent
        may be measuring its next candidate in the same workspace."""

        unread = self._unread
        mark = self._mark(manifest)
        if self._unread and self._unread != unread:
            await self._note(f"{self._unread}; no keep is counted until it does")
        if mark is None or mark == watch.mark:
            return
        if isinstance(mark, int) and isinstance(watch.mark, int) and mark < watch.mark and manifest.snapshots is None:
            watch.base = watch.seq
        seq = watch.base + mark if isinstance(mark, int) else watch.seq + 1
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
        else:
            if watch.pending.uncopied:
                names = ", ".join(watch.pending.uncopied)
                await self._note(
                    f"keep {seq}: the kernel tables of {names} were not copied, too large; imported as set"
                )
            await self._record_dropped(watch, watch.pending)
        await self._learn(manifest.ledger, watch)
        watch.save(self._place.workspace / WATCH_FILE)

    async def _record_dropped(self, watch: KeepWatch, kept: KeptChange) -> None:
        """Record in ``watch`` the variables of the agent's launch the measurement sets itself, noting each once."""

        new = {name: value for name, value in kept.dropped.items() if watch.refused_envs.get(name) != value}
        if new:
            watch.refused_envs.update(new)
            names = ", ".join(sorted(new))
            await self._note(f"keep {kept.seq} is imported without {names}: the measurement sets them, not the launch")

    async def _learn(self, ledger: LedgerSource | None, watch: KeepWatch) -> None:
        """Append each finding of the agent's ``ledger`` the branch has not learnt as it reads now as the branch's
        learning, superseding the one an earlier version of it became; one the blackboard refuses is noted once.

        The watch is saved after each finding: :meth:`stop` reads it from its file, and a finding learnt by a run that
        ended before saving it would be learnt again."""

        if ledger is None:
            return
        for finding in await asyncio.to_thread(read_findings, self._agent_dir / SOURCE_DIR / ledger.path, ledger):
            digest = hashlib.sha256(finding.content.encode()).hexdigest()[:16]
            earlier, said = watch.learnt.get(finding.id, ["", ""])
            if said == digest:
                continue
            learning = Learning(self._start.branch, finding.kind, finding.content, supersedes=earlier or None)
            try:
                earlier = (await self._place.blackboard.append_learning(learning)).entry_id
            except ValueError as error:
                await self._note(f"the agent's finding {finding.id} is not learnt: {error}")
            watch.learnt[finding.id] = [earlier, digest]
            watch.save(self._place.workspace / WATCH_FILE)

    async def _submit(self, manifest: AgentManifest, watch: KeepWatch) -> None:
        """Submit the pending keep once nothing of the branch is measured or being judged, against the head then.

        A keep whose launch the toolbox cannot compile (``delta_not_applicable``) goes again without the part of its
        launch that is refused (:meth:`_narrowed`): keeps are cumulative, so a launch entry the measurement refuses,
        such as a variable its launcher sets itself, would otherwise refuse every keep after it, and is left out of
        them from then on."""

        kept = watch.pending
        if kept is None or await self._busy():
            return
        head = await self._setup.repo.head(self._start.branch)
        launch = (await self._setup.repo.read_recipe(head)).launch
        kept = _without_refused(kept, launch, watch)
        try:
            await self._submitted(kept, head, await self._delta(kept, head))
        except Backpressure:
            return
        except RequestRejected as error:
            narrowed = await self._narrowed(kept, launch, head, watch) if error.code == LAUNCH_REFUSED else None
            if narrowed is None:
                await self._note(f"keep {kept.seq} was refused: {error.reason}")
            else:
                left_out = ", ".join([*watch.refused_envs, *(["its server arguments"] if watch.refused_args else [])])
                await self._note(f"keep {kept.seq} was refused: {error.reason}; imported without {left_out}")
                await self._submit_narrowed(narrowed, head)
        except RecipeError as error:
            await self._note(f"keep {kept.seq} is no recipe change and is skipped: {error}")
        except WorktreeError as error:
            await self._note(f"keep {kept.seq} cannot be read against the branch head and is skipped: {error}")
        watch.pending = None
        watch.save(self._place.workspace / WATCH_FILE)

    async def _submit_narrowed(self, kept: KeptChange, head: str) -> None:
        try:
            await self._submitted(kept, head, await self._delta(kept, head))
        except (RequestRejected, RecipeError) as error:
            await self._note(f"keep {kept.seq} was refused again: {error}")

    async def _narrowed(self, kept: KeptChange, launch: LaunchConfig, head: str, watch: KeepWatch) -> KeptChange | None:
        """``kept`` without the least of its launch the toolbox's acceptance takes it without, recorded in ``watch``:
        one changed variable, then every changed variable, then its server arguments, then both; ``None`` when even
        its source alone is refused. Each try is an acceptance check, which queues nothing."""

        changed = sorted(name for name in {*kept.envs, *launch.envs} if kept.envs.get(name) != launch.envs.get(name))
        args = () if kept.server_args == launch.server_args else (kept.server_args,)
        tries = [((name,), ()) for name in changed] + [(tuple(changed), ()), ((), args), (tuple(changed), args)]
        for names, refused_args in tries:
            if not names and not refused_args:
                continue
            candidate = _reverted(kept, launch, names, bool(refused_args))
            if await self._accepted(candidate, head):
                watch.refused_envs.update({name: kept.envs.get(name) for name in names})
                watch.refused_args = refused_args[0] if refused_args else watch.refused_args
                return candidate
        return None

    async def _accepted(self, kept: KeptChange, head: str) -> bool:
        try:
            delta = await self._delta(kept, head)
            if delta is None:
                return False
            await self._setup.toolbox.check_test(
                TestRequest(TestKind.TEST, Purpose.EXPLORE, self._start.branch, head, delta)
            )
        except (RequestRejected, RecipeError):
            return False
        return True

    async def _delta(self, kept: KeptChange, head: str) -> RecipeDelta | None:
        repo, component = self._setup.repo, self._setup.component
        recipe = await repo.read_recipe(head)
        root = await repo.read_recipe(self._start.start_commit)
        targets = {kernel_load_env(kernel): kernel.target for kernel in root.kernels}
        stores = KeepStores(repo.store_patch, partial(self._kernel_artifact, targets), partial(self._increment, head))
        delta = await import_delta(kept, root=root, head=recipe, component=component, stores=stores)
        if delta is None or all(item.component != component for item in delta.components):
            return delta
        if any(item.component == component for item in recipe.components):
            return delta
        upstream = await asyncio.to_thread(self._setup.upstream, component)
        return delta if upstream is None else added_at(delta, recipe, upstream)

    async def _increment(self, head: str, imported: tuple[PatchRef, ...], cumulative: str) -> str:
        """The patch from the state ``head``'s ``imported`` patches make to the one the keep's ``cumulative`` patch
        makes, both over the workspace's root."""

        texts = [await self._setup.repo.patch_text(head, ref.path) for ref in imported]
        worktree, component = self._place.worktree, self._setup.component
        return await asyncio.to_thread(worktree.diff_between, self._root_commit, texts, cumulative, component)

    async def _kernel_artifact(self, targets: dict[str, str], env: str, table: str, rows_only: bool) -> KernelArtifact:
        """The table ``env`` loads, kept in the session's blob store under the target the session bound ``env`` to, or,
        when no table was put for it yet, under the target of the start commit's table for it (``targets``), else
        ``env``'s own name: the service reads one file per variable, and a table on another target would be a second."""

        put = self._setup.toolbox.put_kernel_artifact
        try:
            return await put(KernelKind.GEMM, None, env, table.encode(), rows_only)
        except RequestRejected as error:
            if error.code != TARGET_REQUIRED:
                raise
            return await put(KernelKind.GEMM, targets.get(env, env.lower()), env, table.encode(), rows_only)

    async def _submitted(self, kept: KeptChange, head: str, delta: RecipeDelta | None) -> None:
        """Submit ``kept`` against ``head``; raise what the toolbox raises when it takes no request now."""

        if delta is None:
            await self._note(f"keep {kept.seq} changes nothing on the branch head")
            return
        request = TestRequest(TestKind.TEST, Purpose.EXPLORE, self._start.branch, head, delta)
        request_id = await self._setup.toolbox.submit_test(request)
        await self._note(f"keep {kept.seq} imported as {request_id}: {kept.summary or 'no summary'}")

    def _mark(self, manifest: AgentManifest) -> int | str | None:
        """What the agent's keep file says of its keeps now; ``None`` when it has none, cannot be read mid-write, or
        holds no counter the manifest names, which :attr:`_unread` then says."""

        if manifest.snapshots is not None:
            return self._latest(manifest.snapshots_outbox)
        if manifest.keep is None:
            return None
        path = self._agent_dir / SOURCE_DIR / manifest.keep.path
        try:
            mark = manifest.keep.mark(path.read_bytes())
        except json.JSONDecodeError as error:
            log.debug("the keep file %s is being written: %s", path, error)
            return None
        except OSError as error:
            self._unread = f"the keep file {manifest.keep.path} cannot be read ({error.strerror})"
            return None
        except (ValueError, KeyError, IndexError) as error:
            self._unread = (
                f"the keep file {manifest.keep.path} holds no keep count at {manifest.keep.counter!r} ({error!r})"
            )
            return None
        self._unread = ""
        return mark

    def _latest(self, outbox: str) -> int | None:
        """The keep the agent's newest snapshot holds; ``None`` before the first, or when ``LATEST`` cannot be read,
        which :attr:`_unread` then says."""

        try:
            latest = latest_snapshot(self._agent_dir / SOURCE_DIR / outbox)
        except OSError as error:
            self._unread = f"the snapshot outbox {outbox} cannot be read ({error.strerror})"
            return None
        except ValueError as error:
            self._unread = f"the LATEST of {outbox} names no keep ({error})"
            return None
        self._unread = ""
        return latest

    def _read(self, manifest: AgentManifest, root: str, seq: int) -> KeptChange:
        """The agent's keep ``seq``, from its snapshot, or its state now; an agent that keeps no config of its own runs
        with the root's."""

        if manifest.snapshots is not None:
            outbox = self._agent_dir / SOURCE_DIR / manifest.snapshots_outbox
            applied = partial(self._place.worktree.diff_applied, root, component=self._setup.component)
            copies = [path for table in self._start_tables.values() for path in table.defaults]
            kept = read_snapshot(outbox, seq, root, applied, ComponentStart(self._root, manifest.config, copies))
            return from_start_tables(kept, self._start_tables)
        workspace = self._place.worktree.path
        patch = without_file(self._place.worktree.diff_from(root, self._setup.component), manifest)
        source, summary = manifest.config_source, self._summary(manifest)
        if source is None:
            return KeptChange(seq, patch, self._root.server_args, dict(self._root.envs), summary)
        where = workspace if source.root == WORKSPACE else self._agent_dir / SOURCE_DIR
        kept = read_kept_change(seq, patch, read_json(where / source.path), source, summary, self._root)
        return with_tables(from_start_tables(kept, self._start_tables), (workspace, self._agent_dir))

    def _summary(self, manifest: AgentManifest) -> str:
        if manifest.keep is None or not manifest.summary_field:
            return ""
        try:
            state = read_json(self._agent_dir / SOURCE_DIR / manifest.keep.path)
            return str(read_field(state, manifest.summary_field))
        except (KeptChangeError, KeyError, IndexError, ValueError):
            return ""

    async def _busy(self) -> bool:
        """Whether a test of the branch is in flight, or measured and not judged yet within the settle time."""

        snapshot = await self._place.blackboard.snapshot()
        settled_before = self._setup.clock() - self._setup.settle_seconds
        return any(
            record.request.branch == self._start.branch and (record.in_flight or _being_judged(record, settled_before))
            for record in snapshot.records
        )

    async def _note(self, content: str) -> None:
        entry = BranchLogEntry(self._start.branch, BranchLogType.RESULT, f"{NOTICE} {content}")
        await self._place.blackboard.append_branch_log(entry)


@dataclass(frozen=True)
class Finding:
    """One finding of an agent's ledger, as the learning it becomes: ``content`` names the agent's kind and id."""

    id: str
    kind: LearningKind
    content: str


def read_findings(path: Path, ledger: LedgerSource) -> list[Finding]:
    """The findings of the kinds ``ledger`` maps that the ledger at ``path`` holds, each in its latest version, in the
    order they first appear; a line the agent is still writing is left for the next read."""

    try:
        lines = path.read_text(encoding=ENCODING).splitlines()
    except FileNotFoundError:
        return []
    latest: dict[str, Finding | None] = {}
    for line in lines:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict) or not isinstance(entry.get(ledger.id), str):
            continue
        kind, text = entry.get(ledger.kind), entry.get(ledger.text)
        content = " ".join(f"{kind} {entry[ledger.id]}: {text}".split())
        if len(content) > LEARNING_MAX_CHARS:
            content = content[: LEARNING_MAX_CHARS - 1] + "\u2026"
        fits = kind in ledger.kinds and isinstance(text, str)
        latest[entry[ledger.id]] = Finding(entry[ledger.id], ledger.kinds[kind], content) if fits else None
    return [finding for finding in latest.values() if finding is not None]


def _without_refused(kept: KeptChange, launch: LaunchConfig, watch: KeepWatch) -> KeptChange:
    """``kept`` with each launch entry the toolbox refused before, and that it still runs with, set back to ``launch``'s."""

    names = tuple(name for name, value in watch.refused_envs.items() if kept.envs.get(name) == value)
    return _reverted(kept, launch, names, kept.server_args == watch.refused_args)


def _reverted(kept: KeptChange, launch: LaunchConfig, names: tuple[str, ...], args: bool) -> KeptChange:
    """``kept`` with the variables ``names``, and its server arguments when ``args``, as ``launch`` has them."""

    envs = {name: value for name, value in kept.envs.items() if name not in names}
    envs.update({name: launch.envs[name] for name in names if name in launch.envs})
    return replace(kept, server_args=launch.server_args if args else kept.server_args, envs=envs)


def collection_margin(records: Iterable[TestRecord], floor: float) -> float:
    """How long before the EXPLORE end an import must go to be measured and judged in EXPLORE: the session's tests in
    flight, then the import's test and its confirm, one after another at the longest test measured so far, as on one
    card; at least ``floor``, which is all there is to go by before any test was measured."""

    records = list(records)
    durations = [
        attempt.finished_at - attempt.started_at
        for record in records
        if record.result is not None
        for attempt in record.result.attempts
        if attempt.started_at is not None and attempt.finished_at is not None
    ]
    if not durations:
        return floor
    queued = sum(record.in_flight for record in records)
    return max(floor, (queued + 2) * max(durations))


def _being_judged(record: TestRecord, settled_before: float) -> bool:
    measured_at = record.measured_at if record.measured_at is not None else record.requested_at
    return record.result is not None and record.judgement is None and measured_at > settled_before


def without_file(patch: str, manifest: AgentManifest) -> str:
    """``patch`` without the sections of the file the agent keeps its config in, which the config delta carries."""

    if manifest.config_source is None or manifest.config_source.root != WORKSPACE:
        return patch
    path = manifest.config_source.path
    sections = patch.split("diff --git ")
    kept = [section for section in sections[1:] if section.split("\n", 1)[0] != f"a/{path} b/{path}"]
    return "".join([sections[0], *(f"diff --git {section}" for section in kept)])


__all__ = [
    "AGENT_DIR",
    "LEASE_FILE",
    "collection_margin",
    "MANIFEST_FILE",
    "SETTLE_SECONDS",
    "STALL_SECONDS",
    "WATCH_FILE",
    "AutonomousSession",
    "AutonomousSetup",
    "Finding",
    "KeepWatch",
    "read_findings",
    "without_file",
]
