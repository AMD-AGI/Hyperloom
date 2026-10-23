# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A branch driven by a third-party agent that optimizes end to end by its own logic: an autonomous branch.

The preflight leases the branch's sandbox, its own cards, from the toolbox, makes the agent ready in it
(:mod:`agent_onboarding`) and keeps the lease and the manifest in the branch's session directory, so a run after a crash
resumes the agent in the same sandbox without onboarding again; the lease goes back when the branch stops. A run then
watches the agent's keep signal and reads each keep whole (:mod:`agent_import`) the moment it shows, since the agent
goes on to its next candidate in the same workspace. Keeps are submitted as claimless explore tests against the branch
head, one at a time: a keep that lands while an import is measured or judged waits, and a later keep replaces it, since
each is cumulative. What the run knows of the keeps is kept beside the manifest, so a keep waiting at a crash is
submitted by the run after it. The agent's own LLM spend is read from its Claude CLI's transcripts (:mod:`agent_spend`)
at each poll and once more when the run ends, and recorded as the branch's. The branch's judge judges the imports as it
judges any branch's tests; nothing of its verdicts reaches the agent. An agent that exits 0 finishes the run once what
it kept last settled; one that exits otherwise is a crash the runtime restarts it from. A session never ends idle: what
the agent keeps, or does not, is its own business, bounded by the session's stops.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from hyperloom.contracts.measurement import Purpose, TestKind, TestRequest
from hyperloom.contracts.recipe import KernelArtifact, KernelKind, LaunchConfig, RecipeDelta, RecipeError
from hyperloom.measure.compile import COMPONENT_LAYOUTS, locate_installed_package
from hyperloom_multibranch.contracts.branch_log import BranchLogEntry, BranchLogType
from hyperloom_multibranch.contracts.notebook.records import TestRecord
from hyperloom_multibranch.contracts.repo import ExperimentRepo
from hyperloom_multibranch.contracts.toolbox import Backpressure, BranchLease, RequestRejected
from hyperloom_multibranch.specialist.agent_import import (
    KeptChange,
    KeptChangeError,
    added_at,
    KeepStores,
    import_delta,
    with_tables,
    read_kept_change,
)
from hyperloom_multibranch.specialist.agent_manifest import AgentManifest, read_field
from hyperloom_multibranch.specialist.agent_onboarding import (
    AgentProcess,
    AgentSandbox,
    ENCODING,
    SOURCE_DIR,
    AgentSpec,
    OnboardingSetup,
    agent_env,
    make_ready,
    tail,
)
from hyperloom_multibranch.specialist.agent_session import BranchPlace
from hyperloom_multibranch.specialist.agent_spend import AgentSpend
from hyperloom_multibranch.specialist.judging import NOTICE
from hyperloom_multibranch.specialist.runtime import PreflightFailed, SessionCrashed, SessionFinished, SessionStart
from hyperloom_multibranch.specialist.tools import sync_to_head
from hyperloom_multibranch.specialist.worktree import Upstream, WorktreeError, installed_upstream

if TYPE_CHECKING:
    from hyperloom_multibranch.toolbox.toolbox import RealToolbox

log = logging.getLogger(__name__)

MANIFEST_FILE = "agent-manifest.json"
LEASE_FILE = "agent-lease.json"
SPEND_FILE = "agent-spend.json"
#: What a run knows of the agent's keeps (:class:`KeepWatch`), so the run after a crash goes on from it.
WATCH_FILE = "agent-watch.json"
AGENT_DIR = "agent"
#: How long a measured import its judge has not judged yet counts as being judged. The judge polls every few seconds
#: and follows a gain with its retest and then its commit; an import submitted meanwhile would be on a stale head.
SETTLE_SECONDS = 120.0
#: The toolbox's refusal of a delta it cannot compile, among them a launch the measurement's launcher overrides.
LAUNCH_REFUSED = "delta_not_applicable"
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
    the repository they are read against, ``component``, the framework the agent edits, and ``spend``, the agent's own LLM
    spend for the directories it works under, read with the file of what was read."""

    onboarding: Callable[[AgentSandbox], OnboardingSetup]
    sandbox: Callable[[BranchLease], AgentSandbox]
    spec: AgentSpec
    toolbox: RealToolbox
    repo: ExperimentRepo
    component: str
    spend: Callable[[tuple[Path, ...], Path], AgentSpend]
    poll_seconds: float = 5.0
    upstream: Callable[[str], Upstream | None] = installed_component_upstream
    settle_seconds: float = SETTLE_SECONDS
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    clock: Callable[[], float] = time.time


@dataclass
class KeepWatch:
    """What a run knows of the agent's keeps: ``root``, the workspace's git commit every keep is read against (the
    agent may commit on top of it); ``seq``, the keeps seen; ``mark``, what the keep file last said of them; ``base``,
    the keeps seen before the agent last started its count over; ``pending``, the latest keep read and not submitted
    yet; and what of the agent's launch the toolbox refused, which every later keep that still runs with it is imported
    without: ``refused_envs``, each variable with the value refused (``None``: the variable left unset), and
    ``refused_args``, server arguments refused as a whole."""

    root: str
    seq: int = 0
    mark: int | str | None = None
    base: int = 0
    pending: KeptChange | None = None
    refused_envs: dict[str, str | None] = field(default_factory=dict)
    refused_args: str | None = None

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
            "refused_envs": self.refused_envs,
            "refused_args": self.refused_args,
        }
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
        self._unread = ""
        self._spend_unread = False
        self._stopped = asyncio.Event()

    async def preflight(self) -> None:
        await sync_to_head(self._setup.repo, self._start.branch, self._place.worktree)
        root = self._place.worktree.commit()
        launch = (await self._setup.repo.read_recipe(self._start.start_commit)).launch
        workspace = str(self._place.worktree.path)
        spec = replace(self._setup.spec, server_args=launch.server_args, envs=dict(launch.envs), workspace=workspace)
        lease = await self._lease(spec.tp)
        spec = replace(spec, cards=lease.gpu_ids)
        onboarding = self._setup.onboarding(self._setup.sandbox(lease))
        ready = await make_ready(self._start.task.agent, spec, self._agent_dir, onboarding)
        self._manifest, self._process = ready.manifest, ready.process
        (self._place.workspace / MANIFEST_FILE).write_text(json.dumps(ready.manifest.to_document()), encoding=ENCODING)
        KeepWatch(root, mark=self._mark(ready.manifest)).save(self._place.workspace / WATCH_FILE)
        await self._note(f"the agent {ready.manifest.source} at {ready.manifest.commit} is running")

    async def run(self) -> None:
        manifest, process = await self._running()
        self._unread = ""
        self._root = (await self._setup.repo.read_recipe(self._start.start_commit)).launch
        watch = KeepWatch.load(self._place.workspace / WATCH_FILE)
        spend = self._setup.spend((self._place.workspace.parent, self._held().root), self._place.workspace / SPEND_FILE)
        try:
            while not self._stopped.is_set():
                ended = process.returncode is not None
                await self._observe(manifest, watch)
                await self._record_spend(spend)
                if ended:
                    await self._ended(manifest, process, watch)
                await self._submit(manifest, watch)
                await self._setup.sleep(self._setup.poll_seconds)
        finally:
            await self._record_spend(spend)

    async def _record_spend(self, spend: AgentSpend) -> None:
        """Record the agent's new LLM spend; a read that fails is noted once and tried again at the next poll."""

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
        sandbox = self._setup.sandbox(self._held())
        process = await sandbox.start(command, self._agent_dir / SOURCE_DIR, agent_env(self._agent_dir))
        self._manifest, self._process = manifest, process
        await self._note(f"the agent resumed with `{command}`")
        return manifest, process

    async def _lease(self, cards: int) -> BranchLease:
        """Lease ``cards`` for the branch's sandbox and keep the lease for :meth:`release` and a later run."""

        # The lease is only given back through its file, so the file's directory must exist before the lease does.
        self._place.workspace.mkdir(parents=True, exist_ok=True)
        try:
            lease = await self._setup.toolbox.lease_branch_sandbox(self._start.branch, cards)
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

        unread = self._unread
        mark = self._mark(manifest)
        if self._unread and self._unread != unread:
            await self._note(f"{self._unread}; no keep is counted until it does")
        if mark is None or mark == watch.mark:
            return
        if isinstance(mark, int) and isinstance(watch.mark, int) and mark < watch.mark:
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
        stores = KeepStores(repo.store_patch, self._kernel_artifact)
        delta = await import_delta(kept, root=root, head=recipe, component=component, stores=stores)
        if delta is None or all(item.component != component for item in delta.components):
            return delta
        if any(item.component == component for item in recipe.components):
            return delta
        upstream = await asyncio.to_thread(self._setup.upstream, component)
        return delta if upstream is None else added_at(delta, recipe, upstream)

    async def _kernel_artifact(self, env: str, table: str) -> KernelArtifact:
        """The table ``env`` loads, kept in the session's blob store under the target the session bound ``env`` to, or
        under ``env``'s own name when no table was put for it yet: the service reads one file per variable, and the
        toolbox refuses a second target for it."""

        put = self._setup.toolbox.put_kernel_artifact
        try:
            return await put(KernelKind.GEMM, None, env, table.encode())
        except RequestRejected as error:
            if error.code != TARGET_REQUIRED:
                raise
            return await put(KernelKind.GEMM, env.lower(), env, table.encode())

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

    def _read(self, manifest: AgentManifest, root: str, seq: int) -> KeptChange:
        """The agent's state now; an agent that keeps no config of its own runs with the root's."""

        patch = without_file(self._place.worktree.diff_from(root, self._setup.component), manifest)
        source, summary = manifest.config_source, self._summary(manifest)
        if source is None:
            return KeptChange(seq, patch, self._root.server_args, dict(self._root.envs), summary)
        kept = read_kept_change(seq, patch, _json(self._place.worktree.path / source.path), source, summary)
        return with_tables(kept, self._place.worktree.path)

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


def _without_refused(kept: KeptChange, launch: LaunchConfig, watch: KeepWatch) -> KeptChange:
    """``kept`` with each launch entry the toolbox refused before, and that it still runs with, set back to ``launch``'s."""

    names = tuple(name for name, value in watch.refused_envs.items() if kept.envs.get(name) == value)
    return _reverted(kept, launch, names, kept.server_args == watch.refused_args)


def _reverted(kept: KeptChange, launch: LaunchConfig, names: tuple[str, ...], args: bool) -> KeptChange:
    """``kept`` with the variables ``names``, and its server arguments when ``args``, as ``launch`` has them."""

    envs = {name: value for name, value in kept.envs.items() if name not in names}
    envs.update({name: launch.envs[name] for name in names if name in launch.envs})
    return replace(kept, server_args=launch.server_args if args else kept.server_args, envs=envs)


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
    "LEASE_FILE",
    "MANIFEST_FILE",
    "SETTLE_SECONDS",
    "WATCH_FILE",
    "AutonomousSession",
    "AutonomousSetup",
    "KeepWatch",
    "without_file",
]
