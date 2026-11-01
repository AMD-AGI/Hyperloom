# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import asyncio
from types import SimpleNamespace
import json
import os
import signal
import subprocess
import time
from collections.abc import Mapping
from dataclasses import asdict, replace, dataclass, field
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import TestRequest
from hyperloom.contracts.recipe import (
    ComponentPatches,
    ConfigDelta,
    KernelArtifact,
    KernelKind,
    LaunchConfig,
    Recipe,
    RecipeDelta,
    apply,
)
from hyperloom.measure.compile import LocalBlobStore, kernel_rows_only
from hyperloom.measure.surface import SurfaceViolation, check_launch_envs
from hyperloom.sandbox import SandboxGone
from hyperloom_multibranch.contracts.lifecycle import BranchTask
from hyperloom_multibranch.contracts.blackboard.vocab import LeaseKind
from hyperloom_multibranch.contracts.toolbox import Backpressure, BranchLease, LeaseCharge, RequestRejected
from hyperloom_multibranch.contracts.blackboard import INTEGRATION_BRANCH, Role, Writer
from hyperloom_multibranch.contracts.blackboard.records import BaselineRecord, ToolResultPointer
from hyperloom_multibranch.blackboard import LocalBlackboard
from hyperloom_multibranch.blackboard.local.blackboard import LEARNING_MAX_CHARS
from hyperloom_multibranch.blackboard.tests.factories import result
from hyperloom_multibranch.specialist.agent_onboarding import AgentSpec, CommandResult, OnboardingSetup
from hyperloom_multibranch.specialist.agent_session import BranchPlace
from hyperloom_multibranch.specialist.agent_manifest import AgentManifest
from hyperloom_multibranch.specialist.autonomous_session import (
    AGENT_DIR,
    MANIFEST_FILE,
    STALL_SECONDS,
    AutonomousSession,
    AutonomousSetup,
    collection_margin,
    without_file,
)
from hyperloom_multibranch.contracts.llm_prices import TokenCounts
from hyperloom_multibranch.specialist.agent_spend import AgentSpend, transcript_dir
from hyperloom_multibranch.specialist.judging import NOTICE
from hyperloom_multibranch.specialist.runtime import (
    PreflightFailed,
    SessionCrashed,
    SessionFinished,
    SessionRefused,
    SessionStart,
)
from hyperloom_multibranch.tests.branch_rig import branch_rig
from hyperloom_multibranch.toolbox.kernel_artifacts import KernelVariables

BRANCH = "a1"
ROOT = Recipe(
    "rocm/vllm@sha256:" + "c" * 64,
    LaunchConfig(server_args="--max-num-seqs 128"),
    components=(ComponentPatches("vllm", "https://github.com/vllm-project/vllm", "1" * 40),),
)

#: An agent that, every ``gap`` seconds, writes each of ``keeps`` to its workspace's config.py and to the config it
#: runs with, commits it when ``commits`` is set, as Arbor does, and records and prints the keep; it ends with ``exit``
#: after ``tail`` more seconds of silence. With ``table`` it also writes that kernel table, for the keep, to its workspace's tuned.csv,
#: or to ``table_at`` in its checkout. It reports ``envs`` as NAME=VALUE words in run/result.json, as GEAK does.
#: Before each keep it appends that keep's entries of ``ledger`` to its ledger, run/ledger.jsonl, as Arbor does.
AGENT = """
import json, pathlib, subprocess, sys, time
config = json.loads(pathlib.Path("campaign.json").read_text())
workspace = pathlib.Path(config["workspace"])
state = pathlib.Path("run/state.json")
state.parent.mkdir(exist_ok=True)
state.write_text(json.dumps({"keeps": [], "note": ""}))
for step, seqs in enumerate(config["keeps"], start=1):
    time.sleep(config["gap"])
    (workspace / "vllm" / "config.py").write_text(f"max_num_seqs = {seqs}\\n")
    (workspace / "agent-config.json").write_text(json.dumps({"args": config["args"].format(seqs=seqs), "envs": config["envs"]}))
    if config["table"]:
        table = pathlib.Path(config["table_at"]) if config["table_at"] else workspace / "tuned.csv"
        table.parent.mkdir(parents=True, exist_ok=True)
        table.write_text(config["table"].format(seqs=seqs))
    accepted = " ".join(f"{name}={value}" for name, value in config["envs"].items())
    pathlib.Path("run/result.json").write_text(json.dumps({"accepted_config": {"flags": "", "env": accepted}}))
    if config["commits"]:
        subprocess.run("git add -A && git -c user.name=a -c user.email=a@a commit -qm k", shell=True, cwd=workspace)
    with pathlib.Path("run/ledger.jsonl").open("a") as ledger:
        for entry in config["ledger"][step - 1] if step <= len(config["ledger"]) else []:
            ledger.write(json.dumps(entry) + "\\n")
    state.write_text(json.dumps({"keeps": [{"decision": "KEEP"}] * step, "note": f"seqs {seqs}"}))
    print(f"kept {seqs}", flush=True)
end = time.time() + config["tail"]
while time.time() < end:
    time.sleep(0.1)
    if config.get("busy"):
        pathlib.Path("run/progress").write_text(str(time.time()))
sys.exit(config["exit"])
"""
#: An agent in Arbor's component mode, its session in run/: every ``gap`` seconds it commits each of ``keeps`` to its
#: workspace, writing a file of its own at the first, and writes the keep's snapshot to its outbox, run/snaps as the
#: config MB wrote renames it: ``change.patch`` against its parent, ``served.patch`` against the root, the launch,
#: ``table`` as the kernel table AITER_CONFIG_GEMM_A8W8 loads after an image default table (listed as too large to copy
#: with ``huge``, with a wrong digest with ``bad_sha``), and an AITER JIT link outside its tree; with ``parts`` the
#: variable's given value instead, ``table`` its own rows appended to it or replacing its last part, every file of it
#: copied in Arbor's multi-part shape; LATEST names it at each
#: keep or, without ``latest_each``, at the last only. The keep ``skip`` gets no snapshot, as one that failed; ``stray`` says it was made
#: on a commit MB never gave it; ``orphan`` is made on the root, its earlier snapshot's commit gone, as ``notes`` say.
#: It removes the last run's exit file as it starts, and ends with ``exit`` after ``tail`` seconds, or 0 on SIGTERM,
#: writing ``reason`` to run/component_exit.json before an exit 0 only; exit 2 says why on stderr.
COMPONENT_AGENT = """
import hashlib, json, os, pathlib, signal, subprocess, sys, time
LISTED = "/sgl-workspace/aiter/aiter/configs/a8w8_tuned_gemm.csv:/arbor/workspace/tuned.csv"
config = json.loads(pathlib.Path("campaign.json").read_text())
plan, workspace, root = config["plan"], pathlib.Path(config["workspace"]), config["component"]["root"]
session = pathlib.Path("run")
outbox = session / config["component"].get("outbox", "keeps")
outbox.mkdir(parents=True, exist_ok=True)
(session / "component_exit.json").unlink(missing_ok=True)


def leave(reason, code):
    if code == 0 and reason:
        (session / "component_exit.json").write_text(json.dumps({"reason": reason, "keeps": 0, "ts": time.time()}))
    if code == 2:
        print("component mode: docker_container must be @host", file=sys.stderr, flush=True)
    sys.exit(code)


def git(*args):
    argv = ["git", "-c", "user.name=a", "-c", "user.email=a@a", *args]
    return subprocess.run(argv, cwd=workspace, check=True, capture_output=True, text=True).stdout


signal.signal(signal.SIGTERM, lambda *_: leave("sigterm", 0))
parent, previous = root, None
for number, seqs in enumerate(plan["keeps"], start=1):
    time.sleep(plan["gap"])
    (workspace / "vllm" / "config.py").write_text(f"max_num_seqs = {seqs}\\n")
    if number == 1:
        (workspace / "vllm" / "first.py").write_text("FIRST = 1\\n")
    git("add", "-A")
    git("commit", "-qm", f"keep {number}")
    commit = git("rev-parse", "HEAD").strip()
    if number == plan["skip"]:
        continue
    made_on = root if number == plan["orphan"] else parent
    built = outbox / f".{number:04d}.tmp"
    built.mkdir()
    (built / "change.patch").write_text(git("diff", "--binary", made_on, commit))
    (built / "served.patch").write_text(git("diff", "--binary", root, commit))
    keep = {"number": number, "previous": previous, "root": root, "parent": made_on, "commit": commit,
            "change": f"seqs {seqs}", "intent": "", "outside": 1, "tables": {}, "tables_outside_workspace": []}
    if number == plan["stray"]:
        keep["parent"] = "f" * 40
    if number == plan["orphan"] and plan["orphan_notes"]:
        keep["notes"] = f"the commit of keep {previous} is gone from the workspace; this change is against the root"
    envs = dict(plan["envs"])
    if plan["parts"]:
        name, own = "AITER_CONFIG_GEMM_A8W8", (session / "own-tuned.csv").resolve()
        own.write_text(plan["table"].format(seqs=seqs))
        given = envs[name].split(os.pathsep)
        parts = {"append": [*given, str(own)], "replace": [*given[:-1], str(own)], "given": given}[plan["parts"]]
        envs[name], files = os.pathsep.join(parts), []
        for k, part in enumerate(parts):
            if os.path.isfile(part):
                data, file = pathlib.Path(part).read_bytes(), f"tables/{name}/{k}-{os.path.basename(part)}"
                (built / file).parent.mkdir(parents=True, exist_ok=True)
                (built / file).write_bytes(data)
                files.append({"path": part, "file": file, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
        keep["tables"][name] = {"path": envs[name], "files": files}
    elif plan["table"]:
        name, table = "AITER_CONFIG_GEMM_A8W8", plan["table"].format(seqs=seqs).encode()
        envs[name] = LISTED
        digest = "0" * 64 if plan["bad_sha"] else hashlib.sha256(table).hexdigest()
        file = None if plan["huge"] else "tuned.csv"
        keep["tables"][name] = {"path": envs[name], "file": file, "sha256": digest, "bytes": len(table)}
        if not plan["huge"]:
            (built / "tables" / name).mkdir(parents=True)
            (built / "tables" / name / "tuned.csv").write_bytes(table)
    (built / "keep.json").write_text(json.dumps(keep))
    (built / "config.json").write_text(json.dumps({"server_flags": plan["flags"].format(seqs=seqs), "server_env": envs}))
    outside = [{"link": "vllm/module.so", "target": "/nonexistent/aiter/jit/module.so", "sha256": "0" * 64, "size": 1}]
    (built / "outside.json").write_text(json.dumps(outside))
    built.rename(outbox / f"{number:04d}")
    if plan["latest_each"] or number == len(plan["keeps"]):
        (outbox / ".LATEST.tmp").write_text(f"{number:04d}")
        os.replace(outbox / ".LATEST.tmp", outbox / "LATEST")
    parent, previous = commit, number
    print(f"kept {seqs}", flush=True)
time.sleep(plan["tail"])
leave(plan["reason"], plan["exit"])
"""
SNAPSHOTS = {"session": "run"}
KEEPS = {"path": "run/state.json", "field": "keeps", "where": {"decision": "KEEP"}}
LEDGER = {
    "path": "run/ledger.jsonl",
    "id": "id",
    "kind": "type",
    "text": "claim",
    "kinds": {"FACT": "insight", "DEAD_END": "pitfall"},
}
CONFIG_SOURCE = {"path": "agent-config.json", "server_args": "args", "envs": "envs"}


@dataclass(frozen=True)
class Agent:
    keeps: tuple[int, ...] = (256, 512)
    gap: float = 0.3
    tail: float = 0.3
    exit: int = 0
    commits: bool = False
    args: str = "--max-num-seqs {seqs}"
    envs: dict[str, str] = field(default_factory=dict)
    keep: object = field(default_factory=lambda: KEEPS)
    config_source: object = None
    resume: str = ""
    table: str = ""
    table_at: str = ""
    ledger: tuple[tuple[dict[str, str], ...], ...] = ()
    busy: bool = False
    install: tuple[str, ...] = ()


@dataclass(frozen=True)
class Component:
    """The plan of :data:`COMPONENT_AGENT`."""

    keeps: tuple[int, ...] = (256, 512)
    gap: float = 0.3
    tail: float = 0.3
    exit: int = 0
    reason: str = "done"
    stray: int = 0
    skip: int = 0
    orphan: int = 0
    orphan_notes: bool = True
    latest_each: bool = True
    envs: dict[str, str] = field(default_factory=dict)
    flags: str = "--max-num-seqs {seqs}"
    #: The launch the config MB wrote starts the agent with, as Arbor's ``server_flags`` and ``server_env``.
    started: dict[str, object] | None = None
    table: str = ""
    huge: bool = False
    bad_sha: bool = False
    #: ``append``, ``replace`` or ``given``: what the agent does to the value of AITER_CONFIG_GEMM_A8W8 it was given.
    parts: str = ""


async def _until(condition, what: str, seconds: float = 10.0) -> None:
    for _ in range(int(seconds / 0.05)):
        if condition():
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


class LocalProcess:
    def __init__(self, process: asyncio.subprocess.Process, log: Path) -> None:
        self._process, self._log = process, log

    @property
    def returncode(self) -> int | None:
        return self._process.returncode

    async def output(self) -> str:
        return self._log.read_text(encoding="utf-8", errors="replace")

    async def kill(self) -> None:
        """SIGTERM the process group, SIGKILL it after 5 s; return once none of it is left, as the sandbox does."""
        if self._process.returncode is not None:
            return
        os.killpg(self._process.pid, signal.SIGTERM)
        for _ in range(100):
            await asyncio.sleep(0.05)
            try:
                os.killpg(self._process.pid, 0)
            except ProcessLookupError:
                break
        else:
            os.killpg(self._process.pid, signal.SIGKILL)
        await self._process.wait()


class LocalSandbox:
    """Runs commands as local processes, each in a process group of its own."""

    def __init__(self, logs: Path) -> None:
        self._logs, self.started, self.envs, self.leases, self.prompts = logs, [], [], [], []
        self.ran: list[str] = []
        #: The leases whose sandbox ended.
        self.gone: set[str] = set()
        self.usage = _Usage()
        self.watch_said: list[str] = []

    def watched(self) -> str:
        return self.watch_said.pop(0) if self.watch_said else ""

    async def run(self, command: str, cwd: Path, env: Mapping[str, str], timeout: float) -> CommandResult:
        self.ran.append(command)
        done = subprocess.run(command, shell=True, cwd=cwd, env={**os.environ, **env}, capture_output=True, text=True)
        return CommandResult(done.returncode, done.stdout + done.stderr)

    async def start(self, command: str, cwd: Path, env: Mapping[str, str]) -> LocalProcess:
        self._logs.mkdir(parents=True, exist_ok=True)
        log = self._logs / f"agent-{len(self.started)}.log"
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd,
            env={**os.environ, **env},
            stdout=log.open("w"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.started.append(command)
        self.envs.append(dict(env))
        return LocalProcess(process, log)


class Toolbox:
    """Accepts each request and writes it to the blackboard, so it is in flight until the test settles it; ``refusals``
    are raised first, one a submit."""

    def __init__(self, blackboard: LocalBlackboard) -> None:
        self._blackboard, self.requests, self.refusals = blackboard, [], []
        self.refused_envs: set[str] = set()
        self.leases: list[tuple[str, int]] = []
        self.lifetimes: list[float | None] = []
        self.charges: list[LeaseCharge | None] = []
        self.released: list[str] = []
        self.cards_left = 1
        self.tables: dict[str, bytes] = {}
        self._variables = KernelVariables()

    async def put_kernel_artifact(
        self, kind: KernelKind, target: str | None, env: str, data: bytes, rows_only: bool = False
    ) -> KernelArtifact:
        artifact = self._variables.put(kind, target, env, data, rows_only)
        self.tables[artifact.sha256] = data
        return artifact

    async def lease_branch_sandbox(
        self,
        branch_id: str,
        cards: int,
        *,
        wait_sec: float | None = None,
        charge: LeaseCharge | None = None,
        lifetime_sec: float | None = None,
    ) -> BranchLease:
        if self.cards_left < cards:
            raise RequestRejected("leasing would leave no card to measure on", "lease_leaves_no_measurement")
        self.leases.append((branch_id, cards))
        self.lifetimes.append(lifetime_sec)
        self.charges.append(charge)
        root = self._blackboard.root.parent / "lease"
        return BranchLease(f"L{len(self.leases)}", branch_id, (3,), root, {"HIP_VISIBLE_DEVICES": "3"}, 1)

    async def release_branch_sandbox(self, lease_id: str) -> None:
        self.released.append(lease_id)

    async def check_test(self, request: TestRequest) -> None:
        try:
            check_launch_envs(request.delta.config.envs)
        except SurfaceViolation as error:
            raise RequestRejected(str(error), "surface_mismatch") from error
        if self.refused_envs & set(request.delta.config.envs):
            names = ", ".join(sorted(self.refused_envs))
            reason = f"the environment does not compile: the renderer dropped or changed launch variables {names}"
            raise RequestRejected(reason, "delta_not_applicable")

    async def submit_test(self, request: TestRequest) -> str:
        if self.refusals:
            raise self.refusals.pop(0)
        await self.check_test(request)
        accepted = TestRequest(**{**request.__dict__, "request_id": f"r{len(self.requests) + 1}"})
        self.requests.append(accepted)
        await self._blackboard.write_request(accepted)
        return accepted.request_id or ""


def _agent_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "agent-repo"
    repo.mkdir()
    (repo / "agent.py").write_text(AGENT, encoding="utf-8")
    (repo / "component.py").write_text(COMPONENT_AGENT, encoding="utf-8")
    for argv in (
        ["git", "init", "-q"],
        ["git", "add", "."],
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init"],
    ):
        subprocess.run(argv, cwd=repo, check=True)
    return repo


def _given(agent: Agent | Component, prompt: str) -> Agent | Component:
    """``agent`` with each ``{NAME}`` of its environment the value of ``NAME`` in the spec ``prompt`` gives it, as an
    agent's own config is derived from that spec."""

    spec = json.loads(prompt[prompt.index("{\n") : prompt.index("\n}\n") + 2])
    given = {name: value.format_map(spec["envs"]) for name, value in agent.envs.items()}
    if not isinstance(agent, Component) or agent.started is None:
        return replace(agent, envs=given)
    started = {
        **agent.started,
        "server_env": {k: v.format_map(spec["envs"]) for k, v in agent.started["server_env"].items()},
    }
    return replace(agent, envs=given, started=started)


def _manifest(workspace: str, agent: Agent | Component) -> str:
    if isinstance(agent, Component):
        return _component_manifest(workspace, agent)
    plan = {"keeps": list(agent.keeps), "gap": agent.gap, "tail": agent.tail, "exit": agent.exit, "busy": agent.busy}
    return json.dumps(
        {
            "source": "agent",
            "commit": "c",
            "install": list(agent.install),
            "config_file": "campaign.json",
            "config": {
                "workspace": workspace,
                "commits": agent.commits,
                "args": agent.args,
                "envs": agent.envs,
                "table": agent.table,
                "table_at": agent.table_at,
                "ledger": [list(entries) for entries in agent.ledger],
                **plan,
            },
            "launch": "python3 agent.py",
            "resume": agent.resume,
            "keep": agent.keep,
            "config_source": agent.config_source,
            "summary_field": "note",
            "needs": [],
            "ledger": LEDGER if agent.ledger else None,
        }
    )


def _component_manifest(workspace: str, agent: Component) -> str:
    return json.dumps(
        {
            "source": "arbor",
            "commit": "c",
            "install": [],
            "config_file": "campaign.json",
            "config": {
                "workspace": workspace,
                "plan": asdict(agent),
                "component": {"outbox": "snaps"},
                **(agent.started or {}),
            },
            "launch": "python3 component.py",
            "resume": "",
            "keep": None,
            "config_source": None,
            "summary_field": "",
            "needs": [],
            "ledger": None,
            "snapshots": SNAPSHOTS,
        }
    )


class _Usage:
    def __init__(self) -> None:
        self.recorded: list = []

    async def record_llm_usage(self, usage) -> None:
        self.recorded.append(usage)


@dataclass
class Rig:
    session: AutonomousSession
    sandbox: LocalSandbox
    toolbox: Toolbox
    toolbox_blackboard: LocalBlackboard
    repo: object
    blackboard: LocalBlackboard
    blobs: LocalBlobStore
    restart: object

    async def notices(self) -> list[str]:
        return [record.entry.content for record in await self.blackboard.branch_log(BRANCH)]

    def patch(self, request: TestRequest) -> str:
        [change] = request.delta.components
        return self.blobs.get(change.add_patches[0].sha256).decode()


async def _rig(
    tmp_path: Path,
    agent: Agent | Component = Agent(),
    *,
    poll: float = 0.05,
    clock=time.time,
    start: Recipe | None = None,
    deadline: float = 0.0,
    stall: float = STALL_SECONDS,
    spend: type[AgentSpend] = AgentSpend,
    start_seconds: float = 5.0,
    aiter: Path | None = None,
    source: Path | None = None,
) -> Rig:
    """A session of the agent on :data:`BRANCH`, opened at the root, or at ``start`` committed onto it the way PRELUDE
    replays a Recipe KB champion; the agent's source is ``source``, or a repository holding :data:`AGENT` and
    :data:`COMPONENT_AGENT`."""

    rig = await branch_rig(tmp_path, ROOT, BRANCH)
    start_commit = rig.root if start is None else await rig.repo.commit(BRANCH, rig.root, start, "champion")
    toolbox_blackboard = LocalBlackboard(tmp_path / "blackboard", Writer(Role.TOOLBOX, "toolbox"))
    worktree = rig.base.worktree(BRANCH, tmp_path / BRANCH / "agent-workspace")
    sandbox, toolbox = LocalSandbox(tmp_path / "logs"), Toolbox(toolbox_blackboard)

    async def onboard(checkout: Path, prompt: str) -> str:
        sandbox.prompts.append(prompt)
        return _manifest(str(worktree.path), _given(agent, prompt))

    def onboarding(agent_sandbox) -> OnboardingSetup:
        return OnboardingSetup(agent_sandbox, onboard, start_seconds=start_seconds, poll_seconds=0.05)

    def sandbox_for(lease: BranchLease) -> LocalSandbox:
        if lease.lease_id in sandbox.gone:
            raise SandboxGone(f"the sandbox of {lease.lease_id} ended")
        sandbox.leases.append(lease)
        return sandbox

    spec = AgentSpec("/models/m", "vllm", 1, "fp8", "mi355x", {}, "output_throughput", "", {}, "", (), 60.0, 31100)

    def spend_of(roots: tuple[Path, ...], ledger: Path) -> AgentSpend:
        return spend(tmp_path / "claude", roots, ledger, BRANCH, "claude-sonnet-5-5", sandbox.usage)

    setup = AutonomousSetup(
        onboarding,
        sandbox_for,
        spec,
        toolbox,
        rig.repo,
        "vllm",
        spend_of,
        poll_seconds=poll,
        settle_seconds=0.3,
        cutoff_seconds=0.5,
        stall_seconds=stall,
        clock=clock,
    )
    if aiter is not None:
        setup = replace(setup, aiter_configs=aiter)
    place = BranchPlace(worktree, rig.blackboard, tmp_path / BRANCH / "session")
    agent_source = str(source or _agent_repo(tmp_path))
    start = SessionStart(BRANCH, start_commit, 1, BranchTask(agent=agent_source, deadline=deadline))

    def restart() -> AutonomousSession:
        return AutonomousSession(start, setup, place)

    blobs = LocalBlobStore(tmp_path / "blobs")
    return Rig(restart(), sandbox, toolbox, toolbox_blackboard, rig.repo, rig.blackboard, blobs, restart)


@pytest.mark.parametrize("commits", [False, True], ids=["edits", "commits"])
async def test_an_agent_s_keeps_are_imported_one_at_a_time_against_the_branch_head(
    tmp_path: Path, commits: bool
) -> None:
    rig = await _rig(tmp_path, Agent(commits=commits))

    await rig.session.preflight()
    running = asyncio.create_task(rig.session.run())
    await _until(lambda: rig.toolbox.requests, "the first import")
    await asyncio.sleep(1.0)
    assert len(rig.toolbox.requests) == 1
    await rig.toolbox_blackboard.write_result(result("r1"))
    await _until(lambda: len(rig.toolbox.requests) == 2, "the second import")
    await rig.toolbox_blackboard.write_result(result("r2"))
    with pytest.raises(SessionFinished):
        await asyncio.wait_for(running, 10)

    first, last = rig.toolbox.requests
    assert (first.branch_id, first.parent_commit, first.intent_fingerprint) == (
        BRANCH, await rig.repo.head(BRANCH), None,
    )  # fmt: skip
    assert "+max_num_seqs = 256" in rig.patch(first)
    assert "+max_num_seqs = 512" in rig.patch(last) and "-max_num_seqs = 128" in rig.patch(last)
    notices = await rig.notices()
    assert any(note.startswith(f"{NOTICE} keep 2") and "seqs 512" in note for note in notices)
    assert notices[-1] == f"{NOTICE} the agent finished its run"


async def test_a_keep_after_one_the_branch_committed_imports_only_its_own_change_on_top(tmp_path: Path) -> None:
    rig = await _rig(tmp_path)

    await rig.session.preflight()
    running = asyncio.create_task(rig.session.run())
    await _until(lambda: rig.toolbox.requests, "the first import")
    first = rig.toolbox.requests[0]
    recipe = await rig.repo.read_recipe(first.parent_commit)
    committed = await rig.repo.commit(BRANCH, first.parent_commit, apply(recipe, first.delta), "keep 1")
    await rig.toolbox_blackboard.write_result(result("r1"))
    await _until(lambda: len(rig.toolbox.requests) == 2, "the second import")
    await rig.toolbox_blackboard.write_result(result("r2"))
    with pytest.raises(SessionFinished):
        await asyncio.wait_for(running, 10)

    last = rig.toolbox.requests[1]
    assert last.parent_commit == committed
    assert "-max_num_seqs = 256" in rig.patch(last) and "+max_num_seqs = 512" in rig.patch(last)
    assert "max_num_seqs = 128" not in rig.patch(last)
    [change] = last.delta.components
    assert change.remove_patches == ()


async def test_an_import_waits_while_the_last_one_is_measured_or_judged(tmp_path: Path) -> None:
    later = [0.0]
    rig = await _rig(tmp_path, Agent(tail=30.0), clock=lambda: time.time() + later[0])

    await rig.session.preflight()
    running = asyncio.create_task(rig.session.run())
    await _until(lambda: rig.toolbox.requests, "the first import")
    await asyncio.sleep(0.6)
    await rig.toolbox_blackboard.write_result(result("r1"))
    await asyncio.sleep(0.2)
    assert len(rig.toolbox.requests) == 1
    later[0] = 1.0
    await _until(lambda: len(rig.toolbox.requests) == 2, "the import once the first one is judged")

    await rig.session.stop()
    await asyncio.wait_for(running, 10)


async def test_an_agent_that_exits_non_zero_is_a_crash_with_its_last_output(tmp_path: Path) -> None:
    rig = await _rig(tmp_path, Agent(keeps=(), exit=3))

    await rig.session.preflight()

    with pytest.raises(SessionCrashed, match="exited 3"):
        await asyncio.wait_for(rig.session.run(), 10)
    await rig.session.stop()


async def test_an_agent_that_goes_silent_is_a_lost_session_once_it_wrote_nothing_for_the_stall_bound(
    tmp_path: Path,
) -> None:
    rig = await _rig(tmp_path, Agent(keeps=(256, 512, 768), gap=0.5, tail=30.0), stall=1.0)
    await rig.session.preflight()
    started = time.monotonic()

    with pytest.raises(SessionCrashed, match="no output") as crashed:
        await asyncio.wait_for(rig.session.run(), 10)
    await rig.session.stop()

    assert crashed.value.lost
    assert time.monotonic() - started > 2.0


async def test_an_agent_silent_on_its_console_while_it_writes_its_workspace_is_not_lost(tmp_path: Path) -> None:
    rig = await _rig(tmp_path, Agent(keeps=(256,), gap=0.1, tail=30.0, busy=True), stall=1.0)
    await rig.session.preflight()
    running = asyncio.create_task(rig.session.run())

    await asyncio.sleep(3.0)
    alive = not running.done()
    await rig.session.stop()
    await asyncio.gather(running, return_exceptions=True)

    assert alive, "the run ended while the agent kept writing its workspace"


class _HungSpend(AgentSpend):
    async def record(self) -> None:
        await asyncio.Event().wait()


async def test_a_run_stuck_in_one_of_its_steps_is_a_lost_session_while_the_agent_still_writes(tmp_path: Path) -> None:
    rig = await _rig(tmp_path, Agent(keeps=tuple(range(1, 30)), gap=0.1), stall=0.5, spend=_HungSpend)
    await rig.session.preflight()

    with pytest.raises(SessionCrashed, match="polled") as crashed:
        await asyncio.wait_for(rig.session.run(), 10)
    await rig.session.stop()

    assert crashed.value.lost


async def test_a_keep_waiting_at_a_crash_is_imported_by_the_resumed_run_with_the_docker_shim(tmp_path: Path) -> None:
    rig = await _rig(tmp_path, Agent(exit=3, resume="sleep 30"))

    await rig.session.preflight()
    with pytest.raises(SessionCrashed):
        await asyncio.wait_for(rig.session.run(), 10)
    await rig.session.stop()
    assert len(rig.toolbox.requests) == 1
    resumed = rig.restart()
    running = asyncio.create_task(resumed.run())
    await rig.toolbox_blackboard.write_result(result("r1"))
    await _until(lambda: len(rig.toolbox.requests) == 2, "the waiting keep")

    assert rig.sandbox.started[-1] == "sleep 30"
    assert rig.sandbox.envs[-1]["PATH"].startswith(str(tmp_path / BRANCH / "session" / AGENT_DIR / "bin"))
    assert "+max_num_seqs = 512" in rig.patch(rig.toolbox.requests[-1])
    await resumed.stop()
    await asyncio.wait_for(running, 10)


async def test_an_agent_whose_sandbox_ended_is_installed_in_the_sandbox_of_a_new_lease_and_resumed(
    tmp_path: Path,
) -> None:
    rig = await _rig(
        tmp_path, Agent(exit=3, resume="sleep 30", install=("true installed",)), deadline=time.time() + 600
    )

    await rig.session.preflight()
    with pytest.raises(SessionCrashed):
        await asyncio.wait_for(rig.session.run(), 10)
    await rig.session.stop()
    rig.sandbox.gone.add("L1")
    resumed = rig.restart()
    running = asyncio.create_task(resumed.run())
    await _until(lambda: rig.sandbox.started[-1] == "sleep 30", "the agent resumed")

    assert rig.toolbox.released == ["L1"] and len(rig.toolbox.leases) == 2
    assert rig.sandbox.ran == ["true installed", "true installed"]
    assert all(590 < lifetime <= 600 for lifetime in rig.toolbox.lifetimes)
    assert any("the agent's sandbox ended (the sandbox of L1 ended)" in notice for notice in await rig.notices())
    await resumed.stop()
    await asyncio.wait_for(running, 10)


async def test_a_keep_file_with_no_counter_is_one_keep_at_each_change(tmp_path: Path) -> None:
    rig = await _rig(tmp_path, Agent(keeps=(256,), keep={"path": "run/state.json"}, tail=30.0))

    await rig.session.preflight()
    running = asyncio.create_task(rig.session.run())
    await _until(lambda: rig.toolbox.requests, "the import")

    assert "+max_num_seqs = 256" in rig.patch(rig.toolbox.requests[0])
    await rig.session.stop()
    await asyncio.wait_for(running, 10)


@pytest.mark.parametrize(
    ("args", "source", "why"),
    [
        ("--max-num-seqs {seqs}", {**CONFIG_SOURCE, "server_args": "envs"}, "cannot be read"),
        ("--max-num-seqs {seqs} /models/m", CONFIG_SOURCE, "is no recipe change"),
    ],
    ids=["unreadable", "positional"],
)
async def test_a_keep_that_cannot_be_imported_is_noted_once_and_the_run_goes_on(
    tmp_path: Path, args: str, source: dict[str, str], why: str
) -> None:
    rig = await _rig(tmp_path, Agent(keeps=(256,), args=args, config_source=source, tail=0.6))

    await rig.session.preflight()
    with pytest.raises(SessionFinished):
        await asyncio.wait_for(rig.session.run(), 10)

    assert not rig.toolbox.requests
    assert len([note for note in await rig.notices() if f"keep 1 {why}" in note]) == 1


async def test_a_keep_made_just_before_the_agent_exits_is_imported(tmp_path: Path) -> None:
    rig = await _rig(tmp_path, Agent(keeps=(256,), gap=0.2, tail=0.0), poll=1.0)

    await rig.session.preflight()
    running = asyncio.create_task(rig.session.run())
    await _until(lambda: rig.toolbox.requests, "the last keep", seconds=5.0)
    await rig.toolbox_blackboard.write_result(result("r1"))

    with pytest.raises(SessionFinished):
        await asyncio.wait_for(running, 10)


def test_only_the_file_the_agent_keeps_its_config_in_leaves_the_source_patch() -> None:
    manifest = AgentManifest.read({**json.loads(_manifest("/ws", Agent())), "config_source": CONFIG_SOURCE})
    sections = [
        f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n-x\n+y\n"
        for path in ("agent-config.json", "vllm/agent-config.json.py")
    ]

    assert without_file("".join(sections), manifest) == sections[1]


async def test_an_agent_runs_on_a_lease_taken_once_for_its_branch_and_given_back_once(tmp_path: Path) -> None:
    rig = await _rig(tmp_path, Agent(exit=3, resume="sleep 30"))

    await rig.session.preflight()
    with pytest.raises(SessionCrashed):
        await asyncio.wait_for(rig.session.run(), 10)
    await rig.session.stop()
    resumed = rig.restart()
    running = asyncio.create_task(resumed.run())
    await _until(lambda: len(rig.sandbox.started) == 2, "the resumed agent")
    await resumed.stop()
    await asyncio.wait_for(running, 10)
    await resumed.release()
    await resumed.release()

    assert rig.toolbox.leases == [(BRANCH, 1)]
    assert rig.toolbox.charges == [LeaseCharge(LeaseKind.AUTONOMOUS, BRANCH)]
    assert [lease.gpu_ids for lease in rig.sandbox.leases] == [(3,), (3,)]
    assert rig.toolbox.released == ["L1", "L1"]


async def test_an_agent_opened_on_a_recipe_kb_champion_starts_from_its_launch(tmp_path: Path) -> None:
    champion = replace(
        ROOT, launch=LaunchConfig("--max-num-seqs 256 --kv-cache-dtype fp8_e4m3", {"SGLANG_USE_AITER": "1"})
    )
    rig = await _rig(tmp_path, start=champion)

    await rig.session.preflight()
    await rig.session.stop()

    [prompt] = rig.sandbox.prompts
    assert '"server_args": "--max-num-seqs 256 --kv-cache-dtype fp8_e4m3"' in prompt
    assert '"SGLANG_USE_AITER": "1"' in prompt


def _spent(transcript: Path, message_id: str, output: int) -> None:
    usage = {"input_tokens": 3, "output_tokens": output}
    message = {"id": message_id, "model": "claude-opus-5-5", "usage": usage, "content": []}
    with transcript.open("a") as handle:
        handle.write(json.dumps({"type": "assistant", "sessionId": "s-1", "message": message}) + "\n")


async def test_what_the_agent_s_own_cli_spends_is_recorded_as_the_branch_s_once_a_minute_and_at_the_end(
    tmp_path: Path,
) -> None:
    rig = await _rig(tmp_path, Agent(tail=30.0))
    await rig.session.preflight()
    running = asyncio.create_task(rig.session.run())
    transcript = transcript_dir(tmp_path / "claude", tmp_path / BRANCH / "session" / "agent" / "src") / "s-1.jsonl"
    transcript.parent.mkdir(parents=True)

    _spent(transcript, "m1", 50)
    await _until(lambda: rig.sandbox.usage.recorded, "the agent's spend")
    _spent(transcript, "m2", 7)
    await asyncio.sleep(0.5)
    assert len(rig.sandbox.usage.recorded) == 1
    await rig.session.stop()
    await asyncio.wait_for(running, 10)

    assert [(used.branch, used.launch, used.tokens) for used in rig.sandbox.usage.recorded] == [
        (BRANCH, "s-1", TokenCounts(3, 50)),
        (BRANCH, "s-1", TokenCounts(3, 7)),
    ]


async def test_an_agent_with_no_keep_signal_has_its_workspace_collected_once_before_the_explore_end(
    tmp_path: Path,
) -> None:
    rig = await _rig(tmp_path, Agent(keep=None, tail=30.0), deadline=time.time() + 1.5)
    await rig.session.preflight()
    running = asyncio.create_task(rig.session.run())

    await _until(lambda: rig.toolbox.requests, "the cut-off collection")
    await asyncio.sleep(1.5)
    await rig.session.stop()
    await asyncio.wait_for(running, 10)

    [request] = rig.toolbox.requests
    assert "+max_num_seqs = 512" in rig.patch(request)
    assert any("EXPLORE ends" in note for note in await rig.notices())


async def test_a_passed_preflight_records_the_agent_s_manifest_in_the_blackboard(tmp_path: Path) -> None:
    rig = await _rig(tmp_path)

    await rig.session.preflight()
    await rig.session.release()

    stored = json.loads((tmp_path / BRANCH / "session" / MANIFEST_FILE).read_text(encoding="utf-8"))
    recorded = (await rig.blackboard.snapshot()).agent_manifests[BRANCH].manifest
    assert recorded["source"] == stored["source"] and recorded["commit"] == stored["commit"]


async def test_an_agent_whose_lease_would_leave_no_card_to_measure_fails_its_preflight(tmp_path: Path) -> None:
    rig = await _rig(tmp_path)
    rig.toolbox.cards_left = 0

    with pytest.raises(PreflightFailed, match="lease_leaves_no_measurement"):
        await rig.session.preflight()
    await rig.session.release()

    assert not rig.sandbox.started and rig.toolbox.released == []


async def test_a_keep_the_toolbox_refuses_is_noted_and_one_it_has_no_room_for_waits(tmp_path: Path) -> None:
    rig = await _rig(tmp_path, Agent(tail=1.0))
    rig.toolbox.refusals = [
        RequestRejected("online tuning is not measured", code="online_tuning_unsupported"),
        Backpressure(BRANCH, 1, 1),
    ]

    await rig.session.preflight()
    running = asyncio.create_task(rig.session.run())
    await _until(lambda: rig.toolbox.requests, "the keep after the refused one")
    await rig.toolbox_blackboard.write_result(result("r1"))
    with pytest.raises(SessionFinished):
        await asyncio.wait_for(running, 10)

    assert len([note for note in await rig.notices() if "keep 1 was refused: online tuning" in note]) == 1
    assert "+max_num_seqs = 512" in rig.patch(rig.toolbox.requests[0])


#: A resumed agent that starts its keep file over, as an agent relaunched from scratch does, then keeps once more.
RESTARTS_ITS_COUNT = (
    "python3 -c \"import json, pathlib, time; state = pathlib.Path('run/state.json'); "
    "state.write_text(json.dumps({'keeps': []})); time.sleep(0.3); "
    "state.write_text(json.dumps({'keeps': [{'decision': 'KEEP'}], 'note': 'after the restart'})); time.sleep(30)\""
)


async def test_a_keep_after_an_agent_restarted_its_count_is_a_new_keep(tmp_path: Path) -> None:
    rig = await _rig(tmp_path, Agent(keeps=(256,), exit=3, resume=RESTARTS_ITS_COUNT))

    await rig.session.preflight()
    with pytest.raises(SessionCrashed):
        await asyncio.wait_for(rig.session.run(), 10)
    await rig.session.stop()
    resumed = rig.restart()
    running = asyncio.create_task(resumed.run())
    await asyncio.sleep(1.0)
    await rig.toolbox_blackboard.write_result(result("r1"))
    await _until(lambda: len(rig.toolbox.requests) == 2, "the keep after the restart")

    assert any(note.startswith(f"{NOTICE} keep 2 imported") for note in await rig.notices())
    await resumed.stop()
    await asyncio.wait_for(running, 10)


async def test_a_keep_file_whose_counter_never_resolves_is_noted_once(tmp_path: Path) -> None:
    rig = await _rig(tmp_path, Agent(keep={"path": "run/state.json", "field": "lessons"}, tail=0.6))

    await rig.session.preflight()
    with pytest.raises(SessionFinished):
        await asyncio.wait_for(rig.session.run(), 10)

    unread = [note for note in await rig.notices() if "run/state.json" in note and "lessons" in note]
    assert len(unread) == 1 and not rig.toolbox.requests


async def test_a_refused_variable_is_left_out_of_later_keeps_while_their_other_launch_changes_go_in(
    tmp_path: Path,
) -> None:
    agent = Agent(envs={"SGLANG_USE_AITER": "1"}, config_source=CONFIG_SOURCE, tail=30.0)
    rig = await _rig(tmp_path, agent)
    rig.toolbox.refused_envs = {"SGLANG_USE_AITER"}

    await rig.session.preflight()
    running = asyncio.create_task(rig.session.run())
    await _until(lambda: rig.toolbox.requests, "the first keep")
    await rig.toolbox_blackboard.write_result(result("r1"))
    await _until(lambda: len(rig.toolbox.requests) == 2, "the second keep")

    configs = [request.delta.config for request in rig.toolbox.requests]
    assert [(config.server_args, dict(config.envs)) for config in configs] == [
        ("--max-num-seqs 256", {}),
        ("--max-num-seqs 512", {}),
    ]
    await rig.session.stop()
    await asyncio.wait_for(running, 10)


async def test_a_keep_whose_launch_the_toolbox_refuses_is_imported_without_it_and_so_are_later_keeps(
    tmp_path: Path,
) -> None:
    # The measurement's launcher sets SGLANG_USE_AITER already; an agent that sets it too keeps it in every keep after.
    agent = Agent(args="--max-num-seqs 128", envs={"SGLANG_USE_AITER": "1"}, config_source=CONFIG_SOURCE, tail=30.0)
    rig = await _rig(tmp_path, agent)
    rig.toolbox.refused_envs = {"SGLANG_USE_AITER"}

    await rig.session.preflight()
    running = asyncio.create_task(rig.session.run())
    await _until(lambda: rig.toolbox.requests, "the first keep without its launch")
    await rig.toolbox_blackboard.write_result(result("r1"))
    await _until(lambda: len(rig.toolbox.requests) == 2, "the second keep")

    assert all(not request.delta.config.envs for request in rig.toolbox.requests)
    assert "+max_num_seqs = 512" in rig.patch(rig.toolbox.requests[1])
    refused = [note for note in await rig.notices() if "SGLANG_USE_AITER" in note]
    assert len(refused) == 1 and "imported without SGLANG_USE_AITER" in refused[0]
    await rig.session.stop()
    await asyncio.wait_for(running, 10)


@pytest.mark.parametrize("bound", ["gemm_a8w8", None], ids=["bound", "unbound"])
async def test_a_kernel_table_the_agent_keeps_in_its_workspace_is_imported_as_the_session_s_artifact(
    tmp_path: Path, bound: str | None
) -> None:
    env = "AITER_CONFIG_GEMM_A8W8"
    table = str(tmp_path / BRANCH / "agent-workspace" / "tuned.csv")
    agent = Agent(keeps=(256,), envs={env: table}, table="M,N,K\n{seqs},5120,5120\n", config_source=CONFIG_SOURCE)
    rig = await _rig(tmp_path, agent)
    if bound:
        await rig.toolbox.put_kernel_artifact(KernelKind.GEMM, bound, env, b"M,N,K\n1,1,1\n")

    await rig.session.preflight()
    running = asyncio.create_task(rig.session.run())
    await _until(lambda: rig.toolbox.requests, "the import")
    Path(table).write_text("rewritten after the keep\n", encoding="utf-8")
    await rig.toolbox_blackboard.write_result(result("r1"))
    with pytest.raises(SessionFinished):
        await asyncio.wait_for(running, 10)

    [request] = rig.toolbox.requests
    [artifact] = request.delta.add_kernels
    assert artifact.target == (bound or env.lower())
    assert rig.toolbox.tables[artifact.sha256] == b"M,N,K\n256,5120,5120\n"
    assert env not in request.delta.config.envs


async def test_a_table_an_agent_reports_in_its_result_file_is_imported_as_rows_over_the_image_s(
    tmp_path: Path,
) -> None:
    """Run 28b3e128's b02: the agent keeps its tuned table in its own checkout and reports the variable it
    accepted in a result file; the keep is that table alone."""

    env = "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE"
    table = tmp_path / BRANCH / "session" / AGENT_DIR / "src" / "run" / "tuned.csv"
    source = {
        "path": "run/result.json",
        "root": "checkout",
        "server_args": "accepted_config.flags",
        "envs": "accepted_config.env",
        "adds": True,
    }
    envs = {env: f"/sgl-workspace/aiter/aiter/configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv:{table}"}
    table_text = "M,N,K\n{seqs},5120,5120\n"
    agent = Agent(keeps=(256,), args="", envs=envs, table=table_text, table_at="run/tuned.csv", config_source=source)
    rig = await _rig(tmp_path, agent)

    await rig.session.preflight()
    running = asyncio.create_task(rig.session.run())
    await _until(lambda: rig.toolbox.requests, "the import")
    await rig.toolbox_blackboard.write_result(result("r1"))
    with pytest.raises(SessionFinished):
        await asyncio.wait_for(running, 10)

    [request] = rig.toolbox.requests
    [artifact] = request.delta.add_kernels
    assert kernel_rows_only(artifact) and rig.toolbox.tables[artifact.sha256] == b"M,N,K\n256,5120,5120\n"
    assert request.delta.config == ConfigDelta()


def _entry(entry_id: str, kind: str, claim: str) -> dict[str, str]:
    return {"id": entry_id, "type": kind, "claim": claim, "evidence": "eval/x.json"}


async def test_the_agent_s_ledger_facts_and_dead_ends_become_the_branch_s_learnings_at_each_keep(
    tmp_path: Path,
) -> None:
    first = (
        _entry("gemm-untuned", "FACT", "dense fp8 GEMMs miss the tuned CSV and cost 64% of the decode step"),
        _entry("triton-slower", "DEAD_END", "aiter triton blockscale GEMM is slower than default CK at M=64"),
        _entry("guess", "ASSUMPTION", "the allreduce is not on the critical path at TP1"),
    )
    second = (
        _entry("gemm-untuned", "FACT", "dense fp8 GEMMs miss the tuned CSV; the asm table cuts them to 4.35 ms"),
        _entry("long", "FACT", "x" * 2000),
    )
    rig = await _rig(tmp_path, Agent(ledger=(first, second)))

    await rig.session.preflight()
    running = asyncio.create_task(rig.session.run())
    await _until(lambda: rig.toolbox.requests, "the first import")
    learnt = await rig.blackboard.learnings(branch=BRANCH)
    assert [(record.learning.kind.value, record.learning.content) for record in learnt] == [
        ("insight", f"FACT gemm-untuned: {first[0]['claim']}"),
        ("pitfall", f"DEAD_END triton-slower: {first[1]['claim']}"),
    ]
    await rig.toolbox_blackboard.write_result(result("r1"))
    await _until(lambda: len(rig.toolbox.requests) == 2, "the second import")
    await rig.toolbox_blackboard.write_result(result("r2"))
    with pytest.raises(SessionFinished):
        await asyncio.wait_for(running, 10)

    revised, long = (await rig.blackboard.learnings(branch=BRANCH))[2:]
    assert revised.learning.content == f"FACT gemm-untuned: {second[0]['claim']}"
    assert revised.learning.supersedes == learnt[0].entry_id
    assert long.learning.content.startswith("FACT long: xxx") and len(long.learning.content) == LEARNING_MAX_CHARS
    assert len(await rig.blackboard.learnings(branch=BRANCH)) == 4


KEPT = "the decode GEMMs miss the tuned CSV"
AFTER = "the fused qk-norm kernel is slower at batch 8"


async def test_ledger_findings_written_after_the_last_keep_are_learnt_when_the_branch_is_stopped(
    tmp_path: Path,
) -> None:
    rig = await _rig(tmp_path, Agent(keeps=(256,), tail=30.0, ledger=((_entry("a", "FACT", KEPT),),)))
    await rig.session.preflight()
    running = asyncio.create_task(rig.session.run())
    await _until(lambda: rig.toolbox.requests, "the import")
    ledger = tmp_path / BRANCH / "session" / AGENT_DIR / "src" / "run" / "ledger.jsonl"
    with ledger.open("a", encoding="utf-8") as lines:
        lines.write(json.dumps(_entry("b", "DEAD_END", AFTER)) + "\n")

    running.cancel()
    await asyncio.gather(running, return_exceptions=True)
    await rig.session.stop()

    learnt = [record.learning.content for record in await rig.blackboard.learnings(branch=BRANCH)]
    assert learnt == [f"FACT a: {KEPT}", f"DEAD_END b: {AFTER}"]


async def test_a_sandbox_watch_started_again_is_noted_in_the_branch_log(tmp_path: Path) -> None:
    rig = await _rig(tmp_path, Agent(keeps=(), tail=30.0))
    await rig.session.preflight()
    rig.sandbox.watch_said.append("the sandbox's watch had ended (exit 1) and was started again")
    running = asyncio.create_task(rig.session.run())

    await _until(lambda: not rig.sandbox.watch_said, "the watch check")
    await rig.session.stop()
    await asyncio.wait_for(running, 10)

    assert f"{NOTICE} the sandbox's watch had ended (exit 1) and was started again" in await rig.notices()


async def test_a_finding_learnt_as_the_agent_finished_is_not_learnt_again_when_the_branch_is_stopped(
    tmp_path: Path,
) -> None:
    rig = await _rig(tmp_path, Agent(keeps=(256,), tail=1.0, ledger=((_entry("a", "FACT", KEPT),),)))
    await rig.session.preflight()
    running = asyncio.create_task(rig.session.run())
    await _until(lambda: rig.toolbox.requests, "the import")
    ledger = tmp_path / BRANCH / "session" / AGENT_DIR / "src" / "run" / "ledger.jsonl"
    with ledger.open("a", encoding="utf-8") as lines:
        lines.write(json.dumps(_entry("b", "DEAD_END", AFTER)) + "\n")
    await rig.toolbox_blackboard.write_result(result("r1"))

    with pytest.raises(SessionFinished):
        await asyncio.wait_for(running, 10)
    await rig.session.stop()

    learnt = [record.learning.content for record in await rig.blackboard.learnings(branch=BRANCH)]
    assert learnt == [f"FACT a: {KEPT}", f"DEAD_END b: {AFTER}"]


def _record(*, measured: tuple[float, float] | None = None) -> SimpleNamespace:
    attempts = () if measured is None else (SimpleNamespace(started_at=measured[0], finished_at=measured[1]),)
    result = None if measured is None else SimpleNamespace(attempts=attempts)
    return SimpleNamespace(result=result, in_flight=measured is None)


def test_the_collection_margin_covers_the_tests_queued_and_the_import_s_test_and_confirm_at_the_longest_test() -> None:
    measured = [_record(measured=(0.0, 500.0)), _record(measured=(1000.0, 1540.0))]

    assert collection_margin([], 2400.0) == 2400.0
    assert collection_margin(measured, 600.0) == 2 * 540.0
    assert collection_margin([*measured, _record(), _record(), _record()], 600.0) == 5 * 540.0
    assert collection_margin([*measured, _record()], 2400.0) == 2400.0


async def test_the_agent_runs_until_its_imports_can_still_be_judged_against_the_gate_s_baseline(
    tmp_path: Path,
) -> None:
    now = time.time()
    rig = await _rig(tmp_path, Agent(keeps=(), tail=30.0), clock=lambda: now, deadline=now + 3600.0)
    gate = LocalBlackboard(tmp_path / "blackboard", Writer(Role.GATE, "gate"))
    rates = {"output_throughput": 0.2}
    root = await rig.repo.head(INTEGRATION_BRANCH)
    await gate.write_baseline(BaselineRecord(root, {"output_throughput": 6290.7}, rates, rates, ("b1",)))

    await rig.session.preflight()

    prompt = rig.sandbox.prompts[0]
    spec = json.loads(prompt[prompt.index("{\n") : prompt.index("\n}\n") + 2])
    assert spec["seconds"] == pytest.approx(3600.0 - 0.5)
    assert spec["baseline"] == 6290.7
    assert spec["knowledge"] == str(tmp_path / BRANCH / "session" / AGENT_DIR / "mb-integration.md")
    await rig.session.stop()


async def test_the_agent_s_integration_notes_follow_the_integration_head(tmp_path: Path) -> None:
    rig = await _rig(tmp_path, Agent(keeps=(), tail=30.0))
    notes = tmp_path / BRANCH / "session" / AGENT_DIR / "mb-integration.md"

    root = await rig.repo.head(INTEGRATION_BRANCH)
    await rig.session.preflight()
    assert root in notes.read_text() and "--max-num-seqs 512" not in notes.read_text()
    running = asyncio.create_task(rig.session.run())
    await rig.repo.create_branch("b2", root)
    other = await rig.repo.commit("b2", root, replace(ROOT, launch=LaunchConfig("--max-num-seqs 512")), "b2")
    head = await rig.repo.commit_integration(root, RecipeDelta(config=ConfigDelta("--max-num-seqs 512")), (other,))
    await _until(lambda: head in notes.read_text(), "the notes of the new integration head")

    assert "--max-num-seqs 512" in notes.read_text()
    await rig.session.stop()
    await asyncio.wait_for(running, 10)


async def test_an_agent_in_component_mode_runs_to_the_deadline_on_the_commit_mb_reads_its_keeps_against(
    tmp_path: Path,
) -> None:
    now = time.time()
    rig = await _rig(
        tmp_path, Component(keeps=(), tail=30.0), clock=lambda: now, deadline=now + 3600.0, start_seconds=0.2
    )

    await rig.session.preflight()
    await rig.session.stop()

    config = json.loads((tmp_path / BRANCH / "session" / AGENT_DIR / "src" / "campaign.json").read_text())
    root = rig.sandbox.prompts[0].split('"root": "')[1].split('"')[0]
    assert (config["mode"], config["docker_container"]) == ("component", "@host")
    assert config["component"] == {"outbox": "snaps", "deadline": int(now + 3600.0 - 0.5), "root": root}
    assert "docs/hosts.md" in rig.sandbox.prompts[0]
    assert root == json.loads((tmp_path / BRANCH / "session" / "agent-watch.json").read_text())["root"]


async def test_an_agent_in_component_mode_reads_the_root_profile_s_op_shares_from_a_file_mb_writes(
    tmp_path: Path,
) -> None:
    rig = await _rig(tmp_path, Component(keeps=(), tail=30.0), start_seconds=0.2)
    root = await rig.repo.head(INTEGRATION_BRANCH)
    rates = {"output_throughput": 0.2}
    gate = LocalBlackboard(tmp_path / "blackboard", Writer(Role.GATE, "gate"))
    await gate.write_baseline(BaselineRecord(root, {"output_throughput": 6290.7}, rates, rates, ("b1",)))
    toolbox = LocalBlackboard(tmp_path / "blackboard", Writer(Role.TOOLBOX, "toolbox"))
    shares = {"gemm": 41.5, "attention": 22.0}
    await toolbox.notify_tool_result(ToolResultPointer("prelude", "job-7", "/a", "trace analysed", root, "p1", shares))

    await rig.session.preflight()
    await rig.session.stop()

    checkout = tmp_path / BRANCH / "session" / AGENT_DIR / "src"
    config = json.loads((checkout / "campaign.json").read_text())
    profile = checkout / SNAPSHOTS["session"] / "op_shares.json"
    assert config["component"]["profile"] == str(profile)
    assert json.loads(profile.read_text()) == {
        "source": "PRELUDE root profile", "commit": root, "job": "job-7", "unit": "percent", "shares": shares,
    }  # fmt: skip


async def test_an_agent_in_component_mode_without_a_root_profile_gets_no_profile(tmp_path: Path) -> None:
    rig = await _rig(tmp_path, Component(keeps=(), tail=30.0), start_seconds=0.2)

    await rig.session.preflight()
    await rig.session.stop()

    checkout = tmp_path / BRANCH / "session" / AGENT_DIR / "src"
    assert "profile" not in json.loads((checkout / "campaign.json").read_text())["component"]
    assert not (checkout / SNAPSHOTS["session"] / "op_shares.json").exists()


async def test_an_agent_in_component_mode_with_no_explore_end_has_a_deadline_mb_s_stop_comes_before(
    tmp_path: Path,
) -> None:
    now = time.time()
    rig = await _rig(tmp_path, Component(keeps=(), tail=30.0), clock=lambda: now, start_seconds=0.2)

    await rig.session.preflight()
    await rig.session.stop()

    config = json.loads((tmp_path / BRANCH / "session" / AGENT_DIR / "src" / "campaign.json").read_text())
    assert config["component"]["deadline"] >= now + 30 * 24 * 3600.0


async def test_the_latest_snapshot_is_imported_with_every_keep_it_is_chained_from_and_its_table(
    tmp_path: Path,
) -> None:
    env = "AITER_CONFIG_GEMM_A8W8"
    agent = Component(keeps=(256, 512), gap=0.1, latest_each=False, table="M,N,K\n{seqs},5120,5120\n")
    rig = await _rig(tmp_path, agent, start_seconds=0.2)

    await rig.session.preflight()
    running = asyncio.create_task(rig.session.run())
    await _until(lambda: rig.toolbox.requests, "the import")
    await rig.toolbox_blackboard.write_result(result("r1"))
    with pytest.raises(SessionFinished):
        await asyncio.wait_for(running, 10)

    [request] = rig.toolbox.requests
    patch = rig.patch(request)
    assert "+FIRST = 1" in patch
    assert "-max_num_seqs = 128" in patch and "+max_num_seqs = 512" in patch
    assert "module.so" not in patch
    assert request.delta.config.server_args == "--max-num-seqs 512"
    [artifact] = request.delta.add_kernels
    assert rig.toolbox.tables[artifact.sha256] == b"M,N,K\n512,5120,5120\n" and kernel_rows_only(artifact)
    assert env not in request.delta.config.envs
    notices = await rig.notices()
    assert any(note.startswith(f"{NOTICE} keep 2 imported") and "seqs 512" in note for note in notices)


async def test_a_snapshot_made_on_a_commit_mb_never_saw_is_refused_in_the_branch_log_and_skipped(
    tmp_path: Path,
) -> None:
    rig = await _rig(tmp_path, Component(keeps=(256, 512, 768), gap=0.6, stray=2, tail=1.0), start_seconds=0.2)

    await rig.session.preflight()
    running = asyncio.create_task(rig.session.run())
    await _until(lambda: rig.toolbox.requests, "the first import")
    await rig.toolbox_blackboard.write_result(result("r1"))
    with pytest.raises(SessionFinished):
        await asyncio.wait_for(running, 10)

    [request] = rig.toolbox.requests
    assert "+max_num_seqs = 256" in rig.patch(request)
    notices = await rig.notices()
    refused = [note for note in notices if "is refused" in note]
    assert [note.split(" cannot")[0] for note in refused] == [f"{NOTICE} keep 2", f"{NOTICE} keep 3"]
    assert "f" * 40 in refused[0]


@pytest.mark.parametrize(
    ("exit", "reason", "said"),
    [(0, "deadline", "deadline"), (0, "map_exhausted", "map_exhausted"), (0, "", "it wrote no exit reason")],
)
async def test_an_agent_in_component_mode_that_finished_records_why(
    tmp_path: Path, exit: int, reason: str, said: str
) -> None:
    rig = await _rig(tmp_path, Component(keeps=(), exit=exit, reason=reason, tail=0.6), start_seconds=0.2)

    await rig.session.preflight()
    with pytest.raises(SessionFinished, match=said):
        await asyncio.wait_for(rig.session.run(), 10)

    assert (await rig.notices())[-1] == f"{NOTICE} the agent finished its run: {said}"


async def _imported(rig: Rig) -> TestRequest:
    """The one import of a component agent's run, once the run finished."""
    await rig.session.preflight()
    running = asyncio.create_task(rig.session.run())
    await _until(lambda: rig.toolbox.requests, "the import")
    await rig.toolbox_blackboard.write_result(result("r1"))
    with pytest.raises(SessionFinished):
        await asyncio.wait_for(running, 10)
    [request] = rig.toolbox.requests
    return request


#: The launch the config MB wrote for Arbor in session MB-20261029T093224Z-658b7c4e (b01): its pack's harness
#: variables sit next to the server's own.
ARBOR_FLAGS = "--tp 1 --max-num-seqs 128"
ARBOR_ENV = {
    "SGLANG_AITER_KV_CACHE_LAYOUT": "vectorized_5d",
    "ISL": "1024",
    "OSL": "1024",
    "NUM_GPUS": "1",
    "BENCH_SEED": "1",
    "GSM8K_LIMIT": "400",
}


async def test_a_snapshot_imports_only_the_launch_changes_made_since_the_start_and_no_workload_variable(
    tmp_path: Path,
) -> None:
    envs = {**ARBOR_ENV, "SGLANG_QWEN3_GATE_UP_PTPC": "1"}
    started = {"server_flags": ARBOR_FLAGS, "server_env": ARBOR_ENV}
    agent = Component(keeps=(256,), gap=0.1, flags=ARBOR_FLAGS, envs=envs, started=started)
    rig = await _rig(tmp_path, agent, start_seconds=0.2)

    request = await _imported(rig)

    assert request.delta.config.envs == {"SGLANG_QWEN3_GATE_UP_PTPC": "1"}
    assert request.delta.config.server_args == "" and request.delta.config.unset_envs == ()
    watch = json.loads((tmp_path / BRANCH / "session" / "agent-watch.json").read_text())
    assert watch["refused_envs"] == {"ISL": "1024", "OSL": "1024"}
    dropped = [note for note in await rig.notices() if "ISL, OSL" in note]
    assert len(dropped) == 1 and "keep 1" in dropped[0]


async def test_a_keep_setting_a_variable_that_needs_a_set_flag_off_is_imported_without_the_flag_and_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # SGLang's Qwen3.8 monokernel asserts with ReplaySSM's spec path on, which the start's launch sets
    monokernel, replayssm = "SGLANG_ENABLE_QWEN4_MONOKERNEL", "--enable-linear-replayssm-spec"
    flags = f"--max-num-seqs 128 {replayssm}"
    agent = Component(keeps=(256,), gap=0.1, flags=flags, envs={monokernel: "1"})
    rig = await _rig(tmp_path, agent, start=replace(ROOT, launch=LaunchConfig(flags)), start_seconds=0.2)

    with caplog.at_level("WARNING"):
        request = await _imported(rig)

    assert replayssm in request.delta.config.remove_args and request.delta.config.envs == {monokernel: "1"}
    warned = [note for note in await rig.notices() if replayssm in note]
    assert len(warned) == 1 and "which was set" in warned[0] and f"{monokernel}=1 needs it off" in warned[0]
    assert any(replayssm in record.getMessage() for record in caplog.records if record.levelname == "WARNING")


async def test_a_snapshot_whose_launch_is_the_one_it_started_with_imports_no_launch_change(tmp_path: Path) -> None:
    started = {"server_flags": ARBOR_FLAGS, "server_env": ARBOR_ENV}
    agent = Component(keeps=(256,), gap=0.1, flags=ARBOR_FLAGS, envs=ARBOR_ENV, started=started)
    rig = await _rig(tmp_path, agent, start_seconds=0.2)

    request = await _imported(rig)

    assert request.delta.config == ConfigDelta()
    assert "+max_num_seqs = 256" in rig.patch(request)


async def test_a_snapshot_after_one_that_was_never_written_carries_both_keeps(tmp_path: Path) -> None:
    rig = await _rig(tmp_path, Component(keeps=(256, 512, 768), gap=0.1, skip=2, latest_each=False), start_seconds=0.2)

    patch = rig.patch(await _imported(rig))

    assert "+FIRST = 1" in patch and "+max_num_seqs = 768" in patch
    assert any(note.startswith(f"{NOTICE} keep 3 imported") for note in await rig.notices())


async def test_a_snapshot_whose_earlier_commit_is_gone_is_read_whole_from_the_root_as_its_notes_say(
    tmp_path: Path,
) -> None:
    rig = await _rig(tmp_path, Component(keeps=(256, 512), gap=0.1, orphan=2, latest_each=False), start_seconds=0.2)

    patch = rig.patch(await _imported(rig))

    assert "+FIRST = 1" in patch and "-max_num_seqs = 128" in patch and "+max_num_seqs = 512" in patch


async def test_a_snapshot_made_on_the_root_after_another_with_no_notes_is_refused(tmp_path: Path) -> None:
    agent = Component(keeps=(256, 512), gap=0.1, orphan=2, orphan_notes=False, latest_each=False, tail=0.6)
    rig = await _rig(tmp_path, agent, start_seconds=0.2)

    await rig.session.preflight()
    with pytest.raises(SessionFinished):
        await asyncio.wait_for(rig.session.run(), 10)

    assert not rig.toolbox.requests
    assert any("snapshot 0002 is refused" in note for note in await rig.notices())


async def test_a_table_too_large_to_copy_is_noted_and_its_variable_imported_as_the_agent_set_it(
    tmp_path: Path,
) -> None:
    agent = Component(keeps=(256,), gap=0.1, table="M,N,K\n{seqs},5120,5120\n", huge=True)
    rig = await _rig(tmp_path, agent, start_seconds=0.2)

    request = await _imported(rig)

    assert request.delta.add_kernels == ()
    assert request.delta.config.envs["AITER_CONFIG_GEMM_A8W8"].endswith(":/arbor/workspace/tuned.csv")
    assert any("AITER_CONFIG_GEMM_A8W8" in note and "not copied" in note for note in await rig.notices())


async def test_a_table_whose_digest_does_not_match_its_listing_refuses_the_snapshot(tmp_path: Path) -> None:
    agent = Component(keeps=(256,), gap=0.1, table="M,N,K\n{seqs},5120,5120\n", bad_sha=True, tail=0.6)
    rig = await _rig(tmp_path, agent, start_seconds=0.2)

    await rig.session.preflight()
    with pytest.raises(SessionFinished):
        await asyncio.wait_for(rig.session.run(), 10)

    assert not rig.toolbox.requests
    assert any("keep 1 cannot be read" in note and "sha256" in note for note in await rig.notices())


async def test_an_agent_in_component_mode_that_crashed_is_resumed_as_any_crash(tmp_path: Path) -> None:
    rig = await _rig(tmp_path, Component(keeps=(), exit=3, tail=0.6), start_seconds=0.2)

    await rig.session.preflight()
    with pytest.raises(SessionCrashed, match="the agent exited 3: ") as crashed:
        await asyncio.wait_for(rig.session.run(), 10)
    await rig.session.stop()

    assert not crashed.value.lost


async def test_an_agent_in_component_mode_that_refused_its_config_says_why_and_is_not_restarted(
    tmp_path: Path,
) -> None:
    rig = await _rig(tmp_path, Component(keeps=(), exit=2, tail=0.6), start_seconds=0.2)

    await rig.session.preflight()
    with pytest.raises(SessionRefused, match="refused its config.*docker_container must be @host"):
        await asyncio.wait_for(rig.session.run(), 10)
    await rig.session.stop()

    assert issubclass(SessionRefused, SessionFinished) and not issubclass(SessionRefused, SessionCrashed)


async def test_an_agent_in_component_mode_stopped_by_mb_records_its_sigterm_exit(tmp_path: Path) -> None:
    rig = await _rig(tmp_path, Component(keeps=(), tail=30.0), start_seconds=0.2)

    await rig.session.preflight()
    running = asyncio.create_task(rig.session.run())
    await asyncio.sleep(0.5)
    await rig.session.stop()
    await asyncio.wait_for(running, 10)

    assert (await rig.notices())[-1] == f"{NOTICE} the agent stopped: sigterm"


#: A stand-in for the image's aiter install: one table variable, its default table, a per-model table and the untuned
#: table its rows' shape key is read from.
AITER_CORE = """
import os

AITER_ROOT_DIR = "/unused"
AITER_CONFIG_GEMM_A8W8 = os.getenv("AITER_CONFIG_GEMM_A8W8", f"{AITER_ROOT_DIR}/aiter/configs/a8w8_tuned_gemm.csv")


class AITER_CONFIG:
    @property
    def AITER_CONFIG_GEMM_A8W8_FILE(self):
        return self.get_config_file("AITER_CONFIG_GEMM_A8W8", AITER_CONFIG_GEMM_A8W8, "a8w8_tuned_gemm")
"""
TABLE_ENV = "AITER_CONFIG_GEMM_A8W8"
IMAGE_DEFAULT = "M,N,K,us\n1,5120,5120,9\n64,5120,5120,10\n"
IMAGE_MODEL = "M,N,K,us\n8192,5120,5120,30\n"
CHAMPION_ROWS = "M,N,K,us\n64,5120,5120,7\n"


def _aiter(tmp_path: Path) -> Path:
    configs = tmp_path / "image" / "aiter" / "configs"
    (configs / "model_configs").mkdir(parents=True)
    (configs.parent / "jit").mkdir()
    (configs.parent / "jit" / "core.py").write_text(AITER_CORE, encoding="utf-8")
    (configs / "a8w8_untuned_gemm.csv").write_text("M,N,K\n", encoding="utf-8")
    (configs / "a8w8_tuned_gemm.csv").write_text(IMAGE_DEFAULT, encoding="utf-8")
    (configs / "model_configs" / "a8w8_tuned_gemm_qwen.csv").write_text(IMAGE_MODEL, encoding="utf-8")
    return configs


def _replayed(blobs: LocalBlobStore) -> Recipe:
    """The root with a Recipe KB champion's tuned rows, kept in the session's blob store as a replay keeps them."""
    sha256 = blobs.put(CHAMPION_ROWS.encode())
    load = {"load": {"via": "env_file", "env": TABLE_ENV, "rows_only": True}}
    table = KernelArtifact(KernelKind.GEMM, "gemm_a8w8", f"kernel/gemm/gemm_a8w8/{sha256}.csv", sha256, load)
    return replace(ROOT, kernels=(table,))


def _spec_envs(rig: Rig) -> dict[str, str]:
    prompt = rig.sandbox.prompts[0]
    return json.loads(prompt[prompt.index("{\n") : prompt.index("\n}\n") + 2])["envs"]


async def test_an_agent_opened_on_a_recipe_kb_champion_runs_with_its_tuned_rows_as_a_test_loads_them(
    tmp_path: Path,
) -> None:
    """Session MB-20261029T093224Z-658b7c4e's b01: the champion's tuned GEMM rows were left out of the agent's launch."""

    configs = _aiter(tmp_path)
    rig = await _rig(tmp_path, start=_replayed(LocalBlobStore(tmp_path / "blobs")), aiter=configs)

    await rig.session.preflight()
    await rig.session.stop()

    *defaults, rows = _spec_envs(rig)[TABLE_ENV].split(os.pathsep)
    assert Path(rows).read_text() == CHAMPION_ROWS
    assert Path(rows).is_relative_to(tmp_path / BRANCH / "session" / AGENT_DIR)
    assert [Path(path).read_text() for path in defaults] == ["M,N,K,us\n1,5120,5120,9\n", IMAGE_MODEL]
    assert (configs / "a8w8_tuned_gemm.csv").read_text() == IMAGE_DEFAULT


async def test_a_snapshot_that_leaves_the_start_s_tables_as_they_were_given_imports_no_table_change(
    tmp_path: Path,
) -> None:
    given = {TABLE_ENV: "{%s}" % TABLE_ENV}
    started = {"server_flags": "--max-num-seqs 128", "server_env": given}
    agent = Component(keeps=(256,), gap=0.1, flags="--max-num-seqs 128", envs=given, started=started)
    start = _replayed(LocalBlobStore(tmp_path / "blobs"))
    rig = await _rig(tmp_path, agent, start=start, start_seconds=0.2, aiter=_aiter(tmp_path))

    request = await _imported(rig)

    assert (request.delta.add_kernels, request.delta.remove_kernels) == ((), ())
    assert request.delta.config == ConfigDelta()


async def test_a_snapshot_that_loads_a_table_of_its_own_imports_its_rows_over_the_start(tmp_path: Path) -> None:
    agent = Component(keeps=(256,), gap=0.1, table="M,N,K,us\n{seqs},5120,5120,3\n")
    start = _replayed(LocalBlobStore(tmp_path / "blobs"))
    rig = await _rig(tmp_path, agent, start=start, start_seconds=0.2, aiter=_aiter(tmp_path))

    request = await _imported(rig)

    [artifact] = request.delta.add_kernels
    assert kernel_rows_only(artifact) and rig.toolbox.tables[artifact.sha256] == b"M,N,K,us\n256,5120,5120,3\n"
    assert artifact.target == "gemm_a8w8" and TABLE_ENV not in request.delta.config.envs


async def test_rows_an_agent_lists_after_the_start_s_tables_are_imported_with_the_start_s_rows(tmp_path: Path) -> None:
    own = tmp_path / BRANCH / "agent-workspace" / "tuned.csv"
    source = {"path": "run/result.json", "root": "checkout", "server_args": "accepted_config.flags",
              "envs": "accepted_config.env", "adds": True}  # fmt: skip
    envs = {TABLE_ENV: "{%s}%s%s" % (TABLE_ENV, os.pathsep, own)}
    agent = Agent(keeps=(256,), args="", envs=envs, table="M,N,K,us\n{seqs},5120,5120,3\n", config_source=source)
    start = _replayed(LocalBlobStore(tmp_path / "blobs"))
    rig = await _rig(tmp_path, agent, start=start, aiter=_aiter(tmp_path))

    request = await _imported(rig)

    [artifact] = request.delta.add_kernels
    table = b"M,N,K,us\n64,5120,5120,7\n256,5120,5120,3\n"
    assert kernel_rows_only(artifact) and rig.toolbox.tables[artifact.sha256] == table
    assert artifact.target == "gemm_a8w8"
    assert request.delta.config == ConfigDelta()


async def test_a_keep_that_reports_the_start_s_tables_as_they_were_given_imports_no_table_change(
    tmp_path: Path,
) -> None:
    source = {"path": "run/result.json", "root": "checkout", "server_args": "accepted_config.flags",
              "envs": "accepted_config.env", "adds": True}  # fmt: skip
    agent = Agent(keeps=(256,), args="", envs={TABLE_ENV: "{%s}" % TABLE_ENV}, config_source=source)
    start = _replayed(LocalBlobStore(tmp_path / "blobs"))
    rig = await _rig(tmp_path, agent, start=start, aiter=_aiter(tmp_path))

    request = await _imported(rig)

    assert (request.delta.add_kernels, request.delta.remove_kernels) == ((), ())
    assert request.delta.config == ConfigDelta()


async def _multi_part(tmp_path: Path, parts: str) -> tuple[Rig, TestRequest]:
    given = {TABLE_ENV: "{%s}" % TABLE_ENV}
    started = {"server_flags": "--max-num-seqs 128", "server_env": given} if parts == "given" else None
    agent = Component(keeps=(256,), gap=0.1, flags="--max-num-seqs 128", envs=given, started=started,
                      table="M,N,K,us\n{seqs},5120,5120,3\n", parts=parts)  # fmt: skip
    start = _replayed(LocalBlobStore(tmp_path / "blobs"))
    rig = await _rig(tmp_path, agent, start=start, start_seconds=0.2, aiter=_aiter(tmp_path))
    return rig, await _imported(rig)


async def test_a_snapshot_that_appends_its_rows_to_the_start_s_tables_imports_the_start_s_rows_and_its_own(
    tmp_path: Path,
) -> None:
    """Arbor copies every file of a multi-part value; the private default copies carry no rows of the agent's."""

    rig, request = await _multi_part(tmp_path, "append")

    [artifact] = request.delta.add_kernels
    table = b"M,N,K,us\n64,5120,5120,7\n256,5120,5120,3\n"
    assert kernel_rows_only(artifact) and rig.toolbox.tables[artifact.sha256] == table
    assert artifact.target == "gemm_a8w8" and TABLE_ENV not in request.delta.config.envs


async def test_a_snapshot_that_replaces_the_start_s_rows_with_its_own_copy_imports_that_copy_s_rows(
    tmp_path: Path,
) -> None:
    rig, request = await _multi_part(tmp_path, "replace")

    [artifact] = request.delta.add_kernels
    assert kernel_rows_only(artifact) and rig.toolbox.tables[artifact.sha256] == b"M,N,K,us\n256,5120,5120,3\n"
    assert artifact.target == "gemm_a8w8" and TABLE_ENV not in request.delta.config.envs


async def test_a_multi_part_snapshot_of_the_start_s_tables_as_they_were_given_imports_no_table_change(
    tmp_path: Path,
) -> None:
    _, request = await _multi_part(tmp_path, "given")

    assert (request.delta.add_kernels, request.delta.remove_kernels) == ((), ())
    assert request.delta.config == ConfigDelta()
