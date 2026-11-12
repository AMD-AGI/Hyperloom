# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import asyncio
from types import SimpleNamespace
import json
import os
import select
import signal
import stat
import subprocess
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, replace, dataclass, field
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import Attempt, Purpose, TestKind, TestRequest, Validity
from hyperloom.contracts.recipe import (
    ConfigDelta,
    KernelKind,
    LaunchConfig,
    Recipe,
    RecipeDelta,
    apply,
)
from hyperloom.measure.compile import LocalBlobStore, PackageBase, kernel_rows_only
from hyperloom.sandbox import SandboxGone
from hyperloom.sandbox.socket import SandboxUnavailable
from hyperloom_multibranch.contracts.lifecycle import BranchTask
from hyperloom_multibranch.contracts.blackboard.vocab import LeaseKind
from hyperloom_multibranch.contracts.toolbox import Backpressure, BranchLease, LeaseCharge, RequestRejected
from hyperloom_multibranch.contracts.blackboard import INTEGRATION_BRANCH, Role, Writer
from hyperloom_multibranch.contracts.blackboard.records import BaselineRecord, ToolResultPointer
from hyperloom_multibranch.blackboard import LocalBlackboard
from hyperloom_multibranch.blackboard.local.blackboard import LEARNING_MAX_CHARS
from hyperloom_multibranch.blackboard.tests.factories import result
from hyperloom_multibranch.specialist import (
    arbor_adapter,
    autonomous_session,
    keep_requeue,
    worktree as worktree_module,
)
from hyperloom_multibranch.specialist.agent_onboarding import SHIM_DIR, OnboardingSetup
from hyperloom_multibranch.specialist.agent_spec import AgentSpec
from hyperloom_multibranch.specialist.agent_session import BranchPlace
from hyperloom_multibranch.specialist.agent_manifest import AgentManifest
from hyperloom_multibranch.specialist.autonomous_session import (
    AGENT_DIR,
    MANIFEST_FILE,
    START_TABLES_DIR,
    WATCH_FILE,
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
from hyperloom_multibranch.specialist.worktree import Upstream
from hyperloom_common.tests.shared_accounts import Accounts, other_accounts
from hyperloom_multibranch.tests.branch_rig import branch_rig
from hyperloom_multibranch.tests.agent_rig import (
    BRANCH,
    CHAMPION_ROWS,
    COMPONENT_AGENT,
    IMAGE_DEFAULT,
    IMAGE_MODEL,
    ROOT,
    TABLE_ENV,
    Component,
    LocalProcess,
    LocalSandbox,
    Toolbox,
    _aiter,
    _arbor_repo,
    _replayed,
    _until,
)


#: An agent that, every ``gap`` seconds, writes each of ``keeps`` to its workspace's config.py and to the config it
#: runs with, commits it when ``commits`` is set, as Arbor does, and records and prints the keep; it ends with ``exit``
#: after ``tail`` more seconds of silence. With ``table`` it also writes that kernel table, for the keep, to its workspace's tuned.csv,
#: or to ``table_at`` in its checkout. It reports ``envs`` as NAME=VALUE words in run/result.json, as GEAK does.
#: Before each keep it appends that keep's entries of ``ledger`` to its ledger, run/ledger.jsonl, as Arbor does.
#: With a keep file, it keeps nothing until ``after`` exists, the session's watch file: the preflight reads the keep
#: file's baseline once the agent started, and a keep made before that read would be taken for one it had before.
AGENT = """
import json, pathlib, subprocess, sys, time
config = json.loads(pathlib.Path("campaign.json").read_text())
workspace = pathlib.Path(config["workspace"])
state = pathlib.Path("run/state.json")
state.parent.mkdir(exist_ok=True)
state.write_text(json.dumps({"keeps": [], "note": ""}))
until = time.time() + 10
while config["after"] and not pathlib.Path(config["after"]).exists() and time.time() < until:
    time.sleep(0.01)
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


def _manifest(workspace: str, agent: Agent | Component, after: str = "") -> str:
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
                "after": after if agent.keep is not None else "",
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


@pytest.fixture(autouse=True)
def fast_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """The session's waits, short enough for a test to see an agent through."""

    waits = {"POLL_SECONDS": 0.05, "SETTLE_SECONDS": 0.3, "CUTOFF_SECONDS": 0.5, "LEASE_POLL_SECONDS": 0.05}
    for name, seconds in waits.items():
        monkeypatch.setattr(autonomous_session, name, seconds)


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
    clock=time.time,
    sleep=asyncio.sleep,
    start: Recipe | None = None,
    deadline: float = 0.0,
    spend: type[AgentSpend] = AgentSpend,
    start_seconds: float = 5.0,
    aiter: Path | None = None,
    source: Path | None = None,
    least_run: float = 0.0,
    spec_fields: Mapping[str, object] | None = None,
) -> Rig:
    """A session of the agent on :data:`BRANCH`, opened at the root, or at ``start`` committed onto it the way PRELUDE
    replays a Recipe KB champion; the agent's source is ``source``, or a repository holding :data:`AGENT` and
    :data:`COMPONENT_AGENT`, and its spec has ``spec_fields``."""

    rig = await branch_rig(tmp_path, ROOT, BRANCH)
    start_commit = rig.root if start is None else await rig.repo.commit(BRANCH, rig.root, start, "champion")
    toolbox_blackboard = LocalBlackboard(tmp_path / "blackboard", Writer(Role.TOOLBOX, "toolbox"))
    worktree = rig.base.worktree(BRANCH, tmp_path / BRANCH / "agent-workspace")
    sandbox, toolbox = LocalSandbox(tmp_path / "logs"), Toolbox(toolbox_blackboard)
    session = tmp_path / BRANCH / "session"

    async def onboard(checkout: Path, prompt: str) -> str:
        sandbox.prompts.append(prompt)
        return _manifest(str(worktree.path), _given(agent, prompt), str(session / WATCH_FILE))

    def onboarding(agent_sandbox) -> OnboardingSetup:
        return OnboardingSetup(agent_sandbox, onboard, start_seconds=start_seconds, poll_seconds=0.05)

    def sandbox_for(lease: BranchLease) -> LocalSandbox:
        if lease.lease_id in sandbox.gone:
            raise SandboxGone(f"the sandbox of {lease.lease_id} ended")
        sandbox.leases.append(lease)
        return sandbox

    spec = AgentSpec("/models/m", "vllm", 1, "fp8", "mi355x", {}, "output_throughput", "", {}, "", (), 60.0, 31100)
    spec = replace(spec, **(spec_fields or {}))

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
        sleep=sleep,
        clock=clock,
        least_run_seconds=least_run,
    )
    if aiter is not None:
        setup = replace(setup, aiter_configs=aiter)
    place = BranchPlace(worktree, rig.blackboard, session)
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


#: vLLM as the image installs it from a wheel: its version names an abbreviated commit, and no checkout holds it.
WHEEL = PackageBase("vllm", Path("/usr/lib/python3/site-packages/vllm"), "vllm", "0.33.0rc1.dev711+gdd4e40142.rocm100")


@pytest.mark.parametrize(
    ("start", "upstream", "unkept"),
    [
        (Recipe(ROOT.base_image_digest, ROOT.launch), None, True),
        (
            Recipe(ROOT.base_image_digest, ROOT.launch),
            Upstream("https://github.com/vllm-project/vllm", "d" * 40),
            False,
        ),
        (ROOT, None, False),
    ],
    ids=["new-component-from-a-wheel", "new-component-with-its-upstream", "component-in-the-recipe"],
)
async def test_an_agent_whose_source_changes_cannot_land_has_the_reason_in_its_spec_and_the_branch_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, start: Recipe, upstream: Upstream | None, unkept: bool
) -> None:
    specs: list[AgentSpec] = []
    make_ready = autonomous_session.make_ready

    async def recorded(source: str, spec: AgentSpec, *args):
        specs.append(spec)
        return await make_ready(source, spec, *args)

    monkeypatch.setattr(autonomous_session, "make_ready", recorded)
    monkeypatch.setattr(worktree_module, "locate_installed_package", lambda _layout: WHEEL)
    monkeypatch.setattr(worktree_module, "installed_upstream", lambda package: upstream if package == WHEEL else None)
    rig = await _rig(tmp_path, start=start)

    await rig.session.preflight()
    await rig.session.stop()

    noted = [note for note in await rig.notices() if "only launch and env changes can" in note]
    assert bool(specs[0].source_unkept) is bool(noted) is unkept
    if unkept:
        assert "0.33.0rc1.dev711+gdd4e40142.rocm100" in specs[0].source_unkept and "wheel" in specs[0].source_unkept
        assert specs[0].source_unkept in noted[0]


#: The toolbox's refusal of a source change to vLLM on a recipe that does not carry it, given without its upstream.
UNLABELLED = RequestRejected(
    "the delta does not fit the parent recipe: vllm is a new component; give its upstream repository and commit",
    "delta_not_applicable",
)
#: A start commit that does not carry vLLM, whose source changes then add it to the recipe.
WITHOUT_VLLM = Recipe(ROOT.base_image_digest, ROOT.launch)


def _from_a_wheel(monkeypatch: pytest.MonkeyPatch) -> None:
    """vLLM installed from a wheel whose upstream commit nothing resolves."""
    monkeypatch.setattr(worktree_module, "locate_installed_package", lambda _layout: WHEEL)
    monkeypatch.setattr(worktree_module, "installed_upstream", lambda _package: None)
    monkeypatch.setattr(autonomous_session, "installed_component_upstream", lambda _component: None)


def _told(monkeypatch: pytest.MonkeyPatch) -> None:
    """The agent made ready as Arbor is: told why no change to its component's source can be kept."""
    make_ready = autonomous_session.make_ready

    async def told(source: str, spec: AgentSpec, *args):
        return replace(await make_ready(source, spec, *args), source_unkept=spec.source_unkept)

    monkeypatch.setattr(autonomous_session, "make_ready", told)


async def test_an_agent_told_its_source_changes_cannot_land_that_keeps_one_ends_its_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _from_a_wheel(monkeypatch)
    _told(monkeypatch)
    rig = await _rig(tmp_path, Agent(tail=30.0), start=WITHOUT_VLLM)
    rig.toolbox.refusals = [UNLABELLED]

    await rig.session.preflight()
    with pytest.raises(SessionRefused, match="kept a source change to vllm after being told it cannot land"):
        await asyncio.wait_for(rig.session.run(), 10)
    await rig.session.stop()
    await rig.session.release()

    notices = await rig.notices()
    assert notices[-1].startswith(f"{NOTICE} keep 1 was refused: the delta does not fit the parent recipe")
    assert "the agent kept a source change to vllm after being told it cannot land" in notices[-1]
    assert rig.toolbox.requests == [] and rig.toolbox.released == ["L1"]


async def test_an_onboarded_agent_never_told_its_source_changes_cannot_land_goes_on_past_their_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _from_a_wheel(monkeypatch)
    rig = await _rig(tmp_path, start=WITHOUT_VLLM)
    rig.toolbox.refusals = [UNLABELLED]

    await rig.session.preflight()
    watch = json.loads((tmp_path / BRANCH / "session" / "agent-watch.json").read_text())
    assert watch["source_unkept"] == "", "a run after a crash holds the agent to what it was told"
    running = asyncio.create_task(rig.session.run())
    await _until(lambda: rig.toolbox.requests, "the keep after the refused one")
    await rig.toolbox_blackboard.write_result(result("r1"))
    with pytest.raises(SessionFinished) as finished:
        await asyncio.wait_for(running, 10)

    assert not isinstance(finished.value, SessionRefused)
    notices = await rig.notices()
    assert any("only launch and env changes can" in note for note in notices)
    assert any(note.startswith(f"{NOTICE} keep 1 was refused: the delta does not fit") for note in notices)
    assert not any("after being told" in note for note in notices)


async def test_a_launch_keep_of_an_agent_told_its_source_changes_cannot_land_is_imported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _from_a_wheel(monkeypatch)
    _told(monkeypatch)
    rig = await _rig(
        tmp_path, Agent(keeps=(128,), args="--max-num-seqs 64", config_source=CONFIG_SOURCE), start=WITHOUT_VLLM
    )

    await rig.session.preflight()
    running = asyncio.create_task(rig.session.run())
    await _until(lambda: rig.toolbox.requests, "the launch keep")
    await rig.toolbox_blackboard.write_result(result("r1"))
    with pytest.raises(SessionFinished) as finished:
        await asyncio.wait_for(running, 10)

    assert not isinstance(finished.value, SessionRefused)
    [request] = rig.toolbox.requests
    assert request.delta.components == () and "--max-num-seqs 64" in request.delta.config.server_args


async def test_an_agent_told_its_source_changes_cannot_land_is_held_to_it_after_a_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _from_a_wheel(monkeypatch)
    _told(monkeypatch)
    rig = await _rig(tmp_path, Agent(exit=3, resume="sleep 30"), start=WITHOUT_VLLM)

    await rig.session.preflight()
    with pytest.raises(SessionCrashed):
        await asyncio.wait_for(rig.session.run(), 10)
    await rig.session.stop()
    rig.toolbox.refusals = [UNLABELLED]
    resumed = rig.restart()
    running = asyncio.create_task(resumed.run())
    await rig.toolbox_blackboard.write_result(result("r1"))
    with pytest.raises(SessionRefused, match="after being told it cannot land"):
        await asyncio.wait_for(running, 10)
    await resumed.stop()

    assert len(rig.toolbox.requests) == 1


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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(autonomous_session, "STALL_SECONDS", 1.0)
    rig = await _rig(tmp_path, Agent(keeps=(256, 512, 768), gap=0.5, tail=30.0))
    await rig.session.preflight()
    started = time.monotonic()

    with pytest.raises(SessionCrashed, match="no output") as crashed:
        await asyncio.wait_for(rig.session.run(), 10)
    await rig.session.stop()

    assert crashed.value.lost
    assert time.monotonic() - started > 2.0


async def test_an_agent_silent_on_its_console_while_it_writes_its_workspace_is_not_lost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(autonomous_session, "STALL_SECONDS", 1.0)
    rig = await _rig(tmp_path, Agent(keeps=(256,), gap=0.1, tail=30.0, busy=True))
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


async def test_a_run_stuck_in_one_of_its_steps_is_a_lost_session_while_the_agent_still_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(autonomous_session, "STALL_SECONDS", 0.5)
    rig = await _rig(tmp_path, Agent(keeps=tuple(range(1, 30)), gap=0.1), spend=_HungSpend)
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


async def test_a_new_sandbox_whose_install_failed_is_not_resumed_in_but_leased_and_installed_again(
    tmp_path: Path,
) -> None:
    count = tmp_path / "installs"
    # The second install, the first in the new sandbox, fails.
    command = f"n=$(cat {count} 2>/dev/null || echo 0); echo $((n + 1)) > {count}; test $n -ne 1"
    rig = await _rig(tmp_path, Agent(exit=3, resume="sleep 30", install=(command,)), deadline=time.time() + 600)

    await rig.session.preflight()
    with pytest.raises(SessionCrashed):
        await asyncio.wait_for(rig.session.run(), 10)
    await rig.session.stop()
    rig.sandbox.gone.add("L1")
    failed = rig.restart()
    with pytest.raises(SessionCrashed, match="install in its new sandbox failed"):
        await asyncio.wait_for(failed.run(), 10)
    await failed.stop()
    resumed = rig.restart()
    running = asyncio.create_task(resumed.run())
    await _until(lambda: rig.sandbox.started[-1] == "sleep 30", "the agent resumed")

    assert count.read_text().strip() == "3"
    assert rig.toolbox.released == ["L1", "L2"] and len(rig.toolbox.leases) == 3
    await resumed.stop()
    await asyncio.wait_for(running, 10)
    await resumed.release()
    assert rig.toolbox.released == ["L1", "L2", "L3"]


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


async def test_a_keep_made_just_before_the_agent_exits_is_imported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(autonomous_session, "POLL_SECONDS", 1.0)
    rig = await _rig(tmp_path, Agent(keeps=(256,), gap=0.2, tail=0.0))

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


NO_ROOM = RequestRejected(
    "leasing 1 of the 1 cards would leave fewer than one test needs", "lease_leaves_no_measurement"
)


@pytest.mark.parametrize(
    "busy", [NO_ROOM, SandboxUnavailable("no 1 card(s) are free now")], ids=["no_measurement", "cards_busy"]
)
async def test_an_agent_whose_cards_are_taken_waits_for_them_and_starts(tmp_path: Path, busy: Exception) -> None:
    rig = await _rig(tmp_path, deadline=time.time() + 3600.0, least_run=60.0)
    rig.toolbox.lease_refusals = [busy, busy]

    await rig.session.preflight()
    await rig.session.release()

    assert rig.toolbox.lease_tries == 3 and rig.sandbox.started
    assert len([note for note in await rig.notices() if "waits for cards" in note]) == 1


async def test_an_agent_whose_lease_is_refused_for_another_reason_fails_its_preflight(tmp_path: Path) -> None:
    rig = await _rig(tmp_path, deadline=time.time() + 3600.0)
    rig.toolbox.lease_refusals = [RequestRejected("the branch holds a lease", "lease_exists")]

    with pytest.raises(PreflightFailed, match="lease_exists"):
        await rig.session.preflight()
    await rig.session.release()

    assert rig.toolbox.lease_tries == 1 and not rig.sandbox.started and rig.toolbox.released == []


async def test_an_agent_that_waits_for_cards_past_the_time_it_needs_to_run_fails_its_preflight(
    tmp_path: Path,
) -> None:
    now = [0.0]

    async def sleep(seconds: float) -> None:
        now[0] += seconds

    rig = await _rig(tmp_path, clock=lambda: now[0], sleep=sleep, deadline=3.0, least_run=2.0)
    rig.toolbox.cards_left = 0

    with pytest.raises(PreflightFailed, match="waited .* for cards.*lease_leaves_no_measurement"):
        await rig.session.preflight()
    await rig.session.release()

    assert rig.toolbox.lease_tries > 1 and not rig.sandbox.started and rig.toolbox.released == []


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
    """The agent keeps its tuned table in its own checkout and reports the variable it
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
    assert collection_margin([*measured, _record()], 2400.0) == 3 * 540.0, "once measured, the tests decide"


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


async def test_the_agent_is_asked_for_the_start_s_launch_with_the_resolved_flags_it_removes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    specs: list[AgentSpec] = []
    make_ready = autonomous_session.make_ready

    async def recorded(agent: str, spec: AgentSpec, *args):
        specs.append(spec)
        return await make_ready(agent, spec, *args)

    monkeypatch.setattr(autonomous_session, "make_ready", recorded)
    start = replace(ROOT, launch=LaunchConfig("--max-num-seqs 128", remove_args=("--enforce-eager",)))
    rig = await _rig(tmp_path, Agent(keeps=(), tail=30.0), start=start)

    await rig.session.preflight()

    assert (specs[0].server_args, specs[0].remove_args) == ("--max-num-seqs 128", ("--enforce-eager",))
    await rig.session.stop()


async def test_the_agent_stops_by_the_session_s_own_tests_once_one_was_measured(tmp_path: Path) -> None:
    now = time.time()
    rig = await _rig(tmp_path, Agent(keeps=(), tail=30.0), clock=lambda: now, deadline=now + 3600.0)
    root = await rig.repo.head(INTEGRATION_BRANCH)
    measured = TestRequest(
        kind=TestKind.TEST,
        purpose=Purpose.BASELINE,
        branch_id=INTEGRATION_BRANCH,
        parent_commit=root,
        delta=RecipeDelta(),
        request_id="root-1",
    )
    await rig.toolbox_blackboard.write_request(measured)
    attempt = Attempt(started_at=now - 1000.0, finished_at=now - 400.0)
    await rig.toolbox_blackboard.write_result(replace(result("root-1"), attempts=(attempt,)))

    await rig.session.preflight()

    prompt = rig.sandbox.prompts[0]
    spec = json.loads(prompt[prompt.index("{\n") : prompt.index("\n}\n") + 2])
    assert spec["seconds"] == pytest.approx(3600.0 - 2 * 600.0), "its import's test and confirm, at 600 s each"
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


async def test_a_keep_s_files_outside_the_component_never_reach_its_import_and_are_named_in_the_branch_log(
    tmp_path: Path,
) -> None:
    rig = await _rig(tmp_path, Component(keeps=(256,), gap=0.1, scratch=True), start_seconds=0.2)

    await rig.session.preflight()
    running = asyncio.create_task(rig.session.run())
    await _until(lambda: rig.toolbox.requests, "the import")
    await rig.toolbox_blackboard.write_result(result("r1"))
    with pytest.raises(SessionFinished):
        await asyncio.wait_for(running, 10)

    [request] = rig.toolbox.requests
    patch = rig.patch(request)
    assert "+FIRST = 1" in patch and "eval/" not in patch
    notices = await rig.notices()
    assert any("keep 1" in note and "eval/gpucore.1.gpu" in note and "eval/server.log" in note for note in notices)


async def test_a_keep_whose_source_change_is_over_the_size_cap_is_refused_in_the_branch_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(autonomous_session, "MAX_SOURCE_PATCH_BYTES", 64)
    rig = await _rig(tmp_path, Component(keeps=(256,), gap=0.1, tail=1.0), start_seconds=0.2)

    await rig.session.preflight()
    with pytest.raises(SessionFinished):
        await asyncio.wait_for(rig.session.run(), 10)

    assert not rig.toolbox.requests
    notices = await rig.notices()
    assert any("keep 1 cannot be read" in note and "64 bytes" in note for note in notices)


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


#: The launch the config MB wrote for an Arbor branch: its pack's harness
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
    # GSM8K_LIMIT is Arbor's evaluation setting, which MB gives it (arbor_adapter.EVAL_ENVS)
    assert watch["refused_envs"] == {"GSM8K_LIMIT": "400", "ISL": "1024", "OSL": "1024"}
    dropped = [note for note in await rig.notices() if "GSM8K_LIMIT, ISL, OSL" in note]
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


async def test_a_stop_whose_kill_times_out_is_noted_and_the_lease_still_goes_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = await _rig(tmp_path, Agent(keeps=(), tail=30.0))
    kill = LocalProcess.kill

    async def unavailable(self: LocalProcess) -> None:
        await kill(self)
        raise SandboxUnavailable("the sandbox service timed out")

    await rig.session.preflight()
    running = asyncio.create_task(rig.session.run())
    await asyncio.sleep(0.3)
    monkeypatch.setattr(LocalProcess, "kill", unavailable)
    await rig.session.stop()
    await asyncio.wait_for(running, 10)
    await rig.session.release()

    assert any("the sandbox service timed out" in notice for notice in await rig.notices())
    assert rig.toolbox.released == ["L1"]


@pytest.mark.parametrize("shared", [True, False], ids=["shared", "private"])
@pytest.mark.parametrize("tables", [False, True], ids=["root", "kb-champion"])
async def test_a_preflight_makes_the_branch_s_directories_the_group_s_only_on_shared_storage(
    tmp_path: Path, shared: bool, tables: bool
) -> None:
    given = {"start": _replayed(LocalBlobStore(tmp_path / "blobs")), "aiter": _aiter(tmp_path)} if tables else {}
    rig = await _rig(tmp_path, **given)
    (tmp_path / BRANCH).chmod(0o2775 if shared else 0o755)
    previous = os.umask(0o022)
    try:
        await rig.session.preflight()
    finally:
        os.umask(previous)
    await rig.session.stop()

    agent = tmp_path / BRANCH / "session" / AGENT_DIR
    made = [agent.parent, agent, agent / "src", *([agent / START_TABLES_DIR] if tables else [])]
    modes = {path.name: oct(path.stat().st_mode & 0o7777) for path in made}
    assert modes == dict.fromkeys(modes, oct(0o2775 if shared else 0o755))
    written = [path for path in (agent.parent.iterdir()) if path.is_file()] + [agent / "mb-integration.md"]
    written += [path for path in (agent / START_TABLES_DIR).rglob("*") if path.is_file()] if tables else []
    modes = {str(path): oct(path.stat().st_mode & 0o777) for path in written}
    assert len(modes) > 4 and modes == dict.fromkeys(modes, oct(0o664 if shared else 0o644))


def _under_git(path: Path, names: tuple[str, ...]) -> bool:
    parts = path.parts
    return any(parts[i] == ".git" and parts[i + 1] in names for i in range(len(parts) - 1))


@pytest.mark.parametrize("arbor", [False, True], ids=["agent", "arbor"])
async def test_a_preflight_below_shared_storage_leaves_the_group_write_on_all_it_made_but_git_s_own_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arbor: bool
) -> None:
    tmp_path.chmod(0o2775)
    previous = os.umask(0o022)
    try:
        given: dict[str, object] = {}
        if arbor:
            monkeypatch.setattr(arbor_adapter, "INSTALL", ())
            plan, dossier = tmp_path / "plan.json", tmp_path / "dossier" / "d1"
            plan.write_text(json.dumps({"directions": ["fuse the decode norm"]}))
            dossier.mkdir(parents=True)
            (dossier / "index.md").write_text("# dossier\n")
            source = _arbor_repo(tmp_path, Component(keeps=(), tail=0.6))
            given = {
                "source": source,
                "start_seconds": 0.2,
                "spec_fields": {"plan": str(plan), "dossier": str(dossier)},
            }
        rig = await _rig(tmp_path, **given)
        await rig.session.preflight()
    finally:
        os.umask(previous)
    await rig.session.stop()

    # Git writes every object read-only, and the docker shim is replaced by rename, never rewritten.
    made = [
        path
        for path in (tmp_path / BRANCH).rglob("*")
        if not path.is_symlink()
        and not (path.is_file() and _under_git(path, ("objects",)))
        and path != tmp_path / BRANCH / "session" / AGENT_DIR / SHIM_DIR / "docker"
    ]
    git = [path for path in made if _under_git(path, ("config", "hooks", "info"))]
    base = tmp_path / "source" / ".git"
    git += [base / "config", base / "info", *(base / "info").iterdir()]
    private = [str(path) for path in made if path not in git and not path.stat().st_mode & stat.S_IWGRP]
    widened = [str(path) for path in git if path.stat().st_mode & 0o022]
    assert len(made) > 20 and len(git) > 6
    assert private == [] and widened == []


def _as_member(accounts: Accounts, uid: int, leg: Callable[[], Awaitable[None]]) -> str:
    """Run ``leg`` in a child process as ``uid`` of the accounts' group under umask ``022``: what it raised, or
    ``"finished"``."""
    read, write = os.pipe()
    pid = os.fork()
    if pid == 0:
        said = "finished"
        try:
            os.setgroups([accounts.group])
            os.setresgid(uid, uid, uid)
            os.setresuid(uid, uid, uid)
            os.umask(0o022)
            asyncio.run(leg())
        except BaseException as error:  # noqa: BLE001 - everything the leg raised goes back to the test
            said = repr(error)
        os.write(write, said.encode())
        os._exit(0)
    os.close(write)
    if not select.select([read], [], [], 120)[0]:
        os.kill(pid, signal.SIGKILL)  # forked from a process with threads, it may hang on a lock one of them held
    with os.fdopen(read) as told:
        said = told.read()
    os.waitpid(pid, 0)
    return said


@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
@pytest.mark.parametrize("arbor", [False, True], ids=["agent", "arbor"])
def test_another_account_of_the_group_resumes_a_branch_whose_files_a_first_leg_wrote_under_umask_022(
    monkeypatch: pytest.MonkeyPatch, arbor: bool
) -> None:
    with other_accounts() as accounts:
        tmp_path = accounts.shared_dir("run")
        blobs = LocalBlobStore(tmp_path / "blobs")
        given: dict[str, object] = {"start": _replayed(blobs), "aiter": _aiter(tmp_path)}
        if arbor:
            monkeypatch.setattr(arbor_adapter, "INSTALL", ())
            plan = tmp_path / "plan.json"
            plan.write_text(json.dumps({"directions": ["fuse the decode norm"]}))
            source = _arbor_repo(tmp_path, Component(keeps=(), exit=3, tail=0.6))
            given |= {"source": source, "start_seconds": 0.2, "spec_fields": {"plan": str(plan)}}
        agent = Agent(keeps=(), exit=3, resume="exit 3")
        rig = asyncio.run(_rig(tmp_path, agent, **given))

        async def first_leg() -> None:
            # The integration head holds a kernel table, laid out below the notes, and moves once the first leg ends.
            root = await rig.repo.head(INTEGRATION_BRANCH)
            held = await rig.repo.commit(INTEGRATION_BRANCH, root, _replayed(blobs), "kb")
            await rig.session.preflight()
            with pytest.raises(SessionCrashed):
                await asyncio.wait_for(rig.session.run(), 10)
            await rig.session.stop()
            moved = replace(_replayed(blobs), launch=LaunchConfig("--max-num-seqs 512"))
            await rig.repo.commit(INTEGRATION_BRANCH, held, moved, "moved")

        previous = os.umask(0o022)
        try:
            asyncio.run(first_leg())
        finally:
            os.umask(previous)
        session = tmp_path / BRANCH / "session"
        for path in (session, *session.rglob("*")):
            os.lchown(path, accounts.first, -1)
        (tmp_path / "logs").chmod(0o2775)  # the fake sandbox's own logs

        async def resumed_leg() -> None:
            resumed = rig.restart()
            await resumed.preflight()
            with pytest.raises(SessionCrashed, match="exited 3"):
                await asyncio.wait_for(resumed.run(), 10)
            await resumed.stop()

        assert _as_member(accounts, accounts.second, resumed_leg) == "finished"
        assert "--max-num-seqs 512" in (session / AGENT_DIR / "mb-integration.md").read_text(encoding="utf-8")


def _spec_envs(rig: Rig) -> dict[str, str]:
    prompt = rig.sandbox.prompts[0]
    return json.loads(prompt[prompt.index("{\n") : prompt.index("\n}\n") + 2])["envs"]


async def test_an_agent_opened_on_a_recipe_kb_champion_runs_with_its_tuned_rows_as_a_test_loads_them(
    tmp_path: Path,
) -> None:
    """The champion's tuned GEMM rows were left out of the agent's launch."""

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


INFRA_REASON = "4 attempts failed for infrastructure reasons: foreign_process_on_card: pid 4242 on card 3"


@pytest.fixture
def quick_requeue(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(keep_requeue, "REQUEUE_SECONDS", 0.3)


def _failed(request_id: str, validity: Validity, reason: str = INFRA_REASON):
    return result(request_id, validity=validity, failure_reason=reason)


async def test_a_keep_whose_import_measured_nothing_for_infrastructure_goes_once_more_and_is_measured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The agent exits while the requeue still waits: its run finishes only once the requeued import settled.
    monkeypatch.setattr(keep_requeue, "REQUEUE_SECONDS", 2.0)
    rig = await _rig(tmp_path, Agent(keeps=(256,), tail=0.3))

    await rig.session.preflight()
    running = asyncio.create_task(rig.session.run())
    await _until(lambda: rig.toolbox.requests, "the import")
    await rig.toolbox_blackboard.write_result(_failed("r1", Validity.MEASUREMENT_FAILED))
    await _until(lambda: len(rig.toolbox.requests) == 2, "the requeued import")
    await rig.toolbox_blackboard.write_result(result("r2"))
    with pytest.raises(SessionFinished):
        await asyncio.wait_for(running, 10)

    first, again = rig.toolbox.requests
    assert rig.patch(again) == rig.patch(first) and again.delta.config == first.delta.config, "the same keep again"
    notices = await rig.notices()
    assert any(f"keep 1's import r1 measured nothing ({INFRA_REASON}); it goes once more" in n for n in notices)
    assert notices[-1] == f"{NOTICE} the agent finished its run", "the run waited for the requeued import"


async def test_a_requeue_waits_its_time_after_the_import_ended(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(keep_requeue, "REQUEUE_SECONDS", 3.0)
    rig = await _rig(tmp_path, Agent(keeps=(256,), tail=30.0))

    await rig.session.preflight()
    running = asyncio.create_task(rig.session.run())
    await _until(lambda: rig.toolbox.requests, "the import")
    await rig.toolbox_blackboard.write_result(_failed("r1", Validity.MEASUREMENT_FAILED))
    await asyncio.sleep(1.5)
    assert len(rig.toolbox.requests) == 1, "not before its time"
    await _until(lambda: len(rig.toolbox.requests) == 2, "the requeued import")

    await rig.session.stop()
    await asyncio.wait_for(running, 10)


@pytest.mark.parametrize(
    "validity, reason",
    [
        (Validity.CANDIDATE_FAILED, "server_boot_timeout: no ready line"),
        (Validity.MEASUREMENT_FAILED, "measurement refused: the spec was rejected"),
        (Validity.MEASUREMENT_FAILED, "inert_patch: process 1 imported aiter from the image, not from the farm"),
    ],
    ids=["candidate", "program_error", "repeats_in_every_sandbox"],
)
async def test_a_keep_whose_import_failed_otherwise_is_not_requeued(
    tmp_path: Path, quick_requeue: None, validity: Validity, reason: str
) -> None:
    rig = await _rig(tmp_path, Agent(keeps=(256,), tail=1.0))

    await rig.session.preflight()
    running = asyncio.create_task(rig.session.run())
    await _until(lambda: rig.toolbox.requests, "the import")
    await rig.toolbox_blackboard.write_result(_failed("r1", validity, reason))
    with pytest.raises(SessionFinished):
        await asyncio.wait_for(running, 10)
    await asyncio.sleep(0.6)

    assert len(rig.toolbox.requests) == 1
    assert not any("goes once more" in note for note in await rig.notices())


async def test_a_keep_is_requeued_once_and_after_the_cap_of_failed_requeues_none_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, quick_requeue: None
) -> None:
    monkeypatch.setattr(keep_requeue, "REQUEUE_FAILURES", 1)
    rig = await _rig(tmp_path, Agent(keeps=(256, 512), gap=4.0, tail=1.0))

    await rig.session.preflight()
    running = asyncio.create_task(rig.session.run())
    await _until(lambda: rig.toolbox.requests, "the first import")
    await rig.toolbox_blackboard.write_result(_failed("r1", Validity.MEASUREMENT_FAILED))
    await _until(lambda: len(rig.toolbox.requests) == 2, "the requeued import")
    await rig.toolbox_blackboard.write_result(_failed("r2", Validity.MEASUREMENT_FAILED))
    await _until(lambda: len(rig.toolbox.requests) == 3, "the second keep's import", seconds=15.0)
    await rig.toolbox_blackboard.write_result(_failed("r3", Validity.MEASUREMENT_FAILED))
    with pytest.raises(SessionFinished):
        await asyncio.wait_for(running, 15)
    await asyncio.sleep(0.6)

    assert "+max_num_seqs = 512" in rig.patch(rig.toolbox.requests[2]) and len(rig.toolbox.requests) == 3
    notices = await rig.notices()
    assert any("keep 1's import r2 measured nothing" in n and "it went once more already" in n for n in notices)
    assert any(
        "keep 2's import r3 measured nothing" in n and "the last 1 requeued imports measured nothing again" in n
        for n in notices
    ), "a node that stays broken does not loop"


async def test_a_later_keep_carries_the_change_of_one_that_measured_nothing_and_no_requeue_goes(
    tmp_path: Path, quick_requeue: None
) -> None:
    rig = await _rig(tmp_path, Agent(keeps=(256, 512), gap=0.3, tail=1.0))

    await rig.session.preflight()
    running = asyncio.create_task(rig.session.run())
    await _until(lambda: rig.toolbox.requests, "the first import")
    await asyncio.sleep(1.0)
    await rig.toolbox_blackboard.write_result(_failed("r1", Validity.MEASUREMENT_FAILED))
    await _until(lambda: len(rig.toolbox.requests) == 2, "the second keep's import")
    await rig.toolbox_blackboard.write_result(result("r2"))
    with pytest.raises(SessionFinished):
        await asyncio.wait_for(running, 10)
    await asyncio.sleep(0.6)

    assert len(rig.toolbox.requests) == 2 and "+max_num_seqs = 512" in rig.patch(rig.toolbox.requests[1])
    assert not any("goes once more" in note for note in await rig.notices())


async def test_a_requeue_due_while_a_later_keep_waits_gives_way_to_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The first keep's requeue falls due while another test keeps the branch busy and the second keep lands: once the
    # branch is free, the second keep, which carries the first's change, goes, and the requeue does not take its place.
    monkeypatch.setattr(keep_requeue, "REQUEUE_SECONDS", 1.0)
    rig = await _rig(tmp_path, Agent(keeps=(256, 512), gap=3.0, tail=1.0))
    watch_file = rig.session._place.workspace / autonomous_session.WATCH_FILE

    def watched() -> dict:
        return json.loads(watch_file.read_text(encoding="utf-8")) if watch_file.exists() else {}

    await rig.session.preflight()
    running = asyncio.create_task(rig.session.run())
    await _until(lambda: rig.toolbox.requests, "the first import")
    await rig.toolbox_blackboard.write_result(_failed("r1", Validity.MEASUREMENT_FAILED))
    await _until(lambda: watched().get("requeue_at"), "the requeue's time")
    await rig.toolbox_blackboard.write_request(replace(rig.toolbox.requests[0], request_id="x1"))
    await _until(lambda: (watched().get("pending") or {}).get("seq") == 2, "the second keep")
    await _until(lambda: time.time() > watched()["requeue_at"], "the requeue falling due")
    await rig.toolbox_blackboard.write_result(result("x1"))
    await _until(lambda: len(rig.toolbox.requests) == 2, "the next import")
    await rig.toolbox_blackboard.write_result(result("r2"))
    with pytest.raises(SessionFinished):
        await asyncio.wait_for(running, 10)

    assert len(rig.toolbox.requests) == 2 and "+max_num_seqs = 512" in rig.patch(rig.toolbox.requests[1])
    assert not any("goes once more" in note for note in await rig.notices())
