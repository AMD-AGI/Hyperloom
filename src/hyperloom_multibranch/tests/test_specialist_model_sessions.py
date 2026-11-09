# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import asyncio
import shlex
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.common.agent_isolation import BwrapIsolation
from hyperloom.measure.compile import PackageBase
from hyperloom_common.bwrap import BwrapProbe
from hyperloom.sandbox import SandboxGone
from hyperloom_multibranch.contracts.toolbox import BranchLease
from hyperloom_multibranch.specialist import agent_sessions
from hyperloom_multibranch.specialist.agent_sandbox import UNCONFINED
from hyperloom_multibranch.specialist.agent_spec import AgentSpec
from hyperloom_multibranch.specialist.arbor_adapter import MAPPED
from hyperloom_multibranch.specialist.dossier_pass import DossierOutcome, DossierSetup, DossierStart
from hyperloom_multibranch.specialist.agent_sessions import (
    AgentSessions,
    LocalAgentSandboxes,
    RemoteAgentSandboxes,
    with_agents,
)
from hyperloom_multibranch.specialist.autonomous_session import AutonomousSession
from hyperloom_multibranch.specialist.gate_client import read_gate_client
from hyperloom_multibranch.specialist.agent_session import AgentCommand, AgentSession, AgentSetup, BranchPlace
from hyperloom_multibranch.specialist.model_sessions import ModelSessions, model_sessions
from hyperloom_multibranch.specialist.prompts import role_prompt
from hyperloom_multibranch.contracts.lifecycle import BranchTask
from hyperloom_multibranch.specialist.runtime import SessionStart
from hyperloom_multibranch.specialist import worktree as worktree_module
from hyperloom_multibranch.specialist.worktree import SourceBase
from hyperloom_multibranch.toolbox.hosted_leases import HostedBranchLease


def test_a_branch_s_sessions_and_its_worktree_tools_share_one_worktree_on_one_source_base(tmp_path, monkeypatch):
    package = tmp_path / "site-packages" / "vllm"
    package.mkdir(parents=True)
    (package / "config.py").write_bytes(b"max_num_seqs = 128\n")
    installed = PackageBase("vllm", package, None, None)
    monkeypatch.setattr(worktree_module, "locate_installed_package", lambda _layout: installed)
    role = role_prompt("qwen3-8b on vllm, TP 1, concurrency 64")
    command = AgentCommand("claude", "claude", model="claude-opus-5-5")
    setup = AgentSetup(command, role, {}, object(), object(), object(), object())

    def built() -> ModelSessions:
        return model_sessions(
            setup,
            base=SourceBase(tmp_path / "MB-source"),
            components=("vllm",),
            branch_dir=lambda branch: tmp_path / "MB-branches" / branch,
            blackboard_for=lambda branch: object(),
        )

    sessions = built()

    assert isinstance(sessions.factory(SessionStart("s1", "c0", 2, BranchTask("leads"))), AgentSession)
    worktree = sessions.worktrees("s1")
    assert worktree.path == tmp_path / "MB-branches" / "s1" / "worktree"
    assert (worktree.path / "vllm" / "config.py").read_bytes() == b"max_num_seqs = 128\n"
    assert built().worktrees("s1").path == worktree.path


def _agent_sessions() -> AgentSessions:
    spec = AgentSpec("/models/m", "vllm", 1, "bf16", "mi355x", {}, "output_throughput", "", {}, "", (), 3600.0, 31100)
    command = AgentCommand("claude", "claude", model="claude-opus-5-5")
    return AgentSessions(
        spec, command, {}, object(), object(), LocalAgentSandboxes(object(), choose=lambda: UNCONFINED)
    )


def test_an_agent_branch_waits_for_cards_while_it_would_still_run_the_session_s_least(tmp_path):
    agents = replace(_agent_sessions(), least_run_seconds=3780.0)
    place = BranchPlace(SimpleNamespace(path=tmp_path / "worktree"), object(), tmp_path / "session")

    session = agents.session(SessionStart("a1", "c0", 1, BranchTask(agent="arbor")), place)

    assert session._setup.least_run_seconds == 3780.0


def test_a_branch_whose_task_names_an_agent_runs_it_on_ports_of_its_own_in_the_sandbox_it_leases(tmp_path, monkeypatch):
    package = tmp_path / "site-packages" / "vllm"
    package.mkdir(parents=True)
    (package / "config.py").write_bytes(b"max_num_seqs = 128\n")
    installed = PackageBase("vllm", package, None, None)
    monkeypatch.setattr(worktree_module, "locate_installed_package", lambda _layout: installed)
    setup = AgentSetup(AgentCommand("claude", "claude"), "role", {}, object(), object(), object(), object())
    base = SourceBase(tmp_path / "MB-source")

    def branch_dir(branch: str) -> Path:
        return tmp_path / "MB-branches" / branch

    def blackboard_for(branch: str) -> object:
        return object()

    sessions = model_sessions(
        setup, base=base, components=("vllm",), branch_dir=branch_dir, blackboard_for=blackboard_for
    )
    built = with_agents(sessions, _agent_sessions(), branch_dir=branch_dir, blackboard_for=blackboard_for)
    first, second, third = (
        built.factory(SessionStart(branch, "c0", 1, BranchTask(agent="https://example.com/agent.git")))
        for branch in ("a1", "a2", "a3")
    )
    again = built.factory(SessionStart("a1", "c0", 1, BranchTask(agent="https://example.com/agent.git")))

    assert all(isinstance(session, AutonomousSession) for session in (first, second, third))
    specs = [session._setup.spec for session in (first, second, third, again)]
    assert [spec.port for spec in specs] == [31100, 31110, 31120, 31100]
    lease = BranchLease("lease-1", "a1", (6,), tmp_path / "lease-1", {"SGLANG_USE_AITER": "1"}, 1)
    sandbox = first._setup.sandbox(lease)
    assert (sandbox.cards, sandbox.root) == ((6,), lease.root)
    assert sandbox.env({})["SGLANG_USE_AITER"] == "1"
    assert isinstance(built.factory(SessionStart("s1", "c0", 1, BranchTask("leads"))), AgentSession)
    with pytest.raises(ValueError, match="no specialist stands for"):
        sessions.factory(SessionStart("a1", "c0", 1, BranchTask(agent="https://example.com/agent.git")))


#: The canonical measurement as the toolbox renders it, the fields the gate's client is run with.
CANONICAL = """\
benchmark:
  framework: sglang
  model: /models/Qwen3-Next-80B-A3B-Instruct-FP8
  envs:
    TP: 2
    CONC: '64'
    ISL: 1024
    OSL: 1024
    RANDOM_RANGE_RATIO: 1
    NUM_PROMPTS: 320
    NUM_WARMUPS: 8
    RUN_EVAL: 'true'
    BENCH_TRUST_REMOTE_CODE: '1'
  benchmark_script: sglang_mi355x.sh
  inferencex_path: /cache/InferenceX
"""


def test_each_agent_s_spec_carries_its_branch_s_share_of_the_directions(tmp_path):
    agents = _agent_sessions()
    place = BranchPlace(SimpleNamespace(path=tmp_path / "worktree"), object(), tmp_path / "session")
    tasks = (BranchTask(agent="arbor", plan="/shares/share-1.json"), BranchTask(agent="arbor"))

    specs = [
        agents.session(SessionStart(branch, "c0", 1, task), place)._setup.spec
        for branch, task in zip(("a1", "a2"), tasks)
    ]

    assert [spec.plan for spec in specs] == ["/shares/share-1.json", None]


async def test_the_dossier_pass_runs_on_a_port_block_of_its_own_and_its_dossier_reaches_later_branches(
    tmp_path, monkeypatch
):
    agents = _agent_sessions()
    place = BranchPlace(SimpleNamespace(path=tmp_path / "worktree"), object(), tmp_path / "session")
    ran: list[DossierSetup] = []

    async def run(start: DossierStart, setup: DossierSetup) -> DossierOutcome:
        ran.append(setup)
        return DossierOutcome(tmp_path / "directions.json", tmp_path / "docs" / "dossier" / "qwen3-8b", MAPPED)

    monkeypatch.setattr(agent_sessions, "run_dossier_pass", run)
    outcome = await agents.dossier_pass(DossierStart("arbor", "c0", 1.0), lambda: place)
    spec = agents.session(SessionStart("a1", "c0", 1, BranchTask(agent="arbor")), place)._setup.spec

    assert outcome.reason == MAPPED
    [setup] = ran
    assert (setup.spec.port, setup.agent_dir) == (31100, tmp_path / "session" / "agent")
    assert (spec.port, spec.dossier) == (31110, str(tmp_path / "docs" / "dossier" / "qwen3-8b"))


def test_a_run_given_a_reviewed_dossier_gives_it_to_every_agent_branch(tmp_path):
    agents = replace(_agent_sessions(), dossier="/reviewed/dossier/qwen3-8b")
    place = BranchPlace(SimpleNamespace(path=tmp_path / "worktree"), object(), tmp_path / "session")

    specs = [
        agents.session(SessionStart(branch, "c0", 1, BranchTask(agent="arbor")), place)._setup.spec
        for branch in ("a1", "a2")
    ]

    assert [spec.dossier for spec in specs] == ["/reviewed/dossier/qwen3-8b"] * 2


def _flags(argv: list[str]) -> dict[str | None, object]:
    """One pass of the gate's client: its interpreter and module under ``None``, then each flag's value (``True`` for a
    switch); the ``NAME=value`` environment prefix is left out."""

    argv = [token for token in argv if "=" not in token or token.startswith("-")]
    flags: dict[str | None, object] = {None: argv[:3]}
    rest = argv[3:]
    for index, token in enumerate(rest):
        if token.startswith("--"):
            following = rest[index + 1] if index + 1 < len(rest) else "--"
            flags[token] = True if following.startswith("--") else following
    return flags


def test_each_agent_is_given_the_client_the_gate_measures_with_on_its_own_port(tmp_path):
    canonical = tmp_path / "canonical.yaml"
    canonical.write_text(CANONICAL, encoding="utf-8")
    agents = replace(_agent_sessions(), client=read_gate_client(canonical, tmp_path, 0))
    places = [
        BranchPlace(SimpleNamespace(path=tmp_path / branch / "worktree"), object(), tmp_path / branch / "session")
        for branch in ("a1", "a2")
    ]

    first, second = (
        agents.session(SessionStart(branch, "c0", 1, BranchTask(agent="agent")), place)
        for branch, place in zip(("a1", "a2"), places)
    )

    warmup, measured = (_flags(shlex.split(part)) for part in second._setup.spec.client.split(" && "))
    assert measured[None] == ["python3", "-m", "infx.bench_serving.benchmark_serving"]
    assert {key: measured[key] for key in ("--model", "--backend", "--base-url", "--dataset-name")} == {
        "--model": "/models/Qwen3-Next-80B-A3B-Instruct-FP8",
        "--backend": "vllm",
        "--base-url": "http://0.0.0.0:31110",
        "--dataset-name": "random",
    }
    assert [measured[key] for key in ("--random-input-len", "--random-output-len", "--max-concurrency")] == [
        "1024", "1024", "64",
    ]  # fmt: skip
    assert (measured["--num-prompts"], warmup["--num-prompts"]) == ("320", "8")
    assert float(measured["--random-range-ratio"]) == 1
    assert "--ignore-eos" in measured and "--trust-remote-code" in measured
    assert "http://0.0.0.0:31100" in first._setup.spec.client


def test_each_agentx_agent_is_given_the_gate_s_replay_on_its_own_port_bounded_as_the_gate_s_round(
    tmp_path, monkeypatch
):
    from hyperloom.orchestrator.actions.executors import TOOL_CACHE_ENV
    from hyperloom_multibranch.tests.test_specialist_gate_client import _agentx_session
    from inference_suite.cli import read_config

    monkeypatch.setenv(TOOL_CACHE_ENV, str(tmp_path / "tools"))
    canonical, _ = _agentx_session(tmp_path, tmp_path / "tools")
    agents = replace(_agent_sessions(), client=read_gate_client(canonical, canonical.parent.parent, 4800))
    places = [
        BranchPlace(SimpleNamespace(path=tmp_path / branch / "worktree"), object(), tmp_path / branch / "session")
        for branch in ("a1", "a2")
    ]

    specs = [
        agents.session(SessionStart(branch, "c0", 1, BranchTask(agent="agent")), place)._setup.spec
        for branch, place in zip(("a1", "a2"), places)
    ]

    for spec in specs:
        words = shlex.split(spec.client)
        config = read_config(Path(words[words.index("--config") + 1]), tmp_path / "out")
        assert (config.server.port, spec.client_seconds) == (spec.port, 4800)
    assert [spec.port for spec in specs] == [31100, 31110]


def test_a_hosted_agent_runs_in_its_lease_s_sandbox_found_again_by_the_lease_alone(tmp_path):
    sandboxes = RemoteAgentSandboxes()
    place = BranchPlace(SimpleNamespace(path=tmp_path / "worktree"), object(), tmp_path / "a1" / "session")
    handle = SimpleNamespace(sandbox_id="sbx-1")
    hosted = HostedBranchLease("lease-1", "a1", (0, 1), Path("/app/work"), {}, 6, "dev-2", 3600, handle)

    first = sandboxes.sandbox("a1", place, tmp_path / "src", hosted)
    again = sandboxes.sandbox(
        "a1", place, tmp_path / "src", BranchLease("lease-1", "a1", (0, 1), Path("/app/work"), {}, 6)
    )

    assert first.sandbox is again.sandbox is handle and again.root == place.workspace / "sandbox"
    assert again.cards == 2 and again.python_path == (tmp_path / "src",)
    assert sandboxes.claude_config(place) == place.workspace / "sandbox" / "home" / ".claude"
    with pytest.raises(SandboxGone):
        sandboxes.sandbox("a1", place, tmp_path / "src", BranchLease("lease-2", "a1", (0,), tmp_path, {}, 6))


def test_where_bubblewrap_sandboxes_an_agent_its_sandbox_writes_its_branch_s_trees_and_keeps_its_own_claude_dir(
    tmp_path: Path,
) -> None:
    asked: list[object] = []
    isolation = BwrapIsolation(BwrapProbe("/usr/bin/bwrap", (0, 13, 1), ()))

    def choose() -> BwrapIsolation:
        asked.append(isolation)
        return isolation

    worktree = SimpleNamespace(path=tmp_path / "worktree", linked=(tmp_path / "base" / "3rdparty",))
    for path in (worktree.path, *worktree.linked, tmp_path / "branch" / "session"):
        path.mkdir(parents=True)
    place = BranchPlace(worktree, object(), tmp_path / "branch" / "session")
    sandboxes = LocalAgentSandboxes(object(), choose=choose)
    lease = BranchLease("lease-1", "a1", (6,), tmp_path / "lease-1", {}, 1)

    sandbox = sandboxes.sandbox("a1", place, worktree.path / "python", lease)
    try:
        home = place.workspace / "sandbox" / "home"
        assert sandbox.isolation is isolation and sandbox.paths.home == home
        assert sandbox.paths.writable == (place.workspace, worktree.path)
        assert sandbox.paths.readable == worktree.linked
        assert sandboxes.claude_config(place) == home / ".claude" == Path(sandbox.env({})["CLAUDE_CONFIG_DIR"])
    finally:
        asyncio.run(sandbox.stop())
    assert len(asked) == 1, "the session asks once how its sandboxes run"
    unconfined = LocalAgentSandboxes(object(), choose=lambda: UNCONFINED)
    assert unconfined.claude_config(place) != home / ".claude"
