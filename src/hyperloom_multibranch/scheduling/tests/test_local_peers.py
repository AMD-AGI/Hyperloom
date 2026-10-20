# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The real peers of a session: built inside the session directory on model-backed specialist sessions and on this
machine's measurement host, and refused when this machine has no agent CLI for them, its pins cannot be read or the
image lacks the framework's package."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import urllib.request
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from hyperloom.common.benchmark_config import BENCHMARK_CONFIG_ENV, benchmark_config_path, is_magpie_agentx
from hyperloom.contracts.measurement import AttachmentRef
from hyperloom.inference_optimizer import inferencex_checkout
from hyperloom.inference_optimizer.cli.benchmark_config import configure_benchmark
from hyperloom.inference_optimizer.cli.parser import _build_parser
from hyperloom.contracts.recipe import Recipe, RecipeDelta
from hyperloom.common.env import forge_explicitly_enabled
from hyperloom.common.perf_metric import agentx_enabled
from hyperloom.common.visible_devices import COUNTING_VISIBLE_DEVICE_VARS
from hyperloom.measure import compile as compile_module
from hyperloom.measure.compile import ComponentLayout, LocalBlobStore, PackageBase, UnsupportedComponentError
from hyperloom.measure.hosted_spec import SpecContext
from hyperloom.measure.pins import ActiveImage, pinned_tool_commits
from hyperloom_multibranch.contracts.gate import RegressionOutcome
from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH, Role, Writer
from hyperloom_multibranch.contracts.notebook.records import LineageEvent
from hyperloom_multibranch.contracts.notebook.vocab import Conclusion, InboxKind, LineageKind
from hyperloom_multibranch.notebook import STALL_TESTS, LocalNotebook
from hyperloom_multibranch.notebook.tests.factories import intent, judgement, request, result
from hyperloom_multibranch.contracts.toolbox import SCOPE_SPECIALIST, SpecialistGrant
from hyperloom_multibranch.repo.git_repo import GitExperimentRepo
from hyperloom_multibranch.rulegate.critic import NoCritic
from hyperloom_multibranch.rulegate.fix_case import FIX_CASE_ID, FixCase, read_case
from hyperloom_multibranch.rulegate.gate import RuleGate
from hyperloom_multibranch.rulegate.regression import RegressionSet
from hyperloom_multibranch.rulegate.rules import AgentXRule, InferenceXRule
from hyperloom_multibranch.rulegate.service import BASELINE_TESTS, GateService
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.decisions import DecisionLog
from hyperloom_multibranch.scheduling.forks import ForkDesk
from hyperloom_multibranch.scheduling.layout import SESSION_PREFIX, SessionLayout, create_session
from hyperloom_multibranch.scheduling import local_peers as local_peers_module
from hyperloom_multibranch.scheduling.local_peers import (
    AGENTX_ENV,
    InferenceXCheckout,
    MeasurementHost,
    SessionWiring,
    local_peers,
    machine_cards,
    machine_parallel_tests,
    measurement_host,
    require_framework_package,
    specialist_command,
    workload_environment,
    workload_rule,
    _specialist_role,
)
from hyperloom_multibranch.scheduling.peers import Peers, PeerSetup, PeersUnavailable
from hyperloom_multibranch.scheduling.prelude import prelude_argv
from hyperloom_multibranch.scheduling.processes import ProcessRegistry
from hyperloom_multibranch.scheduling.roster import Roster, RosterCaps
from hyperloom_multibranch.scheduling.settings import InferenceXShape, KernelOptimizer, Workload
from hyperloom_multibranch.specialist.model_sessions import ModelSessions
from hyperloom_multibranch.specialist.runtime import SessionStart, SpecialistRuntime
from hyperloom_multibranch.specialist.worktree import BranchWorktree
from hyperloom_multibranch.toolbox import local_hosted
from hyperloom_multibranch.toolbox.local_hosted import LocalHostedToolbox
from hyperloom_multibranch.toolbox.mcp_tools import McpSpecialistTokens
from hyperloom_multibranch.toolbox.queue import QueuePolicy

WORKLOAD = Workload(
    "/models/Llama-3-8B", "sglang", 2, 16, "bf16", "mi355x", KernelOptimizer.FORGE, InferenceXShape(1024, 512)
)
AGENTX = replace(WORKLOAD, inferencex=None)
IMAGE = ActiveImage("registry.example/sglang:v1", "sha256:" + "2" * 64)
HOST = MeasurementHost((0, 1, 2, 3), IMAGE, "a" * 40, InferenceXCheckout("b" * 40, "/opt/InferenceX"))
IMAGE_VARS = ("HYPERLOOM_ACTIVE_IMAGE_REF", "HYPERLOOM_ACTIVE_IMAGE_DIGEST")


def _call_tool(grant: SpecialistGrant, name: str, arguments: dict[str, str] | None = None) -> dict[str, object]:
    message = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments or {}},
    }
    request = urllib.request.Request(
        grant.url,
        data=json.dumps(message).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {grant.token}"},
    )
    with urllib.request.urlopen(request, timeout=10) as reply:
        return json.loads(reply.read())


def _list_tools(grant: SpecialistGrant) -> set[str]:
    message = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    request = urllib.request.Request(
        grant.url,
        data=json.dumps(message).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {grant.token}"},
    )
    with urllib.request.urlopen(request, timeout=10) as reply:
        return {tool["name"] for tool in json.loads(reply.read())["result"]["tools"]}


def test_the_workload_reaches_the_renderer_as_hyperloom_optimize_exports_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HL_MODEL_BASE", raising=False)
    assert workload_environment(WORKLOAD) == {
        "FRAMEWORK": "sglang",
        "MODEL_PATH": "/models/Llama-3-8B",
        "TP": "2",
        "ISL": "1024",
        "OSL": "512",
        "PRECISION": "bf16",
        "GPU_TYPE": "mi355x",
        "TARGET_GPU_TYPE": "mi355x",
        "KERNEL_OPT_BACKEND_ORDER": "forge",
    }


def test_an_agentx_workload_reaches_the_renderer_and_the_prelude_child_as_agentx_without_sequence_lengths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HL_MODEL_BASE", raising=False)
    environment = workload_environment(AGENTX)

    assert not {"ISL", "OSL"} & set(environment)
    assert environment[AGENTX_ENV] == "1" and agentx_enabled(environment)
    assert AGENTX_ENV not in workload_environment(WORKLOAD)


def test_the_rule_follows_the_workload(tmp_path: Path) -> None:
    notebook = LocalNotebook(tmp_path, Writer(Role.GATE, "rule-gate"))

    assert workload_rule(WORKLOAD, notebook) == InferenceXRule(notebook, WORKLOAD.conc)
    assert isinstance(workload_rule(AGENTX, notebook), AgentXRule)


def test_the_specialists_are_told_an_agentx_workload_and_no_sequence_lengths() -> None:
    assert _specialist_role(WORKLOAD) == "Llama-3-8B on sglang, bf16, TP 2, ISL 1024, OSL 512, concurrency 16, mi355x"
    assert _specialist_role(AGENTX) == "Llama-3-8B on sglang, bf16, TP 2, AgentX trace replay, concurrency 16, mi355x"


@pytest.mark.parametrize("optimizer", list(KernelOptimizer))
def test_the_kernel_optimizer_flag_picks_the_backend_kernel_tuning_runs(
    optimizer: KernelOptimizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in workload_environment(replace(WORKLOAD, kernel_optimizer=optimizer)).items():
        monkeypatch.setenv(name, value)
    assert forge_explicitly_enabled() is (optimizer is KernelOptimizer.FORGE)


def test_the_model_is_exported_as_the_directory_a_server_loads_it_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "Llama-3-8B").mkdir()
    monkeypatch.setenv("HL_MODEL_BASE", str(tmp_path))
    hub_id = Workload(
        "meta-llama/Llama-3-8B", "vllm", 1, 1, "fp8", "mi355x", KernelOptimizer.FORGE, InferenceXShape(1, 1)
    )
    assert workload_environment(hub_id)["MODEL_PATH"] == str(tmp_path / "Llama-3-8B")


def test_the_specialists_run_the_agent_cli_found_on_path(tmp_path: Path) -> None:
    cli = tmp_path / "claude"
    cli.write_text("#!/bin/sh\nexit 0\n")
    cli.chmod(0o755)

    command = specialist_command(
        {"PATH": str(tmp_path), "ANTHROPIC_API_KEY": "dummy", "CLAUDE_MODEL": "claude-sonnet-5-5"}
    )

    assert (command.backend, command.executable, command.model) == ("claude", str(cli), "claude-sonnet-5-5")


def test_without_the_agent_cli_the_peers_are_refused(tmp_path: Path) -> None:
    with pytest.raises(PeersUnavailable, match="claude CLI the specialists run is not on PATH"):
        specialist_command({"PATH": str(tmp_path), "ANTHROPIC_API_KEY": "dummy"})


def test_a_specialist_model_the_price_sheet_lacks_runs_on_its_tokens_and_says_so(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    cli = tmp_path / "claude"
    cli.write_text("#!/bin/sh\nexit 0\n")
    cli.chmod(0o755)

    command = specialist_command({"PATH": str(tmp_path), "ANTHROPIC_API_KEY": "dummy", "CLAUDE_MODEL": "glm-5-3"})

    assert command.model == "glm-5-3"
    assert "the specialists run on glm-5-3, which the price sheet does not list" in caplog.text


@pytest.fixture
def four_cards(monkeypatch: pytest.MonkeyPatch) -> None:
    """This machine shows four cards, the upper half of an eight-card node."""

    for name in COUNTING_VISIBLE_DEVICE_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "4,5,6,7")


def _inferencex_checkout(root: Path) -> Path:
    (root / "benchmarks").mkdir(parents=True)
    (root / "benchmarks" / "benchmark_lib.sh").write_text("")
    return root


@pytest.mark.usefixtures("four_cards")
def test_the_host_is_this_machines_cards_on_the_launchers_image_and_the_pinned_tool_commits(tmp_path: Path) -> None:
    checkout = _inferencex_checkout(tmp_path / "InferenceX")
    env = {IMAGE_VARS[0]: IMAGE.ref, IMAGE_VARS[1]: IMAGE.digest, "INFERENCEX_PATH": str(checkout)}

    host = measurement_host(env, WORKLOAD)

    tools = pinned_tool_commits({})
    assert host == MeasurementHost(
        (4, 5, 6, 7), IMAGE, tools.magpie, InferenceXCheckout(tools.inferencex, str(checkout))
    )
    assert machine_parallel_tests(2) == 2


@pytest.mark.usefixtures("four_cards")
def test_a_machine_without_an_inferencex_checkout_refuses_the_peers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(inferencex_checkout, "detect", lambda: "")
    env = {IMAGE_VARS[0]: IMAGE.ref, IMAGE_VARS[1]: IMAGE.digest}

    with pytest.raises(PeersUnavailable, match="no InferenceX checkout"):
        measurement_host(env, WORKLOAD)
    with pytest.raises(PeersUnavailable, match=f"no InferenceX checkout .*{tmp_path.name}"):
        measurement_host({**env, "INFERENCEX_PATH": str(tmp_path)}, WORKLOAD)


def _git_checkout(root: Path) -> str:
    """A Magpie-layout InferenceX repository at ``root``, its project in ``inferencex-e2e``; its HEAD commit."""

    _inferencex_checkout(root / "inferencex-e2e")
    git = ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "c"], check=True)
    return subprocess.run([*git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def _magpie(monkeypatch: pytest.MonkeyPatch, default_dir: Path, **pin: str) -> None:
    """The installed Magpie, as its InferenceX module: ``pin`` its ``INFERENCEX_DEFAULT_REF`` (absent when empty),
    ``default_dir`` its default clone location, and its root resolver taking the ``inferencex-e2e`` project."""

    def resolve_inferencex_root(path: str) -> str:
        project = Path(path) / "inferencex-e2e"
        if not (project / "benchmarks" / "benchmark_lib.sh").is_file():
            raise RuntimeError(f"Expected one supported InferenceX project at {path}")
        return str(project)

    module = SimpleNamespace(
        **pin,
        InferenceXManager=lambda: SimpleNamespace(default_dir=str(default_dir)),
        resolve_inferencex_root=resolve_inferencex_root,
    )
    monkeypatch.setattr(local_peers_module, "_magpie_inferencex", lambda: module)


@pytest.mark.usefixtures("four_cards")
def test_an_agentx_host_measures_on_the_checkout_at_the_commit_magpie_pins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    default, named = tmp_path / "default", tmp_path / "named"
    default_commit, named_commit = _git_checkout(default), _git_checkout(named)
    env = {IMAGE_VARS[0]: IMAGE.ref, IMAGE_VARS[1]: IMAGE.digest, "INFERENCEX_PATH": str(tmp_path / "synthetic")}
    tools = pinned_tool_commits({})

    _magpie(monkeypatch, default, INFERENCEX_DEFAULT_REF=default_commit)
    assert measurement_host(env, AGENTX) == MeasurementHost(
        (4, 5, 6, 7), IMAGE, tools.magpie, InferenceXCheckout(default_commit, str(default / "inferencex-e2e"))
    )
    _magpie(monkeypatch, default, INFERENCEX_DEFAULT_REF=named_commit)
    host = measurement_host({**env, "MAGPIE_INFERENCEX_PATH": str(named)}, AGENTX)
    assert host.inferencex == InferenceXCheckout(named_commit, str(named / "inferencex-e2e"))


@pytest.mark.usefixtures("four_cards")
def test_an_agentx_host_refuses_a_checkout_off_the_magpie_pin_and_a_magpie_without_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _git_checkout(tmp_path / "InferenceX")
    env = {IMAGE_VARS[0]: IMAGE.ref, IMAGE_VARS[1]: IMAGE.digest}
    pinned = "4" * 40

    _magpie(monkeypatch, tmp_path / "InferenceX", INFERENCEX_DEFAULT_REF=pinned)
    with pytest.raises(PeersUnavailable, match=f"no InferenceX checkout at {pinned}.* at {tmp_path}/InferenceX"):
        measurement_host(env, AGENTX)
    with pytest.raises(PeersUnavailable, match=f"no InferenceX checkout at {pinned}.* at {tmp_path}/missing"):
        measurement_host({**env, "MAGPIE_INFERENCEX_PATH": str(tmp_path / "missing")}, AGENTX)
    _magpie(monkeypatch, tmp_path / "InferenceX")
    with pytest.raises(PeersUnavailable, match=f"INFERENCEX_DEFAULT_REF.*MAGPIE_REF {pinned_tool_commits({}).magpie}"):
        measurement_host(env, AGENTX)


@pytest.mark.parametrize("mask", ["ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"])
def test_the_cards_are_the_ids_the_visible_device_mask_names(mask: str, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in COUNTING_VISIBLE_DEVICE_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(mask, "6,2")

    assert machine_cards() == (6, 2)


@pytest.mark.parametrize("framework", ["tensorrt-llm", "sglang"])
def test_a_framework_whose_source_cannot_be_laid_out_refuses_the_peers(
    framework: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def not_installed(layout: ComponentLayout) -> PackageBase:
        raise UnsupportedComponentError(f"package {layout.import_name} is not installed in this image")

    monkeypatch.setattr(compile_module, "locate_installed_package", not_installed)

    with pytest.raises(PeersUnavailable, match=f"{framework} .*(no package layout|not installed in this image)"):
        require_framework_package(framework)


@pytest.mark.usefixtures("four_cards")
def test_a_session_nobody_names_the_image_of_refuses_the_peers() -> None:
    with pytest.raises(PeersUnavailable, match=f"{IMAGE_VARS[0]}, {IMAGE_VARS[1]} not set"):
        measurement_host({}, WORKLOAD)


@pytest.mark.parametrize(
    ("env", "named"),
    [
        ({IMAGE_VARS[0]: IMAGE.ref}, IMAGE_VARS[1]),
        ({IMAGE_VARS[0]: IMAGE.ref, IMAGE_VARS[1]: "sha256:" + "0" * 64}, "not a real sha256"),
        ({IMAGE_VARS[0]: IMAGE.ref, IMAGE_VARS[1]: IMAGE.digest, "MAGPIE_REF": "main"}, "MAGPIE_REF"),
    ],
    ids=["half-named-image", "zero-digest", "moving-tool-ref"],
)
@pytest.mark.usefixtures("four_cards")
def test_pins_that_cannot_be_read_refuse_the_peers(env: dict[str, str], named: str) -> None:
    with pytest.raises(PeersUnavailable, match=named):
        measurement_host(env, WORKLOAD)


def test_a_machine_that_shows_no_card_fits_no_test(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "")
    with pytest.raises(ValueError, match="0 cards"):
        machine_parallel_tests(1)


@dataclass(frozen=True)
class _Built:
    layout: SessionLayout
    setup: PeerSetup
    peers: Peers
    wiring: SessionWiring


@asynccontextmanager
async def _peers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workload: Workload, host: MeasurementHost = HOST
) -> AsyncIterator[_Built]:
    """The real peers of a session of ``workload`` on ``host`` in ``tmp_path``, with the wiring their specialist
    sessions were built on; every variable the build exports is restored afterwards."""

    for name in (*workload_environment(workload), BENCHMARK_CONFIG_ENV):
        monkeypatch.setenv(name, "")
    monkeypatch.delenv("GPU_TYPE", raising=False)
    monkeypatch.delenv("HYPERLOOM_KB_URL", raising=False)
    clock = Clock()
    layout = create_session(tmp_path, workload.model_basename, clock)
    desk = ForkDesk(Roster(RosterCaps(1, 1, 1), layout.roster, clock), DecisionLog(layout.decisions, clock), clock)
    blobs = LocalBlobStore(layout.blobs)
    setup = PeerSetup(layout, workload, desk, ProcessRegistry(layout.pids, clock), blobs, NoCritic())
    wired: list[SessionWiring] = []

    def sessions(wiring: SessionWiring) -> ModelSessions:
        wired.append(wiring)

        def factory(start: SessionStart) -> object:
            raise AssertionError("no specialist runs here")

        def worktrees(branch: str) -> BranchWorktree:
            raise AssertionError("no worktree is used here")

        return ModelSessions(factory, worktrees)

    peers = local_peers(host, sessions, setup)
    [wiring] = wired
    try:
        yield _Built(layout, setup, peers, wiring)
    finally:
        peers.mcp.stop()
        await peers.toolbox.close()


@pytest.fixture
async def built(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[_Built]:
    """The real peers of an InferenceX session in ``tmp_path``."""

    monkeypatch.delenv("HL_MODEL_BASE", raising=False)
    async with _peers(tmp_path, monkeypatch, WORKLOAD) as peers:
        yield peers


async def test_an_agentx_session_measures_the_agentx_benchmark_its_prelude_child_is_bound_to(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "models" / "Llama-3-8B").mkdir(parents=True)
    monkeypatch.setenv("HL_MODEL_BASE", str(tmp_path / "models"))
    workload = replace(AGENTX, model="meta-llama/Llama-3-8B")
    host = replace(HOST, inferencex=InferenceXCheckout("c" * 40, "/opt/InferenceX-agentx/inferencex-e2e"))
    specs: list[SpecContext] = []
    build = local_hosted.build_local_toolbox

    def signed(*args: object) -> LocalHostedToolbox:
        specs.append(args[-1])
        return build(*args)

    monkeypatch.setattr(local_hosted, "build_local_toolbox", signed)

    async with _peers(tmp_path / "workspace", monkeypatch, workload, host) as built:
        toolbox = built.layout.toolbox
        assert benchmark_config_path() == (toolbox / "benchmark.yaml").resolve()
        canonical = yaml.safe_load((toolbox / "canonical" / "canonical.yaml").read_text())["benchmark"]
        assert is_magpie_agentx(canonical) and canonical["agentx"]["mode"] == "canonical"
        assert canonical["inferencex_path"] == host.inferencex.path
        assert [(spec.magpie_commit, spec.inferencex_commit) for spec in specs] == [
            (host.magpie_commit, host.inferencex.commit)
        ]
        assert (canonical["model"], canonical["envs"]["MODEL_PATH"]) == (
            workload.model,
            str(tmp_path / "models" / "Llama-3-8B"),
        )
        assert (str(canonical["envs"]["TP"]), str(canonical["envs"]["CONC"])) == (str(workload.tp), str(workload.conc))
        argv = prelude_argv(workload, 1.0, built.layout.prelude_launch)
        args = _build_parser().parse_args(list(argv[argv.index("optimize") :]))
        configure_benchmark(args)
        assert (args.model, args.framework, args.tp, args.ep, args.conc, args.precision) == (
            workload.model,
            workload.framework,
            workload.tp,
            1,
            workload.conc,
            workload.precision,
        )


async def test_the_real_peers_live_inside_the_session_directory(built: _Built) -> None:
    layout, setup, peers, wiring = built.layout, built.setup, built.peers, built.wiring
    assert (wiring.setup, wiring.repo, wiring.briefs) == (setup, peers.repo, peers.briefs)
    assert isinstance(peers.lifecycle, SpecialistRuntime) and peers.briefs.warm == ()
    assert peers.briefs.repo is peers.repo
    assert isinstance(peers.toolbox, LocalHostedToolbox) and isinstance(peers.repo, GitExperimentRepo)
    canonical = yaml.safe_load((layout.toolbox / "canonical" / "canonical.yaml").read_text())["benchmark"]
    assert (canonical["model"], canonical["runner_type"], canonical["inferencex_path"]) == (
        WORKLOAD.model,
        WORKLOAD.gpu_type,
        HOST.inferencex.path,
    )
    assert not (layout.toolbox / "measurement-config.json").exists()
    assert benchmark_config_path() is None and not (layout.toolbox / "benchmark.yaml").exists()
    assert peers.base_image_digest == IMAGE.digest
    assert peers.mcp.url.startswith("http://127.0.0.1:")
    assert {name: os.environ[name] for name in workload_environment(WORKLOAD)} == workload_environment(WORKLOAD)
    root = await peers.repo.init_root(Recipe(IMAGE.digest))
    assert await peers.repo.head(INTEGRATION_BRANCH) == root
    patch = await peers.repo.store_patch("sglang", "--- a/x\n+++ b/x\n")
    assert LocalBlobStore(layout.blobs).get(patch.sha256) == b"--- a/x\n+++ b/x\n"
    assert isinstance(peers.gate, GateService) and await peers.gate.baseline_test_count() == BASELINE_TESTS
    await peers.toolbox.start()
    await peers.toolbox.set_policy(QueuePolicy(per_branch_k=1, global_concurrency=2))
    state = await peers.toolbox.queue_state()
    assert (state.queued, state.running) == (0, 0)
    assert isinstance(peers.reserved, RuleGate) and not await peers.reserved.land_split("b09")
    assert isinstance(peers.regression, RegressionSet)
    scheduler = LocalNotebook(layout.notebook, Writer(Role.SCHEDULER, "test-scheduler"))
    await scheduler.record_lineage(LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id=root))
    assert await peers.regression.check(root, RecipeDelta(), sources=()) == RegressionOutcome(True)
    state = await peers.toolbox.queue_state()
    assert (state.queued, state.running) == (0, 0)
    assert isinstance(wiring.tokens, McpSpecialistTokens)
    grant = wiring.tokens.issue("b01", frozenset({SCOPE_SPECIALIST}))
    assert grant.url == peers.mcp.url
    reply = await asyncio.to_thread(_call_tool, grant, "queue_state")
    assert reply["result"]["isError"] is False
    tools = await asyncio.to_thread(_list_tools, grant)
    served = {
        "worktree_sync",
        "worktree_check",
        "worktree_patch",
        "notebook_brief",
        "request_repair",
        "decline_repair",
        "fix_case",
    }
    assert served <= tools
    assert all(entry.name.startswith(SESSION_PREFIX) for entry in layout.root.iterdir())


async def test_a_fix_case_a_specialist_declares_is_stored_in_the_session_blob_store(built: _Built) -> None:
    grant = built.wiring.tokens.issue("b01", frozenset({SCOPE_SPECIALIST}))
    case = FixCase("#!/bin/bash\nexit 0\n", "fp8 scales overflow at 8k context; clamp them before the cast")

    reply = await asyncio.to_thread(
        _call_tool, grant, "fix_case", {"repro_sh": case.repro_sh, "diagnosis": case.diagnosis}
    )

    assert reply["result"]["isError"] is False
    ref = AttachmentRef.from_document(json.loads(reply["result"]["content"][0]["text"]))
    assert (ref.id, ref.kind) == (FIX_CASE_ID, "dir")
    assert read_case(built.setup.blobs.get(ref.sha256)) == case


async def test_the_specialist_notebook_handle_briefs_a_branch_after_stall_tests_judged_tests_without_a_commit(
    built: _Built,
) -> None:
    root = built.layout.notebook
    scheduler = LocalNotebook(root, Writer(Role.SCHEDULER, "test-scheduler"))
    await scheduler.record_lineage(LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id="c0"))
    await scheduler.record_lineage(LineageEvent(LineageKind.OPEN, "b01", from_commit="c0"))
    toolbox = LocalNotebook(root, Writer(Role.TOOLBOX, "test-toolbox"))
    gate = LocalNotebook(root, Writer(Role.GATE, "test-gate"))
    handle = built.wiring.notebooks("b01")
    assert isinstance(handle, LocalNotebook)

    async def judged(count: int) -> list[str]:
        for n in range(count):
            request_id = f"r{n}-{count}"
            await toolbox.write_request(request(request_id, intent("b01", "c0", target=request_id)))
            await toolbox.write_result(result(request_id))
            await gate.write_judgement(request_id, judgement(0.1, (request_id,)), Conclusion.RECORDED_ONLY, None)
        return [item.summary for item in (await handle.poll("b01")).items if item.kind is InboxKind.BRIEF]

    assert await judged(STALL_TESTS - 1) == []
    assert [summary.split(":")[0] for summary in await judged(1)] == ["stall"]
