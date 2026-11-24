# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What protects an Implementer lane, and who is asked to guarantee it.

A lane edits its own copy of the workspace, and what it leaves is applied to the
canonical tree and measured there by the loop. The measurement surface still has
to survive the session: a lane diff that touches the driver, the harness or the
oracle is refused at the boundary, and the refusal takes the implementation
edits in the same diff with it -- after the session has already been paid for in
full. A hook denies that edit while the agent is still working, which is the
only point at which the rest of the session can still be saved.

A round's lanes share one device. A lane's session sees none of its own: every
command that needs one goes through the lane's runner, which waits its turn, and
the stop check that measures the lane's candidate waits in the same queue.

That protection, and the private build cache a lane compiles into, are both
things the provider does on the lane's behalf. Neither is checked by the lane
code, so the second half of this file is about the provider being made to
declare them before a round of lanes is allowed to start.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import kernelforge.workflows.forge_loop.gate.insession_gate as gate_module
import kernelforge.workflows.forge_loop.roles.agent as agent_module
from kernelforge.agent_backends.base import (
    AgentCapabilities,
    AgentRunResult,
    AgentRunSpec,
)
from kernelforge.agent_backends.registry import AgentProvider, register_agent_provider
from kernelforge.workflows.forge_loop.bootstrap import lanes as cli_lanes
from kernelforge.workflows.forge_loop.bootstrap.errors import ForgeLoopError
from kernelforge.config import Config
from kernelforge.utils.device_queue import RUNNER_CONFIG_NAME, RUNNER_NAME
from kernelforge.workflows.forge_loop.iteration.fanout import LaneAssignment, LaneSeat
from kernelforge.evaluation.scoring import GateScoring


@pytest.fixture(autouse=True)
def _isolate_provider_registry(isolated_provider_registry):
    """Apply the shared registry isolation to every test in this module."""


def _campaign(tmp_path: Path) -> tuple[Config, Path]:
    """A campaign workspace with a driver and one declared source file."""
    workspace = tmp_path / "workspace"
    _tree(workspace)
    config = Config(
        workspace=str(workspace),
        agent_backend="claude",
        agent_model="claude-test",
        agent_precheck=False,
    )
    return config, workspace


def _tree(root: Path) -> Path:
    """Write the driver and kernel layout a lane copy is given."""
    (root / "src").mkdir(parents=True)
    (root / "forge_driver.py").write_text("print('allclose: True')\n")
    (root / "src" / "kernel.py").write_text("VALUE = 0\n")
    return root


def _record_backend_specs(monkeypatch, *, stop_hooks: bool = True) -> list[AgentRunSpec]:
    """Route every implementer session to a backend that records its spec."""
    specs: list[AgentRunSpec] = []

    class Backend:
        """Stand in for a provider without running one."""

        name = "claude"
        capabilities = AgentCapabilities(resumable=True, stop_hooks=stop_hooks)

        def __init__(self, runtime):
            self.runtime = runtime

        async def run(self, spec: AgentRunSpec, usage=None) -> AgentRunResult:
            """Record what this session would have been started with."""
            specs.append(spec)
            return AgentRunResult(text="PLAN: tune it")

    monkeypatch.setattr(
        agent_module,
        "create_registered_backend",
        lambda runtime, **_kwargs: Backend(runtime),
    )
    return specs


def _lane_factory(tmp_path: Path):
    """Build the factory a round uses to bind one session to one lane."""
    config, workspace = _campaign(tmp_path)
    return cli_lanes._make_lane_agent_factory(
        make_agent=agent_module.make_agent_fn,
        config=config,
        binding=cli_lanes.LaneBinding(
            workspace_dir=str(workspace),
            driver=str(workspace / "forge_driver.py"),
            source_files=[str(workspace / "src" / "kernel.py")],
        ),
        setup=agent_module.ImplementerSetup(
            agent_module.ImplementerTask(program_md="Optimize VALUE."),
            session=agent_module.SessionOptions(agent_backend="claude"),
        ),
    )


def _seat(tmp_path: Path, lane_dir: Path) -> LaneSeat:
    """Where a round seats one lane: its copy, and the runner and queue it reaches the device through."""
    return LaneSeat(
        assignment=LaneAssignment("1"),
        lane_dir=lane_dir,
        device_runner=lane_dir / RUNNER_NAME,
        queue_root=tmp_path / "device-queue",
        usage_log=tmp_path / "device_usage.jsonl",
    )


def _run_lane_session(
    tmp_path: Path,
    monkeypatch,
    *,
    gain_required: bool = False,
    stop_hooks: bool = True,
) -> tuple[AgentRunSpec, Path]:
    """Run one lane session the way a round runs it."""
    specs = _record_backend_specs(monkeypatch, stop_hooks=stop_hooks)
    factory = _lane_factory(tmp_path)
    lane_dir = _tree(tmp_path / "lanes" / "1")

    session = factory(_seat(tmp_path, lane_dir), gain_required=gain_required)
    asyncio.run(session(str(lane_dir / "src" / "kernel.py"), "plan 1"))

    assert len(specs) == 1
    return specs[0], lane_dir


@pytest.mark.parametrize("fails", [False, True])
def test_every_lane_round_releases_the_backend_it_built(tmp_path, monkeypatch, fails):
    """Each round builds a fresh backend per lane, so nothing but the lane can release it.

    The DeepSeek backend keeps one harness process per workspace for resumes. A
    lane session is never resumed once it returns, and the next round builds a
    new backend that cannot see the old one's processes.
    """
    backends = []

    class Backend:
        """Stand in for a provider that holds a process between sessions."""

        name = "claude"
        capabilities = AgentCapabilities(resumable=True, stop_hooks=True)

        def __init__(self, runtime):
            self.runtime = runtime
            self.closed = False
            backends.append(self)

        async def run(self, spec: AgentRunSpec, usage=None) -> AgentRunResult:
            """Answer, or fail the way a lost provider does."""
            if fails:
                raise RuntimeError("provider lost")
            return AgentRunResult(text="PLAN: tune it")

        def close(self) -> None:
            """Record the release."""
            self.closed = True

    monkeypatch.setattr(agent_module, "create_registered_backend", lambda runtime, **_kwargs: Backend(runtime))
    factory = _lane_factory(tmp_path)

    for round_number in (1, 2):
        lane_dir = _tree(tmp_path / f"round{round_number}" / "1")
        session = factory(_seat(tmp_path, lane_dir), gain_required=True)
        if fails:
            with pytest.raises(RuntimeError):
                asyncio.run(session(str(lane_dir / "src" / "kernel.py"), "plan"))
        else:
            asyncio.run(session(str(lane_dir / "src" / "kernel.py"), "plan"))

    assert len(backends) == 2
    assert all(backend.closed for backend in backends)


def test_a_lane_session_is_given_no_extra_directory(tmp_path, monkeypatch):
    spec, _lane_dir = _run_lane_session(tmp_path, monkeypatch)

    assert spec.additional_directories == []
    assert "## Reasoning log" not in spec.user_prompt
    assert "reasoning log" not in spec.system_prompt.lower()


def test_a_lane_session_sink_reaches_an_agent_that_declares_it(tmp_path, monkeypatch):
    """The lesson turn resumes the lane session, so the sink has to reach the agent that names it."""
    _record_backend_specs(monkeypatch)
    lane_dir = _tree(tmp_path / "lanes" / "1")
    session = _lane_factory(tmp_path)(_seat(tmp_path, lane_dir), gain_required=True)
    sink: dict = {}

    asyncio.run(session(str(lane_dir / "src" / "kernel.py"), "plan 2", session_sink=sink))

    assert sink["session_started"] is True
    assert sink["end_reason"]
    assert "progress_log" in sink


def test_a_hookless_provider_is_given_no_hooks_to_drop(tmp_path, monkeypatch):
    """The gate's callbacks are built for the provider that runs them, or not at all.

    A backend that ignores ``AgentRunSpec.hooks`` drops the whole group without
    a word. Attaching one anyway made the call site read as protection the
    session does not have, which is the inference the lane gate's old refusal
    was there to prevent and the one this leaves no room for.
    """
    spec, _lane_dir = _run_lane_session(tmp_path, monkeypatch, stop_hooks=False)

    assert spec.hooks is None
    # The rest of the lane's protection is unchanged: it does not travel in the
    # hooks, and none of it depends on the provider running them.
    assert spec.driver_script
    assert spec.target_files
    assert spec.allow_dirty_baseline is True


def test_a_lane_session_is_given_the_protected_path_hooks(tmp_path, monkeypatch):
    """A lane must reach its provider carrying the gate's edit and Bash denials."""
    spec, _lane_dir = _run_lane_session(tmp_path, monkeypatch)

    assert spec.hooks is not None
    assert {hook.matcher for hook in spec.hooks.pre_tool_use} == {
        gate_module._EDIT_TOOL_MATCHER,
        "Bash",
    }
    assert spec.hooks.post_tool_use == []


def test_a_lane_session_keeps_the_stop_check_and_queues_it_for_the_device(tmp_path, monkeypatch):
    """A lane is the same gated session a round's only session is; what differs is that its stop check waits its turn."""
    spec, lane_dir = _run_lane_session(tmp_path, monkeypatch)

    assert len(spec.hooks.stop) == 1
    gate = spec.hooks.stop[0].callback.__self__
    assert gate.device.queue_root == str(tmp_path / "device-queue")
    assert gate.device.runner == str(lane_dir / RUNNER_NAME)
    assert "ONE self-correcting session" in spec.system_prompt


def test_a_lane_is_told_to_reach_the_gpu_through_its_runner(tmp_path, monkeypatch):
    spec, lane_dir = _run_lane_session(tmp_path, monkeypatch)
    runner = lane_dir / RUNNER_NAME

    assert "## The GPU is shared: run every command that needs it through the device runner" in spec.system_prompt
    assert f"python3 {runner} <command> [args...]" in spec.system_prompt
    # The driver commands the session is shown already go through it.
    assert f"python3 {runner} python3 forge_driver.py --warmup 3 --iters 20 --bench-mode" in spec.system_prompt


def test_a_lane_hook_denies_a_command_that_selects_a_device_around_the_queue(tmp_path, monkeypatch):
    spec, lane_dir = _run_lane_session(tmp_path, monkeypatch)
    deny_bash = next(hook.callback for hook in spec.hooks.pre_tool_use if hook.matcher == "Bash")

    def decide(command: str) -> dict:
        return asyncio.run(deny_bash({"tool_name": "Bash", "tool_input": {"command": command}}, None, None))

    denied = decide("HIP_VISIBLE_DEVICES=0 python3 bench.py")
    assert denied["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert str(lane_dir / RUNNER_NAME) in denied["hookSpecificOutput"]["permissionDecisionReason"]
    assert (
        decide("export CUDA_VISIBLE_DEVICES=1; python3 bench.py")["hookSpecificOutput"]["permissionDecision"] == "deny"
    )
    assert decide(f"python3 {lane_dir / RUNNER_NAME} python3 forge_driver.py --bench-mode") == {}


@pytest.mark.parametrize("name", [RUNNER_NAME, RUNNER_CONFIG_NAME])
def test_a_lane_hook_denies_an_edit_to_the_device_runner(tmp_path, monkeypatch, name):
    """The runner and its configuration decide which device a command sees, so they are not the session's to change."""
    spec, lane_dir = _run_lane_session(tmp_path, monkeypatch)
    deny_edit = next(
        hook.callback for hook in spec.hooks.pre_tool_use if hook.matcher == gate_module._EDIT_TOOL_MATCHER
    )

    denied = asyncio.run(
        deny_edit({"tool_name": "Edit", "tool_input": {"file_path": str(lane_dir / name)}}, None, None)
    )

    assert denied["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.parametrize(
    ("gain_required", "told", "not_told"),
    [
        (True, "correct but not faster", "Other sessions are working from"),
        (False, "Other sessions are working from", "correct but not faster"),
    ],
    ids=["a round's only lane", "one lane of several"],
)
def test_a_lane_is_told_how_its_stop_is_judged(tmp_path, monkeypatch, gain_required, told, not_told):
    """Alone, a lane has to beat the current best; beside others it has to be correct, and the merge uses the rest."""
    spec, _lane_dir = _run_lane_session(tmp_path, monkeypatch, gain_required=gain_required)
    gate = spec.hooks.stop[0].callback.__self__
    prompt = " ".join(spec.system_prompt.split())

    assert gate.gain_required is gain_required
    assert told in prompt
    assert not_told not in prompt


def test_a_lane_hook_denies_an_edit_to_the_lane_driver(tmp_path, monkeypatch):
    """The installed hook refuses the edit rather than the finished candidate."""
    spec, lane_dir = _run_lane_session(tmp_path, monkeypatch)
    deny_edit = next(
        hook.callback for hook in spec.hooks.pre_tool_use if hook.matcher == gate_module._EDIT_TOOL_MATCHER
    )

    denied = asyncio.run(
        deny_edit(
            {
                "tool_name": "Edit",
                "tool_input": {"file_path": str(lane_dir / "forge_driver.py")},
            },
            None,
            None,
        )
    )
    allowed = asyncio.run(
        deny_edit(
            {
                "tool_name": "Edit",
                "tool_input": {"file_path": str(lane_dir / "src" / "kernel.py")},
            },
            None,
            None,
        )
    )

    decision = denied["hookSpecificOutput"]
    assert decision["permissionDecision"] == "deny"
    assert "forge_driver.py" in decision["permissionDecisionReason"]
    assert allowed == {}


def test_a_lane_hook_denies_a_shell_write_to_the_lane_driver(tmp_path, monkeypatch):
    """The Bash denial travels with the lane too, not just the edit tools."""
    spec, _lane_dir = _run_lane_session(tmp_path, monkeypatch)
    deny_bash = next(hook.callback for hook in spec.hooks.pre_tool_use if hook.matcher == "Bash")

    denied = asyncio.run(
        deny_bash(
            {
                "tool_name": "Bash",
                "tool_input": {"command": "sed -i 's/1/2/' forge_driver.py"},
            },
            None,
            None,
        )
    )
    allowed = asyncio.run(
        deny_bash(
            {
                "tool_name": "Bash",
                "tool_input": {"command": "python3 forge_driver.py"},
            },
            None,
            None,
        )
    )

    assert denied["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert allowed == {}


def _bash_hook(spec: AgentRunSpec):
    """The Bash denial the session carries to its provider."""
    return next(hook.callback for hook in spec.hooks.pre_tool_use if hook.matcher == "Bash")


def _bash_decision(spec: AgentRunSpec, command: str) -> dict:
    """What the session's Bash hook answers for one command."""
    return asyncio.run(
        _bash_hook(spec)(
            {"tool_name": "Bash", "tool_input": {"command": command}},
            None,
            None,
        )
    )


def test_the_canonical_session_keeps_its_benchmarking_stop_hook(tmp_path, monkeypatch):
    """Pin the single-session path: it self-corrects against the scoring it was handed, and that has not changed."""
    specs = _record_backend_specs(monkeypatch)
    workspace = _tree(tmp_path / "workspace")

    agent_fn = agent_module.make_agent_fn(
        Config(
            workspace=str(workspace),
            agent_backend="claude",
            agent_model="claude-test",
            agent_precheck=False,
        ),
        agent_module.ImplementerSetup(
            agent_module.ImplementerTask(program_md="Optimize VALUE."),
            measurement=agent_module.GateMeasurement(driver_script=str(workspace / "forge_driver.py")),
            session=agent_module.SessionOptions(agent_backend="claude", insession_gate=True),
        ),
    )
    asyncio.run(agent_fn(str(workspace / "src" / "kernel.py"), "", gate_scoring=GateScoring({"case": 2.0}, 1.25)))

    spec = specs[0]
    assert spec.hooks is not None
    assert len(spec.hooks.stop) == 1
    assert "ONE self-correcting session" in spec.system_prompt
    gate = spec.hooks.stop[0].callback.__self__
    assert (gate.baseline_case_times, gate.best_mean_case_speedup) == ({"case": 2.0}, 1.25)
    assert "Current best pristine-relative score: 1.25." in spec.system_prompt
    # It has the device to itself, so it is told nothing about a runner.
    assert RUNNER_NAME not in spec.system_prompt
    assert gate.device is None


def _register_provider(name: str, **capabilities: bool) -> None:
    """Register one provider declaring exactly the capabilities named."""

    def factory(runtime):
        """Never called: the lane path refuses before it builds a backend."""
        raise AssertionError(f"provider {runtime.provider} must not be built")

    register_agent_provider(
        AgentProvider(
            name=name,
            factory=factory,
            default_model="fake-model",
            capabilities=AgentCapabilities(**capabilities),
        )
    )


def test_lanes_warn_but_run_on_a_provider_that_does_not_run_our_hooks(capsys):
    """The hooks save a lane's work; they are not what makes its result true.

    This was a refusal, on the ground that a provider ignoring ``spec.hooks``
    leaves a lane with no in-session protection at all. It buys a real thing --
    an edit to the driver, harness or oracle is denied while the session can
    still be saved, and so is a driver run that skips the shared device lock --
    but nothing a campaign publishes rests on it. A candidate touching the
    measurement surface is refused on its patch paths by
    ``IterationRunner._lane_rejection``, the canonical driver is re-checked
    byte-for-byte once the patch applies, and every candidate is re-measured one
    at a time in the canonical tree, so a lane's own timings never decide a
    KEEP. Unlocked concurrent benchmarking is caught by the lane teardown's
    contention report, which voids the round out loud.

    What is left is wasted budget, reported when it happens. That is a cost to
    warn about, not a result to refuse -- and refusing it is what shut the whole
    rewrite OPTIMIZE phase to every hookless provider.
    """
    _register_provider("hooklesscli", session_env=True)

    assert cli_lanes._require_lane_provider_capabilities("hooklesscli", 2) is None

    warning = capsys.readouterr().out
    assert "hooklesscli" in warning
    assert "stop_hooks" in warning
    assert "WARNING" in warning


def test_lanes_are_refused_on_a_provider_that_ignores_the_session_environment():
    """A lane's private build cache is carried by AgentRunSpec.env, or not at all.

    A provider that drops it puts every lane back into one cache, where aiter
    imports a module by name and a lane measures a binary a sibling compiled.
    """
    _register_provider("sharedenvcli", stop_hooks=True)

    with pytest.raises(ForgeLoopError) as refusal:
        cli_lanes._require_lane_provider_capabilities("sharedenvcli", 2)

    message = str(refusal.value)
    assert "sharedenvcli" in message
    assert "session_env" in message
    assert "stop_hooks" not in message


def test_a_lane_refusal_names_every_missing_guarantee(capsys):
    """One re-run has to be enough, so the operator is told all of it at once.

    The refusal and the advisory reach the operator together: switching to a
    provider that clears the refusal must not be the moment they first hear
    what that provider still will not do for a lane.
    """
    _register_provider("plaincli")

    with pytest.raises(ForgeLoopError) as refusal:
        cli_lanes._require_lane_provider_capabilities("plaincli", 4)

    message = str(refusal.value)
    assert "session_env" in message
    assert "--lanes 4" in message
    assert "--lanes 1" in message

    warning = capsys.readouterr().out
    assert "stop_hooks" in warning


def test_lanes_are_refused_rather_than_quietly_reduced():
    """The operator asked for N sessions and gets N or an explanation.

    Answering with fewer lanes would be the same silent downgrade the refusal
    exists to prevent, only with the evidence for it thrown away.
    """
    _register_provider("plaincli")

    with pytest.raises(ForgeLoopError):
        cli_lanes._require_lane_provider_capabilities("plaincli", 8)


def test_one_lane_is_never_refused():
    """A single session needs none of this: there is nothing to isolate it from.

    It is also what a refusal offers as the way forward, so it cannot itself
    depend on the guarantees that were missing.
    """
    _register_provider("plaincli")

    assert cli_lanes._require_lane_provider_capabilities("plaincli", 1) is None


def test_lanes_run_on_a_provider_that_declares_both():
    """The check reads what a provider declares, not who wrote it."""
    _register_provider("fullcli", stop_hooks=True, session_env=True)

    assert cli_lanes._require_lane_provider_capabilities("fullcli", 2) is None


def test_a_provider_that_declares_both_is_warned_about_nothing(capsys):
    """The advisory is about a missing guarantee, not a note on every run."""
    _register_provider("fullcli", stop_hooks=True, session_env=True)

    cli_lanes._require_lane_provider_capabilities("fullcli", 2)

    assert capsys.readouterr().out == ""


def test_the_builtin_hook_capable_provider_passes_the_lane_check():
    """Tie the built-in declaration to the rule that reads it.

    Claude runs the hooks and applies the session environment; if either
    declaration were dropped, concurrent lanes would stop being available at all
    rather than quietly losing a guarantee.
    """
    assert cli_lanes._require_lane_provider_capabilities("claude", 2) is None


def test_the_builtin_codex_provider_runs_concurrent_lanes(capsys):
    """Codex applies the session environment; hooks stay off the lane path."""
    assert cli_lanes._require_lane_provider_capabilities("codex", 2) is None
    warning = capsys.readouterr().out
    assert "codex" in warning
    assert "stop_hooks" in warning
    assert "WARNING" in warning
