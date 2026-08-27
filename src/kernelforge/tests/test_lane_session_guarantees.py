# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What protects an Implementer lane, and who is asked to guarantee it.

A lane edits its own copy of the workspace, and its candidate is measured later,
once, by the loop. The measurement surface still has to survive the session: a
lane diff that touches the driver, the harness or the oracle is refused at the
boundary, and the refusal takes the implementation edits in the same diff with
it -- after the session has already been paid for in full.

A hook denies that edit while the agent is still working, which is the only
point at which the rest of the session can still be saved. The lane runs the
in-session gate for those denials alone: the gate's Stop hook benchmarks, and
lanes run concurrently while the device measures one thing at a time.

That protection, and the private build cache a lane compiles into, are both
things the provider does on the lane's behalf. Neither is checked by the lane
code, so the second half of this file is about the provider being made to
declare them before a round of lanes is allowed to start.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import click
import pytest

import kernelforge.agent_backends.registry as registry
import kernelforge.loop.insession_gate as gate_module
import kernelforge.orchestrator.agent as agent_module
from kernelforge.agent_backends.base import (
    AgentCapabilities,
    AgentRunResult,
    AgentRunSpec,
)
from kernelforge.agent_backends.registry import AgentProvider, register_agent_provider
from kernelforge import cli
from kernelforge.config import Config
from kernelforge.loop import fanout


@pytest.fixture(autouse=True)
def isolated_provider_registry(monkeypatch):
    """Give every test in this module its own copy of the provider registry.

    Same reason as the fixture of the same name in ``test_provider_registry.py``:
    ``register_agent_provider`` writes module-level state that no API removes, so
    a fake registered below would stay visible to every later test in the same
    worker process. Discovery runs first so the snapshot already holds the
    built-ins, then the globals are rebound to copies monkeypatch drops.
    """
    registry.discover_agent_providers()
    monkeypatch.setattr(registry, "_providers", dict(registry._providers))
    monkeypatch.setattr(registry, "_plugin_errors", dict(registry._plugin_errors))


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


def _record_backend_specs(monkeypatch) -> list[AgentRunSpec]:
    """Route every implementer session to a backend that records its spec."""
    specs: list[AgentRunSpec] = []

    class Backend:
        """Stand in for a hook-capable provider without running one."""

        name = "claude"
        capabilities = AgentCapabilities(resumable=True, stop_hooks=True)

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
    return cli._make_lane_agent_factory(
        make_agent=agent_module.make_agent_fn,
        config=config,
        workspace_dir=str(workspace),
        driver=str(workspace / "forge_driver.py"),
        source_files=[str(workspace / "src" / "kernel.py")],
        session_kwargs={
            "program_md": "Optimize VALUE.",
            "agent_backend": "claude",
        },
    )


def _run_lane_session(
    tmp_path: Path,
    monkeypatch,
    *,
    serialized_driver: str | None = None,
) -> tuple[AgentRunSpec, Path]:
    """Run one lane session the way a fan-out round runs it."""
    specs = _record_backend_specs(monkeypatch)
    factory = _lane_factory(tmp_path)
    lane_dir = _tree(tmp_path / "lanes" / "1")

    session = factory(str(lane_dir), serialized_driver)
    asyncio.run(session(str(lane_dir / "src" / "kernel.py"), "plan 1"))

    assert len(specs) == 1
    return specs[0], lane_dir


def test_a_lane_session_is_given_the_protected_path_hooks(tmp_path, monkeypatch):
    """A lane must reach its provider carrying the gate's edit and Bash denials."""
    spec, _lane_dir = _run_lane_session(tmp_path, monkeypatch)

    assert spec.hooks is not None
    assert {hook.matcher for hook in spec.hooks.pre_tool_use} == {
        gate_module._EDIT_TOOL_MATCHER,
        "Bash",
    }
    assert [hook.matcher for hook in spec.hooks.post_tool_use] == [gate_module._EDIT_TOOL_MATCHER]


def test_a_lane_session_is_not_given_the_benchmarking_stop_hook(tmp_path, monkeypatch):
    """The Stop hook runs correctness and a benchmark; lanes are concurrent.

    Lanes overlap in time and the device measures one thing at a time, so a lane
    that benchmarked at the end of its own session would time its kernel against
    whatever its siblings were running.
    """
    spec, _lane_dir = _run_lane_session(tmp_path, monkeypatch)

    assert spec.hooks is not None
    assert spec.hooks.stop == []


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


@pytest.mark.parametrize(
    "command",
    [
        "python3 forge_driver.py",
        "python3 forge_driver.py --warmup 3 --iters 20 --bench-mode",
        "python forge_driver.py",
        "/usr/bin/python3 forge_driver.py",
        "timeout 600 python3 forge_driver.py --bench-mode",
        "./forge_driver.py --bench-mode",
        "cd src && python3 ../forge_driver.py",
        "bash -c 'python3 forge_driver.py --bench-mode'",
        "timeout 600 sh -c 'python3 forge_driver.py'",
        "python3 -W ignore forge_driver.py",
        "python3 -X dev forge_driver.py --bench-mode",
        # An interpreter reaches the driver by module too, and -m names it
        # without the suffix a path carries.
        "python3 -m forge_driver",
        "python3 -mforge_driver --bench-mode",
        "python3 -X dev -m forge_driver",
        "python3 -m tools.forge_driver",
        # A shell's -c clusters and attaches like any short option.
        "bash -lc 'python3 forge_driver.py'",
        "bash -c'python3 forge_driver.py'",
        "sh -ic 'python3 forge_driver.py --bench-mode'",
    ],
)
def test_a_lane_hook_refuses_a_driver_run_that_would_skip_the_lock(tmp_path, monkeypatch, command):
    """A prompt is a preference; the number the round is judged on is not.

    The wrapper is what holds the device lock, so a driver run that goes around
    it times this lane against whichever sibling is benchmarking at the same
    moment -- and corrupts that sibling's number too, which is the part no
    lesson can attribute to anything.
    """
    wrapper = str(tmp_path / "lanes" / "1" / fanout.SERIALIZED_DRIVER_NAME)
    spec, _lane_dir = _run_lane_session(tmp_path, monkeypatch, serialized_driver=wrapper)

    decision = _bash_decision(spec, command)["hookSpecificOutput"]

    assert decision["permissionDecision"] == "deny"
    assert wrapper in decision["permissionDecisionReason"]


@pytest.mark.parametrize(
    "command",
    [
        "python3 {wrapper}",
        "python3 {wrapper} --warmup 3 --iters 20 --bench-mode",
        "cat forge_driver.py",
        "grep -n case_ms forge_driver.py",
        "python3 -c 'import torch; print(torch.__version__)'",
        "bash -c 'python3 {wrapper} --bench-mode'",
        # Reading -m must not turn every module run into a driver run.
        "python3 -m pytest tests/",
        "python3 -m pip install --no-deps triton",
        "python3 -m json.tool config.json",
        "bash -lc 'python3 {wrapper} --bench-mode'",
    ],
)
def test_a_lane_hook_allows_the_locked_run_and_every_read(tmp_path, monkeypatch, command):
    """Only executing the driver is refused, and only outside its wrapper.

    Reading the driver is how a lane learns what it is being scored on, and the
    wrapper is the command it was told to measure through.
    """
    wrapper = str(tmp_path / "lanes" / "1" / fanout.SERIALIZED_DRIVER_NAME)
    spec, _lane_dir = _run_lane_session(tmp_path, monkeypatch, serialized_driver=wrapper)

    assert _bash_decision(spec, command.format(wrapper=wrapper)) == {}


def test_a_session_without_a_wrapper_still_runs_the_driver_itself(tmp_path, monkeypatch):
    """The refusal belongs to an interposed command, not to the gate at large.

    Every ordinary session runs the driver directly and must keep doing so; the
    rule exists only where a wrapper was put in front of it.
    """
    spec, _lane_dir = _run_lane_session(tmp_path, monkeypatch)

    assert _bash_decision(spec, "python3 forge_driver.py --bench-mode") == {}


def test_a_lane_is_told_to_run_the_driver_through_its_own_lock(tmp_path, monkeypatch):
    """The lock lives in the wrapper, so it binds only if the session runs it.

    The wrapper used to reach the lane in a per-invocation note alone, while the
    factory argument carrying it went unused. A requirement that holds for the
    whole session belongs in the session's own instructions, which a long run
    keeps in view long after its first message.
    """
    wrapper = str(tmp_path / "lanes" / "1" / fanout.SERIALIZED_DRIVER_NAME)

    spec, _lane_dir = _run_lane_session(tmp_path, monkeypatch, serialized_driver=wrapper)

    assert f"python3 {wrapper}" in spec.system_prompt
    assert "Never `python3 forge_driver.py` directly" in spec.system_prompt


def test_a_session_without_a_wrapper_is_told_nothing_about_one(tmp_path, monkeypatch):
    """No wrapper means no interposition, so the prompt is exactly as it was."""
    spec, _lane_dir = _run_lane_session(tmp_path, monkeypatch)

    assert fanout.SERIALIZED_DRIVER_NAME not in spec.system_prompt
    assert "Run the driver through this command" not in spec.system_prompt


def test_a_lane_keeps_the_prompt_of_the_session_it_actually_runs(tmp_path, monkeypatch):
    """No Stop hook means no gate to send the agent back, so it is not promised one.

    The self-correcting prompt describes a gate that re-checks correctness and
    speed and rejects a stop that does not converge. A lane has no such gate.
    """
    spec, _lane_dir = _run_lane_session(tmp_path, monkeypatch)

    assert "ONE self-correcting session" not in spec.system_prompt
    assert "Authoritative mean case scoring state" not in spec.system_prompt


def test_the_canonical_session_keeps_its_benchmarking_stop_hook(tmp_path, monkeypatch):
    """Pin the single-session path: it self-corrects, and that has not changed."""
    specs = _record_backend_specs(monkeypatch)
    workspace = _tree(tmp_path / "workspace")

    agent_fn = agent_module.make_agent_fn(
        config=Config(
            workspace=str(workspace),
            agent_backend="claude",
            agent_model="claude-test",
            agent_precheck=False,
        ),
        program_md="Optimize VALUE.",
        agent_backend="claude",
        insession_gate=True,
        driver_script=str(workspace / "forge_driver.py"),
    )
    asyncio.run(agent_fn(str(workspace / "src" / "kernel.py"), ""))

    spec = specs[0]
    assert spec.hooks is not None
    assert len(spec.hooks.stop) == 1
    assert "ONE self-correcting session" in spec.system_prompt


@pytest.mark.parametrize("stop_check", [True, False])
def test_both_gate_modes_share_one_protected_path_rule(tmp_path, stop_check):
    """Neither mode may re-derive what counts as the measurement surface."""
    workspace = _tree(tmp_path / "workspace")
    gate = gate_module.InSessionGate(
        driver_script=str(workspace / "forge_driver.py"),
        snr_threshold=30.0,
        kernel_file=str(workspace / "src" / "kernel.py"),
    )

    hooks = gate.make_agent_hooks(stop_check=stop_check)

    pre_edit_callbacks = {hook.callback for hook in hooks.pre_tool_use}
    assert gate._on_pre_edit in pre_edit_callbacks
    assert gate._on_pre_bash in pre_edit_callbacks
    assert [hook.callback for hook in hooks.post_tool_use] == [gate._on_edit]


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


def test_lanes_are_refused_on_a_provider_that_does_not_run_our_hooks():
    """Without the hooks the lane protection is a promise nothing keeps.

    The gate builds them and the spec carries them, and a provider that ignores
    ``spec.hooks`` drops them in silence -- leaving a lane exactly where it
    started, losing whole candidates at the boundary check.
    """
    _register_provider("hooklesscli", session_env=True)

    with pytest.raises(click.ClickException) as refusal:
        cli._require_lane_provider_capabilities("hooklesscli", 2)

    message = str(refusal.value)
    assert "hooklesscli" in message
    assert "stop_hooks" in message
    assert "session_env" not in message


def test_lanes_are_refused_on_a_provider_that_ignores_the_session_environment():
    """A lane's private build cache is carried by AgentRunSpec.env, or not at all.

    A provider that drops it puts every lane back into one cache, where aiter
    imports a module by name and a lane measures a binary a sibling compiled.
    """
    _register_provider("sharedenvcli", stop_hooks=True)

    with pytest.raises(click.ClickException) as refusal:
        cli._require_lane_provider_capabilities("sharedenvcli", 2)

    message = str(refusal.value)
    assert "sharedenvcli" in message
    assert "session_env" in message
    assert "stop_hooks" not in message


def test_a_lane_refusal_names_every_missing_guarantee():
    """One re-run has to be enough, so the operator is told all of it at once."""
    _register_provider("plaincli")

    with pytest.raises(click.ClickException) as refusal:
        cli._require_lane_provider_capabilities("plaincli", 4)

    message = str(refusal.value)
    assert "stop_hooks" in message
    assert "session_env" in message
    assert "--lanes 4" in message
    assert "--lanes 1" in message


def test_lanes_are_refused_rather_than_quietly_reduced():
    """The operator asked for N sessions and gets N or an explanation.

    Answering with fewer lanes would be the same silent downgrade the refusal
    exists to prevent, only with the evidence for it thrown away.
    """
    _register_provider("plaincli")

    with pytest.raises(click.ClickException):
        cli._require_lane_provider_capabilities("plaincli", 8)


def test_one_lane_is_never_refused():
    """A single session needs none of this: there is nothing to isolate it from.

    It is also what a refusal offers as the way forward, so it cannot itself
    depend on the guarantees that were missing.
    """
    _register_provider("plaincli")

    assert cli._require_lane_provider_capabilities("plaincli", 1) is None


def test_lanes_run_on_a_provider_that_declares_both():
    """The check reads what a provider declares, not who wrote it."""
    _register_provider("fullcli", stop_hooks=True, session_env=True)

    assert cli._require_lane_provider_capabilities("fullcli", 2) is None


def test_the_builtin_hook_capable_provider_passes_the_lane_check():
    """Tie the built-in declaration to the rule that reads it.

    Claude runs the hooks and applies the session environment; if either
    declaration were dropped, concurrent lanes would stop being available at all
    rather than quietly losing a guarantee.
    """
    assert cli._require_lane_provider_capabilities("claude", 2) is None


@pytest.mark.parametrize(
    "command",
    [
        'pgrep -af "forge_lane_driver.py|forge_driver.py"',
        'rg -n "forge_driver.py|chunk.py" src/',
        'grep -E "forge_driver.py|kernel.py" build.log',
    ],
)
def test_a_pipe_inside_one_argument_is_not_a_second_command(tmp_path, monkeypatch, command):
    """Observed in a live round: a lane's `pgrep` was refused as a driver run.

    The operators that separate commands were found by a regex over the raw
    text, so a pipe inside a quoted argument cut the string in half and the
    tail became a command whose verb was a file the session never ran.
    """
    wrapper = str(tmp_path / "lanes" / "1" / fanout.SERIALIZED_DRIVER_NAME)
    spec, _lane_dir = _run_lane_session(tmp_path, monkeypatch, serialized_driver=wrapper)

    assert _bash_decision(spec, command) == {}
