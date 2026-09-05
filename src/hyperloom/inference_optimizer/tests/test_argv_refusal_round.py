# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A refused argv costs a probe, not a round, and it terminates by name.

The failure this reproduces used to look like every other launch failure: the
server died a few seconds in, the round was consumed, and the enablement
backstop -- which dispatches an authoring specialist for any non-blank launch
log, because an unrecognised wall may still be one a patch gets past -- spent
the next round writing a framework patch for an argument the framework simply
does not have.

Played here across ticks on the real executor, the real lane, the real round
store and the real registry, with the launch subprocess scripted. What is
asserted is the negative space: the scripted backend is never asked to launch,
no repair task is opened, and the run stops under a name of its own.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

from hyperloom.common.bringup import LadderStage
from hyperloom.orchestrator.actions.executors import baseline as bl
from hyperloom.orchestrator.bringup import argv_preflight as pf
from hyperloom.orchestrator.bringup import load_boot_observation
from hyperloom.orchestrator.rehearsal import LaunchScenario, ScriptedLaunchBackend

from .test_bringup_round_scenario import _ctx, _lane, registry, round_slot  # noqa: F401

#: One clean boot, offered and never taken: the assertion is that the preflight
#: refuses before anything reaches it.
_SCENARIO = {"name": "never-launched", "attempts": [{"name": "would-boot", "outcome": "ready"}]}

#: A value the installed parser rejects. Nothing here may be rewritten: the
#: harness does not know what the framework meant by it.
_BAD_ARGS = "--tp 8 --attention-backend fa4"

#: A flag the installed parser does not have, in the spelling an older version
#: accepted. This one is droppable, exactly once.
_STALE_FLAG_ARGS = "--tp 8 --moe-backend triton"

_PROBE_PYTHON = "/opt/probe/bin/python"


@pytest.fixture
def refusing_probe(monkeypatch):
    """Answer the identity probes as one install, and the parser probe as a refusal.

    The subprocess itself is scripted rather than run: what this test is about
    is what the round does with a refusal, and the parser's own verdict is
    established against a real parser elsewhere.
    """
    identity = '{"executable": "/opt/probe/bin/python", "python": "3.12.1", "origin": "/site/sglang/__init__.py", "dist": "0.5.1"}'
    refusal = '{"status": "invalid", "message": "argument --attention-backend: invalid choice: \'fa4\'", "unknown": []}'
    calls: list[list[str]] = []

    def _probe(argv, env, timeout_sec):
        calls.append(list(argv))
        program = argv[2] if len(argv) > 2 else ""
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=(refusal if "_build_parser" in program else identity) + "\n",
            stderr="",
        )

    monkeypatch.setattr(pf, "_default_probe", _probe)
    monkeypatch.setattr(pf, "_resolve_probe_interpreter", lambda _framework: _PROBE_PYTHON)
    return calls


async def _attempt(session: Path, slot: Path, *, task_id: str, server_args: str) -> dict:
    """Run one baseline round whose config carries ``server_args``."""
    executor = bl.BaselineExecutor(session_dir=session, magpie_python="/usr/bin/python3")
    config = slot / f"{task_id}.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "framework": "sglang",
                    "model": "scripted/Model",
                    "envs": {
                        "EXTRA_SGLANG_ARGS": server_args,
                        pf.SERVING_PYTHON_ENV: _PROBE_PYTHON,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    return await executor._run_single_benchmark(
        config_path=config,
        output_dir=slot,
        timeout_sec=1800,
        override_result_dir=None,
        resolved_model="scripted/Model",
        materialized_config_path=config,
        inferencex_path="",
        effective_extra_server_args="",
        params={"framework": "sglang"},
        ctx=_ctx(task_id),
    )


@pytest.mark.asyncio
async def test_a_refused_argv_never_launches_and_never_opens_a_repair_round(
    round_slot,  # noqa: F811
    registry,  # noqa: F811
    launch_backend,
    refusing_probe,
    monkeypatch,
):
    """No launch, one named terminal, and nothing for the backstop to author against."""
    from hyperloom.orchestrator.actions.executors import _multi_node_env as multi_node

    monkeypatch.setattr(multi_node, "is_multi_node", lambda: False)

    session, slot = round_slot
    tasks, rounds, _locks = registry
    launches = ScriptedLaunchBackend(scenario=LaunchScenario.from_dict(_SCENARIO))
    launch_backend(launches)

    result = await _attempt(session, slot, task_id="baseline-0", server_args=_BAD_ARGS)

    # Nothing was launched: the round was refused for the cost of a probe.
    assert launches.served == 0
    assert result["status"] == "failed"
    assert result["error_class"] == pf.ARGV_INVALID

    # And the refusal was recorded where every other bring-up observation goes,
    # at the rung the argv is parsed on.
    loaded = load_boot_observation(result.get("boot_observation_path"))
    assert loaded.observation is not None, loaded.degraded
    assert loaded.observation.stage_failed == LadderStage.ARGV_PARSE
    assert pf.is_argv_invalid(loaded.observation)

    lane = _lane(session, tasks, rounds, result["enablement_launch_log"], attempts=0)
    lane.shared_state.enablement.launch_observation_path = result["boot_observation_path"]

    # Tick after tick, the backstop declines to author against it and the run
    # carries the terminal that names what actually happened.
    for _tick in range(3):
        assert await lane._maybe_enqueue_enablement_specialist() == ""
        assert lane.shared_state.stop_reason == pf.ARGV_INVALID
    assert await rounds.held() is None
    assert launches.served == 0


@pytest.mark.asyncio
async def test_an_argv_the_parser_accepts_still_reaches_the_launch(
    round_slot,  # noqa: F811
    launch_backend,
    monkeypatch,
):
    """The check must be invisible on the path it does not stop."""
    from hyperloom.orchestrator.actions.executors import _multi_node_env as multi_node

    monkeypatch.setattr(multi_node, "is_multi_node", lambda: False)
    accepted = '{"status": "ok", "message": "", "unknown": []}'
    identity = '{"executable": "/opt/probe/bin/python", "python": "3.12.1", "origin": "/site/sglang/__init__.py", "dist": "0.5.1"}'

    def _probe(argv, env, timeout_sec):
        program = argv[2] if len(argv) > 2 else ""
        return subprocess.CompletedProcess(
            argv, 0, stdout=(accepted if "_build_parser" in program else identity) + "\n", stderr=""
        )

    monkeypatch.setattr(pf, "_default_probe", _probe)
    monkeypatch.setattr(pf, "_resolve_probe_interpreter", lambda _framework: _PROBE_PYTHON)

    session, slot = round_slot
    launches = ScriptedLaunchBackend(scenario=LaunchScenario.from_dict(_SCENARIO))
    launch_backend(launches)

    await _attempt(session, slot, task_id="baseline-0", server_args="--tp 8")
    assert launches.served == 1


@pytest.mark.asyncio
async def test_the_one_allowed_drop_is_written_back_before_the_launch(
    round_slot,  # noqa: F811
    launch_backend,
    monkeypatch,
):
    """A repair the config does not carry is a repair the server never receives."""
    from hyperloom.orchestrator.actions.executors import _multi_node_env as multi_node
    from hyperloom.orchestrator.actions.executors._server_argv import config_server_argv

    monkeypatch.setattr(multi_node, "is_multi_node", lambda: False)
    identity = '{"executable": "/opt/probe/bin/python", "python": "3.12.1", "origin": "/site/sglang/__init__.py", "dist": "0.5.1"}'

    def _probe(argv, env, timeout_sec):
        program = argv[2] if len(argv) > 2 else ""
        if "_build_parser" not in program:
            return subprocess.CompletedProcess(argv, 0, stdout=identity + "\n", stderr="")
        refused = "--moe-backend" in argv[3]
        body = (
            '{"status": "invalid", "message": "unrecognized arguments: --moe-backend triton",'
            ' "unknown": ["--moe-backend", "triton"]}'
            if refused
            else '{"status": "ok", "message": "", "unknown": []}'
        )
        return subprocess.CompletedProcess(argv, 0, stdout=body + "\n", stderr="")

    monkeypatch.setattr(pf, "_default_probe", _probe)
    monkeypatch.setattr(pf, "_resolve_probe_interpreter", lambda _framework: _PROBE_PYTHON)

    session, slot = round_slot
    launches = ScriptedLaunchBackend(scenario=LaunchScenario.from_dict(_SCENARIO))
    launch_backend(launches)

    await _attempt(session, slot, task_id="baseline-0", server_args=_STALE_FLAG_ARGS)

    assert launches.served == 1
    assert config_server_argv(slot / "baseline-0.yaml").argv == ("--tp", "8")


@pytest.mark.asyncio
async def test_an_unavailable_probe_does_not_stop_the_round(
    round_slot,  # noqa: F811
    launch_backend,
    monkeypatch,
):
    """An unprovable interpreter leaves the launch as the only verdict, as before."""
    from hyperloom.orchestrator.actions.executors import _multi_node_env as multi_node

    monkeypatch.setattr(multi_node, "is_multi_node", lambda: False)
    monkeypatch.setattr(pf, "_resolve_probe_interpreter", lambda _framework: "")

    session, slot = round_slot
    launches = ScriptedLaunchBackend(scenario=LaunchScenario.from_dict(_SCENARIO))
    launch_backend(launches)

    await _attempt(session, slot, task_id="baseline-0", server_args=_BAD_ARGS)
    assert launches.served == 1
