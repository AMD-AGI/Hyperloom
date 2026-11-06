# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A refused argv costs a probe, not a round, and it terminates by name."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

from hyperloom.common.bringup import LadderStage
from hyperloom.orchestrator.actions.executors import _benchmark_interpreter
from hyperloom.orchestrator.actions.executors import baseline as bl
from hyperloom.orchestrator.actions.executors.benchmark_run.records import BenchmarkAttempt
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
#: accepted.
_STALE_FLAG = "--moe-backend"

_PROBE_PYTHON = "/opt/probe/bin/python"


@pytest.fixture
def refusing_probe(monkeypatch):
    """Answer the identity probes as one install, and the parser probe as a refusal."""
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
    monkeypatch.setattr(
        _benchmark_interpreter, "_resolve_probe_python", lambda _framework="vllm", *, env=None: _PROBE_PYTHON
    )
    return calls


async def _attempt(session: Path, slot: Path, *, task_id: str, server_args: str) -> dict:
    """Run one baseline round whose config carries ``server_args``."""
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
    return await _run(session, slot, config, task_id=task_id)


async def _run(session: Path, slot: Path, config: Path, *, task_id: str) -> dict:
    """Run one baseline round over an already materialised ``config``."""
    executor = bl.BaselineExecutor(session_dir=session, magpie_python="/usr/bin/python3")
    return await executor._run_single_benchmark(
        BenchmarkAttempt(
            timeout_sec=1800,
            override_result_dir=None,
            resolved_model="scripted/Model",
            materialized_config_path=config,
            inferencex_path="",
            effective_extra_server_args="",
            params={"framework": "sglang"},
            ctx=_ctx(task_id),
        ),
        config_path=config,
        output_dir=slot,
    )


@pytest.mark.asyncio
async def test_a_refused_argv_never_launches_and_never_opens_a_repair_round(
    round_slot,
    registry,
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

    lane = _lane(session, tasks, rounds, result["enablement_launch_log"])
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
    round_slot,
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
    monkeypatch.setattr(
        _benchmark_interpreter, "_resolve_probe_python", lambda _framework="vllm", *, env=None: _PROBE_PYTHON
    )

    session, slot = round_slot
    launches = ScriptedLaunchBackend(scenario=LaunchScenario.from_dict(_SCENARIO))
    launch_backend(launches)

    await _attempt(session, slot, task_id="baseline-0", server_args="--tp 8")
    assert launches.served == 1


@pytest.mark.parametrize("source", ("recipe", "user", "candidate"))
@pytest.mark.asyncio
async def test_an_unknown_flag_from_any_source_is_refused_by_name_and_never_dropped(
    round_slot,
    launch_backend,
    monkeypatch,
    source,
):
    """Whichever layer brought the flag, the round fails naming it and the config still carries it."""
    from hyperloom.orchestrator.actions.executors import _multi_node_env as multi_node
    from hyperloom.orchestrator.actions.executors import _workload_envs as we
    from hyperloom.orchestrator.actions.executors._server_argv import config_server_argv
    from hyperloom.orchestrator.actions.executors._workload_envs import LaunchLayer, MaterializeOptions

    from ._workload_envs_helpers import clear_env, write_config

    monkeypatch.setattr(multi_node, "is_multi_node", lambda: False)
    clear_env(monkeypatch)
    monkeypatch.delenv("INFERENCE_OPTIMIZER_SERVER_ARGS", raising=False)
    stale = f"{_STALE_FLAG} triton"
    recipe_args = f"--tp 8 {stale}" if source == "recipe" else "--tp 8"
    monkeypatch.setattr(we, "resolve_reference_launch", lambda: (recipe_args, {}, {}))
    if source == "user":
        monkeypatch.setenv("INFERENCE_OPTIMIZER_SERVER_ARGS", stale)
    candidate = LaunchLayer(extra_server_args=stale if source == "candidate" else "--context-length 4096")

    identity = '{"executable": "/opt/probe/bin/python", "python": "3.12.1", "origin": "/site/sglang/__init__.py", "dist": "0.5.1"}'

    def _probe(argv, env, timeout_sec):
        program = argv[2] if len(argv) > 2 else ""
        if "_build_parser" not in program:
            return subprocess.CompletedProcess(argv, 0, stdout=identity + "\n", stderr="")
        body = (
            f'{{"status": "invalid", "message": "unrecognized arguments: {stale}", "unknown": ["{_STALE_FLAG}", "triton"]}}'
            if _STALE_FLAG in argv[3]
            else '{"status": "ok", "message": "", "unknown": []}'
        )
        return subprocess.CompletedProcess(argv, 0, stdout=body + "\n", stderr="")

    monkeypatch.setattr(pf, "_default_probe", _probe)
    monkeypatch.setattr(
        _benchmark_interpreter, "_resolve_probe_python", lambda _framework="vllm", *, env=None: _PROBE_PYTHON
    )

    session, slot = round_slot
    src = write_config(slot / "base.yaml", envs={pf.SERVING_PYTHON_ENV: _PROBE_PYTHON})
    config = we.materialize_config_with_envs(
        src, slot, launch=candidate, options=MaterializeOptions(out_name="baseline-0.yaml", agentx_mode=False)
    )
    assert _STALE_FLAG in config_server_argv(config).argv

    launches = ScriptedLaunchBackend(scenario=LaunchScenario.from_dict(_SCENARIO))
    launch_backend(launches)
    result = await _run(session, slot, config, task_id="baseline-0")

    assert launches.served == 0
    assert result["error_class"] == pf.ARGV_INVALID
    assert _STALE_FLAG in result["error"]
    assert result["server_argv_unknown_flags"] == [_STALE_FLAG]
    assert _STALE_FLAG in config_server_argv(config).argv


@pytest.mark.asyncio
async def test_an_unavailable_probe_does_not_stop_the_round(
    round_slot,
    launch_backend,
    monkeypatch,
):
    """An unprovable interpreter leaves the launch as the only verdict, as before."""
    from hyperloom.orchestrator.actions.executors import _multi_node_env as multi_node

    monkeypatch.setattr(multi_node, "is_multi_node", lambda: False)
    monkeypatch.setattr(pf, "resolve_serving_interpreter", lambda _framework, _env: "")

    session, slot = round_slot
    launches = ScriptedLaunchBackend(scenario=LaunchScenario.from_dict(_SCENARIO))
    launch_backend(launches)

    await _attempt(session, slot, task_id="baseline-0", server_args=_BAD_ARGS)
    assert launches.served == 1


def test_a_state_recorded_with_argv_repairs_still_loads():
    """Sessions written before the drop repair was removed carry its ledger; it is ignored."""
    from hyperloom.orchestrator.state._shared_state.enablement_round import EnablementRound

    loaded = EnablementRound.from_dict({"argv_repairs": ["0" * 64], "launch_log": "boot failed"})
    assert loaded.launch_log == "boot failed"
    assert not hasattr(loaded, "argv_repairs")
