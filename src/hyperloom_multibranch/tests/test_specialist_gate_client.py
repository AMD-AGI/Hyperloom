# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The gate's client, read from a rendered canonical configuration, sends the passes a fixed round sends, or the
replay an AgentX round sends, against an agent's server."""

from __future__ import annotations

import os
import shlex
import stat
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

import inference_suite
from hyperloom.orchestrator.actions.executors import TOOL_CACHE_ENV, SuiteRoundRefused, fixed_workload
from hyperloom.orchestrator.actions.executors._suite.config import suite_config
from hyperloom.orchestrator.actions.executors.tests.test_suite_accuracy_round import _agentx_base, _agentx_point
from hyperloom.orchestrator.actions.executors.tests.test_suite_config import _settings
from hyperloom.orchestrator.state.tests._session_base import freeze_base
from hyperloom_multibranch.specialist.gate_client import AgentxGateClient, FixedGateClient, read_gate_client
from inference_suite import inferencex_root
from inference_suite.cli import read_config
from inference_suite.contracts import Lifecycle, LifecycleMode, WorkloadKind

MODEL = "Qwen/Qwen3-14B-FP8"


def _canonical(tmp_path: Path, **envs: str) -> Path:
    bench = {"framework": "vllm", "model": MODEL, "inferencex_path": "/opt/InferenceX", "envs": envs}
    path = tmp_path / "canonical.yaml"
    path.write_text(yaml.safe_dump({"benchmark": bench}), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def tool_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cache = tmp_path / "tool-cache"
    monkeypatch.setenv(TOOL_CACHE_ENV, str(cache))
    return cache


def _read(canonical: Path) -> FixedGateClient:
    client = read_gate_client(canonical, canonical.parent, 7200)
    assert isinstance(client, FixedGateClient)
    return client


def _flag(words: list[str], flag: str) -> str:
    return words[words.index(flag) + 1]


def test_the_client_sends_the_warmup_then_the_measured_pass_of_the_gate_s_round(
    tmp_path: Path, tool_cache: Path
) -> None:
    envs = {"ISL": "1024", "OSL": "1024", "CONC": "8", "RANDOM_RANGE_RATIO": "0.8", "NUM_PROMPTS": "80"}
    canonical = _canonical(tmp_path, **envs, NUM_WARMUPS="16", BENCH_TRUST_REMOTE_CODE="1")
    client = _read(canonical)
    assert client.workload == fixed_workload(yaml.safe_load(canonical.read_text(encoding="utf-8")))

    warmup, measured = (shlex.split(part) for part in client.stage(31110, tmp_path / "out").split(" && "))

    for words in (warmup, measured):
        assert words[:4] == [
            f"PYTHONPATH={inferencex_root(tool_cache)}",
            "python3",
            "-m",
            "infx.bench_serving.benchmark_serving",
        ]
        assert (_flag(words, "--base-url"), _flag(words, "--model"), _flag(words, "--served-model-name")) == (
            "http://0.0.0.0:31110",
            MODEL,
            MODEL,
        )
        assert "--trust-remote-code" in words and _flag(words, "--num-warmups") == "0"
    assert (_flag(warmup, "--seed"), _flag(warmup, "--num-prompts"), "--save-result" in warmup) == ("1", "16", False)
    assert (_flag(measured, "--seed"), _flag(measured, "--num-prompts")) == ("0", "80")
    assert (_flag(measured, "--result-dir"), _flag(measured, "--result-filename")) == (
        str(tmp_path / "out"),
        "inferencex_result.json",
    )


def test_the_client_runs_from_the_staged_inferencex_the_gate_s_round_runs_not_the_canonical_checkout(
    tmp_path: Path, tool_cache: Path
) -> None:
    client = _read(_canonical(tmp_path, ISL="128", OSL="128", CONC="4", NUM_PROMPTS="40", NUM_WARMUPS="0"))

    assert client.package_root == str(inferencex_root(tool_cache))
    assert "/opt/InferenceX" not in client.stage(8000, tmp_path / "out")


def test_a_workload_without_warmups_is_the_measured_pass_alone(tmp_path: Path) -> None:
    envs = {"ISL": "128", "OSL": "128", "CONC": "4", "NUM_PROMPTS": "40", "NUM_WARMUPS": "0"}
    client = _read(_canonical(tmp_path, **envs))

    command = client.stage(8000, tmp_path / "out")

    assert " && " not in command and "--save-result" in shlex.split(command)


def test_a_fixed_client_is_bounded_by_the_agent_s_own_bench_cap(tmp_path: Path) -> None:
    client = _read(_canonical(tmp_path, ISL="128", OSL="128", CONC="4", NUM_PROMPTS="40", NUM_WARMUPS="0"))

    assert client.seconds == 0


CORPUS = "semianalysis_cc_traces_weka_062126"


def _base():
    base = _agentx_base()
    return replace(base, client=replace(base.client, env={"WEKA_LOADER_OVERRIDE": CORPUS}))


def _agentx_session(tmp_path: Path, tool_cache: Path, **envs: str) -> tuple[Path, dict]:
    """A session that froze an AgentX base and staged its tools, and the AgentX canonical it rendered."""

    base = _base()
    session = tmp_path / "toolbox"
    session.mkdir()
    freeze_base(session, base)
    tool_cache.mkdir()
    (tool_cache / "staging_manifest.json").write_text('{"schema": "inference-suite-staging/1", "tools": {}}\n')
    canonical = _agentx_point(base)
    canonical["benchmark"]["envs"].update(envs)
    path = session / "canonical" / "canonical.yaml"
    path.parent.mkdir()
    path.write_text(yaml.safe_dump(canonical), encoding="utf-8")
    return path, canonical


def test_an_agentx_client_runs_the_suite_s_replay_of_the_gate_s_round_against_the_agent_s_server(
    tmp_path: Path, tool_cache: Path
) -> None:
    canonical, document = _agentx_session(tmp_path, tool_cache, DURATION="1200")
    client = read_gate_client(canonical, canonical.parent.parent, 4800)
    assert isinstance(client, AgentxGateClient)
    out = tmp_path / "agent" / "mb-bench"

    words = shlex.split(client.stage(31110, out))

    packages = Path(inference_suite.__file__).resolve().parents[1]
    assert words[:5] == [f"PYTHONPATH={packages}", sys.executable, "-m", "inference_suite", "run"]
    assert _flag(words, "--run-dir") == str(out)
    config = read_config(Path(_flag(words, "--config")), tmp_path / "iter_1" / "host_bench")
    gate = suite_config({"benchmark": {**document["benchmark"], "timeout_seconds": 4800}}, _base(), _settings())
    assert config.agentx == gate.agentx and config.agentx.duration_sec == 1200
    assert (config.workload, config.lifecycle) == (WorkloadKind.AGENTX, Lifecycle(LifecycleMode.EXTERNAL))
    assert (config.server.host, config.server.port, config.server.served_model_name) == (
        "0.0.0.0",
        31110,
        gate.server.served_model_name,
    )
    assert config.paths.run_dir == str(tmp_path / "iter_1" / "host_bench")
    assert (config.paths.tool_cache, config.telemetry.gpu.enabled) == (str(tool_cache), False)
    assert client.seconds == config.deadlines.round_sec == 4800


def test_each_agent_s_agentx_client_measures_its_own_port(tmp_path: Path, tool_cache: Path) -> None:
    canonical, _ = _agentx_session(tmp_path, tool_cache)
    client = read_gate_client(canonical, canonical.parent.parent, 4800)

    first, second = (
        read_config(Path(_flag(shlex.split(client.stage(port, tmp_path / name / "mb-bench")), "--config")), tmp_path)
        for port, name in ((31100, "a1"), (31110, "a2"))
    )

    assert (first.server.port, second.server.port) == (31100, 31110)


def test_an_agentx_client_stages_its_config_as_the_group_s_below_shared_storage(
    tmp_path: Path, tool_cache: Path
) -> None:
    canonical, _ = _agentx_session(tmp_path, tool_cache)
    client = read_gate_client(canonical, canonical.parent.parent, 4800)
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o2775)
    previous = os.umask(0o022)
    try:
        config = Path(_flag(shlex.split(client.stage(31110, shared / "agent" / "mb-bench")), "--config"))
    finally:
        os.umask(previous)

    assert stat.S_IMODE(config.parent.stat().st_mode) == 0o2775
    assert stat.S_IMODE(config.stat().st_mode) == 0o664


def test_an_agentx_client_is_refused_when_the_session_froze_no_base_or_staged_no_tools(
    tmp_path: Path, tool_cache: Path
) -> None:
    canonical, _ = _agentx_session(tmp_path, tool_cache)
    (tool_cache / "staging_manifest.json").unlink()

    with pytest.raises(SuiteRoundRefused, match="staging manifest"):
        read_gate_client(canonical, canonical.parent.parent, 4800)
    with pytest.raises(SuiteRoundRefused, match="froze no base launch"):
        read_gate_client(canonical, tmp_path, 4800)
