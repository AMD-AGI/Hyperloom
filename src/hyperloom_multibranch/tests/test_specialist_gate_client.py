# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The gate's client, read from a rendered canonical configuration, sends the passes a fixed round sends."""

from __future__ import annotations

import shlex
from pathlib import Path

import pytest
import yaml

from hyperloom.orchestrator.actions.executors import TOOL_CACHE_ENV, fixed_workload
from hyperloom_multibranch.specialist.gate_client import GateClient
from inference_suite import inferencex_root

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


def _flag(words: list[str], flag: str) -> str:
    return words[words.index(flag) + 1]


def test_the_client_sends_the_warmup_then_the_measured_pass_of_the_gate_s_round(
    tmp_path: Path, tool_cache: Path
) -> None:
    envs = {"ISL": "1024", "OSL": "1024", "CONC": "8", "RANDOM_RANGE_RATIO": "0.8", "NUM_PROMPTS": "80"}
    canonical = _canonical(tmp_path, **envs, NUM_WARMUPS="16", BENCH_TRUST_REMOTE_CODE="1")
    client = GateClient.read(canonical)
    assert client is not None
    assert client.workload == fixed_workload(yaml.safe_load(canonical.read_text(encoding="utf-8")))

    warmup, measured = (shlex.split(part) for part in client.command(31110, tmp_path / "out").split(" && "))

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
    client = GateClient.read(_canonical(tmp_path, ISL="128", OSL="128", CONC="4", NUM_PROMPTS="40", NUM_WARMUPS="0"))
    assert client is not None

    assert client.package_root == str(inferencex_root(tool_cache))
    assert "/opt/InferenceX" not in client.command(8000, tmp_path / "out")


def test_a_workload_without_warmups_is_the_measured_pass_alone(tmp_path: Path) -> None:
    envs = {"ISL": "128", "OSL": "128", "CONC": "4", "NUM_PROMPTS": "40", "NUM_WARMUPS": "0"}
    client = GateClient.read(_canonical(tmp_path, **envs))
    assert client is not None

    command = client.command(8000, tmp_path / "out")

    assert " && " not in command and "--save-result" in shlex.split(command)


def test_an_agentx_canonical_has_no_client(tmp_path: Path) -> None:
    assert GateClient.read(_canonical(tmp_path, CONC="8", AGENTX_DURATION_SEC="600")) is None
