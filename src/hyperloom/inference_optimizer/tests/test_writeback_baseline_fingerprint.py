# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Baseline fingerprint and workload-extra extraction used by writeback."""

from __future__ import annotations

import pytest

from hyperloom.orchestrator.loop.writeback import WritebackCollaborator, _parse_baseline_workload_extra


def test_parse_baseline_workload_extra_missing(tmp_path):
    assert _parse_baseline_workload_extra(str(tmp_path / "nope.yaml")) == {}


def test_parse_baseline_workload_extra_full(tmp_path):
    yaml_path = tmp_path / "base.yaml"
    yaml_path.write_text(
        "benchmark:\n"
        "  workload_mode: serving\n"
        "  quant_scheme: fp8\n"
        "  envs:\n"
        "    EXTRA_SGLANG_ARGS: '--max-running-requests 256 --enable-chunked-prefill"
        " --enable-torch-compile'\n",
        encoding="utf-8",
    )
    out = _parse_baseline_workload_extra(str(yaml_path))
    assert out["workload_mode"] == "serving"
    assert out["quant_scheme"] == "fp8"
    assert out["max_running_requests"] == 256
    assert out["chunked_prefill_enabled"] is True
    assert out["enable_torch_compile"] is True


def test_parse_baseline_workload_extra_torch_compile_env(tmp_path):
    yaml_path = tmp_path / "base.yaml"
    yaml_path.write_text(
        "benchmark:\n"
        "  envs:\n"
        "    ENABLE_TORCH_COMPILE: 'true'\n"
        "    EXTRA_SGLANG_ARGS: '--disable-chunked-prefill --max-num-seqs 32'\n",
        encoding="utf-8",
    )
    out = _parse_baseline_workload_extra(str(yaml_path))
    assert out["enable_torch_compile"] is True
    assert out["chunked_prefill_enabled"] is False
    assert out["max_num_seqs"] == 32


@pytest.mark.parametrize(("framework", "env_key"), [("vllm", "EXTRA_VLLM_ARGS"), ("atom", "EXTRA_ATOM_ARGS")])
def test_parse_baseline_workload_extra_reads_own_framework_args(tmp_path, framework, env_key):
    yaml_path = tmp_path / "base.yaml"
    yaml_path.write_text(
        f"benchmark:\n  framework: {framework}\n  envs:\n    {env_key}: '--max-running-requests 8'\n",
        encoding="utf-8",
    )
    assert _parse_baseline_workload_extra(str(yaml_path))["max_running_requests"] == 8


def test_parse_baseline_workload_extra_ignores_another_frameworks_args(tmp_path):
    # An operator --extra-env lands in benchmark.envs unfiltered, so a key for
    # another framework can sit beside the one this server is launched with.
    yaml_path = tmp_path / "base.yaml"
    yaml_path.write_text(
        "benchmark:\n  framework: vllm\n  envs:\n"
        "    EXTRA_SGLANG_ARGS: '--max-running-requests 99'\n"
        "    EXTRA_VLLM_ARGS: '--max-running-requests 8'\n",
        encoding="utf-8",
    )
    assert _parse_baseline_workload_extra(str(yaml_path))["max_running_requests"] == 8


def test_parse_baseline_workload_extra_non_dict_benchmark(tmp_path):
    yaml_path = tmp_path / "base.yaml"
    yaml_path.write_text("benchmark: not-a-dict\n", encoding="utf-8")
    assert _parse_baseline_workload_extra(str(yaml_path)) == {}


def test_baseline_params_fingerprint():
    out = WritebackCollaborator.baseline_params_fingerprint(
        {
            "benchmark_script": "b.sh",
            "extra_envs": {"B": "2", "A": "1"},
        }
    )
    assert out["benchmark_script"] == "b.sh"
    assert out["model_path"] is None
    assert out["extra_envs"] == [["A", "1"], ["B", "2"]]


def test_baseline_params_fingerprint_bad_envs():
    out = WritebackCollaborator.baseline_params_fingerprint({"extra_envs": "oops"})
    assert out["extra_envs"] is None
