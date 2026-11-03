# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A framework with no validated runner for the board is refused at launch, on both CLI entry paths."""

import asyncio

import pytest

from hyperloom.common.benchmark_config import BENCHMARK_CONFIG_ENV
from hyperloom.inference_optimizer import cli
from hyperloom.inference_optimizer.cli import launch_shape
from hyperloom.inference_optimizer.cli.benchmark_config import configure_benchmark
from hyperloom.inference_optimizer.cli.parser import _build_parser

from .conftest import cli_call_sequence


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    for key in (
        BENCHMARK_CONFIG_ENV,
        "FRAMEWORK",
        "FRAMEWORK_VERSION",
        "GPU_TYPE",
        "TARGET_GPU_TYPE",
        "MODEL_PATH",
        "TP",
        "EP",
        "CONC",
        "ISL",
        "OSL",
        "PRECISION",
        "MAX_MODEL_LEN",
        "INFERENCE_OPTIMIZER_EXTRA_ENV",
        "INFERENCE_OPTIMIZER_SERVER_ARGS",
        "INFERENCE_OPTIMIZER_NODES",
    ):
        monkeypatch.setenv(key, "")


def _seed(tmp_path, monkeypatch, framework: str) -> None:
    args = _build_parser().parse_args(
        ["optimize", "--model", "/model", "--framework", framework, "--gpu-type", "r9700"]
    )
    configure_benchmark(args)
    launch_shape._resolve_run_topology(args)
    for name in ("_enforce_expected_framework", "_apply_operator_supplied_paths", "_check_gfx_arch_resolvable"):
        monkeypatch.setattr(cli, name, lambda *a, **kw: None)
    monkeypatch.setattr(cli, "_should_remote_probe_gpu", lambda args: False)
    monkeypatch.setattr(cli, "_autodetect_gpu_type", lambda: None)
    monkeypatch.setattr(cli, "resolve_framework_version", lambda args: "test-version")
    monkeypatch.setattr(cli, "make_session_dir", lambda **kw: tmp_path)
    monkeypatch.setattr(cli, "_acquire_session_lock_or_exit", lambda path: None)

    class ManifestReached(Exception):
        pass

    def manifest(session_dir, *, args):
        raise ManifestReached

    monkeypatch.setattr(cli, "write_manifest", manifest)
    with pytest.raises(ManifestReached):
        asyncio.run(cli._seed_fresh_session(args, nodes_resolved=1))


def test_fresh_r9700_sglang_launch_exits_2(tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit) as exc:
        _seed(tmp_path, monkeypatch, "sglang")
    assert exc.value.code == 2
    assert "no validated gfx12 runner" in capsys.readouterr().err


def test_fresh_r9700_vllm_launch_passes_the_gate(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, "vllm")


def test_resume_runs_the_gate():
    assert "_require_custom_entrypoint" in cli_call_sequence("_resume_session")
