# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A framework not validated on the board is refused at launch, on both CLI entry paths."""

import asyncio
import os

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


def _seed(tmp_path, monkeypatch, framework: str, gpu_type: str = "r9700", *extra: str) -> None:
    args = _build_parser().parse_args(
        ["optimize", "--model", "/model", "--framework", framework, "--gpu-type", gpu_type, *extra]
    )
    configure_benchmark(args)
    launch_shape._resolve_run_topology(args)
    for name in ("_enforce_expected_framework", "_apply_operator_supplied_paths", "_check_gfx_arch_resolvable"):
        monkeypatch.setattr(cli, name, lambda *a, **kw: None)
    monkeypatch.setattr(cli, "_autodetect_gpu_type", lambda agents=None: None)
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


@pytest.mark.parametrize(("product", "validated"), [("r9700", "llamacpp, vllm"), ("w7900", "vllm")])
def test_fresh_radeon_sglang_launch_exits_2(tmp_path, monkeypatch, capsys, product, validated):
    with pytest.raises(SystemExit) as exc:
        _seed(tmp_path, monkeypatch, "sglang", product)
    assert exc.value.code == 2
    assert f"{product} is currently validated only for {validated}" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("product", "flags"),
    [("r9700", ()), ("w7900", ()), ("w7900", ("--no-kernel",))],
    ids=["r9700", "w7900", "w7900-no-kernel"],
)
def test_fresh_radeon_vllm_launch_passes_the_gate_and_exports_the_board_itself(tmp_path, monkeypatch, product, flags):
    _seed(tmp_path, monkeypatch, "vllm", product, *flags)
    assert os.environ["GPU_TYPE"] == os.environ["TARGET_GPU_TYPE"] == product


def test_resume_runs_the_gate():
    assert "_require_custom_entrypoint" in cli_call_sequence("_resume_session")
