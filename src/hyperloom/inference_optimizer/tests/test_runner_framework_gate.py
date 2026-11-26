# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A framework not validated on the board is refused at launch, on both CLI entry paths."""

import asyncio
import os

import pytest

from hyperloom.common.benchmark_config import BENCHMARK_CONFIG_ENV
from hyperloom.inference_optimizer import cli
from hyperloom.inference_optimizer.cli import base_launch, launch_shape
from hyperloom.inference_optimizer.cli.benchmark_config import configure_benchmark
from hyperloom.inference_optimizer.cli.parser import _build_parser
from inference_suite import SourceSnapshot
from inference_suite.recipe.resolve import UnreadSource

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


def _seed(tmp_path, monkeypatch, framework: str, board: str) -> None:
    args = _build_parser().parse_args(["optimize", "--model", "/model", "--framework", framework, "--gpu-type", board])
    offline = SourceSnapshot(None, 0, unread=UnreadSource("InferenceX", "offline"))
    monkeypatch.setattr(base_launch, "fetch_sources", lambda *args, **kwargs: offline)
    configure_benchmark(args)
    launch_shape._resolve_run_topology(args)
    for name in ("_enforce_expected_framework", "_apply_operator_supplied_paths", "_check_gfx_arch_resolvable"):
        monkeypatch.setattr(cli, name, lambda *a, **kw: None)
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


@pytest.mark.parametrize("board", ["r9700", "r9600d"])
def test_fresh_atom_launch_exits_2(tmp_path, monkeypatch, capsys, board):
    with pytest.raises(SystemExit) as exc:
        _seed(tmp_path, monkeypatch, "atom", board)
    assert exc.value.code == 2
    assert "framework 'atom' has not been validated on it" in capsys.readouterr().err


@pytest.mark.parametrize("board", ["r9700", "r9600d"])
@pytest.mark.parametrize("framework", ["vllm", "sglang"])
def test_fresh_launch_passes_the_gate_and_exports_the_board(tmp_path, monkeypatch, board, framework):
    _seed(tmp_path, monkeypatch, framework, board)
    assert os.environ["GPU_TYPE"] == os.environ["TARGET_GPU_TYPE"] == board


def test_resume_runs_the_gate():
    assert "_require_custom_entrypoint" in cli_call_sequence("_resume_session")
