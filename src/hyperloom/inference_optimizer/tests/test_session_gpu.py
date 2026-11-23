# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A session's GPU on both CLI entry paths: the card it reads and publishes, and the label it files results under."""

import argparse
import asyncio
import os
from types import SimpleNamespace

import pytest

from hyperloom_common import gpu_card
from hyperloom.common.benchmark_config import BENCHMARK_CONFIG_ENV
from hyperloom.common.visible_devices import VISIBLE_DEVICE_VARS
from hyperloom.inference_optimizer import cli
from hyperloom.inference_optimizer.cli import launch_shape, session_resume
from hyperloom.inference_optimizer.cli.benchmark_config import configure_benchmark
from hyperloom.inference_optimizer.cli.parser import _build_parser
from hyperloom.inference_optimizer.multi_node._internal import gpu_probe
from hyperloom.orchestrator.tests._fixtures import MI325X_CARD, MI355X_CARD


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
    for mask in VISIBLE_DEVICE_VARS:
        monkeypatch.delenv(mask, raising=False)


def _host(monkeypatch, *cards: gpu_card.GpuCard) -> None:
    monkeypatch.setattr(gpu_card, "read_cards", lambda: list(cards))


def _args(*flags: str) -> argparse.Namespace:
    return argparse.Namespace(gpu_type=dict(zip(flags[::2], flags[1::2])).get("--gpu-type"))


def _seed(tmp_path, monkeypatch, *flags: str) -> list:
    """Start a fresh session up to its manifest; returns the card the quantization prelude saw."""
    args = _build_parser().parse_args(["optimize", "--model", "/model", *flags])
    configure_benchmark(args)
    launch_shape._resolve_run_topology(args)
    for name in ("_enforce_expected_framework", "_apply_operator_supplied_paths"):
        monkeypatch.setattr(cli, name, lambda *a, **kw: None)
    monkeypatch.setattr(cli, "resolve_framework_version", lambda args: "test-version")
    monkeypatch.setattr(cli, "make_session_dir", lambda **kw: tmp_path)
    monkeypatch.setattr(cli, "_acquire_session_lock_or_exit", lambda path: None)
    prelude_saw = []

    async def prelude(args):
        prelude_saw.append(gpu_card.session_card())

    monkeypatch.setattr(cli, "_run_quantization_prelude", prelude)

    class ManifestReached(Exception):
        pass

    def manifest(session_dir, *, args):
        raise ManifestReached

    monkeypatch.setattr(cli, "write_manifest", manifest)
    with pytest.raises(ManifestReached):
        asyncio.run(cli._seed_fresh_session(args, nodes_resolved=1))
    return prelude_saw


def test_a_fresh_session_publishes_its_card_before_the_quantization_prelude(tmp_path, monkeypatch):
    _host(monkeypatch, MI355X_CARD, MI355X_CARD)

    assert _seed(tmp_path, monkeypatch, "--framework", "vllm") == [MI355X_CARD]
    assert gpu_card.session_card() == MI355X_CARD
    assert os.environ["GPU_TYPE"] == os.environ["TARGET_GPU_TYPE"] == "mi355x"


def test_any_name_is_accepted_as_the_gpu_type():
    args = _build_parser().parse_args(["optimize", "--model", "/model", "--gpu-type", "MI300A"])
    assert args.gpu_type == "mi300a"


def test_a_given_gpu_type_counts_even_when_the_card_names_itself_otherwise(monkeypatch, capsys):
    _host(monkeypatch, MI355X_CARD)
    args = _args("--gpu-type", "mi350x")

    cli._resolve_target_gpu(args, nodes=1)

    assert args.gpu_type == os.environ["GPU_TYPE"] == "mi350x"
    assert gpu_card.session_card() == MI355X_CARD
    assert "'mi355x'" in capsys.readouterr().err


def test_the_environment_names_the_gpu_type_when_the_flag_does_not(monkeypatch):
    _host(monkeypatch, MI355X_CARD)
    monkeypatch.setenv("TARGET_GPU_TYPE", "MI350X")
    monkeypatch.setenv("GPU_TYPE", "mi300x")

    cli._resolve_target_gpu(args := _args(), nodes=1)

    assert args.gpu_type == "mi350x"


def test_a_host_whose_cards_cannot_be_read_runs_without_a_card(monkeypatch, capsys):
    """Not a refusal: the first round stops a host without GPUs, and a given name still labels the session."""
    gpu_card.publish_session_card(MI355X_CARD)

    cli._resolve_target_gpu(args := _args("--gpu-type", "mi300x"), nodes=1)

    assert gpu_card.session_card() is None
    assert args.gpu_type == "mi300x"
    assert "none read (amd-smi static could not run (test))" in capsys.readouterr().out


def test_mixed_cards_refuse_the_session_and_list_each_kind(monkeypatch, capsys):
    _host(monkeypatch, MI325X_CARD, MI355X_CARD)

    with pytest.raises(SystemExit) as exc:
        cli._resolve_target_gpu(_args(), nodes=1)

    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "AMD Instinct MI325X" in err and "AMD Instinct MI355X" in err


def test_the_visible_device_mask_picks_the_cards_by_rocm_number(monkeypatch):
    _host(monkeypatch, MI325X_CARD, MI355X_CARD)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "1")

    cli._resolve_target_gpu(_args(), nodes=1)

    assert gpu_card.session_card() == MI355X_CARD


def test_a_multi_node_session_reads_a_remote_nodes_cards(monkeypatch):
    """This host's cards are not the ones the benchmark runs on."""
    _host(monkeypatch, MI325X_CARD)
    monkeypatch.setattr(gpu_probe, "remote_cards", lambda: [MI355X_CARD] * 8)

    cli._resolve_target_gpu(args := _args(), nodes=2)

    assert gpu_card.session_card() == MI355X_CARD
    assert args.gpu_type == "mi355x"


def test_a_resume_restores_the_label_and_reads_the_card_again(tmp_path, monkeypatch):
    """The label is the session's; the facts are the machine's this leg runs on."""
    _host(monkeypatch, MI355X_CARD)
    gpu_card.publish_session_card(MI325X_CARD)
    state = SimpleNamespace(model_path="", framework="vllm", gpu_type="mi350x", nodes=1)

    session_resume._reexport_session_identity(argparse.Namespace(nodes=1), state, tmp_path)

    assert os.environ["GPU_TYPE"] == "mi350x"
    assert gpu_card.session_card() == MI355X_CARD
