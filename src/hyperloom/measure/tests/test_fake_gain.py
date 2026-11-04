# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The prefix-cache check of an AgentX baseline round, read from the point it ran and the log it landed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from hyperloom.measure.fake_gain import PREFIX_CACHE_OFF_ERROR_CLASS, baseline_prefix_cache_refusal

SGLANG_OFF = "server_args=ServerArgs(model_path='m', disable_radix_cache=True, tp_size=8)\n"
SGLANG_ON = "server_args=ServerArgs(model_path='m', disable_radix_cache=False, tp_size=8)\n"
VLLM_OFF = "INFO 01-01 00:00:00 api_server.py:1 non-default args: {'model': 'm', 'enable_prefix_caching': False}\n"
VLLM_ON = "INFO 01-01 00:00:00 api_server.py:1 non-default args: {'model': 'm'}\n"


def _round(tmp_path: Path, framework: str, log: str | None, atom_args: str = "") -> tuple[Path, Path]:
    point = tmp_path / "point.yaml"
    envs = {"EXTRA_ATOM_ARGS": atom_args} if framework == "atom" else {}
    point.write_text(yaml.safe_dump({"benchmark": {"framework": framework, "envs": envs}}), encoding="utf-8")
    directory = tmp_path / "round"
    directory.mkdir()
    if log is not None:
        (directory / "server.log").write_text(log, encoding="utf-8")
    return point, directory


@pytest.mark.parametrize(
    ("framework", "log", "refused"),
    [
        ("sglang", SGLANG_OFF, True),
        ("sglang", SGLANG_ON, False),
        ("vllm", VLLM_OFF, True),
        ("vllm", VLLM_ON, False),
        ("sglang", None, False),
    ],
)
def test_a_round_is_refused_only_when_its_server_log_says_the_cache_was_off(
    tmp_path: Path, framework: str, log: str | None, refused: bool
) -> None:
    refusal = baseline_prefix_cache_refusal(*_round(tmp_path, framework, log))
    assert bool(refusal) is refused
    if refused:
        assert "AgentX baseline refused" in refusal


@pytest.mark.parametrize(("atom_args", "refused"), [("--level 2", True), ("--level 2 --enable_prefix_caching", False)])
def test_an_atom_round_is_read_from_the_server_args_of_its_point(tmp_path: Path, atom_args: str, refused: bool) -> None:
    assert bool(baseline_prefix_cache_refusal(*_round(tmp_path, "atom", None, atom_args))) is refused


ATOM_LAUNCH = ["python3", "-m", "atom.entrypoints.openai_server", "--model", "m", "--level", "2"]


def _agentx_round(tmp_path: Path, framework: str, argv: list[str] | None, log: str | None = None) -> tuple[Path, Path]:
    """An AgentX point, which carries its launch in ``agentx.launch_overrides``, and the round Magpie reported for it."""
    point = tmp_path / "point.yaml"
    overrides = {"version": 1, "append_args": ["--level", "2", "--enable_prefix_caching"]}
    benchmark = {"framework": framework, "model": "m", "agentx": {"enabled": True, "launch_overrides": overrides}}
    point.write_text(yaml.safe_dump({"benchmark": {**benchmark, "envs": {"TP": 8}}}), encoding="utf-8")
    directory = tmp_path / "round"
    directory.mkdir()
    if log is not None:
        (directory / "server.log").write_text(log, encoding="utf-8")
    if argv is not None:
        metrics = {
            "execution_owner": "magpie",
            "server_launch": {"effective_argv": argv, "effective_env": {}},
            "recipe": {"tp": 8},
            "requests": {},
        }
        report = {"scenario": "agentx", "success": True, "benchmark_valid": True, "publishable": True}
        (directory / "benchmark_report.json").write_text(json.dumps({**report, "agentx_metrics": metrics}))
    return point, directory


@pytest.mark.parametrize(("argv", "refused"), [(ATOM_LAUNCH, True), ([*ATOM_LAUNCH, "--enable_prefix_caching"], False)])
def test_an_agentx_atom_round_is_read_from_the_launch_magpie_reports_it_ran(
    tmp_path: Path, argv: list[str], refused: bool
) -> None:
    refusal = baseline_prefix_cache_refusal(*_agentx_round(tmp_path, "atom", argv))
    assert bool(refusal) is refused, refusal
    if refused:
        assert "without --enable_prefix_caching" in refusal


def test_an_agentx_sglang_round_is_read_from_its_server_log(tmp_path: Path) -> None:
    point, directory = _agentx_round(tmp_path, "sglang", ["python3", "-m", "sglang.launch_server"], SGLANG_OFF)
    assert "disable_radix_cache=True" in baseline_prefix_cache_refusal(point, directory)


def test_an_agentx_round_magpie_reported_no_launch_for_is_left_alone(tmp_path: Path) -> None:
    assert baseline_prefix_cache_refusal(*_agentx_round(tmp_path, "atom", None)) == ""


def test_the_launch_of_a_warmup_round_is_read_when_the_round_has_no_log_of_its_own(tmp_path: Path) -> None:
    point, directory = _round(tmp_path, "sglang", None)
    (directory / "warmup_round").mkdir()
    (directory / "warmup_round" / "server.log").write_text(SGLANG_OFF, encoding="utf-8")
    assert "disable_radix_cache=True" in baseline_prefix_cache_refusal(point, directory)


def test_the_log_in_the_round_directory_wins_over_a_warmup_launch(tmp_path: Path) -> None:
    point, directory = _round(tmp_path, "sglang", SGLANG_ON)
    (directory / "warmup_round").mkdir()
    (directory / "warmup_round" / "server.log").write_text(SGLANG_OFF, encoding="utf-8")
    assert baseline_prefix_cache_refusal(point, directory) == ""


def test_the_refusal_carries_the_agentx_baseline_error_class_name() -> None:
    assert PREFIX_CACHE_OFF_ERROR_CLASS == "agentx_baseline_prefix_cache_off"
