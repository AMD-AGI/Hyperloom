# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The prefix-cache check of an AgentX baseline round, read from the point it ran and the log it landed."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hyperloom.measure.fake_gain import PREFIX_CACHE_OFF_ERROR_CLASS, baseline_prefix_cache_refusal

SGLANG_OFF = "server_args=ServerArgs(model_path='m', disable_radix_cache=True, tp_size=8)\n"
SGLANG_ON = "server_args=ServerArgs(model_path='m', disable_radix_cache=False, tp_size=8)\n"
VLLM_OFF = "INFO 10-14 api_server.py:1 non-default args: {'model': 'm', 'enable_prefix_caching': False}\n"
VLLM_ON = "INFO 10-14 api_server.py:1 non-default args: {'model': 'm'}\n"


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


def test_the_refusal_carries_main_s_error_class_name() -> None:
    assert PREFIX_CACHE_OFF_ERROR_CLASS == "agentx_baseline_prefix_cache_off"
