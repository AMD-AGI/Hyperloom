# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The prefix-cache check of an AgentX baseline round, read from the run_dir it landed in."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from hyperloom.common.tests.test_round_measurement import _agentx, _agentx_result
from hyperloom.measure.fake_gain import PREFIX_CACHE_OFF_ERROR_CLASS, RoundUnreadable, baseline_prefix_cache_refusal
from inference_suite.contracts import ProfileResult, Report, RunLayout
from inference_suite.contracts.base_launch import Workload
from inference_suite.contracts.launch import Launch, LaunchArg, LayerSource
from inference_suite.tests.test_compose import _launch

SGLANG_OFF = "server_args=ServerArgs(model_path='m', disable_radix_cache=True, tp_size=8)\n"
SGLANG_ON = "server_args=ServerArgs(model_path='m', disable_radix_cache=False, tp_size=8)\n"
VLLM_OFF = "INFO 01-01 00:00:00 api_server.py:1 non-default args: {'model': 'm', 'enable_prefix_caching': False}\n"
VLLM_ON = "INFO 01-01 00:00:00 api_server.py:1 non-default args: {'model': 'm'}\n"


def _suite_round(tmp_path: Path, report: Report, log: str | None, launch: Launch | None = None) -> Path:
    """The inference_suite run_dir an AgentX baseline landed in, with its report, launch and server log."""
    directory = tmp_path / "round"
    directory.mkdir()
    if log is not None:
        (directory / "server.log").write_text(log, encoding="utf-8")
    layout = RunLayout(directory)
    layout.report.write_text(json.dumps(report.to_document()), encoding="utf-8")
    layout.launch.write_text(json.dumps((launch or _launch()).to_document()), encoding="utf-8")
    return directory


def _framework_launch(framework: str, entrypoint: tuple[str, ...], *args: tuple[str, ...]) -> Launch:
    """The suite launch of ``framework`` whose final argv is ``entrypoint`` and ``args``.

    The suite composes no ATOM AgentX base, so an ATOM launch is built on a fixed-workload base.
    """
    launch = _launch()
    query = replace(launch.base.query, framework=framework)
    if framework == "atom":
        query = replace(query, workload=Workload.FIXED, isl=1024, osl=1024)
    base = replace(launch.base, framework=framework, query=query, entrypoint=entrypoint)
    flags = tuple(LaunchArg(flag, tuple(values), LayerSource.CANDIDATE) for flag, *values in args)
    return replace(launch, base=base, framework=replace(launch.framework, name=framework), args=flags)


@pytest.mark.parametrize(("log", "refused"), [(SGLANG_OFF, True), (SGLANG_ON, False)], ids=["cache_off", "cache_on"])
def test_a_valid_suite_round_is_read_from_its_server_log(tmp_path: Path, log: str, refused: bool) -> None:
    assert bool(baseline_prefix_cache_refusal(_suite_round(tmp_path, _agentx(), log))) is refused


@pytest.mark.parametrize(
    "report",
    [
        _agentx(_agentx_result(stopped_by_threshold=True)),
        _agentx(profile=ProfileResult(trace_dirs=("profile/torch_trace",), has_step_traces=False)),
    ],
    ids=["invalid", "diagnostic"],
)
def test_a_suite_round_the_mapper_does_not_accept_is_left_alone(tmp_path: Path, report: Report) -> None:
    assert baseline_prefix_cache_refusal(_suite_round(tmp_path, report, SGLANG_OFF)) == ""


@pytest.mark.parametrize(("log", "refused"), [(VLLM_OFF, True), (VLLM_ON, False)], ids=["cache_off", "cache_on"])
def test_a_vllm_suite_round_is_read_as_the_vllm_launch_it_recorded(tmp_path: Path, log: str, refused: bool) -> None:
    launch = _framework_launch("vllm", ("vllm", "serve"), ("--tensor-parallel-size", "8"))
    assert bool(baseline_prefix_cache_refusal(_suite_round(tmp_path, _agentx(), log, launch))) is refused


@pytest.mark.parametrize(
    ("args", "refused"),
    [((("--level", "2"),), True), ((("--level", "2"), ("--enable_prefix_caching",)), False)],
    ids=["cache_off", "cache_on"],
)
def test_an_atom_suite_round_is_read_from_the_argv_its_launch_recorded(
    tmp_path: Path, args: tuple[tuple[str, ...], ...], refused: bool
) -> None:
    launch = _framework_launch("atom", ("python3", "-m", "atom.entrypoints.openai_server"), *args)
    assert bool(baseline_prefix_cache_refusal(_suite_round(tmp_path, _agentx(), None, launch))) is refused


def test_a_suite_round_whose_report_is_off_contract_is_refused_as_unreadable(tmp_path: Path) -> None:
    directory = _suite_round(tmp_path, _agentx(), SGLANG_ON)
    RunLayout(directory).report.write_text("{", encoding="utf-8")
    with pytest.raises(RoundUnreadable):
        baseline_prefix_cache_refusal(directory)


def test_a_directory_that_is_no_run_dir_is_refused_as_unreadable(tmp_path: Path) -> None:
    (tmp_path / "server.log").write_text(SGLANG_OFF, encoding="utf-8")
    with pytest.raises(RoundUnreadable, match="not an inference_suite run_dir"):
        baseline_prefix_cache_refusal(tmp_path)


def test_the_launch_of_a_warmup_round_is_read_when_the_round_has_no_log_of_its_own(tmp_path: Path) -> None:
    directory = _suite_round(tmp_path, _agentx(), None)
    (directory / "warmup_round").mkdir()
    (directory / "warmup_round" / "server.log").write_text(SGLANG_OFF, encoding="utf-8")
    assert "disable_radix_cache=True" in baseline_prefix_cache_refusal(directory)


def test_the_log_in_the_round_directory_wins_over_a_warmup_launch(tmp_path: Path) -> None:
    directory = _suite_round(tmp_path, _agentx(), SGLANG_ON)
    (directory / "warmup_round").mkdir()
    (directory / "warmup_round" / "server.log").write_text(SGLANG_OFF, encoding="utf-8")
    assert baseline_prefix_cache_refusal(directory) == ""


def test_the_refusal_carries_the_agentx_baseline_error_class_name() -> None:
    assert PREFIX_CACHE_OFF_ERROR_CLASS == "agentx_baseline_prefix_cache_off"
