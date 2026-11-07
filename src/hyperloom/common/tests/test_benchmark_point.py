# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hyperloom.common.benchmark_config import (
    BENCHMARK_CONFIG_ENV,
    BenchmarkPoint,
    PointError,
    configured_benchmark,
    read_benchmark_point,
)


def _write(tmp_path: Path, document: object) -> Path:
    path = tmp_path / "point.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def test_reads_every_field_of_a_fixed_point(tmp_path: Path) -> None:
    point = read_benchmark_point(
        _write(
            tmp_path,
            {
                "benchmark": {
                    "framework": "vllm",
                    "model": "/models/m",
                    "precision": "bf16",
                    "runner_type": "mi355x",
                    "benchmark_script": "vllm_mi300x.sh",
                    "envs": {"TP": 8, "EXTRA_VLLM_ARGS": "--x"},
                }
            },
        )
    )

    assert point == BenchmarkPoint(
        framework="vllm",
        model="/models/m",
        precision="bf16",
        runner_type="mi355x",
        benchmark_script="vllm_mi300x.sh",
        envs={"TP": 8, "EXTRA_VLLM_ARGS": "--x"},
        agentx=False,
        agentx_launch=point.agentx_launch,
    )
    assert dict(point.agentx_launch.env) == {}
    assert point.agentx_launch.executable == ""


def test_absent_fields_read_empty() -> None:
    point = BenchmarkPoint.from_document({"benchmark": {}})

    assert (point.framework, point.model, point.benchmark_script, dict(point.envs), point.agentx) == (
        "",
        "",
        "",
        {},
        False,
    )


@pytest.mark.parametrize(
    ("agentx", "enabled"),
    [("enable", True), (True, True), ({"enabled": False}, False), ({}, True), ("off", False)],
)
def test_agentx_follows_magpie_spelling(agentx: object, enabled: bool) -> None:
    assert BenchmarkPoint.from_document({"benchmark": {"agentx": agentx}}).agentx is enabled


def test_agentx_launch_is_read_from_the_overrides() -> None:
    point = BenchmarkPoint.from_document(
        {"benchmark": {"agentx": {"launch_overrides": {"env": {"A": "1"}, "executable": "/py"}}}}
    )

    assert point.agentx
    assert dict(point.agentx_launch.env) == {"A": "1"}
    assert point.agentx_launch.executable == "/py"


def test_the_point_is_a_copy_the_reader_cannot_change() -> None:
    document = {"benchmark": {"envs": {"TP": 1}}}
    point = BenchmarkPoint.from_document(document)

    document["benchmark"]["envs"]["TP"] = 2
    with pytest.raises(TypeError):
        point.envs["TP"] = 3  # type: ignore[index]
    assert point.envs["TP"] == 1


@pytest.mark.parametrize(
    "document",
    [
        None,
        [],
        {},
        {"benchmark": None},
        {"benchmark": "x"},
        {"benchmark": {"envs": "x"}},
        {"benchmark": {"agentx": {"launch_overrides": []}}},
    ],
)
def test_a_misshapen_point_is_refused(document: object) -> None:
    with pytest.raises(PointError, match="the point"):
        BenchmarkPoint.from_document(document)


def test_a_refusal_names_the_file(tmp_path: Path) -> None:
    path = _write(tmp_path, {"other": {}})

    with pytest.raises(PointError, match=str(path)):
        read_benchmark_point(path)


def test_the_operator_config_is_refused_through_the_same_check(tmp_path: Path) -> None:
    path = _write(tmp_path, {"benchmark": []})

    with pytest.raises(ValueError, match="must contain a benchmark mapping"):
        configured_benchmark({BENCHMARK_CONFIG_ENV: str(path)})
