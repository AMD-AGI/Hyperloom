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
                    "workload_kind": "server",
                    "inferencex_path": "/ix ",
                    "envs": {"TP": 8, "EXTRA_VLLM_ARGS": "--x"},
                    "profiler": {"torch_profiler": {"enabled": True}},
                    "server_lifecycle": {"server_ready_timeout_s": 900},
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
        workload_kind="server",
        inferencex_path="/ix",
        envs={"TP": 8, "EXTRA_VLLM_ARGS": "--x"},
        agentx=False,
        agentx_launch=point.agentx_launch,
        torch_profiler=True,
        server_ready_timeout_s=900.0,
    )
    assert dict(point.agentx_launch.env) == {}
    assert point.agentx_launch.executable == ""
    assert point.agentx_launch.unset_env == ()
    assert not point.agentx_launch.composed


def test_absent_fields_read_empty() -> None:
    point = BenchmarkPoint.from_document({"benchmark": {}})

    assert (
        point.framework,
        point.model,
        point.benchmark_script,
        point.workload_kind,
        point.inferencex_path,
        dict(point.envs),
    ) == (
        "",
        "",
        "",
        "",
        "",
        {},
    )
    assert (point.agentx, point.torch_profiler, point.server_ready_timeout_s) == (False, False, None)
    assert (point.agentx_launch.append_args, point.agentx_launch.remove_args) == ((), ())
    assert (point.agentx_launch.replace_args, point.agentx_launch.unset_env) == (False, ())
    assert dict(point.agentx_launch.source_files) == {}


@pytest.mark.parametrize("enabled", ["true", 1, None])
def test_only_a_true_enabled_turns_the_torch_profiler_on(enabled: object) -> None:
    document = {"benchmark": {"profiler": {"torch_profiler": {"enabled": enabled}}}}

    assert BenchmarkPoint.from_document(document).torch_profiler is False


@pytest.mark.parametrize("carried", ["900", True, None])
def test_a_ready_timeout_that_is_not_a_number_is_none(carried: object) -> None:
    document = {"benchmark": {"server_lifecycle": {"server_ready_timeout_s": carried}}}

    assert BenchmarkPoint.from_document(document).server_ready_timeout_s is None


@pytest.mark.parametrize(
    ("agentx", "enabled"),
    [("enable", True), (True, True), ({"enabled": False}, False), ({}, True), ("off", False)],
)
def test_agentx_follows_magpie_spelling(agentx: object, enabled: bool) -> None:
    assert BenchmarkPoint.from_document({"benchmark": {"agentx": agentx}}).agentx is enabled


def test_agentx_launch_is_read_from_the_overrides() -> None:
    point = BenchmarkPoint.from_document(
        {
            "benchmark": {
                "agentx": {
                    "launch_overrides": {
                        "env": {"A": "1"},
                        "executable": "/py",
                        "append_args": ["--a", 2],
                        "remove_args": ["--b"],
                        "replace_args": True,
                        "unset_env": ["B"],
                        "source_files": {"/dst.py": "/src.py"},
                    }
                }
            }
        }
    )

    assert point.agentx
    assert dict(point.agentx_launch.env) == {"A": "1"}
    assert point.agentx_launch.executable == "/py"
    assert point.agentx_launch.append_args == ("--a", "2")
    assert point.agentx_launch.remove_args == ("--b",)
    assert point.agentx_launch.replace_args is True
    assert point.agentx_launch.unset_env == ("B",)
    assert dict(point.agentx_launch.source_files) == {"/dst.py": "/src.py"}
    assert point.agentx_launch.composed


@pytest.mark.parametrize("append_args", [[], ["--x"]])
def test_a_launch_is_composed_once_it_carries_append_args(append_args: list[str]) -> None:
    point = BenchmarkPoint.from_document({"benchmark": {"agentx": {"launch_overrides": {"append_args": append_args}}}})

    assert point.agentx_launch.composed


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
        {"benchmark": {"agentx": {"launch_overrides": {"append_args": "--a"}}}},
        {"benchmark": {"agentx": {"launch_overrides": {"unset_env": "B"}}}},
        {"benchmark": {"agentx": {"launch_overrides": {"source_files": ["/src.py"]}}}},
        {"benchmark": {"profiler": {"torch_profiler": True}}},
        {"benchmark": {"server_lifecycle": "on"}},
        {"benchmark": {"profiler": "x"}},
        {"benchmark": {"profiler": {"torch_profiler": "x"}}},
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
