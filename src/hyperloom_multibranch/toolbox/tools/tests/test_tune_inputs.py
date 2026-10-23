# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What ``kernel_tune`` reads of a test's landed output: the workload its server ran, and the shapes its log reports untuned,
read by Forge's own reader from the lines of a real server log."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from hyperloom_multibranch.toolbox.tests.kernel_tune_rig import (
    ENV,
    LOG_SHAPES,
    SERVER_LOG,
    TABLE,
    TUNER,
    landed_test,
)
from hyperloom_multibranch.toolbox.tools.base import ToolFailure
from hyperloom_multibranch.toolbox.tools.tune_inputs import (
    DENSE_TABLES,
    demand_from_logs,
    explicit_demand,
    served_workload,
    write_demand_of,
)

BF16 = "AITER_CONFIG_GEMM_BF16"


def keys(demand: dict) -> set[tuple[str, str, str]]:
    return {(key["M"], key["N"], key["K"]) for entry in demand["demands"] for key in entry["keys"]}


def test_the_demand_is_what_forges_reader_takes_of_the_logs_misses_for_the_dense_tables(tmp_path: Path) -> None:
    landed = landed_test(tmp_path, tmp_path / "model")

    demand = demand_from_logs(landed)

    (entry,) = demand["demands"]
    assert (entry["table"], entry["tuner"], entry["env_var"]) == (TABLE, TUNER, ENV)
    assert keys(demand) == LOG_SHAPES and entry["miss_count"] == len(LOG_SHAPES)


def test_the_logs_of_every_point_are_read_together(tmp_path: Path) -> None:
    landed = landed_test(tmp_path, tmp_path / "model")
    other = SERVER_LOG.replace("M:496", "M:480")
    (landed / "variant_00_c32").mkdir(exist_ok=True)
    (landed / "variant_00_c32" / "server_1.log").write_text(other, encoding="utf-8")

    demand = demand_from_logs(landed)

    assert {m for m, _, _ in keys(demand)} == {"496", "480"}


def test_a_log_with_no_miss_of_a_dense_table_is_a_failure_by_name(tmp_path: Path) -> None:
    boot = "[aiter] import [module_aiter_core] under /sgl-workspace/aiter/aiter/jit/module_aiter_core.so\n"
    with pytest.raises(ToolFailure, match="report no untuned dense GEMM shape"):
        demand_from_logs(landed_test(tmp_path, tmp_path / "model", log=boot))


def test_a_test_that_landed_no_server_log_is_a_failure_by_name(tmp_path: Path) -> None:
    with pytest.raises(ToolFailure, match="landed no server log"):
        demand_from_logs(landed_test(tmp_path, tmp_path / "model", log=None))


def test_only_dense_tables_are_asked_for() -> None:
    assert ENV in DENSE_TABLES and BF16 in DENSE_TABLES
    assert "AITER_CONFIG_FMOE" not in DENSE_TABLES and all(env.startswith("AITER_CONFIG_GEMM") for env in DENSE_TABLES)


def test_explicit_shapes_become_one_demand_entry_per_table_ranked_by_how_often_they_were_named() -> None:
    shapes = [
        {"env": ENV, "M": 64, "N": 5120, "K": 5120},
        {"env": BF16, "M": 8, "N": 256, "K": 5120, "dtype": "torch.bfloat16", "otype": "torch.float32"},
        {"env": ENV, "M": 128, "N": 5120, "K": 5120},
        {"env": ENV, "M": 128, "N": 5120, "K": 5120},
    ]

    demand = explicit_demand(shapes)

    by_env = {entry["env_var"]: entry for entry in demand["demands"]}
    assert set(by_env) == {ENV, BF16} and by_env[ENV]["tuner"] == TUNER and by_env[BF16]["tuner"] == "sglang_dense_bf16"
    assert [(k["M"], k["requests"]) for k in by_env[ENV]["keys"]] == [("128", 2), ("64", 1)]
    assert by_env[BF16]["keys"][0]["dtype"] == "torch.bfloat16" and by_env[BF16]["miss_count"] == 1
    assert by_env[ENV]["key_schema"] == ["M", "N", "K"]


def test_a_demand_is_written_as_the_document_forge_loads(tmp_path: Path) -> None:
    from kernelforge.gemm_tune.evidence import demand_for_tuner, demand_shapes, load_demand

    (entry,) = demand_from_logs(landed_test(tmp_path, tmp_path / "model"))["demands"]
    path = write_demand_of(entry, tmp_path / "demand.json")

    loaded = load_demand(path)
    wanted = demand_for_tuner(loaded, TUNER)
    assert wanted is not None and {(s["M"], s["N"], s["K"]) for s in demand_shapes(wanted, bucket=False)} == {
        (int(m), int(n), int(k)) for m, n, k in LOG_SHAPES
    }
    assert json.loads(path.read_text())["schema"] == loaded["schema"]


def test_the_workload_is_the_one_the_points_of_the_test_ran_with_the_highest_concurrency(tmp_path: Path) -> None:
    workload = served_workload(landed_test(tmp_path, tmp_path / "model", concs=(32, 64, 128)))

    assert (workload.framework, workload.precision, workload.tp, workload.conc) == ("sglang", "fp8", 1, 128)
    assert workload.model_path == str(tmp_path / "model")


def test_a_test_with_no_readable_point_configuration_is_a_failure_by_name(tmp_path: Path) -> None:
    with pytest.raises(ToolFailure, match="landed no readable point configuration"):
        served_workload(landed_test(tmp_path, tmp_path / "model", concs=()))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"framework": "vllm"}, "tunes aiter tables for sglang; the test served 'vllm'"),
        ({"precision": ""}, "names no precision"),
        ({"model": "/nonexistent/model"}, "is not a model directory"),
        ({"envs": {"TP": 0, "CONC": "1"}}, "no usable TP"),
        ({"envs": {"TP": 1, "CONC": "many"}}, "no usable CONC"),
    ],
)
def test_a_workload_this_tool_cannot_tune_for_is_a_failure_that_says_which(
    tmp_path: Path, change: dict, message: str
) -> None:
    landed = landed_test(tmp_path, tmp_path / "model", concs=(64,))
    point = landed / "variant_00_c64" / "config.yaml"
    document = yaml.safe_load(point.read_text())
    document["benchmark"].update(change)
    point.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(ToolFailure, match=message):
        served_workload(landed)


def test_the_table_asked_for_most_comes_first_so_a_short_time_reaches_it_first() -> None:
    shapes = [
        {"env": ENV, "M": 64, "N": 5120, "K": 5120},
        {"env": BF16, "M": 8, "N": 256, "K": 5120},
        {"env": BF16, "M": 16, "N": 256, "K": 5120},
    ]

    demand = explicit_demand(shapes)

    assert [entry["env_var"] for entry in demand["demands"]] == [BF16, ENV]
