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
    GRAPH_SIZES,
    HIT_LOG,
    HIT_SHAPES,
    LOG_SHAPES,
    SERVER_ARGS,
    SERVER_LOG,
    TABLE,
    TUNER,
    landed_test,
)
from hyperloom_multibranch.toolbox.tools.base import ToolFailure
from hyperloom_multibranch.toolbox.tools.tune_inputs import (
    DENSE_TABLES,
    cuda_graph_batch_sizes,
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


# --- the decode CUDA-graph batch sizes of the servers ---------------------------------------------------------------------


def test_the_graph_sizes_are_the_decode_list_of_the_servers_resolved_arguments_up_to_the_size_the_workload_replays() -> (
    None
):
    """A step of ``conc`` requests replays the graph of the smallest captured size at least ``conc``: 128 here (it is in the
    list), so the sizes above it are never replayed, and neither the prefill list nor an unset flag is read as the list."""
    assert cuda_graph_batch_sizes(SERVER_ARGS, 128) == tuple(size for size in GRAPH_SIZES if size <= 128)
    assert cuda_graph_batch_sizes(SERVER_ARGS, 100) == tuple(size for size in GRAPH_SIZES if size <= 104)
    assert cuda_graph_batch_sizes(SERVER_ARGS, 1) == (1,)


def test_a_workload_above_every_captured_size_replays_all_of_them() -> None:
    assert cuda_graph_batch_sizes(SERVER_ARGS, 100_000) == GRAPH_SIZES


@pytest.mark.parametrize(
    "line",
    [
        "server_args=ServerArgs(model_path='/m', cuda_graph_max_bs=160, cuda_graph_bs=[1, 2, 4, 8, 16, 24], tp_size=1)",
        "server_args={'tp_size': 1, 'cuda_graph_bs': [1, 2, 4, 8, 16, 24], 'disable_cuda_graph': False}",
    ],
)
def test_the_list_of_an_older_layout_is_read_too(line: str) -> None:
    assert cuda_graph_batch_sizes(line, 64) == (1, 2, 4, 8, 16, 24)


@pytest.mark.parametrize(
    "text",
    [
        "",
        SERVER_LOG,
        "server_args={'cuda_graph_bs_decode': None, 'cuda_graph_bs_prefill': None}",
        "server_args={'cuda_graph_config': {'decode': {'backend': 'disabled', 'bs': None}, 'prefill': {'bs': [4, 8]}}}",
    ],
)
def test_a_log_that_does_not_say_what_the_decode_graphs_were_captured_for_has_no_sizes(text: str) -> None:
    assert cuda_graph_batch_sizes(text, 64) == ()


def test_the_sizes_of_every_server_log_of_the_test_are_taken_together() -> None:
    other = SERVER_ARGS.replace("'bs': [1, 2, 4", "'bs': [3, 5, 4", 1)
    assert cuda_graph_batch_sizes(SERVER_ARGS + other, 128) == tuple(
        sorted({3, 5, *GRAPH_SIZES} - {136, 144, 256, 512})
    )


def test_every_graph_size_is_added_as_the_m_of_each_n_k_of_a_demand_and_counts_for_no_miss(tmp_path: Path) -> None:
    landed = landed_test(tmp_path, tmp_path / "model", log=SERVER_ARGS + SERVER_LOG)

    demand = demand_from_logs(landed, conc=128)

    (entry,) = demand["demands"]
    sizes = tuple(size for size in GRAPH_SIZES if size <= 128)
    assert demand["cuda_graph_batch_sizes"] == list(sizes)
    by_group: dict[tuple[str, str], dict[str, int]] = {}
    for key in entry["keys"]:
        by_group.setdefault((key["N"], key["K"]), {})[key["M"]] = key["requests"]
    assert set(by_group) == {(n, k) for _, n, k in LOG_SHAPES}
    for group in by_group.values():
        assert {int(m) for m in group} == {*sizes, 496}, "every size, and the M the log asked for"
        assert group["496"] == 1 and all(group[str(size)] == 0 for size in sizes)
    assert entry["miss_count"] == len(LOG_SHAPES), "an added key is no miss"
    assert entry["distinct_keys"] == len(LOG_SHAPES) * (len(sizes) + 1)


def test_a_size_the_log_asked_for_keeps_its_count(tmp_path: Path) -> None:
    log = SERVER_ARGS + SERVER_LOG.replace("M:496", "M:64")
    demand = demand_from_logs(landed_test(tmp_path, tmp_path / "model", log=log), conc=128)

    (entry,) = demand["demands"]
    counts = {(key["M"], key["N"], key["K"]): key["requests"] for key in entry["keys"]}
    assert counts[("64", "7168", "5120")] == 1 and counts[("32", "7168", "5120")] == 0
    assert len([key for key in entry["keys"] if (key["M"], key["N"], key["K"]) == ("64", "7168", "5120")]) == 1


def test_no_concurrency_given_means_no_sizes_are_added(tmp_path: Path) -> None:
    demand = demand_from_logs(landed_test(tmp_path, tmp_path / "model", log=SERVER_ARGS + SERVER_LOG))

    assert keys(demand) == LOG_SHAPES and demand["cuda_graph_batch_sizes"] == []


def test_a_log_that_names_no_graph_sizes_leaves_the_demand_as_the_log_has_it(tmp_path: Path) -> None:
    demand = demand_from_logs(landed_test(tmp_path, tmp_path / "model"), conc=128)

    assert keys(demand) == LOG_SHAPES and demand["cuda_graph_batch_sizes"] == []


# --- retune: the shapes the tables serve ------------------------------------------------------------------------------------


def hit_keys(demand: dict) -> dict[str, set[tuple[str, str, str]]]:
    return {entry["env_var"]: {(k["M"], k["N"], k["K"]) for k in entry["keys"]} for entry in demand["demands"]}


def test_a_log_whose_tables_served_every_lookup_is_nothing_to_tune_and_the_failure_names_retune(tmp_path: Path) -> None:
    landed = landed_test(tmp_path, tmp_path / "model", log=HIT_LOG)

    with pytest.raises(ToolFailure, match=r"report no untuned dense GEMM shape.*set retune=true"):
        demand_from_logs(landed)


def test_retune_makes_the_rows_the_tables_served_the_demand_of_each_dense_table(tmp_path: Path) -> None:
    demand = demand_from_logs(landed_test(tmp_path, tmp_path / "model", log=HIT_LOG), retune=True)

    assert hit_keys(demand) == HIT_SHAPES
    assert {entry["env_var"]: entry["miss_count"] for entry in demand["demands"]} == {
        ENV: 0,
        "AITER_CONFIG_GEMM_BF16": 0,
    }
    bf16 = next(entry for entry in demand["demands"] if entry["env_var"] == "AITER_CONFIG_GEMM_BF16")
    assert bf16["keys"][0]["dtype"] == "torch.bfloat16" and bf16["keys"][0]["bpreshuffle"] == "False"
    assert demand["served_keys"] == 3, "the per-token a8w8 line prints no q_dtype_w, which keys its table: no key"
    assert demand["served_lines_unread"] == 1


def test_a_served_row_and_a_missed_shape_are_one_demand(tmp_path: Path) -> None:
    landed = landed_test(tmp_path, tmp_path / "model", log=SERVER_LOG + HIT_LOG)

    demand = demand_from_logs(landed, retune=True)

    served = hit_keys(demand)[ENV]
    assert served == LOG_SHAPES | HIT_SHAPES[ENV]
    (blockscale,) = [entry for entry in demand["demands"] if entry["env_var"] == ENV]
    assert blockscale["miss_count"] == len(LOG_SHAPES), "a served row is no miss"


def test_without_retune_a_log_with_misses_and_hits_asks_for_the_misses_only(tmp_path: Path) -> None:
    demand = demand_from_logs(landed_test(tmp_path, tmp_path / "model", log=SERVER_LOG + HIT_LOG))

    assert hit_keys(demand) == {ENV: LOG_SHAPES} and demand["served_keys"] == 0


def test_retune_on_a_log_with_no_lookup_at_all_says_that_the_server_may_not_have_logged_its_hits(
    tmp_path: Path,
) -> None:
    boot = "[aiter] import [module_aiter_core] under /sgl-workspace/aiter/aiter/jit/module_aiter_core.so\n"

    with pytest.raises(ToolFailure, match=r"no dense GEMM lookup.*AITER_LOG_TUNED_CONFIG.*nothing to retune"):
        demand_from_logs(landed_test(tmp_path, tmp_path / "model", log=boot), retune=True)


def test_retune_on_a_log_whose_only_hits_are_of_a_table_it_cannot_key_says_so_and_points_to_shapes(
    tmp_path: Path,
) -> None:
    a8w8 = "".join(line for line in HIT_LOG.splitlines(keepends=True) if "a8w8_bpreshuffle_tuned_gemm" in line)

    with pytest.raises(
        ToolFailure, match=r"1 hit line\(s\) name rows of tables this reader cannot key.*q_dtype_w.*params.shapes"
    ):
        demand_from_logs(landed_test(tmp_path, tmp_path / "model", log=a8w8), retune=True)
