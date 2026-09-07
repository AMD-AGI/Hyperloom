# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Distributed driver contract checks for collective tasks routed through KRC."""

from __future__ import annotations

import ast

from kernelforge.loop.task_preparer import _distributed_static_checks
from kernelforge.resources import resource_path


def _reference_driver() -> str:
    return str(resource_path("examples") / "aiter-allreduce-forge-loop" / "driver.py")


def test_reference_driver_passes_distributed_static_checks():
    ok, reasons = _distributed_static_checks(_reference_driver(), require_ranks=4)
    assert ok is True
    assert reasons == []


def test_reference_driver_self_launches_under_torchrun():
    text = open(_reference_driver(), encoding="utf-8").read()
    assert "torch.distributed.run" in text
    assert "--nproc-per-node=" in text


def test_reference_driver_refuses_to_oversubscribe_gpus():
    text = open(_reference_driver(), encoding="utf-8").read()
    assert "visible < nproc" in text


def test_bench_does_not_resynchronise_between_samples():
    text = open(_reference_driver(), encoding="utf-8").read()
    bench = text.split("def bench_case(")[1].split("def profile_case(")[0]
    assert bench.count("dist.barrier(") == 1
    barrier_at = bench.index("dist.barrier(")
    assert barrier_at < bench.index("start.record()")


def test_parity_gate_and_cross_rank_max_are_present():
    text = open(_reference_driver(), encoding="utf-8").read()
    assert "ReduceOp.MAX" in text or "_reduce_max" in text


def test_bench_times_a_captured_graph():
    text = open(_reference_driver(), encoding="utf-8").read()
    assert "torch.cuda.CUDAGraph()" in text
    bench = text.split("def bench_case(")[1].split("def profile_case(")[0]
    assert "graph.replay()" in bench


def test_inputs_are_rank_distinct():
    text = open(_reference_driver(), encoding="utf-8").read()
    assert "manual_seed(seed + " in text


def test_static_checks_reject_a_mean_reduction_driver(tmp_path):
    driver = tmp_path / "driver.py"
    text = open(_reference_driver(), encoding="utf-8").read()
    text = text.replace("def _reduce_max", "def _reduce_mean").replace("_reduce_max(", "_reduce_mean(")
    text = text.replace("dist.ReduceOp.MAX", "dist.ReduceOp.SUM")
    driver.write_text(text, encoding="utf-8")
    ok, reasons = _distributed_static_checks(str(driver), require_ranks=4)
    assert ok is False
    assert any("MAX" in reason for reason in reasons)


def test_static_checks_ignore_single_rank_tasks():
    ok, reasons = _distributed_static_checks(_reference_driver(), require_ranks=1)
    assert ok is True
    assert reasons == []


def test_reference_driver_destroys_process_group():
    text = open(_reference_driver(), encoding="utf-8").read()
    assert "destroy_distributed_environment" in text


def test_reference_driver_uses_a_torch_distributed_reference():
    text = open(_reference_driver(), encoding="utf-8").read()
    tree = ast.parse(text)
    assert any(
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "dist"
        for node in ast.walk(tree)
    )
