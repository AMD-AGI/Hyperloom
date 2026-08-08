###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Tests for the generated torchrun driver of a collective kernel.

The generated rig must be valid Python before forge ever runs it: a syntax error
surfaces deep inside the loop where it is expensive to diagnose.
"""

from __future__ import annotations

import ast
import py_compile
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from collective_driver_generator import generate_collective_driver  # noqa: E402


def _candidate(**extra) -> dict:
    item = {
        "name": "hipLaunchKernel->_ZN5aiter18all_reduce_kernel... (Synthetic Op)",
        "device_kernel_name": "all_reduce_cross_device",
        "source_file": "aiter/dist/device_communicators/custom_all_reduce.py",
        "source_line": 811,
        "source_function": "fused_ar_rms",
        "gpu_pct": 3.2,
        "kernel_contract": {"kind": "collective", "collective_op": "all_reduce", "world_size": 4},
        "input_shapes": [{"shape": "(4096, 7168)", "call_num": 63}],
        "input_dtypes": ["bf16"],
    }
    item.update(extra)
    return item


def _gen(tmp_path: Path, **extra) -> tuple[str, str]:
    res = generate_collective_driver(_candidate(**extra), tmp_path, tp=8)
    return Path(res["driver"]).read_text(encoding="utf-8"), Path(res["program"]).read_text(encoding="utf-8")


# --- The rig must be valid, runnable Python ----------------------------------


def test_generated_driver_compiles(tmp_path):
    res = generate_collective_driver(_candidate(), tmp_path)
    py_compile.compile(res["driver"], doraise=True)


def test_generated_driver_exposes_the_expected_entry_points(tmp_path):
    driver, _ = _gen(tmp_path)
    tree = ast.parse(driver)
    funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for name in ("self_launch", "init_worker", "make_inputs", "snr_db",
                 "run_candidate", "run_reference", "check_case", "bench_case",
                 "profile_case", "build_parser", "main"):
        assert name in funcs, name


# --- forge-loop's CLI contract ------------------------------------------------
#
# forge-loop invokes the driver as
#     <driver> --warmup N --iters N --bench-mode
# (kernel_agents/loop/task_preparer.py), passes --repeat when --bench-repeat > 1,
# and requires a permissive parser. A mismatch here fails inside the loop, where
# it is expensive to diagnose.


@pytest.mark.parametrize(
    "flag",
    ["--bench-mode", "--profile-run", "--warmup", "--iters", "--repeat", "--snr-threshold"],
)
def test_driver_accepts_every_flag_forge_loop_passes(tmp_path, flag):
    driver, _ = _gen(tmp_path)
    assert flag in driver


def test_driver_uses_a_permissive_parser(tmp_path):
    """An unknown flag from a future loop revision must not fail the run."""
    driver, _ = _gen(tmp_path)
    assert "parse_known_args" in driver


def test_bench_mode_is_not_named_benchmark(tmp_path):
    """forge-loop passes --bench-mode; --benchmark would be an argparse error."""
    driver, _ = _gen(tmp_path)
    assert '"--benchmark"' not in driver


def test_correctness_is_the_default_mode(tmp_path):
    """No flags => correctness, matching the reference driver's behaviour."""
    driver, _ = _gen(tmp_path)
    assert "if args.profile_run:" in driver
    assert "elif args.bench_mode:" in driver


def test_run_candidate_is_left_unimplemented_and_points_at_the_call_site(tmp_path):
    """Signature cannot be inferred from a trace, so it must not be guessed."""
    driver, _ = _gen(tmp_path)
    assert "raise NotImplementedError" in driver
    assert "custom_all_reduce.py(811): fused_ar_rms" in driver


# --- Contract-driven substitution --------------------------------------------


@pytest.mark.parametrize(
    "op,expected",
    [
        ("all_reduce", "dist.all_reduce(out"),
        ("all_gather", "dist.all_gather_into_tensor(gathered"),
        ("reduce_scatter", "dist.reduce_scatter_tensor(scattered"),
        ("all_to_all", "dist.all_to_all_single(shuffled"),
        ("broadcast", "dist.broadcast(out"),
    ],
)
def test_reference_matches_the_collective_op(tmp_path, op, expected):
    driver, _ = _gen(tmp_path, kernel_contract={"kind": "collective", "collective_op": op, "world_size": 4})
    assert expected in driver
    py_compile.compile(str(tmp_path / "driver.py"), doraise=True)


def test_unknown_op_falls_back_to_all_reduce(tmp_path):
    driver, _ = _gen(tmp_path, kernel_contract={"kind": "collective", "collective_op": "exotic_op"})
    assert "dist.all_reduce(out" in driver


def test_caller_tp_overrides_the_contract(tmp_path):
    # The contract's world_size may be the constant 2 that TraceLens stamps on
    # any multi-GPU kernel, so the session's real TP has to win. Benchmarking an
    # 8-rank all-reduce on 2 ranks would measure a different regime entirely.
    driver, _ = _gen(tmp_path, kernel_contract={"kind": "collective", "collective_op": "all_reduce", "world_size": 2})
    assert "WORLD_SIZE = 8" in driver


def test_world_size_comes_from_contract_when_tp_is_unusable(tmp_path):
    res = generate_collective_driver(
        _candidate(kernel_contract={"kind": "collective", "collective_op": "all_reduce", "world_size": 4}),
        tmp_path,
        tp=0,
    )
    assert "WORLD_SIZE = 4" in Path(res["driver"]).read_text(encoding="utf-8")


def test_world_size_falls_back_to_tp_when_contract_is_silent(tmp_path):
    res = generate_collective_driver(_candidate(kernel_contract={"kind": "collective"}), tmp_path, tp=8)
    assert "WORLD_SIZE = 8" in Path(res["driver"]).read_text(encoding="utf-8")


def test_shapes_are_taken_from_the_trace(tmp_path):
    driver, _ = _gen(tmp_path)
    assert "[[4096, 7168]]" in driver


def test_shapes_fall_back_to_a_token_sweep(tmp_path):
    """No usable trace shape must still yield a decode-to-prefill sweep."""
    driver, _ = _gen(tmp_path, input_shapes=[], shapes=[])
    assert "[1, 7168]" in driver and "[4096, 7168]" in driver


def test_dtype_is_detected(tmp_path):
    driver, _ = _gen(tmp_path, input_dtypes=["torch.float16"])
    assert 'DTYPE = "fp16"' in driver


def test_bench_times_a_captured_graph(tmp_path):
    # forge's preflight counts real CUDAGraph replays and rejects a driver that
    # times eagerly, so the generated rig has to capture and replay by itself
    # rather than leaving the agent to retrofit a harness.
    driver, program = _gen(tmp_path)
    assert "def capture_chain(" in driver
    assert "from aiter.dist.parallel_state import graph_capture" in driver
    assert "torch.cuda.CUDAGraph()" in driver
    assert "graph.replay()" in driver
    # The chain total must be normalised back to a per-call figure.
    assert "start.elapsed_time(end) / chain" in driver
    assert "Timing replays a captured graph" in program


def test_capture_failure_falls_back_to_eager(tmp_path):
    driver, _ = _gen(tmp_path)
    assert "return None" in driver.split("def capture_chain(")[1].split("def bench_case(")[0]
    assert "if graph is not None:" in driver


# --- Gates that must survive generation --------------------------------------


def test_parity_gate_and_cross_rank_max_are_present(tmp_path):
    driver, _ = _gen(tmp_path)
    assert "SNR_FLOOR_DB = 30.0" in driver
    # A collective is bounded by its slowest rank, so medians reduce with MAX.
    assert "dist.ReduceOp.MAX" in driver


def test_inputs_are_rank_distinct(tmp_path):
    """Seeding by rank is what stops a rank-dropping collective passing parity."""
    driver, _ = _gen(tmp_path)
    assert "manual_seed(seed + ctx.rank)" in driver


def test_driver_self_launches_under_torchrun(tmp_path):
    driver, _ = _gen(tmp_path)
    assert "torch.distributed.run" in driver
    assert "--nproc-per-node=" in driver


def test_driver_refuses_to_oversubscribe_gpus(tmp_path):
    """Two ranks on one device would measure intra-device copies, and the
    resulting 'speedup' would not transfer to the real multi-GPU path."""
    driver, _ = _gen(tmp_path)
    assert "visible < WORLD_SIZE" in driver
    assert "world_size > visible" in driver
    # Each rank must own a device, so binding follows LOCAL_RANK rather than a
    # modulo of the visible count.
    assert "rank % torch.cuda.device_count()" not in driver
    assert 'os.environ.get("LOCAL_RANK", rank)' in driver


# --- Task brief ---------------------------------------------------------------


def test_program_brief_carries_the_call_site_and_gates(tmp_path):
    _driver, program = _gen(tmp_path)
    assert "custom_all_reduce.py(811): fused_ar_rms" in program
    assert "all_reduce" in program
    assert "30.0 dB" in program
    assert "run_candidate" in program


def test_program_brief_handles_an_unresolved_source(tmp_path):
    _driver, program = _gen(tmp_path, source_file="", source_line=None, source_function=None)
    assert "unresolved" in program
