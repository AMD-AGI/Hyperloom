###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Tests for the generated torchrun collective driver."""

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
    res = generate_collective_driver(_candidate(), tmp_path, tp=8)
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


@pytest.mark.parametrize(
    "flag",
    ["--bench-mode", "--profile-run", "--warmup", "--iters", "--repeat", "--snr-threshold"],
)
def test_driver_accepts_every_flag_forge_loop_passes(tmp_path, flag):
    driver, _ = _gen(tmp_path)
    assert flag in driver


def test_driver_uses_a_strict_parser(tmp_path):
    driver, _ = _gen(tmp_path)
    assert "parse_args(argv)" in driver


def test_bench_mode_is_not_named_benchmark(tmp_path):
    """forge-loop passes --bench-mode; --benchmark would be an argparse error."""
    driver, _ = _gen(tmp_path)
    assert '"--benchmark"' not in driver


def test_correctness_is_the_default_mode(tmp_path):
    """No flags => correctness, matching the reference driver's behaviour."""
    driver, _ = _gen(tmp_path)
    assert "if args.profile_run:" in driver
    assert "elif args.bench_mode:" in driver


def test_driver_reports_runtime_failures_as_json(tmp_path):
    """Rank-zero failures must remain machine-readable."""
    driver, _ = _gen(tmp_path)
    main = next(
        node
        for node in ast.parse(driver).body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    assert any(
        isinstance(node, ast.Try) and node.handlers
        for node in ast.walk(main)
    )
    assert "error_class" in driver
    assert "allow_nan=False" in driver


def test_run_candidate_is_left_unimplemented_and_points_at_the_call_site(tmp_path):
    """Signature cannot be inferred from a trace, so it must not be guessed."""
    driver, _ = _gen(tmp_path)
    assert "raise NotImplementedError" in driver
    assert "custom_all_reduce.py(811): fused_ar_rms" in driver


# --- Contract-driven substitution --------------------------------------------


def test_reference_uses_all_reduce(tmp_path):
    driver, _ = _gen(tmp_path)
    assert "dist.all_reduce(out" in driver
    py_compile.compile(str(tmp_path / "driver.py"), doraise=True)


def test_driver_preserves_prefill_and_decode_shapes(tmp_path):
    """Each traced workload regime must remain a benchmark case."""
    driver, _ = _gen(
        tmp_path,
        input_shapes=[
            {"shape": "(1024, 5120)", "call_num": 40},
            {"shape": "(64, 5120)", "call_num": 40},
        ],
        input_dtypes=["bf16", "bf16"],
    )

    assert "CASES = [[1024, 5120], [64, 5120]]" in driver


@pytest.mark.parametrize(
    "op",
    ["all_to_all", "broadcast", "gather", "exotic_op"],
)
def test_rejects_unsupported_collective_ops(tmp_path, op):
    contract = {"kind": "collective", "collective_op": op, "world_size": 4}
    with pytest.raises(ValueError, match="unsupported collective operation"):
        _gen(tmp_path, kernel_contract=contract)


@pytest.mark.parametrize(
    "op, reference",
    [
        ("all_reduce", "dist.all_reduce(out, op=dist.ReduceOp.SUM, group=ctx.group)"),
        ("reduce_scatter", "dist.reduce_scatter_tensor(out, src, op=dist.ReduceOp.SUM, group=ctx.group)"),
        ("all_gather", "dist.all_gather_into_tensor(out, src, group=ctx.group)"),
    ],
)
def test_each_supported_op_gets_its_own_distributed_reference(tmp_path, op, reference):
    """Parity is only meaningful against a reference that is itself distributed."""
    contract = {"kind": "collective", "collective_op": op, "world_size": 4}
    driver, program = _gen(tmp_path, kernel_contract=contract)
    body = driver.split("def run_reference(")[1].split("def check_case(")[0]
    assert reference in body
    assert f"torch.distributed.{op}" in program


@pytest.mark.parametrize(
    "op, numerator, gathered",
    [
        ("all_reduce", "2", "False"),
        ("reduce_scatter", "1", "False"),
        ("all_gather", "1", "True"),
    ],
)
def test_bench_reports_bytes_and_bus_bandwidth(tmp_path, op, numerator, gathered):
    """Latency alone cannot separate a faster transfer from a cheaper barrier."""
    contract = {"kind": "collective", "collective_op": op, "world_size": 4}
    driver, _ = _gen(tmp_path, kernel_contract=contract)

    assert f"BUSBW_NUMERATOR = {numerator}" in driver
    assert f"GATHERED_OUTPUT = {gathered}" in driver
    assert "def case_bandwidth(" in driver
    assert 'result["bandwidth"]' in driver
    assert "algbw_gbps" in driver and "busbw_gbps" in driver
    assert "case_bw:" in driver


def test_reduce_scatter_requires_a_divisible_leading_extent(tmp_path):
    """An indivisible extent would compare against a truncated reference."""
    contract = {"kind": "collective", "collective_op": "reduce_scatter", "world_size": 8}
    with pytest.raises(ValueError, match="must divide across 8 ranks"):
        _gen(
            tmp_path,
            kernel_contract=contract,
            input_shapes=[{"shape": "(1023, 5120)", "call_num": 40}],
            input_dtypes=["bf16"],
        )


def test_caller_tp_overrides_the_contract(tmp_path):
    """The session TP defines the measured world size."""
    driver, _ = _gen(tmp_path, kernel_contract={"kind": "collective", "collective_op": "all_reduce", "world_size": 2})
    assert "WORLD_SIZE = 8" in driver


def test_unusable_tp_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="tp must be greater than one"):
        generate_collective_driver(_candidate(), tmp_path, tp=0)


def test_explicit_tp_does_not_require_contract_world_size(tmp_path):
    contract = {"kind": "collective", "collective_op": "all_reduce"}
    res = generate_collective_driver(
        _candidate(kernel_contract=contract), tmp_path, tp=8
    )
    assert "WORLD_SIZE = 8" in Path(res["driver"]).read_text(encoding="utf-8")


def test_shapes_are_taken_from_the_trace(tmp_path):
    driver, _ = _gen(tmp_path)
    assert "[[4096, 7168]]" in driver


def test_missing_trace_shape_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="no traced input shape"):
        _gen(tmp_path, input_shapes=[], shapes=[])


def test_dtype_is_detected(tmp_path):
    driver, _ = _gen(tmp_path, input_dtypes=["torch.float16"])
    assert 'DTYPE = "fp16"' in driver


def test_bench_times_a_captured_graph(tmp_path):
    """The benchmark must execute at least the requested graph replays."""
    driver, program = _gen(tmp_path)
    assert "def capture_chain(" in driver
    assert "torch.cuda.CUDAGraph()" in driver
    assert "torch.cuda.graph(graph, stream=capture_stream)" in driver
    assert "aiter.dist.parallel_state" not in driver
    bench = driver.split("def bench_case(")[1].split("def profile_case(")[0]
    assert "chain = max(1, iters)" in bench
    assert "capture_chain(inputs, ctx, chain)" in bench
    assert "graph.replay()" in bench
    assert "start.elapsed_time(end) / chain" in driver
    assert "Latency replays every iteration as one captured chain" in program


def test_bench_does_not_resynchronise_between_samples(tmp_path):
    """One barrier per sample would hide arrival skew and reward its removal.

    A barrier immediately before each replay resets the ranks to a fully
    synchronised state, which is the condition under which deleting an internal
    barrier looks free. The timed region must hold at most the single entry
    barrier.
    """
    driver, program = _gen(tmp_path)
    bench = driver.split("def bench_case(")[1].split("def profile_case(")[0]
    assert bench.count("dist.barrier(group=ctx.group)") == 1
    barrier_at = bench.index("dist.barrier(group=ctx.group)")
    assert barrier_at < bench.index("start.record()")
    assert "Do not add a barrier inside the timed region" in program


def test_parity_validates_two_back_to_back_calls(tmp_path):
    """Scratch reused across consecutive collectives needs two live calls."""
    driver, program = _gen(tmp_path)
    check = driver.split("def check_case(")[1].split("def capture_chain(")[0]
    assert "seeds = (seed, seed + 1)" in check
    assert "got = [run_candidate(inputs, ctx) for inputs in pending]" in check
    # Both results are compared only after both calls have been issued.
    assert check.index("got = [run_candidate") < check.index("min(snr_db(")
    assert "Parity issues two calls back to back" in program


def test_capture_has_no_eager_timing_path(tmp_path):
    driver, _ = _gen(tmp_path)
    capture = driver.split("def capture_chain(")[1].split("def bench_case(")[0]
    assert "return graph" in capture
    assert "return None" not in capture


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
    assert 'os.environ["LOCAL_RANK"]' in driver


# --- Task brief ---------------------------------------------------------------


def test_program_brief_carries_the_call_site_and_gates(tmp_path):
    _driver, program = _gen(tmp_path)
    assert "custom_all_reduce.py(811): fused_ar_rms" in program
    assert "all_reduce" in program
    assert "30.0 dB" in program
    assert "run_candidate" in program


def test_unresolved_source_is_rejected(tmp_path):
    with pytest.raises(
        ValueError,
        match="requires source_file and source_function",
    ):
        _gen(
            tmp_path,
            source_file="",
            source_line=None,
            source_function=None,
        )


def test_existing_driver_is_not_overwritten(tmp_path):
    """A resumed campaign must preserve an implemented driver."""
    generate_collective_driver(_candidate(), tmp_path, tp=8)
    driver = tmp_path / "driver.py"
    driver.write_text("# implemented\n", encoding="utf-8")

    generate_collective_driver(_candidate(), tmp_path, tp=8)

    assert driver.read_text(encoding="utf-8") == "# implemented\n"
