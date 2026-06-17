# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for aiter op_test harness generation (compiled-kernel step A).

Validates that maybe_generate_harness recognizes the aiter @benchmark/
run_perftest idiom and emits a Forge-contract harness. Runtime execution of the
generated harness requires aiter's HIP/CK JIT runtime (image-gated); these tests
only cover generation + structure.
"""

from __future__ import annotations

import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import harness_generator as hg  # noqa: E402


_AITER_TEST_SRC = '''import torch
import torch.nn.functional as F
import aiter
from aiter.test_common import run_perftest, checkAllclose, benchmark


def torch_silu_and_mul(input):
    d = input.shape[-1] // 2
    x, y = input.split([d, d], dim=-1)
    return F.silu(x) * y


@benchmark()
def test_scaled_silu_and_mul(m, n, dtype):
    input = torch.randn(m, n, dtype=dtype, device="cuda")
    out = torch.empty((m, n // 2), dtype=dtype, device="cuda")
    ref = torch_silu_and_mul(input)
    _, us_aiter = run_perftest(aiter.scaled_silu_and_mul, out, input)
    err = checkAllclose(ref.to(torch.float), out.to(torch.float))
    ret = {}
    ret["us"] = us_aiter
    ret["err"] = err
    return ret


for m in [128, 256]:
    for n in [4096]:
        test_scaled_silu_and_mul(m, n, torch.bfloat16)
'''


def _gen(tmp_path, candidate):
    bench = tmp_path / "aiter" / "op_tests" / "test_activation.py"
    bench.parent.mkdir(parents=True)
    bench.write_text(_AITER_TEST_SRC)
    src = tmp_path / "aiter" / "csrc" / "hip_act_and_mul.cuh"
    src.parent.mkdir(parents=True)
    src.write_text("// kernel\n")
    return hg.maybe_generate_harness(
        benchmark_file=str(bench), candidate=candidate,
        source_file=str(src), out_dir=tmp_path / "out",
    )


def test_generates_aiter_harness(tmp_path):
    hr = _gen(tmp_path, {"input_shapes": {"M": 8192, "N": 4096}, "precision": "bf16"})
    assert hr is not None, "expected an aiter harness to be generated"
    code = Path(hr.harness_path).read_text()
    # Reuses the aiter test fn + emits the Forge benchmark contract.
    assert "test_scaled_silu_and_mul" in code
    assert "GEAK_RESULT_LATENCY_MS" in code
    assert "aiter.scaled_silu_and_mul" in code
    # Candidate shapes baked into the call.
    assert "8192" in code and "4096" in code
    # dtype mapped from precision.
    assert "torch.bfloat16" in code
    # Harness is syntactically valid.
    import ast
    ast.parse(code)
    assert hr.test_command.endswith("--correctness")


def test_dtype_fp16_mapping(tmp_path):
    hr = _gen(tmp_path, {"input_shapes": {"M": 128, "N": 256}, "precision": "fp16"})
    assert hr is not None
    assert "torch.float16" in Path(hr.harness_path).read_text()


def test_non_aiter_file_not_handled_by_aiter_path(tmp_path):
    # A triton-ish file without aiter import must not be mistaken for aiter.
    bench = tmp_path / "bench_triton.py"
    bench.write_text(
        "import torch\n"
        "from aiter.test_common import benchmark\n"  # has benchmark but...
    )
    # No @benchmark functions -> aiter path returns None (falls through).
    src = tmp_path / "k.py"
    src.write_text("x=1\n")
    hr = hg.maybe_generate_harness(
        benchmark_file=str(bench), candidate={"input_shapes": {"M": 8}},
        source_file=str(src), out_dir=tmp_path / "out2",
    )
    assert hr is None
