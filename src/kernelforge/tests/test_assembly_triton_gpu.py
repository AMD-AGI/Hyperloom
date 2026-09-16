# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Real Triton/Gluon/HIP capture, execution probes, graph checks and patch replay."""

from __future__ import annotations

import asyncio
import re
import shutil
import subprocess
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.version.hip or not torch.cuda.is_available():
    pytest.skip("requires ROCm PyTorch and an AMD GPU", allow_module_level=True)
triton = pytest.importorskip("triton")
tl = pytest.importorskip("triton.language")

from kernelforge.assembly.prepare import AssemblyPreparationError, prepare_assembly
from kernelforge.config import Config
from kernelforge.loop.validation import run_validation_pipeline
from kernelforge.mcp_server.tools.bench import bench_wallclock


@pytest.mark.parametrize(
    "frontend,wiring",
    [(name, "independent") for name in ("triton", "gluon", "hip")]
    + [("triton", role) for role in ("source_before", "source_after", "candidate")],
)
def test_compiler_capture_wrong_instruction_and_replay(tmp_path, frontend, wiring):
    example = Path(__file__).resolve().parents[3] / "examples/triton2asm-vector-add"
    workspace = tmp_path / "campaign"
    shutil.copytree(example, workspace)
    if frontend == "gluon":
        pytest.importorskip("triton.experimental.gluon")
        path = workspace / "kernel.py"
        source = (
            path.read_text()
            .replace(
                "import triton.language as tl",
                "from triton.experimental import gluon\nfrom triton.experimental.gluon import language as tl",
            )
            .replace("@triton.jit", "@gluon.jit")
        )
        source = source.replace(
            "tl.arange(0, BLOCK)", "tl.arange(0, BLOCK, layout=tl.BlockedLayout([1], [64], [4], [0]))"
        )
        path.write_text(source)
    if frontend == "hip":
        _hip_workspace(workspace)
    if wiring != "independent":
        path = workspace / "driver.py"
        text = path.read_text()
        callback = "source_run" if wiring == "candidate" else "run"
        text = text.replace(
            "row[role] = measure_outputs(callback, reference, repetitions=5)",
            f"row[role] = measure_outputs({callback} if role == {wiring!r} else callback, reference, repetitions=5)",
        )
        path.write_text(text)

    def git(*args, cwd=workspace):
        return subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True).stdout.strip()

    git("init")
    git("config", "user.name", "Forge")
    git("config", "user.email", "forge@example.com")
    git("add", ".")
    git("commit", "-m", "source")
    base = git("rev-parse", "HEAD")
    target = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    driver = str(workspace / "driver.py")

    async def run_preparation():
        return await prepare_assembly(
            config=Config(workspace=str(workspace), gpu_target=target),
            kernel=str(workspace / "kernel.py"),
            driver=driver,
            sources=[],
            base_commit=base,
            threshold=50.0,
            deadline=time.time() + 1800,
        )

    if wiring != "independent":
        with pytest.raises(AssemblyPreparationError, match="numerical execution probe"):
            asyncio.run(run_preparation())
        assert git("rev-parse", "HEAD") == base
        assert not (workspace / "kernel.s").exists()
        assert not (workspace / "forge_experiments/assembly_preparation/result.json").exists()
        return
    record = asyncio.run(run_preparation())
    assert record["origin"] == frontend + "_compiler"
    assert record["build_failure_probe_passed"] and record["execution_probe_passed"]
    for row in record["numerical_execution_probe_evidence"]["cases"]:
        assert not row["candidate"]["finite"]
        assert row["source_before"]["finite"] and row["source_after"]["finite"]
        assert row["source_before"]["oracle_errors"] == row["source_after"]["oracle_errors"] == [0] * 5
    assembly = workspace / "kernel.s"
    original = assembly.read_text()
    edited, count = re.subn(r"\bv_add_f32(?P<encoding>_e32|_e64)?\b", r"v_sub_f32\g<encoding>", original)
    assert count > 0
    assembly.write_text(edited)
    try:
        wrong = asyncio.run(run_validation_pipeline(driver))
        assert wrong.failed_outcome == "correctness_failure", wrong.failed_output
    finally:
        assembly.write_text(original)

    replay = tmp_path / "replay"
    git("clone", str(workspace), str(replay))
    git("checkout", base, cwd=replay)
    patch = tmp_path / "candidate.patch"
    patch.write_bytes(
        subprocess.run(
            ["git", "-C", str(workspace), "diff", "--binary", base, record["preparation_commit"]],
            check=True,
            capture_output=True,
        ).stdout
    )
    git("apply", str(patch), cwd=replay)
    assert not (replay / "forge_experiments").exists()
    report = asyncio.run(run_validation_pipeline(str(replay / "driver.py")))
    assert report.all_passed, report.failed_output
    result = asyncio.run(
        bench_wallclock(
            str(replay / "driver.py"),
            driver_args=["--bench-case", "random"],
            warmup_iters=2,
            bench_iters=10,
            repeat=2,
        )
    )
    assert result["success"], result


def _hip_workspace(workspace):
    (workspace / "vector_add.hip").write_text("""#include <hip/hip_runtime.h>
extern "C" __global__ void add(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}
""")
    source = """from pathlib import Path
import tempfile
import torch
from kernelforge.assembly.hip_source import compile_hip
from kernelforge.assembly.hip import HipKernel
N = 4103
class VectorAdd:
    def __init__(self):
        self.build = tempfile.TemporaryDirectory()
        target = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
        binary = compile_hip(Path(__file__).with_name("vector_add.hip"), Path(self.build.name)/"kernel.hsaco", gpu_target=target)
        self.kernel = HipKernel(binary, "add", ["ptr", "ptr", "ptr", "i32"])
    def __call__(self, a, b, out):
        self.kernel.launch([a.data_ptr(),b.data_ptr(),out.data_ptr(),N],grid=((N+255)//256,1,1),block=(256,1,1),stream=torch.cuda.current_stream().cuda_stream)
    def close(self):
        self.kernel.close()
        self.build.cleanup()
"""
    (workspace / "kernel.py").write_text(source)
    (workspace / "reference.py").write_text(source)
    path = workspace / "driver.py"
    driver = path.read_text().replace("import triton\n", "")
    driver = driver.replace(
        "from kernel import N, VectorAdd, _vector_add",
        "from kernel import N, VectorAdd\nfrom reference import VectorAdd as ReferenceVectorAdd",
    )
    driver = driver.replace(
        "source = _vector_add.warmup(*args, N, BLOCK=256, num_warps=4, grid=(triton.cdiv(N, 256),))",
        "source = ReferenceVectorAdd()",
    )
    driver = driver.replace("source[(triton.cdiv(N, 256), 1, 1)](*args, N, 256)", "source(*args)")
    driver = driver.replace(
        "            rows.append(row)\n        del graph",
        "            rows.append(row)\n        del graph\n        source.close()",
    )
    assert "triton" not in driver
    path.write_text(driver)


@triton.jit
def _atomic_increment(output):
    tl.atomic_add(output, 1)


def test_binding_does_not_double_atomic_output(tmp_path):
    from kernelforge.assembly.triton import TritonAssembly

    target = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    output = torch.zeros(1, dtype=torch.int32, device="cuda")
    export = TritonAssembly(str(tmp_path / "kernel.py"), "kernel.s", target, export=True)
    export(_atomic_increment)[(1,)](output)
    assert output.item() == 1
    candidate = TritonAssembly(str(tmp_path / "kernel.py"), "kernel.s", target)(_atomic_increment)
    output.zero_()
    candidate[(1,)](output)
    assert output.item() == 1
    output.zero_()
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        candidate[(1,)](output)
    output.zero_()
    graph.replay()
    graph.replay()
    assert output.item() == 2
