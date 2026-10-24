# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Prepare a pinned LocalMoE or MegaMoE experiment in a reviewer-owned workspace."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from importlib.util import find_spec
import json
import os
from pathlib import Path
import re
import subprocess
import sys

AVE_REVISIONS = {
    "localmoe": "f2c2496e227d838e0edbc6edf878d9799239af85",
    "megamoe": "809d959251d640108b493b7a558955e6ff7c0964",
}
LLVM_REVISION = "b064d77d623617ddfc3adbec6f77f025c3b9a963"
AITER_REVISION = "eec768a47c1da52e175f9089c640e695f8b572a3"
TORCH_REQUIREMENT = "torch>=2.9.1,<=2.12.0"


def install_aiter(workspace: Path) -> None:
    """Build pinned AITER against the installed Torch and keep its Triton dependency."""
    import torch

    source = workspace / "aiter"
    fetch_source("https://github.com/ROCm/aiter.git", AITER_REVISION, source)
    subprocess.run(["git", "submodule", "update", "--init", "--recursive", "--depth=1"], cwd=source, check=True)
    constraint = workspace / "torch-constraint.txt"
    constraint.write_text(f"torch=={torch.__version__}\n", encoding="utf-8")
    env = {**os.environ, "AITER_USE_SYSTEM_TRITON": "1", "PREBUILD_KERNELS": "0"}
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--constraint",
            str(constraint),
            "setuptools>=77",
            "setuptools_scm[toml]>=10",
            "wheel",
            "-r",
            str(source / "requirements.txt"),
        ],
        env=env,
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-build-isolation", "--no-deps", str(source)],
        env=env,
        check=True,
    )


def prepare_runtime(workspace: Path, suite: str) -> None:
    """Reuse installed GPU packages, or install missing ones from explicit ROCm sources."""
    if find_spec("torch") is None:
        rocm_version = Path("/opt/rocm/.info/version").read_text(encoding="utf-8").strip()
        version = re.match(r"(\d+\.\d+)", rocm_version)
        if version is None:
            raise RuntimeError(f"Cannot determine the ROCm version from {rocm_version!r}")
        index = f"https://download.pytorch.org/whl/rocm{version[1]}"
        print(f"Installing ROCm PyTorch from {index}", flush=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", TORCH_REQUIREMENT, "--index-url", index],
            check=True,
        )
    import torch

    if not torch.version.hip:
        raise RuntimeError(f"{torch.__file__} is not a ROCm PyTorch build; activate the container's ROCm environment")
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm PyTorch cannot access a GPU; expose /dev/kfd and /dev/dri to the container")
    gpus = 8 if suite == "megamoe" else 1
    if torch.cuda.device_count() < gpus:
        raise RuntimeError(f"{suite} requires {gpus} visible GPUs, found {torch.cuda.device_count()}")
    for index in range(gpus):
        arch = torch.cuda.get_device_properties(index).gcnArchName
        if not arch.startswith("gfx950"):
            raise RuntimeError(f"{suite} requires gfx950 GPUs (MI350/MI355), GPU {index} is {arch}")
    if find_spec("aiter") is None:
        install_aiter(workspace)
    import aiter.ops.quant as quant

    for name in ("dynamic_per_group_scaled_quant", "mxfp4_moe_sort_hip"):
        if not hasattr(quant, name):
            raise RuntimeError(f"The container's AITER is missing aiter.ops.quant.{name}")
    print(f"Python: {sys.executable}", flush=True)
    print(f"PyTorch {torch.__version__} from {torch.__file__}; HIP {torch.version.hip}; GPU {arch}", flush=True)
    print(f"AITER quantization from {quant.__file__}", flush=True)


def write_prompts(workspace: Path, suite: str, max_steps: int | None) -> Path:
    """Render an ordered recipe from shared knowledge and the fixed reviewer workload."""
    from ave.prompts import build_prompts

    example = Path(__file__).resolve().parent / suite
    recipe = json.loads((example / "recipe.json").read_text(encoding="utf-8"))
    recipe["steps"] = recipe["steps"][:max_steps]
    driver_source = f"import runpy\nrunpy.run_path({str(example / 'driver.py')!r}, run_name='__main__')\n"
    target = (
        "dynamic_mxfp4_mega_moe from the selected AveLang checkout: Kimi K3 EP8 on eight gfx950 GPUs "
        "(896 experts, 112 local experts, top-16, hidden 3584, intermediate 3072). "
        "Use the existing torchrun driver with all eight ranks and its collective reference. "
        "It validates two consecutive calls per case with distinct per-rank inputs, requires worst-rank SNR >= 30 dB, "
        "and benchmarks kimi_M8/kimi_M32/kimi_M128 using the slowest rank's graph timings. "
        "The repository's test/examples/gemm/amdgpu/test_mega_moe.py additionally covers 32 experts (4 local). "
        "Compute symmetric-heap byte offsets passed to al.subview/make_rsrc in al.u64, as weight_view does; "
        "u32 arithmetic overflows and causes GPU memory-access faults. "
        "Wrap scratch distributed tests in timeout and clean up their own process trees before retrying or finishing"
        if suite == "megamoe"
        else "dynamic_mxfp4_moe from the selected AveLang checkout on one gfx950 GPU. "
        "It runs six correctness cases and benchmarks gptoss/dsv3/dsv4 at M8/M128/M1024"
    )
    workload = (
        f"Hyperloom's custom benchmark runs {target}. "
        "During optimization, inspect and edit the selected worktree using its existing AveLang/MFMA APIs. "
        "Run the supplied correctness and benchmark commands, read failures and repair them in the same session. "
        f"Read the fixed workload driver at {example / 'driver.py'} and entrypoint at {example / 'custom_mi355x.sh'} "
        "alongside the baseline report to establish its call path, shapes and correctness reference. "
        "The fixed driver already implements no-argument correctness and --bench-mode timing. "
        "It imports kernels from the current worktree through PYTHONPATH. After verifying its call path, "
        "use the following complete driver_source to delegate to it; do not duplicate or rewrite the driver. "
        "Do not change workload cases, the fixed benchmark entrypoint, or the reference tests. "
        "Benchmark source edits are permitted only when the step explicitly requires them.\n"
        f"```python\n{driver_source}```"
    )
    steps = build_prompts(recipe, context=workload)
    path = workspace / "prompts.json"
    path.write_text(json.dumps({"steps": [asdict(step) for step in steps]}, indent=2) + "\n", encoding="utf-8")
    return path


def fetch_source(url: str, revision: str, destination: Path) -> None:
    """Fetch one revision, or reuse an existing clean checkout of that revision."""
    if not destination.exists():
        subprocess.run(["git", "init", str(destination)], check=True)
    if not (destination / ".git").is_dir():
        raise ValueError(f"Not a source checkout created by this script: {destination}")
    head = subprocess.run(["git", "rev-parse", "--verify", "HEAD"], cwd=destination, capture_output=True, text=True)
    if head.returncode:
        subprocess.run(["git", "fetch", "--depth=1", url, revision], cwd=destination, check=True)
        subprocess.run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=destination, check=True)
    elif head.stdout.strip() != revision:
        raise ValueError(f"{destination} must be at {revision}; use a new workspace")
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=destination,
        capture_output=True,
        text=True,
        check=True,
    )
    if status.stdout.strip():
        raise ValueError(f"{destination} has tracked changes; use a clean checkout")


def build_llvm(source: Path) -> Path:
    """Build matching Clang, LLVM and MLIR using the historical configuration."""
    build = source / "build"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(source / "llvm"),
            "-B",
            str(build),
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DLLVM_ENABLE_PROJECTS=clang;lld;mlir",
            "-DLLVM_ENABLE_RUNTIMES=compiler-rt",
            "-DLLVM_TARGETS_TO_BUILD=AMDGPU;X86",
            "-DLLVM_ENABLE_DUMP=ON",
            "-DLLVM_ENABLE_RTTI=ON",
            "-DLLVM_ENABLE_ASSERTIONS=ON",
        ],
        check=True,
    )
    subprocess.run(["cmake", "--build", str(build)], check=True)
    return build


def main() -> None:
    """Prepare sources and install their compiler into the active Python environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path, help="Directory for downloaded sources and compiler build")
    parser.add_argument("--suite", choices=AVE_REVISIONS, default="localmoe")
    parser.add_argument("--max-steps", type=int, help="Run at most the first N prompts (default: all)")
    args = parser.parse_args()
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    workspace = args.workspace.expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    try:
        prepare_runtime(workspace, args.suite)
    except RuntimeError as exc:
        parser.error(str(exc))
    experiment = workspace / args.suite
    experiment.mkdir(parents=True, exist_ok=True)
    ave = experiment / "avelang"
    llvm = workspace / "llvm-project"
    fetch_source("https://github.com/xindubawukong/avelang.git", AVE_REVISIONS[args.suite], ave)
    fetch_source("https://github.com/ROCm/llvm-project.git", LLVM_REVISION, llvm)
    prompt = write_prompts(experiment, args.suite, args.max_steps)
    build = build_llvm(llvm)
    env = dict(os.environ)
    env["CMAKE_ARGS"] = " ".join(
        [
            "-DAVE_LANG_BACKEND=rocm",
            f"-DMLIR_DIR={build}/lib/cmake/mlir",
            f"-DLLVM_DIR={build}/lib/cmake/llvm",
            f"-DCMAKE_PREFIX_PATH={build};/opt/rocm",
            f"-DCMAKE_C_COMPILER={build}/bin/clang",
            f"-DCMAKE_CXX_COMPILER={build}/bin/clang++",
            "-DCMAKE_MODULE_LINKER_FLAGS=-Wl,--exclude-libs,ALL",
            "-DCMAKE_SHARED_LINKER_FLAGS=-Wl,--exclude-libs,ALL",
        ]
    )
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", str(ave)], env=env, check=True)
    print(f"Ready for hyperloom optimize: source={ave}, AVE_PROMPTS_FILE={prompt}. See the example README.")


if __name__ == "__main__":
    main()
