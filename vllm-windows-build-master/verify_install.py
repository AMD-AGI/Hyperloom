"""Validate the portable vLLM runtime contract used by install.bat."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import sys
from pathlib import Path

EXPECTED_VLLM_VERSION = "0.33.0"

REQUIRED_MODULES = (
    "llguidance",
    "multi_turboquant",
    "xgrammar",
    "vllm._C_stable_libtorch",
    "vllm._moe_C_stable_libtorch",
    "vllm.cumem_allocator",
    "vllm.fs_io_C",
    "vllm.spinloop",
    "vllm.model_executor.models.qwen3_5",
    "vllm.model_executor.models.qwen3_vl",
    "vllm.v1.kv_offload.cpu.gpu_worker",
    "vllm.v1.kv_offload.cpu.shared_offload_region",
    "vllm.v1.kv_offload.file_mapper",
    "vllm.v1.kv_offload.tiering.fs.io",
    "vllm.vllm_flash_attn.layers.rotary",
    "vllm.vllm_flash_attn.ops.triton.rotary",
    "vllm.vllm_flash_attn._vllm_fa2_C",
)


def validate_vllm_versions(runtime_version: str, distribution_version: str) -> None:
    """Validate both the wheel build tag and vLLM's upstream runtime version."""

    if distribution_version != EXPECTED_VLLM_VERSION:
        raise RuntimeError(
            "vLLM distribution version is "
            f"{distribution_version!r}, expected {EXPECTED_VLLM_VERSION!r}"
        )

    expected_runtime_version = EXPECTED_VLLM_VERSION.partition("+")[0]
    runtime_base_version = runtime_version.partition("+")[0]
    if runtime_base_version != expected_runtime_version:
        raise RuntimeError(
            "vLLM runtime version is "
            f"{runtime_version!r}, expected base version {expected_runtime_version!r}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    site_packages = (root / "python" / "Lib" / "site-packages").resolve()

    print("Checking PyTorch, Triton, and vLLM imports...", flush=True)
    import torch
    import triton
    import vllm
    import multi_turboquant

    installed_vllm_version = importlib.metadata.version("vllm")
    validate_vllm_versions(vllm.__version__, installed_vllm_version)
    if not torch.__version__.startswith("2.14.0") or torch.version.cuda != "13.0":
        raise RuntimeError(
            f"PyTorch is {torch.__version__!r} with CUDA {torch.version.cuda!r}, "
            "expected 2.14.0+cu130"
        )
    if not triton.__version__.startswith("3.8.0"):
        raise RuntimeError(f"Triton version is {triton.__version__!r}, expected 3.8.0")
    if multi_turboquant.__version__ != "0.1.0":
        raise RuntimeError(
            f"Multi-TurboQuant version is {multi_turboquant.__version__!r}, expected '0.1.0'"
        )

    vllm_file = Path(vllm.__file__).resolve()
    if site_packages not in vllm_file.parents:
        raise RuntimeError(
            f"vLLM loaded from {vllm_file}, expected it under {site_packages}"
        )

    for module_name in REQUIRED_MODULES:
        print(f"Checking {module_name}...", flush=True)
        importlib.import_module(module_name)

    from vllm.v1.attention.ops.multi_turboquant_kv import get_packed_dim

    for cache_dtype in (
        "isoquant3",
        "isoquant4",
        "planarquant3",
        "planarquant4",
        "turboquant25",
        "turboquant35",
    ):
        packed_dim = get_packed_dim(cache_dtype, 128)
        if not 0 < packed_dim <= 128:
            raise RuntimeError(f"invalid {cache_dtype} packed dimension: {packed_dim}")

    if args.cuda:
        print("Checking Triton and FlashAttention CUDA execution...", flush=True)
        import triton.runtime.driver as driver
        from vllm.vllm_flash_attn.layers.rotary import apply_rotary_emb

        active = getattr(driver, "active", None)
        active = active if active is not None else driver.driver.active
        target = active.get_current_target()
        if target.backend != "cuda":
            raise RuntimeError(f"Triton backend is {target.backend!r}, expected 'cuda'")
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch cannot access CUDA")

        x = torch.randn((1, 4, 2, 8), device="cuda", dtype=torch.float16)
        cos = torch.randn((4, 4), device="cuda", dtype=torch.float16)
        sin = torch.randn((4, 4), device="cuda", dtype=torch.float16)
        output = apply_rotary_emb(x, cos, sin)
        torch.cuda.synchronize()
        if output.shape != x.shape or output.device.type != "cuda":
            raise RuntimeError("FlashAttention rotary CUDA smoke test returned invalid output")
        print(f"Triton CUDA target: {target.backend} {target.arch}")

    print(
        f"vLLM runtime {vllm.__version__} / distribution "
        f"{installed_vllm_version} contract passed from {vllm_file}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
