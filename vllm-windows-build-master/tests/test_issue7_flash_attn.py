"""Regression test for issue #7's missing FlashAttention rotary package."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package-root",
        type=Path,
        help="Optional pip --target directory containing the wheel under test.",
    )
    args = parser.parse_args()
    if args.package_root:
        sys.path.insert(0, str(args.package_root.resolve()))

    print("Importing torch...", flush=True)
    import torch

    print("Importing vLLM...", flush=True)
    import vllm

    print("Importing Qwen3-VL and FlashAttention rotary...", flush=True)
    import vllm.model_executor.models.qwen3_vl  # noqa: F401
    from vllm.vllm_flash_attn.layers.rotary import apply_rotary_emb

    if args.package_root:
        package_root = args.package_root.resolve()
        vllm_file = Path(vllm.__file__).resolve()
        assert package_root in vllm_file.parents, (
            f"loaded vLLM from {vllm_file}, expected it under {package_root}"
        )

    assert torch.cuda.is_available(), "CUDA is required for the rotary kernel test"
    x = torch.randn((1, 4, 2, 8), device="cuda", dtype=torch.float16)
    cos = torch.randn((4, 4), device="cuda", dtype=torch.float16)
    sin = torch.randn((4, 4), device="cuda", dtype=torch.float16)
    output = apply_rotary_emb(x, cos, sin)
    torch.cuda.synchronize()
    assert output.shape == x.shape
    assert output.device.type == "cuda"

    print(f"vLLM: {vllm.__version__} from {vllm.__file__}")
    print(f"CUDA rotary regression passed: shape={tuple(output.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
