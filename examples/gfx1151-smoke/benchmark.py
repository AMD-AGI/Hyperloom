# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Small GPU attention smoke test with a CPU reference, not an LLM benchmark."""

from __future__ import annotations

import json
import math
import os
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def main() -> None:
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("A ROCm PyTorch build and accessible GPU are required")
    properties = torch.cuda.get_device_properties(0)
    if properties.gcnArchName.split(":", 1)[0] != "gfx1151":
        raise RuntimeError(f"Expected gfx1151, got {properties.gcnArchName}")
    torch.manual_seed(42)
    q, k, v = [torch.randn(1, 8, 128, 64) for _ in range(3)]
    reference = (q @ k.transpose(-2, -1) / math.sqrt(64)).softmax(-1) @ v
    q, k, v = [x.to(device="cuda", dtype=torch.float16) for x in (q, k, v)]
    with torch.inference_mode():
        for _ in range(5):
            actual = F.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
        samples = []
        for _ in range(5):
            start = time.perf_counter()
            for _ in range(20):
                actual = F.scaled_dot_product_attention(q, k, v)
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) / 20)
    error = (actual.float().cpu() - reference).abs()
    passed = bool(
        torch.isfinite(error).all() and torch.allclose(actual.float().cpu(), reference, atol=0.002, rtol=0.02)
    )
    report = {
        "framework": "custom",
        "workload_kind": "scriptable",
        "throughput_unit": "attention_calls/s",
        "output_throughput": 1 / statistics.median(samples),
        "quality_gate": {"passed": passed, "max_abs_error": error.max().item()},
        "gpu_arch": properties.gcnArchName,
        "hip_multiprocessors": properties.multi_processor_count,
        "torch_version": torch.__version__,
    }
    output = Path(os.environ["RESULT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "inferencex_result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise RuntimeError("GPU attention failed the CPU-reference correctness check")


if __name__ == "__main__":
    main()
