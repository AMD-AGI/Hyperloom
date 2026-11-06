# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A FlyDSL rewrite spec over a minimal workspace, shared by the rewrite and KB recipe tests."""

from __future__ import annotations

from kernelforge.workflows.flydsl_rewrite.spec import RewriteSpec


def _spec(tmp_path):
    workspace = tmp_path / "workspace"
    source = workspace / "vllm" / "softmax.py"
    source.parent.mkdir(parents=True)
    source.write_text("import triton\n@triton.jit\ndef softmax_kernel(x):\n    return x\n")
    kernel = workspace / "kernel.py"
    kernel.write_text("import flydsl\ndef build_softmax_module(config):\n    return lambda inputs: inputs['x']\n")
    driver = workspace / "driver.py"
    driver.write_text("# stable rewrite driver contract\n")
    return (
        RewriteSpec(
            op_name="softmax",
            source_kernel=str(source),
            target_functions=["softmax_kernel"],
            framework="vllm",
            flydsl_kernel=str(kernel),
            workspace=str(workspace),
            snr_threshold=30.0,
        ),
        driver,
    )
