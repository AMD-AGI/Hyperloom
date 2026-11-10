# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""fix(kernelforge): use RDNA guidance and rocprofv3 first on gfx1201."""

import hashlib

from kernelforge.kernel_backends.triton.prompts import build_system_prompt
from kernelforge.gpu_guidance import profiler_for_target


def test_triton_authoring_context_names_the_real_rdna_mxfp4_path():
    prompt = build_system_prompt("gfx1201", "target cards", "canonical gate")
    assert "Use wave32 and WMMA" in prompt
    assert "BF16 activation × MXFP4 weight" in prompt
    assert "matmul_ogs" in prompt
    assert "rocprofv3 kernel tracing" in prompt
    assert "wavefront=64" not in prompt and "CDNA-sane" not in prompt


def test_profiler_choice_keeps_cdna_default():
    assert profiler_for_target("gfx1201") == "rocprofv3"
    assert profiler_for_target("gfx942") == "rocprof-compute"
    assert profiler_for_target("gfx950") == "rocprof-compute"


def test_cdna_triton_prompts_match_main():
    expected = {
        "gfx942": "b7babf0bc36af5ae2a03d19c2284ce3eaa04a273021a091374dc5a96e4691a0c",
        "gfx950": "1011cf3121bf14ac3a1dad1dc5b5480542989506af11bc5190040971f3bd8f90",
    }
    for gpu_target, digest in expected.items():
        prompt = build_system_prompt(gpu_target, "target cards", "canonical gate")
        assert hashlib.sha256(prompt.encode()).hexdigest() == digest
