# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The directions a session hands its specialist branches, so its first branches do not all sweep the same settings.

A branch started with no direction of its own and no repair takes the first one no branch of the session has held
yet, so the first four are the ones a four-branch session works on; once each was held, the one the fewest branches
held, then the one the fewest running branches hold, the earlier one in this order on a tie.
"""

from __future__ import annotations

DIRECTIONS = (
    "Fast path checks: make sure every setting the recipe asks for and every patch really takes its fast path. Read "
    "the server.log of your branch's tests for what fell back: an eager or plain torch path, CUDA graphs off, a kernel "
    "or a feature the engine turned off. Where the log cannot tell, add a one-time log line on the path to prove it "
    "ran. For each operator the profile ranks highest, read the branch conditions on its call path (is_cuda or is_hip, "
    "an SM, gfx or capability check, a table keyed by TP size or cu_num): a tuned constant or a fused path that sits "
    "only in another branch never runs here, and the log does not say so. Fix the cause with a source patch or the "
    "setting that removes it; errors and warnings in the log are leads, and untuned GEMM shapes belong to the GEMM "
    "tuning branch.",
    "GEMM tuning: from the GEMM shapes the profile shows for this workload (decode and prefill), tune the aiter and "
    "hipBLASLt GEMM configurations for those shapes, and test each tuned table as the kernel artifact put_kernel_artifact "
    "returns. Fill the shapes the server.log reports as untuned first; for a shape with no tuned row, switch its "
    "kernel path before picking rows by hand.",
    "Fusion: fuse the elementwise work around the GEMMs and the attention (rmsnorm with the residual add, silu with "
    "mul, rope) with a source patch or a kernel, starting from the operators the profile ranks highest.",
    "Server arguments and environment variables: yours is the one branch that tests settings alone; test the settings "
    "the profile or server.log point at, not a blind sweep.",
    "ROCm code gaps: find the code paths where ROCm runs a slower or missing implementation where CUDA has a fast one "
    "(a fallback, a missing fused or tuned path), and port or align the CUDA implementation.",
    "Attention and decode kernels: speed up the hottest attention and decode kernels the profile shows, with a kernel "
    "artifact or a source patch.",
    "Existing PRs and patches: scan the GitHub PRs of aiter and of this session's framework (sglang, vllm or atom), "
    "merged or open, for ones on this model's architecture, its hot operators or ROCm and gfx950; check each against "
    "the version installed in the image, which can predate a merged PR, and port the ones it lacks as a patch to test. "
    "Search with WebSearch, once per hot operator at least (the architecture, ROCm or gfx950 and the operator, on "
    "site:github.com/sgl-project/sglang or site:github.com/ROCm/aiter), read a PR's diff with WebFetch, tell from the "
    "installed version's commit whether the image has it, and record each PR's verdict with notebook_learn (in the "
    "image, missing, or not for this model) so no branch searches it again.",
)


def direction_name(direction: str) -> str:
    """The short name a direction opens with, as prompts and notices cite it."""
    return direction.split(":", 1)[0]


__all__ = ["DIRECTIONS", "direction_name"]
