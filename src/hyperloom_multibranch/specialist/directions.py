# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The directions a session hands its specialist branches, so its first branches do not all sweep the same settings.

A branch started with no direction of its own and no repair takes the one the fewest running branches hold, the
earlier one in this order on a tie: the first four are the ones a four-branch session works on.
"""

from __future__ import annotations

DIRECTIONS = (
    "server.log fixes: read the server.log of your branch's tests, fix every error and warning it shows, and make sure "
    "every setting the recipe asks for is really applied, with no fallback to an eager or plain torch path; a fix is a "
    "source patch or the setting that removes the cause.",
    "GEMM tuning: from the GEMM shapes the profile shows for this workload (decode and prefill), tune the aiter and "
    "hipBLASLt GEMM configurations for those shapes, and test each tuned table as the kernel artifact put_kernel_artifact "
    "returns.",
    "Fusion: fuse the elementwise work around the GEMMs and the attention (rmsnorm with the residual add, silu with "
    "mul, rope) with a source patch or a kernel, starting from the operators the profile ranks highest.",
    "Server arguments and environment variables: yours is the one branch that tests settings alone; test the settings "
    "the profile or server.log point at, not a blind sweep.",
    "ROCm code gaps: find the code paths where ROCm runs a slower or missing implementation where CUDA has a fast one "
    "(a fallback, a missing fused or tuned path), and port or align the CUDA implementation.",
    "Attention and decode kernels: speed up the hottest attention and decode kernels the profile shows, with a kernel "
    "artifact or a source patch.",
    "Existing PRs and patches: find upstream sglang and aiter PRs or patches that this image lacks and that target this "
    "workload's hot paths, port them as a patch and test them.",
)

__all__ = ["DIRECTIONS"]
