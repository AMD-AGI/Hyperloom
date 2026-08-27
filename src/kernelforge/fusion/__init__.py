# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Kernel fusion - autonomous source-level fusion discovery + validation.

Given a decode trace, a model and a framework (sglang/vllm), the pipeline:

1. diagnoses whether the decode path is launch-bound (a fusion candidate),
2. locates which launch-bound op chain to fuse (model-agnostic pattern library +
   source localization) and produces a concrete recipe,
3. authors an env-gated fused kernel,
4. validates it at the KERNEL level (numerical parity vs the real eager op chain
   + an isolated microbenchmark speedup) -- e2e is intentionally out of scope and
   left to Hyperloom / the caller,
5. emits a fixed JSON manifest + a git patch describing the kernel and framework
   changes.

Reached from the CLI as ``kernelforge forge-fuse``.
"""

__version__ = "0.1.0"
