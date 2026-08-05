# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``hyperloom.forge_kernels`` — runtime dispatcher for KernelForge kernel packs.

This is the ONLY Hyperloom package that executes *inside the served framework
process* (sglang / vLLM). Everything here therefore imports the stdlib plus
``torch`` and (lazily) ``flydsl`` only: never ``orchestrator`` /
``inference_optimizer`` / ``agents``, and never anything that would drag the
optimizer's dependency tree into a serving worker.

KernelForge emits standalone kernel modules under
``$FORGE_PATH/serving_patches/kernels/<pack>/kernel.py``. They are builder modules,
not patches: ``build_softmax_module(M, N, dtype) -> launch(A, C, m_rows,
stream=)``. Hyperloom's orchestrator installs a *pack* (the kernel module, its
manifest, and a preflight report pinning the shapes that were actually verified
on this machine) and patches the framework call site to route through the
``op_*`` entry points below.

Contract for the framework-side patch: every entry point returns ``None``
whenever the pack is disabled, missing, unverified for this shape, or unsafe to
build right now. The patched call site MUST treat ``None`` as "run the original
code". That keeps every patch backward-compatible and makes the default (no
``HYPERLOOM_FORGE_KERNEL_PACKS`` in the environment) a strict no-op.
"""

from __future__ import annotations

from ._dispatch import enabled_packs
from ._dispatch import is_enabled
from ._dispatch import rowwise_softmax
from ._dispatch import stats

__all__ = [
    "enabled_packs",
    "is_enabled",
    "rowwise_softmax",
    "stats",
]
