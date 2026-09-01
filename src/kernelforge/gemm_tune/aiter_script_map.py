# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Where aiter is installed, and where it keeps each tuner script.

A leaf module on purpose: ``utils`` wants the preferred path per tuner and
``script_discovery`` wants both the hints and the search patterns, while
``script_discovery`` also has to know where csrc lives. Holding any of that in
either of those two made them import each other. Nothing here imports anything
from the package, so there is no cycle to break in the first place.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

# Preference-ordered relative paths under csrc/. First existing file wins.
#
# sglang_dense_bf16 lists the direct tuner ahead of gemm_tuner.py on purpose:
# the latter is a shim that rewrites the tuner's exit code (1 -> 0), which hides
# whether anything was produced. We judge by row count either way, but the
# direct script keeps the signal honest.
TUNER_SCRIPT_HINTS: dict[str, tuple[str, ...]] = {
    "fmoe_ck": ("ck_gemm_moe_2stages_codegen/gemm_moe_tune.py",),
    "a8w8": ("ck_gemm_a8w8/gemm_a8w8_tune.py",),
    "a8w8_blockscale": ("ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py",),
    "a8w8_bpreshuffle": ("ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.py",),
    "a8w8_blockscale_bpreshuffle": ("ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py",),
    "a4w4_blockscale": ("ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale_tune.py",),
    "sglang_dense_bf16": (
        "gemm_a16w16/gemm_a16w16_tune.py",
        "gemm_a16w16/gemm_tuner.py",
    ),
}

# Filename globs used when every hint misses. Exact filenames, so
# ``gemm_a8w8_tune.py`` never matches ``batched_gemm_a8w8_tune.py``.
TUNER_SCRIPT_PATTERNS: dict[str, tuple[str, ...]] = {
    "fmoe_ck": ("**/gemm_moe_tune.py",),
    "a8w8": ("**/gemm_a8w8_tune.py",),
    "a8w8_blockscale": ("**/gemm_a8w8_blockscale_tune.py",),
    "a8w8_bpreshuffle": ("**/gemm_a8w8_bpreshuffle_tune.py",),
    "a8w8_blockscale_bpreshuffle": ("**/gemm_a8w8_blockscale_tune.py",),
    "a4w4_blockscale": ("**/gemm_a4w4_blockscale_tune.py",),
    "sglang_dense_bf16": ("**/gemm_a16w16_tune.py", "**/gemm_tuner.py"),
}


def resolve_aiter_root() -> Path | None:
    """Find aiter installation root (AITER_ROOT_DIR or package location)."""
    root_env = os.environ.get("AITER_ROOT_DIR", "").strip()
    if root_env and Path(root_env).is_dir():
        return Path(root_env)
    with contextlib.suppress(ImportError):
        import aiter

        pkg_dir = Path(aiter.__file__).parent
        # Source installs keep csrc beside the aiter package. Some wheel
        # layouts split metadata and tuner scripts into a sibling aiter_meta
        # package under the same site-packages directory.
        for candidate in (pkg_dir.parent, pkg_dir.parent / "aiter_meta"):
            if (candidate / "csrc").is_dir():
                return candidate
    # Fallback well-known paths
    for p in ("/sgl-workspace/aiter", "/opt/aiter"):
        if Path(p).is_dir():
            return Path(p)
    return None


def resolve_aiter_csrc() -> Path | None:
    """Return the aiter csrc directory containing tuner scripts."""
    root = resolve_aiter_root()
    if root is None:
        return None
    csrc = root / "csrc"
    return csrc if csrc.is_dir() else None
