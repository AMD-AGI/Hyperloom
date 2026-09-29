# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Bridging repos a framework-agent round scouts; a framework's own repo is its ``FrameworkSpec.repo_url``."""

from __future__ import annotations

# Enablement bridging repos, keyed by ``bridge_layer``.
_BRIDGE_LAYER_TO_REPO_URLS: dict[str, tuple[str, ...]] = {
    "rocm_hip": (
        "https://github.com/ROCm/aiter.git",
        "https://github.com/ROCm/HIP.git",
        "https://github.com/ROCm/ROCm.git",
    ),
    "build": ("https://github.com/ROCm/aiter.git",),
}


def bridge_repo_urls(bridge_layer: str) -> tuple[str, ...]:
    """Return the bridging repo URLs to scout for a failure's ``bridge_layer``."""
    return _BRIDGE_LAYER_TO_REPO_URLS.get((bridge_layer or "").strip().lower(), ())


__all__ = ["bridge_repo_urls"]
