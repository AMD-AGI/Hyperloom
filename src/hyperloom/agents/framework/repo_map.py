# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Upstream repos a framework-agent round scouts: the framework's own, and the bridging ones."""

from __future__ import annotations

from hyperloom.inference_optimizer import framework_registry

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


def repo_url_for_framework(framework: str) -> str:
    """Return the canonical GitHub repo URL for ``framework``."""
    return framework_registry.repo_url(framework)


__all__ = ["bridge_repo_urls", "repo_url_for_framework"]
