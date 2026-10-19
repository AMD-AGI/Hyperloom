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


#: Upstreams of the forks above. New-model support lands upstream first, so a
#: fork's own PR list rarely carries the change a Day-1 model needs.
_FORK_TO_UPSTREAM_REPO_URLS: dict[str, tuple[str, ...]] = {
    "https://github.com/ROCm/vllm.git": ("https://github.com/vllm-project/vllm.git",),
}


def upstream_repo_urls(repo_url: str) -> tuple[str, ...]:
    """Return the upstream repo URLs of a fork, or ``()`` when ``repo_url`` is not a known fork."""
    return _FORK_TO_UPSTREAM_REPO_URLS.get(repo_url.strip(), ())


def bridge_repo_urls(bridge_layer: str) -> tuple[str, ...]:
    """Return the bridging repo URLs to scout for a failure's ``bridge_layer``."""
    return _BRIDGE_LAYER_TO_REPO_URLS.get((bridge_layer or "").strip().lower(), ())


__all__ = ["bridge_repo_urls", "upstream_repo_urls"]
