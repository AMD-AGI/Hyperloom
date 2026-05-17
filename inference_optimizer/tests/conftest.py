"""Pytest hooks and session-wide defaults for inference_optimizer tests."""

from __future__ import annotations

import os
from pathlib import Path


def _bootstrap_kernel_agent_env() -> None:
    """Point HYPERLOOM_KERNEL_AGENT_ROOT at the in-repo kernel-agent checkout.

    ``kernel_request_handlers`` snapshots this path at import time from
    ``os.environ``. Tests that need ``apply_kernel_patch.py`` /
    ``kernel_optimization.py`` must see a valid root before any handler
    module import, so we set the env here (conftest loads before test modules).
    """
    if os.environ.get("HYPERLOOM_KERNEL_AGENT_ROOT"):
        return
    # conftest.py -> tests -> inference_optimizer -> Hyperloom
    repo = Path(__file__).resolve().parents[2]
    kernel_agent = repo / "kernel-agent"
    if kernel_agent.is_dir():
        os.environ["HYPERLOOM_KERNEL_AGENT_ROOT"] = str(kernel_agent)


_bootstrap_kernel_agent_env()
