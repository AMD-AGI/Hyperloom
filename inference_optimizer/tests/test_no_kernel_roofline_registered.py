# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Blocker #1 regression — ``--no-kernel`` boot must still register the analysis executors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


class _StubSub:
    def __init__(self) -> None:
        self.executor_registry: dict[str, Any] = {}

    def register_executor(self, kind: str, fn: Any) -> None:
        self.executor_registry[kind] = fn


class _StubCoord:
    def __init__(self) -> None:
        self.sub = _StubSub()
        # RooflineExecutor refuses ``shared_state is None``; any truthy stand-in is enough.
        self.shared_state = object()


@pytest.mark.parametrize("no_kernel", [True, False])
def test_roofline_and_profile_registered_in_both_modes(
    tmp_path: Path, no_kernel: bool,
) -> None:
    """Both ``roofline`` and ``profile`` must register in every boot configuration."""
    from inference_optimizer.cli import _register_executors

    coord = _StubCoord()
    _register_executors(
        coord, no_kernel=no_kernel, session_dir=tmp_path,
        specialist_executor=None,
    )
    registry = coord.sub.executor_registry
    assert "roofline" in registry, (
        "roofline executor missing (no_kernel=%s): PRELUDE auto-enqueue "
        "would fail with no_executor" % no_kernel
    )
    assert "profile" in registry, (
        "profile executor missing (no_kernel=%s): --no-enable-roofline "
        "PRELUDE auto-enqueue would fail with no_executor" % no_kernel
    )
