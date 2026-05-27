"""Blocker #1 regression — ``--no-kernel`` boot must still register the
PRELUDE-initial analysis executors.

Pre-PR-321 the ``roofline`` registration sat *below* the ``no_kernel``
early-return in :func:`cli._register_executors`. The single-path refactor
keeps the Coordinator auto-enqueueing an analysis task at PRELUDE
regardless of mode, so a ``--no-kernel`` session would enqueue the task
and then fail with ``no_executor`` on dispatch.

This test pins the ordering so the regression cannot recur: both kinds
must be present in the SubAgentRunner's ``executor_registry`` even when
``no_kernel=True``.
"""

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
        # RooflineExecutor refuses ``shared_state is None`` so any
        # truthy stand-in is enough — registration does not call into
        # it, this object is just stored on the executor.
        self.shared_state = object()


@pytest.mark.parametrize("no_kernel", [True, False])
def test_roofline_and_profile_registered_in_both_modes(
    tmp_path: Path, no_kernel: bool,
) -> None:
    """Single-path lifecycle: both ``roofline`` and ``profile`` are
    Coordinator-internal analysis executors and must dispatch in every
    boot configuration.

    ``profile`` is the ``--no-enable-roofline`` alternative and travels
    in ``_REAL_EXECUTORS_FULL`` (always registered). ``roofline`` is
    minted via :func:`make_roofline_executor` and was the regression
    site — its registration moved above the ``no_kernel`` early-return.
    """
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
