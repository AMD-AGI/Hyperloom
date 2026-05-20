"""Mock Framework agent backend (deterministic envelope source).

Peer of :class:`MockKernelBackend`. P1 mock e2e DOES NOT route through
this backend — handlers return canned envelopes directly (see
:mod:`framework_request_handlers`). This class exists for two reasons:

1. Unit-testing the ``Backend`` interface contract for the framework
   role without needing the real subprocess bridge.
2. PR-F can swap ``FrameworkAgentBackend`` (real subprocess) for this
   class behind ``--framework-mock`` so CI smoke runs without the
   ``fa agent`` runtime + libcst dependency installed.

Selected by passing ``--framework-mock`` on the CLI. Defaults match the
:func:`framework_optimize_handler` canned reply so e2e routing is
self-consistent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import Backend, BackendTurnResult


class FrameworkMockBackend(Backend):
    """Deterministic mock backend for the framework role.

    Does not call any external service. Used by ``--framework-mock`` CI
    runs and by PR-F's backend-contract unit tests.
    """

    name = "framework_mock"

    def __init__(self, *, session_dir: Path | None = None) -> None:
        self._session_dir = Path(session_dir) if session_dir is not None else None

    async def run_turn(self, *args: Any, **kwargs: Any) -> BackendTurnResult:
        """Heartbeat-only turn loop.

        Framework role is responder-only and its handlers are
        programmatic, so the backend never actually emits intents on a
        normal P1 run. We return an empty BackendTurnResult so callers
        that probe the backend liveness (e.g. ``_preflight``) get a
        consistent shape.
        """
        return BackendTurnResult(
            intents=[],
            raw_assistant_text="",
            stop_reason="end_turn",
            usage={},
        )

    def run_optimize(
        self,
        *,
        session_dir: str,
        target_framework: str,
        kb_partition: str,
        ast_scan_enabled: bool = True,
        ast_frameworks: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Direct envelope source used by PR-F's contract tests.

        Returns the OptimizeSuccess payload only — wrapping into
        ``{kind, status, result}`` is the handler's job.
        """
        return {
            "payload_kind": "OptimizeSuccess",
            "patch_path": f"{session_dir}/runs/framework/fw-mock/proposal.diff",
            "predicted_gain_pct": 5.0,
            "rationale": "FrameworkMockBackend canned response",
            "discovered_flags": {},
            "target_framework": target_framework,
            "stage_a_elapsed_ms": 50,
        }


__all__ = ["FrameworkMockBackend"]
