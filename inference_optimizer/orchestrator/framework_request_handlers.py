"""Coordinator-side handlers for Framework-agent REQUEST kinds.

Counterpart of :mod:`kernel_request_handlers` for the 5th Framework
agent role (hyperloom-framework-agent-design.md §5 / §6).

Lifecycle:

* **P1 mock** (this commit, PR-B) — handlers return canned envelopes
  directly. No backend invocation, no subprocess. Lets the
  ``--framework-mock`` smoke test verify the 5-role protocol mesh
  without API keys / GPU / vllm-or-sglang source mount.
* **P2 PR-F** — :func:`framework_optimize_handler` invokes
  :class:`FrameworkAgentBackend` (subprocess JSON-over-stdio bridge to
  the sibling ``framework-agent/agent/cli.py``) and emits
  ``UPDATE_STATE(discovered_flags)`` once the AST scan returns.
* **P3 PR-G** — same handler returns a real ``patch_path`` produced by
  the LLM patch proposer + KB priors.
* **P3 PR-H** — :func:`framework_integrate_handler` applies the patch,
  restarts the server, runs Magpie + accuracy gate, and verdicts
  KEEP / REVERT / NEEDS_REVIEW.

Handler signature (matches kernel_request_handlers for symmetry)::

    async def handler(payload: dict, *, session_dir: Path) -> dict:
        # returns the dict that becomes RESPONSE.payload['result']

Dispatch table is exposed via :data:`FRAMEWORK_REQUEST_HANDLERS` so
callers / tests can monkey-patch.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Awaitable, Callable


log = logging.getLogger(__name__)


HandlerFn = Callable[..., Awaitable[dict[str, Any]]]


async def framework_optimize_handler(
    payload: dict[str, Any],
    *,
    session_dir: Path,
) -> dict[str, Any]:
    """P1 mock OptimizeSuccess envelope (predicted_gain_pct=5% > 3%
    threshold so orchestration re-proposes ``framework_integrate``
    on the next tick — keeps the P1 e2e KEEP path testable).

    PR-F upgrades this to invoke ``FrameworkAgentBackend.run_optimize``
    via subprocess and emit ``UPDATE_STATE(discovered_flags)``.
    PR-G upgrades the patch_path to point at a real unified diff.
    """
    target_fw = (str(payload.get("target_framework") or "sglang")
                 .strip().lower())
    log.info(
        "framework_optimize_handler[P1 mock] session=%s target=%s",
        Path(session_dir).name, target_fw,
    )
    return {
        "status": "succeeded",
        "payload_kind": "OptimizeSuccess",
        "patch_path": str(
            Path(session_dir) / "runs" / "framework"
            / "fw-mock" / "proposal.diff"
        ),
        "predicted_gain_pct": 5.0,
        "rationale": "P1 mock — handler returns canned OptimizeSuccess",
        "discovered_flags": {},
        "target_framework": target_fw,
        "stage_a_elapsed_ms": 100,
    }


async def framework_integrate_handler(
    payload: dict[str, Any],
    *,
    session_dir: Path,
) -> dict[str, Any]:
    """P1 mock IntegrateSuccess(KEEP) envelope.

    PR-H upgrades to apply patch + restart server + Magpie + accuracy
    gate + verdict decision.
    """
    patch_id = str(payload.get("patch_id") or "fw-mock")
    log.info(
        "framework_integrate_handler[P1 mock] session=%s patch_id=%s",
        Path(session_dir).name, patch_id,
    )
    return {
        "status": "succeeded",
        "payload_kind": "IntegrateSuccess",
        "verdict": "KEEP",
        "patch_id": patch_id,
        "tput_before": 0.0,
        "tput_after": 0.0,
        "accuracy_before": 0.0,
        "accuracy_after": 0.0,
        "accuracy_drop": 0.0,
        "stage_b_elapsed_ms": 200,
    }


FRAMEWORK_REQUEST_HANDLERS: dict[str, HandlerFn] = {
    "framework_optimize": framework_optimize_handler,
    "framework_integrate": framework_integrate_handler,
}


def get_framework_handler(kind: str) -> HandlerFn | None:
    """Return the registered handler for ``kind``, or None when unknown."""
    return FRAMEWORK_REQUEST_HANDLERS.get(kind)


__all__ = [
    "FRAMEWORK_REQUEST_HANDLERS",
    "framework_integrate_handler",
    "framework_optimize_handler",
    "get_framework_handler",
]
