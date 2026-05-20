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

import asyncio
import logging
from pathlib import Path
from typing import Any, Awaitable, Callable


log = logging.getLogger(__name__)


HandlerFn = Callable[..., Awaitable[dict[str, Any]]]


# Module-level singleton -- the Coordinator sets ``framework_backend``
# in PR-B after constructing FrameworkAgentBackend / FrameworkMockBackend.
# The handler reads it lazily so unit tests can monkey-patch a fixture
# in without touching the Coordinator wiring. None means "P1 mock path"
# -- handler returns canned envelopes directly.
_FRAMEWORK_BACKEND: Any = None


def set_framework_backend(backend: Any) -> None:
    """Inject the framework backend used by the real-path handlers.

    Called by the Coordinator after ``_build_backends`` selects which
    concrete backend to use (FrameworkAgentBackend /
    FrameworkMockBackend / CodexBackend). ``None`` disables the real
    path so the handler stays on the P1 mock branch -- this is what
    the legacy 4-role tests rely on.
    """
    global _FRAMEWORK_BACKEND
    _FRAMEWORK_BACKEND = backend


async def framework_optimize_handler(
    payload: dict[str, Any],
    *,
    session_dir: Path,
) -> dict[str, Any]:
    """Drive Stage A: AST scan + envelope build.

    P1 (no backend injected): returns the canned OptimizeSuccess
    envelope so the protocol mesh keeps end-to-end testable.
    P2 PR-F (backend injected): invokes
    ``FrameworkAgentBackend.run_optimize`` which performs real AST
    scan + envelope validation via ``fa agent commit-result``.
    P3 PR-G upgrades the patch_path to a real proposal.diff.
    """
    target_fw = (str(payload.get("target_framework") or "sglang")
                 .strip().lower())
    backend = _FRAMEWORK_BACKEND
    if backend is None or not hasattr(backend, "run_optimize"):
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
            "rationale": "P1 mock -- handler returns canned OptimizeSuccess",
            "discovered_flags": {},
            "target_framework": target_fw,
            "stage_a_elapsed_ms": 100,
        }

    log.info(
        "framework_optimize_handler[P2 real] session=%s target=%s "
        "ast_scan_enabled=%s ast_frameworks=%r",
        Path(session_dir).name, target_fw,
        payload.get("ast_scan_enabled", True),
        payload.get("ast_frameworks") or (target_fw,),
    )
    # FrameworkAgentBackend.run_optimize is a sync subprocess call --
    # offload to a thread so the Coordinator reactor's event loop
    # doesn't block on `fa agent prepare-task`.
    try:
        envelope: dict[str, Any] = await asyncio.to_thread(
            backend.run_optimize,
            session_dir=str(session_dir),
            target_framework=target_fw,
            kb_partition=str(payload.get("kb_partition")
                             or "framework_optimization"),
            ast_scan_enabled=bool(payload.get("ast_scan_enabled", True)),
            ast_frameworks=tuple(payload.get("ast_frameworks") or ()),
        )
    except NotImplementedError as exc:
        log.warning("framework backend not yet implemented: %s", exc)
        return {
            "status": "failed",
            "payload_kind": "OptimizeFailure",
            "reason": "backend_not_implemented",
            "detail": str(exc),
            "stage_a_elapsed_ms": 0,
        }
    except Exception as exc:  # noqa: BLE001
        log.exception("framework_optimize_handler[P2 real] crashed")
        return {
            "status": "failed",
            "payload_kind": "OptimizeFailure",
            "reason": "handler_exception",
            "detail": repr(exc),
            "stage_a_elapsed_ms": 0,
        }
    # Coordinator wrapping (kind/status/result) is added in the
    # framework REQUEST routing branch; the handler just returns the
    # envelope plus a top-level ``status`` so the reactor can pick a
    # response priority. PR-F propagates the OptimizeFailure status as
    # "failed", everything else as "succeeded".
    kind = envelope.get("payload_kind")
    envelope.setdefault(
        "status",
        "failed" if kind == "OptimizeFailure" else "succeeded",
    )
    return envelope


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
