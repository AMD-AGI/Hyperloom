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


# Hook-injection points for the P3 PR-H real path. Production wires
# these to IO's server lifecycle + Magpie + accuracy gate; tests inject
# fixtures so the 20-cell fault matrix is exercisable without GPU.
#
# Signatures (all sync; handler offloads via asyncio.to_thread):
#   server_restart_hook(session_dir, patch_id) -> None | raises ServerRestartError
#   bench_hook(session_dir, patch_id) -> {"tput": float, "ok": bool,
#                                          "reason": str (when not ok)}
#   accuracy_gate_hook(session_dir, patch_id) -> float | None
_SERVER_RESTART_HOOK: Any = None
_BENCH_HOOK: Any = None
_ACCURACY_GATE_HOOK: Any = None


class ServerRestartError(RuntimeError):
    """Raised by the server-restart hook to signal a hard failure."""


def set_integrate_hooks(
    *,
    server_restart: Any | None = None,
    bench: Any | None = None,
    accuracy_gate: Any | None = None,
) -> None:
    """Inject the 3 hooks used by the real integrate flow.

    ``None`` keeps the previous binding (so the CLI can wire only the
    pieces it has ready). Tests pass all 3 fixtures explicitly.
    """
    global _SERVER_RESTART_HOOK, _BENCH_HOOK, _ACCURACY_GATE_HOOK
    if server_restart is not None:
        _SERVER_RESTART_HOOK = server_restart
    if bench is not None:
        _BENCH_HOOK = bench
    if accuracy_gate is not None:
        _ACCURACY_GATE_HOOK = accuracy_gate


def reset_integrate_hooks() -> None:
    """Drop all hooks. Used by test teardown + the P1 mock branch."""
    global _SERVER_RESTART_HOOK, _BENCH_HOOK, _ACCURACY_GATE_HOOK
    _SERVER_RESTART_HOOK = None
    _BENCH_HOOK = None
    _ACCURACY_GATE_HOOK = None


async def framework_integrate_handler(
    payload: dict[str, Any],
    *,
    session_dir: Path,
) -> dict[str, Any]:
    """Apply a KEEP'd patch -> restart server -> bench + accuracy gate
    -> KEEP / REVERT / NEEDS_REVIEW verdict.

    Two operating modes:

    * **P1 mock branch** (no hooks injected via
      :func:`set_integrate_hooks`): returns a canned IntegrateSuccess
      (KEEP) envelope so the 5-role protocol smoke keeps green
      without real server / bench / gate plumbing.
    * **P3 real branch** (hooks injected): runs the full lifecycle
      from :mod:`._patch_lifecycle` (backup -> apply -> restart ->
      bench -> gate -> verdict -> rollback-if-REVERT) and returns
      IntegrateSuccess / IntegrateFailure per design §4.8 fault tree.
    """
    started_ms = int(asyncio.get_event_loop().time() * 1000) if False else 0
    import time as _time
    started = _time.monotonic()

    patch_id_in = str(payload.get("patch_id") or "").strip()
    if (
        _SERVER_RESTART_HOOK is None
        or _BENCH_HOOK is None
        or _ACCURACY_GATE_HOOK is None
    ):
        # P1 mock branch -- behave like the original canned response.
        log.info(
            "framework_integrate_handler[P1 mock] session=%s patch_id=%s",
            Path(session_dir).name, patch_id_in or "fw-mock",
        )
        return {
            "status": "succeeded",
            "payload_kind": "IntegrateSuccess",
            "verdict": "KEEP",
            "patch_id": patch_id_in or "fw-mock",
            "tput_before": 0.0,
            "tput_after": 0.0,
            "accuracy_before": 0.0,
            "accuracy_after": 0.0,
            "accuracy_drop": 0.0,
            "stage_b_elapsed_ms": 200,
        }

    # ----- P3 real path -----
    from ._patch_lifecycle import (
        BackupRef,
        PatchApplyError,
        VerdictInputs,
        apply_patch,
        backup_files,
        decide_verdict,
        generate_patch_id,
        rollback_backup,
    )

    patch_path = str(payload.get("patch_path") or "").strip()
    if not patch_path:
        return _integrate_failure(
            patch_id_in or "fw-unknown", started, "missing_patch_path",
            "framework_integrate payload missing patch_path",
        )
    patch_pp = Path(patch_path).expanduser()
    if not patch_pp.is_file():
        return _integrate_failure(
            patch_id_in or "fw-unknown", started, "patch_not_found",
            f"patch file does not exist: {patch_pp}",
        )

    patch_id = patch_id_in or generate_patch_id("fw")
    files_touched = _extract_files_touched(patch_pp)
    if not files_touched:
        return _integrate_failure(
            patch_id, started, "patch_empty",
            "patch contains no +++ b/<file> entries -- nothing to back up",
        )
    baseline_tput = float(payload.get("baseline_tput") or 0.0)
    baseline_accuracy = float(payload.get("baseline_accuracy") or 0.0)

    # Stage 1: backup. Hard fail -> IntegrateFailure(backup_failed).
    try:
        ref = await asyncio.to_thread(
            backup_files, patch_id, files_touched, session_dir=session_dir,
        )
    except (FileNotFoundError, OSError) as exc:
        return _integrate_failure(
            patch_id, started, "backup_failed", repr(exc),
        )

    # Stage 2: apply. ``cwd=/`` so ``+++ b/<absolute-rel>`` resolves to
    # ``/<absolute-rel>`` -- matches how the test fixtures + production
    # patches (with /sgl-workspace/... paths) are anchored. Failure ->
    # IntegrateFailure(patch_apply_failed).
    try:
        await asyncio.to_thread(apply_patch, patch_pp, cwd=Path("/"))
    except PatchApplyError as exc:
        # No need to rollback -- apply failed before mutation.
        return _integrate_failure(
            patch_id, started, "patch_apply_failed", str(exc),
        )

    # Stage 3: server restart. Failure -> rollback + fail.
    try:
        await asyncio.to_thread(
            _SERVER_RESTART_HOOK, session_dir, patch_id,
        )
    except ServerRestartError as exc:
        _ = await asyncio.to_thread(rollback_backup, ref)
        return _integrate_failure(
            patch_id, started, "server_restart_failed", str(exc),
        )

    # Stage 4: bench + accuracy gate -> verdict.
    bench: dict[str, Any] = {}
    try:
        bench = await asyncio.to_thread(_BENCH_HOOK, session_dir, patch_id)
    except Exception as exc:  # noqa: BLE001
        log.exception("bench_hook crashed")
        _ = await asyncio.to_thread(rollback_backup, ref)
        return _integrate_failure(
            patch_id, started, "bench_failed", repr(exc),
        )
    bench_ok = bool(bench.get("ok", True))
    tput_after = float(bench.get("tput") or 0.0)
    bench_reason = str(bench.get("reason") or "")

    accuracy_after: float | None = None
    try:
        accuracy_after = await asyncio.to_thread(
            _ACCURACY_GATE_HOOK, session_dir, patch_id,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("accuracy_gate_hook crashed")
        # Treat as missing accuracy -> NEEDS_REVIEW unless other gates fail.
        accuracy_after = None
        bench_reason = (bench_reason + "; " if bench_reason else "") + (
            f"accuracy_gate_exception: {exc!r}"
        )

    verdict = decide_verdict(VerdictInputs(
        baseline_tput=baseline_tput,
        baseline_accuracy=baseline_accuracy,
        tput_after=tput_after,
        accuracy_after=accuracy_after,
        bench_ok=bench_ok,
        bench_reason=bench_reason,
    ))
    if verdict.verdict == "REVERT":
        _ = await asyncio.to_thread(rollback_backup, ref)
    elif verdict.verdict == "KEEP":
        # PR-I: append the KEEP outcome to framework_optimization KB
        # partition so subsequent runs see this lesson as a prior. Best
        # effort -- KB write failure must not flip a KEEP into a
        # failure envelope.
        try:
            from framework_agent.agent.kb_write import append_keep_lesson
            await asyncio.to_thread(
                append_keep_lesson,
                framework=str(payload.get("target_framework") or "").strip(),
                patch_id=patch_id,
                summary=str(payload.get("summary") or
                            f"patch {patch_id} gain={verdict.gain_pct:.2f}%"),
                rationale=str(payload.get("rationale") or verdict.reason),
                gain_pct=verdict.gain_pct,
                session_id=Path(session_dir).name,
            )
        except Exception:  # noqa: BLE001
            log.exception("append_keep_lesson failed -- KEEP verdict still stands")

    elapsed_ms = int((_time.monotonic() - started) * 1000)
    return {
        "status": "succeeded",
        "payload_kind": "IntegrateSuccess",
        "verdict": verdict.verdict,
        "patch_id": patch_id,
        "tput_before": baseline_tput,
        "tput_after": tput_after,
        "accuracy_before": baseline_accuracy,
        "accuracy_after": accuracy_after if accuracy_after is not None else 0.0,
        "accuracy_drop": verdict.accuracy_drop_pct,
        "gain_pct": verdict.gain_pct,
        "reason": verdict.reason,
        "stage_b_elapsed_ms": elapsed_ms,
    }


def _extract_files_touched(patch_path: Path) -> list[Path]:
    """Read +++ b/<rel> entries from a unified diff and return absolute
    paths.

    Path resolution order (first hit wins):

    1. Treat ``rel`` (with ``b/`` prefix stripped) as a filesystem-root
       relative path: ``/`` + ``rel``. Apply_patch runs with cwd=``/``
       in tests so this matches how the diff's hunks are anchored.
    2. Resolve against ``patch_path.parent`` (fixture-style relative diff).
    3. Resolve against the current working directory.
    """
    files: list[Path] = []
    try:
        text = patch_path.read_text(encoding="utf-8")
    except OSError:
        return []
    base = patch_path.parent
    for line in text.splitlines():
        if not line.startswith("+++ "):
            continue
        rel = line[4:].strip()
        if rel.startswith("b/"):
            rel = rel[2:]
        if not rel or rel == "/dev/null":
            continue
        candidates = [
            Path("/" + rel),
            (base / rel).resolve(),
            Path(rel).resolve(),
        ]
        for cand in candidates:
            if cand.is_file():
                files.append(cand)
                break
    return files


def _integrate_failure(
    patch_id: str,
    started: float,
    reason: str,
    detail: str,
) -> dict[str, Any]:
    import time as _time
    return {
        "status": "failed",
        "payload_kind": "IntegrateFailure",
        "reason": reason,
        "detail": detail,
        "patch_id": patch_id,
        "stage_b_elapsed_ms": int((_time.monotonic() - started) * 1000),
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
    "ServerRestartError",
    "framework_integrate_handler",
    "framework_optimize_handler",
    "get_framework_handler",
    "reset_integrate_hooks",
    "set_framework_backend",
    "set_integrate_hooks",
]
