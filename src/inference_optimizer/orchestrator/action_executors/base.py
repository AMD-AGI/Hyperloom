"""``ActionExecutor`` base class + executor registry.

An ActionExecutor is the *Python adapter* for one action in the
catalogue. It receives an :class:`ExecutorContext` (task, action
metadata, lanes held, session dir, env vars, an emit-intent callback)
and is responsible for:

1. Validating its required env vars are present (raise
   :class:`ExecutorEnvError` early if not — the runner will fall back to
   the LLM-driven path).
2. Shelling out to the bundled skill scripts (resolved via
   :func:`paths.skill_script`).
3. Parsing the script's outputs (``metrics.json`` / ``results.tsv`` /
   ``eval_summary_<task>.json``) into structured fields on
   :class:`ExecutorResult`.
4. Emitting at least one ``update_state`` intent so the Conductor's
   :class:`SharedState` learns about the new measurement.

Failure semantics mirror :class:`SubAgentRunner.run`:

* Recoverable env / setup problem → raise :class:`ExecutorEnvError`. The
  runner will fall back to the LLM path; the task is *not* terminated.
* Real run failed (bad exit code, missing artefact) → return
  ``ExecutorResult(status="failed", ...)``. The runner marks the task
  ``failed`` in the registry and moves on.
* Successful run → ``status="succeeded"`` with metrics + artifacts
  populated; the runner marks the task ``succeeded``.
"""
from __future__ import annotations

import logging
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

if TYPE_CHECKING:  # pragma: no cover - type-only
    from ..action_registry import ActionMetadata
    from ..intent_parser import Intent
    from ..task_registry import Task


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error / result types
# ---------------------------------------------------------------------------
class ExecutorEnvError(RuntimeError):
    """Required env var / file missing — caller should fall back to LLM path.

    The Conductor uses the message to record a structured ``observation``
    event so subsequent reactor turns can see why the executor opted out.
    """


@dataclass
class ExecutorResult:
    """Return value of :meth:`ActionExecutor.run`."""

    status: str  # "succeeded" | "failed" | "needs_manual_review"
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    intents: list["Intent"] = field(default_factory=list)
    notes: str = ""
    rc: int | None = None  # exit code of the underlying script, if any


@dataclass
class ExecutorContext:
    """Bundle of inputs an executor receives from :class:`SubAgentRunner`."""

    task: "Task"
    action_meta: "ActionMetadata"
    lanes_held: list[str]
    session_dir: Path
    env: dict[str, str] = field(default_factory=dict)
    # Optional — when present, the executor can publish intents directly
    # instead of returning them in ``ExecutorResult.intents``. Useful for
    # actions that want to stream progress mid-run.
    on_event: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None

    # Convenience accessors --------------------------------------------------
    @property
    def task_id(self) -> str:
        return getattr(self.task, "task_id", "unknown")

    @property
    def task_params(self) -> dict[str, Any]:
        """Resolve the ``params`` sub-dict the user originally passed
        (``delegate.payload.params``). The task row stores it nested
        under ``task.params["params"]`` per :class:`Conductor._handle_delegate`."""
        raw = getattr(self.task, "params", None) or {}
        if not isinstance(raw, dict):
            return {}
        inner = raw.get("params") if isinstance(raw.get("params"), dict) else None
        return dict(inner or {})

    def env_get(self, key: str, default: str = "") -> str:
        return self.env.get(key, default) or default

    def require_env(self, *keys: str) -> dict[str, str]:
        """Resolve every key from :attr:`env`; raise if any is missing."""
        out: dict[str, str] = {}
        missing: list[str] = []
        for k in keys:
            v = self.env.get(k)
            if v is None or v == "":
                missing.append(k)
            else:
                out[k] = v
        if missing:
            raise ExecutorEnvError(
                f"missing required env vars for "
                f"{self.action_meta.name!r}: {missing!r}"
            )
        return out

    def results_dir(self) -> Path:
        """Per-task scratch directory under ``<session>/results/<task_id>/``."""
        d = self.session_dir / "results" / self.task_id
        d.mkdir(parents=True, exist_ok=True)
        return d


# ---------------------------------------------------------------------------
# Subprocess seam
# ---------------------------------------------------------------------------
async def run_subprocess(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
    timeout_s: float | None = None,
    log_path: Path | None = None,
) -> int:
    """Tiny async subprocess wrapper used by every executor.

    Streams stdout+stderr to ``log_path`` (created+truncated) so the
    user can ``tail -f`` it. Returns the exit code; never raises on
    non-zero rc — callers decide what to do with it. Raises only on
    timeout / OS-level failures.

    Tests monkey-patch this function rather than calling
    ``asyncio.create_subprocess_exec`` directly so the executor logic
    stays exercised without spawning real processes.
    """
    import asyncio

    log.info("subprocess: %s (cwd=%s)", _safe_cmd(cmd), cwd or "<inherit>")
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = log_path.open("w", encoding="utf-8", buffering=1)
    else:
        log_fh = None
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(cwd) if cwd else None,
            env=env,
        )
        assert proc.stdout is not None  # nosec - PIPE is set
        if timeout_s is not None:
            try:
                async with asyncio.timeout(timeout_s):
                    rc = await _drain(proc, log_fh)
            except TimeoutError:
                proc.kill()
                await proc.wait()
                raise
        else:
            rc = await _drain(proc, log_fh)
        return rc
    finally:
        if log_fh is not None:
            try:
                log_fh.close()
            except OSError:
                pass


async def _drain(proc, log_fh) -> int:
    """Stream proc stdout into ``log_fh`` (and INFO-level log) until EOF."""
    assert proc.stdout is not None
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        text = line.decode("utf-8", errors="replace").rstrip()
        if log_fh is not None:
            log_fh.write(text + "\n")
        log.debug("proc> %s", text)
    return await proc.wait()


def _safe_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(c)) for c in cmd)


# ---------------------------------------------------------------------------
# ActionExecutor ABC
# ---------------------------------------------------------------------------
class ActionExecutor:
    """Base class — subclasses implement :meth:`run`.

    Subclasses MUST set the class-level ``name`` attribute to match the
    ``ActionMetadata.name`` they handle. The :func:`register_executor`
    decorator + module-import in :mod:`__init__.py` wires them into
    :data:`EXECUTOR_REGISTRY`.
    """

    name: str = ""
    timeout_s: float | None = 60 * 60  # 1 hour default cap

    async def run(self, ctx: ExecutorContext) -> ExecutorResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
EXECUTOR_REGISTRY: dict[str, ActionExecutor] = {}


def _normalize(name: str) -> str:
    """Match ``kernel-opt`` and ``kernel_opt`` to the same registry slot."""
    return name.replace("-", "_")


def register_executor(executor: ActionExecutor) -> ActionExecutor:
    """Decorator-friendly registration. Idempotent on re-import."""
    if not executor.name:
        raise ValueError(f"{type(executor).__name__} must set a non-empty .name")
    EXECUTOR_REGISTRY[_normalize(executor.name)] = executor
    return executor


def get_executor(action_name: str) -> ActionExecutor | None:
    """Resolve an executor for a given action name (hyphen-insensitive)."""
    return EXECUTOR_REGISTRY.get(_normalize(action_name))


__all__ = [
    "ActionExecutor",
    "EXECUTOR_REGISTRY",
    "ExecutorContext",
    "ExecutorEnvError",
    "ExecutorResult",
    "_normalize",
    "_safe_cmd",
    "get_executor",
    "register_executor",
    "run_subprocess",
]
