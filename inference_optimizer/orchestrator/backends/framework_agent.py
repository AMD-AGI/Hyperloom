"""Real Framework agent backend — subprocess bridge to ``fa agent`` (P2 PR-F).

Drives the sibling ``framework-agent/agent/cli.py`` subprocess via the
``fa agent prepare-task`` + ``fa agent commit-result`` two-stage
protocol (design §9.1). Selected by the ``--framework-agent`` CLI flag.

PR-F scope (this commit):

* ``run_optimize`` — writes a ``task.json`` under
  ``runs/framework/<task_id>/``, invokes ``fa agent prepare-task`` to
  produce a bundle, runs the AST scan in-process (P2 PR-E gives us
  ``scan_framework_args``), validates a synthesized OptimizeSuccess
  envelope via ``fa agent commit-result``, returns the envelope.
  **No LLM invocation yet** — P3 PR-G adds the patch_proposer LLM loop
  and a real ``patch_path``.
* ``run_integrate`` — still raises NotImplementedError; PR-H implements
  the apply + bench + accuracy gate flow.

The subprocess bridge is intentionally thin so PR-G can swap the
"synthesize OptimizeSuccess envelope" step with a real LLM loop
without touching the IO side wiring.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from .base import Backend, BackendError, BackendTurnResult


log = logging.getLogger(__name__)


class FrameworkAgentBackend(Backend):
    """Subprocess bridge to ``framework-agent/agent/`` runtime."""

    name = "framework_agent"

    def __init__(
        self,
        *,
        framework_agent_root: Path,
        session_dir: Path,
        options: dict[str, Any] | None = None,
    ) -> None:
        self._root = Path(framework_agent_root)
        self._session_dir = Path(session_dir)
        self._options: dict[str, Any] = dict(options or {})
        self._fa_python = self._options.get("python") or sys.executable
        self._fa_module = "framework_agent.runtime.cli"

    async def run(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        tools: list[str] | None = None,
        max_turns: int = 1,
    ) -> BackendTurnResult:
        """Heartbeat-only -- the active path is the programmatic handler.

        Coordinator's reactor only routes intents for non-handler
        REQUEST kinds through ``backend.run``. P1/P2/P3 framework
        REQUEST kinds all have programmatic handlers, so this method
        never carries real traffic.
        """
        return BackendTurnResult(intents=[], raw_text="", metadata={})

    # ------------------------------------------------------------------
    # Stage A wrapper -- AST scan + envelope build via fa agent CLI
    # ------------------------------------------------------------------
    def run_optimize(
        self,
        *,
        session_dir: str,
        target_framework: str,
        kb_partition: str = "framework_optimization",
        ast_scan_enabled: bool = True,
        ast_frameworks: tuple[str, ...] = (),
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """Run the framework_optimize Stage A flow.

        Returns the OptimizeSuccess (or OptimizeFailure) payload dict
        only; the handler is responsible for wrapping it into the
        ``{kind, status, result}`` reactor envelope.
        """
        started = time.monotonic()
        tid = task_id or f"fw-{time.strftime('%Y%m%d')}-{uuid.uuid4().hex[:8]}"
        task_dir = (
            Path(session_dir) / "runs" / "framework" / tid
        ).resolve()
        task_dir.mkdir(parents=True, exist_ok=True)

        task = {
            "task_id": tid,
            "kind": "framework_optimize",
            "session_dir": str(Path(session_dir).resolve()),
            "target_framework": target_framework,
            "ast_scan_enabled": bool(ast_scan_enabled),
            "ast_frameworks": list(ast_frameworks),
            "kb_partition": kb_partition,
        }
        task_path = task_dir / "task.json"
        task_path.write_text(json.dumps(task, indent=2), encoding="utf-8")

        bundle_path = task_dir / "bundle.json"
        try:
            self._invoke_fa(
                "agent", "prepare-task",
                "--task", str(task_path),
                "--output-bundle", str(bundle_path),
            )
        except BackendError as exc:
            return self._failure(tid, started, "prepare_task_failed", str(exc))

        # PR-F runs AST scan in-process for now. PR-G swaps this with
        # the patch_proposer LLM loop (which itself consumes AST
        # findings). Importing here avoids a hard libcst dependency
        # at IO import time.
        discovered_flags: dict[str, list[str]] = {}
        ast_mode = "skipped"
        if ast_scan_enabled:
            try:
                discovered_flags, ast_mode = self._run_ast_scan(
                    target_framework=target_framework,
                    ast_frameworks=tuple(ast_frameworks),
                )
            except FileNotFoundError as exc:
                return self._failure(
                    tid, started, "source_not_found", str(exc),
                )
            except Exception as exc:  # noqa: BLE001
                log.exception("AST scan failed for task=%s", tid)
                return self._failure(
                    tid, started, "ast_scan_failed", repr(exc),
                )

        # PR-F envelope: flags-only OptimizeSuccess (no patch yet).
        # PR-G upgrades the path with a real patch_proposer.diff.
        envelope = {
            "payload_kind": "OptimizeSuccess",
            "patch_path": "",  # PR-G fills with proposal.diff path
            "predicted_gain_pct": 0.0,
            "rationale": (
                f"PR-F: AST scan ({ast_mode}) surfaced "
                f"{sum(len(v) for v in discovered_flags.values())} flags; "
                "patch proposer lands in PR-G"
            ),
            "discovered_flags": discovered_flags,
            "target_framework": target_framework,
            "stage_a_elapsed_ms": int((time.monotonic() - started) * 1000),
        }

        # Validate via fa agent commit-result (round-trips through
        # the sibling skill's jsonschema validator).
        env_path = task_dir / "envelope.json"
        env_path.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
        try:
            self._invoke_fa(
                "agent", "commit-result",
                "--envelope", str(env_path),
                "--task-id", tid,
                "--session-dir", str(Path(session_dir).resolve()),
            )
        except BackendError as exc:
            return self._failure(tid, started, "envelope_validate_failed", str(exc))

        return envelope

    def run_integrate(
        self,
        *,
        session_dir: str,
        patch_path: str,
        patch_id: str,
    ) -> dict[str, Any]:
        """PR-H lands the patch lifecycle + bench + accuracy gate."""
        raise NotImplementedError(
            "FrameworkAgentBackend.run_integrate is not implemented in P2. "
            "PR-H lands the patch lifecycle + bench + accuracy gate."
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _invoke_fa(self, *args: str) -> subprocess.CompletedProcess:
        """Spawn ``python -m framework_agent.runtime.cli ARGS``.

        Returns the completed process on rc=0; raises
        :class:`BackendError` otherwise. Uses the framework-agent root
        as cwd so editable installs resolve correctly without forcing
        the IO venv to live in the same path.
        """
        cmd = [self._fa_python, "-m", self._fa_module, *args]
        log.debug("fa agent invoke cmd=%s", cmd)
        proc = subprocess.run(
            cmd,
            cwd=str(self._root),
            check=False,
            capture_output=True,
            text=True,
            timeout=int(self._options.get("subprocess_timeout_sec", 60)),
        )
        if proc.returncode != 0:
            raise BackendError(
                f"fa agent {args[0] if args else '<no-args>'} failed "
                f"rc={proc.returncode}: stderr={proc.stderr.strip()[:400]!r}"
            )
        return proc

    def _run_ast_scan(
        self,
        *,
        target_framework: str,
        ast_frameworks: tuple[str, ...],
    ) -> tuple[dict[str, list[str]], str]:
        """Run AST scan via the agent package; return (flags_by_fw, mode)."""
        from framework_agent.agent.ast_scanner import scan_framework_args
        from framework_agent.agent.flag_discovery import cli_flag_names
        from framework_agent.agent.source_resolver import (
            FrameworkSourceMissing,
            resolve_framework_sources,
        )

        # ``ast_frameworks`` empty -> derive from target_framework
        # (mirrors CLI parse default in cli._seed_shared_state).
        wanted = tuple(ast_frameworks) or (target_framework,)
        resolved = resolve_framework_sources(wanted)
        if not resolved:
            raise FrameworkSourceMissing(
                f"could not resolve any of {wanted!r} source roots; "
                "set VLLM_SOURCE_ROOT / SGLANG_SOURCE_ROOT or use "
                "--no-framework-ast to skip AST scan"
            )
        flags_by_fw: dict[str, list[str]] = {}
        agg_mode = "libcst"
        for fw, root in resolved.items():
            result = scan_framework_args(fw, root)
            flags_by_fw[fw] = cli_flag_names(result.flags)
            if result.mode == "grep_fallback":
                agg_mode = "grep_fallback"
        return flags_by_fw, agg_mode

    @staticmethod
    def _failure(
        task_id: str, started: float, reason: str, detail: str,
    ) -> dict[str, Any]:
        """Build an OptimizeFailure envelope (matches §4.6 schema)."""
        return {
            "payload_kind": "OptimizeFailure",
            "reason": reason,
            "detail": detail,
            "task_id": task_id,
            "stage_a_elapsed_ms": int((time.monotonic() - started) * 1000),
        }


__all__ = ["FrameworkAgentBackend"]
