"""Real Framework agent backend — placeholder until PR-F.

Counterpart of :class:`CriticAgentBackend` / :class:`RobustnessAgentBackend`:
drives the sibling ``framework-agent/agent/cli.py`` subprocess via
JSON-over-stdio. P2 PR-F will:

* implement ``run_optimize`` (calls ``fa agent prepare-task`` to bundle
  the LLM input, then ``fa agent commit-result`` to consume the envelope);
* validate the envelope against the §4.6 jsonschema before returning;
* surface per-stage timing into the orchestrator's ``stage_log``.

PR-A1 / PR-B keep this as a deliberate ``NotImplementedError``
placeholder so the CLI can wire ``--framework-agent`` to it without
shipping subprocess code yet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import Backend, BackendError, BackendTurnResult


class FrameworkAgentBackend(Backend):
    """Subprocess bridge to ``framework-agent/agent/`` runtime.

    Selected by ``--framework-agent`` CLI flag. Not yet implemented —
    raises on first use; PR-F will fill the run_optimize /
    run_integrate methods.
    """

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

    async def run_turn(self, *args: Any, **kwargs: Any) -> BackendTurnResult:
        # Heartbeat-equivalent: framework agent is responder-only and
        # the active path is the programmatic handler. The reactor
        # loop calls into this only when no handler is registered for
        # the incoming REQUEST kind, which P1 never does (both
        # framework_optimize and framework_integrate have handlers).
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
        """Invoke ``fa agent prepare-task`` + ``fa agent commit-result``.

        PR-F fills the real subprocess invocation. P1 raises so any
        accidental wire-up surfaces immediately.
        """
        raise NotImplementedError(
            "FrameworkAgentBackend.run_optimize is not implemented in P1. "
            "Use --framework-mock for protocol smoke; PR-F lands the "
            "real subprocess bridge."
        )

    def run_integrate(
        self,
        *,
        session_dir: str,
        patch_path: str,
        patch_id: str,
    ) -> dict[str, Any]:
        """Invoke ``fa agent apply-patch`` (PR-H).

        P1 raises. PR-H implements the full apply + bench + gate flow.
        """
        raise NotImplementedError(
            "FrameworkAgentBackend.run_integrate is not implemented in P1. "
            "PR-H lands the patch lifecycle + bench + accuracy gate."
        )


__all__ = ["FrameworkAgentBackend"]
