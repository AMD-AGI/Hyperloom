# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""RobustnessAgentBackend — bridges the ``hyperloom.agents.robustness``
runtime into the Coordinator as a real Robustness Backend.

Each tick is one ``runtime.cli tick`` invocation whose ``emit.json`` carries
an ``intent_envelope``. Test seam: ``runtime_caller_factory`` bypasses the
subprocess.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from hyperloom.inference_optimizer.protocol.intent import (
    IntentValidationError,
    NoIntentEmitted,
    validate_envelope,
)
from hyperloom.inference_optimizer.session.session_paths import allocate_turn_workdir
from ._runtime_bridge import RuntimeCall, RuntimeCaller, invoke_runtime_cli
from .base import BackendError, BackendTurnResult


log = logging.getLogger(__name__)


ROBUSTNESS_AGENT_RUNTIME_TIMEOUT_SEC = 30
ROBUSTNESS_AGENT_WORKDIR_KEEP_COUNT = 50


def _default_runtime_caller(call: RuntimeCall) -> None:
    """Real implementation — runs ``python -m hyperloom.agents.robustness.runtime.cli tick``.

    Args:
        call: The invocation descriptor with phase, request / output paths,
            working directory, and subprocess env.

    Raises:
        BackendError: If the phase is not ``tick``, the subprocess times out,
            cannot start, or exits non-zero.
    """
    if call.phase != "tick":
        raise BackendError(f"RobustnessAgentBackend: unsupported runtime phase {call.phase!r} (expected 'tick')")
    invoke_runtime_cli(
        call,
        module="hyperloom.agents.robustness.runtime.cli",
        agent_label="robustness-agent",
        timeout_sec=ROBUSTNESS_AGENT_RUNTIME_TIMEOUT_SEC,
    )


# ---------------------------------------------------------------------------
@dataclass
class RobustnessAgentBackend:
    """Real Robustness backend that drives the robustness-agent runtime.

    Parameters
    ----------
    robustness_agent_root:
        Directory containing ``runtime/cli.py``
        (``src/hyperloom/agents/robustness/``). The CLI is invoked as the
        package-qualified ``python -m hyperloom.agents.robustness.runtime.cli``
        with ``cwd=robustness_agent_root``.
    session_dir:
        Coordinator session directory; scopes per-turn workdirs and is
        forwarded into ``request.options.session_dir``.
    options:
        Optional ``request.options`` overrides forwarded into every tick
        request. ``session_dir`` is auto-injected.
    runtime_caller_factory:
        Test seam returning a :data:`RuntimeCaller` that bypasses the subprocess.
    name:
        Backend instance name surfaced in the Coordinator startup banner.
    """

    robustness_agent_root: Path
    session_dir: Path
    options: dict[str, Any] | None = None
    runtime_caller_factory: Callable[[], RuntimeCaller] | None = None
    name: str = "robustness-agent"

    _turn_idx: int = field(default=0, init=False, repr=False)
    _options: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    calls: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        """Normalise paths, verify the CLI module, and select the runtime caller.

        Coerces ``robustness_agent_root`` and ``session_dir`` to :class:`Path`,
        confirms ``runtime/cli.py`` exists, wires up either the test
        ``runtime_caller_factory`` or the real subprocess caller, and
        snapshots the forwarded ``options``.

        Raises:
            BackendError: If the expected ``runtime/cli.py`` module is not found
                under ``robustness_agent_root``.
        """
        self.robustness_agent_root = Path(self.robustness_agent_root)
        self.session_dir = Path(self.session_dir)
        cli_module = self.robustness_agent_root / "runtime" / "cli.py"
        if not cli_module.is_file():
            raise BackendError(
                f"RobustnessAgentBackend: runtime/cli.py "
                f"not found under {self.robustness_agent_root!s} — set "
                f"ROBUSTNESS_AGENT_ROOT or check the install"
            )

        if self.runtime_caller_factory is not None:
            object.__setattr__(
                self,
                "_runtime_caller",
                self.runtime_caller_factory(),
            )
        else:
            object.__setattr__(
                self,
                "_runtime_caller",
                _default_runtime_caller,
            )

        self._options = dict(self.options or {})

    async def run(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        tools: list[str] | None = None,
        max_turns: int = 1,
    ) -> BackendTurnResult:
        """One Robustness turn — drive a single reactor tick over subprocess.

        Writes a ``coordinator_inbox`` request for the rendered ``prompt``,
        invokes one ``runtime.cli tick`` (off the event loop), reads back the
        emitted ``intent_envelope``, validates it into intents, and records
        per-turn telemetry.

        Args:
            prompt (str): The Coordinator-rendered prompt for this tick.
            system_prompt (str | None): Unused; robustness ticks are
                deterministic and ignore it.
            tools (list[str] | None): Unused; the runtime owns its own tooling.
            max_turns (int): Unused; one reactor tick is run per call.

        Returns:
            BackendTurnResult: The validated intents plus session/tick metadata
            and any parse warnings from the runtime.

        Raises:
            BackendError: If ``emit.json`` cannot be read or is missing a dict
                ``intent_envelope``.
            NoIntentEmitted: If the emitted envelope fails intent validation.
        """
        del system_prompt, tools, max_turns

        turn_idx = self._turn_idx
        self._turn_idx += 1

        workdir = allocate_turn_workdir(
            self.session_dir,
            "robustness-workdir",
            turn_idx,
            keep=ROBUSTNESS_AGENT_WORKDIR_KEEP_COUNT,
        )
        request_path = workdir / "request.json"
        emit_path = workdir / "emit.json"

        session_id = self.session_dir.name
        merged_options = dict(self._options)
        merged_options.setdefault("session_dir", str(self.session_dir))
        request: dict[str, Any] = {
            "kind": "coordinator_inbox",
            "session_id": session_id,
            "raw_prompt": prompt,
            "context": {"tick_index": turn_idx},
            "options": merged_options,
        }
        request_path.write_text(
            json.dumps(request, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        env = self._build_runtime_env()

        await asyncio.to_thread(
            self._runtime_caller,
            RuntimeCall(
                phase="tick",
                request_path=request_path,
                review_path=None,
                out_path=emit_path,
                cwd=self.robustness_agent_root,
                env=env,
            ),
        )

        try:
            emit = json.loads(emit_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BackendError(f"RobustnessAgentBackend: failed to read emit.json from {emit_path}: {exc}") from exc

        envelope = emit.get("intent_envelope")
        if not isinstance(envelope, dict):
            raise BackendError(
                f"RobustnessAgentBackend: emit.json missing intent_envelope (got keys={sorted(emit.keys())!r})"
            )
        try:
            intents = validate_envelope(envelope)
        except IntentValidationError as exc:
            raise NoIntentEmitted(f"robustness_agent_envelope_invalid: {exc}") from exc

        parse_warnings = list(emit.get("parse_warnings") or [])
        runtime_tick_index = emit.get("tick_index")
        intent_summary = [(i.type.value, i.payload.get("severity") or i.payload.get("topic")) for i in intents]
        log.info(
            "robustness_agent_backend turn=%d session=%s tick_index=%s intents=%s parse_warnings=%d",
            turn_idx,
            session_id,
            runtime_tick_index,
            intent_summary,
            len(parse_warnings),
        )
        self.calls.append(
            {
                "turn_idx": turn_idx,
                "tick_index": runtime_tick_index,
                "intents": intent_summary,
                "parse_warnings": parse_warnings,
                "workdir": str(workdir),
            }
        )

        # Record the robustness signal before stale workdirs are pruned.
        try:
            from hyperloom.inference_optimizer.breakdown.recorder import instrument

            instrument.record_robustness_signal(
                self.session_dir,
                workdir=workdir,
            )
        except Exception:  # noqa: BLE001 — author-time capture must never break the agent loop
            log.debug("robustness breakdown capture failed", exc_info=True)

        metadata: dict[str, Any] = {
            "session_id": session_id,
            "turn_idx": turn_idx,
            "tick_index": runtime_tick_index,
            "parse_warnings": parse_warnings,
        }
        # Fold any RCA-LLM token spend into canonical metadata counters.
        self._merge_llm_usage(metadata, emit.get("llm_usage"))

        return BackendTurnResult(
            intents=intents,
            raw_text="(robustness-agent)",
            metadata=metadata,
        )

    @staticmethod
    def _merge_llm_usage(metadata: dict[str, Any], usage: Any) -> None:
        """Map a runtime ``llm_usage`` block onto canonical metadata counters.

        The robustness runtime reports ``{input_tokens, output_tokens, calls,
        latency_ms, model}``. We surface ``input_tokens`` / ``output_tokens``
        (the keys the reactor trace looks for) plus ``model`` so the ledger row
        carries a model name. No-op when ``usage`` is missing or shapeless.
        """
        if not isinstance(usage, dict):
            return
        it = usage.get("input_tokens")
        ot = usage.get("output_tokens")
        if it is None and ot is None:
            return
        metadata["input_tokens"] = it
        metadata["output_tokens"] = ot
        if usage.get("model"):
            metadata["model"] = usage.get("model")


    def _build_runtime_env(self) -> dict[str, str]:
        """Build the subprocess environment for ``runtime.cli`` invocations.

        The module resolves via the installed ``hyperloom`` namespace, so no
        ``PYTHONPATH`` prepending is needed.

        Returns:
            A copy of the current environment with the robustness
            session-dir hint applied.
        """
        env = dict(os.environ)
        env.setdefault("ROBUSTNESS_AGENT_SESSION_DIR", str(self.session_dir))
        return env


__all__ = [
    "ROBUSTNESS_AGENT_RUNTIME_TIMEOUT_SEC",
    "ROBUSTNESS_AGENT_WORKDIR_KEEP_COUNT",
    "RobustnessAgentBackend",
    "_default_runtime_caller",
]
