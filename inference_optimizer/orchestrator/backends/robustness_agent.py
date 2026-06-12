# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""RobustnessAgentBackend — bridges the ``robustness-agent`` runtime into
the Coordinator as a real Robustness Backend.

Mirrors :class:`CriticAgentBackend`'s subprocess transport, simplified
because the robustness reactor is deterministic (no LLM, KB, or two-phase
handshake). Each tick is one ``runtime.cli tick`` invocation whose
``emit.json`` carries an ``intent_envelope`` like critic-agent's. Test seam:
``runtime_caller_factory`` bypasses the subprocess.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ...protocol.intent import (
    IntentValidationError,
    NoIntentEmitted,
    validate_envelope,
)
from .base import BackendError, BackendTurnResult
from .critic_agent import RuntimeCall, RuntimeCaller


log = logging.getLogger(__name__)


ROBUSTNESS_AGENT_RUNTIME_TIMEOUT_SEC = 30
ROBUSTNESS_AGENT_WORKDIR_KEEP_COUNT = 50


def _default_runtime_caller(call: RuntimeCall) -> None:
    """Real implementation — runs ``python -m robustness_agent.runtime.cli tick``.

    Sets ``PYTHONPATH=<root>/src`` + ``cwd=<root>`` so the package resolves
    without a pip-install (matching the critic-agent convention).
    """
    if call.phase != "tick":
        raise BackendError(
            f"RobustnessAgentBackend: unsupported runtime phase {call.phase!r} "
            f"(expected 'tick')"
        )
    cmd = [
        sys.executable, "-m", "robustness_agent.runtime.cli", "tick",
        "--request", str(call.request_path),
        "--out", str(call.out_path),
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(call.cwd),
            env=call.env,
            capture_output=True,
            text=True,
            timeout=ROBUSTNESS_AGENT_RUNTIME_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired as exc:
        raise BackendError(
            f"robustness-agent runtime.cli tick timed out after "
            f"{ROBUSTNESS_AGENT_RUNTIME_TIMEOUT_SEC}s (cwd={call.cwd})"
        ) from exc
    except FileNotFoundError as exc:
        raise BackendError(
            f"robustness-agent runtime.cli tick could not start "
            f"(python={sys.executable!r}, cwd={call.cwd}): {exc}"
        ) from exc

    if proc.returncode != 0:
        raise BackendError(
            f"robustness-agent runtime.cli tick exited rc={proc.returncode}: "
            f"stderr={proc.stderr.strip()[:500]!r}"
        )


# ---------------------------------------------------------------------------
@dataclass
class RobustnessAgentBackend:
    """Real Robustness backend that drives the robustness-agent runtime.

    Parameters
    ----------
    robustness_agent_root:
        Package root containing ``src/robustness_agent/runtime/cli.py``
        (invoked with ``cwd=root`` + ``PYTHONPATH=<root>/src``).
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
        confirms ``src/robustness_agent/runtime/cli.py`` exists, wires up either
        the test ``runtime_caller_factory`` or the real subprocess caller, and
        snapshots the forwarded ``options``.

        Raises:
            BackendError: If the expected ``runtime/cli.py`` module is not found
                under ``robustness_agent_root``.
        """
        self.robustness_agent_root = Path(self.robustness_agent_root)
        self.session_dir = Path(self.session_dir)
        cli_module = (
            self.robustness_agent_root / "src" / "robustness_agent"
            / "runtime" / "cli.py"
        )
        if not cli_module.is_file():
            raise BackendError(
                f"RobustnessAgentBackend: src/robustness_agent/runtime/cli.py "
                f"not found under {self.robustness_agent_root!s} — set "
                f"ROBUSTNESS_AGENT_ROOT or check the install"
            )

        if self.runtime_caller_factory is not None:
            object.__setattr__(
                self, "_runtime_caller", self.runtime_caller_factory(),
            )
        else:
            object.__setattr__(
                self, "_runtime_caller", _default_runtime_caller,
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

        workdir = self._allocate_workdir(turn_idx)
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
            raise BackendError(
                f"RobustnessAgentBackend: failed to read emit.json from "
                f"{emit_path}: {exc}"
            ) from exc

        envelope = emit.get("intent_envelope")
        if not isinstance(envelope, dict):
            raise BackendError(
                f"RobustnessAgentBackend: emit.json missing intent_envelope "
                f"(got keys={sorted(emit.keys())!r})"
            )
        try:
            intents = validate_envelope(envelope)
        except IntentValidationError as exc:
            raise NoIntentEmitted(
                f"robustness_agent_envelope_invalid: {exc}"
            ) from exc

        parse_warnings = list(emit.get("parse_warnings") or [])
        runtime_tick_index = emit.get("tick_index")
        intent_summary = [
            (i.type.value, i.payload.get("severity") or i.payload.get("topic"))
            for i in intents
        ]
        log.info(
            "robustness_agent_backend turn=%d session=%s tick_index=%s "
            "intents=%s parse_warnings=%d",
            turn_idx, session_id, runtime_tick_index,
            intent_summary, len(parse_warnings),
        )
        self.calls.append({
            "turn_idx": turn_idx,
            "tick_index": runtime_tick_index,
            "intents": intent_summary,
            "parse_warnings": parse_warnings,
            "workdir": str(workdir),
        })

        # Author-time breakdown capture: record this robustness signal before
        # the backend prunes old workdirs (composed into critic_robustness).
        try:
            from ...breakdown.recorder import instrument
            instrument.record_robustness_signal(
                self.session_dir, workdir=workdir,
            )
        except Exception:  # noqa: BLE001
            pass

        return BackendTurnResult(
            intents=intents,
            raw_text="(robustness-agent)",
            metadata={
                "session_id": session_id,
                "turn_idx": turn_idx,
                "tick_index": runtime_tick_index,
                "parse_warnings": parse_warnings,
            },
        )

    def _allocate_workdir(self, turn_idx: int) -> Path:
        """Create and return a per-turn workdir, pruning stale ones first.

        Args:
            turn_idx (int): Zero-based index of the current turn, used as the
                zero-padded subdirectory name.

        Returns:
            Path: The created ``<session>/robustness-workdir/<turn>/`` directory.
        """
        root = self.session_dir / "robustness-workdir"
        root.mkdir(parents=True, exist_ok=True)
        self._prune_old_workdirs(root, keep=ROBUSTNESS_AGENT_WORKDIR_KEEP_COUNT)
        wd = root / f"{turn_idx:06d}"
        wd.mkdir(parents=True, exist_ok=True)
        return wd

    @staticmethod
    def _prune_old_workdirs(root: Path, *, keep: int) -> None:
        """Remove all but the ``keep`` most recent ``<turn>/`` subdirs.

        Best-effort: directory-listing and removal errors are swallowed so a
        cleanup hiccup never fails a turn.

        Args:
            root (Path): The ``robustness-workdir`` parent directory to prune.
            keep (int): Number of most recent turn subdirectories to retain.
        """
        try:
            entries = sorted(
                (p for p in root.iterdir() if p.is_dir()),
                key=lambda p: p.name,
            )
        except OSError:
            return
        if len(entries) <= keep:
            return
        for stale in entries[: len(entries) - keep]:
            try:
                for child in stale.rglob("*"):
                    if child.is_file():
                        child.unlink(missing_ok=True)
                for child in sorted(stale.rglob("*"), key=lambda p: -len(p.parts)):
                    if child.is_dir():
                        try:
                            child.rmdir()
                        except OSError:
                            pass
                stale.rmdir()
            except OSError:
                continue

    def _build_runtime_env(self) -> dict[str, str]:
        """Build subprocess env with ``<root>/src`` prepended to PYTHONPATH
        (preserving any existing value) so the CLI module resolves."""
        env = dict(os.environ)
        src = str(self.robustness_agent_root / "src")
        existing = env.get("PYTHONPATH", "").strip()
        if existing:
            env["PYTHONPATH"] = src + os.pathsep + existing
        else:
            env["PYTHONPATH"] = src
        env.setdefault("ROBUSTNESS_AGENT_SESSION_DIR", str(self.session_dir))
        return env


__all__ = [
    "ROBUSTNESS_AGENT_RUNTIME_TIMEOUT_SEC",
    "ROBUSTNESS_AGENT_WORKDIR_KEEP_COUNT",
    "RobustnessAgentBackend",
    "_default_runtime_caller",
]
