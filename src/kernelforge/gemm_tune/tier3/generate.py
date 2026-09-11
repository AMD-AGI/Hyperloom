# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Ask an agent to author one tuner from a mandate.

The optional LLM dependency is imported lazily. Writable sessions run in an
isolated git worktree so the guard can snapshot and roll back the generated file.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .mandate import TunerMandate

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 1800

#: Allow backend discovery and script authoring; ``timeout_s`` remains the hard limit.
MAX_AUTHORING_TURNS = 120

_SYSTEM_PROMPT = """\
You author one GPU kernel tuning script, to a fixed contract, and nothing else.

Your script proposes candidate configurations. It does not decide whether they
are good: a separate harness re-times whatever you propose with its own clock,
and only those measurements count. Write the script that finds genuinely fast
configurations and describes them precisely enough to be re-dispatched by code
that has never seen it.

Obey the mandate exactly, especially the correctness and timing requirements --
they exist because ignoring either has already produced confident wrong answers
on this hardware.
"""


@dataclass
class GeneratedTuner:
    """The outcome of asking for a tuner."""

    ok: bool
    script_path: Path | None = None
    reason: str = ""
    provider: str = ""
    session_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "script": str(self.script_path) if self.script_path else None,
            "reason": self.reason,
            "provider": self.provider,
            "session_id": self.session_id,
        }


def _user_prompt(mandate: TunerMandate, script_path: Path, retry_note: str) -> str:
    parts = [
        mandate.render(),
        "",
        "## Deliverable",
        f"Write a single self-contained Python 3 script to `{script_path}`.",
        "It must run with no arguments and produce both output files named above.",
    ]
    if retry_note:
        parts += [
            "",
            "## The previous attempt was rejected",
            retry_note,
            "Fix exactly this and keep everything else that worked.",
        ]
    return "\n".join(parts)


def generate_tuner(
    mandate: TunerMandate,
    work_dir: Path,
    *,
    model: str = "",
    timeout_s: int = DEFAULT_TIMEOUT_S,
    retry_note: str = "",
) -> GeneratedTuner:
    """Author a tuner script into ``work_dir``; never raises."""
    work_dir.mkdir(parents=True, exist_ok=True)
    script_path = work_dir / "tuner.py"

    try:
        from kernelforge.agent_backends.base import AgentRunSpec, AgentToolPolicy
        from kernelforge.agent_backends.registry import (
            create_registered_backend,
            resolve_agent_runtime,
            select_default_agent_provider,
        )
        from kernelforge.config import resolve_agent_model, resolve_agent_reasoning_effort
    except ImportError as exc:
        return GeneratedTuner(
            False,
            None,
            f"no agent provider available in this install ({exc}); "
            "generation is skipped and tuning continues without it",
        )

    try:
        # ``resolve_agent_runtime`` needs a provider name; picking one is a separate step that also checks the CLI is
        # actually installed.
        chosen = select_default_agent_provider(model)
        runtime = resolve_agent_runtime(
            chosen.name,
            # The model variable is per-provider, so it is only readable once
            # the provider is settled.
            model=model or resolve_agent_model(chosen.name),
            timeout_sec=timeout_s,
            reasoning_effort=resolve_agent_reasoning_effort(),
        )
        backend = create_registered_backend(runtime)
    except Exception as exc:  # noqa: BLE001 - provider setup must not fail tuning
        return GeneratedTuner(False, None, f"agent provider unusable: {exc!r}")

    isolated = _isolate(work_dir)
    if isolated is not None:
        return isolated

    spec = AgentRunSpec(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=_user_prompt(mandate, script_path, retry_note),
        cwd=str(work_dir),
        writable=True,
        timeout_sec=timeout_s,
        # Pre-approve tools because unattended sessions cannot answer permission
        # prompts; shell access is required to enumerate installed kernel ids.
        tool_policy=AgentToolPolicy(read=True, search=True, write=True, shell=True, max_turns=MAX_AUTHORING_TURNS),
        target_files=[str(script_path)],
        allow_untracked=True,
    )

    try:
        result = _run(backend, spec)
    except Exception as exc:  # noqa: BLE001
        return GeneratedTuner(False, None, f"authoring session failed: {exc!r}")

    provider = str(getattr(backend, "name", "") or "")
    session = str(getattr(result, "session_id", "") or "")
    if not script_path.is_file():
        # Preserve the agent's explanation for diagnosis and the next retry.
        said = " ".join(str(getattr(result, "text", "") or "").split())[-600:]
        return GeneratedTuner(
            False,
            None,
            f"the session ended ({getattr(result, 'end_reason', '?')}) without writing {script_path.name}"
            + (f"; it said: {said}" if said else "; it said nothing"),
            provider,
            session,
        )
    log.info("tier3: %s authored %s", provider or "agent", script_path)
    return GeneratedTuner(True, script_path, "", provider, session)


def _isolate(work_dir: Path) -> GeneratedTuner | None:
    """Initialize ``work_dir`` as its own git repository for workspace safety.

    This provides a local baseline and rollback target without exposing an
    enclosing operator checkout. Return an error or None.
    """
    if (work_dir / ".git").exists():
        return None
    steps = (
        ("init", "-q"),
        ("config", "user.email", "tier3@kernelforge.invalid"),
        ("config", "user.name", "kernelforge tier3"),
        # A baseline commit, so rollback has something to roll back to.
        ("commit", "-q", "--allow-empty", "-m", "empty sandbox"),
    )
    for args in steps:
        try:
            done = subprocess.run(
                ["git", *args],
                cwd=str(work_dir),
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return GeneratedTuner(False, None, f"could not prepare a sandbox worktree: git {args[0]}: {exc}")
        if done.returncode != 0:
            detail = (done.stderr or done.stdout or "").strip()[:300]
            return GeneratedTuner(
                False,
                None,
                f"could not prepare a sandbox worktree: git {args[0]} failed: {detail}",
            )
    log.info("tier3: initialised a sandbox worktree at %s for the authoring session", work_dir)
    return None


def _run(backend: Any, spec: Any) -> Any:
    """Drive the backend, whichever calling convention it offers."""
    run = backend.run
    import asyncio
    import inspect

    if inspect.iscoroutinefunction(run):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(run(spec))
        raise RuntimeError(
            "generate_tuner was called from a running event loop; call it from a "
            "worker thread so the authoring session can own its own loop"
        )
    return run(spec)
