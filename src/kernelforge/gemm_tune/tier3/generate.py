# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Ask an agent to author a tuner from a mandate.

``kernelforge.llm`` is imported inside the call, never at module scope. The standalone
wheel is meant to be the only thing a GPU box has to install to tune, and a test
asserts it imports with no ``kernelforge`` present; pulling an agent provider
in at import time would quietly make the LLM stack a tuning dependency. Absent,
this returns "unavailable" and the caller carries on without a generated tuner,
which is the same outcome as the gate being closed.

The agent writes one file and is told what it will be judged on. It is not shown
the existing tuners: the point of this tier is a capability nothing else has, and
a script derived from one that does is either the wrong shape or evidence the
gate should not have opened.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .mandate import TunerMandate

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 1800

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
    """Author a tuner script into ``work_dir``; never raises.

    Args:
        mandate: What the script has to cover and produce.
        work_dir: Sandbox directory; the agent may only write here.
        model: Provider model override, or "" for the configured default.
        timeout_s: Wall clock for the authoring session.
        retry_note: Why the previous attempt was rejected, when retrying.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    script_path = work_dir / "tuner.py"

    try:
        from kernelforge.agent_backends.base import AgentRunSpec
        from kernelforge.agent_backends.registry import (
            create_registered_backend,
            resolve_agent_runtime,
            select_default_agent_provider,
        )
    except ImportError as exc:
        return GeneratedTuner(
            False,
            None,
            f"no agent provider available in this install ({exc}); "
            "generation is skipped and tuning continues without it",
        )

    try:
        # ``resolve_agent_runtime`` needs a provider name; picking one is a
        # separate step that also checks the CLI is actually installed. Passing
        # the model lets a Codex model route to Codex rather than to whichever
        # backend happens to be registered first.
        chosen = select_default_agent_provider(model)
        runtime = resolve_agent_runtime(chosen.name, model=model, timeout_sec=timeout_s)
        backend = create_registered_backend(runtime)
    except Exception as exc:  # noqa: BLE001 - provider setup must not fail tuning
        return GeneratedTuner(False, None, f"agent provider unusable: {exc!r}")

    spec = AgentRunSpec(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=_user_prompt(mandate, script_path, retry_note),
        cwd=str(work_dir),
        writable=True,
        timeout_sec=timeout_s,
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
        return GeneratedTuner(
            False,
            None,
            f"the session ended ({getattr(result, 'end_reason', '?')}) without writing {script_path.name}",
            provider,
            session,
        )
    log.info("tier3: %s authored %s", provider or "agent", script_path)
    return GeneratedTuner(True, script_path, "", provider, session)


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
