"""Persona distillation — DESIGN §5.3.

Marathon mode only. Personas accumulate notes from each agent over hours;
once they cross 8K tokens (or every 4h, whichever first) we ask the
agent's backend to self-distill into a leaner narrative.

STATUS (v0.7):
    Pure-Python implementation. ``distill_persona`` invokes
    ``backend.run`` only when the backend is real; for offline / mock
    runs the caller may pass ``backend=None`` and we will perform a
    deterministic *truncation* distillation that keeps the most recent
    ``KEEP_TAIL_TOKENS`` tokens — good enough for tests + dry-runs.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from .execution_mode import ExecutionMode
from .intent_parser import Intent, IntentType


__all__ = [
    "HARD_TOKEN_LIMIT",
    "SOFT_TIME_HOURS",
    "KEEP_TAIL_TOKENS",
    "estimate_tokens",
    "should_distill",
    "distill_persona",
    "archive_old_persona",
]


HARD_TOKEN_LIMIT: int = 8_000
SOFT_TIME_HOURS: float = 4.0
KEEP_TAIL_TOKENS: int = 4_000


def estimate_tokens(text: str) -> int:
    """Rough char/4 estimate; deterministic and dependency-free.

    Switch to ``tiktoken`` once we have a real runtime budget for it.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def _coerce_mode(mode: ExecutionMode | str) -> ExecutionMode:
    if isinstance(mode, ExecutionMode):
        return mode
    try:
        return ExecutionMode(str(mode))
    except ValueError:
        return ExecutionMode.QUICK_PARAM_SWEEP


def should_distill(
    persona_path: Path,
    mode: ExecutionMode | str,
    *,
    last_distill_ts: float | None = None,
    keep_just_happened: bool = False,
    now: float | None = None,
) -> bool:
    """True when the persona file should be re-distilled.

    Triggers (DESIGN §5.3 / §14):

        * marathon only — quick / guided always return False (ADR-22)
        * file ``> HARD_TOKEN_LIMIT`` tokens
        * ``last_distill_ts`` is None or older than ``SOFT_TIME_HOURS``
        * ``keep_just_happened`` (post-KEEP review forces a fresh distill)
    """
    if _coerce_mode(mode) is not ExecutionMode.MARATHON_MULTI_AGENT:
        return False
    persona_path = Path(persona_path)
    if not persona_path.is_file():
        return False
    try:
        body = persona_path.read_text(encoding="utf-8")
    except OSError:
        return False
    if estimate_tokens(body) >= HARD_TOKEN_LIMIT:
        return True
    now_ts = float(now) if now is not None else time.time()
    if last_distill_ts is None:
        return True
    if now_ts - float(last_distill_ts) >= SOFT_TIME_HOURS * 3600.0:
        return True
    if keep_just_happened:
        return True
    return False


def archive_old_persona(
    agent_name: str, raw_text: str, archive_dir: Path
) -> Path:
    """Save ``personas/archive/<agent>-<ts>.md`` before overwriting the live file."""
    archive_dir = Path(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    out = archive_dir / f"{agent_name}-{ts}.md"
    out.write_text(raw_text, encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Distillation
# ---------------------------------------------------------------------------
def _truncate_to_tail(text: str, max_tokens: int) -> str:
    """Deterministic fallback distillation: drop the head, keep the tail."""
    if estimate_tokens(text) <= max_tokens:
        return text
    target_chars = max_tokens * 4
    return "<!-- distilled (truncate-to-tail) -->\n" + text[-target_chars:]


def distill_persona(
    agent_name: str,
    backend: Any,
    *,
    persona_path: Path,
    archive_dir: Path | None = None,
) -> str:
    """Re-write ``persona_path`` with a leaner narrative.

    If ``backend`` is ``None`` we do a deterministic tail-truncation. If
    ``backend`` exposes a ``run`` coroutine that returns intents, we feed
    the persona body in as a prompt and look for an ``UPDATE_PERSONA``
    intent to pick up the new body. Anything goes wrong → tail truncation.
    """
    persona_path = Path(persona_path)
    if not persona_path.is_file():
        return ""
    body = persona_path.read_text(encoding="utf-8")

    if archive_dir is not None:
        archive_old_persona(agent_name, body, archive_dir)

    new_body: str | None = None
    if backend is not None and hasattr(backend, "run"):
        try:
            new_body = _ask_backend_to_distill(backend, agent_name, body)
        except Exception:  # noqa: BLE001 — fallback to truncation
            new_body = None

    if not new_body:
        new_body = _truncate_to_tail(body, KEEP_TAIL_TOKENS)

    persona_path.write_text(new_body, encoding="utf-8")
    return new_body


def _ask_backend_to_distill(backend: Any, agent_name: str, body: str) -> str | None:
    prompt = (
        f"# Persona distillation for {agent_name}\n"
        f"## Existing persona body\n{body}\n\n"
        "## Task\n"
        "Re-write this persona keeping only the actionable lessons / "
        "preferences / failure modes that future you will benefit from "
        "remembering. Emit the new body via an `update_persona` intent."
    )
    coro = backend.run(prompt, agent_name=agent_name, allowed_tools=("emit_intent",))
    intents = asyncio.run(coro) if asyncio.iscoroutine(coro) else coro
    if not intents:
        return None
    for intent in intents:
        if isinstance(intent, Intent) and intent.type == IntentType.UPDATE_PERSONA:
            new_body = str(intent.payload.get("body_md", ""))
            if new_body.strip():
                return new_body
    return None
