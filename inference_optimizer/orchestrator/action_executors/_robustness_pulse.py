# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Best-effort robustness "pulse" used at long-action variant boundaries.

The Coordinator only ticks robustness between actions, so a mid-grid leak /
crash / ROCm error spike goes unobserved until the whole grid finishes. This
spawns the robustness-agent runtime CLI as a one-shot subprocess between
variants (LLM RCA disabled, ~5s wall time), persisting findings via the normal
sink. Best-effort (every failure swallowed; never on the critical path), reads
``SESSION_DIR`` / ``ROBUSTNESS_AGENT_SESSION_DIR``, opt-out via
``HYPERLOOM_GRID_ROBUSTNESS_PULSE=0``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


log = logging.getLogger(__name__)


_PULSE_TIMEOUT_SEC = 8.0
_OFF_VALUES = frozenset({"0", "false", "no", "off", ""})


def _enabled() -> bool:
    # Disable inside pytest — the pulse spawns a real subprocess that bypasses
    # test mocks. Mirrors ``_run_magpie``'s guard.
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    val = os.environ.get("HYPERLOOM_GRID_ROBUSTNESS_PULSE", "1").strip().lower()
    return val not in _OFF_VALUES


def _resolve_session_dir() -> Path | None:
    """Resolve the session dir from the robustness/session env vars.

    Returns:
        Path | None: The first existing directory named by
            ``ROBUSTNESS_AGENT_SESSION_DIR`` or ``SESSION_DIR``, else ``None``.
    """
    for env_key in ("ROBUSTNESS_AGENT_SESSION_DIR", "SESSION_DIR"):
        raw = os.environ.get(env_key, "").strip()
        if raw:
            p = Path(raw)
            if p.is_dir():
                return p
    return None


def _build_request(session_dir: Path, *, tick_index: int) -> dict[str, Any]:
    """Build the one-shot robustness-agent tick request payload.

    Args:
        session_dir (Path): The session directory the tick should run against.
        tick_index (int): The monotonic tick index for this pulse.

    Returns:
        dict[str, Any]: The ``coordinator_inbox`` request dict (LLM RCA
            disabled) to hand to the robustness-agent CLI.
    """
    session_id = session_dir.name or "default"
    return {
        "kind": "coordinator_inbox",
        "session_id": session_id,
        "raw_prompt": (
            "=== Shared session state ===\n"
            f"session_id={session_id}\n"
        ),
        "context": {"tick_index": tick_index},
        "options": {
            "session_dir": str(session_dir),
            "llm_rca_enabled": False,
        },
    }


async def pulse(*, tick_index: int = 0, timeout_s: float = _PULSE_TIMEOUT_SEC) -> bool:
    """Fire one deterministic robustness tick. Returns True on clean exit.

    Always safe to ``await`` from inside another asyncio task — failures
    are logged at DEBUG and swallowed.

    Args:
        tick_index (int): The monotonic tick index for this pulse.
        timeout_s (float): Hard wall-clock budget for the tick subprocess.

    Returns:
        bool: True on a clean (exit-0) tick; False on disabled / missing
            session / spawn failure / timeout / non-zero exit.
    """
    if not _enabled():
        return False
    session_dir = _resolve_session_dir()
    if session_dir is None:
        return False

    request = _build_request(session_dir, tick_index=tick_index)
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8",
        ) as fh:
            json.dump(request, fh)
            req_path = fh.name
    except OSError as exc:
        log.debug("robustness pulse: tempfile create failed: %r", exc)
        return False

    try:
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "robustness_agent.runtime.cli", "tick",
                "--request", req_path,
                "--out", "-",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
        except (FileNotFoundError, OSError) as exc:
            log.debug("robustness pulse: subprocess spawn failed: %r", exc)
            return False
        try:
            _, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            log.info(
                "robustness pulse timed out after %.1fs at tick=%d; killing",
                timeout_s, tick_index,
            )
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()
            return False
        if proc.returncode == 0:
            log.debug("robustness pulse tick=%d ok", tick_index)
            return True
        log.debug(
            "robustness pulse tick=%d exit=%d stderr=%r",
            tick_index, proc.returncode,
            (stderr or b"")[:400].decode("utf-8", errors="replace"),
        )
        return False
    finally:
        try:
            os.unlink(req_path)
        except OSError:
            pass


__all__ = ["pulse"]
