"""Append KEEP lessons to the framework_optimization KB partition.

PR-I write path. Counterpart of :mod:`kb_priors`. Invoked from
:mod:`framework_request_handlers.framework_integrate_handler` only
when the verdict is ``KEEP`` -- REVERT / NEEDS_REVIEW outcomes never
write to KB (design §4.7 / §10.1 contributing rule).

Atomic-append safety: writes to a tmp file under the partition then
``os.replace`` onto ``empirical_kb.md``. Concurrent writers across
sessions are rare (sessions are single-user per Hyperloom contract)
but the race-safe write keeps the file uncorrupted in adversarial
ordering.
"""

from __future__ import annotations

import logging
import os
import secrets
import time
from pathlib import Path
from typing import Any

from .kb_priors import _LESSONS_FILE, _PARTITION_NAME, resolve_kb_root


log = logging.getLogger(__name__)


def append_keep_lesson(
    *,
    framework: str,
    patch_id: str,
    summary: str,
    rationale: str,
    gain_pct: float,
    session_id: str = "",
    kb_root: Path | None = None,
) -> Path | None:
    """Append a KEEP lesson to ``empirical_kb.md`` and return its path.

    Returns ``None`` when no KB root is configured (skipping write,
    the integrate handler continues normally). The block format mirrors
    the parser in :mod:`kb_priors._parse_lessons_file`.
    """
    root = kb_root or resolve_kb_root()
    if root is None:
        log.debug(
            "append_keep_lesson: no KB root configured; skipping "
            "(framework=%s patch_id=%s)", framework, patch_id,
        )
        return None
    partition = root / _PARTITION_NAME
    partition.mkdir(parents=True, exist_ok=True)
    target = partition / _LESSONS_FILE

    fw_lower = (framework or "").strip().lower() or "unknown"
    timestamp = time.strftime("%Y%m%d", time.gmtime())
    entry_id = f"fw-keep-{timestamp}-{secrets.token_hex(4)}"
    block_lines = [
        f"# {entry_id}  KEEP: {fw_lower} {summary.strip()}",
        f"Framework: {fw_lower}",
        f"Source: {session_id or '(no-session-id)'}",
        f"Patch: {patch_id}",
        f"Gain: {gain_pct:.2f}%",
        "",
        rationale.strip(),
        "",
    ]
    block = "\n".join(block_lines) + "\n"

    # Atomic-ish append: read current content (if any) -> write the
    # combined buffer to a temp file -> os.replace onto the target.
    tmp = target.with_suffix(target.suffix + f".tmp.{os.getpid()}.{secrets.token_hex(4)}")
    try:
        existing = target.read_text(encoding="utf-8") if target.is_file() else ""
        tmp.write_text(existing + block, encoding="utf-8")
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    log.info(
        "append_keep_lesson: %s -> %s (gain=%.2f%%)",
        entry_id, target, gain_pct,
    )
    return target


__all__ = ["append_keep_lesson"]
