# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Archive enablement round artifacts under ``reports/enablement/<task_id>/``.

``reports/`` is retained by the session-archive collector whereas
``runs/specialist/<tid>/patches/`` and ``runs/integrate_patch/<tid>/`` are
excluded wholesale.  A single call to :func:`snapshot_round` copies the
deliverables that matter into an archived location so they survive upload.

Everything here is best-effort: a failure to copy one file must never propagate
to the caller or interrupt the enablement rearm loop.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from hyperloom.common.io import atomic_write_text

log = logging.getLogger(__name__)

_FILE_SIZE_LIMIT = 2 * 1024 * 1024   # 2 MB per file
_ROUND_SIZE_LIMIT = 8 * 1024 * 1024  # 8 MB total per round


def snapshot_round(
    session_dir: str | Path,
    res: dict[str, Any],
    *,
    specialist_workspace: str | Path | None = None,
) -> None:
    """Copy enablement round artifacts into ``reports/enablement/<task_id>/``.

    Writes a ``round.json`` summary plus patches, ``specialist_done.json``,
    ``prompt.md``, and the accepted launch config yaml if present.  Each file
    is capped at :data:`_FILE_SIZE_LIMIT`; the total round write is capped at
    :data:`_ROUND_SIZE_LIMIT` — oversized content is silently truncated and
    ``round.json`` records ``"truncated": true``.

    Args:
        session_dir: Session root directory.
        res: The result dict returned by the ``integrate_patch`` executor.
        specialist_workspace: Explicit workspace path of the enablement
            specialist.  When ``None`` the path is inferred from
            ``res["specialist_task_id"]`` and the standard ``runs/`` layout.
    """
    if not session_dir or not isinstance(res, dict):
        return

    from hyperloom.inference_optimizer.session.session_paths import (
        enablement_round_dir,
        runs_dir,
    )

    task_id = str(res.get("specialist_task_id") or "").strip()
    round_dir = enablement_round_dir(Path(session_dir), task_id or "unknown")

    try:
        round_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        log.debug("enablement_artifacts: cannot create %s", round_dir, exc_info=True)
        return

    bytes_written = 0
    truncated = False

    def _write_json(name: str, payload: Any) -> None:
        nonlocal bytes_written, truncated
        if bytes_written >= _ROUND_SIZE_LIMIT:
            truncated = True
            return
        try:
            import json as _json
            text = _json.dumps(payload, indent=2, ensure_ascii=False, default=str)
            if len(text.encode()) > _FILE_SIZE_LIMIT:
                text = text[: _FILE_SIZE_LIMIT] + "\n"
                truncated = True
            atomic_write_text(round_dir / name, text, make_parents=False)
            bytes_written += len(text.encode())
        except Exception:
            log.debug("enablement_artifacts: failed writing %s", name, exc_info=True)

    def _copy_file(src: Path, dest_name: str, sub: str = "") -> None:
        nonlocal bytes_written, truncated
        if bytes_written >= _ROUND_SIZE_LIMIT:
            truncated = True
            return
        try:
            if not src.is_file():
                return
            raw = src.read_bytes()
            if len(raw) > _FILE_SIZE_LIMIT:
                raw = raw[: _FILE_SIZE_LIMIT]
                truncated = True
            dest_dir = round_dir / sub if sub else round_dir
            dest_dir.mkdir(parents=True, exist_ok=True)
            (dest_dir / dest_name).write_bytes(raw)
            bytes_written += len(raw)
        except Exception:
            log.debug("enablement_artifacts: failed copying %s", src, exc_info=True)

    # --- round.json ---
    launch_log = str(res.get("enablement_launch_log") or "")
    round_summary = {
        "status": res.get("status"),
        "specialist_task_id": task_id,
        "patches_applied": res.get("patches_applied") or [],
        "config_changes_applied": res.get("config_changes_applied") or {},
        "extra_envs_applied": res.get("extra_envs_applied") or {},
        "extra_server_args_applied": res.get("extra_server_args_applied") or "",
        "setup_commands_applied": res.get("setup_commands_applied") or [],
        "after_signature": res.get("after_signature") or {},
        "enablement_accepted_config_path": res.get("enablement_accepted_config_path") or "",
        "enablement_effective_config": res.get("enablement_effective_config") or {},
        "launch_log_excerpt": launch_log[:1200] if launch_log else "",
    }
    _write_json("round.json", round_summary)

    # --- patches ---
    for patch_path_str in (res.get("patches_applied") or []):
        patch_path = Path(str(patch_path_str))
        if patch_path.is_file():
            _copy_file(patch_path, patch_path.name, sub="patches")

    # --- specialist workspace files ---
    ws: Path | None = None
    if specialist_workspace:
        ws = Path(specialist_workspace)
    elif task_id and session_dir:
        try:
            candidate = Path(runs_dir(Path(session_dir), "specialist", task_id))
            if candidate.is_dir():
                ws = candidate
        except ValueError:
            pass

    if ws and ws.is_dir():
        _copy_file(ws / "specialist_done.json", "specialist_done.json")
        _copy_file(ws / "prompt.md", "prompt.md")

        already = {Path(p).name for p in (res.get("patches_applied") or [])}
        for base in (ws, ws / "worktree"):
            pd = base / "patches"
            if not pd.is_dir():
                continue
            for ext in ("*.patch", "*.diff"):
                for pf in sorted(pd.glob(ext)):
                    if pf.name not in already:
                        _copy_file(pf, pf.name, sub="patches")
                        already.add(pf.name)

    # --- accepted launch config ---
    config_path_str = str(res.get("enablement_accepted_config_path") or "").strip()
    if config_path_str:
        _copy_file(Path(config_path_str), "launch_config.yaml")

    # --- mark truncation in round.json if needed ---
    if truncated:
        try:
            import json as _json
            existing = _json.loads((round_dir / "round.json").read_text(encoding="utf-8"))
            existing["truncated"] = True
            atomic_write_text(
                round_dir / "round.json",
                _json.dumps(existing, indent=2, ensure_ascii=False, default=str) + "\n",
            )
        except Exception:
            log.debug("enablement_artifacts: could not mark truncation", exc_info=True)

    log.debug(
        "enablement_artifacts: snapshot wrote %d bytes to %s (truncated=%s)",
        bytes_written,
        round_dir,
        truncated,
    )


__all__ = ["snapshot_round"]
