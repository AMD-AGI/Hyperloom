from __future__ import annotations

import json
import logging
import os
import random
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests

log = logging.getLogger("optimize-submit")

from . import records as _records

globals().update({k: v for k, v in vars(_records).items() if not k.startswith("__")})

# ── Manifest ────────────────────────────────────────────────────────────────────


def write_manifest(
    out_dir: Path,
    records: list[SubmissionRecord],
    base_url: str,
    register_workspace: str,
    submit_workspace: str,
    volume: str,
) -> None:
    """Write the submission manifest as JSON and a markdown summary table.

    Args:
        out_dir (Path): Output directory (created if absent).
        records (list[SubmissionRecord]): The submission records to serialize.
        base_url (str): SaFE API base URL recorded in the manifest.
        register_workspace (str): Workspace used for registration.
        submit_workspace (str): Workspace used for submission.
        volume (str): Storage volume name recorded in the manifest.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "api_url": base_url,
        "register_workspace": register_workspace,
        "submit_workspace": submit_workspace,
        "volume": volume,
        "records": [asdict(r) for r in records],
    }
    (out_dir / "submission_manifest.json").write_text(json.dumps(payload, indent=2))

    md = [
        "# SaFE Optimization Submission Manifest",
        f"- API: `{base_url}`",
        f"- Register workspace: `{register_workspace}`",
        f"- Submit workspace: `{submit_workspace}`",
        f"- Volume: `{volume}`",
        f"- Submitted at: {payload['submitted_at']}",
        "",
        "| Pool | Model | Category | Image | Duration | Submit | Final | CI | Phase | Task ID | Display Name | Artifacts | Note |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        # final_status is only meaningful when --wait-for-completion was on.
        final = r.final_status or ("-" if r.status == "submitted" else "")
        phase = "-" if r.final_phase is None else str(r.final_phase)
        artifacts_cell = f"{r.artifact_count} files in `{r.artifacts_dir}`" if r.artifact_count else "-"
        note_parts = []
        if r.error:
            note_parts.append(r.error)
        if r.final_message:
            note_parts.append(r.final_message)
        note = " \\| ".join(note_parts).replace("|", "\\|")[:200]
        pool_cell = "-"
        if r.pool:
            pool_id = r.pool.get("pool_id") or "-"
            pool_idx = r.pool.get("pool_index")
            batch_idx = r.pool.get("batch_index")
            batch_size = r.pool.get("batch_size")
            pool_cell = (
                f"`{pool_id}`"
                f"<br/>idx={pool_idx if pool_idx not in (None, '') else '-'}"
                f"<br/>batch={batch_idx if batch_idx not in (None, '') else '-'}/"
                f"{batch_size if batch_size not in (None, '') else '-'}"
            )
        # Image cell: tag suffix only for readability; full path is in JSON.
        image_cell = "-"
        image_full = (r.detected or {}).get("image", "") if r.detected else ""
        if image_full:
            image_cell = "`" + image_full.split("/")[-1] + "`"
        # Duration cell: rounded minutes; ms-precision is in JSON.
        duration_cell = "-"
        if r.sandbox_duration_seconds is not None:
            mins = r.sandbox_duration_seconds / 60.0
            duration_cell = f"{mins:.1f}m"
        category_cell = r.category or "-"
        ci_cell = r.ci_status or ("Succeeded" if r.final_status == "Succeeded" else "-")
        if r.ci_success and r.delivery_reason:
            ci_cell = f"{ci_cell}<br/>{r.delivery_reason}"
        md.append(
            f"| {pool_cell} | `{r.model}` | {category_cell} | {image_cell} | {duration_cell} | "
            f"{r.status} | {final or '-'} | {ci_cell} | {phase} | "
            f"`{r.task_id or '-'}` | {r.display_name or '-'} | {artifacts_cell} | {note} |"
        )
    (out_dir / "submission_manifest.md").write_text("\n".join(md) + "\n")
    log.info("manifest written to %s", out_dir)
