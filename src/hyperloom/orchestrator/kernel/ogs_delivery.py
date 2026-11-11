# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Verify that a matmul_ogs tuned row reached the measured serving process."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def verify_ogs_apply(logs: list[Path], table_path: str) -> dict[str, Any]:
    """Block a KEEP unless the latest integrate server logged a matching row receipt."""
    table = Path(table_path)
    if not table.is_file():
        return {"verdict": "artifact_missing", "blocks_keep": True, "detail": "matmul_ogs flag table is absent"}
    try:
        digest = hashlib.sha256(table.read_bytes()).hexdigest()
        latest = logs[-1].read_text(encoding="utf-8", errors="replace") if logs else ""
    except OSError as exc:
        return {"verdict": "receipt_unreadable", "blocks_keep": True, "detail": str(exc)}
    marker = f"HL_MATMUL_OGS_FLAGS_APPLIED sha256={digest}"
    if marker in latest:
        return {"verdict": "applied", "blocks_keep": False, "detail": "matching tuned row in latest server log"}
    return {"verdict": "no_serving_receipt", "blocks_keep": True, "detail": "latest server did not log a tuned row"}
