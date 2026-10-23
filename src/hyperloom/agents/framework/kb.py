# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Framework PR ledger: KB root resolution and the ledger reader."""

from __future__ import annotations

import json
import os
from pathlib import Path

from hyperloom.inference_optimizer.session.paths import workspace_root


# Per-framework KB partition root under ``<KB_ROOT>/framework_optimization/``.
_FRAMEWORK_OPTIMIZATION_ROOT: str = "framework_optimization"

#: Append-log filename inside the framework_optimization/ directory; stable so existing ledgers stay readable.
LESSONS_FILE: str = "lessons.jsonl"

#: The only supported override for the mutable KB root; both this module and
#: ``kb_writeback`` honour it. It reaches the process through the
#: ``INFERENCE_OPTIMIZER_`` prefix rule in the ``common/env_safety`` dotenv
#: allowlist, which is a prefix rule rather than an entry for this name.
KB_ROOT_ENV: str = "INFERENCE_OPTIMIZER_FA_KB_PATH"

#: Workspace subdirectory holding this KB; the recipe root is ``<workspace>/knowledge``.
_MUTABLE_KB_DIRNAME: str = "framework-kb"


def mutable_kb_root() -> Path:
    """Root of the KB partition this session reads and writes."""
    override = os.environ.get(KB_ROOT_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    return workspace_root() / _MUTABLE_KB_DIRNAME


def framework_optimization_root() -> Path:
    """The partition holding the lessons ledger, for reader and writer alike."""
    return mutable_kb_root() / _FRAMEWORK_OPTIMIZATION_ROOT


def read_pr_ledger() -> list[dict]:
    """Read the framework PR outcome ledger from ``lessons.jsonl``."""
    path = framework_optimization_root() / LESSONS_FILE
    if not path.is_file():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


__all__ = [
    "LESSONS_FILE",
    "framework_optimization_root",
    "read_pr_ledger",
]
