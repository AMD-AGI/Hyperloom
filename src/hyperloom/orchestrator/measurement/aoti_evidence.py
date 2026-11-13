# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Aggregate the per-process export counters a benchmark run left behind.

Written by ``assets/host_probe/hl_aoti_evidence.py``, one file per pid. Only
``invoked`` decides: a dead helper, a caught fallback to eager, and a package
compiled but never used all parse as an export and all benchmark fine. The last of
those is why the package on disk is not enough either.
"""

from __future__ import annotations

import glob
import json
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

EVIDENCE_DIR_NAME = "aoti_evidence"
EVIDENCE_DIR_ENV = "HYPERLOOM_AOTI_EVIDENCE_DIR"


@dataclass(frozen=True)
class AotiEvidence:
    """What a run's processes recorded. ``observed`` is false when none reported."""

    observed: bool
    compiled: int = 0
    loaded: int = 0
    invoked: int = 0
    processes: int = 0

    @property
    def exported(self) -> bool:
        """Whether the exported path actually ran during the benchmark."""
        return self.observed and self.invoked > 0


def evidence_dir(workspace: Path | str) -> Path:
    """Where a run under ``workspace`` writes its counters."""
    return Path(workspace) / EVIDENCE_DIR_NAME


def read_evidence(workspace: Path | str) -> AotiEvidence:
    """Aggregate the per-process counters under ``workspace``.

    Searched recursively, as ``_accuracy_gate`` does for ``inferencex_result.json``:
    the benchmark's working directory nests under the workspace at a depth that is
    the harness's business.

    A process that reported without instrumenting is dropped rather than counted as a
    zero -- it wrapped nothing, so its zeros are evidence of nothing. ``observed``
    false means no process reported, which the caller must not read as "did not
    export".
    """
    pattern = str(Path(workspace) / "**" / EVIDENCE_DIR_NAME / "*.json")
    totals = {"compiled": 0, "loaded": 0, "invoked": 0}
    processes = 0
    for path in glob.glob(pattern, recursive=True):
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            log.warning("aoti evidence: unreadable %s", path)
            continue
        if not isinstance(payload, dict) or not payload.get("instrumented"):
            continue
        processes += 1
        for key in totals:
            value = payload.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                totals[key] += value
    return AotiEvidence(observed=processes > 0, processes=processes, **totals)
