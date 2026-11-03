# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read what a benchmark run recorded about its ahead-of-time export.

The counters are written by ``assets/host_probe/hl_aoti_evidence.py`` inside the
benchmark process, one file per pid. This is the orchestrator side: aggregate them
and answer the one question the export gate asks.

Reading the module's source cannot answer it. The AOTI markers appear in a helper
nobody calls, in a branch under ``if False``, and in the ``except`` arm of a
fallback to eager; all three parse identically to a working export, and all three
produce a valid benchmark measurement. Measured on gfx1151, the four shapes a
candidate can take:

===================  ========  ======  =======
shape                compiled  loaded  invoked
===================  ========  ======  =======
real export                 1       1        1
dead code                   0       0        0
caught fallback             0       0        0
compiled, not used          1       1        0
===================  ========  ======  =======

Only ``invoked`` separates the first from the rest, which is why it alone decides.
The last row is why the package on disk is not enough either.
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

    Searched recursively: the benchmark's working directory is nested under the
    workspace and its depth is the harness's business, not ours. Mirrors how
    ``_accuracy_gate`` finds ``inferencex_result.json``.

    A process that reported without instrumenting is dropped rather than counted as
    a zero: it means the shim loaded but never wrapped torch, which is evidence of
    nothing. ``observed`` is false when no process reported at all -- the shim never
    ran -- and the caller must not read that as "did not export".
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
