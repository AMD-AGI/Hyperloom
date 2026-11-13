# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Aggregate the per-process export counters a benchmark run left behind, and judge them.

Written by ``assets/host_probe/hl_aoti_evidence.py``, one file per pid. The counters
decide, never the module's source: a dead helper, a caught fallback to eager, and a
package compiled but never used all parse as an export and all benchmark fine. The last
of those is why the package on disk is not enough either.

The judgement lives here rather than beside either of its two callers -- the KEEP gate
and the lift that writes ``current_best`` -- because the two must agree on what counts as
an export. They disagreed once: the gate reverted non-exporting candidates while the
explore lift, which has no gate, accepted them.
"""

from __future__ import annotations

import glob
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    boundary: int = 0
    timed_invoked: int = 0
    timed_boundary: int = 0
    processes: int = 0

    @property
    def exported(self) -> bool:
        """Whether the timed benchmark ran the exported path, every call.

        ``invoked > 0`` alone is not enough: a module that calls the package once to
        validate it and then serves every inference from the eager model satisfies it.
        An exporting module runs the package at least once per call at the boundary, so
        the timed window's own two counts settle it. The weaker rule stands in only
        where no window was reported -- a harness that predates it, or a workload that
        does not run through one at all.
        """
        if not self.observed:
            return False
        if self.timed_boundary > 0:
            return self.timed_invoked >= self.timed_boundary
        return self.invoked > 0

    def summary(self) -> str:
        """The counts as a refusal names them."""
        return (
            f"compiled={self.compiled} loaded={self.loaded} invoked={self.invoked} "
            f"timed_invoked={self.timed_invoked}/{self.timed_boundary} across {self.processes} process(es)"
        )


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
    totals = {"compiled": 0, "loaded": 0, "invoked": 0, "boundary": 0, "timed_invoked": 0, "timed_boundary": 0}
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


def export_regression_refusal(shared_state: Any, workspace: str) -> str:
    """Why a candidate whose benchmark did not export cannot KEEP.

    An eager module is not a slower deliverable, it is not a deliverable: the operator
    declared ``aoti.enabled``, so a configuration that does not export has nothing to
    ship. It is refused on that ground alone, whatever its throughput, and whatever holds
    the slot keeps it.

    Applied from the first round, not only once an export has landed. ``current_best`` is
    what the run reports and what it ships, so letting an eager candidate occupy it means
    every figure afterwards describes a path the deployment never takes. After a landing
    it also stops the loss that would otherwise be the *expected* outcome: the first
    export is accepted on correctness alone precisely because it is usually slower than
    eager, so the next eager candidate would beat it honestly and take the slot back.

    The cost is that a KEEP is the only thing the integrate lane accumulates -- a kept
    patch stays in the tree for the next specialist, a reverted one does not -- so a model
    needing several rounds of work before it can export has no way to bank the
    intermediate steps. Observed runs landed their export on the first or second patch, so
    the risk is currently hypothetical; see the upstreaming plan's open items for the fix
    if a model turns out to need it.

    Absent counters stay asymmetric, because that is about what the evidence can prove
    rather than about policy. Before the first landing they cannot be told apart from a
    harness that was never instrumented, so they are not read as a verdict. The landing
    settles it -- that KEEP is proof the shim runs and writes -- so counters going missing
    afterwards are a harness regression and the conservative reading is to refuse.

    Args:
        workspace: The graded leg's own run directory, not the attempt's task root.
            ``read_evidence`` sums every process below the path it is given, and the task
            root holds more than the candidate: the switch-off parity leg runs the same
            patch with the switches *unset* and reports no export by design, and the root
            is keyed by task id and reused, so an earlier attempt's counters are still
            sitting there.
    """
    from hyperloom.common.optimization_scope import aoti_config

    if shared_state is None:
        return ""
    try:
        config = aoti_config()
    except ValueError:
        return ""
    if config is None or not config.enabled:
        return ""
    landed = bool(getattr(shared_state, "export_landed", False))
    tail = "an eager module is not a deliverable under aoti.enabled"
    if not workspace:
        if not landed:
            return ""
        return f"this candidate's benchmark reported no workspace, so it cannot be shown to export; {tail}"
    evidence = read_evidence(workspace)
    if not evidence.observed:
        if not landed:
            # Indistinguishable from a harness that was never instrumented, and refusing
            # on it would revert every candidate for the rest of the campaign.
            return ""
        return (
            f"no run counters under {workspace!r}: the start-up shim did not run, so this candidate "
            f"cannot be shown to export. An export has already landed, which proves the shim works, "
            f"so their absence is a harness regression rather than an unknown; {tail}"
        )
    if evidence.exported:
        return ""
    if not landed:
        return f"this candidate's benchmark did not export ({evidence.summary()}); {tail}"
    return f"the accepted configuration exports and this one does not ({evidence.summary()}); {tail}"


def gives_up_a_landed_export(shared_state: Any, workspace: str) -> str:
    """Why this configuration cannot take ``current_best`` from one that exports.

    The same judgement as the KEEP gate, narrowed to after the first landing. Before it
    nothing exports yet, and the lanes that run then -- the baseline, enablement, the
    patch work the gate admits -- have to be free to move ``current_best`` or the campaign
    cannot start. The integrate lane keeps the wider rule because it is the lane that is
    supposed to be producing the export.
    """
    if not getattr(shared_state, "export_landed", False):
        return ""
    return export_regression_refusal(shared_state, workspace)
