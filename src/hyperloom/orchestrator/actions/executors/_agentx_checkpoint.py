# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The AgentX checkpoint: median interactivity over the first half hour of profiling, and the early stop it rules on.

Both sides of the comparison are read the same way from aiperf's per-request export: the reference when its round
finished, the candidate while its round is still running.
"""

from __future__ import annotations

import logging
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from hyperloom_common.coerce import optional_positive_float
from hyperloom_common.io import atomic_write_json
from hyperloom_common.jsonio import read_json
from hyperloom.common.agentx_early_stop import EarlyStopMode, early_stop_mode
from hyperloom.common.outside_facts import AIPERF_PROFILING_PHASE
from hyperloom.common.perf_metric import (
    AGENTX_CHECKPOINT_MAX_DROP_PCT,
    AGENTX_CHECKPOINT_SEC,
    GRADED_INTVTY_P50_CHECKPOINT,
)

from inference_suite.telemetry.agentx_timeline import RequestRecord, find_profile_export, parse_profile_export
from ._subprocess_kill import AgentxCheckpointMissed

log = logging.getLogger(__name__)

__all__ = [
    "CHECKPOINT_ARTIFACT_NAME",
    "AgentxCheckpoint",
    "CheckpointJudge",
    "SuiteRoundCheckpoint",
    "checkpoint_p50",
    "checkpoint_summary",
    "read_checkpoint",
    "round_checkpoint",
    "workspace_checkpoint_p50",
]

#: Written into the round's slot when the checkpoint rules.
CHECKPOINT_ARTIFACT_NAME = "agentx_checkpoint.json"

# aiperf flushes its export about once a second, so a window still open on one read has closed on a later one.
_REREAD_SEC = 30.0


def checkpoint_p50(records: Iterable[RequestRecord], at_sec: float = AGENTX_CHECKPOINT_SEC) -> float | None:
    """Median e2e-normalised interactivity of profiling requests done ``at_sec`` in; None until that window closed."""
    profiling = [r for r in records if r.phase == AIPERF_PROFILING_PHASE]
    if not profiling:
        return None
    cutoff = min(r.start for r in profiling) + at_sec
    if max(r.end for r in profiling) < cutoff:
        return None
    sec_per_token = [
        r.latency / r.output_tokens for r in profiling if r.ok and r.end <= cutoff and r.latency and r.output_tokens
    ]
    return 1.0 / statistics.median(sec_per_token) if sec_per_token else None


def workspace_checkpoint_p50(workspace: Path | str | None) -> float | None:
    """The checkpoint median of the round whose aiperf export is under ``workspace``."""
    path = find_profile_export(workspace) if workspace else None
    return checkpoint_p50(parse_profile_export(path)) if path else None


@dataclass(frozen=True)
class AgentxCheckpoint:
    """The reference's checkpoint median a round is compared with, and whether trailing it stops the round."""

    reference_p50: float
    enforce: bool = False

    def gain_pct(self, candidate_p50: float) -> float:
        return (candidate_p50 / self.reference_p50 - 1.0) * 100.0


@dataclass(frozen=True)
class SuiteRoundCheckpoint:
    """The checkpoint of an inference_suite round: it rules into the grid slot and reads aiperf's export from the run_dir.

    The round reports no phase to its caller, so the judge starts counting when the round starts: profiling opens no
    earlier, and the export is re-read until it shows the window closed.
    """

    checkpoint: AgentxCheckpoint
    slot: Path
    run_dir: Path

    def judge(self) -> CheckpointJudge:
        """The judge for this round, counting from now."""
        judge = CheckpointJudge(self.checkpoint, self.slot, export_root=self.run_dir)
        judge.note_phase("measured", time.monotonic())
        return judge

    def clear(self) -> None:
        """Drop a ruling an earlier round left in the slot, so the slot's ruling is this round's alone."""
        (self.slot / CHECKPOINT_ARTIFACT_NAME).unlink(missing_ok=True)


def round_checkpoint(reference: Mapping[str, Any] | None) -> AgentxCheckpoint | None:
    """The checkpoint for a round graded against ``reference``; None when it is off or the reference has none."""
    mode = early_stop_mode(os.environ, EarlyStopMode.ENFORCE)
    reference_p50 = optional_positive_float((reference or {}).get(GRADED_INTVTY_P50_CHECKPOINT))
    if mode is EarlyStopMode.OFF or reference_p50 is None:
        return None
    return AgentxCheckpoint(reference_p50, enforce=mode is EarlyStopMode.ENFORCE)


def checkpoint_summary(ruling: Mapping[str, Any]) -> str:
    """One line naming both medians and the drop that stops a round."""
    if not ruling:
        return "trailed the reference at the AgentX checkpoint"
    return (
        f"intvty p50 at {ruling['at_sec']:.0f}s {ruling['reference_p50']:.1f}->{ruling['candidate_p50']:.1f} "
        f"({ruling['gain_pct']:+.2f}%; stops below -{ruling['max_drop_pct']:.1f}%)"
    )


def read_checkpoint(slot: Path) -> dict[str, Any] | None:
    """The ruling a round's checkpoint wrote into ``slot``, if it ruled."""
    return read_json(slot / CHECKPOINT_ARTIFACT_NAME, None, require_dict=True)


class CheckpointJudge:
    """Round recorder that rules once at the checkpoint and reaps a trailing round when the checkpoint enforces."""

    def __init__(self, checkpoint: AgentxCheckpoint, slot: Path, *, export_root: Path | None = None) -> None:
        self._checkpoint = checkpoint
        self._slot = slot
        self._export_root = slot if export_root is None else export_root
        self._due: float | None = None
        self._ruled = False

    def note_phase(self, phase: str, mono: float) -> None:
        if phase == "measured" and self._due is None:
            self._due = mono + AGENTX_CHECKPOINT_SEC

    def tick(self, mono: float) -> None:
        if self._ruled or self._due is None or mono < self._due:
            return
        candidate = workspace_checkpoint_p50(self._export_root)
        if candidate is None:
            self._due = mono + _REREAD_SEC
            return
        self._ruled = True
        gain = self._checkpoint.gain_pct(candidate)
        trails = gain < -AGENTX_CHECKPOINT_MAX_DROP_PCT
        ruling = {
            "at_sec": AGENTX_CHECKPOINT_SEC,
            "candidate_p50": candidate,
            "reference_p50": self._checkpoint.reference_p50,
            "gain_pct": gain,
            "max_drop_pct": AGENTX_CHECKPOINT_MAX_DROP_PCT,
            "trails": trails,
            "stopped": trails and self._checkpoint.enforce,
        }
        try:
            atomic_write_json(self._slot / CHECKPOINT_ARTIFACT_NAME, ruling)
        except OSError as exc:
            log.warning("agentx_checkpoint: could not record the ruling in %s (%s)", self._slot, exc)
        log.info(
            "agentx_checkpoint: %s%s", checkpoint_summary(ruling), "; stopping the round" if ruling["stopped"] else ""
        )
        if ruling["stopped"]:
            raise AgentxCheckpointMissed(checkpoint_summary(ruling))

    def close(self, *, aborted: bool = False) -> None:
        """Nothing to flush: the ruling is written when it is made."""
