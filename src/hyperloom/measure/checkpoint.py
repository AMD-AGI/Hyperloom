# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The AgentX checkpoint a test's measured round is held to, as the sheet carries it, and the ruling it leaves.

The caller decides whether a test is judged and in which mode; the sandbox only follows the sheet. The local backend
judges with the single-lineage :class:`~hyperloom.orchestrator.actions.executors.AgentxCheckpoint`, so both flows
compute the same median against the same allowed drop and write the same ``agentx_checkpoint.json`` into the round's
slot.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hyperloom.common.agentx_early_stop import EarlyStopMode
from hyperloom.contracts.measurement import AgentxCheckpointRuling, MeasurementError
from hyperloom.measure.docs import DocumentError, Reader

if TYPE_CHECKING:
    from hyperloom.orchestrator.actions.executors import AgentxCheckpoint

__all__ = ["CheckpointPlan", "read_ruling"]

#: A sheet never carries ``off``: a test that is not judged carries no plan.
_SHEET_MODES = (EarlyStopMode.SHADOW.value, EarlyStopMode.ENFORCE.value)


@dataclass(frozen=True)
class CheckpointPlan:
    """The reference's checkpoint median the measured round is compared with, and whether trailing it stops the round
    (``enforce``) or is only recorded (``shadow``)."""

    reference_p50: float
    mode: EarlyStopMode

    def __post_init__(self) -> None:
        if self.mode.value not in _SHEET_MODES:
            raise DocumentError(f"checkpoint: mode must be one of {', '.join(_SHEET_MODES)}, not {self.mode.value}")
        if isinstance(self.reference_p50, bool) or not math.isfinite(self.reference_p50) or self.reference_p50 <= 0:
            raise DocumentError("checkpoint: reference_p50 must be a positive number")

    def on(self, measured: bool) -> AgentxCheckpoint | None:
        """The checkpoint a round runs under: a test's measured round is judged, its warmup and evaluation are not."""
        from hyperloom.orchestrator.actions.executors import AgentxCheckpoint  # not at the top: see read_ruling

        return AgentxCheckpoint(self.reference_p50, enforce=self.mode is EarlyStopMode.ENFORCE) if measured else None

    def to_document(self) -> dict[str, Any]:
        return {"reference_p50": self.reference_p50, "mode": self.mode.value}

    @classmethod
    def parse(cls, value: Any, path: str) -> CheckpointPlan:
        r = Reader(value, path)
        plan = cls(
            reference_p50=r.number("reference_p50") or 0.0,
            mode=EarlyStopMode(r.choice("mode", _SHEET_MODES)),
        )
        r.done()
        return plan


def read_ruling(output_dir: Path) -> AgentxCheckpointRuling | None:
    """The ruling the checkpoint wrote into the measured round's slot below ``output_dir``; ``None`` when it ruled on
    none (no plan, or the window never closed).

    Raises:
        ValueError: The test left more than one ruling, or one that is not a ruling.
    """
    # Here, not at the top: the sheet's readers, which the scheduler imports, must not load the single-lineage flow.
    from hyperloom.orchestrator.actions.executors import CHECKPOINT_ARTIFACT_NAME

    found = sorted(Path(output_dir).glob(f"*/{CHECKPOINT_ARTIFACT_NAME}"))
    if not found:
        return None
    if len(found) > 1:
        raise ValueError(f"{output_dir} holds {len(found)} checkpoint rulings; a test's one measured round rules once")
    try:
        return AgentxCheckpointRuling.from_document(json.loads(found[0].read_text(encoding="utf-8")))
    except MeasurementError as exc:
        raise ValueError(f"{found[0]}: {exc}") from exc
