# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Cumulative gain measured against the baseline, and the watermark it moves."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Mapping

from hyperloom.common.gain_math import gain_pct_or_zero
from hyperloom.common.perf_metric import with_output_per_gpu
from ..state.shared_state import resolve_graded_comparison
from ._current_best_lift import _graded_source

log = logging.getLogger(__name__)


class ValidationMixin:
    """Validating cumulative gain and keeping the watermark honest."""

    def validate(
        self,
        new_tput: float,
        measurement: Mapping[str, Any],
        *,
        source: str = "writeback",
        measurement_basis: str = "e2e_rebench",
        ts: str | None = None,
    ) -> bool:
        """Validate cumulative gain against baseline, updating the watermark.

        Args:
            new_tput: The newly measured output throughput.
            measurement: The measurement the promotion came from. Candidate and
                baseline are read off it as a pair, so a total-graded session
                cannot divide an output numerator by a total denominator.
            source: Which promotion path produced this figure, recorded so the
                breakdown can name it.
            measurement_basis: How the reading was obtained — ``e2e_rebench``
                for a full-stack revalidation, ``e2e_decision_round`` for the
                round an explore variant was graded on.
            ts: Author-time stamp the caller already minted for this
                promotion; defaults to now.

        Returns:
            Whether a comparable measurement updated the validation watermark.
        """
        if self.shared_state.baseline_tput <= 0:
            return False
        graded_source = with_output_per_gpu(_graded_source(measurement, new_tput), self.shared_state.tp)
        graded = resolve_graded_comparison(self.shared_state, graded_source, against_baseline=True)
        if not graded.comparable:
            log.info("cumulative gain held: measurement not comparable (%s)", graded.degrade_reason)
            return False
        validated_gain = gain_pct_or_zero(graded.candidate, graded.reference)
        ts = str(ts or datetime.now(timezone.utc).isoformat())
        self.shared_state.cumulative_gain_validated = float(validated_gain)
        self.shared_state.cumulative_gain_validated_ts = ts
        self.shared_state.cumulative_gain_validated_stack_len = len(self.shared_state.optimization_stack)
        self.shared_state.validated_recipe_generation = int(self.shared_state.working_recipe_generation or 0)
        # The breakdown's own total is the sum of its ledger, so without this
        # record there is nothing for it to disagree with.
        from hyperloom.inference_optimizer.breakdown.recorder import stack_event

        stack_event.record_validation(
            stack_event.StackValidation(
                stack_len=self.shared_state.cumulative_gain_validated_stack_len,
                graded=graded,
                validated_gain_pct=float(validated_gain),
                # The figures grading actually read, not the raw measurement: the caller's resolved output
                # throughput is stamped into it, so the axes recorded here are the ones the verdict was reached on.
                measurement=graded_source,
                source=source,
                measurement_basis=measurement_basis,
                ts=ts,
            )
        )
        return True

    async def _validate_and_watermark(
        self,
        new_tput: float,
        measurement: Mapping[str, Any],
        *,
        source: str = "writeback",
        measurement_basis: str = "e2e_rebench",
        watermark_reason: str,
    ) -> bool:
        """Validate cumulative gain and fire a watermark roofline when it updates."""
        if self.validate(new_tput, measurement, source=source, measurement_basis=measurement_basis):
            await self._coord.phase_kernel.maybe_enqueue_watermark_roofline(reason=watermark_reason)
            return True
        return False

    async def _compensate_validation_watermark(self, report: dict[str, Any]) -> None:
        state = self.shared_state
        if not state.optimization_stack_has_unvalidated_keeps():
            return
        stack = [e for e in (state.optimization_stack or []) if isinstance(e, dict)]
        generation = int(state.working_recipe_generation or 0)
        report["warnings"].append(
            {
                "kind": "resume_unvalidated_keeps",
                "validated_stack_len": int(state.cumulative_gain_validated_stack_len or 0),
                "stack_len": len(stack),
                "working_recipe_generation": generation,
                "validated_recipe_generation": state.validated_recipe_generation,
            }
        )
        fix = await self.enqueue_internal_stack_rebench(
            reason="resume_unvalidated_keeps",
            idempotency_key=f"resume-stack-revalidate-g{generation}",
        )
        report["fixes"].append({"kind": "queued_resume_stack_rebench", **fix})
