# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""When a variant that restates the round's opening stack may be skipped."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Mapping
from typing import Any

from ._grid_base import GridVariant

log = logging.getLogger(__name__)

__all__ = ["BaselineNoopPlan"]

# What a KEEP changes about the stack; the anchor figures that ride along are not the launch.
_STACK_FIELDS = ("extra_server_args", "extra_envs", "remove_args", "unset_envs", "args_mode")


class BaselineNoopPlan:
    """The variants found to restate the stack's observed launch, and when skipping them is still right.

    The observed launch is the stack the round opens on. A KEEP earlier in the
    round moves the stack, after which a variant that restated the opening stack
    can be a real change, so the verdict holds only while the stack has not moved.
    """

    def __init__(self, grid: list[GridVariant], kept: list[GridVariant], judged: list[tuple[str, str]]) -> None:
        kept_ids = {id(gv) for gv in kept}
        # The filter keeps grid order, so the variants it dropped pair with its reasons in order.
        dropped = [gv for gv in grid if id(gv) not in kept_ids]
        self._reasons = {id(gv): reason for gv, (_name, reason) in zip(dropped, judged, strict=True)}
        self.dropped_now = 0

    def _skip(self, gv: GridVariant, skipped: list[dict[str, Any]]) -> None:
        reason = self._reasons[id(gv)]
        log.info("explore: skipping baseline-noop variant %s (%s)", gv.name, reason)
        skipped.append({"name": gv.name, "reason": "baseline_noop", "detail": reason})

    def slot(self, gv: GridVariant, fp: str) -> object:
        """The key ``gv`` takes in the round's dedup of identical launches.

        A restatement takes one of its own: whether it is a duplicate depends on what has launched by its turn, which
        ``turns`` decides, so it must neither hold a later variant's slot nor lose its own to an earlier restatement
        that the unmoved stack will skip.
        """
        return (fp, id(gv)) if id(gv) in self._reasons else fp

    def drop_leading(self, runnable: list[GridVariant], skipped: list[dict[str, Any]]) -> list[GridVariant]:
        """Drop the restatements with no other variant ahead of them, which can only launch on the opening stack.

        Every other one stays in the round and is decided by ``turns`` when its turn comes.
        """
        planned: list[GridVariant] = []
        for gv in runnable:
            if not planned and id(gv) in self._reasons:
                self._skip(gv, skipped)
                self.dropped_now += 1
                continue
            planned.append(gv)
        return planned

    def turns(
        self,
        runnable: list[GridVariant],
        stack_now: Callable[[], Mapping[str, Any]],
        skipped: list[dict[str, Any]],
    ) -> Iterator[tuple[int, GridVariant]]:
        """``enumerate(runnable)``, minus each restatement whose turn comes while the stack has not moved.

        ``stack_now`` is read at each turn, after the variants before it have run: a KEEP among them moves the stack,
        and a variant that restated the opening stack is then a change to the stack it launches on, so it runs --
        unless another variant of the round launches the same fingerprint, the round's usual dedup. Only a
        restatement gives way: every other variant runs exactly as it would without this plan.
        """
        opening = stack_now()
        # Fingerprints other variants of the round launch, at whatever turn; the dedup gave each a slot of its own.
        taken = {getattr(gv, "canonical_fp", None) for gv in runnable if id(gv) not in self._reasons}
        for idx, gv in enumerate(runnable):
            fp = getattr(gv, "canonical_fp", None)
            if id(gv) not in self._reasons:
                yield idx, gv
                continue
            if not any(opening.get(field) != stack_now().get(field) for field in _STACK_FIELDS):
                self._skip(gv, skipped)
                continue
            if fp in taken:
                log.info("explore: skipping %s: another variant of this round launches the same fingerprint", gv.name)
                skipped.append({"name": gv.name, "fingerprint": fp, "reason": "round_dup"})
                continue
            log.info("explore: running %s: it restates the round's opening stack, but a KEEP has moved it", gv.name)
            taken.add(fp)
            yield idx, gv
