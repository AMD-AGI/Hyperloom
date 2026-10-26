# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Merge stacking: once the search stalls, measure two archived rejected gains applied together."""

from __future__ import annotations

from typing import NamedTuple

from kernelforge.loop.merge_candidates import (
    MERGE_ATTEMPT_STALL_THRESHOLD,
    MERGE_PRECEDENCE_STREAK_LIMIT,
    MergeCandidate,
    attempted_pairs,
    eligible_candidates,
    select_merge_pair,
)
from kernelforge.loop.run_state import make_event


class StagedMerge(NamedTuple):
    """A stacked pair staged for measurement, and the diff that stacks it."""

    pair: tuple | None
    diff: str


class MergeStackingMixin:
    """Select, stage and report stacked attempts for the iteration loop.

    The host supplies ``ic``, ``run_state``, ``state_store``, ``archive``, ``_lane_queue``,
    ``_merge_precedence_streak``, ``_declined_merge_pairs``, ``_scored_incumbent_case_times``,
    ``_working_tree_diff``, ``_git_apply_patch`` and ``_git_discard_worktree``.
    """

    # The one obstacle a later iteration clears on its own, and so the one the caller must not hold against the pair:
    # a tree carrying work is this iteration's accident, not a fact about two archived diffs.
    TREE_ALREADY_DIRTY_OBSTACLE = "the working tree already carried uncommitted work"

    def _select_merge_attempt(
        self,
    ) -> tuple[MergeCandidate, MergeCandidate] | None:
        """Two rejected gains worth measuring stacked, once single patches stall."""
        if not self.ic.merge_stacking:
            return None
        if self.run_state.stall.unresolved_stall_iters < MERGE_ATTEMPT_STALL_THRESHOLD:
            return None
        incumbent_case_times = self._scored_incumbent_case_times()
        if not incumbent_case_times:
            return None
        index = self.archive.load_index()
        metas = []
        for row in index:
            meta = self.archive.load_meta(int(row.get("iter") or 0))
            if meta:
                metas.append(meta)
        return select_merge_pair(
            eligible_candidates(metas, incumbent_case_times),
            already_attempted=(
                attempted_pairs([str(row.get("plan") or "") for row in index]) | frozenset(self._declined_merge_pairs)
            ),
        )

    def _merge_attempt_refusal(self) -> str:
        """Why a selected pair may not be measured this iteration, or \"\"."""
        if self._merge_precedence_streak >= MERGE_PRECEDENCE_STREAK_LIMIT:
            return (
                f"{self._merge_precedence_streak} stacked iterations have run "
                "back to back without the queue being reached"
            )
        return ""

    def _stage_merge_attempt(
        self,
        pair: tuple[MergeCandidate, MergeCandidate] | None,
    ) -> tuple[str, str]:
        """Put both candidates' diffs in the tree; the diff, or why there is none."""
        if pair is None:
            return "", ""
        if self._working_tree_diff().strip():
            return "", self.TREE_ALREADY_DIRTY_OBSTACLE
        for candidate in pair:
            patch = self.archive.read_candidate_file(candidate.iteration, "change.diff")
            if not str(patch or "").strip():
                # Reported apart from a conflict because the two ask for opposite responses.
                self._git_discard_worktree()
                return "", (f"iteration {candidate.iteration}'s archived diff is missing or unreadable")
            if not self._git_apply_patch(patch):
                self._git_discard_worktree()
                return "", (f"iteration {candidate.iteration}'s diff would not apply over the other's")
        staged = self._working_tree_diff()
        if not staged.strip():
            self._git_discard_worktree()
            return "", "both diffs applied but changed nothing against HEAD"
        return staged, ""

    def _decline_merge_attempt(
        self,
        iteration: int,
        pair: tuple[MergeCandidate, MergeCandidate],
        obstacle: str,
        *,
        about_the_iteration: bool = False,
    ) -> None:
        """Report a pair that was selected and not measured, and drop it or not."""
        print(f"  [merge] declined: {obstacle}")
        if not about_the_iteration and obstacle != self.TREE_ALREADY_DIRTY_OBSTACLE:
            self._declined_merge_pairs.add(frozenset({pair[0].iteration, pair[1].iteration}))
        self.state_store.append_event(
            make_event(
                "merge_attempt_declined",
                iteration,
                first_iteration=pair[0].iteration,
                second_iteration=pair[1].iteration,
                obstacle=obstacle,
            )
        )

    def _stage_merge(self, *, iteration, hazard, fan_out_plan) -> StagedMerge:
        """Stack two rejected gains, and say which pair and what it staged.

        Stacking costs a measurement but no session, so it is tried before
        spending another Implementer round on a search that has stopped
        producing a new best -- and, once that search has stalled, before
        draining a candidate the same search bought.
        """
        merge_pair = None if hazard or fan_out_plan is not None else self._select_merge_attempt()
        merge_refusal = "" if merge_pair is None else self._merge_attempt_refusal()
        lane_queue_depth = 0 if hazard else len(self._lane_queue)
        if merge_refusal:
            merge_diff, merge_obstacle = "", merge_refusal
        else:
            merge_diff, merge_obstacle = self._stage_merge_attempt(merge_pair)
        self._merge_precedence_streak = self._merge_precedence_streak + 1 if merge_diff else 0
        if merge_diff and lane_queue_depth:
            # Distinct from ``merge_attempt_staged``, and not foldable into it: that counts every stack measured,
            # this counts the ones that went ahead of a queue a round already paid for, which is the only thing
            # precedence can cost.
            print(f"  [merge] precedence over a lane queue {lane_queue_depth} deep")
            self.state_store.append_event(
                make_event(
                    "merge_took_precedence",
                    iteration,
                    lane_queue_depth=lane_queue_depth,
                    first_iteration=merge_pair[0].iteration,
                    second_iteration=merge_pair[1].iteration,
                    unresolved_stall_iters=(self.run_state.stall.unresolved_stall_iters),
                )
            )
        if merge_pair is not None and merge_obstacle:
            self._decline_merge_attempt(
                iteration,
                merge_pair,
                merge_obstacle,
                # A refusal did not reach the pair's diffs, so it is not evidence about them and must not be
                # remembered against them.
                about_the_iteration=bool(merge_refusal),
            )
        return StagedMerge(merge_pair, merge_diff)


__all__ = ["MergeStackingMixin", "StagedMerge"]
