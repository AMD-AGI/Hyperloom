# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A round's lanes: running their sessions in workspace copies, and merging what they left into one candidate."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from hyperloom_common.io import DURABLE, JsonFormat, atomic_write_json, atomic_write_text
from kernelforge.config.edit_policy import FORGE_PROTECTED_PATHS
from kernelforge.evaluation.scoring import GateScoring, beats_current_best
from kernelforge.utils.git import git
from kernelforge.workflows.forge_loop.campaign.experiments_layout import ExperimentsLayout
from kernelforge.workflows.forge_loop.iteration.fanout import (
    LaneAssignment,
    LaneResult,
    LaneRound,
    LaneSeat,
    lane_diff,
)
from kernelforge.workflows.forge_loop.iteration.iteration_types import (
    IterationResult,
    RoundSession,
    _append_finding,
    _IterationFrame,
)

log = logging.getLogger(__name__)

# How much of a lane's final message the merge session is shown; the rest is in the lane's record.
_RATIONALE_CHARS = 600


def _patch_paths(patch: str, *, cwd: str) -> list[str]:
    """Every workspace path a patch writes, as git itself reads them."""
    handle = tempfile.NamedTemporaryFile("w", suffix=".diff", encoding="utf-8", delete=False)
    try:
        handle.write(patch if patch.endswith("\n") else patch + "\n")
        handle.close()
        completed = git("apply", "--numstat", "-z", handle.name, cwd=cwd, check=False)
    finally:
        os.unlink(handle.name)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip() or f"git apply --numstat exited {completed.returncode}"
        raise ValueError(f"the paths in the diff could not be read: {detail}")
    paths: list[str] = []
    for record in (record for record in completed.stdout.split("\0") if record):
        fields = record.split("\t")
        if len(fields) < 3:
            raise ValueError(
                f"the paths in the diff could not be read: unreadable git apply --numstat record {record!r}"
            )
        paths.append(fields[2])
    for line in patch.splitlines():
        for prefix in ("rename from ", "copy from "):
            if line.startswith(prefix):
                paths.append(line[len(prefix) :].strip())
    return sorted({path for path in paths if path})


def _lane_label(lane: LaneResult, width: int) -> str:
    """How a lane is named in the round's record: nothing for a round's only lane."""
    return f"lane {lane.lane_id}" if width > 1 else ""


def _case_times(lane: LaneResult) -> dict:
    """The per-case times the lane's stop check measured, when it measured any."""
    measurement = lane.sink.get("benchmark_measurement")
    return dict(measurement.get("case_times") or {}) if isinstance(measurement, dict) else {}


class LaneRuntimeMixin:
    """Run a round's lanes, choose where the merge starts, and tell the merge session what the lanes left."""

    def _round_assignments(self) -> list[LaneAssignment]:
        """This round's lanes: one per focus the Analysis gave, up to the campaign's width; one lane without them."""
        focuses = self._lane_focuses()[: max(1, int(self.ic.lanes))]
        if len(focuses) < 2:
            return [LaneAssignment("1")]
        return [LaneAssignment(str(index), focus) for index, focus in enumerate(focuses, start=1)]

    async def _run_lanes(self, frame: _IterationFrame, *, agent_factory, digest) -> list[LaneResult]:
        """Run the round's lane sessions side by side, each in its own copy, and read what each one changed."""
        iteration = frame.iteration
        assignments = self._round_assignments()
        width = len(assignments)
        print(f"  [lanes] round of {width} lane(s)")
        # Beside the workspace, not in /tmp: a lane copy carries the build outputs, and /tmp is typically a smaller
        # local filesystem than the one sized for the campaign.
        self._lane_round = LaneRound(
            workspace_dir=self.ic.workspace_dir,
            parent_dir=str(Path(self.ic.workspace_dir).resolve().parent),
            lanes=assignments,
        )
        await self._lane_round.open()
        kernel_relative = self._workspace_path(self.ic.kernel_file)
        scoring = GateScoring(dict(self._baseline_case_times), self.best_mean_case_speedup)

        async def session(seat: LaneSeat) -> tuple[str, dict]:
            sink: dict = {}
            self._bind_session_producer(sink, self.run_state.agent_relay.active_profile_id)
            call = agent_factory(seat, gain_required=width == 1)
            history = self._implementer_history(digest=digest, direction=self._lane_direction(seat, assignments))
            _error, rationale = await self._run_session(
                call, str(seat.lane_dir / kernel_relative), history, sink=sink, gate_scoring=scoring
            )
            return rationale, sink

        results = await self._lane_round.run(session)
        for lane in results:
            self._read_lane_candidate(lane)
            self._file_lane(iteration, lane)
        self._record_lane_contention(iteration, results)
        return results

    def _read_lane_candidate(self, lane: LaneResult) -> None:
        """Read the lane's diff, refusing one that touches the measurement surface."""
        if lane.error:
            print(f"  [lane {lane.lane_id}] session lost: {lane.error}")
            return
        try:
            lane.diff = lane_diff(lane.lane_dir, self._lane_new_paths(lane.lane_dir))
            rejection = self._lane_rejection(lane.diff) if lane.diff.strip() else ""
        except (OSError, RuntimeError, ValueError) as error:
            rejection = f"its changes could not be read: {error}"
        if rejection:
            lane.error = rejection
            print(f"  [lane {lane.lane_id}] candidate rejected: {rejection}")
        elif lane.sink.get("integrity_violation") is True:
            print(f"  [lane {lane.lane_id}] candidate rejected: {lane.sink.get('integrity_reason') or 'integrity'}")

    def _lane_new_paths(self, lane_dir: Path) -> list[str]:
        """The new files a lane created that a KEEP may commit: allowlisted, and not there when the round began."""
        if not self.ic.commit_new_paths:
            return []
        present_before = self._pre_untracked or set()
        admitted, _refused = self._new_paths(str(lane_dir))
        return [path for path in admitted if path not in present_before]

    def _lane_rejection(self, patch: str) -> str:
        """Why a lane candidate must not reach the canonical tree, or \"\"."""
        paths = _patch_paths(patch, cwd=self.ic.workspace_dir)
        protected = sorted(
            path
            for path in paths
            if FORGE_PROTECTED_PATHS.is_protected(
                path,
                workspace=self.ic.workspace_dir,
                # The campaign driver carries no protected name of its own.
                exact_paths=(self.ic.driver_script,),
            )
        )
        return "it changes the measurement surface: " + ", ".join(protected) if protected else ""

    def _lane_artifact_dir(self, iteration: int, lane_id: str) -> Path:
        """Where one lane's patch and measurement are kept for the merge session and for later sessions."""
        return ExperimentsLayout.of(Path(self.ic.workspace_dir).resolve()).round(iteration) / f"lane_{lane_id}"

    def _file_lane(self, iteration: int, lane: LaneResult) -> None:
        """Publish what one lane left: its patch, and its stop check's verdict and per-case times."""
        root = self._lane_artifact_dir(iteration, lane.lane_id)
        try:
            root.mkdir(parents=True, exist_ok=True)
            if lane.diff.strip():
                atomic_write_text(root / "change.diff", lane.diff, options=DURABLE)
            atomic_write_json(
                root / "lane.json",
                {
                    "focus": lane.assignment.focus,
                    "end_reason": lane.sink.get("end_reason", ""),
                    "plan": lane.sink.get("plan", ""),
                    "error": lane.error,
                    "verified": lane.verified,
                    "mean_case_speedup": lane.score,
                    "case_times": _case_times(lane),
                },
                fmt=JsonFormat(trailing_newline=True),
                options=DURABLE,
            )
        except OSError as error:
            log.warning("lane %s artifacts not written: %s", lane.lane_id, error)

    def _record_lane_contention(self, iteration: int, results: list[LaneResult]) -> None:
        """Record a device a lane's teardown could not clear: the canonical measurement shares it."""
        contended = [lane for lane in results if lane.contended]
        if not contended:
            return
        hazard = self.device_hazard.record(
            iteration=iteration,
            detail="; ".join(f"lane {lane.lane_id}: {lane.reaped.describe()}" for lane in contended),
            pids={pid for lane in contended for pid in lane.reaped.blockers},
        )
        print(
            f"  [lanes] {len(contended)} of {len(results)} lanes left the device contended; this round measures "
            f"nothing. {hazard.describe()}"
        )

    def _close_lane_round(self) -> None:
        """Remove the round's workspace copies, once its sessions have written their records."""
        if self._lane_round is not None:
            self._lane_round.close()
            self._lane_round = None

    def _unmeasurable_on_a_held_device(self, *, iteration: int, detail: str, session_sink: dict) -> IterationResult:
        """File an iteration that refused to measure on a device it does not own."""
        summary = (
            "REVERT (workspace contention): canonical correctness and benchmark "
            "were skipped because the device is still held by processes this "
            f"campaign could not clear. {detail}"
        )
        _append_finding(session_sink, summary)
        print("  [REVERT] Device still contended; nothing measured this iteration")
        return IterationResult(
            iteration=iteration,
            duration_sec=0.0,
            validation_passed=False,
            validation_summary=summary,
            kept=False,
            workspace_contention=detail,
        )

    def _merge_start(self, changed: list[LaneResult]) -> LaneResult | None:
        """The lane the round's candidate starts from, or None for the round's base.

        A round with one changed lane has nothing to merge: that lane is the candidate, and the canonical measurement
        judges it. Otherwise the start is the lane whose own stop check scored highest above the current best, and the
        base when none did.
        """
        if len(changed) == 1:
            return changed[0]
        ahead = [
            lane
            for lane in changed
            if lane.verified and beats_current_best(lane.score, best_mean_case_speedup=self.best_mean_case_speedup)
        ]
        return max(ahead, key=lambda lane: lane.score) if ahead else None

    def _apply_lane(self, lane: LaneResult) -> bool:
        """Put one lane's changes on the canonical tree, leaving the tree clean when they do not apply."""
        if self._git_apply_patch(lane.diff):
            return True
        self._git_discard_worktree()
        lane.error = "its patch did not apply to the round's base"
        print(f"  [lane {lane.lane_id}] {lane.error}")
        return False

    def _lane_sessions(self, results: list[LaneResult]) -> list[RoundSession]:
        """The lane sessions that owe the round a record, lowest stop-check score first.

        Each record turn rewrites the campaign's insight list, so the lane that measured best writes last.
        """
        ran = [lane for lane in results if lane.sink.get("session_started") is True]
        ran.sort(key=lambda lane: lane.score if lane.score is not None else float("-inf"))
        return [RoundSession(_lane_label(lane, len(results)), lane.sink) for lane in ran]

    def _lane_direction(self, seat: LaneSeat, assignments: list[LaneAssignment]) -> str:
        """What a lane's session works on: a direction of its own choosing, inside its focus when it has one."""
        direction = self._session_direction()
        focus = seat.assignment.focus
        if not focus:
            return direction
        others = "\n".join(
            f"- lane {other.lane_id}: {other.focus}" for other in assignments if other.lane_id != seat.assignment.lane_id
        )
        return (
            f"{direction}\n\n"
            "## This lane's focus\n"
            f"This round runs {len(assignments)} sessions side by side from the same starting point, each in its own "
            "copy of the workspace, and a merge session combines what they leave. Yours is lane "
            f"{seat.assignment.lane_id}. Its focus:\n\n{focus}\n\n"
            f"The other lanes' focuses, so that you do not repeat their work:\n{others}\n\n"
            "The focus says where to start, not where to stop: once your own measurements show it has no room left, "
            "take the largest gain the other lanes are not working on. Where the focus names scored cases, judge a "
            "change by those cases."
        )

    def _merge_direction(self, iteration: int, start: LaneResult | None, rest: list[LaneResult]) -> str:
        """What the merge session is to do with the lanes the canonical tree does not hold yet."""
        best = self.best_mean_case_speedup
        if start is None:
            standing = (
                f"This workspace holds the round's starting point (score {best}): no lane's own result beat it. "
                "A lane can still be faster on some cases."
            )
        else:
            standing = (
                f"This workspace already holds lane {start.lane_id}'s result, the round's best by its own stop check "
                f"(score {start.score}; per-case ms {_case_times(start)}). That score is what this session has to beat."
            )
        lanes = "\n".join(self._lane_brief(iteration, lane) for lane in rest)
        return (
            "## Direction: merge this round's lanes\n"
            f"This round ran {len(rest) + (start is not None)} session(s) side by side from one starting point, each "
            f"in its own copy of the workspace. {standing}\n\n"
            "Bring the changes of the lanes below into this workspace so that the result scores higher than what it "
            f"holds now:\n{lanes}\n\n"
            "How to merge is yours to decide: apply a patch as it is, rewrite the places where two lanes changed the "
            "same code, dispatch per case between two lanes' implementations where each wins on different cases, "
            "re-tune a constant where they meet. A lane whose overall score did not rise may still be faster on some "
            "cases: take those. A lane marked unverified ended before its own stop check passed, so check its "
            "correctness before relying on it. This session opens no optimization direction of its own: it ends when "
            "each lane's change is either in the kernel or measured as not worth taking. In your final message, say "
            "for each lane whether its change is in, and for one that is not, what you measured and why."
        )

    def _lane_brief(self, iteration: int, lane: LaneResult) -> str:
        """One lane as the merge session reads it: its focus, where its patch and files are, and what it measured."""
        root = self._lane_artifact_dir(iteration, lane.lane_id)
        if lane.verified:
            measured = f"stop check passed, score {lane.score}, per-case ms {_case_times(lane)}"
        else:
            measured = "unverified"
        focus = f" Focus: {lane.assignment.focus}" if lane.assignment.focus else ""
        said = " ".join(lane.rationale.split())[:_RATIONALE_CHARS]
        return (
            f"- lane {lane.lane_id}: {measured}.{focus}\n"
            f"  patch against the starting point: {root / 'change.diff'}\n"
            f"  its workspace, with every file as the lane left it: {lane.lane_dir}\n"
            f"  its final message: {said}"
        )
