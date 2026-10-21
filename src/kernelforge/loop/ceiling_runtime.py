# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Roofline ceiling estimation, adoption and attainment reporting for the forge-loop."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any


log = logging.getLogger(__name__)


# Distinguishes "the ceiling report has not been looked for yet" from "it was looked for and is not there", so a
# missing or corrupt report is read from disk once per campaign rather than once per iteration.
_CEILING_UNLOADED = object()


class CeilingRuntimeMixin:
    """Own the campaign's roofline ceiling: establish it, read it back, and report attainment."""

    async def _establish_ceiling(self) -> None:
        """Estimate this kernel's per-shape ceiling once per campaign, before its first round.

        Runs only when the caller supplied an estimator, which is to say the
        operator turned the roofline ceiling on. A fresh campaign estimates. A
        resumed one reads back the ceiling its run state records, and estimates
        only when that report can no longer be used. The estimate costs a
        profiler pass and an analyst session, and it must not change for the
        life of the campaign: the ceiling is a property of the operator and the
        box, not of the current implementation, and the stop rule divides by it,
        so a second estimate would move the target between segments.

        The case set and per-case latencies come from the baseline measured just
        above, which is both a better clock than a single run and one driver run
        the estimator no longer has to pay for.

        An estimate that ends in one of ``ESTIMATE_ERRORS`` is reported and
        dropped. Without a ceiling the campaign simply has no attainment target
        and runs to its time budget, which is what every campaign did before
        this existed. Any other exception is a bug in how the estimator was
        wired, and raises rather than passing for that degraded path.
        """
        if self._ceiling_estimator is None:
            return
        scored = self._scored_case_ids()
        if not scored:
            return
        if self.resume and self._adopt_recorded_ceiling(scored):
            return
        from kernelforge.roofline_ceiling.estimate import ESTIMATE_ERRORS

        anchor = self._best_case_times or self._baseline_case_times
        case_ms = {case_id: anchor[case_id] for case_id in scored}
        print("Estimating the roofline ceiling for this kernel...")
        try:
            outcome = await self._ceiling_estimator(case_ids=scored, case_ms=case_ms)
        except ESTIMATE_ERRORS as exc:
            log.warning("roofline ceiling unavailable: %s", exc, exc_info=True)
            print(f"  [roofline] no ceiling for this campaign: {str(exc) or type(exc).__name__}")
            return

        self._ceiling_report_path = str(outcome.report_path)
        self._ceiling_report = outcome.report
        self._record_ceiling()
        for note in outcome.notes:
            print(f"  [roofline] {note}")
        standing = self._roofline_attainment()
        if standing is not None and standing.usable:
            print(
                f"  [roofline] ceiling published ({outcome.source}); attainment "
                f"{standing.mean * 100:.1f}% of estimate across {len(standing.cases)} case(s)"
            )
        else:
            print(f"  [roofline] ceiling published ({outcome.source}); no attainment figure yet")
        for case_id, reason in sorted((standing.excluded if standing else {}).items()):
            print(f"  [roofline] {case_id}: {reason}")

    def _adopt_recorded_ceiling(self, scored: list[str]) -> bool:
        """Take back the ceiling this campaign estimated in an earlier session.

        Returns whether one was adopted. A recorded report that can no longer be
        read, or that has no figure for a scored case, is reported and estimated
        again: the alternative is a campaign with no target for the rest of its
        run, and the gate would refuse to rule on a partial one anyway.
        """
        path = str(self.run_state.ceiling_report_path or "").strip()
        if not path:
            return False
        self._ceiling_report_path = path
        self._ceiling_report = _CEILING_UNLOADED
        report = self._ceiling()
        if report is None:
            reason = "cannot be read back"
        else:
            missing = sorted(set(scored) - set(report.ideal_ms()))
            reason = f"has no figure for {', '.join(missing)}" if missing else ""
        if reason:
            self._ceiling_report_path = ""
            self._ceiling_report = _CEILING_UNLOADED
            print(f"  [roofline] the ceiling this campaign published at {path} {reason}, so estimating again")
            return False
        print(f"  [roofline] resumed with the ceiling this campaign published: {path}")
        return True

    def _record_ceiling(self) -> None:
        """Checkpoint where this campaign's ceiling was published, for a resume to read back.

        ``LoopStateStore.save`` records its own write failures as degraded
        persistence rather than raising, so there is nothing to catch here.
        """
        self.run_state.ceiling_report_path = self._ceiling_report_path
        self.state_store.save(self.run_state)

    def _with_ceiling_standing(self, context):
        """Attach the roofline standing to a planning context, when a ceiling is published.

        This is where the ceiling steers the campaign: the planner decides which
        cases the round's effort goes to, so it is the one that has to see which
        still have headroom, and the supervisor reviewing a stall has to see
        whether the kernel is near its ceiling or far from it. Each case carries
        its own figure; the context carries the mean, the target and how to
        read both. The implementer's copy, from ``_render_ceiling_advisory``,
        arrives after the round's cases are chosen. Like the rest of the
        planning evidence it is guidance for the next round and never enters a
        KEEP.
        """
        report = self._ceiling()
        standing = self._roofline_attainment()
        if report is None or standing is None:
            return context
        from kernelforge.orchestrator.contracts import CampaignRoofline, CaseRoofline

        ceilings = report.ideal_ms()
        scored = {entry.case_id: entry for entry in standing.cases}
        incumbent = self._best_case_times or self._baseline_case_times

        def roofline(case_id: str) -> CaseRoofline | None:
            if case_id not in ceilings:
                return None
            entry = scored.get(case_id)
            if entry is not None:
                return CaseRoofline(
                    ceiling_ms=entry.t_ideal_ms,
                    incumbent_ms=entry.t_current_ms,
                    attainment=entry.attainment,
                )
            measured = incumbent.get(case_id)
            return CaseRoofline(
                ceiling_ms=ceilings[case_id],
                incumbent_ms=measured if measured and measured > 0 else None,
                excluded=standing.excluded.get(case_id, ""),
            )

        scored_ids = self._scored_case_ids()
        target = float(self.ic.roofline_target or 0.0)
        return replace(
            context,
            cases=tuple(replace(case, roofline=roofline(case.case_id)) for case in context.cases),
            roofline=CampaignRoofline(
                scored_cases=len(scored_ids),
                covered_cases=len(set(scored_ids) & set(scored)),
                mean_attainment=standing.mean,
                target=target if target > 0 else None,
            ),
        )

    def _ceiling(self) -> Any | None:
        """The published ceiling report for this kernel, or ``None``.

        Loaded on demand rather than in ``__init__`` so a ceiling published
        part-way through a campaign is picked up at the next iteration, and a
        missing or corrupt one costs a log line rather than the run.
        """
        path = self._ceiling_report_path.strip()
        if not path:
            return None
        cached = getattr(self, "_ceiling_report", _CEILING_UNLOADED)
        if cached is _CEILING_UNLOADED:
            try:
                from kernelforge.roofline_ceiling.report import read_report

                cached = read_report(path)
            except (OSError, ValueError) as exc:
                log.warning("ceiling unavailable from %s: %s", path, exc)
                cached = None
            self._ceiling_report = cached
        return cached

    def _roofline_attainment(self) -> Any | None:
        """Score the published ceiling against the incumbent's own per-case times.

        The divisor is the incumbent, not the pristine anchor: attainment has to
        move as the campaign improves the kernel, and the anchor by definition
        does not move at all.
        """
        report = self._ceiling()
        if report is None:
            return None
        from kernelforge.roofline_ceiling.attainment import measure_attainment

        return measure_attainment(
            report,
            self._best_case_times or self._baseline_case_times,
            unscored_cases=self._unscored_cases,
        )

    def _is_roofline_target_met(self) -> bool:
        """Whether mean attainment has reached the target the operator asked for.

        Two things have to hold, and the second is not a second opinion -- it is
        what makes the first a number at all.

        The mean must cover every scored case. A case excluded from the mean is
        excluded from the objective the mean claims to report, so dropping the
        three shapes furthest from their ceilings leaves the four easiest
        averaging comfortably above target. The commonest exclusion is a ceiling
        that sits below the latency already measured, which is the work model
        contradicting itself -- exactly the estimate that must not be allowed to
        end a campaign.

        Nothing here touches KEEP. Whether one candidate beats another stays a
        measurement against the incumbent; this decides only whether to buy
        another round.
        """
        target = float(self.ic.roofline_target or 0.0)
        if target <= 0:
            return False
        standing = self._roofline_attainment()
        if standing is None or not standing.usable:
            return False
        if not standing.covers(self._scored_case_ids()):
            for case_id, reason in sorted(standing.excluded.items()):
                print(f"  [roofline] {case_id} has no attainment figure, so the target cannot be ruled on: {reason}")
            return False
        return standing.mean >= target

    def _render_roofline_progress(self) -> str:
        """One log line with the incumbent's mean attainment, or ``""`` when the campaign has no ceiling.

        A mean taken over fewer cases than the suite scores says so, because
        that mean is not the objective and must not read as if it were.
        """
        standing = self._roofline_attainment()
        if standing is None:
            return ""
        if not standing.usable:
            return "  [roofline] attainment: no scored case has a usable ceiling"
        scored = self._scored_case_ids()
        covered = {entry.case_id for entry in standing.cases} & set(scored)
        coverage = "" if standing.covers(scored) else f" over {len(covered)} of {len(scored)} scored cases"
        target = float(self.ic.roofline_target or 0.0)
        suffix = f" (target {target * 100:.0f}%)" if target > 0 else ""
        return f"  [roofline] attainment {standing.mean * 100:.1f}% of the estimated ceiling{coverage}{suffix}"

    def _render_ceiling_advisory(self) -> str:
        """Render the roofline standing for the implementer, when a ceiling is published.

        The planner has already chosen the round's cases from the same standing,
        carried as case evidence by ``_with_ceiling_standing``; this is the
        implementer's view of it, so the session working a case knows how far
        that case sits from its ceiling. Guidance, not a verdict: because the
        ceiling is fixed for the campaign, pointing the agent at attainment and
        pointing it at latency ask for the same thing, so this block adds a
        direction without adding an incentive.
        """
        report = self._ceiling()
        if report is None:
            return ""
        from kernelforge.roofline_ceiling.report import render_for_prompt

        return render_for_prompt(
            report,
            self._best_case_times or self._baseline_case_times,
            target=float(self.ic.roofline_target or 0.0),
            unscored_cases=sorted(self._unscored_cases),
        )
