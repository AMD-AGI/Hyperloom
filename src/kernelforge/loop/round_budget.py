# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Whether the remaining campaign budget can pay for another round.

The loop's original admission guard was one constant: it refused to START a new
session below ``budget_reserve_sec`` and asked nothing else. That answers "is
some time left", not "can this round finish", and the two differ by the cost of
the round itself. Two campaigns of ten 11-hour production runs (2026-08-17)
opened a three-lane round with half an hour left, spent every minute of it
planning, and were killed by the external timeout with their lane sessions still
running and no report written.

Replaying the policy over the 82 rounds of those campaigns says WHEN the
decision is taken matters more than what it is priced from. At round start the
two rounds that died had 30 and 32 minutes left and four rounds that finished
had 33, 42, 49 and 50 -- interleaved, so no threshold on that number separates
them. Measured again once planning has returned, the two deaths sit at 7.3 and
8.3 minutes and the worst survivor at 24.8: a threefold gap. Planning is why.
It is the dominant term and the most variable one -- 12.5 to 31.6 minutes over
75 measured rounds -- so a pre-planning threshold that covers its high water
refuses rounds that would have succeeded, and the constants this module first
shipped with refused 14 of the 82, including the largest single gain any of the
ten campaigns found.

So the decision is taken twice, for two different questions:

* :func:`admit_round`, before planning, asks only whether the round can
  possibly run. It is priced as a LOWER bound -- the cheapest planning anything
  has been observed to do, plus the least an execution can cost -- so it
  refuses only what provably cannot finish, and narrows before it refuses.
* :func:`admit_dispatch`, after planning has returned and its cost is a
  measurement rather than an estimate, decides whether the round's session may
  actually be started. This is the check that separates the production deaths
  from the production survivors.

The two questions are also answered from different evidence, and the same
number cannot serve both. Before planning nothing is committed, so being
generous only refuses a round that would have worked; after planning the loop
is about to start a session it cannot interrupt, so being ungenerous starts a
session the external timeout kills. Hence two session prices -- the p25 for the
first question (:data:`ADMISSION_SESSION_SEC`) and the median for the second
(:data:`DISPATCH_SESSION_SEC`) -- and a floor under the second requirement
(:data:`DISPATCH_FLOOR_SEC`) that this campaign's own observations may raise
past but never lower: what the dispatch check guards against is an external
deadline the loop neither sets nor measures, and that deadline does not move
because this campaign's validation got faster.

One consequence is worth naming: because the two checks are priced apart, a
round can be admitted to planning and then refused at dispatch even if planning
costs exactly what it was estimated to. The first check is a bound on whether a
round could run at all, not a promise that it will be dispatched. That costs
the planning of a round in a band a few minutes wide -- and only really costs
it for a round narrowed to a single lane, since a fan-out round's plans are
published before dispatch and the next session picks them up.

The finalize reserve is deliberately charged to neither. Both functions are
handed ``remaining_sec`` with nothing subtracted -- the same unreserved number
the loop compares against the reserve -- so the reserve is a bound of its own
beside these, not a term inside them. A round runs when what remains clears the
reserve AND clears the round's price, each on its own; the larger of the two
binds, and no round has to cover ``reserve + its own cost``. That is the point:
the loop already holds the reserve back before every iteration
(``budget_reserve_sec``), and adding it again on top of a round's own cost is
what made the first version of this module refuse rounds that went on to
produce a KEEP.
"""

from __future__ import annotations

from dataclasses import dataclass

from kernelforge.loop.run_state import RoundCost
from kernelforge.orchestrator.plan_critic import PLAN_CRITIC_TIMEOUT_SEC

# The fastest of the 75 measured rounds. Planning is priced here only as a lower
# bound, so this is what a campaign that has observed no round of its own
# assumes planning will cost, and nothing derived below claims to plan faster
# than the fastest round anything has ever observed.
PLANNING_FLOOR_SEC = 750.0

# What an Implementer session is priced at BEFORE planning, where the only
# question is whether a round could run at all. Over 219 Implementer sessions
# of the same ten campaigns a session ran 1.7 minutes at its fastest, 8.0 at
# the p25, 12.3 at the median, 22.2 at the p75 and 34.6 at the p90; 215 of the
# 219 produced at least one edit, and the fastest of those was also 1.7
# minutes. This is the p25, for what the constant means here: one session in
# four returned within it, so a round admitted on it has a real chance of
# returning something, while the median and above price a session at more than
# most sessions needed -- which is what refused rounds that went on to produce
# a KEEP. The observed minimum would be worse in the other direction: it is one
# session in 219, and a round admitted on it can only finish if this session is
# the fastest ever seen. The asymmetry that picks the p25 is local to this
# check: nothing is committed yet, so a bound that is too generous costs a
# round that would have worked, which is the failure this was calibrated away
# from. Dispatch faces the opposite asymmetry and prices the same session at
# :data:`DISPATCH_SESSION_SEC`. This is an admission input, not a timeout -- a
# session already running is never cut off here, and a session that needs
# longer than this is not shortened by it.
ADMISSION_SESSION_SEC = 480.0

# What an Implementer session is priced at AFTER planning, where the loop is
# about to start something it cannot interrupt. Here too small starts a session
# the external timeout kills mid-flight, and the round produces nothing at all;
# too large costs the campaign one more iteration. So this is the median of the
# same 219 sessions rather than their p25: half of every session ever observed
# finished inside it, against one in four for the pre-planning bound. It is the
# largest of the observed percentiles the evidence permits, and the next one up
# is not close: priced at the p75 (22.2 minutes), a campaign that has yet to
# observe its own measurement cycle would demand 32.2 minutes after planning
# and refuse the survivor that came out of planning with 24.8 and went on to a
# measured candidate. What that survivor actually caps this at is 14.8 minutes
# (it must leave room for the 10-minute cycle a campaign assumes before it has
# measured one), so the median is a choice inside the permitted range rather
# than its edge.
DISPATCH_SESSION_SEC = 738.0

# The floor under the dispatch requirement, and the one number here that this
# campaign's own speed cannot lower.
#
# Every other estimate in this module tracks what the campaign has observed,
# and rightly so: they price what the round is about to SPEND, and a campaign
# that validates in 150 seconds really does need less time than one that takes
# 600. The dispatch check is not only pricing that. It is the last decision
# before the loop commits to a session it cannot interrupt, and what kills such
# a session is the EXTERNAL timeout -- which the loop does not set, cannot
# measure and cannot stop. A session is not cut short when the budget runs out;
# its own wall-clock bound is sized from the campaign's TOTAL length, not from
# what is left of it, so a round dispatched with ten minutes remaining still
# runs for however long the session takes. Nothing about that deadline gets
# cheaper because this campaign's validation got faster, so the requirement it
# guards must not follow the campaign's measurements down. Without a floor it
# does, and by a lot: one observed 150-second cycle took the check as first
# shipped from 18 minutes to 10.5, which is 2.2 minutes above the deaths this
# module exists to prevent. 150 seconds is the WORST of the 171 production
# cycles and the estimate converges to a campaign's own worst, so most
# campaigns land under it and fall further -- at the production median of 36
# seconds the unfloored check is 8.6 minutes, three tenths of a minute above
# the round that died with 8.3. Pricing the session at the median rather than
# the p25 only moves those to 14.8 and 12.9; what stops the fall is this floor,
# not the session price above it.
#
# The value is derived from the kill it protects against, and then checked
# against the rounds that survived. Production gave these campaigns a
# 10.75-hour internal budget against an 11-hour external kill: 15 minutes of
# grace past the loop's own deadline. A session dispatched with X left overruns
# that deadline by (session length - X), so it dies when session length exceeds
# X + 15 minutes. At X = 34.6 - 15 = 19.6 minutes every session shorter than
# the p90 -- 9 in 10 of the ones production ran -- ends inside the grace, and
# the shorter the session the more of that grace is left for the measurement
# cycle that follows it: the median session leaves 22 minutes, the p90 leaves
# none. So the floor sizes the SESSION against the kill and no more. Paying for
# the cycle on top of it is the estimate's job, which is why the estimate is
# what binds whenever it is the larger of the two.
#
# 19.6 minutes is also close to as high as the floor can go. It has to fit
# under the worst round that survived, which came out of planning with 24.8
# minutes and went on to a measured candidate, and above the two that died with
# 7.3 and 8.3 -- which it clears by more than twice their margin. A floor that
# covered the 10-minute cycle a campaign assumes before it has measured one on
# top of that p90 session would be 29.6 minutes and would refuse that survivor.
# This is the strongest floor the evidence supports, not the strongest one
# imaginable.
#
# It does assume the deployment leaves grace between the budget the loop counts
# down and the deadline that kills it. A campaign given --max-hours equal to
# its external timeout has no grace at all, and no constant here can invent it.
DISPATCH_FLOOR_SEC = 1176.0

# What the canonical validation and benchmark cost a round that has observed
# none of its own. Over 171 canonical validate-and-benchmark cycles of the same
# campaigns the cycle cost 5 seconds at its fastest, 36 at the median, 110 at
# the p90 and 150 at its worst. This is four times that worst case: ten kernels
# are not every kernel, and a cold JIT rebuild, a wider case set or a slower
# device can legitimately cost far more than anything those campaigns saw. It
# is still a fourteenth of the per-step timeout ceilings this used to be priced
# from, which is what a round buys at worst rather than what it costs.
FIRST_ROUND_MEASUREMENT_SEC = 600.0


@dataclass(frozen=True)
class RoundAdmission:
    """The decision on planning one round, and every number it was made from.

    ``lanes`` is the width the round may be planned at; it is meaningful on a
    refusal too, where it names the narrowest round that was still too
    expensive.
    """

    admitted: bool
    lanes: int
    remaining_sec: float
    required_sec: float
    planning_sec: float
    execution_sec: float
    # True when the round was admitted at less than the width it asked for.
    narrowed: bool

    def summary(self) -> str:
        """One line naming the decision's cost breakdown, in minutes."""
        return (
            f"{self.required_sec / 60:.0f} min needed at {self.lanes} lane(s) "
            f"at least (planning {self.planning_sec / 60:.0f}, session and "
            f"measurement {self.execution_sec / 60:.0f}); "
            f"{self.remaining_sec / 60:.0f} min remain"
        )


@dataclass(frozen=True)
class DispatchAdmission:
    """The decision on dispatching a round whose plans are already bought."""

    admitted: bool
    remaining_sec: float
    required_sec: float
    session_sec: float
    measurement_sec: float
    # True when what this round is estimated to spend came in under
    # :data:`DISPATCH_FLOOR_SEC` and the floor is what is being required. Then
    # ``required_sec`` is deliberately more than the parts below it add up to,
    # and this says so rather than leaving the sum looking wrong.
    floored: bool

    def summary(self) -> str:
        """One line naming the decision's cost breakdown, in minutes."""
        priced = f"session {self.session_sec / 60:.0f}, measurement {self.measurement_sec / 60:.0f}"
        if self.floored:
            priced = f"external-timeout floor over {priced}"
        return (
            f"{self.required_sec / 60:.0f} min needed after planning "
            f"({priced}); {self.remaining_sec / 60:.0f} min remain"
        )


def estimate_measurement_sec(history: list[RoundCost]) -> float:
    """Seconds the canonical validation and benchmark are expected to take.

    High-water over what this campaign has observed, the way planning was
    priced before it became a measurement: the cycle's cost is a property of
    the driver, the case set and the device, and an estimate built on the
    middle of the observations is beaten by every round slower than typical.
    A round that failed validation early observed a cheap cycle, which is a
    true observation of what that round spent and is simply outranked by any
    fuller one in the window.
    """
    observed = [cost.measurement_sec for cost in history if cost.measurement_sec > 0]
    if not observed:
        return FIRST_ROUND_MEASUREMENT_SEC
    return max(observed)


def estimate_planning_sec(history: list[RoundCost], *, lanes: int) -> float:
    """A LOWER bound on what a round of ``lanes`` lanes will spend planning.

    A bound rather than an expectation, because the only thing it decides is
    whether planning is worth buying at all -- what the round costs once its
    plans exist is settled afterwards, against a measurement. Refusing a round
    that would have planned faster than anything ever observed is the failure
    this was re-calibrated away from.

    This campaign's own rounds at this width come first. Failing that, a wider
    round's cheapest observation narrowed by what the Critic no longer has to
    read: it is given :data:`PLAN_CRITIC_TIMEOUT_SEC` per plan, and reading is
    the only part of planning that grows with the width of a round -- dispatch
    and the specialists are shared, and the lane syntheses run concurrently.
    Failing that, a narrower round's cheapest, which a wider round has never
    been observed to beat.
    """
    observed = [cost.planning_sec for cost in history if cost.lanes == lanes]
    if observed:
        return min(observed)
    wider = [cost for cost in history if cost.lanes > lanes]
    if wider:
        cheapest = min(wider, key=lambda cost: cost.planning_sec)
        unread_plans = cheapest.lanes - lanes
        # Held at the floor -- or at this campaign's own cheapest round, on the
        # campaign that plans faster than production ever did.
        return max(
            min(PLANNING_FLOOR_SEC, cheapest.planning_sec),
            cheapest.planning_sec - unread_plans * PLAN_CRITIC_TIMEOUT_SEC,
        )
    narrower = [cost.planning_sec for cost in history if cost.lanes < lanes]
    if narrower:
        return min(narrower)
    return PLANNING_FLOOR_SEC


def admit_round(
    *,
    remaining_sec: float,
    requested_lanes: int,
    history: list[RoundCost],
    measurement_sec: float,
) -> RoundAdmission:
    """Decide whether -- and how wide -- the next round may be PLANNED.

    Every width from the requested one down to a single lane is tried, and the
    first the remaining budget covers is the one returned. The intermediate
    widths are tried rather than jumped over, because the bound falls with each
    plan the Critic no longer has to read: a three-lane round that does not fit
    may fit at two, and two lanes search twice as widely as one. A refusal
    carries the numbers for the single-lane round, because that is the cheapest
    round the campaign could not afford.

    This is the cheap half of the decision and it is priced as a lower bound:
    it exists so that a round which provably cannot run does not buy planning
    first. Passing it is not a promise of a dispatch -- that is decided by
    :func:`admit_dispatch` once planning has returned and its cost is known,
    against a requirement priced higher than this one.
    """
    requested = max(1, int(requested_lanes))
    # What a round costs once its plans exist: the least a session can be given
    # and still return something, and the canonical measurement that judges it.
    # Deliberately the MINIMUM viable round rather than the typical one -- the
    # question here is whether the campaign can still buy a whole round, and a
    # narrow one it can finish is worth more than a wide one it cannot.
    execution_sec = ADMISSION_SESSION_SEC + max(0.0, measurement_sec)

    def priced(lanes: int) -> RoundAdmission:
        planning_sec = estimate_planning_sec(history, lanes=lanes)
        required_sec = planning_sec + execution_sec
        return RoundAdmission(
            admitted=remaining_sec >= required_sec,
            lanes=lanes,
            remaining_sec=remaining_sec,
            required_sec=required_sec,
            planning_sec=planning_sec,
            execution_sec=execution_sec,
            narrowed=lanes < requested,
        )

    for lanes in range(requested, 1, -1):
        decision = priced(lanes)
        if decision.admitted:
            return decision
    # The single-lane round is both the last width tried and the one a refusal
    # reports, so it is priced outside the loop and its verdict IS the answer.
    # "Some decision is always returned" is then a property of the code rather
    # than of an assertion, which ``python -O`` would strip.
    return priced(1)


def admit_dispatch(
    *,
    remaining_sec: float,
    measurement_sec: float,
) -> DispatchAdmission:
    """Decide whether a round whose plans are bought may be dispatched.

    The decisive check, taken here because here planning's cost is a
    measurement rather than an estimate. What is left to buy is one Implementer
    session and the canonical measurement that judges what it wrote; a round
    that cannot pay for both cannot produce a measured candidate, and starting
    it spends the campaign's last minutes on a candidate nobody will ever see
    -- which is exactly how the two production rounds died.

    What the plans cost is spent either way. A refused round therefore keeps
    them rather than discarding them: a fan-out round's plans are published
    before dispatch and the next session picks them up unplanned.

    The requirement is what the round is estimated to spend, held at
    :data:`DISPATCH_FLOOR_SEC` from below. The estimate still follows this
    campaign's own measurement cycle, so a campaign whose cycle is genuinely
    expensive requires more than the floor; what it may not do is require less,
    because the deadline that kills a dispatched session is external to the
    loop and does not recede when this campaign gets faster.
    """
    measurement = max(0.0, measurement_sec)
    estimated_sec = DISPATCH_SESSION_SEC + measurement
    required_sec = max(DISPATCH_FLOOR_SEC, estimated_sec)
    return DispatchAdmission(
        admitted=remaining_sec >= required_sec,
        remaining_sec=remaining_sec,
        required_sec=required_sec,
        session_sec=DISPATCH_SESSION_SEC,
        measurement_sec=measurement,
        floored=estimated_sec < DISPATCH_FLOOR_SEC,
    )
