"""The decision procedure: run the validator, apply the gate, emit a verdict.

WHAT AN ITERATION DECLARES and WHAT HAPPENS TO IT are two different things, and
they live apart: ``arbor.measure.intents`` owns the first (WIN / FIX /
FIX-CORRECTNESS / LEARN, one row each, chosen before the measurement) and this
module owns the second. Every per-intent rule the gate applies is read off that
row, so a fifth intent is a fifth row rather than a fifth branch here.

Three verdicts, and only two of them touch the tree. ``NOT_MEASURED`` is the
important one: a bench that never ran, a server that never came up, a 404 from
the wrong wire name, a missing metric key — none of these say anything about
the change. They used to be reported as REVERT, which discarded working code,
incremented ``null_streak``, and drove tier escalation off manufactured
evidence. NOT-MEASURED means the tree stands, the counters do not move, and the
caller retries.

``GateResult.decision`` keeps emitting the legacy
``KEEP``/``REVERT``/``PROBE``/``INVALID`` vocabulary, derived from the
(intent, verdict) pair. That is not a transitional shim: everything downstream
reads it, including ``check_review_consistency.py``, which counts
``l["decision"] == "KEEP"`` straight out of ``session.json``.

A pack that never declares an intent gets ``WIN`` for everything — the
strictest bar — and a pack that configures no secondary metric gets the old
single-axis behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from arbor.measure.intents import (
    DEFAULT_CORRECTNESS_CAP_PCT,
    DEFAULT_NOISE_BAND_PCT,
    KEEP,
    NOT_MEASURED,
    REGISTRY,
    REVERT,
    Intent,
    legacy_decision,
    resolve,
    tolerance,
)

# Re-exported: these were part of this module's surface before the split, and
# the correctness cap is now reached through the registry's tolerance lookup
# rather than read here directly.
__all__ = [
    "GateResult", "run_gate", "CLAIM_CLASSES",
    "DEFAULT_NOISE_BAND_PCT", "DEFAULT_CORRECTNESS_CAP_PCT",
]

# Every legacy class name, for callers that still validate against a list.
CLAIM_CLASSES = tuple(sorted({a for i in REGISTRY for a in i.aliases}))

# Correctness-check errors that describe the harness, not the model. A 404 for
# the wrong wire name and an unreachable server are infrastructure; a 200 whose
# body has no content is a real correctness failure and must still REVERT.
#
# `correctness inconclusive` is the general form, emitted by a validator that
# ran but could not ADJUDICATE — a reasoning model that spent its whole token
# budget thinking, a reply too long to judge, a crashed adjudicator. Without it
# the specific markers below only catch the cases that happen to name an HTTP
# code in their message, and a truncated reply reverts a tree the gate never
# judged. That is the failure class this whole list exists to prevent.
_INFRA_CORRECTNESS_MARKERS = (
    "correctness inconclusive",
    "server unreachable",
    "does not exist",
    "model_not_found",
    "http 404",
    "http 500",
    "http 502",
    "http 503",
    "http 000",
    "non-json",
    # An eval dataset that cannot be loaded (no egress, an offline cache without it): the
    # Hub's own errors, for a validator that reports lm_eval's log tail as it is. The packs'
    # scripts say "correctness inconclusive" for it themselves (Hyperloom on Lux, 2026-10-29).
    "couldn't reach",
    "offlinemodeisenabled",
    "datasetnotfounderror",
    "localentrynotfounderror",
)


@dataclass
class GateResult:
    # KEEP | REVERT | NOT_MEASURED — what happens to the tree, and nothing else.
    verdict: str
    metric_value: float
    best: float | None
    delta_pct: float
    correctness_passed: bool
    integrity_ok: bool
    reason: str
    # What the iteration declared it was buying. The row, not a string.
    intent: Intent = field(default_factory=lambda: resolve(None))
    # Whether the caller should `git revert` the tree. False for INVALID: the
    # measurement failed, the change did not, so the code stays and is retried.
    revert_tree: bool = True
    # True only when the change cleared `threshold_rel` — i.e. it is a claimed
    # win, not merely worth keeping. Drives escalation, not retention.
    booked: bool = False
    # Second axis, when the pack configures one.
    secondary_value: float | None = None
    secondary_best: float | None = None
    secondary_delta_pct: float | None = None
    # Set when the change improves one axis beyond noise and degrades the other
    # beyond noise: a new point on the frontier rather than a strict advance.
    frontier: bool = False
    # Set on a REVERT the SECOND axis decided alone: the primary stayed inside
    # its noise band and the secondary did not. Says nothing about this
    # iteration on its own -- one is ordinary. It exists so the state machine
    # can count a RUN of them across unrelated changes, which is evidence about
    # the secondary incumbent rather than about any of the changes. See
    # state.SECONDARY_SUSPECT_STREAK.
    secondary_decided: bool = False

    # ── Legacy views ────────────────────────────────────────────────────────
    # Derived, never stored, so they cannot drift from the pair above. Kept
    # because everything downstream still reads them — including
    # check_review_consistency.py, which counts l["decision"] == "KEEP"
    # straight out of session.json.

    @property
    def decision(self) -> str:
        """KEEP | REVERT | PROBE | INVALID — the pre-split vocabulary."""
        return legacy_decision(self.intent, self.verdict)

    @property
    def claim_class(self) -> str:
        """A | A-CORRECTNESS | B | PROBE — the pre-split intent name."""
        return self.intent.aliases[0]


@dataclass(frozen=True)
class _Axis:
    """How one measured axis moved against its incumbent.

    ``gain``  — improved past ``threshold``; a real, claimable advance.
    ``ok``    — did not regress past the noise band.
    ``delta`` — signed percent, positive = better. ``None`` when this axis is
                not configured or has no incumbent to compare against.

    An axis the pack does not configure is ``gain=False, ok=True``: it can
    never win and never blocks, which makes every rule below collapse to the
    single-axis behaviour without a special case.
    """

    gain: bool
    ok: bool
    delta: float | None
    key: str | None = None

    @property
    def configured(self) -> bool:
        return self.delta is not None

    def note(self) -> str:
        return f", {self.key} {self.delta:+.1f}%" if self.configured else ""


NEUTRAL_AXIS = _Axis(gain=False, ok=True, delta=None)


def _is_better(value: float, best: float, direction: str, threshold: float) -> bool:
    if direction == "max":
        return value > best * threshold
    return value < best / threshold


def _within_noise(value: float, best: float, direction: str, band_pct: float) -> bool:
    """True when ``value`` has not regressed past the noise band."""
    if direction == "max":
        return value >= best * (1.0 - band_pct / 100.0)
    return value <= best * (1.0 + band_pct / 100.0)


def _delta_pct(value: float, best: float, direction: str) -> float:
    if not best:
        return 0.0
    if direction == "max":
        return (value - best) / best * 100.0
    return (best - value) / best * 100.0


def _measure_axis(value, best, direction, threshold, band_pct, key) -> _Axis:
    """Score one axis, or return NEUTRAL_AXIS when there is nothing to score.

    ``best`` is compared with ``is None`` rather than truthiness on purpose: a
    legitimate incumbent of 0.0 is falsy and would silently disable the axis.
    """
    if value is None or best is None:
        return NEUTRAL_AXIS
    return _Axis(
        gain=_is_better(value, best, direction, threshold),
        ok=_within_noise(value, best, direction, band_pct),
        delta=_delta_pct(value, best, direction),
        key=key,
    )


def _not_measured(reason: str, best, intent: Intent, correctness_passed=True,
                  integrity_ok=True) -> GateResult:
    """Not measured. The change is untouched and the counters must not move."""
    return GateResult(
        NOT_MEASURED, 0.0, best, 0.0,
        correctness_passed, integrity_ok, reason, intent,
        revert_tree=False,
    )


def _correctness_is_infrastructure(error: str) -> bool:
    e = (error or "").lower()
    return any(m in e for m in _INFRA_CORRECTNESS_MARKERS)


def run_gate(
    objective: dict,
    best: float | None,
    bench_result: dict[str, Any],
    correctness_result: dict[str, Any],
    integrity_ok: bool,
    claim_class: str | Intent | None = None,
    secondary_best: float | None = None,
    intent: str | Intent | None = None,
    arm: dict | None = None,
    bundle: dict | None = None,
    accuracy: dict | None = None,
) -> GateResult:
    """Pure gate logic. Callers provide the raw results; this decides the outcome.

    ``intent`` and ``claim_class`` are the same argument under both
    vocabularies; either may be passed. An unknown or absent declaration
    resolves to WIN, the strictest bar.

    ``arm`` is the live server's comparison with the arm's claim
    (arbor/measure/armcheck.py), passed only when the pack enforces it.
    ``bundle`` is a declared bundle's check (arbor/measure/bundles.py), passed
    only when the pack allows bundles. ``accuracy`` is the full-set eval of a
    numerics change (arbor/measure/accuracy.py), passed only when the pack's
    accuracy gate is on; absent, the gate is what it was.
    """
    metric_key = objective["metric"]
    direction = objective["direction"]
    measurement = objective.get("measurement", {})
    threshold = measurement.get("threshold_rel", 1.01)
    noise_band = measurement.get("noise_band_pct", DEFAULT_NOISE_BAND_PCT)
    kind = objective["kind"]

    # Second axis. Absent → NEUTRAL_AXIS below, i.e. exactly the old behaviour.
    sec_key = measurement.get("secondary_metric")
    sec_direction = measurement.get("secondary_direction", "max")
    sec_noise = measurement.get("secondary_noise_band_pct", noise_band)
    sec_threshold = measurement.get("secondary_threshold_rel", threshold)

    intent = resolve(intent if intent is not None else claim_class)
    # How far THIS intent may regress. For a correctness fix that is the
    # catastrophe cap, for everything else the noise band — the one thing that
    # distinguishes FIX-CORRECTNESS from FIX.
    intent_tolerance = tolerance(intent, measurement)

    correctness_passed = correctness_result.get("passed", False)

    # ── Not-measured cases ──────────────────────────────────────────────────
    # A changed validator instrument means no measurement from this iteration
    # can be trusted. That is a statement about the harness, not the change.
    if not integrity_ok:
        return _not_measured(
            "validator integrity check failed — measurement not trustworthy, "
            "iteration NOT counted",
            best, intent, correctness_passed, integrity_ok=False,
        )

    # The same for the arm: a server that ran without what the arm claims
    # measured some other change. A capture that could not read the server is
    # "unknown" and passes; only a read that shows the difference refuses.
    if arm and arm.get("status") == "mismatch":
        return _not_measured(
            f"arm differed from claim: {arm.get('reason', '')} — iteration NOT counted",
            best, intent, correctness_passed,
        )

    # A bundle is gated as one change only when it is the bundle it names: every
    # member set and live. Otherwise the number belongs to some other set.
    if bundle and not bundle.get("ok"):
        return _not_measured(f"{bundle.get('reason', 'bundle not proven')} — iteration NOT counted",
                             best, intent, correctness_passed)

    # A read-only LEARN skips the server run entirely, so it also skips
    # correctness — there is no serving path to check. The caller is
    # responsible for only marking one read-only when the diff is confined to
    # inert paths; see _probe_is_readonly in the loop.
    if intent.skips_server_when_readonly and bench_result.get("status") == "skipped":
        return GateResult(
            KEEP, 0.0, best, 0.0, True, True,
            "read-only probe — no server run required", intent,
        )

    if not correctness_passed:
        err = correctness_result.get("error", "unknown")
        if _correctness_is_infrastructure(err):
            # The server was never reachable under the name the check used.
            # Iterations 5 and 8 lost a working FP8 o-projection port to
            # exactly this: a 404 for `/model_weights` against a Rust frontend
            # that only knew the HF name.
            return _not_measured(
                f"correctness could not reach the model ({err}) — harness "
                f"failure, iteration NOT counted",
                best, intent, correctness_passed=False,
            )
        return GateResult(
            REVERT, 0.0, best, 0.0, False, True,
            f"correctness failed: {err}", intent,
        )

    # ── Accuracy ────────────────────────────────────────────────────────────
    # A change that can move numerics is held to the full eval set against the
    # baseline's own score: the per-iteration check is only a floor, and a
    # throughput number cannot see a drop (numerics-knobs.md §0). A score below
    # the bar is a result about the change; an eval that could not produce one
    # is not.
    if accuracy and accuracy.get("triggered"):
        score, gold = accuracy.get("score"), accuracy.get("golden")
        if accuracy.get("error") and not score:
            return _not_measured(
                f"accuracy eval did not run ({accuracy['error']}) — iteration NOT counted",
                best, intent)
        if not accuracy.get("measurable", True) or not isinstance(score, (int, float)):
            return _not_measured(
                "this pack cannot measure accuracy: its correctness output has no score "
                "— iteration NOT counted", best, intent)
        if gold is None:
            return _not_measured(
                f"accuracy: {accuracy.get('source', 'no golden score')} to compare "
                f"{score} with — iteration NOT counted", best, intent)
        tol = float(accuracy.get("tolerance") or 0.0)
        if score < gold - tol:
            by = ", ".join(accuracy.get("by") or [])
            return GateResult(
                REVERT, 0.0, best, 0.0, False, True,
                f"accuracy: {score} < {gold} − {tol} ({accuracy.get('source', '')}; "
                f"full GSM8K, for {by})", intent,
            )

    if bench_result.get("status") != "ok":
        return _not_measured(
            f"bench did not run ({bench_result.get('error', 'unknown')}) — "
            f"iteration NOT counted", best, intent,
        )

    if metric_key not in bench_result:
        return _not_measured(
            f"bench output missing metric key '{metric_key}' — validator "
            f"script is broken, iteration NOT counted", best, intent,
        )

    metric_value = bench_result[metric_key]
    if not isinstance(metric_value, (int, float)) or isinstance(metric_value, bool):
        return _not_measured(
            f"metric '{metric_key}' is not numeric: {metric_value!r} — "
            f"iteration NOT counted", best, intent,
        )

    # Zero is a broken bench for a perf campaign, but it is the ORDINARY
    # negative result for find_witness ("no reproducer this time"), which must
    # stay a REVERT rather than being retried forever as not-measured.
    if metric_value <= 0 and kind != "find_witness":
        n_ok = bench_result.get("num_requests_successful", -1)
        detail = f"num_requests_successful={n_ok}" if n_ok >= 0 else "no traffic"
        return _not_measured(
            f"metric '{metric_key}'={metric_value} — bench returned no traffic "
            f"({detail}), iteration NOT counted", best, intent,
        )

    # ── Measured ────────────────────────────────────────────────────────────
    sec_value = bench_result.get(sec_key) if sec_key else None
    if not isinstance(sec_value, (int, float)) or isinstance(sec_value, bool):
        sec_value = None

    pri = _measure_axis(metric_value, best, direction, threshold,
                        noise_band, metric_key)
    sec = _measure_axis(sec_value, secondary_best, sec_direction,
                        sec_threshold, sec_noise, sec_key)

    def _result(decision, reason, *, booked=False, frontier=False) -> GateResult:
        return GateResult(
            decision, metric_value, best, pri.delta or 0.0,
            True, True, reason, intent,
            revert_tree=True, booked=booked,
            secondary_value=sec_value, secondary_best=secondary_best,
            secondary_delta_pct=sec.delta, frontier=frontier,
            # Computed here rather than at each REVERT branch so it cannot be
            # forgotten by one of them. `pri.ok and not sec.ok` is exactly "the
            # primary had no objection and the secondary did".
            secondary_decided=(decision == REVERT and pri.ok and not sec.ok),
        )

    if kind == "find_witness":
        if metric_value > 0:
            return _result(KEEP, "witness found", booked=True)
        return _result(REVERT, "no witness")

    if best is None:
        return _result(KEEP, "first measurement — accepted as baseline",
                       booked=True)

    # ── Pareto shape, computed once and shared by every class ───────────────
    # The leaderboard this campaign targets is a throughput-vs-interactivity
    # CURVE. A change that buys one axis by spending the other is a new point
    # on the frontier, not a failure — a single-axis gate cannot see that and
    # reverted the same FP8 o-projection port twice.
    both_ok = pri.ok and sec.ok
    is_frontier = (pri.gain and not sec.ok) or (sec.gain and not pri.ok)
    delta_s = f"{pri.delta:+.1f}%{sec.note()}"

    # ── Cap on what a frontier trade may spend ──────────────────────────────
    # `secondary_noise_band_pct` does NOT bound a frontier point: once the
    # primary clears `threshold_rel`, is_frontier fires and the band is never
    # consulted, so a +1% primary win could be kept at ANY secondary cost.
    # That is the right default for a genuine Pareto curve — the frontier
    # branch exists because a single-axis gate reverted the same FP8
    # o-projection port twice — but it is unbounded, and an unbounded trade is
    # indistinguishable from a broken measurement.
    #
    # The budget is EARNED, not flat. A trade may spend
    #
    #     allowed_loss = min(secondary_regression_cap_pct,
    #                        secondary_exchange_rate * primary_gain_pct)
    #
    # so the size of the primary win sets how much of the second axis it may
    # buy, and an absolute ceiling still bounds the whole thing. A flat cap got
    # this wrong in both directions at once: set low enough to refuse a
    # marginal +1% win spending 5%, it also refused a decisive +8% win spending
    # 5%, which is a real point on the frontier.
    #
    # `secondary_exchange_rate` defaults to 1.0 when a cap is set: never give
    # up more percent of the second axis than you gained on the first. Set it
    # >1 to allow a primary-favouring trade, <1 to demand the primary win
    # dominate. The absolute cap is what still catches the pathological shape —
    # at C4, output_throughput == sum(OSL)/duration/4 exactly, so a large
    # divergence between the axes means the run served a different token total,
    # and the first suspect is the DRAW, not the code.
    #
    # Absent (None), behaviour is exactly as before, so packs that do not set a
    # cap are unaffected.
    #
    # NOTE ON WHAT THIS IS NOT: it is not a per-iteration allowance. Both bars
    # are HIGH-WATER MARKS (state.record_keep takes max()), so every accepted
    # tree is scored against the best secondary ever seen, and cumulative drift
    # is bounded by the cap rather than accumulating cap% per iteration.
    sec_cap = measurement.get("secondary_regression_cap_pct")
    sec_rate = measurement.get("secondary_exchange_rate", 1.0)
    sec_allowed = None
    if sec_cap is not None:
        sec_allowed = abs(float(sec_cap))
        if pri.gain and pri.delta is not None and pri.delta > 0:
            sec_allowed = min(sec_allowed, abs(float(sec_rate)) * pri.delta)
    sec_regression_capped = (
        sec_allowed is not None
        and sec.delta is not None
        and sec.delta < -sec_allowed
    )

    def _frontier_reason() -> str:
        won, lost = (pri, sec) if pri.gain else (sec, pri)
        return (f"frontier point: {won.key} {won.delta:+.1f}% bought at "
                f"{lost.key} {lost.delta:+.1f}%")

    def _cap_reason() -> str:
        return (f"{delta_s} — {sec.key} regressed beyond the "
                f"{abs(float(sec_cap)):.1f}% cap on a frontier trade; a "
                f"primary gain does not license an unbounded secondary loss. "
                f"Check the draw fingerprint before re-running.")

    # A correctness fix carries no performance claim on EITHER axis, so it is
    # not judged on one and the Pareto shape never decides it. Only a
    # catastrophe refuses it.
    if intent.pareto_exempt:
        if _within_noise(metric_value, best, direction, intent_tolerance):
            return _result(KEEP,
                           f"correctness fix ships regardless of throughput "
                           f"({delta_s}, cap {intent_tolerance:.1f}%)",
                           booked=intent.books and pri.gain)
        return _result(REVERT,
                       f"{delta_s} — beyond the {intent_tolerance:.1f}% "
                       f"catastrophe cap even for a correctness fix")

    # A probe makes no metric claim. It only has to not leave the tree worse,
    # and a frontier trade is not "worse" — it is a different point.
    if not intent.books:
        if both_ok:
            return _result(KEEP, f"probe recorded, {delta_s} "
                                 f"(within {noise_band:.1f}% noise band)")
        if is_frontier:
            if sec_regression_capped:
                return _result(REVERT, _cap_reason())
            return _result(KEEP, f"probe recorded, {_frontier_reason()}",
                           frontier=True)
        return _result(REVERT, f"probe regressed {delta_s} — beyond the "
                               f"{noise_band:.1f}% noise band")

    if is_frontier:
        if sec_regression_capped:
            return _result(REVERT, _cap_reason())
        return _result(KEEP, _frontier_reason(), frontier=True)

    if both_ok and (pri.gain or sec.gain):
        return _result(KEEP, f"{delta_s} — cleared the bar ({threshold}x)",
                       booked=intent.books and pri.gain)

    if both_ok:
        # Neutral. Work that ships on evidence is kept; a metric claim is not,
        # because it made a claim it did not meet.
        if not intent.requires_bar:
            return _result(KEEP, f"neutral ({delta_s}) — class A ships on "
                                 f"evidence")
        # Two different shapes reach here and they are not the same news. A
        # move that cleared the noise band but not the bar is a real upward
        # reading that this campaign declines to buy; calling that "inside the
        # noise band" misreports the measurement and reads as a null. ONE bar
        # keeps and books (see the module docstring), so the honest sentence is
        # that it fell short of the bar, not that nothing happened.
        if pri.delta is not None and pri.delta > noise_band:
            return _result(REVERT, f"{delta_s} — short of the {threshold}x "
                                     f"bar; one bar both keeps and books")
        return _result(REVERT, f"{delta_s} — inside the noise band, no "
                                 f"measurable gain on either axis")

    return _result(REVERT, f"{delta_s} — regressed beyond the noise band "
                             f"with no compensating gain")
