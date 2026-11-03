"""Resumable campaign state with enforced escalation.

Serialized as ``session.json`` — backward-compatible with ``evals/collect.py``.
Escalation rules are deterministic code, not prompt suggestions.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from arbor import rebaseline

DEFAULT_LEVELS = ["config", "framework", "kernel", "feature"]

# Escalation thresholds — these are hard rules, not suggestions.
CONFIG_MAX_ITERS = 3
ESCALATE_NULL_STREAK = 3

# A PROBE buys knowledge, not throughput. Capping consecutive probes keeps the
# loop from converting an optimization campaign into a survey: after this many
# in a row the next iteration must carry a real change.
MAX_CONSECUTIVE_PROBES = 2

# When the loop has produced this many nulls in a row AND has no tier left to
# escalate into, it does not know what to do next — which is precisely when it
# should buy knowledge rather than guess again. At that point a probe is
# MANDATED, not merely permitted. The dsv41flash campaign sat at null_streak 7
# on the terminal tier with 30 unresolved ledger findings and never probed,
# because every counter it could see treated probing as failure.
PROBE_MANDATE_NULL_STREAK = 3

# Consecutive not-measured iterations that mean the harness, not the campaign,
# is broken. The loop stops rather than burning an allocation retrying.
MAX_CONSECUTIVE_INVALID = 3

# Consecutive REVERTs decided by the SECOND axis alone -- primary inside its
# noise band, secondary past its band -- that mean the secondary incumbent is
# suspect rather than the changes.
#
# WHY THIS COUNTER EXISTS. `best_secondary` is a high-water mark over
# single-replay measurements, so it is an UPWARD-BIASED estimator: each ratchet
# keeps whichever draw happened to be high, and the bias grows with the number
# of ratchet events. The cuda-graph campaign ratcheted three times and landed
# 1.78% above its own matched-draw cohort, which ate 71% of a 2.5% secondary
# band before any real effect. Iterations 41, 44, 46 and 48 were then each
# REVERTED on the secondary axis alone -- 46 was a BYTE-IDENTICAL tree.
#
# The inference is statistical, not about any one change: several UNRELATED
# changes reading null on the primary and failing only on the secondary is far
# better explained by a high secondary incumbent than by that many real
# secondary-only regressions. One such iteration is ordinary; three is a
# statement about the instrument.
#
# It deliberately does NOT auto-correct the bar. Moving an incumbent without a
# measurement is how a campaign talks itself into a number no tree produces.
# What it does is make the situation VISIBLE and mandate the cheap experiment
# that settles it -- a null re-bench of the unchanged tree. The cuda-graph
# campaign reached that experiment on its own at iteration 46, but only because
# that agent was exceptional; nothing in the harness asked for it.
SECONDARY_SUSPECT_STREAK = 3

# ── The BARREN knob ─────────────────────────────────────────────────────────
# Iterations since the last BOOKED advance on either axis. See
# docs/barren-and-learn-design.md for the full argument; the short version:
#
# `null_streak` is wired to three consumers that want different things —
# escalation, the LEARN mandate, and a reset from LEARN-with-findings. That
# third wire is correct (without it the old dsv41flash campaign sat at
# null_streak 7 with 30 unresolved findings and never probed) but it makes the
# counter unable to answer "are we converting effort into wins?". The 2b
# campaign went 31 of 63 iterations without moving either bar while
# `null_streak` read 0, because a REVERT raised it to 1 and the next
# LEARN — closing 7-13 genuine findings — reset it to 0. It never reached
# ESCALATE_NULL_STREAK, so nothing was ever rendered into the prompt.
#
# So: one counter per question. `null_streak` keeps its meaning ("is this tier
# dry?"), which foundation work legitimately answers. `barren_streak` answers
# "is the campaign's picture of where the cost is still correct?", and NOTHING
# resets it but a real advance.
BARREN_NOTICE = 8
BARREN_ACT = 16

# The one wire between the two knobs. A drought WIDENS the LEARN budget rather
# than tightening it, which is the correction: today the system rations
# exploration hardest at the moment exploitation has proven least productive
# (2b sat at probe_streak 2 — budget exhausted — 29 iterations into its
# drought). A LEARN touching only `eval/` and the ledger skips the server run,
# so this is nearly free in wall-clock.
#
# It is not circular: LEARN resets `null_streak` but NEVER `barren_streak`, so
# a campaign can spend every granted LEARN while the drought counter keeps
# climbing. Exploration is made cheap, never self-justifying.
LEARN_BUDGET_BARREN = 4


@dataclass
class Policy:
    """Every pace threshold in one place, overridable per pack.

    Defaults are the module constants above, so a pack that declares nothing
    behaves exactly as before. Persisted into session.json so a resume
    reproduces the regime it started under and the monitor can report which
    regime a run was on — five of the six pre-existing thresholds were the
    integer 3 meaning five unrelated things, none of them configurable.
    """

    config_max_iters: int = CONFIG_MAX_ITERS
    escalate_null_streak: int = ESCALATE_NULL_STREAK
    max_consecutive_probes: int = MAX_CONSECUTIVE_PROBES
    probe_mandate_null_streak: int = PROBE_MANDATE_NULL_STREAK
    max_consecutive_invalid: int = MAX_CONSECUTIVE_INVALID
    secondary_suspect_streak: int = SECONDARY_SUSPECT_STREAK
    barren_notice: int = BARREN_NOTICE
    barren_act: int = BARREN_ACT
    learn_budget_barren: int = LEARN_BUDGET_BARREN
    # Map rounds (arbor/directions.py, which documents them); map_rounds 0 turns them off.
    map_rounds: int = 1
    map_worked_flag: int = 4
    map_update_max_iters: int = 8
    map_bootstrap_waiver: int = 2
    map_round_additions: int = 3
    # Does the gate hold the live server to the arm's claim (arbor/measure/armcheck.py):
    # "off", "record" (shown to the agent and the critic; the verdict stands) or "enforce"
    # (a mismatch is NOT_MEASURED).
    arm_check: str = "record"
    # May the agent claim a catalogued bundle (serving-playbook knobs.json) as one change,
    # with `BUNDLE: <id>` (arbor/measure/bundles.py)? Off: a declared bundle is gated as an
    # ordinary change.
    bundles: bool = False
    # Is a change to a numerics knob (knobs.json), or one the agent marks `NUMERICS: yes`,
    # held to the full GSM8K set against the baseline's score minus the tolerance
    # (arbor/measure/accuracy.py)? 0.015 is about 2 sigma at n = 1319.
    accuracy_gate: bool = False
    accuracy_tolerance: float = 0.015

    @classmethod
    def from_raw(cls, raw: Any) -> "Policy":
        """Build from a dict, ignoring unknown keys. Accepts a Policy or None.

        Each value is cast to its field's type, so a pack's `"6"` is an int and
        its `"enforce"` stays a string.
        """
        if isinstance(raw, cls):
            return raw
        if not isinstance(raw, dict):
            return cls()
        fields = cls.__dataclass_fields__
        return cls(**{k: _cast(fields[k].default, v) for k, v in raw.items()
                      if k in fields and v is not None})


def _cast(default: Any, value: Any) -> Any:
    if isinstance(default, bool):   # before int: a bool is an int
        return value if isinstance(value, bool) else str(value).strip().lower() in ("1", "true", "yes", "on")
    return type(default)(value)


@dataclass
class CampaignState:
    identity: str = ""
    pack_name: str = ""
    goal: str = ""
    baseline: float | None = None
    best_metric: float | None = None
    # Second Pareto axis (e.g. p90_interactivity). None when the pack declares
    # no secondary metric, in which case every gate path is single-axis.
    baseline_secondary: float | None = None
    best_secondary: float | None = None
    iteration: int = 0
    tier: str = "config"
    null_streak: int = 0
    tier_iters: int = 0
    probe_streak: int = 0
    probe_iters: int = 0
    invalid_streak: int = 0
    invalid_iters: int = 0
    # Consecutive REVERTs the SECOND axis decided on its own. See
    # SECONDARY_SUSPECT_STREAK. Defaults to 0 so an older session.json loads
    # unchanged and starts counting from where it resumes.
    secondary_only_streak: int = 0
    # Iterations since the last BOOKED advance on either axis. See the BARREN
    # block above. Defaults to 0 so an older session.json loads unchanged and
    # starts counting from where it resumes.
    barren_streak: int = 0
    # When the iteration being recorded started (set by run_iteration, used once by
    # _lesson): every gate row carries its own wall time, which the map sums per
    # direction (arbor/directions.py). 0 when nothing set it.
    iteration_started: float = 0.0
    # What the iteration being recorded adds to its row (the arm check, ...), set by
    # run_iteration and used once by _lesson, like iteration_started.
    iteration_notes: dict[str, Any] = field(default_factory=dict)
    current_env: dict[str, str] = field(default_factory=dict)
    current_flags: list[str] = field(default_factory=list)
    current_patches: list[str] = field(default_factory=list)
    lessons: list[dict[str, Any]] = field(default_factory=list)
    validator_digests: dict[str, str] = field(default_factory=dict)
    start_epoch: float = field(default_factory=time.time)
    metric_name: str = ""
    metric_direction: str = "max"
    secondary_name: str = ""
    secondary_direction: str = "max"
    metric_unit: str = ""
    workload_kind: str = ""
    policy: Policy = field(default_factory=Policy)
    # "standalone" or "component" (arbor/component.py): set at init from the config,
    # so a resume keeps it; the driver refuses a config whose mode differs. A session
    # from before modes recorded none: it loads as standalone, and the driver records
    # whichever mode first resumes it.
    mode: str = "standalone"
    # The human's approval at the review gate (arbor/review.py): who, when, the note and the
    # files changed since the request. Empty when the campaign started without the gate.
    review: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> CampaignState:
        p = Path(path)
        if not p.exists():
            return cls()
        raw = json.loads(p.read_text())
        if "best_tput" in raw and "best_metric" not in raw:
            raw["best_metric"] = raw.pop("best_tput")
        known = {f.name for f in cls.__dataclass_fields__.values()}
        kw = {k: v for k, v in raw.items() if k in known}
        # Serialized as a plain dict by asdict(); rebuild it rather than
        # letting a bare dict reach the threshold accessors.
        kw["policy"] = Policy.from_raw(kw.get("policy"))
        return cls(**kw)

    @classmethod
    def from_campaign(cls, campaign) -> CampaignState:
        """Create initial state from a campaign. The only way to start a new campaign."""
        obj = campaign.pack.objective
        return cls(
            identity=campaign.identity,
            pack_name=campaign.pack.name,
            goal=campaign.goal,
            metric_name=obj.get("metric", ""),
            metric_direction=obj.get("direction", "max"),
            secondary_name=obj.get("measurement", {}).get("secondary_metric", ""),
            secondary_direction=obj.get("measurement", {}).get(
                "secondary_direction", "max"),
            metric_unit=obj.get("unit", ""),
            workload_kind=campaign.pack.workload.get("kind", ""),
            # Config wins over pack, pack wins over the module defaults, so a
            # single campaign can be re-regimed without editing a shared pack.
            policy=Policy.from_raw({
                **(getattr(campaign.pack, "policy", None) or {}),
                **(campaign.config.get("policy") or {}),
            }),
            mode=getattr(campaign, "mode", "standalone"),
        )

    def record_baseline(self, metric: float) -> None:
        """Record the baseline measurement. Must be called before any iterations."""
        self.baseline = metric
        self.best_metric = metric
        self.lessons.append({
            "iteration": 0,
            "decision": "BASELINE",
            "metric": metric,
            "change": "baseline measurement",
            "tier": self.tier,
            "ts": time.time(),
        })

    def save(self, path: str | Path) -> None:
        """Write session.json atomically: a temp file beside it, then a rename.

        Other processes read this file while a campaign runs (a parallel run's
        coordinator, rebase_bar, the dashboards). A plain write_text lets them
        see a half-written file and fail on JSONDecodeError.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2) + "\n")
        os.replace(tmp, p)

    def check_escalation(self, levels: list[str] | None = None) -> str | None:
        """Return the tier the campaign MUST move to, or None if no escalation needed.

        This is deterministic code — the agent cannot override it.
        """
        levels = levels or DEFAULT_LEVELS
        idx = levels.index(self.tier) if self.tier in levels else 0

        # Config tier is warmup — force out after CONFIG_MAX_ITERS iterations
        if self.tier == "config" and self.tier_iters >= self.policy.config_max_iters:
            if idx + 1 < len(levels):
                return levels[idx + 1]

        # Any tier: consecutive failures force escalation
        if self.null_streak >= self.policy.escalate_null_streak:
            if idx + 1 < len(levels):
                return levels[idx + 1]

        return None

    def apply_escalation(self, new_tier: str) -> None:
        """Force-move to a new tier. Resets streak counters."""
        self.tier = new_tier
        self.null_streak = 0
        self.tier_iters = 0
        self.lessons.append({
            "iteration": self.iteration,
            "decision": "ESCALATE",
            "change": f"forced escalation to {new_tier}",
            "tier": new_tier,
            "ts": time.time(),
        })

    def apply_map_round(self, round_no: int, phase: str, change: str) -> None:
        """A map-round event (arbor/directions.py), as an annotation row.

        ``start`` begins a new map like a new campaign: the streaks that measure
        how long this map has gone without an advance start again. The tier,
        the not-measured streak and the secondary-only streak carry on: they
        describe the harness and the incumbent, not the map.
        """
        if phase == "start":
            self.null_streak = 0
            self.barren_streak = 0
            self.probe_streak = 0
        self.lessons.append({
            "iteration": self.iteration,
            "decision": "MAP_ROUND",
            "map_round": round_no,
            "phase": phase,
            "change": change,
            "tier": self.tier,
            "ts": time.time(),
        })

    # ── The recorder ────────────────────────────────────────────────────────
    # One iteration, one call. What varies is (intent, verdict), and the four
    # named methods below are thin wrappers that name a common pair.
    #
    # THE ASYMMETRIES ARE REAL — this table is the specification, and three of
    # its cells are the whole reason this is not a for-loop over a dict:
    #
    #                    KEEP(books)   REVERT    KEEP(probe)   NOT_MEASURED
    #   invalid_streak   = 0           = 0       = 0           += 1
    #   null_streak      0 if advanced += 1      0 if findings  UNTOUCHED
    #   probe_streak     = 0           = 0       += 1           UNTOUCHED
    #   tier             set           NOT SET   set            set
    #   tier_iters       += 1          += 1      += 1           NOT INCREMENTED
    #   best_metric      ratcheted     —         —              —
    #   lesson metric    value         value     value          None
    #
    # NOT_MEASURED freezing every counter is deliberate and load-bearing (see
    # record_invalid's docstring). REVERT not assigning `tier` is NOT obviously
    # deliberate — every caller passes the tier the campaign is already on, so
    # it has never mattered — but it is preserved verbatim here. Changing
    # behaviour inside a refactor is how a "cleanup" silently moves a verdict.

    def record(
        self,
        verdict: str,
        *,
        iteration: int,
        change: str,
        tier: str,
        intent: "Intent | str | None" = None,
        metric: float | None = None,
        reason: str = "",
        findings: list[str] | None = None,
        advanced_metric: bool = True,
        secondary: float | None = None,
        frontier: bool = False,
        secondary_decided: bool = False,
    ) -> None:
        """Record one iteration's outcome. See the table above."""
        from arbor.measure.intents import (
            KEEP, NOT_MEASURED, REVERT, VERDICTS, legacy_decision, resolve,
        )

        # Accept the legacy spellings, whose meaning is unambiguous, and REFUSE
        # anything else. An unrecognised verdict used to fall through to the
        # REVERT branch, which would silently discard a working tree on a typo
        # — the exact failure class NOT_MEASURED was introduced to stop. A
        # programming error must be loud and leave the tree alone.
        verdict = (verdict or "").strip().upper()
        if verdict == "INVALID":
            verdict = NOT_MEASURED
        elif verdict == "PROBE":
            # The legacy `decision` PROBE carried BOTH axes at once: it meant
            # KEEP *and* it meant the intent was a probe. Mapping only the
            # verdict half would let `record("PROBE", intent="B")` ratchet
            # best_metric on a survey — the one thing record_probe exists to
            # prevent ("Deliberately does NOT touch best_metric"). Restore both
            # halves, and ignore any conflicting intent rather than honouring
            # it, because the caller spelled the intent in the verdict.
            verdict, intent = KEEP, "PROBE"
        if verdict not in VERDICTS:
            raise ValueError(
                f"unknown verdict {verdict!r}; expected one of {VERDICTS}"
            )

        intent_row = resolve(intent)
        decision = legacy_decision(intent_row, verdict)
        findings = findings or []

        # `is None`, not truthiness: 0.0 is a legitimate reading and a
        # falsy test would silently rewrite it.
        value = 0.0 if metric is None else metric

        # A run of secondary-only REVERTs is evidence about the INCUMBENT, so
        # it must survive only while that evidence is uninterrupted. A KEEP
        # means the bar was beaten and the suspicion is answered; a REVERT the
        # primary decided breaks the chain. NOT_MEASURED leaves it alone,
        # because nothing was learned either way -- the same rule every other
        # counter follows.
        if verdict == KEEP:
            self.secondary_only_streak = 0
        elif verdict == REVERT:
            self.secondary_only_streak = (
                self.secondary_only_streak + 1 if secondary_decided else 0
            )

        # ── The BARREN knob ──────────────────────────────────────────────
        # Only a real advance clears it. A LEARN that closed a dozen findings
        # does not, which is the entire point: `null_streak` already credits
        # that, and crediting it twice is how 2b spent 31 iterations barren
        # while every visible counter read zero.
        #
        # FIX-CORRECTNESS is EXEMPT (not merely reset-proof): a correctness
        # fix that costs throughput is still the right move, and counting it
        # as drought would pressure the agent away from fixing bugs.
        # NOT_MEASURED freezes it, as it freezes every other counter — nothing
        # was learned about the candidate either way.
        # THE RATCHET IS AUTHORITATIVE, not `advanced_metric`. A neutral KEEP
        # normally reads below the incumbent and is a drought iteration, but it
        # CAN set a new high-water: 2d's iteration 32 shipped `booked=False`
        # and still moved both bars (291.44 over 290.31, 53.43 over 52.93),
        # because _record_keep ratchets whatever it is handed. Calling that
        # barren would have the prompt assert "no advance in 11 iterations"
        # while the monitor showed the bar moving 7 iterations ago. The drought
        # is a statement about THE BARS; if a bar moved, it is over.
        if verdict != NOT_MEASURED and intent_row.name != "FIX-CORRECTNESS":
            booked = bool(
                verdict == KEEP and intent_row.books
                and (frontier or self._would_ratchet(value, secondary))
            )
            self.barren_streak = 0 if booked else self.barren_streak + 1

        if verdict == NOT_MEASURED:
            self._record_not_measured(iteration, change, tier, reason,
                                      intent_row, decision, findings)
        elif verdict == KEEP and not intent_row.books:
            self._record_probe(iteration, change, tier, reason, value,
                               intent_row, decision, findings)
        elif verdict == KEEP:
            self._record_keep(iteration, value, change, tier,
                              intent_row, decision, findings, advanced_metric,
                              secondary, frontier)
        else:
            self._record_revert(iteration, value, change, tier, reason,
                                intent_row, decision, findings)

    def _would_ratchet(self, metric: float | None,
                       secondary: float | None) -> bool:
        """Would this reading move EITHER high-water bar?

        Called BEFORE the bars are mutated, so it must not be moved below the
        _record_* dispatch. Either axis counts: the two peak at different
        iterations (v2's throughput peaked at 55, its interactivity at 58), and
        tying them together would lose every second-axis-only advance.
        """
        def better(v, best, direction):
            if v is None:
                return False
            if best is None:
                return True
            return v < best if direction == "min" else v > best

        return (better(metric, self.best_metric, self.metric_direction)
                or better(secondary, self.best_secondary,
                          self.secondary_direction or "max"))

    def _lesson(self, **kw) -> None:
        kw["ts"] = time.time()
        # Stamp the regime on every row so a drought is legible after the fact
        # from session.json alone, without re-deriving it from the bars.
        kw.setdefault("barren", self.barren_streak)
        # And the factors that put this row's numbers on the campaign's first node's
        # scale (arbor/rebaseline.py): a rolling chunk rebases the bar on its node, and
        # the dashboard reads every row on one scale. The collector ships to the login
        # node as one ssh command, with no room left to compute them there.
        kw["scale"], kw["secondary_scale"] = rebaseline.current(self.lessons)
        # The iteration's wall time, start to verdict. Used once: a row recorded by a
        # path that set no start (a CLI gate, a test) carries none, never a stale one.
        if self.iteration_started and kw["ts"] >= self.iteration_started:
            kw["started"] = round(self.iteration_started, 1)
            kw["wall_s"] = round(kw["ts"] - self.iteration_started, 1)
        self.iteration_started = 0.0
        for k, v in self.iteration_notes.items():
            kw.setdefault(k, v)
        self.iteration_notes = {}
        self.lessons.append(kw)

    def _record_keep(
        self, iteration, metric, change, tier, intent, decision, findings,
        advanced_metric, secondary, frontier,
    ) -> None:
        self.invalid_streak = 0
        if self.baseline is None:
            self.baseline = metric
        # A frontier point trades one axis for the other, so it must NOT ratchet
        # EITHER bar. Ratcheting only the axis that won would leave the next
        # change competing against a composite point that no single
        # configuration ever achieved, and the bar would drift out of reach one
        # trade at a time.
        if self.best_metric is None:
            self.best_metric = metric
        elif frontier:
            pass
        elif self.metric_direction == "min":
            self.best_metric = min(self.best_metric, metric)
        else:
            self.best_metric = max(self.best_metric, metric)
        if secondary is not None and not frontier:
            if self.baseline_secondary is None:
                self.baseline_secondary = secondary
            if self.best_secondary is None:
                self.best_secondary = secondary
            elif (self.secondary_direction or "max") == "min":
                self.best_secondary = min(self.best_secondary, secondary)
            else:
                self.best_secondary = max(self.best_secondary, secondary)
        # A class-A KEEP that landed NEUTRAL shipped on evidence, not on the
        # metric — so it is not progress against the objective and must not
        # reset the escalation counter. Otherwise a campaign can ship neutral
        # foundation steps indefinitely and never escalate, which is exactly
        # the stuck state null_streak exists to detect. record_probe already
        # takes this care; class A needs it too.
        if advanced_metric:
            self.null_streak = 0
        else:
            self.null_streak += 1
        self.probe_streak = 0
        self.iteration = iteration
        self.tier = tier
        self.tier_iters += 1
        self._lesson(
            iteration=iteration,
            decision=decision,
            verdict="KEEP",
            metric=metric,
            secondary=secondary,
            frontier=frontier,
            booked=advanced_metric,
            change=change,
            tier=tier,
            claim_class=intent.aliases[0],
            intent=intent.name,
            findings=findings,
        )

    def _record_revert(
        self, iteration, metric, change, tier, reason, intent, decision,
        findings,
    ) -> None:
        self.invalid_streak = 0
        self.null_streak += 1
        self.probe_streak = 0
        self.iteration = iteration
        # NOTE: `self.tier` is deliberately NOT assigned here; see the table
        # above. Preserved from the pre-split implementation.
        self.tier_iters += 1
        self._lesson(
            iteration=iteration,
            decision=decision,
            verdict="REVERT",
            metric=metric,
            change=change,
            tier=tier,
            reason=reason,
            claim_class=intent.aliases[0],
            intent=intent.name,
            findings=findings,
        )

    def _record_probe(
        self, iteration, change, tier, reason, metric, intent, decision,
        findings,
    ) -> None:
        self.invalid_streak = 0
        if findings:
            self.null_streak = 0
        else:
            self.null_streak += 1
        self.probe_streak += 1
        self.probe_iters += 1
        self.iteration = iteration
        self.tier = tier
        self.tier_iters += 1
        self._lesson(
            iteration=iteration,
            decision=decision,
            verdict="KEEP",
            metric=metric,
            change=change,
            tier=tier,
            reason=reason,
            claim_class=intent.aliases[0],
            intent=intent.name,
            findings=findings,
        )

    def _record_not_measured(
        self, iteration, change, tier, reason, intent, decision, findings,
    ) -> None:
        self.invalid_streak += 1
        self.invalid_iters += 1
        self.iteration = iteration
        self.tier = tier
        self._lesson(
            iteration=iteration,
            decision=decision,
            verdict="NOT_MEASURED",
            metric=None,
            change=change,
            tier=tier,
            reason=reason,
            claim_class=intent.aliases[0],
            intent=intent.name,
            findings=findings,
        )

    # ── Named wrappers ──────────────────────────────────────────────────────
    # The pre-split vocabulary, kept so every call site and test reads
    # unchanged. Each names one (intent, verdict) pair.

    def record_keep(
        self,
        iteration: int,
        metric: float,
        change: str,
        tier: str,
        claim_class: str = "B",
        findings: list[str] | None = None,
        advanced_metric: bool = True,
        secondary: float | None = None,
        frontier: bool = False,
    ) -> None:
        self.record(
            "KEEP", iteration=iteration, metric=metric, change=change,
            tier=tier, intent=claim_class, findings=findings,
            advanced_metric=advanced_metric, secondary=secondary,
            frontier=frontier,
        )

    def record_revert(
        self,
        iteration: int,
        metric: float,
        change: str,
        tier: str,
        reason: str,
        claim_class: str = "B",
        findings: list[str] | None = None,
        secondary_decided: bool = False,
    ) -> None:
        self.record(
            "REVERT", iteration=iteration, metric=metric, change=change,
            tier=tier, reason=reason, intent=claim_class, findings=findings,
            secondary_decided=secondary_decided,
        )

    def record_probe(
        self,
        iteration: int,
        change: str,
        tier: str,
        findings: list[str] | None = None,
        reason: str = "",
        metric: float = 0.0,
    ) -> None:
        """A probe resolves an uncertainty; it does not claim the metric.

        Deliberately does NOT touch ``best_metric``. ``probe_streak`` is what
        bounds probing.

        It DOES reset ``null_streak`` when it closed at least one ledger
        finding. The original rule — never reset — was written to stop a
        campaign surveying forever, but combined with the probe cap it made
        every visible counter treat probing as failure: probe_streak is a
        limit, null_streak kept climbing, best_metric could not move. The
        dsv41flash campaign stopped probing after iteration 2 and accumulated
        30 unresolved findings while sitting at null_streak 7. A probe that
        closes a real uncertainty IS progress; a probe that closes nothing is
        still counted as a null, and MAX_CONSECUTIVE_PROBES still bounds the
        survey.
        """
        self.record(
            "KEEP", iteration=iteration, change=change, tier=tier,
            reason=reason, metric=metric, intent="PROBE", findings=findings,
        )

    def record_invalid(
        self,
        iteration: int,
        change: str,
        tier: str,
        reason: str,
        claim_class: str = "B",
        findings: list[str] | None = None,
    ) -> None:
        """The iteration was NOT MEASURED. Nothing about the change is known.

        A bench that never ran, a server that never came up, a 404 from the
        wrong wire name, a pinned instrument that moved — none of these are
        evidence about the candidate. Reporting them as REVERT discarded
        working code, incremented ``null_streak``, and drove tier escalation
        off manufactured nulls. Six of eleven REVERTs in the dsv41flash
        campaign were this, including five attempts at one FP8 o-projection
        port that was correct the whole time.

        So: the tree is NOT reverted, ``null_streak`` does NOT move, the tier
        counters do NOT move, and the caller retries. Only ``invalid_streak``
        advances, so a systemically broken harness still stops the loop.
        """
        self.record(
            "NOT_MEASURED", iteration=iteration, change=change, tier=tier,
            reason=reason, intent=claim_class, findings=findings,
        )

    def secondary_bar_suspect(self, limit: int | None = None) -> bool:
        """True when a run of REVERTs the second axis decided alone says the
        SECONDARY INCUMBENT is more likely wrong than the changes are.

        Deliberately not a correction. The only honest way to move an incumbent
        is to measure it: bench the CURRENT tree unchanged and compare. That
        run is cheap relative to what it settles — the cuda-graph campaign spent
        four iterations (38, 41, 44, 48) being refused on this axis before one
        agent thought to try it at iteration 46, and the answer was that a
        byte-identical tree read -5.1%.
        """
        n = self.policy.secondary_suspect_streak if limit is None else limit
        return n > 0 and self.secondary_only_streak >= n

    def harness_broken(self) -> bool:
        """True when consecutive not-measured iterations mean: stop and look."""
        return self.invalid_streak >= self.policy.max_consecutive_invalid

    # ── The BARREN knob ─────────────────────────────────────────────────────

    def barren_stage(self) -> str:
        """``"clear"`` | ``"notice"`` | ``"act"`` — the drought regime.

        A threshold of 0 disables its stage, so a pack can opt out of either
        half without the harness growing a second switch.
        """
        p = self.policy
        if p.barren_act > 0 and self.barren_streak >= p.barren_act:
            return "act"
        if p.barren_notice > 0 and self.barren_streak >= p.barren_notice:
            return "notice"
        return "clear"

    def learn_budget(self) -> int:
        """Consecutive LEARNs allowed right now.

        THE ONE WIRE between the two knobs. A drought WIDENS this rather than
        tightening it. That is counterintuitive and it is the correction: the
        system's only existing answer to no-progress is pressure to ship,
        which is an exploitation signal, and if N iterations of exploitation
        produced nothing then more exploitation pressure is exactly wrong.
        """
        p = self.policy
        return (p.learn_budget_barren if self.barren_stage() != "clear"
                else p.max_consecutive_probes)

    def probes_exhausted(self) -> bool:
        """True when the next iteration must carry a real change, not a probe."""
        return self.probe_streak >= self.learn_budget()

    def probe_mandated(self, levels: list[str] | None = None) -> bool:
        """True when the loop should buy knowledge instead of guessing again.

        ``levels`` is accepted and ignored, for call-site compatibility. It
        used to gate this on ALSO sitting at the terminal tier, on the theory
        that escalation is the loop's normal answer to being stuck and the
        mandate is for when that answer is exhausted. In practice every live
        pack declares ``space.levels: None``, so the tier half rarely fired and
        the mandate was mostly dead code. The streak half carries the rule.
        """
        if self.probes_exhausted():
            return False
        return self.null_streak >= self.policy.probe_mandate_null_streak

    def diagnose(self) -> str:
        """One line naming the regime, for the prompt header and the monitor.

        2b's 31-iteration drought was invisible because every counter anyone
        looked at faithfully reported `null_streak=2`.
        """
        return (f"iteration {self.iteration} | tier {self.tier} | "
                f"barren {self.barren_streak} ({self.barren_stage()}) | "
                f"null {self.null_streak} | "
                f"LEARN {self.probe_streak}/{self.learn_budget()} | "
                f"secondary-only {self.secondary_only_streak}")
