"""Action scoring — heuristic priors + dynamics consumed by the Orchestration prompt.

This module is intentionally pure: no I/O, no logging, no global state. It
holds the data class (:class:`ActionScore`) that the Coordinator persists in
``SharedState.action_scores`` plus the helpers used to seed, update, and rank
those scores.

Design (see plan ``action-scoring-in-shared-state``):

* Each enabled action has one ``ActionScore`` record.
* ``base_score`` is seeded once at session start. Default is auto-computed
  from :class:`ActionMetadata` (expected_gain midpoint / cost_p50, discounted
  by accuracy & crash risks). When the model_class has a marathon prior
  (table from ``/wekafs/zgong/TBO/inference_optimization/marathon/skills/SKILL.md``
  L832–839) we use that instead — marathon's priors encode lessons from
  past long-run experiments.
* ``score_mult`` evolves as the Coordinator processes KEEP/DISCARD outcomes.
* ``effective_score`` combines ``base_score * score_mult * risk_discounts *
  target_gap_multiplier`` plus a UCB-style exploration bonus and a linear
  aging bonus. Locked or cooldown'd actions return -1.0 so they sort to the
  bottom and the prompt renderer can mark them as unavailable.

Anti-loop: every completed action sets ``cooldown_until_tick`` and KEEPs
also decay ``score_mult`` (diminishing returns).

Anti-starvation: the UCB bonus prefers under-sampled actions; the aging bonus
grows linearly with ticks since the action was last run, so a long-ignored
action surfaces into the top eventually regardless of low base.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

from .action_registry import ActionMetadata, ActionRegistry


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Diminishing-returns decay applied to ``score_mult`` after each KEEP. The
# multiplier is ``max(KEEP_DECAY_FLOOR, 1 - KEEP_DECAY_RATE * gain_pct)`` so a
# strong KEEP (e.g. +5%) halves the mult while a tiny KEEP barely touches it.
KEEP_DECAY_RATE: float = 0.1
KEEP_DECAY_FLOOR: float = 0.5

# Discard multiplier — kept as a deprecated constant for backward compat with
# old state.json snapshots and the thin ``apply_discard`` wrapper below.
# New code paths should use :func:`apply_failure` / :func:`apply_no_promote`
# which only nudge ``score_mult`` after a streak of 3 consecutive bad
# outcomes (see ``STREAK_*`` below).
DISCARD_MULT: float = 0.7

# Exploration constants. UCB_C controls how aggressively under-sampled actions
# bubble up; AGING_RATE bumps idle actions linearly per tick.
UCB_C: float = 0.6
AGING_RATE: float = 0.05


# Streak-based dampening. Replaces the mandatory per-action cooldown so the
# Orchestration LLM can re-select an action immediately if it has good reason
# to retry, while still discouraging an action that keeps failing or keeps
# succeeding-without-promoting. Threshold 3 with a 0.85 multiplier and a 0.2
# floor means a degenerate streak shaves the row to ~30% of its base over six
# strikes — enough to bubble alternatives up without permanently burying the
# action.
STREAK_THRESHOLD: int = 3
STREAK_PENALTY_MULT: float = 0.85
STREAK_PENALTY_FLOOR: float = 0.2


# Legacy cooldown table — preserved so old state.json snapshots that still
# carry ``cooldown_until_tick`` values can round-trip and so we can revert
# the streak-based design without resurrecting the table from git. The
# active scheduler ignores this map: ``_cooldown_for`` returns 0 for every
# action and ``apply_keep`` / ``apply_failure`` no longer set
# ``cooldown_until_tick``.
COOLDOWN_TICKS: dict[str, int] = {
    "backends": 2,
    "params": 2,
    "sweep": 6,
    "kernel_opt": 3,
    "integrate": 0,
    "operator_tuning": 4,
    "compiler_tuning": 4,
    "framework_rebuild": 6,
    "comm_optimization": 4,
    "deep_kernel_analysis": 4,
    "vendor_kernel_config": 3,
    "dream": 10,
    "re_explore": 8,
    "recover": 0,
    "profile": 4,
    "pmc_roofline": 4,
    "validate_stack": 0,
    "baseline": 0,
    "report": 0,
    "target_analysis": 0,
}


# Marathon priors per model_class. Mirrors the table at
# ``/wekafs/zgong/TBO/inference_optimization/marathon/skills/SKILL.md`` L832-839.
# Only the six actions explicitly tabulated in marathon are listed; every
# other action falls back to ``compute_initial_priors_from_metadata``.
#
# Marathon names use kebab-case (deep-kernel-analysis); inference_optimizer
# uses snake_case (deep_kernel_analysis). The mapping below uses the
# snake_case names so seed_action_scores can look them up directly.
MARATHON_PRIORS: dict[str, dict[str, float]] = {
    "dense": {
        "deep_kernel_analysis": 9.0,
        "operator_tuning": 4.0,
        "kernel_opt": 8.0,
        "framework_rebuild": 3.0,
        "comm_optimization": 2.0,
        "compiler_tuning": 6.0,
    },
    "moe_mla": {
        "deep_kernel_analysis": 8.0,
        "operator_tuning": 7.0,
        "kernel_opt": 6.0,
        "framework_rebuild": 4.0,
        "comm_optimization": 5.0,
        "compiler_tuning": 3.0,
    },
    "moe_swa": {
        "deep_kernel_analysis": 8.0,
        "operator_tuning": 7.0,
        "kernel_opt": 6.0,
        "framework_rebuild": 4.0,
        "comm_optimization": 5.0,
        "compiler_tuning": 3.0,
    },
    "moe_mla_nsa": {
        "deep_kernel_analysis": 8.0,
        "operator_tuning": 7.0,
        "kernel_opt": 6.0,
        "framework_rebuild": 4.0,
        "comm_optimization": 6.0,
        "compiler_tuning": 3.0,
    },
}


# Locked sentinels used by ``effective_score`` to flag rows that should not
# be picked. They sort to the bottom and the renderer surfaces the reason.
_LOCKED_SCORE: float = -1.0


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------
@dataclass
class ActionScore:
    """Per-action heuristic state persisted in :class:`SharedState`.

    Stored as a plain dict inside ``SharedState.action_scores`` so JSON
    serialisation is automatic; ``from_dict`` / ``to_dict`` adapt back and
    forth on the boundary. Use :meth:`SharedState.get_action_score` to
    materialise an ``ActionScore`` from the stored dict.
    """

    base_score: float = 0.0
    score_mult: float = 1.0
    runs: int = 0
    keeps: int = 0
    discards: int = 0
    last_run_tick: int = -1
    last_gain_pct: float = 0.0
    ema_gain_pct: float = 0.0
    # Retained for state.json backward-compat; the scheduler no longer
    # writes to it on KEEP / DISCARD outcomes. Only ``apply_lock`` and
    # similar explicit lockouts should touch this field going forward.
    cooldown_until_tick: int = 0
    locked_reason: str = ""
    # Streak counters consumed by :func:`apply_failure` /
    # :func:`apply_no_promote`. Both reset to 0 on every KEEP/promote.
    consecutive_failures: int = 0
    consecutive_no_promote: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_score": float(self.base_score),
            "score_mult": float(self.score_mult),
            "runs": int(self.runs),
            "keeps": int(self.keeps),
            "discards": int(self.discards),
            "last_run_tick": int(self.last_run_tick),
            "last_gain_pct": float(self.last_gain_pct),
            "ema_gain_pct": float(self.ema_gain_pct),
            "cooldown_until_tick": int(self.cooldown_until_tick),
            "locked_reason": str(self.locked_reason or ""),
            "consecutive_failures": int(self.consecutive_failures),
            "consecutive_no_promote": int(self.consecutive_no_promote),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "ActionScore":
        if not isinstance(raw, dict):
            return cls()
        return cls(
            base_score=float(raw.get("base_score", 0.0) or 0.0),
            score_mult=float(raw.get("score_mult", 1.0) or 1.0),
            runs=int(raw.get("runs", 0) or 0),
            keeps=int(raw.get("keeps", 0) or 0),
            discards=int(raw.get("discards", 0) or 0),
            last_run_tick=int(raw.get("last_run_tick", -1) if raw.get("last_run_tick") is not None else -1),
            last_gain_pct=float(raw.get("last_gain_pct", 0.0) or 0.0),
            ema_gain_pct=float(raw.get("ema_gain_pct", 0.0) or 0.0),
            cooldown_until_tick=int(raw.get("cooldown_until_tick", 0) or 0),
            locked_reason=str(raw.get("locked_reason", "") or ""),
            consecutive_failures=int(raw.get("consecutive_failures", 0) or 0),
            consecutive_no_promote=int(raw.get("consecutive_no_promote", 0) or 0),
        )


# ---------------------------------------------------------------------------
# Initial seeding
# ---------------------------------------------------------------------------
def compute_initial_priors_from_metadata(meta: ActionMetadata) -> float:
    """Auto-compute a base score from :class:`ActionMetadata`.

    Formula mirrors marathon's per-action heuristic:

        ((gain_lo + gain_hi) / 2) / max(cost_p50, 1.0)
            * (1 - accuracy_risk) * (1 - crash_risk)

    Returns a non-negative float; 0.0 for actions with zero expected gain
    (e.g. ``profile``, ``report``) which are then scored solely via the
    aging + UCB bonuses (and the Coordinator's required-step gates).
    """
    lo, hi = meta.expected_gain_pct
    midpoint = (float(lo) + float(hi)) / 2.0
    cost = max(float(meta.cost_minutes_p50 or 0.0), 1.0)
    risk_factor = max(0.0, 1.0 - float(meta.accuracy_risk)) * max(
        0.0, 1.0 - float(meta.crash_risk)
    )
    return max(0.0, (midpoint / cost) * risk_factor)


def _normalize_model_class(model_class: str) -> str:
    raw = (model_class or "").strip().lower()
    if not raw:
        return ""
    # Tolerate alternate forms ("MoE+MLA" / "moe-mla" / "moe_mla").
    return (
        raw.replace("+", "_")
        .replace("-", "_")
        .replace(" ", "_")
    )


def seed_action_scores(
    registry: ActionRegistry,
    *,
    model_class: str,
    enabled: Iterable[str],
) -> dict[str, dict[str, Any]]:
    """Return a fresh ``{name: ActionScore.to_dict()}`` map for the enabled set.

    For each enabled action:

    * if the model_class has a marathon prior for the action, that wins;
    * otherwise we fall back to the auto-computed prior from registry metadata.

    The returned dict is plain JSON-roundtrippable; the caller is expected
    to drop it into ``SharedState.action_scores``.
    """
    mc = _normalize_model_class(model_class)
    marathon = MARATHON_PRIORS.get(mc, {}) if mc else {}
    out: dict[str, dict[str, Any]] = {}
    for name in enabled:
        meta = registry.get(name)
        if meta is None:
            continue
        auto = compute_initial_priors_from_metadata(meta)
        marathon_prior = marathon.get(name)
        base = (
            float(marathon_prior)
            if marathon_prior is not None
            else auto
        )
        out[name] = ActionScore(base_score=base).to_dict()
    return out


# ---------------------------------------------------------------------------
# Effective score
# ---------------------------------------------------------------------------
def target_gap_multiplier(
    *, target_gap_pct: float, cumulative_gain: float
) -> float:
    """Map the remaining target gap to a [0.1, 1.6] multiplier.

    ``target_gap_pct == 0`` (no target / target hit) returns 1.0 — the
    Orchestration prompt should already be in `report`/stop state when the
    objective is reached.
    """
    if target_gap_pct is None or target_gap_pct <= 0:
        return 1.0
    remaining = max(0.0, float(target_gap_pct) - float(cumulative_gain or 0.0))
    if remaining <= 0.0:
        return 0.1
    if remaining < 5.0:
        return 0.8
    if remaining < 15.0:
        return 1.0
    if remaining < 30.0:
        return 1.3
    return 1.6


def _ucb_bonus(*, total_runs: int, runs: int) -> float:
    """UCB-1 exploration bonus.

    ``total_runs`` is the sum of all action runs in the registry; we add 2 to
    the log argument so it stays defined when nothing has been run yet.
    """
    return UCB_C * math.sqrt(math.log(max(2, int(total_runs))) / (1 + max(0, int(runs))))


def _aging_bonus(*, tick: int, last_run_tick: int) -> float:
    """Linear aging bonus — grows monotonically the longer the action sits idle."""
    if last_run_tick < 0:
        age = tick + 1
    else:
        age = max(0, tick - last_run_tick)
    return AGING_RATE * float(age)


def effective_score(
    a: ActionScore,
    *,
    meta: ActionMetadata | None,
    tick: int,
    total_runs: int,
    target_gap_mult: float = 1.0,
) -> float:
    """Compute the effective sort key for one action at the current tick.

    Returns :data:`_LOCKED_SCORE` (= -1.0) only when the action carries an
    explicit ``locked_reason`` (e.g. ``params/grid_exhausted``). Mandatory
    cooldowns were retired in favour of the streak-based penalty in
    :func:`apply_failure` / :func:`apply_no_promote`, so a non-zero
    ``cooldown_until_tick`` on a row no longer suppresses it — keep the
    field intact for backward compat with archived state.json snapshots.
    """
    if a.locked_reason:
        return _LOCKED_SCORE
    acc_risk = float(meta.accuracy_risk) if meta is not None else 0.0
    crash_risk = float(meta.crash_risk) if meta is not None else 0.0
    base = (
        float(a.base_score)
        * float(a.score_mult)
        * max(0.0, 1.0 - acc_risk)
        * max(0.0, 1.0 - crash_risk)
        * float(target_gap_mult)
    )
    return base + _ucb_bonus(total_runs=total_runs, runs=a.runs) + _aging_bonus(
        tick=tick, last_run_tick=a.last_run_tick,
    )


# ---------------------------------------------------------------------------
# Mutators
# ---------------------------------------------------------------------------
def _cooldown_for(action_name: str) -> int:
    """Return 0 — mandatory cooldowns are disabled.

    Kept as a function (rather than inlined) so the COOLDOWN_TICKS table
    above can still be inspected by diagnostics and so re-enabling
    cooldowns later is a one-line revert.
    """
    return 0


def apply_keep(
    a: ActionScore,
    *,
    gain_pct: float,
    tick: int,
    action_name: str,
) -> ActionScore:
    """Record a KEEP outcome. Returns the same instance (mutated)."""
    gain = max(0.0, float(gain_pct or 0.0))
    decay = max(KEEP_DECAY_FLOOR, 1.0 - KEEP_DECAY_RATE * gain)
    a.score_mult = max(KEEP_DECAY_FLOOR, float(a.score_mult) * decay)
    a.runs = int(a.runs) + 1
    a.keeps = int(a.keeps) + 1
    a.last_run_tick = int(tick)
    a.last_gain_pct = gain
    # Exponential moving average — 0.4 weight on the newest observation keeps
    # the EMA responsive without flapping when one bad measurement lands.
    a.ema_gain_pct = 0.6 * float(a.ema_gain_pct) + 0.4 * gain
    # A genuine win resets both penalty streaks — even if the previous
    # two runs were failures or no-promotes, a fresh promote means the
    # action is paying off.
    a.consecutive_failures = 0
    a.consecutive_no_promote = 0
    return a


def apply_failure(
    a: ActionScore,
    *,
    tick: int,
    action_name: str,
) -> ActionScore:
    """Record a FAILED task outcome (``result.status == 'failed' / 'error'``).

    Bumps ``consecutive_failures``. Only after three back-to-back failures
    is ``score_mult`` shaved by :data:`STREAK_PENALTY_MULT` (floored at
    :data:`STREAK_PENALTY_FLOOR`); the streak counter is reset on
    application so the next penalty also requires another 3 strikes.
    """
    a.runs = int(a.runs) + 1
    a.discards = int(a.discards) + 1
    a.last_run_tick = int(tick)
    a.last_gain_pct = 0.0
    a.ema_gain_pct = 0.6 * float(a.ema_gain_pct)
    a.consecutive_no_promote = 0
    a.consecutive_failures = int(a.consecutive_failures) + 1
    if a.consecutive_failures >= STREAK_THRESHOLD:
        a.score_mult = max(
            STREAK_PENALTY_FLOOR,
            float(a.score_mult) * STREAK_PENALTY_MULT,
        )
        a.consecutive_failures = 0
    return a


def apply_no_promote(
    a: ActionScore,
    *,
    tick: int,
    action_name: str,
) -> ActionScore:
    """Record a successful task that did NOT promote ``current_best``.

    This is the "we ran backends/params/sweep and nothing was better than
    the existing baseline" case. Same shape as :func:`apply_failure` but
    tracked on a separate streak so the diagnostics surface the two
    pathologies independently.
    """
    a.runs = int(a.runs) + 1
    a.discards = int(a.discards) + 1
    a.last_run_tick = int(tick)
    a.last_gain_pct = 0.0
    a.ema_gain_pct = 0.6 * float(a.ema_gain_pct)
    a.consecutive_failures = 0
    a.consecutive_no_promote = int(a.consecutive_no_promote) + 1
    if a.consecutive_no_promote >= STREAK_THRESHOLD:
        a.score_mult = max(
            STREAK_PENALTY_FLOOR,
            float(a.score_mult) * STREAK_PENALTY_MULT,
        )
        a.consecutive_no_promote = 0
    return a


def apply_discard(
    a: ActionScore,
    *,
    tick: int,
    action_name: str,
) -> ActionScore:
    """Backward-compatible alias for :func:`apply_failure`.

    Prefer :func:`apply_failure` (task returned failed/error) or
    :func:`apply_no_promote` (task succeeded but didn't promote
    current_best) in new code — they bucket the two pathologies into
    distinct streaks for clearer diagnostics.
    """
    return apply_failure(a, tick=tick, action_name=action_name)


def apply_lock(a: ActionScore, reason: str) -> ActionScore:
    """Mark the action as locked. No-op if already locked (preserve first reason)."""
    if not a.locked_reason:
        a.locked_reason = str(reason or "locked")
    return a


def apply_unlock(a: ActionScore) -> ActionScore:
    a.locked_reason = ""
    return a


def boost_action(
    scores: dict[str, dict[str, Any]],
    name: str,
    *,
    to: float | None = None,
    mult: float | None = None,
) -> None:
    """Marathon rules 1–5: bump one action's mult upward.

    ``to`` pins ``score_mult`` to at least the given value; ``mult``
    multiplies the existing mult by the factor. Both may be combined.
    """
    raw = scores.get(name)
    if not isinstance(raw, dict):
        return
    a = ActionScore.from_dict(raw)
    if to is not None:
        a.score_mult = max(float(a.score_mult), float(to))
    if mult is not None:
        a.score_mult = float(a.score_mult) * float(mult)
    scores[name] = a.to_dict()


def dampen_action(
    scores: dict[str, dict[str, Any]],
    name: str,
    mult: float,
) -> None:
    """Marathon rules 6–7: dampen one action's mult (mult should be in (0, 1])."""
    raw = scores.get(name)
    if not isinstance(raw, dict):
        return
    a = ActionScore.from_dict(raw)
    a.score_mult = max(0.0, float(a.score_mult) * float(mult))
    scores[name] = a.to_dict()


# ---------------------------------------------------------------------------
# Ranking / rendering
# ---------------------------------------------------------------------------
def rank_top_k(
    scores: dict[str, dict[str, Any]],
    registry: ActionRegistry,
    *,
    tick: int,
    target_gap_mult: float = 1.0,
    k: int = 12,
) -> list[tuple[str, float, ActionScore]]:
    """Return ``[(name, eff_score, ActionScore), ...]`` sorted by eff_score desc.

    Locked / cooldown rows are returned with ``eff_score == -1.0`` AFTER all
    positive rows, so the renderer can include them in the visible top-K to
    surface "you have backends and params on cooldown" context. Among the
    locked rows we keep deterministic ordering by name.
    """
    if not isinstance(scores, dict) or not scores:
        return []
    total_runs = sum(
        int(ActionScore.from_dict(v).runs)
        for v in scores.values()
        if isinstance(v, dict)
    )
    rows: list[tuple[str, float, ActionScore]] = []
    for name, raw in scores.items():
        if not isinstance(raw, dict):
            continue
        a = ActionScore.from_dict(raw)
        meta = registry.get(name)
        eff = effective_score(
            a,
            meta=meta,
            tick=tick,
            total_runs=total_runs,
            target_gap_mult=target_gap_mult,
        )
        rows.append((name, eff, a))
    # Primary: descending effective score (locked rows tie at -1.0).
    # Secondary tiebreaker: name asc so render is stable.
    rows.sort(key=lambda r: (-r[1], r[0]))
    if k > 0:
        rows = rows[: int(k)]
    return rows


__all__ = [
    "ActionScore",
    "AGING_RATE",
    "COOLDOWN_TICKS",
    "DISCARD_MULT",
    "KEEP_DECAY_FLOOR",
    "KEEP_DECAY_RATE",
    "MARATHON_PRIORS",
    "STREAK_PENALTY_FLOOR",
    "STREAK_PENALTY_MULT",
    "STREAK_THRESHOLD",
    "UCB_C",
    "apply_discard",
    "apply_failure",
    "apply_keep",
    "apply_lock",
    "apply_no_promote",
    "apply_unlock",
    "boost_action",
    "compute_initial_priors_from_metadata",
    "dampen_action",
    "effective_score",
    "rank_top_k",
    "seed_action_scores",
    "target_gap_multiplier",
]
