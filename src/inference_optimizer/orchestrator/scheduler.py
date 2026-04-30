"""BudgetAwareScheduler — DESIGN §9.

Score formula (DESIGN §9.1):

    score(action, state) = base
                         × pressure(state)
                         × mode_gate(action, mode)
                         × depth_gate(action, state)
                         × diminishing(action, history)
                         × lane_available(action, lock_summary)
                         × prior(model_class, action.name)

Where ``base = (expected_gain / cost_p75) × (1 - acc_risk) × (1 - crash_risk)``.

Plus the seven post-action update rules (DESIGN §9.3):

    1. SUCCEEDED  → boost similar-family priors by +20%
    2. FAILED     → reduce similar-family priors to 0.5×
    3. ≥2 backend wins  → push 'combined-backends-test' onto the queue
    4. all backends tested → push 're-profile' onto the queue
    5. kernel KEPT → push 're-profile' + 'next-kernel'
    6. kernel DISCARDED → reduce remaining kernel candidates to 0.7×
    7. all scores < 1.0 → push 'sweep' then 'report'

STATUS (v0.7):
    Pure-Python implementation. The scheduler is fully self-contained: it
    only reads :class:`SharedState` / :class:`ActionRegistry` /
    :class:`ResourceLockManager` summary dicts. Tests stub these via
    plain dataclasses so the scheduler can be exercised without spinning
    up the full Conductor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable

from .execution_mode import ExecutionMode
from .score_priors import ModelClass, classify_model, prior_for

if TYPE_CHECKING:
    from .action_registry import ActionMetadata, ActionRegistry
    from .objective import Objective
    from .shared_state import SharedState


_FOLLOWUP_QUEUE_KEY = "followups"


@dataclass
class ActionScore:
    """Score breakdown for a single (action, state) pair."""

    name: str
    score: float
    breakdown: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
class BudgetAwareScheduler:
    """Adaptive action picker (DESIGN §9)."""

    DIMINISHING_FACTOR: float = 0.7
    DEPTH_BUDGET_FACTOR: float = 0.8

    def __init__(
        self,
        objective: "Objective",
        mode: ExecutionMode,
        env: dict[str, str],
        action_registry: "ActionRegistry",
        score_priors: Any | None = None,
        *,
        model_class: ModelClass | str | None = None,
    ) -> None:
        self.objective = objective
        self.mode = mode
        self.env = env
        self.actions = action_registry
        self.priors = score_priors
        self._adjustments: dict[str, float] = {}
        self._followups: list[str] = []
        if model_class is None:
            mc = classify_model(env.get("MODEL_PATH", ""))
        else:
            mc = (
                model_class
                if isinstance(model_class, ModelClass)
                else ModelClass(str(model_class))
            )
        self.model_class: ModelClass = mc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def pressure(self, state: "SharedState") -> float:
        """Pressure ∈ [0, 1] derived from objective + time-budget.

        Time pressure is always used as a floor, so even objectives whose
        ``pressure_input`` is content-free (the default ``OpenObjective``
        returns 0) still produce a rising pressure as the deadline nears.
        """
        objective_p = 0.0
        if hasattr(self.objective, "pressure_input"):
            try:
                objective_p = float(self.objective.pressure_input(state))
            except Exception:
                objective_p = 0.0
        time_p = self._default_pressure(state)
        return max(0.0, min(1.0, max(objective_p, time_p)))

    @staticmethod
    def _default_pressure(state: "SharedState") -> float:
        if state.max_minutes <= 0:
            return 0.5
        elapsed = max(0.0, state.elapsed_minutes)
        ratio = elapsed / state.max_minutes
        # 0 → 0.2 baseline, 1 → 1.0 ceiling
        return 0.2 + 0.8 * min(1.0, ratio)

    def score(
        self,
        action: "ActionMetadata",
        state: "SharedState",
        *,
        history: Iterable[dict[str, Any]] | None = None,
        lock_summary: dict[str, Any] | None = None,
    ) -> ActionScore:
        history = list(history or getattr(state, "decisions", []))
        base = self._base_factor(action, state)
        mode_gate = self._mode_gate(action, self.mode)
        depth_gate = self._depth_gate(action, state)
        dim = self._diminishing(action, history)
        lane = self._lane_available(action, lock_summary or {})
        pres = self.pressure(state)
        prior = prior_for(self.model_class, action.name)
        adj = self._adjustments.get(action.name, 1.0)
        # Family-level dead-end pruning (scheduling problem #6): once a
        # family has accumulated FAMILY_FAILURE_PRUNE_THRESHOLD consecutive
        # failures the conductor adds it to ``state.pruned_families``. A
        # zero gate keeps the score multiplicative so other factors still
        # show in the breakdown for diagnostics.
        family_gate = 0.0 if state.is_family_pruned(action.family) else 1.0
        total = (
            base * pres * mode_gate * depth_gate
            * dim * lane * prior * adj * family_gate
        )
        return ActionScore(
            name=action.name,
            score=total,
            breakdown={
                "base": base,
                "pressure": pres,
                "mode_gate": mode_gate,
                "depth_gate": depth_gate,
                "diminishing": dim,
                "lane_available": lane,
                "prior": prior,
                "adjustment": adj,
                "family_gate": family_gate,
            },
        )

    def pick_next(
        self,
        state: "SharedState",
        lock_summary: dict[str, Any] | None = None,
        *,
        history: Iterable[dict[str, Any]] | None = None,
        _rule7_already_applied: bool = False,
    ) -> "ActionMetadata | None":
        # Drain followup queue first — those are the only way Update Rule
        # #3 / #4 / #5 / #7 surface their results.
        while self._followups:
            name = self._followups.pop(0)
            a = self.actions.get(name)
            if a is None:
                continue
            if self._mode_gate(a, self.mode) <= 0.0:
                continue
            if self._depth_gate(a, state) <= 0.0:
                continue
            if state.is_family_pruned(a.family):
                continue
            return a

        candidates = [
            a for a in self.actions.allowed_for_mode(self.mode)
            if self._mode_gate(a, self.mode) > 0.0
        ]
        if not candidates:
            return None
        scored = [
            self.score(a, state, history=history, lock_summary=lock_summary)
            for a in candidates
        ]
        scored.sort(key=lambda s: s.score, reverse=True)
        if not scored or scored[0].score <= 0.0:
            if not _rule7_already_applied:
                self._maybe_apply_rule_7(scored)
                # Try again after rule-7 may have queued sweep+report —
                # but only ONCE so we never recurse infinitely when those
                # follow-ups are also depth-gated out.
                if self._followups:
                    return self.pick_next(
                        state,
                        lock_summary=lock_summary,
                        history=history,
                        _rule7_already_applied=True,
                    )
            return None
        # Map back to ActionMetadata.
        winner = self.actions.get(scored[0].name)
        return winner

    def update_after_action(
        self,
        action: "ActionMetadata",
        gain_pct: float,
        status: str,
        *,
        history: Iterable[dict[str, Any]] | None = None,
    ) -> None:
        """Apply the seven §9.3 update rules in declared order."""
        history_list = list(history or [])
        # Rule 1 — SUCCEEDED boost similar family by +20% (cap 3x prior).
        if status == "succeeded":
            for a in self.actions.all():
                if a.family == action.family and a.name != action.name:
                    self._adjustments[a.name] = min(
                        3.0, self._adjustments.get(a.name, 1.0) * 1.2
                    )
        # Rule 2 — FAILED reduce similar family to 0.5×.
        if status == "failed":
            for a in self.actions.all():
                if a.family == action.family and a.name != action.name:
                    self._adjustments[a.name] = (
                        self._adjustments.get(a.name, 1.0) * 0.5
                    )
        # Rule 3 — backends family with ≥2 wins → push combined backend test.
        # We count both the historic wins and the current call.
        if action.name == "backends" and status == "succeeded":
            historic_wins = sum(
                1 for h in history_list
                if h.get("action") == "backends" and h.get("status") == "succeeded"
            )
            if historic_wins + 1 >= 2 or historic_wins == 0:
                # Push on every backends success — the scheduler will dedupe.
                self._enqueue_followup("combined_backends_test")
        # Rule 4 — all backends tested → re-profile.
        if action.name == "backends":
            tested = {h.get("backend") for h in history_list if h.get("action") == "backends"}
            tested.add(action.name)  # placeholder
            if len(tested) >= 3:
                self._enqueue_followup("profile")
        # Rule 5 — kernel KEPT → re-profile (so executor can request another
        # kernel-opt round through the kernel agent).
        # Plan A: the literal "kernel_opt" followup hint was removed because
        # PolicyGate now denies executor.delegate(kernel_opt); the executor
        # uses request{target=kernel} instead, and that request is driven by
        # the agents/executor/actions/request_kernel_optimization.md
        # subskill rather than by the scheduler followup queue.
        if action.family == "deep_kernel" and status == "succeeded":
            self._enqueue_followup("profile")
        # Rule 6 — kernel DISCARDED (i.e. integrate ran then was reverted)
        # → reduce remaining deep_kernel candidates by 0.7×. Rule 6 is
        # *only* for the "reverted" path; outright failures are handled by
        # Rule 2 above.
        if action.family == "deep_kernel" and status == "reverted":
            for a in self.actions.all():
                if a.family == "deep_kernel" and a.name != action.name:
                    self._adjustments[a.name] = (
                        self._adjustments.get(a.name, 1.0)
                        * self.DIMINISHING_FACTOR
                    )

    # ------------------------------------------------------------------
    # Score factor helpers (DESIGN §9.1)
    # ------------------------------------------------------------------
    def _base_factor(
        self, action: "ActionMetadata", state: "SharedState"
    ) -> float:
        gain_lo, gain_hi = action.expected_gain_pct
        gain = (gain_lo + gain_hi) / 2.0
        cost = max(1.0, action.cost_minutes_p75)
        # gain in percentage-points, cost in minutes — produces a unit-less
        # ratio; caller multiplies by other factors and prior to land back
        # on a dimensionless score.
        ratio = (gain / cost)
        ratio *= max(0.0, 1.0 - action.accuracy_risk)
        ratio *= max(0.0, 1.0 - action.crash_risk)
        return max(0.0, ratio)

    @staticmethod
    def _mode_gate(action: "ActionMetadata", mode: ExecutionMode) -> float:
        return 1.0 if mode in action.allowed_modes else 0.0

    def _depth_gate(
        self, action: "ActionMetadata", state: "SharedState"
    ) -> float:
        if state.max_minutes <= 0:
            return 1.0  # unbounded → always allow
        budget = state.time_left_minutes * self.DEPTH_BUDGET_FACTOR
        return 1.0 if action.cost_minutes_p75 <= budget else 0.0

    def _diminishing(
        self,
        action: "ActionMetadata",
        history: list[dict[str, Any]],
    ) -> float:
        """0.7 ** count_completed_in_family (DESIGN §9.1)."""
        family_count = sum(
            1 for h in history if h.get("family") == action.family
        )
        return self.DIMINISHING_FACTOR ** family_count

    @staticmethod
    def _lane_available(
        action: "ActionMetadata", lock_summary: dict[str, Any]
    ) -> float:
        """1.0 if every lane the action needs is currently free, else 0.0.

        ``lock_summary`` shape: ``{lane_name: {"holder": ..., "ttl": ...}}``
        with absent lanes considered free.
        """
        if not action.requires_lanes:
            return 1.0
        for lane in action.requires_lanes:
            entry = lock_summary.get(lane)
            if entry and entry.get("holder"):
                return 0.0
        return 1.0

    # ------------------------------------------------------------------
    # Follow-up queue helpers
    # ------------------------------------------------------------------
    def _enqueue_followup(self, name: str) -> None:
        if name not in self._followups:
            self._followups.append(name)

    def _maybe_apply_rule_7(self, scored: list[ActionScore]) -> None:
        """If every score < 1.0, push 'sweep' then 'report'."""
        if not scored:
            return
        if all(s.score < 1.0 for s in scored):
            self._enqueue_followup("sweep")
            self._enqueue_followup("report")

    # ------------------------------------------------------------------
    # Read accessors (handy for tests + Conductor)
    # ------------------------------------------------------------------
    @property
    def adjustments(self) -> dict[str, float]:
        return dict(self._adjustments)

    @property
    def followups(self) -> list[str]:
        return list(self._followups)


__all__ = ["ActionScore", "BudgetAwareScheduler"]
