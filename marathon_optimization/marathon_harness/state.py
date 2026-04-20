"""MarathonState — full 40+ field state with scoring, tiers, save/load."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import random
import tempfile
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

log = logging.getLogger(__name__)

try:
    import numpy as np
except ImportError:
    np = None


# ---------------------------------------------------------------------------
# Scoring priors  (expected_gain_pct, cost_minutes, accuracy_risk, crash_risk)
# Rows: action type.  Columns: model class.
# ---------------------------------------------------------------------------
SCORE_PRIORS: dict[str, dict[str, tuple[float, float, float, float]]] = {
    "deep-kernel-analysis": {
        "dense":        (0.0, 30, 0.0, 0.0),
        "moe_mla":      (0.0, 30, 0.0, 0.0),
        "moe_swa":      (0.0, 30, 0.0, 0.0),
        "moe_mla_nsa":  (0.0, 30, 0.0, 0.0),
    },
    "operator-tuning": {
        "dense":        (3.0, 30, 0.02, 0.02),
        "moe_mla":      (4.0, 30, 0.02, 0.02),
        "moe_swa":      (3.5, 30, 0.02, 0.02),
        "moe_mla_nsa":  (4.0, 30, 0.02, 0.02),
    },
    "deep-kernel-opt": {
        "dense":        (5.0, 45, 0.05, 0.05),
        "moe_mla":      (6.0, 45, 0.05, 0.05),
        "moe_swa":      (5.5, 45, 0.05, 0.05),
        "moe_mla_nsa":  (6.0, 45, 0.05, 0.05),
    },
    "framework-rebuild": {
        "dense":        (2.0, 15, 0.15, 0.10),
        "moe_mla":      (2.5, 15, 0.15, 0.10),
        "moe_swa":      (2.0, 15, 0.15, 0.10),
        "moe_mla_nsa":  (2.5, 15, 0.15, 0.10),
    },
    "comm-optimization": {
        "dense":        (2.0, 30, 0.05, 0.05),
        "moe_mla":      (3.0, 30, 0.05, 0.05),
        "moe_swa":      (2.5, 30, 0.05, 0.05),
        "moe_mla_nsa":  (3.0, 30, 0.05, 0.05),
    },
    "compiler-tuning": {
        "dense":        (1.5, 20, 0.05, 0.03),
        "moe_mla":      (2.0, 20, 0.05, 0.03),
        "moe_swa":      (1.5, 20, 0.05, 0.03),
        "moe_mla_nsa":  (2.0, 20, 0.05, 0.03),
    },
}

HANDOFF_BOOSTS: dict[str, int] = {
    "marathon-candidate": 3,
    "register-pressure-fixable": 3,
    "shape-tuning-untested": 2,
    "oob-untested": 2,
}

BASE_SCORES: dict[str, int] = {
    "deep-kernel-analysis": 9,
    "operator-tuning": 7,
    "deep-kernel-opt": 6,
    "comm-optimization": 5,
}

TIER_BOUNDARIES: list[float] = [3.0, 8.0, 24.0]  # hours


def _tier_for_hours(h: float) -> str:
    if h < TIER_BOUNDARIES[0]:
        return "tier1"
    if h < TIER_BOUNDARIES[1]:
        return "tier2"
    if h < TIER_BOUNDARIES[2]:
        return "tier3"
    return "tier4"


@dataclass
class ActionPosterior:
    """Beta distribution posterior for (action_type, code_region) success rate."""
    alpha: float = 1.0
    beta: float = 1.0
    total_gain: float = 0.0
    count: int = 0

    def sample(self) -> float:
        """Thompson Sampling: draw from posterior for exploration-aware scoring."""
        p = random.betavariate(max(self.alpha, 0.01), max(self.beta, 0.01))
        mean_gain = (self.total_gain / self.count) if self.count > 0 else 3.0
        return p * max(mean_gain, 0.0)

    def update(self, success: bool, gain_pct: float = 0.0) -> None:
        if success:
            self.alpha += 1.0
        else:
            self.beta += 1.0
        self.total_gain += max(gain_pct, 0.0)
        self.count += 1


def compute_score(
    expected_gain: float,
    cost_minutes: float,
    accuracy_risk: float = 0.0,
    crash_risk: float = 0.0,
    target_gap_pct: float = 100.0,
) -> float:
    """Canonical scoring formula from SKILL.md."""
    if cost_minutes <= 0:
        cost_minutes = 1
    gap_mult = max(0.1, min(target_gap_pct / 100, 2.0))
    return (expected_gain / cost_minutes) * (1 - accuracy_risk) * (1 - crash_risk) * gap_mult


# ---------------------------------------------------------------------------
# State dataclass
# ---------------------------------------------------------------------------

@dataclass
class MarathonState:
    # --- identity ---
    session_id: str = ""
    model_name: str = ""
    model_class: str = "dense"
    framework: str = "sglang"
    tp: int = 8
    gpu_type: str = "MI355X"
    gpu_count: int = 8
    num_nodes: int = 1

    # --- paths ---
    base_dir: str = ""
    session_dir: str = ""

    # --- performance ---
    sprint_tput_per_gpu: float = 0.0
    baseline_tput_per_gpu: float = 0.0
    current_tput_per_gpu: float = 0.0
    best_tput_per_gpu: float = 0.0
    cumulative_gain_pct: float = 0.0
    target_tput_per_gpu: float = 0.0
    target_gap_pct: float = 100.0

    # --- deep analysis ---
    kernel_dispatch_map: dict[str, Any] = field(default_factory=dict)
    untuned_shapes: list[str] = field(default_factory=list)
    dispatch_bugs_found: int = 0

    # --- accuracy ---
    baseline_accuracy: float = 0.0
    accuracy_threshold: float = 0.01

    # --- DFS ---
    action_stack: list[dict[str, Any]] = field(default_factory=list)
    completed_actions: list[dict[str, Any]] = field(default_factory=list)
    kernel_candidates: list[dict[str, Any]] = field(default_factory=list)

    # --- legacy async ---
    pending_kernel_tasks: list[dict[str, Any]] = field(default_factory=list)
    kernel_results: list[dict[str, Any]] = field(default_factory=list)

    # --- kernel-manager IPC ---
    kernel_manager_last_seen_id: str = ""
    kernel_manager_targets_pushed: int = 0
    kernel_manager_merges_completed: int = 0
    kernel_manager_merges_kept: int = 0

    # --- watchdog IPC ---
    watchdog_last_seen_finding_id: str = ""
    watchdog_last_seen_event_id: str = ""
    watchdog_findings_consumed: int = 0
    watchdog_hw_blocked_kernels: list[str] = field(default_factory=list)
    events_written: int = 0

    # --- kernel manager IPC ---
    kernel_manager_processed_ids: list[str] = field(default_factory=list)

    # --- infra ---
    current_time_tier: str = "tier1"
    checkpoint_path: str = ""
    dream_count: int = 0
    last_dream_ts: float = 0.0
    crash_count: int = 0
    crash_log: list[str] = field(default_factory=list)
    strategies_tested: list[str] = field(default_factory=list)
    tier_breakdown: dict[str, Any] = field(default_factory=dict)
    loop_signatures: list[str] = field(default_factory=list)

    # --- tracking ---
    total_wall_minutes: float = 0.0
    total_kernel_opt_submissions: int = 0
    consecutive_discards: int = 0
    backend_wins: dict[str, int] = field(default_factory=dict)
    frameworks_rebuilt: list[str] = field(default_factory=list)

    # --- cost ---
    total_llm_calls: int = 0
    total_llm_cost_usd: float = 0.0
    total_llm_turns: int = 0

    # --- phase ---
    warm_started: bool = False
    profiled: bool = False
    phase: str = "warm_start"  # warm_start | profile | analysis | dfs | sweep | report

    # --- error ---
    consecutive_failures: int = 0
    # `consecutive_failures` counts CRASHED actions; it triggers re-analyze.
    # `consecutive_regressions` counts actions that ran-clean but lost
    # throughput (and were reverted).  The DFS branch may be exhausted /
    # over-committed even when no action crashes — without this counter
    # the orchestrator can spin forever trying near-relatives of the
    # same losing hypothesis.  Triggers a forced re-explore.
    consecutive_regressions: int = 0
    actions_since_gain: int = 0
    actions_since_rescore: int = 0
    actions_since_bench: int = 0

    # --- timing ---
    start_time: float = field(default_factory=time.time)
    last_checkpoint_time: float = 0.0
    last_rescore_time: float = 0.0

    # --- kernel uniqueness tracking ---
    discovered_kernels: dict[str, dict[str, Any]] = field(default_factory=dict)
    kernel_attempt_count: int = 0
    unique_kernel_count: int = 0

    # --- server ---
    server_config: dict[str, Any] = field(default_factory=dict)

    # --- dashboard history ---
    score_history: list[dict[str, Any]] = field(default_factory=list)
    branch_log: list[dict[str, Any]] = field(default_factory=list)

    # --- discovery engine ---
    visit_map: dict[str, int] = field(default_factory=dict)
    action_type_history: list[str] = field(default_factory=list)
    transfer_candidates: list[dict[str, Any]] = field(default_factory=list)
    posteriors: dict[str, dict[str, float]] = field(default_factory=dict)
    failure_journal: list[dict[str, Any]] = field(default_factory=list)
    last_known_good: dict[str, Any] = field(default_factory=dict)
    defensive_rules: list[dict[str, Any]] = field(default_factory=list)
    insights_last_seen_id: str = ""

    # --- multi-objective ---
    pareto_frontier: list[dict[str, Any]] = field(default_factory=list)
    benchmark_history: list[dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Action stack ops
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_score(v: Any) -> float:
        """Coerce a score value to float (LLM may return strings)."""
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    _REJECTED_PATTERNS: ClassVar[tuple[str, ...]] = (
        "max-concurrency",
        "concurrency-128",
        "concurrency-256",
        "increase-concurrency",
        "change-concurrency",
        "conc-sweep",
    )

    def push_action(self, action: dict[str, Any]) -> None:
        action["score"] = self._safe_score(action.get("score", 0))
        aid = (action.get("id") or action.get("description") or "").lower()
        for pat in self._REJECTED_PATTERNS:
            if pat in aid:
                log.info("Rejected action %r (matches blocked pattern %r)", action.get("id"), pat)
                return
        self.action_stack.append(action)

    def pop_action(self) -> dict[str, Any] | None:
        if not self.action_stack:
            return None
        best_idx = max(range(len(self.action_stack)),
                       key=lambda i: self._safe_score(self.action_stack[i].get("score", 0)))
        return self.action_stack.pop(best_idx)

    def update_tier(self) -> str | None:
        """Recompute tier; return new tier name if boundary crossed, else None."""
        elapsed_h = (time.time() - self.start_time) / 3600
        new_tier = _tier_for_hours(elapsed_h)
        if new_tier != self.current_time_tier:
            old = self.current_time_tier
            self.current_time_tier = new_tier
            return new_tier
        return None

    def time_regime_epsilon(self, max_wall_hours: float = 24.0) -> float:
        """Exploration rate that adapts to remaining wall time."""
        elapsed = time.time() - self.start_time
        remaining_frac = max(0.0, 1.0 - elapsed / (max_wall_hours * 3600))
        if remaining_frac > 0.75:
            return 0.7
        elif remaining_frac > 0.25:
            return 0.4
        else:
            return 0.15

    def record_visit(self, file_path: str) -> None:
        """Track that an agent read this source file."""
        self.visit_map[file_path] = self.visit_map.get(file_path, 0) + 1

    def get_posterior(self, action_type: str, code_region: str = "") -> ActionPosterior:
        key = f"{action_type}|{code_region}" if code_region else action_type
        raw = self.posteriors.get(key)
        if raw and isinstance(raw, dict):
            return ActionPosterior(**{k: raw[k] for k in ActionPosterior.__dataclass_fields__ if k in raw})
        return ActionPosterior()

    def update_posterior(self, action_type: str, code_region: str,
                         success: bool, gain_pct: float = 0.0) -> None:
        key = f"{action_type}|{code_region}" if code_region else action_type
        post = self.get_posterior(action_type, code_region)
        post.update(success, gain_pct)
        self.posteriors[key] = asdict(post)

    def record_failure(self, action: dict[str, Any], symptom: str,
                       root_cause: str = "", fix: str = "") -> None:
        self.failure_journal.append({
            "timestamp": time.time(),
            "action_id": action.get("id", ""),
            "action_type": action.get("action", ""),
            "target": action.get("target_kernel", ""),
            "symptom": symptom,
            "root_cause": root_cause,
            "fix": fix,
            "files_changed": action.get("files_changed", []),
        })
        if len(self.failure_journal) > 200:
            self.failure_journal = self.failure_journal[-200:]

    def snapshot_known_good(self, benchmark_result: dict[str, Any]) -> None:
        self.last_known_good = {
            "timestamp": time.time(),
            "tput_per_gpu": self.current_tput_per_gpu,
            "cumulative_gain_pct": self.cumulative_gain_pct,
            "benchmark": benchmark_result,
            "server_config": dict(self.server_config),
            "completed_count": len(self.completed_actions),
        }

    def state_summary(self) -> str:
        """Compact JSON for LLM context windows."""
        return json.dumps({
            "model": self.model_name,
            "gpu": f"{self.gpu_count}x {self.gpu_type}",
            "tput": f"{self.best_tput_per_gpu or self.current_tput_per_gpu:.1f} tok/s/GPU (+{self.cumulative_gain_pct:.1f}%)",
            "target_gap": f"{self.target_gap_pct:.1f}%",
            "phase": self.phase,
            "tier": self.current_time_tier,
            "stack_size": len(self.action_stack),
            "completed": len(self.completed_actions),
            "km_kept": self.kernel_manager_merges_kept,
            "events": self.events_written,
            "cost_usd": round(self.total_llm_cost_usd, 2),
        }, indent=2)

    # ------------------------------------------------------------------
    # Kernel uniqueness tracking
    # ------------------------------------------------------------------

    def register_kernel(self, kernel_name: str, source_file: str = "",
                        shapes: list[str] | None = None,
                        source_hash: str = "") -> bool:
        """Register a kernel and return True if it's genuinely new (not seen before)."""
        fingerprint = self._kernel_fingerprint(kernel_name, source_file, shapes or [])
        self.kernel_attempt_count += 1
        if fingerprint in self.discovered_kernels:
            self.discovered_kernels[fingerprint]["attempt_count"] += 1
            return False
        self.discovered_kernels[fingerprint] = {
            "kernel_name": kernel_name,
            "source_file": source_file,
            "shapes": shapes or [],
            "source_hash": source_hash,
            "discovered_at": time.time(),
            "attempt_count": 1,
        }
        self.unique_kernel_count = len(self.discovered_kernels)
        return True

    @staticmethod
    def _kernel_fingerprint(kernel_name: str, source_file: str,
                            shapes: list[str]) -> str:
        text = f"{kernel_name}|{source_file}|{'|'.join(sorted(shapes))}"
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def score_action(self, action: dict[str, Any]) -> float:
        action_type = action.get("action", "")
        target_file = action.get("source_file", "")
        target_region = action.get("target_kernel", target_file)

        # --- exploitation score (original formula with priors) ---
        priors = SCORE_PRIORS.get(action_type, {}).get(self.model_class)
        if priors:
            gain = action.get("expected_gain_pct", priors[0])
            cost = action.get("cost_minutes", priors[1])
            a_risk = action.get("accuracy_risk", priors[2])
            c_risk = action.get("crash_risk", priors[3])
        else:
            gain = action.get("expected_gain_pct", 0)
            cost = action.get("cost_minutes", 30)
            a_risk = action.get("accuracy_risk", 0)
            c_risk = action.get("crash_risk", 0)
        exploitation = compute_score(gain, cost, a_risk, c_risk, self.target_gap_pct)

        # --- exploration: Thompson Sampling ---
        epsilon = self.time_regime_epsilon()
        if random.random() < epsilon:
            posterior = self.get_posterior(action_type, target_region)
            thompson = posterior.sample()
            if thompson > 0:
                exploitation = max(exploitation, thompson)

        # --- UCB confidence bonus ---
        total_completed = max(len(self.completed_actions), 1)
        region_key = f"{action_type}|{target_region}"
        region_attempts = sum(
            1 for a in self.completed_actions
            if f"{a.get('action', '')}|{a.get('target_kernel', a.get('source_file', ''))}" == region_key
        )
        ucb = 2.0 * math.sqrt(math.log(total_completed) / (1 + region_attempts))

        # --- visit novelty ---
        novelty = 1.0 / (1 + self.visit_map.get(target_file, 0)) if target_file else 0.0

        # --- transfer bonus ---
        transfer = 0.0
        for tc in self.transfer_candidates:
            if tc.get("pattern") == action.get("pattern") and tc.get("target") != target_region:
                transfer = max(transfer, tc.get("success_rate", 0.5) * 2.0)

        # --- diversity penalty ---
        diversity_penalty = 0.0
        recent = self.action_type_history[-8:] if self.action_type_history else []
        if recent:
            same_type_count = sum(1 for t in recent if t == action_type)
            diversity_penalty = 0.5 * same_type_count / len(recent)

        # --- stagnation multiplier ---
        stagnation = 1.0 + 0.3 * max(0, self.actions_since_gain - 3)

        exploration = stagnation * (ucb + novelty + transfer)
        return max(exploitation + exploration - diversity_penalty, 0.1)

    def apply_handoff_boosts(self, action: dict[str, Any]) -> float:
        boost = 0
        for tag in action.get("tags", []):
            boost += HANDOFF_BOOSTS.get(tag, 0)
        return boost

    # ------------------------------------------------------------------
    # Score update rules 1-8
    # ------------------------------------------------------------------

    def apply_update_rules(self, trigger: str, context: dict[str, Any] | None = None) -> None:
        ctx = context or {}
        if trigger == "dispatch_bug_found":
            for a in self.action_stack:
                if a.get("action") == "deep-kernel-opt":
                    a["score"] = max(self._safe_score(a.get("score", 0)), 10)
        elif trigger == "rebuild_success":
            for a in self.action_stack:
                if a.get("action") == "framework-rebuild":
                    a["score"] = self._safe_score(a.get("score", 0)) * 1.5
        elif trigger == "comm_gain":
            if ctx.get("gain_pct", 0) > 2:
                for a in self.action_stack:
                    if a.get("action") == "comm-optimization":
                        a["score"] = self._safe_score(a.get("score", 0)) * 1.5
        elif trigger == "backend_discard":
            backend = ctx.get("backend")
            if backend:
                for a in self.action_stack:
                    if a.get("backend") == backend:
                        a["score"] = self._safe_score(a.get("score", 0)) * 0.8
        elif trigger == "post_crash":
            action_id = ctx.get("action_id")
            for a in self.action_stack:
                if a.get("id") == action_id:
                    a["score"] = self._safe_score(a.get("score", 0)) * 0.3

    def apply_dream_rescores(self) -> None:
        """Post-dream score adjustments — uses bandit posteriors + structural bonuses."""
        tested = set(self.strategies_tested)
        all_strategies = {"A", "B", "C", "D", "E", "F", "G"}
        untested = all_strategies - tested

        for a in self.action_stack:
            strategy = a.get("strategy_letter", "")
            # Untested strategy categories still get a boost
            if strategy in untested:
                a["score"] = self._safe_score(a.get("score", 0)) * 1.3
            # Structural bonuses
            bugs = self.dispatch_bugs_found
            bug_count = len(bugs) if isinstance(bugs, list) else int(bugs or 0)
            if a.get("action") == "framework-rebuild" and bug_count > 0:
                a["score"] = self._safe_score(a.get("score", 0)) * 2.0
            if a.get("action") == "operator-tuning" and self.untuned_shapes:
                a["score"] = self._safe_score(a.get("score", 0)) * 1.5
            # Re-score with bandit (may boost or lower based on empirical data)
            action_type = a.get("action", "")
            target = a.get("target_kernel", a.get("source_file", ""))
            post = self.get_posterior(action_type, target)
            if post.count >= 3:
                empirical_rate = post.alpha / (post.alpha + post.beta)
                if empirical_rate < 0.2:
                    a["score"] = self._safe_score(a.get("score", 0)) * 0.5

    def detect_loop_graduated(self) -> float:
        """Returns loop severity 0.0 (none) to 1.0 (stuck). Graduated response."""
        window = self.completed_actions[-8:]
        if len(window) < 3:
            return 0.0
        tuples = [
            (a.get("action", ""), a.get("target_kernel", ""),
             a.get("result", {}).get("status", ""))
            for a in window
        ]
        unique_ratio = len(set(tuples)) / len(tuples)
        failure_count = sum(1 for t in tuples if t[2] in ("error", "crash", "reverted", "segfault"))
        severity = (1.0 - unique_ratio) * (1.0 + 0.2 * failure_count)
        return min(severity, 1.0)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    _LIST_CAPS: ClassVar[dict[str, int]] = {
        "completed_actions": 200,
        "crash_log": 100,
        "loop_signatures": 100,
        "strategies_tested": 200,
        "kernel_results": 200,
        "kernel_manager_processed_ids": 500,
        "score_history": 200,
        "branch_log": 200,
        "action_type_history": 50,
        "transfer_candidates": 50,
        "failure_journal": 50,
        "defensive_rules": 50,
        "pareto_frontier": 100,
        "benchmark_history": 200,
    }

    _DICT_CAPS: ClassVar[dict[str, int]] = {
        "discovered_kernels": 500,
        "visit_map": 500,
        "posteriors": 500,
    }

    def _cap_lists(self) -> None:
        for attr, cap in self._LIST_CAPS.items():
            lst = getattr(self, attr, None)
            if isinstance(lst, list) and len(lst) > cap:
                setattr(self, attr, lst[-cap:])
        for attr, cap in self._DICT_CAPS.items():
            d = getattr(self, attr, None)
            if isinstance(d, dict) and len(d) > cap:
                keys = list(d.keys())
                for k in keys[:-cap]:
                    del d[k]

    def save(self, path: str | Path | None = None) -> Path:
        self._cap_lists()
        if self.current_tput_per_gpu > (self.best_tput_per_gpu or 0):
            self.best_tput_per_gpu = self.current_tput_per_gpu
        p = Path(path) if path else Path(self.session_dir) / "state.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2, default=str))
        tmp.rename(p)
        return p

    @classmethod
    def load(cls, path: str | Path) -> "MarathonState":
        data = json.loads(Path(path).read_text())
        st = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        if st.current_tput_per_gpu > (st.best_tput_per_gpu or 0):
            st.best_tput_per_gpu = st.current_tput_per_gpu
        return st

    @classmethod
    def load_or_create(cls, session_dir: str, **overrides: Any) -> "MarathonState":
        p = Path(session_dir) / "state.json"
        if p.exists():
            st = cls.load(p)
            for k, v in overrides.items():
                if hasattr(st, k):
                    setattr(st, k, v)
            return st
        st = cls(session_dir=session_dir, **overrides)
        st.start_time = time.time()
        return st

    def checkpoint(self, label: str = "") -> Path:
        ts = int(time.time())
        ckpt_dir = Path(self.session_dir) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"checkpoint_{ts}.json"
        self.save(ckpt_path)
        latest_tmp = ckpt_dir / "latest.tmp"
        latest = ckpt_dir / "latest"
        try:
            latest_tmp.unlink(missing_ok=True)
            latest_tmp.symlink_to(ckpt_path.name)
            latest_tmp.rename(latest)
        except OSError:
            latest.unlink(missing_ok=True)
            latest.symlink_to(ckpt_path.name)
        self.checkpoint_path = str(ckpt_path)
        self.last_checkpoint_time = time.time()
        return ckpt_path


class StateLock:
    """Async lock protecting concurrent mutations to a shared MarathonState.

    Usage::

        slock = StateLock()

        async with slock.mutate():
            state.push_action(action)
            state.save()
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def mutate(self):
        async with self._lock:
            yield
