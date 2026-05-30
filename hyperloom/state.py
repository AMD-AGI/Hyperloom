"""Session state persistence — JSON-based, simple.

One state.json per session. Tracks throughput history, dispatched agents,
actions taken, and current optimization status.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class AgentRecord:
    """Record of a dispatched agent."""

    agent_id: str
    role: str  # "kernel", "framework", "config", etc.
    status: str = "dispatched"  # dispatched, running, completed, failed, timeout
    dispatched_at: float = 0.0
    completed_at: float = 0.0
    result_summary: str = ""
    gpu_ids: list[int] = field(default_factory=list)


@dataclass
class ActionRecord:
    """Record of an action taken during optimization."""

    action: str
    timestamp: float = 0.0
    throughput_before: float = 0.0
    throughput_after: float = 0.0
    accepted: bool = False
    details: str = ""


@dataclass
class SessionState:
    """Persistent session state."""

    session_id: str = ""
    started_at: str = ""
    model_path: str = ""
    baseline_throughput: float = 0.0
    current_throughput: float = 0.0
    best_throughput: float = 0.0
    target_gain_pct: float = 10.0
    actions: list[ActionRecord] = field(default_factory=list)
    agents: list[AgentRecord] = field(default_factory=list)
    iteration: int = 0
    status: str = "initialized"  # initialized, running, target_reached, timed_out, stopped

    @property
    def gain_pct(self) -> float:
        if self.baseline_throughput <= 0:
            return 0.0
        return (self.best_throughput / self.baseline_throughput - 1) * 100

    @property
    def target_reached(self) -> bool:
        return self.gain_pct >= self.target_gain_pct


def init_session(session_dir: str, model_path: str, target_gain: float) -> SessionState:
    """Initialize a new session state."""
    ts = datetime.now(timezone.utc).isoformat()
    session_id = Path(session_dir).name or datetime.now().strftime("%Y%m%d-%H%M%S")
    state = SessionState(
        session_id=session_id,
        started_at=ts,
        model_path=model_path,
        target_gain_pct=target_gain,
        status="initialized",
    )
    save_session(session_dir, state)
    return state


def load_session(session_dir: str) -> SessionState | None:
    """Load session state from disk. Returns None if not found."""
    path = Path(session_dir) / "state.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        actions = [ActionRecord(**a) for a in data.pop("actions", [])]
        agents = [AgentRecord(**a) for a in data.pop("agents", [])]
        return SessionState(**data, actions=actions, agents=agents)
    except (json.JSONDecodeError, TypeError, KeyError):
        return None


def save_session(session_dir: str, state: SessionState) -> None:
    """Atomically save session state to disk."""
    path = Path(session_dir) / "state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(state), indent=2, default=str) + "\n")
    os.replace(tmp, path)


def record_action(
    state: SessionState,
    action: str,
    throughput_before: float,
    throughput_after: float,
    accepted: bool,
    details: str = "",
) -> None:
    """Record an optimization action in the session history."""
    state.actions.append(ActionRecord(
        action=action,
        timestamp=time.time(),
        throughput_before=throughput_before,
        throughput_after=throughput_after,
        accepted=accepted,
        details=details,
    ))
    if accepted and throughput_after > state.best_throughput:
        state.best_throughput = throughput_after
    state.current_throughput = throughput_after
    state.iteration += 1


def record_agent(state: SessionState, agent_id: str, role: str, gpu_ids: list[int] | None = None) -> None:
    """Record a dispatched agent."""
    state.agents.append(AgentRecord(
        agent_id=agent_id,
        role=role,
        status="dispatched",
        dispatched_at=time.time(),
        gpu_ids=gpu_ids or [],
    ))


def update_agent_status(state: SessionState, agent_id: str, status: str, result_summary: str = "") -> None:
    """Update the status of a dispatched agent."""
    for agent in state.agents:
        if agent.agent_id == agent_id:
            agent.status = status
            agent.result_summary = result_summary
            if status in ("completed", "failed", "timeout"):
                agent.completed_at = time.time()
            break


def get_intervention_mix(state: SessionState) -> dict[str, Any]:
    """Analyze the mix of config vs code-patch interventions."""
    config_count = 0
    code_count = 0
    for a in state.actions:
        change_type = "config"
        if "patch" in a.action.lower() or "code" in a.action.lower():
            change_type = "code_patch"
        if change_type == "config":
            config_count += 1
        else:
            code_count += 1

    consecutive_config = 0
    for a in reversed(state.actions):
        if "patch" in a.action.lower() or "code" in a.action.lower():
            break
        consecutive_config += 1

    return {
        "config_count": config_count,
        "code_patch_count": code_count,
        "total": config_count + code_count,
        "consecutive_config_only": consecutive_config,
        "config_heavy": consecutive_config >= 2 and code_count == 0,
    }


# ─── Depth tracking (prevents premature stopping) ─────────────────────────────


@dataclass
class DepthTracker:
    """Tracks exploration depth to prevent premature stopping.

    The orchestrator must exhaust multiple levels of exploration before
    it is allowed to declare a plateau and stop.
    """

    kb_items_tried: list[str] = field(default_factory=list)
    research_scout_runs: int = 0
    prs_fetched: list[str] = field(default_factory=list)
    pr_diffs_read: list[str] = field(default_factory=list)
    nvidia_refs_compared: list[str] = field(default_factory=list)
    code_patches_attempted: int = 0
    config_changes_attempted: int = 0
    specialist_failures: int = 0
    specialist_retries: int = 0
    consecutive_reverts: int = 0

    def record_revert(self) -> None:
        self.consecutive_reverts += 1

    def record_kept(self) -> None:
        self.consecutive_reverts = 0

    def record_research_scout(self) -> None:
        self.research_scout_runs += 1

    def record_pr_fetch(self, pr_id: str) -> None:
        if pr_id not in self.prs_fetched:
            self.prs_fetched.append(pr_id)

    def record_pr_diff_read(self, pr_id: str) -> None:
        if pr_id not in self.pr_diffs_read:
            self.pr_diffs_read.append(pr_id)

    def record_nvidia_ref(self, ref: str) -> None:
        if ref not in self.nvidia_refs_compared:
            self.nvidia_refs_compared.append(ref)

    def record_specialist_failure(self) -> None:
        self.specialist_failures += 1

    def record_specialist_retry(self) -> None:
        self.specialist_retries += 1

    def should_stop(self) -> tuple[bool, str]:
        """Tiered stopping: only stop if multiple exploration levels exhausted."""
        if self.consecutive_reverts < 3:
            return False, ""

        blockers = []

        if self.research_scout_runs < 2:
            blockers.append(
                f"Only {self.research_scout_runs} research scout runs — "
                f"re-run scout for new optimizations"
            )
        if len(self.prs_fetched) < 5:
            blockers.append(
                f"Only {len(self.prs_fetched)} PRs fetched — "
                f"search GitHub for model/architecture keywords"
            )
        if len(self.pr_diffs_read) < 3:
            blockers.append(
                f"Only {len(self.pr_diffs_read)} PR diffs read — "
                f"read actual diffs to find portable optimizations"
            )
        if len(self.nvidia_refs_compared) < 2:
            blockers.append(
                f"Only {len(self.nvidia_refs_compared)} NVIDIA references compared — "
                f"study NVIDIA configs and translate to AMD"
            )
        if self.code_patches_attempted < 1:
            blockers.append(
                "No code patches attempted — dispatch code-patch specialist"
            )

        if blockers:
            return False, (
                f"Cannot stop yet — {self.consecutive_reverts} consecutive reverts but "
                f"exploration is shallow:\n"
                + "\n".join(f"  - {b}" for b in blockers)
            )

        return True, (
            f"Plateau confirmed: {self.consecutive_reverts} reverts, "
            f"{self.research_scout_runs} scout runs, "
            f"{len(self.prs_fetched)} PRs, "
            f"{self.code_patches_attempted} code patches tried."
        )

    def next_exploration_action(self) -> str:
        """Suggest the next deepening action when reverts stack up."""
        if self.research_scout_runs < 2:
            return (
                "DEEPEN: Re-run Research Scout. Focus on NVIDIA differences, "
                "recent merged PRs, and kernel-level tuning flags."
            )
        if len(self.prs_fetched) < 5:
            return (
                "DEEPEN: Search GitHub PRs using model architecture keywords "
                "across vLLM, SGLang, TRT-LLM, ROCm/aiter repos."
            )
        if len(self.pr_diffs_read) < 3:
            return (
                "DEEPEN: Read actual PR diffs for the most promising PRs. "
                "Extract env vars, flags, and code changes."
            )
        if len(self.nvidia_refs_compared) < 2:
            return (
                "DEEPEN: Study NVIDIA's full serving stack for this model. "
                "Compare every flag with ours."
            )
        if self.code_patches_attempted < 1:
            return (
                "DEEPEN: Config tuning exhausted. Dispatch code-patch specialist "
                "for scheduler/kernel/memory modifications."
            )
        return (
            "All exploration levels exhausted. Consider stopping or trying "
            "completely novel approaches."
        )

    def to_dict(self) -> dict:
        return {
            "kb_items_tried": self.kb_items_tried,
            "research_scout_runs": self.research_scout_runs,
            "prs_fetched": self.prs_fetched,
            "pr_diffs_read": self.pr_diffs_read,
            "nvidia_refs_compared": self.nvidia_refs_compared,
            "code_patches_attempted": self.code_patches_attempted,
            "config_changes_attempted": self.config_changes_attempted,
            "specialist_failures": self.specialist_failures,
            "specialist_retries": self.specialist_retries,
            "consecutive_reverts": self.consecutive_reverts,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DepthTracker":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def load_depth_tracker(session_dir: str) -> DepthTracker:
    path = Path(session_dir) / "depth_tracker.json"
    if not path.exists():
        return DepthTracker()
    try:
        data = json.loads(path.read_text())
        return DepthTracker.from_dict(data)
    except (json.JSONDecodeError, TypeError):
        return DepthTracker()


def save_depth_tracker(session_dir: str, tracker: DepthTracker) -> None:
    path = Path(session_dir) / "depth_tracker.json"
    tmp = path.with_suffix(".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(tracker.to_dict(), indent=2) + "\n")
    os.replace(tmp, path)


# ─── Competitor target benchmarks ──────────────────────────────────────────────


@dataclass
class TargetDatapoint:
    """A single benchmark data point from a competitor platform."""

    concurrency: int
    throughput_per_gpu: float
    tpot_ms: float
    ttft_ms: float = 0.0

    @property
    def interactivity(self) -> float:
        return 1000.0 / self.tpot_ms if self.tpot_ms > 0 else 0.0


@dataclass
class CompetitorTarget:
    """Benchmark targets from a competitor for gap-aware optimization."""

    name: str
    gpu_type: str
    gpu_count: int
    tp: int
    framework: str
    model: str
    isl: int
    osl: int
    mtp_enabled: bool = False
    datapoints: list[TargetDatapoint] = field(default_factory=list)
    notes: str = ""

    def gap_analysis(
        self,
        our_throughput_per_gpu: float,
        our_tpot_ms: float,
        concurrency: int,
    ) -> dict[str, Any]:
        """Compute gap between our results and target at a given concurrency."""
        match = None
        for dp in self.datapoints:
            if dp.concurrency == concurrency:
                match = dp
                break
        if not match:
            match = min(self.datapoints, key=lambda d: abs(d.concurrency - concurrency))

        throughput_gap_pct = (
            (match.throughput_per_gpu - our_throughput_per_gpu) / match.throughput_per_gpu * 100
            if match.throughput_per_gpu > 0 else 0
        )
        tpot_ratio = our_tpot_ms / match.tpot_ms if match.tpot_ms > 0 else 0
        interactivity_gap_pct = (
            (match.interactivity - (1000.0 / our_tpot_ms)) / match.interactivity * 100
            if match.interactivity > 0 and our_tpot_ms > 0 else 0
        )

        return {
            "concurrency": concurrency,
            "target_throughput": match.throughput_per_gpu,
            "our_throughput": our_throughput_per_gpu,
            "throughput_gap_pct": round(throughput_gap_pct, 1),
            "target_tpot_ms": match.tpot_ms,
            "our_tpot_ms": our_tpot_ms,
            "tpot_ratio": round(tpot_ratio, 2),
            "interactivity_gap_pct": round(interactivity_gap_pct, 1),
        }

    def full_gap_summary(self, our_datapoints: list[dict]) -> str:
        """Generate human-readable gap summary for orchestrator prompt."""
        lines = [
            f"## Gap: Ours vs {self.name} ({self.gpu_type}x{self.gpu_count} TP={self.tp})",
            f"Target: {self.model} on {self.framework}, ISL={self.isl}/OSL={self.osl}",
            "",
        ]

        worst_tpot_ratio = 0.0
        worst_throughput_gap = 0.0

        for dp in our_datapoints:
            gap = self.gap_analysis(dp["throughput_per_gpu"], dp["tpot_ms"], dp["concurrency"])
            lines.append(
                f"  C={gap['concurrency']:>3}: "
                f"tput {gap['our_throughput']:.1f} vs {gap['target_throughput']:.1f} "
                f"({gap['throughput_gap_pct']:+.1f}%), "
                f"TPOT {gap['our_tpot_ms']:.1f}ms vs {gap['target_tpot_ms']:.1f}ms "
                f"({gap['tpot_ratio']:.2f}x)"
            )
            if gap["tpot_ratio"] > worst_tpot_ratio:
                worst_tpot_ratio = gap["tpot_ratio"]
            if gap["throughput_gap_pct"] > worst_throughput_gap:
                worst_throughput_gap = gap["throughput_gap_pct"]

        lines.append("")
        if worst_tpot_ratio > 1.3:
            lines.append(">>> INTERACTIVITY (TPOT) IS THE PRIMARY GAP <<<")
        if worst_throughput_gap > 10:
            lines.append(">>> THROUGHPUT GAP IS SIGNIFICANT <<<")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "gpu_type": self.gpu_type,
            "gpu_count": self.gpu_count,
            "tp": self.tp,
            "framework": self.framework,
            "model": self.model,
            "isl": self.isl,
            "osl": self.osl,
            "mtp_enabled": self.mtp_enabled,
            "notes": self.notes,
            "datapoints": [
                {"concurrency": d.concurrency, "throughput_per_gpu": d.throughput_per_gpu,
                 "tpot_ms": d.tpot_ms, "ttft_ms": d.ttft_ms}
                for d in self.datapoints
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CompetitorTarget":
        dps = [TargetDatapoint(**d) for d in data.get("datapoints", [])]
        return cls(
            name=data["name"],
            gpu_type=data["gpu_type"],
            gpu_count=data["gpu_count"],
            tp=data["tp"],
            framework=data["framework"],
            model=data["model"],
            isl=data["isl"],
            osl=data["osl"],
            mtp_enabled=data.get("mtp_enabled", False),
            datapoints=dps,
            notes=data.get("notes", ""),
        )


def load_targets(path: str) -> list[CompetitorTarget]:
    """Load competitor target benchmarks from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    targets = data if isinstance(data, list) else [data]
    return [CompetitorTarget.from_dict(t) for t in targets]


def save_targets(path: str, targets: list[CompetitorTarget]) -> None:
    """Save competitor targets to a JSON file."""
    data = [t.to_dict() for t in targets]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2) + "\n")
