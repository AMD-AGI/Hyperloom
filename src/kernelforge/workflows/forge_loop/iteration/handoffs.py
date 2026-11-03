"""Immutable per-iteration handoffs for planning and recovery consumers."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from kernelforge.workflows.forge_loop.campaign.experiments_layout import ExperimentsLayout, tagged_iteration
from hyperloom_common.io import DURABLE, atomic_write_text


@dataclass(frozen=True)
class IterationHandoff:
    """Compact machine-readable outcome passed to the next planning cycle."""

    iteration: int
    analysis_commit: str
    canonical_verdict: str
    search_mode: str = "EXPLOIT"
    search_reason_codes: tuple[str, ...] = ()
    search_objective: str = "IMMEDIATE_CANONICAL_GAIN"
    search_mode_residence_remaining: int = 0
    diversification_cycle_complete: bool = False
    optimization_plan_path: str = ""
    supervisor_ruling_path: str = ""
    plan: str = ""
    lesson_path: str = ""
    orchestration_artifacts: str = ""
    candidate_archive: str = ""
    agent_profile_id: str = ""
    agent_provider: str = ""
    agent_model: str = ""
    blockage_epoch: int = 0
    intervention_kind: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.iteration, bool) or not isinstance(self.iteration, int) or self.iteration <= 0:
            raise ValueError("handoff iteration must be a positive integer")
        if not self.analysis_commit.strip():
            raise ValueError("handoff analysis_commit is required")
        if not self.canonical_verdict.strip():
            raise ValueError("handoff canonical_verdict is required")
        if self.search_mode not in {"EXPLOIT", "DIVERSIFY"}:
            raise ValueError("handoff search_mode is unsupported")
        if self.search_mode_residence_remaining < 0:
            raise ValueError("handoff search_mode_residence_remaining must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "complete": True,
            "iteration": self.iteration,
            "analysis_commit": self.analysis_commit,
            "canonical_verdict": self.canonical_verdict,
            "search_policy": {
                "mode": self.search_mode,
                "reason_codes": list(self.search_reason_codes),
                "objective_kind": self.search_objective,
                "residence_iterations_remaining": (self.search_mode_residence_remaining),
                "diversification_cycle_complete": (self.diversification_cycle_complete),
            },
            "optimization_plan_path": self.optimization_plan_path,
            "supervisor_ruling_path": self.supervisor_ruling_path,
            "plan": self.plan,
            "lesson_path": self.lesson_path,
            "orchestration_artifacts": self.orchestration_artifacts,
            "candidate_archive": self.candidate_archive,
            "agent_profile_id": self.agent_profile_id,
            "agent_provider": self.agent_provider,
            "agent_model": self.agent_model,
            "blockage_epoch": self.blockage_epoch,
            "intervention_kind": self.intervention_kind,
        }


class HandoffStore:
    """Atomically persist and retrieve immutable iteration handoffs."""

    def __init__(self, workspace_dir: str) -> None:
        self.workspace = Path(workspace_dir).resolve()
        self._layout = ExperimentsLayout.of(self.workspace)
        self.root = self._layout.handoffs
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, iteration: int) -> Path:
        return self._layout.handoff(iteration)

    @staticmethod
    def _without_timestamp(payload: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in payload.items() if key != "created_at"}

    def write(self, handoff: IterationHandoff) -> Path:
        """Write once; repeated identical writes are idempotent."""
        destination = self.path(handoff.iteration)
        payload = handoff.to_dict()
        if destination.is_file():
            existing = json.loads(destination.read_text())
            if self._without_timestamp(existing) != payload:
                raise ValueError(f"handoff conflicts with existing iteration {handoff.iteration}")
            return destination

        payload = {
            **payload,
            "created_at": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(),
            ),
        }
        atomic_write_text(destination, json.dumps(payload, indent=2, sort_keys=True) + "\n", options=DURABLE)
        return destination

    def read(self, iteration: int) -> dict[str, Any]:
        """Read one complete handoff, returning an empty dict when absent."""
        path = self.path(iteration)
        if not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid handoff: {path}") from error
        if not isinstance(payload, dict):
            raise ValueError(f"handoff must be an object: {path}")
        if payload.get("complete") is not True:
            raise ValueError(f"incomplete handoff: {path}")
        relay_fields = {
            "agent_profile_id",
            "agent_provider",
            "agent_model",
            "blockage_epoch",
            "intervention_kind",
        }
        if not relay_fields & set(payload):
            payload.update(
                {
                    "agent_profile_id": "",
                    "agent_provider": "",
                    "agent_model": "",
                    "blockage_epoch": 0,
                    "intervention_kind": "",
                }
            )
        expected = {
            "complete",
            "iteration",
            "analysis_commit",
            "canonical_verdict",
            "search_policy",
            "optimization_plan_path",
            "supervisor_ruling_path",
            "plan",
            "lesson_path",
            "orchestration_artifacts",
            "candidate_archive",
            "agent_profile_id",
            "agent_provider",
            "agent_model",
            "blockage_epoch",
            "intervention_kind",
            "created_at",
        }
        missing = expected - set(payload)
        unknown = set(payload) - expected
        if missing:
            raise ValueError("handoff missing fields: " + ", ".join(sorted(missing)))
        if unknown:
            raise ValueError("handoff has unknown fields: " + ", ".join(sorted(unknown)))
        return payload

    def latest(self) -> tuple[Path, dict[str, Any]] | None:
        """Return the latest complete handoff."""
        for path in sorted(self.root.glob("*.json"), reverse=True):
            iteration = tagged_iteration(path.stem)
            if iteration is None:
                continue
            payload = self.read(iteration)
            if payload:
                return path, payload
        return None
