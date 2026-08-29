# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Auto-evolution pipeline — continuous knowledge base growth.

Hooks into the experiment lifecycle so learning happens automatically:

  1. AFTER every benchmark → log to the tuning DB
  2. AFTER an experiment ends → run the postmortem, extract lessons,
     discover transfer rules
"""

from __future__ import annotations

from typing import Any

from kernelforge.config import Config
from kernelforge.learning.postmortem import PostMortem
from kernelforge.learning.tuning_db import TuningDatabase
from kernelforge.resources import writable_knowledge_root
from kernelforge.tracker.schema import Experiment
from kernelforge.loop.scoring import DEFAULT_SNR_THRESHOLD_DB


class AutoEvolver:
    """Drives the tuning database and the postmortem from loop events."""

    def __init__(
        self,
        tuning_db: TuningDatabase,
        postmortem: PostMortem,
    ):
        self.tuning_db = tuning_db
        self.postmortem = postmortem

    @classmethod
    def from_config(cls, config: Config) -> AutoEvolver:
        """Create an AutoEvolver from standard config.

        Both sinks are *writers*, so they target the writable knowledge root --
        a directory next to the user's experiments, not anything inside the
        installed package.
        """
        kb_dir = writable_knowledge_root()
        return cls(
            tuning_db=TuningDatabase(kb_dir / "tuning_db"),
            postmortem=PostMortem(kb_dir),
        )

    # ─── Trigger 1: After every benchmark ───

    def on_benchmark(
        self,
        operation: str,
        backend: str,
        shape: dict[str, int],
        config: dict[str, Any],
        wall_ms: float,
        snr_db: float | None = None,
        passed_correctness: bool = True,
        pmc_diagnosis: str = "",
        vgpr: int | None = None,
        experiment_id: str = "",
        gpu_target: str = "gfx950",
        dtype: str = "bf16",
    ) -> None:
        """Log benchmark result to tuning DB. Called after every bench."""
        self.tuning_db.log(
            operation=operation,
            backend=backend,
            gpu_target=gpu_target,
            dtype=dtype,
            shape=shape,
            config=config,
            wall_ms=wall_ms,
            snr_db=snr_db,
            passed_correctness=passed_correctness,
            pmc_diagnosis=pmc_diagnosis,
            vgpr=vgpr,
            experiment_id=experiment_id,
        )

    # ─── Trigger 2: After experiment completes ───

    def on_experiment_complete(self, experiment: Experiment) -> dict:
        """Full post-experiment learning. Returns summary of what was learned."""
        results = {
            "lessons": [],
            "skills": [],
            "transfer_rules": [],
        }

        # Extract lessons
        lessons = self.postmortem.analyze(experiment)
        if lessons:
            self.postmortem.save_lessons(lessons)
            results["lessons"] = [l.title for l in lessons]

        # Log all iterations to tuning DB (if not already logged)
        for it in experiment.iterations:
            if it.wall_ms is not None and it.snr_db is not None:
                self.tuning_db.log(
                    operation=experiment.task_id,
                    backend=experiment.backend,
                    gpu_target="gfx950",
                    dtype="bf16",
                    shape=it.config.get("shape", {}),
                    config={k: v for k, v in it.config.items() if k != "shape"},
                    wall_ms=it.wall_ms,
                    snr_db=it.snr_db,
                    passed_correctness=it.snr_db >= DEFAULT_SNR_THRESHOLD_DB,
                    pmc_diagnosis=it.pmc_diagnosis,
                    vgpr=it.vgpr,
                    experiment_id=experiment.experiment_id,
                )

        # Discover new transfer rules from accumulated data
        new_rules = self.tuning_db.discover_transfer_rules()
        for rule in new_rules:
            self.tuning_db.add_transfer_rule(
                rule_id=rule.rule_id,
                description=rule.description,
                scope=rule.scope,
                parameter=rule.parameter,
                recommended_value=rule.recommended_value,
                anti_value=rule.anti_value,
                evidence=rule.evidence,
            )
            results["transfer_rules"].append(rule.description)

        return results
