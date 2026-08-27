# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PostMortem — extract lessons from experiments and grow the knowledge base.

After each experiment completes, the PostMortem analyzer:
  1. Reviews the full iteration history
  2. Identifies what worked and what failed
  3. Extracts reusable lessons as structured knowledge
  4. Writes new knowledge files or updates existing ones
  5. Captures non-obvious findings (the "surprises")

This is how agents get stronger over time — each experiment
leaves behind knowledge that future experiments can use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from kernelforge.tracker.schema import Experiment
from kernelforge.loop.scoring import DEFAULT_SNR_THRESHOLD_DB


@dataclass
class Lesson:
    """A lesson extracted from an experiment."""

    title: str
    category: str  # "pitfall", "optimization", "methodology", "config"
    backend: str  # "ck", "flydsl", "triton", "aiter", "shared"
    description: str
    evidence: str  # What measurement/observation supports this
    actionable: str  # What to do differently next time
    experiment_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class PostMortem:
    """Extracts and persists lessons from completed experiments.

    Usage:
        pm = PostMortem(knowledge_dir=writable_knowledge_root())
        lessons = pm.analyze(experiment)
        pm.save_lessons(lessons)
    """

    def __init__(self, knowledge_dir: str | Path):
        self.knowledge_dir = Path(knowledge_dir)

    def analyze(self, experiment: Experiment) -> list[Lesson]:
        """Analyze an experiment and extract lessons.

        Looks for:
          - Configurations that caused regressions (pitfalls to avoid)
          - Changes that gave big improvements (optimizations to remember)
          - Unexpected PMC counter patterns
          - Occupancy cliffs (VGPR transitions)
          - Plateau patterns (what was tried when stuck)
        """
        lessons = []

        if not experiment.iterations:
            return lessons

        # 1. Find big regressions — these are pitfalls
        for i, it in enumerate(experiment.iterations[1:], 1):
            prev = experiment.iterations[i - 1]
            if it.wall_ms and prev.wall_ms and it.wall_ms > prev.wall_ms * 1.15:  # >15% regression
                lessons.append(
                    Lesson(
                        title=f"Config regression: {it.config}",
                        category="pitfall",
                        backend=experiment.backend,
                        description=(
                            f"Iteration {it.iteration_id} regressed {prev.wall_ms:.3f} → "
                            f"{it.wall_ms:.3f} ms (+{(it.wall_ms / prev.wall_ms - 1) * 100:.0f}%)"
                        ),
                        evidence=f"Config: {it.config}, PMC: {it.pmc_diagnosis}",
                        actionable=f"Avoid this configuration. Decision was: {it.decision}",
                        experiment_id=experiment.experiment_id,
                    )
                )

        # 2. Find big improvements — these are optimizations
        for i, it in enumerate(experiment.iterations[1:], 1):
            prev = experiment.iterations[i - 1]
            if it.wall_ms and prev.wall_ms and it.wall_ms < prev.wall_ms * 0.85:  # >15% improvement
                lessons.append(
                    Lesson(
                        title=f"Effective optimization: {it.decision}",
                        category="optimization",
                        backend=experiment.backend,
                        description=(
                            f"Iteration {it.iteration_id} improved {prev.wall_ms:.3f} → "
                            f"{it.wall_ms:.3f} ms ({prev.wall_ms / it.wall_ms:.2f}x speedup)"
                        ),
                        evidence=f"Config: {it.config}, PMC: {it.pmc_diagnosis}",
                        actionable="This optimization worked. Consider for similar kernels.",
                        experiment_id=experiment.experiment_id,
                    )
                )

        # 3. Occupancy transitions
        for i, it in enumerate(experiment.iterations[1:], 1):
            prev = experiment.iterations[i - 1]
            if it.vgpr and prev.vgpr:
                # Crossed the 256 boundary
                if (prev.vgpr <= 256 and it.vgpr > 256) or (prev.vgpr > 256 and it.vgpr <= 256):
                    direction = "dropped" if it.vgpr > 256 else "gained"
                    lessons.append(
                        Lesson(
                            title=f"Occupancy {direction}: VGPR {prev.vgpr} → {it.vgpr}",
                            category="pitfall" if direction == "dropped" else "optimization",
                            backend=experiment.backend,
                            description=(
                                f"VGPR crossed 256 boundary: {prev.vgpr} → {it.vgpr}. "
                                f"Wall time: {prev.wall_ms} → {it.wall_ms} ms"
                            ),
                            evidence="gfx950 occupancy=2 requires VGPR ≤ 256",
                            actionable=(
                                f"Watch for occupancy cliff. "
                                f"{'Reduce register pressure.' if direction == 'dropped' else 'This register reduction paid off.'}"
                            ),
                            experiment_id=experiment.experiment_id,
                        )
                    )

        # 4. SNR failures — correctness traps
        for it in experiment.iterations:
            if it.snr_db is not None and it.snr_db < DEFAULT_SNR_THRESHOLD_DB:
                lessons.append(
                    Lesson(
                        title=f"Correctness failure: SNR {it.snr_db:.1f} dB",
                        category="pitfall",
                        backend=experiment.backend,
                        description=(
                            f"Config {it.config} produced SNR {it.snr_db:.1f} dB "
                            f"(< the {DEFAULT_SNR_THRESHOLD_DB:g} dB pre-filter)"
                        ),
                        evidence=f"Iteration {it.iteration_id}",
                        actionable="This configuration causes numerical issues. Do not use.",
                        experiment_id=experiment.experiment_id,
                    )
                )

        # 5. Plateau analysis — what was the state when stuck
        if experiment.is_plateaued():
            last_few = experiment.iterations[-3:]
            lessons.append(
                Lesson(
                    title=f"Plateau at {last_few[-1].wall_ms:.3f} ms",
                    category="methodology",
                    backend=experiment.backend,
                    description=(
                        f"Plateaued after {len(experiment.iterations)} iterations. "
                        f"Last 3 wall_ms: {[it.wall_ms for it in last_few]}"
                    ),
                    evidence=f"PMC at plateau: {last_few[-1].pmc_diagnosis}",
                    actionable=(
                        "At this plateau, consider: "
                        "1) Switch to a different backend, "
                        "2) Try hybrid strategy, "
                        "3) Move to module-level optimization"
                    ),
                    experiment_id=experiment.experiment_id,
                )
            )

        return lessons

    def save_lessons(self, lessons: list[Lesson]) -> list[Path]:
        """Write lessons to the knowledge base as markdown files.

        New lessons are appended to the appropriate backend's learned/ directory.
        ``knowledge_dir`` must be a writable root (see
        ``kernelforge.resources.writable_knowledge_root``), never the packaged
        curated tree.
        """
        saved = []
        for lesson in lessons:
            # Determine target directory
            backend_dir = self.knowledge_dir / lesson.backend
            learned_dir = backend_dir / "learned"
            learned_dir.mkdir(parents=True, exist_ok=True)

            # Generate filename from title
            safe_title = "".join(c if c.isalnum() or c in "-_ " else "" for c in lesson.title)
            safe_title = safe_title.strip().replace(" ", "_")[:60]
            filename = f"{lesson.category}_{safe_title}.md"
            filepath = learned_dir / filename

            # Write or append
            content = f"""# {lesson.title}

**Category**: {lesson.category}
**Backend**: {lesson.backend}
**Experiment**: {lesson.experiment_id}
**Date**: {lesson.timestamp}

## What Happened

{lesson.description}

## Evidence

{lesson.evidence}

## What To Do

{lesson.actionable}
"""
            filepath.write_text(content)
            saved.append(filepath)

        return saved

    def summary(self, lessons: list[Lesson]) -> str:
        """Generate a human-readable summary of lessons learned."""
        if not lessons:
            return "No lessons extracted from this experiment."

        lines = [f"## Lessons Learned ({len(lessons)} findings)\n"]
        by_category = {}
        for l in lessons:
            by_category.setdefault(l.category, []).append(l)

        for cat, cat_lessons in by_category.items():
            lines.append(f"\n### {cat.title()} ({len(cat_lessons)})")
            for l in cat_lessons:
                lines.append(f"- **{l.title}**: {l.actionable}")

        return "\n".join(lines)
