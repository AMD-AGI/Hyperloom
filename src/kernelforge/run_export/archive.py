# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What Kernel Forge archives for an iteration.

An iteration's archive is its candidate (``candidates/iter_NNN``), the plan it was built from, among the planning
files of its round (``orchestration/iter_NNN``), the reasoning log the implementer kept beside that plan
(``<plan>.reasoning.md``), its handoff (``handoffs/iter_NNN.json``), and its lesson (``lessons/iter_NNN.md``). An
Experience keeps the whole reasoning log, which ``implementer_fields`` reads, and excerpts of the rest of the prose:
``plan_text`` for a plan, which keeps the whole Summary line a plan opens with, and ``first_sentences`` for the
others.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kernelforge.knowledge.kb_store.identity.normalization import UNKNOWN_SEGMENT
from kernelforge.knowledge.kb_store.identity.run import LoopKernel, resolve_loop_identity
from kernelforge.orchestrator.summary_line import read_summary, reasoning_log_path
from kernelforge.run_export.files import read_json, read_text
from kernelforge.run_export.units import CandidateOrigin

#: How many sentences of a plan without a Summary line, of a single session's rationale, or of a lesson an
#: Experience keeps.
_EXCERPT_SENTENCES = 3
#: The sentence KernelForge opens a plan with when it could not revise the draft and put the critic's review in front
#: of it instead (``_render_critic_revision_fallback`` in ``kernelforge/orchestrator/orchestration.py`` at 1c14e79e1).
_FALLBACK_PREFACE = "The Orchestration revision was unavailable."
_FALLBACK_DRAFT = "\n## Original draft\n"
_HEADING = re.compile(r"^#{1,6} (\S.*)$")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z`*\"'(\[])")
_LIST_MARKER = re.compile(r"^(?:[-*+]|\d+[.)])\s+")


def first_sentences(text: str, count: int = _EXCERPT_SENTENCES) -> str | None:
    """The first ``count`` sentences of ``text``'s prose, copied.

    Headings, code blocks, tables, and rules are skipped, and the end of a paragraph ends a sentence.
    """

    sentences: list[str] = []
    paragraph: list[str] = []
    fenced = False
    for line in [*text.splitlines(), ""]:
        stripped = line.strip()
        if stripped.startswith("```"):
            fenced = not fenced
        elif not fenced and stripped and not _HEADING.match(stripped) and not stripped.startswith("|"):
            if not set(stripped) <= set("-*_="):
                paragraph.append(_LIST_MARKER.sub("", stripped))
                continue
        if paragraph:
            sentences.extend(_SENTENCE_END.split(" ".join(paragraph)))
            paragraph.clear()
    return " ".join(sentences[:count]).strip() or None


def _draft(plan: str) -> str:
    """The plan as its planner wrote it: the original draft when KernelForge could not revise it and put the critic's
    review in front of it instead."""

    if _FALLBACK_PREFACE in plan and _FALLBACK_DRAFT in plan:
        return plan.split(_FALLBACK_DRAFT, 1)[1]
    return plan


def _plan_body(plan: str) -> str:
    """The plan from its first heading on.

    Lines before the first heading are the planner's own narration, which KernelForge archives with the plan.
    """

    lines = plan.splitlines()
    first = next((index for index, line in enumerate(lines) if _HEADING.match(line)), 0)
    return "\n".join(lines[first:])


def plan_text(plan: str) -> str | None:
    """What an Experience keeps of a whole plan file: the whole Summary line its draft opens with, or the first
    sentences of the draft's body when it has none."""

    draft = _draft(plan)
    return read_summary(draft) or first_sentences(_plan_body(draft))


def campaign_identity(config: dict[str, Any]) -> dict[str, Any]:
    """The run's conditions as its ``campaign_config.json`` records them; empty and unknown ones are left out.

    ``kernel`` is the name Kernel Forge files the run's own KB records under, which it derives from the operator.
    """

    filed, _operation, _framework = resolve_loop_identity(
        LoopKernel(
            kernel_path=str(config.get("kernel_path") or ""),
            kernel_backend=str(config.get("kernel_backend") or ""),
            target_functions=config.get("target_functions"),
            source_files=config.get("source_files"),
            framework=str(config.get("framework") or ""),
            operator_name=str(config.get("operator_name") or ""),
        ),
        gpu_type=str(config.get("gpu_type") or ""),
    )
    identity = {
        "operator": config.get("operator_name"),
        "kernel": filed.kernel_name,
        "framework": config.get("framework"),
        "backend": config.get("kernel_backend"),
        "gpu": config.get("gpu_type"),
    }
    return {key: value for key, value in identity.items() if value not in (None, "", UNKNOWN_SEGMENT)}


def _lane_plan_file(experiments: Path, seq: int, plan: str) -> Path | None:
    """The file of the fan-out round that holds this lane's plan: optimization_plan.md for the round's first lane,
    lane_NNN.md for its others.

    A round's later lanes are measured in later iterations, which have no planning directory of their own.
    """

    for number in range(seq, 0, -1):
        directory = experiments / "orchestration" / f"iter_{number:03d}"
        for path in [directory / "optimization_plan.md", *sorted(directory.glob("lane_[0-9][0-9][0-9].md"))]:
            if path.is_file() and read_text(path).strip() == plan:
                return path
    return None


@dataclass(frozen=True)
class IterationArchive:
    """What Kernel Forge archived for iteration ``seq`` under a run's ``forge_experiments`` directory."""

    experiments: Path
    seq: int

    @property
    def tag(self) -> str:
        return f"iter_{self.seq:03d}"

    @property
    def candidate(self) -> Path:
        return self.experiments / "candidates" / self.tag

    @property
    def handoff(self) -> Path:
        return self.experiments / "handoffs" / f"{self.tag}.json"

    @property
    def lesson(self) -> Path:
        return self.experiments / "lessons" / f"{self.tag}.md"

    def meta(self) -> dict[str, Any] | None:
        """The candidate's ``meta.json``, which Kernel Forge writes for every attempt that produced a diff."""

        return read_json(self.candidate / "meta.json")

    def plan_file(self, origin: CandidateOrigin, plan: str) -> Path | None:
        """The plan the candidate was built from: its lane's plan, or the plan of the round a single session ran.

        ``plan`` is the plan ``meta.json`` records, which for a lane is the whole of the lane's plan.
        """

        found: Path | None = None
        if origin is CandidateOrigin.LANE and plan:
            found = _lane_plan_file(self.experiments, self.seq, plan)
        elif origin is CandidateOrigin.SESSION:
            relative = str((read_json(self.handoff) or {}).get("optimization_plan_path") or "")
            found = self.experiments.parent / relative if relative else None
        return found if found is not None and found.is_file() else None

    def fields(self, meta: dict[str, Any], plan_file: Path | None) -> dict[str, Any]:
        """The unit fields the archive holds: the plan, the baseline time, the kernel, the lesson.

        The baseline time is the pristine one the run's state records, the original kernel's. ``meta.json`` holds the
        time of the kernel the search started from, which is the same unless the run was given ``--baseline-json``,
        so it stands in only for a run state that records none.
        """

        fields: dict[str, Any] = {}
        if plan_file is not None:
            fields["plan_file"] = str(plan_file.resolve())
            fields["plan"] = plan_text(read_text(plan_file))
        pristine = (read_json(self.experiments / "run_state.json") or {}).get("pristine_baseline_wall_ms")
        fields["source_ms"] = pristine if pristine is not None else meta.get("baseline_wall_ms")
        kernel = str((meta.get("files") or {}).get("kernel") or "")
        if kernel and (self.candidate / kernel).is_file():
            fields["source_code"] = str((self.candidate / kernel).resolve())
        if self.lesson.is_file():
            fields["lesson_file"] = str(self.lesson.resolve())
            fields["lesson"] = first_sentences(read_text(self.lesson))
        return fields


def implementer_fields(plan_file: Path | None) -> dict[str, Any]:
    """What the session that made a candidate recorded of its work: the whole reasoning log it kept beside
    ``plan_file``, the reasoning whatever else gave one; empty when it kept none."""

    log = reasoning_log_path(plan_file) if plan_file is not None else None
    record = read_text(log).strip() if log is not None and log.is_file() else ""
    return {"reasoning": record} if record else {}
