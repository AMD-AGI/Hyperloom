# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What Kernel Forge archives for an iteration.

An iteration's archive is its candidate (``candidates/iter_NNN``), the plan it was built from, among the planning
files of its round (``orchestration/iter_NNN``), the diff of every file the commit touched
(``candidates/iter_NNN/change.diff``), the report of the Analysis generation the round planned on
(``analysis/<commit>/generation-NNN/report.md``), its handoff (``handoffs/iter_NNN.json``), and its lesson
(``lessons/iter_NNN.md``). An Experience keeps the whole lesson as its reasoning, which ``lesson_reasoning`` reads,
the whole Analysis report, which ``analysis_report`` finds, and excerpts of the rest of the prose: ``plan_text`` for
a plan, which keeps the whole Summary line a plan opens with, and ``first_sentences`` for the others.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from hyperloom_kb.schema import TEXT_MAX_BYTES
from kernelforge.knowledge.kb_store.identity.normalization import UNKNOWN_SEGMENT
from kernelforge.knowledge.kb_store.identity.run import LoopKernel, resolve_loop_identity
from kernelforge.loop.experiments_layout import (
    CANDIDATE_META,
    LANE_PLAN_GLOB,
    ROUND_CONTEXT,
    ExperimentsLayout,
    lane_plan_name,
)
from kernelforge.orchestrator.critic_fallback import original_draft
from kernelforge.orchestrator.summary_line import read_summary
from kernelforge.run_export.files import read_json, read_text
from kernelforge.run_export.units import CandidateOrigin

#: How many sentences of a plan without a Summary line, or of a single session's rationale when it wrote no lesson,
#: an Experience keeps.
_EXCERPT_SENTENCES = 3
_HEADING = re.compile(r"^#{1,6} (\S.*)$")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z`*\"'(\[])")
_LIST_MARKER = re.compile(r"^(?:[-*+]|\d+[.)])\s+")
#: A published Analysis generation's directory; an Analysis still at work writes under ``analysis/work/`` instead.
_GENERATION = re.compile(r"^generation-\d{3,}$")


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

    draft = original_draft(plan)
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
        # The benchmark driver bytes. Two runs of one operator on different drivers must not share a repeat group.
        "driver_sha256": config.get("driver_sha256"),
    }
    return {key: value for key, value in identity.items() if value not in (None, "", UNKNOWN_SEGMENT)}


def _lane_plan_file(layout: ExperimentsLayout, seq: int, plan: str) -> Path | None:
    """The file of the fan-out round that holds this lane's plan: optimization_plan.md for the round's first lane,
    lane_NNN.md for its others.

    A round's later lanes are measured in later iterations, which have no planning directory of their own.
    """

    for number in range(seq, 0, -1):
        directory = layout.round(number)
        for path in [directory / lane_plan_name(1), *sorted(directory.glob(LANE_PLAN_GLOB))]:
            if path.is_file() and read_text(path).strip() == plan:
                return path
    return None


@dataclass(frozen=True)
class IterationArchive:
    """What Kernel Forge archived for iteration ``seq`` of the run ``layout`` lays out."""

    layout: ExperimentsLayout
    seq: int

    @property
    def candidate(self) -> Path:
        return self.layout.candidate(self.seq)

    @property
    def handoff(self) -> Path:
        return self.layout.handoff(self.seq)

    @property
    def lesson(self) -> Path:
        return self.layout.lesson(self.seq)

    def meta(self) -> dict[str, Any] | None:
        """The candidate's ``meta.json``, which Kernel Forge writes for every attempt that produced a diff."""

        return read_json(self.candidate / CANDIDATE_META)

    def plan_file(self, origin: CandidateOrigin, plan: str) -> Path | None:
        """The plan the candidate was built from: its lane's plan, or the plan of the round a single session ran.

        ``plan`` is the plan ``meta.json`` records, which for a lane is the whole of the lane's plan.
        """

        found: Path | None = None
        if origin is CandidateOrigin.LANE and plan:
            found = _lane_plan_file(self.layout, self.seq, plan)
        elif origin is CandidateOrigin.SESSION:
            relative = str((read_json(self.handoff) or {}).get("optimization_plan_path") or "")
            found = self.layout.root.parent / relative if relative else None
        return found if found is not None and found.is_file() else None

    def fields(self, meta: dict[str, Any], plan_file: Path | None) -> dict[str, Any]:
        """The unit fields the archive holds: the plan, the baseline time, the kernel, and the archived diff.

        The baseline time is the pristine one the run's state records, the original kernel's. ``meta.json`` holds the
        time of the kernel the search started from, which is the same unless the run was given ``--baseline-json``,
        so it stands in only for a run state that records none. The diff is every file the commit touched; the kernel
        file is only the anchor, and neither record contains the other.
        """

        fields: dict[str, Any] = {}
        if plan_file is not None:
            fields["plan_file"] = str(plan_file.resolve())
            fields["plan"] = plan_text(read_text(plan_file))
        pristine = (read_json(self.layout.run_state) or {}).get("pristine_baseline_wall_ms")
        fields["source_ms"] = pristine if pristine is not None else meta.get("baseline_wall_ms")
        files = meta.get("files") or {}
        kernel, diff = str(files.get("kernel") or ""), str(files.get("diff") or "")
        if kernel and (self.candidate / kernel).is_file():
            fields["source_code"] = str((self.candidate / kernel).resolve())
        if diff and (self.candidate / diff).is_file():
            fields["change_diff_file"] = str((self.candidate / diff).resolve())
        return fields


def lesson_reasoning(lesson: Path) -> dict[str, Any]:
    """The whole lesson, which an Experience keeps as its reasoning; empty when the iteration wrote none."""

    record = read_text(lesson).strip() if lesson.is_file() else ""
    return {"reasoning": record} if record else {}


@dataclass(frozen=True)
class AnalysisReport:
    """The report.md of a published Analysis generation: the generation, ``analysis/<commit>/generation-NNN`` under
    ``forge_experiments``, and the report's text."""

    generation: str
    text: str

    @property
    def fits(self) -> bool:
        """Whether a text field holds the whole report."""

        return len(self.text.encode()) <= TEXT_MAX_BYTES

    def provenance(self) -> dict[str, str]:
        """The generation the report comes from, and why its text is left out when a text field cannot hold it."""

        record = {"analysis_generation": self.generation}
        if not self.fits:
            size = len(self.text.encode())
            record["analysis_report_omitted"] = (
                f"report.md is {size} bytes, over the {TEXT_MAX_BYTES} a text field holds"
            )
        return record


def analysis_report(layout: ExperimentsLayout, plan_file: Path | None) -> AnalysisReport | None:
    """The report of the published Analysis generation the round that wrote ``plan_file`` planned on.

    The round's ``context.json``, beside its plans, names that generation as its first ``analysis_bundle`` evidence,
    the bundle Kernel Forge itself takes a context's to be (``kernelforge/loop/analysis_evidence.py``). It names it by
    the path it had where the run was made, so the path's last three components, ``analysis/<commit>/generation-NNN``,
    are looked up under the run's own ``forge_experiments``. A merge, which no round planned, has no plan file and so
    no report.
    """

    if plan_file is None:
        return None
    context = read_json(plan_file.parent / ROUND_CONTEXT) or {}
    bundle = next(
        (
            str(ref.get("path") or "")
            for ref in context.get("evidence_refs") or []
            if isinstance(ref, dict) and ref.get("kind") == "analysis_bundle"
        ),
        "",
    )
    tail = PurePosixPath(bundle).parts[-3:]
    if len(tail) < 3 or tail[0] != "analysis" or not _GENERATION.match(tail[2]):
        return None
    report = layout.root.joinpath(*tail, "report.md")
    return AnalysisReport("/".join(tail), read_text(report).strip()) if report.is_file() else None
