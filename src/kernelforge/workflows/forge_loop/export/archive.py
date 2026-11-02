# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What Kernel Forge archives for an iteration.

An iteration's archive is its candidate (``candidates/iter_NNN``), the plan it was built from, among the planning
files of its round (``orchestration/iter_NNN``), the diff of every file the commit touched
(``candidates/iter_NNN/change.diff``), the report of the Analysis generation the round planned on
(``analysis/<commit>/generation-NNN/report.md``), its handoff (``handoffs/iter_NNN.json``), and its lesson
(``lessons/iter_NNN.md``). An Experience keeps the reasoning and the headline the Implementer declared before the
candidate was measured, which ``declared_account`` reads, the whole lesson written after it as its reflection, which
``archived_lesson`` reads, the whole Analysis report, which ``analysis_report`` finds, and excerpts of the rest of
the prose: ``plan_text`` for a plan, which keeps the whole Summary line a plan opens with, and ``first_sentences``
for the others.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from hyperloom_kb.schema import TEXT_MAX_BYTES
from kernelforge.workflows.forge_loop.campaign.experiments_layout import (
    CANDIDATE_META,
    LANE_PLAN_GLOB,
    ROUND_CONTEXT,
    ExperimentsLayout,
    lane_plan_name,
)
from kernelforge.workflows.forge_loop.roles.critic_fallback import original_draft
from kernelforge.workflows.forge_loop.gate.gate_tag import strip_gate_outcome_tag
from kernelforge.workflows.forge_loop.roles.summary_line import read_summary
from kernelforge.workflows.forge_loop.export.files import read_json, read_text
from kernelforge.workflows.forge_loop.export.units import CandidateOrigin

#: How many sentences of a plan without a Summary line an Experience keeps.
_EXCERPT_SENTENCES = 3
#: The rationales Kernel Forge itself recorded for an iteration before it kept the declared reasoning: a session that
#: gave no text, one that failed after editing, an iteration no agent ran, a round that failed to plan, and one that
#: found the device held.
_KERNELFORGE_RATIONALES = (
    "no rationale provided",
    "agent session ended with error",
    "no-agent (baseline measurement)",
    "planning failed",
    "device held",
)
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


def _names(value: Any) -> list[str]:
    return [str(name).strip() for name in value if str(name).strip()] if isinstance(value, list) else []


def declared_techniques(meta: dict[str, Any]) -> dict[str, Any]:
    """The techniques the candidate's Implementer declared its change applies, as its ``meta.json`` records them;
    empty when no declaration was made, which ``meta.json`` shows by recording neither list."""

    if "techniques" not in meta and "techniques_unlisted" not in meta:
        return {}
    return {
        "techniques": _names(meta.get("techniques")),
        "techniques_unlisted": _names(meta.get("techniques_unlisted")),
        "techniques_declared": True,
    }


def declared_account(origin: CandidateOrigin, meta: dict[str, Any]) -> dict[str, Any]:
    """The PLAN headline and the RATIONALE line the candidate's Implementer declared before it was measured, as
    ``summary`` and ``reasoning``; each is left out when it declared none.

    A candidate archived before ``meta.json`` kept them records neither key. A single session's headline is then its
    plan, and its reasoning is the rationale it recorded without the gate tag in front, unless Kernel Forge wrote that
    rationale itself. A lane's rationale then names only its lane, and a merge was made by no Implementer.
    """

    if "headline" in meta or "reasoning" in meta:
        headline, reasoning = str(meta.get("headline") or ""), str(meta.get("reasoning") or "")
    elif origin is CandidateOrigin.SESSION:
        headline = str(meta.get("plan") or "")
        reasoning = strip_gate_outcome_tag(str(meta.get("rationale") or "")).strip()
        if reasoning.startswith(_KERNELFORGE_RATIONALES):
            reasoning = ""
    else:
        headline = reasoning = ""
    account = {"summary": headline.strip(), "reasoning": reasoning.strip()}
    return {key: value for key, value in account.items() if value}


@dataclass(frozen=True)
class ArchivedLesson:
    """The lesson an iteration wrote after its candidate was measured, ``lessons/iter_NNN.md``."""

    name: str
    text: str

    @property
    def fits(self) -> bool:
        """Whether a text field holds the whole lesson."""

        return len(self.text.encode()) <= TEXT_MAX_BYTES

    def fields(self) -> dict[str, Any]:
        """The whole lesson as the Experience's reflection, when a text field holds it."""

        return {"lesson": self.text} if self.text and self.fits else {}

    def provenance(self) -> dict[str, str]:
        """Why the lesson is left out, when a text field cannot hold it."""

        if not self.text or self.fits:
            return {}
        size = len(self.text.encode())
        return {"lesson_omitted": f"{self.name} is {size} bytes, over the {TEXT_MAX_BYTES} a text field holds"}


def archived_lesson(lesson: Path) -> ArchivedLesson:
    """The lesson the iteration wrote; empty when it wrote none."""

    return ArchivedLesson(lesson.name, read_text(lesson).strip() if lesson.is_file() else "")


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
    the bundle Kernel Forge itself takes a context's to be (``kernelforge/workflows/forge_loop/roles/analysis_evidence.py``). It names it by
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
