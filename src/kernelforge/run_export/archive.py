# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What Kernel Forge archives for an iteration, and the agents' Claude Code transcripts, read alike by every layout.

An iteration's archive is its candidate (``candidates/iter_NNN``), the plan it was built from, its handoff
(``handoffs/iter_NNN.json``), its lesson (``lessons/iter_NNN.md``), its ``best/iter_NNN`` bundle, and the planning
files of its round (``orchestration/iter_NNN``), among them the reasoning log the implementer kept beside the plan it
executed (``<plan>.reasoning.md``). An Experience keeps excerpts of the prose among them: ``plan_text`` for a plan,
``implementer_fields`` for the implementer's own account of its work, and ``first_sentences`` for the rest. A plan or
a reasoning log that opens with a Summary line is excerpted as that line.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kernelforge.orchestrator.summary_line import read_reasoning_log_section, read_summary, reasoning_log_path
from kernelforge.run_export.files import inventory, read_json, read_text
from kernelforge.run_export.units import CandidateOrigin

#: How many sentences of a plan, of the agent's reasoning, or of a lesson an Experience keeps.
_EXCERPT_SENTENCES = 3
#: The sentence KernelForge opens a plan with when it could not revise the draft and put the critic's review in front
#: of it instead (``_render_critic_revision_fallback`` in ``kernelforge/orchestrator/orchestration.py`` at 1c14e79e1).
_FALLBACK_PREFACE = "The Orchestration revision was unavailable."
_FALLBACK_DRAFT = "\n## Original draft\n"
_HEADING = re.compile(r"^#{1,6} (\S.*)$")
_PLAN_LINE = re.compile(r"^PLAN:\s*(\S.*?)\s*$", re.M)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z`*\"'(\[])")
_LIST_MARKER = re.compile(r"^(?:[-*+]|\d+[.)])\s+")
_SUMMARY_PROMPT = re.compile(r"^Stop working on the kernel\. Iteration \d+ is over")
#: The in-session gate's outcome, which KernelForge puts in front of the agent's own text when it records the
#: session's rationale (``_gate_outcome_tag`` in ``kernelforge/orchestrator/agent.py``).
_GATE_TAG = re.compile(r"^\[gate edits=[^\]]*\]\s*")


def first_sentences(text: str, count: int = _EXCERPT_SENTENCES) -> str | None:
    """The first ``count`` sentences of ``text``'s prose, copied.

    Headings, code blocks, tables, and rules are skipped, and the end of a paragraph, such as one of the agent's
    messages, ends a sentence.
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


def _summary(text: str) -> str | None:
    """The Summary line of ``text``, cut to the sentences an excerpt keeps; None when it has none."""

    summary = read_summary(text)
    return first_sentences(summary) if summary else None


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
    """What an Experience keeps of a whole plan file: the Summary line its draft opens with, or the first sentences of
    the draft's body when it has none."""

    draft = _draft(plan)
    return _summary(draft) or first_sentences(_plan_body(draft))


def reasoning_text(messages: Sequence[str]) -> str | None:
    """What an Experience keeps of the implementer's own account of its work, given as its messages in order."""

    return first_sentences("\n\n".join(messages))


def rationale_text(rationale: str) -> str | None:
    """What an Experience keeps of the rationale KernelForge recorded for a session, without the gate's outcome."""

    return reasoning_text([_GATE_TAG.sub("", rationale, count=1)])


def plan_title(plan: str) -> str | None:
    """The plan's first Markdown heading, at any level."""

    return next((match.group(1).strip() for line in plan.splitlines() if (match := _HEADING.match(line))), None)


def campaign_identity(config: dict[str, Any], **more: str | None) -> dict[str, Any]:
    """The run's conditions as its ``campaign_config.json`` records them, with ``more`` from elsewhere; empty ones
    are left out."""

    framework = config.get("framework")
    identity = {
        "kernel": config.get("operator_name"),
        "framework": framework if framework != "unknown" else None,
        "backend": config.get("kernel_backend"),
        "gpu": config.get("gpu_type"),
        **more,
    }
    return {key: value for key, value in identity.items() if value not in (None, "")}


def _files_in(directory: Path) -> list[Path]:
    return [record.path for record in inventory(directory)] if directory.is_dir() else []


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
    def orchestration(self) -> Path:
        return self.experiments / "orchestration" / self.tag

    @property
    def handoff(self) -> Path:
        return self.experiments / "handoffs" / f"{self.tag}.json"

    @property
    def lesson(self) -> Path:
        return self.experiments / "lessons" / f"{self.tag}.md"

    def meta(self) -> dict[str, Any] | None:
        """The candidate's ``meta.json``, which Kernel Forge writes for every attempt that produced a diff."""

        return read_json(self.candidate / "meta.json")

    def plan_file(self, origin: CandidateOrigin | None, plan: str) -> Path | None:
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

    def fields(
        self, origin: CandidateOrigin | None, meta: dict[str, Any] | None, plan_file: Path | None
    ) -> dict[str, Any]:
        """The unit fields the archive holds: the plan, the baseline time, the PLAN headline, the kernel, the lesson.

        The baseline time is the pristine one the run's state records, which every score divides by. ``meta.json``
        holds the time of the kernel the search started from, which is the same unless the run was given
        ``--baseline-json``, so it stands in only for a run state that records none.
        """

        fields: dict[str, Any] = {}
        if plan_file is not None:
            fields["plan_file"] = str(plan_file.resolve())
            fields["plan"] = plan_text(read_text(plan_file))
        if meta is not None:
            pristine = (read_json(self.experiments / "run_state.json") or {}).get("pristine_baseline_wall_ms")
            fields["source_ms"] = pristine if pristine is not None else meta.get("baseline_wall_ms")
            headline = str(meta.get("plan") or "").strip()
            if origin is CandidateOrigin.SESSION and headline and "\n" not in headline:
                fields["approach"] = headline
            kernel = str((meta.get("files") or {}).get("kernel") or "")
            if kernel and (self.candidate / kernel).is_file():
                fields["source_code"] = str((self.candidate / kernel).resolve())
        if self.lesson.is_file():
            fields["lesson_file"] = str(self.lesson.resolve())
            fields["lesson"] = first_sentences(read_text(self.lesson))
        return fields

    def files(self, plan_file: Path | None) -> list[Path]:
        """The archive's files: the candidate, the round's planning files, the best bundle, the handoff, the lesson."""

        files = _files_in(self.candidate) + _files_in(self.orchestration)
        files += _files_in(self.experiments / "best" / self.tag)
        if plan_file is not None and plan_file.parent != self.orchestration:
            files += _files_in(plan_file.parent)
        return files + [path for path in (self.handoff, self.lesson) if path.is_file()]

    def sessions(self, plan_file: Path | None, transcripts: Iterable[Transcript]) -> list[Transcript]:
        """Of this run's ``transcripts``, those of the sessions told to keep ``plan_file``'s reasoning log.

        A prompt names the log by the absolute path it had where the session ran, which for a run made in a container
        is not the path it has here, so the log is matched by its path from the workspace's own directory on. Another
        run's workspace of the same name would match too, so ``transcripts`` must hold only this run's.
        """

        if plan_file is None:
            return []
        log = f"/{reasoning_log_path(plan_file).relative_to(self.experiments.parent.parent).as_posix()}"
        return [transcript for transcript in transcripts if (transcript.reasoning_log or "").endswith(log)]


@dataclass(frozen=True)
class Transcript:
    """One Claude Code session, read once.

    ``cwds`` are the working directories its records name, in the order they first appear, and ``prompt`` is its first
    user message. ``messages`` are the agent's text blocks until KernelForge resumed the session to ask for its record
    of the iteration.
    """

    path: Path
    cwds: tuple[str, ...]
    started: str
    prompt: str
    messages: tuple[str, ...]

    @property
    def reasoning_log(self) -> str | None:
        """The reasoning log the session was told to keep, by the path it had where the session ran."""

        return read_reasoning_log_section(self.prompt)


class _TranscriptReader:
    """A transcript's records, taken in session order."""

    def __init__(self) -> None:
        self.cwds: list[str] = []
        self.started = ""
        self.prompt = ""
        self.messages: list[str] = []
        self._summarized = False
        self._seen: set[tuple[Any, int, str]] = set()

    def add(self, item: dict[str, Any]) -> None:
        cwd = item.get("cwd")
        if isinstance(cwd, str) and cwd not in self.cwds:
            self.cwds.append(cwd)
        stamp = str(item.get("timestamp") or "")[:23]
        if stamp and not self.started:
            self.started = stamp
        message = raw if isinstance(raw := item.get("message"), dict) else {}
        kind = item.get("type")
        if kind == "queue-operation":
            self._summarized |= _SUMMARY_PROMPT.match(str(item.get("content") or "")) is not None
        elif kind == "user":
            self._user(message.get("content"))
        elif kind == "assistant":
            self._assistant(message)

    def _user(self, content: Any) -> None:
        if isinstance(content, str):
            text = content
        else:
            text = "".join(
                str(block.get("text") or "")
                for block in content or []
                if isinstance(block, dict) and block.get("type") == "text"
            )
        if text and not self.prompt:
            self.prompt = text
        elif text and _SUMMARY_PROMPT.match(text):
            self._summarized = True

    def _assistant(self, message: dict[str, Any]) -> None:
        content = message.get("content")
        if not isinstance(content, list):
            return
        for position, block in enumerate(content):
            if not isinstance(block, dict):
                continue
            key = (message.get("id"), position, json.dumps(block, sort_keys=True))
            if key in self._seen:
                continue
            self._seen.add(key)
            text = str(block.get("text") or "").strip()
            if block.get("type") == "text" and text and not self._summarized:
                self.messages.append(text)


def is_transcript(path: Path) -> bool:
    """Whether ``path`` is a Claude Code transcript: ``projects/<cwd>/<session>.jsonl``."""

    return path.suffix == ".jsonl" and path.parent.parent.name == "projects"


def read_transcript(path: Path) -> Transcript:
    """One Claude Code transcript (``<session>.jsonl``); lines that are not JSON objects are skipped."""

    reader = _TranscriptReader()
    for raw in read_text(path).splitlines():
        try:
            item = json.loads(raw)
        except ValueError:
            continue
        if isinstance(item, dict):
            reader.add(item)
    return Transcript(path, tuple(reader.cwds), reader.started, reader.prompt, tuple(reader.messages))


def _transcript_fields(transcript: Transcript, root: Path) -> dict[str, Any]:
    """What the transcript of the session that made a candidate gives its unit.

    The reasoning and the last PLAN headline come from the agent's messages; the transcript itself is named by its
    path under the log root.
    """

    fields: dict[str, Any] = {}
    if transcript.messages:
        fields["reasoning"] = reasoning_text(transcript.messages)
        headlines = _PLAN_LINE.findall("\n".join(transcript.messages))
        if headlines:
            fields["approach"] = headlines[-1]
    fields["reasoning_file"] = transcript.path.resolve().relative_to(root.resolve()).as_posix()
    return fields


def implementer_fields(transcripts: Sequence[Transcript], plan_file: Path | None, root: Path) -> dict[str, Any]:
    """What the session that made a candidate recorded of its work: its transcripts, then the reasoning log it kept.

    Each transcript gives its unit what ``_transcript_fields`` reads, the last one winning. The reasoning log beside
    ``plan_file`` is the unit's ``reasoning_log``, and its Summary line, the implementer's own account of the problem
    and its approach, is the reasoning whatever else gave one.
    """

    fields: dict[str, Any] = {}
    for transcript in transcripts:
        fields.update(_transcript_fields(transcript, root))
    log = reasoning_log_path(plan_file) if plan_file is not None else None
    if log is not None and log.is_file():
        fields["reasoning_log"] = str(log.resolve())
        summary = _summary(read_text(log))
        if summary:
            fields["reasoning"] = summary
    return fields
