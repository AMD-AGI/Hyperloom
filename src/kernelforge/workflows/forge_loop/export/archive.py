# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What Kernel Forge archives for an iteration.

An iteration's archive is its candidate (``candidates/iter_NNN``), the diff of every file the commit touched
(``candidates/iter_NNN/change.diff``), its handoff (``handoffs/iter_NNN.json``), and its lesson
(``lessons/iter_NNN.md``). An Experience keeps the whole lesson as its reasoning, which ``lesson_reasoning`` reads,
and an excerpt of the session's own rationale, which ``first_sentences`` takes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kernelforge.knowledge.kb_store.identity.normalization import UNKNOWN_SEGMENT
from kernelforge.knowledge.kb_store.identity.run import LoopKernel, resolve_loop_identity
from kernelforge.workflows.forge_loop.campaign.experiments_layout import (
    CANDIDATE_META,
    ExperimentsLayout,
)
from kernelforge.workflows.forge_loop.export.files import read_json, read_text

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

    def fields(self, meta: dict[str, Any]) -> dict[str, Any]:
        """The unit fields the archive holds: the baseline time, the kernel, and the archived diff.

        The baseline time is the pristine one the run's state records, the original kernel's. ``meta.json`` holds the
        time of the kernel the search started from, which is the same unless the run was given ``--baseline-json``,
        so it stands in only for a run state that records none. The diff is every file the commit touched; the kernel
        file is only the anchor, and neither record contains the other.
        """

        fields: dict[str, Any] = {}
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
