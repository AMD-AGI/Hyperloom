# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A plain ``kernelforge forge-loop`` run: one unit per candidate Kernel Forge archived.

The run is a workspace's ``forge_experiments/`` directory, recognized by its ``campaign_config.json`` and
``run_state.json``. A directory that never wrote those two files is not a run this export reports. A record that
exists but is not readable JSON, and a candidate whose ``meta.json`` is missing or malformed, is one entry saying
why. Every attempt that produced a diff is archived as ``candidates/iter_NNN/meta.json``, and each one is a unit:
seq N, the candidate iteration N measured and Kernel Forge's decision on it, under the run's ``campaign_id`` as
``run_id``. The forge-loop's stdout log has no timestamps, so a unit's times come from the run's ``events.jsonl``.
The log is read only for the experiment each session announces and the record of Kernel Forge's own experience KB a
kept candidate was written as. Each ``iteration_result`` event records the ``experiment_id`` of the session that
measured that iteration, and a kept candidate takes the record that session printed under its iteration. A log whose
experiment no exported run recorded, and two logs that disagree on one session's record, leave ``kb_record`` unset.
Nothing about a log stops the export, and no path is compared, so two runs on one kernel file each keep their own
record, as does a run whose ``forge_experiments`` was moved out of the workspace it was made in. A unit does not
name the log: the only files a unit names are its plan, its kernel, and the diff of every file the commit touched.

Kernel Forge writes naive local times, and the units carry them as it wrote them; collection reads a naive time as
UTC. That is right for a run made in a container, which normally runs in UTC, and shifts the times of a run made on a
host in another time zone.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from kernelforge.loop.experiments_layout import (
    ARCHIVE_TIME_FORMAT,
    CAMPAIGN_CONFIG,
    CANDIDATE_META,
    RUN_STATE,
    ExperimentsLayout,
    tagged_iteration,
)
from kernelforge.loop.fanout import LANE_ROOT_PREFIX, is_lane_rationale
from kernelforge.loop.loop_log import EXPERIMENT_LINE, ITERATION_HEADER, KB_WRITTEN, LOOP_BANNER
from kernelforge.loop.merge_candidates import MERGE_PLAN_PREFIX
from kernelforge.loop.run_state import ITERATION_RESULT, ITERATION_STARTED
from kernelforge.orchestrator.gate_tag import strip_gate_outcome_tag
from kernelforge.run_export.archive import (
    IterationArchive,
    analysis_report,
    campaign_identity,
    first_sentences,
    lesson_reasoning,
)
from kernelforge.run_export.files import FileRecord, inventory, read_json, read_text
from kernelforge.run_export.units import CandidateOrigin, Run, read_error
from kernelforge.workspace.edit_policy import FORGE_LOOP_OUTPUT_DIRNAME


def _session_records(text: str) -> dict[tuple[str, int], str]:
    """The first record of Kernel Forge's own experience KB each session printed, by experiment id and iteration.

    A banner opens a session and the first experiment line after it names that session. The forge-loop writes a record
    after each KEEP and may write the best one again after its last iteration, so the first record for a key wins, and
    only a kept candidate's unit takes its iteration's record. A line before the session's experiment line, or before
    its first iteration header, is not attributed.
    """

    records: dict[tuple[str, int], str] = {}
    experiment: str | None = None
    iteration: int | None = None
    in_session = False
    for line in text.splitlines():
        if line.strip() == LOOP_BANNER:
            in_session = True
            experiment = None
            iteration = None
        elif in_session and experiment is None and (announced := EXPERIMENT_LINE.match(line)) is not None:
            experiment = announced.group(1)
        elif (header := ITERATION_HEADER.match(line)) is not None:
            iteration = int(header.group(1))
        elif experiment is not None and iteration is not None and (written := KB_WRITTEN.match(line)) is not None:
            records.setdefault((experiment, iteration), written.group(1))
    return records


def _kb_records(records: list[FileRecord]) -> dict[tuple[str, int], str]:
    """Every banner-bearing log's records. A key whose logs disagree is left out, so neither record is guessed."""

    found: dict[tuple[str, int], str | None] = {}
    for record in records:
        if record.path.suffix != ".log" or record.size == 0:
            continue
        text = read_text(record.path)
        if LOOP_BANNER not in text:
            continue
        for key, value in _session_records(text).items():
            if key not in found:
                found[key] = value
            elif found[key] != value:
                found[key] = None
    return {key: value for key, value in found.items() if value is not None}


def _experiments_dirs(records: list[FileRecord], root: Path) -> list[Path]:
    found: list[Path] = []
    for record in records:
        path = record.path
        if path.name != CAMPAIGN_CONFIG or path.parent.name != FORGE_LOOP_OUTPUT_DIRNAME:
            continue
        lane_copy = any(part.startswith(LANE_ROOT_PREFIX) for part in path.relative_to(root).parts)
        if not lane_copy and ExperimentsLayout(path.parent).run_state.is_file():
            found.append(path.parent)
    return found


def _iso(stamp: Any) -> str | None:
    """A Kernel Forge timestamp, ``ARCHIVE_TIME_FORMAT`` in naive local time, as ISO 8601."""

    try:
        return datetime.strptime(str(stamp or ""), ARCHIVE_TIME_FORMAT).isoformat()  # noqa: DTZ007 - the archive stamps naive local time
    except ValueError:
        return None


@dataclass(frozen=True)
class _RunEvents:
    """When each iteration last started and reported its result, and which session measured it."""

    times: dict[tuple[str, int], str]
    experiments: dict[int, str]


def _run_events(path: Path) -> _RunEvents:
    """The times and the measuring session ``events.jsonl`` records for each iteration."""

    times: dict[tuple[str, int], str] = {}
    experiments: dict[int, str] = {}
    lines = read_text(path).splitlines() if path.is_file() else []
    for raw in lines:
        try:
            event = json.loads(raw)
        except ValueError:
            continue
        if not isinstance(event, dict) or event.get("type") not in (ITERATION_STARTED, ITERATION_RESULT):
            continue
        iteration = event.get("iter")
        if not isinstance(iteration, int):
            continue
        if (stamp := _iso(event.get("ts"))) is not None:
            times[(event["type"], iteration)] = stamp
        experiment = event.get("experiment_id")
        if event.get("type") == ITERATION_RESULT and isinstance(experiment, str) and experiment:
            experiments[iteration] = experiment
    return _RunEvents(times, experiments)


def _origin(meta: dict[str, Any]) -> CandidateOrigin:
    if str(meta.get("plan") or "").startswith(MERGE_PLAN_PREFIX):
        return CandidateOrigin.MERGE
    if is_lane_rationale(str(meta.get("rationale") or "")):
        return CandidateOrigin.LANE
    return CandidateOrigin.SESSION


def _origin_fields(origin: CandidateOrigin, meta: dict[str, Any], plan: str) -> dict[str, Any]:
    """What ``meta.json`` says of the candidate that depends on where it came from.

    A merge's plan is Kernel Forge's line naming the two stacked candidates, and no agent reasoned about it. A single
    session's ``rationale`` is the agent's own text. A lane's records neither, only which lane it was.
    """

    if origin is CandidateOrigin.MERGE:
        return {"plan": first_sentences(plan)}
    if origin is CandidateOrigin.SESSION:
        return {"reasoning": first_sentences(strip_gate_outcome_tag(str(meta.get("rationale") or "")))}
    return {}


@dataclass(frozen=True)
class _PlainRun:
    """One plain run: the layout of its ``forge_experiments`` directory, its event times, and the KB records its logs
    show each iteration writing."""

    run: Run
    layout: ExperimentsLayout
    times: dict[tuple[str, int], str]
    kb_records: dict[int, str]

    def candidate_unit(self, seq: int) -> dict[str, Any]:
        """Seq N: the candidate iteration N archived, and Kernel Forge's decision on it."""

        archive = IterationArchive(self.layout, seq)
        meta = archive.meta()
        if meta is None:
            message = f"{archive.candidate / CANDIDATE_META} is not a readable JSON object"
            return read_error(f"{self.run.label}/iteration-{seq}", message)
        origin = _origin(meta)
        plan = str(meta.get("plan") or "").strip()
        plan_file = archive.plan_file(origin, plan)
        report = analysis_report(self.layout, plan_file)
        stamp = _iso(meta.get("timestamp"))
        kb_record = self.kb_records.get(seq) if meta.get("kept") else None
        unit = self.run.unit(
            f"iteration-{seq}",
            seq,
            kb_record=kb_record,
            candidate_origin=origin.value,
            meta=meta,
            **(report.provenance() if report is not None else {}),
        )
        unit.update(
            started_at=self.times.get((ITERATION_STARTED, seq), stamp),
            completed_at=self.times.get((ITERATION_RESULT, seq), stamp),
            best_wall_ms_before=meta.get("best_wall_ms_before"),
            decision=meta.get("decision"),
            candidate_wall_ms=meta.get("wall_ms"),
        )
        unit.update(archive.fields(meta, plan_file))
        unit.update(_origin_fields(origin, meta, plan))
        # The lesson is the reasoning. A session that wrote none keeps the rationale excerpt _origin_fields set.
        unit.update(lesson_reasoning(archive.lesson))
        if report is not None and report.fits:
            unit["analysis_report"] = report.text
        return unit


def _run_units(experiments: Path, kb_records: dict[tuple[str, int], str]) -> list[dict[str, Any]]:
    """The units of one run. A kept candidate takes the record its own session printed, when the logs agree on it."""

    layout = ExperimentsLayout(experiments)
    workspace = experiments.parent
    config = read_json(layout.campaign_config)
    state = read_json(layout.run_state)
    campaign_id = str((state or {}).get("campaign_id") or "")
    if config is None or not campaign_id:
        problem = (
            f"{CAMPAIGN_CONFIG} is not a readable JSON object"
            if config is None
            else f"{RUN_STATE} names no campaign_id"
        )
        return [read_error(f"{workspace.name}/session", problem)]
    events = _run_events(layout.events)
    run = Run(
        label=workspace.name,
        run_id=campaign_id,
        identity=campaign_identity(config),
    )
    records = {
        seq: kb_records[(experiment, seq)]
        for seq, experiment in events.experiments.items()
        if (experiment, seq) in kb_records
    }
    plain = _PlainRun(run, layout, events.times, records)
    candidates = layout.candidates
    numbers = sorted(
        number
        for path in (candidates.iterdir() if candidates.is_dir() else ())
        if path.is_dir() and (number := tagged_iteration(path.name)) is not None
    )
    return [plain.candidate_unit(seq) for seq in numbers]


def forge_loop_runs(root: Path) -> list[dict[str, Any]]:
    """The units of every plain forge-loop run under ``root``, runs in path order."""

    root = root.resolve()
    records = inventory(root)
    runs = _experiments_dirs(records, root)
    if not runs:
        return []
    kb_records = _kb_records(records)
    return [unit for experiments in runs for unit in _run_units(experiments, kb_records)]
