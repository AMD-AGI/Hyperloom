# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A plain ``kernelforge forge-loop`` run: one unit per candidate Kernel Forge archived, and per iteration that archived
none.

The run is a workspace's ``forge_experiments/`` directory, recognized by its ``campaign_config.json`` and
``run_state.json``. A directory that never wrote those two files is not a run this export reports. A record that
exists but is not readable JSON, and a candidate whose ``meta.json`` is missing or malformed, is one entry saying
why. Every attempt that produced a diff is archived as ``candidates/iter_NNN/meta.json``, and each one is a unit:
seq N, the candidate iteration N measured and Kernel Forge's decision on it, under the run's ``campaign_id`` as
``run_id``. An iteration that archived nothing, such as one whose session changed nothing, is a unit from its
``iteration_result`` event, which records its decision and what its Implementer declared; the mapping skips one no
agent gave an account of. The forge-loop's stdout log has no timestamps, so a unit's times come from the run's ``events.jsonl``.
Each ``iteration_result`` event records the ``experiment_id`` of the session that measured that iteration, and a kept
candidate takes the record of Kernel Forge's own experience KB that session wrote under its iteration, which a
``kb_record_written`` event records. A run made before that event existed has it only in its stdout log, which is
then read for the experiment each session announces and the record it printed. A log whose experiment no exported
run recorded, and two logs that disagree on one session's record, leave ``kb_record`` unset. Nothing about a log
stops the export, and no path is compared, so two runs on one kernel file each keep their own record, as does a run
whose ``forge_experiments`` was moved out of the workspace it was made in. A unit does not name the log: the only
files a unit names are its plan, its kernel, and the diff of every file the commit touched.

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

from kernelforge.workflows.forge_loop.campaign.experiments_layout import (
    ARCHIVE_TIME_FORMAT,
    CAMPAIGN_CONFIG,
    CANDIDATE_META,
    RUN_STATE,
    ExperimentsLayout,
    tagged_iteration,
)
from kernelforge.workflows.forge_loop.iteration.fanout import LANE_ROOT_PREFIX, is_lane_rationale
from kernelforge.workflows.forge_loop.campaign.loop_log import (
    EXPERIMENT_LINE,
    ITERATION_HEADER,
    KB_WRITTEN,
    LOOP_BANNER,
)
from kernelforge.workflows.forge_loop.iteration.merge_candidates import MERGE_PLAN_PREFIX
from kernelforge.knowledge.kb_store.identity.run import campaign_identity
from kernelforge.workflows.forge_loop.campaign.run_state import ITERATION_RESULT, ITERATION_STARTED, KB_RECORD_WRITTEN
from kernelforge.workflows.forge_loop.export.archive import (
    IterationArchive,
    analysis_report,
    archived_lesson,
    declared_account,
    declared_techniques,
    first_sentences,
)
from kernelforge.workflows.forge_loop.export.files import FileRecord, inventory, read_json, read_text
from kernelforge.workflows.forge_loop.export.units import CandidateOrigin, Run, read_error
from kernelforge.config.edit_policy import FORGE_LOOP_OUTPUT_DIRNAME


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
    """When each iteration last started and reported its result, which session measured it, and the experience KB
    record each session wrote, by experiment id and iteration, the first for a key winning as in a log."""

    times: dict[tuple[str, int], str]
    experiments: dict[int, str]
    kb_records: dict[tuple[str, int], str]
    # Each iteration's last ``iteration_result`` event.
    results: dict[int, dict[str, Any]]


def _run_events(path: Path) -> _RunEvents:
    """The times, the measuring session, and the experience KB records ``events.jsonl`` records for each iteration."""

    found = _RunEvents({}, {}, {}, {})
    lines = read_text(path).splitlines() if path.is_file() else []
    for raw in lines:
        try:
            event = json.loads(raw)
        except ValueError:
            continue
        if not isinstance(event, dict) or event.get("type") not in (
            ITERATION_STARTED,
            ITERATION_RESULT,
            KB_RECORD_WRITTEN,
        ):
            continue
        iteration, experiment = event.get("iter"), event.get("experiment_id")
        if not isinstance(iteration, int):
            continue
        if event["type"] == KB_RECORD_WRITTEN:
            if isinstance(experiment, str) and experiment and isinstance(event.get("record"), str):
                found.kb_records.setdefault((experiment, iteration), event["record"])
            continue
        if (stamp := _iso(event.get("ts"))) is not None:
            found.times[(event["type"], iteration)] = stamp
        if event["type"] == ITERATION_RESULT:
            found.results[iteration] = event
            if isinstance(experiment, str) and experiment:
                found.experiments[iteration] = experiment
    return found


def _origin(meta: dict[str, Any]) -> CandidateOrigin:
    if str(meta.get("plan") or "").startswith(MERGE_PLAN_PREFIX):
        return CandidateOrigin.MERGE
    if is_lane_rationale(str(meta.get("rationale") or "")):
        return CandidateOrigin.LANE
    return CandidateOrigin.SESSION


def _origin_fields(origin: CandidateOrigin, plan: str) -> dict[str, Any]:
    """What ``meta.json`` says of the candidate that depends on where it came from: a merge's plan is Kernel Forge's
    line naming the two stacked candidates, and no agent reasoned about it."""

    return {"plan": first_sentences(plan)} if origin is CandidateOrigin.MERGE else {}


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
        return self._unit(archive, meta, _origin(meta))

    def unarchived_unit(self, seq: int, event: dict[str, Any]) -> dict[str, Any]:
        """Seq N for an iteration N that archived no candidate, such as one whose session changed nothing: Kernel
        Forge's decision and what its Implementer declared, from the iteration's ``iteration_result`` event."""

        return self._unit(IterationArchive(self.layout, seq), _event_meta(event), CandidateOrigin.SESSION)

    def _unit(self, archive: IterationArchive, meta: dict[str, Any], origin: CandidateOrigin) -> dict[str, Any]:
        seq = archive.seq
        plan = str(meta.get("plan") or "").strip()
        plan_file = archive.plan_file(origin, plan)
        report = analysis_report(self.layout, plan_file)
        stamp = _iso(meta.get("timestamp"))
        kb_record = self.kb_records.get(seq) if meta.get("kept") else None
        lesson = archived_lesson(archive.lesson)
        unit = self.run.unit(
            f"iteration-{seq}",
            seq,
            kb_record=kb_record,
            candidate_origin=origin.value,
            meta=meta,
            **(report.provenance() if report is not None else {}),
            **lesson.provenance(),
        )
        unit.update(
            started_at=self.times.get((ITERATION_STARTED, seq), stamp),
            completed_at=self.times.get((ITERATION_RESULT, seq), stamp),
            best_wall_ms_before=meta.get("best_wall_ms_before"),
            best_mean_case_speedup_before=meta.get("best_mean_case_speedup_before"),
            decision=meta.get("decision"),
            candidate_wall_ms=meta.get("wall_ms"),
            mean_case_speedup=meta.get("mean_case_speedup"),
        )
        unit.update(archive.fields(meta, plan_file))
        unit.update(declared_techniques(meta))
        unit.update(declared_account(origin, meta))
        unit.update(_origin_fields(origin, plan))
        unit.update(lesson.fields())
        if report is not None and report.fits:
            unit["analysis_report"] = report.text
        return unit


#: The scalar fields of an ``iteration_result`` event an iteration that archived no candidate takes as its metadata.
_EVENT_META_KEYS = (
    "decision",
    "validation_passed",
    "wall_ms",
    "mean_case_speedup",
    "snr_db",
    "session_end_reason",
    "turns",
    "agent_profile_id",
    "agent_provider",
    "agent_model",
    "blockage_epoch",
    "plan",
    "rationale",
)


def _event_meta(event: dict[str, Any]) -> dict[str, Any]:
    """An ``iteration_result`` event as the ``meta.json`` of an iteration that archived no candidate.

    It kept nothing, so the best it was measured against is the best the event says stood after it. Its ``plan`` is
    the Implementer's headline. An event that records ``reasoning`` holds what the Implementer declared, empty when it
    declared none; one written before it did gives only the rationale, which ``declared_account`` reads as an archived
    candidate's.
    """

    meta = {key: event[key] for key in _EVENT_META_KEYS if key in event}
    meta.update(
        kept=False,
        timestamp=event.get("ts"),
        best_wall_ms_before=event.get("best_after_ms"),
        best_mean_case_speedup_before=event.get("best_after_mean_case_speedup"),
    )
    if "reasoning" in event:
        meta.update(headline=str(event.get("plan") or ""), reasoning=str(event.get("reasoning") or ""))
    techniques = event.get("techniques")
    if isinstance(techniques, dict):
        meta.update(techniques=techniques.get("listed") or [], techniques_unlisted=techniques.get("unlisted") or [])
    return meta


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
    written = {**kb_records, **events.kb_records}
    records = {
        seq: written[(experiment, seq)]
        for seq, experiment in events.experiments.items()
        if (experiment, seq) in written
    }
    plain = _PlainRun(run, layout, events.times, records)
    candidates = layout.candidates
    archived = {
        number
        for path in (candidates.iterdir() if candidates.is_dir() else ())
        if path.is_dir() and (number := tagged_iteration(path.name)) is not None
    }
    units = {seq: plain.candidate_unit(seq) for seq in archived}
    units.update(
        {seq: plain.unarchived_unit(seq, event) for seq, event in events.results.items() if seq not in archived}
    )
    return [units[seq] for seq in sorted(units)]


def forge_loop_runs(root: Path) -> list[dict[str, Any]]:
    """The units of every plain forge-loop run under ``root``, runs in path order."""

    root = root.resolve()
    records = inventory(root)
    runs = _experiments_dirs(records, root)
    if not runs:
        return []
    kb_records = _kb_records(records)
    return [unit for experiments in runs for unit in _run_units(experiments, kb_records)]
