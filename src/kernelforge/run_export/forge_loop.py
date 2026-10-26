# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A plain ``kernelforge forge-loop`` run: one unit per candidate Kernel Forge archived.

The run is a workspace's ``forge_experiments/`` directory, recognized by its ``campaign_config.json`` and
``run_state.json``. A directory that never wrote those two files is not a run this export reports. A record that
exists but is not readable JSON, and a candidate whose ``meta.json`` is missing or malformed, is one entry saying
why. Every attempt that produced a diff is archived as ``candidates/iter_NNN/meta.json``, and each one is a unit:
seq N, the candidate iteration N measured and Kernel Forge's decision on it, under the run's ``campaign_id`` as
``run_id``. The forge-loop's stdout log has no timestamps, so a unit's times come from the run's ``events.jsonl``,
and the log is read only for the kernel it announces and the record of Kernel Forge's own experience KB each kept
candidate was written as.

A log belongs to the run whose ``campaign_config.json`` ``kernel_path`` the kernel line names, when only that run
records it: the same path, or an absolute line that ends with the relative path the config stores. That still finds
a run whose ``forge_experiments`` was moved out of the workspace it was made in. When several runs record that
kernel, the log belongs to the run whose workspace path shares the most trailing components with the path the line
announces, so a run made in a container is still found when only the leading directories differ, and a path two runs
fit equally stops the export rather than giving both runs the log. A log is read for what it records, and a unit
does not name it: the only files a unit names are its plan, its kernel, and the diff of every file the commit touched.

Kernel Forge writes naive local times, and the units carry them as it wrote them; collection reads a naive time as
UTC. That is right for a run made in a container, which normally runs in UTC, and shifts the times of a run made on a
host in another time zone.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
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
from kernelforge.loop.loop_log import ITERATION_HEADER, KB_WRITTEN, KERNEL_LINE, LOOP_BANNER
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


@dataclass(frozen=True)
class _LoopLog:
    """One forge-loop stdout log: the kernels its runs announced, and the KB records its iterations wrote."""

    path: Path
    kernels: tuple[str, ...]
    kb_records: dict[int, str]


class AmbiguousOwnership(Exception):
    """A log names a path that fits more than one run's workspace equally well."""


def _refuse_ambiguous(evidence: Path, named: str, owners: list[Path]) -> None:
    """Stop the export: ``named`` fits every run in ``owners`` and no trail separates them."""

    workspaces = " and ".join(str(owner.parent) for owner in owners)
    raise AmbiguousOwnership(
        f"{evidence} names {named}, which fits the runs in {workspaces} equally well; "
        "export them from separate log roots"
    )


def _shared_tail(named: tuple[str, ...], workspace: Path) -> int:
    """How many of ``workspace``'s trailing path components ``named`` ends with."""

    shared = 0
    for mine, theirs in zip(reversed(named), reversed(workspace.parts)):
        if mine != theirs:
            break
        shared += 1
    return shared


def _owner(evidence: Path, named: str, candidates: dict[Path, tuple[str, ...]]) -> Path | None:
    """The ``forge_experiments`` directory of the run that ``evidence``, which names the path ``named``, belongs to.

    ``candidates`` maps each run it can belong to onto the workspace ``named`` would put that run in, and the run whose
    workspace shares the most trailing components with it owns it.
    """

    scores = {experiments: _shared_tail(workspace, experiments.parent) for experiments, workspace in candidates.items()}
    best = max(scores.values(), default=0)
    owners = sorted(experiments for experiments, score in scores.items() if score == best) if best else []
    if len(owners) > 1:
        _refuse_ambiguous(evidence, named, owners)
    return owners[0] if owners else None


def _names_kernel(announced: PurePosixPath, recorded: str) -> bool:
    """Whether ``announced`` is the file ``recorded`` stores, as the same path or as its absolute form."""

    kernel = PurePosixPath(recorded)
    if announced == kernel:
        return True
    parts = kernel.parts
    return len(announced.parts) > len(parts) and announced.parts[-len(parts) :] == parts


def _owner_of_kernel(evidence: Path, named: str, kernel_paths: dict[Path, str]) -> Path | None:
    """The run whose ``kernel_path`` ``named`` is.

    One run that records that path owns the log even when its archive sits outside the workspace. Several runs that
    record it are still separated by the workspace path the line announces.
    """

    announced = PurePosixPath(named)
    exact = sorted(
        experiments for experiments, path in kernel_paths.items() if path and PurePosixPath(path) == announced
    )
    if len(exact) > 1:
        _refuse_ambiguous(evidence, named, exact)
    if len(exact) == 1:
        return exact[0]
    matches = sorted(
        experiments for experiments, path in kernel_paths.items() if path and _names_kernel(announced, path)
    )
    if len(matches) == 1:
        return matches[0]
    if len(matches) < 2:
        return None
    candidates = {
        experiments: announced.parts[: -len(PurePosixPath(kernel_paths[experiments]).parts)]
        for experiments in matches
    }
    owner = _owner(evidence, named, candidates)
    if owner is None:
        _refuse_ambiguous(evidence, named, matches)
    return owner


def _log_owners(log: _LoopLog, kernel_paths: dict[Path, str]) -> set[Path]:
    """The runs whose kernel ``log`` announces; ``kernel_paths`` holds each run's kernel path in its workspace."""

    owners: set[Path] = set()
    for kernel in log.kernels:
        owner = _owner_of_kernel(log.path, kernel.strip(), kernel_paths)
        if owner is not None:
            owners.add(owner)
    return owners


def _kb_records(text: str) -> dict[int, str]:
    """The first record of Kernel Forge's own experience KB written during each iteration, by iteration.

    The forge-loop writes one after each KEEP and may write the best one again after its last iteration, which is why
    only a kept candidate's unit takes its iteration's record.
    """

    records: dict[int, str] = {}
    iteration: int | None = None
    for line in text.splitlines():
        if (header := ITERATION_HEADER.match(line)) is not None:
            iteration = int(header.group(1))
        elif iteration is not None and (written := KB_WRITTEN.match(line)) is not None:
            records.setdefault(iteration, written.group(1))
    return records


def _loop_logs(records: list[FileRecord]) -> list[_LoopLog]:
    logs: list[_LoopLog] = []
    for record in records:
        if record.path.suffix != ".log" or record.size == 0:
            continue
        text = read_text(record.path)
        if LOOP_BANNER not in text:
            continue
        logs.append(_LoopLog(record.path, tuple(KERNEL_LINE.findall(text)), _kb_records(text)))
    return logs


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


def _event_times(path: Path) -> dict[tuple[str, int], str]:
    """When each iteration last started and when it reported its result, keyed by event type and iteration."""

    times: dict[tuple[str, int], str] = {}
    lines = read_text(path).splitlines() if path.is_file() else []
    for raw in lines:
        try:
            event = json.loads(raw)
        except ValueError:
            continue
        if not isinstance(event, dict) or event.get("type") not in (ITERATION_STARTED, ITERATION_RESULT):
            continue
        stamp, iteration = _iso(event.get("ts")), event.get("iter")
        if stamp and isinstance(iteration, int):
            times[(event["type"], iteration)] = stamp
    return times


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


def _run_units(experiments: Path, logs: list[_LoopLog]) -> list[dict[str, Any]]:
    """The units of one run, given the logs that belong to it."""

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
    times = _event_times(layout.events)
    run = Run(
        label=workspace.name,
        run_id=campaign_id,
        identity=campaign_identity(config),
    )
    kb_records = {seq: record for log in logs for seq, record in log.kb_records.items()}
    plain = _PlainRun(run, layout, times, kb_records)
    candidates = layout.candidates
    numbers = sorted(
        number
        for path in (candidates.iterdir() if candidates.is_dir() else ())
        if path.is_dir() and (number := tagged_iteration(path.name)) is not None
    )
    return [plain.candidate_unit(seq) for seq in numbers]


def _logs_by_run(logs: list[_LoopLog], runs: list[Path]) -> dict[Path, list[_LoopLog]]:
    """Each run's forge-loop logs: those announcing its kernel."""

    kernel_paths = {
        experiments: str((read_json(ExperimentsLayout(experiments).campaign_config) or {}).get("kernel_path") or "")
        for experiments in runs
    }
    owned: dict[Path, list[_LoopLog]] = {experiments: [] for experiments in runs}
    for log in logs:
        for owner in _log_owners(log, kernel_paths):
            owned[owner].append(log)
    return owned


def forge_loop_runs(root: Path) -> list[dict[str, Any]]:
    """The units of every plain forge-loop run under ``root``, runs in path order."""

    root = root.resolve()
    records = inventory(root)
    runs = _experiments_dirs(records, root)
    if not runs:
        return []
    logs = _logs_by_run(_loop_logs(records), runs)
    return [unit for experiments in runs for unit in _run_units(experiments, logs[experiments])]
