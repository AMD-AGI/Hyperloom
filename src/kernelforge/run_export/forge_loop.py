# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A plain ``kernelforge forge-loop`` run: one unit per candidate Kernel Forge archived.

The run is a workspace's ``forge_experiments/`` directory, recognized by its ``campaign_config.json`` and
``run_state.json``. Every attempt that produced a diff is archived as ``candidates/iter_NNN/meta.json``, and each one
is a unit: seq N, the candidate iteration N measured and Kernel Forge's decision on it, under the run's
``campaign_id`` as ``run_id``. The forge-loop's stdout log has no timestamps, so a unit's times come from the run's
``events.jsonl``, and the log is read only for the Implementer it names and the record of Kernel Forge's own
experience KB each kept candidate was written as.

A log belongs to the run whose workspace path shares the most trailing components with the kernel path the log
announces. That is the path the run had where it was made, so a run made in a container is still found, and a path two
runs fit equally stops the export rather than giving both runs the log. A log is read for what it records, and a unit
does not name it: the only files a unit names are its plan, its kernel, and its lesson.

Kernel Forge writes naive local times, and the units carry them as it wrote them; collection reads a naive time as
UTC. That is right for a run made in a container, which normally runs in UTC, and shifts the times of a run made on a
host in another time zone.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from kernelforge.loop.merge_candidates import MERGE_PLAN_PREFIX
from kernelforge.run_export.archive import (
    IterationArchive,
    analysis_report,
    campaign_identity,
    first_sentences,
    implementer_fields,
)
from kernelforge.run_export.files import FileRecord, inventory, read_json, read_text
from kernelforge.run_export.units import CandidateOrigin, Run, read_error

_EXPERIMENTS = "forge_experiments"
_CONFIG = "campaign_config.json"
_STATE = "run_state.json"
#: Kernel Forge's copies of a workspace for a fan-out round's lanes, which carry the run's whole archive.
_LANE_COPY = "forge-lanes-"
_STARTED = "iteration_started"
_RESULT = "iteration_result"
_LOOP_BANNER = "Starting autonomous iteration loop"
_KERNEL_LINE = re.compile(r"^  Kernel: (.+)$", re.M)
_IMPLEMENTER_LINE = re.compile(r"^[ \t]*Implementer: (\S+) / (\S+)[ \t]*$", re.M)
_ITERATION_HEADER = re.compile(r"^--- Iteration (\d+)\b")
_KB_WRITTEN = re.compile(r"^\s*\[kb\] experience written: (\S+)")
_CANDIDATE_DIR = re.compile(r"^iter_(\d+)$")
_LANE_RATIONALE = re.compile(r"^lane \d+ of a fan-out round$")
#: The in-session gate's outcome, which Kernel Forge puts in front of the agent's own text.
_GATE_TAG = re.compile(r"^\[gate edits=[^\]]*\]\s*")
_CLOCK = re.compile(r"^(\d{4}-\d\d-\d\d) (\d\d:\d\d:\d\d)$")


@dataclass(frozen=True)
class _LoopLog:
    """One forge-loop stdout log: the kernels its runs announced, the first Implementer it names, and the KB records
    its iterations wrote."""

    path: Path
    kernels: tuple[str, ...]
    implementer: dict[str, str]
    kb_records: dict[int, str]


class AmbiguousOwnership(Exception):
    """A log names a path that fits more than one run's workspace equally well."""


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
        workspaces = " and ".join(str(owner.parent) for owner in owners)
        raise AmbiguousOwnership(
            f"{evidence} names {named}, which fits the runs in {workspaces} equally well; "
            "export them from separate log roots"
        )
    return owners[0] if owners else None


def _log_owners(log: _LoopLog, kernel_paths: dict[Path, str]) -> set[Path]:
    """The runs whose kernel ``log`` announces; ``kernel_paths`` holds each run's kernel path in its workspace."""

    inners = {experiments: PurePosixPath(path).parts for experiments, path in kernel_paths.items() if path}
    owners: set[Path] = set()
    for kernel in log.kernels:
        named = PurePosixPath(kernel.strip()).parts
        candidates = {
            experiments: named[: -len(inner)] for experiments, inner in inners.items() if named[-len(inner) :] == inner
        }
        owner = _owner(log.path, kernel.strip(), candidates)
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
        if (header := _ITERATION_HEADER.match(line)) is not None:
            iteration = int(header.group(1))
        elif iteration is not None and (written := _KB_WRITTEN.match(line)) is not None:
            records.setdefault(iteration, written.group(1))
    return records


def _loop_logs(records: list[FileRecord]) -> list[_LoopLog]:
    logs: list[_LoopLog] = []
    for record in records:
        if record.path.suffix != ".log" or record.size == 0:
            continue
        text = read_text(record.path)
        if _LOOP_BANNER not in text:
            continue
        implementer = _IMPLEMENTER_LINE.search(text)
        named = {"llm_backend": implementer.group(1), "llm_model": implementer.group(2)} if implementer else {}
        logs.append(_LoopLog(record.path, tuple(_KERNEL_LINE.findall(text)), named, _kb_records(text)))
    return logs


def _experiments_dirs(records: list[FileRecord], root: Path) -> list[Path]:
    found: list[Path] = []
    for record in records:
        path = record.path
        if path.name != _CONFIG or path.parent.name != _EXPERIMENTS:
            continue
        lane_copy = any(part.startswith(_LANE_COPY) for part in path.relative_to(root).parts)
        if not lane_copy and (path.parent / _STATE).is_file():
            found.append(path.parent)
    return found


def _iso(stamp: Any) -> str | None:
    """A Kernel Forge timestamp, ``%Y-%m-%d %H:%M:%S`` in naive local time, as ISO 8601."""

    match = _CLOCK.match(str(stamp or ""))
    return f"{match.group(1)}T{match.group(2)}" if match else None


def _event_times(path: Path) -> dict[tuple[str, int], str]:
    """When each iteration last started and when it reported its result, keyed by event type and iteration."""

    times: dict[tuple[str, int], str] = {}
    lines = read_text(path).splitlines() if path.is_file() else []
    for raw in lines:
        try:
            event = json.loads(raw)
        except ValueError:
            continue
        if not isinstance(event, dict) or event.get("type") not in (_STARTED, _RESULT):
            continue
        stamp, iteration = _iso(event.get("ts")), event.get("iter")
        if stamp and isinstance(iteration, int):
            times[(event["type"], iteration)] = stamp
    return times


def _origin(meta: dict[str, Any]) -> CandidateOrigin:
    if str(meta.get("plan") or "").startswith(MERGE_PLAN_PREFIX):
        return CandidateOrigin.MERGE
    if _LANE_RATIONALE.match(str(meta.get("rationale") or "")):
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
        return {"reasoning": first_sentences(_GATE_TAG.sub("", str(meta.get("rationale") or ""), count=1))}
    return {}


@dataclass(frozen=True)
class _PlainRun:
    """One plain run: its ``forge_experiments`` directory, its event times, and the KB records its logs show each
    iteration writing."""

    run: Run
    experiments: Path
    times: dict[tuple[str, int], str]
    kb_records: dict[int, str]

    def candidate_unit(self, seq: int) -> dict[str, Any]:
        """Seq N: the candidate iteration N archived, and Kernel Forge's decision on it."""

        archive = IterationArchive(self.experiments, seq)
        meta = archive.meta()
        if meta is None:
            message = f"{archive.candidate / 'meta.json'} is not a readable JSON object"
            return read_error(f"{self.run.label}/iteration-{seq}", message)
        origin = _origin(meta)
        plan = str(meta.get("plan") or "").strip()
        plan_file = archive.plan_file(origin, plan)
        report = analysis_report(self.experiments, plan_file)
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
            started_at=self.times.get((_STARTED, seq), stamp),
            completed_at=self.times.get((_RESULT, seq), stamp),
            current_best_mean_ms=meta.get("best_wall_ms_before"),
            decision=meta.get("decision"),
            candidate_mean_ms=meta.get("wall_ms"),
        )
        unit.update(archive.fields(meta, plan_file))
        unit.update(_origin_fields(origin, meta, plan))
        unit.update(implementer_fields(plan_file))
        if report is not None and report.fits:
            unit["analysis_report"] = report.text
        return unit


def _run_units(experiments: Path, logs: list[_LoopLog]) -> list[dict[str, Any]]:
    """The units of one run, given the logs that belong to it."""

    workspace = experiments.parent
    config = read_json(experiments / _CONFIG)
    state = read_json(experiments / _STATE)
    campaign_id = str((state or {}).get("campaign_id") or "")
    if config is None or not campaign_id:
        problem = f"{_CONFIG} is not a readable JSON object" if config is None else f"{_STATE} names no campaign_id"
        return [read_error(f"{workspace.name}/session", problem)]
    times = _event_times(experiments / "events.jsonl")
    run = Run(
        label=workspace.name,
        run_id=campaign_id,
        identity=campaign_identity(config),
        provenance=next((log.implementer for log in logs if log.implementer), {}),
    )
    kb_records = {seq: record for log in logs for seq, record in log.kb_records.items()}
    plain = _PlainRun(run, experiments, times, kb_records)
    candidates = experiments / "candidates"
    numbers = sorted(
        int(match.group(1))
        for path in (candidates.iterdir() if candidates.is_dir() else ())
        if path.is_dir() and (match := _CANDIDATE_DIR.match(path.name))
    )
    return [plain.candidate_unit(seq) for seq in numbers]


def _logs_by_run(logs: list[_LoopLog], runs: list[Path]) -> dict[Path, list[_LoopLog]]:
    """Each run's forge-loop logs: those announcing its kernel."""

    kernel_paths = {
        experiments: str((read_json(experiments / _CONFIG) or {}).get("kernel_path") or "") for experiments in runs
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
