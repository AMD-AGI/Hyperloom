# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The two layouts that hold one run of KernelForge on one task and one GPU, launched by AgentKernelArena's ``forge``
template.

* A session directory holding ``driver.log`` and ``arena_logs/`` (KernelForge's forge-rewrite-by-flydsl pipeline),
  with the task's directory. The task's conditions come from the task's own files.
* A campaign log (the launcher's log of one run, under ``logs/``) next to the run's KernelForge artifacts directory
  (``<workspace>/<run>/<task>-forge-<id>/engine/forge_experiments``) and the Claude Code transcripts of its agents
  (``.claude/projects/<cwd>/<session>.jsonl``). An iteration's fields also come from KernelForge's archive of it,
  from the implementer's messages in the transcript of the session that produced its candidate, which is linked by
  its path under the log root, and from the reasoning log that session kept beside its plan. The transcripts of the
  round's planners are raw logs of the run.

Every unit of a run carries the launcher's run name as its ``run_id``. The launcher's own evaluation of the kernel a
session delivered is not a KernelForge run, so it is no unit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from kernelforge.run_export.archive import (
    IterationArchive,
    Transcript,
    campaign_identity,
    implementer_fields,
    is_transcript,
    plan_title,
    read_transcript,
)
from kernelforge.run_export.files import FileRecord, is_under, read_json, read_text, read_yaml
from kernelforge.run_export.session_log import LogLine, Setup, iteration_unit, iterations, port_unit, timeline
from kernelforge.run_export.units import CandidateOrigin, Run, log_files, read_error

#: How much of a log is read to tell whether it is a campaign log.
_HEAD_BYTES = 65_536
_DRIVER_LOG = "driver.log"
_ARENA_START = "AgentKernelArena Framework Started"
_RUNTIME_ENTRYPOINT = re.compile(r"runtime entrypoint (\S+?)\.?(?:\s|$)")
_README_ENTRY = re.compile(r"baseline entry is `([^`]+)`")
_FAMILY_POLICY = re.compile(r"^\s*([a-z0-9][a-z0-9.-]*) family policy")
_SESSION_TXT = re.compile(r"^(\w[\w ]*?)\s*: (.*)$")
_CHECKOUT = re.compile(r"^\S+ \(([0-9a-f]{7,40})(?: (\S+))?\)$")
_MODEL_LINE = re.compile(r"^(\S+)\s+\(effort=([^,)]+)")
_KB_LINE = re.compile(r"^mode=(\S+)")
_FORGE_ARTIFACTS = re.compile(r"^Forge artifacts: (\S+)$")
_RUN_STAMP = re.compile(r"^run_\d{8}_\d{6}_")
_ENGINE_BANNER = re.compile(r"^forge_engine=(\S+) ([0-9a-f]{7,40})\s*$", re.M)
_IMAGE_BANNER = re.compile(r"^Docker runtime: .*?\bimage=(\S+)", re.M)
_CONFIG_BANNER = re.compile(r"^config_ok=(\S+)\s*$", re.M)
_IMPLEMENTER = re.compile(r"^Implementer: (\S+) / (\S+)$")
_MODEL_CLASS = re.compile(r"'--model-class', '([^']+)'")
#: The line KernelForge ends every implementer's prompt with (``kernelforge/orchestrator/agent.py``), and no planner's.
_IMPLEMENTER_PROMPT_END = "Make your change(s) now."
_ROUND_DIR = re.compile(r"^iter_(\d+)$")
_PIPELINE = "forge-rewrite-by-flydsl"


@dataclass(frozen=True)
class ArenaRuns:
    """The units one arena run yielded, and the files under the log root that hold that run."""

    units: list[dict[str, Any]]
    claimed: frozenset[Path]


# ── session layout ──────────────────────────────────────────────────────────────────────────────────────────


def session_dirs(records: list[FileRecord]) -> list[Path]:
    """The directories holding a ``driver.log``, one forge-rewrite session each."""

    return sorted({record.path.parent for record in records if record.path.name == _DRIVER_LOG})


def session_runs(session_dir: Path, records: list[FileRecord]) -> ArenaRuns:
    """The PORT step and each forge-loop iteration of one session directory."""

    files = [record.path for record in records if is_under(record.path, session_dir)]
    return ArenaRuns(_session_units(session_dir, files), frozenset(files))


def _session_units(session_dir: Path, files: list[Path]) -> list[dict[str, Any]]:
    name = session_dir.name
    arena_logs = sorted(path for path in files if path.parent == session_dir / "arena_logs" and path.suffix == ".log")
    run_dirs = sorted(
        path for path in session_dir.glob("arena_ws_*/run_*") if path.is_dir() and path.parent.parent == session_dir
    )
    task_dirs = sorted(path for path in run_dirs[0].glob("Aiter-task_*") if path.is_dir()) if run_dirs else []
    if len(arena_logs) != 1 or len(run_dirs) != 1 or len(task_dirs) != 1:
        message = (
            f"expected one arena log, one arena run directory, and one task directory; found "
            f"{len(arena_logs)}, {len(run_dirs)}, {len(task_dirs)}"
        )
        return [read_error(f"{name}/session", message, files)]
    run_dir, task_dir = run_dirs[0], task_dirs[0]
    session_txt = session_dir / "session.txt"
    session_config = session_dir / "config.yaml"
    driver_log = session_dir / _DRIVER_LOG
    logs = [path for path in (session_txt, session_config, driver_log, arena_logs[0]) if path.is_file()]
    try:
        lines = timeline([read_text(path) for path in (arena_logs[0], driver_log) if path.is_file()])
        header = _session_header(session_txt) if session_txt.is_file() else {}
        task = _TaskReader(task_dir, run_dir / ".task-sessions" / task_dir.name).read(session_config)
    except (OSError, ValueError, yaml.YAMLError) as error:
        return [read_error(f"{name}/session", f"{type(error).__name__}: {error}", files)]
    setup = Setup.read(lines)
    run = Run(
        label=name,
        run_id=setup.run_name or run_dir.name,
        identity=task.identity,
        provenance=_provenance(header, setup, task.hip),
        source_ms=setup.source_ms,
    )
    identity_logs = log_files(logs + task.files)
    return [port_unit(run, setup, identity_logs)] + [
        iteration_unit(run, iteration, identity_logs) for iteration in iterations(setup)
    ]


def _session_header(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in read_text(path).splitlines():
        match = _SESSION_TXT.match(line)
        if match:
            values[match.group(1).strip()] = match.group(2).strip()
    return values


def _provenance(header: dict[str, str], setup: Setup, hip: str | None) -> dict[str, Any]:
    extra: dict[str, Any] = {}
    if setup.forge_rewrite:
        extra["pipeline"] = _PIPELINE
    if hip:
        extra["hip"] = hip
    for key in ("kernelforge", "arena"):
        match = _CHECKOUT.match(header.get(key, ""))
        if match:
            extra[f"{key}_commit"] = match.group(1)
            if match.group(2):
                extra[f"{key}_branch"] = match.group(2)
    model = _MODEL_LINE.match(header.get("model", ""))
    if model:
        extra["llm_model"] = model.group(1)
        extra["llm_effort"] = model.group(2).strip()
    kb = _KB_LINE.match(header.get("KB", ""))
    if kb:
        extra["kb_mode"] = kb.group(1)
    for key in ("host", "started"):
        if header.get(key):
            extra[key] = header[key]
    return extra


@dataclass(frozen=True)
class _Task:
    """The task's conditions, its HIP runtime, and the files they were read from."""

    identity: dict[str, Any]
    hip: str | None
    files: list[Path]


class _TaskReader:
    """A task's conditions, read from the task's own files; each file a condition came from is recorded."""

    def __init__(self, task_dir: Path, task_session: Path) -> None:
        self.task_dir = task_dir
        self.task_session = task_session
        self.identity: dict[str, Any] = {}
        self.hip: str | None = None
        self.files: list[Path] = []

    def read(self, session_config: Path) -> _Task:
        self._config()
        if session_config.is_file():
            gpu = read_yaml(session_config).get("target_gpu_model")
            if gpu:
                self.identity["gpu"] = str(gpu).lower()
        self._runtime()
        self._workload()
        identity = {key: value for key, value in self.identity.items() if value not in (None, "")}
        return _Task(identity, self.hip or None, self.files)

    def _config(self) -> None:
        path = self.task_dir / "config.yaml"
        config = read_yaml(path)
        self.files.append(path)
        kernel_identity = config.get("kernel_identity") or {}
        baseline = config.get("baseline") or {}
        self.identity["kernel"] = kernel_identity.get("logical_operator")
        self.identity["framework"] = kernel_identity.get("source_owner")
        self.identity["backend"] = (config.get("candidate") or {}).get("language")
        family = _FAMILY_POLICY.match(str(baseline.get("diagnostic_reason") or ""))
        if family:
            self.identity["model"] = family.group(1)
        entry = _RUNTIME_ENTRYPOINT.search(str(config.get("description") or "")) or self._readme_entry()
        if entry is not None:
            self.identity["operator"] = entry.group(1).rsplit(".", 1)[-1]

    def _readme_entry(self) -> re.Match[str] | None:
        for readme in (self.task_dir / "README.md", self.task_dir / "BUNDLE_README.md"):
            found = _README_ENTRY.search(read_text(readme)) if readme.is_file() else None
            if found:
                self.files.append(readme)
                return found
        return None

    def _runtime(self) -> None:
        path = self.task_session / "runtime_identity.json"
        runtime = read_json(path)
        if runtime is not None:
            self.files.append(path)
            self.identity["backend_version"] = runtime.get(str(self.identity.get("backend") or ""))
            self.hip = runtime.get("hip")

    def _workload(self) -> None:
        candidates = (self.task_dir / "scripts" / "workload.json", self.task_dir / "workload.json")
        path = next((path for path in candidates if path.is_file()), None)
        workload = read_json(path) if path is not None else None
        if path is None or workload is None:
            return
        self.files.append(path)
        dtype = _workload_dtype(workload)
        self.identity["dtype"] = dtype if dtype is not None else self._validated_dtype()

    def _validated_dtype(self) -> str | None:
        """The first case's dtype as the arena's task validation recorded it, when it names one dtype."""

        validate = next(iter(sorted(self.task_session.glob("action-*-task-validate-task.json"))), None)
        if validate is None:
            return None
        cases = ((read_json(validate) or {}).get("result") or {}).get("cases") or [{}]
        dtype = cases[0].get("dtype")
        if not isinstance(dtype, str) or dtype.startswith("{"):
            return None
        self.files.append(validate)
        return dtype


def _workload_dtype(workload: dict[str, Any]) -> str | None:
    """The first input's dtype, as the workload declares it."""

    definition = workload.get("definition")
    if not isinstance(definition, dict):
        return None
    inputs = definition.get("inputs")
    first = next(iter(inputs.values()), None) if isinstance(inputs, dict) else None
    return first.get("dtype") if isinstance(first, dict) else None


# ── campaign layout: a campaign log, KernelForge's artifacts directory, and the agents' transcripts ─────────


def campaign_logs(records: list[FileRecord], claimed: set[Path]) -> list[Path]:
    """Arena logs outside a driver.log session that name a KernelForge artifacts directory."""

    logs: list[Path] = []
    for record in records:
        path = record.path
        if path.suffix != ".log" or path in claimed or record.size == 0:
            continue
        try:
            head = read_text(path, min(record.size, _HEAD_BYTES))
        except OSError:
            continue
        if _ARENA_START in head and "Forge artifacts: " in head:
            logs.append(path)
    return sorted(logs)


class _Role(Enum):
    """What one of the run's agents did, as its working directory tells, or in the workspace itself its prompt."""

    LANE = "lane"
    PLANNING = "planning"
    ANALYSIS = "analysis"
    IMPLEMENTER = "implementer"
    OTHER = "other"


@dataclass(frozen=True)
class _AgentSession:
    """A transcript of one of the run's agents, and its working directory relative to the artifacts directory."""

    transcript: Transcript
    where: str

    @property
    def role(self) -> _Role:
        """The planners run in the workspace itself, as does the implementer of a kernel at its root, whose prompt
        names a reasoning log or ends as every implementer's does."""

        if self.where.startswith("forge-lanes-"):
            return _Role.LANE
        if self.where == "engine":
            prompt = self.transcript.prompt
            implements = self.transcript.reasoning_log is not None or _IMPLEMENTER_PROMPT_END in prompt
            return _Role.IMPLEMENTER if implements else _Role.PLANNING
        if self.where.startswith("engine/forge_experiments/analysis"):
            return _Role.ANALYSIS
        if self.where.startswith("engine/"):
            return _Role.IMPLEMENTER
        return _Role.OTHER

    @property
    def lane(self) -> int | None:
        parts = self.where.split("/")
        return int(parts[1]) if self.role is _Role.LANE and len(parts) > 1 and parts[1].isdigit() else None


def _within(stamp: str, iteration: dict[str, Any]) -> bool:
    end = iteration.get("completed_at")
    return bool(stamp and end) and str(iteration.get("started_at") or "") <= stamp <= str(end)


def _implementer_sessions(iteration: dict[str, Any], plan: str, sessions: list[_AgentSession]) -> list[_AgentSession]:
    """The session that produced the iteration's candidate, when none was told to keep the reasoning log beside its
    plan, as in runs made before KernelForge named one.

    A single session is told its round's plan by path, so its prompt names ``orchestration/iter_NNN``. A lane session
    is given its lane's plan inline, so its prompt holds ``plan``, the plan KernelForge archived for the candidate it
    measured; the plan's first heading stands for it, or the whole plan when it has none.
    """

    origin = iteration.get("change_kind")
    if origin is CandidateOrigin.SESSION:
        marker = f"forge_experiments/orchestration/iter_{iteration['seq']:03d}/optimization_plan.md"
        implementers = [session for session in sessions if session.role is _Role.IMPLEMENTER]
        return [session for session in implementers if marker in session.transcript.prompt] or [
            session for session in implementers if _within(session.transcript.started, iteration)
        ]
    if origin is CandidateOrigin.LANE and plan:
        key = plan_title(plan) or plan
        end = str(iteration.get("completed_at") or "~")
        found = [
            session
            for session in sessions
            if session.role is _Role.LANE
            and session.lane == iteration.get("lane_number")
            and key in session.transcript.prompt
            and session.transcript.started <= end
        ]
        return found[-1:]
    return []


@dataclass(frozen=True)
class _Campaign:
    """One campaign's KernelForge artifacts, the transcripts of its agents, the iterations its log shows by seq, and
    the files all its units name."""

    root: Path
    experiments: Path
    sessions: list[_AgentSession]
    rounds: dict[int, dict[str, Any]]
    files: list[Path]

    def iteration_unit(self, run: Run, iteration: dict[str, Any]) -> dict[str, Any]:
        """An iteration's unit from its log lines, KernelForge's archive of it, and its implementer's transcript.

        forge-rewrite's logged source baseline is the original kernel's time, so it wins over the archive's, which is
        the time of the port the forge-loop started from when the run state records no pristine baseline.
        """

        archive = IterationArchive(self.experiments, iteration["seq"])
        meta = archive.meta()
        origin = iteration.get("change_kind")
        plan = str((meta or {}).get("plan") or "").strip()
        plan_file = archive.plan_file(origin, plan)
        unit = iteration_unit(run, iteration, [])
        archived = archive.fields(origin, meta, plan_file)
        if run.source_ms is not None:
            archived.pop("source_ms", None)
        unit.update(archived)
        transcripts = archive.sessions(plan_file, [session.transcript for session in self.sessions]) or [
            session.transcript for session in _implementer_sessions(iteration, plan, self.sessions)
        ]
        unit.update(implementer_fields(transcripts, plan_file, self.root))
        files = archive.files(plan_file) + [transcript.path for transcript in transcripts]
        unit["log_files"] = log_files(self.files + files + self._planners(plan_file, iteration))
        return unit

    def _planners(self, plan_file: Path | None, iteration: dict[str, Any]) -> list[Path]:
        """The transcripts of the planners of the round that planned ``plan_file``, or of ``iteration``'s own.

        A fan-out round plans all its lanes at once, and its later lanes are measured in later iterations.
        """

        named = _ROUND_DIR.match(plan_file.parent.name) if plan_file is not None else None
        planned = self.rounds.get(int(named.group(1)), iteration) if named else iteration
        return [
            session.transcript.path
            for session in self.sessions
            if session.role is _Role.PLANNING and _within(session.transcript.started, planned)
        ]


def _agent_sessions(records: list[FileRecord], artifacts: str) -> list[_AgentSession]:
    """The Claude Code sessions whose working directory is inside this run's KernelForge artifacts directory."""

    project = re.sub(r"[^A-Za-z0-9]", "-", artifacts)
    found: list[_AgentSession] = []
    for record in records:
        path = record.path
        if not is_transcript(path):
            continue
        if path.parent.name != project and not path.parent.name.startswith(project + "-"):
            continue
        transcript = read_transcript(path)
        where = next(
            (
                cwd[len(artifacts) + 1 :]
                for cwd in transcript.cwds
                if cwd == artifacts or cwd.startswith(artifacts + "/")
            ),
            None,
        )
        if where is not None:
            found.append(_AgentSession(transcript, where))
    return sorted(found, key=lambda session: (session.transcript.started, str(session.transcript.path)))


def _artifacts_dir(records: list[FileRecord], run_name: str, artifacts: str) -> Path | None:
    """The local copy of ``<workspace>/<run>/<task>-forge-<id>``, which the log names by its in-container path."""

    name = artifacts.rsplit("/", 1)[-1]
    for record in records:
        parts = record.path.parts
        if name in parts:
            index = parts.index(name)
            if index > 0 and parts[index - 1] == run_name:
                return Path(*parts[: index + 1])
    return None


def _campaign_provenance(text: str, lines: list[LogLine], setup: Setup) -> dict[str, Any]:
    extra: dict[str, Any] = {}
    if setup.forge_rewrite:
        extra["pipeline"] = _PIPELINE
    if (engine := _ENGINE_BANNER.search(text)) is not None:
        extra["kernelforge_commit"] = engine.group(2)
    if (image := _IMAGE_BANNER.search(text)) is not None:
        extra["image"] = image.group(1)
    if (config := _CONFIG_BANNER.search(text)) is not None:
        extra["arena_config"] = config.group(1)
    implementer = next(
        (match for line in lines if line.forge and (match := _IMPLEMENTER.match(line.body.strip()))), None
    )
    if implementer is not None:
        extra["llm_backend"], extra["llm_model"] = implementer.group(1), implementer.group(2)
    return extra


def campaign_runs(log: Path, records: list[FileRecord], root: Path) -> ArenaRuns:
    """The PORT step, when the campaign ran forge-rewrite, and each forge-loop iteration of one campaign log."""

    try:
        text = read_text(log)
    except OSError as error:
        unit = read_error(f"{log.parent.name}/{log.stem}/session", f"{type(error).__name__}: {error}", [log])
        return ArenaRuns([unit], frozenset({log}))
    lines = timeline([text])
    setup = Setup.read(lines)
    run_name = setup.run_name
    artifacts = next(
        (
            match.group(1).rstrip("/")
            for line in lines
            if not line.forge and (match := _FORGE_ARTIFACTS.match(line.body.strip()))
        ),
        None,
    )
    label = _RUN_STAMP.sub("", run_name) if run_name else f"{log.parent.name}/{log.stem}"
    forge_dir = _artifacts_dir(records, run_name, artifacts) if run_name and artifacts else None
    if forge_dir is None or run_name is None or artifacts is None:
        message = (
            f"no KernelForge artifacts directory {artifacts or '(not logged)'} for run "
            f"{run_name or '(not logged)'} under this log root"
        )
        return ArenaRuns([read_error(f"{label}/session", message, [log])], frozenset({log}))
    experiments = forge_dir / "engine" / "forge_experiments"
    config_path, state_path = experiments / "campaign_config.json", experiments / "run_state.json"
    model_class = next(
        (match.group(1) for line in lines if not line.forge and (match := _MODEL_CLASS.search(line.body))), None
    )
    run = Run(
        label=label,
        run_id=run_name,
        identity=campaign_identity(read_json(config_path) or {}, operator=model_class),
        provenance=_campaign_provenance(text, lines, setup),
        source_ms=setup.source_ms,
    )
    logged = iterations(setup)
    campaign = _Campaign(
        root=root,
        experiments=experiments,
        sessions=_agent_sessions(records, artifacts),
        rounds={iteration["seq"]: iteration for iteration in logged},
        files=[log] + [path for path in (config_path, state_path) if path.is_file()],
    )
    units = [port_unit(run, setup, log_files(campaign.files))] if setup.port_start is not None else []
    units += [campaign.iteration_unit(run, iteration) for iteration in logged]
    claimed = {log} | {record.path for record in records if is_under(record.path, forge_dir)}
    return ArenaRuns(units, frozenset(claimed))
