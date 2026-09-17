"""Read-only specialist sessions and bounded parallel execution."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence

from kernelforge.agent_backends import (
    AgentBackend,
    AgentProviderError,
    AgentRunSpec,
    AgentToolPolicy,
    StdioMcpServer,
    watchdog_timeout_sec,
)
from kernelforge.agent_backends.session_resume import is_api_failure
from kernelforge.llm.process_reaping import ReapReport, reap_processes_under
from kernelforge.mcp_server.probe_stdio_server import (
    ANALYSIS_RESERVE_SEC,
    BUDGET_EXHAUSTED,
    BUDGET_SEC_ENV,
    DEVICE_LOCK_ENV,
    LEDGER_ENV,
    MAX_PROBES_ENV,
    MEASURED,
    PROBE_TOOL_GRACE_SEC,
    ROUND_BUDGET_ENV,
    SCRATCH_ENV,
    SESSION_DEADLINE_ENV,
    TOOL_NAMES as PROBE_TOOL_NAMES,
    WORKSPACE_ENV,
    ProbeSandboxError,
    load_sandbox,
    probe_primitive_status,
    probe_timeout_sec,
)
from kernelforge.orchestrator.contracts import (
    OrchestrationContext,
    SpecialistAssignment,
    SpecialistDefinition,
    SpecialistFailure,
    SpecialistOutcome,
)

log = logging.getLogger(__name__)


_SPECIALIST_SYSTEM_PROMPT = """\
You are a GPU-kernel optimization specialist with read-only access to the
workspace.

Analyze only the assigned cases and evidence. Do not edit files, run shell
commands, alter measurement inputs, or present an inference as a profiler fact.
Any measurement you make comes from a tool listed below, if one is listed at
all. Cite the evidence paths that support important claims.

Write a concise technical analysis for the orchestration planner. Focus on
high-value mechanisms, concrete implementation options, feasibility, expected
impact, dependencies, and correctness or performance risks. Use ordinary
Markdown; no fixed schema is required.
"""


_PROBE_SERVER_KEY = "specialist_probe"
_PROBE_LEDGER_NAME = "probe_ledger.jsonl"
_PROBE_BUDGET_NAME = "round_budget.json"
_PROBE_SECTION_TITLE = "## Scratch probe ledger"

# What the probe's MCP child needs from this process and would not otherwise get.
_PROBE_CHILD_ENV_VARS = (
    # The child's own launch: `python -m kernelforge...` from a checkout.
    "PYTHONPATH",
    "PYTHONHOME",
    "VIRTUAL_ENV",
    "CONDA_PREFIX",
    # ROCm toolchain and runtime: without these the driver finds no compiler and no libraries.
    "ROCM_PATH",
    "HIP_PATH",
    "HIP_PLATFORM",
    "HIP_CLANG_PATH",
    "LD_LIBRARY_PATH",
    "PYTORCH_ROCM_ARCH",
    "GPU_TARGET",
    # Which device this campaign may touch.
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
    # Build caches, and the aiter cache isolation this campaign installs in
    # ``loop.aiter_cache.configure_aiter_cache_isolation``.
    "TRITON_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "AITER_ROOT_DIR",
    "AITER_JIT_DIR",
    # FlyDSL is the third compiler behind that isolation.
    "FLYDSL_RUNTIME_CACHE_DIR",
    "FORGE_AITER_CACHE_ROOT",
    "FORGE_AITER_CACHE_OWNER_PID",
    "TMPDIR",
    # The rank count the campaign measures under.
    "FORGE_NPROC_PER_NODE",
)
# Whole families rather than named members: the HSA and AMD runtime knobs a node is configured with are open-ended,
# and a probe that ran without them would not be measuring the configuration the campaign measures.
_PROBE_CHILD_ENV_PREFIXES = ("HSA_", "AMD_", "ROCM_", "TRITON_")


def _device_lock_path(workspace: Path) -> Path:
    """The campaign sentinel a fan-out lane's serialized driver locks."""
    from kernelforge.loop.fanout import campaign_device_lock_path

    return campaign_device_lock_path(workspace)


def _probe_child_env() -> dict[str, str]:
    """Collect what the probe's MCP child needs from this process's environment."""
    return {
        name: value
        for name, value in os.environ.items()
        if value.strip() and (name in _PROBE_CHILD_ENV_VARS or name.startswith(_PROBE_CHILD_ENV_PREFIXES))
    }


@dataclass(frozen=True)
class SpecialistProbeConfig:
    """Bound the scratch measurement one analysis phase may run."""

    scratch_root: str
    max_probes: int = 6
    budget_sec: float = 600.0

    def __post_init__(self) -> None:
        if not self.scratch_root.strip():
            raise ValueError("scratch_root must not be empty")
        if self.max_probes <= 0:
            raise ValueError("max_probes must be greater than zero")
        if self.budget_sec <= 0:
            raise ValueError("budget_sec must be greater than zero")


@dataclass
class _ProbeRound:
    """One analysis phase's scratch tree and the budget its specialists share."""

    root: Path | None = None
    budget_path: Path | None = None
    error: str = ""
    # What the teardown found still running in the round's tree.
    reaped: ReapReport | None = None


@asynccontextmanager
async def _probe_round(probe: SpecialistProbeConfig | None):
    """Give one round its own scratch tree, and take it away when the round ends."""
    if probe is None:
        yield None
        return
    root = Path(probe.scratch_root).expanduser().resolve()
    try:
        root.mkdir(parents=True, exist_ok=True)
        round_root = Path(tempfile.mkdtemp(prefix="round-", dir=str(root)))
    except OSError as error:
        # Not fatal to the round: the specialists still analyse, they just cannot measure, and ``_prepare_probe``
        # reports why.
        log.warning("specialist probe scratch root unusable: %s", error)
        yield _ProbeRound(error=f"the round scratch tree could not be created: {error}")
        return
    opened = _ProbeRound(root=round_root, budget_path=round_root / _PROBE_BUDGET_NAME)
    try:
        yield opened
    finally:
        opened.reaped = await reap_processes_under(round_root, description=f"left running in probe round {round_root}")
        shutil.rmtree(round_root, ignore_errors=True)


@dataclass(frozen=True)
class _ProbeSetup:
    """Describe what measurement, if any, one specialist session may perform."""

    enabled: bool = False
    scratch_dir: Path | None = None
    ledger_path: Path | None = None
    workspace: str = ""
    unavailable_reason: str = ""
    config: SpecialistProbeConfig | None = None
    # The counters this round's specialists share.
    budget_path: Path | None = None
    # The campaign's device sentinel, the same file a fan-out lane's driver flocks.
    device_lock: Path | None = None
    # This session's own wall clock, threaded through so the probe can refuse a measurement that would leave the
    # analysis unwritten.
    session_timeout_sec: float = 0.0
    # None means the parent did not say.
    session_deadline: float | None = None

    def server_env(self) -> dict[str, str]:
        if self.config is None:
            return {}
        return {
            # The parent environment the MCP client does not forward for us -- see ``_PROBE_CHILD_ENV_VARS``.
            **_probe_child_env(),
            SCRATCH_ENV: str(self.scratch_dir),
            WORKSPACE_ENV: self.workspace,
            LEDGER_ENV: str(self.ledger_path),
            MAX_PROBES_ENV: str(self.config.max_probes),
            BUDGET_SEC_ENV: str(self.config.budget_sec),
            ROUND_BUDGET_ENV: str(self.budget_path or ""),
            DEVICE_LOCK_ENV: str(self.device_lock or ""),
            **({SESSION_DEADLINE_ENV: f"{self.session_deadline:.3f}"} if self.session_deadline is not None else {}),
        }

    def probe_ceiling_sec(self) -> int:
        """The longest one probe may run, as the prompt states it."""
        if self.config is None:
            return 0
        return probe_timeout_sec(
            budget_remaining=self.config.budget_sec,
            session_remaining=self.session_timeout_sec,
            requested=self.config.budget_sec,
        )

    def mcp_servers(self) -> dict[str, StdioMcpServer]:
        if not self.enabled or self.config is None:
            return {}
        return {
            _PROBE_SERVER_KEY: StdioMcpServer(
                command=sys.executable,
                args=("-m", "kernelforge.mcp_server.probe_stdio_server"),
                env=self.server_env(),
                startup_timeout_sec=15,
                # The ceiling the prompt states, plus the grace the server allows itself past it: a client that timed
                # out first would kill the call before the server wrote its ledger line, and the ledger is the only
                # channel back to this process.
                tool_timeout_sec=self.probe_ceiling_sec() + PROBE_TOOL_GRACE_SEC,
                tools=self.tool_names(),
            )
        }

    def tool_names(self) -> tuple[str, ...]:
        if not self.enabled:
            return ()
        return tuple(f"mcp__{_PROBE_SERVER_KEY}__{name}" for name in PROBE_TOOL_NAMES)


def _deadline_prompt_section(session_timeout_sec: float) -> str:
    """Tell the specialist how long it has, because nothing else does."""
    total = max(0.0, float(session_timeout_sec))
    reserve = f"{ANALYSIS_RESERVE_SEC:.0f}s"
    return f"""
Time limit -- read this before you plan the work:
You have {total:.0f}s ({total / 60:.0f} min), and it is HARD. At that moment the
session is killed and anything you have not already written is lost. A killed
session returns no analysis at all, which the round reads as an infrastructure
failure rather than as a short answer, and a round whose specialists all return
nothing is abandoned. Nobody writes it for you, so a brief analysis delivered
beats a thorough one that never lands.

The real deadline is {reserve} before that: the last {reserve} are for writing,
not for work. Before every further step, ask whether its answer can still reach
the page. If it cannot, stop investigating AT ONCE and write what you have,
naming what you left unresolved. A question reported as unresolved is useful; an
answer you never wrote down is not.
"""


def _probe_prompt_section(setup: _ProbeSetup) -> str:
    """Tell an otherwise read-only specialist what it may measure."""
    if not setup.enabled or setup.config is None:
        return ""
    tool = setup.tool_names()[0]
    return f"""
Bounded measurement:
You may settle a question rather than only argue it. `{tool}` re-runs the
workspace benchmark driver for one named case with declared dispatch constants
overridden, in a scratch directory of its own; it edits nothing, and the
workspace stays read-only to you. Its record of this session is kept under
{setup.scratch_dir}, outside the canonical tree. This round gets
{setup.config.max_probes} probes and {setup.config.budget_sec:.0f}s of wall
clock IN TOTAL, shared with the other specialists analysing it at the same
time, and no single probe may run longer than {setup.probe_ceiling_sec()}s.
Every result carries three numbers: the probes and the seconds left of the
round's shared budget, and the seconds left of your own session. Spend your
share on the constants your recommendation turns on.

Every call costs one of the count, including one that is refused or comes back
without a measurement. A probe is also refused outright once too little of your
own session is left to write the analysis, and it may spend part of the budget
waiting for the GPU, which measures one thing at a time. When a result says a
budget or the session clock is spent, stop probing.

Mark every claim that came from a probe as measured and cite the probe label;
leave the rest marked as argued. Report a probe that failed or an exhausted
budget in your analysis -- an unmeasured question is a different thing from one
nobody asked.
"""


def _read_probe_ledger(setup: _ProbeSetup) -> tuple[list[dict], str]:
    """Read back what the probe server recorded, and any reason it cannot be."""
    if setup.ledger_path is None or not setup.ledger_path.exists():
        return [], ""
    records: list[dict] = []
    try:
        lines = setup.ledger_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        return [], f"the probe ledger could not be read: {error}"
    for line in lines:
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            return records, f"the probe ledger has an unreadable entry: {line[:200]}"
        if isinstance(entry, dict):
            records.append(entry)
    return records, ""


def _render_probe_record(record: dict) -> str:
    label = str(record.get("label") or "unlabelled")
    case_id = str(record.get("case_id") or "unknown case")
    status = str(record.get("status") or "unknown")
    detail = str(record.get("detail") or "").strip()
    if status == MEASURED:
        detail = f"{record.get('case_ms')} ms" + (f" ({detail})" if detail else "")
    return f"- probe {record.get('probe_index')} `{label}` on case `{case_id}`: {status}" + (
        f" -- {detail}" if detail else ""
    )


def _render_probe_report(setup: _ProbeSetup) -> str:
    """Render what this specialist measured, so a reader can tell it from argument."""
    if not setup.enabled and not setup.unavailable_reason:
        return ""
    if setup.unavailable_reason:
        return (
            f"{_PROBE_SECTION_TITLE}\n"
            f"No probe ran: {setup.unavailable_reason}. Every claim above is "
            "argued, not measured."
        )
    records, ledger_error = _read_probe_ledger(setup)
    lines = [_PROBE_SECTION_TITLE]
    if ledger_error:
        lines.append(
            f"The probe was offered but its record is incomplete: {ledger_error}. "
            "Treat the probe evidence below as partial."
        )
    if not records:
        lines.append("The probe was offered and never called: nothing above is a measured claim.")
        return "\n".join(lines)
    lines.extend(_render_probe_record(record) for record in records)
    exhausted = [record for record in records if record.get("status") == BUDGET_EXHAUSTED]
    measured = sum(1 for record in records if record.get("status") == MEASURED)
    lines.append(
        f"{measured} of {len(records)} probe attempts produced a measurement"
        + ("; the budget was exhausted and the remaining questions stay unmeasured." if exhausted else ".")
    )
    return "\n".join(lines)


def _summarize_probes(setup: _ProbeSetup) -> str:
    """Compress the probe outcome to one clause for a failed specialist."""
    if not setup.enabled:
        return f"no probe: {setup.unavailable_reason}" if setup.unavailable_reason else ""
    records, ledger_error = _read_probe_ledger(setup)
    if ledger_error:
        return f"probe ledger incomplete: {ledger_error}"
    if not records:
        return "probe offered and never called"
    measured = sum(1 for record in records if record.get("status") == MEASURED)
    return f"{measured} of {len(records)} probe attempts measured"


def build_specialist_prompts(
    *,
    definition: SpecialistDefinition,
    assignment: SpecialistAssignment,
    context: OrchestrationContext,
    session_timeout_sec: float,
    probe_setup: _ProbeSetup | None = None,
) -> tuple[str, str]:
    """Build one evidence-scoped specialist prompt."""
    if assignment.role_id != definition.role_id:
        raise ValueError("specialist assignment role does not match definition")
    unknown_cases = set(assignment.target_case_ids) - context.case_ids
    if unknown_cases:
        raise ValueError("specialist assignment references unknown cases: " + ", ".join(sorted(unknown_cases)))

    system_prompt = (
        f"{_SPECIALIST_SYSTEM_PROMPT}"
        # Ahead of the probe section, which speaks of "your own session" and needs that clock established first.
        f"{_deadline_prompt_section(session_timeout_sec)}"
        f"{_probe_prompt_section(probe_setup) if probe_setup else ''}\n"
        f"Specialist role: {definition.description}\n\n"
        f"Role instructions:\n{definition.instructions.strip()}"
    )
    global_analysis_refs = tuple(
        reference
        for reference in context.evidence_refs
        if reference.kind
        in {
            "analysis_artifact_catalog",
            "analysis_bundle",
            "analysis_cumulative_diff",
            "analysis_summary",
            "analysis_source_map",
            "analysis_workflow",
        }
    )
    scoped_evidence = tuple(
        {
            reference.path: reference
            for reference in (
                *assignment.evidence_refs,
                *global_analysis_refs,
            )
        }.values()
    )
    payload = {
        "assignment": assignment.to_dict(),
        "context": context.to_prompt_dict(
            case_ids=assignment.target_case_ids,
            evidence_refs=scoped_evidence,
        ),
    }
    user_prompt = (
        "Analyze the assigned optimization problem and produce recommendations "
        "that another planner can combine with other specialists. Do not merely "
        "enumerate generic ideas: compare the most relevant options and explain "
        "what is worth implementing.\n\n" + json.dumps(payload, indent=2, sort_keys=True)
    )
    return system_prompt, user_prompt


class SpecialistAgent:
    """Run one registered specialist role through an injected backend."""

    def __init__(
        self,
        *,
        definition: SpecialistDefinition,
        backend: AgentBackend,
        timeout_sec: int,
        max_turns: int,
        probe: SpecialistProbeConfig | None = None,
    ) -> None:
        if timeout_sec <= 0:
            raise ValueError("timeout_sec must be greater than zero")
        if max_turns <= 0:
            raise ValueError("max_turns must be greater than zero")
        self.definition = definition
        self.backend = backend
        self.timeout_sec = timeout_sec
        self.max_turns = max_turns
        self.probe = probe

    async def run(
        self,
        assignment: SpecialistAssignment,
        context: OrchestrationContext,
        *,
        usage=None,
        probe_round: _ProbeRound | None = None,
    ) -> SpecialistOutcome:
        """Run one isolated specialist and normalize every failure."""
        started = time.monotonic()
        probe_setup = _ProbeSetup()
        try:
            probe_setup = self._prepare_probe(assignment, context, probe_round)
            system_prompt, user_prompt = build_specialist_prompts(
                definition=self.definition,
                assignment=assignment,
                context=context,
                session_timeout_sec=self.timeout_sec,
                probe_setup=probe_setup,
            )
            result = await asyncio.wait_for(
                self.backend.run(
                    AgentRunSpec(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        role=f"specialist {self.definition.role_id}",
                        cwd=context.workspace,
                        writable=False,
                        timeout_sec=self.timeout_sec,
                        tool_policy=AgentToolPolicy(
                            read=True,
                            search=True,
                            write=False,
                            shell=False,
                            max_turns=self.max_turns,
                            extra_tools=probe_setup.tool_names(),
                        ),
                        protected_globs=["*"],
                        mcp_servers=probe_setup.mcp_servers(),
                    ),
                    usage=usage,
                ),
                timeout=watchdog_timeout_sec(self.timeout_sec),
            )
            if is_api_failure(result):
                detail = (
                    result.stderr_tail or result.end_reason or "specialist backend failed before producing an answer"
                )
                return self._failure(
                    assignment,
                    started,
                    kind="backend_failure",
                    message=detail,
                    probe_setup=probe_setup,
                )
            content = (result.text or "").strip()
            if not content:
                return self._failure(
                    assignment,
                    started,
                    kind="empty_output",
                    message="specialist returned no analysis",
                    probe_setup=probe_setup,
                )
            report = _render_probe_report(probe_setup)
            return SpecialistOutcome(
                assignment_id=assignment.assignment_id,
                role_id=assignment.role_id,
                duration_sec=time.monotonic() - started,
                content=f"{content}\n\n{report}" if report else content,
            )
        except asyncio.TimeoutError:
            return self._failure(
                assignment,
                started,
                kind="timeout",
                message=f"specialist exceeded {self.timeout_sec}s timeout",
                probe_setup=probe_setup,
            )
        except AgentProviderError as error:
            log.warning(
                "specialist %s provider failure: %s",
                assignment.role_id,
                error,
            )
            return self._failure(
                assignment,
                started,
                kind="backend_failure",
                message=f"{type(error).__name__}: {error}",
                probe_setup=probe_setup,
            )
        except Exception as error:  # noqa: BLE001 - failures are isolated by design
            log.exception(
                "specialist %s failed unexpectedly",
                assignment.role_id,
            )
            return self._failure(
                assignment,
                started,
                kind="backend_error",
                message=f"{type(error).__name__}: {error}",
                probe_setup=probe_setup,
            )

    def _prepare_probe(
        self,
        assignment: SpecialistAssignment,
        context: OrchestrationContext,
        probe_round: _ProbeRound | None = None,
    ) -> _ProbeSetup:
        """Create this assignment's scratch root, or say why it has none."""
        if self.probe is None:
            return _ProbeSetup()
        _, unusable = probe_primitive_status()
        if unusable:
            return self._no_probe(assignment, unusable)
        capabilities = getattr(self.backend, "capabilities", None)
        if not getattr(capabilities, "mcp", False):
            return self._no_probe(
                assignment,
                "the specialist backend does not serve MCP tools, so the probe could not be offered",
            )
        # A session with no room for one probe must not be offered one.
        if float(self.timeout_sec) - ANALYSIS_RESERVE_SEC <= 0:
            return self._no_probe(
                assignment,
                f"this specialist session is {self.timeout_sec}s long and "
                f"{ANALYSIS_RESERVE_SEC:.0f}s of it is reserved for writing the "
                "analysis, which leaves no room for a probe",
            )
        if probe_round is not None and probe_round.error:
            return self._no_probe(assignment, probe_round.error)
        workspace = Path(context.workspace).expanduser().resolve()
        scratch_root = (
            probe_round.root
            if probe_round is not None and probe_round.root is not None
            else Path(self.probe.scratch_root).expanduser().resolve()
        )
        scratch_dir = scratch_root / assignment.assignment_id
        ledger_path = scratch_dir / _PROBE_LEDGER_NAME
        candidate = _ProbeSetup(
            enabled=True,
            scratch_dir=scratch_dir,
            ledger_path=ledger_path,
            workspace=str(workspace),
            config=self.probe,
            budget_path=(probe_round.budget_path if probe_round is not None else None),
            # The campaign's sentinel, not one of the probe's own: the GPU a probe times on is the one a fan-out lane
            # drives, so a probe and a lane queue on the same file.
            device_lock=_device_lock_path(workspace),
            session_timeout_sec=float(self.timeout_sec),
            session_deadline=time.time() + float(self.timeout_sec),
        )
        try:
            scratch_dir.mkdir(parents=True, exist_ok=True)
            # An assignment id repeats across rounds.
            ledger_path.unlink(missing_ok=True)
        except OSError as error:
            return self._no_probe(assignment, f"the scratch root could not be prepared: {error}")
        # The server validates its own environment and would refuse a session it cannot serve; a refusal it cannot
        # write to the ledger reads downstream like a probe nobody called, so the same check runs here, where "not
        # offered, and here is why" is still a thing the parent can report.
        try:
            load_sandbox(candidate.server_env())
        except ProbeSandboxError as error:
            return self._no_probe(assignment, str(error))
        # Create after load_sandbox: a refused probe must not leave an orphaned sentinel.
        try:
            candidate.device_lock.parent.mkdir(parents=True, exist_ok=True)
            candidate.device_lock.touch(exist_ok=True)
        except OSError as error:
            return self._no_probe(assignment, f"could not create device sentinel: {error}")
        return candidate

    def _no_probe(self, assignment: SpecialistAssignment, reason: str) -> _ProbeSetup:
        """Disable the probe for one assignment, in the log and in the analysis."""
        log.warning(
            "specialist probe not offered for %s (%s): %s",
            assignment.assignment_id,
            assignment.role_id,
            reason,
        )
        return _ProbeSetup(unavailable_reason=reason)

    @staticmethod
    def _failure(
        assignment: SpecialistAssignment,
        started: float,
        *,
        kind: str,
        message: str,
        probe_setup: _ProbeSetup | None = None,
    ) -> SpecialistOutcome:
        summary = _summarize_probes(probe_setup) if probe_setup else ""
        return SpecialistOutcome(
            assignment_id=assignment.assignment_id,
            role_id=assignment.role_id,
            duration_sec=time.monotonic() - started,
            failure=SpecialistFailure(
                kind=kind,
                message=f"{message}; {summary}" if summary else message,
            ),
        )


@dataclass(frozen=True)
class SpecialistRunResult:
    """What one analysis phase produced, and what it left on the device."""

    outcomes: tuple[SpecialistOutcome, ...] = ()
    # The round scratch tree's teardown report, None when the round had no tree.
    reaped: ReapReport | None = None

    @property
    def contended(self) -> bool:
        """Whether this round left the device unsafe to measure on."""
        return self.reaped is not None and self.reaped.contended


class SpecialistPool:
    """Run specialist assignments concurrently with bounded failure isolation."""

    def __init__(
        self,
        agents: Mapping[str, SpecialistAgent],
        *,
        max_parallel: int,
    ) -> None:
        if max_parallel <= 0:
            raise ValueError("max_parallel must be greater than zero")
        if not agents:
            raise ValueError("agents must not be empty")
        if set(agents) != {agent.definition.role_id for agent in agents.values()}:
            raise ValueError("agent mapping keys must match specialist role_id values")
        self._agents = dict(agents)
        self._max_parallel = max_parallel

    async def run(
        self,
        assignments: Sequence[SpecialistAssignment],
        context: OrchestrationContext,
        *,
        usage=None,
    ) -> SpecialistRunResult:
        """Run all assignments without letting one failure cancel siblings."""
        assignment_ids = [assignment.assignment_id for assignment in assignments]
        if len(set(assignment_ids)) != len(assignment_ids):
            raise ValueError("assignment_id values must be unique")
        semaphore = asyncio.Semaphore(self._max_parallel)
        probe = next(
            (agent.probe for agent in self._agents.values() if agent.probe is not None),
            None,
        )

        async def run_one(
            assignment: SpecialistAssignment,
            probe_round: _ProbeRound | None,
        ) -> SpecialistOutcome:
            agent = self._agents.get(assignment.role_id)
            if agent is None:
                return SpecialistOutcome(
                    assignment_id=assignment.assignment_id,
                    role_id=assignment.role_id,
                    duration_sec=0.0,
                    failure=SpecialistFailure(
                        kind="unknown_role",
                        message=(f"no specialist registered for {assignment.role_id!r}"),
                    ),
                )
            async with semaphore:
                return await agent.run(assignment, context, usage=usage, probe_round=probe_round)

        async with _probe_round(probe) as probe_round:
            outcomes = await asyncio.gather(*(run_one(item, probe_round) for item in assignments))
        # Read after the block, not inside it: the teardown that writes it runs as the context manager closes.
        return SpecialistRunResult(
            outcomes=tuple(
                sorted(
                    outcomes,
                    key=lambda item: (item.role_id, item.assignment_id),
                )
            ),
            reaped=probe_round.reaped if probe_round is not None else None,
        )
