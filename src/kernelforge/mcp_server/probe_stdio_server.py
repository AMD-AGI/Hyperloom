# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Stdio MCP server that lets a read-only specialist measure one variant."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import importlib
import inspect
import json
import logging
import math
import os
from pathlib import Path
import time
from typing import Any

from kernelforge.mcp_server import stdio_transport

log = logging.getLogger(__name__)

#: Re-exported so the two servers stay one import away from the shared wire
#: format; both spellings name the same objects.
InvalidParamsError = stdio_transport.InvalidParamsError
_write_message = stdio_transport.write_message
_write_error = stdio_transport.write_error

SERVER_NAME = "kernelforge-specialist-probe"
# Agents see these as mcp__specialist_probe__<name>.
TOOL_NAMES = ("probe_variant",)

SCRATCH_ENV = "FORGE_PROBE_SCRATCH"
WORKSPACE_ENV = "FORGE_PROBE_WORKSPACE"
LEDGER_ENV = "FORGE_PROBE_LEDGER"
MAX_PROBES_ENV = "FORGE_PROBE_MAX"
BUDGET_SEC_ENV = "FORGE_PROBE_BUDGET_SEC"
# The round's shared counters; see ``ProbeBudget``.
ROUND_BUDGET_ENV = "FORGE_PROBE_ROUND_BUDGET"
# The campaign-wide sentinel a run must flock before it touches the GPU, from
# ``kernelforge.loop.fanout.campaign_device_lock_path``.
DEVICE_LOCK_ENV = "FORGE_PROBE_DEVICE_LOCK"
# Unix timestamp at which the specialist session this server serves is killed.
SESSION_DEADLINE_ENV = "FORGE_PROBE_SESSION_DEADLINE"

PRIMITIVE_MODULE = "kernelforge.mcp_server.tools.bench"
PRIMITIVE_ATTR = "sweep_case"
PRIMITIVE_PATH = f"{PRIMITIVE_MODULE}.{PRIMITIVE_ATTR}"
# The keywords this server calls the primitive with.
PRIMITIVE_KEYWORDS = (
    "driver_script",
    "case_id",
    "constants",
    "timeout_sec",
    "prefix_constants",
)

# Ledger statuses.
MEASURED = "measured"
FAILED = "failed"
BUDGET_EXHAUSTED = "budget_exhausted"
UNAVAILABLE = "unavailable"
REFUSED = "refused"
DEVICE_BUSY = "device_busy"

DEFAULT_PROBE_TIMEOUT_SEC = 300
# Seconds the server waits past a probe's own ceiling before abandoning it, so a primitive that is a moment late is
# reported as late rather than lost.
PROBE_TOOL_GRACE_SEC = 5
# One probe's report; a compile log can be arbitrarily long.
MAX_DETAIL_CHARS = 2_000

# Seconds of the specialist's session that no probe may take.
ANALYSIS_RESERVE_SEC = 120.0
# The most of what is left of a session one probe budget may claim.
SESSION_PROBE_FRACTION = 0.5
# How often a probe waiting for the device retries the sentinel.
DEVICE_LOCK_POLL_SEC = 1.0

# Attempts one ledger holds.
MAX_LEDGER_RECORDS = 200


class ProbeSandboxError(RuntimeError):
    """The scratch sandbox this server was configured with is unusable."""


def wall_clock() -> float:
    """Now, on the clock the session deadline is expressed in."""
    return time.time()


def monotonic_clock() -> float:
    """Elapsed-time clock for the device wait and a probe's own duration."""
    return time.monotonic()


@dataclass(frozen=True)
class ProbeSandbox:
    """Hold the scratch root, the budgets and the ledger for one specialist."""

    scratch_root: Path
    workspace: Path
    ledger_path: Path
    max_probes: int
    budget_sec: float
    # Shared counters for the round this session belongs to; None keeps them in this process.
    budget_path: Path | None = None
    # The campaign's device sentinel.
    device_lock: Path | None = None
    # When the specialist session is killed.
    session_deadline: float | None = None

    def session_remaining_sec(self) -> float:
        """Seconds left in the specialist session, or infinity if unbounded."""
        if self.session_deadline is None:
            return math.inf
        return self.session_deadline - wall_clock()


def probe_budget_sec(*, configured_remaining: float, session_remaining: float) -> float:
    """What is really left to spend on probing, given the session's own clock."""
    return max(0.0, min(configured_remaining, session_remaining * SESSION_PROBE_FRACTION))


def probe_timeout_sec(
    *,
    budget_remaining: float,
    session_remaining: float,
    requested: Any = None,
) -> int:
    """Ceiling for one probe: the budget, the session, and what was asked for."""
    allowed = min(budget_remaining, session_remaining - ANALYSIS_RESERVE_SEC)
    # ``requested > 0`` belongs in this test, not under it: a numeric zero or a negative -- both of which an agent can
    # send -- would otherwise match the outer branch, fail the inner one, and escape every clamp.
    if isinstance(requested, (int, float)) and not isinstance(requested, bool) and requested > 0:
        allowed = min(allowed, float(requested))
    else:
        allowed = min(allowed, float(DEFAULT_PROBE_TIMEOUT_SEC))
    return max(1, int(allowed))


@contextmanager
def _locked_json(path: Path):
    """Read one small JSON object under an exclusive lock and write it back."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8", errors="replace") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0)
            raw = handle.read().strip()
            try:
                state = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                state = {}
            if not isinstance(state, dict):
                state = {}
            yield state
            handle.seek(0)
            handle.truncate()
            handle.write(json.dumps(state, sort_keys=True))
            handle.flush()
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@dataclass
class ProbeBudget:
    """Track what one ROUND has already spent against its two ceilings."""

    path: Path | None = None
    attempts: int = 0
    seconds_used: float = 0.0
    # Attempts THIS process made, which is this assignment's ledger's own numbering.
    own_attempts: int = 0
    # Why the round's shared counters could not be reached.
    shared_error: str = ""

    def refresh(self) -> None:
        """Re-read what the round's other specialists have spent."""
        if self.path is None:
            return
        self._apply(attempts=0, seconds=0.0)

    def spend(self, *, attempts: int = 1, seconds: float = 0.0) -> None:
        """Charge one attempt, and its wall clock, to the round."""
        self._apply(attempts=attempts, seconds=seconds)

    def _apply(self, *, attempts: int, seconds: float) -> None:
        self.own_attempts += attempts
        if self.path is None:
            self.attempts += attempts
            self.seconds_used += seconds
            return
        try:
            with _locked_json(self.path) as state:
                state["attempts"] = int(state.get("attempts", 0) or 0) + attempts
                state["seconds_used"] = float(state.get("seconds_used", 0.0) or 0.0) + seconds
                self.attempts = state["attempts"]
                self.seconds_used = state["seconds_used"]
        except (OSError, ValueError) as error:
            # Two failure modes, one choice.
            self.attempts += attempts
            self.seconds_used += seconds
            if not self.shared_error:
                log.warning(
                    "the round's shared probe budget at %s cannot be reached "
                    "(%s: %s); no probe may measure against a budget that is "
                    "not being counted",
                    self.path,
                    type(error).__name__,
                    error,
                )
            self.shared_error = (
                f"the round's shared probe budget at {self.path} cannot be reached ({type(error).__name__}: {error})"
            )


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "probe_variant",
        "description": (
            "Time ONE named case at ONE point in the dispatch-constant space, "
            "by re-running the workspace driver in a scratch directory of its "
            "own. Nothing in the canonical workspace is edited. Use it to "
            "settle a question about a constant that you would otherwise have "
            "to argue. A result from this tool is exploratory, not an "
            "acceptance-gate result.\n"
            "Three numbers come back with every result: how many probes and "
            "how many seconds of wall clock are left of this ROUND's budget, "
            "both shared with the other specialists analysing it at the same "
            "time, and how many seconds are left of YOUR OWN session -- so a "
            "spent round budget and a session that is nearly over are things "
            "you can tell apart. No probe may run that would "
            "leave too little of your session to write the analysis, and a "
            "refused or unavailable probe costs one of the count just as a "
            "measured one does. The GPU is measured one run at a time, so a "
            "probe may spend part of its budget waiting for the device and be "
            "abandoned if it does not come free. When a result says the budget "
            "or the session clock is spent, stop probing and report the "
            "remaining questions as unmeasured."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "description": (
                        "Short name for the question this probe settles, cited "
                        "from the analysis, e.g. 'block-1024-vs-256'."
                    ),
                },
                "driver_script": {
                    "type": "string",
                    "description": (
                        "Benchmark driver to run, as a path inside the "
                        "canonical workspace. A path outside it is refused."
                    ),
                },
                "case_id": {
                    "type": "string",
                    "description": "Scored case to time, by name.",
                },
                "constants": {
                    "type": "object",
                    "description": (
                        "Declared dispatch constants to vary, upper-case name "
                        "-> value; they reach the driver as environment "
                        "variables named FORGE_SWEEP_<NAME>, which only a knob "
                        "instrumented for forge reads. Empty measures the "
                        "unmodified source as this probe's own reference."
                    ),
                },
                "prefix_constants": {
                    "type": "boolean",
                    "description": (
                        "Default true. Set false to export each name EXACTLY as "
                        "written, which is the only way to reach a knob the "
                        "source already reads under its own name (e.g. "
                        "GPTOSS_SWIGLU_MXFP4_BF16_BOUND). Such a knob does not "
                        "print forge's 'sweep_const:' echo, so the result comes "
                        "back marked unread and unconfirmed: measure a probe "
                        "with no constants in the same round and compare, or "
                        "the number says nothing. A name the measurement itself "
                        "runs on (PATH, HIP_VISIBLE_DEVICES, a cache directory) "
                        "is refused: those are not knobs of the kernel."
                    ),
                },
                "timeout_sec": {
                    "type": "integer",
                    "description": (
                        "Ceiling for this probe; clamped down to the remaining "
                        "wall-clock budget and to what your session can spare."
                    ),
                },
            },
            "required": ["label", "driver_script", "case_id"],
        },
    },
]


def resolve_probe_primitive() -> Any:
    """Return PR-1's single-case sweep primitive, or None if absent."""
    try:
        module = importlib.import_module(PRIMITIVE_MODULE)
    except ImportError:
        return None
    return getattr(module, PRIMITIVE_ATTR, None)


def probe_primitive_status() -> tuple[Any, str]:
    """Return the callable probe primitive, or None and why it is unusable."""
    try:
        primitive = resolve_probe_primitive()
    except Exception as error:  # noqa: BLE001 - an unimportable seam is reported
        return None, (
            f"the measurement primitive {PRIMITIVE_PATH} could not be imported: {type(error).__name__}: {error}"
        )
    if not callable(primitive):
        return None, (f"the measurement primitive {PRIMITIVE_PATH} is absent from this build")
    try:
        signature = inspect.signature(primitive)
    except (TypeError, ValueError) as error:
        return None, (f"the measurement primitive {PRIMITIVE_PATH} is not introspectable: {error}")
    accepted = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind in (parameter.KEYWORD_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    }
    if any(parameter.kind is parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return primitive, ""
    missing = [name for name in PRIMITIVE_KEYWORDS if name not in accepted]
    if missing:
        return None, (
            f"the measurement primitive {PRIMITIVE_PATH} does not accept "
            f"{', '.join(missing)}, so the probe cannot call it"
        )
    return primitive, ""


def _is_inside(child: Path, parent: Path) -> bool:
    """Whether ``child`` resolves to ``parent`` or below it."""
    return child == parent or parent in child.parents


def load_sandbox(environ: dict[str, str] | None = None) -> ProbeSandbox:
    """Read the sandbox this server was started for and validate its isolation."""
    env = os.environ if environ is None else environ
    scratch_raw = str(env.get(SCRATCH_ENV) or "").strip()
    workspace_raw = str(env.get(WORKSPACE_ENV) or "").strip()
    if not scratch_raw or not workspace_raw:
        raise ProbeSandboxError(f"{SCRATCH_ENV} and {WORKSPACE_ENV} must both be set")
    scratch_root = Path(scratch_raw).expanduser().resolve()
    workspace = Path(workspace_raw).expanduser().resolve()
    if _is_inside(scratch_root, workspace) or _is_inside(workspace, scratch_root):
        raise ProbeSandboxError(f"probe scratch root {scratch_root} overlaps the canonical tree {workspace}")
    if not scratch_root.is_dir():
        raise ProbeSandboxError(f"probe scratch root is not a directory: {scratch_root}")
    ledger_raw = str(env.get(LEDGER_ENV) or "").strip()
    ledger_path = Path(ledger_raw).expanduser().resolve() if ledger_raw else scratch_root / "probe_ledger.jsonl"
    if not _is_inside(ledger_path, scratch_root):
        raise ProbeSandboxError(f"probe ledger {ledger_path} lies outside the scratch root {scratch_root}")
    try:
        max_probes = int(env.get(MAX_PROBES_ENV, "0"))
        budget_sec = float(env.get(BUDGET_SEC_ENV, "0"))
    except ValueError as error:
        raise ProbeSandboxError(f"probe budget is not numeric: {error}") from error
    if max_probes <= 0 or budget_sec <= 0:
        raise ProbeSandboxError(f"{MAX_PROBES_ENV} and {BUDGET_SEC_ENV} must both be greater than zero")
    budget_raw = str(env.get(ROUND_BUDGET_ENV) or "").strip()
    device_raw = str(env.get(DEVICE_LOCK_ENV) or "").strip()
    deadline_raw = str(env.get(SESSION_DEADLINE_ENV) or "").strip()
    # An absent deadline is fail-open on purpose -- the configured probe budget still bounds every probe, and a
    # session run outside a round has no deadline to declare.
    session_deadline: float | None = None
    if deadline_raw:
        try:
            session_deadline = float(deadline_raw)
        except ValueError as error:
            raise ProbeSandboxError(f"{SESSION_DEADLINE_ENV} is not a Unix timestamp: {error}") from error
        if not math.isfinite(session_deadline) or session_deadline <= 0:
            raise ProbeSandboxError(f"{SESSION_DEADLINE_ENV} is not a Unix timestamp: {deadline_raw!r}")
    return ProbeSandbox(
        scratch_root=scratch_root,
        workspace=workspace,
        ledger_path=ledger_path,
        max_probes=max_probes,
        budget_sec=budget_sec,
        budget_path=(Path(budget_raw).expanduser().resolve() if budget_raw else None),
        device_lock=(Path(device_raw).expanduser().resolve() if device_raw else None),
        session_deadline=session_deadline,
    )


def _append_line(path: Path, record: dict[str, Any]) -> None:
    """Append one attempt to a ledger, unless that ledger is already full."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = 0
    if path.exists():
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            existing = sum(1 for line in handle if line.strip())
    if existing > MAX_LEDGER_RECORDS:
        return
    if existing == MAX_LEDGER_RECORDS:
        record = {
            "status": REFUSED,
            "label": "ledger-full",
            "detail": (
                f"this ledger reached its cap of {MAX_LEDGER_RECORDS} attempts; "
                "every later attempt is dropped and unrecorded"
            ),
        }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str, sort_keys=True) + "\n")


def append_ledger(sandbox: ProbeSandbox, record: dict[str, Any]) -> None:
    """Append one attempt to the ledger the parent reads after the session."""
    _append_line(sandbox.ledger_path, record)


def _try_device_lock(path: Path):
    """Take the device sentinel without waiting, or return None."""
    try:
        handle = path.open("r+", encoding="utf-8")
    except OSError:
        return None
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


async def acquire_device_lock(path: Path, *, timeout_sec: float):
    """Hold the campaign's device sentinel, or give up before the wait costs more."""
    deadline = monotonic_clock() + max(0.0, timeout_sec)
    while True:
        handle = await asyncio.to_thread(_try_device_lock, path)
        if handle is not None:
            return handle
        left = deadline - monotonic_clock()
        if left <= 0:
            return None
        await asyncio.sleep(min(DEVICE_LOCK_POLL_SEC, left))


def release_device_lock(handle) -> None:
    """Drop the device sentinel."""
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def _detail(text: Any) -> str:
    return str(text or "")[:MAX_DETAIL_CHARS]


async def probe_variant(
    arguments: dict[str, Any],
    *,
    sandbox: ProbeSandbox,
    budget: ProbeBudget,
) -> dict[str, Any]:
    """Run one bounded probe and record it, whatever the outcome."""
    label = str(arguments.get("label") or "").strip()
    case_id = str(arguments.get("case_id") or "").strip()
    if not label or not case_id:
        raise InvalidParamsError("label and case_id must be non-empty strings")
    constants = arguments.get("constants")
    if constants is None:
        constants = {}
    if not isinstance(constants, dict):
        raise InvalidParamsError("constants must be an object")
    prefix_constants = arguments.get("prefix_constants", True)
    if not isinstance(prefix_constants, bool):
        raise InvalidParamsError("prefix_constants must be a boolean")

    # What the round's other specialists have spent since the last call.
    budget.refresh()
    session_remaining = sandbox.session_remaining_sec()
    configured_remaining = sandbox.budget_sec - budget.seconds_used
    budget_remaining = probe_budget_sec(
        configured_remaining=configured_remaining,
        session_remaining=session_remaining,
    )
    base = {
        # This ledger's own numbering: ``budget.attempts`` is the round's and skips what a sibling specialist spent.
        "probe_index": budget.own_attempts + 1,
        "label": label,
        "case_id": case_id,
        "constants": constants,
    }

    if budget.shared_error:
        return _record(
            sandbox,
            budget,
            {
                **base,
                "status": UNAVAILABLE,
                "detail": f"{budget.shared_error}; nothing was measured",
                "duration_sec": 0.0,
            },
        )

    if budget.attempts >= sandbox.max_probes or configured_remaining <= 0:
        exhausted = (
            f"probe count budget of {sandbox.max_probes} is spent"
            if budget.attempts >= sandbox.max_probes
            else f"wall-clock budget of {sandbox.budget_sec:.0f}s is spent"
        )
        return _record(
            sandbox,
            budget,
            {
                **base,
                "status": BUDGET_EXHAUSTED,
                "detail": (f"{exhausted} for this round; this question stays unmeasured and must be reported as such"),
                "duration_sec": 0.0,
            },
        )

    # Gated on whether there is time to PRODUCE the analysis, not on a reserve of the probe's own: a session killed
    # mid-probe returns nothing at all, and the round treats that as infrastructure failure rather than as a thin
    # answer.
    if session_remaining - ANALYSIS_RESERVE_SEC <= 0 or budget_remaining <= 0:
        return _record(
            sandbox,
            budget,
            {
                **base,
                "status": BUDGET_EXHAUSTED,
                "detail": (
                    f"only {max(0.0, session_remaining):.0f}s of your session is "
                    f"left and {ANALYSIS_RESERVE_SEC:.0f}s of it is reserved for "
                    "writing the analysis; no further probe can run, so stop "
                    "probing and report the remaining questions as unmeasured"
                ),
                "duration_sec": 0.0,
            },
        )

    primitive, unusable = probe_primitive_status()
    if primitive is None:
        return _record(
            sandbox,
            budget,
            {
                **base,
                "status": UNAVAILABLE,
                "detail": f"{unusable}; nothing was measured",
                "duration_sec": 0.0,
            },
        )

    driver_raw = str(arguments.get("driver_script") or "").strip()
    if not driver_raw:
        raise InvalidParamsError("driver_script must be a non-empty string")
    driver = Path(driver_raw)
    if not driver.is_absolute():
        driver = sandbox.workspace / driver
    driver = driver.expanduser().resolve()
    if not _is_inside(driver, sandbox.workspace) or not driver.is_file():
        return _record(
            sandbox,
            budget,
            {
                **base,
                "status": REFUSED,
                "driver_script": str(driver),
                "detail": (
                    f"{driver} is not a file inside the canonical workspace {sandbox.workspace}; nothing was measured"
                ),
                "duration_sec": 0.0,
            },
        )
    base["driver_script"] = str(driver)

    if sandbox.device_lock is None or not sandbox.device_lock.is_file():
        missing = (
            f"{DEVICE_LOCK_ENV} names no device sentinel"
            if sandbox.device_lock is None
            else f"the device sentinel {sandbox.device_lock} does not exist"
        )
        return _record(
            sandbox,
            budget,
            {
                **base,
                "status": UNAVAILABLE,
                "detail": (
                    f"{missing}, so this probe would time the GPU while something else uses it; nothing was measured"
                ),
                "duration_sec": 0.0,
            },
        )

    started = monotonic_clock()
    # The wait is bounded by the probe's own budget, and what it costs is charged to the budget: a specialist that
    # blocked here until the device came free would spend its session doing nothing.
    handle = await acquire_device_lock(sandbox.device_lock, timeout_sec=budget_remaining)
    waited = monotonic_clock() - started
    if handle is None:
        return _record(
            sandbox,
            budget,
            {
                **base,
                "status": DEVICE_BUSY,
                "detail": (
                    f"the device was still held after {waited:.0f}s, which is "
                    "this probe's whole budget; nothing was measured"
                ),
                "duration_sec": waited,
            },
        )

    # The gate again, on what the wait left.
    budget_remaining -= waited
    session_remaining -= waited
    if session_remaining - ANALYSIS_RESERVE_SEC <= 0 or budget_remaining <= 0:
        release_device_lock(handle)
        return _record(
            sandbox,
            budget,
            {
                **base,
                "status": BUDGET_EXHAUSTED,
                "detail": (
                    f"{waited:.0f}s went on waiting for the device, which "
                    f"leaves {max(0.0, session_remaining):.0f}s of your session "
                    f"and {max(0.0, budget_remaining):.0f}s of this round's "
                    "budget; nothing was measured, so report this question as "
                    "unmeasured"
                ),
                "duration_sec": waited,
            },
        )

    timeout_sec = probe_timeout_sec(
        budget_remaining=budget_remaining,
        session_remaining=session_remaining,
        requested=arguments.get("timeout_sec"),
    )
    try:
        try:
            result = await asyncio.wait_for(
                primitive(
                    driver_script=str(driver),
                    case_id=case_id,
                    constants=dict(constants),
                    timeout_sec=timeout_sec,
                    prefix_constants=prefix_constants,
                ),
                timeout=timeout_sec + PROBE_TOOL_GRACE_SEC,
            )
        except asyncio.TimeoutError:
            return _record(
                sandbox,
                budget,
                {
                    **base,
                    "status": FAILED,
                    "detail": f"probe exceeded its {timeout_sec}s ceiling",
                    "duration_sec": monotonic_clock() - started,
                },
            )
        except Exception as error:  # noqa: BLE001 - a broken probe is a reported probe
            return _record(
                sandbox,
                budget,
                {
                    **base,
                    "status": FAILED,
                    "detail": f"{type(error).__name__}: {error}",
                    "duration_sec": monotonic_clock() - started,
                },
            )
    finally:
        release_device_lock(handle)

    payload = result if isinstance(result, dict) else {}
    # The primitive omits every timing field on failure rather than reporting a zero, so a result carrying no
    # ``case_ms`` is a failure whatever else it says.
    case_ms = payload.get("case_ms")
    succeeded = bool(payload.get("success")) and isinstance(case_ms, (int, float))
    record = {
        **base,
        "status": MEASURED if succeeded else FAILED,
        "detail": _detail(payload.get("message") or f"the primitive returned no measurement: {payload or result!r}"),
        "duration_sec": monotonic_clock() - started,
    }
    if succeeded:
        record["case_ms"] = case_ms
        record["kind"] = payload.get("kind", "")
        # ``narrowed`` false means other cases were timed too, so the cost was not one case and the reported spread is
        # not this case's; ``case_selection`` says whether the flag is what narrowed it.
        record["narrowed"] = bool(payload.get("narrowed", True))
        record["case_selection"] = str(payload.get("case_selection", ""))
        # Which overrides the source was seen to read.
        consumption = payload.get("override_consumption")
        if isinstance(consumption, dict) and consumption:
            record["override_consumption"] = consumption
    return _record(sandbox, budget, record)


def _record(
    sandbox: ProbeSandbox,
    budget: ProbeBudget,
    record: dict[str, Any],
) -> dict[str, Any]:
    """Charge one attempt to the round, persist it, and return it."""
    budget.spend(attempts=1, seconds=float(record.get("duration_sec") or 0.0))
    record = {
        **record,
        "probes_remaining": max(0, sandbox.max_probes - budget.attempts),
        "seconds_remaining": probe_budget_sec(
            configured_remaining=sandbox.budget_sec - budget.seconds_used,
            session_remaining=sandbox.session_remaining_sec(),
        ),
        # The third number the tool description promises.
        "session_seconds_remaining": max(0.0, sandbox.session_remaining_sec()),
    }
    append_ledger(sandbox, record)
    return {
        **record,
        "evidence": "exploratory scratch measurement, not an acceptance-gate result",
    }


def refuse_to_ledger(
    record: dict[str, Any],
    environ: dict[str, str] | None = None,
) -> str:
    """Record a refusal the sandbox itself could not, and say if that failed."""
    env = os.environ if environ is None else environ
    raw = str(env.get(LEDGER_ENV) or "").strip()
    if not raw:
        return f"{LEDGER_ENV} is unset, so this refusal reaches no ledger"
    path = Path(raw).expanduser()
    try:
        _append_line(path, record)
    except OSError as error:
        return f"this refusal could not be written to {path}: {error}"
    return ""


class ProbeServer:
    """Serve the probe tool for one specialist session over stdio."""

    def __init__(self) -> None:
        self._sandbox: ProbeSandbox | None = None
        self._sandbox_error = ""
        self._budget = ProbeBudget()
        self._refusals = 0

    def _resolve_sandbox(self) -> ProbeSandbox | None:
        if self._sandbox is None and not self._sandbox_error:
            try:
                self._sandbox = load_sandbox()
            except ProbeSandboxError as error:
                self._sandbox_error = str(error)
        return self._sandbox

    async def handle_tool_call(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Invoke the probe tool and wrap its record as MCP content."""
        if name != "probe_variant":
            raise InvalidParamsError(f"unknown tool: {name}")
        sandbox = self._resolve_sandbox()
        if sandbox is None:
            # A refusal costs a count of its own: there is no sandbox to charge it to, and a free refusal is one the
            # session can repeat until it ends.
            self._refusals += 1
            if self._refusals > MAX_LEDGER_RECORDS:
                # One last line first, on the call that crosses the cap: a ledger that stops without saying it is full
                # cannot be told from a session that simply made few calls, which is the whole point of the marker.
                if self._refusals == MAX_LEDGER_RECORDS + 1:
                    refuse_to_ledger(
                        {
                            "status": REFUSED,
                            "label": str(arguments.get("label") or ""),
                            "detail": (f"probe sandbox unusable: {self._sandbox_error}"),
                        }
                    )
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "status": REFUSED,
                                    "detail": (
                                        f"this session made {self._refusals} "
                                        "refused probe calls; the probe is "
                                        "unusable here, stop calling it"
                                    ),
                                }
                            ),
                        }
                    ]
                }
            result = {
                "probe_index": self._refusals,
                "status": REFUSED,
                "label": str(arguments.get("label") or ""),
                "case_id": str(arguments.get("case_id") or ""),
                "detail": f"probe sandbox unusable: {self._sandbox_error}",
            }
            unrecorded = refuse_to_ledger(result)
            if unrecorded:
                result["detail"] = f"{result['detail']}; {unrecorded}"
        else:
            result = await probe_variant(
                arguments,
                sandbox=sandbox,
                budget=self._budget,
            )
        return {"content": [{"type": "text", "text": json.dumps(result, default=str)}]}

    async def dispatch(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Dispatch one supported MCP request and return its result object."""
        return await stdio_transport.dispatch_envelope(
            method,
            params,
            server_name=SERVER_NAME,
            tool_definitions=TOOL_DEFINITIONS,
            handle_tool_call=self.handle_tool_call,
        )


async def _serve() -> None:
    """Serve JSON-RPC requests until stdin closes or an exit notification arrives."""
    await stdio_transport.serve(ProbeServer().dispatch)


def main() -> None:
    """Run the specialist probe MCP server over standard input and output."""
    asyncio.run(_serve())


if __name__ == "__main__":
    main()
