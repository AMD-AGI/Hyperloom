# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Stdio MCP server that lets a read-only specialist measure one variant.

The specialist a round is planned from cannot write to the canonical tree, so a
question about a constant can only be argued. This server answers it instead of
a shell: one tool, whose every call times one declared case at one point in the
dispatch-constant space, and appends the attempt -- refusals and failures
included -- to a ledger under a scratch root the parent created outside the
canonical tree and reads back after the session.

Three things bound a call. The round's probe count and wall-clock budget, which
the round's concurrently running specialists share through one small locked file
rather than each getting a copy. The specialist's own session clock, which no
probe may eat into far enough to leave the analysis unwritten. And the
campaign's device sentinel -- the same file a fan-out lane's driver flocks --
because the GPU times one thing at a time and a probe that waits for it is
spending a session that is running out.

The measurement itself belongs to PR-1's ``sweep_case``, resolved by name at
call time and checked against the keywords this server calls it with. When that
primitive is absent, or present with a signature the probe cannot call, the tool
reports the seam; this server never grows a measurement path of its own.
"""

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
import sys
import time
from typing import Any

log = logging.getLogger(__name__)

SERVER_NAME = "kernelforge-specialist-probe"
# Agents see these as mcp__specialist_probe__<name>.
TOOL_NAMES = ("probe_variant",)

SCRATCH_ENV = "FORGE_PROBE_SCRATCH"
WORKSPACE_ENV = "FORGE_PROBE_WORKSPACE"
LEDGER_ENV = "FORGE_PROBE_LEDGER"
MAX_PROBES_ENV = "FORGE_PROBE_MAX"
BUDGET_SEC_ENV = "FORGE_PROBE_BUDGET_SEC"
# The round's shared counters; see ``ProbeBudget``. Absent, the budget is this
# process's own.
ROUND_BUDGET_ENV = "FORGE_PROBE_ROUND_BUDGET"
# The campaign-wide sentinel a run must flock before it touches the GPU, from
# ``kernelforge.loop.fanout.campaign_device_lock_path``. Absent, the probe
# refuses to measure rather than timing against whatever else is running.
DEVICE_LOCK_ENV = "FORGE_PROBE_DEVICE_LOCK"
# Unix timestamp at which the specialist session this server serves is killed.
# Absent, only the configured probe budget bounds a probe.
SESSION_DEADLINE_ENV = "FORGE_PROBE_SESSION_DEADLINE"

PRIMITIVE_MODULE = "kernelforge.mcp_server.tools.bench"
PRIMITIVE_ATTR = "sweep_case"
PRIMITIVE_PATH = f"{PRIMITIVE_MODULE}.{PRIMITIVE_ATTR}"
# The keywords this server calls the primitive with. Checked rather than
# assumed: a primitive that landed under this name with a different signature
# would otherwise fail once per probe, as a TypeError inside a compile report.
PRIMITIVE_KEYWORDS = (
    "driver_script",
    "case_id",
    "constants",
    "timeout_sec",
    "prefix_constants",
)

# Ledger statuses. The parent renders every one of them: a probe that was
# refused, or one whose primitive is missing, must not read like a probe nobody
# ran.
MEASURED = "measured"
FAILED = "failed"
BUDGET_EXHAUSTED = "budget_exhausted"
UNAVAILABLE = "unavailable"
REFUSED = "refused"
DEVICE_BUSY = "device_busy"

DEFAULT_PROBE_TIMEOUT_SEC = 300
# Seconds the server waits past a probe's own ceiling before abandoning it, so
# a primitive that is a moment late is reported as late rather than lost. The
# MCP client's ``tool_timeout_sec`` is given the same grace: a client that timed
# out first would kill the call before ``_record`` appended anything, and the
# ledger is the only channel this server has back to the parent.
PROBE_TOOL_GRACE_SEC = 5
# One probe's report; a compile log can be arbitrarily long.
MAX_DETAIL_CHARS = 2_000

# Seconds of the specialist's session that no probe may take. This is about
# whether there is time to PRODUCE the analysis, not about a reserve of the
# probe's own: a session killed mid-probe returns no analysis at all, and a
# round whose specialists all probed themselves to death is a dead round rather
# than a degraded one. Same reasoning as ``lessons.SUMMARY_MIN_SECONDS``.
ANALYSIS_RESERVE_SEC = 120.0
# The most of what is left of a session one probe budget may claim. The probe
# is there to settle a question the analysis turns on, so it may take a large
# share -- but never the share that leaves reading and writing no room.
SESSION_PROBE_FRACTION = 0.5
# How often a probe waiting for the device retries the sentinel.
DEVICE_LOCK_POLL_SEC = 1.0

# Attempts one ledger holds. ``max_probes`` bounds the probes; nothing else
# bounds a session that keeps calling a tool which refuses it, and the parent
# reads this file back in full. See ``_append_line``.
MAX_LEDGER_RECORDS = 200


class InvalidParamsError(ValueError):
    """Invalid agent-supplied MCP tool arguments."""


class ProbeSandboxError(RuntimeError):
    """The scratch sandbox this server was configured with is unusable."""


def wall_clock() -> float:
    """Now, on the clock the session deadline is expressed in.

    A function rather than a call site so a test can drive the budget
    arithmetic without sleeping.
    """
    return time.time()


def monotonic_clock() -> float:
    """Elapsed-time clock for the device wait and a probe's own duration.

    A function for the same reason ``wall_clock`` is: the gate that re-runs
    after a device wait is arithmetic, and a test must be able to drive it
    without holding the device for two minutes.
    """
    return time.monotonic()


@dataclass(frozen=True)
class ProbeSandbox:
    """Hold the scratch root, the budgets and the ledger for one specialist."""

    scratch_root: Path
    workspace: Path
    ledger_path: Path
    max_probes: int
    budget_sec: float
    # Shared counters for the round this session belongs to; None keeps them
    # in this process.
    budget_path: Path | None = None
    # The campaign's device sentinel. None means no probe may measure: timing
    # against whatever else holds the GPU produces a number, not a measurement.
    device_lock: Path | None = None
    # When the specialist session is killed. None means unbounded, which only
    # happens when the parent did not say.
    session_deadline: float | None = None

    def session_remaining_sec(self) -> float:
        """Seconds left in the specialist session, or infinity if unbounded."""
        if self.session_deadline is None:
            return math.inf
        return self.session_deadline - wall_clock()


def probe_budget_sec(*, configured_remaining: float, session_remaining: float) -> float:
    """What is really left to spend on probing, given the session's own clock.

    The configured budget is a ceiling, not an entitlement: a specialist that
    spent it in full could be killed by its session timeout before writing
    anything, and a round in which every specialist did that raises rather than
    degrades. So the budget is capped at a fraction of what is left of the
    session, and it shrinks as the session does.
    """
    return max(0.0, min(configured_remaining, session_remaining * SESSION_PROBE_FRACTION))


def probe_timeout_sec(
    *,
    budget_remaining: float,
    session_remaining: float,
    requested: Any = None,
) -> int:
    """Ceiling for one probe: the budget, the session, and what was asked for.

    ``int()`` on a fractional budget would truncate to zero, which some backends
    read as "time out immediately", so the result is never below one second --
    a probe that cannot fit is refused by the caller's gate rather than started
    with an impossible ceiling.
    """
    allowed = min(budget_remaining, session_remaining - ANALYSIS_RESERVE_SEC)
    # ``requested > 0`` belongs in this test, not under it: a numeric zero or a
    # negative -- both of which an agent can send -- would otherwise match the
    # outer branch, fail the inner one, and escape every clamp.
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
    """Track what one ROUND has already spent against its two ceilings.

    The unit of account is the round, not one assignment. A round's specialists
    run concurrently behind one server process each, and how many assignments a
    round has is chosen by a model at runtime -- so a per-assignment budget
    bounds nothing an operator can predict. The counters therefore live in one
    small JSON file under the round's scratch root, read and written under an
    ``fcntl.flock``; with no ``path`` they are this process's own, which is what
    a session run outside a round gets.

    ``attempts`` counts every attempt that reached the ledger, refusals and
    unavailable primitives included -- see ``_record``. An outcome that cost
    nothing could be asked for again for the whole session.
    """

    path: Path | None = None
    attempts: int = 0
    seconds_used: float = 0.0
    # Attempts THIS process made, which is this assignment's ledger's own
    # numbering. ``attempts`` is the round's and skips whatever a sibling
    # spent, so a single ledger numbered with it read 1, 3, 4.
    own_attempts: int = 0
    # Why the round's shared counters could not be reached. Non-empty means no
    # probe may measure; see ``_apply``.
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
            # Two failure modes, one choice. Falling back to this process's own
            # counters would give every specialist of the round a full
            # ``max_probes`` and ``budget_sec`` of its own, so N concurrent
            # specialists would overspend N-fold with nothing in the log. So
            # this file makes the same call it makes for a missing device
            # sentinel: report the probe unavailable rather than measure under
            # a budget nobody is counting. The local counters still move, which
            # is what stops a session repeating a free attempt.
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
    """Return PR-1's single-case sweep primitive, or None if absent.

    Anything the primitive's module raises at import time other than a missing
    module propagates to ``probe_primitive_status``, which reports it as an
    unavailable seam; a server that crashed on it would take the specialist
    session with it.
    """
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
    # An absent deadline is fail-open on purpose -- the configured probe budget
    # still bounds every probe, and a session run outside a round has no
    # deadline to declare. A deadline that is PRESENT and nonsense is not: nan
    # made ``min(600, nan)`` return 600 and the session constraint disappear,
    # and a zero or past value refused every probe for the rest of the session.
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
    """Append one attempt to a ledger, unless that ledger is already full.

    ``max_probes`` bounds the probes but not the calls: a tool that refuses
    every call refuses as many as the session makes, and the parent reads this
    file back in full. At the cap the record is dropped and one final line says
    so, so a truncated ledger cannot be misread as a short one.
    """
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
    """Take the device sentinel without waiting, or return None.

    Opened without creating it. A sentinel this process made is a fresh private
    file that serializes nothing, so a misconfigured ``FORGE_PROBE_DEVICE_LOCK``
    would have produced a number labelled ``measured`` while a lane was on the
    device. The caller checks the path exists and refuses when it does not.
    """
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
    """Hold the campaign's device sentinel, or give up before the wait costs more.

    The same ``fcntl.flock`` on the same file a fan-out lane's serialized driver
    takes (``fanout.campaign_device_lock_path``), so a probe queues behind a
    lane and a lane behind a probe. Polled rather than blocked on, because the
    waiting session's own clock keeps running: a wait that outlasts the probe's
    budget is a probe that must be abandoned, not one that blocks.
    """
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
    """Run one bounded probe and record it, whatever the outcome.

    Three clocks bound a probe and every one of them can refuse it: the round's
    probe count, the round's wall-clock budget, and what is left of THIS
    specialist's session once the time to write the analysis is set aside.
    Waiting for the device counts against the second.
    """
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
        # This ledger's own numbering: ``budget.attempts`` is the round's and
        # skips what a sibling specialist spent.
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

    # Gated on whether there is time to PRODUCE the analysis, not on a reserve
    # of the probe's own: a session killed mid-probe returns nothing at all, and
    # the round treats that as infrastructure failure rather than as a thin
    # answer. Said in the refusal so the agent stops asking.
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
    # The wait is bounded by the probe's own budget, and what it costs is
    # charged to the budget: a specialist that blocked here until the device
    # came free would spend its session doing nothing.
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

    # The gate again, on what the wait left. Recomputing only the ceiling let a
    # probe start under the ``max(1, ...)`` clamp it could not possibly meet,
    # and the ledger then read "the probe was too slow" for a session that had
    # run out -- after the wait had held the device the whole time.
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
    # The primitive omits every timing field on failure rather than reporting a
    # zero, so a result carrying no ``case_ms`` is a failure whatever else it
    # says.
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
        # ``narrowed`` false means other cases were timed too, so the cost was
        # not one case and the reported spread is not this case's;
        # ``case_selection`` says whether the flag is what narrowed it.
        record["narrowed"] = bool(payload.get("narrowed", True))
        record["case_selection"] = str(payload.get("case_selection", ""))
        # Which overrides the source was seen to read. A verbatim-named knob
        # that echoed nothing leaves the number unconfirmed, and a ledger entry
        # that dropped this would read exactly like a confirmed one.
        consumption = payload.get("override_consumption")
        if isinstance(consumption, dict) and consumption:
            record["override_consumption"] = consumption
    return _record(sandbox, budget, record)


def _record(
    sandbox: ProbeSandbox,
    budget: ProbeBudget,
    record: dict[str, Any],
) -> dict[str, Any]:
    """Charge one attempt to the round, persist it, and return it.

    Every recorded attempt is charged, and every outcome is recorded: a refused
    or unavailable probe that cost nothing could be asked for again for the
    whole session, and the ledger it appends to is unbounded in nothing else.
    """
    budget.spend(attempts=1, seconds=float(record.get("duration_sec") or 0.0))
    record = {
        **record,
        "probes_remaining": max(0, sandbox.max_probes - budget.attempts),
        "seconds_remaining": probe_budget_sec(
            configured_remaining=sandbox.budget_sec - budget.seconds_used,
            session_remaining=sandbox.session_remaining_sec(),
        ),
        # The third number the tool description promises. ``seconds_remaining``
        # already folds the session in, so without this one the model cannot
        # tell a spent round budget from a session that is nearly over.
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
    """Record a refusal the sandbox itself could not, and say if that failed.

    A refusal is the one outcome that arrives with no validated sandbox to write
    it to, and the parent reads an empty ledger as "the probe was offered and
    never called". So the refusal goes to the raw ``LEDGER_ENV`` path, ahead of
    any validation; the returned string is empty when it landed there.
    """
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
            # A refusal costs a count of its own: there is no sandbox to charge
            # it to, and a free refusal is one the session can repeat until it
            # ends. Past the cap the tool still answers, but records nothing --
            # ``_append_line`` has already said so on the ledger's last line.
            self._refusals += 1
            if self._refusals > MAX_LEDGER_RECORDS:
                # One last line first, on the call that crosses the cap: a
                # ledger that stops without saying it is full cannot be told
                # from a session that simply made few calls, which is the whole
                # point of the marker. ``_append_line`` substitutes it.
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
        if method == "initialize":
            return {
                "protocolVersion": params.get("protocolVersion") or "2024-11-05",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": SERVER_NAME, "version": "0.1.0"},
            }
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": TOOL_DEFINITIONS}
        if method == "tools/call":
            arguments = params.get("arguments")
            if arguments is None:
                arguments = {}
            if not isinstance(arguments, dict):
                raise InvalidParamsError("tools/call arguments must be an object")
            return await self.handle_tool_call(str(params.get("name") or ""), arguments)
        if method in {"resources/list", "prompts/list"}:
            return {"resources": []} if method == "resources/list" else {"prompts": []}
        if method in {"logging/setLevel", "shutdown"}:
            return {}
        raise NotImplementedError(f"unsupported MCP method: {method}")


def _write_message(payload: dict[str, Any]) -> None:
    """Write one newline-delimited JSON-RPC message to stdout."""
    sys.stdout.write(json.dumps(payload, separators=(",", ":"), default=str) + "\n")
    sys.stdout.flush()


def _write_error(request_id: Any, code: int, message: str) -> None:
    """Write one JSON-RPC error response."""
    _write_message(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }
    )


async def _serve() -> None:
    """Serve JSON-RPC requests until stdin closes or an exit notification arrives."""
    server = ProbeServer()
    while True:
        raw = await asyncio.to_thread(sys.stdin.buffer.readline)
        if not raw:
            return
        try:
            message = json.loads(raw.decode())
        except (UnicodeDecodeError, json.JSONDecodeError):
            _write_error(None, -32700, "Parse error")
            continue
        if not isinstance(message, dict):
            _write_error(None, -32600, "Invalid Request")
            continue
        method = str(message.get("method") or "")
        request_id = message.get("id")
        if method == "exit":
            return
        if request_id is None:
            continue
        params = message.get("params")
        if params is None:
            params = {}
        if not isinstance(params, dict):
            _write_error(request_id, -32602, "params must be an object")
            continue
        try:
            result = await server.dispatch(method, params)
            _write_message({"jsonrpc": "2.0", "id": request_id, "result": result})
        except NotImplementedError as exc:
            _write_error(request_id, -32601, str(exc))
        except InvalidParamsError as exc:
            _write_error(request_id, -32602, str(exc))
        except Exception as exc:  # noqa: BLE001 - convert failures to JSON-RPC
            _write_error(request_id, -32603, f"{type(exc).__name__}: {exc}")


def main() -> None:
    """Run the specialist probe MCP server over standard input and output."""
    asyncio.run(_serve())


if __name__ == "__main__":
    main()
