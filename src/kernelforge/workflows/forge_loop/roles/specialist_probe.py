# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Let a read-only specialist measure one variant inside its sandbox, budget and session."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from functools import partial
import importlib
import inspect
import json
import logging
import math
import os
from pathlib import Path
import time
from typing import Any, Awaitable, Callable

from kernelforge.evaluation.benchmark.runner import SWEEP_BENCH_ITERS, SWEEP_WARMUP_ITERS, BenchRun
from kernelforge.utils.file_lock import locked_json_state, release_lock, try_lock_exclusive

log = logging.getLogger(__name__)

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
# ``kernelforge.workflows.forge_loop.iteration.fanout.campaign_device_lock_path``.
DEVICE_LOCK_ENV = "FORGE_PROBE_DEVICE_LOCK"
# Unix timestamp at which the specialist session this probe serves is killed.
SESSION_DEADLINE_ENV = "FORGE_PROBE_SESSION_DEADLINE"

PRIMITIVE_MODULE = "kernelforge.evaluation.benchmark.runner"
PRIMITIVE_ATTR = "sweep_case"
PRIMITIVE_PATH = f"{PRIMITIVE_MODULE}.{PRIMITIVE_ATTR}"
# The keywords the probe calls the primitive with.
PRIMITIVE_KEYWORDS = (
    "bench",
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
# Seconds the caller waits past a probe's own ceiling before abandoning it, so a primitive that is a moment late is
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
    """The scratch sandbox this probe was configured with is unusable."""


class ProbeRequestError(ValueError):
    """Invalid agent-supplied probe arguments."""


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
            with locked_json_state(self.path) as state:
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
    except Exception as error:  # isolation boundary: the optional probe primitive; callers report it unusable
        log.warning("probe primitive %s could not be imported: %s", PRIMITIVE_PATH, error, exc_info=True)
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
    """Read the sandbox this probe was started for and validate its isolation."""
    env = os.environ if environ is None else environ
    scratch_raw = env.get(SCRATCH_ENV, "").strip()
    workspace_raw = env.get(WORKSPACE_ENV, "").strip()
    if not scratch_raw or not workspace_raw:
        raise ProbeSandboxError(f"{SCRATCH_ENV} and {WORKSPACE_ENV} must both be set")
    scratch_root = Path(scratch_raw).expanduser().resolve()
    workspace = Path(workspace_raw).expanduser().resolve()
    if _is_inside(scratch_root, workspace) or _is_inside(workspace, scratch_root):
        raise ProbeSandboxError(f"probe scratch root {scratch_root} overlaps the canonical tree {workspace}")
    if not scratch_root.is_dir():
        raise ProbeSandboxError(f"probe scratch root is not a directory: {scratch_root}")
    ledger_raw = env.get(LEDGER_ENV, "").strip()
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
    budget_raw = env.get(ROUND_BUDGET_ENV, "").strip()
    device_raw = env.get(DEVICE_LOCK_ENV, "").strip()
    deadline_raw = env.get(SESSION_DEADLINE_ENV, "").strip()
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


async def acquire_device_lock(path: Path, *, timeout_sec: float):
    """Hold the campaign's device sentinel, or give up before the wait costs more."""
    deadline = monotonic_clock() + max(0.0, timeout_sec)
    while True:
        handle = await asyncio.to_thread(try_lock_exclusive, path)
        if handle is not None:
            return handle
        left = deadline - monotonic_clock()
        if left <= 0:
            return None
        await asyncio.sleep(min(DEVICE_LOCK_POLL_SEC, left))


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
        raise ProbeRequestError("label and case_id must be non-empty strings")
    constants = arguments.get("constants")
    if constants is not None and not isinstance(constants, dict):
        raise ProbeRequestError("constants must be an object")
    constants = constants or {}
    prefix_constants = arguments.get("prefix_constants", True)
    if not isinstance(prefix_constants, bool):
        raise ProbeRequestError("prefix_constants must be a boolean")

    # What the round's other specialists have spent since the last call.
    budget.refresh()
    session_remaining = sandbox.session_remaining_sec()
    configured_remaining = sandbox.budget_sec - budget.seconds_used
    budget_remaining = probe_budget_sec(configured_remaining=configured_remaining, session_remaining=session_remaining)
    # This ledger's own numbering: ``budget.attempts`` is the round's and skips what a sibling specialist spent.
    base = {"probe_index": budget.own_attempts + 1, "label": label, "case_id": case_id, "constants": constants}
    record = partial(_record, sandbox, budget, base)

    if budget.shared_error:
        return record({"status": UNAVAILABLE, "detail": f"{budget.shared_error}; nothing was measured"})
    if budget.attempts >= sandbox.max_probes or configured_remaining <= 0:
        exhausted = (
            f"probe count budget of {sandbox.max_probes} is spent"
            if budget.attempts >= sandbox.max_probes
            else f"wall-clock budget of {sandbox.budget_sec:.0f}s is spent"
        )
        detail = f"{exhausted} for this round; this question stays unmeasured and must be reported as such"
        return record({"status": BUDGET_EXHAUSTED, "detail": detail})
    # Gated on whether there is time to PRODUCE the analysis, not on a reserve of the probe's own: a session killed
    # mid-probe returns nothing at all, and the round treats that as infrastructure failure rather than as a thin
    # answer.
    if session_remaining - ANALYSIS_RESERVE_SEC <= 0 or budget_remaining <= 0:
        detail = (
            f"only {max(0.0, session_remaining):.0f}s of your session is "
            f"left and {ANALYSIS_RESERVE_SEC:.0f}s of it is reserved for "
            "writing the analysis; no further probe can run, so stop "
            "probing and report the remaining questions as unmeasured"
        )
        return record({"status": BUDGET_EXHAUSTED, "detail": detail})
    primitive, unusable = probe_primitive_status()
    if primitive is None:
        return record({"status": UNAVAILABLE, "detail": f"{unusable}; nothing was measured"})

    driver_raw = str(arguments.get("driver_script") or "").strip()
    if not driver_raw:
        raise ProbeRequestError("driver_script must be a non-empty string")
    # Joining an absolute path onto the workspace yields that path, so a relative one alone is resolved against it.
    driver = (sandbox.workspace / driver_raw).expanduser().resolve()
    if not _is_inside(driver, sandbox.workspace) or not driver.is_file():
        detail = f"{driver} is not a file inside the canonical workspace {sandbox.workspace}; nothing was measured"
        return record({"status": REFUSED, "driver_script": str(driver), "detail": detail})
    base["driver_script"] = str(driver)
    run = partial(
        primitive,
        bench=BenchRun(str(driver), warmup_iters=SWEEP_WARMUP_ITERS, bench_iters=SWEEP_BENCH_ITERS),
        case_id=case_id,
        constants=dict(constants),
        prefix_constants=prefix_constants,
    )
    return record(
        await _measure_on_device(sandbox, run, arguments.get("timeout_sec"), budget_remaining, session_remaining)
    )


async def _measure_on_device(
    sandbox: ProbeSandbox,
    run: Callable[..., Awaitable[Any]],
    requested_timeout: Any,
    budget_remaining: float,
    session_remaining: float,
) -> dict[str, Any]:
    """Time the probe while holding the device; the ledger fields of what came of it, measured or not."""
    if sandbox.device_lock is None or not sandbox.device_lock.is_file():
        missing = (
            f"{DEVICE_LOCK_ENV} names no device sentinel"
            if sandbox.device_lock is None
            else f"the device sentinel {sandbox.device_lock} does not exist"
        )
        detail = f"{missing}, so this probe would time the GPU while something else uses it; nothing was measured"
        return {"status": UNAVAILABLE, "detail": detail}

    started = monotonic_clock()
    # The wait is bounded by the probe's own budget, and what it costs is charged to the budget: a specialist that
    # blocked here until the device came free would spend its session doing nothing.
    handle = await acquire_device_lock(sandbox.device_lock, timeout_sec=budget_remaining)
    waited = monotonic_clock() - started
    if handle is None:
        detail = (
            f"the device was still held after {waited:.0f}s, which is this probe's whole budget; nothing was measured"
        )
        return {"status": DEVICE_BUSY, "detail": detail, "duration_sec": waited}

    # The gate again, on what the wait left.
    budget_remaining -= waited
    session_remaining -= waited
    if session_remaining - ANALYSIS_RESERVE_SEC <= 0 or budget_remaining <= 0:
        release_lock(handle)
        detail = (
            f"{waited:.0f}s went on waiting for the device, which leaves {max(0.0, session_remaining):.0f}s of your "
            f"session and {max(0.0, budget_remaining):.0f}s of this round's budget; nothing was measured, so report "
            "this question as unmeasured"
        )
        return {"status": BUDGET_EXHAUSTED, "detail": detail, "duration_sec": waited}

    timeout_sec = probe_timeout_sec(
        budget_remaining=budget_remaining, session_remaining=session_remaining, requested=requested_timeout
    )
    try:
        result = await asyncio.wait_for(run(timeout_sec=timeout_sec), timeout=timeout_sec + PROBE_TOOL_GRACE_SEC)
    except asyncio.TimeoutError:
        failure = f"probe exceeded its {timeout_sec}s ceiling"
        return {"status": FAILED, "detail": failure, "duration_sec": monotonic_clock() - started}
    except (OSError, ValueError) as error:
        return {
            "status": FAILED,
            "detail": f"{type(error).__name__}: {error}",
            "duration_sec": monotonic_clock() - started,
        }
    finally:
        release_lock(handle)

    payload = result if isinstance(result, dict) else {}
    # The primitive omits every timing field on failure rather than reporting a zero, so a result carrying no
    # ``case_ms`` is a failure whatever else it says.
    case_ms = payload.get("case_ms")
    succeeded = bool(payload.get("success")) and isinstance(case_ms, (int, float))
    outcome = {
        "status": MEASURED if succeeded else FAILED,
        "detail": _detail(payload.get("message") or f"the primitive returned no measurement: {payload or result!r}"),
        "duration_sec": monotonic_clock() - started,
    }
    if succeeded:
        outcome["case_ms"] = case_ms
        outcome["kind"] = payload.get("kind", "")
        # ``narrowed`` false means other cases were timed too, so the cost was not one case and the reported spread is
        # not this case's; ``case_selection`` says whether the flag is what narrowed it.
        outcome["narrowed"] = bool(payload.get("narrowed", True))
        outcome["case_selection"] = str(payload.get("case_selection", ""))
        # Which overrides the source was seen to read.
        consumption = payload.get("override_consumption")
        if isinstance(consumption, dict) and consumption:
            outcome["override_consumption"] = consumption
    return outcome


def _record(
    sandbox: ProbeSandbox,
    budget: ProbeBudget,
    base: dict[str, Any],
    outcome: dict[str, Any],
) -> dict[str, Any]:
    """Charge one attempt to the round, persist it, and return it; an outcome without a duration took no time."""
    record = {**base, **outcome}
    record.setdefault("duration_sec", 0.0)
    budget.spend(attempts=1, seconds=float(record["duration_sec"]))
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


class ProbeSession:
    """Run the probe for one specialist session: one sandbox, one round budget, counted refusals."""

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

    async def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Run one probe, or refuse it when this session's sandbox is unusable."""
        sandbox = self._resolve_sandbox()
        if sandbox is not None:
            return await probe_variant(arguments, sandbox=sandbox, budget=self._budget)
        return self._refuse(arguments)

    def _refuse(self, arguments: dict[str, Any]) -> dict[str, Any]:
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
                "status": REFUSED,
                "detail": (
                    f"this session made {self._refusals} refused probe calls; the probe is unusable here, stop calling it"
                ),
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
        return result
