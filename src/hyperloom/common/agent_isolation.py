# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How an agent is launched: as another account, in a bubblewrap sandbox, or unconfined as before.

There is no switch. :func:`choose_agent_isolation` takes the account tier (:class:`AccountIsolation`) when
``HYPERLOOM_SPECIALIST_USER`` names an account this process can start a child as and later signal, else bubblewrap
(:class:`BwrapIsolation`) when its probe passes. When neither works, a launch that must be isolated refuses with
:class:`IsolationRefused`, naming the reasons of both; any other launch runs unconfined as before
(:class:`NoIsolation`, carrying both reasons), which :func:`announce_agent_isolation` says once at start-up with a
WARNING and :func:`agent_isolation_report` records for the session report. A launch that runs unconfined says so
itself (:func:`unconfined_launch`): with the process's one WARNING when start-up did not say it already (a probe that
passed then and gave no answer in time at the launch), and in the session report's lines
(:func:`session_isolation_lines`).

Isolation is required in a hosted run (its entry calls :func:`require_agent_isolation` once at start-up, which marks
the process) and, for a specialist, wherever ``HYPERLOOM_SPECIALIST_USER`` is set. An agent sandbox's commands
(:func:`choose_sandbox_isolation`) never run as an account, so only a hosted run requires theirs.

A hosted run launches only its hosting layer's bubblewrap, by the path its entry names, never one found on ``PATH``
(the framework image may carry its own, possibly setuid); a local run uses the ``bwrap`` on ``PATH``, and the probe
refuses a setuid one. The probe is :func:`hyperloom_common.bwrap.bwrap_probe`: once per process for each binary,
interpreter, ids, namespaces and user-namespace sysctls, and an answer that did not come in time is not kept.

Under bubblewrap the agent is the app's own kernel uid, confined by its namespaces and an empty root: no app-side trust
may derive from the ownership, mode, mtime or content of a path the agent can write.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from hyperloom_common.bwrap import BwrapPolicy, BwrapPolicyError, BwrapProbe, bwrap_probe
from hyperloom_common.env import env_str

from hyperloom.common.env_safety import (
    SPECIALIST_USER_ENV,
    RunAs,
    RunAsError,
    is_secret_shaped_env_name,
    missing_launch_capabilities,
    resolve_run_as,
)

__all__ = [
    "ACCOUNT",
    "BWRAP",
    "NONE",
    "RAY_ACTOR_PATH",
    "REFUSED",
    "SANDBOX_ACCOUNT",
    "AccountIsolation",
    "AgentIsolation",
    "BwrapIsolation",
    "IsolationRefused",
    "IsolationReport",
    "NoIsolation",
    "agent_isolation_report",
    "announce_agent_isolation",
    "bwrap_isolation",
    "choose_agent_isolation",
    "choose_sandbox_isolation",
    "isolation_required",
    "require_agent_isolation",
    "session_isolation_lines",
    "unconfined_launch",
]

log = logging.getLogger(__name__)

#: The account tier's reason on the Ray GPU actor path, which cannot start its child as another account.
RAY_ACTOR_PATH = "the Ray GPU actor path cannot run the specialist as another user"
#: The account tier's reason for an agent sandbox, whose commands run as the app's own account.
SANDBOX_ACCOUNT = "an agent sandbox runs its commands as the app's own account"
#: The bubblewrap tier's reason in a hosted run whose entry named no hosting layer bubblewrap.
NO_LAYER_BWRAP = "bubblewrap: a hosted run launches agents only with its hosting layer's bubblewrap, and none was named"

#: What a WARNING says of agents that run unconfined.
UNCONFINED = "run unconfined on this host, as before: they can read this account's files and credentials"

#: How a launch is isolated, as the report names it.
ACCOUNT = "account"
BWRAP = "bwrap"
NONE = "none"
REFUSED = "refused"

ProbeBwrap = Callable[[str | None], BwrapProbe]


class IsolationRefused(RunAsError):
    """No tier can isolate an agent that must be. A :class:`RunAsError`, so every launch path that refuses on one
    refuses on this. ``transient``: bubblewrap's probe gave no answer in time, so a later attempt may succeed."""

    def __init__(self, reasons: Sequence[str], *, transient: bool = False) -> None:
        self.reasons = tuple(reasons)
        self.transient = transient
        super().__init__("agents cannot be isolated on this host: " + "; ".join(self.reasons))


@dataclass(frozen=True)
class AccountIsolation:
    """The agent runs as ``run_as``, another account."""

    run_as: RunAs
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class BwrapIsolation:
    """The agent runs in a bubblewrap sandbox with the probed bubblewrap; ``notes`` say why not as an account."""

    probe: BwrapProbe
    notes: tuple[str, ...] = ()

    @property
    def shown(self) -> str:
        """``X.Y.Z PATH`` of the bubblewrap in use."""
        version = ".".join(map(str, self.probe.version or ()))
        return f"{version} {self.probe.path}"

    def launch(
        self,
        argv: Sequence[str],
        policy: BwrapPolicy,
        env: Mapping[str, str],
        *,
        marks: Sequence[str] = (),
        status_fd: int | None = None,
    ) -> tuple[list[str], dict[str, str]]:
        """``argv`` in ``policy``'s sandbox: the bubblewrap command line, and the environment to start it with.

        The variables ``marks`` names leave ``env`` and are set inside with ``--setenv``, so bubblewrap's own
        processes carry no mark and a reaper's TERM reaches only the sandbox (their values show in bubblewrap's
        command line: never a secret). ``--die-with-parent`` is kept only on the main thread: it follows the thread
        that starts bubblewrap, and one that ends first would SIGKILL the sandbox with no grace.

        Raises:
            IsolationRefused: ``policy`` names a problem for this bubblewrap's version (nothing was started).
            ValueError: a mark's name is secret-shaped.
        """
        secret = [name for name in marks if is_secret_shaped_env_name(name)]
        if secret:
            raise ValueError(f"marks show in bubblewrap's command line, so never a secret: {', '.join(secret)}")
        child_env = dict(env)
        moved = tuple((name, child_env.pop(name)) for name in marks if name in child_env)
        on_main = threading.current_thread() is threading.main_thread()
        policy = dataclasses.replace(
            policy, marks=(*policy.marks, *moved), die_with_parent=policy.die_with_parent and on_main
        )
        try:
            return policy.argv(self.probe, argv, status_fd=status_fd), child_env
        except BwrapPolicyError as refused:
            raise IsolationRefused(refused.problems) from None


@dataclass(frozen=True)
class NoIsolation:
    """Unconfined, as before: a launch that need not be isolated, where no tier works. ``reasons`` name both tiers'."""

    reasons: tuple[str, ...]

    @property
    def why(self) -> str:
        """The reasons as one line."""
        return "; ".join(self.reasons)


AgentIsolation = AccountIsolation | BwrapIsolation | NoIsolation


@dataclass(frozen=True)
class IsolationReport:
    """How this process's agents are isolated: ``specialists`` and ``sandboxes`` are each :data:`ACCOUNT`,
    :data:`BWRAP`, :data:`NONE` or :data:`REFUSED`; ``required`` whether a specialist must be isolated here;
    ``bwrap`` the bubblewrap in use (``X.Y.Z PATH``); ``reasons`` why a launch is not as another account."""

    specialists: str
    sandboxes: str
    required: bool
    bwrap: str | None
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """The report as a JSON document."""
        return {**dataclasses.asdict(self), "reasons": list(self.reasons)}

    def line(self) -> str:
        """The report as one line, for a session report."""
        text = f"agent isolation: specialists {self.specialists}, sandboxes {self.sandboxes}"
        if self.bwrap:
            text += f" (bubblewrap {self.bwrap})"
        return text + (": " + "; ".join(self.reasons) if self.reasons else "")


@dataclass
class _Process:
    """What this process was told at start-up, what it has logged (``warned_unconfined``: the WARNING that agents run
    unconfined), the report start-up announced, and the agent launches that ran unconfined since, by their reasons."""

    hosted: bool = False
    layer_bwrap: str | None = None
    logged: set[str] = field(default_factory=set)
    warned_unconfined: bool = False
    announced: IsolationReport | None = None
    unconfined: dict[str, int] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)


_STATE = _Process()


def _log_once(level: int, message: str) -> None:
    with _STATE.lock:
        if message in _STATE.logged:
            return
        _STATE.logged.add(message)
    log.log(level, "%s", message)


def _warn_unconfined(message: str) -> None:
    """Log ``message`` as this process's one WARNING that agents run unconfined, or at INFO once one was."""
    with _STATE.lock:
        first, _STATE.warned_unconfined = not _STATE.warned_unconfined, True
    _log_once(logging.WARNING if first else logging.INFO, message)


def _configured(user: str | None) -> str:
    return ((env_str(SPECIALIST_USER_ENV) if user is None else user) or "").strip()


def isolation_required(user: str | None = None) -> bool:
    """Whether a specialist launched now must be isolated: in a hosted run, or with an account configured (``user``,
    default ``HYPERLOOM_SPECIALIST_USER``)."""
    return _STATE.hosted or bool(_configured(user))


def _account(name: str, allow_account: bool) -> tuple[RunAs | None, str]:
    """The account to run the agent as, or why there is none (the checks of the launch paths' account test)."""
    if not name:
        return None, f"{SPECIALIST_USER_ENV} is not set"
    if not allow_account:
        return None, RAY_ACTOR_PATH
    try:
        run_as = resolve_run_as(name)
    except RunAsError as exc:
        return None, str(exc)
    if run_as is None:
        return None, f"{SPECIALIST_USER_ENV} is not set"
    missing = missing_launch_capabilities()
    if missing:
        return None, f"running a specialist as another user needs root or {', '.join(missing)}"
    return run_as, ""


def _bwrap(probe: ProbeBwrap) -> BwrapProbe:
    """The probe of the bubblewrap this process may use: a hosted run's layer one, else the one on ``PATH``. A hosted
    run whose layer path is empty or relative is refused without a probe, which takes an empty path for the
    bubblewrap on ``PATH``."""
    if not _STATE.hosted:
        return probe(None)
    layer = _STATE.layer_bwrap
    if not layer:
        return BwrapProbe(None, None, (NO_LAYER_BWRAP,))
    if not os.path.isabs(layer):
        return BwrapProbe(layer, None, (f"bubblewrap at {layer} is not an absolute path",))
    return probe(layer)


def _bwrap_reasons(found: BwrapProbe) -> tuple[str, ...]:
    """The probe's problems, each naming the bubblewrap where its own line does not."""
    where = ", ".join(str(part) for part in (found.path, found.version and ".".join(map(str, found.version))) if part)
    prefix = f"bubblewrap ({where}): " if where else "bubblewrap: "
    return tuple(line if line.startswith("bubblewrap") else prefix + line for line in found.problems)


def _unaccounted(account_reason: str, *, required: bool, warn: bool, probe: ProbeBwrap) -> BwrapIsolation | NoIsolation:
    """Bubblewrap, else a refusal (``required``) or unconfined; ``warn`` logs once that an account was asked for."""
    note = f"running them as an account: {account_reason}"
    found = _bwrap(probe)
    if found.ok:
        if warn:
            _log_once(logging.WARNING, f"agents run under bubblewrap: {note}")
        return BwrapIsolation(found, (note,))
    reasons = (note, *_bwrap_reasons(found))
    if required:
        raise IsolationRefused(reasons, transient=found.transient)
    _log_once(logging.INFO, "agents cannot run under bubblewrap here: " + "; ".join(reasons[1:]))
    return NoIsolation(reasons)


def choose_agent_isolation(
    user: str | None = None, *, allow_account: bool = True, probe: ProbeBwrap = bwrap_probe
) -> AgentIsolation:
    """How to launch a specialist now (see the module doc). ``user`` defaults to ``HYPERLOOM_SPECIALIST_USER``;
    ``allow_account=False`` (the Ray GPU actor path) never gives the account tier.

    Raises:
        IsolationRefused: The specialist must be isolated and neither tier works here.
    """
    name = _configured(user)
    run_as, reason = _account(name, allow_account)
    if run_as is not None:
        return AccountIsolation(run_as)
    return _unaccounted(reason, required=isolation_required(name), warn=bool(name), probe=probe)


def choose_sandbox_isolation(*, probe: ProbeBwrap = bwrap_probe) -> BwrapIsolation | NoIsolation:
    """How to launch an agent sandbox's commands now: bubblewrap, else unconfined, which only a hosted run refuses.

    Raises:
        IsolationRefused: In a hosted run, where bubblewrap cannot run a sandbox.
    """
    return _unaccounted(SANDBOX_ACCOUNT, required=_STATE.hosted, warn=False, probe=probe)


def bwrap_isolation(*, probe: ProbeBwrap = bwrap_probe) -> BwrapIsolation | None:
    """Bubblewrap alone, for a launcher with an account tier of its own (a native compile); ``None`` when its probe
    finds a problem."""
    found = _bwrap(probe)
    return BwrapIsolation(found) if found.ok else None


def require_agent_isolation(
    *, bwrap: str | None, allow_account: bool = True, probe: ProbeBwrap = bwrap_probe
) -> AgentIsolation:
    """At a hosted run's start-up: mark this process as one whose agents must be isolated, with the hosting layer's
    bubblewrap at ``bwrap`` (an absolute path; ``None`` or empty when the run has none, which refuses every launch
    bubblewrap would make) and no other, and choose now, so a host where neither tier works refuses before anything
    runs. Where the account is chosen, bubblewrap is probed as well: the Ray GPU actor path never starts a specialist
    as an account, so where bubblewrap cannot confine one either, one WARNING says that path refuses its specialists
    and why, and the run goes on (the account isolates every other specialist).

    Raises:
        IsolationRefused: Neither tier works here.
    """
    with _STATE.lock:
        _STATE.hosted = True
        _STATE.layer_bwrap = bwrap
    chosen = choose_agent_isolation(allow_account=allow_account, probe=probe)
    if isinstance(chosen, AccountIsolation):
        found = _bwrap(probe)
        if not found.ok:
            _log_once(
                logging.WARNING,
                f"specialists run as the account {chosen.run_as.user}, but {RAY_ACTOR_PATH} and bubblewrap cannot"
                f" confine one here, so that path refuses its specialists: {'; '.join(_bwrap_reasons(found))}",
            )
    return chosen


@dataclass(frozen=True)
class _Decision:
    """One choice as the report names it, its reasons, and the bubblewrap it uses."""

    how: str
    reasons: tuple[str, ...] = ()
    bwrap: BwrapProbe | None = None


def _decided(choose: Callable[[], AgentIsolation]) -> _Decision:
    """What ``choose`` gives now; a refusal is a decision too."""
    try:
        found = choose()
    except IsolationRefused as refused:
        return _Decision(REFUSED, refused.reasons)
    if isinstance(found, AccountIsolation):
        return _Decision(ACCOUNT)
    if isinstance(found, BwrapIsolation):
        return _Decision(BWRAP, found.notes, found.probe)
    return _Decision(NONE, found.reasons)


def agent_isolation_report(*, probe: ProbeBwrap = bwrap_probe) -> IsolationReport:
    """How a specialist and an agent sandbox launched now would be isolated, and why not as an account; never
    raises."""
    specialists = _decided(lambda: choose_agent_isolation(probe=probe))
    sandboxes = _decided(lambda: choose_sandbox_isolation(probe=probe))
    in_use = specialists.bwrap or sandboxes.bwrap
    return IsolationReport(
        specialists=specialists.how,
        sandboxes=sandboxes.how,
        required=isolation_required(),
        bwrap=BwrapIsolation(in_use).shown if in_use else None,
        reasons=tuple(dict.fromkeys((*specialists.reasons, *sandboxes.reasons))),
    )


def announce_agent_isolation(*, probe: ProbeBwrap = bwrap_probe) -> IsolationReport:
    """Say once at start-up how this process's agents are isolated: one WARNING when some run unconfined or would be
    refused, naming which and why, else one INFO line. Returns the report."""
    report = agent_isolation_report(probe=probe)
    decisions = (("specialists", report.specialists), ("agent sandboxes", report.sandboxes))
    unconfined = [name for name, how in decisions if how == NONE]
    refused = [name for name, how in decisions if how == REFUSED]
    said = []
    if unconfined:
        what = "agents" if len(unconfined) == len(decisions) else unconfined[0]
        said.append(f"{what} {UNCONFINED}")
    if refused:
        said.append(f"{' and '.join(refused)} will be refused: they must be isolated here, and no tier can")
    message = f"{'; '.join(said)} ({'; '.join(report.reasons)})"
    if unconfined:
        _warn_unconfined(message)
    elif refused:
        _log_once(logging.WARNING, message)
    else:
        _log_once(logging.INFO, report.line())
    with _STATE.lock:
        _STATE.announced = report
    return report


def unconfined_launch(isolation: NoIsolation) -> None:
    """Record that an agent was just launched unconfined, as ``isolation`` says why, for the session report, and say
    so: this process's one WARNING that agents run unconfined when nothing said it yet, else one INFO line for each
    reason."""
    with _STATE.lock:
        _STATE.unconfined[isolation.why] = _STATE.unconfined.get(isolation.why, 0) + 1
    _warn_unconfined(f"agents {UNCONFINED} ({isolation.why})")


def session_isolation_lines() -> tuple[str, ...]:
    """What a session report says of how its agents were isolated: the report start-up announced
    (:func:`announce_agent_isolation`), and how many agent launches ran unconfined since, for each reason; none when
    neither happened in this process."""
    with _STATE.lock:
        announced, unconfined = _STATE.announced, dict(_STATE.unconfined)
    lines = [announced.line()] if announced is not None else []
    for why, count in unconfined.items():
        lines.append(f"agent isolation: {count} agent launch{'' if count == 1 else 'es'} ran unconfined: {why}")
    return tuple(lines)
