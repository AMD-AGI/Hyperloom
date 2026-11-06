# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How a session's launches on this host are isolated, as the path that makes each one makes it.

A session's manifest and report say how its agents ran, and a hosted run refuses to start one unconfined beside its
credentials (:func:`~hyperloom_multibranch.scheduling.hosted_run.require_hosted_isolation`), so each kind is read from
the path that launches it, never from a chooser that path does not call:

- a specialist is started by the agent runner as the account
  :func:`~hyperloom.orchestrator.specialists.launch_env.resolve_launch_user` gives, the function the runner calls, and
  unconfined when it gives none;
- an agent or dev sandbox on this host runs its commands as the application's own account, with its environment;
- a test measured on this host's cards runs as the application's own account.

A session measured in a grant's sandboxes runs its agent and dev sandboxes and its tests there (:data:`HOSTED`), not on
this host.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass

from hyperloom.common.agent_isolation import ACCOUNT, NONE, REFUSED, isolation_required
from hyperloom.common.env_safety import SPECIALIST_USER_ENV, RunAsError
from hyperloom.orchestrator.specialists.launch_env import resolve_launch_user

log = logging.getLogger(__name__)

#: A launch made in a grant's sandboxes, not on this host.
HOSTED = "hosted"
#: Why a specialist is started unconfined.
NO_ACCOUNT = f"the agent runner starts a specialist as the account {SPECIALIST_USER_ENV} names, and it is not set"
#: Why an agent or dev sandbox on this host is unconfined.
LOCAL_SANDBOX = (
    "an agent or dev sandbox on this host runs its commands as the application's own account, with its environment"
)
#: Why a test on this host's cards is unconfined.
LOCAL_TEST = "a test on this host's cards runs as the application's own account"


@dataclass(frozen=True)
class LaunchIsolation:
    """How a session's ``specialists``, its agent and dev ``sandboxes`` and its ``tests`` are launched: each
    :data:`~hyperloom.common.agent_isolation.ACCOUNT`, :data:`~hyperloom.common.agent_isolation.NONE` (unconfined),
    :data:`~hyperloom.common.agent_isolation.REFUSED` or, for sandboxes and tests, :data:`HOSTED`; ``required`` whether
    a specialist must be isolated here; ``reasons`` why a launch on this host is not isolated."""

    specialists: str
    sandboxes: str
    tests: str
    required: bool
    reasons: tuple[str, ...]

    def kinds(self, how: str) -> tuple[str, ...]:
        """The kinds launched ``how``, as people name them."""
        kinds = (("specialists", self.specialists), ("agent and dev sandboxes", self.sandboxes), ("tests", self.tests))
        return tuple(name for name, launched in kinds if launched == how)

    def problems(self) -> str:
        """What runs unconfined on this host and what would be refused, and why; empty when nothing does."""
        unconfined, refused = self.kinds(NONE), self.kinds(REFUSED)
        said = [f"{' and '.join(unconfined)} run unconfined on this host"] if unconfined else []
        said += [f"{' and '.join(refused)} will be refused"] if refused else []
        return f"{'; '.join(said)} ({'; '.join(self.reasons)})" if said else ""

    def to_dict(self) -> dict[str, object]:
        """The record as a JSON document."""
        return {**dataclasses.asdict(self), "reasons": list(self.reasons)}

    def line(self) -> str:
        """The record as one line, for a session report."""
        text = f"agent isolation: specialists {self.specialists}, sandboxes {self.sandboxes}, tests {self.tests}"
        return text + (": " + "; ".join(self.reasons) if self.reasons else "")


def launch_isolation(*, local: bool) -> LaunchIsolation:
    """How this process launches a session's agents and tests, ``local`` when its sandboxes and tests run on this
    host's cards; never raises."""

    try:
        account = resolve_launch_user(None)
    except RunAsError as refused:
        specialists, why = REFUSED, (str(refused),)
    else:
        specialists, why = (NONE, (NO_ACCOUNT,)) if account is None else (ACCOUNT, ())
    where, here = (NONE, (LOCAL_SANDBOX, LOCAL_TEST)) if local else (HOSTED, ())
    return LaunchIsolation(specialists, where, where, isolation_required(), (*why, *here))


def announce_launch_isolation(isolation: LaunchIsolation) -> None:
    """Say how a session launches its agents, once at its start: one WARNING when some run unconfined on this host or
    would be refused, naming which and why, else one INFO line."""

    problems = isolation.problems()
    if not problems:
        log.info("%s", isolation.line())
        return
    exposed = ": they can read this account's files and credentials, as before" if isolation.kinds(NONE) else ""
    log.warning("%s%s", problems, exposed)


__all__ = [
    "HOSTED",
    "LOCAL_SANDBOX",
    "LOCAL_TEST",
    "NO_ACCOUNT",
    "LaunchIsolation",
    "announce_launch_isolation",
    "launch_isolation",
]
