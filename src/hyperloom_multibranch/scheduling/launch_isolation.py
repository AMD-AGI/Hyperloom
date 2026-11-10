# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How a session's launches on this host are isolated, as the path that makes each one makes it.

A session's manifest and report say how its agents ran, so each kind is read from the path that launches it, never
from a chooser that path does not call:

- a specialist is started by the agent runner as
  :func:`~hyperloom.orchestrator.specialists.launch_env.resolve_launch_isolation` chooses, the function the runner
  calls: as an account, under bubblewrap, unconfined, or not at all where it must be isolated and cannot be;
- an agent or dev sandbox on this host (:class:`~hyperloom_multibranch.specialist.agent_sandbox.LocalAgentSandbox`)
  runs its commands as the app's own account and asks no chooser, so it runs unconfined (:data:`SANDBOX_UNCONFINED`);
- a test measured on this host's cards runs in a sandbox the local toolbox's provider makes, which names no account
  for it and is given no runner plan (:class:`~hyperloom.sandbox.local.ProcessUsers`), so it runs unconfined
  (:data:`TESTS_UNCONFINED`).

A session measured in a grant's sandboxes runs its agent and dev sandboxes and its tests there (:data:`HOSTED`), not on
this host.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass

from hyperloom.common.agent_isolation import (
    ACCOUNT,
    BWRAP,
    NONE,
    REFUSED,
    AccountIsolation,
    BwrapIsolation,
    IsolationRefused,
    isolation_required,
)
from hyperloom.common.env_safety import RunAsError
from hyperloom.orchestrator.specialists.launch_env import resolve_launch_isolation

log = logging.getLogger(__name__)

#: A launch made in a grant's sandboxes, not on this host.
HOSTED = "hosted"
#: Why an agent or dev sandbox on this host is not isolated.
SANDBOX_UNCONFINED = "an agent or dev sandbox on this host runs its commands as the app's own account, unconfined"
#: Why a test on this host's cards is not isolated.
TESTS_UNCONFINED = "a test on this host runs as the app's own account, unconfined: its provider is given no runner plan"


@dataclass(frozen=True)
class LaunchIsolation:
    """How a session's ``specialists``, its agent and dev ``sandboxes`` and its ``tests`` are launched: each
    :data:`~hyperloom.common.agent_isolation.BWRAP`, :data:`~hyperloom.common.agent_isolation.NONE` (unconfined) or
    :data:`~hyperloom.common.agent_isolation.REFUSED`, specialists also
    :data:`~hyperloom.common.agent_isolation.ACCOUNT`, and sandboxes and tests :data:`HOSTED`; ``required`` whether a
    launch must be isolated here; ``reasons`` why a launch on this host is not isolated."""

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

    specialists, why = _specialist_isolation()
    where, here = _local_sandbox_isolation() if local else (HOSTED, ())
    tests, test_why = _local_test_isolation() if local else (HOSTED, ())
    reasons = tuple(dict.fromkeys((*why, *here, *test_why)))
    return LaunchIsolation(specialists, where, tests, isolation_required(), reasons)


def _specialist_isolation() -> tuple[str, tuple[str, ...]]:
    """How the agent runner launches a specialist now, and why it is not isolated; a session's specialists are never
    launched on the Ray actor path."""

    try:
        found = resolve_launch_isolation(None)
    except IsolationRefused as refused:
        return REFUSED, refused.reasons
    except RunAsError as refused:
        return REFUSED, (str(refused),)
    if isinstance(found, AccountIsolation):
        return ACCOUNT, ()
    return (BWRAP, ()) if isinstance(found, BwrapIsolation) else (NONE, found.reasons)


def _local_sandbox_isolation() -> tuple[str, tuple[str, ...]]:
    """How an agent or dev sandbox on this host is launched, and why it is not isolated: unconfined, as its launch
    path asks no chooser, in a hosted run too."""

    return NONE, (SANDBOX_UNCONFINED,)


def _local_test_isolation() -> tuple[str, tuple[str, ...]]:
    """How a test on this host's cards is launched, and why it is not isolated: unconfined, as the local toolbox's
    provider confines no test sandbox without a runner plan, in a hosted run too."""

    return NONE, (TESTS_UNCONFINED,)


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
    "SANDBOX_UNCONFINED",
    "TESTS_UNCONFINED",
    "LaunchIsolation",
    "announce_launch_isolation",
    "launch_isolation",
]
