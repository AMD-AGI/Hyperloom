# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a toolbox with a card pool of its own shares between its backends: the leases it counts, and the pump they restart.

The local toolbox leases cards from its provider's pool, the hosted one counts the cards of its grant; both hand
branches a :class:`~hyperloom_multibranch.contracts.toolbox.BranchLease`, report the cards tests can still run on and
start the tests that cards given back let run. This module holds that, so a backend keeps only what its leases are.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Protocol

from hyperloom.measure.archive import DirResultArchive, ResultArchive
from hyperloom.measure.hosted import HostedConfig, HostedMeasurementBackend
from hyperloom.sandbox.socket import SandboxProvider
from hyperloom_multibranch.contracts.blackboard.vocab import LeaseKind
from hyperloom_multibranch.contracts.toolbox import BranchLease, LeaseCharge, MeasurementCapacity
from hyperloom_multibranch.blackboard import LocalBlackboard
from hyperloom_multibranch.toolbox.lease_ledger import LeaseLedger
from hyperloom_multibranch.toolbox.pending_builds import BuildDriver
from hyperloom_multibranch.toolbox.queue import QueuePolicy
from hyperloom_multibranch.toolbox.long_context import LongContextSetup, ServedLongContext
from hyperloom_multibranch.toolbox.short_prompt_pool import POOL_FILE, ShortPromptPool
from hyperloom_multibranch.toolbox.short_prompts import ShortPromptSetup
from hyperloom_multibranch.toolbox.submission_log import SubmissionLog
from hyperloom_multibranch.toolbox.tools import AsyncTool, ToolJobs
from hyperloom_multibranch.toolbox.toolbox import AcceptanceSetup, RealToolbox, ToolboxStores

__all__ = [
    "LeaseBook",
    "LeasedToolbox",
    "capped_policy",
    "session_archive",
    "session_stores",
    "session_tools",
    "started_backend",
    "with_long_context",
    "with_short_prompts",
]

#: Where a session's verified bundles, submit log and tool results are, below its directory; every toolbox uses the same.
_ARCHIVE, _SUBMISSIONS, _TOOL_RESULTS = "archive", "toolbox/submissions.jsonl", "tool-results"


class LeaseBook(Protocol):
    """The leases of a toolbox's cards, as the toolbox reads them; both backends' lease classes fit."""

    on_release: Callable[[], None]
    ledger: LeaseLedger

    def capacity(self) -> MeasurementCapacity: ...

    def tests_at_once(self) -> int: ...

    async def lease(
        self, holder: str, cards: int, /, *, charge: LeaseCharge, wait_sec: float | None = None
    ) -> BranchLease: ...

    def close(self) -> None: ...


class LeasedToolbox(RealToolbox):
    """A :class:`RealToolbox` whose measurement cards are what its ``leases`` leave.

    ``leases`` is ``None`` for a toolbox that cannot count its cards: its capacity is then unknown, nothing is leased
    (the interface's own refusal) and tests are bounded only by the queue policy. Either way the start ends the leases
    an earlier toolbox left open in the blackboard, and the close those this one still holds
    (:class:`~hyperloom_multibranch.toolbox.lease_ledger.LeaseLedger`).
    """

    leases: LeaseBook | None = None
    _loop: asyncio.AbstractEventLoop | None = None
    _ledger: LeaseLedger | None = None

    async def measurement_capacity(self) -> MeasurementCapacity:
        """The cards the leases count, with the cards leased to branch sandboxes in ``leased_cards``."""

        if self.leases is None:
            return await super().measurement_capacity()
        return self.leases.capacity()

    async def lease_branch_sandbox(
        self,
        branch_id: str,
        cards: int,
        *,
        wait_sec: float | None = None,
        charge: LeaseCharge | None = None,
        lifetime_sec: float | None = None,
    ) -> BranchLease:
        """Take ``cards`` cards for ``branch_id``'s agent, waiting for running tests to give them back; they stay until
        released, whatever ``lifetime_sec`` says."""

        if self.leases is None:
            return await super().lease_branch_sandbox(branch_id, cards, wait_sec=wait_sec, charge=charge)
        charge = charge or LeaseCharge(LeaseKind.DEV, branch_id)
        return await self.leases.lease(branch_id, cards, charge=charge, wait_sec=wait_sec)

    def attach_leases(self, leases: LeaseBook) -> None:
        """Take ``leases`` as the leases on the toolbox's cards: the tests the cards they give back let run start, and
        the blackboard records their grants and returns."""

        self.leases = leases
        leases.on_release = self._pump_soon
        leases.ledger = self._lease_ledger()

    def _lease_ledger(self) -> LeaseLedger:
        if self._ledger is None:
            self._ledger = LeaseLedger(self._lease_sink, self._clock)
        return self._ledger

    def _pump_soon(self) -> None:
        """Start what the cards just given back let run, on the toolbox's loop; safe from any thread, and nothing before
        :meth:`start` or after the loop ended."""

        if self._loop is not None:
            with contextlib.suppress(RuntimeError):
                self._loop.call_soon_threadsafe(self._pump)

    def _parallel_limit(self) -> int | None:
        return None if self.leases is None else self.leases.tests_at_once()

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        ledger = self._lease_ledger()
        ledger.bind(self._loop)
        await ledger.recover()
        await super().start()

    async def close(self) -> None:
        """Close as :class:`RealToolbox` does, then end in the blackboard every lease still out, before the backend gives
        the cards back."""

        await super().close()
        await self._lease_ledger().close()


def capped_policy(policy: QueuePolicy | None, parallel: int) -> QueuePolicy:
    """``policy`` with its global concurrency lowered to what the cards can run in parallel."""
    base = policy or QueuePolicy()
    return replace(base, global_concurrency=min(base.global_concurrency, parallel))


def with_short_prompts(acceptance: AcceptanceSetup, setup: ShortPromptSetup | None) -> AcceptanceSetup:
    """``acceptance`` that runs the short-prompt check with ``setup``, a builder's option; unchanged when it names none.

    Raises:
        ValueError: ``acceptance`` already names another prompt set: the set is named once.
    """
    held = acceptance.short_prompts
    if setup is None or setup == held:
        return acceptance
    if held is not None:
        raise ValueError(
            f"the options name the short-prompt set {setup.prompt_set} and the acceptance setup already names "
            f"{held.prompt_set}: name it once"
        )
    return replace(acceptance, short_prompts=setup)


def _set_name(setup: LongContextSetup | ServedLongContext) -> str:
    return setup.prompt_set if isinstance(setup, LongContextSetup) else "of the model at the served context"


def with_long_context(
    acceptance: AcceptanceSetup, setup: LongContextSetup | ServedLongContext | None
) -> AcceptanceSetup:
    """``acceptance`` that runs the long-context check with ``setup``, a builder's option or the environment's (the model's
    own set at the served context, :class:`~.long_context.ServedLongContext`); unchanged when it names none.

    Raises:
        ValueError: ``acceptance`` already names another prompt set: the set is named once.
    """
    held = acceptance.long_context
    if setup is None or setup == held:
        return acceptance
    if held is not None:
        raise ValueError(
            f"the options name the long-context set {_set_name(setup)} and the acceptance setup already names "
            f"{_set_name(held)}: name it once"
        )
    return replace(acceptance, long_context=setup)


def started_backend(provider: SandboxProvider, config: HostedConfig) -> HostedMeasurementBackend:
    """A hosted backend over ``provider``, started."""
    backend = HostedMeasurementBackend(provider, config)
    backend.start()
    return backend


def session_archive(archive: ResultArchive | None, directory: Path) -> ResultArchive:
    """``archive``, or the directory archive below the session's ``archive/``."""
    return archive or DirResultArchive(directory / _ARCHIVE)


def session_stores(
    blackboard: LocalBlackboard, directory: Path, run_id: str, builds: BuildDriver | None = None
) -> ToolboxStores:
    """The submit log and the short-prompt pool below the session directory and the blackboard as every port the toolbox
    writes to; ``builds`` drives the native build store when the host configured one (without it no test waits for a
    build)."""
    return ToolboxStores(
        log=SubmissionLog(directory / _SUBMISSIONS, run_id),
        short_prompt_pool=ShortPromptPool(directory / POOL_FILE),
        sink=blackboard,  # type: ignore[arg-type]
        health=blackboard,  # type: ignore[arg-type]
        leases=blackboard,
        claims=blackboard,  # type: ignore[arg-type]
        records=blackboard,  # type: ignore[arg-type]
        builds=builds,
    )


def session_tools(tools: Sequence[AsyncTool], blackboard: LocalBlackboard, directory: Path) -> ToolJobs:
    """``tools`` as the jobs ``start_analysis`` starts, their results landing below the session's ``tool-results/``."""
    return ToolJobs(tools, blackboard, directory / _TOOL_RESULTS)  # type: ignore[arg-type]
