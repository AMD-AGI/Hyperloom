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
from hyperloom_multibranch.contracts.toolbox import BranchLease, MeasurementCapacity
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.toolbox.queue import QueuePolicy
from hyperloom_multibranch.toolbox.long_context import LongContextSetup
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

    def capacity(self) -> MeasurementCapacity: ...

    def tests_at_once(self) -> int: ...

    async def lease(self, holder: str, cards: int, /, *, wait_sec: float | None = None) -> BranchLease: ...

    def close(self) -> None: ...


class LeasedToolbox(RealToolbox):
    """A :class:`RealToolbox` whose measurement cards are what its ``leases`` leave.

    ``leases`` is ``None`` for a toolbox that cannot count its cards: its capacity is then unknown, nothing is leased
    (the interface's own refusal) and tests are bounded only by the queue policy.
    """

    leases: LeaseBook | None = None
    _loop: asyncio.AbstractEventLoop | None = None

    async def measurement_capacity(self) -> MeasurementCapacity:
        """The cards the leases count, with the cards leased to branch sandboxes in ``leased_cards``."""

        if self.leases is None:
            return await super().measurement_capacity()
        return self.leases.capacity()

    async def lease_branch_sandbox(self, branch_id: str, cards: int, *, wait_sec: float | None = None) -> BranchLease:
        """Take ``cards`` cards for ``branch_id``'s agent, waiting for running tests to give them back."""

        if self.leases is None:
            return await super().lease_branch_sandbox(branch_id, cards, wait_sec=wait_sec)
        return await self.leases.lease(branch_id, cards, wait_sec=wait_sec)

    def attach_leases(self, leases: LeaseBook) -> None:
        """Take ``leases`` as the leases on the toolbox's cards: the tests the cards they give back let run start."""

        self.leases = leases
        leases.on_release = self._pump_soon

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
        await super().start()


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


def with_long_context(acceptance: AcceptanceSetup, setup: LongContextSetup | None) -> AcceptanceSetup:
    """``acceptance`` that runs the long-context check with ``setup``, a builder's option; unchanged when it names none.

    Raises:
        ValueError: ``acceptance`` already names another prompt set: the set is named once.
    """
    held = acceptance.long_context
    if setup is None or setup == held:
        return acceptance
    if held is not None:
        raise ValueError(
            f"the options name the long-context set {setup.prompt_set} and the acceptance setup already names "
            f"{held.prompt_set}: name it once"
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


def session_stores(notebook: LocalNotebook, directory: Path, run_id: str) -> ToolboxStores:
    """The submit log below the session directory and the notebook as every port the toolbox writes to."""
    return ToolboxStores(
        log=SubmissionLog(directory / _SUBMISSIONS, run_id),
        sink=notebook,  # type: ignore[arg-type]
        health=notebook,  # type: ignore[arg-type]
        claims=notebook,  # type: ignore[arg-type]
        records=notebook,  # type: ignore[arg-type]
    )


def session_tools(tools: Sequence[AsyncTool], notebook: LocalNotebook, directory: Path) -> ToolJobs:
    """``tools`` as the jobs ``start_analysis`` starts, their results landing below the session's ``tool-results/``."""
    return ToolJobs(tools, notebook, directory / _TOOL_RESULTS)  # type: ignore[arg-type]
