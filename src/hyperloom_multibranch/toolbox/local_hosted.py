# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The real toolbox with the local-sandbox executor: one call builds it (handover 3.3, MB1 on one machine).

Every test runs in a fresh local sandbox started from the shipped runner (``hyperloom.measure.local_templates``), is
measured on cards leased from the card pool, and is accepted only as a signed bundle that verified against the key of
that sandbox. At most ``8 / TP`` tests run at once on one node.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from hyperloom.common.rule_pack import rule_pack_identity
from hyperloom.measure.archive import ResultArchive
from hyperloom.measure.hosted import FILES_ROOT, HostedConfig, HostedMeasurementBackend
from hyperloom.measure.hosted_log import STATE_FINISHED
from hyperloom.measure.hosted_spec import SpecContext
from hyperloom.measure.local import SUPPORTED_KINDS
from hyperloom.measure.framework_base import DEFAULT_CONFIGS, framework_envs
from hyperloom.measure.local_templates import local_runner_template
from hyperloom.measure.rule_pack_stage import rule_pack_envs
from hyperloom.sandbox.local import LocalSandboxProvider
from hyperloom.sandbox.templates import LocalTemplate
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.toolbox.attempts import DEFAULT_MAX_INFRA_RETRIES
from hyperloom_multibranch.toolbox.branch_leases import BranchLeases
from hyperloom_multibranch.toolbox.hosted_executor import DEFAULT_TEST_TIMEOUT_SEC, HostedExecutor
from hyperloom_multibranch.toolbox.leased_toolbox import (
    LeasedToolbox,
    capped_policy,
    session_archive,
    session_stores,
    session_tools,
    started_backend,
)
from hyperloom_multibranch.toolbox.queue import QueuePolicy
from hyperloom_multibranch.toolbox.tools import AsyncTool, kernel_tune_tool, tracelens_tool
from hyperloom_multibranch.toolbox.toolbox import AcceptanceSetup, ToolboxTuning

__all__ = [
    "LocalHostedToolbox",
    "LocalOptions",
    "LocalSession",
    "MAX_CARDS_PER_NODE",
    "build_local_toolbox",
    "last_output",
]

#: One node serves at most eight cards, so at most ``8 / TP`` tests run at once (handover 5, item 7).
MAX_CARDS_PER_NODE = 8
_SANDBOXES = "sandboxes"
_BRANCH_SANDBOXES = "branch-sandboxes"
_FRAMEWORK = "framework"
_RULE_PACK = "rule-pack"
_POLL_INTERVAL_SEC = 2.0
LEAKED_CARDS_REASON = "sandbox processes not provably gone"
#: The logs of a running test that show whether it still makes progress; the one written last is reported.
_LIVE_LOGS = frozenset({"server.log", "benchmark_stderr.log", "benchmark_stdout.log"})
#: A running test's files below its sandbox's directory: a local sandbox's root is its ``fs`` directory.
_LIVE_FILES = Path("fs") / FILES_ROOT.lstrip("/")
_TAIL_BYTES = 64 * 1024
_LINE_CHARS = 300
_ESCAPES = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]+")
logger = logging.getLogger(__name__)


class LocalHostedToolbox(LeasedToolbox):
    """A :class:`RealToolbox` that also owns its hosted backend and local sandbox provider; ``close`` releases them.

    ``sandboxes`` is the provider's directory, where a running test's live files are read for :meth:`progress`. The
    cards tests can run on are the provider's pool less the cards it lost for good (``provider.quarantined``) and the
    leases; node quarantine does not reduce them.
    """

    backend: HostedMeasurementBackend
    provider: LocalSandboxProvider
    leases: BranchLeases
    sandboxes: Path
    _reported_cards: frozenset[int] = frozenset()

    @property
    def single_node(self) -> bool:
        """Every test runs in a sandbox of the one machine the local provider stands for."""

        return True

    async def release_branch_sandbox(self, lease_id: str) -> None:
        self.leases.release(lease_id)

    async def start(self) -> None:
        """Start, after taking back the leaks an earlier run of this host recorded in the notebook."""

        await self._restore_leaked_cards()
        await super().start()

    async def _restore_leaked_cards(self) -> None:
        """A card an earlier process lost for good stays lost: its old holder may still run.

        The notebook holds those cards as events of this node with ``gpu_ids``. The provider is fresh and owns every
        card, so they are leased once here (the pool has no call to take out one card) and never released, exactly as
        a leak in this process; the others go straight back.
        """

        node = self.provider.hostname
        recorded = {gpu for event in await self._quarantine.cards() if event.node == node for gpu in event.gpu_ids}
        pool = self.provider.pool
        lost = frozenset(recorded & set(pool.free_cards()))
        if not lost:
            return
        leased = pool.acquire(len(pool.free_cards()), 0.0)
        pool.release([card for card in leased if card not in lost])
        self.provider.quarantined.update(lost)
        self._reported_cards |= lost

    def _leaked(self) -> frozenset[int]:
        return frozenset(self.provider.quarantined)

    async def _housekeeping(self) -> None:
        """Tell the notebook about cards the provider lost since last time: one event for all the new ones.

        The provider only grows a plain set; this polls it after each settled test and on every tick, so the sandbox
        layer knows nothing of the notebook. A refused write is retried at the next poll.
        """

        new = self._leaked() - self._reported_cards
        if not new:
            return
        try:
            await self._quarantine.add_cards(self.provider.hostname, new, LEAKED_CARDS_REASON)
        except Exception:
            logger.warning("could not record leaked cards in the notebook", exc_info=True)
            return
        self._reported_cards |= new

    async def close(self) -> None:
        await super().close()
        self.leases.close()
        await asyncio.to_thread(self.backend.close)
        await asyncio.to_thread(self.provider.close)
        await self._housekeeping()  # a sandbox that could not be reaped at shutdown has leaked its cards too

    async def _last_output(self, request_id: str) -> tuple[float, str] | None:
        """The newest live log in the sandbox the backend's journal names for the test's unfinished attempt."""

        return await asyncio.to_thread(self._live_log, request_id)

    def _live_log(self, request_id: str) -> tuple[float, str] | None:
        record = self.backend.journal.latest(request_id)
        if record is None or record.state == STATE_FINISHED or not record.sandbox_id:
            return None
        return last_output(self.sandboxes / record.sandbox_id / _LIVE_FILES, time.time())


def last_output(directory: Path, now: float) -> tuple[float, str] | None:
    """``(seconds since it was written, its last non-empty line)`` of the live log below ``directory`` written last.

    ``None`` when there is none or it cannot be read. The line has terminal escapes and control characters removed and
    keeps its last ``_LINE_CHARS`` characters, where a progress line has its newest figures.
    """
    found = []
    for root, _, names in os.walk(directory):  # a directory that goes away during the walk is skipped, not raised
        for path in (Path(root) / name for name in _LIVE_LOGS.intersection(names)):
            with contextlib.suppress(OSError):
                found.append((path.stat().st_mtime, path))
    if not found:
        return None
    written, newest = max(found)
    try:
        with newest.open("rb") as handle:
            handle.seek(max(0, handle.seek(0, os.SEEK_END) - _TAIL_BYTES))
            tail = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    return max(0.0, now - written), _last_line(tail)


def _last_line(text: str) -> str:
    for line in reversed(_ESCAPES.sub("", text).splitlines()):
        cleaned = _CONTROL_CHARS.sub(" ", line).strip()
        if cleaned:
            return cleaned[-_LINE_CHARS:]
    return ""


def max_parallel_tests(cards: int, tp: int) -> int:
    """How many tests of ``tp`` cards run at once on ``cards`` cards: ``min(cards, 8) // tp``, at least one."""
    if tp < 1:
        raise ValueError("tp must be at least 1")
    if cards < tp:
        raise ValueError(f"{cards} cards cannot run a test of TP={tp}")
    return min(cards, MAX_CARDS_PER_NODE) // tp


@dataclass(frozen=True)
class LocalSession:
    """Where and on what the toolbox runs.

    ``session_dir`` is the session directory. Below it: ``sandboxes/`` (the sandboxes), ``archive/`` (verified
    bundles, unless an archive is given), ``hosted/`` (the backend's journal; a restart re-attaches from it),
    ``toolbox/submissions.jsonl`` (the submit log) and ``tests/<request_id>/attempt-<n>`` (landed output).
    ``run_id`` is the run's id, the first part of every sandbox's idempotency key. ``cards`` is the card pool, for
    example ``[6, 7]``; at most eight are used for concurrency. ``tp`` is the tensor-parallel size of every test, that
    is the cards of one runner sandbox. ``hostname`` is the node name the local provider reports; this machine's by
    default.
    """

    session_dir: Path
    run_id: str
    cards: Iterable[int]
    tp: int = 1
    hostname: str | None = None


@dataclass(frozen=True)
class LocalOptions:
    """What :func:`build_local_toolbox` may be told besides the session, the notebook, acceptance and the spec.

    ``policy`` is the quota and priority policy; its ``global_concurrency`` is lowered to ``8 // tp`` when higher.
    ``runner_envs`` is the environment the runner template carries (for example ``INFERENCEX_REPO``). ``archive`` is
    the result archive, a directory archive below ``session_dir`` when omitted. ``python`` is the interpreter the
    runner and the measurement run with. ``timeout_sec`` is the timeout of one attempt, ``max_infra_retries`` the
    infrastructure retries per test before it ends as ``measurement_failed`` and ``poll_interval_sec`` how often the
    backend polls a runner for its result. ``tools`` are the asynchronous tools ``start_analysis`` may reach; ``None``
    is the local set: TraceLens, and ``kernel_tune`` when the pool can spare a card for it without leaving fewer than one
    test needs (it leases the card from the pool the branches' sandboxes lease from). Roofline profiles in the host
    process on host cards outside the card leases, so it is not offered here. ``framework_configs`` is the directory of aiter's tuned tables that every test reads private
    copies of, snapshotted below ``session_dir/framework`` as the toolbox is built: the image's when omitted, ``""`` for
    none (the tests then share the tables).
    """

    policy: QueuePolicy | None = None
    runner_envs: Mapping[str, str] | None = None
    archive: ResultArchive | None = None
    python: str = sys.executable
    timeout_sec: int = DEFAULT_TEST_TIMEOUT_SEC
    max_infra_retries: int = DEFAULT_MAX_INFRA_RETRIES
    poll_interval_sec: float = _POLL_INTERVAL_SEC
    tools: Sequence[AsyncTool] | None = None
    framework_configs: str | None = None


def _runner_template(session: LocalSession, options: LocalOptions) -> LocalTemplate:
    """The runner template: ``runner_envs`` plus the variable that names the session's rule pack."""
    template = local_runner_template(
        f"runner-{session.tp}", gpus=session.tp, python=options.python, envs=options.runner_envs
    )
    pack = rule_pack_envs(Path(session.session_dir) / _RULE_PACK)
    return replace(template, envs={**template.envs, **pack})


def _provider(
    session: LocalSession, cards: tuple[int, ...], template: LocalTemplate, options: LocalOptions
) -> LocalSandboxProvider:
    framework = framework_envs(Path(session.session_dir) / _FRAMEWORK, options.framework_configs)
    template = replace(template, envs={**template.envs, **framework})
    return LocalSandboxProvider(
        [template], gpus=cards, base_dir=Path(session.session_dir) / _SANDBOXES, hostname=session.hostname
    )


def _local_tools(leases: BranchLeases, options: LocalOptions) -> Sequence[AsyncTool]:
    """What ``options`` serves: its own tools, or TraceLens and the tuner, which is served only where a card can be
    spared for it."""
    if options.tools is not None:
        return options.tools
    tuner = (
        (kernel_tune_tool(leases, Path(options.framework_configs or DEFAULT_CONFIGS)),) if leases.can_lease(1) else ()
    )
    return (tracelens_tool(), *tuner)


def _accepting_local_kinds(acceptance: AcceptanceSetup) -> AcceptanceSetup:
    """``acceptance`` that refuses, before queueing, the kinds the local backend does not run."""
    return replace(acceptance, supported_kinds=SUPPORTED_KINDS)


def build_local_toolbox(
    session: LocalSession,
    notebook: LocalNotebook,
    acceptance: AcceptanceSetup,
    spec: SpecContext,
    *,
    options: LocalOptions = LocalOptions(),
) -> LocalHostedToolbox:
    """Build a :class:`RealToolbox` whose tests run in local sandboxes; call ``await toolbox.start()`` before use.

    Args:
        session: Where and on what the toolbox runs, see :class:`LocalSession`.
        notebook: The notebook; it takes the requests and results, holds the quarantine list and the claims, and is
            searched when a restart recovers unfinished tests.
        acceptance: The repository the parent recipes are read from, the planner (its ``plan`` must return a
            ``hyperloom.measure.points.AcceptedMeasurement``; ``HostedPlanner`` does), the active framework image
            and the campaign's workload, see :class:`~hyperloom_multibranch.toolbox.toolbox.AcceptanceSetup`.
        spec: What the instruction sheet needs besides the call: image, GPU type, commits, blob store.
        options: Everything else, see :class:`LocalOptions`.
    """

    directory, cards, tp = Path(session.session_dir), tuple(session.cards), session.tp
    parallel = max_parallel_tests(len(cards), tp)
    config = HostedConfig(
        session_dir=directory,
        run_id=session.run_id,
        spec=spec,
        runner_templates={tp: f"runner-{tp}"},
        rule_pack_digest=rule_pack_identity()["digest"],
        archive=session_archive(options.archive, directory),
        max_parallel_tests=parallel,
        poll_interval_sec=options.poll_interval_sec,
    )
    template = _runner_template(session, options)
    provider = _provider(session, cards, template, options)
    backend = started_backend(provider, config)
    leases = BranchLeases(provider, directory / _BRANCH_SANDBOXES, tp, template.envs)
    executor = HostedExecutor(
        backend,
        directory,
        timeout_sec=options.timeout_sec,
        magpie_python=options.python,
        workload_kind=acceptance.workload_kind,
    )
    toolbox = LocalHostedToolbox(
        session_stores(notebook, directory, session.run_id),
        executor,
        _accepting_local_kinds(acceptance),
        tools=session_tools(_local_tools(leases, options), notebook, directory),
        tuning=ToolboxTuning(
            policy=capped_policy(options.policy, parallel), max_infra_retries=options.max_infra_retries
        ),
    )
    toolbox.backend, toolbox.provider, toolbox.trace_root = backend, provider, directory
    toolbox.attach_leases(leases)
    toolbox.sandboxes = directory / _SANDBOXES
    toolbox.blobs = spec.blobs
    return toolbox
