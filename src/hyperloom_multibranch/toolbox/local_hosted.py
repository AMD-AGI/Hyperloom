# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The real toolbox with the local-sandbox executor: one call builds it (handover 3.3, MB1 on one machine).

Every test runs in a fresh local sandbox started from the shipped runner (``hyperloom.measure.local_templates``), is
measured on cards leased from the card pool, and is accepted only as a signed bundle that verified against the key of
that sandbox. At most ``8 / TP`` tests run at once on one node.

The shared compiled-code cache (design 4.21) is off unless a directory is named: ``LocalOptions.jit_store``, or the
``HYPERLOOM_JIT_STORE`` environment variable of the host process when the option is not given (a command line that
builds the toolbox with the default options is switched on by exporting it). Tests then read the directory's verified snapshots before their
server starts, and what an accepted test compiled is published to it by this host.

The short-prompt consistency check is off the same way: ``LocalOptions.short_prompts``, or the key named by
``HYPERLOOM_SHORT_PROMPTS`` in the host environment when the option is not given. The set is read in the runner cache of
this host (``HYPERLOOM_RUNNER_CACHE``, where ``python -m hyperloom.measure.short_prompts.build`` staged it) before the
toolbox is built, and the runners are told that directory; one that is not staged there refuses the build.

The long-context check is off the same way: ``LocalOptions.long_context``, or the key named by ``HYPERLOOM_LONG_CONTEXT``
(staged by ``python -m hyperloom.measure.long_context.build``). It runs for the requests that set ``long_context_check``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from hyperloom.common.outside_facts.rule_pack import rule_pack_identity
from hyperloom.measure.archive import ResultArchive
from hyperloom.measure.hosted import FILES_ROOT, HostedConfig, HostedMeasurementBackend
from hyperloom.measure.hosted_log import STATE_FINISHED
from hyperloom.measure.hosted_spec import SpecContext
from hyperloom.measure.jit.store import DirJitStore
from hyperloom.measure.live_log import last_output
from hyperloom.measure.local import SUPPORTED_KINDS
from hyperloom.measure.framework.base import DEFAULT_CONFIGS, framework_envs
from hyperloom.measure.local_templates import local_runner_template
from hyperloom.measure.rule_pack_stage import rule_pack_envs
from hyperloom.measure.runner.jit import JIT_CACHE_ENV
from hyperloom.measure.short_prompts.staging import CACHE_ENV as RUNNER_CACHE_ENV
from hyperloom.measure.short_prompts.staging import cache_root
from hyperloom.sandbox.card_occupancy import CARD_GUARD_ENV, CardGuard, CardReading, card_guard, read_holders
from hyperloom.sandbox.gated_pool import GatedPool
from hyperloom.sandbox.gpu_pool import GpuPool
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
    with_long_context,
    with_short_prompts,
)
from hyperloom_multibranch.toolbox.queue import QueuePolicy
from hyperloom_multibranch.toolbox.long_context import LongContextSetup, staged_long_context
from hyperloom_multibranch.toolbox.short_prompts import ShortPromptSetup, staged_short_prompts
from hyperloom_multibranch.toolbox.tools import (
    AsyncTool,
    UpstreamPrSearch,
    kernel_tune_tool,
    roofline_tool,
    tracelens_tool,
)
from hyperloom_multibranch.toolbox.toolbox import AcceptanceSetup, ToolboxTuning

__all__ = [
    "JIT_STORE_ENV",
    "LocalHostedToolbox",
    "LocalOptions",
    "LocalSession",
    "MAX_CARDS_PER_NODE",
    "build_local_toolbox",
]

#: One node serves at most eight cards, so at most ``8 / TP`` tests run at once (handover 5, item 7).
MAX_CARDS_PER_NODE = 8
#: The host environment variable that names the shared compiled-code cache when ``LocalOptions.jit_store`` is not given.
JIT_STORE_ENV = "HYPERLOOM_JIT_STORE"
_SANDBOXES = "sandboxes"
_BRANCH_SANDBOXES = "branch-sandboxes"
_FRAMEWORK = "framework"
_RULE_PACK = "rule-pack"
_POLL_INTERVAL_SEC = 2.0
LEAKED_CARDS_REASON = "sandbox processes not provably gone"
#: A running test's files below its sandbox's directory: a local sandbox's root is its ``fs`` directory.
_LIVE_FILES = Path("fs") / FILES_ROOT.lstrip("/")
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
        leased = GpuPool.acquire(pool, len(pool.free_cards()), 0.0)  # the pool's own: a leak is kept whoever holds it
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
    is the local set: TraceLens, the roofline of a profile test's trace (``bypass`` analysis, no card and no model), the
    upstream PR search (GitHub is asked from this process, never from a sandbox) and ``kernel_tune`` when the pool can spare a
    card for it without leaving fewer than one test needs (it leases the card from the pool the branches' sandboxes lease
    from). The orchestrator's own roofline profiles on a host card outside the card leases, so it is not offered.
    ``framework_configs`` is the directory of aiter's tuned tables that every test reads private
    copies of, snapshotted below ``session_dir/framework`` as the toolbox is built: the image's when omitted, ``""`` for
    none (the tests then share the tables). ``jit_store`` is the directory of the shared compiled-code cache: the
    runners read it (``HYPERLOOM_JIT_CACHE`` in their template, which therefore wins over the same name in
    ``runner_envs``) and this host publishes into it; ``None`` takes ``HYPERLOOM_JIT_STORE`` of the host environment,
    and without either this host publishes nothing (a ``HYPERLOOM_JIT_CACHE`` in ``runner_envs`` alone is a cache the
    runners only read). ``card_guard`` is what is done about a process that holds a card of a test or of a lease: observed
    (kept and recorded) or enforced (the test refused and retried, the card not handed out); ``None`` takes
    ``HYPERLOOM_CARD_GUARD`` of the host environment, and without it the guard only observes. The runners get the same
    setting. ``card_holders`` says who holds some cards, which the pool asks before it hands a card out: the kernel's list
    of the machine unless a test of this module says otherwise. ``short_prompts`` is the staged prompt set of the
    short-prompt consistency check (``acceptance.short_prompts`` of a builder that names it there says the same and must
    not differ); ``None`` takes the key in ``HYPERLOOM_SHORT_PROMPTS`` of the host environment, and without either no
    check runs. It is read in this host's runner cache (``HYPERLOOM_RUNNER_CACHE``, ``/runner-cache`` when unset) and
    the build refuses one that is not staged there; the runners read the same directory (``HYPERLOOM_RUNNER_CACHE`` in
    their template, which therefore wins over the same name in ``runner_envs``), the branches' sandboxes are not told.
    ``long_context`` is the staged prompt set of the long-context check, named and read the same way (``None`` takes the
    key in ``HYPERLOOM_LONG_CONTEXT``); a request that sets ``long_context_check`` runs the check with it.
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
    jit_store: Path | None = None
    card_guard: CardGuard | None = None
    card_holders: Callable[[Sequence[int]], CardReading] = read_holders
    short_prompts: ShortPromptSetup | None = None
    long_context: LongContextSetup | None = None


def card_guard_of(options: LocalOptions, environ: Mapping[str, str] | None = None) -> CardGuard:
    """The guard ``options`` (or ``HYPERLOOM_CARD_GUARD``) names; observing when neither does.

    Raises:
        ValueError: When the environment names something that is not a guard.
    """
    return options.card_guard or card_guard(os.environ if environ is None else environ)


def jit_store_root(options: LocalOptions, environ: Mapping[str, str] | None = None) -> Path | None:
    """The absolute directory of the shared compiled-code cache ``options`` (or ``HYPERLOOM_JIT_STORE``) names, if any.

    The runner starts in a directory of its own, so a relative path is made absolute here, once.
    """
    source = os.environ if environ is None else environ
    named = options.jit_store or source.get(JIT_STORE_ENV, "").strip() or None
    return None if named is None else Path(named).expanduser().resolve()


def _runner_template(session: LocalSession, options: LocalOptions) -> LocalTemplate:
    """The runner template: ``runner_envs`` plus the variable that names the session's rule pack."""
    template = local_runner_template(
        f"runner-{session.tp}", gpus=session.tp, python=options.python, envs=options.runner_envs
    )
    pack = rule_pack_envs(Path(session.session_dir) / _RULE_PACK)
    return replace(template, envs={**template.envs, **pack})


def _shared_dirs(jit_store: Path | None, eval_sets: bool, environ: Mapping[str, str]) -> dict[str, str]:
    """The host directories the runners are told, by the variable each reads, for the shared stores that are on: the
    compiled-code cache, and the runner cache the check sets are read from when a check is on (made absolute: a runner
    starts elsewhere)."""
    named = {
        JIT_CACHE_ENV: jit_store,
        RUNNER_CACHE_ENV: cache_root(environ).expanduser().resolve() if eval_sets else None,
    }
    return {name: str(path) for name, path in named.items() if path is not None}


def _provider(
    session: LocalSession,
    cards: tuple[int, ...],
    template: LocalTemplate,
    options: LocalOptions,
    shared_dirs: Mapping[str, str],
) -> LocalSandboxProvider:
    """The provider whose runner template also carries the session's framework tables, the card guard and the directories
    of the shared stores that are on: none is part of the environment the branches' own sandboxes get. Its cards are
    handed out, to a test's sandbox and to a lease alike, under the same guard."""
    framework = framework_envs(Path(session.session_dir) / _FRAMEWORK, options.framework_configs)
    guard = {CARD_GUARD_ENV: card_guard_of(options).value}
    template = replace(template, envs={**template.envs, **framework, **shared_dirs, **guard})
    provider = LocalSandboxProvider(
        [template], gpus=cards, base_dir=Path(session.session_dir) / _SANDBOXES, hostname=session.hostname
    )
    provider.pool = GatedPool(cards, card_guard_of(options), read=options.card_holders)
    return provider


def _local_tools(leases: BranchLeases, options: LocalOptions) -> Sequence[AsyncTool]:
    """What ``options`` serves: its own tools, or TraceLens, the roofline, the upstream PR search and the tuner, which is
    served only where a card can be spared for it. The first three need no card: they read a trace that is already there, or
    ask GitHub from this process."""
    if options.tools is not None:
        return options.tools
    tuner = (
        (kernel_tune_tool(leases, Path(options.framework_configs or DEFAULT_CONFIGS)),) if leases.can_lease(1) else ()
    )
    return (tracelens_tool(), roofline_tool(), UpstreamPrSearch(), *tuner)


def _accepting_local_kinds(acceptance: AcceptanceSetup) -> AcceptanceSetup:
    """``acceptance`` that refuses, before queueing, the kinds the local backend does not run."""
    return replace(acceptance, supported_kinds=SUPPORTED_KINDS)


def _acceptance_with_prompts(
    acceptance: AcceptanceSetup, options: LocalOptions, environ: Mapping[str, str]
) -> AcceptanceSetup:
    """``acceptance`` with the short-prompt set the options (else the acceptance setup, else ``HYPERLOOM_SHORT_PROMPTS``)
    name, read in this host's runner cache: the runners read the same files.

    Raises:
        ShortPromptsNotStaged: A set is asked for and is not staged there.
        ValueError: The options and the acceptance setup name different sets.
    """
    named = with_short_prompts(acceptance, options.short_prompts).short_prompts
    long_named = with_long_context(acceptance, options.long_context).long_context
    return replace(
        acceptance,
        short_prompts=staged_short_prompts(named, environ),
        long_context=staged_long_context(long_named, environ),
    )


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

    Raises:
        ShortPromptsNotStaged: The short-prompt check is asked for (the option or ``HYPERLOOM_SHORT_PROMPTS``) and its
            set is not staged in this host's runner cache; nothing has been started.
        LongContextNotStaged: The same for the long-context check (the option or ``HYPERLOOM_LONG_CONTEXT``).
        ValueError: The options and the acceptance setup name different short-prompt or long-context sets.
    """

    directory, cards, tp = Path(session.session_dir), tuple(session.cards), session.tp
    parallel = max_parallel_tests(len(cards), tp)
    jit_root = jit_store_root(options)
    acceptance = _acceptance_with_prompts(acceptance, options, os.environ)
    config = HostedConfig(
        session_dir=directory,
        run_id=session.run_id,
        spec=spec,
        runner_templates={tp: f"runner-{tp}"},
        rule_pack_digest=rule_pack_identity()["digest"],
        archive=session_archive(options.archive, directory),
        max_parallel_tests=parallel,
        poll_interval_sec=options.poll_interval_sec,
        jit_store=None if jit_root is None else DirJitStore(jit_root),
    )
    template = _runner_template(session, options)
    eval_sets = acceptance.short_prompts is not None or acceptance.long_context is not None
    shared_dirs = _shared_dirs(jit_root, eval_sets, os.environ)
    provider = _provider(session, cards, template, options, shared_dirs)
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
