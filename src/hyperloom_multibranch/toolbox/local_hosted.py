# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The real toolbox with the local-sandbox executor: one call builds it, on one machine.

Every test runs in a fresh local sandbox started from the shipped runner (``hyperloom.measure.local_templates``), is
measured on cards leased from the card pool, and is accepted only as a signed bundle that verified against the key of
that sandbox. At most ``8 / TP`` tests run at once on one node.

The shared compiled-code cache is in ``LocalOptions.jit_store``, else in the directory the
``HYPERLOOM_JIT_STORE`` environment variable of the host process names, else in :data:`DEFAULT_JIT_STORE` and this
account's uid below this host's temporary directory (``TMPDIR`` honoured), not beside the session root: that lies on the
session's shared storage, where a CK module build crawls. Tests read the directory's verified snapshots before their
server starts, and what an accepted test compiled is published to it by this host. The runners load what it holds as
compiled code, so the build refuses a link there, and a directory that is not this account's own or that its group or
others can write: another account could otherwise put snapshots of its own there.

The short-prompt consistency check is on by default: ``LocalOptions.short_prompts``, or the key named by
``HYPERLOOM_SHORT_PROMPTS`` in the host environment when the option is not given (``off`` turns it off), else the set
for the session model's tokenizer, staged here at the build when it is missing (``short_prompts.default_short_prompts``;
a failure leaves the check off with a warning that says why). The set is read in the runner cache of
this host (``HYPERLOOM_RUNNER_CACHE``, where ``python -m hyperloom.measure.short_prompts.build`` staged it) before the
toolbox is built, and the runners are told that directory; one that is not staged there refuses the build.

The long-context check is on by default the same way: ``LocalOptions.long_context``, or the key named by
``HYPERLOOM_LONG_CONTEXT`` (``off`` turns it off), else the set ``python -m hyperloom.measure.long_context.build`` makes for
the session model at the context the test's launch serves, staged here when acceptance first plans a test of that context
and it is missing (``long_context.ServedLongContext``). It runs under the short-prompt check's triggers, on the test's own
server.

A test's server runs a branch's patches, so where bubblewrap can sandbox it, every sandbox runs confined
(:mod:`hyperloom.sandbox.confined`): it writes its own tree and reads its toolchain, its own signing key and what
:func:`runner_plan` names (the model, the Hugging Face hub cache and the trees its template's variables name), never the
session's other files, another sandbox's key, the app's credentials or its processes. Where bubblewrap cannot, tests run
as before, said once at WARNING; a hosted run refuses to build the toolbox. A model whose entry links into a credential
location (``~/.ssh``, ``~/.claude``, a login file) runs each test as before, said at WARNING, and a hosted run refuses
its tests: no mask would hide what that bind shows.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import pwd
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final

from hyperloom.common.confinement import credential_paths
from hyperloom.common.outside_facts.rule_pack import PACK_ENV, rule_pack_identity
from hyperloom.measure.archive import ResultArchive
from hyperloom.measure.hosted import FILES_ROOT, HostedConfig, HostedMeasurementBackend
from hyperloom.measure.hosted_log import STATE_FINISHED
from hyperloom.measure.hosted_spec import SpecContext
from hyperloom.measure.jit.key import build_surface
from hyperloom.measure.jit.store import DirJitStore
from hyperloom.measure.live_log import last_output_below
from hyperloom.measure.local import SUPPORTED_KINDS
from hyperloom.measure.framework.base import (
    SESSION_SNAPSHOT_DIR,
    SNAPSHOT_ENV,
    FrameworkBaseError,
    framework_envs,
    image_configs,
    load_snapshot,
)
from hyperloom.measure.jit.native.config import BUILD_STORE_ENV, BUILD_TRUST_ENV
from hyperloom.measure.local_templates import image_env, local_runner_template
from hyperloom.measure.rule_pack_stage import rule_pack_envs
from hyperloom.measure.runner.jit import JIT_CACHE_ENV
from hyperloom.measure.short_prompts.staging import CACHE_ENV as RUNNER_CACHE_ENV
from hyperloom.measure.short_prompts.staging import cache_root
from hyperloom.sandbox.card_occupancy import (
    CARD_GUARD_ENV,
    CardGuard,
    CardReading,
    card_guard,
    read_holders,
    read_listed,
)
from hyperloom.sandbox.gated_pool import GatedPool
from hyperloom.sandbox.gpu_pool import GpuPool
from hyperloom.sandbox.local import LocalSandboxProvider, ProcessUsers, RunnerPlan
from hyperloom.sandbox.rooted import open_rooted
from hyperloom.sandbox.socket import SandboxDenied
from hyperloom.sandbox.templates import LocalTemplate
from hyperloom_common.fs_utils import is_private_dir
from hyperloom_multibranch.blackboard import LocalBlackboard
from hyperloom_multibranch.contracts.blackboard.interfaces import BlackboardError
from hyperloom_multibranch.toolbox.attempts import DEFAULT_MAX_INFRA_RETRIES
from hyperloom_multibranch.toolbox.branch_leases import BranchLeases, CardDrain
from hyperloom_multibranch.toolbox.build_preflight import changeable_by_others
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
from hyperloom_multibranch.toolbox.native_builds import native_builds, with_native_plan
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
    "DEFAULT_JIT_STORE",
    "HOST_TOOL_REPO_ENVS",
    "JIT_STORE_ENV",
    "JitStoreNotPrivate",
    "LocalHostedToolbox",
    "LocalOptions",
    "LocalSession",
    "MAX_CARDS_PER_NODE",
    "build_local_toolbox",
    "host_tool_repos",
    "hub_cache",
    "private_jit_store",
    "runner_cache_dir",
    "runner_plan",
]

#: One node serves at most eight cards, so at most ``8 / TP`` tests run at once.
MAX_CARDS_PER_NODE = 8
#: The host environment variable that names the shared compiled-code cache when ``LocalOptions.jit_store`` is not given.
JIT_STORE_ENV = "HYPERLOOM_JIT_STORE"
#: The shared compiled-code cache below this host's temporary directory when neither the options nor the environment
#: name one, followed by ``-<uid>``: each account on the host keeps its own.
DEFAULT_JIT_STORE = "hyperloom-jit-store"
#: The variables that name where a runner clones the tools its tests run (InferenceX) from. A local runner starts from an
#: emptied environment, so :func:`host_tool_repos` carries them from this host's.
HOST_TOOL_REPO_ENVS: Final = ("INFERENCEX_REPO",)
#: The runner template's variables naming what a confined runner reads: the framework snapshot, the staged rule pack,
#: the compiled-code store, the runner cache (the inference_suite staging every test runs and the check sets), the
#: native build store and its trust file, and a local InferenceX repository.
RUNNER_READS: Final = (
    SNAPSHOT_ENV,
    PACK_ENV,
    JIT_CACHE_ENV,
    RUNNER_CACHE_ENV,
    BUILD_STORE_ENV,
    BUILD_TRUST_ENV,
    *HOST_TOOL_REPO_ENVS,
)
_FILE_URL = "file://"
_HUB = "hub"
_HUGGINGFACE = "huggingface"
#: A Hugging Face hub repository's directory starts so; its snapshots' files, nested ones too, link into its ``blobs``.
_HUB_REPOS = ("models--", "datasets--", "spaces--")
_BLOBS = "blobs"
_SANDBOXES = "sandboxes"
_BRANCH_SANDBOXES = "branch-sandboxes"
_RULE_PACK = "rule-pack"
_POLL_INTERVAL_SEC = 2.0
LEAKED_CARDS_REASON = "sandbox processes not provably gone"
FREED_CARDS_REASON = "nothing of the sandbox that held them runs any more"
#: A local sandbox's root, below its directory: a running test's files are below it at ``FILES_ROOT``.
_ROOT = "fs"
logger = logging.getLogger(__name__)


class LocalHostedToolbox(LeasedToolbox):
    """A :class:`RealToolbox` that also owns its hosted backend and local sandbox provider; ``close`` releases them.

    ``sandboxes`` is the provider's directory, where a running test's live files are read for :meth:`progress`. The
    cards tests can run on are the provider's pool less the cards it lost (``provider.quarantined``) and the leases;
    node quarantine does not reduce them. A card lost by a sandbox of this process comes back once nothing of that
    sandbox runs; one an earlier process lost stays lost.
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
        """Start, after taking back the leaks an earlier run of this host recorded in the blackboard."""

        await self._restore_leaked_cards()
        await super().start()

    async def _restore_leaked_cards(self) -> None:
        """A card an earlier process lost for good stays lost: its old holder may still run.

        The blackboard holds those cards as events of this node with ``gpu_ids``. The provider is fresh and owns every
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
        """Take back the cards the provider freed, then tell the blackboard about cards it lost since last time: one
        event for all the new ones.

        The provider only grows a plain set; this polls it after each settled test and on every tick, so the sandbox
        layer knows nothing of the blackboard. A refused write is retried at the next poll.
        """

        await self._take_back_freed_cards()
        new = self._leaked() - self._reported_cards
        if not new:
            return
        try:
            await self._quarantine.add_cards(self.provider.hostname, new, LEAKED_CARDS_REASON)
        except (BlackboardError, OSError):  # the blackboard refused or was not reached: the next poll retries
            logger.warning("could not record leaked cards in the blackboard", exc_info=True)
            return
        self._reported_cards |= new

    async def _take_back_freed_cards(self) -> None:
        """Put back in the pool the cards of ended sandboxes nothing of which runs any more
        (:meth:`~hyperloom.sandbox.local.LocalSandboxProvider.release_freed_cards`), lift their record in the blackboard
        and start the tests they let run.

        A refused lift is not retried: the blackboard keeps the cards recorded as lost, which is read back only by a
        restart into this session, and keeps them out of the pool there.
        """

        freed = frozenset(await asyncio.to_thread(self.provider.release_freed_cards))
        if not freed:
            return
        logger.info("cards %s are back in the pool: %s", sorted(freed), FREED_CARDS_REASON)
        recorded = freed & self._reported_cards
        if recorded:
            try:
                await self._quarantine.return_cards(self.provider.hostname, recorded, FREED_CARDS_REASON)
            except (BlackboardError, OSError):  # refused or not reached: the cards stay recorded lost
                logger.warning(
                    "could not record in the blackboard that cards %s are back", sorted(recorded), exc_info=True
                )
            else:
                self._reported_cards -= recorded
        self._pump_soon()

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
        """The live log of the sandbox the backend's journal names for the test's unfinished attempt, reached from the
        sandbox's root by a walk no link its processes made leads out of (they write ``/out``). ``None`` when there is
        none or it cannot be read: the sandbox's processes change that tree as it is walked, and a test's progress is
        reported without a live line rather than not at all."""
        record = self.backend.journal.latest(request_id)
        if record is None or record.state == STATE_FINISHED or not record.sandbox_id:
            return None
        try:
            files = open_rooted(self.sandboxes / record.sandbox_id / _ROOT, FILES_ROOT, os.O_RDONLY | os.O_DIRECTORY)
        except (OSError, SandboxDenied):
            return None
        try:
            return last_output_below(files, time.time())
        except OSError:
            logger.debug("the live log of sandbox %s could not be read", record.sandbox_id, exc_info=True)
            return None
        finally:
            os.close(files)


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
    """What :func:`build_local_toolbox` may be told besides the session, the blackboard, acceptance and the spec.

    ``policy`` is the quota and priority policy; its ``global_concurrency`` is lowered to ``8 // tp`` when higher.
    ``runner_envs`` add to the image's variables the runner template carries (for example ``INFERENCEX_REPO``) and win
    over the same name in the host's ``INFERENCEX_REPO``, which the template carries when set.
    ``archive`` is the result archive, a directory archive below ``session_dir`` when omitted. ``python`` is the interpreter the
    runner runs with. ``timeout_sec`` is the timeout of one attempt, ``max_infra_retries`` the
    infrastructure retries per test before it ends as ``measurement_failed`` and ``poll_interval_sec`` how often the
    backend polls a runner for its result. ``tools`` are the asynchronous tools ``start_analysis`` may reach; ``None``
    is the local set: TraceLens, the roofline of a profile test's trace (``non-llm`` analysis, no card and no model), the
    upstream PR search (GitHub is asked from this process, never from a sandbox) and ``kernel_tune`` when the pool can spare a
    card for it without leaving fewer than one test needs (it leases the card from the pool the branches' sandboxes lease
    from). The orchestrator's own roofline profiles on a host card outside the card leases, so it is not offered.
    ``framework_configs`` is the directory of aiter's tuned tables that every test reads private
    copies of, snapshotted below ``session_dir/framework`` as the toolbox is built: the image's when omitted, ``""`` for
    none (the tests then share the tables). ``jit_store`` is the directory of the shared compiled-code cache: the
    runners read it (``HYPERLOOM_JIT_CACHE`` in their template, which therefore wins over the same name in
    ``runner_envs``) and this host publishes into it; ``None`` takes ``HYPERLOOM_JIT_STORE`` of the host environment,
    and without either :data:`DEFAULT_JIT_STORE` and this account's uid below this host's temporary directory; the
    build refuses a link there, and one that is not this account's own or that its group or others can write.
    ``card_guard`` is what is done about a process that holds a card of a test or of a lease: observed
    (kept and recorded) or enforced (the test refused and retried, the card not handed out); ``None`` takes
    ``HYPERLOOM_CARD_GUARD`` of the host environment, and without it the guard enforces. The runners get the same
    setting. ``card_holders`` says who holds some cards, which the pool asks before it hands a card out: the kernel's list
    of the machine unless a test of this module says otherwise; ``card_memory`` lists what the kernel has on some cards,
    which a released lease waits on before its cards go back (:func:`~hyperloom.sandbox.card_occupancy.read_listed`
    unless a test says otherwise). ``short_prompts`` is the staged prompt set of the
    short-prompt consistency check (``acceptance.short_prompts`` of a builder that names it there says the same and must
    not differ); ``None`` takes the key in ``HYPERLOOM_SHORT_PROMPTS`` of the host environment (``off`` for no check),
    and without either the set for the session model's tokenizer, staged at the build when missing. It is read in this
    host's runner cache (``HYPERLOOM_RUNNER_CACHE``, ``/runner-cache`` when unset) and the build refuses one that is not staged
    there; the runners read the same directory (``HYPERLOOM_RUNNER_CACHE`` in
    their template, which therefore wins over the same name in ``runner_envs``), the branches' sandboxes are not told.
    ``long_context`` is the staged prompt set of the long-context check, named, found and read the same way (``None`` takes
    the key in ``HYPERLOOM_LONG_CONTEXT``, ``off`` for no check, and without either the model's own set at the context
    each test's launch serves).
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
    card_memory: Callable[[Sequence[int]], CardReading] = read_listed
    short_prompts: ShortPromptSetup | None = None
    long_context: LongContextSetup | None = None


def card_guard_of(options: LocalOptions, environ: Mapping[str, str] | None = None) -> CardGuard:
    """The guard ``options`` (or ``HYPERLOOM_CARD_GUARD``) names; enforcing when neither does.

    Raises:
        ValueError: When the environment names something that is not a guard.
    """
    return options.card_guard or card_guard(os.environ if environ is None else environ)


class JitStoreNotPrivate(ValueError):
    """The compiled-code cache is not a directory of this account's that only it can write, or another account could
    change a directory above it."""


def jit_store_root(options: LocalOptions, environ: Mapping[str, str] | None = None) -> Path:
    """The absolute directory of the shared compiled-code cache: the one ``options`` (or ``HYPERLOOM_JIT_STORE``) names,
    else :data:`DEFAULT_JIT_STORE` and this account's uid below this host's temporary directory.

    The runner starts in a directory of its own, so a relative path is made absolute here, once. A link is not resolved:
    :func:`private_jit_store` judges what stands under the name, and a link there is refused, never followed.
    """
    source = os.environ if environ is None else environ
    default = Path(tempfile.gettempdir(), f"{DEFAULT_JIT_STORE}-{os.getuid()}")
    named = options.jit_store or source.get(JIT_STORE_ENV, "").strip() or default
    return Path(named).expanduser().absolute()


def private_jit_store(root: Path) -> Path:
    """``root``, made (mode 0700) when nothing stands under its name.

    Raises:
        JitStoreNotPrivate: ``root`` is not a directory of this account's (a link to one is refused too), or its group or
            others can write it: the runners load what it holds as compiled code, and the snapshots prove only their own
            content. Another account can plant the default name, which stops the build here but feeds it nothing. Nor
            may a directory above it be another account's or writable by its group or others unless it is sticky (as a
            host's temporary directory is): whoever can change one could move the store aside and put one of their own
            in its place (:func:`~.build_preflight.changeable_by_others`, the build store's check).
    """
    with contextlib.suppress(FileExistsError):  # what already stands there, a link included, is judged below
        root.mkdir(mode=0o700, parents=True)
    if not is_private_dir(root):
        raise JitStoreNotPrivate(
            f"the compiled-code cache {root} is not a directory of this account's that only it can write (nor may it "
            f"be a link to one); name one that is with {JIT_STORE_ENV}"
        )
    problem = changeable_by_others(root, os.getuid())
    if problem is not None:
        raise JitStoreNotPrivate(
            f"users other than this account could change the compiled-code cache {root}: {problem}; name one below "
            f"directories only this account or root can change with {JIT_STORE_ENV}"
        )
    return root


def aiter_builds_missing(runner_envs: Mapping[str, str]) -> str | None:
    """Why the tests of runners with ``runner_envs`` cannot build the aiter modules a ``csrc``/``hsa`` patch touches,
    ``None`` when they can: a runner builds them in the test's own aiter JIT directory, which only the shared
    compiled-code cache gives (the CPU build of touched modules seeds that directory too), unless every module is
    rebuilt (``AITER_REBUILD``)."""
    if runner_envs.get(JIT_CACHE_ENV, "").strip() or build_surface((), runner_envs, {}).rebuild:
        return None
    return f"no JIT store ({JIT_STORE_ENV}) gives its tests an aiter JIT directory to build the touched modules in"


def host_tool_repos(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """The variables of :data:`HOST_TOOL_REPO_ENVS` that ``environ`` (this process's by default) sets to a value.

    A local runner does not inherit the host's environment, yet it clones InferenceX from the repository these name:
    without them a session whose commit exists only in a local mirror fails at the runner's start. Hosted (sandbox)
    runners get theirs from their own template and never from here.
    """
    source = os.environ if environ is None else environ
    return {name: source[name] for name in HOST_TOOL_REPO_ENVS if source.get(name, "").strip()}


def hub_cache(environ: Mapping[str, str] | None = None) -> Path | None:
    """This host's Hugging Face hub cache, where a model's snapshot links point: ``HF_HUB_CACHE``, else ``hub`` below
    ``HF_HOME``, ``$XDG_CACHE_HOME/huggingface`` or ``~/.cache/huggingface``; ``None`` when that is not a directory.
    Never the login files beside it."""
    source = os.environ if environ is None else environ
    if source.get("HF_HUB_CACHE"):
        hub = Path(source["HF_HUB_CACHE"])
    elif source.get("HF_HOME"):
        hub = Path(source["HF_HOME"], _HUB)
    elif source.get("XDG_CACHE_HOME"):
        hub = Path(source["XDG_CACHE_HOME"], _HUGGINGFACE, _HUB)
    elif source.get("HOME"):
        hub = Path(source["HOME"], ".cache", _HUGGINGFACE, _HUB)
    else:
        return None
    return hub if hub.is_absolute() and hub.is_dir() else None


def _named_path(value: str) -> Path | None:
    """The local path a variable's value names (an absolute path, or a ``file://`` URL); ``None`` for anything else."""
    path = value.removeprefix(_FILE_URL) if value.startswith(_FILE_URL) else value
    return Path(path) if os.path.isabs(path) else None


def _real(path: str | os.PathLike[str]) -> Path:
    return Path(os.path.realpath(path))


def _at_or_below(path: Path, tree: Path) -> bool:
    return path == tree or tree in path.parents


def _passwd_home() -> Path | None:
    try:
        return Path(pwd.getpwuid(os.geteuid()).pw_dir)
    except KeyError:
        return None


@dataclass(frozen=True)
class _LinkRule:
    """Where a model entry's link may lead (all paths resolved): never to or below one of the credential
    ``locations``, except below one of the ``homes`` outside its hidden entries, unless a tree the plan reads whole
    anyway (``bound``) holds the link's target and is itself in that credential location."""

    homes: tuple[Path, ...]
    locations: tuple[Path, ...]
    bound: tuple[Path, ...]

    def refusal(self, entry: Path, tree: Path) -> str | None:
        """Why ``entry``'s bind of ``tree`` would show credentials no mask hides (masks open a credential location up
        around a bound path, so nothing in that path is masked); ``None`` when it would not: ``tree`` is in no
        credential location (below a home it may be, outside the hidden entries, ``~/.ssh``, ``~/.config``,
        ``~/.netrc``, where tools keep their logins by name), or each one it is in holds a tree the plan reads whole
        that holds ``tree`` too, whose masks are opened up around it already. A tree the plan reads whole above the
        location (a hub cache that is a home's parent) is no excuse: its masks hide the location, and the link's bind
        would open them up."""
        for where, location in self._credentials(tree):
            if not any(_at_or_below(tree, bound) and _at_or_below(bound, location) for bound in self.bound):
                return f"{entry} links to {tree}, {where}, which its bind would show unmasked"
        return None

    def _credentials(self, tree: Path) -> Iterator[tuple[str, Path]]:
        """Each credential location ``tree`` is at or below, said, with the location: one of ``locations``, or a
        hidden entry of one of ``homes``."""
        for location in dict.fromkeys((*self.locations, *self.homes)):
            if tree == location:
                yield "a credential location", location
            elif location in tree.parents:
                hidden = location / tree.relative_to(location).parts[0]
                if location not in self.homes:
                    yield f"below the credential location {location}", location
                elif hidden.name.startswith("."):
                    yield f"in {hidden}, a hidden entry of a home where tools keep their logins", hidden


def _link_rule(
    model_dir: Path, hub: Path | None, envs: Mapping[str, str], environ: Mapping[str, str] | None
) -> _LinkRule:
    """The rule for the links of the model at ``model_dir``: the homes of the app (``environ``, this process's by
    default, and its passwd entry) and of the runner (``envs``), every credential location of either
    (:func:`~hyperloom.common.confinement.credential_paths` with nothing allowed names each, whole), and the model's
    directory and the hub cache ``hub``, which the plan reads whole."""
    app = os.environ if environ is None else environ
    homes = (app.get("HOME"), envs.get("HOME"), _passwd_home())
    return _LinkRule(
        homes=tuple(dict.fromkeys(_real(home) for home in homes if home and os.path.isabs(home))),
        locations=credential_paths(envs, app_environ=app),
        bound=tuple(_real(tree) for tree in (model_dir, hub) if tree is not None),
    )


def _model_trees(model_dir: Path, rule: _LinkRule) -> tuple[tuple[Path, ...], tuple[str, ...]]:
    """The model's directory and what its entries link to, and why a link of it cannot be bound safely. Each link
    binds its target itself, never the directory it is in (a link to a file in the home would show the whole home),
    except a hub snapshot's links, whose repository's ``blobs`` is bound whole, so that the snapshot's nested links
    resolve too. A link whose bind ``rule`` refuses (one into a credential location: ``~/.ssh``, ``~/.claude``, a
    login file) binds nothing; the reason is kept instead, and a sandbox of the plan runs as before, said at WARNING,
    or is refused in a hosted run."""
    linked, refused = [], []
    with contextlib.suppress(OSError):
        for entry in model_dir.iterdir():
            if not entry.is_symlink():
                continue
            tree = _linked_tree(_real(entry))
            why = rule.refusal(entry, tree)
            if why is None:
                linked.append(tree)
            else:
                refused.append(why)
    return (model_dir, *dict.fromkeys(linked)), tuple(refused)


def _linked_tree(target: Path) -> Path:
    """What a model entry linking to ``target`` (resolved) binds: the hub repository's ``blobs`` directory for a
    blob, else ``target``."""
    blobs = target.parent
    return blobs if blobs.name == _BLOBS and blobs.parent.name.startswith(_HUB_REPOS) else target


def runner_plan(
    session_dir: Path,
    model_dir: Path | None,
    envs: Mapping[str, str],
    python: str,
    environ: Mapping[str, str] | None = None,
) -> RunnerPlan:
    """What a confined runner of the session at ``session_dir`` reads besides its toolchain (its launcher's
    interpreter's, this process's), its own tree and its key: the interpreter ``python`` its template starts it with,
    the trees its template's variables ``envs`` name (:data:`RUNNER_READS`; a repository named by URL names none), the
    session's model and what its entries link to (:func:`_model_trees`), and this host's Hugging Face hub cache
    (:func:`hub_cache` of ``environ``, this process's by default), and the shared tuned tables the framework snapshot
    was taken from (:func:`_snapshot_source`), which each test hashes to show it changed none of them, read-only as
    every other tree here. No bind may show the session's directory whole: it
    holds the blackboard, every sandbox and key, and the leases. A model link into a credential location is refused
    (:attr:`~hyperloom.sandbox.local.RunnerPlan.refused`), so no test sandbox of the plan runs confined with it."""
    named = [_named_path(envs.get(name, "")) for name in RUNNER_READS]
    hub = hub_cache(environ)
    model: tuple[Path, ...] = ()
    refused: tuple[str, ...] = ()
    if model_dir is not None:
        model, refused = _model_trees(model_dir, _link_rule(model_dir, hub, envs, environ))
    found = (_named_path(python), *named, *_snapshot_source(envs), *model, hub)
    return RunnerPlan(
        readable=tuple(dict.fromkeys(path for path in found if path is not None)),
        app_only=(Path(session_dir),),
        refused=refused,
    )


def _snapshot_source(envs: Mapping[str, str]) -> tuple[Path, ...]:
    """The directory of shared tuned tables the framework snapshot ``envs`` names was taken from; none without one."""
    directory = envs.get(SNAPSHOT_ENV, "")
    if not directory:
        return ()
    try:
        return (load_snapshot(Path(directory)).source,)
    except FrameworkBaseError:
        return ()  # no snapshot was taken: each test says so in its result


def _runner_template(session: LocalSession, options: LocalOptions) -> LocalTemplate:
    """The runner template: the image's variables, the host's tool repositories (:func:`host_tool_repos`), ``runner_envs``
    over both, and the variable that names the session's rule pack. ``runner_envs`` add to the image's variables, never
    replace them: without ``ROCM_PATH`` and ``LD_LIBRARY_PATH`` the runner cannot import torch on an image whose ROCm
    libraries are not in the loader's path."""
    envs = {**image_env(), **host_tool_repos(), **(options.runner_envs or {})}
    template = local_runner_template(f"runner-{session.tp}", gpus=session.tp, python=options.python, envs=envs)
    pack = rule_pack_envs(Path(session.session_dir) / _RULE_PACK)
    return replace(template, envs={**template.envs, **pack})


def runner_cache_dir(environ: Mapping[str, str]) -> Path:
    """This host's runner cache as its runners are told it, made absolute: a runner starts elsewhere. It holds the
    inference_suite staging every test runs and the check sets."""
    return cache_root(environ).expanduser().resolve()


def _shared_dirs(jit_store: Path | None, environ: Mapping[str, str]) -> dict[str, str]:
    """The host directories the runners are told, by the variable each reads, for the shared stores: the compiled-code
    cache when it is on, and always the runner cache, which holds the inference_suite staging every test runs. The
    runner's environment is cleared, so without it a runner reads the default cache, where nothing was published."""
    named = {JIT_CACHE_ENV: jit_store, RUNNER_CACHE_ENV: runner_cache_dir(environ)}
    return {name: str(path) for name, path in named.items() if path is not None}


def _measured_template(
    session: LocalSession, template: LocalTemplate, options: LocalOptions, shared_dirs: Mapping[str, str]
) -> LocalTemplate:
    """The runner template that also carries the session's framework tables, the card guard and the directories of the
    shared stores that are on: none is part of the environment the branches' own sandboxes get."""
    framework = framework_envs(Path(session.session_dir) / SESSION_SNAPSHOT_DIR, options.framework_configs)
    guard = {CARD_GUARD_ENV: card_guard_of(options).value}
    return replace(template, envs={**template.envs, **framework, **shared_dirs, **guard})


def _provider(
    session: LocalSession,
    cards: tuple[int, ...],
    template: LocalTemplate,
    options: LocalOptions,
    model_dir: Path | None,
) -> LocalSandboxProvider:
    """The provider of the runner ``template`` (:func:`_measured_template`). Its cards are handed out, to a test's
    sandbox and to a lease alike, under the same guard. Its sandboxes are confined by the runner's plan
    (:func:`runner_plan`) where bubblewrap can sandbox a test.

    Raises:
        IsolationRefused: In a hosted run, where bubblewrap cannot sandbox a test.
    """
    plan = runner_plan(Path(session.session_dir), model_dir, template.envs, options.python)
    provider = LocalSandboxProvider(
        [template],
        gpus=cards,
        base_dir=Path(session.session_dir) / _SANDBOXES,
        hostname=session.hostname,
        users=ProcessUsers(plan=plan),
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
        (kernel_tune_tool(leases, Path(options.framework_configs or image_configs())),) if leases.can_lease(1) else ()
    )
    return (tracelens_tool(), roofline_tool(), UpstreamPrSearch(), *tuner)


def _accepting_local_kinds(acceptance: AcceptanceSetup) -> AcceptanceSetup:
    """``acceptance`` that refuses, before queueing, the kinds the local backend does not run."""
    return replace(acceptance, supported_kinds=SUPPORTED_KINDS)


def _acceptance_with_prompts(
    acceptance: AcceptanceSetup, options: LocalOptions, environ: Mapping[str, str], model_dir: Path | None
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
        short_prompts=staged_short_prompts(named, environ, model_dir, stage=True),
        long_context=staged_long_context(long_named, environ, model_dir, stage=True),
    )


def build_local_toolbox(
    session: LocalSession,
    blackboard: LocalBlackboard,
    acceptance: AcceptanceSetup,
    spec: SpecContext,
    *,
    options: LocalOptions = LocalOptions(),
) -> LocalHostedToolbox:
    """Build a :class:`RealToolbox` whose tests run in local sandboxes; call ``await toolbox.start()`` before use.

    Args:
        session: Where and on what the toolbox runs, see :class:`LocalSession`.
        blackboard: The blackboard; it takes the requests and results, holds the quarantine list and the claims, and is
            searched when a restart recovers unfinished tests.
        acceptance: The repository the parent recipes are read from, the planner (its ``plan`` must return a
            ``hyperloom.measure.points.AcceptedMeasurement``; ``HostedPlanner`` does), the active framework image
            and the campaign's workload, see :class:`~hyperloom_multibranch.toolbox.toolbox.AcceptanceSetup`.
        spec: What the instruction sheet needs besides the call: image, GPU type, commits, blob store.
        options: Everything else, see :class:`LocalOptions`.

    Raises:
        JitStoreNotPrivate: The compiled-code cache is not this account's own directory that only it can write; nothing
            has been started.
        ShortPromptsNotStaged: The short-prompt check is asked for (the option or ``HYPERLOOM_SHORT_PROMPTS``) and its
            set is not staged in this host's runner cache; nothing has been started.
        LongContextNotStaged: The same for the long-context check (the option or ``HYPERLOOM_LONG_CONTEXT``).
        ValueError: The options and the acceptance setup name different short-prompt or long-context sets.
        IsolationRefused: In a hosted run, where bubblewrap cannot sandbox a test; nothing has been started.
    """

    directory, cards, tp = Path(session.session_dir), tuple(session.cards), session.tp
    parallel = max_parallel_tests(len(cards), tp)
    jit_root = private_jit_store(jit_store_root(options))
    acceptance = _acceptance_with_prompts(acceptance, options, os.environ, spec.model_dir)
    native = native_builds(os.environ, spec.blobs, acceptance.active_image_digest or spec.image_digest)
    acceptance = with_native_plan(acceptance, native)
    config = HostedConfig(
        session_dir=directory,
        run_id=session.run_id,
        spec=spec,
        runner_templates={tp: f"runner-{tp}"},
        rule_pack_digest=rule_pack_identity()["digest"],
        archive=session_archive(options.archive, directory),
        max_parallel_tests=parallel,
        poll_interval_sec=options.poll_interval_sec,
        jit_store=DirJitStore(jit_root),
    )
    template = _runner_template(session, options)
    shared_dirs = {**_shared_dirs(jit_root, os.environ), **(native.runner_envs if native else {})}
    acceptance = replace(acceptance, aiter_builds_missing=aiter_builds_missing({**template.envs, **shared_dirs}))
    measured = _measured_template(session, template, options, shared_dirs)
    provider = _provider(session, cards, measured, options, spec.model_dir)
    backend = started_backend(provider, config)
    leases = BranchLeases(provider, directory / _BRANCH_SANDBOXES, tp, template.envs, CardDrain(options.card_memory))
    executor = HostedExecutor(
        backend,
        directory,
        timeout_sec=options.timeout_sec,
        workload_kind=acceptance.workload_kind,
    )
    toolbox = LocalHostedToolbox(
        session_stores(blackboard, directory, session.run_id, native.driver if native else None),
        executor,
        _accepting_local_kinds(acceptance),
        tools=session_tools(_local_tools(leases, options), blackboard, directory),
        tuning=ToolboxTuning(
            policy=capped_policy(options.policy, parallel), max_infra_retries=options.max_infra_retries
        ),
    )
    toolbox.backend, toolbox.provider, toolbox.trace_root = backend, provider, directory
    toolbox.attach_leases(leases)
    toolbox.sandboxes = directory / _SANDBOXES
    toolbox.blobs = spec.blobs
    return toolbox
