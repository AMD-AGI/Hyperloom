# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The real toolbox over the sandbox service: one call builds it from plain values.

Every test runs in a fresh authoritative sandbox the service creates from the runner template of the session's card
count, and is accepted only as a signed bundle that verified against the key the service reported for that sandbox
(:mod:`hyperloom.measure.hosted`). While it runs, its progress is the last log line its runner reports. A branch's agent
works in a ``dev`` sandbox of the same grant (:mod:`hyperloom_multibranch.toolbox.hosted_leases`). The service, not the
toolbox, owns the cards: the toolbox counts the grant's GPUs and gives the ones its leases hold back to nobody else.

:class:`HostedToolboxInputs` holds plain values only, so the host entry (which reads the host's variables and may not
import this package) can pass them as keyword arguments.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from hyperloom.common.outside_facts.rule_pack import rule_pack_identity
from hyperloom.measure.archive import ResultArchive
from hyperloom.measure.framework.base import SESSION_SNAPSHOT_DIR, framework_envs
from hyperloom.measure.hosted import HostedConfig, HostedMeasurementBackend
from hyperloom.measure.hosted_spec import SpecContext
from hyperloom.measure.jit.store import DirJitStore
from hyperloom.measure.runner.cache_layout import MANIFEST_PIN_ENV
from hyperloom.measure.sandboxes import (
    DEFAULT_MAX_PARALLEL_TESTS,
    MEASUREMENT_ROLE,
    RUN_ID_KEY,
)
from hyperloom.sandbox.e2b import E2BSandboxProvider, RetryPolicy
from hyperloom.sandbox.node import node_name
from hyperloom.sandbox.socket import Role, SandboxError, SandboxProvider
from hyperloom_multibranch.blackboard import LocalBlackboard
from hyperloom_multibranch.contracts.blackboard.vocab import LeaseKind
from hyperloom_multibranch.contracts.toolbox import BranchLease, LeaseCharge
from hyperloom_multibranch.toolbox.attempts import DEFAULT_MAX_INFRA_RETRIES, INFRA_BACKOFF_SECONDS
from hyperloom_multibranch.toolbox.hosted_executor import DEFAULT_TEST_TIMEOUT_SEC, HostedExecutor
from hyperloom_multibranch.toolbox.jit_store import REFUSED_HOSTED, named_jit_store, session_jit_store
from hyperloom_multibranch.toolbox.merge_smoke import merge_smoke_from
from hyperloom_multibranch.toolbox.hosted_leases import (
    DevSession,
    GrantLimits,
    HostedBranchLeases,
    LeaseTuning,
)
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
from hyperloom_multibranch.toolbox.resume_guard import MeasuredOn
from hyperloom_multibranch.toolbox.native_builds import base_bins_env, native_builds, with_native_plan
from hyperloom_multibranch.toolbox.long_context import environment_long_context, sees_runner_cache
from hyperloom_multibranch.toolbox.short_prompts import ShortPromptSetup, environment_short_prompts
from hyperloom_multibranch.toolbox.tools import AsyncTool, tracelens_tool
from hyperloom_multibranch.toolbox.toolbox import AcceptanceSetup, ToolboxTuning

__all__ = [
    "DEV_TEMPLATES",
    "MEASUREMENT_TEMPLATES",
    "RUNNER_TEMPLATES",
    "SOURCE_SHA_ENV",
    "HostedOptions",
    "HostedToolbox",
    "HostedToolboxInputs",
    "build_hosted_toolbox",
    "dev_templates",
    "parallel_tests_for",
]

#: The roles of the template list (``HYPERLOOM_SANDBOX_TEMPLATES``) the toolbox reads, named as the grant names them;
#: the others are the sub-runs'. ``runner`` is the older name of ``measurement``, read when ``measurement`` is absent.
MEASUREMENT_TEMPLATES, RUNNER_TEMPLATES, DEV_TEMPLATES = "measurement", "runner", "dev"
#: What the runner's start-up script installs from: the full commit hash that names its entry of the runner cache. It
#: and ``MANIFEST_PIN_ENV`` (when the host has the manifest's digest) are the runner template's accepted create
#: environment, passed on every ``measurement`` create.
SOURCE_SHA_ENV = "HYPERLOOM_SOURCE_SHA"
_MANIFEST_DIGEST = re.compile(r"[0-9a-f]{64}")
#: Where the base builds of the frameworks built from a pinned checkout are laid, below the session directory the runner
#: sandboxes read at the same path.
_BASE_BUILDS = "base-builds"

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class HostedToolboxInputs:
    """What the host gives the toolbox, as plain values: build it with ``HostedToolboxInputs(**kwargs)``.

    Attributes:
        run_id: The run's id, the first part of every sandbox's idempotency key.
        session_dir: The session directory; the backend's journal, the archive and the landed test output are below it.
        api_url: The sandbox service's control plane (``E2B_API_URL``).
        sandbox_url: Its data plane (``E2B_SANDBOX_URL``).
        measurement_key: The API key of the ``measurement`` role, as the string itself. Never in ``repr`` or a log.
        sandbox_templates: The template list as the host gave it (``HYPERLOOM_SANDBOX_TEMPLATES``, parsed). The toolbox
            reads the roles ``measurement`` (or its older name ``runner``, never both) and ``dev``; a role's value maps
            a card count, a decimal string, to the template name the grant issued for it
            (``{"measurement": {"1": "<name>", "8": "<name>"}}``). A name is opaque: its cards are its key, never read
            from the name, so a role given as a bare name or a list says no card count and is refused.
        source_sha: The full lowercase commit hash that names the runner cache entry the runner's start-up script
            installs from (``HYPERLOOM_SOURCE_SHA``).
        tp: The session's tensor-parallel size: the cards of one runner sandbox and of one ``dev`` lease.
        dev_key: The API key of the ``dev`` role. Without it no branch sandbox is leased. Never in ``repr`` or a log.
        grant_limits: The grant's totals, a mapping with all of ``max_sandboxes``, ``max_gpus``, ``gpu_hours`` and
            ``max_runtime_sec`` (``HYPERLOOM_GRANT_LIMITS``, parsed). Without it the toolbox cannot count the grant's
            cards: see :func:`build_hosted_toolbox`.
        runner_cache_manifest_sha256: The SHA-256 of the manifest file of the runner cache entry, 64 lowercase hex
            characters, as the cache builder reported it. Given, it is passed on every ``measurement`` create
            (``HYPERLOOM_RUNNER_CACHE_MANIFEST_SHA256``) and the runner's bootstrap accepts only that manifest; without
            it the manifest vouches only for the entry it sits in.
        session_id: The session's id when ``run_id`` is the run's, which a restart of the run that starts its session
            again shares: it keeps the new session's request ids, and so its sandboxes, apart from the closed one's.
        tests_end_by: When every test of the session must have ended, Unix seconds: the grant's end less what the
            session keeps after its last test, which the session works out
            (:attr:`~hyperloom_multibranch.scheduling.hosted_run.GrantEnd.tests_end_by`). An infrastructure retry never
            waits past one typical test before it (:class:`~.toolbox.ToolboxTuning`). ``None``: no such time.
        sandbox_workspace: The workspace every sandbox is created in (``HYPERLOOM_SANDBOX_WORKSPACE``), sent on each
            create as ``agent-sandbox.io/workspace``. A SaFE key used directly needs it when its user is in more than
            one workspace, and a ``system-admin`` key always does; a Claw grant's key names its own, so a Claw host
            leaves it ``None`` and the creates are as they were.

    Raises:
        ValueError: When ``dev_key`` or ``sandbox_workspace`` is empty, ``tp`` is below one,
            ``runner_cache_manifest_sha256`` is not a SHA-256 or the limits are not a valid grant; the message names the
            field, never a value.
    """

    run_id: str
    session_dir: Path
    api_url: str
    sandbox_url: str
    measurement_key: str = field(repr=False)
    sandbox_templates: Mapping[str, object]
    source_sha: str
    tp: int = 1
    dev_key: str | None = field(default=None, repr=False)
    grant_limits: Mapping[str, float] | None = None
    runner_cache_manifest_sha256: str | None = None
    sandbox_workspace: str | None = None
    session_id: str | None = None
    tests_end_by: float | None = None

    def __post_init__(self) -> None:
        if self.dev_key is not None and not self.dev_key.strip():
            raise ValueError("dev_key is empty; leave it None for a session with no dev sandboxes")
        if self.sandbox_workspace is not None and not self.sandbox_workspace.strip():
            raise ValueError("sandbox_workspace is empty; leave it None for a key that names its own workspace")
        if self.tp < 1:
            raise ValueError("tp must be at least 1")
        pin = self.runner_cache_manifest_sha256
        if pin is not None and not _MANIFEST_DIGEST.fullmatch(pin):
            raise ValueError(
                "runner_cache_manifest_sha256 is not a 64-character lowercase hex SHA-256; leave it None to accept "
                "the manifest of the cache entry itself"
            )
        self.grant()

    def create_envs(self) -> dict[str, str]:
        """The environment every ``measurement`` sandbox is created with: the commit that names its runner cache entry
        and, when the host has it, the digest of the manifest it must be."""
        envs = {SOURCE_SHA_ENV: self.source_sha}
        if self.runner_cache_manifest_sha256 is not None:
            envs[MANIFEST_PIN_ENV] = self.runner_cache_manifest_sha256
        return envs

    def grant(self) -> GrantLimits | None:
        """The limits as :class:`GrantLimits`, or ``None`` when the host gave none."""
        return None if self.grant_limits is None else GrantLimits(**self.grant_limits)  # type: ignore[arg-type]


@dataclass(frozen=True)
class HostedOptions:
    """What :func:`build_hosted_toolbox` may be told besides the inputs, the blackboard, acceptance and the spec.

    ``policy`` is the quota and priority policy; its ``global_concurrency`` is lowered to what the grant runs at once
    when higher. ``archive`` is the result archive, a directory archive below ``session_dir`` when omitted.
    ``timeout_sec`` is the timeout of one attempt, ``boot_budget_sec`` how long, at least, its server may take to be ready
    (``TestCall.boot_budget_sec``; ``None`` for the default share of the timeout), ``max_infra_retries`` the
    infrastructure retries per test before it ends as ``measurement_failed``, ``infra_backoff_seconds`` the wait before each
    one after a failure that is not the node's (:class:`~.attempts.InfraBackoff`) and ``poll_interval_sec`` how often
    the backend polls a runner sandbox for its result.
    ``tools`` are the asynchronous tools ``start_analysis`` may reach; ``None`` is the hosted set, TraceLens: it reads the
    traces a profile test lands below ``session_dir``. ``kernel_tune`` leases a card of a local pool and the roofline
    profiles on host cards, and the hosted toolbox has neither. ``retry`` is how the service client repeats a control-plane
    call, ``lease_tuning`` the lifetime and waits of a branch's ``dev`` sandbox. ``short_prompts`` is the staged prompt set
    of the short-prompt consistency check (``acceptance.short_prompts`` of a builder that names it there says the same and
    must not differ). The set is an entry of the runner cache the runner template mounts (``<cache>/evalsets/<key>``,
    keyed by a hash of its generator and inputs, its prompts named by their SHA-256 in every sheet): the option is
    carried as given (a key or hash that is not well formed is refused) and a runner that does not find that set, or finds
    other prompts under its key, refuses the test, saying so, before it starts a server. ``None`` takes the key in
    ``HYPERLOOM_SHORT_PROMPTS`` of the host environment, whose hash is read from the manifest of the entry in this host's
    own view of the runner cache (``HYPERLOOM_RUNNER_CACHE``) and which refuses the build when it is not staged there;
    ``off`` there runs no check. Without either the check is on with the session model's own set
    (``short_prompts.default_short_prompts``). A host that sees the runner cache (the directory ``HYPERLOOM_RUNNER_CACHE``
    names exists here, as it does for a hosted session, which builds the commit's entry there, in the directory the runner
    sandboxes mount) stages that set there when it is missing, from the model's tokenizer, as the local toolbox does; a
    host that does not see it stages nothing and the check is off, logged. The long-context check is found the same way
    from ``acceptance.long_context``, ``HYPERLOOM_LONG_CONTEXT`` and ``long_context.ServedLongContext`` (the model's own
    set at the context each test's launch serves, staged by a host that sees the runner cache when a test first plans
    that context; a host that does not see it runs no such check and logs why).
    ``framework_configs`` is the directory of aiter's tuned tables
    every test reads private copies of, snapshotted below ``session_dir/framework`` as the toolbox is built, where the
    runner sandboxes read it at the same path: the image's when omitted, ``""`` for none. The snapshot is named to every
    runner sandbox by ``HYPERLOOM_FRAMEWORK_SNAPSHOT`` and ``HYPERLOOM_FRAMEWORK_SESSION_SHA256``, which the runner
    template must therefore accept; without it a test that adds tuned rows to a default table is refused. A session
    whose framework is built from its pinned checkout lays that base build below ``session_dir`` and names it to every
    runner sandbox by ``HYPERLOOM_BASE_BINS``, which the template must accept too: a test that does not rebuild the
    framework runs it, never whatever the sandbox's ``PATH`` finds.
    ``dev_leases`` says whether the session lends ``dev`` sandboxes, to its specialists or its autonomous agents; a
    session that lends none builds no branch leases (``HostedToolbox.leases`` is ``None``) and reads no ``dev`` template,
    whatever ``dev`` key the host gave. A hosted session sets it from its ``dev_leases`` and ``agents``
    (:class:`~hyperloom_multibranch.scheduling.hosted_peers.HostedSandboxes`).
    ``jit_store`` (else ``HYPERLOOM_JIT_STORE``) is the shared compiled-code cache this host checks and prunes as the
    session starts (:func:`~.jit_store.session_jit_store`, which says what happens to one it refuses). Nothing arrives
    in it yet: no sandbox runner reads it or stages a contribution, so nothing is published. It belongs on this host's
    own disk, where no sandbox can write it (a sandbox's code could put snapshots there that later tests load), and the
    ``measurement`` template must neither mount it nor set ``HYPERLOOM_JIT_CACHE``.
    """

    policy: QueuePolicy | None = None
    archive: ResultArchive | None = None
    timeout_sec: int = DEFAULT_TEST_TIMEOUT_SEC
    boot_budget_sec: int | None = None
    max_infra_retries: int = DEFAULT_MAX_INFRA_RETRIES
    infra_backoff_seconds: tuple[float, ...] = INFRA_BACKOFF_SECONDS
    poll_interval_sec: float = 5.0
    tools: Sequence[AsyncTool] | None = None
    retry: RetryPolicy = RetryPolicy()
    lease_tuning: LeaseTuning = LeaseTuning()
    short_prompts: ShortPromptSetup | None = None
    framework_configs: str | None = None
    dev_leases: bool = True
    jit_store: Path | None = None


class HostedToolbox(LeasedToolbox):
    """A :class:`RealToolbox` that also owns its hosted backend and sandbox provider; ``close`` releases them.

    ``leases`` is ``None`` when the session lends no ``dev`` sandboxes, has no grant limits or no ``dev`` key. ``run_id`` is the run's id, which every
    sandbox of the run carries whichever process, container or account of the run created it.
    """

    backend: HostedMeasurementBackend
    provider: SandboxProvider
    leases: HostedBranchLeases | None
    run_id: str
    #: The ``measurement`` sandboxes of the run that were running when :meth:`start` began, so an earlier process's:
    #: ended by the first :meth:`withdraw_all`.
    _earlier: tuple[str, ...] = ()

    async def lease_branch_sandbox(
        self,
        branch_id: str,
        cards: int,
        *,
        wait_sec: float | None = None,
        charge: LeaseCharge | None = None,
        lifetime_sec: float | None = None,
    ) -> BranchLease:
        """Take a ``dev`` sandbox for ``branch_id``'s agent, living as long as ``lifetime_sec`` asks
        (:meth:`HostedBranchLeases.lifetime_sec`)."""

        if self.leases is None:
            return await super().lease_branch_sandbox(branch_id, cards, wait_sec=wait_sec, charge=charge)
        charge = charge or LeaseCharge(LeaseKind.DEV, branch_id)
        return await self.leases.lease(branch_id, cards, charge=charge, wait_sec=wait_sec, lifetime_sec=lifetime_sec)

    async def release_branch_sandbox(self, lease_id: str) -> None:
        if self.leases is not None:
            await self.leases.release(lease_id)

    async def _last_output(self, request_id: str) -> tuple[float, str] | None:
        """The live output the test's runner reported at the backend's latest poll of its sandbox.

        Read in a thread like every other call into the backend: its readings are guarded by a lock the polling threads
        take, which the event loop never waits for.
        """

        return await asyncio.to_thread(self.backend.live_output, request_id)

    async def start(self) -> None:
        """End the lease sandboxes and the measurement sandboxes no unfinished test names that an earlier process of
        this run left behind, note the ``measurement`` sandboxes it left running for those tests, then start as
        :class:`LeasedToolbox` does, which closes the leases' records ``toolbox_lost`` and re-attaches the unfinished
        tests."""

        if self.leases is not None:
            await asyncio.to_thread(self.leases.end_leftovers)
        await asyncio.to_thread(self.backend.end_leftovers)
        self._earlier = await asyncio.to_thread(self._running_measurements)
        await super().start()

    async def withdraw_all(self, reason: str = "session stopped") -> tuple[str, ...]:
        """Withdraw every test as :class:`RealToolbox` does, then, the first time, end the ``measurement`` sandboxes an
        earlier process of the run left running (noted by :meth:`start`).

        A recovered test ends its sandbox only once its attempt re-attached to it, and a test withdrawn before that, or
        still queued behind the grant's slots, never does; nor does a sandbox the service made for a create that process
        did not live to journal. A later leg of the run, in another container and maybe as another account, takes the
        session over by withdrawing what it recovered: without this those sandboxes would measure on for nobody,
        holding the grant's slots and cards until their own timeout. Every test they ran is withdrawn by now."""

        withdrawn = await super().withdraw_all(reason)
        earlier, self._earlier = self._earlier, ()
        if earlier:
            await asyncio.to_thread(self._end_measurements, earlier)
        return withdrawn

    def _running_measurements(self) -> tuple[str, ...]:
        try:
            found = self.provider.find({RUN_ID_KEY: self.run_id}, role=MEASUREMENT_ROLE)
        except SandboxError:
            log.exception("the measurement sandboxes an earlier process of the run left could not be listed")
            return ()
        return tuple(info.sandbox_id for info in found)

    def _end_measurements(self, sandbox_ids: Sequence[str]) -> None:
        for sandbox_id in sandbox_ids:
            self.backend.pool.discard(sandbox_id)  # one already gone is fine, one unreachable is logged
        log.info("ended %d measurement sandbox(es) an earlier process of the run left", len(sandbox_ids))

    async def close(self) -> None:
        """End the tests' bookkeeping, then the lease sandboxes and the measurement sandboxes of this run."""

        await super().close()
        try:
            if self.leases is not None:
                await asyncio.to_thread(self.leases.close)
        finally:
            await asyncio.to_thread(self.backend.close)


def _role_templates(templates: Mapping[str, object], role: str) -> dict[int, str]:
    """Card count to template name for ``role``: none when the list has no such role.

    Raises:
        ValueError: When the role's value is not a mapping of card counts (decimal strings of at least 1) to names, or
            two keys name one card count; the message names the role.
    """
    value = templates.get(role)
    if value is None:
        return {}
    if not isinstance(value, Mapping) or not value:
        raise ValueError(
            f'the {role!r} templates must map card counts to names ({{"<cards>": "<name>"}}); a bare name or a list '
            "does not say its cards"
        )
    by_cards: dict[int, str] = {}
    for key, name in value.items():
        cards = _cards(key)
        if cards is None or not isinstance(name, str) or not name.strip():
            raise ValueError(f"the {role!r} templates must map card counts (decimal, >= 1) to names; got key {key!r}")
        if by_cards.setdefault(cards, name) != name:
            raise ValueError(f"the {role!r} templates name {cards} cards twice")
    return by_cards


def _cards(key: object) -> int | None:
    if isinstance(key, str) and key.isascii() and key.isdecimal() and int(key) >= 1:
        return int(key)
    return None


def _measurement_templates(templates: Mapping[str, object]) -> dict[int, str]:
    """The ``measurement`` role's templates, read from ``runner`` when only that older name is given.

    Raises:
        ValueError: When both names are given, or neither.
    """
    if MEASUREMENT_TEMPLATES in templates and RUNNER_TEMPLATES in templates:
        raise ValueError(
            f"the template list gives both {MEASUREMENT_TEMPLATES!r} and its older name {RUNNER_TEMPLATES!r}; "
            "which one the tests run on is ambiguous"
        )
    role = RUNNER_TEMPLATES if RUNNER_TEMPLATES in templates else MEASUREMENT_TEMPLATES
    by_cards = _role_templates(templates, role)
    if not by_cards:
        raise ValueError(f"the template list has no {MEASUREMENT_TEMPLATES!r} role")
    return by_cards


def _runner_template(inputs: HostedToolboxInputs) -> str:
    by_cards = _measurement_templates(inputs.sandbox_templates)
    if inputs.tp not in by_cards:
        raise ValueError(
            f"no {MEASUREMENT_TEMPLATES!r} template has the session's {inputs.tp} cards; templates serve {sorted(by_cards)}"
        )
    return by_cards[inputs.tp]


def _service_provider(inputs: HostedToolboxInputs, retry: RetryPolicy) -> E2BSandboxProvider:
    keys: dict[Role, str] = {MEASUREMENT_ROLE: inputs.measurement_key}
    if inputs.dev_key is not None:
        keys["dev"] = inputs.dev_key
    return E2BSandboxProvider(
        keys, api_url=inputs.api_url, sandbox_url=inputs.sandbox_url, retry=retry, workspace=inputs.sandbox_workspace
    )


def _leases(provider: SandboxProvider, inputs: HostedToolboxInputs, tuning: LeaseTuning) -> HostedBranchLeases | None:
    """The branch sandbox leases: offered only where the grant is known to count their cards and a ``dev`` key makes
    them."""
    grant = inputs.grant()
    if grant is None or inputs.dev_key is None:
        return None
    return HostedBranchLeases(
        provider, grant, DevSession(inputs.run_id, inputs.tp, dev_templates(inputs.sandbox_templates)), tuning
    )


def dev_templates(templates: Mapping[str, object]) -> dict[int, str]:
    """Card count to ``dev`` template name in the host's template list ``templates``: none when it has no ``dev`` role.

    Raises:
        ValueError: When the ``dev`` role does not map card counts to names, naming the role.
    """

    return _role_templates(templates, DEV_TEMPLATES)


def parallel_tests_for(grant: GrantLimits | None, tp: int) -> int:
    """How many tests of TP=``tp`` may run at once under ``grant``: ``min(max_gpus // tp, max_sandboxes)``, at least 1.

    This is the one source of the grant's parallel-tests rule; the hosted toolbox sizes its
    measurement pool with it, and anything else that needs the number (a scheduler's cap, a planner) calls it rather
    than repeating the arithmetic. Without a grant it is the pool's default.

    Raises:
        ValueError: When the grant has fewer GPUs than one test of TP=``tp`` needs.
    """
    if grant is None:
        return DEFAULT_MAX_PARALLEL_TESTS
    if grant.max_gpus < tp:
        raise ValueError(f"a grant of {grant.max_gpus} GPUs cannot run a test of TP={tp}")
    return max(1, min(grant.max_gpus // tp, grant.max_sandboxes))


def _acceptance_with_prompts(
    acceptance: AcceptanceSetup, options: HostedOptions, environ: Mapping[str, str], model_dir: Path | None
) -> AcceptanceSetup:
    """``acceptance`` with the short-prompt set the options (else the acceptance setup, else ``HYPERLOOM_SHORT_PROMPTS``)
    name, and the long-context set the acceptance setup (else ``HYPERLOOM_LONG_CONTEXT``, else the model's own set) names.

    The model's own sets are staged in the runner cache when they are missing and this host sees it
    (:func:`~.long_context.sees_runner_cache`): the runners read that directory, and nothing else puts the sets there.

    Raises:
        ShortPromptsNotStaged: The environment names a key that is not staged in this host's view of the runner cache.
        LongContextNotStaged: The same for the long-context check.
        ValueError: The options and the acceptance setup name different sets.
    """
    stage = sees_runner_cache(environ)
    named = with_short_prompts(acceptance, options.short_prompts)
    if named.short_prompts is None:
        named = with_short_prompts(named, environment_short_prompts(environ, model_dir, stage=stage))
    if named.long_context is None:
        named = with_long_context(named, environment_long_context(environ, model_dir, stage=stage))
    return named


def _session_jit_store(options: HostedOptions, session_dir: Path) -> DirJitStore | None:
    """The compiled-code cache this host publishes to, checked and pruned; ``None`` when none is named or it is refused."""
    root = named_jit_store(options.jit_store, os.environ)
    return session_jit_store(root, session_dir, REFUSED_HOSTED) if root is not None else None


def build_hosted_toolbox(
    inputs: HostedToolboxInputs,
    blackboard: LocalBlackboard,
    acceptance: AcceptanceSetup,
    spec: SpecContext,
    *,
    options: HostedOptions = HostedOptions(),
) -> HostedToolbox:
    """Build a :class:`RealToolbox` whose tests run in sandboxes of the sandbox service; call ``await toolbox.start()``
    before use and ``await toolbox.close()`` after.

    At most ``min(max_gpus // tp, max_sandboxes)`` tests run at once. The cards are counted only for a
    session that lends ``dev`` sandboxes (:attr:`HostedOptions.dev_leases`) and was given grant limits and a ``dev``
    key: the grant's GPUs are the cards, ``measurement_capacity`` reports them, with the cards of the branch sandboxes
    leased as ``leased_cards``, and ``lease_branch_sandbox`` leases them. Any other session -- one that lends none,
    whatever limits the grant has, included -- reports its capacity as unknown and ``lease_branch_sandbox`` refuses
    with ``unsupported_kind``. Without grant limits at most as many tests run at once as the sandbox pool's default;
    the service's own quota still bounds them (a refused create waits).
    Sandboxes of other runs are never touched.

    A session that reads another rule pack than the runner's has its results refused as ``rule_pack_mismatch``: the
    runner reads the pack of the commit it installs, and the backend takes only results that read the session's.

    Args:
        inputs: What the host gave, see :class:`HostedToolboxInputs`.
        blackboard: The blackboard; it takes the requests and results, holds the quarantine list and the claims, and is
            searched when a restart recovers unfinished tests.
        acceptance: The repository the parent recipes are read from, the planner (its ``plan`` must return a
            ``hyperloom.measure.points.AcceptedMeasurement``; ``HostedPlanner`` does), the active framework image
            and the campaign's workload, see :class:`~hyperloom_multibranch.toolbox.toolbox.AcceptanceSetup`.
        spec: What the instruction sheet needs besides the call: image, GPU type, commits, blob store.
        options: Everything else, see :class:`HostedOptions`.

    Raises:
        ShortPromptsNotStaged: ``HYPERLOOM_SHORT_PROMPTS`` names a set that is not staged in this host's runner cache; nothing
            has been started.
        ValueError: When the templates have no runner template for ``tp`` (or no ``dev`` one, when the session lends
            ``dev`` sandboxes and the host gave a ``dev`` key and grant limits),
            the grant cannot run one test, or the options and the acceptance setup name different short-prompt sets.
    """

    directory = Path(inputs.session_dir)
    acceptance = _acceptance_with_prompts(acceptance, options, os.environ, spec.model_dir)
    native = native_builds(os.environ, spec.blobs, acceptance.active_image_digest or spec.image_digest)
    acceptance = with_native_plan(acceptance, native)
    grant = inputs.grant()
    # A grant that cannot run one test is refused before the templates are read.
    parallel = parallel_tests_for(grant, inputs.tp)
    runner = _runner_template(inputs)  # refused before the lease templates are read
    provider = _service_provider(inputs, options.retry)
    leases = _leases(provider, inputs, options.lease_tuning) if options.dev_leases else None
    config = HostedConfig(
        session_dir=directory,
        run_id=inputs.run_id,
        spec=spec,
        runner_templates={inputs.tp: runner},
        rule_pack_digest=rule_pack_identity()["digest"],
        create_envs={
            **inputs.create_envs(),
            **framework_envs(directory / SESSION_SNAPSHOT_DIR, options.framework_configs),
            **(native.runner_envs if native else {}),
            **base_bins_env(native, directory / _BASE_BUILDS),  # built once the inputs are known to run a test
        },
        archive=session_archive(options.archive, directory),
        max_parallel_tests=parallel,
        poll_interval_sec=options.poll_interval_sec,
        jit_store=_session_jit_store(options, directory),
    )
    backend = started_backend(provider, config)
    executor = HostedExecutor(
        backend,
        directory,
        timeout_sec=options.timeout_sec,
        boot_budget_sec=options.boot_budget_sec,
        workload_kind=acceptance.workload_kind,
    )
    executor.smoke = merge_smoke_from(os.environ)
    toolbox = HostedToolbox(
        session_stores(
            blackboard, directory, inputs.run_id, native.driver if native else None, inputs.session_id or ""
        ),
        executor,
        acceptance,
        tools=session_tools(options.tools if options.tools is not None else (tracelens_tool(),), blackboard, directory),
        tuning=ToolboxTuning(
            policy=capped_policy(options.policy, parallel),
            max_infra_retries=options.max_infra_retries,
            infra_backoff_seconds=options.infra_backoff_seconds,
            tests_end_by=inputs.tests_end_by,
        ),
    )
    toolbox.backend, toolbox.provider, toolbox.trace_root, toolbox.blobs = backend, provider, directory, spec.blobs
    toolbox.run_id = inputs.run_id
    # The runner template's cards, not the app's node: a restart reschedules the app, and its tests run in the grant.
    toolbox.measured_on = MeasuredOn(directory, node_name(), spec.image_digest, inputs.tp)
    if leases is not None:
        toolbox.attach_leases(leases)
    return toolbox
