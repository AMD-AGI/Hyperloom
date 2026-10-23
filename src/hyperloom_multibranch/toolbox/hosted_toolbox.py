# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The real toolbox over the sandbox service: one call builds it from plain values (handover 3.3, MB2).

Every test runs in a fresh authoritative sandbox the service creates from the runner template of the session's card
count, and is accepted only as a signed bundle that verified against the key the service reported for that sandbox
(:mod:`hyperloom.measure.hosted`). A branch's agent works in a ``dev`` sandbox of the same grant
(:mod:`hyperloom_multibranch.toolbox.hosted_leases`). The service, not the toolbox, owns the cards: the toolbox counts
the grant's GPUs and gives the ones its leases hold back to nobody else.

:class:`HostedToolboxInputs` holds plain values only, so the host entry (which reads the host's variables and may not
import this package) can pass them as keyword arguments.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from hyperloom.common.rule_pack import rule_pack_identity
from hyperloom.measure.archive import ResultArchive
from hyperloom.measure.hosted import HostedConfig, HostedMeasurementBackend
from hyperloom.measure.hosted_spec import SpecContext
from hyperloom.measure.sandboxes import DEFAULT_MAX_PARALLEL_TESTS, MEASUREMENT_ROLE
from hyperloom.sandbox.e2b import E2BSandboxProvider, RetryPolicy
from hyperloom.sandbox.socket import Role, SandboxProvider
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.toolbox.attempts import DEFAULT_MAX_INFRA_RETRIES
from hyperloom_multibranch.toolbox.hosted_executor import DEFAULT_TEST_TIMEOUT_SEC, HostedExecutor
from hyperloom_multibranch.toolbox.hosted_leases import (
    DevSession,
    GrantLimits,
    HostedBranchLeases,
    LeaseTuning,
    templates_by_cards,
)
from hyperloom_multibranch.toolbox.leased_toolbox import (
    LeasedToolbox,
    capped_policy,
    session_archive,
    session_stores,
    session_tools,
    started_backend,
)
from hyperloom_multibranch.toolbox.queue import QueuePolicy
from hyperloom_multibranch.toolbox.tools import AsyncTool, tracelens_tool
from hyperloom_multibranch.toolbox.toolbox import AcceptanceSetup, ToolboxTuning

__all__ = [
    "DEV_TEMPLATES",
    "RUNNER_TEMPLATES",
    "SOURCE_REPO_ENV",
    "SOURCE_SHA_ENV",
    "HostedOptions",
    "HostedToolbox",
    "HostedToolboxInputs",
    "build_hosted_toolbox",
]

#: The roles of the template list (``HYPERLOOM_SANDBOX_TEMPLATES``) the toolbox reads; the others are the sub-runs'.
RUNNER_TEMPLATES, DEV_TEMPLATES = "runner", "dev"
#: What the runner's start-up script installs from: the repository and the full commit hash. They are the runner
#: template's accepted create environment, passed on every ``measurement`` create.
SOURCE_REPO_ENV, SOURCE_SHA_ENV = "HYPERLOOM_SOURCE_REPO", "HYPERLOOM_SOURCE_SHA"
_COMMIT = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_GRANT_KEYS = ("max_sandboxes", "max_gpus", "gpu_hours", "max_runtime_sec")
_REQUIRED_TEXT = ("run_id", "api_url", "sandbox_url", "measurement_key", "source_repo")


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
            reads the roles ``runner`` and ``dev``; a role's value is one name, a list of names or a mapping whose values
            are names, and each name carries its card count as the suffix ``-g<cards>`` (``ar-01ja-runner-g8``).
        source_repo: The repository the runner's start-up script fetches its code from (``HYPERLOOM_SOURCE_REPO``).
        source_sha: The full lowercase commit hash it checks out (``HYPERLOOM_SOURCE_SHA``).
        tp: The session's tensor-parallel size: the cards of one runner sandbox and of one ``dev`` lease.
        dev_key: The API key of the ``dev`` role. Without it no branch sandbox is leased. Never in ``repr`` or a log.
        grant_limits: The grant's totals, a mapping with all of ``max_sandboxes``, ``max_gpus``, ``gpu_hours`` and
            ``max_runtime_sec`` (``HYPERLOOM_GRANT_LIMITS``, parsed). Without it the toolbox cannot count the grant's
            cards: see :func:`build_hosted_toolbox`.

    Raises:
        ValueError: When a text is empty, ``tp`` is below one, ``source_sha`` is not a full commit hash or the limits are
            not a valid set; the message names the field, never a value.
    """

    run_id: str
    session_dir: Path
    api_url: str
    sandbox_url: str
    measurement_key: str = field(repr=False)
    sandbox_templates: Mapping[str, object]
    source_repo: str
    source_sha: str
    tp: int = 1
    dev_key: str | None = field(default=None, repr=False)
    grant_limits: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        for name in _REQUIRED_TEXT:
            if not getattr(self, name).strip():
                raise ValueError(f"{name} is empty")
        if self.dev_key is not None and not self.dev_key.strip():
            raise ValueError("dev_key is empty; leave it None for a session with no dev sandboxes")
        if self.tp < 1:
            raise ValueError("tp must be at least 1")
        if not _COMMIT.fullmatch(self.source_sha):
            raise ValueError("source_sha is not a full lowercase commit hash")
        self.grant()

    def grant(self) -> GrantLimits | None:
        """The limits as :class:`GrantLimits`, or ``None`` when the host gave none."""
        limits = self.grant_limits
        if limits is None:
            return None
        missing = [key for key in _GRANT_KEYS if key not in limits]
        if missing:
            raise ValueError(f"grant_limits has no {', '.join(missing)}")
        return GrantLimits(
            _whole(limits, "max_sandboxes"),
            _whole(limits, "max_gpus"),
            float(limits["gpu_hours"]),
            _whole(limits, "max_runtime_sec"),
        )


def _whole(limits: Mapping[str, float], key: str) -> int:
    if float(limits[key]) != int(limits[key]):
        raise ValueError(f"grant_limits {key} is not a whole number")
    return int(limits[key])


@dataclass(frozen=True)
class HostedOptions:
    """What :func:`build_hosted_toolbox` may be told besides the inputs, the notebook, acceptance and the spec.

    ``policy`` is the quota and priority policy; its ``global_concurrency`` is lowered to what the grant runs at once
    when higher. ``archive`` is the result archive, a directory archive below ``session_dir`` when omitted.
    ``timeout_sec`` is the timeout of one attempt, ``max_infra_retries`` the infrastructure retries per test before it
    ends as ``measurement_failed`` and ``poll_interval_sec`` how often the backend polls a runner sandbox for its result.
    ``tools`` are the asynchronous tools ``start_analysis`` may reach; ``None`` is the hosted set, TraceLens: it reads the
    traces a profile test lands below ``session_dir``. ``kernel_tune`` leases a card of a local pool and the roofline
    profiles on host cards, and the hosted toolbox has neither. ``retry`` is how the service client repeats a control-plane
    call, ``lease_tuning`` the lifetime and waits of a branch's ``dev`` sandbox.
    """

    policy: QueuePolicy | None = None
    archive: ResultArchive | None = None
    timeout_sec: int = DEFAULT_TEST_TIMEOUT_SEC
    max_infra_retries: int = DEFAULT_MAX_INFRA_RETRIES
    poll_interval_sec: float = 5.0
    tools: Sequence[AsyncTool] | None = None
    retry: RetryPolicy = RetryPolicy()
    lease_tuning: LeaseTuning = LeaseTuning()


class HostedToolbox(LeasedToolbox):
    """A :class:`RealToolbox` that also owns its hosted backend and sandbox provider; ``close`` releases them.

    ``leases`` is ``None`` when the session has no grant limits or no ``dev`` key.
    """

    backend: HostedMeasurementBackend
    provider: SandboxProvider
    leases: HostedBranchLeases | None

    async def release_branch_sandbox(self, lease_id: str) -> None:
        if self.leases is not None:
            await self.leases.release(lease_id)

    async def close(self) -> None:
        """End the tests' bookkeeping, then the lease sandboxes and the measurement sandboxes of this run."""

        await super().close()
        try:
            if self.leases is not None:
                await asyncio.to_thread(self.leases.close)
        finally:
            await asyncio.to_thread(self.backend.close)


def _names(templates: Mapping[str, object], role: str) -> list[str]:
    """The template names of ``role``: none when the list has no such role."""
    value = templates.get(role)
    if value is None:
        return []
    if isinstance(value, str):
        names: list[object] = [value]
    elif isinstance(value, Mapping):
        names = list(value.values())
    elif isinstance(value, (list, tuple)):
        names = list(value)
    else:
        names = [value]
    if not all(isinstance(name, str) and name for name in names):
        raise ValueError(f"the {role!r} templates must be names: a name, or a list or mapping of names")
    return names  # type: ignore[return-value]


def _runner_template(inputs: HostedToolboxInputs) -> str:
    by_cards = templates_by_cards(_names(inputs.sandbox_templates, RUNNER_TEMPLATES))
    if inputs.tp not in by_cards:
        raise ValueError(f"no runner template has the session's {inputs.tp} cards; templates serve {sorted(by_cards)}")
    return by_cards[inputs.tp]


def _service_provider(inputs: HostedToolboxInputs, retry: RetryPolicy) -> E2BSandboxProvider:
    keys: dict[Role, str] = {MEASUREMENT_ROLE: inputs.measurement_key}
    if inputs.dev_key is not None:
        keys["dev"] = inputs.dev_key
    return E2BSandboxProvider(keys, api_url=inputs.api_url, sandbox_url=inputs.sandbox_url, retry=retry)


def _leases(provider: SandboxProvider, inputs: HostedToolboxInputs, tuning: LeaseTuning) -> HostedBranchLeases | None:
    """The branch sandbox leases: offered only where the grant is known to count their cards and a ``dev`` key makes
    them."""
    grant = inputs.grant()
    if grant is None or inputs.dev_key is None:
        return None
    dev = templates_by_cards(_names(inputs.sandbox_templates, DEV_TEMPLATES))
    return HostedBranchLeases(provider, grant, DevSession(inputs.run_id, inputs.tp, dev), tuning)


def _parallel_tests(grant: GrantLimits | None, tp: int) -> int:
    """At most as many tests at once as the grant's GPUs and sandboxes allow; the pool's default without a grant."""
    if grant is None:
        return DEFAULT_MAX_PARALLEL_TESTS
    if grant.max_gpus < tp:
        raise ValueError(f"a grant of {grant.max_gpus} GPUs cannot run a test of TP={tp}")
    return max(1, min(grant.max_gpus // tp, grant.max_sandboxes))


def build_hosted_toolbox(
    inputs: HostedToolboxInputs,
    notebook: LocalNotebook,
    acceptance: AcceptanceSetup,
    spec: SpecContext,
    *,
    options: HostedOptions = HostedOptions(),
) -> HostedToolbox:
    """Build a :class:`RealToolbox` whose tests run in sandboxes of the sandbox service; call ``await toolbox.start()``
    before use and ``await toolbox.close()`` after.

    The grant's GPUs are the cards: ``measurement_capacity`` reports them, with the cards of the branch sandboxes
    leased as ``leased_cards``, and at most ``min(max_gpus // tp, max_sandboxes)`` tests run at once. Without grant
    limits the toolbox cannot count the cards: the capacity is unknown, ``lease_branch_sandbox`` refuses with
    ``unsupported_kind``, and at most as many tests run at once as the sandbox pool's default; the service's own quota
    still bounds them (a refused create waits). Sandboxes of other runs are never touched.

    A session that reads another rule pack than the runner's has its results refused as ``rule_pack_mismatch``: the
    runner reads the pack of the commit it installs, and the backend takes only results that read the session's.

    Args:
        inputs: What the host gave, see :class:`HostedToolboxInputs`.
        notebook: The notebook; it takes the requests and results, holds the quarantine list and the claims, and is
            searched when a restart recovers unfinished tests.
        acceptance: The repository the parent recipes are read from, the planner (its ``plan`` must return a
            ``hyperloom.measure.points.AcceptedMeasurement``; ``HostedPlanner`` does), the active framework image
            and the campaign's workload, see :class:`~hyperloom_multibranch.toolbox.toolbox.AcceptanceSetup`.
        spec: What the instruction sheet needs besides the call: image, GPU type, commits, blob store.
        options: Everything else, see :class:`HostedOptions`.

    Raises:
        ValueError: When the templates have no runner template for ``tp`` (or no ``dev`` one, when leases are offered)
            or the grant cannot run one test.
    """

    directory = Path(inputs.session_dir)
    grant = inputs.grant()
    parallel = _parallel_tests(grant, inputs.tp)
    config = HostedConfig(
        session_dir=directory,
        run_id=inputs.run_id,
        spec=spec,
        runner_templates={inputs.tp: _runner_template(inputs)},
        rule_pack_digest=rule_pack_identity()["digest"],
        create_envs={SOURCE_REPO_ENV: inputs.source_repo, SOURCE_SHA_ENV: inputs.source_sha},
        archive=session_archive(options.archive, directory),
        max_parallel_tests=parallel,
        poll_interval_sec=options.poll_interval_sec,
    )
    provider = _service_provider(inputs, options.retry)
    leases = _leases(provider, inputs, options.lease_tuning)
    backend = started_backend(provider, config)
    executor = HostedExecutor(
        backend, directory, timeout_sec=options.timeout_sec, workload_kind=acceptance.workload_kind
    )
    toolbox = HostedToolbox(
        session_stores(notebook, directory, inputs.run_id),
        executor,
        acceptance,
        tools=session_tools(options.tools if options.tools is not None else (tracelens_tool(),), notebook, directory),
        tuning=ToolboxTuning(
            policy=capped_policy(options.policy, parallel), max_infra_retries=options.max_infra_retries
        ),
    )
    toolbox.backend, toolbox.provider, toolbox.trace_root, toolbox.blobs = backend, provider, directory, spec.blobs
    if leases is not None:
        toolbox.attach_leases(leases)
    return toolbox
