# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``build_hosted_toolbox`` from plain values, over the E2B stand-in and the fake runner.

The stand-in runs real envd processes behind a simulated control plane. It writes no platform metadata, so the tests put
a platform's part in front of the E2B provider: the signing key the service reports for a runner sandbox (the fake runner
signs with it) and its node. The tests that need the stand-in skip where it cannot run (root, ``unshare``, an envd binary
named by ``HYPERLOOM_E2B_ENVD``); the ones that only build the toolbox always run.
"""

from __future__ import annotations

import asyncio
import functools
import os
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

import hyperloom
from hyperloom.common.rule_pack import rule_pack_identity
from hyperloom.contracts.measurement import TestResult, Validity
from hyperloom.measure import signing
from hyperloom.measure.hosted_log import STATE_SENT
from hyperloom.measure.sandboxes import BRANCH_ID_KEY, PURPOSE_KEY
from hyperloom.measure.tests.hosted_fixtures import FAKE_RUNNER, point_yaml, spec_context
from hyperloom.measure.tests.rule_pack_fixtures import identity_of, other_pack, reading_pack  # noqa: F401
from hyperloom.sandbox.e2b import E2BSandboxProvider, RetryPolicy
from hyperloom.sandbox.socket import (
    IDEMPOTENCY_KEY,
    NODE_NAME_KEY,
    SIGNING_PUBLIC_KEY,
    CreateOptions,
    Role,
    Sandbox,
    SandboxInfo,
)
from hyperloom.sandbox.tests.e2b_standin import (
    ENVD_ENV,
    StandIn,
    StandInTemplate,
    interpreter_binds,
    unavailable_reason,
)
from hyperloom_multibranch.contracts.notebook import Role as NotebookRole, Writer
from hyperloom_multibranch.contracts.toolbox import MeasurementCapacity, RequestRejected, WorkloadKind
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.toolbox import hosted_toolbox
from hyperloom_multibranch.toolbox.hosted_leases import LEASE_PURPOSE, RUN_ID_KEY, LeaseTuning
from hyperloom_multibranch.toolbox.hosted_toolbox import (
    SOURCE_REPO_ENV,
    SOURCE_SHA_ENV,
    HostedOptions,
    HostedToolbox,
    HostedToolboxInputs,
    build_hosted_toolbox,
)
from hyperloom_multibranch.toolbox.queue import QueuePolicy
from hyperloom_multibranch.toolbox.tests.helpers import IMAGE, FakeRepo, req
from hyperloom_multibranch.toolbox.tests.test_hosted_executor import PointPlanner
from hyperloom_multibranch.toolbox.toolbox import AcceptanceSetup
from hyperloom_multibranch.toolbox.tools import TRACELENS_TOOL

_RUNNERS = ("run-runner-g1", "run-runner-g2")
_DEVS = ("run-dev-g1", "run-dev-g2")
_SOURCE_REPO, _SOURCE_SHA = "https://source.invalid/hyperloom.git", "ab" * 20
_GRANT = {"max_sandboxes": 8, "max_gpus": 8, "gpu_hours": 400.0, "max_runtime_sec": 7200}
_SOURCE_FILE = "/out/source-envs.txt"
#: The runner template's start command: note the source the sandbox was given, then start the fake runner.
_START = (
    "/bin/sh",
    "-c",
    f'printenv {SOURCE_REPO_ENV} {SOURCE_SHA_ENV} > {_SOURCE_FILE}; exec "$0" "$@"',
    *FAKE_RUNNER,
)
_NODE = "standin-node"
_FAST = LeaseTuning(wait_sec=20.0, first_backoff_sec=0.02, max_backoff_sec=0.05)
_RETRY = RetryPolicy(budget_sec=3.0, first_delay_sec=0.02, max_delay_sec=0.1)
_HANG = {"fake_mode": "hang"}


@dataclass
class Created:
    """One create the toolbox asked the service for."""

    template: str
    role: Role
    metadata: Mapping[str, str]
    options: CreateOptions


class _Reporting:
    """A sandbox as the platform reports it: a runner's info carries the signing key and the node."""

    def __init__(self, inner: Sandbox, platform: Mapping[str, str]) -> None:
        self._inner = inner
        self._platform = platform
        self.sandbox_id = inner.sandbox_id

    def info(self) -> SandboxInfo:
        info = self._inner.info()
        return SandboxInfo(info.sandbox_id, info.template, {**info.metadata, **self._platform})

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _Platform(hosted_toolbox._RunProvider):
    """The toolbox's provider plus the platform's metadata on ``measurement`` sandboxes, remembering every create it is
    asked for (before the run's id and the source are added to a measurement create)."""

    def __init__(self, keys: Mapping[Role, str], *, public_key: str, creates: list[Created], **service: Any) -> None:
        super().__init__(keys, **service)
        self._platform = {SIGNING_PUBLIC_KEY: public_key, NODE_NAME_KEY: _NODE}
        self._creates = creates

    def create(
        self,
        template: str,
        *,
        role: Role,
        timeout_sec: int,
        metadata: Mapping[str, str],
        options: CreateOptions = CreateOptions(),
    ) -> Sandbox:
        self._creates.append(Created(template, role, dict(metadata), options))
        made = super().create(template, role=role, timeout_sec=timeout_sec, metadata=metadata, options=options)
        return _Reporting(made, self._platform if role == "measurement" else {})

    def connect(self, sandbox_id: str, *, role: Role) -> Sandbox:
        return _Reporting(super().connect(sandbox_id, role=role), self._platform if role == "measurement" else {})


@dataclass
class Service:
    """A stand-in with runner and ``dev`` templates, a way to look at it from outside, and what the toolbox asked of it."""

    standin: StandIn
    creates: list[Created] = field(default_factory=list)

    def observer(self) -> E2BSandboxProvider:
        """A client that is not the toolbox's: for what another run does and for looking at what is running."""
        return E2BSandboxProvider(self.standin.keys, api_url=self.standin.api_url, sandbox_url=self.standin.sandbox_url)

    def running(self, role: Role, run_id: str | None = None) -> list[str]:
        where = {} if run_id is None else {RUN_ID_KEY: run_id}
        return [info.sandbox_id for info in self.observer().find(where, role=role)]


def _runner_template(name: str, key_file: Path) -> StandInTemplate:
    source = Path(hyperloom.__file__).resolve().parents[1]
    return StandInTemplate(
        name,
        "measurement",
        start=_START,
        binds=(*interpreter_binds(), str(source), str(key_file.parent)),
        envs={
            "HYPERLOOM_SANDBOX_ROOT": "/",
            "HYPERLOOM_SIGNING_KEY_FILE": str(key_file),
            "PYTHONPATH": str(source),
        },
    )


@pytest.fixture
def service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Service]:
    """The stand-in, and the toolbox's provider replaced by one that reports what the platform would."""
    reason = unavailable_reason(os.environ.get(ENVD_ENV))
    if reason is not None:
        pytest.skip(reason)
    key = signing.generate_private_key()
    key_file = tmp_path / "keys" / "signing.pem"
    key_file.parent.mkdir()
    key_file.write_bytes(signing.private_key_pem(key))
    key_file.chmod(0o600)
    templates = [*(_runner_template(name, key_file) for name in _RUNNERS), *(StandInTemplate(n, "dev") for n in _DEVS)]
    workdir = tmp_path / "standin"
    workdir.mkdir()
    with ExitStack() as stack:
        standin = stack.enter_context(StandIn(Path(os.environ[ENVD_ENV]), workdir, templates))
        made = Service(standin)
        provider = functools.partial(_Platform, public_key=signing.public_key_pem(key), creates=made.creates)
        monkeypatch.setattr(hosted_toolbox, "_RunProvider", provider)
        yield made


def _inputs(service: Service | None, tmp_path: Path, **changed: Any) -> HostedToolboxInputs:
    keys: Mapping[Role, str] = service.standin.keys if service else {"measurement": "m-key", "dev": "d-key"}
    values: dict[str, Any] = {
        "run_id": "run-1",
        "session_dir": tmp_path / "session",
        "api_url": service.standin.api_url if service else "http://service.invalid",
        "sandbox_url": service.standin.sandbox_url if service else "http://sandboxes.invalid",
        "measurement_key": keys["measurement"],
        "dev_key": keys["dev"],
        "sandbox_templates": {"runner": list(_RUNNERS), "dev": {"1": _DEVS[0], "2": _DEVS[1]}, "subrun": "x-sub"},
        "source_repo": _SOURCE_REPO,
        "source_sha": _SOURCE_SHA,
        "tp": 1,
        "grant_limits": _GRANT,
    }
    return HostedToolboxInputs(**{**values, **changed})


def _toolbox(
    service: Service | None, tmp_path: Path, *, planned: Mapping[str, Any] | None = None, **changed: Any
) -> HostedToolbox:
    """The toolbox for ``_inputs(**changed)``; ``planned`` is what the fake runner is told to do (the point's YAML)."""
    point = point_yaml(tmp_path, "planned.yaml", **(planned or {}))
    options = changed.pop("options", None) or HostedOptions(
        policy=QueuePolicy(per_branch_k=2, global_concurrency=64),
        poll_interval_sec=0.05,
        timeout_sec=60,
        max_infra_retries=0,
        retry=_RETRY,
        lease_tuning=_FAST,
    )
    return build_hosted_toolbox(
        _inputs(service, tmp_path, **changed),
        LocalNotebook(tmp_path / "notebook", Writer(NotebookRole.TOOLBOX, "toolbox")),
        AcceptanceSetup(FakeRepo(), PointPlanner(point), IMAGE, WorkloadKind.AGENTX),
        spec_context(tmp_path),
        options=options,
    )


async def _until(condition: Callable[[], bool], what: str, seconds: float = 30.0) -> None:
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        await asyncio.sleep(0.02)


async def _result_of(toolbox: HostedToolbox, request_id: str) -> TestResult:
    await asyncio.wait_for(toolbox.drain(), 60)
    entry = toolbox._log.get(request_id)
    assert entry is not None and entry.outcome is not None and "result" in entry.outcome, entry
    return TestResult.from_document(entry.outcome["result"])


# -- the inputs ----------------------------------------------------------------------------------------------------


def test_the_inputs_never_print_a_key_and_a_refusal_names_the_field_not_the_value(tmp_path: Path) -> None:
    inputs = _inputs(None, tmp_path)
    assert "m-key" not in repr(inputs) and "d-key" not in repr(inputs)
    for changed, field_name in (({"measurement_key": " "}, "measurement_key"), ({"run_id": ""}, "run_id")):
        with pytest.raises(ValueError, match=f"{field_name} is empty"):
            _inputs(None, tmp_path, **changed)
    with pytest.raises(ValueError, match="dev_key is empty") as refused:
        _inputs(None, tmp_path, dev_key="  ")
    assert "m-key" not in str(refused.value)


@pytest.mark.parametrize(
    ("changed", "message"),
    [
        ({"tp": 0}, "tp must be at least 1"),
        ({"source_sha": "main"}, "source_sha is not a full lowercase commit hash"),
        ({"source_sha": "AB" * 20}, "source_sha is not a full lowercase commit hash"),
        ({"source_sha": "ab" * 19}, "source_sha is not a full lowercase commit hash"),
        ({"grant_limits": {"max_sandboxes": 8, "max_gpus": 8}}, "grant_limits has no gpu_hours, max_runtime_sec"),
        ({"grant_limits": {**_GRANT, "max_gpus": 7.5}}, "max_gpus is not a whole number"),
        ({"grant_limits": {**_GRANT, "max_sandboxes": 0}}, "at least one sandbox"),
    ],
)
def test_inputs_that_cannot_describe_a_session_are_refused(
    tmp_path: Path, changed: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _inputs(None, tmp_path, **changed)


def test_the_source_of_a_sha256_repository_is_a_commit_hash_too(tmp_path: Path) -> None:
    assert _inputs(None, tmp_path, source_sha="cd" * 32).source_sha == "cd" * 32


# -- what is built from them ---------------------------------------------------------------------------------------


async def test_the_toolbox_is_built_from_the_inputs(tmp_path: Path) -> None:
    toolbox = _toolbox(None, tmp_path, tp=2)
    assert isinstance(toolbox, HostedToolbox)
    assert toolbox.trace_root == tmp_path / "session" and toolbox.blobs is not None
    assert isinstance(toolbox.provider, hosted_toolbox._RunProvider)
    assert isinstance(toolbox.provider, E2BSandboxProvider) and toolbox.provider.name == "e2b"
    assert toolbox.backend.pool.template_for(2) == "run-runner-g2"
    assert (await toolbox.measurement_capacity()) == MeasurementCapacity(8, 8)
    assert toolbox._parallel_limit() == 4 and toolbox._queue.policy.global_concurrency == 4, "8 GPUs, TP=2"
    assert toolbox.analysis_tools == (TRACELENS_TOOL,), "kernel_tune and roofline are the local toolbox's"
    assert not toolbox.single_node


def test_the_grants_sandboxes_and_gpus_bound_the_tests_at_once(tmp_path: Path) -> None:
    toolbox = _toolbox(None, tmp_path, grant_limits={**_GRANT, "max_sandboxes": 3})
    assert toolbox._parallel_limit() == 3 and toolbox._queue.policy.global_concurrency == 3
    with pytest.raises(ValueError, match="a grant of 2 GPUs cannot run a test of TP=4"):
        _toolbox(None, tmp_path, tp=4, grant_limits={**_GRANT, "max_gpus": 2}, sandbox_templates={"runner": "r-g4"})


def test_the_rule_pack_digest_is_the_sessions(tmp_path: Path, reading_pack) -> None:
    named = other_pack(tmp_path)
    for pack in (None, named):
        reading_pack(pack)
        toolbox = _toolbox(None, tmp_path)
        assert toolbox.backend._config.rule_pack_digest == rule_pack_identity()["digest"]
    assert toolbox.backend._config.rule_pack_digest == identity_of(named)["digest"]


async def test_without_grant_limits_the_capacity_is_unknown_and_nothing_is_leased(tmp_path: Path) -> None:
    toolbox = _toolbox(None, tmp_path, grant_limits=None)
    assert toolbox.leases is None
    assert (await toolbox.measurement_capacity()) == MeasurementCapacity.unknown()
    with pytest.raises(RequestRejected) as refused:
        await toolbox.lease_branch_sandbox("b1", 1)
    assert refused.value.code == "unsupported_kind"
    await toolbox.release_branch_sandbox("a lease that never was")
    assert toolbox._parallel_limit() is None and toolbox._queue.policy.global_concurrency == 8, "the pool's default"


@pytest.mark.parametrize("changed", [{"dev_key": None}, {"grant_limits": None}])
async def test_leases_need_both_the_dev_key_and_the_grant(tmp_path: Path, changed: dict[str, Any]) -> None:
    toolbox = _toolbox(None, tmp_path, **changed)
    assert toolbox.leases is None
    with pytest.raises(RequestRejected) as refused:
        await toolbox.lease_branch_sandbox("b1", 1)
    assert refused.value.code == "unsupported_kind"


@pytest.mark.parametrize(
    ("templates", "tp", "message"),
    [
        ({"dev": _DEVS[0]}, 1, r"no runner template has the session's 1 cards; templates serve \[\]"),
        ({"runner": list(_RUNNERS), "dev": _DEVS[0]}, 2, r"no dev template has the session's 2 cards"),
        ({"runner": "runner-without-cards", "dev": _DEVS[0]}, 1, "does not name its cards"),
        ({"runner": ["a-g1", "b-g1"], "dev": _DEVS[0]}, 1, "both have 1 cards"),
        ({"runner": 7, "dev": _DEVS[0]}, 1, "templates must be names"),
    ],
)
def test_templates_that_cannot_serve_the_session_are_refused_before_anything_starts(
    tmp_path: Path, templates: dict[str, object], tp: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _toolbox(None, tmp_path, sandbox_templates=templates, tp=tp)
    assert not (tmp_path / "session" / "hosted" / "specs").exists(), "the backend was not started"


def test_the_runner_templates_of_a_role_may_be_a_name_a_list_or_a_mapping(tmp_path: Path) -> None:
    for runner in ("r-g1", ["r-g1", "r-g2"], {"1": "r-g1", "2": "r-g2"}):
        inputs = _inputs(None, tmp_path, sandbox_templates={"runner": runner, "dev": "d-g1"})
        assert hosted_toolbox._runner_template(inputs) == "r-g1"


# -- through the stand-in ------------------------------------------------------------------------------------------


async def test_a_test_is_measured_in_a_sandbox_of_the_service_and_its_result_verified(
    service: Service, tmp_path: Path
) -> None:
    toolbox = _toolbox(service, tmp_path)
    try:
        await toolbox.start()
        request_id = await toolbox.submit_test(req())
        result = await _result_of(toolbox, request_id)
        record = toolbox.backend.journal.latest(request_id)
        assert result.validity is Validity.SUCCEEDED and result.failure_class is None
        assert record is not None and record.extra["verified"] is True and record.state != STATE_SENT
        assert result.manifest_sha256 == record.extra["manifest_sha256"]
        (measured,) = result.rounds
        assert measured.phase == "measure" and measured.metrics["output_throughput"] == 5000.0
        assert result.sandbox is not None and result.sandbox.backend == "hosted"
        assert result.sandbox.node_name == _NODE and result.sandbox.sandbox_id == record.sandbox_id
        (made,) = [c for c in service.creates if c.role == "measurement"]
        assert made.template == "run-runner-g1" and made.metadata[IDEMPOTENCY_KEY] == f"run-1:{request_id}:1"
        assert service.running("measurement") == [], "the sandbox ended with the test"
    finally:
        await toolbox.close()


async def test_the_runner_sandbox_is_given_the_source_it_installs(service: Service, tmp_path: Path) -> None:
    toolbox = _toolbox(service, tmp_path, planned=_HANG)
    try:
        await toolbox.start()
        await toolbox.submit_test(req())
        await _until(lambda: service.running("measurement", "run-1") != [], "the runner sandbox")
        (sandbox_id,) = service.running("measurement", "run-1")
        sandbox = service.observer().connect(sandbox_id, role="measurement")
        seen: list[bytes] = []

        def noted() -> bool:
            try:
                seen.append(sandbox.read_file(_SOURCE_FILE))
            except FileNotFoundError:
                return False
            return True

        await _until(noted, "the start command's note of its environment")
        assert seen[-1].decode().split() == [_SOURCE_REPO, _SOURCE_SHA]
    finally:
        await toolbox.close()


async def test_a_lease_is_a_dev_sandbox_and_its_cards_leave_the_measurement_capacity(
    service: Service, tmp_path: Path
) -> None:
    toolbox = _toolbox(service, tmp_path, tp=2)
    try:
        await toolbox.start()
        lease = await toolbox.lease_branch_sandbox("b1", 2)
        assert lease.template == "run-dev-g2" and lease.sandbox.run("echo hi").stdout == "hi\n"  # type: ignore[attr-defined]
        assert (await toolbox.measurement_capacity()) == MeasurementCapacity(8, 8, leased_cards=2)
        assert lease.remaining_cards == 6 and toolbox._parallel_limit() == 3
        (created,) = [c for c in service.creates if c.role == "dev"]
        assert created.metadata[PURPOSE_KEY] == LEASE_PURPOSE and created.metadata[BRANCH_ID_KEY] == "b1"
        assert created.metadata[RUN_ID_KEY] == "run-1" and not created.options.envs, "a dev sandbox is given no source"
        await toolbox.release_branch_sandbox(lease.lease_id)
        assert (await toolbox.measurement_capacity()) == MeasurementCapacity(8, 8) and toolbox._parallel_limit() == 4
        assert service.running("dev") == []
    finally:
        await toolbox.close()


async def test_a_release_starts_the_test_the_cards_let_run_without_waiting_for_another_test_to_end(
    service: Service, tmp_path: Path
) -> None:
    grant = {**_GRANT, "max_gpus": 2}
    toolbox = _toolbox(service, tmp_path, grant_limits=grant, planned={"fake_delay": 6})
    try:
        await toolbox.start()
        lease = await toolbox.lease_branch_sandbox("b1", 1)
        for _ in range(2):
            await toolbox.submit_test(req())
        await _until(lambda: len(toolbox._tasks) == 1, "the one test the other card runs")
        await asyncio.sleep(0.5)
        assert len(toolbox._tasks) == 1 and service.running("measurement") != []
        await toolbox.release_branch_sandbox(lease.lease_id)
        await _until(lambda: len(toolbox._tasks) == 2, "the second test to start on the card given back", seconds=3.0)
        await asyncio.wait_for(toolbox.drain(), 60)
    finally:
        await toolbox.close()


async def test_close_ends_this_runs_sandboxes_and_no_other_runs(service: Service, tmp_path: Path) -> None:
    other = service.observer()
    elsewhere = {
        "dev": other.create(
            "run-dev-g1", role="dev", timeout_sec=600, metadata={PURPOSE_KEY: LEASE_PURPOSE, RUN_ID_KEY: "run-2"}
        ),
        "measurement": other.create(
            "run-runner-g1", role="measurement", timeout_sec=600, metadata={RUN_ID_KEY: "run-2"}
        ),
    }
    toolbox = _toolbox(service, tmp_path, planned=_HANG)
    try:
        await toolbox.start()
        await toolbox.lease_branch_sandbox("b1", 1)
        await toolbox.submit_test(req())
        await _until(lambda: service.running("measurement", "run-1") != [], "the runner sandbox")
        assert len(service.running("dev")) == 2 and len(service.running("measurement")) == 2
    finally:
        await toolbox.close()
    for role, sandbox in elsewhere.items():
        assert service.running(role) == [sandbox.sandbox_id], f"only run-2's {role} sandbox is left"  # type: ignore[arg-type]


async def test_a_result_of_a_runner_that_read_another_rule_pack_is_refused(
    service: Service,
    tmp_path: Path,
    reading_pack,
) -> None:
    other = other_pack(tmp_path)
    reading_pack(other)
    toolbox = _toolbox(service, tmp_path)
    try:
        await toolbox.start()
        request_id = await toolbox.submit_test(req())
        result = await _result_of(toolbox, request_id)
        assert result.failure_reason and "(rule_pack_mismatch)" in result.failure_reason, result.failure_reason
        assert identity_of(other)["digest"] in result.failure_reason and result.manifest_sha256 is None
    finally:
        await toolbox.close()
