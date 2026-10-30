# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``build_hosted_toolbox`` from plain values, over the E2B stand-in and the fake runner.

The stand-in runs real envd processes behind a simulated control plane. Its runner templates are sealed and list the
commit and the runner cache manifest's digest as the environment a create may pass, as the service's do. It writes no platform metadata, so the
tests put a platform's part in front of the E2B provider: the signing key the service reports for a runner sandbox (the
fake runner signs with it) and its node. The tests that need the stand-in skip where it cannot run (root, ``unshare``, an envd binary
named by ``HYPERLOOM_E2B_ENVD``); the ones that only build the toolbox always run.
"""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import json
import os
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

import hyperloom
from hyperloom.common.outside_facts.rule_pack import rule_pack_identity
from hyperloom.contracts.measurement import TestResult, Validity
from hyperloom.hosted.host_info import read_host_info
from hyperloom.hosted.toolbox_inputs import hosted_toolbox_inputs
from hyperloom.measure import signing
from hyperloom.measure.framework.base import SESSION_SHA_ENV, SNAPSHOT_ENV
from hyperloom.measure.hosted_log import STATE_SENT
from hyperloom.measure.runner.cache_layout import MANIFEST_PIN_ENV
from hyperloom.measure.sandboxes import BRANCH_ID_KEY, DEFAULT_MAX_PARALLEL_TESTS, PURPOSE_KEY
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
from hyperloom_multibranch.contracts.blackboard import Role as BlackboardRole, Writer
from hyperloom_multibranch.contracts.blackboard.vocab import LeaseEndReason, LeaseKind
from hyperloom_multibranch.blackboard import LocalBlackboard
from hyperloom_multibranch.contracts.toolbox import MeasurementCapacity, RequestRejected, TestProgress, WorkloadKind
from hyperloom_multibranch.toolbox import hosted_toolbox
from hyperloom_multibranch.toolbox.hosted_leases import (
    LEASE_PURPOSE,
    LIFETIME_MARGIN_SEC,
    RUN_ID_KEY,
    GrantLimits,
    LeaseTuning,
)
from hyperloom_multibranch.toolbox.hosted_toolbox import (
    SOURCE_SHA_ENV,
    HostedOptions,
    HostedToolbox,
    HostedToolboxInputs,
    build_hosted_toolbox,
    parallel_tests_for,
)
from hyperloom_multibranch.toolbox.queue import QueuePolicy
from hyperloom_multibranch.toolbox.tests.helpers import IMAGE, FakeRepo, req, lineage_blackboard
from hyperloom_multibranch.toolbox.tests.test_hosted_executor import PointPlanner
from hyperloom_multibranch.toolbox.toolbox import AcceptanceSetup
from hyperloom_multibranch.toolbox.tools import TRACELENS_TOOL

#: Template names are opaque (no card count in them): a template's cards are its key in the template list.
_RUNNERS = ("ar-01ja-measurement-7f3c", "ar-01ja-measurement-b20e")
_NARROW = "ar-01ja-narrow-55aa"  # a 1-card runner template that accepts the commit but not the manifest pin
_DEVS = ("ar-01ja-dev-c41d", "ar-01ja-dev-9e02")
_RUNNER_MAP = {"1": _RUNNERS[0], "2": _RUNNERS[1]}
_DEV_MAP = {"1": _DEVS[0], "2": _DEVS[1]}
_SOURCE_REPO, _SOURCE_SHA = "https://source.invalid/hyperloom.git", "ab" * 20
_MANIFEST = "cd" * 32  # the SHA-256 of the manifest file of a runner cache entry
#: What the runner no longer reads: the repository its start-up script once fetched from. The host may still send it.
_RETIRED_REPO_ENV = "HYPERLOOM_SOURCE_REPO"
_GRANT = {"max_sandboxes": 8, "max_gpus": 8, "gpu_hours": 400.0, "max_runtime_sec": 7200}
_SOURCE_FILE = "/out/source-envs.txt"
#: The runner template's start command: note the source variables the sandbox was given (whatever their names are
#: among the ones the runner's bootstrap reads, or used to read), then start the fake runner.
_START = (
    "/bin/sh",
    "-c",
    f'env | grep -E "^({_RETIRED_REPO_ENV}|{SOURCE_SHA_ENV}|{MANIFEST_PIN_ENV})=" | sort > {_SOURCE_FILE}; exec "$0" "$@"',
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


class _Platform(E2BSandboxProvider):
    """The toolbox's provider plus the platform's metadata on ``measurement`` sandboxes, remembering every create it is
    asked for."""

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


def _runner_template(
    name: str, key_file: Path, accepted_envs: tuple[str, ...] = (SOURCE_SHA_ENV, MANIFEST_PIN_ENV)
) -> StandInTemplate:
    source = Path(hyperloom.__file__).resolve().parents[1]
    return StandInTemplate(
        name,
        "measurement",
        sealed=True,
        accepted_envs=accepted_envs,
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
    templates = [
        *(_runner_template(name, key_file) for name in _RUNNERS),
        _runner_template(_NARROW, key_file, (SOURCE_SHA_ENV,)),
        *(StandInTemplate(n, "dev") for n in _DEVS),
    ]
    workdir = tmp_path / "standin"
    workdir.mkdir()
    with ExitStack() as stack:
        standin = stack.enter_context(StandIn(Path(os.environ[ENVD_ENV]), workdir, templates))
        made = Service(standin)
        provider = functools.partial(_Platform, public_key=signing.public_key_pem(key), creates=made.creates)
        monkeypatch.setattr(hosted_toolbox, "E2BSandboxProvider", provider)
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
        "sandbox_templates": {"measurement": _RUNNER_MAP, "dev": _DEV_MAP, "subrun": {"1": "x-sub"}},
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
        # No tables: on a host whose image holds aiter's, the default would name a snapshot these tests do not expect.
        framework_configs="",
    )
    return build_hosted_toolbox(
        _inputs(service, tmp_path, **changed),
        lineage_blackboard(tmp_path / "blackboard"),
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


_NOT_A_DIGEST = "runner_cache_manifest_sha256 is not a 64-character lowercase hex SHA-256"


@pytest.mark.parametrize(
    ("changed", "message"),
    [
        ({"tp": 0}, "tp must be at least 1"),
        ({"source_sha": "main"}, "source_sha is not a full lowercase commit hash"),
        ({"source_sha": "AB" * 20}, "source_sha is not a full lowercase commit hash"),
        ({"source_sha": "ab" * 19}, "source_sha is not a full lowercase commit hash"),
        ({"runner_cache_manifest_sha256": ""}, _NOT_A_DIGEST),
        ({"runner_cache_manifest_sha256": "CD" * 32}, _NOT_A_DIGEST),
        ({"runner_cache_manifest_sha256": "cd" * 31}, _NOT_A_DIGEST),
        ({"runner_cache_manifest_sha256": "cd" * 33}, _NOT_A_DIGEST),
        ({"runner_cache_manifest_sha256": "cg" * 32}, _NOT_A_DIGEST),
        ({"runner_cache_manifest_sha256": _MANIFEST + "\n"}, _NOT_A_DIGEST),
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


#: What the host entry builds the inputs from: a plain dict whose keys are these names (``HostedToolboxInputs(**d)``).
#: Fields are only ever added, and only with a default, so the required names are fixed. ``source_repo`` was required
#: until the runner stopped fetching from a repository: a host may still send it, and one may leave it out.
_REQUIRED_FIELDS = (
    "run_id",
    "session_dir",
    "api_url",
    "sandbox_url",
    "measurement_key",
    "sandbox_templates",
    "source_sha",
)
_OPTIONAL_FIELDS = ("source_repo", "tp", "dev_key", "grant_limits", "runner_cache_manifest_sha256")


def test_a_plain_dict_with_exactly_the_field_names_builds_the_inputs(tmp_path: Path) -> None:
    plain: dict[str, Any] = {
        "run_id": "run-1",
        "session_dir": tmp_path,
        "api_url": "http://service.invalid",
        "sandbox_url": "http://sandboxes.invalid",
        "measurement_key": "m-key",
        "sandbox_templates": {"measurement": {"1": _RUNNERS[0]}, "dev": {"1": _DEVS[0]}},
        "source_repo": _SOURCE_REPO,
        "source_sha": _SOURCE_SHA,
        "tp": 1,
        "dev_key": "d-key",
        "grant_limits": dict(_GRANT),
        "runner_cache_manifest_sha256": _MANIFEST,
    }
    assert set(plain) == {*_REQUIRED_FIELDS, *_OPTIONAL_FIELDS}
    inputs = HostedToolboxInputs(**plain)
    assert {name: getattr(inputs, name) for name in plain} == plain
    required = {f.name for f in dataclasses.fields(inputs) if f.default is f.default_factory is dataclasses.MISSING}
    assert required == set(_REQUIRED_FIELDS) and {f.name for f in dataclasses.fields(inputs)} == set(plain)
    only_required = {name: plain[name] for name in _REQUIRED_FIELDS}
    assert HostedToolboxInputs(**only_required).dev_key is None


@pytest.mark.parametrize(
    "left_out",
    [
        ("runner_cache_manifest_sha256",),  # a host that predates the manifest pin
        ("source_repo",),  # a host that no longer has a repository to name
        ("source_repo", "runner_cache_manifest_sha256"),
        _OPTIONAL_FIELDS,
    ],
)
def test_a_plain_dict_without_the_newer_or_the_retired_keys_still_builds_the_inputs(
    tmp_path: Path, left_out: tuple[str, ...]
) -> None:
    plain = {**vars(_inputs(None, tmp_path)), "runner_cache_manifest_sha256": _MANIFEST}
    fewer = {name: value for name, value in plain.items() if name not in left_out}
    assert set(left_out) < set(plain) and set(fewer) >= set(_REQUIRED_FIELDS)
    inputs = HostedToolboxInputs(**fewer)
    for name in left_out:
        assert getattr(inputs, name) == {"tp": 1}.get(name), f"{name} falls back to its default"
    pinned = {} if "runner_cache_manifest_sha256" in left_out else {MANIFEST_PIN_ENV: _MANIFEST}
    assert inputs.create_envs() == {SOURCE_SHA_ENV: _SOURCE_SHA, **pinned}


@pytest.mark.parametrize("source_repo", [None, "", "  ", _SOURCE_REPO])
def test_the_source_repository_is_accepted_whatever_it_is_and_never_reaches_the_runner(
    tmp_path: Path, source_repo: str | None
) -> None:
    inputs = _inputs(None, tmp_path, source_repo=source_repo)
    assert inputs.source_repo == source_repo, "the field is kept as given"
    assert inputs.create_envs() == {SOURCE_SHA_ENV: _SOURCE_SHA}, "and the runner is not told of it"
    assert _toolbox(None, tmp_path, source_repo=source_repo).backend._config.create_envs == inputs.create_envs()


def test_a_refused_manifest_digest_is_never_in_the_message(tmp_path: Path) -> None:
    secret_looking = "XY" * 32
    with pytest.raises(ValueError, match=_NOT_A_DIGEST) as refused:
        _inputs(None, tmp_path, runner_cache_manifest_sha256=secret_looking)
    assert secret_looking not in str(refused.value)


def test_a_dict_with_one_key_more_or_one_required_key_less_is_refused(tmp_path: Path) -> None:
    plain = vars(_inputs(None, tmp_path)).copy()
    with pytest.raises(TypeError, match="unexpected keyword argument 'extra_field'"):
        HostedToolboxInputs(**plain, extra_field=1)
    for name in _REQUIRED_FIELDS:
        fewer = {key: value for key, value in plain.items() if key != name}
        with pytest.raises(TypeError, match=rf"missing 1 required positional argument: '{name}'"):
            HostedToolboxInputs(**fewer)


def test_the_source_of_a_sha256_repository_is_a_commit_hash_too(tmp_path: Path) -> None:
    assert _inputs(None, tmp_path, source_sha="cd" * 32).source_sha == "cd" * 32


# -- what is built from them ---------------------------------------------------------------------------------------


async def test_the_toolbox_is_built_from_the_inputs(tmp_path: Path) -> None:
    toolbox = _toolbox(None, tmp_path, tp=2)
    assert isinstance(toolbox, HostedToolbox)
    assert toolbox.trace_root == tmp_path / "session" and toolbox.blobs is not None
    assert type(toolbox.provider) is E2BSandboxProvider and toolbox.provider.name == "e2b"
    assert toolbox.backend._config.create_envs == {SOURCE_SHA_ENV: _SOURCE_SHA}, "no repository, no manifest pin"
    assert toolbox.backend.pool.template_for(2) == _RUNNERS[1]
    assert (await toolbox.measurement_capacity()) == MeasurementCapacity(8, 8)
    assert toolbox._parallel_limit() == 4 and toolbox._queue.policy.global_concurrency == 4, "8 GPUs, TP=2"
    assert toolbox.analysis_tools == (TRACELENS_TOOL,), "kernel_tune and roofline are the local toolbox's"
    assert not toolbox.single_node


def test_every_runner_sandbox_is_named_the_sessions_snapshot_of_the_framework_tables(tmp_path: Path) -> None:
    """A test that adds tuned rows to a default table needs the snapshot, or the runner refuses it (#156 c4084)."""
    configs = tmp_path / "aiter-configs"
    configs.mkdir()
    (configs / "a8w8_blockscale_tuned_gemm.csv").write_text("M,N,K,us\n1,2,3,4\n")
    options = HostedOptions(retry=_RETRY, lease_tuning=_FAST, framework_configs=str(configs))

    envs = _toolbox(None, tmp_path, options=options).backend._config.create_envs

    snapshot = tmp_path / "session" / "framework"
    assert envs[SNAPSHOT_ENV] == str(snapshot) and any(snapshot.rglob("a8w8_blockscale_tuned_gemm.csv"))
    assert len(envs[SESSION_SHA_ENV]) == 64
    assert envs[SOURCE_SHA_ENV] == _SOURCE_SHA


def test_no_framework_tables_names_no_snapshot(tmp_path: Path) -> None:
    options = HostedOptions(retry=_RETRY, lease_tuning=_FAST, framework_configs="")
    envs = _toolbox(None, tmp_path, options=options).backend._config.create_envs
    assert SNAPSHOT_ENV not in envs and SESSION_SHA_ENV not in envs


def test_the_runner_cache_manifest_digest_is_a_create_environment_of_every_runner_sandbox(tmp_path: Path) -> None:
    toolbox = _toolbox(None, tmp_path, runner_cache_manifest_sha256=_MANIFEST)
    assert toolbox.backend._config.create_envs == {SOURCE_SHA_ENV: _SOURCE_SHA, MANIFEST_PIN_ENV: _MANIFEST}
    assert MANIFEST_PIN_ENV == "HYPERLOOM_RUNNER_CACHE_MANIFEST_SHA256", "the name the runner's bootstrap reads"
    assert toolbox.backend.pool._create_envs == toolbox.backend._config.create_envs, "the pool creates with them"


def test_the_grants_sandboxes_and_gpus_bound_the_tests_at_once(tmp_path: Path) -> None:
    toolbox = _toolbox(None, tmp_path, grant_limits={**_GRANT, "max_sandboxes": 3})
    assert toolbox._parallel_limit() == 3 and toolbox._queue.policy.global_concurrency == 3
    with pytest.raises(ValueError, match="a grant of 2 GPUs cannot run a test of TP=4"):
        _toolbox(
            None, tmp_path, tp=4, grant_limits={**_GRANT, "max_gpus": 2}, sandbox_templates={"measurement": {"4": "r"}}
        )


@pytest.mark.parametrize(
    ("sandboxes", "gpus", "tp", "expected"),
    [
        (8, 8, 2, 4),  # the GPUs bound it: 8 // 2
        (3, 8, 2, 3),  # the sandboxes bound it
        (8, 7, 2, 3),  # a GPU short of a whole test does not count: 7 // 2
        (1, 16, 1, 1),
        (8, 4, 4, 1),
    ],
)
def test_the_parallel_tests_rule_is_the_lesser_of_whole_tests_and_sandboxes(
    sandboxes: int, gpus: int, tp: int, expected: int
) -> None:
    grant = GrantLimits(max_sandboxes=sandboxes, max_gpus=gpus, gpu_hours=1.0, max_runtime_sec=60)
    assert parallel_tests_for(grant, tp) == expected
    assert "parallel_tests_for" in hosted_toolbox.__all__


def test_the_parallel_tests_rule_without_a_grant_is_the_pools_default_and_refuses_a_grant_too_small() -> None:
    assert parallel_tests_for(None, 8) == DEFAULT_MAX_PARALLEL_TESTS
    small = GrantLimits(max_sandboxes=8, max_gpus=2, gpu_hours=1.0, max_runtime_sec=60)
    with pytest.raises(ValueError, match="a grant of 2 GPUs cannot run a test of TP=4"):
        parallel_tests_for(small, 4)


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
        ({"dev": _DEV_MAP}, 1, r"the template list has no 'measurement' role"),
        (
            {"measurement": _RUNNER_MAP, "dev": _DEV_MAP},
            4,
            r"no 'measurement' template has the session's 4 cards; "
            r"templates serve \[1, 2\]",
        ),
        (
            {"runner": {"8": "r8"}, "dev": _DEV_MAP},
            1,
            r"no 'measurement' template has the session's 1 cards; "
            r"templates serve \[8\]",
        ),
        (
            {"measurement": _RUNNER_MAP, "dev": {"1": _DEVS[0]}},
            2,
            r"no dev template has the session's 2 cards; "
            r"templates serve \[1\]",
        ),
        ({"measurement": _RUNNER_MAP, "runner": _RUNNER_MAP, "dev": _DEV_MAP}, 1, "both 'measurement' and its older"),
        ({"measurement": "opaque-name", "dev": _DEV_MAP}, 1, "'measurement' templates must map card counts to names"),
        ({"measurement": list(_RUNNERS), "dev": _DEV_MAP}, 1, "'measurement' templates must map card counts"),
        ({"measurement": _RUNNER_MAP, "dev": _DEVS[0]}, 1, "'dev' templates must map card counts to names"),
        ({"measurement": {"one": "r"}, "dev": _DEV_MAP}, 1, "got key 'one'"),
        ({"measurement": {"0": "r"}, "dev": _DEV_MAP}, 1, "got key '0'"),
        ({"measurement": {"1": ""}, "dev": _DEV_MAP}, 1, "got key '1'"),
        ({"measurement": {"1": "a", "01": "b"}, "dev": _DEV_MAP}, 1, "name 1 cards twice"),
    ],
)
def test_templates_that_cannot_serve_the_session_are_refused_before_anything_starts(
    tmp_path: Path, templates: dict[str, object], tp: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _toolbox(None, tmp_path, sandbox_templates=templates, tp=tp)
    assert not (tmp_path / "session" / "hosted" / "specs").exists(), "the backend was not started"


def test_the_grants_template_map_gives_each_role_its_cards_by_key_and_never_reads_the_name(tmp_path: Path) -> None:
    """The map Claw builds from the grant's names: role -> {cards: opaque name}; a name that looks like another card
    count (``-g8``) is still the template of the key it is filed under."""
    templates = {
        "measurement": {"1": "tpl-a", "8": "tpl-b-g1"},
        "dev": {"1": "tpl-c", "8": "tpl-d"},
        "subrun": {"1": "tpl-e", "8": "tpl-f"},
    }
    for tp, runner, dev in ((1, "tpl-a", "tpl-c"), (8, "tpl-b-g1", "tpl-d")):
        (tmp_path / str(tp)).mkdir()
        toolbox = _toolbox(None, tmp_path / str(tp), sandbox_templates=templates, tp=tp, grant_limits=_GRANT)
        assert toolbox.backend.pool.template_for(tp) == runner
        assert toolbox.leases is not None and toolbox.leases._session.template == dev


def test_the_hosts_template_variable_reaches_the_toolbox_unchanged(tmp_path: Path) -> None:
    """``HYPERLOOM_SANDBOX_TEMPLATES`` as Claw sets it, read by the host entry and handed on as the toolbox's inputs."""
    templates = {"measurement": {"1": "tpl-a", "8": "tpl-b"}, "dev": {"1": "tpl-c", "8": "tpl-d"}}
    spec = tmp_path / "spec.json"
    spec.write_text("{}", encoding="utf-8")
    env = {
        "HYPERLOOM_INPUT_FILE": str(spec),
        "HYPERLOOM_RUN_ID": "run-1",
        "HYPERLOOM_RUN_CREATED_AT": "2026-10-28T12:00:00Z",
        "HYPERLOOM_STATE_DIR": str(tmp_path),
        "E2B_API_URL": "http://service.invalid",
        "E2B_SANDBOX_URL": "http://sandboxes.invalid",
        "HYPERLOOM_SANDBOX_KEY_MEASUREMENT": "m-key-000000000000",
        "HYPERLOOM_SANDBOX_TEMPLATES": json.dumps(templates),
        "HYPERLOOM_SOURCE_REPO": _SOURCE_REPO,
        "HYPERLOOM_SOURCE_SHA": _SOURCE_SHA,
    }
    given = hosted_toolbox_inputs(read_host_info(env, require=()), session_dir=tmp_path / "session", tp=8)
    assert given["sandbox_templates"] == templates
    assert hosted_toolbox._runner_template(HostedToolboxInputs(**given)) == "tpl-b"  # type: ignore[arg-type]


def test_runner_is_the_older_name_of_the_measurement_role(tmp_path: Path) -> None:
    inputs = _inputs(None, tmp_path, sandbox_templates={"runner": {"1": "r1", "8": "r8"}, "dev": _DEV_MAP})
    assert hosted_toolbox._runner_template(inputs) == "r1"
    assert hosted_toolbox._runner_template(dataclasses.replace(inputs, tp=8)) == "r8"


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
        assert made.template == _RUNNERS[0] and made.metadata[IDEMPOTENCY_KEY] == f"run-1:{request_id}:1"
        assert made.metadata[RUN_ID_KEY] == "run-1"
        assert dict(made.options.envs or {}) == {SOURCE_SHA_ENV: _SOURCE_SHA}, "the commit alone: no repo, no pin"
        assert service.running("measurement") == [], "the sandbox ended with the test"
    finally:
        await toolbox.close()


@pytest.mark.parametrize("manifest", [None, _MANIFEST])
async def test_the_runner_sandbox_is_given_the_commit_it_installs_and_the_manifest_it_must_be(
    service: Service, tmp_path: Path, manifest: str | None
) -> None:
    """The host also sends a repository (``_inputs`` does): the sandbox is not given it, the bootstrap does not read it."""
    toolbox = _toolbox(service, tmp_path, planned=_HANG, runner_cache_manifest_sha256=manifest)
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
        given = dict(line.split("=", 1) for line in seen[-1].decode().splitlines())
        pinned = {} if manifest is None else {MANIFEST_PIN_ENV: manifest}
        assert given == {SOURCE_SHA_ENV: _SOURCE_SHA, **pinned}, f"{_RETIRED_REPO_ENV} must not be among them"
    finally:
        await toolbox.close()


async def test_every_runner_sandbox_is_created_with_the_manifest_digest_when_the_host_has_one(
    service: Service, tmp_path: Path
) -> None:
    toolbox = _toolbox(service, tmp_path, runner_cache_manifest_sha256=_MANIFEST)
    try:
        await toolbox.start()
        first, second = await toolbox.submit_test(req()), await toolbox.submit_test(req(server_args="--other"))
        for request_id in (first, second):
            assert (await _result_of(toolbox, request_id)).validity is Validity.SUCCEEDED
        made = [c for c in service.creates if c.role == "measurement"]
        assert len(made) == 2, "one runner sandbox for each test"
        for created in made:
            assert dict(created.options.envs or {}) == {SOURCE_SHA_ENV: _SOURCE_SHA, MANIFEST_PIN_ENV: _MANIFEST}
    finally:
        await toolbox.close()


async def test_a_runner_template_that_does_not_list_a_create_environment_name_refuses_the_test(
    service: Service, tmp_path: Path
) -> None:
    """``_NARROW`` accepts the commit and not the manifest digest, so the host that has one is refused by name."""
    toolbox = _toolbox(
        service,
        tmp_path,
        sandbox_templates={"measurement": {"1": _NARROW}, "dev": _DEV_MAP},
        runner_cache_manifest_sha256=_MANIFEST,
    )
    try:
        await toolbox.start()
        request_id = await toolbox.submit_test(req())
        result = await _result_of(toolbox, request_id)
        assert result.validity is not Validity.SUCCEEDED and result.manifest_sha256 is None
        assert result.failure_reason is not None and "sandbox_denied" in result.failure_reason, result.failure_reason
        assert f"env_not_allowed: template '{_NARROW}' does not accept {MANIFEST_PIN_ENV}" in result.failure_reason
        assert _MANIFEST not in result.failure_reason, "the refusal names the variable, not its value"
        assert service.running("measurement") == [], "a refused create made no sandbox"
    finally:
        await toolbox.close()


async def test_a_runner_template_that_lists_only_the_commit_runs_the_test_of_a_host_with_no_manifest_digest(
    service: Service, tmp_path: Path
) -> None:
    toolbox = _toolbox(service, tmp_path, sandbox_templates={"measurement": {"1": _NARROW}, "dev": _DEV_MAP})
    try:
        await toolbox.start()
        result = await _result_of(toolbox, await toolbox.submit_test(req()))
        assert result.validity is Validity.SUCCEEDED, result.failure_reason
    finally:
        await toolbox.close()


async def _progress_with_a_line(toolbox: HostedToolbox, request_id: str) -> TestProgress:
    deadline = time.monotonic() + 30
    while True:
        progress = await toolbox.progress(request_id)
        if progress is not None and progress.last_line:
            return progress
        assert time.monotonic() < deadline, f"timed out waiting for a live line, last progress {progress}"
        await asyncio.sleep(0.02)


async def test_a_running_tests_progress_is_the_last_line_its_runner_reports_from_the_sandbox(
    service: Service, tmp_path: Path
) -> None:
    toolbox = _toolbox(service, tmp_path, planned={**_HANG, "fake_log_line": "decode batch 8"})
    try:
        await toolbox.start()
        request_id = await toolbox.submit_test(req())
        progress = await _progress_with_a_line(toolbox, request_id)
        assert progress.state == "running" and progress.last_line == "decode batch 8"
        assert progress.last_output_seconds_ago is not None and 0.0 <= progress.last_output_seconds_ago < 30.0
        assert progress.running_seconds is not None and progress.queued_ahead is None
    finally:
        await toolbox.close()


async def test_a_running_test_whose_runner_has_written_no_log_has_no_live_line(
    service: Service, tmp_path: Path
) -> None:
    toolbox = _toolbox(service, tmp_path, planned=_HANG)
    try:
        await toolbox.start()
        request_id = await toolbox.submit_test(req())
        await _until(lambda: service.running("measurement", "run-1") != [], "the runner sandbox")
        (sandbox_id,) = service.running("measurement", "run-1")
        sandbox = service.observer().connect(sandbox_id, role="measurement")

        def running() -> bool:
            try:
                return json.loads(sandbox.read_file("/out/progress.json")).get("phase") == "running"
            except FileNotFoundError:
                return False

        await _until(running, "the runner's progress to say it runs")
        await asyncio.sleep(0.5)  # ten polls of that file
        progress = await toolbox.progress(request_id)
        assert progress is not None and progress.state == "running" and progress.running_seconds is not None
        assert (progress.last_line, progress.last_output_seconds_ago) == (None, None)
    finally:
        await toolbox.close()


async def test_a_lease_is_a_dev_sandbox_and_its_cards_leave_the_measurement_capacity(
    service: Service, tmp_path: Path
) -> None:
    toolbox = _toolbox(service, tmp_path, tp=2)
    try:
        await toolbox.start()
        lease = await toolbox.lease_branch_sandbox("b1", 2, lifetime_sec=600)
        assert lease.template == _DEVS[1] and lease.sandbox.run("echo hi").stdout == "hi\n"  # type: ignore[attr-defined]
        assert lease.lifetime_sec == 600 + LIFETIME_MARGIN_SEC  # type: ignore[attr-defined]
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


async def test_a_lease_is_recorded_as_its_sandbox_from_the_grant_to_one_return_and_the_close_ends_one_left_open(
    service: Service, tmp_path: Path
) -> None:
    toolbox = _toolbox(service, tmp_path, tp=2)
    reader = LocalBlackboard(tmp_path / "blackboard", Writer(BlackboardRole.SCHEDULER, "scheduler"))
    try:
        await toolbox.start()
        lease = await toolbox.lease_branch_sandbox("b1", 2)
        await toolbox.lease_branch_sandbox("b02", 2)
        await toolbox.release_branch_sandbox(lease.lease_id)
        released = time.time()
    finally:
        await toolbox.close()

    released_hold, kept_hold = await reader.card_holds()
    start = released_hold.start
    assert (start.holder, start.branch_id, start.kind, start.cards, start.gpu_ids) == ("b1", "b1", LeaseKind.DEV, 2, ())
    assert start.sandbox_id == lease.sandbox.sandbox_id  # type: ignore[attr-defined]
    (back,) = set(released_hold.returned.values())
    assert sorted(released_hold.returned) == [0, 1] and back <= released, "the sandbox's cards go back at once"
    assert released_hold.end_reason is LeaseEndReason.RELEASED
    assert kept_hold.end_reason is LeaseEndReason.SESSION_CLOSE and min(kept_hold.returned.values()) >= released
    (row,) = [row for row in await reader.progress() if row.branch == "b1"]
    assert row.held_card_seconds == pytest.approx(2 * (back - start.started_at))


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
            _DEVS[0], role="dev", timeout_sec=600, metadata={PURPOSE_KEY: LEASE_PURPOSE, RUN_ID_KEY: "run-2"}
        ),
        "measurement": other.create(_RUNNERS[0], role="measurement", timeout_sec=600, metadata={RUN_ID_KEY: "run-2"}),
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


async def test_start_ends_the_lease_sandboxes_an_earlier_process_of_the_run_left(
    service: Service, tmp_path: Path
) -> None:
    other = service.observer()
    left = other.create(
        _DEVS[0], role="dev", timeout_sec=600, metadata={PURPOSE_KEY: LEASE_PURPOSE, RUN_ID_KEY: "run-1"}
    )
    elsewhere = other.create(
        _DEVS[0], role="dev", timeout_sec=600, metadata={PURPOSE_KEY: LEASE_PURPOSE, RUN_ID_KEY: "run-2"}
    )
    toolbox = _toolbox(service, tmp_path)
    try:
        await toolbox.start()
        assert service.running("dev") == [elsewhere.sandbox_id], "this run's leftover is ended, run-2's is not"
        assert left.sandbox_id not in service.running("dev")
    finally:
        await toolbox.close()


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
