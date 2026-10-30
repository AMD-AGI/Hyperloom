# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the sandbox contract suite runs against: one harness per implementation.

Every test of the suite runs once per implementation, chosen by the ``harness`` fixture's parameter
(``contract/conftest.py``):

- ``local``: :class:`~hyperloom.sandbox.local.LocalSandboxProvider`. It implements the Claw sandbox service's own policy
  (sealed paths, signing identity, platform metadata, avoid-nodes), so it runs every test.
- ``e2b``: :class:`~hyperloom.sandbox.e2b.E2BSandboxProvider`. By default it talks to the E2B stand-in
  (:mod:`hyperloom.sandbox.tests.e2b_standin`: real envd processes behind a simulated control plane), which cannot
  honour that policy, so the tests marked ``claw_policy`` are skipped there. When ``HYPERLOOM_E2B_CONTRACT_API_URL`` is
  set the same suite runs against the real Claw sandbox service instead, ``claw_policy`` tests included; the service's
  data-plane router is ``HYPERLOOM_E2B_CONTRACT_SANDBOX_URL``, the keys are
  ``HYPERLOOM_SANDBOX_KEY_{MEASUREMENT,SUBRUN,DEV}``, and ``HYPERLOOM_E2B_CONTRACT_TEMPLATES`` names its templates as
  ``sealed,dev,subrun`` (the sealed one must run a runner that copies ``/in/spec.json`` to ``/out/status.json``). The
  keys must belong to an otherwise idle application run: a test lists everything its key can see.

Another implementation joins by adding a builder here and a parameter to the fixture.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from hyperloom.measure.runner.bootstrap import runner_start_command
from hyperloom.measure.runner.tests.bootstrap_rig import Rig
from hyperloom.sandbox.local import LocalSandboxProvider, LocalTemplate
from hyperloom.sandbox.socket import CreateOptions, Role, Sandbox, SandboxError, SandboxInfo, SandboxProvider
from hyperloom.sandbox.tests.e2b_standin import (
    ENVD_ENV,
    StandIn,
    StandInTemplate,
    interpreter_binds,
    unavailable_reason,
    without_network,
)

log = logging.getLogger(__name__)

#: The fake runner of the local and the Claw service legs: it waits for /in/spec.json and answers by copying it to
#: /out/status.json. The stand-in's leg starts the runner's start-up script, whose rig runner does the same.
FAKE_RUNNER = (
    "import os,time,shutil\n"
    "root=os.environ['HYPERLOOM_SANDBOX_ROOT']\n"
    "src=root+'/in/spec.json'\n"
    "while not os.path.exists(src): time.sleep(0.02)\n"
    "time.sleep(0.05)\n"
    "shutil.copy(src, root+'/out/status.json')\n"
    "time.sleep(600)\n"
)

SERVICE_URL_ENV = "HYPERLOOM_E2B_CONTRACT_API_URL"
_SERVICE_SANDBOX_URL_ENV = "HYPERLOOM_E2B_CONTRACT_SANDBOX_URL"
_SERVICE_TEMPLATES_ENV = "HYPERLOOM_E2B_CONTRACT_TEMPLATES"
# A real service starts a sandbox on a scheduled pod, which takes minutes where the stand-in takes a second.
_SERVICE_WAIT_ENV = "HYPERLOOM_E2B_CONTRACT_WAIT_SEC"
_SERVICE_WAIT_SEC = 600.0


@dataclass(frozen=True)
class Harness:
    """What the contract suite needs from an implementation.

    Attributes:
        provider: The provider under test.
        sealed_template: A measurement-role sealed template that signs, and
            whose runner copies ``/in/spec.json`` to ``/out/status.json``.
        dev_template: An unsealed template for the ``dev`` role.
        subrun_template: An unsealed template for the ``subrun`` role.
        claw_policy: Whether the implementation enforces the Claw sandbox service's own policy; the tests marked
            ``claw_policy`` need it.
    """

    provider: SandboxProvider
    sealed_template: str
    dev_template: str
    subrun_template: str
    claw_policy: bool = True


class _KillsWhatItMade:
    """A provider that remembers the sandboxes it created, so a run against a shared service ends exactly those."""

    def __init__(self, provider: SandboxProvider, min_wait_sec: float = 0.0) -> None:
        self._provider = provider
        self._min_wait_sec = min_wait_sec
        self.name = provider.name
        self._made: list[Sandbox] = []

    def create(
        self,
        template: str,
        *,
        role: Role,
        timeout_sec: int,
        metadata: Mapping[str, str],
        options: CreateOptions = CreateOptions(),
    ) -> Sandbox:
        if options.wait_sec < self._min_wait_sec:
            options = replace(options, wait_sec=self._min_wait_sec)
        sandbox = self._provider.create(
            template, role=role, timeout_sec=timeout_sec, metadata=metadata, options=options
        )
        self._made.append(sandbox)
        return sandbox

    def connect(self, sandbox_id: str, *, role: Role) -> Sandbox:
        return self._provider.connect(sandbox_id, role=role)

    def find(self, metadata: Mapping[str, str], *, role: Role) -> list[SandboxInfo]:
        return self._provider.find(metadata, role=role)

    def kill_all(self) -> None:
        for sandbox in self._made:
            try:
                sandbox.kill()
            except SandboxError:
                log.warning("the contract suite could not end sandbox %s", sandbox.sandbox_id, exc_info=True)


__all__ = ["FAKE_RUNNER", "SERVICE_URL_ENV", "Harness", "claw_service_harness", "local_harness", "standin_harness"]


@contextmanager
def local_harness() -> Iterator[Harness]:
    templates = [
        LocalTemplate(
            "runner", "measurement", command=(sys.executable, "-c", FAKE_RUNNER), gpus=1, sealed=True, signing=True
        ),
        LocalTemplate("dev", "dev"),
        LocalTemplate("sub", "subrun"),
    ]
    with LocalSandboxProvider(templates, gpus=[0, 1, 2, 3], hostname="node-a") as provider:
        yield Harness(provider, "runner", "dev", "sub")


@contextmanager
def standin_harness(workdir: Path, rig: Rig) -> Iterator[Harness]:
    """The stand-in's leg: its sealed template starts the runner's own start-up script on ``rig``.

    The sandbox has no network, as a measurement sandbox has none, and sees the rig's runner cache read-only. The script
    builds a venv, installs the rig's hash lock from the cache and starts the rig's runner, which answers a sheet like
    :data:`FAKE_RUNNER` does, so the suite's sealed tests also exercise the script inside a sandbox.
    """
    from hyperloom.sandbox.e2b import E2BSandboxProvider

    reason = unavailable_reason(os.environ.get(ENVD_ENV))
    if reason is not None:
        pytest.skip(reason)
    templates = [
        StandInTemplate(
            "runner",
            "measurement",
            sealed=True,
            start=without_network(runner_start_command()),
            binds=(*interpreter_binds(), str(rig.root)),
            envs=rig.sandbox_env(),
        ),
        StandInTemplate("dev", "dev"),
        StandInTemplate("sub", "subrun"),
    ]
    with StandIn(Path(os.environ[ENVD_ENV]), workdir, templates) as standin:
        provider = E2BSandboxProvider(standin.keys, api_url=standin.api_url, sandbox_url=standin.sandbox_url)
        yield Harness(provider, "runner", "dev", "sub", claw_policy=False)


@contextmanager
def claw_service_harness() -> Iterator[Harness]:
    from hyperloom.sandbox.e2b import E2BSandboxProvider

    environ = {
        **os.environ,
        "E2B_API_URL": os.environ[SERVICE_URL_ENV],
        "E2B_SANDBOX_URL": os.environ.get(_SERVICE_SANDBOX_URL_ENV, os.environ[SERVICE_URL_ENV]),
    }
    sealed, dev, subrun = os.environ[_SERVICE_TEMPLATES_ENV].split(",")
    wait_sec = float(os.environ.get(_SERVICE_WAIT_ENV, _SERVICE_WAIT_SEC))
    provider = _KillsWhatItMade(E2BSandboxProvider.from_environ(environ), min_wait_sec=wait_sec)
    try:
        yield Harness(provider, sealed, dev, subrun)
    finally:
        provider.kill_all()
