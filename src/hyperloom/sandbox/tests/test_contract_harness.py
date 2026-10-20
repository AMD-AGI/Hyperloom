# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The contract suite's leg against a real Claw sandbox service, driven here against the stand-in.

A real service is only reachable from an integration environment, so this checks the part that can be checked anywhere:
that the environment variables the integration run sets are read as documented, the suite's sandboxes are created with
the right keys and templates, and they are ended afterwards.
"""

from __future__ import annotations

import pytest

from hyperloom.sandbox.e2b import E2BSandboxProvider
from hyperloom.sandbox.socket import SandboxDenied
from hyperloom.sandbox.tests.contract_harness import SERVICE_URL_ENV, claw_service_harness
from hyperloom.sandbox.tests.e2b_standin import StandIn


def test_the_service_leg_reads_its_environment_and_ends_the_sandboxes_it_made(
    standin: StandIn, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(SERVICE_URL_ENV, standin.api_url)
    monkeypatch.setenv("HYPERLOOM_E2B_CONTRACT_SANDBOX_URL", standin.sandbox_url)
    monkeypatch.setenv("HYPERLOOM_E2B_CONTRACT_TEMPLATES", "sealed,dev,sub")
    for role, key in standin.keys.items():
        monkeypatch.setenv(f"HYPERLOOM_SANDBOX_KEY_{role.upper()}", key)
    bystander = E2BSandboxProvider(
        {"dev": standin.keys["dev"]}, api_url=standin.api_url, sandbox_url=standin.sandbox_url
    )
    other = bystander.create("dev", role="dev", timeout_sec=60, metadata={"owner": "someone else"})
    with claw_service_harness() as harness:
        assert (harness.sealed_template, harness.dev_template, harness.subrun_template) == ("sealed", "dev", "sub")
        assert harness.claw_policy
        made = harness.provider.create("dev", role="dev", timeout_sec=60, metadata={"owner": "the suite"})
        assert made.run("echo up").stdout == "up\n"
        with pytest.raises(SandboxDenied):
            harness.provider.create("sub", role="dev", timeout_sec=60, metadata={})
    assert [i.sandbox_id for i in bystander.find({}, role="dev")] == [other.sandbox_id]
    other.kill()
