# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fixtures that bind the sandbox contract suite to one implementation.

Another implementation runs the same suite by overriding the ``harness``
fixture (for example in a conftest of its own that wraps its provider).
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from dataclasses import dataclass

import pytest

from hyperloom.sandbox.local import LocalSandboxProvider, LocalTemplate
from hyperloom.sandbox.socket import SandboxProvider

#: The fake runner every sealed template of this suite starts: it waits for
#: /in/spec.json and answers by copying it to /out/status.json.
FAKE_RUNNER = (
    "import os,time,shutil\n"
    "root=os.environ['HYPERLOOM_SANDBOX_ROOT']\n"
    "src=root+'/in/spec.json'\n"
    "while not os.path.exists(src): time.sleep(0.02)\n"
    "time.sleep(0.05)\n"
    "shutil.copy(src, root+'/out/status.json')\n"
    "time.sleep(600)\n"
)


@dataclass(frozen=True)
class Harness:
    """What the contract suite needs from an implementation.

    Attributes:
        provider: The provider under test.
        sealed_template: A measurement-role sealed template that signs, and
            whose runner copies ``/in/spec.json`` to ``/out/status.json``.
        dev_template: An unsealed template for the ``dev`` role.
        subrun_template: An unsealed template for the ``subrun`` role.
    """

    provider: SandboxProvider
    sealed_template: str
    dev_template: str
    subrun_template: str


@pytest.fixture
def harness() -> Iterator[Harness]:
    templates = [
        LocalTemplate(
            "runner", "measurement", command=(sys.executable, "-c", FAKE_RUNNER), gpus=1, sealed=True, signing=True
        ),
        LocalTemplate("dev", "dev"),
        LocalTemplate("sub", "subrun"),
    ]
    with LocalSandboxProvider(templates, gpus=[0, 1, 2, 3], hostname="node-a") as provider:
        yield Harness(provider, "runner", "dev", "sub")
