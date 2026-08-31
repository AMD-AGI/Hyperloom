# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the source arm still shares with the framework agent.

The Coordinator no longer shells out to ``fa phase-*`` -- discovery is a
specialist -- so what is left to cover is the repo lookup, the scriptable
framework specs it feeds, and that the agent CLI itself still starts.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

from hyperloom.orchestrator.framework import client as fac

_FA_MODULE = "hyperloom.agents.framework.runtime.cli"


def test_repo_url_for_framework_known_and_unknown() -> None:
    assert fac.repo_url_for_framework("sglang").endswith("sglang.git")
    assert fac.repo_url_for_framework("xdit") == "https://github.com/xdit-project/xDiT.git"
    assert fac.repo_url_for_framework("nope") == ""


def test_scriptable_framework_registry_specs() -> None:
    from hyperloom.inference_optimizer import framework_registry

    xdit_spec = framework_registry.FRAMEWORKS["xdit"]
    assert xdit_spec.repo_url == "https://github.com/xdit-project/xDiT.git"
    assert xdit_spec.kind == framework_registry.SCRIPTABLE
    assert xdit_spec.extra_args_env == "EXTRA_XDIT_ARGS"
    assert xdit_spec.throughput_unit == "img/s"
    assert framework_registry.primary_metric_name("xdit") == "e2el_mean_ms"


def test_module_entry_starts_in_real_subprocess() -> None:
    """Smoke: ``python -m <module> schema`` launches and emits valid JSON."""
    proc = subprocess.run(
        [sys.executable, "-m", _FA_MODULE, "schema"],
        capture_output=True,
        text=True,
        timeout=60,
        env=os.environ.copy(),
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert "schema" in payload.get("subcommands_available", [])
