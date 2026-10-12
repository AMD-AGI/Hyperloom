# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The profile run's start-up shim keeps GPU events in spawned engine traces."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from hyperloom.orchestrator.actions.executors import _framework_rewrite_evidence as evidence

_REPORT = (
    "import os, hl_host_probe; "
    "print(os.environ.get('ROCPROFILER_REGISTER_LIBRARY', '<unset>'), "
    "os.environ.get('HL_OVERLAY_RUNS'), hl_host_probe.active() is not None)"
)


def _run(tmp_path: Path, entries: list[str], **env_overrides: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("HYPERLOOM_HOST_PROBE")}
    env["PYTHONPATH"] = os.pathsep.join(entries)
    env["ROCPROFILER_REGISTER_LIBRARY"] = "/opt/rocm/lib/librocprofiler-sdk.so.1"
    env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-c", _REPORT],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )


def _overlay(tmp_path: Path, body: str = "") -> str:
    """A non-chaining ``sitecustomize``, as an authored-kernel overlay writes."""
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "sitecustomize.py").write_text(
        "import os\nos.environ['HL_OVERLAY_RUNS'] = os.environ.get('HL_OVERLAY_RUNS', '') + 'x'\n" + body
    )
    return str(overlay)


def test_shim_drops_the_variable_and_chains_the_overlay_once(tmp_path):
    """Without the probe armed: the variable is gone, the overlay loads once, no probe is installed."""
    out = _run(tmp_path, [str(evidence.probe_asset_dir()), _overlay(tmp_path)])

    assert out.stdout.split() == ["<unset>", "x", "False"]


def test_armed_probe_installs_after_the_overlay(tmp_path):
    probe_dir = tmp_path / "probe"
    probe_dir.mkdir()
    out = _run(
        tmp_path,
        [str(evidence.probe_asset_dir()), _overlay(tmp_path)],
        HYPERLOOM_HOST_PROBE="1",
        HYPERLOOM_HOST_PROBE_DIR=str(probe_dir),
    )

    assert out.stdout.split() == ["<unset>", "x", "True"]


def test_a_failing_overlay_is_reported_and_the_probe_still_installs(tmp_path):
    probe_dir = tmp_path / "probe"
    probe_dir.mkdir()
    out = _run(
        tmp_path,
        [str(evidence.probe_asset_dir()), _overlay(tmp_path, "raise RuntimeError('overlay broke')\n")],
        HYPERLOOM_HOST_PROBE="1",
        HYPERLOOM_HOST_PROBE_DIR=str(probe_dir),
    )

    assert "overlay broke" in out.stderr
    assert out.stdout.split() == ["<unset>", "x", "True"]
