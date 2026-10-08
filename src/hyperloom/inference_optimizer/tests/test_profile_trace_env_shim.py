# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The profile run's start-up shim that keeps GPU events in spawned engine traces."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml

from hyperloom.orchestrator.actions.executors import _framework_rewrite_evidence as evidence
from hyperloom.orchestrator.actions.executors.profile import ProfileExecutor

_SHIM_DIR = Path(__file__).resolve().parents[1] / "assets" / "profile_trace_env"


def _write_profile_config(path: Path, envs: dict | None = None) -> None:
    path.write_text(yaml.safe_dump({"benchmark": {"framework": "custom", "envs": dict(envs or {})}}), encoding="utf-8")


def _pythonpath(config: Path) -> list[str]:
    return yaml.safe_load(config.read_text(encoding="utf-8"))["benchmark"]["envs"]["PYTHONPATH"].split(os.pathsep)


def test_shim_goes_right_after_the_host_probe(tmp_path, monkeypatch):
    """The shim sits between the host probe and the rest, and re-arming does not stack it."""
    monkeypatch.delenv(evidence.ENABLE_ENV, raising=False)
    config = tmp_path / "profile.yaml"
    _write_profile_config(config, {"PYTHONPATH": "/opt/overlay:/opt/framework/lib"})
    executor = ProfileExecutor()

    executor._inject_host_probe(config, tmp_path / "ws")
    executor._inject_trace_env_shim(config)
    executor._inject_trace_env_shim(config)

    assert _pythonpath(config) == [
        str(evidence.probe_asset_dir()),
        str(_SHIM_DIR),
        "/opt/overlay",
        "/opt/framework/lib",
    ]


def test_shim_goes_first_without_the_host_probe(tmp_path, monkeypatch):
    monkeypatch.setenv(evidence.ENABLE_ENV, "0")
    config = tmp_path / "profile.yaml"
    _write_profile_config(config, {"PYTHONPATH": "/opt/overlay"})

    ProfileExecutor()._inject_trace_env_shim(config)

    assert _pythonpath(config) == [str(_SHIM_DIR), "/opt/overlay"]


def _run_with_pythonpath(tmp_path: Path, entries: list[str]) -> list[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(entries)
    env["ROCPROFILER_REGISTER_LIBRARY"] = "/opt/rocm/lib/librocprofiler-sdk.so.1"
    env[evidence.ENABLE_ENV] = "0"
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os; print(os.environ.get('ROCPROFILER_REGISTER_LIBRARY', '<unset>'), os.environ.get('HL_OVERLAY_RUNS'))",
        ],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return out.stdout.split()


def _overlay(tmp_path: Path) -> str:
    """A non-chaining ``sitecustomize``, as an authored-kernel overlay writes."""
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "sitecustomize.py").write_text(
        "import os\nos.environ['HL_OVERLAY_RUNS'] = os.environ.get('HL_OVERLAY_RUNS', '') + 'x'\n"
    )
    return str(overlay)


def test_shim_drops_the_variable_and_every_hook_runs_once(tmp_path):
    """Host probe, shim, overlay: the variable is gone and the overlay still loads, once."""
    overlay = _overlay(tmp_path)

    assert _run_with_pythonpath(tmp_path, [str(evidence.probe_asset_dir()), str(_SHIM_DIR), overlay]) == [
        "<unset>",
        "x",
    ]


def test_a_hook_chaining_back_to_the_shim_does_not_loop(tmp_path):
    """The host probe after the shim chains back to it; the second run chains nothing."""
    assert _run_with_pythonpath(tmp_path, [str(_SHIM_DIR), str(evidence.probe_asset_dir())]) == ["<unset>", "None"]
