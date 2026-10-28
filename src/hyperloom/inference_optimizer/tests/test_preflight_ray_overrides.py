# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Ray configuration stays consistent between package installation and CLI probes."""

from __future__ import annotations

import subprocess
import sys

import pytest

from hyperloom.inference_optimizer.cli import preflight

from .test_preflight_ray_version_floor import _RAY_START_OPTION_DECLS, write_ray_cli_shim


@pytest.fixture
def ray_packages(tmp_path, monkeypatch):
    ray = tmp_path / "ray"
    (ray / "scripts").mkdir(parents=True)
    (ray / "scripts/__init__.py").write_text("")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.delenv("RAY_VERSION", raising=False)

    def configure(ray_version, start_options=None):
        write_ray_cli_shim(ray, ray_version=ray_version, start_options=start_options)

    return configure


@pytest.mark.parametrize(
    "ray_override,installed_ray",
    [
        (None, "2.44.1"),
        ("", "2.44.1"),
        (None, "2.58.0"),
        ("", "2.58.0"),
        ("2.58.0", "2.58.0"),
        ("2.55.1", "2.55.1"),
    ],
)
def test_configured_installed_ray_avoids_reinstallation(monkeypatch, ray_packages, ray_override, installed_ray):
    ray_packages(installed_ray)
    if ray_override is not None:
        monkeypatch.setenv("RAY_VERSION", ray_override)
    actual_run = subprocess.run

    def probe_only(argv, **kwargs):
        assert argv[:2] == [sys.executable, "-c"], "An already matching runtime must not be reinstalled"
        return actual_run(argv, **kwargs)

    monkeypatch.setattr(preflight.subprocess, "run", probe_only)
    result = preflight._ensure_ray(sys.executable, [])
    assert result["status"] == "already_present"
    assert result["version_after"] == installed_ray
    assert result["spec"] == f"ray[default]=={ray_override or preflight._RAY_MIN_VERSION}"


@pytest.mark.parametrize("ray_version", ["2.44.1", "2.55.0"])
def test_override_keeps_version_check(ray_packages, ray_version):
    ray_packages(ray_version)
    result = preflight._ray_smoke(sys.executable, "2.58.0")
    assert result.returncode != 0
    assert "version mismatch" in result.stderr


def test_override_keeps_cli_import_check(ray_packages, tmp_path):
    ray_packages("2.58.0")
    (tmp_path / "ray/scripts/scripts.py").write_text("raise ImportError('broken CLI')\n")
    result = preflight._ray_smoke(sys.executable, "2.58.0")
    assert result.returncode != 0
    assert "ray CLI import failed" in result.stderr


def test_override_keeps_the_cli_parse_check(ray_packages):
    # Whatever click the override's Ray is built on, the smoke passes exactly
    # when that CLI parses the argv the orchestrator hands `ray start`.
    offered = {opt: _RAY_START_OPTION_DECLS[opt] for opt in ("--head", "--port")}
    ray_packages("2.58.0", start_options=offered)
    result = preflight._ray_smoke(sys.executable, "2.58.0")
    assert result.returncode != 0
    assert f"ray start does not accept {' '.join(preflight._RAY_START_ARGV)}" in result.stderr

    ray_packages("2.58.0")
    assert preflight._ray_smoke(sys.executable, "2.58.0").returncode == 0


@pytest.mark.parametrize("post_install_returncode", [0, 1])
def test_install_and_recheck_share_overrides(monkeypatch, post_install_returncode):
    monkeypatch.setenv("RAY_VERSION", "2.58.0")
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        code = 1 if len(calls) == 1 else post_install_returncode if len(calls) == 3 else 0
        return subprocess.CompletedProcess(argv, code, stdout="", stderr="probe failed" if code else "")

    monkeypatch.setattr(preflight.subprocess, "run", fake_run)
    if post_install_returncode:
        with pytest.raises(RuntimeError, match="smoke test still failed"):
            preflight._ensure_ray("/runtime/python", ["--break-system-packages"])
    else:
        result = preflight._ensure_ray("/runtime/python", ["--break-system-packages"])
        assert result["status"] == "applied"
        assert result["version_after"] == "2.58.0"
    assert calls[0] == calls[2]
    assert calls[0][-1] == "2.58.0"
    assert calls[1] == [
        "/runtime/python",
        "-m",
        "pip",
        "install",
        "--quiet",
        "--break-system-packages",
        "ray[default]==2.58.0",
    ]


def test_explicit_ray_pin_does_not_fall_back_to_another_release(monkeypatch):
    monkeypatch.setenv("RAY_VERSION", "2.58.0")
    calls = []

    def fail_run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="No matching distribution found")

    monkeypatch.setattr(preflight.subprocess, "run", fail_run)
    with pytest.raises(RuntimeError, match=r"Ray install failed for ray\[default\]==2.58.0"):
        preflight._ensure_ray("/runtime/python", [])
    assert len(calls) == 2
    assert calls[1][-1] == "ray[default]==2.58.0"


def test_floor_fallback_when_the_default_ray_pin_cannot_resolve(monkeypatch):
    monkeypatch.delenv("RAY_VERSION", raising=False)
    calls = []
    results = iter([(1, ""), (1, ""), (0, ""), (0, "2.58.0\n")])

    def fake_run(argv, **kwargs):
        calls.append(argv)
        code, stdout = next(results)
        return subprocess.CompletedProcess(argv, code, stdout=stdout, stderr="unavailable" if code else "")

    monkeypatch.setattr(preflight.subprocess, "run", fake_run)
    result = preflight._ensure_ray("/runtime/python", [])
    assert calls[0] == calls[3]
    assert calls[0][-1] == ""
    assert calls[1][-1] == "ray[default]==2.44.1"
    assert calls[2][-1] == "ray[default]>=2.44.1"
    assert result["spec"] == "ray[default]>=2.44.1"
    assert result["version_after"] == "2.58.0"


def test_newer_ray_and_click_pass_without_an_override(ray_packages):
    # The pair that preflight used to refuse: the CLI parses what Hyperloom
    # passes, so the probe accepts it and no repair is attempted.
    ray_packages("2.44.1")
    result = preflight._ray_smoke(sys.executable, "")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "2.44.1"
