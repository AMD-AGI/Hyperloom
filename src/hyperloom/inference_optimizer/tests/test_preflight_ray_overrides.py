# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Ray configuration stays consistent between package installation and CLI probes."""

from __future__ import annotations

import subprocess
import sys

import pytest

from hyperloom.inference_optimizer.cli import preflight


@pytest.fixture
def ray_packages(tmp_path, monkeypatch):
    ray = tmp_path / "ray"
    (ray / "scripts").mkdir(parents=True)
    (ray / "scripts/__init__.py").write_text("")
    (ray / "scripts/scripts.py").write_text("def main(): pass\n")
    metadata = tmp_path / "click-8.2.1.dist-info"
    metadata.mkdir()
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.delenv("RAY_VERSION", raising=False)
    monkeypatch.delenv("RAY_CLI_CLICK_MAX_VERSION", raising=False)

    def configure(ray_version, click_version):
        (ray / "__init__.py").write_text(f"__version__ = {ray_version!r}\n")
        (metadata / "METADATA").write_text(f"Name: click\nVersion: {click_version}\n")

    return configure


@pytest.mark.parametrize(
    "ray_override,click_override,installed_ray,installed_click",
    [
        (None, None, "2.44.1", "8.2.1"),
        ("", "", "2.44.1", "8.2.1"),
        ("2.58.0", None, "2.58.0", "8.2.1"),
        ("2.55.1", "8.6.0", "2.55.1", "8.5.0"),
    ],
)
def test_configured_installed_ray_avoids_reinstallation(
    monkeypatch, ray_packages, ray_override, click_override, installed_ray, installed_click
):
    ray_packages(installed_ray, installed_click)
    for name, value in [("RAY_VERSION", ray_override), ("RAY_CLI_CLICK_MAX_VERSION", click_override)]:
        if value is not None:
            monkeypatch.setenv(name, value)
    actual_run = subprocess.run

    def probe_only(argv, **kwargs):
        assert argv[:2] == [sys.executable, "-c"], "An already matching runtime must not be reinstalled"
        return actual_run(argv, **kwargs)

    monkeypatch.setattr(preflight.subprocess, "run", probe_only)
    result = preflight._ensure_ray(sys.executable, [])
    assert result["status"] == "already_present"
    assert result["version_after"] == installed_ray
    assert result["spec"] == f"ray[default]=={installed_ray}"


@pytest.mark.parametrize("ray_version,click_version", [("2.44.1", "8.2.1"), ("2.58.0", "8.5.0")])
def test_override_keeps_version_and_click_checks(ray_packages, ray_version, click_version):
    ray_packages(ray_version, click_version)
    result = preflight._ray_smoke(sys.executable, "2.58.0", "8.3.0")
    assert result.returncode != 0
    assert "version mismatch" in result.stderr or "click version incompatible" in result.stderr


def test_override_keeps_cli_import_check(ray_packages, tmp_path):
    ray_packages("2.58.0", "8.2.1")
    (tmp_path / "ray/scripts/scripts.py").write_text("raise ImportError('broken CLI')\n")
    result = preflight._ray_smoke(sys.executable, "2.58.0", "8.3.0")
    assert result.returncode != 0
    assert "ray CLI import failed" in result.stderr


@pytest.mark.parametrize("post_install_returncode", [0, 1])
def test_install_and_recheck_share_overrides(monkeypatch, post_install_returncode):
    monkeypatch.setenv("RAY_VERSION", "2.58.0")
    monkeypatch.setenv("RAY_CLI_CLICK_MAX_VERSION", "8.6.0")
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
    assert calls[0][-2:] == ["2.58.0", "8.6.0"]
    assert calls[1] == [
        "/runtime/python",
        "-m",
        "pip",
        "install",
        "--quiet",
        "--break-system-packages",
        "ray[default]==2.58.0",
        "click<8.6.0",
    ]
