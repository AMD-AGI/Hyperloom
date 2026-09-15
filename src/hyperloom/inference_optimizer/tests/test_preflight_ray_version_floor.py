# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for preflight treating the Ray pin as a floor, not an exact version."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from hyperloom.inference_optimizer.cli import preflight as cli_preflight


def _fake_site(tmp_path: Path, *, ray_version: str, click_version: str) -> Path:
    """A site dir with an importable ray shim plus ray/click dist metadata."""
    site = tmp_path / "site"
    ray_pkg = site / "ray"
    (ray_pkg / "scripts").mkdir(parents=True)
    (ray_pkg / "__init__.py").write_text(f'__version__ = "{ray_version}"\n')
    (ray_pkg / "scripts" / "__init__.py").write_text("")
    (ray_pkg / "scripts" / "scripts.py").write_text("def main():\n    return 0\n")
    for name, version in (("ray", ray_version), ("click", click_version)):
        dist = site / f"{name}-{version}.dist-info"
        dist.mkdir()
        (dist / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
    return site


def _run_smoke(site: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", cli_preflight._RAY_SMOKE],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(site), "PATH": "/usr/bin:/bin"},
    )


def test_smoke_accepts_newer_ray_with_newer_click(tmp_path: Path) -> None:
    # The kernel-agent installer picks 2.55.0 on interpreters without a 2.44.1
    # wheel; preflight must not reject it, nor its click 8.5.0.
    site = _fake_site(tmp_path, ray_version="2.55.0", click_version="8.5.0")

    result = _run_smoke(site)

    assert result.returncode == 0, result.stderr


def test_smoke_still_enforces_click_ceiling_on_the_pinned_ray(tmp_path: Path) -> None:
    site = _fake_site(tmp_path, ray_version="2.44.1", click_version="8.5.0")

    result = _run_smoke(site)

    assert result.returncode == 1
    assert "click version incompatible" in result.stderr


def test_smoke_accepts_the_pinned_ray_with_an_allowed_click(tmp_path: Path) -> None:
    site = _fake_site(tmp_path, ray_version="2.44.1", click_version="8.2.1")

    result = _run_smoke(site)

    assert result.returncode == 0, result.stderr


def test_smoke_rejects_ray_below_the_floor(tmp_path: Path) -> None:
    site = _fake_site(tmp_path, ray_version="2.40.0", click_version="8.2.1")

    result = _run_smoke(site)

    assert result.returncode == 1
    assert "ray too old" in result.stderr


def test_probe_env_reindexes_hip_devices_from_rocr(monkeypatch) -> None:
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "3")

    assert cli_preflight._ray_probe_env()["HIP_VISIBLE_DEVICES"] == "0"

    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "2,5,6")
    assert cli_preflight._ray_probe_env()["HIP_VISIBLE_DEVICES"] == "0,1,2"


def test_probe_env_keeps_an_existing_hip_value(monkeypatch) -> None:
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1")

    assert cli_preflight._ray_probe_env()["HIP_VISIBLE_DEVICES"] == "1"


def test_probe_env_leaves_hip_unset_without_rocr(monkeypatch) -> None:
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)

    assert "HIP_VISIBLE_DEVICES" not in cli_preflight._ray_probe_env()


class _InstallRecorder:
    """Stands in for subprocess.run over the pip install attempts."""

    def __init__(self, *, pinned_install_ok: bool) -> None:
        self.pinned_install_ok = pinned_install_ok
        self.install_specs: list[list[str]] = []

    def __call__(self, argv, **kwargs):  # noqa: ANN001, ANN003
        specs = [arg for arg in argv if arg.startswith(("ray[", "click"))]
        self.install_specs.append(specs)
        pinned = any(spec.startswith("ray[default]==") for spec in specs)
        code = 0 if (self.pinned_install_ok or not pinned) else 1
        return subprocess.CompletedProcess(argv, code, "", "No matching distribution found")


def _ensure_ray_with(monkeypatch, recorder: _InstallRecorder, *, version_after: str) -> dict:
    smoke_results = iter(
        [
            subprocess.CompletedProcess([], 1, "", "ray import failed"),
            subprocess.CompletedProcess([], 0, f"{version_after}\n", ""),
        ]
    )
    monkeypatch.setattr(cli_preflight, "_ray_smoke", lambda _exe: next(smoke_results))
    monkeypatch.setattr(cli_preflight.subprocess, "run", recorder)
    return cli_preflight._ensure_ray("/usr/bin/python3", [])


def test_ensure_ray_keeps_the_pin_when_it_resolves(monkeypatch) -> None:
    recorder = _InstallRecorder(pinned_install_ok=True)

    result = _ensure_ray_with(monkeypatch, recorder, version_after="2.44.1")

    assert recorder.install_specs == [["ray[default]==2.44.1", "click<8.3.0"]]
    assert result["version_after"] == "2.44.1"


def test_ensure_ray_falls_back_and_drops_the_click_ceiling(monkeypatch) -> None:
    # 2.44.1 predates cp314 wheels: the pinned install cannot resolve, and the
    # click ceiling must not follow the fallback since it only guards 2.44.1.
    recorder = _InstallRecorder(pinned_install_ok=False)

    result = _ensure_ray_with(monkeypatch, recorder, version_after="2.55.0")

    assert recorder.install_specs == [
        ["ray[default]==2.44.1", "click<8.3.0"],
        ["ray[default]>=2.44.1"],
    ]
    assert result["version_after"] == "2.55.0"
    assert result["spec"] == "ray[default]>=2.44.1"
