# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for preflight treating the Ray pin as a floor, not an exact version."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.cli import preflight as cli_preflight


@pytest.fixture(autouse=True)
def ray_defaults(monkeypatch):
    monkeypatch.delenv("RAY_VERSION", raising=False)
    monkeypatch.delenv("RAY_CLI_CLICK_MAX_VERSION", raising=False)


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

    def __call__(self, argv, **kwargs):
        specs = [arg for arg in argv if arg.startswith(("ray[", "click"))]
        self.install_specs.append(specs)
        pinned = any(spec.startswith("ray[default]==") for spec in specs)
        code = 0 if (self.pinned_install_ok or not pinned) else 1
        return subprocess.CompletedProcess(argv, code, "", "No matching distribution found")


def _ensure_ray_with(
    monkeypatch,
    recorder: _InstallRecorder,
    *,
    version_after: str,
    first_smoke_stderr: str = "ray import failed",
) -> dict:
    smoke_results = iter(
        [
            subprocess.CompletedProcess([], 1, "", first_smoke_stderr),
            subprocess.CompletedProcess([], 0, f"{version_after}\n", ""),
        ]
    )
    monkeypatch.setattr(cli_preflight, "_ray_smoke", lambda _exe, *_overrides: next(smoke_results))
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


def test_ensure_ray_retries_the_click_ceiling_when_the_floor_is_already_met(monkeypatch) -> None:
    # The node already carries the pinned ray 2.44.1 with a too-new click: a
    # bare ">=2.44.1" floor would be a no-op, so the retry must move click.
    recorder = _InstallRecorder(pinned_install_ok=False)

    result = _ensure_ray_with(
        monkeypatch,
        recorder,
        version_after="2.44.1",
        first_smoke_stderr="ray detected: 2.44.1\nclick version incompatible with Ray CLI: 8.5.0 >= 8.3.0\n",
    )

    assert recorder.install_specs == [
        ["ray[default]==2.44.1", "click<8.3.0"],
        ["click<8.3.0"],
    ]
    assert result["version_after"] == "2.44.1"
    assert result["spec"] == "click<8.3.0"


def test_ensure_ray_retry_asks_for_a_strictly_newer_ray_above_the_floor(monkeypatch) -> None:
    # Ray is past the floor and past the click-gated pin: only a newer ray can
    # repair it, and ">" cannot be satisfied by installing nothing.
    recorder = _InstallRecorder(pinned_install_ok=False)

    result = _ensure_ray_with(
        monkeypatch,
        recorder,
        version_after="2.56.0",
        first_smoke_stderr="ray detected: 2.55.0\nray CLI import failed: ImportError: boom\n",
    )

    assert recorder.install_specs[1] == ["ray[default]>2.55.0"]
    assert result["version_after"] == "2.56.0"


def test_ensure_ray_retry_keeps_an_overridden_click_ceiling(monkeypatch) -> None:
    monkeypatch.setenv("RAY_CLI_CLICK_MAX_VERSION", "8.4.0")
    recorder = _InstallRecorder(pinned_install_ok=False)

    _ensure_ray_with(
        monkeypatch,
        recorder,
        version_after="2.55.0",
        first_smoke_stderr="ray detected: 2.55.0\nclick version incompatible with Ray CLI: 8.5.0 >= 8.4.0\n",
    )

    assert recorder.install_specs[1] == ["click<8.4.0"]


def test_ensure_ray_failure_message_names_every_spec_attempted(monkeypatch) -> None:
    recorder = _InstallRecorder(pinned_install_ok=False)
    smoke_results = iter(
        [
            subprocess.CompletedProcess([], 1, "", "ray detected: 2.44.1\nclick version incompatible"),
            subprocess.CompletedProcess([], 1, "", "click version incompatible with Ray CLI"),
        ]
    )
    monkeypatch.setattr(cli_preflight, "_ray_smoke", lambda _exe, *_overrides: next(smoke_results))
    monkeypatch.setattr(cli_preflight.subprocess, "run", recorder)

    with pytest.raises(RuntimeError) as excinfo:
        cli_preflight._ensure_ray("/usr/bin/python3", [])

    message = str(excinfo.value)
    assert "ray[default]==2.44.1 click<8.3.0 then click<8.3.0" in message
    assert "smoke test still failed" in message


def test_smoke_reports_the_detected_ray_version_before_failing(tmp_path: Path) -> None:
    site = _fake_site(tmp_path, ray_version="2.44.1", click_version="8.5.0")

    result = _run_smoke(site)

    assert result.returncode == 1
    assert cli_preflight._smoke_ray_version(result) == "2.44.1"
