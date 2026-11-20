# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Preflight Ray provisioning: smoke-test an interpreter's Ray CLI and install or repair Ray for it."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any


# A floor rather than an exact requirement: interpreters with no 2.44.1 wheel
# (cp314 postdates it) must be allowed to keep the newer release the
# kernel-agent installer resolved for them.
_RAY_MIN_VERSION = "2.44.1"
# Ray 2.44.1's CLI does not import under click 8.3+ ("ValueError: ... is not a valid Sentinel"). The smoke
# exits with this code for that failure on that Ray alone, the one failure a fixed click hold repairs;
# install_kernel_tools.sh applies the same rule.
_RAY_CLICK_HOLD_VERSION = "2.44.1"
_RAY_CLICK_HOLD_SPEC = "click<8.3"
_RAY_SENTINEL_EXIT = 75
# A representative `ray start` argv of the orchestrator's Ray runtime: every option it passes to the head
# this step provisions, with values of its kind. Ray's own parser reading this argv is the certification, so
# a pair whose `ray start` parses what Hyperloom passes is never refused over a version number.
_RAY_START_ARGV = (
    *"--head --port=6379 --dashboard-host=127.0.0.1 --dashboard-port=8265".split(),
    *"--ray-client-server-port=10001 --num-gpus=8 --resources".split(),
    '{"serving_slot": 1}',
)


_RAY_SMOKE_SCRIPT = Path(__file__).resolve().parent.parent / "assets" / "ray_cli_smoke.py"


def _ray_probe_env() -> dict[str, str]:
    """Ray refuses to import on ROCm when only ROCR_VISIBLE_DEVICES is set, and
    preflight clears HIP_VISIBLE_DEVICES for the benchmark path; restore a
    re-indexed value for Ray's own probes so they are not false negatives."""
    env = dict(os.environ)
    if env.get("HIP_VISIBLE_DEVICES"):
        return env
    visible = [part for part in env.get("ROCR_VISIBLE_DEVICES", "").split(",") if part.strip()]
    if visible:
        env["HIP_VISIBLE_DEVICES"] = ",".join(str(index) for index in range(len(visible)))
    return env


def _ray_smoke_command(python_exe: str, ray_version: str) -> list[str]:
    hold = (_RAY_CLICK_HOLD_VERSION, str(_RAY_SENTINEL_EXIT))
    return [python_exe, str(_RAY_SMOKE_SCRIPT), _RAY_MIN_VERSION, *hold, *_RAY_START_ARGV, ray_version]


def _ray_smoke(python_exe: str, ray_version: str = "") -> subprocess.CompletedProcess:
    return subprocess.run(
        _ray_smoke_command(python_exe, ray_version),
        capture_output=True,
        text=True,
        env=_ray_probe_env(),
    )


def _pip_install_ray(python_exe: str, pip_extra: list[str], specs: tuple[str, ...]) -> subprocess.CompletedProcess:
    argv = [python_exe, "-m", "pip", "install", "--quiet", *pip_extra, *specs]
    return subprocess.run(argv, capture_output=True, text=True)


def _smoke_after_click_hold(
    python_exe: str, pip_extra: list[str], ray_version: str, check: subprocess.CompletedProcess
) -> subprocess.CompletedProcess:
    """Hold click below 8.3 and smoke again, but only for the Sentinel import failure of Ray 2.44.1."""
    if check.returncode != _RAY_SENTINEL_EXIT:
        return check
    print(f"Preflight: ray {_RAY_CLICK_HOLD_VERSION} cannot import under this click; holding {_RAY_CLICK_HOLD_SPEC}")
    _pip_install_ray(python_exe, pip_extra, (_RAY_CLICK_HOLD_SPEC,))
    return _ray_smoke(python_exe, ray_version)


def _ray_result(status: str, python_exe: str, spec: str, version_after: str, message: str | None) -> dict[str, Any]:
    """The record the Ray step reports, whether it found Ray or had to install it."""
    return {
        "status": status,
        "skip_reason": None,
        "target": "ray",
        "interpreter": python_exe,
        "spec": spec,
        "version_after": version_after,
        "message": message,
    }


def _install_ray(python_exe: str, pip_extra: list[str], ray_version: str, ray_spec: str, reason: str) -> dict[str, Any]:
    """Install Ray for an interpreter whose smoke test refused the Ray it has."""
    specs: tuple[str, ...] = (ray_spec,)
    pinned_detail = ""
    install = _pip_install_ray(python_exe, pip_extra, specs)
    if install.returncode != 0 and not ray_version:
        pinned_detail = (install.stderr or install.stdout or "no pip output").strip()
        specs = (f"ray[default]>={_RAY_MIN_VERSION}",)
        print(f"Preflight: {ray_spec} does not resolve for {python_exe}; retrying with {' '.join(specs)}")
        install = _pip_install_ray(python_exe, pip_extra, specs)
    if install.returncode != 0:
        detail = (install.stderr or install.stdout or "no pip output").strip()
        raise RuntimeError(f"Ray install failed for {' '.join(specs)}: {detail}")
    check = _smoke_after_click_hold(python_exe, pip_extra, ray_version, _ray_smoke(python_exe, ray_version))
    if check.returncode != 0:
        detail = (check.stderr or check.stdout or "unknown Ray smoke failure").strip()
        # The floor spec is satisfied by whatever Ray is already installed, so a
        # zero exit from pip does not mean anything was installed: report what
        # the repair really did rather than claiming it completed.
        earlier = f" after {ray_spec} failed to install ({pinned_detail})" if pinned_detail else ""
        raise RuntimeError(
            f"pip install {' '.join(specs)} succeeded{earlier} but the Ray smoke test still failed: {detail}"
        )
    version_after = (check.stdout or "").strip() or ray_version or _RAY_MIN_VERSION
    print(f"Preflight: ray installed OK ({version_after})")
    return _ray_result("applied", python_exe, " ".join(specs), version_after, reason)


def _ensure_ray(python_exe: str, pip_extra: list[str]) -> dict[str, Any]:
    """Probe-then-install Ray using the interpreter that will import it."""
    ray_version = os.environ.get("RAY_VERSION", "")
    ray_spec = f"ray[default]=={ray_version or _RAY_MIN_VERSION}"
    check = _ray_smoke(python_exe, ray_version)
    if check.returncode == _RAY_SENTINEL_EXIT:
        held = _smoke_after_click_hold(python_exe, pip_extra, ray_version, check)
        if held.returncode == 0:
            version_after = held.stdout.strip() or ray_version or _RAY_MIN_VERSION
            reason = "ray CLI Sentinel import failure"
            return _ray_result("applied", python_exe, _RAY_CLICK_HOLD_SPEC, version_after, reason)
        check = held
    if check.returncode == 0:
        print("Preflight: ray OK")
        version_after = check.stdout.strip() or ray_version or _RAY_MIN_VERSION
        return _ray_result("already_present", python_exe, ray_spec, version_after, None)
    reason = (check.stderr or check.stdout or "unknown Ray smoke failure").strip().splitlines()[-1]
    print(f"Preflight: ray unusable ({reason}), installing {ray_spec} ...")
    return _install_ray(python_exe, pip_extra, ray_version, ray_spec, reason)
