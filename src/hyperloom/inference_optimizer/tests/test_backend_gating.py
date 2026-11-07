# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Stage 3-gate tests: Magpie install in preflight is gated by benchmark backend."""

from __future__ import annotations

from pathlib import Path

from hyperloom.orchestrator.actions.executors import benchmark_backend as bb
from hyperloom.inference_optimizer.cli import preflight as preflight_mod


def test_backend_resolution_controls_magpie_need(monkeypatch):
    # Default + magpie -> Magpie needed; bypass -> not needed.
    monkeypatch.delenv(bb.BENCHMARK_BACKEND_ENV, raising=False)
    assert bb.resolve_backend_name() == "magpie"
    monkeypatch.setenv(bb.BENCHMARK_BACKEND_ENV, "bypass")
    assert bb.resolve_backend_name() == "bypass"
    # Unknown typos normalize to magpie so preflight still installs Magpie.
    monkeypatch.setenv(bb.BENCHMARK_BACKEND_ENV, "magpiee")
    assert bb.resolve_backend_name() == "magpie"


def test_preflight_skips_magpie_install_under_bypass(monkeypatch):
    """A bypass-only host has no Magpie; preflight must neither import nor install it."""

    def _no_subprocess(*_args, **_kwargs):
        raise AssertionError("Magpie must not be probed or installed under bypass")

    recorded = []
    monkeypatch.delenv("MAGPIE_PATH", raising=False)
    monkeypatch.setattr(preflight_mod.subprocess, "run", _no_subprocess)
    monkeypatch.setattr(preflight_mod, "_record_install_step", lambda _event, step, **_kw: recorded.append(step))

    preflight_mod._ensure_magpie(
        None, magpie_python="python3", benchmark_backend="bypass", magpie_active=False, pip_extra=[]
    )

    assert [(step.step_id, step.status, step.skip_reason) for step in recorded] == [
        ("ensure_magpie", "skipped", "benchmark_backend_not_magpie"),
    ]
    assert "MAGPIE_PATH" not in preflight_mod.os.environ


def test_kernel_install_validates_ray_cli_and_serving_slot():
    install_sh = Path(preflight_mod.__file__).resolve().parents[1] / "assets" / "install_kernel_tools.sh"
    text = install_sh.read_text(encoding="utf-8")

    assert 'RAY_VERSION="${RAY_VERSION:-2.44.1}"' in text
    assert 'RAY_INSTALL_SPEC="ray[default]==${RAY_VERSION}"' in text
    # The CLI is certified by what its parser accepts, not by the version of
    # click it was built on, so no click release is pinned or downgraded here.
    assert "RAY_START_ARGV=" in text
    assert "ray start does not accept " in text
    assert "RAY_CLI_CLICK_MAX_VERSION" not in text
    assert "CLICK_INSTALL_SPEC" not in text
    # `main` is a plain wrapper; the commands live on the `cli` group Ray publishes.
    assert "from ray.scripts.scripts import cli" in text

    assert "ray_head_has_serving_slot() {" in text
    assert '"serving_slot" in resources' in text
    assert "ray head already running with serving_slot" in text
    assert "ray head already running without serving_slot; restarting local Ray head" in text
    assert "--resources='{\"serving_slot\": 1}'" in text
