# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Preflight's installs: Magpie only for a serving framework, the Ray CLI it validates and the serving slot its Ray head
carries."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.inference_optimizer.cli import preflight as preflight_mod


@pytest.mark.parametrize("framework", ["custom", "xdit"])
def test_preflight_skips_magpie_for_a_scriptable_framework(monkeypatch, framework):
    """A scriptable framework's rounds run on the bypass runner; preflight neither imports nor installs Magpie."""

    def _no_subprocess(*_args, **_kwargs):
        raise AssertionError("Magpie must not be probed or installed for a scriptable framework")

    recorded = []
    monkeypatch.delenv("MAGPIE_PATH", raising=False)
    monkeypatch.setattr(preflight_mod.subprocess, "run", _no_subprocess)
    monkeypatch.setattr(preflight_mod, "_record_install_step", lambda _event, step, **_kw: recorded.append(step))

    preflight_mod._ensure_magpie(None, magpie_python="python3", framework=framework, pip_extra=[])

    assert [(step.step_id, step.status, step.skip_reason) for step in recorded] == [
        ("ensure_magpie", "skipped", "scriptable_framework"),
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
