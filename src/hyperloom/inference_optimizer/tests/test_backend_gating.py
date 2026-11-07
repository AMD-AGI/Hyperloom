# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Stage 3-gate tests: Magpie install in preflight and install.sh is gated by benchmark backend."""

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


def test_preflight_skips_magpie_install_and_patches_under_bypass(monkeypatch):
    """A bypass-only host has no Magpie; preflight must neither import, install nor patch it."""

    def _no_subprocess(*_args, **_kwargs):
        raise AssertionError("Magpie must not be probed or installed under bypass")

    recorded = []
    monkeypatch.delenv("MAGPIE_PATH", raising=False)
    monkeypatch.setattr(preflight_mod.subprocess, "run", _no_subprocess)
    monkeypatch.setattr(preflight_mod, "_ensure_client_trust_compat", _no_subprocess)
    monkeypatch.setattr(preflight_mod, "_ensure_eval_concurrency_compat", _no_subprocess)
    monkeypatch.setattr(preflight_mod, "_report_inferencex_patch_anchors", lambda _path: True)
    monkeypatch.setattr(preflight_mod, "_record_install_step", lambda _event, step, **_kw: recorded.append(step))

    preflight_mod._ensure_magpie(
        None, magpie_python="python3", benchmark_backend="bypass", magpie_active=False, pip_extra=[]
    )
    preflight_mod._patch_magpie_compat(None, "/nonexistent/InferenceX", magpie_active=False)

    assert [(step.step_id, step.status, step.skip_reason) for step in recorded] == [
        ("ensure_magpie", "skipped", "benchmark_backend_not_magpie"),
        ("patch_magpie_eval_concurrency", "skipped", "benchmark_backend_not_magpie"),
    ]
    assert "MAGPIE_PATH" not in preflight_mod.os.environ


def test_install_sh_gates_magpie_calls():
    install_sh = Path(preflight_mod.__file__).resolve().parent.parent / "assets" / "install.sh"
    text = install_sh.read_text(encoding="utf-8")
    # Backend-based gate present.
    assert "HYPERLOOM_BENCHMARK_BACKEND" in text
    assert 'if [ "$HYPERLOOM_BENCHMARK_BACKEND_LC" = "bypass" ]; then' in text
    # Normalization must strip ONLY leading/trailing whitespace (mirrors Python's .strip().lower()) so "
    # bypass"/"bypass " skip Magpie at install time too, while an internal-space value like "by pass" stays !=
    # "bypass" (matching runtime, which resolves unknown values back to magpie).
    assert "sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'" in text
    assert "tr -d '[:space:]'" not in text
    # ensure_magpie runs only for non-bypass (inside the first else…fi block).
    gate_idx = text.index("HYPERLOOM_BENCHMARK_BACKEND_LC=")
    else_idx = text.index("else", gate_idx)
    fi_idx = text.index("\nfi\n", gate_idx)
    magpie_idx = text.index("ensure_magpie\n", gate_idx)
    assert else_idx < magpie_idx < fi_idx
    # InferenceX stays unconditional (after the fi).
    inferencex_idx = text.index("ensure_inferencex\n", fi_idx)
    assert inferencex_idx > fi_idx
    # The atomic-scripts patch also scrubs the redundant --concurrent-requests eval flag from the InferenceX benchmark
    # copies, so it must run AFTER ensure_inferencex (which exports $INFERENCEX_PATH) and is gated on non-bypass by
    # its own guard rather than the first else branch.
    patch_guard_idx = text.index('if [ "$HYPERLOOM_BENCHMARK_BACKEND_LC" != "bypass" ]; then', gate_idx)
    patch_idx = text.index("ensure_magpie_compat_patches\n", gate_idx)
    patch_fi_idx = text.index("\nfi\n", patch_guard_idx)
    assert inferencex_idx < patch_guard_idx < patch_idx < patch_fi_idx


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
