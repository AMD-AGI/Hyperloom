# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Stage 3-gate tests: Magpie install/preflight is gated by benchmark backend."""

from __future__ import annotations

import inspect
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


def test_preflight_gates_magpie_on_backend():
    src = inspect.getsource(preflight_mod._preflight)
    # Preflight resolves the active backend and hands the gate to the Magpie step.
    assert "resolve_backend_name" in src
    assert "_magpie_backend_active" in src
    assert "magpie_active=_magpie_backend_active" in src
    # The gate wraps the import check and the clone/install branch.
    magpie_src = inspect.getsource(preflight_mod._ensure_magpie)
    assert "import Magpie" in magpie_src
    assert "if magpie_active and" in magpie_src
    # The synthetic bypass backend still needs InferenceX; its preparation is separate from the Magpie install.
    assert "_prepare_inferencex(install_event, magpie_active=_magpie_backend_active)" in src
    assert "_ensure_inferencex" not in magpie_src


def test_preflight_resolves_interpreter_via_backend_not_magpie():
    """A bypass-only environment must not route installs through Magpie's venv."""
    src = inspect.getsource(preflight_mod._preflight)
    assert "resolve_benchmark_interpreter" in src
    # Ray installs with the backend interpreter, not a hardcoded Magpie one.
    assert "benchmark_python" in src
    assert "_ensure_ray(benchmark_python, pip_extra)" in src
    # Ray availability is probed with the SAME interpreter (not shutil.which, which only inspects PATH and would
    # false-positive on a bypass-only host that has a stray ``ray`` on PATH but cannot run Ray correctly).
    ray_src = inspect.getsource(preflight_mod._ensure_ray)
    assert "_ray_smoke(python_exe," in ray_src
    assert "shutil.which" not in ray_src
    # Magpie interpreter is no longer resolved unconditionally in _preflight; it comes through
    # resolve_benchmark_interpreter for the Magpie backend.
    assert "_resolve_magpie_python" not in src


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


def test_lifecycle_delegates_to_bypass_backend(tmp_path, monkeypatch):
    """bypass backend resolves server_lifecycle from the YAML, not the script name."""
    import yaml
    from hyperloom.orchestrator.actions.executors import _server_lifecycle as sl

    monkeypatch.setenv(bb.BENCHMARK_BACKEND_ENV, "bypass")
    cfg = {
        "benchmark": {
            "framework": "vllm",
            "benchmark_script": "vllm_mi300x.sh",  # would be eligible under magpie
            "envs": {"PORT": 8888},
        }
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    # Pin the free-port picker so the port assertion is deterministic (resolve now assigns a per-session free port on
    # the common path).
    monkeypatch.setattr(sl, "_pick_free_port", lambda: 8888)
    info = sl.resolve_lifecycle_params(cfg_path)
    # bypass now honors the YAML lifecycle block, so a serving framework with profiling off is eligible.
    assert info["eligible"] is True
    assert info["framework"] == "vllm"
    assert info["port"] == 8888


def test_lifecycle_magpie_default_unchanged(tmp_path, monkeypatch):
    """Default (magpie) backend keeps script-name-based eligibility."""
    import yaml
    from hyperloom.orchestrator.actions.executors import _server_lifecycle as sl

    monkeypatch.delenv(bb.BENCHMARK_BACKEND_ENV, raising=False)
    cfg = {
        "benchmark": {
            "framework": "vllm",
            "benchmark_script": "vllm_mi300x.sh",  # a Magpie built-in
            "envs": {"PORT": 8888},
            "profiler": {"torch_profiler": {"enabled": False}},
        }
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    info = sl.resolve_lifecycle_params(cfg_path)
    assert info["eligible"] is True  # magpie built-in script -> eligible


def test_lifecycle_magpie_gfx12_eligible(tmp_path, monkeypatch):
    """magpie backend: the R9700 script vllm_gfx12.sh is lifecycle-eligible like the MI scripts."""
    import yaml
    from hyperloom.orchestrator.actions.executors import _server_lifecycle as sl

    monkeypatch.delenv(bb.BENCHMARK_BACKEND_ENV, raising=False)
    monkeypatch.setattr(sl, "_pick_free_port", lambda: 8888)
    cfg = {
        "benchmark": {
            "framework": "vllm",
            "benchmark_script": "vllm_gfx12.sh",
            "envs": {"PORT": 8888},
            "profiler": {"torch_profiler": {"enabled": False}},
        }
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    info = sl.resolve_lifecycle_params(cfg_path)
    assert info["eligible"] is True
    assert info["framework"] == "vllm"
    assert info["port"] == 8888
    assert info["reason"] == ""


def test_lifecycle_magpie_non_builtin_ineligible(tmp_path, monkeypatch):
    """magpie backend: a non-built-in script stays ineligible (unchanged)."""
    import yaml
    from hyperloom.orchestrator.actions.executors import _server_lifecycle as sl

    monkeypatch.delenv(bb.BENCHMARK_BACKEND_ENV, raising=False)
    cfg = {
        "benchmark": {
            "framework": "vllm",
            "benchmark_script": "custom_thing.sh",
            "envs": {"PORT": 8888},
        }
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    info = sl.resolve_lifecycle_params(cfg_path)
    assert info["eligible"] is False


def test_bypass_lifecycle_ineligible_when_profiling(tmp_path, monkeypatch):
    """bypass lifecycle is ineligible when torch_profiler is enabled."""
    import yaml
    from hyperloom.orchestrator.actions.executors import _server_lifecycle as sl

    monkeypatch.setenv(bb.BENCHMARK_BACKEND_ENV, "bypass")
    cfg = {
        "benchmark": {
            "framework": "vllm",
            "envs": {"PORT": 8888},
            "profiler": {"torch_profiler": {"enabled": True}},
        }
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    info = sl.resolve_lifecycle_params(cfg_path)
    assert info["eligible"] is False
    assert "profiler" in info["reason"]


def test_bypass_lifecycle_ineligible_for_non_serving(tmp_path, monkeypatch):
    """bypass lifecycle is ineligible for a non-serving framework."""
    import yaml
    from hyperloom.orchestrator.actions.executors import _server_lifecycle as sl

    monkeypatch.setenv(bb.BENCHMARK_BACKEND_ENV, "bypass")
    cfg = {"benchmark": {"framework": "xdit", "envs": {"PORT": 8888}}}
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    info = sl.resolve_lifecycle_params(cfg_path)
    assert info["eligible"] is False


def test_bypass_lifecycle_ineligible_when_multi_node(tmp_path, monkeypatch):
    """bypass lifecycle is ineligible on multi-node (reuse is local-only)."""
    import yaml
    from hyperloom.orchestrator.actions.executors import _multi_node_env
    from hyperloom.orchestrator.actions.executors import _server_lifecycle as sl

    monkeypatch.setenv(bb.BENCHMARK_BACKEND_ENV, "bypass")
    monkeypatch.setattr(_multi_node_env, "is_multi_node", lambda: True)
    cfg = {
        "benchmark": {
            "framework": "vllm",
            "benchmark_script": "vllm_mi300x.sh",
            "envs": {"PORT": 8888},
        }
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    info = sl.resolve_lifecycle_params(cfg_path)
    assert info["eligible"] is False
    assert "multi-node" in info["reason"]
