# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Stage 3-gate tests: Magpie install/preflight is gated by benchmark backend.

The bypass backend must not force a Magpie clone/install/import/patch, while
the default (magpie) path stays byte-for-byte unchanged. These assertions are
structural (source-level) to avoid brittle end-to-end mocking of the large
_preflight()/install.sh flows, and they pin the exact gate the runtime relies
on.
"""

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
    # The Magpie import/install must be guarded by the active backend.
    assert "resolve_backend_name" in src
    assert "_magpie_backend_active" in src
    # The gate wraps the import check and the clone/install branch.
    assert 'import Magpie' in src
    assert "if _magpie_backend_active and" in src
    # InferenceX must NOT be gated away (bypass still needs it): the InferenceX
    # section marker exists and is not inside the magpie-only branch.
    assert "3. InferenceX" in src


def test_preflight_resolves_interpreter_via_backend_not_magpie():
    """A bypass-only environment must not route installs through Magpie's venv.

    The benchmark interpreter is resolved via ``resolve_benchmark_interpreter``
    (sys.executable for bypass, the Magpie venv for Magpie), and preflight no
    longer calls ``_resolve_magpie_python()`` unconditionally up front.
    """
    src = inspect.getsource(preflight_mod._preflight)
    assert "resolve_benchmark_interpreter" in src
    # Ray installs with the backend interpreter, not a hardcoded Magpie one.
    assert "benchmark_python" in src
    assert "_ensure_ray(benchmark_python, pip_extra)" in src
    # Ray availability is probed by importing with the SAME interpreter (not
    # shutil.which, which only inspects PATH and would false-positive on a
    # bypass-only host that has a stray ``ray`` on PATH but cannot import it).
    ray_src = inspect.getsource(preflight_mod._ensure_ray)
    assert '[python_exe, "-c", "import ray"]' in ray_src
    assert "shutil.which" not in ray_src
    # Magpie interpreter is no longer resolved unconditionally in _preflight;
    # it comes through resolve_benchmark_interpreter for the Magpie backend.
    assert "_resolve_magpie_python" not in src


def test_install_sh_gates_magpie_calls():
    install_sh = (
        Path(preflight_mod.__file__).resolve().parent.parent
        / "assets"
        / "install.sh"
    )
    text = install_sh.read_text(encoding="utf-8")
    # Backend-based gate present.
    assert "HYPERLOOM_BENCHMARK_BACKEND" in text
    assert 'if [ "$HYPERLOOM_BENCHMARK_BACKEND_LC" = "bypass" ]; then' in text
    # Normalization must strip ONLY leading/trailing whitespace (mirrors
    # Python's .strip().lower()) so " bypass"/"bypass " skip Magpie at install
    # time too, while an internal-space value like "by pass" stays != "bypass"
    # (matching runtime, which resolves unknown values back to magpie). A blanket
    # ``tr -d '[:space:]'`` would collapse "by pass" -> "bypass" and diverge.
    assert "sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'" in text
    assert "tr -d '[:space:]'" not in text
    # Magpie stages are inside the else branch (only run for non-bypass).
    gate_idx = text.index('HYPERLOOM_BENCHMARK_BACKEND_LC=')
    else_idx = text.index("else", gate_idx)
    fi_idx = text.index("\nfi\n", gate_idx)
    magpie_idx = text.index("ensure_magpie\n", gate_idx)
    patch_idx = text.index("ensure_magpie_atomic_scripts_patch\n", gate_idx)
    assert else_idx < magpie_idx < fi_idx
    assert else_idx < patch_idx < fi_idx
    # InferenceX stays unconditional (after the fi).
    inferencex_idx = text.index("ensure_inferencex\n", fi_idx)
    assert inferencex_idx > fi_idx

def test_lifecycle_delegates_to_bypass_backend(tmp_path, monkeypatch):
    """bypass backend reports server_lifecycle ineligible with a clear reason."""
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

    info = sl.resolve_lifecycle_params(cfg_path)
    # bypass now honors the YAML lifecycle block, so a serving framework
    # with profiling off is eligible.
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
    """bypass lifecycle is ineligible on multi-node (reuse is local-only).

    The same single-node-eligible vllm config must flip to ineligible when
    ``is_multi_node()`` is True; the bypass verdict must not short-circuit past
    the multi-node gate that the Magpie path enforces.
    """
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
