# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AgentX switch (``HYPERLOOM_AGENTX``) materialization tests.

Contract:
- OFF (unset / falsey / unrecognized) -> default synthetic path, byte-for-byte
  unchanged (zero regression). No ``AGENTX_*`` leakage.
- ON -> authoritative overwrite of ``benchmark_script`` to ``aiperf_client.sh``
  (even when the gpu_type block pre-pinned a synthetic script), plus
  ``MODEL`` / ``RUN_EVAL`` defaults and ``AGENTX_*`` / ``AIPERF_BIN``
  pass-through into ``benchmark.envs``.
- Parsing is defensive: an unrecognized value is treated as OFF, never raises.
"""

from __future__ import annotations

import os
import subprocess
import sys

import yaml

from hyperloom.orchestrator.actions.executors import _workload_envs as we

_AGENTX_ENV_KEYS = (
    "HYPERLOOM_AGENTX",
    "AGENTX_DATASET",
    "AGENTX_MAX_CTX",
    "AGENTX_NUM_ENTRIES",
    "AGENTX_WARMUP_DURATION",
    "AGENTX_NUM_WARMUP_SESSIONS",
    "AGENTX_KEEP_SERVER",
    "AIPERF_BIN",
    "WEKA_LOADER_OVERRIDE",
    "RUN_EVAL",
    "MODEL_PATH",
)


def _clear_env(monkeypatch):
    for k in _AGENTX_ENV_KEYS:
        monkeypatch.delenv(k, raising=False)


def _write(path, **bench_extra):
    bench = {"framework": "vllm", "model": "/m", "envs": {}}
    bench.update(bench_extra)
    path.write_text(yaml.safe_dump({"benchmark": bench}), encoding="utf-8")
    return path


def _materialize(src, out, **kw):
    res = we.materialize_config_with_envs(src, out, **kw)
    return yaml.safe_load(res.read_text())["benchmark"]


# ── OFF path: zero regression ────────────────────────────────────────────────
def test_switch_off_keeps_synthetic_script(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    src = _write(tmp_path / "base.yaml")
    bench = _materialize(src, tmp_path / "out", gpu_type="mi300x", model_path="/m")
    assert bench["benchmark_script"] == "vllm_mi300x.sh"
    assert not any(k.startswith("AGENTX") for k in bench.get("envs", {}))


def test_switch_off_no_agentx_leakage(tmp_path, monkeypatch):
    """OFF output must carry no trace of the AgentX feature.

    Byte-for-byte zero-regression vs the shipped baseline is separately locked by
    test_workload_envs_golden_lock; here we assert no AgentX/aiperf key or value
    leaks into the default synthetic materialization (a stronger check than
    comparing two identical OFF runs, which determinism alone would satisfy).
    """
    _clear_env(monkeypatch)
    src = _write(tmp_path / "base.yaml")
    bench = _materialize(src, tmp_path / "out", gpu_type="mi300x", model_path="/m")
    dumped = yaml.safe_dump(bench).lower()
    assert "aiperf" not in dumped
    assert "agentx" not in dumped
    assert bench["benchmark_script"] == "vllm_mi300x.sh"


# ── ON path: authoritative overwrite + env injection ─────────────────────────
def test_switch_on_authoritative_overwrite(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    src = _write(tmp_path / "base.yaml")
    # gpu_type pre-pins vllm_mi300x.sh; the switch must overwrite it.
    bench = _materialize(src, tmp_path / "out", gpu_type="mi300x", model_path="/m")
    assert bench["benchmark_script"] == "aiperf_client.sh"
    assert bench["envs"]["AGENTX_PHASE_WAIT_TIMEOUT_S"] == str(bench["timeout_seconds"])


def test_persisted_agentx_mode_switches_without_ambient_env(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    src = _write(tmp_path / "base.yaml")
    bench = _materialize(
        src,
        tmp_path / "out",
        gpu_type="mi300x",
        model_path="/m",
        agentx_mode=True,
    )
    assert bench["benchmark_script"] == "aiperf_client.sh"


def test_switch_on_injects_model_and_run_eval(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "true")
    src = _write(tmp_path / "base.yaml")
    bench = _materialize(src, tmp_path / "out", gpu_type="mi300x", model_path="/model/x")
    envs = bench["envs"]
    assert envs["RUN_EVAL"] == "false"
    assert envs["MODEL"] == "/model/x"


def test_switch_on_passes_agentx_env(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "on")
    monkeypatch.setenv("AGENTX_DATASET", "semianalysis-cc-traces-weka-with-subagents")
    monkeypatch.setenv("AGENTX_NUM_ENTRIES", "8")
    monkeypatch.setenv("AIPERF_BIN", "/venv/bin/aiperf")
    src = _write(tmp_path / "base.yaml")
    bench = _materialize(src, tmp_path / "out", gpu_type="mi300x", model_path="/m")
    envs = bench["envs"]
    assert envs["AGENTX_DATASET"] == "semianalysis-cc-traces-weka-with-subagents"
    assert envs["AGENTX_NUM_ENTRIES"] == "8"
    assert envs["AIPERF_BIN"] == "/venv/bin/aiperf"


def test_switch_forwards_weka_loader_override(tmp_path, monkeypatch):
    """Upstream's own corpus pin has no ``AGENTX_`` prefix.

    ``aiperf_client.sh`` documents ``WEKA_LOADER_OVERRIDE`` as a supported knob,
    but the prefix loop would drop it -- leaving it to work only when the
    benchmark process happens to inherit the full parent environment, which is
    the class of silent difference this path exists to remove.
    """
    _clear_env(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    monkeypatch.setenv("WEKA_LOADER_OVERRIDE", "semianalysis_cc_traces_weka_062126")
    src = _write(tmp_path / "base.yaml")
    bench = _materialize(src, tmp_path / "out", gpu_type="mi300x", model_path="/m")
    assert bench["envs"]["WEKA_LOADER_OVERRIDE"] == "semianalysis_cc_traces_weka_062126"


def test_switch_off_does_not_leak_weka_loader_override(tmp_path, monkeypatch):
    """The synthetic path must not gain an env it never had."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("WEKA_LOADER_OVERRIDE", "semianalysis_cc_traces_weka_062126")
    src = _write(tmp_path / "base.yaml")
    bench = _materialize(src, tmp_path / "out", gpu_type="mi300x", model_path="/m")
    assert "WEKA_LOADER_OVERRIDE" not in (bench.get("envs") or {})


# ── A3: defensive parsing ────────────────────────────────────────────────────
def test_switch_unrecognized_value_is_off_no_raise(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "ture")  # typo -> OFF, must not raise
    src = _write(tmp_path / "base.yaml")
    bench = _materialize(src, tmp_path / "out", gpu_type="mi300x", model_path="/m")
    assert bench["benchmark_script"] == "vllm_mi300x.sh"


def test_switch_only_serving_frameworks(tmp_path, monkeypatch):
    """Scriptable (image) frameworks must never be swapped to aiperf."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    src = _write(tmp_path / "base.yaml", framework="xdit")
    bench = _materialize(src, tmp_path / "out", model_path="/m")
    assert bench.get("benchmark_script") != "aiperf_client.sh"


# ── Regression: the shared grid/baseline/profile rebuild path (E1 bug) ────────
def test_runtime_overrides_honor_agentx_on(monkeypatch):
    """apply_runtime_benchmark_overrides must apply the switch, else the
    gpu_type-derived synthetic script silently reverts a materialize-time swap
    (the exact defect E1 caught: run_grid rebuilt to vllm_mi300x.sh)."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    from hyperloom.orchestrator.actions.executors._grid_server_args import (
        apply_runtime_benchmark_overrides,
    )

    bench = {"framework": "vllm"}
    apply_runtime_benchmark_overrides(bench, model_path="/m", gpu_type="mi300x")
    assert bench["benchmark_script"] == "aiperf_client.sh"
    assert bench["envs"]["RUN_EVAL"] == "false"


def test_runtime_overrides_off_keeps_synthetic(monkeypatch):
    _clear_env(monkeypatch)  # HYPERLOOM_AGENTX cleared => OFF
    from hyperloom.orchestrator.actions.executors._grid_server_args import (
        apply_runtime_benchmark_overrides,
    )

    bench = {"framework": "vllm"}
    apply_runtime_benchmark_overrides(bench, model_path="/m", gpu_type="mi300x")
    assert bench["benchmark_script"] == "vllm_mi300x.sh"


def test_runtime_overrides_preserve_materialized_agentx_without_env(monkeypatch):
    _clear_env(monkeypatch)
    from hyperloom.orchestrator.actions.executors._grid_server_args import (
        apply_runtime_benchmark_overrides,
    )

    bench = {"framework": "vllm", "benchmark_script": "aiperf_client.sh"}
    apply_runtime_benchmark_overrides(bench, model_path="/m", gpu_type="mi300x")
    assert bench["benchmark_script"] == "aiperf_client.sh"


# ── A2: OFF path never imports the agentx package (lazy-import guarantee) ─────
_PROBE_OFF = """
import sys
from pathlib import Path
from hyperloom.orchestrator.actions.executors import _grid_runner  # noqa: F401
from hyperloom.orchestrator.actions.executors import _workload_envs as we

assert not we.agentx_enabled()
we.materialize_config_with_envs(
    Path(sys.argv[1]), Path(sys.argv[2]), gpu_type="mi300x", model_path="/m"
)
leaked = [m for m in sys.modules if m.startswith("hyperloom.inference_optimizer.agentx")]
assert not leaked, "agentx package imported on OFF path: %r" % (leaked,)
print("OFF_OK")
"""


def test_agentx_package_not_imported_on_off_path(tmp_path):
    """A2: the default (OFF) benchmark path must never import the agentx package.

    ``sys.modules`` is process-global and sibling tests import the agentx
    package, so a same-process assertion would false-negative. A fresh
    interpreter runs the OFF materialize path (importing the core executor
    modules too) and proves the agentx deploy/preflight/runtime imports stay
    lazy behind ``agentx_enabled`` -- a default install loads none of them.
    """
    src = tmp_path / "base.yaml"
    out = tmp_path / "out"
    src.write_text("benchmark:\n  framework: vllm\n  model: /m\n  envs: {}\n", encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("HYPERLOOM_AGENTX", "AGENTX_", "AIPERF"))}
    r = subprocess.run(
        [sys.executable, "-c", _PROBE_OFF, str(src), str(out)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, "stdout=%s\nstderr=%s" % (r.stdout, r.stderr)
    assert "OFF_OK" in r.stdout


_PROBE_CONTRAST = """
import sys

assert not any(m.startswith("hyperloom.inference_optimizer.agentx") for m in sys.modules)
from hyperloom.inference_optimizer.agentx.runtime import maybe_prepare_agentx  # noqa: F401

assert any(m.startswith("hyperloom.inference_optimizer.agentx") for m in sys.modules)
print("CONTRAST_OK")
"""


def test_agentx_package_importable_contrast():
    """Contrast: the agentx package is real and importable (the ON _grid_runner
    branch does exactly this lazy import), so the OFF assertion is not vacuous."""
    r = subprocess.run(
        [sys.executable, "-c", _PROBE_CONTRAST],
        env=dict(os.environ),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, "stdout=%s\nstderr=%s" % (r.stdout, r.stderr)
    assert "CONTRAST_OK" in r.stdout


# ── Finding 1: framework injected so the wrapper delegates to the right builtin ─
def test_switch_on_injects_framework_for_delegation(tmp_path, monkeypatch):
    """ON must inject ``benchmark.framework`` into ``envs.FRAMEWORK`` so
    aiperf_client.sh delegates to ``{framework}_{gpu}.sh``. Without it an sglang
    task falls back to the wrapper default and would boot vllm."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    for fw in ("sglang", "vllm"):
        src = _write(tmp_path / f"{fw}.yaml", framework=fw)
        bench = _materialize(src, tmp_path / f"out_{fw}", gpu_type="mi300x", model_path="/m")
        assert bench["benchmark_script"] == "aiperf_client.sh"
        assert bench["envs"]["FRAMEWORK"] == fw
