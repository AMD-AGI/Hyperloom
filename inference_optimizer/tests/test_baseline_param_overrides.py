"""Baseline parameter override tests.

Pins the executor-boundary contract for the two Magpie leak-path
recovery knobs that Orchestration drives via ``task.params``:

* ``benchmark_script`` — rewrites ``benchmark.benchmark_script`` in the
  materialized YAML AFTER the gpu_type-driven pop, so the operator's
  pick wins over Magpie's runner_type → script auto-selection.
* ``result_dir``       — forwarded as ``$RESULT_DIR`` for the Magpie
  subprocess; ``$RESULT_DIR`` defaults to the per-task workspace when
  the override is absent.

Both knobs go through :func:`sanitize_script_name` /
:func:`sanitize_result_dir` at the executor boundary so malformed
overrides land as ``error_class='bad_param'`` instead of an unsafe
subprocess.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from inference_optimizer.orchestrator.action_executors._grid_runner import (
    sanitize_result_dir,
    sanitize_script_name,
)
from inference_optimizer.orchestrator.action_executors._workload_envs import (
    materialize_config_with_envs,
)
from inference_optimizer.orchestrator.action_executors.baseline import (
    BaselineExecutor,
)


@pytest.fixture(autouse=True)
def _isolate_leak_root(tmp_path_factory, monkeypatch):
    """Pin ``INFERENCE_OPTIMIZER_LEAK_ROOTS`` to an empty sandbox so
    ``BaselineExecutor.__call__``'s always-on artifact harvest does
    not scrape the host's real ``/workspace`` directory during this
    test module's subprocess-mocked runs.
    """
    sandbox = tmp_path_factory.mktemp("isolated_leak_root")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_LEAK_ROOTS", str(sandbox))


# ---------------------------------------------------------------------------
# Sanitization
# ---------------------------------------------------------------------------
def test_sanitize_script_name_accepts_bare_filename():
    assert sanitize_script_name("sglang_mi300x.sh") == "sglang_mi300x.sh"
    assert sanitize_script_name(" sglang_mi300x.sh ") == "sglang_mi300x.sh"
    assert sanitize_script_name("DSR1_FP8.sh") == "DSR1_FP8.sh"
    assert sanitize_script_name("a-b_c.0.sh") == "a-b_c.0.sh"


def test_sanitize_script_name_empty_returns_none():
    assert sanitize_script_name(None) is None
    assert sanitize_script_name("") is None
    assert sanitize_script_name("   ") is None


@pytest.mark.parametrize("bad", [
    "../etc/passwd.sh",
    "scripts/sglang.sh",
    "no_extension",
    "with space.sh",
    "trailing.SH",            # case-sensitive *.sh
    "../sglang_mi300x.sh",
    "sglang_mi300x.sh; rm -rf /",
    "$(evil).sh",
])
def test_sanitize_script_name_rejects_unsafe(bad):
    with pytest.raises(ValueError):
        sanitize_script_name(bad)


def test_sanitize_result_dir_accepts_paths():
    assert sanitize_result_dir("/workspace/hyperloom/runs/baseline/t1") == (
        "/workspace/hyperloom/runs/baseline/t1"
    )
    assert sanitize_result_dir("runs/baseline/t1") == "runs/baseline/t1"
    assert sanitize_result_dir(" /tmp/leak ") == "/tmp/leak"


def test_sanitize_result_dir_empty_returns_none():
    assert sanitize_result_dir(None) is None
    assert sanitize_result_dir("") is None
    assert sanitize_result_dir("   ") is None


@pytest.mark.parametrize("bad", [
    "/tmp/with space",
    "/tmp/leak;rm -rf /",
    "/tmp/$(evil)",
    "/tmp/leak`whoami`",
    "/tmp/leak\nrm",
    "/tmp/leak|rm",
    "/tmp/leak&rm",
    "/tmp/leak<other",
])
def test_sanitize_result_dir_rejects_unsafe(bad):
    with pytest.raises(ValueError):
        sanitize_result_dir(bad)


# ---------------------------------------------------------------------------
# materialize_config_with_envs honors benchmark_script after gpu_type pop
# ---------------------------------------------------------------------------
def _write_yaml(path: Path, *, benchmark_script: str | None = None) -> None:
    cfg: dict = {
        "benchmark": {
            "framework": "sglang",
            "model": "/wekafs/models/Qwen-Qwen3-8B",
            "precision": "bf16",
            "run_mode": "local",
            "envs": {"TP": 1, "CONC": 8, "ISL": 256, "OSL": 256},
            "timeout_seconds": 600,
            "profiler": {
                "torch_profiler": {"enabled": False},
                "system_profiler": {"enabled": False},
                "tracelens": {"enabled": False},
            },
            "gpu_selection": {"auto": False},
        }
    }
    if benchmark_script is not None:
        cfg["benchmark"]["benchmark_script"] = benchmark_script
    with path.open("w") as f:
        yaml.safe_dump(cfg, f)


def test_materialize_config_with_envs_pins_benchmark_script_after_gpu_pop(
    tmp_path,
):
    base = tmp_path / "base.yaml"
    _write_yaml(base, benchmark_script="dsr1_fp8_mi300x.sh")
    out = tmp_path / "out"
    out.mkdir()

    materialized = materialize_config_with_envs(
        base, out,
        model_path="/wekafs/models/DeepSeek-R1",
        gpu_type="mi300x",
        benchmark_script="sglang_mi300x.sh",
    )
    cfg = yaml.safe_load(materialized.read_text())
    # gpu_type would normally pop benchmark_script; the override re-pins
    # AFTER that pop so the operator wins.
    assert cfg["benchmark"]["benchmark_script"] == "sglang_mi300x.sh"
    assert cfg["benchmark"]["runner_type"] == "mi300x"


def test_materialize_config_with_envs_forces_generic_without_override(tmp_path):
    """Without an explicit ``benchmark_script`` override, gpu_type now
    force-pins the generic ``{framework}_{gpu_type}.sh`` so Magpie's
    resolver hits priority 1 (explicit user override) and never silently
    falls through to the InferenceX native script (e.g.
    ``dsr1_fp8_mi300x.sh``) which hardcodes ``--result-dir /workspace/``.
    See ``design/magpie-generic-script-and-user-data-path.md`` §3."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, benchmark_script="dsr1_fp8_mi300x.sh")
    out = tmp_path / "out"
    out.mkdir()

    materialized = materialize_config_with_envs(
        base, out,
        gpu_type="mi300x",
    )
    cfg = yaml.safe_load(materialized.read_text())
    # framework in _write_yaml fixture is "sglang".
    assert cfg["benchmark"]["benchmark_script"] == "sglang_mi300x.sh"


# ---------------------------------------------------------------------------
# TP auto-clamp against visible GPU count (real Qwen3-8B failure regression)
# ---------------------------------------------------------------------------
def _write_yaml_with_tp(path: Path, tp: int) -> None:
    """Like ``_write_yaml`` but lets the test pin the YAML's default TP."""
    cfg: dict = {
        "benchmark": {
            "framework": "sglang",
            "model": "/wekafs/models/Qwen-Qwen3-8B",
            "precision": "bf16",
            "run_mode": "local",
            "envs": {"TP": tp, "CONC": 8, "ISL": 256, "OSL": 256},
            "timeout_seconds": 600,
            "profiler": {
                "torch_profiler": {"enabled": False},
                "system_profiler": {"enabled": False},
                "tracelens": {"enabled": False},
            },
            "gpu_selection": {"auto": False},
        }
    }
    with path.open("w") as f:
        yaml.safe_dump(cfg, f)


def test_materialize_config_with_envs_clamps_tp_to_visible_gpus(
    tmp_path, monkeypatch, caplog,
):
    """A 4-GPU pod must not launch sglang/vllm with ``TP=8``.

    Regression: the shipped ``baseline_sglang.yaml`` ships with ``TP: 8``
    (full DGX-style node). On a 4-GPU sandbox the unclamped value caused
    ``HIP error: invalid device ordinal`` deep inside the subprocess, which
    looked like a Magpie failure even though the root cause was the
    operator forgetting to ``export TP=...``. baseline survived because
    Coordinator-driven recovery layered ``--tp 1`` into
    ``EXTRA_SGLANG_ARGS``; profile / params / backends did not inherit
    that recovery and looped forever on TP=8. The materializer is now
    the single source of truth: clamp TP to the actual visible GPU count.
    """
    base = tmp_path / "base.yaml"
    _write_yaml_with_tp(base, tp=8)
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.delenv("TP", raising=False)
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", raising=False)

    with patch(
        "inference_optimizer.orchestrator.action_executors._workload_envs._visible_gpu_count",
        return_value=4,
    ):
        materialized = materialize_config_with_envs(base, out)

    cfg = yaml.safe_load(materialized.read_text())
    assert cfg["benchmark"]["envs"]["TP"] == 4
    assert cfg["benchmark"]["envs"]["ROCR_VISIBLE_DEVICES"] == "0,1,2,3"


def test_materialize_config_with_envs_clamp_respects_env_override(
    tmp_path, monkeypatch,
):
    """When the operator explicitly sets ``$TP``, the clamp still fires
    because the alternative is a subprocess crash. ``DISABLE_TP_CLAMP=1``
    is the documented bypass for the rare case where a user wants to
    deliberately force an oversubscribed launch (e.g. running TP=8 on a
    single MI300X for a controlled OOM repro)."""
    base = tmp_path / "base.yaml"
    _write_yaml_with_tp(base, tp=1)
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setenv("TP", "8")
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")

    with patch(
        "inference_optimizer.orchestrator.action_executors._workload_envs._visible_gpu_count",
        return_value=4,
    ):
        materialized = materialize_config_with_envs(base, out)

    cfg = yaml.safe_load(materialized.read_text())
    # Bypass keeps the operator-requested TP=8 even though it will fail.
    assert cfg["benchmark"]["envs"]["TP"] == 8
    assert cfg["benchmark"]["envs"]["ROCR_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"


def test_materialize_config_with_envs_no_clamp_when_visible_zero(
    tmp_path, monkeypatch,
):
    """``_visible_gpu_count`` returns 0 on CPU-only test boxes / rocm-smi
    failures. The materializer must NOT clamp to 0 (which would render
    ROCR_VISIBLE_DEVICES empty and break sglang's CLI parsing); the
    correct behaviour is to leave the YAML TP intact and let downstream
    surface the real "no GPU" error."""
    base = tmp_path / "base.yaml"
    _write_yaml_with_tp(base, tp=2)
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.delenv("TP", raising=False)
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", raising=False)

    with patch(
        "inference_optimizer.orchestrator.action_executors._workload_envs._visible_gpu_count",
        return_value=0,
    ):
        materialized = materialize_config_with_envs(base, out)

    cfg = yaml.safe_load(materialized.read_text())
    assert cfg["benchmark"]["envs"]["TP"] == 2
    # ROCR was unset upstream → derived from TP (no clamp interference).
    assert cfg["benchmark"]["envs"]["ROCR_VISIBLE_DEVICES"] == "0,1"


# ---------------------------------------------------------------------------
# BaselineExecutor.__call__ end-to-end (subprocess mocked)
# ---------------------------------------------------------------------------
def _fake_workspace(slot: Path, *, tput: float = 1500.0) -> Path:
    import json

    ws = slot / "benchmark_sglang_20260513_010101"
    ws.mkdir(parents=True)
    (ws / "benchmark_report.json").write_text(json.dumps({
        "success": True,
        "framework": "sglang",
        "model": "/wekafs/models/Qwen-Qwen3-8B",
        "throughput": {
            "request_throughput": tput / 256,
            "output_throughput": tput,
            "total_token_throughput": tput * 2,
            "completed_requests": 64,
            "duration_seconds": 25.0,
        },
        "latency": {
            "ttft": {"mean_ms": 100.0, "p99_ms": 120.0},
            "e2el": {"mean_ms": 2000.0, "p99_ms": 2300.0},
        },
    }))
    return ws


def _make_ctx(params: dict) -> SimpleNamespace:
    task = SimpleNamespace(task_id="t-baseline-1", params=params)
    return SimpleNamespace(task=task, extra={})


def _run(coro):
    return asyncio.run(coro)


def test_baseline_executor_forwards_override_to_yaml_and_env(tmp_path):
    base = tmp_path / "base.yaml"
    _write_yaml(base, benchmark_script="dsr1_fp8_mi300x.sh")
    output_dir = tmp_path / "ws"
    captured: dict = {}

    def fake_run(cmd, *args, **kwargs):
        cfg_idx = cmd.index("--benchmark-config")
        out_idx = cmd.index("--output-dir")
        cfg_path = Path(cmd[cfg_idx + 1])
        slot = Path(cmd[out_idx + 1])
        captured["cfg"] = yaml.safe_load(cfg_path.read_text())
        captured["env"] = dict(kwargs.get("env") or {})
        _fake_workspace(slot)
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    executor = BaselineExecutor(
        magpie_python="/opt/venv/bin/python",
        default_config_path=base,
        session_dir=tmp_path,
    )
    ctx = _make_ctx({
        "output_dir": str(output_dir),
        "timeout_sec": 10,
        "model_path": "/wekafs/models/DeepSeek-R1",
        "gpu_type": "mi300x",
        "benchmark_script": "sglang_mi300x.sh",
        "result_dir": str(tmp_path / "redirect_leak"),
    })

    with patch(
        "inference_optimizer.orchestrator.action_executors.baseline."
        "run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    bench = captured["cfg"]["benchmark"]
    assert bench["benchmark_script"] == "sglang_mi300x.sh"
    assert bench["runner_type"] == "mi300x"
    assert captured["env"]["RESULT_DIR"] == str(tmp_path / "redirect_leak")


def test_baseline_executor_defaults_result_dir_to_workspace(tmp_path):
    base = tmp_path / "base.yaml"
    _write_yaml(base)
    output_dir = tmp_path / "ws"
    captured: dict = {}

    def fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        captured["env"] = dict(kwargs.get("env") or {})
        _fake_workspace(slot)
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    executor = BaselineExecutor(
        magpie_python="/opt/venv/bin/python",
        default_config_path=base,
        session_dir=tmp_path,
    )
    ctx = _make_ctx({"output_dir": str(output_dir), "timeout_sec": 10})

    with patch(
        "inference_optimizer.orchestrator.action_executors.baseline."
        "run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    # Always-on default = the per-task workspace.
    assert captured["env"]["RESULT_DIR"] == str(output_dir)


def test_baseline_executor_pins_magpie_inferencex_path(tmp_path, monkeypatch):
    """#210 fix (Deval, comment 8): the baseline executor's Magpie
    subprocess must inherit ``MAGPIE_INFERENCEX_PATH=$INFERENCEX_PATH``
    so Magpie's ``_resolve_default_inferencex_dir`` picks the same
    InferenceX checkout Hyperloom's ``_inferencex_patcher`` patched.
    Symmetric to the ``_grid_runner._run_magpie`` test in
    test_p2_3_param_executors — both Magpie invocation sites must
    set this env."""
    monkeypatch.setenv("INFERENCEX_PATH", "/wekafs/hyperloom/InferenceX")
    base = tmp_path / "base.yaml"
    _write_yaml(base)
    output_dir = tmp_path / "ws"
    captured: dict = {}

    def fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        captured["env"] = dict(kwargs.get("env") or {})
        _fake_workspace(slot)
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    executor = BaselineExecutor(
        magpie_python="/opt/venv/bin/python",
        default_config_path=base,
        session_dir=tmp_path,
    )
    ctx = _make_ctx({"output_dir": str(output_dir), "timeout_sec": 10})

    with patch(
        "inference_optimizer.orchestrator.action_executors.baseline."
        "run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    assert captured["env"].get("MAGPIE_INFERENCEX_PATH") == (
        "/wekafs/hyperloom/InferenceX"
    ), (
        "MAGPIE_INFERENCEX_PATH must equal $INFERENCEX_PATH so Magpie "
        "loads the patched checkout (#210 root cause)"
    )


def test_baseline_executor_rejects_bad_param(tmp_path):
    base = tmp_path / "base.yaml"
    _write_yaml(base)
    output_dir = tmp_path / "ws"

    def fake_run(cmd, *args, **kwargs):  # pragma: no cover — must not run
        raise AssertionError("subprocess.run should not be invoked")

    executor = BaselineExecutor(
        magpie_python="/opt/venv/bin/python",
        default_config_path=base,
        session_dir=tmp_path,
    )
    ctx = _make_ctx({
        "output_dir": str(output_dir),
        "timeout_sec": 10,
        "benchmark_script": "../etc/passwd.sh",
    })

    with patch(
        "inference_optimizer.orchestrator.action_executors.baseline."
        "run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "failed"
    assert result["error_class"] == "bad_param"
    assert "benchmark_script" in result["error"]


def test_baseline_executor_rejects_bad_result_dir(tmp_path):
    base = tmp_path / "base.yaml"
    _write_yaml(base)
    output_dir = tmp_path / "ws"

    def fake_run(cmd, *args, **kwargs):  # pragma: no cover
        raise AssertionError("subprocess.run should not be invoked")

    executor = BaselineExecutor(
        magpie_python="/opt/venv/bin/python",
        default_config_path=base,
        session_dir=tmp_path,
    )
    ctx = _make_ctx({
        "output_dir": str(output_dir),
        "timeout_sec": 10,
        "result_dir": "/tmp/leak;rm -rf /",
    })

    with patch(
        "inference_optimizer.orchestrator.action_executors.baseline."
        "run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "failed"
    assert result["error_class"] == "bad_param"
    assert "result_dir" in result["error"]
