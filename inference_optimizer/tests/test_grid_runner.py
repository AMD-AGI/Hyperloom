"""Consolidated tests for ``orchestrator.action_executors._grid_runner``.

Combines four previously separate modules:

* abort marker (``_write_variant_abort_marker`` + ``VariantResult.error_class``)
* helper-level units (skip-spec parsing, MN/SN invalid-variant filters,
  ``apply_user_skip_list``, ``VariantResult.to_dict``, sanitisers)
* per-variant mtime gating for Magpie leak-path salvage (regression for
  stale ``inferencex_result.json`` adoption)
* parameter override + ``RESULT_DIR`` plumbing
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from inference_optimizer.orchestrator.action_executors import _grid_runner
from inference_optimizer.orchestrator.action_executors import _grid_runner as gr
from inference_optimizer.orchestrator.action_executors._grid_runner import (
    GridVariant,
    VariantResult,
    _build_variant_yaml,
    _parse_skip_spec,
    _run_magpie,
    apply_multi_node_invalid_variants,
    apply_runtime_benchmark_overrides,
    apply_single_node_invalid_variants,
    apply_user_skip_list,
    resolve_skip_spec,
    run_grid,
)


# ==============================================================================
# Section 1: _write_variant_abort_marker (formerly test_grid_runner_abort_marker.py)
# ==============================================================================


def _read_marker(slot):
    """Convenience: load abort_reason.json from a variant slot dir."""
    marker_path = slot / "abort_reason.json"
    assert marker_path.exists(), f"marker not written at {marker_path}"
    return json.loads(marker_path.read_text(encoding="utf-8"))


def test_write_variant_abort_marker_creates_file_with_expected_fields(tmp_path):
    slot = tmp_path / "variant_00_max_num_seqs_128"
    _grid_runner._write_variant_abort_marker(
        slot,
        variant_name="max_num_seqs_128",
        error_class="mn_server_restart_failed",
        error_summary=(
            "server /health did not return 200 within 1800s "
            "(url=http://10.245.131.67:8888/health, last_err=...)"
        ),
        extra_args="--max-num-seqs 128",
    )
    marker = _read_marker(slot)
    assert marker["variant"] == "max_num_seqs_128"
    assert marker["error_class"] == "mn_server_restart_failed"
    assert "server /health did not return 200" in marker["error"]
    assert marker["extra_args"] == "--max-num-seqs 128"
    assert marker["aborted_at_utc"].endswith("Z")
    assert len(marker["aborted_at_utc"]) == 20


def test_write_variant_abort_marker_truncates_huge_error_summary(tmp_path):
    slot = tmp_path / "variant"
    huge = "x" * 5000
    _grid_runner._write_variant_abort_marker(
        slot,
        variant_name="big",
        error_class="magpie_timeout",
        error_summary=huge,
    )
    marker = _read_marker(slot)
    assert len(marker["error"]) == 2000


def test_write_variant_abort_marker_creates_parent_dirs(tmp_path):
    slot = tmp_path / "deeper" / "than" / "expected" / "variant"
    _grid_runner._write_variant_abort_marker(
        slot,
        variant_name="x",
        error_class="yaml_build_error",
        error_summary="config.yaml unwritable",
    )
    assert (slot / "abort_reason.json").exists()


def test_write_variant_abort_marker_swallows_oserror(monkeypatch, tmp_path, caplog):
    slot = tmp_path / "variant"
    slot.mkdir()

    def boom(*_args, **_kwargs):
        raise OSError("read-only fs")

    monkeypatch.setattr(_grid_runner.Path, "write_text", boom)
    with caplog.at_level("WARNING"):
        _grid_runner._write_variant_abort_marker(
            slot,
            variant_name="x",
            error_class="mn_server_restart_failed",
            error_summary="oops",
        )
    assert any(
        "failed to write abort_reason.json" in r.message
        for r in caplog.records
    )


def test_write_variant_abort_marker_json_is_stable_sorted(tmp_path):
    slot = tmp_path / "variant"
    _grid_runner._write_variant_abort_marker(
        slot,
        variant_name="v",
        error_class="ec",
        error_summary="msg",
    )
    raw = (slot / "abort_reason.json").read_text(encoding="utf-8")
    assert (
        raw.index('"aborted_at_utc"')
        < raw.index('"error"')
        < raw.index('"error_class"')
        < raw.index('"extra_args"')
        < raw.index('"variant"')
    )


def test_grid_runner_emits_expected_error_class_labels():
    src = inspect.getsource(_grid_runner)
    expected = {
        "yaml_build_error",
        "mn_server_restart_failed",
        "magpie_timeout",
        "no_benchmark_workspace",
        "magpie_nonzero_invalid_measurement",
        "benchmark_report_missing",
        "benchmark_report_invalid_metric",
    }
    missing = [label for label in expected if f'"{label}"' not in src]
    assert not missing, f"missing error_class labels in run_grid: {missing}"


def test_variant_result_carries_error_class_field():
    vr = _grid_runner.VariantResult(
        name="x",
        extra_server_args="--foo",
        extra_envs={},
        status="failed",
        error="boom",
        error_class="mn_server_restart_failed",
    )
    d = vr.to_dict()
    assert d["error_class"] == "mn_server_restart_failed"
    assert d["status"] == "failed"

    vr_ok = _grid_runner.VariantResult(
        name="ok", extra_server_args="", extra_envs={}, status="succeeded",
    )
    assert vr_ok.to_dict()["error_class"] == ""


# ==============================================================================
# Section 2: helper-level units (formerly test_grid_runner_helpers_units.py)
# ==============================================================================


class TestResolveSkipSpec:
    def test_returns_empty_when_no_params_and_no_env(self, monkeypatch):
        monkeypatch.delenv("SKIP_VARIANTS", raising=False)
        assert resolve_skip_spec(None) == ""

    def test_env_used_when_params_absent(self, monkeypatch):
        monkeypatch.setenv("SKIP_VARIANTS", "foo,bar")
        assert resolve_skip_spec(None) == "foo,bar"

    def test_params_list_flattened(self):
        assert resolve_skip_spec({"skip_variants": ["a", "b", None]}) == "a,b"

    def test_params_str_passthrough(self):
        assert resolve_skip_spec({"skip_variants": "x"}) == "x"

    def test_params_override_env(self, monkeypatch):
        monkeypatch.setenv("SKIP_VARIANTS", "env-default")
        assert resolve_skip_spec({"skip_variants": "explicit"}) == "explicit"


class TestParseSkipSpec:
    def test_splits_on_commas_and_whitespace(self):
        assert _parse_skip_spec("a, b\nc d") == ["a", "b", "c", "d"]

    def test_empty_input_returns_empty_list(self):
        assert _parse_skip_spec("") == []


class TestMultiNodeInvalidFilter:
    def test_short_circuits_in_single_node(self, monkeypatch):
        grid = [
            GridVariant(name="a", extra_server_args="--cuda-graph-max-bs 8"),
        ]
        kept, dropped = apply_multi_node_invalid_variants(grid)
        assert kept == grid
        assert dropped == []

    def test_drops_undersized_cuda_graph_max_bs_in_multi_node(self, monkeypatch):
        monkeypatch.setattr(gr, "is_multi_node", lambda: True, raising=False)
        from inference_optimizer.orchestrator.action_executors import (
            _multi_node_env as mne,
        )

        monkeypatch.setattr(mne, "is_multi_node", lambda: True)
        monkeypatch.setenv("CONC", "64")
        grid = [
            GridVariant(name="bad", extra_server_args="--cuda-graph-max-bs 8"),
            GridVariant(name="ok",  extra_server_args="--max-num-seqs 128"),
        ]
        kept, dropped = apply_multi_node_invalid_variants(grid)
        assert [k.name for k in kept] == ["ok"]
        assert [d["name"] for d in dropped] == ["bad"]
        assert "CONC=64" in dropped[0]["reason"]


class TestSingleNodeInvalidFilter:
    def test_drops_multi_node_only_in_single_node(self):
        grid = [
            GridVariant(name="legacy", extra_server_args="--foo 1"),
            GridVariant(
                name="mn_only",
                extra_server_args="--enable-deepep-moe",
                note="multi_node_only_moe",
            ),
        ]
        kept, dropped = apply_single_node_invalid_variants(grid)
        assert [k.name for k in kept] == ["legacy"]
        assert [d["name"] for d in dropped] == ["mn_only"]

    def test_short_circuits_in_multi_node(self, monkeypatch):
        from inference_optimizer.orchestrator.action_executors import (
            _multi_node_env as mne,
        )

        monkeypatch.setattr(mne, "is_multi_node", lambda: True)
        grid = [
            GridVariant(
                name="mn_only",
                extra_server_args="--enable-deepep-moe",
                note="multi_node_only_moe",
            ),
        ]
        kept, dropped = apply_single_node_invalid_variants(grid)
        assert [k.name for k in kept] == ["mn_only"]
        assert dropped == []


class TestApplyUserSkipList:
    def test_empty_spec_keeps_grid(self):
        grid = [GridVariant(name="a"), GridVariant(name="b")]
        kept, dropped = apply_user_skip_list(grid, skip_spec="")
        assert [k.name for k in kept] == ["a", "b"]
        assert dropped == []

    def test_exact_name_drop(self):
        grid = [GridVariant(name="alpha"), GridVariant(name="beta")]
        kept, dropped = apply_user_skip_list(grid, skip_spec="alpha")
        assert [k.name for k in kept] == ["beta"]
        assert dropped[0]["name"] == "alpha"

    def test_glob_matches_drop(self):
        grid = [
            GridVariant(name="cuda_graph_max_bs_8"),
            GridVariant(name="cuda_graph_max_bs_32"),
            GridVariant(name="schedule_lpm"),
        ]
        kept, dropped = apply_user_skip_list(grid, skip_spec="cuda_graph_*")
        assert [k.name for k in kept] == ["schedule_lpm"]
        assert {d["name"] for d in dropped} == {
            "cuda_graph_max_bs_8", "cuda_graph_max_bs_32",
        }


class TestVariantResultToDict:
    def test_succeeded_default_shape(self):
        vr = VariantResult(
            name="v", extra_server_args="--foo 1", extra_envs={"A": "1"},
            status="succeeded",
        )
        out = vr.to_dict()
        assert out["status"] == "succeeded"
        assert out["error_class"] == ""

    def test_failed_round_trip_carries_error_class(self):
        vr = VariantResult(
            name="v", extra_server_args="", extra_envs={},
            status="failed", error="boom", error_class="benchmark_report_missing",
        )
        out = vr.to_dict()
        assert out["status"] == "failed"
        assert out["error_class"] == "benchmark_report_missing"
        assert out["error"] == "boom"


class TestSanitizeScriptName:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("ok.sh", "ok.sh"),
            ("  ok.sh ", "ok.sh"),
            (None, None),
            ("", None),
        ],
    )
    def test_accepts_safe_names(self, raw, expected):
        assert gr.sanitize_script_name(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        ["../danger.sh", "with space.sh", "../sub.sh", "abc.sh; rm -rf /"],
    )
    def test_rejects_unsafe(self, raw):
        with pytest.raises(ValueError):
            gr.sanitize_script_name(raw)


class TestSanitizeResultDir:
    def test_accepts_relative_and_absolute(self):
        assert gr.sanitize_result_dir("runs/x") == "runs/x"
        assert gr.sanitize_result_dir("/workspace/out") == "/workspace/out"

    def test_blank_returns_none(self):
        assert gr.sanitize_result_dir(None) is None
        assert gr.sanitize_result_dir("") is None
        assert gr.sanitize_result_dir("   ") is None

    @pytest.mark.parametrize(
        "raw",
        ["/tmp/with space", "/tmp/leak`whoami`", "/tmp/leak;rm"],
    )
    def test_rejects_unsafe(self, raw):
        with pytest.raises(ValueError):
            gr.sanitize_result_dir(raw)


# ==============================================================================
# Section 3: per-variant mtime gating + param overrides
# (formerly test_grid_runner_mtime_gating.py + test_grid_runner_param_overrides.py)
#
# Both source modules used an autouse fixture pinning
# INFERENCE_OPTIMIZER_LEAK_ROOTS to an empty sandbox so the runner's
# always-on artifact harvest doesn't scrape the host's real /workspace
# during the test run. Hoisted to module-level here so it applies to both
# subsections.
# ==============================================================================


@pytest.fixture(autouse=True)
def _isolate_leak_root(request, tmp_path_factory, monkeypatch):
    """Pin ``INFERENCE_OPTIMIZER_LEAK_ROOTS`` to an empty sandbox for the
    grid-runner subprocess tests below. Applied unconditionally: the
    helper-level tests (sections 1/2) never spin up ``run_grid`` so the
    extra env var is harmless there.
    """
    sandbox = tmp_path_factory.mktemp("isolated_leak_root")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_LEAK_ROOTS", str(sandbox))


# ---- mtime gating subsection helpers ----------------------------------------


def _write_baseline_yaml_mtime(path: Path) -> None:
    cfg = {
        "benchmark": {
            "framework": "sglang",
            "model": "/wekafs/models/Qwen-Qwen3-8B",
            "precision": "bf16",
            "run_mode": "local",
            "envs": {"TP": 1, "CONC": 8, "ISL": 256, "OSL": 256},
            "benchmark_script": "sglang_mi300x.sh",
            "timeout_seconds": 600,
            "profiler": {
                "torch_profiler": {"enabled": False},
                "system_profiler": {"enabled": False},
                "tracelens": {"enabled": False},
            },
            "gpu_selection": {"auto": False},
        },
    }
    with path.open("w") as f:
        yaml.safe_dump(cfg, f)


def _empty_workspace(slot: Path) -> Path:
    ws = slot / "benchmark_sglang_20260513_010101"
    ws.mkdir(parents=True)
    (ws / "benchmark_report.json").write_text(json.dumps({
        "success": False,
        "framework": "sglang",
        "model": "/wekafs/models/Qwen-Qwen3-8B",
    }))
    return ws


def _write_leak(path: Path, *, tput: float = 1761.6, completed: int = 640) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "output_throughput": tput,
        "request_throughput": tput / 10,
        "completed_requests": completed,
        "duration_seconds": 120.0,
    }))


@pytest.mark.asyncio
async def test_run_grid_rejects_stale_leak_from_previous_run(
    tmp_path, monkeypatch,
):
    base = tmp_path / "base.yaml"
    _write_baseline_yaml_mtime(base)
    output_root = tmp_path / "out"

    leak_dir = tmp_path / "stale_leak"
    leak_path = leak_dir / "inferencex_result.json"
    _write_leak(leak_path, tput=9999.0)
    stale_mtime = time.time() - 3600.0
    os.utime(leak_path, (stale_mtime, stale_mtime))
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RESCUE_PATHS", str(leak_dir))

    def fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _empty_workspace(slot)
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    with patch(
        "inference_optimizer.orchestrator.action_executors._grid_runner."
        "run_with_session_kill",
        side_effect=fake_run,
    ):
        results = await run_grid(
            base_yaml_path=base, base_extra_args="",
            grid=[GridVariant("vA")], output_root=output_root,
            variant_timeout_sec=5,
        )

    assert len(results) == 1
    r = results[0]
    assert r.status == "failed"
    assert r.output_throughput is None
    assert all(
        "rescued_from_leaked_path" not in (w or "")
        for w in r.nonfatal_warnings
    )


@pytest.mark.asyncio
async def test_run_grid_salvages_fresh_leak_per_variant(tmp_path, monkeypatch):
    base = tmp_path / "base.yaml"
    _write_baseline_yaml_mtime(base)
    output_root = tmp_path / "out"

    leak_dir = tmp_path / "fresh_leak"
    leak_dir.mkdir()
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RESCUE_PATHS", str(leak_dir))

    def fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _empty_workspace(slot)
        _write_leak(leak_dir / "inferencex_result.json", tput=1234.0)
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    with patch(
        "inference_optimizer.orchestrator.action_executors._grid_runner."
        "run_with_session_kill",
        side_effect=fake_run,
    ):
        results = await run_grid(
            base_yaml_path=base, base_extra_args="",
            grid=[GridVariant("vA")], output_root=output_root,
            variant_timeout_sec=5,
        )

    assert len(results) == 1
    r = results[0]
    assert r.status == "succeeded"
    assert r.output_throughput == pytest.approx(1234.0)
    assert any(
        (w or "").startswith("rescued_from_leaked_path:")
        for w in r.nonfatal_warnings
    )


# ---- param overrides subsection helpers -------------------------------------


def _write_baseline_yaml_overrides(path: Path) -> None:
    cfg = {
        "benchmark": {
            "framework": "sglang",
            "model": "/wekafs/models/Qwen-Qwen3-8B",
            "precision": "bf16",
            "run_mode": "local",
            "envs": {"TP": 1, "CONC": 8, "ISL": 256, "OSL": 256},
            "benchmark_script": "dsr1_fp8_mi300x.sh",
            "timeout_seconds": 600,
            "profiler": {
                "torch_profiler": {"enabled": False},
                "system_profiler": {"enabled": False},
                "tracelens": {"enabled": False},
            },
            "gpu_selection": {"auto": False},
        },
    }
    with path.open("w") as f:
        yaml.safe_dump(cfg, f)


def _fake_workspace(slot: Path, *, tput: float = 800.0) -> Path:
    workspace = slot / "benchmark_sglang_20260513_001122"
    workspace.mkdir(parents=True)
    (workspace / "benchmark_report.json").write_text(json.dumps({
        "success": True,
        "framework": "sglang",
        "model": "/wekafs/models/Qwen-Qwen3-8B",
        "throughput": {
            "request_throughput": tput / 256,
            "output_throughput": tput,
            "total_token_throughput": tput * 2,
            "completed_requests": 80,
            "duration_seconds": 25.0,
        },
        "latency": {
            "ttft": {"mean_ms": 140.0, "p99_ms": 160.0},
            "e2el": {"mean_ms": 2500.0, "p99_ms": 2800.0},
        },
    }))
    return workspace


def test_apply_runtime_overrides_pins_benchmark_script_after_gpu_pop():
    bench = {
        "framework": "sglang",
        "benchmark_script": "dsr1_fp8_mi300x.sh",
        "envs": {},
    }
    apply_runtime_benchmark_overrides(
        bench, gpu_type="mi300x", benchmark_script="sglang_mi300x.sh",
    )
    assert bench["benchmark_script"] == "sglang_mi300x.sh"
    assert bench["runner_type"] == "mi300x"


def test_build_variant_yaml_propagates_benchmark_script(tmp_path):
    base = tmp_path / "base.yaml"
    _write_baseline_yaml_overrides(base)
    out = _build_variant_yaml(
        base,
        base_extra_args="",
        variant=GridVariant("vA", "--attention-backend aiter"),
        output_subdir=tmp_path / "vA",
        gpu_type="mi300x",
        benchmark_script="sglang_mi300x.sh",
    )
    cfg = yaml.safe_load(out.read_text())
    assert cfg["benchmark"]["benchmark_script"] == "sglang_mi300x.sh"
    assert cfg["benchmark"]["runner_type"] == "mi300x"


def test_run_magpie_default_result_dir_is_output_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "skip-kill")
    captured: dict = {}

    def fake_run(cmd, *args, **kwargs):
        captured["env"] = dict(kwargs.get("env") or {})
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    with patch(
        "inference_optimizer.orchestrator.action_executors._grid_runner."
        "run_with_session_kill",
        side_effect=fake_run,
    ):
        _run_magpie(
            magpie_python="/opt/venv/bin/python",
            config_path=tmp_path / "config.yaml",
            output_dir=tmp_path / "slot",
            timeout_sec=5,
            cwd=str(tmp_path),
        )
    assert captured["env"]["RESULT_DIR"] == str(tmp_path / "slot")


def test_run_magpie_explicit_result_dir_overrides_default(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "skip-kill")
    captured: dict = {}

    def fake_run(cmd, *args, **kwargs):
        captured["env"] = dict(kwargs.get("env") or {})
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    with patch(
        "inference_optimizer.orchestrator.action_executors._grid_runner."
        "run_with_session_kill",
        side_effect=fake_run,
    ):
        _run_magpie(
            magpie_python="/opt/venv/bin/python",
            config_path=tmp_path / "config.yaml",
            output_dir=tmp_path / "slot",
            timeout_sec=5,
            cwd=str(tmp_path),
            result_dir="/tmp/redirect_leak",
        )
    assert captured["env"]["RESULT_DIR"] == "/tmp/redirect_leak"


@pytest.mark.asyncio
async def test_run_grid_forwards_benchmark_script_per_variant(tmp_path):
    base = tmp_path / "base.yaml"
    _write_baseline_yaml_overrides(base)
    output_root = tmp_path / "out"

    def fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_workspace(slot)
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    grid = [GridVariant("v0"), GridVariant("v1")]
    with patch(
        "inference_optimizer.orchestrator.action_executors._grid_runner."
        "run_with_session_kill",
        side_effect=fake_run,
    ):
        results = await run_grid(
            base_yaml_path=base, base_extra_args="",
            grid=grid, output_root=output_root, variant_timeout_sec=5,
            gpu_type="mi300x",
            benchmark_script="sglang_mi300x.sh",
        )

    assert len(results) == 2
    for i in range(2):
        slot = output_root / f"variant_{i:02d}_v{i}"
        cfg = yaml.safe_load((slot / "config.yaml").read_text())
        assert cfg["benchmark"]["benchmark_script"] == "sglang_mi300x.sh"


@pytest.mark.asyncio
async def test_run_grid_forwards_result_dir_to_subprocess_env(tmp_path):
    base = tmp_path / "base.yaml"
    _write_baseline_yaml_overrides(base)
    output_root = tmp_path / "out"
    captured_envs: list[dict] = []

    def fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        captured_envs.append(dict(kwargs.get("env") or {}))
        _fake_workspace(slot)
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    grid = [GridVariant("v0"), GridVariant("v1")]
    with patch(
        "inference_optimizer.orchestrator.action_executors._grid_runner."
        "run_with_session_kill",
        side_effect=fake_run,
    ):
        await run_grid(
            base_yaml_path=base, base_extra_args="",
            grid=grid, output_root=output_root, variant_timeout_sec=5,
            result_dir="/tmp/redirect",
        )

    assert len(captured_envs) == 2
    for env in captured_envs:
        assert env["RESULT_DIR"] == "/tmp/redirect"


@pytest.mark.asyncio
async def test_run_grid_default_result_dir_is_per_variant_slot(tmp_path):
    base = tmp_path / "base.yaml"
    _write_baseline_yaml_overrides(base)
    output_root = tmp_path / "out"
    captured_envs: list[tuple[str, str]] = []

    def fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        env = dict(kwargs.get("env") or {})
        captured_envs.append((str(slot), env["RESULT_DIR"]))
        _fake_workspace(slot)
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    grid = [GridVariant("vA"), GridVariant("vB")]
    with patch(
        "inference_optimizer.orchestrator.action_executors._grid_runner."
        "run_with_session_kill",
        side_effect=fake_run,
    ):
        await run_grid(
            base_yaml_path=base, base_extra_args="",
            grid=grid, output_root=output_root, variant_timeout_sec=5,
        )

    for slot_path, result_dir in captured_envs:
        assert slot_path == result_dir
    assert len({rd for _, rd in captured_envs}) == 2


# ==============================================================================
# Framework-aware help-text probe (atom + multi-framework cache)
# ==============================================================================


@pytest.fixture(autouse=False)
def _reset_help_cache():
    """Clear the framework-keyed help-text cache before/after each test
    so per-test mocks don't leak across the class."""
    _grid_runner._HELP_TEXT_CACHE.clear()
    yield
    _grid_runner._HELP_TEXT_CACHE.clear()


def test_probe_server_help_text_atom_returns_help_when_importable(
    _reset_help_cache, monkeypatch,
):
    """The atom branch invokes
    ``atom.model_engine.arg_utils:EngineArgs.add_cli_args``. We mock
    subprocess.run to return a synthetic atom-help payload; the probe
    must return it verbatim and cache it for the second call."""
    call_count = {"n": 0}
    synthetic_help = (
        "usage: atom-engine [-h] [--tensor-parallel-size INT] "
        "[--torch-profiler-dir DIR] ..."
    )

    def fake_run(cmd, *args, **kwargs):
        call_count["n"] += 1
        return subprocess.CompletedProcess(cmd, 0, synthetic_help, "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    out = _grid_runner._probe_server_help_text("atom")
    assert "--tensor-parallel-size" in out
    assert "--torch-profiler-dir" in out
    # Second call must hit the cache, not the subprocess.
    out2 = _grid_runner._probe_server_help_text("atom")
    assert out2 == out
    assert call_count["n"] == 1, (
        "_probe_server_help_text must cache atom's result; subprocess "
        f"called {call_count['n']} times"
    )


def test_probe_server_help_text_atom_returns_empty_on_failure(
    _reset_help_cache, monkeypatch,
):
    """Subprocess failures must surface as ``""`` and NOT be cached —
    a transient failure (e.g. unrelated test-time mock) must not
    poison the slot for the rest of the session."""
    raised = {"n": 0}

    def fake_run(*args, **kwargs):
        raised["n"] += 1
        raise RuntimeError("subprocess refused to run")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert _grid_runner._probe_server_help_text("atom") == ""
    # Re-probe — must invoke subprocess again rather than serving an
    # empty cached value (so transient failures recover automatically).
    assert _grid_runner._probe_server_help_text("atom") == ""
    assert raised["n"] == 2


def test_probe_server_help_text_cache_keyed_by_framework(
    _reset_help_cache, monkeypatch,
):
    """Cache slots must be per-framework so a sglang-default test box
    doesn't leak its help text into the vllm or atom slot."""
    payload_map = {
        "sglang": "USAGE_SGLANG --enable-flashinfer-mla",
        "atom": "USAGE_ATOM --torch-profiler-dir",
    }

    def fake_run(cmd, *args, **kwargs):
        # Identify the framework from the inline source code in cmd[-1].
        src = cmd[-1] if cmd else ""
        if "sglang.launch_server" in src:
            payload = payload_map["sglang"]
        elif "atom.model_engine" in src:
            payload = payload_map["atom"]
        else:
            payload = ""
        return subprocess.CompletedProcess(cmd, 0, payload, "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    sgl = _grid_runner._probe_server_help_text("sglang")
    atom = _grid_runner._probe_server_help_text("atom")
    assert "--enable-flashinfer-mla" in sgl
    assert "--torch-profiler-dir" in atom
    # No cross-contamination — atom slot did NOT inherit sglang's text.
    assert "--enable-flashinfer-mla" not in atom
    assert "--torch-profiler-dir" not in sgl


def test_probe_server_help_text_supports_all_three_frameworks(
    _reset_help_cache, monkeypatch,
):
    """Cross-cutting guard: every first-class framework must have a
    registered probe command and the helper must return a ``str`` for
    each (success or failure path, doesn't matter)."""
    monkeypatch.setattr(
        subprocess, "run",
        lambda cmd, *a, **kw: subprocess.CompletedProcess(
            cmd, 0, f"help for {cmd[-1]!r}", "",
        ),
    )
    for fw in ("sglang", "vllm", "atom"):
        out = _grid_runner._probe_server_help_text(fw)
        assert isinstance(out, str)
        assert out, (
            f"_probe_server_help_text({fw!r}) returned an empty string; "
            f"command registration likely missing"
        )


def test_probe_server_help_text_unknown_framework_returns_empty(
    _reset_help_cache,
):
    """Unregistered framework names short-circuit to ``""`` without
    invoking subprocess. Same conservative shape as the existing
    failure path."""
    assert _grid_runner._probe_server_help_text("tensorrt") == ""
    assert _grid_runner._probe_server_help_text("") == ""


def test_probe_sglang_help_text_back_compat_shim(
    _reset_help_cache, monkeypatch,
):
    """The legacy ``_probe_sglang_help_text`` name is preserved as a
    thin wrapper around the framework-keyed probe. Pre-existing test
    fixtures that monkey-patch this exact name must keep working."""
    monkeypatch.setattr(
        subprocess, "run",
        lambda cmd, *a, **kw: subprocess.CompletedProcess(
            cmd, 0, "USAGE_SGLANG_LEGACY", "",
        ),
    )
    out = _grid_runner._probe_sglang_help_text()
    assert "USAGE_SGLANG_LEGACY" in out
    # The shim populates the framework-keyed cache under the sglang key.
    assert "USAGE_SGLANG_LEGACY" in _grid_runner._HELP_TEXT_CACHE.get("sglang", "")


def test_apply_compatibility_filter_uses_atom_help_when_framework_atom(
    _reset_help_cache, monkeypatch,
):
    """When ``$FRAMEWORK=atom`` the compatibility filter must validate
    variant flag literals against the atom --help output, not sglang's. We mock the probe so atom's
    help advertises one flag but not another and confirm the variant
    with the unrecognised flag is dropped with a reason mentioning
    ``atom --help``."""
    monkeypatch.setenv("FRAMEWORK", "atom")
    # MoE keyword so the model-class predicate doesn't drop the variant
    # before the help-text check runs.
    monkeypatch.setenv("MODEL_PATH", "/wekafs/models/DeepSeek-R1-0528")

    # Pre-populate the cache so the test doesn't have to mock subprocess
    # for the underlying probe — the predicate reads from the cache when
    # populated.
    _grid_runner._HELP_TEXT_CACHE["atom"] = (
        "usage: atom-engine [--tensor-parallel-size INT] "
        "[--enable-deepep-moe]"
    )

    # Two variants: one whose flag IS in the atom help (kept), one that
    # references a sglang-only flag (dropped).
    kept_variant = GridVariant(
        name="atom_compatible",
        extra_server_args="--enable-deepep-moe",
    )
    dropped_variant = GridVariant(
        name="sglang_only",
        extra_server_args="--enable-flashinfer-mla",
    )
    kept, dropped = _grid_runner.apply_compatibility_filter(
        [kept_variant, dropped_variant],
    )
    assert [v.name for v in kept] == ["atom_compatible"]
    assert len(dropped) == 1
    assert dropped[0]["name"] == "sglang_only"
    assert "atom --help" in dropped[0]["reason"], (
        f"reason must mention `atom --help` so log readers can tell "
        f"which framework rejected the variant: {dropped[0]['reason']!r}"
    )
