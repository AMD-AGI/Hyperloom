# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Consolidated tests for ``orchestrator.action_executors._grid_runner``."""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import time
from pathlib import Path
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
    annotate_multi_node_cuda_graph_max_bs,
    apply_runtime_benchmark_overrides,
    apply_user_skip_list,
    coerce_extra_envs,
    resolve_skip_spec,
    run_grid,
)


# Section 1: _write_variant_abort_marker (formerly test_grid_runner_abort_marker.py)


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


# Section 2: helper-level units (formerly test_grid_runner_helpers_units.py)


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


class TestMultiNodeCudaGraphMaxBsAdvisory:
    def test_no_notes_in_single_node(self, monkeypatch):
        grid = [
            GridVariant(name="a", extra_server_args="--cuda-graph-max-bs 8"),
        ]
        notes = annotate_multi_node_cuda_graph_max_bs(grid)
        assert notes == []

    def test_emits_advisory_in_multi_node_without_dropping(self, monkeypatch):
        from inference_optimizer.orchestrator.action_executors import (
            _multi_node_env as mne,
        )

        monkeypatch.setattr(mne, "is_multi_node", lambda: True)
        monkeypatch.setenv("CONC", "64")
        grid = [
            GridVariant(name="bad", extra_server_args="--cuda-graph-max-bs 8"),
            GridVariant(name="ok",  extra_server_args="--max-num-seqs 128"),
        ]
        notes = annotate_multi_node_cuda_graph_max_bs(grid)
        assert [n["name"] for n in notes] == ["bad"]
        assert "CONC=64" in notes[0]["reason"]
        # Grid itself is unchanged: nothing is dropped.
        assert [v.name for v in grid] == ["bad", "ok"]


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


class TestCoerceExtraEnvs:
    """Lock the boundary contract for LLM-supplied grid env overrides."""

    def test_dict_is_passthrough(self):
        assert coerce_extra_envs({"FOO": "1", "BAR": "two"}) == {
            "FOO": "1", "BAR": "two",
        }

    def test_dict_coerces_values_to_str(self):
        assert coerce_extra_envs({"USE_AITER": 1, "DEBUG": True}) == {
            "USE_AITER": "1", "DEBUG": "True",
        }

    def test_space_delimited_string(self):
        assert coerce_extra_envs(
            "SGLANG_USE_AITER=1 VLLM_ROCM_USE_AITER_MHA=1"
        ) == {"SGLANG_USE_AITER": "1", "VLLM_ROCM_USE_AITER_MHA": "1"}

    def test_newline_delimited_string(self):
        assert coerce_extra_envs(
            "FOO=1\nBAR=two\n\nBAZ=3"
        ) == {"FOO": "1", "BAR": "two", "BAZ": "3"}

    def test_semicolon_delimited_string(self):
        assert coerce_extra_envs("A=1;B=2;C=3") == {
            "A": "1", "B": "2", "C": "3",
        }

    def test_string_preserves_url_in_value(self):
        assert coerce_extra_envs(
            "HF_ENDPOINT=https://hf.example.com/api"
        ) == {"HF_ENDPOINT": "https://hf.example.com/api"}

    def test_string_drops_tokens_without_equals(self):
        assert coerce_extra_envs("FOO=1 garbage BAR=2") == {
            "FOO": "1", "BAR": "2",
        }

    def test_list_of_kv_tokens(self):
        assert coerce_extra_envs(["FOO=1", "BAR=2"]) == {
            "FOO": "1", "BAR": "2",
        }

    def test_list_of_dicts(self):
        assert coerce_extra_envs([{"FOO": "1"}, {"BAR": "2"}]) == {
            "FOO": "1", "BAR": "2",
        }

    def test_list_later_entries_win(self):
        assert coerce_extra_envs([{"FOO": "first"}, {"FOO": "last"}]) == {
            "FOO": "last",
        }

    def test_none_returns_empty(self):
        assert coerce_extra_envs(None) == {}

    def test_unknown_type_returns_empty(self):
        assert coerce_extra_envs(42) == {}
        assert coerce_extra_envs(object()) == {}

    def test_used_by_backends_grid_override(self):
        llm_entry = {
            "name": "aiter_mla_off",
            "extra_sglang_args": "--attention-backend aiter --disable-mla",
            "extra_envs": "SGLANG_USE_AITER=1 VLLM_ROCM_USE_AITER_MHA=0",
            "note": "aiter attn without MLA fast path",
        }
        v = GridVariant(
            name=llm_entry["name"],
            extra_sglang_args=llm_entry["extra_sglang_args"],
            extra_envs=coerce_extra_envs(llm_entry["extra_envs"]),
            note=llm_entry["note"],
        )

        assert dict(v.extra_envs.items()) == {
            "SGLANG_USE_AITER": "1",
            "VLLM_ROCM_USE_AITER_MHA": "0",
        }
        assert isinstance(v.fingerprint, str) and len(v.fingerprint) > 0


# Section 3: per-variant mtime gating + param overrides
# The autouse fixture below pins INFERENCE_OPTIMIZER_LEAK_ROOTS to an empty sandbox so the harvest doesn't scrape the host's /workspace.


@pytest.fixture(autouse=True)
def _isolate_leak_root(request, tmp_path_factory, monkeypatch):
    """Pin ``INFERENCE_OPTIMIZER_LEAK_ROOTS`` to an empty sandbox for the grid-runner subprocess tests."""
    sandbox = tmp_path_factory.mktemp("isolated_leak_root")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_LEAK_ROOTS", str(sandbox))


# mtime gating subsection helpers


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


# param overrides subsection helpers


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


def test_apply_runtime_overrides_yaml_tp_wins_over_env_on_resume(monkeypatch):
    """Regression (2026-06-02 conc_sweep bug): a stale ``state.tp`` re-exported as ``os.environ['TP']`` on resume must NOT downgrade a YAML-pinned TP."""
    monkeypatch.setenv("TP", "1")
    bench = {
        "framework": "sglang",
        "envs": {"TP": 2, "CONC": 64, "ISL": 1024, "OSL": 1024},
    }
    apply_runtime_benchmark_overrides(bench, gpu_type="mi355x")
    assert bench["envs"]["TP"] == 2, (
        f"yaml-pinned TP=2 must win over os.environ['TP']=1; "
        f"got TP={bench['envs']['TP']}"
    )
    # Other env keys still flow through normally (no yaml-wins for them).
    monkeypatch.setenv("CONC", "128")
    apply_runtime_benchmark_overrides(bench, gpu_type="mi355x")
    assert bench["envs"]["CONC"] == 128
    assert bench["envs"]["TP"] == 2


def test_apply_runtime_overrides_env_tp_used_when_yaml_silent(monkeypatch):
    """Companion: when yaml has no TP, env TP is still applied (guards against an over-broad fix)."""
    monkeypatch.setenv("TP", "4")
    bench = {"framework": "sglang", "envs": {}}
    apply_runtime_benchmark_overrides(bench, gpu_type="mi355x")
    assert bench["envs"]["TP"] == 4


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


# Framework-aware help-text probe (atom + multi-framework cache)


@pytest.fixture(autouse=False)
def _reset_help_cache():
    """Clear the framework-keyed help-text cache before/after each test."""
    _grid_runner._HELP_TEXT_CACHE.clear()
    yield
    _grid_runner._HELP_TEXT_CACHE.clear()


def test_probe_server_help_text_atom_returns_help_when_importable(
    _reset_help_cache, monkeypatch,
):
    """The atom probe returns the mocked help verbatim and caches it for the second call."""
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
    """Subprocess failures surface as ``""`` and are NOT cached (transient failures must not poison the slot)."""
    raised = {"n": 0}

    def fake_run(*args, **kwargs):
        raised["n"] += 1
        raise RuntimeError("subprocess refused to run")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert _grid_runner._probe_server_help_text("atom") == ""
    # Re-probe invokes subprocess again rather than serving an empty cached value.
    assert _grid_runner._probe_server_help_text("atom") == ""
    assert raised["n"] == 2


def test_probe_server_help_text_cache_keyed_by_framework(
    _reset_help_cache, monkeypatch,
):
    """Cache slots must be per-framework so sglang's help text doesn't leak into the vllm/atom slot."""
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
    """Cross-cutting guard: every first-class framework has a registered probe command and returns a ``str``."""
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
    """Unregistered framework names short-circuit to ``""`` without invoking subprocess."""
    assert _grid_runner._probe_server_help_text("tensorrt") == ""
    assert _grid_runner._probe_server_help_text("") == ""


def test_probe_sglang_help_text_back_compat_shim(
    _reset_help_cache, monkeypatch,
):
    """The legacy ``_probe_sglang_help_text`` name is preserved as a thin wrapper so fixtures patching it keep working."""
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
    """When ``$FRAMEWORK=atom`` the compatibility filter validates variant flags against atom --help, dropping a sglang-only flag with a reason mentioning ``atom --help``."""
    monkeypatch.setenv("FRAMEWORK", "atom")
    # MoE keyword so the model-class predicate doesn't drop the variant first.
    monkeypatch.setenv("MODEL_PATH", "/wekafs/models/DeepSeek-R1-0528")

    # Pre-populate the cache so the predicate reads from it without mocking subprocess.
    _grid_runner._HELP_TEXT_CACHE["atom"] = (
        "usage: atom-engine [--tensor-parallel-size INT] "
        "[--enable-deepep-moe]"
    )

    # One variant's flag IS in the atom help (kept); one references a sglang-only flag (dropped).
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


# Section: dedup_vllm_server_args (#520) — vLLM/atom single-value flag collapse


class TestDedupVllmServerArgs:
    """``dedup_vllm_server_args`` collapses repeated vLLM single-value flags."""

    def test_duplicate_attention_backend_keeps_last(self):
        # The exact #520 repro: YAML base + variant both inject the flag.
        out = _grid_runner.dedup_vllm_server_args(
            "--attention-backend ROCM_AITER_FA --attention-backend ROCM_FLASH",
            "vllm",
        )
        assert out == "--attention-backend ROCM_FLASH"
        assert out.count("--attention-backend") == 1

    def test_identical_duplicate_collapses_to_single(self):
        out = _grid_runner.dedup_vllm_server_args(
            "--attention-backend ROCM_AITER_FA --attention-backend ROCM_AITER_FA",
            "vllm",
        )
        assert out == "--attention-backend ROCM_AITER_FA"

    def test_preserves_surrounding_and_unknown_flags_in_order(self):
        out = _grid_runner.dedup_vllm_server_args(
            "--enforce-eager --attention-backend A --max-model-len 4096 "
            "--attention-backend B --trust-remote-code",
            "vllm",
        )
        # Earlier --attention-backend span removed; everything else stays ordered.
        assert out == (
            "--enforce-eager --max-model-len 4096 "
            "--attention-backend B --trust-remote-code"
        )

    def test_equals_form_is_deduped(self):
        out = _grid_runner.dedup_vllm_server_args(
            "--gpu-memory-utilization=0.9 --gpu-memory-utilization=0.85",
            "vllm",
        )
        assert out == "--gpu-memory-utilization=0.85"

    def test_atom_framework_also_deduped(self):
        out = _grid_runner.dedup_vllm_server_args(
            "--attention-backend A --attention-backend B", "atom",
        )
        assert out == "--attention-backend B"

    def test_sglang_is_noop_repeats_preserved(self):
        # sglang tolerates repeats (last-wins at the server); do not mangle.
        raw = "--attention-backend aiter --attention-backend triton"
        assert _grid_runner.dedup_vllm_server_args(raw, "sglang") == raw

    def test_no_duplicates_returns_unchanged(self):
        raw = "--attention-backend ROCM_AITER_FA --max-model-len 8192"
        assert _grid_runner.dedup_vllm_server_args(raw, "vllm") == raw

    def test_empty_and_none_safe(self):
        assert _grid_runner.dedup_vllm_server_args("", "vllm") == ""
        assert _grid_runner.dedup_vllm_server_args(None, "vllm") == ""

    def test_unparseable_string_returned_as_is(self):
        # Unbalanced quote: leave it for vLLM to report rather than mangling.
        raw = "--attention-backend 'unterminated"
        assert _grid_runner.dedup_vllm_server_args(raw, "vllm") == raw.strip()

    def test_distinct_single_value_flags_untouched(self):
        raw = "--max-num-seqs 256 --block-size 16 --kv-cache-dtype fp8"
        assert _grid_runner.dedup_vllm_server_args(raw, "vllm") == raw
