# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Consolidated tests for ``orchestrator.actions.executors._grid_runner``."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

from hyperloom.orchestrator.actions.executors import _grid_runner
from hyperloom.orchestrator.actions.executors import _grid_runner as gr
from hyperloom.orchestrator.actions.executors import _grid_variant_filter
from hyperloom.orchestrator.actions.executors._subprocess_kill import (
    ORCHESTRATOR_CANCELLED_RETURNCODE,
    SESSION_TIME_EXHAUSTED_RETURNCODE,
)
from hyperloom.orchestrator.actions.executors._grid_runner import (
    _MN_BACKENDS_PRIORITY,
    _MN_PARAMS_PRIORITY,
    SESSION_TIME_EXHAUSTED_CLASS,
    GridVariant,
    VariantResult,
    _build_variant_yaml,
    _parse_skip_spec,
    _run_magpie,
    _SESSION_KILL_GRACE_SEC,
    apply_runtime_benchmark_overrides,
    apply_user_skip_list,
    coerce_extra_envs,
    reorder_grid_for_multi_node,
    resolve_skip_spec,
    run_grid,
)

from .conftest import enable_multi_node, launches_by_round_slot


# Section 1: _write_variant_abort_marker


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
            "server /health did not return 200 within 1800s (url=http://192.0.2.67:8888/health, last_err=...)"
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
    assert any("failed to write abort_reason.json" in r.message for r in caplog.records)


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
        "magpie_nonzero_after_valid_measurement",
        "killed_overtime",
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
        name="ok",
        extra_server_args="",
        extra_envs={},
        status="succeeded",
    )
    assert vr_ok.to_dict()["error_class"] == ""


# Section 2: helper-level units


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


class TestReorderGridForMultiNode:
    """reorder is wired into explore/sweep; single-node MUST be a no-op."""

    def _grid(self):
        # Ordered so a real reorder would change it: low-priority backend first,
        # high-priority param last, untagged in the middle.
        return [
            GridVariant(name="tier5_comm_custom_ar", note="tier5_comm"),
            GridVariant(name="untagged_misc"),
            GridVariant(name="cuda_graph_max_bs_64", note="cuda_graph_max_bs"),
        ]

    def test_single_node_preserves_order_bit_for_bit(self, monkeypatch):
        # Single-node grid order is never altered.
        from hyperloom.orchestrator.actions.executors import (
            _multi_node_env as mne,
        )

        monkeypatch.setattr(mne, "is_multi_node", lambda: False)
        grid = self._grid()
        out = reorder_grid_for_multi_node(
            grid,
            priority_tags=_MN_PARAMS_PRIORITY + _MN_BACKENDS_PRIORITY,
        )
        assert [v.name for v in out] == [v.name for v in grid]

    def test_multi_node_surfaces_likely_winners_first(self, monkeypatch):
        from hyperloom.orchestrator.actions.executors import (
            _multi_node_env as mne,
        )

        monkeypatch.setattr(mne, "is_multi_node", lambda: True)
        grid = self._grid()
        out = reorder_grid_for_multi_node(
            grid,
            priority_tags=_MN_PARAMS_PRIORITY + _MN_BACKENDS_PRIORITY,
        )
        # cuda_graph_max_bs (params tier-1) sorts ahead of tier5_comm; the
        # untagged variant sinks to the end (stable sort).
        assert [v.name for v in out] == [
            "cuda_graph_max_bs_64",
            "tier5_comm_custom_ar",
            "untagged_misc",
        ]


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
            "cuda_graph_max_bs_8",
            "cuda_graph_max_bs_32",
        }


class TestVariantResultToDict:
    def test_succeeded_default_shape(self):
        vr = VariantResult(
            name="v",
            extra_server_args="--foo 1",
            extra_envs={"A": "1"},
            status="succeeded",
        )
        out = vr.to_dict()
        assert out["status"] == "succeeded"
        assert out["error_class"] == ""

    def test_failed_round_trip_carries_error_class(self):
        vr = VariantResult(
            name="v",
            extra_server_args="",
            extra_envs={},
            status="failed",
            error="boom",
            error_class="benchmark_report_missing",
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
            "FOO": "1",
            "BAR": "two",
        }

    def test_dict_coerces_values_to_str(self):
        assert coerce_extra_envs({"USE_AITER": 1, "DEBUG": True}) == {
            "USE_AITER": "1",
            "DEBUG": "True",
        }

    def test_space_delimited_string(self):
        assert coerce_extra_envs("SGLANG_USE_AITER=1 VLLM_ROCM_USE_AITER_MHA=1") == {
            "SGLANG_USE_AITER": "1",
            "VLLM_ROCM_USE_AITER_MHA": "1",
        }

    def test_newline_delimited_string(self):
        assert coerce_extra_envs("FOO=1\nBAR=two\n\nBAZ=3") == {"FOO": "1", "BAR": "two", "BAZ": "3"}

    def test_semicolon_delimited_string(self):
        assert coerce_extra_envs("A=1;B=2;C=3") == {
            "A": "1",
            "B": "2",
            "C": "3",
        }

    def test_string_preserves_url_in_value(self):
        assert coerce_extra_envs("HF_ENDPOINT=https://hf.example.com/api") == {
            "HF_ENDPOINT": "https://hf.example.com/api"
        }

    def test_string_drops_tokens_without_equals(self):
        assert coerce_extra_envs("FOO=1 garbage BAR=2") == {
            "FOO": "1",
            "BAR": "2",
        }

    def test_list_of_kv_tokens(self):
        assert coerce_extra_envs(["FOO=1", "BAR=2"]) == {
            "FOO": "1",
            "BAR": "2",
        }

    def test_list_of_dicts(self):
        assert coerce_extra_envs([{"FOO": "1"}, {"BAR": "2"}]) == {
            "FOO": "1",
            "BAR": "2",
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
            "extra_server_args": "--attention-backend aiter --disable-mla",
            "extra_envs": "SGLANG_USE_AITER=1 VLLM_ROCM_USE_AITER_MHA=0",
            "note": "aiter attn without MLA fast path",
        }
        v = GridVariant(
            name=llm_entry["name"],
            extra_server_args=llm_entry["extra_server_args"],
            extra_envs=coerce_extra_envs(llm_entry["extra_envs"]),
            note=llm_entry["note"],
        )

        assert dict(v.extra_envs.items()) == {
            "SGLANG_USE_AITER": "1",
            "VLLM_ROCM_USE_AITER_MHA": "0",
        }
        assert isinstance(v.fingerprint, str) and len(v.fingerprint) > 0

    def test_drops_hijacking_envs_but_keeps_workload_pins(self):
        v = GridVariant(
            name="mixed",
            extra_envs={
                "SGLANG_USE_AITER": "1",
                "CONC": "64",
                "ISL": "1024",
                "RUN_EVAL": "false",
                "LD_PRELOAD": "/tmp/evil.so",
                "PATH": "/tmp/bin",
                "PYTHONPATH": "/tmp/evil",
                "OPENAI_API_KEY": "must-not-reach-benchmark",
            },
        )

        assert v.extra_envs == {
            "SGLANG_USE_AITER": "1",
            "CONC": "64",
            "ISL": "1024",
            "RUN_EVAL": "false",
        }


# Section 3: per-variant mtime gating + param overrides


@pytest.fixture(autouse=True)
def _isolate_leak_root(request, tmp_path_factory, monkeypatch):
    """Pin ``INFERENCE_OPTIMIZER_LEAK_ROOTS`` to an empty sandbox for the grid-runner subprocess tests."""
    sandbox = tmp_path_factory.mktemp("isolated_leak_root")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_LEAK_ROOTS", str(sandbox))


def _write_baseline_yaml_mtime(path: Path) -> None:
    cfg = {
        "benchmark": {
            "framework": "sglang",
            "model": "/path/models/Qwen-Qwen3-8B",
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
    (ws / "benchmark_report.json").write_text(
        json.dumps(
            {
                "success": False,
                "framework": "sglang",
                "model": "/path/models/Qwen-Qwen3-8B",
            }
        )
    )
    return ws


def _write_leak(path: Path, *, tput: float = 1761.6, completed: int = 640) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "output_throughput": tput,
                "request_throughput": tput / 10,
                "completed_requests": completed,
                "duration_seconds": 120.0,
            }
        )
    )


@pytest.mark.asyncio
async def test_run_grid_rejects_stale_leak_from_previous_run(
    tmp_path,
    monkeypatch,
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
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=fake_run,
    ):
        results = await run_grid(
            base_yaml_path=base,
            base_extra_args="",
            grid=[GridVariant("vA")],
            output_root=output_root,
            variant_timeout_sec=5,
        )

    assert len(results) == 1
    r = results[0]
    assert r.status == "failed"
    assert r.output_throughput is None
    assert all("rescued_from_leaked_path" not in (w or "") for w in r.nonfatal_warnings)


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
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=fake_run,
    ):
        results = await run_grid(
            base_yaml_path=base,
            base_extra_args="",
            grid=[GridVariant("vA")],
            output_root=output_root,
            variant_timeout_sec=5,
        )

    assert len(results) == 1
    r = results[0]
    assert r.status == "succeeded"
    assert r.output_throughput == pytest.approx(1234.0)
    assert any((w or "").startswith("rescued_from_leaked_path:") for w in r.nonfatal_warnings)


def _write_baseline_yaml_overrides(path: Path) -> None:
    cfg = {
        "benchmark": {
            "framework": "sglang",
            "model": "/path/models/Qwen-Qwen3-8B",
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
    (workspace / "benchmark_report.json").write_text(
        json.dumps(
            {
                "success": True,
                "framework": "sglang",
                "model": "/path/models/Qwen-Qwen3-8B",
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
            }
        )
    )
    return workspace


def test_apply_runtime_overrides_pins_benchmark_script_after_gpu_pop():
    bench = {
        "framework": "sglang",
        "benchmark_script": "dsr1_fp8_mi300x.sh",
        "envs": {},
    }
    apply_runtime_benchmark_overrides(
        bench,
        gpu_type="mi300x",
        benchmark_script="sglang_mi300x.sh",
    )
    assert bench["benchmark_script"] == "sglang_mi300x.sh"
    assert bench["runner_type"] == "mi300x"


def test_apply_runtime_overrides_yaml_tp_wins_over_env_on_resume(monkeypatch):
    """A stale ``state.tp`` re-exported as ``os.environ['TP']`` on resume must not downgrade a YAML-pinned TP."""
    monkeypatch.setenv("TP", "1")
    bench = {
        "framework": "sglang",
        "envs": {"TP": 2, "CONC": 64, "ISL": 1024, "OSL": 1024},
    }
    apply_runtime_benchmark_overrides(bench, gpu_type="mi355x")
    assert bench["envs"]["TP"] == 2, f"yaml-pinned TP=2 must win over os.environ['TP']=1; got TP={bench['envs']['TP']}"
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


def test_build_variant_yaml_can_remove_base_args_and_unset_envs(tmp_path):
    base = tmp_path / "base.yaml"
    _write_baseline_yaml_overrides(base)
    cfg = yaml.safe_load(base.read_text())
    cfg["benchmark"]["envs"]["EXTRA_SGLANG_ARGS"] = (
        "--enable-prefix-caching --max-num-seqs 512 --attention-backend aiter"
    )
    cfg["benchmark"]["envs"]["SGLANG_ENABLE_FOO"] = "1"
    base.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    out = _build_variant_yaml(
        base,
        base_extra_args="--enable-bar --block-size 128",
        variant=GridVariant(
            "without_user_foo",
            "--max-num-seqs 256",
            {"SGLANG_KEEP": "1"},
            remove_args=["--enable-prefix-caching", "--attention-backend"],
            unset_envs=["SGLANG_ENABLE_FOO"],
        ),
        output_subdir=tmp_path / "without_user_foo",
    )

    envs = yaml.safe_load(out.read_text())["benchmark"]["envs"]
    args = envs["EXTRA_SGLANG_ARGS"]
    assert "--enable-prefix-caching" not in args
    assert "--attention-backend" not in args
    assert "aiter" not in args
    assert "--enable-bar" in args
    assert "--block-size 128" in args
    assert "--max-num-seqs 256" in args
    assert envs["SGLANG_KEEP"] == "1"
    assert "SGLANG_ENABLE_FOO" not in envs


def test_build_variant_yaml_refuses_to_unset_pinned_envs(tmp_path):
    """A variant that unsets TP would silently shrink the Ray lease to one GPU."""
    base = tmp_path / "base.yaml"
    _write_baseline_yaml_overrides(base)
    cfg = yaml.safe_load(base.read_text())
    cfg["benchmark"]["envs"].update({"TP": 8, "CONC": 64, "SGLANG_ENABLE_FOO": "1"})
    base.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    out = _build_variant_yaml(
        base,
        base_extra_args="",
        variant=GridVariant(
            "unset_the_world",
            unset_envs=["TP", "CONC", "ROCR_VISIBLE_DEVICES", "SGLANG_ENABLE_FOO"],
        ),
        output_subdir=tmp_path / "unset_the_world",
    )

    envs = yaml.safe_load(out.read_text())["benchmark"]["envs"]
    assert envs["TP"] == 8
    assert envs["CONC"] == 64
    # A plain tuning knob is still removable; only the pins are protected.
    assert "SGLANG_ENABLE_FOO" not in envs


def test_run_magpie_default_result_dir_is_output_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "skip-kill")
    captured: dict = {}

    def fake_run(cmd, *args, **kwargs):
        captured["env"] = dict(kwargs.get("env") or {})
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
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


def test_run_magpie_does_not_forward_llm_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "skip-kill")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-benchmark")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-reach-benchmark")
    captured: dict = {}

    def fake_run(cmd, *args, **kwargs):
        captured["env"] = dict(kwargs.get("env") or {})
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=fake_run,
    ):
        _run_magpie(
            magpie_python="/opt/venv/bin/python",
            config_path=tmp_path / "config.yaml",
            output_dir=tmp_path / "slot",
            timeout_sec=5,
            cwd=str(tmp_path),
        )

    assert "OPENAI_API_KEY" not in captured["env"]
    assert "ANTHROPIC_API_KEY" not in captured["env"]


def test_run_magpie_explicit_result_dir_overrides_default(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "skip-kill")
    captured: dict = {}

    def fake_run(cmd, *args, **kwargs):
        captured["env"] = dict(kwargs.get("env") or {})
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
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
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=fake_run,
    ):
        results = await run_grid(
            base_yaml_path=base,
            base_extra_args="",
            grid=grid,
            output_root=output_root,
            variant_timeout_sec=5,
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
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=fake_run,
    ):
        await run_grid(
            base_yaml_path=base,
            base_extra_args="",
            grid=grid,
            output_root=output_root,
            variant_timeout_sec=5,
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
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=fake_run,
    ):
        await run_grid(
            base_yaml_path=base,
            base_extra_args="",
            grid=grid,
            output_root=output_root,
            variant_timeout_sec=5,
        )

    for slot_path, result_dir in captured_envs:
        assert slot_path == result_dir
    assert len({rd for _, rd in captured_envs}) == 2


@pytest.mark.asyncio
async def test_run_grid_benchmark_runs_inside_the_session_that_owns_it(tmp_path):
    """Every grid pass runs from the task workspace, so its children are ours.

    A load generator is what tells the robustness reactor that a refused port is
    a server that died mid-benchmark rather than the idle gap between two
    variants, and it is only believed when it can be tied to this session —
    otherwise a co-tenant's client on a shared node vouches for a port it never
    touched. The tie is the working directory the whole benchmark subtree
    inherits, which the baseline arm already anchors to its own output dir. A
    grid variant launched from the system temp directory carries no anchor at
    all, so the outage it is running through reads as an idle stretch.
    """
    from hyperloom.agents.robustness.role.prompt_inputs import (
        ReactorContext,
        SharedStateSnapshot,
    )
    from hyperloom.agents.robustness.signals.local_health import (
        LocalHealthConfig,
        evaluate_local_health_signals,
    )
    from hyperloom.agents.robustness.sources.base import SourceData

    base = tmp_path / "base.yaml"
    _write_baseline_yaml_overrides(base)
    session_dir = tmp_path / "session"
    output_root = session_dir / "runs" / "explore" / "task-1"
    captured_cwds: list[str] = []

    def fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        captured_cwds.append(str(kwargs.get("cwd") or ""))
        _fake_workspace(slot)
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=fake_run,
    ):
        await run_grid(
            base_yaml_path=base,
            base_extra_args="",
            grid=[GridVariant("vA"), GridVariant("vB")],
            output_root=output_root,
            variant_timeout_sec=5,
        )

    assert captured_cwds
    for cwd in captured_cwds:
        assert Path(cwd).is_relative_to(session_dir), f"benchmark cwd {cwd} is outside the session {session_dir}"

    # The reactor's own reading of that cwd: a client inheriting it is ours even
    # when nothing else on its command line names the session.
    data = SourceData(
        local_processes=[
            {"pid": 8, "rss_mb": 96.0, "cmd": "python benchmark_serving.py --port 30000", "cwd": captured_cwds[0]},
        ],
        local_server_health=[
            {"url": "http://localhost:30000/health", "reachable": False, "status": "error", "error": "connect"},
        ],
    )
    ctx = ReactorContext(
        tick_index=0,
        # The session identity the rule matches on comes from LocalHealthConfig
        # below; the snapshot no longer carries a second copy of it.
        shared_state=SharedStateSnapshot(),
        inbox=[],
        now_unix=1.0,
    )
    matched = [
        s
        for s in evaluate_local_health_signals(ctx, data, config=LocalHealthConfig(session_dir=session_dir))
        if s.name == "local_server_unreachable"
    ]
    assert matched and matched[0].evidence["benchmark_client_seen"] is True


@pytest.mark.asyncio
async def test_run_grid_multi_node_removal_matches_materialized_yaml(tmp_path, monkeypatch):
    base = tmp_path / "base.yaml"
    _write_baseline_yaml_overrides(base)
    cfg = yaml.safe_load(base.read_text())
    cfg["benchmark"]["envs"]["EXTRA_SGLANG_ARGS"] = "--bad-base 1 --keep-base 2"
    cfg["benchmark"]["envs"]["SGLANG_REMOVE_ME"] = "1"
    base.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    from hyperloom.orchestrator.actions.executors import _multi_node_env as mne

    monkeypatch.setattr(mne, "is_multi_node", lambda: True)
    captured_restart: dict = {}

    async def fake_restart_server_for_round(**kwargs):
        captured_restart.update(kwargs)

    def fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_workspace(slot)
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    monkeypatch.setattr(
        "hyperloom.orchestrator.actions.executors._multi_node_server_lifecycle.restart_server_for_round",
        fake_restart_server_for_round,
    )
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=fake_run,
    ):
        await run_grid(
            base_yaml_path=base,
            base_extra_args="--base-extra 3",
            grid=[
                GridVariant(
                    "remove_inherited",
                    "--variant 4",
                    {"SGLANG_KEEP_ME": "1"},
                    remove_args=["--bad-base"],
                    unset_envs=["SGLANG_REMOVE_ME"],
                )
            ],
            output_root=tmp_path / "out",
            variant_timeout_sec=5,
        )

    args = captured_restart["extra_server_args"]
    assert "--bad-base" not in args
    assert "1" not in args.split()
    assert "--keep-base 2" in args
    assert "--base-extra 3" in args
    assert "--variant 4" in args
    assert captured_restart["unset_env"] == ["SGLANG_REMOVE_ME"]
    assert captured_restart["extra_env"] == {"SGLANG_KEEP_ME": "1"}


# Framework-aware help-text probe (atom + multi-framework cache)


@pytest.fixture(autouse=False)
def _reset_help_cache():
    """Clear the framework-keyed help-text caches before/after each test."""
    _grid_runner._HELP_TEXT_CACHE.clear()
    _grid_variant_filter._HELP_PROBE_FAILED_UNTIL.clear()
    yield
    _grid_runner._HELP_TEXT_CACHE.clear()
    _grid_variant_filter._HELP_PROBE_FAILED_UNTIL.clear()


def test_probe_server_help_text_atom_returns_help_when_importable(
    _reset_help_cache,
    monkeypatch,
):
    """The atom probe returns the mocked help verbatim and caches it for the second call."""
    call_count = {"n": 0}
    synthetic_help = "usage: atom-engine [-h] [--tensor-parallel-size INT] [--torch-profiler-dir DIR] ..."

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
        f"_probe_server_help_text must cache atom's result; subprocess called {call_count['n']} times"
    )


def test_probe_server_help_text_atom_returns_empty_on_failure(
    _reset_help_cache,
    monkeypatch,
):
    """A failure surfaces as ``""`` and is held off rather than re-paid at once.

    Re-probing on every variant costs a ten-second import each time on a box
    that does not have the framework; the hold-off expires so a box that gains
    it is picked back up.
    """
    raised = {"n": 0}

    def fake_run(*args, **kwargs):
        raised["n"] += 1
        raise RuntimeError("subprocess refused to run")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert _grid_runner._probe_server_help_text("atom") == ""
    assert _grid_runner._probe_server_help_text("atom") == ""
    assert raised["n"] == 1

    # The hold-off is bounded, so a framework that recovers is picked back up.
    _grid_variant_filter._HELP_PROBE_FAILED_UNTIL["atom"] = 0.0
    assert _grid_runner._probe_server_help_text("atom") == ""
    assert raised["n"] == 2


def test_probe_server_help_text_ignores_a_failed_runs_stderr(
    _reset_help_cache,
    monkeypatch,
):
    """A traceback is not help text.

    Treating it as one makes every gated flag look absent from the help, which
    drops the variants carrying them instead of sparing them.
    """
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, *a, **kw: subprocess.CompletedProcess(cmd, 1, "", "Traceback ... ImportError"),
    )
    assert _grid_runner._probe_server_help_text("atom") == ""


def test_probe_server_help_text_cache_keyed_by_framework(
    _reset_help_cache,
    monkeypatch,
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
    _reset_help_cache,
    monkeypatch,
):
    """Cross-cutting guard: every first-class framework has a registered probe command and returns a ``str``."""
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, *a, **kw: subprocess.CompletedProcess(
            cmd,
            0,
            f"help for {cmd[-1]!r}",
            "",
        ),
    )
    for fw in ("sglang", "vllm", "atom"):
        out = _grid_runner._probe_server_help_text(fw)
        assert isinstance(out, str)
        assert out, f"_probe_server_help_text({fw!r}) returned an empty string; command registration likely missing"


def test_probe_server_help_text_unknown_framework_returns_empty(
    _reset_help_cache,
):
    """Unregistered framework names short-circuit to ``""`` without invoking subprocess."""
    assert _grid_runner._probe_server_help_text("tensorrt") == ""
    assert _grid_runner._probe_server_help_text("") == ""


def test_probe_server_help_text_sglang(
    _reset_help_cache,
    monkeypatch,
):
    """The framework-keyed probe handles sglang and populates the sglang cache."""
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, *a, **kw: subprocess.CompletedProcess(
            cmd,
            0,
            "USAGE_SGLANG_LEGACY",
            "",
        ),
    )
    out = _grid_runner._probe_server_help_text("sglang")
    assert "USAGE_SGLANG_LEGACY" in out
    assert "USAGE_SGLANG_LEGACY" in _grid_runner._HELP_TEXT_CACHE.get("sglang", "")


def test_apply_compatibility_filter_uses_atom_help_when_framework_atom(
    _reset_help_cache,
    monkeypatch,
):
    """When ``$FRAMEWORK=atom`` the compatibility filter validates variant flags against atom --help, dropping a sglang-only flag with a reason mentioning ``atom --help``."""
    monkeypatch.setenv("FRAMEWORK", "atom")
    # MoE keyword so the model-class predicate doesn't drop the variant first.
    monkeypatch.setenv("MODEL_PATH", "/path/models/DeepSeek-R1-0528")

    # Pre-populate the cache so the predicate reads from it without mocking subprocess.
    _grid_runner._HELP_TEXT_CACHE["atom"] = "usage: atom-engine [--tensor-parallel-size INT] [--enable-deepep-moe]"

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
        framework="atom",
        model_path="",
    )
    assert [v.name for v in kept] == ["atom_compatible"]
    assert len(dropped) == 1
    assert dropped[0]["name"] == "sglang_only"
    assert "atom --help" in dropped[0]["reason"], (
        f"reason must mention `atom --help` so log readers can tell "
        f"which framework rejected the variant: {dropped[0]['reason']!r}"
    )


# Section: dedup_vllm_server_args  — vLLM/atom single-value flag collapse


class TestDedupVllmServerArgs:
    """``dedup_vllm_server_args`` collapses repeated vLLM single-value flags."""

    def test_duplicate_attention_backend_keeps_last(self):
        # YAML base + variant both inject the flag.
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
            "--enforce-eager --attention-backend A --max-model-len 4096 --attention-backend B --trust-remote-code",
            "vllm",
        )
        # Earlier --attention-backend span removed; everything else stays ordered.
        assert out == ("--enforce-eager --max-model-len 4096 --attention-backend B --trust-remote-code")

    def test_json_config_flag_quotes_survive_dedup(self):
        # Regression: a variant that duplicates a single-value flag (here
        # --block-size) used to force the shlex-split/rejoin branch, which
        # STRIPPED the inner double quotes of a compact --compilation-config
        # JSON value (``{"cudagraph_mode":"PIECEWISE"}`` -> ``{cudagraph_mode:
        # PIECEWISE}``) and crashed every explore/kernel/integrate variant
        # server with ``Invalid JSON``. JSON-aware tokenization must preserve
        # that value while still collapsing unrelated duplicate flags.
        raw = (
            '--compilation-config {"cudagraph_mode":"PIECEWISE"} '
            "--block-size 128 --block-size 128 --gpu-memory-utilization 0.95"
        )
        out = _grid_runner.dedup_vllm_server_args(raw, "vllm")
        assert '{"cudagraph_mode":"PIECEWISE"}' in out
        assert "{cudagraph_mode:PIECEWISE}" not in out
        assert out.count("--block-size") == 1

    def test_speculative_config_quotes_survive_dedup(self):
        raw = '--speculative-config {"method":"eagle"} --max-num-seqs 256 --max-num-seqs 256'
        out = _grid_runner.dedup_vllm_server_args(raw, "vllm")
        assert '{"method":"eagle"}' in out
        assert out.count("--max-num-seqs") == 1

    def test_equals_form_is_deduped(self):
        out = _grid_runner.dedup_vllm_server_args(
            "--gpu-memory-utilization=0.9 --gpu-memory-utilization=0.85",
            "vllm",
        )
        assert out == "--gpu-memory-utilization=0.85"

    def test_atom_framework_also_deduped(self):
        out = _grid_runner.dedup_vllm_server_args(
            "--attention-backend A --attention-backend B",
            "atom",
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

    def test_mixed_equals_and_space_forms_dedup_last_wins(self):
        # `--flag value` (YAML base) then `--flag=value` (variant): last wins.
        out = _grid_runner.dedup_vllm_server_args(
            "--attention-backend ROCM_AITER_FA --attention-backend=ROCM_FLASH",
            "vllm",
        )
        assert out == "--attention-backend=ROCM_FLASH"

    def test_mixed_equals_then_space_form_dedup_last_wins(self):
        # Reverse order: `--flag=value` then `--flag value`.
        out = _grid_runner.dedup_vllm_server_args(
            "--attention-backend=ROCM_AITER_FA --attention-backend ROCM_FLASH",
            "vllm",
        )
        assert out == "--attention-backend ROCM_FLASH"

    def test_three_occurrences_keep_only_last(self):
        out = _grid_runner.dedup_vllm_server_args(
            "--attention-backend A --attention-backend B --attention-backend C",
            "vllm",
        )
        assert out == "--attention-backend C"

    def test_json_flag_does_not_disable_other_flag_dedup(self):
        raw = "--attention-backend A --attention-backend B --override-generation-config '{\"temperature\": 0.7}'"
        out = _grid_runner.dedup_vllm_server_args(raw, "vllm")
        assert out == '--attention-backend B --override-generation-config {"temperature":0.7}'

    def test_json_string_with_internal_space_fails_closed(self):
        raw = '--attention-backend A --attention-backend B --speculative-config \'{"model":"draft model"}\''
        assert _grid_runner.dedup_vllm_server_args(raw, "vllm") == raw

    def test_quoted_non_json_operand_fragments_fail_closed(self):
        raw = '--tool-call-parser "my parser" --max-num-seqs 512 --max-num-seqs 1024'
        assert _grid_runner.dedup_vllm_server_args(raw, "vllm") == raw

    def test_multi_value_flag_left_untouched(self):
        # cuda-graph-bs takes a list; never collapse a string that carries one.
        raw = "--attention-backend A --cuda-graph-bs 1 2 4 8 --attention-backend B"
        assert _grid_runner.dedup_vllm_server_args(raw, "vllm") == raw


@pytest.mark.asyncio
async def test_run_grid_skips_all_variants_when_budget_already_exhausted(tmp_path):
    base = tmp_path / "base.yaml"
    _write_baseline_yaml_overrides(base)
    ran: list[str] = []

    def fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        ran.append(slot.name)
        _fake_workspace(slot)
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=fake_run,
    ):
        results = await run_grid(
            base_yaml_path=base,
            base_extra_args="",
            grid=[GridVariant("v0"), GridVariant("v1")],
            output_root=tmp_path / "out",
            variant_timeout_sec=5,
            session_deadline_sec=time.monotonic() - 1.0,
        )

    assert ran == []
    assert [r.status for r in results] == ["skipped", "skipped"]
    assert all(r.error_class == "session_time_exhausted" for r in results)


@pytest.mark.asyncio
async def test_run_grid_skips_remaining_when_budget_cannot_fit_a_variant(tmp_path):
    base = tmp_path / "base.yaml"
    _write_baseline_yaml_overrides(base)
    ran: list[str] = []

    def fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        ran.append(slot.name)
        _fake_workspace(slot)
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    # Deadline leaves less than one variant_timeout_sec of budget, so no variant
    # should start and all are skipped (last-variant overrun guard).
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=fake_run,
    ):
        results = await run_grid(
            base_yaml_path=base,
            base_extra_args="",
            grid=[GridVariant("v0"), GridVariant("v1")],
            output_root=tmp_path / "out",
            variant_timeout_sec=600,
            session_deadline_sec=time.monotonic() + 5.0,
        )

    assert ran == []
    assert [r.status for r in results] == ["skipped", "skipped"]


@pytest.mark.asyncio
async def test_run_grid_runs_all_when_no_session_deadline(tmp_path):
    base = tmp_path / "base.yaml"
    _write_baseline_yaml_overrides(base)
    ran: list[str] = []

    def fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        ran.append(slot.name)
        _fake_workspace(slot)
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=fake_run,
    ):
        results = await run_grid(
            base_yaml_path=base,
            base_extra_args="",
            grid=[GridVariant("v0"), GridVariant("v1")],
            output_root=tmp_path / "out",
            variant_timeout_sec=5,
            session_deadline_sec=None,
        )

    assert len(ran) == 2
    assert [r.status for r in results] == ["succeeded", "succeeded"]


def _capture_launches(recorded: list[dict]):
    """A ``run_with_session_kill`` double that records how each round was launched.

    Only benchmark rounds are recorded: the interpreter probe carries no
    ``--output-dir`` and is module-memoized, so counting it would make these
    assertions depend on which test ran first.

    Args:
        recorded: Appended to per launched round, each record carrying the
            ``round_slot`` the round wrote into alongside the launch kwargs.

    Returns:
        A callable usable as ``side_effect``.
    """

    def fake_run(cmd, *args, **kwargs):
        if "--output-dir" not in cmd:
            return subprocess.CompletedProcess(cmd, 0, "ok", "")
        slot = Path(cmd[cmd.index("--output-dir") + 1])
        recorded.append({"round_slot": slot.name, **kwargs})
        _fake_workspace(slot)
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    return fake_run


def _granted_timeouts(recorded: list[dict]) -> list[int]:
    """The hard timeout each recorded round was granted, in launch order."""
    return [int(launch["timeout"]) for launch in recorded]


# Every output slot one variant launches a benchmark process into, in launch
# order: the discarded warmup, the multi-node client warmup, and the measured
# round. All three are full benchmark passes on the GPU.
_GRID_ROUND_SLOTS = ("warmup_round", "mn_warmup", "variant_00_v0")


async def _launch_every_pass_of_one_variant(
    tmp_path,
    monkeypatch,
    *,
    session_deadline_sec: float | None,
) -> list[dict]:
    """Run one variant with every optional pass enabled, recording each launch.

    The passes are independently gated -- the discarded warmup on lifecycle
    eligibility, the client warmup on multi-node -- so a test that wants to reach
    every launch site the grid has must turn all of them on at once.

    Args:
        tmp_path: Test-scoped directory for the config and the output root.
        monkeypatch: Used to put the grid on the multi-node path.
        session_deadline_sec: The session deadline handed to ``run_grid``.

    Returns:
        list[dict]: One record per launched round, as ``_capture_launches`` makes
            them.
    """
    base = tmp_path / "base.yaml"
    _write_baseline_yaml_overrides(base)
    enable_multi_node(monkeypatch)
    recorded: list[dict] = []
    with (
        patch(
            "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
            side_effect=_capture_launches(recorded),
        ),
        patch(
            "hyperloom.orchestrator.actions.executors._server_lifecycle.resolve_lifecycle_params",
            return_value={"eligible": True, "framework": "sglang", "port": 30000, "reason": ""},
        ),
    ):
        await run_grid(
            base_yaml_path=base,
            base_extra_args="",
            grid=[GridVariant("v0")],
            output_root=tmp_path / "out",
            variant_timeout_sec=600,
            session_deadline_sec=session_deadline_sec,
            variant_expected_sec=30.0,
            warmup_before_measure=True,
        )
    return recorded


class TestSessionGridBounds:
    """One definition of the two numbers every benching arm needs.

    A deadline derived one way in one executor and another way in the next
    produces arms that abandon different amounts of the tail budget.
    """

    def test_no_session_means_no_bounds(self):
        assert _grid_runner.session_grid_bounds(None) == (None, None)

    def test_reads_the_deadline_and_the_measured_baseline(self):
        state = MagicMock()
        state.grid_session_deadline_sec.return_value = 4242.0
        state.baseline_runtime_sec = 600.0
        assert _grid_runner.session_grid_bounds(state) == (4242.0, 600.0)

    def test_unmeasured_baseline_yields_no_estimate(self):
        """Zero is "not measured yet", which must not read as "needs 0 seconds"."""
        state = MagicMock()
        state.grid_session_deadline_sec.return_value = 4242.0
        state.baseline_runtime_sec = 0.0
        assert _grid_runner.session_grid_bounds(state) == (4242.0, None)

    def test_unparseable_baseline_yields_no_estimate(self):
        state = MagicMock()
        state.grid_session_deadline_sec.return_value = None
        state.baseline_runtime_sec = "not-a-number"
        assert _grid_runner.session_grid_bounds(state) == (None, None)

    def test_state_without_the_deadline_accessor_is_tolerated(self):
        state = SimpleNamespace(baseline_runtime_sec=600.0)
        assert _grid_runner.session_grid_bounds(state) == (None, 600.0)

    def test_a_variant_is_priced_as_a_boot_and_a_benchmark(self):
        """What a variant actually spends, rather than what the baseline spent.

        A variant cannot re-attach to anyone else's server -- its config differs
        in the very knobs that decide how one comes up -- so it pays a boot and
        then a benchmark. The baseline's 900s cold round is 350s of boot and 550s
        of benchmarking that also paid the first request's kernel compile; the
        variant pays that boot and the 400s a benchmark costs once the compile is
        cached. Admitting on the 900s abandons 150s of every variant's worth of
        tail budget.
        """
        state = SimpleNamespace(
            grid_session_deadline_sec=lambda: 4242.0,
            baseline_runtime_sec=900.0,
            baseline_post_ready_runtime_sec=550.0,
            baseline_warm_runtime_sec=400.0,
        )

        deadline, variant_sec = _grid_runner.session_grid_bounds(state)

        assert deadline == 4242.0
        assert variant_sec == pytest.approx(750.0)

    def test_a_baseline_with_no_hot_pass_prices_the_benchmark_from_the_cold_one(self):
        """The post-ready segment stands in, over-predicting by the compile."""
        state = SimpleNamespace(
            grid_session_deadline_sec=lambda: None,
            baseline_runtime_sec=900.0,
            baseline_post_ready_runtime_sec=550.0,
        )

        assert _grid_runner.session_grid_bounds(state)[1] == pytest.approx(900.0)

    def test_a_baseline_that_never_reported_its_boot_falls_back_to_the_whole_round(self):
        """A scriptable workload runs no server, so there is no split to read."""
        state = SimpleNamespace(
            grid_session_deadline_sec=lambda: None,
            baseline_runtime_sec=900.0,
            baseline_warm_runtime_sec=400.0,
        )

        assert _grid_runner.session_grid_bounds(state)[1] == pytest.approx(900.0)


class TestSessionBudgetAdmission:
    """A variant is admitted on what it is expected to need, not on its backstop.

    ``variant_timeout_sec`` is the catastrophic-hang cap (~baseline x 2 for
    explore). Gating admission on it abandons the tail of the budget: with a
    20-minute baseline the grid refuses to start a round with 30 minutes left.
    """

    @pytest.mark.asyncio
    async def test_variant_runs_when_budget_fits_expected_but_not_the_backstop(self, tmp_path):
        base = tmp_path / "base.yaml"
        _write_baseline_yaml_overrides(base)
        recorded: list[dict] = []

        with patch(
            "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
            side_effect=_capture_launches(recorded),
        ):
            results = await run_grid(
                base_yaml_path=base,
                base_extra_args="",
                grid=[GridVariant("v0")],
                output_root=tmp_path / "out",
                variant_timeout_sec=600,
                session_deadline_sec=time.monotonic() + 120.0,
                variant_expected_sec=30.0,
            )

        assert [r.status for r in results] == ["succeeded"]
        assert len(recorded) == 1

    @pytest.mark.asyncio
    async def test_variant_skipped_when_budget_cannot_fit_the_expected_runtime(self, tmp_path):
        base = tmp_path / "base.yaml"
        _write_baseline_yaml_overrides(base)
        recorded: list[dict] = []

        with patch(
            "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
            side_effect=_capture_launches(recorded),
        ):
            results = await run_grid(
                base_yaml_path=base,
                base_extra_args="",
                grid=[GridVariant("v0")],
                output_root=tmp_path / "out",
                variant_timeout_sec=600,
                session_deadline_sec=time.monotonic() + 10.0,
                variant_expected_sec=300.0,
            )

        assert recorded == []
        assert [r.status for r in results] == ["skipped"]

    @pytest.mark.asyncio
    async def test_without_an_estimate_the_stricter_backstop_check_is_kept(self, tmp_path):
        """Callers that cannot estimate keep the pre-existing, stricter gate."""
        base = tmp_path / "base.yaml"
        _write_baseline_yaml_overrides(base)
        recorded: list[dict] = []

        with patch(
            "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
            side_effect=_capture_launches(recorded),
        ):
            results = await run_grid(
                base_yaml_path=base,
                base_extra_args="",
                grid=[GridVariant("v0")],
                output_root=tmp_path / "out",
                variant_timeout_sec=600,
                session_deadline_sec=time.monotonic() + 120.0,
                variant_expected_sec=None,
            )

        assert recorded == []
        assert [r.status for r in results] == ["skipped"]

    @pytest.mark.asyncio
    async def test_without_an_estimate_agentx_gates_on_its_raised_cap(self, tmp_path, monkeypatch):
        """The AgentX-raised cap, not the declared one, must gate admission.

        Gating on the declared ``variant_timeout_sec`` (600) would admit this
        variant with 700s left on the clock; the round is then handed the
        AgentX-raised cap (10800s) by ``_round_timeout_sec``, which
        ``session_clamped_timeout_sec`` immediately clamps back down to the
        ~700s actually remaining -- reproducing the mid-warmup kill this
        AgentX cap-raise exists to prevent.
        """
        monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
        monkeypatch.setenv("AGENTX_DURATION", "3600")
        monkeypatch.setenv("AGENTX_BASELINE_OVERHEAD_SEC", "7200")
        monkeypatch.delenv("AGENTX_BASELINE_TIMEOUT_SEC", raising=False)
        base = tmp_path / "base.yaml"
        _write_baseline_yaml_overrides(base)
        recorded: list[dict] = []

        with patch(
            "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
            side_effect=_capture_launches(recorded),
        ):
            results = await run_grid(
                base_yaml_path=base,
                base_extra_args="",
                grid=[GridVariant("v0")],
                output_root=tmp_path / "out",
                variant_timeout_sec=600,
                session_deadline_sec=time.monotonic() + 700.0,
                variant_expected_sec=None,
            )

        assert recorded == []
        assert [r.status for r in results] == ["skipped"]


class TestSessionBudgetTimeoutClamp:
    """A granted cap never exceeds what the session can still pay for.

    explore derives caps from the measured baseline (up to 4h) and never
    consulted the budget, so a 3h session could hand a single variant more time
    than the whole run was given.
    """

    @pytest.mark.asyncio
    async def test_granted_cap_is_clamped_to_the_remaining_budget(self, tmp_path):
        base = tmp_path / "base.yaml"
        _write_baseline_yaml_overrides(base)
        recorded: list[dict] = []

        with patch(
            "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
            side_effect=_capture_launches(recorded),
        ):
            await run_grid(
                base_yaml_path=base,
                base_extra_args="",
                grid=[GridVariant("v0")],
                output_root=tmp_path / "out",
                variant_timeout_sec=7800,
                session_deadline_sec=time.monotonic() + 120.0,
                variant_expected_sec=30.0,
            )

        assert len(recorded) == 1
        granted = _granted_timeouts(recorded)[0]
        # The cap is allowed a small grace past the deadline so the in-process
        # session watchdog trips first and attributes the kill correctly.
        assert 60 <= granted <= 120 + _SESSION_KILL_GRACE_SEC, (
            f"expected a cap clamped to the ~120s budget, got {granted}"
        )

    @pytest.mark.asyncio
    async def test_declared_cap_is_kept_when_the_budget_is_larger(self, tmp_path):
        base = tmp_path / "base.yaml"
        _write_baseline_yaml_overrides(base)
        recorded: list[dict] = []

        with patch(
            "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
            side_effect=_capture_launches(recorded),
        ):
            await run_grid(
                base_yaml_path=base,
                base_extra_args="",
                grid=[GridVariant("v0")],
                output_root=tmp_path / "out",
                variant_timeout_sec=600,
                session_deadline_sec=time.monotonic() + 36000.0,
                variant_expected_sec=30.0,
            )

        assert _granted_timeouts(recorded) == [600]

    @pytest.mark.asyncio
    async def test_no_deadline_leaves_the_declared_cap_untouched(self, tmp_path):
        base = tmp_path / "base.yaml"
        _write_baseline_yaml_overrides(base)
        recorded: list[dict] = []

        with patch(
            "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
            side_effect=_capture_launches(recorded),
        ):
            await run_grid(
                base_yaml_path=base,
                base_extra_args="",
                grid=[GridVariant("v0")],
                output_root=tmp_path / "out",
                variant_timeout_sec=600,
                session_deadline_sec=None,
                variant_expected_sec=30.0,
            )

        assert _granted_timeouts(recorded) == [600]


class TestSessionKillAttribution:
    """A round reaped for the session budget is not a verdict about the variant."""

    @pytest.mark.asyncio
    async def test_mid_round_budget_kill_is_recorded_as_skipped_not_failed(self, tmp_path):
        base = tmp_path / "base.yaml"
        _write_baseline_yaml_overrides(base)

        def fake_run(cmd, *args, **kwargs):
            return subprocess.CompletedProcess(cmd, SESSION_TIME_EXHAUSTED_RETURNCODE, "", "")

        with patch(
            "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
            side_effect=fake_run,
        ):
            results = await run_grid(
                base_yaml_path=base,
                base_extra_args="",
                grid=[GridVariant("v0")],
                output_root=tmp_path / "out",
                variant_timeout_sec=600,
                session_deadline_sec=time.monotonic() + 120.0,
                variant_expected_sec=30.0,
            )

        assert [r.status for r in results] == ["skipped"]
        assert results[0].error_class == "session_time_exhausted"
        # Never the overtime label, which asserts the variant is abnormally slow.
        assert not getattr(results[0], "killed_overtime", False)
        assert results[0].output_throughput is None

    @pytest.mark.asyncio
    async def test_the_hard_cap_leaves_room_for_the_session_watchdog_to_win(self, tmp_path):
        """Both fire at the same instant, and the sentinel must get there first.

        The hard cap raises ``TimeoutExpired``, which the ledger reads as a variant
        timeout, so it is granted a small grace past the session deadline.
        """
        base = tmp_path / "base.yaml"
        _write_baseline_yaml_overrides(base)
        recorded: list[dict] = []

        with patch(
            "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
            side_effect=_capture_launches(recorded),
        ):
            await run_grid(
                base_yaml_path=base,
                base_extra_args="",
                grid=[GridVariant("v0")],
                output_root=tmp_path / "out",
                variant_timeout_sec=7800,
                session_deadline_sec=time.monotonic() + 60.0,
                variant_expected_sec=30.0,
            )

        assert len(recorded) == 1
        granted = _granted_timeouts(recorded)[0]
        assert granted > 60, f"hard cap {granted}s must sit past the ~60s deadline, not on it"
        assert granted <= 90, f"the grace must stay small, got {granted}s"

    @pytest.mark.parametrize("round_slot", _GRID_ROUND_SLOTS)
    @pytest.mark.asyncio
    async def test_the_session_deadline_reaches_the_subprocess_layer(self, tmp_path, monkeypatch, round_slot):
        """Regression: the clamped cap alone bounds the round but mislabels the kill.

        Parameterized over every pass a variant costs, because each launch site
        hands the deadline over on its own. A site that leaves it out is still
        bounded by its clamped cap, so the round still ends -- as a
        ``TimeoutExpired`` the ledger reads as a variant too slow to measure.
        """
        deadline = time.monotonic() + 120.0
        recorded = await _launch_every_pass_of_one_variant(
            tmp_path,
            monkeypatch,
            session_deadline_sec=deadline,
        )

        assert launches_by_round_slot(recorded)[round_slot]["session_deadline_sec"] == deadline

    @pytest.mark.asyncio
    async def test_no_pass_of_a_variant_is_launched_without_the_deadline(self, tmp_path, monkeypatch):
        """The net for a pass added later, which no per-slot test would know about."""
        deadline = time.monotonic() + 120.0
        recorded = await _launch_every_pass_of_one_variant(
            tmp_path,
            monkeypatch,
            session_deadline_sec=deadline,
        )

        assert set(launches_by_round_slot(recorded)) >= set(_GRID_ROUND_SLOTS)
        assert [c["round_slot"] for c in recorded if c.get("session_deadline_sec") != deadline] == []


def _reaping_round(returncode: int, *, slot_name: str):
    """A ``run_with_session_kill`` double that reaps one named round of a variant.

    Args:
        returncode: The sentinel the reaped round comes back with.
        slot_name: Output-slot directory name identifying the round to reap;
            every other round succeeds with a valid report.

    Returns:
        tuple: The ``side_effect`` callable, and the list of slot names it
            appends to as rounds are launched.
    """
    launched: list[str] = []

    def fake_run(cmd, *args, **kwargs):
        # The module-memoized interpreter probe is not a benchmark round.
        if "--output-dir" not in cmd:
            return subprocess.CompletedProcess(cmd, 0, "ok", "")
        slot = Path(cmd[cmd.index("--output-dir") + 1])
        launched.append(slot.name)
        if slot.name == slot_name:
            return subprocess.CompletedProcess(cmd, returncode, "", "")
        _fake_workspace(slot)
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    return fake_run, launched


class TestEveryRoundCarriesTheStopThatEndedIt:
    """A round the run stopped is a stop whichever round it was.

    The measured round is not the only full benchmark pass a variant costs: the
    discarded warmup runs the same workload, and so does the multi-node client
    warmup. A reap in either has exactly as much to say about the variant as one
    in the measured round -- nothing -- so grading it as ``warmup_round_failed``
    files a verdict the run never reached, and ignoring the returncode entirely
    keeps launching benchmark rounds after the orchestrator asked the action to
    stop.
    """

    @pytest.mark.asyncio
    async def test_a_warmup_reaped_by_the_budget_is_skipped_not_a_failed_variant(self, tmp_path):
        base = tmp_path / "base.yaml"
        _write_baseline_yaml_overrides(base)
        fake_run, launched = _reaping_round(SESSION_TIME_EXHAUSTED_RETURNCODE, slot_name="warmup_round")

        with (
            patch(
                "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
                side_effect=fake_run,
            ),
            patch(
                "hyperloom.orchestrator.actions.executors._server_lifecycle.resolve_lifecycle_params",
                return_value={"eligible": True, "framework": "sglang", "port": 30000, "reason": ""},
            ),
        ):
            results = await run_grid(
                base_yaml_path=base,
                base_extra_args="",
                grid=[GridVariant("cand0")],
                output_root=tmp_path / "out",
                variant_timeout_sec=600,
                session_deadline_sec=time.monotonic() + 600.0,
                variant_expected_sec=30.0,
                warmup_before_measure=True,
            )

        assert launched == ["warmup_round"], "the measured round must not run after the warmup was reaped"
        assert [r.status for r in results] == ["skipped"]
        assert results[0].error_class == "session_time_exhausted"
        assert _read_marker(tmp_path / "out" / "variant_00_cand0")["error_class"] == "session_time_exhausted"

    @pytest.mark.asyncio
    async def test_a_cancelled_warmup_ends_the_grid_instead_of_booting_the_next_variant(self, tmp_path):
        """Every remaining variant would boot its own server on the Ray path.

        ``run_session_kill`` re-``ensure()``s a lease whose actor the cancel just
        killed, so a grid that keeps going after a cancel starts a fresh actor
        and a fresh GPU server per remaining variant.
        """
        base = tmp_path / "base.yaml"
        _write_baseline_yaml_overrides(base)
        fake_run, launched = _reaping_round(ORCHESTRATOR_CANCELLED_RETURNCODE, slot_name="warmup_round")

        with (
            patch(
                "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
                side_effect=fake_run,
            ),
            patch(
                "hyperloom.orchestrator.actions.executors._server_lifecycle.resolve_lifecycle_params",
                return_value={"eligible": True, "framework": "sglang", "port": 30000, "reason": ""},
            ),
        ):
            results = await run_grid(
                base_yaml_path=base,
                base_extra_args="",
                grid=[GridVariant("c0"), GridVariant("c1"), GridVariant("c2")],
                output_root=tmp_path / "out",
                variant_timeout_sec=600,
                keep_going_on_failure=True,
                session_deadline_sec=time.monotonic() + 600.0,
                variant_expected_sec=30.0,
                warmup_before_measure=True,
            )

        assert launched == ["warmup_round"], f"a cancelled action kept launching rounds: {launched}"
        assert [r.status for r in results] == ["skipped"] * 3
        assert [r.error_class for r in results] == ["orchestrator_cancelled"] * 3

    @pytest.mark.asyncio
    async def test_a_cancelled_multi_node_warmup_ends_the_grid(self, tmp_path, monkeypatch):
        """The multi-node warmup discarded its returncode along with its report."""
        from hyperloom.orchestrator.actions.executors import _multi_node_env as mne
        from hyperloom.orchestrator.actions.executors import _multi_node_server_lifecycle as mnsl

        base = tmp_path / "base.yaml"
        _write_baseline_yaml_overrides(base)
        monkeypatch.setattr(mne, "is_multi_node", lambda: True)
        monkeypatch.setattr(mne, "mn_bench_warmup_enabled", lambda: True)

        async def fake_restart_server_for_round(**_kwargs):
            return None

        monkeypatch.setattr(mnsl, "restart_server_for_round", fake_restart_server_for_round)
        fake_run, launched = _reaping_round(ORCHESTRATOR_CANCELLED_RETURNCODE, slot_name="mn_warmup")

        with patch(
            "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
            side_effect=fake_run,
        ):
            results = await run_grid(
                base_yaml_path=base,
                base_extra_args="",
                grid=[GridVariant("c0"), GridVariant("c1")],
                output_root=tmp_path / "out",
                variant_timeout_sec=600,
                keep_going_on_failure=True,
                session_deadline_sec=time.monotonic() + 600.0,
                variant_expected_sec=30.0,
            )

        assert launched == ["mn_warmup"], f"a cancelled action kept launching rounds: {launched}"
        assert [r.error_class for r in results] == ["orchestrator_cancelled"] * 2


class TestSessionBudgetWarmupRounds:
    """A warmup round costs a full pass, and the measured round is paid first."""

    @pytest.mark.asyncio
    async def test_admission_accounts_for_the_warmup_pass(self, tmp_path):
        """Budget for one round is not budget for a warmup plus a measure round.

        Admitting on a single round's estimate would let a variant in and then
        clamp its measured round to nothing, turning a budget shortfall into a
        ledger full of spurious timeouts.
        """
        base = tmp_path / "base.yaml"
        _write_baseline_yaml_overrides(base)
        recorded: list[dict] = []

        with (
            patch(
                "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
                side_effect=_capture_launches(recorded),
            ),
            patch(
                "hyperloom.orchestrator.actions.executors._server_lifecycle.resolve_lifecycle_params",
                return_value={"eligible": True, "framework": "sglang", "port": 30000, "reason": ""},
            ),
        ):
            results = await run_grid(
                base_yaml_path=base,
                base_extra_args="",
                grid=[GridVariant("v0")],
                output_root=tmp_path / "out",
                variant_timeout_sec=600,
                session_deadline_sec=time.monotonic() + 40.0,
                variant_expected_sec=30.0,
                warmup_before_measure=True,
            )

        assert recorded == []
        assert [r.status for r in results] == ["skipped"]

    @pytest.mark.asyncio
    async def test_the_budget_is_re_checked_after_the_uncapped_server_restart(
        self,
        tmp_path,
        monkeypatch,
    ):
        """The admission gate is taken before the one launch it does not cover.

        The per-variant multi-node server restart sits between the gate at the top
        of the loop and the first pass, and it is under no cap of its own: booting
        a large model across nodes can take longer than a benchmark pass. So a
        variant can be admitted on a budget that fits both its passes and reach the
        warmup with a budget that fits neither -- and the grid has no skip there,
        so it launches the warmup anyway, watches it get killed, swallows that as
        best-effort, and finds the measured round no longer fits.

        Scaled down by a thousand from the field shape (1300s left, 2x600s
        admitted, a 300s restart) so the restart's cost is real elapsed time
        rather than a clock the test pretends about.
        """
        from hyperloom.orchestrator.actions.executors import _multi_node_env as mne
        from hyperloom.orchestrator.actions.executors import _multi_node_server_lifecycle as mnsl

        base = tmp_path / "base.yaml"
        _write_baseline_yaml_overrides(base)
        monkeypatch.setattr(mne, "is_multi_node", lambda: True)
        monkeypatch.setattr(mne, "mn_bench_warmup_enabled", lambda: True)
        restarts: list[float] = []

        async def slow_restart(**_kwargs):
            restarts.append(time.monotonic())
            await asyncio.sleep(0.5)

        monkeypatch.setattr(mnsl, "restart_server_for_round", slow_restart)
        recorded: list[dict] = []

        with patch(
            "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
            side_effect=_capture_launches(recorded),
        ):
            results = await run_grid(
                base_yaml_path=base,
                base_extra_args="",
                grid=[GridVariant("v0")],
                output_root=tmp_path / "out",
                variant_timeout_sec=600,
                session_deadline_sec=time.monotonic() + 1.3,
                variant_expected_sec=0.6,
            )

        assert restarts, "the restart never ran, so this is not the case under test"
        launched = [c["round_slot"] for c in recorded]
        assert launched == [], f"a pass was launched into a budget the restart had spent: {launched}"
        assert [r.status for r in results] == ["skipped"]
        assert [r.error_class for r in results] == [SESSION_TIME_EXHAUSTED_CLASS]

    @pytest.mark.asyncio
    async def test_warmup_cap_reserves_budget_for_the_measured_round(self, tmp_path):
        base = tmp_path / "base.yaml"
        _write_baseline_yaml_overrides(base)
        recorded: list[dict] = []

        with (
            patch(
                "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
                side_effect=_capture_launches(recorded),
            ),
            patch(
                "hyperloom.orchestrator.actions.executors._server_lifecycle.resolve_lifecycle_params",
                return_value={"eligible": True, "framework": "sglang", "port": 30000, "reason": ""},
            ),
        ):
            await run_grid(
                base_yaml_path=base,
                base_extra_args="",
                grid=[GridVariant("v0")],
                output_root=tmp_path / "out",
                variant_timeout_sec=7800,
                session_deadline_sec=time.monotonic() + 300.0,
                variant_expected_sec=60.0,
                warmup_before_measure=True,
            )

        by_round = launches_by_round_slot(recorded)
        warmup = next(int(c["timeout"]) for c in recorded if "warmup" in c["round_slot"])
        measure = next(int(c["timeout"]) for c in recorded if "warmup" not in c["round_slot"])
        assert warmup >= 60, f"a warmup capped under the 60s a pass takes is launched to be killed, got {warmup}"
        assert warmup <= 240 + _SESSION_KILL_GRACE_SEC, (
            f"warmup cap must hold back the measured round's 60s, got {warmup}"
        )
        assert warmup <= measure, "the warmup is never granted more than the round it holds budget back for"
        assert len(by_round) == 2, f"expected a warmup and a measured round, got {list(by_round)}"

    @pytest.mark.asyncio
    async def test_a_warmup_killed_at_a_clamped_cap_is_logged_with_the_cap_it_got(self, tmp_path, caplog):
        """The abort line is the only record of how long the round was allowed.

        The declared cap is a hang backstop; what the warmup was granted is that
        cap minus the reserve, and a round killed after four minutes logged as a
        two-hour timeout reads as a variant that hangs rather than a budget that
        ran out.
        """
        base = tmp_path / "base.yaml"
        _write_baseline_yaml_overrides(base)
        recorded: list[dict] = []

        def fake_run(cmd, *args, **kwargs):
            if "--output-dir" not in cmd:
                return subprocess.CompletedProcess(cmd, 0, "ok", "")
            slot = Path(cmd[cmd.index("--output-dir") + 1])
            recorded.append({"round_slot": slot.name, **kwargs})
            raise subprocess.TimeoutExpired(cmd, float(kwargs.get("timeout") or 0.0))

        with (
            patch(
                "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
                side_effect=fake_run,
            ),
            patch(
                "hyperloom.orchestrator.actions.executors._server_lifecycle.resolve_lifecycle_params",
                return_value={"eligible": True, "framework": "sglang", "port": 30000, "reason": ""},
            ),
            caplog.at_level("WARNING"),
        ):
            results = await run_grid(
                base_yaml_path=base,
                base_extra_args="",
                grid=[GridVariant("v0")],
                output_root=tmp_path / "out",
                variant_timeout_sec=7800,
                session_deadline_sec=time.monotonic() + 300.0,
                variant_expected_sec=60.0,
                warmup_before_measure=True,
            )

        granted = int(recorded[0]["timeout"])
        assert granted < 7800, f"this test needs a clamped cap to be about, got {granted}"
        aborts = [r.message for r in caplog.records if "warmup timeout" in r.message]
        assert aborts, f"the warmup abort was not logged: {[r.message for r in caplog.records]}"
        assert f"timeout_sec={granted}" in aborts[0], f"the abort line reports a cap the round never had: {aborts[0]}"
        assert results[0].error_class == "warmup_magpie_timeout"


class TestCompactJsonServerArgs:
    """JSON-valued flags must be space-free to survive Magpie's unquoted
    ``$EXTRA_VLLM_ARGS`` splice (otherwise spec-decode / compilation-config
    explore variants always crash the server at boot)."""

    def test_compilation_config_separator_space_removed(self):
        out = _grid_runner.compact_json_server_args('--compilation-config {"full_cuda_graph": true}', "vllm")
        assert out == '--compilation-config {"full_cuda_graph":true}'
        # the JSON value is now a single shell word under bash word-splitting
        assert len(out.split()) == 2

    def test_speculative_config_multikey(self):
        out = _grid_runner.compact_json_server_args(
            '--speculative-config {"method": "eagle", "num_speculative_tokens": 3}',
            "vllm",
        )
        assert out == ('--speculative-config {"method":"eagle","num_speculative_tokens":3}')
        assert len(out.split()) == 2

    def test_compact_json_server_args_internal_space_unsupported(self):
        # Spaces inside JSON string values are not collapsed, so the value is not
        # a single shell word under Magpie's unquoted expansion; this flag shape
        # is unsupported. The value is left intact, not corrupted.
        out = _grid_runner.compact_json_server_args('--speculative-config {"model": "draft model name"}', "vllm")
        # Value is preserved verbatim (separator space after ':' removed only).
        assert out == '--speculative-config {"model":"draft model name"}'
        # ...but it still splits into MORE than the ideal 2 words: the two
        # internal spaces of "draft model name" survive, so the shell sees
        # ['--speculative-config', '{"model":"draft', 'model', 'name"}'].
        assert len(out.split()) == 4

    def test_quote_stripped_json_is_repaired(self):
        # A prior shlex round-trip (e.g. in the GEMM shape-capture path) can strip
        # the JSON double-quotes, turning a stored-valid ``{"method":"ngram",...}``
        # into ``{method:ngram,...}`` which vLLM's ``json.loads`` rejects at boot
        # (observed: shape_capture server never boots -> shape_capture_failed).
        # compact must repair the barewords back to valid JSON.
        out = _grid_runner.compact_json_server_args(
            "--speculative-config {method:ngram,num_speculative_tokens:7,prompt_lookup_min:2,prompt_lookup_max:8}",
            "vllm",
        )
        assert out == (
            "--speculative-config "
            '{"method":"ngram","num_speculative_tokens":7,"prompt_lookup_min":2,"prompt_lookup_max":8}'
        )
        blob = out.split(" ", 1)[1]
        json.loads(blob)  # must be valid JSON now

    def test_quote_stripped_json_with_path_value_repaired(self):
        # Bareword string values containing ``/``, ``.``, ``-`` (e.g. a model id)
        # must also be re-quoted.
        out = _grid_runner.compact_json_server_args(
            "--speculative-config {method:eagle3,model:RedHatAI/Llama-3.1-8B-Instruct-speculator}",
            "vllm",
        )
        blob = out.split(" ", 1)[1]
        assert json.loads(blob) == {
            "method": "eagle3",
            "model": "RedHatAI/Llama-3.1-8B-Instruct-speculator",
        }

    def test_quote_stripped_nested_json_is_repaired(self):
        out = _grid_runner.compact_json_server_args(
            "--compilation-config {method:ngram,nested:{a:1}}",
            "vllm",
        )
        blob = out.split(" ", 1)[1]
        assert json.loads(blob) == {
            "method": "ngram",
            "nested": {"a": 1},
        }

    def test_shell_single_quote_wrappers_are_removed(self):
        out = _grid_runner.compact_json_server_args(
            """--speculative-config '{"method":"ngram","num_speculative_tokens":7}' """,
            "vllm",
        )
        assert out.strip() == ('--speculative-config {"method":"ngram","num_speculative_tokens":7}')
        json.loads(out.split(" ", 1)[1].strip())

    def test_unrepairable_json_left_verbatim(self):
        # Genuinely broken blobs that cannot be repaired to valid JSON are left
        # verbatim (no worse than before), never raising.
        raw = "--compilation-config {this is : not ] json"
        out = _grid_runner.compact_json_server_args(raw, "vllm")
        assert "{this is" in out

    def test_other_flags_around_json_untouched(self):
        out = _grid_runner.compact_json_server_args('--kv-cache-dtype fp8 --compilation-config {"level": 3}', "vllm")
        assert out == '--kv-cache-dtype fp8 --compilation-config {"level":3}'

    def test_sglang_json_is_normalized_for_unquoted_transport(self):
        raw = '--speculative-config {"method": "eagle"}'
        assert _grid_runner.compact_json_server_args(raw, "sglang") == ('--speculative-config {"method":"eagle"}')

    def test_sglang_and_missing_framework_remove_legacy_shell_wrappers(self):
        raw = """--json-model-override-args '{"rope_scaling":null}'"""
        expected = '--json-model-override-args {"rope_scaling":null}'
        assert _grid_runner.compact_json_server_args(raw, "sglang") == expected
        assert _grid_runner.compact_json_server_args(raw, None) == expected

    def test_no_json_is_noop(self):
        raw = "--block-size 128 --no-enable-prefix-caching"
        assert _grid_runner.compact_json_server_args(raw, "vllm") == raw

    def test_malformed_json_left_verbatim(self):
        raw = "--compilation-config {not json}"
        assert _grid_runner.compact_json_server_args(raw, "vllm") == raw

    def test_malformed_wrapped_json_keeps_both_shell_wrappers(self):
        raw = "--compilation-config '{not json}'"
        assert _grid_runner.compact_json_server_args(raw, "vllm") == raw

    def test_empty_is_noop(self):
        assert _grid_runner.compact_json_server_args("", "vllm") == ""
        assert _grid_runner.compact_json_server_args(None, "vllm") == ""


class TestRemoveServerArgsPreservesJson:
    """``remove_server_args`` tokenizes with ``shlex.split`` (which strips JSON
    inner double quotes) and runs AFTER ``compact_json_server_args`` in
    ``materialize_config_with_envs`` (GEMM shape-capture always passes
    ``remove_args=['--port']``). It must therefore re-quote/compact the JSON
    blobs itself, or a sibling ``--compilation-config`` / ``--speculative-config``
    is silently corrupted to unquoted barewords that vLLM rejects at boot."""

    def test_remove_port_keeps_sibling_json_valid(self):
        raw = '--compilation-config {"cudagraph_mode":"FULL"} --port 8888'
        out = _grid_runner.remove_server_args(raw, ["--port"])
        assert "--port" not in out
        blob = out.split("--compilation-config ", 1)[1].strip()
        assert json.loads(blob) == {"cudagraph_mode": "FULL"}

    def test_remove_keeps_multikey_speculative_config_valid(self):
        raw = '--speculative-config {"method":"ngram","num_speculative_tokens":7} --port 8888'
        out = _grid_runner.remove_server_args(raw, ["--port"])
        blob = out.split("--speculative-config ", 1)[1].strip()
        assert json.loads(blob) == {"method": "ngram", "num_speculative_tokens": 7}

    def test_remove_noop_when_nothing_matches_keeps_json(self):
        raw = '--compilation-config {"cudagraph_mode":"FULL"}'
        out = _grid_runner.remove_server_args(raw, ["--port"])
        assert json.loads(out.split("--compilation-config ", 1)[1].strip()) == {"cudagraph_mode": "FULL"}

    def test_sign_prefixed_custom_op_survives_removal(self):
        """A ``+``/``-`` prefixed custom-op value must survive the round trip.

        Live regression: every variant of an explore round died at vLLM argv
        parse with ``Invalid JSON: key must be a string`` -- including a control
        leg that added one env var and zero args, because
        ``strip_benchmark_harness_flags`` routes EVERY composed variant through
        ``remove_server_args`` with a non-empty denylist. The old POSIX
        ``shlex.split``/rejoin stripped the JSON quotes, and the bareword repair
        heuristic could not re-quote ``+fused_rms_norm_gated`` (its ``+`` was
        outside the charset), so the corruption reached the server verbatim.
        """
        raw = (
            "--compilation-config "
            '{"mode":3,"custom_ops":["+fused_rms_norm_gated"],'
            '"cudagraph_capture_sizes":[1,2,3]} --port 8888'
        )
        out = _grid_runner.remove_server_args(raw, ["--port"])
        assert "--port" not in out
        blob = json.loads(out.split("--compilation-config ", 1)[1].strip())
        assert blob["custom_ops"] == ["+fused_rms_norm_gated"]
        assert blob["cudagraph_capture_sizes"] == [1, 2, 3]

    def test_compose_preserves_json_for_every_args_mode(self):
        """All four variant shapes keep both JSON flags ``json.loads``-able."""
        cc = '{"mode":3,"custom_ops":["+fused_rms_norm_gated"]}'
        sc = '{"method":"mtp","num_speculative_tokens":2}'
        base = f"--max-num-seqs 20 --compilation-config {cc} --speculative-config {sc}"

        def _blob(text, flag):
            toks = text.split()
            return json.loads(toks[toks.index(flag) + 1])

        for kwargs in (
            {"inherited_args": base},
            {"inherited_args": base, "remove_args": ["--max-num-seqs"]},
            {"base_extra_args": base, "args_mode": "replace"},
        ):
            out = _grid_runner.compose_server_args(**kwargs)
            assert _blob(out, "--compilation-config") == json.loads(cc)
            assert _blob(out, "--speculative-config") == json.loads(sc)
        # Removing the JSON flag and supplying a variant replacement.
        repl = '{"mode":3,"custom_ops":["-rms_norm"]}'
        out = _grid_runner.compose_server_args(
            inherited_args=base,
            remove_args=["--compilation-config"],
            variant_extra_args=f"--compilation-config {repl}",
        )
        assert out.count("--compilation-config") == 1
        assert _blob(out, "--compilation-config") == json.loads(repl)
        assert _blob(out, "--speculative-config") == json.loads(sc)

    def test_shell_quoted_plain_operand_loses_its_wrappers(self):
        """Magpie expands ``EXTRA_*_ARGS`` unquoted, so wrappers must not persist."""
        out = _grid_runner.remove_server_args("--tool-call-parser 'kimi_k3' --port 8888", ["--port"])
        assert out == "--tool-call-parser kimi_k3"


# ---------------------------------------------------------------------------
# benchmark_report settling (real-run race)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_report_read_waits_for_a_report_still_being_written(tmp_path):
    """A report that lands a moment after the process exits must not abort the variant.

    Observed on a live 24-hour session: the reader ran in the same second the
    benchmark finished, read a ``benchmark_report.json`` that was not fully on disk
    yet, and aborted the variant as ``benchmark_report_invalid_metric``. Six of
    thirteen variants died that way while their reports, read afterwards, all held
    valid throughput — including two authored patches worth +4.4% and +4.7% whose
    switch-off parity legs had passed. The measurement was fine; only the moment of
    reading was wrong.
    """
    workspace = tmp_path / "benchmark_ws"
    workspace.mkdir()
    report_path = workspace / "benchmark_report.json"
    # First read sees a truncated file, exactly as a partial flush leaves it.
    report_path.write_text('{"throughput": {"output_th', encoding="utf-8")

    reads = {"n": 0}
    real_parse = _grid_runner._parse_report

    def _parse_then_complete(ws):
        reads["n"] += 1
        result = real_parse(ws)
        if reads["n"] == 1:
            report_path.write_text(
                json.dumps({"throughput": {"output_throughput": 0.351, "completed_requests": 3}}),
                encoding="utf-8",
            )
        return result

    with patch.object(_grid_runner, "_parse_report", _parse_then_complete):
        report, measurement = await _grid_runner._settled_measurement(
            workspace,
            subprocess_started_unix=None,
            settle_seconds=5.0,
            poll_seconds=0.01,
        )

    assert reads["n"] >= 2, "a truncated report must be re-read, not accepted as final"
    assert measurement.get("valid_measurement") is True
    assert report is not None


@pytest.mark.asyncio
async def test_report_read_gives_up_on_a_genuinely_invalid_report(tmp_path):
    """Settling must be bounded: a report that never becomes valid still aborts."""
    workspace = tmp_path / "benchmark_ws"
    workspace.mkdir()
    (workspace / "benchmark_report.json").write_text(
        json.dumps({"throughput": {"output_throughput": 0.0, "completed_requests": 0}}),
        encoding="utf-8",
    )
    started = time.monotonic()
    _report, measurement = await _grid_runner._settled_measurement(
        workspace,
        subprocess_started_unix=None,
        settle_seconds=0.05,
        poll_seconds=0.01,
    )
    assert not measurement.get("valid_measurement")
    assert time.monotonic() - started < 3.0, "a dead report must not stall the grid"


@pytest.mark.asyncio
async def test_report_read_does_not_wait_when_the_process_already_failed(tmp_path):
    """With settling disabled the reader takes one look, so a crashed run fails fast."""
    workspace = tmp_path / "benchmark_ws"
    workspace.mkdir()
    reads = {"n": 0}
    real_parse = _grid_runner._parse_report

    def _counting_parse(ws):
        reads["n"] += 1
        return real_parse(ws)

    with patch.object(_grid_runner, "_parse_report", _counting_parse):
        _report, measurement = await _grid_runner._settled_measurement(
            workspace,
            subprocess_started_unix=None,
            settle_seconds=0.0,
            poll_seconds=0.01,
        )
    assert reads["n"] == 1
    assert not measurement.get("valid_measurement")


# --- the JSON-preserving tokenizer is on the DEFAULT path, not just AgentX -----


class TestServerArgTokenizerOnTheSyntheticPath:
    """``strip_benchmark_harness_flags`` runs for every variant, AgentX or not.

    The PR note "AgentX-off is a no-op" does not hold in this file:
    ``compose_server_args`` always calls ``strip_benchmark_harness_flags``, which
    is ``remove_server_args`` with a non-empty denylist, so every synthetic grid
    variant goes through the replaced tokenizer. These lock the behaviour that
    matters there, with HYPERLOOM_AGENTX unset.
    """

    def _off(self, monkeypatch):
        monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)

    def test_a_plain_synthetic_arg_string_round_trips(self, monkeypatch):
        self._off(monkeypatch)
        args = "--tensor-parallel-size 8 --gpu-memory-utilization 0.9 --max-num-seqs 512"
        assert _grid_runner.compose_server_args(base_extra_args=args) == args

    def test_the_denylisted_flag_is_still_dropped(self, monkeypatch):
        self._off(monkeypatch)
        out = _grid_runner.compose_server_args(base_extra_args="--no-enable-prefix-caching --max-num-seqs 512")
        assert "--no-enable-prefix-caching" not in out
        assert "--max-num-seqs 512" in out

    def test_quoted_operands_do_not_keep_their_wrappers(self, monkeypatch):
        """Magpie expands EXTRA_*_ARGS unquoted, so a wrapper reaches argv literally."""
        self._off(monkeypatch)
        out = _grid_runner.compose_server_args(base_extra_args="--quantization 'fp8' --max-num-seqs 512")
        assert out == "--quantization fp8 --max-num-seqs 512"

    def test_an_unbalanced_quote_leaves_the_string_alone(self, monkeypatch):
        """Untokenizable input is returned untouched rather than guessed at."""
        self._off(monkeypatch)
        broken = "--served-model-name 'oops --max-num-seqs 512"
        assert _grid_runner.remove_server_args(broken, ["--max-num-seqs"]) == broken

    def test_a_synthetic_json_value_survives_a_removal(self, monkeypatch):
        """Synthetic runs carry JSON flags too (compilation-config, and friends)."""
        self._off(monkeypatch)
        args = '--compilation-config {"mode":3} --max-num-seqs 512'
        out = _grid_runner.remove_server_args(args, ["--max-num-seqs"])
        assert out == '--compilation-config {"mode":3}'


def test_the_json_tripwire_sees_damage_from_the_removal_pass(caplog):
    """The window must cover ``remove_server_args``, which is what it is about.

    It compared ``composed`` (already that function's output) against the final
    string, so damage done during removal made the "before" side unparseable
    too, ``healthy_before`` False, and the tripwire silent on precisely the
    failure it was written for.
    """
    from hyperloom.orchestrator.actions.executors import _grid_server_args as gsa

    real = gsa.remove_server_args

    def _lossy(server_args, remove_args):
        out = real(server_args, remove_args)
        return out.replace('{"mode":3}', "{mode:3}")

    args = '--compilation-config {"mode":3} --max-num-seqs 512'
    with patch.object(gsa, "remove_server_args", side_effect=_lossy):
        with caplog.at_level("ERROR"):
            gsa.compose_server_args(base_extra_args=args, remove_args=["--max-num-seqs"])
    assert any("CORRUPTED" in r.getMessage() for r in caplog.records), [r.getMessage() for r in caplog.records]
