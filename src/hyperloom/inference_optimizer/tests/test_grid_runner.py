# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Consolidated tests for ``orchestrator.actions.executors._grid_runner``."""

from __future__ import annotations

import inspect
import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

from hyperloom.orchestrator.actions.executors import _grid_runner
from hyperloom.orchestrator.actions.executors import _grid_runner as gr
from hyperloom.inference_optimizer.grid_server_args import (
    compose_server_args,
    dedup_vllm_server_args,
)
from hyperloom.orchestrator.actions.executors._benchmark_runtime import apply_runtime_benchmark_overrides
from hyperloom.orchestrator.actions.executors._grid_variant_filter import (
    _parse_skip_spec,
    apply_user_skip_list,
    reorder_grid_for_multi_node,
    resolve_skip_spec,
)
from hyperloom.orchestrator.actions.executors._grid_runner import (
    GridVariant,
    VariantResult,
    _build_variant_yaml,
    GridRecipe,
)

from hyperloom.orchestrator.actions.executors._workload_envs import BenchmarkTarget, LaunchLayer


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
        error_class="round_timeout",
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
    from hyperloom.orchestrator.state import failure_evidence

    src = inspect.getsource(_grid_runner) + inspect.getsource(failure_evidence)
    expected = {
        "yaml_build_error",
        "mn_server_restart_failed",
        "no_benchmark_workspace",
        "round_nonzero_invalid_measurement",
        "benchmark_report_missing",
        "benchmark_report_invalid_metric",
        "round_nonzero_after_valid_measurement",
    }
    missing = [label for label in expected if f'"{label}"' not in src]
    assert not missing, f"missing error_class labels in run_grid: {missing}"


def test_grid_evidence_skips_prelaunch_failure(tmp_path: Path) -> None:
    """A refused launch has an abort marker, not a measurement evidence record."""
    output_root = tmp_path / "runs"
    result = _grid_runner.VariantResult(
        name="unsupported",
        extra_server_args="--unsupported",
        extra_envs={},
        status="failed",
        error_class="capability_unsupported",
    )

    _grid_runner._attach_grid_launch_evidence(
        [result],
        grid=[_grid_runner.GridVariant("unsupported")],
        output_root=output_root,
        caller_reused_ready_server=False,
    )

    assert result.launch_evidence == {}
    assert not (output_root / "variant_00_unsupported" / "launch_evidence.json").exists()
    assert not (output_root / "variant_00_unsupported" / "runtime_findings.json").exists()


def test_grid_evidence_writes_runtime_findings_next_to_launch_evidence(tmp_path: Path) -> None:
    output_root = tmp_path / "runs"
    slot = output_root / "variant_00_v"
    slot.mkdir(parents=True)
    (slot / "config.yaml").write_text(
        yaml.safe_dump({"benchmark": {"framework": "vllm", "envs": {"VLLM_FOO": "1", "VLLM_REPO_PATH": "/opt/vllm"}}}),
        encoding="utf-8",
    )
    (slot / "server.log").write_text(
        "WARNING [interface.py:1461] Unknown vLLM environment variable detected: VLLM_FOO\n"
        "WARNING [interface.py:1461] Unknown vLLM environment variable detected: VLLM_PYTHON\n"
        "WARNING [interface.py:1461] Unknown vLLM environment variable detected: VLLM_REPO_PATH\n",
        encoding="utf-8",
    )
    result = _grid_runner.VariantResult(name="v", extra_server_args="", extra_envs={}, status="succeeded")

    _grid_runner._attach_grid_launch_evidence(
        [result],
        grid=[_grid_runner.GridVariant("v")],
        output_root=output_root,
        caller_reused_ready_server=False,
    )

    report = json.loads((slot / "runtime_findings.json").read_text(encoding="utf-8"))
    assert report["log_path"] == str(slot / "server.log")
    assert [(f["rule_id"], f["subject"]) for f in report["findings"] if f["status"] == "detected"] == [
        ("vllm.unknown_env", "VLLM_FOO")
    ]


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
        # Ordered so a real reorder would change it: low-priority backend first, high-priority param last, untagged in
        # the middle.
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
        out = reorder_grid_for_multi_node(grid)
        assert [v.name for v in out] == [v.name for v in grid]

    def test_multi_node_surfaces_likely_winners_first(self, monkeypatch):
        from hyperloom.orchestrator.actions.executors import (
            _multi_node_env as mne,
        )

        monkeypatch.setattr(mne, "is_multi_node", lambda: True)
        grid = self._grid()
        out = reorder_grid_for_multi_node(grid)
        # cuda_graph_max_bs (params tier-1) sorts ahead of tier5_comm; the untagged variant sinks to the end (stable
        # sort).
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
    def test_preserves_measurement_fields(self):
        from dataclasses import asdict

        result = VariantResult(
            name="agentx",
            extra_server_args="--kv-cache-dtype fp8",
            extra_envs={"CONC": "4"},
            status="succeeded",
            output_throughput=100.0,
            input_throughput=900.0,
            total_token_throughput=1000.0,
            intvty_p90=80.0,
            tpot_p90_ms=12.5,
            completed_requests=20,
            duration_seconds=3600.0,
            workspace="/runs/benchmark",
            raw_result_path="/runs/benchmark/inferencex_result.json",
            launch_evidence={"identity": "measured-server"},
        )

        encoded = result.to_dict()
        expected = asdict(result)
        expected["e2e_norm_intvty_p90"] = expected.pop("intvty_p90")
        expected["e2e_norm_intvty_p50"] = expected.pop("intvty_p50")
        expected["e2e_norm_intvty_p50_1800s"] = expected.pop("intvty_p50_checkpoint")
        assert encoded == expected

    def test_preserves_unmeasured_axes(self):
        result = VariantResult(name="legacy", extra_server_args="", extra_envs={}, status="failed")
        encoded = result.to_dict()
        for key in (
            "input_throughput",
            "total_token_throughput",
            "e2e_norm_intvty_p90",
            "e2e_norm_intvty_p50",
            "tpot_p90_ms",
        ):
            assert key in encoded
            assert encoded[key] is None

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


class TestGridVariantEnvSafety:
    """Lock the env-safety filter on LLM-supplied grid env overrides."""

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


def test_apply_runtime_overrides_pins_the_explicit_benchmark_script():
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
    assert "runner_type" not in bench


@pytest.mark.parametrize("script", [None, "vllm_custom.sh"])
def test_apply_runtime_overrides_derives_nothing_from_the_board(script):
    bench = {"framework": "vllm", "envs": {}, **({"benchmark_script": script} if script else {})}
    apply_runtime_benchmark_overrides(bench, gpu_type="r9700")
    assert "runner_type" not in bench
    assert bench.get("benchmark_script") == script


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
        GridRecipe(base_yaml_path=base, target=BenchmarkTarget(gpu_type="mi300x", benchmark_script="sglang_mi300x.sh")),
        GridVariant("vA", "--attention-backend aiter"),
        output_subdir=tmp_path / "vA",
    )
    cfg = yaml.safe_load(out.read_text())
    assert cfg["benchmark"]["benchmark_script"] == "sglang_mi300x.sh"
    assert "runner_type" not in cfg["benchmark"]


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
        GridRecipe(base_yaml_path=base, base=LaunchLayer(extra_server_args="--enable-bar --block-size 128")),
        GridVariant(
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


def test_build_variant_yaml_keeps_a_value_the_replacing_variant_re_adds(tmp_path):
    """A removal scoped to the base's value must not take the variant's own copy of that flag with it."""
    base = tmp_path / "base.yaml"
    _write_baseline_yaml_overrides(base)

    out = _build_variant_yaml(
        GridRecipe(base_yaml_path=base, base=LaunchLayer(extra_server_args="--cuda-graph-max-bs 8 --tp 8")),
        GridVariant(
            "bigger_graphs",
            "--cuda-graph-max-bs 16 --tp 8",
            remove_args=["--cuda-graph-max-bs 8"],
            args_mode="replace",
        ),
        output_subdir=tmp_path / "bigger_graphs",
    )

    args = yaml.safe_load(out.read_text())["benchmark"]["envs"]["EXTRA_SGLANG_ARGS"]
    assert "--cuda-graph-max-bs 16" in args
    assert "--cuda-graph-max-bs 8" not in args


def test_agentx_removals_reach_the_launch_as_bare_names_without_losing_the_re_added_value():
    """An AgentX launch takes one bare option name per removal; only that boundary drops the value a spec is scoped to."""
    from hyperloom.orchestrator.actions.executors.agentx_point import LaunchChanges, apply_launch_changes

    bench = {"framework": "sglang", "agentx": {"enabled": True}}
    apply_launch_changes(
        bench,
        LaunchChanges(
            args="--cuda-graph-max-bs 16 --tp 8",
            remove_args=["--cuda-graph-max-bs 8", "--disable-cuda-graph --enable-dp-attention"],
            replace_args=True,
        ),
    )

    launch = bench["agentx"]["launch_overrides"]
    # The launch matches these against the recipe before our append_args, so the re-added value survives the removal.
    assert launch["remove_args"] == ["--cuda-graph-max-bs", "--disable-cuda-graph", "--enable-dp-attention"]
    argv = launch["append_args"]
    assert argv[argv.index("--cuda-graph-max-bs") + 1] == "16"


def test_build_variant_yaml_refuses_to_unset_pinned_envs(tmp_path):
    """A variant that unsets TP would silently shrink the Ray lease to one GPU."""
    base = tmp_path / "base.yaml"
    _write_baseline_yaml_overrides(base)
    cfg = yaml.safe_load(base.read_text())
    cfg["benchmark"]["envs"].update({"TP": 8, "CONC": 64, "SGLANG_ENABLE_FOO": "1"})
    base.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    out = _build_variant_yaml(
        GridRecipe(base_yaml_path=base),
        GridVariant(
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


def test_an_exported_setting_with_an_overlay_cannot_be_reimported(tmp_path):
    from hyperloom.orchestrator.state.shared_state import SharedState
    from inference_suite import UserFileError, parse_user_file

    overlay = tmp_path / "overlay ' literal"
    overlay.mkdir()
    (overlay / "sitecustomize.py").write_text("")
    session = tmp_path / "session"
    session.mkdir()
    state = SharedState(framework="sglang")
    state.current_best = {"extra_server_args": "--max-running-requests 4", "final_overlay": str(overlay)}
    state.save(session)

    with pytest.raises(UserFileError, match="hyperloom-launch-controls"):
        parse_user_file("current_setting.sh", (session / "current_setting.sh").read_bytes(), framework="sglang")


# Section: dedup_vllm_server_args  — vLLM/atom single-value flag collapse


class TestDedupVllmServerArgs:
    """``dedup_vllm_server_args`` collapses repeated vLLM single-value flags."""

    def test_duplicate_attention_backend_keeps_last(self):
        # YAML base + variant both inject the flag.
        out = dedup_vllm_server_args(
            "--attention-backend ROCM_AITER_FA --attention-backend ROCM_FLASH",
            "vllm",
        )
        assert out == "--attention-backend ROCM_FLASH"
        assert out.count("--attention-backend") == 1

    def test_identical_duplicate_collapses_to_single(self):
        out = dedup_vllm_server_args(
            "--attention-backend ROCM_AITER_FA --attention-backend ROCM_AITER_FA",
            "vllm",
        )
        assert out == "--attention-backend ROCM_AITER_FA"

    def test_preserves_surrounding_and_unknown_flags_in_order(self):
        out = dedup_vllm_server_args(
            "--enforce-eager --attention-backend A --max-model-len 4096 --attention-backend B --trust-remote-code",
            "vllm",
        )
        # Earlier --attention-backend span removed; everything else stays ordered.
        assert out == ("--enforce-eager --max-model-len 4096 --attention-backend B --trust-remote-code")

    def test_json_config_flag_quotes_survive_dedup(self):
        # Regression: a variant that duplicates a single-value flag (here --block-size) used to force the
        # shlex-split/rejoin branch, which STRIPPED the inner double quotes of a compact --compilation-config JSON
        # value (``{"cudagraph_mode":"PIECEWISE"}`` -> ``{cudagraph_mode: PIECEWISE}``) and crashed every
        # explore/kernel/integrate variant server with ``Invalid JSON``.
        raw = (
            '--compilation-config {"cudagraph_mode":"PIECEWISE"} '
            "--block-size 128 --block-size 128 --gpu-memory-utilization 0.95"
        )
        out = dedup_vllm_server_args(raw, "vllm")
        assert '{"cudagraph_mode":"PIECEWISE"}' in out
        assert "{cudagraph_mode:PIECEWISE}" not in out
        assert out.count("--block-size") == 1

    def test_speculative_config_quotes_survive_dedup(self):
        raw = '--speculative-config {"method":"eagle"} --max-num-seqs 256 --max-num-seqs 256'
        out = dedup_vllm_server_args(raw, "vllm")
        assert '{"method":"eagle"}' in out
        assert out.count("--max-num-seqs") == 1

    def test_equals_form_is_deduped(self):
        out = dedup_vllm_server_args(
            "--gpu-memory-utilization=0.9 --gpu-memory-utilization=0.85",
            "vllm",
        )
        assert out == "--gpu-memory-utilization=0.85"

    def test_atom_framework_also_deduped(self):
        out = dedup_vllm_server_args(
            "--attention-backend A --attention-backend B",
            "atom",
        )
        assert out == "--attention-backend B"

    def test_sglang_is_noop_repeats_preserved(self):
        # sglang tolerates repeats (last-wins at the server); do not mangle.
        raw = "--attention-backend aiter --attention-backend triton"
        assert dedup_vllm_server_args(raw, "sglang") == raw

    def test_no_duplicates_returns_unchanged(self):
        raw = "--attention-backend ROCM_AITER_FA --max-model-len 8192"
        assert dedup_vllm_server_args(raw, "vllm") == raw

    def test_empty_and_none_safe(self):
        assert dedup_vllm_server_args("", "vllm") == ""
        assert dedup_vllm_server_args(None, "vllm") == ""

    def test_unparseable_string_returned_as_is(self):
        # Unbalanced quote: leave it for vLLM to report rather than mangling.
        raw = "--attention-backend 'unterminated"
        assert dedup_vllm_server_args(raw, "vllm") == raw.strip()

    def test_distinct_single_value_flags_untouched(self):
        raw = "--max-num-seqs 256 --block-size 16 --kv-cache-dtype fp8"
        assert dedup_vllm_server_args(raw, "vllm") == raw

    def test_mixed_equals_and_space_forms_dedup_last_wins(self):
        # `--flag value` (YAML base) then `--flag=value` (variant): last wins.
        out = dedup_vllm_server_args(
            "--attention-backend ROCM_AITER_FA --attention-backend=ROCM_FLASH",
            "vllm",
        )
        assert out == "--attention-backend=ROCM_FLASH"

    def test_mixed_equals_then_space_form_dedup_last_wins(self):
        # Reverse order: `--flag=value` then `--flag value`.
        out = dedup_vllm_server_args(
            "--attention-backend=ROCM_AITER_FA --attention-backend ROCM_FLASH",
            "vllm",
        )
        assert out == "--attention-backend ROCM_FLASH"

    def test_three_occurrences_keep_only_last(self):
        out = dedup_vllm_server_args(
            "--attention-backend A --attention-backend B --attention-backend C",
            "vllm",
        )
        assert out == "--attention-backend C"

    def test_json_flag_does_not_disable_other_flag_dedup(self):
        raw = "--attention-backend A --attention-backend B --override-generation-config '{\"temperature\": 0.7}'"
        out = dedup_vllm_server_args(raw, "vllm")
        assert out == '--attention-backend B --override-generation-config {"temperature":0.7}'

    def test_json_string_with_internal_space_fails_closed(self):
        raw = '--attention-backend A --attention-backend B --speculative-config \'{"model":"draft model"}\''
        assert dedup_vllm_server_args(raw, "vllm") == raw

    def test_quoted_non_json_operand_fragments_fail_closed(self):
        raw = '--tool-call-parser "my parser" --max-num-seqs 512 --max-num-seqs 1024'
        assert dedup_vllm_server_args(raw, "vllm") == raw

    def test_multi_value_flag_left_untouched(self):
        # cuda-graph-bs takes a list; never collapse a string that carries one.
        raw = "--attention-backend A --cuda-graph-bs 1 2 4 8 --attention-backend B"
        assert dedup_vllm_server_args(raw, "vllm") == raw


# Every output slot one variant launches a benchmark process into, in launch order: the discarded warmup, the
# multi-node client warmup, and the measured round.


class TestSessionGridBounds:
    """One definition of the two numbers every benching arm needs."""

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
        """What a variant actually spends, rather than what the baseline spent."""
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


def test_compose_preserves_json_for_every_args_mode():
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
        out = compose_server_args(**kwargs)
        assert _blob(out, "--compilation-config") == json.loads(cc)
        assert _blob(out, "--speculative-config") == json.loads(sc)
    # Removing the JSON flag and supplying a variant replacement.
    repl = '{"mode":3,"custom_ops":["-rms_norm"]}'
    out = compose_server_args(
        inherited_args=base,
        remove_args=["--compilation-config"],
        variant_extra_args=f"--compilation-config {repl}",
    )
    assert out.count("--compilation-config") == 1
    assert _blob(out, "--compilation-config") == json.loads(repl)
    assert _blob(out, "--speculative-config") == json.loads(sc)


# benchmark_report settling (real-run race)


@pytest.mark.asyncio
async def test_report_read_waits_for_a_report_still_being_written(tmp_path):
    """A report that lands a moment after the process exits must not abort the variant."""
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
    """Every variant's args go through the JSON-preserving tokenizer, AgentX or not, and normalizing them drops no flag."""

    def _off(self, monkeypatch):
        monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)

    def test_a_plain_synthetic_arg_string_round_trips(self, monkeypatch):
        self._off(monkeypatch)
        args = "--tensor-parallel-size 8 --gpu-memory-utilization 0.9 --max-num-seqs 512"
        assert compose_server_args(base_extra_args=args) == args

    def test_the_harness_flag_reaches_the_launch(self, monkeypatch):
        """The baseline keeps the operator's flags, so a variant must launch with them too."""
        self._off(monkeypatch)
        out = compose_server_args(base_extra_args="--no-enable-prefix-caching --max-num-seqs 512")
        assert out == "--no-enable-prefix-caching --max-num-seqs 512"

    def test_quoted_operands_do_not_keep_their_wrappers(self, monkeypatch):
        """EXTRA_*_ARGS travel unquoted, so a wrapper reaches argv literally."""
        self._off(monkeypatch)
        out = compose_server_args(base_extra_args="--quantization 'fp8' --max-num-seqs 512")
        assert out == "--quantization fp8 --max-num-seqs 512"


def test_the_json_tripwire_sees_damage_from_the_removal_pass(caplog):
    """The window must cover ``remove_server_args``, which is what it is about."""
    from hyperloom.inference_optimizer import grid_server_args as gsa
    from inference_suite import argv

    real = argv.remove_server_args

    def _lossy(server_args, remove_args):
        out = real(server_args, remove_args)
        # A wildcard is damage normalization cannot repair, unlike a bareword key.
        return out.replace('{"mode":3}', '{"ignore":[*.mlp.gate]}')

    args = '--compilation-config {"mode":3} --max-num-seqs 512'
    with patch.object(argv, "remove_server_args", side_effect=_lossy):
        with caplog.at_level("ERROR"):
            gsa.compose_server_args(base_extra_args=args, remove_args=["--max-num-seqs"])
    assert any("CORRUPTED" in r.getMessage() for r in caplog.records), [r.getMessage() for r in caplog.records]


# --- graded_objective field in VariantResult and explore decision records ---


def test_variant_result_has_intvty_and_total_fields():
    """VariantResult must carry both graded axes so the explore executor can stamp them."""
    r = VariantResult(
        name="v",
        extra_server_args="",
        extra_envs={},
        status="succeeded",
        output_throughput=183.0,
        input_throughput=25800.0,
        total_token_throughput=25983.0,
        intvty_p90=447.2,
    )
    assert r.total_token_throughput == 25983.0
    assert r.intvty_p90 == 447.2


def test_variant_result_graded_axes_default_to_none():
    r = VariantResult(name="v", extra_server_args="", extra_envs={}, status="succeeded", output_throughput=100.0)
    assert r.total_token_throughput is None
    assert r.intvty_p90 is None


def test_the_total_axis_has_exactly_one_attribute():
    """One measured quantity, one name: `total_throughput` was a second alias for it."""
    r = VariantResult(name="v", extra_server_args="", extra_envs={}, status="succeeded")
    assert not hasattr(r, "total_throughput")
