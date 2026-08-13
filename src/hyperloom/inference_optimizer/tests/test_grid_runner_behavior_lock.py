# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Behavior-lock tests for ``run_grid``: pulse matrix, ``keep_going_on_failure``
asymmetry, and auto-warmup teardown timing."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from hyperloom.orchestrator.actions.executors import _grid_runner as gr
from hyperloom.orchestrator.actions.executors import _multi_node_env as mne
from hyperloom.orchestrator.actions.executors import (
    _multi_node_server_lifecycle as mnsl,
)
from hyperloom.orchestrator.actions.executors import _server_lifecycle as sl
from hyperloom.orchestrator.actions.executors._grid_runner import (
    GridVariant,
    run_grid,
)
from hyperloom.orchestrator.trace.task_progress import progress_scope


@pytest.fixture(autouse=True)
def _isolate_leak_root(tmp_path_factory, monkeypatch):
    sandbox = tmp_path_factory.mktemp("isolated_leak_root_behavior_lock")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_LEAK_ROOTS", str(sandbox))
    # Multi-node client warmup is orthogonal noise; keep it off so the mocked
    # ``run_with_session_kill`` call counts are unambiguous.
    monkeypatch.setenv("INFERENCE_OPTIMIZER_MN_BENCH_WARMUP", "0")


@pytest.fixture(autouse=True)
def _single_node_default(monkeypatch):
    monkeypatch.setattr(mne, "is_multi_node", lambda: False)


def _write_base_yaml(path: Path, *, framework: str = "sglang") -> None:
    cfg = {
        "benchmark": {
            "framework": framework,
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


def _valid_workspace(slot: Path, *, tput: float = 800.0) -> Path:
    ws = slot / "benchmark_sglang_20260101_000000"
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "benchmark_report.json").write_text(
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
    return ws


def _invalid_rc0_workspace(slot: Path) -> Path:
    """A benchmark_* workspace whose report yields no valid measurement."""
    ws = slot / "benchmark_sglang_20260101_000000"
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "benchmark_report.json").write_text('{"success": false, "framework": "sglang"}')
    return ws


# ---------------------------------------------------------------------------
# Robustness-pulse matrix
# ---------------------------------------------------------------------------


class TestPulseMatrix:
    """``_pulse_after_variant`` (delegating to ``_robustness_pulse``) fires on
    every failure path EXCEPT the multi-node ``mn_server_restart_failed`` path,
    which returns/continues before the pulse call."""

    def _run_with_pulse_capture(self, *, multi_node, run_side_effect, base, out, restart=None, keep_going=True, grid_n=1):
        pulse_calls: list = []

        async def fake_pulse(**kwargs):
            pulse_calls.append(kwargs)

        with ExitStack() as st:
            st.enter_context(patch.object(mne, "is_multi_node", lambda: multi_node))
            st.enter_context(patch.object(gr, "_robustness_pulse", side_effect=fake_pulse))
            st.enter_context(
                patch(
                    "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
                    side_effect=run_side_effect,
                )
            )
            if restart is not None:
                st.enter_context(patch.object(mnsl, "restart_server_for_round", restart))
            grid = [GridVariant(name=f"c{i}") for i in range(grid_n)]
            results = asyncio.run(
                run_grid(
                    base_yaml_path=base,
                    base_extra_args="",
                    grid=grid,
                    output_root=out,
                    magpie_python=sys.executable,
                    variant_timeout_sec=10,
                    gpu_type="mi300x",
                    keep_going_on_failure=keep_going,
                )
            )
        return results, pulse_calls

    def test_mn_server_restart_failed_does_not_pulse(self, tmp_path, monkeypatch):
        # Warmup must be off so multi-node truly hits the restart path.
        monkeypatch.setenv("INFERENCE_OPTIMIZER_RUN_GRID_WARMUP", "0")
        base = tmp_path / "base.yaml"
        _write_base_yaml(base)

        async def _restart_fail(**_kwargs):
            raise mnsl.ServerRestartFailed("server /health did not return 200")

        results, pulse_calls = self._run_with_pulse_capture(
            multi_node=True,
            run_side_effect=lambda cmd, *a, **k: subprocess.CompletedProcess(cmd, 0, "ok", ""),
            base=base,
            out=tmp_path / "out",
            restart=_restart_fail,
        )
        assert results[0].status == "failed"
        assert results[0].error_class == "mn_server_restart_failed"
        assert pulse_calls == [], "mn_server_restart_failed must NOT trigger a robustness pulse"

    def test_mn_server_restart_failed_no_pulse_even_for_multiple_variants(self, tmp_path, monkeypatch):
        monkeypatch.setenv("INFERENCE_OPTIMIZER_RUN_GRID_WARMUP", "0")
        base = tmp_path / "base.yaml"
        _write_base_yaml(base)

        async def _restart_fail(**_kwargs):
            raise mnsl.ServerRestartFailed("health probe timed out")

        results, pulse_calls = self._run_with_pulse_capture(
            multi_node=True,
            run_side_effect=lambda cmd, *a, **k: subprocess.CompletedProcess(cmd, 0, "ok", ""),
            base=base,
            out=tmp_path / "out",
            restart=_restart_fail,
            grid_n=2,
        )
        assert [r.error_class for r in results] == [
            "mn_server_restart_failed",
            "mn_server_restart_failed",
        ]
        assert pulse_calls == []

    def test_no_benchmark_workspace_failure_pulses(self, tmp_path, monkeypatch):
        monkeypatch.setenv("INFERENCE_OPTIMIZER_RUN_GRID_WARMUP", "0")
        base = tmp_path / "base.yaml"
        _write_base_yaml(base)

        results, pulse_calls = self._run_with_pulse_capture(
            multi_node=False,
            run_side_effect=lambda cmd, *a, **k: subprocess.CompletedProcess(cmd, 1, "stdout", "boom"),
            base=base,
            out=tmp_path / "out",
        )
        assert results[0].error_class == "no_benchmark_workspace"
        assert len(pulse_calls) == 1
        assert pulse_calls[0]["tick_index"] == 0

    def test_yaml_build_error_failure_pulses(self, tmp_path, monkeypatch):
        monkeypatch.setenv("INFERENCE_OPTIMIZER_RUN_GRID_WARMUP", "0")
        base = tmp_path / "base.yaml"
        _write_base_yaml(base)
        pulse_calls: list = []

        async def fake_pulse(**kwargs):
            pulse_calls.append(kwargs)

        with (
            patch.object(gr, "_robustness_pulse", side_effect=fake_pulse),
            patch.object(gr, "_build_variant_yaml", side_effect=ValueError("bad yaml render")),
        ):
            results = asyncio.run(
                run_grid(
                    base_yaml_path=base,
                    base_extra_args="",
                    grid=[GridVariant(name="c0")],
                    output_root=tmp_path / "out",
                    magpie_python=sys.executable,
                    variant_timeout_sec=10,
                    gpu_type="mi300x",
                )
            )
        assert results[0].error_class == "yaml_build_error"
        assert len(pulse_calls) == 1

    def test_capability_unsupported_failure_pulses(self, tmp_path, monkeypatch):
        monkeypatch.setenv("INFERENCE_OPTIMIZER_RUN_GRID_WARMUP", "0")
        base = tmp_path / "base.yaml"
        _write_base_yaml(base)
        pulse_calls: list = []

        async def fake_pulse(**kwargs):
            pulse_calls.append(kwargs)

        # Force the capability fast-fail branch without needing a real vLLM build.
        with (
            patch.object(gr, "_robustness_pulse", side_effect=fake_pulse),
            patch.object(gr, "unsupported_capability_reason", lambda _v: "aiter fusion shared-experts unsupported"),
        ):
            results = asyncio.run(
                run_grid(
                    base_yaml_path=base,
                    base_extra_args="",
                    grid=[GridVariant(name="c0")],
                    output_root=tmp_path / "out",
                    magpie_python=sys.executable,
                    variant_timeout_sec=10,
                    gpu_type="mi300x",
                )
            )
        assert results[0].error_class == "capability_unsupported"
        assert len(pulse_calls) == 1

    def test_succeeded_variant_pulses(self, tmp_path, monkeypatch):
        monkeypatch.setenv("INFERENCE_OPTIMIZER_RUN_GRID_WARMUP", "0")
        base = tmp_path / "base.yaml"
        _write_base_yaml(base)

        def _ok(cmd, *a, **k):
            out_idx = cmd.index("--output-dir")
            _valid_workspace(Path(cmd[out_idx + 1]))
            return subprocess.CompletedProcess(cmd, 0, "ok", "")

        results, pulse_calls = self._run_with_pulse_capture(
            multi_node=False,
            run_side_effect=_ok,
            base=base,
            out=tmp_path / "out",
        )
        assert results[0].status == "succeeded"
        assert len(pulse_calls) == 1


# ---------------------------------------------------------------------------
# keep_going_on_failure asymmetry
# ---------------------------------------------------------------------------


class TestKeepGoingAsymmetry:
    """The break gates are keyed on ``rc != 0``: an ``rc==0`` failure (invalid
    measurement, or no workspace) ALWAYS continues to the next variant even when
    ``keep_going_on_failure=False``; only an ``rc != 0`` failure breaks."""

    def _run(self, run_side_effect, base, out):
        with patch(
            "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
            side_effect=run_side_effect,
        ):
            return asyncio.run(
                run_grid(
                    base_yaml_path=base,
                    base_extra_args="",
                    grid=[GridVariant(name="c0"), GridVariant(name="c1")],
                    output_root=out,
                    magpie_python=sys.executable,
                    variant_timeout_sec=10,
                    gpu_type="mi300x",
                    keep_going_on_failure=False,
                )
            )

    def test_rc0_invalid_measurement_continues_despite_keep_going_false(self, tmp_path, monkeypatch):
        monkeypatch.setenv("INFERENCE_OPTIMIZER_RUN_GRID_WARMUP", "0")
        base = tmp_path / "base.yaml"
        _write_base_yaml(base)

        def _invalid_rc0(cmd, *a, **k):
            out_idx = cmd.index("--output-dir")
            _invalid_rc0_workspace(Path(cmd[out_idx + 1]))
            return subprocess.CompletedProcess(cmd, 0, "ok", "")

        results = self._run(_invalid_rc0, base, tmp_path / "out")
        # Both variants run: rc==0 invalid measurement never breaks the loop.
        assert len(results) == 2
        assert [r.status for r in results] == ["failed", "failed"]
        assert {r.error_class for r in results} == {"benchmark_report_invalid_metric"}
        assert [r.returncode for r in results] == [0, 0]

    def test_rc0_no_workspace_continues_despite_keep_going_false(self, tmp_path, monkeypatch):
        monkeypatch.setenv("INFERENCE_OPTIMIZER_RUN_GRID_WARMUP", "0")
        base = tmp_path / "base.yaml"
        _write_base_yaml(base)

        def _no_ws_rc0(cmd, *a, **k):
            return subprocess.CompletedProcess(cmd, 0, "ok", "")

        results = self._run(_no_ws_rc0, base, tmp_path / "out")
        assert len(results) == 2
        assert {r.error_class for r in results} == {"no_benchmark_workspace"}
        assert [r.returncode for r in results] == [0, 0]

    def test_rc_nonzero_invalid_measurement_breaks(self, tmp_path, monkeypatch):
        """Contrast: an ``rc != 0`` invalid measurement DOES break the loop."""
        monkeypatch.setenv("INFERENCE_OPTIMIZER_RUN_GRID_WARMUP", "0")
        base = tmp_path / "base.yaml"
        _write_base_yaml(base)

        def _invalid_rc1(cmd, *a, **k):
            out_idx = cmd.index("--output-dir")
            _invalid_rc0_workspace(Path(cmd[out_idx + 1]))
            return subprocess.CompletedProcess(cmd, 1, "out", "err")

        results = self._run(_invalid_rc1, base, tmp_path / "out")
        # Only the first variant ran; the loop broke.
        assert len(results) == 1
        assert results[0].error_class == "magpie_nonzero_invalid_measurement"
        assert results[0].returncode == 1


# ---------------------------------------------------------------------------
# auto-warmup teardown timing
# ---------------------------------------------------------------------------


class TestAutoWarmupTeardown:
    """With auto-warmup engaged (single-node, Magpie built-in script,
    ``INFERENCE_OPTIMIZER_RUN_GRID_WARMUP=1``), ``teardown_lifecycle_server`` is
    called exactly once per variant on each of these paths, keyed on the
    resolved lifecycle framework/port."""

    def _run_capture_teardown(self, run_side_effect, base, out):
        teardown_calls: list = []

        def fake_teardown(**kwargs):
            teardown_calls.append(kwargs)

        with (
            patch.object(sl, "teardown_lifecycle_server", side_effect=fake_teardown),
            patch(
                "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
                side_effect=run_side_effect,
            ),
        ):
            results = asyncio.run(
                run_grid(
                    base_yaml_path=base,
                    base_extra_args="",
                    grid=[GridVariant(name="cand")],
                    output_root=out,
                    magpie_python=sys.executable,
                    variant_timeout_sec=10,
                    gpu_type="mi300x",
                )
            )
        return results, teardown_calls

    def test_warmup_success_measured_success_tears_down_once(self, tmp_path, monkeypatch):
        monkeypatch.setenv("INFERENCE_OPTIMIZER_RUN_GRID_WARMUP", "1")
        # Pin the free-port picker so the teardown-port assertion is
        # deterministic (baseline uses a per-session free port).
        monkeypatch.setattr(
            "hyperloom.orchestrator.actions.executors._server_lifecycle._pick_free_port",
            lambda: 8888,
        )
        base = tmp_path / "base.yaml"
        _write_base_yaml(base)
        state = {"n": 0}

        def _run(cmd, *a, **k):
            out_idx = cmd.index("--output-dir")
            slot = Path(cmd[out_idx + 1])
            state["n"] += 1
            _valid_workspace(slot, tput=270.9 if state["n"] == 1 else 4701.6)
            return subprocess.CompletedProcess(cmd, 0, "ok", "")

        results, teardown_calls = self._run_capture_teardown(_run, base, tmp_path / "out")
        assert results[0].status == "succeeded"
        # warmup (round 1) + measured (round 2) both ran.
        assert state["n"] == 2
        # Exactly one teardown: the measured-round ``finally`` block.
        assert len(teardown_calls) == 1
        assert teardown_calls[0]["framework"] == "sglang"
        assert teardown_calls[0]["port"] == 8888
        assert "run_grid_warmup_discarded_first" in results[0].nonfatal_warnings

    def test_warmup_failed_tears_down_once_and_skips_measured_round(self, tmp_path, monkeypatch):
        monkeypatch.setenv("INFERENCE_OPTIMIZER_RUN_GRID_WARMUP", "1")
        base = tmp_path / "base.yaml"
        _write_base_yaml(base)
        state = {"n": 0}

        def _run(cmd, *a, **k):
            state["n"] += 1
            if state["n"] == 1:
                # Warmup round fails (nonzero, no workspace).
                return subprocess.CompletedProcess(
                    cmd,
                    1,
                    "",
                    "warmup boom OPENAI_API_KEY=ak-warmup-secret-value",
                )
            out_idx = cmd.index("--output-dir")
            _valid_workspace(Path(cmd[out_idx + 1]))
            return subprocess.CompletedProcess(cmd, 0, "ok", "")

        results, teardown_calls = self._run_capture_teardown(_run, base, tmp_path / "out")
        assert results[0].status == "failed"
        assert results[0].error_class == "warmup_round_failed"
        assert "warmup-secret-value" not in results[0].error
        assert "[REDACTED]" in results[0].error
        marker_path = next((tmp_path / "out").glob("variant_*/abort_reason.json"))
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        assert "warmup-secret-value" not in marker["error"]
        # The measured round never ran — only the warmup call was made.
        assert state["n"] == 1
        # Exactly one teardown, from the warmup-failure branch.
        assert len(teardown_calls) == 1
        assert teardown_calls[0]["framework"] == "sglang"
        assert "run_grid_warmup_round_failed" in results[0].nonfatal_warnings

    def test_measured_round_timeout_tears_down_once(self, tmp_path, monkeypatch):
        monkeypatch.setenv("INFERENCE_OPTIMIZER_RUN_GRID_WARMUP", "1")
        base = tmp_path / "base.yaml"
        _write_base_yaml(base)
        state = {"n": 0}

        def _run(cmd, *a, **k):
            state["n"] += 1
            if state["n"] == 1:
                out_idx = cmd.index("--output-dir")
                _valid_workspace(Path(cmd[out_idx + 1]), tput=270.9)
                return subprocess.CompletedProcess(cmd, 0, "ok", "")
            raise subprocess.TimeoutExpired(cmd="magpie", timeout=10)

        results, teardown_calls = self._run_capture_teardown(_run, base, tmp_path / "out")
        assert results[0].status == "failed"
        assert results[0].error_class == "magpie_timeout"
        # warmup succeeded, measured round timed out.
        assert state["n"] == 2
        # Exactly one teardown, from the measured-round ``finally`` block.
        assert len(teardown_calls) == 1
        assert teardown_calls[0]["framework"] == "sglang"

    def test_warmup_round_timeout_tears_down_once(self, tmp_path, monkeypatch):
        """Companion path: the warmup round itself times out (its own teardown
        branch, before the measured round is ever reached)."""
        monkeypatch.setenv("INFERENCE_OPTIMIZER_RUN_GRID_WARMUP", "1")
        base = tmp_path / "base.yaml"
        _write_base_yaml(base)
        state = {"n": 0}

        def _run(cmd, *a, **k):
            state["n"] += 1
            raise subprocess.TimeoutExpired(cmd="magpie", timeout=10)

        results, teardown_calls = self._run_capture_teardown(_run, base, tmp_path / "out")
        assert results[0].status == "failed"
        assert results[0].error_class == "warmup_magpie_timeout"
        assert state["n"] == 1
        assert len(teardown_calls) == 1
        assert teardown_calls[0]["framework"] == "sglang"


# ---------------------------------------------------------------------------
# _resolve_mn_effective_server_args: prefer the materialized variant YAML;
# fall back to recomposing from the base YAML when the variant read fails.
# ---------------------------------------------------------------------------
class TestResolveMnEffectiveServerArgs:
    def _base(self, tmp_path: Path, *, args: str = "--tp 8") -> Path:
        base = tmp_path / "base.yaml"
        base.write_text(
            yaml.safe_dump(
                {"benchmark": {"framework": "sglang", "envs": {"EXTRA_SGLANG_ARGS": args}}}
            ),
            encoding="utf-8",
        )
        return base

    def test_prefers_materialized_variant_yaml(self, tmp_path):
        base = self._base(tmp_path)
        cfg = tmp_path / "variant.yaml"
        cfg.write_text(
            yaml.safe_dump(
                {"benchmark": {"framework": "sglang", "envs": {"EXTRA_SGLANG_ARGS": "--tp 8 --chunked-prefill 4096"}}}
            ),
            encoding="utf-8",
        )
        variant = GridVariant(name="v1", extra_server_args="--should-not-be-composed")

        out = gr._resolve_mn_effective_server_args(
            cfg,
            base,
            variant,
            base_extra_args="--base-extra",
            base_args_mode="append",
        )
        # Read verbatim from the variant YAML; the variant/base extras are NOT
        # re-composed on this happy path.
        assert out == "--tp 8 --chunked-prefill 4096"

    def test_variant_env_absent_returns_empty(self, tmp_path):
        base = self._base(tmp_path)
        cfg = tmp_path / "variant.yaml"
        cfg.write_text(yaml.safe_dump({"benchmark": {"framework": "sglang", "envs": {}}}), encoding="utf-8")
        variant = GridVariant(name="v1")

        out = gr._resolve_mn_effective_server_args(
            cfg, base, variant, base_extra_args="", base_args_mode="append"
        )
        assert out == ""

    def test_falls_back_to_base_compose_when_variant_missing(self, tmp_path):
        base = self._base(tmp_path, args="--tp 8")
        missing_cfg = tmp_path / "does_not_exist.yaml"
        variant = GridVariant(name="v1", extra_server_args="--chunked-prefill 2048")

        out = gr._resolve_mn_effective_server_args(
            missing_cfg,
            base,
            variant,
            base_extra_args="--mem-fraction-static 0.9",
            base_args_mode="append",
        )
        # append mode: inherited base args + base_extra + variant extra all survive.
        assert "--tp 8" in out
        assert "--mem-fraction-static 0.9" in out
        assert "--chunked-prefill 2048" in out

    def test_fallback_replace_mode_drops_inherited_base_args(self, tmp_path):
        base = self._base(tmp_path, args="--tp 8")
        missing_cfg = tmp_path / "does_not_exist.yaml"
        variant = GridVariant(name="v1", extra_server_args="--chunked-prefill 2048")

        out = gr._resolve_mn_effective_server_args(
            missing_cfg,
            base,
            variant,
            base_extra_args="",
            base_args_mode="replace",
        )
        # replace mode drops the inherited base env args.
        assert "--tp 8" not in out
        assert "--chunked-prefill 2048" in out


# ---------------------------------------------------------------------------
# Per-variant progress heartbeat
# ---------------------------------------------------------------------------


class TestVariantHeartbeat:
    """A grid that runs for hours must be distinguishable from one that hung."""

    def _run_capture_progress(self, run_side_effect, base, out, *, grid_n=2, notes=None):
        notes = [] if notes is None else notes

        async def _sink(**note):
            notes.append(note)

        async def _no_pulse(**_kwargs):
            return None

        with (
            progress_scope(_sink),
            patch.object(gr, "_robustness_pulse", side_effect=_no_pulse),
            patch(
                "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
                side_effect=run_side_effect,
            ),
        ):
            results = asyncio.run(
                run_grid(
                    base_yaml_path=base,
                    base_extra_args="",
                    grid=[GridVariant(name=f"c{i}") for i in range(grid_n)],
                    output_root=out,
                    magpie_python=sys.executable,
                    variant_timeout_sec=10,
                    gpu_type="mi300x",
                )
            )
        return results, notes

    def test_each_variant_reports_as_it_lands(self, tmp_path, monkeypatch):
        monkeypatch.setenv("INFERENCE_OPTIMIZER_RUN_GRID_WARMUP", "0")
        base = tmp_path / "base.yaml"
        _write_base_yaml(base)

        def _ok(cmd, *a, **k):
            out_idx = cmd.index("--output-dir")
            _valid_workspace(Path(cmd[out_idx + 1]))
            return subprocess.CompletedProcess(cmd, 0, "ok", "")

        results, notes = self._run_capture_progress(_ok, base, tmp_path / "out")

        landed = [n for n in notes if n["unit"] == "variant"]
        assert [r.status for r in results] == ["succeeded", "succeeded"]
        assert [(n["label"], n["index"], n["total"]) for n in landed] == [("c0", 1, 2), ("c1", 2, 2)]
        assert all(n["status"] == "succeeded" for n in landed)
        assert landed[0]["output_throughput"] == 800.0

    def test_a_failed_variant_reports_too(self, tmp_path, monkeypatch):
        """Progress means "a unit finished", not "a unit worked"."""
        monkeypatch.setenv("INFERENCE_OPTIMIZER_RUN_GRID_WARMUP", "0")
        base = tmp_path / "base.yaml"
        _write_base_yaml(base)

        _, notes = self._run_capture_progress(
            lambda cmd, *a, **k: subprocess.CompletedProcess(cmd, 1, "stdout", "boom"),
            base,
            tmp_path / "out",
            grid_n=1,
        )

        assert [n["status"] for n in notes if n["unit"] == "variant"] == ["failed"]

    def test_a_variant_reports_before_it_blocks(self, tmp_path, monkeypatch):
        """A first variant that hangs inside the benchmark used to emit nothing at all."""
        monkeypatch.setenv("INFERENCE_OPTIMIZER_RUN_GRID_WARMUP", "0")
        base = tmp_path / "base.yaml"
        _write_base_yaml(base)
        notes: list[dict] = []
        at_launch: list[dict] = []

        def _capture_then_fail(cmd, *a, **k):
            at_launch.extend(notes)
            return subprocess.CompletedProcess(cmd, 1, "", "boom")

        self._run_capture_progress(
            _capture_then_fail,
            base,
            tmp_path / "out",
            grid_n=1,
            notes=notes,
        )

        assert [(n["label"], n["status"]) for n in at_launch] == [
            ("c0:variant", "started"),
            ("c0:benchmark", "started"),
        ]
