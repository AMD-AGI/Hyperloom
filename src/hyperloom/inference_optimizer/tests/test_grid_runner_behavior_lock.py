# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Behavior-lock tests for ``run_grid``: variant-boundary progress, ``keep_going_on_failure`` asymmetry, and
auto-warmup teardown timing.
"""

from __future__ import annotations

import asyncio
import json
import signal
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from hyperloom.common.agentx_workload import AGENTX_SERVER_PIDFILE
from hyperloom.orchestrator.actions.executors import _grid_runner as gr
from hyperloom.orchestrator.actions.executors import _multi_node_env as mne
from hyperloom.orchestrator.actions.executors import (
    _multi_node_server_lifecycle as mnsl,
)
from hyperloom.orchestrator.actions.executors import _server_lifecycle as sl
from hyperloom.orchestrator.actions.executors._grid_runner import (
    GridVariant,
    run_grid,
    GridLaunch,
    GridRecipe,
    GridStop,
)
from hyperloom.orchestrator.actions.executors._subprocess_kill import ORCHESTRATOR_CANCELLED_RETURNCODE
from hyperloom.orchestrator.actions.stop_attribution import ORCHESTRATOR_CANCELLED_CLASS, STOPPED_BY_THE_RUN
from hyperloom.inference_optimizer.trace.task_progress import progress_scope

from .conftest import chatty_child
from hyperloom.orchestrator.actions.executors._workload_envs import BenchmarkTarget


@pytest.fixture(autouse=True)
def _isolate_leak_root(tmp_path_factory, monkeypatch):
    sandbox = tmp_path_factory.mktemp("isolated_leak_root_behavior_lock")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_LEAK_ROOTS", str(sandbox))
    # Multi-node client warmup is orthogonal noise; keep it off so the mocked ``run_with_session_kill`` call counts
    # are unambiguous.
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
            "benchmark_script": f"{framework}_mi300x.sh",
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


# Variant-boundary progress matrix


def _run_capturing_variant_notes(
    *,
    multi_node,
    run_side_effect,
    base,
    out,
    restart=None,
    grid_n=1,
    notes=None,
):
    async def collect(**note):
        notes.append(note)

    with ExitStack() as st:
        st.enter_context(patch.object(mne, "is_multi_node", lambda: multi_node))
        st.enter_context(
            patch(
                "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
                side_effect=run_side_effect,
            )
        )
        if restart is not None:
            st.enter_context(patch.object(mnsl, "restart_server_for_round", restart))
        if notes is not None:
            st.enter_context(progress_scope(collect))
        grid = [GridVariant(name=f"c{i}") for i in range(grid_n)]
        results = asyncio.run(
            run_grid(
                grid=grid,
                output_root=out,
                recipe=GridRecipe(base_yaml_path=base, target=BenchmarkTarget(gpu_type="mi300x")),
                launch=GridLaunch(magpie_python=sys.executable),
                stop=GridStop(keep_going_on_failure=True),
            )
        )
    return results, notes


class TestVariantBoundaryReportsEveryOutcome:
    """``_report_finished_variant`` fires on every variant outcome, including the multi-node ``mn_server_restart_failed`` path that used to leave before reaching it."""

    def test_mn_server_restart_failed_reaches_the_variant_boundary(self, tmp_path, monkeypatch):
        """A variant whose remote server never came back still ends its own row."""
        # Warmup must be off so multi-node truly hits the restart path.
        monkeypatch.setenv("INFERENCE_OPTIMIZER_RUN_GRID_WARMUP", "0")
        base = tmp_path / "base.yaml"
        _write_base_yaml(base)
        notes: list[dict] = []

        async def _restart_fail(_server, **_kwargs):
            raise mnsl.ServerRestartFailed("server /health did not return 200")

        results, _ = _run_capturing_variant_notes(
            multi_node=True,
            run_side_effect=lambda cmd, *a, **k: subprocess.CompletedProcess(cmd, 0, "ok", ""),
            base=base,
            out=tmp_path / "out",
            restart=_restart_fail,
            notes=notes,
        )
        assert results[0].status == "failed"
        assert results[0].error_class == "mn_server_restart_failed"
        landed = [(n["label"], n["index"], n["status"]) for n in notes if n["unit"] == "variant"]
        assert landed == [("c0", 1, "failed")]

    def test_mn_server_restart_failed_reports_each_variant_it_ends(self, tmp_path, monkeypatch):
        monkeypatch.setenv("INFERENCE_OPTIMIZER_RUN_GRID_WARMUP", "0")
        base = tmp_path / "base.yaml"
        _write_base_yaml(base)
        notes: list[dict] = []

        async def _restart_fail(_server, **_kwargs):
            raise mnsl.ServerRestartFailed("health probe timed out")

        results, _ = _run_capturing_variant_notes(
            multi_node=True,
            run_side_effect=lambda cmd, *a, **k: subprocess.CompletedProcess(cmd, 0, "ok", ""),
            base=base,
            out=tmp_path / "out",
            restart=_restart_fail,
            grid_n=2,
            notes=notes,
        )
        assert [r.error_class for r in results] == [
            "mn_server_restart_failed",
            "mn_server_restart_failed",
        ]
        # Each variant is named by its own row, not by the tail of the batch.
        landed = [(n["label"], n["index"]) for n in notes if n["unit"] == "variant"]
        assert landed == [("c0", 1), ("c1", 2)]


def _agentx_workspace(slot: Path, **report: object) -> Path:
    """A ``benchmark_*`` workspace whose report is Magpie's native AgentX report, which is ``diagnostic_only`` when the
    round profiles."""
    ws = slot / "benchmark_sglang_20260101_000000"
    ws.mkdir(parents=True, exist_ok=True)
    metrics = {
        "execution_owner": "magpie",
        "server_launch": {"base_argv": ["python3"], "effective_argv": ["python3"], "effective_env": {}},
        "recipe": {"tp": 1, "ep": 1, "conc": 8},
        "throughput": {
            "output_tokens_per_second": 280.0,
            "total_tokens_per_second": 25000.0,
            "duration_seconds": 3600.0,
        },
        "requests": {"successful": 1029, "total": 1072},
        **report.pop("metrics", {}),
    }
    document = {"scenario": "agentx", "success": True, "framework": "sglang", "agentx_metrics": metrics, **report}
    (ws / "benchmark_report.json").write_text(json.dumps(document))
    return ws


class TestADiagnosticAgentxRoundIsAVerdict:
    """Magpie marks a profiled AgentX replay ``diagnostic_only`` and not publishable: it is no valid measurement, and
    the round it ends is still a success, which ``round_succeeded`` states (the main flow's ``ProfileExecutor`` reads
    the same field)."""

    def _run(self, tmp_path, monkeypatch, **report):
        monkeypatch.setattr(gr, "REPORT_SETTLE_SECONDS", 0.0)
        monkeypatch.setenv("INFERENCE_OPTIMIZER_RUN_GRID_WARMUP", "0")
        base = tmp_path / "base.yaml"
        _write_base_yaml(base)

        def _magpie(cmd, *a, **k):
            _agentx_workspace(Path(cmd[cmd.index("--output-dir") + 1]), **report)
            return subprocess.CompletedProcess(cmd, 0, "ok", "")

        results, _ = _run_capturing_variant_notes(
            multi_node=False, run_side_effect=_magpie, base=base, out=tmp_path / "out"
        )
        return results[0]

    def test_a_profiled_replay_succeeds_without_being_a_valid_measurement(self, tmp_path, monkeypatch):
        result = self._run(
            tmp_path,
            monkeypatch,
            benchmark_valid=False,
            publishable=False,
            metrics={"diagnostic_only": True, "profile_capture": {"status": "complete"}},
        )

        assert (result.status, result.error_class) == ("succeeded", "")
        assert result.diagnostic_only is True and result.benchmark_valid is False

    def test_a_replay_that_is_neither_valid_nor_diagnostic_still_fails(self, tmp_path, monkeypatch):
        result = self._run(tmp_path, monkeypatch, benchmark_valid=False, publishable=False)

        assert (result.status, result.error_class) == ("failed", "benchmark_report_invalid_metric")

    def test_a_valid_replay_succeeds(self, tmp_path, monkeypatch):
        result = self._run(tmp_path, monkeypatch, benchmark_valid=True, publishable=True)

        assert result.status == "succeeded" and result.benchmark_valid is True and result.diagnostic_only is False


# keep_going_on_failure asymmetry


class TestKeepGoingAsymmetry:
    """The break gates are keyed on ``rc != 0``: an ``rc==0`` failure (invalid measurement, or no workspace) ALWAYS continues to the next variant even when ``keep_going_on_failure=False``; only an ``rc != 0`` failure breaks."""

    def _run(self, run_side_effect, base, out):
        with patch(
            "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
            side_effect=run_side_effect,
        ):
            return asyncio.run(
                run_grid(
                    grid=[GridVariant(name="c0"), GridVariant(name="c1")],
                    output_root=out,
                    recipe=GridRecipe(base_yaml_path=base, target=BenchmarkTarget(gpu_type="mi300x")),
                    launch=GridLaunch(magpie_python=sys.executable),
                    stop=GridStop(keep_going_on_failure=False),
                )
            )

    def test_rc0_invalid_measurement_continues_despite_keep_going_false(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gr, "REPORT_SETTLE_SECONDS", 0.0)
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
        monkeypatch.setattr(gr, "REPORT_SETTLE_SECONDS", 0.0)
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

    def test_rc_nonzero_blank_pipe_uses_report_errors(self, tmp_path, monkeypatch):
        """Last-resort: empty pipe and no log files, diagnostic only in report.errors."""
        monkeypatch.setattr(gr, "REPORT_SETTLE_SECONDS", 0.0)
        monkeypatch.setenv("INFERENCE_OPTIMIZER_RUN_GRID_WARMUP", "0")
        base = tmp_path / "base.yaml"
        _write_base_yaml(base)

        def _blank_rc2(cmd, *a, **k):
            out_idx = cmd.index("--output-dir")
            slot = Path(cmd[out_idx + 1])
            ws = slot / "benchmark_sglang_20260101_000000"
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "benchmark_report.json").write_text(
                json.dumps(
                    {
                        "success": False,
                        "framework": "sglang",
                        "errors": ["scriptable benchmark script not found for custom_mi355x.sh"],
                    }
                )
            )
            return subprocess.CompletedProcess(cmd, 2, "", "")

        results = self._run(_blank_rc2, base, tmp_path / "out")
        assert len(results) == 1
        assert results[0].error_class == "magpie_nonzero_invalid_measurement"
        assert "custom_mi355x.sh" in (results[0].error or "")
        marker_path = next((tmp_path / "out").glob("variant_*/abort_reason.json"))
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        assert "custom_mi355x.sh" in marker["error"]


# aborting a variant mid-grid


class TestVariantAbort:
    """A variant that never reaches the benchmark still records a result, and ``keep_going_on_failure`` decides
    whether the grid walks on from it.
    """

    def _run(self, *, keep_going, tmp_path, monkeypatch):
        monkeypatch.setenv("INFERENCE_OPTIMIZER_RUN_GRID_WARMUP", "0")
        base = tmp_path / "base.yaml"
        _write_base_yaml(base)
        real_build = gr._build_variant_yaml

        def _fail_on_second(recipe, variant, **kwargs):
            if variant.name == "c1":
                raise RuntimeError("boom")
            return real_build(recipe, variant, **kwargs)

        monkeypatch.setattr(gr, "_build_variant_yaml", _fail_on_second)

        def _no_ws_rc0(cmd, *a, **k):
            return subprocess.CompletedProcess(cmd, 0, "ok", "")

        with patch(
            "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
            side_effect=_no_ws_rc0,
        ):
            return asyncio.run(
                run_grid(
                    grid=[GridVariant(name="c0"), GridVariant(name="c1"), GridVariant(name="c2")],
                    output_root=tmp_path / "out",
                    recipe=GridRecipe(base_yaml_path=base, target=BenchmarkTarget(gpu_type="mi300x")),
                    launch=GridLaunch(magpie_python=sys.executable),
                    stop=GridStop(keep_going_on_failure=keep_going),
                )
            )

    def test_abort_at_a_later_variant_stops_the_grid(self, tmp_path, monkeypatch):
        results = self._run(keep_going=False, tmp_path=tmp_path, monkeypatch=monkeypatch)

        assert [r.name for r in results] == ["c0", "c1"]
        aborted = results[-1]
        assert aborted.status == "failed"
        assert aborted.error_class == "yaml_build_error"
        assert "boom" in (aborted.error or "")
        assert aborted.returncode is None
        assert aborted.workspace is None
        marker = json.loads(
            (next((tmp_path / "out").glob("*_c1/abort_reason.json"))).read_text(encoding="utf-8"),
        )
        assert marker["error_class"] == "yaml_build_error"

    def test_abort_at_a_later_variant_is_walked_past(self, tmp_path, monkeypatch):
        results = self._run(keep_going=True, tmp_path=tmp_path, monkeypatch=monkeypatch)

        assert [r.name for r in results] == ["c0", "c1", "c2"]
        assert [r.error_class for r in results] == [
            "no_benchmark_workspace",
            "yaml_build_error",
            "no_benchmark_workspace",
        ]


# auto-warmup teardown timing


class TestAutoWarmupTeardown:
    """With auto-warmup engaged (single-node, Magpie built-in script, ``INFERENCE_OPTIMIZER_RUN_GRID_WARMUP=1``),
    ``teardown_lifecycle_server`` is called exactly once per variant on each of these paths, keyed on the resolved
    lifecycle framework/port.
    """

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
                    grid=[GridVariant(name="cand")],
                    output_root=out,
                    recipe=GridRecipe(base_yaml_path=base, target=BenchmarkTarget(gpu_type="mi300x")),
                    launch=GridLaunch(magpie_python=sys.executable),
                )
            )
        return results, teardown_calls

    @pytest.mark.parametrize("framework", ["vllm", "sglang", "atom"])
    def test_warmup_success_measured_success_tears_down_once(self, tmp_path, monkeypatch, framework):
        monkeypatch.setenv("INFERENCE_OPTIMIZER_RUN_GRID_WARMUP", "1")
        # Pin the free-port picker so the teardown-port assertion is deterministic (baseline uses a per-session free
        # port).
        monkeypatch.setattr(
            "hyperloom.orchestrator.actions.executors._server_lifecycle._pick_free_port",
            lambda: 8888,
        )
        base = tmp_path / "base.yaml"
        _write_base_yaml(base, framework=framework)
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
        assert teardown_calls[0]["framework"] == framework
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
        """Companion path: the warmup round itself times out (its own teardown branch, before the measured round is ever reached)."""
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


# servers a round's AgentX client boots


@pytest.fixture
def client_servers():
    """Stand-ins for the servers a round's AgentX client boots; whatever the run leaves of them dies at teardown."""
    servers: list[subprocess.Popen] = []
    yield servers
    for server in servers:
        if server.poll() is None:
            server.kill()
        server.wait(timeout=5)


class TestRoundServerTeardown:
    """An AgentX client boots its server setsid'd, outside the round's process tree, so reaping the round never reaches
    it. The variant's teardown stops it however the round ended, and the stop marker only calls the round cleaned up
    once it is gone.
    """

    def _run(self, client_servers, base, out, *, rc, measured=False):
        def _round(cmd, *a, **k):
            slot = Path(cmd[cmd.index("--output-dir") + 1])
            ws = _valid_workspace(slot) if measured else slot / "benchmark_sglang_20260101_000000"
            ws.mkdir(parents=True, exist_ok=True)
            server = subprocess.Popen(
                [sys.executable, "-c", "import time; _='sglang.launch_server'; time.sleep(120)"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            client_servers.append(server)
            (ws / AGENTX_SERVER_PIDFILE).write_text(f"{server.pid}\n", encoding="utf-8")
            return subprocess.CompletedProcess(cmd, rc, "", "")

        with patch("hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill", side_effect=_round):
            return asyncio.run(
                run_grid(
                    grid=[GridVariant(name="cand")],
                    output_root=out,
                    recipe=GridRecipe(base_yaml_path=base, target=BenchmarkTarget(gpu_type="mi300x")),
                    launch=GridLaunch(magpie_python=sys.executable),
                )
            )

    @staticmethod
    def _abort_marker(out: Path) -> dict:
        return json.loads(next(out.glob("variant_*/abort_reason.json")).read_text(encoding="utf-8"))

    def test_cancelled_round_stops_the_server_its_client_booted(self, tmp_path, monkeypatch, client_servers):
        monkeypatch.setenv("INFERENCE_OPTIMIZER_RUN_GRID_WARMUP", "0")
        base = tmp_path / "base.yaml"
        _write_base_yaml(base)
        out = tmp_path / "out"

        [result] = self._run(client_servers, base, out, rc=ORCHESTRATOR_CANCELLED_RETURNCODE)

        [server] = client_servers
        assert server.wait(timeout=5) == -signal.SIGTERM
        stop = STOPPED_BY_THE_RUN[ORCHESTRATOR_CANCELLED_CLASS]
        assert (result.status, result.error_class) == ("skipped", stop.error_class)
        assert not [w for w in result.nonfatal_warnings if w.startswith("server_survived_teardown:")]
        assert self._abort_marker(out)["error"] == f"{stop.interrupted}; tree reaped"
        assert not list(out.rglob(AGENTX_SERVER_PIDFILE))

    def test_a_server_that_outlives_sigkill_is_not_reported_reaped(self, tmp_path, monkeypatch, client_servers):
        monkeypatch.setenv("INFERENCE_OPTIMIZER_RUN_GRID_WARMUP", "0")
        monkeypatch.setattr(sl, "_terminate_group", lambda _pgid: False)
        base = tmp_path / "base.yaml"
        _write_base_yaml(base)
        out = tmp_path / "out"

        [result] = self._run(client_servers, base, out, rc=ORCHESTRATOR_CANCELLED_RETURNCODE)

        [server] = client_servers
        assert f"server_survived_teardown:{server.pid}" in result.nonfatal_warnings
        assert self._abort_marker(out)["error"].endswith(
            f"; tree reaped, but server pid(s) {server.pid} survived SIGKILL"
        )

    def test_measured_round_stops_the_server_its_client_left_running(self, tmp_path, monkeypatch, client_servers):
        monkeypatch.setattr(gr, "REPORT_SETTLE_SECONDS", 0.0)
        monkeypatch.setenv("INFERENCE_OPTIMIZER_RUN_GRID_WARMUP", "0")
        base = tmp_path / "base.yaml"
        _write_base_yaml(base)

        [result] = self._run(client_servers, base, tmp_path / "out", rc=0, measured=True)

        assert result.status == "succeeded"
        [server] = client_servers
        assert server.wait(timeout=5) == -signal.SIGTERM

    def test_timed_out_round_stops_its_server_before_the_next_round(self, tmp_path, monkeypatch, client_servers):
        """A round killed at its cap leaves the server its client booted; the next round must not start beside it."""
        monkeypatch.setenv("INFERENCE_OPTIMIZER_RUN_GRID_WARMUP", "0")
        base = tmp_path / "base.yaml"
        _write_base_yaml(base)
        out = tmp_path / "out"
        alive_at_next_round: list[bool] = []

        def _round(cmd, *a, **k):
            if client_servers:
                alive_at_next_round.append(client_servers[0].poll() is None)
                return subprocess.CompletedProcess(cmd, 1, "", "")
            slot = Path(cmd[cmd.index("--output-dir") + 1])
            ws = slot / "benchmark_sglang_20260101_000000"
            ws.mkdir(parents=True, exist_ok=True)
            server = subprocess.Popen(
                [sys.executable, "-c", "import time; _='sglang.launch_server'; time.sleep(120)"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            client_servers.append(server)
            (ws / AGENTX_SERVER_PIDFILE).write_text(f"{server.pid}\n", encoding="utf-8")
            raise subprocess.TimeoutExpired(cmd, 7800)

        with patch("hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill", side_effect=_round):
            results = asyncio.run(
                run_grid(
                    grid=[GridVariant(name="c28"), GridVariant(name="c20")],
                    output_root=out,
                    recipe=GridRecipe(base_yaml_path=base, target=BenchmarkTarget(gpu_type="mi300x")),
                    launch=GridLaunch(magpie_python=sys.executable),
                )
            )

        assert results[0].error_class == "magpie_timeout"
        assert alive_at_next_round == [False]
        [server] = client_servers
        assert server.wait(timeout=5) == -signal.SIGTERM


# _resolve_mn_effective_server_args: prefer the materialized variant YAML; fall back to recomposing from the base YAML
# when the variant read fails.
class TestResolveMnEffectiveServerArgs:
    def _base(self, tmp_path: Path, *, args: str = "--tp 8") -> Path:
        base = tmp_path / "base.yaml"
        base.write_text(
            yaml.safe_dump({"benchmark": {"framework": "sglang", "envs": {"EXTRA_SGLANG_ARGS": args}}}),
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
        # Read verbatim from the variant YAML; the variant/base extras are NOT re-composed on this happy path.
        assert out == "--tp 8 --chunked-prefill 4096"

    def test_variant_env_absent_returns_empty(self, tmp_path):
        base = self._base(tmp_path)
        cfg = tmp_path / "variant.yaml"
        cfg.write_text(yaml.safe_dump({"benchmark": {"framework": "sglang", "envs": {}}}), encoding="utf-8")
        variant = GridVariant(name="v1")

        out = gr._resolve_mn_effective_server_args(cfg, base, variant, base_extra_args="", base_args_mode="append")
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


# Per-variant progress heartbeat


class TestVariantHeartbeat:
    """A grid that runs for hours must be distinguishable from one that hung."""

    def _run_capture_progress(self, run_side_effect, base, out, *, grid_n=2, notes=None, sink=None):
        notes = [] if notes is None else notes

        async def _collect(**note):
            notes.append(note)

        with (
            progress_scope(sink or _collect),
            patch(
                "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
                side_effect=run_side_effect,
            ),
        ):
            results = asyncio.run(
                run_grid(
                    grid=[GridVariant(name=f"c{i}") for i in range(grid_n)],
                    output_root=out,
                    recipe=GridRecipe(base_yaml_path=base, target=BenchmarkTarget(gpu_type="mi300x")),
                    launch=GridLaunch(magpie_python=sys.executable),
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

    def test_the_note_names_the_variant_that_ran_not_the_last_row(self):
        """The tail of ``results`` is not always the variant that just reported."""
        grid = [GridVariant(name=f"c{i}") for i in range(3)]
        stopped = gr.VariantResult(
            name="c0",
            extra_server_args="",
            extra_envs={},
            status="skipped",
            error_class="orchestrator_cancelled",
        )
        never_ran = [
            gr.VariantResult(
                name=variant.name,
                extra_server_args="",
                extra_envs={},
                status="not_run",
            )
            for variant in grid[1:]
        ]

        assert gr._variant_progress_note(grid, [stopped, *never_ran], 0) == {
            "unit": "variant",
            "label": "c0",
            "index": 1,
            "total": 3,
            "status": "skipped",
            "output_throughput": None,
        }

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

    def test_a_variant_keeps_reporting_while_its_benchmark_blocks(
        self,
        tmp_path,
        monkeypatch,
        progress_cadence,
    ):
        """Entry markers alone leave the row silent for a whole variant timeout."""
        monkeypatch.setenv("INFERENCE_OPTIMIZER_RUN_GRID_WARMUP", "0")
        base = tmp_path / "base.yaml"
        _write_base_yaml(base)

        def _ok(cmd, *_a, **_k):
            out_idx = cmd.index("--output-dir")
            _valid_workspace(Path(cmd[out_idx + 1]))
            return subprocess.CompletedProcess(cmd, 0, "ok", "")

        results, _notes = self._run_capture_progress(
            chatty_child(progress_cadence, _ok, blocks_for_s=600.0, line_every_s=30.0),
            base,
            tmp_path / "out",
            grid_n=1,
            sink=progress_cadence.sink(),
        )

        assert [r.status for r in results] == ["succeeded"]
        running = [note for note in progress_cadence.notes if note["status"] == "running"]
        assert len(running) >= 3
        assert all(note["output_lines"] > 0 for note in running)
        assert progress_cadence.widest_silence() <= 150.0
