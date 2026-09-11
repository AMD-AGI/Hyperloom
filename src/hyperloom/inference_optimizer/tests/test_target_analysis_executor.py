# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Integration + unit tests for :class:`TargetAnalysisExecutor`."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hyperloom.orchestrator.actions.executors import TargetAnalysisExecutor
from hyperloom.orchestrator.actions.executors import target_analysis as ta
from hyperloom.orchestrator.state.task_registry import Task


# Fixtures
@dataclass
class _Ctx:
    task: Task
    lease: Any = None
    extra: dict[str, Any] = None  # type: ignore[assignment]


def _ctx(session_dir: Path, params: dict[str, Any] | None = None) -> _Ctx:
    return _Ctx(
        task=Task(
            task_id="t-target-analysis-1",
            kind="target_analysis",
            params=params or {},
            requires_lanes=(),
            state="running",
            idempotency_key="ta-1",
        ),
        extra={"session_dir": str(session_dir)},
    )


@pytest.fixture
def session_dir(tmp_path: Path) -> Path:
    sd = tmp_path / "sess"
    sd.mkdir()
    return sd


def _ifx_rows() -> list[dict[str, Any]]:
    """A minimal InferenceX-shaped benchmark row set for mocking ``fetch_rows``."""
    return [
        {
            "hardware": "b300",
            "precision": "fp8",
            "isl": 1024,
            "osl": 1024,
            "conc": 64,
            "decode_tp": 2,
            "metrics": {
                "tput_per_gpu": 2781.5,
                "output_tput_per_gpu": 1390.7,
                "mean_ttft": 0.094,  # seconds
                "mean_tpot": 0.022,
                "mean_e2el": 20.6,
            },
            "date": "2026-04-17",
        }
    ]


def _patch_fetch_rows(monkeypatch, rows: list[dict[str, Any]] | None) -> None:
    """Patch the ``fetch_rows`` symbol ``analyze`` uses with a stub."""
    monkeypatch.setattr(
        "hyperloom.inference_optimizer.baseline_comparison.target_analyzer.fetch_rows",
        lambda _name: rows,
    )


# Tests
@pytest.mark.asyncio
async def test_no_flag_writes_skipped_marker(session_dir):
    """Without --compare-against-gpu, the executor still runs and persists a ``no_target_gpu_configured`` marker JSON."""
    executor = TargetAnalysisExecutor(compare_against_gpu="", session_dir=session_dir)
    result = await executor(_ctx(session_dir, {"model_path": "MiniMax-M2.5"}))
    assert result["status"] == "succeeded"
    assert result["baseline_status"] == "skipped"
    assert result["reason"] == "no_target_gpu_configured"
    json_path = session_dir / "target_analysis" / "target_baseline.json"
    assert json_path.exists()
    on_disk = json.loads(json_path.read_text())
    assert on_disk["status"] == "skipped"
    assert on_disk["reason"] == "no_target_gpu_configured"
    assert on_disk["query"]["gpu"] == ""


@pytest.mark.asyncio
async def test_no_inferencex_data_graceful(session_dir, monkeypatch):
    """InferenceX returns no rows for the model → succeeded + no_match."""
    _patch_fetch_rows(monkeypatch, [])
    executor = TargetAnalysisExecutor(compare_against_gpu="b300", session_dir=session_dir)
    params = {
        "model_path": "MiniMax-M2.5",
        "framework": "vllm",
        "precision": "fp8",
        "isl": 1024,
        "osl": 1024,
    }
    result = await executor(_ctx(session_dir, params))
    assert result["status"] == "succeeded"
    assert result["baseline_status"] == "no_match"
    assert result["reason"] == "no_inferencex_data"
    assert (session_dir / "target_analysis" / "target_baseline.json").exists()


@pytest.mark.asyncio
async def test_model_mapping_miss_writes_skipped(session_dir, monkeypatch):
    """Unknown model → skipped without any HTTP traffic."""
    # URL points at a hang-if-hit port; mapping miss must short-circuit before any fetch
    monkeypatch.setenv("INFERENCEX_BASE_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("INFERENCEX_TIMEOUT_SEC", "5.0")
    monkeypatch.setenv("INFERENCEX_MAX_ATTEMPTS", "1")

    executor = TargetAnalysisExecutor(compare_against_gpu="b300", session_dir=session_dir)
    result = await executor(
        _ctx(
            session_dir,
            {
                "model_path": "/path/models/MyCorp-Custom-FT-7B",
                "framework": "vllm",
                "precision": "fp8",
                "isl": 1024,
                "osl": 1024,
            },
        )
    )
    assert result["status"] == "succeeded"
    assert result["baseline_status"] == "skipped"
    assert result["reason"] == "model_mapping_miss"


@pytest.mark.asyncio
async def test_happy_path_writes_files(session_dir, monkeypatch):
    """Full pipeline reading live InferenceX-measured rows (mocked)."""
    _patch_fetch_rows(monkeypatch, _ifx_rows())

    executor = TargetAnalysisExecutor(compare_against_gpu="b300", session_dir=session_dir)
    result = await executor(
        _ctx(
            session_dir,
            {
                "model_path": "/path/models/MiniMaxAI-MiniMax-M2.5",
                "framework": "vllm",
                "precision": "fp8",
                "isl": 1024,
                "osl": 1024,
            },
        )
    )
    assert result["status"] == "succeeded"
    assert result["baseline_status"] == "ok"
    assert result["reason"] == "ok"
    assert result["row_count"] == 1
    assert result["best_tput_per_gpu"] == pytest.approx(2781.5)
    assert result["best_conc"] == 64

    json_path = Path(result["json_path"])
    md_path = Path(result["md_path"])
    assert json_path.exists()
    assert md_path.exists()

    on_disk = json.loads(json_path.read_text())
    assert on_disk["status"] == "ok"
    assert on_disk["reason"] == "ok"
    assert on_disk["best"]["tput_per_gpu"] == pytest.approx(2781.5)
    assert on_disk["query"]["model"] == "MiniMax-M2.5"
    assert on_disk["query"]["gpu"] == "b300"
    # Provenance is the live API URL, never the old ``llm_authored`` marker.
    assert on_disk["source"].startswith("http")
    assert "llm_authored" not in on_disk["source"]

    md_text = md_path.read_text()
    assert "## Reference best" in md_text


@pytest.mark.asyncio
async def test_report_executor_renders_external_baseline_section(tmp_path: Path, monkeypatch):
    """ReportExecutor reads target_baseline.json and injects an advisory section without touching SharedState."""
    from hyperloom.orchestrator.actions.executors import ReportExecutor
    from hyperloom.orchestrator.state.shared_state import SharedState
    from hyperloom.orchestrator.bus.storage.connection import SqliteConnection

    sd = tmp_path / "sess-report"
    sd.mkdir()
    SharedState(session_id=sd.name, model_name="MiniMax-M2.5", baseline_tput=1500.0).save(sd)
    (sd / "storage").mkdir()
    SqliteConnection(sd / "storage" / "coordinator.db").close()

    target_dir = sd / "target_analysis"
    target_dir.mkdir()
    (target_dir / "target_baseline.json").write_text(
        json.dumps(
            {
                "query": {
                    "model": "MiniMax-M2.5",
                    "gpu": "b300",
                    "framework": "vllm",
                    "precision": "fp8",
                    "isl": 1024,
                    "osl": 1024,
                },
                "fetched_at": "2026-05-12T07:00:34Z",
                "row_count": 1,
                "best": {
                    "tput_per_gpu": 2781.5,
                    "output_tput_per_gpu": 1390.7,
                    "conc": 64,
                    "decode_tp": 2,
                    "mean_ttft_ms": 94.0,
                    "mean_tpot_ms": 22.0,
                    "mean_e2el_ms": 20600.0,
                    "date": "2026-04-17",
                },
                "all_concurrencies": [],
                "status": "ok",
                "warning": "",
                "source": "https://inferencex.semianalysis.com/api/v1",
            }
        )
    )

    monkeypatch.setenv("USER_DATA_PATH", str(sd))

    class _ReportCtx:
        task = Task(task_id="r-1", kind="report", params={}, requires_lanes=(), state="running", idempotency_key="r-1")
        lease = None
        extra = {"session_dir": str(sd)}

    result = await ReportExecutor()(_ReportCtx())
    assert result["status"] == "succeeded"
    final_md = Path(result["md_path"]).read_text()
    assert "## External baseline" in final_md
    assert "2781.5" in final_md
    assert "Advisory only" in final_md

    final_json = json.loads(Path(result["json_path"]).read_text())
    assert "external_baseline" in final_json
    assert final_json["external_baseline"]["status"] == "ok"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode_source", ["state", "environment"])
async def test_agentx_executor_fetches_reference_and_maps_derived_id(session_dir, monkeypatch, mode_source):
    from hyperloom.inference_optimizer.baseline_comparison import inferencex_client
    from hyperloom.orchestrator.knowledge import research_hints

    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    ctx = _ctx(session_dir, {"model_path": "/models/GLM-5.2-MXFP4", "precision": "mxfp4", "isl": 1024, "osl": 2048})
    ctx.extra["shared_state"] = SimpleNamespace(benchmark_mode="agentx" if mode_source == "state" else "")
    if mode_source == "environment":
        monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    calls = []

    def fetch(url):
        calls.append(url)
        if "/benchmarks?" in url:
            return json.dumps(
                [
                    {
                        "id": "42",
                        "hardware": "b300",
                        "model": "glm5.2",
                        "precision": "fp4",
                        "benchmark_type": "agentic_traces",
                        "isl": None,
                        "osl": None,
                        "conc": 4,
                        "decode_tp": 8,
                        "metrics": {"tput_per_gpu": 800.0, "mean_tpot": 0.001},
                    }
                ]
            ).encode()
        assert "/derived-agentic-metrics?ids=42" in url
        return b'{"42":{"id":42,"p90_e2e_norm_intvty":20.0}}'

    monkeypatch.setattr(inferencex_client, "_fetch_raw", fetch)
    result = await TargetAnalysisExecutor(compare_against_gpu="b300")(ctx)
    assert result["baseline_status"] == "ok"
    assert result["best_e2e_norm_intvty_p90"] == 20.0
    assert result["best_benchmark_id"] == "42"
    assert len(calls) == 2
    summary = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))
    assert summary["query"]["benchmark_mode"] == "agentx"
    assert summary["query"]["isl"] is None
    assert summary["best"]["tput_per_gpu"] == 800.0
    target = research_hints.load_competitor_target(session_dir)
    assert target["per_conc"][0]["e2e_norm_intvty_p90"] == 20.0
    assert "interactivity" not in target["per_conc"][0]


@pytest.mark.asyncio
async def test_agentx_executor_uses_persisted_model_and_precision(session_dir, monkeypatch):
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    for key in ("MODEL_PATH", "FRAMEWORK", "PRECISION"):
        monkeypatch.delenv(key, raising=False)
    ctx = _ctx(session_dir)
    ctx.extra["shared_state"] = SimpleNamespace(
        benchmark_mode="agentx", model_path="/models/GLM-5.2-MXFP4", framework="sglang", precision="mxfp4"
    )
    captured = []

    def analyze(**kwargs):
        captured.append(kwargs)
        return _DummySummary()

    monkeypatch.setattr(ta, "analyze", analyze)
    await TargetAnalysisExecutor(compare_against_gpu="b300")(ctx)
    assert captured[0]["model_path"] == "/models/GLM-5.2-MXFP4"
    assert captured[0]["precision"] == "mxfp4"
    assert captured[0]["benchmark_mode"] == "agentx"


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["params", "environment"])
async def test_agentx_explicit_precision_override_wins_over_stale_state(session_dir, monkeypatch, source):
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    monkeypatch.delenv("PRECISION", raising=False)
    params = {"model_path": "GLM-5.2"}
    if source == "params":
        params["precision"] = "bf16"
    else:
        monkeypatch.setenv("PRECISION", "bf16")
    ctx = _ctx(session_dir, params)
    ctx.extra["shared_state"] = SimpleNamespace(benchmark_mode="agentx", precision="fp8")
    captured = []

    def analyze(**kwargs):
        captured.append(kwargs)
        return _DummySummary()

    monkeypatch.setattr(ta, "analyze", analyze)
    await TargetAnalysisExecutor(compare_against_gpu="b300")(ctx)
    assert captured[0]["precision"] == "bf16"
    assert ctx.extra["shared_state"].precision == "fp8"


@pytest.mark.asyncio
async def test_agentx_no_gpu_persists_mode_and_clears_previous_target(session_dir, monkeypatch):
    from hyperloom.inference_optimizer.session import session_paths

    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    ctx = _ctx(session_dir, {"model_path": "GLM-5.2"})
    stale = session_paths.competitor_target_json(session_dir)
    stale.write_text('{"old":true}', encoding="utf-8")
    result = await TargetAnalysisExecutor(compare_against_gpu="")(ctx)
    assert result["reason"] == "no_target_gpu_configured"
    assert not stale.exists()
    assert json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))["query"]["benchmark_mode"] == "agentx"


@pytest.mark.asyncio
async def test_analyzer_exception_replaces_previous_reference_artifacts(session_dir, monkeypatch):
    from hyperloom.inference_optimizer.session import session_paths

    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    stale = session_paths.competitor_target_json(session_dir)
    stale.write_text('{"old":true}', encoding="utf-8")
    baseline = session_dir / "target_analysis/target_baseline.json"
    baseline.parent.mkdir()
    baseline.write_text('{"status":"ok","best":{"tput_per_gpu":9999}}', encoding="utf-8")

    def fail(**kwargs):
        raise ValueError("schema mismatch")

    monkeypatch.setattr(ta, "analyze", fail)
    result = await TargetAnalysisExecutor(compare_against_gpu="b300")(_ctx(session_dir, {"model_path": "GLM-5.2"}))
    assert result["reason"] == "analyzer_crash"
    assert not stale.exists()
    persisted = json.loads(baseline.read_text(encoding="utf-8"))
    assert persisted["reason"] == "analyzer_crash"
    assert persisted["best"] is None
    assert persisted["query"]["benchmark_mode"] == "agentx"


@pytest.mark.asyncio
@pytest.mark.parametrize("gpu", ["b300", ""])
async def test_initial_analysis_state_write_failure_keeps_history_and_skips_fetch(session_dir, monkeypatch, gpu):
    from hyperloom.common import io as common_io

    baseline = session_dir / "target_analysis/target_baseline.json"
    baseline.parent.mkdir()
    baseline.write_text(
        json.dumps({"status": "ok", "fetched_at": "2026-01-01T00:00:00Z", "best": {"tput_per_gpu": 42.0}}),
        encoding="utf-8",
    )
    fetched = False
    previous = baseline.read_bytes()
    real_write_text = Path.write_text

    def fetch(_model):
        nonlocal fetched
        fetched = True
        return []

    def fail_write(*_args, **_kwargs):
        raise OSError("state volume is read-only")

    def fail_baseline_write(path, text, **kwargs):
        if path == baseline:
            return fail_write()
        return real_write_text(path, text, **kwargs)

    monkeypatch.setattr(
        "hyperloom.inference_optimizer.baseline_comparison.target_analyzer.fetch_rows",
        fetch,
    )
    monkeypatch.setattr(common_io, "atomic_write_json", fail_write)
    monkeypatch.setattr(Path, "write_text", fail_baseline_write)

    with pytest.raises(OSError, match="state volume is read-only"):
        await TargetAnalysisExecutor(compare_against_gpu=gpu, session_dir=session_dir)(
            _ctx(session_dir, {"model_path": "MiniMax-M2.5"})
        )

    assert fetched is False
    assert baseline.read_bytes() == previous
    persisted = json.loads(baseline.read_text(encoding="utf-8"))
    assert persisted["best"]["tput_per_gpu"] == 42.0
    assert persisted["fetched_at"] == "2026-01-01T00:00:00Z"


@pytest.mark.asyncio
async def test_analysis_and_failure_write_errors_never_report_previous_reference(session_dir, monkeypatch):
    from hyperloom.common import io as common_io
    from hyperloom.inference_optimizer.session import session_paths
    from hyperloom.orchestrator.actions.executors import ReportExecutor
    from hyperloom.orchestrator.state.shared_state import SharedState

    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    monkeypatch.delenv("HYPERLOOM_RESULTS_SERVICE_URL", raising=False)
    monkeypatch.delenv("HYPERLOOM_RESULTS_AUTO_PUBLISH", raising=False)
    SharedState(session_id=session_dir.name, benchmark_mode="agentx", model_name="GLM-5.2").save(session_dir)
    baseline = session_paths.target_baseline_json(session_dir)
    baseline.parent.mkdir()
    baseline.write_text(
        json.dumps(
            {
                "status": "ok",
                "fetched_at": "2026-01-01T00:00:00Z",
                "query": {"model": "GLM-5.2", "gpu": "b300", "benchmark_mode": "agentx", "precision": "fp4"},
                "best": {"tput_per_gpu": 9999, "benchmark_id": "old-benchmark"},
            }
        ),
        encoding="utf-8",
    )
    competitor = session_paths.competitor_target_json(session_dir)
    competitor.write_text('{"old":true}', encoding="utf-8")
    real_write = common_io.atomic_write_json
    writes = 0
    fetched = False

    def fail_after_pending(path, value, **kwargs):
        nonlocal writes
        if Path(path) == baseline:
            writes += 1
            if writes > 1:
                raise OSError("failure summary cannot be written")
        return real_write(path, value, **kwargs)

    def fail_fetch(_model):
        nonlocal fetched
        fetched = True
        pending = json.loads(baseline.read_text(encoding="utf-8"))
        assert pending["status"] == "in_progress"
        assert pending["best"] is None
        assert pending["all_concurrencies"] == []
        assert not competitor.exists()
        raise ValueError("invalid upstream schema")

    monkeypatch.setattr(common_io, "atomic_write_json", fail_after_pending)
    monkeypatch.setattr("hyperloom.inference_optimizer.baseline_comparison.target_analyzer.fetch_rows", fail_fetch)
    result = await TargetAnalysisExecutor(compare_against_gpu="b300")(
        _ctx(session_dir, {"model_path": "GLM-5.2", "precision": "fp4"})
    )
    assert fetched is True
    assert result["reason"] == "analyzer_crash"
    assert writes == 2
    assert json.loads(baseline.read_text(encoding="utf-8"))["status"] == "in_progress"

    report = await ReportExecutor()(_ctx(session_dir))
    final = json.loads(Path(report["json_path"]).read_text(encoding="utf-8"))
    assert final["external_baseline"]["status"] == "in_progress"
    assert final["external_baseline"]["best"] is None
    assert final["external_baseline"]["comparison"]["status"] == "unavailable"
    assert "old-benchmark" not in json.dumps(final["external_baseline"])
    assert "9999" not in Path(report["md_path"]).read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_terminal_state_write_failure_leaves_in_progress_reference(session_dir, monkeypatch):
    from hyperloom.common import io as common_io

    real_write = common_io.atomic_write_json
    writes = 0

    def fail_terminal_write(*args, **kwargs):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("terminal write failed")
        return real_write(*args, **kwargs)

    _patch_fetch_rows(monkeypatch, [])
    monkeypatch.setattr(common_io, "atomic_write_json", fail_terminal_write)

    with pytest.raises(OSError, match="terminal write failed"):
        await TargetAnalysisExecutor(compare_against_gpu="b300", session_dir=session_dir)(
            _ctx(session_dir, {"model_path": "MiniMax-M2.5"})
        )

    persisted = json.loads((session_dir / "target_analysis/target_baseline.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "in_progress"
    assert persisted["best"] is None
    assert writes == 2


@pytest.mark.asyncio
async def test_markdown_write_failure_keeps_authoritative_json(session_dir, monkeypatch):
    real_write = Path.write_text

    def fail_markdown(path, text, **kwargs):
        if path.suffix == ".md":
            raise OSError("markdown projection failed")
        return real_write(path, text, **kwargs)

    _patch_fetch_rows(monkeypatch, _ifx_rows())
    monkeypatch.setattr(Path, "write_text", fail_markdown)

    result = await TargetAnalysisExecutor(compare_against_gpu="b300", session_dir=session_dir)(
        _ctx(
            session_dir,
            {
                "model_path": "MiniMax-M2.5",
                "precision": "fp8",
                "isl": 1024,
                "osl": 1024,
            },
        )
    )

    persisted = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))
    assert result["baseline_status"] == "ok"
    assert persisted["status"] == "ok"
    assert persisted["best"]["tput_per_gpu"] == pytest.approx(2781.5)
    assert not Path(result["md_path"]).exists()


@pytest.mark.asyncio
async def test_agentx_request_records_to_external_reference_and_final_report(session_dir, monkeypatch):
    import yaml

    from hyperloom.inference_optimizer.agentx.deploy import agentx_asset_dir
    from hyperloom.inference_optimizer.baseline_comparison import inferencex_client
    from hyperloom.orchestrator.actions.executors import ReportExecutor
    from hyperloom.orchestrator.knowledge import research_hints
    from hyperloom.orchestrator.state.shared_state import SharedState

    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    monkeypatch.delenv("HYPERLOOM_PERF_METRIC", raising=False)
    monkeypatch.delenv("AGENTX_NONCANONICAL_REASONS", raising=False)
    workspace = session_dir / "accepted"
    workspace.mkdir()
    source = workspace / "profile_export_aiperf.json"
    source.write_text(
        json.dumps(
            {
                "metadata": {"submission_valid": True},
                "output_token_throughput": {"avg": 100.0},
                "input_token_throughput": {"avg": 900.0},
                "total_token_throughput": {"avg": 1000.0},
                "e2e_output_token_throughput": {"p10": 1.9},
            }
        ),
        encoding="utf-8",
    )
    records = [
        {
            "metadata": {"benchmark_phase": "profiling"},
            "metrics": {
                "request_latency": {"value": latency, "unit": "ms"},
                "time_to_first_token": {"value": 10, "unit": "ms"},
                "input_sequence_length": {"value": 128, "unit": "tokens"},
                "output_sequence_length": {"value": 10, "unit": "tokens"},
            },
        }
        for latency in (1000.0, 10000.0)
    ]
    source.with_name("profile_export.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )
    spec = importlib.util.spec_from_file_location("_closed_loop_mapper", agentx_asset_dir() / "map_aiperf.py")
    mapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mapper)
    raw_path = workspace / "inferencex_result.json"
    mapper.main(str(source), str(raw_path))
    recipe = workspace / "config.yaml"
    recipe.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "framework": "sglang",
                    "model": "/models/GLM-5.2-MXFP4",
                    "precision": "mxfp4",
                    "run_mode": "local",
                    "envs": {"TP": 2, "CONC": 4},
                }
            }
        ),
        encoding="utf-8",
    )
    state = SharedState(
        session_id=session_dir.name,
        benchmark_mode="agentx",
        framework="sglang",
        model_name="GLM-5.2-MXFP4",
        model_path="/models/GLM-5.2-MXFP4",
        precision="mxfp4",
        tp=8,
        conc=99,
        baseline_tput=100.0,
        baseline_perf={"total_throughput": 1000.0, "e2e_norm_intvty_p90": 1.9},
        current_best={
            "tput": 100.0,
            "total_throughput": 1000.0,
            "e2e_norm_intvty_p90": 1.9,
            "measurement": {
                "tput": 100.0,
                "raw_result_path": str(raw_path),
                "benchmark_workspace": str(workspace),
                "launch_evidence": {
                    "materialized_config_path": str(recipe),
                    "recipe_digest": "sha256:" + hashlib.sha256(recipe.read_bytes()).hexdigest(),
                    "requested_server_env": {"TP": "2", "CONC": "4"},
                },
            },
        },
    )
    state.save(session_dir)
    calls = []

    def fetch(url):
        calls.append(url)
        if "/benchmarks?" in url:
            return json.dumps(
                [
                    {
                        "id": "42",
                        "hardware": "b300",
                        "precision": "fp4",
                        "benchmark_type": "agentic_traces",
                        "isl": None,
                        "osl": None,
                        "conc": 4,
                        "decode_tp": 8,
                        "metrics": {"tput_per_gpu": 1000.0},
                    }
                ]
            ).encode()
        return json.dumps({"42": {"id": 42, "p90_e2e_norm_intvty": 2.0 / 0.91}}).encode()

    monkeypatch.setattr(inferencex_client, "_fetch_raw", fetch)
    ctx = _ctx(session_dir)
    ctx.extra["shared_state"] = state
    analyzed = await TargetAnalysisExecutor(compare_against_gpu="b300")(ctx)
    assert analyzed["baseline_status"] == "ok"
    advisory = research_hints.gap_for_state(research_hints.load_competitor_target(session_dir), state)
    assert advisory["throughput_gap_pct"] == 50.0
    assert advisory["interactivity_gap_pct"] == pytest.approx(50.0)
    ctx.task.kind = "report"
    report = await ReportExecutor()(ctx)
    final = json.loads(Path(report["json_path"]).read_text(encoding="utf-8"))
    assert final["external_baseline"]["comparison"] == advisory
    assert final["current_best"]["e2e_norm_intvty_p90"] == 1.9
    assert final["external_baseline"]["comparison"]["local_e2e_norm_intvty_p90"] == pytest.approx(1.0 / 0.91)
    assert "same-concurrency" in Path(analyzed["md_path"]).read_text(encoding="utf-8")
    assert "total throughput/GPU" in Path(report["md_path"]).read_text(encoding="utf-8")
    assert len(calls) == 2


# Unit tests


# env helpers


class TestEnvHelpers:
    def test_env_int_uses_default_when_missing(self, monkeypatch):
        monkeypatch.delenv("TARGET_INT_TEST", raising=False)
        assert ta._env_int("TARGET_INT_TEST", default=7) == 7

    def test_env_int_parses_valid(self, monkeypatch):
        monkeypatch.setenv("TARGET_INT_TEST", "42")
        assert ta._env_int("TARGET_INT_TEST") == 42

    def test_env_int_falls_back_on_invalid(self, monkeypatch):
        monkeypatch.setenv("TARGET_INT_TEST", "garbage")
        assert ta._env_int("TARGET_INT_TEST", default=3) == 3


# session_dir resolution


class _DummySummary:
    status = "ok"
    reason = ""
    warning = ""
    row_count = 3
    best = SimpleNamespace(tput_per_gpu=10.0, conc=4, decode_tp=2)


def _unit_ctx(*, params: dict | None = None, extra: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        task=SimpleNamespace(task_id="ta-t1", kind="target_analysis", params=params or {}),
        extra=extra or {},
    )


class TestResolveSessionDir:
    def test_extra_session_dir_wins(self, tmp_path):
        ex = ta.TargetAnalysisExecutor(compare_against_gpu="MI300X")
        ctx = _unit_ctx(extra={"session_dir": str(tmp_path)})
        assert ex._resolve_session_dir(ctx) == tmp_path

    def test_params_session_dir_used(self, tmp_path):
        ex = ta.TargetAnalysisExecutor(compare_against_gpu="MI300X")
        ctx = _unit_ctx(params={"session_dir": str(tmp_path)})
        assert ex._resolve_session_dir(ctx) == tmp_path

    def test_constructor_session_dir_used(self, tmp_path):
        ex = ta.TargetAnalysisExecutor(
            compare_against_gpu="MI300X",
            session_dir=tmp_path,
        )
        ctx = _unit_ctx()
        assert ex._resolve_session_dir(ctx) == tmp_path

    def test_falls_back_to_paths_session_dir(self, tmp_path, monkeypatch):
        ex = ta.TargetAnalysisExecutor(compare_against_gpu="MI300X")
        monkeypatch.setattr(
            "hyperloom.inference_optimizer.session.paths.session_dir",
            lambda: tmp_path,
        )
        ctx = _unit_ctx()
        assert ex._resolve_session_dir(ctx) == tmp_path

    def test_returns_none_when_fallback_missing(self, monkeypatch):
        ex = ta.TargetAnalysisExecutor(compare_against_gpu="MI300X")

        def boom():
            raise RuntimeError("no session")

        monkeypatch.setattr(
            "hyperloom.inference_optimizer.session.paths.session_dir",
            boom,
        )
        assert ex._resolve_session_dir(_unit_ctx()) is None


# Execution branches


class TestExecutor:
    @pytest.mark.asyncio
    async def test_skipped_when_no_session_dir(self, monkeypatch):
        ex = ta.TargetAnalysisExecutor(compare_against_gpu="MI300X")
        monkeypatch.setattr(ex, "_resolve_session_dir", lambda ctx: None)
        result = await ex(_unit_ctx())
        assert result["status"] == "succeeded"
        assert result["baseline_status"] == "skipped"
        assert result["reason"] == "no_session_dir"

    @pytest.mark.asyncio
    async def test_skipped_when_no_session_dir_clears_stale_competitor_target(
        self,
        tmp_path,
        monkeypatch,
    ):
        from hyperloom.inference_optimizer.session import session_paths
        from hyperloom.orchestrator.knowledge import research_hints

        sd = tmp_path / "sess"
        sd.mkdir()
        research_hints.write_competitor_target(
            sd,
            {
                "gpu": "b300",
                "model": "MiniMax-M2.5",
                "per_conc": [{"conc": 64, "tput_per_gpu": 999.0, "source": "scout"}],
            },
        )
        assert session_paths.competitor_target_json(sd).exists()

        ex = ta.TargetAnalysisExecutor(compare_against_gpu="MI300X")
        monkeypatch.setattr(ex, "_resolve_session_dir", lambda ctx: None)
        monkeypatch.setattr(
            "hyperloom.inference_optimizer.session.paths.session_dir",
            lambda: sd,
        )
        result = await ex(_unit_ctx())
        assert result["reason"] == "no_session_dir"
        assert not session_paths.competitor_target_json(sd).exists()
        assert research_hints.load_competitor_target(sd) is None

    @pytest.mark.asyncio
    async def test_writes_skipped_summary_when_no_gpu(self, tmp_path, monkeypatch):
        ex = ta.TargetAnalysisExecutor(compare_against_gpu="")
        monkeypatch.setattr(ex, "_resolve_session_dir", lambda ctx: tmp_path)
        monkeypatch.setattr(
            "hyperloom.orchestrator.actions.executors.target_analysis.analyze",
            lambda **kwargs: _DummySummary(),
        )
        result = await ex(_unit_ctx())
        assert result["status"] == "succeeded"
        assert result["baseline_status"] == "ok"
        assert result["best_tput_per_gpu"] == 10.0

    @pytest.mark.asyncio
    async def test_analyzer_crash_is_swallowed(self, tmp_path, monkeypatch):
        ex = ta.TargetAnalysisExecutor(compare_against_gpu="MI300X")
        monkeypatch.setattr(ex, "_resolve_session_dir", lambda ctx: tmp_path)

        def boom(**_):
            raise RuntimeError("InferenceX 500")

        monkeypatch.setattr(
            "hyperloom.orchestrator.actions.executors.target_analysis.analyze",
            boom,
        )
        result = await ex(_unit_ctx())
        assert result["status"] == "succeeded"
        assert result["baseline_status"] == "fetch_error"
        assert "analyzer crashed" in result["note"]

    @pytest.mark.asyncio
    async def test_analyzer_crash_in_no_gpu_branch_is_swallowed(
        self,
        tmp_path,
        monkeypatch,
    ):
        ex = ta.TargetAnalysisExecutor(compare_against_gpu="")
        monkeypatch.setattr(ex, "_resolve_session_dir", lambda ctx: tmp_path)

        def boom(**_):
            raise RuntimeError("nope")

        monkeypatch.setattr(
            "hyperloom.orchestrator.actions.executors.target_analysis.analyze",
            boom,
        )
        result = await ex(_unit_ctx())
        assert result["baseline_status"] == "fetch_error"

    @pytest.mark.asyncio
    async def test_format_result_uses_summary_without_best(
        self,
        tmp_path,
        monkeypatch,
    ):
        class _NoBestSummary:
            status = "no_data"
            reason = "row_count==0"
            warning = "filtered_to_empty"
            row_count = 0
            best = None

        ex = ta.TargetAnalysisExecutor(compare_against_gpu="MI300X")
        monkeypatch.setattr(ex, "_resolve_session_dir", lambda ctx: tmp_path)
        monkeypatch.setattr(
            "hyperloom.orchestrator.actions.executors.target_analysis.analyze",
            lambda **kwargs: _NoBestSummary(),
        )
        result = await ex(_unit_ctx(params={"model_path": "/m"}))
        assert result["baseline_status"] == "no_data"
        assert "best_tput_per_gpu" not in result
