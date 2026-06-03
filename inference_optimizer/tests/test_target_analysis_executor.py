"""Integration + unit tests for :class:`TargetAnalysisExecutor`.

Integration tests (full-flow reading the LLM-authored competitor target):

* ``test_no_flag_writes_skipped_marker`` — without ``--compare-against-gpu``
  the executor still runs (it is wired unconditionally in
  ``cli._register_executors``) and writes a structured
  ``reason='no_target_gpu_configured'`` marker JSON.
* ``test_no_competitor_target_graceful`` — when no ``competitor_target.json``
  exists the task still returns ``status=succeeded`` and
  ``baseline_status=no_match``.
* ``test_model_mapping_miss`` — unknown model name persists a ``skipped``
  summary.
* ``test_happy_path_writes_files`` — full pipeline reading a sourced
  competitor target; verifies JSON + MD on disk and the bus payload.

Unit tests (TestEnvHelpers / TestResolveSessionDir / TestExecutor) cover
the small helper utilities (env coercion, ctx fallbacks) and the failure
branches so the "never fail" guarantee for the runner stays locked.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from inference_optimizer.orchestrator.action_executors import TargetAnalysisExecutor
from inference_optimizer.orchestrator.action_executors import target_analysis as ta
from inference_optimizer.orchestrator.task_registry import Task


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_no_flag_writes_skipped_marker(session_dir):
    """Without --compare-against-gpu, ``cli._register_executors`` still
    wires the real :class:`TargetAnalysisExecutor` (the `_noop_prep`
    fallback was removed). The executor must run end-to-end and persist
    a structured ``reason='no_target_gpu_configured'`` marker JSON so the
    coordinator gate opens and the report has a deterministic External
    baseline section to render.
    """
    executor = TargetAnalysisExecutor(compare_against_gpu="",
                                       session_dir=session_dir)
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
async def test_no_competitor_target_graceful(session_dir):
    """No ``competitor_target.json`` on disk → succeeded + no_match."""
    executor = TargetAnalysisExecutor(compare_against_gpu="b300",
                                       session_dir=session_dir)
    params = {
        "model_path": "MiniMax-M2.5",
        "framework":  "vllm",
        "precision":  "fp8",
        "isl":        1024,
        "osl":        1024,
    }
    result = await executor(_ctx(session_dir, params))
    assert result["status"] == "succeeded"
    assert result["baseline_status"] == "no_match"
    assert result["reason"] == "no_competitor_target"
    assert (session_dir / "target_analysis" / "target_baseline.json").exists()


@pytest.mark.asyncio
async def test_model_mapping_miss_writes_skipped(session_dir, monkeypatch):
    """Unknown model → skipped without any HTTP traffic."""
    # Even if the env var points somewhere broken, the mapping miss must
    # short-circuit BEFORE the fetch attempt. We deliberately point the
    # URL at a port that would hang if hit; the test will still finish
    # because no HTTP call is made.
    monkeypatch.setenv("INFERENCEX_BASE_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("INFERENCEX_TIMEOUT_SEC", "5.0")
    monkeypatch.setenv("INFERENCEX_MAX_ATTEMPTS", "1")

    executor = TargetAnalysisExecutor(compare_against_gpu="b300",
                                       session_dir=session_dir)
    result = await executor(_ctx(session_dir, {
        "model_path": "/wekafs/models/MyCorp-Custom-FT-7B",
        "framework":  "vllm",
        "precision":  "fp8",
        "isl":        1024,
        "osl":        1024,
    }))
    assert result["status"] == "succeeded"
    assert result["baseline_status"] == "skipped"
    assert result["reason"] == "model_mapping_miss"


@pytest.mark.asyncio
async def test_happy_path_writes_files(session_dir):
    """Full pipeline reading an LLM-authored competitor target."""
    from inference_optimizer.orchestrator import research_hints
    research_hints.write_competitor_target(session_dir, {
        "gpu": "b300", "model": "MiniMax-M2.5",
        "framework": "vllm", "precision": "fp8",
        "per_conc": [
            {"conc": 64, "tput_per_gpu": 2781.5, "tpot_ms": 22.0,
             "source": "https://pr/9"},
        ],
        "notes": "scout-authored",
    })

    executor = TargetAnalysisExecutor(compare_against_gpu="b300",
                                       session_dir=session_dir)
    result = await executor(_ctx(session_dir, {
        "model_path": "/wekafs/models/MiniMaxAI-MiniMax-M2.5",
        "framework":  "vllm",
        "precision":  "fp8",
        "isl":        1024,
        "osl":        1024,
    }))
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
    assert on_disk["source"] == "llm_authored"

    md_text = md_path.read_text()
    assert "## Reference best" in md_text


@pytest.mark.asyncio
async def test_report_executor_renders_external_baseline_section(
    tmp_path: Path, monkeypatch
):
    """ReportExecutor must read target_baseline.json and inject an
    advisory section into final.md / final.json without touching SharedState."""
    from inference_optimizer.orchestrator.action_executors import ReportExecutor
    from inference_optimizer.orchestrator.shared_state import SharedState
    from inference_optimizer.storage.connection import SqliteConnection

    sd = tmp_path / "sess-report"
    sd.mkdir()
    SharedState(session_id=sd.name, model_name="MiniMax-M2.5",
                baseline_tput=1500.0).save(sd)
    (sd / "storage").mkdir()
    SqliteConnection(sd / "storage" / "coordinator.db").close()

    target_dir = sd / "target_analysis"
    target_dir.mkdir()
    (target_dir / "target_baseline.json").write_text(json.dumps({
        "query": {
            "model": "MiniMax-M2.5", "gpu": "b300",
            "framework": "vllm", "precision": "fp8",
            "isl": 1024, "osl": 1024,
        },
        "fetched_at": "2026-05-12T07:00:34Z",
        "row_count": 1,
        "best": {
            "tput_per_gpu": 2781.5, "output_tput_per_gpu": 1390.7,
            "conc": 64, "decode_tp": 2,
            "mean_ttft_ms": 94.0, "mean_tpot_ms": 22.0, "mean_e2el_ms": 20600.0,
            "date": "2026-04-17",
        },
        "all_concurrencies": [],
        "status": "ok",
        "warning": "",
        "source": "https://inferencex.semianalysis.com/api/v1",
    }))

    monkeypatch.setenv("USER_DATA_PATH", str(sd))

    class _ReportCtx:
        task = Task(task_id="r-1", kind="report", params={},
                    requires_lanes=(), state="running",
                    idempotency_key="r-1")
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


# ===========================================================================
# Unit tests (formerly test_target_analysis_units.py)
# ===========================================================================


# ---------------------------------------------------------------------------
# env helpers
# ---------------------------------------------------------------------------

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

    def test_env_str_strips_whitespace(self, monkeypatch):
        monkeypatch.setenv("TARGET_STR_TEST", "  value  ")
        assert ta._env_str("TARGET_STR_TEST") == "value"


# ---------------------------------------------------------------------------
# session_dir resolution
# ---------------------------------------------------------------------------

class _DummySummary:
    status = "ok"
    reason = ""
    warning = ""
    row_count = 3
    best = SimpleNamespace(tput_per_gpu=10.0, conc=4, decode_tp=2)


def _unit_ctx(*, params: dict | None = None, extra: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        task=SimpleNamespace(task_id="ta-t1", kind="target_analysis",
                             params=params or {}),
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
            compare_against_gpu="MI300X", session_dir=tmp_path,
        )
        ctx = _unit_ctx()
        assert ex._resolve_session_dir(ctx) == tmp_path

    def test_falls_back_to_paths_session_dir(self, tmp_path, monkeypatch):
        ex = ta.TargetAnalysisExecutor(compare_against_gpu="MI300X")
        monkeypatch.setattr(
            "inference_optimizer.paths.session_dir", lambda: tmp_path,
        )
        ctx = _unit_ctx()
        assert ex._resolve_session_dir(ctx) == tmp_path

    def test_returns_none_when_fallback_missing(self, monkeypatch):
        ex = ta.TargetAnalysisExecutor(compare_against_gpu="MI300X")

        def boom():
            raise RuntimeError("no session")

        monkeypatch.setattr(
            "inference_optimizer.paths.session_dir", boom,
        )
        assert ex._resolve_session_dir(_unit_ctx()) is None


# ---------------------------------------------------------------------------
# Execution branches
# ---------------------------------------------------------------------------

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
    async def test_writes_skipped_summary_when_no_gpu(self, tmp_path, monkeypatch):
        ex = ta.TargetAnalysisExecutor(compare_against_gpu="")
        monkeypatch.setattr(ex, "_resolve_session_dir", lambda ctx: tmp_path)
        monkeypatch.setattr(
            "inference_optimizer.orchestrator.action_executors.target_analysis.analyze",
            lambda **kwargs: _DummySummary(),
        )
        result = await ex(_unit_ctx())
        assert result["status"] == "succeeded"
        assert result["baseline_status"] == "ok"
        # Best fields propagate through.
        assert result["best_tput_per_gpu"] == 10.0

    @pytest.mark.asyncio
    async def test_analyzer_crash_is_swallowed(self, tmp_path, monkeypatch):
        ex = ta.TargetAnalysisExecutor(compare_against_gpu="MI300X")
        monkeypatch.setattr(ex, "_resolve_session_dir", lambda ctx: tmp_path)

        def boom(**_):
            raise RuntimeError("InferenceX 500")

        monkeypatch.setattr(
            "inference_optimizer.orchestrator.action_executors.target_analysis.analyze",
            boom,
        )
        result = await ex(_unit_ctx())
        assert result["status"] == "succeeded"
        assert result["baseline_status"] == "fetch_error"
        assert "analyzer crashed" in result["note"]

    @pytest.mark.asyncio
    async def test_analyzer_crash_in_no_gpu_branch_is_swallowed(
        self, tmp_path, monkeypatch,
    ):
        ex = ta.TargetAnalysisExecutor(compare_against_gpu="")
        monkeypatch.setattr(ex, "_resolve_session_dir", lambda ctx: tmp_path)

        def boom(**_):
            raise RuntimeError("nope")

        monkeypatch.setattr(
            "inference_optimizer.orchestrator.action_executors.target_analysis.analyze",
            boom,
        )
        result = await ex(_unit_ctx())
        assert result["baseline_status"] == "fetch_error"

    @pytest.mark.asyncio
    async def test_format_result_uses_summary_without_best(
        self, tmp_path, monkeypatch,
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
            "inference_optimizer.orchestrator.action_executors.target_analysis.analyze",
            lambda **kwargs: _NoBestSummary(),
        )
        result = await ex(_unit_ctx(params={"model_path": "/m"}))
        assert result["baseline_status"] == "no_data"
        # Best metrics are absent when ``best`` is None.
        assert "best_tput_per_gpu" not in result
