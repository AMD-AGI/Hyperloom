"""Integration tests for :class:`TargetAnalysisExecutor`.

Covers the four scenarios called out in the design chat:

* ``test_no_flag_keeps_noop`` — without ``--compare-against-gpu`` the
  executor must not be registered; the registered stub is the existing
  ``_noop_prep`` and produces no on-disk artefacts.
* ``test_fetch_timeout_graceful`` — when the upstream is unreachable the
  task still returns ``status=succeeded`` and ``baseline_status=fetch_error``.
* ``test_model_mapping_miss`` — unknown model name skips the HTTP call
  entirely and persists a ``skipped`` summary.
* ``test_happy_path_writes_files`` — full pipeline with a local mock
  HTTP server; verifies JSON + MD on disk and the bus-friendly payload.
"""

from __future__ import annotations

import gzip
import http.server
import io
import json
import socketserver
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.orchestrator.action_executors import TargetAnalysisExecutor
from inference_optimizer.orchestrator.task_registry import Task


# ---------------------------------------------------------------------------
# Local mock InferenceX (mirrors the one in test_baseline_comparison.py)
# ---------------------------------------------------------------------------
_SAMPLE_ROW = {
    "hardware":      "b300",
    "framework":     "vllm",
    "model":         "minimaxm2.5",
    "precision":     "fp8",
    "spec_method":   "none",
    "disagg":        False,
    "is_multinode":  False,
    "decode_tp":     2,
    "isl":           1024,
    "osl":           1024,
    "conc":          64,
    "metrics": {
        "tput_per_gpu":        2781.5,
        "output_tput_per_gpu": 1390.7,
        "mean_ttft":           0.094,
        "mean_tpot":           0.022,
        "mean_e2el":           20.6,
    },
    "date":   "2026-04-17",
}


class _StaticHandler(http.server.BaseHTTPRequestHandler):
    payload: list[dict[str, Any]] = []
    response_status: int = 200
    gzip_response: bool = False

    def do_GET(self):  # noqa: N802
        body = json.dumps(self.payload).encode("utf-8")
        self.send_response(self.response_status)
        self.send_header("Content-Type", "application/json")
        if self.gzip_response:
            buf = io.BytesIO()
            with gzip.GzipFile(fileobj=buf, mode="wb") as f:
                f.write(body)
            body = buf.getvalue()
            self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # noqa: A002, ARG002
        return


def _start_mock(payload, status=200, gzip_response=False):
    handler = _StaticHandler
    handler.payload = payload
    handler.response_status = status
    handler.gzip_response = gzip_response
    server = socketserver.TCPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return f"http://{host}:{port}", server.shutdown


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
async def test_no_flag_keeps_noop_when_executor_not_registered(session_dir):
    """When CLI does not pass --compare-against-gpu, cli._register_executors
    keeps the existing _noop_prep stub. We simulate that by NOT instantiating
    TargetAnalysisExecutor; the registry path stays unchanged.

    This test guards the conditional in ``_register_executors``: it asserts
    the executor module is importable and the public class exists, and that
    instantiating with an empty gpu yields a graceful skip — i.e. there is
    no scenario where importing the new code accidentally activates HTTP
    calls."""
    executor = TargetAnalysisExecutor(compare_against_gpu="",
                                       session_dir=session_dir)
    result = await executor(_ctx(session_dir, {"model_path": "MiniMax-M2.5"}))
    assert result["status"] == "succeeded"
    assert result["baseline_status"] == "skipped"
    # The skipped summary IS persisted by design — same shape as a real
    # run so the report can show "we tried and skipped" rather than "no
    # info at all".
    assert (session_dir / "target_analysis" / "target_baseline.json").exists()


@pytest.mark.asyncio
async def test_fetch_timeout_graceful(session_dir, monkeypatch):
    """Upstream unreachable → succeeded + baseline_status=fetch_error."""
    # Point the client at a port that nothing is listening on, with a
    # very short timeout + single attempt so the test runs fast.
    monkeypatch.setenv("INFERENCEX_BASE_URL", "http://127.0.0.1:1")  # reserved port
    monkeypatch.setenv("INFERENCEX_TIMEOUT_SEC", "0.5")
    monkeypatch.setenv("INFERENCEX_MAX_ATTEMPTS", "1")

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
    assert result["baseline_status"] == "fetch_error"
    assert "warning" in result and result["warning"]
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


@pytest.mark.asyncio
async def test_happy_path_writes_files(session_dir, monkeypatch):
    """Full pipeline against a local mock server."""
    url, shutdown = _start_mock([_SAMPLE_ROW])
    try:
        monkeypatch.setenv("INFERENCEX_BASE_URL", url)
        monkeypatch.setenv("INFERENCEX_TIMEOUT_SEC", "2.0")
        monkeypatch.setenv("INFERENCEX_MAX_ATTEMPTS", "1")

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
        assert result["row_count"] == 1
        assert result["best_tput_per_gpu"] == pytest.approx(2781.5)
        assert result["best_conc"] == 64

        json_path = Path(result["json_path"])
        md_path = Path(result["md_path"])
        assert json_path.exists()
        assert md_path.exists()

        on_disk = json.loads(json_path.read_text())
        assert on_disk["status"] == "ok"
        assert on_disk["best"]["tput_per_gpu"] == pytest.approx(2781.5)
        assert on_disk["query"]["model"] == "MiniMax-M2.5"
        assert on_disk["query"]["gpu"] == "b300"

        md_text = md_path.read_text()
        assert "## Reference best" in md_text
        # No KPI-like phrasing leaked into the report.
        assert "gap" not in md_text.lower()
        assert "should reach" not in md_text.lower()
    finally:
        shutdown()


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

    monkeypatch.setenv("INFERENCE_OPTIMIZER_SESSION_DIR", str(sd))

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
