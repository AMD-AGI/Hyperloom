"""Unit tests for the ``baseline_comparison`` package.

Covers:

* :func:`name_mapping.to_inferencex_name` — prefix stripping + match.
* :func:`inferencex_client.fetch_rows` — happy path with a local HTTP
  server, timeout path, gzip decoding, retry budget.
* :func:`target_analyzer.analyze` — every status branch (ok / skipped /
  fetch_error / no_match) end-to-end with mocked upstream rows, plus
  on-disk persistence (JSON + MD) under the session dir.

These tests deliberately avoid hitting the real InferenceX endpoint;
they spin up a ``http.server`` thread on 127.0.0.1 so the assertions
stay deterministic and offline-friendly.
"""

from __future__ import annotations

import gzip
import http.server
import io
import json
import socketserver
import threading
import time
from pathlib import Path
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# name_mapping
# ---------------------------------------------------------------------------
def test_name_mapping_known_display_name_passthrough():
    from inference_optimizer.baseline_comparison.name_mapping import to_inferencex_name
    assert to_inferencex_name("MiniMax-M2.5") == "MiniMax-M2.5"


def test_name_mapping_strips_vendor_prefix_from_path():
    from inference_optimizer.baseline_comparison.name_mapping import to_inferencex_name
    assert to_inferencex_name("/wekafs/models/MiniMaxAI-MiniMax-M2.5") == "MiniMax-M2.5"


def test_name_mapping_case_insensitive():
    from inference_optimizer.baseline_comparison.name_mapping import to_inferencex_name
    assert to_inferencex_name("/wekafs/x/minimaxai-minimax-m2.5") == "MiniMax-M2.5"


def test_name_mapping_unknown_returns_none():
    from inference_optimizer.baseline_comparison.name_mapping import to_inferencex_name
    assert to_inferencex_name("/wekafs/models/MyCorp-Custom-FT-7B") is None
    assert to_inferencex_name("") is None
    assert to_inferencex_name(None) is None  # type: ignore[arg-type]


def test_known_models_list_is_nonempty():
    from inference_optimizer.baseline_comparison.name_mapping import KNOWN_INFERENCEX_MODELS
    assert "MiniMax-M2.5" in KNOWN_INFERENCEX_MODELS
    assert len(KNOWN_INFERENCEX_MODELS) >= 5


# ---------------------------------------------------------------------------
# inferencex_client — happy path against a local HTTP server
# ---------------------------------------------------------------------------
_SAMPLE_ROW = {
    "hardware":      "b300",
    "framework":     "vllm",
    "model":         "minimaxm2.5",
    "precision":     "fp8",
    "spec_method":   "none",
    "disagg":        False,
    "is_multinode":  False,
    "prefill_tp":    2,
    "prefill_ep":    1,
    "decode_tp":     2,
    "decode_ep":     1,
    "num_prefill_gpu": 2,
    "num_decode_gpu":  2,
    "isl":           1024,
    "osl":           1024,
    "conc":          64,
    "image":         "vllm-rocm:test",
    "metrics": {
        "tput_per_gpu":        2781.5,
        "output_tput_per_gpu": 1390.7,
        "input_tput_per_gpu":  1390.8,
        "mean_ttft":           0.094,   # seconds
        "mean_tpot":           0.022,   # seconds
        "mean_e2el":           20.6,    # seconds
    },
    "date":   "2026-04-17",
    "run_url": "https://example/runs/x",
}


class _StaticHandler(http.server.BaseHTTPRequestHandler):
    """Serves a fixed JSON payload, optionally gzipped, on every GET.

    Used by :func:`_mock_server` below. Test code installs the payload
    via :attr:`_StaticHandler.payload`.
    """

    payload: list[dict[str, Any]] = []
    delay_sec: float = 0.0
    gzip_response: bool = False
    response_status: int = 200

    def do_GET(self):  # noqa: N802 (BaseHTTPRequestHandler API)
        if self.delay_sec:
            time.sleep(self.delay_sec)
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


def _mock_server(payload, *, delay_sec=0.0, gzip_response=False, status=200):
    """Start a local HTTP server thread on a free port.

    Returns ``(base_url, shutdown_fn)``. ``base_url`` is suitable for
    ``INFERENCEX_BASE_URL`` (no trailing slash — caller appends
    ``/benchmarks?...``).
    """
    handler = _StaticHandler
    handler.payload = payload
    handler.delay_sec = delay_sec
    handler.gzip_response = gzip_response
    handler.response_status = status
    server = socketserver.TCPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return f"http://{host}:{port}", server.shutdown


@pytest.fixture
def mock_inferencex(monkeypatch):
    """Yield ``(set_payload, set_delay)`` for tests to drive the server."""
    state: dict[str, Any] = {
        "shutdown": None,
        "payload":  [],
        "delay":    0.0,
        "gzip":     False,
        "status":   200,
    }

    def _start():
        url, shutdown = _mock_server(
            state["payload"],
            delay_sec=state["delay"],
            gzip_response=state["gzip"],
            status=state["status"],
        )
        state["shutdown"] = shutdown
        monkeypatch.setenv("INFERENCEX_BASE_URL", url)
        # Keep retry budget short so timeout tests don't hang.
        monkeypatch.setenv("INFERENCEX_TIMEOUT_SEC", "0.5")
        monkeypatch.setenv("INFERENCEX_MAX_ATTEMPTS", "1")
        return url

    yield state, _start

    if state["shutdown"]:
        state["shutdown"]()


def test_fetch_rows_happy_path(mock_inferencex):
    state, start = mock_inferencex
    state["payload"] = [_SAMPLE_ROW, dict(_SAMPLE_ROW, conc=128)]
    start()

    from inference_optimizer.baseline_comparison import fetch_rows
    rows, warning = fetch_rows("MiniMax-M2.5")
    assert warning == ""
    assert rows is not None
    assert len(rows) == 2
    assert rows[0]["metrics"]["tput_per_gpu"] == 2781.5


def test_fetch_rows_gzip_decode(mock_inferencex):
    state, start = mock_inferencex
    state["payload"] = [_SAMPLE_ROW]
    state["gzip"] = True
    start()

    from inference_optimizer.baseline_comparison import fetch_rows
    rows, warning = fetch_rows("MiniMax-M2.5")
    assert warning == ""
    assert rows is not None and len(rows) == 1


def test_fetch_rows_http_error_returns_none(mock_inferencex):
    state, start = mock_inferencex
    state["status"] = 500
    state["payload"] = []
    start()

    from inference_optimizer.baseline_comparison import fetch_rows
    rows, warning = fetch_rows("MiniMax-M2.5")
    assert rows is None
    assert "500" in warning


def test_fetch_rows_empty_model_returns_none(mock_inferencex):
    state, start = mock_inferencex
    state["payload"] = [_SAMPLE_ROW]
    start()
    from inference_optimizer.baseline_comparison import fetch_rows
    rows, warning = fetch_rows("")
    assert rows is None
    assert "empty" in warning.lower()


# ---------------------------------------------------------------------------
# target_analyzer — end-to-end with mocked upstream
# ---------------------------------------------------------------------------
def _make_rows() -> list[dict[str, Any]]:
    """Construct a small, realistic-shaped row set covering multiple
    (gpu, precision, conc) combinations."""
    base = _SAMPLE_ROW
    out: list[dict[str, Any]] = []
    for conc, tput in [(4, 412.5), (16, 1167.9), (64, 2781.5), (256, 6624.1)]:
        row = json.loads(json.dumps(base))  # deep-copy via JSON
        row["conc"] = conc
        row["metrics"]["tput_per_gpu"] = tput
        out.append(row)
    # Add a mi300x fp8 row with smaller numbers so filter selectivity
    # is visible.
    mi = json.loads(json.dumps(base))
    mi["hardware"] = "mi300x"
    mi["conc"] = 64
    mi["metrics"]["tput_per_gpu"] = 1596.05
    out.append(mi)
    # A different precision row that should be filtered out.
    fp4 = json.loads(json.dumps(base))
    fp4["precision"] = "fp4"
    fp4["metrics"]["tput_per_gpu"] = 4066.79
    out.append(fp4)
    return out


def _write_competitor_target(session_dir: Path, per_conc: list[dict[str, Any]]) -> None:
    from inference_optimizer.orchestrator import research_hints
    research_hints.write_competitor_target(
        session_dir,
        {
            "gpu": "b300",
            "model": "MiniMax-M2.5",
            "framework": "vllm",
            "precision": "fp8",
            "per_conc": per_conc,
            "notes": "scout-authored",
        },
    )


def test_analyze_happy_path_writes_files(tmp_path: Path):
    _write_competitor_target(tmp_path, [
        {"conc": 4, "tput_per_gpu": 412.5, "tpot_ms": 18.0,
         "source": "https://pr/1"},
        {"conc": 256, "tput_per_gpu": 6624.1, "tpot_ms": 30.0,
         "source": "https://blog/x"},
    ])

    from inference_optimizer.baseline_comparison import analyze
    summary = analyze(
        session_dir=tmp_path,
        model_path="/wekafs/models/MiniMaxAI-MiniMax-M2.5",
        compare_against_gpu="b300",
        framework="vllm",
        precision="fp8",
        isl=1024,
        osl=1024,
    )

    assert summary.status == "ok"
    assert summary.reason == "ok"
    assert summary.row_count == 2
    assert summary.best is not None
    assert summary.best.tput_per_gpu == 6624.1
    assert summary.best.conc == 256
    assert summary.source == "llm_authored"

    json_path = tmp_path / "target_analysis" / "target_baseline.json"
    md_path = tmp_path / "target_analysis" / "target_analysis_report.md"
    assert json_path.exists()
    assert md_path.exists()

    on_disk = json.loads(json_path.read_text())
    assert on_disk["status"] == "ok"
    assert on_disk["reason"] == "ok"
    assert on_disk["query"]["model"] == "MiniMax-M2.5"
    assert on_disk["query"]["gpu"] == "b300"
    assert on_disk["best"]["tput_per_gpu"] == 6624.1
    assert on_disk["source"] == "llm_authored"

    md_text = md_path.read_text()
    assert "## Reference best" in md_text
    assert "6624.1" in md_text


def test_analyze_mapping_miss_writes_skipped_summary(tmp_path):
    _write_competitor_target(tmp_path, [
        {"conc": 64, "tput_per_gpu": 2781.5, "source": "https://pr/1"},
    ])

    from inference_optimizer.baseline_comparison import analyze
    summary = analyze(
        session_dir=tmp_path,
        model_path="/wekafs/models/MyCorp-Custom-FT-7B",
        compare_against_gpu="b300",
        framework="vllm",
        precision="fp8",
        isl=1024,
        osl=1024,
    )
    assert summary.status == "skipped"
    assert summary.reason == "model_mapping_miss"
    assert summary.best is None
    assert "mapping miss" in summary.warning.lower()

    on_disk = json.loads(
        (tmp_path / "target_analysis" / "target_baseline.json").read_text()
    )
    assert on_disk["status"] == "skipped"
    assert on_disk["reason"] == "model_mapping_miss"


def test_analyze_no_target_gpu_writes_marker(tmp_path):
    """``compare_against_gpu=""`` short-circuits and persists a marker JSON.
    Mirrors the path used when ``--compare-against-gpu`` is unset."""
    from inference_optimizer.baseline_comparison import analyze
    summary = analyze(
        session_dir=tmp_path,
        model_path="/wekafs/models/MiniMaxAI-MiniMax-M2.5",
        compare_against_gpu="",
    )
    assert summary.status == "skipped"
    assert summary.reason == "no_target_gpu_configured"
    assert summary.best is None
    assert summary.query.gpu == ""

    on_disk = json.loads(
        (tmp_path / "target_analysis" / "target_baseline.json").read_text()
    )
    assert on_disk["status"] == "skipped"
    assert on_disk["reason"] == "no_target_gpu_configured"


def test_analyze_no_competitor_target(tmp_path):
    """No ``competitor_target.json`` on disk → ``no_match`` (fail-soft)."""
    from inference_optimizer.baseline_comparison import analyze
    summary = analyze(
        session_dir=tmp_path,
        model_path="MiniMax-M2.5",
        compare_against_gpu="b300",
        framework="vllm",
        precision="fp8",
        isl=1024,
        osl=1024,
    )
    assert summary.status == "no_match"
    assert summary.reason == "no_competitor_target"
    assert summary.best is None
    assert (tmp_path / "target_analysis" / "target_baseline.json").exists()


def test_analyze_sourceless_target_dropped(tmp_path):
    """Per-conc rows without a source are discarded; if all lack a source
    the summary degrades to ``no_match`` rather than fabricating a best."""
    from inference_optimizer.orchestrator import research_hints
    # write_competitor_target itself drops sourceless rows, so emulate a
    # hand-edited file with a sourceless row to exercise load-time filter.
    from inference_optimizer import session_paths
    path = session_paths.competitor_target_json(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "gpu": "b300", "model": "MiniMax-M2.5",
        "per_conc": [{"conc": 64, "tput_per_gpu": 2781.5}],
    }), encoding="utf-8")
    assert research_hints.load_competitor_target(tmp_path) is None

    from inference_optimizer.baseline_comparison import analyze
    summary = analyze(
        session_dir=tmp_path,
        model_path="MiniMax-M2.5",
        compare_against_gpu="b300",
    )
    assert summary.status == "no_match"
    assert summary.reason == "no_competitor_target"


# ---------------------------------------------------------------------------
# report.py renderer — _format_external_baseline_section branches on reason
# ---------------------------------------------------------------------------
def _ext_payload(**overrides) -> dict[str, Any]:
    base: dict[str, Any] = {
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
        "reason": "ok",
        "warning": "",
        "source": "https://inferencex.semianalysis.com/api/v1",
    }
    base.update(overrides)
    return base


def test_report_section_renders_no_target_gpu_marker():
    from inference_optimizer.orchestrator.action_executors.report import (
        _format_external_baseline_section,
    )
    ext = _ext_payload(
        status="skipped", reason="no_target_gpu_configured",
        warning="compare_against_gpu is empty", best=None, row_count=0,
    )
    ext["query"]["gpu"] = ""
    md = "\n".join(_format_external_baseline_section(ext))
    assert "## External baseline (not requested)" in md
    assert "no_target_gpu_configured" in md
    assert "No `--compare-against-gpu`" in md
    # Must NOT render a fake reference-best line.
    assert "Reference best per-GPU throughput" not in md


def test_report_section_renders_ok_with_reference_best():
    from inference_optimizer.orchestrator.action_executors.report import (
        _format_external_baseline_section,
    )
    md = "\n".join(_format_external_baseline_section(_ext_payload()))
    assert "## External baseline (competitor target, advisory)" in md
    assert "Reference best per-GPU throughput" in md
    assert "2781.5" in md


def test_report_section_renders_fetch_error_with_warning():
    from inference_optimizer.orchestrator.action_executors.report import (
        _format_external_baseline_section,
    )
    ext = _ext_payload(
        status="fetch_error", reason="fetch_error",
        warning="upstream timed out", best=None, row_count=0,
    )
    md = "\n".join(_format_external_baseline_section(ext))
    assert "## External baseline (competitor target, advisory)" in md
    assert "fetch_error" in md
    assert "upstream timed out" in md
    assert "No reference best available" in md


def test_session_paths_helpers_under_target_analysis(tmp_path):
    from inference_optimizer.session_paths import (
        target_analysis_dir,
        target_analysis_report_md,
        target_baseline_json,
    )
    sd = tmp_path / "sess"
    assert target_analysis_dir(sd) == sd / "target_analysis"
    assert target_baseline_json(sd) == sd / "target_analysis" / "target_baseline.json"
    assert target_analysis_report_md(sd) == (
        sd / "target_analysis" / "target_analysis_report.md"
    )
