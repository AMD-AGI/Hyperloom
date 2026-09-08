# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import gzip
import importlib.util
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from hyperloom.inference_optimizer.agentx.deploy import agentx_asset_dir


def _load_phase_gate():
    path = agentx_asset_dir() / "aiperf_phase_gate.py"
    spec = importlib.util.spec_from_file_location("aiperf_phase_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


phase_gate = _load_phase_gate()


class _ProgressHandler(BaseHTTPRequestHandler):
    requests_seen = 0

    def do_GET(self):  # noqa: N802
        type(self).requests_seen += 1
        if type(self).requests_seen < 3:
            payload = {"phases": {"warmup": {"start_ns": 1}}}
        else:
            payload = {"phases": {"profiling": {"start_ns": 123456789}}}
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        return


@pytest.mark.parametrize(
    "framework,body,expected",
    [
        ("sglang", '{"num_steps":8}', True),
        ("vllm", '{"num_steps":8}', False),
        ("", '{"num_steps":8}', False),
        ("sglang", '{"num_steps":true}', False),
        ("sglang", '{"num_steps":8.0}', False),
        ("sglang", '{"num_steps":"8"}', False),
        ("sglang", '{"num_steps":0}', False),
        ("sglang", '{"num_steps":-1}', False),
        ("sglang", '{"num_steps":8,"other":NaN}', False),
        ("sglang", "{}", False),
        ("sglang", "[]", False),
        ("sglang", "not JSON", False),
    ],
)
def test_auto_bounded_requires_native_positive_integer_steps(framework, body, expected):
    assert phase_gate.is_auto_bounded(framework, body) is expected


def _write_trace(path, *, rank=None, gpu=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"traceEvents": [{"cat": "kernel" if gpu else "cpu_op", "ph": "X", "ts": 1, "dur": 2}]}
    if rank is not None:
        payload["distributedInfo"] = {"rank": rank}
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return path


@pytest.mark.parametrize(
    "name",
    [
        "177-TP-0-DECODE.trace.json.gz",
        "worker-rank-0.pt.trace.json.gz",
        "worker-rank0.pt.trace.json.gz",
        "dp0_pp0_tp0_dcp0_ep0_rank0.1787293265778058798.pt.trace.json.gz",
        "rank_0/trace.pt.trace.json.gz",
        "r0.trace.json",
    ],
)
def test_current_trace_proof_supports_framework_rank_names(tmp_path, name):
    dirs = [str(tmp_path)]
    snapshot = phase_gate.snapshot_traces(dirs)
    _write_trace(tmp_path / name)
    assert phase_gate.traces_complete(dirs, snapshot, 1)


def test_current_trace_proof_accepts_rank_metadata_without_rank_filename(tmp_path):
    dirs = [str(tmp_path)]
    snapshot = phase_gate.snapshot_traces(dirs)
    _write_trace(tmp_path / "worker-a.pt.trace.json.gz", rank=0)
    _write_trace(tmp_path / "worker-b.pt.trace.json.gz", rank=1)
    assert phase_gate.traces_complete(dirs, snapshot, 2)


@pytest.mark.parametrize(
    "case",
    [
        "empty",
        "stale",
        "old_mtime",
        "partial_ranks",
        "duplicate_rank",
        "unknown_rank",
        "bad_gzip",
        "bad_json",
        "cpu_only",
        "rank_conflict",
        "bool_rank",
        "graph_capture",
    ],
)
def test_current_trace_proof_rejects_incomplete_or_ambiguous_evidence(tmp_path, case):
    dirs = [str(tmp_path)]
    if case == "stale":
        _write_trace(tmp_path / "r0.trace.json.gz")
        _write_trace(tmp_path / "r1.trace.json.gz")
    snapshot = phase_gate.snapshot_traces(dirs)
    if case not in {"empty", "stale"}:
        first = _write_trace(tmp_path / "r0.trace.json.gz")
        second = tmp_path / "r1.trace.json.gz"
        if case == "duplicate_rank":
            second = tmp_path / "worker-rank-0.trace.json.gz"
        elif case == "unknown_rank":
            second = tmp_path / "worker.trace.json.gz"
        elif case == "graph_capture":
            second = tmp_path / "capture_traces" / "graph_capture_rank1.pt.trace.json.gz"
        if case != "partial_ranks":
            _write_trace(second, gpu=case != "cpu_only")
        if case == "old_mtime":
            os.utime(first, ns=(snapshot["started_ns"] - 1, snapshot["started_ns"] - 1))
        elif case == "bad_gzip":
            second.write_bytes(second.read_bytes()[:-8])
        elif case == "bad_json":
            with gzip.open(second, "wt", encoding="utf-8") as handle:
                handle.write('{"traceEvents":[{"cat":"kernel","ph":"X","ts":1,"dur":2}]')
        elif case in {"rank_conflict", "bool_rank"}:
            _write_trace(second, rank=0 if case == "rank_conflict" else True)
    assert not phase_gate.traces_complete(dirs, snapshot, 2)


def test_current_trace_proof_accepts_rewritten_path_but_requires_known_tp(tmp_path):
    dirs = [str(tmp_path)]
    path = _write_trace(tmp_path / "r0.trace.json")
    snapshot = phase_gate.snapshot_traces(dirs)
    _write_trace(path)
    os.utime(path, ns=(snapshot["started_ns"] + 1, snapshot["started_ns"] + 1))
    assert phase_gate.traces_complete(dirs, snapshot, 1)
    assert not phase_gate.traces_complete(dirs, snapshot, 0)


def test_pick_loopback_port_returns_available_port():
    port = phase_gate.pick_loopback_port()
    assert 0 < port < 65536


def test_wait_for_phase_ignores_warmup_until_profiling_starts():
    _ProgressHandler.requests_seen = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ProgressHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        start_ns = phase_gate.wait_for_phase(
            api_url=f"http://127.0.0.1:{server.server_port}",
            phase="profiling",
            pid=os.getpid(),
            timeout_seconds=2,
            poll_interval_seconds=0.01,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert start_ns == 123456789
    assert _ProgressHandler.requests_seen >= 3


def test_wait_for_phase_fails_when_aiperf_process_exits():
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait(timeout=5)
    with pytest.raises(RuntimeError, match="exited before phase"):
        phase_gate.wait_for_phase(
            api_url="http://127.0.0.1:1",
            phase="profiling",
            pid=proc.pid,
            timeout_seconds=1,
            poll_interval_seconds=0.01,
        )


def test_wait_for_phase_times_out_without_phase():
    class EmptyProgressHandler(_ProgressHandler):
        def do_GET(self):  # noqa: N802
            body = b'{"phases":{}}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), EmptyProgressHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(TimeoutError, match="timed out"):
            phase_gate.wait_for_phase(
                api_url=f"http://127.0.0.1:{server.server_port}",
                phase="profiling",
                pid=os.getpid(),
                timeout_seconds=0.05,
                poll_interval_seconds=0.01,
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_capture_stops_when_phase_completes():
    class PhaseCompletionHandler(_ProgressHandler):
        requests_seen = 0

        def do_GET(self):  # noqa: N802
            type(self).requests_seen += 1
            body = json.dumps(
                {
                    "phases": {
                        "profiling": {
                            "start_ns": 1,
                            "requests_end_ns": (123456789 if type(self).requests_seen >= 3 else None),
                        }
                    }
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), PhaseCompletionHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = phase_gate.wait_for_capture_stop(
            api_url=f"http://127.0.0.1:{server.server_port}",
            phase="profiling",
            pid=os.getpid(),
            max_window_seconds=2,
            poll_interval_seconds=0.01,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert result["stop_reason"] == "phase_complete"


def test_capture_stops_at_wall_clock_limit_without_phase_completion():
    class NoCoverageHandler(_ProgressHandler):
        def do_GET(self):  # noqa: N802
            body = b'{"phases":{"profiling":{"start_ns":1,"requests_completed":0,"requests_end_ns":null}}}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), NoCoverageHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = phase_gate.wait_for_capture_stop(
            api_url=f"http://127.0.0.1:{server.server_port}",
            phase="profiling",
            pid=os.getpid(),
            max_window_seconds=0.05,
            poll_interval_seconds=0.01,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert result["stop_reason"] == "wall_clock_limit"


def test_process_alive_rejects_zombie_state(monkeypatch):
    monkeypatch.setattr(
        phase_gate.Path,
        "read_text",
        lambda _self, **_kwargs: "42 (python) Z 1 2 3",
    )
    assert phase_gate.process_alive(42) is False


def test_wait_for_phase_retries_transient_http_protocol_errors(monkeypatch):
    responses = iter(
        [
            phase_gate.http.client.BadStatusLine("partial"),
            {"start_ns": 123},
        ]
    )

    def _phase_stats(*_args, **_kwargs):
        value = next(responses)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(phase_gate, "phase_stats", _phase_stats)
    monkeypatch.setattr(phase_gate, "process_alive", lambda _pid: True)
    assert (
        phase_gate.wait_for_phase(
            api_url="http://127.0.0.1:1",
            phase="profiling",
            pid=42,
            timeout_seconds=1,
            poll_interval_seconds=0.01,
        )
        == 123
    )


def test_capture_stop_records_transient_api_errors(monkeypatch):
    responses = iter(
        [
            phase_gate.http.client.IncompleteRead(b"partial"),
            {"requests_end_ns": 123},
        ]
    )

    def _phase_stats(*_args, **_kwargs):
        value = next(responses)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(phase_gate, "phase_stats", _phase_stats)
    monkeypatch.setattr(phase_gate, "process_alive", lambda _pid: True)
    result = phase_gate.wait_for_capture_stop(
        api_url="http://127.0.0.1:1",
        phase="profiling",
        pid=42,
        max_window_seconds=1,
        poll_interval_seconds=0.01,
    )
    assert result["stop_reason"] == "phase_complete"
    assert result["api_error_count"] == 1
    assert "IncompleteRead" in result["last_api_error"]


def test_write_capture_status_is_structured_and_atomic(tmp_path):
    output = tmp_path / "agentx_profile_capture.json"
    phase_gate.write_capture_status(
        output=str(output),
        capture_id="capture-1",
        status="succeeded",
        reason="capture_complete",
        phase_start_ns=123,
        requested_window_seconds=20,
        decision_json='{"stop_reason":"wall_clock_limit"}',
    )
    payload = json.loads(output.read_text())
    assert payload["capture_id"] == "capture-1"
    assert payload["status"] == "succeeded"
    assert payload["phase_start_ns"] == 123
    assert payload["decision"]["stop_reason"] == "wall_clock_limit"
    assert not list(tmp_path.glob(".agentx_profile_capture.*"))
