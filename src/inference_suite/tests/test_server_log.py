# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract of ``server_log``: the one ready-marker table, and what the incremental log cursor reports."""

from __future__ import annotations


from inference_suite import server_log
from inference_suite.server_log import READY_MARKERS, LogCursor, throughput_samples


def test_the_ready_table_names_the_vllm_ready_line() -> None:
    assert "OpenAI server is ready to accept requests" in READY_MARKERS


def test_scan_increment_detects_vllm_rust_frontend_ready(tmp_path):
    """vLLM's Rust frontend (`vllm serve`, VLLM_USE_RUST_FRONTEND=1) runs no uvicorn; its own ready line is the ready
    marker, colour codes included, or the boot guard ends the round at its boot share."""
    log_path = tmp_path / "server.log"
    log_path.write_text(
        "\x1b[36m(RustFrontend pid=4242)\x1b[0m \x1b[32mINFO\x1b[0m 10-29 06:18:45 [server/src/lib.rs:331] "
        'OpenAI server is ready to accept requests bind_address="0.0.0.0:45369"\n'
    )
    assert server_log._scan_increment(str(log_path), 0).saw_ready is True


def test_scan_increment_detects_ready_and_progress(tmp_path):
    """The incremental scanner advances its offset and flags ready/progress
    markers only in the newly appended bytes."""
    log_path = tmp_path / "server.log"
    log_path.write_text("INFO loading weights\nApplication startup complete\n")
    first = server_log._scan_increment(str(log_path), 0)
    assert first.saw_ready is True
    assert first.saw_progress is False and first.saw_eval_start is False
    assert first.offset == log_path.stat().st_size
    # Re-scan from the advanced offset: nothing new, no re-trigger.
    second = server_log._scan_increment(str(log_path), first.offset)
    assert second.saw_ready is False and second.saw_progress is False and second.saw_eval_start is False
    assert second.offset == first.offset
    # Append a vLLM throughput line; only the new bytes are scanned.
    with log_path.open("a") as f:
        f.write("Avg generation throughput: 123.4 tokens/s, Running: 8\n")
    third = server_log._scan_increment(str(log_path), second.offset)
    assert third.saw_progress is True and third.saw_ready is False and third.saw_eval_start is False
    assert third.offset == log_path.stat().st_size
    # The eval-start marker is reported independently of ready/progress.
    with log_path.open("a") as f:
        f.write("HYPERLOOM_EVAL_START\n")
    fourth = server_log._scan_increment(str(log_path), third.offset)
    assert fourth.saw_eval_start is True and fourth.saw_ready is False and fourth.saw_progress is False
    assert fourth.offset == log_path.stat().st_size
    # An idle engine keeps printing the same line with no rate on it: the value
    # is the progress signal, not the marker.
    with log_path.open("a") as f:
        f.write("Avg generation throughput: 0.0 tokens/s, Running: 0 reqs\n")
    fifth = server_log._scan_increment(str(log_path), fourth.offset)
    assert fifth.saw_progress is False and fifth.offset == log_path.stat().st_size


def test_log_cursor_tells_the_childs_own_log_from_the_servers(tmp_path):
    """Only one of the resolved logs is written by the process being waited on.

    ``server.log`` is the inference server's; ``benchmark_stderr.log`` holds the
    benchmark body's own stderr, so the parent's pipe stays empty for the whole
    round and that file is the only place the child's own output shows up.
    Liveness that cannot tell them apart either vouches for a wedged client or
    leaves a working one unable to report.
    """
    server_log = tmp_path / "server.log"
    child_log = tmp_path / "benchmark_stderr.log"
    server_log.write_text("Application startup complete\n")
    child_log.write_text("running benchmark\n")
    passed = str(tmp_path / "server.log")
    cursor = LogCursor()
    cursor.advance(passed)

    with server_log.open("a") as f:
        f.write("[atom 22:29:41] Scheduled prefill batch: 4 reqs, 3707 new tokens\n")
    server_only = cursor.advance(passed)
    assert server_only.grew is True and server_only.child_spoke is False

    with child_log.open("a") as f:
        f.write("bench: 128/2000 requests done\n")
    child_only = cursor.advance(passed)
    assert child_only.grew is True and child_only.child_spoke is True


def test_health_probe_lines_are_not_server_activity(tmp_path):
    """An engine that died behind a live HTTP front end still answers /metrics.

    ATOM's ModelRunner crashed on a GPU memory fault while its API server kept
    serving the monitor's /metrics polls, so the log grew every few seconds and
    the silence gate never fired: lm_eval waited on 64 requests nothing would
    serve until the 7800s hard cap.
    """
    log_path = tmp_path / "server.log"
    log_path.write_text("Application startup complete\n", encoding="utf-8")
    cursor = LogCursor()
    cursor.advance(str(log_path))

    with log_path.open("a", encoding="utf-8") as f:
        f.write('INFO:     127.0.0.1:45518 - "GET /metrics HTTP/1.1" 200 OK\n')
        f.write('INFO:     127.0.0.1:45520 - "GET /health HTTP/1.1" 200 OK\n')
    assert cursor.advance(str(log_path)).grew is False

    with log_path.open("a", encoding="utf-8") as f:
        f.write('INFO:     127.0.0.1:45522 - "GET /metrics HTTP/1.1" 200 OK\n')
        f.write("[atom 23:29:42] Request 639 arrived, input tokens: 1295, pending requests: 64\n")
    assert cursor.advance(str(log_path)).grew is True


def test_scan_survives_truncation_by_rescanning_from_the_top(tmp_path):
    """A rotated log is shorter than the consumed offset; markers in the new
    content must still be seen rather than skipped past."""
    log_path = tmp_path / "server.log"
    log_path.write_text("x" * 500 + "\n", encoding="utf-8")
    first = server_log._scan_increment(str(log_path), 0)
    assert first.offset == log_path.stat().st_size

    log_path.write_text("Application startup complete\n", encoding="utf-8")
    after = server_log._scan_increment(str(log_path), first.offset)

    assert after.saw_ready is True
    assert after.offset == log_path.stat().st_size


def test_scan_reports_agentx_phase_boundaries(tmp_path):
    """aiperf prints these itself; we add no marker of our own."""
    log_path = tmp_path / "server.log"
    log_path.write_text("Phase warmup (warmup) started | target: 44 requests\n", encoding="utf-8")
    warm = server_log._scan_increment(str(log_path), 0)
    assert warm.saw_warmup_begin is True and warm.saw_measured_begin is False

    with log_path.open("a", encoding="utf-8") as f:
        f.write("Phase profiling (profiling) started | phase_index=0\n")
    measured = server_log._scan_increment(str(log_path), warm.offset)
    assert measured.saw_measured_begin is True and measured.saw_warmup_begin is False


def test_resolve_scan_logs_includes_the_agentx_client_log(tmp_path):
    """An AgentX workspace has no benchmark_stderr.log, so without this the
    phase lines are unreachable."""
    (tmp_path / "aiperf_artifacts" / "logs").mkdir(parents=True)
    (tmp_path / "server.log").write_text("up\n", encoding="utf-8")
    (tmp_path / "aiperf_artifacts" / "logs" / "aiperf.log").write_text("phases\n", encoding="utf-8")

    resolved = server_log._resolve_scan_logs(str(tmp_path / "server.log"))

    assert any(p.endswith("aiperf.log") for p in resolved)


def test_marker_split_across_a_read_boundary_is_still_seen(tmp_path):
    """Half a marker is in neither chunk, so without a carried tail the phase
    boundary is lost outright -- not seen late, never seen."""
    log_path = tmp_path / "server.log"
    log_path.write_text("Phase profiling (prof", encoding="utf-8")

    residuals: dict[str, str] = {}
    first = server_log._scan_increment(str(log_path), 0, residuals.get("k", ""))
    residuals["k"] = first.residual
    assert first.saw_measured_begin is False

    with log_path.open("a", encoding="utf-8") as f:
        f.write("iling) started | phase_index=0\n")
    second = server_log._scan_increment(str(log_path), first.offset, residuals["k"])

    assert second.saw_measured_begin is True


def test_residual_is_held_per_path(tmp_path):
    """The resolved logs have different writers; splicing one's tail into
    another's next read would invent a line neither of them wrote."""
    (tmp_path / "server.log").write_text("Phase profiling (prof", encoding="utf-8")
    (tmp_path / "benchmark_stderr.log").write_text("iling) started\n", encoding="utf-8")

    cursor = LogCursor()
    scan = cursor.advance(str(tmp_path / "server.log"))

    assert scan.saw_measured_begin is False
    assert len(cursor.residuals) >= 2


def test_truncation_drops_the_carried_tail(tmp_path):
    """The tail belonged to a file that no longer exists."""
    log_path = tmp_path / "server.log"
    log_path.write_text("Phase profiling (prof", encoding="utf-8")
    first = server_log._scan_increment(str(log_path), 0)
    assert first.residual

    log_path.write_text("iling) started\n", encoding="utf-8")
    after = server_log._scan_increment(str(log_path), 10_000, first.residual)

    assert after.saw_measured_begin is False


def test_throughput_samples_reads_sglang_and_vllm_lines_and_drops_non_positive() -> None:
    text = "\n".join(
        [
            "Decode batch. #running-req: 4 gen throughput (token/s): 100.0",
            "Decode batch. #running-req: 4 gen throughput (token/s): 200.5",
            "Avg generation throughput: 300 tokens/s",
            "irrelevant line",
            "gen throughput (token/s): 0",
        ]
    )
    assert throughput_samples(text) == [100.0, 200.5, 300.0]
