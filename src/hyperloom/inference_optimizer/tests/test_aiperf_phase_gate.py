# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import errno
import gzip
import importlib.util
import json
import os
import subprocess
import sys
import textwrap
import threading
import time
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
    now_ns = time.time_ns()
    os.utime(path, ns=(now_ns, now_ns))
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


@pytest.mark.parametrize(
    "name",
    [
        "host_12345.1700.pt.trace.json.gz",
        "vllmpip-7f96d6bc84-abcde_12345.1700000000000000000.pt.trace.json.gz",
        "async_llm_12345.1700000000000000000.pt.trace.json.gz",
    ],
)
def test_current_trace_proof_accepts_single_unranked_tp1(tmp_path, name):
    dirs = [str(tmp_path)]
    snapshot = phase_gate.snapshot_traces(dirs)
    _write_trace(tmp_path / name)
    assert phase_gate.traces_complete(dirs, snapshot, 1)


@pytest.mark.parametrize(
    "gpu_names",
    [("rank0.pt.trace.json.gz",), ("worker.pt.trace.json.gz",), ("rank0.pt.trace.json.gz", "rank1.pt.trace.json.gz")],
)
@pytest.mark.parametrize(
    "frontend_name", ["host_84217.async_llm.1787731415290283310.pt.trace.json.gz", "frontend-rank0.pt.trace.json.gz"]
)
def test_current_trace_proof_accepts_gpu_ranks_with_cpu_frontend(tmp_path, gpu_names, frontend_name):
    dirs = [str(tmp_path)]
    snapshot = phase_gate.snapshot_traces(dirs)
    _write_trace(tmp_path / frontend_name, gpu=False)
    for name in gpu_names:
        _write_trace(tmp_path / name)
    assert len(phase_gate.current_traces(dirs, snapshot)) == len(gpu_names) + 1
    assert phase_gate.traces_complete(dirs, snapshot, len(gpu_names))


def test_cpu_frontend_cache_rechecks_file_when_gpu_events_appear(tmp_path, trace_metadata_reads):
    dirs = [str(tmp_path)]
    snapshot = phase_gate.snapshot_traces(dirs)
    _write_trace(tmp_path / "rank0.pt.trace.json.gz")
    frontend = _write_trace(tmp_path / "host.async_llm.pt.trace.json.gz", gpu=False)
    cache = {}
    assert phase_gate.traces_complete(dirs, snapshot, 1, cache=cache)
    assert phase_gate.traces_complete(dirs, snapshot, 1, cache=cache)
    assert len(trace_metadata_reads) == 2
    _write_trace(frontend, gpu=True)
    assert not phase_gate.traces_complete(dirs, snapshot, 1, cache=cache)
    assert len(trace_metadata_reads) == 3


@pytest.mark.parametrize(
    "case", ["missing_gpu_rank", "duplicate_gpu_rank", "unranked_gpu_tp2", "truncated_frontend", "cpu_only"]
)
def test_cpu_frontend_does_not_mask_incomplete_or_ambiguous_gpu_traces(tmp_path, case):
    dirs = [str(tmp_path)]
    snapshot = phase_gate.snapshot_traces(dirs)
    frontend = _write_trace(tmp_path / "host.async_llm.pt.trace.json.gz", gpu=False)
    if case != "cpu_only":
        _write_trace(tmp_path / "rank0.pt.trace.json.gz")
    if case == "duplicate_gpu_rank":
        _write_trace(tmp_path / "worker-rank0.pt.trace.json.gz")
    elif case == "unranked_gpu_tp2":
        _write_trace(tmp_path / "worker.pt.trace.json.gz")
    elif case == "truncated_frontend":
        frontend.write_bytes(frontend.read_bytes()[:-8])
        now_ns = time.time_ns()
        os.utime(frontend, ns=(now_ns, now_ns))
        assert str(frontend.resolve()) in phase_gate.current_traces(dirs, snapshot)
    tp = 2 if case in {"missing_gpu_rank", "unranked_gpu_tp2"} else 1
    assert not phase_gate.traces_complete(dirs, snapshot, tp)


@pytest.mark.parametrize("case", ["tp2", "explicit_rank", "rank_conflict", "multiple_unranked"])
def test_current_trace_proof_rejects_ambiguous_unranked_fallback(tmp_path, case):
    dirs = [str(tmp_path)]
    snapshot = phase_gate.snapshot_traces(dirs)
    name = "r0.trace.json.gz" if case == "rank_conflict" else "host_12345.1700.pt.trace.json.gz"
    _write_trace(tmp_path / name, rank=1 if case in {"explicit_rank", "rank_conflict"} else None)
    if case in {"tp2", "multiple_unranked"}:
        _write_trace(tmp_path / "host_54321.1700.pt.trace.json.gz")
    assert not phase_gate.traces_complete(dirs, snapshot, 2 if case == "tp2" else 1)


def _write_trace_text(path, text):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as handle:
        handle.write(text)
    now_ns = time.time_ns()
    os.utime(path, ns=(now_ns, now_ns))
    return path


@pytest.mark.parametrize("suffix", [".json", ".json.gz"])
def test_current_trace_proof_streams_trace_with_bounded_reads(tmp_path, monkeypatch, suffix):
    dirs = [str(tmp_path)]
    snapshot = phase_gate.snapshot_traces(dirs)
    events = [{"cat": "cpu_op", "ph": "X", "args": {}} for _ in range(2000)]
    events.append(
        {
            "cat": "kernel",
            "ph": "X",
            "args": {"nested": [None, True, -1.25e3, {"text": '\\"traceEvents\\": [} \\ end\n' * 100}]},
        }
    )
    payload = {
        "unknown": {"nested": [1, {"traceEvents": "not an event array"}]},
        "traceEvents": events,
        "distributedInfo": {"rank": 0},
        "tail": [False, None, {"empty": {}}],
    }
    path = _write_trace_text(tmp_path / f"worker.trace{suffix}", json.dumps(payload))
    original_open = gzip.open if suffix.endswith(".gz") else open
    reads = []

    class BoundedReader:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self.handle, name)

        def read(self, size=-1):
            assert 0 < size <= 1024 * 1024, "trace reads must be bounded"
            reads.append(size)
            return self.handle.read(min(size, 37))

    def bounded_open(*args, **kwargs):
        return BoundedReader(original_open(*args, **kwargs))

    def reject_trace_json_load(handle, **kwargs):
        pytest.fail("trace validation must not materialize the document with json.load")

    monkeypatch.setattr(phase_gate.json, "load", reject_trace_json_load)
    if suffix.endswith(".gz"):
        monkeypatch.setattr(phase_gate.gzip, "open", bounded_open)
    else:
        monkeypatch.setattr(phase_gate, "open", bounded_open, raising=False)
    assert phase_gate.traces_complete(dirs, snapshot, 1)
    assert len(reads) > 100
    assert path.exists()


@pytest.mark.parametrize("suffix", [".json", ".json.gz"])
@pytest.mark.parametrize(
    "number,split",
    [
        ("0", 1),
        ("-0", 1),
        ("12345", 2),
        ("-12.5", 3),
        ("-12.5", 4),
        ("1e+10", 1),
        ("1e+10", 2),
        ("1e+10", 3),
        ("1e+10", 4),
        ("-1.25E-10", 6),
        ("-1.25E-10", 7),
    ],
)
def test_current_trace_proof_accepts_numeric_chunk_boundaries(tmp_path, monkeypatch, suffix, number, split):
    dirs = [str(tmp_path)]
    snapshot = phase_gate.snapshot_traces(dirs)
    prefix = '{"traceEvents":[{"cat":"kernel","ph":"X"}],"metadata":['
    text = prefix + number + "]}"
    _write_trace_text(tmp_path / f"r0.trace{suffix}", text)
    original_open = gzip.open if suffix.endswith(".gz") else open
    boundary = len(prefix) + split
    reads = []

    class SplitReader:
        def __init__(self, handle):
            self.handle = handle
            self.position = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def read(self, size=-1):
            assert size > 0
            if self.position < boundary:
                size = min(size, boundary - self.position)
            chunk = self.handle.read(size)
            self.position += len(chunk)
            reads.append(self.position)
            return chunk

    def split_open(*args, **kwargs):
        return SplitReader(original_open(*args, **kwargs))

    if suffix.endswith(".gz"):
        monkeypatch.setattr(phase_gate.gzip, "open", split_open)
    else:
        monkeypatch.setattr(phase_gate, "open", split_open, raising=False)
    assert phase_gate.traces_complete(dirs, snapshot, 1)
    assert boundary in reads
    assert reads[-1] == len(text)


def test_current_trace_proof_accepts_trace_events_as_final_member(tmp_path):
    dirs = [str(tmp_path)]
    snapshot = phase_gate.snapshot_traces(dirs)
    _write_trace_text(
        tmp_path / "worker.trace.json",
        '{"distributedInfo":{"rank":0},"metadata":{"unknown":[{},[]]},'
        '"traceEvents":[{"cat":"kernel","ph":"X","args":{"nested":[1,{"value":true}]}}]}',
    )
    assert phase_gate.traces_complete(dirs, snapshot, 1)


@pytest.mark.parametrize(
    "text",
    [
        '{"traceEvents":[{"cat":"kernel","ph":"X"},]}',
        '{"traceEvents":[{"cat":"kernel","ph":"X"}],}',
        '{"traceEvents":[{"cat":"kernel","ph":"X"}]} trailing garbage',
        '{"traceEvents":[{"cat":"kernel","ph":"X"}]} {}',
        '{"traceEvents":[{"cat":"cpu_op","ph":"X","args":{"cat":"kernel","ph":"X"}}]}',
        json.dumps({"traceEvents": [], "metadata": '{"cat":"kernel","ph":"X"}'}),
    ],
    ids=[
        "event_trailing_comma",
        "object_trailing_comma",
        "garbage",
        "second_document",
        "nested_kernel",
        "kernel_string",
    ],
)
def test_current_trace_proof_rejects_invalid_or_false_kernel_evidence(tmp_path, text):
    dirs = [str(tmp_path)]
    snapshot = phase_gate.snapshot_traces(dirs)
    _write_trace_text(tmp_path / "r0.trace.json.gz", text)
    assert not phase_gate.traces_complete(dirs, snapshot, 1)


def test_current_trace_proof_rejects_truncated_gzip_after_valid_events(tmp_path):
    dirs = [str(tmp_path)]
    snapshot = phase_gate.snapshot_traces(dirs)
    path = _write_trace(tmp_path / "r0.trace.json.gz")
    path.write_bytes(path.read_bytes()[:-8])
    now_ns = time.time_ns()
    os.utime(path, ns=(now_ns, now_ns))
    assert not phase_gate.traces_complete(dirs, snapshot, 1)


_ONE_MIB = 1024 * 1024


def _repeat_to(unit, target):
    return unit * (target // len(unit) + 1)


def _enforce_bounded_reads(monkeypatch, suffix):
    """Fail the test if trace validation gives up streaming and swallows the whole document."""
    original_open = gzip.open if suffix.endswith(".gz") else open
    sizes = []

    class BoundedReader:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self.handle, name)

        def read(self, size=-1):
            assert 0 < size <= _ONE_MIB, "trace reads must stay bounded"
            sizes.append(size)
            return self.handle.read(size)

    def bounded_open(*args, **kwargs):
        return BoundedReader(original_open(*args, **kwargs))

    def reject_trace_json_load(handle, **kwargs):
        pytest.fail("trace validation must not materialize the document with json.load")

    monkeypatch.setattr(phase_gate.json, "load", reject_trace_json_load)
    if suffix.endswith(".gz"):
        monkeypatch.setattr(phase_gate.gzip, "open", bounded_open)
    else:
        monkeypatch.setattr(phase_gate, "open", bounded_open, raising=False)
    return sizes


def _oversized_valid_payload(case):
    kernel = {"cat": "kernel", "ph": "X", "ts": 1, "dur": 2}
    if case == "huge_event_string":
        stack = _repeat_to('frame "f" \\ /src/kernel.cpp:42\n', 2 * _ONE_MIB)
        return {"traceEvents": [dict(kernel, args={"Call stack": stack})]}
    if case == "many_small_values":
        args = {"values": list(range(400000)), "flags": [True, False, None] * 1000}
        return {"traceEvents": [dict(kernel, args=args)]}
    metadata = _repeat_to("unknown vendor metadata; ", 2 * _ONE_MIB)
    return {"unknown": metadata, "traceEvents": [kernel], "distributedInfo": {"rank": 0}}


@pytest.mark.parametrize("suffix", [".json", ".json.gz"])
@pytest.mark.parametrize("case", ["huge_event_string", "many_small_values", "huge_unknown_metadata"])
def test_current_trace_proof_accepts_valid_values_larger_than_the_buffer_budget(tmp_path, monkeypatch, suffix, case):
    dirs = [str(tmp_path)]
    snapshot = phase_gate.snapshot_traces(dirs)
    text = json.dumps(_oversized_valid_payload(case))
    assert len(text) > 2 * _ONE_MIB
    _write_trace_text(tmp_path / f"r0.trace{suffix}", text)
    sizes = _enforce_bounded_reads(monkeypatch, suffix)
    assert phase_gate.traces_complete(dirs, snapshot, 1)
    assert len(sizes) > 1


def _oversized_invalid_text(case):
    filler = "x" * (2 * _ONE_MIB)
    kernel_first = '{"traceEvents":[{"cat":"kernel","ph":"X","ts":1,"dur":2},'
    if case == "bad_escape":
        return kernel_first + '{"cat":"cpu_op","ph":"X","args":{"stack":"' + filler + '\\q"}}]}'
    if case == "control_char":
        return kernel_first + '{"cat":"cpu_op","ph":"X","args":{"stack":"' + filler + '\x01"}}]}'
    if case == "truncated_string":
        return kernel_first + '{"cat":"cpu_op","ph":"X","args":{"stack":"' + filler
    if case == "false_kernel_in_string":
        return (
            '{"traceEvents":[{"cat":"cpu_op","ph":"X","args":{"stack":"'
            + filler
            + '{\\"cat\\":\\"kernel\\",\\"ph\\":\\"X\\"}"}}]}'
        )
    return '{"traceEvents":[{"cat":"cpu_op","ph":"X","args":{"cat":"kernel","ph":"X","pad":"' + filler + '"}}]}'


@pytest.mark.parametrize("suffix", [".json", ".json.gz"])
@pytest.mark.parametrize(
    "case",
    [
        "bad_escape",
        "control_char",
        "truncated_string",
        "false_kernel_in_string",
        "nested_kernel_in_oversized_event",
    ],
)
def test_current_trace_proof_rejects_oversized_invalid_or_false_kernel_evidence(tmp_path, monkeypatch, suffix, case):
    dirs = [str(tmp_path)]
    snapshot = phase_gate.snapshot_traces(dirs)
    _write_trace_text(tmp_path / f"r0.trace{suffix}", _oversized_invalid_text(case))
    sizes = _enforce_bounded_reads(monkeypatch, suffix)
    assert not phase_gate.traces_complete(dirs, snapshot, 1)
    assert len(sizes) > 1


@pytest.fixture
def trace_metadata_reads(monkeypatch):
    original = phase_gate._read_trace_metadata
    reads = []

    def read_metadata(path, *, deadline):
        reads.append((path, deadline))
        return original(path, deadline=deadline)

    monkeypatch.setattr(phase_gate, "_read_trace_metadata", read_metadata)
    return reads


def test_current_trace_proof_cache_reuses_unchanged_metadata(tmp_path, trace_metadata_reads):
    dirs = [str(tmp_path)]
    snapshot = phase_gate.snapshot_traces(dirs)
    _write_trace(tmp_path / "r0.trace.json.gz")
    cache = {}
    deadline = time.monotonic() + 30
    assert phase_gate.traces_complete(dirs, snapshot, 1, cache=cache, deadline=deadline)
    assert phase_gate.traces_complete(dirs, snapshot, 1, cache=cache, deadline=deadline)
    assert len(trace_metadata_reads) == 1
    assert trace_metadata_reads[0][1] == deadline


def test_current_trace_proof_cache_invalidates_changed_trace(tmp_path, trace_metadata_reads):
    dirs = [str(tmp_path)]
    snapshot = phase_gate.snapshot_traces(dirs)
    path = _write_trace(tmp_path / "r0.trace.json")
    cache = {}
    assert phase_gate.traces_complete(dirs, snapshot, 1, cache=cache)
    original_mtime = path.stat().st_mtime_ns
    _write_trace(path, gpu=False)
    os.utime(path, ns=(original_mtime + 1000, original_mtime + 1000))
    assert not phase_gate.traces_complete(dirs, snapshot, 1, cache=cache)
    assert not phase_gate.traces_complete(dirs, snapshot, 1, cache=cache)
    assert len(trace_metadata_reads) == 2


def test_current_trace_proof_cache_does_not_bypass_snapshot_or_rank_checks(tmp_path, trace_metadata_reads):
    dirs = [str(tmp_path)]
    snapshot = phase_gate.snapshot_traces(dirs)
    _write_trace(tmp_path / "worker.trace.json.gz")
    cache = {}
    assert phase_gate.traces_complete(dirs, snapshot, 1, cache=cache)
    assert not phase_gate.traces_complete(dirs, snapshot, 2, cache=cache)
    next_snapshot = phase_gate.snapshot_traces(dirs)
    assert not phase_gate.traces_complete(dirs, next_snapshot, 1, cache=cache)
    assert len(trace_metadata_reads) == 1


def test_current_trace_proof_cache_keeps_unranked_metadata_unranked_for_tp2(tmp_path):
    dirs = [str(tmp_path)]
    snapshot = phase_gate.snapshot_traces(dirs)
    _write_trace(tmp_path / "worker.trace.json.gz")
    cache = {}
    assert phase_gate.traces_complete(dirs, snapshot, 1, cache=cache)
    _write_trace(tmp_path / "r1.trace.json.gz")
    assert not phase_gate.traces_complete(dirs, snapshot, 2, cache=cache)


def test_current_trace_proof_expired_deadline_rejects_cached_success(tmp_path, monkeypatch, trace_metadata_reads):
    dirs = [str(tmp_path)]
    snapshot = phase_gate.snapshot_traces(dirs)
    _write_trace(tmp_path / "r0.trace.json.gz")
    cache = {}
    monkeypatch.setattr(phase_gate.time, "monotonic", lambda: 10.0)
    assert phase_gate.traces_complete(dirs, snapshot, 1, cache=cache, deadline=11.0)
    monkeypatch.setattr(phase_gate.time, "monotonic", lambda: 11.0)
    assert not phase_gate.traces_complete(dirs, snapshot, 1, cache=cache, deadline=11.0)
    assert len(trace_metadata_reads) == 1


def test_current_trace_proof_deadline_expiring_during_parse_rejects_success(tmp_path, monkeypatch):
    dirs = [str(tmp_path)]
    snapshot = phase_gate.snapshot_traces(dirs)
    _write_trace(tmp_path / "r0.trace.json.gz")
    original = phase_gate._read_trace_metadata
    now = [10.0]
    monkeypatch.setattr(phase_gate.time, "monotonic", lambda: now[0])

    def read_metadata(path, *, deadline):
        result = original(path, deadline=deadline)
        now[0] = 11.0
        return result

    monkeypatch.setattr(phase_gate, "_read_trace_metadata", read_metadata)
    assert not phase_gate.traces_complete(dirs, snapshot, 1, deadline=11.0)


@pytest.mark.parametrize("code", [errno.EMFILE, errno.EIO, errno.ENOMEM])
@pytest.mark.parametrize("injection", ["metadata", "gzip_open"])
def test_current_trace_proof_retries_after_transient_read_failure(tmp_path, monkeypatch, injection, code):
    dirs = [str(tmp_path)]
    snapshot = phase_gate.snapshot_traces(dirs)
    _write_trace(tmp_path / "r0.trace.json.gz")
    original_metadata = phase_gate._read_trace_metadata
    original_open = phase_gate.gzip.open
    attempts = []

    def read_metadata(path, *, deadline):
        attempts.append(path)
        if injection == "metadata" and len(attempts) == 1:
            raise OSError(code, os.strerror(code))
        return original_metadata(path, deadline=deadline)

    def transient_open(*args, **kwargs):
        if injection == "gzip_open" and len(attempts) == 1:
            raise OSError(code, os.strerror(code))
        return original_open(*args, **kwargs)

    monkeypatch.setattr(phase_gate, "_read_trace_metadata", read_metadata)
    monkeypatch.setattr(phase_gate.gzip, "open", transient_open)
    cache = {}
    assert not phase_gate.traces_complete(dirs, snapshot, 1, cache=cache)
    assert phase_gate.traces_complete(dirs, snapshot, 1, cache=cache)
    assert phase_gate.traces_complete(dirs, snapshot, 1, cache=cache)
    assert len(attempts) == 2


@pytest.mark.parametrize("case", ["bad_gzip_header", "truncated_gzip", "bad_json"])
def test_current_trace_proof_keeps_negative_cache_for_persistent_corruption(tmp_path, trace_metadata_reads, case):
    dirs = [str(tmp_path)]
    snapshot = phase_gate.snapshot_traces(dirs)
    path = tmp_path / "r0.trace.json.gz"
    if case == "bad_gzip_header":
        path.write_bytes(b"this is not a gzip stream" * 16)
    elif case == "truncated_gzip":
        _write_trace(path)
        path.write_bytes(path.read_bytes()[:-8])
    else:
        _write_trace_text(path, '{"traceEvents":[{"cat":"kernel","ph":"X","ts":1,"dur":2}]')
    now_ns = time.time_ns()
    os.utime(path, ns=(now_ns, now_ns))
    cache = {}
    assert not phase_gate.traces_complete(dirs, snapshot, 1, cache=cache)
    assert not phase_gate.traces_complete(dirs, snapshot, 1, cache=cache)
    assert len(trace_metadata_reads) == 1


def _run_trace_cli(tmp_path, snapshot, *, budget="5.0", read_mode="record"):
    script = textwrap.dedent(
        """
        import builtins
        import errno
        import importlib.util
        import os
        import sys
        import time

        asset, log_path, read_mode = sys.argv[1:4]
        spec = importlib.util.spec_from_file_location("aiperf_phase_gate_cli", asset)
        gate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gate)

        class ObservedReader:
            def __init__(self, handle):
                self.handle = handle
                self.recorded = False

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return self.handle.__exit__(*args)

            def __getattr__(self, name):
                return getattr(self.handle, name)

            def read(self, size=-1):
                if not self.recorded:
                    with builtins.open(log_path, "a", encoding="utf-8") as log:
                        log.write("read\\n")
                    self.recorded = True
                if read_mode == "transient-error":
                    raise OSError(errno.EMFILE, os.strerror(errno.EMFILE))
                chunk = self.handle.read(size)
                if not chunk and read_mode == "block-eof":
                    time.sleep(5)
                return chunk

        original_open = gate.gzip.open

        def observed_open(*args, **kwargs):
            return ObservedReader(original_open(*args, **kwargs))

        gate.gzip.open = observed_open
        if read_mode == "block-scan":
            original_current_traces = gate.current_traces

            def blocked_current_traces(*args, **kwargs):
                time.sleep(5)
                return original_current_traces(*args, **kwargs)

            gate.current_traces = blocked_current_traces
        raise SystemExit(gate.main(sys.argv[4:]))
        """
    )
    return subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(agentx_asset_dir() / "aiperf_phase_gate.py"),
            str(tmp_path / "trace-reads.log"),
            read_mode,
            "traces-complete",
            "--trace-dir",
            str(tmp_path),
            "--snapshot",
            json.dumps(snapshot),
            "--tp",
            "1",
            "--cache-file",
            str(tmp_path / "trace-cache.json"),
            f"--timeout-seconds={budget}",
        ],
        capture_output=True,
        text=True,
        timeout=3,
    )


def _trace_cli_read_count(tmp_path):
    log = tmp_path / "trace-reads.log"
    return len(log.read_text(encoding="utf-8").splitlines()) if log.exists() else 0


def test_trace_cli_cache_persists_across_invocations(tmp_path):
    snapshot = phase_gate.snapshot_traces([str(tmp_path)])
    _write_trace(tmp_path / "r0.trace.json.gz")
    first = _run_trace_cli(tmp_path, snapshot)
    assert first.returncode == 0, first.stderr
    assert (tmp_path / "trace-cache.json").is_file()
    assert _trace_cli_read_count(tmp_path) == 1
    second = _run_trace_cli(tmp_path, snapshot)
    assert second.returncode == 0, second.stderr
    assert _trace_cli_read_count(tmp_path) == 1


def test_trace_cli_cache_invalidates_changed_trace(tmp_path):
    snapshot = phase_gate.snapshot_traces([str(tmp_path)])
    path = _write_trace(tmp_path / "r0.trace.json.gz")
    first = _run_trace_cli(tmp_path, snapshot)
    assert first.returncode == 0, first.stderr
    mtime = path.stat().st_mtime_ns
    _write_trace(path, gpu=False)
    os.utime(path, ns=(mtime + 1000, mtime + 1000))
    second = _run_trace_cli(tmp_path, snapshot)
    assert second.returncode != 0
    assert _trace_cli_read_count(tmp_path) == 2
    third = _run_trace_cli(tmp_path, snapshot)
    assert third.returncode != 0
    assert _trace_cli_read_count(tmp_path) == 2


def test_trace_cli_cache_is_scoped_to_snapshot(tmp_path):
    snapshot = phase_gate.snapshot_traces([str(tmp_path)])
    _write_trace(tmp_path / "r0.trace.json.gz")
    first = _run_trace_cli(tmp_path, snapshot)
    assert first.returncode == 0, first.stderr
    different_snapshot = dict(snapshot, started_ns=snapshot["started_ns"] - 1)
    second = _run_trace_cli(tmp_path, different_snapshot)
    assert second.returncode == 0, second.stderr
    assert _trace_cli_read_count(tmp_path) == 2
    next_snapshot = phase_gate.snapshot_traces([str(tmp_path)])
    third = _run_trace_cli(tmp_path, next_snapshot)
    assert third.returncode != 0
    assert _trace_cli_read_count(tmp_path) == 2


def test_trace_cli_transient_read_error_is_not_cached_across_invocations(tmp_path):
    snapshot = phase_gate.snapshot_traces([str(tmp_path)])
    _write_trace(tmp_path / "r0.trace.json.gz")
    first = _run_trace_cli(tmp_path, snapshot, read_mode="transient-error")
    assert first.returncode != 0
    assert _trace_cli_read_count(tmp_path) == 1
    second = _run_trace_cli(tmp_path, snapshot)
    assert second.returncode == 0, second.stderr
    assert _trace_cli_read_count(tmp_path) == 2
    third = _run_trace_cli(tmp_path, snapshot)
    assert third.returncode == 0, third.stderr
    assert _trace_cli_read_count(tmp_path) == 2


@pytest.mark.parametrize("budget", ["0", "-0.0", "-1.5", "nan", "inf", "-inf", "1e309"])
def test_trace_cli_rejects_nonpositive_or_nonfinite_budget(tmp_path, budget):
    snapshot = phase_gate.snapshot_traces([str(tmp_path)])
    _write_trace(tmp_path / "r0.trace.json.gz")
    result = _run_trace_cli(tmp_path, snapshot, budget=budget)
    assert result.returncode != 0
    assert _trace_cli_read_count(tmp_path) == 0


@pytest.mark.skipif(os.name != "posix", reason="hard trace deadlines require POSIX timers")
def test_trace_cli_hard_timeout_interrupts_read_without_caching_partial_proof(tmp_path):
    snapshot = phase_gate.snapshot_traces([str(tmp_path)])
    _write_trace(tmp_path / "r0.trace.json.gz")
    result = _run_trace_cli(tmp_path, snapshot, budget="0.05", read_mode="block-eof")
    assert result.returncode != 0
    assert _trace_cli_read_count(tmp_path) == 1
    retry = _run_trace_cli(tmp_path, snapshot)
    assert retry.returncode == 0, retry.stderr
    assert _trace_cli_read_count(tmp_path) == 2
    cached = _run_trace_cli(tmp_path, snapshot)
    assert cached.returncode == 0, cached.stderr
    assert _trace_cli_read_count(tmp_path) == 2


@pytest.mark.skipif(os.name != "posix", reason="hard trace deadlines require POSIX timers")
def test_trace_cli_hard_timeout_cannot_accept_cached_success(tmp_path):
    snapshot = phase_gate.snapshot_traces([str(tmp_path)])
    _write_trace(tmp_path / "r0.trace.json.gz")
    first = _run_trace_cli(tmp_path, snapshot)
    assert first.returncode == 0, first.stderr
    result = _run_trace_cli(tmp_path, snapshot, budget="0.05", read_mode="block-scan")
    assert result.returncode != 0
    assert _trace_cli_read_count(tmp_path) == 1


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
