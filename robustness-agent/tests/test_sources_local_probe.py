# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the LocalProbe fallback source."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from robustness_agent.sources.base import SourceUnavailable
from robustness_agent.sources.local_probe import (
    LocalProbeConfig,
    LocalProbeSource,
)


@pytest.fixture()
def session_dir(tmp_path: Path) -> Path:
    storage = tmp_path / "storage"
    storage.mkdir()
    return tmp_path


def _seed_conductor_db(
    session_dir: Path,
    rows: list[dict],
    *,
    schema: str = "v6",
) -> Path:
    db = session_dir / "storage" / "conductor.db"
    conn = sqlite3.connect(db)
    if schema == "v6":
        conn.execute(
            "CREATE TABLE events ("
            "  seq INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  msg_id TEXT,"
            "  from_agent TEXT,"
            "  to_agent TEXT,"
            "  topic TEXT,"
            "  payload TEXT,"
            "  ts TEXT"
            ")"
        )
        for row in rows:
            conn.execute(
                "INSERT INTO events (msg_id, from_agent, to_agent, topic, payload, ts) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    row.get("msg_id", ""),
                    row.get("agent", ""),
                    row.get("to_agent", ""),
                    row.get("topic", ""),
                    json.dumps(row.get("payload", {})),
                    row.get("ts", ""),
                ),
            )
    else:
        conn.execute(
            "CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, agent TEXT, "
            "intent_type TEXT, payload TEXT, timestamp REAL, topic TEXT)"
        )
        for row in rows:
            conn.execute(
                "INSERT INTO events (agent, intent_type, payload, timestamp, topic) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    row.get("agent", ""),
                    row.get("intent_type", ""),
                    json.dumps(row.get("payload", {})),
                    row.get("timestamp", 0.0),
                    row.get("topic", ""),
                ),
            )
    conn.commit()
    conn.close()
    return db


# ---------------------------------------------------------------------------
# Coordinator events
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_local_probe_reads_v6_events_schema(session_dir: Path):
    _seed_conductor_db(
        session_dir,
        [
            {"agent": "orchestration", "topic": "heartbeat", "payload": {"x": 1}},
            {"agent": "kernel", "topic": "alert", "payload": {"y": 2}},
        ],
        schema="v6",
    )
    cfg = LocalProbeConfig(session_dir=session_dir, disk_mountpoints=())
    probe = LocalProbeSource(cfg)
    data = await probe.fetch(ctx=None)
    assert len(data.coordinator_events) == 2
    topics = sorted(e["topic"] for e in data.coordinator_events)
    assert topics == ["alert", "heartbeat"]
    assert data.sources_used == ["local-probe"]


@pytest.mark.asyncio
async def test_local_probe_reads_legacy_events_schema(session_dir: Path):
    _seed_conductor_db(
        session_dir,
        [{"agent": "kernel", "intent_type": "alert", "topic": "alert", "timestamp": 1.0}],
        schema="legacy",
    )
    cfg = LocalProbeConfig(session_dir=session_dir, disk_mountpoints=())
    data = await LocalProbeSource(cfg).fetch(ctx=None)
    assert len(data.coordinator_events) == 1
    assert data.coordinator_events[0]["agent"] == "kernel"


@pytest.mark.asyncio
async def test_local_probe_returns_disk_usage(tmp_path: Path):
    cfg = LocalProbeConfig(
        session_dir=None,
        disk_mountpoints=(str(tmp_path),),
    )
    data = await LocalProbeSource(cfg).fetch(ctx=None)
    assert str(tmp_path) in data.local_disk
    snap = data.local_disk[str(tmp_path)]
    assert snap["total_gb"] >= 0
    assert "used_pct" in snap


@pytest.mark.asyncio
async def test_local_probe_unavailable_when_no_data(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    cfg = LocalProbeConfig(
        session_dir=None,
        disk_mountpoints=(),
        process_patterns=(),
        server_log_path=None,
        # All optional probes neutralised so we exercise the
        # SourceUnavailable path with no data anywhere. Each new probe
        # generation has to be added here as it's introduced.
        ray_probe_enabled=False,
        fd_probe_enabled=False,
        decision_audit_enabled=False,
        preflight_enabled=False,
        critic_health_enabled=False,
        state_integrity_enabled=False,
        external_deps_enabled=False,
    )

    # Force samplers that read the local host to return empty.
    monkeypatch.setattr(
        "robustness_agent.sources.local_probe._sample_gpu", lambda: {}
    )
    monkeypatch.setattr(
        "robustness_agent.sources.local_probe._sample_aiter_jit",
        lambda _jit_dir: {},
    )

    probe = LocalProbeSource(cfg)
    with pytest.raises(SourceUnavailable):
        await probe.fetch(ctx=None)


@pytest.mark.asyncio
async def test_local_probe_handles_missing_conductor_db(tmp_path: Path):
    cfg = LocalProbeConfig(
        session_dir=tmp_path,
        disk_mountpoints=(str(tmp_path),),
    )
    data = await LocalProbeSource(cfg).fetch(ctx=None)
    assert data.coordinator_events == []
    assert str(tmp_path) in data.local_disk


# ---------------------------------------------------------------------------
# Log tail
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_local_probe_tails_log_file(tmp_path: Path):
    log_path = tmp_path / "server.log"
    log_path.write_text("\n".join(f"line {i}" for i in range(50)) + "\n")
    cfg = LocalProbeConfig(
        session_dir=None,
        disk_mountpoints=(str(tmp_path),),
        server_log_path=log_path,
        log_tail_lines=10,
    )
    data = await LocalProbeSource(cfg).fetch(ctx=None)
    assert len(data.local_log_tail) == 10
    assert data.local_log_tail[-1] == "line 49"


def test_parse_rocm_smi_csv_multi_block():
    from robustness_agent.sources.local_probe import _parse_rocm_smi_csv

    text = (
        "device,GPU use (%)\n"
        "card0,42\n"
        "card1,17\n"
        "\n"
        "device,GPU memory use (%)\n"
        "card0,68\n"
        "card1,55\n"
        "\n"
        "device,Temperature (Sensor edge) (C)\n"
        "card0,72.5\n"
        "card1,69.0\n"
        "\n"
        "device,Average Graphics Package Power (W)\n"
        "card0,210.0\n"
        "card1,198.5\n"
    )
    gpus = _parse_rocm_smi_csv(text)
    assert len(gpus) == 2
    by_id = {g["gpu_id"]: g for g in gpus}
    assert by_id[0]["util_gpu_pct"] == 42.0
    assert by_id[0]["util_mem_pct"] == 68.0
    assert by_id[0]["temperature_c"] == 72.5
    assert by_id[0]["power_watts"] == 210.0
    assert by_id[1]["util_gpu_pct"] == 17.0


def test_parse_rocm_smi_csv_skips_unknown_columns_and_garbage():
    from robustness_agent.sources.local_probe import _parse_rocm_smi_csv

    text = (
        "device,GPU use (%),Unknown column\n"
        "card0,5,N/A\n"
        "card2,??,whatever\n"
        "\n"
        "noheader,row\n"
    )
    gpus = _parse_rocm_smi_csv(text)
    assert [g["gpu_id"] for g in gpus] == [0]
    assert gpus[0]["util_gpu_pct"] == 5.0
    assert "Unknown column" not in gpus[0]


@pytest.mark.asyncio
async def test_local_probe_uses_rocm_smi_when_available(monkeypatch, tmp_path: Path):
    from robustness_agent.sources import local_probe as lp

    captured: dict = {}

    def fake_which(binary: str) -> str | None:
        return "/usr/bin/rocm-smi" if binary == "rocm-smi" else None

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        sample = (
            "device,GPU use (%)\n"
            "card0,55\n"
            "\n"
            "device,Temperature (Sensor edge) (C)\n"
            "card0,80.0\n"
        )

        class _Result:
            returncode = 0
            stdout = sample
            stderr = ""

        return _Result()

    monkeypatch.setattr(lp.shutil, "which", fake_which)
    monkeypatch.setattr(lp.subprocess, "run", fake_run)

    cfg = lp.LocalProbeConfig(
        session_dir=None,
        disk_mountpoints=(str(tmp_path),),
        process_patterns=(),
    )
    data = await lp.LocalProbeSource(cfg).fetch(ctx=None)
    assert data.local_gpu.get("tool") == "rocm-smi"
    gpus = data.local_gpu.get("gpus")
    assert gpus and gpus[0]["util_gpu_pct"] == 55.0
    assert gpus[0]["temperature_c"] == 80.0
    assert "--showpower" in captured["cmd"]


def test_extract_log_errors_finds_known_patterns():
    from robustness_agent.sources.local_probe import _extract_log_errors

    tail = [
        "INFO  starting",
        "RuntimeError: tensor shape mismatch",
        "WARNING something else",
        "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate ...",
    ]
    out = _extract_log_errors(
        tail,
        patterns=("CUDA out of memory", "RuntimeError"),
        window=10,
    )
    assert {e["pattern"] for e in out} == {"CUDA out of memory", "RuntimeError"}
    assert all(len(e["line"]) <= 240 for e in out)


def test_extract_log_errors_truncates_long_lines():
    from robustness_agent.sources.local_probe import _extract_log_errors

    long_line = "RuntimeError: " + ("x" * 1000)
    out = _extract_log_errors([long_line], patterns=("RuntimeError",), window=10)
    assert out and len(out[0]["line"]) == 240


def test_extract_log_errors_returns_empty_when_no_match():
    from robustness_agent.sources.local_probe import _extract_log_errors

    assert _extract_log_errors(["INFO ok"], patterns=("OOM",), window=10) == []


@pytest.mark.asyncio
async def test_local_probe_emits_log_errors_alongside_tail(tmp_path: Path):
    log_path = tmp_path / "server.log"
    log_path.write_text(
        "\n".join(
            [
                "INFO loading",
                "torch.cuda.OutOfMemoryError: CUDA out of memory occurred",
                "INFO recovered",
            ]
        )
        + "\n"
    )
    from robustness_agent.sources.local_probe import (
        LocalProbeConfig,
        LocalProbeSource,
    )

    cfg = LocalProbeConfig(
        session_dir=None,
        disk_mountpoints=(str(tmp_path),),
        process_patterns=(),
        server_log_path=log_path,
        log_tail_lines=10,
        log_error_patterns=("CUDA out of memory",),
    )
    data = await LocalProbeSource(cfg).fetch(ctx=None)
    assert data.local_log_errors
    assert data.local_log_errors[0]["pattern"] == "CUDA out of memory"
    assert "out of memory" in data.local_log_errors[0]["line"].lower()


@pytest.mark.asyncio
async def test_local_probe_runs_health_probes(monkeypatch, tmp_path: Path):
    import httpx
    from robustness_agent.sources import local_probe as lp

    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if "alive" in str(request.url):
            return httpx.Response(200, json={"ok": True})
        if "wedged" in str(request.url):
            return httpx.Response(503, json={"detail": "wedged"})
        raise httpx.ConnectError("connection refused", request=request)

    transport = httpx.MockTransport(handler)

    class _PatchedClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(lp.httpx, "AsyncClient", _PatchedClient)

    # Isolate from host shell pollution: a configured ``$OPENAI_BASE_URL``
    # would otherwise make the external-deps sub-probe fire an extra
    # ``/models`` gateway request through the same mock transport, inflating
    # ``seen`` and breaking the exact-count assertion below. This test only
    # exercises ``_probe_local_servers``, so unset the gateway env vars and
    # disable the orthogonal external-deps probe.
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("SAFE_API_KEY", raising=False)

    cfg = lp.LocalProbeConfig(
        session_dir=None,
        disk_mountpoints=(str(tmp_path),),
        process_patterns=(),
        health_probe_targets=(
            "http://localhost:30000/alive",
            "http://localhost:30001/wedged",
            "http://localhost:30002/dead",
        ),
        health_probe_timeout_s=1.0,
        external_deps_enabled=False,
    )
    data = await lp.LocalProbeSource(cfg).fetch(ctx=None)
    by_url = {entry["url"]: entry for entry in data.local_server_health}
    assert by_url["http://localhost:30000/alive"]["status"] == "ok"
    assert by_url["http://localhost:30000/alive"]["reachable"] is True
    assert by_url["http://localhost:30001/wedged"]["status"] == "http_error"
    assert by_url["http://localhost:30001/wedged"]["reachable"] is False
    assert by_url["http://localhost:30002/dead"]["status"] == "error"
    assert "connect" in by_url["http://localhost:30002/dead"]["error"]
    assert len(seen) == 3


@pytest.mark.asyncio
async def test_local_probe_skips_log_when_path_missing(tmp_path: Path):
    cfg = LocalProbeConfig(
        session_dir=None,
        disk_mountpoints=(str(tmp_path),),
        server_log_path=tmp_path / "no-such.log",
        log_tail_lines=10,
    )
    data = await LocalProbeSource(cfg).fetch(ctx=None)
    assert data.local_log_tail == []


# ===========================================================================
# D2 — multi-source server-log tailing (``_tail_logs``)
# ===========================================================================

import os  # noqa: E402
import subprocess  # noqa: E402
from typing import Any  # noqa: E402
from unittest.mock import patch  # noqa: E402

from robustness_agent.sources import local_probe  # noqa: E402
from robustness_agent.sources.local_probe import (  # noqa: E402
    _is_pid_alive,
    _probe_agent_files,
    _probe_coordinator_pid,
    _probe_external_mounts,
    _probe_state_json,
    _probe_tracelens_cli,
    _probe_wal_size,
    _sample_state_integrity,
    _tail_logs,
)


def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def test_tail_logs_returns_primary_when_no_extras(tmp_path):
    p = tmp_path / "server.log"
    _write(p, "line-1\nline-2\nline-3\n")
    out = _tail_logs(p, None, (), 0, max_lines=10)
    assert out == ["line-1", "line-2", "line-3"]


def test_tail_logs_picks_up_runs_glob(tmp_path):
    primary = tmp_path / "primary.log"
    _write(primary, "PRIMARY-A\n")
    _write(tmp_path / "runs" / "backends" / "t1" / "server.log",
           "VARIANT-1A\nVARIANT-1B\n")
    _write(tmp_path / "runs" / "params" / "t2" / "server.log",
           "VARIANT-2A\n")
    out = _tail_logs(
        primary,
        tmp_path,
        ("runs/*/*/server.log",),
        5,
        max_lines=20,
    )
    assert "PRIMARY-A" in out
    assert any("[server.log] VARIANT-1A" in line for line in out)
    assert any("[server.log] VARIANT-2A" in line for line in out)


def test_tail_logs_dedup_when_primary_matches_glob(tmp_path):
    primary = tmp_path / "runs" / "backends" / "t1" / "server.log"
    _write(primary, "ONLY-LINE\n")
    out = _tail_logs(
        primary, tmp_path, ("runs/*/*/server.log",), 5, max_lines=10,
    )
    matches = [line for line in out if "ONLY-LINE" in line]
    assert len(matches) == 1


def test_tail_logs_cap_extras_by_mtime(tmp_path):
    primary = None
    for i in range(5):
        path = tmp_path / "runs" / "backends" / f"t{i}" / "server.log"
        _write(path, f"variant-{i}\n")
        os.utime(path, (1000.0 + i, 1000.0 + i))
    out = _tail_logs(
        primary, tmp_path, ("runs/*/*/server.log",), 3, max_lines=10,
    )
    body = "\n".join(out)
    assert "variant-4" in body
    assert "variant-3" in body
    assert "variant-2" in body
    assert "variant-0" not in body
    assert "variant-1" not in body


def test_tail_logs_empty_when_no_max_lines(tmp_path):
    _write(tmp_path / "server.log", "x\n")
    assert _tail_logs(tmp_path / "server.log", None, (), 0, max_lines=0) == []


@pytest.mark.asyncio
async def test_local_probe_picks_up_grid_variant_logs(tmp_path):
    """End-to-end: LocalProbe sees a grid variant log under runs/."""
    _write(tmp_path / "runs" / "backends" / "t1" / "server.log",
           "CUDA out of memory at allocator.cc:42\n")
    cfg = LocalProbeConfig(
        session_dir=tmp_path,
        disk_mountpoints=(),
        process_patterns=(),
        ray_probe_enabled=False, fd_probe_enabled=False,
        decision_audit_enabled=False, preflight_enabled=False,
        critic_health_enabled=False,
    )
    data = await LocalProbeSource(cfg).fetch(ctx=None)
    assert any(
        h.get("pattern") == r"CUDA out of memory" for h in data.local_log_errors
    )


# ===========================================================================
# Preflight probes (manifest + kernel breakdown)
# ===========================================================================

@pytest.mark.asyncio
async def test_manifest_loads_when_present(tmp_path):
    sd = tmp_path
    manifest = {
        "schema_version": 2,
        "model_name": "Qwen3-32B",
        "gpu_type": "mi300x",
        "tp": 8,
        "workload": {"precision": "bf16", "max_model_len": 4096, "conc": 16},
    }
    _write_json(sd / "manifest.json", manifest)
    cfg = LocalProbeConfig(
        session_dir=sd, disk_mountpoints=(), process_patterns=(),
        ray_probe_enabled=False, fd_probe_enabled=False,
        decision_audit_enabled=False,
    )
    data = await LocalProbeSource(cfg).fetch(ctx=None)
    assert data.local_manifest == manifest


@pytest.mark.asyncio
async def test_manifest_empty_when_missing(tmp_path):
    sd = tmp_path
    _write_json(sd / "results" / "ci_metrics.json", {"model": "X"})
    cfg = LocalProbeConfig(
        session_dir=sd, disk_mountpoints=(), process_patterns=(),
        ray_probe_enabled=False, fd_probe_enabled=False,
    )
    data = await LocalProbeSource(cfg).fetch(ctx=None)
    assert data.local_manifest == {}


@pytest.mark.asyncio
async def test_manifest_empty_on_malformed_json(tmp_path, monkeypatch):
    sd = tmp_path
    (sd / "manifest.json").write_text("not json at all", encoding="utf-8")
    _write_json(sd / "profiles" / "kernel_breakdown.json", [])
    cfg = LocalProbeConfig(
        session_dir=sd, disk_mountpoints=(), process_patterns=(),
        ray_probe_enabled=False, fd_probe_enabled=False,
        decision_audit_enabled=False,
    )
    monkeypatch.setattr(
        "robustness_agent.sources.local_probe._sample_gpu", lambda: {}
    )
    monkeypatch.setattr(
        "robustness_agent.sources.local_probe._sample_aiter_jit",
        lambda _d: {},
    )
    data = await LocalProbeSource(cfg).fetch(ctx=None)
    assert data.local_manifest == {}


@pytest.mark.asyncio
async def test_kernel_breakdown_aggregates_by_tier(tmp_path):
    sd = tmp_path
    rows = [
        {"name": "triton_red_x", "gpu_pct": 12.0, "tier": "T1_TRITON",
         "count": 100, "duration_us": 1000},
        {"name": "triton_poi_y", "gpu_pct": 8.5, "tier": "T1_TRITON",
         "count": 80, "duration_us": 800},
        {"name": "mha_fwd_kernel", "gpu_pct": 30.0, "tier": "T2_AITER_CK",
         "count": 50, "duration_us": 2000},
        {"name": "rccl_allreduce", "gpu_pct": 20.0, "tier": "T4_COMM",
         "count": 200, "duration_us": 500},
        {"name": "hipblaslt_gemm", "gpu_pct": 25.0, "tier": "T5_COMPILED",
         "count": 30, "duration_us": 1500},
    ]
    _write_json(sd / "profiles" / "kernel_breakdown.json", rows)
    cfg = LocalProbeConfig(
        session_dir=sd, disk_mountpoints=(), process_patterns=(),
        ray_probe_enabled=False, fd_probe_enabled=False,
        decision_audit_enabled=False,
    )
    data = await LocalProbeSource(cfg).fetch(ctx=None)
    breakdown = data.local_kernel_breakdown
    assert breakdown["tier_pcts"]["triton"] == pytest.approx(20.5, rel=1e-3)
    assert breakdown["tier_pcts"]["vendor"] == 30.0
    assert breakdown["tier_pcts"]["comm"] == 20.0
    assert breakdown["tier_pcts"]["compiled"] == 25.0
    assert breakdown["total_kernels"] == 5
    assert breakdown["total_gpu_pct"] == pytest.approx(95.5, rel=1e-3)
    assert breakdown["kernel_breakdown_path"].endswith("kernel_breakdown.json")
    assert isinstance(breakdown["mtime"], float)


@pytest.mark.asyncio
async def test_kernel_breakdown_empty_when_missing(tmp_path):
    sd = tmp_path
    _write_json(sd / "manifest.json", {"model_name": "X"})
    cfg = LocalProbeConfig(
        session_dir=sd, disk_mountpoints=(), process_patterns=(),
        ray_probe_enabled=False, fd_probe_enabled=False,
        decision_audit_enabled=False,
    )
    data = await LocalProbeSource(cfg).fetch(ctx=None)
    assert data.local_kernel_breakdown == {}


@pytest.mark.asyncio
async def test_preflight_disabled_skips_manifest_and_breakdown(
    tmp_path, monkeypatch,
):
    """When ``preflight_enabled=False`` both new slots stay empty even
    when the files exist on disk."""
    sd = tmp_path
    _write_json(sd / "manifest.json", {"model_name": "X"})
    _write_json(sd / "profiles" / "kernel_breakdown.json", [
        {"name": "k", "gpu_pct": 50.0, "tier": "T1_TRITON",
         "count": 1, "duration_us": 100},
    ])
    monkeypatch.setattr(
        "robustness_agent.sources.local_probe._sample_gpu", lambda: {}
    )
    monkeypatch.setattr(
        "robustness_agent.sources.local_probe._sample_aiter_jit",
        lambda _d: {},
    )
    cfg = LocalProbeConfig(
        session_dir=sd,
        disk_mountpoints=(str(sd),),
        process_patterns=(),
        ray_probe_enabled=False, fd_probe_enabled=False,
        decision_audit_enabled=False,
        preflight_enabled=False,
    )
    data = await LocalProbeSource(cfg).fetch(ctx=None)
    assert data.local_manifest == {}
    assert data.local_kernel_breakdown == {}


@pytest.mark.asyncio
async def test_kernel_breakdown_unknown_tier_falls_back_to_lowered_name(tmp_path):
    sd = tmp_path
    rows = [
        {"name": "k1", "gpu_pct": 50.0, "tier": "T6_NEW_TIER",
         "count": 1, "duration_us": 100},
        {"name": "k2", "gpu_pct": 50.0, "tier": "",
         "count": 1, "duration_us": 100},
    ]
    _write_json(sd / "profiles" / "kernel_breakdown.json", rows)
    cfg = LocalProbeConfig(
        session_dir=sd, disk_mountpoints=(), process_patterns=(),
        ray_probe_enabled=False, fd_probe_enabled=False,
        decision_audit_enabled=False,
    )
    data = await LocalProbeSource(cfg).fetch(ctx=None)
    assert data.local_kernel_breakdown["tier_pcts"]["t6_new_tier"] == 50.0
    assert data.local_kernel_breakdown["tier_pcts"]["unknown"] == 50.0


# ===========================================================================
# Ray head probe (``_probe_ray_head`` + ``_parse_ray_pending_count``)
# ===========================================================================
# Regression: legacy regex ``(\d+)\s+pending`` captured trailing hex digits
# of Ray node IDs (node ID ending ``...d3da81`` → bogus pending_tasks=81).

RAY_STATUS_IDLE = """\
======== Autoscaler status: 2026-05-20 03:14:48.677060 ========
Node status
---------------------------------------------------------------
Active:
 1 node_dcd71ad0316b238eb2ab9323d50f23bb8201ef3c78447c14dad3da81
Pending:
 (no pending nodes)
Recent failures:
 (no failures)

Resources
---------------------------------------------------------------
Usage:
 0.0/64.0 CPU
 0.0/8.0 GPU
 0B/829.27GiB memory
 0B/186.26GiB object_store_memory

Demands:
 (no resource demands)
"""

RAY_STATUS_SINGLE_DEMAND = """\
======== Autoscaler status: 2026-05-20 04:00:00 ========
Node status
---------------------------------------------------------------
Active:
 1 node_dcd71ad0316b238eb2ab9323d50f23bb8201ef3c78447c14dad3da81

Resources
---------------------------------------------------------------
Usage:
 64.0/64.0 CPU
 8.0/8.0 GPU

Demands:
 {'CPU': 1.0}: 5+ pending tasks/actors
"""

RAY_STATUS_MULTI_DEMAND = """\
======== Autoscaler status: 2026-05-20 04:01:00 ========
Node status
---------------------------------------------------------------
Active:
 1 node_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa11
 1 node_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb22

Demands:
 {'CPU': 1.0}: 4+ pending tasks
 {'GPU': 1.0}: 12+ pending tasks/actors
 {'CPU': 0.5, 'memory': 1000000000}: 2 pending actors
"""


def test_parse_idle_status_with_digit_terminated_node_id():
    """Regression: legacy regex captured ``81`` from ``...d3da81\\nPending``."""
    assert local_probe._parse_ray_pending_count(RAY_STATUS_IDLE) == 0


def test_parse_single_demand():
    assert local_probe._parse_ray_pending_count(RAY_STATUS_SINGLE_DEMAND) == 5


def test_parse_multiple_demands_sum():
    assert local_probe._parse_ray_pending_count(RAY_STATUS_MULTI_DEMAND) == 18


def test_parse_empty_string():
    assert local_probe._parse_ray_pending_count("") == 0


def test_parse_ignores_node_id_substrings():
    """``\\d+`` inside node-hash tokens must not match without ``pending task|actor`` suffix."""
    text = """\
Active:
 1 node_0000000000000000000000000000000000000000000000000000000000000099
 2 node_0000000000000000000000000000000000000000000000000000000000000123
"""
    assert local_probe._parse_ray_pending_count(text) == 0


def test_parse_ignores_pending_nodes_header():
    text = "Pending:\n (no pending nodes)\n"
    assert local_probe._parse_ray_pending_count(text) == 0


def test_parse_handles_plus_suffix_on_count():
    text = " {'GPU': 8.0}: 99+ pending tasks/actors"
    assert local_probe._parse_ray_pending_count(text) == 99


def test_parse_actor_suffix():
    text = " {'CPU': 0.1}: 7 pending actors"
    assert local_probe._parse_ray_pending_count(text) == 7


def _completed(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["ray", "status"], returncode=returncode, stdout=stdout, stderr="",
    )


def test_probe_returns_empty_when_ray_not_on_path():
    with patch.object(local_probe.shutil, "which", return_value=None):
        assert local_probe._probe_ray_head(1.0) == {}


def test_probe_idle_returns_zero_pending():
    with patch.object(local_probe.shutil, "which", return_value="/usr/bin/ray"), \
         patch.object(local_probe.subprocess, "run",
                      return_value=_completed(RAY_STATUS_IDLE)):
        out = local_probe._probe_ray_head(1.0)
    assert out["healthy"] is True
    assert out["pending_tasks"] == 0
    assert out["returncode"] == 0
    assert "Autoscaler status" in out["stdout_head"]


def test_probe_demand_block_counted():
    with patch.object(local_probe.shutil, "which", return_value="/usr/bin/ray"), \
         patch.object(local_probe.subprocess, "run",
                      return_value=_completed(RAY_STATUS_MULTI_DEMAND)):
        out = local_probe._probe_ray_head(1.0)
    assert out["pending_tasks"] == 18


def test_probe_unhealthy_on_nonzero_exit():
    with patch.object(local_probe.shutil, "which", return_value="/usr/bin/ray"), \
         patch.object(local_probe.subprocess, "run",
                      return_value=_completed("ConnectionError: ...", returncode=1)):
        out = local_probe._probe_ray_head(1.0)
    assert out["healthy"] is False
    assert "exit=1" in out["reason"]
    assert out["returncode"] == 1


def test_probe_unhealthy_on_timeout():
    def _raise(*_a: Any, **_kw: Any) -> Any:
        raise subprocess.TimeoutExpired(cmd="ray status", timeout=1.0)

    with patch.object(local_probe.shutil, "which", return_value="/usr/bin/ray"), \
         patch.object(local_probe.subprocess, "run", side_effect=_raise):
        out = local_probe._probe_ray_head(1.0)
    assert out["healthy"] is False
    assert "timed out" in out["reason"]
    assert out["returncode"] is None


def test_probe_unhealthy_on_oserror():
    def _raise(*_a: Any, **_kw: Any) -> Any:
        raise FileNotFoundError("ray binary missing mid-call")

    with patch.object(local_probe.shutil, "which", return_value="/usr/bin/ray"), \
         patch.object(local_probe.subprocess, "run", side_effect=_raise):
        out = local_probe._probe_ray_head(1.0)
    assert out["healthy"] is False
    assert "FileNotFoundError" in out["reason"]


def test_probe_clamps_negative_timeout():
    captured: dict[str, Any] = {}

    def _capture(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess:
        captured.update(kwargs)
        return _completed(RAY_STATUS_IDLE)

    with patch.object(local_probe.shutil, "which", return_value="/usr/bin/ray"), \
         patch.object(local_probe.subprocess, "run", side_effect=_capture):
        local_probe._probe_ray_head(0.0)
    assert captured["timeout"] >= 0.5


@pytest.mark.parametrize("suffix", ["00", "11", "42", "99", "ab", "ff", "9a"])
def test_regex_never_matches_node_hash_followed_by_pending_header(suffix: str):
    text = f" 1 node_{'0' * 62}{suffix}\nPending:\n (no pending nodes)\n"
    assert local_probe._parse_ray_pending_count(text) == 0


# ===========================================================================
# Decision audit (``_sample_decision_audit``)
# ===========================================================================

@pytest.mark.asyncio
async def test_decision_audit_empty_when_no_session_dir():
    cfg = LocalProbeConfig(session_dir=None)
    data = await LocalProbeSource(cfg).fetch(ctx=None)
    assert data.local_decision_audit == {}


@pytest.mark.asyncio
async def test_decision_audit_scans_integrate_result_files(tmp_path):
    """Newest integrate ``result.json`` files are surfaced sorted by mtime."""
    sd = tmp_path
    integrate_root = sd / "runs" / "integrate"
    patch_path = sd / "patch.diff"
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    patch_path.write_bytes(b"diff --git a/x b/x\n")
    for i, decision in enumerate(["REVERT", "KEEP", "NEEDS_REVIEW"]):
        out = integrate_root / f"task-{i}" / "result.json"
        _write_json(out, {
            "kernel_id": f"k{i}",
            "decision": decision,
            "gain_pct": float(i),
            "base_tput": 100.0,
            "new_tput": 100.0 + i,
            "patch_path": str(patch_path),
        })

    cfg = LocalProbeConfig(
        session_dir=sd,
        disk_mountpoints=(),
        process_patterns=(),
        ray_probe_enabled=False,
        fd_probe_enabled=False,
    )
    data = await LocalProbeSource(cfg).fetch(ctx=None)
    audit = data.local_decision_audit
    entries = audit["recent_integrate"]
    assert len(entries) == 3
    decisions = {e["decision"] for e in entries}
    assert decisions == {"REVERT", "KEEP", "NEEDS_REVIEW"}
    for e in entries:
        assert e["patch_size_bytes"] == len(b"diff --git a/x b/x\n")


@pytest.mark.asyncio
async def test_decision_audit_patch_size_none_when_patch_missing(tmp_path):
    sd = tmp_path
    integrate_root = sd / "runs" / "integrate" / "t1"
    _write_json(integrate_root / "result.json", {
        "kernel_id": "k1",
        "decision": "KEEP",
        "gain_pct": 5.0,
        "patch_path": "/nonexistent/patch.diff",
    })
    cfg = LocalProbeConfig(
        session_dir=sd, disk_mountpoints=(), process_patterns=(),
        ray_probe_enabled=False, fd_probe_enabled=False,
    )
    data = await LocalProbeSource(cfg).fetch(ctx=None)
    entry = data.local_decision_audit["recent_integrate"][0]
    assert entry["patch_size_bytes"] is None


@pytest.mark.asyncio
async def test_decision_audit_loads_ci_metrics_from_results_dir(tmp_path):
    sd = tmp_path
    ci_path = sd / "results" / "ci_metrics.json"
    _write_json(ci_path, {
        "model": "X", "framework": "sglang", "gpu": "MI300X", "tp": 8,
        "baseline_tok_per_gpu": 1500.0,
        "optimized_tok_per_gpu": 1800.0,
        "gain_pct": 20.0,
    })
    cfg = LocalProbeConfig(
        session_dir=sd, disk_mountpoints=(), process_patterns=(),
        ray_probe_enabled=False, fd_probe_enabled=False,
    )
    data = await LocalProbeSource(cfg).fetch(ctx=None)
    audit = data.local_decision_audit
    assert audit["ci_metrics"]["model"] == "X"
    assert audit["ci_metrics_path"].endswith("ci_metrics.json")


@pytest.mark.asyncio
async def test_decision_audit_prefers_final_over_inflight(tmp_path):
    sd = tmp_path
    inflight = sd / "results" / "ci_metrics.json"
    final = sd / "results" / "ci_metrics_final.json"
    _write_json(inflight, {"baseline_tput": 0.0})
    _write_json(final, {"baseline_tput": 1500.0, "model": "X"})
    cfg = LocalProbeConfig(
        session_dir=sd, disk_mountpoints=(), process_patterns=(),
        ray_probe_enabled=False, fd_probe_enabled=False,
    )
    data = await LocalProbeSource(cfg).fetch(ctx=None)
    audit = data.local_decision_audit
    assert audit["ci_metrics"]["baseline_tput"] == 1500.0
    assert audit["ci_metrics_path"].endswith("ci_metrics_final.json")


@pytest.mark.asyncio
async def test_decision_audit_ci_metrics_empty_when_missing(tmp_path):
    sd = tmp_path
    cfg = LocalProbeConfig(
        session_dir=sd, disk_mountpoints=(), process_patterns=(),
        ray_probe_enabled=False, fd_probe_enabled=False,
    )
    integrate_root = sd / "runs" / "integrate" / "t1"
    _write_json(integrate_root / "result.json", {"decision": "KEEP"})
    data = await LocalProbeSource(cfg).fetch(ctx=None)
    audit = data.local_decision_audit
    assert audit["ci_metrics"] == {}
    assert audit["ci_metrics_path"] == ""


@pytest.mark.asyncio
async def test_decision_audit_tails_oob_attempts(tmp_path):
    sd = tmp_path
    oob_path = (
        sd / "kernel-agent" / "runs" / "sess-1"
        / "optimization_attempts.jsonl"
    )
    rows = [
        {
            "kernel_id": "gemm_a8w8",
            "backend": "oob_claude",
            "report_text": "expected speedup ~1.1x",
            "microbench_speedup": None,
            "ts": "2026-05-11T09:30:19",
        },
        {
            "kernel_id": "rmsnorm",
            "backend": "geak",
            "report_text": "measured 1.18x",
            "microbench_speedup": 1.18,
            "ts": "2026-05-11T10:00:00",
        },
    ]
    _write_jsonl(oob_path, rows)
    cfg = LocalProbeConfig(
        session_dir=sd, disk_mountpoints=(), process_patterns=(),
        ray_probe_enabled=False, fd_probe_enabled=False,
    )
    data = await LocalProbeSource(cfg).fetch(ctx=None)
    audit = data.local_decision_audit
    assert len(audit["oob_attempts"]) == 2
    first = audit["oob_attempts"][0]
    assert first["backend"] == "oob_claude"
    assert first["microbench_speedup"] is None


@pytest.mark.asyncio
async def test_decision_audit_max_integrate_caps_count(tmp_path):
    sd = tmp_path
    for i in range(30):
        out = sd / "runs" / "integrate" / f"t{i}" / "result.json"
        _write_json(out, {"decision": "KEEP", "kernel_id": f"k{i}"})
    cfg = LocalProbeConfig(
        session_dir=sd,
        disk_mountpoints=(),
        process_patterns=(),
        decision_audit_max_integrate=5,
        ray_probe_enabled=False, fd_probe_enabled=False,
    )
    data = await LocalProbeSource(cfg).fetch(ctx=None)
    assert len(data.local_decision_audit["recent_integrate"]) == 5


@pytest.mark.asyncio
async def test_decision_audit_disable_returns_empty(tmp_path):
    sd = tmp_path
    integrate_root = sd / "runs" / "integrate" / "t1"
    _write_json(integrate_root / "result.json", {"decision": "KEEP"})
    cfg = LocalProbeConfig(
        session_dir=sd,
        disk_mountpoints=(),
        process_patterns=(),
        decision_audit_enabled=False,
        ray_probe_enabled=False, fd_probe_enabled=False,
    )
    data = await LocalProbeSource(cfg).fetch(ctx=None)
    assert data.local_decision_audit == {}


@pytest.mark.asyncio
async def test_decision_audit_handles_malformed_json_gracefully(tmp_path):
    sd = tmp_path
    out = sd / "runs" / "integrate" / "t1" / "result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("this is not json", encoding="utf-8")
    good = sd / "runs" / "integrate" / "t2" / "result.json"
    _write_json(good, {"decision": "KEEP", "kernel_id": "k2"})
    cfg = LocalProbeConfig(
        session_dir=sd, disk_mountpoints=(), process_patterns=(),
        ray_probe_enabled=False, fd_probe_enabled=False,
    )
    data = await LocalProbeSource(cfg).fetch(ctx=None)
    entries = data.local_decision_audit["recent_integrate"]
    kernels = {e["kernel_id"] for e in entries}
    assert kernels == {"k2"}


# ===========================================================================
# State integrity + external-deps sub-probes (I + J)
# ===========================================================================

def test_probe_state_json_valid(tmp_path):
    state = {"baseline_tput": 100.0, "stop_reason": ""}
    _write(tmp_path / "state.json", json.dumps(state))
    out = _probe_state_json(tmp_path)
    assert out["valid"] is True
    assert out["stop_reason"] == ""
    assert out["size_bytes"] > 0


def test_probe_state_json_missing(tmp_path):
    out = _probe_state_json(tmp_path)
    assert out["valid"] is False
    assert out["error"] == "missing"


def test_probe_state_json_corrupt(tmp_path):
    _write(tmp_path / "state.json", "this is not json")
    out = _probe_state_json(tmp_path)
    assert out["valid"] is False
    assert out["error"] == "json_parse_failed"


def test_probe_state_json_non_dict(tmp_path):
    _write(tmp_path / "state.json", json.dumps([1, 2, 3]))
    out = _probe_state_json(tmp_path)
    assert out["valid"] is False


def test_probe_wal_size_reads_files(tmp_path):
    db = tmp_path / "storage" / "coordinator.db"
    wal = tmp_path / "storage" / "coordinator.db-wal"
    _write(db, "x" * 1000)
    _write(wal, "y" * 5000)
    out = _probe_wal_size(tmp_path)
    assert out["wal_bytes"] == 5000
    assert out["db_bytes"] == 1000


def test_probe_wal_size_silent_when_no_files(tmp_path):
    out = _probe_wal_size(tmp_path)
    assert out["wal_bytes"] == 0
    assert out["db_bytes"] == 0


def test_probe_agent_files_collects_inbox_outbox(tmp_path):
    for role in ("orchestration", "critic"):
        _write(tmp_path / "agents" / role / "inbox.jsonl", "a\nb\n")
        _write(tmp_path / "agents" / role / "outbox.jsonl", "c\n")
    out = _probe_agent_files(tmp_path)
    assert "orchestration" in out
    assert "critic" in out
    assert out["orchestration"]["inbox_bytes"] == 4
    assert out["orchestration"]["outbox_bytes"] == 2


def test_probe_agent_files_silent_when_no_agents(tmp_path):
    assert _probe_agent_files(tmp_path) == {}


def test_probe_coordinator_pid_alive(tmp_path):
    """Current PID is always alive — used as the synthetic positive case."""
    pid = os.getpid()
    _write(tmp_path / "optimizer_runs" / "run_now.pid", f"{pid}\n")
    out = _probe_coordinator_pid(tmp_path, "optimizer_runs")
    assert out["recorded_pid"] == pid
    assert out["alive"] is True


def test_probe_coordinator_pid_dead(tmp_path):
    """A PID we know is unused — pick a very high value unlikely to clash."""
    _write(tmp_path / "optimizer_runs" / "run_x.pid", "9999999\n")
    out = _probe_coordinator_pid(tmp_path, "optimizer_runs")
    assert out["recorded_pid"] == 9999999
    assert out["alive"] is False


def test_probe_coordinator_pid_no_file(tmp_path):
    out = _probe_coordinator_pid(tmp_path, "optimizer_runs")
    assert out["recorded_pid"] is None
    assert out["alive"] is None


def test_probe_coordinator_pid_picks_newest(tmp_path):
    """Multiple pid files → newest mtime wins."""
    old = tmp_path / "optimizer_runs" / "run_old.pid"
    new = tmp_path / "optimizer_runs" / "run_new.pid"
    _write(old, "111\n")
    _write(new, f"{os.getpid()}\n")
    os.utime(old, (1000.0, 1000.0))
    os.utime(new, (2000.0, 2000.0))
    out = _probe_coordinator_pid(tmp_path, "optimizer_runs")
    assert out["recorded_pid"] == os.getpid()


def test_is_pid_alive_self():
    assert _is_pid_alive(os.getpid()) is True


def test_is_pid_alive_invalid():
    assert _is_pid_alive(9999999) is False


def test_sample_state_integrity_aggregates_slots(tmp_path):
    _write(tmp_path / "state.json", json.dumps({"baseline_tput": 5.0}))
    _write(tmp_path / "storage" / "coordinator.db", "x")
    _write(tmp_path / "storage" / "coordinator.db-wal", "y" * 100)
    _write(tmp_path / "agents" / "kernel" / "inbox.jsonl", "data")
    _write(tmp_path / "optimizer_runs" / "run_x.pid", "9999999\n")
    out = _sample_state_integrity(tmp_path, "optimizer_runs")
    assert out["state_json"]["valid"] is True
    assert out["wal"]["wal_bytes"] == 100
    assert "kernel" in out["agents"]
    assert out["coordinator"]["recorded_pid"] == 9999999
    assert out["coordinator"]["alive"] is False


def test_sample_state_integrity_empty_session_dir():
    assert _sample_state_integrity(None, "optimizer_runs") == {}


def test_probe_leases_via_full_probe(tmp_path):
    """Use sqlite3 to write a fake leases table and verify probe reads."""
    db_path = tmp_path / "storage" / "coordinator.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("""
            CREATE TABLE leases (
                task_id TEXT,
                holder_pid INTEGER,
                lane TEXT,
                acquired_at REAL
            )
        """)
        conn.execute(
            "INSERT INTO leases VALUES ('tsk-a', ?, 'lane-1', 1700000000.0)",
            (os.getpid(),),
        )
        conn.execute(
            "INSERT INTO leases VALUES ('tsk-b', 9999999, 'lane-2', 1700000000.0)",
        )
        conn.commit()
    finally:
        conn.close()
    out = _sample_state_integrity(tmp_path, "optimizer_runs")
    leases = out["leases"]
    assert len(leases) == 2
    by_task = {row["task_id"]: row for row in leases}
    assert by_task["tsk-a"]["alive"] is True
    assert by_task["tsk-b"]["alive"] is False


def test_probe_external_mounts_records_latency(monkeypatch, tmp_path):
    monkeypatch.setenv("TRACELENS_ROOT", str(tmp_path))
    monkeypatch.setenv("TRACELENS_INTERNAL_ROOT", str(tmp_path))
    monkeypatch.setenv("INFERENCEX_PATH", "/nonexistent/path/zzz")
    monkeypatch.delenv("OOB_SRC", raising=False)
    out = _probe_external_mounts(timeout_s=5.0)
    by_env = {row["env_name"]: row for row in out}
    assert "TRACELENS_ROOT" in by_env
    assert by_env["TRACELENS_ROOT"]["ok"] is True
    assert "TRACELENS_INTERNAL_ROOT" in by_env
    assert by_env["TRACELENS_INTERNAL_ROOT"]["ok"] is True
    assert "INFERENCEX_PATH" in by_env
    assert by_env["INFERENCEX_PATH"]["ok"] is False
    assert "OOB_SRC" not in by_env


def test_probe_external_mounts_skips_tracelens_root_when_unset(monkeypatch):
    """install.sh now clones AMD-AGI/TraceLens into
    $HYPERLOOM_RUNTIME_DIR/source-mirrors/TraceLens (a session-local
    path), so unset TRACELENS_ROOT must not be probed as a degraded
    external mount. Only operator-set TRACELENS_ROOT (e.g. an explicit
    /wekafs override) should appear in the probe output."""
    monkeypatch.delenv("TRACELENS_ROOT", raising=False)
    monkeypatch.delenv("TRACELENS_INTERNAL_ROOT", raising=False)
    monkeypatch.delenv("INFERENCEX_PATH", raising=False)
    monkeypatch.delenv("OOB_SRC", raising=False)
    out = _probe_external_mounts(timeout_s=5.0)
    by_env = {row["env_name"]: row for row in out}
    assert "TRACELENS_ROOT" not in by_env
    assert "TRACELENS_INTERNAL_ROOT" not in by_env
    assert "INFERENCEX_PATH" not in by_env
    assert "OOB_SRC" not in by_env


def test_probe_tracelens_cli_reports_absent(monkeypatch):
    """In CI env the CLI is not present → ``any_present=False``."""
    monkeypatch.setattr(
        "robustness_agent.sources.local_probe.shutil.which",
        lambda _n: None,
    )
    out = _probe_tracelens_cli()
    assert out["any_present"] is False
    assert all(v is False for v in out["found"].values())


def test_probe_tracelens_cli_reports_present(monkeypatch):
    monkeypatch.setattr(
        "robustness_agent.sources.local_probe.shutil.which",
        lambda name: "/usr/local/bin/" + name
        if name == "TraceLens_generate_perf_report_pytorch_inference"
        else None,
    )
    out = _probe_tracelens_cli()
    assert out["any_present"] is True
    assert out["found"]["TraceLens_generate_perf_report_pytorch_inference"] is True


@pytest.mark.asyncio
async def test_fetch_populates_state_and_deps(tmp_path, monkeypatch):
    """Smoke: fetch() exposes both ``local_state_integrity`` and
    ``local_external_deps``."""
    _write(tmp_path / "state.json", json.dumps({"baseline_tput": 1.0}))
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("TRACELENS_ROOT", str(tmp_path))
    monkeypatch.setenv("TRACELENS_INTERNAL_ROOT", str(tmp_path))
    cfg = LocalProbeConfig(
        session_dir=tmp_path,
        disk_mountpoints=(),
        process_patterns=(),
        ray_probe_enabled=False, fd_probe_enabled=False,
        decision_audit_enabled=False, preflight_enabled=False,
        critic_health_enabled=False,
    )
    data = await LocalProbeSource(cfg).fetch(ctx=None)
    assert data.local_state_integrity["state_json"]["valid"] is True
    assert isinstance(data.local_external_deps.get("mounts"), list)
