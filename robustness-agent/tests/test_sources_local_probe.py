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
