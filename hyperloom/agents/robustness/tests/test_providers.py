"""Tests for metrics providers — factory, ring buffer, LocalProvider,
RobustProvider, and helper plumbing.

These cover the same surface a production HTTP client would, but without
network IO. Subprocess + httpx integration points are patched so tests
run without real ROCm or a Primus-Robust-Internal endpoint.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from hyperloom.agents.robustness.config import Config
from hyperloom.agents.robustness.models import GpuSnapshot
from hyperloom.agents.robustness.providers.hybrid import HybridProvider, create_provider
from hyperloom.agents.robustness.providers.local import LocalProvider, _RingBuffer, _run_cmd
from hyperloom.agents.robustness.providers.robust import RobustProvider


# ---------------------------------------------------------------------------
# _RingBuffer — basic + units
# ---------------------------------------------------------------------------


class TestRingBuffer:

    def test_push_and_get(self) -> None:
        buf = _RingBuffer(max_age_seconds=60)
        snap = GpuSnapshot(
            gpu_id=0, utilization=80, vram_used_mb=1000,
            vram_total_mb=2000, temperature_c=70, power_watts=300,
            timestamp=100.0,
        )
        buf.push(snap)
        result = buf.get(0, 120)
        assert len(result) == 1
        assert result[0].utilization == 80

    def test_expiry(self) -> None:
        buf = _RingBuffer(max_age_seconds=10)
        old = GpuSnapshot(
            gpu_id=0, utilization=50, vram_used_mb=500,
            vram_total_mb=1000, temperature_c=60, power_watts=200,
            timestamp=1.0,
        )
        new = GpuSnapshot(
            gpu_id=0, utilization=90, vram_used_mb=900,
            vram_total_mb=1000, temperature_c=80, power_watts=350,
            timestamp=100.0,
        )
        buf.push(old)
        buf.push(new)
        result = buf.get(0, 20)
        assert len(result) == 1
        assert result[0].utilization == 90

    def test_empty_gpu(self) -> None:
        buf = _RingBuffer()
        assert buf.get(99, 60) == []


def _snapshot(gpu_id: int, ts: float, util: float = 0.0) -> GpuSnapshot:
    return GpuSnapshot(
        gpu_id=gpu_id,
        utilization=util,
        vram_used_mb=0.0,
        vram_total_mb=0.0,
        temperature_c=0.0,
        power_watts=0.0,
        timestamp=ts,
    )


class TestRingBufferUnits:
    def test_push_then_get_within_window(self):
        buf = _RingBuffer(max_age_seconds=100)
        for i in range(5):
            buf.push(_snapshot(0, ts=10.0 + i, util=float(i)))
        out = buf.get(0, window_seconds=2)
        assert [s.utilization for s in out] == [2.0, 3.0, 4.0]

    def test_max_age_drops_stale_entries(self):
        buf = _RingBuffer(max_age_seconds=5)
        buf.push(_snapshot(0, ts=0.0))
        buf.push(_snapshot(0, ts=10.0))
        buf.push(_snapshot(0, ts=12.0))
        out = buf.get(0, window_seconds=100)
        assert [s.timestamp for s in out] == [10.0, 12.0]

    def test_get_returns_empty_for_unknown_gpu(self):
        buf = _RingBuffer()
        assert buf.get(99, window_seconds=10) == []


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestCreateProvider:

    @pytest.mark.asyncio
    async def test_no_url_returns_local(self, tmp_path) -> None:
        cfg = Config(session_dir=tmp_path)
        provider = await create_provider(cfg)
        assert isinstance(provider, LocalProvider)

    @pytest.mark.asyncio
    async def test_with_url_returns_hybrid(self, tmp_path) -> None:
        cfg = Config(session_dir=tmp_path, robust_analyzer_url="http://fake:8085")
        provider = await create_provider(cfg)
        assert isinstance(provider, HybridProvider)


# ---------------------------------------------------------------------------
# _run_cmd
# ---------------------------------------------------------------------------


class _FakeProc:
    def __init__(self, *, returncode: int = 0, stdout: bytes = b"", stderr: bytes = b""):
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr

    async def communicate(self):
        return self._stdout, self._stderr


class TestRunCmd:
    @pytest.mark.asyncio
    async def test_returns_rc_and_stdout(self, monkeypatch):
        async def _spawn(*args, **kwargs):
            return _FakeProc(returncode=0, stdout=b"ok")

        monkeypatch.setattr(asyncio, "create_subprocess_shell", _spawn)
        rc, out = await _run_cmd("true")
        assert rc == 0
        assert out == "ok"

    @pytest.mark.asyncio
    async def test_timeout_returns_minus_one(self, monkeypatch):
        async def _spawn(*args, **kwargs):
            class _Slow(_FakeProc):
                async def communicate(self):
                    await asyncio.sleep(10)

            return _Slow()

        monkeypatch.setattr(asyncio, "create_subprocess_shell", _spawn)
        rc, out = await _run_cmd("sleep 99", timeout=0.05)
        assert rc == -1
        assert out == ""

    @pytest.mark.asyncio
    async def test_subprocess_failure_returns_minus_one(self, monkeypatch):
        async def _spawn(*args, **kwargs):
            raise FileNotFoundError("sh missing")

        monkeypatch.setattr(asyncio, "create_subprocess_shell", _spawn)
        rc, out = await _run_cmd("ls")
        assert rc == -1
        assert out == ""


# ---------------------------------------------------------------------------
# LocalProvider
# ---------------------------------------------------------------------------


class TestLocalProvider:
    @pytest.mark.asyncio
    async def test_get_gpu_metrics_parses_rocm_smi_json(self, monkeypatch):
        rocm_payload = {
            "card0": {
                "GPU use (%)": "55",
                "VRAM Total Used Memory (B)": str(int(2 * 1024 * 1024 * 1024)),
                "VRAM Total Memory (B)": str(int(64 * 1024 * 1024 * 1024)),
                "Temperature (Sensor junction) (C)": "78",
                "Average Graphics Package Power (W)": "200",
            },
            "card1": {
                "GPU Usage (%)": "12",
                "VRAM Total Used Memory (B)": "0",
                "VRAM Total Memory (B)": "0",
                "Temperature (Sensor edge) (C)": "40",
                "Average Graphics Package Power (W)": "100",
            },
        }

        async def fake_run(cmd, timeout=10.0):
            if "rocm-smi" in cmd:
                return 0, json.dumps(rocm_payload)
            return 1, ""

        monkeypatch.setattr(
            "robustness_agent.providers.local._run_cmd", fake_run,
        )
        provider = LocalProvider()
        snaps = await provider.get_gpu_metrics()
        assert len(snaps) == 2
        first = next(s for s in snaps if s.gpu_id == 0)
        assert first.utilization == 55.0
        assert first.vram_used_mb == pytest.approx(2 * 1024)
        assert first.temperature_c == 78.0
        hist = await provider.get_gpu_history(0, window_seconds=60)
        assert len(hist) == 1

    @pytest.mark.asyncio
    async def test_get_gpu_metrics_falls_back_to_nvidia_when_rocm_empty(
        self, monkeypatch,
    ):
        async def fake_run(cmd, timeout=10.0):
            if "rocm-smi" in cmd:
                return 0, ""
            if "nvidia-smi" in cmd:
                return 0, "0,80,1024,2048,70,150\n1,bad,0,0,0,0\n"
            return 1, ""

        monkeypatch.setattr(
            "robustness_agent.providers.local._run_cmd", fake_run,
        )
        provider = LocalProvider()
        snaps = await provider.get_gpu_metrics()
        assert len(snaps) == 1
        assert snaps[0].utilization == 80.0
        assert snaps[0].vram_used_mb == 1024.0

    @pytest.mark.asyncio
    async def test_get_gpu_metrics_filters_by_gpu_id(self, monkeypatch):
        async def fake_run(cmd, timeout=10.0):
            if "rocm-smi" in cmd:
                return 0, json.dumps({
                    "card0": {
                        "GPU use (%)": "10",
                        "VRAM Total Used Memory (B)": "0",
                        "VRAM Total Memory (B)": "0",
                        "Temperature (Sensor junction) (C)": "0",
                        "Average Graphics Package Power (W)": "0",
                    },
                    "card1": {
                        "GPU use (%)": "20",
                        "VRAM Total Used Memory (B)": "0",
                        "VRAM Total Memory (B)": "0",
                        "Temperature (Sensor junction) (C)": "0",
                        "Average Graphics Package Power (W)": "0",
                    },
                })
            return 1, ""

        monkeypatch.setattr(
            "robustness_agent.providers.local._run_cmd", fake_run,
        )
        provider = LocalProvider()
        snaps = await provider.get_gpu_metrics(gpu_id=1)
        assert len(snaps) == 1 and snaps[0].gpu_id == 1

    @pytest.mark.asyncio
    async def test_get_gpu_metrics_handles_malformed_json(self, monkeypatch):
        async def fake_run(cmd, timeout=10.0):
            return 0, "{not json"

        monkeypatch.setattr(
            "robustness_agent.providers.local._run_cmd", fake_run,
        )
        provider = LocalProvider()
        assert await provider.get_gpu_metrics() == []

    @pytest.mark.asyncio
    async def test_get_process_list_parses_ps_output(self, monkeypatch):
        ps_output = "\n".join([
            "root  100  0.0  0.1  1234  4096 ?  S  10:00  0:00  /usr/bin/python3 a.py",
            "root  101  0.0  0.1  1234  notnum ?  S  10:00  0:00  /usr/bin/python3 b.py",
            "garbage too short",
        ])

        async def fake_run(cmd, timeout=10.0):
            assert "ps aux" in cmd
            return 0, ps_output

        monkeypatch.setattr(
            "robustness_agent.providers.local._run_cmd", fake_run,
        )
        provider = LocalProvider()
        procs = await provider.get_process_list()
        assert any(p.pid == 100 for p in procs)
        assert any(p.pid == 101 and p.rss_mb == 0 for p in procs)

    @pytest.mark.asyncio
    async def test_get_disk_usage_parses_df_output(self, monkeypatch):
        df_output = "Filesystem 1G-blocks Used Avail Use% Mounted\n" \
                    "/dev/sda1 100G 60G 40G 60% /\n"

        async def fake_run(cmd, timeout=10.0):
            return 0, df_output

        monkeypatch.setattr(
            "robustness_agent.providers.local._run_cmd", fake_run,
        )
        provider = LocalProvider()
        snapshots = await provider.get_disk_usage("/")
        assert len(snapshots) == 1
        snap = snapshots[0]
        assert snap.mount == "/"
        assert snap.total_gb == 100.0
        assert snap.used_gb == 60.0

    @pytest.mark.asyncio
    async def test_disk_usage_handles_non_zero_rc(self, monkeypatch):
        async def fake_run(cmd, timeout=10.0):
            return 1, ""

        monkeypatch.setattr(
            "robustness_agent.providers.local._run_cmd", fake_run,
        )
        provider = LocalProvider()
        assert await provider.get_disk_usage("/") == []

    @pytest.mark.asyncio
    async def test_get_fault_events_returns_empty(self):
        provider = LocalProvider()
        assert await provider.get_fault_events(0.0) == []

    @pytest.mark.asyncio
    async def test_check_available_always_true(self):
        provider = LocalProvider()
        assert await provider.check_available() is True


# ---------------------------------------------------------------------------
# RobustProvider
# ---------------------------------------------------------------------------


class _FakeHttpResponse:
    def __init__(self, status_code: int = 200, payload: Any = None):
        self.status_code = status_code
        self._payload = payload or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "err", request=None, response=None,  # type: ignore[arg-type]
            )

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, responses: dict[tuple[str, str | None], Any]):
        self._responses = responses
        self.closed = False

    async def get(self, path: str, params=None, timeout=None):
        params = params or {}
        candidates = [path]
        if "query" in params:
            candidates.insert(0, f"{path}?{params['query']}")
        for key in candidates:
            if (key, None) in self._responses:
                resp = self._responses[(key, None)]
                if isinstance(resp, Exception):
                    raise resp
                return resp
        if (path, None) in self._responses:
            resp = self._responses[(path, None)]
            if isinstance(resp, Exception):
                raise resp
            return resp
        return _FakeHttpResponse(200, {"data": {"result": []}})

    async def aclose(self):
        self.closed = True


@pytest.fixture
def provider_with_responses(monkeypatch):
    def _factory(responses):
        provider = RobustProvider("http://robust.example.com", workload_uid="w1")
        provider._client = _FakeClient(responses)
        return provider

    return _factory


class TestRobustProvider:
    @pytest.mark.asyncio
    async def test_get_gpu_metrics_parses_promql_instant(self, provider_with_responses):
        provider = provider_with_responses({
            ("/api/v1/query", None): _FakeHttpResponse(200, {
                "data": {
                    "result": [
                        {"metric": {"gpu_id": "0"}, "value": [1700000000.0, "55"]},
                        {"metric": {"gpu_id": "bogus"}, "value": [1700000000.0, "x"]},
                    ],
                },
            }),
        })
        snaps = await provider.get_gpu_metrics()
        assert len(snaps) == 1
        assert snaps[0].gpu_id == 0
        assert snaps[0].utilization == 55.0

    @pytest.mark.asyncio
    async def test_get_gpu_metrics_filters_by_gpu_id(self, provider_with_responses):
        provider = provider_with_responses({
            ("/api/v1/query", None): _FakeHttpResponse(200, {
                "data": {"result": [
                    {"metric": {"gpu_id": "0"}, "value": [0, "10"]},
                    {"metric": {"gpu_id": "1"}, "value": [0, "20"]},
                ]},
            }),
        })
        out = await provider.get_gpu_metrics(gpu_id=1)
        assert [s.gpu_id for s in out] == [1]

    @pytest.mark.asyncio
    async def test_get_gpu_metrics_handles_query_error(self, provider_with_responses):
        provider = provider_with_responses({
            ("/api/v1/query", None): RuntimeError("network down"),
        })
        assert await provider.get_gpu_metrics() == []

    @pytest.mark.asyncio
    async def test_get_gpu_history_parses_range(self, provider_with_responses):
        provider = provider_with_responses({
            ("/api/v1/query_range", None): _FakeHttpResponse(200, {
                "data": {"result": [
                    {"values": [
                        [1700000000.0, "10"],
                        [1700000005.0, "20"],
                        [1700000010.0, "bad"],
                    ]},
                ]},
            }),
        })
        out = await provider.get_gpu_history(0, window_seconds=10)
        assert [s.utilization for s in out] == [10.0, 20.0]

    @pytest.mark.asyncio
    async def test_get_gpu_history_handles_error(self, provider_with_responses):
        provider = provider_with_responses({
            ("/api/v1/query_range", None): RuntimeError("boom"),
        })
        assert await provider.get_gpu_history(0, window_seconds=10) == []

    @pytest.mark.asyncio
    async def test_get_fault_events_parses_payload(self, provider_with_responses):
        provider = provider_with_responses({
            ("/api/v1/faults", None): _FakeHttpResponse(200, {
                "data": [
                    {"monitor_id": "m", "category": "gpu", "severity": "high",
                     "message": "ECC", "created_at": 1700000000, "node_name": "n0"},
                    {"monitor_id": "bad", "created_at": "not-a-time"},
                ],
            }),
        })
        events = await provider.get_fault_events(0)
        assert len(events) == 1
        assert events[0].monitor_id == "m"
        assert events[0].node == "n0"

    @pytest.mark.asyncio
    async def test_get_fault_events_returns_empty_on_error(
        self, provider_with_responses,
    ):
        provider = provider_with_responses({
            ("/api/v1/faults", None): RuntimeError("503"),
        })
        assert await provider.get_fault_events(0) == []

    @pytest.mark.asyncio
    async def test_check_available_returns_true_on_200(
        self, provider_with_responses,
    ):
        provider = provider_with_responses({
            ("/health", None): _FakeHttpResponse(200, {}),
        })
        assert await provider.check_available() is True

    @pytest.mark.asyncio
    async def test_check_available_returns_false_on_error(
        self, provider_with_responses,
    ):
        provider = provider_with_responses({
            ("/health", None): RuntimeError("connection refused"),
        })
        assert await provider.check_available() is False

    @pytest.mark.asyncio
    async def test_get_process_list_and_disk_usage_are_empty(
        self, provider_with_responses,
    ):
        provider = provider_with_responses({})
        assert await provider.get_process_list() == []
        assert await provider.get_disk_usage("/") == []

    @pytest.mark.asyncio
    async def test_close_closes_underlying_client(self, provider_with_responses):
        provider = provider_with_responses({})
        await provider.close()
        assert provider._client.closed is True

    def test_parse_faults_accepts_dict_with_faults_key(self):
        provider = RobustProvider("http://x")
        events = provider._parse_faults({"faults": [{
            "monitor_id": "m", "category": "g", "severity": "h",
            "output": "ECC error", "timestamp": 1700000000.0,
        }]})
        assert len(events) == 1
        assert events[0].message == "ECC error"

    def test_parse_faults_returns_empty_for_unknown_shape(self):
        provider = RobustProvider("http://x")
        assert provider._parse_faults(None) == []
        assert provider._parse_faults({"unrelated": "shape"}) == []
