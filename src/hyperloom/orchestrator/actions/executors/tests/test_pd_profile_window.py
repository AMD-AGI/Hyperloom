# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PD engine profiling of an atomesh/sglangrouter pod hand-off within a timed window."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from hyperloom.orchestrator.actions.executors import _multi_node_server_lifecycle as life
from hyperloom.orchestrator.actions.executors import profile

_PD_WORKERS = {
    "backend": "sglangrouter",
    "prefill_pods": [{"podIP": "10.0.0.1", "podId": "p0", "enginePort": 8010}],
    "decode_pods": [{"podIP": "10.0.0.2", "podId": "d0", "enginePort": 8020}],
}
_ROLE_BY_PORT = {8010: "prefill", 8020: "decode"}


@pytest.fixture
def engines(monkeypatch: pytest.MonkeyPatch):
    """Route PD engine requests to a mock transport; return the ``(role, path, body)`` log."""
    seen: list[tuple[str, str, dict]] = []
    real_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((_ROLE_BY_PORT[request.url.port], request.url.path, json.loads(request.content or b"{}")))
        return httpx.Response(200)

    monkeypatch.setattr(life, "is_multi_node", lambda: True)
    monkeypatch.setattr(life, "_read_state", lambda: _PD_WORKERS)
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real_client(transport=httpx.MockTransport(handler)))
    monkeypatch.delenv("HYPERLOOM_MN_PROFILE_WITH_STACK", raising=False)
    return seen


def test_pod_engines_get_only_the_output_dir_and_stack_flag(engines, tmp_path: Path) -> None:
    asyncio.run(life.trigger_pd_engine_profile("start", {"output_dir": str(tmp_path)}))
    asyncio.run(life.trigger_pd_engine_profile("stop"))

    assert engines == [
        ("prefill", "/start_profile", {"output_dir": str(tmp_path / "prefill" / "0"), "with_stack": False}),
        ("decode", "/start_profile", {"output_dir": str(tmp_path / "decode" / "0"), "with_stack": False}),
        ("prefill", "/stop_profile", {}),
        ("decode", "/stop_profile", {}),
    ]


def test_profile_window_stops_every_engine_after_the_window(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    async def infera(action: str, body: dict | None = None) -> None:
        events.append(f"infera:{action}")

    async def pd(action: str, body: dict | None = None) -> None:
        events.append(f"pd:{action}")

    monkeypatch.setattr(life, "trigger_infera_engine_profile", infera)
    monkeypatch.setattr(life, "trigger_pd_engine_profile", pd)
    monkeypatch.setenv("HYPERLOOM_MN_PROFILE_WARMUP_S", "0")
    monkeypatch.setenv("HYPERLOOM_MN_PROFILE_WINDOW_S", "0")
    monkeypatch.delenv("PROFILE_EXTRA_BODY", raising=False)

    async def _run_body() -> None:
        async with profile._infera_profile_window("/t"):
            await asyncio.sleep(0.05)

    asyncio.run(_run_body())

    assert events == ["infera:start", "pd:start", "infera:stop", "pd:stop"]
