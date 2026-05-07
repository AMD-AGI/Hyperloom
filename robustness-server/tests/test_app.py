"""App factory smoke test.

Builds the FastAPI app with a Settings instance that disables
side-effecting startup (no DB connection) by patching the lifespan to
a no-op. This guarantees the basic wiring (router registration,
configuration plumbing) compiles and boots without external services.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import robustness_server.app as app_module
from robustness_server.config import Settings


@asynccontextmanager
async def _noop_lifespan(_: FastAPI) -> AsyncIterator[None]:
    yield


def test_create_app_registers_health(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "_lifespan", _noop_lifespan)
    settings = Settings(apply_migrations_on_start=False)

    app = app_module.create_app(settings=settings)

    routes = {r.path for r in app.routes}
    assert "/healthz" in routes
    assert "/readyz" in routes


def test_healthz_returns_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "_lifespan", _noop_lifespan)
    settings = Settings(apply_migrations_on_start=False)
    app = app_module.create_app(settings=settings)

    with TestClient(app) as client:
        resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
