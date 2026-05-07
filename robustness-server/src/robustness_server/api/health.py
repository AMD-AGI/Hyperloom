"""Liveness and readiness probes.

``/healthz`` returns 200 as long as the process is up. ``/readyz``
performs a single ``SELECT 1`` against the database pool so K8s only
routes traffic once Postgres is reachable. Migration application is
gated separately at startup; if migrations failed the lifespan hook
raises and the pod restarts before reaching ready.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..store import get_database

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
async def readyz() -> dict[str, str]:
    db = get_database()
    try:
        async with db.pool.acquire() as conn:
            value = await conn.fetchval("SELECT 1")
        if value != 1:
            raise RuntimeError(f"unexpected SELECT 1 value: {value!r}")
    except Exception as exc:  # noqa: BLE001 — bubble up as 503
        raise HTTPException(status_code=503, detail=f"db not ready: {exc}") from exc
    return {"status": "ok"}
