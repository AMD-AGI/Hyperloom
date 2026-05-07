"""HTTP routers."""

from __future__ import annotations

from fastapi import FastAPI

from .health import router as health_router
from .sessions import router as sessions_router


def register_routes(app: FastAPI) -> None:
    """Mount all HTTP routes on ``app``.

    Routers are kept here so ``app.py`` stays free of routing details
    and order-dependent imports.
    """

    app.include_router(health_router)
    app.include_router(sessions_router)
