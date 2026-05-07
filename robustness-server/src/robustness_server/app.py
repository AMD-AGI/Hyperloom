"""FastAPI application factory.

Wiring is intentionally explicit: ``create_app`` builds the FastAPI
instance, registers routers, and arranges the lifespan hooks for the
database pool, robust-api client, NATS consumer, KV watcher, and
workload reconciler. Each collaborator lives in its own module so it
can be tested in isolation; ``app.py`` only knows how to assemble
them.

Background services are optional via ``Settings.enable_*`` flags so
unit tests can boot the HTTP surface without a NATS broker or a
Kubernetes API. When disabled the lifespan logs once and skips the
start call, which keeps test fixtures simple.
"""

from __future__ import annotations

import contextlib
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI

from .api import register_routes
from .config import Settings, get_settings
from .services import (
    BrainRegistryWatcher,
    NatsEventConsumer,
    RobustAPIClient,
    RobustWorkloadLister,
    SessionRouter,
    WorkloadReconciler,
)
from .store import (
    AssignmentsRepository,
    Database,
    EventsRepository,
    SessionsRepository,
)
from .store.pool import install_database

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start every long-running collaborator in dependency order.

    DB first (everything else needs it), then the shared HTTP client,
    then the orchestrator, then the ingest/reconcile services.
    Shutdown reverses the order so requests don't observe partial
    teardown.
    """

    settings: Settings = app.state.settings

    db = Database(settings)
    await db.connect()
    install_database(db)
    app.state.database = db
    logger.info("Database connected (schema=%s)", settings.database_schema)

    robust_client = RobustAPIClient(settings)
    app.state.robust_client = robust_client

    router = SessionRouter(
        sessions=SessionsRepository(db.pool),
        assignments=AssignmentsRepository(db.pool),
        events=EventsRepository(db.pool),
    )
    app.state.session_router = router

    started: list[Any] = []

    if settings.enable_nats_consumer:
        consumer = NatsEventConsumer(settings=settings, router=router)
        try:
            await consumer.start()
            started.append(consumer)
            app.state.nats_consumer = consumer
        except Exception:
            logger.exception("nats consumer failed to start")
    else:
        logger.info("NATS consumer disabled by configuration")

    if settings.enable_kv_watcher:
        watcher = BrainRegistryWatcher(settings=settings, router=router)
        try:
            await watcher.start()
            started.append(watcher)
            app.state.kv_watcher = watcher
        except Exception:
            logger.exception("brain registry watcher failed to start")
    else:
        logger.info("BRAIN_REGISTRY watcher disabled by configuration")

    if settings.enable_workload_reconciler:
        reconciler = WorkloadReconciler(
            settings=settings,
            assignments=AssignmentsRepository(db.pool),
            list_fn=RobustWorkloadLister(
                settings=settings,
                client=robust_client,
            ),
        )
        try:
            await reconciler.start()
            started.append(reconciler)
            app.state.workload_reconciler = reconciler
        except Exception:
            logger.exception("workload reconciler failed to start")
    else:
        logger.info("Workload reconciler disabled by configuration")

    try:
        yield
    finally:
        for service in reversed(started):
            with contextlib.suppress(Exception):
                await service.stop()
        with contextlib.suppress(Exception):
            await robust_client.aclose()
        await db.disconnect()
        logger.info("All services stopped")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a configured FastAPI app.

    Tests construct their own ``Settings`` (often disabling background
    services) to override the database URL / NATS endpoints; production
    goes through the cached ``get_settings()`` defaults.
    """

    settings = settings or get_settings()
    logging.basicConfig(level=settings.log_level)

    app = FastAPI(
        title="Hyperloom Robustness Server",
        version="0.1.0",
        description=(
            "Bridges Primus-Claw NATS / KV with Primus-Robust "
            "pod-dimension metrics and workload catalogue for "
            "session-level views."
        ),
        lifespan=_lifespan,
    )
    app.state.settings = settings
    register_routes(app)
    return app
