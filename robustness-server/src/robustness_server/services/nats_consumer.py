"""NATS JetStream durable consumer for ``events.>``.

The consumer:

1. Connects to ``settings.nats_servers`` and obtains a JetStream
   handle.
2. Binds (or creates) a *pull* consumer named
   ``settings.nats_durable_name`` against ``settings.nats_stream``
   filtered to ``settings.nats_subject_filter`` (default
   ``events.>``).
3. Pull-fetches in a loop, decodes each message into an
   ``EventEnvelope``, hands it to ``SessionRouter.ingest_event`` and
   ACKs on success. Decode failures are NAK'd with a delay so a
   crashing payload does not poison-queue the stream.

We use a pull consumer (not push) because the server may be horizontally
scaled and pull semantics make load distribution explicit without a
queue group dance.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from datetime import datetime, timezone

import nats
from nats.aio.client import Client as NATSClient
from nats.aio.errors import ErrConnectionClosed, ErrTimeout
from nats.js import JetStreamContext
from nats.js.api import ConsumerConfig, DeliverPolicy
from nats.js.errors import NotFoundError

from ..config import Settings
from ..models import parse_event_envelope
from .session_router import SessionRouter

logger = logging.getLogger(__name__)


class NatsEventConsumer:
    """Long-running JetStream durable consumer.

    Created during the FastAPI lifespan and started as a background
    task; ``stop()`` requests a graceful drain so in-flight messages
    finish ACKing before the pool tears down.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        router: SessionRouter,
        batch_size: int = 32,
        fetch_timeout_seconds: float = 5.0,
    ) -> None:
        self._settings = settings
        self._router = router
        self._batch_size = batch_size
        self._fetch_timeout = fetch_timeout_seconds

        self._nc: NATSClient | None = None
        self._js: JetStreamContext | None = None
        self._sub = None
        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()

    async def start(self) -> None:
        """Connect to NATS, bind the consumer, spawn the pull loop."""

        self._nc = await nats.connect(
            servers=self._settings.nats_servers,
            name="hyperloom-robustness-server",
            error_cb=self._on_error,
        )
        self._js = self._nc.jetstream()
        await self._ensure_consumer()
        self._sub = await self._js.pull_subscribe(
            subject=self._settings.nats_subject_filter,
            durable=self._settings.nats_durable_name,
            stream=self._settings.nats_stream,
        )
        self._task = asyncio.create_task(
            self._run(),
            name="nats-event-consumer",
        )
        logger.info(
            "NATS consumer started: stream=%s durable=%s filter=%s",
            self._settings.nats_stream,
            self._settings.nats_durable_name,
            self._settings.nats_subject_filter,
        )

    async def stop(self) -> None:
        """Signal the loop to drain and close the NATS connection."""

        self._stop_event.set()
        if self._task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        if self._sub is not None:
            with contextlib.suppress(Exception):
                await self._sub.unsubscribe()
        if self._nc is not None:
            with contextlib.suppress(Exception):
                await self._nc.drain()
        logger.info("NATS consumer stopped")

    async def _ensure_consumer(self) -> None:
        """Create the durable consumer on first run; idempotent."""

        assert self._js is not None
        try:
            await self._js.consumer_info(
                self._settings.nats_stream,
                self._settings.nats_durable_name,
            )
            return
        except NotFoundError:
            pass
        await self._js.add_consumer(
            stream=self._settings.nats_stream,
            config=ConsumerConfig(
                durable_name=self._settings.nats_durable_name,
                filter_subject=self._settings.nats_subject_filter,
                deliver_policy=DeliverPolicy.ALL,
                ack_wait=30,
                max_deliver=8,
            ),
        )

    async def _run(self) -> None:
        """Pull → decode → route → ack loop.

        Failures are funneled through ``_handle_message`` so the loop
        itself never propagates an exception (which would kill the
        background task and stall ingest). The loop exits cleanly when
        ``_stop_event`` flips.
        """

        assert self._sub is not None
        while not self._stop_event.is_set():
            try:
                msgs = await self._sub.fetch(
                    batch=self._batch_size,
                    timeout=self._fetch_timeout,
                )
            except (ErrTimeout, asyncio.TimeoutError):
                continue
            except ErrConnectionClosed:
                logger.warning("NATS connection closed; exiting consumer loop")
                return
            except Exception:
                logger.exception("NATS fetch failed; backing off")
                await asyncio.sleep(1.0)
                continue
            for msg in msgs:
                await self._handle_message(msg)

    async def _handle_message(self, msg) -> None:
        try:
            body = json.loads(msg.data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            logger.warning(
                "dropping non-JSON message subject=%s len=%d",
                msg.subject,
                len(msg.data),
            )
            with contextlib.suppress(Exception):
                await msg.ack()
            return

        event = parse_event_envelope(
            subject=msg.subject,
            body=body,
            received_at=datetime.now(tz=timezone.utc),
        )
        if event is None:
            logger.warning(
                "dropping event missing sessionId/type subject=%s keys=%s",
                msg.subject,
                list(body.keys())[:10],
            )
            with contextlib.suppress(Exception):
                await msg.ack()
            return

        try:
            await self._router.ingest_event(event)
        except Exception:
            logger.exception(
                "ingest failed; nak'ing for redelivery (subject=%s session=%s)",
                msg.subject,
                event.session_id,
            )
            with contextlib.suppress(Exception):
                await msg.nak(delay=5)
            return

        with contextlib.suppress(Exception):
            await msg.ack()

    async def _on_error(self, exc: Exception) -> None:
        logger.warning("NATS error: %s", exc)
