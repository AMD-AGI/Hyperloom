"""``BRAIN_REGISTRY`` KV watcher.

Claw's brain pod registers itself in a JetStream KV bucket
(``BRAIN_REGISTRY``) under the key ``lock.<sessionId>`` with a value
naming the brain pod (``<namespace>/<podName>``). The lock has a
short TTL (5 minutes by default) and the brain process renews it; an
expire/delete therefore signals the brain has handed off the session.

The watcher folds PUT/DELETE/expire into ``KVAssignmentChange`` and
hands them to the ``SessionRouter``. We intentionally do *not* invent
session rows for brain registrations because the KV may be populated
before any NATS event lands — keeping ``last_event_at`` driven by
events keeps ordering deterministic.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import datetime, timezone
from typing import Any

import nats
from nats.aio.client import Client as NATSClient
from nats.js import JetStreamContext
from nats.js.errors import NotFoundError
from nats.js.kv import KeyValue

from ..config import Settings
from ..models import KVAssignmentChange
from .session_router import SessionRouter

logger = logging.getLogger(__name__)

# Reserved key prefix used by Claw's BRAIN_REGISTRY. Embedded as a
# constant so a Claw rename only needs an update here.
_LOCK_PREFIX = "lock."


class BrainRegistryWatcher:
    """JetStream KV watcher dedicated to the brain registry."""

    def __init__(
        self,
        *,
        settings: Settings,
        router: SessionRouter,
    ) -> None:
        self._settings = settings
        self._router = router
        self._nc: NATSClient | None = None
        self._js: JetStreamContext | None = None
        self._kv: KeyValue | None = None
        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()

    async def start(self) -> None:
        self._nc = await nats.connect(
            servers=self._settings.nats_servers,
            name="hyperloom-robustness-kv-watch",
        )
        self._js = self._nc.jetstream()
        try:
            self._kv = await self._js.key_value(self._settings.nats_kv_bucket)
        except NotFoundError as exc:
            # The bucket is owned by Claw; if it is missing we surface
            # rather than auto-create — we don't want to mask a
            # misconfigured cluster by inventing the data plane.
            raise RuntimeError(
                f"BRAIN_REGISTRY bucket {self._settings.nats_kv_bucket!r} "
                "not found in NATS"
            ) from exc
        self._task = asyncio.create_task(
            self._run(),
            name="brain-registry-watch",
        )
        logger.info(
            "BRAIN_REGISTRY watcher started bucket=%s prefix=%s",
            self._settings.nats_kv_bucket,
            _LOCK_PREFIX,
        )

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        if self._nc is not None:
            with contextlib.suppress(Exception):
                await self._nc.drain()
        logger.info("BRAIN_REGISTRY watcher stopped")

    async def _run(self) -> None:
        """Watch the bucket for the lifetime of the server.

        ``include_history=True`` ensures we replay current locks on
        startup so a freshly-launched server immediately reflects the
        in-flight brain assignments.
        """

        assert self._kv is not None
        try:
            watcher = await self._kv.watchall(include_history=True)
        except Exception:
            logger.exception("kv watchall failed")
            return

        try:
            while not self._stop_event.is_set():
                try:
                    entry = await asyncio.wait_for(watcher.updates(), timeout=5.0)
                except asyncio.TimeoutError:
                    continue
                except Exception:
                    logger.exception("kv watch updates() raised; backing off")
                    await asyncio.sleep(1.0)
                    continue
                if entry is None:
                    # nats-py uses ``None`` to terminate the watch when
                    # the connection drops; restart the loop on the
                    # next iteration via reconnection.
                    return
                await self._handle_entry(entry)
        finally:
            with contextlib.suppress(Exception):
                await watcher.stop()

    async def _handle_entry(self, entry: Any) -> None:
        key: str = getattr(entry, "key", "")
        if not key.startswith(_LOCK_PREFIX):
            return
        session_id = key[len(_LOCK_PREFIX) :]
        if not session_id:
            return

        operation = getattr(entry, "operation", None)
        operation_name = (
            operation.value if hasattr(operation, "value") else str(operation or "PUT")
        ).upper()
        opening = operation_name == "PUT"

        namespace, pod_name = _parse_value(getattr(entry, "value", None))
        if not pod_name:
            # Without a pod name we cannot anchor the assignment. Log
            # and skip rather than open against an empty pod ref.
            logger.warning(
                "kv entry %s has no usable pod name (op=%s); skipping",
                key,
                operation_name,
            )
            return

        change = KVAssignmentChange(
            session_id=session_id,
            pod_name=pod_name,
            pod_namespace=namespace,
            opening=opening,
            observed_at=datetime.now(tz=timezone.utc),
        )
        try:
            await self._router.ingest_kv_change(change)
        except Exception:
            logger.exception(
                "router failed to ingest kv change session=%s pod=%s",
                session_id,
                pod_name,
            )


def _parse_value(raw: bytes | None) -> tuple[str, str]:
    """Decode a KV value into ``(namespace, pod_name)``.

    Claw stores the value as ``"<namespace>/<podName>"`` today;
    ``"<podName>"`` (no namespace) is accepted as a forward-compatible
    fallback so future format changes don't immediately break us.
    Empty / unparseable values yield ``("", "")``.
    """

    if not raw:
        return "", ""
    try:
        value = raw.decode("utf-8").strip()
    except UnicodeDecodeError:
        return "", ""
    if not value:
        return "", ""
    if "/" in value:
        ns, _, name = value.partition("/")
        return ns, name
    return "", value
