"""Single-owner output dispatcher for the synchronous vLLM engine."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EngineDispatchFailure:
    error: Exception


class EngineDispatcher:
    """Ensure exactly one task calls engine.step() and route outputs by request ID."""

    def __init__(self, engine: Any, logger: logging.Logger | None = None) -> None:
        self.engine = engine
        self.logger = logger or logging.getLogger(__name__)
        self.queues: dict[str, asyncio.Queue[Any]] = {}
        self.running = False

    def add_request(self, request_id: str, prompt: str, params: Any) -> asyncio.Queue[Any]:
        if request_id in self.queues:
            raise ValueError(f"duplicate request ID: {request_id}")
        queue: asyncio.Queue[Any] = asyncio.Queue()
        self.queues[request_id] = queue
        try:
            self.engine.add_request(request_id, prompt, params)
        except Exception:
            self.queues.pop(request_id, None)
            raise
        return queue

    def finish(self, request_id: str) -> None:
        self.queues.pop(request_id, None)

    def abort(self, request_id: str) -> None:
        existed = self.queues.pop(request_id, None) is not None
        if not existed:
            return
        try:
            self.engine.abort_request([request_id])
        except Exception as exc:
            self.logger.warning("Could not abort request %s: %s", request_id, exc)

    def step_once(self) -> bool:
        try:
            outputs = self.engine.step()
        except Exception as exc:
            self.logger.exception("Engine step failed")
            failure = EngineDispatchFailure(exc)
            for queue in tuple(self.queues.values()):
                queue.put_nowait(failure)
            return False

        for output in outputs:
            queue = self.queues.get(output.request_id)
            if queue is not None:
                queue.put_nowait(output)
        return bool(outputs)

    async def run(self) -> None:
        self.running = True
        idle_count = 0
        try:
            while self.running:
                busy = self.step_once()
                idle_count = 0 if busy else idle_count + 1
                await asyncio.sleep(0.001 if idle_count < 10 else 0.003)
        finally:
            self.running = False

    def stop(self) -> None:
        self.running = False
