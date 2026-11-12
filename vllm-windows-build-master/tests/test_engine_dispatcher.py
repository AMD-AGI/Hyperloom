from __future__ import annotations

import asyncio
import unittest
from dataclasses import dataclass

from engine_dispatcher import EngineDispatchFailure, EngineDispatcher


@dataclass
class Output:
    request_id: str
    finished: bool = False


class FakeEngine:
    def __init__(self) -> None:
        self.pending: list[Output] = []
        self.added: list[str] = []
        self.aborted: list[list[str]] = []
        self.step_calls = 0

    def add_request(self, request_id, prompt, params) -> None:
        self.added.append(request_id)

    def abort_request(self, request_ids) -> None:
        self.aborted.append(request_ids)

    def step(self):
        self.step_calls += 1
        outputs, self.pending = self.pending, []
        return outputs


class BrokenEngine(FakeEngine):
    def step(self):
        raise RuntimeError("engine failed")


class EngineDispatcherTests(unittest.IsolatedAsyncioTestCase):
    async def test_routes_interleaved_outputs_without_consuming_other_requests(self) -> None:
        engine = FakeEngine()
        dispatcher = EngineDispatcher(engine)
        first = dispatcher.add_request("first", "a", object())
        second = dispatcher.add_request("second", "b", object())
        engine.pending = [Output("second"), Output("first", finished=True)]

        self.assertTrue(dispatcher.step_once())
        self.assertEqual((await first.get()).request_id, "first")
        self.assertEqual((await second.get()).request_id, "second")
        self.assertEqual(engine.step_calls, 1)

    async def test_abort_removes_queue_and_aborts_engine_request(self) -> None:
        engine = FakeEngine()
        dispatcher = EngineDispatcher(engine)
        dispatcher.add_request("request", "prompt", object())
        dispatcher.abort("request")
        self.assertNotIn("request", dispatcher.queues)
        self.assertEqual(engine.aborted, [["request"]])

    async def test_step_failure_reaches_every_waiter(self) -> None:
        dispatcher = EngineDispatcher(BrokenEngine())
        first = dispatcher.add_request("first", "a", object())
        second = dispatcher.add_request("second", "b", object())
        self.assertFalse(dispatcher.step_once())
        self.assertIsInstance(await first.get(), EngineDispatchFailure)
        self.assertIsInstance(await second.get(), EngineDispatchFailure)


if __name__ == "__main__":
    unittest.main()
