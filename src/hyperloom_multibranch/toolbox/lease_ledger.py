# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The blackboard's record of the toolbox's leases: a start when cards are granted, an end as cards go back.

A lease's card time counts against the session's GPU budget, and the blackboard's lease records are the one place it is
read from (``BranchProgress.held_card_seconds``). The lease classes of both backends tell a :class:`LeaseLedger`:

- :meth:`LeaseLedger.started` when the cards are granted, before the lease is handed out, at the toolbox's clock;
- :meth:`LeaseLedger.returned` as cards go back to the pool: a local lease's cards one return at a time as the drain
  frees them, a hosted lease's sandbox at once (no card list: every card still out);
- :meth:`LeaseLedger.close` at the toolbox's close, which ends every lease still out with ``session_close``;
- :meth:`LeaseLedger.recover` at the toolbox's start, before it writes anything else, which ends every lease an earlier
  process left open with ``toolbox_lost`` at the last time the blackboard saw that toolbox.

A write that fails is handled by which way the error would bend the budget:

- A start that is not recorded refuses the lease. An unrecorded lease would hold its cards outside the budget for as long
  as its holder keeps it, and nothing has run on the cards yet, so a refusal costs the caller only a retry. A start the
  blackboard refuses as invalid (a branch the lineage has not opened) is ``invalid_request``; any other failure is
  :class:`~hyperloom.sandbox.socket.SandboxUnavailable`.
- An end that is not recorded is logged and the cards go back all the same: they are back in the pool, and keeping them
  out would starve the tests. The lease then stays open in the blackboard and keeps charging its branch until the close
  ends it with ``session_close``, or the next start with ``toolbox_lost``: the budget is overcharged, never undercharged.

Ends may come from any thread (a local lease's drain thread); they are written on the toolbox's event loop, the one
thread that writes the blackboard, in the order they were given.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
import time
from collections.abc import Callable, Sequence

from hyperloom.sandbox.socket import SandboxUnavailable
from hyperloom_multibranch.contracts.blackboard.interfaces import LeaseSink
from hyperloom_multibranch.contracts.blackboard.records import LeaseEnd, LeaseStart
from hyperloom_multibranch.contracts.blackboard.vocab import LeaseEndReason
from hyperloom_multibranch.contracts.toolbox import RequestRejected

__all__ = ["LeaseLedger"]

log = logging.getLogger(__name__)


class LeaseLedger:
    """Writes a toolbox's lease starts and ends to ``sink``, the times taken from ``clock`` (the clock the toolbox times
    its tests with); with no ``sink`` it writes nothing, for lease classes used without a blackboard."""

    def __init__(self, sink: LeaseSink | None = None, clock: Callable[[], float] = time.time) -> None:
        self._sink = sink
        self.clock = clock
        self._guard = threading.Lock()
        self._out: dict[str, set[int]] = {}
        self._unrecorded: set[str] = set()
        self._writes: set[concurrent.futures.Future[None]] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._closed = False

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        """Write the ends on ``loop``, the toolbox's."""

        self._loop = loop

    async def recover(self) -> int:
        """End, ``toolbox_lost``, every lease the blackboard holds open, at the last time it saw the toolbox; returns how
        many. Call it before this toolbox writes anything: its own writes move that time."""

        if self._sink is None:
            return 0
        lost = [hold.start for hold in await self._sink.card_holds() if not hold.closed]
        if not lost:
            return 0
        seen = await self._sink.toolbox_seen_at()
        for start in lost:
            at = start.started_at if seen is None else max(seen, start.started_at)
            await self._write(LeaseEnd(lease_id=start.lease_id, ended_at=at, end_reason=LeaseEndReason.TOOLBOX_LOST))
        log.warning("ended %d leases an earlier toolbox left open, as toolbox_lost", len(lost))
        return len(lost)

    async def started(self, start: LeaseStart) -> None:
        """Record ``start`` before the lease is handed out.

        Raises:
            RequestRejected: ``invalid_request`` when the blackboard refuses the start (a branch the lineage has not
                opened, a lease id already started).
            SandboxUnavailable: When the blackboard could not be written.
        """

        if self._sink is not None:
            try:
                await self._sink.record_lease_start(start)
            except ValueError as exc:
                raise RequestRejected(f"the blackboard refused lease {start.lease_id}: {exc}") from None
            except Exception as exc:  # re-raised as the lease's refusal
                log.exception("lease %s could not be recorded in the blackboard; it is refused", start.lease_id)
                raise SandboxUnavailable(f"lease {start.lease_id} could not be recorded: {exc}") from exc
        with self._guard:
            self._out[start.lease_id] = set(start.card_ids)

    def returned(self, lease_id: str, gpu_ids: Sequence[int] = ()) -> None:
        """Record that ``gpu_ids`` of ``lease_id`` are back in the pool now, every card still out when none are named.

        Safe from any thread; does not wait for the write. A lease this ledger did not start, or one already ended, and
        any return after :meth:`close`, write nothing.
        """

        at = self.clock()
        with self._guard:
            out = self._out.get(lease_id)
            if out is None or self._closed:
                return
            out.difference_update(gpu_ids or tuple(out))
            if not out:
                del self._out[lease_id]
        end = LeaseEnd(lease_id=lease_id, ended_at=at, end_reason=LeaseEndReason.RELEASED, gpu_ids=tuple(gpu_ids))
        self._submit(end)

    async def close(self) -> None:
        """Wait for the ends being written, then end every lease still out, and every lease an end of which was not
        recorded, with ``session_close``; nothing is recorded after this."""

        while True:
            with self._guard:
                pending = list(self._writes)
            if not pending:
                break
            await asyncio.gather(*(asyncio.wrap_future(write) for write in pending), return_exceptions=True)
        with self._guard:
            self._closed = True
            still = list(dict.fromkeys([*self._out, *sorted(self._unrecorded)]))
            self._out.clear()
            self._unrecorded.clear()
        at = self.clock()
        for lease_id in still:
            await self._write(LeaseEnd(lease_id=lease_id, ended_at=at, end_reason=LeaseEndReason.SESSION_CLOSE))

    def _submit(self, end: LeaseEnd) -> None:
        """Write ``end`` on the toolbox's loop, from whichever thread returned the cards."""

        if self._sink is None:
            return
        loop = self._loop
        try:
            if loop is None:
                raise RuntimeError("the toolbox has not started")
            write = asyncio.run_coroutine_threadsafe(self._write(end), loop)
        except RuntimeError as exc:
            log.error("the return of lease %s could not be recorded: %s", end.lease_id, exc)
            with self._guard:
                self._unrecorded.add(end.lease_id)
            return
        with self._guard:
            self._writes.add(write)
        write.add_done_callback(self._written)

    def _written(self, write: concurrent.futures.Future[None]) -> None:
        with self._guard:
            self._writes.discard(write)

    async def _write(self, end: LeaseEnd) -> None:
        """Record ``end``; a failure is logged and leaves the lease to the close or the next start."""

        assert self._sink is not None
        try:
            await self._sink.record_lease_end(end)
        except Exception:  # isolates one end: its lease is kept in _unrecorded for the close to end
            log.exception(
                "the %s end of lease %s could not be recorded; the lease stays open in the blackboard until the close "
                "or the next start ends it",
                end.end_reason.value,
                end.lease_id,
            )
            with self._guard:
                self._unrecorded.add(end.lease_id)
