# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import asyncio
from pathlib import Path

from hyperloom_multibranch.toolbox.queue import QueuePolicy
from hyperloom_multibranch.toolbox.submission_log import SubmissionLog
from hyperloom_multibranch.toolbox.tests.helpers import ScriptedExecutor, build, req, settle

K1 = QueuePolicy(per_branch_k=1, global_concurrency=9)


class RefusingSink:
    """The notebook's test record sink, refusing result writes while ``down``."""

    def __init__(self, inner) -> None:
        self._inner, self.down = inner, True

    async def write_request(self, request):
        return await self._inner.write_request(request)

    async def write_result(self, result):
        if self.down:
            raise ConnectionError("notebook unavailable")
        return await self._inner.write_result(result)

    async def write_withdrawal(self, withdrawal):
        return await self._inner.write_withdrawal(withdrawal)


async def test_accepted_tests_survive_a_restart_with_the_same_ids_and_numbering(tmp_path: Path) -> None:
    executor = ScriptedExecutor()
    executor.gate = asyncio.Event()
    rig = build(tmp_path, executor=executor, policy=K1)
    ids = [await rig.toolbox.submit_test(req("b1", server_args=f"--s{n}")) for n in range(3)]
    other = await rig.toolbox.submit_test(req("b2"))
    await settle()
    assert executor.started == [ids[0], other]
    await rig.toolbox.close()  # the process dies with one test running and two queued

    again = build(tmp_path)
    await again.toolbox.start()
    await again.toolbox.drain()
    assert sorted(again.executor.started) == sorted([*ids, other]), "no accepted request is lost"
    for request_id in [*ids, other]:
        assert (await again.notebook.record(request_id)).result is not None
    fresh = await again.toolbox.submit_test(req("b1", server_args="--new"))
    assert fresh == again.toolbox._log.request_id("b1", 4), "numbering continues where it stopped"
    assert fresh not in ids
    await again.toolbox.close()


async def test_recovered_tests_keep_their_priority_and_queue_order(tmp_path: Path) -> None:
    executor = ScriptedExecutor()
    executor.gate = asyncio.Event()
    rig = build(tmp_path, executor=executor, policy=QueuePolicy(per_branch_k=9, global_concurrency=1))
    names = [("b1", 4), ("b2", 2), ("b3", 4), ("b4", 0)]
    ids = [await rig.toolbox.submit_test(req(b, priority=p, server_args=f"--{b}")) for b, p in names]
    await settle()
    await rig.toolbox.close()
    again = build(tmp_path, policy=QueuePolicy(per_branch_k=9, global_concurrency=1))
    await again.toolbox.start()
    await again.toolbox.drain()
    assert again.executor.started == [ids[3], ids[1], ids[0], ids[2]]
    await again.toolbox.close()


async def test_a_result_the_notebook_refused_is_kept_and_written_after_the_restart_without_rerunning(
    tmp_path: Path,
) -> None:
    sinks: list[RefusingSink] = []

    def wrap(notebook):
        sinks.append(RefusingSink(notebook))
        return sinks[0]

    rig = build(tmp_path, wrap_sink=wrap)
    request_id = await rig.toolbox.submit_test(req())
    await rig.toolbox.drain()
    assert (await rig.notebook.record(request_id)).result is None
    entry = rig.toolbox._log.get(request_id)
    assert entry is not None and entry.unwritten

    again = build(tmp_path)
    await again.toolbox.start()
    assert again.executor.jobs == [], "a measured test is not measured again"
    assert (await again.notebook.record(request_id)).result is not None
    await again.toolbox.close()


async def test_tick_retries_an_outcome_the_notebook_refused_earlier(tmp_path: Path) -> None:
    sinks: list[RefusingSink] = []

    def wrap(notebook):
        sinks.append(RefusingSink(notebook))
        return sinks[0]

    rig = build(tmp_path, wrap_sink=wrap)
    request_id = await rig.toolbox.submit_test(req())
    await rig.toolbox.drain()
    sinks[0].down = False
    await rig.toolbox.tick()
    assert (await rig.notebook.record(request_id)).result is not None
    assert rig.toolbox._log.get(request_id).settled  # type: ignore[union-attr]


async def test_a_request_logged_but_never_written_to_the_notebook_is_written_on_recovery(tmp_path: Path) -> None:
    rig = build(tmp_path)
    request = req().accepted(rig.toolbox._log.request_id("b1", 1))
    log = SubmissionLog(tmp_path / "toolbox" / "submissions.jsonl", "campaign-1")
    from hyperloom_multibranch.toolbox.submission_log import LogEntry

    log.accept(LogEntry(request.request_id or "", "b1", log.next_sequence("b1"), request.to_document(), 4, 1.0))
    again = build(tmp_path)
    await again.toolbox.start()
    await again.toolbox.drain()
    assert (await again.notebook.record(request.request_id or "")).result is not None
    await again.toolbox.close()


async def test_the_quarantine_list_is_read_back_from_the_notebook_on_start(tmp_path: Path) -> None:
    from hyperloom.contracts.measurement import FailureClass
    from hyperloom_multibranch.toolbox.tests.helpers import failed, ok

    executor = ScriptedExecutor([lambda j: failed(j, FailureClass.INFRA, "bad"), lambda j: ok(j, "good")])
    rig = build(tmp_path, executor=executor)
    await rig.toolbox.submit_test(req())
    await rig.toolbox.drain()
    again = build(tmp_path)
    await again.toolbox.start()
    assert again.toolbox.quarantined == {"bad"}
    await again.toolbox.close()


def test_a_torn_last_line_is_ignored_and_an_aborted_number_is_not_reused(tmp_path: Path) -> None:
    from hyperloom_multibranch.toolbox.submission_log import LogEntry

    path = tmp_path / "log.jsonl"
    log = SubmissionLog(path, "c")
    entry = LogEntry(log.request_id("b", 1), "b", log.next_sequence("b"), {}, 4, 1.0)
    log.accept(entry)
    log.abort(entry.request_id)
    with path.open("a") as handle:
        handle.write('{"type": "accepted", "request_id": "x", "bra')
    reread = SubmissionLog(path, "c")
    assert reread.next_sequence("b") == 2
    assert reread.get(entry.request_id).aborted  # type: ignore[union-attr]
    assert reread.request_id("b", 1) == entry.request_id
    assert SubmissionLog(tmp_path / "other.jsonl", "c2").request_id("b", 1) != entry.request_id


async def test_nothing_is_dispatched_after_close(tmp_path: Path) -> None:
    executor = ScriptedExecutor()
    executor.gate = asyncio.Event()
    rig = build(tmp_path, executor=executor, policy=K1)
    first = await rig.toolbox.submit_test(req(server_args="--a"))
    await rig.toolbox.submit_test(req(server_args="--b"))
    await settle()
    assert executor.started == [first]
    await rig.toolbox.close()
    await settle()
    assert executor.started == [first], "closing cancelled the running test; its slot must not start the queued one"
    await rig.toolbox.submit_test(req(server_args="--c"))
    await settle()
    assert executor.started == [first], "a closed toolbox accepts into its log but dispatches nothing"
