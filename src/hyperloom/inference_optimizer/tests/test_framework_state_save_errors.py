# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A state save that fails on disk is logged and the FRAMEWORK step goes on; any other error is a defect and surfaces."""

from __future__ import annotations

import itertools
import logging
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any

import pytest

from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.phases.framework_reauthor import _AUTHORED_LANE_MAX_ATTEMPTS
from hyperloom.orchestrator.phases.framework_records import ProgressStamp

from .conftest import make_coordinator

_CAND = {"candidate_id": "https://github.com/o/r/pull/1", "batch_id": "b-1", "title": "t", "framework": "sglang"}
_CAND_ID = _CAND["candidate_id"]


@pytest.fixture
def coord(session_dir) -> Coordinator:
    return make_coordinator(session_dir)


async def _map_authoring_specialist(coord: Coordinator) -> None:
    coord.phase_framework._map_authoring_specialist(SimpleNamespace(task_id="spec-1"), cand_id=_CAND_ID, label="t")


async def _rearm_under_cap(coord: Coordinator) -> None:
    coord.shared_state.framework_agent_batches = [{"batch_id": "b-1", "candidates": [dict(_CAND)]}]
    await coord.phase_framework._maybe_rearm_authored_lane(
        {"lane": "perf_framework", "status": "apply_failed"}, {"framework_agent_candidate_id": _CAND_ID}
    )


async def _rearm_at_cap(coord: Coordinator) -> None:
    coord.shared_state.apply_fail_reauthor_attempts = {_CAND_ID: _AUTHORED_LANE_MAX_ATTEMPTS}
    await _rearm_under_cap(coord)


async def _drain_retry_pending(coord: Coordinator) -> None:
    async def _enqueue(retry: Any) -> str:
        return "spec-retry"

    coord.phase_framework._enqueue_author_specialist = _enqueue  # type: ignore[method-assign]
    coord.shared_state.apply_fail_retry_pending = [
        {"lane": "perf_framework", "attempt": 1, "candidate": dict(_CAND), "batch_id": "b-1"}
    ]
    await coord.phase_framework._drain_apply_fail_retry_pending()


async def _stamp_progress(coord: Coordinator) -> None:
    coord.phase_framework._stamp_framework_progress(ProgressStamp(candidate_id=_CAND_ID, status="dropped"))


# (driver, 1-based index of the save call the handler under test wraps)
_SAVE_SITES: dict[str, tuple[Callable[[Coordinator], Awaitable[None]], int]] = {
    "map_authoring_specialist": (_map_authoring_specialist, 1),
    "rearm_retry_pending": (_rearm_under_cap, 1),
    "rearm_cap_stamp": (_rearm_at_cap, 2),
    "drain_apply_fail": (_drain_retry_pending, 1),
    "stamp_progress": (_stamp_progress, 1),
}


def _fail_save_on(coord: Coordinator, monkeypatch: pytest.MonkeyPatch, *, call: int, exc: BaseException) -> None:
    saves = itertools.count(1)

    def _save(session_dir: Any = None) -> None:
        if next(saves) == call:
            raise exc

    monkeypatch.setattr(coord.shared_state, "save", _save)


@pytest.mark.asyncio
@pytest.mark.parametrize("site", sorted(_SAVE_SITES))
async def test_a_failed_state_write_is_logged_and_the_step_returns(coord, monkeypatch, caplog, site):
    driver, call = _SAVE_SITES[site]
    err = OSError("disk full")
    _fail_save_on(coord, monkeypatch, call=call, exc=err)

    with caplog.at_level(logging.DEBUG, logger="hyperloom.orchestrator.phases"):
        await driver(coord)

    assert [r.exc_info[1] for r in caplog.records if r.exc_info] == [err]


@pytest.mark.asyncio
@pytest.mark.parametrize("site", sorted(_SAVE_SITES))
async def test_a_state_serialization_defect_is_not_swallowed(coord, monkeypatch, site):
    driver, call = _SAVE_SITES[site]
    _fail_save_on(coord, monkeypatch, call=call, exc=TypeError("not JSON serializable"))

    with pytest.raises(TypeError, match="not JSON serializable"):
        await driver(coord)
