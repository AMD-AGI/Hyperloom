# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A configuration the queue already holds is not queued a second time inside another grid."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from hyperloom.orchestrator.loop.dispatcher import DispatcherCollaborator


def _dispatcher(*, queued_grids: tuple[tuple[dict, ...], ...] = ()) -> DispatcherCollaborator:
    """A dispatcher whose only live part is the explore queue."""

    async def queued():
        return [SimpleNamespace(kind="explore", params={"grid": list(g)}) for g in queued_grids]

    async def running():
        return []

    dispatcher = DispatcherCollaborator.__new__(DispatcherCollaborator)
    dispatcher._coord = SimpleNamespace(tasks=SimpleNamespace(queued=queued, running=running))
    return dispatcher


def _variant(args: str, name: str = "v") -> dict:
    return {"name": name, "extra_args": args, "args_mode": "append"}


def test_a_configuration_already_queued_is_dropped_from_a_later_grid():
    """One observed queue held 34 variants under 21 configurations; one lever appeared five times.

    Each copy is a server launch and a benchmark on the single serving lane
    that the cycle's reviewed patches were waiting for.
    """
    d = _dispatcher(queued_grids=((_variant("--enable-aiter-allreduce-fusion"),),))

    kept = asyncio.run(
        d.variants_not_already_queued(
            [_variant("--enable-aiter-allreduce-fusion"), _variant("--scheduler-recv-interval 10")]
        )
    )

    assert [v["extra_args"] for v in kept] == ["--scheduler-recv-interval 10"]


def test_a_grid_that_repeats_itself_is_narrowed_too():
    """The same lever three times in one grid is the same lane cost three times."""
    d = _dispatcher()

    kept = asyncio.run(d.variants_not_already_queued([_variant("--x"), _variant("--x", name="again"), _variant("--y")]))

    assert [v["extra_args"] for v in kept] == ["--x", "--y"]


def test_a_grid_of_entirely_new_configurations_is_untouched():
    d = _dispatcher(queued_grids=((_variant("--already"),),))
    grid = [_variant("--a"), _variant("--b")]

    assert asyncio.run(d.variants_not_already_queued(grid)) == grid


def test_a_grid_whose_every_variant_is_queued_comes_back_empty():
    """The caller refuses the proposal on this, rather than queueing a round that measures nothing."""
    d = _dispatcher(queued_grids=((_variant("--a"), _variant("--b")),))

    assert asyncio.run(d.variants_not_already_queued([_variant("--b"), _variant("--a")])) == []


def test_a_missing_or_malformed_grid_is_empty_rather_than_an_error():
    d = _dispatcher()

    assert asyncio.run(d.variants_not_already_queued(None)) == []
    assert asyncio.run(d.variants_not_already_queued("--not-a-grid")) == []
    assert asyncio.run(d.variants_not_already_queued([_variant("--a"), "junk"])) == [_variant("--a")]


def test_the_prompt_and_the_gate_read_one_set():
    """ConversationCollaborator asks the dispatcher rather than walking the queue itself."""
    from hyperloom.orchestrator.loop.conversation import ConversationCollaborator

    d = _dispatcher(queued_grids=((_variant("--seen"),),))
    collaborator = ConversationCollaborator.__new__(ConversationCollaborator)
    collaborator._coord = SimpleNamespace(dispatcher=d)

    assert asyncio.run(collaborator._explore_grid_fingerprints()) == asyncio.run(d.queued_grid_variants())


def test_an_explore_with_no_grid_is_left_for_the_executor_to_seed():
    """A delegated explore may carry no grid at all; the executor builds one.

    Reading "no grid" as "every variant was a duplicate" would refuse a round
    the session is entitled to run.
    """
    from hyperloom.orchestrator.loop.intent_router import IntentRouter

    router = IntentRouter.__new__(IntentRouter)
    router._coord = SimpleNamespace(dispatcher=_dispatcher(queued_grids=((_variant("--seen"),),)))
    params: dict = {}

    assert asyncio.run(router._keep_unqueued_variants("orchestration", params)) is True
    assert "grid" not in params
