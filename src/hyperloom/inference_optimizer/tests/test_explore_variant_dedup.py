# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A configuration the queue already holds is not queued a second time inside another grid."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.actions.executors.measured_variants import unqueued_variants
from hyperloom.orchestrator.loop._explore_delegate import _keep_unqueued_variants


def _queued(*grids: tuple[dict, ...], stack: dict | None = None) -> list[SimpleNamespace]:
    return [SimpleNamespace(kind="explore", params={**(stack or {}), "grid": list(g)}) for g in grids]


def _variant(args: str, name: str = "v") -> dict:
    return {"name": name, "extra_args": args, "args_mode": "append"}


_OVERLAY = {"name": "patched", "extra_args": "--k 0", "overlay_pythonpath": "/overlays/a"}


@pytest.mark.parametrize(
    ("active", "grid", "params", "kept"),
    [
        pytest.param(
            _queued((_variant("--fusion"),)),
            [_variant("--fusion"), _variant("--recv 10")],
            {},
            [_variant("--recv 10")],
            id="queued-config-dropped",
        ),
        pytest.param(
            [],
            [_variant("--x"), _variant("--x", name="again"), _variant("--y")],
            {},
            [_variant("--x"), _variant("--y")],
            id="repeat-within-grid-dropped",
        ),
        pytest.param(
            _queued((_variant("--already"),)), [_variant("--a"), _variant("--b")], {}, None, id="all-new-untouched"
        ),
        pytest.param(_queued((_variant("--a"), _variant("--b"))), [_variant("--b"), _variant("--a")], {}, [], id="all"),
        pytest.param(
            _queued((_variant("--a"),), stack={"base_tput": 800.0}),
            [_variant("--a")],
            {"base_tput": 820.0},
            None,
            id="other-stack-kept",
        ),
        pytest.param(
            _queued((_OVERLAY,)),
            [{**_OVERLAY, "overlay_pythonpath": "/overlays/b"}],
            {},
            None,
            id="overlay-kept",
        ),
        pytest.param([], [_variant("--a"), "junk"], {}, [_variant("--a")], id="non-dict-dropped"),
    ],
)
def test_unqueued_variants(active, grid, params, kept):
    assert unqueued_variants(grid, params, active) == (grid if kept is None else kept)


def test_an_explore_with_no_grid_is_left_for_the_executor_to_seed():
    """Reading "no grid" as "every variant was a duplicate" would refuse a round the executor seeds itself."""
    params: dict = {}

    assert asyncio.run(_keep_unqueued_variants(None, "orchestration", params, _queued((_variant("--seen"),)))) is True
    assert "grid" not in params
