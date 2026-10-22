# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Two explore grids asking for the same configurations are one task, however they were worded."""

from __future__ import annotations

from hyperloom.orchestrator.loop.intent_router import _explore_grid_key

_DISABLE_RADIX = ["--disable-radix-cache"]


def _grid(name: str, *, extra_args, remove_args=None):
    return {"grid": [{"name": name, "extra_args": extra_args, "remove_args": remove_args}]}


def test_the_same_configuration_under_another_name_is_the_same_task():
    """A specialist re-proposing its finding words it differently each round; the measurement is identical."""
    first = _explore_grid_key(
        "explore", _grid("enable-radix-cache", extra_args=[], remove_args=_DISABLE_RADIX), macro_cycle=0
    )
    again = _explore_grid_key(
        "explore", _grid("enable-radix-prefix-cache", extra_args="", remove_args=_DISABLE_RADIX), macro_cycle=0
    )

    assert first == again
    assert first.startswith("explore-grid-c0-")


def test_a_different_configuration_is_a_different_task():
    one = _explore_grid_key("explore", _grid("a", extra_args=["--chunked-prefill-size", "131072"]), macro_cycle=0)
    two = _explore_grid_key("explore", _grid("b", extra_args=["--chunked-prefill-size", "16384"]), macro_cycle=0)

    assert one != two


def test_removing_a_flag_is_not_the_same_as_leaving_it():
    """``enable-radix-cache`` carries no args at all: only ``remove_args`` tells it from the stack it runs on."""
    removes = _explore_grid_key("explore", _grid("on", extra_args=[], remove_args=_DISABLE_RADIX), macro_cycle=0)
    plain = _explore_grid_key("explore", _grid("on", extra_args=[]), macro_cycle=0)

    assert removes != plain


def test_a_later_macro_cycle_measures_against_a_different_stack():
    grid = _grid("enable-radix-cache", extra_args=[], remove_args=_DISABLE_RADIX)

    assert _explore_grid_key("explore", grid, macro_cycle=0) != _explore_grid_key("explore", grid, macro_cycle=1)


def test_the_order_variants_were_listed_in_does_not_matter():
    a = {"grid": [{"name": "x", "extra_args": ["--a", "1"]}, {"name": "y", "extra_args": ["--b", "2"]}]}
    b = {"grid": [{"name": "y", "extra_args": ["--b", "2"]}, {"name": "x", "extra_args": ["--a", "1"]}]}

    assert _explore_grid_key("explore", a, macro_cycle=0) == _explore_grid_key("explore", b, macro_cycle=0)


def test_anything_that_is_not_an_explore_grid_keeps_the_existing_chain():
    """Returning "" hands the key back to the proposer-supplied / tick-scoped fallback."""
    assert _explore_grid_key("specialist", _grid("x", extra_args=[]), macro_cycle=0) == ""
    assert _explore_grid_key("explore", {"grid": []}, macro_cycle=0) == ""
    assert _explore_grid_key("explore", {}, macro_cycle=0) == ""
    assert _explore_grid_key("explore", {"grid": ["not-a-mapping"]}, macro_cycle=0) == ""
