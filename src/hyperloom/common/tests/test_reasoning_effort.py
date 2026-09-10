# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The shared effort ladder: one vocabulary, and the boundary around it."""

from __future__ import annotations

import pytest

from hyperloom.common.reasoning_effort import (
    DEFAULT_REASONING_EFFORT,
    gateway_reasoning_effort,
    REASONING_EFFORT_LEVELS,
    REASONING_EFFORT_RANK,
    normalize_reasoning_effort,
)


def test_the_ladder_is_what_the_deeper_surface_can_express() -> None:
    """Pinned by value, because the point of the list is what it excludes.

    ``low``..``xhigh`` are common to both surfaces. ``max`` is a Claude level
    the gateway answers with a 400, and it is on the ladder because there is a
    gateway level to project it onto. ``minimal`` and ``none`` are the reverse
    -- the gateway takes them, the Claude CLI does not know them, and there is
    nothing below ``low`` to project them onto -- so they are not levels.
    """
    assert REASONING_EFFORT_LEVELS == ("low", "medium", "high", "xhigh", "max")


def test_rank_orders_cheapest_first() -> None:
    """A ceiling comparison depends on this order, so it is asserted."""
    assert REASONING_EFFORT_RANK["low"] < REASONING_EFFORT_RANK["medium"]
    assert REASONING_EFFORT_RANK["medium"] < REASONING_EFFORT_RANK["high"]
    assert REASONING_EFFORT_RANK["high"] < REASONING_EFFORT_RANK["xhigh"]
    assert REASONING_EFFORT_RANK["xhigh"] < REASONING_EFFORT_RANK["max"]
    assert set(REASONING_EFFORT_RANK) == set(REASONING_EFFORT_LEVELS)


def test_the_default_is_a_level() -> None:
    """A default outside the ladder would fail its own validation."""
    assert DEFAULT_REASONING_EFFORT in REASONING_EFFORT_RANK


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("low", "low"),
        (" HIGH ", "high"),
        ("XHigh", "xhigh"),
        ("max", "max"),
        ("minimal", ""),
        ("none", ""),
        ("turbo", ""),
        ("", ""),
        (None, ""),
    ],
)
def test_normalize_canonicalizes_or_rejects(raw: str | None, expected: str) -> None:
    """Case and whitespace are noise; anything off the ladder is empty string."""
    assert normalize_reasoning_effort(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("max", "xhigh"),
        (" MAX ", "xhigh"),
        ("xhigh", "xhigh"),
        ("low", "low"),
        ("minimal", ""),
        ("turbo", ""),
    ],
)
def test_gateway_projection_lands_max_on_the_deepest_level_it_has(raw: str, expected: str) -> None:
    """The OpenAI protocol 400s on ``max``, so it is sent ``xhigh`` instead.

    Projecting rather than rejecting keeps ``max`` meaning "as deep as this
    surface goes" on both providers, which is what an operator who wrote it
    asked for. Everything else passes through normalization unchanged.
    """
    assert gateway_reasoning_effort(raw) == expected
