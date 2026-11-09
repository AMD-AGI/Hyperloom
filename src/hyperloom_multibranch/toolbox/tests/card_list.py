# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The kernel's list of who holds the cards, as a toolbox test writes it."""

from __future__ import annotations

from collections.abc import Sequence

from hyperloom.sandbox.card_occupancy import CardReading, Holder

__all__ = ["INTRUDER", "CardList"]

#: A process of an agent that is on card 0.
INTRUDER = Holder(card=0, pid=4242, command="python3 -c import torch")


class CardList:
    """The kernel's list of who holds the cards, as the pool reads it; the test says who holds what."""

    def __init__(self) -> None:
        self.holders: tuple[Holder, ...] = ()
        self.reads = 0
        self.arrives_after_first_hand_out = False

    def __call__(self, cards: Sequence[int]) -> CardReading:
        self.reads += 1
        if self.arrives_after_first_hand_out and self.reads > 1:
            self.holders = (INTRUDER,)
        return CardReading(tuple(holder for holder in self.holders if holder.card in cards))
