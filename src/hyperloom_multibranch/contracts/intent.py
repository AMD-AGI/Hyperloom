# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a specialist is about to try on a commit, and the fingerprint claims and duplicate checks match it on."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum


class IntentLayer(str, Enum):
    """Which layer of the stack an intent changes."""

    CONFIG = "config"
    SOURCE = "source"
    KERNEL = "kernel"


def _normalized(text: str) -> str:
    return " ".join(text.lower().split())


def intent_fingerprint(target: str, mechanism: str, layer: IntentLayer) -> str:
    """Equal for the same target, mechanism and layer however they are spelled.

    A claim is keyed on ``(parent_commit, fingerprint)``, so neither the commit nor the branch enters it: two branches
    trying the same thing on one commit must collide.
    """
    canonical = "\x1f".join(_normalized(part) for part in (target, mechanism, layer.value))
    return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass(frozen=True)
class Intent:
    """The structured description a specialist writes, and claims on ``parent_commit``, before it changes anything.

    ``target_share_pct`` is the target's share of GPU time on ``parent_commit``, read from that commit's own profile,
    when the specialist gives one. It bounds what the intent can gain, and takes no part in the fingerprint.
    """

    branch: str
    parent_commit: str
    target: str
    mechanism: str
    layer: IntentLayer
    target_share_pct: float | None = None

    def __post_init__(self) -> None:
        if not _normalized(self.target) or not _normalized(self.mechanism):
            raise ValueError("an intent needs a target and a mechanism")
        if self.target_share_pct is not None and not 0.0 <= self.target_share_pct <= 100.0:
            raise ValueError(f"a target's share of GPU time is a percentage, not {self.target_share_pct}")

    @property
    def fingerprint(self) -> str:
        return intent_fingerprint(self.target, self.mechanism, self.layer)


__all__ = ["Intent", "IntentLayer", "intent_fingerprint"]
