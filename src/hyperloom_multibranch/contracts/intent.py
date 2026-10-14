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
    """The structured description a specialist writes, and claims on ``parent_commit``, before it changes anything."""

    branch: str
    parent_commit: str
    target: str
    mechanism: str
    layer: IntentLayer

    def __post_init__(self) -> None:
        if not _normalized(self.target) or not _normalized(self.mechanism):
            raise ValueError("an intent needs a target and a mechanism")

    @property
    def fingerprint(self) -> str:
        return intent_fingerprint(self.target, self.mechanism, self.layer)


__all__ = ["Intent", "IntentLayer", "intent_fingerprint"]
