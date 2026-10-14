# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Claim generations: which branch holds which (parent commit, intent) key right now.

Every change of a claim (acquired, renewed, attached to a test, released, completed) writes the next generation file,
``<token>.<generation>.json``, exclusively. Only one writer can create a given generation, on any host of a shared
folder, so two writers racing on one claim cannot both win; the loser reads the new generation and decides again. The
highest generation is in force. Expiry is derived from ``expires_at`` and never written.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from hyperloom_multibranch.contracts.notebook.interfaces import ClaimLost
from hyperloom_multibranch.contracts.notebook.records import CLAIM_TOKEN_LENGTH, Claim, ClaimKey
from hyperloom_multibranch.contracts.notebook.vocab import ClaimStatus
from hyperloom_multibranch.notebook import codec
from hyperloom_multibranch.notebook.local.files import encode_json, publish_once

#: How often a change is re-read and retried when other writers keep creating the next generation first.
CHANGE_ATTEMPTS = 8
_GENERATION_DIGITS = 6
_NAME = re.compile(rf"^(?P<token>[0-9a-f]{{{CLAIM_TOKEN_LENGTH}}})\.(?P<generation>\d{{{_GENERATION_DIGITS}}})\.json$")


class LeaseTaken(Exception):
    """A live claim holds the key; ``claim`` is that claim."""

    def __init__(self, claim: Claim) -> None:
        super().__init__(f"{claim.holder} holds the claim on {claim.key.parent_commit}")
        self.claim = claim


def is_live(claim: Claim, now: float) -> bool:
    return claim.status is ClaimStatus.HELD and claim.expires_at > now


def effective(claim: Claim, now: float) -> Claim:
    """``claim`` as callers see it: a held claim nobody renewed in time has expired."""

    expired = claim.status is ClaimStatus.HELD and claim.expires_at <= now
    return replace(claim, status=ClaimStatus.EXPIRED) if expired else claim


class LeaseStore:
    """The claim files of one notebook."""

    def __init__(self, claims_dir: Path) -> None:
        self.root = Path(claims_dir)

    def _path(self, token: str, generation: int) -> Path:
        return self.root / f"{token}.{generation:0{_GENERATION_DIGITS}d}.json"

    def _generations(self, token: str) -> list[int]:
        if not self.root.is_dir():
            return []
        found = (_NAME.match(path.name) for path in self.root.glob(f"{token}.*.json"))
        return sorted(int(match["generation"]) for match in found if match is not None)

    def _load(self, token: str, generation: int) -> Claim:
        return codec.decode(Claim, json.loads(self._path(token, generation).read_text(encoding="utf-8")))

    def current(self, key: ClaimKey) -> Claim | None:
        """The generation in force on ``key``, whatever its status, or ``None`` when it was never claimed."""

        generations = self._generations(key.token)
        return self._load(key.token, generations[-1]) if generations else None

    def all_current(self) -> list[Claim]:
        """The generation in force on every claimed key."""

        if not self.root.is_dir():
            return []
        tokens = sorted({match["token"] for match in map(_NAME.match, (p.name for p in self.root.iterdir())) if match})
        return [self._load(token, self._generations(token)[-1]) for token in tokens]

    def acquire(self, claim: Claim, now: float) -> Claim:
        """Create the next generation for ``claim.key``; raises :class:`LeaseTaken` while a live claim holds it."""

        current = self.current(claim.key)
        if current is not None and is_live(current, now):
            raise LeaseTaken(current)
        acquired = replace(claim, generation=(current.generation if current is not None else 0) + 1)
        if not publish_once(self._path(claim.key.token, acquired.generation), encode_json(codec.encode(acquired))):
            raise LeaseTaken(self._load(claim.key.token, acquired.generation))
        return acquired

    def change(self, key: ClaimKey, update: Callable[[Claim], Claim]) -> Claim:
        """Write the next generation ``update`` makes of the claim in force.

        ``update`` raises :class:`ClaimLost` when the change is not the caller's to make; it is re-applied to the newer
        generation when another writer created one first.
        """

        for _ in range(CHANGE_ATTEMPTS):
            current = self.current(key)
            if current is None:
                raise ClaimLost(f"nobody claimed {key.intent_fingerprint} on {key.parent_commit}")
            changed = replace(update(current), generation=current.generation + 1)
            if publish_once(self._path(key.token, changed.generation), encode_json(codec.encode(changed))):
                return changed
        raise ClaimLost(f"the claim on {key.parent_commit} kept changing under {CHANGE_ATTEMPTS} attempts")


__all__ = ["CHANGE_ATTEMPTS", "LeaseStore", "LeaseTaken", "effective", "is_live"]
