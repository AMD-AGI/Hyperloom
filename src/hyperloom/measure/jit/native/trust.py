# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Who may vouch for a build, and every reason a stored build is refused.

The build service signs a record with its own Ed25519 key through the result-bundle primitives
(:mod:`hyperloom.measure.signing`); the key stays in the service, never in the build container, because a build runs
candidate-controlled code. A reader holds a list of public keys, each with the trust epoch it was issued in, and a minimum
epoch: a record verifies only under a listed key of that epoch or later. As in :func:`hyperloom.measure.signing.verify`,
the key is never taken from the data being checked: every listed key is tried.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from hyperloom.measure.signing import load_private_key, sign, verify

if TYPE_CHECKING:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

__all__ = ["BuildRefused", "Refusal", "Signer", "Trust", "TrustedKey"]

_ASCII = "ascii"


class Refusal(str, Enum):
    """Why a stored build is not used."""

    BAD_SIGNATURE = "bad_signature"
    STALE_EPOCH = "stale_epoch"
    KEY_MISMATCH = "key_mismatch"
    IMAGE_MISMATCH = "image_mismatch"
    BLOB_MISSING = "blob_missing"
    BLOB_MISMATCH = "blob_mismatch"
    RECORD_INVALID = "record_invalid"


class BuildRefused(Exception):
    """A stored build failed a check; :attr:`reason` says which. The entry is left as it is, never rebuilt over."""

    def __init__(self, reason: Refusal, detail: str) -> None:
        super().__init__(f"{reason.value}: {detail}")
        self.reason = reason


@dataclass(frozen=True)
class TrustedKey:
    public_key_pem: str
    epoch: int


@dataclass(frozen=True)
class Trust:
    """The keys a reader accepts: those of :attr:`min_epoch` or later."""

    keys: tuple[TrustedKey, ...]
    min_epoch: int

    def check(self, raw: bytes, signature: bytes) -> None:
        """Pass when ``signature`` is a listed, current key's signature of exactly ``raw``.

        Raises:
            BuildRefused: ``stale_epoch`` when only a key older than the minimum epoch signed it, else ``bad_signature``.
            SigningError: When a listed public key cannot be used (a configuration fault, not evidence about the data).
        """
        try:
            text = signature.decode(_ASCII)
        except UnicodeDecodeError:
            raise BuildRefused(Refusal.BAD_SIGNATURE, "the signature is not base64 text") from None
        stale = False
        for key in self.keys:
            if verify(raw, text, key.public_key_pem):
                if key.epoch >= self.min_epoch:
                    return
                stale = True
        if stale:
            raise BuildRefused(Refusal.STALE_EPOCH, f"signed by a key older than trust epoch {self.min_epoch}")
        raise BuildRefused(Refusal.BAD_SIGNATURE, "no trusted key signed the record")


@dataclass(frozen=True)
class Signer:
    """The build service's private key; it signs records and nothing else."""

    private: Ed25519PrivateKey = field(repr=False)

    @classmethod
    def from_file(cls, path: Path) -> Signer:
        return cls(load_private_key(path))

    def sign(self, raw: bytes) -> bytes:
        return sign(raw, self.private).encode(_ASCII)
