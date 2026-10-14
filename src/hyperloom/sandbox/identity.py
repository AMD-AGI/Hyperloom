# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A sandbox's signing identity: a per-sandbox Ed25519 key pair.

The private key goes to a 0600 file in a 0700 directory that the sandbox file
API cannot reach; only the public key leaves the provider, through metadata.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

__all__ = ["KEY_FILE_NAME", "SigningIdentity", "generate_signing_identity"]

KEY_FILE_NAME = "signing_key.pem"


@dataclass(frozen=True)
class SigningIdentity:
    """A generated identity.

    Attributes:
        private_key_file: Where the PKCS8 PEM private key is stored (0600).
        public_key_pem: The matching public key, SubjectPublicKeyInfo PEM.
    """

    private_key_file: Path
    public_key_pem: str


def generate_signing_identity(directory: Path, owner: tuple[int, int] | None = None) -> SigningIdentity:
    """Generate a fresh key pair and write the private half to ``directory``.

    Args:
        directory: An existing directory owned by the caller; it is made 0700.
        owner: A ``(uid, gid)`` to hand the directory and key file to (needs root).

    Returns:
        SigningIdentity: The key file path and the public key.

    Raises:
        RuntimeError: When the ``cryptography`` package is not installed.
        FileExistsError: When a key file is already there; keys are never reused.
    """
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    except ImportError as exc:
        raise RuntimeError("signing identities need the 'cryptography' package; install hyperloom[hosted]") from exc
    key = Ed25519PrivateKey.generate()
    private_pem = key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    )
    public_pem = key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    directory.chmod(0o700)
    path = directory / KEY_FILE_NAME
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(private_pem)
    if owner is not None:
        os.chown(directory, *owner)
        os.chown(path, *owner)
    return SigningIdentity(path, public_pem.decode("ascii"))
