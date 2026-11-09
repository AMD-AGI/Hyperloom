# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Ed25519 signing of the result manifest.

The runner signs the raw bytes of ``manifest.json`` with the private key the platform mounted into its sandbox; the
application verifies with the public key the platform reports for that sandbox. The application never holds a private
key, so a process on the application side can produce a signature only for a key of its own, which fails against the
public key taken from the sandbox information: :func:`verify` therefore never reads a key from the data it checks.

``cryptography`` is an optional dependency (``hyperloom[hosted]`` or ``hyperloom[runner]``) and is imported on first use.
Errors never carry key material.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

SIGNATURE_BYTES = 64
_ASCII = "ascii"
FINGERPRINT_PREFIX = "sha256:"
_INSTALL_HINT = "Ed25519 signing needs the 'cryptography' package: install hyperloom[hosted] (or hyperloom[runner])"


class SigningError(Exception):
    """A key could not be loaded or is not an Ed25519 key. Never raised for a signature that merely does not match."""


def _crypto() -> tuple[Any, Any]:
    """Return ``(serialization, ed25519)``, or fail with the install hint when ``cryptography`` is absent."""
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import ed25519
    except ImportError as exc:
        raise SigningError(_INSTALL_HINT) from exc
    return serialization, ed25519


def _unreadable_key_errors() -> tuple[type[Exception], ...]:
    """What ``cryptography`` raises for a key it cannot load; none of them may leak key material."""
    from cryptography.exceptions import UnsupportedAlgorithm

    return (ValueError, TypeError, UnsupportedAlgorithm)


def generate_private_key() -> Ed25519PrivateKey:
    """A fresh private key; for the platform's key issuing and for tests, never for the application's own use."""
    _, ed25519 = _crypto()
    return ed25519.Ed25519PrivateKey.generate()


def private_key_pem(key: Ed25519PrivateKey) -> bytes:
    """PKCS#8 PEM, the format the platform mounts into a sandbox."""
    serialization, _ = _crypto()
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def public_key_pem(key: Ed25519PrivateKey) -> str:
    """SubjectPublicKeyInfo PEM, the format sandbox information carries."""
    serialization, _ = _crypto()
    return (
        key.public_key()
        .public_bytes(encoding=serialization.Encoding.PEM, format=serialization.PublicFormat.SubjectPublicKeyInfo)
        .decode(_ASCII)
    )


def load_private_key(path: Path) -> Ed25519PrivateKey:
    """Load a PKCS#8 PEM Ed25519 private key."""
    serialization, ed25519 = _crypto()
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise SigningError(f"cannot read the signing key file: {type(exc).__name__}") from exc
    try:
        key = serialization.load_pem_private_key(data, password=None)
    except _unreadable_key_errors():
        raise SigningError("the signing key file is not an unencrypted PEM private key") from None
    if not isinstance(key, ed25519.Ed25519PrivateKey):
        raise SigningError("the signing key is not an Ed25519 key")
    return key


def _load_public_key(public_key_pem_text: str) -> Any:
    serialization, ed25519 = _crypto()
    try:
        key = serialization.load_pem_public_key(public_key_pem_text.encode(_ASCII))
    except _unreadable_key_errors():
        raise SigningError("the public key is not a PEM SubjectPublicKeyInfo key") from None
    if not isinstance(key, ed25519.Ed25519PublicKey):
        raise SigningError("the public key is not an Ed25519 key")
    return key


def public_key_fingerprint(public_key_pem_text: str) -> str:
    """``sha256:<hex>`` of the DER SubjectPublicKeyInfo encoding, as the manifest's ``signing_key_fingerprint``."""
    serialization, _ = _crypto()
    der = _load_public_key(public_key_pem_text).public_bytes(
        encoding=serialization.Encoding.DER, format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    return FINGERPRINT_PREFIX + hashlib.sha256(der).hexdigest()


def sign(data: bytes, key: Ed25519PrivateKey) -> str:
    """The base64 Ed25519 signature of exactly ``data``."""
    return base64.b64encode(key.sign(data)).decode(_ASCII)


def verify(data: bytes, signature_b64: str, public_key_pem_text: str) -> bool:
    """True only when ``signature_b64`` is a valid signature of exactly ``data`` under the given public key.

    A signature that is malformed, has the wrong length or does not match returns False. A public key that cannot be
    used raises :class:`SigningError`: that is a caller or platform fault, not evidence about the data.
    """
    key = _load_public_key(public_key_pem_text)
    try:
        signature = base64.b64decode(signature_b64.strip().encode(_ASCII), validate=True)
    except (binascii.Error, UnicodeEncodeError, ValueError):
        return False
    if len(signature) != SIGNATURE_BYTES:
        return False
    from cryptography.exceptions import InvalidSignature

    try:
        key.verify(signature, data)
    except InvalidSignature:
        return False
    return True
