# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Where a process finds the native build store and whom it trusts: the variables every reader and the service share.

``HYPERLOOM_BUILD_STORE`` is the store's directory and ``HYPERLOOM_BUILD_TRUST`` a JSON file of the public keys a reader
accepts (:func:`trust_from_file`)::

    {"min_epoch": 1, "keys": [{"public_key_pem": "-----BEGIN PUBLIC KEY-----...", "epoch": 1}]}

A process given neither has no store; one given only one of them is misconfigured, and refuses rather than reading
builds it cannot verify or verifying with nothing to read.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from hyperloom.measure.docs import DocumentError, Reader, parse_json_object
from hyperloom.measure.jit.native.store import BuildStore
from hyperloom.measure.jit.native.trust import Trust, TrustedKey

__all__ = ["BUILD_STORE_ENV", "BUILD_TRUST_ENV", "BuildStoreConfigError", "store_from_environment", "trust_from_file"]

BUILD_STORE_ENV = "HYPERLOOM_BUILD_STORE"
BUILD_TRUST_ENV = "HYPERLOOM_BUILD_TRUST"
_MAX_TRUST_BYTES = 1024 * 1024


class BuildStoreConfigError(ValueError):
    """The build store is named without its trust, or the other way round, or the trust file cannot be read."""


def _key(value: object, path: str) -> TrustedKey:
    reader = Reader(value, path)
    key = TrustedKey(str(reader.text("public_key_pem")), int(reader.integer("epoch", minimum=0) or 0))
    reader.done()
    return key


def trust_from_file(path: Path) -> Trust:
    """The trust a JSON file states (see the module).

    Raises:
        BuildStoreConfigError: The file cannot be read, is too large or is not in that form.
    """
    try:
        with Path(path).open("rb") as stream:
            raw = stream.read(_MAX_TRUST_BYTES + 1)
        if len(raw) > _MAX_TRUST_BYTES:
            raise BuildStoreConfigError(f"the build trust file {path} is larger than {_MAX_TRUST_BYTES} bytes")
        reader = Reader(parse_json_object(raw, what="build trust"), "trust")
        trust = Trust(reader.sequence("keys", _key), int(reader.integer("min_epoch", minimum=0) or 0))
        reader.done()
    except (OSError, DocumentError) as exc:
        raise BuildStoreConfigError(f"the build trust file {path} cannot be used: {exc}") from None
    if not trust.keys:
        raise BuildStoreConfigError(f"the build trust file {path} lists no key")
    return trust


def store_from_environment(environ: Mapping[str, str]) -> BuildStore | None:
    """The store ``HYPERLOOM_BUILD_STORE`` names, read with the trust ``HYPERLOOM_BUILD_TRUST`` names; ``None`` when
    neither is set.

    Raises:
        BuildStoreConfigError: Only one of them is set, or the trust file cannot be used.
    """
    root, trust = environ.get(BUILD_STORE_ENV, "").strip(), environ.get(BUILD_TRUST_ENV, "").strip()
    if not root and not trust:
        return None
    if not root or not trust:
        missing = BUILD_STORE_ENV if not root else BUILD_TRUST_ENV
        raise BuildStoreConfigError(f"the native build store is half configured: {missing} is not set")
    return BuildStore(Path(root), trust_from_file(Path(trust)))
