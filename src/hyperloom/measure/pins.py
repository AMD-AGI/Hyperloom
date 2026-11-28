"""The image the current machine runs.

The image digest has no file or environment variable the platform fills in, so it comes
from two environment variables the operator or the hosting layer sets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

from hyperloom.measure.docs import PREFIXED_SHA256_RE

__all__ = [
    "ActiveImage",
    "ImagePinError",
    "active_image",
]

IMAGE_REF_VAR = "HYPERLOOM_ACTIVE_IMAGE_REF"
IMAGE_DIGEST_VAR = "HYPERLOOM_ACTIVE_IMAGE_DIGEST"

_ZERO_DIGEST = "sha256:" + "0" * 64


class ImagePinError(ValueError):
    """The active image cannot be named by a real digest."""


@dataclass(frozen=True)
class ActiveImage:
    """The framework image the current machine runs."""

    ref: str
    digest: str


def _image(ref: str, digest: str) -> ActiveImage:
    if not ref:
        raise ImagePinError("the active image has no reference")
    if not PREFIXED_SHA256_RE.fullmatch(digest) or digest == _ZERO_DIGEST:
        raise ImagePinError(f"the active image digest {digest!r} is not a real sha256:<64 hex> digest")
    return ActiveImage(ref=ref, digest=digest)


def active_image(env: Mapping[str, str] | None = None) -> ActiveImage:
    """The image from ``HYPERLOOM_ACTIVE_IMAGE_REF`` and ``HYPERLOOM_ACTIVE_IMAGE_DIGEST``.

    Nothing inside a SaFE container states its own image digest, so the operator or the
    hosting layer sets both. Missing or malformed values raise.
    """

    env = os.environ if env is None else env
    missing = [var for var in (IMAGE_REF_VAR, IMAGE_DIGEST_VAR) if not env.get(var)]
    if missing:
        raise ImagePinError(f"cannot name the active image: {', '.join(missing)} not set")
    return _image(env[IMAGE_REF_VAR], env[IMAGE_DIGEST_VAR])
