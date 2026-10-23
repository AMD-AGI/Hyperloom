"""The tool commits Hyperloom pins and the image the current machine runs.

A runner installs Magpie and InferenceX by commit, so a placeholder or a moving ref
cannot be fetched. The only pin lives in the packaged ``install.sh``; it is read from
there and never copied. The image digest has no file or environment variable the
platform fills in, so it comes from the caller (the pod status read through the
Kubernetes API) or from two environment variables the operator sets.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from importlib import resources
from typing import Mapping

from hyperloom.measure.docs import PREFIXED_SHA256_RE

__all__ = [
    "ActiveImage",
    "ImagePinError",
    "ToolCommits",
    "ToolPinError",
    "active_image",
    "image_from_pod_status",
    "pinned_tool_commits",
]

MAGPIE_REF_VAR = "MAGPIE_REF"
INFERENCEX_REF_VAR = "INFERENCEX_REF"
IMAGE_REF_VAR = "HYPERLOOM_ACTIVE_IMAGE_REF"
IMAGE_DIGEST_VAR = "HYPERLOOM_ACTIVE_IMAGE_DIGEST"

_INSTALL_SCRIPT = ("assets", "install.sh")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_ZERO_DIGEST = "sha256:" + "0" * 64
_DIGEST_IN_ID_RE = re.compile(r"@(sha256:[0-9a-f]{64})$")
_ID_SCHEMES = ("docker-pullable://", "docker://", "containerd://")


class ToolPinError(ValueError):
    """A tool commit could not be read from the shipped pin or is not a 40-hex commit."""


class ImagePinError(ValueError):
    """The active image cannot be named by a real digest."""


@dataclass(frozen=True)
class ToolCommits:
    """The commits a runner installs the measurement tools at."""

    magpie: str
    inferencex: str


@dataclass(frozen=True)
class ActiveImage:
    """The framework image the current machine runs."""

    ref: str
    digest: str


def _shipped_default(script: str, var: str) -> str:
    match = re.search(rf'^{var}="\$\{{{var}:-([^}}"]*)\}}"', script, re.MULTILINE)
    if match is None:
        raise ToolPinError(f"the shipped install.sh has no default for {var}")
    return match.group(1)


def _commit(var: str, value: str, origin: str) -> str:
    if not _COMMIT_RE.fullmatch(value):
        raise ToolPinError(f"{var} from {origin} is {value!r}, not a 40-hex commit; a runner pins by commit")
    return value


def pinned_tool_commits(env: Mapping[str, str] | None = None) -> ToolCommits:
    """The Magpie and InferenceX commits, honouring the overrides ``install.sh`` honours.

    A non-empty ``MAGPIE_REF`` / ``INFERENCEX_REF`` wins, as in the script; either value
    must be a 40-hex commit, so a tag or branch override fails here instead of on a runner.
    """

    env = os.environ if env is None else env
    path = resources.files("hyperloom.inference_optimizer").joinpath(*_INSTALL_SCRIPT)
    try:
        script = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ToolPinError(f"the packaged install.sh cannot be read: {exc}") from exc
    values = []
    for var in (MAGPIE_REF_VAR, INFERENCEX_REF_VAR):
        override = env.get(var, "")
        values.append(
            _commit(var, override, "the environment")
            if override
            else _commit(var, _shipped_default(script, var), "install.sh")
        )
    return ToolCommits(magpie=values[0], inferencex=values[1])


def _image(ref: str, digest: str) -> ActiveImage:
    if not ref:
        raise ImagePinError("the active image has no reference")
    if not PREFIXED_SHA256_RE.fullmatch(digest) or digest == _ZERO_DIGEST:
        raise ImagePinError(f"the active image digest {digest!r} is not a real sha256:<64 hex> digest")
    return ActiveImage(ref=ref, digest=digest)


def active_image(env: Mapping[str, str] | None = None) -> ActiveImage:
    """The image from ``HYPERLOOM_ACTIVE_IMAGE_REF`` and ``HYPERLOOM_ACTIVE_IMAGE_DIGEST``.

    Nothing inside a SaFE container states its own image digest, so the operator (or the
    launcher, from ``image_from_pod_status``) sets both. Missing or malformed values raise.
    """

    env = os.environ if env is None else env
    missing = [var for var in (IMAGE_REF_VAR, IMAGE_DIGEST_VAR) if not env.get(var)]
    if missing:
        raise ImagePinError(f"cannot name the active image: {', '.join(missing)} not set")
    return _image(env[IMAGE_REF_VAR], env[IMAGE_DIGEST_VAR])


def image_from_pod_status(pod: Mapping[str, object], container: str) -> ActiveImage:
    """The image a container runs, from a pod object read through the Kubernetes API.

    Uses ``status.containerStatuses[].imageID``, the digest the runtime resolved; the
    spec's tag is not enough.
    """

    status = pod.get("status")
    statuses = status.get("containerStatuses") if isinstance(status, Mapping) else None
    for entry in statuses if isinstance(statuses, list) else []:
        if isinstance(entry, Mapping) and entry.get("name") == container:
            return _image_from_status(entry, container)
    raise ImagePinError(f"the pod status has no container {container!r}: the pod is not running yet")


def _image_from_status(entry: Mapping[str, object], container: str) -> ActiveImage:
    image_id = str(entry.get("imageID") or "")
    for scheme in _ID_SCHEMES:
        image_id = image_id.removeprefix(scheme)
    match = _DIGEST_IN_ID_RE.search(image_id)
    if match is None:
        raise ImagePinError(f"the imageID of container {container!r} carries no sha256 digest")
    return _image(str(entry.get("image") or image_id.rsplit("@", 1)[0]), match.group(1))
