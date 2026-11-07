"""The image the current machine runs, and the tool commits callers outside this branch still ask for.

The image digest has no file or environment variable the platform fills in, so it comes
from the caller (the pod status read through the Kubernetes API) or from two environment
variables the operator sets.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Mapping

from hyperloom.inference_optimizer.inferencex_checkout import DEFAULT_REF as INFERENCEX_DEFAULT_REF
from hyperloom.inference_optimizer.session.paths import MAGPIE_DEFAULT_REF
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

IMAGE_REF_VAR = "HYPERLOOM_ACTIVE_IMAGE_REF"
IMAGE_DIGEST_VAR = "HYPERLOOM_ACTIVE_IMAGE_DIGEST"

_ZERO_DIGEST = "sha256:" + "0" * 64
_DIGEST_IN_ID_RE = re.compile(r"@(sha256:[0-9a-f]{64})$")
_ID_SCHEMES = ("docker-pullable://", "docker://", "containerd://")


class ImagePinError(ValueError):
    """The active image cannot be named by a real digest."""


# ToolPinError, ToolCommits and pinned_tool_commits are a compatibility layer for main's toolbox GPU tests and
# multibranch-dev's scheduling.local_peers, which this branch does not migrate; removed with 5.12 once they are gone.
_TOOL_REFS = (("MAGPIE_REF", MAGPIE_DEFAULT_REF), ("INFERENCEX_REF", INFERENCEX_DEFAULT_REF))
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")


class ToolPinError(ValueError):
    """A tool commit is not a 40-hex commit."""


@dataclass(frozen=True)
class ToolCommits:
    """The Magpie commit preflight installs and the InferenceX commit the checkout Hyperloom detects is held to."""

    magpie: str
    inferencex: str


def pinned_tool_commits(env: Mapping[str, str] | None = None) -> ToolCommits:
    """The Magpie and InferenceX commits, a non-empty ``MAGPIE_REF`` / ``INFERENCEX_REF`` overriding each.

    Raises:
        ToolPinError: When a commit, overridden or not, is not a 40-hex commit; a runner fetches by commit.
    """

    env = os.environ if env is None else env
    commits = []
    for var, default in _TOOL_REFS:
        value, origin = (env[var], "the environment") if env.get(var) else (default, "the shipped pin")
        if not _COMMIT_RE.fullmatch(value):
            raise ToolPinError(f"{var} from {origin} is {value!r}, not a 40-hex commit; a runner pins by commit")
        commits.append(value)
    return ToolCommits(magpie=commits[0], inferencex=commits[1])


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
