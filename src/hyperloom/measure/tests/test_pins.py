"""Tests for the active image and the tool commits callers outside this branch still read."""

from __future__ import annotations

import pytest


# Every name multibranch-dev's scheduling.local_peers imports from this module.
from hyperloom.measure.pins import (
    ActiveImage,
    ImagePinError,
    ToolCommits,
    ToolPinError,
    active_image,
    image_from_pod_status,
    pinned_tool_commits,
)

DIGEST = "sha256:" + "b" * 64


def test_active_image_from_environment() -> None:
    image = active_image({"HYPERLOOM_ACTIVE_IMAGE_REF": "reg/img:tag", "HYPERLOOM_ACTIVE_IMAGE_DIGEST": DIGEST})
    assert (image.ref, image.digest) == ("reg/img:tag", DIGEST)


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({}, "HYPERLOOM_ACTIVE_IMAGE_REF, HYPERLOOM_ACTIVE_IMAGE_DIGEST not set"),
        ({"HYPERLOOM_ACTIVE_IMAGE_REF": "r"}, "HYPERLOOM_ACTIVE_IMAGE_DIGEST not set"),
        ({"HYPERLOOM_ACTIVE_IMAGE_REF": "r", "HYPERLOOM_ACTIVE_IMAGE_DIGEST": "sha256:" + "0" * 64}, "not a real"),
        ({"HYPERLOOM_ACTIVE_IMAGE_REF": "r", "HYPERLOOM_ACTIVE_IMAGE_DIGEST": "sha256:abc"}, "not a real"),
        ({"HYPERLOOM_ACTIVE_IMAGE_REF": "r", "HYPERLOOM_ACTIVE_IMAGE_DIGEST": "b" * 64}, "not a real"),
    ],
)
def test_active_image_refuses_missing_or_fake(env: dict[str, str], message: str) -> None:
    with pytest.raises(ImagePinError, match=message):
        active_image(env)


def _pod(image_id: str, name: str = "main") -> dict[str, object]:
    return {"status": {"containerStatuses": [{"name": name, "image": "reg/img:tag", "imageID": image_id}]}}


@pytest.mark.parametrize("prefix", ["", "docker-pullable://", "containerd://"])
def test_image_from_pod_status(prefix: str) -> None:
    image = image_from_pod_status(_pod(f"{prefix}reg/img@{DIGEST}"), "main")
    assert (image.ref, image.digest) == ("reg/img:tag", DIGEST)


def test_image_from_pod_status_refusals() -> None:
    with pytest.raises(ImagePinError, match="not running yet"):
        image_from_pod_status({"status": {}}, "main")
    with pytest.raises(ImagePinError, match="no container 'main'"):
        image_from_pod_status(_pod(f"r@{DIGEST}", "other"), "main")
    with pytest.raises(ImagePinError, match="carries no sha256 digest"):
        image_from_pod_status(_pod("reg/img:tag"), "main")
    with pytest.raises(ImagePinError, match="not a real"):
        image_from_pod_status(_pod("r@sha256:" + "0" * 64), "main")


_LAST_MAGPIE_REF = "06b7cf6e185fdd49483a512e350cb3964ba3bf68"
_LAST_INFERENCEX_REF = "3d5581562f643f9bdeb8410cd924e2c70906c966"


def test_the_tool_commits_are_the_last_ones_pinned_for_magpie_and_its_inferencex_checkout() -> None:
    commits = pinned_tool_commits({})

    assert commits == ToolCommits(magpie=_LAST_MAGPIE_REF, inferencex=_LAST_INFERENCEX_REF)
    assert isinstance(
        active_image({"HYPERLOOM_ACTIVE_IMAGE_REF": "r", "HYPERLOOM_ACTIVE_IMAGE_DIGEST": DIGEST}), ActiveImage
    )


def test_a_non_empty_override_wins_and_an_empty_one_is_ignored() -> None:
    magpie, inferencex = "a" * 40, "c" * 40

    assert pinned_tool_commits({"MAGPIE_REF": magpie, "INFERENCEX_REF": inferencex}) == ToolCommits(magpie, inferencex)
    assert pinned_tool_commits({"MAGPIE_REF": ""}).magpie == _LAST_MAGPIE_REF


@pytest.mark.parametrize("var", ["MAGPIE_REF", "INFERENCEX_REF"])
def test_an_override_that_is_not_a_commit_is_refused_by_name(var: str) -> None:
    with pytest.raises(ToolPinError, match=f"{var} from the environment is 'main'"):
        pinned_tool_commits({var: "main"})
