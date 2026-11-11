"""Tests for the active image."""

from __future__ import annotations

import pytest

from hyperloom.measure.pins import ImagePinError, active_image, image_from_pod_status

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
