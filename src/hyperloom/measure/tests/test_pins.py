"""Tests for the active image."""

from __future__ import annotations

import pytest

from hyperloom.measure.pins import ImagePinError, active_image

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
