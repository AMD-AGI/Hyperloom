"""Tests for the pinned tool commits and the active image."""

from __future__ import annotations

import pytest

from hyperloom.measure import pins
from hyperloom.measure.pins import (
    ImagePinError,
    ToolPinError,
    active_image,
    image_from_pod_status,
    pinned_tool_commits,
)

GOOD = "a" * 40
DIGEST = "sha256:" + "b" * 64


def test_pins_come_from_the_shipped_install_script() -> None:
    commits = pinned_tool_commits({})
    script = pins.resources.files("hyperloom.inference_optimizer").joinpath("assets", "install.sh").read_text()
    assert f'MAGPIE_REF="${{MAGPIE_REF:-{commits.magpie}}}"' in script
    assert f'INFERENCEX_REF="${{INFERENCEX_REF:-{commits.inferencex}}}"' in script
    assert len(commits.magpie) == len(commits.inferencex) == 40


def test_the_preflight_default_stays_in_lockstep() -> None:
    # Main moved the default to the public ``inferencex_checkout.DEFAULT_REF``; the old private name is read only
    # until that move is merged here.
    try:
        from hyperloom.inference_optimizer.inferencex_checkout import DEFAULT_REF as default_ref
    except ImportError:
        from hyperloom.inference_optimizer.cli import preflight

        default_ref = preflight._INFERENCEX_REF_DEFAULT

    assert pinned_tool_commits({}).inferencex == default_ref


def test_environment_overrides_win_and_empty_does_not() -> None:
    default = pinned_tool_commits({})
    over = pinned_tool_commits({"MAGPIE_REF": GOOD, "INFERENCEX_REF": ""})
    assert over.magpie == GOOD
    assert over.inferencex == default.inferencex


@pytest.mark.parametrize("bad", ["v0.2.0", "main", "e6833b8", "A" * 40, GOOD + "0"])
def test_a_non_commit_override_is_refused(bad: str) -> None:
    with pytest.raises(ToolPinError, match="not a 40-hex commit"):
        pinned_tool_commits({"INFERENCEX_REF": bad})


class _Missing:
    def read_text(self, encoding: str) -> str:
        raise FileNotFoundError("install.sh")


class _Root:
    def __init__(self, text: str | None) -> None:
        self._text = text

    def joinpath(self, *parts: str) -> object:
        if self._text is None:
            return _Missing()
        return self

    def read_text(self, encoding: str) -> str:
        assert self._text is not None
        return self._text


def test_a_missing_asset_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pins.resources, "files", lambda _name: _Root(None))
    with pytest.raises(ToolPinError, match="cannot be read"):
        pinned_tool_commits({})


def test_a_script_without_the_default_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pins.resources, "files", lambda _name: _Root(f'MAGPIE_REF="${{MAGPIE_REF:-{GOOD}}}"\n'))
    with pytest.raises(ToolPinError, match="no default for INFERENCEX_REF"):
        pinned_tool_commits({})


def test_a_tag_default_in_the_script_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    text = f'MAGPIE_REF="${{MAGPIE_REF:-v1}}"\nINFERENCEX_REF="${{INFERENCEX_REF:-{GOOD}}}"\n'
    monkeypatch.setattr(pins.resources, "files", lambda _name: _Root(text))
    with pytest.raises(ToolPinError, match="MAGPIE_REF from install.sh"):
        pinned_tool_commits({})


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
