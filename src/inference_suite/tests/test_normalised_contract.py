# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract of the normalised staging identity: what makes two stagings the same client, and how a difference is named."""

from __future__ import annotations

import pytest

from inference_suite.contracts import NormalisedError, NormalisedManifest, PackageIdentity

_PIN = "1" * 40
_SHA = "a" * 64


def _manifest(*packages: PackageIdentity, interpreter: str = "3.12.15") -> NormalisedManifest:
    return NormalisedManifest(_PIN, interpreter, "b" * 64, packages)


def test_names_are_canonical_and_order_does_not_change_the_sha() -> None:
    one = _manifest(PackageIdentity("Agentx_Harness", "1.0.13", _SHA), PackageIdentity("aiohttp", "3.13.5", "c" * 64))
    other = _manifest(PackageIdentity("aiohttp", "3.13.5", "c" * 64), PackageIdentity("agentx-harness", "1.0.13", _SHA))
    assert one == other
    assert one.sha256() == other.sha256()
    assert [package.name for package in one.packages] == ["agentx-harness", "aiohttp"]


def test_the_document_round_trips_and_refuses_unknown_keys() -> None:
    manifest = _manifest(PackageIdentity("aiohttp", "3.13.5", _SHA))
    assert NormalisedManifest.from_document(manifest.to_document()) == manifest
    with pytest.raises(NormalisedError):
        NormalisedManifest.from_document({**manifest.to_document(), "paths": []})
    with pytest.raises(NormalisedError):
        NormalisedManifest.from_document({**manifest.to_document(), "schema": "inference-suite-normalised/0"})


@pytest.mark.parametrize(
    "packages",
    [
        [("aiohttp", "3.13.5", _SHA), ("AIOHTTP", "3.13.5", _SHA)],
        [("aiohttp", "3.13.5", "not-a-sha")],
        [("aiohttp", "", _SHA)],
    ],
)
def test_a_package_named_twice_or_without_a_version_or_wheel_sha256_is_refused(packages) -> None:
    with pytest.raises(NormalisedError):
        _manifest(*(PackageIdentity(*package) for package in packages))


def test_differences_name_each_package_that_differs_and_nothing_that_matches() -> None:
    same = PackageIdentity("agentx-harness", "1.0.13", _SHA)
    mine = _manifest(same, PackageIdentity("aiohttp", "3.13.5", _SHA), PackageIdentity("numpy", "2.3.4", _SHA))
    theirs = _manifest(same, PackageIdentity("aiohttp", "3.13.4", _SHA), PackageIdentity("orjson", "3.11.7", _SHA))
    found = mine.differences(theirs)
    assert [line.split()[0] for line in found] == ["aiohttp", "numpy", "orjson"]
    assert "3.13.5" in found[0] and "3.13.4" in found[0]
    assert mine.differences(mine) == ()


def test_another_interpreter_or_inferencex_pin_is_a_difference() -> None:
    mine = _manifest()
    assert [line.split()[0] for line in mine.differences(_manifest(interpreter="3.12.16"))] == ["interpreter"]
    other_pin = NormalisedManifest("2" * 40, "3.12.15", "b" * 64, ())
    assert [line.split()[0] for line in mine.differences(other_pin)] == ["inferencex"]
