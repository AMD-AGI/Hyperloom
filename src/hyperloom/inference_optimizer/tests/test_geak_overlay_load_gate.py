# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit coverage for the overlay load gate on the GEAK revalidation path."""

from __future__ import annotations

import json
from pathlib import Path

from hyperloom.orchestrator.loop.coordinator_helpers import (
    _geak_overlay_digest,
    _geak_overlay_is_loadable,
)


def _overlay(root: Path, *, sitecustomize: bool = True, manifest: dict | None = None) -> str:
    root.mkdir(parents=True, exist_ok=True)
    if sitecustomize:
        (root / "sitecustomize.py").write_text("# installs the authored kernel\n")
    if manifest is not None:
        (root / "_overlay_manifest.json").write_text(json.dumps(manifest))
    return str(root)


# The gate


def test_empty_overlay_path_is_not_loadable() -> None:
    assert _geak_overlay_is_loadable("") is False


def test_missing_directory_is_not_loadable(tmp_path: Path) -> None:
    # 25 of the 64 declared overlays name a directory that does not exist.
    assert _geak_overlay_is_loadable(str(tmp_path / "absent")) is False


def test_directory_without_sitecustomize_is_not_loadable(tmp_path: Path) -> None:
    # 30 of the 64 exist but hold no sitecustomize.py: inert on launch.
    path = _overlay(tmp_path / "ov", sitecustomize=False)
    assert _geak_overlay_is_loadable(path) is False


def test_sitecustomize_without_a_manifest_is_loadable(tmp_path: Path) -> None:
    # Absence of evidence is not evidence of an empty overlay.
    path = _overlay(tmp_path / "ov")
    assert _geak_overlay_is_loadable(path) is True


def test_manifest_naming_a_module_is_loadable(tmp_path: Path) -> None:
    path = _overlay(
        tmp_path / "ov",
        manifest={"modules": [{"file": "kern.py"}], "rebinds": [], "captures": []},
    )
    assert _geak_overlay_is_loadable(path) is True


def test_config_only_manifest_is_not_loadable(tmp_path: Path) -> None:
    # GEAK's config-only overlay imports cleanly and installs nothing.
    path = _overlay(
        tmp_path / "ov",
        manifest={
            "modules": [],
            "rebinds": [],
            "note": "config-only result: no kernel overlay accepted",
        },
    )
    assert _geak_overlay_is_loadable(path) is False


def test_unreadable_manifest_is_not_loadable(tmp_path: Path) -> None:
    root = tmp_path / "ov"
    _overlay(root)
    (root / "_overlay_manifest.json").write_text("{not json")
    assert _geak_overlay_is_loadable(str(root)) is False


# The digest — the content check the config fingerprint does not do


def test_digest_is_empty_without_a_manifest(tmp_path: Path) -> None:
    # Empty means "no content evidence", never "mismatch".
    assert _geak_overlay_digest(_overlay(tmp_path / "ov")) == ""
    assert _geak_overlay_digest("") == ""


def test_digest_is_stable_for_the_same_overlay(tmp_path: Path) -> None:
    path = _overlay(tmp_path / "ov", manifest={"modules": [{"file": "kern.py"}]})
    assert _geak_overlay_digest(path) == _geak_overlay_digest(path)
    assert _geak_overlay_digest(path) != ""


def test_digest_changes_when_the_bound_target_changes(tmp_path: Path) -> None:
    a = _overlay(tmp_path / "a", manifest={"modules": [{"file": "kern_a.py"}]})
    b = _overlay(tmp_path / "b", manifest={"modules": [{"file": "kern_b.py"}]})
    assert _geak_overlay_digest(a) != _geak_overlay_digest(b)


def test_digest_separates_two_overlays_that_patch_the_same_symbol(tmp_path: Path) -> None:
    # Measured on the campaign: three unrelated sessions share one manifest digest because all three rebind sglang
    # decode_attention.
    target = {"rebinds": [{"target": "sglang...decode_attention", "impl_module": "impl_one"}]}
    other = {"rebinds": [{"target": "sglang...decode_attention", "impl_module": "impl_two"}]}
    a = _overlay(tmp_path / "a", manifest=target)
    b = _overlay(tmp_path / "b", manifest=other)
    assert _geak_overlay_digest(a) != _geak_overlay_digest(b)
