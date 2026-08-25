# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Overlay loadability and overlay identity for the GEAK revalidation.

``canonical_fingerprint`` covers ``(args, envs)`` only, so the 2b rebench needs
its own evidence that an authored kernel was installed and that the installed
kernel is the one that was dispatched. Two traps found on
``/shared_nfs/hyperloom-claw`` are guarded here:

* GEAK emits a *config-only* overlay -- ``{"modules": [], "rebinds": []}`` plus
  a note -- which imports cleanly and installs nothing. Counting it as loadable
  labels a pure config win as a kernel win.
* ``_overlay_manifest.json`` names the bind *target*, not the kernel body.
  Three unrelated sessions in the campaign share one manifest digest because
  all three patch ``sglang.kernels.ops.attention.decode_attention``.
"""

from __future__ import annotations

import json
from pathlib import Path

from hyperloom.orchestrator.loop.coordinator_helpers import (
    _geak_overlay_digest,
    _geak_overlay_is_loadable,
)

_KERNEL_MANIFEST = {
    "modules": [
        {
            "module": "sglang.kernels.ops.attention.decode_attention",
            "file": "_patched/sglang.kernels.ops.attention.decode_attention.py",
        }
    ],
    "rebinds": [],
    "captures": [],
}

_CONFIG_ONLY_MANIFEST = {
    "modules": [],
    "rebinds": [],
    "note": "config-only result: no kernel overlay accepted (all candidates rejected)",
}


def _overlay(root: Path, manifest: dict | None, body: str = "# kernel\n") -> str:
    """Build an overlay dir; ``manifest=None`` writes no manifest at all."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "sitecustomize.py").write_text("# installs the overlay\n")
    if manifest is not None:
        (root / "_overlay_manifest.json").write_text(json.dumps(manifest))
        for mod in manifest.get("modules") or []:
            target = root / str(mod["file"])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body)
        for rebind in manifest.get("rebinds") or []:
            (root / f"{rebind['impl_module']}.py").write_text(body)
    return root.as_posix()


def test_missing_dir_and_missing_sitecustomize_are_not_loadable(tmp_path: Path) -> None:
    assert _geak_overlay_is_loadable("") is False
    assert _geak_overlay_is_loadable((tmp_path / "absent").as_posix()) is False
    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "_overlay_manifest.json").write_text(json.dumps(_KERNEL_MANIFEST))
    assert _geak_overlay_is_loadable(bare.as_posix()) is False


def test_config_only_overlay_is_not_a_kernel_overlay(tmp_path: Path) -> None:
    """It imports fine and installs nothing, so it must not read as a kernel."""
    path = _overlay(tmp_path / "cfg", _CONFIG_ONLY_MANIFEST)
    assert _geak_overlay_is_loadable(path) is False


def test_manifest_naming_a_bind_is_loadable(tmp_path: Path) -> None:
    assert _geak_overlay_is_loadable(_overlay(tmp_path / "mod", _KERNEL_MANIFEST)) is True
    rebind = {
        "modules": [],
        "rebinds": [
            {
                "target": "vllm._custom_ops:reshape_and_cache",
                "impl_module": "reshape_and_cache_shim",
                "impl_attr": "reshape_and_cache",
            }
        ],
        "captures": [],
    }
    assert _geak_overlay_is_loadable(_overlay(tmp_path / "reb", rebind)) is True


def test_overlay_without_a_manifest_keeps_the_old_behaviour(tmp_path: Path) -> None:
    """Absence of evidence is not evidence of an empty overlay."""
    assert _geak_overlay_is_loadable(_overlay(tmp_path / "nomf", None)) is True


def test_unreadable_manifest_is_refused_not_assumed_good(tmp_path: Path) -> None:
    root = tmp_path / "broken"
    root.mkdir()
    (root / "sitecustomize.py").write_text("# installs\n")
    (root / "_overlay_manifest.json").write_text("{not json")
    assert _geak_overlay_is_loadable(root.as_posix()) is False


def test_digest_separates_same_target_different_body(tmp_path: Path) -> None:
    """The campaign collision: same patched module, different kernel."""
    a = _overlay(tmp_path / "a", _KERNEL_MANIFEST, body="# kernel A\n")
    b = _overlay(tmp_path / "b", _KERNEL_MANIFEST, body="# kernel B\n")
    assert _geak_overlay_digest(a) != _geak_overlay_digest(b)


def test_digest_is_stable_for_identical_content(tmp_path: Path) -> None:
    a = _overlay(tmp_path / "a", _KERNEL_MANIFEST, body="# same\n")
    b = _overlay(tmp_path / "b", _KERNEL_MANIFEST, body="# same\n")
    assert _geak_overlay_digest(a) == _geak_overlay_digest(b)
    assert len(_geak_overlay_digest(a)) == 16


def test_digest_tracks_an_edited_body(tmp_path: Path) -> None:
    path = _overlay(tmp_path / "edit", _KERNEL_MANIFEST, body="# before\n")
    before = _geak_overlay_digest(path)
    target = Path(path) / str(_KERNEL_MANIFEST["modules"][0]["file"])
    target.write_text("# after\n")
    assert _geak_overlay_digest(path) != before


def test_digest_empty_without_a_manifest(tmp_path: Path) -> None:
    """Empty means "no content evidence", never "mismatch"."""
    assert _geak_overlay_digest("") == ""
    assert _geak_overlay_digest(_overlay(tmp_path / "nomf", None)) == ""


def test_digest_survives_a_missing_body(tmp_path: Path) -> None:
    """A referenced body that cannot be read must not collapse the digest."""
    path = _overlay(tmp_path / "gone", _KERNEL_MANIFEST)
    (Path(path) / str(_KERNEL_MANIFEST["modules"][0]["file"])).unlink()
    digest = _geak_overlay_digest(path)
    assert digest and len(digest) == 16
