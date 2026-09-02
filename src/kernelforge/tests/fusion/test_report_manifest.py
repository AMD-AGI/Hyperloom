# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``build_manifest`` carries the nomination envelope, and the combine path stays
byte-identical to the pre-multi-patch shape.

The manifest is the wire contract between KernelForge and Hyperloom. The two keys
added for multi-patch -- top-level ``patches`` (one independent sibling per kept
recipe) and ``nomination`` (the round's summary counts) -- must be present on the
multi-patch path and ABSENT-as-null on the combine path so a legacy consumer that
reads only ``artifacts`` sees no change.
"""

from __future__ import annotations

from kernelforge.fusion.models import Diagnosis
from kernelforge.fusion.report import build_manifest


def _diagnosis() -> Diagnosis:
    """A minimal candidate diagnosis -- enough to build a manifest."""
    return Diagnosis(
        launch_bound_share=0.6,
        busy_fraction_of_wall=0.4,
        dominant_categories=["rmsnorm"],
        kernels_per_step=120.0,
        category_shares={"rmsnorm": 0.6},
        is_candidate=True,
        reason="launch-bound decode",
    )


def _base_kwargs() -> dict:
    return {
        "framework": "sglang",
        "model_path": "/models/zaya",
        "model_type": "zaya",
        "diagnosis": _diagnosis(),
        "recipe": None,
    }


def _sibling(name: str, target: str, speedup: float) -> dict:
    return {
        "kernel_name": name,
        "patch_path": f"/out/{name}.patch",
        "target_file": target,
        "kernel_repo": "/venv/site-packages",
        "snapshot_dir": f"/snap/{name}",
        "base_commit": "abc123",
        "micro_speedup": speedup,
        "kind": "fusion",
    }


def test_combine_path_omits_the_nomination_keys():
    """No ``patches`` argument => both keys are null, legacy shape preserved."""
    manifest = build_manifest(**_base_kwargs())

    assert manifest["patches"] is None
    assert manifest["nomination"] is None


def test_multi_patch_carries_every_sibling_in_order():
    patches = [
        _sibling("fuse_a", "/fw/a.py", 1.4),
        _sibling("fuse_b", "/fw/b.py", 1.2),
    ]
    nomination = {"candidates_seen": 3, "resolved": 2, "selected": 2}

    manifest = build_manifest(**_base_kwargs(), patches=patches, nomination=nomination)

    assert [p["kernel_name"] for p in manifest["patches"]] == ["fuse_a", "fuse_b"]
    assert [p["target_file"] for p in manifest["patches"]] == ["/fw/a.py", "/fw/b.py"]
    assert manifest["nomination"] == nomination


def test_empty_patches_list_is_carried_as_a_kept_nothing_run():
    """An empty list is NOT None: the run ran multi-patch and nominated nobody.

    A consumer must be able to tell "multi-patch, zero keepers" (empty list) from
    "combine path, no nomination contract" (null) -- they enqueue differently.
    """
    manifest = build_manifest(
        **_base_kwargs(),
        patches=[],
        nomination={"candidates_seen": 2, "resolved": 0, "selected": 0},
    )

    assert manifest["patches"] == []
    assert manifest["patches"] is not None
    assert manifest["nomination"]["selected"] == 0


def test_the_manifest_copies_the_envelopes_it_is_handed():
    """Mutating the caller's list after the fact must not rewrite the manifest."""
    patches = [_sibling("fuse_a", "/fw/a.py", 1.4)]
    manifest = build_manifest(**_base_kwargs(), patches=patches)

    patches[0]["kernel_name"] = "mutated"
    patches.append(_sibling("fuse_b", "/fw/b.py", 1.1))

    assert len(manifest["patches"]) == 1
    assert manifest["patches"][0]["kernel_name"] == "fuse_a"
