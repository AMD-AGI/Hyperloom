# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A keeper the run already smoked must survive the wrapper being killed.

``fusion_manifest.json`` is the only artifact that points at a keeper, and it used to land
once every campaign had returned. Session 20260916T050331Z-94ee8477 lost a 5.011x fusion to
that gap: the loop proved it at 57dB SNR and published it at 10:26, the wrapper's 5400s
timeout fired at 11:10 with the next experiment still running, and the lane reported REVERT
with ``patch=null``.
"""

from __future__ import annotations

import json
from pathlib import Path

from kernelforge.fusion.command import _publish_partial_nomination
from kernelforge.fusion.loop import RecipePatch
from kernelforge.fusion.report import write_manifest


def _patch(out: Path, name: str, *, speedup: float | None, env_flag: str, source_file: str) -> RecipePatch:
    patch_path = out / f"fusion_{name}.patch"
    patch_path.write_text(f"diff --git a/qk.py b/qk.py\n+# {name}\n", encoding="utf-8")
    return RecipePatch(
        kernel_name=name,
        patch_path=str(patch_path),
        source_file=source_file,
        micro_speedup=speedup,
        snapshot_dir=str(out / f".pristine_{name}"),
        base_commit="c0ffee",
        env_flag=env_flag,
    )


def _publisher(out: Path):
    """The run's manifest writer, reduced to the two fields the salvage path reads."""

    def publish(patches, *, artifacts=None, loop=None, **_rest):
        manifest = {
            "verdict": "candidate",
            "patches": patches,
            "artifacts": {"patch": artifacts.patch, "repo_root": artifacts.repo_root} if artifacts else {},
            "fusion_loop": loop or {},
            "fusion": {"source_file": patches[0]["target_file"] if patches else ""},
        }
        return manifest, write_manifest(manifest, out)

    return publish


def test_a_smoked_keeper_is_on_disk_before_the_next_campaign_starts(tmp_path):
    """The proof has to land when the keeper passes, not when the whole run returns."""
    out = tmp_path / "out"
    out.mkdir()
    patch = _patch(
        out,
        "llm_qkvgate_split_qknorm_rope",
        speedup=5.011,
        env_flag="QWEN3_NEXT_FUSED_QKGATE_NORM_ROPE",
        source_file="/fw/python/sglang/srt/models/qwen3_next.py",
    )

    _publish_partial_nomination(_publisher(out), [], patch, out=out, repo_root="/fw")

    manifest = json.loads((out / "fusion_manifest.json").read_text(encoding="utf-8"))
    assert manifest["fusion_loop"]["kept"] is True
    assert manifest["fusion_loop"]["termination_reason"] == "in_progress"
    row = manifest["patches"][0]
    # Both are what makes a salvaged row applicable: integrate builds fusion_env_flags from the
    # flag, and boots the re-baseline server un-gated without it -- measuring the eager path and
    # REVERTing the very fusion this exists to save.
    assert row["env_flag"] == "QWEN3_NEXT_FUSED_QKGATE_NORM_ROPE"
    # apply_kernel_patch and record_source_path both expect the absolute recipe.source_file, not
    # the repo-relative path a diff header carries.
    assert row["target_file"] == "/fw/python/sglang/srt/models/qwen3_next.py"
    assert Path(row["target_file"]).is_absolute()
    assert row["snapshot_dir"] and row["base_commit"] == "c0ffee"


def test_the_strongest_smoked_keeper_leads_and_the_rest_ride_along(tmp_path):
    """Siblings are ranked by the loop's own rule, weakest-first-is-wrong."""
    out = tmp_path / "out"
    out.mkdir()
    smoked: list[RecipePatch] = []
    weak = _patch(out, "weak", speedup=1.2, env_flag="FLAG_WEAK", source_file="/fw/a.py")
    strong = _patch(out, "strong", speedup=5.0, env_flag="FLAG_STRONG", source_file="/fw/b.py")
    publish = _publisher(out)

    _publish_partial_nomination(publish, smoked, weak, out=out, repo_root="/fw")
    _publish_partial_nomination(publish, smoked, strong, out=out, repo_root="/fw")

    manifest = json.loads((out / "fusion_manifest.json").read_text(encoding="utf-8"))
    assert [row["kernel_name"] for row in manifest["patches"]] == ["strong", "weak"]
    assert manifest["fusion_loop"]["best_env_flag"] == "FLAG_STRONG"
    assert manifest["artifacts"]["patch"] == strong.patch_path
    # The legacy singular name follows the strongest, which is what the combine-era salvage reads.
    assert (out / "fusion.patch").read_text(encoding="utf-8") == Path(strong.patch_path).read_text(encoding="utf-8")


def test_an_unmeasured_keeper_sorts_below_a_measured_one(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    smoked: list[RecipePatch] = []
    unmeasured = _patch(out, "unmeasured", speedup=None, env_flag="FLAG_U", source_file="/fw/a.py")
    measured = _patch(out, "measured", speedup=1.05, env_flag="FLAG_M", source_file="/fw/b.py")
    publish = _publisher(out)

    _publish_partial_nomination(publish, smoked, unmeasured, out=out, repo_root="/fw")
    _publish_partial_nomination(publish, smoked, measured, out=out, repo_root="/fw")

    manifest = json.loads((out / "fusion_manifest.json").read_text(encoding="utf-8"))
    assert [row["kernel_name"] for row in manifest["patches"]] == ["measured", "unmeasured"]


def test_a_run_with_no_manifest_writer_still_tracks_its_keepers(tmp_path):
    """The compile-pass and combine paths have no publisher; the keeper still counts."""
    out = tmp_path / "out"
    out.mkdir()
    smoked: list[RecipePatch] = []
    patch = _patch(out, "k", speedup=2.0, env_flag="FLAG", source_file="/fw/a.py")

    _publish_partial_nomination(None, smoked, patch, out=out, repo_root="/fw")

    assert smoked == [patch]
    assert not (out / "fusion_manifest.json").exists()
