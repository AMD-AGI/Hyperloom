# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""forge-distill: the curator edits a copy of the skill, and the patch it leaves applies to the original."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from kernelforge.finalize import DeliveryOptions, finalize_campaign
from kernelforge.finalize.distill import REPORT_HEADING, DistillBrief, DistillError, distill_campaign, distill_prompt
from kernelforge.tests.finalize.test_delivery import CLEAN, _campaign, _reviewer


def _skill_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "sikl"
    skill = repo / ".claude" / "skills" / "kernel-optimization"
    (skill / "gemm").mkdir(parents=True)
    (skill / "SKILL.md").write_text("# Kernel optimization\n\n- [split-K](gemm/split_k.md)\n")
    (skill / "gemm" / "split_k.md").write_text("# Split-K\n\nSplit K when the grid is too small.\n")
    run = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*run, "init", "-q"], check=True)
    subprocess.run([*run, "add", "-A"], check=True)
    subprocess.run([*run, "commit", "-qm", "skill"], check=True)
    return skill


def _curator(briefs: list[DistillBrief], *, fail: bool = False):
    async def distill_fn(*, brief: DistillBrief) -> str | None:
        briefs.append(brief)
        if fail:
            return None
        card = Path(brief.skill_copy) / "gemm" / "split_k.md"
        card.write_text(card.read_text() + "\nA fused reduction beats a second launch at small M (field, ~10%).\n")
        (Path(brief.skill_copy) / "gemm" / "reduction.md").write_text("# Reduction\n")
        return f"Done.\n{REPORT_HEADING}\n- addition, gemm/split_k.md: fused reduction; ablation.md round 13 +4%\n"

    return distill_fn


async def test_the_patch_applies_to_the_skill_and_leaves_it_untouched(tmp_path: Path) -> None:
    workspace, base, best = _campaign(tmp_path)
    report = await finalize_campaign(workspace, review_fn=_reviewer([CLEAN], []), options=DeliveryOptions())
    skill = _skill_repo(tmp_path)
    briefs: list[DistillBrief] = []
    out = tmp_path / "out"
    result = await distill_campaign(workspace, skill_dir=skill, out=out, distill_fn=_curator(briefs))
    brief = briefs[0]
    assert (brief.base, brief.delivered, brief.history) == (base, report["final_commit"], "forge-history")
    assert sorted(result["files"]) == ["gemm/reduction.md", "gemm/split_k.md"] and result["failed"] is False
    assert "fused reduction" not in (skill / "gemm" / "split_k.md").read_text()
    patch = (out / "knowledge_patch.diff").read_text()
    assert "b/.claude/skills/kernel-optimization/gemm/split_k.md" in patch
    applied = subprocess.run(
        ["git", "-C", str(skill.parents[2]), "apply", "--check", str(out / "knowledge_patch.diff")]
    )
    assert applied.returncode == 0
    assert "round 13 +4%" in (out / "distill.md").read_text()


async def test_a_failed_curator_and_a_folder_without_a_skill(tmp_path: Path) -> None:
    workspace, _, _ = _campaign(tmp_path)
    skill = _skill_repo(tmp_path)
    result = await distill_campaign(
        workspace, skill_dir=skill, out=tmp_path / "out", distill_fn=_curator([], fail=True)
    )
    assert result["failed"] is True and result["files"] == []
    with pytest.raises(DistillError, match="no SKILL.md"):
        await distill_campaign(workspace, skill_dir=tmp_path, out=tmp_path / "out", distill_fn=_curator([]))


def test_the_curator_is_held_to_the_audit_rules() -> None:
    prompt = distill_prompt(DistillBrief("/copy", "/ws", "b0", "f1", "forge-history", ("/flydsl",)))
    assert "Edit nothing outside that directory" in prompt and "- /flydsl" in prompt
    assert '"(field)"' in prompt and "not knowledge" in prompt and REPORT_HEADING in prompt
