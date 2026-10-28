# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""forge-distill: the curator edits a copy of the skill, and the patch it leaves applies to the original."""

from __future__ import annotations

import subprocess
import sys
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
    review = (out / "distill.md").read_text()
    assert "round 13 +4%" in review and "```diff\n" in review and "+A fused reduction beats" in review
    assert "## Lint" not in review and result["lint"] == {}


_CHECK = """import sys
repo, files = sys.argv[1], sys.argv[2:]
print("repo", repo)
bad = [path for path in files if "(field, ~" in open(path).read()]
print("\\n".join(f"unmarked number: {path}" for path in bad))
sys.exit(1 if bad else 0)
"""


async def test_the_skill_guide_reaches_the_curator_and_lint_judges_the_edited_files(tmp_path: Path) -> None:
    workspace, _, _ = _campaign(tmp_path)
    skill = _skill_repo(tmp_path)
    (skill / "CONTRIBUTING.md").write_text("# Adding to this skill\n\nNew pages go to staging/.\n")
    (skill / "check.py").write_text(_CHECK)
    briefs: list[DistillBrief] = []
    out = tmp_path / "out"
    marks = f"{sys.executable} check.py {{repo}}"
    always = f'{sys.executable} -c "import sys; sys.exit(3)"'
    result = await distill_campaign(
        workspace,
        skill_dir=skill,
        out=out,
        distill_fn=_curator(briefs),
        guides=("CONTRIBUTING.md",),
        lint_commands=(marks, always),
    )
    assert briefs[0].guides == (str(out / "skill" / "CONTRIBUTING.md"),)
    prompt = distill_prompt(briefs[0])
    assert f"rules for adding to it: {out / 'skill' / 'CONTRIBUTING.md'}" in prompt and '"(field)"' not in prompt
    repo = str(skill.parents[2])
    checked = marks.replace("{repo}", repo)
    assert sorted(result["lint"]) == sorted([checked, always])
    assert f"repo {repo}" in result["lint"][checked]["output"]
    assert "unmarked number: gemm/split_k.md" in result["lint"][checked]["output"]
    assert result["lint"][checked]["already_failing"] is False and result["lint"][always]["already_failing"] is True
    review = (out / "distill.md").read_text()
    assert "## Lint" in review and "It already failed on those files in the skill." in review
    with pytest.raises(DistillError, match="is not a file of"):
        await distill_campaign(workspace, skill_dir=skill, out=out, distill_fn=_curator([]), guides=("RULES.md",))


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
    assert "not steps to run" in prompt and "one or two lines" in prompt
    assert '"(field)"' in prompt and "not knowledge" in prompt and REPORT_HEADING in prompt
    assert "`KB: no-match` lines" in prompt
