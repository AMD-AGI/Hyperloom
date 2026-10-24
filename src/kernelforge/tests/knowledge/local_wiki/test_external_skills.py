# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Skills outside the packaged knowledge tree: how ``--skill-dir`` folders are read and pointed at."""

from __future__ import annotations

import subprocess
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from kernelforge.cli import main
from kernelforge.cli.options import _resolve_skill_dirs
from kernelforge.knowledge.local_wiki.index import build_forge_knowledge
from kernelforge.knowledge.local_wiki.skills import resolve_external_skill, skill_files_opened

SKILL_MD = """---
name: flydsl-kernel-optimization
description: >
  Making FlyDSL kernels fast on gfx950. Includes a dense-GEMM package.
allowed-tools: Read
---

# FlyDSL kernel optimization

| task | read |
|---|---|
| small-m GEMM | `gemm/flydsl_gemm_small_m.md` |
"""


@pytest.fixture
def kb(tmp_path: Path) -> Path:
    for rel in ("hardware", "common_methodology", "languages/flydsl"):
        folder = tmp_path / "kb" / rel
        folder.mkdir(parents=True)
        (folder / "INDEX.md").write_text(f"# {rel} map\n\n- card.md — a card\n", encoding="utf-8")
    return tmp_path / "kb"


def _skill(folder: Path) -> Path:
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")
    return folder


def test_a_skill_is_read_from_its_front_matter(tmp_path: Path) -> None:
    skill = resolve_external_skill(_skill(tmp_path / "skills" / "anything"))
    assert skill.name == "flydsl-kernel-optimization"
    assert skill.description == "Making FlyDSL kernels fast on gfx950."
    assert skill.skill_file == tmp_path / "skills" / "anything" / "SKILL.md"
    assert skill.repository_root is None


def test_a_skill_inside_a_repository_knows_that_repository(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    skill = resolve_external_skill(_skill(repo / ".claude" / "skills" / "flydsl-kernel-optimization"))
    assert skill.repository_root == repo.resolve()


def test_a_folder_without_skill_md_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="holds no SKILL.md"):
        resolve_external_skill(tmp_path)
    with pytest.raises(click.BadParameter, match="holds no SKILL.md"):
        _resolve_skill_dirs(None, None, (str(tmp_path),))


def test_no_skills_leaves_the_block_unchanged(kb: Path) -> None:
    assert build_forge_knowledge(kb, language="flydsl", skills=()) == build_forge_knowledge(kb, language="flydsl")


def test_a_skill_is_inlined_after_the_knowledge_maps(kb: Path, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    skill = resolve_external_skill(_skill(repo / ".claude" / "skills" / "flydsl-kernel-optimization"))

    block = build_forge_knowledge(kb, language="flydsl", defer_all=True, skills=(skill,))

    section = block[block.index("## skill flydsl-kernel-optimization/") :]
    assert block.index("## languages/flydsl/") < block.index("## skill flydsl-kernel-optimization/")
    assert f"`{skill.skill_file}` is inlined below and applies to every round" in section
    assert "| small-m GEMM | `gemm/flydsl_gemm_small_m.md` |" in section
    assert "allowed-tools" not in section
    assert f"`{skill.root.parent}/<name>/SKILL.md`" in section
    assert f"resolves against `{repo.resolve()}`" in section


def test_a_skill_outside_a_repository_names_no_repository(kb: Path, tmp_path: Path) -> None:
    skill = resolve_external_skill(_skill(tmp_path / "loose"))
    block = build_forge_knowledge(kb, language="flydsl", skills=(skill,))
    assert "relative to its repository" not in block


def test_a_skill_is_inlined_even_without_a_knowledge_tree(tmp_path: Path) -> None:
    skill = resolve_external_skill(_skill(tmp_path / "loose"))
    block = build_forge_knowledge(tmp_path / "missing", language="flydsl", skills=(skill,))
    assert "# FlyDSL kernel optimization" in block


def test_the_skill_files_a_session_opened_are_named_once_in_order(tmp_path: Path) -> None:
    skill = resolve_external_skill(_skill(tmp_path / "skill"))
    root = skill.root
    calls = [
        ("Bash", {"command": f"cat {root}/SKILL.md; ls -R {root}/"}),
        ("Read", {"file_path": f"{root}/gemm/flydsl_gemm_small_m.md"}),
        ("Read", {"file_path": f"{root}/gemm/flydsl_gemm_small_m.md", "offset": 40}),
        ("Grep", {"pattern": "split", "path": f"{root}/gemm"}),
        ("Read", {"file_path": f"{tmp_path}/elsewhere/notes.md"}),
    ]
    assert skill_files_opened(calls, [skill]) == [
        "flydsl-kernel-optimization/SKILL.md",
        "flydsl-kernel-optimization/gemm/flydsl_gemm_small_m.md",
    ]
    assert skill_files_opened(calls, []) == []


def test_forge_loop_refuses_a_skill_dir_without_skill_md(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        main,
        ["forge-loop", "--kernel", "k.py", "--driver", "d.py", "--workspace", ".", "--skill-dir", str(tmp_path)],
    )
    assert result.exit_code == 2
    assert "holds no SKILL.md" in result.output
