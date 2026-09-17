# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract for the skill amd/skills imports out of this repo.

The catalog vendors this folder nightly and validates it there, so without
these checks a broken edit lands as a red bot pull request in that repo.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
# The path amd/skills declares in .github/federation.json. Moving the folder
# stops the nightly import until a pull request there follows.
SKILL_DIR = ROOT / "examples" / "skills" / "hyperloom-workload-optimizer"
SKILL_MD = SKILL_DIR / "SKILL.md"

# Limits from docs/skill-requirements.md in amd/skills.
MAX_DESCRIPTION_CHARS = 1024
MAX_BODY_LINES = 500

FRONTMATTER = re.compile(r"\A---\s*\n(?P<frontmatter>.*?)\n---\s*\n(?P<body>.*)\Z", re.DOTALL)


@pytest.fixture(scope="module")
def skill() -> tuple[dict, str]:
    match = FRONTMATTER.match(SKILL_MD.read_text(encoding="utf-8"))
    assert match, f"{SKILL_MD.relative_to(ROOT)} does not open with a YAML frontmatter block"
    frontmatter = yaml.safe_load(match["frontmatter"])
    assert isinstance(frontmatter, dict), "frontmatter is not a YAML mapping"
    return frontmatter, match["body"]


def test_skill_is_where_the_catalog_looks_for_it():
    assert SKILL_MD.is_file(), (
        f"{SKILL_MD.relative_to(ROOT)} is the path amd/skills imports; moving it "
        "needs a federation.json pull request there first"
    )


def test_name_matches_the_directory(skill):
    frontmatter, _ = skill
    # The importer renames the folder and this field together, so a mismatch
    # here ships a skill no agent resolves.
    assert frontmatter.get("name") == SKILL_DIR.name


def test_description_fits_the_catalog_limit(skill):
    frontmatter, _ = skill
    description = frontmatter.get("description") or ""
    assert description, "description is the only part always in an agent's context"
    assert len(description) <= MAX_DESCRIPTION_CHARS, f"{len(description)} characters"


def test_body_fits_the_catalog_limit(skill):
    _, body = skill
    assert len(body.splitlines()) <= MAX_BODY_LINES
