# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from hyperloom.orchestrator.kernel.patch_conflict_merge import (
    STRATEGY_LLM,
    STRATEGY_STRICT,
    STRATEGY_THREE_WAY,
    STRATEGY_UNION,
    apply_patch_resolving_conflicts,
)

_GIT_IDENTITY = {
    "GIT_AUTHOR_NAME": "merge-test",
    "GIT_AUTHOR_EMAIL": "merge-test@local",
    "GIT_COMMITTER_NAME": "merge-test",
    "GIT_COMMITTER_EMAIL": "merge-test@local",
}

_MODULE = "mod.py"

_BASE = """import os


def compute(x):
    return x * 2


def reduce(values):
    total = 0
    for value in values:
        total += value
    return total


def report(values):
    return f"{reduce(values)} / {compute(len(values))}"
"""

#: Two lanes each inserting their own sweep helpers directly after the imports:
#: the shape that cost ``flydsl_moe_stage2`` its measured 1.1727x.
_LANE_ONE_HELPERS = """

def _sweep_flag(name, default):
    return os.environ.get("FORGE_SWEEP_" + name, default) == "1"


_PAD_ZERO = _sweep_flag("PAD_ZERO", "1")
"""

_LANE_TWO_HELPERS = """

def _sweep_int(name, default):
    return int(os.environ.get("FORGE_SWEEP_" + name, default))


_TILE_N = _sweep_int("TILE_N", "0")
"""

_LANE_THREE_HELPERS = """

def _sweep_list(name, default):
    return [int(part) for part in os.environ.get("FORGE_SWEEP_" + name, default).split(",")]


_XCD_ORDER = _sweep_list("XCD_ORDER", "0,1")
"""


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        env={**os.environ, **_GIT_IDENTITY},
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def _write(repo: Path, text: str) -> None:
    (repo / _MODULE).write_text(text, encoding="utf-8")


def _capture_patch(repo: Path, patches: Path, name: str, text: str) -> Path:
    """Diff *text* against the committed base and put the worktree back."""
    _write(repo, text)
    patch = patches / f"{name}.patch"
    patch.write_text(_git(repo, "diff"), encoding="utf-8")
    _git(repo, "checkout", "--", _MODULE)
    return patch


def _insert_helpers(helpers: str) -> str:
    return _BASE.replace("import os\n", "import os\n" + helpers, 1)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--initial-branch=main")
    _git(repo, "config", "core.autocrlf", "false")
    _write(repo, _BASE)
    _git(repo, "add", _MODULE)
    _git(repo, "commit", "-m", "base")
    return repo


@pytest.fixture
def patches(tmp_path: Path) -> Path:
    directory = tmp_path / "patches"
    directory.mkdir()
    return directory


def _land(repo: Path, patch: Path) -> None:
    """Commit one lane's patch, the way integration commits a KEEP."""
    _git(repo, "apply", str(patch))
    _git(repo, "add", _MODULE)
    _git(repo, "commit", "-m", f"keep {patch.stem}")


def _dirty(repo: Path) -> str:
    return _git(repo, "status", "--porcelain")


async def test_unconflicted_patch_still_applies_verbatim(repo: Path, patches: Path) -> None:
    patch = _capture_patch(repo, patches, "lane", _insert_helpers(_LANE_ONE_HELPERS))

    outcome = await apply_patch_resolving_conflicts(repo, patch)

    assert outcome.applied
    assert outcome.strategy == STRATEGY_STRICT
    assert not outcome.reconstructed
    assert "_PAD_ZERO" in (repo / _MODULE).read_text(encoding="utf-8")


async def test_neighbouring_edits_are_absorbed_by_three_way(repo: Path, patches: Path) -> None:
    """The landed lane rewrote a line inside the incoming hunk's context."""
    landed = _capture_patch(
        repo, patches, "landed", _BASE.replace("def reduce(values):", "def reduce(values, start=0):")
    )
    incoming = _capture_patch(repo, patches, "incoming", _BASE.replace("return x * 2", "return x * 4"))
    _land(repo, landed)

    outcome = await apply_patch_resolving_conflicts(repo, incoming, landed_patches=[landed])

    assert outcome.applied, outcome.error
    assert outcome.strategy == STRATEGY_THREE_WAY
    merged = (repo / _MODULE).read_text(encoding="utf-8")
    assert "return x * 4" in merged
    assert "def reduce(values, start=0):" in merged


async def test_two_lanes_inserting_at_one_anchor_are_unioned(repo: Path, patches: Path) -> None:
    landed = _capture_patch(repo, patches, "stage1", _insert_helpers(_LANE_ONE_HELPERS))
    incoming = _capture_patch(repo, patches, "stage2", _insert_helpers(_LANE_TWO_HELPERS))
    _land(repo, landed)

    outcome = await apply_patch_resolving_conflicts(
        repo,
        incoming,
        operator_id="stage2",
        landed_operator_ids=["stage1"],
        landed_patches=[landed],
    )

    assert outcome.applied, outcome.error
    assert outcome.strategy == STRATEGY_UNION
    assert outcome.reconstructed
    merged = (repo / _MODULE).read_text(encoding="utf-8")
    for symbol in ("_sweep_flag", "_PAD_ZERO", "_sweep_int", "_TILE_N"):
        assert symbol in merged, symbol
    compile(merged, _MODULE, "exec")


async def test_a_third_lane_merges_on_top_of_two_already_merged(repo: Path, patches: Path) -> None:
    """Lanes keep arriving, so each one merges against every KEEP before it."""
    lanes = [
        _capture_patch(repo, patches, f"stage{index}", _insert_helpers(helpers))
        for index, helpers in enumerate((_LANE_ONE_HELPERS, _LANE_TWO_HELPERS, _LANE_THREE_HELPERS), start=1)
    ]
    _land(repo, lanes[0])

    landed = [lanes[0]]
    for lane in lanes[1:]:
        outcome = await apply_patch_resolving_conflicts(
            repo,
            lane,
            operator_id=lane.stem,
            landed_operator_ids=[patch.stem for patch in landed],
            landed_patches=landed,
        )
        assert outcome.applied, (lane.stem, outcome.error)
        assert outcome.strategy == STRATEGY_UNION
        _git(repo, "add", _MODULE)
        _git(repo, "commit", "-m", f"keep {lane.stem}")
        landed.append(lane)

    merged = (repo / _MODULE).read_text(encoding="utf-8")
    for symbol in ("_PAD_ZERO", "_TILE_N", "_XCD_ORDER"):
        assert symbol in merged, symbol
    compile(merged, _MODULE, "exec")


async def test_overlapping_edits_are_left_to_a_resolver(repo: Path, patches: Path) -> None:
    landed = _capture_patch(repo, patches, "landed", _BASE.replace("return x * 2", "return x * 4"))
    incoming = _capture_patch(repo, patches, "incoming", _BASE.replace("return x * 2", "return x * 8"))
    _land(repo, landed)

    outcome = await apply_patch_resolving_conflicts(repo, incoming, resolver=None)

    assert not outcome.applied
    assert outcome.conflicted
    assert "no resolver available" in outcome.note()
    assert not _dirty(repo)
    assert "return x * 4" in (repo / _MODULE).read_text(encoding="utf-8")


def _resolver_returning(text: str):
    async def resolve(**_: object) -> str:
        return text

    return resolve


async def test_resolver_output_lands_when_it_keeps_both_sides(repo: Path, patches: Path) -> None:
    landed = _capture_patch(repo, patches, "landed", _BASE.replace("return x * 2", "return x * 4"))
    incoming = _capture_patch(repo, patches, "incoming", _BASE.replace("return x * 2", "return x * 8"))
    _land(repo, landed)

    outcome = await apply_patch_resolving_conflicts(
        repo,
        incoming,
        landed_patches=[landed],
        resolver=_resolver_returning(
            _BASE.replace(
                "    return x * 2",
                '    if os.environ.get("LANE_TWO"):\n        return x * 8\n    return x * 4',
            )
        ),
    )

    assert outcome.applied, outcome.error
    assert outcome.strategy == STRATEGY_LLM
    merged = (repo / _MODULE).read_text(encoding="utf-8")
    assert "return x * 4" in merged
    assert "return x * 8" in merged


async def test_resolver_dropping_the_incoming_side_is_rejected(repo: Path, patches: Path) -> None:
    landed = _capture_patch(repo, patches, "landed", _BASE.replace("return x * 2", "return x * 4"))
    incoming = _capture_patch(repo, patches, "incoming", _BASE.replace("return x * 2", "return x * 8"))
    _land(repo, landed)

    outcome = await apply_patch_resolving_conflicts(
        repo,
        incoming,
        landed_patches=[landed],
        resolver=_resolver_returning(_BASE.replace("return x * 2", "return x * 4")),
    )

    assert not outcome.applied
    assert "dropped 1 added line(s)" in outcome.note()
    assert not _dirty(repo)


async def test_resolver_dropping_a_landed_keep_is_rejected(repo: Path, patches: Path) -> None:
    landed = _capture_patch(repo, patches, "landed", _BASE.replace("return x * 2", "return x * 4"))
    incoming = _capture_patch(repo, patches, "incoming", _BASE.replace("return x * 2", "return x * 8"))
    _land(repo, landed)

    outcome = await apply_patch_resolving_conflicts(
        repo,
        incoming,
        landed_patches=[landed],
        resolver=_resolver_returning(_BASE.replace("return x * 2", "return x * 8")),
    )

    assert not outcome.applied
    assert "dropped 1 added line(s)" in outcome.note()
    assert not _dirty(repo)


async def test_resolver_leaving_markers_is_rejected(repo: Path, patches: Path) -> None:
    landed = _capture_patch(repo, patches, "landed", _BASE.replace("return x * 2", "return x * 4"))
    incoming = _capture_patch(repo, patches, "incoming", _BASE.replace("return x * 2", "return x * 8"))
    _land(repo, landed)

    outcome = await apply_patch_resolving_conflicts(
        repo,
        incoming,
        resolver=_resolver_returning("<<<<<<< ours\nA = 1\n=======\nA = 2\n>>>>>>> theirs\n"),
    )

    assert not outcome.applied
    assert "conflict markers survived" in outcome.note()
    assert not _dirty(repo)


async def test_resolver_shadowing_a_module_symbol_is_rejected(repo: Path, patches: Path) -> None:
    landed = _capture_patch(repo, patches, "landed", _BASE.replace("return x * 2", "return x * 4"))
    incoming = _capture_patch(repo, patches, "incoming", _BASE.replace("return x * 2", "return x * 8"))
    _land(repo, landed)

    outcome = await apply_patch_resolving_conflicts(
        repo,
        incoming,
        resolver=_resolver_returning(_BASE + "\n\ndef compute(x):\n    return x * 8\n"),
    )

    assert not outcome.applied
    assert "redefines module-level compute" in outcome.note()
    assert not _dirty(repo)


async def test_unparsable_resolution_is_rejected(repo: Path, patches: Path) -> None:
    landed = _capture_patch(repo, patches, "landed", _BASE.replace("return x * 2", "return x * 4"))
    incoming = _capture_patch(repo, patches, "incoming", _BASE.replace("return x * 2", "return x * 8"))
    _land(repo, landed)

    outcome = await apply_patch_resolving_conflicts(
        repo,
        incoming,
        resolver=_resolver_returning("def compute(x)\n    return x\n"),
    )

    assert not outcome.applied
    assert "does not parse" in outcome.note()
    assert not _dirty(repo)
