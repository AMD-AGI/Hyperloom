# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.common import llm_config
from hyperloom.common.llm_config import ANTHROPIC_CREDENTIAL_ENV_ORDER, _ANTHROPIC_MANAGED_GATEWAY_ENVS
from hyperloom.orchestrator.kernel.patch_conflict_merge import (
    STRATEGY_LLM,
    STRATEGY_STRICT,
    STRATEGY_THREE_WAY,
    STRATEGY_UNION,
    _resolver_backend,
    apply_patch_resolving_conflicts,
    llm_resolution_available,
)
from hyperloom.common.llm_config import DEFAULT_CODEX_MODEL

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

#: A second lane inserting the *same* helper as lane one, plus its own constant:
#: the union keeps both copies, which leaves ``_sweep_flag`` defined twice.
_LANE_TWO_SHARED_HELPER = """

def _sweep_flag(name, default):
    return os.environ.get("FORGE_SWEEP_" + name, default) == "1"


_TILE_N = _sweep_flag("TILE_N", "0")
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


def _capture_patch_creating(repo: Path, patches: Path, name: str, text: str, created: str) -> Path:
    """Diff a lane that also adds a file, and put the worktree back."""
    _write(repo, text)
    (repo / created).write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", "-N", created)
    patch = patches / f"{name}.patch"
    patch.write_text(_git(repo, "diff"), encoding="utf-8")
    _git(repo, "checkout", "--", _MODULE)
    _git(repo, "reset", "-q", "--", created)
    (repo / created).unlink()
    return patch


def _insert_helpers(helpers: str) -> str:
    return _BASE.replace("import os\n", "import os\n" + helpers, 1)


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the default resolver unavailable, so no test can reach a gateway.

    ``resolver=None`` means "build this deployment's resolver if either backend
    can be reached", so a developer with a credential exported for Claude or
    for Codex would otherwise turn the no-resolver tests into live calls.
    """
    for name in (*ANTHROPIC_CREDENTIAL_ENV_ORDER, *_ANTHROPIC_MANAGED_GATEWAY_ENVS, "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)


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


async def test_an_openai_only_box_resolves_through_codex(
    repo: Path, patches: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Forge's rewrite lane runs Codex on such a box, so this rung must too."""
    monkeypatch.setenv("OPENAI_API_KEY", "token")
    client = object()
    posted: dict[str, object] = {}

    async def _achat(called_on: object, **params: object) -> object:
        posted.update(params, client=called_on)
        return SimpleNamespace(
            text=(
                "\n\ndef _sweep_flag(name, default):\n"
                '    return os.environ.get("FORGE_SWEEP_" + name, default) == "1"\n'
                '\n\n_PAD_ZERO = _sweep_flag("PAD_ZERO", "1")\n'
                '_TILE_N = _sweep_flag("TILE_N", "0")\n'
            )
        )

    monkeypatch.setattr(llm_config, "get_async_openai_client", lambda **_: client)
    monkeypatch.setattr(llm_config, "achat_completion", _achat)
    landed = _capture_patch(repo, patches, "stage1", _insert_helpers(_LANE_ONE_HELPERS))
    incoming = _capture_patch(repo, patches, "stage2", _insert_helpers(_LANE_TWO_SHARED_HELPER))
    _land(repo, landed)

    outcome = await apply_patch_resolving_conflicts(repo, incoming, landed_patches=[landed], resolver=None)

    assert outcome.applied, outcome.error
    assert outcome.strategy == STRATEGY_LLM
    assert posted["client"] is client
    assert posted["model"] == DEFAULT_CODEX_MODEL
    assert (posted["component"], posted["operation"]) == ("forge", "patch_conflict_merge")
    system, user = posted["messages"]  # type: ignore[misc]
    assert "never a choice between them" in system["content"]
    assert _MODULE in user["content"]
    assert (repo / _MODULE).read_text(encoding="utf-8").count("def _sweep_flag") == 1


def test_a_dual_configured_box_keeps_the_claude_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tie-break is `preferred_agent_backend`'s, not this module's."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "token")
    monkeypatch.setenv("OPENAI_API_KEY", "token")

    assert _resolver_backend() == "claude"


def test_a_box_credentialed_for_neither_side_has_no_resolver() -> None:
    assert not llm_resolution_available()


async def test_a_rejected_patch_leaves_nothing_of_the_file_it_created(repo: Path, patches: Path) -> None:
    """``git apply -3`` implies ``--index``, and the next lane commits the index."""
    landed = _capture_patch(repo, patches, "landed", _BASE.replace("return x * 2", "return x * 4"))
    incoming = _capture_patch_creating(
        repo, patches, "incoming", _BASE.replace("return x * 2", "return x * 8"), "sweep.py"
    )
    _land(repo, landed)

    outcome = await apply_patch_resolving_conflicts(repo, incoming, resolver=None)

    assert not outcome.applied
    assert not _dirty(repo)
    assert not (repo / "sweep.py").exists()


async def test_a_union_that_duplicates_a_helper_is_rejected_without_a_resolver(repo: Path, patches: Path) -> None:
    landed = _capture_patch(repo, patches, "stage1", _insert_helpers(_LANE_ONE_HELPERS))
    incoming = _capture_patch(repo, patches, "stage2", _insert_helpers(_LANE_TWO_SHARED_HELPER))
    _land(repo, landed)

    outcome = await apply_patch_resolving_conflicts(repo, incoming, landed_patches=[landed], resolver=None)

    assert not outcome.applied
    assert "redefines module-level _sweep_flag" in outcome.note()
    assert not _dirty(repo)


def _resolver_returning(text: str):
    """A resolver that replaces every conflicted region with *text*."""

    async def resolve(**_: object) -> str:
        return text

    return resolve


async def test_resolver_is_shown_one_region_and_its_context(repo: Path, patches: Path) -> None:
    """The model decides the conflicted lines; it never sees the file to rewrite."""
    landed = _capture_patch(repo, patches, "landed", _BASE.replace("return x * 2", "return x * 4"))
    incoming = _capture_patch(repo, patches, "incoming", _BASE.replace("return x * 2", "return x * 8"))
    _land(repo, landed)
    seen: list[dict[str, str]] = []

    async def resolve(**kwargs: str) -> str:
        seen.append(kwargs)
        return "    return x * 4 * 8\n    return x * 8\n"

    await apply_patch_resolving_conflicts(repo, incoming, landed_patches=[landed], resolver=resolve)

    assert len(seen) == 1
    region = seen[0]["region"]
    assert region.splitlines()[0].startswith("<<<<<<<")
    assert "return x * 4" in region and "return x * 8" in region
    # The untouched body of the file is context, never part of the answer.
    assert "def report(values):" not in region
    assert "import os" in seen[0]["context_before"]
    assert "def report(values):" in seen[0]["context_after"]


async def test_resolver_output_lands_when_it_keeps_both_sides(repo: Path, patches: Path) -> None:
    landed = _capture_patch(repo, patches, "landed", _BASE.replace("return x * 2", "return x * 4"))
    incoming = _capture_patch(repo, patches, "incoming", _BASE.replace("return x * 2", "return x * 8"))
    _land(repo, landed)

    outcome = await apply_patch_resolving_conflicts(
        repo,
        incoming,
        landed_patches=[landed],
        resolver=_resolver_returning('    if os.environ.get("LANE_TWO"):\n        return x * 8\n    return x * 4\n'),
    )

    assert outcome.applied, outcome.error
    assert outcome.strategy == STRATEGY_LLM
    merged = (repo / _MODULE).read_text(encoding="utf-8")
    assert "return x * 4" in merged
    assert "return x * 8" in merged
    # Everything outside the region is still the file the lanes started from.
    assert merged.replace('    if os.environ.get("LANE_TWO"):\n        return x * 8\n', "") == _BASE.replace(
        "return x * 2", "return x * 4"
    )
    compile(merged, _MODULE, "exec")


async def test_a_dedented_first_line_is_put_back_where_the_markers_were(repo: Path, patches: Path) -> None:
    """Models indent the body of a region but start line one at the marker's column."""
    landed = _capture_patch(repo, patches, "landed", _BASE.replace("return x * 2", "return x * 4"))
    incoming = _capture_patch(repo, patches, "incoming", _BASE.replace("return x * 2", "return x * 8"))
    _land(repo, landed)

    outcome = await apply_patch_resolving_conflicts(
        repo,
        incoming,
        landed_patches=[landed],
        resolver=_resolver_returning('if os.environ.get("LANE_TWO"):\n        return x * 8\n    return x * 4\n'),
    )

    assert outcome.applied, outcome.error
    merged = (repo / _MODULE).read_text(encoding="utf-8")
    assert '    if os.environ.get("LANE_TWO"):\n' in merged
    compile(merged, _MODULE, "exec")


async def test_a_union_that_duplicates_a_helper_escalates_to_the_resolver(repo: Path, patches: Path) -> None:
    """Both lanes shipped the same helper, so keeping both copies is not a merge."""
    landed = _capture_patch(repo, patches, "stage1", _insert_helpers(_LANE_ONE_HELPERS))
    incoming = _capture_patch(repo, patches, "stage2", _insert_helpers(_LANE_TWO_SHARED_HELPER))
    _land(repo, landed)
    asked = 0

    async def resolve(**_: object) -> str:
        nonlocal asked
        asked += 1
        return (
            "\n\ndef _sweep_flag(name, default):\n"
            '    return os.environ.get("FORGE_SWEEP_" + name, default) == "1"\n'
            '\n\n_PAD_ZERO = _sweep_flag("PAD_ZERO", "1")\n'
            '_TILE_N = _sweep_flag("TILE_N", "0")\n'
        )

    outcome = await apply_patch_resolving_conflicts(repo, incoming, landed_patches=[landed], resolver=resolve)

    assert asked == 1
    assert outcome.applied, outcome.error
    assert outcome.strategy == STRATEGY_LLM
    assert "redefines module-level _sweep_flag" in outcome.note()
    merged = (repo / _MODULE).read_text(encoding="utf-8")
    assert merged.count("def _sweep_flag") == 1
    for symbol in ("_PAD_ZERO", "_TILE_N"):
        assert symbol in merged, symbol
    compile(merged, _MODULE, "exec")


async def test_a_resolver_that_raises_costs_the_lane_and_nothing_else(repo: Path, patches: Path) -> None:
    """An injected callable reaching a network and an SDK fails in many ways."""
    landed = _capture_patch(repo, patches, "landed", _BASE.replace("return x * 2", "return x * 4"))
    incoming = _capture_patch(repo, patches, "incoming", _BASE.replace("return x * 2", "return x * 8"))
    _land(repo, landed)

    async def explode(**_: object) -> str:
        raise RuntimeError("claude-agent-sdk is not installed")

    outcome = await apply_patch_resolving_conflicts(repo, incoming, landed_patches=[landed], resolver=explode)

    assert not outcome.applied
    assert "resolver call failed" in outcome.note()
    assert not _dirty(repo)
    assert "<<<<<<<" not in (repo / _MODULE).read_text(encoding="utf-8")


async def test_resolver_dropping_the_incoming_side_is_rejected(repo: Path, patches: Path) -> None:
    landed = _capture_patch(repo, patches, "landed", _BASE.replace("return x * 2", "return x * 4"))
    incoming = _capture_patch(repo, patches, "incoming", _BASE.replace("return x * 2", "return x * 8"))
    _land(repo, landed)

    outcome = await apply_patch_resolving_conflicts(
        repo,
        incoming,
        landed_patches=[landed],
        resolver=_resolver_returning("    return x * 4\n"),
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
        resolver=_resolver_returning("    return x * 8\n"),
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
        resolver=_resolver_returning("    return x * 4\n\n\ndef compute(x):\n    return x * 8\n"),
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
        resolver=_resolver_returning("    return x * 8\n    if (\n"),
    )

    assert not outcome.applied
    assert "does not parse" in outcome.note()
    assert not _dirty(repo)
