# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for framework-agent pure helpers in ``kb`` and ``models``.

Covers the KB ledger reader, the SDK-message text extractor, the LLM prompt
builder, and the request/field parsers + validation branches in ``models`` --
all pure over dicts / tmp files with no KB backend or network.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.agents.framework.kb import (
    _build_llm_prompt,
    _iter_message_text,
    read_pr_ledger,
)
from hyperloom.agents.framework.models import (
    Candidate,
    ExploreRequest,
    Finding,
    _parse_keywords,
    _parse_pr_states,
    _parse_search_modes,
)


# --------------------------------------------------------------------------
# kb.py
# --------------------------------------------------------------------------
def test_read_pr_ledger_tolerates_malformed_rows(tmp_path: Path) -> None:
    part = tmp_path / "framework_optimization"
    part.mkdir()
    (part / "lessons.jsonl").write_text(
        '{"a": 1}\n'  # valid
        "\n"  # blank -> skipped
        "   \n"  # whitespace -> skipped
        "not json\n"  # malformed -> skipped
        "[1, 2]\n"  # valid json but not a dict -> skipped
        '{"b": 2}\n',
        encoding="utf-8",
    )
    assert read_pr_ledger(kb_root=tmp_path) == [{"a": 1}, {"b": 2}]
    # Missing file -> empty (cold start).
    assert read_pr_ledger(kb_root=tmp_path / "nope") == []


@pytest.mark.parametrize(
    "env",
    [
        pytest.param({}, id="workspace-default"),
        pytest.param({"USER_DATA_PATH": "/tmp/hl-user-data"}, id="user-data-path"),
        pytest.param({"INFERENCE_OPTIMIZER_FA_KB_PATH": "/tmp/hl-explicit-kb"}, id="explicit-override"),
    ],
)
def test_lessons_writer_and_reader_resolve_the_same_file(
    monkeypatch: pytest.MonkeyPatch,
    env: dict[str, str],
) -> None:
    """The PR ledger must be one file, whatever the deployment sets.

    The writer and the reader used to resolve the root independently, so an
    orchestrator run appended lessons under the workspace while ``fa`` read a
    packaged path that does not exist. Nothing raised: the ledger just came
    back empty, and every session re-proposed PRs it had already tried.
    """
    from hyperloom.agents.framework import kb as fa_kb
    from hyperloom.orchestrator.knowledge import kb_writeback

    for name in ("USER_DATA_PATH", "INFERENCE_OPTIMIZER_FA_KB_PATH", "FRAMEWORK_AGENT_KB_DIR", "FRAMEWORK_AGENT_ROOT"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    writer = kb_writeback._default_kb_root() / kb_writeback.LESSONS_FILE
    reader = fa_kb.path_for_framework("") / kb_writeback.LESSONS_FILE

    assert writer == reader
    # The packaged seed is a different, read-only tree and must not be the
    # place a live session writes to.
    assert fa_kb.packaged_kb_root() not in writer.parents


def test_framework_kb_does_not_share_a_root_with_the_recipe_kb(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The framework KB must not land on a directory a recipe store owns.

    ``list_domains`` reports every directory under the framework KB root as a
    framework domain, so sharing a root with a recipe store surfaces recipe
    trees as domains and puts two unrelated writers in one namespace.

    The root that actually matters here is the *legacy* recipe root
    ``<workspace>/kb``: it is where the framework ledger used to be written, it
    still holds recipe data on deployments that predate the split, and the
    one-time recipe migration still reads it. The current recipe root
    (``<workspace>/knowledge``) is checked too, but it never collided — asserting
    against it alone is what let an earlier version of this test pass with the
    framework root set back to ``kb``.
    """
    from hyperloom.agents.framework import kb as fa_kb
    from hyperloom.inference_optimizer.cli.kb import _legacy_recipe_root, _resolve_local_kb_root

    for name in (
        "INFERENCE_OPTIMIZER_FA_KB_PATH",
        "FRAMEWORK_AGENT_KB_DIR",
        "HYPERLOOM_LOCAL_KB_ROOT",
        "KNOWLEDGE_LOCAL_ROOT",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path / "workspace"))

    framework_root = fa_kb.mutable_kb_root()
    recipe_roots = [
        _legacy_recipe_root(os.environ),
        _resolve_local_kb_root(SimpleNamespace(local_kb_root=None)),
    ]

    for recipe_root in recipe_roots:
        assert framework_root != recipe_root
        assert recipe_root not in framework_root.parents
        assert framework_root not in recipe_root.parents


def test_legacy_kb_dirname_agrees_with_the_recipe_side(monkeypatch, tmp_path: Path) -> None:
    """The framework package hardcodes the legacy root's leaf; it must stay in sync.

    ``agents.framework`` cannot import ``inference_optimizer`` (the ``fa`` CLI
    runs standalone), so the one-time partition migration names ``kb`` itself. If
    the recipe side ever renames it, the migration would silently stop finding
    anything to carry over.
    """
    from hyperloom.agents.framework import kb as fa_kb
    from hyperloom.inference_optimizer.cli.kb import _legacy_recipe_root

    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path / "workspace"))

    assert _legacy_recipe_root(os.environ).name == fa_kb._LEGACY_WORKSPACE_KB_DIRNAME


def test_iter_message_text_handles_all_shapes() -> None:
    assert list(_iter_message_text("hello")) == ["hello"]
    assert list(_iter_message_text(SimpleNamespace(text="t"))) == ["t"]
    msg = SimpleNamespace(
        content=[
            SimpleNamespace(text="b1"),
            SimpleNamespace(text="b2"),
            SimpleNamespace(other=1),  # no .text -> skipped
        ]
    )
    assert list(_iter_message_text(msg)) == ["b1", "b2"]


def test_build_llm_prompt_embeds_domain_and_findings() -> None:
    prompt = _build_llm_prompt("kernel_agent", [Finding(title="Speedup", body="2x")])
    assert "kernel_agent" in prompt
    assert "curator" in prompt
    assert "Speedup" in prompt


# --------------------------------------------------------------------------
# models.py
# --------------------------------------------------------------------------
def test_parse_pr_states() -> None:
    assert _parse_pr_states(None) == ("open",)
    assert _parse_pr_states("open") == ("open",)
    assert _parse_pr_states(["open"]) == ("open",)
    with pytest.raises(ValueError):
        _parse_pr_states(123)
    with pytest.raises(ValueError):
        _parse_pr_states(["bogus-state"])


def test_parse_keywords() -> None:
    assert _parse_keywords(None) == ()
    assert _parse_keywords("decode, throughput  moe") == ("decode", "throughput", "moe")
    assert _parse_keywords([" x ", "y", ""]) == ("x", "y")
    with pytest.raises(ValueError):
        _parse_keywords(123)


def test_parse_search_modes() -> None:
    assert _parse_search_modes(None) == ("pr_monitor", "github")
    assert _parse_search_modes("github") == ("github",)
    with pytest.raises(ValueError):
        _parse_search_modes(123)
    with pytest.raises(ValueError):
        _parse_search_modes(["not-a-source"])


def test_explore_request_from_dict_valid() -> None:
    req = ExploreRequest.from_dict(
        {
            "framework": "VLLM",
            "repo_url": "https://github.com/acme/x",
            "work_dir": "/tmp/fa",
            "baseline": {"throughput": 100.0},
            "commands": {"build": {"command": "make"}},
            "outputs": {"summary": "out.json"},
            "pr_monitor": {"base_url": "http://pr_monitor"},
            "pr_filter": {"require_labels": ["perf"]},
            "search_modes": ["github"],
            "pr_states": ["open"],
            "keywords": "decode",
        }
    )
    assert req.framework == "vllm"
    assert req.repo_url == "https://github.com/acme/x"
    assert "build" in req.commands
    assert req.pr_monitor is not None


def test_explore_request_from_dict_validation_errors() -> None:
    base = {
        "framework": "vllm",
        "repo_url": "https://github.com/acme/x",
        "baseline": {"throughput": 100.0},
    }
    with pytest.raises(ValueError):
        ExploreRequest.from_dict({"repo_url": "r", "baseline": {"throughput": 1.0}})  # no framework
    with pytest.raises(ValueError):
        ExploreRequest.from_dict({"framework": "vllm", "baseline": {"throughput": 1.0}})  # no repo
    with pytest.raises(ValueError):
        ExploreRequest.from_dict({**base, "baseline": "x"})
    with pytest.raises(ValueError):
        ExploreRequest.from_dict({**base, "commands": "x"})
    with pytest.raises(ValueError):
        ExploreRequest.from_dict({**base, "outputs": "x"})
    with pytest.raises(ValueError):
        ExploreRequest.from_dict({**base, "pr_monitor": "x"})


def test_candidate_slug_and_pr_number() -> None:
    assert Candidate(ref="feature/Foo@1", repo="r").slug == "feature-foo-1"
    assert Candidate(ref="!!!", repo="r").slug == "candidate"
    assert Candidate(ref="PR:42", repo="r").pr_number == 42
    assert Candidate(ref="branch-x", repo="r").pr_number is None
    assert Candidate(ref="PR:notanum", repo="r").pr_number is None
