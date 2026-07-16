# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for framework-agent pure helpers in ``kb`` and ``models``.

Covers the KB ledger reader, the SDK-message text extractor, the LLM prompt
builder, and the request/field parsers + validation branches in ``models`` --
all pure over dicts / tmp files with no KB backend or network.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.agents.framework import kb
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
    assert _parse_search_modes(None) == ("primus_cortex", "github")
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
            "primus_cortex": {"base_url": "http://primus"},
            "pr_filter": {"require_labels": ["perf"]},
            "search_modes": ["github"],
            "pr_states": ["open"],
            "keywords": "decode",
        }
    )
    assert req.framework == "vllm"
    assert req.repo_url == "https://github.com/acme/x"
    assert "build" in req.commands
    assert req.primus_cortex is not None


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
        ExploreRequest.from_dict({**base, "primus_cortex": "x"})


def test_candidate_slug_and_pr_number() -> None:
    assert Candidate(ref="feature/Foo@1", repo="r").slug == "feature-foo-1"
    assert Candidate(ref="!!!", repo="r").slug == "candidate"
    assert Candidate(ref="PR:42", repo="r").pr_number == 42
    assert Candidate(ref="branch-x", repo="r").pr_number is None
    assert Candidate(ref="PR:notanum", repo="r").pr_number is None
