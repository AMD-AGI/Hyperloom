"""SpecialistPromptInputs.pr_candidates + FRAMEWORK PR CANDIDATES
section tests + ``_warm_specialist_params`` framework_pr_scout
pre-fetch wiring."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.orchestrator.specialist_domains import get_domain
from inference_optimizer.orchestrator.system_prompts.specialist_prompt_builder import (
    SpecialistPromptInputs,
    _section_framework_pr_candidates,
    build_specialist_prompts,
)


# ---------------------------------------------------------------------------
# 1. Section rendering
# ---------------------------------------------------------------------------
def _serving_inp(
    *, pr_candidates: list[dict[str, Any]], sub_kind: str = "framework_pr_scout",
) -> SpecialistPromptInputs:
    return SpecialistPromptInputs(
        task_id="t-1",
        domain=get_domain("serving_specialist"),
        gap_canonical_id="gap.x",
        sub_kind=sub_kind,
        pr_candidates=pr_candidates,
    )


def test_section_renders_candidates_with_curl_template():
    inp = _serving_inp(
        pr_candidates=[
            {
                "repo": "sgl-project/sglang",
                "pr_number": 1234,
                "ref": "PR:1234",
                "title": "MoE expert dispatch fix",
                "summary": "moe, perf",
                "score": 0.92,
                "diff_url": (
                    "https://github.com/sgl-project/sglang/pull/1234.diff"
                ),
            },
        ],
    )
    section = _section_framework_pr_candidates(inp)
    text = "\n".join(section)
    assert "## 6b. FRAMEWORK PR CANDIDATES" in text
    assert "1234" in text and "MoE expert dispatch fix" in text
    assert (
        "https://github.com/sgl-project/sglang/pull/1234.diff" in text
    )
    # curl template + iron rule.
    assert "curl -fsSL" in text
    assert "do NOT commit a raw GitHub diff" in text


def test_section_omitted_when_subkind_mismatch():
    inp = _serving_inp(
        pr_candidates=[
            {"repo": "x/y", "pr_number": 1, "title": "z", "diff_url": ""},
        ],
        sub_kind="",
    )
    assert _section_framework_pr_candidates(inp) == []


def test_section_omitted_when_no_candidates():
    inp = _serving_inp(pr_candidates=[])
    assert _section_framework_pr_candidates(inp) == []


def test_section_caps_at_20_rows():
    cands = [
        {
            "repo": "sgl-project/sglang",
            "pr_number": i,
            "ref": f"PR:{i}",
            "title": f"PR {i}",
            "summary": "",
            "score": 0.5,
            "diff_url": f"https://github.com/sgl-project/sglang/pull/{i}.diff",
        }
        for i in range(1, 31)
    ]
    inp = _serving_inp(pr_candidates=cands)
    section = _section_framework_pr_candidates(inp)
    text = "\n".join(section)
    # The 21st PR (number 21) should NOT appear.
    assert " PR 1 " in text or " | PR 1 |" in text
    assert "PR 21" not in text


def test_build_specialist_prompts_inserts_pr_candidates_after_pr_feed():
    inp = _serving_inp(
        pr_candidates=[
            {
                "repo": "sgl-project/sglang",
                "pr_number": 999,
                "ref": "PR:999",
                "title": "test",
                "summary": "",
                "score": 0.5,
                "diff_url": "",
            },
        ],
    )
    _system, user = build_specialist_prompts(inp)
    pr_feed_idx = user.index("## 6. PR FEED")
    fa_idx = user.index("## 6b. FRAMEWORK PR CANDIDATES")
    source_idx = user.index("## 7. LOCAL SOURCE NAVIGATION HINT")
    assert pr_feed_idx < fa_idx < source_idx


# ---------------------------------------------------------------------------
# 2. Coordinator pre-fetch wiring
# ---------------------------------------------------------------------------
@dataclass
class _State:
    framework: str = "sglang"
    framework_agent_enabled: bool = True
    last_trace_analyze: dict[str, Any] = field(default_factory=dict)
    gpu_type: str = ""
    tp: int = 0
    precision: str = ""
    conc: int = 0
    isl: int = 0
    osl: int = 0
    max_model_len: int = 0
    warm_start_recipe: dict[str, Any] = field(default_factory=dict)
    warm_start_pitfalls: list[dict[str, Any]] = field(default_factory=list)
    warm_start_lessons: list[dict[str, Any]] = field(default_factory=list)
    gaps: list[dict[str, Any]] = field(default_factory=list)

    def find_gap(self, _cid: str):
        return None


def _make_coord(tmp_path: Path, *, state: _State) -> Coordinator:
    c = Coordinator.__new__(Coordinator)
    c.session_dir = tmp_path
    c.shared_state = state
    c.knowledge_plane = None
    return c


@pytest.mark.asyncio
async def test_warm_calls_fetch_pr_candidates_for_framework_pr_scout(
    tmp_path, monkeypatch,
):
    """When ``sub_kind=framework_pr_scout`` and
    ``framework_agent_enabled=True``, the warmer calls
    ``fetch_pr_candidates`` and stamps the result onto
    ``params['pr_candidates']``."""
    captured: dict[str, Any] = {}

    async def fake_fetch(*, gap_description, framework, session_dir, **_kw):
        captured["gap_description"] = gap_description
        captured["framework"] = framework
        captured["session_dir"] = session_dir
        return [
            {
                "repo": "sgl-project/sglang",
                "pr_number": 1,
                "ref": "PR:1",
                "title": "fake",
                "summary": "",
                "score": 0.1,
                "diff_url": "",
            },
        ]

    monkeypatch.setattr(
        "inference_optimizer.orchestrator.framework_agent_client.fetch_pr_candidates",
        fake_fetch,
    )

    coord = _make_coord(tmp_path, state=_State())
    params: dict[str, Any] = {
        "domain": "serving_specialist",
        "sub_kind": "framework_pr_scout",
        "gap_symptom": "MoE routing slow",
        "gap_layer": "kernel",
        "gap_canonical_id": "gap.moe.routing",
    }
    await coord._warm_specialist_params(params)

    assert "pr_candidates" in params
    assert len(params["pr_candidates"]) == 1
    # gap_description = symptom + layer + canonical_id joined.
    assert "MoE routing slow" in captured["gap_description"]
    assert "kernel" in captured["gap_description"]
    assert captured["framework"] == "sglang"


@pytest.mark.asyncio
async def test_warm_skips_pr_candidates_when_subkind_missing(
    tmp_path, monkeypatch,
):
    """No ``sub_kind`` → no PR candidates pre-fetch (default
    serving_specialist path)."""
    called = False

    async def fake_fetch(**_kw):
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(
        "inference_optimizer.orchestrator.framework_agent_client.fetch_pr_candidates",
        fake_fetch,
    )

    coord = _make_coord(tmp_path, state=_State())
    params: dict[str, Any] = {
        "domain": "serving_specialist",
        "gap_symptom": "x",
    }
    await coord._warm_specialist_params(params)
    assert not called
    assert "pr_candidates" not in params


@pytest.mark.asyncio
async def test_warm_skips_pr_candidates_when_toggle_off(tmp_path, monkeypatch):
    """``framework_agent_enabled=False`` → no pre-fetch even with the
    correct sub_kind."""
    called = False

    async def fake_fetch(**_kw):
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(
        "inference_optimizer.orchestrator.framework_agent_client.fetch_pr_candidates",
        fake_fetch,
    )

    state = _State(framework_agent_enabled=False)
    coord = _make_coord(tmp_path, state=state)
    params: dict[str, Any] = {
        "domain": "serving_specialist",
        "sub_kind": "framework_pr_scout",
        "gap_symptom": "x",
    }
    await coord._warm_specialist_params(params)
    assert not called
    assert "pr_candidates" not in params


@pytest.mark.asyncio
async def test_warm_graceful_when_pre_fetch_raises(tmp_path, monkeypatch):
    """Exceptions from ``fetch_pr_candidates`` must not break the
    dispatch — the warmer stamps an empty list and logs."""
    async def boom(**_kw):
        raise RuntimeError("simulated fa outage")

    monkeypatch.setattr(
        "inference_optimizer.orchestrator.framework_agent_client.fetch_pr_candidates",
        boom,
    )
    coord = _make_coord(tmp_path, state=_State())
    params: dict[str, Any] = {
        "domain": "serving_specialist",
        "sub_kind": "framework_pr_scout",
        "gap_symptom": "x",
    }
    await coord._warm_specialist_params(params)
    assert params.get("pr_candidates") == []
