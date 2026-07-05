# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the FRAMEWORK_AGENT phase-discover subcommand. Hermetic - stubs ``sources.enumerate_candidates`` so no network/git is required."""

from __future__ import annotations

import json
from pathlib import Path


import framework_agent.runtime.cli as cli
from framework_agent.models import Candidate


# phase-discover ---------------------------------------------------------


def test_phase_discover_happy_path(monkeypatch, tmp_path: Path, capsys) -> None:
    req = {
        "model": "deepseek-r1",
        "framework": "sglang",
        "gpu_type": "MI300X",
        "repo_url": "https://github.com/sgl-project/sglang.git",
        "work_dir": str(tmp_path / "w"),
        "gaps": [
            {"gap_canonical_id": "gap-decode-1", "gap_description": "low decode tput"},
            {"gap_canonical_id": "gap-prefill-1", "gap_description": "prefill stall"},
        ],
        "max_search_candidates": 3,
        "batch_id": "batch-test",
    }
    req_path = tmp_path / "req.json"
    req_path.write_text(json.dumps(req), encoding="utf-8")

    import framework_agent.sources as src

    def fake_enum(r):
        # Second gap returns a duplicate ref to validate dedup.
        if "decode" in (r.gap_description or ""):
            return [
                Candidate(
                    ref="PR:1234",
                    repo="sgl-project/sglang",
                    source="github",
                    title="speed up decode",
                    html_url="https://github.com/sgl-project/sglang/pull/1234",
                    score=0.9,
                )
            ]
        return [
            Candidate(
                ref="PR:1234",
                repo="sgl-project/sglang",
                source="github",
                title="speed up decode",
                html_url="https://github.com/sgl-project/sglang/pull/1234",
                score=0.9,
            ),
            Candidate(
                ref="PR:5678",
                repo="sgl-project/sglang",
                source="github",
                title="prefill batching",
                html_url="https://github.com/sgl-project/sglang/pull/5678",
                score=0.7,
            ),
        ]

    monkeypatch.setattr(src, "enumerate_candidates", fake_enum)

    rc = cli.main(["phase-discover", "--request", str(req_path)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["batch_id"] == "batch-test"
    assert payload["framework"] == "sglang"
    assert payload["model"] == "deepseek-r1"
    assert payload["candidate_count"] == 2  # dedup across gaps
    refs = sorted(c["ref"] for c in payload["candidates"])
    assert refs == ["PR:1234", "PR:5678"]
    pr1 = next(c for c in payload["candidates"] if c["ref"] == "PR:1234")
    assert pr1["diff_url"].endswith("/pull/1234.diff")
    assert pr1["pr_number"] == 1234
    # Each candidate stamped with the gap it surfaced under.
    assert all("gap_canonical_id" in c for c in payload["candidates"])


def test_phase_discover_enables_search_perf_prs(monkeypatch, tmp_path: Path, capsys) -> None:
    """Regression: phase-discover MUST set search_perf_prs=True (else
    enumerate_candidates short-circuits to explicit-refs-only and always returns
    0 candidates) and degrade to GitHub-only when no primus URL is configured."""
    monkeypatch.delenv("PRIMUS_CORTEX_PR_API", raising=False)
    captured: dict[str, object] = {}

    import framework_agent.sources as src

    def fake_enum(r):
        captured["search_perf_prs"] = r.search_perf_prs
        captured["search_modes"] = list(r.search_modes)
        return [
            Candidate(ref="PR:42", repo="ROCm/vllm", source="github",
                      title="perf: moe fp8", html_url="https://github.com/ROCm/vllm/pull/42", score=0.9),
        ]

    monkeypatch.setattr(src, "enumerate_candidates", fake_enum)
    req = {
        "model": "/m/DeepSeek", "framework": "vllm", "gpu_type": "mi300x",
        "repo_url": "https://github.com/ROCm/vllm.git",
        "gaps": [{"gap_canonical_id": "g1", "gap_description": "moe fp8 decode"}],
        "max_search_candidates": 5, "batch_id": "b1",
    }
    req_path = tmp_path / "req.json"
    req_path.write_text(json.dumps(req), encoding="utf-8")

    rc = cli.main(["phase-discover", "--request", str(req_path)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert captured["search_perf_prs"] is True
    # No primus URL configured -> GitHub-only (no SourceConfigError raise).
    assert captured["search_modes"] == ["github"]
    assert payload["candidate_count"] == 1


def test_phase_discover_uses_primus_when_url_present(monkeypatch, tmp_path: Path, capsys) -> None:
    """With a primus_cortex_url in the request, both modes are enabled."""
    captured: dict[str, object] = {}
    import framework_agent.sources as src

    def fake_enum(r):
        captured["search_modes"] = list(r.search_modes)
        captured["primus_cfg"] = r.primus_cortex is not None
        return []

    monkeypatch.setattr(src, "enumerate_candidates", fake_enum)
    req = {
        "model": "/m/x", "framework": "vllm", "gpu_type": "mi300x",
        "repo_url": "https://github.com/ROCm/vllm.git",
        "gaps": [{"gap_canonical_id": "g1", "gap_description": "x"}],
        "primus_cortex_url": "http://primus.local/v1", "batch_id": "b1",
    }
    req_path = tmp_path / "req.json"
    req_path.write_text(json.dumps(req), encoding="utf-8")
    rc = cli.main(["phase-discover", "--request", str(req_path)])
    assert rc == 0
    assert captured["search_modes"] == ["primus_cortex", "github"]
    assert captured["primus_cfg"] is True


# Step B — hard-dedup + same-PR de-prioritisation ----------------------------
def test_extract_pr_number_from_ref_and_url() -> None:
    assert cli._extract_pr_number("PR:1234") == "1234"
    assert cli._extract_pr_number("https://github.com/sgl-project/sglang/pull/1234") == "1234"
    assert cli._extract_pr_number("PR:abc") == ""
    assert cli._extract_pr_number("") == ""


def test_candidate_excluded_by_memory_matches_id_and_pr_number() -> None:
    # pr_url / ref id match.
    assert cli._candidate_excluded_by_memory(
        pr_url="https://github.com/o/r/pull/7", ref="PR:7", pr_number=7,
        excluded_ids={"PR:7"}, excluded_pr_numbers=set(),
    )
    # PR-number match (from a failed candidate).
    assert cli._candidate_excluded_by_memory(
        pr_url="", ref="PR:9", pr_number=9,
        excluded_ids=set(), excluded_pr_numbers={"9"},
    )
    # Genuinely new candidate passes.
    assert not cli._candidate_excluded_by_memory(
        pr_url="https://github.com/o/r/pull/3", ref="PR:3", pr_number=3,
        excluded_ids={"PR:7"}, excluded_pr_numbers={"9"},
    )


def test_phase_discover_hard_filters_excluded_and_failed(monkeypatch, tmp_path: Path, capsys) -> None:
    """excluded_candidate_ids drops a candidate outright; a same-PR-number in
    failed_candidate_context drops another; only genuinely new PRs survive."""
    req = {
        "framework": "sglang",
        "repo_url": "https://github.com/sgl-project/sglang.git",
        "work_dir": str(tmp_path / "w"),
        "gaps": [{"gap_canonical_id": "g1", "gap_description": "decode"}],
        "max_search_candidates": 5,
        "batch_id": "batch-x",
        # PR:1234 already discovered/finalised → hard-excluded by id.
        "excluded_candidate_ids": ["PR:1234"],
        # PR:5678 already failed this session → dropped by same-PR-number.
        "failed_candidate_context": [{"ref": "PR:5678", "status": "reverted", "why": "no-op"}],
    }
    req_path = tmp_path / "req.json"
    req_path.write_text(json.dumps(req), encoding="utf-8")

    import framework_agent.sources as src

    def fake_enum(r):
        return [
            Candidate(ref="PR:1234", repo="sgl-project/sglang", source="github",
                      title="excluded by id", html_url="https://github.com/sgl-project/sglang/pull/1234", score=0.9),
            Candidate(ref="PR:5678", repo="sgl-project/sglang", source="github",
                      title="excluded by failed pr#", html_url="https://github.com/sgl-project/sglang/pull/5678", score=0.8),
            Candidate(ref="PR:9999", repo="sgl-project/sglang", source="github",
                      title="brand new", html_url="https://github.com/sgl-project/sglang/pull/9999", score=0.7),
        ]

    monkeypatch.setattr(src, "enumerate_candidates", fake_enum)

    rc = cli.main(["phase-discover", "--request", str(req_path)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    refs = sorted(c["ref"] for c in payload["candidates"])
    assert refs == ["PR:9999"]
    assert payload["candidate_count"] == 1
    assert payload["excluded_count"] == 2


def test_phase_discover_missing_request_exits_two(tmp_path: Path) -> None:
    rc = cli.main(["phase-discover", "--request", str(tmp_path / "nope.json")])
    assert rc == 2


def test_phase_discover_bad_root_exits_two(tmp_path: Path) -> None:
    bad = tmp_path / "list.json"
    bad.write_text("[]", encoding="utf-8")
    rc = cli.main(["phase-discover", "--request", str(bad)])
    assert rc == 2
