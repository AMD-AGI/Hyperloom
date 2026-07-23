# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Behavior-lock tests for ``fa phase-discover``: full emitted JSON field set and
per-gap enumerate failure isolation (enumerate_candidates / read_pr_ledger stubbed)."""

from __future__ import annotations

import json
from pathlib import Path

import hyperloom.agents.framework.kb as kb
import hyperloom.agents.framework.runtime.cli as cli
import hyperloom.agents.framework.sources as src
from hyperloom.agents.framework.models import Candidate


def _write_req(tmp_path: Path, req: dict) -> str:
    path = tmp_path / "req.json"
    path.write_text(json.dumps(req), encoding="utf-8")
    return str(path)


def test_phase_discover_json_golden(monkeypatch, tmp_path: Path, capsys) -> None:
    """Lock the full phase-discover output (top-level + per-candidate fields);
    read_pr_ledger stubbed empty so prior_score is 0.0 and prior_rank is enum order."""
    # gap_keywords passed explicitly to avoid depending on extract_keywords.
    req = {
        "model": "deepseek-r1",
        "framework": "sglang",
        "gpu_type": "MI300X",
        "repo_url": "https://github.com/sgl-project/sglang.git",
        "work_dir": str(tmp_path / "w"),
        "gaps": [
            {
                "gap_canonical_id": "gap-decode-1",
                "gap_description": "low decode throughput",
                "gap_keywords": ["decode", "moe"],
            }
        ],
        "max_search_candidates": 3,
        "batch_id": "batch-test",
    }
    req_path = _write_req(tmp_path, req)

    def fake_enum(r):
        return [
            Candidate(
                ref="PR:1234",
                repo="sgl-project/sglang",
                source="github",
                title="speed up decode",
                html_url="https://github.com/sgl-project/sglang/pull/1234",
                score=0.9,
                labels=("perf", "decode"),
                author="alice",
                changed_files=("a.py", "b.py"),
            )
        ]

    monkeypatch.setattr(src, "enumerate_candidates", fake_enum)
    monkeypatch.setattr(kb, "read_pr_ledger", lambda *a, **k: [])

    rc = cli.main(["phase-discover", "--request", req_path])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)

    # Top-level field set is locked exactly.
    assert set(payload.keys()) == {
        "batch_id",
        "framework",
        "repo_url",
        "model",
        "gpu_type",
        "candidate_count",
        "excluded_count",
        "prior_ranking",
        "candidates",
    }
    assert payload["batch_id"] == "batch-test"
    assert payload["framework"] == "sglang"
    assert payload["repo_url"] == "https://github.com/sgl-project/sglang.git"
    assert payload["model"] == "deepseek-r1"
    assert payload["gpu_type"] == "MI300X"
    assert payload["candidate_count"] == 1
    assert payload["excluded_count"] == 0
    assert payload["prior_ranking"] == {
        "enabled": True,
        "ledger_records": 0,
        "ranked_candidates": 0,
    }

    assert len(payload["candidates"]) == 1
    cand = payload["candidates"][0]
    # Per-candidate field set is locked exactly.
    assert set(cand.keys()) == {
        "pr_url",
        "repo",
        "ref",
        "pr_number",
        "title",
        "summary",
        "score",
        "diff_url",
        "labels",
        "author",
        "framework",
        "model_class",
        "gpu_type",
        "precision",
        "gap_canonical_id",
        "gap_description",
        "gap_keywords",
        "changed_files",
        "prior_score",
        "prior_rank",
    }
    assert cand == {
        "pr_url": "https://github.com/sgl-project/sglang/pull/1234",
        "repo": "sgl-project/sglang",
        "ref": "PR:1234",
        "pr_number": 1234,
        "title": "speed up decode",
        # summary is the comma-joined labels.
        "summary": "perf, decode",
        "score": 0.9,
        "diff_url": "https://github.com/sgl-project/sglang/pull/1234.diff",
        "labels": ["perf", "decode"],
        "author": "alice",
        "framework": "sglang",
        # model_class falls back to model when unset.
        "model_class": "deepseek-r1",
        "gpu_type": "MI300X",
        "precision": "",
        "gap_canonical_id": "gap-decode-1",
        "gap_description": "low decode throughput",
        "gap_keywords": ["decode", "moe"],
        "changed_files": ["a.py", "b.py"],
        "prior_score": 0.0,
        "prior_rank": 1,
    }


def test_phase_discover_per_gap_enumerate_failure_is_isolated(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    """A single gap whose enumerate raises must NOT fail the command: it exits
    0, WARNs to stderr, and still emits candidates from the healthy gaps.
    """
    req = {
        "framework": "sglang",
        "repo_url": "https://github.com/sgl-project/sglang.git",
        "work_dir": str(tmp_path / "w"),
        "gaps": [
            {
                "gap_canonical_id": "gap-boom",
                "gap_description": "will fail",
                "gap_keywords": ["x"],
            },
            {
                "gap_canonical_id": "gap-ok",
                "gap_description": "ok",
                "gap_keywords": ["y"],
            },
        ],
        "batch_id": "b1",
    }
    req_path = _write_req(tmp_path, req)

    def fake_enum(r):
        if r.gap_canonical_id == "gap-boom":
            raise RuntimeError("network down")
        return [
            Candidate(
                ref="PR:77",
                repo="sgl-project/sglang",
                source="github",
                title="ok",
                html_url="https://github.com/sgl-project/sglang/pull/77",
                score=0.5,
            )
        ]

    monkeypatch.setattr(src, "enumerate_candidates", fake_enum)
    monkeypatch.setattr(kb, "read_pr_ledger", lambda *a, **k: [])

    rc = cli.main(["phase-discover", "--request", req_path])
    assert rc == 0

    captured = capsys.readouterr()
    # The failing gap is warned about on stderr, not raised.
    assert "WARN: phase-discover gap='gap-boom' enumerate failed" in captured.err
    assert "network down" in captured.err

    payload = json.loads(captured.out)
    # The healthy gap's candidate still surfaces.
    assert payload["candidate_count"] == 1
    assert [c["ref"] for c in payload["candidates"]] == ["PR:77"]
    assert payload["candidates"][0]["gap_canonical_id"] == "gap-ok"
