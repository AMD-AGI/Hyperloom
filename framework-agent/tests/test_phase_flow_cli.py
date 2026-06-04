"""Tests for the FRAMEWORK_PR phase-discover subcommand.

Hermetic - stubs ``sources.enumerate_candidates`` so no network/git is
required.
"""

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
        # Mimic per-gap candidates; the second gap returns a duplicate ref
        # to validate dedup.
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
    # diff_url constructed correctly from html_url
    pr1 = next(c for c in payload["candidates"] if c["ref"] == "PR:1234")
    assert pr1["diff_url"].endswith("/pull/1234.diff")
    assert pr1["pr_number"] == 1234
    # Each candidate stamped with the gap it surfaced under.
    assert all("gap_canonical_id" in c for c in payload["candidates"])


def test_phase_discover_missing_request_exits_two(tmp_path: Path) -> None:
    rc = cli.main(["phase-discover", "--request", str(tmp_path / "nope.json")])
    assert rc == 2


def test_phase_discover_bad_root_exits_two(tmp_path: Path) -> None:
    bad = tmp_path / "list.json"
    bad.write_text("[]", encoding="utf-8")
    rc = cli.main(["phase-discover", "--request", str(bad)])
    assert rc == 2
