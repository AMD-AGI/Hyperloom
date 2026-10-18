# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The dashboard's data and its self-contained page, built from one notebook snapshot."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH, ClaimKey, Role, Writer
from hyperloom_multibranch.contracts.notebook.records import (
    BranchNode,
    Claim,
    NotebookSnapshot,
    QuarantineEvent,
    TestRecord,
)
from hyperloom_multibranch.contracts.notebook.vocab import ClaimStatus, LineageKind, QuarantineAction
from hyperloom_multibranch.contracts.toolbox import QueueState
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.notebook.dashboard import build_dashboard, render_html, write_dashboard
from hyperloom_multibranch.notebook.rsi.frontier import FrontierPolicy, score_progress
from hyperloom_multibranch.notebook.tests.factories import request
from hyperloom_multibranch.notebook.tests.sessions import FP8_KV, GEMM_AGAIN, METRIC

HOSTILE = "<script>alert('x')</script>"


@pytest.fixture
def busy(snapshot: NotebookSnapshot) -> NotebookSnapshot:
    """The scripted session plus a waiting test, a withdrawn one, a held claim, a split branch, and a broken node
    whose reason is markup."""

    waiting = TestRecord(request("r9", GEMM_AGAIN), requested_at=snapshot.taken_at - 30)
    at = snapshot.taken_at
    withdrawn = TestRecord(
        request("r10", GEMM_AGAIN), requested_at=at - 60, withdrawn_at=at - 10, withdrawal_reason="x"
    )
    held = Claim(ClaimKey("c1", FP8_KV.fingerprint), "s1", FP8_KV, "a", 60.0, at, at, at + 45, ClaimStatus.HELD)
    split = BranchNode("s1-fix", LineageKind.SPLIT, "c1", "c1", from_branch="s1")
    lineage = dataclasses.replace(snapshot.lineage, branches=(*snapshot.lineage.branches, split))
    broken = QuarantineEvent("node-3", QuarantineAction.QUARANTINE, HOSTILE, (0, 1))
    return dataclasses.replace(
        snapshot,
        records=(*snapshot.records, waiting, withdrawn),
        claims=(held,),
        lineage=lineage,
        quarantined=(broken,),
    )


def test_the_dashboard_holds_branches_in_flight_work_lineage_and_infrastructure(busy: NotebookSnapshot) -> None:
    rows = score_progress(busy, (), FrontierPolicy(METRIC))

    queue = QueueState({0: 1, 1: 0, 2: 2, 3: 0, 4: 0}, {"s1": 1}, {"s1": 3}, cold_start_seconds=812.0)
    dashboard = build_dashboard(busy, rows, metric=METRIC, queue=queue)

    branches = {row["branch"]: row for row in dashboard["branches"]}
    assert branches["s1"]["gain_pct"] == pytest.approx(4.0)
    assert branches["s1"]["frontier"]["expected_gain_pct"] == pytest.approx(4.0)
    assert branches[INTEGRATION_BRANCH]["gain_pct"] == pytest.approx(4.1)
    assert branches["s2"]["running"] is False and branches["s2"]["frontier"] is None
    assert [(branches[name]["llm_usd"], branches[name]["judged_since_commit"]) for name in ("s1", "s2")] == [
        (1.25, 1),
        (0.5, 2),
    ]
    assert dashboard["spend"] == {"gpu_hours": 1.0, "llm_usd": 2.0}
    (waiting,) = dashboard["in_flight"]["tests"]
    assert (waiting["request_id"], waiting["waiting_s"]) == ("r9", pytest.approx(30.0))
    assert [(claim["branch"], claim["expires_in_s"]) for claim in dashboard["in_flight"]["claims"]] == [("s1", 45.0)]
    assert dashboard["splits"] == [{"branch": "s1-fix", "from_branch": "s1", "start_commit": "c1"}]
    assert [(row["node"], row["tests"]) for row in dashboard["sandboxes"]] == [("node-1", 5), ("node-2", 1)]
    assert dashboard["quarantined"][0]["node"] == "node-3"
    assert dashboard["queue"] == {
        "queued": 3,
        "running": 1,
        "queued_by_priority": {"0": 1, "1": 0, "2": 2, "3": 0, "4": 0},
        "queued_by_branch": {"s1": 3},
        "running_by_branch": {"s1": 1},
        "cold_start_seconds": 812.0,
    }
    assert json.loads(json.dumps(dashboard)) == dashboard


def test_the_page_is_self_contained_and_escapes_every_value(busy: NotebookSnapshot) -> None:
    page = render_html(build_dashboard(busy, score_progress(busy, (), FrontierPolicy(METRIC)), metric=METRIC))

    assert page.startswith("<!DOCTYPE html>") and "<h1>Session sess-1</h1>" in page
    assert HOSTILE not in page and "&lt;script&gt;" in page
    assert not any(marker in page for marker in ("http://", "https://", "<link", "<script", " src="))
    assert "<td>+4.00%</td>" in page and "Toolbox queue" not in page
    assert "spent 1.00 GPU hours and $2.00 of LLM calls" in page
    assert "<td>stopped</td>" in page and "<td>node-3</td>" in page


def test_the_dashboard_is_written_as_json_and_html(busy: NotebookSnapshot, tmp_path: Path) -> None:
    dashboard = build_dashboard(busy, metric=METRIC, queue=QueueState(running_by_branch={"s1": 1}))

    files = write_dashboard(tmp_path / "out", dashboard)

    assert json.loads(files.json.read_text(encoding="utf-8")) == dashboard
    assert files.html.read_text(encoding="utf-8") == render_html(dashboard)
    assert "Toolbox queue" in files.html.read_text(encoding="utf-8")


async def test_an_empty_notebook_has_an_empty_page(tmp_path: Path) -> None:
    empty = await LocalNotebook(tmp_path / "empty", Writer(Role.OBSERVER, "observer")).snapshot()

    page = render_html(build_dashboard(empty))

    assert "<h1>Session without a profile</h1>" in page
    assert page.count('<p class="empty">none</p>') == 7
