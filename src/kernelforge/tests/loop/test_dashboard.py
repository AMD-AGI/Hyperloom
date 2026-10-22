# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The campaign dashboard: what it reads, what it shows, and that it is safe to open."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernelforge.loop.dashboard import DASHBOARD_HTML, LiveStatus, publish_dashboard
from kernelforge.loop.runner import IterationLoop


def _result(iteration: int, decision: str, score: float, best: float, plan: str = "a plan") -> dict:
    return {
        "type": "iteration_result",
        "iter": iteration,
        "ts": f"2026-10-22 10:0{iteration}:00",
        "decision": decision,
        "mean_case_speedup": score,
        "best_after_mean_case_speedup": best,
        "plan": plan,
    }


@pytest.fixture
def campaign(tmp_path: Path) -> Path:
    root = tmp_path / "forge_experiments"
    (root / "supervisor").mkdir(parents=True)
    state = {
        "campaign_id": "gemm_n768_k7168",
        "session_status": "running",
        "iteration": 3,
        "cumulative": {"iterations": 3, "kept": 1, "reverted": 2},
        "best": {"iteration": 2, "mean_case_speedup": 1.05},
        "baseline_case_times": {"m=7": 0.0103, "m=14": 0.0100},
        "best_case_times": {"m=7": 0.0100, "m=14": 0.0101},
    }
    (root / "run_state.json").write_text(json.dumps(state))
    events = [
        {"type": "session_started", "iter": 0, "ts": "2026-10-22 10:00:00"},
        _result(1, "REVERT_PERF", 0.98, 1.0),
        _result(2, "KEEP", 1.05, 1.05),
        _result(3, "REVERT_VALIDATION", None, 1.05, plan="<script>alert(1)</script>"),
    ]
    (root / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
    (root / "llm_usage.json").write_text(
        json.dumps(
            {
                "input_tokens": 1000,
                "output_tokens": 200,
                "total_cost_usd": 3.5,
                "calls": 4,
                "by_role": {"implementer": {"input_tokens": 900, "output_tokens": 150, "total_cost_usd": 3.0}},
            }
        )
    )
    (root / "supervisor" / "latest.md").write_text("Try a deeper pipeline <b>now</b>.\n")
    return root


def _live(**overrides) -> LiveStatus:
    values = {"status": "running", "anchor_label": "SIKL baseline", "serving_speedup": 1.03, "time_remaining_sec": 3600}
    return LiveStatus(**(values | overrides))


def test_the_page_carries_every_panel(campaign: Path) -> None:
    html = publish_dashboard(campaign, _live()).read_text()
    for heading in ("Score by iteration", "Decisions", "Cases", "Latest supervisor ruling", "LLM spend by role"):
        assert heading in html
    assert "scores are speedups over SIKL baseline" in html
    assert (campaign / "dashboard.json").is_file()


def test_the_page_loads_nothing_and_runs_nothing(campaign: Path) -> None:
    html = publish_dashboard(campaign, _live()).read_text()
    assert "<script" not in html
    assert "src=" not in html
    assert "http://" not in html and "https://" not in html


def test_agent_text_is_escaped(campaign: Path) -> None:
    html = publish_dashboard(campaign, _live()).read_text()
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "&lt;b&gt;now&lt;/b&gt;" in html


def test_a_case_exactly_at_the_threshold_is_served(campaign: Path) -> None:
    html = publish_dashboard(campaign, _live()).read_text()
    row = html[html.index("<td>m=7</td>") :].split("</tr>")[0]
    assert "1.030" in row and "✓ yes" in row
    assert "1/2 at or above 1.03×" in html


def test_no_threshold_draws_none(campaign: Path) -> None:
    html = publish_dashboard(campaign, _live(serving_speedup=None)).read_text()
    assert "at or above" not in html


def test_a_render_without_the_loop_keeps_its_last_live_status(campaign: Path) -> None:
    publish_dashboard(campaign, _live(anchor_label="SIKL baseline"))
    html = publish_dashboard(campaign).read_text()
    assert "scores are speedups over SIKL baseline" in html


def test_an_empty_campaign_still_renders(tmp_path: Path) -> None:
    root = tmp_path / "forge_experiments"
    root.mkdir()
    html = publish_dashboard(root, _live()).read_text()
    assert "No measured iteration yet." in html


def test_the_loop_publishes_and_never_raises(campaign: Path, tmp_path: Path) -> None:
    loop = SimpleNamespace(
        ic=SimpleNamespace(deadline_unix=None, max_time_hours=2.0, anchor_label="SIKL baseline", serving_speedup=1.03),
        start_time=1.0,
        termination_reason="",
        config=SimpleNamespace(experiments_dir=campaign),
        state_store=SimpleNamespace(root=campaign),
        persistence_errors=[],
        _time_remaining=lambda: 1800.0,
    )
    IterationLoop._publish_dashboard(loop)
    assert "Score by iteration" in (campaign / DASHBOARD_HTML).read_text()

    not_a_directory = tmp_path / "file"
    not_a_directory.write_text("")
    loop.state_store = SimpleNamespace(root=not_a_directory)
    IterationLoop._publish_dashboard(loop)
    assert loop.persistence_errors and loop.persistence_errors[0].startswith("publish dashboard")
