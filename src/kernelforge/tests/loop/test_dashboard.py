# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The campaign dashboard: what it reads, what it shows, and that it is safe to open."""

from __future__ import annotations

import json
import stat
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernelforge.loop.dashboard import DASHBOARD_HTML, LiveStatus, publish_dashboard
from kernelforge.loop.dashboard.server import make_server
from kernelforge.loop.operator_inbox import CLEAR, OperatorInbox
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
        "search_start_mean_case_speedup": 0.99,
        "baseline_case_times": {"m=14": 0.0100, "m=7": 0.0103},
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
    rounds = {
        2: {"plan": "Direction: d1\nPrefetch B two tiles ahead", "rationale": "Moved B loads up; 5% faster."},
        3: {"plan": "Direction: d2\nSplit K", "rationale": "Split K by 2 and <i>reduced</i> in the epilogue."},
    }
    for iteration, meta in rounds.items():
        folder = root / "candidates" / f"iter_{iteration:03d}"
        folder.mkdir(parents=True)
        (folder / "meta.json").write_text(json.dumps(meta | {"best_mean_case_speedup_before": 1.0}))
    (root / "candidates" / "iter_003" / "validation.txt").write_text("m=14: max_diff 0.5 > atol")
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


def test_the_page_loads_nothing_and_runs_only_its_own_refresh(campaign: Path) -> None:
    html = publish_dashboard(campaign, _live()).read_text()
    assert html.count("<script") == 1 and "<script>" in html
    assert "src=" not in html
    assert "http://" not in html and "https://" not in html
    assert "<noscript><meta http-equiv='refresh'" in html


def test_each_round_says_what_was_planned_what_was_done_and_why_it_failed(campaign: Path) -> None:
    html = publish_dashboard(campaign, _live()).read_text()
    rounds = html[html.index("<h2>Rounds</h2>") :]
    # Newest first, and only the newest open.
    assert rounds.index("id='round-3' open") < rounds.index("id='round-2'>") < rounds.index("id='round-1'>")
    assert "Split K by 2 and &lt;i&gt;reduced&lt;/i&gt; in the epilogue." in rounds
    assert "m=14: max_diff 0.5 &gt; atol" in rounds
    assert "Prefetch B two tiles ahead" in rounds and "(+5.00% vs best 1.0000×)" in rounds


def test_a_round_is_told_from_its_lesson_record_when_there_is_one(campaign: Path) -> None:
    (campaign / "lessons").mkdir()
    (campaign / "lessons" / "iter_002.md").write_text("**What was built:** a two-stage <b>prefetch</b>.\nOUTCOME: KEEP")
    rounds = publish_dashboard(campaign, _live()).read_text().split("<h2>Rounds</h2>")[1]
    round_2 = rounds[rounds.index("id='round-2'") : rounds.index("id='round-1'")]
    assert "a two-stage &lt;b&gt;prefetch&lt;/b&gt;." in round_2 and "Moved B loads up" not in round_2


def test_each_round_names_the_skill_files_it_opened(campaign: Path) -> None:
    events = [json.loads(line) for line in (campaign / "events.jsonl").read_text().splitlines()]
    events[2]["skill_reads"] = ["flydsl-kernel-optimization/SKILL.md", "flydsl-kernel-optimization/gemm/x.md"]
    events[3]["skill_reads"] = []
    (campaign / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
    rounds = publish_dashboard(campaign, _live()).read_text().split("<h2>Rounds</h2>")[1]
    round_3, round_2 = rounds[: rounds.index("id='round-2'")], rounds[rounds.index("id='round-2'") :]
    assert "Skill files it opened</h3><pre>none</pre>" in round_3
    assert "flydsl-kernel-optimization/SKILL.md\nflydsl-kernel-optimization/gemm/x.md" in round_2
    # Round 1 was recorded before skill reads were: it says nothing rather than "none".
    assert rounds.count("Skill files it opened") == 2


def test_a_run_that_ended_says_so_instead_of_counting_down(campaign: Path) -> None:
    html = publish_dashboard(campaign, _live(status="paused", termination_reason="budget_exhausted")).read_text()
    tile = html[html.index(">Time left<") :].split("</div></div>")[0]
    assert ">ended<" in tile and "budget_exhausted" in tile


def test_the_page_carries_a_message_form_even_with_an_empty_inbox(campaign: Path) -> None:
    html = publish_dashboard(campaign, _live()).read_text()
    assert "<form class='say' method='post' action='say'>" in html
    assert "name='message'" in html and "name='clear'" in html
    assert "No message stands." in html


def test_the_published_page_is_readable_by_whoever_syncs_it(campaign: Path) -> None:
    for path in (publish_dashboard(campaign, _live()), campaign / "dashboard.json"):
        assert stat.S_IMODE(path.stat().st_mode) == 0o644


def test_cases_run_in_numeric_order(campaign: Path) -> None:
    html = publish_dashboard(campaign, _live()).read_text()
    assert html.index("<td>m=7</td>") < html.index("<td>m=14</td>")
    assert "<th class='num'>anchor µs</th>" in html and "<td class='num'>10.30</td>" in html


def _serve(campaign: Path):
    server = make_server(campaign.parent, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _post(url: str, fields: dict) -> int:
    request = urllib.request.Request(url, data=urllib.parse.urlencode(fields).encode(), method="POST")
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        return opener.open(request).status
    except urllib.error.HTTPError as error:
        return error.code


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def test_the_served_page_takes_a_message_and_a_clear(campaign: Path) -> None:
    publish_dashboard(campaign, _live())
    server, base = _serve(campaign)
    try:
        page = urllib.request.urlopen(f"{base}/dashboard.html").read().decode()
        assert "<h2>Rounds</h2>" in page
        assert _post(f"{base}/say", {"message": "focus on m=7"}) == 303
        inbox = OperatorInbox(campaign.parent)
        assert [path.read_text().strip() for path in inbox.pending()] == ["focus on m=7"]
        assert "waiting for the next iteration: focus on m=7" in urllib.request.urlopen(base + "/").read().decode()
        assert _post(f"{base}/say", {"message": "ignored", "clear": "1"}) == 303
        assert inbox.pending()[-1].read_text().strip() == CLEAR
        assert _post(f"{base}/say", {"message": "  "}) == 400
        assert _post(f"{base}/elsewhere", {"message": "x"}) == 404
    finally:
        server.shutdown()
        server.server_close()


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


def test_before_any_keep_the_best_is_the_starting_kernel(campaign: Path) -> None:
    state = json.loads((campaign / "run_state.json").read_text())
    (campaign / "run_state.json").write_text(json.dumps(state | {"best": {}}))
    html = publish_dashboard(campaign, _live()).read_text()
    tile = html[html.index("Best speedup") :].split("</div></div>")[0]
    assert "0.9900×" in tile and "the starting kernel" in tile


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
