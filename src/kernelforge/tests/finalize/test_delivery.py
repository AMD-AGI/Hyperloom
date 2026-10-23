# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""forge-finalize: the review round trip, its gates, the confinement, and the one commit it leaves."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from kernelforge.cli import main
from kernelforge.finalize import HISTORY_BRANCH, FinalizeError, finalize_campaign
from kernelforge.finalize.review import REPORT_HEADING, ReviewBrief, review_prompt
from kernelforge.loop.dashboard import publish_dashboard
from kernelforge.loop.campaign_config import CampaignConfigStore, CampaignInputs, create_campaign_config
from kernelforge.loop.run_state import BestRecord, LoopStateStore, RunState, make_event

# Times every case at the kernel's SPEED; correctness fails while it sets CORRECT = False.
DRIVER = """import pathlib, runpy, sys
kernel = runpy.run_path(str(pathlib.Path(__file__).resolve().parent / "kernel.py"))
if "--bench-mode" in sys.argv:
    for case in ("m=8", "m=16"):
        print(f"case_ms: {case} {kernel['SPEED']:.6f}")
    print(f"median_ms: {kernel['SPEED'] * 2:.6f}")
else:
    print(f"SNR: {40.0 if kernel.get('CORRECT', True) else 5.0} dB")
"""

HEADER = "# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.\n# SPDX-License-Identifier: MIT\n\n"
CLEAN = HEADER + "SPEED = 0.8\n"
# What the campaign left: a passenger knob, a debug print, and its own notes.
DELIVERED = HEADER + (
    "# round 2: split K four ways, measured 1.25x\n# round 3: tried 8 ways, slower\n# round 4: kept 4\n# notes\n"
    "SPEED = 0.8\nUNUSED_ARM = False\nprint('debug')\n"
)


def _git(repo: Path, *args: str) -> str:
    identity = ["-c", "user.name=t", "-c", "user.email=t@t"]
    return subprocess.run(["git", "-C", str(repo), *identity, *args], capture_output=True, text=True, check=True).stdout


def _campaign(tmp_path: Path) -> tuple[Path, str, str]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "driver.py").write_text(DRIVER)
    (workspace / "kernel.py").write_text(HEADER + "SPEED = 1.0\n")
    (workspace / "program.md").write_text("Make it fast.\n")
    (workspace / ".gitignore").write_text("forge_experiments/\n__pycache__/\n")
    _git(workspace, "init", "-q", "-b", "forge")
    _git(workspace, "add", "-A")
    _git(workspace, "commit", "-qm", "start")
    base = _git(workspace, "rev-parse", "HEAD").strip()
    CampaignConfigStore(str(workspace)).save(
        create_campaign_config(
            str(workspace), CampaignInputs(kernel="kernel.py", driver="driver.py", kernel_backend="triton")
        )
    )
    store = LoopStateStore(str(workspace))
    for iteration, (text, score) in enumerate([(HEADER + "SPEED = 0.9\n", 1 / 0.9), (DELIVERED, 1.25)], start=1):
        (workspace / "kernel.py").write_text(text)
        _git(workspace, "commit", "-qam", f"keep {iteration}")
        store.append_event(
            make_event(
                "iteration_result",
                iteration,
                decision="KEEP",
                plan=f"change {iteration}",
                commit_hash=_git(workspace, "rev-parse", "HEAD").strip(),
                best_after_mean_case_speedup=score,
            )
        )
    best = _git(workspace, "rev-parse", "HEAD").strip()
    store.save(
        RunState(
            search_start_mean_case_speedup=1.0,
            baseline_case_times={"m=8": 1.0, "m=16": 1.0},
            best=BestRecord(iteration=2, mean_case_speedup=1.25, commit_hash=best),
        )
    )
    return workspace, base, best


def _reviewer(edits: list[str | None], briefs: list[ReviewBrief]):
    """A review that writes each round's kernel text, wanders outside its files, and reports; None fails."""

    async def review_fn(*, workspace: str, brief: ReviewBrief, record: Path) -> str | None:
        briefs.append(brief)
        text = edits.pop(0)
        if text is None:
            return None
        root = Path(workspace)
        (root / "kernel.py").write_text(text)
        (root / "driver.py").write_text("print('rewritten')\n")
        (root / "_probe.py").write_text("print('probe')\n")
        return f"Read it all.\n{REPORT_HEADING}\n- removed the round notes and the unused arm\n- dropped the print\n"

    return review_fn


async def test_the_review_is_kept_and_the_rounds_become_one_commit(tmp_path: Path) -> None:
    workspace, base, best = _campaign(tmp_path)
    briefs: list[ReviewBrief] = []
    report = await finalize_campaign(
        workspace, review_fn=_reviewer([CLEAN], briefs), author="kyle-256 <Kyle.Zhao@amd.com>"
    )
    assert report["gates"]["kept"] is True and report["gates"]["timing"]["mean"] == pytest.approx(1.0)
    assert report["review"]["rounds"][0]["stray"] == ["driver.py", "_probe.py"]
    assert report["review"]["rounds"][0]["style"] == []
    # The review saw what the campaign left: notes over three lines, numbers in them, a debug print.
    assert any("[block]" in line for line in briefs[0].open_findings)
    assert any("[debug]" in line for line in briefs[0].open_findings)
    assert _git(workspace, "rev-list", "--count", f"{base}..HEAD").strip() == "1"
    assert _git(workspace, "rev-parse", HISTORY_BRANCH).strip() == best
    assert _git(workspace, "log", "-1", "--format=%an <%ae>|%cn <%ce>").strip() == (
        "kyle-256 <Kyle.Zhao@amd.com>|kyle-256 <Kyle.Zhao@amd.com>"
    )
    assert (workspace / "kernel.py").read_text() == CLEAN
    assert "rewritten" not in (workspace / "driver.py").read_text() and not (workspace / "_probe.py").exists()
    assert report["paths"] == ["kernel.py"] and report["outside_sources"] == []
    message = _git(workspace, "log", "-1", "--format=%B")
    assert "Review: kept" in message and "removed the round notes" in message and "1.0000x -> 1.2500x" in message
    assert _git(workspace, "status", "--porcelain").strip() == ""
    html = publish_dashboard(workspace / "forge_experiments").read_text()
    assert "<h2>Delivery</h2>" in html and report["final_commit"][:12] in html and "removed the round notes" in html
    with pytest.raises(FinalizeError, match="already delivered"):
        await finalize_campaign(workspace, review_fn=_reviewer([CLEAN], []))


@pytest.mark.parametrize(
    ("kernel", "reason"),
    [
        (CLEAN.replace("SPEED = 0.8", "SPEED = 0.8\nCORRECT = False"), "fails correctness"),
        (CLEAN.replace("SPEED = 0.8", "SPEED = 0.9"), "slower than the best"),
    ],
)
async def test_a_review_that_breaks_or_slows_the_kernel_is_not_delivered(tmp_path: Path, kernel, reason) -> None:
    workspace, base, best = _campaign(tmp_path)
    report = await finalize_campaign(workspace, review_fn=_reviewer([kernel], []))
    assert report["gates"]["kept"] is False and reason in report["gates"]["reason"]
    assert _git(workspace, "diff", best, "HEAD").strip() == ""
    assert "Review: not kept" in _git(workspace, "log", "-1", "--format=%B")


async def test_findings_left_by_a_round_go_to_the_next_and_a_failed_review_delivers_the_best(tmp_path: Path) -> None:
    workspace, base, best = _campaign(tmp_path)
    lint = tmp_path / "lint.py"
    lint.write_text("import sys\nsys.exit(1 if 'UNUSED_ARM' in open(sys.argv[1]).read() else 0)\n")
    briefs: list[ReviewBrief] = []
    still_long = HEADER + "# one\n# two\n# three\n# four\nSPEED = 0.8\nUNUSED_ARM = 1\n"
    report = await finalize_campaign(
        workspace, review_fn=_reviewer([still_long, CLEAN], briefs), lint_commands=(f"python {lint}",)
    )
    assert len(briefs) == 2 and briefs[1].round_number == 2
    assert any("[block]" in line for line in briefs[1].open_findings)
    assert any(f"python {lint}" in line for line in briefs[1].open_findings)
    assert report["review"]["rounds"][1]["style"] == [] and report["gates"]["kept"] is True

    (tmp_path / "failed").mkdir()
    workspace, _, best = _campaign(tmp_path / "failed")
    report = await finalize_campaign(workspace, review_fn=_reviewer([None], []))
    assert report["review"]["rounds"][0]["failed"] is True
    assert report["gates"] == {"kept": False, "reason": "the review changed nothing"}
    assert _git(workspace, "diff", best, "HEAD").strip() == ""
    finalize = json.loads((workspace / "forge_experiments" / "finalize.json").read_text())
    assert finalize["final_commit"] == _git(workspace, "rev-parse", "HEAD").strip()


def test_the_review_is_told_which_rounds_pay(tmp_path: Path) -> None:
    from kernelforge.finalize.delivery import pricing_text

    ladder = {
        "a" * 40: {
            "iteration": 2,
            "lines": 40,
            "status": "measured",
            "within_noise": False,
            "title": "split K",
            "cost": {"mean": 0.2, "lower": 0.18, "upper": 0.22},
        },
        "b" * 40: {
            "iteration": 3,
            "lines": 90,
            "status": "measured",
            "within_noise": True,
            "title": "XCD remap",
            "cost": {"mean": 0.001, "lower": -0.004, "upper": 0.006},
        },
    }
    text = pricing_text({"ladder": ladder, "noise_floor": 0.002})
    assert "aaaaaaaaaaaa (iteration 2, 40 lines): pays +20.00%" in text
    assert "bbbbbbbbbbbb (iteration 3, 90 lines): within noise" in text
    prompt = review_prompt(
        ReviewBrief(
            base="b0",
            best="b9",
            kept_commits=2,
            files=("kernel.py",),
            driver="driver.py",
            pricing=text,
            lint_commands=("ruff check",),
        )
    )
    assert "`ruff check kernel.py`" in prompt and "within noise" in prompt and REPORT_HEADING in prompt
    assert "No ablation priced the rounds" in pricing_text({})


def test_a_running_campaign_is_not_finalized(tmp_path: Path) -> None:
    workspace, _, _ = _campaign(tmp_path)
    with LoopStateStore(str(workspace)).workspace_lock():
        result = CliRunner().invoke(main, ["forge-finalize", "--workspace", str(workspace)])
    assert result.exit_code != 0 and "stop forge-loop" in result.output
