# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""forge-ablate: each kept change priced beside the best, the all-off arm, resume, and the refusals."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from kernelforge.cli import main
from kernelforge.loop import ablation as ablation_module
from kernelforge.loop.ablation import ALL_OFF, AblationError, ablate
from kernelforge.loop.campaign_config import CampaignConfigStore, CampaignInputs, create_campaign_config
from kernelforge.loop.dashboard import publish_dashboard
from kernelforge.loop.run_state import BestRecord, LoopStateStore, RunState, make_event

# Time scales with speed.txt and drops 20% while fast_a exists; correctness fails while needs_fix exists without fix.
DRIVER = """import pathlib, sys
here = pathlib.Path(__file__).resolve().parent
speed = float((here / "speed.txt").read_text()) * (0.8 if (here / "fast_a").exists() else 1.0)
if "--bench-mode" in sys.argv:
    for case in ("m=8", "m=16"):
        print(f"case_ms: {case} {speed:.6f}")
    print(f"median_ms: {speed * 2:.6f}")
else:
    broken = (here / "needs_fix").exists() and not (here / "fix").exists()
    print(f"SNR: {5.0 if broken else 40.0} dB")
"""

# (file edits, score after the KEEP): c1 pays alone; c2's revert conflicts with c3; c3 pays alone but breaks
# correctness until c4, so reverting c4 alone is wrong.
KEEPS = [
    ({"fast_a": ""}, 1.25),
    ({"speed.txt": "0.9"}, 1.25 / 0.9),
    ({"speed.txt": "0.85", "needs_fix": ""}, 1.25 / 0.85),
    ({"fix": ""}, 1.25 / 0.85),
]


def _git(repo: Path, *args: str) -> str:
    identity = ["-c", "user.name=t", "-c", "user.email=t@t"]
    return subprocess.run(["git", "-C", str(repo), *identity, *args], capture_output=True, text=True, check=True).stdout


def _campaign(tmp_path: Path, keeps=KEEPS) -> tuple[Path, list[str]]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "driver.py").write_text(DRIVER)
    (workspace / "speed.txt").write_text("1.0")
    (workspace / ".gitignore").write_text("forge_experiments/\n")
    _git(workspace, "init", "-q", "-b", "forge")
    _git(workspace, "add", "-A")
    _git(workspace, "commit", "-qm", "start")
    base = _git(workspace, "rev-parse", "HEAD").strip()
    CampaignConfigStore(str(workspace)).save(
        create_campaign_config(
            str(workspace), CampaignInputs(kernel="speed.txt", driver="driver.py", kernel_backend="triton")
        )
    )
    store = LoopStateStore(str(workspace))
    commits = []
    for iteration, (edits, score) in enumerate(keeps, start=1):
        for name, text in edits.items():
            (workspace / name).write_text(text)
        _git(workspace, "add", "-A")
        _git(workspace, "commit", "-qm", f"keep {iteration}")
        commits.append(_git(workspace, "rev-parse", "HEAD").strip())
        store.append_event(
            make_event(
                "iteration_result",
                iteration,
                decision="KEEP",
                plan=f"Direction: d{iteration}\nchange {iteration}",
                commit_hash=commits[-1],
                direction_id=f"d{iteration}",
                best_after_mean_case_speedup=score,
            )
        )
    store.save(
        RunState(
            search_start_mean_case_speedup=1.0,
            baseline_case_times={"m=8": 1.0, "m=16": 1.0},
            best=BestRecord(
                iteration=len(keeps),
                mean_case_speedup=keeps[-1][1] if keeps else None,
                commit_hash=commits[-1] if commits else base,
            ),
        )
    )
    return workspace, commits


async def test_each_kept_change_is_priced_beside_the_best(tmp_path: Path) -> None:
    workspace, (c1, c2, c3, c4) = _campaign(tmp_path)
    report = await ablate(workspace, pairs=2, null_pairs=2)
    arms = report["arms"]
    best_ms = 0.85 * 0.8
    assert list(arms) == [ALL_OFF, c1, c2, c3, c4]
    assert arms[ALL_OFF]["cost"]["mean"] == pytest.approx(1.0 / best_ms - 1)
    assert arms[c1]["cost"]["mean"] == pytest.approx(0.85 / best_ms - 1)
    assert arms[c3]["cost"]["mean"] == pytest.approx(0.9 * 0.8 / best_ms - 1)
    assert arms[c1]["within_noise"] is False and arms[c1]["direction_id"] == "d1"
    assert arms[c1]["gain_when_kept"] == pytest.approx(0.25)
    assert arms[c2]["status"] == "not_separable"
    assert arms[c4]["status"] == "required_for_correctness" and "SNR" in arms[c4]["detail"]
    combined = (0.85 / best_ms) * (0.9 * 0.8 / best_ms) - 1
    assert report["singles_combined"] == pytest.approx(combined)
    assert report["unpriced"] == 2 and report["completed"] is True
    assert report["additivity"] == pytest.approx(combined / (1.0 / best_ms - 1))
    assert report["noise_floor"] == 0.0
    root = workspace / "forge_experiments"
    assert "not_separable" in (root / "ablation.md").read_text()
    assert _git(workspace, "worktree", "list").count("\n") == 1
    html = publish_dashboard(root).read_text()
    assert "Ablation" in html and "needed for correctness" in html and "tangled with a later change" in html


async def test_the_ladder_prices_each_round_beside_the_version_before_it(tmp_path: Path) -> None:
    workspace, (c1, c2, c3, c4) = _campaign(tmp_path)
    report = await ablate(workspace, pairs=2, null_pairs=2)
    ladder = report["ladder"]
    assert list(ladder) == [c1, c2, c3, c4] and ladder[c2]["parent"] == c1
    # Every rung is timed, even the one whose change cannot be reverted alone or fails correctness without the next.
    assert ladder[c1]["cost"]["mean"] == pytest.approx(1.0 / 0.8 - 1)
    assert ladder[c2]["cost"]["mean"] == pytest.approx(0.8 / 0.72 - 1)
    assert ladder[c3]["cost"]["mean"] == pytest.approx(0.72 / 0.68 - 1)
    assert ladder[c4]["within_noise"] is True and ladder[c4]["cost"]["mean"] == pytest.approx(0.0)
    assert ladder[c2]["cases"] == pytest.approx({"m=16": 0.8 / 0.72 - 1, "m=8": 0.8 / 0.72 - 1})
    assert ladder[c2]["lines"] == 2
    assert report["ladder_combined"] == pytest.approx(1.0 / 0.68 - 1) and report["ladder_unpriced"] == 0
    text = (workspace / "forge_experiments" / "ablation.md").read_text()
    assert "## Round by round" in text and "- within noise: `" + c4[:12] + "`" in text
    html = publish_dashboard(workspace / "forge_experiments").read_text()
    assert "round, beside the version before it" in html


async def test_a_rerun_prices_only_what_the_budget_did_not_reach(tmp_path: Path, monkeypatch) -> None:
    workspace, commits = _campaign(tmp_path)
    first = await ablate(workspace, pairs=2, null_pairs=2, max_hours=1e-9)
    assert {arm["status"] for arm in first["arms"].values()} == {"skipped_budget"}
    assert first["completed"] is False and first["noise_floor"] is not None
    assert "rerun forge-ablate" in (workspace / "forge_experiments" / "ablation.md").read_text()

    second = await ablate(workspace, pairs=2, null_pairs=2)
    assert second["completed"] is True and second["arms"][commits[0]]["status"] == "measured"

    async def no_timing(*_args, **_kwargs):
        raise AssertionError("a settled arm was timed again")

    monkeypatch.setattr(ablation_module, "measure_pairs", no_timing)
    third = await ablate(workspace, pairs=2, null_pairs=2)
    assert third["arms"] == second["arms"]


async def test_a_campaign_that_kept_nothing_is_refused(tmp_path: Path) -> None:
    workspace, _ = _campaign(tmp_path, keeps=[])
    with pytest.raises(AblationError, match="kept no change"):
        await ablate(workspace)


def test_a_running_campaign_is_not_ablated(tmp_path: Path) -> None:
    workspace, _ = _campaign(tmp_path)
    with LoopStateStore(str(workspace)).workspace_lock():
        result = CliRunner().invoke(main, ["forge-ablate", "--workspace", str(workspace)])
    assert result.exit_code != 0 and "stop forge-loop" in result.output
    assert not (workspace / "forge_experiments" / "ablation.json").exists()
