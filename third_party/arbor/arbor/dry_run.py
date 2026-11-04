"""Dry-run mode — exercises the full harness loop without GPU, server, or container.

Uses a local temp git repo as workspace, synthetic bench results, and real
state management / escalation / gate logic. Completes in ~20 seconds.

    python3 -m arbor run --config configs/agentx-glm52-mi355x.json --dry-run

The temp repo is local, so its commits run here (``@host``): no docker needed.
"""

from __future__ import annotations

import json
import random
import subprocess
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

from arbor.cli.iterate import _find_arbor_root
from arbor.execute import workspace
from arbor.measure.gate import run_gate
from arbor.pack import load_campaign, render_program
from arbor.state import CampaignState


def _make_local_repo(tmp: Path) -> Path:
    repo = tmp / "workspace"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "arbor-dry-run@test"],
                   cwd=repo, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "Arbor Dry Run"],
                   cwd=repo, capture_output=True, check=True)
    (repo / "arbor").mkdir()
    (repo / "arbor" / "config.json").write_text('{"server_env": {}, "server_flags": ""}')
    subprocess.run(["git", "add", "-A"], cwd=repo, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "initial"],
                   cwd=repo, capture_output=True, check=True)
    return repo


def _local_git(_container: str, workspace_dir: str, *args: str) -> str:
    r = subprocess.run(["git", *args], cwd=workspace_dir,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {(r.stderr or r.stdout).strip()}")
    return r.stdout.strip()


def _mock_has_changes(_container: str, workspace_dir: str) -> bool:
    r = subprocess.run(["git", "status", "--porcelain"], cwd=workspace_dir,
                       capture_output=True, text=True)
    return bool(r.stdout.strip())


def main(argv: list[str]) -> int:
    import argparse
    parser = argparse.ArgumentParser(prog="arbor dry-run")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-iters", type=int, default=10)
    parser.add_argument("--cooldown", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    config_path = str(Path(args.config).resolve())
    arbor_root = str(_find_arbor_root(Path(config_path)))
    campaign = load_campaign(config_path, arbor_root)

    with tempfile.TemporaryDirectory(prefix="arbor-dry-run-") as tmp:
        tmp = Path(tmp)
        repo = _make_local_repo(tmp)
        session_dir = tmp / "session"
        session_dir.mkdir()

        state = CampaignState.from_campaign(campaign)
        state_path = session_dir / "session.json"

        # Pin validator digests (synthetic — no Docker)
        state.validator_digests = {"validator/bench.sh": "aaa", "validator/correctness.sh": "bbb"}

        # Baseline
        baseline = 100.0
        state.record_baseline(baseline)
        state.save(state_path)
        print(f"[dry-run] Baseline: {baseline} {state.metric_unit}")

        levels = campaign.space.get("levels", [])
        direction = state.metric_direction
        metric_key = campaign.pack.objective["metric"]
        secondary_key = campaign.pack.objective.get(
            "measurement", {}).get("secondary_metric")
        changes = [
            "set AITER_USE_XCD4_ALLREDUCE=1",
            "set SGLANG_EAGLE_TOPK=3",
            "export HIP_FORCE_DEV_KERNARG=1",
            "patch sglang/srt/layers/moe.py — fused gate",
            "rewrite attention backward kernel",
            "add persistent MoE dispatch",
            "enable flash decoding v2",
            "tune NCCL_ALGO=Ring",
            "set HIPBLASLT_TUNING_FILE=/tmp/gemm.csv",
            "add custom all-reduce kernel",
        ]

        for i in range(1, args.max_iters + 1):
            state = CampaignState.load(state_path)

            # Escalation
            escalation = state.check_escalation(levels)
            if escalation:
                state.apply_escalation(escalation)
                print(f"[dry-run] ESCALATION → {escalation}")

            # Render program (exercises real render logic)
            program = render_program(campaign, state, arbor_root, escalation=escalation)

            change = changes[(i - 1) % len(changes)]
            print(f"\n[dry-run] === Iteration {i} === tier={state.tier} null_streak={state.null_streak}")
            print(f"[dry-run] Change: {change}")

            # Simulate agent edit
            cfg_path = repo / "arbor" / "config.json"
            cfg = json.loads(cfg_path.read_text())
            cfg.setdefault("server_env", {})
            cfg["server_env"][f"DRY_RUN_ITER_{i}"] = "1"
            cfg_path.write_text(json.dumps(cfg, indent=2))

            # Commit using real workspace code (mocked to local git)
            with patch.object(workspace, "git", side_effect=lambda c, w, *a: _local_git(c, str(repo), *a)):
                sha = workspace.commit(workspace.HOST, str(repo), f"iter-{i}: {change}")
            print(f"[dry-run] Committed: {sha[:12]}")

            # Synthetic bench — some better, some worse.
            #
            # The metric key comes from the PACK, not a literal. Hardcoding
            # "output_throughput" made this whole dry run vacuous for any pack
            # gating on something else: agentx-dsv41-6 gates on
            # mean_interactivity, so all twelve iterations returned
            # "bench output missing metric key" and the run exercised nothing
            # but the not-measured path while still exiting 0.
            noise = rng.gauss(0, 3)
            if rng.random() < 0.4:
                noise += 5  # 40% chance of improvement
            metric = (state.best_metric or baseline) + noise
            bench_result = {metric_key: round(metric, 1), "status": "ok",
                            "num_requests_successful": 100, "num_requests_total": 100}
            if secondary_key:
                # Loosely correlated with the primary, so frontier trades occur.
                bench_result[secondary_key] = round(
                    (state.best_secondary or baseline / 2)
                    + rng.gauss(0, 2) - noise * 0.3, 1)
            correctness = {"passed": True, "score": 1.0}

            # Gate using real gate logic
            gate = run_gate(campaign.pack.objective, state.best_metric,
                            bench_result, correctness, integrity_ok=True,
                            secondary_best=state.best_secondary)

            # Save patch on KEEP
            iter_dir = session_dir / f"iter_{i}"
            iter_dir.mkdir(exist_ok=True)
            (iter_dir / "bench_output.json").write_text(json.dumps(bench_result, indent=2))

            if gate.decision == "KEEP":
                with patch.object(workspace, "git", side_effect=lambda c, w, *a: _local_git(c, str(repo), *a)):
                    workspace.save_patch(workspace.HOST, str(repo), str(iter_dir / "patch.diff"))
                state.record_keep(iteration=i, metric=gate.metric_value,
                                  change=change, tier=state.tier,
                                  advanced_metric=gate.booked,
                                  secondary=gate.secondary_value,
                                  frontier=gate.frontier)
                shape = " [frontier]" if gate.frontier else \
                        ("" if gate.booked else " [kept, not booked]")
                print(f"[dry-run] KEEP: {gate.metric_value:.1f} "
                      f"({gate.delta_pct:+.1f}%){shape}")
            elif gate.decision == "INVALID":
                state.record_invalid(iteration=i, change=change, tier=state.tier,
                                     reason=gate.reason)
                print(f"[dry-run] INVALID (not measured): {gate.reason}")
            else:
                with patch.object(workspace, "git", side_effect=lambda c, w, *a: _local_git(c, str(repo), *a)):
                    workspace.revert(workspace.HOST, str(repo))
                state.record_revert(iteration=i, metric=gate.metric_value,
                                    change=change, tier=state.tier, reason=gate.reason)
                print(f"[dry-run] REVERT: {gate.reason}")

            state.save(state_path)
            time.sleep(args.cooldown)

        # Summary
        final = CampaignState.load(state_path)
        keeps = sum(1 for l in final.lessons if l.get("decision") == "KEEP")
        reverts = sum(1 for l in final.lessons if l.get("decision") == "REVERT")
        escalations = sum(1 for l in final.lessons if l.get("decision") == "ESCALATE")

        print(f"\n{'=' * 50}")
        print(f"  DRY-RUN COMPLETE")
        print(f"{'=' * 50}")
        print(f"  Iterations: {final.iteration}")
        print(f"  Baseline:   {final.baseline} {final.metric_unit}")
        print(f"  Best:       {final.best_metric} {final.metric_unit}")
        print(f"  Tier:       {final.tier}")
        print(f"  KEEPs={keeps}  REVERTs={reverts}  ESCALATIONs={escalations}")
        for l in final.lessons:
            d = l.get("decision", "?")
            c = l.get("change", "?")
            m = l.get("metric", "?")
            t = l.get("tier", "?")
            r = l.get("reason", "")
            line = f"    [{d}] [{t}] {c} → {m}"
            if r:
                line += f" ({r})"
            print(line)

        # Verify session dir has expected artifacts
        patches = list(session_dir.glob("iter_*/patch.diff"))
        benches = list(session_dir.glob("iter_*/bench_output.json"))
        print(f"\n  Artifacts: {len(patches)} patches, {len(benches)} bench outputs")
        print(f"  Program doc: {len(program)} chars")

    return 0
