"""CLI entry point for the iteration loop.

Three subcommands that bracket the policy's creative work:

    python3 -m arbor iterate init   --config <path>
    python3 -m arbor iterate begin  --config <path> --session-dir <path> ...
    python3 -m arbor iterate gate   --config <path> --session-dir <path> ...

The harness owns the mechanics. The policy owns what to try.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _find_arbor_root(start: Path) -> Path:
    """Walk up from a path to find the directory containing packs/.

    ARBOR_ROOT, when set to a directory with packs/, wins: a config generated
    outside the checkout (a parallel lane's, in the run's store) has no packs/
    above it.
    """
    env = os.environ.get("ARBOR_ROOT")
    if env and (Path(env) / "packs").is_dir():
        return Path(env)
    p = start if start.is_dir() else start.parent
    for _ in range(10):
        if (p / "packs").is_dir():
            return p
        if p.parent == p:
            break
        p = p.parent
    return start.parent


def _load(config_path: str):
    from arbor.pack import load_campaign
    config_path = Path(config_path).resolve()
    arbor_root = _find_arbor_root(config_path)
    campaign = load_campaign(str(config_path), str(arbor_root))
    return campaign, arbor_root


def _require_state(session_dir: Path):
    """Load state, raising if session.json doesn't exist (missing = misconfigured path)."""
    from arbor.state import CampaignState
    p = session_dir / "session.json"
    if not p.exists():
        raise RuntimeError(
            f"session.json not found at {p} — run 'arbor iterate init' first, "
            "or check that --session-dir points to the right campaign directory."
        )
    return CampaignState.load(p)


def _trace_mark(arbor_root: Path, args: list[str]) -> None:
    """Record a mark via arbor_trace.py. Best-effort — never fails the iteration."""
    try:
        subprocess.run(
            [sys.executable, str(arbor_root / "evals" / "arbor_trace.py"), "mark"] + args,
            timeout=10, capture_output=True,
        )
    except Exception:
        pass


def cmd_init(args: argparse.Namespace) -> int:
    from arbor.execute.workspace import write_workspace_config
    from arbor.measure.integrity import snapshot_digests
    from arbor.state import CampaignState

    campaign, arbor_root = _load(args.config)
    session_dir = campaign.session_root
    session_dir.mkdir(parents=True, exist_ok=True)

    state = CampaignState.from_campaign(campaign)
    from arbor import review
    state.review = review.summary(session_dir)   # the approval, when the review gate ran

    container = campaign.config.get("docker_container", "")
    workspace = campaign.config.get("workspace_root", "")

    # Recover any orphaned iter-N commit ONLY on a genuinely fresh campaign
    # (no existing session.json). On an existing campaign, recover_orphaned_commit
    # with recorded_iteration=0 would revert ALL iter-N commits, destroying KEEPs.
    is_fresh = not (session_dir / "session.json").exists()
    if container and workspace and is_fresh:
        from arbor.execute.workspace import recover_orphaned_commit
        recover_orphaned_commit(container, workspace, recorded_iteration=0)

    # KernelForge pattern: seed arbor/config.json inside the workspace so that
    # config-tier changes (env vars, flags) are git-tracked alongside source patches.
    if container and workspace:
        from arbor import component
        write_workspace_config(
            container, workspace,
            component.seed_env(campaign.config),   # no GPU masks in component mode
            campaign.config.get("server_flags", ""),
        )
        # Auto-commit the seed so has_changes() starts clean.
        from arbor.execute.workspace import commit, has_changes
        if has_changes(container, workspace):
            try:
                commit(container, workspace, "chore: seed arbor/config.json baseline")
            except RuntimeError as e:
                if "nothing to commit" in str(e) or "no changes added" in str(e):
                    print(f"[arbor] No real changes to commit (submodule noise) — continuing",
                          file=sys.stderr)
                else:
                    raise
        print(f"[arbor] Seeded workspace config: {workspace}/arbor/config.json",
              file=sys.stderr)

    protected = campaign.pack.validator.get("protected", [])
    if container and protected:
        from pathlib import PurePosixPath
        bench_script = campaign.config.get("bench_script", "")
        container_pack_root = str(PurePosixPath(bench_script).parent.parent)
        paths = [f"{container_pack_root}/{p}" for p in protected]
        state.validator_digests = snapshot_digests(container, paths)

    state.save(session_dir / "session.json")

    _trace_mark(arbor_root, [
        "session-start",
        "--session-dir", str(session_dir),
        "--config", str(args.config),
    ])

    print(f"[arbor] Campaign initialized: {campaign.identity}", file=sys.stderr)
    print(f"[arbor] Session dir: {session_dir}", file=sys.stderr)
    print(f"[arbor] Metric: {state.metric_name} ({state.metric_direction}) {state.metric_unit}", file=sys.stderr)
    print(f"[arbor] Pack: {campaign.pack.name}", file=sys.stderr)
    return 0


def cmd_begin(args: argparse.Namespace) -> int:
    from arbor.pack import render_program

    campaign, arbor_root = _load(args.config)
    session_dir = Path(args.session_dir).resolve()
    state = _require_state(session_dir)

    # The map first (arbor/directions.py): a new map round resets the streaks
    # escalation reads, and a map that stops the campaign stops it here.
    from arbor import directions
    moved = directions.advance(campaign, state, session_dir)
    if moved == "stop":
        state.save(session_dir / "session.json")
        why = (session_dir / "STOP").read_text(errors="replace").strip() if (session_dir / "STOP").exists() else ""
        print(f"[arbor] Map: the campaign stops ({why or 'MAP_STOP.md'})", file=sys.stderr)
        return 0
    if moved == "round_end":
        # The map closed its round (arbor/rounds.py): the driver runs the A/B first.
        state.save(session_dir / "session.json")
        print("[arbor] Map: the round ends; its A/B runs before the next iteration", file=sys.stderr)
        return 0

    levels = campaign.space.get("levels", [])
    escalation = state.check_escalation(levels)
    if escalation:
        state.apply_escalation(escalation)
        state.save(session_dir / "session.json")  # save before anything that can raise
        _trace_mark(arbor_root, [
            "escalation",
            "--iter", str(args.iter or state.iteration),
            "--null-streak", str(state.null_streak),
        ])
        print(f"[arbor] Escalation: {state.tier} (forced)", file=sys.stderr)

    program = render_program(campaign, state, str(arbor_root), escalation=escalation)

    mark_args = ["branch"]
    if args.iter is not None:
        mark_args += ["--iter", str(args.iter)]
    if args.branch_id:
        mark_args += ["--branch-id", args.branch_id]
    if args.tier:
        mark_args += ["--tier", args.tier]
    if args.hypothesis:
        mark_args += ["--hypothesis", args.hypothesis]
    _trace_mark(arbor_root, mark_args)

    if not escalation:
        state.save(session_dir / "session.json")

    print(program)
    print(f"[arbor] iter={state.iteration} tier={state.tier} "
          f"best={state.best_metric} null_streak={state.null_streak}",
          file=sys.stderr)
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    from arbor.execute.workspace import commit, has_changes, revert, save_patch
    from arbor.measure.runner import run_full_gate

    campaign, arbor_root = _load(args.config)
    session_dir = Path(args.session_dir).resolve()
    state = _require_state(session_dir)

    iter_dir = session_dir / f"iter_{args.iter}"
    tier = state.tier

    # No uncommitted changes — agent may have already committed (prompt says to commit)
    # or may have made no change at all. Check the last commit message to distinguish.
    if not has_changes(args.container, args.workspace):
        from arbor.execute.workspace import git as _git
        try:
            last_msg = _git(args.container, args.workspace, "log", "-1", "--format=%s")
        except RuntimeError:
            last_msg = ""
        # If the agent committed for this iteration, the harness skips its own commit
        # and proceeds to gating. Otherwise, record a no-change REVERT.
        if not last_msg.startswith(f"iter-{args.iter}:"):
            metric_for_revert = state.best_metric if state.best_metric is not None else 0.0
            state.record_revert(
                iteration=args.iter, metric=metric_for_revert,
                change=args.change, tier=tier,
                reason="no changes in workspace — agent made no diff",
            )
            state.save(session_dir / "session.json")
            _trace_mark(arbor_root, [
                "gate", "--iter", str(args.iter),
                "--decision", "REVERT",
                "--metric-value", str(metric_for_revert),
                "--delta-pct", "0",
                "--correctness", "pass",
                "--change", args.change,
            ])
            out = {"decision": "REVERT", "metric_value": 0.0, "delta_pct": 0.0,
                   "reason": "no changes", "best": state.best_metric}
            print(json.dumps(out))
            print("[arbor] REVERT: no changes in workspace", file=sys.stderr)
            return 0
        # Agent already committed — skip our commit, go straight to gate
        print(f"[arbor] Agent committed: {last_msg[:60]}", file=sys.stderr)

    sha = commit(args.container, args.workspace, f"iter-{args.iter}: {args.change}")
    print(f"[arbor] Committed: {sha[:12]}", file=sys.stderr)

    # Scope enforcement — revert immediately if any changed file escapes the workspace.
    # git diff --name-only returns workspace-relative paths, so a valid change is one
    # that does NOT start with ".." (which would traverse above workspace_root).
    from arbor.execute.workspace import changed_files
    changed = changed_files(args.container, args.workspace)
    escaped = [f for f in changed if f.startswith("..") or f.startswith("/")]
    if escaped:
        metric_for_revert = state.best_metric if state.best_metric is not None else 0.0
        reason = f"scope violation — edits outside workspace: {', '.join(escaped)}"
        state.record_revert(
            iteration=args.iter, metric=metric_for_revert,
            change=args.change, tier=tier, reason=reason,
        )
        state.save(session_dir / "session.json")
        revert(args.container, args.workspace)
        out = {"decision": "REVERT", "metric_value": 0.0, "delta_pct": 0.0,
               "reason": reason, "best": state.best_metric}
        print(json.dumps(out))
        print(f"[arbor] REVERT: {reason}", file=sys.stderr)
        return 0
    print(f"[arbor] Scope ok — {len(changed)} file(s): {changed[:3]}", file=sys.stderr)

    # Claude-reported metric: Claude ran the bench itself, saw output, debugged failures.
    # If Claude reported a metric in its DONE line, use it directly — skip the harness bench.
    # The harness still runs correctness independently (cheap, tamper-evident).
    reported_metric = getattr(args, "reported_metric", None)
    if reported_metric is not None:
        print(f"[arbor] Using Claude-reported metric: {reported_metric}", file=sys.stderr)

    try:
        result = run_full_gate(
            campaign, state, str(session_dir), args.iter,
            reported_metric=reported_metric,
        )
    except Exception as gate_exc:
        # Gate raised before deciding — always revert so the workspace is clean.
        print(f"[arbor] Gate error: {gate_exc} — reverting workspace", file=sys.stderr)
        try:
            revert(args.container, args.workspace)
        except Exception as revert_exc:
            print(f"[arbor] Revert also failed: {revert_exc}", file=sys.stderr)
        raise gate_exc  # always re-raise the original gate failure

    if result.decision == "KEEP":
        # Artifact must land on disk before state is recorded — no KEEP without the patch.
        try:
            save_patch(args.container, args.workspace, str(iter_dir / "patch.diff"))
        except Exception as patch_exc:
            print(f"[arbor] save_patch failed: {patch_exc} — downgrading KEEP to REVERT",
                  file=sys.stderr)
            result = result.__class__(
                "REVERT", result.metric_value, result.best, result.delta_pct,
                result.correctness_passed, result.integrity_ok,
                f"save_patch failed: {patch_exc}",
            )
        else:
            state.record_keep(
                iteration=args.iter, metric=result.metric_value,
                change=args.change, tier=tier,
            )
    if result.decision == "INVALID":
        # Not measured: the harness failed, so there is no evidence about the
        # change. Preserve the tree and leave every counter alone.
        state.record_invalid(
            iteration=args.iter, change=args.change, tier=tier,
            reason=result.reason,
        )
    elif result.decision != "KEEP":
        state.record_revert(
            iteration=args.iter, metric=result.metric_value,
            change=args.change, tier=tier, reason=result.reason,
        )
        if result.revert_tree:
            # Archive the diff BEFORE reverting. The revert itself is safe —
            # `git revert` keeps the original commit in history — but that
            # history lives only in the container, and a container lives only on
            # one node whose docker store is not persistent. V6 lost the patches
            # of iterations 41 (+1.29%), 7 (+0.7%) and 14 (+0.76%) that way:
            # three changes the gate scored as primary WINS and rejected for
            # mechanical faults (a draw-contaminated secondary bar, and a
            # correctness.sh crash on a null `content` field). By the time the
            # node was revisited it held zero containers and zero images.
            #
            # Best-effort on purpose: a REVERT must still revert even if the
            # archive fails, so this never raises into the control flow.
            try:
                save_patch(args.container, args.workspace,
                           str(iter_dir / "rejected.patch"))
            except Exception as rej_exc:  # pragma: no cover - diagnostic only
                print(f"[arbor] could not archive rejected patch: {rej_exc}",
                      file=sys.stderr)
            revert(args.container, args.workspace)

    state.save(session_dir / "session.json")

    mark_args = [
        "gate", "--iter", str(args.iter),
        "--decision", result.decision,
        "--metric-value", str(result.metric_value),
        "--delta-pct", str(result.delta_pct),
        "--correctness", "pass" if result.correctness_passed else "fail",
        "--change", args.change,
    ]
    _trace_mark(arbor_root, mark_args)

    out = {
        "decision": result.decision,
        "metric_value": result.metric_value,
        "delta_pct": round(result.delta_pct, 2),
        "reason": result.reason,
        "best": state.best_metric,
    }
    print(json.dumps(out))
    print(f"[arbor] {result.decision}: {result.reason}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="arbor iterate",
        description="Iteration loop — the harness mechanics.",
    )
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("init", help="Initialize a new campaign")
    p.add_argument("--config", required=True)

    p = sub.add_parser("begin", help="Start an iteration — render program, check escalation")
    p.add_argument("--config", required=True)
    p.add_argument("--session-dir", required=True)
    p.add_argument("--iter", type=int, default=None)
    p.add_argument("--branch-id", default=None)
    p.add_argument("--tier", default=None)
    p.add_argument("--hypothesis", default=None)

    p = sub.add_parser("gate", help="Gate the agent's work — commit, measure, keep/revert")
    p.add_argument("--config", required=True)
    p.add_argument("--session-dir", required=True)
    p.add_argument("--container", required=True)
    p.add_argument("--workspace", required=True)
    p.add_argument("--iter", type=int, required=True)
    p.add_argument("--change", required=True)
    p.add_argument("--reported-metric", type=float, default=None,
                   help="Metric value reported by the agent (skips reading bench_output.json)")

    args = parser.parse_args(argv)

    if args.cmd == "init":
        return cmd_init(args)
    if args.cmd == "begin":
        return cmd_begin(args)
    if args.cmd == "gate":
        return cmd_gate(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
