"""The loop. One command: python3 -m arbor run --config <path>

Initializes if needed, then loops: policy picks a candidate, harness gates it.
State persists across iterations via session.json. Stop conditions: target,
time budget, STOP file, max iterations. With a round-end A/B hook (arbor/rounds.py)
the target is not a stop: it ends the round, like an explored map does.

In component mode (arbor/component.py, docs/component-mode.md) a host owns the
clock: an iteration starts only if it fits before the host's deadline, and every
exit writes component_exit.json with its reason before returning 0.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

# Force line-buffered stdout so tee/pipes see each line immediately.
# Without this, block-buffered stdout drops lines when the process is piped.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

from arbor import component, directions, review, rounds
from arbor.cli.iterate import _find_arbor_root, cmd_init
from arbor.execute.workspace import recover_orphaned_commit
from arbor.pack import load_campaign, validate_campaign
from arbor.pack.loader import PACK_RESULTS_CONCERNS, PACK_RESULTS_DIR, PACK_RESULTS_FLAG
from arbor.policy.claude_loop import run_iteration
from arbor.rebaseline import current as rebase_scale
from arbor.state import CampaignState


def _resolve_target(target, metric: str):
    """Resolve a campaign `target` to a number, or None if it carries none.

    A config may express `target` either as a bare number or as a dict of
    documented reference points — e.g. the dsv41flash campaign records
    ``{"_source": ..., "output_tok_per_chip": 328.59, "gap_pct": 26.8}``.
    Comparing `best_metric >= <dict>` raises TypeError, which killed the loop
    on the first iteration that had a metric to compare (iteration 1 was safe
    only because best_metric was still None).
    """
    if target is None or isinstance(target, bool):
        return None
    if isinstance(target, (int, float)):
        return float(target)
    if isinstance(target, dict):
        for key in (metric, f"{metric}_per_chip", "output_tok_per_chip", "value"):
            v = target.get(key)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
    return None


def _target_reached(state: CampaignState, target, direction: str) -> bool:
    if target is None or state.best_metric is None:
        return False
    if not isinstance(target, (int, float)) or isinstance(target, bool):
        return False
    # The target was set on the first node's bar; after a rebase best_metric is
    # this node's, so compare on the first node's scale (arbor/rebaseline.py).
    best = state.best_metric * rebase_scale(state.lessons)[0]
    if direction == "min":
        return best <= target
    return best >= target


def _aborted_detail(session_dir: Path) -> str:
    """ABORTED.md's message (the streak and the last not-measured reason), for the exit record.

    A host has to tell a broken harness from a finished run: an abort recorded as `done`
    read as a normal end under Hyperloom on Cluster C (2026-10-29, a gateway outage).
    """
    try:
        text = (session_dir / "ABORTED.md").read_text()
    except OSError:
        return ""
    body = " ".join(l.strip() for l in text.splitlines() if l.strip() and not l.startswith("#"))
    return body[:500]


def _aborted_exit(session_dir: Path) -> tuple[str, str]:
    """The exit reason and detail for an ABORTED.md: `unreachable` when the model could not
    be reached (exit 3, the host resumes the run), else `aborted` (exit 0)."""
    reason = "unreachable" if component.unreachable(session_dir) else "aborted"
    return reason, _aborted_detail(session_dir)


def _best_text(state: CampaignState, unit: str) -> str:
    """best=, on the campaign's first-node scale once a rolling chunk has rebased the bar."""
    scale = rebase_scale(state.lessons)[0]
    if state.best_metric is None or abs(scale - 1.0) < 1e-9:
        return f"best={state.best_metric} {unit}"
    return (f"best={state.best_metric * scale:.5f} {unit} on the first node's scale "
            f"({state.best_metric} on this node, x{scale:.4f})")


def _root_accuracy(campaign, state_path: Path, comp) -> None:
    """Component mode: the root's full GSM8K, once, before iteration 1 (arbor/measure/accuracy.py).

    A host gives no baseline record, and its own eval of the root reads on another scale, so
    without this the accuracy gate could hold a numerics change only to the pack's GSM8K_MIN.
    The boot and the eval are the run's time, within the deadline; the result goes into
    iteration 1's row of session.json too.
    """
    from arbor.measure import accuracy
    from arbor.policy import claude_loop
    state = CampaignState.load(state_path)
    why = accuracy.root_skip(campaign.config, state.policy, state.iteration, time.time(), comp.deadline)
    if why:
        if state.policy.accuracy_gate:
            print(f"[arbor] Root accuracy: not measured: {why}")
        return
    print("[arbor] Root accuracy: serving the root for the full GSM8K, once: the accuracy gate's golden score")
    rec = accuracy.measure_root(
        campaign.config, start=claude_loop._start_server, healthy=claude_loop._wait_healthy,
        stop=lambda cfg: claude_loop._stop_server(cfg, cfg.get("docker_container", ""), cfg.get("gpu_ids")))
    if rec.get("score") is None:
        print(f"[arbor] Root accuracy: not measured ({rec.get('error')}): GSM8K_MIN stays the floor")
    else:
        print(f"[arbor] Root accuracy: {rec['score']} on {rec['n']} questions in {rec['seconds']:.0f} s")
    state = CampaignState.load(state_path)
    state.iteration_notes["root_accuracy"] = {k: rec.get(k) for k in ("score", "n", "seconds", "error")}
    state.save(state_path)


def _seed_baseline(state_path: Path, campaign, direction: str) -> None:
    """Seed campaign state from the frozen baseline record, if there is one.

    Without this, `best_metric` stays None until the FIRST iteration measures
    something — so iteration 1's own number becomes the bar, and a regression
    measured first would be enshrined as "best". The whole point of running
    validator/baseline.sh three times is to establish that bar beforehand.

    `baseline_record` is an optional config key naming the JSON that
    baseline.sh froze. Absent or unreadable, this is a no-op and the old
    behaviour stands.
    """
    record_path = campaign.config.get("baseline_record")
    if not record_path:
        return
    p = Path(record_path)
    if not p.is_file():
        print(f"[arbor] baseline_record not found: {p} — campaign will seed "
              f"from iteration 1 instead")
        return

    state = CampaignState.load(state_path)
    if state.baseline is not None:
        return  # already seeded; never overwrite a running campaign's bar

    try:
        rec = json.loads(p.read_text())
    except Exception as e:
        print(f"[arbor] baseline_record unreadable ({e}) — not seeding")
        return

    # `mean` is the unbiased estimate across replays. Prefer it; fall back to a
    # bare number for hand-written records.
    value = rec.get("mean")
    if value is None and isinstance(rec.get("runs"), list) and rec["runs"]:
        value = sum(rec["runs"]) / len(rec["runs"])
    if not isinstance(value, (int, float)):
        print(f"[arbor] baseline_record has no usable 'mean' — not seeding")
        return

    state.record_baseline(float(value))
    # A resumed session predates the secondary axis, so re-read the direction
    # from the pack rather than trusting whatever was serialized. Without this
    # a "min" secondary would ratchet the wrong way after an upgrade.
    meas = campaign.pack.objective.get("measurement", {})
    state.secondary_name = meas.get("secondary_metric", "") or state.secondary_name
    state.secondary_direction = meas.get("secondary_direction", "max")
    sec = rec.get("mean_secondary")
    if sec is None and isinstance(rec.get("runs_secondary"), list) and rec["runs_secondary"]:
        sec = sum(rec["runs_secondary"]) / len(rec["runs_secondary"])
    if isinstance(sec, (int, float)) and state.best_secondary is None:
        state.baseline_secondary = float(sec)
        state.best_secondary = float(sec)
        print(f"[arbor] Seeded secondary baseline "
              f"{state.secondary_name}={sec:.3f}")
    elif state.secondary_name and state.best_secondary is None:
        print(f"[arbor] No secondary baseline in {p.name} — "
              f"{state.secondary_name} will seed from the first KEEP")
    state.save(state_path)
    spread = rec.get("relative_spread")
    extra = f", run-to-run spread {spread * 100:.2f}%" if isinstance(spread, (int, float)) else ""
    print(f"[arbor] Seeded baseline from {p.name}: {value:.3f} "
          f"{campaign.pack.objective.get('unit', '')}{extra}")


def _round_end(campaign, session_dir: Path, state_path: Path, roll) -> str:
    """Run the pending round end's A/B and book it (arbor/rounds.py).

    "next" (confirmed: the next round starts), "pending" (incomplete: try again),
    "stop" (STOP written for the operator), "handoff" (too little time left in a
    rolling chunk: the next chunk runs the A/B first) or "exit" (too little time
    left in a job that does not roll).
    """
    pend = rounds.pending(session_dir)
    state = CampaignState.load(state_path)
    if not rounds.hook(campaign):
        out = rounds.stop(state, session_dir, f"round end {pend['end']} ({pend['trigger']}) is pending, "
                                              f"but this launch sets no A/B hook ({rounds.ENV_HOOK})")
        state.save(state_path)
        return out
    need = rounds.knob("round_ab_minutes", campaign) * 60
    now = time.time()
    if roll:
        deadline = roll.switch_at()
    else:
        deadline = now + 2 * need
        job = os.environ.get("SLURM_JOB_ID")
        if job:
            from arbor.cluster.handoff import job_timeleft
            left = job_timeleft(job)
            if left is not None:
                deadline = min(deadline, now + left - 15 * 60)
    if now + need > deadline:
        print(f"[arbor] round: end {pend['end']}'s A/B needs {need / 60:.0f} min and "
              f"{max(0.0, deadline - now) / 60:.0f} remain -- "
              + ("handing off; the next chunk runs it first" if roll else "exiting; the next launch runs it first"))
        return "handoff" if roll else "exit"
    summary = rounds.run_ab(campaign, session_dir, deadline)
    state = CampaignState.load(state_path)
    outcome = rounds.apply(campaign, state, session_dir, summary)
    if outcome == "next" and directions.after_round_end(campaign, state, session_dir,
                                                        pend["trigger"]) == "stop":
        outcome = "stop"
    state.save(state_path)
    if outcome == "pending" and roll:
        return "handoff"
    return outcome


def main(argv: list[str] | None = None) -> int:
    """The loop, and in component mode what ends it from outside.

    SIGTERM from the host (arbor/component.py installs the handler) unwinds the loop
    wherever it is: the iteration in flight is booked NOT MEASURED, our own server is
    stopped, the exit record written, and the driver exits 0 within the host's grace
    (MB: 30 s, then SIGKILL). Any other exception still crashes the driver, which the
    host resumes, but stops our server first.
    """
    held: dict = {}
    try:
        return _main(argv, held)
    except component.Terminated:
        if not held.get("comp"):
            raise
        return held["comp"].terminated()
    except BaseException:
        if held.get("comp"):
            held["comp"].abort()
        raise


def _main(argv: list[str] | None, held: dict) -> int:
    parser = argparse.ArgumentParser(
        prog="arbor run",
        description="Run an Arbor campaign — one command, walk away.",
    )
    parser.add_argument("--config", required=True, help="Campaign config JSON")
    parser.add_argument("--max-iters", type=int, default=0,
                        help="0 = run until target or STOP file")
    parser.add_argument("--cooldown", type=int, default=30,
                        help="Seconds between iterations")
    parser.add_argument(PACK_RESULTS_FLAG, action="store_true",
                        help=f"accept a {PACK_RESULTS_DIR}/ directory inside the pack (refused by "
                             f"default: {PACK_RESULTS_CONCERNS})")
    gate = parser.add_mutually_exclusive_group()
    gate.add_argument("--review-gate", dest="review_gate", action="store_true", default=None,
                      help="a fresh session waits for a human's approval before it runs "
                           "anything (the default; arbor/review.py)")
    gate.add_argument("--no-review-gate", dest="review_gate", action="store_false",
                      help=f"start without the review (also {review.ENV}=0, or "
                           f"\"review_gate\": false in the config)")
    args = parser.parse_args(argv)

    config_path = str(Path(args.config).resolve())
    arbor_root = str(_find_arbor_root(Path(config_path)))
    try:
        campaign = load_campaign(config_path, arbor_root)
    except (*component.UNLOADABLE, yaml.YAMLError) as exc:
        return component.refuse_unloadable(config_path, exc)
    session_dir = campaign.session_root
    if args.allow_pack_results and (campaign.pack.pack_dir / PACK_RESULTS_DIR).is_dir():
        print(f"[arbor] WARNING: {PACK_RESULTS_FLAG}: keeping {campaign.pack.pack_dir / PACK_RESULTS_DIR}/ "
              f"inside the pack. {PACK_RESULTS_CONCERNS[0].upper()}{PACK_RESULTS_CONCERNS[1:]}.",
              file=sys.stderr, flush=True)

    # The mode (arbor/component.py; docs/component-mode.md decision 3): an unknown mode,
    # a component config without a deadline or session_root, off "@host" or under
    # ARBOR_ROLL=1, or a mode the session did not start in, exits 2 before anything else
    # touches the session. In component mode so does a campaign that does not validate
    # (a missing script, a broken pack): it would only crash later, and a host resumes a crash.
    refused = component.refusals(campaign.config, os.environ, session_dir)
    if component.is_component(campaign):
        refused += [e for e in validate_campaign(campaign, allow_results=args.allow_pack_results)
                    if e not in refused]
    if component.stops_at_map(campaign.config) and not directions.enabled(campaign):
        refused.append("component.stop_at \"map\" needs map rounds, which this campaign does not "
                       "run: they need the dossier skill in the pack, a workload that is not a "
                       "document, and map_rounds above 0. It would never write a map")
    if refused:
        return component.refuse(refused, campaign.config, session_dir)

    # Cluster A's rolling chunks (Arbor main only; arbor/cluster/handoff.py): the
    # launcher opts in with ARBOR_ROLL=1. The lease is taken before anything
    # writes the session, so two drivers never share one.
    roll = None
    if os.environ.get("ARBOR_ROLL") == "1":
        from arbor.cluster import handoff
        roll, rc = handoff.start(campaign.config, session_dir)
        if rc is not None:
            return rc
    elif campaign.config.get("docker_container"):
        # Not rolling, the flag KEEPs still go into a fresh container, and each KEEP's are
        # mirrored below: b0's first iteration in a held Cluster B job (2026-10-22) ran
        # without them, without --enable-expert-parallel.
        from arbor.cluster import handoff
        handoff.restore_workspace_config(campaign.config, session_dir)

    # The review gate (arbor/review.py): a fresh session waits here for a human's approval,
    # before cmd_init pins the validators or seeds the workspace and before any agent, server
    # or bench runs. Approved, the campaign is loaded again, so the reviewed files are the
    # ones that run. In component mode the host's clock, heartbeat and SIGTERM cover the wait.
    state_path = session_dir / "session.json"
    gate_on, gate_why = review.decision(campaign.config, args.review_gate)
    if not state_path.exists() and not gate_on and review.status(session_dir) != review.APPROVED:
        review.record_bypass(session_dir, gate_why)   # why it did not wait, for session.json
        print(f"[arbor] Review gate: {'the host reviews' if gate_why == 'host' else 'skipped'} "
              f"({gate_why})")
    if (not state_path.exists() and gate_on
            and review.status(session_dir) != review.APPROVED):
        if component.is_component(campaign):
            held["comp"] = component.Run(campaign.config, session_dir)
            held["comp"].start(campaign.config.get("time_budget_minutes"))
        doc = review.request(campaign, config_path, arbor_root, session_dir)
        print(f"[arbor] Review gate: nothing has run; review {doc}")
        comp = held.get("comp")
        outcome = review.wait(session_dir, deadline=comp.deadline if comp else None,
                              margin=component.DEADLINE_MARGIN_S)
        if outcome != review.APPROVED:
            reason = {review.REJECTED: "review_rejected", "stop": "done"}.get(outcome, "deadline")
            detail = review.outcome_detail(session_dir, outcome)
            print(f"[arbor] Review gate: {reason} ({detail}) — exiting before anything ran.")
            return comp.finish(reason, detail) if comp else 0
        approved = review.load(session_dir) or {}
        print(f"[arbor] Review gate: approved by {approved.get('by')}"
              + (f" ({approved.get('note')})" if approved.get("note") else "")
              + f"; {len(approved.get('changed') or [])} file(s) changed since the request")
        try:
            campaign = load_campaign(config_path, arbor_root)
        except (*component.UNLOADABLE, yaml.YAMLError) as exc:
            return component.refuse_unloadable(config_path, exc)
        refused = component.refusals(campaign.config, os.environ, session_dir)
        if component.is_component(campaign):
            refused += [e for e in validate_campaign(campaign, allow_results=args.allow_pack_results)
                        if e not in refused]
        if refused:
            return component.refuse(refused, campaign.config, session_dir)
        if comp and component.is_component(campaign):
            comp.config = campaign.config            # a reviewed component block
        elif comp:
            comp.close()                             # reviewed out of component mode
            held.pop("comp", None)

    # Auto-init if no session.json
    if not state_path.exists():
        print("[arbor] No session.json — initializing campaign.")
        init_ns = argparse.Namespace(config=config_path)
        cmd_init(init_ns)
    elif CampaignState.load(state_path).mode != campaign.mode:
        # A session from before modes, which recorded none (a recorded one that differs
        # was refused above): it keeps the mode that resumes it from now on.
        state = CampaignState.load(state_path)
        state.mode = campaign.mode
        state.save(state_path)

    start_time = time.time()
    metric_name = campaign.pack.objective.get("metric", "")
    config_target = _resolve_target(campaign.config.get("target"), metric_name)
    _seed_baseline(state_path, campaign, campaign.pack.objective.get("direction", "max"))
    direction = campaign.pack.objective.get("direction", "max")
    time_budget = campaign.config.get("time_budget_minutes")
    unit = campaign.pack.objective.get("unit", "")

    # Component mode: the host's deadline is the clock (component.json keeps the first
    # launch's, so a resume does not restart it), and time_budget_minutes is not read.
    comp = held.get("comp")      # started already when the review gate waited
    if component.is_component(campaign):
        if comp is None:
            comp = held["comp"] = component.Run(campaign.config, session_dir)
            comp.start(time_budget)   # the deadline, the heartbeat, the SIGTERM handler
        time_budget = None
        # The host resumed a run that stopped because its model could not be reached: that
        # ABORTED.md goes aside and the not-measured streak starts again, or the run would
        # end at once, or after one more failure.
        moved = component.clear_unreachable(session_dir)
        if moved:
            st = CampaignState.load(state_path)
            st.invalid_streak = 0
            st.save(state_path)
            print(f"[arbor] Resumed after the model could not be reached: ABORTED.md is now "
                  f"{moved}, and the not-measured streak starts again")
        _root_accuracy(campaign, state_path, comp)
    exit_reason, exit_detail = "done", ""

    print(f"[arbor] Campaign: {campaign.identity}")
    print(f"[arbor] Session dir: {session_dir}")
    print(f"[arbor] Metric: {campaign.pack.objective.get('metric')} ({direction}) {unit}")
    round_target = rounds.target(session_dir, metric_name)
    if comp and (round_target is not None or config_target is not None):
        print(f"[arbor] Target: {round_target if round_target is not None else config_target} {unit}, "
              f"not read in component mode: the host gates and merges")
    elif round_target is not None:
        print(f"[arbor] Target: {round_target} {unit} (set by the last round-end A/B; "
              f"the config's is {config_target})")
    elif config_target is not None:
        print(f"[arbor] Target: {config_target} {unit}")
    if rounds.hook(campaign):
        print("[arbor] Round ends: a reached target or an explored map runs a same-node A/B, "
              "then the next round (arbor/rounds.py)")
    if time_budget is not None:
        print(f"[arbor] Time budget: {time_budget} min")
    print(f"[arbor] Stop: touch {session_dir}/STOP")
    print()

    iteration = 0
    while True:
        if (session_dir / "STOP").exists():
            print("[arbor] STOP file — exiting.")
            break

        # Checked here, not only inside the `except` below: a run of
        # not-measured iterations writes ABORTED.md and returns normally, so
        # without this the loop would keep retrying a broken harness until the
        # allocation ran out.
        if (session_dir / "ABORTED.md").exists():
            print("[arbor] ABORTED.md present — exiting.")
            print((session_dir / "ABORTED.md").read_text())
            exit_reason, exit_detail = _aborted_exit(session_dir)
            break

        state = CampaignState.load(state_path)
        iteration = state.iteration + 1

        # RELOAD THE MEASUREMENT BLOCK EACH ITERATION.
        #
        # `state` is re-read from disk every iteration, so the BAR can be
        # corrected between iterations (scripts/dsv41-campaign/fix_bar.py does
        # exactly that). The BANDS could not: load_campaign() runs once before
        # this loop, and gate.py reads threshold_rel / noise_band_pct /
        # secondary_noise_band_pct off that in-memory pack. Editing pack.yaml
        # mid-run did nothing, and the only way to widen a band was to resubmit
        # the job -- which, since the driver IS the batch script, costs the
        # allocation and a container rebuild.
        #
        # That asymmetry mattered. A band narrower than the measurement noise
        # silently rejects real work, and the noise is routinely underestimated:
        # the 0.79% spread quoted across three campaigns came from two runs on
        # one node, while re-measuring a byte-identical tree on its own node
        # later differed by 3.02%. v2 widened its secondary band 1.7 -> 4.2
        # only at the START of the next campaign, after 47 iterations under a
        # bar it could not reach.
        #
        # Only the measurement block is refreshed. Validators, protected paths
        # and tools stay pinned to what the campaign started with, so a
        # mid-run pack edit cannot change what is measured or what is allowed
        # to change -- only the thresholds that interpret the result.
        try:
            _fresh = load_campaign(config_path, arbor_root)
            _new_meas = _fresh.pack.objective.get("measurement", {})
            _old_meas = campaign.pack.objective.get("measurement", {})
            if _new_meas and _new_meas != _old_meas:
                changed = {k: (_old_meas.get(k), v)
                           for k, v in _new_meas.items() if _old_meas.get(k) != v}
                campaign.pack.objective["measurement"] = _new_meas
                for k, (was, now) in sorted(changed.items()):
                    print(f"[arbor] measurement.{k}: {was} -> {now} (pack reloaded)")
        except Exception as e:
            print(f"[arbor] pack reload failed ({e}) — keeping the loaded measurement block")

        if args.max_iters and iteration > args.max_iters:
            print(f"[arbor] Reached max iterations ({args.max_iters}) — exiting.")
            break

        elapsed_min = (time.time() - start_time) / 60.0
        if time_budget and elapsed_min >= time_budget:
            print(f"[arbor] Time budget exhausted ({elapsed_min:.1f}/{time_budget} min) — exiting.")
            break

        # A round end (arbor/rounds.py) runs its A/B before any iteration. Never in
        # component mode: the host gates and merges, and rounds.hook is None there.
        if not comp and rounds.pending(session_dir):
            outcome = _round_end(campaign, session_dir, state_path, roll)
            if outcome == "handoff":
                return roll.handoff(CampaignState.load(state_path))
            if outcome == "exit":
                break
            if outcome == "pending":
                time.sleep(args.cooldown)
            continue

        # The target a confirmed round-end A/B set, else the config's. Not in component
        # mode: what Arbor keeps is the host's to gate, and its deadline ends the run.
        target = rounds.target(session_dir, metric_name) or config_target
        if not comp and _target_reached(state, target, direction):
            if not rounds.hook(campaign):
                print(f"[arbor] Target reached: {_best_text(state, unit)} — exiting.")
                break
            # Not a stop (operator, 2026-10-21): the round ends, with a same-node A/B first.
            rounds.request(session_dir, state, "target",
                           f"target {target} {unit} reached: {_best_text(state, unit)}")
            state.save(state_path)
            continue

        # Start an iteration only if it can finish before the switch; else hand off.
        if roll and not roll.may_start_iteration(state):
            return roll.handoff(state)

        # The host's dossier pass (component.stop_at "map") ends once its map is set,
        # before any iteration works it: the host splits it among its Arbors.
        if comp and component.stops_at_map(campaign.config):
            mapped = component.map_ready(campaign, state, session_dir)
            if mapped:
                exit_reason, exit_detail = "mapped", mapped
                break

        # Component mode: likewise before the host's deadline; else exit 0, saying why
        # (`deadline`, or `map_exhausted` when the map's update round is what did not fit).
        if comp and not comp.may_start(state):
            exit_reason = comp.reason
            break

        print(f"[arbor] === Iteration {iteration} ===")
        print(f"[arbor] tier={state.tier} best={state.best_metric} "
              f"null_streak={state.null_streak}")

        # S5: recover any commit left orphaned by a prior kill between commit and state.save
        container = campaign.config.get("docker_container", "")
        workspace = campaign.config.get("workspace_root", "")
        if container and workspace:
            recover_orphaned_commit(container, workspace, state.iteration)

        t_iter = time.time()
        if comp:
            comp.begin(iteration, t_iter)
        try:
            result = run_iteration(config_path, arbor_root, str(session_dir), iteration)
            if comp:
                comp.end()
            print(f"[arbor] {result.get('decision')}: {result.get('change')} "
                  f"→ {result.get('metric')}")
            if result.get("decision") in ("STOPPED", "ROUND_END"):
                # The map stopped the campaign, or ended its round, before the agent
                # ran (arbor/directions.py): no iteration to time; the loop's STOP
                # check exits, or its round-end check runs the A/B.
                continue
        except KeyboardInterrupt:
            print("\n[arbor] Interrupted — exiting.")
            break
        except Exception as e:
            print(f"[arbor] Iteration failed: {e}")
            if (session_dir / "ABORTED.md").exists():
                print("[arbor] ABORTED.md — exiting.")
                exit_reason, exit_detail = _aborted_exit(session_dir)
                break

        if comp:
            comp.end()
        state = CampaignState.load(state_path)
        if roll:
            roll.after_iteration(iteration, t_iter, state)
        elif campaign.config.get("docker_container"):
            last = state.lessons[-1] if state.lessons else {}
            if last.get("iteration") == iteration and last.get("decision") == "KEEP":
                from arbor.cluster import handoff
                handoff.mirror_workspace_config(campaign.config, session_dir, os.environ.get(
                    "SLURM_JOB_ID") or os.environ.get("SPUR_JOB_ID") or "local")
        keeps = sum(1 for l in state.lessons if l.get("decision") == "KEEP")
        reverts = sum(1 for l in state.lessons if l.get("decision") == "REVERT")
        invalid = sum(1 for l in state.lessons if l.get("decision") == "INVALID")
        # INVALID is reported separately and on purpose. Folding not-measured
        # iterations into the REVERT count is what made a harness problem look
        # like a run of scientific failures.
        extra = f", {invalid} NOT MEASURED" if invalid else ""
        print(f"[arbor] Cumulative: {keeps} KEEPs, {reverts} REVERTs{extra}, "
              f"{_best_text(state, unit)}")

        print(f"[arbor] Cooling down {args.cooldown}s...")
        time.sleep(args.cooldown)

    rc = 0
    if comp:
        # Every component-mode exit says why (decision 1): `done` for STOP, max iterations
        # and an interrupt; `aborted` for ABORTED.md, a harness that stopped measuring, with
        # its message as the detail; `unreachable` (exit 3, so the host resumes the run) when
        # that harness was a model that could not be reached; the gate's reason when time ran
        # out; `mapped` when a dossier pass has set its map.
        rc = comp.finish(exit_reason, exit_detail)
    print("[arbor] Done.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
