"""The master's side of a session: the setup, what each pass must clear, a round's end, and the campaign's end.

A session is a master and its workers (``arbor/roles.py``). The master:

- **sets the campaign up** (``prepare``): the mode's refusals, the roll lease, the review gate, the skill
  conflicts, ``init``, the recorded baseline, the host's run and the root's accuracy;
- **clears each pass** (``before_pass``): STOP, an abort, the iteration and time limits, a pending round end, the
  target, a rolling chunk's switch, a host's dossier pass whose map is set, and the host's deadline;
- **ends a round** (``round_end``): its same-node A/B, booked (``arbor/rounds.py``);
- **ends the campaign** (``finish``): the ``end`` event and, under a host, the exit record.

The map's and the update round's own passes are iterations like a worker's: ``arbor/driver.py`` runs each pass under
the role ``roles.role_of`` gives it. This code moved from ``driver.py`` as it was.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from arbor import component, conflicts, directions, lifecycle, retraction, review, roles, rounds, triggers
from arbor.cli.iterate import _find_arbor_root, cmd_init
from arbor.pack import load_campaign, validate_campaign
from arbor.pack.loader import PACK_RESULTS_CONCERNS, PACK_RESULTS_DIR, PACK_RESULTS_FLAG
from arbor.policy.claude_loop import KNOWLEDGE_SKILL, _run_claude
from arbor.rebaseline import current as rebase_scale
from arbor.state import CampaignState

def _resolve_target(target, metric: str):
    """Resolve a campaign `target` to a number, or None if it carries none.

    A config may express `target` either as a bare number or as a dict of
    documented reference points — e.g. a campaign may record
    ``{"_source": ..., "output_tok_per_chip": <number>, "gap_pct": <number>}``.
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
    read as a normal end under a host (in a gateway outage).
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


# ── the session the loop carries ──────────────────────────────────────────────

@dataclass
class Session:
    """One run of a session, as ``prepare`` sets it up and the loop carries it."""

    args: argparse.Namespace
    held: dict
    config_path: str
    arbor_root: str
    campaign: object
    session_dir: Path
    state_path: Path
    roll: object
    comp: object
    start_time: float
    metric_name: str
    config_target: float | None
    direction: str
    time_budget: float | None
    unit: str
    exit_reason: str = "done"
    exit_detail: str = ""
    fired_map: dict = field(default_factory=lambda: {"round": None})
    # The last pass's failure, and how many passes running failed that same way (driver._same_failure).
    failures: dict = field(default_factory=lambda: {"last": "", "count": 0})

    def fire(self, event: str, at: int | None = None, **detail) -> None:
        """The lifecycle's events (arbor/triggers.py): journaled for a host, the pack's trigger skills started and the
        host's callbacks called, none of them holding the loop up but a blocking skill, which the time left bounds."""
        triggers.fire(self.campaign, self.session_dir, self.arbor_root, self.config_path, event, at,
                      budget_s=self.time_left(), **detail)

    def time_left(self) -> float | None:
        """Seconds before the run must stop: a host's deadline less its margin, or the time budget's end; None
        without either."""
        ends = []
        if self.comp and getattr(self.comp, "deadline", None):
            ends.append(self.comp.deadline - component.DEADLINE_MARGIN_S)
        if self.time_budget:
            ends.append(self.start_time + self.time_budget * 60)
        return min(ends) - time.time() if ends else None

    def map_set(self) -> None:
        """``map_set``, once per map round in the session: a round already announced, in the journal, is not again, by
        this process or by the next command of a split run."""
        doc, _ = directions.load(self.session_dir)
        if self.fired_map["round"] is None:
            self.fired_map["round"] = (lifecycle.last(self.session_dir, "event", "map_set") or {}).get("map_round")
        if doc and doc.get("phase") == "active" and doc.get("map_round") != self.fired_map["round"]:
            self.fired_map["round"] = doc.get("map_round")
            self.fire("map_set", map_round=doc.get("map_round"),
                      open=[d.get("id") for d in doc.get("directions") or [] if d.get("status") == "open"])

    def target(self):
        """The target a confirmed round-end A/B set, else the config's."""
        return rounds.target(self.session_dir, self.metric_name) or self.config_target

    def limits(self) -> dict:
        """What bounds the run, for a work order (arbor/roles.py)."""
        from arbor.worker import iso
        return {"max_iters": self.args.max_iters or None, "time_budget_minutes": self.time_budget,
                "started": iso(self.start_time),
                "deadline": iso(getattr(self.comp, "deadline", None)) if self.comp else None}


def sync_knowledge(campaign, arbor_root) -> bool:
    """Knowledge search (arbor/knowledge/): a pack that loads its skill gets the campaign's sources synced at their
    pins before anything runs, into ARBOR_KNOWLEDGE_CACHE (the user's cache by default). It is small, and a pin already
    synced is skipped. A source that cannot be synced is logged, and the verbs say they did not search it; nothing
    here stops the campaign. True when the pack loads the skill."""
    plan = lifecycle.declared(campaign)
    if plan is None or KNOWLEDGE_SKILL not in plan.load:
        return False
    try:
        from arbor.knowledge import cache as knowledge_cache
        knowledge_cache.sync_campaign(campaign.config, Path(arbor_root))
    except Exception as e:  # noqa: BLE001 - search is an aid; the campaign runs without it
        print(f"[arbor] knowledge: not synced ({type(e).__name__}: {str(e)[:200]})")
    return True


def prepare(args: argparse.Namespace, held: dict) -> Session | int:
    """The setup before the loop; an exit code when the run ends before it (a refusal, the review, a conflict)."""
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
    # host_launch.json: Arbor's base launch beside the host's record, refused or not, for the
    # host to read (arbor/base_launch.py). Only when the host names a record.
    if component.is_component(campaign) and session_dir:
        from arbor import base_launch
        try:
            base_launch.write(campaign.config, session_dir)
        except OSError as exc:
            print(f"[arbor] could not write {base_launch.HOST_LAUNCH_FILE}: {exc}", file=sys.stderr, flush=True)
    if refused:
        return component.refuse(refused, campaign.config, session_dir)

    # Rolling chunks (arbor/cluster/handoff.py): the
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
        # mirrored below, or the first iteration in a fresh container runs without them.
        from arbor.cluster import handoff
        handoff.restore_workspace_config(campaign.config, session_dir)

    # The review gate (arbor/review.py): a fresh session waits here for a human's approval,
    # before cmd_init pins the validators or seeds the workspace and before any agent, server
    # or bench runs. Approved, the campaign is loaded again, so the reviewed files are the
    # ones that run. In component mode the host's clock, heartbeat and SIGTERM cover the wait.
    state_path = session_dir / "session.json"
    # Skill conflicts are recorded first, so the review document names them and a reviewer can
    # decide them in the same pause (arbor/conflicts.py); they are settled after the review.
    conflicts.check(campaign, arbor_root, session_dir)
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

    # Skill conflicts (arbor/conflicts.py): two loaded skills that write the same thing, or say
    # they overlap. Standalone, a person decides each (one side, or an agent's merge); auto-merge
    # lets an agent merge them without a pause; under a host, nothing waits for a person. Under a
    # host, the run starts before a merge's agent turn, as before the review gate's wait, so the
    # host's clock, heartbeat and SIGTERM cover it.
    if (component.is_component(campaign) and held.get("comp") is None
            and conflicts.check(campaign, arbor_root, session_dir)):
        held["comp"] = component.Run(campaign.config, session_dir)
        held["comp"].start(campaign.config.get("time_budget_minutes"))
    outcome, reasons = conflicts.gate(
        campaign, arbor_root, session_dir, args.skill_conflicts, hosted=component.is_component(campaign),
        run_agent=lambda prompt, label: asyncio.run(_run_claude(
            prompt, arbor_root, allowed_tools=["Bash", "Read", "Edit"], label=label, skills=[], plugins=[])),
        deadline=held["comp"].deadline if held.get("comp") else None, margin=component.DEADLINE_MARGIN_S)
    if outcome == "refuse":
        return component.refuse(reasons, campaign.config, session_dir)
    if outcome != "go":
        print(f"[arbor] Skill conflicts: {outcome} while a person was deciding — exiting before anything ran.")
        comp = held.get("comp")
        reason = "done" if outcome == "stop" else "deadline"
        return comp.finish(reason, f"{outcome} while skill conflicts waited") if comp else 0

    sync_knowledge(campaign, arbor_root)

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

    # A run split into commands (arbor run --split processes, a worker's order) keeps its first start, so a time
    # budget bounds the whole run, not each command.
    start_time = getattr(args, "started", None) or time.time()
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

    s = Session(args=args, held=held, config_path=config_path, arbor_root=arbor_root, campaign=campaign,
                session_dir=session_dir, state_path=state_path, roll=roll, comp=comp, start_time=start_time,
                metric_name=metric_name, config_target=config_target, direction=direction,
                time_budget=time_budget, unit=unit, exit_reason=exit_reason, exit_detail=exit_detail)
    s.fire("session_start", mode=campaign.mode)
    return s


# ── what each pass must clear ─────────────────────────────────────────────────

@dataclass
class Step:
    """What ``before_pass`` decided: ``run`` the pass at ``iteration``; a pending ``round_end``; loop ``again``;
    ``exit`` the loop; or ``return`` ``rc`` at once (a rolling chunk's handoff). The driver's loop adds ``handover``:
    the next pass is the other side's, and this command stops before it (``arbor master``, ``arbor worker``)."""

    kind: str
    iteration: int = 0
    state: CampaignState | None = None
    rc: int = 0


def _reload_measurement(s: Session) -> None:
    # RELOAD THE MEASUREMENT BLOCK EACH ITERATION.
    #
    # `state` is re-read from disk every iteration, so the BAR can be
    # corrected between iterations (a campaign's bar-fixing script does
    # exactly that). The BANDS could not: load_campaign() runs once before
    # this loop, and gate.py reads threshold_rel / noise_band_pct /
    # secondary_noise_band_pct off that in-memory pack. Editing pack.yaml
    # mid-run did nothing, and the only way to widen a band was to resubmit
    # the job -- which, since the driver IS the batch script, costs the
    # allocation and a container rebuild.
    #
    # That asymmetry mattered. A band narrower than the measurement noise
    # silently rejects real work, and the noise is routinely underestimated:
    # a spread from a pair of runs on one node can be several times smaller
    # than what re-measuring a byte-identical tree on its own node shows.
    # Without the refresh, a widened band takes effect only when the next
    # campaign starts, after every iteration under a bar it cannot reach.
    #
    # Only the measurement block is refreshed. Validators, protected paths
    # and tools stay pinned to what the campaign started with, so a
    # mid-run pack edit cannot change what is measured or what is allowed
    # to change -- only the thresholds that interpret the result.
    try:
        _fresh = load_campaign(s.config_path, s.arbor_root)
        _new_meas = _fresh.pack.objective.get("measurement", {})
        _old_meas = s.campaign.pack.objective.get("measurement", {})
        if _new_meas and _new_meas != _old_meas:
            changed = {k: (_old_meas.get(k), v)
                       for k, v in _new_meas.items() if _old_meas.get(k) != v}
            s.campaign.pack.objective["measurement"] = _new_meas
            for k, (was, now) in sorted(changed.items()):
                print(f"[arbor] measurement.{k}: {was} -> {now} (pack reloaded)")
    except Exception as e:
        print(f"[arbor] pack reload failed ({e}) — keeping the loaded measurement block")


def before_pass(s: Session) -> Step:
    """What the next pass must clear, in the order the loop has always checked it."""
    s.map_set()
    if (s.session_dir / "STOP").exists():
        print("[arbor] STOP file — exiting.")
        return Step("exit")

    # Checked here, not only inside the pass: a run of not-measured iterations writes
    # ABORTED.md and returns normally, so without this the loop would keep retrying a
    # broken harness until the allocation ran out.
    if (s.session_dir / "ABORTED.md").exists():
        print("[arbor] ABORTED.md present — exiting.")
        print((s.session_dir / "ABORTED.md").read_text())
        s.exit_reason, s.exit_detail = _aborted_exit(s.session_dir)
        return Step("exit")

    # A KEEP the agent found invalid that could not be taken back (arbor/retraction.py):
    # the run ends, exit 0, saying so, with the finding as the detail. A host retires the
    # branch rather than import a tree that holds the invalid change.
    conflict = s.session_dir / retraction.CONFLICT_FILE
    if conflict.exists():
        detail = (conflict.read_text().splitlines() or [""])[0]
        print(f"[arbor] {retraction.CONFLICT_FILE} present — exiting: {detail}")
        s.exit_reason, s.exit_detail = retraction.CONFLICT_REASON, detail
        return Step("exit")

    state = CampaignState.load(s.state_path)
    iteration = state.iteration + 1
    _reload_measurement(s)

    if s.args.max_iters and iteration > s.args.max_iters:
        print(f"[arbor] Reached max iterations ({s.args.max_iters}) — exiting.")
        return Step("exit")

    elapsed_min = (time.time() - s.start_time) / 60.0
    if s.time_budget and elapsed_min >= s.time_budget:
        print(f"[arbor] Time budget exhausted ({elapsed_min:.1f}/{s.time_budget} min) — exiting.")
        return Step("exit")

    # A round end (arbor/rounds.py) runs its A/B before any iteration. Never in
    # component mode: the host gates and merges, and rounds.hook is None there.
    if not s.comp and rounds.pending(s.session_dir):
        return Step("round_end", iteration, state)

    # The target a confirmed round-end A/B set, else the config's. Not in component
    # mode: what Arbor keeps is the host's to gate, and its deadline ends the run.
    target = s.target()
    if not s.comp and _target_reached(state, target, s.direction):
        if not rounds.hook(s.campaign):
            print(f"[arbor] Target reached: {_best_text(state, s.unit)} — exiting.")
            return Step("exit")
        # Not a stop: the round ends, with a same-node A/B first.
        rounds.request(s.session_dir, state, "target",
                       f"target {target} {s.unit} reached: {_best_text(state, s.unit)}")
        state.save(s.state_path)
        return Step("again")

    # Start an iteration only if it can finish before the switch; else hand off.
    if s.roll and not s.roll.may_start_iteration(state):
        return Step("return", rc=s.roll.handoff(state))

    # The host's dossier pass (component.stop_at "map") ends once its map is set,
    # before any iteration works it: the host splits it among its Arbors.
    if s.comp and component.stops_at_map(s.campaign.config):
        mapped = component.map_ready(s.campaign, state, s.session_dir)
        if mapped:
            s.exit_reason, s.exit_detail = "mapped", mapped
            return Step("exit")

    # Component mode: likewise before the host's deadline; else exit 0, saying why
    # (`deadline`, or `map_exhausted` when the map's update round is what did not fit).
    if s.comp and not s.comp.may_start(state):
        s.exit_reason = s.comp.reason
        return Step("exit")
    return Step("run", iteration, state)


def round_end(s: Session, iteration: int) -> Step:
    """The pending round end's A/B, booked (``_round_end``), and what the loop does next."""
    outcome = _round_end(s.campaign, s.session_dir, s.state_path, s.roll)
    if outcome != "pending":
        s.fire("round_end", iteration, outcome=outcome)
    if outcome == "handoff":
        return Step("return", rc=s.roll.handoff(CampaignState.load(s.state_path)))
    if outcome == "exit":
        return Step("exit")
    if outcome == "pending":
        time.sleep(s.args.cooldown)
    return Step("again")


def session_of(config_path: str) -> Path:
    """The session a config runs in, without setting anything up."""
    config_path = str(Path(config_path).resolve())
    return load_campaign(config_path, str(_find_arbor_root(Path(config_path)))).session_root


def hand_over(s: Session, outcome: str, path, reason: str | None = None, detail: str = "") -> int:
    """A side's command ends (``arbor master``, ``arbor worker``): ``role_exit.json`` says how and names the record it
    wrote; under a host, so does the exit record (``ordered`` or ``reported``, or the run's own reason when the run
    ended). The campaign's ``end`` is the master's (``finish``), so a hand-over fires no event."""
    side = lifecycle.MASTER if outcome == roles.ORDERED else lifecycle.WORKER
    detail = detail or (f"{roles.ORDER_FILE} for a worker" if outcome == roles.ORDERED else
                        f"{roles.REPORT_FILE} for the master")
    roles.write_exit(s.session_dir, side, outcome, path, reason, detail)
    print(f"[arbor] {side.capitalize()}: {outcome}" + (f" ({reason})" if reason else "") + f": {path}")
    if s.comp:
        return s.comp.finish(reason or outcome, detail)
    return 0


def finish(s: Session) -> int:
    """The campaign's end: the ``end`` event and, under a host, the exit record; the exit code."""
    s.map_set()
    s.fire("end", CampaignState.load(s.state_path).iteration, reason=s.exit_reason, detail=s.exit_detail)
    rc = 0
    if s.comp:
        # Every component-mode exit says why (decision 1): `done` for STOP, max iterations
        # and an interrupt; `aborted` for ABORTED.md, a harness that stopped measuring, with
        # its message as the detail; `unreachable` (exit 3, so the host resumes the run) when
        # that harness was a model that could not be reached; the gate's reason when time ran
        # out; `mapped` when a dossier pass has set its map.
        rc = s.comp.finish(s.exit_reason, s.exit_detail)
    print("[arbor] Done.")
    return rc
