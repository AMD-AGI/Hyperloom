"""The loop. One command: python3 -m arbor run --config <path>

Initializes if needed, then loops: policy picks a candidate, harness gates it.
State persists across iterations via session.json. Stop conditions: target,
time budget, STOP file, max iterations. With a round-end A/B hook (arbor/rounds.py)
the target is not a stop: it ends the round, like an explored map does.

In component mode (arbor/component.py, docs/component-mode.md) a host owns the
clock: an iteration starts only if it fits before the host's deadline, and every
exit writes component_exit.json with its reason before returning 0.

A session is a master and its workers. Each
pass is one side's (arbor/roles.py):

- the master's (arbor/master.py): the setup, the map's and the update round's
  passes, a round end, and the campaign's end;
- a worker's (arbor/worker.py): the passes in which the map is active. A worker's
  stretch starts with a work order and ends with a report, both in the session.

Three commands run them, over one loop (``_loop``):

- ``arbor run``: both sides, in this process; ``--split processes`` runs the two
  commands below in turn instead, each a process of its own; ``--split workers``
  runs the master, then several workers at once, each in a session and a slot of
  its own (arbor/dealer.py, plan 3);
- ``arbor master --config``: the master's passes, until a work order is ready or
  the campaign ends;
- ``arbor worker --order``: one worker's stretch, until its report is written.

Each of the two says how it ended in role_exit.json (and, under a host, in
component_exit.json: ``ordered`` or ``reported``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Force line-buffered stdout so tee/pipes see each line immediately.
# Without this, block-buffered stdout drops lines when the process is piped.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

from arbor import component, conflicts, lifecycle, master, review, roles, triggers, worker
from arbor.execute.workspace import recover_orphaned_commit
from arbor.master import (  # noqa: F401 - their old home, which callers and tests import from
    _aborted_detail, _aborted_exit, _best_text, _resolve_target, _root_accuracy, _round_end, _seed_baseline,
    _target_reached,
)
from arbor.pack import load_campaign  # noqa: F401 - callers and tests load campaigns through the driver
from arbor.pack.loader import PACK_RESULTS_CONCERNS, PACK_RESULTS_DIR, PACK_RESULTS_FLAG
from arbor.policy.claude_loop import KNOWLEDGE_SKILL, run_iteration
from arbor.state import CampaignState


def main(argv: list[str] | None = None, command=None) -> int:
    """The loop, and in component mode what ends it from outside.

    SIGTERM from the host (arbor/component.py installs the handler) unwinds the loop
    wherever it is: the iteration in flight is booked NOT MEASURED, our own server is
    stopped, the exit record written, and the driver exits 0 within the host's grace
    (MB: 30 s, then SIGKILL). Any other exception still crashes the driver, which the
    host resumes, but stops our server first.

    ``command`` is the side to run (``_master``, ``_worker``); ``arbor run`` (``_main``) by default.
    """
    held: dict = {}
    try:
        return (command or _main)(argv, held)
    except component.Terminated:
        if not held.get("comp"):
            raise
        return held["comp"].terminated()
    except BaseException:
        if held.get("comp"):
            held["comp"].abort()
        raise


def main_master(argv: list[str] | None = None) -> int:
    """``python3 -m arbor master``: the master's passes, until a work order is ready or the campaign ends."""
    return main(argv, _master)


def main_worker(argv: list[str] | None = None) -> int:
    """``python3 -m arbor worker --order``: one worker's stretch, until its report is written."""
    return main(argv, _worker)


DESCRIPTIONS = {
    "arbor run": "Run an Arbor campaign — one command, walk away.",
    "arbor master": "Run the master's side of a campaign: set it up, write the map, end its rounds; stop once a work "
                    "order is ready for a worker (role_exit.json says ordered) or the campaign ends (ended).",
    "arbor worker": "Run one worker's stretch from a work order: iterate on the map until the master's turn or the "
                    "run's end, then write the worker report (role_exit.json says reported).",
}


SPLITS, SPLIT_ENV = ("loop", "processes", "workers"), "ARBOR_SPLIT"


def _parse(argv: list[str] | None, prog: str = "arbor run") -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog=prog, description=DESCRIPTIONS[prog])
    if prog == "arbor run" and (os.environ.get(SPLIT_ENV) or "loop") not in SPLITS:
        parser.error(f"{SPLIT_ENV} is {os.environ[SPLIT_ENV]!r}, not one of {', '.join(SPLITS)}")
    if prog == "arbor worker":
        parser.add_argument("--order", required=True, help="the work order (work_order.json) the master wrote")
        parser.add_argument("--config", default=None,
                            help="the worker's own config (its session, workspace and node); the order's by default, "
                                 "in the master's session. A session of its own takes the order's directions as its "
                                 "map")
    else:
        parser.add_argument("--config", required=True, help="Campaign config JSON")
    if prog == "arbor run":
        parser.add_argument("--split", choices=SPLITS, default=os.environ.get(SPLIT_ENV) or "loop",
                            help="loop: the master and its workers in this process (the default); processes: "
                                 "each side as its own command, python3 -m arbor master, then worker --order, in turn "
                                 f"(also {SPLIT_ENV}=processes, as a host that launches arbor run passes it); workers: "
                                 "the master deals its map to several workers at once, one per slot of the config's "
                                 "workers block, and integrates what they kept (arbor/dealer.py)")
    else:
        # The run's first start, so a time budget bounds the whole run, not each command: --split processes passes
        # it to the master; a worker takes its order's.
        parser.add_argument("--started", type=float, default=None, help=argparse.SUPPRESS)
    if prog == "arbor master":
        parser.add_argument("--deal", action="store_true",
                            help="the master deals its map to workers of their own (--split workers passes it): it "
                                 "starts the set map before handing over, and ends the campaign when the time left is "
                                 "the integration's reserve")
    parser.add_argument("--max-iters", type=int, default=0,
                        help="0 = run until target or STOP file")
    parser.add_argument("--cooldown", type=int, default=30,
                        help="Seconds between iterations")
    parser.add_argument(PACK_RESULTS_FLAG, action="store_true",
                        help=f"accept a {PACK_RESULTS_DIR}/ directory inside the pack (refused by "
                             f"default: {PACK_RESULTS_CONCERNS})")
    if prog != "arbor worker":       # a worker runs a campaign its master has had reviewed
        gate = parser.add_mutually_exclusive_group()
        gate.add_argument("--review-gate", dest="review_gate", action="store_true", default=None,
                          help="a fresh session waits for a human's approval before it runs "
                               "anything (the default; arbor/review.py)")
        gate.add_argument("--no-review-gate", dest="review_gate", action="store_false",
                          help=f"start without the review (also {review.ENV}=0, or "
                               f"\"review_gate\": false in the config)")
    parser.add_argument("--skill-conflicts", choices=conflicts.POLICIES, default=None,
                        help="two loaded skills that write the same thing: pause for a person to decide (the "
                             "default), or merge them with an agent, without a pause (also "
                             f"{conflicts.ENV}, or \"skill_conflicts\" in the config; arbor/conflicts.py)")
    args = parser.parse_args(argv)
    return args


BOTH = (lifecycle.MASTER, lifecycle.WORKER)


def _loop(s: master.Session, stretch: worker.Stretch, sides: tuple[str, ...] = BOTH) -> master.Step:
    """Run passes while each is one of ``sides``'s (``roles.role_of``), and say why it stopped: ``exit`` (the run
    ends), ``return`` (``rc`` at once: a rolling chunk's handoff), or ``handover`` (the next pass is the other side's;
    its ``state`` the session's)."""
    while True:
        step = master.before_pass(s)
        if step.kind == "round_end":
            if lifecycle.MASTER not in sides:
                return master.Step("handover", step.iteration, step.state)
            stretch.end(*stretch.handover(step.state))   # a round's end is the master's
            step = master.round_end(s, step.iteration)
        if step.kind == "exit":
            return step
        if step.kind == "return":
            stretch.end("handoff", "the rolling chunk hands the session to the next")
            return step
        if step.kind == "again":
            continue
        side = roles.role_of(s.campaign, step.state, s.session_dir)
        if side not in sides or (side == lifecycle.WORKER and stretch.spend(step.state)):
            return master.Step("handover", step.iteration, step.state)
        stretch.switch(side, step.state)
        if _run_pass(s, step.iteration, step.state) == "exit":
            return master.Step("exit")


def _main(argv: list[str] | None, held: dict) -> int:
    """``arbor run``: the master and its workers, in this process (or, with ``--split processes``, as commands)."""
    args = _parse(argv)
    if args.split == "processes":
        return _split(argv, args)
    if args.split == "workers":
        from arbor import dealer
        return dealer.run(argv, args)
    s = master.prepare(args, held)
    if not isinstance(s, master.Session):
        return s
    stretch = worker.Stretch(s)
    step = _loop(s, stretch)
    if step.kind == "return":
        return step.rc
    stretch.end(s.exit_reason, s.exit_detail)
    return master.finish(s)


def _master(argv: list[str] | None, held: dict) -> int:
    """``arbor master``: the master's passes, until the next is a worker's (its work order written, ``ordered``) or
    the campaign ends (``ended``). An order still waiting for its worker's report is the worker's turn: nothing runs."""
    args = _parse(argv, "arbor master")
    s = master.prepare(args, held)
    if not isinstance(s, master.Session):
        return s
    waiting = roles.open_order(s.session_dir)
    if waiting is not None:
        print(f"[arbor] Master: the work order from iteration {waiting['start_iteration']} waits for its worker")
        return master.hand_over(s, roles.ORDERED, s.session_dir / roles.ORDER_FILE)
    stretch = worker.Stretch(s)
    step = _loop(s, stretch, (lifecycle.MASTER,))
    while args.deal and step.kind == "handover":
        nxt = _dealt(s, stretch, step)
        if nxt is None:
            break
        step = nxt
    if step.kind == "return":
        return step.rc
    if step.kind == "handover":
        stretch.start(step.state)
        return master.hand_over(s, roles.ORDERED, s.session_dir / roles.ORDER_FILE)
    roles.write_exit(s.session_dir, lifecycle.MASTER, roles.ENDED, None, s.exit_reason, s.exit_detail)
    return master.finish(s)


def _dealt(s: master.Session, stretch: worker.Stretch, step: master.Step) -> master.Step | None:
    """``arbor master --deal`` at a hand-over (plan 3): a round that starts only if the integration's reserve is left
    after it; the set map started here, since no pass runs in this session while its workers work. None when the map
    is active and the order can go; an ``exit`` when the campaign ends; or the master's next step when the map's start
    left the map the master's (a fresh start)."""
    from arbor import directions, slots

    left, reserve = s.time_left(), slots.reserve_s(s.campaign.config)
    if left is not None and left < reserve:
        s.exit_reason = "done"
        s.exit_detail = f"{left / 60:.0f} min left, under the integration's reserve ({reserve / 60:.0f} min): no round"
        print(f"[arbor] Master: {s.exit_detail}")
        return master.Step("exit")
    doc = directions.current(s.campaign, step.state)
    if not doc or not (doc.get("pending") or doc.get("phase") == "active"):
        s.exit_reason, s.exit_detail = "aborted", "--deal: a master deals its map, and this campaign has none"
        print(f"[arbor] Master: {s.exit_detail}", file=sys.stderr)
        return master.Step("exit")
    started = directions.activate(s.campaign, step.state)
    step.state.save(s.state_path)
    if started:
        s.map_set()
        return None
    if (directions.current(s.campaign, step.state) or {}).get("pending"):
        s.exit_reason, s.exit_detail = "aborted", "--deal: the set map could not be started"
        return master.Step("exit")
    return _loop(s, stretch, (lifecycle.MASTER,))


def _worker(argv: list[str] | None, held: dict) -> int:
    """``arbor worker --order``: one worker's stretch, from the order, until the next pass is the master's or the run
    ends; then its report (``reported``). The master reviewed the campaign, so a worker never waits for the review.

    In the master's session by default. With ``--config`` naming a session of its own, the worker's session takes the
    order's directions as its map, the master's dossier when it has none, and the order, renumbered from its own next
    iteration (``roles.localize``); its report stays there, for the caller."""
    args = _parse(argv, "arbor worker")
    try:
        order = roles.read_order(args.order)
        own = args.config is not None
        args.config, args.review_gate = args.config or order["campaign"]["config"], False
        session_dir = master.session_of(args.config)
    except (OSError, ValueError) as exc:
        print(f"[arbor] Worker: no work order to run: {exc}", file=sys.stderr)
        return 2
    apart = os.path.realpath(session_dir) != os.path.realpath(order["session"])
    if apart and not own:
        print(f"[arbor] Worker: the order's session is {order['session']}, and its config runs in {session_dir}; "
              f"a worker runs in its master's session, or in its own with --config", file=sys.stderr)
        return 2
    if apart and not order["map"].get("directions"):
        print("[arbor] Worker: a worker of its own session works the order's directions, and this order lists none "
              "(map.directions, roles interface 2)", file=sys.stderr)
        return 2
    if apart:
        raw = json.loads(Path(args.config).read_text(encoding="utf-8"))
        if component.block(raw).get("plan") is not None:
            return component.refuse(["a worker's map is its work order's: its config gives no component.plan"],
                                    raw, session_dir)
    if args.started is None:
        args.started = worker.epoch((order.get("limits") or {}).get("started"))
    s = master.prepare(args, held)
    if not isinstance(s, master.Session):
        return s
    if apart:
        local = roles.load(s.session_dir, roles.ORDER_FILE)
        if not local or local.get("id") != order["id"]:          # a resume keeps the order it started with
            local = roles.localize(order, s.session_dir, CampaignState.load(s.state_path))
            roles.write_order(s.session_dir, local)
            from arbor import chapters
            for said in (roles.seed_map(s.session_dir, local),
                         roles.take_dossier(local, chapters.places(s.campaign, s.session_dir, s.arbor_root)["dossier"])):
                if said:
                    print(f"[arbor] Worker: {said}")
        order = local
    if roles.reported(s.session_dir, order):
        print(f"[arbor] Worker: the order from iteration {order['start_iteration']} is reported already")
        return master.hand_over(s, roles.REPORTED, s.session_dir / roles.REPORT_FILE)
    stretch = worker.Stretch(s, spends=apart)
    stretch.resume(order)
    step = _loop(s, stretch, (lifecycle.WORKER,))
    if step.kind == "return":
        return step.rc
    ended, detail = stretch.handover(step.state) if step.kind == "handover" else (s.exit_reason, s.exit_detail)
    stretch.end(ended, detail)
    return master.hand_over(s, roles.REPORTED, s.session_dir / roles.REPORT_FILE,
                            reason=None if step.kind == "handover" else s.exit_reason, detail=detail)


def _without(argv: list[str], flag: str, value: bool = True) -> list[str]:
    """``argv`` less ``flag`` (``--flag X`` or ``--flag=X`` when it takes a ``value``)."""
    out, skip = [], False
    for arg in argv:
        if skip:
            skip = False
        elif arg == flag:
            skip = value
        elif not arg.startswith(flag + "="):
            out.append(arg)
    return out


_stopping: list[int] = []      # the signals --split processes has forwarded: no command starts after one


def _call(argv: list[str]) -> int:
    """Run one side's command and wait for it. SIGTERM goes on to it, so a host that stops ``arbor run`` stops the
    side that holds the cards, which exits as it would have (component mode: within the host's grace)."""
    import signal
    import subprocess

    child = subprocess.Popen(argv)

    def forward(signum, _frame) -> None:
        _stopping.append(signum)
        child.send_signal(signum)

    previous = signal.signal(signal.SIGTERM, forward)
    try:
        return child.wait()
    finally:
        signal.signal(signal.SIGTERM, previous)


def _split(argv: list[str] | None, args: argparse.Namespace) -> int:
    """``arbor run --split processes``: ``arbor master``, then ``arbor worker --order``, in turn, each a process of its
    own (``_call``), until the master ends the campaign or a SIGTERM stops the run; its exit code. Each says how it
    ended in ``role_exit.json``."""
    try:
        session_dir = master.session_of(args.config)
    except Exception as exc:  # noqa: BLE001 - the master refuses it, saying why
        print(f"[arbor] Run: {exc}", file=sys.stderr)
        session_dir = None
    rest = _without(list(argv or []), "--split")
    for_worker = _without(_without(_without(rest, "--config"), "--review-gate", False), "--no-review-gate", False)
    rest += ["--started", repr(time.time())]
    while True:
        if session_dir:      # a command that ends before it says how leaves no answer from an earlier one
            (session_dir / roles.EXIT_FILE).unlink(missing_ok=True)
        rc = _call([sys.executable, "-m", "arbor", "master", *rest])
        done = (roles.load(session_dir, roles.EXIT_FILE) or {}) if session_dir else {}
        if rc != 0 or _stopping or done.get("outcome") != roles.ORDERED:
            return rc
        print(f"[arbor] Run: the master ordered a worker ({done['path']})")
        (session_dir / roles.EXIT_FILE).unlink(missing_ok=True)
        rc = _call([sys.executable, "-m", "arbor", "worker", "--order", str(done["path"]), *for_worker])
        if rc != 0 or _stopping:
            return rc


# Passes running that may fail the same way before the run ends: a failure that repeats unchanged is the harness's or
# the session's, which another attempt does not mend (one run repeated an identical failure well over a hundred times).
SAME_FAILURE_LIMIT = 5


def _same_failure(s: master.Session, iteration: int, error: Exception) -> None:
    """Count a failed pass; the first of its kind prints its traceback, and the SAME_FAILURE_LIMIT-th in a row
    writes ABORTED.md, which ends the run (`aborted`, or `unreachable` when the model could not be reached)."""
    import traceback

    from arbor.policy import claude_loop

    text = f"{type(error).__name__}: {error}"[:500]
    f = s.failures
    f["count"] = f["count"] + 1 if f["last"] == text else 1
    f["last"] = text
    trace = traceback.format_exc()
    if f["count"] == 1:
        print(trace, end="", file=sys.stderr, flush=True)
    if f["count"] < SAME_FAILURE_LIMIT:
        print(f"[arbor] this failure {f['count']} time(s) running; the run ends at {SAME_FAILURE_LIMIT}")
        return
    claude_loop._write_aborted(s.session_dir, (
        f"Iteration {iteration} failed the same way {f['count']} times running, so another attempt would too: "
        f"{text}\n\nThe last traceback:\n\n```\n{trace}```"))
    print(f"[arbor] the same failure {f['count']} times running: ABORTED.md written")


def _run_pass(s: master.Session, iteration: int, state: CampaignState) -> str:
    """One pass, the master's or a worker's: an iteration (arbor/policy/claude_loop.py) and its bookkeeping.

    "next" after its cooldown; "again" when the map stopped the campaign or ended its round before the agent ran;
    "exit" on an interrupt or an abort."""
    campaign, comp, session_dir, roll = s.campaign, s.comp, s.session_dir, s.roll
    print(f"[arbor] === Iteration {iteration} ===")
    print(f"[arbor] tier={state.tier} best={state.best_metric} "
          f"null_streak={state.null_streak}")
    # The agents' shells inherit these: arbor.knowledge logs and budgets each call per iteration (arbor/knowledge/),
    # and arbor.dossier.check holds a dossier to its knowledge citations when the pack loads knowledge:search.
    # Names of their own: ARBOR_SESSION_DIR already tells evals/arbor_trace.py where to write its traces.
    plan = lifecycle.declared(campaign)
    os.environ.update(ARBOR_KNOWLEDGE_SESSION=str(session_dir), ARBOR_KNOWLEDGE_ITERATION=str(iteration),
                      ARBOR_KNOWLEDGE_CONFIG=str(s.config_path),
                      ARBOR_KNOWLEDGE_LOADED="1" if plan is not None and KNOWLEDGE_SKILL in plan.load
                      else "0")

    # S5: recover any commit left orphaned by a prior kill between commit and state.save
    container = campaign.config.get("docker_container", "")
    workspace = campaign.config.get("workspace_root", "")
    if container and workspace:
        recover_orphaned_commit(container, workspace, state.iteration)

    t_iter = time.time()
    if comp:
        comp.begin(iteration, t_iter)
    s.fire("iteration_start", iteration)
    try:
        result = run_iteration(s.config_path, s.arbor_root, str(session_dir), iteration)
        s.failures.update(last="", count=0)
        if comp:
            comp.end()
        print(f"[arbor] {result.get('decision')}: {result.get('change')} "
              f"→ {result.get('metric')}")
        decided = str(result.get("decision") or "").lower()
        if decided in triggers.FIRED:
            s.fire(decided, iteration, change=result.get("change"), metric=result.get("metric"))
        if decided == "invalid" and str(result.get("reason") or "").startswith("agent did not run"):
            s.fire("agent_failed", iteration, reason=result.get("reason"))
        if result.get("decision") in ("STOPPED", "ROUND_END"):
            # The map stopped the campaign, or ended its round, before the agent
            # ran (arbor/directions.py): no iteration to time; the loop's STOP
            # check exits, or its round-end check runs the A/B.
            return "again"
    except KeyboardInterrupt:
        print("\n[arbor] Interrupted — exiting.")
        return "exit"
    except Exception as e:
        print(f"[arbor] Iteration failed: {e}")
        _same_failure(s, iteration, e)
        if (session_dir / "ABORTED.md").exists():
            print("[arbor] ABORTED.md — exiting.")
            s.exit_reason, s.exit_detail = _aborted_exit(session_dir)
            return "exit"

    if comp:
        comp.end()
    state = CampaignState.load(s.state_path)
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
          f"{_best_text(state, s.unit)}")

    print(f"[arbor] Cooling down {s.args.cooldown}s...")
    time.sleep(s.args.cooldown)
    return "next"


if __name__ == "__main__":
    sys.exit(main())
