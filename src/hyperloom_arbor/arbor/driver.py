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

from arbor import component, conflicts, lifecycle, master, objective, review, revisit, roles, triggers, worker
from arbor.execute import kept
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

    Under WSL it first warns, in one line: its clock can step back, and git can then miss an edit
    (arbor/execute/preflight.py, ``warn_wsl``).
    """
    from arbor.execute.preflight import warn_wsl
    warn_wsl()
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
    finally:
        # The session lock (arbor/checkpoints.py), let go when the command ends: in its own process the exit would,
        # and a caller that runs commands in its process (the tests, a host's in-process run) starts the next.
        from arbor import checkpoints
        checkpoints.release_session(held)


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


def parser(prog: str = "arbor run") -> argparse.ArgumentParser:
    """The command's parser, built without parsing: :func:`_parse` parses with it, and the payload's argv and schema
    are made from it (arbor/cli/payload.py)."""
    parser = argparse.ArgumentParser(prog=prog, description=DESCRIPTIONS[prog])
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
        parser.add_argument("--exit-at-handover", action="store_true",
                            help="exit at each hand-over, as arbor master and arbor worker would in turn: the master's "
                                 "passes, then one worker stretch, then at the next master's turn the worker's report "
                                 "(worker_report.json, role_exit.json `reported`) and exit 0. The same command "
                                 "continues: an order still waiting for its report is worked first, else the master's "
                                 "passes run again. For a host that deals each report (with the loop split only)")
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
    parser.add_argument("--opt-objective", default=None, metavar="TEXT",
                        help="optimize this metric in place of the pack's: a name (\"P50 interactivity\", \"total "
                             "throughput\") or a bench key with its direction (\"tpot_p50:min\"). Also "
                             f"{objective.ENV}, or \"{objective.CONFIG_KEY}\" in the config; arbor/objective.py")
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
    return parser


def _parse(argv: list[str] | None, prog: str = "arbor run") -> argparse.Namespace:
    p = parser(prog)
    if prog == "arbor run" and (os.environ.get(SPLIT_ENV) or "loop") not in SPLITS:
        p.error(f"{SPLIT_ENV} is {os.environ[SPLIT_ENV]!r}, not one of {', '.join(SPLITS)}")
    args = p.parse_args(argv)
    # Every load of the campaign in this run, and the agents' arbor commands, read it from the environment.
    if args.opt_objective:
        os.environ[objective.ENV] = args.opt_objective
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
            _checkpoint(s)
        if step.kind == "exit":
            if s.exit_reason == "paused":
                _pause(s)
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
        ran = _run_pass(s, step.iteration, step.state)
        _checkpoint(s)
        if ran == "exit":
            return master.Step("exit")


def _checkpoint(s: master.Session) -> None:
    """The session's checkpoint at this pass boundary (arbor/checkpoints.py); never fails the run."""
    from arbor import checkpoints

    try:
        iteration = CampaignState.load(s.state_path).iteration
    except (OSError, ValueError) as exc:
        print(f"[arbor] checkpoint not written: session.json unreadable ({exc})", file=sys.stderr, flush=True)
        return
    checkpoints.write_safely(s.campaign.config, s.session_dir, iteration, s.campaign, s.arbor_root)


def _pause(s: master.Session) -> None:
    """``arbor pause`` at this boundary: the server stopped, the tree checked committed, the checkpoint pause-NNNN
    written, ``paused.json`` recorded (arbor/checkpoints.py)."""
    from arbor import checkpoints
    from arbor.policy import claude_loop

    try:
        iteration = CampaignState.load(s.state_path).iteration
    except (OSError, ValueError):
        iteration = 0
    done = checkpoints.pause_now(
        s.campaign.config, s.session_dir, iteration, s.campaign, s.arbor_root,
        stop=None if s.comp else (lambda cfg: claude_loop._stop_server(cfg, cfg.get("docker_container", ""),
                                                                       cfg.get("gpu_ids"))))
    s.exit_detail = (f"after iteration {iteration}, checkpoint {done.get('checkpoint') or 'not written'}"
                     + (f" ({done['checkpoint_error']})" if done.get("checkpoint_error") else "")
                     + ("" if done.get("tree_clean") is not False else "; the tree had uncommitted changes")
                     + (f"; {s.exit_detail}" if s.exit_detail else ""))
    print(f"[arbor] Paused {s.exit_detail}. Continue: python3 -m arbor run --config {s.config_path}")


def _main(argv: list[str] | None, held: dict) -> int:
    """``arbor run``: the master and its workers, in this process (or, with ``--split processes``, as commands)."""
    args = _parse(argv)
    if args.exit_at_handover and args.split != "loop":
        print(f"[arbor] Run: --exit-at-handover runs the master and a worker in this process, the loop split; "
              f"--split {args.split} hands over between commands already", file=sys.stderr)
        return 2
    if args.split == "processes":
        return _split(argv, args)
    if args.split == "workers":
        from arbor import dealer
        return dealer.run(argv, args)
    s = master.prepare(args, held)
    if not isinstance(s, master.Session):
        return s
    stretch = worker.Stretch(s)
    if getattr(args, "exit_at_handover", False):
        return _until_handover(s, stretch)
    step = _loop(s, stretch)
    if step.kind == "return":
        return step.rc
    stretch.end(s.exit_reason, s.exit_detail)
    return master.finish(s)


def _until_handover(s: master.Session, stretch: worker.Stretch) -> int:
    """``arbor run --exit-at-handover``: what ``arbor master`` then ``arbor worker`` would run, in this process, until
    the worker's stretch hands over; then its report, ``role_exit.json`` ``reported``, exit 0, for a host that deals
    each report (its map dealer) before it runs the same command again.

    The same command continues where it stopped: a work order still waiting for its report is worked first (a run
    stopped mid-stretch); otherwise the master's passes run (the map, a round's end) until the next worker's turn,
    whose order is written, and that stretch runs. A run that ends instead (the campaign's end, a STOP, the deadline,
    a pause) ends as ``arbor run`` ends, a stretch in progress reported first: ``role_exit.json`` is the master's
    ``ended``, as the last command of ``--split processes`` writes it.

    Under a host's plan (``component.plan``), the share is the stretch's map: once every direction of it is closed,
    the stretch hands it back (``reported``) instead of closing the lane, and the next run of the command takes the
    plan's directions the session does not hold yet as its next share (``component.deal_plan``). With nothing new
    dealt, the master's turn closes the lane and the run ends ``done``, as without the flag."""
    dealt = component.deal_plan(s.campaign.config, s.session_dir, CampaignState.load(s.state_path).iteration + 1)
    if dealt:
        print(f"[arbor] Run: {dealt}")
    order = roles.open_order(s.session_dir)
    if order is None:
        step = _loop(s, stretch, (lifecycle.MASTER,))
        if step.kind == "return":
            return step.rc
        if step.kind != "handover":
            roles.write_exit(s.session_dir, lifecycle.MASTER, roles.ENDED, None, s.exit_reason, s.exit_detail)
            return master.finish(s)
        stretch.start(step.state)
    else:
        stretch.resume(order)
    from arbor import directions
    with directions.lane_hands_back():
        step = _loop(s, stretch, (lifecycle.WORKER,))
    if step.kind == "return":
        return step.rc
    if step.kind != "handover":
        stretch.end(s.exit_reason, s.exit_detail)      # the last stretch's report, then the campaign's end
        roles.write_exit(s.session_dir, lifecycle.MASTER, roles.ENDED, None, s.exit_reason, s.exit_detail)
        return master.finish(s)
    stretch.end(*stretch.handover(step.state))
    return master.hand_over(s, roles.REPORTED, s.session_dir / roles.REPORT_FILE)


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
# the session's, which another attempt does not mend.
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
    # ARBOR_SESSION_DIR tells the trace hooks (evals/arbor_trace.py) where to write. Without it they followed the
    # pointer `init` writes, which a resume on another node does not have, and filed a whole resumed run under
    # ~/.arbor/trajectories/.
    plan = lifecycle.declared(campaign)
    os.environ.update(ARBOR_SESSION_DIR=str(session_dir),
                      ARBOR_KNOWLEDGE_SESSION=str(session_dir), ARBOR_KNOWLEDGE_ITERATION=str(iteration),
                      ARBOR_KNOWLEDGE_CONFIG=str(s.config_path),
                      ARBOR_KNOWLEDGE_LOADED="1" if plan is not None and KNOWLEDGE_SKILL in plan.load
                      else "0")

    # S5: recover any commit left orphaned by a prior kill between commit and state.save
    container = campaign.config.get("docker_container", "")
    workspace = campaign.config.get("workspace_root", "")
    if container and workspace:
        recover_orphaned_commit(container, workspace, state.iteration)
        # The tree now is the kept one (arbor/execute/kept.py): what the harness committed since the last pass (an
        # integration of the workers' KEEPs) is kept too, and an iteration the session never kept goes.
        try:
            moved = kept.refresh(container, campaign.config, session_dir, iteration)
            if moved:
                print(f"[arbor] kept tree: {moved['line']}")
        except Exception as exc:   # noqa: BLE001 - the restore then goes to the older point, as it is recorded
            print(f"[arbor] WARNING: the kept tree was not refreshed: {exc}")

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
        # The tree after the verdict (arbor/execute/kept.py): a KEEP or PROBE is the new kept point; a REVERT or a
        # NOT MEASURED iteration goes back to the last one, an INVALID's change noted first for the revisit phase.
        if container and workspace:
            try:
                done = kept.after_pass(container, campaign.config, session_dir, iteration, result)
                if done:
                    print(f"[arbor] {done}")
            except Exception as exc:   # noqa: BLE001 - bookkeeping must not cost the verdict
                print(f"[arbor] WARNING: the kept tree was not restored or recorded: {exc}")
        # A revisit's verdict on the change it measured again (arbor/revisit.py).
        try:
            closed = revisit.close(session_dir, iteration, result)
            if closed:
                print(f"[arbor] revisit: iteration {iteration} {closed['decision']}")
        except Exception as exc:   # noqa: BLE001 - the next pass closes it from the session's record
            print(f"[arbor] WARNING: the revisit was not closed: {exc}")
        if decided in triggers.FIRED:
            s.fire(decided, iteration, change=result.get("change"), metric=result.get("metric"))
        if decided == "invalid" and str(result.get("reason") or "").startswith("agent did not run"):
            s.fire("agent_failed", iteration, reason=result.get("reason"))
        if result.get("decision") in ("STOPPED", "ROUND_END", "SKIPPED", "HANDED_BACK"):
            # The map stopped the campaign, or ended its round, before the agent
            # ran (arbor/directions.py), or a revisit's change no longer applied
            # (arbor/revisit.py), or a host's share is worked out (--exit-at-handover): no iteration to time; the
            # loop's STOP check exits, its round-end check runs the A/B, the next revisit arms, or the worker hands
            # its share back.
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
                "SLURM_JOB_ID") or "local")
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
