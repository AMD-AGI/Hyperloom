"""Rolling chunks: a campaign longer than one job hands off to the next node.

Cluster A's rules: never hold a node,
each job's work is capped at 4 hours, and longer work rolls from node to node:
the running job submits its successor at the 3-hour mark, on another node and
with no dependency, and the work switches to it at 4 hours. The switch happens
at the last iteration boundary before then: an iteration starts only if it can
finish before the switch, so none is lost.

This is Cluster A's rule, not a universal one: it lives on Arbor ``main`` only (it
began on Arbor's ``<old branch>``). It is off unless the launcher sets ``ARBOR_ROLL=1``
(``scripts/cluster/chunk_lib.sh``); without that the driver never imports this
module and behaves as before.

The split: this module owns the rule (the clock, when the successor is due,
the switch deadline, the iteration estimate, the session lease and the
handoff marker); the launcher owns the mechanics (``sbatch``, setup, waiting
for its predecessor, restore, cleanup, whether the chain goes on).

Everything that crosses a switch is in ``<session_root>/chain/`` on shared
storage:

  lease.json          which job's driver may write the session
  next_of_<job>       the successor this job submitted (one per job id)
  handoff_<job>.json  the predecessor's last word: iteration, times, successor
  iter_times.jsonl    iteration durations, for the estimate
  submissions.log     every successor submission attempt

Exit codes the launcher reads: 75 handed off, 73 lease held by a live job,
78 roll mode asked for but not configured.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

EXIT_HANDOFF = 75
EXIT_LEASE_HELD = 73
EXIT_ROLL_CONFIG = 78

# Slurm states in which a job's processes may still run. PENDING is not live: a
# requeued job is PENDING again, and its processes are already gone.
LIVE_STATES = {"RUNNING", "COMPLETING", "CONFIGURING", "SUSPENDED", "STOPPED",
               "SIGNALING", "RESIZING", "STAGE_OUT"}
# A successor in these states still exists: it is not resubmitted.
QUEUED_STATES = {"PENDING", "REQUEUED", "REQUEUE_HOLD", "REQUEUE_FED"}

DEFAULTS = {
    "wall_min": 240,           # the operator's cap on one job's work
    "successor_at_min": 180,   # when the successor is submitted
    "exit_margin_min": 15,     # the switch comes this long before the wall: USR1 (7 min) + cleanup
    "max_chunks": 0,           # 0: no cap here (the launcher may have its own)
    "est_default_min": 60,     # an iteration's length with nothing measured yet
    "est_floor_min": 15,
    "est_window": 8,           # the last N iterations
    "est_quantile": 0.75,      # an overrun loses the iteration; an early handoff only idles
    "est_factor": 1.15,
    "wait_max_min": 90,        # a successor waits this long for its predecessor's handoff
    "resubmit_limit": 3,       # successors resubmitted after the recorded one vanished
}
ENV_KEYS = {
    "wall_min": "ARBOR_ROLL_WALL_MIN",
    "successor_at_min": "ARBOR_ROLL_SUCCESSOR_AT_MIN",
    "exit_margin_min": "ARBOR_ROLL_EXIT_MARGIN_MIN",
    "max_chunks": "ARBOR_ROLL_MAX_CHUNKS",
    "est_default_min": "ARBOR_ROLL_EST_DEFAULT_MIN",   # a short smoke test's iterations
    "est_floor_min": "ARBOR_ROLL_EST_FLOOR_MIN",
}


def log(msg: str) -> None:
    print(f"[arbor] roll: {msg}", flush=True)


def settings(config: dict | None, env=None) -> dict:
    """Environment first (the launcher's), then config["cluster"]["roll"], then defaults."""
    env = os.environ if env is None else env
    out = dict(DEFAULTS)
    block = ((config or {}).get("cluster") or {}).get("roll") or {}
    for k in DEFAULTS:
        if k in block:
            out[k] = block[k]
    for k, var in ENV_KEYS.items():
        if env.get(var, "") != "":
            out[k] = env[var]
    for k, v in DEFAULTS.items():
        out[k] = type(v)(out[k])
    return out


# -- Slurm --------------------------------------------------------------------------


def parse_timeleft(s: str) -> float | None:
    """Seconds from squeue's %L ("1-00:00:00", "3:58:10", "58:10"); None if none is set."""
    s = (s or "").strip()
    if not s or not s[0].isdigit():
        return None   # UNLIMITED, INVALID, NOT_SET, empty
    try:
        days = 0
        if "-" in s:
            d, s = s.split("-", 1)
            days = int(d)
        parts = [int(p) for p in s.split(":")]
    except ValueError:
        return None
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, sec = parts[-3:]
    return float(days * 86400 + h * 3600 + m * 60 + sec)


def _squeue(args: list[str], run: Callable = subprocess.run) -> str | None:
    try:
        p = run(["squeue", "-h", *args], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return p.stdout if p.returncode == 0 else None


def list_my_jobs(run: Callable = subprocess.run) -> dict[str, str] | None:
    """{job id: state} for this user's jobs; None if squeue could not be read."""
    out = _squeue(["-u", str(os.getuid()), "-o", "%i|%T"], run)
    if out is None:
        return None
    jobs = {}
    for line in out.splitlines():
        jid, _, state = line.strip().partition("|")
        if jid:
            jobs[jid] = state.strip()
    return jobs


def job_timeleft(job: str, run: Callable = subprocess.run) -> float | None:
    out = _squeue(["-j", str(job), "-o", "%L"], run)
    return parse_timeleft(out.splitlines()[0]) if out and out.strip() else None


def alive(job: str, jobs: dict[str, str] | None) -> bool | None:
    """True if the job may still run, False if not, None if it could not be checked."""
    if jobs is None:
        return None
    return jobs.get(str(job)) in LIVE_STATES


# -- small helpers -------------------------------------------------------------------


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def nearest_rank(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[max(0, math.ceil(q * len(xs)) - 1)]


def _hms(t: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(t))


def seed_digest(config: dict) -> str:
    """The campaign config's server_env and server_flags: what a fresh container is seeded with."""
    seed = {"server_env": config.get("server_env", {}), "server_flags": config.get("server_flags", "")}
    return hashlib.sha256(json.dumps(seed, sort_keys=True).encode()).hexdigest()[:16]


# -- the rule ------------------------------------------------------------------------


class Roll:
    """One job's share of a rolling campaign."""

    def __init__(self, config: dict, session_dir: Path, *, job: str, t0: float,
                 submit_cmd: str, pred: str | None = None, cfg: dict | None = None,
                 clock: Callable[[], float] = time.time,
                 jobs_fn: Callable[[], dict | None] = list_my_jobs,
                 timeleft_fn: Callable[[str], float | None] = job_timeleft,
                 run: Callable = subprocess.run):
        self.config = config
        self.session = Path(session_dir)
        self.chain = self.session / "chain"
        self.chain.mkdir(parents=True, exist_ok=True)
        self.job = str(job)
        self.t0 = float(t0)
        self.submit_cmd = submit_cmd
        self.pred = str(pred) if pred else None
        self.cfg = cfg or settings(config)
        self.clock = clock
        self.jobs_fn = jobs_fn
        self.timeleft_fn = timeleft_fn
        self.run = run
        self.host = socket.gethostname().split(".")[0]
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._resubmits = 0
        self.lease_path = self.chain / "lease.json"

    # deadlines
    def job_end(self) -> float:
        end = self.t0 + self.cfg["wall_min"] * 60
        left = self.timeleft_fn(self.job)
        if left is not None:
            end = min(end, self.clock() + left)
        return end

    def switch_at(self) -> float:
        return self.job_end() - self.cfg["exit_margin_min"] * 60

    def successor_due(self) -> float:
        return self.t0 + self.cfg["successor_at_min"] * 60

    # the session lease
    def acquire(self) -> bool:
        """Take the lease, or refuse if a live job holds it (unknown counts as live)."""
        cur = _read_json(self.lease_path)
        if cur and not cur.get("released_at") and str(cur.get("job")) != self.job:
            state = alive(str(cur.get("job")), self.jobs_fn())
            if state is None:
                log(f"lease held by job {cur.get('job')} on {cur.get('host')}, whose state could not be "
                    f"checked -- refusing (unknown is never free)")
                return False
            if state:
                log(f"lease held by live job {cur.get('job')} on {cur.get('host')} -- refusing")
                return False
            log(f"taking over the lease from job {cur.get('job')} (no longer running)")
        lease = {"job": self.job, "host": self.host, "pid": os.getpid(), "since": self.clock(),
                 "released_at": None, "handed_to": None}
        _atomic_write(self.lease_path, json.dumps(lease, indent=1) + "\n")
        time.sleep(0 if self.clock is not time.time else 1)   # let a racing writer land first
        got = _read_json(self.lease_path) or {}
        return str(got.get("job")) == self.job and got.get("pid") == os.getpid()

    def release(self, handed_to: str | None = None) -> None:
        cur = _read_json(self.lease_path)
        if not cur or str(cur.get("job")) != self.job or cur.get("released_at"):
            return
        cur.update(released_at=self.clock(), handed_to=handed_to)
        _atomic_write(self.lease_path, json.dumps(cur, indent=1) + "\n")

    # the estimate
    def samples(self, state=None) -> tuple[list[float], str]:
        """Iteration durations in seconds, newest last, and where they came from."""
        own = []
        p = self.chain / "iter_times.jsonl"
        if p.is_file():
            for line in p.read_text().splitlines():
                try:
                    r = json.loads(line)
                    own.append(float(r["end"]) - float(r["start"]))
                except (ValueError, KeyError, TypeError):
                    continue
        if len(own) >= 3 or state is None:
            return own, "own"
        # Fallback: gaps between consecutive iterations' lessons, dropping any
        # that spans a switch (longer than a whole job).
        ls = sorted((l for l in getattr(state, "lessons", [])
                     if l.get("ts") is not None and l.get("iteration")), key=lambda l: l["ts"])
        wall = self.cfg["wall_min"] * 60
        gaps = [b["ts"] - a["ts"] for a, b in zip(ls, ls[1:])
                if b.get("iteration") == a.get("iteration", 0) + 1 and 0 < b["ts"] - a["ts"] < wall]
        return (own + gaps) if gaps else own, "lessons" if gaps else "own"

    def estimate(self, state=None) -> tuple[float, str]:
        xs, src = self.samples(state)
        xs = xs[-self.cfg["est_window"]:]
        if not xs:
            return self.cfg["est_default_min"] * 60, "default"
        q = nearest_rank(xs, self.cfg["est_quantile"])
        est = max(self.cfg["est_floor_min"] * 60, self.cfg["est_factor"] * q)
        return est, f"q{int(self.cfg['est_quantile'] * 100)} of {len(xs)} ({src}) x{self.cfg['est_factor']}"

    # the successor
    def _chunks(self) -> int:
        return len(list(self.chain.glob("job_*")))

    def ensure_successor(self, reason: str) -> str | None:
        """Submit this job's successor once; return its id (or None)."""
        with self._lock:
            if (self.chain / "CHAIN_STOP").exists():
                return None
            cap = self.cfg["max_chunks"]
            if cap and self._chunks() >= cap:
                if not getattr(self, "_cap_logged", False):
                    log(f"chunk cap {cap} reached -- no successor")
                    self._cap_logged = True
                return None
            rec = self.chain / f"next_of_{self.job}"
            if rec.is_file():
                nxt = rec.read_text().strip()
                jobs = self.jobs_fn()
                if nxt and (jobs is None or jobs.get(nxt) in LIVE_STATES | QUEUED_STATES):
                    return nxt   # running, queued, or unknown: never submit a second one
                if self._resubmits >= self.cfg["resubmit_limit"]:
                    log(f"successor {nxt} is gone and the resubmit limit is reached")
                    return None
                self._resubmits += 1
                log(f"successor {nxt} is gone -- resubmitting ({self._resubmits}/{self.cfg['resubmit_limit']})")
            try:
                p = self.run(["bash", "-c", self.submit_cmd], capture_output=True, text=True, timeout=120)
                out = (p.stdout or "").strip().splitlines()
                nid = out[-1].strip().split(";")[0] if out else ""
                ok = p.returncode == 0 and nid.isdigit()
            except (OSError, subprocess.TimeoutExpired) as exc:
                ok, nid, p = False, "", None
                out = [str(exc)]
            with open(self.chain / "submissions.log", "a") as f:
                f.write(json.dumps({"job": self.job, "at": self.clock(), "reason": reason, "ok": ok,
                                    "successor": nid if ok else None,
                                    "output": (p.stderr if p is not None else "")[-300:] if not ok else ""}) + "\n")
            if not ok:
                log(f"successor submission failed ({reason}) -- will retry")
                return None
            _atomic_write(rec, nid + "\n")
            log(f"successor {nid} submitted ({reason}); it provisions on another node and waits for the handoff")
            return nid

    def start_timer(self) -> None:
        delay = max(0.0, self.successor_due() - self.clock())

        def fire():
            if self.ensure_successor("3-hour mark") is None and not (self.chain / "CHAIN_STOP").exists():
                self._timer = threading.Timer(300, fire)   # failed: try again in 5 minutes
                self._timer.daemon = True
                self._timer.start()
        self._timer = threading.Timer(delay, fire)
        self._timer.daemon = True
        self._timer.start()

    def stop_timer(self) -> None:
        if self._timer:
            self._timer.cancel()

    # the loop's questions
    def may_start_iteration(self, state=None) -> bool:
        now = self.clock()
        if now >= self.successor_due():
            self.ensure_successor("3-hour mark (loop)")
        est, basis = self.estimate(state)
        sw = self.switch_at()
        if now + est <= sw:
            log(f"next iteration est {est / 60:.0f} min ({basis}) ends {_hms(now + est)} <= switch {_hms(sw)}")
            return True
        log(f"next iteration est {est / 60:.0f} min ({basis}) would end {_hms(now + est)} > switch "
            f"{_hms(sw)} -- handing off")
        return False

    def after_iteration(self, iteration: int, started: float, state=None) -> None:
        end = self.clock()
        with open(self.chain / "iter_times.jsonl", "a") as f:
            f.write(json.dumps({"job": self.job, "iteration": iteration, "start": started, "end": end}) + "\n")
        last = (getattr(state, "lessons", None) or [{}])[-1] if state is not None else {}
        if last.get("iteration") == iteration and last.get("decision") == "KEEP":
            self.mirror_workspace_config()

    # the workspace config across containers
    def _container(self) -> tuple[str, str]:
        return self.config.get("docker_container", ""), self.config.get("workspace_root", "")

    def mirror_workspace_config(self) -> None:
        mirror_workspace_config(self.config, self.session, self.job, self.clock)

    def restore_workspace_config(self) -> bool:
        return restore_workspace_config(self.config, self.session)

    def _tree_vs_landed(self) -> str:
        """Whether the container's tree is what landed.patch says (audit only)."""
        container, workspace = self._container()
        landed = self.session / "landed.patch"
        if not container or not workspace or not landed.is_file():
            return "unknown"
        try:
            # The writer's own pathspec: comparing with `:(exclude)arbor/` read every
            # campaign with a served file under arbor/ as "differs".
            from arbor.execute.workspace import git, landed_pathspec
            root = git(container, workspace, "rev-list", "--max-parents=0", "HEAD").strip().splitlines()[0]
            diff = git(container, workspace, "diff", "--binary", root, "HEAD", "--",
                       *landed_pathspec(self.config))
        except Exception:
            return "unknown"
        # git() strips the output and landed.patch is git's own bytes, which end with a
        # blank line after a binary hunk: compare without the trailing newlines.
        if diff.rstrip("\n") == landed.read_text(errors="replace").rstrip("\n"):
            return "same"
        (self.chain / f"handoff_{self.job}.tree.patch").write_text(diff + "\n")
        return "differs"

    # the switch
    def handoff(self, state=None) -> int:
        nxt = self.ensure_successor("handoff")
        self.stop_timer()
        est, basis = self.estimate(state)
        tree = self._tree_vs_landed()
        iteration = getattr(state, "iteration", None)
        last = ((getattr(state, "lessons", None) or [{}])[-1] or {}).get("decision") if state is not None else None
        log(f"handing off after iteration {iteration} to {nxt or 'no successor'}; "
            f"switch {_hms(self.switch_at())}, job end {_hms(self.job_end())}, tree vs landed.patch: {tree}")
        marker = {"job": self.job, "pred": self.pred, "host": self.host, "iteration": iteration,
                  "last_decision": last, "at": self.clock(), "switch_at": self.switch_at(),
                  "job_end": self.job_end(), "estimate_s": est, "estimate_basis": basis,
                  "successor": nxt, "head_vs_landed": tree,
                  "workspace_config_mirrored": (self.session / "workspace_config.json").is_file()}
        _atomic_write(self.chain / f"handoff_{self.job}.json", json.dumps(marker, indent=1) + "\n")
        self.release(handed_to=nxt)
        return EXIT_HANDOFF


# -- the flag KEEPs across containers, rolling or not ------------------------------------
# A driver that does not roll calls these too (arbor/driver.py): a campaign moved to a held
# job with roll mode off got a fresh container seeded from the config, and its first
# iteration there ran without --enable-expert-parallel, a large regression.


def mirror_workspace_config(config: dict, session: Path, job: str = "", clock=time.time) -> None:
    """Copy the container's arbor/config.json (flag KEEPs) to the session, beside landed.patch."""
    container, workspace = config.get("docker_container", ""), config.get("workspace_root", "")
    if not container or not workspace:
        return
    try:
        from arbor.execute.workspace import read_workspace_config
        ws = read_workspace_config(container, workspace)
    except Exception as exc:   # bookkeeping: never overturns a verdict
        log(f"could not read arbor/config.json to mirror it: {exc}")
        return
    if ws:
        _atomic_write(Path(session) / "workspace_config.json",
                      json.dumps({"workspace_config": ws, "seed_digest": seed_digest(config),
                                  "job": job, "at": clock()}, indent=1) + "\n")


def restore_workspace_config(config: dict, session: Path) -> bool:
    """Into a fresh container: the mirrored flag KEEPs, unless the operator changed the seed."""
    m = _read_json(Path(session) / "workspace_config.json")
    container, workspace = config.get("docker_container", ""), config.get("workspace_root", "")
    if not m or not container or not workspace:
        return False
    if m.get("seed_digest") != seed_digest(config):
        log("workspace_config.json predates a change to the config's server_env/server_flags -- "
            "the operator's edit wins, not restoring it")
        return False
    try:
        from arbor.execute.workspace import git, read_workspace_config, write_workspace_config
        cur = read_workspace_config(container, workspace)
        seed = {"server_env": config.get("server_env", {}), "server_flags": config.get("server_flags", "")}
        want = m.get("workspace_config") or {}
        if cur == want:
            return False
        if cur and cur != seed:
            log("the container's arbor/config.json is neither the seed nor the mirror -- leaving it")
            return False
        write_workspace_config(container, workspace, want.get("server_env", {}), want.get("server_flags", ""))
        git(container, workspace, "add", "-f", "arbor/config.json")
        git(container, workspace, "commit", "-q", "-m", "restore: workspace config")
        log("restored arbor/config.json (flag KEEPs) from the session")
        return True
    except Exception as exc:
        log(f"could not restore arbor/config.json: {exc}")
        return False


def start(config: dict, session_dir: Path) -> tuple[Roll | None, int | None]:
    """The driver's entry: read the launcher's environment, take the lease, start the timer."""
    env = os.environ
    missing = [v for v in ("ARBOR_ROLL_T0", "SLURM_JOB_ID", "ARBOR_ROLL_SUBMIT") if not env.get(v)]
    if missing:
        log(f"ARBOR_ROLL=1 but {', '.join(missing)} unset -- refusing to run without a deadline")
        return None, EXIT_ROLL_CONFIG
    roll = Roll(config, session_dir, job=env["SLURM_JOB_ID"], t0=float(env["ARBOR_ROLL_T0"]),
                submit_cmd=env["ARBOR_ROLL_SUBMIT"], pred=env.get("ARBOR_ROLL_PRED"))
    if not roll.acquire():
        return None, EXIT_LEASE_HELD
    import atexit
    atexit.register(roll.release)
    roll.restore_workspace_config()
    roll.start_timer()
    c = roll.cfg
    log(f"job {roll.job} on {roll.host}: wall {c['wall_min']} min, successor at {_hms(roll.successor_due())}, "
        f"switch by {_hms(roll.switch_at())}" + (f", after {roll.pred}" if roll.pred else ""))
    return roll, None


# -- the launcher's side --------------------------------------------------------------


def wait(chain: Path, pred: str, timeout_min: float, poll_s: float = 20,
         jobs_fn: Callable[[], dict | None] = list_my_jobs, clock=time.time, sleep=time.sleep) -> str:
    """Wait for the predecessor: its handoff marker, or the job gone. Unknown keeps waiting."""
    end = clock() + timeout_min * 60
    while True:
        if (chain / f"handoff_{pred}.json").is_file():
            return "handoff"
        if alive(pred, jobs_fn()) is False:
            # Gone without a marker: preempted, crashed or cancelled. NFS can lag a
            # marker written just before the end, so look once more.
            sleep(min(poll_s, 30))
            return "handoff" if (chain / f"handoff_{pred}.json").is_file() else "gone"
        if clock() >= end:
            return "timeout"
        sleep(poll_s)


def _selftest(args) -> int:
    """The driver's loop with sleeps for iterations (the Cluster A smoke test)."""
    session = Path(args.session)
    roll, rc = start({}, session)
    if rc is not None:
        return rc
    i = 0
    while True:
        if (session / "STOP").exists() or (roll.chain / "CHAIN_STOP").exists():
            log("STOP -- exiting")
            return 0
        if not roll.may_start_iteration(None):
            return roll.handoff(None)
        i += 1
        t = roll.clock()
        log(f"selftest iteration {i} ({args.iter_s} s)")
        if args.crash_at and i >= args.crash_at:
            log("selftest crash")
            os._exit(1)
        time.sleep(args.iter_s)
        roll.after_iteration(i, t)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m arbor.cluster.handoff")
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("wait", help="wait for the predecessor's handoff")
    w.add_argument("--chain", required=True)
    w.add_argument("--pred", required=True)
    w.add_argument("--timeout-min", type=float, default=DEFAULTS["wait_max_min"])
    w.add_argument("--poll-s", type=float, default=20)
    e = sub.add_parser("ensure-successor", help="submit this job's successor if it has none")
    e.add_argument("--session", required=True)
    e.add_argument("--reason", default="launcher")
    s = sub.add_parser("status", help="lease, last handoff, successor and the estimate")
    s.add_argument("--session", required=True)
    t = sub.add_parser("selftest", help="the loop with sleeps for iterations")
    t.add_argument("--session", required=True)
    t.add_argument("--iter-s", type=float, default=150)
    t.add_argument("--crash-at", type=int, default=0)
    a = ap.parse_args(argv)
    if a.cmd == "wait":
        r = wait(Path(a.chain), a.pred, a.timeout_min, a.poll_s)
        print(r)
        return 3 if r == "timeout" else 0
    if a.cmd == "ensure-successor":
        env = os.environ
        if not (env.get("SLURM_JOB_ID") and env.get("ARBOR_ROLL_SUBMIT")):
            print("SLURM_JOB_ID and ARBOR_ROLL_SUBMIT are required", file=sys.stderr)
            return EXIT_ROLL_CONFIG
        roll = Roll({}, Path(a.session), job=env["SLURM_JOB_ID"], t0=float(env.get("ARBOR_ROLL_T0") or time.time()),
                    submit_cmd=env["ARBOR_ROLL_SUBMIT"], pred=env.get("ARBOR_ROLL_PRED"))
        return 0 if roll.ensure_successor(a.reason) else 1
    if a.cmd == "status":
        session = Path(a.session)
        chain = session / "chain"
        roll = Roll({}, session, job=os.environ.get("SLURM_JOB_ID", "status"), t0=time.time(),
                    submit_cmd="false", jobs_fn=lambda: None, timeleft_fn=lambda j: None)
        handoffs = sorted(chain.glob("handoff_*.json"), key=lambda p: p.stat().st_mtime)
        est, basis = roll.estimate()
        print(json.dumps({"lease": _read_json(chain / "lease.json"),
                          "last_handoff": _read_json(handoffs[-1]) if handoffs else None,
                          "next_of": {p.name[8:]: p.read_text().strip() for p in chain.glob("next_of_*")},
                          "estimate_min": round(est / 60, 1), "estimate_basis": basis,
                          "samples_min": [round(x / 60, 1) for x in roll.samples()[0][-roll.cfg['est_window']:]]},
                         indent=1))
        return 0
    return _selftest(a)


if __name__ == "__main__":
    sys.exit(main())
