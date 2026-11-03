"""The cluster as the controller sees it: Known, Unknown or Failed.

`Unknown` is what a timed-out or unreadable `squeue`/`sacct` gives, and it
never triggers an action. A restart guard that read "could not check" as
"nothing running" once relaunched a driver into a point that was mid-
measurement (AGENTS.md section 4); here the controller simply waits for the
next tick.

Submission is the same: an `sbatch` that timed out may still have queued the
job, so it is `Unknown`, and the controller looks the job up by its name
before it submits again (`find_by_name`). Every job name a run submits is
unique to its lane and chain (or map step, or A/B, and attempt), so a
duplicate is visible.

`SlurmCluster` runs the commands (Cluster A: sbatch only, nothing held). The
`FakeCluster` runs jobs in-process for the end-to-end tests.

Ported from arbor/mesh3/cluster.py (<fork>/mesh3-confirm a691edc). Added:
`intervals` (sacct's start and end of finished jobs, for a lane's clocks),
and a fake job may keep running across advances or be requeued. Not ported:
the multi-site router.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from typing import Callable, Union

from arbor.cluster.handoff import parse_timeleft

PENDING_STATES = ("PENDING", "CONFIGURING", "REQUEUED", "REQUEUE_HOLD", "RESV_DEL_HOLD", "SUSPENDED")
RUNNING_STATES = ("RUNNING", "COMPLETING", "STAGE_OUT", "SIGNALING", "RESIZING")
FAILED_STATES = ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED",
                 "BOOT_FAIL", "DEADLINE", "REVOKED", "SPECIAL_EXIT")


@dataclass(frozen=True)
class Known:
    state: str                      # PENDING | RUNNING | COMPLETED (normalized)
    node: str | None = None
    reason: str = ""
    elapsed_s: float | None = None
    timeleft_s: float | None = None


@dataclass(frozen=True)
class Failed:
    state: str
    reason: str = ""
    node: str | None = None


@dataclass(frozen=True)
class Unknown:
    why: str


Status = Union[Known, Failed, Unknown]


def _base_state(raw: str) -> str:
    # sacct prints "CANCELLED by 1234"; squeue prints plain names.
    return (raw or "").strip().split(" ")[0].upper()


def _normalize(raw: str, node: str | None, reason: str = "") -> Status:
    s = _base_state(raw)
    node = node if node and node not in ("(null)", "None", "") else None
    if s in PENDING_STATES:
        return Known("PENDING", node, reason)
    if s in RUNNING_STATES:
        return Known("RUNNING", node, reason)
    if s == "COMPLETED":
        return Known("COMPLETED", node, reason)
    if s in FAILED_STATES:
        return Failed(s, reason, node)
    return Unknown(f"unrecognized state {raw!r}")


def parse_squeue(out: str) -> dict[str, Known | Failed | Unknown]:
    """`%i|%T|%N|%r|%M|%L` lines."""
    jobs = {}
    for line in out.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 2 or not parts[0].strip():
            continue
        jid, state = parts[0].strip(), parts[1]
        node = parts[2].strip() if len(parts) > 2 else None
        reason = parts[3].strip() if len(parts) > 3 else ""
        st = _normalize(state, node, reason if reason not in ("None", "(null)") else "")
        if isinstance(st, Known) and len(parts) > 5:
            st = Known(st.state, st.node, st.reason, parse_timeleft(parts[4]), parse_timeleft(parts[5]))
        jobs[jid] = st
    return jobs


def parse_sacct(out: str) -> dict[str, Status]:
    """`JobIDRaw|State|NodeList|ExitCode` lines (-X: allocations only)."""
    jobs = {}
    for line in out.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 2 or not parts[0].strip() or "." in parts[0]:
            continue
        jid = parts[0].strip()
        node = parts[2].strip() if len(parts) > 2 else None
        exit_code = parts[3].strip() if len(parts) > 3 else ""
        jobs[jid] = _normalize(parts[1], node, f"exit {exit_code}" if exit_code else "")
    return jobs


class SlurmCluster:
    """Slurm by its commands: here, on the login node (host None), or over ssh to `host` (a
    controller on a workstation; plan B2). Over ssh only the job's own variables travel, set by
    `env` on the remote side: this machine's environment is never a job's."""

    def __init__(self, run: Callable = subprocess.run, *, timeout_s: float = 30.0,
                 partition: str | None = None, account: str | None = None, host: str | None = None) -> None:
        self.run, self.timeout_s = run, timeout_s
        self.partition, self.account, self.host = partition, account, host

    def _cmd(self, cmd: list[str], env: dict | None = None) -> tuple[int, str, str] | None:
        if self.host:
            remote = " ".join(shlex.quote(x) for x in
                              (["env", *[f"{k}={v}" for k, v in (env or {}).items()]] if env else []) + cmd)
            cmd, env, timeout = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", self.host, remote], None, \
                self.timeout_s + 30
        else:
            timeout = self.timeout_s
        try:
            p = self.run(cmd, capture_output=True, text=True, timeout=timeout,
                         env={**os.environ, **env} if env else None)
        except (OSError, subprocess.TimeoutExpired):
            return None
        if self.host and p.returncode == 255:      # ssh could not reach the host: unknown, not a refusal
            return None
        return p.returncode, p.stdout or "", p.stderr or ""

    def status(self, job_ids: list[str]) -> dict[str, Status]:
        ids = [str(j) for j in job_ids if j]
        if not ids:
            return {}
        r = self._cmd(["squeue", "-h", "-j", ",".join(ids), "-o", "%i|%T|%N|%r|%M|%L"])
        # squeue exits non-zero when every listed job has left the queue; that is not "unknown".
        if r is None or (r[0] != 0 and "Invalid job id" not in r[2]):
            return {j: Unknown("squeue could not be read") for j in ids}
        out = parse_squeue(r[1]) if r[0] == 0 else {}
        gone = [j for j in ids if j not in out]
        if gone:
            r = self._cmd(["sacct", "-n", "-P", "-X", "-j", ",".join(gone),
                           "-o", "JobIDRaw,State,NodeList,ExitCode"])
            acct = parse_sacct(r[1]) if r is not None and r[0] == 0 else None
            for j in gone:
                if acct is None:
                    out[j] = Unknown("left the queue and sacct could not be read")
                else:
                    out[j] = acct.get(j, Unknown("in neither squeue nor sacct"))
        return out

    def find_by_name(self, name: str) -> list[str] | None:
        """Live jobs with this name; None if squeue could not be read."""
        r = self._cmd(["squeue", "-h", "-n", name, "-o", "%i"])
        if r is None or r[0] != 0:
            return None
        return [l.strip() for l in r[1].splitlines() if l.strip()]

    def submit(self, script: str, *, name: str, args: list[str] = (), env: dict | None = None,
               exclude: list[str] = (), time_limit: str | None = None,
               output: str | None = None) -> str | Failed | Unknown:
        cmd = ["sbatch", "--parsable", f"--job-name={name}"]
        if self.partition:
            cmd.append(f"--partition={self.partition}")
        if self.account:
            cmd.append(f"--account={self.account}")
        if time_limit:
            cmd.append(f"--time={time_limit}")
        if exclude:
            cmd.append("--exclude=" + ",".join(sorted(set(exclude))))
        if output:
            cmd.append(f"--output={output}")
        cmd += [script, *map(str, args)]
        r = self._cmd(cmd, env)
        if r is None:
            return Unknown("sbatch timed out; the job may exist")
        rc, out, err = r
        jid = out.strip().split(";")[0]
        if rc == 0 and jid.isdigit():
            return jid
        if rc == 0:
            return Unknown(f"sbatch printed {out.strip()[:80]!r}")
        return Failed("REJECTED", (err or out).strip()[:300])

    def cancel(self, job_id: str) -> bool | None:
        r = self._cmd(["scancel", str(job_id)])
        return None if r is None else r[0] == 0

    def intervals(self, job_ids: list[str]) -> dict[str, list[tuple]] | None:
        """{job: [(start, end, node), ...]} from sacct (every requeue, -D); None if unreadable."""
        ids = [str(j) for j in job_ids if j]
        if not ids:
            return {}
        r = self._cmd(["sacct", "-n", "-P", "-X", "-D", "-j", ",".join(ids), "-o", "JobIDRaw,Start,End,NodeList"])
        if r is None or r[0] != 0:
            return None
        return parse_intervals(r[1])


def _epoch(stamp: str) -> float | None:
    try:
        return time.mktime(time.strptime(stamp.strip(), "%Y-%m-%dT%H:%M:%S"))
    except ValueError:
        return None


def parse_intervals(out: str) -> dict[str, list[tuple]]:
    """`JobIDRaw|Start|End|NodeList` lines; a job not started, or still running, has no interval."""
    found: dict[str, list[tuple]] = {}
    for line in out.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 3 or not parts[0].strip() or "." in parts[0]:
            continue
        start, end = _epoch(parts[1]), _epoch(parts[2])
        if start is not None and end is not None and end >= start:
            node = parts[3].strip() if len(parts) > 3 else None
            found.setdefault(parts[0].strip(), []).append((start, end, node or None))
    return found


# -- the fake ------------------------------------------------------------------------


@dataclass
class FakeJob:
    id: str
    script: str
    name: str
    args: list[str]
    env: dict
    exclude: list[str]
    state: str = "PENDING"
    node: str | None = None
    reason: str = ""
    result: object = None
    restarts: int = 0
    runs: list = field(default_factory=list)      # [start, end, node] per allocation


@dataclass
class FakeCluster:
    """Jobs run in-process when `advance()` is called.

    `handlers` maps a script's basename to `fn(job) -> bool | str`: True or
    "COMPLETED" completes the job, False or a failed state fails it, "RUNNING"
    keeps it running to the next advance, and "REQUEUE" puts it back in the
    queue as a preemption does. A script with no handler completes. `unknown`
    makes the next N status calls Unknown. `clock` stamps each allocation.
    """
    nodes: list[str] = field(default_factory=lambda: [f"n{i:02d}" for i in range(8)])
    handlers: dict[str, Callable[[FakeJob], object]] = field(default_factory=dict)
    jobs: dict[str, FakeJob] = field(default_factory=dict)
    unknown: int = 0
    reject: int = 0
    lose_submit: int = 0            # queue the job but answer Unknown, as a timed-out sbatch can
    clock: Callable[[], float] = time.time
    _next: int = 1000

    def submit(self, script: str, *, name: str, args: list[str] = (), env: dict | None = None,
               exclude: list[str] = (), time_limit: str | None = None,
               output: str | None = None) -> str | Failed | Unknown:
        if self.reject:
            self.reject -= 1
            return Failed("REJECTED", "fake rejection")
        self._next += 1
        jid = str(self._next)
        self.jobs[jid] = FakeJob(jid, script, name, list(map(str, args)), dict(env or {}), list(exclude))
        if self.lose_submit:
            self.lose_submit -= 1
            return Unknown("fake: sbatch timed out")
        return jid

    def status(self, job_ids: list[str]) -> dict[str, Status]:
        if self.unknown:
            self.unknown -= 1
            return {str(j): Unknown("fake: unreadable") for j in job_ids}
        out = {}
        for j in map(str, job_ids):
            job = self.jobs.get(j)
            if job is None:
                out[j] = Unknown("in neither squeue nor sacct")
            else:
                out[j] = _normalize(job.state, job.node, job.reason)
        return out

    def find_by_name(self, name: str) -> list[str] | None:
        return [j.id for j in self.jobs.values() if j.name == name and j.state in ("PENDING", "RUNNING")]

    def cancel(self, job_id: str) -> bool | None:
        job = self.jobs.get(str(job_id))
        if job is None:
            return False
        if job.state in ("PENDING", "RUNNING"):
            self._end(job, "CANCELLED")
        return True

    def intervals(self, job_ids: list[str]) -> dict[str, list[tuple]] | None:
        return {j: [tuple(r) for r in self.jobs[j].runs if r[1] is not None]
                for j in map(str, job_ids) if j in self.jobs}

    def busy_nodes(self) -> set[str]:
        return {j.node for j in self.jobs.values() if j.state == "RUNNING" and j.node}

    def start_pending(self) -> list[FakeJob]:
        started = []
        for job in self.jobs.values():
            if job.state != "PENDING":
                continue
            free = [n for n in self.nodes if n not in self.busy_nodes() and n not in job.exclude]
            if not free:
                continue
            job.state, job.node = "RUNNING", free[0]
            job.runs.append([self.clock(), None, job.node])
            started.append(job)
        return started

    def _end(self, job: FakeJob, state: str) -> None:
        if job.runs and job.runs[-1][1] is None:
            job.runs[-1][1] = self.clock()
        job.state = state

    def finish(self, job: FakeJob) -> None:
        handler = self.handlers.get(os.path.basename(job.script))
        result = True if handler is None else handler(job)
        job.result = result
        if result == "RUNNING" or job.state != "RUNNING":     # still at work, or the handler moved it
            return
        if result == "REQUEUE":
            self._end(job, "PENDING")
            job.node, job.restarts = None, job.restarts + 1
        elif result is True or result == "COMPLETED" or result is None:
            self._end(job, "COMPLETED")
        else:
            self._end(job, result if isinstance(result, str) else "FAILED")

    def advance(self) -> list[FakeJob]:
        """Start what fits, then run every running job once; return the jobs that ran."""
        self.start_pending()
        ran = [j for j in self.jobs.values() if j.state == "RUNNING"]
        for job in ran:
            self.finish(job)
        return ran


# -- one job, tracked across ticks ---------------------------------------------------------


def track_submit(cluster, rec: dict, script: str, *, env: dict, exclude: list[str] = (),
                 time_limit: str | None = None, output: str | None = None) -> dict:
    """Submit the job a record names ({"name": ...}); an Unknown answer is found by name later.

    The caller saves the record, name and all, before calling: a crash between
    the submit and the save leaves SUBMITTING, which `track_poll` resolves by name.
    """
    r = cluster.submit(script, name=rec["name"], env=env, exclude=list(exclude), time_limit=time_limit,
                       output=output)
    if isinstance(r, Failed):
        rec.update(state="FAILED", reason=f"sbatch refused it: {r.reason}")
    elif isinstance(r, Unknown):
        rec.update(state="SUBMITTING")
    else:
        rec.update(id=r, state="PENDING")
    return rec


def track_poll(cluster, rec: dict) -> dict:
    """Move a tracked job's record to what the cluster says; Unknown changes nothing."""
    state = rec.get("state")
    if state == "SUBMITTING":
        found = cluster.find_by_name(rec["name"])
        if found:
            rec.update(id=found[0], state="PENDING")
        elif found == []:
            rec.update(state="LOST")          # the caller resubmits
        return rec
    if state not in ("PENDING", "RUNNING") or not rec.get("id"):
        return rec
    st = cluster.status([rec["id"]]).get(str(rec["id"]))
    if st is None or isinstance(st, Unknown):
        return rec
    if isinstance(st, Failed):
        rec.update(state="FAILED", reason=st.state, node=st.node)
    else:
        rec.update(state=st.state, node=st.node or rec.get("node"))
    return rec
