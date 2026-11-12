"""The master's side of a session: the setup, what each pass must clear, a round's end, and the campaign's end.

A session is a master and its workers (``arbor/roles.py``). The master:

- **sets the campaign up** (``prepare``): the mode's refusals, the roll lease, the review gate, the skill
  conflicts, ``init``, the recorded baseline, the host's run and the root's accuracy;
- **clears each pass** (``before_pass``): STOP, an abort, the iteration and time limits, a pending round end, the
  target, a rolling chunk's switch, a host's dossier pass whose map is set, and the host's deadline; then, with the
  map's directions all closed and time for one more iteration, arms a revisit of a NOT MEASURED change
  (``arbor/revisit.py``);
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
import shlex
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from arbor import (component, conflicts, directions, lifecycle, retraction, review, revisit, roles, rounds, triggers,
                   version)
from arbor.cli.iterate import _find_arbor_root, cmd_init
from arbor.pack import load_campaign, validate_campaign, validate_pack
from arbor.pack.loader import PACK_RESULTS_CONCERNS, PACK_RESULTS_DIR, PACK_RESULTS_FLAG
from arbor.policy.claude_loop import KNOWLEDGE_SKILL, _run_claude
from arbor.rebaseline import current as rebase_scale
from arbor.state import CampaignState

#: <session>/: the Arbor time a standalone session has spent, so a resume counts it against time_budget_minutes.
CLOCK_FILE = "clock.json"


def _spent(session_dir: Path) -> float:
    """Seconds of Arbor time the session spent before this launch, from :data:`CLOCK_FILE`; 0 without one."""
    try:
        return max(0.0, float(json.loads((Path(session_dir) / CLOCK_FILE).read_text()).get("spent_s", 0.0)))
    except (OSError, ValueError, TypeError, AttributeError):
        return 0.0


def _record_spent(session_dir: Path, seconds: float) -> None:
    """Write the session's spent Arbor time. Bookkeeping: a write that fails is said, never raised."""
    try:
        part = Path(session_dir) / f".{CLOCK_FILE}.tmp"
        part.write_text(json.dumps({"spent_s": round(seconds, 1), "ts": time.time()}) + "\n")
        os.replace(part, Path(session_dir) / CLOCK_FILE)
    except OSError as exc:
        print(f"[arbor] could not write {CLOCK_FILE}: {exc}", file=sys.stderr, flush=True)


def _resolve_target(target, metric: str):
    """Resolve a campaign `target` to a number, or None if it carries none.

    A config may express `target` either as a bare number or as a dict of
    documented reference points (``{"_source": ..., "<metric>": <number>}``).
    Comparing `best_metric >= <dict>` raises TypeError, so the dict is resolved
    to the metric's number first.
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
    would read as a normal end under a host.
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


def _root_baseline(campaign, state_path: Path, comp, session_dir: Path) -> None:
    """Component mode: the root's own measurement, once, before iteration 1, when the host gives no baseline_record.

    Without it the first iteration's measurement became the bar, with nothing to compare it to: a change that gained
    nothing was kept as the baseline, even when it read below the root. The bench is the one
    the gate uses: the host's command read by ``component.bench_read``, else the pack's bench script. The boot and the
    bench are the run's time, within the deadline. A host that measured the root with the same client can pass its
    number as ``baseline_record`` instead (``{"mean": ...}``), and nothing is booted: ``_seed_baseline`` runs first,
    and a bar it seeded ends this.
    """
    from arbor.policy import claude_loop
    state = CampaignState.load(state_path)
    cfg = campaign.config
    if state.baseline is not None or state.iteration > 0 or component.stops_at_map(cfg):
        return
    obj = campaign.pack.objective
    meas = obj.get("measurement") or {}
    metric, secondary = obj.get("metric", "output_throughput"), meas.get("secondary_metric") or None
    host, read = component.bench_command(cfg), component.bench_read(cfg)
    if host and not read:
        print("[arbor] Root baseline: not measured: the host's bench command comes without bench_read, so the harness "
              "cannot read its report; iteration 1's measurement is the bar")
        return
    if not host and not cfg.get("bench_script"):
        print("[arbor] Root baseline: not measured: no bench_script; iteration 1's measurement is the bar")
        return
    server_cap = component.phase_seconds(cfg, "server")
    bench_cap = component.turn_seconds(cfg, "bench", floor_minutes=meas.get("bench_minutes"))
    if (server_cap is not None and server_cap <= 0) or (bench_cap is not None and bench_cap <= 0):
        print("[arbor] Root baseline: not measured: no time is left before the deadline")
        return
    repeats = max(1, int(meas.get("bench_repeats", cfg.get("bench_repeats", 1)) or 1))
    out = session_dir / "root_baseline"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[arbor] Root baseline: serving the root and benching it {repeats}x with the gate's bench: the bar for "
          f"iteration 1 ({out})", flush=True)

    def stop() -> None:
        claude_loop._stop_server(cfg, cfg.get("docker_container", ""), cfg.get("gpu_ids"))

    runs: list[dict] = []
    try:
        stop()
        component.phase("server")
        if not claude_loop._start_server(cfg) or not claude_loop._wait_healthy(cfg, int(server_cap or 600)):
            print("[arbor] Root baseline: not measured: the root did not serve; iteration 1's measurement is the bar")
            return
        component.phase("bench")
        port = cfg.get("port", 8888)
        for i in range(1, repeats + 1):
            run_dir = out if repeats == 1 else out / f"run{i}"
            run_dir.mkdir(parents=True, exist_ok=True)
            dest = run_dir / "bench_output.json"
            dest.unlink(missing_ok=True)
            if host:
                report = run_dir / "host_bench"
                report.mkdir(parents=True, exist_ok=True)
                script = (component.fill_bench(host, port, report) + "\n"
                          + claude_loop._read_block(report, str(dest), read, [metric]))
                env = dict(os.environ)
            else:
                script = f"bash {shlex.quote(str(cfg['bench_script']))}"
                env = {**os.environ, **claude_loop._bench_env(cfg, port, str(dest))}
            why = _run_bench_group(script, env, bench_cap, run_dir / "bench.log")
            res = _json_or(dest, {"status": "failed", "error": why or "the bench wrote no output"})
            res.setdefault("status", "ok")
            runs.append(res)
    finally:
        stop()
    ok = [r for r in runs if r.get("status") == "ok" and _number(r.get(metric))]
    if not ok:
        last = runs[-1] if runs else {}
        print(f"[arbor] Root baseline: not measured ({str(last.get('error', ''))[:200]}); iteration 1's measurement "
              f"is the bar")
        return
    value = sum(r[metric] for r in ok) / len(ok)
    secs = [r[secondary] for r in ok if secondary and _number(r.get(secondary))]
    state = CampaignState.load(state_path)
    state.record_baseline(float(value))
    if secs:
        state.secondary_name = secondary
        state.secondary_direction = meas.get("secondary_direction", "max")
        state.baseline_secondary = state.best_secondary = float(sum(secs) / len(secs))
    # The root's per-request records, so iteration 1's guards and second axis compare matched requests too.
    means = {m: sum(v) / len(v) for m in claude_loop._matched_metrics(meas)
             if (v := [r[m] for r in ok if _number(r.get(m))])}
    state.request_ref = claude_loop._request_ref(out, means, meas, len(state.lessons))
    state.iteration_notes["root_baseline"] = {"metric": metric, "value": value, "replays": len(ok),
                                              "runs": [r[metric] for r in ok]}
    state.save(state_path)
    print(f"[arbor] Root baseline: {metric}={value:.3f} over {len(ok)} replay(s)"
          + (f", {secondary}={state.baseline_secondary:.3f}" if secs else "") + ": the bar for iteration 1")


def _group_umask(session_dir) -> bool:
    """Under a group's shared storage, write what the group can read and write: the umask loses its group-write bit
    when the session directory is setgid and group-writable (a host's shared layout: directories 2775, files 664).

    A host may resume a run under another account of the same group. That account appends to the session's logs
    (``ledger.jsonl``, ``lifecycle.jsonl``, the trace) and replaces its files, and a file written 644 refuses it.
    Every child (the agents, the server, the bench) inherits the umask. True when it was changed."""
    import stat
    try:
        mode = os.stat(session_dir).st_mode
    except OSError:
        return False
    if not (mode & stat.S_ISGID and mode & stat.S_IWGRP):
        return False
    old = os.umask(0)
    os.umask(old & ~0o020)
    if old & 0o020:
        print(f"[arbor] The session directory is a group's (setgid, group-writable): umask {old:03o} -> "
              f"{old & ~0o020:03o}, so another account of the group can continue the run")
    return bool(old & 0o020)


def _book_unfinished(campaign, session_dir: Path, state_path: Path) -> int | None:
    """A run stopped between an iteration's start and its verdict without a SIGTERM (a kill, a crash, a lost node)
    left no row for it: the next start books it NOT MEASURED, with the intent and change it declared at its commit
    (``component.CLAIM_FILE``, as a host's SIGTERM books it), so it is neither lost nor its number reused. Its commit
    is then reverted by the pass's kept-tree refresh and noted for the revisit, as every NOT MEASURED change is.

    Only an iteration that committed its change: a pass that stopped before (its agent still working, a STOP found
    at its start) leaves no change to measure, and runs again under the same number. The iteration booked, or None."""
    import calendar

    started = lifecycle.last(session_dir, "event", "iteration_start") or {}
    n = started.get("iteration")
    if not isinstance(n, int) or isinstance(n, bool) or not state_path.exists():
        return None
    state = CampaignState.load(state_path)
    if n <= state.iteration or any(r.get("iteration") == n for r in state.lessons):
        return None
    claim = {}
    try:
        claim = json.loads((Path(session_dir) / f"iter_{n}" / component.CLAIM_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    change = str(claim.get("change") or "") if isinstance(claim, dict) else ""
    cfg = campaign.config
    if not change and cfg.get("docker_container") and cfg.get("workspace_root"):
        # Killed between the commit and its claim: the workspace's log names the change.
        try:
            from arbor.execute.workspace import git
            log = git(cfg["docker_container"], cfg["workspace_root"], "log", "-30", "--format=%s")
        except Exception:  # noqa: BLE001 - a workspace that cannot be read books nothing
            log = ""
        change = next((line.split(":", 1)[1].strip() for line in log.splitlines()
                       if line.startswith(f"iter-{n}:")), "")
    if not change:
        return None
    try:
        state.iteration_started = calendar.timegm(time.strptime(started.get("at", ""), "%Y-%m-%dT%H:%M:%SZ"))
    except (TypeError, ValueError):
        pass
    state.record_invalid(
        iteration=n, change=change[:200], tier=state.tier, findings=[],
        claim_class=str((claim or {}).get("claim_class") or "B"),
        reason=f"the run stopped before iteration {n}'s verdict (no SIGTERM: a kill, a crash or a lost node), and "
               f"the session was started again")
    state.save(state_path)
    print(f"[arbor] Iteration {n} never got its verdict: booked NOT MEASURED ({change[:120]})")
    return n


def _agent_model_line(config: dict) -> str | None:
    """The agents' model at start: the config's ``agent_model`` pins it for every turn (through ``ARBOR_AGENT_MODEL``,
    which the turns read), else that variable, else the CLI's default. Said once, and recorded on every lesson row
    (arbor/prompts.py); before, which model ran was only in the transcripts."""
    from arbor import prompts
    pinned = str(config.get("agent_model") or "").strip()
    if pinned:
        os.environ[prompts.MODEL_ENV] = pinned
        print(f"[arbor] Agent model: {pinned} (the config's agent_model, for every turn)")
    elif os.environ.get(prompts.MODEL_ENV, "").strip():
        pinned = os.environ[prompts.MODEL_ENV].strip()
        print(f"[arbor] Agent model: {pinned} ({prompts.MODEL_ENV}, for every turn)")
    else:
        print("[arbor] Agent model: not pinned (the CLI's default); each lesson row records the models its turns "
              "ran on")
    return pinned or None


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _json_or(path: Path, default: dict) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return dict(default)


def _run_bench_group(script: str, env: dict, cap: float | None, log: Path) -> str | None:
    """Run ``script`` under bash in a process group of its own, its output in ``log``; past ``cap`` seconds stop the
    whole group (a replay's client and its workers). None, or why it did not finish."""
    import signal
    import subprocess
    with open(log, "wb") as fh:
        proc = subprocess.Popen(["bash", "-c", script], env=env, stdout=fh, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            rc = proc.wait(timeout=cap)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=30)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            return f"the bench ran past its {cap / 60:.0f}-min cap"
    return None if rc == 0 else f"the bench exited {rc} (its log: {log})"


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
    # A host may give the record itself, an object, rather than its file (arbor/measure/accuracy.py reads both).
    p = Path("baseline_record") if isinstance(record_path, dict) else Path(record_path)
    if not isinstance(record_path, dict) and not p.is_file():
        print(f"[arbor] baseline_record not found: {p} — campaign will seed "
              f"from iteration 1 instead")
        return

    state = CampaignState.load(state_path)
    if state.baseline is not None:
        return  # already seeded; never overwrite a running campaign's bar

    try:
        rec = dict(record_path) if isinstance(record_path, dict) else json.loads(p.read_text())
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
    _group_umask(session_dir)
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
    # A session written in a format this Arbor does not continue (arbor/checkpoints.py, version.CONTINUES).
    from arbor import checkpoints
    refused += [r for r in checkpoints.refusals(session_dir) if r not in refused]
    # A prompt profile or override that names a file that is not there (arbor/prompts.py): refused now, not at the
    # first agent turn, and never a silent fall back to the built-in instructions.
    from arbor import prompts
    for role in prompts.ROLES:
        try:
            prompts.template(campaign.config, role, session_dir, prompts.base_dir(campaign))
        except prompts.PromptError as exc:
            refused.append(str(exc))
    if component.is_component(campaign):
        refused += [e for e in validate_campaign(campaign, allow_results=args.allow_pack_results)
                    if e not in refused]
    else:
        # The pack's own check standalone too, as warnings: a pack that runs standalone and that a host would refuse
        # (a tools/ directory once was) is found here, not first under the host. The config's keys are not checked:
        # standalone configs differ (a document campaign has no bench).
        for problem in validate_pack(campaign.pack, allow_results=args.allow_pack_results):
            print(f"[arbor] WARNING: a host would refuse this pack: {problem}", file=sys.stderr, flush=True)
    # The objective (arbor/objective.py): one Arbor cannot read, or another than the session started on.
    from arbor import objective as _objective
    wrong = _objective.problem(campaign, session_dir)
    if wrong and wrong not in refused:
        refused.append(wrong)
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
    # A standalone container must see the session where the bench writes: <session_mount>/<name>.
    # Checked at every start, a resume included, before any GPU time.
    if not refused and not component.is_component(campaign) and session_dir:
        from arbor.execute.workspace import session_mount_problem
        session_dir.mkdir(parents=True, exist_ok=True)
        problem = session_mount_problem(campaign.config.get("docker_container", ""), session_dir,
                                        campaign.config.get("session_mount", "/workspace/sessions"))
        if problem:
            refused.append(problem)
    if refused:
        return component.refuse(refused, campaign.config, session_dir)
    # One driver per session (arbor/checkpoints.py): the session lock, exclusive, once the config is accepted (a refusal
    # touches nothing but its record) and before the session's own records are written. A second driver waits a minute
    # for one that is ending (a rolling chunk's predecessor, a host's restart), then exits 73, lease_held; checkpoint
    # create, restore and fork refuse while a driver holds it.
    holder = checkpoints.hold_session(session_dir, held)
    if holder is not None:
        from arbor.cluster import handoff
        print(f"[arbor] Another driver holds this session ({holder}): exit {handoff.EXIT_LEASE_HELD}. Start the "
              f"same command again once it has ended.", file=sys.stderr, flush=True)
        return handoff.EXIT_LEASE_HELD
    # What this campaign optimizes, for a host to check before it trusts the run (arbor/objective.py).
    if session_dir:
        try:
            told = _objective.record(campaign, session_dir)
            if told["source"] != "pack":
                print(f"[arbor] Objective: {told['metric']} ({told['direction']}), asked as \"{told['asked']}\", "
                      f"in place of the pack's {told['pack_metric']}", file=sys.stderr, flush=True)
        except OSError as exc:
            print(f"[arbor] could not write {_objective.FILE}: {exc}", file=sys.stderr, flush=True)
        # How this start was asked for: its parameters, and which came from a payload (arbor/cli/payload.py).
        from arbor.cli import payload as _payload
        try:
            _payload.record(session_dir, args, "worker" if hasattr(args, "order") else
                            "run" if hasattr(args, "split") else "master")
        except OSError as exc:
            print(f"[arbor] could not write {_payload.RECORD}: {exc}", file=sys.stderr, flush=True)
    # The other editable trees (AITER) become git trees with a baseline, at every start, so each
    # iteration commits and reverts them with the workspace (arbor/execute/trees.py).
    if not component.is_component(campaign) and campaign.config.get("docker_container"):
        from arbor.execute import trees as _trees
        try:
            for tree, base in _trees.ensure(campaign.config["docker_container"], campaign.config,
                                            session_dir).items():
                print(f"[arbor] Tracking {tree} from {base[:12]}", file=sys.stderr, flush=True)
        except (RuntimeError, OSError) as exc:
            print(f"[arbor] WARNING: the editable trees are not tracked ({exc}): a change there is not "
                  f"committed or reverted with the iteration", file=sys.stderr, flush=True)

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
    # The kept point a REVERT or a NOT MEASURED iteration goes back to (arbor/execute/kept.py): this container's
    # record, or the tree as it stands at this start, after the seed commit and any restored KEEP.
    if campaign.config.get("docker_container") and campaign.config.get("workspace_root"):
        from arbor.execute import kept as _kept
        try:
            _kept.ensure(campaign.config["docker_container"], campaign.config, session_dir)
        except Exception as exc:   # noqa: BLE001 - without it, a REVERT reverts its own commit, as before
            print(f"[arbor] WARNING: no kept point recorded ({exc})", file=sys.stderr, flush=True)

    # A run split into commands (arbor run --split processes, a worker's order) keeps its first start, so a time
    # budget bounds the whole run, not each command. A resumed standalone session keeps what it already spent
    # (<session>/clock.json, written at every iteration's start): the budget bounds the session, not each launch.
    started = getattr(args, "started", None)
    spent = 0.0 if started or component.is_component(campaign) else _spent(session_dir)
    start_time = started or (time.time() - spent)
    if spent:
        print(f"[arbor] Resumed: {spent / 60:.0f} min of the time budget already spent ({CLOCK_FILE})")
    _agent_model_line(campaign.config)
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
        _root_baseline(campaign, state_path, comp, session_dir)
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
    print(f"[arbor] Pause: python3 -m arbor pause --session {session_dir}")
    print()
    # The session lock is held since the start (above): an iteration the last driver left without its verdict is booked.
    _book_unfinished(campaign, session_dir, state_path)
    paused = session_dir / checkpoints.PAUSED
    if paused.is_file() and state_path.exists():
        try:
            rec = json.loads(paused.read_text(encoding="utf-8"))
            print(f"[arbor] Resuming a session paused after iteration {rec.get('iteration')} "
                  f"(checkpoint {rec.get('checkpoint') or 'not written'}; {paused})")
        except (OSError, ValueError):
            pass

    s = Session(args=args, held=held, config_path=config_path, arbor_root=arbor_root, campaign=campaign,
                session_dir=session_dir, state_path=state_path, roll=roll, comp=comp, start_time=start_time,
                metric_name=metric_name, config_target=config_target, direction=direction,
                time_budget=time_budget, unit=unit, exit_reason=exit_reason, exit_detail=exit_detail)
    s.fire("session_start", mode=campaign.mode, arbor=version.provenance())   # which Arbor this start runs
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

    # `arbor pause` (arbor/checkpoints.py): at this boundary, the driver's loop stops the server, writes the checkpoint
    # pause-NNNN and exits 76; `arbor run` with the same config continues.
    from arbor import checkpoints
    asked = checkpoints.pause_requested(s.session_dir)
    if asked is not None:
        print(f"[arbor] PAUSE asked{(': ' + asked['reason']) if asked.get('reason') else ''} — pausing here.")
        s.exit_reason, s.exit_detail = "paused", str(asked.get("reason") or "")
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
    if not s.comp:
        _record_spent(s.session_dir, elapsed_min * 60.0)
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

    # The revisit phase (arbor/revisit.py): with every direction closed and time for one more iteration, a NOT
    # MEASURED change is measured once more before the round closes. Under a host or a rolling job the deadline was
    # checked above; a standalone run checks its time budget here.
    cfg = s.campaign.config
    if cfg.get("docker_container") and cfg.get("workspace_root"):
        try:
            ok, line = revisit.time_fits(state, None if s.comp else s.time_budget, s.start_time)
            armed = revisit.arm(cfg["docker_container"], cfg, s.session_dir, iteration, state, ok)
            if armed:
                print(f"[arbor] revisit: iteration {iteration} measures iteration {armed['of']}'s change again "
                      f"({line})")
        except Exception as exc:   # noqa: BLE001 - without it the round closes as it did before
            print(f"[arbor] WARNING: no revisit armed: {exc}")
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
    from arbor import checkpoints
    if s.comp:
        # Every component-mode exit says why (decision 1): `done` for STOP, max iterations
        # and an interrupt; `aborted` for ABORTED.md, a harness that stopped measuring, with
        # its message as the detail; `unreachable` (exit 3, so the host resumes the run) when
        # that harness was a model that could not be reached; the gate's reason when time ran
        # out; `mapped` when a dossier pass has set its map.
        rc = s.comp.finish(s.exit_reason, s.exit_detail)
    if s.exit_reason == "paused":
        print(f"[arbor] Paused ({s.exit_detail}); exit {checkpoints.EXIT_PAUSED}.")
        return checkpoints.EXIT_PAUSED
    print("[arbor] Done.")
    return rc
