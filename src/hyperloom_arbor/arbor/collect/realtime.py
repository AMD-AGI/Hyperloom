"""Realtime collector: a campaign's live state, as one JSON snapshot.

Ported from the Cluster A campaign dashboard's collector
(``scripts/arbor_dashboard/collect.py`` on ``<fork>/dsv41-v2.1-ruppell``,
collector version 1), with the same top-level keys so that page can render a
snapshot unchanged. What differs: it runs where the campaign runs, writes the
snapshot to a local directory, and pushes it to the share (``share.py``); it
never samples GPU nodes (a probe during a measured replay perturbs it), and it
takes every path as an argument instead of a Cluster A layout.

Standard library only, and it never raises: every section records its own
failure in ``errors`` so a partial snapshot still renders.

Whether the campaign runs: by default, an ``arbor run`` process on this host.
A campaign whose driver runs inside a Slurm job (a chain of chunks on GPU
nodes) is invisible to that check from a login node, so ``slurm_job`` names
the job instead and its state decides: RUNNING is running, only a PENDING
successor is between chunks, and a ``squeue`` that fails is unknown, never
stopped.

This file is also a script: ``python3 realtime.py --session ...`` prints one
snapshot. The dashboard's local mode sends it inside its ssh command, so the
host needs only python3, not an Arbor checkout.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

COLLECTOR_VERSION = 5   # 1 is the dashboard's collect.py; 2 adds `share`; 3 the map; 4 a lane's stages;
#                         5 the warm chain, the round-end A/B's attempts, a job's restarts


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _mtime(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except OSError:
        return None


def tail(path: Path | None, lines: int, max_bytes: int = 2_000_000) -> list[str]:
    # Last ``lines`` lines without reading a multi-megabyte log end to end.
    if path is None:
        return []
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            if size > max_bytes:
                fh.seek(size - max_bytes)
                fh.readline()  # discard the partial line
            data = fh.read()
        return data.decode("utf-8", errors="replace").splitlines()[-lines:]
    except OSError:
        return []


def _stop_kind(text: str) -> str:
    if text.startswith("chunk-timer"):
        return "chunk"
    if text.startswith("lane-closed"):
        return "lane"
    if text.startswith("round-stop"):
        return "round"
    return "map" if text.startswith("map-stop") else "operator"


def _map(path: Path, errors: list[str]) -> dict | None:
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text())
        doc.pop("events", None)
        return doc
    except Exception as exc:  # noqa: BLE001 - never raise
        errors.append(f"{path.name} unreadable: {exc}")
        return None


def warm_chain(session: Path, lessons: list, end):
    # The first node's scale as arbor/rebaseline.py chains it, but with a warm reading on
    # both sides of each step: the last KEEP's or bar's on the old node over the new
    # node's bar. A rebased bar's warm reading is its record's replays after the
    # container's first, which reads low (scripts/model-m-campaign/rebase_bar.py), unless
    # the record discarded a warm-up itself. Each lesson gets the factor in force after it;
    # returned: the factor now, and the best on this node, warm. Model M day-0 b0's chain
    # read 1.438x the Day-0 bar this way 1.237x, and its round-end A/B 1.24x.
    scale = 1.0
    for x in lessons:
        d, m = x.get("decision"), x.get("metric")
        if d == "REBASELINE":
            try:
                r = json.loads((session / "bars" / Path(x["record"]).name).read_text())
                w = r["runs"] if r.get("warmup_runs") or len(r["runs"]) < 2 else r["runs"][1:]
                m = x["warm_metric"] = sum(w) / len(w)
            except Exception:  # noqa: BLE001 - no record: the lesson's own bar
                pass
            scale *= end / m if end and m else 1
        if d == "ROUND_AB":
            scale = 1.0
        if m and (d in ("REBASELINE", "ROUND_AB") or d == "KEEP" and x.get("booked") is not False):
            end = m  # a frontier KEEP is not booked: the bar stays
        x["warm_scale"] = scale
    return scale, end


def collect_campaign(session: Path, errors: list[str]) -> dict:
    state_path = session / "session.json"
    out: dict = {"present": state_path.is_file(), "session_root": str(session)}
    if not out["present"]:
        return out
    try:
        st = json.loads(state_path.read_text())
    except Exception as exc:  # noqa: BLE001 - never raise
        errors.append(f"session.json unreadable: {exc}")
        return out
    lessons = st.get("lessons") or []
    counts: dict[str, int] = {}
    for item in lessons:
        d = str(item.get("decision", "?"))
        counts[d] = counts.get(d, 0) + 1
    stop = session / "STOP"
    out.update({
        "identity": st.get("identity"),
        "pack": st.get("pack_name"),
        "goal": st.get("goal"),
        "iteration": st.get("iteration"),
        "tier": st.get("tier"),
        "metric_name": st.get("metric_name"),
        "metric_unit": st.get("metric_unit"),
        "baseline": st.get("baseline"),
        "best_metric": st.get("best_metric"),
        "secondary_name": st.get("secondary_name"),
        "secondary_direction": st.get("secondary_direction"),
        "baseline_secondary": st.get("baseline_secondary"),
        "best_secondary": st.get("best_secondary"),
        "null_streak": st.get("null_streak"),
        "counts": counts,
        # Newest first: a long campaign accumulates hundreds of entries.
        "lessons": list(reversed(lessons))[:60],
        "stop_file": stop.is_file(),
        "stop_kind": (_stop_kind(stop.read_text(errors="replace")) if stop.is_file() else None),
        "chain_stop": (session / "chain" / "CHAIN_STOP").is_file(),
        "aborted_file": (session / "ABORTED.md").is_file(),
        "mtime": _mtime(state_path),
        # The map (arbor/directions.py): the dashboard's directions checklist.
        "directions": _map(session / "directions.json", errors),
        # A lane's stage records (arbor/stages.py): every attempt, and what waits to be judged.
        "stages": _map(session / "stages.json", errors),
        # Round ends (arbor/rounds.py): a pending A/B, the ones booked, a stop.
        "round_end": _map(session / "round_end.json", errors),
    })
    try:
        out["round_ab"] = collect_round_ab(session, out["round_end"], st.get("metric_name"), st.get("secondary_name"))
        if counts.get("REBASELINE"):
            out["warm_scale"], out["warm_best"] = warm_chain(session, lessons, st.get("baseline"))
    except Exception as exc:  # noqa: BLE001 - never raise
        errors.append(f"round-end A/B or warm chain: {exc}")
    for base, best, key in (("baseline", "best_metric", "gain_pct"),
                            ("baseline_secondary", "best_secondary", "secondary_gain_pct")):
        if out[base] and out[best]:
            try:
                out[key] = (out[best] / out[base] - 1.0) * 100.0
            except (TypeError, ZeroDivisionError):
                pass
    return out


PHASE_RULES = [
    # In the order they occur inside an iteration: the last match in the log
    # is the current phase.
    (r"^\[arbor\] === Iteration (\d+) ===", "iteration_start", "iteration starting"),
    (r"^\[arbor\] Running agent", "agent", "agent working"),
    (r"^\[arbor\] PROBE: scan recorded", "probe", "probe recorded (no server run)"),
    (r"^\[arbor\] Committed:", "committed", "change committed, preparing to measure"),
    (r"^\[arbor\] Agent starting server", "server", "starting the server"),
    (r"^\[arbor\] GPUs clean", "gpu_check", "GPU preflight"),
    (r"^\[arbor\] Correctness:", "correctness", "correctness check"),
    (r"^\[warmed\] warmup pass", "warmup", "warmup replay (discarded)"),
    (r"^\[warmed\] measured pass", "measuring", "measured replay"),
    (r"^\[arbor\] Agent running bench", "bench", "benchmark running"),
    (r"^\[arbor\] Bench result:", "bench_done", "benchmark finished"),
    (r"^\[arbor\] Running critic", "critic", "critic reviewing"),
    (r"^\[arbor\] Gate: ", "gate", "gate decided"),
    (r"^\[arbor\] Cooling down", "cooldown", "cooling down between iterations"),
    (r"^\[arbor\] roll: handing off", "handoff", "handing over to the next job"),
    # Between iterations: a chunk's start, its bar (scripts/model-m-campaign/campaign_model_m.sbatch),
    # a round end's A/B (round_ab.py; the page shows the A/B's last line).
    (r"^\[chunk .* takes over", "chunk", "a new chunk starts"),
    (r"^\[campaign\] .*(measuring the bar|\[rebase\])", "rebase", "measuring this node's bar"),
    (r"^\[campaign\] .*driver starting", "driver", "driver starting"),
    (r"^\[(round-ab\]|arbor\] round: A/B)", "round_ab", "round-end A/B"),
]


def _running(pattern: str) -> bool:
    try:
        p = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True, timeout=5)
        return p.returncode == 0 and bool(p.stdout.strip())
    except (OSError, subprocess.TimeoutExpired):
        return False


def _lease_holder(session):
    try:
        lease = json.loads((session / "chain" / "lease.json").read_text())
        return None if lease.get("released_at") else str(lease.get("job"))
    except (OSError, ValueError, TypeError):
        return None


def collect_jobs(slurm_job: str, errors: list[str], session: Path | None = None) -> tuple[dict, dict]:
    # The campaign's Slurm jobs by name, and the liveness they imply.
    #
    # A chained campaign has one RUNNING chunk, or none and a PENDING successor
    # between chunks; a rolling one (handoff.py) runs two while one waits. ``running``
    # is None when squeue could not be read: an unanswered check is unknown, never idle.
    # (Comments, not docstrings, in this file: the dashboard ships it minified inside an
    # ssh command, under Git for Windows' 8,000 characters, and a docstring is code.)
    jobs: dict = {"ours": [], "other": []}
    live = {"running": None, "between_chunks": False, "liveness": f"Slurm job {slurm_job}"}
    try:
        p = subprocess.run(["squeue", "-h", "-u", getpass.getuser(), "-n", slurm_job,
                            "-o", "%i|%T|%N|%M|%L|%r"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        errors.append(f"squeue for {slurm_job} failed: {exc}")
        return jobs, live
    if p.returncode != 0:
        errors.append(f"squeue for {slurm_job} exited {p.returncode}: {(p.stderr or p.stdout).strip()[:200]}")
        return jobs, live
    for line in p.stdout.splitlines():
        f = (line.split("|") + [""] * 6)[:6]
        jobs["ours"].append({"id": f[0], "name": slurm_job, "state": f[1], "node": f[2], "elapsed": f[3],
                             "limit": f[4], "reason": f[5],
                             "role": "campaign" if f[1] == "RUNNING" else "next chunk"})
    running = [j for j in jobs["ours"] if j["state"] == "RUNNING"]
    if len(running) > 1:
        holder = _lease_holder(session)
        ids = [j["id"] for j in running]
        main = holder if holder in ids else min(ids, key=lambda i: int(i) if i.isdigit() else 0)
        for j in running:
            j["role"] = "campaign" if j["id"] == main else "successor"
    for j in jobs["ours"]:
        try:  # a preempted job is requeued under its own id (Cluster A's <partition>): Restarts counts them
            m = re.search(r"Restarts=(\d+)", subprocess.run(["scontrol", "show", "job", j["id"]],
                                                            capture_output=True, text=True, timeout=15).stdout)
            j["restarts"] = int(m.group(1)) if m else None
        except (OSError, subprocess.TimeoutExpired):
            pass
    states = {j["state"] for j in jobs["ours"]}
    live["running"] = "RUNNING" in states
    live["between_chunks"] = not live["running"] and bool(states & {"PENDING", "CONFIGURING", "COMPLETING"})
    return jobs, live


def collect_log(session: Path, log: Path | None, errors: list[str],
                live: dict | None = None) -> dict:
    lines = tail(log, 4000)
    if not lines:
        return {"present": bool(log and log.is_file()), "phase": None, "tail": [],
                "verdicts": [], "supervisor": dict(live or {})}
    phase = None
    current_iteration = None
    verdicts = []
    for line in lines:
        for pattern, key, label in PHASE_RULES:
            if re.match(pattern, line):
                phase = {"key": key, "label": label, "line": line[:300]}
        m = re.match(r"^\[arbor\] === Iteration (\d+) ===", line)
        if m:
            # The iteration being worked, which session.json's `iteration` (the
            # last one the gate counted) does not show.
            current_iteration = int(m.group(1))
        # The gate line is the authority: "[arbor] Gate: KEEP (B) — reason".
        m = re.match(r"^\[arbor\] Gate: (\w+)(?: \(([^)]*)\))? — (.*)$", line)
        if m:
            verdicts.append({"decision": m.group(1), "claim_class": m.group(2),
                             "reason": m.group(3)[:240]})
    # Supervisor lines ([sup]), scoped to the run that started last: the log is
    # append-only across restarts, so an earlier run's "campaign complete"
    # would otherwise keep matching.
    all_sup = [l for l in lines if l.startswith("[sup]")]
    start = max([i for i, l in enumerate(all_sup) if "supervisor start" in l], default=0)
    sup = all_sup[start:]
    terminal = None
    for l in sup:
        for needle in ("campaign complete", "deadline reached", "giving up",
                       "STOP file present", "ABORTED.md present"):
            if needle in l:
                terminal = needle
    # Liveness beats log archaeology: if an Arbor driver runs (here, or in the
    # campaign's Slurm job), the campaign runs, whatever the log's history says.
    if live is None:
        live = {"running": _running("arbor run"), "between_chunks": False,
                "liveness": f"arbor run process on {socket.gethostname()}"}
    running = live["running"]
    if running or live["between_chunks"]:
        terminal = None
    iter_started = _mtime(session / f"iter_{current_iteration}") if current_iteration is not None else None
    return {
        "present": True,
        "phase": phase,
        "current_iteration": current_iteration,
        "iteration_dir_mtime": iter_started,
        "invalid_count": sum(1 for l in lines if l.startswith("[arbor] Gate: INVALID")),
        "not_measured_count": sum(1 for l in lines if l.startswith("[arbor] NOT MEASURED")),
        "log_mtime": _mtime(log),
        "tail": lines[-120:],
        "verdicts": list(reversed(verdicts))[:40],
        "supervisor": {"attempts": sum(1 for l in sup if "=== launching arbor" in l),
                       "restarts": sum(1 for l in sup if "restarting in" in l),
                       "terminal": terminal, "running": running,
                       "between_chunks": live["between_chunks"], "liveness": live["liveness"],
                       "runs_in_log": sum(1 for l in all_sup if "supervisor start" in l),
                       "last": sup[-6:]},
        "agent_tail": [l for l in lines if l.startswith("[agent]")][-25:],
    }


def collect_round_ab(session: Path, rt: dict | None, metric, secondary):
    # The pending (else the last) round end's A/B (scripts/model-m-campaign/round_ab.py),
    # attempt by attempt, oldest first: each row [round, arm, concurrency, metric,
    # secondary], or the current arm's full GSM8K as a dict; the node and verdict once
    # the attempt wrote its summary.
    rt = rt or {}
    end = (rt.get("pending") or {}).get("end") or (rt.get("ends") or [{}])[-1].get("end")
    tries = []
    for f in sorted((session / "rounds" / f"end_{end}" / "ab").glob("*/results.jsonl"), key=os.path.getmtime):
        s = _map(f.parent / "summary.json", []) or {}
        rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        tries.append({"dir": f.parent.name, "node": s.get("node"), "verdict": s.get("verdict"),
                      "rows": [r.get("gsm8k_full") or [r.get(k) for k in ("round", "arm", "concurrency", metric, secondary)]
                               for r in rows]})
    return tries


def collect_baselines(results: Path | None, errors: list[str]) -> dict:
    out: dict = {}
    if results is None:
        return out
    for p in sorted(results.glob("*baseline*.json")):
        try:
            d = json.loads(p.read_text())
            out[p.stem] = {k: d.get(k) for k in (
                "concurrency", "mean", "median", "runs", "relative_spread",
                "min_credible_gain", "mean_secondary", "relative_spread_secondary",
                "requests_successful", "duration_seconds")}
        except Exception as exc:  # noqa: BLE001
            errors.append(f"baseline {p.name} unreadable: {exc}")
    return out


def collect_target(config: dict, metric: str | None, log: Path | None,
                   secondary: str | None = None, session: Path | None = None) -> dict:
    # The target as the config states it (resolved as the driver does) and as
    # the running driver last printed it; the secondary metric's too, when the
    # config's target names it. A confirmed round-end A/B's target (arbor/rounds.py,
    # round_end.json) replaces the config's, as it does in the driver.
    out: dict = {"value": None, "set_by": None, "source": None, "driver": None,
                 "secondary_value": None}
    t = config.get("target")
    metric = metric or "output_throughput"
    if isinstance(t, (int, float)) and not isinstance(t, bool):
        out["value"] = float(t)
    elif isinstance(t, dict):
        for key in (metric, f"{metric}_per_chip", "output_tok_per_chip", "value"):
            v = t.get(key)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out["value"] = float(v)
                break
        v = t.get(secondary) if secondary else None
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out["secondary_value"] = float(v)
        out["set_by"], out["source"] = t.get("_set_by"), t.get("_source")
    rt = _map(session / "round_end.json", []) if session else None
    rtt = (rt or {}).get("target") or {}
    if isinstance(rtt.get(metric), (int, float)) and not isinstance(rtt.get(metric), bool):
        ends = (rt or {}).get("ends") or [{}]
        out["value"], out["set_by"] = float(rtt[metric]), "round"
        out["source"] = f"round end {ends[-1].get('end')}'s A/B"
        v = rtt.get(secondary) if secondary else None
        out["secondary_value"] = float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None
    for line in reversed(tail(log, 4000)):
        m = re.match(r"^\[arbor\] Target: ([0-9.]+)", line)
        if m:
            out["driver"] = float(m.group(1))
            break
    return out


def snapshot(session: Path, log: Path | None = None, config: dict | None = None,
             results: Path | None = None, events: Path | None = None,
             slurm_job: str | None = None) -> dict:
    errors: list[str] = []
    campaign = collect_campaign(session, errors)
    jobs, live = collect_jobs(slurm_job, errors, session) if slurm_job else ({"ours": [], "other": []}, None)
    return {
        "collector_version": COLLECTOR_VERSION,
        "collected_at": now_iso(),
        "host": socket.gethostname(),
        "root": str(session.parent),
        "session_dir": str(session),
        "jobs": jobs,
        "campaign": campaign,
        "log": collect_log(session, log, errors, live),
        "baselines": collect_baselines(results, errors),
        "node": {"ok": False, "gpus": [], "containers": [],
                 "error": "the realtime collector does not sample nodes: a probe "
                          "during a measured replay perturbs it"},
        "ab": {},
        "events": tail(events, 40),
        "target": collect_target(config or {}, campaign.get("metric_name"), log,
                                 campaign.get("secondary_name"), session),
        "errors": errors,
    }


def write_json(path: Path, data: dict) -> None:
    # Write in one step: a reader of the local copy never sees half a file.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, default=str) + "\n")
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    """One snapshot on stdout, writing nothing: what the dashboard's local mode runs."""
    p = argparse.ArgumentParser(prog="realtime.py", description=main.__doc__)
    p.add_argument("--session", required=True)
    p.add_argument("--log")
    p.add_argument("--config")
    p.add_argument("--results")
    p.add_argument("--events")
    p.add_argument("--slurm-job", help="the campaign's Slurm job name: its state is the liveness")
    p.add_argument("--campaign", help="the name to show (default: the session directory's)")
    a = p.parse_args(argv)
    session = Path(a.session)
    config: dict = {}
    if a.config:
        try:
            config = json.loads(Path(a.config).read_text())
        except (OSError, ValueError):
            config = {}
    snap = snapshot(session, Path(a.log) if a.log else None, config,
                    Path(a.results) if a.results else None, Path(a.events) if a.events else None,
                    a.slurm_job)
    snap["share"] = {"campaign": a.campaign or session.name, "share_host": None, "destination": None,
                     "share_ok": None, "pushed_at": None, "last_error": None}
    print(json.dumps(snap, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
