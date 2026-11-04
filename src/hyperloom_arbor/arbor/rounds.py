"""The end of an optimization round: a same-node A/B, then the next round.

The target is not a STOP: it triggers an A/B and the next round of optimization,
and the "all plan items verdicted" trigger is unified with it. A campaign once
stopped at its early-exit target while its last KEEP still gained, and the number
that stopped it was the chained first-node scale (arbor/rebaseline.py), each
KEEP's single replay carried forward: an upper bound.

A round ends on either trigger:

- ``target``: the best on the bar's scale reaches the target (arbor/driver.py);
- ``map``: every direction on the map is closed (arbor/directions.py).

Then, before the next iteration:

1. **A/B.** The launcher's hook (``ARBOR_ROUND_AB``, or the config's
   ``round_ab.command``) serves the campaign's base tree and the current tree on this
   node in alternating rounds (scripts/model-m-campaign/round_ab.py) and writes a
   verdict to ``<out>/summary.json``.
2. **Confirmed:** the current tree's A/B mean becomes the bar on this node, the base
   tree's the baseline, and the chained scale starts again from them (a ``ROUND_AB``
   lesson; arbor/rebaseline.py resets there). The next target is 1.40x that bar. With
   map rounds on, the dossier update round follows and a new map starts the next round.
3. **Not confirmed** (the gain is within noise, below the base, or GSM8K fails): the
   campaign stops for the operator (``STOP``, ``ROUND_STOP.md`` with the evidence).
4. **Incomplete** (the A/B could not finish its rounds before the deadline): the round
   end stays pending; a rolling chain hands off and the next chunk runs it first.
   Three incomplete attempts stop for the operator, and so does an A/B that could not
   put the campaign's tree back in the container.

Without a hook the round end keeps today's behavior: the target exits the driver,
and an exhausted map goes straight to its update round. A round end left pending
when a launch sets no hook stops for the operator.

State lives in ``<session>/round_end.json``; only the harness writes it.

Parallel mode (arbor/parallel) is the same loop, faster: its round ends when every
lane has closed and no dependent lane waits; its integration is the A/B, and the next
round maps the integrated tree. A lane runs
no round end of its own: its config has no target and no round_ab, and its launch
sets ROUND_AB=0.

Component mode (arbor/component.py) has no round ends at all: the host gates and
merges what Arbor keeps, so ``hook`` is None whatever the env or the config say, the
driver reads no target, and an exhausted map goes straight to its update round.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

FILE = "round_end.json"
STOP_FILE = "ROUND_STOP.md"
STOP_PREFIX = "round-stop:"   # never "chunk-timer": scripts/cluster/chunk_lib.sh deletes those
ENV_HOOK = "ARBOR_ROUND_AB"
TRIGGERS = ("target", "map")
VERDICTS = ("confirmed", "not_confirmed", "incomplete")

KNOBS = {
    "round_target_ratio": 1.40,   # the next target, over the A/B bar (calibrate_baseline.py's ratio)
    "round_ab_minutes": 150,      # time an A/B needs; less left before a chunk's switch hands off
    "round_ab_attempts": 3,       # incomplete A/Bs before the round end stops for the operator
}


def log(msg: str) -> None:
    print(f"[arbor] round: {msg}", flush=True)


def knob(name: str, campaign=None) -> float:
    for src in ((getattr(campaign, "config", None) or {}).get("policy") or {},
                (getattr(getattr(campaign, "pack", None), "policy", None) or {})):
        if isinstance(src, dict) and name in src:
            try:
                return float(src[name])
            except (TypeError, ValueError):
                pass
    return float(KNOBS[name])


def hook(campaign=None) -> str | None:
    """The A/B command template, or None: the launcher's env first, then the config.

    None in component mode, whatever either says: the host gates and merges, and a
    round end would run an A/B and write STOP or ROUND_STOP.md on the host's node.
    """
    from arbor import component
    if campaign is not None and component.is_component(campaign):
        return None
    cmd = os.environ.get(ENV_HOOK, "").strip()
    if not cmd:
        cfg = (getattr(campaign, "config", None) or {}).get("round_ab") or {}
        cmd = (cfg.get("command") or "").strip() if isinstance(cfg, dict) else ""
    return cmd or None


# ── state ──────────────────────────────────────────────────────────────────────

def path(root) -> Path:
    return Path(root) / FILE


def new_doc() -> dict:
    return {"version": 1, "pending": None, "target": None, "ends": [], "stop": None}


def load(root) -> dict:
    p = path(root)
    if not p.exists():
        return new_doc()
    try:
        doc = json.loads(p.read_text())
    except (OSError, ValueError) as exc:
        log(f"{p} unreadable ({exc}) -- starting a new one")
        return new_doc()
    for k, v in new_doc().items():
        doc.setdefault(k, v)
    return doc


def save(root, doc: dict) -> None:
    p = path(root)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n")
    os.replace(tmp, p)


def pending(root) -> dict | None:
    return load(root).get("pending")


def target(root, metric: str):
    """The target the last confirmed A/B set for ``metric``, or None (the config's then holds)."""
    t = load(root).get("target") or {}
    v = t.get(metric)
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def request(root, state, trigger: str, detail: str, map_round: int | None = None) -> bool:
    """Open a round end. False when one is already pending (the first trigger stands)."""
    if trigger not in TRIGGERS:
        raise ValueError(f"trigger {trigger!r} is not one of {TRIGGERS}")
    doc = load(root)
    if doc.get("pending"):
        return False
    n = len(doc["ends"]) + 1
    doc["pending"] = {"end": n, "trigger": trigger, "detail": detail, "map_round": map_round,
                      "requested_at": state.iteration, "ts": time.time(), "attempts": []}
    doc["stop"] = None
    save(root, doc)
    _annotate(state, n, "requested", f"round end {n} ({trigger}): {detail}")
    log(f"end {n} requested by the {trigger}: {detail}")
    return True


def _annotate(state, n: int, phase: str, change: str, **extra) -> None:
    state.lessons.append({"iteration": state.iteration, "decision": "ROUND_END", "round_end": n,
                          "phase": phase, "change": change, "tier": getattr(state, "tier", None),
                          "ts": time.time(), **extra})


# ── the A/B ────────────────────────────────────────────────────────────────────

def out_dir(root, n: int, attempt: int = 1) -> Path:
    job = os.environ.get("SLURM_JOB_ID", "local")
    rst = os.environ.get("SLURM_RESTART_COUNT", "0")
    return Path(root) / "rounds" / f"end_{n}" / "ab" / f"{job}.r{rst}.a{attempt}"


def run_ab(campaign, root, deadline: float, timeout_s: float | None = None) -> dict:
    """Run the hook for the pending end; its summary, or an ``incomplete`` one on failure.

    The hook is a command template: ``{out}`` is this attempt's directory, ``{deadline}``
    the epoch second by which its last measured pair must end.
    """
    doc = load(root)
    pend = doc.get("pending")
    if not pend:
        raise RuntimeError("no round end is pending")
    cmd = hook(campaign)
    if not cmd:
        raise RuntimeError(f"no A/B hook ({ENV_HOOK}, or the config's round_ab.command)")
    out = out_dir(root, pend["end"], len(pend.get("attempts") or []) + 1)
    if out.exists():
        # An attempt the driver never booked (it died mid-A/B): its files are not this one's.
        out.rename(out.with_name(f"{out.name}.stale-{time.time_ns()}"))
    out.mkdir(parents=True)
    os.chmod(out, 0o770)
    t0 = time.time()
    try:
        argv = shlex.split(cmd.format(out=str(out), deadline=int(deadline)))
    except (KeyError, IndexError, ValueError) as exc:
        return {"verdict": "incomplete", "reason": f"the hook {cmd!r} does not format: {exc!r}",
                "out": str(out), "rc": None, "minutes": 0.0}
    log(f"A/B for end {pend['end']}: {' '.join(argv)}")
    try:
        p = subprocess.run(argv, timeout=timeout_s or max(60.0, deadline - time.time() + 900))
        rc = p.returncode
    except (OSError, subprocess.TimeoutExpired) as exc:
        rc, summary = -1, {"verdict": "incomplete", "reason": f"hook failed: {exc}"}
    else:
        try:
            summary = json.loads((out / "summary.json").read_text())
        except (OSError, ValueError) as exc:
            summary = {"verdict": "incomplete", "reason": f"no summary.json (rc {rc}): {exc}"}
    if summary.get("verdict") not in VERDICTS:
        summary = {**summary, "verdict": "incomplete",
                   "reason": f"verdict {summary.get('verdict')!r} is not one of {VERDICTS}"}
    summary.update(out=str(out), rc=rc, minutes=round((time.time() - t0) / 60, 1))
    return summary


def apply(campaign, state, root, summary: dict) -> str:
    """Book an A/B's verdict: "next" (the next round starts), "pending" or "stop"."""
    doc = load(root)
    pend = doc.get("pending")
    if not pend:
        raise RuntimeError("no round end is pending")
    n, verdict = pend["end"], summary.get("verdict")
    pend["attempts"].append({k: summary.get(k) for k in ("verdict", "reason", "out", "rc", "minutes",
                                                          "rounds", "gain", "node", "restore_failed")})
    if summary.get("restore_failed"):
        # The container no longer serves the campaign's tree: no iteration may run on it.
        save(root, doc)
        return stop(state, root, f"round end {n}'s A/B could not put the campaign's tree back "
                                 f"({summary.get('reason')}); the container is not the session's tree")
    if verdict == "incomplete":
        limit = int(knob("round_ab_attempts", campaign))
        if len(pend["attempts"]) >= limit:
            save(root, doc)
            return stop(state, root, f"round end {n}'s A/B did not complete in {limit} attempts "
                                     f"(last: {summary.get('reason')})")
        save(root, doc)
        _annotate(state, n, "incomplete", f"round end {n}: A/B incomplete ({summary.get('reason')}); "
                                          f"attempt {len(pend['attempts'])} of {limit}")
        log(f"end {n}: A/B incomplete ({summary.get('reason')}) -- still pending")
        return "pending"

    metric = summary.get("metric")
    gain = summary.get("gain")
    text = (f"A/B on {summary.get('node')} ({summary.get('rounds')} measured rounds): {metric} base "
            f"{summary.get('base_mean')} vs current {summary.get('current_mean')} "
            f"({100 * gain:+.2f}%)" if isinstance(gain, (int, float)) else f"A/B: {summary}")
    if verdict == "not_confirmed":
        save(root, doc)
        return stop(state, root, f"round end {n}'s A/B did not confirm the gain: {text}; "
                                 f"{summary.get('reason', '')}".rstrip("; "))

    # Confirmed: the A/B's two means are the bar and the baseline on this node, and the
    # chained scale restarts here (scale 1.0, which the dashboard reads off the newest
    # lesson that carries one).
    new = summary["current_mean"]
    new_sec = summary.get("current_secondary_mean")
    base, base_sec = summary.get("base_mean"), summary.get("base_secondary_mean")
    old, old_sec = state.best_metric, getattr(state, "best_secondary", None)
    old_base = state.baseline
    state.best_metric = new
    if new_sec is not None and hasattr(state, "best_secondary"):
        state.best_secondary = new_sec
    if isinstance(base, (int, float)):
        state.baseline = base
    if isinstance(base_sec, (int, float)) and hasattr(state, "baseline_secondary"):
        state.baseline_secondary = base_sec
    ratio = knob("round_target_ratio", campaign)
    tgt = {metric: round(new * ratio, 5)}
    if new_sec is not None and summary.get("secondary_metric"):
        tgt[summary["secondary_metric"]] = round(new_sec * ratio, 5)
    state.lessons.append({
        "iteration": state.iteration, "decision": "ROUND_AB", "round_end": n,
        "trigger": pend["trigger"], "metric": new, "secondary": new_sec,
        "bar_before": old, "secondary_before": old_sec, "base_mean": base,
        "base_secondary_mean": base_sec, "baseline_before": old_base, "scale": 1.0,
        "secondary_scale": 1.0, "gain": gain, "verdict": verdict, "node": summary.get("node"),
        "record": summary.get("out"), "target_after": tgt, "tier": getattr(state, "tier", None),
        "ts": time.time(),
        "change": (f"round end {n} ({pend['trigger']}) confirmed. {text}. On this node the bar "
                   f"is now the current tree's A/B mean {new} (was {old}) and the baseline the "
                   f"base tree's {base} (was {old_base}); the chained scale restarts here. "
                   f"Next target {tgt.get(metric)} ({ratio:g}x)."),
    })
    doc["ends"].append({**pend, "verdict": verdict, "ab": {k: summary.get(k) for k in (
        "metric", "base_mean", "current_mean", "gain", "rounds", "node", "secondary_metric",
        "base_secondary_mean", "current_secondary_mean", "gsm8k", "out", "digest")},
        "bar_before": old, "bar_after": new, "target_after": tgt, "closed_ts": time.time()})
    doc["pending"] = None
    doc["target"] = tgt
    save(root, doc)
    log(f"end {n} confirmed: bar {old} -> {new}, next target {tgt}")
    return "next"


def stop(state, root, reason: str) -> str:
    """Stop the campaign for the operator: STOP, and ROUND_STOP.md with the evidence."""
    root = Path(root)
    doc = load(root)
    doc["stop"] = {"reason": reason, "iteration": state.iteration, "ts": time.time()}
    save(root, doc)
    ends = "\n".join(f"| {e['end']} | {e['trigger']} | {e.get('verdict')} | "
                     f"{(e.get('ab') or {}).get('gain')} | {e.get('bar_after')} |" for e in doc["ends"]) or "| – | | | | |"
    tries = "\n".join(f"- {json.dumps(a)}" for a in (doc.get("pending") or {}).get("attempts", [])) or "- none"
    (root / STOP_FILE).write_text(
        f"# The round end stopped the campaign\n\n**Why:** {reason}.\n\n"
        f"Stopped after iteration {state.iteration}.\n\n## Earlier round ends\n\n"
        f"| End | Trigger | Verdict | Gain | Bar after |\n|---|---|---|---|---|\n{ends}\n\n"
        f"## This end's A/B attempts\n\n{tries}\n\n"
        f"## To resume\n\nReview the round's KEEPs against the A/B evidence (each attempt's directory "
        f"holds results.jsonl and summary.md). Then remove `STOP` and, in a rolling chain, "
        f"`chain/CHAIN_STOP`, and launch again: the pending round end runs its A/B again first, unless "
        f"you clear `pending` in {FILE}.\n")
    (root / "STOP").write_text(f"{STOP_PREFIX} {reason}\n")
    n = (doc.get("pending") or {}).get("end", 0)
    _annotate(state, n, "stop", reason)
    print(f"[arbor] round: STOP -- {reason} (see {root / STOP_FILE})", file=sys.stderr, flush=True)
    return "stop"
