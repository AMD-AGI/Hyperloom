"""A same-node A/B of two trees of a campaign's workspace, for any pack.

Two uses:

- **A round end's hook** (arbor/rounds.py): the config's ``round_ab.command``, or ``ARBOR_ROUND_AB``, set to
  ``python3 -m arbor.ab --config <config> --out {out} --deadline {deadline}``. The base is the workspace's first
  commit, the current tree its HEAD. Before this, the only hook was one campaign launcher's own round_ab script.
- **An integration's A/B:** ``--base`` and ``--current`` name the two commits, such as the tree integrated so far and
  that tree with one worker's KEEPs merged in.

**An arm is a commit of the workspace.** Its ``arbor/config.json`` (the server's flags and env, which Arbor tracks with
the source) comes with it. A commit without one serves the campaign's own ``server_flags`` and ``server_env``. Each arm
is served by the harness's own helpers (``_start_server``, ``_wait_healthy``, ``_stop_server``), and measured by the
pack's bench script; the current arm's full GSM8K, by the pack's correctness script.

**The rounds** (the estimator of that launcher's hook, which this one generalizes):
- **Round 0** warms each arm (its first server pays the compile), and runs the full GSM8K on the current arm.
- **Rounds 1..N** are measured, alternating which arm goes first (B C, C B, ...), while a pair still fits before the
  deadline.
- **The verdict** (``judge``):
  - ``confirmed``: the current arm is ahead in every pair, its mean gain is above the floor (the pack's
    ``threshold_rel`` and ``noise_band_pct``, or either arm's spread, whichever is largest), and it passes the full
    GSM8K;
  - ``not_confirmed``: any of those fails;
  - ``incomplete``: too few pairs, or no GSM8K result.

**The tree is put back** however the A/B ends: the workspace returns to the commit it started on. When that fails,
``restore_failed`` is set and arbor/rounds.py stops the campaign. A workspace with uncommitted changes is refused: a
checkout would lose them.

The summary goes to ``<out>/summary.json``, in the fields arbor/rounds.py reads, and ``<out>/summary.md``; every
measured row to ``<out>/results.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import sys
import time
from pathlib import Path

from arbor.measure.accuracy import FULL_GSM8K

#: Exit code when the workspace could not be put back on its tree (the caller must not serve from it).
EXIT_RESTORE = 3


class Fail(Exception):
    """The A/B cannot go on: the summary says why, as ``incomplete``."""


def log(msg: str) -> None:
    print(f"[arbor] A/B: {msg}", flush=True)


def better(x: float, y: float, direction: str) -> float:
    """x's gain over y: positive when x is better."""
    return (x / y - 1) if direction != "min" else (y / x - 1)


def spread(values: list[float]) -> float:
    return (max(values) - min(values)) / statistics.fmean(values) if len(values) > 1 else 0.0


def judge(rows: list[dict], metric: str, direction: str, secondary: str | None, floors: dict[str, float],
          gsm: dict | None, min_rounds: int) -> dict:
    """The verdict from the measured rows (round >= 1), as the launcher's round_ab script judges, at one point."""
    by_round: dict[int, dict] = {}
    for r in rows:
        if r.get("round", 0) >= 1 and r.get("status") == "ok" and isinstance(r.get(metric), (int, float)):
            by_round.setdefault(r["round"], {})[r["arm"]] = r
    full = [k for k in sorted(by_round) if {"base", "current"} <= set(by_round[k])]
    b = [by_round[k]["base"][metric] for k in full]
    c = [by_round[k]["current"][metric] for k in full]
    out: dict = {"metric": metric, "direction": direction, "rounds": len(full), "base": b, "current": c}
    if len(full) < min_rounds:
        return {**out, "verdict": "incomplete", "reason": f"{len(full)} measured pair(s), {min_rounds} needed"}
    bm, cm = statistics.fmean(b), statistics.fmean(c)
    per = [better(ci, bi, direction) for bi, ci in zip(b, c)]
    gain = better(cm, bm, direction)
    parts = {**floors, "base_spread": spread(b), "current_spread": spread(c)}
    floor = max(parts.values())
    out.update(base_mean=round(bm, 5), current_mean=round(cm, 5), gain=round(gain, 5),
               per_round_gain=[round(g, 5) for g in per], floor=round(floor, 5),
               floor_parts={k: round(v, 5) for k, v in parts.items()}, gsm8k=gsm)
    if secondary:
        bs = [by_round[k]["base"].get(secondary) for k in full]
        cs = [by_round[k]["current"].get(secondary) for k in full]
        if all(isinstance(v, (int, float)) for v in bs + cs):
            out.update(secondary_metric=secondary, base_secondary_mean=round(statistics.fmean(bs), 5),
                       current_secondary_mean=round(statistics.fmean(cs), 5))
    if gsm is None or gsm.get("passed") is None:
        return {**out, "verdict": "incomplete", "reason": "no full GSM8K result for the current arm"}
    reasons = []
    if not all(g > 0 for g in per):
        reasons.append(f"the current arm was not ahead in every pair ({', '.join(f'{100 * g:+.2f}%' for g in per)})")
    if gain <= floor:
        reasons.append(f"the mean gain {100 * gain:+.2f}% is not above the floor {100 * floor:.2f}%")
    if not gsm.get("passed"):
        reasons.append(f"the current arm fails the full GSM8K ({gsm.get('score')})")
    return {**out, "verdict": "not_confirmed" if reasons else "confirmed",
            "reason": "; ".join(reasons) or (f"{100 * gain:+.2f}% over {len(full)} pairs, ahead in each, above the "
                                             f"{100 * floor:.2f}% floor; GSM8K {gsm.get('score')}")}


class Workspace:
    """The campaign's workspace as two commits: switched with ``git checkout -f``, and put back where it started."""

    def __init__(self, container: str, path: str):
        from arbor.execute.workspace import git
        self._git = lambda *args: git(container, path, *args)
        self.start = self.rev("HEAD")
        self.moved = False          # nothing is checked out, so nothing to put back, until the first switch
        try:                        # the branch it is on, so it goes back on it, not on a detached HEAD
            self.branch = self._git("symbolic-ref", "-q", "--short", "HEAD").strip() or None
        except RuntimeError:
            self.branch = None

    def rev(self, ref: str) -> str:
        try:
            sha = self._git("rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}").strip()
        except RuntimeError:
            sha = ""
        if not sha:
            raise Fail(f"{ref} is not a commit of the workspace")
        return sha

    def first(self) -> str:
        return self._git("rev-list", "--max-parents=0", "HEAD").split()[-1]

    def dirty(self) -> list[str]:
        return self._git("status", "--porcelain", "--untracked-files=no").splitlines()

    def switch(self, sha: str) -> None:
        self.moved = True
        self._git("checkout", "-q", "-f", sha)
        if self.rev("HEAD") != sha:
            raise Fail(f"the workspace did not switch to {sha[:10]}")

    def restore(self) -> None:
        """Back on the commit it started on, and on its branch when it was on one (the next commit, an integration's,
        moves that branch); a workspace never switched is left as it is, uncommitted work and all."""
        if not self.moved:
            return
        self._git("checkout", "-q", "-f", self.branch or self.start)
        if self.rev("HEAD") != self.start:
            raise RuntimeError(f"the workspace is not back on {self.start[:10]}")


def _render(s: dict) -> str:
    lines = [f"# A/B: {s.get('verdict')}", "", str(s.get("reason", "")), ""]
    trees = s.get("trees") or {}
    lines.append(f"- **Trees:** base `{str(trees.get('base'))[:10]}`, current `{str(trees.get('current'))[:10]}`")
    if isinstance(s.get("gain"), (int, float)):
        lines.append(f"- **{s.get('metric')}:** base {s.get('base_mean')}, current {s.get('current_mean')} "
                     f"({100 * s['gain']:+.2f}%), {s.get('rounds')} pairs, floor {100 * s.get('floor', 0):.2f}%")
    lines.append(f"- **GSM8K (current, full):** {s.get('gsm8k')}")
    lines.append(f"- **Node:** {s.get('node')}")
    return "\n".join(lines) + "\n"


def run(config_path: str, out: Path, deadline: float, base: str | None = None, current: str | None = None,
        rounds: int = 3, min_rounds: int = 2, pair_minutes: float = 20.0, gsm8k: bool = True) -> tuple[dict, int]:
    """The A/B; its summary and exit code (``EXIT_RESTORE`` when the tree could not be put back)."""
    from arbor.cli.iterate import _find_arbor_root
    from arbor.execute.bench import run_bench, run_correctness
    from arbor.pack import load_campaign
    from arbor.policy import claude_loop

    out.mkdir(parents=True, exist_ok=True)
    results = out / "results.jsonl"
    config_path = str(Path(config_path).resolve())
    campaign = load_campaign(config_path, str(_find_arbor_root(Path(config_path))))
    config = campaign.config
    obj = campaign.pack.objective
    meas = obj.get("measurement", {}) or {}
    metric, direction = obj.get("metric", "output_throughput"), obj.get("direction", "max")
    secondary = meas.get("secondary_metric") or None
    floors = {"threshold_rel": max(0.0, float(meas.get("threshold_rel", 1.0)) - 1.0),
              "noise_band_pct": float(meas.get("noise_band_pct", 0.0)) / 100.0}
    container, gpu_ids = config.get("docker_container", ""), config.get("gpu_ids")
    summary: dict = {"node": socket.gethostname().split(".")[0], "job": os.environ.get("SLURM_JOB_ID")}

    def stop() -> None:
        claude_loop._stop_server(config, container, gpu_ids)

    ws = None
    rc = 0
    try:
        ws = Workspace(container, config["workspace_root"])
        dirty = ws.dirty()
        if dirty:
            raise Fail(f"the workspace has uncommitted changes ({len(dirty)} paths), which a checkout would lose")
        trees = {"base": ws.rev(base) if base else ws.first(), "current": ws.rev(current) if current else ws.start}
        summary["trees"] = {**trees, "start": ws.start}
        log(f"base {trees['base'][:10]} against current {trees['current'][:10]} on {summary['node']}, judged on "
            f"{metric} ({direction}); deadline {time.strftime('%H:%M:%S', time.localtime(deadline))}")

        def run_arm(rnd: int, arm: str, full_gsm8k: bool) -> dict | None:
            ws.switch(trees[arm])
            tag = out / f"r{rnd}_{arm}"
            stop()
            t0 = time.time()
            if not claude_loop._start_server(config):
                raise Fail(f"{tag.name}: the server did not start")
            if not claude_loop._wait_healthy(config, 900):
                raise Fail(f"{tag.name}: the server did not become healthy")
            boot = time.time() - t0
            res = run_bench(config, tag)
            with results.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"round": rnd, "arm": arm, "tree": trees[arm][:10], "boot_s": round(boot, 1),
                                    **res}, default=str) + "\n")
            log(f"{tag.name}: {metric} {res.get(metric)}" + (f", {secondary} {res.get(secondary)}" if secondary
                                                               else "") + f", status {res.get('status')}")
            g = None
            if full_gsm8k:
                r = run_correctness(config, tag, extra_env={"GSM8K_LIMIT": str(FULL_GSM8K)})
                g = {"passed": r.get("passed"), "score": r.get("score")}
                if r.get("capped") or (r.get("error") and not r.get("score")):
                    # The check did not run to a score (the harness, not the tree): nothing is known.
                    g = {"passed": None, "score": None, "error": str(r.get("error") or "capped")[:300]}
                log(f"{tag.name}: full GSM8K {g}")
            stop()
            return g

        t0 = time.time()
        if t0 + 60 * pair_minutes * 1.3 > deadline:
            raise Fail("not enough time before the deadline for the warm-up round")
        run_arm(0, "base", False)
        gsm = run_arm(0, "current", gsm8k)
        if not gsm8k:
            gsm = {"passed": True, "score": None, "skipped": "--no-gsm8k"}
        log(f"warm-up round done in {(time.time() - t0) / 60:.0f} min")
        for rnd in range(1, rounds + 1):
            if time.time() + 60 * pair_minutes > deadline:
                log(f"round {rnd}: not enough time before the deadline for a pair -- stopping")
                break
            t1 = time.time()
            try:
                for arm in (("base", "current") if rnd % 2 else ("current", "base")):
                    run_arm(rnd, arm, False)
            except Exception as exc:  # noqa: BLE001 - the pairs already measured still count
                log(f"round {rnd} failed ({exc}) -- judging the pairs measured so far")
                break
            pair_minutes = max(pair_minutes, (time.time() - t1) / 60 * 1.1)
        rows = [json.loads(line) for line in results.read_text().splitlines()] if results.exists() else []
        summary.update(judge(rows, metric, direction, secondary, floors, gsm, min_rounds))
    except Exception as exc:  # noqa: BLE001 - whatever failed, the summary says so, and the tree is put back
        log(f"the A/B could not finish: {exc}")
        summary.update(verdict="incomplete", reason=str(exc)[:500])
    finally:
        try:
            stop()
        except Exception:  # noqa: BLE001 - the restore below still runs
            pass
        if ws is not None:
            try:
                ws.restore()
                log(f"the workspace is back on {ws.start[:10]}")
            except Exception as exc:  # noqa: BLE001 - whatever failed, the summary says so
                log(f"RESTORE FAILED: {exc}")
                summary.update(verdict="incomplete", restore_failed=True,
                               reason=f"restore failed: {exc}; " + str(summary.get("reason", "")))
                rc = EXIT_RESTORE
    summary.setdefault("verdict", "incomplete")
    tmp = out / "summary.json.tmp"
    tmp.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, out / "summary.json")
    (out / "summary.md").write_text(_render(summary), encoding="utf-8")
    log(f"verdict: {summary['verdict']} -- {summary.get('reason', '')}")
    return summary, rc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m arbor.ab",
                                 description="A same-node A/B of two trees of a campaign's workspace, for any pack: "
                                             "a round end's hook, or an integration's A/B.")
    ap.add_argument("--config", required=True, help="the campaign's config")
    ap.add_argument("--out", required=True, help="this A/B's directory (a round end's {out})")
    ap.add_argument("--deadline", type=float, required=True, help="epoch s by which the last pair must end")
    ap.add_argument("--base", default=None, help="the base tree (a commit); default the workspace's first commit")
    ap.add_argument("--current", default=None, help="the current tree (a commit); default the workspace's HEAD")
    ap.add_argument("--rounds", type=int, default=3, help="measured rounds after the warm-up round (default 3)")
    ap.add_argument("--min-rounds", type=int, default=2, help="measured pairs a verdict needs (default 2)")
    ap.add_argument("--pair-minutes", type=float, default=20.0, help="the time a measured pair needs (default 20)")
    ap.add_argument("--no-gsm8k", action="store_true", help="skip the current arm's full GSM8K (a pack with no "
                                                            "accuracy check)")
    a = ap.parse_args(argv)
    _summary, rc = run(a.config, Path(a.out), a.deadline, a.base, a.current, a.rounds, a.min_rounds,
                       a.pair_minutes, not a.no_gsm8k)
    return rc


if __name__ == "__main__":
    sys.exit(main())
