"""FIX-STABILITY: a fix for a failure the kept tree shows, held to a cap instead of the noise band, when the harness
sees the failure before the fix and not after it.

A FIX is held to the noise band. A fix for a hang or a stall (a pinned CPU KV buffer the kernel moves, so the GPU
waits; a collective that times out) can cost a percent and still be the change that makes every later reading
trustworthy. Held to the band, such a fix was reverted for costing about a percent, and the hazard stayed for the rest
of the run.

The agent declares ``INTENT: FIX-STABILITY`` and cites the failure, ``FAILURE: iter-K`` (one or more). The harness
holds the fix to three things:

1. **The failure before it.** A cited iteration after the last KEEP (the kept tree, its own change aside) whose
   end-of-run server log (``iter_K/server_end.log``, ``capture``) or recorded reason shows a stability signal, or whose
   reading collapsed: ``stability_collapse_pct`` (10 %) below the best.
2. **None after it.** The candidate's own end-of-run log shows no signal.
3. **The cap.** It may cost up to ``stability_regression_cap_pct`` (3 %) of the metric, as a correctness fix may cost
   its own cap, and the bars rebase to its reading.

Without the failure on record it is gated as a FIX, and the row says so. With the failure in its own run it is a REVERT.
The signals are the pack's ``objective.measurement.stability_signals`` (regular expressions), or ``SIGNALS``.
"""

from __future__ import annotations

import re
from pathlib import Path

LOG_FILE = "server_end.log"     # iter_N/: the server log's tail once the bench is done
LOG_LINES = 50_000
DEFAULT_COLLAPSE_PCT = 10.0

# What a server log shows when the run hung, stalled or lost its device: collective watchdogs, device faults, a dead
# engine. Never throughput lines: a slow run is not a failure here.
SIGNALS = (
    r"Watchdog caught collective operation timeout",
    r"(?:NCCL|RCCL|ProcessGroupNCCL|c10d).{0,80}(?:timed out|timeout)",
    r"[Hh]eartbeat monitor timed out",
    r"Memory access fault by GPU",
    r"HSA_STATUS_ERROR",
    r"GPU [Hh]ang",
    r"hipError(?:LaunchFailure|IllegalAddress)|HIP error: (?:an illegal|unspecified launch)",
    r"EngineDeadError|EngineCore (?:encountered a fatal error|died)",
)
# What a recorded reason says when the iteration failed that way (an INVALID's "bench did not run (timeout ...)").
REASONS = r"\b(?:hang|hung|stall(?:ed)?|watchdog|timed out|timeout|deadlock|collective)\b"

_CITE = re.compile(r"iter[-_ ]?(\d+)", re.I)


def signals(measurement: dict | None) -> list[re.Pattern]:
    pats = (measurement or {}).get("stability_signals") or SIGNALS
    return [re.compile(p) for p in pats]


def found(text: str, pats: list[re.Pattern], limit: int = 3) -> list[str]:
    """The first lines of ``text`` that show a signal."""
    out = []
    for line in (text or "").splitlines():
        if any(p.search(line) for p in pats):
            out.append(line.strip()[:240])
            if len(out) >= limit:
                break
    return out


def capture(log: str, iter_dir: str | Path) -> None:
    """Keep the server log's end-of-run tail beside the iteration's other records."""
    if log:
        Path(iter_dir).mkdir(parents=True, exist_ok=True)
        (Path(iter_dir) / LOG_FILE).write_text(log)


def cited(agent_output: str) -> list[int]:
    """The iterations a ``FAILURE:`` line names."""
    out: list[int] = []
    for line in (agent_output or "").splitlines():
        body = line.strip().strip("*-_` ")
        if body.upper().startswith("FAILURE"):
            out += [int(n) for n in _CITE.findall(body[len("FAILURE"):])]
    return list(dict.fromkeys(out))


def _last_keep(lessons: list[dict], before: int) -> int:
    return max((int(l.get("iteration") or 0) for l in lessons
                if l.get("decision") == "KEEP" and int(l.get("iteration") or 0) < before), default=0)


def check(session_dir: str | Path, lessons: list[dict], iteration: int, agent_output: str,
          measurement: dict | None, best: float | None, direction: str = "max") -> dict:
    """``{"before": {"iteration", "evidence"} | None, "own": [lines], "why": str}`` for a FIX-STABILITY candidate."""
    pats = signals(measurement)
    session_dir = Path(session_dir)
    own = found(_read(session_dir / f"iter_{iteration}" / LOG_FILE), pats)
    since = _last_keep(lessons, iteration)
    collapse = float((measurement or {}).get("stability_collapse_pct", DEFAULT_COLLAPSE_PCT))
    refused = []
    for k in cited(agent_output):
        if not since < k < iteration:
            refused.append(f"iter-{k} is not after the last KEEP (iter-{since})")
            continue
        rows = [l for l in lessons if int(l.get("iteration") or 0) == k]
        evidence = found(_read(session_dir / f"iter_{k}" / LOG_FILE), pats)
        evidence += [f"its row: {r.get('reason')}" for r in rows
                     if re.search(REASONS, str(r.get("reason") or ""), re.I)][:1]
        for r in rows:
            m = r.get("metric")
            if isinstance(m, (int, float)) and m > 0 and isinstance(best, (int, float)) and best > 0:
                drop = (best - m) / best * 100 if direction == "max" else (m - best) / best * 100
                if drop >= collapse:
                    evidence.append(f"its reading {m:g} is {drop:.1f}% below the best {best:g}")
        if evidence:
            return {"before": {"iteration": k, "evidence": evidence[:3]}, "own": own, "why": ""}
        refused.append(f"iter-{k} shows no stability signal")
    why = "; ".join(refused) or "no FAILURE: iter-K line"
    return {"before": None, "own": own, "why": why}


def _read(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""
