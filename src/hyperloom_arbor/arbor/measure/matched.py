"""Guards on matched requests: a per-request metric judged on the requests both runs completed, request for request.

A change that reorders which requests complete inside the bench's window moves a raw p50 or p90 over that window
without making any request faster or slower: with prefill scheduled longest-prefix-first, the measured window holds
another mix of short and long requests. A raw p50 interactivity check can reject a real throughput
win that way: it reads the mix, not the speed, while on the requests both runs completed interactivity holds.

So a guard metric that is a percentile over requests is read again on the matched set:

- **The requests:** aiperf's per-request records (``profile_export.jsonl``, under ``aiperf_artifacts/`` in the
  iteration's directory: ``agentic_run/`` for the pack's bench, ``host_bench/`` for a host's), the measured window only
  (``benchmark_phase`` ``profiling``), cancelled and failed ones left out. A request is
  ``(conversation_id, turn_index)``: the trace's own identity, the same in every replay of it.
- **The metrics** (``METRICS``), recomputed the way the bench defines them: ``p50_interactivity`` /
  ``p90_interactivity`` are 1000 / the p50 (p90) TPOT in ms; ``e2e_norm_intvty_p50`` / ``_p90`` are 1 / the p50 (p90)
  of E2EL in s / OSL. A metric not listed is never matched.
- **The reference:** the kept tree's records (``state.request_ref``): the root baseline's, when the session measured
  it, then the run of the last KEEP or measured PROBE. A frozen ``baseline_record`` carries none, so until the first
  of those the raw readings decide.
- **When it is used:** at least MIN_MATCHED requests in common, and at least MIN_SHARE of the kept run's requests.
  Fewer, and the matched set is a biased few (the requests both runs reached early in the window), so the raw aggregate
  decides, as before, and the row says why.
- **What it decides:** the candidate's reading over the kept tree's, both on the matched set (``ratio``). A guard's band
  and the gate's second axis are judged on that ratio; the raw readings are kept on the row beside it.

Standard library only; each file is read once, a line at a time (a measured iteration's records run to hundreds of MB).
"""

from __future__ import annotations

import json
from pathlib import Path

#: A guard metric that is a percentile over requests: (the per-request cost, its percentile, the scale).
#: The reading is scale / that percentile of the cost.
METRICS = {
    "p50_interactivity": ("tpot_ms", 50, 1000.0),
    "p90_interactivity": ("tpot_ms", 90, 1000.0),
    "e2e_norm_intvty_p50": ("e2e_s_per_token", 50, 1.0),
    "e2e_norm_intvty_p90": ("e2e_s_per_token", 90, 1.0),
}
#: The fewest requests in common for a matched reading: a p50 over 30 requests is within about a quarter of a standard
#: deviation of its own population; a p90 over fewer than 30 rests on two or three requests.
MIN_MATCHED = 30
#: The least share of the kept run's requests the candidate must have completed too: under half, the matched set is the
#: requests both runs reached early in the window, not the window.
MIN_SHARE = 0.5
RECORDS = "profile_export.jsonl"
MEASURED = "profiling"


def find(iter_dir) -> list[Path]:
    """The per-request record files of a bench run: the pack's bench (``agentic_run/``) and a host's (``host_bench/``,
    one directory per replay), in the directory or, for the root baseline's replays, in its ``runK/``. Nothing else in
    the directory is read: another tool's aiperf run there is not the gate's bench."""
    d = Path(iter_dir)
    bases = [d] + (sorted(p for p in d.glob("run*") if p.is_dir()) if d.is_dir() else [])
    out: list[Path] = []
    for base in bases:
        out += [p for p in [base / "agentic_run" / "aiperf_artifacts" / RECORDS] if p.is_file()]
        out += sorted(p for p in (base / "host_bench").glob(f"**/aiperf_artifacts/{RECORDS}") if p.is_file())
    return out


def _value(metrics: dict, name: str):
    v = (metrics.get(name) or {}).get("value") if isinstance(metrics.get(name), dict) else metrics.get(name)
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def read(iter_dir) -> dict[tuple, dict]:
    """``{(conversation_id, turn_index): {"tpot_ms", "e2e_s_per_token"}}`` for the measured window's completed requests.
    A request found in several files (a bench's replays) keeps the mean of each cost."""
    seen: dict[tuple, dict[str, list[float]]] = {}
    for path in find(iter_dir):
        try:
            fh = open(path, encoding="utf-8", errors="replace")
        except OSError:
            continue
        with fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                meta, metrics = rec.get("metadata") or {}, rec.get("metrics") or {}
                if meta.get("benchmark_phase") != MEASURED or rec.get("error") or meta.get("was_cancelled") \
                        or rec.get("was_cancelled"):
                    continue
                key = (meta.get("conversation_id"), meta.get("turn_index"))
                if key[0] is None:
                    continue
                costs = seen.setdefault(key, {"tpot_ms": [], "e2e_s_per_token": []})
                itl, osl, e2e = (_value(metrics, n) for n in ("inter_token_latency", "output_sequence_length",
                                                               "request_latency"))
                if itl and itl > 0:
                    costs["tpot_ms"].append(float(itl))
                if e2e and osl and osl > 0:
                    costs["e2e_s_per_token"].append(float(e2e) / 1000.0 / float(osl))
    return {k: {c: sum(v) / len(v) for c, v in costs.items() if v} for k, costs in seen.items()}


def percentile(values: list[float], q: float) -> float:
    """The q-th percentile, linearly interpolated (numpy's default)."""
    xs = sorted(values)
    if not xs:
        raise ValueError("no values")
    pos = (len(xs) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def reading(records: dict[tuple, dict], metric: str, keys=None) -> float | None:
    """``metric`` over ``records`` (only ``keys`` when given), as the bench defines it; None without a cost."""
    cost, q, scale = METRICS[metric]
    vals = [r[cost] for k, r in records.items() if cost in r and (keys is None or k in keys)]
    if not vals:
        return None
    p = percentile(vals, q)
    return scale / p if p > 0 else None


def compare(metrics, candidate: dict[tuple, dict], kept: dict[tuple, dict]) -> dict[str, dict]:
    """For each per-request metric in ``metrics``: ``{"used", "n", "share", "ratio", "candidate", "kept", "why"}``.
    ``used`` with ``ratio`` (candidate / kept, both on the requests in common) when the match is wide enough."""
    out = {}
    common = set(candidate) & set(kept)
    for metric in metrics:
        if metric not in METRICS:
            continue
        cost = METRICS[metric][0]
        keys = {k for k in common if cost in candidate[k] and cost in kept[k]}
        n_kept = sum(1 for r in kept.values() if cost in r)
        share = len(keys) / n_kept if n_kept else 0.0
        row = {"used": False, "n": len(keys), "share": round(share, 3)}
        if not candidate or not kept:
            row["why"] = "no per-request records" + (" for the candidate" if not candidate else " for the kept tree")
        elif len(keys) < MIN_MATCHED or share < MIN_SHARE:
            row["why"] = (f"{len(keys)} requests in common, {share:.0%} of the kept run's: under {MIN_MATCHED} or "
                          f"{MIN_SHARE:.0%}, the raw reading decides")
        else:
            cand, ref = reading(candidate, metric, keys), reading(kept, metric, keys)
            if cand and ref:
                row.update(used=True, candidate=round(cand, 4), kept=round(ref, 4), ratio=round(cand / ref, 5))
            else:
                row["why"] = "no reading on the requests in common"
        out[metric] = row
    return out
