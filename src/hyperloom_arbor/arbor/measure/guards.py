"""A host's guards: the metrics a KEEP must not give up, held against the kept tree's reading (arbor/objective.py).

A host that gates on one metric often refuses a gain another metric paid for. An AgentX host keeps a P50 interactivity
gain only while P90 interactivity and output throughput hold within its noise band, and the error rate stays under a
cap. Arbor holds a KEEP to the same guards, so an iteration does not keep what the host would refuse:

- **a band guard** ``{metric, direction, band_pct}``: the candidate's reading is at least the kept tree's
  x (1 - band / 100), or at most x (1 + band / 100) for a metric to minimize;
- **a cap** ``{metric, max_pct}``: the reading is at most ``max_pct``, absolute. ``request_error_rate`` is computed
  from the request counts when the bench reports those alone.

A broken guard is a REVERT, whatever the intent. Arbor does not retest; the host's gate still decides what it merges.
The kept tree's readings are the session's ``guard_ref``, taken at every measured KEEP and PROBE (``reading``). A guard
the bench did not report is not checked, and the row says so. A host's ``component.bench_read`` fails the bench instead,
so under it no guard goes unchecked.
"""

from __future__ import annotations


def value(bench: dict, metric: str) -> float | None:
    """``metric``'s reading in ``bench``; None when it has none."""
    v = (bench or {}).get(metric)
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    if metric == "request_error_rate":
        ok, total = (bench or {}).get("num_requests_successful"), (bench or {}).get("num_requests_total")
        if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in (ok, total)) and total > 0:
            return (1 - ok / total) * 100
    return None


def check(measurement: dict, bench: dict, ref: dict | None, matched: dict | None = None) -> dict | None:
    """``{"broken", "held", "unchecked"}``, one line each, for the measurement's guards; None without guards.

    ``matched`` (arbor/measure/matched.py ``compare``): a band guard on a per-request metric with a matched reading is
    judged on the requests both runs completed, the candidate's reading over the kept tree's on them; its line gives the
    raw readings beside it. Without one, the raw readings decide, as before."""
    guards = (measurement or {}).get("guards") or []
    if not guards:
        return None
    out: dict[str, list[str]] = {"broken": [], "held": [], "unchecked": []}
    for g in guards:
        metric = g["metric"]
        v = value(bench, metric)
        if v is None:
            out["unchecked"].append(f"{metric}: the bench did not report it")
            continue
        if "max_pct" in g:
            cap = float(g["max_pct"])
            line = f"{metric} {v:.2f} % against the {cap:g} % cap"
            (out["broken"] if v > cap else out["held"]).append(line)
            continue
        m = (matched or {}).get(metric) or {}
        if m.get("used"):
            band, r = float(g.get("band_pct") or 0.0), float(m["ratio"])
            if g.get("direction", "max") == "min":
                ok, change = r <= 1 + band / 100, (1 - r) * 100
            else:
                ok, change = r >= 1 - band / 100, (r - 1) * 100
            kept = (ref or {}).get(metric)
            raw = f"; raw {v:g} against {kept:g}" if isinstance(kept, (int, float)) else ""
            line = (f"{metric} {m['candidate']:g} against the kept tree's {m['kept']:g} on {m['n']} matched requests "
                    f"({change:+.1f} %, band {band:g} %{raw})")
            (out["held"] if ok else out["broken"]).append(line)
            continue
        kept = (ref or {}).get(metric)
        if not isinstance(kept, (int, float)) or kept == 0:
            out["unchecked"].append(f"{metric}: no reading of the kept tree yet")
            continue
        band = float(g.get("band_pct") or 0.0)
        if g.get("direction", "max") == "min":
            ok, change = v <= kept * (1 + band / 100), (kept - v) / kept * 100
        else:
            ok, change = v >= kept * (1 - band / 100), (v - kept) / kept * 100
        line = f"{metric} {v:g} against the kept tree's {kept:g} ({change:+.1f} %, band {band:g} %)"
        (out["held"] if ok else out["broken"]).append(line)
    return out


def reading(measurement: dict, bench: dict) -> dict[str, float]:
    """The band guards' readings in ``bench``: the kept tree's, once its iteration is kept."""
    out = {}
    for g in (measurement or {}).get("guards") or []:
        if "band_pct" in g and (v := value(bench, g["metric"])) is not None:
            out[g["metric"]] = v
    return out
