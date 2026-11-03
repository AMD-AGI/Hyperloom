"""`ruppell witness RUN ROCPROF_DIR --counters [--also DIR ...] [--same-process]`: gray-parrot's
counters against rocprofv3's `counter_collection.csv` (GP1's exit, runner plan 03 B3 and B5).

- **Separate runs** (the default): the same program run under `rocprofv3 --pmc` with the same
  counters, in witness mode, so its kernel sequence is ruppell's, markers included. Steps pair by
  order: the k-th marker pair in rocprofv3's dispatch sequence is ruppell's k-th step. Within a
  step, dispatches pair by position in enqueue order (dispatch id), and must be the same kernel.
  Each rocprofv3 run (ROCPROF_DIR and every --also) gives one value per kernel, position and
  counter; their range is the spread a counted value is checked against.
- **One process** (--same-process): rocprofv3 counted in ruppell's process; dispatches pair by id.

It passes when deterministic counters (waves, instructions, MFMA operations, bytes written) equal
every rocprofv3 run's value, and every other counter's median, per kernel, lies within rocprofv3's
spread over all paired steps and runs (the operator, 2026-10-22, runner plan 03 decision 4). Each
value inside the range of its position's rocprofv3 runs is reported too, but a fourth draw of one
distribution falls outside three others' range half the time, so it does not decide.
"""

from __future__ import annotations

import csv
import glob
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from ruppell.engine import MARKER_BIT, MARKER_KERNEL, Analysis, Proc, process_dirs, short_name
from ruppell.gray_parrot import Counters, base_name

DETERMINISTIC = ("SQ_WAVES", "SQ_INSTS_", "TCC_EA0_WRREQ", "WRITE_SIZE")


def deterministic(counter: str) -> bool:
    return counter.startswith(DETERMINISTIC)


def _rocprof(rdir: str) -> List[Tuple[int, str, Dict[str, float]]]:
    """One rocprofv3 run's dispatches in dispatch-id order: (id, kernel, {counter: value})."""
    by: Dict[int, list] = {}
    for f in sorted(glob.glob(os.path.join(rdir, "**", "*counter_collection.csv"), recursive=True)):
        with open(f, newline="") as fh:
            for r in csv.DictReader(fh):
                try:
                    dd = int(r["Dispatch_Id"])
                    v = float(r["Counter_Value"])
                except (KeyError, ValueError):
                    continue
                e = by.setdefault(dd, [r.get("Kernel_Name", ""), {}])
                e[1][r.get("Counter_Name", "")] = e[1].get(r.get("Counter_Name", ""), 0.0) + v
    return [(dd, short_name(e[0]), e[1]) for dd, e in sorted(by.items())]


def _rocprof_steps(seq) -> List[list]:
    """rocprofv3's dispatches cut into steps: from each begin marker to the next marker (its end)."""
    marks = [i for i, x in enumerate(seq) if MARKER_KERNEL in x[1]]
    return [seq[marks[k]:marks[k + 1] + 1] for k in range(0, len(marks) - 1, 2)]


def _ruppell_steps(a: Analysis) -> List[Tuple[int, List[int]]]:
    """ruppell's steps in enqueue order: (step id, its dispatch indices from begin marker to end
    marker, by dispatch id)."""
    d = a.proc.d
    order = np.argsort(d["dispatch_id"], kind="stable")
    out, cur, sid = [], None, None
    for i in order.tolist():
        if a.is_marker[i]:
            code = int(d["ext"][i])
            if code & int(MARKER_BIT):
                kind, step = code & 3, (code & ~int(MARKER_BIT)) >> 2
                if kind == 1:
                    cur, sid = [i], step
                    continue
                if kind == 2 and cur is not None:
                    cur.append(i)
                    out.append((sid, cur))
                    cur = None
                    continue
        if cur is not None:
            cur.append(i)
    return out


def compare(run: str, rocprof_dirs: List[str], same_process: bool = False) -> dict:
    runs = [_rocprof(r) for r in rocprof_dirs]
    res = {"mode": "same process" if same_process else "separate runs", "rocprof_runs": rocprof_dirs,
           "rocprof_dispatches": [len(r) for r in runs], "processes": []}
    for pdir in process_dirs(run):
        c = Counters(pdir)
        if not c.recorded:
            continue
        p = Proc(pdir)
        a = Analysis(p)
        d = p.d
        names = {r["counter"]: base_name(r["name"]) for r in c.dispatch_rows()}   # an XCC split, summed
        vals: Dict[int, Dict[str, float]] = defaultdict(dict)
        for dd, ci, x in zip(c.values["key0"].tolist(), c.values["counter"].tolist(), c.values["value"].tolist()):
            if ci in names:
                vals[dd][names[ci]] = vals[dd].get(names[ci], 0.0) + x
        kname = lambda i: short_name(a.kernel_name.get(int(d["kernel_id"][i]), "?"))  # noqa: E731
        pairs: List[Tuple[str, str, float, List[float]]] = []   # kernel, counter, ours, theirs per run
        kernel_mismatch = steps_paired = 0
        note = None
        if same_process:
            idx = {dd: i for i, dd in enumerate(d["dispatch_id"].tolist())}
            theirs = [{dd: (k, v) for dd, k, v in r} for r in runs]
            for dd, mine in vals.items():
                i = idx.get(dd)
                if i is None:
                    continue
                for cn, x in mine.items():
                    ts = [t[dd][1].get(cn) for t in theirs if dd in t]
                    if ts and all(y is not None for y in ts):
                        pairs.append((kname(i), cn, x, ts))
        else:
            ours = _ruppell_steps(a)
            rsteps = [_rocprof_steps(r) for r in runs]
            if any(len(rs) != len(ours) for rs in rsteps):
                note = f"step counts differ: ruppell {len(ours)}, rocprofv3 {[len(rs) for rs in rsteps]}"
            cs = a.counter_step_ids
            for k, (sid, idxs) in enumerate(ours):
                if sid not in cs or any(k >= len(rs) for rs in rsteps):
                    continue
                # within a step, a kernel's n-th dispatch pairs with its n-th there: kernels on
                # other streams can enter the queue in another order from run to run (graph_probe's
                # side stream did, runner job 90857), so positions alone do not pair
                theirs = []
                for rs in rsteps:
                    occ: Dict[Tuple[str, int], dict] = {}
                    seen: Dict[str, int] = defaultdict(int)
                    for _, kn_r, v in rs[k]:
                        occ[(kn_r, seen[kn_r])] = v
                        seen[kn_r] += 1
                    theirs.append(occ)
                seen_m: Dict[str, int] = defaultdict(int)
                paired_here = False
                for i in idxs:
                    kn = kname(i)
                    n = seen_m[kn]
                    seen_m[kn] += 1
                    mine = vals.get(int(d["dispatch_id"][i]))
                    if not mine:
                        continue
                    tv = [t.get((kn, n)) for t in theirs]
                    if any(x is None for x in tv):
                        kernel_mismatch += 1
                        continue
                    paired_here = True
                    for cn, x in mine.items():
                        ts = [t.get(cn) for t in tv]
                        if all(y is not None for y in ts):
                            pairs.append((kn, cn, x, ts))
                steps_paired += int(paired_here)
        by: Dict[Tuple[str, str], dict] = {}
        for kn, cn, x, ts in pairs:
            e = by.setdefault((kn, cn), {"kernel": kn, "counter": cn, "deterministic": deterministic(cn), "compared": 0,
                                          "equal": 0, "within": 0, "max_rel_diff": 0.0, "_ours": [], "_theirs": []})
            lo, hi, med = min(ts), max(ts), float(np.median(ts))
            e["compared"] += 1
            e["equal"] += int(all(x == y for y in ts))
            e["within"] += int(lo <= x <= hi)
            e["_ours"].append(x)
            e["_theirs"] += ts
            if med:
                e["max_rel_diff"] = max(e["max_rel_diff"], abs(x - med) / abs(med))
            elif x:
                e["max_rel_diff"] = max(e["max_rel_diff"], float("inf"))
        for e in by.values():
            # over every paired step: the medians, and whether ours lie within rocprofv3's whole range.
            # A value inside the range of three runs at its own position is a weak test (a fourth
            # draw of the same distribution falls outside the other three's range half the time)
            o, t = np.asarray(e.pop("_ours")), np.asarray(e.pop("_theirs"))
            e["median"], e["rocprof_median"] = float(np.median(o)), float(np.median(t))
            e["median_rel_diff"] = (abs(e["median"] - e["rocprof_median"]) / abs(e["rocprof_median"])
                                    if e["rocprof_median"] else (0.0 if not e["median"] else float("inf")))
            e["within_rocprof_range_all_steps"] = float(((o >= t.min()) & (o <= t.max())).mean())
            # the pass rule for a counter that depends on the caches (the operator, 2026-10-22, plan
            # 03 decision 4): its median lies within rocprofv3's spread over every paired step and run
            e["rocprof_min"], e["rocprof_max"] = float(t.min()), float(t.max())
            e["median_within_spread"] = bool(t.min() <= e["median"] <= t.max())
        rows = sorted(by.values(), key=lambda e: (not e["deterministic"], e["kernel"], e["counter"]))
        det = [e for e in rows if e["deterministic"]]
        oth = [e for e in rows if not e["deterministic"]]
        res["processes"].append({
            "process": p.name, "counter_steps": len(a.counter_step_ids), "steps_paired": steps_paired,
            "dispatches_unpaired": kernel_mismatch, "note": note, "values_compared": len(pairs),
            "deterministic": {"compared": sum(e["compared"] for e in det), "equal": sum(e["equal"] for e in det)},
            "other": {"compared": sum(e["compared"] for e in oth), "within_spread": sum(e["within"] for e in oth),
                      "within_rocprof_range_all_steps": (sum(e["within_rocprof_range_all_steps"] * e["compared"] for e in oth)
                                                         / max(1, sum(e["compared"] for e in oth))),
                      "median_rel_diff_max": max((e["median_rel_diff"] for e in oth), default=None),
                      "kernel_counter_pairs": len(oth),
                      "medians_within_spread": sum(1 for e in oth if e["median_within_spread"]),
                      "medians_outside_spread": [{k: e[k] for k in ("kernel", "counter", "median", "rocprof_min",
                                                                   "rocprof_max", "compared")}
                                                 for e in oth if not e["median_within_spread"]][:20]},
            # vacuously true for a group with no deterministic counter: "deterministic.compared" says 0
            "all_deterministic_equal": all(e["equal"] == e["compared"] for e in det),
            "all_others_medians_within_spread": all(e["median_within_spread"] for e in oth),
            # the earlier, weaker rule, kept for comparison: each value inside the range of its
            # position's rocprofv3 runs
            "all_others_within_spread": all(e["within"] == e["compared"] for e in oth),
            "by_kernel_and_counter": rows,
        })
    ps = res["processes"]
    res["all_deterministic_equal"] = bool(ps) and all(p["all_deterministic_equal"] for p in ps)
    res["all_others_medians_within_spread"] = bool(ps) and all(p["all_others_medians_within_spread"] for p in ps)
    res["all_others_within_spread"] = bool(ps) and all(p["all_others_within_spread"] for p in ps)
    res["passes"] = res["all_deterministic_equal"] and res["all_others_medians_within_spread"]
    return res
