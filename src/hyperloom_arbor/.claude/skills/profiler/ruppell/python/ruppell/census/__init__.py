"""The census: per step and per process, where the GPU time went.

For every step of every process (engine steps and named startup steps):
- the step's GPU interval (period) and busy time (union of its kernels' and copies'
  intervals), kept apart;
- summed kernel time, total and by op, and the unattributed share;
- the step's schema fields;
- the client phase it falls in, when phases are given (realtime seconds),
  aligned through the process's clock pairs;
- side-stream kernels running outside their step (docs/plan-m2.md 9.2).

Output: `census.jsonl` (one line per process and step) and
`census_scopes.npz` + `census_scopes.json` (GPU time per step and scope, in
columns: tens of thousands of steps times hundreds of scopes do not belong
in JSON).
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from ruppell.engine import Analysis, analyze_run


def _offset(a: Analysis) -> Optional[float]:
    c = a.proc.clock
    if not c:
        return None
    diffs = sorted(((x[0] + x[1]) / 2.0) - x[2] for x in c)  # sdk ns - realtime ns
    return diffs[len(diffs) // 2]


def in_repeated_block(name: str) -> bool:
    """A scope inside a repeated block, a layer: a module path with an index segment
    (model.layers.3.mlp, drafter.layers.0; a profile folds the indices to *)."""
    return any(p.isdigit() or p == "*" for p in str(name).split("."))


def _busy(starts: np.ndarray, ends: np.ndarray, group: np.ndarray, ngroups: int) -> np.ndarray:
    """Union of [start, end) intervals per group. Inputs sorted by (group, start)."""
    if not len(starts):
        return np.zeros(ngroups, dtype=np.int64)
    span = int(ends.max() - starts.min()) + 1
    base = starts.min()
    off = group.astype(np.int64) * span
    s = (starts - base).astype(np.int64) + off
    e = (ends - base).astype(np.int64) + off
    runmax = np.maximum.accumulate(e)
    prev = np.empty_like(runmax)
    prev[0] = s[0]
    prev[1:] = runmax[:-1]
    first = np.ones(len(s), dtype=bool)
    first[1:] = group[1:] != group[:-1]
    prev[first] = s[first]
    covered = np.clip(e - np.maximum(s, prev), 0, None)
    return np.bincount(group, weights=covered, minlength=ngroups).astype(np.int64)


def compute(run_dir: str, phases: Optional[Dict[str, Tuple[float, float]]] = None, out_dir: Optional[str] = None) -> Dict:
    res = analyze_run(run_dir)
    out_dir = out_dir or run_dir
    lines: List[dict] = []
    col_proc, col_step, col_scope, col_ns = [], [], [], []
    scope_table: Dict[str, Dict[str, dict]] = {}
    summary = {"processes": {}}
    for a in res["analyses"]:
        p, d = a.proc, a.proc.d
        if not a.step_list or not len(d):
            continue
        label = p.label
        # counter steps (gray-parrot) are not timing: their kernels were serialized by counting
        steps = [s for s in a.step_list if s["step"] not in a.untimed_step_ids]
        if not steps:
            continue
        work = (~a.is_marker) & (a.step >= 0) & ~a.untimed
        wi = np.nonzero(work)[0]
        # step id -> index into steps, vectorized (tens of millions of kernels)
        ids = np.array([s["step"] for s in steps], dtype=np.int64)
        id_order = np.argsort(ids)
        sidx = id_order[np.searchsorted(ids[id_order], a.step[wi].astype(np.int64))] if len(wi) else np.zeros(0, np.int64)
        dur = (d["end"][wi] - d["start"][wi]).astype(np.int64)
        n = len(steps)
        kernel_ns = np.bincount(sidx, weights=dur, minlength=n).astype(np.int64)
        attributed = a.scope[wi] > 0
        attr_ns = np.bincount(sidx[attributed], weights=dur[attributed], minlength=n).astype(np.int64)
        outside = np.bincount(sidx[a.outside_owner[wi]], minlength=n)
        # memory copies and sets in a step (copies.bin): their own time, bytes and count;
        # busy is the union of kernels and copies
        c = p.copies
        ci = np.nonzero((a.copy_step >= 0) & ~a.copy_untimed)[0] if len(c) else np.zeros(0, np.int64)
        csidx = id_order[np.searchsorted(ids[id_order], a.copy_step[ci])] if len(ci) else np.zeros(0, np.int64)
        copy_ns = np.bincount(csidx, weights=(c["end"][ci] - c["start"][ci]).astype(np.float64), minlength=n).astype(np.int64)
        copy_bytes = np.bincount(csidx, weights=c["bytes"][ci].astype(np.float64), minlength=n).astype(np.int64)
        copy_n = np.bincount(csidx, minlength=n)
        bs = np.concatenate([d["start"][wi], c["start"][ci]])
        be = np.concatenate([d["end"][wi], c["end"][ci]])
        bg = np.concatenate([sidx, csidx]).astype(np.int64)
        o = np.lexsort((bs, bg))
        busy = _busy(bs[o], be[o], bg[o], n)

        # by scope and op
        scopes = a.scope[wi]
        ops: Dict[int, str] = {}
        table = {}
        for sc in np.unique(scopes).tolist():
            if sc <= 0:
                continue
            info = a.scope_info(sc) or {"name": str(sc)}
            ops[sc] = str(info.get("op", ""))
            table[str(sc)] = {"name": info["name"], "op": info.get("op"), "shape": info.get("shape")}
        scope_table[label] = table
        key = sidx * (int(scopes.max()) + 2 if len(scopes) else 1) + (scopes + 1)
        uk, inv = np.unique(key, return_inverse=True)
        ns = np.bincount(inv, weights=dur).astype(np.int64)
        width = int(scopes.max()) + 2 if len(scopes) else 1
        k_step = uk // width
        k_scope = uk % width - 1
        col_proc += [label] * len(uk)
        col_step += [steps[int(x)]["step"] for x in k_step]
        col_scope += k_scope.tolist()
        col_ns += ns.tolist()
        by_op_step: Dict[int, Dict[str, int]] = {}
        for st_i, sc, v in zip(k_step.tolist(), k_scope.tolist(), ns.tolist()):
            op = ops.get(sc, "unattributed") if sc > 0 else "unattributed"
            dct = by_op_step.setdefault(st_i, {})
            dct[op] = dct.get(op, 0) + v
        # how finely the time is attributed: on a scope inside a repeated block (a module path
        # with an index, model.layers.3.mlp), per step; and by module path (a path is several
        # scope ids, one per input shape), for the largest scope of each phase's decode steps
        names: Dict[str, int] = {}
        name_idx = np.full(width + 1, -1, dtype=np.int64)
        in_block = np.zeros(width + 1, dtype=bool)
        for sc in ops:
            nm = table[str(sc)]["name"]
            name_idx[sc + 1] = names.setdefault(nm, len(names))
            in_block[sc + 1] = in_repeated_block(nm)
        layer_ns = np.bincount(k_step, weights=np.where(in_block[k_scope + 1], ns, 0), minlength=n).astype(np.int64)

        off = _offset(a)
        for n_i, s in enumerate(steps):
            phase = None
            if phases and off is not None:
                t_rt = (s["start"] - off) / 1e9
                for name, (lo, hi) in phases.items():
                    if lo <= t_rt < hi:
                        phase = name
            lines.append({
                "process": label, "step": s["step"], "start_ns": s["start"], "period_ns": s["end"] - s["start"],
                "busy_ns": int(busy[n_i]), "kernel_ns": int(kernel_ns[n_i]), "attributed_ns": int(attr_ns[n_i]),
                "copy_ns": int(copy_ns[n_i]), "copy_bytes": int(copy_bytes[n_i]), "copies": int(copy_n[n_i]),
                "by_op_ns": by_op_step.get(n_i, {}), "layer_ns": int(layer_ns[n_i]),
                "side_stream_kernels_outside_step": int(outside[n_i]),
                "client_phase": phase, "fields": s["fields"],
            })
        eng = [ln for ln in lines if ln["process"] == label and ln["fields"].get("phase") == "engine"]
        # Time-weighted like the workload census (AGENTS.md §8), within each
        # client phase only: a step's running count holds until the next step
        # begins or the phase ends. Over a whole run it would not mean much:
        # a count would hold across the idle time between phases.
        by_phase: Dict[str, List[int]] = {}
        phase_end = {k: hi * 1e9 + off for k, (lo, hi) in phases.items()} if phases and off is not None else {}
        # The running count is weighted over every engine step, from the fields
        # and host begin time each step records whether its kernels were
        # (detail sampling) or not; census lines exist for traced steps only,
        # and a sampled step's count held across the steps between samples is
        # not the load (job 50402: 1.35 against aiperf's 1.15).
        every = sorted((p.step_cpu[s][0], f.get("running_requests", 0)) for s, f in p.step_fields.items()
                       if f.get("phase") == "engine" and s in p.step_cpu and p.step_cpu[s][0] is not None)
        if every and phase_end:
            for (t, r), (t_next, _) in zip(every, every[1:]):
                ph = next((k for k, (lo, hi) in phases.items() if lo <= (t - off) / 1e9 < hi), None)
                if ph is None:
                    continue
                w = min(t_next, phase_end[ph]) - t
                acc = by_phase.setdefault(ph, [0, 0, 0, 0, 0, 0, 0, 0])
                acc[0] += r * w
                acc[1] += w
        decode_steps: Dict[str, List[int]] = {}
        for x, nxt in zip(eng, eng[1:]):
            if not x["client_phase"]:
                continue
            acc = by_phase.setdefault(x["client_phase"], [0, 0, 0, 0, 0, 0, 0, 0])
            if not every:
                w = min(nxt["start_ns"], phase_end[x["client_phase"]]) - x["start_ns"]
                r = x["fields"].get("running_requests", 0)
                acc[0] += r * w
                acc[1] += w
            acc[2] += 1
            # plan-v0 coverage: decode steps' GPU time attributed to any scope,
            # and to a model or drafter module (a (layer, op) key; the runner's
            # own work is attributed but not to a layer)
            if x["fields"].get("kind") == "decode":
                acc[3] += 1
                acc[4] += x["kernel_ns"]
                acc[5] += x["attributed_ns"]
                acc[6] += x["attributed_ns"] - x["by_op_ns"].get("runner", 0)
                # plan 06's granularity: on a module inside a repeated block (a layer). A compiled
                # region whose modules run no hooks puts its time on the scope around it
                acc[7] += x["layer_ns"]
                decode_steps.setdefault(x["client_phase"], []).append(x["step"])
        top: Dict[str, Tuple[str, int]] = {}
        k_name = name_idx[k_scope + 1]
        k_id = ids[k_step] if len(k_step) else np.zeros(0, np.int64)
        by_name = list(names)
        for ph, sl in decode_steps.items():
            m = np.isin(k_id, sl) & (k_name >= 0)
            t = np.bincount(k_name[m], weights=ns[m], minlength=len(names)) if m.any() else np.zeros(0)
            if len(t) and t.max() > 0:
                j = int(t.argmax())
                top[ph] = (by_name[j], int(t[j]))
        summary["processes"][label] = {
            "steps": n,
            "counter_steps_left_out": len(a.untimed_step_ids),
            "engine_steps": len(eng),
            "records_complete": p.records_complete,
            "kernel_ns": int(kernel_ns.sum()),
            "attributed_ns": int(attr_ns.sum()),
            "engine_busy_over_period": (sum(x["busy_ns"] for x in eng) / sum(x["period_ns"] for x in eng)) if eng else None,
            "mean_running_requests_per_step": (sum(x["fields"].get("running_requests", 0) for x in eng) / len(eng)) if eng else None,
            "by_client_phase": {k: {"engine_steps": v[2], "seconds": v[1] / 1e9,
                                    "mean_running_requests_time_weighted": v[0] / v[1] if v[1] else None,
                                    "decode_steps": v[3], "decode_kernel_ns": v[4],
                                    "decode_attributed_share": v[5] / v[4] if v[4] else None,
                                    "decode_module_share": v[6] / v[4] if v[4] else None,
                                    "decode_layer_share": v[7] / v[4] if v[4] else None,
                                    "decode_top_scope": top[k][0] if k in top else None,
                                    "decode_top_scope_share": top[k][1] / v[4] if k in top and v[4] else None}
                                for k, v in by_phase.items()},
        }

    with open(os.path.join(out_dir, "census.jsonl"), "w") as f:
        for ln in lines:
            f.write(json.dumps(ln, separators=(",", ":")) + "\n")
    np.savez_compressed(os.path.join(out_dir, "census_scopes.npz"), process=np.array(col_proc), step=np.array(col_step, dtype=np.int64),
                        scope=np.array(col_scope, dtype=np.int64), gpu_ns=np.array(col_ns, dtype=np.int64))
    with open(os.path.join(out_dir, "census_scopes.json"), "w") as f:
        json.dump(scope_table, f)
    summary["lines"] = len(lines)
    summary["merge"] = res["merge"]
    # processes with engine steps whose records stop at a periodic flush: their
    # last steps are missing, and ranks then disagree on their engine steps
    summary["incomplete_processes"] = [k for k, v in summary["processes"].items()
                                       if v["engine_steps"] and not v["records_complete"]]
    from ruppell.limits import for_run

    summary["limitations"] = for_run(res["analyses"][0].proc.meta if res["analyses"] else None,
                                     cpu_ops=any(len(a.proc.ops) for a in res["analyses"]),
                                     incomplete=bool(summary["incomplete_processes"]),
                                     copies=all(a.proc.copies_recorded for a in res["analyses"]))
    with open(os.path.join(out_dir, "census_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary
