"""`ruppell report counters`: hardware counters per kernel (gray-parrot GP1).

Each counted dispatch joins to its dispatch record by dispatch id, and through it to its kernel,
grid, step and scope. Per kernel name and grid: each counter's mean, minimum, maximum and number
of samples (counted dispatches), summed over XCCs. Derived metrics and birdeye's table are GP3's.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import numpy as np

from ruppell.engine import Analysis, Proc, process_dirs, short_name
from ruppell.gray_parrot import Counters, base_name


def per_kernel(run: str) -> dict:
    procs = []
    for pdir in process_dirs(run):
        c = Counters(pdir)
        if not c.recorded:
            continue
        p = Proc(pdir)
        a = Analysis(p)
        d = p.d
        pos = {int(x): i for i, x in enumerate(d["dispatch_id"].tolist())}
        names = {r["counter"]: base_name(r["name"]) for r in c.dispatch_rows()}   # an XCC split, summed
        per: Dict[tuple, float] = defaultdict(float)
        missing = 0
        for dd, ci, x in zip(c.values["key0"].tolist(), c.values["counter"].tolist(), c.values["value"].tolist()):
            if ci in names:
                per[(dd, names[ci])] += x
        acc: Dict[tuple, list] = defaultdict(list)
        for (dd, cn), x in per.items():
            i = pos.get(dd)
            if i is None:
                missing += 1
                continue
            kn = short_name(a.kernel_name.get(int(d["kernel_id"][i]), "?"))
            grid = f"{int(d['gx'][i])}x{int(d['gy'][i])}x{int(d['gz'][i])}"
            acc[(kn, grid, cn)].append(x)
        table: List[dict] = []
        for (kn, grid, cn), xs in sorted(acc.items()):
            x = np.asarray(xs, np.float64)
            table.append({"kernel": kn, "grid": grid, "counter": cn, "n": int(len(x)), "mean": float(x.mean()),
                          "min": float(x.min()), "max": float(x.max())})
        meta = p.meta.get("counters")
        procs.append({"process": p.name, "label": p.label, "counter_steps": len(a.counter_step_ids),
                      "counted_dispatches": int(len(c.counted())), "values_without_dispatch_record": missing,
                      "counters": sorted(set(names.values())), "meta": meta, "table": table})
    return {"run": run, "processes": procs}


def per_step(run: str) -> dict:
    """Counter steps' readings (device_counters.bin records with ``step`` set; on CUDA every counter
    step is one reading, plan 04 Part B), per step kind and counter: n, mean, min, max. A step's
    kind is its fields' ``kind`` (the vLLM adapter's decode, prefill, mixed), else its ``phase``.
    Also the counter steps without a reading (the step fields' ``counters`` against the readings)."""
    procs = []
    for pdir in process_dirs(run):
        c = Counters(pdir)
        rows = c.device[c.device["step"] > 0] if len(c.device) else c.device
        if not len(rows) and not c.meta:
            continue
        p = Proc(pdir)
        kind = {int(s): str(f.get("kind") or f.get("phase") or "-") for s, f in p.step_fields.items()}
        counted = sorted(int(s) for s, f in p.step_fields.items() if "counters" in f)
        acc: Dict[tuple, list] = defaultdict(list)
        seen = set()
        for st, ci, x in zip(rows["step"].tolist(), rows["counter"].tolist(), rows["value"].tolist()):
            acc[(kind.get(int(st), "-"), c.name(ci))].append(x)
            seen.add(int(st))
        table = []
        for (k, cn), xs in sorted(acc.items()):
            x = np.asarray(xs, np.float64)
            table.append({"kind": k, "counter": cn, "n": int(len(x)), "mean": float(x.mean()),
                          "min": float(x.min()), "max": float(x.max())})
        procs.append({"process": p.name, "label": p.label, "counter_steps": len(counted),
                      "steps_with_readings": len(seen), "counter_steps_without_reading": len(set(counted) - seen),
                      "meta": c.meta or None, "table": table})
    return {"run": run, "processes": procs}


def markdown_steps(rep: dict, top: int = 0) -> str:
    out = []
    for p in rep["processes"]:
        out.append(f"## {p['label']}: {p['steps_with_readings']} of {p['counter_steps']} counter steps with readings\n")
        out.append("| step kind | counter | n | mean | min | max |")
        out.append("| --- | --- | ---: | ---: | ---: | ---: |")
        for r in (p["table"] if not top else p["table"][:top]):
            out.append(f"| {r['kind']} | {r['counter']} | {r['n']} | {r['mean']:.6g} | {r['min']:.6g} | {r['max']:.6g} |")
        out.append("")
    return "\n".join(out)


def markdown(rep: dict, top: int = 0) -> str:
    out = []
    for p in rep["processes"]:
        out.append(f"## {p['label']}: {p['counted_dispatches']} counted dispatches in {p['counter_steps']} counter steps\n")
        out.append("| kernel | grid | counter | n | mean | min | max |")
        out.append("| --- | --- | --- | ---: | ---: | ---: | ---: |")
        rows = p["table"] if not top else p["table"][:top]
        for r in rows:
            out.append(f"| `{r['kernel'][:80]}` | {r['grid']} | {r['counter']} | {r['n']} | {r['mean']:.6g} | "
                       f"{r['min']:.6g} | {r['max']:.6g} |")
        if p["values_without_dispatch_record"]:
            out.append(f"\n{p['values_without_dispatch_record']} values have no dispatch record.")
        out.append("")
    return "\n".join(out)
