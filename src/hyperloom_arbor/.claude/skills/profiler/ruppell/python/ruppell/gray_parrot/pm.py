"""gray-parrot's device samples on CUDA (GP4, RUPPELL_DEVICE_COUNTERS; runner plan 04, Part D).

The CUDA tool stores PM sampling's raw counter data (device_counters.raw), and `ruppell_pm_eval`,
built beside the tool, evaluates it after the run into device_counters.bin (one record per sample
and metric, step 0): evaluating costs about 216 us a sample, more than a host core at 10 kHz.

- `evaluate(run)`: runs the evaluator on every process with raw samples (the build is the one
  run.json names, or RUPPELL_BUILD).
- `activity(run)`: per process, each metric over the run and per step kind. A step's samples are
  those whose midpoint lies in its GPU span (its first kernel's start to its last kernel's end), so
  only steps with kernel records (the detail sample) have any. Percentages are averaged; byte
  counts become GB/s over the sample's interval.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections import defaultdict
from typing import List, Optional

import numpy as np

from ruppell.engine import Analysis, Proc, process_dirs
from ruppell.gray_parrot import Counters


def _build(run: str) -> str:
    b = os.environ.get("RUPPELL_BUILD", "")
    if b:
        return b
    for path in (os.path.join(run, "run.json"), os.path.join(os.path.dirname(run.rstrip("/")), "run.json")):
        try:
            info = json.load(open(path))
        except (OSError, ValueError):
            continue
        for v in info.values():
            if isinstance(v, str) and v.endswith("libruppell_tool.so"):
                return os.path.dirname(v)
    return ""


def evaluate(run: str, build: Optional[str] = None, threads: Optional[int] = None, force: bool = False) -> List[dict]:
    """Evaluate each process's raw samples; skipped where device_counters.bin is newer than them."""
    out = []
    exe = os.path.join(build or _build(run), "ruppell_pm_eval")
    for pd in process_dirs(run):
        raw = os.path.join(pd, "device_counters.raw")
        if not os.path.exists(raw):
            continue
        done = os.path.join(pd, "device_counters.bin")
        if not force and os.path.exists(done) and os.path.getmtime(done) >= os.path.getmtime(raw):
            out.append({"dir": pd, "skipped": "evaluated before"})
            continue
        if not os.path.exists(exe):
            out.append({"dir": pd, "error": f"{exe} not found (build with runtime/build.sh, or set RUPPELL_BUILD)"})
            continue
        r = subprocess.run([exe, pd] + ([str(threads)] if threads else []), capture_output=True, text=True)
        try:
            out.append(json.loads(r.stdout) if r.returncode == 0 else {"dir": pd, "error": r.stderr.strip()})
        except ValueError:
            out.append({"dir": pd, "error": (r.stdout + r.stderr).strip()[:500]})
    return out


def _is_bytes(name: str) -> bool:
    return "bytes" in name and "pct" not in name


def samples(pd: str):
    """(names, t0, t1, values): the process's samples, one row per sample, one column per counter."""
    c = Counters(pd)
    d = c.device
    d = d[d["step"] == 0] if len(d) else d
    if not len(d):
        return {}, np.zeros(0, np.uint64), np.zeros(0, np.uint64), np.zeros((0, 0))
    ids = sorted({int(x) for x in d["counter"].tolist()})
    t0, inv = np.unique(d["t0"], return_inverse=True)
    t1 = np.zeros(len(t0), np.uint64)
    t1[inv] = d["t1"]
    vals = np.full((len(t0), len(ids)), np.nan)
    col = {k: i for i, k in enumerate(ids)}
    vals[inv, [col[int(x)] for x in d["counter"].tolist()]] = d["value"]
    return {i: c.name(k) for i, k in enumerate(ids)}, t0, t1, vals


def _summary(names, t0, t1, vals, sel) -> dict:
    out = {}
    dt = (t1.astype(np.float64) - t0.astype(np.float64))[sel]
    for i, n in names.items():
        v = vals[sel, i]
        if not len(v):
            continue
        if _is_bytes(n):
            out[n.replace(".sum", "") + " GB/s"] = float(np.nansum(v) / max(np.sum(dt), 1.0))
        else:
            out[n] = float(np.nanmean(v))
    return out


def activity(run: str) -> dict:
    procs = []
    for pd in process_dirs(run):
        names, t0, t1, vals = samples(pd)
        if not len(t0):
            continue
        p = Proc(pd)
        a = Analysis(p)
        d = p.d
        mid = (t0.astype(np.float64) + t1.astype(np.float64)) / 2
        per_kind = defaultdict(lambda: np.zeros(len(t0), bool))
        steps_with = defaultdict(int)
        if len(d):
            work = ~a.is_marker & ~a.is_profiler & (a.step >= 0)
            st, s0, s1 = a.step[work], d["start"][work], d["end"][work]
            if len(st):
                order = np.argsort(st, kind="stable")
                st, s0, s1 = st[order], s0[order], s1[order]
                ids, first = np.unique(st, return_index=True)
                lo = np.minimum.reduceat(s0, first).astype(np.float64)
                hi = np.maximum.reduceat(s1, first).astype(np.float64)
                kinds = {s["step"]: str(s["fields"].get("kind") or s["fields"].get("phase") or "-") for s in a.step_list}
                for sid, l, h in zip(ids.tolist(), lo, hi):
                    k = kinds.get(sid, "-")
                    m = (mid >= l) & (mid <= h)
                    if m.any():
                        per_kind[k] |= m
                        steps_with[k] += 1
        meta = p.meta.get("counters")
        dev = meta.get("device") if isinstance(meta, dict) else None
        procs.append({
            "process": p.name, "label": p.label, "samples": int(len(t0)),
            "interval_ns": float(np.median(t1.astype(np.float64) - t0.astype(np.float64))),
            "span_s": float((int(t1.max()) - int(t0.min())) / 1e9), "device": dev,
            "run": _summary(names, t0, t1, vals, np.ones(len(t0), bool)),
            "per_kind": {k: {"steps": steps_with[k], "samples": int(m.sum()), **_summary(names, t0, t1, vals, m)}
                         for k, m in sorted(per_kind.items())}})
    return {"run": run, "processes": procs}


def markdown(rep: dict) -> str:
    out = []
    for p in rep["processes"]:
        out.append(f"## {p['label']}: {p['samples']} device samples every {p['interval_ns'] / 1e3:.0f} us "
                   f"over {p['span_s']:.1f} s\n")
        keys = list(p["run"])
        out.append("| over | steps | samples | " + " | ".join(keys) + " |")
        out.append("| --- | ---: | ---: | " + " | ".join("---:" for _ in keys) + " |")
        out.append(f"| the run | | {p['samples']} | " + " | ".join(f"{p['run'][k]:.4g}" for k in keys) + " |")
        for kind, s in p["per_kind"].items():
            out.append(f"| {kind} | {s['steps']} | {s['samples']} | " +
                       " | ".join(f"{s[k]:.4g}" if k in s else "" for k in keys) + " |")
        out.append("")
    return "\n".join(out)
