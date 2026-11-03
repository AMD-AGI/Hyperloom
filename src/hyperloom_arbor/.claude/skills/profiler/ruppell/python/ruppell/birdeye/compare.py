"""Several birdeye reports side by side, the first as the baseline.

    python -m ruppell birdeye-compare OUT_DIR SRC_OR_DIR... [--tags a,b,...]

Each argument is a birdeye output directory (it has `birdeye.json`) or a
source, which is reported first into `OUT_DIR/<tag>/`. Tables are merged on
their keys, an outer join, like TraceLens's `compare_perf_reports_pytorch`:

| table | keys | values |
| --- | --- | --- |
| gpu_timeline | rank, type | time ms, percent |
| category_summary | category | Count, Kernel duration (µs)_sum, busy (µs), Percent of kernels time (%) |
| ops_summary_by_category | op category | Count, total_direct_kernel_time_ms, Percentage (%) |
| ops_summary | name | Count, total_direct_kernel_time_ms, Percentage (%) |
| kernel_summary | Kernel name (summed over ops and streams) | Kernel duration (µs)_sum, _count |
| steps | rank, step kind | steps, period (µs)_mean, _median, busy (µs)_mean, busy/period, kernels per step_mean |
| coll_analysis | Collective | Count, Duration (µs)_sum, Transfer (µs)_sum, Wait (µs)_sum_mean_rank |

Columns are `tag::column`, and for every report after the first
`tag::column__diff` (minus the baseline) and `tag::column__pct` (percent
change). Kernel names do not match across vendors or library versions; the
category, timeline and step tables are the ones that compare across them.
Output: `compare_<table>.csv` per table and `COMPARE.md`.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

from ruppell.birdeye.report import _md_table, _trunc, read_csv, write_csv
from ruppell.birdeye.analyses import Sheet

TABLES: List[Tuple[str, List[str], List[str]]] = [
    ("gpu_timeline", ["rank", "type"], ["time ms", "percent"]),
    ("category_summary", ["category"], ["Count", "Kernel duration (µs)_sum", "busy (µs)", "Percent of kernels time (%)"]),
    ("ops_summary_by_category", ["op category"], ["Count", "total_direct_kernel_time_ms", "Percentage (%)"]),
    ("ops_summary", ["name"], ["Count", "total_direct_kernel_time_ms", "Percentage (%)"]),
    ("kernel_summary", ["Kernel name"], ["Kernel duration (µs)_sum", "Kernel duration (µs)_count"]),
    ("steps", ["rank", "step kind"], ["steps", "period (µs)_mean", "period (µs)_median", "busy (µs)_mean",
                                      "busy/period", "kernels per step_mean"]),
    ("coll_analysis", ["Collective"], ["Count", "Duration (µs)_sum", "Transfer (µs)_sum", "Wait (µs)_sum_mean_rank"]),
]
# the column the summary ranks rows by
RANK_BY = {"gpu_timeline": "time ms", "category_summary": "Kernel duration (µs)_sum",
           "ops_summary_by_category": "total_direct_kernel_time_ms", "ops_summary": "total_direct_kernel_time_ms",
           "kernel_summary": "Kernel duration (µs)_sum", "steps": "period (µs)_mean", "coll_analysis": "Duration (µs)_sum"}


def _num(x) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _load(d: str, name: str, keys: List[str], vals: List[str]) -> Optional[Dict[tuple, Dict[str, float]]]:
    p = os.path.join(d, f"{name}.csv")
    if not os.path.exists(p):
        return None
    sh = read_csv(p)
    if not all(k in sh.columns for k in keys):
        return None
    ki = [sh.columns.index(k) for k in keys]
    vi = {v: sh.columns.index(v) for v in vals if v in sh.columns}
    out: Dict[tuple, Dict[str, float]] = {}
    for r in sh.rows:
        key = tuple(r[i] for i in ki)
        acc = out.setdefault(key, {})
        for v, i in vi.items():
            x = _num(r[i])
            if x is None:
                continue
            # rows sharing a key (kernel_summary over ops and streams) add up;
            # percentages and ratios of such rows add up too (shares of one whole)
            if v in acc and name == "kernel_summary":
                acc[v] += x
            else:
                acc[v] = x
    return out


def merge(dirs: List[str], tags: List[str]) -> Dict[str, Sheet]:
    out: Dict[str, Sheet] = {}
    for name, keys, vals in TABLES:
        loaded = [_load(d, name, keys, vals) for d in dirs]
        if all(x is None for x in loaded):
            continue
        allkeys: List[tuple] = []
        seen = set()
        for x in loaded:
            for k in (x or {}):
                if k not in seen:
                    seen.add(k)
                    allkeys.append(k)
        cols = list(keys)
        for j, tag in enumerate(tags):
            for v in vals:
                cols.append(f"{tag}::{v}")
                if j:
                    cols += [f"{tag}::{v}__diff", f"{tag}::{v}__pct"]
        rows = []
        for k in allkeys:
            row = list(k)
            base = (loaded[0] or {}).get(k, {})
            for j, x in enumerate(loaded):
                cur = (x or {}).get(k, {})
                for v in vals:
                    a = cur.get(v)
                    row.append(a)
                    if j:
                        b = base.get(v)
                        row.append(a - b if a is not None and b is not None else None)
                        row.append((a / b - 1) * 100 if a is not None and b not in (None, 0) else None)
            rows.append(row)
        out[name] = Sheet(f"compare_{name}", cols, rows)
    return out


def markdown(sheets: Dict[str, Sheet], tags: List[str], dirs: List[str], top: int = 20) -> str:
    L = ["# birdeye compare", "", f"Baseline: **{tags[0]}**. Reports:", ""]
    L += [f"- **{t}**: `{d}`" for t, d in zip(tags, dirs)]
    L += ["", "Kernel names do not match across vendors or library versions; the timeline, category and step "
              "tables compare across them.", ""]
    for name, keys, vals in TABLES:
        sh = sheets.get(name)
        if sh is None:
            continue
        v = RANK_BY[name]
        cols = sh.columns
        heads = [c for c in cols if c.endswith(f"::{v}")]
        diffs = [c for c in cols if c.endswith(f"::{v}__pct")]
        L += [f"## {name} (`compare_{name}.csv`), by {v}", ""]
        idx = [cols.index(h) for h in heads]

        def size(r):
            return max(abs(r[i]) for i in idx if r[i] is not None) if any(r[i] is not None for i in idx) else 0

        rows = sorted(sh.rows, key=lambda r: -size(r)) if name != "gpu_timeline" else sh.rows
        if name == "gpu_timeline":
            rows = [r for r in rows if r[0] == "mean"] or rows
        show = []
        for r in rows[:top]:
            cells = [f"`{_trunc(x, 60)}`" if j == len(keys) - 1 and name in ("kernel_summary", "ops_summary",
                                                                              "coll_analysis") else str(x)
                     for j, x in enumerate(r[:len(keys)])]
            for h in heads:
                x = r[cols.index(h)]
                cells.append("" if x is None else f"{x:.3f}")
            for dcol in diffs:
                x = r[cols.index(dcol)]
                cells.append("" if x is None else f"{x:+.1f}%")
            show.append(cells)
        L += _md_table(keys + [h.split("::")[0] for h in heads] + [d.split("::")[0] + " vs " + tags[0] for d in diffs],
                       show, set(range(len(keys), len(keys) + len(heads) + len(diffs))))
        L.append("")
    return "\n".join(L)


def compare(out_dir: str, inputs: List[str], tags: Optional[List[str]] = None, **run_kw) -> dict:
    from ruppell.birdeye import run

    if len(inputs) < 2:
        raise ValueError("birdeye-compare needs at least two reports or sources")
    if tags is None:
        tags = []
        for p in inputs:
            t = os.path.basename(p.rstrip("/\\")) or p
            t = t.replace("::", "_")
            while t in tags:
                t += "_"
            tags.append(t)
    if len(tags) != len(inputs):
        raise ValueError(f"{len(tags)} tags for {len(inputs)} inputs")
    os.makedirs(out_dir, exist_ok=True)
    dirs = []
    for p, tag in zip(inputs, tags):
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "birdeye.json")):
            dirs.append(p)
        else:
            d = os.path.join(out_dir, tag)
            run(p, out=d, **run_kw)
            dirs.append(d)
    sheets = merge(dirs, tags)
    files = []
    for name, sh in sheets.items():
        path = os.path.join(out_dir, f"compare_{name}.csv")
        write_csv(path, sh)
        files.append(path)
    with open(os.path.join(out_dir, "COMPARE.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(markdown(sheets, tags, dirs))
    files.append(os.path.join(out_dir, "COMPARE.md"))
    return {"out": out_dir, "tags": tags, "reports": dirs, "files": files}
