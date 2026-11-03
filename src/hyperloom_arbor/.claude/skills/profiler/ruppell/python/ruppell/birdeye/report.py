"""birdeye's outputs: one CSV per table (`OUT/<sheet>.csv`, TraceLens's file
names), `OUT/birdeye.json` (provenance, coverage, what was skipped and why),
and `OUT/REPORT.md`, the readable report."""

from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Optional

from ruppell.birdeye.analyses import Sheet
from ruppell.limits import LIMITS


def _cell(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, (list, tuple, set)):
        return str(sorted(v) if isinstance(v, set) else list(v))
    return v


def write_csv(path: str, sheet: Sheet) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(sheet.columns)
        for r in sheet.rows:
            w.writerow([_cell(x) for x in r])


def read_csv(path: str) -> Sheet:
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        cols = next(r, [])
        rows = [list(x) for x in r]
    return Sheet(os.path.splitext(os.path.basename(path))[0], cols, rows)


# ---- formatting ------------------------------------------------------------------------------

def _f(v, nd=3) -> str:
    if v is None or v == "":
        return ""
    if isinstance(v, float):
        if abs(v) >= 1e5:
            return f"{v:,.0f}"
        return f"{v:.{nd}f}"
    return str(v)


def _pct(v, nd=1) -> str:
    return "" if v is None else f"{v:.{nd}f}%"


def _md_table(cols: List[str], rows: List[list], align_right: Optional[set] = None) -> List[str]:
    def esc(x):
        return str(x).replace("|", "\\|").replace("\n", " ")
    out = ["| " + " | ".join(cols) + " |",
           "| " + " | ".join("---:" if (align_right and i in align_right) else "---" for i in range(len(cols))) + " |"]
    out += ["| " + " | ".join(esc(x) for x in r) + " |" for r in rows]
    return out


def _trunc(s: str, n: int = 90) -> str:
    s = str(s)
    return s if len(s) <= n else s[:n - 3] + "..."


def limitations(info: dict, res: dict) -> List[dict]:
    """The source's own limitations (ruppell runs and exports carry theirs)
    and those birdeye's tables add."""
    out = list(info.get("limitations") or [])
    have = {x["id"] for x in out}
    add = ["comm-by-name", "categories-by-name"]
    if "GEMM" in res["skipped"] and "shapes" in str(res["skipped"]["GEMM"]):
        add.append("gemm-needs-shapes")
    if info.get("platform") == "cuda" or (info.get("build") or {}).get("platform") == "cuda":
        add.append("software-timestamps")
    if info.get("copies_recorded") is False:
        add.append("memcpy-unrecorded")
    if info.get("counter_steps_left_out"):
        add.append("counter-steps")
        if info.get("platform") == "cuda" or (info.get("build") or {}).get("platform") == "cuda":
            add.append("counters-per-step-cuda")
    for i in add:
        if i not in have and i in LIMITS:
            out.append({"id": i, **LIMITS[i]})
    return out


def _provenance(info: dict, analysis_commit: Optional[str]) -> List[list]:
    fmt = info.get("format")
    rows = [["Source", f"`{info.get('source')}`"],
            ["Format", "JSON trace (`--format kineto`)" if fmt == "kineto" else f"`--format {fmt}`"],
            ["Profiler", info.get("profiler")]]
    b = info.get("build")
    if b:
        rows.append(["Recorded by", f"ruppell `{b.get('commit')}`, tool library sha256 `{b.get('tool_library_sha256')}`, "
                                    f"marker library sha256 `{b.get('marker_library_sha256')}` ({b.get('platform')}, "
                                    f"mode {b.get('mode')})"])
        if info.get("builds_differ"):
            rows.append(["", "**processes were recorded by different builds** (see birdeye.json)"])
    elif info.get("format") == "ruppell":
        rows.append(["Recorded by", "**unknown: no run.json** (the commit and library digest cannot be named)"])
    for f in info.get("files") or []:
        rows.append(["File", f"`{f['path']}`, {f['bytes']:,} bytes, sha256 `{f['sha256']}`..."])
    rows.append(["Analysed by", f"ruppell birdeye at `{analysis_commit or 'unknown commit'}`"])
    if info.get("window"):
        rows.append(["Window in the source", f"{info['window']}"])
    return rows


def _stands_out(res: dict, sheets: Dict[str, Sheet], short_us: float) -> List[str]:
    s = res["summary"]
    out: List[str] = []
    rank = s["rank"]
    tl = {r[1]: r[1:] for r in sheets["gpu_timeline"].rows if r[0] == rank}   # type: [type, ms, percent]
    if tl:
        out.append(f"GPU idle {_pct(tl['idle_time'][2])} of the window ({_f(tl['idle_time'][1])} ms "
                   f"of {_f(tl['total_time'][1])} ms), {rank}.")
        if tl["total_comm_time"][1]:
            hidden = 1 - tl["exposed_comm_time"][1] / tl["total_comm_time"][1]
            out.append(f"Collectives take {_f(tl['total_comm_time'][1])} ms; {_pct(hidden * 100)} of it runs under "
                       f"computation, so {_pct(tl['exposed_comm_time'][2], 2)} of the window is exposed communication.")
        else:
            out.append("No collective kernels on this rank (by kernel name).")
    cat = sheets.get("category_summary")
    if cat and cat.rows:
        c = cat.rows[0]
        out.append(f"Largest kernel category: {c[0]}, {_pct(c[4])} of kernel time ({_f(c[2] / 1000)} ms, {c[1]:,} "
                   f"activities).")
    ks = sheets.get("kernel_summary")
    if ks and ks.rows:
        by: Dict[str, list] = {}
        for r in ks.rows:
            x = by.setdefault(r[2], [0.0, 0, r[0]])
            x[0] += r[4]
            x[1] += r[5]
        name, (t, n, c) = max(by.items(), key=lambda kv: kv[1][0])
        tot = sum(v[0] for v in by.values())
        from ruppell.engine import short_name
        out.append(f"Largest kernel: `{_trunc(short_name(name), 80)}` ({c}), {_pct(t / tot * 100)} of kernel time, "
                   f"{n:,} launches.")
    if s["kernels"]:
        out.append(f"{_pct(s['short_kernels'] / s['kernels'] * 100, 0)} of kernels run under {short_us:g} µs; "
                   f"they take {_pct(s['short_kernel_time_ns'] / s['kernel_time_ns'] * 100 if s['kernel_time_ns'] else 0)} "
                   f"of kernel time.")
    if tl and tl["busy_time"][1]:
        ratio = s["kernel_time_ns"] / 1e6 / tl["busy_time"][1]
        if ratio > 1.05:
            out.append(f"Kernels overlap: kernel time is {ratio:.2f}x busy time (several streams at once), so "
                       f"shares of kernel time are not shares of wall time.")
    ops = sheets.get("ops_summary")
    if ops and ops.rows:
        r = ops.rows[0]
        out.append(f"Top op: `{_trunc(r[0], 80)}`, {_pct(r[5])} of direct kernel time ({r[3]:,} invocations).")
    if s["activity_time_ns"]:
        no_op = 1 - s["with_op_time_ns"] / s["activity_time_ns"]
        if no_op > 0.01 and s.get("without_op_by_kernel"):
            k = s["without_op_by_kernel"][0]
            out.append(f"{_pct(no_op * 100)} of GPU time has no op; the largest such kernel is `{_trunc(k['kernel'], 60)}`.")
    st = sheets.get("steps")
    if st and st.rows:
        mine = [r for r in st.rows if r[0] == rank and r[8] is not None and r[3]]
        if mine:
            lo = min(mine, key=lambda r: r[8])
            out.append(f"Least busy step kind on {rank}: {lo[1]}, GPU busy {_pct(lo[8] * 100)} of its period "
                       f"(median period {_f(lo[5], 1)} µs over {lo[2]:,} steps).")
    co = sheets.get("coll_analysis")
    if co and co.rows and len(res["summary"]["ranks"]) > 1:
        tr = sum(r[8] or 0 for r in co.rows)
        wt = sum(r[9] or 0 for r in co.rows)
        if tr or wt:
            out.append(f"Collectives across ranks: transfer {_f(tr / 1000)} ms, waiting {_f(wt / 1000)} ms per rank on "
                       f"average ({_pct(wt / (tr + wt) * 100)} waiting; collective-wait: untrusted when traced).")
    return out


def markdown(info: dict, res: dict, opts: dict, analysis_commit: Optional[str]) -> str:
    sheets: Dict[str, Sheet] = res["sheets"]
    s = res["summary"]
    top = int(opts.get("top", 30))
    src = os.path.basename(str(info.get("source", "")).rstrip("/\\")) or str(info.get("source"))
    L = [f"# birdeye: {src}", ""]
    L += _md_table(["", ""], _provenance(info, analysis_commit))
    L += ["", f"Ranks: {', '.join(s['ranks'])}. Per-rank tables (kernels, ops, categories, short kernels, GEMM) are "
              f"for **{s['rank']}**; the timeline, steps and collectives cover every rank.",
          f"Steps: {opts.get('steps') or 'every step in the source'}"
          + (f"; phases {opts['phases']}" if opts.get("phases") else "")
          + (f" ({info['steps_from']})" if info.get("steps_from") else "") + ".",
          "Collectives: " + ("TraceLens's names (nccl, rccl, cross_device_reduce)" if s["comm"] == "tracelens"
                             else "ruppell's classifier (`ruppell.limits.COLLECTIVE`)") + ".", ""]

    # coverage
    L += ["## Coverage", ""]
    kt = s["kernel_time_ns"]
    allt = max(1, s["activity_time_ns"])
    L.append(f"- {s['kernels']:,} kernels on {s['rank']} ({s['activities']:,} GPU activities with copies and fills), "
             f"{kt / 1e6:.3f} ms of kernel time over a {s['total_time_ns'] / 1e6:.3f} ms window.")
    L.append(f"- With an op: {s['with_op']:,} activities, {_pct(s['with_op_time_ns'] / allt * 100)} of GPU time"
             + (f"; with a ruppell scope: {_pct(s['with_scope_time_ns'] / allt * 100)}" if s["with_scope"] else "") + ".")
    L.append(f"- Unlinked (no launching call in the source): {s['unlinked']:,}. Outside every step: {s['outside_steps']:,}"
             f" of {s['activities']:,}; steps on this rank: {s['steps']:,}.")
    for c in info.get("coverage") or []:
        bits = []
        for k in ("attributed_share", "dropped_records", "records_complete", "graph_replays_refused",
                  "kernels_outside_any_step", "every_kernel_in_exactly_one_step", "unlinked_kernels",
                  "kernels_with_an_op", "cpu_ops_gpu_time_share"):
            if c.get(k) is not None:
                v = c[k]
                bits.append(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}")
        if bits:
            L.append(f"- As recorded, {c.get('process')}: " + ", ".join(bits) + ".")
    if s.get("without_op_by_kernel"):
        L += ["", "Activities without an op, by kernel (this rank):", ""]
        L += _md_table(["kernel", "count", "ms"], [[f"`{_trunc(x['kernel'], 80)}`", f"{x['count']:,}",
                                                    _f(x["time_ns"] / 1e6)] for x in s["without_op_by_kernel"][:10]],
                       {1, 2})
    for n in info.get("notes") or []:
        L.append(f"- Note: {n}")
    lim = limitations(info, res)
    if lim:
        L += ["", "Limitations that hold for these numbers (`ruppell.limits`):", ""]
        L += [f"- **{x['id']}** ({x['applies_to']}): {x['text']}" for x in lim]
    L += ["", "## What stands out", ""]
    L += [f"{i}. {x}" for i, x in enumerate(_stands_out(res, sheets, float(opts.get("short_us", 10))), 1)]

    # gpu timeline, pivoted
    tl = sheets["gpu_timeline"]
    ranks = []
    for r in tl.rows:
        if r[0] not in ranks:
            ranks.append(r[0])
    L += ["", "## GPU timeline (`gpu_timeline.csv`)", "",
          "ms, and percent of the window (last end minus first start), per rank:", ""]
    val = {(r[0], r[1]): r[1:] for r in tl.rows}   # [type, ms, percent]
    types = [r[1] for r in tl.rows if r[0] == ranks[0]] if ranks else []
    L += _md_table(["type"] + ranks, [[ty] + [f"{_f(val[(rk, ty)][1])} ({_pct(val[(rk, ty)][2])})" for rk in ranks]
                                      for ty in types],
                   set(range(1, len(ranks) + 1)))
    tk = sheets.get("gpu_timeline_by_step_kind")
    if tk and tk.rows:
        L += ["", "Inside steps, by step kind (each step's activities clipped to its GPU period; "
                  "`gpu_timeline_by_step_kind.csv`), " + s["rank"] + ":", ""]
        g: Dict[str, dict] = {}
        for r in tk.rows:
            if r[0] == s["rank"]:
                g.setdefault(r[1], {"steps": r[2]})[r[3]] = (r[4], r[5])
        L += _md_table(["step kind", "steps", "total ms", "computation", "exposed comm", "exposed memcpy", "idle"],
                       [[k, f"{v['steps']:,}", _f(v["total_time"][0]), _pct(v["computation_time"][1]),
                         _pct(v["exposed_comm_time"][1], 2), _pct(v["exposed_memcpy_time"][1], 2),
                         _pct(v["idle_time"][1])] for k, v in g.items()], {1, 2, 3, 4, 5, 6})

    cat = sheets.get("category_summary")
    if cat:
        L += ["", "## Kernel categories (`category_summary.csv`)", "",
              "By kernel name (`ruppell/birdeye/categories.py`), every activity on " + s["rank"] + ":", ""]
        L += _md_table(["category", "count", "kernel ms", "busy ms", "% of kernel time", "% of window"],
                       [[r[0], f"{r[1]:,}", _f(r[2] / 1000), _f(r[3] / 1000), _pct(r[4]), _pct(r[5])] for r in cat.rows],
                       {1, 2, 3, 4, 5})

    ks = sheets.get("kernel_summary")
    if ks:
        L += ["", f"## Top kernels (`kernel_summary.csv`, {len(ks.rows):,} rows)", ""]
        from ruppell.engine import short_name
        L += _md_table(["category", "parent op", "kernel", "stream", "ms", "count", "mean µs", "% kernel time"],
                       [[r[0], f"`{_trunc(r[1], 40)}`", f"`{_trunc(short_name(r[2]), 70)}`", r[3], _f(r[4] / 1000),
                         f"{r[5]:,}", _f(r[6], 2), _pct(r[9])] for r in ks.rows[:top]], {3, 4, 5, 6, 7})

    for name, title in (("ops_summary", "Ops"), ("ops_summary_by_category", "Op categories")):
        sh = sheets.get(name)
        L += ["", f"## {title} (`{name}.csv`)", ""]
        if sh is None:
            L.append(f"Not computed: {res['skipped'].get(name)}.")
            continue
        if name == "ops_summary":
            L += ["Direct kernel time: per op invocation, the union of its kernels' intervals (collectives excluded, "
                  "as TraceLens does); summed per op.", ""]
            L += _md_table(["op", "ms", "invocations", "categories", "%", "cumulative %"],
                           [[f"`{_trunc(r[0], 70)}`", _f(r[2]), f"{r[3]:,}", ", ".join(r[4]), _pct(r[5]), _pct(r[6])]
                            for r in sh.rows[:top]], {1, 2, 4, 5})
        else:
            L += _md_table(["op category", "invocations", "ms", "%", "cumulative %"],
                           [[r[0], f"{r[1]:,}", _f(r[2]), _pct(r[3]), _pct(r[4])] for r in sh.rows], {1, 2, 3, 4})

    co = sheets.get("coll_analysis")
    L += ["", "## Collectives (`coll_analysis.csv`)", ""]
    if co is None:
        L.append(f"Not computed: {res['skipped'].get('coll_analysis')}.")
    else:
        L += ["Every rank. Transfer: each aligned launch's shortest duration across ranks; waiting: the rest, mean "
              "per rank (untrusted when traced: collective-wait)." + (f" {co.note}." if co.note else ""), ""]
        L += _md_table(["collective", "ranks", "launches", "ms", "mean µs", "aligned", "transfer ms",
                        "wait ms / rank", "wait %", "skew µs mean", "skew µs max"],
                       [[f"`{_trunc(r[0], 60)}`", r[1], f"{r[2]:,}", _f(r[3] / 1000), _f(r[4], 2), r[7] if r[7] is not None else "",
                         _f(r[8] / 1000 if r[8] is not None else None), _f(r[9] / 1000 if r[9] is not None else None),
                         _pct(r[10]), _f(r[11], 2), _f(r[12], 2)] for r in co.rows[:top]],
                       set(range(1, 11)))
    cb = sheets.get("coll_by_op")
    if cb is not None:
        L += ["", "### By call (`coll_by_op.csv`)", "",
              "The collective calls an adapter recorded (roadmap-v2 M8): operation, backend, size and group, over "
              "every rank; bandwidth with nccl-tests' conventions" + (f" ({cb.note})" if cb.note else "") + ".", ""]
        L += _md_table(["operation", "backend", "dtype", "bytes", "group", "launches", "replayed", "ms",
                        "transfer µs", "algbw GB/s", "busbw GB/s", "wait %", "NCCL"],
                       [[r[0], r[1], r[2], f"{r[4]:,}" if isinstance(r[4], (int, float)) else r[4], r[5], f"{r[7]:,}",
                         f"{r[8]:,}", _f(r[9] / 1000), _f(r[12], 2), _f(r[13], 1), _f(r[14], 1), _pct(r[15]),
                         f"{r[16]}/{r[17]}, {r[18]} ch" if r[16] else ""]
                        for r in cb.rows[:top]], set(range(3, 12)))

    sk = sheets.get("short_kernels_summary")
    L += ["", f"## Short kernels (under {float(opts.get('short_us', 10)):g} µs; `short_kernels_summary.csv`, "
              "`short_kernel_histogram.csv`)", ""]
    if sk is None or not sk.rows:
        L.append("None.")
    else:
        tot_n = sum(r[3] for r in sk.rows)
        tot_t = sum(r[2] for r in sk.rows)
        L += [f"{tot_n:,} short kernels, {_f(tot_t / 1000)} ms, {_pct(tot_t / max(1e-9, s['kernel_time_ns'] / 1000) * 100)} "
              "of kernel time. Largest:", ""]
        from ruppell.engine import short_name
        L += _md_table(["parent op", "kernel", "count", "ms", "mean µs", "% of window"],
                       [[f"`{_trunc(r[0], 40)}`", f"`{_trunc(short_name(r[1]), 70)}`", f"{r[3]:,}", _f(r[2] / 1000),
                         _f(r[4], 2), _pct(r[5], 2)] for r in sk.rows[:min(top, 15)]], {2, 3, 4, 5})

    st = sheets.get("steps")
    L += ["", "## Steps (`steps.csv`)", ""]
    if st is None:
        L.append(f"Not computed: {res['skipped'].get('steps')}.")
    else:
        L += ["Period: begin marker to end marker on the GPU; busy: the union of the step's activities inside it.", ""]
        L += _md_table(["rank", "step kind", "steps", "period µs mean", "median", "p90", "busy µs mean", "busy/period",
                        "kernels/step", "kernel µs/step", "graph", "eager"],
                       [[r[0], r[1], f"{r[2]:,}", _f(r[4], 1), _f(r[5], 1), _f(r[6], 1), _f(r[7], 1),
                         _pct(r[8] * 100 if r[8] is not None else None), _f(r[9], 1), _f(r[10], 1), _f(r[11], 1),
                         _f(r[12], 1)] for r in st.rows], set(range(2, 12)))

    gm = sheets.get("GEMM")
    L += ["", "## GEMM (`GEMM.csv`)", ""]
    if gm is None:
        L.append(f"Not computed: {res['skipped'].get('GEMM')}.")
    else:
        L += ["FLOPs = 2·B·M·N·K; bytes = (M·K + K·N + M·N)·B at the operands' element sizes; times are each "
              "invocation's direct kernel time.", ""]
        L += _md_table(["op", "B", "M", "N", "K", "dtypes", "GFLOP", "TFLOP/s mean", "TB/s mean", "µs mean", "count"],
                       [[r[0], r[1], r[2], r[3], r[4], r[5], _f(r[6], 4), _f(r[9], 2), _f(r[10], 3), _f(r[12], 2),
                         f"{r[13]:,}"] for r in gm.rows[:top]], set(range(1, 11)) - {5})
    skipped = {k: v for k, v in res["skipped"].items()}
    L += ["", "## Files", ""]
    L += [f"- `{k}.csv`" for k in sheets] + ["- `birdeye.json`: provenance, coverage, skipped tables and why"]
    if skipped:
        L += [f"- not written: `{k}.csv` ({v})" for k, v in skipped.items()]
    return "\n".join(L) + "\n"


def write(out_dir: str, info: dict, res: dict, opts: dict, analysis_commit: Optional[str]) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, sheet in res["sheets"].items():
        p = os.path.join(out_dir, f"{name}.csv")
        write_csv(p, sheet)
        paths.append(p)
    meta = {"source": info.get("source"), "format": info.get("format"), "profiler": info.get("profiler"),
            "build": info.get("build"), "files": info.get("files"), "analysis_commit": analysis_commit,
            "options": opts, "summary": res["summary"], "skipped": res["skipped"],
            "coverage": info.get("coverage"), "notes": info.get("notes"), "steps_from": info.get("steps_from"),
            "limitations": limitations(info, res), "merge": info.get("merge"),
            "notes_per_sheet": {k: v.note for k, v in res["sheets"].items() if v.note}}
    with open(os.path.join(out_dir, "birdeye.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, default=str)
    with open(os.path.join(out_dir, "REPORT.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(markdown(info, res, opts, analysis_commit))
    return paths + [os.path.join(out_dir, "birdeye.json"), os.path.join(out_dir, "REPORT.md")]
