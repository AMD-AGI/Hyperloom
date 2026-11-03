"""Compare a ruppell run with an independent collector's trace of the same work.

The witness is rocprofv3's kernel trace (CSV). The strong form runs both tools
in the same processes (`rocprofv3 --kernel-trace -o %pid% -- python -m ruppell
run ...`): they observe the same dispatches through the same SDK, so kernels
are compared by the SDK's kernel id and every dispatch is matched one to one
by dispatch id. With several processes, each witness file is matched to the
ruppell process directory of the same pid (the file name, or its rows' pids).
For two separate runs the kernels are compared by demangled name instead;
demanglers differ in style, so that form is weaker.

On CUDA the witness is Nsight Systems (`nsys profile -t cuda
--cuda-graph-trace=node --export sqlite`), and it cannot share a process
with ruppell: CUPTI takes one subscriber (docs/feasibility-cuda.md). So it
traces a separate run of the same program in witness mode (markers
launched, no tool), and the comparison is the weaker one: kernel counts and
GPU time by name over the run, and, because both collectors see the step
markers, the kernels by name inside each step, step by step.
"""

from __future__ import annotations

import csv
import glob
import os
import re
from collections import Counter
from typing import Dict, List, Optional

import numpy as np

from ruppell.engine import MARKER_KERNEL, Proc, demangle, process_dirs


def _norm(name: str) -> str:
    return " ".join(name.split())


def _agent(v: Optional[str]) -> Optional[int]:
    """rocprofv3's Agent_Id: a number, or "Agent 2" (ROCm 7.2)."""
    if not v:
        return None
    v = v.strip()
    return int(v.split()[-1], 0) if v.lower().startswith("agent") else int(v, 0)


def _read_csv(path: str) -> Dict:
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        for row in reader:
            rows.append({
                "name": row.get("Kernel_Name") or row.get("Kernel-Name") or "",
                "start": int(row.get("Start_Timestamp") or 0),
                "end": int(row.get("End_Timestamp") or 0),
                "dispatch_id": int(row["Dispatch_Id"]) if row.get("Dispatch_Id") else None,
                "kernel_id": int(row["Kernel_Id"]) if row.get("Kernel_Id") else None,
                "pid": int(row["Pid"]) if row.get("Pid") else None,
                "queue": int(row["Queue_Id"], 0) if row.get("Queue_Id") else None,
                # also read by birdeye (ruppell.birdeye.readers)
                "agent": _agent(row.get("Agent_Id")),
                "stream": int(row["Stream_Id"], 0) if row.get("Stream_Id") else None,
                "corr": int(row["Correlation_Id"], 0) if row.get("Correlation_Id") else None,
            })
    return {"path": path, "header": header, "rows": rows}


def _read_nsys(path: str) -> Dict[int, List[dict]]:
    """Kernel rows of an nsys SQLite export, by process id."""
    import sqlite3

    con = sqlite3.connect(path)
    cols = {r[1] for r in con.execute("pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
    if not cols:
        raise ValueError(f"{path}: no CUPTI_ACTIVITY_KIND_KERNEL table (was nsys run with -t cuda?)")
    graph_level = con.execute("select count(*) from sqlite_master where name = 'CUPTI_ACTIVITY_KIND_GRAPH_TRACE'").fetchone()[0]
    if graph_level and con.execute("select count(*) from CUPTI_ACTIVITY_KIND_GRAPH_TRACE").fetchone()[0]:
        raise ValueError(f"{path}: graphs were traced as whole graphs; run nsys with --cuda-graph-trace=node")
    name_col = "mangledName" if "mangledName" in cols else "demangledName"
    # also read by birdeye (ruppell.birdeye.readers): the device, the launch's
    # correlation id and the demangled name, where the export has them
    dem = "d.value" if "demangledName" in cols else "s.value"
    dem_join = "join StringIds d on d.id = k.demangledName " if "demangledName" in cols else ""
    dev = "k.deviceId" if "deviceId" in cols else "0"
    corr = "k.correlationId" if "correlationId" in cols else "0"
    q = (f"select k.globalPid, k.start, k.end, k.streamId, k.graphNodeId, s.value, {dev}, {corr}, {dem} "
         f"from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id = k.{name_col} {dem_join}order by k.start")
    out: Dict[int, List[dict]] = {}
    for gpid, s, e, stream, node, name, device, cid, demangled in con.execute(q):
        pid = (int(gpid) >> 24) & 0xFFFFFF
        out.setdefault(pid, []).append({"name": name, "start": int(s), "end": int(e), "stream": int(stream),
                                        "graph_node": int(node or 0), "dispatch_id": None, "kernel_id": None,
                                        "pid": pid, "device": int(device or 0), "corr": int(cid or 0),
                                        "demangled": demangled})
    return out


# ---- memory copies and sets (ruppell's copies.bin) -----------------------------------------

def _read_copy_csv(path: str) -> List[dict]:
    """rocprofv3 --memory-copy-trace: <pid>_memory_copy_trace.csv."""
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            b = (r.get("Bytes") or r.get("Size") or "").strip()
            rows.append({"corr": int(r["Correlation_Id"], 0) if r.get("Correlation_Id") else None,
                         "start": int(r.get("Start_Timestamp") or 0), "end": int(r.get("End_Timestamp") or 0),
                         "direction": r.get("Direction") or "", "bytes": int(b) if b.isdigit() else None})
    return rows


def compare_copies(proc: Proc, wrows: List[dict]) -> Dict:
    """In the same process, through the same SDK: each copy rocprofv3 saw is
    one of ruppell's with the same correlation id, start and end."""
    c = proc.copies
    mine = Counter(zip(c["corr"].tolist(), c["start"].tolist(), c["end"].tolist()))
    theirs = Counter((r["corr"], r["start"], r["end"]) for r in wrows)
    matched = sum((mine & theirs).values())
    wb = [r["bytes"] for r in wrows if r["bytes"] is not None]
    return {"ruppell": int(len(c)), "witness": len(wrows), "matched_identical": matched,
            "all_matched": matched == len(c) == len(wrows),
            "bytes_ruppell": int(c["bytes"].sum()) if len(c) else 0, "bytes_witness": sum(wb) if wb else None}


def _nsys_copies(path: str) -> Dict[int, Counter]:
    """Per process: copies (by CUPTI copy kind) and sets, count and bytes, from an nsys export."""
    import sqlite3

    con = sqlite3.connect(path)
    tables = {r[0] for r in con.execute("select name from sqlite_master where type = 'table'")}
    out: Dict[int, Counter] = {}
    for table, kind in (("CUPTI_ACTIVITY_KIND_MEMCPY", "copyKind"), ("CUPTI_ACTIVITY_KIND_MEMSET", None)):
        if table not in tables:
            continue
        key = kind or "0"
        for gpid, k, n, b in con.execute(f"select globalPid, {key}, count(*), sum(bytes) from {table} group by 1, 2"):
            pid = (int(gpid) >> 24) & 0xFFFFFF
            name = f"memcpy:{int(k)}" if kind else "memset"
            e = out.setdefault(pid, Counter())
            e[name + ":count"] += int(n)
            e[name + ":bytes"] += int(b or 0)
    return out


def _ruppell_copies_by_kind(proc: Proc) -> Counter:
    """ruppell's copies in _nsys_copies's terms (CUDA copy kinds)."""
    e: Counter = Counter()
    c = proc.copies
    for kind, direction, b in zip(c["kind"].tolist(), c["direction"].tolist(), c["bytes"].tolist()):
        name = "memset" if kind == 2 else f"memcpy:{int(direction)}"
        e[name + ":count"] += 1
        e[name + ":bytes"] += int(b)
    return e


def _marker_steps(names: List[str], starts: List[int], ends: List[int]) -> Optional[List[tuple]]:
    """Step intervals from marker kernels alone (the witness knows no step
    ids): markers alternate begin, end in time order. None if they do not."""
    m = [i for i, n in enumerate(names) if MARKER_KERNEL in n]
    m.sort(key=lambda i: starts[i])
    if len(m) % 2:
        return None
    return [(starts[b], ends[e]) for b, e in zip(m[0::2], m[1::2])]


def per_step(proc: Proc, wrows: List[dict]) -> Dict:
    """Kernels by name inside each step, ruppell against the witness, the
    n-th step of one run against the n-th of the other."""
    d = proc.d
    raw = [proc.kernels.get(int(k), f"kernel_{int(k)}") for k in d["kernel_id"]]
    ours = _marker_steps(raw, d["start"].tolist(), d["end"].tolist())
    wit = _marker_steps([w["name"] for w in wrows], [w["start"] for w in wrows], [w["end"] for w in wrows])
    if ours is None or wit is None:
        return {"compared": False, "reason": "markers do not alternate begin, end in one of the runs"}
    if len(ours) != len(wit):
        return {"compared": False, "reason": f"{len(ours)} steps in the run, {len(wit)} in the witness"}

    def counts(names, starts, steps):
        st = np.array([s for s, _ in steps], dtype=np.int64)
        en = np.array([e for _, e in steps], dtype=np.int64)
        out = [Counter() for _ in steps]
        k = np.searchsorted(st, np.array(starts, dtype=np.int64), side="right") - 1
        for n, kk, s in zip(names, k.tolist(), starts):
            if kk >= 0 and s <= en[kk] and MARKER_KERNEL not in n:
                out[kk][n] += 1
        return out

    a = counts(raw, d["start"].astype(np.int64).tolist(), ours)
    b = counts([w["name"] for w in wrows], [w["start"] for w in wrows], wit)
    diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    return {"compared": True, "steps": len(ours), "steps_identical": len(ours) - len(diff),
            "kernels_in_steps": {"ruppell": sum(sum(c.values()) for c in a), "witness": sum(sum(c.values()) for c in b)},
            "first_differing_steps": [{"step": i, "only_ruppell": dict(a[i] - b[i]), "only_witness": dict(b[i] - a[i])}
                                      for i in diff[:5]]}


def _time_by_name(rows: List[dict]) -> Counter:
    t = Counter()
    for w in rows:
        t[w["name"]] += w["end"] - w["start"]
    return t


def compare_nsys(run_dir: str, sqlite_path: str, time_tolerance: float, spread_sqlite: Optional[str] = None) -> Dict:
    """``spread_sqlite``: a second witness run of the same program. Two runs
    differ in GPU time by themselves (clocks, placement), so with it the time
    check for each kernel allows the larger of the tolerance and the
    witness's own largest run-to-run difference."""
    wit = _read_nsys(sqlite_path)
    wcopies = _nsys_copies(sqlite_path)
    spread_rows = None
    if spread_sqlite:
        w2 = _read_nsys(spread_sqlite)
        spread_rows = max(w2.values(), key=len) if w2 else None
    procs = [pr for pr in (Proc(p) for p in process_dirs(run_dir)) if len(pr.d)]
    # separate runs: pids differ; pair processes by kernel count, largest first
    wp = sorted(wit, key=lambda k: -len(wit[k]))
    procs.sort(key=lambda pr: -len(pr.d))
    results = []
    for pr, pid in zip(procs, wp):
        r = compare_proc(pr, wit[pid], time_tolerance)
        r["witness_pid"] = pid
        r["compared_by"] = "kernel name, separate run"
        r["per_step"] = per_step(pr, wit[pid])
        if spread_rows is not None:
            a, b = _time_by_name(wit[pid]), _time_by_name(spread_rows)
            spread = max((abs(b[n] / a[n] - 1) for n in a if a[n] and b[n]), default=0.0)
            allowed = max(time_tolerance, spread)
            dem = demangle(list(a))
            for row in r["rows"]:
                row["time_within_tolerance"] = row["time_ratio"] is not None and abs(row["time_ratio"] - 1) <= allowed
            r["all_times_within_tolerance"] = all(row["time_within_tolerance"] for row in r["rows"])
            r["time_bar"] = {"tolerance": time_tolerance, "witness_run_to_run_max": round(spread, 4), "allowed": round(allowed, 4),
                             "witness_spread_by_kernel": {_norm(dem.get(n, n)): round(b[n] / a[n] - 1, 4) for n in a if a[n] and b[n]}}
        if pr.copies_recorded:
            # separate runs: copies and sets by kind and direction, count and bytes
            mine, theirs = _ruppell_copies_by_kind(pr), wcopies.get(pid, Counter())
            r["copies"] = {"ruppell": dict(mine), "witness": dict(theirs),
                           "equal": dict(mine) == dict(theirs)}
        results.append(r)
    return {
        "witness": "nsys -t cuda --cuda-graph-trace=node (separate run, witness mode)",
        "witness_files": [sqlite_path] + ([spread_sqlite] if spread_sqlite else []),
        "time_tolerance": time_tolerance,
        "processes": results,
        "unpaired": {"ruppell": len(procs) - len(results), "witness": len(wp) - len(results)},
        "in_process": False,
        "all_counts_equal": bool(results) and all(r["all_counts_equal"] for r in results),
        "all_steps_identical": bool(results) and all(r["per_step"].get("compared") and
                                                     r["per_step"]["steps_identical"] == r["per_step"]["steps"]
                                                     for r in results),
        "all_times_within_tolerance": bool(results) and all(r["all_times_within_tolerance"] for r in results),
        "copies_equal": (all(r["copies"]["equal"] for r in results if "copies" in r)
                         if any("copies" in r for r in results) else None),
    }


def _pid_of_file(path: str) -> int:
    m = re.findall(r"(\d+)", os.path.basename(path))
    return int(m[0]) if m else -1


def _sampled_steps_check(proc: Proc, wrows: List[dict]) -> Dict:
    """For a run with sampled detail: every dispatch the witness saw on a
    step's queue between that sampled step's begin and end markers must be
    in ruppell's records (turning dispatch tracing on and off loses nothing
    inside a sampled step)."""
    d = proc.d
    ext = d["ext"].astype(np.uint64)
    mk = (ext & np.uint64(1 << 63)) != 0
    code = ext[mk] & ~np.uint64(1 << 63)
    step, kind = (code >> np.uint64(2)).astype(np.int64), (code & np.uint64(3)).astype(np.int64)
    begin, end = {}, {}
    for i, s, k in zip(np.nonzero(mk)[0].tolist(), step.tolist(), kind.tolist()):
        (begin if k == 1 else end)[s] = i
    spans = [(int(d["queue_id"][begin[s]]), int(d["start"][begin[s]]), int(d["end"][end[s]]), s)
             for s in begin if s in end]
    sampled_ids = {s for _, _, _, s in spans}
    ours = np.sort(d["dispatch_id"].astype(np.int64))
    n_in = n_missing = 0
    examples: List[dict] = []
    by_queue: Dict[int, List[tuple]] = {}
    for q, lo, hi, s in spans:
        by_queue.setdefault(q, []).append((lo, hi, s))
    for q in by_queue:
        by_queue[q].sort()
    starts = {q: np.array([x[0] for x in v], dtype=np.int64) for q, v in by_queue.items()}
    for w in wrows:
        q = w.get("queue")
        if q is None or q not in by_queue or w["dispatch_id"] is None:
            continue
        j = int(np.searchsorted(starts[q], w["start"], side="right")) - 1
        # inside the step as the engine counts it: from its begin marker's start to
        # its end marker's end (the next step's begin marker can start a few µs
        # before this step's end marker ends; it belongs to that next step)
        if j < 0 or w["end"] > by_queue[q][j][1]:
            continue
        n_in += 1
        k = int(np.searchsorted(ours, w["dispatch_id"]))
        if k >= len(ours) or ours[k] != w["dispatch_id"]:
            n_missing += 1
            if len(examples) < 12:
                lo, hi, s = by_queue[q][j]
                examples.append({"kernel": w["name"][:80], "queue": q, "step": s, "dispatch_id": w["dispatch_id"],
                                 "ns_after_begin": w["start"] - lo, "ns_before_end": hi - w["end"],
                                 "first_of_run": (s - 1) not in sampled_ids, "last_of_run": (s + 1) not in sampled_ids})
    return {"sampled_steps": len(spans), "witness_dispatches_in_sampled_steps": n_in,
            "missing_from_ruppell": n_missing, "complete": n_missing == 0 and n_in > 0, "missing_examples": examples}


def compare_proc(proc: Proc, wrows: List[dict], time_tolerance: float) -> Dict:
    d = proc.d
    sampled = None
    if int(proc.meta.get("detail_toggles") or 0) > 0 and len(d):
        # detail sampling: ruppell recorded the sampled steps' dispatches only;
        # check completeness inside them, then compare on what ruppell recorded
        sampled = _sampled_steps_check(proc, wrows)
        ours = set(d["dispatch_id"].tolist())
        sampled["witness_dispatches_total"] = len(wrows)
        wrows = [w for w in wrows if w["dispatch_id"] in ours]
    have = [w for w in wrows if w["dispatch_id"] is not None]
    matched = same_time = same_kernel = max_dt = 0
    if have and len(d):
        w_id = np.array([w["dispatch_id"] for w in have], dtype=np.int64)
        w_k = np.array([w["kernel_id"] if w["kernel_id"] is not None else -1 for w in have], dtype=np.int64)
        w_s = np.array([w["start"] for w in have], dtype=np.int64)
        w_e = np.array([w["end"] for w in have], dtype=np.int64)
        o = np.argsort(w_id)
        w_id, w_k, w_s, w_e = w_id[o], w_k[o], w_s[o], w_e[o]
        did = d["dispatch_id"].astype(np.int64)
        pos = np.clip(np.searchsorted(w_id, did), 0, len(w_id) - 1)
        hit = w_id[pos] == did
        p = pos[hit]
        dt = np.maximum(np.abs(w_s[p] - d["start"][hit].astype(np.int64)), np.abs(w_e[p] - d["end"][hit].astype(np.int64)))
        matched = int(hit.sum())
        same_time = int((dt == 0).sum())
        same_kernel = int((w_k[p] == d["kernel_id"][hit].astype(np.int64)).sum())
        max_dt = int(dt.max()) if len(dt) else 0
    in_process = bool(have) and matched == len(d) == len(wrows) and same_kernel == matched
    if in_process:
        mode = "kernel_id"
        label = {}
        for w in wrows:
            label.setdefault(w["kernel_id"], _norm(w["name"]))
        ours_key = d["kernel_id"].tolist()
        wit_key = [w["kernel_id"] for w in wrows]
    else:
        mode = "demangled name"
        raw = [proc.kernels.get(int(k), f"kernel_{int(k)}") for k in d["kernel_id"]]
        dem = demangle(raw + [w["name"] for w in wrows])
        ours_key = [_norm(dem.get(r, r)) for r in raw]
        wit_key = [_norm(dem.get(w["name"], w["name"])) for w in wrows]
        label = {k: k for k in set(ours_key) | set(wit_key)}
    dur = (d["end"] - d["start"]).tolist()
    oc, ot, wc, wt = Counter(), Counter(), Counter(), Counter()
    for k, t in zip(ours_key, dur):
        oc[k] += 1
        ot[k] += t
    for k, w in zip(wit_key, wrows):
        wc[k] += 1
        wt[k] += w["end"] - w["start"]
    rows = []
    for k in sorted(set(oc) | set(wc), key=lambda x: str(label.get(x, x))):
        ratio = (ot[k] / wt[k]) if wt[k] else None
        rows.append({"kernel": label.get(k, str(k)), "key": k, "count_ruppell": oc[k], "count_witness": wc[k],
                     "counts_equal": oc[k] == wc[k], "gpu_time_ns_ruppell": ot[k], "gpu_time_ns_witness": wt[k],
                     "time_ratio": round(ratio, 4) if ratio is not None else None,
                     "time_within_tolerance": ratio is not None and abs(ratio - 1.0) <= time_tolerance})
    return {
        "process": proc.label, "in_process": in_process, "compared_by": mode,
        "per_dispatch": {"ruppell_dispatches": int(len(d)), "witness_dispatches": len(wrows), "matched_by_dispatch_id": matched,
                         "same_kernel_id": same_kernel, "identical_start_and_end": same_time, "max_timestamp_difference_ns": max_dt},
        "kernels": len(rows),
        "all_counts_equal": all(r["counts_equal"] for r in rows),
        "all_times_within_tolerance": all(r["time_within_tolerance"] for r in rows),
        "sampled_detail": sampled,
        "rows": rows,
    }


def compare(run_dir: str, witness_dir: str, time_tolerance: float = 0.03, spread_dir: Optional[str] = None) -> Dict:
    sq = sorted(glob.glob(os.path.join(witness_dir, "**", "*.sqlite"), recursive=True))
    if sq:
        sp = sorted(glob.glob(os.path.join(spread_dir, "**", "*.sqlite"), recursive=True)) if spread_dir else []
        return compare_nsys(run_dir, sq[0], time_tolerance, sp[0] if sp else None)
    files = sorted(glob.glob(os.path.join(witness_dir, "**", "*kernel_trace.csv"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no *kernel_trace.csv under {witness_dir}")
    wit = [_read_csv(f) for f in files]
    procs = [Proc(p) for p in process_dirs(run_dir)]
    results = []
    if len(procs) == 1 and len(wit) == 1:
        pairs = [(procs[0], wit[0]["rows"])]
    else:
        pairs = []
        for pr in procs:
            pid = int(pr.meta.get("pid", -1))
            rows = []
            for w in wit:
                if _pid_of_file(w["path"]) == pid:
                    rows = w["rows"]
                    break
                if w["rows"] and w["rows"][0].get("pid") == pid:
                    rows = [r for r in w["rows"] if r.get("pid") == pid]
                    break
            if len(pr.d) or rows:
                pairs.append((pr, rows))
    # memory copies: rocprofv3 --memory-copy-trace, one file per process (-o %pid%)
    cfiles = sorted(glob.glob(os.path.join(witness_dir, "**", "*memory_copy_trace.csv"), recursive=True))
    crows = {f: _read_copy_csv(f) for f in cfiles}
    for pr, rows in pairs:
        r = compare_proc(pr, rows, time_tolerance)
        if pr.copies_recorded and cfiles:
            own = [f for f in cfiles if _pid_of_file(f) == int(pr.meta.get("pid", -1))]
            if own or len(cfiles) == 1 == len(pairs):
                r["copies"] = compare_copies(pr, crows[own[0] if own else cfiles[0]])
        results.append(r)
    return {
        "witness": "rocprofv3 --kernel-trace" + (" --memory-copy-trace" if cfiles else ""),
        "copies_all_matched": (all(r["copies"]["all_matched"] for r in results if "copies" in r)
                               if any("copies" in r for r in results) else None),
        "witness_files": files + cfiles,
        "witness_columns": wit[0]["header"] if wit else [],
        "time_tolerance": time_tolerance,
        "processes": results,
        "in_process": all(r["in_process"] for r in results) if results else False,
        "all_counts_equal": all(r["all_counts_equal"] for r in results) if results else False,
        "all_times_within_tolerance": all(r["all_times_within_tolerance"] for r in results) if results else False,
        # detail sampling: every witness dispatch inside a sampled step is in ruppell's records
        "sampled_steps_complete": (all(r["sampled_detail"]["complete"] for r in results if r["sampled_detail"])
                                   if any(r["sampled_detail"] for r in results) else None),
    }
