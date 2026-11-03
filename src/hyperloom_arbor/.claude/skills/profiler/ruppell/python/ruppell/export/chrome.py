"""Chrome trace (Trace Event Format JSON) from a run directory.

A run holds one directory per traced process (M2) or is one process itself
(M1). Each process gets:
- a host process: one thread per launching CPU thread, HIP API calls;
- one GPU process per agent it used: one thread per hardware queue (with
  overlap lanes: a track cannot hold overlapping slices, and kernels on one
  queue can overlap by a few hundred ns), a "steps" track and a
  "ruppell markers" track;
- flow arrows from each launching call to the kernels it produced.

`steps=(A, B)` exports a window: the engine steps A..B of every process (or,
for runs without engine steps, the steps A..B), with the kernels and calls
that start inside it. The census always covers every step.

Timestamps are the SDK's, shared by processes on one node; the validation
block reports the clock-pair spread between processes.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import numpy as np

from ruppell.engine import (COPY_MEMSET, HOW_NAMES, OP_ATEN, OP_SCOPE, RANGE_MARK, RANGE_PUSH, RANGE_START, Analysis,
                            analyze_run, short_name)
from ruppell.limits import for_run, is_collective
from ruppell.validate.schema import validate_run

STEPS_TID = 0  # lanes 0..99
MARKERS_TID = 100  # lanes 100..199; queue q uses (q + 1) * 1000 + lane
COPIES_TID = 200  # lanes 200..299: memory copies and sets on the GPU they touch
NVTX_TID = 1 << 30  # a host thread's NVTX (ROCTX) ranges: their own track, NVTX_TID + the thread id


def _lanes(items: List[Tuple[int, int, int]]) -> Dict[int, int]:
    """items: (start, end, key). key -> first free lane."""
    ends: List[int] = []
    out: Dict[int, int] = {}
    for s, e, k in sorted(items):
        for li, le in enumerate(ends):
            if le <= s:
                ends[li] = e
                out[k] = li
                break
        else:
            ends.append(e)
            out[k] = len(ends) - 1
    return out


def _window(a: Analysis, steps: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    if steps is None or not a.step_list:
        return None
    lo, hi = steps
    engine = [s for s in a.step_list if s["fields"].get("phase") == "engine"]
    pool = engine if engine else a.step_list
    sel = [s for s in pool if lo <= s["step"] <= hi]
    if not sel:
        return (0, 0)
    return (min(s["start"] for s in sel), max(s["end"] for s in sel))


def _clock_offsets(analyses: List[Analysis]) -> Dict[str, float]:
    """sdk minus realtime, per process (median over its clock pairs)."""
    out = {}
    for a in analyses:
        c = a.proc.clock
        if c:
            diffs = sorted(((x[0] + x[1]) / 2.0) - x[2] for x in c)
            out[a.proc.label] = diffs[len(diffs) // 2]
    return out


def build(run_dir: str, flows: bool = True, steps: Optional[Tuple[int, int]] = None) -> dict:
    res = analyze_run(run_dir)
    analyses: List[Analysis] = res["analyses"]
    windows = [_window(a, steps) for a in analyses]

    def keep_mask(a: Analysis, w, starts: np.ndarray) -> np.ndarray:
        if w is None:
            return np.ones(len(starts), bool)
        return (starts >= w[0]) & (starts <= w[1])

    t0 = None
    for a, w in zip(analyses, windows):
        for arr in (a.proc.d["start"], a.proc.api["start"], a.proc.ops["t0"], a.proc.copies["start"]):
            m = keep_mask(a, w, arr)
            if m.any():
                v = int(arr[m].min())
                t0 = v if t0 is None else min(t0, v)
    t0 = t0 or 0

    def us(t) -> float:
        return round((int(t) - t0) / 1000.0, 3)

    ev: List[dict] = []
    reports = []
    lanes_added = 0
    for k, (a, w) in enumerate(zip(analyses, windows)):
        p, d = a.proc, a.proc.d
        host_pid = 10 * k + 1
        cuda = p.platform == "cuda"
        ev.append({"ph": "M", "name": "process_name", "pid": host_pid,
                   "args": {"name": f"{p.label}: host ({'CUDA' if cuda else 'HIP'} API)"}})
        ev.append({"ph": "M", "name": "process_sort_index", "pid": host_pid, "args": {"sort_index": 10 * k}})
        dm = keep_mask(a, w, d["start"]) if len(d) else np.zeros(0, bool)
        idx = np.nonzero(dm)[0]
        agents: Dict[int, int] = {}
        for ag in sorted({int(x) for x in d["agent_id"][idx]}):
            agents[ag] = 10000 + 100 * k + len(agents)
        qindex: Dict[tuple, int] = {}
        for ag, q in sorted({(int(x), int(y)) for x, y in zip(d["agent_id"][idx], d["queue_id"][idx])}):
            qindex[(ag, q)] = 1 + sum(1 for kk in qindex if kk[0] == ag)
        by_track: Dict[tuple, List[Tuple[int, int, int]]] = {}
        for i in idx.tolist():
            ag = int(d["agent_id"][i])
            track = (ag, "marker") if a.is_marker[i] else (ag, int(d["queue_id"][i]))
            by_track.setdefault(track, []).append((int(d["start"][i]), int(d["end"][i]), i))
        tid_of: Dict[int, int] = {}
        names: Dict[Tuple[int, int], str] = {}
        for (ag, q), items in by_track.items():
            pid = agents[ag]
            for i, lane in _lanes(items).items():
                if q == "marker":
                    tid = MARKERS_TID + lane
                    label = "ruppell markers" + (f" (overlap {lane})" if lane else "")
                else:
                    qi = qindex[(ag, q)]
                    tid = (qi + 1) * 1000 + lane
                    label = (f"stream {q}" if cuda else f"queue {qi} ({q:#x})") + (f" (overlap {lane})" if lane else "")
                tid_of[i] = tid
                names[(pid, tid)] = label
        gpu_pid = next(iter(agents.values()), 10000 + 100 * k)
        st = [s for s in a.step_list if w is None or (s["start"] >= w[0] and s["start"] <= w[1])]
        step_lane = _lanes([(s["start"], s["end"], n) for n, s in enumerate(st)])
        for n, lane in step_lane.items():
            names[(gpu_pid, STEPS_TID + lane)] = "steps (GPU markers)" + (f" (overlap {lane})" if lane else "")
        for ag, pid in agents.items():
            ev.append({"ph": "M", "name": "process_name", "pid": pid,
                       "args": {"name": f"{p.label}: " + (f"GPU {ag}" if cuda else f"GPU agent {ag}")}})
            ev.append({"ph": "M", "name": "process_sort_index", "pid": pid, "args": {"sort_index": pid}})
        if not agents:
            ev.append({"ph": "M", "name": "process_name", "pid": gpu_pid, "args": {"name": f"{p.label}: GPU"}})
        for (pid, tid), label in sorted(names.items()):
            ev.append({"ph": "M", "name": "thread_name", "pid": pid, "tid": tid, "args": {"name": label}})
            ev.append({"ph": "M", "name": "thread_sort_index", "pid": pid, "tid": tid, "args": {"sort_index": tid}})
        lanes_added += sum(1 for n in names.values() if "(overlap" in n)

        for n, s in enumerate(st):
            args = {"step": s["step"], **s["fields"]}
            ev.append({"ph": "X", "cat": "step", "name": f"step {s['step']}", "pid": gpu_pid,
                       "tid": STEPS_TID + step_lane[n], "ts": us(s["start"]), "dur": us(s["end"]) - us(s["start"]),
                       "args": args})

        for i in idx.tolist():
            r = d[i]
            nm = a.kernel_name.get(int(r["kernel_id"]), str(int(r["kernel_id"])))
            if cuda:
                args = {"kernel": nm, "corr": int(r["corr"]), "stream": int(r["queue_id"]),
                        "grid": [int(r["gx"]), int(r["gy"]), int(r["gz"])], "block": [int(r["wx"]), int(r["wy"]), int(r["wz"])]}
                if p.graph_node[i]:
                    args["graph_node_id"] = int(p.graph_node[i])
            else:
                args = {"kernel": nm, "corr": int(r["corr"]), "dispatch_id": int(r["dispatch_id"]),
                        "queue": hex(int(r["queue_id"])), "grid": [int(r["gx"]), int(r["gy"]), int(r["gz"])],
                        "workgroup": [int(r["wx"]), int(r["wy"]), int(r["wz"])]}
            if a.is_marker[i]:
                cat, label = "marker", "step marker"
            else:
                cat, label = "kernel", short_name(nm)
                if is_collective(nm):
                    args["ruppell_limits"] = ["collective-wait"]
                sc = a.scope_info(int(a.scope[i])) if a.scope[i] > 0 else None
                if a.nvtx[i] >= 0:  # NVTX (ROCTX) ranges, outermost first; they sit outside the scope
                    args["nvtx"] = a.range_path(int(a.nvtx[i]))
                if a.coll[i] > 0:  # the collective call that launched it (M8), beside its scope
                    args["collective"] = a.coll_info(int(a.coll[i]))
                nc = a.nccl_info(i)  # NCCL's own record of it (nccl.tsv, M8)
                if nc:
                    args["nccl"] = nc
                if sc:
                    args["scope"] = sc
                    label = f"{sc['name']} · {label}"
                elif a.nvtx[i] >= 0:
                    label = f"{args['nvtx'][-1]} · {label}"
                args["attribution"] = HOW_NAMES[int(a.how[i])]
                if a.graph[i] >= 0:
                    args["graph"] = {"key": a.graphs[int(a.graph[i])]["key"], "replay": int(a.replay[i]), "node": int(a.node[i])}
                if a.cpu_op[i] >= 0:
                    oi = a.op_info(int(a.cpu_op[i]))
                    args["op"] = {"name": oi["name"], "dims": oi["dims"], "types": oi["types"]}
                if a.step[i] >= 0:
                    args["step"] = int(a.step[i])
                    if a.step_rule[i] == 2:
                        args["step_by"] = "launching step (side stream)"
                        if a.outside_owner[i]:
                            args["runs_outside_its_step"] = True
            ev.append({"ph": "X", "cat": cat, "name": label, "pid": agents[int(r["agent_id"])], "tid": tid_of[i],
                       "ts": us(r["start"]), "dur": us(r["end"]) - us(r["start"]), "args": args})

        # memory copies and sets (copies.bin): a "copies" track on the GPU each touches
        # (a ROCm copy names agents; a device-to-host one's destination is the host)
        c = p.copies
        ci = np.nonzero(keep_mask(a, w, c["start"]))[0] if len(c) else np.zeros(0, np.int64)
        by_pid: Dict[int, List[Tuple[int, int, int]]] = {}
        for j in ci.tolist():
            pid = next((agents[dv] for dv in (int(c["dst_device"][j]), int(c["src_device"][j])) if dv in agents), gpu_pid)
            by_pid.setdefault(pid, []).append((int(c["start"][j]), int(c["end"][j]), j))
        for pid, items in by_pid.items():
            lanes = _lanes(items)
            for lane in sorted(set(lanes.values())):
                ev.append({"ph": "M", "name": "thread_name", "pid": pid, "tid": COPIES_TID + lane,
                           "args": {"name": "copies" + (f" (overlap {lane})" if lane else "")}})
                ev.append({"ph": "M", "name": "thread_sort_index", "pid": pid, "tid": COPIES_TID + lane,
                           "args": {"sort_index": COPIES_TID + lane}})
            for j, lane in lanes.items():
                x = c[j]
                args = {"bytes": int(x["bytes"]), "corr": int(x["corr"]), "attribution": HOW_NAMES[int(a.copy_how[j])]}
                if cuda:
                    args["stream"] = int(x["stream_id"])
                    if x["graph_node_id"]:
                        args["graph_node_id"] = int(x["graph_node_id"])
                if x["kind"] == COPY_MEMSET:
                    args["value"] = int(x["value"])
                label = a.copy_names[j]
                sc = a.scope_info(int(a.copy_scope[j])) if a.copy_scope[j] > 0 else None
                if sc:
                    args["scope"] = sc
                    label = f"{sc['name']} · {label}"
                if a.copy_step[j] >= 0:
                    args["step"] = int(a.copy_step[j])
                ev.append({"ph": "X", "cat": "memcpy" if x["kind"] != COPY_MEMSET else "memset", "name": label,
                           "pid": pid, "tid": COPIES_TID + lane, "ts": us(x["start"]),
                           "dur": us(x["end"]) - us(x["start"]), "args": args})

        am = keep_mask(a, w, p.api["start"]) if len(p.api) else np.zeros(0, bool)
        api_by_corr = {}
        for j in np.nonzero(am)[0].tolist():
            x = p.api[j]
            op = p.api_ops.get(int(x["op"]), f"{'cuda' if cuda else 'hip'}_op_{int(x['op'])}")
            ev.append({"ph": "X", "cat": "cuda_api" if cuda else "hip_api", "name": op, "pid": host_pid, "tid": int(x["thread_id"]),
                       "ts": us(x["start"]), "dur": us(x["end"]) - us(x["start"]), "args": {"corr": int(x["corr"])}})
            api_by_corr[int(x["corr"])] = x
        # the CPU op tree (RUPPELL_CPU_OPS): aten ops and ruppell scopes, nested
        # slices on the host threads, around the API calls they made
        if len(p.ops):
            # aten ops and scopes only: a sequence-number record (RUPPELL_OP_SEQ)
            # holds the number in t1 and is no interval (job 57972 exported
            # them as scopes of -33,811 s)
            om = keep_mask(a, w, p.ops["t0"]) & np.isin(p.ops["kind"], (OP_ATEN, OP_SCOPE))
            for j in np.nonzero(om)[0].tolist():
                r = p.ops[j]
                if int(r["kind"]) == OP_ATEN:
                    sig = p.op_sigs.get(int(r["id"]), {})
                    name, cat, oargs = sig.get("name", "op"), "cpu_op", {"dims": sig.get("dims"), "types": sig.get("types")}
                else:
                    name, cat, oargs = a.scope_name(int(r["id"])) or str(int(r["id"])), "scope", {}
                ev.append({"ph": "X", "cat": cat, "name": name, "pid": host_pid, "tid": int(r["thread_id"]),
                           "ts": us(r["t0"]), "dur": us(r["t1"]) - us(r["t0"]), "args": oargs})
        # NVTX (ROCTX) ranges: push ranges as slices on a track of their own per thread (they
        # need not nest with the op tree's slices), start/end ranges as async slices, marks
        # as instants on that track
        if len(p.ranges):
            rr = p.ranges
            nvtx_tids = set()
            for j in np.nonzero(keep_mask(a, w, rr["t0"]))[0].tolist():
                x = rr[j]
                kind, name = int(x["kind"]), a.range_label(j)
                rargs = {"domain": p.range_names.get(int(x["name"]), ("?",))[0]}
                if 1 <= int(x["payload_type"]) <= 6:  # NVTX's scalar payloads; the bits as given
                    rargs["payload"] = {"type": int(x["payload_type"]), "bits": int(x["payload"])}
                elif x["payload_type"]:  # an extension payload (NCCL's): a pointer, gone after the call
                    rargs["payload"] = {"type": int(x["payload_type"]), "extension": True}
                if x["category"]:
                    rargs["category"] = int(x["category"])
                tid = NVTX_TID + int(x["thread_id"])
                if kind == RANGE_PUSH:
                    nvtx_tids.add(int(x["thread_id"]))
                    ev.append({"ph": "X", "cat": "nvtx", "name": name, "pid": host_pid, "tid": tid,
                               "ts": us(x["t0"]), "dur": us(x["t1"]) - us(x["t0"]), "args": rargs})
                elif kind == RANGE_START:
                    rargs["end_thread"] = int(x["end_thread_id"])
                    for ph, t in (("b", x["t0"]), ("e", x["t1"])):
                        ev.append({"ph": ph, "cat": "nvtx", "name": name, "id": f"{k}.{j}", "pid": host_pid,
                                   "tid": int(x["thread_id"]), "ts": us(t), "args": rargs if ph == "b" else {}})
                elif kind == RANGE_MARK:
                    nvtx_tids.add(int(x["thread_id"]))
                    ev.append({"ph": "i", "s": "t", "cat": "nvtx", "name": name, "pid": host_pid, "tid": tid,
                               "ts": us(x["t0"]), "args": rargs})
            for t in sorted(nvtx_tids):
                ev.append({"ph": "M", "name": "thread_name", "pid": host_pid, "tid": NVTX_TID + t,
                           "args": {"name": f"thread {t} nvtx"}})
        for tid in sorted({int(t) for t in p.api["thread_id"][am]}):
            ev.append({"ph": "M", "name": "thread_name", "pid": host_pid, "tid": tid, "args": {"name": f"thread {tid}"}})

        if flows:
            for i in idx.tolist():
                if a.is_marker[i]:
                    continue
                x = api_by_corr.get(int(d["corr"][i]))
                if x is None:
                    continue
                fid = k * 1_000_000_000_000 + i
                # start 1 ns into the launching call (a replay's first kernels can
                # begin before hipGraphLaunch returns); end at the kernel's midpoint
                ev.append({"ph": "s", "cat": "launch", "name": "launch", "id": fid, "pid": host_pid,
                           "tid": int(x["thread_id"]), "ts": us(int(x["start"]) + min(1, int(x["end"] - x["start"])))})
                ev.append({"ph": "f", "bp": "e", "cat": "launch", "name": "launch", "id": fid,
                           "pid": agents[int(d["agent_id"][i])], "tid": tid_of[i],
                           "ts": us(int(d["start"][i]) + int(d["end"][i] - d["start"][i]) // 2)})
        reports.append(a.report)

    offsets = _clock_offsets(analyses)
    spread = (max(offsets.values()) - min(offsets.values())) if len(offsets) > 1 else 0.0
    report = {
        "processes": reports,
        "limitations": for_run(analyses[0].proc.meta if analyses else None,
                               cpu_ops=any(len(a.proc.ops) for a in analyses),
                               copies=all(a.proc.copies_recorded for a in analyses)),
        "merge": {**res["merge"], "clock_offset_spread_ns": spread},
        "schema": validate_run(analyses),
        "window": {"steps": list(steps) if steps else None},
        "flows": flows,
        "tracks": {"lanes_added_for_overlap": lanes_added},
    }
    return {"traceEvents": ev, "displayTimeUnit": "ns", "otherData": {"ruppell": report}}


def export(run_dir: str, out_path: str, flows: bool = True, steps: Optional[Tuple[int, int]] = None) -> dict:
    trace = build(run_dir, flows=flows, steps=steps)
    with open(out_path, "w") as f:
        json.dump(trace, f, separators=(",", ":"))
    return trace["otherData"]["ruppell"]
