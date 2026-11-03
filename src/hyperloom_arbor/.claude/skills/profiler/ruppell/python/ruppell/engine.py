"""The analysis engine: joins one process's records, vectorized with numpy.

M1's rules, at serving size (tens of millions of dispatches per process):

- graph-replayed dispatches join to the launches recorded while their graph
  was captured, by position in the replay (ordered by dispatch id), only if
  every replay of the graph has as many dispatches as the capture had
  operations and runs the same kernel sequence; otherwise the graph is
  refused, with the reason;
- eager dispatches take the scope pushed on the launching thread;
- steps are intervals between marker kernels on the GPU timeline.

Added for M2 (docs/plan-m2.md 9.2): when a process recorded the CPU window of
its steps, only kernels on the queues its markers ran on (the compute
stream) are placed by GPU time; kernels on other queues belong to the step
whose CPU window launched them, and those running outside that step on the
GPU are counted.

On CUDA (record version 2, docs/plan-cuda.md), CUPTI names the graph node
behind every replayed kernel, so replays join by node id instead of by
position: the instantiated node maps back through the clone records to the
node created during capture, which carries the scope that was on top when it
was created. Queues are CUDA streams and agents are devices.
"""

from __future__ import annotations

import glob
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional

import numpy as np

RECORD_VERSION = 1            # ROCm
RECORD_VERSION_CUDA = 2
MARKER_BIT = np.uint64(1 << 63)
MARKER_KERNEL = "ruppell_step_marker"
CAP_BEGIN, CAP_END, CAP_INSTANTIATE, CAP_LAUNCH, CAP_OP, CAP_KEY = 1, 2, 3, 4, 5, 6
CAP_NODE, CAP_CLONE, CAP_NODE_UPDATE = 7, 8, 9  # CUDA
CU_NODE_KERNEL, CU_NODE_MEMCPY, CU_NODE_MEMSET = 0, 1, 2

DISPATCH_DT = np.dtype([
    ("corr", "<u8"), ("ext", "<u8"), ("start", "<u8"), ("end", "<u8"), ("kernel_id", "<u8"),
    ("dispatch_id", "<u8"), ("queue_id", "<u8"), ("agent_id", "<u8"), ("thread_id", "<u8"),
    ("gx", "<u4"), ("gy", "<u4"), ("gz", "<u4"), ("wx", "<u4"), ("wy", "<u4"), ("wz", "<u4"),
])
API_DT = np.dtype([("corr", "<u8"), ("ext", "<u8"), ("start", "<u8"), ("end", "<u8"),
                   ("thread_id", "<u8"), ("kind", "<u4"), ("op", "<u4")])
CAP_DT = np.dtype([("kind", "<u8"), ("thread_id", "<u8"), ("ts", "<u8"),
                   ("a", "<u8"), ("b", "<u8"), ("c", "<u8"), ("d", "<u8")])
SELF_DT = np.dtype([("start", "<u8"), ("end", "<u8"), ("records", "<u8")])
# CUDA (runtime/src/records.h: ruppell_kernel_rec, ruppell_ext_rec)
KERNEL_DT_CUDA = np.dtype([
    ("corr", "<u8"), ("ext", "<u8"), ("start", "<u8"), ("end", "<u8"), ("kernel_id", "<u8"),
    ("graph_node_id", "<u8"), ("stream_id", "<u8"), ("device_id", "<u8"), ("graph_id", "<u8"),
    ("gx", "<u4"), ("gy", "<u4"), ("gz", "<u4"), ("wx", "<u4"), ("wy", "<u4"), ("wz", "<u4"),
])
EXT_DT = np.dtype([("corr", "<u8"), ("ext", "<u8")])
# both platforms (records.h: ruppell_copy_rec): copies.bin, memory copies and (CUDA) sets
COPY_DT = np.dtype([
    ("corr", "<u8"), ("ext", "<u8"), ("start", "<u8"), ("end", "<u8"), ("bytes", "<u8"),
    ("stream_id", "<u8"), ("dst_device", "<u8"), ("src_device", "<u8"), ("graph_node_id", "<u8"),
    ("thread_id", "<u8"), ("graph_id", "<u4"), ("kind", "<u4"), ("direction", "<u4"), ("value", "<u4"),
])
COPY_MEMCPY, COPY_MEMSET = 1, 2
# gray-parrot, both platforms (records.h: ruppell_counter_rec, ruppell_device_counter_rec):
# counters.bin and device_counters.bin; the contract is docs/gray-parrot/records.md
COUNTER_DT = np.dtype([
    ("key0", "<u8"), ("key1", "<u8"), ("seq", "<u8"), ("agent", "<u8"),
    ("step", "<u4"), ("group", "<u2"), ("counter", "<u2"), ("value", "<f8"),
])
DEVICE_COUNTER_DT = np.dtype([
    ("t0", "<u8"), ("t1", "<u8"), ("agent", "<u8"),
    ("step", "<u4"), ("group", "<u2"), ("counter", "<u2"), ("value", "<f8"),
])
# a copy's direction from the platform's own code: CUpti_ActivityMemcpyKind (CUDA),
# rocprofiler_memory_copy_operation_t (ROCm)
COPY_DIRECTION = {
    "cuda": {1: "HtoD", 2: "DtoH", 3: "HtoA", 4: "AtoH", 5: "AtoA", 6: "AtoD", 7: "DtoA", 8: "DtoD", 9: "HtoH",
             10: "PtoP"},
    "rocm": {1: "HtoH", 2: "HtoD", 3: "DtoH", 4: "DtoD"},
}


def copy_name(platform: str, kind: int, direction: int) -> str:
    """A copy's name in every output: "Memcpy HtoD", "Memset"."""
    if int(kind) == COPY_MEMSET:
        return "Memset"
    return "Memcpy " + COPY_DIRECTION.get(platform, {}).get(int(direction), "?")


def _by_corr(keys: np.ndarray, corr: np.ndarray, val: np.ndarray) -> np.ndarray:
    """For each key, the value of the row with that correlation id (0 when none)."""
    if not len(keys) or not len(corr):
        return np.zeros(len(keys), dtype=np.uint64)
    o = np.argsort(corr, kind="stable")
    c, v = corr[o], val[o]
    pos = np.clip(np.searchsorted(c, keys), 0, len(c) - 1)
    return np.where(c[pos] == keys, v[pos], 0).astype(np.uint64)


# CPU op tree (records.h: ruppell_op_rec); kind 1 aten op, 2 ruppell scope, 3 an op's
# autograd sequence number (t1; id the forward op's scope, depth 1 for a backward function)
OP_DT = np.dtype([("t0", "<u8"), ("t1", "<u8"), ("thread_id", "<u8"), ("id", "<u8"), ("kind", "<u4"), ("depth", "<u4")])
OP_ATEN, OP_SCOPE, OP_SEQ = 1, 2, 3
EVAL_PREFIX = "autograd::engine::evaluate_function: "  # autograd's wrapper around each backward node
# scope ids made by the analysis for backward kernels: "bwd:<forward scope>" (BWD_SCOPE_BASE
# + the forward scope's id) and "bwd:op:<backward function>" for a forward op outside any scope
BWD_SCOPE_BASE = 1 << 48
BWD_OP_SCOPE_BASE = 1 << 49
# NVTX and ROCTX ranges (records.h: ruppell_range_rec; roadmap-v2 M7): kind 1 push to pop on one
# thread, 2 start to end, 3 a mark; names in ranges.tsv (id, domain, text)
RANGE_DT = np.dtype([("t0", "<u8"), ("t1", "<u8"), ("thread_id", "<u8"), ("end_thread_id", "<u8"), ("name", "<u8"),
                     ("payload", "<u8"), ("kind", "<u4"), ("depth", "<u4"), ("payload_type", "<u4"), ("category", "<u4")])
RANGE_PUSH, RANGE_START, RANGE_MARK = 1, 2, 3
RANGE_KIND = {RANGE_PUSH: "push", RANGE_START: "start", RANGE_MARK: "mark"}
assert (DISPATCH_DT.itemsize, API_DT.itemsize, CAP_DT.itemsize, SELF_DT.itemsize) == (96, 48, 56, 24)
assert (KERNEL_DT_CUDA.itemsize, EXT_DT.itemsize, COPY_DT.itemsize, RANGE_DT.itemsize) == (96, 16, 96, 64)

HOW_NONE, HOW_EAGER, HOW_GRAPH, HOW_GRAPH_NOSCOPE, HOW_BACKWARD = 0, 1, 2, 3, 4
HOW_NAMES = {HOW_NONE: "none", HOW_EAGER: "eager", HOW_GRAPH: "graph", HOW_GRAPH_NOSCOPE: "graph:no-scope",
             HOW_BACKWARD: "backward"}


def _bin(path: str, dt: np.dtype, truncated: Optional[Dict[str, int]] = None) -> np.ndarray:
    """Records of one file. A process killed mid-write leaves a partial record
    at the end: it is dropped, and its size noted in ``truncated``."""
    if not os.path.exists(path):
        return np.zeros(0, dtype=dt)
    size = os.path.getsize(path)
    if size % dt.itemsize and truncated is not None:
        truncated[os.path.basename(path)] = size % dt.itemsize
    return np.fromfile(path, dtype=dt, count=size // dt.itemsize)


def _tsv(path: str) -> List[List[str]]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", errors="replace") as f:
        return [line.rstrip("\n").split("\t") for line in f if line.strip()]


# ns per tick of NCCL's GPU timer when the run cannot measure it (_nccl_tick): NCCL's is the GPU's
# globaltimer, in ns; RCCL's (ROCm 7.2) is wall_clock64, 100 MHz on gfx9
NCCL_TICK_NS = {"cuda": 1, "rocm": 10}


def _nccl_tick(tasks: List[dict], platform: str) -> dict:
    """ns per tick of the GPU timer in NCCL's channel times, measured against the tasks' host times
    (CLOCK_REALTIME, when the plugin saw the task): the median of (host, GPU) differences between
    the first timed task and each one at least 1 s later. Taken when within 1% of a whole number,
    the platform's otherwise (too short a run, or a timer that is neither)."""
    pts = sorted((k["t"], k["gpu_start"]) for k in tasks if k["gpu_start"])
    r = None
    if len(pts) > 1:
        h, g = np.array(pts, np.float64).T
        far = (h - h[0] >= 1e9) & (g > g[0])
        if far.any():
            r = float(np.median((h[far] - h[0]) / (g[far] - g[0])))
    n = round(r) if r else 0
    if n >= 1 and abs(r - n) <= 0.01 * n:
        return {"ns_per_tick": n, "measured_ns_per_tick": r, "source": "measured"}
    return {"ns_per_tick": NCCL_TICK_NS.get(platform, 1), "measured_ns_per_tick": r, "source": "platform"}


def _read_nccl(path: str, platform: str = "cuda") -> Optional[dict]:
    """nccl.tsv, ruppell's NCCL profiler plugin's lines (runtime/src/nccl_profiler.cpp; roadmap-v2
    M8): communicators, API calls, and tasks (collectives and p2p) in issue order, each with its
    kernel channels' GPU start and stop in ns of the GPU's own timer (NCCL's globaltimer; RCCL's
    wall_clock64 ticks, scaled by _nccl_tick): from the task's chs line (its channels' first start
    and last stop), and from ch lines (one per channel: the plugin before 2026-10-21, and a
    channel whose task's slot was reused). None when the run has none. A process killed
    mid-write can leave a partial last line: lines that do not parse are skipped. A drop line
    counts the lines the plugin could not keep.
    A collective API call (interface v5) that made no task is a launch of its own, from the call:
    function, count, datatype and root, with no algorithm, channels or GPU span. NCCL 2.29 gives
    broadcasts no task (they run from its own queue) but reports each call (DDP at 8 ranks on
    B200, 2026-10-21: 32 broadcasts per rank)."""
    if not os.path.exists(path):
        return None
    comms: Dict[int, dict] = {}
    apis: Dict[int, dict] = {}
    tasks: List[dict] = []
    chans: Dict[int, list] = defaultdict(list)
    spans: Dict[int, tuple] = {}
    bad = dropped = 0
    for x in _tsv(path):
        try:
            if x[0] == "comm":
                comms[int(x[1])] = {"name": x[2], "nodes": int(x[3]), "ranks": int(x[4]), "rank": int(x[5]),
                                    "interface": x[7] if len(x) > 7 else None}
            elif x[0] == "api":
                apis[int(x[2])] = {"kind": x[3], "func": x[4], "count": int(x[5]), "datatype": x[6],
                                   "graph_captured": x[9] == "1", "comm": int(x[1]), "root": int(x[7]),
                                   "t": int(x[10])}
            elif x[0] == "coll":
                tasks.append({"kind": "coll", "comm": int(x[1]), "id": int(x[2]), "api": int(x[3]), "seq": int(x[4]),
                              "func": x[5], "count": int(x[6]), "datatype": x[7], "root": int(x[8]),
                              "channels": int(x[9]), "warps": int(x[10]), "algo": x[11], "proto": x[12], "t": int(x[13])})
            elif x[0] == "p2p":
                tasks.append({"kind": "p2p", "comm": int(x[1]), "id": int(x[2]), "api": int(x[3]), "seq": None,
                              "func": x[4], "count": int(x[5]), "datatype": x[6], "peer": int(x[7]),
                              "channels": int(x[8]), "warps": None, "algo": None, "proto": None, "t": int(x[9])})
            elif x[0] == "chs":
                spans[int(x[2])] = (int(x[3]), int(x[4]), int(x[5]))
            elif x[0] == "ch":
                chans[int(x[2])].append((int(x[3]), int(x[4]), int(x[5])))
            elif x[0] == "drop":
                dropped += int(x[1])
        except (IndexError, ValueError):
            bad += 1
    tasks.sort(key=lambda k: k["id"])
    for k in tasks:
        ch = [c[1:] for c in chans.get(k["id"], []) if c[2]]
        n, sp = len(ch), spans.get(k["id"])
        if sp and sp[0]:
            ch.append(sp[1:])
            n += sp[0]
        k["gpu_start"] = min(c[0] for c in ch) if ch else None
        k["gpu_stop"] = max(c[1] for c in ch) if ch else None
        k["channels_timed"] = n
        k["graph_captured"] = apis.get(k["api"], {}).get("graph_captured", False)
    timer = _nccl_tick(tasks, platform)
    if timer["ns_per_tick"] != 1:
        for k in tasks:
            if k["gpu_start"]:
                k["gpu_start"] *= timer["ns_per_tick"]
                k["gpu_stop"] *= timer["ns_per_tick"]
    made = {k["api"] for k in tasks}
    untasked = [{"kind": "api", "comm": a["comm"], "id": i, "api": i, "seq": None, "func": a["func"],
                 "count": a["count"], "datatype": a["datatype"], "root": a["root"], "channels": None, "warps": None,
                 "algo": None, "proto": None, "t": a["t"], "gpu_start": None, "gpu_stop": None, "channels_timed": 0,
                 "graph_captured": a["graph_captured"]}
                for i, a in apis.items() if a["kind"] == "coll" and i not in made]
    launches = sorted(_launches(tasks) + untasked, key=lambda k: k["id"])
    return {"comms": comms, "apis": apis, "tasks": launches, "task_lines": len(tasks),
            "api_only": len(untasked), "unparsed_lines": bad, "dropped_lines": dropped, "gpu_timer": timer}


def _launches(tasks: List[dict]) -> List[dict]:
    """The plugin's tasks as kernel launches, in issue order. The tasks one API call (interface v5)
    or one group (v4) made run in one kernel, so they are one launch: RCCL at 8 ranks (MI350X,
    2026-10-21) runs an all-gather as 16 p2p tasks, a send and a receive per peer. A launch of one
    task is that task. Of several: its coll task if it has one, otherwise "SendRecv" with a send's
    count (each send carries the call's whole input); the GPU span over all of them, and
    ``parts``. Tasks with no call or group (api 0: v4 files before 2026-10-21) are a launch each."""
    groups: Dict[tuple, List[dict]] = {}
    for k in tasks:
        groups.setdefault((k["comm"], k["api"]) if k["api"] else ("task", k["id"]), []).append(k)
    out = []
    for g in groups.values():
        if len(g) == 1:
            out.append(g[0])
            continue
        colls = [k for k in g if k["kind"] == "coll"]
        if colls:
            base = dict(colls[0])
        else:
            send = next((k for k in g if k["func"] == "Send"), g[0])
            base = {**send, "func": "SendRecv" if any(k["func"] == "Recv" for k in g) else send["func"], "peer": None,
                    "channels": max(k["channels"] for k in g)}
        starts = [k["gpu_start"] for k in g if k["gpu_start"]]
        stops = [k["gpu_stop"] for k in g if k["gpu_stop"]]
        base.update(gpu_start=min(starts) if starts else None, gpu_stop=max(stops) if stops else None,
                    channels_timed=sum(k["channels_timed"] for k in g), parts=len(g),
                    graph_captured=any(k["graph_captured"] for k in g))
        out.append(base)
    out.sort(key=lambda k: k["id"])
    return out


NCCL_DTYPE = {"ncclInt8": ("int8", 1), "ncclUint8": ("uint8", 1), "ncclInt32": ("int32", 4), "ncclUint32": ("uint32", 4),
              "ncclInt64": ("int64", 8), "ncclUint64": ("uint64", 8), "ncclFloat16": ("float16", 2),
              "ncclFloat32": ("float32", 4), "ncclFloat64": ("float64", 8), "ncclBfloat16": ("bfloat16", 2),
              "ncclFloat8e4m3": ("float8_e4m3fn", 1), "ncclFloat8e5m2": ("float8_e5m2", 1)}
NCCL_OP = {"AllReduce": "all_reduce", "AllGather": "all_gather", "ReduceScatter": "reduce_scatter",
           "Broadcast": "broadcast", "Reduce": "reduce", "AllToAll": "all_to_all", "Send": "send", "Recv": "recv"}


def _innermost(t0: np.ndarray, t1: np.ndarray, pts: np.ndarray):
    """Intervals [t0, t1] on one thread and points in time on it: the innermost interval
    open at each point, and each interval's enclosing interval, as positions (-1: none). A
    sweep in time: at equal times an interval starts before a point, a point comes before
    an interval ends, and a longer interval starts first."""
    ni, nk = len(t0), len(pts)
    t = np.concatenate([t0, pts, t1])
    typ = np.concatenate([np.zeros(ni, np.int8), np.ones(nk, np.int8), np.full(ni, 2, np.int8)])
    ref = np.concatenate([np.arange(ni), np.arange(nk), np.arange(ni)])
    tie = np.concatenate([-(t1 - t0).astype(np.int64), np.zeros(nk + ni, np.int64)])
    owner = np.full(nk, -1, np.int64)
    parent = np.full(ni, -1, np.int64)
    stack: List[int] = []
    for j in np.lexsort((tie, typ, t)).tolist():
        kind, x = int(typ[j]), int(ref[j])
        if kind == 0:
            parent[x] = stack[-1] if stack else -1
            stack.append(x)
        elif kind == 1:
            if stack:
                owner[x] = stack[-1]
        elif stack and stack[-1] == x:
            stack.pop()
        elif x in stack:
            stack.remove(x)
    return owner, parent


def process_dirs(path: str) -> List[str]:
    """The process directories of a run: `path` itself (M1 layout) or its
    subdirectories that hold records (M2 layout, one per process)."""
    if os.path.exists(os.path.join(path, "dispatch.bin")) or os.path.exists(os.path.join(path, "meta.json")):
        return [path]
    subs = sorted(d for d in glob.glob(os.path.join(path, "*"))
                  if os.path.isdir(d) and os.path.exists(os.path.join(d, "dispatch.bin")))
    return subs


class Proc:
    """Everything one traced process wrote."""

    def __init__(self, path: str):
        self.path = path
        self.name = os.path.basename(path.rstrip("/"))
        self.meta = {}
        mp = os.path.join(path, "meta.json")
        if os.path.exists(mp):
            try:
                with open(mp) as f:
                    self.meta = json.load(f)
            except ValueError:
                # killed while rewriting it (tools before the atomic rename).
                # Only the CUDA tool writes ext.bin, so the layout is known;
                # otherwise refuse rather than read records with a guessed layout.
                if not os.path.exists(os.path.join(path, "ext.bin")):
                    raise ValueError(f"{path}: meta.json unreadable and the platform unknown; refusing")
                self.meta = {"record_version": RECORD_VERSION_CUDA, "platform": "cuda", "final": False,
                             "meta_unreadable": True}
        self.platform = self.meta.get("platform", "rocm") if self.meta else "rocm"
        want = RECORD_VERSION_CUDA if self.platform == "cuda" else RECORD_VERSION
        if self.meta and self.meta.get("record_version") != want:
            raise ValueError(f"{path}: record version {self.meta.get('record_version')} != {want} "
                             f"for platform {self.platform}; refusing")
        self.truncated: Dict[str, int] = {}
        self.graph_node: Optional[np.ndarray] = None   # CUDA: instantiated graph node per dispatch, 0 = eager
        self.api = _bin(os.path.join(path, "api.bin"), API_DT, self.truncated)
        if self.platform == "cuda":
            self.d = self._cuda_dispatches(path)
        else:
            self.d = _bin(os.path.join(path, "dispatch.bin"), DISPATCH_DT, self.truncated)
        self.cap = _bin(os.path.join(path, "capture.bin"), CAP_DT, self.truncated)
        # copies and sets: a run without copies.bin predates them (or ran with RUPPELL_COPIES=0)
        cp = os.path.join(path, "copies.bin")
        self.copies_recorded = os.path.exists(cp)
        self.copies = _bin(cp, COPY_DT, self.truncated)
        if self.platform == "cuda" and len(self.copies):
            # as for kernels: the external id from ext.bin, the issuing thread from the API record
            ext = _bin(os.path.join(path, "ext.bin"), EXT_DT, None)
            self.copies["ext"] = _by_corr(self.copies["corr"], ext["corr"], ext["ext"])
            self.copies["thread_id"] = _by_corr(self.copies["corr"], self.api["corr"], self.api["thread_id"])
        self.ops = _bin(os.path.join(path, "ops.bin"), OP_DT, self.truncated)
        self.op_sigs: Dict[int, dict] = {}
        for r in _tsv(os.path.join(path, "ops.tsv")):
            # a process killed mid-write can leave a partial line: skip what does not parse
            if len(r) > 1 and r[0].isdigit():
                try:
                    self.op_sigs[int(r[0])] = json.loads(r[1])
                except ValueError:
                    self.op_sigs[int(r[0])] = {"name": r[1]}
        # NVTX and ROCTX ranges: a run without ranges.bin predates them or ran without RUPPELL_NVTX
        rp = os.path.join(path, "ranges.bin")
        self.ranges_recorded = os.path.exists(rp)
        self.ranges = _bin(rp, RANGE_DT, self.truncated)
        self.range_names: Dict[int, tuple] = {}  # id -> (domain, text)
        for r in _tsv(os.path.join(path, "ranges.tsv")):
            if len(r) >= 2 and r[0].isdigit():
                self.range_names[int(r[0])] = (r[1], r[2] if len(r) > 2 else "")
        # an adapter's collective calls recorded as ranges (calls.h; RUPPELL_COLLECTIVE_MARK=range
        # or both): name is the call's kind
        self.calls = _bin(os.path.join(path, "calls.bin"), RANGE_DT, self.truncated)
        self.nccl = _read_nccl(os.path.join(path, "nccl.tsv"), self.platform)
        self.self_ = _bin(os.path.join(path, "self.bin"), SELF_DT, self.truncated)
        self.kernels: Dict[int, str] = {int(r[0]): (r[1] if len(r) > 1 else "") for r in _tsv(os.path.join(path, "kernels.tsv"))}
        self.api_ops: Dict[int, str] = {int(r[0]): r[1] for r in _tsv(os.path.join(path, "api_ops.tsv")) if len(r) > 1}
        self.defs: Dict[int, tuple] = {}
        self.context: Dict[str, object] = {}
        for r in _tsv(os.path.join(path, "defs.tsv")):
            if len(r) < 4:
                continue
            try:
                attrs = json.loads(r[3])
            except ValueError:
                attrs = {"raw": r[3]}
            if r[1] == "context":
                self.context[r[2]] = attrs.get("value", attrs)
            else:
                self.defs[int(r[0])] = (r[1], r[2], attrs)
        self.step_fields: Dict[int, dict] = defaultdict(dict)
        self.step_cpu: Dict[int, list] = defaultdict(lambda: [None, None])
        for r in _tsv(os.path.join(path, "steps.tsv")):
            if len(r) < 3:
                continue
            try:
                step, ts, obj = int(r[0]), int(r[1]), json.loads(r[2])
            except ValueError:
                continue
            if "_cpu" in obj and len(obj) == 1:
                self.step_cpu[step][0 if obj["_cpu"] == 1 else 1] = ts
            else:
                self.step_fields[step].update(obj)
        self.clock = [tuple(int(x) for x in r) for r in _tsv(os.path.join(path, "clock.tsv")) if len(r) == 4]
        run_json = os.path.join(os.path.dirname(path.rstrip("/")), "run.json")
        if not os.path.exists(run_json):
            run_json = os.path.join(path, "run.json")
        self.run_info = json.load(open(run_json)) if os.path.exists(run_json) else {}

    @property
    def records_complete(self) -> bool:
        """Whether the records end at a flush after the process's last work: the
        profiler's final flush (meta.json `final`), or a flush its adapter
        marked complete at shutdown (context `records_complete`). A process
        killed between the periodic flushes is neither: what it recorded after
        its last flush is gone, and `dropped_records` does not count it."""
        finalized = self.meta.get("final", True) if self.meta else False
        return bool(finalized) or bool(self.context.get("records_complete"))

    def _cuda_dispatches(self, path: str) -> np.ndarray:
        """CUDA kernel records in the common layout: streams as queues,
        devices as agents, the external id joined from ext.bin by correlation
        id, the launching thread from the API record, and dispatch ids by
        start time (CUPTI has no enqueue sequence; nothing on CUDA orders by
        them). The graph node of each kernel is kept in ``graph_node``."""
        k = _bin(os.path.join(path, "dispatch.bin"), KERNEL_DT_CUDA, self.truncated)
        ext = _bin(os.path.join(path, "ext.bin"), EXT_DT, self.truncated)
        d = np.zeros(len(k), dtype=DISPATCH_DT)
        for f in ("corr", "start", "end", "kernel_id", "gx", "gy", "gz", "wx", "wy", "wz"):
            d[f] = k[f]
        d["queue_id"] = k["stream_id"]
        d["agent_id"] = k["device_id"]
        order = np.lexsort((k["corr"], k["start"]))
        d["dispatch_id"][order] = np.arange(len(k), dtype=np.uint64)
        if len(ext) and len(k):
            eo = np.argsort(ext["corr"], kind="stable")
            ec, ev = ext["corr"][eo], ext["ext"][eo]
            pos = np.clip(np.searchsorted(ec, k["corr"], side="right") - 1, 0, len(ec) - 1)
            hit = ec[pos] == k["corr"]
            d["ext"] = np.where(hit, ev[pos], 0)
        if len(self.api) and len(k):
            ao = np.argsort(self.api["corr"], kind="stable")
            ac = self.api["corr"][ao]
            pos = np.clip(np.searchsorted(ac, k["corr"]), 0, len(ac) - 1)
            d["thread_id"] = np.where(ac[pos] == k["corr"], self.api["thread_id"][ao][pos], 0)
        self.graph_node = k["graph_node_id"].copy()
        return d

    @property
    def label(self) -> str:
        rank = self.context.get("rank")
        role = self.context.get("role")
        if rank is not None:
            return f"rank {rank}" + (f" ({role})" if role else "")
        return self.name


# ---- names ---------------------------------------------------------------------------------

def _base(name: str) -> str:
    return name[:-3] if name.endswith(".kd") else name


def demangle(names: List[str]) -> Dict[str, str]:
    """GNU c++filt first (rocprofv3's style), then ROCm's llvm-cxxfilt for names it
    leaves mangled; the ".kd" descriptor suffix is dropped first."""
    import shutil
    import subprocess

    names = sorted(set(names))
    result = dict(zip(names, [_base(n) for n in names]))
    rocm = os.environ.get("ROCM_PATH", "/opt/rocm")
    tools = []
    for cand in (shutil.which("c++filt"), os.path.join(rocm, "llvm", "bin", "llvm-cxxfilt"), shutil.which("llvm-cxxfilt")):
        if cand and os.path.exists(cand) and cand not in tools:
            tools.append(cand)
    for tool in tools:
        todo = [n for n in names if result[n].startswith("_Z")]
        if not todo:
            break
        out = subprocess.run([tool], input="\n".join(result[n] for n in todo), capture_output=True, text=True).stdout.splitlines()
        if len(out) == len(todo):
            result.update(zip(todo, out))
    return result


def short_name(demangled: str) -> str:
    """The function's qualified name with its template arguments: the
    parameter list and any return type dropped (a `(` inside template
    arguments, as in `std::enable_if<(N > 0), void>::type f<...>(...)`, is
    not the parameter list)."""
    s = demangled.strip()
    angle = 0
    cut = len(s)
    last_space = -1
    i = 0
    while i < len(s):
        ch = s[i]
        if s.startswith("(anonymous namespace)", i):
            i += len("(anonymous namespace)")
            continue
        if ch == "<":
            angle += 1
        elif ch == ">" and angle > 0:
            # `(4)>(0)` inside template arguments is a comparison, not a close
            if not (i + 1 < len(s) and s[i + 1] == "(" and i > 0 and s[i - 1] == ")"):
                angle -= 1
        elif angle == 0 and ch == "(":
            cut = i
            break
        elif angle == 0 and ch == " ":
            last_space = i
        i += 1
    name = s[last_space + 1:cut].strip() or s[:cut].strip()
    return name if len(name) <= 120 else name[:117] + "..."


# ---- analysis ----------------------------------------------------------------------------------

class Analysis:
    """Per-dispatch results (numpy arrays aligned with proc.d) and a report."""

    def __init__(self, proc: Proc):
        self.proc = proc
        p = proc
        d = p.d
        n = len(d)
        dem = demangle([p.kernels.get(int(k), f"kernel_{int(k)}") for k in np.unique(d["kernel_id"])]) if n else {}
        self.kernel_name: Dict[int, str] = {}
        for k in (np.unique(d["kernel_id"]) if n else []):
            raw = p.kernels.get(int(k), f"kernel_{int(k)}")
            self.kernel_name[int(k)] = dem.get(raw, raw)
        marker_kids = np.array([k for k, nm in self.kernel_name.items() if MARKER_KERNEL in nm], dtype=np.uint64)
        self.is_marker = np.isin(d["kernel_id"], marker_kids) if n else np.zeros(0, bool)
        # gray-parrot on CUDA: CUPTI's Range Profiler launches one WaitNs kernel per counter step,
        # outside every step (runner plan 04, probe A2). It is the profiler's, not the program's:
        # left out of the work, of coverage and of the outside-step counts, and counted on its own.
        self.is_profiler = np.zeros(n, bool)
        if n and p.platform == "cuda" and isinstance(p.meta.get("counters"), dict):
            prof_kids = np.array([k for k, nm in self.kernel_name.items() if nm == "WaitNs"], dtype=np.uint64)
            self.is_profiler = np.isin(d["kernel_id"], prof_kids)

        self.scope = np.zeros(n, dtype=np.int64)          # scope id, 0 = none
        self.how = np.zeros(n, dtype=np.int8)
        self.graph = np.full(n, -1, dtype=np.int32)       # index into self.graphs
        self.replay = np.full(n, -1, dtype=np.int64)
        self.node = np.full(n, -1, dtype=np.int32)
        self.step = np.full(n, -1, dtype=np.int64)        # step id
        self.step_rule = np.zeros(n, dtype=np.int8)        # 1 gpu time, 2 cpu owner
        self.outside_owner = np.zeros(n, dtype=bool)
        # the CPU time and thread a kernel was launched at: its launching call, or for a
        # replayed kernel the launch recorded while its graph was captured (CPU op tree)
        self.launch_ts = np.zeros(n, dtype=np.uint64)
        self.launch_tid = np.zeros(n, dtype=np.uint64)
        self.graphs: List[dict] = []
        self._did_order = None          # dispatch indices by dispatch id (built on first repair)
        self._did_sorted = None
        # collective calls (roadmap-v2 M8): an adapter pushes one around each communication call,
        # either a "collective" definition with its parent scope in it, or (the tool pairing a kind
        # with its scope, runner ruppell-accuracy plan 02, B1) a "collective_at" definition
        # {collective: the kind, parent: the scope}. Its kernels get the call's kind in ``coll``
        # (0: none) and keep the scope the call was made from (the parent) as their scope.
        self.coll = np.zeros(n, dtype=np.int64)
        self._scope_known = np.array(sorted(k for k, v in p.defs.items() if v[0] == "scope"), dtype=np.int64)
        coll_defs = sorted([(k, k, int(v[2].get("parent") or 0)) for k, v in p.defs.items() if v[0] == "collective"]
                           + [(k, int(v[2].get("collective") or 0), int(v[2].get("parent") or 0))
                              for k, v in p.defs.items() if v[0] == "collective_at"])
        self._coll_keys = np.array([k for k, _, _ in coll_defs], dtype=np.int64)
        self._coll_kinds = np.array([c for _, c, _ in coll_defs], dtype=np.int64)
        self._coll_parents = np.array([v for _, _, v in coll_defs], dtype=np.int64)
        self._coll_defined = sum(1 for v in p.defs.values() if v[0] == "collective")

        if p.platform == "cuda":
            self._join_graphs_by_node()
        else:
            self._join_graphs()
        self._eager()
        self._eager_launch_points()
        self._cpu_ops()
        self._ranges()
        self._calls()
        self._backward_scopes()
        self._steps()
        self._copies()
        self._counter_steps()
        self._nccl()
        self.report = self._report()

    # hardware counters (gray-parrot)
    def _counter_steps(self):
        """Counter steps: steps whose fields name the group counted (``counters``). The counting
        serialized their kernels, so every time the analysis reports (here, in the census, birdeye
        and the exports) leaves them out; their kernels are still counted, attributed and placed.
        ``untimed``: per dispatch, in a step left out of timing; ``copy_untimed`` the same for
        copies. The step after each counter step is flagged (``after_counter_step_ids``), and left
        out too with RUPPELL_UNTIMED_AFTER_COUNTER=1, if plan 03's B6 finds it disturbed."""
        d = self.proc.d
        order = sorted(self.step_list, key=lambda s: s["start"])
        cs = {s["step"] for s in order if "counters" in s["fields"]}
        self.counter_step_ids = cs
        self.after_counter_step_ids = {b["step"] for a, b in zip(order, order[1:])
                                       if a["step"] in cs and b["step"] not in cs}
        skip = set(cs)
        if os.environ.get("RUPPELL_UNTIMED_AFTER_COUNTER") == "1":
            skip |= self.after_counter_step_ids
        self.untimed_step_ids = skip
        ids = np.array(sorted(skip), dtype=np.int64)
        self.untimed = np.isin(self.step, ids) if len(ids) and len(d) else np.zeros(len(d), bool)
        cstep = getattr(self, "copy_step", np.zeros(0, np.int64))
        self.copy_untimed = np.isin(cstep, ids) if len(ids) and len(cstep) else np.zeros(len(cstep), bool)

    def _counter_steps_report(self, work: np.ndarray) -> Optional[dict]:
        if not self.counter_step_ids:
            return None
        return {"steps": len(self.counter_step_ids), "kernels": int((work & np.isin(
                    self.step, np.array(sorted(self.counter_step_ids), np.int64))).sum()),
                "after_steps": len(self.after_counter_step_ids),
                "left_out_of_timing": sorted(["counter"] + (["after"] if self.untimed_step_ids - self.counter_step_ids else [])),
                "untimed_kernels": int((work & self.untimed).sum())}

    def _ext_to_scope(self, ext) -> tuple:
        """External ids to (scope ids, collective ids). A scope id stays; a collective call's id
        gives the scope it was called from (its definition's parent) and its kind (itself, or a
        "collective_at" definition's collective); markers and ids that name neither give 0."""
        ext = np.asarray(ext, dtype=np.uint64)
        e = np.where((ext != 0) & ((ext & MARKER_BIT) == 0), ext, 0).astype(np.int64)
        coll = np.zeros(len(e), np.int64)
        if len(self._coll_keys):
            pos = np.clip(np.searchsorted(self._coll_keys, e), 0, len(self._coll_keys) - 1)
            hit = self._coll_keys[pos] == e
            coll[hit] = self._coll_kinds[pos][hit]
            e = np.where(hit, self._coll_parents[pos], e)
        return np.where(np.isin(e, self._scope_known), e, 0), coll

    # graphs
    def _join_graphs(self):
        p, d = self.proc, self.proc.d
        # Handles are reused: PyTorch destroys each hipGraph_t right after
        # instantiating it, so every capture can get the same graph address, and
        # exec handles freed with one capture pass come back in the next. So
        # handles are resolved in record order, to the capture current at the
        # time: a capture gets an id when it ends, an instantiate binds its exec
        # to the graph's current capture, a launch to its exec's current one.
        pending_key: Dict[int, int] = {}
        open_cap: Dict[int, dict] = {}
        caps: List[dict] = []
        graph_cur: Dict[int, int] = {}
        exec_cur: Dict[int, int] = {}
        launch_corr, launch_cap = [], []
        for c in p.cap.tolist():
            kind, tid, ts, a, b, cc, dd = c
            if kind == CAP_KEY:
                pending_key[tid] = a
            elif kind == CAP_BEGIN:
                open_cap[tid] = {"stream": a, "key_id": b or pending_key.pop(tid, 0), "ops": [], "ts": ts, "graph": 0}
            elif kind == CAP_OP and tid in open_cap:
                open_cap[tid]["ops"].append((a, b, cc, dd, ts, tid))
            elif kind == CAP_END:
                cap = open_cap.pop(tid, None)
                if cap is not None and cc == 1 and b:
                    cap["graph"] = b
                    graph_cur[b] = len(caps)
                    caps.append(cap)
            elif kind == CAP_INSTANTIATE:
                exec_cur[a] = graph_cur.get(b, -1)
            elif kind == CAP_LAUNCH:
                launch_corr.append(cc)
                launch_cap.append(exec_cur.get(a, -1))
        self.captures = len(caps)
        self.launch_corr = np.array(launch_corr, dtype=np.uint64)
        self.n_launches = len(launch_corr)
        if not len(d) or not launch_corr:
            return
        lc = self.launch_corr
        order = np.argsort(lc)
        lc_sorted = lc[order]
        lcap = np.array(launch_cap, dtype=np.int64)[order]
        pos = np.searchsorted(lc_sorted, d["corr"])
        pos_c = np.minimum(pos, len(lc_sorted) - 1)
        replayed = lc_sorted[pos_c] == d["corr"]
        self.is_replayed = replayed
        idx = np.nonzero(replayed)[0]
        if not len(idx):
            return
        cap_of = lcap[pos_c[idx]]
        # order by (corr, dispatch id): position within each replay
        o = np.lexsort((d["dispatch_id"][idx], d["corr"][idx]))
        idx, cap_of = idx[o], cap_of[o]
        corr = d["corr"][idx]
        new_group = np.ones(len(idx), dtype=bool)
        new_group[1:] = corr[1:] != corr[:-1]
        group_start = np.nonzero(new_group)[0]
        group_id = np.cumsum(new_group) - 1
        position = np.arange(len(idx)) - group_start[group_id]
        group_size = np.diff(np.append(group_start, len(idx)))
        group_cap = cap_of[group_start]
        kid = d["kernel_id"][idx]

        # replays grouped by capture in one sort (thousands of breakable segments
        # times hundreds of thousands of replays is too many to scan per capture)
        by_cap = np.argsort(group_cap, kind="stable")
        caps_u, first, count = np.unique(group_cap[by_cap], return_index=True, return_counts=True)
        for ci, f0, nrep in zip(caps_u.tolist(), first.tolist(), count.tolist()):
            gi = len(self.graphs)
            cap = caps[ci] if ci >= 0 else None
            gsel = by_cap[f0:f0 + nrep]                       # groups (replays) of this capture, in order
            key_name = None
            if cap and cap["key_id"] and cap["key_id"] in p.defs:
                key_name = p.defs[cap["key_id"]][1]
            rep = {"graph": hex(int(cap["graph"])) if cap else None, "capture": int(ci), "key": key_name, "replays": int(len(gsel)),
                   "captured_kernel_launches": None, "captured_memory_ops": None, "joined": False, "reason": None}
            self.graphs.append(rep)
            sizes = group_size[gsel]
            rep["dispatches_per_replay"] = {str(k): int(v) for k, v in zip(*np.unique(sizes, return_counts=True))}
            if cap is None:
                rep["reason"] = "no capture record for this launch's graph (captured before tracing started, or not by stream capture)"
                continue
            ops = cap["ops"]
            n_kern = sum(1 for o_ in ops if o_[3] == 1)
            n_mem = sum(1 for o_ in ops if o_[3] == 2)
            rep["captured_kernel_launches"], rep["captured_memory_ops"] = n_kern, n_mem
            # Replay by replay (plan-m2 §5: every replay joined or refused with a
            # reason): a replay joins when it has as many dispatches as the
            # capture calls for and runs the graph's usual kernel sequence. One
            # odd replay among 17,503 once refused a whole graph (job 48834).
            if n_mem == 0:
                if not (sizes == n_kern).any():
                    rep["reason"] = f"no replay has as many dispatches as the capture's {n_kern} kernel launches"
                    continue
                per, use = n_kern, [o_ for o_ in ops if o_[3] == 1]
            else:
                # A memory operation in a graph may run as no blit kernel (on a
                # DMA engine), one, or several (job 49224: 1,575 dispatches for
                # 1,568 launches and 6 memory operations). Align the capture to
                # the graph's usual replay by kind, never by count alone: an
                # equal count can hide another arrangement.
                aligned = self._align_memory_ops(ops, sizes, gsel, group_start, idx, kid, p.api_ops)
                if aligned is None:
                    rep["reason"] = (f"no replay has as many dispatches as the capture ({n_kern} kernel launches, "
                                     f"{n_mem} memory operations), and no alignment of its memory operations "
                                     f"to blit kernels fits")
                    continue
                per, use = aligned
                rep["memory_ops_aligned_to_blit_kernels"] = True
            ok_count = sizes == per
            gsel_ok = gsel[ok_count]
            starts = group_start[gsel_ok]
            rows = starts[:, None] + np.arange(per)[None, :]   # (replays, per) indices into idx
            seq = kid[rows]
            uniq, inv, cnt = np.unique(seq, axis=0, return_inverse=True, return_counts=True)
            same = inv.ravel() == int(np.argmax(cnt))          # the graph's usual sequence
            refused = {"different dispatch count": int((~ok_count).sum()),
                       "different kernel sequence": int((~same).sum())}
            rep["replays_joined"] = int(same.sum())
            rep["replays_refused"] = sum(refused.values())
            rep["replays_refused_by_reason"] = {k: v for k, v in refused.items() if v}
            rep["replays_with_a_different_kernel_sequence"] = refused["different kernel sequence"]
            if rep["replays_refused"]:
                bad = np.concatenate([np.nonzero(~ok_count)[0], np.nonzero(ok_count)[0][~same]])[:5]
                rep["refused_replays_sample"] = [
                    {"replay": int(b), "of": int(len(gsel)), "dispatches": int(sizes[b]),
                     "start": int(d["start"][idx[group_start[gsel[b]]]])} for b in sorted(bad.tolist())]
            gsel_join, rows = gsel_ok[same], rows[same]
            rep["joined"] = True
            rep["join_mode"] = "position in replay, by dispatch id"
            ext_ops = np.array([o_[2] for o_ in use], dtype=np.uint64)
            ts_ops = np.array([o_[4] for o_ in use], dtype=np.uint64)
            tid_ops = np.array([o_[5] for o_ in use], dtype=np.uint64)
            def attribute(drows: np.ndarray, replay_no: np.ndarray) -> None:
                """drows: (replays, per) dispatch indices in graph order."""
                flat = drows.ravel()
                posflat = np.tile(np.arange(per), len(drows))
                scope_ids, coll_ids = self._ext_to_scope(ext_ops[posflat])
                self.scope[flat] = scope_ids
                self.coll[flat] = coll_ids
                self.how[flat] = np.where(scope_ids > 0, HOW_GRAPH, HOW_GRAPH_NOSCOPE)
                self.graph[flat] = gi
                self.replay[flat] = np.repeat(replay_no, per)
                self.node[flat] = posflat
                self.launch_ts[flat] = ts_ops[posflat]
                self.launch_tid[flat] = tid_ops[posflat]

            attribute(idx[rows], np.nonzero(ok_count)[0][same])

            # Repair: the SDK has tagged a replay's dispatches on a graph's
            # second queue with another call's correlation id (job 48834: 4 MoE
            # kernels on queue 0x3, once in 17,503 replays). A replay short by
            # k is completed only when its dispatch-id range has exactly k
            # gaps, dispatches with those ids exist, and filling them in gives
            # the graph's usual kernel sequence exactly.
            usual = uniq[int(np.argmax(cnt))]
            rejoined = []
            for b in np.nonzero(sizes < per)[0].tolist():
                s0 = int(group_start[gsel[b]])
                mine = idx[s0:s0 + int(sizes[b])]
                ids = d["dispatch_id"][mine].astype(np.int64)
                inner = np.setdiff1d(np.arange(ids.min(), ids.max() + 1), ids)
                short = per - int(sizes[b]) - len(inner)
                if short < 0:
                    continue
                if self._did_order is None:
                    self._did_order = np.argsort(d["dispatch_id"], kind="stable")
                    self._did_sorted = d["dispatch_id"][self._did_order].astype(np.int64)
                found = None
                # the missing ids lie in the replay's range, or just past its ends
                for after in range(short + 1):
                    gap = np.concatenate([np.arange(ids.min() - (short - after), ids.min()), inner,
                                          np.arange(ids.max() + 1, ids.max() + 1 + after)])
                    pos = np.searchsorted(self._did_sorted, gap)
                    if (pos >= len(self._did_sorted)).any() or (self._did_sorted[np.minimum(pos, len(self._did_sorted) - 1)] != gap).any():
                        continue
                    extra = self._did_order[pos]
                    if self.is_replayed[extra].any():
                        continue                               # already another replay's
                    both = np.concatenate([mine, extra])
                    both = both[np.argsort(d["dispatch_id"][both], kind="stable")]
                    if np.array_equal(d["kernel_id"][both], usual):
                        found = (extra, both)
                        break
                if found is None:
                    continue
                extra, both = found
                attribute(both[None, :], np.array([b]))
                self.is_replayed[extra] = True
                foreign = sorted({int(c) for c in d["corr"][extra]})
                ops = []
                for c in foreign:
                    j = np.nonzero(p.api["corr"] == c)[0]
                    ops.append(p.api_ops.get(int(p.api["op"][j[0]])) if len(j) else None)
                rejoined.append({"replay": int(b), "recorrelated_dispatches": int(len(extra)),
                                 "their_correlation_ids": foreign, "which_belong_to": ops,
                                 "queues": sorted({hex(int(q)) for q in d["queue_id"][extra]})})
            if rejoined:
                rep["replays_rejoined_after_recorrelation"] = rejoined
                rep["replays_joined"] += len(rejoined)
                n = rep["replays_refused_by_reason"].get("different dispatch count", 0) - len(rejoined)
                if n:
                    rep["replays_refused_by_reason"]["different dispatch count"] = n
                else:
                    rep["replays_refused_by_reason"].pop("different dispatch count", None)
                rep["replays_refused"] = sum(rep["replays_refused_by_reason"].values())

    BLIT = "__amd_rocclr_"

    def _align_memory_ops(self, ops, sizes, gsel, group_start, idx, kid, api_ops):
        """Map a graph's captured operations onto its usual replay, where a
        memory operation ran as zero or more blit kernels (the runtime's
        `__amd_rocclr_copyBuffer*` / `fillBuffer*`): each kernel launch takes
        the next non-blit dispatch, each memory operation the blit dispatches
        of its kind that follow. Returns (dispatches per replay, the op for
        each position) or None when the sequence does not fit."""
        vals, cnt = np.unique(sizes, return_counts=True)
        m = int(vals[np.argmax(cnt)])
        b = int(np.nonzero(sizes == m)[0][0])
        s0 = int(group_start[gsel[b]])
        names = [self.kernel_name.get(int(k), "") for k in kid[s0:s0 + m]]
        blit = [n.startswith(self.BLIT) for n in names]
        use, j = [], 0
        for o in ops:
            if o[3] == 1:                                   # a kernel launch
                if j >= m or blit[j]:
                    return None
                use.append(o)
                j += 1
            elif o[3] == 2:                                 # a memory operation
                opname = api_ops.get(int(o[0]), "")
                kind = "fill" if "Memset" in opname else "copy"
                while j < m and blit[j] and kind in names[j][len(self.BLIT):].lower():
                    use.append(o)
                    j += 1
        return (m, use) if j == m else None

    def _join_graphs_by_node(self):
        """CUDA: every replayed kernel names its instantiated graph node; the
        clone records map it to the node created during capture, which
        carries its scope. A kernel whose node was never recorded, or whose
        graph had node parameters changed after capture, is refused with the
        reason; nothing is placed by position."""
        p, d = self.proc, self.proc.d
        pending_key: Dict[int, int] = {}
        open_cap: Dict[int, dict] = {}
        caps: List[dict] = []
        node_ext: Dict[int, int] = {}
        node_type: Dict[int, int] = {}
        node_graph: Dict[int, int] = {}
        node_at: Dict[int, tuple] = {}
        node_cap: Dict[int, int] = {}
        clone_of: Dict[int, int] = {}
        updated: set = set()
        launch_corr: List[int] = []
        outer_launches = 0
        for kind, tid, ts, a, b, cc, dd in p.cap.tolist():
            if kind == CAP_KEY:
                pending_key[tid] = a
            elif kind == CAP_BEGIN:
                open_cap[tid] = {"key_id": b or pending_key.pop(tid, 0), "nodes": [], "graph_ids": set(), "ts": ts}
            elif kind == CAP_NODE:
                node_ext[a], node_type[a], node_graph[a] = b, cc, dd
                node_at[a] = (ts, tid)
                cap = open_cap.get(tid)
                if cap is not None:
                    cap["nodes"].append(a)
                    cap["graph_ids"].add(dd)
            elif kind == CAP_CLONE:
                clone_of[a] = b
            elif kind == CAP_END:
                cap = open_cap.pop(tid, None)
                if cap is not None and cc == 1:
                    ci = len(caps)
                    caps.append(cap)
                    for nid in cap["nodes"]:
                        node_cap[nid] = ci
            elif kind == CAP_NODE_UPDATE:
                updated.add(a)
            elif kind == CAP_LAUNCH:
                launch_corr.append(cc)
                outer_launches += int(dd <= 1)
        self._node_ext, self._clone_of = node_ext, clone_of  # for copies replayed from graphs (_copies)
        # nodes added to a capture's graph from another thread (relaxed capture modes)
        graph_cap = {g: ci for ci, cap in enumerate(caps) for g in cap["graph_ids"]}
        for nid, g in node_graph.items():
            if nid not in node_cap and nid not in clone_of and g in graph_cap:
                node_cap[nid] = graph_cap[g]
                caps[graph_cap[g]]["nodes"].append(nid)
        self.captures = len(caps)
        self.launch_corr = np.array(launch_corr, dtype=np.uint64)
        self.n_launches = outer_launches
        gn = p.graph_node if p.graph_node is not None else np.zeros(len(d), np.uint64)
        replayed = gn != 0
        self.is_replayed = replayed
        idx = np.nonzero(replayed)[0]
        if not len(idx):
            return

        def original(nid: int) -> int:
            seen = 0
            while nid in clone_of and seen < 16:
                nid, seen = clone_of[nid], seen + 1
            return nid

        # per distinct instantiated node: its capture, position among the
        # capture's kernel nodes, scope, and why not if it cannot join
        uniq, inv = np.unique(gn[idx], return_inverse=True)
        kpos = [{nid: i for i, nid in enumerate(n for n in cap["nodes"] if node_type.get(n) == CU_NODE_KERNEL)}
                for cap in caps]
        cap_updated = [any(n in updated for n in cap["nodes"]) for cap in caps]
        for nid in updated:
            o = original(nid)
            if o in node_cap:
                cap_updated[node_cap[o]] = True
        u_cap = np.full(len(uniq), -1, np.int64)
        u_pos = np.full(len(uniq), -1, np.int64)
        u_scope = np.zeros(len(uniq), np.int64)
        u_coll = np.zeros(len(uniq), np.int64)
        u_reason = [None] * len(uniq)
        u_ts = np.zeros(len(uniq), np.uint64)          # when and where the captured node was created
        u_tid = np.zeros(len(uniq), np.uint64)
        for j, nid in enumerate(uniq.tolist()):
            o = original(nid)
            ci = node_cap.get(o, -1)
            if ci < 0:
                u_reason[j] = ("node not recorded at capture (captured before tracing started, or built without stream capture)"
                               if o not in node_ext else "node created outside any stream capture")
                continue
            if cap_updated[ci] or nid in updated:
                u_reason[j] = "graph node parameters changed after capture; the registry holds the captured ones"
                continue
            u_cap[j] = ci
            u_ts[j], u_tid[j] = node_at.get(o, (0, 0))
            u_pos[j] = kpos[ci].get(o, -1)
            e = node_ext.get(o, 0)
            sc_, co_ = self._ext_to_scope([e])
            u_scope[j], u_coll[j] = int(sc_[0]), int(co_[0])
        k_cap = u_cap[inv]
        k_ok = np.array([r is None for r in u_reason], dtype=bool)[inv]
        k_reason = np.array([r or "" for r in u_reason], dtype=object)[inv]
        # a replay (one launch's correlation id) joins whole or not at all
        why_corr = {int(c): r for c, r in zip(d["corr"][idx][~k_ok].tolist(), k_reason[~k_ok].tolist())}
        bad_corr = np.array(sorted(why_corr), dtype=np.uint64)
        resolved_corr = np.unique(d["corr"][idx][k_ok])

        # one graph entry per capture, plus one per reason for kernels that cannot join
        groups: Dict[object, np.ndarray] = {}
        order = np.argsort(k_cap, kind="stable")
        cu, first, count = np.unique(k_cap[order], return_index=True, return_counts=True)
        for c, f0, n in zip(cu.tolist(), first.tolist(), count.tolist()):
            sel = order[f0:f0 + n]
            if c >= 0:
                groups[c] = sel
            else:
                for r in sorted(set(k_reason[sel].tolist())):
                    groups[("refused", r)] = sel[k_reason[sel] == r]
        for key, sel in groups.items():
            gi = len(self.graphs)
            rows = idx[sel]
            corr = d["corr"][rows]
            ucorr, rep_no, sizes = np.unique(corr, return_inverse=True, return_counts=True)
            if isinstance(key, tuple):
                # replays counted under their capture when any of their kernels resolved
                alone = int((~np.isin(ucorr, resolved_corr)).sum())
                self.graphs.append({"graph": None, "capture": None, "key": None, "replays": alone,
                                    "kernels": int(len(rows)),
                                    "dispatches_per_replay": {str(k): int(v) for k, v in zip(*np.unique(sizes, return_counts=True))},
                                    "captured_kernel_launches": None, "captured_memory_ops": None, "joined": False,
                                    "reason": key[1], "replays_joined": 0, "replays_refused": alone,
                                    "replays_refused_by_reason": {key[1]: alone} if alone else {},
                                    "graph_ids": sorted({int(x) for x in np.unique(gn[rows])})[:5]})
                continue
            cap = caps[key]
            key_name = p.defs[cap["key_id"]][1] if cap["key_id"] and cap["key_id"] in p.defs else None
            n_kern = sum(1 for n in cap["nodes"] if node_type.get(n) == CU_NODE_KERNEL)
            n_mem = sum(1 for n in cap["nodes"] if node_type.get(n) in (CU_NODE_MEMCPY, CU_NODE_MEMSET))
            n_other = len(cap["nodes"]) - n_kern - n_mem
            ok = k_ok[sel]
            # a replay joins when every kernel it ran resolves to a captured node
            rep_ok = ~np.isin(ucorr, bad_corr)
            dup = int(len(rows) - len(np.unique(np.stack([corr, gn[rows]]), axis=1)[0]))
            rep = {"graph": ",".join(str(g) for g in sorted(cap["graph_ids"])), "capture": int(key), "key": key_name,
                   "replays": int(len(ucorr)),
                   "dispatches_per_replay": {str(k): int(v) for k, v in zip(*np.unique(sizes, return_counts=True))},
                   "captured_kernel_launches": n_kern, "captured_memory_ops": n_mem, "captured_other_nodes": n_other,
                   "joined": bool(rep_ok.any()), "reason": None,
                   "join_mode": "graph node id (CUPTI), through the instantiation's clone records",
                   "replays_joined": int(rep_ok.sum()), "replays_refused": int((~rep_ok).sum()),
                   "replays_refused_by_reason": {}, "replays_with_a_different_dispatch_count": int((sizes != n_kern).sum()),
                   "kernels_on_one_node_twice_in_a_replay": dup}
            if rep["replays_refused"]:
                rep["replays_refused_by_reason"] = dict(Counter(why_corr[int(c)] for c in ucorr[~rep_ok].tolist()))
            if not rep["joined"]:
                rep["reason"] = next(iter(rep["replays_refused_by_reason"]), "no replay joined")
            self.graphs.append(rep)
            good = rows[ok & rep_ok[rep_no]]
            gsel = sel[ok & rep_ok[rep_no]]
            sc = u_scope[inv[gsel]]
            self.scope[good] = sc
            self.coll[good] = u_coll[inv[gsel]]
            self.how[good] = np.where(sc > 0, HOW_GRAPH, HOW_GRAPH_NOSCOPE)
            self.graph[good] = gi
            self.replay[good] = rep_no[ok & rep_ok[rep_no]]
            self.node[good] = u_pos[inv[gsel]]
            self.launch_ts[good] = u_ts[inv[gsel]]
            self.launch_tid[good] = u_tid[inv[gsel]]

    def _eager_launch_points(self):
        """An eager kernel's launch point: its launching call's start and thread (a replayed
        kernel has its launch during capture from the graph join). The CPU op tree and the
        NVTX ranges and the collective calls recorded as ranges are matched against these, so they
        are found only when one of them was recorded."""
        p, d = self.proc, self.proc.d
        n = len(d)
        if not n or not len(p.api) or not (len(p.ops) or len(p.ranges) or len(p.calls)):
            return
        replayed = getattr(self, "is_replayed", np.zeros(n, bool))
        eager = ~replayed & ~self.is_marker
        if eager.any():
            ao = np.argsort(p.api["corr"], kind="stable")
            ac = p.api["corr"][ao]
            pos = np.clip(np.searchsorted(ac, d["corr"]), 0, len(ac) - 1)
            hit = eager & (ac[pos] == d["corr"])
            self.launch_ts[hit] = p.api["start"][ao][pos][hit]
            self.launch_tid[hit] = p.api["thread_id"][ao][pos][hit]

    # CPU op tree
    def _cpu_ops(self):
        """RUPPELL_CPU_OPS runs: each kernel's aten op is the innermost op open
        on its launching thread at its launch point (the launching call's
        start; for a replayed kernel, its launch while the graph was captured).
        ``cpu_op``: index into proc.ops, -1 when none."""
        p, d = self.proc, self.proc.d
        n = len(d)
        self.cpu_op = np.full(n, -1, dtype=np.int64)
        ops = p.ops
        if not len(ops) or not n:
            return
        aten = np.nonzero(ops["kind"] == OP_ATEN)[0]
        want = np.nonzero(self.launch_ts > 0)[0]
        for tid in np.unique(self.launch_tid[want]).tolist():
            oi = aten[ops["thread_id"][aten] == tid]
            ki = want[self.launch_tid[want] == tid]
            if not len(oi):
                continue
            # a sweep in time: at equal times an op starts before a point, a
            # point comes before an op ends, and a longer op starts first
            t = np.concatenate([ops["t0"][oi], self.launch_ts[ki], ops["t1"][oi]])
            typ = np.concatenate([np.zeros(len(oi), np.int8), np.ones(len(ki), np.int8), np.full(len(oi), 2, np.int8)])
            ref = np.concatenate([oi, ki, oi])
            tie = np.concatenate([-(ops["t1"][oi] - ops["t0"][oi]).astype(np.int64), np.zeros(len(ki) + len(oi), np.int64)])
            stack: List[int] = []
            for j in np.lexsort((tie, typ, t)).tolist():
                kind, r = int(typ[j]), int(ref[j])
                if kind == 0:
                    stack.append(r)
                elif kind == 1:
                    if stack:
                        self.cpu_op[r] = stack[-1]
                elif stack and stack[-1] == r:
                    stack.pop()
                elif r in stack:
                    stack.remove(r)

    # NVTX and ROCTX ranges (roadmap-v2 M7)
    def _ranges(self):
        """Each kernel's NVTX (ROCTX) range: the innermost push range open on its launching
        thread at its launch point, as for the CPU op tree, so a replayed kernel's is the
        range open when its graph was captured. ``nvtx``: index into proc.ranges, -1 when
        none. ``range_parent``: each push range's enclosing push range on its thread (any
        domain), -1 at the top. ruppell's own scopes are not replaced: a range sits outside
        the scope (``scope``) a kernel already has. Start/end ranges and marks attribute no
        kernel."""
        p, d = self.proc, self.proc.d
        r = p.ranges
        self.nvtx = np.full(len(d), -1, dtype=np.int64)
        self.range_parent = np.full(len(r), -1, dtype=np.int64)
        if not len(r):
            return
        push = np.nonzero(r["kind"] == RANGE_PUSH)[0]
        want = np.nonzero(self.launch_ts > 0)[0]
        for tid in np.unique(r["thread_id"][push]).tolist():
            oi = push[r["thread_id"][push] == tid]
            ki = want[self.launch_tid[want] == tid]
            owner, parent = _innermost(r["t0"][oi], r["t1"][oi], self.launch_ts[ki])
            self.range_parent[oi] = np.where(parent >= 0, oi[np.maximum(parent, 0)], -1)
            hit = owner >= 0
            self.nvtx[ki[hit]] = oi[owner[hit]]

    def _calls(self):
        """Collective calls recorded as ranges (proc.calls; runner ruppell-accuracy plan 02, B3):
        a kernel launched inside one, on its thread at its launch point (eager, or when its graph
        was captured), gets the innermost one's kind in ``coll`` where no pushed id gave it a
        call. A run that recorded both (RUPPELL_COLLECTIVE_MARK=both) says how they agree, in
        ``_calls_check``: kernels with a call both ways, the same call or another, and with one
        way only."""
        p, d = self.proc, self.proc.d
        c = p.calls
        self._calls_check: Optional[dict] = None
        if not len(c) or not len(d):
            return
        by_range = np.zeros(len(d), np.int64)
        want = np.nonzero(self.launch_ts > 0)[0]
        for tid in np.unique(c["thread_id"]).tolist():
            ci = np.nonzero(c["thread_id"] == tid)[0]
            ki = want[self.launch_tid[want] == tid]
            owner, _ = _innermost(c["t0"][ci].astype(np.int64), c["t1"][ci].astype(np.int64),
                                  self.launch_ts[ki].astype(np.int64))
            hit = owner >= 0
            by_range[ki[hit]] = c["name"][ci[owner[hit]]].astype(np.int64)
        pushed, ranged = self.coll > 0, by_range > 0
        self._calls_check = {"calls": int(len(c)), "kernels_in_a_call": int(ranged.sum()),
                             "agree": int((pushed & ranged & (self.coll == by_range)).sum()),
                             "differ": int((pushed & ranged & (self.coll != by_range)).sum()),
                             "pushed_only": int((pushed & ~ranged).sum()), "range_only": int((ranged & ~pushed).sum())}
        self.coll = np.where(pushed, self.coll, by_range)

    def range_label(self, i: int) -> Optional[str]:
        """Range ``i`` of proc.ranges by name: its text, prefixed by its domain unless that is
        NVTX's unnamed one or ROCTX's."""
        if i < 0:
            return None
        dom, text = self.proc.range_names.get(int(self.proc.ranges["name"][i]), ("?", "?"))
        return text if dom in ("default", "roctx") else f"{dom}:{text}"

    def range_path(self, i: int) -> List[str]:
        """The push ranges around range ``i``, outermost first, ending with ``i``."""
        out: List[str] = []
        while i >= 0 and len(out) < 64:
            out.append(self.range_label(i))
            i = int(self.range_parent[i])
        return out[::-1]

    def _ranges_report(self, work: np.ndarray) -> Optional[dict]:
        """What the ranges cover. Per range name: instances, the kernels launched inside them
        (nested ranges included, as nsys's NVTX projection counts them) and their GPU time,
        and for kernels launched eagerly inside a range, its projection on the GPU (first
        kernel's start to last kernel's end, summed over instances), which is what nsys
        reports. Replayed kernels are counted apart: their range is the one open at capture,
        which nsys does not project onto replays."""
        p, d = self.proc, self.proc.d
        if not p.ranges_recorded:
            return None
        r = p.ranges
        dur = (d["end"] - d["start"]).astype(np.int64) if len(d) else np.zeros(0, np.int64)
        has = (self.nvtx >= 0) & work
        replayed = getattr(self, "is_replayed", np.zeros(len(d), bool))
        nr = len(r)
        acc = {k: np.zeros(nr, np.int64) for k in ("kernels", "gpu", "rkernels", "rgpu")}
        first = np.full(nr, np.iinfo(np.int64).max, np.int64)
        last = np.zeros(nr, np.int64)
        idx = np.nonzero(has)[0]
        cur = self.nvtx[idx].copy()
        for _ in range(64):  # up the enclosing ranges: a kernel counts in each
            live = cur >= 0
            if not live.any():
                break
            k, c = idx[live], cur[live]
            rep = replayed[k]
            np.add.at(acc["kernels"], c[~rep], 1)
            np.add.at(acc["gpu"], c[~rep], dur[k[~rep]])
            np.add.at(acc["rkernels"], c[rep], 1)
            np.add.at(acc["rgpu"], c[rep], dur[k[rep]])
            np.minimum.at(first, c[~rep], d["start"][k[~rep]].astype(np.int64))
            np.maximum.at(last, c[~rep], d["end"][k[~rep]].astype(np.int64))
            cur[live] = self.range_parent[c]
        proj = np.where(acc["kernels"] > 0, last - first, 0)
        by: Dict[int, dict] = {}
        for i in range(nr):
            nid = int(r["name"][i])
            e = by.setdefault(nid, {"instances": 0, "instances_with_kernels": 0, "kernels": 0, "gpu_time_ns": 0,
                                    "projected_ns": 0, "replayed_kernels": 0, "replayed_gpu_time_ns": 0,
                                    "kind": RANGE_KIND.get(int(r["kind"][i]), str(int(r["kind"][i])))})
            e["instances"] += 1
            e["instances_with_kernels"] += int(acc["kernels"][i] > 0 or acc["rkernels"][i] > 0)
            e["kernels"] += int(acc["kernels"][i])
            e["gpu_time_ns"] += int(acc["gpu"][i])
            e["projected_ns"] += int(proj[i])
            e["replayed_kernels"] += int(acc["rkernels"][i])
            e["replayed_gpu_time_ns"] += int(acc["rgpu"][i])
        rows = []
        for nid, e in by.items():
            dom, text = p.range_names.get(nid, ("?", "?"))
            rows.append({"domain": dom, "name": text, **e})
        rows.sort(key=lambda e: -(e["gpu_time_ns"] + e["replayed_gpu_time_ns"]))
        t_work = int(dur[work].sum()) if len(d) else 0
        t_in = int(dur[has].sum()) if len(d) else 0
        return {
            "source": (p.meta.get("nvtx") or {}).get("source"),
            "meta": p.meta.get("nvtx"),
            "records": int(nr),
            "by_kind": {RANGE_KIND.get(int(k), str(int(k))): int(v) for k, v in zip(*np.unique(r["kind"], return_counts=True))},
            "by_domain": dict(Counter(p.range_names.get(int(x), ("?",))[0] for x in r["name"].tolist())),
            "kernels_in_a_range": {"count": int(has.sum()), "gpu_time_ns": t_in,
                                   "share_of_work_gpu_time": (t_in / t_work) if t_work else None,
                                   "eager": int((has & ~replayed).sum()), "replayed": int((has & replayed).sum())},
            "kernels_with_a_range_and_no_scope": int((has & (self.scope <= 0)).sum()),
            "by_name": rows[:100],
        }

    # collective calls (roadmap-v2 M8)
    def coll_info(self, cid: int) -> Optional[dict]:
        """A collective call by id: its operation and attributes (backend, dtype, numel, bytes,
        world, rank), without its parent scope."""
        v = self.proc.defs.get(int(cid))
        if not v or v[0] != "collective":
            return None
        return {"id": int(cid), "op": v[1], **{k: x for k, x in v[2].items() if k != "parent"}}

    def _collectives_report(self, work: np.ndarray) -> Optional[dict]:
        """What the adapter's collective calls cover: the collective kernels (by name) with a
        call, eager and replayed, and per call kind (operation, backend, dtype, elements, group
        size) its kernels, their GPU time and the bytes each call moves. Kernels inside a call
        that are not collective kernels (copies into a buffer, a layout change after a gather)
        are counted apart. None when the run has no collective definition."""
        p, d = self.proc, self.proc.d
        if not len(self._coll_keys) or not len(d):
            return None
        from ruppell.limits import is_collective

        dur = (d["end"] - d["start"]).astype(np.int64)
        coll_kids = [k for k in np.unique(d["kernel_id"]).tolist() if is_collective(self.kernel_name.get(int(k), ""))]
        is_c = np.isin(d["kernel_id"], np.array(coll_kids, dtype=np.uint64)) & work
        rep = getattr(self, "is_replayed", np.zeros(len(d), bool))
        has = self.coll > 0
        # one pass: each kernel with a call to its call kind
        sel = np.nonzero(has & work)[0]
        ucid, inv = np.unique(self.coll[sel], return_inverse=True)
        kinds: Dict[tuple, int] = {}
        rows: List[dict] = []
        kidx = np.zeros(len(ucid), np.int64)
        for j, cid in enumerate(ucid.tolist()):
            ci = self.coll_info(cid) or {}
            key = (ci.get("op"), ci.get("backend"), ci.get("dtype"), ci.get("numel"), ci.get("world"))
            if key not in kinds:
                kinds[key] = len(rows)
                rows.append({"op": key[0], "backend": key[1], "dtype": key[2], "numel": key[3], "world": key[4],
                             "bytes": ci.get("bytes")})
            kidx[j] = kinds[key]
        g = kidx[inv]
        c_, r_, du = is_c[sel], rep[sel], dur[sel]
        nk = len(rows)
        counts = {"collective_kernels": np.bincount(g[c_], minlength=nk),
                  "replayed": np.bincount(g[c_ & r_], minlength=nk),
                  "gpu_time_ns": np.bincount(g[c_], weights=du[c_], minlength=nk),
                  "other_kernels": np.bincount(g[~c_], minlength=nk)}
        for i, r in enumerate(rows):
            r.update({k: int(v[i]) for k, v in counts.items()})
        out = sorted(rows, key=lambda r: -r["gpu_time_ns"])
        t_c = int(dur[is_c].sum())
        return {
            "calls_defined": int(self._coll_defined),
            "by_range": getattr(self, "_calls_check", None),
            "collective_kernels": int(is_c.sum()),
            "with_a_call": int((is_c & has).sum()),
            "share_with_a_call": float((is_c & has).sum() / is_c.sum()) if is_c.any() else None,
            "share_of_time_with_a_call": float(dur[is_c & has].sum() / t_c) if t_c else None,
            "eager": {"kernels": int((is_c & ~rep).sum()), "with_a_call": int((is_c & ~rep & has).sum())},
            "replayed": {"kernels": int((is_c & rep).sum()), "with_a_call": int((is_c & rep & has).sum())},
            "without_a_call": [{"kernel": n_, "count": c_} for n_, c_ in Counter(
                self.kernel_name.get(int(k), "?") for k in d["kernel_id"][is_c & ~has].tolist()).most_common(10)],
            "by_call": out[:100],
        }

    def _backward_scopes(self):
        """Training: a kernel launched inside a backward function (RecordFunction
        scope BACKWARD_FUNCTION) takes the scope of the forward op it
        differentiates. No module hook runs in the backward: the kernel carries
        no scope, or only the coarse one of the code that called backward()
        (DLRMv4 runs autograd's CUDA work on the calling thread, inside
        `pipeline._backward`, job 57923), so the backward function's scope
        replaces it. The backward function carries the forward op's autograd
        sequence number, and the forward op recorded the innermost scope open
        when it ran (RUPPELL_OP_SEQ records). Scope "bwd:<forward scope>", or
        "bwd:op:<backward function>" when the forward op ran outside any scope
        or there is none (AccumulateGrad); ``how`` is "backward".

        Autograd's engine wraps each node in "autograd::engine::evaluate_function:
        <node>": the node's own record, then its post-hooks (DDP's gradient
        all-reduce runs in AccumulateGrad's), then the accumulation of its
        outputs into the next nodes' input buffers. A kernel anywhere in the
        wrapper belongs to the node (job 57972: 364 ms of DDP all-reduce and 63
        ms of input-buffer adds, over 8 ranks and 5 steps, lay outside the
        node's own record)."""
        p, d = self.proc, self.proc.d
        ops = p.ops
        self.backward_link = None
        if not len(ops) or not len(d):
            return
        seqr = ops[ops["kind"] == OP_SEQ]
        bwd = seqr[seqr["depth"] == 1]
        if not len(bwd):
            return
        fwd = seqr[seqr["depth"] == 0]
        seq_scope: Dict[int, int] = {}
        for s, sid in zip(fwd["t1"].tolist(), fwd["id"].tolist()):
            if sid:
                seq_scope.setdefault(int(s), int(sid))
        # the backward functions' own intervals: their aten records, by thread and start
        bseq = {(int(t), int(t0)): int(s) for t, t0, s in zip(bwd["thread_id"], bwd["t0"], bwd["t1"])}
        aten = np.nonzero(ops["kind"] == OP_ATEN)[0]
        node = np.array([(int(ops["thread_id"][i]), int(ops["t0"][i])) in bseq for i in aten.tolist()], dtype=bool)
        wrap_sigs = [s for s, v in p.op_sigs.items() if str(v.get("name", "")).startswith(EVAL_PREFIX)]
        wrap = np.isin(ops["id"][aten], np.array(wrap_sigs, dtype=np.uint64)) & ~node
        if not node.any():
            return
        bi = aten[node | wrap]
        # each wrapper's node: the first node record starting inside it, on its thread
        wrap_seq: Dict[int, int] = {}
        for tid in np.unique(ops["thread_id"][aten[wrap]]).tolist():
            ni = aten[node][ops["thread_id"][aten[node]] == tid]
            ni = ni[np.argsort(ops["t0"][ni], kind="stable")]
            nt0 = ops["t0"][ni]
            for w in aten[wrap][ops["thread_id"][aten[wrap]] == tid].tolist():
                j = int(np.searchsorted(nt0, ops["t0"][w]))
                if j < len(ni) and nt0[j] <= ops["t1"][w]:
                    wrap_seq[w] = bseq[(int(tid), int(nt0[j]))]
        cand = np.nonzero((self.launch_ts > 0) & ~self.is_marker)[0]
        linked = by_op = 0
        for tid in np.unique(ops["thread_id"][bi]).tolist():
            oi = bi[ops["thread_id"][bi] == tid]
            ki = cand[self.launch_tid[cand] == tid]
            if not len(ki):
                continue
            t = np.concatenate([ops["t0"][oi], self.launch_ts[ki], ops["t1"][oi]])
            typ = np.concatenate([np.zeros(len(oi), np.int8), np.ones(len(ki), np.int8), np.full(len(oi), 2, np.int8)])
            ref = np.concatenate([oi, ki, oi])
            tie = np.concatenate([-(ops["t1"][oi] - ops["t0"][oi]).astype(np.int64), np.zeros(len(ki) + len(oi), np.int64)])
            stack: List[int] = []
            for j in np.lexsort((tie, typ, t)).tolist():
                kind, r = int(typ[j]), int(ref[j])
                if kind == 0:
                    stack.append(r)
                elif kind == 2:
                    if stack and stack[-1] == r:
                        stack.pop()
                    elif r in stack:
                        stack.remove(r)
                elif stack:
                    o = stack[-1]
                    seq = bseq.get((int(tid), int(ops["t0"][o])), wrap_seq.get(o))
                    sid = seq_scope.get(seq) if seq is not None else None
                    if sid is not None and sid in p.defs and p.defs[sid][0] == "scope":
                        bid = BWD_SCOPE_BASE + sid
                        if bid not in p.defs:
                            _, name, attrs = p.defs[sid]
                            p.defs[bid] = ("scope", f"bwd:{name}", {**attrs, "phase": "backward"})
                        linked += 1
                    else:
                        sig = int(ops["id"][o])
                        fn = p.op_sigs.get(sig, {}).get("name", f"op {sig}")
                        fn = fn.split(": ", 1)[-1]
                        bid = BWD_OP_SCOPE_BASE + sig
                        if bid not in p.defs:
                            p.defs[bid] = ("scope", f"bwd:op:{fn}", {"op": fn, "phase": "backward"})
                        by_op += 1
                    self.scope[r] = bid
                    self.how[r] = HOW_BACKWARD
        self.backward_link = {"backward_functions": int(node.sum()), "wrappers": int(wrap.sum()),
                              "forward_ops_with_scope": len(seq_scope),
                              "kernels_by_forward_scope": linked, "kernels_by_backward_op": by_op}

    def _cpu_ops_report(self, work: np.ndarray) -> dict:
        p, d = self.proc, self.proc.d
        ops = p.ops
        dur = (d["end"] - d["start"]).astype(np.int64)
        has = work & (self.cpu_op >= 0)
        rep = getattr(self, "is_replayed", np.zeros(len(d), bool))
        t_work = int(dur[work].sum())
        return {
            "aten_ops": int((ops["kind"] == OP_ATEN).sum()), "scope_intervals": int((ops["kind"] == OP_SCOPE).sum()),
            "signatures": len(p.op_sigs), "threads": int(len(np.unique(ops["thread_id"]))),
            # records of no known kind: the file was written out of record
            # alignment (a forked child writing into it, before the runtime's
            # child_after_fork), so everything after the first is suspect
            "malformed_records": int((~np.isin(ops["kind"], (OP_ATEN, OP_SCOPE, OP_SEQ))).sum()),
            "kernels_with_an_op": int(has.sum()), "of_kernels": int(work.sum()),
            "gpu_time_share_with_an_op": (int(dur[has].sum()) / t_work) if t_work else None,
            "replayed_kernels_with_an_op": int((has & rep).sum()), "of_replayed": int((work & rep).sum()),
        }

    def op_info(self, i: int) -> Optional[dict]:
        """The aten op record ``i`` of proc.ops: name, dims, types, times."""
        if i < 0:
            return None
        r = self.proc.ops[i]
        sig = self.proc.op_sigs.get(int(r["id"]), {})
        return {"name": sig.get("name", f"op {int(r['id'])}"), "dims": sig.get("dims"), "types": sig.get("types"),
                "t0": int(r["t0"]), "t1": int(r["t1"]), "thread": int(r["thread_id"])}

    # eager
    def _eager(self):
        p, d = self.proc, self.proc.d
        if not len(d):
            return
        replayed = getattr(self, "is_replayed", np.zeros(len(d), bool))
        cand = (~replayed) & (~self.is_marker)
        sc, co = self._ext_to_scope(d["ext"])
        ok = cand & (sc > 0)
        self.scope[ok] = sc[ok]
        self.coll[cand] = co[cand]
        self.how[ok] = HOW_EAGER

    # steps
    def _steps(self):
        p, d = self.proc, self.proc.d
        self.step_list: List[dict] = []
        if not len(d):
            return
        m = np.nonzero(self.is_marker)[0]
        code = d["ext"][m]
        coded = (code & MARKER_BIT) != 0
        self.markers_without_code = int((~coded).sum())
        m, code = m[coded], code[coded]
        stepid = ((code & ~MARKER_BIT) >> np.uint64(2)).astype(np.int64)
        kind = (code & np.uint64(3)).astype(np.int64)
        begins, ends = {}, {}
        for i, s, k in zip(m.tolist(), stepid.tolist(), kind.tolist()):
            (begins if k == 1 else ends)[s] = i
        self.compute_queues = sorted({int(q) for q in d["queue_id"][m]})
        unpaired = 0
        for s in sorted(set(begins) | set(ends)):
            b, e = begins.get(s), ends.get(s)
            if b is None or e is None:
                unpaired += 1
                continue
            cpu = p.step_cpu.get(s, [None, None])
            self.step_list.append({"step": s, "start": int(d["start"][b]), "end": int(d["end"][e]),
                                   "cpu_begin": cpu[0], "cpu_end": cpu[1], "fields": dict(p.step_fields.get(s, {}))})
        self.unpaired_markers = unpaired
        self.step_list.sort(key=lambda s: s["start"])
        if not self.step_list:
            return
        st = np.array([s["start"] for s in self.step_list], dtype=np.uint64)
        en = np.array([s["end"] for s in self.step_list], dtype=np.uint64)
        sid = np.array([s["step"] for s in self.step_list], dtype=np.int64)
        work = ~self.is_marker
        have_cpu = all(s["cpu_begin"] is not None and s["cpu_end"] is not None for s in self.step_list)
        self.cpu_rule = have_cpu
        on_compute = np.isin(d["queue_id"], np.array(self.compute_queues, dtype=np.uint64))
        gpu_placed = work & (on_compute | (not have_cpu))

        # GPU time: kernel inside [begin marker start, end marker end]; check the
        # step that starts last before it and the one before that (markers of
        # consecutive steps can overlap by a few hundred ns)
        k = np.searchsorted(st, d["start"], side="right") - 1
        inside = np.zeros(len(d), dtype=np.int8)
        chosen = np.full(len(d), -1, dtype=np.int64)
        for back in (0, 1):
            kk = k - back
            ok = kk >= 0
            kc = np.clip(kk, 0, None)
            hit = ok & (d["start"] >= st[kc]) & (d["end"] <= en[kc])
            chosen = np.where(hit & (chosen < 0), kc, chosen)
            inside += hit.astype(np.int8)
        kc = np.clip(k, 0, None)
        straddle = (k >= 0) & (inside == 0) & (d["start"] < en[kc])
        self.step[gpu_placed & (chosen >= 0)] = sid[chosen[gpu_placed & (chosen >= 0)]]
        self.step[gpu_placed & straddle] = sid[kc[gpu_placed & straddle]]
        self.step_rule[gpu_placed & ((chosen >= 0) | straddle)] = 1
        self._inside = inside
        self._straddle = straddle
        self._gpu_placed = gpu_placed

        # CPU ownership for the other queues (9.2)
        side = work & ~gpu_placed
        self._side = side
        if side.any():
            a = p.api
            if len(a):
                ao = np.argsort(a["corr"])
                ac = a["corr"][ao]
                astart = a["start"][ao]
                pos = np.clip(np.searchsorted(ac, d["corr"][side]), 0, len(ac) - 1)
                found = ac[pos] == d["corr"][side]
                launch_ts = np.where(found, astart[pos], 0).astype(np.uint64)
            else:
                found = np.zeros(int(side.sum()), bool)
                launch_ts = np.zeros(int(side.sum()), np.uint64)
            cb = np.array([s["cpu_begin"] for s in self.step_list], dtype=np.uint64)
            ce = np.array([s["cpu_end"] for s in self.step_list], dtype=np.uint64)
            co = np.argsort(cb)
            cbs, ces, sids, sts, ens = cb[co], ce[co], sid[co], st[co], en[co]
            j = np.searchsorted(cbs, launch_ts, side="right") - 1
            jc = np.clip(j, 0, None)
            owned = found & (j >= 0) & (launch_ts <= ces[jc])
            sidx = np.nonzero(side)[0]
            self.step[sidx[owned]] = sids[jc[owned]]
            self.step_rule[sidx[owned]] = 2
            out = owned & ((d["start"][side] < sts[jc]) | (d["end"][side] > ens[jc]))
            self.outside_owner[sidx[out]] = True

    # NCCL's own records of its calls (nccl.tsv; roadmap-v2 M8)
    @staticmethod
    def _nccl_name_agrees(func: str, kernel: str) -> Optional[bool]:
        """Whether an NCCL task's function is the kernel's (None: the kernel's name does not say,
        as RCCL's ncclDevKernel_Generic does not)."""
        if "Generic" in kernel:
            return None
        if func in ("Send", "Recv"):
            return "SendRecv" in kernel or func in kernel
        return func in kernel

    @staticmethod
    def _nccl_kernel_func(kernel: str) -> Optional[str]:
        """The function an NCCL kernel's name says (ncclDevKernel_AllReduce_Sum_bf16_RING_LL:
        AllReduce), keyed as _nccl_task_func keys a task's; None when it says none."""
        parts = kernel.split("(")[0].split("_")
        f = parts[1] if len(parts) > 1 and parts[0] in ("ncclDevKernel", "ncclKernel") else None
        return None if f in (None, "", "Generic") else f

    @staticmethod
    def _nccl_task_func(func: str) -> str:
        return "SendRecv" if func in ("Send", "Recv", "SendRecv") else func

    def _nccl(self):
        """Each eager NCCL kernel's launch in nccl.tsv, ruppell's NCCL profiler plugin (_launches:
        a task, or the tasks of one call or group). Both sides
        are in issue order: the plugin's launches by id, the kernels by their launching call's start
        (the kernel's own start when the call is not recorded), which holds across streams.
        Where kernels' names say their function, as NCCL's do, the k-th eager task of a function
        pairs with the k-th eager kernel of that function (``paired_by`` "function"), so a kernel
        the library gave no task leaves the others paired: NCCL gave DDP's broadcasts none (B200,
        8 ranks, 2026-10-21), and pairing all by order slipped a call at each. RCCL's kernels
        (ncclDevKernel_Generic) name no function, so there the k-th eager task pairs with the
        k-th eager NCCL kernel ("order"). Graph-captured tasks, which NCCL reports once at
        capture, are left out. The pairing is checked by name, and by NCCL's GPU timer against
        ruppell's kernel start: the timer has its own offset per GPU and drifts a little, so a
        line is fitted per GPU, and a pairing off by a call leaves residuals as large as the
        time between calls. With RCCL the residuals are the only check.
        ``nccl_task``: index into proc.nccl["tasks"], -1 when none."""
        p, d = self.proc, self.proc.d
        self.nccl_task = np.full(len(d), -1, np.int64)
        self._nccl_pairing: Optional[dict] = None
        if not p.nccl or not len(d):
            return
        tasks = p.nccl["tasks"]
        eager_t = [i for i, k in enumerate(tasks) if not k["graph_captured"]]
        rep = getattr(self, "is_replayed", np.zeros(len(d), bool))
        kids = np.array([k for k, nm in self.kernel_name.items() if nm.startswith(("ncclDevKernel", "ncclKernel"))],
                        dtype=np.uint64)
        cand = np.nonzero(np.isin(d["kernel_id"], kids) & ~rep & ~self.is_marker)[0]
        launch = _by_corr(d["corr"][cand], p.api["corr"], p.api["start"]) if len(p.api) else np.zeros(len(cand), np.uint64)
        cand = cand[np.argsort(np.where(launch > 0, launch, d["start"][cand]), kind="stable")].tolist()
        kf = [self._nccl_kernel_func(self.kernel_name.get(int(d["kernel_id"][i]), "")) for i in cand]
        if any(f is not None for f in kf):
            by_t: Dict[str, list] = defaultdict(list)
            by_k: Dict[Optional[str], list] = defaultdict(list)
            for ti in eager_t:
                by_t[self._nccl_task_func(tasks[ti]["func"])].append(ti)
            for i, f in zip(cand, kf):
                by_k[f].append(i)
            pairs = [pr for f, ks in by_k.items() if f is not None for pr in zip(ks, by_t.get(f, []))]
            used = {ti for _, ti in pairs}
            pairs += zip(by_k.get(None, []), [ti for ti in eager_t if ti not in used])  # names that say none
            paired_by = "function"
        else:
            pairs, paired_by = list(zip(cand, eager_t)), "order"
        n = len(pairs)
        for i, ti in pairs:
            self.nccl_task[i] = ti
        used_k, used_t = {i for i, _ in pairs}, {ti for _, ti in pairs}
        left_k = Counter(f or "?" for i, f in zip(cand, kf) if i not in used_k)
        left_t = Counter(tasks[ti]["func"] for ti in eager_t if ti not in used_t)
        agree = Counter()
        offsets: Dict[int, list] = defaultdict(list)
        for i, ti in pairs:
            k = tasks[ti]
            agree[self._nccl_name_agrees(k["func"], self.kernel_name.get(int(d["kernel_id"][i]), ""))] += 1
            if k["gpu_start"]:
                offsets[int(d["agent_id"][i])].append((k["gpu_start"], int(d["start"][i]) - k["gpu_start"]))
        off = {}
        for dev, gv in offsets.items():
            g, v = np.array(gv, np.int64).T
            o = {"median_ns": int(np.median(v)), "spread_p1_p99_ns": int(np.percentile(v, 99) - np.percentile(v, 1))}
            x, y = (g - g.min()).astype(np.float64), (v - o["median_ns"]).astype(np.float64)
            if len(v) > 2 and x.max() > 0:
                # a minority whose GPU start is not its kernel's must not tilt the line: RCCL at 8
                # ranks attaches other kernels' channel events to a task, one 10.8 s old in a 14 s
                # DDP run (Cluster A job 89500), which a fit over all of them followed. So the first fit
                # is on the launches near the median offset, the second on those near that line;
                # within_100us says how many are near it
                far = np.abs(y)
                keep = far <= max(5 * float(np.median(far)), 1_000_000.0)
                b, c = np.polyfit(x[keep], y[keep], 1) if keep.sum() > 2 else np.polyfit(x, y, 1)
                res = y - (b * x + c)
                far = np.abs(res - np.median(res))
                keep = far <= max(5 * float(np.median(far)), 1000.0)
                if keep.sum() > 2 and not keep.all():
                    b, c = np.polyfit(x[keep], y[keep], 1)
                    res = y - (b * x + c)
                o.update(drift_ppm=round(float(b) * 1e6, 3),
                         residual_p1_p99_ns=int(np.percentile(res, 99) - np.percentile(res, 1)),
                         residual_max_ns=int(np.abs(res).max()),
                         residual_p50_abs_ns=int(np.median(np.abs(res))),
                         within_100us=round(float((np.abs(res) <= 100_000).mean()), 4))
            off[str(dev)] = o
        self._nccl_pairing = {"eager_tasks": len(eager_t), "graph_captured_tasks": len(tasks) - len(eager_t),
                              "eager_nccl_kernels": int(len(cand)), "paired": int(n), "paired_by": paired_by,
                              "names_agree": agree[True], "names_differ": agree[False], "names_unknown": agree[None],
                              "unpaired_tasks": len(eager_t) - n, "unpaired_kernels": int(len(cand)) - n,
                              "unpaired_tasks_by_function": dict(left_t), "unpaired_kernels_by_function": dict(left_k),
                              "gpu_timer_offset": off}

    def nccl_info(self, i: int) -> Optional[dict]:
        """Kernel ``i``'s NCCL launch (_launches: one task, or the parts of one call or group): what
        NCCL chose for the call (algorithm, protocol, channels, warps), its sequence number, and
        its kernel channels' span on the GPU (ns)."""
        ti = int(self.nccl_task[i]) if hasattr(self, "nccl_task") else -1
        if ti < 0:
            return None
        k = self.proc.nccl["tasks"][ti]
        span = (k["gpu_stop"] - k["gpu_start"]) if k["gpu_start"] else None
        return {"func": k["func"], "seq": k["seq"], "count": k["count"], "datatype": k["datatype"], "algo": k["algo"],
                "proto": k["proto"], "channels": k["channels"], "warps": k["warps"], "gpu_span_ns": span,
                "comm": k["comm"], "parts": k.get("parts", 1)}

    def _nccl_bytes(self, k: dict) -> Optional[int]:
        """One rank's input bytes for NCCL task k, as the adapter records a call's. NCCL counts an
        all-gather or a broadcast in bytes (ncclInt8), and a reduce-scatter by its output."""
        size = NCCL_DTYPE.get(k["datatype"], (None, 0))[1]
        if not size:
            return None
        n = k["count"] * size
        if k["func"] == "ReduceScatter":
            n *= self.proc.nccl["comms"].get(k["comm"], {}).get("ranks", 1)
        return n

    def _nccl_report(self) -> Optional[dict]:
        """What nccl.tsv holds and how it paired with the NCCL kernels, and per call kind (function,
        datatype, count, algorithm, protocol, channels): the paired kernels, ruppell's GPU time for
        them, and NCCL's own span of their channels (first channel's start to last one's stop).
        ``call_bytes``: for kernels that also carry an adapter's call, the call's bytes against
        NCCL's, the check that the adapter recorded the input it was given."""
        p, d = self.proc, self.proc.d
        if not p.nccl:
            return None
        rows: Dict[tuple, dict] = {}
        sizes = Counter()
        differ: Dict[tuple, int] = Counter()
        dur = (d["end"] - d["start"]).astype(np.int64) if len(d) else np.zeros(0, np.int64)
        coll = getattr(self, "coll", None)
        for i in np.nonzero(self.nccl_task >= 0)[0].tolist():
            k = p.nccl["tasks"][int(self.nccl_task[i])]
            ci = self.coll_info(int(coll[i])) if coll is not None and coll[i] > 0 else None
            if ci and isinstance(ci.get("bytes"), int) and ci["bytes"] >= 0:
                nb = self._nccl_bytes(k)
                sizes[nb == ci["bytes"]] += 1
                if nb != ci["bytes"]:
                    differ[(ci["op"], ci.get("backend"), ci.get("method"), ci["bytes"], k["func"], nb)] += 1
            key = (k["func"], k["datatype"], k["count"], k["algo"], k["proto"], k["channels"])
            r = rows.setdefault(key, {"func": key[0], "datatype": key[1], "count": key[2], "algo": key[3],
                                      "proto": key[4], "channels": key[5], "kernels": 0, "gpu_time_ns": 0,
                                      "nccl_span_ns": 0, "timed": 0})
            r["kernels"] += 1
            if self.untimed[i]:
                continue   # in a counter step: counted, not timed
            r["gpu_time_ns"] += int(dur[i])
            if k["gpu_start"]:
                r["timed"] += 1
                r["nccl_span_ns"] += int(k["gpu_stop"] - k["gpu_start"])
        return {"comms": {str(h): c for h, c in p.nccl["comms"].items()}, "tasks": len(p.nccl["tasks"]),
                "task_lines": p.nccl.get("task_lines", len(p.nccl["tasks"])),
                "api_only": p.nccl.get("api_only", 0),
                "unparsed_lines": p.nccl["unparsed_lines"], "dropped_lines": p.nccl.get("dropped_lines", 0),
                "gpu_timer": p.nccl["gpu_timer"],
                **(self._nccl_pairing or {}),
                "call_bytes": {"agree": sizes[True], "differ": sizes[False],
                               "differing": [{"op": key[0], "backend": key[1], "method": key[2], "call_bytes": key[3],
                                              "nccl": key[4], "nccl_bytes": key[5], "kernels": n}
                                             for key, n in differ.most_common(10)]},
                "by_call": sorted(rows.values(), key=lambda r: -r["gpu_time_ns"])[:100]}

    # copies and sets (copies.bin)
    def _copies(self):
        """Memory copies and sets, attributed by the kernels' rules.
        - Scope: an eagerly issued one takes the scope pushed on its thread
          (external correlation). One replayed from a CUDA graph takes its
          captured node's scope. On ROCm one issued by a graph launch stays
          unscoped (graph:no-scope): the SDK names no node.
        - Step: by GPU time on the compute streams, where the markers run;
          otherwise by the CPU window of the call that issued it, as
          side-stream kernels are. ROCm's DMA copies have no stream, so they
          take the CPU rule."""
        p, c = self.proc, self.proc.copies
        n = len(c)
        self.copy_scope = np.zeros(n, dtype=np.int64)
        self.copy_how = np.zeros(n, dtype=np.int8)
        self.copy_step = np.full(n, -1, dtype=np.int64)
        self.copy_step_rule = np.zeros(n, dtype=np.int8)  # 1 gpu time, 2 cpu owner
        self.copy_names = [copy_name(p.platform, k, dr) for k, dr in zip(c["kind"].tolist(), c["direction"].tolist())]
        if not n:
            return
        # a copy made inside a collective call (M8) keeps the scope the call was made from
        sc, _ = self._ext_to_scope(c["ext"])
        graphed = c["graph_node_id"] != 0
        ok = (~graphed) & (sc > 0)
        self.copy_scope[ok] = sc[ok]
        self.copy_how[ok] = HOW_EAGER
        node_ext, clone_of = getattr(self, "_node_ext", {}), getattr(self, "_clone_of", {})
        for i in np.nonzero(graphed)[0].tolist():
            nid, seen = int(c["graph_node_id"][i]), 0
            while nid in clone_of and seen < 16:
                nid, seen = clone_of[nid], seen + 1
            e = int(self._ext_to_scope([int(node_ext.get(nid, 0))])[0][0])
            if e:
                self.copy_scope[i], self.copy_how[i] = e, HOW_GRAPH
            else:
                self.copy_how[i] = HOW_GRAPH_NOSCOPE
        launches = getattr(self, "launch_corr", np.zeros(0, np.uint64))
        if len(launches):
            from_launch = (self.copy_how == HOW_NONE) & np.isin(c["corr"], launches)
            self.copy_how[from_launch] = HOW_GRAPH_NOSCOPE

        if not self.step_list:
            return
        st = np.array([s["start"] for s in self.step_list], dtype=np.uint64)
        en = np.array([s["end"] for s in self.step_list], dtype=np.uint64)
        sid = np.array([s["step"] for s in self.step_list], dtype=np.int64)
        have_cpu = getattr(self, "cpu_rule", False)
        on_compute = np.isin(c["stream_id"], np.array(getattr(self, "compute_queues", []), dtype=np.uint64)) \
            if p.platform == "cuda" else np.zeros(n, bool)
        gpu = on_compute | (not have_cpu)
        k = np.searchsorted(st, c["start"], side="right") - 1
        kc = np.clip(k, 0, None)
        hit = gpu & (k >= 0) & (c["start"] < en[kc])
        self.copy_step[hit] = sid[kc[hit]]
        self.copy_step_rule[hit] = 1
        side = ~gpu
        if side.any() and len(p.api):
            launch_ts = _by_corr(c["corr"][side], p.api["corr"], p.api["start"])
            cb = np.array([s["cpu_begin"] for s in self.step_list], dtype=np.uint64)
            ce = np.array([s["cpu_end"] for s in self.step_list], dtype=np.uint64)
            co = np.argsort(cb)
            cbs, ces, sids = cb[co], ce[co], sid[co]
            j = np.searchsorted(cbs, launch_ts, side="right") - 1
            jc = np.clip(j, 0, None)
            owned = (launch_ts > 0) & (j >= 0) & (launch_ts <= ces[jc])
            sidx = np.nonzero(side)[0]
            self.copy_step[sidx[owned]] = sids[jc[owned]]
            self.copy_step_rule[sidx[owned]] = 2

    def _copies_report(self) -> Optional[dict]:
        p, c = self.proc, self.proc.copies
        if not p.copies_recorded:
            return None
        # copies in a counter step are counted, not timed
        dur = np.where(self.copy_untimed, 0, (c["end"] - c["start"]).astype(np.int64))
        by = defaultdict(lambda: [0, 0, 0])
        for nm, b, t in zip(self.copy_names, c["bytes"].tolist(), dur.tolist()):
            e = by[nm]
            e[0], e[1], e[2] = e[0] + 1, e[1] + int(b), e[2] + int(t)
        t_total = int(dur.sum())
        return {
            "count": int(len(c)),
            "gpu_time_ns": t_total,
            "attributed_share": (int(dur[self.copy_scope > 0].sum()) / t_total) if t_total else None,
            "attributed_by": dict(Counter(HOW_NAMES[int(h)] for h in self.copy_how)),
            "in_a_step": int((self.copy_step >= 0).sum()),
            "step_rule": {"gpu time": int((self.copy_step_rule == 1).sum()), "cpu owner": int((self.copy_step_rule == 2).sum())},
            "by_name": {k: {"count": v[0], "bytes": v[1], "gpu_time_ns": v[2]} for k, v in sorted(by.items())},
        }

    def _report(self) -> dict:
        p, d = self.proc, self.proc.d
        work = (~self.is_marker & ~self.is_profiler) if len(d) else np.zeros(0, bool)
        timed = work & ~self.untimed   # times leave counter steps out (_counter_steps)
        dur = (d["end"] - d["start"]).astype(np.int64) if len(d) else np.zeros(0, np.int64)
        t_total = int(dur[timed].sum()) if len(d) else 0
        t_attr = int(dur[timed & (self.scope > 0)].sum()) if len(d) else 0
        un = Counter()
        un_t = Counter()
        if len(d):
            sel = timed & (self.scope == 0)
            kids, inv = np.unique(d["kernel_id"][sel], return_inverse=True)
            counts = np.bincount(inv, minlength=len(kids))
            times = np.bincount(inv, weights=dur[sel], minlength=len(kids))
            for k, c, t in zip(kids.tolist(), counts.tolist(), times.tolist()):
                nm = short_name(self.kernel_name.get(int(k), str(k)))
                un[nm] += int(c)
                un_t[nm] += int(t)
        by = Counter(HOW_NAMES[int(h)] for h in self.how[work]) if len(d) else Counter()
        steps_rep = {"count": len(self.step_list)}
        if len(d) and self.step_list:
            gp = self._gpu_placed
            engine_start = min((s["start"] for s in self.step_list if s["fields"].get("phase") == "engine"), default=None)
            before = (d["start"] < engine_start) if engine_start is not None else np.zeros(len(d), bool)
            outside = gp & (self.step < 0) & ~self.is_profiler   # the profiler's WaitNs is not the program's
            ov = [a["end"] - b["start"] for a, b in zip(self.step_list, self.step_list[1:]) if b["start"] < a["end"]]
            steps_rep.update({
                "engine_steps": sum(1 for s in self.step_list if s["fields"].get("phase") == "engine"),
                "named_steps": dict(Counter(s["fields"].get("phase", "unnamed") for s in self.step_list if s["fields"].get("phase") != "engine")),
                "unpaired_markers": self.unpaired_markers,
                "markers_without_code": self.markers_without_code,
                "overlapping_steps": len(ov),
                "max_step_overlap_ns": int(max(ov)) if ov else 0,
                "rule_for_other_queues": "cpu owner" if self.cpu_rule else "gpu time",
                "compute_queues": [hex(q) for q in self.compute_queues],
                "kernels_inside_one_step": int((gp & (self._inside == 1)).sum()),
                "kernels_inside_more_than_one_step": int((gp & (self._inside > 1)).sum()),
                "kernels_straddling_a_step_end": int((gp & self._straddle).sum()),
                "kernels_outside_any_step": int(outside.sum()),
                "kernels_outside_any_step_before_first_engine_step": int((outside & before).sum()),
                "kernels_outside_any_step_after_first_engine_step": int((outside & ~before).sum()),
                "side_stream_kernels": int(self._side.sum()),
                "side_stream_kernels_owned": int((self._side & (self.step >= 0)).sum()),
                "side_stream_kernels_unowned_after_first_engine_step": int((self._side & (self.step < 0) & ~before).sum()),
                "side_stream_kernels_running_outside_their_step": int(self.outside_owner.sum()),
            })
            steps_rep["every_kernel_in_exactly_one_step"] = (
                steps_rep["kernels_inside_more_than_one_step"] == 0
                and steps_rep["kernels_straddling_a_step_end"] == 0
                and steps_rep["kernels_outside_any_step"] == 0
                and steps_rep["side_stream_kernels_owned"] == steps_rep["side_stream_kernels"]
            )
            steps_rep["every_kernel_after_first_engine_step_in_one_step"] = (
                steps_rep["kernels_outside_any_step_after_first_engine_step"] == 0
                and steps_rep["kernels_inside_more_than_one_step"] == 0
            )
            steps_rep["outside_or_straddling_after_first_engine_step"] = self._stray_detail(
                np.nonzero((outside | (gp & self._straddle)) & ~before)[0][:20])
        span = int(d["end"].max() - d["start"].min()) if len(d) else 0
        buf = p.meta.get("buffer", {})
        return {
            "process": p.name,
            "label": p.label,
            "context": p.context,
            "dispatch_rate_per_s": round(len(d) / (span / 1e9), 1) if span else None,
            "dispatches": {"total": int(len(d)), "work": int(work.sum()), "markers": int(self.is_marker.sum()),
                           "profiler": int(self.is_profiler.sum()),
                           "graph_replayed": int(getattr(self, "is_replayed", np.zeros(len(d), bool)).sum())},
            "gpu_time_ns": {"work": t_total, "attributed": t_attr},
            "attributed_share": (t_attr / t_total) if t_total else None,
            "attributed_by": dict(by),
            "unattributed": [{"kernel": k, "count": un[k], "gpu_time_ns": un_t[k]} for k, _ in un.most_common(50)],
            "graphs": {f"{g['key'] or 'capture ' + str(g['capture'])}": g for g in self.graphs},
            "graph_summary": {"captures": getattr(self, "captures", 0), "graphs": len(self.graphs), "joined": sum(1 for g in self.graphs if g["joined"]),
                              "refused": sum(1 for g in self.graphs if not g["joined"]),
                              "replays": sum(g["replays"] for g in self.graphs),
                              "replays_joined": sum(g.get("replays_joined", 0) for g in self.graphs),
                              "replays_refused": sum(g.get("replays_refused", g["replays"] if not g["joined"] else 0) for g in self.graphs),
                              "keys_shared_by_several_graphs": sum(1 for k, v in Counter(g["key"] for g in self.graphs if g["key"]).items() if v > 1)},
            "steps": steps_rep,
            # memory copies and sets (None: the run predates copies.bin, or RUPPELL_COPIES=0)
            "copies": self._copies_report(),
            "cpu_ops": self._cpu_ops_report(timed) if len(p.ops) else None,
            # NVTX and ROCTX ranges (None: the run predates ranges.bin, or ran without RUPPELL_NVTX)
            "nvtx": self._ranges_report(timed),
            # collective calls from an adapter (None: the run has none; roadmap-v2 M8)
            "collectives": self._collectives_report(timed),
            # hardware counters' steps, left out of every time above (None: the run has none; gray-parrot)
            "counter_steps": self._counter_steps_report(work),
            # NCCL's own records of its calls, paired with its kernels (None: no nccl.tsv; M8)
            "nccl": self._nccl_report(),
            "backward_link": getattr(self, "backward_link", None),
            "records": {"api": int(len(p.api)), "capture": int(len(p.cap)), "kernel_symbols": len(p.kernels),
                        "clock_pairs": len(p.clock), "graph_launches": self.n_launches},
            "dropped_records": p.meta.get("dropped_records"),
            # false: the process ended before the profiler finalized; what it
            # recorded up to its last flush is here, anything after is not
            "finalized": p.meta.get("final", True) if p.meta else False,
            # finalized, or its adapter's shutdown flush (ruppell.flush(complete=True))
            "records_complete": p.records_complete,
            "partial_record_bytes_dropped": p.truncated,
            "buffer": {
                **buf,
                "records_per_s": round((p.meta.get("records", {}).get("dispatch", 0) + p.meta.get("records", {}).get("api", 0))
                                       / ((buf.get("last_callback_ts", 0) - buf.get("first_callback_ts", 0)) / 1e9), 1)
                if buf.get("last_callback_ts", 0) > buf.get("first_callback_ts", 0) else None,
            },
            "build": p.run_info,
        }

    def _stray_detail(self, idx: np.ndarray) -> List[dict]:
        """Name each kernel that is not inside exactly one step, with what is
        known about why: who launched it, from which thread, inside which
        step's CPU window, and the GPU steps around it."""
        p, d = self.proc, self.proc.d
        if not len(idx):
            return []
        st = np.array([s["start"] for s in self.step_list], dtype=np.uint64)
        a = p.api
        ao = np.argsort(a["corr"]) if len(a) else None
        out = []
        for i in idx.tolist():
            name = self.kernel_name.get(int(d["kernel_id"][i]), "")
            k = int(np.searchsorted(st, d["start"][i], side="right")) - 1
            prev = self.step_list[k] if k >= 0 else None
            nxt = self.step_list[k + 1] if k + 1 < len(self.step_list) else None
            call = None
            if ao is not None:
                j = int(np.searchsorted(a["corr"][ao], d["corr"][i]))
                if j < len(ao) and a["corr"][ao[j]] == d["corr"][i]:
                    r = a[ao[j]]
                    t = int(r["start"])
                    owners = [s["step"] for s in self.step_list
                              if s["cpu_begin"] is not None and s["cpu_end"] is not None and s["cpu_begin"] <= t <= s["cpu_end"]]
                    call = {"op": p.api_ops.get(int(r["op"])), "thread": int(r["thread_id"]),
                            "launched_before_start_ns": int(d["start"][i]) - t, "inside_cpu_window_of_steps": owners[:3]}
            out.append({
                "kernel": short_name(name), "queue": hex(int(d["queue_id"][i])),
                "duration_ns": int(d["end"][i] - d["start"][i]), "straddles_a_step_end": bool(self._straddle[i]),
                "scope": self.scope_name(int(self.scope[i])) if self.scope[i] > 0 else None,
                "launch": call,
                "previous_step": prev and {"step": prev["step"], "phase": prev["fields"].get("phase"),
                                           "ended_ns_before": int(d["start"][i]) - int(prev["end"])},
                "next_step": nxt and {"step": nxt["step"], "phase": nxt["fields"].get("phase"),
                                      "starts_ns_after": int(nxt["start"]) - int(d["end"][i])},
            })
        return out

    def scope_name(self, sid: int) -> Optional[str]:
        v = self.proc.defs.get(int(sid))
        return v[1] if v and v[0] == "scope" else None

    def scope_info(self, sid: int) -> Optional[dict]:
        v = self.proc.defs.get(int(sid))
        if not v or v[0] != "scope":
            return None
        return {"id": int(sid), "name": v[1], **{k: x for k, x in v[2].items() if k != "parent"}}


# ---- several processes ------------------------------------------------------------------------

def merge_report(analyses: List[Analysis]) -> dict:
    """Cross-process checks: step ids and fingerprints agree across ranks (9.1)."""
    per = {}
    for a in analyses:
        fp = {s["step"]: s["fields"].get("fingerprint") for s in a.step_list if s["fields"].get("phase") == "engine"}
        if fp:
            per[a.proc.label] = fp
    out = {"processes": [a.proc.label for a in analyses], "ranks_with_engine_steps": sorted(per)}
    if len(per) >= 2:
        labels = sorted(per)
        ids = [set(per[l]) for l in labels]
        common = set.intersection(*ids)
        mismatch = []
        for s in sorted(common):
            vals = {per[l][s] for l in labels}
            if len(vals) > 1:
                mismatch.append(s)
        out.update({
            "engine_steps_per_rank": {l: len(per[l]) for l in labels},
            "step_ids_identical_across_ranks": all(x == ids[0] for x in ids),
            "fingerprint_mismatches": len(mismatch),
            "first_divergent_step": mismatch[0] if mismatch else None,
            "merge_ok": all(x == ids[0] for x in ids) and not mismatch,
        })
    else:
        out["merge_ok"] = True
    return out


def analyze_run(path: str) -> Dict:
    dirs = process_dirs(path)
    if not dirs:
        raise FileNotFoundError(f"no process records under {path}")
    analyses = [Analysis(Proc(p)) for p in dirs]
    return {"analyses": analyses, "merge": merge_report(analyses)}
