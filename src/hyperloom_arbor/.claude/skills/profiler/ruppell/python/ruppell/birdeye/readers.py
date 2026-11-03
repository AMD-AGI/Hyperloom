"""Every source birdeye reads, as one normalized kernel table.

One row per GPU activity (kernel, memcpy, memset), in numpy columns:

| column | meaning |
| --- | --- |
| `rank` | index into `Kernels.ranks` (one per process that ran GPU work) |
| `device`, `stream` | GPU and stream (ROCm: queue) the activity ran on |
| `name` | index into `Kernels.names`: the demangled full name; `short()` gives the short one |
| `start`, `end` | GPU times, ns, in the source's own time base |
| `kind` | 0 kernel, 1 memcpy, 2 memset (the source's event kind) |
| `step` | step id, -1 outside every step |
| `scope` | index into `scopes` (ruppell's module path), -1 none |
| `op` | index into `ops`: the aten op (CPU op tree, PyTorch profiler) or ruppell scope's op, -1 none |
| `sig` | index into `sigs`: the op's (input dims, input types), -1 none |
| `inst` | the op invocation (launch group) the activity belongs to |
| `launch` | 0 unknown, 1 eager, 2 graph replay |
| `linked` | the launching call is in the source |

Readers, each reusing what ruppell already has:

- a ruppell run directory: `engine.analyze_run` (graph join, steps, scopes,
  the CPU op tree, backward attribution), not a second parser;
- ruppell's Chrome trace (`ruppell export`): its kernel events carry scope,
  op, step and attribution;
- a PyTorch profiler (Kineto) trace, `.json` or `.json.gz`: each kernel's op
  is the innermost `cpu_op` open on the thread of the runtime call that
  launched it (same correlation id), as TraceLens links them; steps from
  ruppell's step markers when the trace has them, else `ProfilerStep#N`;
- an Nsight Systems SQLite export (`validate.witness._read_nsys`) and a
  rocprofv3 kernel-trace CSV (`validate.witness._read_csv`): kernels, no
  ops; steps from ruppell's step markers (`validate.witness._marker_steps`).
"""

from __future__ import annotations

import csv
import glob
import gzip
import hashlib
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ruppell.engine import MARKER_KERNEL, demangle, short_name

KIND_KERNEL, KIND_MEMCPY, KIND_MEMSET = 0, 1, 2
KIND_NAMES = {KIND_KERNEL: "kernel", KIND_MEMCPY: "memcpy", KIND_MEMSET: "memset"}
LAUNCH_UNKNOWN, LAUNCH_EAGER, LAUNCH_GRAPH = 0, 1, 2
LAUNCH_NAMES = {LAUNCH_UNKNOWN: "unknown", LAUNCH_EAGER: "eager", LAUNCH_GRAPH: "graph"}
# a collective call's fields in the table (the rank and how it was launched stay out, so one
# call kind is one entry across ranks)
COLL_KEYS = ("op", "backend", "dtype", "numel", "bytes", "world")
FORMATS = ("ruppell", "nsys", "rocprof", "kineto")

COLS = {"rank": np.int32, "device": np.int64, "stream": np.int64, "name": np.int32, "start": np.int64,
        "end": np.int64, "kind": np.int8, "step": np.int64, "scope": np.int32, "op": np.int32, "sig": np.int32,
        "inst": np.int64, "launch": np.int8, "linked": bool, "coll": np.int32}


def step_kind(fields: dict) -> str:
    """A step's kind: an engine step's kind (decode, prefill, mixed), else its
    phase (capture, warmup, a named startup step), else "step"."""
    if not fields:
        return "step"
    return str(fields.get("kind") or fields.get("phase") or "step")


class Kernels:
    """The normalized kernel table, its steps, ranks and provenance."""

    def __init__(self):
        self.c: Dict[str, np.ndarray] = {k: np.zeros(0, dt) for k, dt in COLS.items()}
        self.names: List[str] = []
        self.ops: List[str] = []
        self.scopes: List[str] = []
        self.sigs: List[tuple] = []
        # collective calls (an adapter's, roadmap-v2 M8): op, backend, dtype, numel, bytes, world
        self.colls: List[dict] = []
        self.ranks: List[dict] = []
        # steps: rank, step id, GPU start and end (ns), kind (index into step_kinds), fields
        self.steps: Dict[str, np.ndarray] = {"rank": np.zeros(0, np.int32), "step": np.zeros(0, np.int64),
                                             "start": np.zeros(0, np.int64), "end": np.zeros(0, np.int64),
                                             "kind": np.zeros(0, np.int32)}
        self.step_fields: List[dict] = []
        self.step_kinds: List[str] = []
        self.info: dict = {}
        self._short: Optional[List[str]] = None

    def __len__(self) -> int:
        return len(self.c["start"])

    def __getattr__(self, k):
        c = self.__dict__.get("c")
        if c is not None and k in c:
            return c[k]
        raise AttributeError(k)

    @property
    def dur(self) -> np.ndarray:
        return self.c["end"] - self.c["start"]

    def short(self) -> List[str]:
        if self._short is None:
            self._short = [short_name(n) for n in self.names]
        return self._short

    def select(self, mask: np.ndarray, steps_mask: Optional[np.ndarray] = None) -> "Kernels":
        """The rows in ``mask`` (and the steps in ``steps_mask``), sharing the name lists."""
        t = Kernels()
        t.c = {k: v[mask] for k, v in self.c.items()}
        for k in ("names", "ops", "scopes", "sigs", "colls", "ranks", "step_kinds", "info", "_short"):
            setattr(t, k, getattr(self, k))
        sm = steps_mask if steps_mask is not None else np.ones(len(self.steps["step"]), bool)
        t.steps = {k: v[sm] for k, v in self.steps.items()}
        t.step_fields = [f for f, keep in zip(self.step_fields, sm.tolist()) if keep]
        return t


class _Intern:
    def __init__(self):
        self.items: List = []
        self.index: Dict = {}

    def __call__(self, v) -> int:
        if v is None:
            return -1
        j = self.index.get(v)
        if j is None:
            j = self.index[v] = len(self.items)
            self.items.append(v)
        return j

    def map(self, values: Sequence) -> np.ndarray:
        return np.array([self(v) for v in values], dtype=np.int64)


KD_SUFFIX = re.compile(r"(?:\.kd| \[clone \.kd\])$")


def kernel_symbol(name: str) -> str:
    """A kernel's name without the `.kd` of its kernel-descriptor symbol, which
    ROCm's PyTorch profiler (Kineto) keeps (`foo.kd`, demangled `f(...) [clone
    .kd]`) and a ruppell run does not, so every source names a kernel alike
    (MI350, torch 2.13: every Kineto name matched a ruppell one once stripped)."""
    return KD_SUFFIX.sub("", name)


class Builder:
    """Collects one part per rank and interns names, ops, scopes and
    signatures into one table."""

    def __init__(self, info: dict):
        self.info = info
        self.parts: List[dict] = []
        self.names, self.ops, self.scopes, self.sigs, self.kinds = _Intern(), _Intern(), _Intern(), _Intern(), _Intern()
        self.colls = _Intern()

    def add(self, rank: dict, n: int, *, names: List[str], name_idx, start, end, device=None, stream=None,
            kind=None, step=None, ops: Optional[List] = None, op_idx=None, scopes: Optional[List] = None,
            scope_idx=None, sigs: Optional[List] = None, sig_idx=None, inst=None, launch=None, linked=None,
            steps: Optional[List[dict]] = None, colls: Optional[List] = None, coll_idx=None):
        """One rank. ``*_idx``: per-row indices into the part's own lists
        (-1 for none); ``inst``: per-row launch-group ids, unique within the part;
        ``colls``: collective calls as dicts (op, backend, dtype, numel, bytes, world)."""
        def col(v, dt, fill):
            if v is None:
                return np.full(n, fill, dt)
            return np.asarray(v).astype(dt, copy=False)

        def remap(lst, idx, intern):
            if lst is None or idx is None:
                return np.full(n, -1, np.int32)
            m = intern.map(lst) if lst else np.zeros(0, np.int64)
            idx = np.asarray(idx, dtype=np.int64)
            out = np.full(n, -1, np.int32)
            ok = idx >= 0
            out[ok] = m[idx[ok]]
            return out

        sig_keys = None
        if sigs is not None:
            sig_keys = [json.dumps(s, sort_keys=True, default=str) if s is not None else None for s in sigs]
        part = {
            "rank": rank, "n": n,
            "name": remap([kernel_symbol(x) for x in names] if names is not None else None, name_idx, self.names),
            "start": col(start, np.int64, 0), "end": col(end, np.int64, 0),
            "device": col(device, np.int64, 0), "stream": col(stream, np.int64, 0),
            "kind": col(kind, np.int8, KIND_KERNEL), "step": col(step, np.int64, -1),
            "op": remap(ops, op_idx, self.ops), "scope": remap(scopes, scope_idx, self.scopes),
            "sig": remap(sig_keys, sig_idx, self.sigs),
            "inst": col(inst, np.int64, -1) if inst is not None else np.arange(n, dtype=np.int64),
            "launch": col(launch, np.int8, LAUNCH_UNKNOWN), "linked": col(linked, bool, False),
            "coll": remap([json.dumps(c, sort_keys=True, default=str) if c else None for c in colls]
                          if colls is not None else None, coll_idx, self.colls),
            "steps": steps or [],
        }
        self.parts.append(part)

    def build(self) -> Kernels:
        t = Kernels()
        t.info = self.info
        known = [p["rank"].get("rank") for p in self.parts]
        any_known = any(r is not None for r in known)
        cols = {k: [] for k in COLS}
        srank, sstep, sstart, send, skind, sfields = [], [], [], [], [], []
        inst_base = 0
        for i, p in enumerate(self.parts):
            r = p["rank"]
            if r.get("rank") is not None:
                r["label"] = f"rank {r['rank']}"
            else:
                r["label"] = f"process {i}" if any_known else f"rank {i}"
                r["rank_from"] = "order"
            t.ranks.append(r)
            n = p["n"]
            cols["rank"].append(np.full(n, i, np.int32))
            for k in COLS:
                if k in ("rank",):
                    continue
                v = p[k]
                if k == "inst":
                    v = np.where(v >= 0, v + inst_base, -1)
                    inst_base = int(v.max()) + 1 if n else inst_base
                cols[k].append(v)
            for s in p["steps"]:
                srank.append(i)
                sstep.append(int(s["step"]))
                sstart.append(int(s["start"]))
                send.append(int(s["end"]))
                skind.append(self.kinds(s.get("kind") or step_kind(s.get("fields") or {})))
                sfields.append(s.get("fields") or {})
        if not any_known and self.parts:
            t.info.setdefault("notes", []).append("ranks are numbered in the order the source lists processes; "
                                                  "it does not name them")
        for k, dt in COLS.items():
            t.c[k] = np.concatenate(cols[k]).astype(dt, copy=False) if cols[k] else np.zeros(0, dt)
        t.names, t.ops, t.scopes, t.step_kinds = self.names.items, self.ops.items, self.scopes.items, self.kinds.items
        t.sigs = [tuple(json.loads(s)) for s in self.sigs.items]
        t.colls = [json.loads(s) for s in self.colls.items]
        t.steps = {"rank": np.array(srank, np.int32), "step": np.array(sstep, np.int64),
                   "start": np.array(sstart, np.int64), "end": np.array(send, np.int64),
                   "kind": np.array(skind, np.int32)}
        t.step_fields = sfields
        return t


# ---- provenance -----------------------------------------------------------------------------

def file_digest(path: str) -> dict:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return {"path": os.path.abspath(path), "bytes": os.path.getsize(path), "sha256": h.hexdigest()[:16]}


# ---- format detection -----------------------------------------------------------------------

def detect(src: str) -> str:
    if os.path.isdir(src):
        from ruppell.engine import process_dirs
        if process_dirs(src):
            return "ruppell"
        if glob.glob(os.path.join(src, "**", "*kernel_trace.csv"), recursive=True):
            return "rocprof"
        if glob.glob(os.path.join(src, "**", "*.sqlite"), recursive=True):
            return "nsys"
        if _json_files(src):
            return "kineto"
        raise ValueError(f"{src}: no ruppell records, rocprofv3 CSV, nsys SQLite or JSON trace found")
    low = src.lower()
    if low.endswith(".nsys-rep"):
        raise ValueError(f"{src}: export it to SQLite first (nsys export --type sqlite)")
    if low.endswith(".sqlite"):
        return "nsys"
    if low.endswith(".csv"):
        return "rocprof"
    if low.endswith(".json") or low.endswith(".json.gz"):
        return "kineto"
    raise ValueError(f"{src}: cannot tell its format; pass --format")


def _json_files(d: str) -> List[str]:
    out = sorted(glob.glob(os.path.join(d, "*.json")) + glob.glob(os.path.join(d, "*.json.gz")))
    return [p for p in out if not os.path.basename(p).startswith(("run.", "report.", "witness.", "census"))]


def read(src: str, fmt: str = "auto") -> Kernels:
    fmt = detect(src) if fmt == "auto" else fmt
    if fmt == "ruppell":
        return read_ruppell(src)
    if fmt == "nsys":
        return read_nsys(src)
    if fmt == "rocprof":
        return read_rocprof(src)
    if fmt == "kineto":
        return read_json(src)
    raise ValueError(f"unknown format {fmt}")


# ---- a ruppell run ---------------------------------------------------------------------------

def _uniq_index(values: np.ndarray):
    u, inv = np.unique(values, return_inverse=True)
    return u, inv.reshape(-1)


def read_ruppell(run_dir: str) -> Kernels:
    from ruppell.census import _offset
    from ruppell.engine import COPY_MEMSET, HOW_GRAPH, HOW_GRAPH_NOSCOPE, analyze_run
    from ruppell.limits import for_run

    res = analyze_run(run_dir)
    analyses = [a for a in res["analyses"] if len(a.proc.d) and (~a.is_marker).any()]
    if not analyses:
        raise ValueError(f"{run_dir}: no process ran kernels")
    p0 = analyses[0].proc
    builds = {a.proc.label: a.proc.run_info for a in analyses}
    hosts = {a.proc.name.rsplit("-", 1)[0] for a in analyses}
    info = {"format": "ruppell", "source": os.path.abspath(run_dir),
            "profiler": f"ruppell ({p0.platform}, {'CUPTI' if p0.platform == 'cuda' else 'ROCm profiler SDK'})",
            "platform": p0.platform, "build": p0.run_info, "builds_differ": len({json.dumps(b, sort_keys=True)
                                                                               for b in builds.values()}) > 1,
            "op_tree": any(len(a.proc.ops) for a in analyses),
            "same_clock": len(hosts) == 1,
            "merge": res["merge"], "coverage": [], "notes": []}
    b = Builder(info)
    incomplete = False
    for a in analyses:
        p, d = a.proc, a.proc.d
        work = np.nonzero(~a.is_marker)[0]
        n = len(work)
        uk, kinv = _uniq_index(d["kernel_id"][work])
        names = [a.kernel_name.get(int(k), f"kernel_{int(k)}") for k in uk.tolist()]
        # memory copies and sets (copies.bin) follow the kernels in this rank's rows
        cp = p.copies
        m = len(cp)
        # a kernel without a ruppell scope takes its NVTX (ROCTX) range, as "nvtx:<range>"
        # (roadmap-v2 M7: ruppell's scopes stay innermost, ranges sit outside them); those
        # keys are -2 - the range's name id, apart from ruppell's scope ids (> 0)
        ksc = a.scope[work].astype(np.int64)
        nv = a.nvtx[work] if len(p.ranges) else np.full(n, -1, np.int64)
        by_range = (ksc <= 0) & (nv >= 0)
        if by_range.any():
            ksc = np.where(by_range, -2 - p.ranges["name"][np.clip(nv, 0, None)].astype(np.int64), ksc)
        sc = np.concatenate([ksc, a.copy_scope.astype(np.int64)]) if m else ksc
        us_, sinv = _uniq_index(sc)
        scope_names, scope_ops = [], []
        for s in us_.tolist():
            if s <= -2:
                dom, text = p.range_names.get(-2 - s, ("?", "?"))
                lbl = "nvtx:" + (text if dom in ("default", "roctx") else f"{dom}:{text}")
                scope_names.append(lbl)
                scope_ops.append(lbl)
                continue
            si = a.scope_info(s) if s > 0 else None
            scope_names.append(si["name"] if si else None)
            scope_ops.append(str(si.get("op") or si["name"]) if si else None)
        known = np.array([x is not None for x in scope_names], bool)
        co = a.cpu_op[work]
        has_op = co >= 0
        has_op_all = np.concatenate([has_op, np.zeros(m, bool)])  # copies have no aten op
        ops_list: List[Optional[str]] = list(scope_ops)          # op index space: scope ops first, then aten ops
        op_idx = np.where(known[sinv], sinv, -1).astype(np.int64)
        sigs: List[Optional[tuple]] = []
        sig_idx = np.full(n + m, -1, np.int64)
        if has_op.any():
            sig_id = np.where(has_op, p.ops["id"][np.clip(co, 0, None)].astype(np.int64), -1)
            ug, ginv = _uniq_index(sig_id[has_op])
            base = len(ops_list)
            for g in ug.tolist():
                sg = p.op_sigs.get(int(g), {})
                ops_list.append(str(sg.get("name", f"op {g}")))
                sigs.append((sg.get("dims"), sg.get("types")))
            op_idx[has_op_all] = base + ginv
            sig_idx[has_op_all] = ginv
        # the collective call that launched each kernel (an adapter's, roadmap-v2 M8), with what
        # NCCL chose for it (its profiler plugin's record: algorithm, protocol, channels). A
        # kernel NCCL recorded but no adapter did takes its call from NCCL's record. Copies: none.
        from ruppell.engine import NCCL_DTYPE, NCCL_OP

        tasks = (p.nccl or {}).get("tasks", [])
        nt = getattr(a, "nccl_task", np.full(len(d), -1, np.int64))[work]
        tkind = _Intern()
        tk = np.full(n, -1, np.int64)
        for j in np.nonzero(nt >= 0)[0].tolist():
            k = tasks[int(nt[j])]
            tk[j] = tkind((k["func"], k["datatype"], k["count"], k["algo"], k["proto"], k["channels"], k["comm"]))
        upair, pinv = np.unique(np.stack([a.coll[work].astype(np.int64), tk]), axis=1, return_inverse=True)
        pinv = pinv.reshape(-1)
        coll_list = []
        for c, ti in upair.T.tolist():
            ci = a.coll_info(c) if c > 0 else None
            call = {k: ci.get(k) for k in COLL_KEYS} if ci else None
            if ti >= 0:
                func, dt, cnt, algo, proto, ch, comm = tkind.items[ti]
                if call is None:
                    dname, size = NCCL_DTYPE.get(dt, (dt, None))
                    call = {"op": NCCL_OP.get(func, func), "backend": "nccl", "dtype": dname, "numel": cnt,
                            "bytes": cnt * size if size else None,
                            "world": p.nccl["comms"].get(comm, {}).get("ranks")}
                call = {**call, "algo": algo, "proto": proto, "channels": ch}
            coll_list.append(call)
        has_call = np.array([x is not None for x in coll_list], bool)
        coll_idx = np.concatenate([np.where(has_call[pinv], pinv, -1), np.full(m, -1, np.int64)])
        replayed = getattr(a, "is_replayed", np.zeros(len(d), bool))[work]
        corr = d["corr"][work].astype(np.int64)
        # the op invocation: with the CPU op tree, the aten op record (eager) or
        # (op record, graph, replay) for a replayed kernel; otherwise the launching
        # call (eager) or the kernel itself (one op per replayed kernel, as the
        # PyTorch-profiler export writes it)
        typ = np.where(replayed, np.where(has_op, 2, 3), np.where(has_op, 1, 0)).astype(np.int64)
        k1 = np.where(typ == 0, corr, np.where(typ == 3, work.astype(np.int64), co))
        k2 = np.where(typ == 2, a.graph[work].astype(np.int64), 0)
        k3 = np.where(typ == 2, a.replay[work], 0)
        order = np.lexsort((k3, k2, k1, typ))
        key = np.stack([typ[order], k1[order], k2[order], k3[order]], axis=1)
        new = np.ones(n, bool)
        if n > 1:
            new[1:] = (key[1:] != key[:-1]).any(axis=1)
        inst = np.empty(n, np.int64)
        inst[order] = np.cumsum(new) - 1
        api_corr = np.unique(p.api["corr"]) if len(p.api) else np.zeros(0, np.uint64)
        linked = np.isin(d["corr"][work], api_corr)
        rank = {"rank": p.context.get("rank"), "process": p.name, "source_label": p.label,
                "pid": p.meta.get("pid"), "clock_offset_ns": _offset(a)}
        steps = [{"step": s["step"], "start": s["start"], "end": s["end"], "fields": s["fields"],
                  "kind": step_kind(s["fields"])} for s in a.step_list]
        # copies: one invocation each, named by kind and direction ("Memcpy HtoD", "Memset")
        cnames = sorted(set(a.copy_names))
        cidx = np.array([len(names) + cnames.index(x) for x in a.copy_names], np.int64) if m else np.zeros(0, np.int64)
        c_replayed = (cp["graph_node_id"] != 0) | np.isin(a.copy_how, [HOW_GRAPH, HOW_GRAPH_NOSCOPE])
        cat = lambda x, y: np.concatenate([np.asarray(x), np.asarray(y)]) if m else np.asarray(x)
        b.add(rank, n + m, names=names + cnames, name_idx=cat(kinv, cidx),
              start=cat(d["start"][work].astype(np.int64), cp["start"].astype(np.int64)),
              end=cat(d["end"][work].astype(np.int64), cp["end"].astype(np.int64)),
              device=cat(d["agent_id"][work].astype(np.int64), cp["dst_device"].astype(np.int64)),
              stream=cat(d["queue_id"][work].astype(np.int64), cp["stream_id"].astype(np.int64)),
              kind=cat(np.full(n, KIND_KERNEL, np.int8),
                       np.where(cp["kind"] == COPY_MEMSET, KIND_MEMSET, KIND_MEMCPY).astype(np.int8)),
              step=cat(a.step[work], a.copy_step), ops=ops_list, op_idx=op_idx,
              scopes=scope_names, scope_idx=np.where(known[sinv], sinv, -1), sigs=sigs, sig_idx=sig_idx,
              inst=cat(inst, (inst.max() + 1 if n else 0) + np.arange(m, dtype=np.int64)),
              launch=cat(np.where(replayed, LAUNCH_GRAPH, LAUNCH_EAGER),
                         np.where(c_replayed, LAUNCH_GRAPH, LAUNCH_EAGER)),
              linked=cat(linked, np.isin(cp["corr"], api_corr)), steps=steps, colls=coll_list, coll_idx=coll_idx)
        r = a.report
        incomplete |= bool(r["steps"].get("engine_steps")) and not r["records_complete"]
        info["coverage"].append({
            "process": p.label, "attributed_share": r["attributed_share"], "attributed_by": r["attributed_by"],
            "unattributed": r["unattributed"][:10], "dropped_records": r["dropped_records"],
            "records_complete": r["records_complete"], "partial_record_bytes_dropped": r["partial_record_bytes_dropped"],
            "graph_replays_refused": r["graph_summary"].get("replays_refused"),
            "graph_replays": r["graph_summary"].get("replays"),
            "kernels_outside_any_step": r["steps"].get("kernels_outside_any_step"),
            "every_kernel_in_exactly_one_step": r["steps"].get("every_kernel_in_exactly_one_step"),
            "cpu_ops_gpu_time_share": (r.get("cpu_ops") or {}).get("gpu_time_share_with_an_op"),
            "copies": r.get("copies"),
            "nvtx_gpu_time_share": ((r.get("nvtx") or {}).get("kernels_in_a_range") or {}).get("share_of_work_gpu_time"),
        })
    info["copies_recorded"] = all(a.proc.copies_recorded for a in analyses)
    info["limitations"] = for_run(p0.meta, cpu_ops=info["op_tree"], incomplete=incomplete,
                                  copies=info["copies_recorded"])
    if len(hosts) > 1:
        info["notes"].append("processes on several hosts: start skew across ranks is not computed "
                             "(clock alignment across nodes is untested)")
    return b.build()


# ---- JSON traces: ruppell's Chrome trace, PyTorch profiler (Kineto) --------------------------

def _load_json(path: str):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _ns(x) -> int:
    return int(round(float(x) * 1000.0))


def read_json(src: str) -> Kernels:
    paths = _json_files(src) if os.path.isdir(src) else [src]
    if not paths:
        raise ValueError(f"{src}: no JSON trace")
    traces = [(p, _load_json(p)) for p in paths]
    chrome = [isinstance(t, dict) and isinstance((t.get("otherData") or {}).get("ruppell", {}).get("processes"), list)
              for _, t in traces]
    if any(chrome):
        if len(traces) > 1:
            raise ValueError(f"{src}: one ruppell Chrome trace at a time (it holds every rank already)")
        return _read_ruppell_chrome(*traces[0])
    info = {"format": "kineto", "source": os.path.abspath(src), "profiler": "PyTorch profiler (Kineto)",
            "files": [file_digest(p) for p in paths], "same_clock": len(paths) == 1, "coverage": [], "notes": [],
            "op_tree": True}
    if len(paths) > 1:
        info["notes"].append("several files: each has its own time base, so start skew across ranks is not computed")
    b = Builder(info)
    for i, (path, tr) in enumerate(traces):
        _read_torch(tr, path, b, i)
    t = b.build()
    rup = [isinstance(tr, dict) and (tr.get("otherData") or {}).get("ruppell") for _, tr in traces]
    if all(rup):
        info["profiler"] = "ruppell, PyTorch-profiler form (ruppell export --format torch / ruppell tracelens)"
        info["limitations"] = rup[0].get("limitations")
        info["op_tree"] = bool(rup[0].get("cpu_op_tree"))
    return t


def _innermost(op_s: np.ndarray, op_e: np.ndarray, names: List[str], q: np.ndarray) -> np.ndarray:
    """For each query time, the index of the innermost op open at it ([start,
    end], ops nested on one thread), skipping TraceLens's pass-through
    `execute`; -1 when none. A sweep: at equal times an op starts before a
    point, a point comes before an op ends, a longer op starts first."""
    out = np.full(len(q), -1, np.int64)
    if not len(op_s) or not len(q):
        return out
    no, nq = len(op_s), len(q)
    t = np.concatenate([op_s, q, op_e])
    typ = np.concatenate([np.zeros(no, np.int8), np.ones(nq, np.int8), np.full(no, 2, np.int8)])
    ref = np.concatenate([np.arange(no), np.arange(nq), np.arange(no)])
    tie = np.concatenate([-(op_e - op_s), np.zeros(nq + no, np.int64)])
    stack: List[int] = []
    for j in np.lexsort((tie, typ, t)).tolist():
        k, r = int(typ[j]), int(ref[j])
        if k == 0:
            stack.append(r)
        elif k == 1:
            for o in reversed(stack):
                if names[o] != "execute":
                    out[r] = o
                    break
        elif stack and stack[-1] == r:
            stack.pop()
        elif r in stack:
            stack.remove(r)
    return out


STEP_ANN = re.compile(r"^step (\d+)\b")
PROFILER_STEP = re.compile(r"^ProfilerStep#(\d+)")
GPU_CATS = {"kernel": KIND_KERNEL, "gpu_memcpy": KIND_MEMCPY, "memcpy": KIND_MEMCPY, "gpu_memset": KIND_MEMSET,
            "memset": KIND_MEMSET}
RUNTIME_CATS = {"cuda_runtime", "cuda_driver", "hip_runtime", "hip_driver", "runtime"}


def _assign_steps(starts: np.ndarray, iv: List[Tuple[int, int, int]]) -> np.ndarray:
    """Step id per time point: the step whose [start, end] holds it, -1 none."""
    out = np.full(len(starts), -1, np.int64)
    if not iv or not len(starts):
        return out
    iv = sorted(iv)
    s = np.array([x[0] for x in iv], np.int64)
    e = np.array([x[1] for x in iv], np.int64)
    sid = np.array([x[2] for x in iv], np.int64)
    k = np.searchsorted(s, starts, side="right") - 1
    kc = np.clip(k, 0, None)
    ok = (k >= 0) & (starts <= e[kc])
    out[ok] = sid[kc[ok]]
    return out


def _read_torch(tr, path: str, b: Builder, index: int) -> None:
    from ruppell.validate.witness import _marker_steps

    ev = tr["traceEvents"] if isinstance(tr, dict) else tr
    gpu, rt, ops, gann, cann = [], {}, defaultdict(list), [], []
    for e in ev:
        if e.get("ph") != "X":
            continue
        c = str(e.get("cat") or "").lower()
        if c in GPU_CATS:
            gpu.append(e)
        elif c in RUNTIME_CATS:
            cid = (e.get("args") or {}).get("correlation")
            if cid is not None:
                rt[cid] = e
        elif c in ("cpu_op", "operator"):
            ops[(e.get("pid"), e.get("tid"))].append(e)
        elif c == "gpu_user_annotation":
            gann.append(e)
        elif c == "user_annotation":
            cann.append(e)
    rup = (tr.get("otherData") or {}).get("ruppell") if isinstance(tr, dict) else None
    markers = [e for e in gpu if MARKER_KERNEL in str(e.get("name", ""))]
    gpu = [e for e in gpu if MARKER_KERNEL not in str(e.get("name", ""))]
    n = len(gpu)
    names_i = _Intern()
    name_idx = np.array([names_i(str(e.get("name", ""))) for e in gpu], np.int64)
    start = np.array([_ns(e["ts"]) for e in gpu], np.int64)
    end = start + np.array([_ns(e.get("dur", 0)) for e in gpu], np.int64)
    kind = np.array([GPU_CATS[str(e.get("cat")).lower()] for e in gpu], np.int8)
    args = [e.get("args") or {} for e in gpu]
    device = np.array([int(a.get("device", e.get("pid", 0)) or 0) for a, e in zip(args, gpu)], np.int64)
    stream = np.array([int(a.get("stream", e.get("tid", 0)) or 0) for a, e in zip(args, gpu)], np.int64)
    corr = [a.get("correlation") for a in args]
    rts = [rt.get(c) if c is not None else None for c in corr]
    linked = np.array([r is not None for r in rts], bool)
    launch = np.array([LAUNCH_UNKNOWN if r is None else
                       (LAUNCH_GRAPH if "graphlaunch" in str(r.get("name", "")).lower() else LAUNCH_EAGER)
                       for r in rts], np.int8)
    attribution = [a.get("ruppell_attribution") for a in args]
    if any(x is not None for x in attribution):
        launch = np.where(np.array([str(x or "").startswith("graph") for x in attribution]), LAUNCH_GRAPH, launch)
    # each kernel's op: the innermost cpu_op open at its runtime call, on that call's thread
    op_of = np.full(n, -1, np.int64)
    op_events: List[dict] = []
    by_thread: Dict[tuple, List[int]] = defaultdict(list)
    for i, r in enumerate(rts):
        if r is not None:
            by_thread[(r.get("pid"), r.get("tid"))].append(i)
    for key, kis in by_thread.items():
        evs = ops.get(key, [])
        if not evs:
            continue
        os_ = np.array([_ns(o["ts"]) for o in evs], np.int64)
        oe = os_ + np.array([_ns(o.get("dur", 0)) for o in evs], np.int64)
        q = np.array([_ns(rts[i]["ts"]) for i in kis], np.int64)
        inner = _innermost(os_, oe, [str(o.get("name", "")) for o in evs], q)
        base = len(op_events)
        op_events.extend(evs)
        for i, j in zip(kis, inner.tolist()):
            if j >= 0:
                op_of[i] = base + j
    ops_i, sigs_i = _Intern(), _Intern()
    op_idx = np.full(n, -1, np.int64)
    sig_idx = np.full(n, -1, np.int64)
    for i in np.nonzero(op_of >= 0)[0].tolist():
        o = op_events[int(op_of[i])]
        oa = o.get("args") or {}
        op_idx[i] = ops_i(str(o.get("name", "")))
        dims, types = oa.get("Input Dims"), oa.get("Input type")
        if dims is not None:
            sig_idx[i] = sigs_i(json.dumps([dims, types], default=str))
    # a kernel without an op is its own launch group under its runtime call (TraceLens's launcher)
    inst = np.where(op_of >= 0, op_of, -1)
    nxt = int(op_of.max()) + 1 if n else 0
    rt_inst: Dict[object, int] = {}
    for i in np.nonzero(op_of < 0)[0].tolist():
        c = corr[i]
        if c is None or rts[i] is None:
            inst[i] = nxt
            nxt += 1
        else:
            j = rt_inst.get(c)
            if j is None:
                j = rt_inst[c] = nxt
                nxt += 1
            inst[i] = j
    scopes_i = _Intern()
    scope_idx = np.array([scopes_i(a.get("ruppell_scope")) for a in args], np.int64)

    # steps
    step = np.full(n, -1, np.int64)
    steps: List[dict] = []
    rsteps = [a.get("ruppell_step") for a in args]
    if any(s is not None for s in rsteps):
        step = np.array([int(s) if s is not None else -1 for s in rsteps], np.int64)
        for e in gann:
            m = STEP_ANN.match(str(e.get("name", "")))
            if m:
                s0 = _ns(e["ts"])
                steps.append({"step": int(m.group(1)), "start": s0, "end": s0 + _ns(e.get("dur", 0)),
                              "fields": dict(e.get("args") or {})})
        b.info.setdefault("steps_from", "ruppell step ids on each kernel (ruppell_step)")
    elif markers:
        ms = _marker_steps([str(e.get("name")) for e in markers], [_ns(e["ts"]) for e in markers],
                           [_ns(e["ts"]) + _ns(e.get("dur", 0)) for e in markers])
        if ms is None:
            b.info.setdefault("notes", []).append(f"{os.path.basename(path)}: step markers do not alternate "
                                                  "begin, end; no steps")
        else:
            steps = [{"step": k, "start": s, "end": e_, "fields": {}} for k, (s, e_) in enumerate(ms)]
            step = _assign_steps(start, [(s["start"], s["end"], s["step"]) for s in steps])
            b.info.setdefault("steps_from", "ruppell step marker kernels (numbered in time order)")
    else:
        g = [(e, PROFILER_STEP.match(str(e.get("name", "")))) for e in gann]
        g = [(e, m) for e, m in g if m]
        if g:
            steps = [{"step": int(m.group(1)), "start": _ns(e["ts"]), "end": _ns(e["ts"]) + _ns(e.get("dur", 0)),
                      "fields": {"phase": "ProfilerStep"}} for e, m in g]
            step = _assign_steps(start, [(s["start"], s["end"], s["step"]) for s in steps])
            b.info.setdefault("steps_from", "ProfilerStep#N GPU annotations")
        else:
            c = [(e, PROFILER_STEP.match(str(e.get("name", "")))) for e in cann]
            c = [(e, m) for e, m in c if m]
            if c:
                iv = [(_ns(e["ts"]), _ns(e["ts"]) + _ns(e.get("dur", 0)), int(m.group(1))) for e, m in c]
                lt = np.array([_ns(r["ts"]) if r is not None else -1 for r in rts], np.int64)
                step = np.where(lt >= 0, _assign_steps(lt, iv), -1)
                for _, _, sid in iv:
                    sel = step == sid
                    if sel.any():
                        steps.append({"step": sid, "start": int(start[sel].min()), "end": int(end[sel].max()),
                                      "fields": {"phase": "ProfilerStep"}})
                b.info.setdefault("steps_from", "ProfilerStep#N CPU ranges, by each kernel's launching call; "
                                                "a step's GPU interval is its kernels' span")
    rank_no = None
    di = tr.get("distributedInfo") if isinstance(tr, dict) else None
    if isinstance(di, dict) and di.get("rank") is not None:
        rank_no = int(di["rank"])
    elif rup and isinstance(rup.get("process"), str):
        m = re.match(r"rank (\d+)", rup["process"])
        rank_no = int(m.group(1)) if m else None
    b.add({"rank": rank_no, "process": os.path.basename(path), "file": path}, n,
          names=names_i.items, name_idx=name_idx, start=start, end=end, device=device, stream=stream, kind=kind,
          step=step, ops=ops_i.items, op_idx=op_idx, scopes=scopes_i.items, scope_idx=scope_idx,
          sigs=[tuple(json.loads(s)) for s in sigs_i.items], sig_idx=sig_idx, inst=inst, launch=launch,
          linked=linked, steps=steps)
    b.info["coverage"].append({"process": os.path.basename(path), "kernels": n,
                               "unlinked_kernels": int((~linked).sum()),
                               "kernels_with_an_op": int((op_of >= 0).sum())})


def _read_ruppell_chrome(path: str, tr: dict) -> Kernels:
    from ruppell.limits import for_run

    rep = tr["otherData"]["ruppell"]
    pname: Dict[int, str] = {}
    for e in tr["traceEvents"]:
        if e.get("ph") == "M" and e.get("name") == "process_name":
            pname[e.get("pid")] = (e.get("args") or {}).get("name", "")
    rows: Dict[int, List[dict]] = defaultdict(list)
    api_corr: Dict[int, set] = defaultdict(set)
    steps: Dict[int, List[dict]] = defaultdict(list)
    for e in tr["traceEvents"]:
        if e.get("ph") != "X":
            continue
        cat, pid = e.get("cat"), int(e.get("pid", 0))
        if cat == "kernel" and pid >= 10000:
            rows[(pid - 10000) // 100].append(e)
        elif cat in ("cuda_api", "hip_api") and pid < 10000:
            api_corr[(pid - 1) // 10].add((e.get("args") or {}).get("corr"))
        elif cat == "step" and pid >= 10000:
            a = dict(e.get("args") or {})
            s0 = _ns(e["ts"])
            steps[(pid - 10000) // 100].append({"step": int(a.pop("step")), "start": s0,
                                                "end": s0 + _ns(e.get("dur", 0)), "fields": a})
    procs = {p.get("label"): p for p in rep.get("processes", [])}
    builds = [p.get("build") for p in rep.get("processes", []) if p.get("build")]
    info = {"format": "kineto", "source": os.path.abspath(path), "profiler": "ruppell (Chrome trace from ruppell export)",
            "files": [file_digest(path)], "same_clock": True, "coverage": [], "notes": [],
            "build": builds[0] if builds else None, "limitations": rep.get("limitations"),
            "window": rep.get("window"), "merge": rep.get("merge"),
            "op_tree": False}
    b = Builder(info)
    for k in sorted(rows):
        evs = rows[k]
        label = pname.get(10 * k + 1, f"process {k}").split(": host")[0]
        m = re.match(r"rank (\d+)", label)
        n = len(evs)
        args = [e.get("args") or {} for e in evs]
        names_i, ops_i, scopes_i, sigs_i = _Intern(), _Intern(), _Intern(), _Intern()
        name_idx = np.array([names_i(str(a.get("kernel", e.get("name")))) for a, e in zip(args, evs)], np.int64)
        start = np.array([_ns(e["ts"]) for e in evs], np.int64)
        end = start + np.array([_ns(e.get("dur", 0)) for e in evs], np.int64)

        def dev_of(e):
            mm = re.search(r"GPU (?:agent )?(\d+)$", pname.get(e.get("pid"), ""))
            return int(mm.group(1)) if mm else int(e.get("pid", 0))

        def stream_of(a):
            if "stream" in a:
                return int(a["stream"])
            q = a.get("queue")
            return int(q, 16) if isinstance(q, str) and q.startswith("0x") else int(q or 0)

        op_idx, scope_idx, sig_idx = [], [], []
        for a in args:
            sc = a.get("scope") if isinstance(a.get("scope"), dict) else None
            opd = a.get("op") if isinstance(a.get("op"), dict) else None
            scope_idx.append(scopes_i(sc.get("name") if sc else None))
            if opd:
                info["op_tree"] = True
                op_idx.append(ops_i(str(opd.get("name"))))
                sig_idx.append(sigs_i(json.dumps([opd.get("dims"), opd.get("types")], default=str))
                               if opd.get("dims") is not None else -1)
            else:
                op_idx.append(ops_i(str(sc.get("op") or sc.get("name")) if sc else None))
                sig_idx.append(-1)
        colls_i = _Intern()  # the collective call ruppell's export put beside the scope (M8)
        coll_idx = np.array([colls_i(json.dumps({k: a["collective"].get(k) for k in COLL_KEYS}, sort_keys=True, default=str))
                             if isinstance(a.get("collective"), dict) else -1 for a in args], np.int64)
        att = [str(a.get("attribution", "")) for a in args]
        graph = np.array([x.startswith("graph") or "graph" in a for x, a in zip(att, args)], bool)
        corr = [a.get("corr") for a in args]
        # one op per launching call (eager), one per replayed kernel: the
        # PyTorch-profiler export's op per launch, which TraceLens reads
        inst_key = _Intern()
        inst = np.array([inst_key(("k", i)) if g else inst_key(("c", c)) for i, (g, c) in enumerate(zip(graph, corr))],
                        np.int64)
        known = api_corr.get(k, set())
        linked = np.array([c in known for c in corr], bool)
        step = np.array([int(a["step"]) if a.get("step") is not None else -1 for a in args], np.int64)
        b.add({"rank": int(m.group(1)) if m else None, "process": label, "source_label": label}, n,
              names=names_i.items, name_idx=name_idx, start=start, end=end,
              device=np.array([dev_of(e) for e in evs], np.int64), stream=np.array([stream_of(a) for a in args], np.int64),
              step=step, ops=ops_i.items, op_idx=np.array(op_idx, np.int64), scopes=scopes_i.items,
              scope_idx=np.array(scope_idx, np.int64), sigs=[tuple(json.loads(s)) for s in sigs_i.items],
              sig_idx=np.array(sig_idx, np.int64), inst=inst,
              launch=np.where(graph, LAUNCH_GRAPH, LAUNCH_EAGER), linked=linked,
              steps=[dict(s, kind=step_kind(s["fields"])) for s in steps.get(k, [])],
              colls=[json.loads(s) for s in colls_i.items], coll_idx=coll_idx)
        pr = procs.get(label, {})
        info["coverage"].append({"process": label, "kernels": n, "unlinked_kernels": int((~linked).sum()),
                                 "attributed_share": pr.get("attributed_share"),
                                 "attributed_by": pr.get("attributed_by"),
                                 "unattributed": (pr.get("unattributed") or [])[:10],
                                 "dropped_records": pr.get("dropped_records"),
                                 "records_complete": pr.get("records_complete")})
    info["steps_from"] = "ruppell steps (exported with the trace)"
    if not info["limitations"]:
        # exported before limitations travelled with the trace: ruppell's base ones
        info["limitations"] = for_run(None, cpu_ops=info["op_tree"])
    return b.build()


# ---- Nsight Systems (SQLite export) ----------------------------------------------------------

COPY_KINDS = {1: "HtoD", 2: "DtoH", 3: "HtoA", 4: "AtoH", 5: "AtoA", 6: "AtoD", 7: "DtoA", 8: "DtoD", 9: "HtoH",
              10: "PtoP"}


def read_nsys(src: str) -> Kernels:
    import sqlite3

    from ruppell.validate.witness import _marker_steps, _read_nsys

    path = src
    if os.path.isdir(src):
        sq = sorted(glob.glob(os.path.join(src, "**", "*.sqlite"), recursive=True))
        if len(sq) != 1:
            raise ValueError(f"{src}: {len(sq)} SQLite files; name one")
        path = sq[0]
    by_pid = _read_nsys(path)
    con = sqlite3.connect(path)
    tables = {r[0] for r in con.execute("select name from sqlite_master where type = 'table'")}

    def rows_of(table: str):
        if table not in tables:
            return {}
        cols = {r[1] for r in con.execute(f"pragma table_info({table})")}
        sel = ", ".join(c if c in cols else "0" for c in ("globalPid", "start", "end", "deviceId", "streamId",
                                                           "correlationId", "copyKind"))
        out: Dict[int, list] = defaultdict(list)
        for gpid, s, e, dev, st, cid, ck in con.execute(f"select {sel} from {table}"):
            out[(int(gpid or 0) >> 24) & 0xFFFFFF].append((int(s), int(e), int(dev or 0), int(st or 0), int(cid or 0),
                                                            int(ck or 0)))
        return out

    mc, ms = rows_of("CUPTI_ACTIVITY_KIND_MEMCPY"), rows_of("CUPTI_ACTIVITY_KIND_MEMSET")
    runtime = None
    if "CUPTI_ACTIVITY_KIND_RUNTIME" in tables:
        runtime = {int(r[0]) for r in con.execute("select correlationId from CUPTI_ACTIVITY_KIND_RUNTIME")}
    info = {"format": "nsys", "source": os.path.abspath(path), "profiler": "Nsight Systems (SQLite export)",
            "files": [file_digest(path)], "same_clock": True, "coverage": [], "notes": [], "op_tree": False}
    if runtime is None:
        info["notes"].append("no CUPTI_ACTIVITY_KIND_RUNTIME table: launching calls unknown (kernels counted as linked)")
    b = Builder(info)
    for pid in sorted(set(by_pid) | set(mc) | set(ms)):
        k = by_pid.get(pid, [])
        mk = [w for w in k if MARKER_KERNEL in w["name"]]
        k = [w for w in k if MARKER_KERNEL not in w["name"]]
        names_i = _Intern()
        # Demangled by ruppell's own demangler, as a ruppell run's names are, so the two
        # line up name for name: nsys's demangledName spells some differently ("(int)8"
        # for "8"; B200, runner ruppell-accuracy 2026-10-19). nsys's is the fallback.
        dm = demangle([w["name"] for w in k]) if k else {}
        name_idx = [names_i(dm.get(w["name"]) or w.get("demangled") or w["name"]) for w in k]
        start = [w["start"] for w in k]
        end = [w["end"] for w in k]
        dev = [w.get("device", 0) for w in k]
        stream = [w["stream"] for w in k]
        kind = [KIND_KERNEL] * len(k)
        launch = [LAUNCH_GRAPH if w["graph_node"] else LAUNCH_EAGER for w in k]
        corr = [w.get("corr", 0) for w in k]
        for rows, kd in ((mc.get(pid, []), KIND_MEMCPY), (ms.get(pid, []), KIND_MEMSET)):
            for s, e, d_, st, cid, ck in rows:
                name_idx.append(names_i("Memset" if kd == KIND_MEMSET else f"Memcpy {COPY_KINDS.get(ck, ck)}"))
                start.append(s)
                end.append(e)
                dev.append(d_)
                stream.append(st)
                kind.append(kd)
                launch.append(LAUNCH_EAGER)
                corr.append(cid)
        n = len(start)
        start_a = np.array(start, np.int64)
        steps: List[dict] = []
        step = np.full(n, -1, np.int64)
        if mk:
            sp = _marker_steps([w["name"] for w in mk], [w["start"] for w in mk], [w["end"] for w in mk])
            if sp is None:
                info["notes"].append(f"pid {pid}: step markers do not alternate begin, end; no steps")
            else:
                steps = [{"step": i, "start": s, "end": e, "fields": {}} for i, (s, e) in enumerate(sp)]
                step = _assign_steps(start_a, [(s["start"], s["end"], s["step"]) for s in steps])
                info["steps_from"] = "ruppell step marker kernels (numbered in time order)"
        linked = np.ones(n, bool) if runtime is None else np.isin(np.array(corr, np.int64), list(runtime))
        b.add({"rank": None, "process": f"pid {pid}", "pid": pid}, n, names=names_i.items,
              name_idx=np.array(name_idx, np.int64), start=start_a, end=np.array(end, np.int64),
              device=np.array(dev, np.int64), stream=np.array(stream, np.int64), kind=np.array(kind, np.int8),
              step=step, launch=np.array(launch, np.int8), linked=linked, steps=steps)
        info["coverage"].append({"process": f"pid {pid}", "kernels": n, "unlinked_kernels": int((~linked).sum())})
    return b.build()


# ---- rocprofv3 (kernel-trace CSV) ------------------------------------------------------------

def read_rocprof(src: str) -> Kernels:
    from ruppell.validate.witness import _marker_steps, _pid_of_file, _read_csv

    files = (sorted(glob.glob(os.path.join(src, "**", "*kernel_trace.csv"), recursive=True))
             if os.path.isdir(src) else [src])
    if not files:
        raise ValueError(f"{src}: no *kernel_trace.csv")
    info = {"format": "rocprof", "source": os.path.abspath(src), "profiler": "rocprofv3 --kernel-trace (CSV)",
            "files": [file_digest(p) for p in files], "same_clock": True, "coverage": [], "notes": [],
            "op_tree": False, "launches_recorded": False}
    by_pid: Dict[int, List[dict]] = defaultdict(list)
    for f in files:
        rows = _read_csv(f)["rows"]
        fp = _pid_of_file(f)
        for r in rows:
            by_pid[r["pid"] if r.get("pid") is not None else fp].append(r)
        # memory copies beside the kernel trace, when rocprofv3 wrote them
        for mf in sorted(glob.glob(os.path.join(os.path.dirname(f) or ".", "*memory_copy_trace.csv"))):
            if len(files) > 1 and _pid_of_file(mf) != fp:
                continue                      # each process's copies go with its own kernel file
            with open(mf, newline="") as fh:
                for row in csv.DictReader(fh):
                    if not row.get("Start_Timestamp"):
                        continue
                    by_pid[int(row["Pid"]) if row.get("Pid") else fp].append({
                        "name": "MEMORY_COPY_" + (row.get("Direction") or row.get("Kind") or "").upper(),
                        "start": int(row["Start_Timestamp"]), "end": int(row["End_Timestamp"]),
                        "agent": int(row["Destination_Agent_Id"], 0) if row.get("Destination_Agent_Id") else None,
                        "queue": None, "stream": int(row["Stream_Id"], 0) if row.get("Stream_Id") else None,
                        "memcpy": True})
    b = Builder(info)
    for pid in sorted(by_pid):
        rows = by_pid[pid]
        mk = [w for w in rows if MARKER_KERNEL in w["name"]]
        k = [w for w in rows if MARKER_KERNEL not in w["name"]]
        names_i = _Intern()
        name_idx = np.array([names_i(w["name"]) for w in k], np.int64)
        start = np.array([w["start"] for w in k], np.int64)
        end = np.array([w["end"] for w in k], np.int64)
        n = len(k)
        steps: List[dict] = []
        step = np.full(n, -1, np.int64)
        if mk:
            sp = _marker_steps([w["name"] for w in mk], [w["start"] for w in mk], [w["end"] for w in mk])
            if sp is None:
                info["notes"].append(f"pid {pid}: step markers do not alternate begin, end; no steps")
            else:
                steps = [{"step": i, "start": s, "end": e, "fields": {}} for i, (s, e) in enumerate(sp)]
                step = _assign_steps(start, [(s["start"], s["end"], s["step"]) for s in steps])
                info["steps_from"] = "ruppell step marker kernels (numbered in time order)"
        b.add({"rank": None, "process": f"pid {pid}", "pid": pid}, n, names=names_i.items, name_idx=name_idx,
              start=start, end=end, device=np.array([w.get("agent") or 0 for w in k], np.int64),
              stream=np.array([w.get("stream") if w.get("stream") is not None else (w.get("queue") or 0) for w in k],
                              np.int64),
              kind=np.array([KIND_MEMCPY if w.get("memcpy") else KIND_KERNEL for w in k], np.int8), step=step,
              linked=np.zeros(n, bool), steps=steps)
        info["coverage"].append({"process": f"pid {pid}", "kernels": n})
    info["notes"].append("rocprofv3's kernel trace does not say which call launched a kernel or whether a graph did: "
                         "no ops, launch kind unknown")
    return b.build()
