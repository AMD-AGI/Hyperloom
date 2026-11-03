"""The tables, from one normalized kernel table (numpy only).

Sheet names and column names are TraceLens's (`generate_perf_report_pytorch`,
e34b294) where the meaning is the same; docs/birdeye.md says how each is
computed and where it differs. Every time is GPU time from the source's
records; nothing is modelled.

Interval arithmetic is exact, in integer ns: a class of activity
(computation, communication, memcpy, everything) is the union of its
intervals; exposed communication is communication minus computation;
exposed memcpy is memcpy minus computation and communication; busy is the
union of everything; total is the last end minus the first start; idle is
total minus busy. So computation + exposed comm + exposed memcpy + idle =
total, exactly.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from ruppell.birdeye.categories import CATEGORIES, categorize_many, op_category
from ruppell.birdeye.readers import KIND_KERNEL, KIND_MEMCPY, KIND_MEMSET, LAUNCH_EAGER, LAUNCH_GRAPH, Kernels
from ruppell.limits import is_collective

TIMELINE_TYPES = ["computation_time", "exposed_comm_time", "exposed_memcpy_time", "busy_time", "idle_time",
                  "total_time", "total_comm_time", "total_memcpy_time"]
# TraceLens's own collective names (TraceEventUtils.get_communication_regexes at e34b294)
TRACELENS_COMM = re.compile(r"nccl|rccl|cross_device_reduce", re.IGNORECASE)
# TraceLens re-buckets ROCm 7.1's copy kernels as memcpy (TraceEventUtils.is_rocm_legacy_memcpy)
LEGACY_MEMCPY = re.compile(r"^(MEMORY_COPY_(HOST_TO_DEVICE|DEVICE_TO_HOST|DEVICE_TO_DEVICE)"
                           r"|__amd_rocclr_copyBuffer(Rect)?(Aligned)?)(\.kd)?$")
COMP, COMM, MEMCPY = 0, 1, 2
GEMM_OPS = {"aten::mm", "aten::addmm", "aten::bmm", "aten::baddbmm", "aten::matmul", "aten::linear",
            "aten::_scaled_mm"}
DTYPE_BYTES = {"c10::bfloat16": 2, "bfloat16": 2, "c10::half": 2, "half": 2, "float16": 2, "float": 4,
               "float32": 4, "double": 8, "float64": 8, "c10::float8_e4m3fn": 1, "c10::float8_e5m2": 1,
               "c10::float8_e4m3fnuz": 1, "c10::float8_e5m2fnuz": 1, "c10::float8_e8m0fnu": 1, "signed char": 1,
               "unsigned char": 1, "char": 1, "short": 2, "int": 4, "long int": 8, "long": 8, "bool": 1}


class Sheet:
    def __init__(self, name: str, columns: List[str], rows: List[list], note: Optional[str] = None):
        self.name, self.columns, self.rows, self.note = name, columns, rows, note

    def dicts(self) -> List[dict]:
        return [dict(zip(self.columns, r)) for r in self.rows]

    def __len__(self):
        return len(self.rows)


# ---- intervals ------------------------------------------------------------------------------

def merged(s: np.ndarray, e: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The union of [s, e) intervals as sorted disjoint intervals (touching ones merge)."""
    if not len(s):
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    o = np.lexsort((e, s))
    s, e = s[o], e[o]
    run = np.maximum.accumulate(e)
    new = np.ones(len(s), bool)
    new[1:] = s[1:] > run[:-1]
    idx = np.nonzero(new)[0]
    return s[idx], np.maximum.reduceat(e, idx)


def overlap_each(a_s: np.ndarray, a_e: np.ndarray, b_s: np.ndarray, b_e: np.ndarray) -> np.ndarray:
    """For each interval of B, the length of it covered by A (A merged)."""
    if not len(a_s) or not len(b_s):
        return np.zeros(len(b_s), np.int64)
    cum = np.concatenate([[0], np.cumsum(a_e - a_s)])

    def cover(t):
        j = np.searchsorted(a_s, t, side="right") - 1
        jc = np.clip(j, 0, None)
        part = np.clip(np.minimum(t, a_e[jc]) - a_s[jc], 0, None)
        return np.where(j >= 0, cum[jc] + part, 0)

    return cover(b_e) - cover(b_s)


def grouped_busy(s: np.ndarray, e: np.ndarray, g: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Union length of [s, e) per group: (group ids, busy ns)."""
    if not len(s):
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    o = np.lexsort((s, g))
    s, e, g = s[o], e[o], g[o]
    first = np.ones(len(g), bool)
    first[1:] = g[1:] != g[:-1]
    idx = np.nonzero(first)[0]
    gr = np.cumsum(first) - 1
    gmin = s[idx]
    span = int((np.maximum.reduceat(e, idx) - gmin).max()) + 1
    off = gr.astype(np.int64) * span - gmin[gr]
    s2, e2 = s + off, e + off
    run = np.maximum.accumulate(e2)
    prev = np.empty_like(run)
    prev[0] = s2[0]
    prev[1:] = run[:-1]
    prev[first] = s2[first]
    covered = np.clip(e2 - np.maximum(s2, prev), 0, None)
    return g[idx], np.bincount(gr, weights=covered, minlength=len(idx)).astype(np.int64)


def timeline(s: np.ndarray, e: np.ndarray, cls: np.ndarray, total: Optional[int] = None) -> Dict[str, int]:
    """TraceLens's GPU timeline over one set of activities, in ns."""
    if not len(s):
        return {k: 0 for k in TIMELINE_TYPES}
    cs, ce = merged(s[cls == COMP], e[cls == COMP])
    ms, me = merged(s[cls == COMM], e[cls == COMM])
    ys, ye = merged(s[cls == MEMCPY], e[cls == MEMCPY])
    as_, ae = merged(s, e)
    comp = int((ce - cs).sum())
    comm = int((me - ms).sum())
    mem = int((ye - ys).sum())
    exp_comm = comm - int(overlap_each(cs, ce, ms, me).sum())
    us, ue = merged(np.concatenate([cs, ms]), np.concatenate([ce, me]))
    exp_mem = mem - int(overlap_each(us, ue, ys, ye).sum())
    busy = int((ae - as_).sum())
    tot = int(ae[-1] - as_[0]) if total is None else int(total)
    return {"computation_time": comp, "exposed_comm_time": exp_comm, "exposed_memcpy_time": exp_mem,
            "busy_time": busy, "idle_time": tot - busy, "total_time": tot, "total_comm_time": comm,
            "total_memcpy_time": mem}


def timeline_rows(m: Dict[str, float], lead: list) -> List[list]:
    tot = m["total_time"]
    return [lead + [k, m[k] / 1e6, (m[k] / tot * 100.0) if tot else None] for k in TIMELINE_TYPES]


# ---- classification ---------------------------------------------------------------------------

class Ctx:
    """Per-table derived columns, computed once."""

    def __init__(self, t: Kernels, comm: str = "ruppell"):
        self.t = t
        self.comm_mode = comm
        is_comm = TRACELENS_COMM.search if comm == "tracelens" else is_collective
        names = t.names
        cat_names = categorize_many(names)
        if comm == "tracelens":
            # the category map follows the chosen classifier for collectives
            cat_names = ["comm" if TRACELENS_COMM.search(n) else ("other" if c == "comm" else c)
                         for n, c in zip(names, cat_names)]
        self.name_cat = np.array([CATEGORIES.index(c) for c in cat_names], np.int32) if names else np.zeros(0, np.int32)
        self.name_comm = np.array([bool(is_comm(n)) for n in names], bool)
        self.name_legacy_copy = np.array([bool(LEGACY_MEMCPY.match(n)) for n in names], bool)
        nm = t.name
        self.cat = self.name_cat[nm] if len(nm) else np.zeros(0, np.int32)
        k = t.kind
        memcpy = (k == KIND_MEMCPY) | ((k == KIND_KERNEL) & (self.name_legacy_copy[nm] if len(nm) else False))
        comm_ = ~memcpy & (self.name_comm[nm] if len(nm) else False) & (k != KIND_MEMSET)
        self.cls = np.where(memcpy, MEMCPY, np.where(comm_, COMM, COMP)).astype(np.int8)
        self.is_comm = self.cls == COMM
        # op-level category: an op name that decides it (aten GEMM, norm, ...), else the kernel's
        op_cat = np.array([CATEGORIES.index(c) if (c := op_category(o)) else -1 for o in t.ops], np.int32)
        oc = np.where(t.op >= 0, op_cat[np.clip(t.op, 0, None)] if len(op_cat) else -1, -1) if len(nm) else nm
        self.parent_cat = np.where(oc >= 0, oc, self.cat).astype(np.int32) if len(nm) else np.zeros(0, np.int32)

    def parent_op(self) -> Tuple[np.ndarray, List[str]]:
        """Parent cpu_op per row, as (codes, names): the op, or for a kernel
        with none, "graph" (a graph replay), "runtime" (an eager launch),
        "unlinked" (its launching call is not in the source), "" (the source
        records no launches: rocprofv3). Codes, not strings: a serving run has
        tens of millions of kernels."""
        t = self.t
        n = len(t.ops)
        names = list(t.ops) + ["graph", "runtime", "unlinked", ""]
        code = np.where(t.op >= 0, t.op, n + 3).astype(np.int64)
        none = t.op < 0
        graph = t.launch == LAUNCH_GRAPH
        code[none & ~graph & t.linked] = n + 1
        if t.info.get("launches_recorded", True):
            code[none & ~graph & ~t.linked] = n + 2
        code[none & graph] = n
        return code, names


def _rank_name(t: Kernels, i: int) -> str:
    return t.ranks[i]["label"]


# ---- gpu_timeline ------------------------------------------------------------------------------

def gpu_timeline(ctx: Ctx) -> Sheet:
    t = ctx.t
    rows, per = [], []
    for ri in range(len(t.ranks)):
        m = t.rank == ri
        if not m.any():
            continue
        tl = timeline(t.start[m], t.end[m], ctx.cls[m])
        per.append(tl)
        rows += timeline_rows(tl, [_rank_name(t, ri)])
    if len(per) > 1:
        mean = {k: sum(p[k] for p in per) / len(per) for k in TIMELINE_TYPES}
        rows += timeline_rows(mean, ["mean"])
    return Sheet("gpu_timeline", ["rank", "type", "time ms", "percent"], rows)


def _step_index(t: Kernels) -> np.ndarray:
    """Per kernel, its row in t.steps (same rank and step id), -1 none."""
    st = t.steps
    out = np.full(len(t), -1, np.int64)
    if not len(st["step"]) or not len(t):
        return out
    key_s = st["step"].astype(np.int64) * 4096 + st["rank"]
    o = np.argsort(key_s, kind="stable")
    ks = key_s[o]
    key_k = t.step.astype(np.int64) * 4096 + t.rank
    pos = np.clip(np.searchsorted(ks, key_k), 0, len(ks) - 1)
    hit = (ks[pos] == key_k) & (t.step >= 0)
    out[hit] = o[pos[hit]]
    return out


def step_groups(t: Kernels, phase_of_step: Optional[np.ndarray]) -> Tuple[np.ndarray, List[str]]:
    """Per step, its group label index: the step kind, or "phase / kind"."""
    labels: List[str] = []
    lab = {}
    out = np.zeros(len(t.steps["step"]), np.int64)
    for i, k in enumerate(t.steps["kind"].tolist()):
        name = t.step_kinds[k] if k >= 0 else "step"
        if phase_of_step is not None:
            name = f"{phase_of_step[i] or '(no phase)'} / {name}"
        j = lab.get(name)
        if j is None:
            j = lab[name] = len(labels)
            labels.append(name)
        out[i] = j
    return out, labels


def gpu_timeline_by_step_kind(ctx: Ctx, sidx: np.ndarray, sgroup: np.ndarray, glabels: List[str]) -> Optional[Sheet]:
    """The timeline inside steps, summed by step kind: each step's kernels
    clipped to its GPU period (begin marker to end marker), total = the
    periods' sum."""
    t = ctx.t
    st = t.steps
    if not len(st["step"]):
        return None
    have = sidx >= 0
    k_s = np.maximum(t.start[have], st["start"][sidx[have]])
    k_e = np.minimum(t.end[have], st["end"][sidx[have]])
    k_step = sidx[have]
    k_cls = ctx.cls[have]
    keep = k_e > k_s
    k_s, k_e, k_step, k_cls = k_s[keep], k_e[keep], k_step[keep], k_cls[keep]
    rows = []
    for ri in range(len(t.ranks)):
        for gi, label in enumerate(glabels):
            ssel = np.nonzero((st["rank"] == ri) & (sgroup == gi))[0]
            if not len(ssel):
                continue
            periods = st["end"][ssel] - st["start"][ssel]
            span = int(periods.max()) + 1
            pos = np.full(len(st["step"]), -1, np.int64)
            pos[ssel] = np.arange(len(ssel))
            m = pos[k_step] >= 0
            # steps laid end to end, each at its own offset, so they never interact
            off = pos[k_step[m]] * span - st["start"][k_step[m]]
            tl = timeline(k_s[m] + off, k_e[m] + off, k_cls[m], total=int(periods.sum()))
            if not m.any():
                tl["total_time"] = tl["idle_time"] = int(periods.sum())
            rows += timeline_rows(tl, [_rank_name(t, ri), label, len(ssel)])
    return Sheet("gpu_timeline_by_step_kind", ["rank", "step kind", "steps", "type", "time ms", "percent"], rows)


# ---- per-rank sheets -----------------------------------------------------------------------------

def _group_agg(keys: List[np.ndarray], vals: np.ndarray):
    """Rows grouped by the key columns: (first row index of each group, sum,
    count, min, max of vals per group, group id per row)."""
    o = np.lexsort(tuple(reversed(keys)))          # keys[0] is the primary key
    new = np.ones(len(o), bool)
    if len(o) > 1:
        new[1:] = False
        for k in keys:
            ks = k[o]
            new[1:] |= ks[1:] != ks[:-1]
    idx = np.nonzero(new)[0]
    v = vals[o]
    gid = np.empty(len(o), np.int64)
    gid[o] = np.cumsum(new) - 1
    return (o[idx], np.add.reduceat(v, idx), np.diff(np.append(idx, len(o))), np.minimum.reduceat(v, idx),
            np.maximum.reduceat(v, idx), gid)


def kernel_summary(ctx: Ctx, mask: np.ndarray, total_ns: int, parent: Tuple[np.ndarray, List[str]]) -> Sheet:
    t = ctx.t
    m = mask & (t.kind == KIND_KERNEL)
    idx = np.nonzero(m)[0]
    cols = ["Parent op category", "Parent cpu_op", "Kernel name", "Kernel stream", "Kernel duration (µs)_sum",
            "Kernel duration (µs)_count", "Kernel duration (µs)_mean", "Kernel duration (µs)_min",
            "Kernel duration (µs)_max", "Percent of kernels time (%)", "Percent of total time (%)"]
    if not len(idx):
        return Sheet("kernel_summary", cols, [])
    code, pnames = parent
    dur = (t.end[idx] - t.start[idx]).astype(np.float64) / 1000.0
    first, s, c, mn, mx, _ = _group_agg([ctx.parent_cat[idx], code[idx], t.name[idx], t.stream[idx]], dur)
    tot_k = float(dur.sum())
    rows = []
    for f, s_, c_, a, b in zip(first.tolist(), s.tolist(), c.tolist(), mn.tolist(), mx.tolist()):
        i = idx[f]
        rows.append([CATEGORIES[int(ctx.parent_cat[i])], pnames[int(code[i])], t.names[int(t.name[i])], int(t.stream[i]),
                     s_, int(c_), s_ / c_, a, b, (s_ / tot_k * 100) if tot_k else None,
                     (s_ / (total_ns / 1000.0) * 100) if total_ns else None])
    rows.sort(key=lambda r: -r[4])
    return Sheet("kernel_summary", cols, rows)


def _instances(ctx: Ctx, mask: np.ndarray, parent: Tuple[np.ndarray, List[str]]):
    """Op invocations among the rows in mask (collectives excluded, as
    TraceLens's kernel launchers exclude NCCL kernels): per invocation its
    name, category, direct kernel time (union of its kernels, ns), first row."""
    t = ctx.t
    idx = np.nonzero(mask & ~ctx.is_comm)[0]
    if not len(idx):
        return None
    gids, busy = grouped_busy(t.start[idx], t.end[idx], t.inst[idx])
    # the longest kernel of each invocation gives its category when its op does not
    dur = t.end[idx] - t.start[idx]
    o = np.lexsort((-dur, t.inst[idx]))
    first = np.ones(len(o), bool)
    first[1:] = t.inst[idx][o][1:] != t.inst[idx][o][:-1]
    rep = idx[o[first]]                       # one row per invocation, in inst order (= gids order)
    return {"inst": gids, "busy": busy, "row": rep, "code": parent[0][rep], "names": parent[1],
            "cat": ctx.parent_cat[rep]}


def ops_summary(ctx: Ctx, inst: Optional[dict]) -> Tuple[Sheet, Sheet]:
    c1 = ["name", "total_direct_kernel_time_sum", "total_direct_kernel_time_ms", "Count", "Categories",
          "Percentage (%)", "Cumulative Percentage (%)"]
    c2 = ["op category", "Count", "total_direct_kernel_time_ms", "Percentage (%)", "Cumulative Percentage (%)"]
    if inst is None:
        return Sheet("ops_summary", c1, []), Sheet("ops_summary_by_category", c2, [])
    uc, inv = np.unique(inst["code"], return_inverse=True)
    inv = inv.reshape(-1)
    tsum_c = np.bincount(inv, weights=inst["busy"], minlength=len(uc)) / 1000.0
    cnt_c = np.bincount(inv, minlength=len(uc))
    cat_c = np.zeros((len(uc), len(CATEGORIES)), bool)
    cat_c[inv, inst["cat"]] = True
    by: Dict[str, list] = {}          # by name (an op could be named like a placeholder)
    for j, c in enumerate(uc.tolist()):
        x = by.setdefault(inst["names"][c], [0.0, 0, set()])
        x[0] += float(tsum_c[j])
        x[1] += int(cnt_c[j])
        x[2] |= {CATEGORIES[k] for k in np.nonzero(cat_c[j])[0].tolist()}
    total = sum(x[0] for x in by.values())
    rows = []
    for name, (ts, n, cs_) in sorted(by.items(), key=lambda kv: -kv[1][0]):
        rows.append([name, ts, ts / 1000.0, n, sorted(cs_), ts / total * 100 if total else None, None])
    acc = 0.0
    for r in rows:
        acc += r[5] or 0.0
        r[6] = acc
    cs = np.bincount(inst["cat"], weights=inst["busy"], minlength=len(CATEGORIES)) / 1e6
    cc = np.bincount(inst["cat"], minlength=len(CATEGORIES))
    rows2 = []
    tot2 = float(cs.sum())
    for g in np.argsort(-cs, kind="stable").tolist():
        if cc[g]:
            rows2.append([CATEGORIES[g], int(cc[g]), float(cs[g]), float(cs[g]) / tot2 * 100 if tot2 else None, None])
    acc = 0.0
    for r in rows2:
        acc += r[3] or 0.0
        r[4] = acc
    return Sheet("ops_summary", c1, rows), Sheet("ops_summary_by_category", c2, rows2)


def category_summary(ctx: Ctx, mask: np.ndarray, total_ns: int) -> Sheet:
    """birdeye's own: GPU time by kernel category (the name map), every
    activity, collectives and copies included. The table that compares across
    vendors."""
    t = ctx.t
    idx = np.nonzero(mask)[0]
    cols = ["category", "Count", "Kernel duration (µs)_sum", "busy (µs)", "Percent of kernels time (%)",
            "Percent of total time (%)"]
    if not len(idx):
        return Sheet("category_summary", cols, [])
    cat = ctx.cat[idx]
    dur = (t.end[idx] - t.start[idx]).astype(np.float64)
    s = np.bincount(cat, weights=dur, minlength=len(CATEGORIES)) / 1000.0
    c = np.bincount(cat, minlength=len(CATEGORIES))
    g, busy = grouped_busy(t.start[idx], t.end[idx], cat.astype(np.int64))
    b = dict(zip(g.tolist(), (busy / 1000.0).tolist()))
    tot = float(s.sum())
    rows = [[CATEGORIES[j], int(c[j]), float(s[j]), b.get(j, 0.0), float(s[j]) / tot * 100 if tot else None,
             float(s[j]) / (total_ns / 1000.0) * 100 if total_ns else None]
            for j in np.argsort(-s, kind="stable").tolist() if c[j]]
    return Sheet("category_summary", cols, rows)


def short_kernels(ctx: Ctx, mask: np.ndarray, total_ns: int, parent: Tuple[np.ndarray, List[str]], short_us: float,
                  bins: int = 100) -> Tuple[Sheet, Sheet]:
    t = ctx.t
    m = mask & (t.kind == KIND_KERNEL)
    dur = (t.end - t.start).astype(np.float64) / 1000.0
    sm = m & (dur < short_us)
    idx = np.nonzero(sm)[0]
    hist = Sheet("short_kernel_histogram", ["bin_start", "bin_end", "count"], [])
    c2 = ["Parent cpu_op", "Kernel name", "Short Kernel duration (µs) sum", "Short Kernel count",
          "Short Kernel duration (µs) mean", "Short Kernel duration (µs) percent of total time"]
    if not len(idx):
        return hist, Sheet("short_kernels_summary", c2, [])
    counts, edges = np.histogram(dur[idx], bins=bins)
    hist.rows = [[float(a), float(b), int(c)] for a, b, c in zip(edges[:-1], edges[1:], counts)]
    code, pnames = parent
    first, s, c, _, _, _ = _group_agg([code[idx], t.name[idx]], dur[idx])
    rows = []
    for f, s_, c_ in zip(first.tolist(), s.tolist(), c.tolist()):
        i = idx[f]
        rows.append([pnames[int(code[i])], t.names[int(t.name[i])], s_, int(c_), s_ / c_,
                     s_ / (total_ns / 1000.0) * 100 if total_ns else None])
    rows.sort(key=lambda r: -r[2])
    return hist, Sheet("short_kernels_summary", c2, rows)


# ---- GEMM -------------------------------------------------------------------------------------

def _prod(xs) -> int:
    p = 1
    for x in xs:
        p *= int(x)
    return p


def gemm_shape(op: str, dims) -> Optional[Tuple[int, int, int, int]]:
    """(B, M, N, K) of a GEMM op from its input dims, or None."""
    try:
        if op in ("aten::mm", "aten::_scaled_mm"):
            a, b = dims[0], dims[1]
            if len(a) == 2 and len(b) == 2:
                return 1, int(a[0]), int(b[1]), int(a[1])
        elif op == "aten::addmm":
            a, b = dims[1], dims[2]
            if len(a) == 2 and len(b) == 2:
                return 1, int(a[0]), int(b[1]), int(a[1])
        elif op == "aten::bmm":
            a, b = dims[0], dims[1]
            if len(a) == 3 and len(b) == 3:
                return int(a[0]), int(a[1]), int(b[2]), int(a[2])
        elif op == "aten::baddbmm":
            a, b = dims[1], dims[2]
            if len(a) == 3 and len(b) == 3:
                return int(a[0]), int(a[1]), int(b[2]), int(a[2])
        elif op == "aten::linear":
            a, w = dims[0], dims[1]
            if len(a) >= 1 and len(w) == 2:
                return 1, _prod(a[:-1]), int(w[0]), int(a[-1])
        elif op == "aten::matmul":
            a, b = dims[0], dims[1]
            if len(a) >= 2 and len(b) == 2:
                return 1, _prod(a[:-1]), int(b[1]), int(a[-1])
            if len(a) >= 3 and len(b) >= 3:
                return _prod(a[:-2]), int(a[-2]), int(b[-1]), int(a[-1])
    except (TypeError, ValueError, IndexError):
        return None
    return None


def _operand_types(op: str, types) -> Tuple[Optional[str], Optional[str]]:
    if not isinstance(types, (list, tuple)):
        return None, None
    i = 1 if op in ("aten::addmm", "aten::baddbmm") else 0
    a = types[i] if len(types) > i else None
    b = types[i + 1] if len(types) > i + 1 else None
    return a, b


def gemm(ctx: Ctx, inst: Optional[dict]) -> Tuple[Optional[Sheet], Optional[str]]:
    t = ctx.t
    cols = ["name", "param: B", "param: M", "param: N", "param: K", "param: dtype_A_B", "GFLOPS_first",
            "Data Moved (MB)_first", "FLOPS/Byte_first", "TFLOPS/s_mean", "TB/s_mean", "Kernel Time (µs)_sum",
            "Kernel Time (µs)_mean", "name_count"]
    if not t.info.get("op_tree") or not t.sigs:
        return None, ("no aten op shapes in this source: the GEMM sheet needs the CPU op tree "
                      "(RUPPELL_CPU_OPS=1) or a PyTorch profiler trace recorded with shapes (limitation gemm-needs-shapes)")
    if inst is None:
        return None, "no kernels"
    groups: Dict[tuple, list] = {}
    for row, busy in zip(inst["row"].tolist(), inst["busy"].tolist()):
        o, sg = int(t.op[row]), int(t.sig[row])
        if o < 0 or sg < 0:
            continue
        name = t.ops[o]
        if name not in GEMM_OPS:
            continue
        dims, types = t.sigs[sg]
        shp = gemm_shape(name, dims)
        if shp is None or busy <= 0:
            continue
        ta, tb = _operand_types(name, types)
        groups.setdefault((name,) + shp + (repr((ta, tb)),), []).append((busy, ta, tb))
    if not groups:
        return None, "no aten GEMM op (mm, addmm, bmm, baddbmm, matmul, linear, _scaled_mm) with input dims ran kernels"
    rows = []
    for (name, B, M, N, K, dt), lst in groups.items():
        ta, tb = lst[0][1], lst[0][2]
        ba = DTYPE_BYTES.get(str(ta).lower())
        bb = DTYPE_BYTES.get(str(tb).lower())
        flops = 2.0 * B * M * N * K
        nbytes = (M * K * ba + K * N * bb + M * N * ba) * B if ba and bb else None
        times = np.array([x[0] for x in lst], np.float64)        # ns
        rows.append([name, B, M, N, K, dt, flops / 1e9, nbytes / (1 << 20) if nbytes else None,
                     flops / nbytes if nbytes else None, float(np.mean(flops / times / 1e3)),
                     float(np.mean(nbytes / times / 1e3)) if nbytes else None, float(times.sum() / 1e3),
                     float(times.mean() / 1e3), len(lst)])
    rows.sort(key=lambda r: -r[11])
    return Sheet("GEMM", cols, rows), None


# ---- collectives -------------------------------------------------------------------------------

def align_collectives(ctx: Ctx) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Each collective kernel's launch aligned across ranks: same step, same position among the
    step's collectives, the same sequence of collective names on every rank. Per row: its
    launch's transfer (the launch's shortest duration across ranks; -1 when not aligned) and its
    wait (its duration less that; -1). On the first rank's row of each aligned launch, ``first``
    and the launch's start skew across ranks."""
    t = ctx.t
    n = len(t)
    transfer, wait = np.full(n, -1, np.int64), np.full(n, -1, np.int64)
    first, skew = np.zeros(n, bool), np.full(n, -1, np.int64)
    idx = np.nonzero(ctx.is_comm)[0]
    nr = len(t.ranks)
    if nr < 2 or not len(idx):
        return transfer, wait, first, skew
    short = t.short()
    per_rank = []
    for ri in range(nr):
        sel = idx[t.rank[idx] == ri]
        sel = sel[np.lexsort((t.start[sel], t.step[sel]))]
        d: Dict[int, tuple] = {}
        if len(sel):
            cut = np.nonzero(np.diff(t.step[sel]))[0] + 1
            for part in np.split(np.arange(len(sel)), cut):
                rows = sel[part]
                d[int(t.step[rows[0]])] = (tuple(short[int(x)] for x in t.name[rows]), rows)
        per_rank.append(d)
    common = set.intersection(*[set(d) for d in per_rank]) if per_rank else set()
    for s in sorted(common):
        seqs = [d[s][0] for d in per_rank]
        if any(x != seqs[0] for x in seqs):
            continue
        rows = np.stack([d[s][1] for d in per_rank])        # ranks x launches
        durs = t.end[rows] - t.start[rows]
        mn = durs.min(axis=0)
        transfer[rows] = np.broadcast_to(mn, rows.shape)
        wait[rows] = durs - mn
        first[rows[0]] = True
        st = t.start[rows]
        skew[rows[0]] = st.max(axis=0) - st.min(axis=0)
    return transfer, wait, first, skew


def coll_analysis(ctx: Ctx, al: Optional[tuple] = None) -> Optional[Sheet]:
    """Per collective kernel: launches and time over every rank; with several
    ranks, each launch aligned across ranks (align_collectives) and split into
    transfer (its shortest duration across ranks) and waiting (the rest, per
    rank: untrusted when traced, limitation collective-wait); start skew across
    ranks when they share a clock."""
    t = ctx.t
    m = ctx.is_comm
    if not m.any():
        return None
    short = t.short()
    nr = len(t.ranks)
    cols = ["Collective", "Ranks", "Count", "Duration (µs)_sum", "Duration (µs)_mean", "Duration (µs)_min",
            "Duration (µs)_max", "Aligned launches", "Transfer (µs)_sum", "Wait (µs)_sum_mean_rank", "Wait (%)",
            "Start skew (µs)_mean", "Start skew (µs)_max"]
    transfer, wait, first, skew = al if al is not None else align_collectives(ctx)
    idx = np.nonzero(m)[0]
    stats: Dict[str, list] = {}
    align: Dict[str, list] = {}       # name -> [aligned, transfer ns, wait ns (all ranks), skew list]
    for i, nm, r, d in zip(idx.tolist(), (short[int(x)] for x in t.name[idx]), t.rank[idx].tolist(),
                           (t.end[idx] - t.start[idx]).tolist()):
        stats.setdefault(nm, [set(), []])
        stats[nm][0].add(r)
        stats[nm][1].append(d)
        if wait[i] >= 0:
            a = align.setdefault(nm, [0, 0, 0, []])
            a[2] += int(wait[i])
            if first[i]:
                a[0] += 1
                a[1] += int(transfer[i])
                a[3].append(int(skew[i]))
    same_clock = bool(t.info.get("same_clock")) and not any("several hosts" in n for n in t.info.get("notes", []))
    rows = []
    for nm, (ranks, ds) in sorted(stats.items(), key=lambda kv: -sum(kv[1][1])):
        ds = np.array(ds, np.float64) / 1000.0
        a = align.get(nm)
        row = [nm, len(ranks), len(ds), float(ds.sum()), float(ds.mean()), float(ds.min()), float(ds.max())]
        if a:
            wait_mean = a[2] / nr / 1000.0
            row += [a[0], a[1] / 1000.0, wait_mean, wait_mean / (a[1] / 1000.0 + wait_mean) * 100
                    if (a[1] or a[2]) else None]
            row += ([float(np.mean(a[3])) / 1000.0, float(np.max(a[3])) / 1000.0] if same_clock else [None, None])
        else:
            row += [0 if nr > 1 else None, None, None, None, None, None]
        rows.append(row)
    note = None
    if nr > 1 and not same_clock:
        note = "start skew not computed: the ranks do not share a time base"
    return Sheet("coll_analysis", cols, rows, note)


def _bus(op: Optional[str], nbytes: float, n: int) -> Tuple[float, float]:
    """nccl-tests' conventions: the size a collective is measured by, and its bus-bandwidth
    factor, given the bytes of the call's input on one rank and the group size. All-reduce: the
    buffer, 2(n-1)/n. All-gather: the gathered output (the input times n), (n-1)/n.
    Reduce-scatter and all-to-all: the input, (n-1)/n. Broadcast, send and recv: the buffer, 1."""
    if n < 2:
        return nbytes, 1.0
    if op == "all_reduce":
        return nbytes, 2 * (n - 1) / n
    if op == "all_gather":
        return nbytes * n, (n - 1) / n
    if op in ("reduce_scatter", "all_to_all", "all_to_all_dispatch", "all_to_all_combine"):
        return nbytes, (n - 1) / n
    return nbytes, 1.0


def coll_by_op(ctx: Ctx, al: tuple) -> Optional[Sheet]:
    """Per collective call kind (an adapter's records, roadmap-v2 M8: operation, backend, dtype,
    elements, bytes, group size), over every rank: its collective kernels, how many were replayed
    from graphs, and with several ranks the transfer estimate (align_collectives) and the
    algorithmic and bus bandwidth it gives (nccl-tests' conventions, _bus). None when no
    collective kernel carries a call."""
    t = ctx.t
    m = ctx.is_comm & (t.coll >= 0)
    if not m.any():
        return None
    transfer, wait, first, _ = al
    nr = len(t.ranks)
    cols = ["Operation", "Backend", "Dtype", "Elements", "Bytes", "Group size", "Ranks", "Count", "Graph-replayed",
            "Duration (µs)_sum", "Duration (µs)_mean", "Aligned launches", "Transfer (µs)_mean",
            "Algorithmic bandwidth (GB/s)", "Bus bandwidth (GB/s)", "Wait (%)", "NCCL algorithm", "NCCL protocol",
            "NCCL channels"]
    idx = np.nonzero(m)[0]
    idx = idx[np.argsort(t.coll[idx], kind="stable")]
    cut = np.nonzero(np.diff(t.coll[idx]))[0] + 1
    rows = []
    for g in np.split(idx, cut):
        c = t.colls[int(t.coll[g[0]])]
        dur = (t.end[g] - t.start[g]).astype(np.float64)
        fr = g[first[g]]
        world = c.get("world") if isinstance(c.get("world"), int) and c.get("world") > 0 else nr
        tmean = float(transfer[fr].mean()) if len(fr) else None
        nbytes = c.get("bytes")
        algbw = busbw = None
        if tmean and isinstance(nbytes, (int, float)) and nbytes > 0:
            size, factor = _bus(c.get("op"), float(nbytes), int(world))
            algbw = size / tmean          # bytes per ns = GB/s
            busbw = algbw * factor
        w = wait[g][wait[g] >= 0]
        tsum = float(transfer[fr].sum()) if len(fr) else 0.0
        wmean = float(w.sum()) / nr if len(w) else 0.0
        rows.append([c.get("op"), c.get("backend"), c.get("dtype"), c.get("numel"), nbytes, world,
                     len(np.unique(t.rank[g])), len(g), int((t.launch[g] == LAUNCH_GRAPH).sum()),
                     float(dur.sum()) / 1000.0, float(dur.mean()) / 1000.0, len(fr),
                     tmean / 1000.0 if tmean is not None else None, algbw, busbw,
                     wmean / (tsum + wmean) * 100 if len(fr) and (tsum or wmean) else None,
                     c.get("algo"), c.get("proto"), c.get("channels")])
    rows.sort(key=lambda r: -r[9])
    note = ("bandwidth from the transfer estimate: each aligned launch's shortest duration across ranks "
            "(nccl-tests' size and bus factor per operation)" if nr > 1 else
            "one rank: no transfer estimate, so no bandwidth")
    return Sheet("coll_by_op", cols, rows, note)


# ---- steps -------------------------------------------------------------------------------------

def steps_sheet(ctx: Ctx, sidx: np.ndarray, sgroup: np.ndarray, glabels: List[str]) -> Optional[Sheet]:
    """ruppell's own (TraceLens has none): per rank and step kind, the steps'
    GPU periods (begin marker to end marker), busy time inside the period,
    kernels and kernel time per step, graph and eager launches."""
    t = ctx.t
    st = t.steps
    ns = len(st["step"])
    if not ns:
        return None
    have = sidx >= 0
    k_step = sidx[have]
    k_s = np.maximum(t.start[have], st["start"][k_step])
    k_e = np.minimum(t.end[have], st["end"][k_step])
    keep = k_e > k_s
    g, busy = grouped_busy(k_s[keep], k_e[keep], k_step[keep])
    busy_of = np.zeros(ns, np.int64)
    busy_of[g] = busy
    kern = (t.kind[have] == KIND_KERNEL).astype(np.float64)
    dur = (t.end[have] - t.start[have]).astype(np.float64) * kern
    kcount = np.bincount(k_step, weights=kern, minlength=ns)
    ktime = np.bincount(k_step, weights=dur, minlength=ns)
    graph = np.bincount(k_step, weights=kern * (t.launch[have] == LAUNCH_GRAPH), minlength=ns)
    eager = np.bincount(k_step, weights=kern * (t.launch[have] == LAUNCH_EAGER), minlength=ns)
    period = (st["end"] - st["start"]).astype(np.float64)
    cols = ["rank", "step kind", "steps", "steps with kernels", "period (µs)_mean", "period (µs)_median",
            "period (µs)_p90", "busy (µs)_mean", "busy/period", "kernels per step_mean",
            "kernel time (µs) per step_mean", "graph kernels per step_mean", "eager kernels per step_mean"]
    rows = []
    for ri in range(len(t.ranks)):
        for gi, label in enumerate(glabels):
            s = np.nonzero((st["rank"] == ri) & (sgroup == gi))[0]
            if not len(s):
                continue
            p = period[s] / 1000.0
            rows.append([_rank_name(t, ri), label, len(s), int((kcount[s] > 0).sum()), float(p.mean()),
                         float(np.median(p)), float(np.percentile(p, 90)), float(busy_of[s].mean() / 1000.0),
                         float(busy_of[s].sum() / period[s].sum()) if period[s].sum() else None,
                         float(kcount[s].mean()), float(ktime[s].mean() / 1000.0), float(graph[s].mean()),
                         float(eager[s].mean())])
    return Sheet("steps", cols, rows)


# ---- everything --------------------------------------------------------------------------------

def pick_rank(t: Kernels, rank: Optional[int]) -> int:
    present = sorted(set(np.unique(t.rank).tolist()))
    if not present:
        raise ValueError("no kernels")
    if rank is None:
        known = [i for i in present if t.ranks[i].get("rank") is not None]
        return min(known, key=lambda i: t.ranks[i]["rank"]) if known else present[0]
    for i in present:
        if t.ranks[i].get("rank") == rank:
            return i
    if 0 <= rank < len(t.ranks) and rank in present:
        return rank
    raise ValueError(f"no rank {rank}: ranks are {[t.ranks[i]['label'] for i in present]}")


def phase_of(t: Kernels, times: np.ndarray, ranks: np.ndarray, phases: Dict[str, Tuple[float, float]]) -> List[Optional[str]]:
    """Phase name per time (ns, the source's base): realtime seconds through
    a ruppell process's clock pairs, else seconds from the first activity of
    the source."""
    out: List[Optional[str]] = []
    t0 = t.info.get("t0_ns", 0)
    for x, r in zip(times.tolist(), ranks.tolist()):
        off = t.ranks[r].get("clock_offset_ns")
        sec = (x - off) / 1e9 if off is not None else (x - t0) / 1e9
        out.append(next((k for k, (lo, hi) in phases.items() if lo <= sec < hi), None))
    return out


def analyze(t: Kernels, rank: Optional[int] = None, short_us: float = 10.0, comm: str = "ruppell",
            phases: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[str, object]:
    """Every table. Per-rank sheets (TraceLens reads one rank at a time) are
    for the chosen rank; gpu_timeline, steps and coll_analysis cover every rank."""
    ctx = Ctx(t, comm=comm)
    ri = pick_rank(t, rank)
    parent = ctx.parent_op()
    sheets: Dict[str, Sheet] = {}
    skipped: Dict[str, str] = {}
    sheets["gpu_timeline"] = gpu_timeline(ctx)
    mine = t.rank == ri
    total_ns = timeline(t.start[mine], t.end[mine], ctx.cls[mine])["total_time"]
    sidx = _step_index(t)
    pstep = phase_of(t, t.steps["start"], t.steps["rank"], phases) if phases else None
    sgroup, glabels = step_groups(t, pstep)
    tl_steps = gpu_timeline_by_step_kind(ctx, sidx, sgroup, glabels)
    if tl_steps is not None:
        sheets["gpu_timeline_by_step_kind"] = tl_steps
    else:
        skipped["gpu_timeline_by_step_kind"] = "the source has no steps"
    sheets["kernel_summary"] = kernel_summary(ctx, mine, total_ns, parent)
    sheets["category_summary"] = category_summary(ctx, mine, total_ns)
    has_ops = bool((t.op[mine] >= 0).any())
    inst = _instances(ctx, mine, parent)
    if has_ops:
        sheets["ops_summary"], sheets["ops_summary_by_category"] = ops_summary(ctx, inst)
    else:
        skipped["ops_summary"] = skipped["ops_summary_by_category"] = (
            "no op attribution in this source (kernels only): see category_summary")
    hist, short = short_kernels(ctx, mine, total_ns, parent, short_us)
    sheets["short_kernel_histogram"], sheets["short_kernels_summary"] = hist, short
    al = align_collectives(ctx)
    cs = coll_analysis(ctx, al)
    if cs is not None:
        sheets["coll_analysis"] = cs
    else:
        skipped["coll_analysis"] = "no collective kernels"
    cb = coll_by_op(ctx, al)
    if cb is not None:
        sheets["coll_by_op"] = cb
    else:
        skipped["coll_by_op"] = "no collective kernel carries a call (an adapter's collective records, M8)"
    g, why = gemm(ctx, inst)
    if g is not None:
        sheets["GEMM"] = g
    else:
        skipped["GEMM"] = why
    st = steps_sheet(ctx, sidx, sgroup, glabels)
    if st is not None:
        sheets["steps"] = st
    else:
        skipped["steps"] = "the source has no steps"
    kern = mine & (t.kind == KIND_KERNEL)
    dur = t.end - t.start
    summary = {
        "rank_index": ri, "rank": t.ranks[ri]["label"], "ranks": [r["label"] for r in t.ranks],
        "comm": comm, "short_us": short_us,
        "kernels": int(kern.sum()), "activities": int(mine.sum()),
        "kernel_time_ns": int(dur[kern].sum()), "activity_time_ns": int(dur[mine].sum()),
        "total_time_ns": int(total_ns),
        "with_op": int((t.op[mine] >= 0).sum()), "with_op_time_ns": int(dur[mine & (t.op >= 0)].sum()),
        "with_scope": int((t.scope[mine] >= 0).sum()), "with_scope_time_ns": int(dur[mine & (t.scope >= 0)].sum()),
        "unlinked": int((mine & ~t.linked).sum()),
        "outside_steps": int((mine & (t.step < 0)).sum()), "steps": int((t.steps["rank"] == ri).sum()),
        "short_kernels": int((kern & (dur < short_us * 1000)).sum()),
        "short_kernel_time_ns": int(dur[kern & (dur < short_us * 1000)].sum()),
    }
    unattr = mine & (t.op < 0)
    if unattr.any():
        short_names = t.short()
        un, inv = np.unique(t.name[unattr], return_inverse=True)
        tt = np.bincount(inv.reshape(-1), weights=dur[unattr])
        cc = np.bincount(inv.reshape(-1))
        summary["without_op_by_kernel"] = [{"kernel": short_names[int(un[j])], "count": int(cc[j]),
                                            "time_ns": int(tt[j])} for j in np.argsort(-tt)[:15].tolist()]
    return {"sheets": sheets, "skipped": skipped, "summary": summary}
