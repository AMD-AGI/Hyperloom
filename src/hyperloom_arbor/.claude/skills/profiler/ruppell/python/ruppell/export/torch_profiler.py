"""A ruppell run in the PyTorch profiler's trace vocabulary, for tools that
read only that (AMD TraceLens).

Such tools build their op tree from `cpu_op` events on a host thread, the
`cuda_runtime` calls nested inside them, and the kernels those calls
launched (linked by `correlation` and `ac2g` flows). A PyTorch profiler trace
of a graph replay gives them one `cudaGraphLaunch` with every replayed kernel
under it. Here each kernel's ruppell scope becomes the op that launched it:

- eager kernels: the real launching call, with its real CPU interval, is
  wrapped in a `cpu_op` named after the scope's op (module class, or the op
  given to `ruppell.scope`), with `Input Dims` from the scope's shape;
- replayed kernels: the real `cudaGraphLaunch` interval is divided among the
  replay's kernels in node order, one `cpu_op` and one launch per kernel.
  These CPU intervals are synthetic (the replay launched them in one call);
  their GPU times and the attribution are ruppell's.

Kernels with no scope get a `cpu_op` named `unattributed`; kernels whose
launching call was not recorded are left unlinked. The step markers are not
exported; engine steps become `gpu_user_annotation` ranges. One process per
file (these tools read one rank at a time); `write_ranks` writes every rank,
from one analysis, as `ruppell tracelens` does.

Op names (without the CPU op tree, which names ops by aten op): "op", the
scope's op (the module class under the vLLM adapter), or "path", the scope's
module path with layer indices folded (`model.layers.*.attn.o_proj`), which
keeps apart modules of one class and still sums over layers.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from ruppell.engine import HOW_NAMES, OP_ATEN, OP_SCOPE, Analysis, analyze_run, short_name
from ruppell.export.chrome import _window
from ruppell.limits import for_run, is_collective, markdown


LAYER = re.compile(r"\.\d+(?=\.|$)")


def _op_name(a: Analysis, sid: int, op_names: str = "op") -> Tuple[str, Optional[str], list]:
    info = a.scope_info(sid) if sid > 0 else None
    if not info:
        return "unattributed", None, []
    shape = info.get("shape")
    dims = [list(shape)] if isinstance(shape, (list, tuple)) else []
    if op_names == "path":
        return LAYER.sub(".*", info["name"]), info["name"], dims
    return str(info.get("op") or info["name"]), info["name"], dims


def _kernel_analyses(run_dir: str) -> List[Analysis]:
    analyses = [a for a in analyze_run(run_dir)["analyses"] if len(a.proc.d)]
    if not analyses:
        raise ValueError(f"{run_dir}: no process ran kernels")
    return analyses


def build(run_dir: str, process: int = 0, steps: Optional[Tuple[int, int]] = None, op_names: str = "op") -> dict:
    """``process``: the index among processes that ran kernels (ranks, in
    directory order)."""
    return _build(_kernel_analyses(run_dir)[process], run_dir, steps, op_names)


def _t0(a: Analysis) -> int:
    p = a.proc
    t = [int(p.d["start"].min())] if len(p.d) else []
    if len(p.api):
        t.append(int(p.api["start"].min()))
    if len(p.ops):
        t.append(int(p.ops["t0"].min()))
    return min(t) if t else 0


def _build(a: Analysis, run_dir: str, steps: Optional[Tuple[int, int]], op_names: str,
           only: Optional[List[int]] = None, t_zero: Optional[int] = None) -> dict:
    """``only``: export these step ids alone (kernels, the CPU ops inside their
    host windows, their annotations), not everything between ``steps``.
    ``t_zero``: the time every rank's file counts from (ranks on one node share
    the profiler's clock; a per-file zero shifts ranks against each other, and
    TraceLens's collective report then reads the shift as skew)."""
    p, d = a.proc, a.proc.d
    w = _window(a, steps)
    keep = ~a.is_marker & ~a.untimed   # counter steps' kernels were serialized (gray-parrot)
    if w is not None:
        keep &= (d["start"] >= w[0]) & (d["start"] <= w[1])
    if only is not None:
        keep &= np.isin(a.step, np.asarray(sorted(only), dtype=a.step.dtype))
        sel = [s for s in a.step_list if s["step"] in set(only)]
        cpu_iv = sorted((int(s.get("cpu_begin") or s["start"]), int(s.get("cpu_end") or s["end"])) for s in sel)
    idx = np.nonzero(keep)[0]
    api = p.api
    ao = np.argsort(api["corr"], kind="stable")
    ac = api["corr"][ao]
    t0 = int(d["start"][idx].min()) if len(idx) else 0
    t0 = min(t0, int(api["start"].min())) if len(api) else t0
    t0 = min(t0, int(p.ops["t0"].min())) if len(p.ops) else t0
    if t_zero is not None:
        t0 = t_zero

    def us(t) -> float:
        return round((int(t) - t0) / 1000.0, 4)

    # with the CPU op tree (RUPPELL_CPU_OPS), the real ops and scopes are the host
    # side: eager launches keep their real calls inside them, and replayed kernels
    # take the op that was running when their graph was captured
    op_mode = len(p.ops) > 0
    host_pid = int(p.meta.get("pid", 1)) or 1
    ev: List[dict] = [{"ph": "M", "name": "process_name", "pid": host_pid, "args": {"name": f"{p.label} (host)"}}]
    names: Dict[int, str] = {}
    corr_next = 1
    groups: Dict[int, List[int]] = {}
    for i in idx.tolist():
        groups.setdefault(int(d["corr"][i]), []).append(i)
    unlinked = 0
    cpu_lo, cpu_hi = 1 << 64, 0          # the CPU span of the launches exported
    for corr, members in groups.items():
        j = int(np.searchsorted(ac, corr))
        rec = api[ao[j]] if j < len(ac) and ac[j] == corr else None
        if rec is None:
            unlinked += len(members)
        replayed = bool(a.is_replayed[members[0]]) if hasattr(a, "is_replayed") else False
        if replayed:
            members.sort(key=lambda i: (int(a.node[i]), int(d["start"][i])))
            units = [[i] for i in members]         # one op per replayed kernel
        else:
            units = [members]                      # one op per eager call
        if rec is not None:
            s, e, tid = int(rec["start"]), int(rec["end"]), int(rec["thread_id"])
            cpu_lo, cpu_hi = min(cpu_lo, s), max(cpu_hi, e)
            call = p.api_ops.get(int(rec["op"]), "launch")
        for n, unit in enumerate(units):
            i0 = unit[0]
            op, scope, dims = _op_name(a, int(a.scope[i0]), op_names)
            types = ["" for _ in dims]
            info = a.op_info(int(a.cpu_op[i0])) if op_mode else None
            if info:
                op, dims, types = info["name"], info["dims"] or [], info["types"] or []
            c = corr_next
            corr_next += 1
            for i in unit:
                nm = a.kernel_name.get(int(d["kernel_id"][i]), str(int(d["kernel_id"][i])))
                dev = int(d["agent_id"][i])
                names.setdefault(dev, f"GPU {dev}")
                args = {"External id": c, "correlation": c, "device": dev, "stream": int(d["queue_id"][i]),
                        "grid": [int(d["gx"][i]), int(d["gy"][i]), int(d["gz"][i])],
                        "block": [int(d["wx"][i]), int(d["wy"][i]), int(d["wz"][i])],
                        "ruppell_scope": scope, "ruppell_attribution": HOW_NAMES[int(a.how[i])],
                        "ruppell_step": int(a.step[i])}
                if is_collective(nm):
                    args["ruppell_limits"] = ["collective-wait"]
                if a.graph[i] >= 0:
                    args["ruppell_graph"] = a.graphs[int(a.graph[i])]["key"]
                ev.append({"ph": "X", "cat": "kernel", "name": nm, "pid": dev, "tid": int(d["queue_id"][i]),
                           "ts": us(d["start"][i]), "dur": round(int(d["end"][i] - d["start"][i]) / 1000.0, 4), "args": args})
                if rec is not None and i == unit[0]:
                    ev.append({"ph": "f", "bp": "e", "cat": "ac2g", "name": "ac2g", "id": c, "pid": dev,
                               "tid": int(d["queue_id"][i]), "ts": us(d["start"][i])})
            if rec is None:
                continue
            if op_mode and not replayed:
                # the real call, inside the real op the tree recorded
                ev.append({"ph": "X", "cat": "cuda_runtime", "name": call, "pid": host_pid, "tid": tid, "ts": us(s),
                           "dur": round((e - s) / 1000.0, 4), "args": {"External id": c, "correlation": c}})
                ev.append({"ph": "s", "cat": "ac2g", "name": "ac2g", "id": c, "pid": host_pid, "tid": tid, "ts": us(s)})
                continue
            # the call's interval, shared out among its units in order; each op
            # ends 2 ns before the next begins, so no float rounding of ts + dur
            # makes a reader nest one inside the other (TraceLens then drops it)
            span = max(e - s, len(units) * 8)
            a_ = s + span * n // len(units)
            b_ = s + span * (n + 1) // len(units)
            op_args = {"External id": c, "Input Dims": dims, "Input type": types, "ruppell_scope": scope}
            ev.append({"ph": "X", "cat": "cpu_op", "name": op, "pid": host_pid, "tid": tid,
                       "ts": us(a_), "dur": round((b_ - a_ - 2) / 1000.0, 4), "args": op_args})
            ev.append({"ph": "X", "cat": "cuda_runtime", "name": call + (f" [node {int(a.node[i0])}]" if replayed else ""),
                       "pid": host_pid, "tid": tid, "ts": us(a_ + 1), "dur": round((b_ - a_ - 4) / 1000.0, 4),
                       "args": {"External id": c, "correlation": c}})
            ev.append({"ph": "s", "cat": "ac2g", "name": "ac2g", "id": c, "pid": host_pid, "tid": tid, "ts": us(a_ + 1)})
    if op_mode and cpu_hi:
        o = p.ops
        # aten ops and scopes only (a RUPPELL_OP_SEQ record is no interval)
        om = (o["t1"] >= cpu_lo) & (o["t0"] <= cpu_hi) & np.isin(o["kind"], (OP_ATEN, OP_SCOPE))
        if only is not None:
            # only the ops inside the selected steps' host windows
            starts = np.array([x for x, _ in cpu_iv], dtype=np.int64)
            ends = np.array([y for _, y in cpu_iv], dtype=np.int64)
            k = np.searchsorted(starts, o["t0"].astype(np.int64), side="right") - 1
            om &= (k >= 0) & (o["t0"].astype(np.int64) <= ends[np.clip(k, 0, None)])
        for j in np.nonzero(om)[0].tolist():
            r = o[j]
            if int(r["kind"]) == OP_ATEN:
                sig = p.op_sigs.get(int(r["id"]), {})
                ev.append({"ph": "X", "cat": "cpu_op", "name": sig.get("name", "op"), "pid": host_pid, "tid": int(r["thread_id"]),
                           "ts": us(r["t0"]), "dur": round(int(r["t1"] - r["t0"]) / 1000.0, 4),
                           "args": {"Input Dims": sig.get("dims") or [], "Input type": sig.get("types") or []}})
            else:
                ev.append({"ph": "X", "cat": "python_function", "name": f"nn.Module: {a.scope_name(int(r['id'])) or r['id']}",
                           "pid": host_pid, "tid": int(r["thread_id"]), "ts": us(r["t0"]),
                           "dur": round(int(r["t1"] - r["t0"]) / 1000.0, 4), "args": {}})
    for dev, label in names.items():
        ev.append({"ph": "M", "name": "process_name", "pid": dev, "args": {"name": label}})
    if names:  # engine steps, as the ranges a PyTorch trace's user annotations are
        dev0 = min(names)
        for s in a.step_list:
            if w is not None and (s["start"] < w[0] or s["end"] > w[1]):
                continue
            if only is not None and s["step"] not in set(only):
                continue
            f = s["fields"]
            nm = f"step {s['step']} {f.get('kind', f.get('phase', ''))} r{f.get('running_requests', '')} rows{f.get('rows', '')}"
            ev.append({"ph": "X", "cat": "gpu_user_annotation", "name": nm.strip(), "pid": dev0, "tid": 0,
                       "ts": us(s["start"]), "dur": round(int(s["end"] - s["start"]) / 1000.0, 4), "args": dict(f)})
    return {"traceEvents": ev, "displayTimeUnit": "ms",
            "otherData": {"ruppell": {"process": p.label, "run": run_dir, "window_steps": steps,
                                      "steps_exported": sorted(only) if only is not None else None, "kernels": int(len(idx)),
                                      "unlinked_kernels": unlinked,
                                      "cpu_op_tree": op_mode, "op_names": "aten" if op_mode else op_names,
                                      "limitations": for_run(p.meta, export="torch", cpu_ops=op_mode, copies=p.copies_recorded),
                                      "note": "PyTorch-profiler form of a ruppell run; CPU intervals of replayed kernels are synthetic"}}}


def export(run_dir: str, out: str, process: int = 0, steps: Optional[Tuple[int, int]] = None) -> dict:
    trace = build(run_dir, process=process, steps=steps)
    with open(out, "w") as f:
        json.dump(trace, f)
    return trace["otherData"]["ruppell"]


def pick_steps(a: Analysis, steps: Optional[Tuple[int, int]], kind: str, max_steps: int) -> List[int]:
    """Up to ``max_steps`` traced engine steps of ``kind`` within ``steps``,
    evenly spaced over them (every rank has the same step ids and kinds)."""
    ids = [s["step"] for s in a.step_list
           if s["fields"].get("phase") == "engine" and s["fields"].get("kind") == kind
           and (steps is None or steps[0] <= s["step"] <= steps[1]) and s["step"] not in a.untimed_step_ids]
    ids.sort()
    if max_steps and len(ids) > max_steps:
        ids = [ids[i * len(ids) // max_steps] for i in range(max_steps)]
    return ids


def write_ranks(run_dir: str, out_dir: str, steps: Optional[Tuple[int, int]] = None, op_names: str = "path",
                kind: Optional[str] = None, max_steps: int = 0) -> List[str]:
    """Every process that ran kernels, one file each (rank<N>.json), from one
    analysis of the run: for TraceLens's per-rank and collective reports.
    With ``kind``: only up to ``max_steps`` traced steps of that kind (the
    same steps on every rank), for a file TraceLens can read in minutes."""
    os.makedirs(out_dir, exist_ok=True)
    paths: List[str] = []
    analyses = _kernel_analyses(run_dir)
    only = pick_steps(analyses[0], steps, kind, max_steps) if kind else None
    t_zero = min(_t0(a) for a in analyses)  # one zero for every rank's file
    if only is not None and not only:
        raise ValueError(f"no traced {kind} steps in {steps}")
    with open(os.path.join(out_dir, "LIMITATIONS.md"), "w") as f:
        f.write(markdown(for_run(analyses[0].proc.meta, export="torch", cpu_ops=any(len(a.proc.ops) for a in analyses),
                                 copies=all(a.proc.copies_recorded for a in analyses))))
    for i, a in enumerate(analyses):
        rank = a.proc.context.get("rank")
        stem = f"rank{rank}" if rank is not None else f"process{i}"
        path = os.path.join(out_dir, stem + ".json")
        if path in paths:
            path = os.path.join(out_dir, f"{stem}_p{i}.json")
        with open(path, "w") as f:
            json.dump(_build(a, run_dir, steps, op_names, only, t_zero), f)
        paths.append(path)
    return paths
