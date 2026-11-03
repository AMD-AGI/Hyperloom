"""Collective calls in vLLM (roadmap-v2 M8): which call launched each collective kernel.

vLLM picks a backend for each collective call: its own custom all-reduce, AITER's, quick
all-reduce (ROCm), PyTorch's symmetric memory, FlashInfer's, its NCCL wrapper (pynccl), or
torch.distributed. Most of these calls are captured into CUDA/HIP graphs and replayed, and most
are not NCCL's or RCCL's: on Qwen3-30B-A3B at TP2, 98% of collective kernels were vLLM's custom
all-reduce (runner ruppell-accuracy/results/m8_probe/). So the calls are recorded here, at the
framework, where each one is seen with its tensor and its group.

Each wrapped call pushes its kind, a collective definition (api.define_raw):
- the operation, the backend and the dtype;
- the element count and bytes of its input, one rank's contribution, found by parameter name
  (_input_arg: torch.distributed's all_gather_into_tensor and reduce_scatter_tensor and pynccl's
  all_gather and reduce_scatter take their output first);
- the group's size and the caller's rank in it, and whether the stream was capturing.
The tool pairs the kind with the scope the call is made from (api.collective_push_kind, a
"collective_at" id), so a kind is defined once per element count rather than once per layer.
Kernels launched inside the call, eagerly or into a graph being captured, carry it; a replay
carries the call made at capture. The engine keeps the module scope as the kernel's scope and
the call beside it (Analysis.coll). Nested wrappers (the dispatcher, then the backend it picks)
leave the innermost, the backend, on the kernels.

Its cost is host time in each eager call (graph replays pay none), so the hot path is kept short.
- CudaCommunicator.all_reduce and CustomAllreduce.custom_all_reduce are not wrapped: every
  backend the first picks, and every launch of the second (self.all_reduce), is wrapped itself.
  They made the common all-reduce three nested wrappers deep, about 23 us of host time a call on
  MI350X (runner ruppell-accuracy/results/m8, step 8).
- A communicator's and a torch.distributed group's size and rank are looked up once, and a dtype's
  name only when a call kind is defined.
- Inside a traced server a repeated call's id cost 7.9 us and a kind's first use 22 us, nearly
  half of the eager calls (B200, runner ruppell-accuracy/results/m8, step 13). So:
  - a definition's JSON is written by an f-string, defined by api.define_raw (was 15 us);
  - the capture check asks the runtime only while a graph capture is open in the process
    (patch_capture_count; was 3.2 us, a CUDA runtime call that CUPTI traces);
  - the lookup's key is a per-wrapper number, the dtype object and the call's own values;
  - no scope in the key (above): a first use was 45-48% of eager calls, defined per layer scope
    (runner ruppell-accuracy/plans/02_ADAPTER_COST.md, B1).

RUPPELL_COLLECTIVE_MARK says how a call reaches its kernels:
- push (the default): the kind pushed as their external correlation, paired with its scope by
  the tool;
- range: the call recorded as a range in calls.bin (api.collective_enter), and nothing pushed.
  The engine gives a kernel the innermost call open on its launching thread at its launch
  (runner ruppell-accuracy/plans/02_ADAPTER_COST.md, B3);
- both: the two together, so a run shows whether they agree (the report's
  collectives.by_range).

Not covered: calls made before ruppell.init in the worker (at load_model), such as pynccl's
warm-up all-reduce when its communicator is created. RUPPELL_COLLECTIVES=0 turns this off.

Timing the wrapper in the program (roadmap-v2 M13): a tight loop puts it at 2-3 us a call, while
a served model's eager collective launches came 10-20 us later with it than without (runner
ruppell-accuracy/results/m8, step 9). Both switches are read when the methods are wrapped, so
neither costs anything when off:
- RUPPELL_COLLECTIVES_TIMING=1: each wrapped call also notes its phases on CLOCK_MONOTONIC (entry;
  its input tensor found, the capture checked, its kind looked up, its id found, defining it on
  first use; pushed; the backend returned; popped), with its thread, whether it was the call
  kind's first use and whether the stream was capturing.
- RUPPELL_COLLECTIVES=time: the methods are wrapped with the timing alone, no record: the backend's
  time without the adapter's work, for comparison.
Each process appends them to collectives_timing.tsv in its run directory (flush_timing).
"""

from __future__ import annotations

import atexit
import functools
import inspect
import itertools
import os
import socket
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from ruppell import api

# module: {class name, or "*" for every class the module defines: (backend, methods)}
TARGETS: Dict[str, Dict[str, Tuple[str, List[str]]]] = {
    "vllm.distributed.device_communicators.cuda_communicator": {"CudaCommunicator": ("vllm", [
        "all_gather", "all_gatherv", "reduce_scatter", "reduce_scatterv", "send", "recv", "broadcast",
        "dispatch", "combine"])},
    "vllm.distributed.device_communicators.custom_all_reduce": {"CustomAllreduce": ("custom", ["all_reduce"])},
    "vllm.distributed.device_communicators.aiter_custom_all_reduce": {"*": ("aiter", ["custom_all_reduce", "all_reduce"])},
    "vllm.distributed.device_communicators.quick_all_reduce": {"QuickAllReduce": ("quick", ["quick_all_reduce"])},
    "vllm.distributed.device_communicators.symm_mem": {"SymmMemCommunicator": ("symm_mem", ["all_reduce"])},
    "vllm.distributed.device_communicators.flashinfer_all_reduce": {"*": ("flashinfer", ["all_reduce"])},
    "vllm.distributed.device_communicators.pynccl": {"PyNcclCommunicator": ("nccl", [
        "all_reduce", "all_gather", "all_gatherv", "reduce_scatter", "reduce_scatterv", "send", "recv", "broadcast"])},
}
TORCH_DIST = ["all_reduce", "all_gather_into_tensor", "reduce_scatter_tensor", "all_gather", "broadcast", "send", "recv"]
OPS = {"custom_all_reduce": "all_reduce", "quick_all_reduce": "all_reduce", "all_gatherv": "all_gather",
       "all_gather_into_tensor": "all_gather", "reduce_scatterv": "reduce_scatter",
       "reduce_scatter_tensor": "reduce_scatter", "dispatch": "all_to_all_dispatch", "combine": "all_to_all_combine"}

# a call's input tensor, by parameter name: the first of a function's parameters named so
INPUT_NAMES = ("input_tensor", "in_tensor", "input_", "input", "inp", "tensor", "hidden_states")

_ids: Dict[tuple, int] = {}     # (wrapper, dtype, numel, world, rank, captured) -> the kind's id
_groups: Dict[int, Tuple[int, int]] = {}  # id(torch.distributed group) -> (size, rank), once found
_comms: Dict[int, tuple] = {}   # id(communicator) -> (the communicator, size, rank)
_dtypes: Dict[Any, str] = {}    # torch dtype -> its name without "torch."
_wrapped: List[str] = []

# Graph captures open in the process (patch_capture_count). With none open, no stream is capturing,
# so a call needs no runtime check: torch.cuda.is_current_stream_capturing() cost 3.2 us a call
# inside a traced server, where CUPTI traces the runtime call (B200, runner
# ruppell-accuracy/results/m8, step 13).
_capture_counted = False
_open_captures = 0
_capture_lock = threading.Lock()
_wrapper_ids = itertools.count(1)

# RUPPELL_COLLECTIVES_TIMING / RUPPELL_COLLECTIVES=time: one row per wrapped call (TIMING_COLUMNS)
TIMING_COLUMNS = ("call", "captured", "first_use", "numel", "thread", "t_entry", "t_tensor", "t_capturing",
                  "t_looked_up", "t_id", "t_pushed", "t_returned", "t_popped", "captured_runtime")
_times: List[tuple] = []
_times_lock = threading.Lock()
_clock = time.monotonic_ns  # CLOCK_MONOTONIC, which clock.tsv pairs with the records' clock


def enabled() -> bool:
    return os.environ.get("RUPPELL_COLLECTIVES", "1") != "0"


def _timing_path() -> Optional[str]:
    out = os.environ.get("RUPPELL_OUT")
    # the tool names it so, the host cut at its first dot (Cluster A's nodes report a domain)
    host = socket.gethostname().split(".")[0]
    return os.path.join(out, f"{host}-{os.getpid()}", "collectives_timing.tsv") if out else None


def flush_timing() -> None:
    """Append the timed calls noted so far to this process's collectives_timing.tsv."""
    with _times_lock:
        rows, _times[:] = list(_times), []
    path = _timing_path()
    if not rows or not path or not os.path.isdir(os.path.dirname(path)):
        return
    new = not os.path.exists(path)
    with open(path, "a") as f:
        if new:
            f.write("\t".join(TIMING_COLUMNS) + "\n")
        f.writelines("\t".join(str(x) for x in r) + "\n" for r in rows)


def _note(row: tuple) -> None:
    _times.append(row)
    if len(_times) >= 4096:
        flush_timing()


atexit.register(flush_timing)


def _group(obj: Any, kwargs: dict) -> Tuple[int, int]:
    """The group's size and the caller's rank in it: a communicator's own attributes, read once, or
    for torch.distributed the `group` argument."""
    if obj is not None:
        c = _comms.get(id(obj))
        if c is not None and c[0] is obj:
            return c[1], c[2]
        world, rank = getattr(obj, "world_size", None), getattr(obj, "rank_in_group", getattr(obj, "rank", None))
        if isinstance(world, int):
            wr = int(world), int(rank) if isinstance(rank, int) else -1
            _comms[id(obj)] = (obj, *wr)
            return wr
    g = kwargs.get("group")
    wr = _groups.get(id(g))
    if wr is None:
        try:
            import torch.distributed as dist
            wr = _groups[id(g)] = (int(dist.get_world_size(g)), int(dist.get_rank(g)))
        except Exception:  # not initialized yet: not kept, asked again next call
            return -1, -1
    return wr


def patch_capture_count() -> None:
    """Count the graph captures open in the process, at torch.cuda.CUDAGraph's capture_begin and
    capture_end, through which vLLM captures every graph: directly, as breakable segments, and the
    speculator's. Not installed (no torch, or no CUDAGraph): every call asks the runtime."""
    global _capture_counted
    try:
        import torch
        cls = torch.cuda.CUDAGraph
        begin, end = cls.capture_begin, cls.capture_end
    except Exception:  # noqa: BLE001
        return
    if getattr(cls, "_ruppell_capture_count", False):
        _capture_counted = True
        return

    def capture_begin(self, *args, **kwargs):
        global _open_captures
        r = begin(self, *args, **kwargs)
        with _capture_lock:
            _open_captures += 1
        return r

    def capture_end(self, *args, **kwargs):
        global _open_captures
        try:
            return end(self, *args, **kwargs)
        finally:
            with _capture_lock:
                _open_captures = max(0, _open_captures - 1)

    cls.capture_begin, cls.capture_end = capture_begin, capture_end
    cls._ruppell_capture_count = True
    _capture_counted = True


def _input_arg(fn: Callable) -> Tuple[int, str]:
    """The position (-1 when keyword-only) and name of fn's input tensor parameter: the first
    named in INPUT_NAMES. (-1, "") when none is, or fn has no signature."""
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return -1, ""
    for i, prm in enumerate(params):
        if prm.name in INPUT_NAMES:
            return (i if prm.kind in (prm.POSITIONAL_ONLY, prm.POSITIONAL_OR_KEYWORD) else -1), prm.name
    return -1, ""


def _wrap(backend: str, method: str, fn: Callable, is_method: bool) -> Callable:
    import torch

    op = OPS.get(method, method)
    pos, name = _input_arg(fn)
    call = f"{backend}.{method}"
    wid = next(_wrapper_ids)  # stands for (op, backend, method) in the lookup's key

    def tensor(args: tuple, kwargs: dict) -> Any:
        t = args[pos] if 0 <= pos < len(args) else kwargs.get(name) if name else None
        if not isinstance(t, torch.Tensor):  # no input parameter: the first tensor given
            t = next((x for x in list(args) + list(kwargs.values()) if isinstance(x, torch.Tensor)), None)
        return t

    def runtime_capturing() -> bool:
        try:
            return bool(torch.cuda.is_current_stream_capturing())
        except Exception:
            return False

    def call_id(args: tuple, kwargs: dict, marks: Optional[list] = None) -> Tuple[int, bool, int, bool]:
        """The call's kind, defined on its first use: (kind, captured, numel, first use). No scope:
        the tool pairs the kind with it (api.collective_push_kind). marks (the timed wrapper's):
        the clock after the tensor, the capture check, the lookup."""
        t = args[pos] if 0 <= pos < len(args) else kwargs.get(name) if name else None
        if not isinstance(t, torch.Tensor):  # no input parameter: the first tensor given
            t = next((x for x in list(args) + list(kwargs.values()) if isinstance(x, torch.Tensor)), None)
        if marks is not None:
            marks.append(_clock())
        # with no graph capture open in the process, no stream is capturing (patch_capture_count)
        captured = bool((not _capture_counted or _open_captures) and runtime_capturing())
        if marks is not None:
            marks.append(_clock())
        world, rank = _group(args[0] if is_method and args else None, kwargs)
        if t is not None:
            dt, numel = t.dtype, t.numel()
        else:
            dt, numel = None, -1
        key = (wid, dt, numel, world, rank, captured)
        cid = _ids.get(key)
        if marks is not None:
            marks.append(_clock())
        first = cid is None
        if first:
            # the attributes as json.dumps(..., sort_keys=True) writes them, an f-string's cost
            dtype = "" if dt is None else _dtypes.get(dt) or _dtypes.setdefault(dt, str(dt).replace("torch.", ""))
            nbytes = numel * t.element_size() if t is not None else -1
            cid = _ids[key] = api.define_raw("collective", op, (
                f'{{"backend": "{backend}", "bytes": {nbytes}, "captured": {"true" if captured else "false"}, '
                f'"dtype": "{dtype}", "method": "{method}", "numel": {numel}, "rank": {rank}, '
                f'"world": {world}}}'))
        return cid, captured, numel, first

    # how a call is marked on its kernels (RUPPELL_COLLECTIVE_MARK): push its kind, which the tool
    # pairs with its scope (the default); or record it as a range in calls.bin, with no push
    # (plan 02, B3); or both, the check that the two agree
    mark = os.environ.get("RUPPELL_COLLECTIVE_MARK", "push")
    if mark == "range":
        enter, leave = api.collective_enter, api.collective_exit
    elif mark == "both":
        def enter(kind: int) -> tuple:
            return api.collective_push_kind(kind), api.collective_enter(kind)

        def leave(token: tuple) -> None:
            api.collective_exit(token[1])
            api.collective_pop(token[0])
    else:
        enter, leave = api.collective_push_kind, api.collective_pop

    if os.environ.get("RUPPELL_COLLECTIVES") == "time":
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):  # the timing alone, no record
            t0 = _clock()
            try:
                return fn(*args, **kwargs)
            finally:
                t3 = _clock()
                t = tensor(args, kwargs)
                cap = int(runtime_capturing())
                _note((call, cap, -1, int(t.numel()) if t is not None else -1,
                       threading.get_native_id(), t0, t0, t0, t0, t0, t0, t3, t3, cap))
    elif os.environ.get("RUPPELL_COLLECTIVES_TIMING", "0") not in ("", "0"):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):  # the record, its phases timed
            if api._mode != "active" or not api._fine_scopes:
                return fn(*args, **kwargs)
            t0 = _clock()
            marks: list = []
            cid, captured, numel, first = call_id(args, kwargs, marks)
            t1 = _clock()
            token = enter(cid)
            t2 = _clock()
            try:
                return fn(*args, **kwargs)
            finally:
                t3 = _clock()
                leave(token)
                t4 = _clock()
                # the runtime's own answer, after the call: the check that the capture count agrees
                _note((call, int(captured), int(first), numel, threading.get_native_id(), t0, *marks, t1, t2, t3,
                       t4, int(runtime_capturing())))
    else:
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            if api._mode != "active" or not api._fine_scopes:
                return fn(*args, **kwargs)
            token = enter(call_id(args, kwargs)[0])
            try:
                return fn(*args, **kwargs)
            finally:
                leave(token)

    wrapped._ruppell_collective = True
    return wrapped


def patch_module(module: Any) -> None:
    """Wrap the collective methods a vLLM communicator module defines (TARGETS)."""
    for cname, (backend, methods) in TARGETS.get(module.__name__, {}).items():
        if cname == "*":
            classes = [c for c in vars(module).values() if isinstance(c, type) and c.__module__ == module.__name__]
        else:
            classes = [getattr(module, cname)] if hasattr(module, cname) else []
        for cls in classes:
            for meth in methods:
                f = cls.__dict__.get(meth)
                if f is None or isinstance(f, (staticmethod, classmethod)) or getattr(f, "_ruppell_collective", False):
                    continue
                setattr(cls, meth, _wrap(backend, meth, f, True))
                _wrapped.append(f"{cls.__name__}.{meth}")


def patch_torch_distributed() -> None:
    try:
        import torch.distributed as dist
    except ImportError:  # no torch: nothing to patch (the offline tests)
        return

    for name in TORCH_DIST:
        f = getattr(dist, name, None)
        if f is not None and not getattr(f, "_ruppell_collective", False):
            setattr(dist, name, _wrap("torch.distributed", name, f, False))
            _wrapped.append(f"torch.distributed.{name}")


def patches() -> Dict[str, Callable[[Any], None]]:
    """The module patches to install, by module name (the adapter's PATCHES)."""
    return {name: patch_module for name in TARGETS} if enabled() else {}
