"""The API a program or framework adapter calls.

Start the program with `python -m ruppell run --out DIR -- <command>`, which
loads the runtime into the process. Then:

    ruppell.init()
    with ruppell.scope("model.layers.0.mlp.fc1", op="linear", shape=[16, 1024]):
        ...                                   # kernels launched here carry the scope
    with ruppell.capture("decode.b16"):
        ...                                   # a graph captured here is registered under the key
    ruppell.step_begin(n); ...; ruppell.step_end(n)   # marker kernels bound step n on the GPU

Modes:
- "active": the runtime is loaded and tracing;
- "witness": RUPPELL_MODE=witness, for running the same program under an
  independent collector: markers are still launched (so the kernel sequence
  is identical), scopes and captures do nothing.
Anything else refuses: a tracer that silently does nothing produces traces
that look complete and are empty.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional, Tuple

from ruppell import _lib

MARKER_BIT = 1 << 63
_mode: Optional[str] = None
_lock = threading.Lock()
_ids: Dict[Tuple[str, str, str], int] = {}
_next_id = 1
_tls = threading.local()  # the scope names open on this thread, innermost last


def _clean(s: str) -> str:
    return str(s).replace("\t", " ").replace("\n", " ")


def _require() -> str:
    if _mode is None:
        raise RuntimeError("ruppell.init() has not been called")
    return _mode


def init(schema: Optional[str] = None, out_dir: Optional[str] = None, *, detail_sample_rate: float = 0.0) -> str:
    """Check the runtime is loaded and return the mode ("active" or "witness").

    ``schema`` and ``detail_sample_rate`` are accepted for the M2 API and not
    used yet; ``out_dir`` is set by `ruppell run` (RUPPELL_OUT).
    """
    global _mode
    if os.environ.get("RUPPELL_MODE") == "witness":
        _mode = "witness"
        _lib.marker()
        return _mode
    lib = _lib.tool()
    if lib is None or lib.ruppell_active() != 1:
        raise RuntimeError(
            "ruppell runtime is not loaded in this process: start it with "
            "`python -m ruppell run --out DIR -- <command>` (or set RUPPELL_MODE=witness)"
        )
    _lib.marker()
    _mode = "active"
    if schema:
        context(schema=schema)
    from ruppell import cpu_ops

    cpu_ops.start()  # RUPPELL_CPU_OPS: the CPU op tree, off by default
    return _mode


def _define(kind: str, name: str, attrs: Dict[str, Any]) -> int:
    global _next_id
    blob = json.dumps(attrs, sort_keys=True, default=str)
    key = (kind, name, blob)
    with _lock:
        sid = _ids.get(key)
        if sid is None:
            sid = _next_id
            _next_id += 1
            _ids[key] = sid
            # Witness mode loads no tool: the id still exists (callers cache it for
            # push_id, which does nothing in witness mode), but nothing is recorded.
            if _mode != "witness":
                _lib.tool().ruppell_define(sid, kind.encode(), _clean(name).encode(), _clean(blob).encode())
    return sid


def scope_push(name: str, **attrs: Any) -> None:
    """Enter a scope (for hooks that cannot use a with-block)."""
    if _require() != "active":
        return
    sid = _define("scope", name, attrs)
    rc = _lib.tool().ruppell_push(sid)
    if rc != 0:
        raise RuntimeError(f"ruppell_push failed ({rc})")
    _enter(name, sid)


def _enter(name: str, sid: int) -> None:
    """Note a scope entered on this thread: its name (current_scope) and id (current_scope_id)."""
    st = getattr(_tls, "stack", None)
    if st is None:
        st = _tls.stack = []
        _tls.ids = []
    st.append(name)
    _tls.ids.append(sid)


def _leave() -> None:
    st = getattr(_tls, "stack", None)
    if st:
        st.pop()
        _tls.ids.pop()


def scope_id(name: str, **attrs: Any) -> int:
    """The id of a scope, defined once: for callers that push the same scope
    many times (hooks, eager breaks at every replay) and cache the id."""
    return _define("scope", name, attrs)


_fine_scopes = True  # False while detail sampling has kernel records off


def push_id(sid: int, name: str) -> None:
    """Enter a scope by the id scope_id() returned (the hot path: no lookup).
    With pop_id, the scopes of module hooks and eager breaks: skipped while
    detail is off (no kernel record would carry them). detail() only changes
    between steps, never inside a forward, so pushes and pops stay paired."""
    if _mode != "active" or not _fine_scopes:
        return
    _lib.tool().ruppell_push(sid)
    _enter(name, sid)


def pop_id() -> None:
    """Leave a scope entered with push_id."""
    if _mode != "active" or not _fine_scopes:
        return
    _lib.tool().ruppell_pop()
    _leave()


def scope_pop() -> None:
    if _require() != "active":
        return
    _lib.tool().ruppell_pop()
    _leave()


def current_scope() -> Optional[str]:
    """The innermost scope open on this thread (None outside any scope)."""
    st = getattr(_tls, "stack", None)
    return st[-1] if st else None


def current_scope_id() -> int:
    """The id of the innermost scope open on this thread (0 outside any scope)."""
    ids = getattr(_tls, "ids", None)
    return ids[-1] if ids else 0


# Collective calls (roadmap-v2 M8). A framework adapter pushes one around each communication
# call. Its kernels, launched eagerly or into a graph being captured, carry the call, and keep
# the scope it was made from as their scope: the definition's parent. The engine reads both.

def collective_id(op: str, parent: int, **attrs: Any) -> int:
    """The id of a collective call: its operation ("all_reduce", "all_gather", ...), the scope id
    it was called from, and attributes (backend, dtype, numel, bytes, world, rank). Defined once
    per distinct set; callers on a hot path cache it."""
    return _define("collective", op, {**attrs, "parent": int(parent)})


def define_raw(kind: str, name: str, attrs_json: str) -> int:
    """A new id for a definition whose attributes the caller has already written as JSON, with no
    tab or newline in name or attributes: for a hot path that keeps its own cache (the vLLM
    adapter's collective calls), so nothing is looked up, serialized or cleaned here. Each call
    defines a new id. json.dumps and _define's lookup cost a served model's eager collective call
    about 14 us on a call kind's first use, nearly half of them (B200, runner
    ruppell-accuracy/results/m8, step 13)."""
    global _next_id
    with _lock:
        sid = _next_id
        _next_id += 1
    if _mode != "witness":
        _lib.tool().ruppell_define(sid, kind.encode(), name.encode(), attrs_json.encode())
    return sid


def collective_push(cid: int) -> bool:
    """Enter a collective call by its id. Returns whether it was pushed: pass that to
    collective_pop. Skipped, as module scopes are, while detail sampling has records off."""
    if _mode != "active" or not _fine_scopes:
        return False
    return _lib.tool().ruppell_push(cid) != -1


def collective_pop(pushed: bool) -> None:
    if pushed:
        _lib.tool().ruppell_pop()


_push_kind_missing = False


def collective_push_kind(kind: int) -> bool:
    """Enter a collective call by its kind: a "collective" definition with no parent. The tool
    pairs it with the scope the call is made from, the innermost scope pushed on this thread, and
    pushes the pair's own id ("collective_at"), which the engine reads as both. So a kind is
    defined once per element count, not once per layer scope (runner ruppell-accuracy, plan 02,
    B1). A tool built before this pushes the kind alone, and its kernels lose their module scope."""
    global _push_kind_missing
    if _mode != "active" or not _fine_scopes:
        return False
    lib = _lib.tool()
    fn = getattr(lib, "ruppell_push_collective", None)
    if fn is None:
        if not _push_kind_missing:
            _push_kind_missing = True
            print("[ruppell] this tool library has no ruppell_push_collective (built before it): collective "
                  "kernels keep their call but not their module scope; rebuild it", file=sys.stderr, flush=True)
        return lib.ruppell_push(kind) != -1
    return fn(kind) != -1


def collective_enter(kind: int) -> bool:
    """Enter a collective call by its kind, recorded as a range rather than pushed: the tool
    writes it to calls.bin when collective_exit closes it, and the engine gives each kernel
    launched inside it the call, keeping the scope it already has (runner ruppell-accuracy, plan
    02, B3). Returns whether it was entered: pass that to collective_exit. A tool built before it
    records nothing."""
    if _mode != "active" or not _fine_scopes:
        return False
    fn = getattr(_lib.tool(), "ruppell_call_enter", None)
    return fn is not None and fn(kind) == 0


def collective_exit(entered: bool) -> None:
    if entered:
        _lib.tool().ruppell_call_exit()


@contextmanager
def scope(name: str, **attrs: Any) -> Iterator[None]:
    """Name the code launching kernels: a module path, and attributes such as op and shape."""
    scope_push(name, **attrs)
    try:
        yield
    finally:
        scope_pop()


@contextmanager
def capture(graph_key: str, **attrs: Any) -> Iterator[None]:
    """The next graph captured on this thread is registered under ``graph_key``.

    Also opens a scope ``capture:<graph_key>``, so work the framework does
    eagerly while setting up the capture (PyTorch initializes the graph's RNG
    state, for example) is attributed to the capture, and captured launches
    outside any module scope are attributed to the graph.
    """
    if _require() == "active":
        _lib.tool().ruppell_capture_key(_define("graph", graph_key, attrs))
    with scope(f"capture:{graph_key}", op="graph_capture", **attrs):
        yield


def _stream(stream: Any) -> int:
    if stream is not None:
        return int(getattr(stream, "cuda_stream", stream))
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        return int(torch.cuda.current_stream().cuda_stream)
    return 0


def _marker(step_id: int, kind: int, stream: Any) -> None:
    code = MARKER_BIT | (int(step_id) << 2) | kind
    active = _require() == "active"
    if active:
        _lib.tool().ruppell_push(code)
    try:
        rc = _lib.marker().ruppell_marker_launch(_stream(stream), code)
        if rc != 0:
            raise RuntimeError(f"marker launch failed (hipError {rc})")
    finally:
        if active:
            _lib.tool().ruppell_pop()


def step_begin(step_id: int, stream: Any = None, **fields: Any) -> None:
    """Mark step ``step_id``'s start on the GPU timeline (on ``stream``, default the
    current torch stream) and record its workload fields and the CPU time the
    step began (for charging side-stream work to the step that launched it)."""
    if _require() == "active":
        if fields:
            _lib.tool().ruppell_step_fields(int(step_id), _clean(json.dumps(fields, default=str)).encode())
        _lib.tool().ruppell_step_cpu(int(step_id), 1)
    if _fine_scopes or _mode != "active":
        # no marker while detail sampling has kernel records off: nothing would
        # record it; the step's fields and CPU window are still written
        _marker(step_id, 1, stream)


def step_end(step_id: int, stream: Any = None, **fields: Any) -> None:
    """Mark step ``step_id``'s end on the GPU timeline; ``fields`` learned during
    the step (the graphs it replayed, say) are merged into its fields."""
    if _fine_scopes or _mode != "active":
        _marker(step_id, 2, stream)
    if _require() == "active":
        _lib.tool().ruppell_step_cpu(int(step_id), 2)
        if fields:
            _lib.tool().ruppell_step_fields(int(step_id), _clean(json.dumps(fields, default=str)).encode())


def detail(on: bool) -> bool:
    """Kernel dispatch records on or off (detail sampling). Returns whether the
    runtime took it (False in witness mode, or without dispatch tracing)."""
    global _fine_scopes
    if _mode != "active":
        return False
    lib = _lib.tool()
    if hasattr(lib, "ruppell_detail_on_stream"):
        # CUDA: "off" first waits for the GPU to finish the sampled step
        # (runtime/src/tool_cuda.cpp, ruppell_detail_on_stream)
        ok = lib.ruppell_detail_on_stream(1 if on else 0, _stream(None)) == 0
    else:
        ok = lib.ruppell_detail(1 if on else 0) == 0
    if ok:
        _fine_scopes = bool(on)
        from ruppell import cpu_ops

        cpu_ops.pause(not on)  # the CPU op tree follows detail (its cost is per op)
    return ok


# Hardware counters (gray-parrot, docs/gray-parrot/README.md): off unless RUPPELL_COUNTERS is set.
# A counter step's kernels are serialized by the counting, so its times are not timing: the
# analysis leaves counter steps out of every time table.

def _rate(var: str, default: Optional[str], environ: Optional[Dict[str, str]] = None) -> Optional[Tuple[int, int]]:
    v = (os.environ if environ is None else environ).get(var, default or "")
    if v in ("", "all"):
        return None
    k, _, n = v.partition("/")
    try:
        k_, n_ = int(k), int(n)
    except ValueError:
        raise ValueError(f"{var}={v}: want K/N") from None
    if not (0 <= k_ <= n_ and n_ > 0):
        raise ValueError(f"{var}={v}: want K/N with 0 <= K <= N, N > 0")
    return k_, n_


def default_counter_steps(environ: Optional[Dict[str, str]] = None) -> str:
    """RUPPELL_COUNTER_STEPS's default: 1/100 on ROCm (a proposal); 1/1000 on CUDA, where a counter
    step costs the program 60-120 ms of host time, mostly CUPTI's Range Profiler start, which grows
    with the program (B200: 1/100 cost a served Qwen3-8B 10 % of its throughput, 1/1000 less than
    three rounds resolve; runner plan 04, Part B). CUDA is the tool injected by CUDA_INJECTION64_PATH."""
    env = os.environ if environ is None else environ
    return "1/1000" if env.get("CUDA_INJECTION64_PATH") else "1/100"


def counter_steps() -> Optional[Tuple[int, int]]:
    """RUPPELL_COUNTER_STEPS as (K, N), K of every N steps carrying counters (default:
    default_counter_steps(); 0/N configures counting and never starts it). None when
    RUPPELL_COUNTERS is unset."""
    if not os.environ.get("RUPPELL_COUNTERS"):
        return None
    return _rate("RUPPELL_COUNTER_STEPS", default_counter_steps())


def is_counter_step(step_id: int) -> bool:
    """Whether step ``step_id`` carries counters: the last K of every N step ids ((id mod N) >=
    N - K), so every rank picks the same steps. With detail sampling (RUPPELL_DETAIL=K/N, detail
    for id mod N < K), a detail step never carries counters: counter steps come from the steps
    outside the detail sample. With full detail every step is a detail step, and counter steps
    are taken out of timing instead."""
    kn = counter_steps()
    if kn is None or kn[0] == 0:
        return False
    if (int(step_id) % kn[1]) < kn[1] - kn[0]:
        return False
    detail = _rate("RUPPELL_DETAIL", "all")
    return detail is None or (int(step_id) % detail[1]) >= detail[0]


_counters_restore_scopes = False  # counting turned scopes on, for a step detail sampling left off


def counters(on: bool, group: Optional[int] = None, step: int = 0, drain: Optional[bool] = None) -> int:
    """Hardware counters on or off, around a counter step: on before its first kernel is queued,
    off after its last. Counting on turns kernel dispatch records (and scopes) on too, so the
    counted kernels keep their identity, scope and graph join; off restores detail sampling's
    choice. ``group``: the group to count, or None for the next in rotation (every rank rotates
    the same way). ``step``: the counter step's id, which its records carry (counters.bin's
    ``step``). ``drain`` (default on; RUPPELL_COUNTER_DRAIN=0 turns it off): before stopping,
    wait for the device to finish the step's kernels. Stopping the SDK's counting with counted
    kernels still queued hung the GPU (MI350X, SDK 1.3.2, runner plan 03 probe A2: every other step
    counted, both W0 programs), and draining first did not (1,000 starts and stops).
    Returns the group counted (on), 0 (off), or a negative code: -1 without counters (unset,
    refused at start, or not tracing), -2 the profiler did not start or stop counting, -3 a group
    out of range."""
    global _fine_scopes, _counters_restore_scopes
    if _mode != "active":
        return -1
    lib = _lib.tool()
    if not hasattr(lib, "ruppell_counters"):
        return -1
    if on:
        g = lib.ruppell_counters(1, -1 if group is None else int(group), int(step))
        if g >= 0 and not _fine_scopes:
            _fine_scopes = True
            _counters_restore_scopes = True
        return g
    if drain if drain is not None else os.environ.get("RUPPELL_COUNTER_DRAIN", "1") != "0":
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
    rc = lib.ruppell_counters(0, 0, 0)
    if rc == 0 and _counters_restore_scopes:
        _fine_scopes = False
        _counters_restore_scopes = False
    return rc


def flush(complete: bool = False) -> None:
    """Write out everything recorded so far, for a process that may be killed
    before the profiler finalizes (a server's worker at shutdown).

    complete: the caller records nothing more (a worker shutting down). The
    device's work is waited for, and once the flush has returned the records
    are marked complete (context `records_complete`), so the analysis can
    tell this process from one killed earlier, whose last seconds are lost
    without a trace. A process killed during the flush stays unmarked."""
    if _require() == "active":
        if complete:
            torch = sys.modules.get("torch")
            if torch is not None and torch.cuda.is_available():
                torch.cuda.synchronize()
        _lib.tool().ruppell_flush()
        if complete:
            context(records_complete=True)   # defs.tsv is written through at once


def context(**fields: Any) -> None:
    """Record run-level fields (for example the client's phase clock)."""
    if _require() != "active":
        return
    for k, v in fields.items():
        _lib.tool().ruppell_define(0, b"context", _clean(k).encode(), _clean(json.dumps({"value": v}, default=str)).encode())
