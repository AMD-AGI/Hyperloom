"""The vLLM adapter (docs/plan-m2.md §2, §9): steps, scopes and graph keys
from inside vLLM's v2 GPU model runner, with no change to vLLM.

vLLM loads it as a general plugin in every process (entry point
`vllm.general_plugins`); it does nothing unless RUPPELL_VLLM=1. When the
worker imports `vllm.v1.worker.gpu.model_runner`, the adapter wraps:

- `GPUModelRunner.load_model`: initializes ruppell (HIP is up by then),
  hooks every module of the target model (scopes `model.*`) and of the
  drafter (`drafter.*`, modules shared with the target keep their target
  scope), and records the run fields (model, tp, rank, role, cudagraph_mode).
- `profile_run` and `capture_model`: named steps (ids from 2**40), so startup
  work is bounded and labelled.
- `execute_model` / `sample_tokens`: one engine step per non-dummy
  `execute_model` call, numbered by a per-worker counter from 1 (9.1). The
  step opens before the runner updates its state and closes when
  `sample_tokens` returns (or at once when `execute_model` returns output
  itself). Its fields come from the scheduler output, which every rank
  receives identically, and its fingerprint hashes the request ids and
  token counts, so a merge can check the ranks agree.
- `Worker.compile_or_warm_up_model` (all of startup warmup: `kernel_warmup`
  and `warmup_kernels`) and `vllm.v1.worker.gpu.warmup._warmup_kernels`:
  vLLM's startup warmup drives
  execute_model and sample_tokens with made-up batches (an empty one, one of
  max_num_seqs requests); their steps are named `warmup` steps, so engine
  steps are the served workload's, numbered from its first.
- every wrapped method runs inside a scope `runner.<method>`, so the runner's
  own work (input preparation, copies, sampling) is attributed; module
  scopes nest inside it.
- `shutdown`: flushes the records, since a worker can be killed before the
  profiler finalizes.
- `torch.cuda.CUDAGraph.capture_begin`: every graph captured, directly, as
  a breakable segment, or by the speculator, is registered under
  `role|passN|gM` (9.3): role is `target` inside the model's capture and
  `drafter` inside the speculator's, N counts `capture_model` calls, M counts
  graphs within the pass. The key is only a name; the join is by graph.
  No scope is opened here: breakable segments begin and end inside module
  forwards, and a scope opened at one and closed at the other would break
  the stack.
- `BreakableCUDAGraphCapture.add_eager`: the eager function between two
  segments runs again at every replay, outside any module hook; it is
  wrapped in the scope that was innermost when it was added (op
  `eager_break`), so its kernels stay attributed.
- vLLM's compiled path (-O2), where module hooks do not run: each compiled
  piece, each launch site in a piece's generated program, and each attention
  layer get a scope, named as the module's hook would name it (`compiled.py`;
  RUPPELL_VLLM_COMPILED=0 turns it off).

- with RUPPELL_TUNING=FILE, in witness mode too: kernels vLLM tunes at run
  time by timing (Inductor's, `@triton.autotune`) take their choice from
  FILE, so a traced and an untraced server run the same program
  (`ruppell/adapters/tuning.py`).

Before patching, every wrapped method's signature is checked; a mismatch
leaves the process untraced and says which hook and why, rather than
tracing half the points. With RUPPELL_VLLM=1 and no runtime loaded, the
worker fails at load (ruppell.init refuses).
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.abc
import importlib.util
import inspect
import json
import os
import sys
import threading
from typing import Any, List, Optional

TARGET = "vllm.v1.worker.gpu.model_runner"
WARMUP = "vllm.v1.worker.gpu.warmup"
WORKER = "vllm.v1.worker.gpu_worker"
NAMED_BASE = 1 << 40

_tls = threading.local()
_state = {"patched": False, "mode": None, "pass": 0, "graph": 0, "named": 0, "step": 0, "open": None, "warmup": False}


def _log(msg: str) -> None:
    print(f"[ruppell] pid {os.getpid()}: {msg}", file=sys.stderr, flush=True)


# ---- fields ---------------------------------------------------------------------------------------

def step_fields(so: Any) -> dict:
    """Serving-schema fields of one engine step, from vLLM's SchedulerOutput."""
    sched = dict(so.num_scheduled_tokens)
    spec = {k: len(v) for k, v in (so.scheduled_spec_decode_tokens or {}).items()}
    computed = {}
    for r in so.scheduled_new_reqs:
        computed[r.req_id] = int(r.num_computed_tokens)
    cached = so.scheduled_cached_reqs
    for rid, n in zip(getattr(cached, "req_ids", []), getattr(cached, "num_computed_tokens", [])):
        computed[rid] = int(n)
    order = sorted(sched)
    decoding = [sched[r] == 1 + spec.get(r, 0) for r in order]
    if not order:
        kind = "empty"
    elif all(decoding):
        kind = "decode"
    elif not any(decoding):
        kind = "prefill"
    else:
        kind = "mixed"
    fp = hashlib.blake2b(json.dumps([[r, sched[r], spec.get(r, 0)] for r in order]).encode(), digest_size=8).hexdigest()
    return {
        "phase": "engine",
        "running_requests": len(order),
        "rows": int(so.total_num_scheduled_tokens),
        "spec_rows": int(sum(spec.values())),
        "context_lengths": [computed.get(r, 0) + int(sched[r]) for r in order],
        "kind": kind,
        "fingerprint": fp,
    }


# ---- graph keys -----------------------------------------------------------------------------------

def _patch_graph_capture() -> None:
    import torch

    from ruppell import _lib
    from ruppell.api import _define

    cls = torch.cuda.CUDAGraph
    if getattr(cls, "_ruppell_patched", False):
        return
    orig = cls.capture_begin

    def capture_begin(self, *args, **kwargs):
        if _state["mode"] == "active":
            role = getattr(_tls, "role", None) or "other"
            _state["graph"] += 1
            key = f"{role}|pass{_state['pass']}|g{_state['graph']:04d}"
            attrs = {"role": role, "pass": _state["pass"], "seq": _state["graph"]}
            _lib.tool().ruppell_capture_key(_define("graph", key, attrs))
        return orig(self, *args, **kwargs)

    cls.capture_begin = capture_begin
    cls._ruppell_patched = True


def _with_role(fn, role: str):
    if getattr(fn, "_ruppell_role", None):
        return fn

    def wrapped(*args, **kwargs):
        prev = getattr(_tls, "role", None)
        _tls.role = role
        try:
            return fn(*args, **kwargs)
        finally:
            _tls.role = prev

    wrapped._ruppell_role = role
    return wrapped


def _with_scope(fn, name: str):
    """Scope `name` around a speculator's propose, so the eager work it does
    outside the drafter's modules (input preparation, a DFlash drafter's
    context-K/V precompute) is the drafter's, not runner.sample_tokens'."""
    if getattr(fn, "_ruppell_scope", None):
        return fn
    from ruppell.api import scope

    def wrapped(*args, **kwargs):
        with scope(name, op="drafter"):
            return fn(*args, **kwargs)

    wrapped._ruppell_scope = name
    return wrapped


# ---- steps ----------------------------------------------------------------------------------------

def _runner_scope(method: str):
    """Scope `runner.<method>` around a runner method, so its own work (input
    preparation, copies, sampling) is attributed; module scopes nest inside."""
    if _state["mode"] != "active":
        return contextlib.nullcontext()
    from ruppell.api import scope

    return scope(f"runner.{method}", op="runner")


def _named(phase: str, fn):
    import ruppell

    def run(*args, **kwargs):
        _close_open()
        _state["named"] += 1
        sid = NAMED_BASE + _state["named"]
        ruppell.step_begin(sid, phase=phase)
        try:
            with _runner_scope(phase):
                return fn(*args, **kwargs)
        finally:
            ruppell.step_end(sid)

    return run


def _sampling():
    """RUPPELL_DETAIL: "all" (default), or "K/N": kernel detail for K of every N
    engine steps (steps whose id mod N < K, the same on every rank). Kernel
    dispatch tracing is the tracer's main host cost on a served model (M3);
    startup, capture and warmup are always traced."""
    v = os.environ.get("RUPPELL_DETAIL", "all")
    if v in ("", "all"):
        return None
    k, _, n = v.partition("/")
    k, n = int(k), int(n)
    if not 0 < k <= n:
        raise ValueError(f"RUPPELL_DETAIL={v}: want K/N with 0 < K <= N")
    return k, n


def _detail_kinds() -> dict:
    """RUPPELL_DETAIL_KINDS: step kinds (decode, prefill, mixed) traced at a rate
    of their own, whatever RUPPELL_DETAIL samples: "mixed" (every mixed step)
    or "mixed:K/N" (K of every N mixed steps, counted per kind; every rank sees
    the same sequence of kinds, so ranks agree). Sampling by step count starves
    kinds that are few steps and much time: on AgentX at C32, mixed steps were
    55% of engine time after 300 s and 3 of 40 samples (jobs 50403/50531).
    Tracing every mixed step moved the operating point (round e: C32
    throughput -2.5%, C8 in flight +0.7% over the bound)."""
    v = os.environ.get("RUPPELL_DETAIL_KINDS", "")
    out = {}
    for item in (x.strip() for x in v.split(",")):
        if not item:
            continue
        kind, _, rate = item.partition(":")
        kind = kind.strip()
        if kind not in ("decode", "prefill", "mixed", "empty"):
            raise ValueError(f"RUPPELL_DETAIL_KINDS={v}: unknown kind {kind!r}")
        k, n = 1, 1
        if rate:
            a, _, b = rate.partition("/")
            k, n = int(a), int(b)
            if not 0 < k <= n:
                raise ValueError(f"RUPPELL_DETAIL_KINDS={v}: want kind:K/N with 0 < K <= N")
        out[kind] = (k, n)
    return out


def _sample(sid: int, kind: Optional[str] = None) -> Optional[bool]:
    """Detail for engine step ``sid``; returns whether the step is a sampled detail step (None:
    full detail, every step traced)."""
    import ruppell

    kn = _state.get("sampling")
    if kn is None:
        return None
    # the kind comes from the scheduler output every rank receives, so ranks agree
    on = (sid % kn[1]) < kn[0]
    rate = _state.get("detail_kinds", {}).get(kind)
    if rate is not None:
        counts = _state.setdefault("kind_count", {})
        i = counts.get(kind, 0)
        counts[kind] = i + 1
        on = on or (i % rate[1]) < rate[0]
    if on != _state.get("detail", True):
        ruppell.detail(on)
        _state["detail"] = on
    return on


def _counter_step(sid: int, detail_step: Optional[bool]) -> Optional[int]:
    """Hardware counters (gray-parrot) for engine step ``sid`` when RUPPELL_COUNTER_STEPS picks it
    and it is not a sampled detail step (RUPPELL_DETAIL_KINDS' steps included): every rank picks
    the same steps and rotates groups the same way. Counting goes on before the step's begin
    marker and off after its end marker (_close_open). Returns the group counted, or None."""
    import ruppell

    if not _state.get("counter_steps") or detail_step or not ruppell.is_counter_step(sid):
        return None
    g = ruppell.counters(True, step=sid)
    if g < 0:
        return None
    _state["counting"] = True
    return g


def _close_open() -> None:
    import ruppell

    sid = _state["open"]
    if sid is not None:
        _state["open"] = None
        ruppell.step_end(sid)
        if _state.get("counting"):
            _state["counting"] = False
            ruppell.counters(False)


# ---- the runner -----------------------------------------------------------------------------------

def _context(runner: Any) -> dict:
    cfg = runner.vllm_config
    try:
        from vllm.distributed import get_tensor_model_parallel_rank

        rank = int(get_tensor_model_parallel_rank())
    except Exception:  # noqa: BLE001 - not initialized: single process
        rank = 0
    spec = cfg.speculative_config
    return {
        "model": str(cfg.model_config.model),
        "tp": int(cfg.parallel_config.tensor_parallel_size),
        "rank": rank,
        "role": "worker",
        "cudagraph_mode": str(cfg.compilation_config.cudagraph_mode),
        "breakable_cudagraph": os.environ.get("VLLM_USE_BREAKABLE_CUDAGRAPH"),
        "speculative_method": getattr(spec, "method", None) if spec is not None else None,
        "num_speculative_tokens": getattr(spec, "num_speculative_tokens", None) if spec is not None else None,
        "vllm_runner": f"{type(runner).__module__}.{type(runner).__qualname__}",
    }


def _instrument(module: Any, prefix: str) -> int:
    from ruppell.torch_hooks import instrument

    return len(instrument(module, prefix)) // 2


# leading parameters, in order (execute_model reads dummy_run by position too)
EXPECTED = {
    "load_model": ["self"],
    "profile_run": ["self"],
    # profile_only names the profiling capture; builds without it (vLLM in
    # vllm/vllm-openai:deepseekv41-flash-0909) capture only for serving
    "capture_model": ["self"],
    "execute_model": ["self", "scheduler_output", "intermediate_tensors", "dummy_run"],
    "sample_tokens": ["self", "grammar_output"],
}


def check_signatures(R: Any) -> List[str]:
    """Problems with the methods the adapter wraps (empty when all match)."""
    problems = []
    for name, want in EXPECTED.items():
        fn = getattr(R, name, None)
        if fn is None:
            problems.append(f"{name}: missing")
            continue
        params = list(inspect.signature(fn).parameters)
        if params[:len(want)] != want:
            problems.append(f"{name}: parameters {params[:len(want) + 2]}, expected to start with {want}")
    return problems


def _patch_breakable() -> Optional[str]:
    try:
        from vllm.compilation import breakable_cudagraph as bc
    except Exception as e:  # noqa: BLE001
        return f"breakable_cudagraph not importable ({e})"
    C = bc.BreakableCUDAGraphCapture
    if getattr(C, "_ruppell_patched", False):
        return None
    params = list(inspect.signature(C.add_eager).parameters)
    if params != ["self", "fn"]:
        return f"BreakableCUDAGraphCapture.add_eager: parameters {params}, expected ['self', 'fn']"
    orig = C.add_eager

    def add_eager(self, fn):
        from ruppell.api import current_scope, pop_id, push_id, scope_id

        if _state["mode"] != "active":
            return orig(self, fn)
        name = current_scope() or "eager_break"
        # defined now; at every replay only the push (this runs per layer per step)
        sid = scope_id(name, op="eager_break", seq=self._num_eager_breaks)

        def scoped():
            push_id(sid, name)
            try:
                return fn()
            finally:
                pop_id()

        return orig(self, scoped)

    C.add_eager = add_eager
    C._ruppell_patched = True
    return None


def patch_runner(mod: Any) -> None:
    import ruppell

    R = mod.GPUModelRunner
    if getattr(R, "_ruppell_patched", False):
        return
    problems = check_signatures(R)
    if problems:
        _log("vllm adapter NOT active in this process; hooks do not match the runner: " + "; ".join(problems))
        return
    problem = _patch_breakable()
    if problem:
        _log(f"eager breaks will not be scoped: {problem}")
    orig_load, orig_profile = R.load_model, R.profile_run
    orig_capture, orig_exec, orig_sample = R.capture_model, R.execute_model, R.sample_tokens
    orig_shutdown = getattr(R, "shutdown", None)

    def load_model(self, *args, **kwargs):
        if _state["mode"] is None:
            _state["mode"] = ruppell.init(schema="serving")
            _state["sampling"] = _sampling()
            _state["detail_kinds"] = _detail_kinds()
            _state["counter_steps"] = _state["mode"] == "active" and ruppell.counter_steps() is not None
            _patch_graph_capture()
        sid = NAMED_BASE + 0
        ruppell.step_begin(sid, phase="load_model")
        try:
            with _runner_scope("load_model"):
                out = orig_load(self, *args, **kwargs)
        finally:
            ruppell.step_end(sid)
        if _state["mode"] == "active":
            ctx = _context(self)
            # RUPPELL_VLLM_HOOKS=0: no module hooks (a diagnostic of their cost:
            # kernels lose their module scopes)
            hooks = os.environ.get("RUPPELL_VLLM_HOOKS") != "0"
            n_t = _instrument(self.model, "model") if hooks else 0
            drafter = getattr(getattr(self, "speculator", None), "model", None)
            n_d = _instrument(drafter, "drafter") if hooks and drafter is not None and hasattr(drafter, "named_modules") else 0
            spec = getattr(self, "speculator", None)
            if hooks and spec is not None and callable(getattr(spec, "propose", None)):
                spec.propose = _with_scope(spec.propose, "drafter.propose")
            ctx["hooked_modules"] = {"target": n_t, "drafter": n_d}
            if _state.get("compiled"):  # compiled regions' scopes take the hooks' names
                from ruppell.adapters.vllm import compiled

                compiled.set_root(self.model, "model")
            ruppell.context(**ctx)
            _log(f"vllm adapter active: rank {ctx['rank']}/{ctx['tp']}, {n_t} target and {n_d} drafter modules hooked, "
                 f"cudagraph_mode {ctx['cudagraph_mode']}")
            if getattr(self.vllm_config, "shutdown_timeout", None) == 0:
                # vLLM then SIGKILLs the engine core's process tree at once on
                # SIGINT/SIGTERM (kill_process_tree), workers included, mid-flush
                _log("vllm shutdown_timeout is 0: at shutdown the workers are killed before their last flush and "
                     "lose their last seconds of records; serve with --shutdown-timeout 60")
        return out

    def capture_model(self, *args, **kwargs):
        _state["pass"] += 1
        _state["graph"] = 0
        mgr = getattr(self, "cudagraph_manager", None)
        if mgr is not None:
            mgr.capture = _with_role(mgr.capture, "target")
        spec = getattr(self, "speculator", None)
        if spec is not None and hasattr(spec, "capture"):
            spec.capture = _with_role(spec.capture, "drafter")
        profile_only = kwargs.get("profile_only", False)
        out = _named("capture_model_profile" if profile_only else "capture_model", orig_capture)(self, *args, **kwargs)
        if _state.get("compiled") and _state["mode"] == "active":
            from ruppell.adapters.vllm import compiled

            s = compiled.summary()
            if s["programs"]:
                _log(f"compiled regions: {s['programs']} programs, {s['sites']} launch sites, "
                     f"{s['pieces']} pieces scoped, sites placed by {s['resolved']}, splitting ops scoped: "
                     f"{s['inlined']} inlined, {s['splitting']} as modules")
        return out

    def execute_model(self, scheduler_output, *args, **kwargs):
        dummy = kwargs.get("dummy_run", args[1] if len(args) > 1 else False)
        if dummy:
            return orig_exec(self, scheduler_output, *args, **kwargs)
        _close_open()
        fields = step_fields(scheduler_output)
        if _state["warmup"]:
            # vLLM's startup warmup drives execute_model with made-up batches:
            # a named step, not an engine step of the served workload
            _state["named"] += 1
            sid = NAMED_BASE + _state["named"]
            fields["phase"] = "warmup"
        else:
            _state["step"] += 1
            sid = _state["step"]
            g = _counter_step(sid, _sample(sid, fields.get("kind")))
            if g is not None:
                fields["counters"] = g
        ruppell.step_begin(sid, **fields)
        _state["open"] = sid
        with _runner_scope("execute_model"):
            out = orig_exec(self, scheduler_output, *args, **kwargs)
        if out is not None:
            _close_open()
        return out

    def sample_tokens(self, *args, **kwargs):
        try:
            with _runner_scope("sample_tokens"):
                return orig_sample(self, *args, **kwargs)
        finally:
            _close_open()

    def shutdown(self, *args, **kwargs):
        if _state["mode"] == "active":
            _close_open()
            # the worker's last flush, marked complete once it returns: a worker
            # killed before then reads as incomplete rather than as 0 dropped
            from ruppell.adapters.vllm import collectives
            collectives.flush_timing()
            ruppell.flush(complete=True)
        return orig_shutdown(self, *args, **kwargs)

    R.load_model = load_model
    if orig_shutdown is not None:
        R.shutdown = shutdown
    R.profile_run = _named("profile_run", orig_profile)
    R.capture_model = capture_model
    R.execute_model = execute_model
    R.sample_tokens = sample_tokens
    R._ruppell_patched = True
    _log(f"patched {TARGET}.GPUModelRunner")


def patch_warmup(mod: Any) -> None:
    """Flag vLLM's startup warmup (`_warmup_kernels`, which drives
    execute_model and sample_tokens with made-up batches), so its steps are
    named `warmup` rather than counted as engine steps."""
    # `_warmup_kernels`, or `warmup_kernels` in later builds (which the worker
    # imports by name: there Worker.compile_or_warm_up_model, wrapped as a
    # whole by patch_worker, is what names these steps)
    attr = "_warmup_kernels" if hasattr(mod, "_warmup_kernels") else "warmup_kernels"
    fn = getattr(mod, attr, None)
    if fn is None or getattr(fn, "_ruppell_wrapped", False):
        if fn is None:
            _log(f"{WARMUP}: no _warmup_kernels or warmup_kernels; startup warmup is named only through the worker")
        return
    params = list(inspect.signature(fn).parameters)
    if params[:3] != ["model_runner", "worker_execute_model", "worker_sample_tokens"]:
        _log(f"{WARMUP}.{attr}: parameters {params}; startup warmup is named only through the worker")
        return

    setattr(mod, attr, _as_warmup(fn))


def _as_warmup(fn):
    """fn runs as startup warmup: execute_model inside it opens named warmup steps."""

    def wrapped(*args, **kwargs):
        prev = _state["warmup"]
        _state["warmup"] = True
        try:
            return fn(*args, **kwargs)
        finally:
            _close_open()
            _state["warmup"] = prev

    wrapped._ruppell_wrapped = True
    return wrapped


def patch_worker(mod: Any) -> None:
    """All of the worker's `compile_or_warm_up_model` is startup: besides
    `_warmup_kernels` it runs `kernel_warmup`, which also drives
    execute_model (job 48796: three such steps counted as engine steps 1-3)."""
    W = getattr(mod, "Worker", None)
    fn = getattr(W, "compile_or_warm_up_model", None) if W is not None else None
    if fn is None:
        _log(f"{WORKER}.Worker.compile_or_warm_up_model not found: startup steps may count as engine steps")
        return
    if not getattr(fn, "_ruppell_wrapped", False):
        W.compile_or_warm_up_model = _as_warmup(fn)


# ---- installation ---------------------------------------------------------------------------------

class _PostImport(importlib.abc.MetaPathFinder):
    """Patch a module right after it is imported, so processes that never
    import it (the API server, the engine core) are left alone."""

    def __init__(self):
        self._busy = False

    def find_spec(self, name, path, target=None):
        if name not in PATCHES or self._busy:
            return None
        self._busy = True
        try:
            spec = importlib.util.find_spec(name)
        finally:
            self._busy = False
        if spec is None or spec.loader is None:
            return None
        loader = spec.loader
        orig_exec = loader.exec_module

        patch = PATCHES[name]

        def exec_module(module):
            orig_exec(module)
            patch(module)

        loader.exec_module = exec_module
        return spec


PATCHES = {TARGET: patch_runner, WARMUP: patch_warmup, WORKER: patch_worker}


def register() -> None:
    """vLLM general-plugin entry point."""
    checks = [c for c in os.environ.get("RUPPELL_CHECK", "").split(",") if c]
    if checks and not _state.get("check"):  # in-server checks (ruppell.checks), traced or not
        _state["check"] = True
        for check in checks:
            importlib.import_module(f"ruppell.checks.{check}").install()
    if os.environ.get("RUPPELL_VLLM") != "1" or _state["patched"]:
        return
    _state["patched"] = True
    from ruppell.adapters import tuning

    if tuning.enabled():  # RUPPELL_TUNING: run-time tuning pinned across processes and servers
        PATCHES.update(tuning.PATCHES)
    from ruppell.adapters.vllm import collectives

    if collectives.enabled():  # each collective call beside the kernels' scope (roadmap-v2 M8)
        PATCHES.update(collectives.patches())
        collectives.patch_torch_distributed()
        collectives.patch_capture_count()
    from ruppell.adapters.vllm import compiled

    if compiled.enabled():  # scopes inside vLLM's compiled regions (runner ruppell-accuracy plan 06)
        _state["compiled"] = True
        PATCHES.update(compiled.patches())
    pending = False
    for name, patch in PATCHES.items():
        mod = sys.modules.get(name)
        if mod is not None:
            patch(mod)
        else:
            pending = True
    if pending:
        sys.meta_path.insert(0, _PostImport())
