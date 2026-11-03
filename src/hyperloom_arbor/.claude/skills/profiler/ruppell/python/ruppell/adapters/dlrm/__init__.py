"""The DLRMv4 training adapter (runner plans/dlrmv4 Plan 01 section 4.3): steps,
scopes and run fields from inside the training loop of
chriscai-amd/training_dlrmv4 (generative_recommenders.dlrm_v4), with no change
to its code.

Installed in every Python process by `site/sitecustomize.py` (that directory on
PYTHONPATH: the trainer spawns its ranks, so only start-up code reaches them);
it does nothing unless RUPPELL_DLRM=1. Each patch is applied right after its
module is imported, after its signature is checked; a mismatch leaves the
process untraced and says which hook and why.

generative_recommenders.dlrm_v4.train.utils
- `setup` (process group up, so HIP is up): ruppell.init; a named step
  `startup` opens and stays open until the first training step (model build,
  sharding plan, data loaders).
- `make_optimizer_and_shard`: the model it returns gets module scopes
  (`model.*`, DMP/DDP wrappers peeled); run fields recorded.
- `build_train_pipeline`: on the pipeline it returns (one class per call,
  defined inside the function, so the instance is patched):
  - `progress` with grad enabled is one training step (phase "engine", kind
    "train", as serving's engine steps, schemas/training.json): step N opens (GPU
    markers on the current stream) when call N begins and closes when call N+1
    begins, so the loop's own work after progress (metrics, the step fence)
    stays in its iteration. With grad disabled it is eval (the eval pipeline,
    under torch.no_grad): no step of its own.
  - `detach` while training: a named step `eval` until the next training step.
  - scopes `pipeline.<method>` on progress, _optimizer_step, copy_batch_to_gpu,
    start_sparse_data_dist, wait_sparse_data_dist, dequeue_batch, _backward.
- `_window_boundary_barrier`: a named step `window_boundary`.
- `cleanup`: a named step `cleanup`, then the records are flushed.
generative_recommenders.modules.stu
- `STULayer.forward_targets_only`, `STUStack.forward_targets_only`: they
  bypass nn.Module.__call__ (so forward hooks miss them); they run inside the
  scope their __call__ would have pushed.
generative_recommenders.dlrm_v4.utils
- `MetricsLogger.update`, `MetricsLogger.compute_and_log`: scopes `loop.metrics.*`.
torch.cuda.CUDAGraph.capture_begin: none is expected in this loop; a capture is
logged loudly and recorded in the run fields (`graph_captured`).

Backward kernels are attributed by the analysis from the CPU op tree's autograd
sequence numbers (RUPPELL_CPU_OPS; engine._backward_scopes): they need
RUPPELL_CPU_OPS on in the detailed steps.

RUPPELL_DLRM_DETAIL=A:B: kernel detail on training steps A..B only (markers
and fields on every step); default: every step.
"""

from __future__ import annotations

import hashlib
import importlib.abc
import importlib.util
import inspect
import os
import sys
import threading
from typing import Any, Dict, List, Optional

UTILS = "generative_recommenders.dlrm_v4.train.utils"
STU = "generative_recommenders.modules.stu"
METRICS = "generative_recommenders.dlrm_v4.utils"
MLPERF = "generative_recommenders.dlrm_v4.train.mlperf_logging_utils"
NAMED_BASE = 1 << 40
PIPELINE_SCOPES = ("progress", "_optimizer_step", "copy_batch_to_gpu", "start_sparse_data_dist",
                   "wait_sparse_data_dist", "dequeue_batch", "_backward")
# The trainer's own functions between setup() and the first train step, and its
# loops: their collectives (the seed broadcast, the loop's start-up broadcasts)
# wait for the slowest rank, up to 267 ms of a 397 ms startup step on one rank
# with no scope (Plan 02, job 58992). Scopes inside them still win.
LOOP_FUNCTIONS = ("seed_everything", "make_model", "make_train_test_dataloaders", "decorrelate_runtime_rng",
                  "train_loop", "eval_loop", "train_eval_loop", "streaming_train_eval_loop")

_lock = threading.Lock()
_state: Dict[str, Any] = {"installed": False, "mode": None, "step": 0, "named": 0, "open": None,
                          "detail": None, "train_pipelines": 0, "graph_captured": False, "rank": None}


def _log(msg: str) -> None:
    print(f"[ruppell] pid {os.getpid()}: {msg}", file=sys.stderr, flush=True)


def _active() -> bool:
    """Scopes and hooks: only while tracing. In witness mode (RUPPELL_MODE=witness,
    no runtime loaded) the steps still launch their markers, so the program
    matches a traced run's kernel for kernel, and nothing else runs."""
    return _state["mode"] == "active"


def _detail_window():
    v = os.environ.get("RUPPELL_DLRM_DETAIL", "")
    if not v:
        return None
    a, _, b = v.partition(":")
    return int(a), int(b)


# ---- steps -------------------------------------------------------------------------------------

def _close_open() -> None:
    import ruppell

    sid = _state["open"]
    if sid is not None:
        _state["open"] = None
        ruppell.step_end(sid)
        if _state.get("counting"):
            _state["counting"] = False
            ruppell.counters(False)


def _open_named(phase: str, **fields: Any) -> None:
    import ruppell

    _close_open()
    _state["named"] += 1
    sid = NAMED_BASE + _state["named"]
    ruppell.step_begin(sid, phase=phase, **fields)
    _state["open"] = sid


def _open_train(pipe: Any) -> None:
    import ruppell

    _close_open()
    _state["step"] += 1
    n = _state["step"]
    win = _state["detail"]
    if win is not None:
        ruppell.detail(win[0] <= n <= win[1])
    ctx0 = (getattr(pipe, "contexts", None) or [None])[0]
    index = getattr(ctx0, "index", None)
    index = int(index) if isinstance(index, int) else -1
    # phase "engine": the workload's main step, as for serving, so the cross-rank
    # merge, the census and the exports treat training steps the same way
    fields = {"phase": "engine", "kind": "train", "step": n, "batch_index": index,
              "fingerprint": hashlib.sha1(f"{n}:{index}".encode()).hexdigest()[:16]}
    # hardware counters (gray-parrot): the steps RUPPELL_COUNTER_STEPS picks, outside the detail
    # window; on before the begin marker, off after the end marker (_close_open)
    if ruppell.is_counter_step(n) and not (win is not None and win[0] <= n <= win[1]):
        g = ruppell.counters(True, step=n)
        if g >= 0:
            fields["counters"] = g
            _state["counting"] = True
    ruppell.step_begin(n, **fields)
    _state["open"] = n


# ---- scopes --------------------------------------------------------------------------------------

def _scoped(name: str, fn, **attrs):
    from ruppell import api

    sid_box: List[int] = []

    def wrapped(*args, **kwargs):
        if not _active():
            return fn(*args, **kwargs)
        if not sid_box:
            sid_box.append(api.scope_id(name, **attrs))
        # pop what this call pushed, whatever detail says by then: the loop
        # functions' scopes span the detail toggles of the steps inside them
        # (push_id and pop_id each skip while detail is off)
        pushed = api._fine_scopes
        if pushed:
            api.push_id(sid_box[0], name)
        try:
            return fn(*args, **kwargs)
        finally:
            if pushed:
                api.scope_pop()

    wrapped._ruppell_wrapped = True
    return wrapped


def _peel(model: Any) -> Any:
    m = model
    for _ in range(6):
        inner = getattr(m, "module", None)
        if inner is None or inner is m or not hasattr(inner, "named_modules"):
            break
        m = inner
    return m


def _instrument(model: Any) -> int:
    from ruppell import torch_hooks

    core = _peel(model)
    handles = torch_hooks.instrument(core, "model")
    return len(handles) // 2


# ---- patches -------------------------------------------------------------------------------------

def _check(fn: Any, want: List[str], where: str) -> Optional[str]:
    if fn is None:
        return f"{where}: not found"
    try:
        params = list(inspect.signature(fn).parameters)
    except (TypeError, ValueError) as e:
        return f"{where}: no signature ({e})"
    if params[:len(want)] != want:
        return f"{where}: parameters {params[:len(want)]}, expected {want}"
    return None


def patch_utils(mod: Any) -> None:
    import ruppell

    if getattr(mod, "_ruppell_patched", False):
        return
    problems = [p for p in (
        _check(getattr(mod, "setup", None), ["rank", "world_size", "master_port", "device"], "setup"),
        _check(getattr(mod, "make_optimizer_and_shard", None), ["model", "device", "world_size"],
               "make_optimizer_and_shard"),
        _check(getattr(mod, "build_train_pipeline", None), ["model", "optimizer", "device"], "build_train_pipeline"),
        _check(getattr(mod, "cleanup", None), [], "cleanup"),
    ) if p]
    if problems:
        _log("dlrm adapter NOT active in this process; hooks do not match the trainer: " + "; ".join(problems))
        return
    orig_setup, orig_shard = mod.setup, mod.make_optimizer_and_shard
    orig_build, orig_cleanup = mod.build_train_pipeline, mod.cleanup
    orig_boundary = getattr(mod, "_window_boundary_barrier", None)
    # the startup and window-boundary steps run no module code, so their kernels
    # (sharding, the pipeline's construction, the barrier's wait) had no scope:
    # 0.2-0.9 s per rank, against 5.5-7.0 ms unattributed inside 30 train steps
    # (jobs 57972, 58091)
    scoped_shard = _scoped("loop.make_optimizer_and_shard", orig_shard, op="loop")
    scoped_build = _scoped("loop.build_train_pipeline", orig_build, op="loop")
    scoped_boundary = (_scoped("loop.window_boundary_barrier", orig_boundary, op="collective")
                       if orig_boundary is not None else None)

    def setup(rank, world_size, *args, **kwargs):
        out = orig_setup(rank, world_size, *args, **kwargs)
        if _state["mode"] is None:
            _state["mode"] = ruppell.init(schema="training")
            _state["rank"] = rank
            _state["detail"] = _detail_window()
            _patch_graph_capture()
            ruppell.context(rank=rank, world_size=world_size, role="trainer",
                            knobs={k: v for k, v in os.environ.items()
                                   if k.startswith(("DLRM_HSTU_FP8", "HSTU_ATTN", "START_TS", "NUM_TRAIN",
                                                    "NUM_EVAL", "EVAL_EVERY", "METRIC_LOG_FREQ", "TRAIN_PIPELINE",
                                                    "TP_", "RUPPELL_DLRM"))})
            _open_named("startup")
            _log(f"dlrm adapter active: rank {rank}/{world_size}, mode {_state['mode']}")
        return out

    def make_optimizer_and_shard(model, *args, **kwargs):
        out = scoped_shard(model, *args, **kwargs)
        if _state["mode"] == "active" and os.environ.get("RUPPELL_DLRM_HOOKS") != "0":
            dmp = out[0] if isinstance(out, tuple) else out
            n = _instrument(dmp)
            ruppell.context(hooked_modules=n, model_class=type(_peel(dmp)).__name__)
            _log(f"dlrm adapter: {n} modules hooked")
        return out

    def build_train_pipeline(*args, **kwargs):
        pipe = scoped_build(*args, **kwargs)
        if _state["mode"] is None:
            return pipe
        _state["train_pipelines"] += 1
        for meth in PIPELINE_SCOPES:
            fn = getattr(pipe, meth, None)
            if fn is not None and not getattr(fn, "_ruppell_wrapped", False):
                setattr(pipe, meth, _scoped(f"pipeline.{meth}", fn, op="pipeline"))
        scoped_progress = pipe.progress
        orig_detach = pipe.detach

        def progress(dataloader_iter):
            import torch

            if torch.is_grad_enabled():
                _open_train(pipe)
            return scoped_progress(dataloader_iter)

        def detach(*a, **k):
            out = orig_detach(*a, **k)
            import torch

            if torch.is_grad_enabled() and _state["open"] is not None and _state["open"] < NAMED_BASE:
                _open_named("eval")
            return out

        progress._ruppell_wrapped = True
        pipe.progress = progress
        pipe.detach = detach
        return pipe

    def cleanup(*args, **kwargs):
        if _state["mode"] is None:
            return orig_cleanup(*args, **kwargs)
        _open_named("cleanup")
        try:
            return orig_cleanup(*args, **kwargs)
        finally:
            _close_open()
            if _state["mode"] == "active":
                ruppell.flush()

    for name in LOOP_FUNCTIONS:
        fn = getattr(mod, name, None)
        if callable(fn) and not getattr(fn, "_ruppell_wrapped", False):
            setattr(mod, name, _scoped(f"loop.{name}", fn, op="loop"))
    mod.setup = setup
    mod.make_optimizer_and_shard = make_optimizer_and_shard
    mod.build_train_pipeline = build_train_pipeline
    mod.cleanup = cleanup
    if orig_boundary is not None:
        def boundary(*args, **kwargs):
            if _state["mode"] is not None:
                _open_named("window_boundary")
            return scoped_boundary(*args, **kwargs)

        mod._window_boundary_barrier = boundary
    mod._ruppell_patched = True
    _log(f"patched {UTILS}")


def patch_stu(mod: Any) -> None:
    from ruppell import api

    problems = [p for p in (
        _check(getattr(getattr(mod, "STULayer", None), "forward_targets_only", None),
               ["self", "x"], "STULayer.forward_targets_only"),
        _check(getattr(getattr(mod, "STUStack", None), "forward_targets_only", None),
               ["self", "x"], "STUStack.forward_targets_only"),
    ) if p]
    if problems:
        _log("STU targets-only paths will not be scoped: " + "; ".join(problems))
        return
    for cls in (mod.STULayer, mod.STUStack):
        orig = cls.forward_targets_only
        if getattr(orig, "_ruppell_wrapped", False):
            continue

        def wrapped(self, *args, _orig=orig, **kwargs):
            path = getattr(self, "_ruppell_scope", None)
            if not _active() or path is None or not api._fine_scopes:
                return _orig(self, *args, **kwargs)
            shp = tuple(args[0].shape) if args and hasattr(args[0], "shape") else None
            ids = self.__dict__.setdefault("_ruppell_tgt_ids", {})
            sid = ids.get(shp)
            if sid is None:
                sid = ids[shp] = api.scope_id(path, op=type(self).__name__, shape=list(shp) if shp else None)
            api.push_id(sid, path)
            try:
                return _orig(self, *args, **kwargs)
            finally:
                api.pop_id()

        wrapped._ruppell_wrapped = True
        cls.forward_targets_only = wrapped
    _log(f"patched {STU} targets-only paths")


def patch_metrics(mod: Any) -> None:
    C = getattr(mod, "MetricsLogger", None)
    if C is None:
        _log(f"{METRICS}.MetricsLogger not found: metric work will carry only the step's scope")
        return
    for meth in ("update", "compute_and_log"):
        fn = getattr(C, meth, None)
        if fn is not None and not getattr(fn, "_ruppell_wrapped", False):
            setattr(C, meth, _scoped(f"loop.metrics.{meth}", fn, op="metrics"))


def patch_mlperf(mod: Any) -> None:
    # MLPerfLogger barriers all ranks before its synchronized events (INIT_STOP and
    # RUN_START in start-up, through the module's _barrier(), looked up at call
    # time): the last start-up collectives with no scope, up to 256 ms of a rank's
    # wait (Plan 03, job 59201)
    fn = getattr(mod, "_barrier", None)
    if callable(fn) and not getattr(fn, "_ruppell_wrapped", False):
        mod._barrier = _scoped("loop.mlperf_barrier", fn, op="collective")


def _patch_graph_capture() -> None:
    import torch

    G = torch.cuda.CUDAGraph
    if getattr(G.capture_begin, "_ruppell_wrapped", False):
        return
    orig = G.capture_begin

    def capture_begin(self, *args, **kwargs):
        import ruppell

        if not _state["graph_captured"]:
            _state["graph_captured"] = True
            _log("WARNING: a CUDA/HIP graph is being captured; the DLRMv4 adapter expects none and does not key it")
            ruppell.context(graph_captured=True)
        return orig(self, *args, **kwargs)

    capture_begin._ruppell_wrapped = True
    G.capture_begin = capture_begin


# ---- installation ---------------------------------------------------------------------------------

class _PostImport(importlib.abc.MetaPathFinder):
    """Patch a module right after it is imported."""

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


PATCHES = {UTILS: patch_utils, STU: patch_stu, METRICS: patch_metrics, MLPERF: patch_mlperf}


def install() -> None:
    """Called by site/sitecustomize.py in every process when RUPPELL_DLRM=1."""
    if os.environ.get("RUPPELL_DLRM") != "1" or _state["installed"]:
        return
    _state["installed"] = True
    pending = False
    for name, patch in PATCHES.items():
        mod = sys.modules.get(name)
        if mod is not None:
            patch(mod)
        else:
            pending = True
    if pending:
        sys.meta_path.insert(0, _PostImport())
