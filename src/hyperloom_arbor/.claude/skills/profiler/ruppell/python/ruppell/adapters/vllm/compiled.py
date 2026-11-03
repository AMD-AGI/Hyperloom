"""Scopes inside vLLM's compiled regions (runner ruppell-accuracy plan 06, Part C).

vLLM's compiled path (-O2) splits the backbone into pieces: compiled FX subgraphs, each run by a
`PiecewiseBackend`, and between them the ops it splits at (attention). Module hooks do not run
inside a compiled region (torch_hooks.py), so without this a piece's kernels carry the scope
around the region (`model`). With it:

- **Pieces:** each `PiecewiseBackend` call is a scope, `compiled.piece`, with its compile index.
- **Launch sites:** Inductor's generated program for a piece (its `call`) launches each kernel
  or extern call after a comment naming the site's source FX nodes and ATen ops. When a program
  is loaded (`PyCodeCache.load_by_key_path`), its `call` is rewritten in memory, once, to enter
  a scope at every site and leave it at the next or at the return; the file is untouched. The
  scope is the module the site's FX nodes come from, read from the running piece's own FX graph
  (`nn_module_stack`), so pieces that share one program (vLLM's identical middle pieces) each
  get their own layer. A site whose comment names no node takes the FX nodes of its ATen op in
  order (vLLM's IR ops, `vllm.moe_forward`), or else the first node no site names between its
  neighbours (`how` says which; `covers` lists the modules of a site's other nodes). Scopes are named as module hooks name them (`model.` and the
  module's path in the model), so compiled and uncompiled runs compare module by module.
- **Splitting ops:** the ops vLLM splits at (attention, MLA, the sparse indexers) run between
  pieces, each in a splitting submodule the stitching code calls as a module: a forward hook on
  it enters the scope of the op's module. Attention layers also look their layer up
  (`get_attention_context`), and the layer's impl (`forward`, `do_kv_cache_update`) runs in a
  scope named by the layer, as its module hook names it.

Graph capture records these scopes like any other, so replays get them through the registry;
replays run no Python. The scopes need the pieces' FX graphs, which a vLLM compile-cache hit does
not load, and Inductor's site comments: so this sets `VLLM_DISABLE_COMPILE_CACHE=1` (unless
set) and Inductor's provenance tracking (`trace.provenance_tracking_level = 1`), which adds the
site handles as comments. With vLLM's cache off, Triton's cache directory (which vLLM sets under
its cache) is set where Inductor would set it, so a second start finds vLLM's own Triton kernels
(`pin_triton_cache`). RUPPELL_VLLM_COMPILED=0 turns it off.
"""

from __future__ import annotations

import ast
import os
import re
import sys
import textwrap
import threading
from typing import Any, Dict, List, Optional, Tuple

from ruppell import api

PIECES = "vllm.compilation.piecewise_backend"
BACKENDS = "vllm.compilation.backends"
CODEGEN = "vllm.compilation.codegen"
CODECACHE = "torch._inductor.codecache"
ATTENTION = ("vllm.model_executor.layers.attention.attention", "vllm.attention.layer")

_tls = threading.local()
_lock = threading.Lock()
_programs: List[List["Site"]] = []        # program id -> its launch sites, in order
_pieces: set = set()                      # the compile indices of the pieces scoped
_root = {"paths": None, "types": {}, "by_suffix": {}}
_stats = {"programs": 0, "sites": 0, "how": {}}


def _log(msg: str) -> None:
    print(f"[ruppell] pid {os.getpid()}: compiled: {msg}", file=sys.stderr, flush=True)


def enabled() -> bool:
    return os.environ.get("RUPPELL_VLLM_COMPILED") != "0"


class Site:
    __slots__ = ("handle", "sources", "aten")

    def __init__(self, handle: str, sources: List[str], aten: List[str]):
        self.handle, self.sources, self.aten = handle, sources, aten

    def __repr__(self):
        return f"Site({self.handle!r}, {self.sources}, {self.aten})"


# ---- the generated program ---------------------------------------------------------------------

_SRC = re.compile(r"^(\s*)# (?:Topologically Sorted|Unsorted) Source Nodes: \[(.*?)\], Original ATen: \[(.*?)\]\s*$")
_HANDLE = re.compile(r"^\s*# \[Provenance debug handles\] (\S+)")
_RETURN = re.compile(r"^(\s*)return\b")


def _names(s: str) -> List[str]:
    return [x.strip() for x in s.split(",")]


def _call_def(tree: ast.Module) -> Optional[Tuple[Optional[str], ast.FunctionDef]]:
    """the program's call: Runner.call (torch 2.13), or a module-level call (older)"""
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Runner":
            for f in node.body:
                if isinstance(f, ast.FunctionDef) and f.name == "call":
                    return "Runner", f
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "call":
            return None, node
    return None


def rewrite(source: str, prog: int) -> Optional[Tuple[str, Optional[str], List[Site]]]:
    """The program's call, with `_ruppell_enter()` first, `_ruppell_site(prog, k)` at every
    launch site k and `_ruppell_site_end()` before every return; its owner ("Runner" or None),
    and the sites. None when the program has no call or no sites."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    found = _call_def(tree)
    if found is None:
        return None
    owner, fn = found
    lines = source.splitlines()[fn.lineno - 1:fn.end_lineno]
    first = fn.body[0].lineno - fn.lineno       # the body's first statement: the call's frame opens there
    out, sites = [], []
    for i, ln in enumerate(lines):
        if i == first:
            out.append(" " * fn.body[0].col_offset + "_ruppell_enter()")
        m = _SRC.match(ln)
        if m:
            handle = ""
            for nxt in lines[i + 1:i + 3]:
                h = _HANDLE.match(nxt)
                if h:
                    handle = h.group(1)
                    break
            sites.append(Site(handle, _names(m.group(2)), [a for a in _names(m.group(3)) if a]))
            out.append(f"{m.group(1)}_ruppell_site({prog}, {len(sites) - 1})")
        r = _RETURN.match(ln)
        if r and i > 0:
            out.append(f"{r.group(1)}_ruppell_site_end()")
        out.append(ln)
    if not sites:
        return None
    body = textwrap.dedent("\n".join(out)).replace("def call(", "def _ruppell_call(", 1)
    return body + "\n", owner, sites


def install(mod: Any) -> bool:
    """Rewrite a loaded program's call in place (once per module object)."""
    if getattr(mod, "_ruppell_program", None) is not None or not hasattr(mod, "call"):
        return False
    path = getattr(mod, "__file__", None)
    try:
        source = open(path).read() if path else None
    except OSError:
        source = None
    if not source or "Source Nodes:" not in source:
        mod._ruppell_program = -1
        return False
    with _lock:
        prog = len(_programs)
        res = rewrite(source, prog)
        if res is None:
            mod._ruppell_program = -1
            return False
        body, owner, sites = res
        _programs.append(sites)
        _stats["programs"] += 1
        _stats["sites"] += len(sites)
    g = mod.__dict__
    g["_ruppell_enter"] = enter
    g["_ruppell_site"] = site
    g["_ruppell_site_end"] = site_end
    exec(compile(body, f"{path}#ruppell", "exec"), g)
    new = g.pop("_ruppell_call")
    if owner == "Runner":
        mod.Runner.call = new
        runner = getattr(mod, "runner", None)
        if runner is not None:
            mod.call = runner.call
    else:
        mod.call = new
    mod._ruppell_program = prog
    return True


# ---- at run time ---------------------------------------------------------------------------------

def _frames() -> List[bool]:
    """per program call running on this thread, innermost last: whether it has a site open"""
    f = getattr(_tls, "frames", None)
    if f is None:
        f = _tls.frames = []
    return f


def enter() -> None:
    """A program's call begins: a frame for its sites."""
    _frames().append(False)


def site(prog: int, k: int) -> None:
    """Enter launch site k of program prog, leaving this call's previous site. Only a piece's own
    program, one frame above the piece, places its sites: a program called from inside a site
    (a custom op's own compiled code: model-w's moe_forward_shared runs its shared experts so) is
    no piece's, and its kernels keep the scope around it."""
    f = _frames()
    if not f:
        return
    if f[-1]:
        f[-1] = False
        api.pop_id()
    piece = getattr(_tls, "piece", None)
    if piece is None or len(f) != getattr(_tls, "base", 0) + 1:
        return
    tables = piece.__dict__.setdefault("_ruppell_tables", {})   # on the piece: ids get reused
    table = tables.get(prog)
    if table is None:
        table = tables[prog] = _table(piece, prog)
        _pieces.add(_index(piece))
    ent = table[k] if k < len(table) else None
    if ent is None:
        return
    api.push_id(ent[0], ent[1])
    f[-1] = True


def site_end() -> None:
    """A program's call returns: its open site closes, and its frame."""
    f = _frames()
    if f and f.pop():
        api.pop_id()


def _unwind(base: int) -> None:
    """Frames a raising call left behind, down to base"""
    f = _frames()
    while len(f) > base:
        if f.pop():
            api.pop_id()


def _fqn(v: Any) -> str:
    """an nn_module_stack entry's module path, as layers.3.self_attn"""
    s = v[0] if isinstance(v, (tuple, list)) else str(v)
    s = re.sub(r"\[(\d+)\]", r".\1", str(s))
    s = re.sub(r"\._modules\['([^']+)'\]", r".\1", s)
    return re.sub(r"^L\['self'\]\.?", "", s)



def _site_ops(s: Site) -> List[str]:
    """The site's ATen ops to match, its own first: a custom-op site's handle names the op it
    calls (torch.ops.vllm.moe_forward.default:4), and its comment can list the ops feeding it
    before it ([vllm_ir.fused_add_rms_norm, vllm.moe_forward], MI350X Qwen3-30B-A3B TP2)."""
    ops = list(s.aten)
    h = s.handle.rsplit(":", 1)[0]
    if h.startswith("torch.ops."):
        own = ".".join(h[len("torch.ops."):].split(".")[:2])
        ops = [own] + [o for o in ops if o != own]
    return ops


def resolve(sites: List[Site], nodes: List[Tuple[str, str, str]]) -> List[Optional[Tuple[str, str, Tuple[str, ...]]]]:
    """Each site's module path (relative to the compiled module), how it was found, and the
    modules of the nodes it covers when more than one, from the piece's FX nodes in graph order,
    (name, target, module path), computation nodes only:
    - "source": the module of the site's last source node. Inductor lists a site's sources
      topologically sorted, and with them the views and pointwise nodes that feed it (a GEMM's
      sources are [view, linear], or [silu, mul, linear_2]), so the last is the node whose output
      the site produces;
    - "op": the next unclaimed node of the site's ATen op, by name (vLLM's IR ops and custom
      ops, whose nodes the site's comment does not name);
    - "between": the first of the unclaimed nodes between its placed neighbours, the op the
      kernel starts with (vLLM's fused all-reduce and norm goes to the all-reduce's module, as
      the all-reduce does uncompiled; Qwen3-30B-A3B TP2, B200, 2026-10-23);
    None when nothing places it (its kernels keep the piece's scope)."""
    pos = {n: i for i, (n, _, _) in enumerate(nodes)}
    mod = {n: m for n, _, m in nodes}
    claimed = set()
    out: List[Optional[Tuple[str, str, Tuple[str, ...]]]] = [None] * len(sites)
    span: List[Optional[Tuple[int, int]]] = [None] * len(sites)
    used: Dict[str, int] = {}
    for k, s in enumerate(sites):
        named = [n for n in s.sources if n in mod]
        if named:
            claimed.update(named)
            last = max(named, key=lambda n: pos[n])
            mods = tuple(sorted({mod[n] for n in named}))
            out[k] = (mod[last], "source", mods if len(mods) > 1 else ())
            span[k] = (min(pos[n] for n in named), max(pos[n] for n in named))
            continue
        for op in _site_ops(s):
            short = op.split(".")[-1]
            if op.startswith("aten.") or not short:
                continue
            start = used.get(short, 0)
            hit = next((i for i in range(start, len(nodes)) if nodes[i][0].startswith(short) and nodes[i][0] not in claimed), None)
            if hit is not None:
                used[short] = hit + 1
                claimed.add(nodes[hit][0])
                out[k] = (nodes[hit][2], "op", ())
                span[k] = (hit, hit)
                break
    for k, s in enumerate(sites):
        if out[k] is not None:
            continue
        lo = max((span[j][1] for j in range(k) if span[j] is not None), default=-1)
        hi = min((span[j][0] for j in range(k + 1, len(sites)) if span[j] is not None), default=len(nodes))
        cand = [nodes[i] for i in range(lo + 1, hi) if nodes[i][0] not in claimed]
        if not cand:
            continue
        mods = [m for _, _, m in cand]
        distinct = tuple(sorted(set(mods)))
        out[k] = (mods[0], "between", distinct if len(distinct) > 1 else ())
    return out


def _nodes(graph: Any) -> List[Tuple[str, str, str]]:
    out = []
    for node in getattr(graph, "graph", graph).nodes:
        if node.op not in ("call_function", "call_method", "call_module"):
            continue
        st = node.meta.get("nn_module_stack") or {}
        if st:
            out.append((node.name, str(node.target), _fqn(list(st.values())[-1])))
    return out


def _table(piece: Any, prog: int) -> List[Optional[Tuple[int, str]]]:
    """per site of prog: the scope id and name in this piece, or None"""
    graph = getattr(piece, "graph", None)
    sites = _programs[prog]
    if graph is None:
        _log(f"piece {_index(piece)} has no FX graph (a compile-cache hit?): its kernels keep the piece's scope")
        return [None] * len(sites)
    res = resolve(sites, _nodes(graph))
    table: List[Optional[Tuple[int, str]]] = []
    for s, r in zip(sites, res):
        if r is None:
            table.append(None)
            continue
        path, how, covers = r
        name, op = scope_name(path)
        _stats["how"][how] = _stats["how"].get(how, 0) + 1
        attrs = {"op": op, "compiled": s.handle, "how": how}
        if covers:  # the kernel's sources span several modules: listed, the time stays on one
            attrs["covers"] = [scope_name(m)[0] for m in covers]
        table.append((api.scope_id(name, **attrs), name))
    return table


# ---- names ---------------------------------------------------------------------------------------

def set_root(model: Any, prefix: str = "model") -> None:
    """The model's module paths and types, to name a compiled module as its hook would."""
    paths, types, by_suffix = [], {}, {}
    for name, m in model.named_modules():
        if not name:
            continue
        paths.append(name)
        types[name] = type(m).__name__
        parts = name.split(".")
        for i in range(len(parts)):
            suf = ".".join(parts[i:])
            by_suffix[suf] = name if suf not in by_suffix else (by_suffix[suf] if by_suffix[suf] == name else None)
    _root.update(paths=paths, types=types, by_suffix=by_suffix, prefix=prefix)


def scope_name(path: str) -> Tuple[str, str]:
    """A module path relative to a compiled module -> (scope name, op): the hook's name for the
    module in the model (`model.` and its path) when the path is unique in it. A module the model
    does not register (a linear's quant_method.fp8_linear.quant_fp8: quant_method is no
    nn.Module, so no hook names it) takes its nearest enclosing module the model has, as its
    kernels do uncompiled (model-w: o_proj, wq_b, q_proj). Else `compiled.<path>`."""
    parts = path.split(".") if path else []
    for n in range(len(parts), 0, -1):
        full = _root["by_suffix"].get(".".join(parts[:n]))
        if full:
            return f"{_root.get('prefix', 'model')}.{full}", _root["types"].get(full, "compiled")
    return f"compiled.{path}" if path else "compiled", "compiled"


def _index(piece: Any) -> Any:
    return getattr(piece, "piecewise_compile_index", None)


# ---- patches -------------------------------------------------------------------------------------

def pin_triton_cache() -> Optional[str]:
    """Triton's cache directory, set where Inductor would set it, when nothing has. vLLM sets one
    under its compile cache, which this turns off. Inductor sets one only when it builds a kernel
    in this process, and a start whose kernels all come back from its compile workers (a warm
    Inductor cache, the worker pool ready) builds none: vLLM's own Triton kernels (its JIT
    warmup) then compile into ~/.triton/cache, cold, on that start (plan 06 Part E: 30 s of a
    second start). Set before the first program loads, as Inductor's kernels would set it."""
    if os.environ.get("TRITON_CACHE_DIR"):
        return None
    try:
        import torch
        from torch._inductor.runtime.cache_dir_utils import triton_cache_dir

        dev = torch.cuda.current_device() if torch.cuda.is_initialized() else 0
        os.environ["TRITON_CACHE_DIR"] = d = triton_cache_dir(dev)
    except Exception as e:  # noqa: BLE001 - Triton then keeps its own default
        _log(f"Triton's cache directory not set ({e!r})")
        return None
    return d


def patch_codecache(mod: Any) -> None:
    """Programs are rewritten as Inductor loads them; Inductor writes the site handles."""
    try:
        import torch._inductor.config as ic

        ic.trace.provenance_tracking_level = max(1, int(getattr(ic.trace, "provenance_tracking_level", 0) or 0))
    except Exception as e:  # noqa: BLE001
        _log(f"provenance tracking not set ({e!r}): sites are found by their source comments alone")
    P = getattr(mod, "PyCodeCache", None)
    if P is None or getattr(P.load_by_key_path, "_ruppell_wrapped", False):
        return
    orig = P.load_by_key_path.__func__

    def load_by_key_path(cls, *args, **kwargs):
        pin_triton_cache()
        m = orig(cls, *args, **kwargs)
        try:
            install(m)
        except Exception as e:  # noqa: BLE001 - never break a compile over attribution
            _log(f"program {getattr(m, '__file__', '?')} left as it is: {e!r}")
            m._ruppell_program = -1
        return m

    load_by_key_path._ruppell_wrapped = True
    P.load_by_key_path = classmethod(load_by_key_path)


def patch_pieces(mod: Any) -> None:
    B = getattr(mod, "PiecewiseBackend", None)
    if B is None or getattr(B.__call__, "_ruppell_wrapped", False):
        return
    orig = B.__call__
    ids: Dict[Any, int] = {}

    def call(self, *args):
        idx = _index(self)
        sid = ids.get(idx)
        if sid is None:
            sid = ids[idx] = api.scope_id("compiled.piece", op="compiled_piece", index=idx)
        prev, prev_base = getattr(_tls, "piece", None), getattr(_tls, "base", 0)
        _tls.piece, _tls.base = self, len(_frames())
        api.push_id(sid, "compiled.piece")
        try:
            return orig(self, *args)
        finally:
            _unwind(_tls.base)
            api.pop_id()
            _tls.piece, _tls.base = prev, prev_base

    call._ruppell_wrapped = True
    B.__call__ = call


def _patch_backends(mod: Any) -> None:
    patch_backend(mod)
    codegen = sys.modules.get(CODEGEN)
    if codegen is not None:
        patch_codegen(codegen)   # backends.py's own name for compile_execution_fn


def patch_backend(mod: Any) -> None:
    """After vLLM splits a model, each splitting submodule (the ops it splits at: attention, MLA,
    the sparse indexers) gets a scope, its op's module: the stitching code calls it as a module,
    so a forward hook runs at every call, eager or captured."""
    VB = getattr(mod, "VllmBackend", None)
    if VB is None or getattr(VB.__call__, "_ruppell_wrapped", False):
        return
    orig = VB.__call__

    def call(self, graph, example_inputs):
        out = orig(self, graph, example_inputs)
        try:
            n = _scope_splitting(self)
            _stats["splitting"] = _stats.get("splitting", 0) + n
        except Exception as e:  # noqa: BLE001 - never break a compile over attribution
            _log(f"splitting ops not scoped: {e!r}")
        return out

    call._ruppell_wrapped = True
    VB.__call__ = call


def patch_codegen(mod: Any) -> None:
    """vLLM's generated stitching code inlines each splitting submodule as a function of its own
    (`__vllm_inlined_submods__<i>`, vllm/compilation/codegen.py) and calls that, not the module,
    so a module hook never runs (MI350X, model-w's image: 144 hooked, none called). Each inlined
    function is wrapped, where compile_execution_fn binds it, in the scope of its op's module.
    backends.py imports the function by name: both names are patched."""
    for target in (mod, sys.modules.get(BACKENDS)):
        orig = getattr(target, "compile_execution_fn", None) if target is not None else None
        if orig is None or getattr(orig, "_ruppell_wrapped", False):
            continue

        def compile_execution_fn(execution_code, submod_callables, submod_names, *args, _orig=orig, **kwargs):
            fn = _orig(execution_code, submod_callables, submod_names, *args, **kwargs)
            try:
                n = _scope_inlined(fn, submod_callables, submod_names)
                _stats["inlined"] = _stats.get("inlined", 0) + n
            except Exception as e:  # noqa: BLE001 - never break a compile over attribution
                _log(f"inlined splitting ops not scoped: {e!r}")
            return fn

        compile_execution_fn._ruppell_wrapped = True
        target.compile_execution_fn = compile_execution_fn


def _scope_inlined(fn: Any, submod_callables: Dict[str, Any], submod_names: List[str]) -> int:
    g = getattr(getattr(fn, "func", fn), "__globals__", None)
    if g is None:
        return 0
    n = 0
    for idx, name in enumerate(submod_names):
        key = f"__vllm_inlined_submods__{idx}"
        f, sub = g.get(key), submod_callables.get(name)
        if f is None or getattr(f, "_ruppell_wrapped", False) or sub is None or not hasattr(sub, "graph"):
            continue
        nodes = _nodes(sub)
        if not nodes:
            continue
        g[key] = _in_scope(f, nodes[-1][2])
        n += 1
    return n


def _in_scope(f: Any, path: str) -> Any:
    """f, called in the scope of the module at path (named on its first call: the model's paths
    are known by then)"""
    ent: Dict[str, Any] = {}

    def wrapped(*args, **kwargs):
        if "sid" not in ent:
            name, op = scope_name(path)
            ent["name"], ent["sid"] = name, api.scope_id(name, op=op, compiled="splitting")
        api.push_id(ent["sid"], ent["name"])
        try:
            return f(*args, **kwargs)
        finally:
            api.pop_id()

    wrapped._ruppell_wrapped = True
    return wrapped


def _scope_splitting(backend: Any) -> int:
    gm = getattr(backend, "split_gm", None)
    n = 0
    for item in getattr(backend, "piecewise_graphs", None) or []:
        if not getattr(item, "is_splitting_graph", False):
            continue
        sub = gm._modules.get(item.submod_name) if gm is not None else None
        if sub is None or not hasattr(sub, "register_forward_pre_hook") or getattr(sub, "_ruppell_scoped", False):
            continue
        nodes = _nodes(sub)
        if not nodes:
            continue
        path = nodes[-1][2]          # the split op's module (a kv-cache update before it shares it)
        ent: Dict[str, Any] = {}

        def pre(mod, args, _path=path, _ent=ent):
            if "sid" not in _ent:
                name, op = scope_name(_path)
                _ent["name"], _ent["sid"] = name, api.scope_id(name, op=op, compiled="splitting")
            api.push_id(_ent["sid"], _ent["name"])

        def post(mod, args, out):
            api.pop_id()

        sub.register_forward_pre_hook(pre)
        sub.register_forward_hook(post)
        sub._ruppell_scoped = True
        n += 1
    return n


def patch_attention(mod: Any) -> None:
    orig = getattr(mod, "get_attention_context", None)
    if orig is None or getattr(orig, "_ruppell_wrapped", False):
        return

    def get_attention_context(layer_name):
        res = orig(layer_name)
        try:
            _scope_layer(res[1], layer_name)
        except Exception as e:  # noqa: BLE001
            _log(f"attention layer {layer_name!r} not scoped: {e!r}")
        return res

    get_attention_context._ruppell_wrapped = True
    mod.get_attention_context = get_attention_context
    # modules that imported the name before this patch hold the original
    for m in list(sys.modules.values()):
        if getattr(m, "__name__", "").startswith("vllm") and getattr(m, "get_attention_context", None) is orig:
            m.get_attention_context = get_attention_context


def _scope_layer(layer: Any, layer_name: Any) -> None:
    impl = getattr(layer, "impl", None)
    if impl is None or getattr(impl, "_ruppell_scoped", False):
        return
    name = getattr(layer, "layer_name", None) or getattr(layer_name, "value", layer_name)
    full = f"{_root.get('prefix', 'model')}.{name}"
    sid = api.scope_id(full, op=type(layer).__name__)
    for meth in ("forward", "do_kv_cache_update"):
        fn = getattr(impl, meth, None)
        if fn is None:
            continue

        def wrapped(*a, _fn=fn, **k):
            api.push_id(sid, full)
            try:
                return _fn(*a, **k)
            finally:
                api.pop_id()
        setattr(impl, meth, wrapped)
    impl._ruppell_scoped = True


def patches() -> Dict[str, Any]:
    """module -> patch, applied when the module is imported (the adapter's PATCHES)"""
    os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")
    out = {CODECACHE: patch_codecache, PIECES: patch_pieces, BACKENDS: _patch_backends, CODEGEN: patch_codegen}
    for name in ATTENTION:
        out[name] = patch_attention
    return out


def summary() -> Dict[str, Any]:
    return {"programs": _stats["programs"], "sites": _stats["sites"], "resolved": dict(_stats["how"]),
            "pieces": len(_pieces), "splitting": _stats.get("splitting", 0), "inlined": _stats.get("inlined", 0)}
