"""Scopes for PyTorch modules, through forward hooks.

Every submodule gets a scope named by its path, with the module's class as
`op` and the first input's shape; the root module too, named by `prefix`.
Scopes nest, and a kernel carries the innermost one, so work done in a
container's own forward (a residual add in a block, say) is attributed to the
container, and work in a leaf to the leaf.

Hooks run in Python, so during graph capture they run once per captured
launch sequence and the scopes are stored in the graph's registry; during
replay no Python runs and the registry supplies them.

A module is hooked once: instrumenting a second tree that shares modules with
the first (a drafter sharing the target's embedding, say) leaves the shared
ones under their first path.

Inside a region torch.compile traces (vLLM's compiled backbone, say), Dynamo
inlines the hooks into the graph it captures, and could not trace their
first call (a scope definition serializes its attributes; vLLM 0.28.0's
fullgraph compile refused it, 2026-10-22). So they do nothing while Dynamo
traces, and the compiled code they were traced into calls none: kernels of a
compiled region carry the scope around it.
"""

from __future__ import annotations

from typing import Any, List

from ruppell import api

try:
    from torch.compiler import is_compiling as _compiling
except ImportError:  # torch before 2.3
    def _compiling() -> bool:
        return False


def _shape(args: Any):
    if args and hasattr(args[0], "shape"):
        return tuple(args[0].shape)
    return None


def instrument(module: Any, prefix: str = "") -> List[Any]:
    handles = []
    for name, m in module.named_modules():
        if getattr(m, "_ruppell_scope", None) is not None:
            continue
        path = ".".join(p for p in (prefix, name) if p) or type(module).__name__
        op = type(m).__name__

        # the scope id per input shape, defined once: eager forwards (prefills)
        # run every hook, and a lookup by serialized attributes costs more than
        # the push itself
        def pre(mod, args, _path=path, _op=op, _ids={}):
            if _compiling() or not api._fine_scopes:
                return
            shp = _shape(args)
            sid = _ids.get(shp)
            if sid is None:
                sid = _ids[shp] = api.scope_id(_path, op=_op, shape=list(shp) if shp is not None else None)
            api.push_id(sid, _path)

        def post(mod, args, out):
            if _compiling():
                return
            api.pop_id()

        m._ruppell_scope = path
        handles.append(m.register_forward_pre_hook(pre))
        handles.append(m.register_forward_hook(post))
    return handles
