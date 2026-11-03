"""ruppell: a graph-aware, step-anchored tracer for GPU serving stacks.

Start a program under `python -m ruppell run --out DIR -- <command>`, call the
API below from it, then `python -m ruppell export DIR -o trace.json`.
"""

from ruppell.api import (capture, context, counter_steps, counters, detail, flush, init, is_counter_step, scope, scope_pop,
                         scope_push, step_begin, step_end)

__all__ = ["capture", "context", "counter_steps", "counters", "detail", "flush", "init", "instrument", "is_counter_step", "scope",
           "scope_pop", "scope_push", "step_begin", "step_end"]
__version__ = "0.9.1"


def instrument(module, prefix: str = ""):
    """Scope every leaf submodule of a torch module by its path (see ruppell.torch_hooks)."""
    from ruppell.torch_hooks import instrument as _instrument

    return _instrument(module, prefix)
