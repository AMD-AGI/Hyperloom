---
title: edit-surface reach — how a permitted file changes behaviour outside itself
kind: technique
gens: [gfx942, gfx950]
dtypes: [any]
regimes: [prefill, decode, training, both]
updated: 2026-08-21
---

# edit-surface reach — a permitted file is a lever, not a box

## TL;DR
The campaign's declared source set is listed in the planning context as `editable_sources`. That
list is a **floor**, not a ceiling — everything on it is editable, and so is every other tracked,
non-protected implementation file in the workspace; the hard boundary is the protected measurement
surface (driver, harness, tests, scoring, reference), not the list. What the list never bounds is
**what you may change**. A Python module that is imported before the framework consumes it can
rebind anything in the process; a kernel file can carry device source into the compiler through the
framework's own hook; a module-level constant is a constant regardless of where its default came
from; a data or config file is an input to whatever reads it. Before pricing a direction as "out of
reach", answer one question: *does anything I am allowed to edit run, or get read, before the thing
I want to change is used?* If yes, the direction is in reach. Ruling out an axis because "that would
mean patching the framework/library, not this file" is the specific mistake this card exists to
prevent — it has cost real campaigns their largest available win.

## The general move
> **Find the last point, reachable from a file you may edit, at which the behaviour you want is still
> mutable — then change it there.**

Everything below is one *instance* of that move, listed to make the shape recognisable. They are
**not four routes**. Enumerating routes is exactly the failure: an agent that lists two mechanisms,
prices both, and closes has produced a *closed list*, and a slightly longer closed list is not a fix.
Read each class as a question to ask about your own program, and expect the instance that applies to
you to be one that is not written here.

---

## Class 1 — rebind a symbol in an installed package, from the permitted file

**The move:** anything the framework will look up *later* can be replaced *now*, from code that runs
earlier. Import the module that owns the symbol, keep the original, bind your own in its place. The
installed package on disk is untouched; the process sees your version. Applies to functions, methods,
classes, module constants, registry entries, dispatch tables — any name resolved at call time.

*One instance* — an installed codegen package emits a prologue you want to change, and your permitted
file is imported before any kernel is built:

```python
# permitted_kernel.py — imported before the first kernel is compiled
import vendorlib.codegen as vc

_orig_emit_prologue = vc.Pipeliner.emit_prologue

def _emit_prologue(self, stage, *args, **kwargs):
    if stage == 0 and self.tile_k >= 128:
        return _emit_double_buffered_prologue(self, stage, *args, **kwargs)
    return _orig_emit_prologue(self, stage, *args, **kwargs)   # unchanged path intact

vc.Pipeliner.emit_prologue = _emit_prologue
```

**Conditions that make it legitimate:** the rebind happens before first use (import order matters —
verify it, do not assume it); the original stays reachable and is used for every case you did not
mean to change; and the change is correct for every shape in the suite, not only the one you timed.

**Verify it took effect** — a rebind that lands after the framework already captured the symbol is a
silent no-op that benchmarks as "no difference". Print a marker from the replacement, or diff the
generated code/ISA, before believing a negative result.

---

## Class 2 — inject device-side source through the framework's own hook

**The move:** most kernel DSLs have a documented door for source they did not generate — an
`import_source` / `pragma_import_c` string, a custom-intrinsic registration, an inline-asm escape, an
extern-call path. That door is reachable from the permitted file, so the instruction sequence the DSL
will not emit is still available to you. You are not limited to what the DSL's code generator knows
how to produce.

*One instance* — the generated code uses a full-precision reciprocal where the workload tolerates the
fast one:

```python
_DEVICE_SRC = r"""
extern "C" __device__ float kf_fast_recip(float x) {
    return __builtin_amdgcn_rcpf(x);
}
"""

@T.prim_func
def kernel(...):
    T.import_source(_DEVICE_SRC)                     # the framework's own hook
    T.attr("pragma_import_c", _DEVICE_SRC)
    ...
    inv = T.call_extern("float", "kf_fast_recip", denom)
```

The same move with a different door: register a custom intrinsic; emit `asm volatile` for one
instruction the compiler will not select; or wrap a `__builtin_amdgcn_*` the DSL has no surface for.

**Conditions:** the injected code must be correct across the dtype and range the suite actually
exercises (a fast reciprocal, a relaxed rounding mode, or a byte-permute assumes something — write
down what); and the numerics gate still applies
(`[[optimization/lever_numerics.md]]`). Confirm the symbol actually reached the module by reading
the generated source or the ISA dump — a mis-declared extern usually fails loudly, but a shadowed one
does not.

---

## Class 3 — change a module-level constant another module's dispatch reads

**The move:** a constant defined in a file you may edit governs every consumer that imports it,
including consumers in files you may not touch. Bounds, thresholds, tile floors, "small case" cutoffs
and enable flags decide which implementation runs; changing one can move an entire shape class onto a
different kernel without editing that kernel at all. Grep the editable files for module-level
assignments, then grep the tree for who reads them.

*One instance* — a cutoff in an editable module routes small batches to a separate path:

```python
# pkg/dispatch_limits.py   ← on the editable list
_SMALL_BATCH_TILE = int(os.environ.get("PKG_SMALL_BATCH_TILE", "64"))

# pkg/router.py            ← not on the list; reads the constant anyway
if batch <= _SMALL_BATCH_TILE:
    return _small_batch_kernel(...)
return _general_kernel(...)
```

Setting `_SMALL_BATCH_TILE = 0` retires the small-batch path for the whole suite; raising it moves
more shapes onto it. Neither edit touches `router.py`.

### The `os.environ` converse — state it to yourself explicitly
**A constant whose default is read from the environment inside an editable file is editable.** The
presence of an `os.environ.get(...)` on the right-hand side says **nothing** about the edit surface —
it is a default, not a permission boundary. "That is an environment variable, not one of the editable
files" is a category error: the *variable* is environment-supplied, the *constant* is a module-level
assignment in a file you were handed. You may edit the literal, edit the default, or set the variable
for the run — and if the variable is not honoured by the deployment you are scored under, edit the
literal. The same reasoning covers a constant behind `getattr(config, ...)`, a `functools.lru_cache`d
getter, or a value read once at import.

Related: an env-defaulted constant is also a first-class sweep target — see
`[[optimization/lever_cheap_sweeps.md]]`.

---

## Class 4 — append to a permitted data or config file that a lookup consumes

**The move:** a lookup table is code. If a CSV / JSON / YAML on the editable list is what a dispatcher
consults to choose a configuration, adding or correcting a row changes which kernel runs, with no
source edit at all. This is the class most often skipped because the file "is not source" — the
editable list does not distinguish, and neither should you.

*One instance* — a per-shape config table with a generic fallback row:

```
# configs/tile_shapes.csv   ← on the editable list
arch,   M,    N,    K,    tile_m, tile_n, tile_k, waves
gfx950, 8192, 8192, 8192, 256,    128,    64,     4
gfx950, *,    *,    *,    128,    128,    32,     2      # fallback row
```

```python
row = table.get((arch, M, N, K)) or table.get((arch, "*", "*", "*"))
```

Two failure modes live in that one line, and both are worth checking on any table you are handed:

- **A primary hit shadows the fallback.** Appending a better fallback row changes nothing for a shape
  that already has an exact entry. If the benchmark shape is in the table, the row you must edit is
  the exact one, not the general one.
- **A missing primary silently falls back.** A shape absent from the table runs the generic row and
  looks like "the tuned config is not helping". Appending the exact row for the benchmark's shape is
  the whole fix — and it is an append to a data file, not a kernel change.

**Verify the row is actually consumed**: match the key field-for-field (an extra column, a dtype
spelled differently, a flag tuned `true` against a live `false` ⇒ 100% lookup miss and zero effect —
see `[[optimization/lever_autotune.md]]`), and confirm engagement from the log or a marker
rather than from the fact that you edited the file.

---

## Asking the question on your own program
Run this before writing "out of reach" in a plan:

1. **What runs first?** List everything imported, executed, or read from the editable set before the
   behaviour you want to change is used. That is your rebinding window.
2. **Who reads what I own?** For each module-level name in the editable files, find every consumer.
   Consumers outside the editable set are the point of the exercise.
3. **What doors does this framework document?** Source-injection hooks, intrinsic registration,
   extern calls, inline asm, dispatch registries, plugin/backend tables.
4. **What non-source files are on the list?** Every CSV/JSON/YAML there is consumed by something;
   find the lookup and the key.
5. **Did it take effect?** Every class here has a silent-no-op mode (late rebind, shadowed symbol,
   unread constant, shadowed table row). A negative measurement from a change that never engaged is
   worse than no measurement, because it closes the axis.

## Boundaries that are real
Reach is not permission to fake a result. Still forbidden regardless of which file you type in:
editing the driver or the benchmark harness, special-casing on benchmark shapes without verifying the
invariant on the real tensors, weakening a correctness check, or mutating installed packages **on
disk** outside the workspace (a process-local rebind from a permitted file is a different thing — it
travels with the source you deliver). The canonical gate still decides everything
(`[[profiling/measure_protocol.md]]`).

## See also
- `[[optimization/lever_cheap_sweeps.md]]` — once a constant is in reach, measure it in one command.
- `[[optimization/lever_autotune.md]]` — lookup keys, engagement checks, tuned tables.
- `[[optimization/lever_numerics.md]]` — the gate any injected fast path must survive.
