#!/usr/bin/env python3
"""unittest_agent: generate an AgentKernelArena-style unittest harness for a
kernel candidate before handing it off to GEAK.

Design goals
============

1. Reflect the **end-to-end runtime** as closely as possible:

   * Import the kernel from its on-disk source file (in-place under
     ``/sgl-workspace/{aiter,sglang,vllm}/...``) so PyTorch/Triton see the
     exact same module graph as vLLM/SGLang do.
   * Reproduce the SGLang/vLLM environment variables seen by the kernel
     (``SGLANG_*``, ``AITER_*``, ``TRITON_*``, ``HIP_*``, ``ROCR_*``,
     ``CUDA_*``) by exporting them inside ``task_runner.py`` *before* the
     kernel module is imported.
   * Use the **profile-captured** input shapes / dtypes from TraceLens
     (``candidate["input_shapes"]`` / ``candidate["input_dtypes"]``) so the
     unittest exercises a real production decode / prefill shape, not a
     synthetic 64x64 sanity tile.
   * For correctness, snapshot the kernel source bytes at generation time
     and use the *snapshot* as the golden reference; the runner then
     re-imports the *current* source (which GEAK may have rewritten) and
     compares outputs tensor-by-tensor. This avoids hand-writing a torch
     reference for arbitrary kernels and gives GEAK a real correctness gate.

2. Match AgentKernelArena's task layout exactly so future evaluation
   pipelines can pick the unittests up unchanged:

   ::

       <out_dir>/
         config.yaml                 # AgentKernelArena task config
         source/<kernel_name>.py     # symlink into /sgl-workspace/...
         source/_baseline_snapshot/  # frozen golden bytes (CRITICAL)
         scripts/task_runner.py      # compile / correctness / performance
         unittest_meta.json          # generator audit trail + self-verify

3. Self-verify the generated harness before declaring success: run
   ``task_runner.py compile`` and ``task_runner.py correctness`` against
   the **unmodified** kernel (so correctness MUST pass: original-vs-original).
   Any failure is reported back to the caller — the caller can still ship
   the prompt to GEAK as a degraded ``best_effort`` harness, but downstream
   verification weight should be lower.

Public surface
==============

``generate_unittest(candidate, *, out_dir, target_platform, log) -> dict``
    Top-level entry point. Returns a manifest dict with keys::

        {
          "status": "ok" | "degraded" | "failed",
          "out_dir": "<absolute path>",
          "config_yaml": "<path>",
          "task_runner": "<path>",
          "test_command": "python3 <task_runner.py> correctness",
          "performance_command": "python3 <task_runner.py> performance",
          "source_file": "<symlinked kernel source>",
          "kernel_name": "<entry function name>",
          "self_verify": {"compile": "ok|fail", "correctness": "ok|fail|skipped", ...},
          "warnings": [...],
        }

For Python/Triton kernels the generator inspects the module's public callables
to find a plausible host entry point (the function that prepares Triton grid +
calls ``@triton.jit``). For HIP/C++ kernels it emits a wrapper harness around
TraceLens-discovered ``benchmark_files``: at GEAK test time the runner
temporarily overlays the edited ``source/<kernel>`` file onto the live framework
source path, invalidates likely aiter JIT modules, runs the benchmark command,
and restores the live tree afterwards.
"""

from __future__ import annotations

import ast
import gzip
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Dtype helpers
# ---------------------------------------------------------------------------

# Canonical mapping from TraceLens / SGLang dtype strings to torch.dtype.
_DTYPE_ALIASES: dict[str, str] = {
    # floating-point
    "fp16": "torch.float16", "float16": "torch.float16", "half": "torch.float16",
    "f16": "torch.float16",
    "bf16": "torch.bfloat16", "bfloat16": "torch.bfloat16",
    "fp32": "torch.float32", "float32": "torch.float32", "float": "torch.float32",
    "f32": "torch.float32",
    "fp64": "torch.float64", "float64": "torch.float64", "double": "torch.float64",
    "fp8": "torch.float8_e4m3fn", "fp8_e4m3": "torch.float8_e4m3fn",
    "e4m3": "torch.float8_e4m3fn", "fp8_e5m2": "torch.float8_e5m2",
    "e5m2": "torch.float8_e5m2",
    # integer
    "int8": "torch.int8", "i8": "torch.int8",
    "int16": "torch.int16", "i16": "torch.int16",
    "int32": "torch.int32", "i32": "torch.int32",
    "int64": "torch.int64", "i64": "torch.int64", "long": "torch.int64",
    "uint8": "torch.uint8", "u8": "torch.uint8",
    "bool": "torch.bool", "boolean": "torch.bool",
}


def _normalize_dtype(name: Any) -> str:
    """Return a ``torch.<dtype>`` expression string for ``name``.

    Unknown / blank dtypes fall back to ``torch.float16`` (the most common
    inference dtype in vLLM/SGLang). The caller is expected to capture the
    fallback in ``warnings``.
    """
    if name is None:
        return "torch.float16"
    text = str(name).strip().lower()
    if not text or text in ("none", "null", "unknown", "-"):
        return "torch.float16"
    # Strip ``torch.`` prefix if present so the lookup table hits.
    text = text.replace("torch.", "")
    return _DTYPE_ALIASES.get(text, "torch.float16")


def _is_integer_dtype_expr(expr: str) -> bool:
    return any(tok in expr for tok in (
        "int8", "int16", "int32", "int64", "uint8", "bool",
    ))


# ---------------------------------------------------------------------------
# Shape helpers
# ---------------------------------------------------------------------------

_SHAPE_TUPLE_RE = re.compile(r"\(([^()]*)\)")


def _coerce_shape(value: Any) -> list[int] | None:
    """Best-effort: turn a shape descriptor into a list of ints.

    Accepted forms (mirroring TraceLens / GEAK conventions):
      * ``[2, 64, 64, 32]`` -> ``[2, 64, 64, 32]``
      * ``(2, 64, 64, 32)`` -> ``[2, 64, 64, 32]``
      * ``"(2, 64, 64, 32)"`` -> ``[2, 64, 64, 32]``
      * ``"2x64x64x32"`` -> ``[2, 64, 64, 32]``
      * ``{"shape": "(2, 64, 64)"}`` -> ``[2, 64, 64]``
    Returns ``None`` when no integer dimensions can be extracted.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        for k in ("shape", "dims", "Args", "value"):
            if k in value:
                got = _coerce_shape(value[k])
                if got is not None:
                    return got
        return None
    if isinstance(value, (list, tuple)):
        out: list[int] = []
        for elem in value:
            if isinstance(elem, int):
                out.append(int(elem))
            elif isinstance(elem, str):
                try:
                    out.append(int(elem.strip()))
                except ValueError:
                    return None
            else:
                return None
        return out or None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        # Try parenthesised form first; pick the FIRST inner tuple so
        # things like ``(2, 64, 64, 32) torch.float16`` work.
        match = _SHAPE_TUPLE_RE.search(text)
        if match:
            inner = match.group(1)
            parts = [p.strip() for p in inner.split(",") if p.strip()]
            try:
                return [int(p) for p in parts]
            except ValueError:
                return None
        # Fallback: ``MxNxK`` or ``M,N,K``.
        delim = "x" if "x" in text else ","
        parts = [p.strip() for p in text.split(delim) if p.strip()]
        try:
            return [int(p) for p in parts]
        except ValueError:
            return None
    return None


def _collect_input_shapes(candidate: dict[str, Any]) -> tuple[list[list[int]], list[str]]:
    """Return one input-shape list per kernel argument.

    Order of preference (matches ``build_kernel_metadata`` in
    ``kernel_optimization.py``):

      1. ``candidate["input_shapes"]`` — TraceLens-resolved per-arg shapes
         (most reliable; one entry per kernel input tensor).
      2. ``candidate["shapes"]`` / ``candidate["task_group"]["rows"][0]["shapes"]``
         — raw TraceLens row strings. Each entry may be ``(shape) dtype``,
         in which case we extract the shape and dtype together.
      3. None — caller falls back to a degraded ``test_shapes = []`` and the
         harness is marked ``best_effort``.

    Returns ``(shapes, dtypes_hint)`` where ``dtypes_hint`` carries any
    dtype tags we sniffed out of the shape strings (so the caller can use
    them when ``candidate["input_dtypes"]`` is empty).
    """
    dtypes_hint: list[str] = []
    raw_shapes = candidate.get("input_shapes")
    if isinstance(raw_shapes, list) and raw_shapes:
        out: list[list[int]] = []
        for entry in raw_shapes:
            shape = _coerce_shape(entry)
            if shape:
                out.append(shape)
        if out:
            return out, dtypes_hint

    # Fall back to TraceLens row strings.
    row_shapes: list[Any] = []
    if isinstance(candidate.get("shapes"), list):
        row_shapes = list(candidate["shapes"])
    elif isinstance(candidate.get("task_group"), dict):
        rows = candidate["task_group"].get("rows") or []
        if rows and isinstance(rows[0], dict):
            shapes = rows[0].get("shapes")
            if isinstance(shapes, list):
                row_shapes = shapes
    out2: list[list[int]] = []
    for entry in row_shapes:
        shape = _coerce_shape(entry)
        if shape:
            out2.append(shape)
        # Sniff dtype if the entry string also carried one.
        text = entry.get("shape") if isinstance(entry, dict) else entry
        if isinstance(text, str):
            for dtoken in (
                "float16", "bfloat16", "float32", "int32", "int64",
                "float8", "fp16", "bf16", "fp32",
            ):
                if dtoken in text.lower():
                    dtypes_hint.append(dtoken)
                    break
            else:
                dtypes_hint.append("")
    return out2, dtypes_hint


# ---------------------------------------------------------------------------
# Kernel source inspection
# ---------------------------------------------------------------------------

# Names we accept as plausible host entry points when the candidate doesn't
# carry an explicit one. The kernel name itself wins; otherwise we try a
# few common SGLang/vLLM/aiter wrapper conventions.
_HOST_ENTRY_FALLBACKS = (
    "{kernel}",
    "{kernel}_launcher",
    "{kernel}_triton",
    "{kernel}_fwd",
    "{kernel}_forward",
    "run_{kernel}",
    "launch_{kernel}",
)


def _parse_top_level_callables(
    source_path: Path,
) -> tuple[list[tuple[str, int]], list[str]]:
    """Return ``(host_functions, jit_functions)`` from a Python source file.

    ``host_functions`` are top-level ``def`` / ``async def`` that are NOT
    decorated with ``@triton.jit`` (so they're the Python launchers that
    do tensor allocation + grid computation + the ``kernel[grid](...)``
    call). Each entry is ``(name, arg_count)`` so the picker can prefer
    candidates whose signature width matches the captured shape count
    (very important on aiter/sglang launchers where one source file
    contains 10+ small helpers plus the real entry point).

    ``jit_functions`` are the ``@triton.jit`` bodies themselves (just
    names, since they're not callable from torch without preparation).

    Anything we can't parse returns empty lists (caller falls back to the
    candidate's ``name`` field).
    """
    host: list[tuple[str, int]] = []
    jit: list[str] = []
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8", errors="replace"))
    except (SyntaxError, OSError):
        return host, jit
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        is_jit = False
        for dec in node.decorator_list:
            text = ast.unparse(dec) if hasattr(ast, "unparse") else ""
            if "triton.jit" in text or text.endswith(".jit") or text == "jit":
                is_jit = True
                break
        if is_jit:
            jit.append(node.name)
        else:
            n_args = len(node.args.args) + len(node.args.kwonlyargs)
            host.append((node.name, n_args))
    return host, jit


def _pick_host_entry(
    candidate: dict[str, Any],
    source_path: Path,
    num_shape_args: int = 0,
) -> tuple[str | None, list[str], int]:
    """Return ``(host_entry_name, all_host_callable_names)``.

    The host entry MUST be callable from torch without preparation — i.e. a
    function whose arguments are torch tensors / ints / dtypes (not Triton
    block pointers). For most vLLM/SGLang kernel modules this is the only
    non-``@triton.jit`` function; when there are multiple we pick the one
    whose **(name, arg-count)** best matches the candidate's profile.

    Scoring:

      1. Exact name match → win immediately.
      2. Pattern match (``{kernel}_triton`` / ``run_{kernel}`` etc.) → win.
      3. Otherwise score every public host by:
         * +5 if name contains the kernel ``base`` (e.g. ``rms_norm`` in
           ``rmsnorm2d_fwd_with_add``);
         * +10 if arg-count is in ``[num_shape_args, num_shape_args+2]``
           (tolerating a couple of scalar args like ``epsilon`` / dtype);
         * −3 per name prefix underscore (``_helper`` is rarely the entry
           point);
         * −3 if the name is very short (``num_programs`` / ``block_size``
           helpers are 1-arg utilities we don't want to mistake for the
           real launcher).
         Pick the highest scoring; ties broken by file order (top-down).
    """
    host_pairs, _jit = _parse_top_level_callables(source_path)
    if not host_pairs:
        return None, [], 0
    host_names = [name for name, _ in host_pairs]
    pair_lookup = {name: nargs for name, nargs in host_pairs}
    kernel_name = str(
        candidate.get("kernel_name")
        or candidate.get("name")
        or candidate.get("kernel_id")
        or ""
    ).strip()
    if not kernel_name:
        return host_names[0], host_names, pair_lookup[host_names[0]]
    base = re.sub(r"_kernel$", "", kernel_name)
    base = base.lstrip("_") or base  # strip leading underscores from base only
    # 1. Exact match wins.
    for cand in (kernel_name, base):
        if cand in pair_lookup:
            return cand, host_names, pair_lookup[cand]
    # 2. Pattern match wins.
    for pattern in _HOST_ENTRY_FALLBACKS:
        target = pattern.format(kernel=base)
        if target in pair_lookup:
            return target, host_names, pair_lookup[target]
    # 3. Score-based fallback.
    def score(name: str, n_args: int) -> int:
        s = 0
        # name affinity to the kernel base ("rms_norm" → +5 on "rms_norm",
        # "rmsnorm2d_fwd_with_add", etc.; case- and underscore-insensitive).
        base_compact = base.replace("_", "").lower()
        name_compact = name.replace("_", "").lower()
        if base_compact and (base_compact in name_compact or name_compact in base_compact):
            s += 5
        # arg-count proximity (favour launchers that take ~num_shape_args
        # tensor inputs plus 0..2 scalars like epsilon / dtype).
        if num_shape_args:
            if num_shape_args <= n_args <= num_shape_args + 3:
                s += 10
            elif n_args < num_shape_args:
                s -= 5  # missing required tensor args → definitely not the launcher
        # Penalize private / very-short helpers.
        if name.startswith("_"):
            s -= 3
        if n_args <= 1 and num_shape_args >= 2:
            # 1-arg helpers can never be the launcher when the kernel
            # takes multiple tensor inputs.
            s -= 6
        return s

    ranked = sorted(
        host_pairs,
        key=lambda p: (-score(p[0], p[1]), host_pairs.index(p)),
    )
    best_name, best_nargs = ranked[0]
    return best_name, host_names, best_nargs


# ---------------------------------------------------------------------------
# Test data generation
# ---------------------------------------------------------------------------

def _render_arg_init(
    idx: int, shape: list[int], dtype_expr: str, seed_offset: int,
) -> str:
    """Return Python source that builds one positional argument tensor."""
    shape_repr = ", ".join(str(d) for d in shape)
    # Build trailing comma so single-element tuples stay tuples.
    if len(shape) == 1:
        shape_literal = f"({shape[0]},)"
    else:
        shape_literal = f"({shape_repr})"
    if _is_integer_dtype_expr(dtype_expr):
        # Use a small positive range so toy reductions / cumsum tests stay
        # numerically stable. Vocab-style int32 inputs (e.g. token ids) are
        # naturally bounded by the second dimension when present.
        upper = max(2, shape[-1] if shape else 2)
        return (
            f"    arg{idx} = torch.randint(0, {upper}, {shape_literal}, "
            f"dtype={dtype_expr}, device=device)"
        )
    if "float8" in dtype_expr:
        # FP8 doesn't accept randn directly; materialize in fp16 then cast.
        return (
            f"    arg{idx} = torch.randn({shape_literal}, "
            f"dtype=torch.float16, device=device).to({dtype_expr})"
        )
    return (
        f"    arg{idx} = torch.randn({shape_literal}, "
        f"dtype={dtype_expr}, device=device)"
    )


def _render_test_cases(
    shapes: list[list[int]], dtypes: list[str], extra_scalar_args: int = 0,
) -> tuple[str, str]:
    """Return ``(test_shapes_block, args_init_block)``.

    The args init block initializes ``arg0`` .. ``argN-1`` for one test
    case and is parameterized over the loop variable ``test_idx``.

    ``extra_scalar_args`` reserves trailing slots for non-tensor arguments
    the host launcher expects (e.g. ``epsilon`` for RMSNorm, ``num_heads``
    for attention launchers). They're filled with sensible defaults
    (``1e-6`` then ``1`` ...) so the resulting tuple matches the launcher's
    arity without us having to introspect signatures at runtime.
    """
    if not shapes and not extra_scalar_args:
        return "TEST_SHAPES = []", "    args = ()"
    shape_lines: list[str] = ["TEST_SHAPES = ["]
    for s in shapes:
        shape_lines.append(f"    {tuple(s)!r},  # arg shape")
    shape_lines.append("]")
    arg_lines: list[str] = []
    for i, (shape, dtype_expr) in enumerate(zip(shapes, dtypes)):
        arg_lines.append(_render_arg_init(i, shape, dtype_expr, seed_offset=i))
    scalar_defaults = ("1e-6", "1", "False", "None")
    scalar_exprs = []
    for j in range(extra_scalar_args):
        val = scalar_defaults[j] if j < len(scalar_defaults) else "None"
        var = f"scalar{j}"
        arg_lines.append(f"    {var} = {val}  # auto-filled scalar arg (no shape captured)")
        scalar_exprs.append(var)
    args_tuple = list(f"arg{i}" for i in range(len(shapes))) + scalar_exprs
    arg_lines.append("    args = (" + ", ".join(args_tuple) + ",)")
    return "\n".join(shape_lines), "\n".join(arg_lines)


# ---------------------------------------------------------------------------
# Generator entry point
# ---------------------------------------------------------------------------

_TASK_RUNNER_TEMPLATE = '''#!/usr/bin/env python3
"""Auto-generated by Hyperloom unittest_agent.

Reflects the end-to-end runtime for kernel `{kernel_name}` as observed in
the live vLLM/SGLang profile that produced this candidate.

DO NOT EDIT BY HAND — regenerated on every kernel-opt request.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent.parent
TASK_NAME = {task_name!r}
SOURCE_FILE = str(TASK_DIR / "source" / {source_basename!r})
BASELINE_SNAPSHOT = str(TASK_DIR / "source" / "_baseline_snapshot" / {source_basename!r})
HOST_ENTRY = {host_entry!r}
TARGET_KERNELS = {target_kernels!r}

# End-to-end environment captured from the live serving process. These are
# the SGLang / vLLM / aiter / triton flags that were active when the profile
# trace was taken; exporting them here keeps the unittest faithful to the
# runtime shape that GEAK is supposed to optimize for.
RUNTIME_ENV = {env_vars!r}
for _k, _v in RUNTIME_ENV.items():
    os.environ.setdefault(str(_k), str(_v))

{test_shapes_block}

# Per-arg dtype expressions, mirrored from the TraceLens row / SGLang
# tensor dtype (with float16 as the universal fallback).
TEST_DTYPES = [{test_dtypes_repr}]

WARMUP_ITERATIONS = {warmup_iters}
BENCHMARK_ITERATIONS = {bench_iters}

# Tolerances; relaxed for fp16/bf16, tightened for fp32. fp8 tolerances are
# intentionally loose because the kernel may quantize on the host side.
DEFAULT_ATOL = {default_atol}
DEFAULT_RTOL = {default_rtol}


def _load_module(path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {{path}}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _resolve_callable(mod, name: str):
    fn = getattr(mod, name, None)
    if fn is None:
        raise AttributeError(f"module {{mod.__file__!r}} has no callable {{name!r}}")
    return fn


def _materialize_args(test_idx: int):
    """Build the positional args tuple for one test case. Pure-python so
    both correctness and performance reuse the same code path."""
    import torch
    device = "cuda"
    if test_idx >= len(TEST_SHAPES):
        raise IndexError(f"test_idx {{test_idx}} >= TEST_SHAPES len {{len(TEST_SHAPES)}}")
    torch.manual_seed(42 + test_idx)
{materialize_args_block}
    return args


def _to_cpu(obj):
    """Recursively move tensors to CPU for golden comparison."""
    import torch
    if isinstance(obj, torch.Tensor):
        return obj.detach().to(torch.float32).cpu()
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(x) for x in obj)
    if isinstance(obj, dict):
        return {{k: _to_cpu(v) for k, v in obj.items()}}
    return obj


def _flatten_tensors(obj):
    import torch
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (list, tuple)):
        out = []
        for x in obj:
            out.extend(_flatten_tensors(x))
        return out
    if isinstance(obj, dict):
        out = []
        for v in obj.values():
            out.extend(_flatten_tensors(v))
        return out
    return []


def run_compile():
    """AST-parse + import. Mirrors AgentKernelArena's compile gate."""
    try:
        import ast as _ast
        with open(SOURCE_FILE, "r", encoding="utf-8") as f:
            source = f.read()
        _ast.parse(source)
        mod = _load_module(SOURCE_FILE, "candidate_kernel_under_test")
        for sym in TARGET_KERNELS:
            if not hasattr(mod, sym):
                return False, f"missing required symbol: {{sym!r}}"
        if HOST_ENTRY and not hasattr(mod, HOST_ENTRY):
            return False, f"missing host entry point: {{HOST_ENTRY!r}}"
        return True, None
    except Exception as exc:  # noqa: BLE001 - want full diagnostic
        return False, f"{{type(exc).__name__}}: {{exc}}"


def run_correctness():
    """Compare current SOURCE_FILE against BASELINE_SNAPSHOT (golden bytes
    captured at generation time). Both modules are loaded under distinct
    module names so torch/triton cache misses are isolated.

    The runner only emits PASS when *both* shapes succeed AND outputs match
    within DEFAULT_ATOL/DEFAULT_RTOL.
    """
    import torch
    if not TEST_SHAPES:
        return False, "no input shapes captured at generation time; cannot run correctness"
    if not os.path.exists(BASELINE_SNAPSHOT):
        return False, f"baseline snapshot missing: {{BASELINE_SNAPSHOT}}"
    if not HOST_ENTRY:
        return False, "no host entry point resolved; cannot drive correctness"

    try:
        cur_mod = _load_module(SOURCE_FILE, "candidate_kernel_current")
        ref_mod = _load_module(BASELINE_SNAPSHOT, "candidate_kernel_baseline")
    except Exception as exc:  # noqa: BLE001
        return False, f"module load failed: {{type(exc).__name__}}: {{exc}}"

    try:
        cur_fn = _resolve_callable(cur_mod, HOST_ENTRY)
        ref_fn = _resolve_callable(ref_mod, HOST_ENTRY)
    except Exception as exc:  # noqa: BLE001
        return False, f"entry-point resolve failed: {{exc}}"

    # Collect per-shape pass/fail rather than aborting on the first failure.
    # Callers (GEAK select_agent, orchestrator) need the full picture so they
    # can tell apart "one shape regressed" from "everything broke", and so a
    # partial result is still actionable.
    per_shape: list[dict] = []
    first_error: str | None = None
    for test_idx in range(len(TEST_SHAPES)):
        shape_label = list(TEST_SHAPES[test_idx])
        try:
            ref_args = _materialize_args(test_idx)
            cur_args = tuple(a.clone() if hasattr(a, "clone") else a for a in ref_args)

            ref_out = ref_fn(*ref_args)
            torch.cuda.synchronize()
            cur_out = cur_fn(*cur_args)
            torch.cuda.synchronize()

            ref_tensors = _flatten_tensors(ref_out)
            cur_tensors = _flatten_tensors(cur_out)
            if not ref_tensors and not cur_tensors:
                ref_tensors = _flatten_tensors(list(ref_args))
                cur_tensors = _flatten_tensors(list(cur_args))
            if len(ref_tensors) != len(cur_tensors):
                msg = (f"shape {{test_idx + 1}}: tensor count mismatch "
                       f"(ref={{len(ref_tensors)}}, current={{len(cur_tensors)}})")
                per_shape.append({{"shape": shape_label, "status": "fail", "error": msg}})
                if first_error is None:
                    first_error = msg
                continue
            shape_ok = True
            shape_error: str | None = None
            shape_max_diff = 0.0
            for ti, (r, c) in enumerate(zip(ref_tensors, cur_tensors)):
                if r.shape != c.shape:
                    shape_ok = False
                    shape_error = (f"shape {{test_idx + 1}} output {{ti}}: shape mismatch "
                                   f"(ref={{tuple(r.shape)}}, current={{tuple(c.shape)}})")
                    break
                rf = r.to(torch.float32).cpu()
                cf = c.to(torch.float32).cpu()
                if not torch.allclose(rf, cf, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL):
                    shape_ok = False
                    diff = (rf - cf).abs().max().item()
                    shape_error = (f"shape {{test_idx + 1}} output {{ti}}: "
                                   f"max abs diff={{diff:.6g}} > atol={{DEFAULT_ATOL}}")
                    shape_max_diff = max(shape_max_diff, float(diff))
                    break
                shape_max_diff = max(shape_max_diff,
                                     float((rf - cf).abs().max().item()))
            if shape_ok:
                per_shape.append({{"shape": shape_label, "status": "ok",
                                   "max_diff": shape_max_diff}})
            else:
                per_shape.append({{"shape": shape_label, "status": "fail",
                                   "error": shape_error,
                                   "max_diff": shape_max_diff}})
                if first_error is None and shape_error:
                    first_error = shape_error
            # Release per-shape allocations so the next shape starts cleanly.
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
        except Exception as exc:  # noqa: BLE001
            msg = f"shape {{test_idx + 1}} raised: {{type(exc).__name__}}: {{exc}}"
            per_shape.append({{"shape": shape_label, "status": "fail", "error": msg}})
            if first_error is None:
                first_error = msg

    failed = [p for p in per_shape if p.get("status") != "ok"]
    summary = {{"per_shape": per_shape,
                "num_pass": len(per_shape) - len(failed),
                "num_fail": len(failed)}}
    if failed:
        return False, first_error or "one or more shapes failed correctness", summary
    return True, None, summary


def run_performance():
    """Time the *current* SOURCE_FILE with the captured shapes. Mirrors
    AgentKernelArena's performance schema (test_case_id + execution_time_ms
    + params)."""
    import torch
    if not TEST_SHAPES:
        return [{{"test_case_id": "perf0", "execution_time_ms": -1.0,
                  "params": {{"note": "no shapes captured"}}}}]
    try:
        mod = _load_module(SOURCE_FILE, "candidate_kernel_perf")
        fn = _resolve_callable(mod, HOST_ENTRY) if HOST_ENTRY else None
        if fn is None:
            return [{{"test_case_id": "perf0", "execution_time_ms": -1.0,
                      "params": {{"note": "no host entry"}}}}]
    except Exception as exc:  # noqa: BLE001
        return [{{"test_case_id": "perf0", "execution_time_ms": -1.0,
                  "params": {{"error": f"{{type(exc).__name__}}: {{exc}}"}}}}]

    test_cases = []
    for test_idx in range(len(TEST_SHAPES)):
        try:
            args = _materialize_args(test_idx)
            for _ in range(WARMUP_ITERATIONS):
                fn(*[a.clone() if hasattr(a, "clone") else a for a in args])
            torch.cuda.synchronize()
            n_iter = BENCHMARK_ITERATIONS
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
            for j in range(n_iter):
                starts[j].record()
                fn(*[a.clone() if hasattr(a, "clone") else a for a in args])
                ends[j].record()
            torch.cuda.synchronize()
            times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
            elapsed_ms = sum(times) / len(times)
            test_cases.append({{
                "test_case_id": f"perf{{test_idx + 1}}",
                "execution_time_ms": elapsed_ms,
                "params": {{"shape": list(TEST_SHAPES[test_idx])}},
            }})
        except Exception as exc:  # noqa: BLE001
            test_cases.append({{
                "test_case_id": f"perf{{test_idx + 1}}",
                "execution_time_ms": -1.0,
                "params": {{"shape": list(TEST_SHAPES[test_idx]),
                            "error": f"{{type(exc).__name__}}: {{exc}}"}},
            }})
    return test_cases


def main() -> int:
    parser = argparse.ArgumentParser(description=f"Task runner for {{TASK_NAME}}")
    parser.add_argument("mode", choices=["compile", "correctness", "performance"])
    args = parser.parse_args()

    build_dir = TASK_DIR / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "compile":
        ok, err = run_compile()
        (build_dir / "compile_report.json").write_text(
            json.dumps({{"status": "ok" if ok else "fail", "error": err}}, indent=2),
            encoding="utf-8",
        )
        print(f"Compilation: {{'PASS' if ok else 'FAIL'}}")
        if err:
            print(f"Error: {{err}}")
        return 0 if ok else 1

    if args.mode == "correctness":
        result = run_correctness()
        # ``run_correctness`` may return a 2- or 3-tuple depending on whether
        # the gate ran (we collect per-shape detail on success/failure but
        # short-circuit with ``(False, err)`` for hard errors like missing
        # baseline). Normalize both shapes here.
        summary: dict = {{}}
        if isinstance(result, tuple) and len(result) == 3:
            ok, err, summary = result
        else:
            ok, err = result  # type: ignore[misc]
        (build_dir / "correctness_report.json").write_text(
            json.dumps({{"status": "ok" if ok else "fail",
                         "error": err,
                         "num_shapes": len(TEST_SHAPES),
                         "per_shape": (summary or {{}}).get("per_shape", []),
                         "num_pass": (summary or {{}}).get("num_pass"),
                         "num_fail": (summary or {{}}).get("num_fail")}},
                       indent=2),
            encoding="utf-8",
        )
        print(f"Correctness: {{'PASS' if ok else 'FAIL'}}")
        for entry in (summary or {{}}).get("per_shape", []) or []:
            tag = entry.get("status", "?").upper()
            print(f"  shape={{entry.get('shape')}} {{tag}} "
                  f"max_diff={{entry.get('max_diff')!s}}")
        if err:
            print(f"Error: {{err}}")
        return 0 if ok else 1

    if args.mode == "performance":
        cases = run_performance()
        (build_dir / "performance_report.json").write_text(
            json.dumps(cases, indent=2), encoding="utf-8",
        )
        good = [c for c in cases if c.get("execution_time_ms", -1) > 0]
        # Per-shape parseable latency lines so the GEAK select_agent
        # (``parse_shape_latencies_ms`` regex) can compare patches shape-by-shape
        # even when a subset failed. Matches the ``(shape): X ms`` format.
        for c in good:
            params = c.get("params") or {{}}
            shape = params.get("shape") or params.get("input_dims") or []
            if shape:
                print(f"({{list(shape)}}): {{c['execution_time_ms']:.4f}} ms")
        if good:
            total = sum(c["execution_time_ms"] for c in good)
            print(f"Performance: measured {{len(good)}}/{{len(cases)}} cases, "
                  f"total time: {{total:.4f}} ms")
        else:
            print("Performance: FAILED - no test cases measured")
        # Mirror HIP runner: non-zero when nothing measured so callers can
        # tell apart "everything ok" from "every shape errored out".
        return 0 if good else 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
'''


_HIP_TASK_RUNNER_TEMPLATE = '''#!/usr/bin/env python3
"""Auto-generated HIP/C++ harness by Hyperloom unittest_agent.

This runner wraps an existing e2e-style benchmark command for `{kernel_name}`.
For each correctness/performance run it temporarily overlays the edited source
mirror onto the live framework source path, invalidates likely aiter JIT
modules, executes the benchmark, then restores the live source and JIT files.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent.parent
TASK_NAME = {task_name!r}
SOURCE_FILE = TASK_DIR / "source" / {source_basename!r}
BASELINE_SNAPSHOT = TASK_DIR / "source" / "_baseline_snapshot" / {source_basename!r}
LIVE_SOURCE = Path({live_source!r})
KERNEL_REPO = Path({kernel_repo!r}) if {kernel_repo!r} else None
BENCHMARK_COMMANDS = {benchmark_commands!r}
TARGET_KERNELS = {target_kernels!r}
RUNTIME_ENV = {env_vars!r}
JIT_CANDIDATE_ROOTS = [Path(p) for p in {jit_roots!r}]
JIT_MATCH_TOKENS = {jit_match_tokens!r}
SHAPE_CASES = {shape_cases!r}
# Generation-time computed defaults. Env vars still win at runtime so callers
# (orchestrator / GEAK config) can override per-attempt. Both are sized for
# the worst-case "first overlay → JIT re-compile (~60s) → N shape benchmark"
# pattern actually observed for aiter rmsnorm_quant / silu_and_mul on MI300X.
_DEFAULT_CORRECTNESS_TIMEOUT_SEC = {default_correctness_timeout}
_DEFAULT_PERFORMANCE_TIMEOUT_SEC = {default_performance_timeout}
_DEFAULT_PER_SHAPE_TIMEOUT_SEC = {default_per_shape_timeout}

for _k, _v in RUNTIME_ENV.items():
    os.environ.setdefault(str(_k), str(_v))


def _run(cmd: str, *, timeout: int) -> tuple[int, str, float]:
    started = time.time()
    env = os.environ.copy()
    env.setdefault("UNITTEST_AGENT_LIVE_SOURCE", str(LIVE_SOURCE))
    env.setdefault("UNITTEST_AGENT_SOURCE_FILE", str(SOURCE_FILE))
    proc = subprocess.run(
        cmd, shell=True, text=True, capture_output=True,
        timeout=timeout, cwd=str(KERNEL_REPO) if KERNEL_REPO else None, env=env,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, output, time.time() - started


def _candidate_jit_files() -> list[Path]:
    files: list[Path] = []
    for root in JIT_CANDIDATE_ROOTS:
        if not root.is_dir():
            continue
        for p in root.glob("*.so"):
            name = p.name.lower()
            if JIT_MATCH_TOKENS and all(tok in name for tok in JIT_MATCH_TOKENS):
                files.append(p)
    return sorted(set(files))


def _overlay_lockfile() -> Path:
    """System-wide lock so concurrent kernel_opt runs don't race the shared
    aiter JIT cache (each ``_OverlayLiveSource`` unlinks ``module_*.so`` from
    ``/sgl-workspace/aiter/aiter/jit/`` — two parallel overlays would otherwise
    interleave unlink+overwrite+import and end up running each other's source).
    """
    lock_root = Path(os.environ.get("HYPERLOOM_UNITTEST_OVERLAY_LOCK_DIR", "/tmp"))
    try:
        lock_root.mkdir(parents=True, exist_ok=True)
    except OSError:
        lock_root = Path(tempfile.gettempdir())
    return lock_root / f"hl_overlay_{{LIVE_SOURCE.name}}.lock"


def _purge_stale_aiter_batons(grace_sec: int = 60) -> list[str]:
    """Remove aiter ``FileBaton`` files (``build/lock_module_*``) left behind
    by a previous compile that was SIGKILL/SIGTERM'd before
    ``baton.release()`` ran. Without this, the next ``import
    aiter.jit.module_<name>`` deadlocks forever inside ``baton.wait()`` (an
    unconditional spin on ``os.path.exists(lock)``).

    Only purge a lock when (a) it is older than ``grace_sec`` and (b) no
    currently-running ``hipcc`` / ``ninja`` / ``clang++`` / ``amdclang``
    process is targeting the matching module name. We discover targets by
    scanning ``/proc/*/cmdline`` for tokens starting with ``module_``. Caller
    must hold the overlay file-lock to serialize this against concurrent
    Hyperloom overlay enter/exits.

    Importantly, we enumerate batons by globbing
    ``<jit_root>/build/lock_module_*`` rather than deriving the lock name
    from the matching ``<jit_root>/*.so`` file. The ``.so`` may not exist
    yet (first-ever compile of that module → ninja creates the lock
    before producing any output), and that's exactly the case where a
    SIGKILL leaves the most damaging baton behind. Limiting purge scope
    to "modules that already have a ``.so``" was the 2026-05-21
    ``module_rmsnorm_quant`` hang: the kernel never finished its first
    compile so the ``.so`` was missing, the lock was stuck, and every
    subsequent ``aiter`` import spun in ``baton.wait()``.
    """
    purged: list[str] = []
    builder_keywords = ("hipcc", "ninja", "clang++", "amdclang")
    active_modules: set[str] = set()
    try:
        proc_pids = [pid for pid in os.listdir("/proc") if pid.isdigit()]
    except OSError:
        return purged
    for pid in proc_pids:
        try:
            with open(f"/proc/{{pid}}/comm", "r") as f:
                comm = f.read().strip()
        except OSError:
            continue
        if not any(kw in comm for kw in builder_keywords):
            continue
        try:
            with open(f"/proc/{{pid}}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\\x00", b" ").decode("utf-8", "ignore")
        except OSError:
            continue
        for tok in cmd.split():
            idx = tok.find("module_")
            if idx < 0:
                continue
            tail = tok[idx:].split("/")[0]
            # Strip trailing punctuation like commas, semicolons.
            tail = tail.rstrip(",;:()'\\\"")
            if tail.endswith(".so"):
                tail = tail[:-3]
            active_modules.add(tail)
    now = time.time()
    seen_locks: set[str] = set()
    for root in JIT_CANDIDATE_ROOTS:
        build_root = root / "build"
        if not build_root.is_dir():
            continue
        for lock_file in sorted(build_root.glob("lock_module_*")):
            try:
                stem = lock_file.name[len("lock_"):]  # module_xxxx
            except ValueError:
                continue
            # Optional scoping: when JIT_MATCH_TOKENS is set, prefer purging
            # baton names that contain ALL tokens (avoids accidentally
            # racing batons owned by a sibling kernel-opt run on the same
            # node). Locks owned by a still-alive compile are always kept.
            stem_lower = stem.lower()
            if JIT_MATCH_TOKENS and not all(
                tok in stem_lower for tok in JIT_MATCH_TOKENS
            ):
                # Still purge unrelated stale locks too — leaving them
                # poisons sibling imports — but be conservative on grace.
                pass
            if stem in active_modules:
                continue
            try:
                mtime = lock_file.stat().st_mtime
            except OSError:
                continue
            if now - mtime < grace_sec:
                continue
            try:
                lock_file.unlink()
                purged.append(str(lock_file))
                seen_locks.add(stem)
            except OSError:
                pass
            # Also drop the inner ninja per-build lock if it's the same
            # stale window — the outer aiter baton was the only thing
            # blocking import, but a stuck inner lock prevents fresh
            # ninja invocations from making progress.
            inner_lock = build_root / stem / "build" / "lock"
            if inner_lock.exists():
                try:
                    if now - inner_lock.stat().st_mtime >= grace_sec:
                        inner_lock.unlink()
                        purged.append(str(inner_lock))
                except OSError:
                    pass
    return purged


class _OverlayLiveSource:
    def __init__(self, overlay_source: Path | None = None):
        self.overlay_source = overlay_source or SOURCE_FILE
        self.lock_fd = None
        self.skipped_overlay = False
        self.tmp: Path | None = None
        self.live_backup: Path | None = None
        self.jit_backups: list[tuple[Path, Path]] = []

    @staticmethod
    def _sha1(path: Path) -> str:
        import hashlib
        try:
            return hashlib.sha1(path.read_bytes()).hexdigest()
        except OSError:
            return ""

    def __enter__(self):
        if not LIVE_SOURCE.is_file():
            raise FileNotFoundError(f"live source missing: {{LIVE_SOURCE}}")
        if not self.overlay_source.is_file():
            raise FileNotFoundError(f"overlay source missing: {{self.overlay_source}}")
        # Acquire exclusive lock on the shared live-source path so concurrent
        # GEAK runs don't trample each other's JIT cache. Best-effort: if
        # fcntl is unavailable (e.g. non-POSIX) we silently skip locking.
        try:
            import fcntl
            lock_path = _overlay_lockfile()
            self.lock_fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX)
        except Exception:  # noqa: BLE001 — locking is non-fatal
            self.lock_fd = None
        # Now that we hold the overlay lock, sweep aiter baton files left
        # behind by SIGKILL'd compiles. Without this the next
        # ``import aiter.jit.module_<name>`` would spin forever in
        # ``FileBaton.wait()`` (observed after our run got SIGTERM mid-JIT
        # in the previous session). Purge is best-effort and silent on
        # failure so we never block overlay enter.
        try:
            purged = _purge_stale_aiter_batons()
            if purged:
                sys.stderr.write(
                    f"[unittest_agent] purged stale aiter batons: {{purged}}\\n"
                )
        except Exception:  # noqa: BLE001 — purge is best-effort
            pass
        # Fast no-op path: when the overlay candidate is byte-identical to
        # the live source (i.e. GEAK hasn't produced a new patch yet, or
        # the overlay IS the baseline snapshot for the save phase running
        # against an unchanged live tree), skip the JIT ``.so`` unlink +
        # ``shutil.copy2`` entirely. The kept ``.so`` cache lets the next
        # ``import aiter.jit.module_<name>`` reuse the existing build
        # instead of re-running ninja (~60–90s/module on aiter CK-Tile).
        # This reliably halves correctness wall-time on the first GEAK
        # round and is essentially free on subsequent rounds where the
        # mirror diverges (we fall through to the slow path normally).
        live_hash = self._sha1(LIVE_SOURCE)
        overlay_hash = self._sha1(self.overlay_source)
        if live_hash and live_hash == overlay_hash:
            self.skipped_overlay = True
            sys.stderr.write(
                f"[unittest_agent] overlay no-op (mirror==live, "
                f"sha1={{live_hash[:12]}}); skipping JIT invalidate + copy\\n"
            )
            return self
        self.tmp = Path(tempfile.mkdtemp(prefix="hl_hip_unittest_"))
        self.live_backup = self.tmp / LIVE_SOURCE.name
        shutil.copy2(LIVE_SOURCE, self.live_backup)
        for jit in _candidate_jit_files():
            dst = self.tmp / jit.name
            shutil.copy2(jit, dst)
            self.jit_backups.append((jit, dst))
            try:
                jit.unlink()
            except OSError:
                pass
        shutil.copy2(self.overlay_source, LIVE_SOURCE)
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if not self.skipped_overlay:
                if (self.live_backup is not None
                        and self.live_backup.exists()):
                    shutil.copy2(self.live_backup, LIVE_SOURCE)
                for jit, backup in self.jit_backups:
                    try:
                        if backup.exists():
                            shutil.copy2(backup, jit)
                    except OSError:
                        pass
        finally:
            if self.tmp is not None:
                shutil.rmtree(self.tmp, ignore_errors=True)
            if self.lock_fd is not None:
                try:
                    import fcntl
                    fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
                except Exception:  # noqa: BLE001
                    pass
                try:
                    os.close(self.lock_fd)
                except OSError:
                    pass
                self.lock_fd = None
        return False


def run_compile():
    try:
        if not SOURCE_FILE.is_file():
            return False, f"missing generated source: {{SOURCE_FILE}}"
        if not BASELINE_SNAPSHOT.is_file():
            return False, f"missing baseline snapshot: {{BASELINE_SNAPSHOT}}"
        if not LIVE_SOURCE.is_file():
            return False, f"missing live source: {{LIVE_SOURCE}}"
        if not BENCHMARK_COMMANDS:
            return False, "no benchmark command captured for HIP source"
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, f"{{type(exc).__name__}}: {{exc}}"


def _supported_shape_cases() -> list[dict]:
    out = []
    for case in SHAPE_CASES:
        op = str(case.get("op", ""))
        dims = case.get("input_dims") or []
        if not dims:
            continue
        if "aiter::rmsnorm" in op and len(dims) >= 3:
            out.append(case)
        elif "silu_and_mul" in op and len(dims) >= 2:
            out.append(case)
    return out


def _shape_case_script(mode: str, path_a: Path | None = None, path_b: Path | None = None) -> str:
    cases_json = json.dumps(_supported_shape_cases())
    path_a_s = str(path_a) if path_a is not None else ""
    path_b_s = str(path_b) if path_b is not None else ""
    return f"""
import json, time, torch
cases = json.loads({{cases_json!r}})
mode = {{mode!r}}
path_a = {{path_a_s!r}}
path_b = {{path_b_s!r}}

def _dtype(type_names):
    text = ' '.join(str(x) for x in (type_names or [])).lower()
    if 'bfloat16' in text or 'bf16' in text:
        return torch.bfloat16
    if 'float16' in text or 'half' in text or 'fp16' in text:
        return torch.float16
    if 'float32' in text or 'fp32' in text:
        return torch.float32
    return torch.bfloat16

def _make(shape, dtype):
    return torch.randn(tuple(int(x) for x in shape), device='cuda', dtype=dtype)

def _run_case(case, seed):
    import aiter
    torch.manual_seed(seed)
    op = str(case.get('op', ''))
    dims = case.get('input_dims') or []
    type_names = case.get('input_types') or []
    dtype = _dtype(type_names)
    if 'aiter::rmsnorm' in op:
        # TraceLens dims are usually (input, output, weight, epsilon).
        input_shape = dims[0]
        output_shape = dims[1] if len(dims) > 1 and dims[1] else dims[0]
        weight_shape = dims[2] if len(dims) > 2 and dims[2] else [input_shape[-1]]
        x = _make(input_shape, dtype)
        out = torch.empty(tuple(output_shape), device='cuda', dtype=dtype)
        weight = _make(weight_shape, dtype)
        aiter.rmsnorm(out, x, weight, 1e-6)
        torch.cuda.synchronize()
        return [out.detach().float().cpu()]
    if 'silu_and_mul' in op:
        # TraceLens dims are usually (output, input) for silu_and_mul.
        output_shape = dims[0]
        input_shape = dims[1] if len(dims) > 1 and dims[1] else [output_shape[0], output_shape[-1] * 2]
        x = _make(input_shape, dtype)
        out = torch.empty(tuple(output_shape), device='cuda', dtype=dtype)
        aiter.silu_and_mul(out, x)
        torch.cuda.synchronize()
        return [out.detach().float().cpu()]
    raise RuntimeError('unsupported op: ' + op)

def _all_outputs():
    outputs = []
    for idx, case in enumerate(cases):
        outputs.append(_run_case(case, 1234 + idx))
    return outputs

if not cases:
    print(json.dumps({{{{'status': 'unsupported', 'reason': 'no supported shape cases'}}}}))
    raise SystemExit(2)

if mode == 'save':
    outs = _all_outputs()
    torch.save(outs, path_a)
    print(json.dumps({{{{'status': 'ok', 'num_cases': len(outs)}}}}))
elif mode == 'compare':
    ref = torch.load(path_a)
    cur = _all_outputs()
    max_diff = 0.0
    per_case = []
    failed_summary = None
    for case_idx, (rr, cc) in enumerate(zip(ref, cur)):
        case_ok = True
        case_err = None
        case_diff = 0.0
        if len(rr) != len(cc):
            case_ok = False
            case_err = 'tensor count mismatch'
        else:
            for tensor_idx, (r, c) in enumerate(zip(rr, cc)):
                if tuple(r.shape) != tuple(c.shape):
                    case_ok = False
                    case_err = f'shape mismatch case={{{{case_idx}}}} tensor={{{{tensor_idx}}}}'
                    break
                diff = (r - c).abs().max().item()
                case_diff = max(case_diff, float(diff))
                max_diff = max(max_diff, float(diff))
                if not torch.allclose(r, c, atol=1e-2, rtol=1e-2):
                    case_ok = False
                    case_err = f'allclose failed case={{{{case_idx}}}} tensor={{{{tensor_idx}}}} max_diff={{{{diff}}}}'
                    break
        per_case.append({{{{'case_idx': case_idx, 'status': 'ok' if case_ok else 'fail',
                         'max_diff': case_diff, 'error': case_err}}}})
        if not case_ok and failed_summary is None:
            failed_summary = case_err
    status = 'ok' if all(p['status'] == 'ok' for p in per_case) else 'fail'
    print(json.dumps({{{{'status': status, 'num_cases': len(cur), 'max_diff': max_diff,
                      'per_case': per_case, 'error': failed_summary}}}}))
    if status != 'ok':
        raise SystemExit(failed_summary or 'one or more cases failed')
elif mode == 'perf':
    results = []
    for idx, case in enumerate(cases):
        for _ in range(5):
            _run_case(case, 5678 + idx)
        torch.cuda.synchronize()
        n = 20
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        for j in range(n):
            starts[j].record()
            _run_case(case, 5678 + idx)
            ends[j].record()
        torch.cuda.synchronize()
        elapsed = sum(s.elapsed_time(e) for s, e in zip(starts, ends)) / n
        results.append({{{{'test_case_id': f'shape_{{{{idx+1}}}}', 'execution_time_ms': elapsed, 'params': case}}}})
        # Free GPU memory between shapes so the next case isn't OOM-killed
        # (large tensors like (1024, 24576) stay resident otherwise).
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    print(json.dumps({{{{'status': 'ok', 'results': results}}}}))
else:
    raise SystemExit('unknown mode')
"""


def _shape_subprocess_budget(timeout: int, num_cases: int) -> int:
    """Pick a subprocess budget for the multi-shape ``python -c`` runner.

    Each ``_run_shape_subprocess`` call may need (a) one JIT recompile of the
    overlaid kernel (~60s/module on aiter+CK-Tile) plus (b) running ``num_cases``
    correctness / perf cases serially.  We give each shape ``_DEFAULT_PER_SHAPE_TIMEOUT_SEC``
    of work and add a flat 90s JIT/import warmup floor, then cap by the
    outer-level ``timeout`` so we never exceed what the caller budgeted.
    """
    per_shape = max(60, int(_DEFAULT_PER_SHAPE_TIMEOUT_SEC))
    base = max(120, per_shape * max(1, num_cases) + 90)
    if timeout > 0:
        return max(60, min(timeout, base))
    return base


def _run_shape_subprocess(mode: str, *, path_a: Path | None = None, path_b: Path | None = None, timeout: int) -> tuple[int, str, float]:
    started = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _shape_case_script(mode, path_a=path_a, path_b=path_b)],
            text=True, capture_output=True, timeout=timeout,
            cwd=str(KERNEL_REPO) if KERNEL_REPO else None, env=os.environ.copy(),
        )
    except subprocess.TimeoutExpired as exc:
        captured = ((exc.stdout or "") + (exc.stderr or ""))
        return 124, captured + f"\\n[task_runner] subprocess timeout after {{timeout}}s (mode={{mode}})\\n", time.time() - started
    return proc.returncode, (proc.stdout or "") + (proc.stderr or ""), time.time() - started


def _run_shape_correctness(timeout: int) -> tuple[bool, str | None, float, str]:
    cases = _supported_shape_cases()
    if not cases:
        return False, "no supported generated shape cases", 0.0, ""
    started = time.time()
    sub_timeout = _shape_subprocess_budget(timeout, len(cases))
    with tempfile.TemporaryDirectory(prefix="hl_hip_shape_ref_") as tmp:
        ref_path = Path(tmp) / "ref.pt"
        with _OverlayLiveSource(BASELINE_SNAPSHOT):
            rc1, out1, _ = _run_shape_subprocess("save", path_a=ref_path, timeout=sub_timeout)
        if rc1 != 0:
            tag = "baseline shape run timed out" if rc1 == 124 else f"baseline shape run rc={{rc1}}"
            return False, tag, time.time() - started, out1
        # Recompute remaining budget so a slow baseline run never starves the
        # compare phase below.
        remaining = max(60, timeout - int(time.time() - started)) if timeout > 0 else sub_timeout
        sub_timeout_compare = _shape_subprocess_budget(remaining, len(cases))
        with _OverlayLiveSource(SOURCE_FILE):
            rc2, out2, _ = _run_shape_subprocess("compare", path_a=ref_path, timeout=sub_timeout_compare)
        if rc2 != 0:
            tag = "current shape run timed out" if rc2 == 124 else f"current shape run rc={{rc2}}"
            return False, tag, time.time() - started, out1 + out2
        return True, None, time.time() - started, out1 + out2


def _run_shape_performance(timeout: int) -> list[dict] | None:
    cases = _supported_shape_cases()
    if not cases:
        return None
    sub_timeout = _shape_subprocess_budget(timeout, len(cases))
    with _OverlayLiveSource(SOURCE_FILE):
        rc, output, elapsed = _run_shape_subprocess("perf", timeout=sub_timeout)
    if rc != 0:
        err = "shape perf timed out" if rc == 124 else f"shape perf rc={{rc}}"
        return [{{
            "test_case_id": "hip_shape_generated",
            "execution_time_ms": -1.0,
            "params": {{"error": err, "stdout_tail": output[-2000:]}},
        }}]
    try:
        payload = json.loads(output.strip().splitlines()[-1])
        results = payload.get("results") or []
        # Emit machine-parseable per-shape latency lines so the upstream
        # select_agent (which groups by ``(shape): X ms``) can compare patches
        # even when other shapes had partial issues. Format matches GEAK's
        # benchmark_parsing.parse_shape_latencies_ms regex.
        for r in results:
            params = r.get("params") or {{}}
            dims = params.get("input_dims") or params.get("shape") or []
            ms = r.get("execution_time_ms")
            if isinstance(ms, (int, float)) and ms > 0 and dims:
                print(f"({{dims}}): {{ms:.4f}} ms")
        return results
    except Exception as exc:  # noqa: BLE001
        return [{{
            "test_case_id": "hip_shape_generated",
            "execution_time_ms": elapsed * 1000.0,
            "params": {{"parse_error": f"{{type(exc).__name__}}: {{exc}}", "stdout_tail": output[-2000:]}},
        }}]


def run_correctness():
    timeout = int(os.environ.get(
        "UNITTEST_HIP_CORRECTNESS_TIMEOUT_SEC",
        str(_DEFAULT_CORRECTNESS_TIMEOUT_SEC),
    ))
    if _supported_shape_cases():
        return _run_shape_correctness(timeout)
    if not BENCHMARK_COMMANDS:
        return False, "no benchmark command captured for HIP source", 0.0, ""
    try:
        with _OverlayLiveSource():
            rc, output, elapsed = _run(BENCHMARK_COMMANDS[0], timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        return False, f"timeout after {{timeout}}s", float(timeout), (exc.stdout or "") + (exc.stderr or "")
    except Exception as exc:  # noqa: BLE001
        return False, f"{{type(exc).__name__}}: {{exc}}", 0.0, ""
    return rc == 0, None if rc == 0 else f"benchmark rc={{rc}}", elapsed, output


def run_performance():
    timeout = int(os.environ.get(
        "UNITTEST_HIP_PERFORMANCE_TIMEOUT_SEC",
        str(_DEFAULT_PERFORMANCE_TIMEOUT_SEC),
    ))
    shape_results = _run_shape_performance(timeout)
    if shape_results is not None:
        return shape_results
    if not BENCHMARK_COMMANDS:
        return []
    try:
        with _OverlayLiveSource():
            rc, output, elapsed = _run(BENCHMARK_COMMANDS[0], timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        return [{{
            "test_case_id": "hip_benchmark",
            "execution_time_ms": -1.0,
            "params": {{"error": f"timeout after {{timeout}}s",
                       "stdout_tail": ((exc.stdout or "") + (exc.stderr or ""))[-2000:]}},
        }}]
    except Exception as exc:  # noqa: BLE001
        return [{{
            "test_case_id": "hip_benchmark",
            "execution_time_ms": -1.0,
            "params": {{"error": f"{{type(exc).__name__}}: {{exc}}"}},
        }}]
    return [{{
        "test_case_id": "hip_benchmark",
        "execution_time_ms": elapsed * 1000.0 if rc == 0 else -1.0,
        "params": {{"command": BENCHMARK_COMMANDS[0],
                   "returncode": rc,
                   "stdout_tail": output[-2000:]}},
    }}]


def main() -> int:
    parser = argparse.ArgumentParser(description=f"Task runner for {{TASK_NAME}}")
    parser.add_argument("mode", choices=["compile", "correctness", "performance"])
    args = parser.parse_args()
    build_dir = TASK_DIR / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "compile":
        ok, err = run_compile()
        (build_dir / "compile_report.json").write_text(
            json.dumps({{"status": "ok" if ok else "fail", "error": err}}, indent=2),
            encoding="utf-8",
        )
        print(f"Compilation: {{'PASS' if ok else 'FAIL'}}")
        if err:
            print(f"Error: {{err}}")
        return 0 if ok else 1

    if args.mode == "correctness":
        ok, err, elapsed, output = run_correctness()
        (build_dir / "correctness_report.json").write_text(
            json.dumps({{
                "status": "ok" if ok else "fail",
                "error": err,
                "elapsed_s": elapsed,
                "stdout_tail": output[-4000:],
            }}, indent=2),
            encoding="utf-8",
        )
        print(f"Correctness: {{'PASS' if ok else 'FAIL'}}")
        if err:
            print(f"Error: {{err}}")
        if output:
            print(output[-4000:])
        return 0 if ok else 1

    cases = run_performance()
    (build_dir / "performance_report.json").write_text(
        json.dumps(cases, indent=2), encoding="utf-8",
    )
    good = [c for c in cases if c.get("execution_time_ms", -1) > 0]
    if good:
        print(f"Performance: measured {{len(good)}} case(s), total time: "
              f"{{sum(c['execution_time_ms'] for c in good):.4f}} ms")
    else:
        print("Performance: FAILED - no test cases measured")
    # Non-zero exit when nothing was actually measured: callers (GEAK
    # select_agent, orchestrator) must be able to distinguish a successful
    # measurement from "every shape errored out / timed out".
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
'''


_CONFIG_TEMPLATE = """source_file_path:
  - source/{source_basename}

target_kernel_functions:
{target_kernel_lines}

compile_command:
  - python3 scripts/task_runner.py compile

correctness_command:
  - python3 scripts/task_runner.py correctness

performance_command:
  - python3 scripts/task_runner.py performance

task_type: {task_type}

task_result_template: null

prompt:
  source_code: null
  instructions: |
{instructions_block}
  cheatsheet: null
"""


def _shape_padded_dtypes(
    candidate: dict[str, Any], shapes: list[list[int]], dtypes_hint: list[str],
) -> tuple[list[str], list[str]]:
    """Return ``(dtype_exprs, warnings)`` padded to ``len(shapes)``."""
    warnings: list[str] = []
    raw_dtypes = candidate.get("input_dtypes") or []
    if isinstance(raw_dtypes, (list, tuple)):
        normalized = [_normalize_dtype(d) for d in raw_dtypes]
    else:
        normalized = []
    if len(normalized) < len(shapes):
        # Try to backfill from sniffed dtype tokens.
        for i in range(len(normalized), len(shapes)):
            if i < len(dtypes_hint) and dtypes_hint[i]:
                normalized.append(_normalize_dtype(dtypes_hint[i]))
            else:
                normalized.append("torch.float16")
                warnings.append(
                    f"input dtype missing for arg{i}; defaulting to float16"
                )
    elif len(normalized) > len(shapes):
        normalized = normalized[: len(shapes)]
        warnings.append("dropped trailing input dtypes beyond captured shape count")
    return normalized, warnings


def _build_instructions(candidate: dict[str, Any], host_entry: str | None) -> str:
    name = str(
        candidate.get("name")
        or candidate.get("kernel_name")
        or candidate.get("kernel_id")
        or "unknown_kernel"
    )
    pct = candidate.get("gpu_pct") or candidate.get("percent_of_total") or "unknown"
    bound = (
        candidate.get("bound_type")
        or (candidate.get("task_group", {}).get("rows", [{}])[0].get("bound_type")
            if isinstance(candidate.get("task_group"), dict) else None)
        or "unknown"
    )
    lines = [
        f"    Optimize the kernel `{name}` for maximum throughput while maintaining",
        "    numerical correctness against the captured baseline.",
        "",
        f"    Profile context: this kernel accounts for ~{pct} of GPU time and is",
        f"    classified as `{bound}` bound by TraceLens roofline analysis.",
        "",
        "    Constraints:",
        "    - Must keep the host entry function signature stable"
        + (f" (`{host_entry}`)" if host_entry else "")
        + ".",
        "    - Output must match the snapshotted baseline within `DEFAULT_ATOL` / `DEFAULT_RTOL`.",
        "    - You may freely retune block sizes, num_warps, num_stages, and the",
        "      `@triton.jit` body — the harness re-imports the file you edit.",
    ]
    return "\n".join(lines)


def _atol_rtol_for(dtype_exprs: list[str]) -> tuple[float, float]:
    """Pick tolerances based on the *least precise* tensor in the set.

    fp32 → 1e-4, bf16/fp16 → 1e-2, fp8 → 5e-2, int* → 0 (exact).
    """
    if not dtype_exprs:
        return 1e-2, 1e-2
    if any("float8" in d for d in dtype_exprs):
        return 5e-2, 5e-2
    if any("bfloat16" in d or "float16" in d for d in dtype_exprs):
        return 1e-2, 1e-2
    if all("float32" in d or "float64" in d for d in dtype_exprs):
        return 1e-4, 1e-4
    if all(_is_integer_dtype_expr(d) for d in dtype_exprs):
        return 0.0, 0.0
    return 1e-2, 1e-2


def _env_subset_for_runtime(candidate: dict[str, Any]) -> dict[str, str]:
    """Return the env-var subset we should reproduce inside the unittest.

    We pull from ``candidate["env_vars"]`` first (this is the snapshot the
    inference_optimizer captured from the live SGLang/vLLM serving process)
    and then fold in any current-process env that matches the framework
    prefixes Hyperloom's prompt builder also looks at. Keys we never echo:
    anything matching ``KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL``.
    """
    prefixes = ("SGLANG_", "VLLM_", "AITER_", "TRITON_", "HIP_", "ROCR_", "CUDA_")
    sensitive = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "AUTH")
    # Device-selection vars are per-run (caller / GEAK / ray pick the
    # physical GPU at invocation time). Baking them into the harness from
    # the captured profile trace is actively harmful: a candidate trace
    # captured on physical GPU0 leaves ``ROCR_VISIBLE_DEVICES='0'`` in
    # RUNTIME_ENV, and when GEAK later runs the harness on GPU2 the
    # ``setdefault`` injection in task_runner.py overrides the caller's
    # ``HIP_VISIBLE_DEVICES=2`` with the stale ``ROCR=0`` → torch reports
    # ``No HIP GPUs are available`` and every patch is rejected.
    device_select_vars = {
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "AMD_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
        "NVIDIA_VISIBLE_DEVICES",
    }
    base: dict[str, str] = {}
    raw = candidate.get("env_vars") or {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            if not isinstance(k, str):
                continue
            if k in device_select_vars:
                continue
            if any(s in k.upper() for s in sensitive):
                continue
            base[k] = str(v) if v is not None else ""
    # Fold in current-process matches (helpful when the optimizer launched
    # via baseline_config that exports SGLANG_* / AITER_* directly). We
    # also skip device-selection vars here so the harness never freezes a
    # device ordinal — the caller's shell / GEAK / ray must supply that
    # afresh on each invocation.
    for k, v in os.environ.items():
        if k in device_select_vars:
            continue
        if not k.startswith(prefixes):
            continue
        if any(s in k.upper() for s in sensitive):
            continue
        base.setdefault(k, v)
    return base


def _task_name(candidate: dict[str, Any]) -> str:
    raw = str(
        candidate.get("name")
        or candidate.get("kernel_name")
        or candidate.get("kernel_id")
        or "unknown"
    )
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", raw).strip("_")
    return safe or "unknown_kernel"


def _detect_task_type(source_path: Path) -> str:
    text = source_path.read_text(encoding="utf-8", errors="replace")[:8000]
    if "@triton.jit" in text or "import triton" in text:
        return "triton2triton"
    if source_path.suffix in (".cu", ".cuh", ".hip", ".hpp", ".h"):
        return "hip2hip"
    return "triton2triton"


def _compute_hip_timeout_budget(shape_cases: list[dict[str, Any]] | None) -> dict[str, int]:
    """Pick HIP correctness/performance timeouts at harness generation time.

    Observed worst-case on a freshly cleared aiter JIT cache (MI300X, ROCm 6.x):

      * ~60s per kernel module recompile (rmsnorm_quant, activation)
      * baseline overlay → save subprocess does 1 module rebuild
      * current overlay → compare subprocess does another rebuild
      * each shape adds ~5–10s of pure compute

    With the old 900s default that's already cutting it close at N=4 shapes
    on activation_kernels.cu (where CK Tile template instantiation can push
    a single rebuild past 90s). We scale the budget with the captured shape
    count and floor at 30 minutes (correctness) / 30 minutes (performance) /
    300s per-shape, then expose all three to the generated runner so it can
    further subdivide between save/compare phases.
    """
    n_shapes = max(1, len(shape_cases or []))
    per_shape = max(300, int(os.environ.get("UNITTEST_HIP_PER_SHAPE_TIMEOUT_SEC", "0")) or 300)
    base_floor = 1800
    # Reserve 2 × ~90s for the two overlay-induced JIT recompiles, plus
    # ``per_shape`` seconds per case for save/compare/perf, plus a 60s buffer.
    computed = 2 * 90 + per_shape * n_shapes + 60
    correctness = max(base_floor, computed)
    performance = max(base_floor, computed)
    return {"correctness": correctness, "performance": performance, "per_shape": per_shape}


def _benchmark_commands(candidate: dict[str, Any]) -> list[str]:
    commands: list[str] = []
    raw_files: list[Any] = []
    for key in ("benchmark_files", "benchmark_file", "test_harness_path"):
        value = candidate.get(key)
        if isinstance(value, list):
            raw_files.extend(value)
        elif value:
            raw_files.append(value)
    for item in raw_files:
        path = Path(str(item))
        if not path.exists():
            continue
        if path.suffix == ".py":
            commands.append("python " + shlex.quote(str(path)))
        else:
            commands.append(shlex.quote(str(path)))
    return commands


def _jit_tokens_for_source(source_path: Path) -> list[str]:
    stem = source_path.stem.lower()
    tokens = [t for t in re.split(r"[^a-z0-9]+", stem) if t and t != "kernels"]
    # Common aiter module naming drops the trailing `_kernels` and prefixes
    # `module_`, e.g. `rmsnorm_quant_kernels.cu` -> module_rmsnorm_quant.so.
    return tokens[:3]


def _jit_roots_for_source(source_path: Path) -> list[str]:
    roots: list[str] = []
    parts = source_path.parts
    if "/sgl-workspace/aiter" in str(source_path):
        roots.append("/sgl-workspace/aiter/aiter/jit")
    return roots


def _literal_tuple(value: Any) -> list[Any]:
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
    except Exception:
        return []
    if not isinstance(parsed, (list, tuple)):
        return []
    return list(parsed)


def _shape_cases(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    runtime_args = candidate.get("runtime_args")
    if not isinstance(runtime_args, dict):
        runtime_args = {}
    raw_entries = runtime_args.get("tracelens_args") or []
    if not isinstance(raw_entries, list):
        return []
    cases: list[dict[str, Any]] = []
    for entry in raw_entries:
        if not isinstance(entry, dict):
            continue
        dims = _literal_tuple(entry.get("input_dims"))
        if not dims:
            continue
        norm_dims: list[list[int]] = []
        for dim in dims:
            shape = _coerce_shape(dim)
            norm_dims.append(shape or [])
        cases.append({
            "op": str(entry.get("op") or ""),
            "input_dims": norm_dims,
            "input_types": list(_literal_tuple(entry.get("input_types"))),
        })
    if cases:
        return cases
    return _shape_cases_from_profile(candidate)


def _candidate_profile_op_hints(candidate: dict[str, Any]) -> list[str]:
    hints: list[str] = []
    runtime_args = candidate.get("runtime_args")
    if isinstance(runtime_args, dict):
        for entry in runtime_args.get("tracelens_args") or []:
            if isinstance(entry, dict) and entry.get("op"):
                hints.append(str(entry["op"]))
    for key in ("name", "kernel_name", "source_file", "source_path"):
        value = candidate.get(key)
        if value:
            hints.append(str(value))
    text = " ".join(hints).lower()
    semantic: list[str] = []
    if "rmsnorm" in text or "rms_norm" in text:
        semantic.extend(["aiter::rmsnorm", "rmsnorm"])
    if "silu" in text or "act_and_mul" in text or "activation" in text:
        semantic.extend(["sgl_kernel::silu_and_mul", "silu_and_mul"])
    if "fmha" in text or "prefill" in text or "paged" in text:
        semantic.extend(["aiter::mha_batch_prefill", "mha_batch_prefill"])
    out: list[str] = []
    for item in semantic + hints:
        item = str(item).strip()
        if item and item not in out:
            out.append(item)
    return out


def _profile_trace_paths(candidate: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for key in ("trace_input", "trace_file", "last_profile_trace", "profile_trace"):
        value = candidate.get(key)
        if value:
            p = Path(str(value))
            if p.is_file():
                paths.append(p)
            elif p.is_dir():
                paths.extend(sorted(p.rglob("*.trace.json.gz")))
    env_trace = os.environ.get("HYPERLOOM_PROFILE_TRACE") or os.environ.get("LAST_PROFILE_TRACE")
    if env_trace:
        p = Path(env_trace)
        if p.is_file():
            paths.append(p)
    session_dir = Path(os.environ.get("USER_DATA_PATH", ""))
    if session_dir.is_dir():
        paths.extend(sorted((session_dir / "runs" / "profile").glob("**/*.trace.json.gz")))
        paths.extend(sorted((session_dir / "runs" / "profile").glob("**/*trace*.json.gz")))
    unique: list[Path] = []
    seen: set[str] = set()
    for p in paths:
        try:
            rp = str(p.resolve())
        except OSError:
            rp = str(p)
        if rp not in seen and Path(rp).is_file():
            seen.add(rp)
            unique.append(Path(rp))
    # Prefer merged/DECODE traces and newest files; cap to keep generation fast.
    def score(p: Path) -> tuple[int, float]:
        name = p.name.lower()
        rank = 0
        if "merged" in name:
            rank += 20
        if "decode" in name:
            rank += 10
        if "extend" in name:
            rank += 5
        try:
            mt = p.stat().st_mtime
        except OSError:
            mt = 0.0
        return (rank, mt)
    return sorted(unique, key=score, reverse=True)[:8]


def _dims_from_trace_args(args: dict[str, Any]) -> list[list[int]]:
    raw = args.get("Input Dims") or args.get("input_dims")
    if not isinstance(raw, list):
        return []
    dims: list[list[int]] = []
    for entry in raw:
        shape = _coerce_shape(entry)
        dims.append(shape or [])
    return dims


def _types_from_trace_args(args: dict[str, Any]) -> list[str]:
    raw = args.get("Input type") or args.get("Input Types") or args.get("input_types")
    if isinstance(raw, list):
        return [str(x) for x in raw]
    return []


def _shape_cases_from_profile(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    hints = [h.lower() for h in _candidate_profile_op_hints(candidate)]
    if not hints:
        return []
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for trace in _profile_trace_paths(candidate):
        try:
            with gzip.open(trace, "rt", errors="replace") as fh:
                payload = json.load(fh)
        except Exception:
            continue
        for event in payload.get("traceEvents") or []:
            if not isinstance(event, dict):
                continue
            name = str(event.get("name") or "")
            lname = name.lower()
            args = event.get("args")
            if not isinstance(args, dict):
                continue
            if not any(h and (h in lname or lname in h) for h in hints):
                continue
            dims = _dims_from_trace_args(args)
            if not any(dims):
                continue
            types = _types_from_trace_args(args)
            key = json.dumps([name, dims, types], sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            cases.append({"op": name, "input_dims": dims, "input_types": types, "source": str(trace)})
            if len(cases) >= 8:
                return _filter_supported_shape_cases(cases, candidate)
    return _filter_supported_shape_cases(cases, candidate)


def _filter_supported_shape_cases(
    cases: list[dict[str, Any]], candidate: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    head_size = None
    if isinstance(candidate, dict):
        params = candidate.get("kernel_params")
        if isinstance(params, dict):
            try:
                head_size = int(params.get("HEAD_SIZE") or 0) or None
            except (TypeError, ValueError):
                head_size = None
    for case in cases:
        op = str(case.get("op", ""))
        dims = case.get("input_dims") or []
        if not dims:
            continue
        if "aiter::rmsnorm" in op and len(dims) >= 3:
            out.append(case)
        elif "silu_and_mul" in op and len(dims) >= 2:
            out.append(case)
    if head_size:
        preferred = []
        for case in out:
            dims = case.get("input_dims") or []
            if any(isinstance(dim, list) and dim and dim[-1] == head_size for dim in dims):
                preferred.append(case)
        if preferred:
            return preferred
    return out


def _generate_hip_unittest(
    candidate: dict[str, Any],
    *,
    source_path: Path,
    out_dir: Path,
    target_platform: str = "",
    log: Callable[[str], None] | None = None,
    self_verify: bool = True,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    warnings = warnings if warnings is not None else []
    name = str(candidate.get("kernel_name") or candidate.get("name") or candidate.get("kernel_id") or source_path.stem)
    source_basename = source_path.name
    src_dir = out_dir / "source"
    snapshot_dir = src_dir / "_baseline_snapshot"
    (out_dir / "scripts").mkdir(parents=True, exist_ok=True)
    src_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    linked_source = src_dir / source_basename
    shutil.copy2(source_path, linked_source)
    snapshot_target = snapshot_dir / source_basename
    shutil.copy2(source_path, snapshot_target)

    commands = _benchmark_commands(candidate)
    if not commands:
        warnings.append("no existing benchmark_file/test_harness_path found; HIP correctness cannot run")
    env_vars = _env_subset_for_runtime(candidate)
    kernel_repo = str(candidate.get("kernel_repo") or source_path.parent)
    shape_cases = _shape_cases(candidate)
    # Compute per-harness timeouts at generation time so we never ship a stale
    # 900s/1200s default with 4-shape rmsnorm_quant / silu_and_mul cases that
    # routinely need 2× JIT recompile (~60s/module) + N shape benchmarks. Both
    # values are still overridable at runtime via UNITTEST_HIP_*_TIMEOUT_SEC.
    timeout_budget = _compute_hip_timeout_budget(shape_cases)
    task_runner_text = _HIP_TASK_RUNNER_TEMPLATE.format(
        kernel_name=name,
        task_name=f"unittest_agent/{_task_name(candidate)}",
        source_basename=source_basename,
        live_source=str(source_path),
        kernel_repo=kernel_repo,
        benchmark_commands=commands,
        target_kernels=[name],
        env_vars=env_vars,
        jit_roots=_jit_roots_for_source(source_path),
        jit_match_tokens=_jit_tokens_for_source(source_path),
        shape_cases=shape_cases,
        default_correctness_timeout=timeout_budget["correctness"],
        default_performance_timeout=timeout_budget["performance"],
        default_per_shape_timeout=timeout_budget["per_shape"],
    )
    task_runner = out_dir / "scripts" / "task_runner.py"
    task_runner.write_text(task_runner_text, encoding="utf-8")
    task_runner.chmod(0o755)

    target_lines = "\n".join(f"  - {s}" for s in [name])
    instructions = "\n".join([
        f"    Optimize the HIP/C++ kernel `{name}` for the captured serving workload.",
        "    The generated runner temporarily overlays `source/<kernel>` onto",
        "    the live framework source, invalidates likely JIT modules, runs the",
        "    TraceLens-discovered benchmark command, and restores the live tree.",
        "    Preserve public signatures, includes, namespaces, and registration macros.",
    ])
    config_yaml = out_dir / "config.yaml"
    config_yaml.write_text(_CONFIG_TEMPLATE.format(
        source_basename=source_basename,
        target_kernel_lines=target_lines,
        task_type=_detect_task_type(source_path),
        instructions_block=instructions,
    ), encoding="utf-8")

    self_verify_res: dict[str, Any] = {
        "compile": "skipped",
        "correctness": "skipped",
        "correctness_reason": "self_verify disabled",
    }
    if self_verify:
        # Run the full HIP/C++ correctness gate before GEAK sees the harness.
        # This can trigger two overlay-induced JIT rebuilds plus every captured
        # shape benchmark, but it prevents GEAK from spending hours on a
        # generated harness that cannot pass its own baseline correctness test.
        rc, out, err = 1, "", ""
        try:
            proc = subprocess.run(
                [sys.executable, str(task_runner), "compile"],
                capture_output=True, text=True, timeout=120, cwd=str(out_dir),
            )
            rc, out, err = proc.returncode, proc.stdout, proc.stderr
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
        self_verify_res = {
            "compile": "ok" if rc == 0 else "fail",
            "compile_rc": rc,
            "compile_tail": (out or err)[-800:],
            "correctness": "skipped",
            "correctness_reason": "compile failed; correctness not attempted",
        }
        if rc == 0:
            correctness_timeout = timeout_budget["correctness"] + 60
            srcc, sout, serr = 1, "", ""
            try:
                proc = subprocess.run(
                    [sys.executable, str(task_runner), "correctness"],
                    capture_output=True, text=True, timeout=correctness_timeout,
                    cwd=str(out_dir),
                )
                srcc, sout, serr = proc.returncode, proc.stdout, proc.stderr
            except subprocess.TimeoutExpired as exc:
                srcc = 124
                sout = exc.stdout or ""
                serr = exc.stderr or f"correctness timed out after {correctness_timeout}s"
            except Exception as exc:  # noqa: BLE001
                serr = f"{type(exc).__name__}: {exc}"
            self_verify_res["correctness"] = "ok" if srcc == 0 else "fail"
            self_verify_res["correctness_rc"] = srcc
            self_verify_res["correctness_tail"] = (sout or serr)[-1500:]
            self_verify_res["correctness_reason"] = (
                "Hyperloom pre-GEAK full correctness validation"
            )
        if log:
            log(f"[self_verify] HIP compile rc={rc} tail={self_verify_res['compile_tail'][:200]!r}")
            if "correctness_rc" in self_verify_res:
                log(f"[self_verify] HIP correctness rc={self_verify_res['correctness_rc']}")

    # The harness is "ok" iff Hyperloom has already proven the generated
    # correctness command passes on the unmodified baseline. Anything else
    # stays degraded and falls back to legacy benchmark_files/test_harness_path
    # instead of being promoted to GEAK's --test-command.
    compile_ok = self_verify_res.get("compile") == "ok"
    corr_ok = self_verify_res.get("correctness") == "ok"
    status = "ok" if (commands and compile_ok and corr_ok) else "degraded"
    test_command = f"python3 {task_runner}"
    captured_shapes = _collect_input_shapes(candidate)[0]
    if not captured_shapes and shape_cases:
        seen_shapes: list[list[int]] = []
        for case in shape_cases:
            for shape in case.get("input_dims") or []:
                if shape and shape not in seen_shapes:
                    seen_shapes.append(shape)
        captured_shapes = seen_shapes
    manifest: dict[str, Any] = {
        "status": status,
        "out_dir": str(out_dir),
        "warnings": warnings,
        "candidate_kernel_id": candidate.get("kernel_id") or candidate.get("name"),
        "config_yaml": str(config_yaml),
        "task_runner": str(task_runner),
        "test_command": f"{test_command} correctness",
        "performance_command": f"{test_command} performance",
        "source_file": str(linked_source),
        "live_source_file": str(source_path),
        "baseline_snapshot": str(snapshot_target),
        "kernel_name": name,
        "host_entry": "",
        "target_kernels": [name],
        "task_type": _detect_task_type(source_path),
        "num_shapes": len(captured_shapes),
        "shapes": captured_shapes,
        "shape_cases": shape_cases,
        "dtypes": candidate.get("input_dtypes") or [],
        "env_vars_count": len(env_vars),
        "benchmark_commands": commands,
        "jit_match_tokens": _jit_tokens_for_source(source_path),
        "self_verify": self_verify_res,
        # Pre-computed harness budget so GEAK config / orchestrator can pad
        # their own subprocess timeouts (``ctx.timeout``) above it. Without
        # this they'd guess and risk killing the inner shape runner before
        # baseline JIT recompile even finishes.
        "harness_timeout_correctness_sec": timeout_budget["correctness"],
        "harness_timeout_performance_sec": timeout_budget["performance"],
        "harness_per_shape_timeout_sec": timeout_budget["per_shape"],
    }
    meta_path = out_dir / "unittest_meta.json"
    meta_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    manifest["unittest_meta_path"] = str(meta_path)
    if log:
        log(f"[unittest_agent] generated {status} HIP unittest at {out_dir} "
            f"(benchmarks={len(commands)}, "
            f"self_verify={self_verify_res.get('compile')}/{self_verify_res.get('correctness')})")
    return manifest


def _self_verify(
    out_dir: Path, task_runner: Path, log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run compile + correctness against the *unmodified* source. Both MUST
    pass for the harness to be flagged ``ok`` — otherwise we mark it
    ``degraded`` so the caller can decide whether to still hand it to GEAK."""
    res: dict[str, Any] = {"compile": "skipped", "correctness": "skipped"}

    def _run(mode: str, timeout: int) -> tuple[int, str, str]:
        cmd = [sys.executable, str(task_runner), mode]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout,
                cwd=str(out_dir),
            )
            return proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as exc:
            return 124, exc.stdout or "", exc.stderr or "timeout"
        except Exception as exc:  # noqa: BLE001
            return 1, "", f"{type(exc).__name__}: {exc}"

    rc, out, err = _run("compile", timeout=120)
    res["compile"] = "ok" if rc == 0 else "fail"
    res["compile_rc"] = rc
    res["compile_tail"] = (out or err)[-800:]
    if log:
        log(f"[self_verify] compile rc={rc} tail={res['compile_tail'][:200]!r}")
    if rc != 0:
        # Skip correctness — without a successful import there's nothing to compare.
        res["correctness"] = "skipped"
        return res

    rc2, out2, err2 = _run("correctness", timeout=300)
    res["correctness"] = "ok" if rc2 == 0 else "fail"
    res["correctness_rc"] = rc2
    res["correctness_tail"] = (out2 or err2)[-1200:]
    if log:
        log(f"[self_verify] correctness rc={rc2} tail={res['correctness_tail'][:200]!r}")
    return res


def generate_unittest(
    candidate: dict[str, Any],
    *,
    out_dir: Path | str,
    target_platform: str = "",
    log: Callable[[str], None] | None = None,
    self_verify: bool = True,
) -> dict[str, Any]:
    """Generate an AgentKernelArena-style unittest harness for ``candidate``.

    Returns the manifest dict described in the module docstring. Never
    raises on a normal failure path — instead returns ``status="failed"``
    + ``error`` so the caller can decide whether to fall through and let
    GEAK run without the harness.
    """
    out_dir = Path(out_dir).resolve()
    warnings: list[str] = []
    manifest: dict[str, Any] = {
        "status": "failed",
        "out_dir": str(out_dir),
        "warnings": warnings,
        "candidate_kernel_id": candidate.get("kernel_id") or candidate.get("name"),
    }

    source_file = str(
        candidate.get("source_file")
        or candidate.get("kernel_url")
        or candidate.get("kernel_path")
        or ""
    ).strip()
    if not source_file:
        manifest["error"] = "candidate has no source_file"
        return manifest
    source_path = Path(source_file)
    if not source_path.is_file():
        manifest["error"] = f"source_file does not exist: {source_path}"
        return manifest
    if source_path.suffix in (".cu", ".cuh", ".hip", ".hpp", ".h"):
        return _generate_hip_unittest(
            candidate,
            source_path=source_path,
            out_dir=out_dir,
            target_platform=target_platform,
            log=log,
            self_verify=self_verify,
            warnings=warnings,
        )
    if source_path.suffix not in (".py",):
        manifest["status"] = "skipped"
        manifest["error"] = (
            f"unittest_agent only generates Python/Triton or HIP/C++ harnesses; "
            f"got suffix {source_path.suffix!r}"
        )
        return manifest

    # 1. Pick kernel name, host entry, target symbols. We need to know how
    # many tensor arg shapes we captured before picking the entry — that's
    # what lets us prefer a launcher with 3 args (2 tensors + epsilon) over
    # a 1-arg helper like ``num_programs``.
    pre_shapes, _ = _collect_input_shapes(candidate)
    host_entry, host_list, host_arg_count = _pick_host_entry(
        candidate, source_path, num_shape_args=len(pre_shapes),
    )
    if host_entry is None:
        manifest["error"] = (
            f"no host entry point found in {source_path} "
            "(no top-level def without @triton.jit)"
        )
        return manifest
    target_symbols = []
    name = str(
        candidate.get("kernel_name")
        or candidate.get("name")
        or candidate.get("kernel_id")
        or ""
    ).strip()
    if name and name not in target_symbols:
        target_symbols.append(name)
    if host_entry not in target_symbols:
        target_symbols.append(host_entry)
    if not target_symbols:
        target_symbols = host_list[:1]

    # 2. Capture shapes + dtypes.
    shapes, dtypes_hint = _collect_input_shapes(candidate)
    if not shapes:
        warnings.append(
            "no input shapes captured from candidate; correctness mode will "
            "report fail and GEAK will only see the source + env context"
        )
    dtype_exprs, dtype_warnings = _shape_padded_dtypes(candidate, shapes, dtypes_hint)
    warnings.extend(dtype_warnings)

    # 3. Lay out workspace.
    (out_dir / "scripts").mkdir(parents=True, exist_ok=True)
    src_dir = out_dir / "source"
    src_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = src_dir / "_baseline_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # Symlink the live kernel file into source/ so GEAK can rewrite it; the
    # snapshot under source/_baseline_snapshot/ stays read-only (the golden
    # bytes for correctness). For sandboxes that don't support symlinks we
    # fall back to a hard copy.
    source_basename = source_path.name
    linked_source = src_dir / source_basename
    if linked_source.exists() or linked_source.is_symlink():
        linked_source.unlink()
    try:
        linked_source.symlink_to(source_path.resolve())
    except OSError:
        shutil.copy2(source_path, linked_source)
        warnings.append("symlink unsupported; copied kernel into source/ (changes won't propagate back to /sgl-workspace)")
    snapshot_target = snapshot_dir / source_basename
    shutil.copy2(source_path, snapshot_target)

    # 4. Render task_runner.py. If the picked host entry takes more args
    # than we have captured shapes (e.g. ``rms_norm(input, weight, epsilon)``
    # → 3 args, we captured 2 tensor shapes), fill the trailing slots with
    # sensible scalar defaults so the launcher signature matches.
    extra_scalar_args = max(0, host_arg_count - len(shapes))
    if extra_scalar_args:
        warnings.append(
            f"host entry {host_entry!r} takes {host_arg_count} args; only "
            f"{len(shapes)} tensor shapes captured. Auto-filling "
            f"{extra_scalar_args} trailing scalar arg(s) with default values "
            f"(1e-6, 1, False, None). Verify these match the kernel's "
            f"non-tensor inputs (epsilon, num_heads, etc.)."
        )
    test_shapes_block, args_init_block = _render_test_cases(
        shapes, dtype_exprs, extra_scalar_args=extra_scalar_args,
    )
    atol, rtol = _atol_rtol_for(dtype_exprs)
    env_vars = _env_subset_for_runtime(candidate)
    test_dtypes_repr = ", ".join(repr(d) for d in dtype_exprs)
    task_runner_text = _TASK_RUNNER_TEMPLATE.format(
        kernel_name=name or host_entry,
        task_name=f"unittest_agent/{_task_name(candidate)}",
        source_basename=source_basename,
        host_entry=host_entry,
        target_kernels=target_symbols,
        env_vars=env_vars,
        test_shapes_block=test_shapes_block,
        test_dtypes_repr=test_dtypes_repr,
        warmup_iters=5,
        bench_iters=25,
        default_atol=atol,
        default_rtol=rtol,
        materialize_args_block=args_init_block,
    )
    task_runner = out_dir / "scripts" / "task_runner.py"
    task_runner.write_text(task_runner_text, encoding="utf-8")
    task_runner.chmod(0o755)

    # 5. Render config.yaml.
    task_type = _detect_task_type(source_path)
    instructions = _build_instructions(candidate, host_entry)
    target_lines = "\n".join(f"  - {s}" for s in target_symbols)
    config_text = _CONFIG_TEMPLATE.format(
        source_basename=source_basename,
        target_kernel_lines=target_lines,
        task_type=task_type,
        instructions_block=instructions,
    )
    config_yaml = out_dir / "config.yaml"
    config_yaml.write_text(config_text, encoding="utf-8")

    # 6. Self-verify (best-effort; failure → degraded, not fatal).
    self_verify_res: dict[str, Any] = {"compile": "skipped", "correctness": "skipped"}
    if self_verify:
        if log:
            log(f"[unittest_agent] self-verifying harness at {out_dir}")
        self_verify_res = _self_verify(out_dir, task_runner, log=log)

    if self_verify_res.get("compile") == "ok" and self_verify_res.get("correctness") in ("ok", "skipped"):
        status = "ok"
    elif self_verify_res.get("compile") == "ok":
        status = "degraded"
    else:
        status = "degraded"

    test_command = f"python3 {task_runner}"
    manifest.update({
        "status": status,
        "config_yaml": str(config_yaml),
        "task_runner": str(task_runner),
        "test_command": f"{test_command} correctness",
        "performance_command": f"{test_command} performance",
        "source_file": str(linked_source),
        "baseline_snapshot": str(snapshot_target),
        "kernel_name": name or host_entry,
        "host_entry": host_entry,
        "target_kernels": target_symbols,
        "task_type": task_type,
        "num_shapes": len(shapes),
        "shapes": shapes,
        "dtypes": dtype_exprs,
        "env_vars_count": len(env_vars),
        "self_verify": self_verify_res,
    })
    meta_path = out_dir / "unittest_meta.json"
    meta_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str),
                         encoding="utf-8")
    manifest["unittest_meta_path"] = str(meta_path)
    if log:
        log(f"[unittest_agent] generated {status} unittest at {out_dir} "
            f"(shapes={len(shapes)}, env_vars={len(env_vars)}, "
            f"self_verify={self_verify_res.get('compile')}/{self_verify_res.get('correctness')})")
    return manifest


# ---------------------------------------------------------------------------
# CLI for ad-hoc debugging (`python -m unittest_agent ...`)
# ---------------------------------------------------------------------------

def _main(argv: list[str] | None = None) -> int:
    import argparse as _argparse
    parser = _argparse.ArgumentParser(
        description="Generate an AgentKernelArena unittest from a candidate JSON",
    )
    parser.add_argument("--candidate-json", required=True,
                        help="Path to a single-candidate dict OR a list "
                        "(use --kernel-id to pick from a list).")
    parser.add_argument("--kernel-id", default="",
                        help="When candidate-json is a list, pick this kernel.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--target-platform", default="")
    parser.add_argument("--skip-self-verify", action="store_true")
    args = parser.parse_args(argv)

    data = json.loads(Path(args.candidate_json).read_text(encoding="utf-8"))
    if isinstance(data, list):
        if not args.kernel_id:
            raise SystemExit("--kernel-id is required when --candidate-json is a list")
        cand = None
        for item in data:
            if isinstance(item, dict) and (
                item.get("kernel_id") == args.kernel_id
                or item.get("name") == args.kernel_id
            ):
                cand = item
                break
        if cand is None:
            raise SystemExit(f"kernel_id {args.kernel_id!r} not found in candidate list")
        candidate = cand
    elif isinstance(data, dict) and "hot_kernels" in data:
        if not args.kernel_id:
            raise SystemExit("--kernel-id is required when --candidate-json holds hot_kernels")
        cand = None
        for item in data["hot_kernels"]:
            if isinstance(item, dict) and (
                item.get("kernel_id") == args.kernel_id
                or item.get("name") == args.kernel_id
            ):
                cand = item
                break
        if cand is None:
            raise SystemExit(f"kernel_id {args.kernel_id!r} not found in hot_kernels")
        candidate = cand
    else:
        candidate = data

    manifest = generate_unittest(
        candidate,
        out_dir=args.out_dir,
        target_platform=args.target_platform,
        log=lambda msg: print(msg, file=sys.stderr),
        self_verify=not args.skip_self_verify,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    return 0 if manifest.get("status") in ("ok", "degraded") else 1


if __name__ == "__main__":
    raise SystemExit(_main())
