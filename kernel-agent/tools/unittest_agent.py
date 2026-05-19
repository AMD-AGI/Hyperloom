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

The generator is intentionally framework-agnostic — it inspects the kernel
module's public callables to find a plausible host entry point (the function
that prepares Triton grid + calls ``@triton.jit``). For Triton kernels
this is the most common shape downstream (vLLM/SGLang nearly always wrap
their ``@triton.jit`` body in a small Python launcher), and HIP/.cu source
files are deliberately skipped here (HIP unit-testing requires a separate
``hipcc`` build flow that the SKILL already routes through GEAK directly).
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
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

    for test_idx in range(len(TEST_SHAPES)):
        try:
            ref_args = _materialize_args(test_idx)
            cur_args = tuple(a.clone() if hasattr(a, "clone") else a for a in ref_args)

            ref_out = ref_fn(*ref_args)
            torch.cuda.synchronize()
            cur_out = cur_fn(*cur_args)
            torch.cuda.synchronize()

            ref_tensors = _flatten_tensors(ref_out)
            cur_tensors = _flatten_tensors(cur_out)
            # Some kernels mutate inputs in-place and return None; in that
            # case compare the input args back-to-back.
            if not ref_tensors and not cur_tensors:
                ref_tensors = _flatten_tensors(list(ref_args))
                cur_tensors = _flatten_tensors(list(cur_args))
            if len(ref_tensors) != len(cur_tensors):
                return False, (
                    f"shape {{test_idx + 1}}: tensor count mismatch "
                    f"(ref={{len(ref_tensors)}}, current={{len(cur_tensors)}})"
                )
            for ti, (r, c) in enumerate(zip(ref_tensors, cur_tensors)):
                if r.shape != c.shape:
                    return False, (
                        f"shape {{test_idx + 1}} output {{ti}}: shape mismatch "
                        f"(ref={{tuple(r.shape)}}, current={{tuple(c.shape)}})"
                    )
                if not torch.allclose(
                    r.to(torch.float32).cpu(),
                    c.to(torch.float32).cpu(),
                    atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL,
                ):
                    diff = (r.to(torch.float32).cpu()
                            - c.to(torch.float32).cpu()).abs().max().item()
                    return False, (
                        f"shape {{test_idx + 1}} output {{ti}}: "
                        f"max abs diff={{diff:.6g}} > atol={{DEFAULT_ATOL}}"
                    )
        except Exception as exc:  # noqa: BLE001
            return False, (
                f"shape {{test_idx + 1}} raised: {{type(exc).__name__}}: {{exc}}"
            )
    return True, None


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
        ok, err = run_correctness()
        (build_dir / "correctness_report.json").write_text(
            json.dumps({{"status": "ok" if ok else "fail",
                         "error": err,
                         "num_shapes": len(TEST_SHAPES)}}, indent=2),
            encoding="utf-8",
        )
        print(f"Correctness: {{'PASS' if ok else 'FAIL'}}")
        if err:
            print(f"Error: {{err}}")
        return 0 if ok else 1

    if args.mode == "performance":
        cases = run_performance()
        (build_dir / "performance_report.json").write_text(
            json.dumps(cases, indent=2), encoding="utf-8",
        )
        good = [c for c in cases if c.get("execution_time_ms", -1) > 0]
        if good:
            total = sum(c["execution_time_ms"] for c in good)
            print(f"Performance: measured {{len(good)}}/{{len(cases)}} cases, "
                  f"total time: {{total:.4f}} ms")
        else:
            print("Performance: FAILED - no test cases measured")
        return 0

    return 2


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
    base: dict[str, str] = {}
    raw = candidate.get("env_vars") or {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            if not isinstance(k, str):
                continue
            if any(s in k.upper() for s in sensitive):
                continue
            base[k] = str(v) if v is not None else ""
    # Fold in current-process matches (helpful when the optimizer launched
    # via baseline_config that exports SGLANG_* / AITER_* directly).
    for k, v in os.environ.items():
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
    if source_path.suffix in (".cu", ".cuh", ".hip"):
        return "hip2hip"
    return "triton2triton"


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
    if source_path.suffix not in (".py",):
        # HIP / .cu / .cuh kernels require a separate hipcc-based harness;
        # punt and let GEAK use its existing kernel-builder path.
        manifest["status"] = "skipped"
        manifest["error"] = (
            f"unittest_agent only generates Python/Triton harnesses; "
            f"got suffix {source_path.suffix!r} — skipping (GEAK falls back "
            "to its own benchmark generator)"
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
