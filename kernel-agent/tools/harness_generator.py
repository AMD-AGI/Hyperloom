#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Generate GEAK-compatible test harnesses from existing benchmark files.

Kernel-agnostic: AST-analyzes a benchmark .py to classify reference vs kernel
callables and tensor-creation patterns, then wraps them in the GEAK 4-mode template.
"""

from __future__ import annotations

import ast
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable


# Data classes

@dataclass
class FuncInfo:
    """Metadata about a function discovered in the benchmark source.

    Attributes:
        name (str): The function's name.
        params (list[str]): Positional parameter names.
        source (str): The original source text (including decorators).
        decorator (str): The recognised decorator name (``perftest`` /
            ``benchmark`` / empty for orchestrators).
        lineno (int): 1-based line number of the ``def`` statement.
    """
    name: str
    params: list[str]
    source: str
    decorator: str
    lineno: int


@dataclass
class TensorInfo:
    """A tensor-creating assignment extracted from a function body.

    Attributes:
        var_name (str): Name of the assigned variable.
        creation_expr (str): The unparsed creation expression (e.g.
            ``torch.randn(M, N, dtype=...)``).
        shape_args (list[str]): Unparsed positional shape args plus any
            non-device keyword args.
        dtype_expr (str | None): The unparsed ``dtype=`` expression, or
            ``None`` when not specified.
    """
    var_name: str
    creation_expr: str
    shape_args: list[str]
    dtype_expr: str | None


@dataclass
class CallInfo:
    """A captured call site and its arguments.

    Attributes:
        func_name (str): The callee name being matched.
        args (list[str]): Unparsed positional argument expressions.
        kwargs (dict[str, str]): Mapping of keyword name to unparsed
            value expression.
    """
    func_name: str
    args: list[str]
    kwargs: dict[str, str]


# BenchmarkAnalyzer — generic AST-based benchmark file analyzer

class BenchmarkAnalyzer:
    """Analyze a benchmark Python file to extract structure for harness generation."""

    PERF_DECORATORS = {"perftest", "benchmark"}
    REF_HINTS = {"torch", "ref", "native", "baseline", "reference", "gold"}
    KERNEL_HINTS = {"ck", "hip", "triton", "kernel_agent", "optimized", "custom", "fused"}

    def __init__(self, source: str, source_file_module: str = ""):
        """Parse the benchmark source into an AST for later queries.

        Args:
            source (str): Full Python source of the benchmark file.
            source_file_module (str): Dotted module path of the kernel's
                source file, used to recognise kernel calls by package
                prefix. Optional.
        """
        self.source = source
        self.lines = source.splitlines()
        self.tree = ast.parse(source)
        self.source_module = source_file_module

    def get_imports(self) -> list[str]:
        """Return original source lines for all import statements.

        Returns:
            list[str]: One entry per ``import`` / ``from ... import``
                statement, preserving the original source text.
        """
        import_lines: list[str] = []
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                start = node.lineno - 1
                end = node.end_lineno or node.lineno
                block = "\n".join(self.lines[start:end])
                # ast.walk also yields imports nested inside functions / classes /
                # try blocks, which keep their source indentation. Emitted verbatim
                # at the harness module top level they raise "unexpected indent" and
                # break static_check (the whole harness is then unusable). Dedent
                # each block so a function-local ``    import math`` becomes a valid
                # top-level ``import math``; module-level imports (no indent) are
                # unchanged.
                import_lines.append(textwrap.dedent(block))
        return import_lines

    def get_decorated_functions(self) -> dict[str, FuncInfo]:
        """Find all functions decorated with @perftest or @benchmark.

        Returns:
            dict[str, FuncInfo]: Mapping of function name to its
                :class:`FuncInfo` for every function carrying a
                recognised performance decorator.
        """
        result: dict[str, FuncInfo] = {}
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            for dec in node.decorator_list:
                dec_name = self._decorator_name(dec)
                if dec_name in self.PERF_DECORATORS:
                    params = [a.arg for a in node.args.args]
                    start = node.lineno - 1
                    dec_start = node.decorator_list[0].lineno - 1
                    end = node.end_lineno or (start + 1)
                    src = "\n".join(self.lines[dec_start:end])
                    result[node.name] = FuncInfo(
                        name=node.name,
                        params=params,
                        source=src,
                        decorator=dec_name,
                        lineno=node.lineno,
                    )
                    break
        return result

    def _decorator_name(self, dec: ast.expr) -> str:
        """Resolve the simple name of a decorator AST node.

        Args:
            dec (ast.expr): A decorator expression (Name, Call, or
                Attribute).

        Returns:
            str: The decorator's leaf name (e.g. ``perftest``), or an
                empty string when it cannot be resolved.
        """
        if isinstance(dec, ast.Name):
            return dec.id
        if isinstance(dec, ast.Call):
            if isinstance(dec.func, ast.Name):
                return dec.func.id
            if isinstance(dec.func, ast.Attribute):
                return dec.func.attr
        if isinstance(dec, ast.Attribute):
            return dec.attr
        return ""

    def classify_functions(
        self, decorated: dict[str, FuncInfo]
    ) -> tuple[FuncInfo | None, FuncInfo | None]:
        """Classify decorated functions into (reference, kernel); either can be None."""
        ref_candidates: list[FuncInfo] = []
        kernel_candidates: list[FuncInfo] = []

        for fi in decorated.values():
            if fi.decorator == "benchmark":
                continue
            name_lower = fi.name.lower()

            is_ref = any(h in name_lower for h in self.REF_HINTS)
            is_kernel = any(h in name_lower for h in self.KERNEL_HINTS)

            has_torch_functional = (
                "torch.nn.functional" in fi.source
                or "F.rms_norm" in fi.source
                or "F.layer_norm" in fi.source
                or "F.linear" in fi.source
                or "F.softmax" in fi.source
                or "F.scaled_dot_product_attention" in fi.source
            )

            has_source_module_call = False
            if self.source_module:
                top_pkg = self.source_module.split(".")[0]
                has_source_module_call = top_pkg in fi.source

            if is_ref or (has_torch_functional and not is_kernel):
                ref_candidates.append(fi)
            elif is_kernel or has_source_module_call:
                kernel_candidates.append(fi)
            else:
                kernel_candidates.append(fi)

        ref = ref_candidates[0] if ref_candidates else None
        kernel = kernel_candidates[0] if kernel_candidates else None

        # Fallback when classification found nothing.
        if ref is None and kernel is None and len(decorated) >= 1:
            perftest_funcs = [
                fi for fi in decorated.values() if fi.decorator == "perftest"
            ]
            if len(perftest_funcs) == 1:
                kernel = perftest_funcs[0]
            elif len(perftest_funcs) >= 2:
                ref = perftest_funcs[0]
                kernel = perftest_funcs[1]

        return ref, kernel

    def get_test_function(self, decorated: dict[str, FuncInfo]) -> FuncInfo | None:
        """Find the main test/benchmark orchestrator function."""
        # Prefer a @benchmark-decorated function; else a top-level test_*/bench_* caller.
        for fi in decorated.values():
            if fi.decorator == "benchmark":
                return fi

        dec_names = set(decorated.keys())
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name in decorated:
                continue
            if "test" in node.name.lower() or "bench" in node.name.lower():
                body_src = "\n".join(
                    self.lines[node.lineno - 1 : node.end_lineno or node.lineno]
                )
                if any(dn in body_src for dn in dec_names):
                    params = [a.arg for a in node.args.args]
                    start = node.lineno - 1
                    end = node.end_lineno or (start + 1)
                    return FuncInfo(
                        name=node.name,
                        params=params,
                        source="\n".join(self.lines[start:end]),
                        decorator="",
                        lineno=node.lineno,
                    )
        return None

    def extract_tensor_creation(self, func: FuncInfo) -> list[TensorInfo]:
        """Extract torch.randn/empty/zeros/ones calls from a function.

        Args:
            func (FuncInfo): The function whose body is scanned for
                tensor-creating assignments.

        Returns:
            list[TensorInfo]: One :class:`TensorInfo` per recognised
                tensor-creation assignment; empty if the body fails to
                parse or contains none.
        """
        results: list[TensorInfo] = []
        try:
            func_tree = ast.parse(textwrap.dedent(func.source))
        except SyntaxError:
            return results

        for node in ast.walk(func_tree):
            if not isinstance(node, ast.Assign):
                continue
            if not isinstance(node.value, ast.Call):
                continue
            call = node.value
            func_name = self._call_func_name(call)
            if func_name not in (
                "torch.randn", "torch.empty", "torch.zeros",
                "torch.ones", "torch.rand", "torch.empty_like",
                "torch.randn_like",
            ):
                continue
            target = node.targets[0]
            if isinstance(target, ast.Name):
                var_name = target.id
            else:
                continue

            shape_args = []
            dtype_expr = None
            for arg in call.args:
                shape_args.append(ast.unparse(arg))
            for kw in call.keywords:
                if kw.arg == "dtype":
                    dtype_expr = ast.unparse(kw.value)
                elif kw.arg not in ("device",):
                    shape_args.append(f"{kw.arg}={ast.unparse(kw.value)}")

            creation_expr = ast.unparse(node.value)
            results.append(TensorInfo(
                var_name=var_name,
                creation_expr=creation_expr,
                shape_args=shape_args,
                dtype_expr=dtype_expr,
            ))
        return results

    def extract_call_to(self, func: FuncInfo, callee_name: str) -> CallInfo | None:
        """Find a call to callee_name within func's body and extract its args.

        Args:
            func (FuncInfo): The function whose body is searched.
            callee_name (str): The callee name to match (suffix matches
                are accepted, e.g. ``mod.foo`` matches ``foo``).

        Returns:
            CallInfo | None: The matched call's :class:`CallInfo`, or
                ``None`` when no matching call is found / body fails to
                parse.
        """
        try:
            func_tree = ast.parse(textwrap.dedent(func.source))
        except SyntaxError:
            return None

        for node in ast.walk(func_tree):
            if not isinstance(node, ast.Call):
                continue
            name = self._call_func_name(node)
            if name == callee_name or (name and name.endswith(callee_name)):
                args = [ast.unparse(a) for a in node.args]
                kwargs = {kw.arg: ast.unparse(kw.value) for kw in node.keywords if kw.arg}
                return CallInfo(func_name=callee_name, args=args, kwargs=kwargs)
        return None

    def _call_func_name(self, call: ast.Call) -> str:
        """Reconstruct the dotted name of a call's callee.

        Args:
            call (ast.Call): The call node to inspect.

        Returns:
            str: The dotted callee name (e.g. ``torch.randn``), or an
                empty string when it cannot be resolved.
        """
        if isinstance(call.func, ast.Name):
            return call.func.id
        if isinstance(call.func, ast.Attribute):
            parts = []
            node = call.func
            while isinstance(node, ast.Attribute):
                parts.append(node.attr)
                node = node.value
            if isinstance(node, ast.Name):
                parts.append(node.id)
            return ".".join(reversed(parts))
        return ""

    def get_toplevel_statements(self) -> list[str]:
        """Get top-level assignment statements (e.g. torch.set_default_device).

        Returns:
            list[str]: Source lines of top-level call statements that set
                the default device or manual seed.
        """
        results = []
        for node in ast.iter_child_nodes(self.tree):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                line = "\n".join(self.lines[node.lineno - 1 : node.end_lineno or node.lineno])
                if "set_default_device" in line or "manual_seed" in line:
                    results.append(line)
        return results


# Config builder — from TraceLens input_shapes

def _build_configs(candidate: dict) -> tuple[str, str, str]:
    """Build (ALL_CONFIGS, cfg unpack, config_str) code from candidate shapes."""
    input_shapes = candidate.get("input_shapes") or []
    if not input_shapes:
        return _default_configs()

    # Each shape is {"call_num": N, "shape": "(M, N) dtype"}.
    parsed: list[tuple[tuple[int, ...], str, int]] = []
    for entry in input_shapes:
        shape_str = entry.get("shape", "")
        call_num = entry.get("call_num", 0)
        dims, dtype = _parse_shape_string(shape_str)
        if dims:
            parsed.append((dims, dtype, call_num))

    if not parsed:
        return _default_configs()

    # Most frequent first.
    parsed.sort(key=lambda x: -x[2])

    # Keep only the highest-dimensional shapes so all configs share tuple length.
    max_ndim = max(len(p[0]) for p in parsed)
    parsed = [p for p in parsed if len(p[0]) == max_ndim]

    if not parsed:
        return _default_configs()

    # Group by unique (dims, dtype).
    seen: set[tuple] = set()
    unique_configs: list[tuple[tuple[int, ...], str]] = []
    for dims, dtype, _ in parsed:
        key = (dims, dtype)
        if key not in seen:
            seen.add(key)
            unique_configs.append((dims, dtype))

    # Scale variants to reach >= 6 configs.
    base_configs = list(unique_configs)
    for dims, dtype in base_configs:
        if len(unique_configs) >= 8:
            break
        for scale in (2, 4, 8, 16):
            scaled = tuple(d * scale if i == 0 else d for i, d in enumerate(dims))
            key = (scaled, dtype)
            if key not in seen:
                seen.add(key)
                unique_configs.append((scaled, dtype))
            if len(unique_configs) >= 8:
                break

    dim_names = _dim_names(max_ndim)
    dtype_map = {"bf16": "torch.bfloat16", "fp16": "torch.float16",
                 "fp32": "torch.float32", "bfloat16": "torch.bfloat16",
                 "float16": "torch.float16", "float32": "torch.float32"}

    config_entries = []
    for dims, dtype in unique_configs:
        torch_dtype = dtype_map.get(dtype, f"torch.{dtype}" if dtype else "torch.bfloat16")
        dims_str = ", ".join(str(d) for d in dims)
        config_entries.append(f"    ({dims_str}, {torch_dtype})")

    all_configs = "[\n" + ",\n".join(config_entries) + ",\n]"
    unpack = ", ".join(dim_names[:max_ndim]) + ", dtype = cfg"
    config_str_parts = " ".join(
        f"{n}={{{n}}}" for n in dim_names[:max_ndim]
    )
    config_str_code = f'f"{config_str_parts} {{dtype}}"'

    return all_configs, unpack, config_str_code


def _default_configs() -> tuple[str, str, str]:
    """Return a generic 2D ``(M, N, dtype)`` config fallback.

    Returns:
        tuple[str, str, str]: ``(all_configs_code, cfg_unpack_code,
            config_str_code)`` for a default bf16 M×4096 sweep.
    """
    return (
        "[\n    (1, 4096, torch.bfloat16),\n    (32, 4096, torch.bfloat16),\n"
        "    (256, 4096, torch.bfloat16),\n    (1024, 4096, torch.bfloat16),\n"
        "    (4096, 4096, torch.bfloat16),\n    (8192, 4096, torch.bfloat16),\n]",
        "M, N, dtype = cfg",
        'f"M={M} N={N} {dtype}"',
    )


def _is_heterogeneous_multi_tensor(candidate: dict) -> bool:
    """True when the candidate's input_shapes describe several distinct tensors.

    The generic ``_build_configs`` path assumes a single GEMM-like operand whose
    leading dim is swept (M×N×K). When TraceLens captures an op with several
    *different-rank* tensors (e.g. paged attention: query (b,h,d), KV cache
    (pages,1,h,d), kv_indptr (n,), workspace (bytes,) ...) that assumption is
    wrong: ``_build_configs`` keeps only the highest-rank shapes and drops the
    rest, silently fabricating a GEMM harness that mismeasures the real op. We
    detect that here so the caller can REFUSE to fabricate rather than guess.

    Heuristic: >=2 input_shapes with >=2 distinct ranks (dim counts). Same-rank
    multi-shape (a normal shape sweep) is fine and returns False.
    """
    shapes = candidate.get("input_shapes")
    if not isinstance(shapes, list) or len(shapes) < 2:
        return False
    ranks: set[int] = set()
    for entry in shapes:
        shape_str = entry.get("shape", "") if isinstance(entry, dict) else str(entry)
        dims, _ = _parse_shape_string(shape_str)
        if dims:
            ranks.add(len(dims))
    return len(ranks) >= 2


def _parse_shape_string(s: str) -> tuple[tuple[int, ...], str]:
    """Parse '(256, 128) bf16' → ((256, 128), 'bf16').

    Args:
        s (str): A shape string of the form ``(d0, d1, ...) dtype``.

    Returns:
        tuple[tuple[int, ...], str]: The parsed integer dims and dtype
            token; returns ``((), "")`` when the string does not match
            or dims are non-integer. Dtype defaults to ``bfloat16`` when
            omitted.
    """
    m = re.match(r"\(([^)]+)\)\s*(\w+)?", s.strip())
    if not m:
        return (), ""
    dims_str = m.group(1)
    dtype = m.group(2) or "bfloat16"
    try:
        dims = tuple(int(d.strip()) for d in dims_str.split(",") if d.strip())
    except ValueError:
        return (), ""
    return dims, dtype


def _dim_names(ndim: int) -> list[str]:
    """Return human-friendly dimension names for ``ndim`` dimensions.

    Args:
        ndim (int): Number of dimensions to name.

    Returns:
        list[str]: ``["M", "N", "K", ...]`` truncated to ``ndim``, or
            ``["D0", "D1", ...]`` when ``ndim`` exceeds the preset names.
    """
    names = ["M", "N", "K", "L", "P", "Q"]
    return names[:ndim] if ndim <= len(names) else [f"D{i}" for i in range(ndim)]


# Adapter function generator

def _generate_setup_inputs(
    analyzer: BenchmarkAnalyzer,
    test_func: FuncInfo | None,
    cfg_unpack: str,
    ref_func: FuncInfo | None,
    kernel_func: FuncInfo | None,
) -> str:
    """Generate the setup_inputs(cfg) body.

    Inputs are created for the UNION of the kernel and reference function
    parameters, so ``run_ref`` can never reference a key that ``setup_inputs``
    did not create (root cause of the forge smoke-test ``KeyError`` that made
    every attention-kernel session spin with zero gain).

    Parameter element type is inferred from the parameter-name class
    (index / int-scalar / float-scalar / dtype / weight) instead of defaulting
    every argument to a 2D float tensor, which previously produced invalid
    inputs (e.g. float ``block_tables`` / ``seq_lens``) that crashed on the
    very first call.
    """
    dim_vars = cfg_unpack.replace(" = cfg", "").split(", ")
    dim_vars = [v.strip() for v in dim_vars if v.strip() != "dtype"]

    lines = [f"    {cfg_unpack}"]
    lines.append("    torch.manual_seed(42)")

    if not (kernel_func or ref_func):
        shape = ", ".join(dim_vars)
        lines.append(f'    x = torch.randn({shape}, dtype=dtype, device="cuda")')
        lines.append('    return {"x": x}')
        return "\n".join(lines)

    shape_2d = ", ".join(dim_vars[:2]) if len(dim_vars) >= 2 else dim_vars[0]
    shape_1d = dim_vars[-1] if dim_vars else "N"

    # Union of kernel + ref params (dedup by name, preferring a concrete
    # literal call value when one is available).
    used_params: list[tuple[str, str | None]] = []
    index_of: dict[str, int] = {}
    for fn in (kernel_func, ref_func):
        for name, call_value in _collect_used_params(analyzer, test_func, fn):
            if name in index_of:
                i = index_of[name]
                prev = used_params[i][1]
                if (call_value and not _is_variable(call_value)
                        and not (prev and not _is_variable(prev))):
                    used_params[i] = (name, call_value)
                continue
            index_of[name] = len(used_params)
            used_params.append((name, call_value))

    inputs_items: list[str] = []
    for param_name, call_value in used_params:
        p_lower = param_name.lower()

        # A literal call arg is stored directly.
        if call_value and not _is_variable(call_value):
            lines.append(f"    {param_name} = {call_value}")
        elif _is_dtype_param(p_lower):
            lines.append(f"    {param_name} = dtype")
        elif _is_index_param(p_lower):
            lines.append(f'    {param_name} = torch.zeros({shape_2d}, dtype=torch.int32, device="cuda")')
        elif _is_int_scalar_param(p_lower):
            lines.append(f"    {param_name} = 1")
        elif _is_float_scalar_param(p_lower):
            lines.append(f"    {param_name} = 1.0")
        elif _is_scalar_param(p_lower):
            val = "1e-06" if "eps" in p_lower else "0"
            lines.append(f"    {param_name} = {val}")
        elif _is_weight_param(p_lower):
            lines.append(f'    {param_name} = torch.randn({shape_1d}, dtype=dtype, device="cuda")')
        else:
            lines.append(f'    {param_name} = torch.randn({shape_2d}, dtype=dtype, device="cuda")')
        inputs_items.append(f'"{param_name}": {param_name}')

    if not inputs_items:
        lines.append(f'    x = torch.randn({shape_2d}, dtype=dtype, device="cuda")')
        inputs_items.append('"x": x')

    lines.append("    return {" + ", ".join(inputs_items) + "}")
    return "\n".join(lines)


def _collect_used_params(
    analyzer: BenchmarkAnalyzer,
    test_func: FuncInfo | None,
    func: FuncInfo | None,
) -> list[tuple[str, str | None]]:
    """Return [(param, call_value_or_None), ...] for one function's params."""
    if not func:
        return []
    call = analyzer.extract_call_to(test_func, func.name) if test_func else None
    params = [p for p in func.params if p != "self"]
    if call:
        return _match_call_args_to_params(call, params)
    return [(p, None) for p in params]


def _match_call_args_to_params(
    call: CallInfo, params: list[str]
) -> list[tuple[str, str | None]]:
    """Match call args to params, returning [(param, call_value_or_None), ...] (positional + tensor kwargs)."""
    result: list[tuple[str, str | None]] = []

    # Positional args are always required.
    for i, arg_val in enumerate(call.args):
        if i < len(params):
            result.append((params[i], arg_val))

    # Skip kwargs for dtypes, quant settings, modes, flags, etc.
    SKIP_KWARG_HINTS = {
        "dtype", "q_dtype", "quant_dtype", "quant_type", "type",
        "mode", "model_sensitive", "use_model_sensitive", "group_size",
        "shuffle", "out_before_quant",
    }
    matched_params = {p for p, _ in result}
    for kw_name, kw_val in call.kwargs.items():
        if kw_name in params and kw_name not in matched_params:
            if kw_val in ("None",):
                continue
            if kw_name in SKIP_KWARG_HINTS:
                continue
            result.append((kw_name, kw_val))

    return result


def _is_scalar_param(name: str) -> bool:
    """Heuristically decide whether a parameter is a scalar.

    Args:
        name (str): Lowercased parameter name.

    Returns:
        bool: True for known scalar names (e.g. ``eps``, ``dropout``).
    """
    SCALAR_EXACT = {"eps", "epsilon", "p", "dropout", "model_sensitive"}
    SCALAR_CONTAINS = {"use_model_sensitive"}
    return name in SCALAR_EXACT or any(h in name for h in SCALAR_CONTAINS)


def _is_weight_param(name: str) -> bool:
    """Heuristically decide whether a parameter is a 1D weight/bias.

    Args:
        name (str): Lowercased parameter name.

    Returns:
        bool: True for known weight/bias names or ``*_weight``-style
            suffixes (so a 1D tensor is created for them).
    """
    WEIGHT_EXACT = {"weight", "w", "gamma", "bias", "beta"}
    return name in WEIGHT_EXACT or any(name.endswith(f"_{h}") for h in WEIGHT_EXACT)


def _is_index_param(name: str) -> bool:
    """Heuristically decide whether a parameter is an integer index/length tensor.

    These (block tables, sequence lengths, indptr/indices) must be integer
    tensors; building them with ``torch.randn`` (float) crashes the kernel on
    the first call.

    Args:
        name (str): Lowercased parameter name.

    Returns:
        bool: True for index / sequence-length tensor parameters.
    """
    INDEX_EXACT = {
        "block_tables", "block_table", "seq_lens", "seqlens", "context_lens",
        "cu_seqlens", "kv_indptr", "qo_indptr", "block_tables_stride0",
    }
    if name in INDEX_EXACT:
        return True
    return (
        name.endswith("_lens")
        or name.endswith("_indices")
        or name.endswith("_indptr")
        or name.endswith("_idx")
        or name.startswith("block_table")
    )


def _is_int_scalar_param(name: str) -> bool:
    """Heuristically decide whether a parameter is an integer scalar.

    Sizes, counts, lengths and strides are Python ints, not float tensors.

    Args:
        name (str): Lowercased parameter name.

    Returns:
        bool: True for integer scalar parameters.
    """
    INT_EXACT = {
        "max_seq_len", "max_qlen", "num_kv_heads", "num_heads", "num_seqs",
        "head_size", "block_size", "num_queries_per_kv", "high_precision",
        "quant_algo", "partition_size", "max_num_partitions",
    }
    if name in INT_EXACT:
        return True
    return (
        name.startswith("num_")
        or name.startswith("max_")
        or name.startswith("stride")
        or name.endswith("_heads")
        or name.endswith("_size")
        or name.endswith("_len")
        or name.endswith("_stride")
    )


def _is_float_scalar_param(name: str) -> bool:
    """Heuristically decide whether a parameter is a float scalar.

    Softmax scale and similar coefficients are float scalars. ``*_scale_cache``
    style names are excluded since those are per-token scale tensors.

    Args:
        name (str): Lowercased parameter name.

    Returns:
        bool: True for float scalar parameters.
    """
    FLOAT_EXACT = {"scale", "softmax_scale", "sm_scale", "scaling", "alpha"}
    if name in FLOAT_EXACT:
        return True
    return name.endswith("_scale") and "cache" not in name


def _is_dtype_param(name: str) -> bool:
    """Heuristically decide whether a parameter carries a dtype.

    Args:
        name (str): Lowercased parameter name.

    Returns:
        bool: True for dtype-carrying parameters (e.g. ``kv_cache_dtype``).
    """
    return name == "dtype" or name.endswith("_dtype")


def _generate_run_kernel(
    analyzer: BenchmarkAnalyzer,
    test_func: FuncInfo | None,
    kernel_func: FuncInfo | None,
) -> str:
    """Generate the run_kernel(inputs) function body.

    Args:
        analyzer (BenchmarkAnalyzer): Analyzer over the benchmark source.
        test_func (FuncInfo | None): The orchestrator function, used to
            mirror how the kernel is invoked.
        kernel_func (FuncInfo | None): The kernel function to call.

    Returns:
        str: The indented body source for ``run_kernel``; a passthrough
            that returns the first input when no kernel function exists.
    """
    if not kernel_func:
        return '    return inputs.get("x", list(inputs.values())[0])'

    return _generate_run_func_body(analyzer, test_func, kernel_func)


def _generate_run_ref(
    analyzer: BenchmarkAnalyzer,
    test_func: FuncInfo | None,
    ref_func: FuncInfo | None,
    kernel_func: FuncInfo | None,
) -> str:
    """Generate the run_ref(inputs) function body.

    Args:
        analyzer (BenchmarkAnalyzer): Analyzer over the benchmark source.
        test_func (FuncInfo | None): The orchestrator function, used to
            mirror how the reference is invoked.
        ref_func (FuncInfo | None): The reference function to call.
        kernel_func (FuncInfo | None): The kernel function (unused
            directly; present for signature symmetry).

    Returns:
        str: The indented body source for ``run_ref``; delegates to
            ``run_kernel`` when no reference function exists.
    """
    if not ref_func:
        return "    return run_kernel(inputs)"

    return _generate_run_func_body(analyzer, test_func, ref_func)


def _generate_run_func_body(
    analyzer: BenchmarkAnalyzer,
    test_func: FuncInfo | None,
    target_func: FuncInfo,
) -> str:
    """Generate a body that calls target_func with inputs dict values; missing params use defaults."""
    call = None
    if test_func:
        call = analyzer.extract_call_to(test_func, target_func.name)

    params = [p for p in target_func.params if p != "self"]

    if call:
        used = _match_call_args_to_params(call, params)
        # Use .get so a ref-only param missing from setup_inputs degrades to
        # None instead of raising KeyError at smoke-test time.
        args_parts: list[str] = [f'inputs.get("{param_name}")' for param_name, _ in used]
    else:
        args_parts = [f'inputs.get("{p}")' for p in params]

    call_str = f"    result = {target_func.name}({', '.join(args_parts)})"
    lines = [call_str]
    # Unwrap nested @perftest tuples down to the func result.
    lines.append("    while isinstance(result, tuple) and len(result) >= 2:")
    lines.append("        result = result[0]")
    lines.append("    return result")
    return "\n".join(lines)


def _is_variable(s: str) -> bool:
    """Check if a string looks like a Python variable name (not a literal).

    Args:
        s (str): The unparsed argument expression to test.

    Returns:
        bool: True if ``s`` is a bare identifier and not ``True`` /
            ``False`` / ``None``.
    """
    return bool(re.match(r"^[a-zA-Z_]\w*$", s)) and s not in (
        "True", "False", "None",
    )


# HARNESS_TEMPLATE — GEAK FIXED boilerplate

HARNESS_TEMPLATE = '''\
#!/usr/bin/env python3
"""Auto-generated GEAK-compatible test harness."""
import argparse
import os
import sys
import math
import torch

# ══════════════════════════════════════════════════════════════════════
# ██  FIXED BOILERPLATE — do NOT modify                              ██
# ══════════════════════════════════════════════════════════════════════

REPO_ROOT = os.environ.get(
    "GEAK_WORK_DIR",
    os.environ.get("GEAK_REPO_ROOT", {repo_root!r}),
)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# aiter JIT routing (no-op for non-aiter kernels; only aiter's JIT reads these).
# aiter is installed EDITABLE via a sys.meta_path finder, so `import aiter` resolves
# to the ORIGINAL repo regardless of sys.path order, and its JIT compiles
# AITER_CSRC_DIR=$AITER_META_DIR/csrc. Without overriding AITER_META_DIR the JIT
# builds the BASELINE csrc/*.cu -> every speedup is ~1.00x and correctness is blind
# to the patch (the worktree-bypass bug). AITER_JIT_DIR routes build output to a
# per-worktree dir so artifacts don't pollute the source package and parallel slots
# don't collide on a shared build/.so/ninja-lock. Both MUST be set BEFORE import aiter.
os.environ.setdefault("AITER_META_DIR", REPO_ROOT)
os.environ.setdefault("AITER_JIT_DIR", os.path.join(REPO_ROOT, "_aiter_jit"))

WARMUP = 50
ITERATIONS = int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "200"))

# ══════════════════════════════════════════════════════════════════════
# ██  Imports and functions from original benchmark                  ██
# ══════════════════════════════════════════════════════════════════════

{imports_section}

{toplevel_stmts}

{function_defs}

# ══════════════════════════════════════════════════════════════════════
# ██  ADAPT — kernel-specific configuration and adapters             ██
# ══════════════════════════════════════════════════════════════════════

ALL_CONFIGS = {all_configs}


def setup_inputs(cfg):
{setup_inputs_body}


def run_kernel(inputs):
{run_kernel_body}


def run_ref(inputs):
{run_ref_body}


def config_str(cfg):
    {cfg_unpack_code}
    return {config_str_code}


# ══════════════════════════════════════════════════════════════════════
# ██  FIXED BOILERPLATE — benchmark & mode infrastructure            ██
# ══════════════════════════════════════════════════════════════════════

def _pick(configs, count):
    if len(configs) <= count:
        return list(range(len(configs)))
    n = len(configs)
    return [round(i * (n - 1) / (count - 1)) for i in range(count)]


def check_correctness_val(out_ref, out_kernel, dtype=torch.float16):
    tol_map = {{
        torch.float32: (1e-4, 1e-4, 0.05),
        torch.float16: (1e-2, 1e-2, 0.10),
        torch.bfloat16: (1e-2, 1e-2, 0.10),
        torch.float8_e4m3fnuz: (5e-2, 5e-2, 0.20),
        torch.float8_e5m2fnuz: (5e-2, 5e-2, 0.20),
    }}
    rtol, atol, max_err_ratio = tol_map.get(dtype, (1e-2, 1e-2, 0.20))
    isClose = torch.isclose(out_ref, out_kernel, rtol=rtol, atol=atol)
    err_ratio = 0.0 if isClose.all() else (~isClose).sum().item() / out_ref.numel()
    x, y = out_ref.double(), out_kernel.double()
    denom = (x * x + y * y).sum().item()
    cos_diff = 1 - 2 * (x * y).sum().item() / max(denom, 1e-12)
    return err_ratio <= max_err_ratio, err_ratio, cos_diff


def benchmark_kernel(inputs):
    """Benchmark with GPU events. Returns median latency in ms."""
    def fn():
        run_kernel(inputs)
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    latencies = []
    for _ in range(ITERATIONS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        latencies.append(start.elapsed_time(end))
    latencies.sort()
    return latencies[len(latencies) // 2]


def mode_correctness(indices):
    print(f"Running correctness check on {{len(indices)}} configs...")
    all_pass = True
    for idx in indices:
        cfg = ALL_CONFIGS[idx]
        label = config_str(cfg)
        try:
            inputs = setup_inputs(cfg)
            out = run_kernel(inputs)
            ref = run_ref(inputs)
            passed, err_ratio, cos_diff = check_correctness_val(ref, out)
            status = "PASS" if passed else "FAIL"
            print(f"  [{{idx}}] {{label}}  err_ratio={{err_ratio:.4f}} cos_diff={{cos_diff:.2e}}  {{status}}")
            if not passed:
                all_pass = False
        except Exception as e:
            print(f"  [{{idx}}] {{label}}  ERROR: {{e}}")
            all_pass = False
        finally:
            torch.cuda.empty_cache()
    print(f"GEAK_SHAPES_USED={{indices}}")
    if not all_pass:
        print("CORRECTNESS FAILED")
        sys.exit(1)
    print("ALL CORRECTNESS CHECKS PASSED")


def mode_benchmark(indices):
    print(f"Running benchmark on {{len(indices)}} configs...")
    latencies = []
    for idx in indices:
        cfg = ALL_CONFIGS[idx]
        label = config_str(cfg)
        try:
            inputs = setup_inputs(cfg)
            ms = benchmark_kernel(inputs)
            print(f"  {{label}}  {{ms:.4f}}ms")
            latencies.append(ms)
        except Exception as e:
            print(f"  {{label}}  ERROR: {{e}}")
        finally:
            torch.cuda.empty_cache()
    print(f"GEAK_SHAPES_USED={{indices}}")
    if latencies:
        geo_mean = math.exp(sum(math.log(x) for x in latencies) / len(latencies))
        print(f"GEAK_RESULT_LATENCY_MS={{geo_mean:.4f}}")
    else:
        print("No successful benchmarks")
        sys.exit(1)


def mode_profile(indices):
    print(f"Running profile on {{len(indices)}} configs...")
    for idx in indices:
        cfg = ALL_CONFIGS[idx]
        label = config_str(cfg)
        try:
            inputs = setup_inputs(cfg)
            run_kernel(inputs)
            print(f"  {{label}}  OK")
        except Exception as e:
            print(f"  {{label}}  ERROR: {{e}}")
        finally:
            torch.cuda.empty_cache()
    print(f"GEAK_SHAPES_USED={{indices}}")


def main():
    parser = argparse.ArgumentParser(description="Test harness")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--correctness", action="store_true")
    group.add_argument("--benchmark", action="store_true")
    group.add_argument("--full-benchmark", action="store_true")
    group.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    total = len(ALL_CONFIGS)
    print(f"Total configs: {{total}}")
    if args.correctness:
        mode_correctness(_pick(ALL_CONFIGS, 25))
    elif args.benchmark:
        mode_benchmark(_pick(ALL_CONFIGS, 25))
    elif args.full_benchmark:
        mode_benchmark(list(range(total)))
    elif args.profile:
        mode_profile(_pick(ALL_CONFIGS, 5))


if __name__ == "__main__":
    main()
'''


# Main entry point

_AITER_HARNESS_TEMPLATE = '''#!/usr/bin/env python3
"""Auto-generated Forge/GEAK harness for an aiter op_test.

Reuses the aiter test's own @benchmark function (which builds inputs, computes a
torch reference inline, and times aiter.<op> via run_perftest, returning a dict
with ``us`` and ``err``). GEAK-contract compliant: reads GEAK_WORK_DIR /
GEAK_BENCHMARK_ITERATIONS, supports --correctness/--benchmark/--profile/
--full-benchmark, and emits GEAK_SHAPES_USED + GEAK_RESULT_LATENCY_MS. GPU event
timing (torch.cuda.Event(enable_timing=True)) is used as a fallback when aiter's
own run_perftest timing is unavailable.
"""
import argparse
import os
import sys

# GEAK places patched candidate code in GEAK_WORK_DIR; prepend it so edits load.
_GEAK_WORK_DIR = os.environ.get("GEAK_WORK_DIR", "")
if _GEAK_WORK_DIR and _GEAK_WORK_DIR not in sys.path:
    sys.path.insert(0, _GEAK_WORK_DIR)
# aiter JIT routing (no-op for non-aiter kernels; only aiter's JIT reads these).
# aiter is editable via a sys.meta_path finder, so `import aiter` resolves to the
# ORIGINAL repo regardless of sys.path; its JIT compiles AITER_CSRC_DIR=$AITER_META_DIR/csrc.
# Without overriding AITER_META_DIR the JIT builds the BASELINE csrc/*.cu -> ~1.00x and
# correctness is blind to the patch. AITER_JIT_DIR routes build output to a per-worktree
# dir (no source-repo pollution, no parallel-slot collisions). Set BEFORE import aiter.
if _GEAK_WORK_DIR:
    os.environ.setdefault("AITER_META_DIR", _GEAK_WORK_DIR)
    os.environ.setdefault("AITER_JIT_DIR", os.path.join(_GEAK_WORK_DIR, "_aiter_jit"))
# GEAK controls benchmark iteration count via this env var.
GEAK_BENCHMARK_ITERATIONS = int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "30"))

__IMPORTS__

__HELPER_DEFS__

__TEST_FN_SRC__


def _call_args():
    # Shapes baked in from the kernel candidate at generation time.
    return __CALL_KWARGS__


def _event_time_ms(fn):
    """Fallback GPU timing via torch.cuda.Event(enable_timing=True)."""
    import torch
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    fn()  # warmup
    torch.cuda.synchronize()
    start.record()
    for _ in range(GEAK_BENCHMARK_ITERATIONS):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / max(1, GEAK_BENCHMARK_ITERATIONS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--correctness", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--full-benchmark", action="store_true")
    a, _ = ap.parse_known_args()
    kw = _call_args()
    print(f"GEAK_SHAPES_USED={kw}")
    try:
        ret = __TEST_FN_NAME__(**kw)
    except Exception as exc:  # noqa: BLE001
        print(f"correctness: error ({type(exc).__name__}: {exc})")
        sys.exit(1)
    ret = ret if isinstance(ret, dict) else {}
    us = ret.get("us")
    err = ret.get("err")
    if a.benchmark or a.full_benchmark or a.profile:
        ms = None
        if us is not None:
            ms = float(us) / 1000.0  # aiter run_perftest reports us/iter
        else:
            # Fallback: time a re-invocation of the test fn with GPU events.
            try:
                ms = _event_time_ms(lambda: __TEST_FN_NAME__(**kw))
            except Exception as exc:  # noqa: BLE001
                print(f"benchmark: error ({type(exc).__name__}: {exc})")
        if ms is not None:
            print(f"GEAK_RESULT_LATENCY_MS={ms:.6f}")
        sys.exit(0)
    # correctness: aiter checkAllclose returns an error ratio (lower is better).
    if err is None:
        print("correctness: unknown (no err metric)")
    else:
        try:
            e = float(err)
            print(f"allclose: {'True' if e <= 0.05 else 'False'} (err={e:.4f})")
        except (TypeError, ValueError):
            print("allclose: True")
    sys.exit(0)


main()
'''


def _try_generate_aiter_harness(
    analyzer: "BenchmarkAnalyzer",
    decorated: dict,
    candidate: dict,
    source_file: str,
    benchmark_path: "Path",
    out_dir: "Path",
    log: Callable[[str], None],
) -> SimpleNamespace | None:
    """Generate a harness from an aiter op_test's @benchmark function.

    Detects the aiter idiom (imports aiter + a @benchmark function whose body
    calls run_perftest / aiter.<op>), then emits a self-contained harness that
    calls that function with the candidate's shapes and reports timing +
    correctness in the Forge contract. Returns None when the file isn't a
    recognisable aiter op_test.

    NOTE: end-to-end runs require aiter's HIP/CK JIT runtime; generation +
    static_check are validated by unit tests, runtime is gated on the image.
    """
    imports = analyzer.get_imports()
    if not any("aiter" in imp for imp in imports):
        return None
    # Recognize both aiter perf decorators. Many aiter op_tests time the op with
    # @perftest rather than @benchmark (e.g. test_batch_prefill.py), so matching
    # only @benchmark misses them and forces the weaker generic path. A bare
    # passthrough wrapper (e.g. @perftest def profile_func(target_func, *args,
    # **kwargs)) carries no mappable shape params and is filtered by the kwargs
    # guard below, so widening to @perftest only adds real benchmark fns.
    bench_fns = [fi for fi in decorated.values()
                 if fi.decorator in ("benchmark", "perftest")]
    if not bench_fns:
        return None
    # Pick the first perf fn that actually times an op (run_perftest / aiter.<op>);
    # this also skips passthrough wrappers whose body just forwards to target_func.
    test_fn = next(
        (fi for fi in bench_fns
         if "run_perftest" in fi.source or "aiter." in fi.source),
        bench_fns[0],
    )
    log(f"aiter idiom: reusing @benchmark fn {test_fn.name!r}")

    # Map the candidate's traced shapes to the test fn's int params so the
    # harness pins the EXACT serving shape (user_task:production), never the
    # test's own synthetic sweep. Two mapping layers:
    #  (1) NAME-aware: pin params whose name matches a known dim role (token /
    #      model_dim / inter_dim / m / n / k / seqlen / batch). This is what
    #      lets MoE fns like test_fmoe(dtype, token, model_dim, inter_dim, ...)
    #      bind correctly — previously `token`/`model_dim` were unrecognized and
    #      the harness fell back to a CSV token sweep (wrong shapes).
    #  (2) POSITIONAL fallback for the classic (m, n, k) GEMM idiom.
    shape = _aiter_shape_from_candidate(candidate)
    kwargs: dict[str, object] = {}
    # Role -> value from the traced operands.
    role_value = {
        "m": shape.get("M"), "token": shape.get("M"), "tokens": shape.get("M"),
        "tokennum": shape.get("M"), "num_tokens": shape.get("M"),
        "numtokens": shape.get("M"), "seqlen": shape.get("M"), "batch": shape.get("M"),
        "n": shape.get("N"), "model_dim": shape.get("N"), "dim": shape.get("N"),
        "hidden": shape.get("N"), "hidden_size": shape.get("N"),
        "k": shape.get("K"), "inter_dim": shape.get("K"),
        "intermediate": shape.get("K"), "inter": shape.get("K"),
    }
    positional = [shape.get(k) for k in ("M", "N", "K") if shape.get(k)]
    si = 0
    for p in test_fn.params:
        pl = p.lower()
        if pl in ("dtype", "input_dtype"):
            kwargs[p] = "__DTYPE__"
        elif pl in role_value and role_value[pl] is not None:
            kwargs[p] = role_value[pl]
        elif pl in ("m", "n", "k", "b", "batch", "num_tokens", "seqlen", "dim") and si < len(positional):
            kwargs[p] = positional[si]
            si += 1
    # Require that we pinned at least the leading dim (token/M); otherwise the
    # harness would benchmark unfaithful shapes — refuse rather than mislead.
    pinned_dims = [v for k, v in kwargs.items() if v != "__DTYPE__"]
    if not pinned_dims:
        log("aiter idiom: could not map any candidate shape to fn params")
        return None
    log(f"aiter idiom: pinned traced shape params {[(k,v) for k,v in kwargs.items() if v!='__DTYPE__']}")

    # Render kwargs dict; dtype placeholder becomes a torch dtype literal.
    dtype_literal = _aiter_torch_dtype(candidate)
    kw_items = []
    for k, v in kwargs.items():
        kw_items.append(f"{k!r}: {dtype_literal}" if v == "__DTYPE__" else f"{k!r}: {v}")
    call_kwargs = "{" + ", ".join(kw_items) + "}"

    # Copy module-level helper functions (e.g. torch_* references) that the
    # test fn may call, excluding decorated functions (kept separately).
    helper_defs = _aiter_module_funcs(analyzer, exclude=set(decorated.keys()))

    harness_code = (
        _AITER_HARNESS_TEMPLATE
        .replace("__IMPORTS__", "\n".join(imports))
        .replace("__HELPER_DEFS__", "\n\n".join(helper_defs))
        .replace("__TEST_FN_SRC__", test_fn.source)
        .replace("__TEST_FN_NAME__", test_fn.name)
        .replace("__CALL_KWARGS__", call_kwargs)
    )

    harness_dir = out_dir / "unittest"
    harness_dir.mkdir(parents=True, exist_ok=True)
    harness_path = harness_dir / f"harness_aiter_{benchmark_path.stem}.py"
    harness_path.write_text(harness_code)
    log(f"wrote aiter harness: {harness_path}")

    try:
        validator_path = Path(__file__).parent.parent / "skills" / "unittest"
        if str(validator_path) not in sys.path:
            sys.path.insert(0, str(validator_path))
        from validate_harness import static_check
        ok, errs = static_check(str(harness_path))
        if not ok:
            log(f"aiter harness failed static_check: {errs}")
            return None
        log("aiter harness passed static_check")
    except Exception as exc:  # noqa: BLE001
        log(f"aiter harness static_check skipped: {exc}")

    return SimpleNamespace(
        harness_path=str(harness_path),
        test_command=f"python {harness_path} --correctness",
    )


def _parse_traced_operand_dims(candidate: dict) -> list[tuple[int, ...]]:
    """Parse the candidate's traced operand shapes into a list of int-tuples.

    Handles both the dict ``{M,N,K}`` form and the TraceLens list form where
    each entry is a string (or ``{"shape": str}``) of ``<br>``-joined operand
    shapes, e.g. ``"(64,2048) bf16<br>(128,1536,2048) bf16<br>..."``. Returns
    the operand dim-tuples in call order so callers can pin exact serving shapes
    (token=first 2D operand's leading dim, etc.) rather than a synthetic sweep.
    """
    import re as _re

    raw = candidate.get("input_shapes") or candidate.get("shapes") or []
    text = ""
    if isinstance(raw, list) and raw:
        first = raw[0]
        text = first.get("shape", "") if isinstance(first, dict) else str(first)
    elif isinstance(raw, str):
        text = raw
    dims: list[tuple[int, ...]] = []
    for tok in text.split("<br>"):
        m = _re.search(r"\(([\d,\s]+)\)", tok)
        if not m:
            continue
        nums = [int(x) for x in m.group(1).split(",") if x.strip().isdigit()]
        if nums:
            dims.append(tuple(nums))
    return dims


def _aiter_shape_from_candidate(candidate: dict) -> dict:
    """Best-effort {M,N,K} extraction from a candidate's input_shapes."""
    out: dict[str, int] = {}
    shapes = candidate.get("input_shapes") or candidate.get("shapes") or {}
    if isinstance(shapes, dict):
        for k in ("M", "N", "K"):
            v = shapes.get(k) or shapes.get(k.lower())
            if isinstance(v, int):
                out[k] = v
    # List/string TraceLens form: derive M/N from the FIRST 2-D operand (the
    # activation, e.g. (token, model_dim)), and the reduction/inter dim from the
    # expert-weight tensors. For MoE the weights are 3-D (E, *, *):
    #   w1 = (E, 2*inter_dim, model_dim), w2 = (E, model_dim, inter_dim).
    # inter_dim is therefore w2's LAST axis (NOT a middle axis — that earlier
    # heuristic mis-read topk/2*inter and pinned an unfaithful shape). We detect
    # w2 as the 3-D weight whose middle dim == model_dim (N).
    if not out:
        dims = _parse_traced_operand_dims(candidate)
        first_2d = next((d for d in dims if len(d) == 2), None)
        if first_2d:
            out["M"], out["N"] = first_2d[0], first_2d[1]
        weights_3d = [d for d in dims if len(d) == 3]
        model_dim = out.get("N")
        inter = None
        if model_dim is not None:
            # w2 = (E, model_dim, inter_dim): match middle axis to model_dim.
            w2 = next((d for d in weights_3d if d[1] == model_dim), None)
            if w2 is not None:
                inter = w2[2]
        if inter is None and weights_3d:
            # Fallback: smallest trailing dim among 3-D weights.
            inter = min(d[2] for d in weights_3d)
        if inter is not None:
            out["K"] = inter
    return out


def _aiter_torch_dtype(candidate: dict) -> str:
    """Map the candidate precision to a torch dtype literal (default bf16)."""
    prec = str(candidate.get("precision") or candidate.get("dtype") or "bf16").lower()
    return {
        "fp16": "torch.float16", "float16": "torch.float16",
        "fp32": "torch.float32", "float32": "torch.float32",
        "bf16": "torch.bfloat16", "bfloat16": "torch.bfloat16",
    }.get(prec, "torch.bfloat16")


def _aiter_module_funcs(analyzer: "BenchmarkAnalyzer", exclude: set) -> list[str]:
    """Return source of top-level non-decorated functions (torch refs/helpers)."""
    out: list[str] = []
    for node in ast.walk(analyzer.tree):
        if not isinstance(node, ast.FunctionDef) or node.name in exclude:
            continue
        if node.decorator_list:
            continue
        start = node.lineno - 1
        end = node.end_lineno or (start + 1)
        out.append("\n".join(analyzer.lines[start:end]))
    return out


def maybe_generate_harness(
    benchmark_file: str,
    candidate: dict,
    source_file: str,
    out_dir: "Path",
    kernel_repo: str = "",
    log_fn: Callable[[str], None] | None = None,
) -> SimpleNamespace | None:
    """Generate a GEAK-compatible harness from a benchmark file.

    Analyzes the benchmark, classifies reference/kernel functions,
    synthesises config + adapter code, renders the fixed harness
    template, writes it under ``<out_dir>/unittest/``, and validates it
    with ``static_check``. Fail-soft: returns ``None`` on any
    unrecoverable condition (already-valid input, no decorated functions,
    parse error, failed validation).

    Args:
        benchmark_file (str): Path to the source benchmark ``.py`` file.
        candidate (dict): Hot-kernel candidate supplying ``input_shapes``.
        source_file (str): Path to the kernel's source file, used to
            derive the module path and guess the repo root.
        out_dir (Path): Base output directory; the harness is written to
            its ``unittest`` subdirectory.
        kernel_repo (str): Optional repo root; auto-detected from
            ``source_file`` when empty.
        log_fn (Callable[[str], None] | None): Optional logging callback;
            messages are prefixed and exceptions in it are swallowed.

    Returns:
        SimpleNamespace | None: ``SimpleNamespace(harness_path,
            test_command)`` on success, or ``None`` on any failure.
    """
    from pathlib import Path as _Path
    out_dir = _Path(out_dir)

    def _log(msg: str) -> None:
        """Forward a prefixed message to ``log_fn`` if one was provided.

        Args:
            msg (str): The message to log; emitted with a ``[harness_gen]``
                prefix. Failures in the callback are ignored.
        """
        if log_fn:
            try:
                log_fn(f"[harness_gen] {msg}")
            except Exception:
                pass

    benchmark_path = _Path(benchmark_file)
    if not benchmark_path.is_file():
        _log(f"benchmark file not found: {benchmark_file}")
        return None

    # L1: skip generation if the benchmark is already a valid harness.
    try:
        validator_path = _Path(__file__).parent.parent / "skills" / "unittest" / "validate_harness.py"
        if validator_path.is_file():
            sys.path.insert(0, str(validator_path.parent))
            from validate_harness import static_check
            ok, _ = static_check(benchmark_file)
            if ok:
                _log("benchmark file already passes static_check, skipping generation")
                return None
    except Exception as exc:
        _log(f"static_check import failed: {exc}")

    source = benchmark_path.read_text()

    # Derive the source module by finding the top-level package on the path.
    source_module = ""
    if source_file:
        sf = _Path(source_file)
        parts = sf.parts
        for i, p in enumerate(parts):
            if (
                i > 0
                and p != "__init__.py"
                and not p.startswith(".")
                and _Path(*parts[: i + 1]).is_dir()
            ):
                pkg_dir = _Path(*parts[:i]) / p
                if (pkg_dir / "__init__.py").exists():
                    module_parts = list(parts[i:])
                    if module_parts[-1].endswith(".py"):
                        module_parts[-1] = module_parts[-1][:-3]
                    source_module = ".".join(module_parts)
                    break

    _log(f"analyzing benchmark: {benchmark_file}, source_module={source_module}")

    try:
        analyzer = BenchmarkAnalyzer(source, source_module)
    except SyntaxError as exc:
        _log(f"AST parse failed: {exc}")
        return None

    imports = analyzer.get_imports()
    decorated = analyzer.get_decorated_functions()
    toplevel = analyzer.get_toplevel_statements()

    if not decorated:
        _log("no @perftest/@benchmark decorated functions found")
        return None

    ref_func, kernel_func = analyzer.classify_functions(decorated)
    test_func = analyzer.get_test_function(decorated)

    _log(f"found: ref={ref_func.name if ref_func else None}, "
         f"kernel={kernel_func.name if kernel_func else None}, "
         f"test={test_func.name if test_func else None}")

    if not kernel_func and not ref_func:
        # aiter op_tests don't expose standalone kernel/ref funcs: a single
        # @benchmark function builds inputs, computes a torch ref inline, and
        # times aiter.<op> via run_perftest (returning a dict with us/err).
        # Reuse that function directly instead of failing (RCA compiled-kernel A).
        aiter_hr = _try_generate_aiter_harness(
            analyzer, decorated, candidate, source_file, benchmark_path,
            out_dir, _log,
        )
        if aiter_hr is not None:
            return aiter_hr
        _log("could not identify kernel or reference function")
        return None

    # Refuse to fabricate a GEMM harness for a heterogeneous multi-tensor op.
    # _build_configs would keep only the highest-rank shapes and drop the rest,
    # silently mismeasuring (this is what broke paged-attention: the flat
    # multi-tensor shapes were flattened to M/N/K -> broken unpack). First RETRY
    # via the op_test idiom (reuses the kernel's own @perftest/@benchmark fn,
    # which builds the real multi-tensor inputs); only if that also fails do we
    # refuse, so the caller skips-with-reason instead of dispatching a blind one.
    if _is_heterogeneous_multi_tensor(candidate):
        _log("heterogeneous multi-tensor input_shapes; GEMM config builder "
             "unsafe, retrying via op_test idiom")
        aiter_hr = _try_generate_aiter_harness(
            analyzer, decorated, candidate, source_file, benchmark_path,
            out_dir, _log,
        )
        if aiter_hr is not None:
            return aiter_hr
        _log("HARNESS_SPEC_INSUFFICIENT: heterogeneous multi-tensor op with no "
             "usable op_test idiom; refusing to fabricate a GEMM harness")
        return None

    all_configs, cfg_unpack, config_str_code = _build_configs(candidate)

    setup_body = _generate_setup_inputs(
        analyzer, test_func, cfg_unpack, ref_func, kernel_func,
    )
    run_kernel_body = _generate_run_kernel(
        analyzer, test_func, kernel_func,
    )
    run_ref_body = _generate_run_ref(
        analyzer, test_func, ref_func, kernel_func,
    )

    # Copy decorated defs, excluding the test orchestrator (its argparse would conflict).
    func_defs_to_copy: list[str] = []
    for fi in decorated.values():
        if fi.decorator != "benchmark":
            func_defs_to_copy.append(fi.source)

    repo_root = kernel_repo or ""
    if not repo_root and source_file:
        sf = _Path(source_file)
        for parent in sf.parents:
            if (parent / ".git").exists() or (parent / "setup.py").exists() or (parent / "pyproject.toml").exists():
                repo_root = str(parent)
                break

    filtered_imports: list[str] = []
    for imp in imports:
        filtered_imports.append(imp)

    harness_code = HARNESS_TEMPLATE.format(
        repo_root=repo_root,
        imports_section="\n".join(filtered_imports),
        toplevel_stmts="\n".join(toplevel),
        function_defs="\n\n".join(func_defs_to_copy),
        all_configs=all_configs,
        setup_inputs_body=setup_body,
        run_kernel_body=run_kernel_body,
        run_ref_body=run_ref_body,
        cfg_unpack_code=cfg_unpack,
        config_str_code=config_str_code,
    )

    harness_dir = out_dir / "unittest"
    harness_dir.mkdir(parents=True, exist_ok=True)
    harness_path = harness_dir / f"harness_{benchmark_path.stem}.py"
    harness_path.write_text(harness_code)
    _log(f"wrote harness: {harness_path}")

    # L2: Validate with static_check
    try:
        from validate_harness import static_check
        ok, errs = static_check(str(harness_path))
        if not ok:
            _log(f"generated harness failed static_check: {errs}")
            # Keep the file for debugging.
            return None
        _log("generated harness passed static_check")
    except Exception as exc:
        _log(f"static_check validation failed: {exc}")
        # Can't validate, but the harness was generated — try it anyway.

    # Append --correctness so GEAK's SaveAndTest validator (runs test_command verbatim) can execute it.
    test_command = f"python {harness_path} --correctness"
    return SimpleNamespace(harness_path=str(harness_path), test_command=test_command)
