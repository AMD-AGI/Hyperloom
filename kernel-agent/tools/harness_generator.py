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
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Callable


# Data classes

@dataclass
class FuncInfo:
    name: str
    params: list[str]
    source: str
    decorator: str
    lineno: int


@dataclass
class TensorInfo:
    var_name: str
    creation_expr: str
    shape_args: list[str]
    dtype_expr: str | None


@dataclass
class CallInfo:
    func_name: str
    args: list[str]
    kwargs: dict[str, str]


# BenchmarkAnalyzer — generic AST-based benchmark file analyzer

class BenchmarkAnalyzer:
    """Analyze a benchmark Python file to extract structure for harness generation."""

    PERF_DECORATORS = {"perftest", "benchmark"}
    REF_HINTS = {"torch", "ref", "native", "baseline", "reference", "gold"}
    KERNEL_HINTS = {"ck", "hip", "triton", "kernel", "optimized", "custom", "fused"}

    def __init__(self, source: str, source_file_module: str = ""):
        self.source = source
        self.lines = source.splitlines()
        self.tree = ast.parse(source)
        self.source_module = source_file_module

    def get_imports(self) -> list[str]:
        """Return original source lines for all import statements."""
        import_lines: list[str] = []
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                start = node.lineno - 1
                end = node.end_lineno or node.lineno
                import_lines.append("\n".join(self.lines[start:end]))
        return import_lines

    def get_decorated_functions(self) -> dict[str, FuncInfo]:
        """Find all functions decorated with @perftest or @benchmark."""
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
            body_lower = fi.source.lower()

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
            elif has_torch_functional:
                ref_candidates.append(fi)
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
        """Extract torch.randn/empty/zeros/ones calls from a function."""
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
        """Find a call to callee_name within func's body and extract its args."""
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
        """Get top-level assignment statements (e.g. torch.set_default_device)."""
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
    return (
        "[\n    (1, 4096, torch.bfloat16),\n    (32, 4096, torch.bfloat16),\n"
        "    (256, 4096, torch.bfloat16),\n    (1024, 4096, torch.bfloat16),\n"
        "    (4096, 4096, torch.bfloat16),\n    (8192, 4096, torch.bfloat16),\n]",
        "M, N, dtype = cfg",
        'f"M={M} N={N} {dtype}"',
    )


def _parse_shape_string(s: str) -> tuple[tuple[int, ...], str]:
    """Parse '(256, 128) bf16' → ((256, 128), 'bf16')."""
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
    """Generate the setup_inputs(cfg) body, creating inputs only for args the test actually passes."""
    dim_vars = cfg_unpack.replace(" = cfg", "").split(", ")
    dim_vars = [v.strip() for v in dim_vars if v.strip() != "dtype"]

    lines = [f"    {cfg_unpack}"]
    lines.append("    torch.manual_seed(42)")

    target_func = kernel_func or ref_func
    if not target_func:
        shape = ", ".join(dim_vars)
        lines.append(f'    x = torch.randn({shape}, dtype=dtype, device="cuda")')
        lines.append('    return {"x": x}')
        return "\n".join(lines)

    call = None
    if test_func:
        call = analyzer.extract_call_to(test_func, target_func.name)

    params = [p for p in target_func.params if p != "self"]

    if call:
        used_params = _match_call_args_to_params(call, params)
    else:
        used_params = [(p, None) for p in params]

    shape_2d = ", ".join(dim_vars[:2]) if len(dim_vars) >= 2 else dim_vars[0]
    shape_1d = dim_vars[-1] if dim_vars else "N"

    inputs_items: list[str] = []
    for param_name, call_value in used_params:
        p_lower = param_name.lower()

        # A literal call arg is stored directly.
        if call_value and not _is_variable(call_value):
            lines.append(f"    {param_name} = {call_value}")
            inputs_items.append(f'"{param_name}": {param_name}')
            continue

        if _is_scalar_param(p_lower):
            val = "1e-06" if "eps" in p_lower else "0"
            lines.append(f"    {param_name} = {val}")
            inputs_items.append(f'"{param_name}": {param_name}')
        elif _is_weight_param(p_lower):
            lines.append(f'    {param_name} = torch.randn({shape_1d}, dtype=dtype, device="cuda")')
            inputs_items.append(f'"{param_name}": {param_name}')
        else:
            lines.append(f'    {param_name} = torch.randn({shape_2d}, dtype=dtype, device="cuda")')
            inputs_items.append(f'"{param_name}": {param_name}')

    if not inputs_items:
        lines.append(f'    x = torch.randn({shape_2d}, dtype=dtype, device="cuda")')
        inputs_items.append('"x": x')

    lines.append("    return {" + ", ".join(inputs_items) + "}")
    return "\n".join(lines)


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
    SCALAR_EXACT = {"eps", "epsilon", "p", "dropout", "model_sensitive"}
    SCALAR_CONTAINS = {"use_model_sensitive"}
    return name in SCALAR_EXACT or any(h in name for h in SCALAR_CONTAINS)


def _is_weight_param(name: str) -> bool:
    WEIGHT_EXACT = {"weight", "w", "gamma", "bias", "beta"}
    return name in WEIGHT_EXACT or any(name.endswith(f"_{h}") for h in WEIGHT_EXACT)


def _generate_run_kernel(
    analyzer: BenchmarkAnalyzer,
    test_func: FuncInfo | None,
    kernel_func: FuncInfo | None,
) -> str:
    """Generate the run_kernel(inputs) function body."""
    if not kernel_func:
        return '    return inputs.get("x", list(inputs.values())[0])'

    return _generate_run_func_body(analyzer, test_func, kernel_func)


def _generate_run_ref(
    analyzer: BenchmarkAnalyzer,
    test_func: FuncInfo | None,
    ref_func: FuncInfo | None,
    kernel_func: FuncInfo | None,
) -> str:
    """Generate the run_ref(inputs) function body."""
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
        args_parts: list[str] = []
        for param_name, call_value in used:
            if call_value and not _is_variable(call_value):
                args_parts.append(f'inputs["{param_name}"]')
            else:
                args_parts.append(f'inputs["{param_name}"]')
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
    """Check if a string looks like a Python variable name (not a literal)."""
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

def maybe_generate_harness(
    benchmark_file: str,
    candidate: dict,
    source_file: str,
    out_dir: "Path",
    kernel_repo: str = "",
    log_fn: Callable[[str], None] | None = None,
) -> SimpleNamespace | None:
    """Generate a GEAK-compatible harness from a benchmark file.

    Returns SimpleNamespace(harness_path, test_command) on success, None on failure.
    """
    from pathlib import Path as _Path
    out_dir = _Path(out_dir)

    def _log(msg: str) -> None:
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
        _log("could not identify kernel or reference function")
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
        pass

    # Append --correctness so GEAK's SaveAndTest validator (runs test_command verbatim) can execute it.
    test_command = f"python {harness_path} --correctness"
    return SimpleNamespace(harness_path=str(harness_path), test_command=test_command)
