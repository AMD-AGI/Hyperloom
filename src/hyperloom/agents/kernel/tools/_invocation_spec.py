# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build a durable, backend-neutral kernel invocation evidence contract."""

from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path
from typing import Any

from _io_utils import atomic_write_json
from _task_group_contract import (
    CASE_SELECTOR_KEY,
    logical_operator_name,
    task_group_shape_cases,
)


SCHEMA_VERSION = 2
_SHAPE_RE = re.compile(r"\(([^()]*)\)\s*([A-Za-z0-9_:.\-]+)?")
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_SOURCE_LOCATOR_RE = re.compile(
    r"^(.*?\.(?:py|pyi|c|cc|cpp|cu|cuh|h|hpp|hip|so))(?:\(\d+\))?(?::.*)?$",
    re.IGNORECASE,
)
_REFERENCE_NAME_HINTS = ("torch", "ref", "reference", "baseline", "gold", "native")
_KERNEL_NAME_HINTS = ("kernel", "gemm", "matmul", "attention", "fused", "custom", "triton", "hip", "ck")
_SENSITIVE_KEY_RE = re.compile(r"(?:token|secret|password|api[_-]?key|authorization)", re.IGNORECASE)
_PATH_KEY_RE = re.compile(r"(?:path|file|dir|root|config)$", re.IGNORECASE)
_MAX_MODEL_CONFIG_BYTES = 1_048_576
_MAX_BENCHMARK_REPORT_BYTES = 16 * 1024 * 1024
_MODEL_EXECUTION_KEYS = (
    "architectures",
    "model_type",
    "torch_dtype",
    "hidden_size",
    "intermediate_size",
    "head_dim",
    "num_attention_heads",
    "num_key_value_heads",
    "num_hidden_layers",
    "max_position_embeddings",
    "sliding_window",
    "use_sliding_window",
    "rope_scaling",
    "rope_theta",
    "quantization_config",
    "num_experts",
    "num_local_experts",
    "num_experts_per_tok",
)
_OMIT = object()
_SOURCE_FRAMEWORK_ALIASES = {
    "vllm": "vllm",
    "sglang": "sglang",
    "aiter": "aiter",
    "aiter_meta": "aiter",
}


def _normalize_dtype(value: Any) -> str:
    """Normalize common profiler/PyTorch dtype spellings without inventing one."""
    raw = str(value or "").strip()
    low = raw.lower()
    if not low:
        return ""
    if "float8" in low or "fp8" in low:
        return "fp8"
    if "bfloat16" in low or low == "bf16":
        return "bf16"
    if "float16" in low or low in {"fp16", "half", "c10::half"}:
        return "fp16"
    if "float32" in low or low in {"fp32", "float", "c10::float"}:
        return "fp32"
    if "float64" in low or low in {"fp64", "double", "c10::double"}:
        return "fp64"
    if "int64" in low or low in {"long", "long int", "c10::long"}:
        return "int64"
    if "int32" in low or low in {"int", "c10::int"}:
        return "int32"
    if "uint8" in low or "unsigned char" in low:
        return "uint8"
    if "bool" in low:
        return "bool"
    return raw


def _parse_dims(raw: str) -> list[int] | None:
    """Parse one comma-separated dimension group; preserve scalar ``()`` as []."""
    text = raw.strip()
    if not text:
        return []
    dims: list[int] = []
    for part in text.split(","):
        token = part.strip()
        if not token:
            continue
        try:
            dims.append(int(token))
        except ValueError:
            return None
    return dims


def _absolute_path(value: Any, *, base_dir: str = "") -> str:
    """Return an absolute filesystem path, or an empty string on bad input."""
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        path = Path(text).expanduser()
        if not path.is_absolute():
            base = Path(base_dir).expanduser() if base_dir else Path.cwd()
            path = base / path
        return str(path.resolve(strict=False))
    except (OSError, RuntimeError, ValueError):
        return ""


def _absolute_existing_path(value: Any, *, base_dir: str = "") -> str:
    """Resolve a path only when it identifies an existing filesystem entry."""
    absolute = _absolute_path(value, base_dir=base_dir)
    if not absolute:
        return ""
    try:
        return absolute if Path(absolute).exists() else ""
    except OSError:
        return ""


def _path_from_source_locator(value: Any, *, base_dir: str = "") -> str:
    """Extract and absolutize the file portion of ``path(line): symbol``."""
    text = str(value or "").strip()
    if not text:
        return ""
    match = _SOURCE_LOCATOR_RE.match(text)
    path_text = match.group(1) if match else text
    return _absolute_path(path_text, base_dir=base_dir)


def _absolute_paths(values: Any, *, base_dir: str = "") -> list[str]:
    rows = values if isinstance(values, list) else [values] if isinstance(values, str) else []
    resolved: list[str] = []
    for value in rows:
        if not isinstance(value, (str, Path)):
            continue
        path = _absolute_path(value, base_dir=base_dir)
        if path and path not in resolved:
            resolved.append(path)
    return resolved


def _is_directory(path: str) -> bool:
    try:
        return bool(path) and Path(path).is_dir()
    except OSError:
        return False


def _argument_records(
    entries: Any,
    fallback_dtypes: Any,
    *,
    root: str,
) -> list[dict[str, Any]]:
    """Convert candidate shape rows into ordered argument/output records."""
    rows = entries if isinstance(entries, list) else []
    dtypes = fallback_dtypes if isinstance(fallback_dtypes, list) else []
    records: list[dict[str, Any]] = []
    logical_index = 0

    for row_index, entry in enumerate(rows):
        call_num = None
        explicit_dtype = ""
        raw_shape: Any = entry
        if isinstance(entry, dict):
            raw_shape = entry.get("shape")
            call_num = entry.get("call_num")
            explicit_dtype = str(entry.get("dtype") or "")

        parsed: list[tuple[list[int] | None, str, str]] = []
        if isinstance(raw_shape, (list, tuple)) and all(isinstance(x, int) for x in raw_shape):
            parsed.append(([int(x) for x in raw_shape], explicit_dtype, str(raw_shape)))
        elif isinstance(raw_shape, str):
            matches = list(_SHAPE_RE.finditer(raw_shape))
            if matches:
                for match in matches:
                    parsed.append(
                        (
                            _parse_dims(match.group(1)),
                            explicit_dtype or str(match.group(2) or ""),
                            match.group(0).strip(),
                        )
                    )
            elif raw_shape.strip():
                parsed.append((None, explicit_dtype, raw_shape.strip()))
        elif raw_shape is not None:
            parsed.append((None, explicit_dtype, str(raw_shape)))

        for dims, inline_dtype, raw in parsed:
            fallback = dtypes[logical_index] if logical_index < len(dtypes) else ""
            dtype_raw = inline_dtype or str(fallback or "")
            record: dict[str, Any] = {
                "path": f"{root}[{logical_index}]",
                "position": logical_index,
                "shape": dims,
                "dtype": _normalize_dtype(dtype_raw),
                "dtype_raw": dtype_raw,
                "raw": raw,
                "source_row": row_index,
            }
            if call_num not in (None, ""):
                record["call_count"] = call_num
            records.append(record)
            logical_index += 1

    while logical_index < len(dtypes):
        dtype_raw = str(dtypes[logical_index] or "")
        records.append(
            {
                "path": f"{root}[{logical_index}]",
                "position": logical_index,
                "shape": None,
                "dtype": _normalize_dtype(dtype_raw),
                "dtype_raw": dtype_raw,
                "raw": "",
                "source_row": None,
            }
        )
        logical_index += 1
    return records


def _decorator_name(node: ast.expr) -> str:
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _qualified_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _qualified_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _call_targets(node: ast.AST) -> list[str]:
    targets: list[str] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        name = _qualified_name(child.func)
        if name and name not in targets and name not in {"perftest", "benchmark"}:
            targets.append(name)
    return targets


def _analyze_benchmark(path: Path) -> dict[str, Any]:
    """Extract function-role and call-target evidence from one Python benchmark."""
    evidence: dict[str, Any] = {"path": str(path)}
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
    except (OSError, SyntaxError) as exc:
        evidence["analysis_error"] = f"{type(exc).__name__}: {exc}"
        return evidence

    functions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    perf_functions: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    benchmark_functions: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for func in functions:
        decorators = {_decorator_name(dec) for dec in func.decorator_list}
        if "perftest" in decorators:
            perf_functions.append(func)
        if "benchmark" in decorators:
            benchmark_functions.append(func)

    reference = next(
        (func for func in perf_functions if any(hint in func.name.lower() for hint in _REFERENCE_NAME_HINTS)),
        None,
    )
    kernel = next(
        (
            func
            for func in perf_functions
            if func is not reference and any(hint in func.name.lower() for hint in _KERNEL_NAME_HINTS)
        ),
        None,
    )
    if kernel is None:
        kernel = next((func for func in perf_functions if func is not reference), None)
    test = (
        benchmark_functions[0]
        if benchmark_functions
        else next(
            (func for func in functions if func.name.startswith(("test_", "bench_"))),
            None,
        )
    )

    evidence["reference_function"] = reference.name if reference else ""
    evidence["kernel_function"] = kernel.name if kernel else ""
    evidence["test_function"] = test.name if test else ""
    evidence["reference_call_targets"] = _call_targets(reference) if reference else []
    evidence["kernel_call_targets"] = _call_targets(kernel) if kernel else []
    evidence["test_call_targets"] = _call_targets(test) if test else []
    return evidence


def _benchmark_evidence(
    candidate: dict[str, Any],
    *,
    base_dir: str = "",
) -> tuple[list[str], list[dict[str, Any]]]:
    raw = candidate.get("benchmark_files") or []
    if isinstance(raw, str):
        raw = [raw]
    files = _absolute_paths(raw, base_dir=base_dir)
    evidence: list[dict[str, Any]] = []
    for path in files[:8]:
        try:
            if Path(path).is_file():
                evidence.append(_analyze_benchmark(Path(path)))
        except OSError:
            continue
    return files, evidence


def _primary_benchmark_evidence(evidence: list[dict[str, Any]]) -> dict[str, Any]:
    """Select and compact the benchmark evidence most useful for driver authoring."""
    primary = next(
        (row for row in evidence if row.get("kernel_function") and row.get("kernel_call_targets")),
        None,
    )
    if primary is None:
        primary = next((row for row in evidence if row.get("test_function")), None)
    if primary is None:
        return {}

    compact: dict[str, Any] = {"path": str(primary.get("path") or "")}
    for key in ("kernel_function", "reference_function", "test_function"):
        value = str(primary.get(key) or "")
        if value:
            compact[key] = value

    public_targets = [
        str(target)
        for target in (primary.get("kernel_call_targets") or [])
        if "." in str(target) and not str(target).startswith(("torch.", "F.", "triton."))
    ]
    reference_targets = [
        str(target)
        for target in (primary.get("reference_call_targets") or [])
        if str(target).startswith(("torch.", "F."))
    ]
    if public_targets:
        compact["public_call_targets"] = public_targets
    if reference_targets:
        compact["reference_call_targets"] = reference_targets
    return compact


def _source_symbol(source_file: str, raw_symbols: list[str]) -> str:
    """Resolve a complete editable Python function from a truncated runtime name."""
    if not source_file or not source_file.endswith((".py", ".pyi")):
        return ""
    try:
        tree = ast.parse(Path(source_file).read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError):
        return ""
    function_names = [node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    prefixes = [symbol.removesuffix("...") for symbol in raw_symbols if symbol]
    matches = [
        name
        for name in function_names
        if any(prefix.startswith(name) or name.startswith(prefix) for prefix in prefixes)
    ]
    return max(matches, key=len) if matches else ""


def _benchmark_report_from_trace(candidate: dict[str, Any]) -> Path | None:
    """Locate the profile benchmark report through TraceLens' input manifest."""
    trace_report = str(candidate.get("trace_report_path") or "").strip()
    if not trace_report:
        return None
    try:
        analysis_path = Path(trace_report).expanduser().resolve()
        manifest_path = analysis_path.parent.parent / "trace_input_manifest.json"
        if not manifest_path.is_file():
            return None
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        trace_input = str(manifest.get("trace_input") or "").strip() if isinstance(manifest, dict) else ""
        if not trace_input:
            return None
        report = Path(trace_input).expanduser().resolve().parent / "benchmark_report.json"
        if report.is_file() and report.stat().st_size <= _MAX_BENCHMARK_REPORT_BYTES:
            return report
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
        return None
    return None


def _runtime_symbols(candidate: dict[str, Any], raw_symbols: list[str]) -> list[str]:
    """Recover complete runtime symbols from the original benchmark report."""
    complete = [symbol for symbol in raw_symbols if symbol and not symbol.endswith("...")]
    truncated_prefixes = [symbol.removesuffix("...") for symbol in raw_symbols if symbol.endswith("...")]
    if not truncated_prefixes:
        return list(dict.fromkeys(complete))

    report = _benchmark_report_from_trace(candidate)
    if report is None:
        return list(dict.fromkeys(complete))
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return list(dict.fromkeys(complete))
    if not isinstance(payload, dict):
        return list(dict.fromkeys(complete))

    report_symbols: list[str] = []
    for row in payload.get("kernel_summary") or []:
        if isinstance(row, dict) and row.get("name"):
            report_symbols.append(str(row["name"]))
    for symbol in payload.get("top_bottlenecks") or []:
        if symbol:
            report_symbols.append(str(symbol))

    for symbol in report_symbols:
        if any(symbol.startswith(prefix) for prefix in truncated_prefixes):
            complete.append(symbol)
    return list(dict.fromkeys(complete))


def _safe_json_value(value: Any, *, key: str = "", base_dir: str = "") -> Any:
    """Recursively sanitize JSON evidence and absolutize explicit path fields."""
    if isinstance(value, dict):
        return {
            str(child_key): _safe_json_value(
                child_value,
                key=str(child_key),
                base_dir=base_dir,
            )
            for child_key, child_value in value.items()
            if not _SENSITIVE_KEY_RE.search(str(child_key))
        }
    if isinstance(value, list):
        return [_safe_json_value(item, base_dir=base_dir) for item in value]
    if isinstance(value, tuple):
        return [_safe_json_value(item, base_dir=base_dir) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str) and _PATH_KEY_RE.search(key):
            return _absolute_path(value, base_dir=base_dir)
        return value
    return str(value)


def _safe_mapping(value: Any, *, base_dir: str = "") -> dict[str, Any]:
    """Copy a JSON mapping while dropping secrets and normalizing path fields."""
    if not isinstance(value, dict):
        return {}
    sanitized = _safe_json_value(value, base_dir=base_dir)
    return sanitized if isinstance(sanitized, dict) else {}


def _load_model_config(model_path: str) -> tuple[str, dict[str, Any]]:
    """Read a bounded local model config and return its path and tuning summary."""
    if not model_path:
        return "", {}
    try:
        config_path = Path(model_path) / "config.json"
        if not config_path.is_file() or config_path.stat().st_size > _MAX_MODEL_CONFIG_BYTES:
            return "", {}
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return "", {}
        safe_payload = _safe_mapping(payload, base_dir=model_path)
        summary = {key: safe_payload[key] for key in _MODEL_EXECUTION_KEYS if key in safe_payload}
        return str(config_path.resolve(strict=False)), summary
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
        return "", {}


def _numeric_context_value(value: Any) -> int | float | None:
    """Return a numeric deployment value, omitting unknown symbolic text."""
    if isinstance(value, bool) or value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    try:
        text = str(value).strip()
        parsed = float(text) if "." in text else int(text)
        if isinstance(parsed, float) and not math.isfinite(parsed):
            return None
        return parsed
    except (TypeError, ValueError):
        return None


def _compact_json(value: Any, *, key: str = "") -> Any:
    """Remove non-actionable empty fields while preserving false/zero/scalar shapes."""
    if isinstance(value, dict):
        compacted: dict[str, Any] = {}
        for child_key, child_value in value.items():
            cleaned = _compact_json(child_value, key=str(child_key))
            if cleaned is not _OMIT:
                compacted[str(child_key)] = cleaned
        return compacted if compacted else _OMIT
    if isinstance(value, list):
        cleaned_items = [cleaned for item in value if (cleaned := _compact_json(item)) is not _OMIT]
        if cleaned_items or key == "shape":
            return cleaned_items
        return _OMIT
    if value is None or value == "":
        return _OMIT
    return value


def _deployment_context(candidate: dict[str, Any], *, repo_root: str = "") -> dict[str, Any]:
    """Build fail-soft serving, sequence, and model context for kernel tuning."""
    runtime_args = candidate.get("runtime_args") if isinstance(candidate.get("runtime_args"), dict) else {}
    workload = dict(runtime_args.get("workload")) if isinstance(runtime_args.get("workload"), dict) else {}
    for key in (
        "batch_size",
        "micro_batch_size",
        "conc",
        "num_prompts",
        "num_warmups",
        "tp",
        "ep",
        "isl",
        "osl",
        "max_model_len",
        "profile_osl",
    ):
        if key not in workload and runtime_args.get(key) not in (None, ""):
            workload[key] = runtime_args[key]

    batch: dict[str, Any] = {}
    batch_sources = (
        ("batch_size", "batch_size"),
        ("micro_batch_size", "micro_batch_size"),
        ("conc", "serving_concurrency"),
        ("num_prompts", "request_count"),
        ("num_warmups", "warmup_request_count"),
        ("tp", "tensor_parallel_size"),
        ("ep", "expert_parallel_size"),
    )
    for source_key, target_key in batch_sources:
        value = _numeric_context_value(workload.get(source_key))
        if value is not None:
            batch[target_key] = value

    sequence: dict[str, Any] = {}
    sequence_sources = (
        ("isl", "input_tokens"),
        ("osl", "output_tokens"),
        ("max_model_len", "max_model_len"),
        ("profile_osl", "profile_output_tokens"),
    )
    for source_key, target_key in sequence_sources:
        value = _numeric_context_value(workload.get(source_key))
        if value is not None:
            sequence[target_key] = value
    input_tokens = sequence.get("input_tokens")
    output_tokens = sequence.get("output_tokens")
    if input_tokens is not None and output_tokens is not None:
        try:
            sequence["request_tokens"] = int(sequence["input_tokens"]) + int(sequence["output_tokens"])
        except (TypeError, ValueError, OverflowError):
            sequence.pop("request_tokens", None)

    model_ref = str(runtime_args.get("model") or candidate.get("model_path") or "").strip()
    local_model_path = _absolute_existing_path(model_ref, base_dir=repo_root)
    model: dict[str, Any] = {}
    if _is_directory(local_model_path):
        model["model_path"] = local_model_path
        config_path, summary = _load_model_config(local_model_path)
        if config_path:
            model["config_path"] = config_path
        if summary:
            model["config_summary"] = summary
    elif model_ref:
        model["model_id"] = model_ref

    materialized_config = _absolute_path(runtime_args.get("materialized_config"), base_dir=repo_root)
    context: dict[str, Any] = {}
    if batch:
        context["batch"] = batch
    if sequence:
        context["sequence"] = sequence
    if model:
        context["model"] = model
    if materialized_config:
        context["materialized_runtime_config_path"] = materialized_config
    return context


def _task_group_contract(
    candidate: dict[str, Any],
    *,
    repo_root: str = "",
) -> dict[str, Any]:
    """Compact a TraceLens task group into actionable workload cases."""
    group = candidate.get("task_group")
    if not isinstance(group, dict):
        return {}
    rows_by_kernel_id = {
        str(row.get("kernel_id") or ""): row for row in (group.get("rows") or []) if isinstance(row, dict)
    }
    cases: list[dict[str, Any]] = []
    for normalized in task_group_shape_cases(candidate):
        kernel_ids = [str(kernel_id) for kernel_id in (normalized.get("kernel_ids") or []) if str(kernel_id)]
        source_row = next(
            (rows_by_kernel_id[kernel_id] for kernel_id in kernel_ids if kernel_id in rows_by_kernel_id),
            {},
        )
        arguments = _argument_records(
            normalized.get("input_shapes") or [],
            normalized.get("input_dtypes") or [],
            root="args",
        )
        case: dict[str, Any] = {
            "case_id": str(normalized.get("case_id") or ""),
            "selector": dict(normalized.get("selector") or {}),
            "kernel_ids": kernel_ids,
            "operation": str(normalized.get("operation") or ""),
            "source_file": _absolute_path(
                source_row.get("source_file"),
                base_dir=repo_root,
            ),
            "call_count": normalized.get("call_count"),
            "gpu_pct": source_row.get("gpu_pct"),
            "arguments": arguments,
            "outputs": _argument_records(
                normalized.get("output_shapes") or [],
                normalized.get("output_dtypes") or [],
                root="outputs",
            ),
            "raw_arg_spec": _safe_mapping(
                normalized.get("raw_arg_spec"),
                base_dir=repo_root,
            ),
        }
        compact_case = _compact_json(case)
        if isinstance(compact_case, dict):
            cases.append(compact_case)
    return {
        "task_group_id": str(group.get("task_group_id") or ""),
        "task_group_key": str(group.get("task_group_key") or ""),
        "primary_kernel_id": str(group.get("primary_kernel_id") or ""),
        "kernel_ids": [str(item) for item in (group.get("kernel_ids") or []) if str(item)],
        "aggregate_gpu_pct": group.get("aggregate_gpu_pct"),
        "aggregate_call_count": group.get("aggregate_call_count"),
        "cases": cases,
    }


def _driver_contract(task_group: dict[str, Any]) -> dict[str, Any]:
    """Describe the shape-selection behavior required from a Forge driver."""
    cases = task_group.get("cases")
    if not isinstance(cases, list) or not cases:
        return {}
    return {
        "shape_argument": "--shape",
        "case_selector_key": CASE_SELECTOR_KEY,
        "requires_all_cases": len(cases) > 1,
        "case_selectors": [
            dict(case.get("selector") or {})
            for case in cases
            if isinstance(case, dict) and isinstance(case.get("selector"), dict)
        ],
    }


def invocation_spec_filename(candidate: dict[str, Any]) -> str:
    """Return a path-safe, operator-identifying invocation-spec filename."""
    identity = str(candidate.get("name") or candidate.get("operation") or "unknown_operator").strip()
    safe_identity = _SAFE_FILENAME_RE.sub("_", identity).strip("._-")[:96]
    if not safe_identity:
        safe_identity = "unknown_operator"
    return f"invocation_spec_{safe_identity}.json"


def _logical_operator(candidate: dict[str, Any]) -> str:
    """Return the logical operator shared with the Forge handoff."""
    return logical_operator_name(candidate)


def _source_framework(candidate: dict[str, Any], sources: list[str]) -> str:
    """Resolve source ownership independently of the serving framework."""
    explicit = str(candidate.get("source_framework") or "").strip().lower()
    if explicit in _SOURCE_FRAMEWORK_ALIASES:
        return _SOURCE_FRAMEWORK_ALIASES[explicit]
    for source in sources:
        for component in Path(source).parts:
            framework = _SOURCE_FRAMEWORK_ALIASES.get(component.lower())
            if framework:
                return framework
    return ""


def _effective_kernel_kind(candidate: dict[str, Any]) -> str:
    """Return explicit kind or one proven by source classification."""
    explicit = str(candidate.get("kernel_kind") or "").strip().lower()
    if explicit:
        return explicit.replace("-", "_")
    source_type = str(candidate.get("source_type") or "").strip().lower()
    return source_type if source_type in {"triton", "flydsl", "ck"} else ""


def _source_level_symbols(source_files: list[str]) -> list[str]:
    """Parse stable editable definitions without using runtime-specialized names."""
    symbols: list[str] = []
    for source_file in source_files:
        path = Path(source_file)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if path.suffix.lower() in {".py", ".pyi"}:
            try:
                tree = ast.parse(text)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                decorators: list[str] = []
                for decorator in node.decorator_list:
                    target = decorator.func if isinstance(decorator, ast.Call) else decorator
                    if isinstance(target, ast.Name):
                        decorators.append(target.id)
                    elif isinstance(target, ast.Attribute):
                        decorators.append(target.attr)
                if "jit" in decorators and node.name not in symbols:
                    symbols.append(node.name)
            continue
        for match in re.finditer(
            r"\b(?:__global__|__device__)\b[^;{}]*?\b"
            r"([A-Za-z_][A-Za-z0-9_]*)\s*\(",
            text,
        ):
            symbol = match.group(1)
            if symbol not in symbols:
                symbols.append(symbol)
    return symbols


def build_invocation_spec(
    candidate: dict[str, Any],
    *,
    source_file: str = "",
) -> dict[str, Any]:
    """Build a versioned invocation evidence document from a TraceLens candidate."""
    raw_repo_root = str(candidate.get("kernel_repo") or "").strip()
    repo_root = _absolute_path(raw_repo_root) if raw_repo_root else ""
    inputs = _argument_records(
        candidate.get("input_shapes") or candidate.get("shapes") or [],
        candidate.get("input_dtypes") or candidate.get("dtypes") or [],
        root="args",
    )
    outputs = _argument_records(
        candidate.get("output_shapes") or [],
        candidate.get("output_dtypes") or [],
        root="outputs",
    )
    benchmark_files, benchmark_evidence = _benchmark_evidence(
        candidate,
        base_dir=repo_root,
    )
    primary_benchmark = _primary_benchmark_evidence(benchmark_evidence)

    callable_candidates: list[dict[str, str]] = []
    for evidence in benchmark_evidence:
        for target in evidence.get("kernel_call_targets") or []:
            if "." not in target or target.startswith(("torch.", "F.", "triton.")):
                continue
            row = {"target": target, "source": str(evidence.get("path") or "")}
            if row not in callable_candidates:
                callable_candidates.append(row)

    missing: list[str] = []
    if not callable_candidates:
        missing.append("public_callable")
    if not inputs:
        missing.append("inputs")
    if not outputs:
        missing.extend(("output_shapes", "output_dtypes"))

    source = _absolute_path(
        source_file or candidate.get("source_file"),
        base_dir=repo_root,
    )
    launcher_source = _path_from_source_locator(
        candidate.get("launcher_source_file"),
        base_dir=repo_root,
    )
    kernel_sources = _absolute_paths(
        candidate.get("kernel_sources") or [],
        base_dir=repo_root,
    )
    deployment = _deployment_context(candidate, repo_root=repo_root)
    observed_leading_dims = sorted(
        {
            int(record["shape"][0])
            for record in inputs
            if isinstance(record.get("shape"), list) and record["shape"] and isinstance(record["shape"][0], int)
        }
    )
    if observed_leading_dims:
        batch_context = deployment.setdefault("batch", {})
        if isinstance(batch_context, dict):
            batch_context["observed_tensor_leading_dimensions"] = observed_leading_dims
    raw_device_symbols = candidate.get("device_kernel_names") or []
    if isinstance(raw_device_symbols, str):
        raw_device_symbols = [raw_device_symbols]
    elif not isinstance(raw_device_symbols, list):
        raw_device_symbols = []
    raw_device_symbols = [str(item) for item in raw_device_symbols if str(item)]
    if candidate.get("device_kernel_name"):
        primary_raw_symbol = str(candidate["device_kernel_name"])
        if primary_raw_symbol not in raw_device_symbols:
            raw_device_symbols.insert(0, primary_raw_symbol)
    source_symbol = _source_symbol(source, raw_device_symbols)
    runtime_symbols = _runtime_symbols(candidate, raw_device_symbols)
    raw_target_symbols = candidate.get("target_functions") or []
    if isinstance(raw_target_symbols, str):
        raw_target_symbols = [value.strip() for value in raw_target_symbols.split(",") if value.strip()]
    elif not isinstance(raw_target_symbols, list):
        raw_target_symbols = []
    parsed_source_symbols = _source_level_symbols(list(dict.fromkeys([source, *kernel_sources])))
    curated_source_symbol = str(candidate.get("source_symbol") or "").strip()
    if not curated_source_symbol and not parsed_source_symbols:
        curated_source_symbol = source_symbol
    implementation_symbols = list(
        dict.fromkeys(
            symbol
            for symbol in (
                curated_source_symbol,
                *[str(item) for item in raw_target_symbols],
                *parsed_source_symbols,
            )
            if symbol and not symbol.endswith("...")
        )
    )
    unresolved_prefixes = [
        symbol.removesuffix("...")
        for symbol in raw_device_symbols
        if symbol.endswith("...") and not any(full.startswith(symbol.removesuffix("...")) for full in runtime_symbols)
    ]
    runtime_args = candidate.get("runtime_args") if isinstance(candidate.get("runtime_args"), dict) else {}
    runtime_flags = candidate.get("runtime_flags") if isinstance(candidate.get("runtime_flags"), dict) else {}
    task_group_contract = _task_group_contract(candidate, repo_root=repo_root)
    execution: dict[str, Any] = {}
    execution_fields = {
        "framework": candidate.get("framework") or candidate.get("backend"),
        "precision": runtime_args.get("precision"),
        "target_platform": runtime_flags.get("target_platform"),
        "is_multigpu": bool(candidate.get("is_multigpu")),
        "num_gpus_recommended": candidate.get("num_gpus_recommended"),
        "runtime_backend": candidate.get("runtime_backend"),
    }
    for key, value in execution_fields.items():
        if value not in (None, ""):
            execution[key] = value
    spec: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete" if not missing else "partial",
        "missing_fields": missing,
        "logical_operator": _logical_operator(candidate),
        "source_framework": _source_framework(
            candidate,
            [*kernel_sources, source],
        ),
        "implementation": {
            "sources": list(dict.fromkeys([source, *kernel_sources])),
            "kernel_kind": _effective_kernel_kind(candidate),
            "symbols": implementation_symbols,
            "runtime_backend": str(candidate.get("runtime_backend") or ""),
        },
        "kernel": {
            "kernel_id": str(candidate.get("kernel_id") or ""),
            "name": str(candidate.get("name") or ""),
            "operation": str(candidate.get("operation") or ""),
            "kernel_category": str(candidate.get("kernel_category") or ""),
            "tracelens_category": str(candidate.get("tracelens_category") or ""),
            "source_type": str(candidate.get("source_type") or ""),
            "kernel_kind": _effective_kernel_kind(candidate),
        },
        "edit_target": {
            "source_file": source,
            "repo_root": repo_root,
            "source_symbol": source_symbol,
            "runtime_symbols": runtime_symbols,
            "unresolved_runtime_symbol_prefixes": unresolved_prefixes,
            "kernel_sources": kernel_sources,
            "resolution_method": str(candidate.get("source_resolution_method") or ""),
        },
        "invocation": {
            "launcher_source_file": launcher_source,
            "launcher_locator": str(candidate.get("tracelens_launcher_path") or ""),
            "public_callable_candidates": callable_candidates,
            "arguments": inputs,
            "outputs": outputs,
            "kernel_contract": (
                candidate.get("kernel_contract") if isinstance(candidate.get("kernel_contract"), dict) else {}
            ),
        },
        "tests": {
            "primary_benchmark": primary_benchmark,
            "related_files": benchmark_files,
            "driver_contract": _driver_contract(task_group_contract),
        },
        "workload": {
            "call_count": candidate.get("call_count"),
            "task_group": task_group_contract,
        },
        "execution": execution,
        "deployment": deployment,
        "provenance": {
            "trace_report_path": _absolute_path(
                candidate.get("trace_report_path"),
                base_dir=repo_root,
            ),
            "shape_provenance": str(candidate.get("shape_provenance") or ""),
            "source_resolution_status": str(candidate.get("op_to_source_status") or ""),
            "source_resolution_reason": str(candidate.get("op_to_source_reason") or ""),
        },
    }
    raw_arg_spec = candidate.get("raw_arg_spec")
    if isinstance(raw_arg_spec, dict) and raw_arg_spec:
        spec["invocation"]["raw_arg_spec"] = _safe_mapping(
            raw_arg_spec,
            base_dir=repo_root,
        )
    compacted = _compact_json(spec)
    return compacted if isinstance(compacted, dict) else {}


def write_invocation_spec(path: Path, spec: dict[str, Any]) -> None:
    """Atomically persist an invocation spec as UTF-8 JSON."""
    atomic_write_json(path, spec, ensure_ascii=False)
