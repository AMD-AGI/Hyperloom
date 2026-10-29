# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Where a decode step's device time goes, by model module, read straight off the raw capture.

A GLM-5.3 VERIFY step is one CUDA-graph replay that TraceLens cannot decompose, yet the raw ROCm trace records
each of its ~2,000 kernels with a duration inside the step's annotation. Summing them per module (MoE, dense GEMM,
attention, the DSA indexer, collectives, quantisation) gives the table a specialist needs to pick a target, without
TraceLens. Profiling stretches each kernel by under a microsecond, so absolute milliseconds read high by about a
tenth while the ranking holds.
"""

from __future__ import annotations

import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple

from kernelforge.serving_windows import StepWindow, _kernels_in, _short, step_windows
from kernelforge.trace_events import load_trace_events, resolve_capture

#: First match wins, so a name that mentions two modules lands in the earlier one (an allreduce fused with an
#: RMSNorm is a collective; a MoE quant is MoE). Checked on GLM-5.3 VERIFY: 99.96% of kernel time is classified.
MODULE_RULES: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (module, re.compile(pattern))
    for module, pattern in (
        (
            "allreduce/comm",
            r"allreduce|all_reduce|cross_device|local_device_load|ranksignals|reduce_scatter|"
            r"allgather|all_gather|rccl|nccl|quickreduce|mscclpp|custom_ar",
        ),
        ("moe", r"moe|expert|topk_softmax|grouped_topk|_routing"),
        ("indexer/topk", r"indexer|dsa_topk|mqa_logits|hadamard|act_quant_kernel|kn_entry_2c_sbhd_cached"),
        ("attention/mla", r"mla|attn|attention|fmha|flash|batched_gemm_a8w8"),
        ("norm", r"rmsnorm|rms_norm|layernorm|layer_norm"),
        ("gemm", r"cijk_|tensile|gemm|matmul|_bhs_|gemv|hipblaslt|wvsplitk"),
        ("quant", r"quant|cvt"),
        ("sampling/spec", r"sample|argmax|verify|eagle|accept|tree|softmax|multinomial|gumbel"),
        ("elementwise/copy", r"elementwise|copy|memcpy|fill|index|gather|scatter|cat|arange|add|mul|vectorized"),
    )
)
_OTHER = "other"
_US_PER_MS = 1000.0
#: A text token names a capture kernel when it equals the kernel's name, or is this long and inside the name;
#: shorter fragments (``fill``, ``max_num_seqs``) are too generic to identify one.
_SPECIFIC_CHARS = 16
_TOKEN = re.compile(r"\w+")
#: Decode classes shown side by side; the most-stepped ones when a capture holds more widths.
_MAX_COLUMNS = 6


def module_of(kernel_name: str) -> str:
    """The model module a kernel belongs to, by its name."""
    lowered = kernel_name.lower()
    return next((module for module, pattern in MODULE_RULES if pattern.search(lowered)), _OTHER)


class _Capture(NamedTuple):
    table: dict[str, dict[str, Any]]
    #: Each decode kernel's lowercased name and its module.
    kernels: dict[str, str]


def decode_modules(trace_path: str | Path | None) -> dict[str, dict[str, Any]]:
    """Per decode window class, its device time per step by module, largest first; empty without decode steps."""
    return _decode_capture(trace_path).table


def _decode_capture(trace_path: str | Path | None) -> _Capture:
    if not str(trace_path or "").strip():
        return _Capture({}, {})
    events = load_trace_events(trace_path)
    by_class: dict[str, list[StepWindow]] = defaultdict(list)
    for window in step_windows(events):
        if window.is_decode:
            by_class[window.window_class].append(window)
    table: dict[str, dict[str, Any]] = {}
    kernels: dict[str, str] = {}
    for window_class in sorted(by_class, key=_class_order):
        members = by_class[window_class]
        totals = _kernels_in(events, members)
        kernels.update({name.lower(): module_of(name) for name in totals})
        rows = _module_rows(totals, steps=len(members))
        if rows:
            table[window_class] = {"steps": len(members), "modules": rows}
    return _Capture(table, kernels)


def _class_order(window_class: str) -> tuple[str, int]:
    kind, _, width = window_class.partition(" bs=")
    return kind, int(width) if width.isdigit() else 0


def _module_rows(totals: dict[str, float], *, steps: int) -> list[dict[str, Any]]:
    """Kernel microseconds summed per module, as ms per step and share of the class, with each module's top kernel."""
    module_us: dict[str, float] = defaultdict(float)
    top: dict[str, tuple[float, str]] = {}
    for name, us in totals.items():
        module = module_of(name)
        module_us[module] += us
        if us > top.get(module, (0.0, ""))[0]:
            top[module] = (us, name)
    total = sum(module_us.values())
    if not total:
        return []
    return [
        {
            "module": module,
            "ms_per_step": round(us / steps / _US_PER_MS, 3),
            "pct": round(us / total * 100.0, 1),
            "top_kernel": top[module][1],
        }
        for module, us in sorted(module_us.items(), key=lambda kv: kv[1], reverse=True)
    ]


def decode_module_lines(trace: Any, *, module: str = "") -> list[str]:
    """The rendered table for a profile's capture (a file or the directory of its ranks); empty when unreadable.

    A *module* keeps that module's row alone, empty when no decode class has it.
    """
    table = _capture(trace).table
    if module:
        table = {
            window_class: {**entry, "modules": [row for row in entry["modules"] if row["module"] == module]}
            for window_class, entry in table.items()
        }
    return render_decode_modules(table)


def decode_module_ranking(trace: Any) -> list[str]:
    """The modules of a capture's decode table in its row order, the most time per step first."""
    return _by_time(_grid(_capture(trace).table)[1])


def named_modules(trace: Any, text: str) -> set[str]:
    """The modules of the capture's decode kernels that *text* names."""
    tokens = {token.lower() for token in _TOKEN.findall(text)}
    return {
        module
        for name, module in _capture(trace).kernels.items()
        if any(token == name or (len(token) >= _SPECIFIC_CHARS and token in name) for token in tokens)
    }


def _capture(trace: Any) -> _Capture:
    capture = resolve_capture(trace)
    return _parsed(capture, Path(capture).stat().st_mtime) if capture else _Capture({}, {})


@lru_cache(maxsize=4)
def _parsed(capture: str, mtime: float) -> _Capture:
    """Parsing a capture takes seconds; every specialist of a cycle reads the same one."""
    del mtime
    return _decode_capture(capture)


def render_decode_modules(table: dict[str, dict[str, Any]] | None) -> list[str]:
    """Render modules as rows and decode classes as columns, ``ms/step (share)`` in each cell."""
    classes, cells = _grid(table or {})
    if not cells:
        return []
    lines = [f"| module | {' | '.join(classes)} | top kernel |", f"|---|{'---:|' * len(classes)}---|"]
    for module in _by_time(cells):
        row = cells[module]
        values = [f"{row[c]['ms_per_step']:.3f} ms ({row[c]['pct']:.1f}%)" if c in row else "—" for c in classes]
        heaviest = max(row.values(), key=lambda r: r["ms_per_step"])
        lines.append(f"| {module} | {' | '.join(values)} | `{_short(heaviest['top_kernel'])}` |")
    return lines


def _grid(table: dict[str, dict[str, Any]]) -> tuple[list[str], dict[str, dict[str, dict[str, Any]]]]:
    """The shown decode classes and each module's row per shown class."""
    classes = sorted(sorted(table, key=lambda c: table[c]["steps"], reverse=True)[:_MAX_COLUMNS], key=_class_order)
    cells: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for window_class in classes:
        for row in table[window_class]["modules"]:
            cells[row["module"]][window_class] = row
    return classes, cells


def _by_time(cells: dict[str, dict[str, dict[str, Any]]]) -> list[str]:
    return sorted(cells, key=lambda m: sum(r["ms_per_step"] for r in cells[m].values()), reverse=True)


__all__ = [
    "MODULE_RULES",
    "decode_module_lines",
    "decode_module_ranking",
    "decode_modules",
    "module_of",
    "named_modules",
    "render_decode_modules",
]
