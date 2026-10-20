# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Render what Hyperloom hands KernelForge for a SIKL schema: its scored shapes, SIKL's rules, and who measures."""

from __future__ import annotations

from pathlib import Path

from hyperloom.common.io import atomic_write_text
from hyperloom.sikl.report import SERVING_SPEEDUP
from hyperloom.sikl.schema import Schema
from hyperloom.sikl.workspace import KERNEL_DECORATORS

SCHEMA_FILENAME = "schema.md"


def _axes_line(schema: Schema) -> str:
    constants = ", ".join(f"{name}={value}" for name, value in schema.constant_axes.items())
    variables = ", ".join(schema.variable_axes)
    return f"constant `{constants or 'none'}`, variable `{variables or 'none'}`"


def _workload_table(schema: Schema) -> list[str]:
    axis_names = list(schema.variable_axes)
    header = "| case | " + " | ".join(f"`{name}`" for name in axis_names) + " |"
    rule = "|---" * (len(axis_names) + 1) + "|"
    rows = [
        f"| `{workload.case_id}` | " + " | ".join(str(workload.axes.get(name, "")) for name in axis_names) + " |"
        for workload in schema.workloads
    ]
    return [header, rule, *rows]


def build_schema_md(schema: Schema) -> str:
    """The workload document: what is scored, and under whose rules."""
    lines = [
        f"# Schema: {schema.name}",
        "",
        f"- **Operator:** `{schema.op_type}`",
        f"- **Axes:** {_axes_line(schema)}",
        f"- **Attributes:** `{schema.attrs}`",
        "- **Inputs:** " + ", ".join(f"`{name}`" for name in schema.inputs),
        "- **Outputs:** " + ", ".join(f"`{name}`" for name in schema.outputs),
        f"- **Comparison:** `{schema.compare}`",
        f"- **Target:** `{schema.targets}`",
        f"- **Scored workloads:** {len(schema.workloads)}",
        "",
        "SIKL serves a kernel workload by workload: where its measured speedup over the baseline",
        f"reaches {SERVING_SPEEDUP}x and its output is correct. A workload it loses still runs the",
        "baseline, but a wrong output on any workload rejects the kernel.",
        "",
        *_workload_table(schema),
        "",
        "## Measurement ownership",
        "",
        "Correctness and timing come from `python -m sikl validate`, the check SIKL's CI runs:",
        "its audit of the source, its float64 reference and error bounds, and its graph-replay",
        "timing against the baseline in the same window. The driver reports what it returns",
        "and measures nothing of its own.",
        "",
    ]
    if schema.description:
        lines += ["## Upstream description", "", schema.description.strip(), ""]
    return "\n".join(lines)


def build_program_md(schema: Schema, *, kernel_backend: str, sikl_root: Path) -> str:
    """The brief the implementer agent reads."""
    constants = ", ".join(f"{name}={value}" for name, value in schema.constant_axes.items())
    return "\n".join(
        [
            f"# Program: implement {schema.name} in {kernel_backend}",
            "",
            f"**Backend**: {kernel_backend}",
            f"**Target**: {schema.targets}",
            "",
            "## Objective",
            "",
            "Make `run` in `kernel.py` faster than SIKL's baseline on as many of the",
            f"{len(schema.workloads)} scored workloads as possible while keeping every one of them",
            "correct. The loop runs SIKL's correctness checks before it benchmarks, so a change",
            "that wins on time and loses a workload is rejected, not traded off.",
            "",
            "## Contract",
            "",
            f"- Fixed axes: `{constants or 'none'}`",
            f"- Varying axes: `{', '.join(schema.variable_axes) or 'none'}` — see `handoff/{SCHEMA_FILENAME}`",
            "- `run` takes SIKL's call keys as keyword arguments and writes each output into the",
            "  preallocated output argument; it returns nothing and never allocates an output.",
            f"- SIKL compares with `{schema.compare}`. A `bitwise` determinism rule rejects any",
            "  reduction whose order varies between runs, atomics across workgroups included.",
            f"- SIKL audits the source before it reaches a GPU. Its rules are in `{sikl_root}/sikl/ops/README.md`",
            "  (§ The kernel section and ABI): imports only from torch, flydsl, triton and a few standard",
            "  modules; scratch memory only from `torch.empty` inside the call; no `.to`, `.contiguous` or",
            "  `.clone` in launch code; compiled code cached only through a module-level `@functools.cache`.",
            f"- Every GPU kernel `run` launches is one `kernel.py` defines with `@{KERNEL_DECORATORS[kernel_backend]}`.",
            "  The correctness pass profiles one call per workload and fails a workload that launches",
            "  anything else: a vendor library such as hipBLASLt, an aiter kernel, or one of torch's",
            "  own kernels, a cast or a copy included.",
            "",
            "## What the starting point is",
            "",
            "`kernel.py` starts from an implementation SIKL passes on every workload, so the first",
            "measurement anchors on a working kernel. It is the one kernel exempt from the rule above:",
            f"replace its body with {kernel_backend} kernels.",
            "",
            "## Where the numbers come from",
            "",
            "`driver.py` runs `python -m sikl validate` on the kernel and reports SIKL's per-workload",
            "verdict and graph time. It is the measurement boundary and is not editable; a change that",
            "only moves the reported number is a change to nothing.",
            "",
        ]
    )


def write(schema: Schema, handoff_dir: Path, program_md: Path, *, kernel_backend: str, sikl_root: Path) -> Path:
    """Write the schema document and the implementer's brief, one copy each."""
    atomic_write_text(
        Path(handoff_dir) / SCHEMA_FILENAME,
        build_schema_md(schema),
        make_parents=True,
        fsync=True,
        fsync_dir=True,
    )
    atomic_write_text(
        Path(program_md),
        build_program_md(schema, kernel_backend=kernel_backend, sikl_root=sikl_root),
        make_parents=True,
        fsync=True,
        fsync_dir=True,
    )
    return Path(handoff_dir)
