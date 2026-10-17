# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Render what Hyperloom hands KernelForge for a SIKL definition: its scored shapes, rule and measurement owner."""

from __future__ import annotations

from pathlib import Path

from hyperloom.common.io import atomic_write_text
from hyperloom.sikl.definition import Definition
from hyperloom.sikl.workspace import KERNEL_DECORATORS

DEFINITION_FILENAME = "definition.md"


def _axes_line(definition: Definition) -> str:
    constants = ", ".join(f"{name}={value}" for name, value in definition.constant_axes.items())
    variables = ", ".join(definition.variable_axes)
    return f"constant `{constants or 'none'}`, variable `{variables or 'none'}`"


def _workload_table(definition: Definition) -> list[str]:
    axis_names = list(definition.variable_axes)
    header = "| case | " + " | ".join(f"`{name}`" for name in axis_names) + " |"
    rule = "|---" * (len(axis_names) + 1) + "|"
    rows = [
        "| `" + workload.uuid + "` | " + " | ".join(str(workload.axes.get(name, "")) for name in axis_names) + " |"
        for workload in definition.workloads
    ]
    return [header, rule, *rows]


def build_definition_md(definition: Definition) -> str:
    """The workload document: what is scored, and under whose rules."""
    lines = [
        f"# Definition: {definition.name}",
        "",
        f"- **Operator:** `{definition.op_type}`",
        f"- **Axes:** {_axes_line(definition)}",
        "- **Inputs:** " + ", ".join(f"`{name}`" for name in definition.inputs),
        "- **Outputs:** " + ", ".join(f"`{name}`" for name in definition.outputs),
        f"- **Target:** `{definition.target}`",
        f"- **Scored workloads:** {len(definition.workloads)}",
        "",
        "One implementation has to cover every workload below; the acceptance rule is a",
        "property of the set, not of any single point.",
        "",
        *_workload_table(definition),
        "",
        "## Measurement ownership",
        "",
        "Numerical correctness and timing come from `sikl.benchmarks.api`, which applies the",
        "definition's own `initialize`, `reference` and `compare` callbacks. The driver",
        "reports what that harness returns and measures nothing of its own; the final score",
        "times the baseline in the same window as the delivered kernel.",
        "",
    ]
    if definition.description:
        lines += ["## Upstream description", "", definition.description.strip(), ""]
    return "\n".join(lines)


def build_program_md(definition: Definition, *, kernel_backend: str, starting_point: str) -> str:
    """The brief the implementer agent reads."""
    constants = ", ".join(f"{name}={value}" for name, value in definition.constant_axes.items())
    return "\n".join(
        [
            f"# Program: implement {definition.name} in {kernel_backend}",
            "",
            f"**Backend**: {kernel_backend}",
            f"**Target**: {definition.target}",
            "",
            "## Objective",
            "",
            "Make `run` in `kernel.py` faster than the SIKL baseline across all",
            f"{len(definition.workloads)} scored workloads while keeping every one of them",
            "correct. The loop runs the full correctness suite before it benchmarks, so a",
            "change that wins on time and loses a case is rejected, not traded off.",
            "",
            "## Contract",
            "",
            f"- Fixed axes: `{constants or 'none'}`",
            f"- Varying axes: `{', '.join(definition.variable_axes) or 'none'}` — see `handoff/{DEFINITION_FILENAME}`",
            f"- Inputs by keyword: {', '.join(f'`{name}`' for name in definition.inputs)}",
            f"- Returns: {', '.join(f'`{name}`' for name in definition.outputs)}",
            "- Inputs are functional. Do not mutate them, and do not cache outputs across",
            "  calls: the harness refills the inputs and replays the measured invocation.",
            f"- Every GPU kernel `run` launches is one `kernel.py` defines with `@{KERNEL_DECORATORS[kernel_backend]}`.",
            "  The correctness pass profiles one call per workload and fails a case that launches",
            "  anything else: a vendor library such as hipBLASLt, an aiter kernel, or one of torch's",
            "  own kernels, a cast or a copy included.",
            "",
            "## What the starting point is",
            "",
            f"`kernel.py` starts as the definition's {starting_point}, which passes every workload",
            "under the harness, so the first measurement anchors on a working implementation.",
            f"It is the one kernel exempt from the rule above: replace its body with {kernel_backend} kernels.",
            "",
            "## Where the numbers come from",
            "",
            "`driver.py` shells out to SIKL's harness and reports its per-case latency. It is",
            "the measurement boundary and is not editable; a change that only moves the",
            "reported number is a change to nothing.",
            "",
        ]
    )


def write(
    definition: Definition, handoff_dir: Path, program_md: Path, *, kernel_backend: str, starting_point: str
) -> Path:
    """Write the definition document and the implementer's brief, one copy each."""
    atomic_write_text(
        Path(handoff_dir) / DEFINITION_FILENAME,
        build_definition_md(definition),
        make_parents=True,
        fsync=True,
        fsync_dir=True,
    )
    atomic_write_text(
        Path(program_md),
        build_program_md(definition, kernel_backend=kernel_backend, starting_point=starting_point),
        make_parents=True,
        fsync=True,
        fsync_dir=True,
    )
    return Path(handoff_dir)
