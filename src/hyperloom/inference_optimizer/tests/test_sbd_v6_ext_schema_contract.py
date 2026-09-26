# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Top-level ``ext`` blocks stay aligned with their TypedDicts.

The conc_sweep timeline already compares nested blocks key for key. Every other
event only had an assembler and a TypedDict that nobody compared, which is how
``V6KernelExt`` lost ``measurements`` and ``V6WarmStartExt`` renamed ``request``
to ``requested`` without the wire noticing. This contract pins the top-level
keys of every assembler that has a TypedDict; nested shape drift stays the
concern of each event's own timeline tests.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.breakdown import schema

_RECORDER_DIR = Path(__file__).resolve().parents[1] / "breakdown" / "recorder"

#: Assembler name → TypedDict that declares its top-level ``ext`` keys.
_EXT_CONTRACTS: dict[str, type] = {
    "assemble_baseline_ext": schema.V6BaselineExt,
    "assemble_conc_sweep_ext": schema.V6ConcSweepExt,
    "assemble_enablement_ext": schema.V6EnablementExt,
    "assemble_framework_ext": schema.V6FrameworkExt,
    "assemble_kernel_ext": schema.V6KernelExt,
    "assemble_phase_ext": schema.V6PhaseExt,
    "assemble_roofline_ext": schema.V6RooflineExt,
    "assemble_stack_ext": schema.V6StackExt,
    "assemble_warm_replay_ext": schema.V6WarmReplayExt,
    "assemble_warm_start_ext": schema.V6WarmStartExt,
}


def _wire_keys(fn: ast.FunctionDef) -> set[str]:
    """Collect top-level keys an assembler puts on its ``ext`` dict."""
    keys: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.startswith("ext") and isinstance(node.value, ast.Dict):
                    keys |= {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id.startswith("ext")
                    and isinstance(target.slice, ast.Constant)
                ):
                    keys.add(target.slice.value)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id.startswith("ext"):
            if isinstance(node.value, ast.Dict):
                keys |= {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple) and node.value.elts:
            first = node.value.elts[0]
            if isinstance(first, ast.Dict):
                keys |= {k.value for k in first.keys if isinstance(k, ast.Constant)}
    return keys


def _assemblers() -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for path in sorted(_RECORDER_DIR.glob("*_event.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in _EXT_CONTRACTS:
                found[node.name] = _wire_keys(node)
    return found


@pytest.mark.parametrize("assembler,declared", sorted(_EXT_CONTRACTS.items()))
def test_every_recorded_ext_block_matches_its_typeddict(assembler: str, declared: type) -> None:
    wire = _assemblers().get(assembler)
    assert wire is not None, f"{assembler} not found under {_RECORDER_DIR}"
    declared_keys = set(declared.__annotations__)
    assert wire == declared_keys, (
        f"{assembler} <-> {declared.__name__}: "
        f"recorded-only={sorted(wire - declared_keys)} "
        f"declared-only={sorted(declared_keys - wire)}"
    )
