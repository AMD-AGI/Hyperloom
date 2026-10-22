# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read the fields a forge-loop run needs out of one SIKL schema: an op contract narrowed to one scenario, its
workloads, and the kernel section a candidate fills in."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


class SchemaError(ValueError):
    """The SIKL checkout does not hold the requested schema, or holds it ambiguously."""


@dataclass(frozen=True)
class Workload:
    """One scored point of a schema: its variable axes."""

    axes: dict[str, int]

    @property
    def case_id(self) -> str:
        """The workload's name in the driver's output: its axes in schema order, one token."""
        return ",".join(f"{name}={value}" for name, value in self.axes.items())


@dataclass(frozen=True)
class Schema:
    """A SIKL schema as the run sees it: where it came from, what it computes, and what it is scored on."""

    path: Path
    op_dir: str
    name: str
    op_type: str
    description: str
    axes: dict[str, dict]
    attrs: dict[str, object]
    inputs: dict[str, dict]
    outputs: dict[str, dict]
    compare: dict[str, object]
    targets: list[dict]
    workloads: tuple[Workload, ...]

    @property
    def constant_axes(self) -> dict[str, int]:
        return {name: spec["value"] for name, spec in self.axes.items() if spec.get("type") == "const"}

    @property
    def variable_axes(self) -> tuple[str, ...]:
        return tuple(name for name, spec in self.axes.items() if spec.get("type") == "var")


def find(sikl_root: Path, name: str) -> Path:
    """The kernel-less schema file named ``name`` in a SIKL checkout: the problem a candidate is written for."""
    pattern = f"sikl/ops/*/impls/*/{name}/*.json"
    matches = [
        path for path in sorted(sikl_root.glob(pattern)) if not json.loads(path.read_text())["kernel"]["sources"]
    ]
    if not matches:
        raise SchemaError(f"no kernel-less schema {name!r} under {sikl_root} ({pattern})")
    if len(matches) > 1:
        raise SchemaError(f"{len(matches)} kernel-less schemas named {name!r} under {sikl_root}; expected one")
    return matches[0]


def load(path: Path) -> Schema:
    """Load the schema file at ``path``, or explain which part of it is missing."""
    path = path.resolve()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        workloads = tuple(Workload(axes=dict(workload["axes"])) for workload in data["workloads"])
        schema = Schema(
            path=path,
            # sikl/ops/<op>/impls/<language>/<name>/<uuid>.json
            op_dir=path.parents[3].name,
            name=str(data["name"]),
            op_type=str(data["op_type"]),
            description=str(data.get("description", "")),
            axes=dict(data["axes"]),
            attrs=dict(data.get("attrs", {})),
            inputs=dict(data["inputs"]),
            outputs=dict(data["outputs"]),
            compare=dict(data.get("compare", {})),
            targets=list(data["kernel"]["targets"]),
            workloads=workloads,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SchemaError(f"{path}: not a SIKL schema: {error}") from None
    if not schema.workloads:
        raise SchemaError(f"{path}: the schema serves no workload")
    return schema
