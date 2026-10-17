# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read the fields a forge-loop run needs out of one definition of an exported SIKL schema collection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


class BundleError(ValueError):
    """The bundle does not describe the requested definition."""


@dataclass(frozen=True)
class Workload:
    """One scored point of a definition."""

    uuid: str
    axes: dict[str, int]


@dataclass(frozen=True)
class Definition:
    """A SIKL definition and the baseline its candidates are measured against."""

    name: str
    op_type: str
    description: str
    axes: dict[str, dict]
    inputs: dict[str, dict]
    outputs: dict[str, dict]
    target: list[dict]
    language: str
    reference_source: str
    baseline_source: str | None
    workloads: tuple[Workload, ...]

    @property
    def constant_axes(self) -> dict[str, int]:
        return {name: spec["value"] for name, spec in self.axes.items() if spec.get("type") == "const"}

    @property
    def variable_axes(self) -> tuple[str, ...]:
        return tuple(name for name, spec in self.axes.items() if spec.get("type") == "var")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _find(root: Path, pattern: str, name: str, kind: str) -> Path:
    matches = [path for path in sorted(root.rglob(pattern)) if path.is_file()]
    if not matches:
        raise BundleError(f"no {kind} for {name!r} under {root}")
    if len(matches) > 1:
        raise BundleError(f"{len(matches)} {kind} files for {name!r} under {root}; expected one")
    return matches[0]


def _baseline(bundle: Path, name: str) -> dict:
    return _read_json(_find(bundle / "solutions" / "baseline", f"{name}/*.json", name, "baseline"))


def _baseline_candidate_source(baseline: dict) -> str | None:
    """The baseline's source when it can stand in as a candidate's `kernel.py::run`, else None."""
    spec, sources = baseline["spec"], baseline.get("sources") or []
    if spec.get("destination_passing_style") or len(sources) != 1:
        return None
    if spec.get("entry_point") != f"{sources[0]['path']}::run":
        return None
    return str(sources[0]["content"])


def _workloads(bundle: Path, name: str) -> tuple[Workload, ...]:
    path = _find(bundle / "workloads", f"{name}.jsonl", name, "workload")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise BundleError(f"workload file for {name!r} is empty")
    return tuple(Workload(uuid=str(row["workload"]["uuid"]), axes=dict(row["workload"]["axes"])) for row in rows)


def load(bundle: Path, name: str, *, language: str) -> Definition:
    """Load ``name`` from ``bundle``, or explain which part of it is missing."""
    bundle = bundle.resolve()
    data = _read_json(_find(bundle / "definitions", f"{name}.json", name, "definition"))
    if data.get("schema_version") != 2:
        raise BundleError(f"{name!r} is schema_version {data.get('schema_version')!r}; this path requires 2")
    baseline = _baseline(bundle, name)
    return Definition(
        name=name,
        op_type=str(data["op_type"]),
        description=str(data.get("description", "")),
        axes=dict(data["axes"]),
        inputs=dict(data["inputs"]),
        outputs=dict(data["outputs"]),
        target=list(baseline["spec"]["target"]),
        language=language or str(baseline["spec"]["language"]),
        reference_source=str(data["reference"]),
        baseline_source=_baseline_candidate_source(baseline),
        workloads=_workloads(bundle, name),
    )
