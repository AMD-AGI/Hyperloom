# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the ``kernel_tune`` tool reads of a test's landed output: the workload its server ran and the shapes aiter's
log reports untuned.

The workload (model, framework, precision, TP, concurrency) is the one the test's own point configurations
(``variant_*/config.yaml``) say, never a caller's word. The shapes are Forge's demand: the keys the server looked up in
aiter's tuned tables and did not find, which ``kernelforge.gemm_tune.evidence.parse_log`` reads out of the server logs
(its reader, not a second one here), or a list the caller names. Either way they reach the tuner as a ``--demand``
document, the shape source every dense Forge tuner ranks first, and only dense GEMM tables are asked for.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from kernelforge.gemm_tune.evidence import (
    KEY_FIELDS,
    MOE_TABLE,
    SCHEMA_VERSION,
    TABLE_KEY_SCHEMA,
    TABLE_TO_TUNER,
    Demand,
    parse_log,
    write_demand,
)

from hyperloom.common.outside_facts import ENV_CONC, ENV_TP
from hyperloom_multibranch.toolbox.tools.base import ToolFailure

__all__ = [
    "DENSE_TABLES",
    "POINT_CONFIG_GLOB",
    "SERVER_LOG_GLOB",
    "ServedWorkload",
    "demand_from_logs",
    "explicit_demand",
    "read_benchmark",
    "served_workload",
    "write_demand_of",
]

#: The YAML the grid runner wrote for each point a test ran, in the test's landed directory.
POINT_CONFIG_GLOB = "variant_*/config.yaml"
SERVER_LOG_GLOB = "server*.log"
POINT_CONFIG_MAX_BYTES = 1 << 20
#: The dense GEMM tables Forge tunes, by the variable aiter reads each from: (the file aiter merges it under, the tuner).
DENSE_TABLES: Mapping[str, tuple[str, str]] = {
    env: (table, tuner) for table, (tuner, env) in TABLE_TO_TUNER.items() if table != MOE_TABLE
}
#: The framework whose tuned-table lookups this tool tunes for; another one's tables are named differently.
SERVED_FRAMEWORK = "sglang"
_UTF8 = "utf-8"


def read_benchmark(path: Path) -> Mapping[str, Any]:
    """The ``benchmark`` section of the point YAML at ``path``; empty when it is too large, unreadable or not a mapping."""
    try:
        if path.stat().st_size > POINT_CONFIG_MAX_BYTES:
            return {}
        section = (yaml.safe_load(path.read_text(encoding=_UTF8)) or {})["benchmark"]
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError):
        return {}
    return section if isinstance(section, Mapping) else {}


@dataclass(frozen=True)
class ServedWorkload:
    """What the test's server served, as Forge routes and sizes a tuning run by it."""

    model_path: str
    framework: str
    precision: str
    tp: int
    conc: int


def _envs(benchmark: Mapping[str, Any]) -> Mapping[str, Any]:
    envs = benchmark.get("envs")
    return envs if isinstance(envs, Mapping) else {}


def _positive(value: Any, what: str) -> int:
    try:
        number = int(str(value))
    except ValueError:
        number = 0
    if number < 1:
        raise ToolFailure(f"the test's point configuration has no usable {what} ({value!r})")
    return number


def served_workload(landed: Path) -> ServedWorkload:
    """The workload of the test whose landed output is ``landed``: the model, framework, precision and TP of its first
    point, the highest concurrency of any point.

    Raises:
        ToolFailure: When the test left no readable point, serves a framework this tool does not tune for, or its model
            is not a directory with a ``config.json`` here (Forge reads the model's shapes and routing from it).
    """
    points = [found for found in (read_benchmark(path) for path in sorted(landed.glob(POINT_CONFIG_GLOB))) if found]
    if not points:
        raise ToolFailure(
            f"the test landed no readable point configuration ({POINT_CONFIG_GLOB}) to take its workload from"
        )
    first = points[0]
    framework = str(first.get("framework") or "").strip().lower()
    if framework != SERVED_FRAMEWORK:
        raise ToolFailure(
            f"kernel_tune tunes aiter tables for {SERVED_FRAMEWORK}; the test served {framework or 'no framework'!r}"
        )
    model, precision = str(first.get("model") or "").strip(), str(first.get("precision") or "").strip().lower()
    if not precision:
        raise ToolFailure("the test's point configuration names no precision")
    if not (Path(model) / "config.json").is_file():
        raise ToolFailure(f"the model of the test ({model!r}) is not a model directory (config.json) readable here")
    tp = _positive(_envs(first).get(ENV_TP, 1), ENV_TP)
    conc = max(_positive(_envs(point).get(ENV_CONC, 1), ENV_CONC) for point in points)
    return ServedWorkload(model, framework, precision, tp, conc)


def _dense_only(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": report.get("schema", SCHEMA_VERSION),
        "demands": [entry for entry in report.get("demands") or [] if entry.get("env_var") in DENSE_TABLES],
    }


def demand_from_logs(landed: Path) -> dict[str, Any]:
    """Forge's demand document for the dense tables the server logs below ``landed`` report lookups that missed.

    Raises:
        ToolFailure: When the test landed no server log, or its logs report no untuned dense GEMM shape.
    """
    logs = sorted(landed.rglob(SERVER_LOG_GLOB))
    if not logs:
        raise ToolFailure(f"the test landed no server log ({SERVER_LOG_GLOB}) to read untuned shapes from")
    text = "\n".join(log.read_text(encoding=_UTF8, errors="replace") for log in logs)
    demand = _dense_only(parse_log(text))
    if not demand["demands"]:
        raise ToolFailure(
            f"the {len(logs)} server log(s) of the test report no untuned dense GEMM shape (aiter's 'not found tuned config' "
            "lines for a dense table): there is nothing to tune from them"
        )
    return demand


def explicit_demand(shapes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Forge's demand document for ``shapes``: each names its table's variable (``env``), ``M``, ``N``, ``K`` and, for a table
    keyed on more, those columns (``KEY_FIELDS``). The same shape twice counts as asked for twice, and the table asked
    for most comes first, as in the demand of a log."""
    keys: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = {}
    for shape in shapes:
        key = {name: None if shape.get(name) is None else str(shape[name]) for name in KEY_FIELDS}
        found = keys.setdefault(str(shape["env"]), {}).setdefault(tuple(key.values()), {**key, "requests": 0})
        found["requests"] += 1
    demands = []
    for env, wanted in keys.items():
        table, tuner = DENSE_TABLES[env]
        rows = sorted(wanted.values(), key=lambda row: -row["requests"])
        demands.append(
            Demand(
                table=table,
                tuner=tuner,
                env_var=env,
                key_schema=list(TABLE_KEY_SCHEMA.get(table, ("M", "N", "K"))),
                logged_fields=[name for name in KEY_FIELDS if any(row[name] is not None for row in rows)],
                miss_count=sum(row["requests"] for row in rows),
                keys=rows,
            ).to_dict()
        )
    return {"schema": SCHEMA_VERSION, "demands": sorted(demands, key=lambda demand: -demand["miss_count"])}


def write_demand_of(demand: Mapping[str, Any], path: Path) -> Path:
    """Write the document that asks Forge for one table's shapes; the tuner of that table picks its entry out of it."""
    return write_demand({"schema": SCHEMA_VERSION, "demands": [dict(demand)]}, path)
