# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the ``kernel_tune`` tool reads of a test's landed output: the workload its server ran and the shapes aiter's
log reports untuned.

The workload (model, framework, precision, TP, concurrency) is the one the test's own point configurations
(``variant_*/config.yaml``) say, never a caller's word. The shapes are Forge's demand: the keys the server looked up in
aiter's tuned tables and did not find, which ``kernelforge.gemm_tune.evidence.parse_log`` reads out of the server logs
(its reader, not a second one here), or a list the caller names. Either way they reach the tuner as a ``--demand``
document, the shape source every dense Forge tuner ranks first, and only dense GEMM tables are asked for.

A demand read from the logs can carry two additions. A server whose tables cover every lookup logs no miss, so a call may
ask for the keys the tables served (the rows aiter's hit lines name) as the demand. And the batch sizes the servers'
decode CUDA graphs are captured for, which the ``server_args`` line of the log holds, are added as M of every (N, K) the
demand names: a decode step replays a graph of exactly one of those sizes, so each is the M a GEMM of that step asks for.
"""

from __future__ import annotations

import re
from collections import Counter
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
    canonical_table_name,
    parse_log,
    write_demand,
)

from hyperloom.common.outside_facts import (
    AITER_LOG_TUNED_CONFIG_ENV,
    AITER_TUNED_HIT,
    ENV_CONC,
    ENV_TP,
    SGLANG_CUDA_GRAPH_DECODE_BS,
)
from hyperloom.measure.framework.tables import AITER_VARIABLE_PREFIX
from hyperloom_multibranch.toolbox.tools.base import ToolFailure

__all__ = [
    "DENSE_TABLES",
    "POINT_CONFIG_GLOB",
    "SERVER_LOG_GLOB",
    "ServedWorkload",
    "cuda_graph_batch_sizes",
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
#: The hit line of aiter (``AITER_TUNED_HIT`` in the rule pack): the group that names the file the row was found in, and the
#: private copy of a table (``tables/<variable>/<digest>/``) a test that put one makes its server read.
_HIT = re.compile(AITER_TUNED_HIT)
_PRIVATE_COPY = re.compile(rf"/tables/({AITER_VARIABLE_PREFIX}\w+)/")
#: The variable of each dense table, by the name of the file aiter's tables of it are merged under.
_ENV_OF_TABLE = {table: env for env, (table, _tuner) in DENSE_TABLES.items()}
_GRAPH_GROUP = "sizes"


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


def _key_row(key: Mapping[str, Any]) -> dict[str, Any]:
    """The columns of a demand key, as Forge's reader writes one: a column the table's lookup does not log is ``None``."""
    return {name: None if key.get(name) is None else str(key[name]) for name in KEY_FIELDS}


def _demand_entry(env: str, keys: Sequence[Mapping[str, Any]], miss_count: int) -> dict[str, Any]:
    """The demand entry of the table ``env`` names, for ``keys`` (each a key's columns and its ``requests``), the most asked
    for first."""
    table, tuner = DENSE_TABLES[env]
    rows = sorted(keys, key=lambda row: -row["requests"])
    return Demand(
        table=table,
        tuner=tuner,
        env_var=env,
        key_schema=list(TABLE_KEY_SCHEMA.get(table, ("M", "N", "K"))),
        logged_fields=[name for name in KEY_FIELDS if any(row[name] is not None for row in rows)],
        miss_count=miss_count,
        keys=rows,
    ).to_dict()


def _server_text(landed: Path) -> str:
    logs = sorted(landed.rglob(SERVER_LOG_GLOB))
    if not logs:
        raise ToolFailure(f"the test landed no server log ({SERVER_LOG_GLOB}) to read untuned shapes from")
    return "\n".join(log.read_text(encoding=_UTF8, errors="replace") for log in logs)


def _hit_env(path: str) -> str | None:
    """The variable of the dense table a hit line's ``path`` is a file of: the private copy of a table the test put names it
    (``tables/<variable>/``), and any other file is the table of that name (the merged view of one, for instance)."""
    copy = _PRIVATE_COPY.search(path)
    env = copy.group(1) if copy else _ENV_OF_TABLE.get(canonical_table_name(path))
    return env if env in DENSE_TABLES else None


def _hit_key(printed: Mapping[str, str | None], env: str) -> dict[str, Any] | None:
    """The key of the row a hit line names, as the demand of that table keys it; ``None`` when the table is keyed on a
    column the line does not print (the per-token a8w8 tables on ``q_dtype_w``), so no key can be made of it."""
    schema = TABLE_KEY_SCHEMA.get(DENSE_TABLES[env][0], ("M", "N", "K"))
    if any(name not in KEY_FIELDS or printed.get(name) is None for name in schema):
        return None
    return _key_row(printed)


def _hit_demands(text: str) -> tuple[dict[str, Counter[tuple[Any, ...]]], int]:
    """The keys of the dense rows the lookups of the log resolved in, per table variable, each with the lines that name it;
    and how many hit lines no key was made of (another table's, or a table keyed on a column the line does not print)."""
    keys: dict[str, Counter[tuple[Any, ...]]] = {}
    unread = 0
    for line in text.splitlines():
        for found in _HIT.finditer(line):
            printed = {**found.groupdict(), "otype": found.groupdict().get("outdtype")}
            env = _hit_env(printed.get("table") or "")
            key = None if env is None else _hit_key(printed, env)
            if env is None or key is None:
                unread += 1
                continue
            keys.setdefault(env, Counter())[tuple(key.values())] += 1
    return keys, unread


def _with_hits(report: dict[str, Any], hits: Mapping[str, Mapping[tuple[Any, ...], int]]) -> dict[str, Any]:
    """``report`` with the keys the tables served added to the demand of each table, each counted by the lines that name it."""
    entries = {entry["env_var"]: entry for entry in report["demands"]}
    for env, served in hits.items():
        known = entries.get(env, {"keys": [], "miss_count": 0})
        have = {tuple(_key_row(key).values()) for key in known["keys"]}
        added = [
            {**dict(zip(KEY_FIELDS, key, strict=True)), "requests": lines}
            for key, lines in served.items()
            if key not in have
        ]
        entries[env] = _demand_entry(env, [*known["keys"], *added], known["miss_count"])
    return {**report, "demands": sorted(entries.values(), key=lambda entry: -entry["miss_count"])}


def cuda_graph_batch_sizes(text: str, conc: int) -> tuple[int, ...]:
    """The batch sizes the decode CUDA graphs of the servers whose log is ``text`` were captured for, and a decode step of
    up to ``conc`` running requests can replay.

    A step of ``n`` requests replays the graph of the smallest captured size that is at least ``n``, so the sizes above the
    smallest one that is at least ``conc`` are never replayed by this workload. Empty when the log holds no ``server_args``
    line that says (no server of the test captured decode graphs, or a server that words it differently).

    Raises:
        ToolFailure: When the rule pack's pattern has no ``sizes`` group (the pack and this reader disagree).
    """
    pattern = re.compile(SGLANG_CUDA_GRAPH_DECODE_BS)
    if _GRAPH_GROUP not in pattern.groupindex:
        raise ToolFailure(f"the rule pack's sglang_cuda_graph_decode_bs pattern has no group {_GRAPH_GROUP!r}")
    captured = sorted({int(size) for found in pattern.finditer(text) for size in found.group(_GRAPH_GROUP).split(",")})
    replayed = min((size for size in captured if size >= conc), default=None)
    return tuple(size for size in captured if replayed is None or size <= replayed)


def _with_graph_sizes(report: dict[str, Any], sizes: Sequence[int]) -> dict[str, Any]:
    """``report`` with each of ``sizes`` added as the M of every other column set (N, K and the dtype and flag columns) the
    demand of a table names; a key the demand already has keeps its count, and an added one counts as asked for no time."""
    if not sizes:
        return report
    demands = []
    for entry in report["demands"]:
        rows = {tuple(_key_row(key).values()): dict(key) for key in entry["keys"]}
        for key in list(rows.values()):
            for size in sizes:
                row = _key_row({**key, "M": size})
                rows.setdefault(tuple(row.values()), {**row, "requests": 0})
        demands.append(_demand_entry(entry["env_var"], list(rows.values()), entry["miss_count"]))
    return {**report, "demands": demands}


def demand_from_logs(landed: Path, *, retune: bool = False, conc: int = 0) -> dict[str, Any]:
    """Forge's demand document for the dense tables the server logs below ``landed`` report lookups that missed, with the
    keys the tables served added when ``retune`` is set, and, when ``conc`` is given, the decode CUDA-graph batch sizes a
    workload of that concurrency replays added as M. ``cuda_graph_batch_sizes`` and ``served_keys`` say what was added.

    Raises:
        ToolFailure: When the test landed no server log, or its logs report no untuned dense GEMM shape and no served one
            was asked for (or none was logged).
    """
    text = _server_text(landed)
    demand = _dense_only(parse_log(text))
    served, unread = _hit_demands(text) if retune else ({}, 0)
    demand = _with_hits(demand, served)
    if not demand["demands"]:
        logs = len(sorted(landed.rglob(SERVER_LOG_GLOB)))
        raise ToolFailure(_nothing_to_retune(logs, unread) if retune else _nothing_to_tune(logs))
    sizes = cuda_graph_batch_sizes(text, conc) if conc else ()
    return {
        **_with_graph_sizes(demand, sizes),
        "cuda_graph_batch_sizes": list(sizes),
        "served_keys": sum(len(keys) for keys in served.values()),
        "served_lines_unread": unread,
    }


def _nothing_to_retune(logs: int, unread: int) -> str:
    unkeyed = (
        f"; {unread} hit line(s) name rows of tables this reader cannot key (a table keyed on a column the line does not "
        "print, such as the per-token a8w8 tables on q_dtype_w): name those shapes with params.shapes"
        if unread
        else ""
    )
    return (
        f"the {logs} server log(s) of the test report no dense GEMM lookup of a table this tool tunes, neither a miss "
        f"nor a served row (aiter logs the rows it serves only when {AITER_LOG_TUNED_CONFIG_ENV} is set in the "
        f"server's environment): there is nothing to retune from them{unkeyed}"
    )


def _nothing_to_tune(logs: int) -> str:
    return (
        f"the {logs} server log(s) of the test report no untuned dense GEMM shape (aiter's 'not found tuned config' "
        "lines for a dense table): there is nothing to tune from them; if the tables cover every shape, set "
        "retune=true to tune the shapes they serve"
    )


def explicit_demand(shapes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Forge's demand document for ``shapes``: each names its table's variable (``env``), ``M``, ``N``, ``K`` and, for a table
    keyed on more, those columns (``KEY_FIELDS``). The same shape twice counts as asked for twice, and the table asked
    for most comes first, as in the demand of a log."""
    keys: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = {}
    for shape in shapes:
        key = _key_row(shape)
        found = keys.setdefault(str(shape["env"]), {}).setdefault(tuple(key.values()), {**key, "requests": 0})
        found["requests"] += 1
    demands = [
        _demand_entry(env, list(wanted.values()), sum(row["requests"] for row in wanted.values()))
        for env, wanted in keys.items()
    ]
    return {"schema": SCHEMA_VERSION, "demands": sorted(demands, key=lambda demand: -demand["miss_count"])}


def write_demand_of(demand: Mapping[str, Any], path: Path) -> Path:
    """Write the document that asks Forge for one table's shapes; the tuner of that table picks its entry out of it."""
    return write_demand({"schema": SCHEMA_VERSION, "demands": [dict(demand)]}, path)
