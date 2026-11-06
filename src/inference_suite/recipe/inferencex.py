# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""InferenceX as a recipe source: AMD single-node master entries, the points they expand to, and the launch each binds.

What InferenceX states as YAML is read here: its entries, the board a runner label stands for, the matrix points an
entry expands to and the environment its launcher hands each one. Which variant a point binds and its golden
acceptance length are InferenceX's own code, run once when the snapshot is taken and read here as recorded data. A
bound recipe becomes a command line by the rules of srtctl at the snapshot's srt-slurm commit.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any

import yaml

from inference_suite.argv import argv_flag_occurrences, speculative_config
from inference_suite.contracts.base_launch import (
    AGENTX_FRAMEWORKS,
    GoldenCurve,
    LaunchQuery,
    LaunchSource,
    SourceKind,
    Topology,
    Workload,
)
from inference_suite.recipe.model import (
    ATOM_ENTRYPOINT,
    ATOM_RUNNER_OWNED,
    InferencexSelection,
    InferencexSnapshot,
    RecipeSourceError,
    RepoBundle,
    SGLANG_ENTRYPOINT,
    SGLANG_RUNNER_OWNED,
    SourceGolden,
    SourceLaunch,
    SourceMiss,
    VLLM_ENTRYPOINT,
    VLLM_RUNNER_OWNED,
    adaptive_verification_reason,
    boards_interchangeable,
)

INFERENCEX_REPO = "https://github.com/SemiAnalysisAI/InferenceX"
#: The InferenceX project inside the repository; recipe and variant paths are relative to it.
PROJECT = "inferencex-e2e"
MASTER = f"{PROJECT}/configs/amd-master.yaml"
RUNNERS = f"{PROJECT}/configs/runners.yaml"
GOLDEN_DIR = f"{PROJECT}/infx/golden_al_distribution"

# The values below are InferenceX's at the pinned commit: infx/matrix/validation.py (recipe directory),
# infx/matrix/generate.py (CPU DRAM offload) and infx/srt_slurm/generate.py (the launcher's point environment).
_SINGLE_NODE_RECIPES = "benchmarks/single_node/srt-slurm-recipes"
_CLUSTER = "cluster:"
_MAX_AGENTIC_CPU_DRAM_MIB = 2_861_022
_BYTES_PER_MIB = 1024 * 1024
_BYTES_PER_GB = 1_000_000_000
_RANDOM_RANGE_RATIO = "0.8"
_THINKING_MODE = "thinking_on"

#: What a matrix point writes into ``benchmark.env`` (InferenceX ``workload.BOUND_ENV`` for single-node points, and
#: the workload values ``validate_recipe`` holds a recipe to); the rest of it configures the client for any point.
_POINT_CLIENT_ENV = frozenset(
    {
        *("MODEL", "ISL", "OSL", "CONC", "CONC_LIST", "RANDOM_RANGE_RATIO", "USE_CHAT_TEMPLATE"),
        *("RESULT_DIR", "AGENTIC_OUTPUT_DIR", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "ENABLE_AGENTX_POWER"),
        *("IMAGE", "MODEL_PREFIX", "FRAMEWORK", "PRECISION", "DISAGG", "DURATION", "MAX_MODEL_LEN"),
        *("KV_OFFLOAD_BACKEND", "KV_OFFLOAD_BACKEND_METADATA", "ROUTER_METADATA", "KV_P2P_TRANSFER"),
        *("POWER", "REQUIRE_POWER", "IS_AGENTIC", "SCENARIO_TYPE", "RUN_EVAL", "EVAL_ONLY"),
        *("SPEC_ALGORITHM", "SPEC_DRAFT_MODEL", "SPEC_DECODING", "KV_OFFLOADING", "TOTAL_CPU_DRAM_GB"),
        *("TP", "EP_SIZE", "PP_SIZE", "DCP_SIZE", "PCP_SIZE", "DP_ATTENTION", "GPU_COUNT"),
    }
)


class Scenario(str, Enum):
    FIXED = "fixed-seq-len"
    AGENTIC = "agentic-coding"


_SCENARIO_OF = {Workload.FIXED: Scenario.FIXED, Workload.AGENTX: Scenario.AGENTIC}


@dataclass(frozen=True)
class InferencexPoint:
    """One single-node matrix point of an AMD master entry.

    ``group`` indexes the entry's sequence (fixed) or scenario block (AgentX) and ``arm`` the search-space item in
    it; the points of one arm differ only in concurrency. ``recipe`` is ``<file>[:<selector>]`` under the project.
    """

    entry: str
    scenario: Scenario
    group: int
    arm: int
    conc: int
    framework: str
    model: str
    model_prefix: str
    precision: str
    image: str
    runner: str
    board: str
    recipe: str
    tp: int
    ep: int
    pp: int
    dcp: int
    pcp: int
    dp_attention: bool
    spec_decoding: str
    isl: int | None
    osl: int | None
    kv_offloading: str | None
    kv_offload_backend: Mapping[str, Any] | None
    router: Mapping[str, Any] | None
    total_cpu_dram_gb: int
    require_power: bool

    @property
    def arm_key(self) -> str:
        return f"{self.entry}/{self.scenario.value}[{self.group}]/search-space[{self.arm}]"

    @property
    def key(self) -> str:
        """The point's name in a snapshot's selections."""
        return f"{self.arm_key}/conc{self.conc}"

    @property
    def gpu_count(self) -> int:
        return self.tp * self.pp * self.pcp

    def environment(self) -> dict[str, str]:
        """The environment InferenceX's launcher hands the point to ``select_recipe`` (``point_environment``)."""
        agentic = self.scenario is Scenario.AGENTIC
        environment = {
            "IMAGE": self.image,
            "MODEL": self.model,
            "MODEL_PREFIX": self.model_prefix,
            "PRECISION": self.precision,
            "FRAMEWORK": self.framework,
            "SPEC_DECODING": self.spec_decoding,
            "IS_AGENTIC": "1" if agentic else "0",
            "ISL": str(self.isl or 0),
            "OSL": str(self.osl or 0),
            "RANDOM_RANGE_RATIO": _RANDOM_RANGE_RATIO,
            "THINKING_MODE": _THINKING_MODE,
            "RUN_EVAL": "false",
            "EVAL_ONLY": "false",
            "REQUIRE_POWER": "1" if self.require_power else "0",
            "ROUTER_METADATA": json.dumps(self.router) if self.router else "",
            "KV_OFFLOAD_BACKEND_METADATA": json.dumps(self.kv_offload_backend) if self.kv_offload_backend else "",
            "TOTAL_CPU_DRAM_GB": str(self.total_cpu_dram_gb),
        }
        if agentic:
            environment["KV_OFFLOADING"] = str(self.kv_offloading)
        return {
            **environment,
            "CONC": str(self.conc),
            "TP": str(self.tp),
            "EP_SIZE": str(self.ep),
            "PP_SIZE": str(self.pp),
            "DCP_SIZE": str(self.dcp),
            "PCP_SIZE": str(self.pcp),
            "DP_ATTENTION": str(self.dp_attention).lower(),
            "GPU_COUNT": str(self.gpu_count),
        }


def _yaml_mapping(bundle: RepoBundle, path: str) -> Mapping[str, Any]:
    document = yaml.safe_load(bundle.text(path))
    if not isinstance(document, Mapping):
        raise RecipeSourceError(f"{path} is not a mapping")
    return document


def _runner_board(runner: str, labels: Mapping[str, Sequence[str]]) -> str:
    """The board a runner label stands for: the label itself, or for a cluster label the one board label whose
    nodes include all of the cluster's."""
    if not runner.startswith(_CLUSTER):
        return runner
    nodes = set(labels.get(runner) or ())
    boards = [
        label
        for label, members in labels.items()
        if not label.startswith(_CLUSTER) and nodes and nodes <= set(members or ())
    ]
    if len(boards) != 1:
        raise RecipeSourceError(f"runner {runner} is on {len(boards)} board labels, not one: {boards}")
    return boards[0]


def _concurrencies(benchmark: Mapping[str, Any]) -> list[int]:
    """``conc-list``, else ``conc-start`` doubled up to ``conc-end``, the last step capped at it."""
    if benchmark.get("conc-list"):
        return list(benchmark["conc-list"])
    conc, end = benchmark["conc-start"], benchmark["conc-end"]
    values = []
    while conc <= end:
        values.append(conc)
        if conc == end:
            break
        conc = min(conc * 2, end)
    return values


def _cpu_dram_gb(
    runner: str, block: Mapping[str, Any], benchmark: Mapping[str, Any], runners: Mapping[str, Any]
) -> int:
    """The CPU DRAM in GB an AgentX point offloads KV cache to: its GPUs' share of the node's DRAM, 0 without."""
    if benchmark["kv-offloading"] != "dram":
        return 0
    cluster = (
        (runners.get("clusters") or {}).get(runner.removeprefix(_CLUSTER)) if runner.startswith(_CLUSTER) else None
    )
    if cluster is None:
        raise RecipeSourceError(f"runner {runner} offloads to CPU DRAM but is no cluster with hardware facts")
    available = min(cluster["available-cpu-dram-mib"], _MAX_AGENTIC_CPU_DRAM_MIB)
    per_node = cluster["gpus-per-node"]
    gpus = benchmark["tp"] * benchmark.get("pp", 1) * benchmark.get("pcp-size", 1)
    if gpus > per_node:
        raise RecipeSourceError(f"a {gpus}-GPU point of runner {runner} exceeds its {per_node} GPUs per node")
    share = Decimal(available) * _BYTES_PER_MIB * Decimal(str(block["dram-utilization"])) * gpus / per_node
    return int(share / _BYTES_PER_GB)


@dataclass(frozen=True)
class _Arm:
    """One search-space item of an entry's scenario block, with where it sits and the block holding it."""

    entry_name: str
    entry: Mapping[str, Any]
    scenario: Scenario
    group: int
    index: int
    block: Mapping[str, Any]
    benchmark: Mapping[str, Any]


def _arms(master: Mapping[str, Any]) -> Iterator[_Arm]:
    for name, entry in master.items():
        if entry.get("multinode", False):
            continue
        for scenario in Scenario:
            for group, block in enumerate(entry["scenarios"].get(scenario.value) or ()):
                for index, benchmark in enumerate(block["search-space"]):
                    yield _Arm(name, entry, scenario, group, index, block, benchmark)


def _arm_points(arm: _Arm, runners: Mapping[str, Any]) -> Iterator[InferencexPoint]:
    entry, block, benchmark = arm.entry, arm.block, arm.benchmark
    fixed = arm.scenario is Scenario.FIXED
    runner = entry["runner"]
    power_from = block if fixed else benchmark
    for conc in _concurrencies(benchmark):
        yield InferencexPoint(
            entry=arm.entry_name,
            scenario=arm.scenario,
            group=arm.group,
            arm=arm.index,
            conc=conc,
            framework=entry["framework"],
            model=entry["model"],
            model_prefix=entry["model-prefix"],
            precision=entry["precision"],
            image=entry["image"],
            runner=runner,
            board=_runner_board(runner, runners["labels"]),
            recipe=f"{_SINGLE_NODE_RECIPES}/{entry['srt-recipe-dir']}/{benchmark['srt-recipe']}",
            tp=benchmark["tp"],
            ep=benchmark.get("ep") or 1,
            pp=benchmark.get("pp", 1),
            dcp=benchmark.get("dcp-size", 1),
            pcp=benchmark.get("pcp-size", 1),
            dp_attention=bool(benchmark.get("dp-attn") or False),
            spec_decoding=benchmark.get("spec-decoding", "none"),
            isl=block["isl"] if fixed else None,
            osl=block["osl"] if fixed else None,
            kv_offloading=None if fixed else benchmark["kv-offloading"],
            kv_offload_backend=None if fixed else benchmark.get("kv-offload-backend"),
            router=benchmark.get("router", entry.get("router")),
            total_cpu_dram_gb=0 if fixed else _cpu_dram_gb(runner, block, benchmark, runners),
            require_power=bool(power_from.get("require-power", power_from.get("require_power", False))),
        )


def inferencex_points(bundle: RepoBundle) -> tuple[InferencexPoint, ...]:
    """Every fixed-sequence and AgentX point of the single-node entries in InferenceX's AMD master config."""
    runners = _yaml_mapping(bundle, RUNNERS)
    return tuple(point for arm in _arms(_yaml_mapping(bundle, MASTER)) for point in _arm_points(arm, runners))


def srtctl_cli_args(engine: str, config: Mapping[str, Any]) -> tuple[str, ...]:
    """srtctl's ``_config_to_cli_args`` for ``engine``: keys sorted, ``True`` a bare flag, ``False`` and ``None``
    left out, a list one flag with several values. ATOM keeps its mixed spelling; the others take ``_`` as ``-``."""
    args: list[str] = []
    for key, value in sorted(config.items()):
        flag = f"--{key if engine == 'atom' else key.replace('_', '-')}"
        if value is True:
            args.append(flag)
        elif value is False or value is None:
            continue
        elif isinstance(value, list):
            args.extend([flag, *map(str, value)])
        else:
            args.extend([flag, str(value)])
    return tuple(args)


def _override_value(raw: str) -> Any:
    """srtctl's ``parse_value``: a YAML scalar, with a mapping kept as its literal text."""
    if raw == "":
        return ""
    try:
        parsed = yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw
    return raw if isinstance(parsed, dict) else parsed


def _with_overrides(recipe: Mapping[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """``recipe`` with InferenceX's golden-acceptance ``--set``/``--unset`` overrides applied as srtctl applies them."""
    result = copy.deepcopy(dict(recipe))
    if len(overrides) % 2:
        raise RecipeSourceError(f"golden overrides come in operation/target pairs, got {list(overrides)}")
    for operation, target in zip(overrides[::2], overrides[1::2]):
        path, _, raw = target.partition("=")
        if any(mark in path for mark in "[]\"'"):
            raise RecipeSourceError(f"golden override path {path!r} is not a plain dotted path")
        *parents, last = path.split(".")
        node = result
        for key in parents:
            node = node.setdefault(key, {})
        if operation == "--set":
            node[last] = _override_value(raw)
        elif operation == "--unset":
            node.pop(last, None)
        else:
            raise RecipeSourceError(f"unknown golden override operation {operation!r}")
    return result


@dataclass(frozen=True)
class _EngineRules:
    """How srtctl starts one engine's aggregated single-node worker.

    ``owned`` are the arguments srtctl itself sets or allocates, which the runner owns here (compared in srtctl's
    spelling with ``_`` read as ``-``); ``unported`` are arguments whose rendering needs the srtctl job topology.
    """

    entrypoint: tuple[str, ...]
    frontend: str
    owned: frozenset[str]
    unported: frozenset[str] = frozenset()


_ENGINES = {
    "sglang": _EngineRules(
        entrypoint=SGLANG_ENTRYPOINT,
        frontend="sglang",
        owned=SGLANG_RUNNER_OWNED,
    ),
    "vllm": _EngineRules(
        entrypoint=VLLM_ENTRYPOINT,
        frontend="vllm",
        owned=VLLM_RUNNER_OWNED,
        unported=frozenset({"connector"}),
    ),
    "atom": _EngineRules(
        entrypoint=ATOM_ENTRYPOINT,
        frontend="atomesh",
        owned=ATOM_RUNNER_OWNED,
        unported=frozenset({"tp", "tensor-parallel-size", "kv-transfer-config", "extra-kv-connectors"}),
    ),
}


def _engine_type(recipe: Mapping[str, Any]) -> str:
    engine = recipe.get("engine")
    return str(engine.get("type") if isinstance(engine, Mapping) else engine)


def _unported(recipe: Mapping[str, Any], engine: str) -> str | None:
    """What of the recipe's deployment this source cannot reproduce outside srtctl, or ``None``."""
    if engine not in _ENGINES:
        return f"it runs engine {engine}"
    block = recipe.get("engine")
    if isinstance(block, Mapping) and (set(block) - {"type", "connector"} or block.get("connector") is not None):
        return f"its engine block {dict(block)} configures more than the engine type"
    frontend = recipe.get("frontend") or {}
    if frontend.get("type", _ENGINES[engine].frontend) != _ENGINES[engine].frontend:
        return f"it serves through the {frontend['type']} frontend"
    if frontend.get("enable_multiple_frontends"):
        return "it runs several frontends"
    if set(recipe.get("roles") or {}) != {"agg"}:
        return f"it runs roles {sorted(recipe.get('roles') or {})}, not one aggregated worker"
    for extra in ("setup_script", "services"):
        if recipe.get(extra):
            return f"it needs {extra} {recipe[extra]!r}"
    args = recipe["roles"]["agg"].get("args") or {}
    topology_args = sorted(key for key in args if key.replace("_", "-") in _ENGINES[engine].unported)
    if topology_args:
        return f"srtctl renders {', '.join(topology_args)} from its job topology"
    return None


@dataclass(frozen=True)
class _ServerCommand:
    """The worker's argv after its entrypoint, and the runner-owned flags taken out of the role's args."""

    args: tuple[str, ...]
    stripped: tuple[str, ...]


def _server_command(engine: str, role: Mapping[str, Any]) -> _ServerCommand:
    owned = _ENGINES[engine].owned
    config = dict(role.get("args") or {})
    stripped = tuple(f"--{key}" for key in sorted(config) if key.replace("_", "-") in owned)
    config = {key: value for key, value in config.items() if key.replace("_", "-") not in owned}
    prefix: tuple[str, ...] = ()
    if engine == "sglang" and not {"enable-metrics", "enable_metrics"} & config.keys():
        prefix = ("--enable-metrics",)
    elif engine == "vllm":
        config = {key: _kv_transfer_text(key, value) for key, value in config.items()}
    elif engine == "atom":
        prefix = ("-tp", str(_atom_tensor_parallel_size(role["gpus"], config)))
    return _ServerCommand((*prefix, *srtctl_cli_args(engine, config)), stripped)


def _kv_transfer_text(key: str, value: Any) -> Any:
    if key.replace("_", "-") == "kv-transfer-config" and isinstance(value, Mapping):
        return json.dumps(dict(value))
    return value


_ATOM_STAGE_ARGS = frozenset({"pipeline-parallel-size", "prefill-context-parallel-size"})


def _atom_tensor_parallel_size(gpus: int, config: Mapping[str, Any]) -> int:
    """The worker's GPUs left for tensor parallelism after its pipeline and prefill-context stages, as srtctl
    sizes ATOM's ``-tp``."""
    stages = 1
    for key, value in config.items():
        if key.replace("_", "-") in _ATOM_STAGE_ARGS and value is not None and value is not False:
            stages *= int(value)
    if stages < 1 or gpus % stages:
        raise RecipeSourceError(f"an ATOM worker of {gpus} GPUs cannot split into {stages} stages")
    return gpus // stages


def _no_golden_reason(point: InferencexPoint, engine: str, args: Sequence[str]) -> str:
    if point.spec_decoding == "none":
        return "the launch drafts no tokens"
    if engine == "vllm" and (adaptive := adaptive_verification_reason(speculative_config(argv_flag_occurrences(args)))):
        return adaptive
    return "InferenceX simulates no acceptance length for this launch"


def _golden(
    point: InferencexPoint, selection: InferencexSelection, bundle: RepoBundle, engine: str, args: Sequence[str]
) -> SourceGolden | None:
    if point.scenario is not Scenario.AGENTIC:
        return None
    if selection.acceptance_length is None:
        return SourceGolden(None, reason=_no_golden_reason(point, engine, args))
    path = f"{GOLDEN_DIR}/{selection.golden_curve}.yaml"
    curve = GoldenCurve(repo=bundle.repo, commit=bundle.commit, path=path, sha256=bundle.sha256(path))
    return SourceGolden(selection.acceptance_length, curve=curve)


def _warnings(point: InferencexPoint, query: LaunchQuery, target: int) -> tuple[str, ...]:
    warnings = []
    if point.conc != target:
        warnings.append(f"InferenceX has no point at concurrency {target} in {point.arm_key}; took {point.conc}")
    if point.board != query.board:
        warnings.append(f"InferenceX tuned {point.entry} on {point.board}, not {query.board}")
    return tuple(warnings)


def _launch(
    point: InferencexPoint, query: LaunchQuery, snapshot: InferencexSnapshot, target: int
) -> SourceLaunch | str:
    """The launch ``point`` binds, or why this source cannot reproduce it."""
    if point.key in snapshot.rejections:
        return f"InferenceX refused it: {snapshot.rejections[point.key]}"
    selection = snapshot.selections.get(point.key)
    if selection is None:
        raise RecipeSourceError(f"the InferenceX snapshot records no selection for {point.key}")
    recipe = _with_overrides(selection.recipe, selection.golden_overrides)
    engine = _engine_type(recipe)
    unported = _unported(recipe, engine)
    if unported is not None:
        return unported
    role = recipe["roles"]["agg"]
    command = _server_command(engine, role)
    bundle = snapshot.bundle
    files = {MASTER: bundle.sha256(MASTER), RUNNERS: bundle.sha256(RUNNERS)}
    files.update({f"{PROJECT}/{path}": sha256 for path, sha256 in selection.files.items()})
    source = LaunchSource(
        kind=SourceKind.INFERENCEX,
        repo=bundle.repo,
        commit=bundle.commit,
        path=f"{PROJECT}/{point.recipe.partition(':')[0]}",
        entry=point.entry,
        locator=selection.variant,
        files=files,
        tools={"srt-slurm": snapshot.srt_slurm_commit},
    )
    client = (recipe.get("benchmark") or {}).get("env") or {}
    return SourceLaunch(
        source=source,
        framework=engine,
        entrypoint=_ENGINES[engine].entrypoint,
        topology=Topology(tp=point.tp, ep=point.ep),
        conc=point.conc,
        exact=point.conc == target,
        image=point.image,
        args=command.args,
        env={
            key: str(value) for key, value in {**(role.get("env") or {}), **(recipe.get("environment") or {})}.items()
        },
        client_env={key: str(value) for key, value in client.items() if key not in _POINT_CLIENT_ENV},
        golden=_golden(point, selection, bundle, engine, command.args),
        kv_offloading=point.kv_offloading or "none",
        stripped=command.stripped,
        warnings=_warnings(point, query, target),
    )


def _misfit(point: InferencexPoint, query: LaunchQuery) -> str | None:
    """Why ``point`` cannot serve ``query``, which it already matches on model, framework and board."""
    if point.scenario is not _SCENARIO_OF[query.workload]:
        return f"it runs {point.scenario.value}, not {_SCENARIO_OF[query.workload].value}"
    if query.workload is Workload.FIXED:
        if point.spec_decoding != "none":
            return f"it drafts with {point.spec_decoding}, and a fixed workload runs no speculative decoding"
        if (point.isl, point.osl) != (query.isl, query.osl):
            return f"it runs {point.isl}/{point.osl}, not {query.isl}/{query.osl}"
    elif point.framework not in AGENTX_FRAMEWORKS:
        return f"an AgentX launch runs {sorted(AGENTX_FRAMEWORKS)}, not {point.framework}"
    for name, asked in (("tp", query.tp), ("ep", query.ep)):
        if asked is not None and asked != getattr(point, name):
            return f"it runs {name} {getattr(point, name)}, not {asked}"
    return None


def _arm_point(points: Sequence[InferencexPoint], target: int) -> InferencexPoint | None:
    """The arm's point at ``target``, else its nearest larger one: a launch tuned for more concurrency serves less."""
    fitting = [point for point in points if point.conc >= target]
    return min(fitting, key=lambda point: point.conc) if fitting else None


def inferencex_launches(query: LaunchQuery, snapshot: InferencexSnapshot) -> tuple[SourceLaunch, ...] | SourceMiss:
    """The launch each InferenceX arm that fits ``query`` binds at ``max(query.concurrencies)``, one per arm."""
    served = [
        point
        for point in inferencex_points(snapshot.bundle)
        if (point.model, point.framework) == (query.model_id, query.framework)
        and boards_interchangeable(query.board, point.board)
    ]
    if not served:
        return SourceMiss(
            SourceKind.INFERENCEX, f"no entry serves {query.model_id} with {query.framework} on a {query.board}"
        )
    target = max(query.concurrencies)
    arms: dict[str, list[InferencexPoint]] = {}
    reasons: dict[str, str] = {}
    for point in served:
        misfit = _misfit(point, query)
        if misfit is None:
            arms.setdefault(point.arm_key, []).append(point)
        else:
            reasons.setdefault(point.arm_key, misfit)
    launches = []
    for arm_key, points in arms.items():
        point = _arm_point(points, target)
        launch = (
            f"it has no point at concurrency {target} or above"
            if point is None
            else _launch(point, query, snapshot, target)
        )
        if isinstance(launch, str):
            reasons[arm_key] = launch
        else:
            launches.append(launch)
    if launches:
        return tuple(launches)
    return SourceMiss(SourceKind.INFERENCEX, "; ".join(f"{arm}: {reason}" for arm, reason in sorted(reasons.items())))


__all__ = [
    "GOLDEN_DIR",
    "INFERENCEX_REPO",
    "MASTER",
    "PROJECT",
    "RUNNERS",
    "InferencexPoint",
    "Scenario",
    "inferencex_launches",
    "inferencex_points",
    "srtctl_cli_args",
]
