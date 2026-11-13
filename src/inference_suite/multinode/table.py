# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The PD default table: InferenceX ``3d55815``'s multi-node model tables, composed per role.

``pd_table/models.yaml`` and ``pd_table/models_atom.yaml`` are byte copies of
``benchmarks/multi_node/amd_utils/models.yaml`` and ``models_atom.yaml`` at that commit. :func:`compose` turns one row
into the prefill and decode launches ``server_sglang.sh`` / ``server_atom.sh`` at that commit build for the same
topology, in their order: the args a script puts on the engine command after its runner-set flags, and the env it
sets for that engine. Host and fabric env (``IBDEVICES``, library paths) and benchmark-only branches (HiCache, eval
mode, simulated MTP acceptance) are not part of a launch.
"""

from __future__ import annotations

import ast
import hashlib
import operator
import shlex
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import resources
from typing import Any

import yaml

from inference_suite.frameworks import Framework

#: The InferenceX commit the table rows are copied from.
TABLE_REPO = "https://github.com/SemiAnalysisAI/InferenceX"
TABLE_COMMIT = "3d5581562f643f9bdeb8410cd924e2c70906c966"
_UPSTREAM_DIR = "benchmarks/multi_node/amd_utils"
_ON, _OFF, _AUTO, _WARNING = "true", "false", "auto", "WARNING"
_POLICY, _RANDOM = "--policy", "random"
_CONTEXT_FLAG = "--context-length"
#: The CUDA-graph batch-size ceilings the scripts default to: no-DP roles, EP-only decode and DP decode.
_NO_DP_BATCH, _EP_ONLY_BATCH, _DP_DECODE_BATCH = 128, 256, 160


@dataclass(frozen=True)
class TableFile:
    """One vendored table: its file in the package, its path upstream and the sha256 of its bytes."""

    name: str
    upstream: str
    sha256: str


SGLANG_TABLE = TableFile(
    "models.yaml", f"{_UPSTREAM_DIR}/models.yaml", "456db6f7f07400d0fe84789fbd80e3ce0617d548eb22ab296a5226112bad14a4"
)
ATOM_TABLE = TableFile(
    "models_atom.yaml",
    f"{_UPSTREAM_DIR}/models_atom.yaml",
    "434de8485ca4e6a45e6c3b9752c4040bdb9df47adf58f2470db036f5777e71a3",
)
TABLES = {Framework.SGLANG: SGLANG_TABLE, Framework.ATOM: ATOM_TABLE}


class TableError(ValueError):
    """A vendored table is damaged, or a row cannot be composed for the topology asked."""


@dataclass(frozen=True)
class RoleShape:
    """One role's parallelism as the scripts take it: TP size and the EP and DP-attention switches."""

    tp: int
    ep: bool = False
    dp: bool = False


@dataclass(frozen=True)
class TableQuery:
    """What the scripts read besides the row: the model directory name, both roles' shapes, the MTP depth, the
    largest benchmark concurrency and whether the run is agentic."""

    model_name: str
    prefill: RoleShape
    decode: RoleShape
    mtp: int = 0
    max_concurrency: int = 512
    agentic: bool = False


@dataclass(frozen=True)
class RoleArgs:
    """One role's engine args after the runner-set flags, and the env the script sets for that engine."""

    args: tuple[str, ...]
    env: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TableHit:
    """A composed table row: where it came from, both roles and the router's policy args."""

    table: TableFile
    key: str
    prefill: RoleArgs
    decode: RoleArgs
    router_args: tuple[str, ...]


def table_text(table: TableFile) -> str:
    """The vendored bytes of ``table``, refused when they are not the bytes copied from InferenceX."""
    raw = resources.files("inference_suite.multinode").joinpath("pd_table", table.name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != table.sha256:
        raise TableError(f"pd_table/{table.name} is not {table.upstream} at {TABLE_COMMIT}")
    return raw.decode("utf-8")


def _rows(table: TableFile) -> Mapping[str, Any]:
    document = yaml.safe_load(table_text(table))
    if not isinstance(document, Mapping):
        raise TableError(f"pd_table/{table.name} is not a mapping")
    return document


def _truthy(value: Any) -> bool:
    """A table switch the way the scripts test it: ``True`` or ``true``."""
    return str(value) in ("True", _ON)


def _graph_default(branch: Mapping[str, Any], ceiling: int) -> list[str]:
    """The branch's ``cuda_graph_bs_range``, else ``1-<ceiling>``, as batch sizes."""
    return _graph_range(branch.get("cuda_graph_bs_range", f"1-{ceiling}"), 1, ceiling)


def _graph_range(spec: Any, start: int, end: int) -> list[str]:
    """``parse_range`` then ``seq``: a ``start-end`` spec, else the default bounds."""
    text = str(spec)
    if "-" in text:
        low, high = text.split("-")
        start, end = int(low), int(high)
    return [str(n) for n in range(start, end + 1)]


_FORMULA_OPS: dict[type, Callable[[int, int], int]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
}


def _formula(value: Any, names: Mapping[str, int]) -> str:
    """``eval_formula``: an integer as given, else an arithmetic expression over ``names``."""
    if isinstance(value, (int, float)):
        return str(int(value))

    def walk(node: ast.AST) -> int:
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.Name) and node.id in names:
            return names[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _FORMULA_OPS:
            return _FORMULA_OPS[type(node.op)](walk(node.left), walk(node.right))
        raise TableError(f"cannot evaluate table formula {value!r}")

    return str(walk(ast.parse(str(value), mode="eval").body))


def _select(rows: Mapping[str, Any], model_name: str, agentic: bool) -> str:
    """The row key ``server_sglang.sh`` reads: ``<model>-AgentX`` or ``<model>-DI``, else the bare name."""
    variant = f"{model_name}-{'AgentX' if agentic else 'DI'}"
    if variant in rows:
        return variant
    if model_name in rows:
        return model_name
    raise TableError(f"the PD table has no row {variant} or {model_name}")


# --- sglang (server_sglang.sh) -------------------------------------------------------------------------------------

#: env.sh's SGLang block at 3d55815: the engine env every role runs with.
_SGLANG_ENV = {
    "SGLANG_USE_AITER": "1",
    "AITER_LOG_LEVEL": "ERROR",
    "SGLANG_MORI_DISPATCH_DTYPE": _AUTO,
    "SGLANG_MORI_QP_PER_TRANSFER": "4",
    "SGLANG_MORI_NUM_WORKERS": "4",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "3600",
    "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "3600",
    "SGLANG_HEALTH_CHECK_TIMEOUT": "600",
    "MORI_SHMEM_MODE": "ISOLATION",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "0",
    "SGLANG_LOG_MS": _ON,
    "SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS": "32",
    "MORI_IO_SQ_BACKOFF_TIMEOUT_US": "50000",
    "MORI_IO_QP_MAX_SEND_WR": "16384",
    "MORI_IO_QP_MAX_CQE": "32768",
    "MORI_IO_QP_MAX_SGE": "2",
    "MORI_IO_TC_DISABLE": "0",
    "MORI_EP_LAUNCH_CONFIG_MODE": "AUTO",
    "MORI_APP_LOG_LEVEL": _WARNING,
    "TORCH_NCCL_BLOCKING_WAIT": "1",
    "NCCL_BLOCKING_WAIT": "1",
}


@dataclass(frozen=True)
class _MoriProfile:
    """The MoRI sizing env.sh sets for a model: dispatch tokens per role, MoE input caps, pinned per-rank dispatch
    tokens, a pinned inter-kernel switch threshold, combine dtypes and the model's extra env."""

    dispatch: tuple[int, int] = (8192, 512)
    moe_input: tuple[str, str] | None = ("32768", "2703")
    per_rank: tuple[str, str] | None = None
    switch: int | None = None
    combine: tuple[str, str] = ("", "")
    env: Mapping[str, str] = field(default_factory=dict)


_DSV4_ENV = {
    "SGLANG_AITER_MLA_PERSIST": "0",
    "HSA_NO_SCRATCH_RECLAIM": "0",
    "MORI_IO_QP_MAX_SEND_WR": "32767",
    "SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1",
    "SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS": "1",
    "SGLANG_MORI_DISPATCH_DTYPE": _AUTO,
    "SGLANG_DEFAULT_THINKING": "1",
    "SGLANG_DSV4_REASONING_EFFORT": "high",
    "SGLANG_OPT_DEEPGEMM_HC_PRENORM": _OFF,
    "SGLANG_USE_AITER": "1",
    "SGLANG_USE_ROCM700A": "0",
    "SGLANG_OPT_USE_FUSED_COMPRESS": _ON,
    "SGLANG_HACK_FLASHMLA_BACKEND": "unified_kv_triton",
    "SGLANG_OPT_FP8_WO_A_GEMM": _OFF,
    "SGLANG_OPT_USE_JIT_INDEXER_METADATA": _OFF,
    "SGLANG_OPT_USE_TOPK_V2": _OFF,
    "SGLANG_OPT_USE_AITER_INDEXER": _ON,
    "SGLANG_OPT_USE_TILELANG_INDEXER": _OFF,
    "SGLANG_OPT_USE_TILELANG_MHC_PRE": _OFF,
    "SGLANG_OPT_USE_TILELANG_MHC_POST": _OFF,
    "SGLANG_FP8_PAGED_MQA_LOGITS_TORCH": "1",
    "SGLANG_OPT_USE_FUSED_COMPRESS_TRITON": _ON,
    "SGLANG_OPT_USE_MULTI_STREAM_OVERLAP": _OFF,
    "SGLANG_ROCM_USE_MULTI_STREAM": _OFF,
    "AITER_BF16_FP8_MOE_BOUND": "0",
    "SGLANG_EAGER_INPUT_NO_COPY": _ON,
    "SGLANG_SHARED_EXPERT_TP1": "1",
    "SGLANG_DP_SHARED_EXPERT_LOCAL": "1",
    "SGLANG_DP_USE_GATHERV": "1",
    "SGLANG_DP_USE_REDUCE_SCATTER": "1",
    "GPU_MAX_HW_QUEUES": "5",
}
#: env.sh's per-model blocks, keyed by model directory name.
_MORI_PROFILES = {
    "GLM-5-FP8": _MoriProfile(
        env={"SGLANG_ROCM_FUSED_DECODE_MLA": "0", "ROCM_QUICK_REDUCE_QUANTIZATION": "INT4", "SAFETENSORS_FAST_GPU": "1"}
    ),
    "DeepSeek-V4-Pro": _MoriProfile(
        dispatch=(8192, 64),
        moe_input=None,
        per_rank=("16384", "128"),
        switch=4096,
        combine=(_AUTO, _AUTO),
        env=_DSV4_ENV,
    ),
}
_SGLANG_DI_ROUTER = (
    *(_POLICY, _RANDOM, "--prefill-policy", _RANDOM, "--decode-policy", _RANDOM),
    *("--cb-timeout-duration-secs", "15", "--retry-max-retries", "3"),
)
_SGLANG_AGENTIC_ROUTER = (
    *(_POLICY, "consistent_hashing", "--dp-aware", "--cache-threshold", "0.3"),
    *("--balance-abs-threshold", "2", "--balance-rel-threshold", "1.1", "--disable-circuit-breaker"),
    *("--health-failure-threshold", "100", "--health-check-timeout-secs", "600", "--health-check-interval-secs", "30"),
)


def _prefill_mode(row: Mapping[str, Any], query: TableQuery, chunk_names: Mapping[str, int]) -> list[str]:
    """``PREFILL_MODE_FLAGS``."""
    prefill = row.get("prefill") or {}
    dp_on = query.prefill.dp
    branch = (prefill.get("dp") if dp_on else prefill.get("no_dp")) or {}
    if dp_on:
        graph = str(branch.get("cuda_graph_bs", "1 2 3")).split()
        running = str(branch.get("max_running_requests", 24))
    else:
        graph = _graph_default(branch, _NO_DP_BATCH)
        running = str(branch.get("max_running_requests", _NO_DP_BATCH))
    if dp_on and query.prefill.ep:
        running = str(query.max_concurrency)
    flags = ["--mem-fraction-static", str(prefill.get("mem_fraction_static", 0.8)), "--max-running-requests", running]
    flags += ["--chunked-prefill-size", _formula(branch.get("chunked_prefill_size", 262144), chunk_names)]
    if _truthy(prefill.get("disable_cuda_graph", False)):
        flags.append("--disable-cuda-graph")
    else:
        flags += ["--cuda-graph-bs", *graph]
    if _truthy(prefill.get("disable_radix_cache", True)) and not query.agentic:
        flags.append("--disable-radix-cache")
    for key, flag in (("context_length", _CONTEXT_FLAG), ("max_total_tokens", "--max-total-tokens")):
        if str(branch.get(key, "")):
            flags += [flag, str(branch[key])]
    if dp_on and _truthy(branch.get("enable_two_batch_overlap", False)):
        flags.append("--enable-two-batch-overlap")
    return flags


@dataclass(frozen=True)
class _DecodeBranch:
    """What the decode branch fixes: CUDA-graph batch sizes, max running requests and context length."""

    graph: list[str]
    running: str
    context: str


def _decode_branch(row: Mapping[str, Any], query: TableQuery) -> _DecodeBranch:
    """The decode branch ``server_sglang.sh`` takes: DP, else EP-only, else no-DP."""
    decode, shape = row.get("decode") or {}, query.decode
    if shape.dp:
        branch = decode.get("dp") or {}
        graph = _graph_default(branch, _DP_DECODE_BATCH)
        running = str(int(graph[-1]) * shape.tp)
    elif shape.ep:
        branch = decode.get("ep_only") or {}
        graph = _graph_default(branch, _EP_ONLY_BATCH)
        running = str(branch.get("max_running_requests", _EP_ONLY_BATCH))
    else:
        branch = decode.get("no_dp") or {}
        graph = _graph_default(branch, _NO_DP_BATCH)
        running = str(branch.get("max_running_requests", _NO_DP_BATCH))
    if shape.dp and shape.ep:
        running = str(query.max_concurrency)
    return _DecodeBranch(graph, running, str(branch.get("context_length", "")))


def _decode_mode(row: Mapping[str, Any], query: TableQuery, prefill_context: str) -> list[str]:
    """``DECODE_MODE_FLAGS``; decode serves prefill's context length when its branch names none."""
    decode = row.get("decode") or {}
    branch = _decode_branch(row, query)
    flags = ["--mem-fraction-static", str(decode.get("mem_fraction_static", 0.85))]
    flags += ["--max-running-requests", branch.running, "--cuda-graph-bs", *branch.graph]
    if _truthy(decode.get("prefill_round_robin_balance", True)):
        flags.append("--prefill-round-robin-balance")
    context = branch.context or prefill_context
    if context:
        flags += [_CONTEXT_FLAG, context]
    if _truthy(decode.get("disagg_decode_enable_radix_cache", False)):
        flags.append("--disaggregation-decode-enable-radix-cache")
    return flags


def _sglang_role(row: Mapping[str, Any], shape: RoleShape, mode: list[str], query: TableQuery) -> list[str]:
    """``build_server_config``: parallel args, base, EP, MTP and DP flags, then the role's mode flags."""
    args = ["--tp-size", str(shape.tp)]
    if shape.ep:
        args += ["--ep-size", str(shape.tp)]
    if shape.dp:
        args += ["--dp-size", str(shape.tp)]
    args += _row_words(row, "base_flags")
    if shape.ep:
        args += _row_words(row, "ep_flags")
    if query.mtp > 0:
        args += _row_words(row, "mtp_flags")
        args += ["--speculative-num-steps", str(query.mtp), "--speculative-num-draft-tokens", str(query.mtp + 1)]
    if shape.dp:
        args += _row_words(row, "dp_flags")
    return args + mode


def _row_words(row: Mapping[str, Any], key: str) -> list[str]:
    """A flags string of the row as the engine receives it: the script prints it inside double quotes, which drops
    one level of backslash escapes, and the engine command then splits it as shell words."""
    text = str(row.get(key, "") or "")
    for escaped, plain in (('\\"', '"'), ("\\$", "$"), ("\\`", "`"), ("\\\\", "\\")):
        text = text.replace(escaped, plain)
    return shlex.split(text)


def _sglang_env(query: TableQuery, prefill_mode: Sequence[str]) -> tuple[dict[str, str], dict[str, str]]:
    """The env each engine runs with: env.sh's block and the model's own, then the engine command's prefix."""
    profile = _MORI_PROFILES.get(query.model_name, _MoriProfile())
    prefill_tokens, decode_tokens = profile.dispatch
    switch = profile.switch if profile.switch is not None else decode_tokens * 2
    if query.decode.dp and query.decode.ep:
        decode_tokens = query.max_concurrency // query.decode.tp
        switch = decode_tokens * 2
    if query.mtp > 0:
        decode_tokens *= query.mtp + 1
    common = {**_SGLANG_ENV, **profile.env, "SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD": str(switch)}
    roles = []
    for index, tokens in enumerate((prefill_tokens, decode_tokens)):
        env = dict(common)
        if profile.combine[index]:
            env["SGLANG_MORI_COMBINE_DTYPE"] = profile.combine[index]
        if profile.moe_input is not None:
            env["SGLANG_MORI_MOE_MAX_INPUT_TOKENS"] = profile.moe_input[index]
        per_rank = profile.per_rank[index] if profile.per_rank is not None else str(tokens)
        env["SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK"] = per_rank
        roles.append(env)
    if "--enable-two-batch-overlap" in prefill_mode:
        roles[0]["MORI_ENABLE_SDMA"] = _ON
    return roles[0], roles[1]


def _compose_sglang(rows: Mapping[str, Any], query: TableQuery) -> TableHit:
    key = _select(rows, query.model_name, query.agentic)
    row = rows[key]
    if not isinstance(row, Mapping):
        raise TableError(f"PD table row {key} is not a mapping")
    dispatch = _MORI_PROFILES.get(query.model_name, _MoriProfile()).dispatch
    names = {
        "MORI_MAX_DISPATCH_TOKENS_PREFILL": dispatch[0],
        "MORI_MAX_DISPATCH_TOKENS_DECODE": dispatch[1],
        "PREFILL_TP_SIZE": query.prefill.tp,
        "DECODE_TP_SIZE": query.decode.tp,
    }
    prefill_mode = _prefill_mode(row, query, names)
    context = _flag_value(prefill_mode, _CONTEXT_FLAG)
    decode_mode = _decode_mode(row, query, context)
    prefill_env, decode_env = _sglang_env(query, prefill_mode)
    return TableHit(
        table=SGLANG_TABLE,
        key=key,
        prefill=RoleArgs(tuple(_sglang_role(row, query.prefill, prefill_mode, query)), prefill_env),
        decode=RoleArgs(tuple(_sglang_role(row, query.decode, decode_mode, query)), decode_env),
        router_args=_SGLANG_AGENTIC_ROUTER if query.agentic else _SGLANG_DI_ROUTER,
    )


def _flag_value(args: Sequence[str], flag: str) -> str:
    """The value after the last ``flag`` in ``args``, empty without one."""
    value = ""
    for index, word in enumerate(args[:-1]):
        if word == flag:
            value = args[index + 1]
    return value


# --- atom (server_atom.sh) -----------------------------------------------------------------------------------------

_ATOM_ROUTER = (
    *(_POLICY, _RANDOM, "--log-level", "info", "--disable-health-check", "--disable-circuit-breaker"),
    *("--prometheus-port", "29100"),
)
_ATOM_CUDAGRAPH_SIZES = (
    "[1,2,4,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,200,208,216,224,232,"
    "240,248,256]"
)
#: env_atom.sh at 3d55815: the engine env every role runs with (its library path is the host's).
_ATOM_ENV = {
    "PYTHONUNBUFFERED": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "SAFETENSORS_FAST_GPU": "1",
    "VLLM_LOG_LEVEL": _WARNING,
    "ATOM_LOG_LEVEL": _WARNING,
    "AITER_LOG_LEVEL": _WARNING,
    "LOG_LEVEL": _WARNING,
    "LOGLEVEL": _WARNING,
}


def _pairs(text: Any) -> dict[str, str]:
    """Space-separated ``KEY=VALUE`` pairs."""
    return dict(pair.split("=", 1) for pair in str(text or "").split())


def _atom_parallel(row: Mapping[str, Any], role: str, shape: RoleShape) -> tuple[list[str], dict[str, str]]:
    """``-tp`` and the role's DP-attention flags, and the env that DP mode exports."""
    args = ["-tp", str(shape.tp)]
    if not shape.dp:
        return args, {}
    kind = "ep_dp" if shape.ep else "tp_dp"
    flags = row.get(f"{role}_{kind}_flags", row.get(f"{kind}_flags", ""))
    return args + shlex.split(str(flags)), _pairs(row.get(f"{kind}_env"))


def _atom_tail(row: Mapping[str, Any]) -> list[str]:
    """``--max-model-len``, ``--max-num-batched-tokens`` and ``--scheduler-delay-factor`` when the row sets them."""
    tail: list[str] = []
    for key, flag in (
        ("max_model_len", "--max-model-len"),
        ("max_num_batched_tokens", "--max-num-batched-tokens"),
        ("scheduler_delay_factor", "--scheduler-delay-factor"),
    ):
        if str(row.get(key, "")):
            tail += [flag, str(row[key])]
    return tail


def _atom_role(row: Mapping[str, Any], parallel: list[str], max_num_seqs: str, quant: list[str]) -> list[str]:
    """One engine's args between ``--trust-remote-code`` and ``--kv-transfer-config``."""
    args = [*parallel, *shlex.split(str(row.get("kv_cache_flags", "")))]
    args += ["--block-size", str(row.get("block_size") or 16)]
    args += ["--gpu-memory-utilization", str(row.get("mem_frac_static") or 0.85), "--max-num-seqs", max_num_seqs]
    return [*args, *_atom_tail(row), "--no-enable_prefix_caching", *quant]


def _atom_quant(row: Mapping[str, Any], prefill_dp: bool) -> list[str]:
    """``--online_quant_config``, chosen by prefill's DP mode and given to both roles."""
    plain = row.get("online_quant_config", "")
    text = row.get("online_quant_dpa_config", plain) if prefill_dp else plain
    return ["--online_quant_config", str(text)] if text else []


def _compose_atom(rows: Mapping[str, Any], query: TableQuery) -> TableHit:
    key = query.model_name
    row = rows.get(key)
    if not isinstance(row, Mapping):
        raise TableError(f"the PD table has no atom row {key}")
    prefill_parallel, prefill_env = _atom_parallel(row, "prefill", query.prefill)
    decode_parallel, decode_env = _atom_parallel(row, "decode", query.decode)
    mtp_flags = shlex.split(str(row.get("mtp_flags", "")))
    spec = [*mtp_flags, str(query.mtp)] if query.mtp > 0 and mtp_flags else []
    quant = _atom_quant(row, query.prefill.dp)
    env = {**_ATOM_ENV, **prefill_env, **decode_env, **_pairs(row.get("env"))}
    prefill = _atom_role(row, [*prefill_parallel, *spec], str(row.get("max_num_seqs") or 256), quant)
    decode = _atom_role(row, [*decode_parallel, *spec], str(query.max_concurrency), quant)
    decode += ["--cudagraph-capture-sizes", _ATOM_CUDAGRAPH_SIZES]
    return TableHit(
        table=ATOM_TABLE,
        key=key,
        prefill=RoleArgs(tuple(prefill), dict(env)),
        decode=RoleArgs(tuple(decode), dict(env)),
        router_args=_ATOM_ROUTER,
    )


def compose(framework: Framework, query: TableQuery) -> TableHit:
    """The table row for ``query.model_name`` composed for ``query``'s topology, as the framework's script builds it.

    Raises:
        TableError: the table has no row for the model, or a row cannot be composed.
    """
    if framework is Framework.SGLANG:
        return _compose_sglang(_rows(SGLANG_TABLE), query)
    if framework is Framework.ATOM:
        return _compose_atom(_rows(ATOM_TABLE), query)
    raise TableError(f"the PD table has no {framework.value} rows")


__all__ = [
    "ATOM_TABLE",
    "SGLANG_TABLE",
    "TABLES",
    "TABLE_COMMIT",
    "TABLE_REPO",
    "RoleArgs",
    "RoleShape",
    "TableError",
    "TableFile",
    "TableHit",
    "TableQuery",
    "compose",
    "table_text",
]
