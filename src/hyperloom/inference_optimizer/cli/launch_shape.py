# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Settle the run's launch shape (workload knobs, topology, operator pins) and project it into the environment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from hyperloom.common.perf_metric import agentx_enabled as _agentx_enabled
from hyperloom.common.workload_defaults import (
    DEFAULT_ISL,
    DEFAULT_OSL,
    DEFAULT_CONC,
    DEFAULT_TP,
    DEFAULT_EP,
    DEFAULT_PRECISION,
)
from .bootstrap import parse_operator_extra_env
from .model_gate import (
    _load_model_max_position_embeddings,
    _resolve_max_model_len,
)
from .multi_node import _resolve_gpus_per_node, _resolve_mn_backend
from .parser import _positive_int_arg


def _resolve_run_max_model_len(args: argparse.Namespace) -> tuple[int, str]:
    """Resolve planning context; native AgentX serving context remains in its YAML."""
    if getattr(args, "max_model_len", None):
        return int(args.max_model_len), "--max-model-len"
    max_model_len_env = os.environ.get("MAX_MODEL_LEN", "").strip()
    if max_model_len_env:
        try:
            return _positive_int_arg(max_model_len_env), "$MAX_MODEL_LEN"
        except argparse.ArgumentTypeError as exc:
            print(
                f"ERROR: MAX_MODEL_LEN={max_model_len_env!r} is invalid: {exc}",
                file=sys.stderr,
            )
            raise SystemExit(2)
    # AgentX replays real agentic traces whose lengths come from the corpus, so ISL/OSL are meaningless placeholders
    # here (they default to 1024/1024) and ``ISL+OSL+headroom`` would pin the server at ~6k against traces reaching
    # ~1M tokens -- the corpus would then be silently reduced to whatever fits.
    if _agentx_enabled():
        native = _load_model_max_position_embeddings(str(args.model or ""))
        if native:
            return int(native), "agentx-native-context"
        # Model not on disk yet (uncached HF id).
        print(
            "NOTE: Model context is unavailable locally; using a planning estimate. "
            "Magpie resolves the AgentX serving context from its workload configuration.",
            file=sys.stderr,
        )
    return (
        _resolve_max_model_len(
            args.isl,
            args.osl,
            str(args.model or ""),
        ),
        "auto",
    )


def _detect_checkpoint_precision(model_path: str | None) -> str:
    """Detect weight precision from model config.json; returns '' when it names none."""
    if not model_path:
        return ""
    from hyperloom.inference_optimizer.model_config_utils import summarize_model_config

    summary = summarize_model_config(str(model_path))
    quant = (summary.get("quantization") or "").strip().lower()
    if quant:
        if quant.startswith("fp8"):
            return "fp8"
        if quant in ("fp4", "mxfp4", "nvfp4"):
            return "fp4"
        if quant in ("int8", "w8a8_int8"):
            return "int8"
        if quant in ("int4", "awq", "gptq"):
            return "int4"
        return quant
    dtype = (summary.get("torch_dtype") or "").strip().lower()
    _DTYPE_MAP = {
        "bfloat16": "bf16",
        "float16": "fp16",
        "float32": "fp32",
        "float8_e4m3fn": "fp8",
        "float8_e5m2": "fp8",
    }
    return _DTYPE_MAP.get(dtype, dtype) if dtype else ""


def _positive_env_int(name: str) -> int:
    """``$name`` as a positive int, or 0 so the caller falls through to the next rung of its ladder."""
    try:
        value = int(os.environ.get(name.upper(), "").strip())
    except ValueError:
        return 0
    return value if value > 0 else 0


def _seed_env_from_resumed_state(state: Any, pins: dict[str, str]) -> None:
    """Put the session's recorded workload in the environment, so a resume outranks whatever the shell exported.

    Hyperloom's own repro scripts ``export TP``/``CONC``/``ISL``/``OSL``, and sourcing one before a resume must not
    silently re-measure the session at another shape. ``pins`` -- this resume's ``--extra-env``, or, when it
    passes none, the pins persisted from the launch -- is skipped, so changing the workload stays an explicit act.
    """
    for name in ("ISL", "OSL", "CONC", "TP", "EP", "MAX_MODEL_LEN"):
        recorded = getattr(state, name.lower(), 0)
        if recorded and name not in pins:
            os.environ[name] = str(recorded)


def _resolve_workload_knobs(
    args: argparse.Namespace,
    state: Any | None = None,
) -> None:
    """Fill the unset numeric workload knobs on ``args``: flag, environment, resumed state, default.

    The environment rung carries the ``--extra-env`` pins :func:`_export_operator_launch_shape` has already written,
    and on a resume also the recorded workload :func:`_seed_env_from_resumed_state` writes over any shell export -- a
    resumed value therefore arrives through that rung rather than through ``state``. Precision has its own function
    because it reads the checkpoint, so it cannot run this early.
    """
    int_knobs = (
        ("isl", DEFAULT_ISL),
        ("osl", DEFAULT_OSL),
        ("conc", DEFAULT_CONC),
        ("tp", DEFAULT_TP),
        ("ep", DEFAULT_EP),
    )
    for name, default in int_knobs:
        val = getattr(args, name, None)
        if val is None:
            from_env = _positive_env_int(name)
            persisted = int(getattr(state, name, 0) or 0) if state is not None else 0
            val = from_env or (persisted if persisted > 0 else default)
        setattr(args, name, int(val))


def _resolve_precision(args: argparse.Namespace, state: Any | None = None) -> None:
    """Fill ``args.precision``: flag, environment, resumed state, checkpoint detection, default.

    Must run after the model is resolved: the detection rung reads the checkpoint's ``config.json``, and finding
    nothing there settles ``args.precision`` on the default with no later call able to correct it. A ``PRECISION``
    reaching this function through the environment is treated as an explicit flag, mismatch warning included.
    """
    precision = getattr(args, "precision", None)
    if not precision:
        precision = os.environ.get("PRECISION", "").strip()
    if not precision:
        persisted = (getattr(state, "precision", "") or "").strip() if state is not None else ""
        if persisted:
            precision = persisted
        else:
            detected = _detect_checkpoint_precision(getattr(args, "model", None))
            precision = detected or DEFAULT_PRECISION
    else:
        detected = _detect_checkpoint_precision(getattr(args, "model", None))
        if detected and detected != precision:
            print(
                f"WARN: --precision={precision!r} but checkpoint dtype is {detected!r}; "
                "using --precision flag as specified.",
                file=sys.stderr,
            )
    args.precision = precision


def _export_workload_envs_for_optimize(
    args: argparse.Namespace,
    *,
    tp_resolved: int,
    ep_resolved: int,
) -> None:
    """Project resolved workload knobs (TP/CONC/EP) into env for downstream Magpie YAMLs."""
    os.environ["TP"] = str(max(1, int(tp_resolved or 1)))
    os.environ["CONC"] = str(max(1, int(getattr(args, "conc", DEFAULT_CONC) or DEFAULT_CONC)))
    os.environ["EP"] = str(max(1, int(ep_resolved or 1)))


def _enforce_topology_gates(*, nodes: int, gpus_per_node: int, tp: int, ep: int) -> None:
    """Fail fast on a TP/EP shape the cluster cannot place, rather than on a cryptic launcher crash mid-cold-start.

    Multi-node only: a single node places whatever its own GPU count allows and the launcher reports that itself.
    """
    if nodes < 2:
        return
    total_gpus = nodes * gpus_per_node
    # Gate 1: total cluster GPUs (nodes*gpus_per_node) must hold the model's TP shards.
    if total_gpus < tp:
        print(
            f"ERROR: TP={tp} exceeds total GPU count "
            f"({nodes} nodes * {gpus_per_node} "
            f"gpus_per_node = {total_gpus}). Either lower --tp, raise "
            "--nodes, or use a larger --gpus-per-node pod "
            "template.",
            file=sys.stderr,
        )
        sys.exit(2)
    # Gate 2: EP cannot exceed TP (can't place more expert shards than ranks); fail before bootstrap.
    if ep > tp:
        print(
            f"ERROR: EP={ep} > TP={tp}. Expert-parallel "
            "size must be <= tensor-parallel size. Either lower --ep or "
            "raise --tp.",
            file=sys.stderr,
        )
        sys.exit(2)


def _export_operator_launch_shape(
    *,
    server_args: str,
    extra_env: dict[str, str],
) -> None:
    """Project the operator's ``--server-args`` / ``--extra-env`` into env.

    Every pin is exported under its own name, with no exceptions: Hyperloom's control variables are read with a
    bare ``os.environ.get``, so a pin the JSON blob alone carries is invisible to all of them. The blob is kept
    because it reaches ``state.json`` and names the previous launch's pins, which a resume dropping one must unset.
    """
    from hyperloom.common.env_safety import OPERATOR_EXTRA_ENV_VAR, operator_extra_env

    if server_args:
        os.environ["INFERENCE_OPTIMIZER_SERVER_ARGS"] = server_args
    else:
        os.environ.pop("INFERENCE_OPTIMIZER_SERVER_ARGS", None)
    # Read before the blob is rewritten: it names what the previous launch exported, which a resume dropping a pin
    # -- or a second session in the same shell -- has to unset rather than leave behind.
    for name in operator_extra_env():
        if name not in extra_env:
            os.environ.pop(name, None)
    os.environ.update(extra_env)
    if extra_env:
        os.environ[OPERATOR_EXTRA_ENV_VAR] = json.dumps(extra_env)
    else:
        os.environ.pop(OPERATOR_EXTRA_ENV_VAR, None)


def _export_predictor_settings(args: argparse.Namespace) -> None:
    """Project ``--primatune-endpoint`` / ``--primatune-mode`` into the env the predictor reads.

    An absent flag leaves the variable alone, so a value exported in the shell
    still applies. Nothing is persisted: a ``--resume-from`` relaunch needs the
    flags again unless the shell carries the variables.
    """
    from hyperloom.common import predictor_config

    endpoint = str(getattr(args, "primatune_endpoint", "") or "").strip()
    if endpoint:
        os.environ[predictor_config.ENV_ENDPOINT] = endpoint
    mode = str(getattr(args, "primatune_mode", "") or "").strip()
    if mode:
        os.environ[predictor_config.ENV_MODE] = mode
    conf = predictor_config.load()
    if conf.enabled:
        print(f"  predictor          : {conf.mode} at {conf.endpoint}")


def _resolve_run_topology(args: argparse.Namespace) -> int:
    """Settle this run's launch shape, and return the node count the caller gates on.

    Everything else it settles -- the operator's pins, TP/EP, ISL/OSL/CONC, the
    PD split, ``SKIP_VARIANTS`` -- leaves through ``os.environ``, which is how
    the subprocess executors that inherit it read these. Only the node count has
    a caller in this process, so only it is returned. A resume runs its own
    ladder against the session's recorded values and skips the projection.
    """
    # Surface --nodes (CLI flag wins) before _preflight runs.
    nodes_resolved = max(1, int(args.nodes))
    # The operator's pins have to be in the environment before the ladder below reads them.
    _export_operator_launch_shape(
        server_args=str(getattr(args, "server_args", "") or "").strip(),
        extra_env=parse_operator_extra_env(args),
    )
    # Settle the knobs before the gates, the projection and ``_preflight`` derive from them. A resume runs its own
    # ladder once the state is loaded.
    _is_resume = bool(getattr(args, "resume_from", None))
    if not _is_resume:
        _resolve_workload_knobs(args)
    tp_resolved = max(1, int(getattr(args, "tp", 1) or 1))
    ep_resolved = max(1, int(getattr(args, "ep", 1) or 1))
    gpus_per_node_resolved = _resolve_gpus_per_node(args)

    # Project before ``_preflight``: ``check_gpu_visibility`` reads ``$TP`` to warn when the run asks for more
    # shards than rocm-smi can see.
    if not _is_resume:
        _export_workload_envs_for_optimize(args, tp_resolved=tp_resolved, ep_resolved=ep_resolved)

    _enforce_topology_gates(
        nodes=nodes_resolved,
        gpus_per_node=gpus_per_node_resolved,
        tp=tp_resolved,
        ep=ep_resolved,
    )

    os.environ["INFERENCE_OPTIMIZER_NODES"] = str(nodes_resolved)
    # Multi-node topology handoff: export the CLI-flag-resolved backend / gpus-per-node so downstream subprocesses
    # (kernel agent, benchmark, KB topology, state synthesis) read a single stable source.
    if nodes_resolved >= 2:
        os.environ["INFERENCE_OPTIMIZER_GPUS_PER_NODE"] = str(gpus_per_node_resolved)
        os.environ["INFERENCE_OPTIMIZER_MN_BACKEND"] = _resolve_mn_backend(args)
    _export_predictor_settings(args)
    # The partition shape is deliberately NOT exported here.

    # User-declared grid skip list; re-export so subprocess executors inherit it (empty clears stale values).
    # Not pinnable: it is the policy for one run, not part of the session's measurement contract, so a resume takes
    # it from the flag it was passed rather than from what the original launch pinned.
    skip_variants_resolved = (getattr(args, "skip_variants", "") or "").strip()
    os.environ["SKIP_VARIANTS"] = skip_variants_resolved
    # Surface PD_* knobs for executors; empty means "resolve from state.json", pd_mode always exported.
    pd_mode = (getattr(args, "pd_mode", "") or "aggregated").lower()
    if pd_mode == "disaggregated" and nodes_resolved < 2:
        # PD disaggregation needs >=2 nodes (separate prefill + decode pods); fail at parse time.
        print(
            f"ERROR: --pd-mode disaggregated requires --nodes >= 2 "
            f"(got --nodes {nodes_resolved}). PD splits the cluster "
            "into prefill + decode groups; a single pod cannot host "
            "both. Either drop --pd-mode (defaults to aggregated) or "
            "raise --nodes.",
            file=sys.stderr,
        )
        sys.exit(2)
    os.environ["PD_MODE"] = pd_mode
    if pd_mode == "disaggregated":
        for cli_attr, env_key in (
            ("pd_prefill_nodes", "PD_PREFILL_NODES"),
            ("pd_decode_nodes", "PD_DECODE_NODES"),
            ("pd_prefill_tp", "PD_PREFILL_TP"),
            ("pd_decode_tp", "PD_DECODE_TP"),
            ("pd_prefill_ep", "PD_PREFILL_EP"),
            ("pd_decode_ep", "PD_DECODE_EP"),
        ):
            v = int(getattr(args, cli_attr, 0) or 0)
            if v > 0:
                os.environ[env_key] = str(v)
        for cli_attr, env_key in (
            ("pd_transfer_backend", "PD_TRANSFER_BACKEND"),
            ("pd_ib_device", "PD_IB_DEVICE"),
            ("pd_prefill_extra_args", "PD_PREFILL_EXTRA_ARGS"),
            ("pd_decode_extra_args", "PD_DECODE_EXTRA_ARGS"),
        ):
            v = (getattr(args, cli_attr, "") or "").strip()
            if v:
                os.environ[env_key] = v
    return nodes_resolved


def _refuse_multinode_partition_shape(nodes: int, mode: str) -> None:
    """Warn that a multi-node session records no partition shape; exit 2 when a mode was asserted."""
    # Unconditional: the misleading record is produced by observing at all, not by asking for a mode.
    print(
        f"WARN: this session has --nodes {nodes}, and a compute-partition shape "
        f"describes one card. The benchmark node's topology cannot be read from here, "
        f"so no shape is recorded for this session.",
        file=sys.stderr,
    )
    if mode:
        print(
            f"ERROR: --compute-partition-mode {mode} cannot be checked on a "
            f"--nodes {nodes} session: the assertion is about the benchmark node's "
            f"card, which this process cannot read. An unverifiable assertion is not "
            f"a satisfied one. Drop the flag to run multi-node.",
            file=sys.stderr,
        )
        sys.exit(2)
