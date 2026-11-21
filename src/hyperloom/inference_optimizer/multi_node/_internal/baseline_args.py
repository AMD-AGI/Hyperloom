# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Baseline launch arguments for atomesh and sglangrouter clusters.

Sources, in priority order (Magpie ships no multi-node recipe):

1. ``manual``: operator-supplied ``--server-args`` / ``--pd-prefill-extra-args`` /
   ``--pd-decode-extra-args``. Used as given; no other source is mixed in.
2. ``previous``: the args a previous launch of the same session recorded for the
   same model and EP.
3. ``inferencex``: the model's entry in ``benchmarks/multi_node/amd_utils/models.yaml``
   of the InferenceX checkout at ``INFERENCEX_PATH``, composed the way
   ``server_sglang.sh`` composes it (no DP attention, no MTP).

No source is a configuration error: the run cannot reproduce a baseline.
"""

from __future__ import annotations

import os
import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom.common.perf_metric import agentx_enabled

SOURCE_MANUAL = "manual"
SOURCE_INFERENCEX = "inferencex"

_MODELS_YAML = Path("benchmarks") / "multi_node" / "amd_utils" / "models.yaml"
_TRANSFER_FLAG = "--disaggregation-transfer-backend"
_HICACHE_FLAG = "--enable-hierarchical-cache"


@dataclass(frozen=True)
class BaselineArgs:
    """Resolved per-role baseline arguments and where they came from."""

    source: str
    prefill: str
    decode: str
    transfer_backend: str = ""
    detail: str = ""


class BaselineArgsMissing(RuntimeError):
    """Raised when no source supplies launch arguments for the cluster."""


def _manual(env: Mapping[str, str]) -> BaselineArgs | None:
    """Operator-supplied args, or None when the operator gave none."""
    shared = env.get("INFERENCE_OPTIMIZER_SERVER_ARGS", "").strip()
    prefill = env.get("PD_PREFILL_EXTRA_ARGS", "").strip()
    decode = env.get("PD_DECODE_EXTRA_ARGS", "").strip()
    if not (shared or prefill or decode):
        return None
    return BaselineArgs(SOURCE_MANUAL, prefill, decode, detail="operator-supplied server args")


def _identity(env: Mapping[str, str]) -> dict[str, Any]:
    """The launch inputs the baseline args depend on: the model and each role's expert parallelism."""
    return {
        "model": env.get("MODEL_PATH", "").strip(),
        "prefill_ep": _role_ep(env, "prefill"),
        "decode_ep": _role_ep(env, "decode"),
    }


def _previous(previous: Mapping[str, Any] | None, env: Mapping[str, str]) -> BaselineArgs | None:
    """Args a prior launch of this session recorded for the same model and EP, so a resume keeps its baseline."""
    if not previous:
        return None
    source = str(previous.get("baseline_args_source") or "")
    if source not in (SOURCE_MANUAL, SOURCE_INFERENCEX):
        return None
    if previous.get("baseline_identity") != _identity(env):
        return None
    return BaselineArgs(
        source,
        str(previous.get("baseline_prefill_args") or ""),
        str(previous.get("baseline_decode_args") or ""),
        transfer_backend=str(previous.get("baseline_transfer_backend") or ""),
        detail="recorded by a previous launch of this session",
    )


def _inferencex_models(env: Mapping[str, str]) -> tuple[Path, Mapping[str, Any]] | str:
    """The models.yaml path and entries of the InferenceX checkout at ``INFERENCEX_PATH``, or why they are missing."""
    root = env.get("INFERENCEX_PATH", "").strip()
    if not root:
        return "INFERENCEX_PATH is unset"
    path = Path(root) / _MODELS_YAML
    if not path.is_file():
        return f"{path} does not exist"
    import yaml

    try:
        models = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        return f"cannot read {path}: {exc}"
    if not isinstance(models, Mapping):
        return f"{path} is not a mapping"
    return path, models


def _model_entry(models: Mapping[str, Any], model_path: str, agentic: bool) -> tuple[str, Mapping[str, Any]] | None:
    """The models.yaml entry for the model directory name, trying ``<name>-AgentX`` first on an agentic run, then
    ``<name>``, then ``<name>-DI``; keys match case-insensitively when no exact key exists."""
    name = Path(model_path.rstrip("/")).name
    if not name:
        return None
    candidates = ([f"{name}-AgentX"] if agentic else []) + [name, f"{name}-DI"]
    lowered = {key.lower(): key for key in models if isinstance(key, str)}
    for candidate in candidates:
        key = candidate if candidate in models else lowered.get(candidate.lower())
        if key is not None and isinstance(models[key], Mapping):
            return key, models[key]
    return None


def _split_transfer_backend(flags: str) -> tuple[list[str], str]:
    """Drop ``--disaggregation-transfer-backend`` from ``flags`` and return its value."""
    words = shlex.split(flags or "")
    for i, word in enumerate(words):
        name, sep, value = word.partition("=")
        if name == _TRANSFER_FLAG:
            end = i + 1 if sep else min(i + 2, len(words))
            return words[:i] + words[end:], value if sep else " ".join(words[i + 1 : end])
    return words, ""


def _graph_bs(spec: Any, default: str) -> list[str]:
    """Expand a ``start-end`` cuda-graph range into explicit batch sizes."""
    text = str(spec or default)
    if "-" not in text:
        text = default
    start, end = (int(part) for part in text.split("-", 1))
    return [str(n) for n in range(start, end + 1)]


def _is_true(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def _prefill_flags(prefill: Mapping[str, Any], keep_prefix_cache: bool) -> tuple[list[str], str]:
    """Prefill mode flags for the no-DP branch, plus its context length; ``keep_prefix_cache`` drops
    ``--disable-radix-cache``."""
    no_dp = prefill.get("no_dp") or {}
    flags = [
        "--mem-fraction-static",
        str(prefill.get("mem_fraction_static", 0.8)),
        "--max-running-requests",
        str(no_dp.get("max_running_requests", 128)),
        "--chunked-prefill-size",
        str(no_dp.get("chunked_prefill_size", 262144)),
    ]
    if _is_true(prefill.get("disable_cuda_graph", False)):
        flags.append("--disable-cuda-graph")
    else:
        flags.extend(["--cuda-graph-bs", *_graph_bs(no_dp.get("cuda_graph_bs_range"), "1-128")])
    if _is_true(prefill.get("disable_radix_cache", True)) and not keep_prefix_cache:
        flags.append("--disable-radix-cache")
    context = str(no_dp.get("context_length") or "")
    if context:
        flags.extend(["--context-length", context])
    if no_dp.get("max_total_tokens"):
        flags.extend(["--max-total-tokens", str(no_dp["max_total_tokens"])])
    return flags, context


def _decode_flags(decode: Mapping[str, Any], ep_on: bool, prefill_context: str) -> list[str]:
    """Decode mode flags for the EP-only or no-DP branch."""
    branch = (decode.get("ep_only") if ep_on else decode.get("no_dp")) or {}
    default_range = "1-256" if ep_on else "1-128"
    default_requests = 256 if ep_on else 128
    flags = [
        "--mem-fraction-static",
        str(decode.get("mem_fraction_static", 0.85)),
        "--max-running-requests",
        str(branch.get("max_running_requests", default_requests)),
        "--cuda-graph-bs",
        *_graph_bs(branch.get("cuda_graph_bs_range"), default_range),
    ]
    if _is_true(decode.get("prefill_round_robin_balance", True)):
        flags.append("--prefill-round-robin-balance")
    context = str(branch.get("context_length") or "") or prefill_context
    if context:
        flags.extend(["--context-length", context])
    if _is_true(decode.get("disagg_decode_enable_radix_cache", False)):
        flags.append("--disaggregation-decode-enable-radix-cache")
    return flags


def _role_ep(env: Mapping[str, str], role: str) -> bool:
    """True when the role runs expert parallel (per-role EP, else the shared EP)."""
    for name in (f"PD_{role.upper()}_EP", "EP"):
        raw = env.get(name, "").strip()
        if raw:
            try:
                return int(raw) > 1
            except ValueError:
                return False
    return False


def _inferencex(env: Mapping[str, str]) -> BaselineArgs | str:
    """Per-role args from the model's InferenceX multi-node entry, or why it supplies none."""
    model_path = env.get("MODEL_PATH", "").strip()
    found = _inferencex_models(env)
    if isinstance(found, str):
        return found
    path, models = found
    gap = f"{path} has no entry for MODEL_PATH={model_path or '<unset MODEL_PATH>'}"
    if not model_path:
        return gap
    agentic = agentx_enabled(env)
    match = _model_entry(models, model_path, agentic)
    if match is None:
        return gap
    key, entry = match
    base, transfer = _split_transfer_backend(str(entry.get("base_flags") or ""))
    ep_flags = shlex.split(str(entry.get("ep_flags") or ""))
    keep_prefix_cache = agentic or _HICACHE_FLAG in base
    prefill_mode, prefill_context = _prefill_flags(entry.get("prefill") or {}, keep_prefix_cache)
    prefill_ep = _role_ep(env, "prefill")
    decode_ep = _role_ep(env, "decode")
    prefill = base + (ep_flags if prefill_ep else []) + prefill_mode
    decode = (
        base + (ep_flags if decode_ep else []) + _decode_flags(entry.get("decode") or {}, decode_ep, prefill_context)
    )
    return BaselineArgs(
        SOURCE_INFERENCEX,
        shlex.join(prefill),
        shlex.join(decode),
        transfer_backend=transfer,
        detail=f"{path} entry {key}",
    )


def resolve_baseline_args(
    env: Mapping[str, str] | None = None,
    previous: Mapping[str, Any] | None = None,
) -> BaselineArgs:
    """Pick the baseline args source; raise ``BaselineArgsMissing`` when none applies."""
    lookup = env if env is not None else os.environ
    inferencex = _inferencex(lookup)
    for choice in (_manual(lookup), _previous(previous, lookup), inferencex):
        if isinstance(choice, BaselineArgs):
            return choice
    raise BaselineArgsMissing(
        "hyperloom cannot optimize this multi-node task: required launch arguments are missing. "
        "Pass --server-args / --pd-prefill-extra-args / --pd-decode-extra-args with the cluster's "
        f"launch arguments, or set INFERENCEX_PATH to an InferenceX checkout whose {_MODELS_YAML} lists the model "
        f"(InferenceX: {inferencex}; Magpie has no multi-node recipe)."
    )


def export_baseline_args(choice: BaselineArgs) -> None:
    """Expose resolved per-role args to the restart path; operator values are never overwritten."""
    if choice.prefill and not os.environ.get("PD_PREFILL_EXTRA_ARGS", "").strip():
        os.environ["PD_PREFILL_EXTRA_ARGS"] = choice.prefill
    if choice.decode and not os.environ.get("PD_DECODE_EXTRA_ARGS", "").strip():
        os.environ["PD_DECODE_EXTRA_ARGS"] = choice.decode
    if choice.transfer_backend and not os.environ.get("PD_TRANSFER_BACKEND", "").strip():
        os.environ["PD_TRANSFER_BACKEND"] = choice.transfer_backend


def record_baseline_args(state: dict[str, Any], choice: BaselineArgs, env: Mapping[str, str] | None = None) -> None:
    """Write the chosen source, per-role args, and the model/EP they were resolved for into the multi-node state."""
    state["baseline_identity"] = _identity(env if env is not None else os.environ)
    state["baseline_args_source"] = choice.source
    state["baseline_prefill_args"] = choice.prefill
    state["baseline_decode_args"] = choice.decode
    state["baseline_transfer_backend"] = choice.transfer_backend
