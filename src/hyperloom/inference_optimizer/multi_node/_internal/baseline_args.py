# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Baseline launch arguments for atom and sglangrouter clusters.

Sources, in priority order:

1. ``manual``: operator-supplied ``--server-args`` / ``--pd-prefill-extra-args`` /
   ``--pd-decode-extra-args``. Used as given; no other source is mixed in.
2. ``magpie``: a Magpie multi-node recipe. The pinned Magpie revision ships none.
3. ``inferencex``: the model's entry in InferenceX
   ``benchmarks/multi_node/amd_utils/models.yaml``, composed the way
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

SOURCE_MANUAL = "manual"
SOURCE_MAGPIE = "magpie"
SOURCE_INFERENCEX = "inferencex"

_MODELS_YAML = Path("benchmarks") / "multi_node" / "amd_utils" / "models.yaml"
_TRANSFER_FLAG = "--disaggregation-transfer-backend"


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


def _previous(previous: Mapping[str, Any] | None) -> BaselineArgs | None:
    """Args a prior launch of this session recorded, so a resume keeps its baseline."""
    if not previous:
        return None
    source = str(previous.get("baseline_args_source") or "")
    if source not in (SOURCE_MANUAL, SOURCE_MAGPIE, SOURCE_INFERENCEX):
        return None
    return BaselineArgs(
        source,
        str(previous.get("baseline_prefill_args") or ""),
        str(previous.get("baseline_decode_args") or ""),
        transfer_backend=str(previous.get("baseline_transfer_backend") or ""),
        detail="recorded by a previous launch of this session",
    )


def _magpie() -> BaselineArgs | None:
    """Magpie multi-node recipe. The pinned Magpie rejects multi-node recipes, so there is none."""
    return None


def _inferencex_root(env: Mapping[str, str]) -> Path | None:
    """The InferenceX checkout preflight selected, or None."""
    explicit = env.get("INFERENCEX_PATH", "").strip()
    if explicit:
        return Path(explicit)
    try:
        from ...inferencex_checkout import detect
    except ImportError:
        return None
    found = detect()
    return Path(found) if found else None


def _model_entry(models: Mapping[str, Any], model_path: str) -> tuple[str, Mapping[str, Any]] | None:
    """The models.yaml entry whose key matches the model directory name."""
    name = Path(model_path.rstrip("/")).name
    if not name:
        return None
    entry = models.get(name)
    if isinstance(entry, Mapping):
        return name, entry
    lowered = name.lower()
    for key, value in models.items():
        if isinstance(key, str) and key.lower() == lowered and isinstance(value, Mapping):
            return key, value
    return None


def _split_transfer_backend(flags: str) -> tuple[list[str], str]:
    """Drop ``--disaggregation-transfer-backend`` from ``flags`` and return its value."""
    words = shlex.split(flags or "")
    out: list[str] = []
    value = ""
    i = 0
    while i < len(words):
        word = words[i]
        if word == _TRANSFER_FLAG:
            value = words[i + 1] if i + 1 < len(words) else ""
            i += 2
            continue
        if word.startswith(_TRANSFER_FLAG + "="):
            value = word.split("=", 1)[1]
            i += 1
            continue
        out.append(word)
        i += 1
    return out, value


def _graph_bs(spec: Any, default: str) -> list[str]:
    """Expand a ``start-end`` cuda-graph range into explicit batch sizes."""
    text = str(spec or default)
    if "-" not in text:
        text = default
    start, end = (int(part) for part in text.split("-", 1))
    return [str(n) for n in range(start, end + 1)]


def _is_true(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def _prefill_flags(prefill: Mapping[str, Any]) -> tuple[list[str], str]:
    """Prefill mode flags for the no-DP branch, plus its context length."""
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
    if _is_true(prefill.get("disable_radix_cache", True)):
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


def _inferencex(env: Mapping[str, str]) -> BaselineArgs | None:
    """Per-role args from the model's InferenceX multi-node entry, or None."""
    model_path = env.get("MODEL_PATH", "").strip()
    root = _inferencex_root(env)
    if not model_path or root is None:
        return None
    path = root / _MODELS_YAML
    if not path.is_file():
        return None
    import yaml

    try:
        models = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(models, Mapping):
        return None
    match = _model_entry(models, model_path)
    if match is None:
        return None
    key, entry = match
    base, transfer = _split_transfer_backend(str(entry.get("base_flags") or ""))
    ep_flags = shlex.split(str(entry.get("ep_flags") or ""))
    prefill_mode, prefill_context = _prefill_flags(entry.get("prefill") or {})
    prefill_ep = _role_ep(env, "prefill")
    decode_ep = _role_ep(env, "decode")
    prefill = base + (ep_flags if prefill_ep else []) + prefill_mode
    decode = base + (ep_flags if decode_ep else []) + _decode_flags(entry.get("decode") or {}, decode_ep, prefill_context)
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
    for choice in (_manual(lookup), _previous(previous), _magpie(), _inferencex(lookup)):
        if choice is not None:
            return choice
    model = lookup.get("MODEL_PATH", "").strip() or "<unset MODEL_PATH>"
    raise BaselineArgsMissing(
        "hyperloom cannot optimize this multi-node task: required launch arguments are missing. "
        "Pass --server-args / --pd-prefill-extra-args / --pd-decode-extra-args with the cluster's "
        f"launch arguments, or use a model listed in InferenceX {_MODELS_YAML} "
        f"(MODEL_PATH={model} has no entry; Magpie has no multi-node recipe)."
    )


def export_baseline_args(choice: BaselineArgs) -> None:
    """Expose resolved per-role args to the restart path; operator values are never overwritten."""
    if choice.prefill and not os.environ.get("PD_PREFILL_EXTRA_ARGS", "").strip():
        os.environ["PD_PREFILL_EXTRA_ARGS"] = choice.prefill
    if choice.decode and not os.environ.get("PD_DECODE_EXTRA_ARGS", "").strip():
        os.environ["PD_DECODE_EXTRA_ARGS"] = choice.decode
    if choice.transfer_backend and not os.environ.get("PD_TRANSFER_BACKEND", "").strip():
        os.environ["PD_TRANSFER_BACKEND"] = choice.transfer_backend


def record_baseline_args(state: dict[str, Any], choice: BaselineArgs) -> None:
    """Write the chosen source and per-role args into the multi-node state."""
    state["baseline_args_source"] = choice.source
    state["baseline_prefill_args"] = choice.prefill
    state["baseline_decode_args"] = choice.decode
    state["baseline_transfer_backend"] = choice.transfer_backend
