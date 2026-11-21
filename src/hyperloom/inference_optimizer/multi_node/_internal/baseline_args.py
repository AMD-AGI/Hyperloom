# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Baseline launch arguments for atomesh and sglangrouter clusters.

Sources, in priority order:

1. ``manual``: operator-supplied ``--server-args`` / ``--pd-prefill-extra-args`` /
   ``--pd-decode-extra-args``. Used as given; no other source is mixed in.
2. ``previous``: the args a previous launch of the same session recorded for the
   same model and EP.

No source is a configuration error: the run cannot reproduce a baseline.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

SOURCE_MANUAL = "manual"


@dataclass(frozen=True)
class BaselineArgs:
    """Resolved per-role baseline arguments and where they came from."""

    source: str
    prefill: str
    decode: str
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
    if str(previous.get("baseline_args_source") or "") != SOURCE_MANUAL:
        return None
    if previous.get("baseline_identity") != _identity(env):
        return None
    return BaselineArgs(
        SOURCE_MANUAL,
        str(previous.get("baseline_prefill_args") or ""),
        str(previous.get("baseline_decode_args") or ""),
        detail="recorded by a previous launch of this session",
    )


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


def resolve_baseline_args(
    env: Mapping[str, str] | None = None,
    previous: Mapping[str, Any] | None = None,
) -> BaselineArgs:
    """Pick the baseline args source; raise ``BaselineArgsMissing`` when none applies."""
    lookup = env if env is not None else os.environ
    for choice in (_manual(lookup), _previous(previous, lookup)):
        if choice is not None:
            return choice
    raise BaselineArgsMissing(
        "hyperloom cannot optimize this multi-node task: required launch arguments are missing. "
        "Pass --server-args / --pd-prefill-extra-args / --pd-decode-extra-args with the cluster's "
        "launch arguments, or resume a session whose previous launch recorded them for the same model and EP."
    )


def export_baseline_args(choice: BaselineArgs) -> None:
    """Expose resolved per-role args to the restart path; operator values are never overwritten."""
    if choice.prefill and not os.environ.get("PD_PREFILL_EXTRA_ARGS", "").strip():
        os.environ["PD_PREFILL_EXTRA_ARGS"] = choice.prefill
    if choice.decode and not os.environ.get("PD_DECODE_EXTRA_ARGS", "").strip():
        os.environ["PD_DECODE_EXTRA_ARGS"] = choice.decode


def record_baseline_args(state: dict[str, Any], choice: BaselineArgs, env: Mapping[str, str] | None = None) -> None:
    """Write the chosen source, per-role args, and the model/EP they were resolved for into the multi-node state."""
    state["baseline_identity"] = _identity(env if env is not None else os.environ)
    state["baseline_args_source"] = choice.source
    state["baseline_prefill_args"] = choice.prefill
    state["baseline_decode_args"] = choice.decode
