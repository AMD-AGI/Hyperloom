# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Resolve a fresh session's base launch and freeze it as ``base_launch.json``.

The base comes from the operator's ``--reference-script`` when one is given, otherwise from the first recipe source
that serves the session, otherwise from the default table. Resolving reads every source once, here; a resume reads the
frozen file instead. Scriptable workloads launch no server and AgentX sessions still take their launch from their
own recipe path, so neither resolves a base.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NoReturn

from hyperloom.common.env_safety import is_allowed_external_env_key, is_secret_shaped_env_name
from hyperloom.common.model_paths import hf_repo_id
from hyperloom.common.perf_metric import is_agentx_mode
from hyperloom.common.visible_devices import detect_gpu_count
from inference_suite import (
    LaunchQuery,
    Miss,
    SourceFetchError,
    UserFile,
    UserFileError,
    fetch_response,
    fetch_sources,
    parse_user_file,
    resolve_base,
)
from inference_suite.contracts import BaseLaunch, BaseLaunchError, Redacted, Workload

from ..framework_registry import is_scriptable
from ..session.base_launch import write_base_launch
from .multi_node import _resolve_gpus_per_node

_HTTP_OK = 200


def _refuse(message: str) -> NoReturn:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(2)


def _read_user_file(location: str, framework: str) -> UserFile:
    if location.startswith(("http://", "https://")):
        response = fetch_response(location)
        if response.status != _HTTP_OK:
            raise UserFileError(f"{location} answered HTTP {response.status}")
        content = response.body.encode("utf-8")
    else:
        content = Path(location).read_bytes()
    return parse_user_file(location, content, framework=framework)


def _model_id(args: argparse.Namespace) -> str:
    """The name every recipe source is asked for the session's model by: its Hugging Face id when it names one.

    A display name the session was given (a quantized export's) names a model no recipe serves; so does a local
    checkpoint that records no HF id, which is asked for by its directory name.
    """
    display = (getattr(args, "model_display_name", "") or "").strip()
    if display:
        return display
    model = str(getattr(args, "model", "") or "").strip().rstrip("/")
    return hf_repo_id(model) or Path(model).name or model


def _query(args: argparse.Namespace, *, framework: str, concurrencies: Sequence[int]) -> LaunchQuery:
    board = (getattr(args, "gpu_type", None) or "").strip()
    if not board:
        _refuse("the base launch is resolved for the target GPU board, and none was detected; pass --gpu-type")
    return LaunchQuery(
        model_id=_model_id(args),
        framework=framework,
        board=board,
        workload=Workload.FIXED,
        concurrencies=tuple(sorted(set(concurrencies))),
        tp=args.operator_tp,
        ep=args.operator_ep,
        isl=int(args.isl),
        osl=int(args.osl),
    )


def _session_gpus(args: argparse.Namespace) -> int:
    """The GPUs the base launch may span: every node's on a multi-node session, else the ones this host sees.

    A multi-node head need not hold a GPU itself, so its count comes from the declared shape, not from a probe.
    """
    nodes = int(getattr(args, "nodes", 1) or 1)
    if nodes >= 2:
        return nodes * _resolve_gpus_per_node(args)
    gpus = detect_gpu_count()
    if gpus < 1:
        _refuse(
            "the base launch is chosen for the GPUs this host sees, and none is visible; "
            "check ROCR_VISIBLE_DEVICES and rocm-smi"
        )
    return gpus


def _run_at_base_topology(args: argparse.Namespace, base: BaseLaunch) -> None:
    """Settle the session's TP and EP on ``base``, which chose them wherever the operator left them open.

    The executors size their GPU masks and launches from ``$TP`` and ``$EP``, and the manifest and state record them.
    """
    args.tp, args.ep = base.topology.tp, base.topology.ep
    os.environ["TP"], os.environ["EP"] = str(base.topology.tp), str(base.topology.ep)


def _redacted(base: BaseLaunch) -> BaseLaunch:
    """``base`` with every credential-shaped env value kept out of the file it is frozen in."""
    env = {
        name: Redacted(hashlib.sha256(value.encode()).hexdigest())
        if isinstance(value, str) and is_secret_shaped_env_name(name)
        else value
        for name, value in base.env.items()
    }
    return base if env == base.env else dataclasses.replace(base, env=env)


def _announce(base: BaseLaunch) -> None:
    where = base.source.locator or base.source.path or base.source.entry or base.source.repo or ""
    print(f"Base launch     : {base.source.kind.value} {where} ({len(base.args)} arg tokens, {len(base.env)} env(s))")
    for warning in base.warnings:
        print(f"WARNING: base launch: {warning}", file=sys.stderr)


def resolve_base_launch(
    args: argparse.Namespace,
    session_dir: Path,
    *,
    framework: str,
    benchmark_mode: str,
    concurrencies: Sequence[int],
) -> BaseLaunch | None:
    """Resolve the base of a fresh ``framework`` session and freeze it in ``session_dir``.

    The session then runs at the base's TP and EP, so this runs before anything records them. ``None`` when the
    session resolves no base. Exits 2 when the operator's launch file cannot serve as the base, the query is
    malformed, a single-node host sees no GPU, a source cannot be read, or no source and no default serves the
    session.
    """
    location = (getattr(args, "reference_script", None) or "").strip()
    if is_scriptable(framework):
        if location:
            _refuse(f"--reference-script is a server launch; a {framework} workload launches no server")
        return None
    if is_agentx_mode(benchmark_mode):
        return None
    try:
        query = _query(args, framework=framework, concurrencies=concurrencies)
        gpus = _session_gpus(args)
        user_file = _read_user_file(location, framework) if location else None
        found = resolve_base(query, fetch_sources(query, user_file=user_file), gpus=gpus)
    except (OSError, SourceFetchError, UserFileError, BaseLaunchError) as exc:
        _refuse(f"the base launch could not be resolved: {exc}")
    if isinstance(found, Miss):
        _refuse("no source serves this session's base launch:\n  " + "\n  ".join(found.reasons()))
    base = _redacted(found)
    write_base_launch(session_dir, base)
    _run_at_base_topology(args, base)
    _announce(base)
    return base


def reference_launch_fields(base: BaseLaunch | None) -> dict[str, Any]:
    """The ``SharedState.reference_*`` fields ``base`` seeds while their readers still read them (units 1.14, 1.15)."""
    if base is None:
        return {}
    return {
        "reference_server_args": " ".join(base.args),
        "reference_envs": {
            name: value
            for name, value in base.env.items()
            if isinstance(value, str) and is_allowed_external_env_key(name)
        },
    }


__all__ = ["reference_launch_fields", "resolve_base_launch"]
