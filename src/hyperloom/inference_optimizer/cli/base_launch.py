# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Resolve a fresh session's base launch and freeze it as ``base_launch.json``.

The base comes from the operator's ``--reference-script`` when one is given, otherwise from the first recipe source
that serves the session, otherwise, for a fixed workload only, from the default table, which also runs a fixed session
when a recipe source cannot be read. An AgentX session no recipe serves is refused. Resolving reads every source once,
here; a resume reads the frozen file instead. Scriptable workloads launch no server, so they resolve no base. The MLPerf
AgentX client drives its own traffic at a server that serves any request, so its session resolves a fixed workload's
base, which the default table always serves.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

from hyperloom.common.agentx_workload import is_mlperf_backend
from hyperloom.common.launch_credentials import credentials_in_the_clear, redact_credential_env
from hyperloom.common.model_paths import hf_repo_id
from hyperloom.common.perf_metric import is_agentx_mode
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
from inference_suite.contracts import BaseLaunch, BaseLaunchError, Workload

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


def _model_path(args: argparse.Namespace) -> str | None:
    return str(getattr(args, "model", "") or "").strip() or None


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


def _query(
    args: argparse.Namespace, *, framework: str, workload: Workload, concurrencies: Sequence[int]
) -> LaunchQuery:
    """The session's query; an AgentX trace replay names no sequence shape."""
    board = (getattr(args, "gpu_type", None) or "").strip()
    if not board:
        _refuse("the base launch is resolved for the target GPU board, and none was detected; pass --gpu-type")
    fixed = workload is Workload.FIXED
    return LaunchQuery(
        model_id=_model_id(args),
        framework=framework,
        board=board,
        workload=workload,
        concurrencies=tuple(sorted(set(concurrencies))),
        tp=args.operator_tp,
        ep=args.operator_ep,
        isl=int(args.isl) if fixed else None,
        osl=int(args.osl) if fixed else None,
    )


def _session_gpus(args: argparse.Namespace) -> int:
    """The GPUs the base launch may span: the declared shape, eight per node unless ``--gpus-per-node`` says otherwise.

    The session may be planned on a host that holds no GPU, so the count never comes from a probe of this one.
    """
    return int(args.nodes) * _resolve_gpus_per_node(args)


def _run_at_base_topology(args: argparse.Namespace, base: BaseLaunch) -> None:
    """Settle the session's TP and EP on ``base``, which chose them wherever the operator left them open.

    The executors size their GPU masks and launches from ``$TP`` and ``$EP``, and the manifest and state record them.
    """
    args.tp, args.ep = base.topology.tp, base.topology.ep
    os.environ["TP"], os.environ["EP"] = str(base.topology.tp), str(base.topology.ep)


def _without_credentials(base: BaseLaunch) -> BaseLaunch:
    """``base`` with its credential-named env frozen as sha256s; a credential it holds anywhere else exits 2."""
    redacted = redact_credential_env(base)
    found = credentials_in_the_clear(redacted)
    if found:
        _refuse(
            f"the base launch holds a credential in {', '.join(found)}; pass it through the environment instead, "
            "which is frozen as its sha256 and read from the runner host's variable of the same name"
        )
    return redacted


def _announce(base: BaseLaunch) -> None:
    where = base.source.locator or base.source.path or base.source.entry or base.source.repo or ""
    print(f"Base launch     : {base.source.kind.value} {where} ({len(base.args)} arg tokens, {len(base.env)} env(s))")


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
    malformed, or no source and no default serves the session; an AgentX session has no default.
    """
    location = (getattr(args, "reference_script", None) or "").strip()
    if is_scriptable(framework):
        if location:
            _refuse(f"--reference-script is a server launch; a {framework} workload launches no server")
        return None
    replay = is_agentx_mode(benchmark_mode) and not is_mlperf_backend()
    try:
        workload = Workload.AGENTX if replay else Workload.FIXED
        query = _query(args, framework=framework, workload=workload, concurrencies=concurrencies)
        gpus = _session_gpus(args)
        user_file = _read_user_file(location, framework) if location else None
        found = resolve_base(query, fetch_sources(query, user_file=user_file), gpus=gpus)
    except (OSError, SourceFetchError, UserFileError, BaseLaunchError) as exc:
        _refuse(f"the base launch could not be resolved: {exc}")
    if isinstance(found, Miss):
        _refuse("no source serves this session's base launch:\n  " + "\n  ".join(found.reasons()))
    base = _without_credentials(
        dataclasses.replace(found, model=dataclasses.replace(found.model, path=_model_path(args)))
    )
    write_base_launch(session_dir, base)
    _run_at_base_topology(args, base)
    _announce(base)
    return base


__all__ = ["resolve_base_launch"]
