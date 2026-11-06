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
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NoReturn

from hyperloom.common.env_safety import is_allowed_external_env_key, is_secret_shaped_env_name
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
from inference_suite.contracts import BaseLaunch, BaseLaunchError, Redacted, Workload

from ..framework_registry import is_scriptable
from ..session.base_launch import write_base_launch

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


def _is_hub_id(text: str) -> bool:
    return text.count("/") == 1 and not text.startswith((".", "/", "~")) and not Path(text).exists()


def _config_hub_id(model_dir: Path) -> str | None:
    try:
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    name = config.get("_name_or_path") if isinstance(config, dict) else None
    return name if isinstance(name, str) and _is_hub_id(name) else None


def _model_id(args: argparse.Namespace) -> str:
    """The Hugging Face id every recipe source names the session's model by.

    A ``--model`` that is no local directory is that id already; a local checkpoint names it in its HF cache path or
    its ``config.json``. A display name the session was given (a quantized export's) names a model no recipe serves.
    """
    display = (getattr(args, "model_display_name", "") or "").strip()
    if display:
        return display
    model = str(getattr(args, "model", "") or "").strip().rstrip("/")
    path = Path(model).expanduser()
    if not path.is_dir():
        return model
    cached = next((part.removeprefix("models--") for part in path.parts if part.startswith("models--")), None)
    if cached:
        return cached.replace("--", "/", 1)
    return _config_hub_id(path) or path.name


def _query(args: argparse.Namespace, *, framework: str, concurrencies: Sequence[int]) -> LaunchQuery:
    board = (getattr(args, "gpu_type", None) or "").strip()
    if not board:
        _refuse("the base launch is resolved for the target GPU board, and none was detected; pass --gpu-type")
    # The workload knobs are settled, and the session's TP and EP exported, before the session is seeded; the base
    # has to run that topology.
    return LaunchQuery(
        model_id=_model_id(args),
        framework=framework,
        board=board,
        workload=Workload.FIXED,
        concurrencies=tuple(sorted(set(concurrencies))),
        tp=int(args.tp),
        ep=int(args.ep),
        isl=int(args.isl),
        osl=int(args.osl),
    )


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

    ``None`` when the session resolves no base. Exits 2 when the operator's launch file cannot serve as the base,
    the query is malformed, a source cannot be read, or no source and no default serves the session.
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
        user_file = _read_user_file(location, framework) if location else None
        found = resolve_base(query, fetch_sources(query, user_file=user_file), gpus=int(args.tp))
    except (OSError, SourceFetchError, UserFileError, BaseLaunchError) as exc:
        _refuse(f"the base launch could not be resolved: {exc}")
    if isinstance(found, Miss):
        _refuse("no source serves this session's base launch:\n  " + "\n  ".join(found.reasons()))
    base = _redacted(found)
    write_base_launch(session_dir, base)
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
