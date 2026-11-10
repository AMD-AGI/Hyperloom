# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Resolve a fresh session's base launch and freeze it as ``base_launch.json``.

The base comes from the operator's ``--reference-script`` when one is given, otherwise from the first recipe source
that serves the session, otherwise, for a fixed workload only, from the default table, which also runs a fixed session
when a recipe source cannot be read. An AgentX session no recipe serves is refused, and so is a fixed one under
``--inferencex-commit``, which never takes the default table. Resolving reads every source once,
here, InferenceX at ``--inferencex-commit`` when one is given; a resume reads the frozen file instead. A scriptable
session launches no server: its base is the script it runs, the operator's from ``--benchmark-scripts-dir`` when it
holds one, else the one its framework ships. The MLPerf AgentX client drives its own traffic at a server that serves
any request, so its session resolves a fixed workload's base, which the default table always serves.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from hyperloom.common.agentx_workload import is_mlperf_backend
from hyperloom.common.env_safety import BLOCKED_EXTERNAL_ENV_NAMES, is_credential_name
from hyperloom.common.launch_credentials import credentials_in_the_clear, redact_credential_env
from hyperloom.common.model_paths import hf_repo_id
from hyperloom.common.perf_metric import is_agentx_mode
from hyperloom.common.visible_devices import detect_gpu_count
from inference_suite import (
    InferencexCommitRefused,
    LaunchQuery,
    Miss,
    SourceFetchError,
    UserFile,
    UserFileError,
    fetch_response,
    fetch_sources,
    parse_user_file,
    resolve_base,
    scriptable_base,
)
from inference_suite.contracts import BaseLaunch, BaseLaunchError, LaunchModel, Topology, Workload

from ..framework_paths import operator_benchmark_script
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


@dataclass(frozen=True)
class SessionLaunch:
    """What a session's base launch is resolved for: the model it serves (a path or a hub id) and the name recipes are
    asked for it by when it was given one, the framework, the target board, the workload and its concurrencies, the
    GPUs the launch may span, the TP and EP the operator fixed (``None`` leaves them to the source), a fixed workload's
    sequence shape, the operator's launch file (``""`` for none), and the commit InferenceX is read at (``""`` for the
    head of its ``main``)."""

    model: str
    framework: str
    board: str
    workload: Workload
    concurrencies: tuple[int, ...]
    gpus: int
    tp: int | None = None
    ep: int | None = None
    isl: int | None = None
    osl: int | None = None
    model_display_name: str = ""
    reference: str = ""
    inferencex_commit: str = ""

    @property
    def model_path(self) -> str | None:
        return self.model.strip() or None

    @property
    def model_id(self) -> str:
        """The name every recipe source is asked for the model by: its Hugging Face id when it names one.

        A display name the session was given (a quantized export's) names a model no recipe serves; so does a local
        checkpoint that records no HF id, which is asked for by its directory name.
        """
        if self.model_display_name.strip():
            return self.model_display_name.strip()
        model = self.model.strip().rstrip("/")
        return hf_repo_id(model) or Path(model).name or model

    def query(self) -> LaunchQuery:
        """The session's query; raises :class:`BaseLaunchError` when it is malformed."""
        return LaunchQuery(
            model_id=self.model_id,
            framework=self.framework,
            board=self.board,
            workload=self.workload,
            concurrencies=tuple(sorted(set(self.concurrencies))),
            tp=self.tp,
            ep=self.ep,
            isl=self.isl,
            osl=self.osl,
        )


class BaseLaunchUnresolved(Exception):
    """No base launch can be resolved for a session; the message says why."""


def _session_launch(
    args: argparse.Namespace, *, framework: str, workload: Workload, concurrencies: Sequence[int]
) -> SessionLaunch:
    """The command line's session; an AgentX trace replay and a script name no sequence shape."""
    board = (getattr(args, "gpu_type", None) or "").strip()
    if not board:
        _refuse("the base launch is resolved for the target GPU board, and none was detected; pass --gpu-type")
    fixed = workload is Workload.FIXED
    return SessionLaunch(
        model=str(getattr(args, "model", "") or ""),
        framework=framework,
        board=board,
        workload=workload,
        concurrencies=tuple(concurrencies),
        gpus=_session_gpus(args),
        tp=args.operator_tp,
        ep=args.operator_ep,
        isl=int(args.isl) if fixed else None,
        osl=int(args.osl) if fixed else None,
        model_display_name=getattr(args, "model_display_name", "") or "",
        reference=(getattr(args, "reference_script", None) or "").strip(),
        inferencex_commit=(getattr(args, "inferencex_commit", None) or "").strip(),
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


def _refused_env(base: BaseLaunch) -> list[str]:
    """The env names of ``base`` no launch may take from a source: a loader, a path, a traffic route or a workload key.
    A credential among them is frozen as its sha256 and filled by the runner, so it is not one."""
    layers = {**base.env, **(base.golden_al.env if base.golden_al is not None else {})}
    return [f"env {name}" for name in layers if name in BLOCKED_EXTERNAL_ENV_NAMES and not is_credential_name(name)]


def _frozen_form(base: BaseLaunch) -> BaseLaunch:
    """``base`` with its credential-named env frozen as sha256s; raises :class:`BaseLaunchUnresolved` for a credential
    it holds anywhere else, or an env no launch may take from a source, which every server of the session would run."""
    redacted = redact_credential_env(base)
    found = credentials_in_the_clear(redacted)
    if found:
        raise BaseLaunchUnresolved(
            f"the base launch holds a credential in {', '.join(found)}; pass it through the environment instead, "
            "which is frozen as its sha256 and read from the runner host's variable of the same name"
        )
    refused = _refused_env(redacted)
    if refused:
        raise BaseLaunchUnresolved(
            f"the base launch sets {', '.join(refused)}, which the session or the runner owns or no launch may take "
            "from a source; drop it from the launch"
        )
    return redacted


def _announce(base: BaseLaunch) -> None:
    where = base.source.locator or base.source.path or base.source.entry or base.source.repo or ""
    print(f"Base launch     : {base.source.kind.value} {where} ({len(base.args)} arg tokens, {len(base.env)} env(s))")


def _scriptable_base(launch: SessionLaunch, args: argparse.Namespace) -> BaseLaunch:
    """The base of a scriptable session: its script, at the TP and EP the session resolved."""
    query = launch.query()
    script = operator_benchmark_script(launch.framework, query.board.lower())
    model = LaunchModel(launch.model_id, launch.model_path)
    return scriptable_base(query, script, model, Topology(tp=int(args.tp), ep=int(args.ep)))


def resolve_server_base(launch: SessionLaunch) -> BaseLaunch:
    """The base launch of a server session: the operator's launch file when it names one, else the first recipe source
    that serves it, else, for a fixed workload with InferenceX unpinned, the default table; its credential-named env
    frozen as sha256s.

    Raises:
        BaseLaunchUnresolved: The launch file cannot serve as the base, the query is malformed, the InferenceX commit
            is refused, no source and no default serves the session (an AgentX session, or one under a pinned
            InferenceX, has no default), or the base holds a credential outside its env or an env no launch may take
            from a source.
    """
    try:
        query = launch.query()
        user_file = _read_user_file(launch.reference, launch.framework) if launch.reference else None
        snapshot = fetch_sources(query, user_file=user_file, inferencex_commit=launch.inferencex_commit)
        found = resolve_base(query, snapshot, gpus=launch.gpus)
    except (OSError, SourceFetchError, UserFileError, BaseLaunchError, InferencexCommitRefused) as exc:
        raise BaseLaunchUnresolved(f"the base launch could not be resolved: {exc}") from exc
    if isinstance(found, Miss):
        raise BaseLaunchUnresolved("no source serves this session's base launch:\n  " + "\n  ".join(found.reasons()))
    return _frozen_form(dataclasses.replace(found, model=dataclasses.replace(found.model, path=launch.model_path)))


def resolve_base_launch(
    args: argparse.Namespace,
    session_dir: Path,
    *,
    framework: str,
    benchmark_mode: str,
    concurrencies: Sequence[int],
) -> BaseLaunch:
    """Resolve the base of a fresh ``framework`` session and freeze it in ``session_dir``.

    The session then runs at the base's TP and EP, so this runs before anything records them. Exits 2 when the
    operator's launch file cannot serve as the base, the query is malformed, ``--inferencex-commit`` is refused, no
    source and no default serves the session (an AgentX session has no default), a scriptable session has no script to run, or a single-node session's
    base needs more GPUs than this host sees.
    """
    if is_scriptable(framework):
        launch = _session_launch(args, framework=framework, workload=Workload.SCRIPTABLE, concurrencies=concurrencies)
        if launch.reference:
            _refuse(f"--reference-script is a server launch; a {framework} workload launches no server")
        try:
            script_base = _scriptable_base(launch, args)
        except BaseLaunchError as exc:
            _refuse(f"the scriptable base could not be frozen: {exc}")
        return _freeze(args, session_dir, script_base)
    replay = is_agentx_mode(benchmark_mode) and not is_mlperf_backend()
    workload = Workload.AGENTX if replay else Workload.FIXED
    launch = _session_launch(args, framework=framework, workload=workload, concurrencies=concurrencies)
    try:
        base = resolve_server_base(launch)
    except BaseLaunchUnresolved as exc:
        _refuse(str(exc))
    return _freeze(args, session_dir, base)


def _refuse_a_base_this_host_cannot_hold(args: argparse.Namespace, base: BaseLaunch) -> None:
    """Exit 2 when a single-node session's base needs more GPUs than this host sees.

    Every round of the session runs at the base's topology, so a host short of its cards can run none of them. A host
    whose GPUs cannot be counted may be planning a session that runs elsewhere, and a multi-node base spans other
    hosts' cards, so neither is checked.
    """
    if int(args.nodes) >= 2:
        return
    visible = detect_gpu_count()
    if 0 < visible < base.cards:
        _refuse(
            f"the base launch runs TP {base.topology.tp} and needs {base.cards} GPUs, but this host sees {visible} "
            "(its visible-device mask, else rocm-smi); expose more GPUs, or pass --tp or --gpus-per-node to resolve a "
            "base that fits"
        )


def _freeze(args: argparse.Namespace, session_dir: Path, base: BaseLaunch) -> BaseLaunch:
    _refuse_a_base_this_host_cannot_hold(args, base)
    write_base_launch(session_dir, base)
    _run_at_base_topology(args, base)
    _announce(base)
    return base


__all__ = ["BaseLaunchUnresolved", "SessionLaunch", "resolve_base_launch", "resolve_server_base"]
