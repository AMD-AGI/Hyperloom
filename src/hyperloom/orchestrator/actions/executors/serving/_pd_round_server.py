# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PD node-list sessions: the per-round cluster restart, the engine profiling calls and the per-node trace choice."""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any

from hyperloom_common.env import env_flag
from hyperloom_common.io import safe_mtime, safe_size
from inference_suite.multinode import PD_ROLES

from .._multi_node_env import _read_state
from ._multi_node_restart_settings import PROFILE_START_TIMEOUT_S, ServerRestartFailed, profile_stop_timeout

log = logging.getLogger(__name__)

PD_PROFILE_ROLES = tuple(role.value for role in PD_ROLES)

_RANK_TRACE_RE = re.compile(r"\A(?P<batch>.+)-TP-(?P<rank>\d+)(?:-(?P<stage>[A-Z]+))?\.trace\.json\.gz\Z")
_ROLE_STAGE = {"prefill": "EXTEND", "decode": "DECODE"}


async def restart_pd_round(extra_server_args: str, poll_timeout_s: int) -> None:
    """Restart a PD session's cluster with the round's args over its frozen launch; both roles get them.

    The caller has validated the args and holds the restart lock, the round's forwarded env and its trace dir.
    """
    from hyperloom.inference_optimizer.multi_node.cli import restart_pd_round as restart_cluster
    from hyperloom.inference_optimizer.multi_node.pd_session import PdSessionError

    log.info("restart_server_for_round: PD cluster, round args=%r", extra_server_args)
    try:
        await asyncio.to_thread(restart_cluster, extra_server_args, poll_timeout_s)
    except PdSessionError as exc:
        raise ServerRestartFailed(str(exc)) from exc


def pd_worker_trace_dir(root: str, role: str, worker: int) -> str:
    """Return the torch trace dir of one engine worker of ``role`` under a profiling round's trace root."""
    return str(Path(root) / role / str(worker))


def _first_gpu_traces(paths: list[Path], role: str, nodes: int) -> list[str]:
    """The first GPU's trace of each node from the newest batch in ``paths``; the heaviest file when none is per-rank.

    SGLang numbers TP ranks across a worker's nodes, so node ``n`` starts at rank ``n * ranks_per_node``. A stage-split
    trace prefers the role's own stage.
    """
    batches: dict[str, dict[int, dict[str, Path]]] = {}
    newest: dict[str, float] = {}
    for path in paths:
        match = _RANK_TRACE_RE.match(path.name)
        if match is None:
            continue
        batch = match["batch"]
        batches.setdefault(batch, {}).setdefault(int(match["rank"]), {})[match["stage"] or ""] = path
        newest[batch] = max(newest.get(batch, 0.0), safe_mtime(path))
    if not batches:
        return [str(max(paths, key=lambda p: (safe_size(p), p.name)))] if paths else []
    ranks = batches[max(newest, key=lambda batch: newest[batch])]
    per_node = -(-(max(ranks) + 1) // nodes)
    picks: list[str] = []
    for node in range(nodes):
        stages = ranks.get(node * per_node)
        if stages:
            picks.append(str(stages.get(_ROLE_STAGE.get(role, "")) or stages.get("") or sorted(stages.values())[0]))
    return picks


def select_pd_node_traces(root: str, since_unix: float) -> dict[str, list[str]]:
    """Return, per PD role, the first GPU's trace of each node written since ``since_unix``, worker by worker under
    ``<root>/<role>/<worker>``; nodes are numbered across the role's workers in order."""
    from hyperloom.inference_optimizer.multi_node.pd_session import is_pd_state, session_worker_hosts

    state = _read_state()
    workers = session_worker_hosts(state) if is_pd_state(state) else {}
    selected: dict[str, list[str]] = {}
    for role in PD_PROFILE_ROLES:
        picks: list[str] = []
        for worker, hosts in enumerate(workers.get(role, [])):
            fresh = [
                path
                for path in Path(pd_worker_trace_dir(root, role, worker)).glob("*.trace.json.gz")
                if path.is_file() and safe_mtime(path) >= since_unix
            ]
            picks += _first_gpu_traces(fresh, role, len(hosts))
        if picks:
            selected[role] = picks
    return selected


async def trigger_pd_engine_profile(action: str, body: dict | None = None) -> None:
    """POST /start_profile or /stop_profile to every prefill and decode engine of a PD session.

    Each engine worker is reached at its first host on its role's engine port, as the frozen launch is placed on the
    node list; the router writes no GPU traces. A start sends ``output_dir`` narrowed to
    ``<output_dir>/<role>/<worker>`` and ``with_stack`` (``HYPERLOOM_MN_PROFILE_WITH_STACK``, default off); step-window
    fields crash disaggregated SGLang.
    """
    from hyperloom.inference_optimizer.multi_node.pd_session import session_engine_urls

    urls = session_engine_urls(_read_state())
    targets = [(role, worker, url) for role in PD_PROFILE_ROLES for worker, url in enumerate(urls.get(role, []))]
    if not targets:
        log.warning("trigger_pd_engine_profile(%s): no prefill/decode engines to profile", action)
        return
    import httpx as _httpx

    starting = action == "start"
    route = "start_profile" if starting else "stop_profile"
    output_dir = str((body or {}).get("output_dir") or "").strip() if starting else ""
    with_stack = env_flag("HYPERLOOM_MN_PROFILE_WITH_STACK", default=False)

    async def _post(client: Any, role: str, worker: int, base_url: str) -> None:
        """Send ``route`` to one engine; a transport error is logged, not raised."""
        url = f"{base_url}/{route}"
        payload: dict = {}
        if output_dir:
            worker_dir = pd_worker_trace_dir(output_dir, role, worker)
            Path(worker_dir).mkdir(parents=True, exist_ok=True)
            payload["output_dir"] = worker_dir
        if starting:
            payload["with_stack"] = with_stack
        try:
            resp = await client.post(url, json=payload)
            log.info("pd profile %s %s -> %s HTTP %d", role, route, url, resp.status_code)
        except _httpx.HTTPError as exc:
            log.warning("pd profile %s -> %s failed (%s); continuing", route, url, exc)

    # stop_profile returns after the trace export; engines start and stop together so each records the same window.
    async with _httpx.AsyncClient(timeout=PROFILE_START_TIMEOUT_S if starting else profile_stop_timeout()) as client:
        await asyncio.gather(*(_post(client, role, worker, base_url) for role, worker, base_url in targets))
