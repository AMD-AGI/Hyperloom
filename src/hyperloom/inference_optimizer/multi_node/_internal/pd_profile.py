# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Profiling of atomesh and sglangrouter PD engines: start/stop requests, trace dirs, and trace selection."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import yaml
from hyperloom_common.env import env_flag, env_float
from hyperloom_common.io import safe_mtime
from inference_suite.contracts.layout import CAPTURE_TRACE_DIRS

from .pd_cluster import ENGINE_PORT, pd_pod_trace_dir, record_profile_paths, role_engine_pods
from .topology_file import PD_ROUTER_BACKENDS

log = logging.getLogger(__name__)

PD_PROFILE_ROLES = ("prefill", "decode")
# Seconds between checks for a role's trace while waiting on step-window engines.
PD_TRACE_POLL_S = 5.0

_RANK_TRACE_RE = re.compile(r"\A(?P<batch>.+)-TP-(?P<rank>\d+)(?:-(?P<stage>[A-Z]+))?\.trace\.json\.gz\Z")
_ROLE_STAGE = {"prefill": "EXTEND", "decode": "DECODE"}
# Per role: the step-count env override, its default, and whether the trace is split by forward stage.
_PD_ROLE_PROFILE_STEPS = {
    "prefill": ("HYPERLOOM_MN_PD_PREFILL_PROFILE_STEPS", 8, True),
    "decode": ("HYPERLOOM_MN_PD_DECODE_PROFILE_STEPS", 256, False),
}
_START_TIMEOUT_S = 30.0
_GRAPH_CAPTURE_PROFILE_ENV_KEYS = (
    "SGLANG_GRAPH_BATCH_CAPTURE",
    "DEBUG_CLR_GRAPH_PACKET_CAPTURE",
    "DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING",
    "DEBUG_HIP_GRAPH_CLASSIC_PATH",
)
PROFILE_CUDA_GRAPH_FLAG = "--enable-profile-cuda-graph"


def _is_pd_cluster(state: dict[str, Any]) -> bool:
    """Whether ``state`` describes an atomesh or sglangrouter cluster."""
    return str(state.get("backend") or "").strip().lower() in PD_ROUTER_BACKENDS


def drop_graph_captures(torch_profiler_dir: str) -> None:
    """Remove the CUDA-graph capture folders under ``torch_profiler_dir``, so they hold only the next server's captures.

    A relaunched server rewrites only the batch sizes it captures; a capture of the previous server left beside them
    is matched against replays it never ran.
    """
    root = Path(torch_profiler_dir)
    for name in (*CAPTURE_TRACE_DIRS, "graph_capture"):
        for folder in (root / name, *root.glob(f"*/{name}"), *root.glob(f"*/*/{name}")):
            if not folder.is_dir():
                continue
            try:
                shutil.rmtree(folder)
            except OSError as exc:
                log.warning("restart_server_for_round: cannot remove stale graph captures %s: %s", folder, exc)


def _engine_trace_dirs(state: dict[str, Any], root: str, role: str) -> dict[str, int]:
    """Trace dir and node count of each ``role`` engine under ``root``, in pod order.

    An atomesh/sglangrouter pod runs its own engine under :func:`pd_pod_trace_dir`; an Infera role runs one engine
    across all its pods under ``<root>/<role>``. A role with no pods in state counts as one engine on one node.
    """
    pods = max(1, len(role_engine_pods(state, role)))
    if _is_pd_cluster(state):
        return {pd_pod_trace_dir(root, role, index): 1 for index in range(pods)}
    return {str(Path(root) / role): pods}


def _first_gpu_traces(paths: list[Path], role: str, nodes: int) -> list[str]:
    """The first GPU's trace of each node from the newest batch in ``paths``; the heaviest file when none is per-rank.

    SGLang numbers TP ranks across an engine's nodes, so node ``n`` starts at rank ``n * ranks_per_node``. A
    stage-split trace prefers the role's own stage.
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
        return [str(max(paths, key=lambda p: (p.stat().st_size, p.name)))] if paths else []
    ranks = batches[max(newest, key=lambda batch: newest[batch])]
    per_node = -(-(max(ranks) + 1) // nodes)
    picks: list[str] = []
    for node in range(nodes):
        stages = ranks.get(node * per_node)
        if stages:
            picks.append(str(stages.get(_ROLE_STAGE.get(role, "")) or stages.get("") or sorted(stages.values())[0]))
    return picks


def pd_node_traces(state: dict[str, Any], root: str, since_unix: float) -> dict[str, list[str]]:
    """Return, per PD role, the first GPU's trace of each node written since ``since_unix``, nodes in pod order."""
    selected: dict[str, list[str]] = {}
    for role in PD_PROFILE_ROLES:
        picks: list[str] = []
        for trace_dir, nodes in _engine_trace_dirs(state, root, role).items():
            fresh = [
                path
                for path in Path(trace_dir).glob("*.trace.json.gz")
                if path.is_file() and safe_mtime(path) >= since_unix
            ]
            picks.extend(_first_gpu_traces(fresh, role, nodes))
        if picks:
            selected[role] = picks
    return selected


def pd_profile_by_steps() -> bool:
    """Whether PD engines profile a step window: ``HYPERLOOM_MN_PD_PROFILE_MODE`` is ``steps`` (default), not ``time``."""
    return os.environ.get("HYPERLOOM_MN_PD_PROFILE_MODE", "").strip().lower() != "time"


def _pd_step_window(role: str) -> dict[str, Any]:
    """The step-window fields of ``role``'s start body: its step count, stage split and shape recording."""
    env_key, default_steps, by_stage = _PD_ROLE_PROFILE_STEPS[role]
    try:
        steps = int(os.environ.get(env_key, "") or default_steps)
    except ValueError:
        steps = default_steps
    return {"num_steps": max(1, steps), "profile_by_stage": by_stage, "record_shapes": True}


def stop_profile_timeout_s() -> float:
    """Seconds a stop may take to export traces, ``HYPERLOOM_MN_STOP_PROFILE_TIMEOUT_S`` (default 600)."""
    return env_float("HYPERLOOM_MN_STOP_PROFILE_TIMEOUT_S", 600.0)


def _profile_targets(state: dict[str, Any]) -> list[tuple[str, int, str]]:
    """``(role, pod index, engine base URL)`` of every prefill and decode engine in ``state``.

    Each engine listens on its pod's ``enginePort``, else the worker port. A role with no pods in state is reached
    through ``HYPERLOOM_MN_EXT_<ROLE>_URL`` as pod 0.
    """
    port = os.environ.get("HYPERLOOM_MN_WORKER_PORT", "").strip() or str(ENGINE_PORT)
    targets: list[tuple[str, int, str]] = []
    for role in PD_PROFILE_ROLES:
        role_pods = role_engine_pods(state, role)
        targets.extend(
            (role, index, f"http://{pod['podIP']}:{pod.get('enginePort') or port}")
            for index, pod in enumerate(role_pods)
        )
        engine_url = os.environ.get(f"HYPERLOOM_MN_EXT_{role.upper()}_URL", "").strip().rstrip("/")
        if not role_pods and engine_url:
            targets.append((role, 0, engine_url))
    return targets


def _pod_output_dir(output_dir: str, role: str, pod_index: int) -> str:
    """The trace dir of one engine under ``output_dir``, created when possible."""
    pod_dir = pd_pod_trace_dir(output_dir, role, pod_index)
    try:
        Path(pod_dir).mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log.debug("pd-cluster profile: cannot mkdir %s: %s", pod_dir, exc)
    return pod_dir


async def post_pd_engine_profile(state: dict[str, Any], action: str, body: dict | None = None) -> list[str]:
    """POST /start_profile or /stop_profile on every atomesh/sglangrouter prefill and decode engine of ``state``.

    The router does not write GPU traces. A start sends each engine its pod's :func:`pd_pod_trace_dir` under
    ``output_dir``, and ``with_stack`` (``HYPERLOOM_MN_PROFILE_WITH_STACK``, default off). In ``steps`` mode
    (:func:`pd_profile_by_steps`) it also sends the role's step window: prefill records 8 steps split by stage, decode
    256 steps (``HYPERLOOM_MN_PD_<ROLE>_PROFILE_STEPS``), both with shapes, and each engine closes its own window.
    In ``time`` mode the window lasts until the stop.

    Returns the roles with at least one engine that answered without an error status.
    """
    if not _is_pd_cluster(state):
        return []
    targets = _profile_targets(state)
    if not targets:
        log.warning("post_pd_engine_profile(%s): no prefill/decode pods or engine URLs", action)
        return []
    import httpx

    start = action == "start"
    route = "start_profile" if start else "stop_profile"
    output_dir = str((body or {}).get("output_dir") or "").strip() if start else ""
    with_stack = env_flag("HYPERLOOM_MN_PROFILE_WITH_STACK")
    by_steps = pd_profile_by_steps()

    async def _post(client: Any, role: str, pod_index: int, base_url: str) -> bool:
        """Send ``route`` to one engine and return whether it answered without an error status; errors are logged."""
        url = f"{base_url}/{route}"
        payload: dict[str, Any] = {}
        if start:
            payload = {**(_pd_step_window(role) if by_steps else {}), "with_stack": with_stack}
            if output_dir:
                payload["output_dir"] = _pod_output_dir(output_dir, role, pod_index)
        try:
            resp = await client.post(url, json=payload)
            log.info("pd-cluster profile %s %s -> %s HTTP %d", role, route, url, resp.status_code)
        except httpx.HTTPError as exc:
            log.warning("pd-cluster profile %s -> %s failed (%s); continuing", route, url, exc)
            return False
        return resp.status_code < 400

    # stop_profile returns after the trace export; engines start and stop together so each records the same window.
    async with httpx.AsyncClient(timeout=_START_TIMEOUT_S if start else stop_profile_timeout_s()) as client:
        answered = await asyncio.gather(*(_post(client, *target) for target in targets))
    return list(dict.fromkeys(role for (role, _index, _url), ok in zip(targets, answered) if ok))


async def await_pd_role_traces(select: Callable[[], dict[str, list[str]]], roles: list[str], root: str) -> None:
    """Wait until ``select()`` returns a trace for each of ``roles``.

    Polls every ``PD_TRACE_POLL_S`` seconds for at most :func:`stop_profile_timeout_s`.
    """
    timeout_s = stop_profile_timeout_s()
    deadline = time.monotonic() + timeout_s
    while True:
        found = await asyncio.to_thread(select)
        missing = [role for role in roles if role not in found]
        if not missing:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            log.warning(
                "profile_executor: no PD trace from %s under %s after %.0fs; sending the safety stop",
                missing,
                root,
                timeout_s,
            )
            return
        await asyncio.sleep(min(PD_TRACE_POLL_S, remaining))


def trace_fs_round_start(round_trace_root: str, fallback_unix: float) -> float:
    """Return the round start on the trace filesystem's clock, read from a marker file's mtime.

    Trace mtimes are stamped by the filesystem that stores them, so this round's
    traces are compared against a time from that same clock; ``fallback_unix``
    is returned when the marker cannot be written.
    """
    marker = Path(round_trace_root) / ".round_start"
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(f"{fallback_unix:.3f}\n", encoding="utf-8")
        return marker.stat().st_mtime
    except OSError as exc:
        log.warning(
            "profile_executor: cannot stamp round start under %s (%s); using the local clock", round_trace_root, exc
        )
        return fallback_unix


def graph_capture_profile_settings(config_path: Path) -> tuple[dict[str, str], str]:
    """Graph-capture profiling env and server flag the profile YAML sets, as ``(env, flag)``; empty when unset."""
    try:
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        log.warning("profile_executor: cannot read %s for graph-capture profiling settings: %s", config_path, exc)
        return {}, ""
    bench = cfg.get("benchmark") if isinstance(cfg, dict) else None
    envs = bench.get("envs") if isinstance(bench, dict) else None
    if not isinstance(envs, dict):
        return {}, ""
    env = {key: str(envs[key]) for key in _GRAPH_CAPTURE_PROFILE_ENV_KEYS if key in envs}
    flag = (
        PROFILE_CUDA_GRAPH_FLAG if PROFILE_CUDA_GRAPH_FLAG in str(envs.get("EXTRA_SGLANG_ARGS") or "").split() else ""
    )
    return env, flag


def pd_graph_capture_profiling(
    state: dict[str, Any], server_args: str, config_path: Path
) -> tuple[str, dict[str, str]]:
    """Server args and env that turn graph-capture profiling on for an atomesh/sglangrouter profiling restart.

    Other backends get ``server_args`` back with no env. The flag is not added when the args disable CUDA graphs.
    """
    if not _is_pd_cluster(state):
        return server_args, {}
    env, flag = graph_capture_profile_settings(config_path)
    tokens = server_args.split()
    if flag and flag not in tokens and "--disable-cuda-graph" not in tokens:
        server_args = f"{server_args} {flag}".strip()
    return server_args, env


def record_analysed_profile(trace_path: str, steady_state_trace: str) -> None:
    """Record an analysed trace and its profile dir in the multi-node state; a failed write is logged, not raised."""
    profile_dir = os.environ.get("HYPERLOOM_MN_PROFILE_TRACE_DIR", "").strip() or str(Path(trace_path).parent)
    try:
        record_profile_paths(profile_trace=profile_dir, steady_state_trace=steady_state_trace or trace_path)
    except (OSError, RuntimeError, ValueError, TypeError):
        log.debug("roofline: could not record profile paths into multi-node state", exc_info=True)


def infera_profile_payload(state: dict[str, Any], action: str, body: dict | None) -> dict[str, Any]:
    """The Infera admin profile request body for ``action``; ``stop`` sends none.

    A PD-disaggregated launch drops ``output_dir``: each role's workers write under their own
    ``SGLANG_TORCH_PROFILER_DIR``.
    """
    payload = dict(body or {}) if action == "start" else {}
    mode = str(state.get("last_restart_pd_mode") or state.get("pd_mode") or "").strip().lower()
    if mode == "disaggregated":
        payload.pop("output_dir", None)
    return payload


# The TraceLens input each PD role's trace yields: a decode engine's trace holds decode steps only, so its steady-state
# chunk is taken with or without the decode-only label; a prefill engine's few steps are analysed whole.
_ROLE_STEADY_STATE_MODE = {"prefill": "unsplit", "decode": "decode_only_or_generic"}


def _pd_main_role(profile_result: Any, trace_path: str) -> str:
    """The PD role whose trace the profile adopted as its main trace, or '' outside a PD cluster."""
    role_traces = profile_result.get("role_traces") if isinstance(profile_result, dict) else None
    if not isinstance(role_traces, dict):
        return ""
    return next((role for role, path in role_traces.items() if path and path == trace_path), "")


def _pd_multi_trace_input(profile_result: dict[str, Any]) -> dict[str, Any] | None:
    """The multi-trace analysis input of a PD profile with decode nodes and two or more nodes in all, else ``None``."""
    node_traces = profile_result.get("role_node_traces")
    if not isinstance(node_traces, dict) or not node_traces.get("decode"):
        return None
    if sum(len(paths or []) for paths in node_traces.values()) < 2:
        return None
    return {"role_node_traces": node_traces}


def add_pd_analysis(payload: dict[str, Any], profile_result: Any, trace_path: str) -> None:
    """Add the PD analysis settings of a PD profile's main role to a trace_analyze payload; no-op outside PD.

    An operator-set ``INFERENCE_OPTIMIZER_STEADY_STATE_MODE`` replaces the role's default steady-state mode.
    """
    main_role = _pd_main_role(profile_result, trace_path)
    if not main_role:
        return
    if not os.environ.get("INFERENCE_OPTIMIZER_STEADY_STATE_MODE", "").strip():
        payload.setdefault("steady_state_mode", _ROLE_STEADY_STATE_MODE[main_role])
    payload.setdefault("split_llm_inference", False)
    payload["clamp_stream_overlaps"] = True
    pd_input = _pd_multi_trace_input(profile_result)
    if pd_input:
        payload["pd_node_traces"] = pd_input
