# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Profiling of atomesh and sglangrouter pod hand-off engines: their URLs, stale graph captures, and analysed traces."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any

from inference_suite.contracts.layout import CAPTURE_TRACE_DIRS

from .pd_cluster import DECODE, ENGINE_PORT, PREFILL, record_profile_paths, role_engine_pods
from .pd_handoff import PD_BACKENDS, is_pd_state

log = logging.getLogger(__name__)


def pod_engine_urls(state: dict[str, Any]) -> dict[str, list[str]]:
    """The HTTP base URL of every prefill and decode engine of an atomesh/sglangrouter pod hand-off, in pod order.

    Each engine listens on its pod's ``enginePort``, else ``HYPERLOOM_MN_WORKER_PORT``, else the engine port. A role
    with no pods in state is reached through ``HYPERLOOM_MN_EXT_<ROLE>_URL``. Empty for any other state.
    """
    if str(state.get("backend") or "") not in PD_BACKENDS or is_pd_state(state):
        return {}
    port = os.environ.get("HYPERLOOM_MN_WORKER_PORT", "").strip() or str(ENGINE_PORT)
    urls: dict[str, list[str]] = {}
    for role in (PREFILL, DECODE):
        role_pods = role_engine_pods(state, role)
        role_urls = [f"http://{pod['podIP']}:{pod.get('enginePort') or port}" for pod in role_pods]
        engine_url = os.environ.get(f"HYPERLOOM_MN_EXT_{role.upper()}_URL", "").strip().rstrip("/")
        if not role_pods and engine_url:
            role_urls.append(engine_url)
        if role_urls:
            urls[role] = role_urls
    return urls


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
