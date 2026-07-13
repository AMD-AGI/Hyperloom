# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""``hyperloom.inference_optimizer.multi_node`` — single-entry sandbox CLI managing one session-scoped RayJob.

Subcommands (``python3 -m hyperloom.inference_optimizer.multi_node <sub>``):
``create-rayjob`` (create via SaFE + wait for Running, checkpointing
``rayjob_id`` so retries don't leak a second workload), ``bootstrap``,
``verify``, ``restart-server`` (kill + relaunch nohup'd server,
idempotent), ``kill-inference``, ``stop-rayjob``.

State lives in ``$MULTI_NODE_STATE_FILE`` (default:
``$INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR/runtime/multi_node_state.json``
when the session is pinned, else legacy ``/tmp/multi_node_state.json``).
HTTP polls under the sandbox's 120s ceiling (ADDENDUM-09) and surface
progress on stderr. Credentials must already be in sandbox env
(ADDENDUM-13); this module never invents URLs or keys.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import secrets
import time
from typing import Any

from ...session.paths import mn_profile_trace_root
from .._internal import safe_client, ray_dashboard, workload_spec
from .._internal.log import info, warn

import logging
log = logging.getLogger(__name__)

def _mn_cli():
    """Import the parent CLI lazily to keep command modules acyclic."""
    from .. import cli as mn_cli

    return mn_cli


# SaFE read-after-write lag: GET /workloads/{id} may 404 briefly post-create.
_SAFE_GET_WORKLOAD_404_GRACE_S = 30.0

# RayJob phase strings reported by SaFE.
_TERMINAL_FAIL_PHASES = {"Failed", "Stopped", "Cancelled"}

_TERMINAL_OK_PHASES = {"Running"}

def _checkpoint_create_rayjob_state(
    *,
    wid: str,
    workspace: str,
    args: argparse.Namespace,
) -> None:
    """Persist ``rayjob_id`` as soon as SaFE returns it, so a concurrent ``create-rayjob`` doesn't leak a second RayJob.

    Args:
        wid (str): The SaFE workload id to checkpoint.
        workspace (str): The SaFE workspace the workload belongs to.
        args (argparse.Namespace): Parsed ``create-rayjob`` args (supplies
            ``nodes`` / ``gpus_per_node`` / ``image``).
    """
    prev = _mn_cli()._load_state()
    old = (prev.get("rayjob_id") or "").strip()
    state: dict[str, Any] = dict(prev)
    if old and old != wid:
        for k in ("head_pod_ip", "ray_dashboard_url", "last_bootstrap_submission_id"):
            state.pop(k, None)
    state["rayjob_id"] = wid
    state["workspace"] = workspace
    state["nodes"] = args.nodes
    state["gpus_per_node"] = args.gpus_per_node
    state["service_url"] = f"http://{wid}.{workspace}.svc.cluster.local:8888"
    state["last_create_request"] = {
        "image": args.image,
        "nodes": args.nodes,
        "gpus_per_node": args.gpus_per_node,
    }
    if old != wid:
        state["head_pod_ip"] = ""
        state["ray_dashboard_url"] = ""
    else:
        state.setdefault("head_pod_ip", "")
        state.setdefault("ray_dashboard_url", "")
    state.setdefault("ray_address", "")
    _mn_cli()._save_state(state)
    info(f"checkpointed rayjob_id={wid} to {_mn_cli()._state_file()}")

def _write_rayjob_meta(
    *,
    wid: str,
    workspace: str,
    session_id: str | None,
    owner_id: str | None,
    display_name: str,
    nodes: int,
    gpus_per_node: int,
) -> None:
    """Write per-session meta JSON at ``profile-traces/<rayjob_id>/<session_id>`` tying the RayJob to its sandbox session.

    Skipped when ``session_id`` is empty (it's the filename). Best-effort:
    filesystem errors are logged at WARN and swallowed.

    Args:
        wid (str): The SaFE workload (RayJob) id.
        workspace (str): The SaFE workspace.
        session_id (str | None): The sandbox session id (used as the filename);
            the write is skipped when empty.
        owner_id (str | None): The owning sandbox workload id, if any.
        display_name (str): Human-readable RayJob name.
        nodes (int): Node count.
        gpus_per_node (int): GPUs per pod.
    """
    if not session_id:
        return
    meta_path = mn_profile_trace_root() / wid / session_id
    payload: dict[str, Any] = {
        "rayjob_id": wid,
        "session_id": session_id,
        "owner_id": owner_id,
        "workspace": workspace,
        "display_name": display_name,
        "nodes": nodes,
        "gpus_per_node": gpus_per_node,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    try:
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        info(f"rayjob meta written to {meta_path}")
    except OSError as exc:
        warn(f"failed to write rayjob meta to {meta_path}: {exc}")

# Helpers
def ray_gcs_address(head_pod_ip: str) -> str:
    """Ray driver address for ``ray.init(address=...)`` (GCS on head, default port).

    Args:
        head_pod_ip (str): The head pod IP.

    Returns:
        str: ``<ip>:6379`` for a non-empty IP, otherwise an empty string.
    """
    ip = (head_pod_ip or "").strip()
    if not ip:
        return ""
    return f"{ip}:6379"

def _is_safe_get_workload_404(exc: BaseException) -> bool:
    """Check whether an exception is a transient SaFE GET-workload 404.

    Args:
        exc (BaseException): The exception to classify.

    Returns:
        bool: ``True`` if ``exc`` is a SaFE 404 on a GET workload endpoint
        (expected briefly right after create), otherwise ``False``.
    """
    return (
        isinstance(exc, safe_client.SafeApiError)
        and exc.status == 404
        and "GET /api/v1/workloads/" in (exc.endpoint or "")
    )

def _summarize_workload_failure(workload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Build a one-line diagnostic + JSON-safe snapshot (phase/message/per-pod status/dispatch) from a SaFE GetWorkloadResponse.

    Args:
        workload (dict[str, Any]): A SaFE ``GetWorkloadResponse`` dict.

    Returns:
        tuple[str, dict[str, Any]]: ``(diag, snapshot)`` — a one-line human
        diagnostic and a JSON-safe structured failure snapshot.
    """
    phase = str(workload.get("phase") or "?")
    msg = str(workload.get("message") or "").strip()
    queue = workload.get("queuePosition")
    dispatch = workload.get("dispatchCount")
    pods = workload.get("pods") or []
    pod_summary: list[dict[str, Any]] = []
    for p in pods:
        if not isinstance(p, dict):
            continue
        pod_summary.append(
            {
                "podId": p.get("podId"),
                "phase": p.get("phase"),
                "resourceId": p.get("resourceId"),
                "node": p.get("adminNodeName"),
                "podIP": p.get("podIP"),
                "failedMessage": p.get("failedMessage"),
            }
        )

    parts = [f"phase={phase}"]
    if msg:
        parts.append(f"message={msg!r}")
    if dispatch is not None:
        parts.append(f"dispatchCount={dispatch}")
    if queue is not None and phase in ("Pending", "Updating"):
        parts.append(f"queuePosition={queue}")
    bad_pods = [p for p in pod_summary if p.get("phase") in ("Failed", "Unknown") or p.get("failedMessage")]
    if bad_pods:
        # First 3 failing pods inline; full list in snapshot.
        bp_strs = []
        for bp in bad_pods[:3]:
            bp_strs.append(
                f"{bp.get('podId') or '?'}({bp.get('phase') or '?'}"
                + (f": {bp['failedMessage']}" if bp.get("failedMessage") else "")
                + ")"
            )
        parts.append("failed_pods=[" + ", ".join(bp_strs) + (", ..." if len(bad_pods) > 3 else "") + "]")
    elif pod_summary:
        parts.append(f"pods={len(pod_summary)}")
    diag = " ".join(parts)
    snapshot = {
        "workloadId": workload.get("workloadId"),
        "phase": phase,
        "message": msg,
        "dispatchCount": dispatch,
        "queuePosition": queue,
        "pods": pod_summary,
    }
    return diag, snapshot

def _find_head_pod_ip(workload: dict) -> str:
    """Pick the KubeRay head pod's PodIp from a GetWorkloadResponse.

    Priority: a ``podId`` containing ``-head-``, then ``resourceId == 0``,
    then the first pod with a ``podIP`` (the submitter pod isn't the head).

    Args:
        workload (dict): A SaFE ``GetWorkloadResponse`` dict.

    Returns:
        str: The resolved head pod IP, or ``""`` when no pod carries one.
    """
    pods = workload.get("pods") or []
    if not pods:
        return ""
    for p in pods:
        pid = str(p.get("podId") or "")
        if "-head-" in pid and p.get("podIP"):
            return str(p["podIP"])
    for p in pods:
        if p.get("resourceId") == 0 and p.get("podIP"):
            return str(p["podIP"])
    for p in pods:
        if p.get("podIP"):
            return str(p["podIP"])
    return ""

# Subcommand: create-rayjob
def cmd_create_rayjob(args: argparse.Namespace) -> int:
    """Create the RayJob, checkpoint ``rayjob_id`` early, then poll for Running.

    Reuses an existing non-terminal workload from state (unless
    ``--recreate``), persists the state file, and prints the merged state
    as JSON.

    Args:
        args (argparse.Namespace): Parsed ``create-rayjob`` arguments.

    Returns:
        int: ``0`` on success.

    Raises:
        RuntimeError: If no workspace can be resolved.
        WorkloadTerminalFailure: If the workload enters a terminal failure
            phase while polling.
        TransientFailure: If polling times out before the workload is
            Running.
    """
    extra_env = _mn_cli()._parse_kv_list(args.extra_env)
    extra_labels = _mn_cli()._parse_kv_list(args.extra_label)
    pending_dashboard_token: str | None = None

    # ownerId: --owner-id > $WORKLOAD_ID (the sandbox workload, for SaFE GC
    # cascade); omitted when neither is set.
    owner_id = args.owner_id or os.environ.get("WORKLOAD_ID", "").strip() or None
    if owner_id and not args.owner_id:
        info(f"ownerId derived from $WORKLOAD_ID: {owner_id}")

    # workspace: --workspace > $SAFE_WORKSPACE; bail fast with a clear error if neither is set.
    workspace = args.workspace or os.environ.get("SAFE_WORKSPACE", "").strip()
    if not workspace:
        raise RuntimeError(
            "workspace is required: pass --workspace <safe-workspace> "
            "or export $SAFE_WORKSPACE in the sandbox env. "
            "Brain normally sets this at sandbox startup."
        )
    if not args.workspace:
        info(f"workspace derived from $SAFE_WORKSPACE: {workspace}")

    # display_name: $DISPLAY_NAME > --display-name > fallback.
    display_name = os.environ.get("DISPLAY_NAME", "").strip() or args.display_name or f"multi_node_{int(time.time())}"
    info(f"displayName: {display_name}")

    # session_id from $CLAW_SESSION_ID; when unset the label is skipped (dev/local runs).
    session_id = (os.environ.get("CLAW_SESSION_ID") or "").strip() or None
    if session_id:
        info(f"sessionId derived from $CLAW_SESSION_ID: {session_id}")

    body: dict[str, Any] | None = None

    with safe_client.from_env() as safe:
        # Idempotency guard: reuse a state ``rayjob_id`` that's still
        # non-terminal in SaFE (--recreate forces a fresh workload).
        wid: str | None = None
        existing = _mn_cli()._load_state()
        prior_wid = (existing.get("rayjob_id") or "").strip()
        if prior_wid and not getattr(args, "recreate", False):
            try:
                prior_wl = safe.get_workload(prior_wid)
            except safe_client.SafeApiError as exc:
                if exc.status == 404:
                    info(f"prior rayjob_id={prior_wid} no longer exists in SaFE; will create a fresh workload")
                else:
                    raise
            else:
                prior_phase = str(prior_wl.get("phase") or "?")
                if prior_phase in _TERMINAL_FAIL_PHASES:
                    info(
                        f"prior rayjob_id={prior_wid} is in terminal phase "
                        f"{prior_phase!r}; will create a fresh workload"
                    )
                else:
                    info(
                        f"reusing existing rayjob_id={prior_wid} from state file "
                        f"(phase={prior_phase}); will resume polling instead of "
                        f"creating a new workload"
                    )
                    wid = prior_wid

        if wid is None:
            pending_dashboard_token = secrets.token_urlsafe(32)
            env = dict(extra_env)
            env["RAY_DASHBOARD_TOKEN"] = pending_dashboard_token
            body = workload_spec.build_rayjob_workload_body(
                workspace=workspace,
                display_name=display_name,
                image=args.image,
                nodes=args.nodes,
                gpus_per_node=args.gpus_per_node,
                cpus_per_node=args.cpus_per_node,
                mem_gi_per_node=args.mem_per_node,
                ephemeral_gi_per_node=args.ephemeral_per_node,
                description=args.description,
                owner_id=owner_id,
                session_id=session_id,
                extra_env=env,
                extra_labels=extra_labels,
            )
            info(f"creating RayJob workload (workspace={workspace} nodes={args.nodes})")
            wid = safe.create_workload(body)
            info(f"workload created: {wid}")
            _write_rayjob_meta(
                wid=wid,
                workspace=workspace,
                session_id=session_id,
                owner_id=owner_id,
                display_name=display_name,
                nodes=args.nodes,
                gpus_per_node=args.gpus_per_node,
            )

        _checkpoint_create_rayjob_state(wid=wid, workspace=workspace, args=args)

        head_pod_ip = ""
        if args.no_wait:
            info("--no-wait set; not polling for Running")
        else:

            def _fetch():
                """Fetch the workload and summarize its phase.

                Returns:
                    tuple: ``(workload_dict, summary_str)`` for the poll loop.
                """
                wl = safe.get_workload(wid)
                phase = wl.get("phase", "?")
                summary = f"phase={phase}"
                return wl, summary

            workload = _mn_cli()._short_poll(
                label=f"workload {wid}",
                fetch=_fetch,
                is_ok=lambda w: w.get("phase") in _TERMINAL_OK_PHASES,
                is_fail=lambda w: w.get("phase") in _TERMINAL_FAIL_PHASES,
                interval_s=args.poll_interval,
                timeout_s=_mn_cli()._poll_timeout_from_args(args),
                failure_diag=_summarize_workload_failure,
                quiet_fetch_error_grace_s=_SAFE_GET_WORKLOAD_404_GRACE_S,
                is_quiet_fetch_error=_is_safe_get_workload_404,
            )
            head_pod_ip = _find_head_pod_ip(workload)
            if not head_pod_ip:
                warn(
                    "workload Running but no head pod IP yet; SaFE may need "
                    "another sync. Re-run create-rayjob to refresh."
                )

    merged = dict(_mn_cli()._load_state())
    prior_token = str(merged.get("ray_dashboard_token") or "").strip()
    if pending_dashboard_token:
        prior_token = pending_dashboard_token
    merged.update(
        {
            "backend": "rayjob",
            "rayjob_id": wid,
            "workspace": workspace,
            "nodes": args.nodes,
            "gpus_per_node": args.gpus_per_node,
            "head_pod_ip": head_pod_ip,
            "ray_address": ray_gcs_address(head_pod_ip),
            "ray_dashboard_url": (ray_dashboard.dashboard_url(head_pod_ip) if head_pod_ip else ""),
            "ray_dashboard_token": prior_token,
            "service_url": f"http://{wid}.{workspace}.svc.cluster.local:8888",
            "last_create_request": {
                "image": args.image,
                "nodes": args.nodes,
                "gpus_per_node": args.gpus_per_node,
            },
        }
    )
    _mn_cli()._save_state(merged)
    info(f"state written to {_mn_cli()._state_file()}")
    print(json.dumps(merged, indent=2, sort_keys=True))
    return 0
