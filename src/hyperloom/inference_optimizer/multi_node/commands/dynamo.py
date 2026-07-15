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
import base64
import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

from .._internal import safe_client, workload_spec
from .._internal import ssh_client, dynamo_support
from .._internal.env_safety import filter_forward_env
from .._internal.log import info, warn, err
from .._internal.server_args_safety import ServerArgsRejected, validate_server_args

import logging
log = logging.getLogger(__name__)

class _MnCliProxy:
    """Lazy proxy preserving ``dyn._mn_cli`` monkeypatch compatibility."""

    def __getattr__(self, name: str) -> Any:
        from .. import cli as mn_cli

        return getattr(mn_cli, name)


_mn_cli = _MnCliProxy()


EXIT_CONFIG_ERROR = 3
EXIT_TRANSIENT = 1
# rayjob.py is imported by cli.py first, so its symbols are already resolvable.
from .rayjob import (
    _TERMINAL_FAIL_PHASES,
    _TERMINAL_OK_PHASES,
    _SAFE_GET_WORKLOAD_404_GRACE_S,
    _is_safe_get_workload_404,
    _summarize_workload_failure,
)


def cmd_create_dynamo(args: argparse.Namespace) -> int:
    """Create an idle multi-node DynamoDeployment, then poll for Running.

    Generates a session SSH keypair (public key -> MN_SSH_AUTHORIZED_KEY in the
    workload body), creates/reuses the workload, waits for Running, then
    discovers the worker pod IPs and the frontend service URL into the state
    file so restart-server can SSH-fan-out the inference server.

    Args:
        args (argparse.Namespace): Parsed ``create-dynamo`` arguments.

    Returns:
        int: ``0`` on success.

    Raises:
        RuntimeError: If no workspace can be resolved.
    """
    extra_env = _mn_cli._parse_kv_list(args.extra_env)
    extra_labels = _mn_cli._parse_kv_list(args.extra_label)
    # Dynamo inference pods run sglang/vllm only and never call an LLM/agent
    # endpoint, so no credentials are baked into their container env; only
    # operator --extra-env values are forwarded (agents get creds over SSH).
    env = dict(extra_env)

    owner_id = args.owner_id or os.environ.get("WORKLOAD_ID", "").strip() or None
    workspace = args.workspace or os.environ.get("SAFE_WORKSPACE", "").strip()
    if not workspace:
        raise RuntimeError("workspace is required: pass --workspace or export $SAFE_WORKSPACE")
    display_name = os.environ.get("DISPLAY_NAME", "").strip() or args.display_name or f"dynamo_mn_{int(time.time())}"
    session_id = (os.environ.get("CLAW_SESSION_ID") or "").strip() or None

    # Session SSH keypair: public key authorises the controller on every pod.
    priv_key, pub_key = ssh_client.generate_session_keypair(_mn_cli._dynamo_ssh_dir())

    pd_mode = (getattr(args, "pd_mode", "") or "aggregated").lower()
    body = workload_spec.build_dynamo_workload_body(
        workspace=workspace,
        display_name=display_name,
        image=args.image,
        nodes=args.nodes,
        gpus_per_node=args.gpus_per_node,
        cpus_per_node=args.cpus_per_node,
        mem_gi_per_node=args.mem_per_node,
        ephemeral_gi_per_node=args.ephemeral_per_node,
        ssh_authorized_key=pub_key,
        backend_framework=args.backend_framework,
        kv_transfer_backend=args.kv_transfer_backend,
        ssh_port=args.ssh_port,
        shared_mem_gi=args.shared_mem_per_node,
        pd_mode=pd_mode,
        pd_prefill_nodes=int(getattr(args, "pd_prefill_nodes", 0) or 0),
        pd_decode_nodes=int(getattr(args, "pd_decode_nodes", 0) or 0),
        pd_prefill_tp=int(getattr(args, "pd_prefill_tp", 0) or 0),
        pd_decode_tp=int(getattr(args, "pd_decode_tp", 0) or 0),
        description=args.description,
        owner_id=owner_id,
        session_id=session_id,
        extra_env=env,
        extra_labels=extra_labels,
    )

    with safe_client.from_env() as safe:
        wid: str | None = None
        existing = _mn_cli._load_state()
        prior_wid = (existing.get("rayjob_id") or "").strip()
        prior_is_dynamo = existing.get("backend") == "dynamo"
        if prior_wid and prior_is_dynamo and not getattr(args, "recreate", False):
            try:
                prior_wl = safe.get_workload(prior_wid)
            except safe_client.SafeApiError as exc:
                if exc.status == 404:
                    info(f"prior dynamo workload {prior_wid} gone; creating fresh")
                else:
                    raise
            else:
                if str(prior_wl.get("phase") or "?") in _TERMINAL_FAIL_PHASES:
                    info(f"prior dynamo workload {prior_wid} terminal; creating fresh")
                else:
                    info(f"reusing dynamo workload {prior_wid}; resume polling")
                    wid = prior_wid

        if wid is None:
            info(f"creating DynamoDeployment (workspace={workspace} nodes={args.nodes})")
            wid = safe.create_workload(body)
            info(f"workload created: {wid}")

        # Checkpoint id immediately for idempotent retries.
        st = dict(_mn_cli._load_state())
        st.update(
            {
                "backend": "dynamo",
                "rayjob_id": wid,
                "workspace": workspace,
                "nodes": args.nodes,
                "gpus_per_node": args.gpus_per_node,
                "ssh_key_path": str(priv_key),
                "ssh_port": args.ssh_port,
                "framework": args.backend_framework,
                "pd_mode": pd_mode,
                "kv_transfer_backend": args.kv_transfer_backend,
                "service_url": dynamo_support.frontend_service_url(wid, workspace),
            }
        )
        _mn_cli._save_state(st)

        workload: dict[str, Any] = {}
        if args.no_wait:
            info("--no-wait set; not polling for Running")
        else:

            def _fetch():
                """Fetch the workload and summarize its phase.

                Returns:
                    tuple: ``(workload_dict, summary_str)`` for the poll loop.
                """
                wl = safe.get_workload(wid)
                return wl, f"phase={wl.get('phase', '?')}"

            workload = _mn_cli._short_poll(
                label=f"dynamo workload {wid}",
                fetch=_fetch,
                is_ok=lambda w: w.get("phase") in _TERMINAL_OK_PHASES,
                is_fail=lambda w: w.get("phase") in _TERMINAL_FAIL_PHASES,
                interval_s=args.poll_interval,
                timeout_s=_mn_cli._poll_timeout_from_args(args),
                failure_diag=_summarize_workload_failure,
                quiet_fetch_error_grace_s=_SAFE_GET_WORKLOAD_404_GRACE_S,
                is_quiet_fetch_error=_is_safe_get_workload_404,
            )

        # Resolve the frontend service URL from live service info when possible.
        service_url = st["service_url"]
        try:
            svc = safe.get_workload_service(wid)
            service_url = dynamo_support.frontend_service_url(wid, workspace, svc)
        except safe_client.SafeApiError as exc:
            warn(f"get_workload_service failed ({exc}); using conventional DNS")

        # Discover worker pod IPs from SaFE GetWorkload .pods.
        roles = (
            dynamo_support.discover_role_pods(workload, pd_mode=pd_mode)
            if workload
            else {"frontend": [], "prefill": [], "decode": [], "worker": []}
        )
        worker_ips = [p["podIP"] for p in roles["worker"]]
        prefill_ips = [p["podIP"] for p in roles["prefill"]]
        decode_ips = [p["podIP"] for p in roles["decode"]]
        gpu_ips = (prefill_ips + decode_ips) if pd_mode == "disaggregated" else worker_ips
        if workload and not gpu_ips:
            warn(
                "discovered 0 GPU pod IPs from GetWorkload.pods; SaFE may still "
                "be syncing DGD pods. Re-run create-dynamo to refresh."
            )

    merged = dict(_mn_cli._load_state())
    merged.update(
        {
            "backend": "dynamo",
            "rayjob_id": wid,
            "workspace": workspace,
            "nodes": args.nodes,
            "gpus_per_node": args.gpus_per_node,
            "ssh_key_path": str(priv_key),
            "ssh_port": args.ssh_port,
            "framework": args.backend_framework,
            "pd_mode": pd_mode,
            "kv_transfer_backend": args.kv_transfer_backend,
            "service_url": service_url,
            "worker_pod_ips": worker_ips,
            "prefill_pod_ips": prefill_ips,
            "decode_pod_ips": decode_ips,
            "last_create_request": {
                "image": args.image,
                "nodes": args.nodes,
                "gpus_per_node": args.gpus_per_node,
                "kind": "DynamoDeployment",
                "pd_mode": pd_mode,
            },
        }
    )
    _mn_cli._save_state(merged)

    # Record pod SSH host keys (Dynamo-only control plane).
    if gpu_ips:
        try:
            kh = _mn_cli._refresh_dynamo_known_hosts(gpu_ips, args.ssh_port, state=merged)
            merged["ssh_known_hosts"] = str(kh)
            _mn_cli._save_state(merged)
        except RuntimeError as exc:
            warn(f"ssh-keyscan failed (non-fatal): {exc}")

    # Best-effort SSH reachability probe (non-fatal: pods may still be booting).
    if gpu_ips:
        kh_path = _mn_cli._dynamo_known_hosts_path(merged)
        reachable = sum(
            1
            for ip in gpu_ips
            if ssh_client.probe_ssh(
                ip,
                key_path=priv_key,
                known_hosts=kh_path,
                port=args.ssh_port,
            )
        )
        info(f"ssh reachable GPU pods: {reachable}/{len(gpu_ips)}")

    info(f"state written to {_mn_cli._state_file()}")
    print(json.dumps(merged, indent=2, sort_keys=True))
    return 0

def _dynamo_require_state() -> dict[str, Any]:
    """Load dynamo state; require an ssh key + at least one GPU pod IP.

    Returns:
        dict[str, Any]: The loaded dynamo state.

    Raises:
        RuntimeError: If the state backend is not ``dynamo``, no GPU pod IPs
            are recorded, or the ssh key path is missing.
    """
    state = _mn_cli._load_state()
    if state.get("backend") != "dynamo":
        raise RuntimeError("state backend is not 'dynamo'; run create-dynamo first")
    has_gpu_pods = bool(state.get("worker_pod_ips") or state.get("prefill_pod_ips") or state.get("decode_pod_ips"))
    if not has_gpu_pods:
        raise RuntimeError(
            "no GPU pod IPs in state; re-run create-dynamo (LWS pods may "
            "not have had IPs yet when the workload reached Running)"
        )
    if not state.get("ssh_key_path"):
        raise RuntimeError("no ssh_key_path in state; re-run create-dynamo")
    return state

# Env-var prefixes forwarded from the controller's os.environ to the
# SSH-launched framework child (sandbox-side tuning vars not present in the pod
# container env and not recovered from pid1).
_FORWARD_ENV_PREFIXES = ("MORI_", "SGLANG_MORI_", "SGLANG_DISAGGREGATION_")

def _collect_forward_env() -> dict[str, str]:
    """Read prompt-provided tuning vars from os.environ for SSH forwarding.

    Returns:
        dict[str, str]: Prefix-matched tuning vars, the translated torch
        profiler dir, and any explicit per-variant overrides (which win on
        key collisions).
    """
    fwd = {k: v for k, v in os.environ.items() if any(k.startswith(p) for p in _FORWARD_ENV_PREFIXES)}
    # Translate the controller's shared-FS trace dir into
    # SGLANG_TORCH_PROFILER_DIR so the SSH-launched sglang emits traces to a
    # path readable by both the pods and the sandbox (else traces go to pod-local
    # /tmp and roofline fails).
    trace_dir = os.environ.get("HYPERLOOM_MN_PROFILE_TRACE_DIR", "").strip()
    if trace_dir and "SGLANG_TORCH_PROFILER_DIR" not in fwd:
        fwd["SGLANG_TORCH_PROFILER_DIR"] = trace_dir
    # Explicit per-variant env overrides come through HYPERLOOM_MN_EXTRA_FWD_ENV
    # as a JSON object; forwarded verbatim regardless of prefix and take
    # precedence over prefix-matched values for the same key.
    extra_fwd = os.environ.get("HYPERLOOM_MN_EXTRA_FWD_ENV", "").strip()
    if extra_fwd:
        try:
            parsed = json.loads(extra_fwd)
            if isinstance(parsed, dict):
                for k, v in parsed.items():
                    fwd[str(k)] = str(v)
        except (ValueError, TypeError):
            warn("HYPERLOOM_MN_EXTRA_FWD_ENV is not valid JSON; skipping per-variant env forwarding")
    return filter_forward_env(fwd, warn_on_drop=True)

def _dynamo_fanout_launch(
    state: dict[str, Any],
    launch_args: str,
    worker_ips: list[str],
    *,
    label: str,
    poll_timeout: int,
    print_logs: bool,
) -> tuple[int, list[dict]]:
    """Ship + run launch_dynamo_node.py on each pod in ``worker_ips`` over SSH.

    Returns ``(rc, per_pod_results)``. rc != 0 if any pod's launcher exits
    non-zero. The SAME launch_args go to every pod in the group — each
    self-determines its node-rank from $LWS_WORKER_INDEX pod-side.

    Args:
        state (dict[str, Any]): The dynamo state (ssh key / port).
        launch_args (str): The launcher argv string sent to every pod.
        worker_ips (list[str]): The pod IPs to SSH into.
        label (str): Human-readable label used in log lines.
        poll_timeout (int): Per-pod SSH timeout in seconds.
        print_logs (bool): When ``True``, print each pod's stdout / stderr.

    Returns:
        tuple[int, list[dict]]: ``(rc, per_pod_results)`` where ``rc`` is
        non-zero if any pod's launcher failed.
    """
    script = _mn_cli._read_pod_script("launch_dynamo_node.py")
    forward_env = _collect_forward_env()
    if forward_env:
        info(f"{label}: forwarding {len(forward_env)} tuning env vars to SSH child")
    results: list[dict] = []
    rc_total = 0
    for ip in worker_ips:
        info(f"{label}: ssh -> {ip}:{int(state.get('ssh_port') or ssh_client.DEFAULT_SSH_PORT)}")
        try:
            cp = _mn_cli._dynamo_ssh_run_script(
                state,
                ip,
                script,
                "python3",
                launch_args,
                timeout=poll_timeout,
                env=forward_env,
            )
        except subprocess.TimeoutExpired:
            warn(f"{label}: {ip} timed out after {poll_timeout}s")
            results.append({"podIP": ip, "rc": 124, "error": "timeout"})
            rc_total = 1
            continue
        parsed = _mn_cli._extract_pod_json(cp.stdout or "")
        rec = {"podIP": ip, "rc": cp.returncode, "summary": parsed}
        if cp.returncode != 0:
            rec["stderr"] = (cp.stderr or "")[-1500:]
            rc_total = 1
        results.append(rec)
        if print_logs:
            print(f"--- {ip} stdout ---\n{cp.stdout}\n--- {ip} stderr ---\n{cp.stderr}")
    return rc_total, results

def _dynamo_restart_server(args: argparse.Namespace) -> int:
    """Dynamo restart: SSH fan-out launch_dynamo_node.py to every worker pod.

    Each pod kills its prior server (PID file) and relaunches dynamo.sglang /
    dynamo.vllm wired with --nnodes/--node-rank/--dist-init-addr (rank from the
    pod's own $LWS_WORKER_INDEX). The launcher detaches immediately, so this
    returns once every rank has SPAWNED — readiness (MoE cold start) is polled
    sandbox-side against the frontend service_url, never blocked here.

    Args:
        args (argparse.Namespace): Parsed ``restart-server`` arguments.

    Returns:
        int: ``0`` when every launcher spawned, ``1`` when at least one failed.

    Raises:
        RuntimeError: For an unsupported framework, or when PD disaggregation
            is requested on a non-sglang framework.
    """
    state = _dynamo_require_state()
    framework = (args.framework or state.get("framework") or "sglang").lower()
    if framework not in ("sglang", "vllm"):
        raise RuntimeError(f"unsupported framework: {framework!r}")
    shared_extra = getattr(args, "extra_args", "") or ""
    try:
        validate_server_args(shared_extra, context="dynamo restart-server --extra-args")
        if getattr(args, "pd_prefill_extra_args", ""):
            validate_server_args(
                getattr(args, "pd_prefill_extra_args", "") or "",
                context="dynamo restart-server --pd-prefill-extra-args",
            )
        if getattr(args, "pd_decode_extra_args", ""):
            validate_server_args(
                getattr(args, "pd_decode_extra_args", "") or "",
                context="dynamo restart-server --pd-decode-extra-args",
            )
    except ServerArgsRejected as exc:
        err(str(exc))
        return EXIT_CONFIG_ERROR
    # Topology is fixed at create time, so state.pd_mode is authoritative: a PD
    # deployment must restart in PD mode even if --pd-mode defaulted otherwise.
    pd_mode = (
        "disaggregated"
        if (getattr(args, "pd_mode", "") or "").lower() == "disaggregated" or state.get("pd_mode") == "disaggregated"
        else "aggregated"
    )
    kv = getattr(args, "pd_transfer_backend", "") or state.get("kv_transfer_backend") or ""
    poll_timeout = _mn_cli._poll_timeout_from_args(args)
    print_logs = getattr(args, "print_logs", False)
    rc_total = 0
    all_results: dict[str, Any] = {}

    if pd_mode == "disaggregated":
        if framework != "sglang":
            raise RuntimeError("PD disaggregation is sglang-only on the Dynamo backend")
        # Prefill and decode are each their own LWS; send per-group tp/nnodes and
        # the matching --disaggregation-mode.
        pn = int(getattr(args, "pd_prefill_nodes", 0) or 0) or len(state.get("prefill_pod_ips") or [])
        dn = int(getattr(args, "pd_decode_nodes", 0) or 0) or len(state.get("decode_pod_ips") or [])
        ptp = int(getattr(args, "pd_prefill_tp", 0) or 0) or int(args.tp)
        dtp = int(getattr(args, "pd_decode_tp", 0) or 0) or int(args.tp)
        # Per-role EP / extra-args; 0 / "" falls back to the shared --ep /
        # --extra-args. The shared --extra-args is the base and the per-role
        # string is appended after it (role-specific flags win, last-wins).
        shared_ep = int(getattr(args, "ep", 1) or 1)
        shared_extra = getattr(args, "extra_args", "") or ""
        pep = int(getattr(args, "pd_prefill_ep", 0) or 0) or shared_ep
        dep = int(getattr(args, "pd_decode_ep", 0) or 0) or shared_ep
        prefill_extra = (shared_extra + " " + (getattr(args, "pd_prefill_extra_args", "") or "")).strip()
        decode_extra = (shared_extra + " " + (getattr(args, "pd_decode_extra_args", "") or "")).strip()
        for role, ips, rnnodes, rtp, rep, rextra in (
            ("prefill", state.get("prefill_pod_ips") or [], pn, ptp, pep, prefill_extra),
            ("decode", state.get("decode_pod_ips") or [], dn, dtp, dep, decode_extra),
        ):
            if not ips:
                continue
            launch_args = dynamo_support.build_node_launch_args(
                framework=framework,
                model=args.model,
                tp=rtp,
                nnodes=max(1, rnnodes),
                ep=rep,
                extra_args=rextra,
                health_wait_sec=0,
                disagg_mode=role,
                kv_transfer_backend=kv,
            )
            info(
                f"dynamo restart-server PD {role}: tp={rtp} ep={rep} "
                f"nnodes={rnnodes} pods={len(ips)} kv={kv} extra={rextra!r}"
            )
            rc, results = _dynamo_fanout_launch(
                state,
                launch_args,
                list(ips),
                label=f"restart-{role}",
                poll_timeout=poll_timeout,
                print_logs=print_logs,
            )
            rc_total = rc_total or rc
            all_results[role] = results
    else:
        nnodes = int(state.get("nodes") or 1)
        worker_ips = state.get("worker_pod_ips") or []
        launch_args = dynamo_support.build_node_launch_args(
            framework=framework,
            model=args.model,
            tp=args.tp,
            nnodes=nnodes,
            ep=int(getattr(args, "ep", 1) or 1),
            extra_args=getattr(args, "extra_args", "") or "",
            health_wait_sec=0,
        )
        info(
            f"dynamo restart-server: framework={framework} model={args.model} "
            f"tp={args.tp} nnodes={nnodes} workers={len(worker_ips)}"
        )
        rc_total, results = _dynamo_fanout_launch(
            state,
            launch_args,
            list(worker_ips),
            label="restart",
            poll_timeout=poll_timeout,
            print_logs=print_logs,
        )
        all_results["worker"] = results

    state["last_restart_framework"] = framework
    state["last_restart_model"] = args.model
    state["last_restart_tp"] = int(args.tp)
    state["last_restart_ep"] = int(getattr(args, "ep", 1) or 1)
    state["last_restart_pd_mode"] = pd_mode
    state["last_restart_extra_args"] = _mn_cli._normalize_extra_args(getattr(args, "extra_args", ""))
    if pd_mode == "disaggregated":
        # Persist per-role knobs so a state-only resume reproduces the topology.
        state["last_restart_pd_prefill_nodes"] = int(getattr(args, "pd_prefill_nodes", 0) or 0)
        state["last_restart_pd_decode_nodes"] = int(getattr(args, "pd_decode_nodes", 0) or 0)
        state["last_restart_pd_prefill_tp"] = int(getattr(args, "pd_prefill_tp", 0) or 0)
        state["last_restart_pd_decode_tp"] = int(getattr(args, "pd_decode_tp", 0) or 0)
        state["last_restart_pd_prefill_ep"] = int(getattr(args, "pd_prefill_ep", 0) or 0)
        state["last_restart_pd_decode_ep"] = int(getattr(args, "pd_decode_ep", 0) or 0)
        state["last_restart_pd_prefill_extra_args"] = getattr(args, "pd_prefill_extra_args", "") or ""
        state["last_restart_pd_decode_extra_args"] = getattr(args, "pd_decode_extra_args", "") or ""
    state["last_restart_results"] = all_results
    _mn_cli._save_state(state)
    print(
        json.dumps(
            {"backend": "dynamo", "pd_mode": pd_mode, "rc": rc_total, "results": all_results},
            indent=2,
        )
    )
    if rc_total != 0:
        info("dynamo restart: at least one launcher failed; see results")
        return 1
    info("dynamo servers launched; benchmark via $service_url (frontend :8000)")
    return 0

def _dynamo_all_gpu_ips(state: dict[str, Any]) -> list[str]:
    """Every GPU pod IP to act on: PD => prefill+decode, else worker.

    Args:
        state (dict[str, Any]): The dynamo state.

    Returns:
        list[str]: The GPU pod IPs (prefill + decode when disaggregated, else
        worker).
    """
    pd = (state.get("pd_mode") or "aggregated").lower()
    if pd == "disaggregated":
        return list(state.get("prefill_pod_ips") or []) + list(state.get("decode_pod_ips") or [])
    return list(state.get("worker_pod_ips") or [])

def _dynamo_kill_inference(args: argparse.Namespace) -> int:
    """Dynamo kill: SSH fan-out launch_dynamo_node.py --kill-only to every GPU pod.

    Args:
        args (argparse.Namespace): Parsed ``kill-inference`` arguments.

    Returns:
        int: ``0`` when every pod's kill succeeded, ``1`` otherwise.
    """
    state = _dynamo_require_state()
    framework = (state.get("last_restart_framework") or state.get("framework") or "sglang").lower()
    gpu_ips = _dynamo_all_gpu_ips(state)
    launch_args = dynamo_support.build_node_launch_args(
        framework=framework,
        model="",
        tp=0,
        nnodes=int(state.get("nodes") or 1),
        kill_only=True,
    )
    info(f"dynamo kill-inference: framework={framework} pods={len(gpu_ips)}")
    rc, results = _dynamo_fanout_launch(
        state,
        launch_args,
        gpu_ips,
        label="kill",
        poll_timeout=_mn_cli._poll_timeout_from_args(args),
        print_logs=getattr(args, "print_logs", False),
    )
    state["last_kill_results"] = results
    _mn_cli._save_state(state)
    print(
        json.dumps(
            {"backend": "dynamo", "action": "kill", "rc": rc, "results": results},
            indent=2,
        )
    )
    return 0 if rc == 0 else 1

def _dynamo_ssh_node_op(
    state: dict[str, Any],
    ip: str,
    op_args: str,
    *,
    timeout: int,
) -> tuple[dict | None, dict]:
    """Ship kernel_node_ops.py to one pod over SSH and run one subcommand.

    Returns ``(parsed_json_or_None, transport)`` where transport carries the
    ssh rc / stderr for diagnostics.

    Args:
        state (dict[str, Any]): The dynamo state (ssh key / port).
        ip (str): The pod IP to SSH into.
        op_args (str): The ``kernel_node_ops.py`` subcommand argv string.
        timeout (int): SSH timeout in seconds.

    Returns:
        tuple[dict | None, dict]: ``(parsed_json_or_None, transport)`` where
        ``transport`` carries the ssh rc / stderr.
    """
    script = _mn_cli._read_bundled_pod_python_script("kernel_node_ops.py")
    try:
        cp = _mn_cli._dynamo_ssh_run_script(
            state,
            ip,
            script,
            "python3",
            op_args,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None, {"rc": 124, "stderr": f"timeout after {timeout}s"}
    return _mn_cli._extract_pod_json(cp.stdout or ""), {
        "rc": cp.returncode,
        "stderr": (cp.stderr or "")[-1500:],
    }

def _dynamo_apply_tracelens_patch(args: argparse.Namespace) -> int:
    """Dynamo apply-tracelens-patch: SSH fan-out the TraceLens SGLang patch
    set to every GPU pod via ``apply_tracelens_patch_multinode.py --local``.

    Each GPU pod runs the patcher locally over SSH. Idempotent: the in-pod
    script returns status=skipped on already-patched pods, so it is safe to
    call on every restart.

    Args:
        args (argparse.Namespace): Parsed ``apply-tracelens-patch`` arguments.

    Returns:
        int: ``0`` when every pod applied / skipped, ``1`` on any failure, or
        ``EXIT_CONFIG_ERROR`` when the tracelens root / GPU pods are missing.
    """
    state = _dynamo_require_state()
    tracelens_root = args.tracelens_root or os.environ.get("TRACELENS_ROOT", "").strip()
    if not tracelens_root:
        err(
            "apply-tracelens-patch (dynamo) requires --tracelens-root or "
            "$TRACELENS_ROOT (an NFS path visible from every GPU pod)"
        )
        return EXIT_CONFIG_ERROR
    gpu_ips = _dynamo_all_gpu_ips(state)
    if not gpu_ips:
        err("apply-tracelens-patch (dynamo): no GPU pod IPs in state")
        return EXIT_CONFIG_ERROR
    script = _mn_cli._read_pod_script("apply_tracelens_patch_multinode.py")
    pin = getattr(args, "sglang_version_pin", None) or ""
    op_args = f"--local --tracelens-root {shlex.quote(str(tracelens_root))}"
    if pin:
        op_args += f" --sglang-version-pin {shlex.quote(str(pin))}"
    timeout = _mn_cli._poll_timeout_from_args(args)
    per_pod: list[dict] = []
    failures: list[dict] = []
    # Pod-side interpreter: sglang lives in /opt/venv on the canonical images
    # (/usr/bin/python3 lacks it). Override via $HYPERLOOM_MN_POD_PYTHON.
    pod_python = os.environ.get("HYPERLOOM_MN_POD_PYTHON", "/opt/venv/bin/python")
    for ip in gpu_ips:
        info(f"apply-tracelens-patch (dynamo): ssh -> {ip}")
        try:
            cp = _mn_cli._dynamo_ssh_run_script(
                state,
                ip,
                script,
                pod_python,
                op_args,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            failures.append({"host": ip, "error": f"timeout after {timeout}s"})
            continue
        parsed = _mn_cli._extract_pod_json(cp.stdout or "")
        pods = (parsed or {}).get("per_pod") or []
        if parsed and str(parsed.get("status")) in ("applied", "skipped") and pods:
            for r in pods:
                r["host"] = ip
                per_pod.append(r)
        else:
            failures.append(
                {
                    "host": ip,
                    "error": (parsed or {}).get("error") or (cp.stderr or "")[-800:] or "unknown",
                    "rc": cp.returncode,
                }
            )
    overall = "applied" if not failures else "failed"
    if overall == "applied" and per_pod and all(r.get("status") == "skipped" for r in per_pod):
        overall = "skipped"
    print(
        json.dumps(
            {
                "command": "apply-tracelens-patch",
                "backend": "dynamo",
                "status": overall,
                "per_pod": per_pod,
                "failures": failures,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not failures else 1

def _dynamo_apply_patch(args: argparse.Namespace) -> int:
    """Dynamo apply-patch: SSH fan-out kernel_node_ops.py apply to every GPU pod.

    Emits the SAME JSON shape as kernel_patch_multinode.py (command/status/
    per_node/failures), with ``per_node[].host`` keyed by the pod IP so the
    sandbox builds an IP->backup_path revert map.

    Args:
        args (argparse.Namespace): Parsed ``apply-patch`` arguments.

    Returns:
        int: ``0`` when every pod applied the patch, ``1`` on any failure, or
        ``EXIT_CONFIG_ERROR`` when the patch file is missing.
    """
    state = _dynamo_require_state()
    patch_path = Path(args.patch_file)
    if not patch_path.is_file():
        err(f"patch_file does not exist: {patch_path}")
        return EXIT_CONFIG_ERROR
    patch_b64 = base64.b64encode(patch_path.read_bytes()).decode("ascii")
    gpu_ips = _dynamo_all_gpu_ips(state)
    op_args = (
        f"apply --target-path {shlex.quote(str(args.target_path))} "
        f"--patch-b64 {shlex.quote(str(patch_b64))} "
        f"--backup-dir {shlex.quote(str(args.backup_dir))} "
        f"--kernel-id {shlex.quote(str(args.kernel_id))}"
    )
    per_node: list[dict] = []
    failures: list[dict] = []
    for ip in gpu_ips:
        info(f"apply-patch (dynamo): ssh -> {ip}")
        parsed, tx = _dynamo_ssh_node_op(state, ip, op_args, timeout=args.timeout_sec)
        if parsed and str(parsed.get("status")) == "ok":
            # Key host by pod IP so revert targets the same pod.
            parsed["host"] = ip
            per_node.append(parsed)
        else:
            failures.append({"host": ip, "error": (parsed or {}).get("error") or tx.get("stderr") or "unknown", **tx})
    payload = {
        "command": "apply",
        "target_path": args.target_path,
        "kernel_id": args.kernel_id,
        "backup_dir": args.backup_dir,
        "per_node": per_node,
        "failures": failures,
        "status": "ok" if not failures else "partial",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 1

def _dynamo_revert_patch(args: argparse.Namespace) -> int:
    """Dynamo revert-patch: SSH each pod in the IP->backup_path map + restore.

    Args:
        args (argparse.Namespace): Parsed ``revert-patch`` arguments.

    Returns:
        int: ``0`` when every pod reverted, ``1`` on any failure, or
        ``EXIT_CONFIG_ERROR`` when the backup map is missing / invalid.
    """
    state = _dynamo_require_state()
    try:
        backup_map = json.loads(args.backup_map_json or "{}")
    except json.JSONDecodeError as exc:
        err(f"--backup-map-json not valid JSON: {exc}")
        return EXIT_CONFIG_ERROR
    if not backup_map:
        err("--backup-map-json must be a non-empty {host: backup_path} object")
        return EXIT_CONFIG_ERROR
    per_node: list[dict] = []
    failures: list[dict] = []
    for ip, backup_path in backup_map.items():
        info(f"revert-patch (dynamo): ssh -> {ip}")
        op_args = (
            f"revert --target-path {shlex.quote(str(args.target_path))} "
            f"--backup-path {shlex.quote(str(backup_path))}"
        )
        parsed, tx = _dynamo_ssh_node_op(state, ip, op_args, timeout=args.timeout_sec)
        if parsed and str(parsed.get("status")) in ("restored", "noop_missing_backup"):
            per_node.append({"host": ip, **parsed})
        else:
            failures.append({"host": ip, "error": (parsed or {}).get("error") or tx.get("stderr") or "unknown", **tx})
    payload = {
        "command": "revert",
        "target_path": args.target_path,
        "per_node": per_node,
        "failures": failures,
        "status": "ok" if not failures else "partial",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 1

def _dynamo_kernel_bench(args: argparse.Namespace) -> int:
    """Dynamo kernel-bench: run the micro-benchmark on ONE GPU pod over SSH.

    Args:
        args (argparse.Namespace): Parsed ``kernel-bench`` arguments.

    Returns:
        int: ``0`` when the bench succeeded, ``1`` when it failed, or
        ``EXIT_CONFIG_ERROR`` / ``EXIT_TRANSIENT`` on missing GPU pods / no
        pod JSON.
    """
    state = _dynamo_require_state()
    gpu_ips = _dynamo_all_gpu_ips(state)
    if not gpu_ips:
        err("kernel-bench (dynamo): no GPU pod IPs in state")
        return EXIT_CONFIG_ERROR
    ip = gpu_ips[0]
    if args.files_b64_json:
        try:
            json.loads(args.files_b64_json)
        except json.JSONDecodeError as exc:
            err(f"--files-b64-json not valid JSON: {exc}")
            return EXIT_CONFIG_ERROR
    op_args = (
        f"bench --workspace {shlex.quote(str(args.workspace))} "
        f"--bench-command {shlex.quote(str(args.bench_command))} "
        f"--files-b64-json {shlex.quote(str(args.files_b64_json or '{}'))} "
        f"--result-glob {shlex.quote(str(args.result_glob))} "
        f"--timeout-sec {int(args.timeout_sec)}"
    )
    info(f"kernel-bench (dynamo): ssh -> {ip}")
    parsed, tx = _dynamo_ssh_node_op(
        state,
        ip,
        op_args,
        timeout=args.timeout_sec + 60,
    )
    if parsed is None:
        err(f"kernel-bench (dynamo): no JSON from pod (ssh rc={tx.get('rc')})")
        if getattr(args, "print_logs", False):
            print(tx.get("stderr", ""))
        return EXIT_TRANSIENT
    payload = {"command": "bench", "status": "ok" if str(parsed.get("status")) == "ok" else "failed", "result": parsed}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "ok" else 1

def _resolve_geak_src(explicit: str | None) -> str:
    """Resolve the shared-FS GEAK source dir the sandbox install.sh cloned.

    Resolution: --geak-src > $HYPERLOOM_GEAK_SRC > $HYPERLOOM_ROOT/geak >
    $USER_DATA_PATH/runtime/geak. Must be a path both sandbox and pod see
    (under $USER_DATA_PATH).

    Args:
        explicit (str | None): The ``--geak-src`` flag value, or ``None``.

    Returns:
        str: The resolved GEAK source dir, or ``""`` when no source resolves.
    """
    if explicit and explicit.strip():
        return explicit.strip()
    env = os.environ.get("HYPERLOOM_GEAK_SRC", "").strip()
    if env:
        return env
    root = os.environ.get("HYPERLOOM_ROOT", "").strip()
    if root:
        return f"{root.rstrip('/')}/geak"
    udp = os.environ.get("USER_DATA_PATH", "").strip()
    if udp:
        return f"{udp.rstrip('/')}/runtime/geak"
    return ""

def cmd_install_geak(args: argparse.Namespace) -> int:
    """Install the GEAK CLI on every Dynamo GPU pod over SSH (idempotent).

    pip-installs the shared-FS GEAK checkout into each pod's framework venv so
    the kernel-agent SSH placement (run_geak_over_ssh) finds ``geak`` on PATH.
    Dynamo-only (kernel-agent on RayJob uses the Ray runtime, not this).

    Args:
        args (argparse.Namespace): Parsed ``install-geak`` arguments.

    Returns:
        int: ``0`` when every pod installed / skipped, non-zero on any failure
        (or ``EXIT_CONFIG_ERROR`` when the GEAK source can't be resolved).
    """
    state = _dynamo_require_state()
    geak_src = _resolve_geak_src(getattr(args, "geak_src", None))
    if not geak_src:
        err("install-geak: cannot resolve GEAK source dir; pass --geak-src or set $HYPERLOOM_ROOT / $USER_DATA_PATH")
        return EXIT_CONFIG_ERROR
    script = _mn_cli._read_pod_script("install_geak_node.sh")
    gpu_ips = _dynamo_all_gpu_ips(state)
    info(f"install-geak (dynamo): geak_src={geak_src} pods={len(gpu_ips)}")
    results: list[dict] = []
    rc_total = 0
    for ip in gpu_ips:
        info(f"install-geak: ssh -> {ip}")
        try:
            cp = _mn_cli._dynamo_ssh_run_script(
                state,
                ip,
                script,
                "bash",
                shlex.quote(str(geak_src)),
                timeout=_mn_cli._poll_timeout_from_args(args),
            )
        except subprocess.TimeoutExpired:
            results.append({"host": ip, "status": "failed", "reason": "timeout"})
            rc_total = 1
            continue
        parsed = _mn_cli._extract_pod_json(cp.stdout or "") or {
            "status": "failed",
            "reason": (cp.stderr or "")[-500:],
        }
        parsed["host"] = ip
        results.append(parsed)
        if str(parsed.get("status")) not in ("installed", "skipped"):
            rc_total = 1
        if getattr(args, "print_logs", False):
            print(f"--- {ip} ---\n{cp.stdout}\n{cp.stderr}")
    print(
        json.dumps(
            {"command": "install-geak", "results": results, "status": "ok" if rc_total == 0 else "partial"}, indent=2
        )
    )
    return rc_total
