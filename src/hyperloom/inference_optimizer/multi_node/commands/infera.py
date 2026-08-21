# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Infera (InferaDeployment) idle-pod backend command cluster.

Drives an InferaDeployment the platform already provisioned: idle worker pods
(mn-idle.sh) and an SSH control plane instead of a RayJob with the Ray
Dashboard. The benchmark entry point is the Infera frontend, NOT sglang rank-0
:8888. Each GPU role binds a distinct per-role sshd port
(decode offset by the role stride); ``restart-server`` SSH-fans-out
``launch_infera_node.py`` to every worker pod, which launches
``infera.engine.sglang`` / ``infera.engine.vllm`` per rank.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

from .._internal import infera_support
from .._internal.env_safety import filter_forward_env
from .._internal.log import info, warn, err
from .._internal.server_args_safety import ServerArgsRejected, validate_server_args

import logging

log = logging.getLogger(__name__)


class _MnCliProxy:
    """Lazy proxy preserving ``inf._mn_cli`` monkeypatch compatibility."""

    def __getattr__(self, name: str) -> Any:
        from .. import cli as mn_cli

        return getattr(mn_cli, name)


_mn_cli = _MnCliProxy()


EXIT_CONFIG_ERROR = 3
EXIT_TRANSIENT = 1


def _infera_all_gpu_targets(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Every GPU pod SSH target (PD => prefill+decode, else worker).

    Args:
        state (dict[str, Any]): The infera state.

    Returns:
        list[dict[str, Any]]: The GPU pod SSH target dicts.
    """
    return infera_support.gpu_ssh_targets_from_state(state)


def _infera_target_for_host(state: dict[str, Any], ip: str) -> dict[str, Any]:
    """Resolve a legacy host IP to a full SSH target (with per-role port).

    Args:
        state (dict[str, Any]): The infera state.
        ip (str): The pod IP to resolve.

    Returns:
        dict[str, Any]: The matching SSH target, or a synthesized one using
        the session default SSH port.
    """
    ip = str(ip or "").strip()
    for t in _infera_all_gpu_targets(state):
        if str(t.get("podIP") or "").strip() == ip:
            return dict(t)
    return {"podIP": ip, "sshPort": _mn_cli._infera_default_ssh_port(state), "podId": ""}


def _infera_require_state() -> dict[str, Any]:
    """Load infera state; require an ssh key + at least one GPU pod IP.

    Returns:
        dict[str, Any]: The loaded infera state.

    Raises:
        RuntimeError: If the state backend is not ``infera``, no GPU pod IPs
            are recorded, or the ssh key path is missing.
    """
    state = _mn_cli._load_state()
    if state.get("backend") != "infera":
        raise _mn_cli.ConfigurationError(
            "state backend is not 'infera'; the platform selects the backend, check "
            "INFERENCE_OPTIMIZER_MN_BACKEND and the HYPERLOOM_MN_EXT_* hand-off"
        )
    has_gpu_pods = bool(_infera_all_gpu_targets(state))
    if not has_gpu_pods:
        raise _mn_cli.ConfigurationError(
            f"no GPU pod IPs in state for pd_mode={state.get('pd_mode')!r}. That mode reads "
            f"{'HYPERLOOM_MN_EXT_PREFILL_IPS + _DECODE_IPS' if state.get('pd_mode') == 'disaggregated' else 'HYPERLOOM_MN_EXT_WORKER_IPS'} "
            f"and nothing else, so a hand-off whose IPs live under the other names looks empty here: "
            f"a PD cluster needs PD_MODE=disaggregated set too. (LWS pods may also not have had IPs "
            f"yet when the workload reached Running.)"
        )
    if not state.get("ssh_key_path"):
        raise _mn_cli.ConfigurationError("no ssh_key_path in state; check HYPERLOOM_MN_EXT_SSH_KEY")
    return state


# Env-var prefixes forwarded from the controller's os.environ to the
# SSH-launched framework child (sandbox-side tuning vars not present in the pod
# container env and not recovered from pid1).
_FORWARD_ENV_PREFIXES = ("MORI_", "SGLANG_MORI_", "SGLANG_DISAGGREGATION_")


def _collect_forward_env() -> dict[str, str]:
    """Read prompt-provided tuning vars from os.environ for SSH forwarding.

    Returns:
        dict[str, str]: Prefix-matched tuning vars, the translated torch
        profiler dir, the shared-FS server-log dir, an optional AITER_REBUILD
        signal, and any explicit per-variant overrides (which win on key
        collisions).
    """
    fwd = {k: v for k, v in os.environ.items() if any(k.startswith(p) for p in _FORWARD_ENV_PREFIXES)}
    # Multi-node torch profiler: the infera SSH path (unlike the RayJob path in
    # launch_multinode.py) never pins SGLANG_TORCH_PROFILER_DIR, so sglang writes
    # traces to pod-local /tmp where the sandbox cannot read them -> roofline's
    # profile_no_trace_failed. Translate the controller's shared-FS trace dir
    # (HYPERLOOM_MN_PROFILE_TRACE_DIR, set by restart_server_for_round) into
    # SGLANG_TORCH_PROFILER_DIR so the SSH-launched sglang emits traces to the
    # wekafs path both server pods and the sandbox mount.
    trace_dir = os.environ.get("HYPERLOOM_MN_PROFILE_TRACE_DIR", "").strip()
    if trace_dir and "SGLANG_TORCH_PROFILER_DIR" not in fwd:
        fwd["SGLANG_TORCH_PROFILER_DIR"] = trace_dir
    unset_fwd = os.environ.get("HYPERLOOM_MN_UNSET_FWD_ENV", "").strip()
    if unset_fwd:
        try:
            parsed_unset = json.loads(unset_fwd)
            if isinstance(parsed_unset, list):
                for key in parsed_unset:
                    fwd.pop(str(key), None)
        except (ValueError, TypeError):
            warn("HYPERLOOM_MN_UNSET_FWD_ENV is not valid JSON; skipping per-variant env unsets")
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
    # Expand any $VAR (e.g. $USER_DATA_PATH) left in the profiler dir so the
    # SSH-launched sglang on the pod (where those vars are undefined) writes
    # traces to an absolute shared-FS path, not an unresolved literal.
    if fwd.get("SGLANG_TORCH_PROFILER_DIR"):
        fwd["SGLANG_TORCH_PROFILER_DIR"] = os.path.expandvars(fwd["SGLANG_TORCH_PROFILER_DIR"])
    # Forward a shared-FS (WekaFS) server-log dir so the SSH-launched sglang
    # writes server.log to shared storage the client can read, not pod-local
    # /tmp. Absolute-only; launch_infera_node adds a per-pod suffix to avoid
    # prefill/decode collisions.
    _slog = os.path.expandvars(
        os.environ.get("HYPERLOOM_MN_SERVER_LOG_DIR", "").strip() or "$USER_DATA_PATH/server_logs"
    )
    if _slog.startswith("/") and "$" not in _slog:
        fwd["HYPERLOOM_MN_SERVER_LOG_DIR"] = _slog
    # aiter/cpp_itfs runtime-compiled kernels (GH #458): the integrate re-baseline
    # sets AITER_REBUILD=1 (transiently) so aiter wipes its build dir and
    # recompiles the patched kernel on import. Forward it to the SSH-launched
    # sglang, else the pod reuses the params-hashed pre-patch .so (measures the
    # unpatched kernel). Absent on normal rounds, so this is a no-op then.
    aiter_rebuild = os.environ.get("AITER_REBUILD", "").strip()
    if aiter_rebuild:
        fwd["AITER_REBUILD"] = aiter_rebuild
    return filter_forward_env(fwd, warn_on_drop=True)


def _infera_fanout_launch(
    state: dict[str, Any],
    launch_args: str,
    targets: list[dict[str, Any]],
    *,
    label: str,
    poll_timeout: int,
    print_logs: bool,
) -> tuple[int, list[dict]]:
    """Ship + run launch_infera_node.py on each GPU pod over SSH.

    Returns ``(rc, per_pod_results)``. rc != 0 if any pod's launcher exits
    non-zero. The SAME launch_args go to every pod in the group — each
    self-determines its node-rank from $LWS_WORKER_INDEX pod-side.

    Args:
        state (dict[str, Any]): The infera state (ssh key / port / known_hosts).
        launch_args (str): The launcher argv string sent to every pod.
        targets (list[dict]): SSH targets ``{podIP, sshPort, podId?, role?}``.
        label (str): Human-readable label used in log lines.
        poll_timeout (int): Per-pod SSH timeout in seconds.
        print_logs (bool): When ``True``, print each pod's stdout / stderr.

    Returns:
        tuple[int, list[dict]]: ``(rc, per_pod_results)`` where ``rc`` is
        non-zero if any pod's launcher failed.
    """
    script = _mn_cli._read_pod_script("launch_infera_node.py")
    forward_env = _collect_forward_env()
    if forward_env:
        info(f"{label}: forwarding {len(forward_env)} tuning env vars to SSH child")
    results: list[dict] = []
    rc_total = 0
    for target in targets:
        ip = str(target.get("podIP") or "").strip()
        if not ip:
            continue
        port = int(target.get("sshPort") or _mn_cli._infera_default_ssh_port(state))
        info(f"{label}: ssh -> {ip}:{port}")
        try:
            cp = _mn_cli._infera_ssh_run_script(
                state,
                ip,
                script,
                "python3",
                launch_args,
                timeout=poll_timeout,
                env=forward_env,
                port=port,
            )
        except subprocess.TimeoutExpired:
            warn(f"{label}: {ip}:{port} timed out after {poll_timeout}s")
            results.append({"podIP": ip, "sshPort": port, "rc": 124, "error": "timeout"})
            rc_total = 1
            continue
        parsed = _mn_cli._extract_pod_json(cp.stdout or "")
        rec = {"podIP": ip, "sshPort": port, "rc": cp.returncode, "summary": parsed}
        if target.get("podId"):
            rec["podId"] = target["podId"]
        if cp.returncode != 0:
            rec["stderr"] = (cp.stderr or "")[-1500:]
            rc_total = 1
        results.append(rec)
        if print_logs:
            print(f"--- {ip}:{port} stdout ---\n{cp.stdout}\n--- {ip}:{port} stderr ---\n{cp.stderr}")
    return rc_total, results


# Pod-side one-liner: is the recorded server PID still alive? Emits MN_ALIVE /
# MN_DEAD so the controller can decide whether an Infera resume is safe.
_INFERA_PID_PROBE = (
    'pid="$(cat /tmp/mn_infera_server.pid 2>/dev/null || true)"; '
    'if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then echo MN_ALIVE; else echo MN_DEAD; fi'
)


def _pd_role_pod_counts(
    state: dict[str, Any],
    *,
    prefill_targets: list[dict[str, Any]] | None = None,
    decode_targets: list[dict[str, Any]] | None = None,
) -> tuple[int, int]:
    """Return prefill/decode pod counts from targets or persisted state lists.

    Args:
        state: Multi-node state dict (may carry ``prefill_pod_ips`` / ``decode_pod_ips``).
        prefill_targets: Optional resolved prefill SSH targets (wins over state).
        decode_targets: Optional resolved decode SSH targets (wins over state).

    Returns:
        tuple[int, int]: ``(prefill_count, decode_count)``.
    """
    if prefill_targets is not None:
        prefill_n = len(prefill_targets)
    else:
        prefill_n = len(state.get("prefill_pod_ips") or state.get("prefill_pods") or [])
    if decode_targets is not None:
        decode_n = len(decode_targets)
    else:
        decode_n = len(state.get("decode_pod_ips") or state.get("decode_pods") or [])
    return prefill_n, decode_n


def _resolve_pd_node_counts(
    args: argparse.Namespace,
    state: dict[str, Any],
    *,
    prefill_targets: list[dict[str, Any]] | None = None,
    decode_targets: list[dict[str, Any]] | None = None,
) -> tuple[int, int]:
    """Resolve PD group sizes: explicit CLI wins, else pod-list length.

    Matches ``external_state`` synthesis and lifecycle ``_resolve_pd_args`` so
    persisted ``last_restart_pd_*`` equals the ``nnodes`` used at launch.

    Args:
        args: Parsed ``restart-server`` / create arguments.
        state: Multi-node state for pod-list fallback.
        prefill_targets: Optional prefill targets (wins over state lists).
        decode_targets: Optional decode targets (wins over state lists).

    Returns:
        tuple[int, int]: ``(pd_prefill_nodes, pd_decode_nodes)``.
    """
    prefill_n, decode_n = _pd_role_pod_counts(
        state,
        prefill_targets=prefill_targets,
        decode_targets=decode_targets,
    )
    pn = int(getattr(args, "pd_prefill_nodes", 0) or 0) or prefill_n
    dn = int(getattr(args, "pd_decode_nodes", 0) or 0) or decode_n
    return pn, dn


def _infera_restart_config_matches(
    state: dict[str, Any],
    args: argparse.Namespace,
    framework: str,
    pd_mode: str,
) -> bool:
    """Whether the requested restart matches the last successful Infera launch.

    Mirrors the RayJob resume fast-path config check: same framework / model /
    tp / ep / pd_mode / normalized extra-args (plus per-role PD knobs when
    disaggregated). A mismatch means the round changed a served flag, so the
    server MUST be relaunched (never resumed) or the benchmark would measure
    stale config.

    Args:
        state (dict[str, Any]): The infera multi-node state (carries
            ``last_restart_*``).
        args (argparse.Namespace): Parsed ``restart-server`` arguments.
        framework (str): Resolved framework for this restart.
        pd_mode (str): Resolved PD mode (``"aggregated"`` / ``"disaggregated"``).

    Returns:
        bool: True only when every served-config field matches the last launch.
    """
    if not state.get("last_restart_framework"):
        return False
    base_match = (
        str(state.get("last_restart_framework") or "") == framework
        and str(state.get("last_restart_model") or "") == str(args.model)
        and int(state.get("last_restart_tp") or 0) == int(args.tp)
        and int(state.get("last_restart_ep") or 1) == int(getattr(args, "ep", 1) or 1)
        and str(state.get("last_restart_pd_mode") or "aggregated") == pd_mode
        and _mn_cli._normalize_extra_args(state.get("last_restart_extra_args"))
        == _mn_cli._normalize_extra_args(getattr(args, "extra_args", ""))
    )
    if not base_match:
        return False
    if pd_mode != "disaggregated":
        return True
    # Compare effective PD topology (CLI explicit > pod-list inference), not raw
    # CLI zeros left unset by the operator.
    prefill_n, decode_n = _pd_role_pod_counts(state)
    args_pn, args_dn = _resolve_pd_node_counts(args, state)
    state_pn = int(state.get("last_restart_pd_prefill_nodes") or 0) or prefill_n
    state_dn = int(state.get("last_restart_pd_decode_nodes") or 0) or decode_n
    return (
        args_pn == state_pn
        and args_dn == state_dn
        and int(state.get("last_restart_pd_prefill_tp") or 0) == int(getattr(args, "pd_prefill_tp", 0) or 0)
        and int(state.get("last_restart_pd_decode_tp") or 0) == int(getattr(args, "pd_decode_tp", 0) or 0)
        and int(state.get("last_restart_pd_prefill_ep") or 0) == int(getattr(args, "pd_prefill_ep", 0) or 0)
        and int(state.get("last_restart_pd_decode_ep") or 0) == int(getattr(args, "pd_decode_ep", 0) or 0)
        and (state.get("last_restart_pd_prefill_extra_args") or "")
        == (getattr(args, "pd_prefill_extra_args", "") or "")
        and (state.get("last_restart_pd_decode_extra_args") or "") == (getattr(args, "pd_decode_extra_args", "") or "")
    )


def _infera_servers_alive(
    state: dict[str, Any],
    targets: list[dict[str, Any]],
    *,
    timeout: int,
) -> bool:
    """Whether EVERY GPU pod still has its prior server process alive.

    SSH-probes each pod's ``/tmp/mn_infera_server.pid`` (``kill -0``). Returns
    True only when every pod reports the recorded PID alive. Any dead /
    unreachable / malformed pod yields False (fail-safe: the caller then does a
    full kill+relaunch rather than risk resuming a dead server).

    Args:
        state (dict[str, Any]): The infera multi-node state.
        targets (list[dict[str, Any]]): GPU pod SSH targets to probe.
        timeout (int): Per-pod SSH timeout in seconds.

    Returns:
        bool: True iff every pod's recorded server PID is alive.
    """
    if not targets:
        return False
    for target in targets:
        ip = str(target.get("podIP") or "").strip()
        if not ip:
            return False
        port = int(target.get("sshPort") or _mn_cli._infera_default_ssh_port(state))
        try:
            cp = _mn_cli._infera_ssh_bash_with_env(state, ip, _INFERA_PID_PROBE, None, timeout=timeout, port=port)
        except subprocess.TimeoutExpired:
            return False
        if cp.returncode != 0 or "MN_ALIVE" not in (cp.stdout or ""):
            return False
    return True


def _infera_restart_server(args: argparse.Namespace) -> int:
    """Infera restart: SSH fan-out launch_infera_node.py to every worker pod.

    Each pod kills its prior server (PID file) and relaunches
    infera.engine.sglang / infera.engine.vllm wired with
    --nnodes/--node-rank/--dist-init-addr (rank from the pod's own
    $LWS_WORKER_INDEX). The launcher detaches immediately, so this returns
    once every rank has SPAWNED — readiness (MoE cold start) is polled
    sandbox-side against the frontend service_url, never blocked here.

    Args:
        args (argparse.Namespace): Parsed ``restart-server`` arguments.

    Returns:
        int: ``0`` when every launcher spawned, ``1`` when at least one failed.

    Raises:
        RuntimeError: For an unsupported framework, or when PD disaggregation
            is requested on a non-sglang framework.
    """
    state = _infera_require_state()
    framework = (args.framework or state.get("framework") or "sglang").lower()
    if framework not in ("sglang", "vllm"):
        raise _mn_cli.ConfigurationError(f"unsupported framework: {framework!r}")
    shared_extra = getattr(args, "extra_args", "") or ""
    try:
        validate_server_args(shared_extra, context="infera restart-server --extra-args")
        if getattr(args, "pd_prefill_extra_args", ""):
            validate_server_args(
                getattr(args, "pd_prefill_extra_args", "") or "",
                context="infera restart-server --pd-prefill-extra-args",
            )
        if getattr(args, "pd_decode_extra_args", ""):
            validate_server_args(
                getattr(args, "pd_decode_extra_args", "") or "",
                context="infera restart-server --pd-decode-extra-args",
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
    # Empty is a valid answer: it leaves the flag off so sglang picks its own
    # default, which is the backend preferred on this fabric anyway.
    kv = getattr(args, "pd_transfer_backend", "") or ""
    poll_timeout = _mn_cli._poll_timeout_from_args(args)
    print_logs = getattr(args, "print_logs", False)
    rc_total = 0
    all_results: dict[str, Any] = {}
    pd_prefill_nodes = 0
    pd_decode_nodes = 0

    # Resume fast-path (parity with the RayJob path): if this restart's config
    # matches the last successful launch AND every GPU pod's prior server is
    # still alive, skip the SSH kill+relaunch (which re-triggers a multi-minute
    # MoE cold start). Disabled via MULTI_NODE_RESTART_RESUME_RUNNING=0, which
    # restart_server_for_round scopes when force_full_restart is set (e.g. after
    # a kernel patch, so the pod re-imports patched modules). Fail-safe: any
    # unreachable/dead pod falls through to a full relaunch; the sandbox-side
    # /health wait + reclaim-retry are the final backstop for a stale resume.
    resume_enabled = os.environ.get("MULTI_NODE_RESTART_RESUME_RUNNING", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    if resume_enabled and _infera_restart_config_matches(state, args, framework, pd_mode):
        probe_timeout = min(30, max(10, int(poll_timeout)))
        if _infera_servers_alive(state, _infera_all_gpu_targets(state), timeout=probe_timeout):
            info(
                f"infera resume: same config (framework={framework} model={args.model} "
                f"tp={args.tp} pd_mode={pd_mode}) and all pods alive; skipping kill+launch"
            )
            print(
                json.dumps(
                    {"backend": "infera", "pd_mode": pd_mode, "rc": 0, "resumed": True},
                    indent=2,
                )
            )
            return 0

    if pd_mode == "disaggregated":
        if framework != "sglang":
            raise _mn_cli.ConfigurationError("PD disaggregation is sglang-only on the Infera backend")
        # Prefill group + decode group: each is its own LWS, each pod uses its
        # own $LWS_WORKER_INDEX/$LWS_LEADER_ADDRESS. We send per-group tp/nnodes
        # and the matching --disaggregation-mode.
        prefill_targets = infera_support.pod_targets_from_lists(
            state.get("prefill_pods"),
            state.get("prefill_pod_ips"),
            default_port=_mn_cli._infera_default_ssh_port(state),
            default_role="prefill",
        )
        decode_targets = infera_support.pod_targets_from_lists(
            state.get("decode_pods"),
            state.get("decode_pod_ips"),
            default_port=_mn_cli._infera_default_ssh_port(state) + infera_support.INFERA_SSH_PORT_ROLE_STRIDE,
            default_role="decode",
        )
        # A PD restart needs both legs. Skipping an empty one used to leave
        # rc_total at 0, so a --pd-mode disaggregated run against a state whose
        # pod lists are aggregated made no SSH connection at all and still
        # reported "infera servers launched" and exit 0 -- after which the round
        # benchmarked whatever was already running and called it a result.
        missing = [role for role, targets in (("prefill", prefill_targets), ("decode", decode_targets)) if not targets]
        if missing:
            raise _mn_cli.ConfigurationError(
                f"PD disaggregation was requested but the state records no {' or '.join(missing)} pods. "
                f"state pd_mode={state.get('pd_mode')!r}: the pod lists are chosen by the state's own mode, "
                f"so a PD hand-off needs PD_MODE=disaggregated (or --pd-mode disaggregated on `optimize`, "
                f"which exports it) alongside HYPERLOOM_MN_EXT_PREFILL_IPS / _DECODE_IPS."
            )
        pd_prefill_nodes, pd_decode_nodes = _resolve_pd_node_counts(
            args,
            state,
            prefill_targets=prefill_targets,
            decode_targets=decode_targets,
        )
        pn, dn = pd_prefill_nodes, pd_decode_nodes
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
        for role, targets, rnnodes, rtp, rep, rextra in (
            ("prefill", prefill_targets, pn, ptp, pep, prefill_extra),
            ("decode", decode_targets, dn, dtp, dep, decode_extra),
        ):
            launch_args = infera_support.build_node_launch_args(
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
                f"infera restart-server PD {role}: tp={rtp} ep={rep} "
                f"nnodes={rnnodes} pods={len(targets)} kv={kv} extra={rextra!r}"
            )
            rc, results = _infera_fanout_launch(
                state,
                launch_args,
                list(targets),
                label=f"restart-{role}",
                poll_timeout=poll_timeout,
                print_logs=print_logs,
            )
            rc_total = rc_total or rc
            all_results[role] = results
    else:
        nnodes = int(state.get("nodes") or 1)
        worker_targets = infera_support.pod_targets_from_lists(
            state.get("worker_pods"),
            state.get("worker_pod_ips"),
            default_port=_mn_cli._infera_default_ssh_port(state),
            default_role="worker",
        )
        launch_args = infera_support.build_node_launch_args(
            framework=framework,
            model=args.model,
            tp=args.tp,
            nnodes=nnodes,
            ep=int(getattr(args, "ep", 1) or 1),
            extra_args=getattr(args, "extra_args", "") or "",
            health_wait_sec=0,
        )
        info(
            f"infera restart-server: framework={framework} model={args.model} "
            f"tp={args.tp} nnodes={nnodes} workers={len(worker_targets)}"
        )
        rc_total, results = _infera_fanout_launch(
            state,
            launch_args,
            list(worker_targets),
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
        # Persist inferred PD topology so resume fast-path and KB keys match launch.
        state["pd_prefill_nodes"] = pd_prefill_nodes
        state["pd_decode_nodes"] = pd_decode_nodes
        state["last_restart_pd_prefill_nodes"] = pd_prefill_nodes
        state["last_restart_pd_decode_nodes"] = pd_decode_nodes
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
            {"backend": "infera", "pd_mode": pd_mode, "rc": rc_total, "results": all_results},
            indent=2,
        )
    )
    if rc_total != 0:
        info("infera restart: at least one launcher failed; see results")
        return 1
    info("infera servers launched; benchmark via $service_url (frontend :8000)")
    return 0


def _infera_kill_inference(args: argparse.Namespace) -> int:
    """Infera kill: SSH fan-out launch_infera_node.py --kill-only to every GPU pod.

    Args:
        args (argparse.Namespace): Parsed ``kill-inference`` arguments.

    Returns:
        int: ``0`` when every pod's kill succeeded, ``1`` otherwise.
    """
    state = _infera_require_state()
    framework = (state.get("last_restart_framework") or state.get("framework") or "sglang").lower()
    gpu_targets = _infera_all_gpu_targets(state)
    launch_args = infera_support.build_node_launch_args(
        framework=framework,
        model="",
        tp=0,
        nnodes=int(state.get("nodes") or 1),
        kill_only=True,
    )
    info(f"infera kill-inference: framework={framework} pods={len(gpu_targets)}")
    rc, results = _infera_fanout_launch(
        state,
        launch_args,
        gpu_targets,
        label="kill",
        poll_timeout=_mn_cli._poll_timeout_from_args(args),
        print_logs=getattr(args, "print_logs", False),
    )
    state["last_kill_results"] = results
    _mn_cli._save_state(state)
    print(
        json.dumps(
            {"backend": "infera", "action": "kill", "rc": rc, "results": results},
            indent=2,
        )
    )
    return 0 if rc == 0 else 1


def _infera_ssh_node_op(
    state: dict[str, Any],
    target: dict[str, Any],
    op_args: str,
    *,
    timeout: int,
) -> tuple[dict | None, dict]:
    """Ship kernel_node_ops.py to one pod over SSH and run one subcommand.

    Returns ``(parsed_json_or_None, transport)`` where transport carries the
    ssh rc / stderr for diagnostics.

    Args:
        state (dict[str, Any]): The infera state (ssh key / port).
        target (dict[str, Any]): SSH target ``{podIP, sshPort, podId?}``.
        op_args (str): The ``kernel_node_ops.py`` subcommand argv string.
        timeout (int): SSH timeout in seconds.

    Returns:
        tuple[dict | None, dict]: ``(parsed_json_or_None, transport)`` where
        ``transport`` carries the ssh rc / stderr.
    """
    ip = str(target.get("podIP") or "").strip()
    port = int(target.get("sshPort") or _mn_cli._infera_default_ssh_port(state))
    script = _mn_cli._read_bundled_pod_python_script("kernel_node_ops.py")
    try:
        cp = _mn_cli._infera_ssh_run_script(
            state,
            ip,
            script,
            "python3",
            op_args,
            timeout=timeout,
            port=port,
        )
    except subprocess.TimeoutExpired:
        return None, {"rc": 124, "stderr": f"timeout after {timeout}s", "podIP": ip, "sshPort": port}
    return _mn_cli._extract_pod_json(cp.stdout or ""), {
        "rc": cp.returncode,
        "stderr": (cp.stderr or "")[-1500:],
        "podIP": ip,
        "sshPort": port,
    }


def _infera_apply_tracelens_patch(args: argparse.Namespace) -> int:
    """Infera apply-tracelens-patch: SSH fan-out the TraceLens SGLang patch
    set to every GPU pod via ``apply_tracelens_patch_multinode.py --local``.

    The ray path submits a Ray-actor fan-out; the infera path has no ray, so
    each GPU pod runs the patcher locally over SSH. Both annotate the sglang
    torch.profiler output the same way, so the NFS-shared trace dir (see
    SGLANG_TORCH_PROFILER_DIR forwarding in _collect_forward_env) is consumable
    by TraceLens identically — only the dispatch differs. Idempotent: the
    in-pod script sentinel-greps and returns status=skipped on already-patched
    pods, so it is safe to call on every restart_server_for_round.

    Args:
        args (argparse.Namespace): Parsed ``apply-tracelens-patch`` arguments.

    Returns:
        int: ``0`` when every pod applied / skipped, ``1`` on any failure, or
        ``EXIT_CONFIG_ERROR`` when the tracelens root / GPU pods are missing.
    """
    state = _infera_require_state()
    tracelens_root = args.tracelens_root or os.environ.get("TRACELENS_ROOT", "").strip()
    if not tracelens_root:
        err(
            "apply-tracelens-patch (infera) requires --tracelens-root or "
            "$TRACELENS_ROOT (an NFS path visible from every GPU pod)"
        )
        return EXIT_CONFIG_ERROR
    gpu_targets = _infera_all_gpu_targets(state)
    if not gpu_targets:
        err("apply-tracelens-patch (infera): no GPU pod IPs in state")
        return EXIT_CONFIG_ERROR
    script = _mn_cli._read_pod_script("apply_tracelens_patch_multinode.py")
    pin = getattr(args, "sglang_version_pin", None) or ""
    op_args = f"--local --tracelens-root {shlex.quote(str(tracelens_root))}"
    if pin:
        op_args += f" --sglang-version-pin {shlex.quote(str(pin))}"
    timeout = _mn_cli._poll_timeout_from_args(args)
    per_pod: list[dict] = []
    failures: list[dict] = []
    # Pod-side interpreter: sglang lives in /opt/venv on the canonical
    # ROCm sglang-infera images; /usr/bin/python3 lacks sglang so
    # _apply_on_pod's `import sglang` fails with "No module named 'sglang'".
    # Allow override via $HYPERLOOM_MN_POD_PYTHON.
    pod_python = os.environ.get("HYPERLOOM_MN_POD_PYTHON", "/opt/venv/bin/python")
    for target in gpu_targets:
        ip = str(target.get("podIP") or "").strip()
        port = int(target.get("sshPort") or _mn_cli._infera_default_ssh_port(state))
        info(f"apply-tracelens-patch (infera): ssh -> {ip}:{port}")
        try:
            cp = _mn_cli._infera_ssh_run_script(
                state,
                ip,
                script,
                pod_python,
                op_args,
                timeout=timeout,
                port=port,
            )
        except subprocess.TimeoutExpired:
            failures.append({"host": ip, "sshPort": port, "error": f"timeout after {timeout}s"})
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
                    "sshPort": port,
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
                "backend": "infera",
                "status": overall,
                "per_pod": per_pod,
                "failures": failures,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not failures else 1


def _infera_apply_patch(args: argparse.Namespace) -> int:
    """Infera apply-patch: SSH fan-out kernel_node_ops.py apply to every GPU pod.

    Emits the SAME JSON shape as kernel_patch_multinode.py (command/status/
    per_node/failures), with ``per_node[].host`` keyed by the pod IP so the
    sandbox builds an IP->backup_path revert map.

    Args:
        args (argparse.Namespace): Parsed ``apply-patch`` arguments.

    Returns:
        int: ``0`` when every pod applied the patch, ``1`` on any failure, or
        ``EXIT_CONFIG_ERROR`` when the patch file is missing.
    """
    state = _infera_require_state()
    patch_path = Path(args.patch_file)
    if not patch_path.is_file():
        err(f"patch_file does not exist: {patch_path}")
        return EXIT_CONFIG_ERROR
    patch_b64 = base64.b64encode(patch_path.read_bytes()).decode("ascii")
    gpu_targets = _infera_all_gpu_targets(state)
    op_args = (
        f"apply --target-path {shlex.quote(str(args.target_path))} "
        f"--patch-b64 {shlex.quote(str(patch_b64))} "
        f"--backup-dir {shlex.quote(str(args.backup_dir))} "
        f"--kernel-id {shlex.quote(str(args.kernel_id))} "
        f"--jit-build-dir "
        f"{shlex.quote(str(getattr(args, 'jit_build_dir', '') or ''))}"
    )
    per_node: list[dict] = []
    failures: list[dict] = []
    for target in gpu_targets:
        ip = str(target.get("podIP") or "").strip()
        port = int(target.get("sshPort") or _mn_cli._infera_default_ssh_port(state))
        info(f"apply-patch (infera): ssh -> {ip}:{port}")
        parsed, tx = _infera_ssh_node_op(state, target, op_args, timeout=args.timeout_sec)
        if parsed and str(parsed.get("status")) == "ok":
            # Key host by pod IP so revert targets the same pod.
            parsed["host"] = ip
            per_node.append(parsed)
        else:
            failures.append(
                {
                    "host": ip,
                    "sshPort": port,
                    "error": (parsed or {}).get("error") or tx.get("stderr") or "unknown",
                    **tx,
                }
            )
    rollback: list[dict] = []
    if failures:
        for record in per_node:
            ip = str(record.get("host") or "")
            target = _infera_target_for_host(state, ip)
            op_args = f"revert --records-json {shlex.quote(json.dumps([record], sort_keys=True))}"
            parsed, tx = _infera_ssh_node_op(
                state,
                target,
                op_args,
                timeout=args.timeout_sec,
            )
            rollback.append({"host": ip, **(parsed or tx)})
    payload = {
        "command": "apply",
        "target_path": args.target_path,
        "kernel_id": args.kernel_id,
        "backup_dir": args.backup_dir,
        "per_node": per_node,
        "failures": failures,
        "rollback": rollback,
        "status": "ok" if not failures else "partial",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 1


def _infera_revert_patch(args: argparse.Namespace) -> int:
    """Infera revert-patch: SSH each pod in the IP->backup_path map + restore.

    Args:
        args (argparse.Namespace): Parsed ``revert-patch`` arguments.

    Returns:
        int: ``0`` when every pod reverted, ``1`` on any failure, or
        ``EXIT_CONFIG_ERROR`` when the backup map is missing / invalid.
    """
    state = _infera_require_state()
    try:
        records_by_host = json.loads(getattr(args, "records_json", "") or "{}")
        backup_map = json.loads(args.backup_map_json or "{}")
    except json.JSONDecodeError as exc:
        err(f"--backup-map-json not valid JSON: {exc}")
        return EXIT_CONFIG_ERROR
    if not records_by_host and not backup_map:
        err("revert requires non-empty records or backup map")
        return EXIT_CONFIG_ERROR
    if not records_by_host:
        records_by_host = {
            host: [
                {
                    "target_path": args.target_path,
                    "backup_path": backup_path,
                }
            ]
            for host, backup_path in backup_map.items()
        }
    per_node: list[dict] = []
    failures: list[dict] = []
    for ip, records in records_by_host.items():
        target = _infera_target_for_host(state, str(ip))
        port = int(target.get("sshPort") or _mn_cli._infera_default_ssh_port(state))
        info(f"revert-patch (infera): ssh -> {ip}:{port}")
        op_args = f"revert --records-json {shlex.quote(json.dumps(records, sort_keys=True))}"
        parsed, tx = _infera_ssh_node_op(state, target, op_args, timeout=args.timeout_sec)
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


def _infera_finalize_patch(args: argparse.Namespace) -> int:
    """Delete accepted patch backups on each Infera pod."""
    state = _infera_require_state()
    try:
        records_by_host = json.loads(args.records_json or "{}")
    except json.JSONDecodeError as exc:
        err(f"--records-json not valid JSON: {exc}")
        return EXIT_CONFIG_ERROR
    per_node, failures = [], []
    for ip, records in records_by_host.items():
        target = _infera_target_for_host(state, str(ip))
        op_args = f"finalize --records-json {shlex.quote(json.dumps(records, sort_keys=True))}"
        parsed, tx = _infera_ssh_node_op(
            state,
            target,
            op_args,
            timeout=args.timeout_sec,
        )
        if parsed and str(parsed.get("status")) == "finalized":
            per_node.append({"host": ip, **parsed})
        else:
            failures.append(
                {
                    "host": ip,
                    "error": (parsed or {}).get("error") or tx.get("stderr") or "unknown",
                    **tx,
                }
            )
    print(
        json.dumps(
            {
                "command": "finalize",
                "per_node": per_node,
                "failures": failures,
                "status": "ok" if not failures else "partial",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not failures else 1


def _infera_kernel_bench(args: argparse.Namespace) -> int:
    """Infera kernel-bench: run the micro-benchmark on ONE GPU pod over SSH.

    Args:
        args (argparse.Namespace): Parsed ``kernel-bench`` arguments.

    Returns:
        int: ``0`` when the bench succeeded, ``1`` when it failed, or
        ``EXIT_CONFIG_ERROR`` / ``EXIT_TRANSIENT`` on missing GPU pods / no
        pod JSON.
    """
    state = _infera_require_state()
    gpu_targets = _infera_all_gpu_targets(state)
    if not gpu_targets:
        err("kernel-bench (infera): no GPU pod IPs in state")
        return EXIT_CONFIG_ERROR
    target = gpu_targets[0]
    ip = str(target.get("podIP") or "").strip()
    port = int(target.get("sshPort") or _mn_cli._infera_default_ssh_port(state))
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
    info(f"kernel-bench (infera): ssh -> {ip}:{port}")
    parsed, tx = _infera_ssh_node_op(
        state,
        target,
        op_args,
        timeout=args.timeout_sec + 60,
    )
    if parsed is None:
        err(f"kernel-bench (infera): no JSON from pod (ssh rc={tx.get('rc')})")
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
    """Install the GEAK CLI on every Infera GPU pod over SSH (idempotent).

    pip-installs the shared-FS GEAK checkout into each pod's framework venv so
    the kernel-agent SSH placement (run_geak_over_ssh) finds ``geak`` on PATH.
    Infera-only (kernel-agent on RayJob uses the Ray runtime, not this).

    Args:
        args (argparse.Namespace): Parsed ``install-geak`` arguments.

    Returns:
        int: ``0`` when every pod installed / skipped, non-zero on any failure
        (or ``EXIT_CONFIG_ERROR`` when the GEAK source can't be resolved).
    """
    state = _infera_require_state()
    geak_src = _resolve_geak_src(getattr(args, "geak_src", None))
    if not geak_src:
        err("install-geak: cannot resolve GEAK source dir; pass --geak-src or set $HYPERLOOM_ROOT / $USER_DATA_PATH")
        return EXIT_CONFIG_ERROR
    script = _mn_cli._read_pod_script("install_geak_node.sh")
    gpu_targets = _infera_all_gpu_targets(state)
    info(f"install-geak (infera): geak_src={geak_src} pods={len(gpu_targets)}")
    results: list[dict] = []
    rc_total = 0
    for target in gpu_targets:
        ip = str(target.get("podIP") or "").strip()
        port = int(target.get("sshPort") or _mn_cli._infera_default_ssh_port(state))
        info(f"install-geak: ssh -> {ip}:{port}")
        try:
            cp = _mn_cli._infera_ssh_run_script(
                state,
                ip,
                script,
                "bash",
                shlex.quote(str(geak_src)),
                timeout=_mn_cli._poll_timeout_from_args(args),
                port=port,
            )
        except subprocess.TimeoutExpired:
            results.append({"host": ip, "sshPort": port, "status": "failed", "reason": "timeout"})
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
            print(f"--- {ip}:{port} ---\n{cp.stdout}\n{cp.stderr}")
    print(
        json.dumps(
            {"command": "install-geak", "results": results, "status": "ok" if rc_total == 0 else "partial"}, indent=2
        )
    )
    return rc_total
