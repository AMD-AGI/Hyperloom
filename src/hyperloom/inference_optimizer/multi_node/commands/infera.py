# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Infera (InferaDeployment) idle-pod backend command cluster."""

from __future__ import annotations

import argparse
import base64
import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .._internal import infera_support
from .._internal.env_safety import filter_forward_env
from .._internal.log import info, warn, err
from .._internal.server_args_safety import ServerArgsRejected, validate_server_args
from .._internal.ssh_client import RemoteScript


class _MnCliProxy:
    """Lazy proxy preserving ``inf._mn_cli`` monkeypatch compatibility."""

    def __getattr__(self, name: str) -> Any:
        from .. import cli as mn_cli

        return getattr(mn_cli, name)


_mn_cli = _MnCliProxy()


def _infera_target_for_host(state: dict[str, Any], ip: str) -> dict[str, Any]:
    """Resolve a legacy host IP to a full SSH target (with per-role port)."""
    ip = str(ip or "").strip()
    for t in infera_support.gpu_ssh_targets_from_state(state):
        if str(t.get("podIP") or "").strip() == ip:
            return dict(t)
    return {"podIP": ip, "sshPort": _mn_cli._infera_default_ssh_port(state), "podId": ""}


def _infera_require_state() -> dict[str, Any]:
    """Load infera state; require an ssh key + at least one GPU pod IP."""
    state = _mn_cli._load_state()
    if state.get("backend") != "infera":
        raise _mn_cli.ConfigurationError(
            "state backend is not 'infera'; the platform selects the backend, check "
            "INFERENCE_OPTIMIZER_MN_BACKEND and the HYPERLOOM_MN_EXT_* hand-off"
        )
    has_gpu_pods = bool(infera_support.gpu_ssh_targets_from_state(state))
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


# Env-var prefixes forwarded from the controller's os.environ to the SSH-launched framework child (sandbox-side tuning
# vars not present in the pod container env and not recovered from pid1).
_FORWARD_ENV_PREFIXES = ("MORI_", "SGLANG_MORI_", "SGLANG_DISAGGREGATION_")


def _collect_forward_env() -> dict[str, str]:
    """Read prompt-provided tuning vars from os.environ for SSH forwarding."""
    fwd = {k: v for k, v in os.environ.items() if any(k.startswith(p) for p in _FORWARD_ENV_PREFIXES)}
    # Multi-node torch profiler: the infera SSH path (unlike the RayJob path in launch_multinode.py) never pins
    # SGLANG_TORCH_PROFILER_DIR, so sglang writes traces to pod-local /tmp where the sandbox cannot read them ->
    # roofline's profile_no_trace_failed.
    trace_dir = os.environ.get("HYPERLOOM_MN_PROFILE_TRACE_DIR", "").strip()
    if trace_dir and "SGLANG_TORCH_PROFILER_DIR" not in fwd:
        fwd["SGLANG_TORCH_PROFILER_DIR"] = trace_dir
    # Forward no-patch shape-discovery config; the pod-side launcher sets
    # PYTHONPATH itself (it is blocked from SSH forwarding).
    for _shape_key in (
        "TRACELENS_ROOT",
        "TRACELENS_SHAPE_DISCOVERY",
        "HYPERLOOM_SGLANG_VERSION_PIN",
    ):
        _shape_val = os.environ.get(_shape_key, "").strip()
        if _shape_val and _shape_key not in fwd:
            fwd[_shape_key] = _shape_val
    # Explicit per-variant overrides are forwarded verbatim regardless of prefix and take precedence over
    # prefix-matched values for the same key; the unsets are applied first so an override can reinstate a key.
    overrides = _mn_cli.per_round_forward_overrides()
    for key in overrides["unset"]:
        fwd.pop(key, None)
    fwd.update(overrides["set"])
    # Expand any $VAR (e.g. $USER_DATA_PATH) left in the profiler dir so the SSH-launched sglang on the pod (where
    # those vars are undefined) writes traces to an absolute shared-FS path, not an unresolved literal.
    if fwd.get("SGLANG_TORCH_PROFILER_DIR"):
        fwd["SGLANG_TORCH_PROFILER_DIR"] = os.path.expandvars(fwd["SGLANG_TORCH_PROFILER_DIR"])
    # Forward a shared-FS (WekaFS) server-log dir so the SSH-launched sglang writes server.log to shared storage the
    # client can read, not pod-local /tmp.
    _slog = os.path.expandvars(
        os.environ.get("HYPERLOOM_MN_SERVER_LOG_DIR", "").strip() or "$USER_DATA_PATH/server_logs"
    )
    if _slog.startswith("/") and "$" not in _slog:
        fwd["HYPERLOOM_MN_SERVER_LOG_DIR"] = _slog
    # aiter/cpp_itfs runtime-compiled kernels (GH #458): the integrate re-baseline sets AITER_REBUILD=1 (transiently)
    # so aiter wipes its build dir and recompiles the patched kernel on import.
    aiter_rebuild = os.environ.get("AITER_REBUILD", "").strip()
    if aiter_rebuild:
        fwd["AITER_REBUILD"] = aiter_rebuild
    return filter_forward_env(fwd)


@dataclass(frozen=True)
class _FanoutOptions:
    """How one infera command runs its SSH fan-outs: the per-pod timeout and whether pod output is echoed."""

    poll_timeout: int
    print_logs: bool


def _infera_fanout_launch(
    state: dict[str, Any],
    launch_args: str,
    targets: list[dict[str, Any]],
    *,
    label: str,
    options: _FanoutOptions,
) -> tuple[int, list[dict]]:
    """Ship + run launch_infera_node.py on each GPU pod over SSH."""
    poll_timeout = options.poll_timeout
    script = _mn_cli._read_bundled_pod_python_script("launch_infera_node.py", _mn_cli._LAUNCHER_DEPS)
    forward_env = _collect_forward_env()
    if forward_env:
        info(f"{label}: forwarding {len(forward_env)} tuning env vars to SSH child")
    results: list[dict] = []
    rc_total = 0
    for target in targets:
        ip = target["podIP"]
        port = int(target.get("sshPort") or _mn_cli._infera_default_ssh_port(state))
        info(f"{label}: ssh -> {ip}:{port}")
        try:
            cp = _mn_cli._infera_ssh_run_script(
                state,
                ip,
                RemoteScript(script, "python3", launch_args, env=forward_env),
                timeout=poll_timeout,
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
        if options.print_logs:
            print(f"--- {ip}:{port} stdout ---\n{cp.stdout}\n--- {ip}:{port} stderr ---\n{cp.stderr}")
    return rc_total, results


# Pod-side one-liner: is the recorded server PID still alive?
_INFERA_PID_PROBE = (
    f'pid="$(cat {shlex.quote(infera_support.INFERA_POD_PID_FILE)} 2>/dev/null || true)"; '
    'if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then echo MN_ALIVE; else echo MN_DEAD; fi'
)


def _pd_role_pod_counts(
    state: dict[str, Any],
    *,
    prefill_targets: list[dict[str, Any]] | None = None,
    decode_targets: list[dict[str, Any]] | None = None,
) -> tuple[int, int]:
    """Return prefill/decode pod counts from targets or persisted state lists."""
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
    """Resolve PD group sizes: explicit CLI wins, else pod-list length."""
    prefill_n, decode_n = _pd_role_pod_counts(
        state,
        prefill_targets=prefill_targets,
        decode_targets=decode_targets,
    )
    pn = int(getattr(args, "pd_prefill_nodes", 0) or 0) or prefill_n
    dn = int(getattr(args, "pd_decode_nodes", 0) or 0) or decode_n
    return pn, dn


# Per-role PD overrides a PD launch records as ``last_restart_<name>``, with the value that leaves the role on the
# shared --tp / --ep / --extra-args.
_PD_ROLE_OVERRIDES = {
    "pd_prefill_tp": 0,
    "pd_decode_tp": 0,
    "pd_prefill_ep": 0,
    "pd_decode_ep": 0,
    "pd_prefill_extra_args": "",
    "pd_decode_extra_args": "",
}


def _infera_restart_config_matches(
    state: dict[str, Any],
    args: argparse.Namespace,
    framework: str,
    pd_mode: str,
    kv_transfer_backend: str = "",
) -> bool:
    """Whether the requested restart matches the last successful Infera launch."""
    base_match = (
        state.get("last_restart_framework") == framework
        and state.get("last_restart_model") == args.model
        and state.get("last_restart_tp") == int(args.tp)
        and int(state.get("last_restart_ep") or 1) == int(getattr(args, "ep", 1) or 1)
        and str(state.get("last_restart_pd_mode") or "aggregated") == pd_mode
        and _mn_cli._normalize_extra_args(state.get("last_restart_extra_args"))
        == _mn_cli._normalize_extra_args(getattr(args, "extra_args", ""))
        # The servers were launched over SSH with these, so a round that changes only them needs a relaunch to take
        # effect; resuming would benchmark the previous environment and report the new one.
        and state.get("last_restart_forward_env") == _mn_cli.per_round_forward_overrides()
    )
    if not base_match:
        return False
    if pd_mode != "disaggregated":
        return True
    for name, unset in _PD_ROLE_OVERRIDES.items():
        if (state.get(f"last_restart_{name}") or unset) != getattr(args, name):
            return False
    # Compare effective PD topology (CLI explicit > pod-list inference), not raw CLI zeros left unset by the operator.
    prefill_n, decode_n = _pd_role_pod_counts(state)
    args_pn, args_dn = _resolve_pd_node_counts(args, state)
    return (
        args_pn == (state.get("last_restart_pd_prefill_nodes") or prefill_n)
        and args_dn == (state.get("last_restart_pd_decode_nodes") or decode_n)
        and kv_transfer_backend.strip() == (state.get("last_restart_pd_transfer_backend") or "").strip()
    )


def _infera_servers_alive(
    state: dict[str, Any],
    targets: list[dict[str, Any]],
    *,
    timeout: int,
) -> bool:
    """Whether EVERY GPU pod still has its prior server process alive."""
    if not targets:
        return False
    for target in targets:
        ip = target["podIP"]
        port = int(target.get("sshPort") or _mn_cli._infera_default_ssh_port(state))
        try:
            cp = _mn_cli._infera_ssh_bash_with_env(state, ip, _INFERA_PID_PROBE, timeout=timeout, port=port)
        except subprocess.TimeoutExpired:
            return False
        if cp.returncode != 0 or "MN_ALIVE" not in (cp.stdout or ""):
            return False
    return True


def _infera_launch_pd_groups(
    state: dict[str, Any], args: argparse.Namespace, framework: str, kv: str, fanout: _FanoutOptions
) -> tuple[int, dict[str, Any]]:
    """Launch the prefill and decode groups, then record in ``state`` the PD topology they were launched with."""
    if framework != "sglang":
        raise _mn_cli.ConfigurationError("PD disaggregation is sglang-only on the Infera backend")
    # Prefill group + decode group: each is its own LWS, each pod uses its own
    # $LWS_WORKER_INDEX/$LWS_LEADER_ADDRESS.
    prefill_targets = infera_support.role_ssh_targets(state, "prefill")
    decode_targets = infera_support.role_ssh_targets(state, "decode")
    # A PD restart needs both legs.
    missing = [role for role, targets in (("prefill", prefill_targets), ("decode", decode_targets)) if not targets]
    if missing:
        raise _mn_cli.ConfigurationError(
            f"PD disaggregation was requested but the state records no {' or '.join(missing)} pods. "
            f"state pd_mode={state.get('pd_mode')!r}: the pod lists are chosen by the state's own mode, "
            f"so a PD hand-off needs PD_MODE=disaggregated (or --pd-mode disaggregated on `optimize`, "
            f"which exports it) alongside HYPERLOOM_MN_EXT_PREFILL_IPS / _DECODE_IPS."
        )
    pn, dn = _resolve_pd_node_counts(args, state, prefill_targets=prefill_targets, decode_targets=decode_targets)
    shared_ep = int(getattr(args, "ep", 1) or 1)
    shared_extra = getattr(args, "extra_args", "") or ""
    rc_total = 0
    all_results: dict[str, Any] = {}
    for role, targets, rnnodes in (("prefill", prefill_targets, pn), ("decode", decode_targets, dn)):
        # Per-role TP / EP / extra-args; 0 / "" falls back to the shared --tp / --ep / --extra-args.
        rtp = int(getattr(args, f"pd_{role}_tp", 0) or 0) or int(args.tp)
        rep = int(getattr(args, f"pd_{role}_ep", 0) or 0) or shared_ep
        rextra = (shared_extra + " " + (getattr(args, f"pd_{role}_extra_args", "") or "")).strip()
        launch_args = infera_support.build_node_launch_args(
            infera_support.NodeLaunch(
                framework=framework,
                model=args.model,
                tp=rtp,
                nnodes=max(1, rnnodes),
                ep=rep,
                extra_args=rextra,
                disagg_mode=role,
                kv_transfer_backend=kv,
            )
        )
        info(
            f"infera restart-server PD {role}: tp={rtp} ep={rep} "
            f"nnodes={rnnodes} pods={len(targets)} kv={kv} extra={rextra!r}"
        )
        rc, all_results[role] = _infera_fanout_launch(
            state, launch_args, list(targets), label=f"restart-{role}", options=fanout
        )
        rc_total = rc_total or rc
    # Persist inferred PD topology so resume fast-path and KB keys match launch.
    state["pd_prefill_nodes"] = pn
    state["pd_decode_nodes"] = dn
    state["last_restart_pd_prefill_nodes"] = pn
    state["last_restart_pd_decode_nodes"] = dn
    # The orchestrator's PD arg resolution reads this key as its fallback tier between an explicit argument and
    # $PD_TRANSFER_BACKEND.
    state["last_restart_pd_transfer_backend"] = kv
    for name in _PD_ROLE_OVERRIDES:
        state[f"last_restart_{name}"] = getattr(args, name)
    return rc_total, all_results


def _infera_launch_workers(
    state: dict[str, Any], args: argparse.Namespace, framework: str, fanout: _FanoutOptions
) -> tuple[int, dict[str, Any]]:
    """Launch one aggregated server group across every worker pod."""
    nnodes = int(state.get("nodes") or 1)
    worker_targets = infera_support.role_ssh_targets(state, "worker")
    launch_args = infera_support.build_node_launch_args(
        infera_support.NodeLaunch(
            framework=framework,
            model=args.model,
            tp=args.tp,
            nnodes=nnodes,
            ep=int(getattr(args, "ep", 1) or 1),
            extra_args=getattr(args, "extra_args", "") or "",
        )
    )
    info(
        f"infera restart-server: framework={framework} model={args.model} "
        f"tp={args.tp} nnodes={nnodes} workers={len(worker_targets)}"
    )
    rc, results = _infera_fanout_launch(state, launch_args, list(worker_targets), label="restart", options=fanout)
    return rc, {"worker": results}


def _infera_restart_server(args: argparse.Namespace) -> int:
    """Infera restart: SSH fan-out launch_infera_node.py to every worker pod."""
    state = _infera_require_state()
    framework = (args.framework or state.get("framework") or "sglang").lower()
    if framework not in ("sglang", "vllm"):
        raise _mn_cli.ConfigurationError(f"unsupported framework: {framework!r}")
    try:
        for flag in ("extra_args", "pd_prefill_extra_args", "pd_decode_extra_args"):
            validate_server_args(
                getattr(args, flag, "") or "", context=f"infera restart-server --{flag.replace('_', '-')}"
            )
    except ServerArgsRejected as exc:
        err(str(exc))
        return _mn_cli.EXIT_CONFIG_ERROR
    # Topology is fixed at create time, so state.pd_mode is authoritative: a PD deployment must restart in PD mode
    # even if --pd-mode defaulted otherwise.
    pd_mode = (
        "disaggregated"
        if (getattr(args, "pd_mode", "") or "").lower() == "disaggregated" or state.get("pd_mode") == "disaggregated"
        else "aggregated"
    )
    # Empty is a valid answer: it leaves the flag off so sglang picks its own default, which is the backend preferred
    # on this fabric anyway.
    kv = getattr(args, "pd_transfer_backend", "") or ""
    fanout = _FanoutOptions(_mn_cli._poll_timeout_from_args(args), getattr(args, "print_logs", False))

    # Resume fast-path (parity with the RayJob path): if this restart's config matches the last successful launch AND
    # every GPU pod's prior server is still alive, skip the SSH kill+relaunch (which re-triggers a multi-minute MoE
    # cold start).
    resume_enabled = os.environ.get("MULTI_NODE_RESTART_RESUME_RUNNING", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    if resume_enabled and _infera_restart_config_matches(state, args, framework, pd_mode, kv):
        probe_timeout = min(30, max(10, int(fanout.poll_timeout)))
        if _infera_servers_alive(state, infera_support.gpu_ssh_targets_from_state(state), timeout=probe_timeout):
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
        rc_total, all_results = _infera_launch_pd_groups(state, args, framework, kv, fanout)
    else:
        rc_total, all_results = _infera_launch_workers(state, args, framework, fanout)

    state["last_restart_framework"] = framework
    state["last_restart_model"] = args.model
    state["last_restart_tp"] = int(args.tp)
    state["last_restart_ep"] = int(getattr(args, "ep", 1) or 1)
    state["last_restart_pd_mode"] = pd_mode
    state["last_restart_extra_args"] = _mn_cli._normalize_extra_args(getattr(args, "extra_args", ""))
    state["last_restart_forward_env"] = _mn_cli.per_round_forward_overrides()
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
    """Infera kill: SSH fan-out launch_infera_node.py --kill-only to every GPU pod."""
    state = _infera_require_state()
    framework = (state.get("last_restart_framework") or state.get("framework") or "sglang").lower()
    gpu_targets = infera_support.gpu_ssh_targets_from_state(state)
    launch_args = infera_support.build_node_kill_args(framework)
    info(f"infera kill-inference: framework={framework} pods={len(gpu_targets)}")
    rc, results = _infera_fanout_launch(
        state,
        launch_args,
        gpu_targets,
        label="kill",
        options=_FanoutOptions(_mn_cli._poll_timeout_from_args(args), getattr(args, "print_logs", False)),
    )
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
    """Ship kernel_node_ops.py to one pod over SSH and run one subcommand."""
    ip = target["podIP"]
    port = int(target.get("sshPort") or _mn_cli._infera_default_ssh_port(state))
    script = _mn_cli._read_bundled_pod_python_script("kernel_node_ops.py", _mn_cli._KERNEL_NODE_OPS_DEPS)
    try:
        cp = _mn_cli._infera_ssh_run_script(
            state,
            ip,
            RemoteScript(script, "python3", op_args),
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
    """Infera apply-tracelens-patch: SSH fan-out the TraceLens SGLang patch set to every GPU pod via ``apply_tracelens_patch_multinode.py --local``."""
    state = _infera_require_state()
    tracelens_root = args.tracelens_root or os.environ.get("TRACELENS_ROOT", "").strip()
    if not tracelens_root:
        err(
            "apply-tracelens-patch (infera) requires --tracelens-root or "
            "$TRACELENS_ROOT (an NFS path visible from every GPU pod)"
        )
        return _mn_cli.EXIT_CONFIG_ERROR
    gpu_targets = infera_support.gpu_ssh_targets_from_state(state)
    script = _mn_cli._read_bundled_pod_python_script("apply_tracelens_patch_multinode.py", _mn_cli._SHAPE_GATE_DEPS)
    op_args = f"--local --tracelens-root {shlex.quote(str(tracelens_root))}"
    if args.sglang_version_pin:
        op_args += f" --sglang-version-pin {shlex.quote(str(args.sglang_version_pin))}"
    patch_set = getattr(args, "patch_set", None)
    if patch_set:
        op_args += f" --patch-set {shlex.quote(str(patch_set))}"
    timeout = _mn_cli._poll_timeout_from_args(args)
    per_pod: list[dict] = []
    failures: list[dict] = []
    # Pod-side interpreter: sglang lives in /opt/venv on the canonical ROCm sglang-infera images; /usr/bin/python3
    # lacks sglang so _apply_on_pod's `import sglang` fails with "No module named 'sglang'".
    pod_python = os.environ.get("HYPERLOOM_MN_POD_PYTHON", "/opt/venv/bin/python")
    for target in gpu_targets:
        ip = target["podIP"]
        port = int(target.get("sshPort") or _mn_cli._infera_default_ssh_port(state))
        info(f"apply-tracelens-patch (infera): ssh -> {ip}:{port}")
        try:
            cp = _mn_cli._infera_ssh_run_script(
                state,
                ip,
                RemoteScript(script, pod_python, op_args),
                timeout=timeout,
                port=port,
            )
        except subprocess.TimeoutExpired:
            failures.append({"host": ip, "sshPort": port, "error": f"timeout after {timeout}s"})
            continue
        parsed = _mn_cli._extract_pod_json(cp.stdout) or {}
        pods = parsed.get("per_pod")
        if str(parsed.get("status")) in ("applied", "skipped") and pods:
            for r in pods:
                r["host"] = ip
                per_pod.append(r)
        else:
            failures.append(
                {
                    "host": ip,
                    "sshPort": port,
                    "error": parsed.get("error") or cp.stderr[-800:] or "unknown",
                    "rc": cp.returncode,
                }
            )
    overall = "applied" if not failures else "failed"
    if overall == "applied" and all(r.get("status") == "skipped" for r in per_pod):
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
    """Infera apply-patch: SSH fan-out kernel_node_ops.py apply to every GPU pod."""
    state = _infera_require_state()
    patch_path = Path(args.patch_file)
    if not patch_path.is_file():
        err(f"patch_file does not exist: {patch_path}")
        return _mn_cli.EXIT_CONFIG_ERROR
    patch_b64 = base64.b64encode(patch_path.read_bytes()).decode("ascii")
    gpu_targets = infera_support.gpu_ssh_targets_from_state(state)
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
        ip = target["podIP"]
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
    """Infera revert-patch: SSH each pod in the IP->backup_path map + restore."""
    state = _infera_require_state()
    try:
        records_by_host = json.loads(getattr(args, "records_json", "") or "{}")
        backup_map = json.loads(args.backup_map_json or "{}")
    except json.JSONDecodeError as exc:
        err(f"--backup-map-json not valid JSON: {exc}")
        return _mn_cli.EXIT_CONFIG_ERROR
    if not records_by_host and not backup_map:
        err("revert requires non-empty records or backup map")
        return _mn_cli.EXIT_CONFIG_ERROR
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
        if parsed and str(parsed.get("status")) == "restored":
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
        return _mn_cli.EXIT_CONFIG_ERROR
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


def _resolve_geak_src(explicit: str | None) -> str:
    """Resolve the shared-FS GEAK source dir the sandbox install.sh cloned."""
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
    """Install the GEAK CLI on every Infera GPU pod over SSH (idempotent)."""
    state = _infera_require_state()
    geak_src = _resolve_geak_src(getattr(args, "geak_src", None))
    if not geak_src:
        err("install-geak: cannot resolve GEAK source dir; pass --geak-src or set $HYPERLOOM_ROOT / $USER_DATA_PATH")
        return _mn_cli.EXIT_CONFIG_ERROR
    script = _mn_cli._read_pod_script("install_geak_node.sh")
    gpu_targets = infera_support.gpu_ssh_targets_from_state(state)
    info(f"install-geak (infera): geak_src={geak_src} pods={len(gpu_targets)}")
    results: list[dict] = []
    rc_total = 0
    for target in gpu_targets:
        ip = target["podIP"]
        port = int(target.get("sshPort") or _mn_cli._infera_default_ssh_port(state))
        info(f"install-geak: ssh -> {ip}:{port}")
        try:
            cp = _mn_cli._infera_ssh_run_script(
                state,
                ip,
                RemoteScript(script, "bash", shlex.quote(str(geak_src))),
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
