# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``hyperloom multi-node`` — single-entry sandbox CLI driving one handed-over cluster."""

from __future__ import annotations

import argparse
import base64
import json
import os
import tempfile
import traceback
from pathlib import Path
from typing import Any

import httpx

from hyperloom_common.env import EnvValueError

from ._internal.log import info, warn, err
from ._internal.server_args_safety import ServerArgsRejected, validate_server_args
from .cluster_access import (
    EXIT_CONFIG_ERROR,
    EXIT_INTERRUPT,
    EXIT_OK,
    EXIT_TRANSIENT,
    ConfigurationError,
    TransientFailure,
    DEFAULT_POLL_INTERVAL_S,
    _DEFAULT_POLL_TIMEOUT_S,
    _TERMINAL_OK_STATUSES,
    _load_state,
    _multinode_server_log_dir,
    _normalize_extra_args,
    _poll_timeout_from_args,
    _ray_dashboard_client,
    _require_state,
    _save_state,
    _short_poll,
    per_round_forward_overrides,
    resolve_poll_timeout_s,
)
from .pod_entrypoints import (
    KernelPatchApply,
    _build_kill_single_entrypoint,
    _build_multinode_apply_patch_entrypoint,
    _build_multinode_apply_tracelens_patch_entrypoint,
    _build_multinode_finalize_patch_entrypoint,
    _build_multinode_kill_entrypoint,
    _build_multinode_launch_entrypoint,
    _build_multinode_revert_patch_entrypoint,
    _build_multinode_router_entrypoint,
    _build_restart_entrypoint,
    _extract_launcher_summary,
    extract_pod_json,
    read_pod_script,
)
from .restart_pd_args import add_restart_pd_role_arguments, add_restart_pd_shape_arguments
from ._internal.env_safety import filter_forward_env
from ._internal.serving_probe import cluster_is_serving
from ._internal.topology_file import PD_ROUTER_BACKENDS


# Overall budget for the HTTP serving probe, spanning every request it makes: one /health per leg, /v1/models, then a
# /v1/completions that has to generate real tokens.
_RESUME_PROBE_TIMEOUT_ENV = "HYPERLOOM_MN_RESUME_PROBE_TIMEOUT_S"
_DEFAULT_RESUME_PROBE_TIMEOUT_S = 30


# Infera idle-pod backend helpers


def install_geak_on_pods_best_effort() -> int:
    """Install GEAK on the Infera GPU pods (provisioner hook); raised errors propagate, the return code is the caller's."""
    if _load_state().get("backend") != "infera":
        return 0
    ns = argparse.Namespace(
        geak_src=None,
        print_logs=False,
        poll_interval=DEFAULT_POLL_INTERVAL_S,
        poll_timeout=resolve_poll_timeout_s(),
    )
    return cmd_install_geak(ns)


# Subcommand `bootstrap`
def cmd_bootstrap(args: argparse.Namespace) -> int:
    """Run the BYOI bootstrap script inside the RayJob via Ray Dashboard REST."""
    # Only head_pod_ip: the Ray Dashboard client addresses the head pod directly, and a cluster the platform
    # handed over carries no rayjob_id.
    state = _require_state("head_pod_ip")

    if args.script:
        # Operator override: assume the path is pod-visible.
        entrypoint = f"bash {args.script}" + (" --force" if args.force else "")
    else:
        bootstrap_sh = read_pod_script("bootstrap.sh")
        force_arg = " --force" if args.force else ""
        entrypoint = (
            "set -euo pipefail; "
            'WORK_DIR=/tmp/multi_node_pod_scripts; mkdir -p "$WORK_DIR"; '
            "cat > \"$WORK_DIR/bootstrap.sh\" <<'__MN_BOOT_EOF__'\n"
            f"{bootstrap_sh}__MN_BOOT_EOF__\n"
            f'chmod +x "$WORK_DIR/bootstrap.sh"; '
            f'"$WORK_DIR/bootstrap.sh"{force_arg}'
        )

    with _ray_dashboard_client(state) as ray:
        info(f"submitting bootstrap entrypoint: {entrypoint}")
        sub_id = ray.submit_job(entrypoint)
        info(f"submission_id={sub_id}")

        def _fetch():
            """Fetch the bootstrap job and summarize its status."""
            j = ray.get_job(sub_id)
            return j, f"status={j.get('status', '?')}"

        result = _short_poll(
            label=f"bootstrap {sub_id}",
            fetch=_fetch,
            interval_s=args.poll_interval,
            timeout_s=_poll_timeout_from_args(args),
        )
        info(f"bootstrap done: {result.get('status')}")
        if args.print_logs:
            logs = ray.get_job_logs(sub_id)
            print(logs)
    return 0


# Subcommand `verify`
def cmd_verify(args: argparse.Namespace) -> int:
    """Sanity-check the toolchain bootstrap installed inside the RayJob."""
    state = _require_state("head_pod_ip")

    # Source the env file, then verify ``ray`` is on PATH.
    script = (
        "set -e; "
        "if [ -f /etc/profile.d/hyperloom-env.sh ]; "
        "then source /etc/profile.d/hyperloom-env.sh; fi; "
        "for bin in ray; do "
        '  echo "-- which $bin --"; '
        '  which "$bin" || { echo "MISSING: $bin" >&2; exit 1; }; '
        "done; "
        "echo OK"
    )
    entrypoint = script  # RayDashboardClient wraps as bash (-lc breaks PATH).
    with _ray_dashboard_client(state) as ray:
        info("submitting verify entrypoint")
        sub_id = ray.submit_job(entrypoint)
        info(f"submission_id={sub_id}")

        def _fetch():
            """Fetch the verify job and summarize its status."""
            j = ray.get_job(sub_id)
            return j, f"status={j.get('status', '?')}"

        result = _short_poll(
            label=f"verify {sub_id}",
            fetch=_fetch,
            interval_s=args.poll_interval,
            timeout_s=_poll_timeout_from_args(args),
        )
        if args.print_logs:
            logs = ray.get_job_logs(sub_id)
            print(logs)
        info(f"verify done: {result.get('status')}")
    return 0


# Subcommand `restart-server`
def _exec_kill_submission(
    state: dict[str, Any],
    entrypoint: str,
    *,
    label: str,
    args: argparse.Namespace,
) -> None:
    """Submit a kill entrypoint via Ray Dashboard and poll to SUCCEEDED."""
    with _ray_dashboard_client(state) as ray:
        kill_sub = ray.submit_job(entrypoint)
        info(f"{label} submission_id={kill_sub}")

        def _fetch_kill():
            """Fetch the kill job status for the poll loop."""
            j = ray.get_job(kill_sub)
            return j, f"kill status={j.get('status', '?')}"

        _short_poll(
            label=f"{label} {kill_sub}",
            fetch=_fetch_kill,
            interval_s=args.poll_interval,
            timeout_s=_poll_timeout_from_args(args),
        )
        if getattr(args, "print_logs", False):
            logs = ray.get_job_logs(kill_sub)
            print(logs)


def _forward_runtime_env(overrides: dict[str, Any]) -> dict[str, Any] | None:
    """Build the Ray runtime_env carrying per-round env overrides to every rank."""
    env_vars = filter_forward_env(dict(overrides.get("set") or {}))
    if not env_vars:
        return None
    info(f"forwarding per-round env to all ranks: {sorted(env_vars)}")
    return {"env_vars": env_vars}


def _submit_and_collect_pod_json(
    state: dict[str, Any],
    entrypoint: str,
    *,
    label: str,
    poll_interval: int,
    poll_timeout: int,
) -> tuple[int, dict | None, str]:
    """Submit ``entrypoint``, poll to terminal, parse the per-pod JSON, and return ``(returncode, parsed_or_None, logs)``."""
    with _ray_dashboard_client(state) as ray:
        sub_id = ray.submit_job(entrypoint)
        info(f"{label} submission_id={sub_id}")

        def _fetch():
            """Fetch the submission status for the poll loop."""
            j = ray.get_job(sub_id)
            return j, f"{label} status={j.get('status', '?')}"

        _short_poll(
            label=label,
            fetch=_fetch,
            interval_s=poll_interval,
            timeout_s=poll_timeout,
        )
        logs = ray.get_job_logs(sub_id)
        parsed = extract_pod_json(logs)
        # Dashboard SUCCEEDED; the sub-script's own ``status`` says if the fan-out worked.
        if parsed is None:
            return EXIT_TRANSIENT, None, logs
        sub_status = str(parsed.get("status", "")).lower()
        return (EXIT_OK if sub_status == "ok" else EXIT_TRANSIENT), parsed, logs


# Subcommand: apply-patch / revert-patch (multi-node only) Cohesive rayjob/infera clusters live in
# commands/{rayjob,infera}.py.
from .commands.infera import (
    _infera_restart_server as _infera_restart_server,
    _infera_kill_inference as _infera_kill_inference,
    _infera_apply_tracelens_patch as _infera_apply_tracelens_patch,
    _infera_apply_patch as _infera_apply_patch,
    _infera_finalize_patch as _infera_finalize_patch,
    _infera_revert_patch as _infera_revert_patch,
    cmd_install_geak as cmd_install_geak,
)


def cmd_apply_patch(args: argparse.Namespace) -> int:
    """Fan out a kernel patch to every pod (head + workers)."""
    if _load_state().get("backend") == "infera":
        return _infera_apply_patch(args)
    state = _load_state()
    head_ip = (state.get("head_pod_ip") or "").strip()
    if not head_ip:
        err("apply-patch requires head_pod_ip in state file; the platform hands it over via HYPERLOOM_MN_EXT_HEAD_IP")
        return EXIT_CONFIG_ERROR

    patch_path = Path(args.patch_file)
    if not patch_path.is_file():
        err(f"patch_file does not exist: {patch_path}")
        return EXIT_CONFIG_ERROR
    try:
        patch_bytes = patch_path.read_bytes()
    except OSError as exc:
        err(f"failed to read patch_file {patch_path}: {exc}")
        return EXIT_CONFIG_ERROR
    patch_b64 = base64.b64encode(patch_bytes).decode("ascii")

    info(
        f"apply-patch: target={args.target_path} kernel_id={args.kernel_id!r} "
        f"backup_dir={args.backup_dir} bytes={len(patch_bytes)}"
    )

    patch = KernelPatchApply(
        target_path=args.target_path,
        patch_b64=patch_b64,
        backup_dir=args.backup_dir,
        kernel_id=args.kernel_id,
        jit_build_dir=str(args.jit_build_dir or ""),
    )
    entrypoint = _build_multinode_apply_patch_entrypoint(patch, args.timeout_sec)
    rc, parsed, logs = _submit_and_collect_pod_json(
        state,
        entrypoint,
        label="apply-patch",
        poll_interval=args.poll_interval,
        poll_timeout=_poll_timeout_from_args(args),
    )
    if parsed is None:
        err("apply-patch: could not parse per-pod JSON from dashboard logs")
        if args.print_logs:
            print(logs)
        return EXIT_TRANSIENT
    print(json.dumps(parsed, indent=2, sort_keys=True))
    return rc


def cmd_revert_patch(args: argparse.Namespace) -> int:
    """Fan out a kernel patch revert across the pods that originally received it."""
    if _load_state().get("backend") == "infera":
        return _infera_revert_patch(args)
    state = _load_state()
    head_ip = (state.get("head_pod_ip") or "").strip()
    if not head_ip:
        err("revert-patch requires head_pod_ip in state file; the platform hands it over via HYPERLOOM_MN_EXT_HEAD_IP")
        return EXIT_CONFIG_ERROR

    try:
        decoded_records = json.loads(args.records_json or "{}")
        decoded_map = json.loads(args.backup_map_json or "{}")
    except json.JSONDecodeError as exc:
        err(f"revert records JSON is invalid: {exc}")
        return EXIT_CONFIG_ERROR
    if not decoded_records and not decoded_map:
        err("revert requires non-empty --records-json or --backup-map-json")
        return EXIT_CONFIG_ERROR

    hosts = list((decoded_records or decoded_map).keys())
    info(f"revert-patch: target={args.target_path} backup_hosts={hosts}")

    entrypoint = _build_multinode_revert_patch_entrypoint(
        args.target_path,
        args.backup_map_json,
        args.timeout_sec,
        args.records_json,
    )
    rc, parsed, logs = _submit_and_collect_pod_json(
        state,
        entrypoint,
        label="revert-patch",
        poll_interval=args.poll_interval,
        poll_timeout=_poll_timeout_from_args(args),
    )
    if parsed is None:
        err("revert-patch: could not parse per-pod JSON from dashboard logs")
        if args.print_logs:
            print(logs)
        return EXIT_TRANSIENT
    print(json.dumps(parsed, indent=2, sort_keys=True))
    return rc


def cmd_finalize_patch(args: argparse.Namespace) -> int:
    """Delete backups after a patch becomes the accepted baseline."""
    if _load_state().get("backend") == "infera":
        return _infera_finalize_patch(args)
    state = _load_state()
    if not (state.get("head_pod_ip") or "").strip():
        return EXIT_CONFIG_ERROR
    try:
        records = json.loads(args.records_json or "{}")
    except json.JSONDecodeError:
        return EXIT_CONFIG_ERROR
    if not records:
        return EXIT_CONFIG_ERROR
    entrypoint = _build_multinode_finalize_patch_entrypoint(
        args.records_json,
        args.timeout_sec,
    )
    rc, parsed, logs = _submit_and_collect_pod_json(
        state,
        entrypoint,
        label="finalize-patch",
        poll_interval=args.poll_interval,
        poll_timeout=_poll_timeout_from_args(args),
    )
    if parsed is None:
        err("finalize-patch: could not parse per-pod JSON from dashboard logs")
        if args.print_logs:
            print(logs)
        return EXIT_TRANSIENT
    print(json.dumps(parsed, indent=2, sort_keys=True))
    return rc


def cmd_apply_tracelens_patch(args: argparse.Namespace) -> int:
    """Fan out the TraceLens SGLang patch set to every pod via apply_tracelens_patch_multinode.py; multi-node only."""
    backend = _load_state().get("backend")
    if backend == "infera":
        return _infera_apply_tracelens_patch(args)
    if backend in PD_ROUTER_BACKENDS:
        from .commands.pd_cluster import apply_tracelens_patch

        return apply_tracelens_patch(args)
    state = _load_state()
    head_ip = (state.get("head_pod_ip") or "").strip()
    if not head_ip:
        err(
            "apply-tracelens-patch requires head_pod_ip in state file; the platform hands it over via HYPERLOOM_MN_EXT_HEAD_IP"
        )
        return EXIT_CONFIG_ERROR

    tracelens_root = args.tracelens_root or os.environ.get("TRACELENS_ROOT", "").strip()
    if not tracelens_root:
        err(
            "apply-tracelens-patch requires --tracelens-root or "
            "$TRACELENS_ROOT to point at the TraceLens checkout "
            "(must be visible from every pod, typically a wekafs path)"
        )
        return EXIT_CONFIG_ERROR

    patch_set = getattr(args, "patch_set", None) or ""
    info(
        f"apply-tracelens-patch: tracelens_root={tracelens_root!r} version_pin={args.sglang_version_pin!r} "
        f"patch_set={patch_set or 'pod-gated'!r}"
    )

    entrypoint = _build_multinode_apply_tracelens_patch_entrypoint(
        tracelens_root,
        args.sglang_version_pin or "",
        patch_set,
    )
    rc, parsed, logs = _submit_and_collect_pod_json(
        state,
        entrypoint,
        label="apply-tracelens-patch",
        poll_interval=args.poll_interval,
        poll_timeout=_poll_timeout_from_args(args),
    )
    if parsed is None:
        err("apply-tracelens-patch: could not parse per-pod JSON from dashboard logs")
        if args.print_logs:
            print(logs)
        return EXIT_TRANSIENT
    print(json.dumps(parsed, indent=2, sort_keys=True))
    # This patcher uses ``applied``/``skipped`` status, so derive the exit code.
    if parsed.get("status") in ("applied", "skipped"):
        return EXIT_OK
    return EXIT_TRANSIENT


def _resume_probe_timeout_s() -> int:
    """Overall budget for the HTTP serving probe, across all of its requests."""
    try:
        return max(1, int(os.environ.get(_RESUME_PROBE_TIMEOUT_ENV, "") or _DEFAULT_RESUME_PROBE_TIMEOUT_S))
    except ValueError:
        return _DEFAULT_RESUME_PROBE_TIMEOUT_S


def _rayjob_topology_fingerprint(
    args: argparse.Namespace,
    nnodes: int,
    forward_env: dict[str, str],
) -> dict[str, Any]:
    """Every field that changes what the RayJob launcher spawns, as one record.

    ``forward_env`` is the runtime_env the launch submission carries, so the servers run
    with it: a round that changes only these would otherwise resume the prior cluster and
    benchmark the previous environment while reporting the new one.
    """
    pd_mode = (getattr(args, "pd_mode", "") or "aggregated").lower()
    fingerprint: dict[str, Any] = {
        "framework": str(args.framework),
        "model": str(args.model),
        "tp": int(args.tp),
        "ep": int(getattr(args, "ep", 1) or 1),
        "nnodes": int(nnodes),
        "pd_mode": pd_mode,
        "extra_args": _normalize_extra_args(getattr(args, "extra_args", "")),
        "forward_env": dict(sorted(forward_env.items())),
    }
    if pd_mode == "disaggregated":
        # Only meaningful under PD; leaving them out when aggregated keeps a stale value from an earlier PD run out of
        # the comparison.
        fingerprint.update(
            {
                "pd_prefill_nodes": int(getattr(args, "pd_prefill_nodes", 0) or 0),
                "pd_decode_nodes": int(getattr(args, "pd_decode_nodes", 0) or 0),
                "pd_prefill_tp": int(getattr(args, "pd_prefill_tp", 0) or 0),
                "pd_decode_tp": int(getattr(args, "pd_decode_tp", 0) or 0),
                "pd_transfer_backend": (getattr(args, "pd_transfer_backend", "") or "").strip(),
                "pd_ib_device": (getattr(args, "pd_ib_device", "") or "").strip(),
                "pd_bootstrap_port": int(getattr(args, "pd_bootstrap_port", 0) or 0),
            }
        )
    return fingerprint


def _reject_unsupported_restart_args(args: argparse.Namespace) -> int:
    """The exit code for a restart whose arguments this backend cannot honour, else 0."""
    try:
        validate_server_args(getattr(args, "extra_args", "") or "", context="restart-server --extra-args")
    except ServerArgsRejected as exc:
        err(str(exc))
        return EXIT_CONFIG_ERROR
    if (getattr(args, "pd_mode", "") or "").lower() != "disaggregated":
        return 0
    # launch_multinode.py has no per-role EP or server args; only the infera launcher does.
    per_role = [
        flag
        for flag, value in (
            ("--pd-prefill-ep", getattr(args, "pd_prefill_ep", 0)),
            ("--pd-decode-ep", getattr(args, "pd_decode_ep", 0)),
            ("--pd-prefill-extra-args", getattr(args, "pd_prefill_extra_args", "")),
            ("--pd-decode-extra-args", getattr(args, "pd_decode_extra_args", "")),
        )
        if value
    ]
    if per_role:
        err(f"restart-server: {', '.join(per_role)} are supported on the infera backend only, not rayjob")
        return EXIT_CONFIG_ERROR
    return 0


def _resume_prior_launch(args: argparse.Namespace, state: dict, *, topology: dict[str, Any]) -> tuple[str, bool]:
    """Reuse a prior launch when the whole topology record matches and it is still good.

    Returns the submission id to keep polling, and whether the resume was granted
    because the cluster is already serving -- which in PD means its router is up
    and must not be rebuilt. ``("", False)`` asks for a full KILL+LAUNCH.
    """
    resume_enabled = os.environ.get("MULTI_NODE_RESTART_RESUME_RUNNING", "1").lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    prev_sub = str(state.get("last_restart_submission_id") or "").strip()
    if not resume_enabled or not prev_sub or state.get("last_restart_topology") != topology:
        return "", False
    prev_status = ""
    try:
        with _ray_dashboard_client(state) as probe:
            prev_status = str(probe.get_job(prev_sub).get("status", "")).upper()
    except (httpx.HTTPError, httpx.InvalidURL, OSError, RuntimeError, ValueError, AttributeError) as exc:
        info(f"resume probe failed: {exc!r}; falling back to KILL+LAUNCH")
    if prev_status == "RUNNING":
        info(
            f"resume: reusing launch_sub={prev_sub} "
            f"(framework={args.framework} model={args.model} "
            f"tp={args.tp} ep={getattr(args, 'ep', 1)}) "
            f"— skipping KILL+LAUNCH, just polling"
        )
        return prev_sub, False
    if prev_status not in _TERMINAL_OK_STATUSES or getattr(args, "no_wait_health", False):
        return "", False
    # The driver waits for the servers it spawned before exiting, so a terminal-OK status means this
    # cluster served once.
    if cluster_is_serving(state, pd_mode=topology["pd_mode"], timeout_s=_resume_probe_timeout_s()):
        info(
            f"resume: prior launch_sub={prev_sub} SUCCEEDED and every leg serves generated tokens; skipping KILL+LAUNCH"
        )
        return prev_sub, True
    info(f"resume: prior launch_sub={prev_sub} SUCCEEDED but the cluster no longer serves; doing a full KILL+LAUNCH")
    return "", False


def _start_pd_router(
    ray,
    args: argparse.Namespace,
    *,
    launch_sub: str,
    pid_dir: str,
    log_dir: str,
) -> tuple[str, dict, int]:
    """Submit the router that binds 8888 in front of the two PD legs.

    Returns ``(submission_id, leg_urls, exit_code)``; a non-zero exit code is
    fatal -- a PD cluster with no router has no public port.
    """
    router_state = _extract_launcher_summary(ray.get_job_logs(launch_sub))
    prefill_url = str(router_state.get("pd_prefill_url") or "").strip()
    decode_url = str(router_state.get("pd_decode_url") or "").strip()
    if not prefill_url or not decode_url:
        info("ERROR PD launcher summary missing prefill/decode URL; cannot start router. Inspect launch logs above.")
        return "", router_state, 1

    info(f"PD launcher summary: prefill={prefill_url} decode={decode_url}; submitting router entrypoint")
    router_ep = _build_multinode_router_entrypoint(args, prefill_url, decode_url, pid_dir, log_dir)
    router_sub = ray.submit_job(router_ep)
    info(f"router submission_id={router_sub} (detaches launch_router.py; dashboard exits when router is alive)")

    def _fetch_router():
        """Fetch the router job status for the poll loop."""
        job = ray.get_job(router_sub)
        return job, f"router status={job.get('status', '?')}"

    _short_poll(
        label=f"router {router_sub}",
        fetch=_fetch_router,
        interval_s=args.poll_interval,
        timeout_s=_poll_timeout_from_args(args),
    )
    if args.print_logs:
        print(ray.get_job_logs(router_sub))
    return router_sub, router_state, 0


def _persist_multinode_restart(state: dict, args: argparse.Namespace, *, router_sub: str, router_state: dict) -> None:
    """Complete the early checkpoint once the launch is up, so a later invocation can resume or fall back to it."""
    # Persist PD state so later invocations can fall back when a flag is omitted.
    if state["last_restart_pd_mode"] == "disaggregated":
        state["last_restart_pd_prefill_nodes"] = int(getattr(args, "pd_prefill_nodes", 0) or 0)
        state["last_restart_pd_decode_nodes"] = int(getattr(args, "pd_decode_nodes", 0) or 0)
        state["last_restart_pd_prefill_tp"] = int(getattr(args, "pd_prefill_tp", 0) or 0)
        state["last_restart_pd_decode_tp"] = int(getattr(args, "pd_decode_tp", 0) or 0)
        state["last_restart_pd_transfer_backend"] = getattr(args, "pd_transfer_backend", "") or ""
        state["last_restart_pd_ib_device"] = getattr(args, "pd_ib_device", "") or ""
        state["last_router_submission_id"] = router_sub
        state["pd_prefill_url"] = router_state.get("pd_prefill_url", "")
        state["pd_decode_url"] = router_state.get("pd_decode_url", "")
    _save_state(state)


def _restart_multinode(args: argparse.Namespace, state: dict, nnodes: int) -> int:
    """Restart the server group across every node, with a dir-based PID/log layout (one file per rank)."""
    pid_dir = args.pid_file or state.get("last_server_pid_dir") or str(Path(tempfile.gettempdir()) / "multi_node_pids")
    log_dir = args.log_file or state.get("last_server_log_dir") or _multinode_server_log_dir()
    info(f"restart-server (multi-node): framework={args.framework} model={args.model} tp={args.tp} nnodes={nnodes}")

    kill_ep = _build_multinode_kill_entrypoint(pid_dir)
    launch_ep = _build_multinode_launch_entrypoint(args, nnodes, pid_dir, log_dir)

    # Collected once: the same runtime_env decides both whether the prior launch can be resumed and what the
    # launch below actually sends. extra_args is normalized on both sides so whitespace alone does not miss the
    # fast path, and it is part of the record because it carries every variant flag.
    runtime_env = _forward_runtime_env(per_round_forward_overrides())
    topology = _rayjob_topology_fingerprint(args, nnodes, dict((runtime_env or {}).get("env_vars") or {}))
    launch_sub, resumed_serving = _resume_prior_launch(args, state, topology=topology)

    if not launch_sub:
        _exec_kill_submission(state, kill_ep, label="restart kill", args=args)

    with _ray_dashboard_client(state) as ray:
        if not launch_sub:
            launch_sub = ray.submit_job(launch_ep, runtime_env=runtime_env)
            info(f"launch submission_id={launch_sub} (driver waits for actors, then returns; servers detached)")

        # Early checkpoint: persist the launch identity + config before the (potentially long) _short_poll, so a
        # poll-timeout retry can hit the resume fast path instead of restarting the bootstrap from zero.
        state["last_server_pid_dir"] = pid_dir
        state["last_server_log_dir"] = log_dir
        state["last_restart_submission_id"] = launch_sub
        state["last_restart_framework"] = args.framework
        state["last_restart_model"] = args.model
        state["last_restart_tp"] = topology["tp"]
        state["last_restart_ep"] = topology["ep"]
        state["last_restart_pd_mode"] = topology["pd_mode"]
        state["last_restart_extra_args"] = topology["extra_args"]
        # A poll-timeout retry resumes from this checkpoint, so it carries the topology record as well; the
        # per-field pd_* keys below are written only once the poll returns.
        state["last_restart_topology"] = topology
        _save_state(state)

        def _fetch_launch():
            """Fetch the launch job status for the poll loop."""
            job = ray.get_job(launch_sub)
            return job, f"launch status={job.get('status', '?')}"

        _short_poll(
            label=f"launch {launch_sub}",
            fetch=_fetch_launch,
            interval_s=args.poll_interval,
            timeout_s=_poll_timeout_from_args(args),
        )
        if args.print_logs:
            print(ray.get_job_logs(launch_sub))

        router_sub = ""
        router_state: dict = {}
        if topology["pd_mode"] == "disaggregated" and resumed_serving:
            # The serving probe just drove a completion through this cluster's public port, which in PD is the
            # router: it is up and routing.
            router_sub = str(state.get("last_router_submission_id") or "")
            router_state = {
                "pd_prefill_url": str(state.get("pd_prefill_url") or ""),
                "pd_decode_url": str(state.get("pd_decode_url") or ""),
            }
            info("resume: keeping the router that is already serving")
        elif topology["pd_mode"] == "disaggregated":
            router_sub, router_state, router_rc = _start_pd_router(
                ray, args, launch_sub=launch_sub, pid_dir=pid_dir, log_dir=log_dir
            )
            if router_rc:
                return router_rc

    _persist_multinode_restart(state, args, router_sub=router_sub, router_state=router_state)
    info("multi-node servers launched; benchmarks should target $service_url")
    return 0


def _restart_single_node(args: argparse.Namespace, state: dict) -> int:
    """Restart the one server this pod runs."""
    if (getattr(args, "pd_mode", "") or "aggregated").lower() == "disaggregated":
        # PD is meaningless on one pod; fail loudly.
        info(
            "ERROR --pd-mode disaggregated is not supported in single-node "
            "mode (state.json says nodes=1). PD requires >=2 nodes so "
            "prefill and decode can run on disjoint pods. Re-run optimize with "
            "--nodes >=2 so the platform provisions a multi-node cluster before "
            "asking for PD."
        )
        return 2
    temp_root = tempfile.gettempdir()
    pid_file = args.pid_file or state.get("last_server_pid_file") or str(Path(temp_root) / "multi_node_server.pid")
    log_file = args.log_file or str(Path(temp_root) / "multi_node_server.log")
    entrypoint = _build_restart_entrypoint(args, pid_file, log_file)

    with _ray_dashboard_client(state) as ray:
        info(f"restart-server (single-node): framework={args.framework} model={args.model} tp={args.tp}")
        sub_id = ray.submit_job(entrypoint)
        info(f"submission_id={sub_id} (entrypoint will exit after launch; server keeps running via nohup)")

        def _fetch():
            """Fetch the restart job status for the poll loop."""
            job = ray.get_job(sub_id)
            return job, f"status={job.get('status', '?')}"

        _short_poll(
            label=f"restart {sub_id}",
            fetch=_fetch,
            interval_s=args.poll_interval,
            timeout_s=_poll_timeout_from_args(args),
        )
        if args.print_logs:
            print(ray.get_job_logs(sub_id))

    state["last_server_pid_file"] = pid_file
    state["last_restart_submission_id"] = sub_id
    state["last_restart_framework"] = args.framework
    state["last_restart_model"] = args.model
    state["last_restart_tp"] = args.tp
    _save_state(state)
    info("server launch acknowledged; subsequent benchmark jobs can target $service_url")
    return 0


def cmd_restart_server(args: argparse.Namespace) -> int:
    """Kill any prior vllm/sglang server and launch a new one."""
    backend = _load_state().get("backend")
    if backend == "infera":
        return _infera_restart_server(args)
    if backend in PD_ROUTER_BACKENDS:
        from .commands.pd_cluster import restart_pd_cluster

        return restart_pd_cluster(args)
    rejected = _reject_unsupported_restart_args(args)
    if rejected:
        return rejected
    state = _require_state("head_pod_ip")
    nnodes = int(state.get("nodes") or 1)
    if nnodes >= 2:
        return _restart_multinode(args, state, nnodes)
    return _restart_single_node(args, state)


def _record_kill_and_invalidate_launch(state: dict) -> None:
    """Drop the launch identity a kill terminated."""
    state.pop("last_restart_submission_id", None)
    _save_state(state)


def cmd_kill_inference(args: argparse.Namespace) -> int:
    """Kill vllm/sglang on the RayJob without starting replacements."""
    backend = _load_state().get("backend")
    if backend == "infera":
        return _infera_kill_inference(args)
    if backend in PD_ROUTER_BACKENDS:
        from .commands.pd_cluster import kill_pd_cluster

        return kill_pd_cluster(args)
    state = _require_state("head_pod_ip")
    nnodes = int(state.get("nodes") or 1)

    if nnodes >= 2:
        pid_dir = (
            args.pid_file or state.get("last_server_pid_dir") or str(Path(tempfile.gettempdir()) / "multi_node_pids")
        )
        info(f"kill-inference (multi-node): pid_dir={pid_dir}")
        _exec_kill_submission(state, _build_multinode_kill_entrypoint(pid_dir), label="kill-inference", args=args)
        _record_kill_and_invalidate_launch(state)
        return 0

    pid_file = (
        args.pid_file or state.get("last_server_pid_file") or str(Path(tempfile.gettempdir()) / "multi_node_server.pid")
    )
    info(f"kill-inference (single-node): pid_file={pid_file}")
    _exec_kill_submission(state, _build_kill_single_entrypoint(pid_file), label="kill-inference", args=args)
    _record_kill_and_invalidate_launch(state)
    return 0


def kill_inference_for_kernel_agent_best_effort() -> None:
    """Best-effort inference teardown before kernel-agent Ray GPU tasks; command errors are warned about."""
    ns = argparse.Namespace(
        pid_file=None,
        print_logs=False,
        poll_interval=DEFAULT_POLL_INTERVAL_S,
        poll_timeout=resolve_poll_timeout_s(),
    )
    try:
        cmd_kill_inference(ns)
    except (RuntimeError, OSError) as exc:
        warn(f"kill-inference skipped: {type(exc).__name__}: {exc}")


def _add_common_poll_flags(p: argparse.ArgumentParser) -> None:
    """Register the shared ``--poll-interval`` / ``--poll-timeout`` flags."""
    p.add_argument(
        "--poll-interval",
        type=int,
        default=DEFAULT_POLL_INTERVAL_S,
        help=f"seconds between polls (default {DEFAULT_POLL_INTERVAL_S})",
    )
    p.add_argument(
        "--poll-timeout",
        type=int,
        default=None,
        help=(
            "max seconds before this CLI invocation gives up "
            f"(default: HYPERLOOM_MN_POLL_TIMEOUT_S env or {_DEFAULT_POLL_TIMEOUT_S}; "
            f"use 1800 for MoE JIT cold-start). Re-run the same subcommand to "
            "keep polling."
        ),
    )


def _add_restart_server_parser(sub: argparse._SubParsersAction) -> None:
    """Register the ``restart-server`` subcommand."""
    # restart-server
    sp = sub.add_parser("restart-server", help="kill prior vllm/sglang server and start a new one")
    sp.add_argument("--framework", required=True, choices=("sglang", "vllm"))
    sp.add_argument("--model", required=True, help="model path or HF id")
    sp.add_argument("--tp", type=int, required=True)
    sp.add_argument(
        "--ep",
        type=int,
        default=1,
        help="expert-parallel size for MoE inference. 1 (default) keeps "
        "experts sharded by TP. >=2 enables EP: sglang gets "
        "`--enable-ep-moe --ep-size N`, vllm gets "
        "`--enable-expert-parallel`. EP > TP is rejected by the "
        "orchestrator helper before this CLI is invoked.",
    )
    add_restart_pd_shape_arguments(sp)
    add_restart_pd_role_arguments(sp)
    sp.add_argument("--extra-args", default="", help="extra CLI args appended verbatim to the framework launch command")
    sp.add_argument(
        "--pid-file", default=None, help="PID file path inside the head pod; defaults to /tmp/multi_node_server.pid"
    )
    sp.add_argument(
        "--log-file", default=None, help="server log path inside the head pod; defaults to /tmp/multi_node_server.log"
    )
    sp.add_argument(
        "--no-wait-health",
        action="store_true",
        help="exit the dashboard job immediately after launch instead of waiting for /health",
    )
    sp.add_argument("--print-logs", action="store_true")
    _add_common_poll_flags(sp)
    sp.set_defaults(func=cmd_restart_server)


def _add_kernel_patch_parsers(sub: argparse._SubParsersAction) -> None:
    """Register the ``apply-patch``, ``revert-patch`` and ``finalize-patch`` subcommands."""
    # apply-patch (multi-node only)
    sp = sub.add_parser(
        "apply-patch",
        help="fan-out a kernel patch to every pod (head + workers); multi-node only",
    )
    sp.add_argument(
        "--patch-file",
        required=True,
        help="path to the patch source on sandbox filesystem; contents will be base64-encoded into the dashboard entrypoint",
    )
    sp.add_argument(
        "--target-path",
        required=True,
        help="absolute file path on each pod to overwrite (e.g. /sgl-workspace/aiter/aiter/ops/gemm.py)",
    )
    sp.add_argument(
        "--backup-dir",
        required=True,
        help="directory on each pod where the pre-patch original is saved (e.g. /var/kernel_patch_backups)",
    )
    sp.add_argument("--kernel-id", default="", help="optional id used to construct backup filename")
    sp.add_argument("--jit-build-dir", default="")
    sp.add_argument("--timeout-sec", type=int, default=120, help="per-actor timeout (default 120s)")
    sp.add_argument("--print-logs", action="store_true", help="dump full dashboard job_logs on parse failure")
    _add_common_poll_flags(sp)
    sp.set_defaults(func=cmd_apply_patch)

    # revert-patch (multi-node only)
    sp = sub.add_parser(
        "revert-patch",
        help="fan-out a kernel patch revert; multi-node only",
    )
    sp.add_argument("--target-path", default="")
    sp.add_argument("--records-json", default="")
    sp.add_argument(
        "--backup-map-json",
        default="",
        help="JSON object {hostname: backup_path} from the matching apply-patch result",
    )
    sp.add_argument("--timeout-sec", type=int, default=60)
    sp.add_argument("--print-logs", action="store_true")
    _add_common_poll_flags(sp)
    sp.set_defaults(func=cmd_revert_patch)

    sp = sub.add_parser(
        "finalize-patch",
        help="delete backups for a patch accepted as the new baseline",
    )
    sp.add_argument("--records-json", required=True)
    sp.add_argument("--timeout-sec", type=int, default=60)
    sp.add_argument("--print-logs", action="store_true", help="dump full dashboard job_logs on parse failure")
    _add_common_poll_flags(sp)
    sp.set_defaults(func=cmd_finalize_patch)


def _add_apply_tracelens_patch_parser(sub: argparse._SubParsersAction) -> None:
    """Register the ``apply-tracelens-patch`` subcommand."""
    # apply-tracelens-patch (multi-node only)
    sp = sub.add_parser(
        "apply-tracelens-patch",
        help=(
            "fan-out the TraceLens SGLang patch set to every pod; "
            "multi-node only. Idempotent: skips pods already patched."
        ),
    )
    sp.add_argument(
        "--tracelens-root",
        default=None,
        help=(
            "absolute path to public TraceLens checkout (must be visible "
            "from every pod, typically /shared/...). Defaults to "
            "$TRACELENS_ROOT."
        ),
    )
    sp.add_argument(
        "--sglang-version-pin",
        default=None,
        help=("advisory pin (e.g. '0.5.11'); logged on mismatch with the sglang installed in the pod. Optional."),
    )
    sp.add_argument(
        "--patch-set",
        choices=("roofline", "graph-capture"),
        default=None,
        help=(
            "patch set every pod applies (the controller's resolved SGLang shape mode). "
            "Optional; when omitted each pod gates on its own SGLang version."
        ),
    )
    sp.add_argument("--print-logs", action="store_true", help="dump full dashboard job_logs on parse failure")
    _add_common_poll_flags(sp)
    sp.set_defaults(func=cmd_apply_tracelens_patch)


def _add_install_geak_parser(sub: argparse._SubParsersAction) -> None:
    """Register the ``install-geak`` subcommand."""
    # install-geak (infera only)
    sp = sub.add_parser(
        "install-geak",
        help="install the GEAK CLI on every Infera GPU pod over SSH "
        "(idempotent; pip-installs the shared-FS GEAK checkout); infera only",
    )
    sp.add_argument(
        "--geak-src",
        default=None,
        help="GEAK source dir on the shared mount (default: "
        "$HYPERLOOM_GEAK_SRC > $HYPERLOOM_ROOT/geak > "
        "$USER_DATA_PATH/runtime/geak)",
    )
    sp.add_argument("--print-logs", action="store_true")
    _add_common_poll_flags(sp)
    sp.set_defaults(func=cmd_install_geak)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argparse parser with every subcommand."""
    p = argparse.ArgumentParser(
        prog="hyperloom multi-node",
        description=(
            "Drive the multi-node cluster the platform provisioned and handed over "
            "via HYPERLOOM_MN_EXT_*. State persists in $MULTI_NODE_STATE_FILE."
        ),
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # bootstrap
    sp = sub.add_parser(
        "bootstrap",
        help="install Hyperloom RayJob toolchain inside the RayJob via Ray Dashboard REST. "
        "Default: uses multi_node/scripts/bootstrap.sh from this checkout.",
    )
    sp.add_argument(
        "--script",
        default=None,
        help="optional override: absolute path to a bootstrap.sh "
        "ALREADY VISIBLE inside the RayJob pod (e.g. baked into "
        "the image). When omitted, the bundled script is streamed in.",
    )
    sp.add_argument("--force", action="store_true", help="re-run bootstrap even if the marker file says it's done")
    sp.add_argument("--print-logs", action="store_true")
    _add_common_poll_flags(sp)
    sp.set_defaults(func=cmd_bootstrap)

    # verify
    sp = sub.add_parser("verify", help="confirm ray is on PATH inside the RayJob")
    sp.add_argument("--print-logs", action="store_true")
    _add_common_poll_flags(sp)
    sp.set_defaults(func=cmd_verify)

    _add_restart_server_parser(sub)

    sp = sub.add_parser(
        "kill-inference",
        help="kill vllm/sglang on the RayJob only (no new server; frees GPU for kernel-agent)",
    )
    sp.add_argument(
        "--pid-file",
        default=None,
        help="single-node: head-pod PID file path; multi-node: pid-dir path "
        "(same override semantics as restart-server --pid-file)",
    )
    sp.add_argument("--print-logs", action="store_true")
    _add_common_poll_flags(sp)
    sp.set_defaults(func=cmd_kill_inference)

    _add_kernel_patch_parsers(sub)

    _add_apply_tracelens_patch_parser(sub)

    _add_install_geak_parser(sub)

    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint with stable exit codes: 0 ok, 1 transient (retryable, including a terminal Ray job failure), 2 usage error, 3 config error, 130 SIGINT."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        err("interrupted")
        return EXIT_INTERRUPT
    except TransientFailure as tf:
        err(str(tf))
        return EXIT_TRANSIENT
    except (ConfigurationError, EnvValueError) as cfg:
        # Retrying cannot fix the environment, so never report it as transient.
        err(f"{type(cfg).__name__}: {cfg}")
        return EXIT_CONFIG_ERROR
    except (RuntimeError, ValueError) as exc:
        # Caller-fixable input/state errors -> config error; else transient.
        msg = str(exc)
        if any(s in msg for s in ("missing required keys", "unsupported framework")):
            err(f"{type(exc).__name__}: {exc}")
            return EXIT_CONFIG_ERROR
        err(f"{type(exc).__name__}: {exc}")
        return EXIT_TRANSIENT
    except Exception as exc:  # noqa: BLE001 - outermost CLI main: reported as transient with its traceback
        err(traceback.format_exc().rstrip())
        err(f"{type(exc).__name__}: {exc}")
        return EXIT_TRANSIENT
