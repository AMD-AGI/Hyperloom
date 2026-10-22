# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Multi-node helpers for the ``optimize`` CLI: cluster adoption, backend resolution, kept-kernel-patch restore."""

from __future__ import annotations

import argparse
import logging
import os
import sys

from ..session.paths import (
    session_dir as _session_dir_resolve,
)

log = logging.getLogger(__name__)


def _resolve_mn_backend(args: argparse.Namespace) -> str:
    """Multi-node backend selector: --mn-backend > $INFERENCE_OPTIMIZER_MN_BACKEND > rayjob."""
    backend = (
        (getattr(args, "mn_backend", None) or "").strip()
        or os.environ.get("INFERENCE_OPTIMIZER_MN_BACKEND", "").strip()
        or os.environ.get("MN_BACKEND", "").strip()
        or "rayjob"
    ).lower()
    if backend not in ("rayjob", "infera"):
        print(
            f"ERROR: --mn-backend must be 'rayjob' or 'infera', got {backend!r}",
            file=sys.stderr,
        )
        sys.exit(2)
    return backend


def _prepare_multi_node_state(args: argparse.Namespace) -> None:
    """Adopt the platform-provisioned cluster for a ``--nodes >= 2`` run."""
    nodes = max(1, int(args.nodes))
    if nodes < 2:
        return

    from ..multi_node._internal.external_state import (
        build_external_state_from_env,
        external_has_server_control,
        external_has_ssh_control,
        external_service_url,
    )
    from ..multi_node.cli import _save_state
    from hyperloom.orchestrator.actions.executors._multi_node_env import export_ray_address_to_os

    if not external_service_url():
        print(
            "ERROR: --nodes >= 2 requires a cluster provisioned by the platform. "
            "Set HYPERLOOM_MN_EXT_SERVICE_URL (plus HYPERLOOM_MN_EXT_SSH_KEY and "
            "one of _PREFILL_IPS / _DECODE_IPS / _WORKER_IPS for infera, or "
            "HYPERLOOM_MN_EXT_HEAD_IP for rayjob); see multi_node/SKILL.md "
            '"Cluster hand-off".',
            file=sys.stderr,
        )
        sys.exit(2)

    ext_state = build_external_state_from_env()
    ext_state["backend"] = _resolve_mn_backend(args)
    if ext_state["backend"] == "infera" and not external_has_ssh_control():
        print(
            "ERROR: the infera backend requires SSH control. Set "
            "HYPERLOOM_MN_EXT_SSH_KEY plus HYPERLOOM_MN_EXT_PREFILL_IPS/"
            "DECODE_IPS (or WORKER_IPS). rayjob uses Ray, not SSH.",
            file=sys.stderr,
        )
        sys.exit(2)
    # The mirror image. --mn-backend defaults to rayjob, so an infera-shaped hand-off whose operator forgot the flag
    # arrives here as rayjob and would report server_control=yes -- SSH control is real, it is simply not the control
    # this backend uses.
    if ext_state["backend"] == "rayjob" and not ext_state.get("head_pod_ip"):
        if external_has_ssh_control():
            print(
                "ERROR: --mn-backend rayjob (the default) needs "
                "HYPERLOOM_MN_EXT_HEAD_IP, but this hand-off carries SSH "
                "credentials and GPU pod IPs instead -- it is an infera "
                "cluster. Pass --mn-backend infera.",
                file=sys.stderr,
            )
            sys.exit(2)
        print(
            "multi-node: rayjob with no HYPERLOOM_MN_EXT_HEAD_IP -- this cluster cannot be "
            "restarted, so the run is benchmark-only (see the warning below).",
            file=sys.stderr,
        )
    try:
        _save_state(ext_state)
    except OSError as exc:
        print(f"ERROR: cannot write multi-node state: {exc}", file=sys.stderr)
        sys.exit(2)
    os.environ["BENCHMARK_BASE_URL"] = ext_state["service_url"]
    os.environ["MAGPIE_RUN_PHASE"] = "client"
    export_ray_address_to_os()
    # Report server *control*, not SSH specifically: rayjob restarts through the Ray dashboard with no SSH at all, so
    # keying this on SSH labelled a fully controllable rayjob cluster "benchmark-only" and gave the genuinely
    # uncontrollable one the same words -- exactly backwards on the one line an operator reads to find out whether the
    # run can tune anything.
    has_control = external_has_server_control()
    print(
        "multi-node: adopted the platform-provisioned cluster. "
        f"url={ext_state['service_url']} prefill={ext_state['prefill_pod_ips']} "
        f"decode={ext_state['decode_pod_ips']} worker={ext_state['worker_pod_ips']} "
        f"server_control={'yes' if has_control else 'no (benchmark-only)'} "
        f"ssh_control={'yes' if external_has_ssh_control() else 'no'}"
    )
    if not has_control:
        # Loud, not fatal: benchmark-only is a documented mode (multi_node/SKILL.md), but without a restart every
        # candidate re-measures the one unchanged server, so the run still reports gains that no config produced.
        print(
            "WARNING: no server control -- per-round restarts will be skipped, so every "
            "candidate config measures the SAME unchanged server and the reported gains "
            "are meaningless. Set HYPERLOOM_MN_EXT_HEAD_IP (rayjob) or "
            "HYPERLOOM_MN_EXT_SSH_KEY plus one of _PREFILL_IPS / _DECODE_IPS / _WORKER_IPS "
            "(infera) to tune. Continuing in benchmark-only mode.",
            file=sys.stderr,
        )
    _prepare_adopted_cluster(
        args,
        str(ext_state["backend"]),
        head_ip=str(ext_state.get("head_pod_ip") or ""),
    )


def _prepare_adopted_cluster(args: argparse.Namespace, backend: str, *, head_ip: str) -> None:
    """Make an adopted cluster usable by this session."""
    from ..multi_node.cli import cmd_bootstrap, install_geak_on_pods_best_effort

    if backend == "rayjob":
        if not head_ip:
            print(
                "multi-node: no HYPERLOOM_MN_EXT_HEAD_IP -- skipping the head-pod bootstrap "
                "(benchmark-only hand-off has no head pod to prepare).",
                file=sys.stderr,
            )
        else:
            ns_boot = argparse.Namespace(
                script=None,
                force=False,
                print_logs=False,
                poll_interval=6,
                # Adoption runs in-process, so it is not bound by the foreground/MCP 120s ceiling; a slow head pod
                # should not abort the run.
                poll_timeout=int(os.environ.get("HYPERLOOM_MN_POLL_TIMEOUT_S", "600") or 600),
            )
            rc_boot = cmd_bootstrap(ns_boot)
            if rc_boot != 0:
                sys.exit(rc_boot)

    if not getattr(args, "no_kernel", False):
        install_geak_on_pods_best_effort()

    _restore_kept_kernel_patches()


def _restore_kept_kernel_patches() -> None:
    """Put the session's kept kernel patches on the adopted pods before anything measures them."""
    from hyperloom.orchestrator.kernel.kept_patch_restore import KeptPatchRestoreError, restore_kept_patches_on_pods

    from ..multi_node._internal.external_state import external_has_server_control
    from ..session.session_paths import kernel_patch_backup_roots

    if not external_has_server_control():
        print(
            "multi-node: no server control -- skipping the kept kernel patch restore, since no restart would load it.",
            file=sys.stderr,
        )
        return
    try:
        restored = restore_kept_patches_on_pods(kernel_patch_backup_roots(_session_dir_resolve()))
    except KeptPatchRestoreError as exc:
        print(f"ERROR: {exc}; the pods would serve without this session's kept kernel patches.", file=sys.stderr)
        sys.exit(1)
    print(f"multi-node: restored {len(restored)} kept kernel file(s) on the pods")
