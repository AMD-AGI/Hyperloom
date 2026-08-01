# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI entry — ``optimize`` subcommand wiring Claude+Codex backends, executors, objective, and Coordinator.run().

Env vars consumed: MODEL_PATH, OPENAI_BASE_URL + SAFE_API_KEY, ROCR_VISIBLE_DEVICES,
CLAUDE_MODEL, CODEX_MODEL, USER_DATA_PATH.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from ..session.paths import (
    mn_profile_trace_root,
    session_dir as _session_dir_resolve,
)

log = logging.getLogger(__name__)


def _gc_old_profile_traces(
    root: str | None = None,
    retention_days: int = 7,
    keep: str | None = None,
) -> None:
    """Best-effort GC of stale per-RayJob profile-trace dirs older than ``retention_days`` (``keep`` name-guarded).

    Never blocks startup (errors swallowed). Env knobs: HYPERLOOM_MN_TRACE_RETENTION_DAYS,
    HYPERLOOM_MN_TRACE_GC_DISABLE.

    Args:
        root (str | None): The trace-root directory to scan; defaults to the
            multi-node profile-trace root.
        retention_days (int): Age threshold in days before a trace dir is
            removed (env-overridable).
        keep (str | None): A directory name to always preserve (name-matched).
    """
    if os.environ.get("HYPERLOOM_MN_TRACE_GC_DISABLE", "").strip() in (
        "1",
        "true",
        "yes",
    ):
        return
    try:
        retention_days = int(os.environ.get("HYPERLOOM_MN_TRACE_RETENTION_DAYS") or retention_days)
    except ValueError:
        retention_days = 7
    base = Path(root) if root is not None else mn_profile_trace_root()
    if not base.is_dir():
        return
    cutoff = time.time() - retention_days * 86400
    keep_name = Path(keep).name if keep else ""
    removed = 0
    kept = 0
    try:
        for child in base.iterdir():
            if not child.is_dir():
                continue
            if keep_name and child.name == keep_name:
                kept += 1
                continue
            try:
                mtime = child.stat().st_mtime
            except OSError:
                continue
            if mtime >= cutoff:
                kept += 1
                continue
            try:
                shutil.rmtree(child)
                removed += 1
            except OSError as exc:
                print(
                    f"WARN multi-node GC: failed to rm {child}: {exc}",
                    file=sys.stderr,
                )
    except OSError as exc:
        print(f"WARN multi-node GC: scan failed under {base}: {exc}", file=sys.stderr)
        return
    if removed or kept:
        print(f"multi-node: GC profile-traces removed={removed} kept={kept} retention={retention_days}d root={base}")


def _resolve_mn_backend(args: argparse.Namespace) -> str:
    """Multi-node backend selector: --mn-backend > $INFERENCE_OPTIMIZER_MN_BACKEND > rayjob.

    Args:
        args (argparse.Namespace): The parsed CLI namespace (reads
            ``mn_backend``).

    Returns:
        str: The resolved multi-node backend (``rayjob`` or ``infera``).

    Raises:
        SystemExit: With code 2 when the resolved backend is invalid.
    """
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
    """Adopt the platform-provisioned cluster for a ``--nodes >= 2`` run.

    The cluster — RayJob head+workers, or an idle InferaDeployment whose pods run
    sshd — is provisioned by the platform before the optimizer starts and handed
    over through the ``HYPERLOOM_MN_EXT_*`` env vars. This synthesizes
    ``multi_node_state.json`` from them so every downstream step
    (``restart-server``, SSH fan-out, GPU sampling, Magpie client mode) reads one
    stable source, and points benchmarks at the cluster's frontend.

    Nothing here creates or releases a cluster; that is the platform's job. The
    session-side setup the adopted cluster still needs runs in
    :func:`_prepare_adopted_cluster`. No-op when ``--nodes < 2``.

    Args:
        args: Parsed CLI namespace (reads ``nodes`` and ``mn_backend``).

    Raises:
        SystemExit: With code 2 when ``--nodes >= 2`` but no cluster was handed
            over, when an infera cluster arrives without SSH control, or when the
            synthesized state cannot be written; with the bootstrap return code
            when that fails.
    """
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
    try:
        _save_state(ext_state)
    except OSError as exc:
        print(f"ERROR: cannot write multi-node state: {exc}", file=sys.stderr)
        sys.exit(2)
    os.environ["BENCHMARK_BASE_URL"] = ext_state["service_url"]
    os.environ["MAGPIE_RUN_PHASE"] = "client"
    export_ray_address_to_os()
    # Report server *control*, not SSH specifically: rayjob restarts through the
    # Ray dashboard with no SSH at all, so keying this on SSH labelled a fully
    # controllable rayjob cluster "benchmark-only" and gave the genuinely
    # uncontrollable one the same words -- exactly backwards on the one line an
    # operator reads to find out whether the run can tune anything.
    has_control = external_has_server_control()
    print(
        "multi-node: adopted the platform-provisioned cluster. "
        f"url={ext_state['service_url']} prefill={ext_state['prefill_pod_ips']} "
        f"decode={ext_state['decode_pod_ips']} worker={ext_state['worker_pod_ips']} "
        f"server_control={'yes' if has_control else 'no (benchmark-only)'} "
        f"ssh_control={'yes' if external_has_ssh_control() else 'no'}"
    )
    if not has_control:
        # Loud, not fatal: benchmark-only is a documented mode (multi_node/SKILL.md),
        # but without a restart every candidate re-measures the one unchanged
        # server, so the run still reports gains that no config produced.
        print(
            "WARNING: no server control -- per-round restarts will be skipped, so every "
            "candidate config measures the SAME unchanged server and the reported gains "
            "are meaningless. Set HYPERLOOM_MN_EXT_HEAD_IP (rayjob) or "
            "HYPERLOOM_MN_EXT_SSH_KEY plus one of _PREFILL_IPS / _DECODE_IPS / _WORKER_IPS "
            "(infera) to tune. Continuing in benchmark-only mode.",
            file=sys.stderr,
        )
    _prepare_adopted_cluster(args, str(ext_state["backend"]))


def _prepare_adopted_cluster(args: argparse.Namespace, backend: str) -> None:
    """Make an adopted cluster usable by this session.

    Adopting only records where the cluster is. These steps are what the run
    still needs from it, and none of them provision anything -- they went on
    working against a handed-over cluster once the create path was removed:

    * rayjob: the BYOI bootstrap renders ``/etc/profile.d/hyperloom-env.sh`` in
      the head pod, which every later Ray Dashboard REST job sources to find the
      framework venv. It is submitted unconditionally rather than tracked in the
      state file: the synthesized state carries no submission id, and
      ``bootstrap.sh`` already self-skips on its pod-side marker.
    * infera: GEAK is installed on the GPU pods over SSH so the kernel agent
      finds it on PATH. Best-effort, and the helper no-ops for other backends.
    * both: kernel patches applied earlier in this session are replayed, so a
      cluster adopted mid-session does not serve from the pre-patch state.

    Args:
        args: Parsed CLI namespace (reads ``nodes`` and ``no_kernel``).
        backend: The resolved multi-node backend.

    Raises:
        SystemExit: With the bootstrap return code when the head pod fails it.
    """
    from ..multi_node.cli import cmd_bootstrap, install_geak_on_pods_best_effort

    if backend == "rayjob":
        ns_boot = argparse.Namespace(
            script=None,
            force=False,
            print_logs=False,
            poll_interval=6,
            # Adoption runs in-process, so it is not bound by the foreground/MCP
            # 120s ceiling; a slow head pod should not abort the run.
            poll_timeout=int(os.environ.get("HYPERLOOM_MN_POLL_TIMEOUT_S", "600") or 600),
        )
        rc_boot = cmd_bootstrap(ns_boot)
        if rc_boot != 0:
            sys.exit(rc_boot)

    if not getattr(args, "no_kernel", False):
        install_geak_on_pods_best_effort()

    _replay_kernel_patches_for_multi_node(args)


def _replay_kernel_patches_for_multi_node(args: argparse.Namespace) -> None:
    """Replay every applied kernel-agent patch (manifest status=applied + multinode block) onto RayJob pods.

    Idempotent ``apply-patch`` fan-out, run only when ``--nodes>=2``. Best-effort: per-patch failures warn.

    Args:
        args: Parsed CLI arguments; reads ``nodes`` and resolves the session
            workspace to locate applied-patch manifests.
    """
    nodes = max(1, int(getattr(args, "nodes", 1) or 1))
    if nodes < 2:
        return
    session_dir = _session_dir_resolve()
    workspace_root = session_dir / "kernel-agent-workspace"
    if not workspace_root.is_dir():
        return

    manifests: list[Path] = sorted(workspace_root.rglob("manifest.json"))
    if not manifests:
        return

    replayed = 0
    skipped = 0
    failed = 0
    for mpath in manifests:
        try:
            data = json.loads(mpath.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(
                f"WARN multi-node patch replay: skipping unreadable manifest {mpath}: {exc}",
                file=sys.stderr,
            )
            skipped += 1
            continue
        if str(data.get("status", "")).lower() != "applied":
            continue
        mn = data.get("multinode") or {}
        if not mn:
            continue
        target_file = data.get("target_file") or ""
        patch_path = data.get("patch_path") or ""
        kernel_id = data.get("kernel_id") or ""
        backup_dir_on_pod = mn.get("backup_dir_on_pod") or "/var/kernel_patch_backups"
        if not target_file or not patch_path:
            skipped += 1
            continue
        if not Path(patch_path).is_file():
            print(
                f"WARN multi-node patch replay: source patch missing for "
                f"{target_file} (manifest={mpath} patch_path={patch_path}); "
                f"skipping",
                file=sys.stderr,
            )
            skipped += 1
            continue
        cmd = [
            sys.executable,
            "-m",
            "hyperloom.inference_optimizer.multi_node",
            "apply-patch",
            "--patch-file",
            str(patch_path),
            "--target-path",
            str(target_file),
            "--backup-dir",
            str(backup_dir_on_pod),
            "--kernel-id",
            str(kernel_id),
        ]
        print(
            f"multi-node patch replay: target={target_file} kernel_id={kernel_id!r} (from {mpath})",
        )
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            failed += 1
            print(
                f"WARN multi-node patch replay failed for {target_file} "
                f"rc={proc.returncode} stderr={(proc.stderr or '')[-1000:]!r}",
                file=sys.stderr,
            )
            continue
        replayed += 1
    if replayed or failed or skipped:
        print(
            f"multi-node patch replay: applied={replayed} "
            f"skipped={skipped} failed={failed} "
            f"(scanned {len(manifests)} manifest(s) under {workspace_root})"
        )


def _dump_mn_input_params(args: argparse.Namespace, nodes_resolved: int) -> None:
    """Dump resolved multi-node input params (CLI args + relevant env) to
    ``$USER_DATA_PATH/optimizer_runs`` for tracing the env->CLI migration.

    Many knobs moved from env vars to CLI flags; this snapshot records both
    the parsed CLI namespace and the multi-node-relevant env at launch so a
    mismatch (e.g. a flag not picking up its old env, or a stale env still
    leaking) is auditable post-hoc. Secrets are redacted; best-effort and
    never raises.

    Args:
        args: The parsed optimize CLI namespace.
        nodes_resolved: The resolved node count (>=2 for multi-node).
    """
    try:
        import datetime as _dt

        base = os.path.expandvars("$USER_DATA_PATH/optimizer_runs")
        if "$" in base or not base.startswith("/"):
            base = str(Path(tempfile.gettempdir()) / "optimizer_runs")
        os.makedirs(base, exist_ok=True)
        ts = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

        def _redact(key: str, val: Any) -> Any:
            kl = str(key).lower()
            if any(tok in kl for tok in ("api_key", "secret", "token", "password")):
                return "***REDACTED***"
            return val

        cli: dict[str, Any] = {}
        for k, v in sorted(vars(args).items()):
            try:
                json.dumps(v)
                cli[k] = _redact(k, v)
            except (TypeError, ValueError):
                cli[k] = _redact(k, repr(v))

        env_prefixes = (
            "INFERENCE_OPTIMIZER_",
            "HYPERLOOM_MN_",
            "PD_",
            "SAFE_",
            "MAGPIE_",
            "SGLANG_",
            "NCCL_",
            "MC_",
            "MORI_",
            "AITER_",
        )
        env_exact = (
            "MODEL_PATH",
            "FRAMEWORK",
            "TP",
            "EP",
            "NODES",
            "MN_BACKEND",
            "MODEL_CLASS",
            "PRECISION",
            "USER_DATA_PATH",
            "BENCHMARK_BASE_URL",
            "SKIP_VARIANTS",
            "RUN_EVAL",
            "GPU_TYPE",
            "ISL",
            "OSL",
            "CONC",
            "RANDOM_RANGE_RATIO",
            "INFERENCEX_PATH",
            "TRACELENS_ROOT",
        )
        env: dict[str, str] = {}
        for k, v in sorted(os.environ.items()):
            if k.startswith(env_prefixes) or k in env_exact:
                env[k] = _redact(k, v)

        out = os.path.join(base, "mn_input_params_" + ts + ".json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(
                {"ts": ts, "nodes_resolved": nodes_resolved, "cli_args": cli, "env": env},
                f,
                indent=2,
                sort_keys=True,
            )
        print("multi-node input params dumped -> " + out)
        log.info("multi-node input params (nodes=%d) dumped to %s", nodes_resolved, out)
    except Exception as exc:  # noqa: BLE001 - tracing aid must never break the run
        log.warning("failed to dump multi-node input params: %r", exc)
