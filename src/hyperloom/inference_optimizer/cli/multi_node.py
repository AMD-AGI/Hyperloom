# Copyright Advanced Micro Devices, Inc. All rights reserved.

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


def _per_node_resources(args: argparse.Namespace) -> tuple[int, int]:
    """Resolve (cpus, mem_gib) per node from the optimize CLI flags.

    Both the RayJob and Infera provision paths honour ``--cpus-per-node`` /
    ``--mem-per-node``; when a flag is unset the create-* subcommand defaults
    (96 CPUs, 1024 GiB) apply. No environment variables are consulted.
    """
    cpus = getattr(args, "cpus_per_node", None)
    mem = getattr(args, "mem_per_node", None)
    return (
        int(cpus) if cpus is not None else 96,
        int(mem) if mem is not None else 1024,
    )


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

def _provision_multi_node_infera_stack(args: argparse.Namespace) -> None:
    """When ``--nodes >= 2`` and ``--mn-backend infera``, create an idle
    InferaDeployment and export the frontend service_url for benchmarks.

    No Ray head / bootstrap / RAY_ADDRESS: the worker pods are idle (sshd),
    and ``restart-server`` (routed by ``state.backend == 'infera'``) SSHes in
    to launch ``infera.engine.sglang``/``infera.engine.vllm``. Benchmarks
    target the Infera frontend (:8000) via ``state.service_url`` — picked up
    automatically by ``_multi_node_env.benchmark_env_for_subprocess``.

    Args:
        args (argparse.Namespace): The parsed CLI namespace (reads ``nodes``,
            ``model``, ``mn_image``, ``gpus_per_node``, and PD flags).

    Raises:
        SystemExit: With code 2 when a required Infera image is not resolvable.
    """
    nodes = max(1, int(args.nodes))
    from ..multi_node.cli import cmd_create_infera, _load_state
    from ..multi_node.state_paths import resolve_state_file

    state_path = resolve_state_file()
    image = (getattr(args, "mn_image", None) or "").strip() or os.environ.get(
        "INFERENCE_OPTIMIZER_MN_IMAGE", ""
    ).strip()
    if not image and state_path.is_file():
        try:
            prior = json.loads(state_path.read_text(encoding="utf-8"))
            image = str((prior.get("last_create_request") or {}).get("image") or "").strip()
        except (OSError, json.JSONDecodeError, TypeError):
            image = ""
    if not image:
        print(
            "ERROR: --nodes >= 2 --mn-backend infera requires an Infera image "
            "WITH the sshd layer (mn-idle.sh). Pass --mn-image <harbor/...> "
            "or set INFERENCE_OPTIMIZER_MN_IMAGE.",
            file=sys.stderr,
        )
        sys.exit(2)

    # The Infera frontend needs the model path for --router-tokenizer-path.
    model = (str(getattr(args, "model", None) or "").strip()) or os.environ.get("MODEL_PATH", "").strip()
    if not model:
        print(
            "ERROR: --nodes >= 2 --mn-backend infera requires --model (or "
            "$MODEL_PATH) for the Infera frontend --router-tokenizer-path.",
            file=sys.stderr,
        )
        sys.exit(2)

    gpn = getattr(args, "gpus_per_node", None)
    if gpn is None:
        try:
            gpn = int(os.environ.get("INFERENCE_OPTIMIZER_GPUS_PER_NODE", "8") or 8)
        except ValueError:
            gpn = 8

    poll_timeout = int(os.environ.get("HYPERLOOM_MN_POLL_TIMEOUT_S", "110") or 110)
    # Hyperloom patch (operator-local): forward --pd-* flags so the auto-created
    # IDEP honours the operator's PD topology AND so cmd_create_infera's reuse
    # path classifies pods under the correct prefill/decode roles when writing
    # /tmp/multi_node_state.json. Without these forwards the helper defaults
    # pd_mode to 'aggregated' regardless of the optimize CLI, and downstream
    # restart_server_for_round finds zero prefill/decode pods → no SSH launch →
    # baseline gets 0 completed requests → baseline_failed after 3 attempts.
    _pd_kv_backend = (getattr(args, "pd_transfer_backend", "") or "").strip() or os.environ.get(
        "INFERENCE_OPTIMIZER_INFERA_KV_BACKEND", "mori"
    )
    cpus_per_node, mem_per_node = _per_node_resources(args)
    ns_create = argparse.Namespace(
        workspace=None,
        image=image,
        model=model,
        nodes=nodes,
        gpus_per_node=int(gpn),
        cpus_per_node=cpus_per_node,
        mem_per_node=mem_per_node,
        ephemeral_per_node=400,
        shared_mem_per_node=200,
        backend_framework=(getattr(args, "framework", None) or "sglang"),
        kv_transfer_backend=_pd_kv_backend,
        ssh_port=int(os.environ.get("MN_SSH_PORT", "2222") or 2222),
        display_name=None,
        description=None,
        owner_id=None,
        extra_env=list(getattr(args, "rayjob_extra_env", None) or []),
        extra_label=[],
        no_wait=False,
        recreate=False,
        poll_interval=6,
        poll_timeout=poll_timeout,
        # PD topology forward (see comment above).
        pd_mode=(getattr(args, "pd_mode", "") or "aggregated"),
        pd_prefill_nodes=int(getattr(args, "pd_prefill_nodes", 0) or 0),
        pd_decode_nodes=int(getattr(args, "pd_decode_nodes", 0) or 0),
        pd_prefill_tp=int(getattr(args, "pd_prefill_tp", 0) or 0),
        pd_decode_tp=int(getattr(args, "pd_decode_tp", 0) or 0),
    )
    rc = cmd_create_infera(ns_create)
    if rc != 0:
        sys.exit(rc)

    state = _load_state()
    su = str(state.get("service_url") or "").strip()
    if su:
        # Also export here so the frontend URL is visible to any early
        # shell-level Magpie call.
        os.environ["BENCHMARK_BASE_URL"] = su
        os.environ["MAGPIE_RUN_PHASE"] = "client"
        print(f"multi-node(infera): BENCHMARK_BASE_URL={su} (frontend :8000)")

    # Install GEAK kernel tooling on the GPU pods (SSH-installed so the
    # kernel-agent finds `geak` on PATH). Skipped when the run opted out of
    # the kernel phase. Best-effort (failures surface later as clear pod-side
    # errors) and Infera-only (the helper no-ops for other backends).
    from ..multi_node.cli import install_geak_on_pods_best_effort

    if not getattr(args, "no_kernel", False):
        install_geak_on_pods_best_effort()

def _provision_multi_node_rayjob_stack(args: argparse.Namespace) -> None:
    """When ``--nodes >= 2``, create/reuse SaFE RayJob, bootstrap once, export RAY_ADDRESS.

    When SaFE is absent and ``HYPERLOOM_MN_EXT_SERVICE_URL`` is set, synthesize
    ``multi_node_state.json`` from env and skip all SaFE create/init (external
    mode). Downstream restart / SSH / Magpie client mode read the synthetic state.

    For ``--mn-backend infera`` this delegates to
    :func:`_provision_multi_node_infera_stack` (idle InferaDeployment + SSH).

    No-op when ``--nodes < 2``. The RayJob path resolves the container image
    (CLI flag → env → prior state file), creates or reuses the RayJob, runs the
    one-time bootstrap if it hasn't run yet, exports ``RAY_ADDRESS`` for
    kernel-agent Ray tasks, sets ``HYPERLOOM_MN_PROFILE_TRACE_DIR`` to a
    cluster-shared trace directory namespaced by ``rayjob_id`` (GC'ing older
    sibling dirs), and replays previously-applied kernel patches onto the
    (possibly fresh) pods.

    Raises:
        SystemExit: With code 2 when ``--nodes >= 2`` but no RayJob image is
            configured, when external infera lacks SSH control, or with the
            create/bootstrap return code on failure.
    """
    from ..multi_node._internal.external_state import (
        build_external_state_from_env,
        external_has_ssh_control,
        external_service_url,
    )

    if external_service_url():
        from ..multi_node.cli import _save_state
        from hyperloom.orchestrator.actions.executors._multi_node_env import export_ray_address_to_os

        ext_state = build_external_state_from_env()
        ext_state["backend"] = _resolve_mn_backend(args)
        if ext_state["backend"] == "infera" and not external_has_ssh_control():
            print(
                "ERROR: external infera mode requires SSH control. Set "
                "HYPERLOOM_MN_EXT_SSH_KEY plus HYPERLOOM_MN_EXT_PREFILL_IPS/"
                "DECODE_IPS (or WORKER_IPS). rayjob external uses Ray, not SSH.",
                file=sys.stderr,
            )
            sys.exit(2)
        try:
            _save_state(ext_state)
        except OSError as exc:
            print(f"ERROR: external mode: cannot write synthetic state: {exc}", file=sys.stderr)
            sys.exit(2)
        os.environ["BENCHMARK_BASE_URL"] = ext_state["service_url"]
        os.environ["MAGPIE_RUN_PHASE"] = "client"
        export_ray_address_to_os()
        print(
            "multi-node(external): SaFE bypassed; state synthesized from env. "
            f"url={ext_state['service_url']} prefill={ext_state['prefill_pod_ips']} "
            f"decode={ext_state['decode_pod_ips']} worker={ext_state['worker_pod_ips']} "
            f"ssh_control={'yes' if external_has_ssh_control() else 'no (benchmark-only)'}"
        )
        return

    nodes = max(1, int(args.nodes))
    if nodes < 2:
        return

    if _resolve_mn_backend(args) == "infera":
        _provision_multi_node_infera_stack(args)
        return

    from ..multi_node.cli import cmd_bootstrap, cmd_create_rayjob, _load_state
    from ..multi_node.state_paths import resolve_state_file
    from hyperloom.orchestrator.actions.executors._multi_node_env import export_ray_address_to_os

    state_path = resolve_state_file()
    image = (getattr(args, "mn_image", None) or "").strip() or os.environ.get(
        "INFERENCE_OPTIMIZER_MN_IMAGE", ""
    ).strip()
    if not image and state_path.is_file():
        try:
            prior = json.loads(state_path.read_text(encoding="utf-8"))
            image = str((prior.get("last_create_request") or {}).get("image") or "").strip()
        except (OSError, json.JSONDecodeError, TypeError):
            image = ""
    if not image:
        print(
            "ERROR: --nodes >= 2 requires a RayJob container image. Pass "
            "--mn-image <harbor/...> or set INFERENCE_OPTIMIZER_MN_IMAGE.",
            file=sys.stderr,
        )
        sys.exit(2)

    gpn = getattr(args, "gpus_per_node", None)
    if gpn is None:
        try:
            gpn = int(os.environ.get("INFERENCE_OPTIMIZER_GPUS_PER_NODE", "8") or 8)
        except ValueError:
            gpn = 8

    # Forward agent-supplied prompt env verbatim; no-op on RayJob reuse (see multi_node/SKILL.md).
    rayjob_extra_env = list(getattr(args, "rayjob_extra_env", None) or [])

    cpus_per_node, mem_per_node = _per_node_resources(args)
    ns_create = argparse.Namespace(
        workspace=None,
        image=image,
        nodes=nodes,
        gpus_per_node=int(gpn),
        cpus_per_node=cpus_per_node,
        mem_per_node=mem_per_node,
        ephemeral_per_node=400,
        display_name=None,
        description=None,
        owner_id=None,
        extra_env=rayjob_extra_env,
        extra_label=[],
        no_wait=False,
        recreate=False,
        poll_interval=6,
        poll_timeout=int(os.environ.get("HYPERLOOM_MN_POLL_TIMEOUT_S", "110") or 110),
    )
    rc = cmd_create_rayjob(ns_create)
    if rc != 0:
        sys.exit(rc)

    state = _load_state()
    if not state.get("last_bootstrap_submission_id"):
        ns_boot = argparse.Namespace(
            script=None,
            force=False,
            print_logs=False,
            poll_interval=6,
            poll_timeout=int(os.environ.get("HYPERLOOM_MN_POLL_TIMEOUT_S", "110") or 110),
        )
        rc_boot = cmd_bootstrap(ns_boot)
        if rc_boot != 0:
            sys.exit(rc_boot)

    export_ray_address_to_os()
    ra = os.environ.get("RAY_ADDRESS", "")
    if ra:
        print(f"multi-node: exported RAY_ADDRESS={ra} for kernel-agent Ray tasks")

    # Server pods write torch traces to a sandbox-readable wekafs path, namespaced by rayjob_id.
    state_after = _load_state()
    rid = (state_after.get("rayjob_id") or "").strip()
    if rid:
        # Anchor the torch-profile shared root on $USER_DATA_PATH so sandbox and pods agree.
        trace_root_path = mn_profile_trace_root() / rid / "torch_trace"
        trace_root = str(trace_root_path)
        try:
            trace_root_path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            print(
                f"WARN multi-node: cannot mkdir {trace_root}: {exc}; server traces will fall back to per-pod /tmp",
                file=sys.stderr,
            )
        else:
            os.environ["HYPERLOOM_MN_PROFILE_TRACE_DIR"] = trace_root
            print(f"multi-node: exported HYPERLOOM_MN_PROFILE_TRACE_DIR={trace_root}")
            # GC older sibling RayJob trace dirs (active rayjob_id name-guarded).
            _gc_old_profile_traces(keep=rid)

    # RayJob recreate path: replay promoted patches since fresh pods lost them. Best-effort.
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
            "INFERENCE_OPTIMIZER_", "HYPERLOOM_MN_", "PD_", "SAFE_", "MAGPIE_",
            "SGLANG_", "NCCL_", "MC_", "MORI_", "AITER_",
        )
        env_exact = (
            "MODEL_PATH", "FRAMEWORK", "TP", "EP", "NODES", "MN_BACKEND",
            "MODEL_CLASS", "PRECISION", "USER_DATA_PATH", "SAFE_WORKSPACE",
            "BENCHMARK_BASE_URL", "SKIP_VARIANTS", "RUN_EVAL",
            "GPU_TYPE", "ISL", "OSL", "CONC", "RANDOM_RANGE_RATIO",
            "INFERENCEX_PATH", "TRACELENS_ROOT",
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
