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
import time
from pathlib import Path

from .cli_executors import (  # noqa: F401 - re-exported for callers/tests
    _NOOP_KINDS_KERNEL_ONLY,
    _REAL_EXECUTORS_FULL,
    _build_specialist_executor,
    _noop_prep,
    _register_executors,
)
from .cli_kb import (  # noqa: F401 - re-exported for callers/tests
    _bootstrap_cortex_kb,
    _bootstrap_knowledge_plane,
    _build_recipe_kb_dispatcher,
    _resolve_local_kb_root,
)
from .cli_backends import (  # noqa: F401 - re-exported for callers/tests
    _MULTI_NODE_WORKLOAD_UID_ENV_KEYS,
    _build_backends,
    _build_proposal_scorer,
    _build_robustness_options,
    _robustness_server_configured,
)
from .cli_model_gate import (  # noqa: F401 - re-exported for callers/tests
    _AMD_GPU_TYPES,
    _AMD_UNSUPPORTED_ARCHITECTURES,
    _AMD_UNSUPPORTED_MODEL_TYPES,
    _AMD_UNSUPPORTED_QUANT_ALGOS,
    _AMD_UNSUPPORTED_QUANT_METHODS,
    _CONTEXT_HEADROOM_DEFAULT,
    _CONTEXT_HEADROOM_ENV,
    _GEMMA2_ARCHITECTURES,
    _GFX_TO_RUNNER,
    _MAX_MODEL_LEN_HEADROOM,
    _MAXPOS_CONFIG_KEYS,
    _NESTED_ONLY_UNRECOGNIZED_MODEL_TYPES,
    _PHI3_ROPE_TYPES,
    _ROPE_CONFIG_KEYS,
    _SAFETENSORS_HEADER_LIMIT,
    _SUPPORTED_ARCH_MARKERS,
    _SUPPORTED_MODEL_TYPES,
    _TEXT_COERCIBLE_MODEL_TYPES,
    _TEXT_DECODER_CONFIG_KEYS,
    _TOKENIZER_ARTIFACT_FILES,
    _UNRECOGNIZED_ARCHITECTURES,
    _UNRECOGNIZED_MODEL_TYPES,
    _UNREGISTERED_CUSTOM_CONFIG_TYPES,
    _UNSUPPORTED_ARCHITECTURES,
    _UNSUPPORTED_CONFIG_KEYS,
    _UNSUPPORTED_MODEL_TYPES,
    _VERDICT_TEXT_COERCIBLE,
    _VERDICT_VISION_ONLY,
    _VOCAB_WEIGHT_NAMES,
    _arch_is_supported_text_generation,
    _autodetect_gpu_type,
    _config_architectures,
    _config_declares_text_decoder,
    _context_headroom_tokens,
    _detect_amd_unsupported_quant,
    _detect_gemma2_missing_hidden_act,
    _detect_incompatible_model_config,
    _detect_missing_tokenizer_files,
    _detect_phi3_rope_scaling_incompatible,
    _detect_unrecognized_architecture,
    _detect_unsupported_model,
    _detect_vocab_weight_shape_mismatch,
    _gpu_runner_type,
    _load_model_arch,
    _load_model_config_dict,
    _load_model_config_tags,
    _load_model_max_position_embeddings,
    _model_has_dual_chunk_attention,
    _model_is_moe,
    _preflight_context_window,
    _preflight_model_config_compat,
    _preflight_unsupported_model_arch,
    _read_safetensors_header,
    _resolve_amd_gpu_type,
    _resolve_gpu_type,
    _resolve_max_model_len,
)
from .model_config_utils import (  # noqa: F401 - re-exported for callers/tests
    _model_is_gemma2,
    summarize_model_config,
)
from .cli_bootstrap import (  # noqa: F401 - re-exported for callers/tests
    _default_target_summary,
    _parse_conc_sweep_concs,
    _print_final_summary,
    _print_kernel_opt_summary_line,
    _print_session_skeleton,
    _read_failure_summary,
    _reconcile_crash_count,
    _resolve_reference_recipe,
    _resolve_session_dir_for_summary,
    _seed_shared_state,
    _snapshot_system_prompts,
    resolve_model_display_name,
)
from .session.paths import (
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
        str: The resolved multi-node backend (``rayjob`` or ``dynamo``).

    Raises:
        SystemExit: With code 2 when the resolved backend is invalid.
    """
    backend = (
        (getattr(args, "mn_backend", None) or "").strip()
        or os.environ.get("INFERENCE_OPTIMIZER_MN_BACKEND", "").strip()
        or os.environ.get("MN_BACKEND", "").strip()
        or "rayjob"
    ).lower()
    if backend not in ("rayjob", "dynamo"):
        print(
            f"ERROR: --mn-backend must be 'rayjob' or 'dynamo', got {backend!r}",
            file=sys.stderr,
        )
        sys.exit(2)
    return backend

def _provision_multi_node_dynamo_stack(args: argparse.Namespace) -> None:
    """When ``--nodes >= 2`` and ``--mn-backend dynamo``, create an idle
    DynamoDeployment and export the frontend service_url for benchmarks.

    No Ray head / bootstrap / RAY_ADDRESS: the worker pods are idle (sshd),
    and ``restart-server`` (routed by ``state.backend == 'dynamo'``) SSHes in
    to launch ``dynamo.sglang``/``dynamo.vllm``. Benchmarks target the Dynamo
    frontend (:8000) via ``state.service_url`` — picked up automatically by
    ``_multi_node_env.benchmark_env_for_subprocess``.

    Args:
        args (argparse.Namespace): The parsed CLI namespace (reads ``nodes``,
            ``rayjob_image``, ``rayjob_gpus_per_node``, and PD flags).

    Raises:
        SystemExit: With code 2 when a required Dynamo image is not resolvable.
    """
    nodes = max(1, int(args.nodes))
    from .multi_node.cli import cmd_create_dynamo, _load_state

    state_path = Path(os.environ.get("MULTI_NODE_STATE_FILE", "/tmp/multi_node_state.json"))
    image = (getattr(args, "rayjob_image", None) or "").strip() or os.environ.get(
        "INFERENCE_OPTIMIZER_RAYJOB_IMAGE", ""
    ).strip()
    if not image and state_path.is_file():
        try:
            prior = json.loads(state_path.read_text(encoding="utf-8"))
            image = str((prior.get("last_create_request") or {}).get("image") or "").strip()
        except (OSError, json.JSONDecodeError, TypeError):
            image = ""
    if not image:
        print(
            "ERROR: --nodes >= 2 --mn-backend dynamo requires a Dynamo image "
            "WITH the sshd layer (mn-idle.sh). Pass --rayjob-image <harbor/...> "
            "or set INFERENCE_OPTIMIZER_RAYJOB_IMAGE.",
            file=sys.stderr,
        )
        sys.exit(2)

    gpn = getattr(args, "rayjob_gpus_per_node", None)
    if gpn is None:
        try:
            gpn = int(os.environ.get("INFERENCE_OPTIMIZER_GPUS_PER_NODE", "8") or 8)
        except ValueError:
            gpn = 8

    poll_timeout = int(os.environ.get("HYPERLOOM_MN_POLL_TIMEOUT_S", "110") or 110)
    # Hyperloom patch (operator-local): forward --pd-* flags so the auto-created
    # DGD honours the operator's PD topology AND so cmd_create_dynamo's reuse
    # path classifies pods under the correct prefill/decode roles when writing
    # /tmp/multi_node_state.json. Without these forwards the helper defaults
    # pd_mode to 'aggregated' regardless of the optimize CLI, and downstream
    # restart_server_for_round finds zero prefill/decode pods → no SSH launch →
    # baseline gets 0 completed requests → baseline_failed after 3 attempts.
    _pd_kv_backend = (getattr(args, "pd_transfer_backend", "") or "").strip() or os.environ.get(
        "INFERENCE_OPTIMIZER_DYNAMO_KV_BACKEND", "nixl"
    )
    ns_create = argparse.Namespace(
        workspace=None,
        image=image,
        nodes=nodes,
        gpus_per_node=int(gpn),
        cpus_per_node=96,
        mem_per_node=1024,
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
        # PD topology forward (missing in upstream; see comment above).
        pd_mode=(getattr(args, "pd_mode", "") or "aggregated"),
        pd_prefill_nodes=int(getattr(args, "pd_prefill_nodes", 0) or 0),
        pd_decode_nodes=int(getattr(args, "pd_decode_nodes", 0) or 0),
        pd_prefill_tp=int(getattr(args, "pd_prefill_tp", 0) or 0),
        pd_decode_tp=int(getattr(args, "pd_decode_tp", 0) or 0),
    )
    rc = cmd_create_dynamo(ns_create)
    if rc != 0:
        sys.exit(rc)

    state = _load_state()
    su = str(state.get("service_url") or "").strip()
    if su:
        # Belt-and-suspenders: benchmark_env_for_subprocess also reads
        # state.service_url, but exporting here makes the frontend URL
        # visible to any early shell-level Magpie call too.
        os.environ["BENCHMARK_BASE_URL"] = su
        os.environ["MAGPIE_RUN_PHASE"] = "client"
        print(f"multi-node(dynamo): BENCHMARK_BASE_URL={su} (frontend :8000)")

    # Install GEAK on the GPU pods over SSH (one-time, idempotent) so the
    # kernel-agent SSH placement finds `geak` on PATH. Skipped when the run
    # opted out of the kernel phase. Best-effort: a failure here surfaces
    # later as a clear pod-side "geak CLI not found" rather than aborting
    # provisioning. Dynamo-only (the helper no-ops for other backends).
    if not getattr(args, "no_kernel", False):
        from .multi_node.cli import install_kernel_tools_on_pods_best_effort

        install_kernel_tools_on_pods_best_effort()

def _provision_multi_node_rayjob_stack(args: argparse.Namespace) -> None:
    """When ``--nodes >= 2``, create/reuse SaFE RayJob, bootstrap once, export RAY_ADDRESS.

    For ``--mn-backend dynamo`` this delegates to
    :func:`_provision_multi_node_dynamo_stack` (idle DynamoDeployment + SSH).

    No-op when ``--nodes < 2``. The RayJob path resolves the container image
    (CLI flag → env → prior state file), creates or reuses the RayJob, runs the
    one-time bootstrap if it hasn't run yet, exports ``RAY_ADDRESS`` for
    kernel-agent Ray tasks, sets ``HYPERLOOM_MN_PROFILE_TRACE_DIR`` to a
    cluster-shared trace directory namespaced by ``rayjob_id`` (GC'ing older
    sibling dirs), and replays previously-applied kernel patches onto the
    (possibly fresh) pods.

    Raises:
        SystemExit: With code 2 when ``--nodes >= 2`` but no RayJob image is
            configured, or with the create/bootstrap return code on failure.
    """
    nodes = max(1, int(args.nodes))
    if nodes < 2:
        return

    if _resolve_mn_backend(args) == "dynamo":
        _provision_multi_node_dynamo_stack(args)
        return

    from .multi_node.cli import cmd_bootstrap, cmd_create_rayjob, _load_state
    from hyperloom.orchestrator.actions.executors._multi_node_env import export_ray_address_to_os

    state_path = Path(os.environ.get("MULTI_NODE_STATE_FILE", "/tmp/multi_node_state.json"))
    image = (getattr(args, "rayjob_image", None) or "").strip() or os.environ.get(
        "INFERENCE_OPTIMIZER_RAYJOB_IMAGE", ""
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
            "--rayjob-image <harbor/...> or set INFERENCE_OPTIMIZER_RAYJOB_IMAGE.",
            file=sys.stderr,
        )
        sys.exit(2)

    gpn = getattr(args, "rayjob_gpus_per_node", None)
    if gpn is None:
        try:
            gpn = int(os.environ.get("INFERENCE_OPTIMIZER_GPUS_PER_NODE", "8") or 8)
        except ValueError:
            gpn = 8

    # Forward agent-supplied prompt env verbatim; no-op on RayJob reuse (see multi_node/SKILL.md).
    rayjob_extra_env = list(getattr(args, "rayjob_extra_env", None) or [])

    ns_create = argparse.Namespace(
        workspace=None,
        image=image,
        nodes=nodes,
        gpus_per_node=int(gpn),
        cpus_per_node=96,
        mem_per_node=1024,
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

    # Multi-node: server pods must write torch traces to a sandbox-readable wekafs path, namespaced by rayjob_id.
    state_after = _load_state()
    rid = (state_after.get("rayjob_id") or "").strip()
    if rid:
        # Anchor torch-profile shared root on $USER_DATA_PATH so sandbox and RayJob pods see the same path.
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
            # Best-effort GC of older sibling RayJob trace dirs (active rayjob_id name-guarded).
            _gc_old_profile_traces(keep=rid)

    # RayJob recreate path: replay promoted patches from optimization_stack since fresh pods lost them.
    # Best-effort; failures degrade to warnings (orchestrator re-runs kernel-agent on missing speedups).
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
