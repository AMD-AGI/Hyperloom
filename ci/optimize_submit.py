#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""ci/optimize_submit.py — Hyperloom CI variant of SaFE optimize_submit.

Submits SaFE inference optimization tasks. Reuses the same SaFE bearer token
as the rest of Hyperloom CI (CLAW_API_KEY).

Tracks the SaFE script's API contract (Primus-SaFE/scripts/optimize_submit.py
as of 2026-05-06):
  POST /api/v1/playground/models   body = {source, workspace, target.volume}
  GET  /api/v1/playground/models/{id}
  POST /api/v1/optimization/tasks  body = {modelId, mode=local, framework, ...}

Notes on tools / mode:
  - SaFE backend hard-codes Claw Tools=[16,18] for optimization tasks
    (apiserver/.../optimization/handler.go), so the client never sends a
    tools field. This is independent of the [67] used by Hyperloom's existing
    Claw-direct CI (ci-config.yaml) — different code path.
  - mode=local (default): prompt tells the agent "SandboxImage: ..." and the
    agent runs benchmarks directly in the sandbox.
  - mode=claw: prompt warns the agent it cannot reach /shared_nfs directly
    and must go through Claw (RayJob fan-out).

Usage:
  # Auto mode — single model
  python3 optimize_submit.py --model Qwen/Qwen3-8B

  # Auto mode — multiple models
  python3 optimize_submit.py --model Qwen/Qwen3-8B meta-llama/Llama-3.1-70B-Instruct

  # Auto mode — top-N from HuggingFace, filtered by size
  python3 optimize_submit.py --hf-top 10 --min-params 7

  # Dry run + write manifest for CI artifact
  python3 optimize_submit.py --hf-top 5 --dry-run --output-dir submit-output

Env vars (all optional, CLI flags take precedence):
  CLAW_API_KEY | SAFE_API_KEY        bearer token (ak-xxx)
  SAFE_BASE_URL | SAFE_API_URL       base URL (default: https://core42.example-internal-host.invalid)
  HF_TOKEN                           HuggingFace token (gated models)
  SAFE_OPTIMIZE_WORKSPACE            override default 'core42-hyperloom'
  SAFE_OPTIMIZE_VOLUME               override default '/wekafs'

Implementation note:
  This file is intentionally kept as the stable CLI/import facade. The
  implementation lives under ci/optimize_submit_lib/ so individual modules stay
  reviewable while workflows keep running python3 optimize_submit.py.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path

# Keep local sibling imports (model_compat, optimize_submit_lib) available when
# the script is executed directly from ci/ or imported by tests.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from optimize_submit_lib import artifacts as _artifacts
from optimize_submit_lib import artifact_paths as _artifact_paths
from optimize_submit_lib import backfill as _backfill
from optimize_submit_lib import config as _config
from optimize_submit_lib import delivery as _delivery
from optimize_submit_lib import detect as _detect
from optimize_submit_lib import flow as _flow
from optimize_submit_lib import hf_client as _hf_client
from optimize_submit_lib import manifest as _manifest
from optimize_submit_lib import nfs_collect as _nfs_collect
from optimize_submit_lib import records as _records
from optimize_submit_lib import safe_client as _safe_client

for _module in (
    _config,
    _hf_client,
    _detect,
    _safe_client,
    _records,
    _artifact_paths,
    _delivery,
    _backfill,
    _nfs_collect,
    _artifacts,
    _flow,
    _manifest,
):
    globals().update({k: v for k, v in vars(_module).items() if not k.startswith("__")})

log = logging.getLogger("optimize-submit")

_FACADE_SYNC_MODULES = (
    _config,
    _hf_client,
    _detect,
    _safe_client,
    _records,
    _artifact_paths,
    _delivery,
    _backfill,
    _nfs_collect,
    _artifacts,
    _flow,
    _manifest,
)
_FACADE_SYNC_SKIP = {
    "wait_and_collect_one",
    "process_completion",
}


def _sync_facade_overrides() -> None:
    """Keep legacy monkeypatches on optimize_submit.* visible to implementation modules."""
    facade = globals()
    for module in _FACADE_SYNC_MODULES:
        for name in list(vars(module)):
            if name.startswith("__") or name in _FACADE_SYNC_SKIP or name not in facade:
                continue
            value = facade[name]
            if getattr(module, name) is not value:
                setattr(module, name, value)


def wait_and_collect_one(*args, **kwargs):
    """Compatibility wrapper for tests/importers that monkeypatch facade hooks."""
    _sync_facade_overrides()
    return _artifacts.wait_and_collect_one(*args, **kwargs)


def process_completion(*args, **kwargs):
    """Compatibility wrapper for completion processing through the facade."""
    _sync_facade_overrides()
    return _artifacts.process_completion(*args, **kwargs)


# ── CLI ─────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for the optimize-submit CLI.

    Returns:
        argparse.ArgumentParser: The configured parser with model selection,
            override, SaFE connection, and collection options.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--model", nargs="+", metavar="HF_REPO", help="HuggingFace repo IDs, e.g. Qwen/Qwen3-8B")
    src.add_argument(
        "--hf-top", type=int, metavar="N", help="Auto-select top-N text-gen models from HuggingFace by downloads"
    )
    parser.add_argument(
        "--min-params", type=float, default=0.0, metavar="B", help="Filter HF top-N to models with >=B billion params"
    )

    parser.add_argument("--manual", action="store_true", help="Manual mode: skip auto-detect; --framework is required")
    parser.add_argument("--framework", choices=["sglang", "vllm"], help="Override detected framework")
    parser.add_argument("--precision", choices=["FP8", "FP4", "BF16", "INT4"], help="Override detected precision")
    parser.add_argument(
        "--tp",
        type=int,
        choices=[1, 2, 4, 8, 16, 32],
        help="Override detected tensor parallel size. Values >8 "
        "require --nodes>1 (multi-node RayJob); tp must be "
        "<= nodes*8.",
    )
    parser.add_argument("--concurrency", type=int, help="Override detected concurrency")
    parser.add_argument("--image", help="Override container image")
    parser.add_argument("--isl", type=int, default=1024)
    parser.add_argument("--osl", type=int, default=1024)
    parser.add_argument(
        "--mode",
        choices=["local", "claw"],
        default="local",
        help="Execution mode passed to SaFE (default: local — "
        "agent runs in sandbox directly; 'claw' routes via RayJob)",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=1,
        metavar="N",
        help="Node count for the run. N>1 spreads the model across "
        "an N-node RayJob (8 GPUs/node): forces --mode claw and "
        "injects the RayJob topology (image, per-node resources, "
        "NODES=N, bnxt tar) into the agent prompt — mirrors the "
        "validated ci-config.yaml multi-node entries. Default 1.",
    )
    parser.add_argument(
        "--rayjob-image",
        default="",
        help="Container image for the multi-node RayJob (used only when --nodes>1). Falls back to --image when empty.",
    )

    parser.add_argument("--api-url", default="", help="SaFE base URL (defaults to $SAFE_BASE_URL or $SAFE_API_URL)")
    parser.add_argument("--api-key", default="", help="SaFE bearer token (defaults to $CLAW_API_KEY or $SAFE_API_KEY)")
    parser.add_argument(
        "--register-workspace",
        default="",
        help=f"Workspace where models are registered + downloaded "
        f"(defaults to $SAFE_OPTIMIZE_REGISTER_WORKSPACE "
        f"then '{DEFAULT_REGISTER_WORKSPACE}')",
    )
    parser.add_argument(
        "--submit-workspace",
        default="",
        help=f"Workspace where the optimization task runs "
        f"(defaults to $SAFE_OPTIMIZE_SUBMIT_WORKSPACE "
        f"then '{DEFAULT_SUBMIT_WORKSPACE}'). Used when "
        f"--submit-workspaces is empty.",
    )
    parser.add_argument(
        "--submit-workspaces",
        default="",
        help="Comma-separated list of submit workspaces for "
        "round-robin task distribution (e.g. "
        "'core42-sandbox,core42-hyperloom'). When set, "
        "overrides --submit-workspace and spreads the "
        "batch evenly across the listed workspaces. "
        "Each must independently accept the same model "
        "(register_workspace stays single). Defaults to "
        "$SAFE_OPTIMIZE_SUBMIT_WORKSPACES.",
    )
    parser.add_argument(
        "--workspace",
        default="",
        help="Shorthand: set both --register-workspace and --submit-workspace to the same value (back-compat)",
    )
    parser.add_argument(
        "--volume",
        default="",
        help=f"Wekafs volume mounted RW in --register-workspace "
        f"(defaults to $SAFE_OPTIMIZE_VOLUME then '{DEFAULT_VOLUME}')",
    )
    parser.add_argument(
        "--gpu-type",
        default="",
        help=f"GPU type tag for the prompt (defaults to "
        f"$SAFE_OPTIMIZE_GPU_TYPE then '{DEFAULT_GPU_TYPE}'). "
        f"Known profiles: {', '.join(GPU_PROFILES)}. "
        f"SaFE backend default is MI355X — must override on core42.",
    )
    parser.add_argument(
        "--inferencex-path",
        default="",
        help="Explicit InferenceX checkout path inside the "
        "sandbox (dev override; also $SAFE_OPTIMIZE_"
        "INFERENCEX_PATH). Leave empty (the default) so "
        "install.sh clones a writable per-session copy "
        "instead of pinning a shared read-only mount.",
    )
    parser.add_argument(
        "--oob-path",
        default="",
        help="Optional OOB checkout override inside the sandbox. "
        "Default is unset: sandbox-side install.sh prepares "
        "and exports OOB paths.",
    )
    parser.add_argument(
        "--tracelens-root",
        default="",
        help="TraceLens checkout path inside the sandbox. "
        "Leave empty (the default) so install.sh clones "
        "AMD-AGI/TraceLens into "
        "$HYPERLOOM_RUNTIME_DIR/source-mirrors/TraceLens "
        "and pins it to a fixed SHA. Override via "
        "$SAFE_OPTIMIZE_TRACELENS_ROOT or this flag only "
        "to point at an existing cluster checkout.",
    )
    parser.add_argument(
        "--prompt-prefix",
        default=_load_default_prompt_prefix(),
        help="Free-form prefix prepended to the SaFE-generated "
        "Hyperloom prompt. Default resolves to "
        "$SAFE_OPTIMIZE_PROMPT_PREFIX -> ci/prompt_prefix.txt "
        "-> empty. Pass an empty string explicitly to suppress.",
    )
    parser.add_argument(
        "--prompt-suffix",
        default=os.environ.get("SAFE_OPTIMIZE_PROMPT_SUFFIX", ""),
        help="Optional free-form suffix appended to the SaFE-generated "
        "Hyperloom prompt. (env: $SAFE_OPTIMIZE_PROMPT_SUFFIX)",
    )
    parser.add_argument(
        "--kernel-opt-backends",
        default=os.environ.get("SAFE_OPTIMIZE_KERNEL_BACKENDS", ""),
        help="Comma-separated kernel optimization backends to send "
        "to SaFE's kernelBackends field. Aliases: geak, "
        "claude, codex, cursor. Default: geak,claude,codex "
        "(GEAK first).",
    )
    parser.add_argument(
        "--max-hours",
        type=float,
        default=float(os.environ.get("SAFE_OPTIMIZE_MAX_HOURS", DEFAULT_MAX_HOURS)),
        help="Max hours passed to the Hyperloom optimizer prompt (default: 6).",
    )
    parser.add_argument(
        "--target-gain",
        type=float,
        default=float(os.environ.get("SAFE_OPTIMIZE_TARGET_GAIN", DEFAULT_TARGET_GAIN)),
        help="Target gain %% passed to the Hyperloom optimizer prompt (default: 500).",
    )
    parser.add_argument(
        "--results-path",
        default=os.environ.get("SAFE_OPTIMIZE_RESULTS_PATH", DEFAULT_RESULTS_PATH),
        help="Results path passed to SaFE's prompt builder "
        "(default: $RESULT_DIR so the prompt respects the "
        "CI-selected persistent/ephemeral result root).",
    )
    parser.add_argument(
        "--hf-token", default=os.environ.get("HF_TOKEN", ""), help="HuggingFace token (or set $HF_TOKEN)"
    )
    parser.add_argument(
        "--hf-token-2",
        default=os.environ.get("HF_TOKEN_2", ""),
        help="Secondary HuggingFace token to alternate with on 429 (or set $HF_TOKEN_2)",
    )

    # Production-pool audit metadata: copied into submission_manifest.json (does
    # not affect submission) to trace which pool entry a task reran.
    parser.add_argument("--pool-id", default=os.environ.get("HYPERLOOM_POOL_ID", ""))
    parser.add_argument("--pool-index", default=os.environ.get("HYPERLOOM_POOL_INDEX", ""))
    parser.add_argument("--pool-batch-index", default=os.environ.get("HYPERLOOM_POOL_BATCH_INDEX", ""))
    parser.add_argument("--pool-batch-size", default=os.environ.get("HYPERLOOM_POOL_BATCH_SIZE", ""))
    parser.add_argument("--pool-source-task-id", default=os.environ.get("HYPERLOOM_POOL_SOURCE_TASK_ID", ""))
    # HF download count for this model (carried from the candidates pool via the
    # matrix). When it parses as an int < 100, the submit pins the orchestration
    # model to claude-opus-4-6 via session env (CLAUDE_MODEL). Missing / non-int
    # leaves the model unset (no-op).
    parser.add_argument("--downloads", default=os.environ.get("HYPERLOOM_DOWNLOADS", ""))

    parser.add_argument(
        "--session-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Session-scoped env var forwarded to SaFE (body.env) and relayed by "
        "Claw into the sandbox process environment (the reliable channel the "
        "inference_optimizer process actually reads — unlike a shell export in "
        "--prompt-prefix, which depends on unverified cross-tool-call shell "
        "persistence). Repeatable. e.g. "
        "--session-env FRAMEWORK_AGENT_CROSS_DISCOVER_TAG=1",
    )

    parser.add_argument(
        "--dry-run", action="store_true", help="Auto-detect and print plan without registering or submitting"
    )
    parser.add_argument(
        "--output-dir", default="", help="Write submission_manifest.{json,md} to this dir (for CI artifacts)"
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    # Post-submission: wait for tasks + collect artifacts.
    wait_group = parser.add_mutually_exclusive_group()
    wait_group.add_argument(
        "--wait-for-completion",
        dest="wait_for_completion",
        action="store_true",
        default=True,
        help="(default) After submitting, poll each task until it reaches Succeeded/Failed/Interrupted/Timeout",
    )
    wait_group.add_argument(
        "--no-wait-for-completion",
        dest="wait_for_completion",
        action="store_false",
        help="Fire-and-forget: exit immediately after submitting",
    )

    collect_group = parser.add_mutually_exclusive_group()
    collect_group.add_argument(
        "--collect-artifacts",
        dest="collect_artifacts",
        action="store_true",
        default=True,
        help="(default) Download each finished task's artifacts to --artifacts-dir (implies --wait-for-completion)",
    )
    collect_group.add_argument(
        "--no-collect-artifacts",
        dest="collect_artifacts",
        action="store_false",
        help="Skip artifact download even when waiting",
    )

    parser.add_argument(
        "--all-artifacts",
        action="store_true",
        help=f"Download every artifact (default keeps only files matching {', '.join(DEFAULT_ARTIFACT_PATTERNS)})",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="task-artifacts",
        help="Local directory where per-task artifacts land (default: ./task-artifacts)",
    )
    parser.add_argument(
        "--task-timeout-min", type=int, default=420, help="Per-task wait timeout in minutes (default: 420 = 7h)"
    )
    parser.add_argument(
        "--poll-interval-s", type=int, default=60, help="How often to poll task status, seconds (default: 60)"
    )
    parser.add_argument(
        "--wait-parallel", type=int, default=8, help="How many tasks to wait for in parallel (default: 8)"
    )
    parser.add_argument(
        "--submit-jitter-sec",
        type=int,
        default=int(os.environ.get("SAFE_OPTIMIZE_SUBMIT_JITTER_SEC", "0") or 0),
        help="Pre-submit random delay window in seconds. Each (parallel matrix) "
        "job sleeps random(0..N) before touching SaFE so register/submit "
        "calls de-sync instead of stampeding Claw-session creation all at "
        "once (the thundering herd that the backend answers with HTTP 500 "
        "'failed to create Claw session' / 504). 0 = off (default).",
    )
    return parser


def main() -> int:
    """CLI entry point: register, submit, wait/collect, and write the manifest.

    Parses arguments, resolves the model set and SaFE connection, submits
    optimization tasks, optionally waits for completion and collects artifacts,
    and writes the submission manifest.

    Returns:
        int: Process exit code (0 on success, non-zero on fatal errors).
    """
    args = _build_parser().parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    base_url = args.api_url or os.environ.get("SAFE_BASE_URL") or os.environ.get("SAFE_API_URL") or DEFAULT_API_URL
    api_key = args.api_key or os.environ.get("CLAW_API_KEY") or os.environ.get("SAFE_API_KEY") or ""
    # --workspace shorthand sets both to the same value (back-compat); explicit
    # --register-workspace / --submit-workspace override it.
    shared_ws = args.workspace or os.environ.get("SAFE_OPTIMIZE_WORKSPACE") or ""
    register_workspace = (
        args.register_workspace
        or os.environ.get("SAFE_OPTIMIZE_REGISTER_WORKSPACE")
        or shared_ws
        or DEFAULT_REGISTER_WORKSPACE
    )
    submit_workspace = (
        args.submit_workspace
        or os.environ.get("SAFE_OPTIMIZE_SUBMIT_WORKSPACE")
        or shared_ws
        or DEFAULT_SUBMIT_WORKSPACE
    )
    # Round-robin pool: --submit-workspaces overrides single submit_workspace;
    # empty -> single-workspace mode.
    submit_workspaces_raw = args.submit_workspaces or os.environ.get("SAFE_OPTIMIZE_SUBMIT_WORKSPACES") or ""
    submit_workspaces_pool = [w.strip() for w in submit_workspaces_raw.split(",") if w and w.strip()]
    volume = args.volume or os.environ.get("SAFE_OPTIMIZE_VOLUME") or DEFAULT_VOLUME
    gpu_type_input = args.gpu_type or os.environ.get("SAFE_OPTIMIZE_GPU_TYPE") or DEFAULT_GPU_TYPE
    gpu_type = canonical_gpu_type(gpu_type_input)
    gpu_profile = normalize_gpu_profile(gpu_type, warn=False) or DEFAULT_GPU_PROFILE
    # Unset by default (install.sh clones a writable per-session copy); only an
    # explicit path pins one. Empty -> inferencexPath="" suppresses SaFE's default.
    inferencex_path = args.inferencex_path or os.environ.get("SAFE_OPTIMIZE_INFERENCEX_PATH") or ""
    oob_path = args.oob_path or os.environ.get("SAFE_OPTIMIZE_OOB_PATH") or ""
    tracelens_root = args.tracelens_root or os.environ.get("SAFE_OPTIMIZE_TRACELENS_ROOT", "")
    try:
        kernel_backends = parse_kernel_backends(args.kernel_opt_backends)
    except ValueError as e:
        log.error("%s", e)
        return 2

    if not api_key and not args.dry_run:
        log.error("no API key set (CLAW_API_KEY / SAFE_API_KEY / --api-key)")
        return 2

    log.info(
        "SaFE base_url=%s register_workspace=%s submit_workspace=%s volume=%s",
        base_url,
        register_workspace,
        submit_workspace,
        volume,
    )
    log.info(
        "Cluster prompt fields: gpu_type=%s gpu_profile=%s inferencex_path=%s oob_path=%s tracelens_root=%s",
        gpu_type,
        gpu_profile,
        inferencex_path,
        oob_path,
        tracelens_root,
    )
    log.info("Kernel backends: %s", ", ".join(kernel_backends))
    if submit_workspaces_pool:
        log.info("submit round-robin pool: %s (overrides --submit-workspace)", ",".join(submit_workspaces_pool))
    if register_workspace != submit_workspace and not submit_workspaces_pool:
        log.info(
            "cross-workspace mode — needs SaFE selectLocalPath path-accessible "
            "fallback to be deployed; will 400 on submit_task otherwise"
        )

    hf_seed = ",".join(getattr(args, "model", None) or []) or os.environ.get("GITHUB_RUN_ID", "")
    hf = HuggingFaceClient(args.hf_token, tokens=[args.hf_token_2], seed=hf_seed)
    if args.hf_token_2:
        log.info("HF token pool: 2 tokens (alternate on 429)")
    # Dry-run never hits SaFE, so a placeholder token is fine.
    safe = SafeOptimizeClient(
        base_url,
        api_key or "dry-run",
        register_workspace=register_workspace,
        submit_workspace=submit_workspace,
        volume=volume,
        submit_workspaces_pool=submit_workspaces_pool or None,
    )
    if submit_workspaces_pool and args.pool_index:
        try:
            safe._submit_ws_counter = max(int(args.pool_index), 0)
            log.info(
                "submit round-robin offset seeded from pool_index=%s",
                args.pool_index,
            )
        except ValueError:
            log.warning("invalid pool_index=%r; round-robin starts at 0", args.pool_index)

    if args.hf_top:
        log.info("fetching HF top-%d (>=%.1fB)", args.hf_top, args.min_params)
        try:
            repos = hf.top_models(args.hf_top, min_params_b=args.min_params)
        except Exception as e:
            log.error("HF top-N fetch failed: %s", e)
            return 1
        log.info("selected %d models: %s", len(repos), repos)
    else:
        repos = list(args.model or [])

    if not repos:
        log.error("no models to process")
        return 1

    overrides = {
        "framework": args.framework,
        "precision": args.precision,
        "tp": args.tp,
        "concurrency": args.concurrency,
        "image": args.image,
    }
    pool_metadata = {
        "pool_id": args.pool_id,
        "pool_index": args.pool_index,
        "batch_index": args.pool_batch_index,
        "batch_size": args.pool_batch_size,
        "source_task_id": args.pool_source_task_id,
    }

    # ── Multi-node resolution ──────────────────────────────────────────────
    # --nodes>1 spans an N-node RayJob. The SaFE task body has no node count, so
    # force mode=claw and append the RayJob topology to the prompt suffix
    # (mirrors the Claw-direct CI). --nodes is global but effectively per-model.
    effective_mode = args.mode
    effective_prompt_suffix = args.prompt_suffix or None
    _nodes = args.nodes or 1
    # tp must fit nodes*8 GPUs (enforced for single-node too) so a stray --tp 16
    # is rejected at submit time, not at runtime on an 8-GPU sandbox.
    if args.tp and args.tp > _nodes * 8:
        log.error(
            "--tp %d exceeds --nodes %d * 8 GPUs = %d; lower --tp or raise --nodes (tp>8 requires multi-node).",
            args.tp,
            _nodes,
            _nodes * 8,
        )
        return 2
    if _nodes > 1:
        if effective_mode != "claw":
            log.warning("--nodes %d > 1 needs RayJob fan-out; forcing --mode claw (was %r)", _nodes, effective_mode)
            effective_mode = "claw"
        rayjob_image = (args.rayjob_image or args.image or "").strip()
        if not rayjob_image:
            log.warning(
                "--nodes %d > 1 but no --rayjob-image/--image set; the agent must pick a RayJob image itself",
                args.nodes,
            )
        effective_prompt_suffix = (
            (args.prompt_suffix or "") + _multinode_prompt_suffix(args.nodes, rayjob_image)
        ) or None
        log.info(
            "multi-node: nodes=%d tp=%s mode=%s rayjob_image=%s",
            args.nodes,
            args.tp,
            effective_mode,
            rayjob_image or "(agent-chosen)",
        )

    # Pre-submit jitter. When a large matrix fans out, every job otherwise hits
    # register/submit (and Claw-session creation) in the same instant — the
    # backend then sheds load with HTTP 500 "failed to create Claw session" /
    # 504. A per-process random(0..N) sleep here spreads the herd across an
    # N-second window so the backend sees a trickle rather than a spike. The
    # submit_task retry loop still backstops any residual collision.
    jitter = max(0, args.submit_jitter_sec)
    if jitter > 0 and not args.dry_run:
        d = random.uniform(0, jitter)
        log.info(
            "submit jitter: sleeping %.1fs (window 0-%ds) to de-sync from other parallel jobs before hitting SaFE",
            d,
            jitter,
        )
        time.sleep(d)

    # Parse --session-env KEY=VALUE pairs once, up front, so a malformed pair
    # fails fast before any task is submitted (rather than silently dropped).
    cli_session_env: dict[str, str] = {}
    for pair in args.session_env or []:
        if "=" not in pair:
            log.error("invalid --session-env %r: expected KEY=VALUE", pair)
            return 2
        key, value = pair.split("=", 1)
        key = key.strip()
        if not key:
            log.error("invalid --session-env %r: empty key", pair)
            return 2
        cli_session_env[key] = value
    if cli_session_env:
        log.info("session env from --session-env: %s", cli_session_env)

    records: list[SubmissionRecord] = []
    for repo in repos:
        log.info("=" * 60)
        log.info("Model: %s", repo)
        # Low-download models (HF downloads < 100) pin the orchestration model
        # to claude-opus-4-6 via session env. A missing or non-integer downloads
        # value is a no-op (env stays None).
        submit_env: dict | None = None
        try:
            if int(str(args.downloads).strip()) < 100:
                submit_env = {"CLAUDE_MODEL": "claude-opus-4-6"}
        except (TypeError, ValueError):
            submit_env = None
        # Merge explicit --session-env KEY=VALUE pairs (reliable session_env
        # channel). These win over the CLAUDE_MODEL default on key collision.
        if cli_session_env:
            submit_env = {**(submit_env or {}), **cli_session_env}
        rec = process_model(
            repo,
            hf,
            safe,
            overrides,
            args.isl,
            args.osl,
            args.dry_run,
            args.hf_token,
            manual_mode=args.manual,
            mode=effective_mode,
            gpu_type=gpu_type,
            inferencex_path=inferencex_path,
            oob_path=oob_path,
            tracelens_root=tracelens_root,
            prompt_prefix=args.prompt_prefix or None,
            prompt_suffix=effective_prompt_suffix,
            kernel_backends=kernel_backends,
            max_hours=args.max_hours,
            target_gain=args.target_gain,
            results_path=args.results_path,
            pool_metadata=pool_metadata,
            env=submit_env,
        )
        records.append(rec)

    submitted = sum(1 for r in records if r.status == "submitted")
    submit_failed = [r for r in records if r.status == "failed"]
    log.info("=" * 60)
    log.info("Submitted: %d ok, %d failed, %d total", submitted, len(submit_failed), len(records))
    for r in submit_failed:
        log.warning("  submit failed: %s — %s", r.model, r.error)

    # Wait + collect (default on); skip on dry-run.
    if not args.dry_run and submitted > 0 and args.wait_for_completion:
        process_completion(
            safe,
            records,
            artifacts_dir=Path(args.artifacts_dir),
            task_timeout_min=args.task_timeout_min,
            poll_s=args.poll_interval_s,
            collect=args.collect_artifacts,
            all_artifacts=args.all_artifacts,
            parallel=args.wait_parallel,
        )

        from collections import Counter

        final_counts = Counter(r.final_status or "Pending" for r in records if r.task_id)
        delivery_counts = Counter(r.ci_status or "Unknown" for r in records if r.task_id)
        log.info("=" * 60)
        log.info("Final task statuses: %s", ", ".join(f"{k}={v}" for k, v in sorted(final_counts.items())))
        log.info("CI delivery statuses: %s", ", ".join(f"{k}={v}" for k, v in sorted(delivery_counts.items())))
        delivered_non_success = [r for r in records if r.task_id and r.final_status != "Succeeded" and r.ci_success]
        if delivered_non_success:
            log.warning(
                "SaFE terminal status was non-success, but CI artifacts were delivered: %s",
                ", ".join(f"{r.model}:{r.final_status or 'Pending'}->{r.ci_status}" for r in delivered_non_success),
            )
        non_success = [r for r in records if r.task_id and r.final_status != "Succeeded" and not r.ci_success]
    else:
        non_success = []

    # Manifest written after wait/collect so it captures final_status etc.
    if args.output_dir:
        write_manifest(Path(args.output_dir), records, base_url, register_workspace, submit_workspace, volume)

    if args.dry_run:
        return 0
    if non_success:
        log.error(
            "Non-success terminal statuses without deliverable artifacts: %s",
            ", ".join(f"{r.model}:{r.final_status or 'Pending'}:{r.ci_status or 'Unknown'}" for r in non_success),
        )
        return 2
    context_skipped = [r for r in records if r.status == "skipped" and (r.error or "").startswith("context_too_short:")]
    if submitted == 0 and records and len(context_skipped) == len(records):
        log.info("All models skipped by policy: context_too_short")
        return 0
    return 0 if submitted > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
