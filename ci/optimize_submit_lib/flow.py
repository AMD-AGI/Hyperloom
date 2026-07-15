from __future__ import annotations

import json
import logging
import os
import random
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests

log = logging.getLogger("optimize-submit")

from . import config as _config

globals().update({k: v for k, v in vars(_config).items() if not k.startswith("__")})

from . import hf_client as _hf_client

globals().update({k: v for k, v in vars(_hf_client).items() if not k.startswith("__")})

import model_compat  # local ci sibling module
from . import safe_client as _safe_client
from . import records as _records
from . import artifacts as _artifacts

globals().update({k: v for k, v in vars(_safe_client).items() if not k.startswith("__")})
globals().update({k: v for k, v in vars(_records).items() if not k.startswith("__")})
globals().update({k: v for k, v in vars(_artifacts).items() if not k.startswith("__")})

# ── Per-model flow ──────────────────────────────────────────────────────────────


def process_model(
    repo_id: str,
    hf: HuggingFaceClient,
    safe: SafeOptimizeClient,
    overrides: dict,
    isl: int,
    osl: int,
    dry_run: bool,
    hf_token: str,
    manual_mode: bool,
    mode: str,
    gpu_type: str | None = None,
    inferencex_path: str | None = None,
    oob_path: str | None = None,
    tracelens_root: str | None = None,
    prompt_prefix: str | None = None,
    prompt_suffix: str | None = None,
    kernel_backends: list[str] | None = None,
    max_hours: float | None = None,
    target_gain: float | None = None,
    results_path: str | None = None,
    pool_metadata: dict | None = None,
    env: dict | None = None,
) -> SubmissionRecord:
    """Run the full submit flow for one model: detect, register, submit.

    Auto-detects (or uses manual overrides for) the launch config, ensures the
    model is registered and Ready in SaFE (preferring prewarmed local_path
    mode), then submits the optimization task. Short-circuits for dry-run and
    for skip/failure conditions.

    Args:
        repo_id (str): HuggingFace repo id.
        hf (HuggingFaceClient): Client for HF metadata lookups.
        safe (SafeOptimizeClient): Client for SaFE register/submit calls.
        overrides (dict): Field overrides (framework/precision/tp/...).
        isl (int): Input sequence length.
        osl (int): Output sequence length.
        dry_run (bool): When True, plan only without registering/submitting.
        hf_token (str): HuggingFace token for gated downloads.
        manual_mode (bool): Skip auto-detect; requires ``framework`` override.
        mode (str): Execution mode passed to SaFE (``local`` / ``claw``).
        gpu_type (str | None): GPU type override for the prompt.
        inferencex_path (str | None): InferenceX checkout path override.
        oob_path (str | None): OOB checkout path override.
        tracelens_root (str | None): TraceLens checkout path override.
        prompt_prefix (str | None): Prompt prefix forwarded to SaFE.
        prompt_suffix (str | None): Prompt suffix forwarded to SaFE.
        kernel_backends (list[str] | None): Kernel optimization backends.
        max_hours (float | None): Max optimization wall-clock hours.
        target_gain (float | None): Target gain percentage.
        results_path (str | None): Results path passed to the prompt builder.
        pool_metadata (dict | None): Production-pool audit metadata.

    Returns:
        SubmissionRecord: The record describing the outcome of this model.
    """
    rec = SubmissionRecord(
        model=repo_id,
        overrides={k: v for k, v in overrides.items() if v is not None},
        pool={k: v for k, v in (pool_metadata or {}).items() if v not in (None, "")},
    )
    gpu_type = canonical_gpu_type(gpu_type)
    gpu_profile = normalize_gpu_profile(gpu_type, warn=False) or DEFAULT_GPU_PROFILE

    detected = None if manual_mode else auto_detect(hf, repo_id, gpu_type=gpu_type)
    if not detected and not manual_mode:
        rec.status = "skipped"
        rec.error = "auto-detect failed"
        return rec
    if manual_mode and not overrides.get("framework"):
        rec.status = "skipped"
        rec.error = "manual mode requires --framework"
        return rec
    if detected:
        rec.detected = asdict(detected)
        rec.category = _category_from_arch(rec.detected.get("arch", ""))
        # Absolute small-context floor: skip models whose config.json
        # max_position_embeddings is present and <= MIN_MAX_POSITION_EMBEDDINGS.
        mpe = int(rec.detected.get("max_position_embeddings") or 0)
        if mpe and mpe <= MIN_MAX_POSITION_EMBEDDINGS:
            rec.status = "skipped"
            rec.error = (
                "context_too_short: "
                f"max_position_embeddings={mpe} <= {MIN_MAX_POSITION_EMBEDDINGS}"
            )
            log.warning("[%s] skipping: %s", repo_id, rec.error)
            return rec
        max_context = int(rec.detected.get("max_context_tokens") or 0)
        if context_too_short(max_context, isl, osl):
            required = isl + osl + DEFAULT_CONTEXT_RESERVE_TOKENS
            rec.status = "skipped"
            rec.error = (
                "context_too_short: "
                f"max_context_tokens={max_context} < required={required} "
                f"(isl={isl}, osl={osl}, reserve={DEFAULT_CONTEXT_RESERVE_TOKENS})"
            )
            log.warning("[%s] skipping: %s", repo_id, rec.error)
            return rec

        # Shared structural compatibility pre-flight, using the downloaded config
        # + local model dir so doomed models are skipped before a Claw session.
        compat = model_compat.unrunnable_reason(
            detected.raw_config,
            repo=repo_id,
            model_dir=os.path.join(
                os.environ.get("CI_MODELS_DIR", "/wekafs/models"),
                repo_id.replace("/", "-")),
            whitelist=model_compat.load_whitelist(),
            gpu_type=gpu_type,
        )
        if compat:
            reason, detail = compat
            rec.status = "skipped"
            rec.error = f"{reason}: {detail}"
            log.warning(
                "[%s] PRE-FLIGHT FILTERED — repo_id=%s rule=%s reason=%s "
                "arch=%s params=%.1fB max_position_embeddings=%s framework=%s — "
                "skipping: NOT submitting, no Claw session created",
                repo_id,
                repo_id,
                reason,
                detail,
                detected.arch,
                detected.params_b,
                detected.max_position_embeddings or "?",
                detected.framework,
            )
            return rec

        # Online fallback for missing_tokenizer: probe the HF file listing so
        # not-yet-cached repos lacking tokenizer.* files are skipped before a
        # Claw session is created.
        hf_tokens = [t for t in (os.environ.get("HF_TOKEN", ""),
                                  os.environ.get("HF_TOKEN_2", "")) if t]
        tok_reason = model_compat.hf_missing_tokenizer(repo_id, hf_tokens)
        if tok_reason:
            rec.status = "skipped"
            rec.error = f"{tok_reason}: weights present on HF but no tokenizer files"
            log.warning(
                "[%s] PRE-FLIGHT FILTERED — rule=%s — skipping: NOT submitting, "
                "no Claw session created", repo_id, tok_reason)
            return rec

    framework = overrides.get("framework") or (detected.framework if detected else "")
    precision = overrides.get("precision") or (detected.precision if detected else "FP8")
    tp = overrides.get("tp") or (detected.tp if detected else 1)
    conc = overrides.get("concurrency") or (detected.concurrency if detected else 64)
    image = overrides.get("image") or (detected.image if detected else detect_image(framework, repo_id))

    log.info(
        "[%s] => mode=%s framework=%s precision=%s tp=%d conc=%d image=%s",
        repo_id,
        mode,
        framework,
        precision,
        tp,
        conc,
        image,
    )

    display_name = f"{repo_id.split('/')[-1]}-{precision.lower()}-{framework}-{gpu_profile}"
    rec.display_name = display_name
    rec.overrides["mode"] = mode
    if kernel_backends:
        rec.overrides["kernel_backends"] = kernel_backends
    if max_hours:
        rec.overrides["max_hours"] = max_hours
    if target_gain:
        rec.overrides["target_gain"] = target_gain
    if results_path:
        rec.overrides["results_path"] = results_path

    if dry_run:
        rec.status = "dry-run"
        return rec

    # If prewarm already populated /wekafs/models/<slug>/, use local_path mode so
    # SaFE sets phase=Ready immediately without re-downloading.
    nfs_root = os.environ.get("NFS_ROOT", "/wekafs")
    target_slug = repo_id.replace("/", "-")
    target_dir = f"{nfs_root}/models/{target_slug}"
    use_local_path = False
    try:
        if os.path.isdir(target_dir):
            # Heuristic floor: any real HF repo has >=5 files.
            n_files = sum(1 for _ in os.scandir(target_dir))
            if n_files >= 5:
                use_local_path = True
                log.info(
                    "[%s] prewarm complete (%d files at %s) — registering "
                    "via local_path mode (skips SaFE Download Job)",
                    repo_id,
                    n_files,
                    target_dir,
                )
            else:
                log.info(
                    "[%s] %s has only %d entries — falling back to SaFE download via accessMode=local",
                    repo_id,
                    target_dir,
                    n_files,
                )
    except OSError as e:
        log.warning("[%s] could not probe %s: %s — falling back to SaFE download", repo_id, target_dir, e)

    # Find existing SaFE record OR register fresh. Stale phase=Failed records are
    # re-registered so the Download Job re-runs against prewarmed files.
    safe_model = safe.find_model(repo_id)
    if safe_model and safe_model.get("phase") != "Failed":
        model_id = safe_model["id"]
        phase = safe_model.get("phase", "")
        log.info("[%s] found in SaFE: id=%s phase=%s", repo_id, model_id, phase)
        if phase != "Ready" and not safe.wait_ready(model_id):
            rec.status = "failed"
            rec.error = "model never reached Ready"
            return rec
    else:
        if safe_model:
            log.info(
                "[%s] existing model %s is %s — re-registering (prewarm should have populated /wekafs/models/ already)",
                repo_id,
                safe_model.get("id"),
                safe_model.get("phase"),
            )
        try:
            model_id = safe.register_model(
                repo_id,
                hf_token,
                local_path=target_dir if use_local_path else "",
            )
        except Exception as e:
            rec.status = "failed"
            rec.error = f"register: {e}"
            return rec
        if not model_id:
            rec.status = "failed"
            rec.error = "register returned empty id"
            return rec
        if safe_model and model_id == safe_model.get("id"):
            log.warning(
                "[%s] SaFE returned the same id %s as the existing "
                "Failed record — backend deduped by sourceURL and did "
                "not reset phase. DELETE the record manually and rerun.",
                repo_id,
                model_id,
            )
        if not safe.wait_ready(model_id):
            rec.status = "failed"
            rec.error = "model never reached Ready"
            return rec

    try:
        task_prompt_prefix = prompt_prefix
        result = safe.submit_task(
            model_id,
            display_name,
            framework,
            precision,
            tp,
            conc,
            isl,
            osl,
            image,
            mode=mode,
            gpu_type=gpu_type,
            inferencex_path=inferencex_path,
            oob_path=oob_path,
            tracelens_root=tracelens_root,
            prompt_prefix=task_prompt_prefix,
            prompt_suffix=prompt_suffix,
            kernel_backends=kernel_backends,
            max_hours=max_hours,
            target_gain=target_gain,
            results_path=results_path,
            env=env,
        )
    except Exception as e:
        rec.status = "failed"
        rec.error = f"submit_task: {e}"
        return rec

    rec.status = "submitted"
    rec.task_id = result.get("id", "?")
    log.info("[%s] OK — task_id=%s display=%s", repo_id, rec.task_id, display_name)
    return rec
