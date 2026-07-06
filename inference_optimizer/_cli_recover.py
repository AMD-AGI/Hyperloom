# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""CLI entry — ``optimize`` subcommand wiring Claude+Codex backends, executors, objective, and Coordinator.run().

Env vars consumed: MODEL_PATH, OPENAI_BASE_URL + SAFE_API_KEY, ROCR_VISIBLE_DEVICES,
CLAUDE_MODEL, CODEX_MODEL, USER_DATA_PATH.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

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

log = logging.getLogger(__name__)

def _session_recovery_status(session_dir: Path) -> dict[str, Any]:
    """Inspect on-disk artifacts to judge whether a session finished cleanly.

    Pure read of state.json / session_breakdown.json / langfuse_receipt.json.
    Returns flags used by :func:`_run_recover_session` to decide whether the
    session still needs a (re)build + Langfuse push.

    Args:
        session_dir (Path): The session directory to inspect.

    Returns:
        dict[str, Any]: A status mapping with ``close_done``,
            ``breakdown_exists``, ``breakdown_recorded``, ``counts_final``,
            and ``looks_complete`` flags.
    """

    from .breakdown import BREAKDOWN_FILENAME

    state_path = session_dir / "state.json"
    close_done = False
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            close_done = bool((state or {}).get("close_sequence_done"))
        except (json.JSONDecodeError, OSError):
            pass

    breakdown_exists = (session_dir / BREAKDOWN_FILENAME).exists()

    from .orchestrator.trace.langfuse_emitter import read_receipt

    receipt = read_receipt(session_dir) or {}
    counts = receipt.get("counts") or {}
    breakdown_recorded = bool(counts.get("breakdown_recorded"))
    counts_final = bool(receipt.get("counts_final"))

    return {
        "close_done": close_done,
        "breakdown_exists": breakdown_exists,
        "breakdown_recorded": breakdown_recorded,
        "counts_final": counts_final,
        "looks_complete": close_done and breakdown_recorded,
    }

def _run_recover_session(args: argparse.Namespace) -> int:
    """Offline recovery for a session that exited abnormally.

    Rebuilds ``session_breakdown.json`` from the crash-time recorder fragments
    (the merge step), reconciles + flushes Langfuse, splices the post-flush
    receipt into the breakdown, and attaches the full breakdown JSON to the
    session's trace. Idempotent across processes (guarded by the persisted
    Langfuse receipt), so re-running is safe.

    Args:
        args (argparse.Namespace): The parsed CLI namespace (reads
            ``session_dir``, ``force``, and ``backfill_trace``).

    Returns:
        int: The process exit code (``0`` on success, ``2`` when the session
            dir is missing, ``1`` on breakdown rebuild failure).
    """
    session_dir = args.session_dir.resolve()
    if not session_dir.is_dir():
        print(f"ERROR: session dir not found: {session_dir}", file=sys.stderr)
        return 2

    status = _session_recovery_status(session_dir)
    print(
        f"recover-session   : {session_dir}\n"
        f"  close_sequence_done={status['close_done']} "
        f"breakdown_exists={status['breakdown_exists']} "
        f"breakdown_recorded={status['breakdown_recorded']} "
        f"counts_final={status['counts_final']}"
    )
    if status["looks_complete"] and not args.force:
        print("  -> already complete (breakdown built and recorded to Langfuse); pass --force to rebuild anyway.")
        return 0

    # 1) Rebuild/merge the breakdown from whatever fragments survived the crash.
    try:
        from .breakdown import write_breakdown_json

        breakdown_path = write_breakdown_json(session_dir)
        print(f"  rebuilt breakdown : {breakdown_path}")
    except Exception:  # noqa: BLE001
        log.exception("recover-session: breakdown rebuild failed")
        return 1

    # 2) Reconcile + flush Langfuse, splice the final receipt, attach the SBD.
    try:
        from .breakdown import patch_breakdown_langfuse
        from .orchestrator.trace.langfuse_emitter import (
            flush_session,
            record_session_breakdown,
        )

        flush_session(session_dir)
        patch_breakdown_langfuse(session_dir)
        record_session_breakdown(session_dir)
        print("  langfuse          : flushed + breakdown attached")
    except Exception:  # noqa: BLE001
        log.exception("recover-session: langfuse push failed (non-fatal)")

    # 3) Optional full generation replay (off by default; duplicates if the
    #    live emitter already ran for this session).
    if args.backfill_trace:
        try:
            from .scripts.backfill_langfuse import build_plan, ingest

            rc = ingest(build_plan(session_dir))
            print(f"  trace backfill    : rc={rc}")
        except Exception:  # noqa: BLE001
            log.exception("recover-session: trace backfill failed (non-fatal)")

    # 4) Re-package the artifact bundle so /workspace carries the recovered SBD.
    try:
        from .breakdown import package_session_artifacts

        pkg_path = package_session_artifacts(session_dir)
        if pkg_path is not None:
            print(f"  artifact package  : {pkg_path}")
    except Exception:  # noqa: BLE001
        log.exception("recover-session: artifact package failed (non-fatal)")

    return 0
