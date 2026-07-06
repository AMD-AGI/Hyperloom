# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""CLI entry — ``optimize`` subcommand wiring Claude+Codex backends, executors, objective, and Coordinator.run().

Env vars consumed: MODEL_PATH, OPENAI_BASE_URL + SAFE_API_KEY, ROCR_VISIBLE_DEVICES,
CLAUDE_MODEL, CODEX_MODEL, USER_DATA_PATH.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
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

log = logging.getLogger(__name__)

def _quantization_enabled_via_env() -> bool:
    """Return ``True`` iff the deterministic quantization master switch is on.

    Quantization is gated on ``$HYPERLOOM_QUANTIZE_ENABLED`` (truthy = ``1`` /
    ``true`` / ``yes`` / ``on``, case-insensitive). This makes the on/off
    decision a deterministic, frontend-settable env flag rather than something
    an LLM agent infers from natural language. Anything else — including unset —
    disables quantization.

    Returns:
        ``True`` when the env var is set to a recognized truthy value.
    """
    return os.environ.get("HYPERLOOM_QUANTIZE_ENABLED", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

async def _run_quantization_prelude(args: argparse.Namespace) -> None:
    """Run the quantization-agent once before the optimization loop.

    No-op unless ``--quantize "<prompt>"`` was passed. When set, this drives
    AMD Quark PTQ from the prompt via the ``quantization_request_handlers``
    adapter, then rewrites ``args.model`` (+ ``$MODEL_PATH``) to the exported
    quantized model so every downstream phase (baseline / profile / sweep /
    kernel) optimizes the quantized model instead of the source.

    Contract:
      * Skipped on ``--resume`` (a resumed session already has its model
        pinned in the manifest; re-quantizing would diverge from it).
      * On a failed/blocked quantization the process exits with code 3 —
        we must not silently fall through and optimize the un-quantized
        source model when the user explicitly asked for quantization.
      * On a scheme/GPU mismatch (e.g. an MI355X-only scheme on an mi300x
        target), the structured ``--quantize-scheme`` path reports the error
        and *skips* quantization, then continues optimizing the un-quantized
        model. The mismatch is a config error caught before any Quark work
        runs, not a mid-run failure, so the run proceeds rather than aborting.
        The skip is made **detectable** so a launcher / UI never mistakes the
        run for quantized: a ``QUANTIZATION_SKIPPED:`` marker line on stdout
        plus the ``$HYPERLOOM_QUANTIZATION_SKIPPED`` env var (set to the reason).

    Args:
        args: Parsed CLI arguments; reads ``quantize`` / ``quantize_scheme`` /
            ``gpu_type`` / ``resume`` and rewrites ``args.model`` in place to
            the exported quantized model path on success.
    """
    # Free-text --quantize wins; otherwise resolve the structured
    # --quantize-scheme enum (the UI/backend path) to a prompt.
    prompt = getattr(args, "quantize", None)
    if not prompt:
        from hyperloom.orchestrator.quantization_schemes import (
            SchemeNotSupportedError,
            resolve_scheme_prompt,
            validate_scheme,
        )

        scheme = getattr(args, "quantize_scheme", None)
        # Constrain the scheme by the target GPU. The real GPU is probed later;
        # use the --gpu-type / $GPU_TYPE hint here (empty => no enforcement).
        gpu_hint = (getattr(args, "gpu_type", None) or os.environ.get("GPU_TYPE", "")).strip().lower()
        try:
            validate_scheme(scheme, gpu_hint)
        except SchemeNotSupportedError as exc:
            # Pre-flight config error (caught before any Quark work): per the
            # documented contract we SKIP quantization and continue on the
            # un-quantized model rather than hard-stopping. Make the skip
            # explicit + machine-detectable (stdout marker + env var) so a
            # launcher / UI surfaces "requested quantization was skipped"
            # instead of silently believing the run is quantized.
            reason = str(exc)
            os.environ["HYPERLOOM_QUANTIZATION_SKIPPED"] = reason
            print(
                f"QUANTIZATION_SKIPPED: {reason}; continuing optimization on the "
                "un-quantized model. Pick a scheme supported by this GPU TYPE "
                "(or change GPU_TYPE) to actually quantize."
            )
            print(f"ERROR: quantization skipped — {reason}", file=sys.stderr)
            return
        prompt = resolve_scheme_prompt(scheme)
    if not prompt:
        return
    if getattr(args, "resume", False):
        print("Quantization prelude: skipped (--resume); using model from manifest.")
        return

    # Deterministic master switch: quantization runs ONLY when
    # $HYPERLOOM_QUANTIZE_ENABLED is explicitly truthy. This decouples the
    # on/off decision from any agent's natural-language judgement — even if a
    # --quantize / --quantize-scheme flag reached us (e.g. an in-sandbox agent
    # added it from the prompt), we refuse to quantize unless the env switch is
    # on. Absent / false => skip and continue on the un-quantized model, made
    # detectable via the QUANTIZATION_SKIPPED marker + $HYPERLOOM_QUANTIZATION_SKIPPED.
    if not _quantization_enabled_via_env():
        reason = "HYPERLOOM_QUANTIZE_ENABLED is not set to a truthy value"
        os.environ["HYPERLOOM_QUANTIZATION_SKIPPED"] = reason
        print(
            f"QUANTIZATION_SKIPPED: {reason}; continuing optimization on the "
            "un-quantized model. Set HYPERLOOM_QUANTIZE_ENABLED=1 to quantize."
        )
        return

    from .paths import workspace_root

    source_model = str(args.model)
    workspace = workspace_root() / "quantization" / Path(source_model).name
    workspace.mkdir(parents=True, exist_ok=True)

    # Adapter lives in the orchestrator package; lazy-import so the CLI keeps
    # importing cleanly even in environments without the quantization deps.
    # _run_optimize already runs under asyncio.run, so await the async form
    # directly (the sync wrapper would call asyncio.run inside a live loop).
    from hyperloom.orchestrator.quantization_request_handlers import (
        run_quantization_prelude_async,
    )

    quantized_model_dir = await run_quantization_prelude_async(
        prompt=prompt,
        source_model=source_model,
        workspace=workspace,
    )

    args.model = Path(quantized_model_dir)
    os.environ["MODEL_PATH"] = str(quantized_model_dir)
    # Preserve the SOURCE model identity for session naming / display. The export
    # dir basename is always "quantized" (see quantization_request_handlers), so
    # deriving the model name from args.model would collapse every quantized run
    # to "quantized". Pin "<source>-quantized" so the session dir, SharedState,
    # and manifest carry the real model name and quantized runs stay distinct.
    args.model_display_name = f"{Path(source_model).name}-quantized"
    print(f"Quantization prelude: model -> {quantized_model_dir}")
