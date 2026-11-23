# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Model / GPU gate for the CLI: arch / config loading, unsupported-model detection, and the
pre-flight gates that run before a session is born.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import framework_registry
from ...common.model_paths import resolve_local_model_dir
from ..session import sbd_v6
from hyperloom_common.timeutil import now_iso
from ..model_config_utils import (
    _config_architectures,
    load_model_config_dict,
    _load_model_max_position_embeddings,
)
from .model_compat import _detect_incompatible_model_config

log = logging.getLogger(__name__)

_SUPPORTED_ARCH_MARKERS = (
    "ForCausalLM",
    "LMHeadModel",
    "ForCausalLMWithValueHead",
)

_SUPPORTED_MODEL_TYPES = frozenset(
    {
        "llama",
        "mistral",
        "mixtral",
        "qwen2",
        "qwen2_moe",
        "qwen3",
        "qwen3_moe",
        "gemma",
        "gemma2",
        "phi",
        "phi3",
        "phimoe",
        "starcoder2",
        "codellama",
        "deepseek_v2",
        "deepseek_v3",
        "falcon",
        "gpt_neox",
        "gpt2",
        "opt",
        "bloom",
        "internlm",
        "internlm2",
        "yi",
        "baichuan",
        "chatglm",
        "glm",
        "glm4",
        "command-r",
        "cohere",
        "cohere2",
        "dbrx",
        "mpt",
        "olmo",
        "olmo2",
        "jamba",
        "arctic",
        "exaone",
        "granite",
        "granitemoeshared",
        "stablelm",
        "persimmon",
    }
)


_UNSUPPORTED_CONFIG_KEYS = (
    "vision_config",
    "image_token_id",
    "image_token_index",
    "mm_config",
    "multi_modal_config",
    "vision_tower",
    "vision_tower_cfg",
    "image_processor_type",
    "projector_config",
    "mm_projector_type",
)

_TEXT_DECODER_CONFIG_KEYS = (
    "text_config",
    "language_config",
    "llm_config",
)

_VERDICT_TEXT_COERCIBLE = "text_coercible"

_VERDICT_VISION_ONLY = "vision_only"
_UNKNOWN_MODEL_FIELD = "<unknown>"
_UNSUPPORTED_ARCH_DEGRADED = (
    "DEGRADED MODE: model '{name}' carries a multimodal signal "
    "({signal}; architecture '{architecture}', model_type '{model_type}') but exposes a text-generation path. "
    "Hyperloom is running it on the TEXT path only — any image/audio inputs are ignored, so "
    "benchmark numbers reflect the text decoder alone. Pass --no-allow-mm-text-fallback to fail-fast instead."
)
_UNSUPPORTED_ARCH_REJECTED = (
    "Unsupported model '{name}': architecture '{architecture}' (model_type '{model_type}') is not a supported "
    "text-generation model. Hyperloom only supports decoder-only causal LM models (architectures "
    "containing ForCausalLM or LMHeadModel). Rejected because: {signal}. Submit a text-generation "
    "checkpoint instead."
)

_TEXT_COERCIBLE_MODEL_TYPES = frozenset(
    {
        "kimi_k25",
        "qwen3_5_moe",
    }
)


def _read_preseeded_model_arch(arch_path: Path) -> str | None:
    """Return the pre-seeded ``$HYPERLOOM_MODEL_ARCH_FILE`` text, or ``None``."""
    src = (os.environ.get("HYPERLOOM_MODEL_ARCH_FILE") or "").strip()
    if not src:
        return None
    try:
        text = Path(src).expanduser().read_text(encoding="utf-8")
    except OSError as exc:
        logging.warning("model_arch_preseed_unreadable: %s (%s)", src, exc)
        return None
    try:
        arch_path.parent.mkdir(parents=True, exist_ok=True)
        arch_path.write_text(text, encoding="utf-8")
    except OSError as exc:
        logging.warning("model_arch_preseed_copy_failed: %s -> %s (%s)", src, arch_path, exc)
    logging.info("model_arch_preseeded_from: %s", src)
    return text


def _load_model_arch(
    workspace_root: Path,
    model_name: str,
    launched_model: str = "",
) -> dict:
    """Best-effort loader for the advisory ``<workspace_root>/model_arch.json`` profile (prompts only)."""
    from hyperloom.common.model_paths import model_identities_match

    arch_path = workspace_root / "model_arch.json"
    try:
        raw = arch_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raw = _read_preseeded_model_arch(arch_path)
        if raw is None:
            return {}
    except OSError as exc:
        logging.warning("model_arch_unreadable: %s (%s)", arch_path, exc)
        return {}
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        logging.warning("model_arch_invalid_json: %s (%s)", arch_path, exc)
        return {}
    if not isinstance(data, dict):
        logging.warning("model_arch_not_a_dict: %s (got %s)", arch_path, type(data).__name__)
        return {}
    declared = str(data.get("model_name") or "").strip()
    if not declared:
        logging.warning("model_arch_missing_model_name: %s (cannot verify freshness)", arch_path)
        return {}
    if not model_identities_match(declared, model_name, launched_model):
        logging.warning(
            "model_arch_stale_or_mismatch: %s declares model_name=%r but "
            "launching model_name=%r (--model=%r) — ignoring",
            arch_path,
            declared,
            model_name,
            launched_model,
        )
        return {}
    return data


def _load_model_config_tags(model_path: str) -> dict:
    """Best-effort loader for KB architecture-identity tags (``architectures`` + ``model_type``) from config.json."""
    data = load_model_config_dict(model_path)
    if data is None:
        return {}
    out: dict = {}
    arches = _config_architectures(data)
    if arches:
        out["architectures"] = arches
    model_type = str(data.get("model_type") or "").strip()
    if model_type:
        out["model_type"] = model_type
    return out


def _arch_is_supported_text_generation(arch: str) -> bool:
    """True when an architecture class name denotes a supported text-generation (decoder-only causal LM) model."""
    a = (arch or "").strip()
    if not a:
        return False
    return any(marker in a for marker in _SUPPORTED_ARCH_MARKERS)


def _config_declares_text_decoder(config: dict, architectures: list[str], model_type_l: str) -> bool:
    """True when config positively identifies a usable text decoder."""
    if model_type_l in _TEXT_COERCIBLE_MODEL_TYPES:
        return True
    if any(_arch_is_supported_text_generation(a) for a in architectures):
        return True

    for key in _TEXT_DECODER_CONFIG_KEYS:
        nested = config.get(key)
        if not isinstance(nested, dict):
            continue
        nested_architectures = _config_architectures(nested)
        if any(_arch_is_supported_text_generation(a) for a in nested_architectures):
            return True

        nested_model_type = str(nested.get("model_type") or "").strip().lower()
        if nested_model_type in _SUPPORTED_MODEL_TYPES or nested_model_type in _TEXT_COERCIBLE_MODEL_TYPES:
            return True

        # Some multimodal configs expose a text_config with decoder dimensions but an unseen model_type; scoped to a
        # named text block, so this does not widen fallback for a top-level mislabeled VLM.
        has_vocab = isinstance(nested.get("vocab_size"), int) and nested["vocab_size"] > 0
        has_decoder_shape = any(
            isinstance(nested.get(field), int) and nested[field] > 0
            for field in ("hidden_size", "num_hidden_layers", "intermediate_size")
        )
        if has_vocab and has_decoder_shape:
            return True

    return False


def _unsupported_hit(architecture: str, model_type: str, signal: str, verdict: str = _VERDICT_VISION_ONLY) -> dict:
    """One ``_detect_unsupported_model`` hit: what matched, and whether the text path can still serve the model."""
    return {
        "architecture": architecture or _UNKNOWN_MODEL_FIELD,
        "model_type": model_type or _UNKNOWN_MODEL_FIELD,
        "signal": signal,
        "verdict": verdict,
    }


def _detect_unsupported_model(model_path: str) -> dict | None:
    """Best-effort classify a model's text-serving viability."""
    config = load_model_config_dict(model_path)
    if config is None:
        return None
    nested = config.get("text_config")
    nested = nested if isinstance(nested, dict) else {}
    # Wrapper models may nest the real arch under text_config.
    architectures = list(dict.fromkeys(_config_architectures(config) + _config_architectures(nested)))
    first_arch = architectures[0] if architectures else ""
    model_type = str(config.get("model_type") or "").strip()
    model_type_l = model_type.lower()

    # A multimodal config key is a degrade signal: with a text decoder the model runs on the text path with a warning.
    _has_text_decoder = _config_declares_text_decoder(config, architectures, model_type_l)
    for key in _UNSUPPORTED_CONFIG_KEYS:
        if key in config:
            verdict = _VERDICT_TEXT_COERCIBLE if _has_text_decoder else _VERDICT_VISION_ONLY
            return _unsupported_hit(first_arch, model_type, f"multimodal config key '{key}'", verdict)

    if any(_arch_is_supported_text_generation(a) for a in architectures):
        return None

    if model_type_l in _SUPPORTED_MODEL_TYPES:
        return None

    if architectures:
        return _unsupported_hit(
            first_arch,
            model_type,
            f"architecture '{first_arch}' does not match any supported text-generation pattern "
            f"({', '.join(_SUPPORTED_ARCH_MARKERS)})",
        )
    if model_type:
        return _unsupported_hit(
            "", model_type, f"model_type '{model_type}' is not in the supported text-generation allowlist"
        )
    return _unsupported_hit("", "", "config.json has neither architectures nor model_type")


# Pre-flight gates: validate the requested context window + model-config compatibility before a run is born.
_CONTEXT_HEADROOM_ENV = "HYPERLOOM_CONTEXT_HEADROOM_TOKENS"

_CONTEXT_HEADROOM_DEFAULT = 512

_MAX_MODEL_LEN_HEADROOM = 4096

_MODEL_GATE_ORDER = (
    "unsupported_model_arch",
    "model_config_compat",
    "context_window",
)


def _model_gate_workload(args: argparse.Namespace) -> dict[str, Any]:
    model_path = str(getattr(args, "model", "") or "")
    return {
        "model_path": model_path,
        "model_name": str(getattr(args, "model_display_name", "") or "")
        or (Path(model_path).name if model_path else ""),
        "framework": str(getattr(args, "framework", "") or os.environ.get("FRAMEWORK", "")),
        "gpu_type": str(getattr(args, "gpu_type", "") or os.environ.get("TARGET_GPU_TYPE", "")),
        "isl": int(getattr(args, "isl", 0) or 0),
        "osl": int(getattr(args, "osl", 0) or 0),
        "allow_mm_text_fallback": bool(getattr(args, "allow_mm_text_fallback", True)),
        "headroom_tokens": _context_headroom_tokens(),
        "headroom_env": _CONTEXT_HEADROOM_ENV,
    }


def _new_model_gate_event(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "type": "model_gate",
        "kind": "model_gate",
        "status": "succeeded",
        "start_time": now_iso(timespec="seconds"),
        "end_time": "",
        "ext": {
            "run_kind": "fresh",
            "skip_reason": None,
            "failed_gate_id": None,
            "workload": _model_gate_workload(args),
            "checks": [],
            "degraded": {"active": False, "warnings": []},
        },
    }


def _load_model_gate_event(args: argparse.Namespace, session_dir: Path) -> dict[str, Any]:
    return sbd_v6.read_timeline_event_for_update(session_dir, "model_gate") or _new_model_gate_event(args)


def _record_model_gate_warning(session_dir: Path, *, component: str, exc: BaseException) -> None:
    """Best-effort retain a model-gate observability failure for export."""
    if not sbd_v6.record_write_warning(session_dir, component=component, exc=exc):
        log.debug("failed to persist SBD V6 model-gate warning", exc_info=exc)


def _model_gate_status(checks: list[dict[str, Any]]) -> str:
    statuses = {str(check.get("status") or "") for check in checks}
    if "failed" in statuses:
        return "failed"
    if "warned" in statuses or "unknown" in statuses:
        return "degraded"
    return "succeeded"


def _model_gate_check_order(check: dict[str, Any]) -> int:
    return int(check["order"])


def _record_model_gate_check(
    args: argparse.Namespace,
    session_dir: Path,
    check: dict[str, Any],
    *,
    failure: dict[str, Any] | None = None,
    degraded_warning: dict[str, Any] | None = None,
) -> None:
    try:
        event = _load_model_gate_event(args, session_dir)
        ext = event["ext"]
        checks = [
            row for row in ext.get("checks", []) if isinstance(row, dict) and row.get("gate_id") != check.get("gate_id")
        ]
        checks.append(check)
        checks.sort(key=_model_gate_check_order)
        if failure is not None:
            failed_order = int(check.get("order") or 0)
            present = {str(row.get("gate_id") or "") for row in checks}
            for order, gate_id in enumerate(_MODEL_GATE_ORDER, start=1):
                if order > failed_order and gate_id not in present:
                    checks.append(
                        {
                            "gate_id": gate_id,
                            "order": order,
                            "status": "skipped",
                            "skip_reason": "prior_gate_failed",
                            "detail": {},
                        }
                    )
            checks.sort(key=_model_gate_check_order)
            ext["failed_gate_id"] = str(check.get("gate_id") or "")
            ext["failure"] = failure
            event["end_time"] = now_iso(timespec="seconds")
        if degraded_warning is not None:
            degraded = ext.setdefault("degraded", {"active": False, "warnings": []})
            degraded["active"] = True
            degraded.setdefault("warnings", []).append(degraded_warning)
        ext["checks"] = checks
        event["status"] = _model_gate_status(checks)
        sbd_v6.write_timeline_event_at(session_dir, event)
    except (OSError, ValueError, TypeError) as exc:
        log.warning("failed to record SBD V6 model-gate check", exc_info=True)
        _record_model_gate_warning(session_dir, component="model_gate.check", exc=exc)


def _start_model_gate(args: argparse.Namespace, session_dir: Path) -> None:
    """Create the model-gate event before the first check executes."""
    try:
        sbd_v6.write_timeline_event_at(session_dir, _new_model_gate_event(args))
    except (OSError, ValueError, TypeError) as exc:
        log.warning("failed to initialize SBD V6 model-gate event", exc_info=True)
        _record_model_gate_warning(session_dir, component="model_gate.start", exc=exc)


def _finish_model_gate(args: argparse.Namespace, session_dir: Path) -> None:
    """Finalize a successfully completed three-check model-gate chain."""
    try:
        event = _load_model_gate_event(args, session_dir)
        event["status"] = _model_gate_status(event["ext"]["checks"])
        event["end_time"] = now_iso(timespec="seconds")
        sbd_v6.write_timeline_event_at(session_dir, event)
    except (OSError, ValueError, TypeError) as exc:
        log.warning("failed to finalize SBD V6 model-gate event", exc_info=True)
        _record_model_gate_warning(session_dir, component="model_gate.finish", exc=exc)


def _record_resumed_model_gate(
    args: argparse.Namespace,
    session_dir: Path,
    *,
    workload_overrides: Mapping[str, Any] | None = None,
) -> None:
    """Persist the explicit V6 skip required by the resume path."""
    try:
        timestamp = now_iso(timespec="seconds")
        event = _new_model_gate_event(args)
        event["status"] = "skipped"
        event["start_time"] = timestamp
        event["end_time"] = timestamp
        event["ext"]["run_kind"] = "resume"
        event["ext"]["skip_reason"] = "resume"
        if workload_overrides:
            event["ext"]["workload"].update(workload_overrides)
        event["ext"]["checks"] = [
            {
                "gate_id": gate_id,
                "order": order,
                "status": "skipped",
                "skip_reason": "resume",
                "detail": {},
            }
            for order, gate_id in enumerate(_MODEL_GATE_ORDER, start=1)
        ]
        sbd_v6.write_timeline_event_at(session_dir, event)
    except (OSError, ValueError, TypeError) as exc:
        log.warning("failed to record resumed SBD V6 model-gate event", exc_info=True)
        _record_model_gate_warning(session_dir, component="model_gate.resume", exc=exc)


def _write_model_gate_breakdown(
    session_dir: Path,
    *,
    failure_label: str,
) -> None:
    """Write the fail-fast SBD once without masking the gate failure."""
    try:
        from ..breakdown import write_breakdown_json

        write_breakdown_json(session_dir)
    except (OSError, ValueError, TypeError) as exc:
        log.warning("session_breakdown.json write failed on %s fail-fast", failure_label, exc_info=True)
        print(
            f"WARNING: failed to write session_breakdown.json on {failure_label} fail-fast: {exc!r}",
            file=sys.stderr,
        )
        _record_model_gate_warning(session_dir, component=f"model_gate.{failure_label}.breakdown", exc=exc)


def _context_headroom_tokens() -> int:
    """Resolve the context headroom (tokens); env override, else default."""
    raw = os.environ.get(_CONTEXT_HEADROOM_ENV, "").strip()
    if not raw:
        return _CONTEXT_HEADROOM_DEFAULT
    try:
        val = int(raw)
    except ValueError:
        return _CONTEXT_HEADROOM_DEFAULT
    return val if val >= 0 else _CONTEXT_HEADROOM_DEFAULT


def _resolve_max_model_len(isl: int, osl: int, model_path: str) -> int:
    """Resolve ``MAX_MODEL_LEN`` = ISL+OSL+headroom, clamped to ``max_position_embeddings`` (never stretch context)."""
    desired = int(isl) + int(osl) + _MAX_MODEL_LEN_HEADROOM
    maxpos = _load_model_max_position_embeddings(model_path)
    if maxpos:
        return min(desired, maxpos)
    return desired


def _emit_breakdown_to_langfuse(session_dir: Path) -> None:
    """Best-effort: push the just-written ``session_breakdown.json`` to Langfuse."""
    try:
        from ..breakdown import patch_breakdown_langfuse
        from hyperloom.inference_optimizer.trace.langfuse_emitter import (
            flush_session,
            record_session_breakdown,
        )

        flush_session(session_dir)
        patch_breakdown_langfuse(session_dir)
        record_session_breakdown(session_dir)
    except (OSError, ValueError) as exc:
        log.warning("session_breakdown Langfuse push failed on fail-fast", exc_info=True)
        print(
            f"WARNING: failed to emit session_breakdown to Langfuse on fail-fast: {exc!r}",
            file=sys.stderr,
        )


def _persist_gate_stop_report(session_dir: Path, *, stop_reason: str, reason: str, warning_label: str) -> None:
    """Persist the gate stop reason to state.json and the final session report files."""
    try:
        from hyperloom.orchestrator.state.shared_state import SharedState
        from hyperloom.orchestrator.actions.executors.report import write_stop_report

        state = SharedState.load_or_init(session_dir)
        # Validated writer keeps the vocab-closed invariant Inv-8.3.
        state.set_stop_reason(stop_reason)
        from ..breakdown.recorder.close_out import record_close_safety_net
        from ..breakdown.recorder import record_stage_reached

        record_close_safety_net(session_dir)
        record_stage_reached(session_dir, "model_gate")
        state.closing_phase = True
        state.save(session_dir)
        write_stop_report(session_dir, state, stop_detail=reason)
    except (OSError, ValueError, TypeError) as exc:
        log.warning("%s stop report persist failed", warning_label, exc_info=True)
        print(
            f"WARNING: failed to persist {warning_label} stop report: {exc!r}",
            file=sys.stderr,
        )


def _record_context_window_unset(args: argparse.Namespace, session_dir: Path, *, isl: int, osl: int) -> None:
    """Record the context-window gate as skipped because ISL or OSL is unset."""
    _record_model_gate_check(
        args,
        session_dir,
        {
            "gate_id": "context_window",
            "order": 3,
            "status": "skipped",
            "skip_reason": "isl_osl_unset",
            "detail": {
                "isl": isl,
                "osl": osl,
                "headroom": _context_headroom_tokens(),
                "required": None,
                "max_position_embeddings": None,
                "fits": None,
                "policy": "no_context_length_override",
            },
        },
    )


@dataclass(frozen=True)
class _ContextDemand:
    """A workload's context demand measured against the model's context window."""

    isl: int
    osl: int
    headroom: int
    required: int
    maxpos: int


def _refuse_context_window(args: argparse.Namespace, session_dir: Path, demand: _ContextDemand) -> None:
    """Record the context-window gate failure, persist its stop report and breakdown, and print the refusal."""
    isl, osl, headroom, required, maxpos = demand.isl, demand.osl, demand.headroom, demand.required, demand.maxpos
    reason = (
        f"model max_position_embeddings={maxpos} < required {required} "
        f"(ISL={isl} + OSL={osl} + headroom={headroom}). The workload exceeds "
        f"the model context window; every request would 400. Refusing to run "
        f"(no --context-length override by policy). Lower ISL/OSL for this "
        f"model, or lower {_CONTEXT_HEADROOM_ENV} if the headroom is too "
        f"conservative (it is added to `required`, so raising it makes "
        f"admission stricter, not looser)."
    )
    # Persist the stop reason for CI and session diagnostics.
    _persist_gate_stop_report(
        session_dir,
        stop_reason="model_context_window_too_small",
        reason=reason,
        warning_label="context-window",
    )
    _record_model_gate_check(
        args,
        session_dir,
        {
            "gate_id": "context_window",
            "order": 3,
            "status": "failed",
            "skip_reason": None,
            "detail": {
                "isl": isl,
                "osl": osl,
                "headroom": headroom,
                "required": required,
                "max_position_embeddings": maxpos,
                "fits": False,
                "policy": "no_context_length_override",
            },
        },
        failure={
            "gate_id": "context_window",
            "stop_reason": "model_context_window_too_small",
            "exit_code": 2,
            "message": reason,
            "artifacts": {
                "final_json": "reports/final.json" if (session_dir / "reports" / "final.json").is_file() else None,
            },
        },
    )
    # Delivery-artifact parity: emit session_breakdown.json here too since fail-fast exits before coordinator.run()'s
    # finally.
    _write_model_gate_breakdown(session_dir, failure_label="context")
    # Langfuse parity: this gate exits before coordinator.run()'s finally, so push the breakdown to Langfuse here too.
    _emit_breakdown_to_langfuse(session_dir)
    print(f"ERROR: {reason}", file=sys.stderr)


def _preflight_context_window(args: argparse.Namespace, session_dir: Path) -> bool:
    """Fail fast when ``max_position_embeddings < ISL+OSL+headroom`` (no --context-length stretch by policy)."""
    isl = int(getattr(args, "isl", 0) or 0)
    osl = int(getattr(args, "osl", 0) or 0)
    if isl <= 0 or osl <= 0:
        _record_context_window_unset(args, session_dir, isl=isl, osl=osl)
        return False
    maxpos = _load_model_max_position_embeddings(str(getattr(args, "model", "") or ""))
    if not maxpos:
        headroom = _context_headroom_tokens()
        _record_model_gate_check(
            args,
            session_dir,
            {
                "gate_id": "context_window",
                "order": 3,
                "status": "skipped",
                "skip_reason": "max_position_unknown",
                "detail": {
                    "isl": isl,
                    "osl": osl,
                    "headroom": headroom,
                    "required": isl + osl + headroom,
                    "max_position_embeddings": None,
                    "fits": None,
                    "policy": "no_context_length_override",
                },
            },
        )
        return False
    headroom = _context_headroom_tokens()
    required = isl + osl + headroom
    if maxpos >= required:
        _record_model_gate_check(
            args,
            session_dir,
            {
                "gate_id": "context_window",
                "order": 3,
                "status": "passed",
                "skip_reason": None,
                "detail": {
                    "isl": isl,
                    "osl": osl,
                    "headroom": headroom,
                    "required": required,
                    "max_position_embeddings": maxpos,
                    "fits": True,
                    "policy": "no_context_length_override",
                },
            },
        )
        return False

    _refuse_context_window(
        args,
        session_dir,
        _ContextDemand(isl=isl, osl=osl, headroom=headroom, required=required, maxpos=maxpos),
    )
    return True


def _refuse_model_config(args: argparse.Namespace, session_dir: Path, *, model: str, detail: str) -> None:
    """Record the model-config gate failure, persist its stop report and breakdown, and print the refusal."""
    name = Path(model).name or model
    reason = (
        f"Model '{name}' has an incompatible config: {detail} Refusing to run "
        f"before the heavy server bring-up. Upgrade the framework/transformers "
        f"to a version that supports this model, or skip it on this hardware."
    )
    _persist_gate_stop_report(
        session_dir,
        stop_reason="model_config_incompatible",
        reason=reason,
        warning_label="model-config",
    )
    model_dir = resolve_local_model_dir(model) or Path(model)
    config_path = model_dir / "config.json"
    _record_model_gate_check(
        args,
        session_dir,
        {
            "gate_id": "model_config_compat",
            "order": 2,
            "status": "failed",
            "skip_reason": None,
            "detail": {
                "config_path": str(config_path) if config_path.is_file() else None,
                "incompatible": True,
                "reason": detail,
                "detector": None,
            },
        },
        failure={
            "gate_id": "model_config_compat",
            "stop_reason": "model_config_incompatible",
            "exit_code": 2,
            "message": reason,
            "artifacts": {
                "final_json": "reports/final.json" if (session_dir / "reports" / "final.json").is_file() else None,
            },
        },
    )
    _write_model_gate_breakdown(session_dir, failure_label="config")
    # Langfuse parity: this gate exits before coordinator.run()'s finally, so push the breakdown to Langfuse here too.
    _emit_breakdown_to_langfuse(session_dir)
    print(f"ERROR: {reason}", file=sys.stderr)


def _preflight_model_config_compat(
    args: argparse.Namespace,
    session_dir: Path,
) -> bool:
    """Fail fast when the model config is statically known to be incompatible."""
    model = str(getattr(args, "model", "") or "")
    framework = (str(getattr(args, "framework", "") or "") or os.environ.get("FRAMEWORK", "")).strip().lower() or None
    detail = _detect_incompatible_model_config(
        model,
        str(getattr(args, "gpu_type", "") or "") or None,
        framework=framework,
    )
    if detail is None:
        model_dir = resolve_local_model_dir(model) or Path(model)
        config_path = model_dir / "config.json"
        absent = not config_path.is_file()
        _record_model_gate_check(
            args,
            session_dir,
            {
                "gate_id": "model_config_compat",
                "order": 2,
                "status": "skipped" if absent else "passed",
                "skip_reason": "config_absent_soft_pass" if absent else None,
                "detail": {
                    "config_path": str(config_path) if config_path.is_file() else None,
                    "incompatible": False,
                    "reason": None,
                    "detector": None,
                },
            },
        )
        return False
    _refuse_model_config(args, session_dir, model=model, detail=detail)
    return True


def _unsupported_arch_check_row(
    status: str,
    allow: bool,
    detail: Mapping[str, Any],
    skip_reason: str | None = None,
    verdict: str | None = None,
) -> dict[str, Any]:
    """One ``unsupported_model_arch`` check row with the gate's fixed key order."""
    return {
        "gate_id": "unsupported_model_arch",
        "order": 1,
        "status": status,
        "skip_reason": skip_reason,
        "verdict": verdict,
        "detail": {
            "architecture": detail.get("architecture"),
            "model_type": detail.get("model_type"),
            "signal": detail.get("signal"),
            "allow_mm_text_fallback": allow,
            "action": detail.get("action", "proceed"),
        },
    }


def _warn_text_coercible_arch(
    args: argparse.Namespace,
    session_dir: Path,
    hit: Mapping[str, Any],
    name: str,
    allow: bool,
) -> None:
    """Persist degraded mode and record a warned unsupported_model_arch check."""
    warning = _UNSUPPORTED_ARCH_DEGRADED.format(name=name, **hit)
    print(f"WARNING: {warning}", file=sys.stderr)
    log.warning(warning)
    from hyperloom.orchestrator.state.shared_state import SharedState

    detail = {"architecture": hit["architecture"], "model_type": hit["model_type"], "signal": hit["signal"]}
    try:
        state = SharedState.load_or_init(session_dir)
        state.degraded_mode = True
        state.model_warnings = [
            *state.model_warnings,
            {"kind": "multimodal_text_fallback", "model_name": name, **detail, "detail": warning},
        ]
        state.save(session_dir)
    except (OSError, ValueError) as exc:  # the marker is advisory; the run proceeds without it
        print(f"WARNING: failed to persist degraded-mode marker: {exc!r}", file=sys.stderr)
    _record_model_gate_check(
        args,
        session_dir,
        _unsupported_arch_check_row("warned", allow, detail, verdict=hit["verdict"]),
        degraded_warning={"kind": "multimodal_text_fallback", **detail},
    )


def _reject_unsupported_arch(
    args: argparse.Namespace,
    session_dir: Path,
    hit: Mapping[str, Any],
    name: str,
    allow: bool,
) -> None:
    """Fail-fast: stop report, failed gate check, breakdown, Langfuse."""
    reason = _UNSUPPORTED_ARCH_REJECTED.format(name=name, **hit)
    _persist_gate_stop_report(
        session_dir, stop_reason="unsupported_model_arch", reason=reason, warning_label="unsupported-model"
    )
    final_json = "reports/final.json" if (session_dir / "reports" / "final.json").is_file() else None
    _record_model_gate_check(
        args,
        session_dir,
        _unsupported_arch_check_row("failed", allow, {**hit, "action": "fail_fast"}, verdict=hit["verdict"]),
        failure={
            "gate_id": "unsupported_model_arch",
            "stop_reason": "unsupported_model_arch",
            "exit_code": 2,
            "message": reason,
            "artifacts": {"final_json": final_json},
        },
    )
    _write_model_gate_breakdown(session_dir, failure_label="unsupported-model")
    _emit_breakdown_to_langfuse(session_dir)
    print(f"ERROR: {reason}", file=sys.stderr)


def _preflight_unsupported_model_arch(
    args: argparse.Namespace,
    session_dir: Path,
) -> bool:
    """Gate multimodal/vision models before expensive bring-up."""
    allow = bool(getattr(args, "allow_mm_text_fallback", True))
    # Scriptable diffusion frameworks (xDiT) are server-less image workloads, not decoder-only causal LMs.
    if framework_registry.is_scriptable(getattr(args, "framework", "")):
        _record_model_gate_check(
            args,
            session_dir,
            _unsupported_arch_check_row("skipped", allow, {}, skip_reason="scriptable_framework"),
        )
        return False
    model = str(getattr(args, "model", "") or "")
    hit = _detect_unsupported_model(model)
    if hit is None:
        config = load_model_config_dict(model)
        passed = config is not None
        architectures = _config_architectures(config) if passed else []
        _record_model_gate_check(
            args,
            session_dir,
            _unsupported_arch_check_row(
                "passed" if passed else "unknown",
                allow,
                {
                    "architecture": architectures[0] if architectures else None,
                    "model_type": str(config.get("model_type") or "") if passed else None,
                },
                verdict="plain_text" if passed else None,
            ),
        )
        return False
    name = Path(model).name or model
    if hit["verdict"] == _VERDICT_TEXT_COERCIBLE and allow:
        _warn_text_coercible_arch(args, session_dir, hit, name, allow)
        return False
    _reject_unsupported_arch(args, session_dir, hit, name, allow)
    return True
