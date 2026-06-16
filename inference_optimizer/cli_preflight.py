# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Pre-flight gates for the CLI: context-window + model-config compatibility.

Extracted from ``cli.py`` (phase 4). Validates the requested context window and
model-config compatibility before a run is born. Imports stdlib + cli_model_gate
only; must not import ``cli`` (one-way dependency).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from .orchestrator.shared_state import SharedState
from .cli_model_gate import (
    _detect_incompatible_model_config,
    _detect_unsupported_model,
    _load_model_max_position_embeddings,
    _VERDICT_TEXT_COERCIBLE,
    _VERDICT_VISION_ONLY,
)

log = logging.getLogger(__name__)

_CONTEXT_HEADROOM_ENV = "HYPERLOOM_CONTEXT_HEADROOM_TOKENS"

_CONTEXT_HEADROOM_DEFAULT = 512

_MAX_MODEL_LEN_HEADROOM = 4096

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

def _preflight_context_window(args: argparse.Namespace, session_dir: Path) -> bool:
    """Fail fast when ``max_position_embeddings < ISL+OSL+headroom`` (no --context-length stretch by policy).

    Persists a stop reason and returns True (caller should exit) when the workload does NOT fit; False
    when it fits or the model's max length is unknown.
    """
    isl = int(getattr(args, "isl", 0) or 0)
    osl = int(getattr(args, "osl", 0) or 0)
    if isl <= 0 or osl <= 0:
        return False
    maxpos = _load_model_max_position_embeddings(str(getattr(args, "model", "") or ""))
    if not maxpos:
        return False
    headroom = _context_headroom_tokens()
    required = isl + osl + headroom
    if maxpos >= required:
        return False

    reason = (
        f"model max_position_embeddings={maxpos} < required {required} "
        f"(ISL={isl} + OSL={osl} + headroom={headroom}). The workload exceeds "
        f"the model context window; every request would 400. Refusing to run "
        f"(no --context-length override by policy). Lower ISL/OSL for this "
        f"model, or lower {_CONTEXT_HEADROOM_ENV} if the headroom is too "
        f"conservative (it is added to `required`, so raising it makes "
        f"admission stricter, not looser)."
    )
    # Persist the stop reason so CI / the robustness monitor read it from state.json instead of the log.
    try:
        from .orchestrator.shared_state import SharedState
        from .orchestrator.action_executors.report import (
            _build_summary_dict,
            _format_md,
        )
        from .session_paths import reports_dir

        state = SharedState.load_or_init(session_dir)
        # Validated writer keeps the vocab-closed invariant Inv-8.3 (term registered in STOP_REASON_VOCAB).
        state.set_stop_reason("model_context_window_too_small")
        state.closing_phase = True
        state.save(session_dir)
        summary = _build_summary_dict(state, {}, [], external_baseline=None)
        summary["stop_detail"] = reason
        rdir = reports_dir(session_dir)
        rdir.mkdir(parents=True, exist_ok=True)
        (rdir / "final.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8",
        )
        (rdir / "final.md").write_text(_format_md(summary), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 — don't mask the reason on a writer bug
        print(
            f"WARNING: failed to persist context-window stop report: {exc!r}",
            file=sys.stderr,
        )
    # Delivery-artifact parity: emit session_breakdown.json here too since fail-fast exits before
    # coordinator.run()'s finally, so CI's delivery contract sees a clean skip not "Missing artifacts".
    try:
        from .breakdown import write_breakdown_json
        write_breakdown_json(session_dir)
    except Exception as exc:  # noqa: BLE001 — best-effort; never mask the reason
        print(
            f"WARNING: failed to write session_breakdown.json on context "
            f"fail-fast: {exc!r}",
            file=sys.stderr,
        )
    print(f"ERROR: {reason}", file=sys.stderr)
    return True

def _preflight_model_config_compat(
    args: argparse.Namespace, session_dir: Path,
) -> bool:
    """Fail fast when the model config is statically known to be incompatible.

    Catches configs that crash vLLM/transformers at load (corrupt config.json,
    or a RoPE block without any max-position field) so we persist a clear stop
    reason instead of booting a server that dies cryptically in engine init.

    Returns True when incompatible (caller should exit); False otherwise.
    """
    model = str(getattr(args, "model", "") or "")
    detail = _detect_incompatible_model_config(
        model, str(getattr(args, "gpu_type", "") or "") or None,
    )
    if detail is None:
        return False
    name = Path(model).name or model
    reason = (
        f"Model '{name}' has an incompatible config: {detail} Refusing to run "
        f"before the heavy server bring-up. Upgrade the framework/transformers "
        f"to a version that supports this model, or skip it on this hardware."
    )
    try:
        from .orchestrator.shared_state import SharedState
        from .orchestrator.action_executors.report import (
            _build_summary_dict,
            _format_md,
        )
        from .session_paths import reports_dir

        state = SharedState.load_or_init(session_dir)
        state.set_stop_reason("model_config_incompatible")
        state.closing_phase = True
        state.save(session_dir)
        summary = _build_summary_dict(state, {}, [], external_baseline=None)
        summary["stop_detail"] = reason
        rdir = reports_dir(session_dir)
        rdir.mkdir(parents=True, exist_ok=True)
        (rdir / "final.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8",
        )
        (rdir / "final.md").write_text(_format_md(summary), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 — don't mask the reason on a writer bug
        print(
            f"WARNING: failed to persist model-config stop report: {exc!r}",
            file=sys.stderr,
        )
    try:
        from .breakdown import write_breakdown_json
        write_breakdown_json(session_dir)
    except Exception as exc:  # noqa: BLE001 — best-effort; never mask the reason
        print(
            f"WARNING: failed to write session_breakdown.json on config "
            f"fail-fast: {exc!r}",
            file=sys.stderr,
        )
    print(f"ERROR: {reason}", file=sys.stderr)
    return True

def _preflight_unsupported_model_arch(
    args: argparse.Namespace, session_dir: Path,
) -> bool:
    """Gate multimodal/vision models before expensive bring-up.

    Best-effort (an unreadable config.json is not a hard block). Three outcomes:

    * plain text model → returns False (run proceeds normally).
    * ``text_coercible`` (multimodal signal but a text decoder exists) →
      when ``--allow-mm-text-fallback`` is on (default), records a degraded-mode
      warning on SharedState, emits a loud stderr/log warning, and returns False
      so the run proceeds on the text path. When the flag is off, falls through
      to fail-fast.
    * ``vision_only`` (true VLM / unclassifiable) → persists
      ``stop_reason=unsupported_model_arch`` and returns True (caller exits).
    """
    model = str(getattr(args, "model", "") or "")
    hit = _detect_unsupported_model(model)
    if hit is None:
        return False

    name = Path(model).name or model
    arch = hit.get("architecture") or "<unknown>"
    mt = hit.get("model_type") or "<unknown>"
    verdict = str(hit.get("verdict") or _VERDICT_VISION_ONLY)
    allow_fallback = bool(getattr(args, "allow_mm_text_fallback", True))

    if verdict == _VERDICT_TEXT_COERCIBLE and allow_fallback:
        warning = (
            f"DEGRADED MODE: model '{name}' carries a multimodal signal "
            f"({hit.get('signal', 'multimodal config')}; architecture '{arch}', "
            f"model_type '{mt}') but exposes a text-generation path. Hyperloom "
            f"is running it on the TEXT path only — any image/audio inputs are "
            f"ignored, so benchmark numbers reflect the text decoder alone. "
            f"Pass --no-allow-mm-text-fallback to fail-fast instead."
        )
        print(f"WARNING: {warning}", file=sys.stderr)
        log.warning(warning)
        try:
            from .orchestrator.shared_state import SharedState

            state = SharedState.load_or_init(session_dir)
            state.degraded_mode = True
            state.model_warnings = list(state.model_warnings or []) + [{
                "kind": "multimodal_text_fallback",
                "model_name": name,
                "architecture": arch,
                "model_type": mt,
                "signal": str(hit.get("signal") or ""),
                "detail": warning,
            }]
            state.save(session_dir)
        except Exception as exc:  # noqa: BLE001 — never block the run on advisory write
            print(
                f"WARNING: failed to persist degraded-mode marker: {exc!r}",
                file=sys.stderr,
            )
        return False

    reason = (
        f"Unsupported model '{name}': architecture '{arch}' (model_type "
        f"'{mt}') is not a supported text-generation model. Hyperloom only "
        f"supports decoder-only causal LM models (architectures containing "
        f"ForCausalLM or LMHeadModel). Rejected because: "
        f"{hit.get('signal', 'unknown architecture')}. Submit a "
        f"text-generation checkpoint instead."
    )
    # Persist the stop reason so CI / the robustness monitor read it from state.json instead of the log.
    try:
        from .orchestrator.shared_state import SharedState
        from .orchestrator.action_executors.report import (
            _build_summary_dict,
            _format_md,
        )
        from .session_paths import reports_dir

        state = SharedState.load_or_init(session_dir)
        # Validated writer keeps the vocab-closed invariant Inv-8.3 (term registered in STOP_REASON_VOCAB).
        state.set_stop_reason("unsupported_model_arch")
        state.closing_phase = True
        state.save(session_dir)
        summary = _build_summary_dict(state, {}, [], external_baseline=None)
        summary["stop_detail"] = reason
        rdir = reports_dir(session_dir)
        rdir.mkdir(parents=True, exist_ok=True)
        (rdir / "final.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8",
        )
        (rdir / "final.md").write_text(_format_md(summary), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 — don't mask the reason on a writer bug
        print(
            f"WARNING: failed to persist unsupported-model stop report: {exc!r}",
            file=sys.stderr,
        )
    # Delivery-artifact parity: emit session_breakdown.json here too since fail-fast exits before
    # coordinator.run()'s finally, so CI's delivery contract sees a clean skip not "Missing artifacts".
    try:
        from .breakdown import write_breakdown_json
        write_breakdown_json(session_dir)
    except Exception as exc:  # noqa: BLE001 — best-effort; never mask the reason
        print(
            f"WARNING: failed to write session_breakdown.json on unsupported-"
            f"model fail-fast: {exc!r}",
            file=sys.stderr,
        )
    print(f"ERROR: {reason}", file=sys.stderr)
    return True

