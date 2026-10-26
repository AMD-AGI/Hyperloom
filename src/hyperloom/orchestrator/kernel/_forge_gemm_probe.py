# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Whether the forge GEMM tuner can run on this host, and at which precision and quant type."""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from hyperloom.common.checkpoint_quant import checkpoint_format
from hyperloom.common.env import is_truthy
from hyperloom.inference_optimizer.model_config_utils import load_model_config_dict

from . import forge_gemm_tuning
from ._integrate_payload import _normalize_precision, _requested_framework, _requested_setting
from ._untuned_gemm_csv import _resolve_fp8_quant_type

log = logging.getLogger(__name__)


def _parse_forge_gemm_sentinel(stdout: str) -> dict[str, Any] | None:
    """Parse FORGE_GEMM_TUNE_RESULT_BEGIN/END sentinel block from stdout."""
    m = re.search(
        r"FORGE_GEMM_TUNE_RESULT_BEGIN\s*\n(.*?)\nFORGE_GEMM_TUNE_RESULT_END",
        stdout,
        re.DOTALL,
    )
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except (json.JSONDecodeError, ValueError):
        return None


def _read_forge_result_json(workspace: Path) -> dict[str, Any]:
    """Read forge's on-disk ``result.json`` from the tuning workspace.

    forge always writes the full report (including ``tuners_skipped``) to
    ``<output_dir>/result.json``, even when the stdout sentinel omits some
    fields. Returns ``{}`` when missing or unparseable.
    """
    try:
        path = workspace / "result.json"
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            if isinstance(data, dict):
                return data
    except (OSError, json.JSONDecodeError, ValueError):
        pass
    return {}


def _derive_gemm_skip_reason(tuners_skipped: Any) -> str:
    """Join forge per-tuner skip reasons into one concise human-readable string."""
    if not isinstance(tuners_skipped, list):
        return ""
    parts: list[str] = []
    for entry in tuners_skipped:
        if not isinstance(entry, dict):
            continue
        reason = str(entry.get("skip_reason") or "").strip()
        if not reason:
            continue
        tuner = str(entry.get("tuner") or "").strip()
        parts.append(f"{tuner}: {reason}" if tuner else reason)
    return "; ".join(parts)


_FORGE_GEMM_PREFLIGHT_TIMEOUT_SEC = 30


def _forge_gemm_tune_available() -> bool:
    """Check exactly what ``forge_gemm_tuning.build_command`` will run, in the interpreter it runs in.

    The tuner is a subpackage of the ``kernelforge`` that ships in this
    distribution, invoked as ``sys.executable -m kernelforge.cli gemm-tune run``.
    Vendoring forge in-tree removes the cross-checkout failures this probe was
    built for, but not the reason it is a subprocess: ``find_spec`` proves the
    module is importable and says nothing about whether ``gemm-tune`` is
    registered on the CLI, which is the thing ``build_command`` actually needs. So
    ask the subcommand itself -- in a subprocess, so a heavy CLI import cannot
    land in the orchestrator's own process. ``--help`` exits 0 only if
    ``kernelforge.cli`` imported and ``gemm-tune`` is registered on it.
    """
    try:
        proc = subprocess.run(
            forge_gemm_tuning.gemm_tune_argv("--help"),
            capture_output=True,
            text=True,
            timeout=_FORGE_GEMM_PREFLIGHT_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:
        log.warning(
            "forge-gemm-tune preflight timed out after %ss in %s",
            _FORGE_GEMM_PREFLIGHT_TIMEOUT_SEC,
            sys.executable,
        )
        return False
    except (OSError, subprocess.SubprocessError) as exc:
        log.info("forge-gemm-tune preflight could not start in %s: %s", sys.executable, exc)
        return False
    if proc.returncode == 0:
        return True
    log.info(
        "forge-gemm-tune preflight failed (rc=%s): %s",
        proc.returncode,
        ((proc.stderr or proc.stdout or "").strip()[-400:] or "(no output)"),
    )
    return False


def _resolve_aiter_root_for_forge() -> str:
    """Resolve AITER's source root from split wheels or editable installs."""
    explicit = os.environ.get("AITER_ROOT_DIR", "").strip()
    if explicit:
        return explicit
    try:
        spec = importlib.util.find_spec("aiter_meta")
    except (ModuleNotFoundError, ValueError):
        spec = None
    locations = getattr(spec, "submodule_search_locations", None) or []
    for location in locations:
        root = Path(location)
        if (root / "csrc").is_dir():
            return str(root)
    try:
        spec = importlib.util.find_spec("aiter")
    except (ModuleNotFoundError, ValueError):
        spec = None
    locations = list(getattr(spec, "submodule_search_locations", None) or [])
    origin = getattr(spec, "origin", None)
    if origin:
        locations.append(str(Path(origin).parent))
    for location in locations:
        package = Path(location)
        root = package.parent
        if package.name == "aiter" and (root / "csrc").is_dir():
            return str(root)
    return ""


def _resolve_forge_precision_and_quant(state, payload: dict) -> tuple[str, str]:
    """Resolve the actual runtime precision and quant_type for forge tuning.

    Priority:
    1. Explicit payload override
    2. --quantization from current_best server args (actual runtime)
    3. state.precision (session-level, may be stale)
    4. Default: bf16

    Returns (precision, quant_type) tuple.
    """
    from hyperloom.inference_optimizer.grid_server_args import server_arg_value
    from hyperloom.inference_optimizer.roofline_ceiling import resolve_runtime_workload

    quant_type = str(payload.get("quant_type") or "auto").strip()

    if payload.get("precision"):
        precision = _normalize_precision(payload["precision"])
        if precision == "fp8" and quant_type.lower() == "auto":
            quant_type = _requested_fp8_quant_type(state, payload)
        return precision, quant_type

    # Resolve from actual server args (baseline yaml + current_best overlay).
    current_best = getattr(state, "current_best", None) or {}
    server_args = resolve_runtime_workload(state, arm="current_best").server_args
    extra_envs = dict(current_best.get("extra_envs") or {})
    ref_envs = dict(getattr(state, "reference_envs", None) or {})
    per_token_signal = is_truthy(extra_envs.get("SGLANG_USE_AITER_FP8_PER_TOKEN")) or is_truthy(
        ref_envs.get("SGLANG_USE_AITER_FP8_PER_TOKEN")
    )

    quantization_arg = server_arg_value(server_args, "--quantization").lower()

    if quantization_arg == "fp8":
        precision = "fp8"
        # Hand forge the fp8 GEMM path the model runs: explicit per-token env wins,
        # else the checkpoint's static format, else "auto".
        if per_token_signal:
            quant_type = "per_token"
        else:
            quant_type = _requested_fp8_quant_type(state, payload)
        return precision, quant_type

    if quantization_arg in ("fp4", "mxfp4"):
        return quantization_arg, "fp4"

    # Fall back to session precision. When that is just the format a Quark or compressed-tensors checkpoint nests, it
    # is one label for layers the checkpoint can store differently (MXFP4 experts, fp8 attention), and forge can tell
    # them apart only when it is left to route each lane itself.
    precision = _normalize_precision(state.precision)
    model_path = _requested_setting(payload, state, "model_path")
    if precision and precision == checkpoint_format(
        (load_model_config_dict(model_path) or {}).get("quantization_config")
    ):
        return "auto", quant_type
    if not precision:
        precision = "bf16"
    if precision == "fp8" and quant_type.lower() == "auto":
        quant_type = _requested_fp8_quant_type(state, payload)
    return precision, quant_type


def _requested_fp8_quant_type(state, payload: dict) -> str:
    return _resolve_fp8_quant_type(
        _requested_setting(payload, state, "model_path"),
        _requested_setting(payload, state, "gpu_type"),
        _requested_framework(payload, state),
    )
