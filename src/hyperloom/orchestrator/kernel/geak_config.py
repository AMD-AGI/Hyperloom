# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GEAK accepted-config normalisation, result inspection, and overlay helpers."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shlex
from pathlib import Path
from typing import Any

from hyperloom.common.env_safety import (
    filter_untrusted_env_mapping,
    is_allowed_external_env_key,
    is_allowed_variant_env_key,
)

log = logging.getLogger(__name__)


def _split_env_and_flags(env_str: str) -> tuple[dict[str, str], str]:
    """Split a bench-style config string into (env dict, flags string)."""
    envs: dict[str, str] = {}
    flag_tokens: list[str] = []
    quoted = True
    try:
        tokens = shlex.split(str(env_str or ""))
    except ValueError:
        tokens = str(env_str or "").split()
        quoted = False
    expects_value = False
    for tok in tokens:
        if tok.startswith("-"):
            flag_tokens.append(tok)
            expects_value = "=" not in tok
        elif re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tok):
            k, v = tok.split("=", 1)
            envs[k] = v
            expects_value = False
        elif expects_value:
            flag_tokens.append(tok)
            expects_value = False
    return envs, shlex.join(flag_tokens) if quoted else " ".join(flag_tokens)


def _accepted_config_as_variant(cfg: Any) -> tuple[str, dict[str, str]]:
    """Normalize a GEAK ``accepted_config`` into the ``(args, envs)`` a variant runs."""
    cfg = cfg if isinstance(cfg, dict) else {}
    _accepted_config_controls(cfg)
    flags = str(cfg.get("flags") or "").strip()
    legacy_envs, extra_flags = _split_env_and_flags(str(cfg.get("env") or ""))
    if cfg.get("env_unparsed"):
        log.warning("GEAK accepted_config.env_unparsed reports discarded source text")
        from hyperloom.inference_optimizer.grid_server_args import remove_server_args

        extra_flags = remove_server_args(extra_flags, cfg["env_unparsed"])
    if extra_flags:
        log.warning("GEAK accepted_config.env contains server flags; retaining them alongside accepted_config.flags")
        flags = (flags + " " + extra_flags).strip()
    if "env_map" in cfg:
        envs = cfg["env_map"]
        if not isinstance(envs, dict) or any(
            not isinstance(key, str)
            or not isinstance(value, str)
            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)
            or "\0" in value
            for key, value in envs.items()
        ):
            raise ValueError("GEAK accepted_config.env_map must map strings to strings")
    else:
        envs = legacy_envs
    envs, _dropped = filter_untrusted_env_mapping(envs, allow_predicate=is_allowed_variant_env_key)
    return flags, envs


def _accepted_config_controls(cfg: Any, *, inherited_remove_args: Any = None) -> dict[str, Any]:
    """Normalize explicit GEAK launch controls; unmarked flags remain a delta.

    ``args_mode=replace`` attests that ``flags`` is complete. Neither a result
    schema version nor an empty environment mapping carries that meaning.
    Environment removals precede current assignments; an assignment re-enables
    the name even when inherited controls still list it in ``unset_envs``.
    Flag-delta mode similarly re-enables removed arguments. Complete flags
    are snapshots subject to the final removals; GEAK clears superseded removal
    specs before returning that snapshot. An omitted removal list inherits the
    prior stack's controls, while an explicit empty list clears them.
    """
    from ..actions.executors._proposal_identity import controls_of, normalize_proposal

    cfg = cfg if isinstance(cfg, dict) else {}
    controls = controls_of(normalize_proposal(cfg))
    if any(not is_allowed_external_env_key(name) for name in controls.get("unset_envs", [])):
        raise ValueError("GEAK accepted_config.unset_envs contains a forbidden environment name")
    if controls.get("args_mode") == "replace" and "remove_args" not in cfg and inherited_remove_args:
        controls["remove_args"] = list(inherited_remove_args)
    return controls


def _geak_revalidation_decision(
    *,
    measured: Any,
    baseline: Any,
    got_hash: str,
    expected_hash: str,
    min_engaged_gain_pct: float,
    current_best: Any = None,
) -> str:
    """Decide a geak same-harness (2b) rebench outcome."""
    measured_ok = isinstance(measured, (int, float)) and measured > 0
    baseline_ok = isinstance(baseline, (int, float)) and baseline > 0
    if not (measured_ok and baseline_ok):
        return "fallback"
    cfg_ok = (not expected_hash) or (str(got_hash or "") == str(expected_hash))
    engaged = float(measured) >= float(baseline) * (1.0 + float(min_engaged_gain_pct) / 100.0)
    if not (cfg_ok and engaged):
        return "fallback"
    if isinstance(current_best, (int, float)) and current_best > 0 and float(measured) <= float(current_best):
        return "no_promote"
    return "validated"


def _geak_result_has_material(
    result: Any,
    *,
    prev_best_flags: str = "",
    prev_best_envs: Any = None,
    prev_best_controls: Any = None,
) -> bool:
    """Decide whether a GEAK result carries a material optimization product."""
    from hyperloom.inference_optimizer.canonical_fingerprint import (
        canonical_fingerprint,
    )

    def _has_nonempty(entries: Any) -> bool:
        # A list whose items are all empty/blank (e.g. ``[""]``) is not material.
        if not isinstance(entries, (list, tuple, set)):
            return bool(entries)
        return any(str(e).strip() for e in entries)

    if not isinstance(result, dict) or not result:
        return True
    if _has_nonempty(result.get("accepted_kernels")):
        return True
    if _has_nonempty(result.get("accepted_heads")):
        return True
    if str(result.get("final_overlay") or "").strip():
        return True
    if str(result.get("final_patch") or "").strip():
        return True
    accepted_flags, parsed_envs = _accepted_config_as_variant(result.get("accepted_config"))
    prior_controls = _accepted_config_controls(prev_best_controls)
    controls = _accepted_config_controls(
        result.get("accepted_config"), inherited_remove_args=prior_controls.get("remove_args")
    )
    # A missing / all-empty accepted_config carries no config optimization; a
    # bare fingerprint mismatch against a non-empty current_best is NOT material
    # (promoting it would wipe the existing config to empty).
    if not accepted_flags and not parsed_envs and not controls:
        return False
    # Both sides go through the same guard: a resume can hand current_best the raw accepted_config, and an untrusted
    # key on one side only reads as a diff.
    prev_envs, _dropped = filter_untrusted_env_mapping(
        dict(prev_best_envs or {}),
        allow_predicate=is_allowed_variant_env_key,
    )
    got_fp = canonical_fingerprint(accepted_flags, parsed_envs, **controls)
    prev_fp = canonical_fingerprint(
        str(prev_best_flags or ""), prev_envs, **(_accepted_config_controls(prev_best_controls) if controls else {})
    )
    return got_fp != prev_fp


def _normalize_geak_overlay_dir(overlay: str) -> str:
    """Normalize a GEAK ``final_overlay`` path to the loadable overlay dir."""
    if not overlay:
        return overlay
    try:
        p = Path(overlay)
        child = p / "overlay"
        if p.is_dir() and child.is_dir():
            return str(child)
    except (OSError, ValueError):
        return overlay
    return overlay


# A GEAK candidate slot tag (``cand_c0_triton``, ``c1_triton``), as opposed to the name of the kernel the slot
# produced.
_GEAK_CAND_TAG_RE = re.compile(r"^(cand[_-])?c\d+([_-]|$)", re.IGNORECASE)


def geak_is_cand_tag(name: Any) -> bool:
    """Return True when ``name`` is a GEAK slot tag, not a kernel symbol."""
    text = str(name or "").strip()
    return bool(text) and bool(_GEAK_CAND_TAG_RE.match(text))


def geak_spec_name(spec: Any) -> str:
    """Return the display name of one GEAK acceptance entry."""
    if isinstance(spec, str):
        return spec.strip()
    if not isinstance(spec, dict):
        return ""
    return str(spec.get("short_name") or spec.get("kernel_id") or spec.get("cand_tag") or "").strip()


def geak_spec_kind(spec: Any) -> str | None:
    """Return the acceptance ``kind``, or ``None`` when the source omits it."""
    if not isinstance(spec, dict):
        return None
    raw = spec.get("kind")
    if raw is None:
        return None
    text = str(raw).strip().lower()
    return text or None


def geak_spec_is_env(spec: Any) -> bool:
    """Return True only when the acceptance is *known* to be an env selection."""
    return geak_spec_kind(spec) == "env"


def _geak_accepted_kernel_specs(result: Any) -> list[dict[str, Any]]:
    """Return the authored kernels a GEAK result accepted, both lanes, deduped."""
    if not isinstance(result, dict):
        return []
    out: list[dict[str, Any]] = []
    index: dict[tuple[str, str], int] = {}
    lanes = (result.get("accepted_kernels") or []) + (result.get("accepted_heads") or [])
    for k in lanes:
        if not isinstance(k, dict):
            continue
        if geak_spec_is_env(k):
            continue
        try:
            delta = float(k.get("e2e_delta_pct") or 0.0)
        except (TypeError, ValueError):
            continue
        if delta <= 0.0:
            continue
        name = str(k.get("short_name") or k.get("kernel_id") or k.get("cand_tag") or "").strip()
        if not name:
            continue
        twin = (str(k.get("op_kind") or ""), f"{delta:.4f}")
        pos = index.get(twin)
        if pos is None:
            index[twin] = len(out)
            out.append(k)
            continue
        existing_name = geak_spec_name(out[pos])
        if _GEAK_CAND_TAG_RE.match(existing_name) and not _GEAK_CAND_TAG_RE.match(name):
            out[pos] = k
            continue
        if _GEAK_CAND_TAG_RE.match(name) and not _GEAK_CAND_TAG_RE.match(existing_name):
            continue
        if name == existing_name:
            continue
        out.append(k)
    return out


def _geak_has_accepted_kernel(result: Any) -> bool:
    """Report whether a GEAK result carries an accepted kernel that gained."""
    return bool(_geak_accepted_kernel_specs(result))


def _geak_overlay_is_loadable(overlay: str) -> bool:
    """Report whether an overlay dir can actually install an authored kernel."""
    from hyperloom.common.overlay import overlay_is_loadable

    return overlay_is_loadable(overlay)


def _geak_overlay_digest(overlay: str) -> str:
    """Digest the overlay's bind manifest, or ``\"\"`` when it has none."""
    if not overlay:
        return ""
    root = Path(overlay)
    try:
        raw = (root / "_overlay_manifest.json").read_bytes()
    except (OSError, ValueError):
        return ""
    hasher = hashlib.sha256()
    hasher.update(raw)
    try:
        spec = json.loads(raw)
    except (ValueError, json.JSONDecodeError):
        spec = None
    if isinstance(spec, dict):
        bodies: set[str] = set()
        for mod in spec.get("modules") or []:
            if isinstance(mod, dict) and str(mod.get("file") or "").strip():
                bodies.add(str(mod["file"]).strip())
        for rebind in spec.get("rebinds") or []:
            if isinstance(rebind, dict) and str(rebind.get("impl_module") or "").strip():
                bodies.add(f"{str(rebind['impl_module']).strip()}.py")
        for rel in sorted(bodies):
            hasher.update(rel.encode("utf-8", "replace"))
            try:
                hasher.update(hashlib.sha256((root / rel).read_bytes()).digest())
            except (OSError, ValueError):
                continue
    return hasher.hexdigest()[:16]


def _geak_sweep_measured_tput(res: dict[str, Any]) -> float | None:
    """The value a ``sweep_via_geak`` replay measured on its requested axis, or None.

    ``measured_value`` is the axis-neutral field; the ``output_throughput`` fallback keeps results recorded before it
    existed readable. Off the output axis only the neutral field is populated, because there the number is not a
    throughput.
    """
    if not isinstance(res, dict):
        return None
    best = res.get("promotion_measurement")
    if not isinstance(best, dict):
        return None
    tput = best.get("measured_value")
    if not isinstance(tput, (int, float)):
        tput = best.get("output_throughput")
    return float(tput) if isinstance(tput, (int, float)) and tput > 0 else None
