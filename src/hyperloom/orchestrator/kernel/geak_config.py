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
from hyperloom.inference_optimizer.session.geak_rebench_verdict import (
    REBENCH_FALLBACK,
    REBENCH_NO_PROMOTE,
    REBENCH_VALIDATED,
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


# Minimum over-baseline gain a same-harness revalidation must show to count as "engaged"; detects a collapse back to
# ~baseline.
_MIN_KERNEL_ENGAGED_GAIN_PCT: float = 2.0


def _geak_revalidation_decision(
    *,
    measured: Any,
    baseline: Any,
    got_hash: str,
    expected_hash: str,
    current_best: Any = None,
) -> str:
    """Decide a geak same-harness (2b) rebench outcome."""
    measured_ok = isinstance(measured, (int, float)) and measured > 0
    baseline_ok = isinstance(baseline, (int, float)) and baseline > 0
    if not (measured_ok and baseline_ok):
        return REBENCH_FALLBACK
    cfg_ok = (not expected_hash) or (str(got_hash or "") == str(expected_hash))
    engaged = float(measured) >= float(baseline) * (1.0 + float(_MIN_KERNEL_ENGAGED_GAIN_PCT) / 100.0)
    if not (cfg_ok and engaged):
        return REBENCH_FALLBACK
    if isinstance(current_best, (int, float)) and current_best > 0 and float(measured) <= float(current_best):
        return REBENCH_NO_PROMOTE
    return REBENCH_VALIDATED


def _geak_result_has_artifact(result: dict[str, Any]) -> bool:
    """Whether a GEAK result carries an authored kernel, head, overlay or patch."""

    def _has_nonempty(entries: Any) -> bool:
        # A list whose items are all empty/blank (e.g. ``[""]``) carries nothing.
        if not isinstance(entries, (list, tuple, set)):
            return bool(entries)
        return any(str(e).strip() for e in entries)

    return (
        _has_nonempty(result.get("accepted_kernels"))
        or _has_nonempty(result.get("accepted_heads"))
        or bool(str(result.get("final_overlay") or "").strip())
        or bool(str(result.get("final_patch") or "").strip())
    )


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


def geak_acceptance_specs(result: Any) -> list[dict[str, Any]]:
    """Return every GEAK acceptance of both lanes, tagged with its ``lane`` and deduped.

    GEAK's own ``e2e_delta_pct`` never decides whether a kernel counts: the rebench
    measures it. The delta is read only as twin evidence -- a slot tag and the symbol
    it produced report the same number, and the tag collapses onto the symbol, which
    lists it in ``aliases``. An exact duplicate name drops; an absent or unreadable
    delta never collapses; ``env`` selections are kept as they are.
    """
    if not isinstance(result, dict):
        return []
    lanes = [
        *(("kernelQueue", row) for row in result.get("accepted_kernels") or []),
        *(("headQueue", row) for row in result.get("accepted_heads") or []),
    ]
    out: list[dict[str, Any]] = []
    index: dict[tuple[str, str], int] = {}
    for lane, raw in lanes:
        name = geak_spec_name(raw)
        if not isinstance(raw, dict) or not name:
            continue
        row = {**raw, "lane": lane, "alias_collapsed": False}
        if geak_spec_is_env(raw):
            out.append({**row, "kind": "env"})
            continue
        try:
            twin = (str(raw.get("op_kind") or ""), f"{float(raw.get('e2e_delta_pct')):.4f}")
        except (TypeError, ValueError):
            out.append(row)
            continue
        pos = index.setdefault(twin, len(out))
        if pos == len(out):
            out.append(row)
            continue
        kept = out[pos]
        kept_name = geak_spec_name(kept)
        if name == kept_name:
            continue
        if geak_is_cand_tag(kept_name) == geak_is_cand_tag(name):
            out.append(row)
            continue
        survivor = row if geak_is_cand_tag(kept_name) else kept
        out[pos] = survivor
        survivor["alias_collapsed"] = True
        aliases = {*(survivor.get("aliases") or []), kept_name, name} - {geak_spec_name(survivor)}
        survivor["aliases"] = sorted(aliases)
    return out


def _geak_accepted_kernel_specs(result: Any) -> list[dict[str, Any]]:
    """Return the authored kernels a GEAK result accepted: its acceptances without the ``env`` selections."""
    return [spec for spec in geak_acceptance_specs(result) if not geak_spec_is_env(spec)]


def _geak_has_accepted_kernel(result: Any) -> bool:
    """Report whether a GEAK result names an authored kernel for the rebench to measure."""
    return bool(_geak_accepted_kernel_specs(result))


def geak_overlay_is_loadable(overlay: str) -> bool:
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
