# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The variant grid one explore round measures.

Where a round's variants come from -- the action payload, the framework-rewrite
levers, the framework's own seed grid -- and which of them are worth launching:
a variant whose settings the server already runs changes nothing, so measuring
it spends a benchmark pass to re-measure the stack.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

import yaml

from hyperloom.common.coerce import to_str_list
from hyperloom.common.launch_log_evidence import settings_the_engine_rewrote
from hyperloom.inference_optimizer.grid_server_args import MULTI_VALUE_FLAGS
from . import _framework_switch_manifest as _switch_manifest
from ._grid_runner import GridVariant, _build_variant_yaml
from ._launch_restatement import config_envs, named_settings, restates_the_baseline
from ._proposal_identity import effective_fingerprint, normalize_proposal
from ._server_argv import ConfigUnreadable, config_server_argv
from ._spec_decode_context_copy import context_copy_refusal

log = logging.getLogger(__name__)


def _action_reasoning(raw: dict[str, Any]) -> tuple[str, str]:
    """Return the action-authored rationale and the payload field that supplied it."""
    for key in ("reasoning", "rationale", "reason", "note"):
        value = str(raw.get(key) or "").strip()
        if value:
            return value, f"action_payload.{key}"
    return "", ""


def _grid_variants_from_payload(payload: list[Any]) -> list[GridVariant]:
    """Convert the LLM/specialist grid payload into GridVariant objects."""
    out: list[GridVariant] = []
    for raw in payload or []:
        if not isinstance(raw, dict) or not raw.get("name"):
            continue
        fields = normalize_proposal(raw)
        reasoning, reasoning_origin = _action_reasoning(raw)
        gv = GridVariant(
            name=fields["name"],
            extra_server_args=fields["extra_args"],
            extra_envs=fields["extra_envs"],
            note=reasoning,
            remove_args=fields["remove_args"],
            unset_envs=fields["unset_envs"],
            args_mode=fields["args_mode"],
        )
        # Stash extra metadata on the GridVariant so the ledger writer can pull provenance.
        gv.provenance = str(raw.get("provenance") or "default_grid")  # type: ignore[attr-defined]
        gv.reasoning_origin = reasoning_origin  # type: ignore[attr-defined]
        gv.scope = str(raw.get("scope") or "")  # type: ignore[attr-defined]
        # Authored-kernel overlay dir (PYTHONPATH prefix); "" for env/flag variants.
        gv.overlay_pythonpath = str(raw.get("overlay_pythonpath") or "")  # type: ignore[attr-defined]
        # Authored kernels this variant's overlay installs.
        gv.accepted_kernels = [  # type: ignore[attr-defined]
            str(k).strip() for k in (raw.get("accepted_kernels") or []) if str(k).strip()
        ]
        # Validated against what the proposing agent was shown before the grid reached this executor.
        gv.experience_citations = [  # type: ignore[attr-defined]
            dict(item) for item in (raw.get("experience_citations") or []) if isinstance(item, dict)
        ]
        out.append(gv)
    return out


def framework_lever_grid(shared_state: Any) -> list[dict[str, Any]]:
    """Build explore variants that attribute each registered rewrite lever."""
    if shared_state is None:
        return []
    rows = list(getattr(shared_state, "authored_framework_levers", None) or [])
    if not rows:
        return []
    pending = [row for row in rows if isinstance(row, dict) and row.get("attributed_gain_pct") is None]
    if not pending:
        return []
    dormant = [row for row in pending if not row.get("default_on")]
    active = [row for row in pending if row.get("default_on")]
    payload: list[dict[str, Any]] = []
    if dormant:
        for variant in _switch_manifest.additive_variants(dormant):
            payload.append({**variant, "framework_lever_source": "additive"})
    if active:
        for variant in _switch_manifest.leave_one_out_variants(active):
            payload.append({**variant, "framework_lever_source": "leave_one_out"})
    return payload


def _framework_lever_attributions(
    per_variant_outcomes: list[dict[str, Any]],
    lever_payload: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Derive each rewrite lever's own contribution from this round's outcomes."""
    if not lever_payload:
        return []
    by_name = {
        str(v.get("name") or ""): v
        for v in lever_payload
        if isinstance(v, dict) and str(v.get("framework_lever") or "")
    }
    if not by_name:
        return []
    out: list[dict[str, Any]] = []
    for row in per_variant_outcomes:
        seed = by_name.get(str(row.get("variant_name") or ""))
        if seed is None:
            continue
        measured = (row.get("metrics") or {}).get("gain_pct")
        if not isinstance(measured, (int, float)):
            continue
        source = str(seed.get("framework_lever_source") or "")
        gain = -float(measured) if source == "leave_one_out" else float(measured)
        out.append(
            {
                "switch": str(seed["framework_lever"]),
                "gain_pct": round(gain, 4),
                "source": source,
                "variant_name": str(row.get("variant_name") or ""),
                "outcome": str(row.get("outcome") or ""),
            }
        )
    return out


# Curated MTP-capable model class set (needs multi-token-prediction heads).
_ATOM_MTP_CAPABLE_MODEL_CLASSES: frozenset[str] = frozenset(
    {
        "moe_mla",
        "moe_mla_nsa",
    }
)


def _atom_default_grid(
    *,
    model_class: str,
    conc: int,
    isl: int = 0,
    osl: int = 0,
) -> list[GridVariant]:
    """Atom default explore grid, seeded from atom's known perf knobs."""
    mc_l = (model_class or "").strip().lower()
    is_moe = "moe" in mc_l
    is_mla = "mla" in mc_l
    is_fp8 = "fp8" in mc_l or mc_l.endswith("_fp8")
    is_mtp_capable = mc_l in _ATOM_MTP_CAPABLE_MODEL_CLASSES

    variants: list[GridVariant] = []

    def _add(name: str, args: str) -> None:
        """Append a ``default_grid``-provenance variant to the grid."""
        gv = GridVariant(
            name=name,
            extra_server_args=args,
            extra_envs={},
            note="default_grid",
        )
        gv.provenance = "default_grid"  # type: ignore[attr-defined]
        variants.append(gv)

    # ``atom_level_3`` is atom's default, so use ``atom_level_2`` as the off-default contrast.
    _add("atom_level_2", "--level 2")
    _add("atom_prefix_cache", "--enable_prefix_caching")

    if is_fp8:
        _add("atom_kv_fp8", "--kv_cache_dtype fp8")

    if is_moe:
        _add("atom_ep", "--enable-expert-parallel")

    if is_mla:
        _add("atom_dp_attn", "--enable-dp-attention")

    if is_mtp_capable:
        _add(
            "atom_mtp_3",
            "--method mtp --num-speculative-tokens 3",
        )
        _add(
            "atom_mtp_1",
            "--method mtp --num-speculative-tokens 1",
        )

    if conc and conc > 0:
        # Bracket the live concurrency so cudagraph captures the actual decode batch sizes the workload spends most of
        # its time at.
        cg_sizes = sorted({1, 2, 4, 8, 16, int(conc)})
        cg_str = "[" + ",".join(str(s) for s in cg_sizes) + "]"
        _add(
            "atom_cudagraph_bracket",
            f"--cudagraph-capture-sizes {cg_str}",
        )

    return variants


def _xdit_default_grid(*, model_class: str) -> list[GridVariant]:
    """xDiT (diffusion) default explore grid, seeded from the empirical KB."""
    variants: list[GridVariant] = []

    def _add(name: str, *, envs: dict[str, str]) -> None:
        """Append a ``default_grid``-provenance env-only variant."""
        gv = GridVariant(
            name=name,
            extra_server_args="",
            extra_envs=dict(envs),
            note="default_grid",
        )
        gv.provenance = "default_grid"  # type: ignore[attr-defined]
        variants.append(gv)

    # AMD buffer load/store instructions — directionally correct, BF16-safe.
    _add("xdit_buffer_ops", envs={"AMDGCN_USE_BUFFER_OPS": "1"})
    # torch.compile reduce-overhead: expose an explicit on/off contrast.
    _add("xdit_compile_reduce_overhead", envs={"XDIT_USE_TORCH_COMPILE": "1"})
    _add("xdit_no_compile", envs={"XDIT_USE_TORCH_COMPILE": "0"})
    # Confirm the safe attention backend (aiter).
    _add("xdit_attn_aiter", envs={"XDIT_ATTENTION_BACKEND": "aiter"})
    return variants


_CONFIG_REPLAY_PROVENANCE = frozenset({"geak_revalidate"})


def _is_config_replay_variant(variant: Any) -> bool:
    """Whether a variant replays an already-validated config verbatim."""
    return str(getattr(variant, "provenance", "") or "").strip() in _CONFIG_REPLAY_PROVENANCE


class ObservedLaunch(NamedTuple):
    """What the server was running when the round opened, as its launch evidence settled it."""

    config: Mapping[str, Any]
    env: Mapping[str, str]


def observed_launch_from_state(state: Any) -> tuple[dict[str, Any], dict[str, str]]:
    """The running server's resolved config and env, empty unless ``current_best`` observed them.

    A setting the engine rewrote after parsing it is left out of the config: its
    resolved value is not the value a flag would pass, so it reads as unknown
    and a variant touching it runs.
    """
    measurement = getattr(state, "current_best_measurement", None)
    if not isinstance(measurement, dict) or not measurement:
        return {}, {}
    evidence = measurement.get("launch_evidence")
    if not isinstance(evidence, Mapping):
        return {}, {}

    raw_config = evidence.get("observed_server_config")
    config = dict(raw_config) if isinstance(raw_config, Mapping) else {}
    raw_adjusted = evidence.get("engine_adjusted_settings")
    for name in settings_the_engine_rewrote(config, raw_adjusted if isinstance(raw_adjusted, Mapping) else None):
        config.pop(name, None)
    raw_env = evidence.get("observed_server_env")
    # Same shape as a variant's extra_envs, so the two compare directly.
    env = {str(k): str(v) for k, v in raw_env.items()} if isinstance(raw_env, Mapping) else {}
    return config, env


def _variant_control_fields(variant: Any) -> dict[str, Any]:
    """Return non-default remove/unset/replace controls for identity and ledger rows."""
    remove_args = to_str_list(getattr(variant, "remove_args", []))
    unset_envs = to_str_list(getattr(variant, "unset_envs", []))
    args_mode = str(getattr(variant, "args_mode", "append") or "append").strip().lower()
    out: dict[str, Any] = {}
    if remove_args:
        out["remove_args"] = remove_args
    if unset_envs:
        out["unset_envs"] = unset_envs
    if args_mode == "replace":
        out["args_mode"] = "replace"
    return out


def _stamp_canonical_fingerprints(
    grid: list[GridVariant], *, base_remove_args: list[str], base_unset_envs: list[str], base_args_mode: str
) -> None:
    """Attach each variant's effective fingerprint, so no later reader has to recompute it."""
    for gv in grid:
        gv.canonical_fp = effective_fingerprint(  # type: ignore[attr-defined]
            gv.extra_server_args,
            gv.extra_envs,
            controls=_variant_control_fields(gv),
            base_remove_args=base_remove_args,
            base_unset_envs=base_unset_envs,
            base_args_mode=base_args_mode,
        )


def _skip_baseline_noop(name: str, reason: str, skipped: list[dict[str, Any]]) -> None:
    """Record a variant skipped because it launches the server the stack is already running."""
    log.info("explore: skipping baseline-noop variant %s (%s)", name, reason)
    skipped.append({"name": name, "reason": "baseline_noop", "detail": reason})


def _setting_already_holds(proposed: list[str], observed: Any) -> bool:
    """Whether ``observed`` is already the value ``proposed`` asks for."""
    if not proposed:
        # A bare flag asks for the feature on; anything else is a change.
        return observed is True
    if isinstance(observed, (list, tuple)):
        return [str(item) for item in observed] == proposed
    if len(proposed) != 1:
        return False
    text = proposed[0]
    if isinstance(observed, bool):
        return text.lower() == str(observed).lower()
    if isinstance(observed, (int, float)):
        try:
            return float(text) == float(observed)
        except ValueError:
            return False
    if observed is None:
        return False
    return text == str(observed)


def _launch_delta(
    variant_argv: str,
    variant_envs: Mapping[str, str],
    base_argv: str,
    base_envs: Mapping[str, str],
    framework: str,
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """What a variant changes about the baseline launch, in argv settings and envs alike.

    Only the delta is comparable to the running server. The rest is the
    baseline's own request, which materialization, the recipe and the engine all
    rewrite on the way to a launch -- a requested ``--watchdog-timeout 1800``
    resolves to ``3600.0`` -- so weighing it would reject every variant. Those
    rewrites hit base and variant alike, so the delta cancels them.
    """
    base_settings = named_settings(base_argv, framework)
    settings = {
        name: values
        for name, values in named_settings(variant_argv, framework).items()
        if base_settings.get(name) != values
    }
    envs = {name: value for name, value in variant_envs.items() if base_envs.get(name) != value}
    return settings, envs


def _settings_already_hold(settings: Mapping[str, list[str]], observed_config: Mapping[str, Any]) -> bool:
    """Whether every named setting already holds that value in the running server.

    Checked setting by setting rather than as one argv string: the observed side
    is the engine's whole resolved config, which no config's argv can equal.
    """
    return all(
        name in observed_config and _setting_already_holds(values, observed_config[name])
        for name, values in settings.items()
    )


def _envs_already_hold(envs: Mapping[str, str], observed_env: Mapping[str, str]) -> bool:
    """Whether every named env already carries that value in the running server."""
    return all(observed_env.get(name) == value for name, value in envs.items())


def _subtracts_from_the_launch(variant: GridVariant) -> bool:
    """Whether ``variant`` removes something, which the probe's config cannot show."""
    return bool(getattr(variant, "remove_args", None) or getattr(variant, "unset_envs", None))


def _touches_a_multi_value_flag(argv_text: str) -> bool:
    """Whether ``argv_text`` names a list-valued flag, whose ordering the merged config does not preserve."""
    return any(flag in argv_text for flag in MULTI_VALUE_FLAGS)


# Separate from _CONFIG_REPLAY_PROVENANCE (shared with filter_operator_pinned_envs):
# a revalidation re-measures its own config, so it must always run.
_NOOP_FILTER_ALWAYS_RUNS = frozenset({"resume_stack_revalidate"})

#: Names the throwaway variant that materializes the unchanged baseline to diff against.
_NOOP_PROBE_BASE_NAME = "__baseline_noop_probe_base__"


def _is_noop_filter_exempt(variant: GridVariant) -> bool:
    """Whether ``variant`` must skip the baseline-noop probe outright and always run."""
    if _is_config_replay_variant(variant):
        return True
    return str(getattr(variant, "provenance", "") or "").strip() in _NOOP_FILTER_ALWAYS_RUNS


def _materialized_launch(
    base_yaml_path: Path, base_extra_args: str, gv: GridVariant, output_subdir: Path, **build: Any
) -> tuple[str, dict[str, str]]:
    """The server argv and benchmark envs ``run_grid`` would launch *gv* with."""
    out_path = _build_variant_yaml(base_yaml_path, base_extra_args, gv, output_subdir=output_subdir, **build)
    return config_server_argv(out_path).text, config_envs(out_path)


def filter_context_copy_drafts(
    grid: list[GridVariant], *, base_yaml_path: Path, base_extra_args: str, **build: Any
) -> tuple[list[GridVariant], list[tuple[str, str]]]:
    """Drop variants that introduce or change a context-copying speculative-decoding draft (synthetic sessions only).

    Each variant and its base are materialized through ``_build_variant_yaml``, the way ``run_grid`` builds the round,
    so a draft reached through args, replace mode, envs or a server-args env counts. A launch that cannot be
    materialized is judged on the variant's own args and envs against an empty base.

    Returns the variants still to run, plus ``(name, reason)`` for each drop.
    """
    if not grid:
        return [], []
    kept: list[GridVariant] = []
    dropped: list[tuple[str, str]] = []
    with tempfile.TemporaryDirectory(prefix="explore_spec_probe_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        try:
            base = _materialized_launch(
                base_yaml_path, base_extra_args, GridVariant(_NOOP_PROBE_BASE_NAME), tmp_root / "base", **build
            )
        except (OSError, ValueError, yaml.YAMLError, ConfigUnreadable) as exc:
            log.warning("explore: could not materialize the base to read its speculative decoding (%s)", exc)
            base = ("", {})
        for idx, gv in enumerate(grid):
            against = base
            try:
                launch = _materialized_launch(base_yaml_path, base_extra_args, gv, tmp_root / f"v{idx}", **build)
            except (OSError, ValueError, yaml.YAMLError, ConfigUnreadable) as exc:
                log.warning("explore: could not materialize %s to read its speculative decoding (%s)", gv.name, exc)
                launch = (str(gv.extra_server_args or ""), {str(k): str(v) for k, v in (gv.extra_envs or {}).items()})
                against = ("", {})
            reason = context_copy_refusal(against[0], against[1], launch[0], launch[1])
            if reason:
                dropped.append((gv.name, reason))
            else:
                kept.append(gv)
    return kept, dropped


def _only_restates_the_baseline(gv: GridVariant, base_argv: str, base_envs: dict[str, str], framework: str) -> bool:
    """Whether a variant with no launch delta is one the baseline launch already holds.

    A change the merged config hides leaves no delta either, so only a request
    that itself restates the baseline marks the variant as a noop -- and only in
    ``append`` mode, where the request is all the variant adds.
    """
    return gv.args_mode == "append" and restates_the_baseline(
        gv.extra_server_args, gv.extra_envs, base_argv, base_envs, framework
    )


def filter_baseline_noop_variants(
    grid: list[GridVariant],
    *,
    framework: str,
    base_yaml_path: Path,
    base_extra_args: str,
    observed_server_config: Mapping[str, Any] | None = None,
    observed_server_env: Mapping[str, str] | None = None,
    **build: Any,
) -> tuple[list[GridVariant], list[tuple[str, str]]]:
    """Drop variants that would launch the server the stack is already running.

    The baseline and each variant are materialized through ``_build_variant_yaml``
    into throwaway configs, and which rule applies depends on whether the computed
    settings and env delta of the variant against the baseline (``_launch_delta``)
    is empty. That delta lists only what the variant sets differently, so a
    ``"replace"`` variant that drops a baseline flag can still have an empty one:

    1. Empty delta: the variant is dropped only when its ``args_mode`` is
       ``"append"`` and its own request names only settings and envs the baseline
       launch already holds (``restates_the_baseline``). This rule reads the
       baseline config, not the observed launch. Every other empty-delta variant
       -- a ``"replace"`` request, an empty request, or a change the merged config
       hides -- is kept without consulting the observed launch.
    2. Non-empty delta: only what the variant *changes* about the baseline is
       judged, those settings against ``observed_server_config`` -- the engine's
       own resolved settings -- and those envs against ``observed_server_env``.
       The variant is dropped when both halves are already in effect.

    A variant that removes something, or names a list-valued flag, is exempt:
    neither is visible here, so both would read as a restatement of the stack.

    Returns the variants still to run, plus ``(name, reason)`` for each drop.
    Filters nothing when the stack's launch was never observed.
    """
    observed_config = dict(observed_server_config or {})
    if not observed_config or not grid:
        return list(grid), []
    observed_env = dict(observed_server_env or {})
    kept: list[GridVariant] = []
    dropped: list[tuple[str, str]] = []
    with tempfile.TemporaryDirectory(prefix="explore_noop_probe_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        try:
            base_argv, base_envs = _materialized_launch(
                base_yaml_path, base_extra_args, GridVariant(_NOOP_PROBE_BASE_NAME), tmp_root / "base", **build
            )
        except (OSError, ValueError, yaml.YAMLError) as exc:
            log.warning(
                "explore: baseline-noop probe could not materialize the baseline (%s); keeping every variant", exc
            )
            return list(grid), []
        for idx, gv in enumerate(grid):
            if _is_noop_filter_exempt(gv) or _subtracts_from_the_launch(gv):
                kept.append(gv)
                continue
            if _touches_a_multi_value_flag(str(getattr(gv, "extra_server_args", "") or "")):
                kept.append(gv)
                continue
            try:
                variant_argv, variant_envs = _materialized_launch(
                    base_yaml_path, base_extra_args, gv, tmp_root / f"v{idx}", **build
                )
            except (OSError, ValueError, yaml.YAMLError) as exc:
                log.warning("explore: baseline-noop probe failed for variant %s (%s); keeping it", gv.name, exc)
                kept.append(gv)
                continue
            delta_settings, delta_envs = _launch_delta(variant_argv, variant_envs, base_argv, base_envs, framework)
            if not delta_settings and not delta_envs:
                if _only_restates_the_baseline(gv, base_argv, base_envs, framework):
                    dropped.append((gv.name, "names only settings the baseline launch already holds (baseline noop)"))
                    continue
                kept.append(gv)
                continue
            if _settings_already_hold(delta_settings, observed_config) and _envs_already_hold(delta_envs, observed_env):
                dropped.append((gv.name, "merged launch already active in the observed server (baseline noop)"))
                continue
            kept.append(gv)
    return kept, dropped


def filter_operator_pinned_envs(
    grid: list[GridVariant],
    baseline_envs: dict[str, Any] | None,
) -> tuple[list[GridVariant], list[tuple[str, str]]]:
    """Drop variants that overwrite an env the operator pinned in the baseline."""
    pinned = {str(k).strip().upper() for k in (baseline_envs or {}) if str(k).strip()}
    if not pinned:
        return list(grid), []

    kept: list[GridVariant] = []
    dropped: list[tuple[str, str]] = []
    for gv in grid:
        clash = sorted(
            key for key in (str(k).strip().upper() for k in (getattr(gv, "extra_envs", None) or {})) if key in pinned
        )
        # A replay reproduces a config another component already measured, so it carries the pinned values verbatim by
        # construction.
        if clash and not _is_config_replay_variant(gv):
            dropped.append(
                (
                    str(getattr(gv, "name", "?")),
                    f"overwrites baseline-pinned env {', '.join(clash)}, which would make "
                    "its number incomparable to the baseline",
                )
            )
            continue
        kept.append(gv)
    return kept, dropped


def _default_grid_for_framework(
    framework: str,
    *,
    model_class: str,
    conc: int = 0,
    isl: int = 0,
    osl: int = 0,
) -> list[GridVariant]:
    """Framework-keyed default grid dispatch."""
    fw = (framework or "").strip().lower()
    if fw == "atom":
        return _atom_default_grid(
            model_class=model_class,
            conc=conc,
            isl=isl,
            osl=osl,
        )
    if fw == "xdit":
        return _xdit_default_grid(model_class=model_class)
    return []


def _seed_lever_variants(grid_payload: Any, lever_payload: Any) -> Any:
    """Put this round's framework-rewrite lever variants in front of the grid.

    A lever already named by the grid is not seeded twice; the fresh ones go
    first so a round that runs out of budget still measures what the rewrite
    was for.
    """
    if not lever_payload:
        return grid_payload
    existing_names = {str(v.get("name") or "") for v in grid_payload if isinstance(v, dict)}
    fresh = [v for v in lever_payload if str(v.get("name") or "") not in existing_names]
    if fresh:
        log.info(
            "explore: seeding %d framework-rewrite lever variant(s) for attribution",
            len(fresh),
        )
        return fresh + list(grid_payload)
    return grid_payload


def _seed_hint(params: Mapping[str, Any], yaml_envs: Mapping[str, str], key: str) -> Any:
    """A workload size the seed grid brackets: the params, then the launch yaml, then the environment."""
    return params.get(key.lower()) or yaml_envs.get(key) or os.environ.get(key) or 0


def _seed_grid_payload(
    params: Mapping[str, Any],
    *,
    framework: str,
    yaml_envs: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Build the framework's programmatic seed grid for a round the LLM left empty.

    A round with no variants of its own falls through to this rather than
    failing the task; the concurrency and model class come from the params,
    the launch yaml and the environment, in that order.
    """
    seed_model_class = str(params.get("model_class") or "").strip() or os.environ.get("MODEL_CLASS", "").strip()
    seed_conc = 0
    try:
        seed_conc = int(_seed_hint(params, yaml_envs, "CONC"))
    except (TypeError, ValueError):
        seed_conc = 0
    seed_isl = 0
    seed_osl = 0
    try:
        seed_isl = int(_seed_hint(params, yaml_envs, "ISL"))
        seed_osl = int(_seed_hint(params, yaml_envs, "OSL"))
    except (TypeError, ValueError):
        # Non-integer seed hint; fall back to the default grid below.
        pass
    seed = _default_grid_for_framework(
        framework,
        model_class=seed_model_class,
        conc=seed_conc,
        isl=seed_isl,
        osl=seed_osl,
    )
    if seed:
        log.info(
            "explore: empty grid for framework=%s; falling through "
            "to %d default_grid seed variants "
            "(model_class=%r conc=%d)",
            framework or "?",
            len(seed),
            seed_model_class or "?",
            seed_conc,
        )
        return [
            {
                "name": v.name,
                "extra_server_args": v.extra_server_args,
                "extra_envs": dict(v.extra_envs or {}),
                "note": v.note or "default_grid",
                "provenance": getattr(v, "provenance", "default_grid"),
            }
            for v in seed
        ]
    return []
