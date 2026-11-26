# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""EXPLORE variants dropped on their materialized launch: baseline restatements, context-copy drafts, pinned envs."""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import yaml

from hyperloom_common.coerce import to_str_list
from hyperloom.common.launch_log_evidence import settings_the_engine_rewrote
from ..framework_change._spec_decode_context_copy import CONTEXT_COPY_REFUSAL, context_copy_refusal
from ..grid._explore_variants import variant_fingerprint
from ..grid._grid_base import GridVariant
from ..grid._grid_runner import GridRecipe, _build_variant_yaml
from ..serving._server_argv import ConfigUnreadable, config_server_argv
from ..workload._launch_restatement import Launch, config_envs, launch_change, read_launch
from ._remeasure import _is_config_replay_variant
from .spec_decode_golden import GoldenLookup, synthetic_draft_refusal

log = logging.getLogger(__name__)


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
    raw_env = evidence.get("requested_server_env")
    # Same shape as a variant's extra_envs, so the two compare directly.
    env = {str(k): str(v) for k, v in raw_env.items()} if isinstance(raw_env, Mapping) else {}
    return config, env


#: Names the throwaway variant that materializes the unchanged baseline to diff against.
_NOOP_PROBE_BASE_NAME = "__baseline_noop_probe_base__"


def _materialized_launch(recipe: GridRecipe, gv: GridVariant, output_subdir: Path) -> tuple[str, dict[str, str]]:
    """The server argv and benchmark envs ``run_grid`` would launch *gv* with."""
    out_path = _build_variant_yaml(recipe, gv, output_subdir=output_subdir)
    return config_server_argv(out_path).text, config_envs(out_path)


#: What reading a probe launch can raise; a launch that cannot be read is never judged unchanged.
_UNREADABLE_LAUNCH = (OSError, ValueError, yaml.YAMLError, ConfigUnreadable)


def _probe_launch(recipe: GridRecipe, gv: GridVariant, output_subdir: Path) -> Launch:
    """The whole launch ``run_grid`` would start *gv* with on ``recipe``'s base."""
    out_path = _build_variant_yaml(recipe, gv, output_subdir=output_subdir)
    return read_launch(out_path, [*to_str_list(recipe.base.unset_envs), *to_str_list(gv.unset_envs)])


#: Why a candidate's speculative decoding is refused, as ``(reason code, detail)``, judged on its materialized launch
#: and its base's (argv, envs each); ``None`` when it is not.
_DraftJudge = Callable[[str, Mapping[str, Any], str, Mapping[str, Any]], tuple[str, str] | None]
#: The variants a draft filter keeps, and ``(name, reason code, detail)`` for each it drops.
_DraftFilter = tuple[list[GridVariant], list[tuple[str, str, str]]]


def _filter_drafts(grid: list[GridVariant], recipe: GridRecipe, judge: _DraftJudge) -> _DraftFilter:
    """``grid`` split into the variants ``judge`` passes and ``(name, reason code, detail)`` for each it refuses.

    Each variant and its base are materialized through ``_build_variant_yaml``, the way ``run_grid`` builds the round,
    so a draft reached through args, replace mode, envs or a server-args env counts. A launch that cannot be
    materialized is judged on the variant's own args and envs against an empty base.
    """
    if not grid:
        return [], []
    kept: list[GridVariant] = []
    dropped: list[tuple[str, str, str]] = []
    with tempfile.TemporaryDirectory(prefix="explore_spec_probe_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        try:
            base = _materialized_launch(recipe, GridVariant(_NOOP_PROBE_BASE_NAME), tmp_root / "base")
        except _UNREADABLE_LAUNCH as exc:
            log.warning("explore: could not materialize the base to read its speculative decoding (%s)", exc)
            base = ("", {})
        for idx, gv in enumerate(grid):
            against = base
            try:
                launch = _materialized_launch(recipe, gv, tmp_root / f"v{idx}")
            except _UNREADABLE_LAUNCH as exc:
                log.warning("explore: could not materialize %s to read its speculative decoding (%s)", gv.name, exc)
                launch = (str(gv.extra_server_args or ""), {str(k): str(v) for k, v in (gv.extra_envs or {}).items()})
                against = ("", {})
            refused = judge(against[0], against[1], launch[0], launch[1])
            if refused:
                dropped.append((gv.name, *refused))
            else:
                kept.append(gv)
    return kept, dropped


def _context_copy(
    base_argv: str, base_envs: Mapping[str, Any], argv: str, envs: Mapping[str, Any]
) -> tuple[str, str] | None:
    reason = context_copy_refusal(base_argv, base_envs, argv, envs)
    return (CONTEXT_COPY_REFUSAL, reason) if reason else None


def filter_context_copy_drafts(
    grid: list[GridVariant], *, recipe: GridRecipe
) -> tuple[list[GridVariant], list[tuple[str, str]]]:
    """Drop variants that introduce or change a context-copying speculative-decoding draft (synthetic sessions only).

    Returns the variants still to run, plus ``(name, reason)`` for each drop.
    """
    kept, dropped = _filter_drafts(grid, recipe, _context_copy)
    return kept, [(name, detail) for name, _, detail in dropped]


def filter_synthetic_drafts(grid: list[GridVariant], *, recipe: GridRecipe, lookup: GoldenLookup) -> _DraftFilter:
    """Drop variants whose speculative decoding the synthetic benchmark refuses (synthetic sessions only): a
    context-copying draft they introduce or change, or a draft they turn on at another acceptance than InferenceX
    records for it (:func:`~.spec_decode_golden.synthetic_draft_refusal`).

    Returns the variants still to run, plus ``(name, reason code, detail)`` for each drop.
    """
    return _filter_drafts(
        grid,
        recipe,
        lambda base_argv, base_envs, argv, envs: synthetic_draft_refusal(base_argv, base_envs, argv, envs, lookup),
    )


def filter_baseline_noop_variants(
    grid: list[GridVariant],
    *,
    observed: tuple[Mapping[str, Any], Mapping[str, str]],
    recipe: GridRecipe,
) -> tuple[list[GridVariant], list[tuple[str, str]]]:
    """Drop variants that would launch the server the stack is already running.

    The baseline and each variant are materialized through ``_build_variant_yaml`` and read back whole, so a
    removal, a replaced argv, an overlay or an AgentX launch override shows as the change it is; ``launch_change``
    then discounts what the observed server already resolved. A variant whose launch cannot be read always runs;
    a stack revalidation round never reaches this filter.

    Returns the variants still to run, plus ``(name, reason)`` for each drop.
    """
    if not grid:
        return [], []
    kept: list[GridVariant] = []
    dropped: list[tuple[str, str]] = []
    with tempfile.TemporaryDirectory(prefix="explore_noop_probe_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        try:
            base = _probe_launch(recipe, GridVariant(_NOOP_PROBE_BASE_NAME), tmp_root / "base")
        except _UNREADABLE_LAUNCH as exc:
            log.warning(
                "explore: baseline-noop probe could not read the baseline launch (%s); keeping every variant", exc
            )
            return list(grid), []
        for idx, gv in enumerate(grid):
            try:
                launch = _probe_launch(recipe, gv, tmp_root / f"v{idx}")
            except _UNREADABLE_LAUNCH as exc:
                log.warning("explore: baseline-noop probe could not read variant %s (%s); keeping it", gv.name, exc)
                kept.append(gv)
                continue
            if launch_change(base, launch, observed):
                kept.append(gv)
            else:
                dropped.append((gv.name, "launches the server the stack already runs (baseline noop)"))
    return kept, dropped


def restates_the_baseline_launch(state: Any, args: str, envs: Mapping[str, Any]) -> bool:
    """Whether ``args`` and ``envs`` laid onto the baseline's materialized config start the baseline's own server.

    Judged as a variant on that config is. The observed launch discounts settings only when ``current_best`` was
    observed running that very config.
    """
    config_path = str(getattr(state, "baseline_config_path", "") or "")
    if not config_path:
        return False
    measurement = getattr(state, "current_best_measurement", None)
    evidence = measurement.get("launch_evidence") if isinstance(measurement, Mapping) else None
    observed_here = isinstance(evidence, Mapping) and evidence.get("materialized_config_path") == config_path
    _, dropped = filter_baseline_noop_variants(
        [GridVariant("warm_replay", str(args or ""), {str(k): str(v) for k, v in envs.items()})],
        observed=observed_launch_from_state(state) if observed_here else ({}, {}),
        recipe=GridRecipe(base_yaml_path=Path(config_path)),
    )
    return bool(dropped)


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


def launch_identity(recipe: GridRecipe, gv: GridVariant) -> str:
    """The server ``gv`` would launch on ``recipe``; its fingerprint when that cannot be read."""
    with tempfile.TemporaryDirectory(prefix="explore_launch_") as tmp_dir:
        try:
            return _probe_launch(recipe, gv, Path(tmp_dir)).identity
        except _UNREADABLE_LAUNCH as exc:
            log.warning("explore: could not read the launch of %s (%s); deduping on its fingerprint", gv.name, exc)
            return variant_fingerprint(gv)
