# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Measurement points: the YAML each test is run from, generated from the canonical configuration (design 4.17).

A point is the canonical measurement configuration with the recipe's launch settings, the concurrency level and the
``RUN_EVAL`` switch the request names, rendered by the same function the single-lineage flow materializes with. The
result is read back and held to the request before it is handed out: a launch variable the renderer dropped, a
``RUN_EVAL`` or concurrency that is not the requested one, or a surface that differs from the canonical one refuses
the request instead of measuring something else.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from hyperloom.contracts.measurement import TestKind, TestRequest
from hyperloom.contracts.recipe import LaunchConfig, Recipe, RecipeError
from hyperloom.inference_optimizer import framework_registry
from hyperloom.measure.compile import CompiledEnvironment, EnvironmentCompiler
from hyperloom.measure.surface import (
    MeasurementSurface,
    benchmark_envs,
    check_launch_envs,
    check_point,
    check_surface,
    extract_surface,
    point_values,
)
from hyperloom.orchestrator.actions.executors import (
    default_baseline_config,
    materialize_config_with_envs,
)


class PointError(RecipeError):
    """A point cannot be generated the way the request describes it."""


@dataclass(frozen=True)
class CanonicalMeasurement:
    """The active canonical measurement configuration: a base YAML, how it is rendered, and its surface.

    ``materialize`` holds the keyword arguments the base is rendered with (model path, GPU type, InferenceX path,
    benchmark script), the same for every test of the session.
    """

    config_path: Path
    surface: MeasurementSurface
    primary_concurrency: int
    materialize: Mapping[str, Any] = field(default_factory=dict)


def _render(canonical: CanonicalMeasurement, out_dir: Path, name: str, **overrides: Any) -> dict[str, Any]:
    path = materialize_config_with_envs(
        canonical.config_path, out_dir, out_name=name, **{**canonical.materialize, **overrides}
    )
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _reference(canonical: CanonicalMeasurement, pins: Mapping[str, str]) -> dict[str, Any]:
    """The canonical configuration rendered with ``pins``, which a generated point is compared with."""

    scratch = Path(tempfile.mkdtemp(prefix="hl-ref-"))
    try:
        return _render(canonical, scratch, "reference.yaml", extra_envs=pins)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def default_canonical(
    out_dir: Path, *, primary_concurrency: int, config_path: Path | None = None, **materialize: Any
) -> CanonicalMeasurement:
    """Render the shipped baseline configuration at the primary operating point and take its surface."""

    base = config_path or default_baseline_config()
    canonical = CanonicalMeasurement(base, MeasurementSurface({}), primary_concurrency, materialize)
    config = _render(
        canonical, out_dir, "canonical.yaml", extra_envs={"CONC": str(primary_concurrency), "RUN_EVAL": "true"}
    )
    return CanonicalMeasurement(base, extract_surface(config), primary_concurrency, materialize)


def _concurrencies(requested: Sequence[int], canonical: CanonicalMeasurement) -> tuple[int, ...]:
    levels = tuple(requested) or (canonical.primary_concurrency,)
    if any(isinstance(c, bool) or not isinstance(c, int) or c <= 0 for c in levels) or len(set(levels)) != len(levels):
        raise PointError(f"concurrency must be distinct positive integers, got {list(levels)}")
    return levels


def _check_launch_present(config: Mapping[str, Any], launch: LaunchConfig, envs: Mapping[str, str]) -> None:
    """Every variable and argument the recipe launches with must be in the YAML, or the test would not use it."""

    present = benchmark_envs(config)
    missing = sorted(name for name, value in {**launch.envs, **envs}.items() if str(present.get(name)) != value)
    if missing:
        raise PointError(f"the renderer dropped or changed launch variables {', '.join(missing)}")
    framework = str(config.get("benchmark", {}).get("framework") or "").lower()
    spec = framework_registry.FRAMEWORKS.get(framework)
    if spec is None:
        return
    carried = str(present.get(spec.extra_args_env, ""))
    flags = {token.split("=", 1)[0] for token in launch.server_args.split() if token.startswith("--")}
    absent = sorted(flag for flag in flags if flag not in carried)
    if absent:
        raise PointError(f"the renderer dropped launch arguments {', '.join(absent)}")


def generate_points(
    environment: CompiledEnvironment,
    canonical: CanonicalMeasurement,
    out_dir: Path,
    *,
    concurrency: Sequence[int] = (),
    accuracy: bool = False,
) -> tuple[Path, ...]:
    """One YAML per concurrency level, all sharing one service, each checked against its request.

    Raises :class:`~hyperloom.measure.surface.SurfaceViolation` when the recipe or the rendered point changes the
    measurement surface, and :class:`PointError` when the rendered point does not carry the recipe's launch settings.
    """

    launch = environment.recipe.launch
    check_launch_envs([*launch.envs, *environment.envs])
    out_dir.mkdir(parents=True, exist_ok=True)
    points = []
    for index, level in enumerate(_concurrencies(concurrency, canonical)):
        run_eval = "true" if accuracy else "false"
        pins = {"CONC": str(level), "RUN_EVAL": run_eval}
        reference = _reference(canonical, pins)
        name = f"point-{index}-c{level}.yaml"
        config = _render(
            canonical,
            out_dir,
            name,
            extra_server_args=launch.server_args,
            extra_envs={**launch.envs, **environment.envs, **pins},
        )
        check_surface(config, canonical.surface)
        check_point(config, concurrency=level, accuracy=accuracy, reference=point_values(reference))
        _check_launch_present(config, launch, environment.envs)
        points.append(out_dir / name)
    return tuple(points)


#: Surface fields a profile point may change: the profile run lowers its output length to fit the capture window.
_PROFILE_SURFACE_EXEMPT = ("envs.OSL",)
#: What the profile point takes from the canonical rendering, so the profile configuration's own defaults for the
#: workload (a different input length, another TP) cannot make it profile a workload the tests do not run.
_PROFILE_PINNED = ("TP", "EP", "ISL", "RANDOM_RANGE_RATIO", "MAX_MODEL_LEN")


def _profile_canonical(canonical: CanonicalMeasurement) -> CanonicalMeasurement:
    """The canonical measurement rendered from the shipped profile configuration of its framework."""

    framework = canonical.surface.fields.get("benchmark.framework") or framework_registry.DEFAULT_FRAMEWORK
    name = framework_registry.shipped_config_name("profile", str(framework))
    path = default_baseline_config().parent / name
    if not path.is_file():
        raise PointError(f"there is no shipped profile configuration for framework {framework!r}")
    return CanonicalMeasurement(path, canonical.surface, canonical.primary_concurrency, canonical.materialize)


def generate_profile_point(
    environment: CompiledEnvironment, canonical: CanonicalMeasurement, out_dir: Path
) -> tuple[Path, ...]:
    """The one point of a ``profile`` test: the recipe's launch on the shipped profile configuration.

    The profile configuration turns the torch profiler on (``PROFILE=1``, ``PROFILE_EXTRA_BODY``, the framework's
    capture flags) and the renderer sizes the capture window from the workload, as the single-lineage profile does.
    The point runs at the primary concurrency without accuracy; apart from the output length the measurement surface
    must still be the canonical one.
    """

    launch = environment.recipe.launch
    check_launch_envs([*launch.envs, *environment.envs])
    out_dir.mkdir(parents=True, exist_ok=True)
    level = canonical.primary_concurrency
    pins = {"CONC": str(level), "RUN_EVAL": "false"}
    reference = benchmark_envs(_reference(canonical, pins))
    workload = {name: str(reference[name]) for name in _PROFILE_PINNED if reference.get(name) is not None}
    config = _render(
        _profile_canonical(canonical),
        out_dir,
        "point-profile.yaml",
        extra_server_args=launch.server_args,
        extra_envs={**launch.envs, **environment.envs, **workload, **pins},
    )
    check_surface(config, canonical.surface, ignore=_PROFILE_SURFACE_EXEMPT)
    check_point(config, concurrency=level, accuracy=False, reference={})
    _check_launch_present(config, launch, environment.envs)
    return (out_dir / "point-profile.yaml",)


@dataclass(frozen=True)
class AcceptedMeasurement:
    """What acceptance produces for a test request: its environment and the points it runs."""

    environment: CompiledEnvironment
    points: tuple[Path, ...]


def accept_measurement(
    request: TestRequest,
    parent_recipe: Recipe,
    compiler: EnvironmentCompiler,
    canonical: CanonicalMeasurement,
    out_dir: Path,
) -> AcceptedMeasurement:
    """Acceptance steps 2 to 4 (design 4.17): compile, generate the points, check the measurement surface.

    ``parent_recipe`` is the recipe of ``request.parent_commit``. Any refusal is a
    :class:`~hyperloom.contracts.recipe.RecipeError`, raised before the request is queued.
    """

    environment = compiler.compile(parent_recipe, request.delta)
    if request.kind is TestKind.PROFILE:
        return AcceptedMeasurement(environment, generate_profile_point(environment, canonical, out_dir))
    points = generate_points(
        environment, canonical, out_dir, concurrency=request.concurrency, accuracy=request.accuracy
    )
    return AcceptedMeasurement(environment, points)


__all__ = [
    "AcceptedMeasurement",
    "CanonicalMeasurement",
    "PointError",
    "accept_measurement",
    "default_canonical",
    "generate_points",
    "generate_profile_point",
]
