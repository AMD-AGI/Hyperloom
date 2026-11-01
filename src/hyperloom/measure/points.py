# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Measurement points: the YAML each test is run from, generated from the canonical configuration (design 4.17).

A point is the canonical measurement configuration with the recipe's launch settings, the concurrency level and the
``RUN_EVAL`` switch the request names, rendered by the same function the single-lineage flow materializes with. An
AgentX canonical (``benchmark.agentx`` enabled) is rendered as that flow renders it: the launch goes into
``agentx.launch_overrides``, the concurrency into ``benchmark.envs.CONC``, and there is no evaluation round. The
result is read back and held to the request before it is handed out: a launch variable the renderer dropped, a
``RUN_EVAL`` or concurrency that is not the requested one, or a surface that differs from the canonical one refuses
the request instead of measuring something else.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from hyperloom.common.benchmark_config import is_magpie_agentx
from hyperloom.common.env_safety import filter_untrusted_env_mapping, is_allowed_variant_env_key
from hyperloom.common.outside_facts import (
    ENV_CONC,
    ENV_EP,
    ENV_ISL,
    ENV_MAX_MODEL_LEN,
    ENV_OSL,
    ENV_RANDOM_RANGE_RATIO,
    ENV_RUN_EVAL,
    ENV_TP,
)
from hyperloom.contracts.measurement import TestKind, TestRequest
from hyperloom.contracts.recipe import LaunchConfig, Recipe, RecipeError
from hyperloom.inference_optimizer import framework_registry
from hyperloom.measure.compile import CompiledEnvironment, EnvironmentCompiler
from hyperloom.measure.eval_context import check_eval_context
from hyperloom.measure.magpie_launchers import install_launchers
from hyperloom.measure.long_context.plan import LongContextPlan
from hyperloom.measure.short_prompts.plan import ShortPromptPlan
from hyperloom.measure.surface import (
    SCRIPT_FIELD,
    MeasurementSurface,
    benchmark_envs,
    check_launch_envs,
    check_point,
    check_surface,
    extract_surface,
    point_values,
)
from hyperloom.orchestrator.actions.executors import (
    BenchmarkTarget,
    LaunchLayer,
    MaterializeOptions,
    default_baseline_config,
    materialize_benchmark_config,
)
from hyperloom.orchestrator.actions.executors.benchmark_backend import resolve_backend_name
from hyperloom.orchestrator.actions.executors.magpie_agentx import enable_profile, native_config

_UTF8 = "utf-8"


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

    @property
    def agentx(self) -> bool:
        """Whether the base configuration is a Magpie AgentX benchmark, the main flow's own test for its workload."""

        return native_config(self.config_path)


def _render(canonical: CanonicalMeasurement, out_dir: Path, name: str, **overrides: Any) -> dict[str, Any]:
    """The base configuration rendered as the single-lineage flow renders it, at the ``CONC`` the pins name.

    An AgentX benchmark carries its concurrency in ``benchmark.envs``, not in the server environment the pins feed, so
    ``CONC`` is taken out of them and written there. The AgentX materializer hands every variable it is given to the
    server, so the variables are held to the allowlist the InferenceX renderer applies first; one that does not pass
    is missing from the point, and :func:`_check_launch_present` refuses it. ``inferencex_path`` names the InferenceX
    checkout the point runs from; the renderer reads it from ``INFERENCEX_PATH`` alone, so it is written over that.
    """

    options = {**canonical.materialize, **overrides, "out_name": name}
    inferencex = str(options.pop("inferencex_path", None) or "").strip()
    concurrency = None
    if canonical.agentx:
        pins = dict(options.get("extra_envs") or {})
        concurrency = pins.pop(ENV_CONC)
        options["extra_envs"], _ = filter_untrusted_env_mapping(pins, allow_predicate=is_allowed_variant_env_key)
    path = materialize_benchmark_config(canonical.config_path, out_dir, **_materialize_args(options))
    with path.open(encoding=_UTF8) as handle:
        config = yaml.safe_load(handle) or {}
    if concurrency is not None:
        config["benchmark"]["envs"][ENV_CONC] = int(concurrency)
    if inferencex:
        config["benchmark"]["inferencex_path"] = inferencex
    if concurrency is not None or inferencex:
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding=_UTF8)
    return config


def _materialize_args(options: Mapping[str, Any]) -> dict[str, Any]:
    """``options``, keyword by keyword, as the launch, target and options values the renderer takes them in."""

    left = dict(options)
    split = {
        name: kind(**{item.name: left.pop(item.name) for item in fields(kind) if item.name in left})
        for name, kind in (("launch", LaunchLayer), ("target", BenchmarkTarget), ("options", MaterializeOptions))
    }
    if left:
        raise TypeError(f"the renderer takes no {', '.join(sorted(left))}")
    return split


def _reference(canonical: CanonicalMeasurement, pins: Mapping[str, str]) -> dict[str, Any]:
    """The canonical configuration rendered with ``pins``, which a generated point is compared with."""

    scratch = Path(tempfile.mkdtemp(prefix="hl-ref-"))
    try:
        return _render(canonical, scratch, "reference.yaml", extra_envs=pins)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


#: The file :func:`default_canonical` renders the canonical measurement to, in its ``out_dir``.
CANONICAL_FILE = "canonical.yaml"
#: The round's shape ``run_grid`` takes from the process environment over a point's own ``envs`` (the old flow's CLI
#: exports them): a runner clears them, or a value left in its process measures every point at that value.
AMBIENT_SHAPE_ENVS = (ENV_ISL, ENV_OSL, ENV_MAX_MODEL_LEN, ENV_TP, ENV_CONC)


def default_canonical(
    out_dir: Path, *, primary_concurrency: int, config_path: Path | None = None, **materialize: Any
) -> CanonicalMeasurement:
    """Render the shipped baseline configuration at the primary operating point and take its surface.

    The evaluation settings are part of the surface, so an InferenceX canonical is rendered with ``RUN_EVAL`` on; an
    AgentX one has no evaluation round, and the main flow refuses it.
    """

    base = config_path or default_baseline_config()
    canonical = CanonicalMeasurement(base, MeasurementSurface({}), primary_concurrency, materialize)
    evaluation = {} if canonical.agentx else {ENV_RUN_EVAL: "true"}
    config = _render(canonical, out_dir, CANONICAL_FILE, extra_envs={ENV_CONC: str(primary_concurrency), **evaluation})
    surface = _canonical_surface(config, out_dir / CANONICAL_FILE)
    return CanonicalMeasurement(base, surface, primary_concurrency, materialize)


def _canonical_surface(config: Mapping[str, Any], point: Path) -> MeasurementSurface:
    """The surface of the canonical ``config`` (rendered at ``point``), its script hashed as the tests will find it.

    Under Magpie the script a test runs is Magpie's own launcher, which Magpie copies into the InferenceX checkout when
    its first benchmark starts; a checkout whose pinned commit does not track the launcher has none until then. A surface
    taken before that records no script hash, and every test, which reports the hash of the copy, would differ from it.
    So the launcher is put there first, by the function the runner and the local backend use before they hash a test's
    point, and a canonical that still has no hash for a launcher it names is refused: ``None`` is never recorded as the
    canonical value of a field a test will report. A checkout with no ``benchmarks`` directory has nowhere to put the
    launcher, for a test either, and is left to the host check that refuses a tree without the benchmark library.
    """

    under_magpie = resolve_backend_name() == "magpie"
    if under_magpie:
        install_launchers([point])
    surface = extract_surface(config)
    bench = config.get("benchmark") or {}
    if under_magpie and _launcher_missing(bench, surface):
        raise PointError(
            f"the canonical measurement runs {bench['benchmark_script']} from the InferenceX checkout "
            f"{bench['inferencex_path']}, which has no such file and Magpie has none to put there: a test would report "
            "a script hash the canonical surface does not hold. Point MAGPIE_PATH at a Magpie that ships the launcher, "
            "or use a checkout that tracks it"
        )
    return surface


def _launcher_missing(bench: Mapping[str, Any], surface: MeasurementSurface) -> bool:
    """Whether ``bench`` names a launcher and a checkout it could be put in, and the surface has no hash for it."""

    checkout = bench.get("inferencex_path")
    placeable = bool(bench.get("benchmark_script") and checkout) and (Path(str(checkout)) / "benchmarks").is_dir()
    return placeable and surface.fields.get(SCRIPT_FIELD) is None


def _concurrencies(requested: Sequence[int], canonical: CanonicalMeasurement) -> tuple[int, ...]:
    levels = tuple(requested) or (canonical.primary_concurrency,)
    if any(isinstance(c, bool) or not isinstance(c, int) or c <= 0 for c in levels) or len(set(levels)) != len(levels):
        raise PointError(f"concurrency must be distinct positive integers, got {list(levels)}")
    return levels


def launch_carriers(config: Mapping[str, Any]) -> tuple[Mapping[str, Any], str | None]:
    """Where a rendered point carries the recipe's launch: the variables, and the text holding its server arguments.

    The text is ``None`` for a framework the registry does not know. An AgentX point carries both in
    ``agentx.launch_overrides``; any other in ``benchmark.envs``.
    """

    benchmark = config.get("benchmark") or {}
    if is_magpie_agentx(benchmark):
        overrides = benchmark["agentx"]["launch_overrides"]
        return overrides.get("env") or {}, " ".join(overrides.get("append_args") or [])
    present = benchmark_envs(config)
    spec = framework_registry.FRAMEWORKS.get(str(benchmark.get("framework") or "").lower())
    return present, None if spec is None else str(present.get(spec.extra_args_env, ""))


def _check_removals(config: Mapping[str, Any], launch: LaunchConfig) -> None:
    """Every option the recipe removes from the launch the framework resolves must be among an AgentX point's
    removals, or Magpie would launch with it. Any other point carries its whole launch in the recipe's own arguments,
    so a removal it could not make is refused, not left on."""

    if not launch.remove_args:
        return
    names = ", ".join(launch.remove_args)
    benchmark = config.get("benchmark") or {}
    if not is_magpie_agentx(benchmark):
        raise PointError(f"only a Magpie AgentX point removes options of the launch it resolves, so {names} would stay")
    removed = set(benchmark["agentx"]["launch_overrides"].get("remove_args") or ())
    kept = sorted(name for name in launch.remove_args if name not in removed)
    if kept:
        raise PointError(f"the renderer did not remove launch arguments {', '.join(kept)}")


def _check_launch_present(config: Mapping[str, Any], launch: LaunchConfig, envs: Mapping[str, str]) -> None:
    """Every variable and argument the recipe launches with must be in the YAML, or the test would not use it, and
    every option it removes must be removed (:func:`_check_removals`)."""

    _check_removals(config, launch)
    present, carried = launch_carriers(config)
    missing = sorted(name for name, value in {**launch.envs, **envs}.items() if str(present.get(name)) != value)
    if missing:
        raise PointError(f"the renderer dropped or changed launch variables {', '.join(missing)}")
    if carried is None:
        return
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
    measurement surface, :class:`PointError` when the rendered point does not carry the recipe's launch settings, and
    :class:`~hyperloom.measure.eval_context.EvalContextError` when an accuracy point's launch serves less context than
    its evaluation runs in.
    """

    if accuracy and canonical.agentx:
        raise PointError("an AgentX workload has no accuracy test: Magpie's replay validity is its check")
    launch = environment.recipe.launch
    check_launch_envs([*launch.envs, *environment.envs])
    out_dir.mkdir(parents=True, exist_ok=True)
    points = []
    for index, level in enumerate(_concurrencies(concurrency, canonical)):
        run_eval = "true" if accuracy else "false"
        pins = {ENV_CONC: str(level), ENV_RUN_EVAL: run_eval}
        reference = _reference(canonical, pins)
        name = f"point-{index}-c{level}.yaml"
        config = _render(
            canonical,
            out_dir,
            name,
            extra_server_args=launch.server_args,
            extra_envs={**launch.envs, **environment.envs, **pins},
            remove_args=launch.remove_args,
        )
        check_surface(config, canonical.surface)
        check_point(config, concurrency=level, accuracy=accuracy, reference=point_values(reference))
        _check_launch_present(config, launch, environment.envs)
        if accuracy:
            envs, carried = launch_carriers(config)
            check_eval_context(envs, carried or "")
        points.append(out_dir / name)
    return tuple(points)


#: Surface fields a profile point may change: the profile run lowers its output length to fit the capture window.
_PROFILE_SURFACE_EXEMPT = (f"envs.{ENV_OSL}",)
#: What the profile point takes from the canonical rendering, so the profile configuration's own defaults for the
#: workload (a different input length, another TP) cannot make it profile a workload the tests do not run.
_PROFILE_PINNED = (ENV_TP, ENV_EP, ENV_ISL, ENV_RANDOM_RANGE_RATIO, ENV_MAX_MODEL_LEN)


def _profile_canonical(canonical: CanonicalMeasurement) -> CanonicalMeasurement:
    """The canonical measurement rendered from the shipped profile configuration of its framework."""

    framework = canonical.surface.fields.get("benchmark.framework") or framework_registry.DEFAULT_FRAMEWORK
    name = framework_registry.shipped_config_name("profile", str(framework))
    path = default_baseline_config().parent / name
    if not path.is_file():
        raise PointError(f"there is no shipped profile configuration for framework {framework!r}")
    return CanonicalMeasurement(path, canonical.surface, canonical.primary_concurrency, canonical.materialize)


#: The serving steps an AgentX profile captures when its point names no count: half the single-lineage default (256), at
#: which a TP4 MoE head wrote more trace than the 4 GiB a result may carry.
AGENTX_PROFILE_STEPS = 128


def _limit_agentx_capture(point: Path) -> None:
    """Give the AgentX profile ``point`` the capture window of :data:`AGENTX_PROFILE_STEPS` steps, unless it names one."""
    config = yaml.safe_load(point.read_text(encoding=_UTF8)) or {}
    profiler = config.setdefault("benchmark", {}).setdefault("profiler", {}).setdefault("torch_profiler", {})
    if "num_steps" in profiler:
        return
    profiler["num_steps"] = AGENTX_PROFILE_STEPS
    point.write_text(yaml.safe_dump(config, sort_keys=False), encoding=_UTF8)


def generate_profile_point(
    environment: CompiledEnvironment, canonical: CanonicalMeasurement, out_dir: Path
) -> tuple[Path, ...]:
    """The one point of a ``profile`` test: the recipe's launch on the shipped profile configuration.

    The profile configuration turns the torch profiler on (``PROFILE=1``, ``PROFILE_EXTRA_BODY``, the framework's
    capture flags) and the renderer sizes the capture window from the workload, as the single-lineage profile does.
    The point runs at the primary concurrency without accuracy; apart from the output length the measurement surface
    must still be the canonical one. An AgentX workload profiles the point it measures: Magpie captures the traces
    when the point's ``profiler.torch_profiler`` is enabled, as the single-lineage AgentX profile turns it on.
    """

    if canonical.agentx:
        (point,) = generate_points(environment, canonical, out_dir, concurrency=(canonical.primary_concurrency,))
        _limit_agentx_capture(point)
        enable_profile(point)
        return (point,)
    launch = environment.recipe.launch
    check_launch_envs([*launch.envs, *environment.envs])
    out_dir.mkdir(parents=True, exist_ok=True)
    level = canonical.primary_concurrency
    pins = {ENV_CONC: str(level), ENV_RUN_EVAL: "false"}
    reference = benchmark_envs(_reference(canonical, pins))
    workload = {name: str(reference[name]) for name in _PROFILE_PINNED if reference.get(name) is not None}
    config = _render(
        _profile_canonical(canonical),
        out_dir,
        "point-profile.yaml",
        extra_server_args=launch.server_args,
        extra_envs={**launch.envs, **environment.envs, **workload, **pins},
        remove_args=launch.remove_args,
    )
    check_surface(config, canonical.surface, ignore=_PROFILE_SURFACE_EXEMPT)
    check_point(config, concurrency=level, accuracy=False, reference={})
    _check_launch_present(config, launch, environment.envs)
    return (out_dir / "point-profile.yaml",)


@dataclass(frozen=True)
class AcceptedMeasurement:
    """What acceptance produces for a test request: its environment and the points it runs, and the short-prompt
    consistency check and the long-context check the toolbox decided the test runs (``None`` for none)."""

    environment: CompiledEnvironment
    points: tuple[Path, ...]
    short_prompts: ShortPromptPlan | None = None
    long_context: LongContextPlan | None = None


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
    "AGENTX_PROFILE_STEPS",
    "AMBIENT_SHAPE_ENVS",
    "CANONICAL_FILE",
    "AcceptedMeasurement",
    "CanonicalMeasurement",
    "PointError",
    "accept_measurement",
    "default_canonical",
    "generate_points",
    "generate_profile_point",
    "launch_carriers",
]
