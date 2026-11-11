# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Compose a round's launch from its frozen base and the config's layers.

The order is fixed: base, user removals, user additions, candidate removals, candidate additions, the golden layer (when
the round applies it), the guard (the round's model path, address and GPU mask), the runner's layer (with the
framework's metrics flag when the round scrapes ``/metrics``, and the engine seed :data:`DEFAULT_SEED` when no earlier
layer names one), then the profiler layer of a profile round. A layer's
flag beats every earlier group of the same flag; a later alias of it wins only because the engine applies the last
one. Neither the user nor the candidate may set what the guard or the runner owns, nor, under a golden layer, what
golden owns; such a layer is refused, not overridden, so the refusal names whose layer it was.

The server runs as the framework's server module under ``server.framework_python``, the interpreter the version probe
and the parser check run, never the base's entrypoint through ``PATH``. A base whose entrypoint is not its framework's
server command names a program the round would not run, so it is refused.
"""

from __future__ import annotations

import shlex
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from inference_suite.argv import argv_flag_occurrences, canonical_flag_name, normalize_argv, parse_removal_specs
from inference_suite.contracts.base_launch import BaseLaunch, EnvValue, RunnerOwned
from inference_suite.contracts.config import GoldenUse, LaunchDelta, RunnerLayer, ServerConfig, SuiteConfig
from inference_suite.contracts.launch import (
    ClientInvocation,
    EvalInvocation,
    FrameworkCheck,
    Launch,
    LaunchArg,
    LaunchEnv,
    LaunchError,
    LayerSource,
    Removal,
    ToolPins,
    VersionCheck,
)
from inference_suite.contracts.report import FailedStep
from inference_suite.frameworks import Framework, ServerFlags, framework_facts
from inference_suite.golden import GOLDEN_OWNED_ENV_PREFIXES, golden_owned_settings, is_golden_owned_flag

#: The mask that names the server's cards, by their machine numbers. ROCr enumerates only these, so no process of the
#: server opens another card: under a HIP mask alone ROCr opens every GPU and keeps memory on each, which a test running
#: beside the round on the other cards takes for a foreign holder of its own.
GPU_MASK_ENV = "ROCR_VISIBLE_DEVICES"
#: HIP's mask beneath it: ROCr numbers the cards it enumerates from 0, so HIP takes all of them, in that order. It must
#: be set, since Ray (which vLLM imports at start) refuses a ROCr mask without it.
HIP_MASK_ENV = "HIP_VISIBLE_DEVICES"
#: Every device mask a launch's processes read; the guard sets the first two and no layer may set any.
_DEVICE_MASK_ENVS = (GPU_MASK_ENV, HIP_MASK_ENV, "CUDA_VISIBLE_DEVICES")
#: The engine seed a launch runs with when no layer names one. Unpinned, the engine draws a fresh one at every start,
#: and under speculative decoding's rejection sampling the accept length, and with it the throughput, moves with that
#: draw, so two cold starts of one recipe would not measure the same launch.
DEFAULT_SEED = 0


@dataclass(frozen=True)
class RoundFacts:
    """What the round established before composing: the installed framework, the tools staged, the client and eval."""

    config_sha256: str
    installed_version: str
    version_check: VersionCheck
    tool_pins: Mapping[str, str] = field(default_factory=dict)
    client: ClientInvocation | None = None
    eval: EvalInvocation | None = None
    #: The profile round's profiler layer, from ``inference_suite.profile.inject.profiler_layer``.
    profiler: LaunchDelta | None = None


@dataclass(frozen=True)
class Refusal:
    """Why no launch was composed, and whose layer caused it."""

    step: FailedStep
    source: LayerSource
    detail: str


@dataclass(frozen=True)
class _Touch:
    """The flags (canonical names) and environment names one layer sets or removes, and the golden keys it sets."""

    flags: frozenset[str]
    env: frozenset[str]
    golden: tuple[str, ...]


def _delta_touch(delta: LaunchDelta) -> _Touch:
    added = argv_flag_occurrences(delta.args_add)
    removed = {canonical_flag_name(flag) for flag, _ in parse_removal_specs(delta.args_remove)}
    golden = set(golden_owned_settings(added, delta.env_set))
    golden |= {f"--{name}" for name in removed if is_golden_owned_flag(name)}
    golden |= {name for name in delta.env_unset if name.startswith(GOLDEN_OWNED_ENV_PREFIXES)}
    return _Touch(
        flags=frozenset({name for name, _ in added} | removed),
        env=frozenset({*delta.env_set, *delta.env_unset}),
        golden=tuple(sorted(golden)),
    )


def _layer_refusal(source: LayerSource, touch: _Touch, owned: _Touch, golden_held: bool) -> Refusal | None:
    owned_hit = sorted(f"--{name}" for name in touch.flags & owned.flags) + sorted(touch.env & owned.env)
    if owned_hit:
        detail = f"the {source.value} layer sets runner-owned {', '.join(owned_hit)}"
        return Refusal(FailedStep.CONFIG_INVALID, source, detail)
    if golden_held and touch.golden:
        detail = f"the {source.value} layer sets golden-owned {', '.join(touch.golden)}"
        return Refusal(FailedStep.GOLDEN_OWNED_TOUCHED, source, detail)
    return None


def runner_owned(base: BaseLaunch) -> RunnerOwned:
    """What no user or candidate layer on ``base`` may set: the base's runner-owned keys, the guard's and the metrics
    flag. A launch a round composed spells them, so a layer replayed from it leaves them out."""
    spec = framework_facts(Framework(base.framework))
    server_flags = spec.server_flags
    guard = (
        (server_flags.model, server_flags.host, server_flags.port, server_flags.served_model_name)
        if server_flags is not None
        else ()
    )
    flags = {*base.runner_owned.args, *(flag for flag in (*guard, spec.metrics_flag) if flag)}
    return RunnerOwned(args=tuple(sorted(flags)), env=tuple(sorted({*base.runner_owned.env, *_DEVICE_MASK_ENVS})))


def _owned(base: BaseLaunch) -> _Touch:
    owned = runner_owned(base)
    return _Touch(flags=owned.flags(), env=frozenset(owned.env), golden=())


def _first_refusal(server: ServerConfig, base: BaseLaunch, owned: _Touch) -> Refusal | None:
    base_touch = _Touch(frozenset(name for name, _ in argv_flag_occurrences(base.args)), frozenset(base.env), golden=())
    layers = (
        (LayerSource.BASE, base_touch),
        (LayerSource.USER, _delta_touch(server.user)),
        (LayerSource.CANDIDATE, _delta_touch(server.candidate)),
    )
    golden_held = base.golden_al is not None
    return next(
        (refusal for source, touch in layers if (refusal := _layer_refusal(source, touch, owned, golden_held))),
        None,
    )


def _groups(tokens: Iterable[str]) -> list[tuple[str, tuple[str, ...]]]:
    """``tokens`` as flag groups: each flag with the values that follow it."""
    groups: list[tuple[str, list[str]]] = []
    for token in tokens:
        if argv_flag_occurrences([token]):
            groups.append((token, []))
        elif groups:
            groups[-1][1].append(token)
        else:
            raise LaunchError(f"a launch's args start with a flag, got {token!r}")
    return [(flag, tuple(values)) for flag, values in groups]


class _Composer:
    """The launch as the layers so far leave it."""

    def __init__(self, base: BaseLaunch) -> None:
        self.args = [LaunchArg(flag, values, LayerSource.BASE) for flag, values in _groups(base.args)]
        self.env = {name: LaunchEnv(name, value, LayerSource.BASE) for name, value in base.env.items()}
        self.removed: list[Removal] = []

    def remove_args(self, specs: tuple[str, ...], by: LayerSource) -> None:
        for flag, value in parse_removal_specs(list(specs)):
            name = canonical_flag_name(flag)
            scope = None if value is None else normalize_argv((value,))
            kept = [arg for arg in self.args if not (_named(arg, name) and scope in (None, arg.values))]
            if len(kept) < len(self.args):
                self.removed.append(Removal(flag, by))
            self.args = kept

    def add_args(self, tokens: tuple[str, ...], source: LayerSource) -> None:
        groups = _groups(tokens)
        names = {canonical_flag_name(flag) for flag, _ in groups}
        beaten = [arg for arg in self.args if canonical_flag_name(arg.flag) in names]
        self.args = [arg for arg in self.args if canonical_flag_name(arg.flag) not in names]
        for flag, values in groups:
            overrode = _beaten_sources(arg for arg in beaten if _named(arg, canonical_flag_name(flag)))
            self.args.append(LaunchArg(flag, values, source, overrode))

    def unset_env(self, names: tuple[str, ...], by: LayerSource) -> None:
        for name in names:
            if self.env.pop(name, None) is not None:
                self.removed.append(Removal(name, by))

    def set_env(self, env: Mapping[str, EnvValue], source: LayerSource) -> None:
        for name, value in env.items():
            previous = self.env.get(name)
            overrode = () if previous is None else _beaten_sources([previous])
            self.env[name] = LaunchEnv(name, value, source, overrode)

    def apply_delta(self, delta: LaunchDelta, source: LayerSource) -> None:
        self.remove_args(delta.args_remove, source)
        self.add_args(delta.args_add, source)
        self.unset_env(delta.env_unset, source)
        self.set_env(delta.env_set, source)

    def apply_runner(self, runner: RunnerLayer, metrics_flag: str | None) -> None:
        """Merge ``runner``, then ``metrics_flag`` when the round scrapes the server's ``/metrics``."""
        prepended: dict[str, EnvValue] = {}
        for name, value in runner.env_prepend.items():
            previous = self.env.get(name)
            if previous is not None and not isinstance(previous.value, str):
                raise LaunchError(f"the runner prepends to {name}, whose value the base redacts")
            prepended[name] = value if previous is None else f"{value}:{previous.value}"
        self.set_env({**prepended, **runner.env_set}, LayerSource.RUNNER)
        for flag, floor in runner.args_floor.items():
            current = next((arg for arg in reversed(self.args) if _named(arg, canonical_flag_name(flag))), None)
            if current is None or len(current.values) != 1 or float(current.values[0]) < floor:
                self.add_args((flag, _number_text(floor)), LayerSource.RUNNER)
        if metrics_flag is not None:
            self.add_args((metrics_flag,), LayerSource.RUNNER)

    def default_arg(self, flag: str | None, value: str) -> None:
        """Set ``flag`` to ``value`` as the runner's when the framework takes it and no layer so far names it."""
        if flag is not None and not any(_named(arg, canonical_flag_name(flag)) for arg in self.args):
            self.args.append(LaunchArg(flag, (value,), LayerSource.RUNNER))


def _named(arg: LaunchArg, name: str) -> bool:
    return canonical_flag_name(arg.flag) == name


def _beaten_sources(entries: Iterable[LaunchArg | LaunchEnv]) -> tuple[LayerSource, ...]:
    sources: list[LayerSource] = []
    for entry in entries:
        for source in (*entry.overrode, entry.source):
            if source not in sources:
                sources.append(source)
    return tuple(sources)


def _number_text(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def _guard(server: ServerConfig, server_flags: ServerFlags) -> tuple[tuple[str, ...], dict[str, EnvValue]]:
    args = [server_flags.model, server.model_path, server_flags.host, server.host, server_flags.port, str(server.port)]
    if server_flags.served_model_name is not None:
        args += [server_flags.served_model_name, server.served_model_name]
    masks = {GPU_MASK_ENV: ",".join(map(str, server.gpus)), HIP_MASK_ENV: ",".join(map(str, range(len(server.gpus))))}
    return tuple(args), masks


def compose(config: SuiteConfig, base: BaseLaunch, facts: RoundFacts) -> Launch | Refusal:
    """The launch ``config`` runs on ``base``, or why none can run.

    Raises:
        ConfigError: When ``base`` cannot serve the config's workload or golden use.
        LaunchError: When ``base`` is a scriptable framework's, which launches no server.
    """
    server = config.server
    if base.digest() != server.base.digest:
        detail = f"the config names {server.base.digest}, the base gives {base.digest()}"
        return Refusal(FailedStep.BASE_DIGEST_MISMATCH, LayerSource.BASE, detail)
    config.check_base(base)
    framework = Framework(base.framework)
    spec = framework_facts(framework)
    server_flags, command = spec.server_flags, spec.server_command
    if server_flags is None or command is None:
        raise LaunchError(f"{framework.value} runs no server, so it has no launch to compose")
    if base.entrypoint != command.entrypoint:
        detail = (
            f"the base runs {shlex.join(base.entrypoint)}, not {framework.value}'s {shlex.join(command.entrypoint)}; "
            f"the round runs {shlex.join(command.module)} under server.framework_python, so name the install there"
        )
        return Refusal(FailedStep.CONFIG_INVALID, LayerSource.BASE, detail)
    refusal = _first_refusal(server, base, _owned(base))
    if refusal is not None:
        return refusal

    composer = _Composer(base)
    composer.apply_delta(server.user, LayerSource.USER)
    composer.apply_delta(server.candidate, LayerSource.CANDIDATE)
    if server.golden is GoldenUse.APPLY and base.golden_al is not None:
        composer.add_args(base.golden_al.args, LayerSource.GOLDEN)
        composer.set_env(base.golden_al.env, LayerSource.GOLDEN)
    guard_args, guard_env = _guard(server, server_flags)
    composer.add_args(guard_args, LayerSource.GUARD)
    composer.set_env(guard_env, LayerSource.GUARD)
    composer.apply_runner(server.runner, spec.metrics_flag if config.telemetry.kv else None)
    composer.default_arg(server_flags.seed, str(DEFAULT_SEED))
    if facts.profiler is not None:
        composer.apply_delta(facts.profiler, LayerSource.PROFILE)

    return Launch(
        base=base,
        config_sha256=facts.config_sha256,
        golden_use=server.golden,
        entrypoint=command.argv(server.framework_python),
        args=tuple(composer.args),
        env_sources=tuple(composer.env.values()),
        removed=tuple(composer.removed),
        framework=FrameworkCheck(
            name=framework,
            installed_version=facts.installed_version,
            python=server.framework_python,
            recipe_min_version=base.image.framework_version,
            version_check=facts.version_check,
        ),
        client=facts.client,
        eval=facts.eval,
        tools=ToolPins(config.paths.tool_manifest_sha256, facts.tool_pins),
        source_files=server.source_files,
    )


__all__ = ["DEFAULT_SEED", "GPU_MASK_ENV", "Refusal", "RoundFacts", "compose", "runner_owned"]
