# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The runner's side of the shared compiled-code cache: from the instruction sheet to a session (design 4.21).

The sheet fixes what the compilers build from: the image, the GPU, the patch laid over the framework, the tuned tables
and the environment of the server. This module reads those facts, asks :mod:`hyperloom.measure.jit.key` for the key of
each kind and hands the backend a :class:`~hyperloom.measure.jit.session.JitCacheSession`. Nothing here is a reason
for a test to fail: a fact that cannot be read switches the kind that needs it off, with a note, and the test builds cold.

``HYPERLOOM_JIT_CACHE`` names the cache's directory; the template mounts it read-only, and without it there is no cache.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path, PurePosixPath

import yaml

from hyperloom.common.gpu_identity import gfx_arch_for_gpu_type
from hyperloom.common.outside_facts import AITER_META_DIR_ENV, AITER_META_WHEEL_DIR
from hyperloom.measure.farm import (
    Farm,
    FarmError,
    image_aiter_meta,
    locate_package,
    patches_aiter_sources,
    server_environment,
    server_python,
)
from hyperloom.measure.jit.builds import BuildLog, scan_builds, server_logs
from hyperloom.measure.jit.key import (
    AITER_PACKAGE,
    REBUILD_VARIABLE,
    BuildSurface,
    Machine,
    aiter_key,
    build_surface,
    scan_env_names,
    touches_build_surface,
    tree_identity,
    triton_key,
)
from hyperloom.measure.jit_kinds import AITER_CPP, AITER_JIT, TRITON, CacheKind
from hyperloom.measure.jit.modules import (
    CONFIG_FILE,
    SourceTree,
    Touched,
    config_path,
    prebuilt_names,
    read_config,
    touched_entries,
    touched_modules,
)
from hyperloom.measure.jit.native.config import BuildStoreConfigError, store_from_environment
from hyperloom.measure.jit.native.laying import NamedBuild, NativeLayError, SheetBuilds, sheet_builds
from hyperloom.measure.jit.native.record import COMPILE_ENV_PREFIX, BuildRecord
from hyperloom.measure.jit.session import GivenModules, JitCacheSession, KindPlan
from hyperloom.measure.framework.table_modules import file_identity, stale_table_modules
from hyperloom.measure.jit.store import DirJitStore
from hyperloom.measure.round_marks import boot_parts
from hyperloom.measure.runner.hardware import GpuFacts
from hyperloom.measure.runner.service import RunnerFailure
from hyperloom.measure.spec import EnvRoot, MeasurementSpec

__all__ = ["JIT_CACHE_ENV", "ColdStart", "Surroundings", "build_session", "cold_start"]

JIT_CACHE_ENV = "HYPERLOOM_JIT_CACHE"
STAGING_DIR = "jit-cache"
_TABLE_PREFIX = "AITER_CONFIG_"
_PYTHONPATH, _FRAMEWORK_PYTHON = "PYTHONPATH", "HYPERLOOM_FRAMEWORK_PYTHON"
_DIRECTORY = "dir"
_UTF8 = "utf-8"
_PREFLIGHT_MISMATCH = "preflight_mismatch"
_SPEC_REJECTED = "spec_rejected"
_RUNTIME_KIND = "runtime"
_GPU_ARCHS, _NATIVE = "GPU_ARCHS", "native"
_MODULE_SUFFIX = ".so"
_NEEDS_JIT_DIR = (
    "the patch changes aiter's csrc or hsa, which is measured only by building the modules it touches in the test's own "
    "aiter JIT directory, and this runner cannot give one: {why}"
)


@dataclass(frozen=True)
class Surroundings:
    """What the runner has besides the sheet: its environment, where a test's own cache directories and its staged
    contribution go, how to read a blob of the sheet, and where to leave notes."""

    environ: Mapping[str, str]
    private_root: Path
    staging: Path
    read_blob: Callable[[str], bytes]
    notes: list[str]


@dataclass(frozen=True)
class _Inputs:
    """The facts the keys are made of, read once."""

    spec: MeasurementSpec
    machine: Machine
    environment: Mapping[str, str]
    sheet_environment: Mapping[str, str]
    farm: Farm | None
    python: str
    around: Surroundings
    #: The environment of the server of each point: what the tuned tables of that point resolve under.
    servers: tuple[Mapping[str, str], ...] = ()
    #: The CPU build of aiter's touched modules the sheet names, proven.
    modules: _ModuleBuild | None = None


@dataclass(frozen=True)
class _ModuleBuild:
    """The signed record of a CPU build of aiter's modules, and how to copy its files out of the store."""

    record: BuildRecord
    lay: Callable[[Path], None]


def build_session(
    spec: MeasurementSpec, points: list[Path], farm: Farm | None, gpu: GpuFacts | None, around: Surroundings
) -> JitCacheSession | None:
    """The session of this test, or ``None`` when no cache is configured or none of its kinds can be used.

    A test that patches aiter's ``csrc`` or ``hsa`` cannot run without the aiter kind: the image's prebuilt modules would
    be what runs. It is refused (``preflight_mismatch``) when this runner cannot give the test its own JIT directory.
    """
    needs_jit_dir = patches_aiter_sources(spec.environment.roots)
    root = around.environ.get(JIT_CACHE_ENV, "").strip()
    arch = (gpu.gfx_arch if gpu else None) or gfx_arch_for_gpu_type(spec.hardware.gpu_type)
    if not root or arch is None:
        why = f"{JIT_CACHE_ENV} is not set" if not root else "the GPU architecture is not known"
        if needs_jit_dir:
            raise RunnerFailure(_PREFLIGHT_MISMATCH, _NEEDS_JIT_DIR.format(why=why))
        if root:
            around.notes.append("jit cache: the GPU architecture is not known, so the compiled-code cache is not used")
        return None
    config = yaml.safe_load(points[0].read_text(encoding=_UTF8)) or {}
    sheet = _sheet_environment(spec, config)
    inherited = {name: value for name, value in around.environ.items() if name not in spec.unset_envs}
    environment = {**inherited, **sheet}
    servers = (environment, *({**inherited, **_sheet_environment(spec, _point(path))} for path in points[1:]))
    machine = Machine(spec.image.digest, arch, spec.hardware.gpu_type, spec.hardware.partition_mode)
    python = server_python(config, default="python3")
    inputs = _Inputs(spec, machine, environment, sheet, farm, python, around, servers, _module_build(spec, around))
    plans, disabled = _plans(inputs)
    if needs_jit_dir:
        _require_jit_dir(plans, disabled, environment)
    if not plans and not disabled:
        return None
    return JitCacheSession(
        DirJitStore(Path(root)), plans, private_root=around.private_root, staging=around.staging, disabled=disabled
    )


def _module_build(spec: MeasurementSpec, around: Surroundings) -> _ModuleBuild | None:
    """The CPU build of aiter's modules the sheet names for its aiter package root, proven; ``None`` when it names none.

    Raises:
        RunnerFailure: ``spec_rejected`` when it names one for another package, or one this runner cannot prove.
    """
    roots = {root.id: root for root in spec.environment.roots}
    named = [ref for ref in spec.environment.builds if roots[ref.root_id].kind != _RUNTIME_KIND]
    if not named:
        return None
    foreign = [ref.root_id for ref in named if roots[ref.root_id].import_name != AITER_PACKAGE]
    if foreign:
        raise RunnerFailure(_SPEC_REJECTED, f"{foreign[0]}: a package root whose modules no CPU build makes")
    try:
        store = store_from_environment(around.environ)
        builds = sheet_builds(
            tuple(NamedBuild(r.root_id, r.key, r.manifest_sha256) for r in named), store, spec.image.digest
        )
        record = builds.module_record(named[0].root_id)  # type: ignore[union-attr]  # sheet_builds refused None
    except (BuildStoreConfigError, NativeLayError) as exc:
        raise RunnerFailure(_SPEC_REJECTED, str(exc)) from None
    return _ModuleBuild(record, partial(_fetch, builds, record))


def _fetch(builds: SheetBuilds, record: BuildRecord, directory: Path) -> None:
    try:
        builds.fetch(record, directory)
    except NativeLayError as exc:
        raise ValueError(str(exc)) from None


def _point(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding=_UTF8)) or {}


def _sheet_environment(spec: MeasurementSpec, config: Mapping) -> dict[str, str]:
    """What the sheet sets in the server of the point whose YAML is ``config``."""
    declared = {str(name): str(value) for name, value in server_environment(config).items()}
    return {**spec.process_env, **declared}


def _require_jit_dir(plans: list[KindPlan], disabled: Mapping[str, str], environment: Mapping[str, str]) -> None:
    """Refuse unless the aiter kind has a plan, or the test rebuilds every module itself (``AITER_REBUILD``)."""
    if any(plan.kind is AITER_JIT for plan in plans) or build_surface((), environment, {}).rebuild:
        return
    raise RunnerFailure(_PREFLIGHT_MISMATCH, _NEEDS_JIT_DIR.format(why=disabled.get(AITER_JIT.name, "it has no plan")))


def _plans(inputs: _Inputs) -> tuple[list[KindPlan], dict[str, str]]:
    plans: list[KindPlan] = []
    disabled: dict[str, str] = {}
    for kind in (TRITON, AITER_JIT, AITER_CPP):
        if kind.env_var in inputs.sheet_environment:
            disabled[kind.name] = f"the sheet sets {kind.env_var} itself"
    if TRITON.name not in disabled:
        plans.append(KindPlan(TRITON, triton_key(inputs.machine), read_only=_triton_read_only(inputs)))
    aiter_kinds = [kind for kind in (AITER_JIT, AITER_CPP) if kind.name not in disabled]
    if aiter_kinds:
        found = _aiter_plans(inputs, aiter_kinds)
        if isinstance(found, str):
            disabled.update({kind.name: found for kind in aiter_kinds})
        else:
            plans.extend(found)
    return plans, disabled


def _directory_attachments(spec: MeasurementSpec) -> list[str]:
    """The attachments that are directories: unlike a table, a directory can hold code the server runs."""
    return [attachment.id for attachment in spec.attachments if attachment.kind == _DIRECTORY]


def _triton_read_only(inputs: _Inputs) -> str | None:
    """Why this test may use the Triton cache but not add to it: Triton's key covers no patch, so any patched code (a
    framework, aiter or Triton package, or a directory attachment) can write files that other tests would then load."""
    patched = any(entry.op == "upsert" for root in inputs.spec.environment.roots for entry in root.entries)
    if patched or _directory_attachments(inputs.spec):
        return "the test carries patched code that the key of Triton's cache does not cover"
    return None


def _aiter_plans(inputs: _Inputs, kinds: list[CacheKind]) -> list[KindPlan] | str:
    """The plans of the aiter kinds, or the reason they cannot be used."""
    try:
        base = locate_package(AITER_PACKAGE, inputs.python, inputs.around.private_root.parent)
    except FarmError:
        return "aiter cannot be imported by the server's interpreter"
    farm = inputs.farm
    effective = farm.site / AITER_PACKAGE if farm is not None and AITER_PACKAGE in farm.packages else base
    meta_dirs = [farm.aiter_meta] if farm is not None and farm.meta_patched and farm.aiter_meta else []
    names = scan_env_names([effective, base], meta_dirs)
    if not names:
        return "no aiter build path was found in the package"
    environment = _build_environment(inputs.environment, farm)
    surface = build_surface({*names, _PYTHONPATH, _FRAMEWORK_PYTHON}, environment, _table_content(names, environment))
    if surface.rebuild:
        return f"{REBUILD_VARIABLE} is set, so the test rebuilds what the cache would give"
    tree = _tree(inputs, names)
    key = aiter_key(inputs.machine, tree.identity, surface)
    uncovered = tree.uncovered or _directory_attachments(inputs.spec)
    read_only = "the test carries patched code that the key of aiter's caches does not cover" if uncovered else None
    touched = _touched(inputs, base, effective)
    if isinstance(touched, str):
        return touched
    given = _given(inputs, surface, effective, touched.names)
    return [
        KindPlan(kind, key, effective / "jit", read_only, touched.names, given)
        if kind is AITER_JIT
        else KindPlan(kind, key, None, read_only)
        for kind in kinds
    ]


def _given(inputs: _Inputs, surface: BuildSurface, effective: Path, touched: frozenset[str]) -> GivenModules | None:
    """The touched modules the sheet's CPU build made that this server may load: built for this GPU, under the variables
    this server's build would read, from tables none of its variables change. Whatever is left out the server builds."""
    built = inputs.modules
    if built is None:
        return None
    names = frozenset(PurePosixPath(o.path).name.removesuffix(_MODULE_SUFFIX) for o in built.record.outputs) & touched
    problem = _build_differs(built.record, inputs.machine.gfx_arch, surface)
    if problem is not None:
        inputs.around.notes.append(f"jit cache: the CPU build of aiter's modules is not used: {problem}")
        return None
    try:
        tabled = stale_table_modules(read_config(config_path(effective)), effective, effective, inputs.servers)
    except (OSError, ValueError):
        tabled = names
    if names & tabled:
        inputs.around.notes.append(
            f"jit cache: {', '.join(sorted(names & tabled))} not taken from their CPU build: the server's tuned tables differ"
        )
    names -= tabled
    return GivenModules(names, built.lay) if names else None


def _build_differs(record: BuildRecord, gfx_arch: str, surface: BuildSurface) -> str | None:
    """How the CPU build's target and variables differ from what this server's build would have, ``None`` when not."""
    if record.inputs.gfx_arch != gfx_arch:
        return f"it was built for {record.inputs.gfx_arch} and this GPU is {gfx_arch}"
    built = {
        name.removeprefix(COMPILE_ENV_PREFIX): value
        for name, value in record.toolchain.items()
        if name.startswith(COMPILE_ENV_PREFIX)
    }
    # The interpreter and the import path are the image's on both sides; what they could change is in the farm.
    server = {k: v for k, v in surface.environment.items() if k not in (_PYTHONPATH, _FRAMEWORK_PYTHON)}
    if server.get(_GPU_ARCHS) in (_NATIVE, gfx_arch):
        del server[_GPU_ARCHS]
    differing = sorted(name for name in set(built) | set(server) if built.get(name) != server.get(name))
    if differing:
        return f"the server's {', '.join(differing)} differ from its build's"
    return None


def _touched(inputs: _Inputs, base: Path, effective: Path) -> Touched | str:
    """The prebuilt modules the sheet's patch to aiter's sources or its tuned tables make stale, or why that cannot be
    told. A table reaches the server through a variable it sets or through a patched aiter package (its ``configs``)."""
    roots = inputs.spec.environment.roots
    patched = [entry.path for root in roots if root.import_name == AITER_META_WHEEL_DIR for entry in root.entries]
    config_patched = any(
        entry.path == f"jit/{CONFIG_FILE}"
        for root in roots
        if root.import_name == AITER_PACKAGE
        for entry in root.entries
    )
    tables = effective != base or any(
        name.startswith(_TABLE_PREFIX) and value.strip() for server in inputs.servers for name, value in server.items()
    )
    if not patched and not config_patched and not tables:
        return Touched()
    farm = inputs.farm
    meta = farm.aiter_meta if farm is not None and farm.aiter_meta else image_aiter_meta(base)
    try:
        before, after = read_config(config_path(base)), read_config(config_path(effective))
    except (OSError, ValueError) as exc:
        return f"the module config of aiter cannot be read ({type(exc).__name__}), so which modules the patch or the tables touch is not known"
    stale = stale_table_modules(after, base, effective, inputs.servers) if tables else frozenset()
    prebuilt = prebuilt_names(base)
    return touched_modules(before, after, SourceTree(meta), patched, prebuilt) | touched_entries(stale, after, prebuilt)


def _build_environment(environment: Mapping[str, str], farm: Farm | None) -> dict[str, str]:
    """The server's environment with the places of the test's own farm left out, its place on ``PYTHONPATH`` and its copy
    of the meta directory: a path of this test's own is no input (the patched tree is, and the key holds that)."""
    result = dict(environment)
    if farm is not None and _PYTHONPATH in result:
        parts = [part for part in result.pop(_PYTHONPATH).split(os.pathsep) if part and part != str(farm.site)]
        if parts:
            result[_PYTHONPATH] = os.pathsep.join(parts)
    if farm is not None and farm.aiter_meta is not None and result.get(AITER_META_DIR_ENV) == str(farm.aiter_meta):
        del result[AITER_META_DIR_ENV]
    return result


def _table_content(names: Iterable[str], environment: Mapping[str, str]) -> dict[str, str]:
    """For each tuned-table variable the server sets, the content of the files it names: what a code generator reads."""
    content = {}
    for name in sorted(name for name in names if name.startswith(_TABLE_PREFIX) and environment.get(name)):
        content[name] = ":".join(file_identity(part) for part in environment[name].split(os.pathsep) if part)
    return content


@dataclass(frozen=True)
class _Tree:
    """The identity of the aiter source tree a test builds from, and the patched files of other packages that it does not
    cover (they cannot change the build, but the code in them runs in the server and can write into the cache)."""

    identity: str
    uncovered: list[str]


def _tree(inputs: _Inputs, names: frozenset[str]) -> _Tree:
    environment = inputs.spec.environment
    aiter_roots = [root for root in environment.roots if root.import_name == AITER_PACKAGE]
    meta_roots = [root for root in environment.roots if root.import_name == AITER_META_WHEEL_DIR]
    overlay = [(e.path, e.op, e.blob or "", e.mode or "") for root in aiter_roots for e in root.entries]
    overlay += [(root.id, e.path, e.op, e.blob or "", e.mode or "") for root in meta_roots for e in root.entries]
    runtime = environment.runtime.manifest_sha256 if environment.runtime is not None else None
    others = [root for root in environment.roots if root.import_name not in (AITER_PACKAGE, AITER_META_WHEEL_DIR)]
    touching, uncovered = _framework_patches(others, names, inputs.around.read_blob)
    return _Tree(tree_identity(overlay, runtime, touching), uncovered)


def _framework_patches(
    roots: Iterable[EnvRoot], names: frozenset[str], read_blob: Callable[[str], bytes]
) -> tuple[list[tuple[str, ...]], list[str]]:
    """The patched files of other packages that can change what aiter builds from inside the server, which join the key,
    and the rest, which do not."""
    touching: list[tuple[str, ...]] = []
    other: list[str] = []
    for root in roots:
        for entry in root.entries:
            if entry.op != "upsert" or entry.blob is None:
                continue
            text = read_blob(entry.blob).decode(_UTF8, errors="replace")
            if touches_build_surface(text, names):
                touching.append((root.import_name, entry.path, entry.blob))
            else:
                other.append(f"{root.import_name}/{entry.path}")
    return touching, other


@dataclass(frozen=True)
class ColdStart:
    """What the server's logs (``logs`` of them) say of its start: what aiter built and how long loading the weights took."""

    builds: BuildLog
    model_load_sec: float | None
    logs: int


def cold_start(files: Path) -> ColdStart:
    """Read the server logs under ``files`` for the two parts of a start that the result states beside its phases."""
    logs = server_logs(files)
    loads = [seconds for log in logs if (seconds := boot_parts(log).weights_load) is not None]
    return ColdStart(scan_builds(logs), max(loads, default=None), len(logs))
