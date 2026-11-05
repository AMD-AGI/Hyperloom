# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The aiter modules an image's own launch needs and its ``aiter/jit`` lacks, built on CPU before the first test.

aiter builds a module when its ``<name>.so`` is missing from the JIT directory, at the first call that needs it. An image
that does not ship a GEMM module generated from its tuned tables (:func:`~hyperloom.measure.framework.table_modules.
table_modules`) therefore compiles it at the first prefill of every fresh container, and sglang's scheduler watchdog
(300 s unless the launch sets ``--watchdog-timeout``) can fire on every rank while it does.

:func:`prewarm` names those modules in its report and removes the cost: it builds the ones aiter can build by name
(:func:`~hyperloom.measure.runner.native.aiter_modules.buildable_modules`) with the CPU build runner of aiter's touched
modules, from the image's own tree with no root changed, and installs each into the image's ``aiter/jit`` with
``os.replace``, then records it in the manifest beside them
(:data:`~hyperloom.orchestrator.actions.executors._aiter_jit.AITER_PREWARM_MANIFEST`, whose format the launch preflight
owns and reads).

This writes into the image deliberately, and that is safe: no candidate's code or table takes part (the farm is laid from
the image's package and meta directory alone, and the build drops every ``AITER_CONFIG_*``, so aiter resolves the image's
own tables), and aiter would build the same module from the same inputs at the first call. A module already present is
left as it is. A build that fails installs nothing and is only logged: the launch then builds at its first prefill, as
before, under the watchdog above.

Given the card's CU count (``cu_num``), the build reads it as ``CU_NUM``, so aiter compiles the tables' rows of exactly
that ``(gfx, cu_num)`` (it takes the count of ``GFX_CU_NUM_MAP`` otherwise, which a partitioned card does not have), and
the built modules must pass the launch preflight's own coverage test for it
(:func:`~hyperloom.orchestrator.actions.executors._aiter_jit.serving_modules_cover_csv`, over every table the image's
launch loads) before anything is installed, or that preflight would move them aside again.

The build runs under the environment a test's server has when nothing of the test sets a build variable
(:func:`build_environment`: the image's variables the local runner's template carries, not every variable of the process
that runs this), and the manifest records the value of each variable a module's build reads
(:func:`~hyperloom.measure.jit.prewarmed.build_record`): a test whose server sets one otherwise is not given the module
(:func:`~hyperloom.measure.jit.prewarmed.unlike_prewarmed`) and builds its own.

The measure runner runs it before each test for the modules the launch preflight's check of the image's own tables moves
aside (:func:`preflight_build`, which the runner gives
:func:`~hyperloom.orchestrator.actions.executors._aiter_serving_once.build_image_own_once`): each test's servers build
into a JIT directory of their own, so without it every test would build them again. It can also be run by hand before
the first test of a container: ``python -m hyperloom.measure.runner.native.prewarm --gfx gfx950``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import stat
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from hyperloom.common.aiter_layout import image_aiter_meta
from hyperloom.measure.compile import EnvRoot, LocalBlobStore
from hyperloom.measure.farm import FarmError, locate_package
from hyperloom.measure.framework.table_modules import table_modules
from hyperloom.measure.jit.key import AITER_PACKAGE
from hyperloom.measure.jit.modules import build_env_names, config_path, module_build_seconds, read_config
from hyperloom.measure.jit.prewarmed import CU_NUM, build_record
from hyperloom.measure.jit.native.record import READY
from hyperloom.measure.jit.native.store import BuildInfraError
from hyperloom.measure.jit.store import BLOBS_DIR
from hyperloom.measure.local_templates import image_env
from hyperloom.measure.runner.cache_layout import file_sha256
from hyperloom.measure.runner.native.aiter_modules import (
    AITER_COMPONENT,
    AITER_RECIPE_FACTORY,
    AiterModules,
    buildable_modules,
)
from hyperloom.measure.runner.native.builder import BuildMachine
from hyperloom.measure.runner.native.isolated import IsolatedBuilds, Isolation, local_isolation
from hyperloom.measure.runner.native.planning import NativePlanner, NativeSite
from hyperloom.orchestrator.actions.executors._aiter_jit import (
    AITER_ENV_TO_SERVING_MODULES,
    AITER_ENV_TO_TUNED_FILE,
    AITER_PREWARM_MANIFEST,
    AITER_PREWARM_SCHEMA,
    csvs_aiter_will_load,
    read_prewarm_manifest,
    serving_modules_cover_csv,
    tuned_tables_sha256,
)

__all__ = [
    "ModuleBuilder",
    "PrewarmError",
    "PrewarmReport",
    "build_environment",
    "isolated_builder",
    "main",
    "preflight_build",
    "prewarm",
]

_JIT_DIR = "jit"
_SO = ".so"
_CONFIGS_DIR = "configs"
_MODULES = "modules"
_INSTALLED_MODE = 0o644
_UTF8 = "utf-8"
_LOG_TAIL = 400
#: Nothing is stored or signed: the build's key only names it within this process.
_LOCAL_IMAGE = "prewarm"
#: The image's interpreter, which imports the aiter being prewarmed.
_IMAGE_PYTHON = "python3"
_GFX_ENVS = ("HYPERLOOM_BUILD_GFX", "GPU_ARCHS")
#: The distribution of the image's aiter as the build's root names it: a sheet's root must name one, and the farm lays the
#: package from the image whatever it says.
_UNREAD_DIST = (AITER_PACKAGE, "image")

logger = logging.getLogger(__name__)

#: ``builder(package, modules, scratch) -> {module: built .so}``: the files may live below ``scratch`` only.
ModuleBuilder = Callable[[Path, frozenset[str], Path], Mapping[str, Path]]


class PrewarmError(Exception):
    """The modules could not be built or installed; nothing of the image was changed."""


@dataclass(frozen=True)
class PrewarmReport:
    """What one :func:`prewarm` did: ``built`` were installed, ``present`` were there already, ``left`` aiter cannot build
    by name (the launch builds them), and ``failure`` says why nothing was installed, when it was not."""

    built: tuple[str, ...] = ()
    present: tuple[str, ...] = ()
    left: tuple[str, ...] = ()
    failure: str = ""
    manifest: str = ""
    seconds: float = 0.0


def _wanted(config: Mapping[str, Any], tables: Collection[str] | None) -> list[str]:
    """The config's table-generated modules, those reading one of ``tables`` when given."""
    names = set(tables) if tables is not None else None
    return sorted(key for key, use in table_modules(config).items() if names is None or use.variables & names)


def _missing(jit: Path, config: Mapping[str, Any], tables: Collection[str] | None) -> list[str]:
    return [name for name in _wanted(config, tables) if not (jit / f"{name}{_SO}").is_file()]


def build_environment(environ: Mapping[str, str], cu_num: int | None) -> dict[str, str]:
    """The environment the modules are built under: what a test's server has when nothing of the test sets a build
    variable -- the image's variables of ``environ`` that the local runner's template carries
    (:func:`~hyperloom.measure.local_templates.image_env`) -- with the card's CU count as ``CU_NUM`` when it is given. The
    build sets ``GPU_ARCHS`` to the target itself."""
    built = image_env(environ)
    built.pop(CU_NUM, None)
    if cu_num is not None:
        built[CU_NUM] = str(cu_num)
    return built


def isolated_builder(
    machine: BuildMachine,
    *,
    isolation: Callable[[], Isolation | None] = local_isolation,
    factory: str = AITER_RECIPE_FACTORY,
    environ: Mapping[str, str] | None = None,
) -> ModuleBuilder:
    """The CPU build runner of aiter's touched modules
    (:class:`~hyperloom.measure.runner.native.isolated.IsolatedBuilds`) over a farm laid from the image's aiter with no
    file changed; ``machine.python`` must find the package being prewarmed."""

    def build(package: Path, modules: frozenset[str], scratch: Path) -> Mapping[str, Path]:
        located = locate_package(AITER_PACKAGE, machine.python, scratch)
        if located.resolve() != Path(package).resolve():
            raise PrewarmError(f"{machine.python} imports aiter from {located}, not from {package}")
        # The compile's user passes through ``scratch`` to its work directory below it, without listing it.
        os.chmod(scratch, stat.S_IMODE(os.stat(scratch).st_mode) | stat.S_IXOTH)
        blobs = LocalBlobStore(scratch / BLOBS_DIR)
        planner = NativePlanner(NativeSite(machine, {}, scratch / "sources", Path(package).parent), blobs)
        image = EnvRoot(AITER_PACKAGE, AITER_PACKAGE, *_UNREAD_DIST, ())
        planned = AiterModules(planner, environ=environ)((image,), read_config(config_path(package)), modules)
        if planned is None:
            return {}
        work = scratch / "build"
        work.mkdir(mode=0o700)
        staged = IsolatedBuilds(planner, blobs, isolation(), {AITER_COMPONENT: factory})(planned.inputs)(work)
        if staged.status != READY:
            raise PrewarmError(f"the build failed: {staged.failure}: {staged.log.strip()[-_LOG_TAIL:]}")
        outputs = {
            output.path.removesuffix(_SO): staged.staging / BLOBS_DIR / output.sha256 for output in staged.outputs
        }
        return {name: path for name, path in outputs.items() if name in modules}

    return build


@dataclass(frozen=True)
class _Build:
    """What the modules are built for and under: the target, the environment (:func:`build_environment`) and the module
    config."""

    gfx: str
    cu_num: int | None
    environ: Mapping[str, str]
    config: Mapping[str, Any]

    def target(self) -> str:
        return self.gfx if self.cu_num is None else f"{self.gfx}/{self.cu_num}"


def _install(jit: Path, built: Mapping[str, Path], entries: Mapping[str, dict[str, Any]]) -> None:
    """Copy every built module beside its place first, then ``os.replace`` each into it and the manifest after them: a
    copy that fails leaves the image as it was, and a module installed without its manifest entry is only one the old
    flow's cleanup moves aside (it is then built at the first prefill)."""
    pending: list[tuple[Path, Path]] = []
    try:
        for name, source in sorted(built.items()):
            temporary = jit / f".{name}{_SO}.prewarm-{uuid.uuid4().hex}"
            pending.append((temporary, jit / f"{name}{_SO}"))
            shutil.copyfile(source, temporary)
            os.chmod(temporary, _INSTALLED_MODE)
        manifest = jit / f".{AITER_PREWARM_MANIFEST}.{uuid.uuid4().hex}"
        pending.append((manifest, jit / AITER_PREWARM_MANIFEST))
        document = {"schema": AITER_PREWARM_SCHEMA, _MODULES: [entries[name] for name in sorted(entries)]}
        with open(manifest, "x", encoding=_UTF8) as out:
            json.dump(document, out, indent=2, sort_keys=True)
        os.chmod(manifest, _INSTALLED_MODE)
    except OSError:
        for temporary, _ in pending:
            temporary.unlink(missing_ok=True)
        raise
    for temporary, target in pending:
        os.replace(temporary, target)


def _entries(package: Path, built: Mapping[str, Path], build: _Build) -> dict[str, dict[str, Any]]:
    """The manifest (:data:`~hyperloom.orchestrator.actions.executors._aiter_jit.AITER_PREWARM_MANIFEST`) after
    installing ``built``: its earlier entries, with the built modules' replaced by ``{"name", "sha256", "tables_sha256",
    "gfx", "build_variables", "build_environment", "built_at"}`` and ``cu_num`` when the build was given one.

    A cleanup that moves session-built modules aside may leave a module listed whose file still has the listed ``sha256``
    and whose tables are still the listed ones
    (:func:`~hyperloom.orchestrator.actions.executors._aiter_jit.tuned_tables_sha256`), when the round leaves its table
    unset: it is what aiter builds at the first call."""
    common: dict[str, Any] = {"tables_sha256": tuned_tables_sha256(package / _CONFIGS_DIR), "gfx": build.gfx}
    if build.cu_num is not None:
        common["cu_num"] = build.cu_num
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    entries = read_prewarm_manifest(package / _JIT_DIR)
    meta = image_aiter_meta(package)
    for name, path in built.items():
        variables = build_record(build_env_names(package, meta, build.config, (name,)), build.environ)
        entries[name] = {"name": name, "sha256": file_sha256(path), **common, **variables, "built_at": stamp}
    return entries


def prewarm(
    image_root: Path,
    builder: ModuleBuilder,
    *,
    gfx: str,
    cu_num: int | None = None,
    tables: Collection[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> PrewarmReport:
    """Build the modules the image's own tuned tables generate (by its ``optCompilerConfig.json``) that its ``aiter/jit``
    lacks and aiter can build by name, with ``builder``, and install them into it (see the module); with ``cu_num``, only
    when they cover the image's tables for ``(gfx, cu_num)``. ``image_root`` is the image's aiter package directory (the
    one holding ``jit/`` and ``configs/``); ``tables`` narrows the modules to those generated from one of these
    ``AITER_CONFIG_*`` variables' tables (every tuned table when ``None``). ``environ`` is what ``builder`` builds under,
    which the manifest records (:func:`build_environment` of this process's when ``None``). Never raises for a build or
    an install that fails: the report says why, the image is unchanged, and the launch builds them at its first
    prefill."""
    started = time.monotonic()
    package = Path(image_root)
    jit = package / _JIT_DIR
    try:
        config = read_config(config_path(package))
    except (OSError, ValueError) as exc:
        return _ended(PrewarmReport(failure=f"the module config cannot be read: {exc}"), started)
    missing = _missing(jit, config, tables)
    present = tuple(name for name in _wanted(config, tables) if name not in missing)
    buildable = buildable_modules(config, missing)
    left = tuple(sorted(set(missing) - buildable))
    if not buildable:
        return _ended(PrewarmReport(present=present, left=left), started)
    build = _Build(gfx, cu_num, build_environment(os.environ, cu_num) if environ is None else environ, config)
    logger.warning(
        "aiter prewarm: building %s for %s on CPU, about %d s (each module's typical build on a card)",
        ", ".join(sorted(buildable)),
        build.target(),
        sum(module_build_seconds(name, config) for name in buildable),
    )
    try:
        built = _built_and_installed(package, builder, buildable, build)
    except (PrewarmError, BuildInfraError, FarmError, OSError, ValueError) as exc:
        report = PrewarmReport(present=present, left=left, failure=f"{type(exc).__name__}: {exc}")
        return _ended(report, started)
    report = PrewarmReport(tuple(sorted(built)), present, left, manifest=str(jit / AITER_PREWARM_MANIFEST))
    return _ended(report, started)


def _built_and_installed(
    package: Path, builder: ModuleBuilder, buildable: frozenset[str], build: _Build
) -> Mapping[str, Path]:
    """Build ``buildable`` with ``builder`` and install them with their manifest entries, or raise before installing any."""
    with tempfile.TemporaryDirectory(prefix="hyperloom-prewarm-") as scratch:
        built = dict(builder(package, frozenset(buildable), Path(scratch)))
        absent = sorted(buildable - set(built))
        if absent:
            raise PrewarmError(f"the build made no {', '.join(absent)}")
        if build.cu_num is not None:
            uncovered = _uncovered(package, built, (build.gfx, build.cu_num), Path(scratch) / "cover")
            if uncovered:
                raise PrewarmError(f"the built modules lack kernels of {build.target()} rows of {', '.join(uncovered)}")
        _install(package / _JIT_DIR, built, _entries(package, built, build))
    return built


def _uncovered(package: Path, built: Mapping[str, Path], target: tuple[str, int], view: Path) -> list[str]:
    """The tables the image's own launch loads (every ``AITER_CONFIG_*`` unset) whose rows for ``target`` name a kernel
    the built modules lack, by the launch preflight's test; ``view`` is a new directory the test reads them from."""
    view.mkdir()
    for name, path in built.items():
        os.symlink(path, view / f"{name}{_SO}")
    uncovered: list[str] = []
    for variable, modules in AITER_ENV_TO_SERVING_MODULES.items():
        tuned = AITER_ENV_TO_TUNED_FILE.get(variable)
        if not tuned or not set(modules) & set(built):
            continue
        for table in csvs_aiter_will_load(package / _CONFIGS_DIR, tuned, ""):
            if table.is_file() and not serving_modules_cover_csv(view, modules, table, frozenset({target})):
                uncovered.append(table.name)
    return uncovered


def _ended(report: PrewarmReport, started: float) -> PrewarmReport:
    report = replace(report, seconds=round(time.monotonic() - started, 3))
    if report.failure:
        logger.warning("aiter prewarm installed nothing; the launch builds at its first prefill: %s", report.failure)
    elif report.built:
        logger.info("aiter prewarm installed %s", ", ".join(report.built))
    return report


def preflight_build(package: Path, tables: Sequence[str], gfx: str, cu_num: int) -> dict[str, Any]:
    """Build into the image's aiter at ``package`` the modules of the ``AITER_CONFIG_*`` variables ``tables`` for ``gfx``
    and ``cu_num``, with the CPU build runner and the image's interpreter, and give the report as a mapping: the ``build``
    the measure runner hands
    :func:`~hyperloom.orchestrator.actions.executors._aiter_serving_once.build_image_own_once` for what the launch
    preflight's check of the image's own tables moved aside."""
    return asdict(_prewarm_image(package, gfx, cu_num, tables, _IMAGE_PYTHON))


def _prewarm_image(
    package: Path, gfx: str, cu_num: int | None, tables: Collection[str] | None, python: str
) -> PrewarmReport:
    """:func:`prewarm` of the aiter at ``package`` with the CPU build runner over ``python`` (which must import it),
    under :func:`build_environment` of this process's."""
    environ = build_environment(os.environ, cu_num)
    builder = isolated_builder(BuildMachine(_LOCAL_IMAGE, _LOCAL_IMAGE, gfx, python), environ=environ)
    return prewarm(package, builder, gfx=gfx, cu_num=cu_num, tables=tables, environ=environ)


def _gfx(given: str | None) -> str:
    gfx = given or next((os.environ[name] for name in _GFX_ENVS if os.environ.get(name, "").strip()), "")
    if not gfx or gfx == "native":
        raise SystemExit(f"prewarm needs --gfx (or one of {', '.join(_GFX_ENVS)}): the CPU build asks no GPU")
    return gfx


def _positive(value: str | None) -> int | None:
    """``value`` as a CU count: a positive integer, else ``None``."""
    try:
        number = int(str(value).strip())
    except ValueError:
        return None
    return number if number > 0 else None


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m hyperloom.measure.runner.native.prewarm --gfx <arch>``: prewarm the aiter ``--python`` imports (or
    ``--package``, which it must import) and print the report as JSON. Exits 0 when nothing could be installed too, so the
    launch that follows still runs."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gfx", help="the GPU architecture the modules are built for, e.g. gfx950")
    parser.add_argument(
        "--cu-num",
        type=int,
        help="the card's CU count: the build's CU_NUM, and the rows the built modules must cover (default: CU_NUM)",
    )
    parser.add_argument("--python", default=_IMAGE_PYTHON, help="the image's interpreter (default: python3)")
    parser.add_argument("--package", type=Path, help="the aiter package to prewarm (default: the one --python imports)")
    parser.add_argument("--table", action="append", dest="tables", help="only the modules of this AITER_CONFIG_* table")
    args = parser.parse_args(argv)
    gfx = _gfx(args.gfx)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        package = args.package or locate_package(AITER_PACKAGE, args.python, Path.cwd())
    except FarmError as exc:
        report = PrewarmReport(failure=f"the image's aiter cannot be found: {exc}")
    else:
        cu_num = args.cu_num if args.cu_num is not None else _positive(os.environ.get(CU_NUM))
        report = _prewarm_image(package, gfx, cu_num, args.tables, args.python)
    json.dump(asdict(report), sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
