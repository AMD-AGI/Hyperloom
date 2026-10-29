# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""aiter's touched modules built on a CPU build runner before the test (#125 stage 2b).

A patch to aiter's ``csrc`` or ``hsa`` (or to its module config, or to a tuned table below its ``configs``) leaves some of
the image's prebuilt modules stale (:mod:`hyperloom.measure.jit.modules`). Without this the test's server builds each on
its card when it first calls it. With a host that builds natively, :class:`AiterModules` plans one build of the modules
it can make, keyed like every native build (:class:`~hyperloom.measure.jit.native.key.BuildInputs`, component
``aiter``): the test waits for it without a card (``pending_build``), and the runner seeds the test's aiter JIT directory
with its outputs, so the server loads them and builds nothing.

The compile is aiter's own JIT build, the one its ``compile_ops`` runs when a module is missing: ``get_args_of_build`` and
``build_module`` of ``aiter/jit/core.py``, for each module by name, with ``GPU_ARCHS`` naming the target (no GPU is
asked), a private ``AITER_JIT_DIR`` the module is installed into (aiter copies it there and ``os.replace``-s it into
place) and ``AITER_META_DIR`` naming the tree it builds from. That tree is the farm the test's runner lays
(:func:`~hyperloom.measure.farm.build_farm`): the patched package and the mirror of the image's meta directory with the
patched ``csrc`` and ``hsa`` over it, so the sources, the config and the tables are the server's. What aiter needs to load
a module without building it is its ``<name>.so`` in the JIT directory, built for the running GPU's architecture
(``core.py`` checks only that the file is there and that its offload bundles name that architecture); the outputs are
those files, nothing else of the JIT directory.

A module is built here only when aiter can build it by name alone: an entry of the config, a Python module, not
standalone, with no third-party checkout to fetch (the build has no network). The rest, the variants aiter names at run
time among them, are built on the card as before. The variables the build reads are the build service's own (the
image's, since it runs in the image's digest), recorded in the build's record so the runner can refuse a build made
under other values than its server's.
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from hyperloom.common.aiter_layout import image_aiter_meta
from hyperloom.common.outside_facts import AITER_META_DIR_ENV, AITER_META_WHEEL_DIR
from hyperloom.contracts.recipe import DIGEST_PREFIX
from hyperloom.measure.compile import BlobStore, EnvRoot
from hyperloom.measure.docs import canonical_json
from hyperloom.measure.farm import FarmError, build_farm, locate_package
from hyperloom.measure.jit.key import AITER_PACKAGE, build_surface
from hyperloom.measure.jit.modules import ESTIMATED_MODULE_SECONDS, build_env_names
from hyperloom.measure.jit.native.key import BuildInputs
from hyperloom.measure.jit.native.needs import ModuleBuild
from hyperloom.measure.jit.native.record import READY, BuildRecord
from hyperloom.measure.jit.native.store import BuildInfraError
from hyperloom.measure.runner.native.builder import BuildMachine, LaidTree, tree_dir
from hyperloom.measure.runner.native.planning import NativePlanner
from hyperloom.measure.runner.jit import FRAMEWORK_PYTHON_ENV, interpreter_identity
from hyperloom.measure.runner.native.recipes import PYTHON_PLACEHOLDER, NativeRecipe, runner_preflight
from hyperloom.measure.spec import EnvRoot as SheetRoot
from hyperloom.orchestrator.enablement.runtime.targeted_build import AiterBuild, default_budget_sec

__all__ = [
    "AITER_COMPONENT",
    "AITER_RECIPE_FACTORY",
    "MODULES_FILE",
    "AiterModules",
    "aiter_module_recipe",
    "buildable_modules",
]

AITER_COMPONENT = "aiter"
_RECIPE_VERSION = "aiter_jit/1"
#: The file below the laid tree that names the modules to build and where the tree's parts are.
MODULES_FILE = "hyperloom-aiter-modules.json"
#: The JIT directory aiter installs the modules into, below the laid tree; the outputs are its ``.so`` files.
_JIT_DIR = "jit"
_SO = ".so"
#: Variables the build sets itself, whatever the service's environment holds.
_SET_BY_BUILD = frozenset({"GPU_ARCHS", AITER_META_DIR_ENV, "AITER_JIT_DIR"})
_TRUE, _FALSE = "True", "False"
_EMPTY_LISTS = (None, "[]", "''", '""')
#: What the compile runs (``python -I -B -c DRIVER MODULES_FILE``, in the laid tree): aiter's own JIT build of each module,
#: called as ``compile_ops`` calls it when a module is missing.
_DRIVER = """\
import json, os, sys
plan = json.load(open(sys.argv[1], encoding="utf-8"))
tree = os.getcwd()
jit = os.path.join(tree, plan["jit"])
os.makedirs(jit, exist_ok=True)
os.environ["AITER_JIT_DIR"] = jit
os.environ["AITER_META_DIR"] = os.path.join(tree, plan["meta"])
os.environ.pop("AITER_REBUILD", None)
sys.path.insert(0, os.path.dirname(os.path.join(tree, plan["package"])))
sys.path.insert(0, os.path.join(tree, plan["package"]))
from jit import core
for name in plan["modules"]:
    args = core.get_args_of_build(name)
    clang = args.get("hip_clang_path")
    if clang is not None and os.path.exists(clang):
        os.environ["HIP_CLANG_PATH"] = clang
    core.build_module(
        name, args["srcs"], args["flags_extra_cc"], args["flags_extra_hip"], args["blob_gen_cmd"],
        args["extra_include"], args["extra_ldflags"], args["verbose"], args["is_python_module"],
        args["is_standalone"], args["torch_exclude"], args.get("third_party", []), args.get("hipify", False),
        flags_extra_hip_per_source=args.get("flags_extra_hip_per_source", {}),
    )
    os.environ.pop("HIP_CLANG_PATH", None)
    if not os.path.isfile(os.path.join(jit, name + ".so")):
        sys.exit(f"aiter's build of {name} installed no {name}.so")
"""


def aiter_module_recipe() -> NativeRecipe:
    """The compile of aiter's modules: the single-lineage aiter recipe's preflight, ``MAX_JOBS`` and budget."""
    return NativeRecipe(
        component=AITER_COMPONENT,
        import_name=AITER_PACKAGE,
        version=_RECIPE_VERSION,
        source_dir="",
        package_dir=_JIT_DIR,
        outputs=(f"*{_SO}",),
        command=(PYTHON_PLACEHOLDER, "-I", "-B", "-c", _DRIVER, MODULES_FILE),
        budget_sec=default_budget_sec(AITER_COMPONENT),
        max_jobs=AiterBuild.default_max_jobs,
        max_output_bytes=2 * 1024**3,
        preflight=partial(runner_preflight, AiterBuild, AITER_COMPONENT, AITER_COMPONENT),
        gfx_variable="GPU_ARCHS",
        against_image=False,
    )


#: The build runner's child makes the recipe by this factory.
AITER_RECIPE_FACTORY = f"{__name__}:{aiter_module_recipe.__name__}"


def buildable_modules(config: Mapping[str, Any], candidates: Collection[str]) -> frozenset[str]:
    """The ``candidates`` aiter can build by name alone: config entries of Python modules, not standalone, with no
    third-party checkout to fetch. An entry in any other form is left to the server."""
    return frozenset(name for name in candidates if _by_name(config.get(name)))


def _by_name(entry: Any) -> bool:
    if not isinstance(entry, Mapping):
        return False
    return (
        entry.get("is_python_module", _TRUE) == _TRUE
        and entry.get("is_standalone", _FALSE) == _FALSE
        and entry.get("third_party") in _EMPTY_LISTS
    )


class AiterModules:
    """The :class:`~hyperloom.measure.jit.native.needs.ModulePlan` of a host that builds natively (see the module).

    ``planner`` keeps what is planned for the build runner; ``history`` gives the ready aiter builds of the image, which
    :meth:`seconds` estimates from; ``environ`` is the service's environment, whose build variables the compile gets.
    """

    def __init__(
        self,
        planner: NativePlanner,
        *,
        history: Callable[[], Iterable[BuildRecord]] = tuple,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self._planner, self._history = planner, history
        self._environ = os.environ if environ is None else environ
        self._recipe = aiter_module_recipe()
        self._package: Path | None = None

    def __call__(
        self, roots: tuple[EnvRoot, ...], config: Mapping[str, Any], candidates: Collection[str]
    ) -> ModuleBuild | None:
        modules = buildable_modules(config, candidates)
        if not modules:
            return None
        machine = self._planner.site.machine
        aiter = tuple(root for root in roots if root.import_name in (AITER_PACKAGE, AITER_META_WHEEL_DIR))
        variables = self._build_variables(machine.python, config, modules)
        digest = _digest(aiter, modules, variables)
        inputs = BuildInputs(
            machine.image_digest,
            machine.builder_image_digest,
            AITER_COMPONENT,
            machine.gfx_arch,
            digest,
            _RECIPE_VERSION,
        )
        self._planner.remember(inputs, _Planned(self._recipe, machine, aiter, modules, variables))
        return ModuleBuild(inputs, modules)

    def seconds(self, modules: Collection[str]) -> float:
        """How long the CPU build of ``modules`` takes: the median CPU seconds a module took in the image's earlier aiter
        builds, spread over the recipe's ``MAX_JOBS``; with none yet, what a module takes to build on a card."""
        per_module = [
            record.build_seconds / len(record.outputs)
            for record in self._history()
            if record.status == READY and record.outputs
        ]
        if not per_module:
            return float(len(modules) * ESTIMATED_MODULE_SECONDS)
        return len(modules) * statistics.median(per_module) / self._recipe.max_jobs

    def _build_variables(self, python: str, config: Mapping[str, Any], modules: Collection[str]) -> dict[str, str]:
        """The variables of the service's environment that the build of ``modules`` reads (``build_env_names``, the
        runner compares the same ones), less locations, the tuned tables and what the build sets itself, and the
        interpreter the compile runs: what the compile gets and its record states."""
        package = self._image_package(python)
        names = build_env_names(package, image_aiter_meta(package), config, modules)
        surface = build_surface(names, self._environ, {})
        variables = {k: v for k, v in surface.environment.items() if k not in _SET_BY_BUILD}
        variables[FRAMEWORK_PYTHON_ENV] = interpreter_identity(python)
        return variables

    def _image_package(self, python: str) -> Path:
        if self._package is None:
            try:
                self._package = locate_package(AITER_PACKAGE, python, Path.cwd())
            except FarmError as exc:
                raise BuildInfraError(f"the image's aiter cannot be found: {exc}") from None
        return self._package


def _blob(blobs: BlobStore, address: str) -> bytes:
    return blobs.get(address.removeprefix(DIGEST_PREFIX))


def _digest(roots: Iterable[EnvRoot], modules: Collection[str], variables: Mapping[str, str]) -> str:
    """The identity of what the build makes the modules from: the patched files of aiter's roots, the modules, the
    variables."""
    files = sorted(
        [root.id, entry.path, entry.op, entry.blob or "", entry.mode or ""] for root in roots for entry in root.entries
    )
    document = {"files": files, "modules": sorted(modules), "environment": dict(sorted(variables.items()))}
    return hashlib.sha256(canonical_json(document).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _Planned:
    """One planned build of aiter's modules, laid as the test's farm is."""

    recipe: NativeRecipe
    machine: BuildMachine
    roots: tuple[EnvRoot, ...]
    modules: frozenset[str]
    variables: Mapping[str, str]

    def lay(self, blobs: BlobStore, work: Path) -> LaidTree:
        """Lay the farm below ``work``'s tree from the roots as a sheet carries them, with no meta directory inherited,
        and name the modules to build."""
        tree = tree_dir(work)
        tree.mkdir()
        roots = [SheetRoot.parse(root.to_document(), f"roots[{index}]") for index, root in enumerate(self.roots)]
        try:
            farm = build_farm(roots, tree, python=self.machine.python, read_blob=partial(_blob, blobs), inherited={})
        except FarmError as exc:
            raise BuildInfraError(f"the tree of aiter's modules cannot be laid: {exc}") from None
        if farm is None or farm.aiter_meta is None:
            raise BuildInfraError("the tree of aiter's modules has no meta directory")
        plan = {
            "modules": sorted(self.modules),
            "jit": _JIT_DIR,
            "package": (farm.site / AITER_PACKAGE).relative_to(tree).as_posix(),
            "meta": farm.aiter_meta.relative_to(tree).as_posix(),
        }
        descriptor = os.open(tree / MODULES_FILE, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8") as out:
            json.dump(plan, out)
        image_modules = farm.site / AITER_PACKAGE / _JIT_DIR
        return LaidTree(self.recipe, self.machine, tree, image_modules, frozenset(), dict(self.variables))
