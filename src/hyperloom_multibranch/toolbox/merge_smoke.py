# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The merge smoke: whether a test's patched framework compiles and imports, checked on CPU before any card is leased.

A patch stack whose Python does not byte-compile, or whose framework no longer imports, fails every test it is measured
in, and the hosted executor would only learn it from the server's boot log after a sandbox, a cold start and GPU minutes.
:meth:`MergeSmoke.check` asks first, in seconds, as the first step of every test that carries patches (a merge test and
any candidate test alike):

1. every ``.py`` file the test's whole patch stack leaves different from the image (the compiled environment's entries:
   the net result of each patch's targets, :func:`~hyperloom.measure.compile.patch_targets`, over every component) is
   byte-compiled with ``compile()`` on its source, so nothing is written next to it;
2. the top-level module of each component the stack patches is imported from a farm of the patched package laid the way
   the runner lays it (:func:`~hyperloom.measure.farm.build_farm`), by the interpreter the test's server runs with (the
   point's ``HYPERLOOM_FRAMEWORK_PYTHON``, else the interpreter the toolbox compiled the stack against). The child sees
   no GPU (``ROCR_VISIBLE_DEVICES``, ``HIP_VISIBLE_DEVICES`` and ``CUDA_VISIBLE_DEVICES`` are empty: the first is
   ROCm's runtime-level mask, beneath HIP's, so a code path that reaches the ROCm runtime without HIP would still see the
   host's cards if it were left set), writes no bytecode, launches no server and is stopped at
   :attr:`MergeSmoke.timeout_sec`.

A file that does not compile, or a module whose import fails with an error judged below while the image's own copy
imports, is the candidate's error: :meth:`MergeSmoke.check` returns its first line, and the executor ends the test as a
candidate failure with the reason ``merge_smoke``. A check that cannot run (no interpreter, a blob missing, a farm that cannot be laid, a child that
timed out or died) raises :class:`SmokeNotRun`, which the executor reports as an infrastructure failure
(``smoke_not_run``), so an environment fault never bars a good commit. A module the image's own copy cannot import
either here (a framework that needs a GPU to import) is not judged: the test goes on to be measured as before.

Only an error that a missing GPU cannot cause is judged, since the image's own copy never runs the patch's new lines and
so cannot clear one: a ``SyntaxError`` in a file the stack patched; an ``ImportError`` (not a ``ModuleNotFoundError``)
for a name a patched module lacks (its ``path`` is a file the stack patched); an ``AttributeError`` raised on a patched
module object. This is the catch the smoke is for: one patch removes or renames a name that another patched file
imports. Anything else (a ``NameError``, a missing module, a name or attribute an unpatched module or value lacks, such
as one the image defines only where a GPU is seen or a device query that returns ``None``, a ``RuntimeError`` of a
device probe, a ``KeyError`` of a variable the host lacks) is logged and not judged, and the test goes on to its
sandbox. The import child gets every variable the point launches its server with, over the host's environment, still
with every GPU hidden.

Not checked: native and kernel sources (``.cu``, ``.hip``, Triton's JIT), anything that fails only at run time, and the
modules a top-level import does not reach. A runtime root (a stack whose native build the test waits for) has its
``.py`` files compiled but is not imported, since its native outputs are not built yet; and since the other components
may import what its package adds, an import failure of a stack that holds one is not judged either.

``HYPERLOOM_MERGE_SMOKE=off`` in the toolbox host's environment turns the check off (:func:`merge_smoke_from`).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import yaml

from hyperloom.common.outside_facts import AITER_META_DIR_ENV, AITER_META_WHEEL_DIR, ROCR_VISIBLE_DEVICES_ENV
from hyperloom.measure.compile import BlobError, CompiledEnvironment, LocalBlobStore
from hyperloom.measure.farm import FarmError, build_farm, named_python, prepend_pythonpath, server_environment
from hyperloom.measure.spec import EnvRoot as SheetRoot

__all__ = [
    "MERGE_SMOKE",
    "MERGE_SMOKE_ENV",
    "SMOKE_HIDDEN_DEVICES",
    "SMOKE_NOT_RUN",
    "SMOKE_TIMEOUT_SEC",
    "MergeSmoke",
    "SmokeNotRun",
    "merge_smoke_from",
]

log = logging.getLogger(__name__)

#: The reason a test whose patched Python does not compile or import ends with, as a candidate failure.
MERGE_SMOKE = "merge_smoke"
#: The reason a test whose smoke could not run ends with, as an infrastructure failure.
SMOKE_NOT_RUN = "smoke_not_run"
#: The toolbox host's variable that turns the smoke off (``off``); unset, empty or ``on`` keeps it on.
MERGE_SMOKE_ENV = "HYPERLOOM_MERGE_SMOKE"
#: How long one child of the smoke (the compile, the import, the image's import) may run.
SMOKE_TIMEOUT_SEC = 120.0
#: The variables that hide every GPU from the smoke's children, each set empty. ``ROCR_VISIBLE_DEVICES`` is ROCm's
#: runtime-level mask, applied beneath HIP's, so leaving the host's value would keep its cards visible to the runtime.
SMOKE_HIDDEN_DEVICES = (ROCR_VISIBLE_DEVICES_ENV, "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
_UPSERT = "upsert"
_RUNTIME = "runtime"
_DETAIL_CHARS = 400
_UTF8 = "utf-8"

#: Byte-compiles each ``[name, path]`` of ``argv[1]`` with ``compile()`` (no bytecode is written) and writes the first
#: error, or none, to ``argv[2]``.
_COMPILE = """\
import json, sys
files, out = json.load(open(sys.argv[1])), sys.argv[2]
error = None
for name, path in files:
    with open(path, 'rb') as handle:
        source = handle.read()
    try:
        compile(source, name, 'exec', dont_inherit=True)
    except SyntaxError as exc:
        error = '%s:%s: %s: %s' % (name, exc.lineno, type(exc).__name__, exc.msg)
        break
    except ValueError as exc:
        error = '%s: %s: %s' % (name, type(exc).__name__, exc)
        break
with open(out, 'w') as handle:
    json.dump({'error': error}, handle)
"""
#: Imports each module of ``argv[1]`` (``modules``) and writes the first that fails, with where it was raised and whether
#: it is one the smoke judges, or none, to ``argv[2]``. ``patched`` holds the farm's paths of the files the stack changes;
#: judged: a ``SyntaxError`` in one of them, an ``ImportError`` that is not a ``ModuleNotFoundError`` whose ``path`` (the
#: module lacking the name) is one of them, or an ``AttributeError`` raised on a module object whose file is one of them.
#: Where it was raised is the last frame of the traceback outside the import machinery's own frozen frames.
_IMPORT = """\
import importlib, json, os, sys, traceback
argument, out = json.load(open(sys.argv[1])), sys.argv[2]
patched = set(os.path.normpath(path) for path in argument['patched'])
error = None
for module in argument['modules']:
    try:
        importlib.import_module(module)
    except BaseException as exc:
        frames = [f for f in traceback.extract_tb(exc.__traceback__) if not f.filename.startswith('<frozen ')]
        if isinstance(exc, SyntaxError) and exc.filename:
            filename, lineno = exc.filename, exc.lineno
        elif frames:
            filename, lineno = frames[-1].filename, frames[-1].lineno
        else:
            filename, lineno = '', None
        if isinstance(exc, SyntaxError):
            lacking = filename
        elif isinstance(exc, ImportError) and not isinstance(exc, ModuleNotFoundError):
            lacking = exc.path
        elif isinstance(exc, AttributeError) and type(getattr(exc, 'obj', None)) is type(sys):
            lacking = getattr(exc.obj, '__file__', None)
        else:
            lacking = None
        judged = isinstance(lacking, str) and bool(lacking) and os.path.normpath(lacking) in patched
        where = '%s:%s' % (filename, lineno) if filename else ''
        error = {'module': module, 'error': '%s: %s' % (type(exc).__name__, exc), 'where': where, 'judged': judged}
        break
with open(out, 'w') as handle:
    json.dump({'error': error}, handle)
"""


class _Step(NamedTuple):
    """One child of the smoke: its name in a reason, its script and the interpreter flags it runs with."""

    name: str
    script: str
    flags: tuple[str, ...] = ()


#: ``-I``: the compile reads no ``PYTHON*`` variable and puts no directory of the host on its path.
_COMPILE_STEP = _Step("compile", _COMPILE, ("-I",))
_IMPORT_STEP = _Step("import", _IMPORT)
_IMAGE_IMPORT_STEP = _Step("image-import", _IMPORT)


class SmokeNotRun(Exception):
    """The smoke could not give an answer about the candidate: the environment failed, not the change."""


@dataclass(frozen=True)
class MergeSmoke:
    """The merge smoke of one toolbox.

    ``default_python`` is the interpreter a point that names none is smoked with: the toolbox's own, which the compile
    found the image's packages with. Each check lays its farm in a directory of its own under the host's temporary
    directory and removes it after the check.
    """

    timeout_sec: float = SMOKE_TIMEOUT_SEC
    default_python: str = sys.executable

    def check(self, environment: object, points: Sequence[Path]) -> str | None:
        """The first error of the candidate's patched Python, or ``None`` when it compiles and imports (or carries no
        patch to check). Raises :class:`SmokeNotRun` when the check cannot run."""

        if not isinstance(environment, CompiledEnvironment) or not environment.roots:
            return None
        config = _point_config(points)
        python = _resolved(named_python(config) or self.default_python)
        work = Path(tempfile.mkdtemp(prefix="hl-merge-smoke-"))
        try:
            store = LocalBlobStore(environment.blob_root)
            error = self._compile(python, environment, store, work)
            if error is not None:
                return error
            return self._import(python, environment, store, config, work)
        finally:
            shutil.rmtree(work, ignore_errors=True)

    # -- step 1: byte-compile ----------------------------------------------------------------------------------------

    def _compile(self, python: str, environment: CompiledEnvironment, store: LocalBlobStore, work: Path) -> str | None:
        files: list[list[str]] = []
        sources = work / "sources"
        for root in environment.roots:
            for entry in _patched_py(root):
                target = sources / str(len(files))
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(_blob(store, entry.blob or ""))
                files.append([f"{root.import_name}/{entry.path}", str(target)])
        if not files:
            return None
        found = self._child(_COMPILE_STEP, python, files, work, os.environ)
        return _one_line(str(found)) if found else None

    # -- step 2: import ----------------------------------------------------------------------------------------------

    def _import(
        self,
        python: str,
        environment: CompiledEnvironment,
        store: LocalBlobStore,
        config: Mapping[str, Any],
        work: Path,
    ) -> str | None:
        roots = [root for root in environment.roots if root.kind != _RUNTIME]
        modules = list(dict.fromkeys(root.import_name for root in roots if root.import_name != AITER_META_WHEEL_DIR))
        if not modules:
            return None
        sheet = [SheetRoot.parse(root.to_document(), "environment.roots") for root in roots]
        yaml_envs = {name: str(value) for name, value in server_environment(config).items() if value is not None}
        image_env = _smoke_env(os.environ, yaml_envs)
        try:
            farm = build_farm(
                sheet, work / "farm", python=python, read_blob=lambda a: _blob(store, a), inherited=image_env
            )
        except (FarmError, OSError) as exc:
            raise SmokeNotRun(f"the patched framework cannot be laid for the import: {_one_line(str(exc))}") from None
        if farm is None:
            return None
        patched_envs = dict(yaml_envs)
        prepend_pythonpath(patched_envs, farm.site)
        if farm.aiter_meta is not None:
            patched_envs.setdefault(AITER_META_DIR_ENV, str(farm.aiter_meta))
        files = [str(farm.site / root.import_name / entry.path) for root in roots for entry in _patched_py(root)]
        argument = {"modules": modules, "patched": files}
        failed = self._child(_IMPORT_STEP, python, argument, work, _smoke_env(os.environ, patched_envs))
        if not failed:
            return None
        site = f"{farm.site}{os.sep}"
        where = str(failed.get("where") or "").removeprefix(site)
        error = str(failed.get("error")).replace(site, "")
        return self._verdict(python, environment, {**failed, "where": where, "error": error}, work, image_env)

    def _verdict(
        self,
        python: str,
        environment: CompiledEnvironment,
        failed: Mapping[str, Any],
        work: Path,
        image_env: Mapping[str, str],
    ) -> str | None:
        """The candidate's error of the failed import ``failed`` (its ``where`` relative to the farm), or ``None``
        (logged) when it is not judged."""
        module = str(failed.get("module"))
        why = self._not_judged(python, environment, failed, work, image_env)
        if why is not None:
            log.warning(
                "merge smoke: %s does not import here, but %s, so the import is not judged: %s",
                module,
                why,
                _one_line(str(failed.get("error"))),
            )
            return None
        where = failed.get("where")
        return _one_line(f"import {module}: {failed.get('error')}" + (f" ({where})" if where else ""))

    def _not_judged(
        self,
        python: str,
        environment: CompiledEnvironment,
        failed: Mapping[str, Any],
        work: Path,
        image_env: Mapping[str, str],
    ) -> str | None:
        """Why the failed import ``failed`` says nothing about the candidate, or ``None`` when it does."""
        if not failed.get("judged"):
            where = failed.get("where") or "no frame"
            return f"the error (at {where}) is not a name or attribute a patched module lacks, which no GPU could cause"
        native = [root.import_name for root in environment.roots if root.kind == _RUNTIME]
        if native:
            return f"the native stacks {', '.join(native)} are not laid until their builds are ready"
        image_import = {"modules": [str(failed.get("module"))], "patched": []}
        if self._child(_IMAGE_IMPORT_STEP, python, image_import, work, image_env):
            return "the image's own copy does not import here either"
        return None

    # -- a child -----------------------------------------------------------------------------------------------------

    def _child(self, step: _Step, python: str, argument: object, work: Path, env: Mapping[str, str]) -> Any:
        """Run ``step`` with ``python`` on ``argument``; what it found (``None``: nothing). Raises :class:`SmokeNotRun`
        when it cannot be started, times out or ends without an answer."""

        name = step.name
        directory = work / name
        directory.mkdir(parents=True, exist_ok=True)
        argv_file, out = directory / "in.json", directory / "out.json"
        argv_file.write_text(json.dumps(argument), encoding=_UTF8)
        command = [python, *step.flags, "-B", "-c", step.script, str(argv_file), str(out)]
        try:
            done = subprocess.run(
                command,
                cwd=directory,
                env={**env, "PYTHONDONTWRITEBYTECODE": "1", **dict.fromkeys(SMOKE_HIDDEN_DEVICES, "")},
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=self.timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise SmokeNotRun(f"the {name} check timed out after {self.timeout_sec:g} s") from None
        except OSError as exc:
            raise SmokeNotRun(f"the {name} check could not start {python}: {_one_line(str(exc))}") from None
        try:
            answer = json.loads(out.read_text(encoding=_UTF8))
        except (OSError, ValueError):
            tail = _one_line(done.stderr.strip()[-_DETAIL_CHARS:]) or "no output"
            raise SmokeNotRun(f"the {name} check ended without an answer (exit {done.returncode}): {tail}") from None
        return answer.get("error") if isinstance(answer, dict) else None


def merge_smoke_from(environ: Mapping[str, str]) -> MergeSmoke | None:
    """The smoke the host's environment asks for: on unless :data:`MERGE_SMOKE_ENV` is ``off``.

    Raises:
        ValueError: When the variable holds anything but ``on``, ``off`` or nothing.
    """

    value = environ.get(MERGE_SMOKE_ENV, "").strip().lower()
    if value in ("", "on"):
        return MergeSmoke()
    if value == "off":
        return None
    raise ValueError(f"{MERGE_SMOKE_ENV}={value!r}: expected 'on' or 'off'")


def _point_config(points: Sequence[Path]) -> Mapping[str, Any]:
    if not points:
        return {}
    try:
        loaded = yaml.safe_load(Path(points[0]).read_text(encoding=_UTF8))
    except (OSError, yaml.YAMLError) as exc:
        raise SmokeNotRun(f"the test's point cannot be read: {_one_line(str(exc))}") from None
    return loaded if isinstance(loaded, dict) else {}


def _resolved(python: str) -> str:
    found = shutil.which(python)
    if found is None:
        raise SmokeNotRun(f"the framework interpreter {python} is not here")
    return found


def _blob(store: LocalBlobStore, address: str) -> bytes:
    try:
        return store.get(address.removeprefix("sha256:"))
    except BlobError as exc:
        raise SmokeNotRun(f"a patched file cannot be read: {exc}") from None


def _patched_py(root: Any) -> list[Any]:
    """The ``.py`` files ``root`` writes."""
    return [entry for entry in root.entries if entry.op == _UPSERT and entry.path.endswith(".py")]


def _smoke_env(environ: Mapping[str, str], yaml_envs: Mapping[str, str]) -> dict[str, str]:
    """This host's environment, without its ``PYTHONPATH``, with every variable the point launches its server with over
    it, so a module that reads a recipe's variable at import finds it. The GPUs stay hidden (:meth:`MergeSmoke._child`
    sets :data:`SMOKE_HIDDEN_DEVICES` last)."""
    env = {name: value for name, value in environ.items() if name != "PYTHONPATH"}
    env.update(yaml_envs)
    return env


def _one_line(text: str) -> str:
    return " ".join(text.split())[:_DETAIL_CHARS]
