# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The key a compiled-code cache is stored under: exactly the inputs that decide what the compiler produces (design 4.21).

Each kind of cache is keyed by what its own compiler contract leaves uncovered, and nothing else, so two candidates that
differ only in server arguments, in a framework patch that does not touch the build, or in the environment of the
request share one cache:

* **Triton** names a kernel's directory by the hash of its source, signature, options, target and its own compiler, so
  an entry is content-addressed already; the key is the image (which fixes Triton and the toolchain) and the GPU
  architecture.
* **aiter** names a module only by ``md_name`` and builds nothing when ``<md_name>.so`` exists, whatever changed under
  it. The name does encode what a call asks for (``gen_func`` puts dtype, bias, mask and the like into ``md_name`` and
  the code generator command; a cpp_itfs library's directory is ``md_name`` plus the hash of every template argument),
  so calls only choose among entries. What the name does not cover is the build context, and the key states all of it:
  the image, the GPU, the aiter source tree (the patch laid over the image's), the content of the tuned tables the code
  generators read, and the value of every environment variable the build path reads.

Which variables the build path reads is found in the source the test will run (:func:`scan_env_names`), not listed here.
A framework patch can change the build context from inside the server (set a variable, patch aiter's builder); that is
not visible in the sheet, so a patch whose text shows such a thing is made part of the key (:func:`touches_build_surface`).
That test errs toward fewer shared entries, never toward a leak.

Variables that only name a location are found by that scan and left out of the key (:func:`build_surface`), because what
they name is where something is kept or made, never an input of what is made:

* ``AITER_JIT_DIR``, ``AITER_ROOT_DIR`` and ``TRITON_CACHE_DIR`` are the cache directories themselves; the session sets
  each of them to the test's own for every kind it caches, so the value the sheet or the runner had never reaches the server.
* ``HOME`` and ``USERPROFILE`` are read by aiter for one purpose: to default ``AITER_ROOT_DIR`` to ``$HOME/.aiter``
  (``csrc/cpp_itfs/utils.py``) and the user's ``jit`` directory to ``~/.aiter/jit`` (``aiter/jit/core.py``), the roots of
  the two caches above. Every local sandbox has a home of its own, so keeping them made every test a different key and the
  cache never hit.
* ``TMPDIR``, ``TEMP`` and ``TMP`` are where the compiler and ninja put scratch files that are gone when the build is done,
  and the ``XDG_*`` variables are the base directories under which a tool keeps its own per-user cache, configuration and state.

What a build compiles is decided by the source, the flags and the variables that select them, all of which stay in the key.
A tool of the image that read its configuration from one of these directories would be reading a per-sandbox accident that
no test is meant to depend on; the key already fixes the image, and treats code the candidate itself can run as read only
(``KindPlan.read_only``).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from hyperloom.common.outside_facts import AITER_LOG_TUNED_CONFIG_ENV
from hyperloom.measure.docs import canonical_json

__all__ = [
    "AITER_PACKAGE",
    "BuildSurface",
    "Machine",
    "aiter_key",
    "build_surface",
    "scan_env_names",
    "touches_build_surface",
    "tree_identity",
    "triton_key",
]

#: 2: Triton's group files are stored with paths relative to the cache directory, so a snapshot of schema 1 is not read.
SCHEMA = 2
AITER_PACKAGE = "aiter"
#: Variables that only name a location, one that differs between sandboxes; the scan finds them, the surface leaves them out.
_LOCATIONS = frozenset(
    {
        "AITER_JIT_DIR",
        "AITER_ROOT_DIR",
        "TRITON_CACHE_DIR",
        "HOME",
        "USERPROFILE",
        "TMPDIR",
        "TEMP",
        "TMP",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_STATE_HOME",
        "XDG_RUNTIME_DIR",
    }
)
#: Variables that change how fast or how loudly aiter builds, not what it builds.
_NEUTRAL = frozenset(
    {"MAX_JOBS", "PREBUILD_THREAD_NUM", "AITER_JIT_VERBOSE", "AITER_LOG_MORE", AITER_LOG_TUNED_CONFIG_ENV}
)
#: ``AITER_REBUILD`` discards what is cached and builds again: a test that sets it has no cache to use or to add to.
REBUILD_VARIABLE = "AITER_REBUILD"
#: Variables the tuned tables are named by; their files' content, not their paths, is what a code generator reads.
_TABLE_PREFIX = "AITER_CONFIG_"
_ENV_READ = re.compile(
    r"""(?:os\.environ\.get|os\.getenv|os\.environ\.setdefault)\(\s*["']([A-Za-z_][A-Za-z0-9_]*)["']"""
    r"""|os\.environ\[\s*["']([A-Za-z_][A-Za-z0-9_]*)["']\s*\]"""
    r"""|["']([A-Za-z_][A-Za-z0-9_]*)["']\s+(?:not\s+)?in\s+os\.environ"""
)
#: Where, below the aiter package and the checkout around it, the code that builds and generates code is.
_BUILD_PATH = ("jit", "../csrc", "../hsa", "../aiter_meta")
_MAX_SCANNED_BYTES = 2 * 1024**2
#: Text of a patch that can change what aiter builds from inside the server process, other than by naming a variable.
_BUILD_SURFACE_WORDS = ("AITER_", "aiter.jit", "build_module", "compile_ops", "putenv", "unsetenv")
_ENV_WRITE = re.compile(
    r"environ\s*\.\s*(?:update|setdefault|pop|clear)\b"  # a mapping call that writes
    r"|environ\s*\[\s*[^\s'\"\]][^\]]*\]\s*=(?!=)"  # a store under a name that is not a literal
)


@dataclass(frozen=True)
class Machine:
    """What the compiled code is for: the image (toolchain, torch, Triton, aiter's base tree) and the GPU it targets.

    ``partition_mode`` stands for ``cu_num``, which aiter reads from the card and which follows from the board and how
    it is partitioned.
    """

    image_digest: str
    gfx_arch: str
    gpu_type: str
    partition_mode: str | None

    def document(self) -> dict[str, object]:
        return {
            "image": self.image_digest,
            "gfx": self.gfx_arch,
            "gpu_type": self.gpu_type,
            "partition": self.partition_mode,
        }


@dataclass(frozen=True)
class BuildSurface:
    """What the aiter build path reads of the server's environment: tuned-table content and plain variable values."""

    tables: Mapping[str, str] = field(default_factory=dict)
    environment: Mapping[str, str] = field(default_factory=dict)
    rebuild: bool = False


def _hash(document: dict[str, object]) -> str:
    return hashlib.sha256(canonical_json(document).encode("utf-8")).hexdigest()


def triton_key(machine: Machine) -> str:
    """Triton's own hashing makes an entry content-addressed; the key only fixes the compiler and the target."""
    return _hash({"schema": SCHEMA, "kind": "triton", "image": machine.image_digest, "gfx": machine.gfx_arch})


def aiter_key(machine: Machine, tree: str, surface: BuildSurface) -> str:
    """The key of the aiter caches: the machine, the source tree's identity and what the build path reads."""
    return _hash(
        {
            "schema": SCHEMA,
            "kind": AITER_PACKAGE,
            "machine": machine.document(),
            "tree": tree,
            "tables": dict(sorted(surface.tables.items())),
            "environment": dict(sorted(surface.environment.items())),
        }
    )


def tree_identity(
    overlay: Iterable[tuple[str, ...]], runtime_manifest: str | None, framework: Iterable[tuple[str, ...]]
) -> str:
    """The identity of the aiter source tree the test builds from, beyond the image's own.

    ``overlay`` is every entry the sheet lays over the aiter package (path, operation, blob, mode); ``runtime_manifest`` the
    hash of a runtime snapshot (it carries the sources outside the package, so its presence is part of the identity);
    ``framework`` the (package, path, blob) of each framework patch that touches the build surface.
    """
    return _hash(
        {
            "overlay": sorted(list(entry) for entry in overlay),
            "runtime": runtime_manifest,
            "framework": sorted(list(entry) for entry in framework),
        }
    )


def scan_env_names(package_dirs: Iterable[Path]) -> frozenset[str]:
    """The environment variables the aiter build path reads, found in the source a test will run.

    Reads the Python files of the build path below each aiter package directory: the package's ``jit`` directory, the
    ``csrc`` code generators, ``hsa/codegen.py`` and the ``aiter_meta`` copy of a wheel install. A variable is named when
    the code reads it with ``os.getenv``, ``os.environ.get``, ``os.environ[...]`` or ``in os.environ``; a name found there that
    the build never reads only makes the key finer.
    """
    names: set[str] = set()
    for package in package_dirs:
        for relative in _BUILD_PATH:
            for path in _python_files((Path(package) / relative).resolve()):
                names.update(_names_in(path))
    return frozenset(names)


def _python_files(directory: Path) -> Iterable[Path]:
    if not directory.is_dir():
        return ()
    return (path for path in directory.rglob("*.py") if path.is_file())


def _names_in(path: Path) -> set[str]:
    try:
        with open(path, "rb") as handle:
            text = handle.read(_MAX_SCANNED_BYTES).decode("utf-8", errors="replace")
    except OSError:
        return set()
    return {name for match in _ENV_READ.finditer(text) for name in match.groups() if name}


def build_surface(
    names: Iterable[str], environment: Mapping[str, str], table_content: Mapping[str, str]
) -> BuildSurface:
    """The part of ``environment`` the build reads: a table variable by the content ``table_content`` gives for it, any
    other variable by its value, minus the variables that only name a location or change only speed or noise (``names`` is
    what :func:`scan_env_names` found, which still has them)."""
    named = set(names) - _LOCATIONS - _NEUTRAL - {REBUILD_VARIABLE}
    tables = {
        name: table_content[name] for name in sorted(named) if name.startswith(_TABLE_PREFIX) and name in table_content
    }
    plain = {
        name: environment[name] for name in sorted(named) if not name.startswith(_TABLE_PREFIX) and name in environment
    }
    return BuildSurface(tables, plain, rebuild=_rebuilds(environment))


def _rebuilds(environment: Mapping[str, str]) -> bool:
    value = (environment.get(REBUILD_VARIABLE) or "0").strip()
    return value.lstrip("+-").isdigit() and int(value) > 0


def touches_build_surface(text: str, names: Iterable[str]) -> bool:
    """Whether a framework patch's text can change what aiter builds from inside the server process.

    It does when it mentions a variable the build reads (or the ``AITER_`` family), aiter's builder, or writes the
    environment other than by a store under a literal name (``environ.update``, ``setdefault``, ``pop``, ``putenv``, a
    store under a computed name). Reading some other variable, or storing under a named one, does not.
    """
    if any(word in text for word in _BUILD_SURFACE_WORDS):
        return True
    if any(re.search(rf"\b{re.escape(name)}\b", text) for name in names):
        return True
    return _ENV_WRITE.search(text) is not None
