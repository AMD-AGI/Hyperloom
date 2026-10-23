# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The environment an image check looks things up in: the real one, and the one an example describes.

``HostImage`` answers from the machine it runs on. It never imports the projects it asks about, because importing aiter,
SGLang or vLLM loads GPU libraries: where a package lives is asked of the interpreter with ``importlib.util.find_spec``
in a child process, a module's source is read as a file, and a server's flags are read from the help its own command
line prints. ``ExampleImage`` answers every question with the one piece of content an ``examples`` entry carries, which
is how a pack's checks are pinned before anything is run for real.

A question the environment cannot answer (the interpreter does not start, a command times out, a file cannot be read)
raises ``ImageReadError``, which a check reports as unverifiable. An answer of ``None`` means the thing is absent, which
a check may report as a failure.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple, Protocol

from hyperloom.common.outside_facts.image_check_spec import FileType

__all__ = [
    "CommandResult",
    "ExampleImage",
    "HostImage",
    "Image",
    "ImageReadError",
    "Runner",
    "run_command",
]

_MAX_FILE_BYTES = 16 * 1024 * 1024
_DEFAULT_TIMEOUT_SEC = 120.0
_TAIL_CHARS = 200
_DECODE = "replace"
#: Run in the image's own interpreter, from a neutral directory: ``find_spec`` of a top-level name imports nothing.
_PROBE = """\
import importlib.metadata as md, importlib.util as u, json, sys
owners = md.packages_distributions()
found = {}
for name in sys.argv[1:]:
    try:
        spec = u.find_spec(name)
    except (ImportError, ValueError):
        continue
    if spec is None:
        continue
    try:
        version = md.version((owners.get(name) or [name])[0])
    except md.PackageNotFoundError:
        version = None
    found[name] = [spec.origin, list(spec.submodule_search_locations or []), version]
print(json.dumps({"python": sys.version.split()[0], "packages": found}))
"""


class ImageReadError(Exception):
    """The environment could not answer a question; what was asked is neither present nor absent."""


class CommandResult(NamedTuple):
    """How a command ended and what it printed."""

    returncode: int
    stdout: str
    stderr: str


Runner = Callable[[Sequence[str], Mapping[str, str], float], CommandResult]


def run_command(argv: Sequence[str], env: Mapping[str, str], timeout: float) -> CommandResult:
    """Run ``argv`` without a shell from ``/``, so that no directory of ours shadows a package of the image."""
    try:
        done = subprocess.run(
            list(argv),
            capture_output=True,
            text=True,
            errors=_DECODE,
            timeout=timeout,
            cwd="/",
            env=dict(env),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ImageReadError(f"{argv[0]} did not finish within {timeout:.0f}s") from exc
    except (OSError, subprocess.SubprocessError) as exc:
        raise ImageReadError(f"{argv[0]} could not be run: {type(exc).__name__}") from exc
    return CommandResult(done.returncode, done.stdout, done.stderr)


class Image(Protocol):
    """What a check may ask of an environment."""

    def prepare(self, packages: Iterable[str]) -> None:
        """Learn about ``packages`` in one step, so that the checks that follow ask nothing further of the interpreter."""

    def getenv(self, name: str) -> str | None:
        """The value of an environment variable, ``None`` when it is unset or empty."""

    def installed(self, package: str) -> bool:
        """Whether the interpreter can import the top-level ``package``."""

    def module_source(self, module: str) -> str | None:
        """The source text of ``module``, ``None`` when the package has no such module file."""

    def path_type(self, path: str) -> FileType | None:
        """Whether the absolute ``path`` is a file or a directory, ``None`` when it is neither."""

    def read_text(self, path: str) -> str | None:
        """The text of the file at the absolute ``path``, ``None`` when there is none."""

    def csv_columns(self, path: str) -> tuple[str, ...] | None:
        """The header of the CSV at ``path``, ``None`` when there is no such file."""

    def help_text(self, module: str, option: str) -> str:
        """What ``python -m <module> <option>`` prints when it succeeds."""

    def describe(self) -> dict[str, Any]:
        """What identifies the environment in a report: the interpreter and what it knows of the packages asked about."""


class _Package(NamedTuple):
    origin: str | None
    locations: tuple[str, ...]
    version: str | None


class HostImage:
    """The machine this process runs on, seen through ``python``: the interpreter whose packages count."""

    def __init__(
        self,
        *,
        python: str | None = None,
        root: Path | str = "/",
        environ: Mapping[str, str] | None = None,
        run: Runner = run_command,
        timeout: float = _DEFAULT_TIMEOUT_SEC,
    ) -> None:
        self._python = python or sys.executable
        self._root = Path(root)
        self._environ = dict(os.environ if environ is None else environ)
        self._run = run
        self._timeout = timeout
        self._packages: dict[str, _Package | None] = {}
        self._python_version: str | None = None
        self._probe_error: ImageReadError | None = None
        self._helps: dict[tuple[str, str], str | ImageReadError] = {}
        self._lock = threading.Lock()

    def prepare(self, packages: Iterable[str]) -> None:
        wanted = sorted({name for name in packages if name not in self._packages})
        if not wanted or self._probe_error is not None:
            return
        try:
            found = self._probe(wanted)
        except ImageReadError as exc:
            self._probe_error = exc
            raise
        for name in wanted:
            entry = found.get(name)
            self._packages[name] = None if entry is None else _Package(entry[0], tuple(entry[1]), entry[2])

    def _probe(self, wanted: Sequence[str]) -> dict[str, list[Any]]:
        done = self._run([self._python, "-c", _PROBE, *wanted], self._environ, self._timeout)
        if done.returncode != 0:
            raise ImageReadError(f"{self._python} could not report its packages (exit {done.returncode})")
        try:
            report = json.loads(done.stdout.strip().splitlines()[-1])
            self._python_version = report["python"]
            return dict(report["packages"])
        except (ValueError, IndexError, KeyError, TypeError) as exc:
            raise ImageReadError(f"{self._python} reported its packages in a form that cannot be read") from exc

    def _package(self, name: str) -> _Package | None:
        if self._probe_error is not None:
            raise self._probe_error
        if name not in self._packages:
            self.prepare([name])
        return self._packages[name]

    def getenv(self, name: str) -> str | None:
        return self._environ.get(name) or None

    def installed(self, package: str) -> bool:
        return self._package(package) is not None

    def module_source(self, module: str) -> str | None:
        top, *rest = module.split(".")
        found = self._package(top)
        if found is None:
            return None
        if not rest:
            return self._read(Path(found.origin)) if found.origin and found.origin.endswith(".py") else None
        for base in found.locations:
            for candidate in (Path(base, *rest).with_suffix(".py"), Path(base, *rest, "__init__.py")):
                if self._type_of(candidate) is FileType.FILE:
                    return self._read(candidate)
        return None

    def _under_root(self, path: str) -> Path:
        return self._root / path.lstrip("/")

    @staticmethod
    def _type_of(path: Path) -> FileType | None:
        try:
            if path.is_dir():
                return FileType.DIR
            return FileType.FILE if path.is_file() else None
        except OSError as exc:
            raise ImageReadError(f"{path.name} cannot be examined: {type(exc).__name__}") from exc

    def path_type(self, path: str) -> FileType | None:
        return self._type_of(self._under_root(path))

    @staticmethod
    def _read(path: Path) -> str:
        try:
            if path.stat().st_size > _MAX_FILE_BYTES:
                raise ImageReadError(f"{path.name} is larger than {_MAX_FILE_BYTES} bytes")
            return path.read_text(encoding="utf-8", errors=_DECODE)
        except OSError as exc:
            raise ImageReadError(f"{path.name} cannot be read: {type(exc).__name__}") from exc

    def read_text(self, path: str) -> str | None:
        target = self._under_root(path)
        return self._read(target) if self._type_of(target) is FileType.FILE else None

    def csv_columns(self, path: str) -> tuple[str, ...] | None:
        target = self._under_root(path)
        if self._type_of(target) is not FileType.FILE:
            return None
        try:
            with target.open(newline="", encoding="utf-8", errors=_DECODE) as handle:
                return tuple(next(csv.reader(handle), []))
        except (OSError, csv.Error) as exc:
            raise ImageReadError(f"{target.name} cannot be read as CSV: {type(exc).__name__}") from exc

    def help_text(self, module: str, option: str) -> str:
        key = (module, option)
        with self._lock:
            known = self._helps.get(key)
        if known is None:
            known = self._ask_for_help(module, option)
            with self._lock:
                self._helps[key] = known
        if isinstance(known, ImageReadError):
            raise known
        return known

    def _ask_for_help(self, module: str, option: str) -> str | ImageReadError:
        try:
            done = self._run([self._python, "-m", module, option], self._environ, self._timeout)
        except ImageReadError as exc:
            return exc
        if done.returncode != 0:
            printed = (done.stderr or done.stdout).strip().splitlines()
            tail = printed[-1][-_TAIL_CHARS:] if printed else "it printed nothing"
            return ImageReadError(f"python -m {module} {option} exited {done.returncode}: {tail}")
        return f"{done.stdout}\n{done.stderr}"

    def describe(self) -> dict[str, Any]:
        packages = {
            name: {"installed": found is not None, "version": None if found is None else found.version}
            for name, found in self._packages.items()
        }
        return {"python": self._python, "python_version": self._python_version, "packages": packages}


class ExampleImage:
    """The environment one ``examples`` entry describes: every question is answered from the content it carries."""

    def __init__(self, example: Mapping[str, Any]) -> None:
        self._example = example
        self._variables: Mapping[str, str] = example.get("env", {})
        self._tree = frozenset(example.get("tree", ()))

    def prepare(self, packages: Iterable[str]) -> None:
        """Nothing to learn: an example is about the image its check is about."""

    def getenv(self, name: str) -> str | None:
        return self._variables.get(name) or None

    def installed(self, package: str) -> bool:
        return True

    def module_source(self, module: str) -> str | None:
        return self._example.get("source")

    def path_type(self, path: str) -> FileType | None:
        if path in self._variables.values() or any(entry.startswith(path.rstrip("/") + "/") for entry in self._tree):
            return FileType.DIR
        return FileType.FILE if path in self._tree else None

    def read_text(self, path: str) -> str | None:
        return self._example.get("source")

    def csv_columns(self, path: str) -> tuple[str, ...] | None:
        header = self._example.get("header")
        return None if header is None else tuple(next(csv.reader([header])))

    def help_text(self, module: str, option: str) -> str:
        return self._example["help"]

    def describe(self) -> dict[str, Any]:
        return {}
