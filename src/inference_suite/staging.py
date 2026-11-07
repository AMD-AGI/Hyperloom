# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Offline staging (D11): the pinned tools copied into the tool cache ahead of time, and the check a round runs on them.

The tool cache holds each tool under ``<tool>/<commit>/`` and one manifest, ``staging_manifest.json``, naming every
staged file with its sha256. A round fetches nothing: :func:`verify` lists what is absent, changed or staged at a
commit other than its pin, including a manifest that does not stage a file a load runs (:data:`INFERENCEX_REQUIRED`),
and a round with anything listed refuses with ``staging_missing``. The manifest's own sha256 is the round's
``tool_manifest_sha256``, so a launch names exactly the bytes it ran.

The AgentX harness runs in its own venv beside the InferenceX subset, installed from the requirements InferenceX pins
at its commit. Its manifest entry is the venv's interpreter and each installed distribution's ``RECORD``, which lists
every installed file with its sha256, so the manifest names the exact resolved set without hashing the whole venv.

``python -m inference_suite.staging stage --source <InferenceX checkout> --tool-cache <dir>`` stages, and ``check
--tool-cache <dir>`` reports whether the cache holds the current pins intact; Hyperloom's ``install.sh`` runs both.
:func:`ensure_staged`, behind ``python -m inference_suite stage`` and this package's ``install.sh``, stages only what
a cache lacks.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from hyperloom_common.contract_fields import ContractFields, field_names
from inference_suite.contracts.report import FailedStep
from inference_suite.pins import (
    INFERENCEX_REPO_URL,
    INFERENCEX_TOOL,
    LM_EVAL_DISTRIBUTION,
    LM_EVAL_REQUIREMENT,
    LM_EVAL_VERSION,
    PINNED_TOOLS,
)

STAGING_SCHEMA = "inference-suite-staging/1"
MANIFEST_FILE = "staging_manifest.json"
#: The project directory of an InferenceX checkout that holds the ``infx`` package.
INFERENCEX_PROJECT_DIR = "inferencex-e2e"
#: The GSM8K task file the eval scores, under the staged InferenceX root.
GSM8K_TASK = "infx/evals/gsm8k.yaml"
#: The module the fixed load runs, from the staged InferenceX root.
CLIENT_MODULE = "infx.bench_serving.benchmark_serving"
#: The file :data:`CLIENT_MODULE` loads, relative to the staged InferenceX root.
CLIENT_FILE = f"{CLIENT_MODULE.replace('.', '/')}.py"
#: The AgentX harness requirements InferenceX pins, under the staged InferenceX root.
AGENTX_REQUIREMENTS = "infx/bench/agentic/requirements.txt"
#: The module that aggregates an AgentX replay's export, from the staged InferenceX root.
AGENTX_AGGREGATOR_MODULE = "infx.results.agentic.process_agentic_result"
AGENTX_AGGREGATOR_FILE = f"{AGENTX_AGGREGATOR_MODULE.replace('.', '/')}.py"
#: The package the aggregator imports its builders from, staged whole.
AGENTX_RESULTS_PACKAGE = "infx/results"
#: The harness venv, under the staged InferenceX root, and the interpreter and CLI module the AgentX load runs.
AGENTX_VENV_DIR = "agentx-venv"
AGENTX_PYTHON = f"{AGENTX_VENV_DIR}/bin/python"
AGENTX_CLI_MODULE = "agentx_cli"
#: The interpreters the harness supports (``agentx-harness`` 1.0.13 declares 3.11 to 3.13).
AGENTX_PYTHON_VERSIONS = tuple((3, minor) for minor in range(11, 14))
#: What a round and its eval run from InferenceX, relative to its project directory.
INFERENCEX_SUBSET = (
    "infx/__init__.py",
    CLIENT_MODULE.rpartition(".")[0].replace(".", "/"),
    GSM8K_TASK,
    AGENTX_RESULTS_PACKAGE,
    AGENTX_REQUIREMENTS,
)
#: The files a staged InferenceX must hold for a round to run, whatever else its subset holds at the pin.
INFERENCEX_REQUIRED = (CLIENT_FILE, GSM8K_TASK, AGENTX_AGGREGATOR_FILE, AGENTX_PYTHON)
#: The file each installed distribution lists its installed files in, with their sha256.
_DIST_RECORD = "RECORD"

_COMMIT = re.compile(r"[0-9a-f]{40}")
_GIT = "git"
_SKIPPED_DIRS = frozenset({"__pycache__"})
_SCHEMA_KEY = "schema"
_TOOLS_KEY = "tools"


class StagingError(ValueError):
    """A staging manifest breaks its contract, or a tool cannot be staged."""


_FIELDS = ContractFields(StagingError)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class StagedTool:
    """One tool as staged: its commit and each file under its root, by relative path, with the file's sha256."""

    commit: str
    files: Mapping[str, str]

    def __post_init__(self) -> None:
        _FIELDS.matching(_COMMIT, self.commit, "the staged commit")
        files = _FIELDS.string_map(self.files, "the staged files")
        if not files:
            raise StagingError(f"the tool staged at {self.commit} lists no files")
        for path, digest in files.items():
            if Path(path).is_absolute() or ".." in Path(path).parts:
                raise StagingError(f"the staged file {path!r} is not a path inside the tool's root")
            _FIELDS.sha256(digest, f"the sha256 of {path}")
        object.__setattr__(self, "files", MappingProxyType(dict(sorted(files.items()))))

    def to_document(self) -> dict[str, object]:
        return {"commit": self.commit, "files": dict(self.files)}

    @classmethod
    def from_document(cls, document: object) -> StagedTool:
        keys = field_names(cls)
        fields = _FIELDS.known(document, keys, "a staged tool")
        return cls(*(fields.get(key) for key in keys))


@dataclass(frozen=True)
class StagingManifest:
    """Every staged tool, by the name :mod:`inference_suite.pins` gives it."""

    tools: Mapping[str, StagedTool]

    def __post_init__(self) -> None:
        tools = _FIELDS.mapping(self.tools, "the staged tools")
        if not tools:
            raise StagingError("the staging manifest lists no tools")
        for name, tool in tools.items():
            _FIELDS.text(name, "a staged tool name")
            if not isinstance(tool, StagedTool):
                raise StagingError(f"the staged tool {name} must be a StagedTool, got {tool!r}")
        object.__setattr__(self, "tools", MappingProxyType(dict(sorted(tools.items()))))

    def to_document(self) -> dict[str, object]:
        return {
            _SCHEMA_KEY: STAGING_SCHEMA,
            _TOOLS_KEY: {name: tool.to_document() for name, tool in self.tools.items()},
        }

    @classmethod
    def from_document(cls, document: object) -> StagingManifest:
        fields = _FIELDS.known(document, (_SCHEMA_KEY, _TOOLS_KEY), "the staging manifest")
        if fields.get(_SCHEMA_KEY) != STAGING_SCHEMA:
            raise StagingError(f"the staging manifest schema is {fields.get(_SCHEMA_KEY)!r}, not {STAGING_SCHEMA}")
        tools = _FIELDS.mapping(fields.get(_TOOLS_KEY), "the staged tools")
        return cls({name: StagedTool.from_document(tool) for name, tool in tools.items()})

    def to_bytes(self) -> bytes:
        """The manifest file's bytes: canonical, so one set of staged files has one sha256."""
        return (json.dumps(self.to_document(), sort_keys=True, indent=2) + "\n").encode()


@dataclass(frozen=True)
class StagingReport:
    """What :func:`verify` found: the manifest checked, each staged tool's commit, and everything missing."""

    manifest_sha256: str
    pins: Mapping[str, str]
    missing: tuple[str, ...]

    @property
    def failed_step(self) -> FailedStep | None:
        return FailedStep.STAGING_MISSING if self.missing else None

    @property
    def detail(self) -> str:
        return "; ".join(self.missing)


def tool_root(tool_cache: Path, tool: str, commit: str) -> Path:
    """Where ``tool`` staged at ``commit`` lives in ``tool_cache``."""
    return tool_cache / tool / commit


def inferencex_root(tool_cache: Path) -> Path:
    """The staged InferenceX at its pin: the directory holding the ``infx`` package."""
    return tool_root(tool_cache, INFERENCEX_TOOL, PINNED_TOOLS[INFERENCEX_TOOL])


def _unpinned(manifest: StagingManifest) -> Iterator[str]:
    for name, pin in PINNED_TOOLS.items():
        staged = manifest.tools.get(name)
        if staged is None or staged.commit != pin:
            found = "not staged" if staged is None else f"staged at {staged.commit}"
            yield f"{name} at its pin {pin} ({found})"


def _incomplete(manifest: StagingManifest) -> Iterator[str]:
    staged = manifest.tools.get(INFERENCEX_TOOL)
    if staged is not None:
        for required in INFERENCEX_REQUIRED:
            if required not in staged.files:
                yield f"{INFERENCEX_TOOL}'s {required} (not staged)"


def _damaged(tool_cache: Path, manifest: StagingManifest) -> Iterator[str]:
    for name, tool in manifest.tools.items():
        root = tool_root(tool_cache, name, tool.commit)
        for relative, digest in tool.files.items():
            path = root / relative
            if not path.is_file():
                yield f"{path} (absent)"
            elif _sha256(path.read_bytes()) != digest:
                yield f"{path} (changed since staging)"


def _absent_manifest(path: Path) -> str:
    return f"the staging manifest {path} (absent)"


def manifest_sha256(tool_cache: Path) -> str | None:
    """The sha256 of the manifest ``tool_cache`` holds, the ``tool_manifest_sha256`` a round names; ``None`` with none."""
    try:
        return _sha256((tool_cache / MANIFEST_FILE).read_bytes())
    except FileNotFoundError:
        return None


def _parse_manifest(path: Path, data: bytes) -> StagingManifest | str:
    """The manifest in ``data``, or what to report when it does not parse."""
    try:
        return StagingManifest.from_document(json.loads(data))
    except ValueError as exc:
        return f"a readable staging manifest at {path} ({exc})"


def verify(tool_cache: Path, manifest_sha256: str) -> StagingReport:
    """Check ``tool_cache`` against the manifest whose sha256 is ``manifest_sha256`` and the pins in force."""
    path = tool_cache / MANIFEST_FILE
    if not path.is_file():
        return StagingReport(manifest_sha256, {}, (_absent_manifest(path),))
    data = path.read_bytes()
    actual = _sha256(data)
    if actual != manifest_sha256:
        return StagingReport(manifest_sha256, {}, (f"the staging manifest {manifest_sha256} ({path} is {actual})",))
    manifest = _parse_manifest(path, data)
    if isinstance(manifest, str):
        return StagingReport(manifest_sha256, {}, (manifest,))
    pins = {name: tool.commit for name, tool in manifest.tools.items()}
    return StagingReport(
        manifest_sha256, pins, (*_unpinned(manifest), *_incomplete(manifest), *_damaged(tool_cache, manifest))
    )


def _checkout_commit(source: Path) -> str:
    done = subprocess.run([_GIT, "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise StagingError(f"{source} is not a git checkout: {done.stderr.strip()}")
    return done.stdout.strip()


def _subset_files(project: Path) -> Iterator[Path]:
    wanted = (*INFERENCEX_SUBSET, CLIENT_FILE, AGENTX_AGGREGATOR_FILE)
    absent = [entry for entry in wanted if not (project / entry).exists()]
    if absent:
        raise StagingError(f"{project} has no {', '.join(absent)}")
    for entry in INFERENCEX_SUBSET:
        path = project / entry
        if path.is_file():
            yield path
            continue
        for item in sorted(path.rglob("*")):
            if item.is_file() and not _SKIPPED_DIRS.intersection(item.relative_to(project).parts):
                yield item


def _run_staging_step(argv: Sequence[str], what: str) -> None:
    done = subprocess.run(list(argv), capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise StagingError(f"{what} exited with code {done.returncode}: {done.stderr.strip()[-2000:]}")


def _pip_install(python: Path | str, wheels: Path | None) -> list[str]:
    """``pip install`` under ``python``; with ``wheels``, from that directory's wheels alone, never from an index."""
    install = [str(python), "-m", "pip", "install", "--no-input", "--disable-pip-version-check"]
    return install if wheels is None else [*install, "--no-index", "--find-links", str(wheels), "--only-binary=:all:"]


def _install_agentx_runtime(root: Path, wheels: Path | None) -> dict[str, str]:
    """Build the harness venv under ``root`` from InferenceX's pinned requirements; returns its manifest files.

    The venv is built in place: its scripts carry their interpreter's absolute path, so it cannot be moved after.
    """
    if sys.version_info[:2] not in AGENTX_PYTHON_VERSIONS:
        supported = ", ".join(".".join(map(str, version)) for version in AGENTX_PYTHON_VERSIONS)
        raise StagingError(f"the AgentX harness needs Python {supported}; staging runs {sys.version.split()[0]}")
    venv = root / AGENTX_VENV_DIR
    _run_staging_step([sys.executable, "-m", "venv", str(venv)], "creating the AgentX harness venv")
    python = root / AGENTX_PYTHON
    install = _pip_install(python, wheels)
    _run_staging_step([*install, "-r", str(root / AGENTX_REQUIREMENTS)], "installing the AgentX harness")
    staged = [python, *sorted(venv.glob(f"lib/python*/site-packages/*.dist-info/{_DIST_RECORD}"))]
    return {path.relative_to(root).as_posix(): _sha256(path.read_bytes()) for path in staged}


def _write_atomically(path: Path, data: bytes) -> None:
    partial = path.with_name(f".{path.name}.{os.getpid()}")
    partial.write_bytes(data)
    os.replace(partial, path)


def stage(source: Path, tool_cache: Path, wheels: Path | None = None) -> str:
    """Copy InferenceX's subset from the checkout ``source`` into ``tool_cache`` and build the AgentX harness venv
    beside it, from the wheel directory ``wheels`` alone when one is given; returns the manifest's sha256.

    Raises:
        StagingError: ``source`` is not a checkout at the pinned commit, lacks part of the subset, or the harness
            cannot be installed under the interpreter staging runs.
    """
    pin = PINNED_TOOLS[INFERENCEX_TOOL]
    commit = _checkout_commit(source)
    if commit != pin:
        raise StagingError(f"{source} is checked out at {commit!r}; {INFERENCEX_TOOL} is pinned at {pin}")
    project = source / INFERENCEX_PROJECT_DIR
    root = tool_root(tool_cache, INFERENCEX_TOOL, pin)
    partial = root.with_name(f".{pin}.{os.getpid()}")
    shutil.rmtree(partial, ignore_errors=True)
    files: dict[str, str] = {}
    for path in _subset_files(project):
        relative = path.relative_to(project)
        target = partial / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        files[relative.as_posix()] = _sha256(target.read_bytes())
    shutil.rmtree(root, ignore_errors=True)
    os.replace(partial, root)
    files.update(_install_agentx_runtime(root, wheels))
    manifest = StagingManifest({INFERENCEX_TOOL: StagedTool(pin, files)})
    data = manifest.to_bytes()
    _write_atomically(tool_cache / MANIFEST_FILE, data)
    return _sha256(data)


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _install_lm_eval(wheels: Path | None) -> None:
    """Install the pinned lm-eval into this interpreter, the one a round runs its eval under, unless it is there.

    An offline staging (``wheels``) installs nothing into this interpreter: a sealed runner's lm-eval is the one its
    own lock installed, so one off the pin is refused.
    """
    installed = _distribution_version(LM_EVAL_DISTRIBUTION)
    if installed == LM_EVAL_VERSION:
        return
    if wheels is not None:
        raise StagingError(f"{sys.executable} has {LM_EVAL_DISTRIBUTION} {installed}, not {LM_EVAL_REQUIREMENT}")
    install = [sys.executable, "-m", "pip", "install", "--no-input", "--disable-pip-version-check"]
    _run_staging_step([*install, LM_EVAL_REQUIREMENT], f"installing {LM_EVAL_REQUIREMENT}")


def _intact_digest(tool_cache: Path) -> str | None:
    """The manifest's sha256 when ``tool_cache`` holds the pinned tools intact; ``None`` when it is to be staged.

    Raises:
        StagingError: a staged file was removed or edited, or the manifest does not parse.
    """
    digest = manifest_sha256(tool_cache)
    if digest is None or verify(tool_cache, digest).failed_step is None:
        return digest
    path = tool_cache / MANIFEST_FILE
    manifest = _parse_manifest(path, path.read_bytes())
    damage = (manifest,) if isinstance(manifest, str) else tuple(_damaged(tool_cache, manifest))
    if damage:
        raise StagingError("; ".join(damage))
    return None


def _fetch_inferencex(checkout: Path) -> None:
    """Check out InferenceX at its pin from :data:`INFERENCEX_REPO_URL` into the empty directory ``checkout``."""
    pin = PINNED_TOOLS[INFERENCEX_TOOL]
    git = [_GIT, "-C", str(checkout)]
    _run_staging_step([_GIT, "init", "-q", str(checkout)], f"creating {checkout}")
    _run_staging_step([*git, "fetch", "-q", "--depth", "1", INFERENCEX_REPO_URL, pin], f"fetching InferenceX {pin}")
    _run_staging_step([*git, "checkout", "-q", "FETCH_HEAD"], f"checking out InferenceX {pin}")


def ensure_staged(tool_cache: Path, source: Path | None, wheels: Path | None = None) -> str:
    """Stage the pinned tools into ``tool_cache`` unless it holds them intact; returns the manifest's sha256.

    A cache whose manifest verifies stages nothing. One staged at a pin no longer in force, or without a file a load
    now runs, is staged again. One whose staged files were removed or edited, or whose manifest does not parse, is
    refused as it stands, never overwritten. InferenceX is staged from the checkout ``source``, or with none,
    fetched at its pin. lm-eval is installed at its pin into the interpreter running this.

    With ``wheels`` the staging is offline: nothing is fetched, the harness is installed from that directory's wheels
    alone, ``source`` is required, and an lm-eval off its pin is refused rather than installed.

    Raises:
        StagingError: the cache is damaged, or a tool cannot be fetched, staged or installed, including when git or
            the cache directory is unavailable.
    """
    try:
        return _stage_unless_intact(tool_cache, source, wheels)
    except OSError as exc:
        raise StagingError(f"staging into {tool_cache} failed: {exc}") from exc


def _stage_unless_intact(tool_cache: Path, source: Path | None, wheels: Path | None) -> str:
    digest = _intact_digest(tool_cache)
    _install_lm_eval(wheels)
    if digest is not None:
        return digest
    if source is None and wheels is not None:
        raise StagingError("an offline staging (--wheels) fetches nothing; name the InferenceX checkout with --source")
    tool_cache.mkdir(parents=True, exist_ok=True)
    if source is not None:
        return stage(source, tool_cache, wheels)
    checkout = tool_cache / f".source-{PINNED_TOOLS[INFERENCEX_TOOL]}"
    shutil.rmtree(checkout, ignore_errors=True)
    try:
        _fetch_inferencex(checkout)
        return stage(checkout, tool_cache)
    finally:
        shutil.rmtree(checkout, ignore_errors=True)


def main(argv: Sequence[str] | None = None) -> int:
    """``stage`` prints the new manifest's sha256; ``check`` prints the cache's, or what is missing and exits 1."""
    parser = argparse.ArgumentParser(prog="python -m inference_suite.staging")
    commands = parser.add_subparsers(dest="command", required=True)
    stage_parser = commands.add_parser("stage", help="stage the pinned tools from local checkouts")
    stage_parser.add_argument("--source", type=Path, required=True, help="an InferenceX checkout at the pin")
    stage_parser.add_argument("--tool-cache", type=Path, required=True)
    check_parser = commands.add_parser("check", help="verify the tool cache against the pins in force")
    check_parser.add_argument("--tool-cache", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "stage":
        args.tool_cache.mkdir(parents=True, exist_ok=True)
        print(stage(args.source, args.tool_cache))
        return 0
    digest = manifest_sha256(args.tool_cache)
    if digest is None:
        print(
            f"{FailedStep.STAGING_MISSING.value}: {_absent_manifest(args.tool_cache / MANIFEST_FILE)}", file=sys.stderr
        )
        return 1
    report = verify(args.tool_cache, digest)
    if report.failed_step is not None:
        print(f"{report.failed_step.value}: {report.detail}", file=sys.stderr)
        return 1
    print(report.manifest_sha256)
    return 0


__all__ = [
    "AGENTX_AGGREGATOR_FILE",
    "AGENTX_AGGREGATOR_MODULE",
    "AGENTX_CLI_MODULE",
    "AGENTX_PYTHON",
    "AGENTX_PYTHON_VERSIONS",
    "AGENTX_REQUIREMENTS",
    "AGENTX_RESULTS_PACKAGE",
    "AGENTX_VENV_DIR",
    "CLIENT_FILE",
    "CLIENT_MODULE",
    "GSM8K_TASK",
    "INFERENCEX_PROJECT_DIR",
    "INFERENCEX_REQUIRED",
    "INFERENCEX_SUBSET",
    "MANIFEST_FILE",
    "STAGING_SCHEMA",
    "StagedTool",
    "StagingError",
    "StagingManifest",
    "StagingReport",
    "ensure_staged",
    "inferencex_root",
    "main",
    "manifest_sha256",
    "stage",
    "tool_root",
    "verify",
]


if __name__ == "__main__":
    sys.exit(main())
