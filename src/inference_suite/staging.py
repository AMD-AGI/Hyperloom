# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Offline staging (D11): the pinned tools copied into the tool cache ahead of time, and the check a round runs on them.

The tool cache holds each tool under ``<tool>/<commit>/`` and one manifest, ``staging_manifest.json``, naming every
staged file with its sha256. A round fetches nothing: :func:`verify` lists what is absent, changed or staged at a
commit other than its pin, including a manifest that does not stage a file a load runs (:data:`INFERENCEX_REQUIRED`),
and a round with anything listed refuses with ``staging_missing``. The manifest's own sha256 is the round's
``tool_manifest_sha256``, so a launch names exactly the bytes it ran.

The AgentX harness runs in its own environment beside the InferenceX subset (:mod:`inference_suite.client_env`): the
pinned CPython build with exactly the wheels ``agentx-client.lock`` pins. Its manifest entries are the interpreter, the
build record, the lock, each installed distribution's ``RECORD``, which lists every installed file with its sha256, and
the two aiperf files staging patches so the harness exports ``error_osl``.
The file manifest names bytes, some of which hold the directory the staging was built in; what two hosts compare is
:func:`normalised_manifest`, the staging's identity with every path left out, and the lm-eval this interpreter runs.

``python -m inference_suite.staging stage --source <InferenceX repository> --tool-cache <dir>`` stages; without
``--source`` it stages nothing and only completes a cache that already verifies. Either installs the pinned lm-eval
into the interpreter running it, from the sealed runner's hash lock (:func:`runner_lock`). ``check --tool-cache <dir>`` reports whether a round could identify the staging: the
cache holds the current pins intact and the interpreter running it has the pinned lm-eval. Hyperloom's ``install.sh``
runs all three.
:func:`ensure_staged`, behind ``python -m inference_suite stage`` and this package's ``install.sh``, stages only what
a cache lacks.

:func:`publish` stages into a directory several hosts share, with no lock: it stages into a private
``.staging-<key>-<uuid>`` beside the published one, syncs it and renames it to ``<key>`` (:func:`staging_key`, which
every input of a staging changes). A loser of the rename removes only its own directory and uses the winner once the
winner's normalised manifest is the one it staged. A published directory is never written again, and a reader looks
up ``<key>`` alone, so it never sees a ``.staging-*`` directory.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import importlib.metadata
import io
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from hyperloom_common.contract_fields import ContractFields, field_names
from hyperloom_common.shared_storage import make_shared_dirs
from inference_suite import aiperf_patch, client_env
from inference_suite.client_env import CLIENT_PYTHON, PATCHED_FILES, ClientEnvError
from inference_suite.contracts.normalised import NormalisedManifest, PackageIdentity, canonical_name
from inference_suite.contracts.report import FailedStep
from inference_suite.pins import (
    AGENTX_PYTHON_BUILD,
    INFERENCEX_REPO_URL,
    INFERENCEX_TOOL,
    LM_EVAL_DISTRIBUTION,
    LM_EVAL_REQUIREMENT,
    LM_EVAL_VERSION,
    LM_EVAL_WHEEL_SHA256,
    PINNED_TOOLS,
)

STAGING_SCHEMA = "inference-suite-staging/1"
MANIFEST_FILE = "staging_manifest.json"
#: The project directory of an InferenceX tree that holds the ``infx`` package.
INFERENCEX_PROJECT_DIR = "inferencex-e2e"
#: The GSM8K task file the eval scores, under the staged InferenceX root.
GSM8K_TASK = "infx/evals/gsm8k.yaml"
#: The module the fixed load runs, from the staged InferenceX root.
CLIENT_MODULE = "infx.bench_serving.benchmark_serving"
#: The file :data:`CLIENT_MODULE` loads, relative to the staged InferenceX root.
CLIENT_FILE = f"{CLIENT_MODULE.replace('.', '/')}.py"
#: A file that runs :data:`CLIENT_MODULE` from the staged root named in :data:`CLIENT_ROOT_ENV`, for a harness that runs
#: its client as a file.
CLIENT_SCRIPT = Path(__file__).parent / "load" / "client_script.py"
CLIENT_ROOT_ENV = "INFERENCE_SUITE_CLIENT_ROOT"
#: The AgentX harness requirements InferenceX pins, under the staged InferenceX root.
AGENTX_REQUIREMENTS = "infx/bench/agentic/requirements.txt"
#: The module that aggregates an AgentX replay's export, from the staged InferenceX root.
AGENTX_AGGREGATOR_MODULE = "infx.results.agentic.process_agentic_result"
AGENTX_AGGREGATOR_FILE = f"{AGENTX_AGGREGATOR_MODULE.replace('.', '/')}.py"
#: The package the aggregator imports its builders from, staged whole.
AGENTX_RESULTS_PACKAGE = "infx/results"
#: The interpreter of the client environment, under the staged InferenceX root, and the CLI module the AgentX load runs.
AGENTX_PYTHON = CLIENT_PYTHON
AGENTX_CLI_MODULE = "agentx_cli"
#: What a round and its eval run from InferenceX, relative to its project directory.
INFERENCEX_SUBSET = (
    "infx/__init__.py",
    CLIENT_MODULE.rpartition(".")[0].replace(".", "/"),
    GSM8K_TASK,
    AGENTX_RESULTS_PACKAGE,
    AGENTX_REQUIREMENTS,
)
#: The files a staged InferenceX must hold for a round to run, whatever else its subset holds at the pin. The patched
#: client files are among them, so a staging made before the patch is staged again.
INFERENCEX_REQUIRED = (CLIENT_FILE, GSM8K_TASK, AGENTX_AGGREGATOR_FILE, AGENTX_PYTHON, *PATCHED_FILES)
#: The prefix of a staging :func:`publish` has not renamed into place yet; no reader looks below one.
PUBLISHING_PREFIX = ".staging-"

#: The Hyperloom source tree this package runs from. Its runner lock pins the lm-eval a round's eval runs, and that
#: release's dependency tree, with hashes; the package ships no lock of its own for it, so it stages lm-eval only from
#: a source tree, as ``install.sh`` runs it.
_SOURCE_TREE = Path(__file__).resolve().parents[2]
_RUNNER_LOCK = "requirements/runner.lock"
_VERSION_LOCK = "requirements/runner-cp{digits}.lock"
_COMMIT = re.compile(r"[0-9a-f]{40}")
_GIT = "git"
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


def verified_manifest_sha256(tool_cache: Path) -> str | None:
    """The sha256 of the manifest ``tool_cache`` holds when the cache verifies against the pins in force; ``None``
    when it does not, or holds no manifest."""
    digest = manifest_sha256(tool_cache)
    if digest is None or verify(tool_cache, digest).failed_step is not None:
        return None
    return digest


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


def _git_output(source: Path, *args: str) -> bytes:
    done = subprocess.run([_GIT, "-C", str(source), *args], capture_output=True, check=False)
    if done.returncode != 0:
        raise StagingError(f"git {args[0]} in {source} failed: {done.stderr.decode(errors='replace').strip()}")
    return done.stdout


def _extract_pinned_subset(source: Path, pin: str, partial: Path) -> None:
    """Write InferenceX's subset as the commit ``pin`` in the repository ``source`` holds it into ``partial``; a
    checkout's working tree never reaches the staging."""
    found = subprocess.run(
        [_GIT, "-C", str(source), "rev-parse", "--verify", "--quiet", f"{pin}^{{commit}}"],
        capture_output=True,
        check=False,
    )
    if found.returncode != 0:
        raise StagingError(f"{source} has no commit {pin}; {INFERENCEX_TOOL} is pinned at {pin}")
    tree = f"{pin}:{INFERENCEX_PROJECT_DIR}"
    wanted = (*INFERENCEX_SUBSET, CLIENT_FILE, AGENTX_AGGREGATOR_FILE)
    listed = _git_output(source, "ls-tree", "-r", "--name-only", "-z", tree, "--", *INFERENCEX_SUBSET)
    paths = [path for path in listed.decode().split("\0") if path]
    absent = [entry for entry in wanted if not any(path == entry or path.startswith(f"{entry}/") for path in paths)]
    if absent:
        raise StagingError(f"{source / INFERENCEX_PROJECT_DIR} has no {', '.join(absent)} at {pin}")
    archive = _git_output(source, "archive", "--format=tar", tree, "--", *INFERENCEX_SUBSET)
    with tarfile.open(fileobj=io.BytesIO(archive)) as members:
        members.extractall(partial, filter="data")


def _run_staging_step(argv: Sequence[str], what: str) -> None:
    done = subprocess.run(list(argv), capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise StagingError(f"{what} exited with code {done.returncode}: {done.stderr.strip()[-2000:]}")


def _write_atomically(path: Path, data: bytes) -> None:
    partial = path.with_name(f".{path.name}.{os.getpid()}")
    partial.write_bytes(data)
    os.replace(partial, path)


def stage(source: Path, tool_cache: Path) -> str:
    """Copy InferenceX's subset as committed at its pin in the repository ``source`` into ``tool_cache`` and build the
    AgentX client environment beside it from the pinned build and lock, both downloaded; returns the manifest's sha256.

    Raises:
        StagingError: ``source`` is not a repository holding the pinned commit, the commit lacks part of the subset, or
            the client environment cannot be fetched or installed.
    """
    pin = PINNED_TOOLS[INFERENCEX_TOOL]
    root = tool_root(tool_cache, INFERENCEX_TOOL, pin)
    partial = root.with_name(f".{pin}.{os.getpid()}")
    shutil.rmtree(partial, ignore_errors=True)
    _extract_pinned_subset(source, pin, partial)
    files = {
        path.relative_to(partial).as_posix(): _sha256(path.read_bytes())
        for path in sorted(partial.rglob("*"))
        if path.is_file()
    }
    try:
        with tempfile.TemporaryDirectory(dir=tool_cache, prefix=".inputs-") as inputs:
            client_env.fetch_inputs(Path(inputs))
            client_env.install(Path(inputs), partial)
    except ClientEnvError as exc:
        raise StagingError(str(exc)) from exc
    files.update(
        {path.relative_to(partial).as_posix(): _sha256(path.read_bytes()) for path in client_env.staged_files(partial)}
    )
    shutil.rmtree(root, ignore_errors=True)
    os.replace(partial, root)
    manifest = StagingManifest({INFERENCEX_TOOL: StagedTool(pin, files)})
    data = manifest.to_bytes()
    _write_atomically(tool_cache / MANIFEST_FILE, data)
    return _sha256(data)


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _lm_eval_off_pin() -> str | None:
    """What this interpreter, the one a round runs its eval under, has instead of the pinned lm-eval; ``None`` when it
    has the pinned release."""
    installed = _distribution_version(LM_EVAL_DISTRIBUTION)
    if installed == LM_EVAL_VERSION:
        return None
    return f"{sys.executable} has {LM_EVAL_DISTRIBUTION} {installed}, not {LM_EVAL_REQUIREMENT}"


def _runner_lock(python_version: str) -> Path:
    # The rule of hyperloom.measure.runner.cache_layout.lock_file, which this package may not import; a test holds the
    # two to the same file for every Python.
    own = _SOURCE_TREE / _VERSION_LOCK.format(digits=python_version.replace(".", ""))
    return own if own.is_file() else _SOURCE_TREE / _RUNNER_LOCK


def runner_lock() -> Path:
    """The hash lock the sealed runner installs on this interpreter's Python, in the source tree staging runs from."""
    return _runner_lock(f"{sys.version_info[0]}.{sys.version_info[1]}")


def _locked_lm_eval(lock: Path) -> str:
    """The pinned lm-eval requirement with every hash ``lock`` allows for it, as a hash-checking pip line.

    Raises:
        StagingError: ``lock`` is absent, or pins no lm-eval at :data:`LM_EVAL_VERSION`.
    """
    if not lock.is_file():
        raise StagingError(f"{lock} is absent: lm-eval is installed only from a Hyperloom source tree's runner lock")
    pin = f"{canonical_name(LM_EVAL_DISTRIBUTION)}=={LM_EVAL_VERSION}"
    lines = [line.strip().removesuffix("\\").strip() for line in lock.read_text().splitlines()]
    if pin not in lines:
        raise StagingError(f"{lock} pins no {pin}")
    hashes = itertools.takewhile(lambda line: line.startswith("--hash="), lines[lines.index(pin) + 1 :])
    return " ".join([LM_EVAL_REQUIREMENT, *hashes]) + "\n"


def _install_lm_eval() -> None:
    """Install the pinned lm-eval into this interpreter unless it is there: its own hashes and every dependency's come
    from :func:`runner_lock`, so pip installs nothing the lock does not name."""
    if _lm_eval_off_pin() is None:
        return
    lock = runner_lock()
    requirement = _locked_lm_eval(lock)
    with tempfile.TemporaryDirectory(prefix="inference-suite-lm-eval-") as scratch:
        requirement_file = Path(scratch, "lm-eval.txt")
        requirement_file.write_text(requirement)
        install = [sys.executable, "-m", "pip", "install", "--no-input", "--disable-pip-version-check"]
        install += ["--require-hashes", "--constraint", str(lock), "--requirement", str(requirement_file)]
        _run_staging_step(install, f"installing {LM_EVAL_REQUIREMENT}")


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


def _fetch_inferencex(repository: Path) -> None:
    """Fetch InferenceX's pinned commit from :data:`INFERENCEX_REPO_URL` into a new repository at ``repository``."""
    pin = PINNED_TOOLS[INFERENCEX_TOOL]
    _run_staging_step([_GIT, "init", "-q", str(repository)], f"creating {repository}")
    fetch = [_GIT, "-C", str(repository), "fetch", "-q", "--depth", "1", INFERENCEX_REPO_URL, pin]
    _run_staging_step(fetch, f"fetching InferenceX {pin}")


def ensure_staged(tool_cache: Path, source: Path | None) -> str:
    """Stage the pinned tools into ``tool_cache`` unless it holds them intact; returns the manifest's sha256.

    A cache whose manifest verifies stages nothing. One staged at a pin no longer in force, or without a file a load
    now runs, is staged again. One whose staged files were removed or edited, or whose manifest does not parse, is
    refused as it stands, never overwritten. InferenceX is staged from the repository ``source``, or with none,
    fetched at its pin. lm-eval is installed at its pin into the interpreter running this.

    Raises:
        StagingError: the cache is damaged, or a tool cannot be fetched, staged or installed, including when git or
            the cache directory is unavailable.
    """
    try:
        return _stage_unless_intact(tool_cache, source)
    except OSError as exc:
        raise StagingError(f"staging into {tool_cache} failed: {exc}") from exc


def _stage_unless_intact(tool_cache: Path, source: Path | None) -> str:
    digest = _intact_digest(tool_cache)
    _install_lm_eval()
    if digest is not None:
        return digest
    tool_cache.mkdir(parents=True, exist_ok=True)
    return _stage_from(source, tool_cache)


def _stage_from(source: Path | None, tool_cache: Path) -> str:
    """:func:`stage` from the repository ``source``, or from InferenceX fetched at its pin into ``tool_cache``."""
    if source is not None:
        return stage(source, tool_cache)
    repository = tool_cache / f".source-{PINNED_TOOLS[INFERENCEX_TOOL]}"
    shutil.rmtree(repository, ignore_errors=True)
    try:
        _fetch_inferencex(repository)
        return stage(repository, tool_cache)
    finally:
        shutil.rmtree(repository, ignore_errors=True)


def _staged_identity(tool_cache: Path) -> NormalisedManifest:
    try:
        return client_env.client_identity(inferencex_root(tool_cache), PINNED_TOOLS[INFERENCEX_TOOL])
    except ClientEnvError as exc:
        raise StagingError(str(exc)) from exc


def normalised_manifest(tool_cache: Path) -> NormalisedManifest:
    """The normalised identity of the staging in ``tool_cache`` and of the lm-eval this interpreter runs the eval with.

    Raises:
        StagingError: the staged client environment is not what its lock pins, or this interpreter's lm-eval is not
            the pinned release.
    """
    off_pin = _lm_eval_off_pin()
    if off_pin is not None:
        raise StagingError(off_pin)
    lm_eval = PackageIdentity(LM_EVAL_DISTRIBUTION, LM_EVAL_VERSION, LM_EVAL_WHEEL_SHA256)
    return _staged_identity(tool_cache).with_packages((lm_eval,))


def staging_key() -> str:
    """The directory name :func:`publish` publishes the staging of the inputs in force under: the sha256 of the
    staging schema, the InferenceX pin, the interpreter build, the client lock and the client's patch, so any changed
    input gets a new one."""
    inputs = {
        _SCHEMA_KEY: STAGING_SCHEMA,
        INFERENCEX_TOOL: PINNED_TOOLS[INFERENCEX_TOOL],
        "python_build": AGENTX_PYTHON_BUILD.sha256,
        "client_lock": _sha256(client_env.AGENTX_LOCK.read_bytes()),
        "client_patch": aiperf_patch.PATCH_SHA256,
    }
    return _sha256(json.dumps(inputs, sort_keys=True).encode())


def _fsync_tree(root: Path) -> None:
    """Flush every file and directory below ``root``, and ``root`` itself, to storage."""
    for directory, _, names in os.walk(root):
        for name in names:
            path = Path(directory, name)
            if not path.is_symlink():
                _fsync(path, os.O_RDONLY)
        _fsync(Path(directory), os.O_RDONLY | os.O_DIRECTORY)


def _fsync(path: Path, flags: int) -> None:
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _published_identity(published: Path) -> NormalisedManifest:
    """The identity of the published staging ``published`` once its files verify against its own manifest."""
    digest = manifest_sha256(published)
    report = verify(published, digest) if digest is not None else None
    if report is None or report.failed_step is not None:
        found = _absent_manifest(published / MANIFEST_FILE) if report is None else report.detail
        raise StagingError(f"the published staging {published} does not verify, and is never rewritten: {found}")
    return _staged_identity(published)


def publish(stagings: Path, source: Path | None) -> Path:
    """The staging of the inputs in force, published as ``stagings/<staging_key()>``; staged here when absent.

    ``stagings`` is made as :func:`~hyperloom_common.shared_storage.make_shared_dirs` makes it: ``2775`` below storage
    set up to be shared (a runner cache the deployment made setgid in the group of its accounts), as the umask leaves
    it anywhere else. The staging is built in
    ``stagings/.staging-<key>-<uuid>``, synced and renamed into place; when another host renamed its own first, this
    one's directory is removed and the winner used once its normalised manifest is the one staged here.

    Raises:
        StagingError: the staging cannot be built, the published one does not verify, or it is not the software this
            host staged (each package that differs named).
    """
    try:
        return _publish(stagings, source)
    except OSError as exc:
        raise StagingError(f"publishing a staging into {stagings} failed: {exc}") from exc


def _publish(stagings: Path, source: Path | None) -> Path:
    key = staging_key()
    published = make_shared_dirs(stagings) / key
    if published.is_dir():
        _published_identity(published)
        return published
    private = make_shared_dirs(stagings / f"{PUBLISHING_PREFIX}{key}-{uuid.uuid4().hex}")
    try:
        _stage_from(source, private)
        staged = _staged_identity(private)
        _fsync_tree(private)
    except BaseException:
        shutil.rmtree(private, ignore_errors=True)
        raise
    try:
        os.rename(private, published)
    except OSError as exc:
        shutil.rmtree(private, ignore_errors=True)
        if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
            raise
        differences = _published_identity(published).differences(staged)
        if differences:
            raise StagingError(
                f"the published staging {published} is not the one staged here: {'; '.join(differences)}"
            ) from None
        return published
    _fsync(stagings, os.O_RDONLY | os.O_DIRECTORY)
    return published


def _cache_report(tool_cache: Path) -> StagingReport:
    """:func:`verify` against whatever manifest ``tool_cache`` holds."""
    digest = manifest_sha256(tool_cache)
    if digest is None:
        return StagingReport("", {}, (_absent_manifest(tool_cache / MANIFEST_FILE),))
    return verify(tool_cache, digest)


def _stage_command(tool_cache: Path, source: Path | None) -> str:
    """:func:`stage` from ``source``, or with none, the digest of ``tool_cache`` as it verifies; either way with the
    pinned lm-eval installed into this interpreter.

    Raises:
        StagingError: with no ``source``, the cache does not verify; or a tool cannot be staged or installed.
    """
    if source is None:
        report = _cache_report(tool_cache)
        if report.failed_step is not None:
            raise StagingError(report.detail)
        _install_lm_eval()
        return report.manifest_sha256
    _install_lm_eval()
    tool_cache.mkdir(parents=True, exist_ok=True)
    return stage(source, tool_cache)


def main(argv: Sequence[str] | None = None) -> int:
    """``stage`` prints the manifest's sha256; ``check`` prints the cache's when a round could identify it. Either
    prints what is missing as ``staging_missing`` and exits 1 otherwise."""
    parser = argparse.ArgumentParser(prog="python -m inference_suite.staging")
    commands = parser.add_subparsers(dest="command", required=True)
    stage_parser = commands.add_parser(
        "stage", help="stage the pinned tools and install the pinned lm-eval into this interpreter"
    )
    stage_parser.add_argument(
        "--source",
        type=Path,
        help="an InferenceX repository holding the pinned commit; when omitted, the cache must already verify",
    )
    stage_parser.add_argument("--tool-cache", type=Path, required=True)
    check_parser = commands.add_parser(
        "check", help="verify the tool cache against the pins in force and this interpreter's lm-eval against its pin"
    )
    check_parser.add_argument("--tool-cache", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "stage":
        try:
            print(_stage_command(args.tool_cache, args.source))
        except StagingError as exc:
            print(f"{FailedStep.STAGING_MISSING.value}: {exc}", file=sys.stderr)
            return 1
        return 0
    report = _cache_report(args.tool_cache)
    off_pin = _lm_eval_off_pin()
    missing = report.missing if off_pin is None else (*report.missing, off_pin)
    if missing:
        print(f"{FailedStep.STAGING_MISSING.value}: {'; '.join(missing)}", file=sys.stderr)
        return 1
    print(report.manifest_sha256)
    return 0


__all__ = [
    "AGENTX_AGGREGATOR_FILE",
    "AGENTX_AGGREGATOR_MODULE",
    "AGENTX_CLI_MODULE",
    "AGENTX_PYTHON",
    "AGENTX_REQUIREMENTS",
    "AGENTX_RESULTS_PACKAGE",
    "CLIENT_FILE",
    "CLIENT_MODULE",
    "GSM8K_TASK",
    "INFERENCEX_PROJECT_DIR",
    "INFERENCEX_REQUIRED",
    "INFERENCEX_SUBSET",
    "MANIFEST_FILE",
    "PUBLISHING_PREFIX",
    "STAGING_SCHEMA",
    "StagedTool",
    "StagingError",
    "StagingManifest",
    "StagingReport",
    "ensure_staged",
    "inferencex_root",
    "main",
    "manifest_sha256",
    "normalised_manifest",
    "publish",
    "runner_lock",
    "stage",
    "staging_key",
    "tool_root",
    "verified_manifest_sha256",
    "verify",
]


if __name__ == "__main__":
    sys.exit(main())
