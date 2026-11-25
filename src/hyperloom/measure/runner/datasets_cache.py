# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A link tree of the test's datasets in a cache of its own, when the cache ``datasets`` would use is not writable.

``datasets`` opens a lock file for writing on every load, a cached one too: in the cache directory
(``<cache>/_<cache with _>_openai___gsm8k_main_0.0.0_<hash>.lock``) and beside the prepared dataset. A cache on storage
mounted read-only (``EROFS``) or owned by another account (``EACCES``, ``EPERM``) therefore fails every load with a
``PermissionError``, though the prepared Arrow files are only read. The AgentX client loads its trace corpus that way
and the evaluation its task's dataset, so such a cache would fail every AgentX test and every evaluation.

The cache is the one ``datasets`` resolves (:func:`_effective_cache`): ``HF_DATASETS_CACHE``, else ``$HF_HOME/datasets``,
else ``$XDG_CACHE_HOME/huggingface/datasets``, else ``~/.cache/huggingface/datasets``. It is probed by making and removing
one file in it; a writable cache is left as it is, and so is the cache when the sheet's ``process_env`` names one of its
own (its ``HF_DATASETS_CACHE``, or with none in the runner's environment its ``HF_HOME`` or ``XDG_CACHE_HOME``).

When the cache is not writable, the runner makes below the test's scratch a tree of real directories mirroring each
dataset directory the test loads (``<namespace>___<name>``), with one symbolic link per file into the cache, and
``HF_DATASETS_CACHE`` names that tree while the test runs; the server, the client and the evaluation inherit it from the
runner's environment, and the variable is put back (removed when it was not set) when the test ends. ``datasets`` takes
its locks in the tree's directories and reads the files through the links; one link per dataset directory is not
enough, since an online load writes inside that directory. Locks are not linked. A dataset directory that is itself a
link is the directory it names; below it a link is followed only to a file inside that directory, and a linked
directory is not followed.

Which datasets a test loads is read from its sheet: the trace corpus of each AgentX point (the suite's own rule,
:func:`~hyperloom.orchestrator.actions.executors._suite.config.agentx_trace`), resolved to its Hub repository through the
client's loader registry in the staged client environment, and the dataset the staged GSM8K task names
(``dataset_path``) when the test runs accuracy. When no dataset is named or the cache holds none of them, the variable is
left as given, and the test meets the cache as it would have.

The tree is made before the test's timeout is fixed, so its time is the test's. A file system error while making it
ends the test as infrastructure: ``disk_full`` for a full disk or quota, else ``hf_cache_permission`` (a directory that
cannot be listed, a file that cannot be read, ``ESTALE``, ``EIO``).
"""

from __future__ import annotations

import contextlib
import errno
import logging
import os
import pwd
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from hyperloom.common.benchmark_config import is_agentx
from hyperloom.measure.failures import DISK_FULL, HF_CACHE_PERMISSION, is_disk_full
from hyperloom.measure.runner.service import RunnerFailure
from hyperloom.orchestrator.actions.executors._suite.config import agentx_trace
from inference_suite import CLIENT_SITE_PACKAGES, GSM8K_TASK

__all__ = ["DatasetsNeeded", "datasets_needed", "local_datasets_cache"]

log = logging.getLogger(__name__)

DATASETS_CACHE_ENV = "HF_DATASETS_CACHE"
#: The test's link tree, below its work directory.
LOCAL_DATASETS_DIR = "hf_datasets"
#: Where the client's loader registry names each public dataset's Hub repository, below its site-packages.
_LOADER_REGISTRY = "aiperf/plugin/plugins.yaml"
_PUBLIC_LOADERS = "public_dataset_loader"
_HUB_NAME = "hf_dataset_name"
_REFUSED = frozenset({errno.EROFS, errno.EACCES, errno.EPERM})
_LOCK_SUFFIX = ".lock"
_PROBE_PREFIX = ".hl-write-probe-"
_HF_HOME_ENV = "HF_HOME"
_XDG_CACHE_ENV = "XDG_CACHE_HOME"
_REPO_PART = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
# datasets.naming.camelcase_to_snakecase, which names a prepared dataset's directory.
_UPPER_UPPER = re.compile(r"([A-Z]+)([A-Z][a-z])")
_LOWER_UPPER = re.compile(r"([a-z\d])([A-Z])")


@dataclass(frozen=True)
class DatasetsNeeded:
    """The Hub repositories the test loads through ``datasets``, and the trace corpora no repository was found for."""

    repos: tuple[str, ...] = ()
    unresolved: tuple[str, ...] = ()


@dataclass
class _LinkReport:
    """What a link tree holds: its links, the repositories linked, those the cache holds no directory of, the links in
    the cache left out, and the seconds it took."""

    files: int = 0
    linked: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    skipped_links: int = 0
    seconds: float = 0.0

    def summary(self, source: Path, target: Path) -> str:
        text = (
            f"{DATASETS_CACHE_ENV} {source} is not writable (datasets takes a lock on every load); linked "
            f"{', '.join(self.linked)} ({self.files} files) below {target} in {self.seconds:.1f} s"
        )
        if self.missing:
            text += f"; the cache holds no {', '.join(self.missing)}"
        if self.skipped_links:
            text += f"; {self.skipped_links} links in the cache leaving their dataset were left out"
        return text


def _is_writable(directory: Path) -> bool:
    """Whether a file can be made and removed in ``directory``: the open ``datasets`` makes of its locks.

    ``EROFS``, ``EACCES`` and ``EPERM`` say no alike; any other error is the cache's own and says nothing of this, so
    the cache counts as writable and is left as it was.
    """
    try:
        handle, name = tempfile.mkstemp(prefix=_PROBE_PREFIX, dir=directory)
    except OSError as exc:
        if exc.errno in _REFUSED:
            return False
        log.warning("%s %s could not be probed for writing: %s", DATASETS_CACHE_ENV, directory, exc)
        return True
    os.close(handle)
    with contextlib.suppress(OSError):
        os.unlink(name)
    return True


def _cache_dir_name(repo: str) -> str | None:
    """The directory below the cache ``datasets`` prepares ``repo`` in (``openai/gsm8k`` is ``openai___gsm8k``,
    ``datasets.naming.camelcase_to_snakecase`` on the name), or ``None`` when ``repo`` is no Hub repository id."""
    parts = repo.strip().strip("/").split("/")
    if not all(_REPO_PART.fullmatch(part) and part not in {".", ".."} for part in parts):
        return None
    last = _LOWER_UPPER.sub(r"\1_\2", _UPPER_UPPER.sub(r"\1_\2", parts[-1]))
    return "___".join([*parts[:-1], last.lower()])


def _loader_registry(staged_root: Path) -> Mapping[str, str]:
    """Each public dataset the staged AgentX client knows, by the Hub repository it loads; empty when unreadable."""
    path = staged_root / CLIENT_SITE_PACKAGES / _LOADER_REGISTRY
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        log.warning("the AgentX client's loader registry %s is unreadable: %s", path, exc)
        return {}
    loaders = document.get(_PUBLIC_LOADERS) if isinstance(document, Mapping) else None
    if not isinstance(loaders, Mapping):
        return {}
    registry = {}
    for name, entry in loaders.items():
        metadata = entry.get("metadata") if isinstance(entry, Mapping) else None
        repo = metadata.get(_HUB_NAME) if isinstance(metadata, Mapping) else None
        if isinstance(repo, str) and repo.strip():
            registry[str(name)] = repo.strip()
    return registry


def _eval_dataset(staged_root: Path) -> str | None:
    """The Hub repository the staged GSM8K task loads (its ``dataset_path``); ``None`` when it names none."""
    path = staged_root / GSM8K_TASK
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        log.warning("the staged GSM8K task %s is unreadable: %s", path, exc)
        return None
    named = document.get("dataset_path") if isinstance(document, Mapping) else None
    return (named.strip() or None) if isinstance(named, str) else None


def _benchmark(point_yaml: str) -> Mapping[str, object]:
    try:
        document = yaml.safe_load(point_yaml)
    except yaml.YAMLError:
        return {}
    bench = document.get("benchmark") if isinstance(document, Mapping) else None
    return bench if isinstance(bench, Mapping) else {}


def _point_envs(bench: Mapping[str, object]) -> dict[str, str]:
    envs = bench.get("envs")
    if not isinstance(envs, Mapping):
        return {}
    return {str(name): str(value) for name, value in envs.items() if value is not None}


def datasets_needed(
    points: Iterable[str], *, base_client_env: Mapping[str, str], accuracy: bool, staged_root: Path | None
) -> DatasetsNeeded:
    """The datasets a test of the point YAMLs ``points`` loads: each AgentX point's trace corpus as the staged client's
    loader registry names its repository, and the staged GSM8K task's dataset when the test runs accuracy. A corpus the
    registry does not name, or any corpus with no staging, is unresolved."""
    registry: Mapping[str, str] | None = None
    repos: list[str] = []
    unresolved: list[str] = []
    for point in points:
        bench = _benchmark(point)
        if not is_agentx(bench):
            continue
        trace = (agentx_trace(_point_envs(bench), base_client_env) or "").strip()
        if not trace:
            continue
        if registry is None:
            registry = _loader_registry(staged_root) if staged_root is not None else {}
        repo = registry.get(trace)
        (repos if repo else unresolved).append(repo or trace)
    if accuracy and staged_root is not None:
        evaluated = _eval_dataset(staged_root)
        if evaluated:
            repos.append(evaluated)
    return DatasetsNeeded(tuple(dict.fromkeys(repos)), tuple(dict.fromkeys(unresolved)))


def _link_tree(source: Path, target: Path, inside: Path, report: _LinkReport) -> None:
    """Mirror the directory ``source`` as real directories below ``target``, one link per file into ``source``; locks
    left out, a link followed only to a file below ``inside``. Every error raises: a directory that cannot be listed is
    not reported linked."""
    target.mkdir()
    with os.scandir(source) as entries:
        for entry in entries:
            if entry.name.endswith(_LOCK_SUFFIX):
                continue
            path = Path(entry.path)
            if entry.is_symlink():
                named = path.resolve()
                if not named.is_relative_to(inside) or not named.is_file():
                    report.skipped_links += 1
                    continue
            elif entry.is_dir(follow_symlinks=False):
                _link_tree(path, target / entry.name, inside, report)
                continue
            elif entry.is_file(follow_symlinks=False):
                named = path
            else:
                continue
            if not os.access(named, os.R_OK):
                raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), str(named))
            os.symlink(named, target / entry.name)
            report.files += 1


def _link_datasets(source: Path, target: Path, repos: Iterable[str]) -> _LinkReport:
    """The link tree below ``target`` of each of ``repos`` prepared below the cache ``source``."""
    started = time.monotonic()
    report = _LinkReport()
    target.mkdir(parents=True)
    for repo in repos:
        name = _cache_dir_name(repo)
        if name is None or not (source / name).is_dir():
            report.missing.append(repo)
            continue
        inside = (source / name).resolve()
        _link_tree(inside, target / name, inside, report)
        report.linked.append(repo)
    report.seconds = time.monotonic() - started
    return report


def _expand_user(path: str, environ: Mapping[str, str]) -> str:
    """``os.path.expanduser`` of ``path`` with ``environ``'s ``HOME`` for ``~``, as ``datasets`` expands ``HF_HOME``."""
    if path != "~" and not path.startswith("~/"):
        return os.path.expanduser(path)  # no ``~``, or ``~user``, which reads no variable
    home = environ["HOME"] if "HOME" in environ else pwd.getpwuid(os.getuid()).pw_dir
    return (home.rstrip("/") + path[1:]) or "/"


def _effective_cache(environ: Mapping[str, str]) -> Path | None:
    """The cache ``datasets`` uses under ``environ`` (``datasets.config``): ``HF_DATASETS_CACHE`` as given, else
    ``HF_HOME`` with ``~`` expanded, else ``$XDG_CACHE_HOME/huggingface`` (``XDG_CACHE_HOME`` defaulting to ``~/.cache``),
    followed by ``datasets``. A variable set empty counts as set, as there. ``None`` when that path is relative: the
    test's processes resolve it against their own working directories."""
    if DATASETS_CACHE_ENV in environ:
        cache = environ[DATASETS_CACHE_ENV]
    else:
        home = environ.get(_HF_HOME_ENV, os.path.join(environ.get(_XDG_CACHE_ENV, "~/.cache"), "huggingface"))
        cache = os.path.join(_expand_user(home, environ), "datasets")
    return Path(cache) if os.path.isabs(cache) else None


@contextlib.contextmanager
def local_datasets_cache(
    environ: MutableMapping[str, str],
    work: Path,
    needed: Callable[[], DatasetsNeeded],
    notes: list[str],
    *,
    sheet_env: Mapping[str, str] | None = None,
) -> Iterator[Path | None]:
    """While the block runs, point ``environ``'s ``HF_DATASETS_CACHE`` at a link tree below ``work`` of the datasets
    ``needed()`` names when the cache ``datasets`` would use (:func:`_effective_cache`) is not writable; yield the tree,
    or ``None`` when the cache is left as it is.

    Nothing changes when that cache is no directory, the sheet names a cache of its own (``sheet_env``: its
    ``HF_DATASETS_CACHE``, or with none in ``environ`` its ``HF_HOME`` or ``XDG_CACHE_HOME``), the cache is writable, or
    the cache holds none of the datasets named. ``environ`` is put back (the variable removed when it was not set) and
    the tree removed when the block ends. What was linked and its time are logged and added to ``notes``.

    Raises:
        RunnerFailure: the tree could not be made: ``disk_full`` for a full disk or quota, else ``hf_cache_permission``.
    """
    given = environ.get(DATASETS_CACHE_ENV)
    own = sheet_env or {}
    overridden = (DATASETS_CACHE_ENV,) if given is not None else (DATASETS_CACHE_ENV, _HF_HOME_ENV, _XDG_CACHE_ENV)
    source = _effective_cache(environ)
    if source is None or any(name in own for name in overridden) or not source.is_dir() or _is_writable(source):
        yield None
        return
    wanted = needed()
    if wanted.unresolved:
        unnamed = (
            f"{DATASETS_CACHE_ENV} {source} is not writable and {', '.join(wanted.unresolved)} names no dataset "
            "repository the staged client knows, so it is not linked"
        )
        log.warning("%s", unnamed)
        notes.append(unnamed)
    if not wanted.repos:
        yield None
        return
    target = work / LOCAL_DATASETS_DIR
    shutil.rmtree(target, ignore_errors=True)
    try:
        report = _link_datasets(source, target, wanted.repos)
    except OSError as exc:
        shutil.rmtree(target, ignore_errors=True)
        error_class = DISK_FULL if is_disk_full(exc) else HF_CACHE_PERMISSION
        raise RunnerFailure(
            error_class, f"{DATASETS_CACHE_ENV} {source} is not writable and its datasets could not be linked: {exc}"
        ) from exc
    except BaseException:
        shutil.rmtree(target, ignore_errors=True)
        raise
    if not report.linked:
        shutil.rmtree(target, ignore_errors=True)
        absent = f"{DATASETS_CACHE_ENV} {source} is not writable and holds none of {', '.join(report.missing)}"
        log.warning("%s", absent)
        notes.append(absent)
        yield None
        return
    message = report.summary(source, target)
    log.info("%s", message)
    notes.append(message)
    environ[DATASETS_CACHE_ENV] = str(target)
    try:
        yield target
    finally:
        if given is None:
            environ.pop(DATASETS_CACHE_ENV, None)
        else:
            environ[DATASETS_CACHE_ENV] = given
        shutil.rmtree(target, ignore_errors=True)
