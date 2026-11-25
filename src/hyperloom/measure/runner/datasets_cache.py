# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A copy of the test's datasets in a cache of its own, when the ``HF_DATASETS_CACHE`` the runner is given is not
writable.

``datasets`` opens a lock file for writing on every load, a cached one too: below the cache directory
(``<cache>/_<cache with _>_openai___gsm8k_main_0.0.0_<hash>.lock``) and beside the prepared dataset. A cache on storage
mounted read-only (``EROFS``) or owned by another account (``EACCES``, ``EPERM``) therefore fails every load with a
``PermissionError``, though the prepared Arrow files are only read. The AgentX client loads its trace corpus that way
and the evaluation its task's dataset, so such a cache would fail every AgentX test and every evaluation.

When the cache is not writable, the datasets the test loads are copied into a directory below the test's own scratch
and ``HF_DATASETS_CACHE`` names that copy while the test runs; the server, the client and the evaluation inherit it
from the runner's environment. A prepared dataset is its directory below the cache (``<namespace>___<name>``), which an
offline load finds by itself; the locks are not copied, since ``datasets`` takes new ones in the copy. A writable cache
is left as it is.

Which datasets a test loads is read from its sheet: the trace corpus of each AgentX point (its
``WEKA_LOADER_OVERRIDE``, else its base's client env), resolved to its Hub repository through the client's own loader
registry in the staged client environment, and the evaluation's dataset (the staged GSM8K task's ``dataset_path``) when
the test runs accuracy. When a corpus cannot be resolved, the whole cache is copied if it is small, else nothing is
and the test runs on the cache as it was given.
"""

from __future__ import annotations

import contextlib
import errno
import logging
import os
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from hyperloom.common.benchmark_config import is_agentx
from inference_suite.client_env import CLIENT_SITE_PACKAGES
from inference_suite.recipe.inferencex import AGENTX_TRACE_ENV
from inference_suite.staging import GSM8K_TASK

__all__ = [
    "DATASETS_CACHE_ENV",
    "LOCAL_DATASETS_DIR",
    "WHOLE_CACHE_MAX_BYTES",
    "CopyReport",
    "DatasetsNeeded",
    "cache_dir_name",
    "copy_datasets",
    "datasets_needed",
    "is_writable",
    "local_datasets_cache",
]

log = logging.getLogger(__name__)

DATASETS_CACHE_ENV = "HF_DATASETS_CACHE"
#: The test's copy, below its work directory.
LOCAL_DATASETS_DIR = "hf_datasets"
#: The largest cache copied whole when the test's datasets cannot all be named.
WHOLE_CACHE_MAX_BYTES = 4 << 30
#: The evaluation's dataset when the staged task does not say (InferenceX's ``infx/evals/gsm8k.yaml``).
DEFAULT_EVAL_DATASET = "openai/gsm8k"
#: Where the client's loader registry names each public dataset's Hub repository, below its site-packages.
_LOADER_REGISTRY = "aiperf/plugin/plugins.yaml"
_PUBLIC_LOADERS = "public_dataset_loader"
_HUB_NAME = "hf_dataset_name"
_REFUSED = frozenset({errno.EROFS, errno.EACCES, errno.EPERM})
_LOCK_SUFFIX = ".lock"
_PROBE_PREFIX = ".hl-write-probe-"
_DATASET_PATH = re.compile(r"^dataset_path:\s*['\"]?([^'\"\s#]+)", re.MULTILINE)
# datasets.naming.camelcase_to_snakecase, which names a prepared dataset's directory.
_UPPER_UPPER = re.compile(r"([A-Z]+)([A-Z][a-z])")
_LOWER_UPPER = re.compile(r"([a-z\d])([A-Z])")
_MIB = 1 << 20


@dataclass(frozen=True)
class DatasetsNeeded:
    """The Hub repositories the test loads through ``datasets``, and the trace corpora no repository was found for."""

    repos: tuple[str, ...] = ()
    unresolved: tuple[str, ...] = ()


@dataclass(frozen=True)
class CopyReport:
    """What a copy took: its bytes and files, its seconds, and the repositories the cache holds no directory of."""

    bytes: int
    files: int
    seconds: float
    copied: tuple[str, ...]
    missing: tuple[str, ...] = ()
    skipped_links: int = 0

    def summary(self, source: str, target: Path) -> str:
        what = ", ".join(self.copied) or "nothing"
        text = (
            f"{DATASETS_CACHE_ENV} {source} is not writable (datasets takes a lock on every load); copied {what} "
            f"({self.bytes / _MIB:.1f} MiB, {self.files} files) to {target} in {self.seconds:.1f} s"
        )
        if self.missing:
            text += f"; the cache holds no {', '.join(self.missing)}"
        if self.skipped_links:
            text += f"; {self.skipped_links} links leaving the cache were not copied"
        return text


def is_writable(directory: Path) -> bool:
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


def cache_dir_name(repo: str) -> str:
    """The directory below the cache ``datasets`` prepares ``repo`` in: ``openai/gsm8k`` is ``openai___gsm8k``."""
    parts = repo.strip().strip("/").split("/")
    last = _LOWER_UPPER.sub(r"\1_\2", _UPPER_UPPER.sub(r"\1_\2", parts[-1]))
    return "___".join([*parts[:-1], last.lower()])


def _loader_registry(staged_root: Path | None) -> Mapping[str, str]:
    """Each public dataset the staged AgentX client knows, by the Hub repository it loads; empty when unreadable."""
    if staged_root is None:
        return {}
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


def _eval_dataset(staged_root: Path | None) -> str:
    """The Hub repository the staged GSM8K task loads (its ``dataset_path``), else :data:`DEFAULT_EVAL_DATASET`."""
    if staged_root is not None:
        try:
            found = _DATASET_PATH.search((staged_root / GSM8K_TASK).read_text(encoding="utf-8"))
        except OSError:
            found = None
        if found:
            return found.group(1)
    return DEFAULT_EVAL_DATASET


def _benchmark(point_yaml: str) -> Mapping[str, object]:
    try:
        document = yaml.safe_load(point_yaml)
    except yaml.YAMLError:
        return {}
    bench = document.get("benchmark") if isinstance(document, Mapping) else None
    return bench if isinstance(bench, Mapping) else {}


def _trace(bench: Mapping[str, object], base_client_env: Mapping[str, str]) -> str | None:
    envs = bench.get("envs")
    named = envs.get(AGENTX_TRACE_ENV) if isinstance(envs, Mapping) else None
    trace = str(named).strip() if named is not None else ""
    return trace or (base_client_env.get(AGENTX_TRACE_ENV) or "").strip() or None


def datasets_needed(
    points: Iterable[str], *, base_client_env: Mapping[str, str], accuracy: bool, staged_root: Path | None
) -> DatasetsNeeded:
    """The datasets a test of the point YAMLs ``points`` loads: each AgentX point's trace corpus, and the evaluation's
    dataset when the test runs accuracy. A corpus the registry does not name and that is no ``<namespace>/<name>``
    repository itself is unresolved."""
    registry: Mapping[str, str] | None = None
    repos: list[str] = []
    unresolved: list[str] = []
    for point in points:
        bench = _benchmark(point)
        if not is_agentx(bench):
            continue
        trace = _trace(bench, base_client_env)
        if trace is None:
            continue
        if registry is None:
            registry = _loader_registry(staged_root)
        repo = registry.get(trace) or (trace if "/" in trace else None)
        (repos if repo else unresolved).append(repo or trace)
    if accuracy:
        repos.append(_eval_dataset(staged_root))
    return DatasetsNeeded(tuple(dict.fromkeys(repos)), tuple(dict.fromkeys(unresolved)))


@dataclass
class _Tally:
    bytes: int = 0
    files: int = 0
    skipped_links: int = 0


def _copy_tree(source: Path, target: Path, tally: _Tally) -> None:
    """Copy ``source`` to ``target`` without its locks; a link is copied as what it names when that lies inside
    ``source``, else left out, so nothing outside the cache is read."""
    inside = source.resolve()
    for directory, dirnames, filenames in os.walk(source, followlinks=False):
        here = Path(directory)
        made = target / here.relative_to(source)
        made.mkdir(parents=True, exist_ok=True)
        for name in list(dirnames):
            if (here / name).is_symlink():
                dirnames.remove(name)
                tally.skipped_links += 1
        for name in filenames:
            if name.endswith(_LOCK_SUFFIX):
                continue
            path = here / name
            if path.is_symlink():
                resolved = path.resolve()
                if not resolved.is_relative_to(inside) or not resolved.is_file():
                    tally.skipped_links += 1
                    continue
            shutil.copy2(path, made / name)
            tally.bytes += (made / name).stat().st_size
            tally.files += 1


def copy_datasets(source: Path, target: Path, repos: Sequence[str]) -> CopyReport:
    """Copy each of ``repos`` prepared below the cache ``source`` into the cache ``target``, locks left out."""
    started = time.monotonic()
    tally = _Tally()
    copied, missing = [], []
    target.mkdir(parents=True, exist_ok=True)
    for repo in repos:
        name = cache_dir_name(repo)
        if not (source / name).is_dir():
            missing.append(repo)
            continue
        _copy_tree(source / name, target / name, tally)
        copied.append(repo)
    return CopyReport(
        tally.bytes, tally.files, time.monotonic() - started, tuple(copied), tuple(missing), tally.skipped_links
    )


def _tree_bytes(root: Path, limit: int) -> int | None:
    """The bytes of the regular files below ``root``, links not followed; ``None`` once past ``limit``."""
    total = 0
    for directory, _, filenames in os.walk(root, followlinks=False):
        for name in filenames:
            path = Path(directory) / name
            with contextlib.suppress(OSError):
                if not path.is_symlink():
                    total += path.stat().st_size
            if total > limit:
                return None
    return total


def _copy_whole(source: Path, target: Path) -> CopyReport | None:
    """The whole cache copied, locks left out, when it is at most :data:`WHOLE_CACHE_MAX_BYTES`; else ``None``."""
    if _tree_bytes(source, WHOLE_CACHE_MAX_BYTES) is None:
        return None
    started = time.monotonic()
    tally = _Tally()
    _copy_tree(source, target, tally)
    return CopyReport(
        tally.bytes, tally.files, time.monotonic() - started, ("the whole cache",), (), tally.skipped_links
    )


def _copy(source: Path, target: Path, wanted: DatasetsNeeded, named: str, notes: list[str]) -> CopyReport:
    """Copy what the test loads from ``source`` to ``target``: the whole cache when a corpus could not be named and the
    cache is small, else the named datasets."""
    report = _copy_whole(source, target) if wanted.unresolved else None
    if report is None:
        if wanted.unresolved:
            # The corpora that could not be named stay unloadable, as on the cache given; the rest load from the copy.
            unnamed = (
                f"{DATASETS_CACHE_ENV} {named} is not writable and {', '.join(wanted.unresolved)} names no dataset "
                f"repository; the cache is over {WHOLE_CACHE_MAX_BYTES >> 30} GiB, so only the named ones are copied"
            )
            log.warning("%s", unnamed)
            notes.append(unnamed)
        report = copy_datasets(source, target, wanted.repos)
    return report


@contextlib.contextmanager
def local_datasets_cache(
    environ: MutableMapping[str, str],
    work: Path,
    needed: Callable[[], DatasetsNeeded],
    notes: list[str],
    *,
    sheet_env: Mapping[str, str] | None = None,
) -> Iterator[Path | None]:
    """While the test runs, point ``environ``'s ``HF_DATASETS_CACHE`` at a copy of the datasets ``needed()`` names below
    ``work`` when the one it names is not writable; yield the copy, or ``None`` when the cache is left as it is.

    Nothing changes when ``environ`` names no cache, the sheet names one of its own (``sheet_env``), the test loads no
    dataset, or the cache is writable. ``environ`` is put back and the copy removed when the test ends. What was copied,
    its size and its time are logged and added to ``notes``.
    """
    named = (environ.get(DATASETS_CACHE_ENV) or "").strip()
    if (
        not named
        or (sheet_env or {}).get(DATASETS_CACHE_ENV, "").strip()
        or not Path(named).is_dir()
        or is_writable(Path(named))
    ):
        yield None
        return
    wanted = needed()
    if not (wanted.repos or wanted.unresolved):
        yield None
        return
    source, target = Path(named), work / LOCAL_DATASETS_DIR
    shutil.rmtree(target, ignore_errors=True)
    try:
        report = _copy(source, target, wanted, named, notes)
    except BaseException:
        shutil.rmtree(target, ignore_errors=True)
        raise
    message = report.summary(named, target)
    log.info("%s", message)
    notes.append(message)
    environ[DATASETS_CACHE_ENV] = str(target)
    try:
        yield target
    finally:
        environ[DATASETS_CACHE_ENV] = named
        shutil.rmtree(target, ignore_errors=True)
