# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""ruppell, Arbor's graph-aware GPU tracer, beside TraceLens in an SGLang profile test, at sampled detail.

ruppell attributes every kernel, graph-replayed ones included, to the module that launched it, and keeps kernel detail
for a sample of the serving steps only (``RUPPELL_DETAIL=5/1000``, which cost -0.4 % of output throughput on the profile
recipe; board ``hyperloom-multibranch#150``). A profile test runs it on a best-effort basis: it never changes the test's
return code, and a profile whose traced attempt failed runs again without it.

* **Source.** A measurement sandbox has no network, so the pinned tree comes from the runner cache, staged there on a
  machine with egress by :mod:`hyperloom.measure.ruppell_stage`::

      <cache>/tools/ruppell-<commit>/source/     the tree at the commit, with ``.commit`` (ruppell reads it without .git)
      <cache>/tools/ruppell-<commit>/manifest    JSON, written last: the commit and the tree's digest

  This file pins the commit and the digest (:data:`RUPPELL_COMMIT`, :data:`RUPPELL_TREE_SHA256`); the tree a test uses
  is a copy of the entry whose digest is the pinned one, so nothing the server writes lands in the cache.
* **Build.** ``runtime/build.sh`` in the test's own image, before the server starts: the tool library links the image's
  ROCm profiler SDK and the GPU code is built for the card's architecture.
* **Launch.** Magpie and InferenceX own the server's command line, so the server is not wrapped in ``ruppell run``: the
  environment ``ruppell run`` would give its command goes into the point's ``benchmark.envs``. It is taken from
  ``ruppell run`` itself (a probe command that prints its environment), so ruppell's own rules apply (the tool library,
  its early load, the NCCL plugin, packet capture off on ROCm 7.2.0-7.2.2, a value the point sets kept). ruppell's
  start-up hook goes first on ``PYTHONPATH``: it runs the ``sitecustomize`` it shadows by looking from the start of
  ``sys.path``, so anywhere else it would run a farm hook a second time.
* **Report.** After the round: ``ruppell census``, ``ruppell export`` (for ``check_m2``'s ``report.json``; the trace
  itself is not kept) and ``tests/m2/check_m2.py``, then the GPU time of each module by step kind
  (:data:`MODULE_SHARES_FILE`). What is kept goes to ``<output_dir>/ruppell``, beside ``profile/`` and never in it: a
  ``.json`` below ``profile/`` would read as a trace. The raw records are not kept.
"""

from __future__ import annotations

import contextlib
import csv
import gzip
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from hyperloom.common.benchmark_config import is_magpie_agentx
from hyperloom.measure.short_prompts.staging import CACHE_ENV, DEFAULT_CACHE_DIR

__all__ = [
    "CHECK_FILE",
    "COMMIT_FILE",
    "DETAIL",
    "MODULE_SHARES_FILE",
    "OUTPUT_DIR",
    "RUPPELL_COMMIT",
    "RUPPELL_TREE_SHA256",
    "RuppellError",
    "Tracer",
    "entry_dir",
    "entry_manifest",
    "entry_problem",
    "module_shares",
    "prepare",
    "tree_sha256",
]

#: The ruppell commit every profile test builds, and the digest (:func:`tree_sha256`) of its staged
#: ``source`` directory, ``.commit`` included. ``0ef9b0d`` is the first commit that turns HIP graph packet capture off on
#: ROCm 7.2.0-7.2.2, where kernel dispatch tracing otherwise crashes the first decode graph replay.
RUPPELL_COMMIT = "0ef9b0dac1cfd3caf533346aca5bdbbc925c918b"
RUPPELL_TREE_SHA256 = "32710ea380a08db3a2611965979875533adf7d99b706d68fc158ab10a178e6fa"
#: The detail sampling of a profile test: kernel detail for 5 of every 1000 serving steps.
DETAIL = "5/1000"
#: The ROCm releases ruppell supports, [from, to): its QUICKSTART says 7.0.2 to 7.2.3, its checks covered 7.2.4, and
#: it has never been built on ROCm 10.
SUPPORTED_ROCM = ((7, 0, 2), (8, 0, 0))

TOOLS_DIR = "tools"
ENTRY_PREFIX = "ruppell-"
SOURCE_DIR = "source"
MANIFEST_NAME = "manifest"
COMMIT_FILE = ".commit"
#: Below a test's output directory, beside ``profile``.
OUTPUT_DIR = "ruppell"
MODULE_SHARES_FILE = "module_shares.csv"
CHECK_FILE = "check.json"
SUMMARY_FILE = "census_summary.json"
LOG_FILE = "ruppell.log"
#: The tail of the server log of a traced attempt that failed, kept when the profile ran again without ruppell.
ATTEMPT_LOG_FILE = "attempt_server.log"
MODULE_SHARES_COLUMNS = ("kind", "module", "gpu_ms", "share_percent", "steps")
UNATTRIBUTED = "unattributed"

_SCHEMA = 1
_UTF8 = "utf-8"
_BUILD_TIMEOUT_SEC = 300.0
_PROBE_TIMEOUT_SEC = 60.0
_REPORT_TIMEOUT_SEC = 600.0
_LOG_TAIL_BYTES = 64 * 1024
_PYTHONPATH = "PYTHONPATH"
_SGLANG_PLUGIN = "ruppell_sglang-0.0.dist-info"
_PLUGIN_METADATA = "Metadata-Version: 2.1\nName: ruppell-sglang\nVersion: 0.0\n"
_PLUGIN_ENTRY_POINTS = "[sglang.srt.plugins]\nruppell = ruppell.adapters.sglang:register\n"
#: What the profile's server gets on top of what ``ruppell run`` sets: the SGLang adapter on, and sampled detail.
_TRACED_ENVS = {"RUPPELL_SGLANG": "1", "RUPPELL_DETAIL": DETAIL}
_EARLY_HOOK = ("python", "ruppell", "_rocm_early")
_CHECK_SCRIPT = ("tests", "m2", "check_m2.py")
_DUMP_ENV = "import json, os, sys; json.dump(dict(os.environ), sys.stdout)"
_VERSION = re.compile(r"(\d+)\.(\d+)\.(\d+)")
_GFX = re.compile(r"gfx[0-9a-f]+")
#: How the log says how each command ended and how long it took (the build's time is measured on the first GPU run).
_ELAPSED = "# exit"


class RuppellError(Exception):
    """ruppell could not be made ready, or its report could not be made; the message says which step and why."""


# --- the cache entry ----------------------------------------------------------------------------------------------


def entry_dir(cache: Path, commit: str) -> Path:
    """Where the runner cache keeps the tree of ``commit``."""
    return Path(cache) / TOOLS_DIR / f"{ENTRY_PREFIX}{commit}"


def entry_manifest(commit: str, files: int, size: int, sha256: str) -> bytes:
    """The manifest of an entry: its commit and the digest of its ``source`` directory (:func:`tree_sha256`)."""
    document = {"schema": _SCHEMA, "commit": commit, "source": {"files": files, "size": size, "sha256": sha256}}
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode(_UTF8)


def tree_sha256(root: Path) -> str:
    """The SHA-256 ``hyperloom.measure.runner.cache_layout.tree_digest`` gives the tree at ``root``: one line per file
    or link, in path order, of its kind (``f``, ``x`` executable, ``l`` link), size, content hash and path.

    The runner cache's own digest, computed here because the runner package imports this backend; a test pins the two
    to the same value.
    """
    outer = hashlib.sha256()
    root = Path(root)
    for path in _files(root):
        info = path.lstat()
        if path.is_symlink():
            target = os.readlink(path)
            kind, content, size = "l", hashlib.sha256(target.encode()).hexdigest(), len(target)
        else:
            kind, content, size = "x" if info.st_mode & 0o111 else "f", _file_sha256(path), info.st_size
        outer.update(f"{kind}\t{size}\t{content}\t{path.relative_to(root).as_posix()}\n".encode())
    return outer.hexdigest()


def _files(directory: Path) -> Iterator[Path]:
    for entry in sorted(os.scandir(directory), key=lambda item: item.name):
        if entry.is_dir(follow_symlinks=False):
            yield from _files(Path(entry.path))
        else:
            yield Path(entry.path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def entry_problem(entry: Path, commit: str, wanted: str) -> str:
    """What is wrong with the entry at ``entry`` for ``commit`` and the tree digest ``wanted``, or ``""``.

    The manifest must name both, and the ``source`` directory must have that digest now, not only when it was staged.
    """
    try:
        manifest = json.loads((Path(entry) / MANIFEST_NAME).read_text(encoding=_UTF8))
        recorded = (manifest["commit"], manifest["source"]["sha256"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return f"{entry} has no readable manifest ({type(exc).__name__}): not staged, or unfinished"
    if recorded != (commit, wanted):
        return f"{entry} holds {recorded[0]} with tree {recorded[1]}, not {commit} with tree {wanted}"
    found = tree_sha256(Path(entry) / SOURCE_DIR)
    if found != wanted:
        return f"the source of {entry} is not the tree its manifest records ({found})"
    return ""


# --- preparing a test ---------------------------------------------------------------------------------------------


@dataclass
class Tracer:
    """ruppell built for one test: its copy of the source, the build, and where the server writes its records."""

    work: Path
    source: Path
    build: Path

    @property
    def records(self) -> Path:
        return self.work / "records"

    @property
    def log(self) -> Path:
        return self.work / LOG_FILE

    def traced_points(self, points: Sequence[Path], environ: Mapping[str, str]) -> tuple[Path, ...]:
        """Copies of ``points``, below the work directory, whose servers run under ruppell.

        Raises:
            RuppellError: ``ruppell run`` did not give its environment.
        """
        site = self._plugin_site()
        traced = []
        for index, point in enumerate(points):
            config = yaml.safe_load(Path(point).read_text(encoding=_UTF8)) or {}
            benchmark = config.setdefault("benchmark", {})
            envs = benchmark.get("envs") or {}
            benchmark["envs"] = {**envs, **self._server_envs(envs, environ, site)}
            target = self.work / f"point_{index}.yaml"
            target.write_text(yaml.safe_dump(config, sort_keys=False), encoding=_UTF8)
            traced.append(target)
        return tuple(traced)

    def _plugin_site(self) -> Path:
        """A directory whose distribution registers ruppell's SGLang adapter (SGLang finds plugins by entry point)."""
        site = self.work / "site"
        info = site / _SGLANG_PLUGIN
        info.mkdir(parents=True, exist_ok=True)
        (info / "METADATA").write_text(_PLUGIN_METADATA, encoding=_UTF8)
        (info / "entry_points.txt").write_text(_PLUGIN_ENTRY_POINTS, encoding=_UTF8)
        return site

    def _server_envs(self, envs: Mapping[str, Any], environ: Mapping[str, str], site: Path) -> dict[str, str]:
        """What the point's ``envs`` gain: the variables ``ruppell run`` sets for its command, the adapter's and the
        ``PYTHONPATH`` in the order the hooks need."""
        given = {str(key): str(value) for key, value in envs.items() if value is not None}
        base = given[_PYTHONPATH] if _PYTHONPATH in given else environ.get(_PYTHONPATH, "")
        added = self._probe({**environ, **given, **_TRACED_ENVS, _PYTHONPATH: base})
        early = str(self.source.joinpath(*_EARLY_HOOK))
        parts = [early, *base.split(os.pathsep), str(self.source / "python"), str(site)]
        path = os.pathsep.join(dict.fromkeys(part for part in parts if part))
        return {**_TRACED_ENVS, **added, _PYTHONPATH: path}

    def _probe(self, environ: Mapping[str, str]) -> dict[str, str]:
        """The variables ``ruppell run`` adds to or changes in ``environ`` for its command, read from a command that
        prints its environment; ``PYTHONPATH`` is left out: the server's is laid out by the caller."""
        command = [
            sys.executable, "-m", "ruppell", "run", "--build", str(self.build), "--out", str(self.records), "--",
            sys.executable, "-I", "-c", _DUMP_ENV,
        ]  # fmt: skip
        probe = {**environ, _PYTHONPATH: _join(str(self.source / "python"), environ.get(_PYTHONPATH, ""))}
        probe["PYTHONDONTWRITEBYTECODE"] = "1"
        printed = _run(command, probe, _PROBE_TIMEOUT_SEC, self.log, cwd=self.work, capture=True)
        try:
            seen = json.loads(printed)
        except ValueError as exc:
            raise RuppellError(f"ruppell run printed no environment: {exc}") from exc
        if not isinstance(seen, dict):
            raise RuppellError("ruppell run printed no environment")
        return {
            str(key): str(value)
            for key, value in seen.items()
            if key != _PYTHONPATH and probe.get(str(key)) != str(value)
        }

    def report(self, output_dir: Path, ranks: int) -> str:
        """Analyse the records into ``<output_dir>/ruppell``; return the note the test carries.

        Never raises: a report that fails is a note, and the test's return code is the round's.
        """
        kept = Path(output_dir) / OUTPUT_DIR
        kept.mkdir(parents=True, exist_ok=True)
        try:
            return self._report(kept, ranks)
        except (RuppellError, OSError, ValueError) as exc:
            return f"ruppell: report failed ({_first_line(str(exc))})"
        finally:
            with contextlib.suppress(OSError):
                shutil.copyfile(self.log, kept / LOG_FILE)

    def _report(self, kept: Path, ranks: int) -> str:
        deadline = time.monotonic() + _REPORT_TIMEOUT_SEC
        records = self.records
        if not (records / "run.json").is_file():
            raise RuppellError("the server left no records")
        env = {**os.environ, _PYTHONPATH: str(self.source / "python"), "PYTHONDONTWRITEBYTECODE": "1"}
        ruppell = [sys.executable, "-m", "ruppell"]
        _run([*ruppell, "census", str(records)], env, _left(deadline), self.log, cwd=self.work)
        trace = self.work / "trace.json"
        _run(
            [*ruppell, "export", str(records), "-o", str(trace), "--no-flows"],
            env,
            _left(deadline),
            self.log,
            cwd=self.work,
        )
        with contextlib.suppress(OSError):
            trace.unlink()
        check = [sys.executable, str(self.source.joinpath(*_CHECK_SCRIPT)), str(records), "--expect-ranks", str(ranks)]
        verdict = _run(check, env, _left(deadline), self.log, cwd=self.work, capture=True, accept=(0, 1))
        (kept / CHECK_FILE).write_text(verdict, encoding=_UTF8)
        for name in ("run.json", SUMMARY_FILE, "census_scopes.json", "census_scopes.npz"):
            shutil.copyfile(records / name, kept / name)
        census = records / "census.jsonl"
        with census.open("rb") as source, gzip.open(kept / "census.jsonl.gz", "wb") as target:
            shutil.copyfileobj(source, target)
        write_module_shares(module_shares(census), kept / MODULE_SHARES_FILE)
        return _status(json.loads((kept / SUMMARY_FILE).read_text(encoding=_UTF8)), json.loads(verdict))

    def keep_attempt(self, server_log: str | None, output_dir: Path) -> None:
        """Keep the tail of a failed traced attempt's server log beside the test's output, where it says why."""
        if not server_log:
            return
        kept = Path(output_dir) / OUTPUT_DIR
        kept.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(OSError):
            with open(server_log, "rb") as handle:
                handle.seek(0, os.SEEK_END)
                handle.seek(max(0, handle.tell() - _LOG_TAIL_BYTES))
                (kept / ATTEMPT_LOG_FILE).write_bytes(handle.read())

    def close(self) -> None:
        shutil.rmtree(self.work, ignore_errors=True)


def prepare(points: Sequence[Path], environ: Mapping[str, str]) -> Tracer | str:
    """ruppell built for a profile of ``points``, or why the profile runs without it.

    No step needs a GPU: the point must be an SGLang server that is not an AgentX launch, the image's ROCm one ruppell
    supports, the pinned tree staged in the runner cache, and the build must succeed in this image.
    """
    benchmark = (yaml.safe_load(Path(points[0]).read_text(encoding=_UTF8)) or {}).get("benchmark") or {}
    framework = str(benchmark.get("framework") or "").lower()
    if "sglang" not in framework:
        return f"ruppell: skipped (the framework is {framework or 'unnamed'}, not SGLang)"
    if is_magpie_agentx(benchmark):
        return "ruppell: skipped (an AgentX launch takes its environment elsewhere)"
    rocm = Path(environ.get("ROCM_PATH") or "/opt/rocm")
    version = rocm_version(rocm)
    lo, hi = SUPPORTED_ROCM
    if version is None or not lo <= version < hi:
        shown = ".".join(map(str, version)) if version else "unknown"
        return f"ruppell: skipped (ROCm {shown} is outside {_dotted(lo)} to {_dotted(hi)} (exclusive))"
    entry = entry_dir(Path(environ.get(CACHE_ENV) or DEFAULT_CACHE_DIR), RUPPELL_COMMIT)
    if not entry.is_dir():
        return f"ruppell: skipped (not staged: no {entry})"
    arch = gpu_arch(rocm)
    if arch is None:
        return f"ruppell: skipped (no GPU architecture from {rocm}/bin/rocm_agent_enumerator)"
    work = Path(tempfile.mkdtemp(prefix="hl-ruppell-"))
    try:
        return _built(entry, work, arch, rocm, environ)
    except (RuppellError, OSError) as exc:
        shutil.rmtree(work, ignore_errors=True)
        return f"ruppell: skipped ({_first_line(str(exc))})"


def _built(entry: Path, work: Path, arch: str, rocm: Path, environ: Mapping[str, str]) -> Tracer:
    """Copy the staged tree, check that the copy is the pinned one, and build it."""
    source = work / SOURCE_DIR
    shutil.copytree(entry / SOURCE_DIR, source, symlinks=True)
    problem = entry_problem(entry, RUPPELL_COMMIT, RUPPELL_TREE_SHA256)
    found = tree_sha256(source)
    if problem or found != RUPPELL_TREE_SHA256:
        raise RuppellError(problem or f"the copy of {entry} has tree {found}, not {RUPPELL_TREE_SHA256}")
    tracer = Tracer(work, source, work / "build")
    env = {
        **environ,
        "RUPPELL_PLATFORM": "rocm",
        "RUPPELL_ARCH": arch,
        "RUPPELL_BUILD": str(tracer.build),
        "ROCM_PATH": str(rocm),
    }
    _run(["bash", str(source / "runtime" / "build.sh")], env, _BUILD_TIMEOUT_SEC, tracer.log, cwd=work)
    return tracer


def rocm_version(rocm: Path) -> tuple[int, int, int] | None:
    """The ROCm release at ``rocm``, from ``.info/version``, as ruppell reads it."""
    try:
        found = _VERSION.match((Path(rocm) / ".info" / "version").read_text(encoding=_UTF8).strip())
    except OSError:
        return None
    return (int(found[1]), int(found[2]), int(found[3])) if found else None


def gpu_arch(rocm: Path) -> str | None:
    """The first GPU architecture ``rocm_agent_enumerator`` names (``gfx950``), or ``None``: ``build.sh`` would
    otherwise build for ``gfx950`` whatever the card is."""
    try:
        done = subprocess.run(
            [str(Path(rocm) / "bin" / "rocm_agent_enumerator")], capture_output=True, text=True, timeout=60, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    found = [
        line.strip() for line in done.stdout.splitlines() if _GFX.fullmatch(line.strip()) and line.strip() != "gfx000"
    ]
    return found[0] if found else None


# --- the report ---------------------------------------------------------------------------------------------------


def module_shares(census: Path) -> list[dict[str, Any]]:
    """The GPU time of each module in the census's engine steps, by step kind (decode, prefill, mixed).

    A census line holds one traced step of one process; ``by_op_ns`` is the step's kernel time by the op of the scope
    that launched it (for SGLang, the module's class), and ``unattributed`` the time no scope launched. Each kind's
    shares add up to 100 % of its kernel time. Startup steps are not engine steps and are left out.
    """
    time_by: dict[tuple[str, str], int] = {}
    steps: dict[str, int] = {}
    with Path(census).open(encoding=_UTF8) as lines:
        for line in lines:
            if not line.strip():
                continue
            step = json.loads(line)
            fields = step.get("fields") or {}
            if fields.get("phase") != "engine":
                continue
            kind = str(fields.get("kind") or "other")
            steps[kind] = steps.get(kind, 0) + 1
            for module, ns in (step.get("by_op_ns") or {}).items():
                key = (kind, str(module) or "(no op)")
                time_by[key] = time_by.get(key, 0) + int(ns)
    totals: dict[str, int] = {}
    for (kind, _), ns in time_by.items():
        totals[kind] = totals.get(kind, 0) + ns
    rows = [
        {
            "kind": kind,
            "module": module,
            "gpu_ms": round(ns / 1e6, 3),
            "share_percent": round(100.0 * ns / totals[kind], 3) if totals[kind] else 0.0,
            "steps": steps[kind],
        }
        for (kind, module), ns in time_by.items()
    ]
    return sorted(rows, key=lambda row: (row["kind"], -row["gpu_ms"], row["module"]))


def write_module_shares(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    with Path(path).open("w", encoding=_UTF8, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MODULE_SHARES_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _status(summary: Mapping[str, Any], verdict: Mapping[str, Any]) -> str:
    processes = (summary.get("processes") or {}).values()
    detail = max((int(p.get("engine_steps") or 0) for p in processes), default=0)
    parts = [f"{detail} detail steps", "check ok" if verdict.get("ok") is True else "check failed"]
    if summary.get("incomplete_processes"):
        parts.append("records incomplete")
    return f"ruppell: ok, {', '.join(parts)}"


# --- processes ----------------------------------------------------------------------------------------------------


def _run(
    command: list[str],
    env: Mapping[str, str],
    timeout: float,
    log: Path,
    *,
    cwd: Path,
    capture: bool = False,
    accept: tuple[int, ...] = (0,),
) -> str:
    """Run ``command`` in a process group of its own, its stderr (and its stdout unless captured) appended to ``log``;
    return what it printed when ``capture``.

    Raises:
        RuppellError: It could not start, ran past ``timeout`` (its whole group is killed) or exited with another code
            than ``accept``.
    """
    name = " ".join(Path(part).name for part in command[:4])
    with Path(log).open("ab") as sink:
        sink.write(f"$ {' '.join(command)}\n".encode(_UTF8))
        sink.flush()
        started = time.monotonic()
        try:
            process = subprocess.Popen(
                command,
                env=dict(env),
                cwd=cwd,
                stdout=subprocess.PIPE if capture else sink,
                stderr=sink,
                start_new_session=True,
            )
        except OSError as exc:
            raise RuppellError(f"{name} could not start: {exc}") from exc
        try:
            out, _ = process.communicate(timeout=max(timeout, 1.0))
        except subprocess.TimeoutExpired as exc:
            with contextlib.suppress(OSError):
                os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            raise RuppellError(f"{name} ran past {timeout:.0f}s") from exc
        finally:
            sink.write(f"{_ELAPSED} {process.returncode} after {time.monotonic() - started:.1f}s\n".encode(_UTF8))
    if process.returncode not in accept:
        raise RuppellError(f"{name} exited {process.returncode}: {_log_tail(log)}")
    return (out or b"").decode(_UTF8, errors="replace") if capture else ""


def _log_tail(log: Path) -> str:
    try:
        text = Path(log).read_text(encoding=_UTF8, errors="replace")
        lines = [line for line in text.splitlines() if line.strip() and not line.startswith(_ELAPSED)]
    except OSError:
        return ""
    return lines[-1][:300] if lines else ""


def _left(deadline: float) -> float:
    left = deadline - time.monotonic()
    if left <= 0:
        raise RuppellError(f"the report ran past {_REPORT_TIMEOUT_SEC:.0f}s")
    return left


def _join(*parts: str) -> str:
    return os.pathsep.join(part for part in parts if part)


def _first_line(text: str) -> str:
    return (text.strip().splitlines() or [""])[0][:300]


def _dotted(version: tuple[int, ...]) -> str:
    return ".".join(map(str, version))
