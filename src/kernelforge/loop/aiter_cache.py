# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Per-attempt AITER build-cache isolation and owned-lock cleanup."""

from __future__ import annotations

import atexit
import contextlib
import hashlib
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from kernelforge.durable_io import atomic_write_text


_OWNER_FILE = ".forge_cache_owner.json"
_LOCK_NAMES = {"lock", ".ninja_lock"}
_REGISTERED_SHARDS: set[str] = set()
_SOURCE_KEY_SCHEMA = b"forge-aiter-source-cache-v2\0"
DEFAULT_AITER_CACHE_MAX_BYTES = 4 * 1024**3
DEFAULT_AITER_CACHE_TARGET_BYTES = 3 * 1024**3

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AiterCachePolicy:
    """Disk budget for one Forge attempt's private AITER source shards."""

    max_bytes: int = DEFAULT_AITER_CACHE_MAX_BYTES
    target_bytes: int = DEFAULT_AITER_CACHE_TARGET_BYTES


_CACHE_POLICIES: dict[str, AiterCachePolicy] = {}


@dataclass(frozen=True)
class AiterCacheIsolation:
    """A Forge process's private AITER build roots."""

    cache_root: Path
    aiter_root_dir: Path
    aiter_jit_dir: Path
    flydsl_cache_dir: Path
    owner_file: Path
    owner_pid: int


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, sort_keys=True))


def configure_aiter_cache_isolation(
    experiments_dir: Path,
    *,
    max_cache_bytes: int = DEFAULT_AITER_CACHE_MAX_BYTES,
) -> AiterCacheIsolation:
    """Route every AITER-adjacent runtime compiler to one private Forge tree.

    Three of them reach the run's workspace: ``cpp_itfs`` (``AITER_ROOT_DIR``),
    ``compile_ops`` (``AITER_JIT_DIR``) and FlyDSL
    (``FLYDSL_RUNTIME_CACHE_DIR``). Missing any one leaves its build products in
    a git-visible directory -- see the FlyDSL note below for what that costs.
    """
    cache_root = (experiments_dir / "aiter_cache").resolve()
    max_cache_bytes = max(0, int(max_cache_bytes))
    target_cache_bytes = min(
        max_cache_bytes,
        int(max_cache_bytes * 0.75),
    )
    policy = AiterCachePolicy(
        max_bytes=max_cache_bytes,
        target_bytes=target_cache_bytes,
    )
    _CACHE_POLICIES[str(cache_root)] = policy
    aiter_root_dir = cache_root / "cpp_itfs"
    aiter_jit_dir = cache_root / "jit"
    flydsl_cache_dir = cache_root / "flydsl_cache"
    aiter_root_dir.mkdir(parents=True, exist_ok=True)
    aiter_jit_dir.mkdir(parents=True, exist_ok=True)
    flydsl_cache_dir.mkdir(parents=True, exist_ok=True)
    _seed_flydsl_cache(flydsl_cache_dir)
    owner_file = cache_root / _OWNER_FILE
    owner_pid = os.getpid()
    _atomic_write_json(
        owner_file,
        {
            "schema_version": 1,
            "owner_pid": owner_pid,
            "created_unix": time.time(),
            "aiter_root_dir": str(aiter_root_dir),
            "aiter_jit_dir": str(aiter_jit_dir),
            "flydsl_cache_dir": str(flydsl_cache_dir),
            "max_cache_bytes": policy.max_bytes,
            "target_cache_bytes": policy.target_bytes,
        },
    )

    # cpp_itfs uses AITER_ROOT_DIR/build while compile_ops uses
    # AITER_JIT_DIR/build. Both must be redirected; setting only the latter
    # leaves paged-attention locks in ~/.aiter/build.
    os.environ["AITER_ROOT_DIR"] = str(aiter_root_dir)
    os.environ["AITER_JIT_DIR"] = str(aiter_jit_dir)
    # THREE runtime compilers reach this tree, not two. FlyDSL is the third, and
    # it was the one left out: `aiter/__init__.py` points FLYDSL_RUNTIME_CACHE_DIR
    # at `<aiter package>/jit/flydsl_cache` on import, and that package lives
    # inside the run's workspace, so every FlyDSL kernel wrote its cache into a
    # git-visible directory the workspace .gitignore does not cover.
    #
    # That is not merely untidy. The guard the default backend grew in #22 fails
    # a session on any new non-ignored file, and FlyDSL names each cache entry
    # after a hash of the kernel source -- so every edit the agent makes creates
    # a *new* directory, which `allow_dirty_baseline` cannot forgive because it
    # only pardons state that predates the session. Across 2026-08-23-1200 and
    # 08-24-0000 this voided 16 iterations outright (correctness and benchmark
    # both skipped) and burned 840 minutes; on one run it took 43% of the budget.
    #
    # aiter only sets the variable when it is absent, and FlyDSL re-reads it from
    # the environment on every access (`flydsl.utils.env.OptStr` is a descriptor),
    # so claiming it here is sufficient regardless of import order.
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = str(flydsl_cache_dir)
    os.environ["FORGE_AITER_CACHE_ROOT"] = str(cache_root)
    os.environ["FORGE_AITER_CACHE_OWNER_PID"] = str(owner_pid)
    os.environ.pop("AITER_REBUILD", None)

    isolation = AiterCacheIsolation(
        cache_root=cache_root,
        aiter_root_dir=aiter_root_dir,
        aiter_jit_dir=aiter_jit_dir,
        flydsl_cache_dir=flydsl_cache_dir,
        owner_file=owner_file,
        owner_pid=owner_pid,
    )
    atexit.register(cleanup_owned_aiter_locks, isolation)
    atexit.register(cleanup_owned_aiter_cache, isolation)
    return isolation


def child_cache_environment(cache_root: Path) -> dict[str, str]:
    """Create one private build cache and return the env that selects it.

    All three runtime compilers are redirected exactly as
    :func:`configure_aiter_cache_isolation` redirects them -- ``cpp_itfs`` reads
    ``AITER_ROOT_DIR``, ``compile_ops`` reads ``AITER_JIT_DIR`` and FlyDSL reads
    ``FLYDSL_RUNTIME_CACHE_DIR`` -- but the
    values are returned instead of written to ``os.environ``, so a caller that
    is one of several running concurrently in this process cannot overwrite what
    the others are using. The caller applies them to one spawned subprocess.

    ``FORGE_AITER_CACHE_ROOT`` names the private root so a source-keyed
    activation inside that subprocess shards under it rather than under the
    shared cache. ``FORGE_AITER_CACHE_OWNER_PID`` stays this process's pid,
    because this process creates the root and is the one that removes it.

    No prebuilt AITER module is seeded into the shard (see
    :func:`seed_prebuilt_modules`), so a subprocess that edits a source compiles
    that source instead of importing a ``.so`` built from another one. The
    FlyDSL shard *is* seeded, because its entries are keyed by a hash of the
    kernel source: an edit lands on a different key and compiles, so a warm
    entry can never stand in for it the way a name-keyed ``.so`` can.

    Creating the directories here is what makes a failure loud -- the caller
    gets an ``OSError`` rather than a cache root it cannot use.
    """
    cache_root = Path(cache_root).resolve()
    aiter_root_dir = cache_root / "cpp_itfs"
    aiter_jit_dir = cache_root / "jit"
    flydsl_cache_dir = cache_root / "flydsl_cache"
    aiter_root_dir.mkdir(parents=True, exist_ok=True)
    aiter_jit_dir.mkdir(parents=True, exist_ok=True)
    flydsl_cache_dir.mkdir(parents=True, exist_ok=True)
    _seed_flydsl_cache(flydsl_cache_dir)
    return {
        "AITER_ROOT_DIR": str(aiter_root_dir),
        "AITER_JIT_DIR": str(aiter_jit_dir),
        "FLYDSL_RUNTIME_CACHE_DIR": str(flydsl_cache_dir),
        "FORGE_AITER_CACHE_ROOT": str(cache_root),
        "FORGE_AITER_CACHE_OWNER_PID": str(os.getpid()),
    }


def _source_digest(source_files: list[str]) -> str:
    """Hash only the declared build inputs, independent of unrelated repo state."""
    digest = hashlib.sha256()
    digest.update(_SOURCE_KEY_SCHEMA)
    paths = {Path(str(raw_path)).expanduser().resolve(strict=False) for raw_path in source_files if raw_path}
    for path in sorted(paths, key=str):
        digest.update(str(path).encode("utf-8", errors="replace"))
        digest.update(b"\0")
        try:
            digest.update(path.read_bytes())
        except OSError:
            digest.update(b"<unreadable>")
        digest.update(b"\0")
    return digest.hexdigest()[:24]


def _read_owner(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}


def _directory_size(path: Path) -> int:
    """Return recursively allocated bytes without following symlinks."""
    total = 0
    try:
        for root, _dirs, files in os.walk(path):
            for name in files:
                try:
                    stat = (Path(root) / name).stat(follow_symlinks=False)
                    allocated = getattr(stat, "st_blocks", 0) * 512
                    total += allocated or stat.st_size
                except OSError:
                    continue
    except OSError:
        return 0
    return total


def _source_shard_isolation(path: Path, owner_pid: int) -> AiterCacheIsolation:
    return AiterCacheIsolation(
        cache_root=path,
        aiter_root_dir=path / "cpp_itfs",
        aiter_jit_dir=path / "jit",
        flydsl_cache_dir=path / "flydsl_cache",
        owner_file=path / _OWNER_FILE,
        owner_pid=owner_pid,
    )


def prune_aiter_cache_shards(
    cache_root: Path,
    *,
    protected_shard: Path,
) -> dict[str, Any]:
    """Prune least-recently-used inactive shards to the attempt's target size."""
    cache_root = cache_root.resolve()
    protected_shard = protected_shard.resolve()
    policy = _CACHE_POLICIES.get(str(cache_root), AiterCachePolicy())
    stats: dict[str, Any] = {
        "cache_root": str(cache_root),
        "max_bytes": policy.max_bytes,
        "target_bytes": policy.target_bytes,
        "before_bytes": 0,
        "after_bytes": 0,
        "deleted_bytes": 0,
        "deleted_shards": [],
        "skipped_live_shards": [],
        "errors": 0,
    }
    sources_root = cache_root / "sources"
    if policy.max_bytes <= 0 or not sources_root.is_dir():
        return stats

    owner_pid = os.getpid()
    shards: list[tuple[float, Path, int]] = []
    try:
        candidates = [path for path in sources_root.iterdir() if path.is_dir()]
    except OSError:
        stats["errors"] += 1
        return stats
    for path in candidates:
        owner = _read_owner(path / _OWNER_FILE)
        try:
            last_used = float(owner.get("last_used_unix") or owner.get("created_unix") or path.stat().st_mtime)
        except (OSError, TypeError, ValueError):
            last_used = 0.0
        size = _directory_size(path)
        stats["before_bytes"] += size
        shards.append((last_used, path, size))

    stats["after_bytes"] = stats["before_bytes"]
    if stats["before_bytes"] <= policy.max_bytes:
        return stats

    for _last_used, path, size in sorted(shards, key=lambda item: item[0]):
        if stats["after_bytes"] <= policy.target_bytes:
            break
        if path.resolve() == protected_shard:
            continue
        isolation = _source_shard_isolation(path, owner_pid)
        live_users = _live_cache_users(isolation)
        if live_users is None or live_users:
            stats["skipped_live_shards"].append(str(path))
            continue
        try:
            shutil.rmtree(path)
            stats["after_bytes"] = max(0, stats["after_bytes"] - size)
            stats["deleted_bytes"] += size
            stats["deleted_shards"].append(str(path))
            _REGISTERED_SHARDS.discard(str(path))
        except OSError:
            stats["errors"] += 1

    if stats["after_bytes"] > policy.max_bytes:
        log.warning(
            "AITER cache remains over budget: root=%s size=%d max=%d",
            cache_root,
            stats["after_bytes"],
            policy.max_bytes,
        )
    elif stats["deleted_shards"]:
        log.info(
            "pruned %d AITER cache shard(s), freeing %d bytes",
            len(stats["deleted_shards"]),
            stats["deleted_bytes"],
        )
    return stats


def activate_aiter_cache_for_sources(
    source_files: list[str],
) -> AiterCacheIsolation | None:
    """Select a cache shard keyed by the current editable source contents."""
    cache_root_raw = os.environ.get("FORGE_AITER_CACHE_ROOT", "").strip()
    if not cache_root_raw:
        return None
    cache_root = Path(cache_root_raw).resolve() / "sources" / _source_digest(source_files)
    aiter_root_dir = cache_root / "cpp_itfs"
    aiter_jit_dir = cache_root / "jit"
    flydsl_cache_dir = cache_root / "flydsl_cache"
    aiter_root_dir.mkdir(parents=True, exist_ok=True)
    aiter_jit_dir.mkdir(parents=True, exist_ok=True)
    flydsl_cache_dir.mkdir(parents=True, exist_ok=True)
    # Every source shard is a fresh directory, so without seeding each one
    # cold-compiles the kernels the edit never touched. Content addressing is
    # what makes that safe: the edited kernel hashes to a key no seeded entry
    # occupies.
    _seed_flydsl_cache(flydsl_cache_dir)
    owner_file = cache_root / _OWNER_FILE
    owner_pid = os.getpid()
    now = time.time()
    existing_owner = _read_owner(owner_file)
    _atomic_write_json(
        owner_file,
        {
            "schema_version": 1,
            "owner_pid": owner_pid,
            "created_unix": existing_owner.get("created_unix", now),
            "last_used_unix": now,
            "aiter_root_dir": str(aiter_root_dir),
            "aiter_jit_dir": str(aiter_jit_dir),
            "flydsl_cache_dir": str(flydsl_cache_dir),
        },
    )
    os.environ["AITER_ROOT_DIR"] = str(aiter_root_dir)
    os.environ["AITER_JIT_DIR"] = str(aiter_jit_dir)
    # Shard FlyDSL with the rest. Its entries are content-addressed, so sharing
    # one directory would be correct -- but concurrent lanes each write a `.lock`
    # beside the entry they build, and the lane copies are what this shard keeps
    # apart in the first place.
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = str(flydsl_cache_dir)
    os.environ["FORGE_AITER_CACHE_OWNER_PID"] = str(owner_pid)
    os.environ.pop("AITER_REBUILD", None)
    isolation = AiterCacheIsolation(
        cache_root=cache_root,
        aiter_root_dir=aiter_root_dir,
        aiter_jit_dir=aiter_jit_dir,
        flydsl_cache_dir=flydsl_cache_dir,
        owner_file=owner_file,
        owner_pid=owner_pid,
    )
    key = str(cache_root)
    if key not in _REGISTERED_SHARDS:
        _REGISTERED_SHARDS.add(key)
        atexit.register(cleanup_owned_aiter_locks, isolation)
    prune_aiter_cache_shards(
        Path(cache_root_raw),
        protected_shard=cache_root,
    )
    return isolation


def _global_aiter_jit_dirs() -> list[Path]:
    """Locate all prebuilt JIT directories where warm ``.so`` may live.

    Checked in order: an explicit ``FORGE_AITER_WARM_JIT_DIR`` override (returned
    alone), then the installed package's own ``aiter/jit``, then ``~/.aiter/jit``.
    Both package and user dirs are returned so the caller can pick the one that
    actually holds the relevant content; stopping at the first existing dir would
    always return the package dir and never reach the user cache, which is where
    read-only wheel installs store compiled modules.
    """
    override = os.environ.get("FORGE_AITER_WARM_JIT_DIR", "").strip()
    if override:
        candidate = Path(override).expanduser()
        return [candidate] if candidate.is_dir() else []
    candidates: list[Path] = []
    try:
        import importlib.util

        spec = importlib.util.find_spec("aiter")
    except (ImportError, ValueError):
        spec = None
    if spec is not None and spec.origin:
        candidates.append(Path(spec.origin).parent / "jit")
    candidates.append(Path.home() / ".aiter" / "jit")
    return [c for c in candidates if c.is_dir()]


def _seed_flydsl_cache(target: Path) -> None:
    """Symlink precompiled FlyDSL artefacts from the global cache into *target*."""
    target.mkdir(parents=True, exist_ok=True)
    source = next(
        (d / "flydsl_cache" for d in _global_aiter_jit_dirs() if (d / "flydsl_cache").is_dir()),
        None,
    )
    if source is None:
        return
    try:
        hash_dirs = sorted(source.iterdir())
    except OSError:
        return
    for entry in hash_dirs:
        try:
            if not entry.is_dir():
                continue
        except OSError:
            continue
        destination = target / entry.name
        if destination.exists() or destination.is_symlink():
            continue
        try:
            os.symlink(entry.resolve(), destination)
        except OSError as error:
            log.warning("aiter-cache: could not link FlyDSL cache %s: %s", entry.name, error)


def seed_prebuilt_modules(jit_dir: Path) -> dict[str, Any]:
    """Symlink the package's prebuilt AITER modules into a fresh BASELINE shard.

    aiter's ``get_module`` imports a module by name from ``AITER_JIT_DIR`` and
    never validates the ``.so`` against source content (aiter/jit/core.py:
    ``importlib.import_module(md_name)``). An empty isolated shard therefore
    cold-compiles the full CK instance-factory TU (measured >26 min, gfx950)
    on first use, which blows the preflight timeout.

    For the baseline task-preparation preflight the kernel source is pristine —
    byte-identical to what the shipped ``.so`` were built from — so pointing the
    shard at those prebuilt modules is correct AND skips the compile entirely.

    NEVER call this for an edited-source shard: aiter would import the stale
    ``.so`` in place of the edit and silently measure the wrong kernel. Callers
    must invoke this only on the pristine baseline shard (different edits get a
    fresh content-keyed shard dir and compile normally).
    """
    stats: dict[str, Any] = {"seeded": 0, "skipped": 0, "src": "", "errors": 0}
    global_dirs = _global_aiter_jit_dirs()
    global_dir = next((d for d in global_dirs if any(d.glob("*.so"))), None) or (
        global_dirs[0] if global_dirs else None
    )
    if global_dir is None:
        return stats
    stats["src"] = str(global_dir)
    jit_dir = Path(jit_dir)
    try:
        jit_dir.mkdir(parents=True, exist_ok=True)
        candidates = sorted(global_dir.glob("*.so"))
    except OSError:
        stats["errors"] += 1
        return stats
    for so in candidates:
        dest = jit_dir / so.name
        if dest.is_symlink() and not dest.exists():
            # Dangling symlink (target vanished): exists()==False but is_symlink()
            # ==True, so without this it would be treated as "already present" and
            # skipped forever, leaving aiter with a broken link that imports
            # nothing. Remove it so we re-link to the current prebuilt .so below.
            with contextlib.suppress(OSError):
                dest.unlink()
        if dest.exists() or dest.is_symlink():
            stats["skipped"] += 1
            continue
        try:
            os.symlink(so.resolve(), dest)
            stats["seeded"] += 1
        except OSError:
            stats["errors"] += 1
    if stats["seeded"] == 0:
        # Zero seeds means the baseline preflight will cold-compile the full CK
        # instance-factory TU (>26 min, gfx950) and blow its timeout. Surface it
        # loudly rather than let the silent slow path look like a hang: the warm
        # dir was empty/missing or FORGE_AITER_WARM_JIT_DIR points somewhere
        # without prebuilt modules.
        log.warning(
            "seed_prebuilt_modules: seeded 0 modules from %s (skipped=%d, "
            "errors=%d); baseline preflight will cold-compile and may time out. "
            "Check FORGE_AITER_WARM_JIT_DIR / the aiter package jit dir.",
            stats["src"] or "<no source dir>",
            stats["skipped"],
            stats["errors"],
        )
    return stats


def _live_cache_users(isolation: AiterCacheIsolation) -> list[int] | None:
    """Return other processes inheriting this cache, or None if uncertain."""
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return None
    expected_root = str(isolation.aiter_root_dir).encode()
    expected_jit = str(isolation.aiter_jit_dir).encode()
    live: list[int] = []
    uncertain = False
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == os.getpid():
            continue
        try:
            environ = (entry / "environ").read_bytes().split(b"\0")
        except FileNotFoundError:
            continue
        except (OSError, PermissionError):
            uncertain = True
            continue
        if b"AITER_ROOT_DIR=" + expected_root in environ or b"AITER_JIT_DIR=" + expected_jit in environ:
            live.append(pid)
    if live:
        return live
    return None if uncertain else []


def cleanup_owned_aiter_locks(isolation: AiterCacheIsolation) -> dict[str, Any]:
    """Delete only orphaned locks in the cache owned by this Forge process."""
    stats: dict[str, Any] = {
        "cache_root": str(isolation.cache_root),
        "scanned": 0,
        "deleted": 0,
        "errors": 0,
        "skipped_live_pids": [],
        "owner_verified": False,
    }
    try:
        owner = json.loads(isolation.owner_file.read_text(encoding="utf-8"))
        stats["owner_verified"] = int(owner.get("owner_pid", -1)) == isolation.owner_pid
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return stats
    if not stats["owner_verified"]:
        return stats

    live_users = _live_cache_users(isolation)
    if live_users is None:
        stats["errors"] += 1
        return stats
    if live_users:
        stats["skipped_live_pids"] = live_users
        return stats

    for root in (isolation.aiter_root_dir / "build", isolation.aiter_jit_dir / "build"):
        if not root.is_dir():
            continue
        try:
            candidates = list(root.rglob("*"))
        except OSError:
            stats["errors"] += 1
            continue
        for path in candidates:
            if not path.is_file() or not (path.name in _LOCK_NAMES or path.name.startswith("lock_")):
                continue
            stats["scanned"] += 1
            try:
                path.unlink()
                stats["deleted"] += 1
            except OSError:
                stats["errors"] += 1
    return stats


def cleanup_owned_aiter_cache(isolation: AiterCacheIsolation) -> dict[str, Any]:
    """Delete one finished attempt's private cache when no child still uses it."""
    stats: dict[str, Any] = {
        "cache_root": str(isolation.cache_root),
        "deleted": False,
        "errors": 0,
        "skipped_live_pids": [],
        "owner_verified": False,
    }
    try:
        owner = json.loads(isolation.owner_file.read_text(encoding="utf-8"))
        stats["owner_verified"] = int(owner.get("owner_pid", -1)) == isolation.owner_pid
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return stats
    if not stats["owner_verified"]:
        return stats

    candidates = [isolation]
    sources_root = isolation.cache_root / "sources"
    if sources_root.is_dir():
        try:
            candidates.extend(
                _source_shard_isolation(path, isolation.owner_pid) for path in sources_root.iterdir() if path.is_dir()
            )
        except OSError:
            stats["errors"] += 1
            return stats
    for candidate in candidates:
        live_users = _live_cache_users(candidate)
        if live_users is None:
            stats["errors"] += 1
            return stats
        if live_users:
            stats["skipped_live_pids"].extend(live_users)
    if stats["skipped_live_pids"]:
        stats["skipped_live_pids"] = sorted(set(stats["skipped_live_pids"]))
        return stats

    try:
        shutil.rmtree(isolation.cache_root)
        stats["deleted"] = True
        _CACHE_POLICIES.pop(str(isolation.cache_root.resolve()), None)
        prefix = str(isolation.cache_root.resolve()) + os.sep
        for key in ("AITER_ROOT_DIR", "AITER_JIT_DIR", "FLYDSL_RUNTIME_CACHE_DIR"):
            value = os.environ.get(key, "")
            if value.startswith(prefix):
                os.environ.pop(key, None)
        if os.environ.get("FORGE_AITER_CACHE_ROOT") == str(isolation.cache_root.resolve()):
            os.environ.pop("FORGE_AITER_CACHE_ROOT", None)
            os.environ.pop("FORGE_AITER_CACHE_OWNER_PID", None)
    except OSError:
        stats["errors"] += 1
    return stats


def cleanup_current_aiter_cache() -> dict[str, Any] | None:
    """Delete the current Forge attempt's private AITER cache, if configured."""
    cache_root_raw = os.environ.get("FORGE_AITER_CACHE_ROOT", "").strip()
    owner_pid_raw = os.environ.get("FORGE_AITER_CACHE_OWNER_PID", "").strip()
    try:
        owner_pid = int(owner_pid_raw)
    except ValueError:
        return None
    if owner_pid != os.getpid() or not cache_root_raw:
        return None
    cache_root = Path(cache_root_raw).resolve()
    return cleanup_owned_aiter_cache(
        AiterCacheIsolation(
            cache_root=cache_root,
            aiter_root_dir=cache_root / "cpp_itfs",
            aiter_jit_dir=cache_root / "jit",
            flydsl_cache_dir=cache_root / "flydsl_cache",
            owner_file=cache_root / _OWNER_FILE,
            owner_pid=owner_pid,
        )
    )


def cleanup_current_owned_aiter_locks() -> dict[str, Any] | None:
    """Clean the current Forge cache after a child timeout, if configured."""
    owner_pid_raw = os.environ.get("FORGE_AITER_CACHE_OWNER_PID", "").strip()
    root_raw = os.environ.get("AITER_ROOT_DIR", "").strip()
    jit_raw = os.environ.get("AITER_JIT_DIR", "").strip()
    try:
        owner_pid = int(owner_pid_raw)
    except ValueError:
        return None
    if owner_pid != os.getpid() or not root_raw or not jit_raw:
        return None
    root = Path(root_raw).resolve()
    jit = Path(jit_raw).resolve()
    if root.parent != jit.parent:
        return None
    isolation = AiterCacheIsolation(
        cache_root=root.parent,
        aiter_root_dir=root,
        aiter_jit_dir=jit,
        flydsl_cache_dir=root.parent / "flydsl_cache",
        owner_file=root.parent / _OWNER_FILE,
        owner_pid=owner_pid,
    )
    return cleanup_owned_aiter_locks(isolation)


__all__ = [
    "AiterCachePolicy",
    "AiterCacheIsolation",
    "DEFAULT_AITER_CACHE_MAX_BYTES",
    "DEFAULT_AITER_CACHE_TARGET_BYTES",
    "child_cache_environment",
    "cleanup_current_aiter_cache",
    "cleanup_current_owned_aiter_locks",
    "cleanup_owned_aiter_cache",
    "cleanup_owned_aiter_locks",
    "configure_aiter_cache_isolation",
    "activate_aiter_cache_for_sources",
    "seed_prebuilt_modules",
    "prune_aiter_cache_shards",
]
