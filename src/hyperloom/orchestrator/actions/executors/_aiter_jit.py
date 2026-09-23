# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared aiter build-cache helpers and stale-lock cleanup."""

from __future__ import annotations

import csv
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from hyperloom.common.aiter_jit_cache import invalidate_jit_cache, resolve_package_root, resolve_serving_context

log = logging.getLogger(__name__)


# < N .so files under aiter jit/ ⇒ COLD start (first-time JIT compile pending).
COLD_START_KERNEL_THRESHOLD = 20
# COLD-start benchmark cap, including first-time compilation and graph capture.
BASELINE_COLD_START_TIMEOUT_SEC = 9000

# Fallback probe paths for aiter's JIT cache dir; first existing wins.
AITER_JIT_PROBE_PATHS: tuple[str, ...] = (
    "/sgl-workspace/aiter/aiter/jit",
    "/sgl-workspace/aiter/aiter/jit/build",
    "/usr/local/lib/python3.10/dist-packages/aiter/jit",
    "/usr/local/lib/python3.12/dist-packages/aiter/jit",
    "/usr/local/lib/python3.10/site-packages/aiter/jit",
    "/usr/local/lib/python3.12/site-packages/aiter/jit",
    "/opt/venv/lib/python3.10/site-packages/aiter/jit",
    "/opt/venv/lib/python3.12/site-packages/aiter/jit",
)

# Fallback locations for cpp_itfs template builds.
AITER_CPP_BUILD_PROBE_PATHS: tuple[str, ...] = ("/root/.aiter/build",)

# Mtime gate (minutes) for the lock sweep.
AITER_LOCK_STALE_MINUTES = 5

# Process names that indicate an in-flight aiter/ninja compile. hipcc is a wrapper whose ``name`` can surface as
# ``perl``/``sh``, so we also match on the cmdline's first token (see ``_any_live_compiler``).
COMPILER_PROCESS_NAMES = frozenset(
    {
        "hipcc",
        "hipcc.bin",
        "ninja",
        "cc1plus",
        "clang",
        "clang++",
        "clang-cpp",
    }
)

# Lock file names left by aiter / ninja under the jit dir.
_LOCK_NAMES = {"lock", ".ninja_lock"}
_BATON_WAIT_MARKER = "waiting for baton release at"
_BATON_LOG_NAMES = {"server.log", "benchmark_stderr.log", "benchmark_stdout.log"}


def _resolve_aiter_jit_dir_dynamic() -> list[str]:
    """Locate aiter's ``jit/`` dir via Python's import machinery."""
    aiter_root = resolve_package_root()
    if aiter_root is None:
        return []
    return [
        str(aiter_root / "jit"),
        str(aiter_root / "jit" / "build"),
    ]


def probe_aiter_jit_cache() -> dict[str, Any]:
    """Inspect aiter's JIT cache and classify the next start as cold or warm."""
    info: dict[str, Any] = {
        "path": None,
        "kernel_count": 0,
        "size_mb": 0,
        "is_cold": None,
        "probe_status": "not_found",
    }
    candidates: list[str] = []
    override = os.environ.get("INFERENCE_OPTIMIZER_AITER_JIT_DIR", "").strip()
    if override:
        candidates.append(override)
    candidates.extend(_resolve_aiter_jit_dir_dynamic())
    candidates.extend(AITER_JIT_PROBE_PATHS)

    try:
        chosen: Path | None = None
        for raw in candidates:
            path = Path(raw)
            if path.exists() and path.is_dir():
                chosen = path
                break
        if chosen is None:
            return info
        info["path"] = str(chosen)

        total_bytes = 0
        kernel_count = 0
        for so_path in chosen.rglob("*.so"):
            try:
                total_bytes += so_path.stat().st_size
                kernel_count += 1
            except OSError:
                continue
        info["kernel_count"] = kernel_count
        info["size_mb"] = total_bytes // (1024 * 1024)
        info["is_cold"] = kernel_count < COLD_START_KERNEL_THRESHOLD
        info["probe_status"] = "found"
        return info
    except Exception as exc:  # noqa: BLE001
        log.warning("aiter_jit: cache probe failed: %s", exc)
        info["probe_status"] = "error"
        info["is_cold"] = None
        return info


def _any_live_compiler(
    build_dirs: list[Path] | None = None,
) -> bool | None:
    """Return True if a compiler associated with the build trees is alive."""
    try:
        import psutil
    except ImportError:
        return None
    try:
        normalized_dirs = [str(path.resolve()) for path in (build_dirs or [])]
        unknown = False
        for proc in psutil.process_iter(["name", "cmdline", "cwd", "status"]):
            try:
                info = proc.info
                if info.get("status") == psutil.STATUS_ZOMBIE:
                    continue
                name = (info.get("name") or "").strip()
                cmdline = info.get("cmdline") or []
                is_compiler = name in COMPILER_PROCESS_NAMES
                if cmdline:
                    first = os.path.basename(str(cmdline[0]).strip())
                    if first in COMPILER_PROCESS_NAMES:
                        is_compiler = True
                if not is_compiler:
                    if info.get("name") is None or info.get("cmdline") is None:
                        unknown = True
                    continue
                if not normalized_dirs:
                    return True
                cwd = str(info.get("cwd") or "")
                command = "\0".join(str(arg) for arg in cmdline)
                if any(
                    cwd == build_dir or cwd.startswith(f"{build_dir}{os.sep}") or build_dir in command
                    for build_dir in normalized_dirs
                ):
                    return True
                if info.get("cwd") is None or info.get("cmdline") is None:
                    unknown = True
            except psutil.AccessDenied:
                unknown = True
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
    except Exception as exc:  # noqa: BLE001 — enumeration failed entirely
        log.warning("aiter_jit: compiler-liveness scan failed: %s", exc)
        return None
    return None if unknown else False


def _dedupe_existing_dirs(candidates: list[Path], unreadable: list[str] | None = None) -> list[Path]:
    """Return existing candidate directories once, preserving priority."""
    resolved: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            normalized = candidate.expanduser().resolve()
        except OSError:
            normalized = candidate.expanduser().absolute()
        key = str(normalized)
        if key in seen:
            continue
        try:
            if not normalized.is_dir():
                continue
        except OSError as exc:
            # is_dir() re-raises EACCES (not in pathlib's ignored errnos), so a fallback under an unreadable root is
            # skipped rather than fatal.
            log.warning("aiter lock sweep: cannot inspect %s (%s); skipping", key, exc)
            if unreadable is not None:
                unreadable.append(key)
            continue
        seen.add(key)
        resolved.append(normalized)
    return resolved


def _resolve_lock_sweep_dirs(aiter_jit_dir: Path | None, unreadable: list[str] | None = None) -> list[Path]:
    """Resolve every active aiter build tree that may contain baton locks."""
    if aiter_jit_dir is not None:
        return [aiter_jit_dir]

    candidates: list[Path] = []
    aiter_root = os.environ.get("AITER_ROOT_DIR", "").strip()
    if aiter_root:
        candidates.append(Path(aiter_root) / "build")
    else:
        home = os.environ.get("HOME", "").strip()
        if home:
            candidates.append(Path(home) / ".aiter" / "build")
        candidates.extend(Path(path) for path in AITER_CPP_BUILD_PROBE_PATHS)

    aiter_jit_override = os.environ.get("AITER_JIT_DIR", "").strip()
    if aiter_jit_override:
        override_path = Path(aiter_jit_override)
        candidates.extend([override_path / "build", override_path])
    if aiter_root and aiter_jit_override:
        # Forge sets both variables for a private attempt.
        return _dedupe_existing_dirs(candidates, unreadable)
    override = os.environ.get("INFERENCE_OPTIMIZER_AITER_JIT_DIR", "").strip()
    if override:
        override_path = Path(override)
        # Preserve the legacy explicit-override contract: callers use this variable to constrain a diagnostic/test
        # sweep to one tree.
        return _dedupe_existing_dirs([override_path / "build", override_path], unreadable)
    package_root = resolve_package_root()
    if package_root is not None:
        candidates.append(package_root / "jit" / "build")
    candidates.extend(
        Path(path)
        for path in (
            "/sgl-workspace/aiter/aiter/jit/build",
            "/usr/local/lib/python3.10/dist-packages/aiter/jit/build",
            "/usr/local/lib/python3.12/dist-packages/aiter/jit/build",
            "/opt/venv/lib/python3.10/site-packages/aiter/jit/build",
            "/opt/venv/lib/python3.12/site-packages/aiter/jit/build",
        )
    )
    return _dedupe_existing_dirs(candidates, unreadable)


def _resolve_lock_sweep_dir(aiter_jit_dir: Path | None) -> Path | None:
    """Compatibility wrapper returning the first resolved build tree."""
    if aiter_jit_dir is None:
        override = os.environ.get("INFERENCE_OPTIMIZER_AITER_JIT_DIR", "").strip()
        if override:
            override_path = Path(override)
            preferred = _dedupe_existing_dirs([override_path / "build", override_path])
            if preferred:
                return preferred[0]
    resolved = _resolve_lock_sweep_dirs(aiter_jit_dir)
    return resolved[0] if resolved else None


def clean_stale_aiter_locks(
    aiter_jit_dir: Path | None = None,
    stale_minutes: int = AITER_LOCK_STALE_MINUTES,
) -> dict[str, Any]:
    """Sweep aiter's JIT build trees for stale plain-file locks left by killed runs."""
    stats: dict[str, Any] = {
        "dir": None,
        "dirs": [],
        "scanned": 0,
        "deleted": 0,
        "skipped_fresh": 0,
        "errors": 0,
    }

    unreadable: list[str] = []
    resolved_dirs = _resolve_lock_sweep_dirs(aiter_jit_dir, unreadable)
    stats["errors"] += len(unreadable)
    stats["unreadable"] = unreadable
    if not resolved_dirs:
        return stats

    primary = _resolve_lock_sweep_dir(None) if aiter_jit_dir is None else resolved_dirs[0]
    stats["dir"] = str(primary or resolved_dirs[0])
    stats["dirs"] = [str(path) for path in resolved_dirs]

    threshold_seconds = float(stale_minutes) * 60.0
    now = time.time()
    for resolved in resolved_dirs:
        try:
            walker = os.walk(str(resolved))
            for root, _dirs, files in walker:
                for fname in files:
                    if not (fname in _LOCK_NAMES or fname.startswith("lock_")):
                        continue
                    stats["scanned"] += 1
                    fpath = Path(root) / fname
                    try:
                        age = now - fpath.stat().st_mtime
                    except OSError:
                        stats["errors"] += 1
                        continue
                    if age < threshold_seconds:
                        stats["skipped_fresh"] += 1
                        continue
                    try:
                        fpath.unlink()
                        stats["deleted"] += 1
                    except OSError:
                        stats["errors"] += 1
        except OSError:
            stats["errors"] += 1

    return stats


def sweep_stale_aiter_locks_if_dead(
    aiter_jit_dir: Path | None = None,
) -> dict[str, Any]:
    """Sweep orphaned aiter JIT locks, gated on no live compiler process."""
    resolved_dirs = _resolve_lock_sweep_dirs(aiter_jit_dir)
    alive = _any_live_compiler(resolved_dirs)
    if alive is not False:
        if alive is None:
            log.warning("aiter lock sweep skipped: compiler liveness is unknown")
        return {
            "dir": None,
            "dirs": [],
            "scanned": 0,
            "deleted": 0,
            "skipped_fresh": 0,
            # An unverified sweep must not look like a successfully empty tree.
            "errors": int(alive is None),
            "compiler_alive": alive,
            "skipped_live": alive is True,
        }
    stats = clean_stale_aiter_locks(
        aiter_jit_dir,
        stale_minutes=AITER_LOCK_STALE_MINUTES,
    )
    stats["compiler_alive"] = False
    return stats


def find_aiter_baton_wait(
    search_root: Path,
    *,
    since_unix: float | None = None,
    max_files: int = 20,
    tail_bytes: int = 256 * 1024,
) -> dict[str, str] | None:
    """Find bounded log evidence of an aiter process blocked on a FileBaton."""
    try:
        candidates = [path for path in search_root.rglob("*") if path.is_file() and path.name in _BATON_LOG_NAMES]
    except OSError:
        return None

    def _mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    candidates.sort(key=_mtime, reverse=True)
    if since_unix is not None:
        candidates = [path for path in candidates if _mtime(path) >= since_unix]
    marker_lower = _BATON_WAIT_MARKER.lower()
    for path in candidates[:max_files]:
        try:
            with path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - tail_bytes))
                text = handle.read().decode(errors="replace")
        except OSError:
            continue
        marker_index = text.lower().rfind(marker_lower)
        if marker_index < 0:
            continue
        line_end = text.find("\n", marker_index)
        excerpt_end = len(text) if line_end < 0 else min(len(text), line_end + 512)
        return {
            "log_path": str(path),
            "excerpt": text[marker_index:excerpt_end].strip(),
        }
    return None


COMPILED_REGISTRY_MARKER = "not present in the compiled registry"

# kernelName values with these libtypes are not linked into serving ``module_*.so``.
NON_JIT_SO_LIBTYPES = frozenset({"asm", "triton", "gluon", "opus", "flydsl"})

# Longest prefix first: a bpreshuffle CSV may carry ``libtype=cktile`` rows whose
# kernelName is ``a8w8_blockscale_cktile_*`` (no bpreshuffle infix).
_KERNEL_PREFIX_TO_MODULE: tuple[tuple[str, dict[str, str]], ...] = (
    (
        "a8w8_blockscale_bpreshuffle",
        {
            "ck": "module_gemm_a8w8_blockscale_bpreshuffle",
            "cktile": "module_gemm_a8w8_blockscale_bpreshuffle_cktile",
            "asm": "module_gemm_a8w8_blockscale_bpreshuffle_asm",
        },
    ),
    (
        "a8w8_blockscale",
        {
            "ck": "module_gemm_a8w8_blockscale",
            "cktile": "module_gemm_a8w8_blockscale_cktile",
            "asm": "module_gemm_a8w8_blockscale_asm",
        },
    ),
    (
        "a8w8_bpreshuffle",
        {
            "ck": "module_gemm_a8w8_bpreshuffle",
            "cktile": "module_gemm_a8w8_bpreshuffle_cktile",
        },
    ),
    ("a8w8", {"ck": "module_gemm_a8w8"}),
    ("a4w4_blockscale", {"ck": "module_gemm_a4w4_blockscale"}),
    ("a4w4", {"ck": "module_gemm_a4w4_blockscale"}),
)

# Serving modules whose codegen reads the matching AITER_CONFIG_* tune file.
AITER_ENV_TO_SERVING_MODULES: dict[str, tuple[str, ...]] = {
    "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE": (
        "module_gemm_a8w8_blockscale_bpreshuffle",
        "module_gemm_a8w8_blockscale_bpreshuffle_cktile",
        "module_gemm_a8w8_blockscale_bpreshuffle_asm",
    ),
    "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE": (
        "module_gemm_a8w8_blockscale",
        "module_gemm_a8w8_blockscale_cktile",
        "module_gemm_a8w8_blockscale_asm",
    ),
    "AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE": (
        "module_gemm_a8w8_bpreshuffle",
        "module_gemm_a8w8_bpreshuffle_cktile",
    ),
    "AITER_CONFIG_GEMM_A8W8": ("module_gemm_a8w8",),
    "AITER_CONFIG_GEMM_A4W4": ("module_gemm_a4w4_blockscale",),
}

# The ``tuned_file_name`` aiter resolves each variable under, which is both the shipped
# table's stem and the glob it discovers model overlays with.
AITER_ENV_TO_TUNED_FILE: dict[str, str] = {
    "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE": "a8w8_blockscale_bpreshuffle_tuned_gemm",
    "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE": "a8w8_blockscale_tuned_gemm",
    "AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE": "a8w8_bpreshuffle_tuned_gemm",
    "AITER_CONFIG_GEMM_A8W8": "a8w8_tuned_gemm",
    "AITER_CONFIG_GEMM_A4W4": "a4w4_blockscale_tuned_gemm",
}


def is_aiter_jit_registry_mismatch(*texts: str) -> bool:
    """True when logs show a tuned CSV kernel name missing from the compiled .so."""
    blob = "\n".join(t for t in texts if t).lower()
    return COMPILED_REGISTRY_MARKER in blob


#: How aiter names the kernel it could not find: ``kernel '<name>' is not present``.
_MISSING_KERNEL_RE = re.compile(r"kernel '([^']+)' is not present", re.IGNORECASE)


def registry_mismatch_modules(*texts: str) -> tuple[str, ...]:
    """Serving modules that own the kernels a registry mismatch named.

    The env a round carries does not always reach the module at fault: a round tuning
    one CSV still boots against every CSV aiter merges, so the missing kernel can
    belong to a variable the round never set -- and to one absent from
    :data:`AITER_ENV_TO_SERVING_MODULES` entirely, which leaves the env-keyed drop
    with no module to invalidate and the retry certain to fail the same way. The error text
    names the kernel, and the kernel names its module.

    Args:
        texts: Error strings or log excerpts from the failed round.

    Returns:
        Module stems to invalidate, deduplicated, empty when no kernel was named.
    """
    blob = "\n".join(t for t in texts if t)
    modules: list[str] = []
    for kernel in _MISSING_KERNEL_RE.findall(blob):
        # libtype is not in the message; the prefix table answers for ck, and a
        # cktile kernel carries it in its own name.
        resolved = serving_module_for_kernel(kernel, "cktile" if "cktile" in kernel else "ck")
        if resolved:
            modules.append(resolved)
    return tuple(dict.fromkeys(modules))


def serving_module_for_kernel(kernel_name: str, libtype: str) -> str | None:
    """Serving ``module_*.so`` that should contain this JIT kernel name."""
    lib = str(libtype or "").strip().lower() or "ck"
    for prefix, by_lib in _KERNEL_PREFIX_TO_MODULE:
        if kernel_name.startswith(prefix):
            return by_lib.get(lib)
    return None


def csv_jit_kernel_rows(csv_path: Path) -> list[tuple[str, str]]:
    """``(kernelName, libtype)`` rows that are linked into a serving ``module_*.so``.

    The tables this reads ship with aiter, so their encoding and field sizes are not
    ours to assume: a BOM raises ``UnicodeDecodeError`` and an overlong field raises
    ``csv.Error``, and either escaping here would skip the whole coverage check.
    """
    rows: list[tuple[str, str]] = []
    try:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                name = str(row.get("kernelName") or "").strip()
                if not name:
                    continue
                libtype = str(row.get("libtype") or "").strip().lower()
                if libtype in NON_JIT_SO_LIBTYPES:
                    continue
                if not libtype and name.startswith("_ZN"):
                    continue
                rows.append((name, libtype))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        log.warning("aiter_jit: cannot read tuned CSV %s (%s); treating it as covered", csv_path, exc)
        return []
    return rows


def serving_modules_cover_csv(jit_dir: Path, modules: tuple[str, ...], csv_path: Path) -> bool:
    """True when every JIT ``kernelName`` in the CSV appears in its serving .so.

    ``libtype=asm`` (and other non-JIT backends) names are not linked into
    ``module_*.so``. A name is looked up in the module implied by its prefix and
    ``libtype``, falling back to ``modules``. Missing .so files mean the next start
    will compile from the current CSV, so that case is treated as covered.
    """
    rows = csv_jit_kernel_rows(csv_path)
    if not rows:
        return True
    blobs: dict[str, bytes | None] = {}

    def _read(module: str) -> bytes | None:
        if module not in blobs:
            so_path = jit_dir / f"{module}.so"
            data: bytes | None = None
            if so_path.is_file():
                try:
                    data = so_path.read_bytes()
                except OSError:
                    data = None
            blobs[module] = data
        return blobs[module]

    for name, libtype in rows:
        resolved = serving_module_for_kernel(name, libtype)
        search = (resolved,) if resolved else modules
        encoded = name.encode("utf-8")
        saw_so = False
        found = False
        for module in dict.fromkeys(search):
            data = _read(module)
            if data is None:
                continue
            saw_so = True
            if encoded in data:
                found = True
                break
        if saw_so and not found:
            return False
    return True


def _invalidate_jit_build(jit_build: Path, backup_dir: Path, modules: tuple[str, ...]) -> dict[str, Any]:
    """Invalidate selected modules and build state, surfacing transaction failure."""
    record = invalidate_jit_cache(jit_build, backup_dir, modules=modules)
    if record["status"] == "failed":
        raise OSError(record["error"])
    return record


def _modules_for_envs(envs: dict[str, str] | None) -> tuple[str, ...]:
    if not envs:
        modules: list[str] = []
        for names in AITER_ENV_TO_SERVING_MODULES.values():
            modules.extend(names)
        return tuple(dict.fromkeys(modules))
    modules: list[str] = []
    for key in envs:
        modules.extend(AITER_ENV_TO_SERVING_MODULES.get(str(key), ()))
    return tuple(dict.fromkeys(modules))


def csvs_aiter_will_load(configs_dir: Path, tuned_file_name: str, value: str) -> list[Path]:
    """The CSV set this boot resolves to, by aiter's own two-branch rule.

    ``jit/core.py::get_config_file`` either takes the env var literally or, when it is
    unset, discovers per-model overlays and merges them on top of the shipped default.
    A coverage check that only looks at what a round names therefore misses the overlay
    an *unset* variable pulls in -- which is how ``fmoe_ck``, tuning only
    ``AITER_CONFIG_FMOE``, booted against the dsv3 bpreshuffle overlay and failed on a
    kernel it never tuned.

    Args:
        configs_dir: aiter's ``configs/`` directory.
        tuned_file_name: The table's stem, e.g. ``a8w8_blockscale_bpreshuffle_tuned_gemm``.
        value: The env var's value, empty when unset.

    Returns:
        The CSV paths this boot will read, shipped default first when it applies.
    """
    if value.strip():
        # Set: precisely the ``:``-joined paths. The shipped default is not prepended and
        # model overlays are not discovered.
        return [Path(p) for p in value.split(":") if p.strip()]
    overlays = sorted(
        p for p in (configs_dir / "model_configs").glob(f"*{tuned_file_name}*.csv") if "untuned" not in p.name
    )
    return [configs_dir / f"{tuned_file_name}.csv", *overlays]


def prepare_serving_so_for_csvs(
    envs: dict[str, str],
    *,
    backup_dir: Path | None = None,
) -> dict[str, Any]:
    """Skip when serving .so already covers the CSVs this boot loads; otherwise re-JIT.

    Args:
        envs: The round's ``AITER_CONFIG_*`` values; a variable it does not name is
            checked on aiter's unset branch, which is where the model overlays come in.
        backup_dir: Where to back up selected modules and ``jit/build`` for invalidation.

    Returns:
        A status dict with ``action`` of ``skip``, ``invalidate``, or ``noop``.
    """
    context = resolve_serving_context(
        os.environ.get("INFERENCE_OPTIMIZER_AITER_JIT_DIR"), jit_probe_paths=AITER_JIT_PROBE_PATHS
    )
    if context is None:
        return {"action": "noop", "reason": "aiter jit dir not found"}
    package_root, jit_build = context
    jit_dir = jit_build.parent
    configs_dir = package_root / "configs"
    modules_needed: list[str] = []
    for env_var, modules in AITER_ENV_TO_SERVING_MODULES.items():
        tuned_file_name = AITER_ENV_TO_TUNED_FILE.get(env_var, "")
        if not tuned_file_name:
            continue
        for csv_path in csvs_aiter_will_load(configs_dir, tuned_file_name, str(envs.get(env_var) or "")):
            if not csv_path.is_file():
                continue
            if serving_modules_cover_csv(jit_dir, modules, csv_path):
                continue
            modules_needed.extend(modules)
            for name, libtype in csv_jit_kernel_rows(csv_path):
                resolved = serving_module_for_kernel(name, libtype)
                if resolved:
                    modules_needed.append(resolved)
    modules_needed_t = tuple(dict.fromkeys(modules_needed))
    if not modules_needed_t:
        return {"action": "skip", "jit_dir": str(jit_dir)}
    dest = backup_dir or (jit_dir / "hyperloom_jit_backup")
    invalidation = _invalidate_jit_build(jit_build, dest, modules_needed_t)
    removed = [str(Path(invalidation["src"]).parent / name) for name in invalidation["module_names"]]
    log.info(
        "aiter serving so does not cover tuned CSV; invalidated %d module(s) jit_build=%s",
        len(removed),
        invalidation.get("status"),
    )
    return {
        "action": "invalidate",
        "jit_dir": str(jit_dir),
        "removed": removed,
        "jit_build": invalidation,
    }


def drop_serving_so_for_envs(
    envs: dict[str, str] | None = None,
    *,
    backup_dir: Path | None = None,
    also_modules: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Back up serving GEMM modules and ``jit/build`` so a later start rebuilds.

    Args:
        envs: ``AITER_CONFIG_*`` the round carried; ``None`` or empty means every known module.
        backup_dir: Where to park the invalidated cache.
        also_modules: Modules to invalidate on top of the env's own, for a mismatch whose
            kernel belongs to a variable this round never set. Without them an env
            that maps to no module invalidates none and the retry repeats the failure.
    """
    context = resolve_serving_context(
        os.environ.get("INFERENCE_OPTIMIZER_AITER_JIT_DIR"), jit_probe_paths=AITER_JIT_PROBE_PATHS
    )
    if context is None:
        return {"action": "noop", "reason": "aiter jit dir not found"}
    _, jit_build = context
    jit_dir = jit_build.parent
    modules = tuple(dict.fromkeys((*_modules_for_envs(envs), *also_modules)))
    dest = backup_dir or (jit_dir / "hyperloom_jit_backup")
    invalidation = _invalidate_jit_build(jit_build, dest, modules)
    removed = [str(Path(invalidation["src"]).parent / name) for name in invalidation["module_names"]]
    return {
        "action": "invalidate",
        "jit_dir": str(jit_dir),
        "removed": removed,
        "jit_build": invalidation,
    }


def result_is_aiter_jit_registry_mismatch(result: dict[str, Any] | None) -> bool:
    """True when an integrate/baseline result is a compiled-registry miss."""
    if not isinstance(result, dict):
        return False
    if str(result.get("error_class") or "") == "aiter_jit_registry_mismatch":
        return True
    return is_aiter_jit_registry_mismatch(str(result.get("error") or ""))


__all__ = [
    "AITER_CPP_BUILD_PROBE_PATHS",
    "AITER_ENV_TO_SERVING_MODULES",
    "AITER_ENV_TO_TUNED_FILE",
    "AITER_JIT_PROBE_PATHS",
    "AITER_LOCK_STALE_MINUTES",
    "BASELINE_COLD_START_TIMEOUT_SEC",
    "COLD_START_KERNEL_THRESHOLD",
    "COMPILER_PROCESS_NAMES",
    "COMPILED_REGISTRY_MARKER",
    "NON_JIT_SO_LIBTYPES",
    "clean_stale_aiter_locks",
    "csv_jit_kernel_rows",
    "csvs_aiter_will_load",
    "drop_serving_so_for_envs",
    "find_aiter_baton_wait",
    "is_aiter_jit_registry_mismatch",
    "prepare_serving_so_for_csvs",
    "probe_aiter_jit_cache",
    "registry_mismatch_modules",
    "result_is_aiter_jit_registry_mismatch",
    "serving_module_for_kernel",
    "serving_modules_cover_csv",
    "sweep_stale_aiter_locks_if_dead",
    "_any_live_compiler",
    "_resolve_aiter_jit_dir_dynamic",
    "_resolve_lock_sweep_dirs",
]
