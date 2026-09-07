# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Framework source-root resolution and path containment."""

from __future__ import annotations

import importlib.util
import logging
import os
import re
import site
import sys
import sysconfig
from collections.abc import Sequence
from pathlib import Path

log = logging.getLogger(__name__)

#: Framework-agnostic way to name the source tree a session may patch. Accepted in
#: addition to ``<FRAMEWORK>_REPO_PATH`` / ``<FRAMEWORK>_DIR``, which keep
#: precedence; see :func:`_discover_explicit_framework_root`.
GENERIC_FRAMEWORK_ROOT_ENV: str = "FRAMEWORK_REPO_PATH"

_DEFAULT_SOURCE_ROOTS: tuple[str, ...] = (
    "/sgl-workspace/aiter/",
    "/sgl-workspace/sglang/",
    "/sgl-workspace/vllm/",
    # atom's editable-install layout.
    "/app/ATOM/atom/",
    # xDiT editable install (pure-Python).
    "/app/xDiT/",
)

#: Every package whose installed tree counts as framework source. The single
#: authoritative list: importlib discovery, the ``$VIRTUAL_ENV`` glob and the
#: install-parent glob all derive their patterns from it, so a package added
#: here reaches all three. Naming one in only some of them is how a standalone
#: ``sgl_kernel`` wheel stayed invisible to root discovery while the tool that
#: greps for kernel source listed it -- and a root that is never searched reads
#: downstream exactly like a kernel whose source is not on this host.
FRAMEWORK_SOURCE_PACKAGES: tuple[str, ...] = (
    "aiter",
    "aiter_meta",
    "sglang",
    "sgl_kernel",
    "vllm",
    "atom",
    "xfuser",
)

#: Backwards-compatible private alias.
_FRAMEWORK_PACKAGES: tuple[str, ...] = FRAMEWORK_SOURCE_PACKAGES

#: Packages an isolated vLLM venv may hold. Deliberately narrower than
#: :data:`FRAMEWORK_SOURCE_PACKAGES`: that tree exists because vLLM needs its
#: own interpreter, so only vLLM and the kernel library it links against are
#: expected to live there.
_VLLM_VENV_PACKAGES: tuple[str, ...] = ("vllm", "aiter", "aiter_meta")

# Parents scanned for ``python*/{site,dist}-packages/<pkg>`` wheel layouts.
_INSTALL_GLOB_PARENTS: tuple[Path, ...] = (
    Path("/usr/local/lib"),
    Path("/opt/venv/lib"),
)


def _site_packages_patterns(packages: Sequence[str], *, flavours: Sequence[str]) -> tuple[str, ...]:
    """Build ``python*/<flavour>-packages/<pkg>`` globs for each package."""
    return tuple(f"python*/{flavour}-packages/{package}" for flavour in flavours for package in packages)


# aiter device sources often live in the sibling ``aiter_meta`` package.
_AITER_META_CSRC_ROOT = "/aiter_meta/csrc/"

# ROCm / HIP source roots for the enablement path, always merged into the allowlist.
_ROCM_HIP_SOURCE_ROOTS: tuple[str, ...] = ("/opt/rocm/",)


def resolve_rocm_hip_source_roots() -> tuple[str, ...]:
    """Return the ROCm/HIP source roots for the enablement path."""
    return _ROCM_HIP_SOURCE_ROOTS


#: File types an enablement patch may write under the ROCm/HIP roots.
#: Runtime objects (``.so``, binaries) stay readable via the source allowlist
#: but are not writable through localization or artifact install.
_ROCM_HIP_WRITE_SUFFIXES: frozenset[str] = frozenset(
    {
        ".c",
        ".cc",
        ".cpp",
        ".cxx",
        ".cu",
        ".cuh",
        ".h",
        ".hh",
        ".hpp",
        ".hxx",
        ".hip",
        ".cl",
        ".cmake",
        ".txt",
        ".in",
        ".py",
        ".s",
        ".asm",
        ".inc",
        ".inl",
    }
)
_ROCM_HIP_WRITE_NAMES: frozenset[str] = frozenset({"cmakelists.txt", "makefile"})


def is_rocm_hip_path(value: str) -> bool:
    """True when ``value`` resolves under a ROCm/HIP source root."""
    return any(resolved_within(value, root) for root in resolve_rocm_hip_source_roots())


def is_rocm_hip_writable_path(value: str) -> bool:
    """True when ``value`` may be written (not a ROCm runtime object)."""
    if not is_rocm_hip_path(value):
        return True
    try:
        path = Path(str(value)).resolve()
    except (OSError, RuntimeError):
        return False
    if path.name.lower() in _ROCM_HIP_WRITE_NAMES:
        return True
    return path.suffix.lower() in _ROCM_HIP_WRITE_SUFFIXES


# FlyDSL checkout roots. Env overrides come first, then the image defaults.
_FLYDSL_ROOT_ENV_KEYS: tuple[str, ...] = ("DSL2_ROOT", "FLYDSL_ROOT")
_FLYDSL_DEFAULT_ROOTS: tuple[str, ...] = ("/opt/flydsl/", "/sgl-workspace/flydsl/")


def resolve_flydsl_source_roots() -> tuple[str, ...]:
    """Return the FlyDSL checkout roots for patch-target matching."""
    out: list[str] = []
    for key in _FLYDSL_ROOT_ENV_KEYS:
        root = _normalize_root(os.environ.get(key, ""))
        if root:
            out.extend((root, root.lower()))
    out.extend(_FLYDSL_DEFAULT_ROOTS)
    return _merge_roots(tuple(out))


#: FlyDSL hashes every ``.py`` under these dirs into its JIT cache key.
ENV_FLYDSL_EXTRA_SOURCE_DIRS = "FLYDSL_EXTRA_SOURCE_DIRS"


def flydsl_extra_source_dirs() -> str:
    """Value for ``$FLYDSL_EXTRA_SOURCE_DIRS``: the FlyDSL roots that exist."""
    found: list[str] = []
    preset = os.environ.get(ENV_FLYDSL_EXTRA_SOURCE_DIRS, "").strip()
    if preset:
        found.extend(p for p in preset.split(":") if p.strip())
    for root in resolve_flydsl_source_roots():
        path = Path(root.rstrip("/"))
        if path.is_dir() and str(path) not in found:
            found.append(str(path))
    return ":".join(found)


# Minimal static fallbacks when importlib/glob find nothing (image defaults).
_STATIC_PATCH_FALLBACK_ROOTS: tuple[str, ...] = (
    "/opt/venv/lib/python3.10/site-packages/aiter/",
    "/opt/venv/lib/python3.10/site-packages/sglang/",
    "/opt/venv/lib/python3.10/site-packages/vllm/",
    "/opt/venv/lib/python3.10/site-packages/atom/",
    "/opt/venv/lib/python3.12/site-packages/aiter/",
    "/opt/venv/lib/python3.12/site-packages/sglang/",
    "/opt/venv/lib/python3.12/site-packages/vllm/",
    "/opt/venv/lib/python3.12/site-packages/atom/",
    "/usr/local/lib/python3.12/dist-packages/aiter/",
    "/usr/local/lib/python3.12/dist-packages/sglang/",
    "/usr/local/lib/python3.12/dist-packages/vllm/",
    "/usr/local/lib/python3.12/dist-packages/atom/",
    "/usr/local/lib/python3.10/dist-packages/aiter/",
    "/usr/local/lib/python3.10/dist-packages/sglang/",
    "/usr/local/lib/python3.10/dist-packages/vllm/",
    "/usr/local/lib/python3.10/dist-packages/atom/",
    "/app/ATOM/atom/",
    "/app/xDiT/",
    _AITER_META_CSRC_ROOT,
)


def _normalize_root(path: str) -> str:
    """Normalise a root path to a trailing-slash form."""
    p = str(path or "").strip()
    if not p:
        return ""
    return p if p.endswith("/") else f"{p}/"


def _merge_roots(*groups: tuple[str, ...]) -> tuple[str, ...]:
    """Concatenate root groups, dropping blanks and duplicates."""
    seen: set[str] = set()
    out: list[str] = []
    for group in groups:
        for root in group:
            if root and root not in seen:
                seen.add(root)
                out.append(root)
    return tuple(out)


def _find_spec_origin(module_name: str) -> Path | None:
    """Return the package directory for an importable module."""
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ModuleNotFoundError, ValueError):
        return None
    if spec is None or not spec.origin:
        return None
    origin = Path(spec.origin)
    return origin.parent


def _glob_install_package_roots() -> tuple[str, ...]:
    """Discover framework package dirs under common lib layouts."""
    patterns = _site_packages_patterns(FRAMEWORK_SOURCE_PACKAGES, flavours=("dist", "site"))
    found: list[str] = []
    seen: set[str] = set()
    parents: list[Path] = list(_INSTALL_GLOB_PARENTS)
    prefix_lib = Path(sys.prefix) / "lib"
    if prefix_lib.is_dir() and prefix_lib not in parents:
        parents.append(prefix_lib)
    for parent in parents:
        if not parent.is_dir():
            continue
        for pattern in patterns:
            for match in sorted(parent.glob(pattern)):
                if not match.is_dir():
                    continue
                root = _normalize_root(str(match))
                if root and root not in seen:
                    seen.add(root)
                    found.append(root)
    return tuple(found)


def _discover_installed_framework_roots() -> tuple[str, ...]:
    """Runtime discovery via importlib and filesystem globs."""
    found: list[str] = []
    seen: set[str] = set()

    def add(path: str | Path) -> None:
        """Append a normalised root to ``found`` if new and non-empty."""
        root = _normalize_root(str(path))
        if root and root not in seen:
            seen.add(root)
            found.append(root)

    for mod in _FRAMEWORK_PACKAGES:
        origin = _find_spec_origin(mod)
        if origin is not None:
            add(origin)

    venv = os.environ.get("VIRTUAL_ENV", "").strip()
    if venv:
        site = Path(venv) / "lib"
        if site.is_dir():
            for pattern in _site_packages_patterns(FRAMEWORK_SOURCE_PACKAGES, flavours=("site",)):
                for match in sorted(site.glob(pattern)):
                    if match.is_dir():
                        add(match)

    # Isolated vLLM lives outside $VIRTUAL_ENV; only fall back to the installer's VLLM_VENV_ROOT when no vllm root was
    # found in the main venv above.
    if not any(r.rstrip("/").endswith("/vllm") for r in found):
        vllm_venv = os.environ.get("VLLM_VENV_ROOT", "").strip()
        if vllm_venv:
            site = Path(vllm_venv) / "lib"
            if site.is_dir():
                for pattern in _site_packages_patterns(_VLLM_VENV_PACKAGES, flavours=("site",)):
                    for match in sorted(site.glob(pattern)):
                        if match.is_dir():
                            add(match)

    for root in _glob_install_package_roots():
        add(root)

    return tuple(found)


def _scriptable_frameworks() -> tuple[str, ...]:
    """Return the registered scriptable framework names (empty on import error)."""
    try:
        from hyperloom.inference_optimizer import framework_registry as _reg

        return tuple(name for name in _reg.names() if _reg.is_scriptable(name))
    except Exception:  # noqa: BLE001 - discovery must never break path resolution
        return ()


def _framework_repo_dirname(framework: str) -> str:
    """Return the checkout directory name implied by a framework's repo URL."""
    try:
        from hyperloom.inference_optimizer import framework_registry as _reg

        spec = _reg.FRAMEWORKS.get(framework)
        url = str(getattr(spec, "repo_url", "") or "").strip()
    except Exception:  # noqa: BLE001
        return ""
    if not url:
        return ""
    name = url.rstrip("/").rsplit("/", 1)[-1]
    return name[:-4] if name.endswith(".git") else name


def _discover_scriptable_repo_roots() -> tuple[str, ...]:
    """Discover git-checkout roots for scriptable frameworks."""
    found: list[str] = []
    seen: set[str] = set()
    for framework in _scriptable_frameworks():
        prefix = framework.upper()
        for var in (f"{prefix}_REPO_PATH", f"{prefix}_DIR"):
            candidate = os.environ.get(var, "").strip()
            if not candidate or not Path(candidate).is_dir():
                continue
            root = _normalize_root(candidate)
            if root and root not in seen:
                seen.add(root)
                found.append(root)
    return tuple(found)


def _discover_explicit_framework_root() -> tuple[str, ...]:
    """Discover the framework checkout named by the framework-agnostic env var."""
    candidate = os.environ.get(GENERIC_FRAMEWORK_ROOT_ENV, "").strip()
    if not candidate or not Path(candidate).is_dir():
        return ()
    root = _normalize_root(candidate)
    return (root,) if root else ()


def _discover_installed_package_roots() -> tuple[str, ...]:
    """Return active site/dist-packages roots available to specialists."""
    candidates: list[Path] = []
    try:
        candidates.extend(Path(p) for p in site.getsitepackages())
    except (AttributeError, OSError):
        pass
    try:
        user_site = site.getusersitepackages()
        if user_site:
            candidates.append(Path(user_site))
    except (AttributeError, OSError):
        pass
    for key in ("purelib", "platlib"):
        value = sysconfig.get_path(key)
        if value:
            candidates.append(Path(value))
    candidates.extend(Path(p) for p in sys.path if p and Path(p).name in {"site-packages", "dist-packages"})
    for env_name in ("VIRTUAL_ENV", "VLLM_VENV_ROOT"):
        root = Path(os.environ.get(env_name, "").strip())
        lib = root / "lib"
        if lib.is_dir():
            candidates.extend(lib.glob("python*/site-packages"))
            candidates.extend(lib.glob("python*/dist-packages"))

    found: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        root = _normalize_root(str(candidate))
        if root and root not in seen:
            seen.add(root)
            found.append(root)
    return tuple(found)


def resolve_source_file_allowlist() -> tuple[str, ...]:
    """Return trusted source roots available to specialists and integration."""
    env = os.environ.get("INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS", "").strip()
    kept: list[str] = []
    for raw in env.split(":") if env else ():
        entry = raw.strip()
        if not entry:
            continue
        if not Path(entry).is_absolute():
            log.warning("ignoring non-absolute framework source root: %r", entry)
            continue
        kept.append(_normalize_root(entry))
    env_roots = tuple(kept)
    return _merge_roots(
        _DEFAULT_SOURCE_ROOTS,
        _discover_installed_package_roots(),
        _discover_installed_framework_roots(),
        _discover_scriptable_repo_roots(),
        _discover_explicit_framework_root(),
        env_roots,
        resolve_rocm_hip_source_roots(),
    )


def resolve_session_framework_root() -> str:
    """The one source tree this session was explicitly pointed at, or ``\"\"``."""
    framework = os.environ.get("FRAMEWORK", "").strip().upper()
    if framework:
        for key in (f"{framework}_REPO_PATH", f"{framework}_DIR"):
            candidate = os.environ.get(key, "").strip()
            if candidate and Path(candidate).is_dir():
                return _normalize_root(candidate)
        generic = _discover_explicit_framework_root()
        return generic[0] if generic else ""

    # Compatibility for callers that set one prefixed root but not FRAMEWORK.
    prefixed = _discover_scriptable_repo_roots()
    if len(prefixed) == 1:
        return prefixed[0]

    generic = _discover_explicit_framework_root()
    return generic[0] if generic else ""


def resolve_framework_tree(framework: str) -> str:
    """Return the source tree belonging to ``framework``, or ``\"\"``."""
    pkg = str(framework or "").strip().lower()
    if not pkg:
        return ""
    for key in (f"{pkg.upper()}_REPO_PATH", f"{pkg.upper()}_DIR", GENERIC_FRAMEWORK_ROOT_ENV):
        candidate = os.environ.get(key, "").strip()
        if candidate and Path(candidate).is_dir():
            return _normalize_root(candidate)
    origin = _find_spec_origin(pkg)
    if origin is not None:
        return _normalize_root(str(origin))
    for default in _DEFAULT_SOURCE_ROOTS:
        if default.rstrip("/").endswith(f"/{pkg}") and Path(default).is_dir():
            return _normalize_root(default)
    return ""


def resolve_patch_target_roots() -> tuple[str, ...]:
    """Roots for substring matching in patch apply + kernel classifiers."""
    return _merge_roots(
        resolve_source_file_allowlist(),
        _STATIC_PATCH_FALLBACK_ROOTS,
        resolve_flydsl_source_roots(),
    )


def resolve_kernel_search_roots() -> tuple[str, ...]:
    """Roots to grep when locating the source that defines a GPU kernel."""
    merged = _merge_roots(
        _discover_installed_framework_roots(),
        _discover_scriptable_repo_roots(),
        _discover_explicit_framework_root(),
        _DEFAULT_SOURCE_ROOTS,
        resolve_flydsl_source_roots(),
    )
    return tuple(root for root in merged if Path(root.rstrip("/")).is_dir())


def probe_framework_source_roots_for_env() -> str:
    """Colon-separated roots for ``INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS``."""
    found: list[str] = []
    for root in resolve_source_file_allowlist():
        p = Path(root.rstrip("/"))
        if p.is_dir():
            found.append(_normalize_root(str(p)))
    return ":".join(found)


# Ordered for deterministic substring matching (atom before vllm/sglang).
_FRAMEWORK_BUCKETS: tuple[str, ...] = ("atom", "vllm", "sglang", "aiter", "xdit", "custom")


def summarise_framework_root_discovery(roots: str) -> str:
    """Return ``\"sglang=ok atom=missing ...\"``-style one-line summary."""
    parts: list[str] = []
    items = [p.strip().lower() for p in (roots or "").split(":") if p.strip()]
    for fw in _FRAMEWORK_BUCKETS:
        # A checkout directory rarely matches the framework name, so accept the repo dirname the registry implies too.
        tokens = [f"/{fw}/"]
        dirname = _framework_repo_dirname(fw)
        if dirname:
            tokens.append(f"/{dirname.lower()}/")
        status = "ok" if any(item.endswith(t) for item in items for t in tokens) else "missing"
        parts.append(f"{fw}={status}")
    return " ".join(parts)


# A profile trace names a frame as ``<path>(<line>): <function>``.
_TRACE_FRAME_SUFFIX = re.compile(r"\(\d+\)\s*:.*$")


def resolved_within(value: str, root: str) -> bool:
    """Return whether ``value`` resolves to or under ``root`` (symlinks resolved)."""
    try:
        v = Path(str(value)).resolve()
        r = Path(str(root)).resolve()
    except (OSError, RuntimeError):
        return False
    return v == r or v.is_relative_to(r)


def source_file_candidates(value: str) -> tuple[str, ...]:
    """Return the path forms a ``source_file`` value may legitimately take."""
    raw = str(value).strip()
    out: list[str] = [raw]
    bare = _TRACE_FRAME_SUFFIX.sub("", raw).strip()
    if bare and bare != raw:
        out.append(bare)
    if bare and not Path(bare).is_absolute():
        root = resolve_session_framework_root()
        if root:
            out.append(str(Path(root) / bare))
        for allow_root in resolve_source_file_allowlist():
            joined = Path(allow_root) / bare
            try:
                exists = joined.is_file()
            except OSError:
                continue
            if exists:
                candidate = str(joined)
                if candidate not in out:
                    out.append(candidate)
    return tuple(out)


__all__ = [
    "FRAMEWORK_SOURCE_PACKAGES",
    "probe_framework_source_roots_for_env",
    "resolve_framework_tree",
    "resolve_kernel_search_roots",
    "resolve_patch_target_roots",
    "resolve_rocm_hip_source_roots",
    "is_rocm_hip_path",
    "is_rocm_hip_writable_path",
    "resolve_session_framework_root",
    "resolve_source_file_allowlist",
    "resolved_within",
    "source_file_candidates",
    "summarise_framework_root_discovery",
]
