# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Framework source-root resolution and path containment.

Centralises probe order across container layouts (``/sgl-workspace/...``,
``/app/ATOM/atom``, ``/app/xDiT``, site/dist-packages) so PolicyGate, AST
discovery, install.sh, and ``apply_kernel_patch`` all agree. First-class
frameworks: atom, sglang, vllm, xdit (``xfuser`` package); aiter is in the
allowlist as a shared kernel library.

Owning the roots and the containment test together keeps every caller on one
boundary rule: :func:`resolved_within` against a root from this module.
"""

from __future__ import annotations

import importlib.util
import os
import re
import site
import sys
import sysconfig
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, NamedTuple

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

_FRAMEWORK_PACKAGES: tuple[str, ...] = ("aiter", "sglang", "vllm", "atom", "xfuser")

# Parents scanned for ``python*/{site,dist}-packages/<pkg>`` wheel layouts.
_INSTALL_GLOB_PARENTS: tuple[Path, ...] = (
    Path("/usr/local/lib"),
    Path("/opt/venv/lib"),
)

# aiter device sources often live in the sibling ``aiter_meta`` package.
_AITER_META_CSRC_ROOT = "/aiter_meta/csrc/"

# ROCm / HIP source roots for the enablement path, always merged into the
# allowlist.
_ROCM_HIP_SOURCE_ROOTS: tuple[str, ...] = ("/opt/rocm/",)


def resolve_rocm_hip_source_roots() -> tuple[str, ...]:
    """Return the ROCm/HIP source roots for the enablement path.

    Always included in :func:`resolve_source_file_allowlist`.

    Returns:
        tuple[str, ...]: :data:`_ROCM_HIP_SOURCE_ROOTS`.
    """
    return _ROCM_HIP_SOURCE_ROOTS


# FlyDSL checkout roots. Env overrides come first, then the image defaults.
_FLYDSL_ROOT_ENV_KEYS: tuple[str, ...] = ("DSL2_ROOT", "FLYDSL_ROOT")
_FLYDSL_DEFAULT_ROOTS: tuple[str, ...] = ("/opt/flydsl/", "/sgl-workspace/flydsl/")


def resolve_flydsl_source_roots() -> tuple[str, ...]:
    """Return the FlyDSL checkout roots for patch-target matching.

    Included in :func:`resolve_patch_target_roots` but deliberately not in
    :func:`resolve_source_file_allowlist`: FlyDSL is a rewrite target for the
    kernel agent, not a framework the specialist may edit.

    An env-supplied root is emitted both case-preserved and lower-cased,
    because the patchability and apply gates match a lower-cased path against
    these roots verbatim while path-resolving consumers need the real case.

    Returns:
        tuple[str, ...]: The de-duplicated FlyDSL roots.
    """
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
    """Value for ``$FLYDSL_EXTRA_SOURCE_DIRS``: the FlyDSL roots that exist.

    FlyDSL's cache key covers the traced function and same-directory helpers
    only, so an edited helper in a sibling directory does not invalidate it and
    the stale binary is reused. Listing the roots here folds their sources into
    the key, re-compiling only the kernels that actually changed.

    Any operator-supplied value is preserved and comes first.

    Returns:
        str: Existing roots joined by ``:`` (empty when none exist).
    """
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
    """Normalise a root path to a trailing-slash form.

    Args:
        path (str): Raw path string (may be empty / whitespace).

    Returns:
        str: The stripped path with a guaranteed trailing ``/``, or an
            empty string when the input was blank.
    """
    p = str(path or "").strip()
    if not p:
        return ""
    return p if p.endswith("/") else f"{p}/"


def _merge_roots(*groups: tuple[str, ...]) -> tuple[str, ...]:
    """Concatenate root groups, dropping blanks and duplicates.

    Args:
        *groups (tuple[str, ...]): One or more ordered groups of root
            strings to merge.

    Returns:
        tuple[str, ...]: The merged roots in first-seen order with
            duplicates and empty strings removed.
    """
    seen: set[str] = set()
    out: list[str] = []
    for group in groups:
        for root in group:
            if root and root not in seen:
                seen.add(root)
                out.append(root)
    return tuple(out)


def _find_spec_origin(module_name: str) -> Path | None:
    """Return the package directory for an importable module.

    Args:
        module_name (str): Importable module / package name to locate.

    Returns:
        Path | None: The directory containing the module's origin (its
            parent dir, whether or not it's a package ``__init__.py``), or
            None when the module cannot be found / has no origin.
    """
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ModuleNotFoundError, ValueError):
        return None
    if spec is None or not spec.origin:
        return None
    origin = Path(spec.origin)
    return origin.parent


def _glob_install_package_roots() -> tuple[str, ...]:
    """Discover framework package dirs under common lib layouts.

    Globs ``python*/{site,dist}-packages/<pkg>`` under the known install
    parents plus ``sys.prefix/lib``.

    Returns:
        tuple[str, ...]: Normalised, de-duplicated package root paths.
    """
    patterns = (
        "python*/dist-packages/aiter",
        "python*/dist-packages/aiter_meta",
        "python*/dist-packages/sglang",
        "python*/dist-packages/vllm",
        "python*/dist-packages/atom",
        "python*/dist-packages/xfuser",
        "python*/site-packages/aiter",
        "python*/site-packages/aiter_meta",
        "python*/site-packages/sglang",
        "python*/site-packages/vllm",
        "python*/site-packages/atom",
        "python*/site-packages/xfuser",
    )
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
    """Runtime discovery via importlib and filesystem globs.

    Combines ``importlib`` spec origins for each framework package, a
    ``$VIRTUAL_ENV`` site-packages glob, and the common install-parent
    globs.

    Returns:
        tuple[str, ...]: Normalised, de-duplicated discovered root paths.
    """
    found: list[str] = []
    seen: set[str] = set()

    def add(path: str | Path) -> None:
        """Append a normalised root to ``found`` if new and non-empty.

        Args:
            path (str | Path): Candidate root path to record.
        """
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
            for pattern in (
                "python*/site-packages/vllm",
                "python*/site-packages/sglang",
                "python*/site-packages/aiter",
                "python*/site-packages/aiter_meta",
                "python*/site-packages/atom",
                "python*/site-packages/xfuser",
            ):
                for match in sorted(site.glob(pattern)):
                    if match.is_dir():
                        add(match)

    # Isolated vLLM lives outside $VIRTUAL_ENV; only fall back to the installer's
    # VLLM_VENV_ROOT when no vllm root was found in the main venv above.
    if not any(r.rstrip("/").endswith("/vllm") for r in found):
        vllm_venv = os.environ.get("VLLM_VENV_ROOT", "").strip()
        if vllm_venv:
            site = Path(vllm_venv) / "lib"
            if site.is_dir():
                for pattern in (
                    "python*/site-packages/vllm",
                    "python*/site-packages/aiter",
                    "python*/site-packages/aiter_meta",
                ):
                    for match in sorted(site.glob(pattern)):
                        if match.is_dir():
                            add(match)

    for root in _glob_install_package_roots():
        add(root)

    return tuple(found)


def _scriptable_frameworks() -> tuple[str, ...]:
    """Return the registered scriptable framework names (empty on import error).

    Imported lazily: ``framework_registry`` lives in ``inference_optimizer`` and
    importing it at module scope would close a cycle back through this package.

    Returns:
        tuple[str, ...]: Scriptable framework names, or ``()`` when the registry
            cannot be imported.
    """
    try:
        from hyperloom.inference_optimizer import framework_registry as _reg

        return tuple(name for name in _reg.names() if _reg.is_scriptable(name))
    except Exception:  # noqa: BLE001 - discovery must never break path resolution
        return ()


def _framework_repo_dirname(framework: str) -> str:
    """Return the checkout directory name implied by a framework's repo URL.

    ``my-framework.git`` -> ``my-framework``. Used so a checkout whose directory
    name differs from the framework name still registers as discovered.

    Args:
        framework (str): Registered framework name.

    Returns:
        str: The bare repo directory name, or ``""`` when unknown.
    """
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
    """Discover git-checkout roots for scriptable frameworks.

    A scriptable framework runs out of a repo checkout
    instead of a pip-installed package, so importlib spec origins and the
    site-packages globs never see them. Materialization exports the resolved
    checkout as ``<FRAMEWORK>_REPO_PATH`` / ``<FRAMEWORK>_DIR``; without those
    roots PolicyGate rejects every patch against the framework's own source and
    framework-agent cannot touch the code it is meant to optimize.

    Returns:
        tuple[str, ...]: Normalised, de-duplicated checkout roots that exist.
    """
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
    """Discover the framework checkout named by the framework-agnostic env var.

    ``<FRAMEWORK>_REPO_PATH`` requires the operator to know the framework name
    before the right variable can be set, and to change variable names when
    switching frameworks — for a value that cannot collide, since a session is
    single-framework by construction (the CLI locks ``$FRAMEWORK``). This accepts
    the same thing without the prefix, and unlike the scriptable discovery it is
    not restricted to registered scriptable frameworks: an editable checkout of a
    normally pip-installed framework is invisible to both importlib and the
    site-packages scan, and this is how it gets pointed at.

    A prefixed value keeps precedence, because it is the more specific statement.

    Returns:
        tuple[str, ...]: The normalised checkout root, or empty when unset or absent.
    """
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
    """Return trusted source roots available to specialists and integration.

    Includes editable framework trees and every active site/dist-packages root.
    File-level editability is decided during reviewed integration rather than by
    restricting specialist discovery to named framework packages.

    Returns:
        tuple[str, ...]: The merged, de-duplicated allowlist roots.
    """
    env = os.environ.get("INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS", "").strip()
    env_roots = tuple(_normalize_root(p) for p in env.split(":") if p.strip()) if env else ()
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
    """The one source tree this session was explicitly pointed at, or ``""``.

    :func:`resolve_source_file_allowlist` answers "may this be edited", and its
    order is an artefact of how the roots were discovered — ``/sgl-workspace/aiter/``
    heads the static defaults, so it comes first whatever the session is
    optimising. Anything that needs to name *the* tree under optimisation must
    ask for it, not read position 0 of a permission set: a session that picked
    the head of the allowlist got an aiter checkout, and every patch naming a
    file in the real tree failed to apply against it.

    Only the explicitly-named checkout counts. Discovery by import or by
    globbing site-packages finds whatever the image happens to ship, which is
    the same guess with more steps.

    Returns:
        str: The normalised checkout root, or ``""`` when the session named none.
    """
    framework = os.environ.get("FRAMEWORK", "").strip().upper()
    if framework:
        for key in (f"{framework}_REPO_PATH", f"{framework}_DIR"):
            candidate = os.environ.get(key, "").strip()
            if candidate and Path(candidate).is_dir():
                return _normalize_root(candidate)
        generic = _discover_explicit_framework_root()
        return generic[0] if generic else ""

    # Compatibility for callers that set one prefixed root but not FRAMEWORK.
    # More than one is ambiguous and must not be resolved by probe order.
    prefixed = _discover_scriptable_repo_roots()
    if len(prefixed) == 1:
        return prefixed[0]

    generic = _discover_explicit_framework_root()
    return generic[0] if generic else ""


def resolve_patch_target_roots() -> tuple[str, ...]:
    """Roots for substring matching in patch apply + kernel classifiers.

    Same as :func:`resolve_source_file_allowlist` plus static fallbacks for
    layouts that are not importable until first use (e.g. ``aiter_meta/csrc``)
    and the FlyDSL checkout roots.

    Returns:
        tuple[str, ...]: The allowlist roots merged with the static patch
            fallback roots and the FlyDSL roots.
    """
    return _merge_roots(
        resolve_source_file_allowlist(),
        _STATIC_PATCH_FALLBACK_ROOTS,
        resolve_flydsl_source_roots(),
    )


def probe_framework_source_roots_for_env() -> str:
    """Colon-separated roots for ``INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS``.

    Filters the resolved allowlist down to roots that exist on disk.

    Returns:
        str: Existing roots joined by ``:`` (empty string when none exist).
    """
    found: list[str] = []
    for root in resolve_source_file_allowlist():
        p = Path(root.rstrip("/"))
        if p.is_dir():
            found.append(_normalize_root(str(p)))
    return ":".join(found)


# Ordered for deterministic substring matching (atom before vllm/sglang).
_FRAMEWORK_BUCKETS: tuple[str, ...] = ("atom", "vllm", "sglang", "aiter", "xdit", "custom")


def summarise_framework_root_discovery(roots: str) -> str:
    """Return ``"sglang=ok atom=missing ..."``-style one-line summary.

    Input is the colon-separated string from
    ``probe_framework_source_roots_for_env``; emitted in ``_FRAMEWORK_BUCKETS``
    order for stable output.

    Args:
        roots: Colon-separated source roots to summarise.

    Returns:
        A one-line ``fw=ok``/``fw=missing`` summary in bucket order.
    """
    parts: list[str] = []
    items = [p.strip().lower() for p in (roots or "").split(":") if p.strip()]
    for fw in _FRAMEWORK_BUCKETS:
        # A checkout directory rarely matches the framework name, so accept the
        # repo dirname the registry implies too.
        tokens = [f"/{fw}/"]
        dirname = _framework_repo_dirname(fw)
        if dirname:
            tokens.append(f"/{dirname.lower()}/")
        status = "ok" if any(item.endswith(t) for item in items for t in tokens) else "missing"
        parts.append(f"{fw}={status}")
    return " ".join(parts)


_WARM_REPLAY_FRAMEWORK_ROOT_TOKENS: tuple[str, ...] = ("sglang", "vllm")
_WARM_REPLAY_KERNEL_ROOT_TOKENS: tuple[str, ...] = (
    "aiter",
    "aiter_meta",
    "rocm",
    "flydsl",
)
_WARM_REPLAY_FRAMEWORK_STATIC_ROOTS: tuple[str, ...] = (
    "/sgl-workspace/sglang/",
    "/sgl-workspace/vllm/",
)
_WARM_REPLAY_KERNEL_STATIC_ROOTS: tuple[str, ...] = (
    "/sgl-workspace/aiter/",
    _AITER_META_CSRC_ROOT,
    *_ROCM_HIP_SOURCE_ROOTS,
    *_FLYDSL_DEFAULT_ROOTS,
)


_WARM_REPLAY_PATCH_PATH_KEYS: tuple[str, ...] = (
    "patch_ref",
    "patch_path",
    "patch_file",
)


class WarmReplayRootResolution(NamedTuple):
    """Result of resolving one warm-replay patch apply root."""

    root: str
    source: str
    reason: str
    allowlist: tuple[str, ...]


class WarmReplayPatchSource(NamedTuple):
    """One warm-replay patch, as a file on disk and/or an inline diff."""

    path: Path | None
    content: str


def warm_replay_patch_sources(
    entries: Iterable[Any] | None = None,
    patch_paths: Sequence[Path] | None = None,
) -> tuple[WarmReplayPatchSource, ...]:
    """Collect patch files and inline diffs from warm-replay patch entries.

    Args:
        entries: Warm-recipe patch rows, each naming a file and/or an inline
            diff. Rows carrying neither are skipped.
        patch_paths: Additional patch files with no row of their own.

    Returns:
        One source per usable patch, entries first.
    """
    sources: list[WarmReplayPatchSource] = []
    for entry in entries or ():
        if not isinstance(entry, dict):
            continue
        path: Path | None = None
        for key in _WARM_REPLAY_PATCH_PATH_KEYS:
            raw = str(entry.get(key) or "").strip()
            if raw:
                path = Path(raw)
                break
        content = str(entry.get("patch_content") or "")
        if path is None and not content.strip():
            continue
        sources.append(WarmReplayPatchSource(path=path, content=content))
    sources.extend(
        WarmReplayPatchSource(path=Path(str(path)), content="") for path in (patch_paths or ()) if str(path).strip()
    )
    return tuple(sources)


def _root_path_matches_tokens(root: str, tokens: tuple[str, ...]) -> bool:
    normalized = root.lower().rstrip("/")
    return any(f"/{token}/" in f"{normalized}/" for token in tokens)


def _filter_roots_by_tokens(
    roots: Sequence[str],
    tokens: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(root for root in roots if root and _root_path_matches_tokens(root, tokens))


def _warm_replay_framework_patch_roots() -> tuple[str, ...]:
    """Return sglang/vllm roots warm-replay framework patches may target."""
    return _merge_roots(
        _WARM_REPLAY_FRAMEWORK_STATIC_ROOTS,
        _filter_roots_by_tokens(
            _discover_installed_framework_roots(),
            _WARM_REPLAY_FRAMEWORK_ROOT_TOKENS,
        ),
    )


def _warm_replay_kernel_patch_roots() -> tuple[str, ...]:
    """Return aiter/ROCm/FlyDSL roots warm-replay kernel patches may target."""
    return _merge_roots(
        _WARM_REPLAY_KERNEL_STATIC_ROOTS,
        _filter_roots_by_tokens(
            _discover_installed_framework_roots(),
            _WARM_REPLAY_KERNEL_ROOT_TOKENS,
        ),
    )


def _patch_source_diff(source: WarmReplayPatchSource) -> str:
    """Return the diff text used for target matching, or ``""`` when unavailable.

    An entry may carry its diff inline instead of on disk, so a ``patch_*`` ref
    that names no local file is not by itself evidence against a root.
    Falls back to the inline content on any I/O error so a deleted previous-session
    patch does not abort the caller (consistent with the is_file() fallback above).
    """
    if source.path is None or not source.path.is_file():
        return source.content
    try:
        return source.path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return source.content


def _resolve_warm_replay_patch_root(
    *,
    patch_sources: Sequence[WarmReplayPatchSource],
    allowlist: tuple[str, ...],
    missing_patch_reason: str,
    missing_allowlist_reason: str,
    explicit_root: str | None = None,
) -> WarmReplayRootResolution:
    from hyperloom.orchestrator.specialists.patch_safety import (
        resolve_patch_apply_root,
    )

    diffs = [diff for diff in (_patch_source_diff(source) for source in patch_sources) if diff.strip()]
    candidate_roots = tuple(Path(candidate.rstrip("/")) for candidate in allowlist)
    from_env = bool(explicit_root)
    resolution = resolve_patch_apply_root(
        diffs,
        explicit_root=Path(explicit_root.rstrip("/")) if explicit_root else None,
        candidate_roots=candidate_roots,
    )
    # The declared root is preferred, not mandatory: a session pointed at one
    # framework can still hold a replayable patch for a sibling tree.
    if resolution.root is None and resolution.reason == "explicit_root_target_mismatch":
        from_env = False
        resolution = resolve_patch_apply_root(
            diffs,
            explicit_root=None,
            candidate_roots=candidate_roots,
        )
    if resolution.root is not None:
        return WarmReplayRootResolution(
            root=_normalize_root(str(resolution.root)),
            source="env" if from_env else "allowlist",
            reason="",
            allowlist=() if from_env else allowlist,
        )
    reason = {
        "patch_content_missing": missing_patch_reason,
        # Only these two are about the allowlist: nothing offered, nothing
        # matched. Any other reason keeps its own wording, which names a fault
        # the allowlist message would misdirect from.
        "no_candidate_roots": missing_allowlist_reason,
        "no_matching_root": missing_allowlist_reason,
    }.get(resolution.reason, resolution.reason)
    return WarmReplayRootResolution(
        root="",
        source="",
        reason=reason,
        allowlist=allowlist,
    )


def resolve_warm_replay_framework_root(
    *,
    patch_paths: Sequence[Path] | None = None,
    patch_entries: Iterable[Any] | None = None,
) -> WarmReplayRootResolution:
    """Resolve the framework patch root for warm replay (env first, then allowlist)."""
    return _resolve_warm_replay_patch_root(
        patch_sources=warm_replay_patch_sources(patch_entries, patch_paths),
        allowlist=_warm_replay_framework_patch_roots(),
        missing_patch_reason="active_framework_root_missing",
        missing_allowlist_reason="framework_patch_root_not_in_allowlist",
        explicit_root=resolve_session_framework_root() or None,
    )


def resolve_warm_replay_kernel_root(
    *,
    patch_paths: Sequence[Path] | None = None,
    patch_entries: Iterable[Any] | None = None,
    precomputed_allowlist: tuple[str, ...] | None = None,
) -> WarmReplayRootResolution:
    """Resolve the kernel patch root from the kernel-specific allowlist.

    Args:
        patch_paths: Patch file paths to resolve.
        patch_entries: KB plan entry dicts to resolve.
        precomputed_allowlist: Pre-resolved allowlist tuple, used to avoid
            re-running the filesystem discovery when the caller iterates many
            entries. When absent, the allowlist is computed fresh.
    """
    return _resolve_warm_replay_patch_root(
        patch_sources=warm_replay_patch_sources(patch_entries, patch_paths),
        allowlist=precomputed_allowlist if precomputed_allowlist is not None else _warm_replay_kernel_patch_roots(),
        missing_patch_reason="active_kernel_patch_root_missing",
        missing_allowlist_reason="kernel_patch_root_not_in_allowlist",
    )


# A profile trace names a frame as ``<path>(<line>): <function>``. The suffix is
# not part of the path and the path is relative to the tree being profiled.
_TRACE_FRAME_SUFFIX = re.compile(r"\(\d+\)\s*:.*$")


def resolved_within(value: str, root: str) -> bool:
    """Return whether ``value`` resolves to or under ``root`` (symlinks resolved).

    Resolving both sides is what rejects ``..`` traversal, symlink escapes, a
    root substring embedded in an unrelated directory, and shared-prefix
    boundary tricks such as ``/x/aiter`` versus ``/x/aiterX``.

    Args:
        value (str): the candidate path string.
        root (str): an allowlist root (may carry a trailing slash).

    Returns:
        bool: True when the resolved ``value`` equals or is nested under the
            resolved ``root``; False on any resolution error or escape.
    """
    try:
        v = Path(str(value)).resolve()
        r = Path(str(root)).resolve()
    except (OSError, RuntimeError):
        return False
    return v == r or v.is_relative_to(r)


def source_file_candidates(value: str) -> tuple[str, ...]:
    """Return the path forms a ``source_file`` value may legitimately take.

    Roofline evidence reaches the orchestration prompt as trace frames and the
    model cites them verbatim, so a relative frame must be resolved against the
    session's framework tree rather than the process CWD, or every citation is
    denied. Each candidate is still bounded by :func:`resolved_within`.

    A pip-installed framework names no checkout, so
    :func:`resolve_session_framework_root` is empty and that join never fires --
    while the frame's file does sit under an allowlist root. Those roots are
    tried too, but only where the join names a file that exists: a root is a
    permission set, not evidence that the frame belongs to it, and admitting
    every root would turn one unresolvable frame into a path in each of them.

    Args:
        value (str): The raw field value.

    Returns:
        tuple[str, ...]: ``value`` first, then the de-annotated form, then that
            form resolved against the tree this session is optimizing, then
            against each allowlist root that holds the named file.
    """
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
    "WarmReplayPatchSource",
    "WarmReplayRootResolution",
    "probe_framework_source_roots_for_env",
    "resolve_patch_target_roots",
    "resolve_rocm_hip_source_roots",
    "resolve_session_framework_root",
    "resolve_source_file_allowlist",
    "resolve_warm_replay_framework_root",
    "resolve_warm_replay_kernel_root",
    "resolved_within",
    "source_file_candidates",
    "summarise_framework_root_discovery",
    "warm_replay_patch_sources",
]
