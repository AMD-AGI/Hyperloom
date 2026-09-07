# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Framework source-root resolution.

Centralises probe order across container layouts (``/sgl-workspace/...``,
``/app/ATOM/atom``, ``/app/xDiT``, site/dist-packages) so navigation hints,
AST discovery, install.sh, and ``apply_kernel_patch`` all agree. First-class
frameworks: atom, sglang, vllm, xdit (``xfuser`` package); aiter is discovered
as a shared kernel library.

Three resolvers, three questions, not interchangeable.
:func:`resolve_framework_tree` names *the* tree a session is optimising, keyed by
the framework's own name. :func:`resolve_kernel_search_roots` lists the trees
worth searching, filtered to what exists here. :func:`resolve_known_source_prefixes`
matches a path string against every layout that could hold source, including
those absent from this host. None is a permission set: what may be written is
decided by the integration step that applies the patch.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import re
import sys
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
    """Build ``python*/<flavour>-packages/<pkg>`` globs for each package.

    Args:
        packages: Package names to match.
        flavours: ``site`` / ``dist`` -- both spellings exist depending on how
            the interpreter was built.

    Returns:
        One pattern per (flavour, package) pair, in that order.
    """
    return tuple(f"python*/{flavour}-packages/{package}" for flavour in flavours for package in packages)


# FlyDSL checkout roots. Env overrides come first, then the image defaults.
_FLYDSL_ROOT_ENV_KEYS: tuple[str, ...] = ("DSL2_ROOT", "FLYDSL_ROOT")
_FLYDSL_DEFAULT_ROOTS: tuple[str, ...] = ("/opt/flydsl/", "/sgl-workspace/flydsl/")


def resolve_flydsl_source_roots() -> tuple[str, ...]:
    """Return the FlyDSL checkout roots for patch-target matching.

    FlyDSL is a rewrite target for the kernel agent, so its roots join the
    search roots rather than being discovered as a framework package.

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
            for pattern in _site_packages_patterns(FRAMEWORK_SOURCE_PACKAGES, flavours=("site",)):
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
                for pattern in _site_packages_patterns(_VLLM_VENV_PACKAGES, flavours=("site",)):
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
    roots the framework's own source is invisible to search and to patch
    grounding, and framework-agent cannot touch the code it is meant to optimize.

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


def _env_source_roots() -> tuple[str, ...]:
    """Roots named by ``$INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS``.

    Written by install.sh's ``_probe_framework_source_roots`` so an operator can
    point a session at a tree that neither importlib nor the site-packages globs
    would find.

    Returns:
        tuple[str, ...]: Normalised absolute roots; non-absolute entries are
            dropped with a warning.
    """
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
    return tuple(kept)


def resolve_session_framework_root() -> str:
    """The one source tree this session was explicitly pointed at, or ``""``.

    :func:`resolve_kernel_search_roots` answers "where might this code be", and
    its order is an artefact of how the roots were discovered — ``/sgl-workspace/aiter/``
    heads the static defaults, so it comes first whatever the session is
    optimising. Anything that needs to name *the* tree under optimisation must
    ask for it, not read position 0 of a discovery order: a session that picked
    the head of that list got an aiter checkout, and every patch naming a file
    in the real tree failed to apply against it.

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


def resolve_framework_tree(framework: str) -> str:
    """Return the source tree belonging to ``framework``, or ``""``.

    Every source consulted here is keyed by the framework's own name — its env
    vars, its Python package, its default path. That is what distinguishes this
    from :func:`resolve_kernel_search_roots`, whose order reflects only how
    roots were discovered and so cannot name the tree a session is optimising.

    Args:
        framework: Framework name, e.g. ``"sglang"``.

    Returns:
        str: The normalised tree root, or ``""`` when the framework has none.
    """
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


#: Wheel layouts on the serving images that the orchestrator process cannot
#: import, so discovery never finds them. String prefixes, not directories.
_STATIC_SOURCE_LAYOUTS: tuple[str, ...] = (
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
    # aiter device sources often live in the sibling ``aiter_meta`` package.
    "/aiter_meta/csrc/",
)


def resolve_known_source_prefixes() -> tuple[str, ...]:
    """Root prefixes for recognising a path *string* as framework source.

    Not existence-filtered, unlike :func:`resolve_kernel_search_roots`: the paths
    classified here come from traces and patch manifests produced on a serving
    pod, so a root absent from this host still names real source on that one.

    Membership permits nothing; a path that matches no prefix is reported as
    unrecognised.

    Returns:
        tuple[str, ...]: Discovered roots, image defaults, static wheel layouts
            and FlyDSL roots, de-duplicated in that order.
    """
    return _merge_roots(
        _discover_installed_framework_roots(),
        _discover_scriptable_repo_roots(),
        _discover_explicit_framework_root(),
        _env_source_roots(),
        _DEFAULT_SOURCE_ROOTS,
        _STATIC_SOURCE_LAYOUTS,
        resolve_flydsl_source_roots(),
    )


def resolve_kernel_search_roots() -> tuple[str, ...]:
    """Roots to grep when locating the source that defines a GPU kernel.

    Named framework trees only: the bare site/dist-packages parents would pull
    in every installed package, torch included, on each keyword.

    Only roots that exist on this host are returned. An empty result therefore
    means "there is nothing here to search", which a caller must surface as a
    misconfiguration -- grepping absent directories yields no hits and is
    indistinguishable from a kernel whose source genuinely is not present.

    Returns:
        tuple[str, ...]: Existing framework package dirs, editable checkouts,
            env-supplied and FlyDSL roots, de-duplicated in discovery order.
    """
    merged = _merge_roots(
        _discover_installed_framework_roots(),
        _discover_scriptable_repo_roots(),
        _discover_explicit_framework_root(),
        _env_source_roots(),
        _DEFAULT_SOURCE_ROOTS,
        resolve_flydsl_source_roots(),
    )
    return tuple(root for root in merged if Path(root.rstrip("/")).is_dir())


def probe_framework_source_roots_for_env() -> str:
    """Colon-separated roots for ``INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS``.

    Returns:
        str: The search roots joined by ``:`` (empty string when none exist).
    """
    return ":".join(resolve_kernel_search_roots())


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
        root (str): a source root (may carry a trailing slash).

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
    session's framework tree rather than the process CWD.

    A pip-installed framework names no checkout, so
    :func:`resolve_session_framework_root` is empty and that join never fires --
    while the frame's file does sit under a search root. Those roots are tried
    too, but only where the join names a file that exists: a root is a discovery
    order, not evidence that the frame belongs to it, and admitting every root
    would turn one unresolvable frame into a path in each of them.

    Args:
        value (str): The raw field value.

    Returns:
        tuple[str, ...]: ``value`` first, then the de-annotated form, then that
            form resolved against the tree this session is optimizing, then
            against each search root that holds the named file.
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
        for search_root in resolve_kernel_search_roots():
            joined = Path(search_root) / bare
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
    "resolve_known_source_prefixes",
    "resolve_session_framework_root",
    "resolved_within",
    "source_file_candidates",
    "summarise_framework_root_discovery",
]
