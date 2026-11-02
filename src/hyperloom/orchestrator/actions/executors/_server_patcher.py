# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""TraceLens's profiling patches for the vLLM, SGLang and xDiT a profile runs: which patch set fits the installed
version, and where it is laid (:mod:`._patch_laying`).

vLLM and SGLang get the patch in a copy of the installed package, which the server that profiles imports first
(:func:`ensure_vllm_patched_for_tracelens`, :func:`ensure_sglang_patched_for_tracelens`), or in a test's farm
(:func:`lay_tracelens_patch`); their installed trees are never written. xDiT's checkout is still patched in place:
Magpie's xDiT script starts its process, and where that script imports ``xfuser`` from is not set here.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Sequence

from hyperloom.common.unified_diff import parse_unified_diff
from hyperloom.inference_optimizer.session import paths

from ._git import _git_head_sha
from ._patch_laying import PatchedServer, _ensure_patched, _ensure_patched_in_place, _lay_in, _PatchPlan

log = logging.getLogger(__name__)


# SGLang version gate: a minor-version allowlist (not an exact pin).
_SGLANG_DEFAULT_ALLOWED_MINORS: tuple[str, ...] = ("0.5",)
_SGLANG_EXACT_VERSIONS_ENV = "HYPERLOOM_SGLANG_PATCH_EXACT_VERSIONS"
_SGLANG_ALLOWED_MINORS_ENV = "HYPERLOOM_SGLANG_PATCH_ALLOWED_MINORS"
_VLLM_EXACT_VERSIONS_ENV = "HYPERLOOM_VLLM_PATCH_EXACT_VERSIONS"

# Required in ``vllm/config/profiler.py`` for the server to accept the flags.
_VLLM_PROFILER_SENTINELS: tuple[str, ...] = (
    "capture_torch_profiler",
    "detailed_trace_annotation",
)

# From 0.26 the fields above are upstream, so they no longer prove the patch landed.
_VLLM_GRAPH_CAPTURE_MIN_VERSION: tuple[int, ...] = (0, 26)
_VLLM_GRAPH_CAPTURE_SENTINEL: tuple[str, ...] = ("vllm", "profiler", "graph_capture.py")
_VLLM_GRAPH_CAPTURE_MARKERS: tuple[str, ...] = ("graph_capture_profiler", "capture_traces")

# xDiT: SHA-based version gating.  ``pip show xfuser`` always reports
# ``0.4.5`` regardless of the actual commit, so version detection reads
# ``git rev-parse HEAD`` from the editable install and matches against
# known image SHAs.
_XDIT_PATCH_RE = re.compile(r"^config_xdit_v(\d+(?:\.\d+)*)\.patch$")
_XDIT_PATCHES_LEAF = "xdit_patches"


def _xdit_default_source_root() -> Path:
    from hyperloom.inference_optimizer import framework_registry

    return Path(framework_registry.source_root("xdit") or "/app/xDiT")


_XDIT_SENTINEL_REL: tuple[str, ...] = ("xfuser", "config", "args.py")
_XDIT_SENTINEL_MARKERS: tuple[str, ...] = ("profile_capture_phase",)
# Known image tag → head commit SHA (full 40-char).
_XDIT_SHA_TO_VERSION: dict[str, str] = {
    "c30266378c815c2f31f95b9d21d2818276050547": "26.7",
    "2f74acb12830163d1fe0dd1abb5422f0bff7acf0": "26.8",
}

# Vendor-shipped manifest filename(s).
_SUPPORTED_VERSIONS_MANIFEST_NAMES: tuple[str, ...] = (
    "SUPPORTED_VERSIONS.txt",
    "SUPPORTED_VERSIONS",
)


def _load_supported_versions_from_manifest(
    patches_dir: Path,
) -> frozenset[str] | None:
    """Read the vendor-shipped ``SUPPORTED_VERSIONS`` manifest if present."""
    for name in _SUPPORTED_VERSIONS_MANIFEST_NAMES:
        manifest = patches_dir / name
        if not manifest.is_file():
            continue
        try:
            raw = manifest.read_text(encoding="utf-8")
        except OSError as e:
            log.warning(
                "_server_patcher: cannot read version manifest %s (%s); falling back to hardcoded allowlist",
                manifest,
                e,
            )
            return None
        versions: set[str] = set()
        for line in raw.splitlines():
            stripped = line.split("#", 1)[0].strip()
            if stripped:
                versions.add(stripped)
        log.info(
            "_server_patcher: loaded %d version(s) from vendor manifest %s",
            len(versions),
            manifest,
        )
        return frozenset(versions)
    return None


def _version_accepted(
    version: str,
    *,
    patches_dir: Path | None = None,
) -> bool:
    """Return True iff an SGLang ``version`` is in the resolved allowlist."""
    text = (version or "").strip()
    if not text:
        return False
    exact = os.environ.get(_SGLANG_EXACT_VERSIONS_ENV, "").strip()
    if exact:
        allowed_exact = {v.strip() for v in exact.split(",") if v.strip()}
        return text in allowed_exact
    minors_env = os.environ.get(_SGLANG_ALLOWED_MINORS_ENV, "").strip()
    if minors_env:
        minors = tuple(v.strip() for v in minors_env.split(",") if v.strip())
        return any(text == minor or text.startswith(f"{minor}.") for minor in minors)
    # Vendor manifest, when present, fully replaces the hardcoded default.
    if patches_dir is not None:
        manifest_versions = _load_supported_versions_from_manifest(patches_dir)
        if manifest_versions is not None:
            return text in manifest_versions
    return any(text == minor or text.startswith(f"{minor}.") for minor in _SGLANG_DEFAULT_ALLOWED_MINORS)


# Path within the TraceLens checkout that hosts the patch sets.
_PATCH_TREE_REL = ("examples", "custom_workflows", "inference_analysis")

# Both patch trees this module applies are SGLang's, and each ships its subdirs under this prefix.
_PATCH_SUBDIR_PREFIX = "sglang"

#: Suffix of the patch sets cut against SGLang's release branch rather than its
#: point-release tag. The two differ in the files a patch expects, so a build off
#: the release branch only applies cleanly against this variant.
_RELEASE_BRANCH_SUFFIX = "sgldev"


def _is_release_branch_build(version: str, source_root: Path | None = None) -> bool:
    """Whether the installed SGLang is a build off its release branch.

    setuptools_scm marks those with a ``.dev`` segment, a ``+g<sha>`` local part,
    or both (``0.5.18.dev20260825+g0c7ff19e3b``). SETUPTOOLS_SCM_PRETEND_VERSION
    flattens that to a bare release number, so a source checkout counts on the
    strength of its own git dir: a point release is never installed that way.
    """
    text = (version or "").strip()
    if text and (".dev" in text or re.search(r"\+g[0-9a-f]{7,}", text)):
        return True
    return source_root is not None and (source_root / ".git").exists()


def _versioned_patches_subdir_names(version: str, source_root: Path | None = None) -> list[str]:
    """Patch subdir names to try for an SGLang version, best match first."""
    text = (version or "").strip()
    if not text:
        return []
    # Strip dev/local suffixes so point-release tags still resolve.
    head = text.split("-", 1)[0].split("+", 1)[0]
    parts = head.split(".") if head else []
    # Keep the leading run of numeric components (``0.5.10.dev4`` -> 0_5_10).
    numeric: list[str] = []
    for p in parts:
        if p.isdigit():
            numeric.append(p)
        else:
            break
    if len(numeric) < 2:
        return []
    base = f"{_PATCH_SUBDIR_PREFIX}_" + "_".join(numeric)
    if _is_release_branch_build(version, source_root):
        return [f"{base}_{_RELEASE_BRANCH_SUFFIX}", base]
    return [base]


def _versioned_patches_subdir_name(version: str, source_root: Path | None = None) -> str | None:
    """The preferred patch subdir name for a version, or ``None`` if unparseable."""
    names = _versioned_patches_subdir_names(version, source_root)
    return names[0] if names else None


def _subdir_version_tuple(name: str) -> tuple[int, ...] | None:
    """Parse a ``sglang_0_5_11`` patch subdir name back into ``(0, 5, 11)``."""
    head = f"{_PATCH_SUBDIR_PREFIX}_"
    if not name.startswith(head):
        return None
    numeric: list[int] = []
    for part in name[len(head) :].split("_"):
        if part.isdigit():
            numeric.append(int(part))
        else:
            break
    return tuple(numeric) if len(numeric) >= 2 else None


def _subdir_is_release_branch(name: str) -> bool:
    """Whether a patch subdir name is the release-branch variant."""
    return name.endswith(f"_{_RELEASE_BRANCH_SUFFIX}")


def _nearest_not_newer(available: dict[tuple[int, ...], Path], running: tuple[int, ...]) -> Path | None:
    """The newest entry not newer than ``running``, preferring its own minor series."""
    if len(running) >= 2:
        same_minor = [vt for vt in available if vt[:2] == running[:2] and vt <= running]
        if same_minor:
            return available[max(same_minor)]
    not_higher = [vt for vt in available if vt <= running]
    if not_higher:
        return available[max(not_higher)]
    return None


def _resolve_versioned_patches_dir(
    patches_root: Path,
    version: str,
    source_root: Path | None = None,
) -> Path | None:
    """Locate the per-version patches dir for the running SGLang version."""

    def _qualifies(d: Path) -> bool:
        return any(d.glob("*.patch"))

    for subdir_name in _versioned_patches_subdir_names(version, source_root):
        candidate = patches_root / subdir_name
        if candidate.is_dir() and _qualifies(candidate):
            return candidate

    # Graceful fallback to the nearest not-newer patched subdir.
    if not patches_root.is_dir():
        return None
    running = _version_tuple(version)
    if running is None:
        return None
    # Keyed by version alone, the plain and release-branch subdirs of one version
    # collide and directory order decides the winner; prefer the variant matching
    # the running build instead.
    want_branch = _is_release_branch_build(version, source_root)
    available: dict[tuple[int, ...], Path] = {}
    for d in sorted(patches_root.iterdir()):
        if not d.is_dir() or not _qualifies(d):
            continue
        vt = _subdir_version_tuple(d.name)
        if not vt:
            continue
        if vt in available and _subdir_is_release_branch(d.name) is not want_branch:
            continue
        available[vt] = d
    return _nearest_not_newer(available, running)


# Public API


def ensure_vllm_patched_for_tracelens(checkout: Path | None = None) -> PatchedServer | None:
    """This host's copy of the installed vLLM with the TraceLens patch laid in, whose server accepts the TraceLens
    profiler flags; ``None`` when the patch cannot be laid.

    ``checkout`` is the vLLM source checkout the server imports ahead of the installed vLLM (the image source
    ``install.sh`` prepares, whose files a session changes): the copy then mirrors its ``vllm``, as the copy goes ahead of
    it and would otherwise put the installed files back in front of the server.
    """
    return ensure_patched_copy("vllm", checkout)


def ensure_sglang_patched_for_tracelens() -> PatchedServer | None:
    """This host's copy of the installed SGLang with the roofline set (< 0.5.18) or ``sglang_gc_patch`` (>= 0.5.18)
    laid in; ``None`` when it cannot be laid."""
    if resolve_sglang_shape_mode() == "sitecustomize":
        return _ensure_sglang_gc_patched()
    plan = _discover_sglang_plan()
    return None if plan is None else _ensure_patched(plan)


def ensure_patched_copy(package: str, checkout: Path | None = None) -> PatchedServer | None:
    """This host's patched copy of the installed package named ``package``, or of a vLLM ``checkout`` the server imports
    ahead of it (:func:`ensure_vllm_patched_for_tracelens`); ``None`` when it cannot be laid, or this module patches no
    package of that name."""
    plan = _plan_for(package, checkout)
    return None if plan is None else _ensure_patched(plan)


def lay_tracelens_patch(package: str, site: Path) -> bool:
    """Lay TraceLens's patch for the installed package named ``package`` in ``site``, the directory holding a test's own
    mirror of that package (its farm); whether the mirror carries the patch.

    A link of the mirror to an installed file or directory the patch writes is first replaced by a copy of its own, so
    nothing is written but the mirror."""
    plan = _plan_for(package)
    return plan is not None and _lay_in(plan, site)


def _vllm_plan(checkout: Path | None = None) -> _PatchPlan | None:
    """The plan for the installed vLLM, laid in the ``vllm`` of ``checkout`` instead when it holds one: ``install.sh``
    activates a checkout only at the installed wheel's commit, so the wheel's version picks its patch."""
    install = _discover_vllm_install()
    if install is not None and checkout is not None and (checkout / "vllm" / "__init__.py").is_file():
        install = (install[0], checkout)
    return None if install is None else _discover_vllm_plan(install=install)


def _sglang_plan() -> _PatchPlan | None:
    """The plan for the installed SGLang: the roofline set, or ``sglang_gc_patch`` in sitecustomize mode."""
    if resolve_sglang_shape_mode() != "sitecustomize":
        return _discover_sglang_plan()
    version = _detect_installed_sglang_version()
    return None if version is None else _discover_sglang_gc_plan(version)


def _plan_for(package: str, checkout: Path | None = None) -> _PatchPlan | None:
    """The plan for the installed package named ``package``; ``None`` for a package this module does not patch."""
    if package == "vllm":
        return _vllm_plan(checkout)
    if package == "sglang":
        return _sglang_plan()
    return None


# SGLang shape-discovery gate: >= 0.5.18 uses the no-patch TraceLens tool
# (PYTHONPATH + sitecustomize + TRACELENS_SHAPE_DISCOVERY); older uses git-apply.
_SGLANG_SITECUSTOMIZE_MIN_VERSION: tuple[int, ...] = (0, 5, 18)
# No-patch tool location, relative to TRACELENS_ROOT.
_KERNEL_SHAPE_TOOL_REL: tuple[str, ...] = ("TraceLens", "TraceUtils", "kernel_shape_tool")


def sglang_shape_mode(version: str) -> str:
    """Return the shape-discovery mechanism for an SGLang version.

    ``"sitecustomize"`` (>= 0.5.18) uses the no-patch TraceLens tool;
    ``"patched"`` (< 0.5.18) uses the legacy ``git apply`` flow.
    """
    vt = _version_tuple(version)
    if vt is None:
        # Unparseable version: keep the safe legacy mechanism.
        return "patched"
    return "sitecustomize" if vt >= _SGLANG_SITECUSTOMIZE_MIN_VERSION else "patched"


def kernel_shape_tool_dir() -> Path | None:
    """Resolve the no-patch ``kernel_shape_tool`` dir under TRACELENS_ROOT, or None."""
    root = _resolve_tracelens_root()
    if root is None:
        return None
    tool = root.joinpath(*_KERNEL_SHAPE_TOOL_REL)
    return tool if tool.is_dir() else None


def _detect_installed_sglang_version() -> str | None:
    """Return the locally-installed SGLang version, or ``None`` if unimportable."""
    try:
        import sglang  # type: ignore[import-not-found]
    except (ImportError, OSError):
        return None
    return (getattr(sglang, "__version__", "") or "").strip() or None


def resolve_sglang_shape_mode() -> str:
    """Resolve the SGLang shape mode from the local install, else the MN version pin.

    The version comes from the locally-installed SGLang (single-node / sandbox)
    or, when SGLang is not importable in the controller (multi-node),
    ``HYPERLOOM_SGLANG_VERSION_PIN``.
    Falls back to ``"patched"`` (legacy) when the version cannot be determined.
    """
    version = _detect_installed_sglang_version() or os.environ.get("HYPERLOOM_SGLANG_VERSION_PIN", "").strip()
    return sglang_shape_mode(version)


def resolve_sglang_patch_set() -> str:
    """Patch set the multi-node pods must apply: ``graph-capture`` in sitecustomize mode, else ``roofline``.

    Resolved on the controller and passed to every pod, which cannot see the
    controller's version pin and would otherwise decide alone.
    """
    return "graph-capture" if resolve_sglang_shape_mode() == "sitecustomize" else "roofline"


def _gc_patch_deferred_to_pods() -> bool:
    """True on a multi-node controller, where the pod fan-out applies the patch."""
    from ._multi_node_env import is_multi_node

    return is_multi_node()


def _ensure_sglang_gc_patched() -> PatchedServer | None:
    """This host's copy of the installed SGLang with ``sglang_gc_patch`` laid in."""
    version = _detect_installed_sglang_version()
    if version is None:
        if _gc_patch_deferred_to_pods():
            log.info(
                "_server_patcher: sglang not importable on this controller; graph-capture patch is applied on the pods"
            )
            return PatchedServer(None)
        log.warning("_server_patcher: sglang not importable; skip graph-capture patch")
        return None
    plan = _discover_sglang_gc_plan(version)
    return None if plan is None else _ensure_patched(plan)


def ensure_xdit_patched_for_tracelens() -> bool:
    """Apply TraceLens profiling patches to the xDiT installation.

    xDiT is an editable install at ``/app/xDiT`` (no pip version);
    version detection matches ``git rev-parse HEAD`` against known
    image SHAs.  Unknown SHAs are skipped (no patch attempted).
    The nearest-lower fallback applies only when a known version
    has no exact patch file.
    """
    plan = _discover_xdit_plan()
    if plan is None:
        return False
    return _ensure_patched_in_place(plan)


def _resolve_xdit_patch_file(patches_dir: Path, version: str) -> Path | None:
    """Pick the TraceLens xDiT patch for ``version`` with nearest-lower fallback."""
    exact = patches_dir / f"config_xdit_v{version}.patch"
    if exact.is_file():
        return exact
    if not patches_dir.is_dir():
        return None
    available: dict[tuple[int, ...], Path] = {}
    for p in patches_dir.glob("config_xdit_v*.patch"):
        m = _XDIT_PATCH_RE.match(p.name)
        if not m:
            continue
        vt = _version_tuple(m.group(1))
        if vt:
            available[vt] = p
    if not available:
        return None
    running = _version_tuple(version)
    if running is None:
        return None
    not_higher = [vt for vt in available if vt <= running]
    if not_higher:
        return available[max(not_higher)]
    return None


def _discover_xdit_plan() -> _PatchPlan | None:
    """Build the xDiT patch plan for the installed xDiT version.

    Only attempts patching when the HEAD SHA matches a known image.
    Unknown SHAs are skipped — no patch is attempted.
    """
    source_root = Path(os.environ.get("XDIT_PATH", "").strip() or str(_xdit_default_source_root()))
    if not source_root.is_dir():
        log.info("_server_patcher: xDiT source root %s not found; skip patch", source_root)
        return None

    sha = _git_head_sha(source_root)
    if not sha:
        log.info("_server_patcher: cannot read xDiT HEAD SHA at %s; skip patch", source_root)
        return None

    version = _XDIT_SHA_TO_VERSION.get(sha)
    if version is None:
        log.warning(
            "_server_patcher: xDiT SHA %s not in known image table; capture-phase profiling will be unavailable",
            sha[:12],
        )
        return None

    tracelens_root = _resolve_tracelens_root()
    if tracelens_root is None:
        log.info("_server_patcher: TRACELENS_ROOT unset/missing — skip xDiT patch")
        return None

    patches_dir = _patch_tree(tracelens_root, _XDIT_PATCHES_LEAF)
    patch_file = _resolve_xdit_patch_file(patches_dir, version)
    if patch_file is None:
        log.warning(
            "_server_patcher: no TraceLens patch for xDiT %s "
            "(searched %s); capture-phase profiling will be unavailable",
            version,
            patches_dir,
        )
        return None

    sentinel = source_root.joinpath(*_XDIT_SENTINEL_REL)
    if not sentinel.is_file():
        log.info(
            "_server_patcher: xDiT layout unexpected (no %s); skip patch",
            sentinel,
        )
        return None

    return _PatchPlan(
        framework="xdit",
        version=version,
        package=source_root / _XDIT_SENTINEL_REL[0],
        patches=(patch_file,),
        sentinel_file=sentinel,
        sentinel_text=_XDIT_SENTINEL_MARKERS,
    )


# Plan discovery


#: Annotation-pipeline sentinels as ``(path under the sglang package, markers)``.
#: A release that does not patch a given file simply drops out of the set.
_SGLANG_ANNOTATION_SENTINELS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("srt", "managers", "scheduler.py"), ("detailed_annotations",)),
    (
        ("srt", "managers", "scheduler_components", "profiler_manager.py"),
        ("detailed_annotations", "shape_discovery", "kernel_shape_profiler"),
    ),
    (("srt", "managers", "io_struct.py"), ("shape_discovery", "detailed_annotations")),
    (("srt", "model_executor", "step_span_utils.py"), ("detailed_annotations",)),
)


def _patch_target_paths(patches: Sequence[Path]) -> frozenset[str]:
    """Return the slash-joined paths a patch set writes to."""
    targets: set[str] = set()
    for patch in patches:
        # An unreadable patch would silently shrink the sentinel set, which is the detection hole this derivation
        # exists to close.
        text = patch.read_text(encoding="utf-8", errors="replace")
        targets.update(change.path.replace("\\", "/") for change in parse_unified_diff(text) if not change.is_deleted)
    return frozenset(targets)


def _patch_set_writes(targets: frozenset[str], parts: tuple[str, ...]) -> bool:
    """Return whether any patch target ends with this package-relative path."""
    suffix = "/".join(("sglang", *parts))
    return any(target.endswith(suffix) for target in targets)


def _resolve_tracelens_root() -> Path | None:
    """Resolve :func:`paths.tracelens_root`; ``None`` when it is not on disk."""
    root = paths.tracelens_root()
    return root if root.is_dir() else None


def _patch_tree(tracelens_root: Path, leaf: str) -> Path:
    """Build a path under the TraceLens inference-analysis patch tree."""
    return tracelens_root.joinpath(*_PATCH_TREE_REL, leaf)


_VLLM_PATCH_RE = re.compile(r"^config_vllm_v(\d+(?:\.\d+)*)\.patch$")


def _version_tuple(v: str) -> tuple[int, ...] | None:
    """Parse the leading dotted-numeric run of a version into a tuple."""
    m = re.match(r"^\s*v?(\d+(?:\.\d+)*)", str(v or ""))
    if not m:
        return None
    return tuple(int(part) for part in m.group(1).split("."))


def _resolve_vllm_patch_file(patches_dir: Path, version: str) -> Path | None:
    """Pick the TraceLens vLLM patch for ``version`` with graceful fallback."""
    exact = patches_dir / f"config_vllm_v{version}.patch"
    if exact.is_file():
        return exact
    if not patches_dir.is_dir():
        return None
    available: dict[tuple[int, ...], Path] = {}
    for p in patches_dir.glob("config_vllm_v*.patch"):
        m = _VLLM_PATCH_RE.match(p.name)
        if not m:
            continue
        vt = _version_tuple(m.group(1))
        if vt:
            available[vt] = p
    if not available:
        return None
    env_pin = os.environ.get(_VLLM_EXACT_VERSIONS_ENV, "").strip()
    if env_pin:
        for token in (t.strip() for t in env_pin.split(",")):
            vt = _version_tuple(token)
            if vt and vt in available:
                return available[vt]
    running = _version_tuple(version)
    return _nearest_not_newer(available, running) if running else None


def _probe_isolated_vllm() -> tuple[str, Path] | None:
    """Resolve version and site-packages for an isolated-venv vLLM install."""
    venv_root = os.environ.get("VLLM_VENV_ROOT", "").strip()
    if not venv_root:
        return None
    lib = Path(venv_root) / "lib"
    if not lib.is_dir():
        return None

    site: Path | None = None
    for match in sorted(lib.glob("python*/site-packages/vllm")):
        if match.is_dir():
            site = match.parent
            break

    version = ""
    imported_root: Path | None = None
    vllm_python = os.environ.get("VLLM_PYTHON", "").strip() or str(Path(venv_root) / "bin" / "python")
    if Path(vllm_python).exists():
        try:
            proc = subprocess.run(
                [vllm_python, "-c", "import vllm; print(vllm.__version__); print(vllm.__file__)"],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
            )
            lines = proc.stdout.strip().splitlines()
            if proc.returncode == 0 and len(lines) >= 2:
                version = lines[-2].strip()
                imported_root = Path(lines[-1].strip()).resolve().parent.parent
        except (OSError, subprocess.SubprocessError) as e:
            log.info("_server_patcher: VLLM_PYTHON version probe failed (%s)", e)

    # A source build installs vLLM editable, so site-packages holds only an egg-link or .pth and the package lives
    # in the checkout; only the venv's own interpreter can say where that is.
    if site is None:
        site = imported_root
    if site is None:
        return None

    if not version:
        for dist in sorted(site.glob("vllm-*.dist-info")):
            name = dist.name
            if name.endswith(".dist-info"):
                version = name.removeprefix("vllm-").removesuffix(".dist-info")
                break
    if not version:
        return None
    return version, site


def _vllm_graph_capture_sentinel(version: str, install_root: Path) -> tuple[tuple[Path, tuple[str, ...]], ...]:
    """Extra sentinels proving the graph-capture patch landed, for 0.26+."""
    vt = _version_tuple(version)
    if vt is None or vt < _VLLM_GRAPH_CAPTURE_MIN_VERSION:
        return ()
    return ((install_root.joinpath(*_VLLM_GRAPH_CAPTURE_SENTINEL), _VLLM_GRAPH_CAPTURE_MARKERS),)


def _discover_vllm_install() -> tuple[str, Path] | None:
    """Resolve ``(version, install_root)`` for the vLLM this run will profile."""
    version = ""
    install_root: Path | None = None
    try:
        import vllm  # type: ignore[import-not-found]

        version = (getattr(vllm, "__version__", "") or "").strip()
        install_root = Path(vllm.__file__).resolve().parent.parent
    except (ImportError, OSError, RuntimeError, TypeError) as e:
        probed = _probe_isolated_vllm()
        if probed is None:
            log.info(
                "_server_patcher: vllm not importable (%s) and no isolated $VLLM_VENV_ROOT vllm; skip patch",
                e,
            )
            return None
        version, install_root = probed
        log.info(
            "_server_patcher: main-process vllm unimportable; using isolated vllm %s at %s",
            version,
            install_root,
        )

    if not version:
        log.info("_server_patcher: vllm has no __version__; skip patch")
        return None
    return version, install_root


def _discover_vllm_plan(*, install: tuple[str, Path]) -> _PatchPlan | None:
    """Build the vLLM patch plan for the installed vLLM version."""
    tracelens_root = _resolve_tracelens_root()
    if tracelens_root is None:
        log.info("_server_patcher: TRACELENS_ROOT (public) unset/missing — skip vLLM patch")
        return None

    version, install_root = install

    patches_dir = _patch_tree(tracelens_root, "vllm_patches")
    patch_file = _resolve_vllm_patch_file(patches_dir, version)
    if patch_file is None:
        log.warning(
            "_server_patcher: no TraceLens patch for vLLM %s "
            "(searched %s incl. minor/nearest-lower fallback); "
            "TraceLens annotations will be unavailable",
            version,
            patches_dir,
        )
        return None
    if patch_file.name != f"config_vllm_v{version}.patch":
        log.info(
            "_server_patcher: no exact vLLM patch for %s; using nearest %s "
            "(backward-compatible; git apply --check still guards)",
            version,
            patch_file.name,
        )

    # Apply root for the ``a/vllm/...`` prefix is site-packages (resolved above from the main venv or the isolated
    # $VLLM_VENV_ROOT).
    sentinel = install_root / "vllm" / "config" / "profiler.py"
    if not sentinel.is_file():
        log.info(
            "_server_patcher: vLLM install layout unexpected (no %s); skip patch",
            sentinel,
        )
        return None

    # Requiring both substrings collapses the false-positive surface to ~zero.
    return _PatchPlan(
        framework="vllm",
        version=version,
        package=install_root / "vllm",
        patches=(patch_file,),
        sentinel_file=sentinel,
        sentinel_text=_VLLM_PROFILER_SENTINELS,
        extra_sentinels=_vllm_graph_capture_sentinel(version, install_root),
    )


#: Basename substrings marking SGLang patches that are non-critical to kernel
#: shape profiling. These patch eagle speculative-decoding cuda-graph runners,
#: unrelated to the shape-profiler pipeline; when one fails ``git apply --check``
#: against a point-release-shifted SGLang it is skipped instead of aborting the
#: whole atomic set (which would also roll back the critical profiler patch).
_SGLANG_OPTIONAL_PATCH_MARKERS = ("eagle",)


def _discover_sglang_plan() -> _PatchPlan | None:
    """Build the SGLang patch plan for the installed SGLang version."""
    tracelens_root = _resolve_tracelens_root()
    if tracelens_root is None:
        log.warning(
            "_server_patcher: TRACELENS_ROOT (public) unset/missing — skip SGLang patch "
            "(kernel shape profiling will be unavailable)"
        )
        return None

    try:
        import sglang  # type: ignore[import-not-found]
    except (ImportError, OSError, RuntimeError) as e:
        log.warning("_server_patcher: sglang not importable (%s); skip patch", e)
        return None

    version = (getattr(sglang, "__version__", "") or "").strip()

    # Resolve the patches dir before the version check so the gate can consult the TraceLens-shipped manifest.
    patches_root = _patch_tree(tracelens_root, "sglang_roofline_patches")
    if not patches_root.is_dir():
        log.warning(
            "_server_patcher: SGLang patches root missing (%s); skip — kernel shape profiling will be unavailable",
            patches_root,
        )
        return None

    # Resolved before the patch set so its git dir can tell a source checkout
    # from a point release when the version string cannot.
    sglang_module = Path(sglang.__file__).resolve()
    source_root = _sglang_source_root(sglang_module)
    if source_root is None:
        return None

    # Per-version subdir layout required.
    patches_dir = _resolve_versioned_patches_dir(patches_root, version, source_root)
    if patches_dir is None:
        log.warning(
            "_server_patcher: no SGLang patches found under %s/%s/ for "
            "version %s; upgrade TraceLens to Hyperloom_integration_v0.3.1+ — "
            "kernel shape profiling will be unavailable",
            patches_root,
            _versioned_patches_subdir_name(version, source_root) or "<unknown>",
            version,
        )
        return None

    if not _version_accepted(version, patches_dir=patches_dir):
        log.warning(
            "_server_patcher: SGLang %s not in supported version list "
            "(consulted: $HYPERLOOM_SGLANG_PATCH_EXACT_VERSIONS, "
            "$HYPERLOOM_SGLANG_PATCH_ALLOWED_MINORS, %s/SUPPORTED_VERSIONS, "
            "then built-in minor allowlist %s); skip — kernel shape profiling "
            "will be unavailable",
            version,
            patches_dir,
            _SGLANG_DEFAULT_ALLOWED_MINORS,
        )
        return None
    patches = tuple(sorted(patches_dir.glob("*.patch")))
    if not patches:
        log.warning("_server_patcher: SGLang patches directory empty; skip")
        return None

    # Sentinel: the kernel_shape_profiler patch creates a new file at ``sglang/srt/utils/kernel_shape_profiler.py`` in
    # both layouts.
    sentinel = sglang_module.parent / "srt" / "utils" / "kernel_shape_profiler.py"
    sglang_pkg = sglang_module.parent
    # Also verify the annotation pipeline so a partial apply (main sentinel present but annotations missing) is still
    # detected: scheduler callback -> profiler_manager toggle -> io_struct request fields -> step-span aggregates.
    written = _patch_target_paths(patches)
    extra_sentinels: tuple[tuple[Path, tuple[str, ...]], ...] = tuple(
        (sglang_pkg.joinpath(*parts), markers)
        for parts, markers in _SGLANG_ANNOTATION_SENTINELS
        if _patch_set_writes(written, parts)
    )
    optional_patches = frozenset(
        p.name for p in patches if any(m in p.name.lower() for m in _SGLANG_OPTIONAL_PATCH_MARKERS)
    )
    return _PatchPlan(
        framework="sglang",
        version=version,
        package=sglang_pkg,
        patches=patches,
        sentinel_file=sentinel,
        # Sentinel file alone is insufficient; extra_sentinels require the annotation pipeline.
        sentinel_text=("kernel_shape_profiler",),
        extra_sentinels=extra_sentinels,
        apply_strip=_SGLANG_PATCH_STRIP,
        optional_patches=optional_patches,
    )


#: TraceLens cuts its SGLang patches in the checkout (``a/python/sglang/...``); two components off, their paths are
#: relative to the directory holding ``sglang``, in a checkout and an installed wheel alike.
_SGLANG_PATCH_STRIP = 2


def _sglang_source_root(sglang_module: Path) -> Path | None:
    """The checkout the active SGLang is installed from, or its package directory when it is a wheel; ``None`` for an
    unexpected layout."""
    if sglang_module.parent.parent.name == "python":
        return sglang_module.parent.parent.parent
    sglang_dir = sglang_module.parent
    if sglang_dir.name == "sglang":
        return sglang_dir
    log.info(
        "_server_patcher: SGLang install at %s has unexpected layout (parent dir name=%r); skip patching",
        sglang_module,
        sglang_dir.name,
    )
    return None


_SGLANG_GC_SENTINELS: tuple[str, ...] = (
    "_set_profile_trace_tag",
    "_profile_runner_name",
)


def _discover_sglang_gc_plan(version: str) -> _PatchPlan | None:
    """Build the ``sglang_gc_patch`` plan for an exact ``sglang_<X_Y_Z>`` dir; strict ``git apply`` only.

    No nearest-version fallback and no fuzzy apply: the patch adds an unconditional
    ``_set_profile_trace_tag`` call to the capture loop, so a drifted apply could
    break CUDA-graph capture, and newer releases may already carry the upstream fix.
    """
    tracelens_root = _resolve_tracelens_root()
    if tracelens_root is None:
        log.warning(
            "_server_patcher: TRACELENS_ROOT unset/missing — skip SGLang graph-capture "
            "patch (per-batch-size CUDA-graph capture can IndexError on multi-variant models)"
        )
        return None

    try:
        import sglang  # type: ignore[import-not-found]
    except (ImportError, OSError, RuntimeError) as e:
        log.warning("_server_patcher: sglang not importable (%s); skip graph-capture patch", e)
        return None
    if not getattr(sglang, "__file__", None):
        log.warning("_server_patcher: sglang.__file__ is unset; skip graph-capture patch")
        return None

    patches_root = _patch_tree(tracelens_root, "sglang_gc_patch")
    if not patches_root.is_dir():
        log.warning(
            "_server_patcher: SGLang graph-capture patches root missing (%s); skip",
            patches_root,
        )
        return None

    sglang_module = Path(sglang.__file__).resolve()
    source_root = _sglang_source_root(sglang_module)
    if source_root is None:
        return None

    candidates = [patches_root / name for name in _versioned_patches_subdir_names(version, source_root)]
    patches_dir = next((d for d in candidates if d.is_dir() and any(d.glob("*.patch"))), None)
    if patches_dir is None:
        log.warning(
            "_server_patcher: no sglang_gc_patch set for SGLang %s under %s (exact version dir required, "
            "tried %s); skip — per-batch-size CUDA-graph capture can IndexError on multi-variant models",
            version,
            patches_root,
            [d.name for d in candidates] or "<unparseable version>",
        )
        return None
    patches = tuple(sorted(patches_dir.glob("*.patch")))

    sentinel = sglang_module.parent / "srt" / "model_executor" / "runner" / "decode_cuda_graph_runner.py"
    if not sentinel.is_file():
        log.warning(
            "_server_patcher: SGLang install layout unexpected (no %s); skip graph-capture patch",
            sentinel,
        )
        return None

    return _PatchPlan(
        framework="sglang-gc",
        version=version,
        package=sglang_module.parent,
        patches=patches,
        sentinel_file=sentinel,
        sentinel_text=_SGLANG_GC_SENTINELS,
        apply_strip=_SGLANG_PATCH_STRIP,
        allow_fuzzy=False,
    )


__all__ = [
    "ensure_patched_copy",
    "ensure_vllm_patched_for_tracelens",
    "ensure_sglang_patched_for_tracelens",
    "ensure_xdit_patched_for_tracelens",
    "lay_tracelens_patch",
]
