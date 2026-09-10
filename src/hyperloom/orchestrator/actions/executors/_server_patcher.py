# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Idempotent run-time patcher for vLLM and SGLang server installs."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from ._file_lock import best_effort_file_lock

log = logging.getLogger(__name__)


# System-wide lock file.
_LOCK_PATH = str(Path(tempfile.gettempdir()) / "hyperloom_server_patcher.lock")

# Per-``git`` invocation timeout (defensive against hung NFS).
_GIT_TIMEOUT_SEC = 30

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


def _versioned_patches_subdir_name(version: str) -> str | None:
    """Map an SGLang version to the per-version patch subdir name (e.g."""
    text = (version or "").strip()
    if not text:
        return None
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
        return None
    return f"{_PATCH_SUBDIR_PREFIX}_" + "_".join(numeric)


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


def _resolve_versioned_patches_dir(
    patches_root: Path,
    version: str,
) -> Path | None:
    """Locate the per-version patches dir for the running SGLang version."""

    def _qualifies(d: Path) -> bool:
        return any(d.glob("*.patch"))

    subdir_name = _versioned_patches_subdir_name(version)
    if subdir_name is not None:
        candidate = patches_root / subdir_name
        if candidate.is_dir() and _qualifies(candidate):
            return candidate

    # Graceful fallback to the nearest not-newer patched subdir.
    if not patches_root.is_dir():
        return None
    running = _version_tuple(version)
    if running is None:
        return None
    available: dict[tuple[int, ...], Path] = {}
    for d in patches_root.iterdir():
        if not d.is_dir() or not _qualifies(d):
            continue
        vt = _subdir_version_tuple(d.name)
        if vt:
            available[vt] = d
    if not available:
        return None
    if len(running) >= 2:
        same_minor = [vt for vt in available if vt[:2] == running[:2] and vt <= running]
        if same_minor:
            return available[max(same_minor)]
    not_higher = [vt for vt in available if vt <= running]
    if not_higher:
        return available[max(not_higher)]
    return None


# Public API


def ensure_vllm_patched_for_tracelens(
    tracelens_root: Path | str | None = None,
) -> bool:
    """Ensure the installed vLLM accepts the TraceLens profiler flags."""
    install = _discover_vllm_install()
    if install is None:
        return False
    plan = _discover_vllm_plan(tracelens_root, install=install)
    if plan is None:
        return False
    return _ensure_patched(plan)


def ensure_sglang_patched_for_tracelens(
    tracelens_root: Path | str | None = None,
) -> bool:
    """SGLang counterpart of :func:`ensure_vllm_patched_for_tracelens`."""
    plan = _discover_sglang_plan(tracelens_root)
    if plan is None:
        return False
    return _ensure_patched(plan)


# SGLang shape-discovery mechanism gate.
#
# From 0.5.18 the kernel shape profiler is delivered as the no-patch TraceLens
# tool (PYTHONPATH + sitecustomize + TRACELENS_SHAPE_DISCOVERY) instead of a
# ``git apply`` of the shape-profiler patches; ``detailed_annotations`` is
# upstream by then, so dropping the patch loses only shapes (the tool restores
# them). Older versions keep the patch mechanism.
_SGLANG_SITECUSTOMIZE_MIN_VERSION: tuple[int, ...] = (0, 5, 18)
_SGLANG_SHAPE_MODE_ENV = "HYPERLOOM_SGLANG_SHAPE_MODE"
# The no-patch kernel shape tool lives in the TraceLens package under
# ``TraceLens/TraceUtils/kernel_shape_tool`` (relative to TRACELENS_ROOT).
_KERNEL_SHAPE_TOOL_REL: tuple[str, ...] = ("TraceLens", "TraceUtils", "kernel_shape_tool")


def sglang_shape_mode(version: str) -> str:
    """Return the shape-discovery mechanism for an SGLang version.

    ``"sitecustomize"`` (>= 0.5.18) uses the no-patch TraceLens tool;
    ``"patched"`` (< 0.5.18) uses the legacy ``git apply`` flow.
    ``HYPERLOOM_SGLANG_SHAPE_MODE=patch|sitecustomize`` overrides the gate
    (``auto`` / unset = version-based).
    """
    override = os.environ.get(_SGLANG_SHAPE_MODE_ENV, "auto").strip().lower()
    if override in {"patch", "patched"}:
        return "patched"
    if override == "sitecustomize":
        return "sitecustomize"
    vt = _version_tuple(version)
    if vt is None:
        # Unparseable version: keep the safe legacy mechanism.
        return "patched"
    return "sitecustomize" if vt >= _SGLANG_SITECUSTOMIZE_MIN_VERSION else "patched"


def kernel_shape_tool_dir(tracelens_root: Path | str | None = None) -> Path | None:
    """Resolve the no-patch ``kernel_shape_tool`` dir under TRACELENS_ROOT, or None."""
    root = _resolve_tracelens_root(tracelens_root)
    if root is None:
        return None
    tool = root.joinpath(*_KERNEL_SHAPE_TOOL_REL)
    return tool if tool.is_dir() else None


def _detect_installed_sglang_version() -> str | None:
    """Return the locally-installed SGLang version, or ``None`` if unimportable."""
    try:
        import sglang  # type: ignore  # noqa: I001 - runtime probe
    except Exception:  # noqa: BLE001
        return None
    return (getattr(sglang, "__version__", "") or "").strip() or None


def resolve_sglang_shape_mode() -> str:
    """Resolve the SGLang shape mode from override -> local install -> MN version pin.

    ``HYPERLOOM_SGLANG_SHAPE_MODE`` wins; otherwise the version comes from the
    locally-installed SGLang (single-node / sandbox) or, when SGLang is not
    importable in the controller (multi-node), ``HYPERLOOM_SGLANG_VERSION_PIN``.
    Falls back to ``"patched"`` (legacy) when the version cannot be determined.
    """
    version = _detect_installed_sglang_version() or os.environ.get(
        "HYPERLOOM_SGLANG_VERSION_PIN", ""
    ).strip()
    return sglang_shape_mode(version)


def ensure_sglang_patched_for_ck_blockscale(
    kernelforge_root: Path | str | None = None,
) -> bool:
    """Apply the KernelForge fp8 block-scale CK-routing patch to SGLang."""
    plan = _discover_sglang_ck_plan(kernelforge_root)
    if plan is None:
        return False
    return _ensure_patched(plan)


# Plan discovery


@dataclass(frozen=True)
class _PatchPlan:
    """All information needed to apply (or verify) a patch set."""

    framework: str
    version: str
    apply_root: Path  # cwd for ``git apply``
    patches: tuple[Path, ...]  # in apply order
    sentinel_file: Path  # file we grep to detect "already patched"
    # Substrings that must ALL be present in ``sentinel_file`` to count as patched; multi-element tuples lower the
    # false-positive risk.
    sentinel_text: tuple[str, ...]
    # Extra file-local markers required for a multi-file set to count as complete (SGLang can partial-patch while
    # leaving the historical sentinel).
    extra_sentinels: tuple[tuple[Path, tuple[str, ...]], ...] = ()
    # Per-plan ``-p<N>`` strip count: ``-p1`` for editable / vLLM, ``-p3`` for wheel SGLang.
    apply_strip: int = 1
    # Patch names that may fail ``git apply --check`` and be skipped instead of rolling back the whole atomic set
    # (e.g. eagle-draft patches whose context drifted across same-version different-commit sglang builds).
    optional_patches: frozenset[str] = frozenset()


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
        for line in text.splitlines():
            if not line.startswith("+++ "):
                continue
            target = line[4:].split("\t", 1)[0].strip()
            if target and target != "/dev/null":
                targets.add(target.replace("\\", "/"))
    return frozenset(targets)


def _patch_set_writes(targets: frozenset[str], parts: tuple[str, ...]) -> bool:
    """Return whether any patch target ends with this package-relative path."""
    suffix = "/".join(("sglang", *parts))
    return any(target.endswith(suffix) for target in targets)


def _resolve_tracelens_root(arg: Path | str | None) -> Path | None:
    """Resolve TRACELENS_ROOT from arg → env → None; fail-soft when unset or missing on disk."""
    if arg:
        root = Path(arg)
    else:
        env = os.environ.get("TRACELENS_ROOT", "").strip()
        if not env:
            return None
        root = Path(env)
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
    if running is None:
        return None
    if len(running) >= 2:
        same_minor = [vt for vt in available if vt[:2] == running[:2] and vt <= running]
        if same_minor:
            return available[max(same_minor)]
    not_higher = [vt for vt in available if vt <= running]
    if not_higher:
        return available[max(not_higher)]
    return None


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
    if site is None:
        return None

    version = ""
    vllm_python = os.environ.get("VLLM_PYTHON", "").strip()
    if vllm_python and Path(vllm_python).exists():
        try:
            proc = subprocess.run(
                [vllm_python, "-c", "import vllm; print(vllm.__version__)"],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
            )
            if proc.returncode == 0:
                version = proc.stdout.strip()
        except (OSError, subprocess.SubprocessError) as e:
            log.info("_server_patcher: VLLM_PYTHON version probe failed (%s)", e)

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
        import vllm  # type: ignore  # noqa: I001 - runtime probe

        version = (getattr(vllm, "__version__", "") or "").strip()
        install_root = Path(vllm.__file__).resolve().parent.parent
    except Exception as e:  # noqa: BLE001 - any import failure → isolated-venv fallback
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


def _discover_vllm_plan(
    arg: Path | str | None,
    *,
    install: tuple[str, Path] | None = None,
) -> _PatchPlan | None:
    """Build the vLLM patch plan for the installed vLLM version."""
    tracelens_root = _resolve_tracelens_root(arg)
    if tracelens_root is None:
        log.info("_server_patcher: TRACELENS_ROOT (public) unset/missing — skip vLLM patch")
        return None

    resolved = install if install is not None else _discover_vllm_install()
    if resolved is None:
        return None
    version, install_root = resolved

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
        apply_root=install_root,
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


def _discover_sglang_plan(arg: Path | str | None) -> _PatchPlan | None:
    """Build the SGLang patch plan for the installed SGLang version."""
    tracelens_root = _resolve_tracelens_root(arg)
    if tracelens_root is None:
        log.warning(
            "_server_patcher: TRACELENS_ROOT (public) unset/missing — skip SGLang patch "
            "(kernel shape profiling will be unavailable)"
        )
        return None

    try:
        import sglang  # type: ignore  # noqa: I001 - runtime probe
    except Exception as e:  # noqa: BLE001
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

    # Per-version subdir layout required.
    patches_dir = _resolve_versioned_patches_dir(patches_root, version)
    if patches_dir is None:
        log.warning(
            "_server_patcher: no SGLang patches found under %s/%s/ for "
            "version %s; upgrade TraceLens to Hyperloom_integration_v0.3.1+ — "
            "kernel shape profiling will be unavailable",
            patches_root,
            _versioned_patches_subdir_name(version) or "<unknown>",
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

    # Support both editable and wheel layouts (see _resolve_sglang_apply_root).
    sglang_module = Path(sglang.__file__).resolve()
    apply_resolution = _resolve_sglang_apply_root(sglang_module)
    if apply_resolution is None:
        return None
    apply_root, apply_strip = apply_resolution

    filtered_patches: list[Path] = list(patches)

    # Sentinel: the kernel_shape_profiler patch creates a new file at ``sglang/srt/utils/kernel_shape_profiler.py`` in
    # both layouts.
    sentinel = sglang_module.parent / "srt" / "utils" / "kernel_shape_profiler.py"
    sglang_pkg = sglang_module.parent
    # Also verify the annotation pipeline so a partial apply (main sentinel present but annotations missing) is still
    # detected: scheduler callback -> profiler_manager toggle -> io_struct request fields -> step-span aggregates.
    written = _patch_target_paths(filtered_patches)
    extra_sentinels: tuple[tuple[Path, tuple[str, ...]], ...] = tuple(
        (sglang_pkg.joinpath(*parts), markers)
        for parts, markers in _SGLANG_ANNOTATION_SENTINELS
        if _patch_set_writes(written, parts)
    )
    optional_patches = frozenset(
        p.name for p in filtered_patches if any(m in p.name.lower() for m in _SGLANG_OPTIONAL_PATCH_MARKERS)
    )
    return _PatchPlan(
        framework="sglang",
        version=version,
        apply_root=apply_root,
        patches=tuple(filtered_patches),
        sentinel_file=sentinel,
        # Sentinel file alone is insufficient; extra_sentinels require the annotation pipeline.
        sentinel_text=("kernel_shape_profiler",),
        extra_sentinels=extra_sentinels,
        apply_strip=apply_strip,
        optional_patches=optional_patches,
    )


def _resolve_sglang_apply_root(sglang_module: Path) -> tuple[Path, int] | None:
    """Pick ``(apply_root, strip_count)`` for the active SGLang install."""
    if sglang_module.parent.parent.name == "python":
        return sglang_module.parent.parent.parent, 1
    sglang_dir = sglang_module.parent
    if sglang_dir.name == "sglang":
        return sglang_dir, 3
    log.info(
        "_server_patcher: SGLang install at %s has unexpected layout (parent dir name=%r); skip patching",
        sglang_module,
        sglang_dir.name,
    )
    return None


# CK fp8 block-scale routing markers added to ``fp8_utils.py`` by the KernelForge-owned patch; all three must be
# present to count as patched.
_SGLANG_CK_BLOCKSCALE_SENTINELS: tuple[str, ...] = (
    "_fp8_blockscale_ck_max_m",
    "SGLANG_FP8_BLOCKSCALE_CK_MAX_M",
    "ck_gemm_a8w8_blockscale",
)


def _resolve_serving_patches_root(arg: Path | str | None) -> Path | None:
    """Resolve KernelForge's ``serving_patches`` tree; fail-soft."""
    if arg:
        candidate = Path(arg) / "serving_patches"
        if candidate.is_dir():
            log.warning(
                "_server_patcher: patching SGLang from an explicit serving_patches tree at %s, "
                "not the one packaged with kernelforge",
                candidate,
            )
            return candidate
        # An override that does not resolve falls through to the packaged tree, which is the right fail-soft behaviour
        # but the wrong silence: the caller asked for a specific tree and got a different one.
        log.warning(
            "_server_patcher: explicit KernelForge root %s has no serving_patches directory; "
            "falling back to the packaged tree, so the requested patches are NOT the ones applied",
            arg,
        )

    try:
        from kernelforge.resources import default_project_root, packaged_data_root, resource_path

        resolved = resource_path("serving_patches", default_project_root(), missing_ok=True)
        packaged_root = packaged_data_root()
    except ImportError:
        return None
    if not resolved.is_dir():
        return None
    if resolved.parent != packaged_root:
        log.warning(
            "_server_patcher: patching SGLang from %s (KERNELFORGE_PROJECT_ROOT override), "
            "not the tree packaged with kernelforge at %s",
            resolved,
            packaged_root / "serving_patches",
        )
    return resolved


def _discover_sglang_ck_plan(arg: Path | str | None) -> _PatchPlan | None:
    """Build the SGLang fp8 block-scale CK-routing patch plan."""
    serving_patches_root = _resolve_serving_patches_root(arg)
    if serving_patches_root is None:
        log.info(
            "_server_patcher: no KernelForge serving_patches tree resolved from the packaged "
            "kernelforge — skip SGLang fp8 block-scale CK patch "
            "(SGLANG_FP8_BLOCKSCALE_CK_MAX_M will no-op on the unpatched tree)"
        )
        return None

    try:
        import sglang  # type: ignore  # noqa: I001 - runtime probe
    except Exception as e:  # noqa: BLE001 - any import failure → fail-soft
        log.warning(
            "_server_patcher: sglang not importable (%s); skip CK block-scale patch",
            e,
        )
        return None

    version = (getattr(sglang, "__version__", "") or "").strip()

    # KernelForge layout: ``serving_patches/sglang/`` holds the per-version subdirs plus the SUPPORTED_VERSIONS
    # manifest.
    patches_root = serving_patches_root / "sglang"
    if not patches_root.is_dir():
        log.warning(
            "_server_patcher: KernelForge SGLang patches root missing (%s); skip CK block-scale patch",
            patches_root,
        )
        return None

    patches_dir = _resolve_versioned_patches_dir(patches_root, version)
    if patches_dir is None:
        log.warning(
            "_server_patcher: no KernelForge CK block-scale patch found under %s/%s/ for sglang %s; skip",
            patches_root,
            _versioned_patches_subdir_name(version) or "<unknown>",
            version,
        )
        return None

    # KernelForge ships the manifest at patches_root (one level above the per-version subdir), so consult patches_root
    # for the version gate.
    if not _version_accepted(version, patches_dir=patches_root):
        log.warning(
            "_server_patcher: SGLang %s not in supported version list "
            "(consulted: $HYPERLOOM_SGLANG_PATCH_EXACT_VERSIONS, "
            "$HYPERLOOM_SGLANG_PATCH_ALLOWED_MINORS, %s/SUPPORTED_VERSIONS, "
            "then built-in minor allowlist %s); skip CK block-scale patch",
            version,
            patches_root,
            _SGLANG_DEFAULT_ALLOWED_MINORS,
        )
        return None

    patches = tuple(sorted(patches_dir.glob("*.patch")))
    if not patches:
        log.warning(
            "_server_patcher: KernelForge CK block-scale patches directory empty; skip",
        )
        return None

    sglang_module = Path(sglang.__file__).resolve()
    apply_resolution = _resolve_sglang_apply_root(sglang_module)
    if apply_resolution is None:
        return None
    apply_root, apply_strip = apply_resolution

    # Sentinel: the patch edits ``sglang/srt/layers/quantization/fp8_utils.py`` in place (both layouts).
    sentinel = sglang_module.parent / "srt" / "layers" / "quantization" / "fp8_utils.py"
    if not sentinel.is_file():
        log.warning(
            "_server_patcher: SGLang install layout unexpected (no %s); skip CK block-scale patch",
            sentinel,
        )
        return None

    return _PatchPlan(
        framework="sglang-ck",
        version=version,
        apply_root=apply_root,
        patches=patches,
        sentinel_file=sentinel,
        sentinel_text=_SGLANG_CK_BLOCKSCALE_SENTINELS,
        apply_strip=apply_strip,
    )


# Application core


def _ensure_patched(plan: _PatchPlan) -> bool:
    """Drive a plan to patched state: fast check, lock, re-check, apply."""
    if _is_patched(plan):
        return True
    with best_effort_file_lock(_LOCK_PATH, label="_server_patcher"):
        if _is_patched(plan):
            return True
        return _apply_atomic(plan)


def _markers_present(path: Path, markers: Sequence[str]) -> bool:
    """True iff ``path`` exists and contains every marker substring."""
    try:
        if not path.exists():
            return False
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return all(marker in content for marker in markers)


def _is_patched(plan: _PatchPlan) -> bool:
    """True iff the sentinel file (and every extra_sentinel) exists with all of its marker substrings present."""
    if not _markers_present(plan.sentinel_file, plan.sentinel_text):
        return False
    # The plan only keeps sentinels this patch set writes, so an absent file is an incomplete apply, not an
    # inapplicable layout.
    return all(_markers_present(path, markers) for path, markers in plan.extra_sentinels)


def _apply_atomic(plan: _PatchPlan) -> bool:
    """Apply every patch in ``plan.patches`` as a transaction."""
    git = shutil.which("git")
    if git is None:
        log.warning(
            "_server_patcher: `git` not on PATH — cannot apply %s patches "
            "(install git in the runtime image to enable TraceLens flags)",
            plan.framework,
        )
        return False
    patch_bin = shutil.which("patch")  # may be ``None`` — fuzzy fallback then disabled

    # Per-patch precheck: each must pass ``git apply --check`` (strict) OR the fuzzy ``patch --fuzz=2 --dry-run``
    # fallback; if neither accepts a patch the whole set fail-softs.
    strip_arg = f"-p{plan.apply_strip}"

    apply_modes: dict[Path, str] = {}
    for p in plan.patches:
        if _git(git, ("apply", "--check", strip_arg, str(p)), plan.apply_root):
            apply_modes[p] = "git"
            continue
        if patch_bin and _patch_dry_run(patch_bin, p, plan.apply_root, plan.apply_strip):
            log.warning(
                "_server_patcher: %s patch %s did not apply cleanly with "
                "`git apply --check %s`; falling back to `patch %s --fuzz=2` "
                "(TraceLens patch may lag deployed %s by a point release)",
                plan.framework,
                p.name,
                strip_arg,
                strip_arg,
                plan.framework,
            )
            apply_modes[p] = "patch"
            continue
        # Forward apply fails: check whether this patch is ALREADY APPLIED (a clean ``git apply -R --check`` succeeds
        # iff the tree already contains its post-image).
        if _git(git, ("apply", "-R", "--check", strip_arg, str(p)), plan.apply_root):
            log.info(
                "_server_patcher: %s patch %s already applied (reverse "
                "`git apply -R --check %s` is clean); skipping it from the "
                "transaction so the remaining patches still apply",
                plan.framework,
                p.name,
                strip_arg,
            )
            apply_modes[p] = "skip"
            continue
        # Symmetric check for patches previously applied via fuzzy `patch` (git reverse may fail on fuzzy-applied
        # hunks).
        if patch_bin and _patch_reverse_dry_run(
            patch_bin,
            p,
            plan.apply_root,
            plan.apply_strip,
        ):
            log.info(
                "_server_patcher: %s patch %s already applied (fuzzy reverse dry-run clean); skipping it",
                plan.framework,
                p.name,
            )
            apply_modes[p] = "skip"
            continue
        if p.name in plan.optional_patches:
            log.warning(
                "_server_patcher: optional %s patch %s does not apply "
                "(`git apply --check %s` + fuzzy both failed, not already "
                "applied) — skipping it so the critical patches still apply "
                "atomically (version %s; non-critical to kernel shape profiling)",
                plan.framework,
                p.name,
                strip_arg,
                plan.version,
            )
            apply_modes[p] = "skip"
            continue
        log.warning(
            "_server_patcher: `git apply --check %s` AND fuzzy `patch %s "
            "--dry-run` both failed for %s (version %s, patch %s), and it is "
            "not already applied (reverse check failed); fail-soft skip",
            strip_arg,
            strip_arg,
            plan.framework,
            plan.version,
            p.name,
        )
        return False

    applied: list[tuple[Path, str]] = []
    skipped = 0
    for p in plan.patches:
        mode = apply_modes[p]
        if mode == "skip":
            # Already applied; nothing to do and nothing to roll back.
            skipped += 1
            continue
        if mode == "git":
            ok = _git(git, ("apply", strip_arg, str(p)), plan.apply_root)
        else:
            ok = _patch_apply(
                patch_bin,
                p,
                plan.apply_root,
                plan.apply_strip,  # type: ignore[arg-type]
            )
        if ok:
            applied.append((p, mode))
            continue
        log.error(
            "_server_patcher: %s patch %s failed during apply after "
            "passing precheck (mode=%s); rolling back %d previously-applied "
            "patches",
            plan.framework,
            p.name,
            mode,
            len(applied),
        )
        _rollback_applied(applied, plan, git, patch_bin, strip_arg)
        return False

    fuzzy_count = sum(1 for _, mode in applied if mode == "patch")
    log.info(
        "_server_patcher: applied %d patch(es) for %s %s "
        "(strict=%d, fuzzy=%d, already-applied/skipped=%d) (issue #194 §4/§5)",
        len(applied),
        plan.framework,
        plan.version,
        len(applied) - fuzzy_count,
        fuzzy_count,
        skipped,
    )
    # Post-apply sentinel verification: confirm all sentinel markers are present (catches all-skipped,
    # semantically-wrong fuzzy apply, or missing annotation-pipeline markers).
    if not _is_patched(plan):
        log.error(
            "_server_patcher: post-apply sentinel check FAILED for %s %s — "
            "patches were applied/skipped (%d applied, %d skipped) but the "
            "sentinel markers are not all present; rolling back %d "
            "applied patch(es) so the framework tree matches the reported "
            "failure (no half-patched fallback)",
            plan.framework,
            plan.version,
            len(applied),
            skipped,
            len(applied),
        )
        _rollback_applied(applied, plan, git, patch_bin, strip_arg)
        return False
    return True


def _rollback_applied(
    applied: list[tuple[Path, str]],
    plan: _PatchPlan,
    git: str,
    patch_bin: str | None,
    strip_arg: str,
) -> None:
    """Reverse-apply ``applied`` patches (newest first) to restore the tree."""
    for prev, prev_mode in reversed(applied):
        if prev_mode == "git":
            rolled_back = _git(
                git,
                ("apply", "-R", strip_arg, str(prev)),
                plan.apply_root,
            )
        else:
            rolled_back = patch_bin is not None and _patch_apply(
                patch_bin,
                prev,
                plan.apply_root,
                plan.apply_strip,
                reverse=True,
            )
        if not rolled_back:
            log.error(
                "_server_patcher: rollback of %s (mode=%s) also failed — "
                "install may be in inconsistent state; manual review "
                "required",
                prev.name,
                prev_mode,
            )


# fuzz=2 tolerates whitespace / single-line drift but rejects multi-line drift hard (a higher fuzz could apply hunks
# to a semantically wrong location).
_FUZZ = 2


def _run_patch(
    patch_bin: str,
    patch_file: Path,
    cwd: Path,
    *extra_flags: str,
    strip: int = 1,
    spawn_log: Callable[..., None],
    log_stderr: bool = False,
    reverse_label: str = "",
) -> bool:
    """Run ``patch -p<strip> --fuzz=2 <extra_flags>`` with ``patch_file`` on stdin."""
    try:
        with patch_file.open("rb") as fh:
            result = subprocess.run(
                (patch_bin, f"-p{strip}", f"--fuzz={_FUZZ}", *extra_flags),
                cwd=str(cwd),
                stdin=fh,
                capture_output=True,
                timeout=_GIT_TIMEOUT_SEC,
            )
    except (OSError, subprocess.TimeoutExpired) as e:
        spawn_log(
            "_server_patcher: patch%s in %s failed to spawn (%s)",
            reverse_label,
            cwd,
            e,
        )
        return False
    if result.returncode != 0:
        if log_stderr:
            err = result.stderr.decode("utf-8", errors="replace")[:500] if result.stderr else ""
            log.debug(
                "_server_patcher: patch%s rc=%d stderr=%r",
                reverse_label,
                result.returncode,
                err,
            )
        return False
    return True


def _patch_dry_run(
    patch_bin: str,
    patch_file: Path,
    cwd: Path,
    strip: int = 1,
) -> bool:
    """Probe ``patch -p<strip> --fuzz=2 --dry-run`` for a single patch."""
    return _run_patch(
        patch_bin,
        patch_file,
        cwd,
        "--dry-run",
        "--silent",
        strip=strip,
        spawn_log=log.warning,
        reverse_label=" --dry-run",
    )


def _patch_reverse_dry_run(
    patch_bin: str,
    patch_file: Path,
    cwd: Path,
    strip: int = 1,
) -> bool:
    """Probe ``patch -R -p<strip> --fuzz=2 --dry-run`` for a single patch."""
    return _run_patch(
        patch_bin,
        patch_file,
        cwd,
        "-R",
        "--dry-run",
        "--silent",
        strip=strip,
        spawn_log=log.debug,
        reverse_label=" -R --dry-run",
    )


def _patch_apply(
    patch_bin: str,
    patch_file: Path,
    cwd: Path,
    strip: int = 1,
    *,
    reverse: bool = False,
) -> bool:
    """Real ``patch -p<strip> --fuzz=2`` apply (or reverse)."""
    flags = ("--reverse", "--silent") if reverse else ("--silent",)
    return _run_patch(
        patch_bin,
        patch_file,
        cwd,
        *flags,
        strip=strip,
        spawn_log=log.warning,
        log_stderr=True,
        reverse_label=" --reverse" if reverse else "",
    )


def _git(git: str, args: Sequence[str], cwd: Path) -> bool:
    """Run ``git <args>`` in ``cwd``."""
    try:
        result = subprocess.run(
            (git, *args),
            cwd=str(cwd),
            capture_output=True,
            timeout=_GIT_TIMEOUT_SEC,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        log.warning(
            "_server_patcher: git %s in %s failed to spawn (%s)",
            list(args),
            cwd,
            e,
        )
        return False
    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="replace")[:500] if result.stderr else ""
        log.debug(
            "_server_patcher: git %s rc=%d stderr=%r",
            list(args),
            result.returncode,
            err,
        )
        return False
    return True


__all__ = [
    "ensure_vllm_patched_for_tracelens",
    "ensure_sglang_patched_for_tracelens",
    "ensure_sglang_patched_for_ck_blockscale",
]
