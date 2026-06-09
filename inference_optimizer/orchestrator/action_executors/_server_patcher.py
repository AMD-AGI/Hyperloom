# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Idempotent run-time patcher for vLLM and SGLang server installs
(Hyperloom issue #194 §4 / §5).

The TraceLens profiling skill needs flags that exist only in TraceLens-patched
vLLM / SGLang builds (``--profiler-config.capture_torch_profiler_dir`` /
``detailed_trace_annotation``; ``--enable-shape-discovery-for-cuda-graph-
profile``); without the patch the server fails to start. As a fallback to
rebuilding the docker image, this runtime-patches the in-container install at
the start of each profile run.

Contract: per-framework independent patchers; fail-soft (any failure returns
``False`` and callers skip the TraceLens flags); idempotent via a sentinel
substring; concurrency-safe via ``fcntl.flock``; all-or-nothing for the
multi-patch SGLang set (``--check`` all, rollback on mid-apply failure).
Patches are TraceLens's responsibility (filenames/dirs are discovered, never
hardcoded) and backward-compatible, so they're safe to leave applied (no
revert path).
"""

from __future__ import annotations

import fcntl
import logging
import os
import shutil
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

log = logging.getLogger(__name__)


# System-wide lock file (``/tmp`` is writable; cross-reboot persistence not
# needed).
_LOCK_PATH = "/tmp/hyperloom_server_patcher.lock"

# Per-``git`` invocation timeout (defensive against hung NFS).
_GIT_TIMEOUT_SEC = 30

# SGLang version gate: a minor-version allowlist (not an exact pin) so the fuzzy
# fallback can apply TraceLens patches against a freshly bumped point release.
# Widening is safe — a real context conflict fail-softs anyway. Override via
# ``HYPERLOOM_SGLANG_PATCH_ALLOWED_MINORS`` (csv) or, for exact pins,
# ``HYPERLOOM_SGLANG_PATCH_EXACT_VERSIONS`` (csv, wins over minors).
_SGLANG_DEFAULT_ALLOWED_MINORS: tuple[str, ...] = ("0.5",)

# TraceLens-shipped manifest filename(s). When present in the SGLang patches
# dir, the manifest is the source of truth for supported versions (#194 §5),
# bypassing the hardcoded default; operator env pins still win. Format: one
# version per line, ``#`` comments, blank lines ignored.
_SGLANG_SUPPORTED_VERSIONS_MANIFEST_NAMES: tuple[str, ...] = (
    "SUPPORTED_VERSIONS.txt",
    "SUPPORTED_VERSIONS",
)


def _load_sglang_supported_versions_from_manifest(
    patches_dir: Path,
) -> frozenset[str] | None:
    """Read the TraceLens-shipped ``SUPPORTED_VERSIONS`` manifest if present.

    Returns ``None`` when no manifest exists (the common case today; a
    forward-compatible hook), or a frozenset of versions (empty = reject all).
    """
    for name in _SGLANG_SUPPORTED_VERSIONS_MANIFEST_NAMES:
        manifest = patches_dir / name
        if not manifest.is_file():
            continue
        try:
            raw = manifest.read_text(encoding="utf-8")
        except OSError as e:
            log.warning(
                "_server_patcher: cannot read SGLang version manifest %s (%s);"
                " falling back to hardcoded allowlist", manifest, e,
            )
            return None
        versions: set[str] = set()
        for line in raw.splitlines():
            stripped = line.split("#", 1)[0].strip()
            if stripped:
                versions.add(stripped)
        log.info(
            "_server_patcher: loaded %d SGLang version(s) from TraceLens "
            "manifest %s (PR-D §5: decoupled from Hyperloom hardcoded "
            "allowlist)", len(versions), manifest,
        )
        return frozenset(versions)
    return None


def _sglang_version_accepted(
    version: str, *, patches_dir: Path | None = None,
) -> bool:
    """Return True iff ``version`` is in the configured allowlist.

    Precedence: ``$HYPERLOOM_SGLANG_PATCH_EXACT_VERSIONS`` (exact pins) >
    ``$HYPERLOOM_SGLANG_PATCH_ALLOWED_MINORS`` (minor prefixes, ``0.5`` matches
    ``0.5.9`` not ``0.50.0``) > TraceLens ``SUPPORTED_VERSIONS`` manifest in
    ``patches_dir`` > :data:`_SGLANG_DEFAULT_ALLOWED_MINORS`.
    """
    text = (version or "").strip()
    if not text:
        return False
    exact = os.environ.get("HYPERLOOM_SGLANG_PATCH_EXACT_VERSIONS", "").strip()
    if exact:
        allowed_exact = {v.strip() for v in exact.split(",") if v.strip()}
        return text in allowed_exact
    minors_env = os.environ.get(
        "HYPERLOOM_SGLANG_PATCH_ALLOWED_MINORS", "",
    ).strip()
    if minors_env:
        minors = tuple(v.strip() for v in minors_env.split(",") if v.strip())
        return any(
            text == minor or text.startswith(f"{minor}.")
            for minor in minors
        )
    # Vendor manifest, when present, fully replaces the hardcoded default;
    # absent -> fall through to the default below.
    if patches_dir is not None:
        manifest_versions = _load_sglang_supported_versions_from_manifest(
            patches_dir,
        )
        if manifest_versions is not None:
            return text in manifest_versions
    return any(
        text == minor or text.startswith(f"{minor}.")
        for minor in _SGLANG_DEFAULT_ALLOWED_MINORS
    )

# Path within the TraceLens checkout that hosts the patch sets.
_PATCH_TREE_REL = ("examples", "custom_workflows", "inference_analysis")


def _versioned_patches_subdir_name(version: str) -> str | None:
    """Map ``sglang.__version__`` to the per-version patch subdir name (e.g.
    ``0.5.11`` -> ``sglang_0_5_11``). Returns ``None`` when ``version`` has no
    dotted numeric head."""
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
    return "sglang_" + "_".join(numeric)


def _resolve_sglang_patches_dir(
    patches_root: Path, version: str,
) -> Path | None:
    """Locate the SGLang patches dir for the running ``sglang`` version.

    Requires the per-version subdir layout (``sglang_0_5_11/``, ...); the flat
    v0.3 layout is unsupported. Returns the subdir when it exists and has at
    least one ``*.patch``, else ``None`` (caller fail-softs).
    """
    subdir_name = _versioned_patches_subdir_name(version)
    if subdir_name is None:
        return None
    candidate = patches_root / subdir_name
    if candidate.is_dir() and any(candidate.glob("*.patch")):
        return candidate
    return None


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def ensure_vllm_patched_for_tracelens(
    tracelens_root: Path | str | None = None,
) -> bool:
    """Apply the TraceLens config patch matching the installed vLLM version.

    Returns ``True`` when patched at exit, ``False`` on any fail-soft outcome
    (callers MUST then skip the TraceLens-only profiler flags).
    """
    plan = _discover_vllm_plan(tracelens_root)
    if plan is None:
        return False
    return _ensure_patched(plan)


def ensure_sglang_patched_for_tracelens(
    tracelens_root: Path | str | None = None,
) -> bool:
    """SGLang counterpart of :func:`ensure_vllm_patched_for_tracelens`.
    Applies the 10-patch roofline / shape-discovery set as a single
    atomic transaction (``--check`` all first, rollback on mid-apply
    failure).

    Args:
        tracelens_root (Path | str | None): TraceLens checkout root;
            falls back to ``$TRACELENS_ROOT`` when ``None``.

    Returns:
        bool: ``True`` if the SGLang install is in patched state at exit.
    """
    plan = _discover_sglang_plan(tracelens_root)
    if plan is None:
        return False
    return _ensure_patched(plan)


# ---------------------------------------------------------------------
# Plan discovery
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class _PatchPlan:
    """All information needed to apply (or verify) a patch set."""
    framework: str
    version: str
    apply_root: Path                       # cwd for ``git apply``
    patches: tuple[Path, ...]              # in apply order
    sentinel_file: Path                    # file we grep to detect "already patched"
    # Substrings that must ALL be present in ``sentinel_file`` to count as
    # patched; multi-element tuples lower the false-positive risk.
    sentinel_text: tuple[str, ...]
    # Extra file-local markers required for a multi-file set to count as
    # complete (SGLang can partial-patch while leaving the historical sentinel).
    extra_sentinels: tuple[tuple[Path, tuple[str, ...]], ...] = ()
    # Per-plan ``-p<N>`` strip count: ``-p1`` for editable / vLLM, ``-p3`` for
    # wheel SGLang. Passed to both ``git apply`` and ``patch``.
    apply_strip: int = 1


def _resolve_tracelens_root(arg: Path | str | None) -> Path | None:
    """Resolve TRACELENS_ROOT from arg → env → None; fail-soft when unset or
    missing on disk."""
    if arg:
        root = Path(arg)
    else:
        env = os.environ.get("TRACELENS_ROOT", "").strip()
        if not env:
            return None
        root = Path(env)
    return root if root.is_dir() else None


def _patch_tree(tracelens_root: Path, leaf: str) -> Path:
    """Build a path under the TraceLens inference-analysis patch tree.

    Args:
        tracelens_root (Path): The resolved TraceLens checkout root.
        leaf (str): The trailing path component (e.g. a patch subdir).

    Returns:
        Path: ``<tracelens>/examples/custom_workflows/inference_analysis/<leaf>``.
    """
    return tracelens_root.joinpath(*_PATCH_TREE_REL, leaf)


def _discover_vllm_plan(arg: Path | str | None) -> _PatchPlan | None:
    """Build the vLLM patch plan for the installed vLLM version.

    Probes the importable ``vllm`` module, locates the matching
    TraceLens patch file and install layout, and assembles the
    sentinel markers used to detect an already-patched install.

    Args:
        arg (Path | str | None): TraceLens checkout root, or ``None``
            to read ``$TRACELENS_ROOT``.

    Returns:
        _PatchPlan | None: A fully-resolved plan, or ``None`` on any
        fail-soft condition (TraceLens missing, vLLM not importable,
        no matching patch, unexpected install layout).
    """
    tracelens_root = _resolve_tracelens_root(arg)
    if tracelens_root is None:
        log.info(
            "_server_patcher: TRACELENS_ROOT (public) unset/missing — skip vLLM patch"
        )
        return None

    try:
        import vllm  # type: ignore  # noqa: I001 - runtime probe
    except Exception as e:  # noqa: BLE001 - any import failure → fail-soft
        log.info("_server_patcher: vllm not importable (%s); skip patch", e)
        return None

    version = (getattr(vllm, "__version__", "") or "").strip()
    if not version:
        log.info("_server_patcher: vllm has no __version__; skip patch")
        return None

    patch_file = _patch_tree(tracelens_root, "vllm_patches") / (
        f"config_vllm_v{version}.patch"
    )
    if not patch_file.is_file():
        log.info(
            "_server_patcher: no TraceLens patch for vLLM %s "
            "(looked for %s); skip", version, patch_file,
        )
        return None

    # Apply root for the ``a/vllm/...`` prefix is site-packages (parent of the
    # ``vllm/`` package dir).
    install_root = Path(vllm.__file__).resolve().parent.parent
    sentinel = install_root / "vllm" / "config" / "profiler.py"
    if not sentinel.is_file():
        log.info(
            "_server_patcher: vLLM install layout unexpected "
            "(no %s); skip patch", sentinel,
        )
        return None

    # Both substrings live in the dataclass body the TraceLens patch adds;
    # requiring both collapses the false-positive surface to ~zero.
    return _PatchPlan(
        framework="vllm",
        version=version,
        apply_root=install_root,
        patches=(patch_file,),
        sentinel_file=sentinel,
        sentinel_text=("capture_torch_profiler_dir", "detailed_trace_annotation"),
    )


def _discover_sglang_plan(arg: Path | str | None) -> _PatchPlan | None:
    """Build the SGLang patch plan for the installed SGLang version.

    Resolves the per-version patch directory, enforces the version
    allowlist, picks the right apply root / strip count for editable
    vs wheel layouts, and assembles the multi-file sentinel markers.

    Args:
        arg (Path | str | None): TraceLens checkout root, or ``None``
            to read ``$TRACELENS_ROOT``.

    Returns:
        _PatchPlan | None: A fully-resolved plan, or ``None`` on any
        fail-soft condition (TraceLens missing, sglang not importable,
        unsupported version, no patches, unexpected install layout).
    """
    tracelens_root = _resolve_tracelens_root(arg)
    if tracelens_root is None:
        log.info(
            "_server_patcher: TRACELENS_ROOT (public) unset/missing — skip SGLang patch"
        )
        return None

    try:
        import sglang  # type: ignore  # noqa: I001 - runtime probe
    except Exception as e:  # noqa: BLE001
        log.info("_server_patcher: sglang not importable (%s); skip patch", e)
        return None

    version = (getattr(sglang, "__version__", "") or "").strip()

    # Resolve the patches dir before the version check so the gate can consult
    # the TraceLens-shipped manifest.
    patches_root = _patch_tree(tracelens_root, "sglang_roofline_patches")
    if not patches_root.is_dir():
        log.info(
            "_server_patcher: SGLang patches root missing (%s); skip",
            patches_root,
        )
        return None

    # Per-version subdir layout required (pre-v0.3.1 flat checkouts must be
    # upgraded — see README "Prepare Source Trees").
    patches_dir = _resolve_sglang_patches_dir(patches_root, version)
    if patches_dir is None:
        log.info(
            "_server_patcher: no SGLang patches found under %s/%s/ for "
            "version %s; upgrade TraceLens to Hyperloom_integration_v0.3.1+",
            patches_root,
            _versioned_patches_subdir_name(version) or "<unknown>",
            version,
        )
        return None

    if not _sglang_version_accepted(version, patches_dir=patches_dir):
        log.info(
            "_server_patcher: SGLang %s not in supported version list "
            "(consulted: $HYPERLOOM_SGLANG_PATCH_EXACT_VERSIONS, "
            "$HYPERLOOM_SGLANG_PATCH_ALLOWED_MINORS, %s/SUPPORTED_VERSIONS, "
            "then built-in minor allowlist %s); skip",
            version, patches_dir, _SGLANG_DEFAULT_ALLOWED_MINORS,
        )
        return None
    patches = tuple(sorted(patches_dir.glob("*.patch")))
    if not patches:
        log.info("_server_patcher: SGLang patches directory empty; skip")
        return None

    # Support both layouts: editable (``-p1`` from repo root) and wheel
    # (``-p3`` from inside the wheel sglang/ dir). See
    # :func:`_resolve_sglang_apply_root`.
    sglang_module = Path(sglang.__file__).resolve()
    apply_resolution = _resolve_sglang_apply_root(sglang_module)
    if apply_resolution is None:
        return None
    apply_root, apply_strip = apply_resolution

    # The per-version subdir ships a complete set authored for this release.
    filtered_patches: list[Path] = list(patches)

    # Sentinel: the kernel_shape_profiler patch creates a new file at
    # ``sglang/srt/utils/kernel_shape_profiler.py`` in both layouts.
    sentinel = sglang_module.parent / "srt" / "utils" / "kernel_shape_profiler.py"
    sglang_pkg = sglang_module.parent
    filtered_names = {p.name for p in filtered_patches}
    core_annotation_patch_names = {
        "scheduler.patch",
        "scheduler_profiler_mixin.patch",
        "io_struct.patch",
        "http_server.patch",
    }
    extra_sentinels: tuple[tuple[Path, tuple[str, ...]], ...] = ()
    if core_annotation_patch_names.issubset(filtered_names):
        extra_sentinels = (
            (
                sglang_pkg / "srt" / "managers" / "scheduler.py",
                ("_build_profile_annotation", "profile_annotation"),
            ),
            (
                sglang_pkg / "srt" / "managers" / "scheduler_profiler_mixin.py",
                ("roofline_annotations", "execute_", "torch.profiler.record_function"),
            ),
            (
                sglang_pkg / "srt" / "managers" / "io_struct.py",
                ("shape_discovery", "roofline_annotations"),
            ),
            (
                sglang_pkg / "srt" / "entrypoints" / "http_server.py",
                ("shape_discovery", "roofline_annotations"),
            ),
        )
    return _PatchPlan(
        framework="sglang",
        version=version,
        apply_root=apply_root,
        patches=tuple(filtered_patches),
        sentinel_file=sentinel,
        # Sentinel file alone is insufficient (a partial apply leaves it
        # present); the extra_sentinels below require the annotation pipeline.
        sentinel_text=("kernel_shape_profiler",),
        extra_sentinels=extra_sentinels,
        apply_strip=apply_strip,
    )


def _resolve_sglang_apply_root(sglang_module: Path) -> tuple[Path, int] | None:
    """Pick ``(apply_root, strip_count)`` for the active SGLang install.

    Editable (``<repo>/python/sglang/``): ``(repo_root, 1)``. Wheel
    (``site-packages/sglang/`` with no ``python/`` parent):
    ``(<site-packages>/sglang, 3)``. Anything else: ``None`` (fail-soft).
    """
    if sglang_module.parent.parent.name == "python":
        return sglang_module.parent.parent.parent, 1
    sglang_dir = sglang_module.parent
    if sglang_dir.name == "sglang":
        return sglang_dir, 3
    log.info(
        "_server_patcher: SGLang install at %s has unexpected layout "
        "(parent dir name=%r); skip patching",
        sglang_module, sglang_dir.name,
    )
    return None


# ---------------------------------------------------------------------
# Application core
# ---------------------------------------------------------------------


def _ensure_patched(plan: _PatchPlan) -> bool:
    """Drive a plan to patched state: fast check, lock, re-check, apply.

    Args:
        plan (_PatchPlan): The resolved patch plan to enforce.

    Returns:
        bool: ``True`` if the install is patched at exit, else ``False``.
    """
    if _is_patched(plan):
        return True
    with _file_lock(_LOCK_PATH):
        if _is_patched(plan):
            return True
        return _apply_atomic(plan)


def _is_patched(plan: _PatchPlan) -> bool:
    """True iff the sentinel file (and every extra_sentinel) exists with all of
    its marker substrings present. The all-of-N rule lowers false positives."""
    try:
        if not plan.sentinel_file.exists():
            return False
        content = plan.sentinel_file.read_text(
            encoding="utf-8", errors="replace",
        )
        if not all(marker in content for marker in plan.sentinel_text):
            return False
        for path, markers in plan.extra_sentinels:
            if not path.exists():
                return False
            extra = path.read_text(encoding="utf-8", errors="replace")
            if not all(marker in extra for marker in markers):
                return False
        return True
    except OSError:
        return False


@contextmanager
def _file_lock(path: str) -> Iterator[None]:
    """Best-effort cross-process exclusion; proceeds unsynchronized if ``/tmp``
    is read-only (the second patcher re-checks the sentinel and short-circuits)."""
    try:
        fp = open(path, "w")
    except OSError as e:
        log.warning(
            "_server_patcher: cannot open lock %s (%s); proceeding without "
            "exclusion", path, e,
        )
        yield
        return
    try:
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fp.fileno(), fcntl.LOCK_UN)
        finally:
            fp.close()


def _apply_atomic(plan: _PatchPlan) -> bool:
    """Apply every patch in ``plan.patches`` as a transaction.

    ``git apply --check`` every patch first (any failure → apply none); then
    apply one at a time, reverse-applying the already-applied ones if a later
    patch fails.
    """
    git = shutil.which("git")
    if git is None:
        log.warning(
            "_server_patcher: `git` not on PATH — cannot apply %s patches "
            "(install git in the runtime image to enable TraceLens flags)",
            plan.framework,
        )
        return False
    patch_bin = shutil.which("patch")  # may be ``None`` — fuzzy fallback then disabled

    # Per-patch precheck: each must pass ``git apply --check`` (strict) OR the
    # fuzzy ``patch --fuzz=2 --dry-run`` fallback (minor context drift); if
    # neither accepts a patch the whole set fail-softs.
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
                plan.framework, p.name, strip_arg, strip_arg, plan.framework,
            )
            apply_modes[p] = "patch"
            continue
        log.warning(
            "_server_patcher: `git apply --check %s` AND fuzzy `patch %s "
            "--dry-run` both failed for %s (version %s, patch %s); fail-soft "
            "skip", strip_arg, strip_arg, plan.framework, plan.version, p.name,
        )
        return False

    applied: list[tuple[Path, str]] = []
    for p in plan.patches:
        mode = apply_modes[p]
        if mode == "git":
            ok = _git(git, ("apply", strip_arg, str(p)), plan.apply_root)
        else:
            ok = _patch_apply(
                patch_bin, p, plan.apply_root, plan.apply_strip,  # type: ignore[arg-type]
            )
        if ok:
            applied.append((p, mode))
            continue
        log.error(
            "_server_patcher: %s patch %s failed during apply after "
            "passing precheck (mode=%s); rolling back %d previously-applied "
            "patches",
            plan.framework, p.name, mode, len(applied),
        )
        for prev, prev_mode in reversed(applied):
            if prev_mode == "git":
                rolled_back = _git(
                    git, ("apply", "-R", strip_arg, str(prev)), plan.apply_root,
                )
            else:
                rolled_back = (
                    patch_bin is not None
                    and _patch_apply(
                        patch_bin, prev, plan.apply_root, plan.apply_strip,
                        reverse=True,
                    )
                )
            if not rolled_back:
                log.error(
                    "_server_patcher: rollback of %s (mode=%s) also failed — "
                    "install may be in inconsistent state; manual review "
                    "required", prev.name, prev_mode,
                )
        return False

    fuzzy_count = sum(1 for _, mode in applied if mode == "patch")
    log.info(
        "_server_patcher: applied %d TraceLens patch(es) for %s %s "
        "(strict=%d, fuzzy=%d) (issue #194 §4/§5)",
        len(applied), plan.framework, plan.version,
        len(applied) - fuzzy_count, fuzzy_count,
    )
    return True


# fuzz=2 (GNU patch's default), not 10: fuzz=10 could silently apply change
# lines to a semantically wrong location (misleading profile data); fuzz=2
# tolerates whitespace / single-line drift but rejects multi-line drift hard.
# Kept explicit so it's grep-discoverable and survives a GNU default change.
_FUZZ = 2


def _patch_dry_run(
    patch_bin: str, patch_file: Path, cwd: Path, strip: int = 1,
) -> bool:
    """Probe ``patch -p<strip> --fuzz=2 --dry-run`` for a single patch.

    Fuzzy fallback (zero side effects) when ``git apply --check`` rejects a
    patch for minor context drift. ``strip`` matches git apply's ``-p<N>``.
    See :data:`_FUZZ`.
    """
    try:
        with patch_file.open("rb") as fh:
            result = subprocess.run(
                (patch_bin, f"-p{strip}", f"--fuzz={_FUZZ}", "--dry-run", "--silent"),
                cwd=str(cwd),
                stdin=fh,
                capture_output=True,
                timeout=_GIT_TIMEOUT_SEC,
            )
    except (OSError, subprocess.TimeoutExpired) as e:
        log.warning(
            "_server_patcher: patch --dry-run in %s failed to spawn (%s)",
            cwd, e,
        )
        return False
    return result.returncode == 0


def _patch_apply(
    patch_bin: str, patch_file: Path, cwd: Path, strip: int = 1, *,
    reverse: bool = False,
) -> bool:
    """Real ``patch -p<strip> --fuzz=2`` apply (or reverse). Mirrors
    :func:`_patch_dry_run` but actually mutates the working tree.

    Args:
        patch_bin (str): Path to the ``patch`` executable.
        patch_file (Path): The patch file to apply.
        cwd (Path): Working directory the patch is applied relative to.
        strip (int): The ``-p<N>`` strip count. Defaults to ``1``.
        reverse (bool): Apply the patch in reverse when ``True``.

    Returns:
        bool: ``True`` iff the apply exits with return code 0.
    """
    args = [patch_bin, f"-p{strip}", f"--fuzz={_FUZZ}", "--silent"]
    if reverse:
        args.append("--reverse")
    try:
        with patch_file.open("rb") as fh:
            result = subprocess.run(
                args,
                cwd=str(cwd),
                stdin=fh,
                capture_output=True,
                timeout=_GIT_TIMEOUT_SEC,
            )
    except (OSError, subprocess.TimeoutExpired) as e:
        log.warning(
            "_server_patcher: patch%s in %s failed to spawn (%s)",
            " --reverse" if reverse else "", cwd, e,
        )
        return False
    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="replace")[:500] \
            if result.stderr else ""
        log.debug(
            "_server_patcher: patch%s rc=%d stderr=%r",
            " --reverse" if reverse else "", result.returncode, err,
        )
        return False
    return True


def _git(git: str, args: Sequence[str], cwd: Path) -> bool:
    """Run ``git <args>`` in ``cwd``.

    Args:
        git (str): Path to the ``git`` executable.
        args (Sequence[str]): Arguments passed after ``git``.
        cwd (Path): Working directory for the invocation.

    Returns:
        bool: ``True`` iff the command exits with return code 0.
    """
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
            list(args), cwd, e,
        )
        return False
    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="replace")[:500] \
            if result.stderr else ""
        log.debug(
            "_server_patcher: git %s rc=%d stderr=%r",
            list(args), result.returncode, err,
        )
        return False
    return True


__all__ = [
    "ensure_vllm_patched_for_tracelens",
    "ensure_sglang_patched_for_tracelens",
]
