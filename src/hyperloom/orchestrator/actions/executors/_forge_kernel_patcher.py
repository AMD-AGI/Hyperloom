# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Install KernelForge optimized-kernel packs into the serving engine.

KernelForge publishes two kinds of serving-side asset under
``serving_patches/``, and Hyperloom is the consumer of both:

- ``<framework>/<framework>_<ver>/*.patch`` — plain source patches, handled by
  :func:`._server_patcher.ensure_sglang_patched_for_ck_blockscale`.
- ``kernels/<pack>/`` — *generated kernels*, handled here. Each pack is a
  standalone builder module (``kernel.py`` exposing
  ``build_softmax_module(M, N, dtype) -> launch``) plus a ``pack.yaml``
  manifest naming the versioned patch that routes a framework call site into
  it. The patch itself is a versioned asset like any other, so it lives in the
  framework tree and is resolved by the same version matching.

A pack is not a patch, so it cannot simply be `git apply`-ed. This module runs
the three steps that turn one into something a server can use:

1. **Install** — copy the pack into ``<workspace>/runtime/forge-kernel-packs/``
   and normalize ``pack.yaml`` to ``pack.json`` so nothing in the serving
   process needs a YAML parser. The KernelForge checkout is never written to.
2. **Preflight** — run :mod:`hyperloom.forge_kernels.preflight` in a subprocess
   on a real GPU. It builds every candidate shape, scores it against the
   framework reference and micro-benchmarks it, writing the allowlist the
   runtime dispatcher consults. A generated kernel routinely fails here (FlyDSL
   API drift against the serving image's pin, or a shape whose internal
   strategy is not launchable) and that must cost a subprocess, not a server.
3. **Patch** — apply the pack's framework patch through the same atomic,
   idempotent, sentinel-checked, rollback-on-failure machinery
   :mod:`._server_patcher` uses.

Everything is fail-soft: any step returning ``False`` leaves the framework
unpatched, and the patch itself no-ops unless
``$HYPERLOOM_FORGE_KERNEL_PACKS`` names the pack, so an applied patch is a
strict no-op against upstream behaviour until the optimizer opts in.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._file_lock import best_effort_file_lock
from ._server_patcher import _PatchPlan
from ._server_patcher import _ensure_patched
from ._server_patcher import _resolve_kernelforge_root
from ._server_patcher import _resolve_sglang_apply_root
from ._server_patcher import _resolve_versioned_patches_dir
from ._server_patcher import _version_accepted
from ._server_patcher import _version_gate_for
from ._server_patcher import _versioned_patches_subdir_name

log = logging.getLogger(__name__)

#: Where KernelForge publishes serving-side assets inside its checkout.
_SERVING_PATCHES_SUBDIR = "serving_patches"
#: Where KernelForge publishes packs, relative to the checkout root.
_PACKS_SUBDIR = (_SERVING_PATCHES_SUBDIR, "kernels")

#: csv of pack names the serving process should activate. Also the switch the
#: optimizer flips to turn a landed patch from a no-op into a live kernel.
ENV_ENABLED_PACKS = "HYPERLOOM_FORGE_KERNEL_PACKS"
#: Where the installed (not source) packs live, for the serving process.
ENV_PACK_ROOT = "HYPERLOOM_FORGE_KERNEL_PACK_ROOT"

_PREFLIGHT_TIMEOUT_SEC = 1800
_LOCK_NAME = "hyperloom_forge_kernel_packs.lock"


@dataclass(frozen=True)
class PackTarget:
    """The framework call site a pack knows how to take over.

    ``versions`` selects the call site (which framework releases have the shape
    this patch was written against); the versioned patch tree and its
    ``SUPPORTED_VERSIONS`` manifest independently decide whether a patch is
    actually available for the running version.
    """

    framework: str
    versions: tuple[str, ...]
    patch_name: str
    sentinel_file: str
    sentinel_markers: tuple[str, ...]


@dataclass(frozen=True)
class SourcePack:
    """A pack as published in the KernelForge checkout."""

    name: str
    source_dir: Path
    manifest: dict[str, Any]
    targets: tuple[PackTarget, ...]


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------


def ensure_framework_patched_for_forge_kernels(
    framework: str,
    pack_names: tuple[str, ...] | list[str],
    *,
    kernelforge_root: Path | str | None = None,
    install_root: Path | str | None = None,
    force_preflight: bool = False,
) -> tuple[str, ...]:
    """Install, gate and wire up the named packs; return the ones that landed.

    Args:
        framework: Serving framework being launched (``vllm`` / ``sglang``).
        pack_names: Packs the caller wants active, as named in
            ``$HYPERLOOM_FORGE_KERNEL_PACKS``.
        kernelforge_root: KernelForge checkout; falls back to ``$FORGE_PATH``.
        install_root: Installed-pack root; falls back to
            ``<workspace_root>/runtime/forge-kernel-packs``.
        force_preflight: Re-run the GPU gate even when a report already exists.

    Returns:
        The subset of ``pack_names`` that is installed, passed preflight, and
        whose framework patch is applied. Callers should narrow
        ``$HYPERLOOM_FORGE_KERNEL_PACKS`` to exactly this tuple so the serving
        process never asks for a pack that did not land.
    """
    wanted = tuple(dict.fromkeys(n for n in pack_names if n))
    if not wanted:
        return ()

    root = _resolve_kernelforge_root(kernelforge_root)
    if root is None:
        log.info(
            "_forge_kernel_patcher: KernelForge root unset/missing ($FORGE_PATH) — skipping optimized-kernel packs %s",
            list(wanted),
        )
        return ()

    dest_root = Path(install_root) if install_root else _default_install_root()
    landed: list[str] = []
    with best_effort_file_lock(
        str(Path(dest_root).parent / _LOCK_NAME),
        label="_forge_kernel_patcher",
    ):
        for name in wanted:
            if _land_one(framework, name, root, dest_root, force_preflight=force_preflight):
                landed.append(name)
    return tuple(landed)


def discover_packs(kernelforge_root: Path | str | None = None) -> tuple[SourcePack, ...]:
    """Every readable pack published under ``<root>/serving_patches/kernels/``."""
    root = _resolve_kernelforge_root(kernelforge_root)
    if root is None:
        return ()
    packs_root = root.joinpath(*_PACKS_SUBDIR)
    if not packs_root.is_dir():
        return ()
    found: list[SourcePack] = []
    for entry in sorted(packs_root.iterdir()):
        pack = _read_source_pack(entry)
        if pack is not None:
            found.append(pack)
    return tuple(found)


# ---------------------------------------------------------------------
# Step 1: read + install
# ---------------------------------------------------------------------


def _default_install_root() -> Path:
    from hyperloom.inference_optimizer.session.paths import runtime_dir

    return runtime_dir() / "forge-kernel-packs"


def _read_source_pack(pack_dir: Path) -> SourcePack | None:
    """Parse ``<pack_dir>/pack.yaml``; ``None`` when it is not a usable pack."""
    if not pack_dir.is_dir():
        return None
    manifest_path = pack_dir / "pack.yaml"
    if not manifest_path.is_file():
        # A bare kernel.py with no manifest is a KernelForge work-in-progress:
        # there is no declared op, entrypoint or call site to wire up, so there
        # is nothing Hyperloom could safely do with it.
        if (pack_dir / "kernel.py").is_file():
            log.info(
                "_forge_kernel_patcher: %s has kernel.py but no pack.yaml; "
                "skipping (KernelForge must declare op/builder/targets before "
                "a serving engine can route into it)",
                pack_dir,
            )
        return None
    try:
        import yaml

        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001 - unreadable manifest => not a pack
        log.warning("_forge_kernel_patcher: cannot parse %s (%s)", manifest_path, e)
        return None
    if not isinstance(manifest, dict):
        return None

    name = str(manifest.get("name") or pack_dir.name)
    targets: list[PackTarget] = []
    for raw in manifest.get("targets") or ():
        if not isinstance(raw, dict):
            continue
        sentinel = raw.get("sentinel") or {}
        targets.append(
            PackTarget(
                framework=str(raw.get("framework") or "").lower(),
                versions=tuple(str(v) for v in (raw.get("versions") or ())),
                patch_name=str(raw.get("patch_name") or ""),
                sentinel_file=str(sentinel.get("file") or ""),
                sentinel_markers=tuple(str(m) for m in (sentinel.get("markers") or ())),
            )
        )
    return SourcePack(name=name, source_dir=pack_dir, manifest=manifest, targets=tuple(targets))


def _install(pack: SourcePack, dest_root: Path) -> Path | None:
    """Copy ``kernel.py`` + a normalized ``pack.json`` into ``dest_root``.

    Returns the installed pack dir, or ``None`` on any I/O failure. Re-copies
    on every call so an updated KernelForge artifact is picked up; the
    preflight report is keyed on the kernel's content hash so a changed kernel
    forces a re-gate.
    """
    dest = dest_root / pack.name
    try:
        dest.mkdir(parents=True, exist_ok=True)
        module_name = str(pack.manifest.get("module") or "kernel.py")
        shutil.copy2(pack.source_dir / module_name, dest / "kernel.py")
        (dest / "pack.json").write_text(
            json.dumps(_normalized_manifest(pack), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as e:
        log.warning("_forge_kernel_patcher: cannot install pack %r into %s (%s)", pack.name, dest, e)
        return None
    return dest


def _normalized_manifest(pack: SourcePack) -> dict[str, Any]:
    """Flatten the YAML manifest into the JSON the serving process reads."""
    manifest = pack.manifest
    correctness = manifest.get("correctness") or {}
    performance = manifest.get("performance") or {}
    return {
        "schema_version": 1,
        "name": pack.name,
        "op": str(manifest.get("op") or ""),
        "language": str(manifest.get("language") or ""),
        "builder": str(manifest.get("builder") or ""),
        "source_dir": str(pack.source_dir),
        "kernel_sha256": _sha256(pack.source_dir / str(manifest.get("module") or "kernel.py")),
        "min_snr_db": float(correctness.get("min_snr_db", 30.0)),
        "min_graph_speedup": float(performance.get("min_graph_speedup", 1.0)),
        "probe_shapes": list(manifest.get("probe_shapes") or ()),
    }


def _sha256(path: Path) -> str:
    import hashlib

    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


# ---------------------------------------------------------------------
# Step 2: preflight
# ---------------------------------------------------------------------


def _preflight(installed: Path, manifest: dict[str, Any], *, force: bool) -> bool:
    """Run (or reuse) the GPU gate for an installed pack.

    Returns:
        True when a report exists that is ``ok`` and was produced from the
        kernel bytes currently installed.
    """
    report_path = installed / "preflight.json"
    if not force:
        cached = _read_json(report_path)
        if cached is not None and cached.get("kernel_sha256") == manifest.get("kernel_sha256"):
            if cached.get("ok"):
                log.info(
                    "_forge_kernel_patcher: reusing preflight for %r (%d verified shape(s))",
                    manifest.get("name"),
                    len(cached.get("verified") or ()),
                )
                return True
            log.warning(
                "_forge_kernel_patcher: pack %r previously failed preflight (%s); not patching",
                manifest.get("name"),
                cached.get("reason"),
            )
            return False

    cmd = [
        sys.executable,
        "-m",
        "hyperloom.forge_kernels.preflight",
        "--pack-dir",
        str(installed),
        "--out",
        str(report_path),
        "--min-snr-db",
        str(manifest.get("min_snr_db", 30.0)),
        "--min-speedup",
        str(manifest.get("min_graph_speedup", 1.0)),
    ]
    log.info("_forge_kernel_patcher: preflighting pack %r on GPU", manifest.get("name"))
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_PREFLIGHT_TIMEOUT_SEC,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as e:
        log.warning("_forge_kernel_patcher: preflight subprocess failed to run (%s)", e)
        return False

    report = _read_json(report_path)
    if report is None:
        log.warning(
            "_forge_kernel_patcher: preflight wrote no report (rc=%d); stderr tail: %s",
            proc.returncode,
            (proc.stderr or "")[-800:],
        )
        return False

    # Stamp the gated kernel's hash so a KernelForge update re-gates instead of
    # inheriting a stale verdict.
    report["kernel_sha256"] = manifest.get("kernel_sha256", "")
    try:
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except OSError:
        pass

    if not report.get("ok"):
        log.warning(
            "_forge_kernel_patcher: pack %r failed preflight (%s); leaving the framework unpatched",
            manifest.get("name"),
            report.get("reason") or "no shape passed",
        )
        return False
    log.info(
        "_forge_kernel_patcher: pack %r passed preflight (%d verified, %d rejected)",
        manifest.get("name"),
        len(report.get("verified") or ()),
        len(report.get("rejected") or ()),
    )
    return True


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


# ---------------------------------------------------------------------
# Step 3: patch the framework
# ---------------------------------------------------------------------


def _land_one(
    framework: str,
    name: str,
    kernelforge_root: Path,
    dest_root: Path,
    *,
    force_preflight: bool,
) -> bool:
    pack = _read_source_pack(kernelforge_root.joinpath(*_PACKS_SUBDIR, name))
    if pack is None:
        log.warning(
            "_forge_kernel_patcher: pack %r not published under %s; skipping",
            name,
            kernelforge_root.joinpath(*_PACKS_SUBDIR),
        )
        return False

    installed = _install(pack, dest_root)
    if installed is None:
        return False
    manifest = _read_json(installed / "pack.json") or {}
    if not _preflight(installed, manifest, force=force_preflight):
        return False

    plan = _build_patch_plan(framework, pack, kernelforge_root)
    if plan is None:
        return False
    if not _ensure_patched(plan):
        log.warning(
            "_forge_kernel_patcher: could not apply the %s patch for pack %r; the pack stays installed but inert",
            framework,
            name,
        )
        return False
    log.info("_forge_kernel_patcher: pack %r is installed, gated and patched into %s", name, framework)
    return True


def _build_patch_plan(
    framework: str,
    pack: SourcePack,
    kernelforge_root: Path,
) -> _PatchPlan | None:
    """Resolve the framework install and assemble the patch plan for ``pack``."""
    fw = (framework or "").lower()
    resolved = _resolve_framework_install(fw)
    if resolved is None:
        return None
    version, apply_root, apply_strip, package_dir = resolved

    target = _select_target(pack, fw, version)
    if target is None:
        log.info(
            "_forge_kernel_patcher: pack %r declares no target for %s %s; skipping",
            pack.name,
            fw,
            version,
        )
        return None

    patch_file = _resolve_target_patch(pack, target, fw, version, kernelforge_root)
    if patch_file is None:
        return None
    sentinel = package_dir / target.sentinel_file
    if not sentinel.is_file():
        log.warning(
            "_forge_kernel_patcher: %s install layout unexpected (no %s); skipping pack %r",
            fw,
            sentinel,
            pack.name,
        )
        return None

    return _PatchPlan(
        framework=f"{fw}-forge-pack:{pack.name}",
        version=version,
        apply_root=apply_root,
        patches=(patch_file,),
        sentinel_file=sentinel,
        sentinel_text=target.sentinel_markers,
        apply_strip=apply_strip,
    )


def _resolve_target_patch(
    pack: SourcePack,
    target: PackTarget,
    framework: str,
    version: str,
    kernelforge_root: Path,
) -> Path | None:
    """Locate the target's patch in the KernelForge versioned patch tree.

    The patch shares the tree, the per-version subdir naming and the
    ``SUPPORTED_VERSIONS`` gate with every other patch KernelForge ships for
    this framework. Returns ``None`` on any fail-soft condition.
    """
    if not target.patch_name:
        log.warning(
            "_forge_kernel_patcher: pack %r target for %s declares no patch_name; skipping",
            pack.name,
            framework,
        )
        return None

    patches_root = kernelforge_root / _SERVING_PATCHES_SUBDIR / framework
    if not patches_root.is_dir():
        log.warning(
            "_forge_kernel_patcher: KernelForge %s patch tree missing (%s); skipping pack %r",
            framework,
            patches_root,
            pack.name,
        )
        return None

    gate = _version_gate_for(framework)
    if not _version_accepted(version, patches_dir=patches_root, gate=gate):
        log.warning(
            "_forge_kernel_patcher: %s %s not in the supported version list "
            "(consulted: $%s, $%s, %s/SUPPORTED_VERSIONS, then built-in minors %s); skipping pack %r",
            framework,
            version,
            gate.exact_env,
            gate.minors_env,
            patches_root,
            gate.default_minors,
            pack.name,
        )
        return None

    patches_dir = _resolve_versioned_patches_dir(
        patches_root,
        version,
        prefix=framework,
        required_patch=target.patch_name,
    )
    if patches_dir is None:
        log.warning(
            "_forge_kernel_patcher: no %s under %s/%s/ for %s %s; skipping pack %r",
            target.patch_name,
            patches_root,
            _versioned_patches_subdir_name(version, prefix=framework) or "<unknown>",
            framework,
            version,
            pack.name,
        )
        return None
    return patches_dir / target.patch_name


def _select_target(pack: SourcePack, framework: str, version: str) -> PackTarget | None:
    """Pick the pack target for ``framework``/``version``.

    ``versions`` entries are prefixes, so ``0.25`` covers ``0.25.0`` and
    ``0.25.1`` but not ``0.250``. An entry with no ``versions`` matches any.
    """
    for target in pack.targets:
        if target.framework != framework:
            continue
        if not target.versions:
            return target
        for allowed in target.versions:
            if version == allowed or version.startswith(f"{allowed}."):
                return target
    return None


def _resolve_framework_install(framework: str) -> tuple[str, Path, int, Path] | None:
    """Probe the installed framework.

    Returns:
        ``(version, apply_root, apply_strip, package_dir)`` where ``apply_root``
        is the cwd for ``git apply`` and ``package_dir`` is the importable
        package root used to locate sentinels, or ``None`` when the framework is
        not importable / has an unrecognized layout.
    """
    if framework == "vllm":
        try:
            import vllm  # type: ignore  # noqa: I001 - runtime probe
        except Exception as e:  # noqa: BLE001
            log.warning("_forge_kernel_patcher: vllm not importable (%s); skip", e)
            return None
        version = (getattr(vllm, "__version__", "") or "").strip()
        package_dir = Path(vllm.__file__).resolve().parent
        # Patches carry ``a/vllm/...`` paths, matching the TraceLens vLLM
        # convention: apply from the dir that CONTAINS the vllm package.
        return version, package_dir.parent, 1, package_dir

    if framework == "sglang":
        try:
            import sglang  # type: ignore  # noqa: I001 - runtime probe
        except Exception as e:  # noqa: BLE001
            log.warning("_forge_kernel_patcher: sglang not importable (%s); skip", e)
            return None
        version = (getattr(sglang, "__version__", "") or "").strip()
        module = Path(sglang.__file__).resolve()
        resolution = _resolve_sglang_apply_root(module)
        if resolution is None:
            return None
        apply_root, apply_strip = resolution
        return version, apply_root, apply_strip, module.parent

    log.info("_forge_kernel_patcher: framework %r has no pack support; skip", framework)
    return None


# ---------------------------------------------------------------------
# Env wiring for the serving process
# ---------------------------------------------------------------------


def pack_envs(landed: tuple[str, ...], install_root: Path | str | None = None) -> dict[str, str]:
    """Env the serving process needs to actually activate ``landed`` packs.

    Returns an empty dict when nothing landed, so a caller can unconditionally
    merge the result without accidentally switching the feature on.
    """
    if not landed:
        return {}
    root = Path(install_root) if install_root else _default_install_root()
    return {
        ENV_ENABLED_PACKS: ",".join(landed),
        ENV_PACK_ROOT: str(root),
    }


def packs_requested_from_env(envs: dict[str, str] | None = None) -> tuple[str, ...]:
    """Pack names named in ``envs`` (else ``os.environ``)."""
    source = envs if envs is not None else os.environ
    raw = str(source.get(ENV_ENABLED_PACKS, "") or "").strip()
    if not raw:
        return ()
    return tuple(n.strip() for n in raw.split(",") if n.strip())


__all__ = [
    "ENV_ENABLED_PACKS",
    "ENV_PACK_ROOT",
    "PackTarget",
    "SourcePack",
    "discover_packs",
    "ensure_framework_patched_for_forge_kernels",
    "pack_envs",
    "packs_requested_from_env",
]
