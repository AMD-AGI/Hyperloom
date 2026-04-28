"""Portable Node + Claude Code CLI installer.

Strategy (no sudo, fully reproducible):

1. **Node.js**: download the official portable distribution from
   ``https://nodejs.org/dist/<version>/`` for the matching OS+arch, extract
   into ``<cache_dir>/node-<version>/``.
2. **Claude CLI**: run ``npm install -g --prefix=<cache_dir>/npm-prefix
   @anthropic-ai/claude-code`` using the just-installed Node, leaving the
   ``claude`` binary at ``<cache_dir>/npm-prefix/bin/claude`` (or
   ``.../npm-prefix/claude.cmd`` on Windows).

Tests monkeypatch :func:`_download` / :func:`_extract` so we never hit the
network. The retry loop and progress banner go through this same surface.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import tarfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .errors import InstallFailed, UnsupportedPlatform


# Latest LTS as of design time. Bump together with the SDK requirement.
DEFAULT_NODE_VERSION = "20.18.0"

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "inference-optimizer"

# Package name on npm.
CLAUDE_NPM_PACKAGE = "@anthropic-ai/claude-code"

NODE_DIST_BASE = "https://nodejs.org/dist"

# Mapping (system, machine) -> (archive name template, extension)
# %s is the Node version (without leading "v").
_DIST_TABLE: dict[tuple[str, str], tuple[str, str]] = {
    ("Linux", "x86_64"):  ("node-v%s-linux-x64.tar.xz",   "tar.xz"),
    ("Linux", "aarch64"): ("node-v%s-linux-arm64.tar.xz", "tar.xz"),
    ("Darwin", "x86_64"): ("node-v%s-darwin-x64.tar.gz",  "tar.gz"),
    ("Darwin", "arm64"):  ("node-v%s-darwin-arm64.tar.gz", "tar.gz"),
    ("Windows", "AMD64"): ("node-v%s-win-x64.zip",        "zip"),
    ("Windows", "ARM64"): ("node-v%s-win-arm64.zip",      "zip"),
}


# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NodeInstall:
    """Where the portable Node landed."""

    install_dir: Path
    node_bin: Path  # path to ``node`` (or ``node.exe``)
    npm_bin: Path   # path to ``npm`` (or ``npm.cmd``)
    bin_dir: Path   # the dir to prepend to PATH


@dataclass(frozen=True)
class ClaudeInstall:
    """Where the global ``@anthropic-ai/claude-code`` was placed."""

    prefix_dir: Path
    claude_bin: Path
    bin_dir: Path  # prepend to PATH so reactor sees claude


# ---------------------------------------------------------------------------
def _platform_key() -> tuple[str, str]:
    return platform.system(), platform.machine()


def _resolve_archive(version: str) -> tuple[str, str, str]:
    """Return ``(filename, full_url, extension)`` for the running host."""
    key = _platform_key()
    entry = _DIST_TABLE.get(key)
    if entry is None:
        raise UnsupportedPlatform(
            f"no portable Node archive registered for platform {key!r}; "
            f"set INFERENCE_OPTIMIZER_NODE_BIN to a pre-installed node binary"
        )
    template, ext = entry
    fname = template % version
    url = f"{NODE_DIST_BASE}/v{version}/{fname}"
    return fname, url, ext


def _download(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest``. Tests monkeypatch this."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310
            with dest.open("wb") as fh:
                shutil.copyfileobj(resp, fh)
    except (urllib.error.URLError, OSError) as exc:
        raise InstallFailed(
            f"failed to download {url}: {exc}",
            step="download_node",
            cause=exc,
        ) from exc


def _extract(archive: Path, target_dir: Path) -> Path:
    """Extract ``archive`` into ``target_dir``; return the top-level extracted dir."""
    target_dir.mkdir(parents=True, exist_ok=True)
    suffix = "".join(archive.suffixes[-2:]) if archive.suffix == ".xz" else archive.suffix
    try:
        if archive.name.endswith((".tar.xz", ".tar.gz")):
            with tarfile.open(archive, "r:*") as tar:
                tar.extractall(target_dir)  # noqa: S202 — official Node dist
                roots = {
                    Path(name).parts[0] for name in tar.getnames() if name.strip()
                }
        elif archive.name.endswith(".zip"):
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(target_dir)
                roots = {
                    Path(n).parts[0] for n in zf.namelist() if n.strip()
                }
        else:  # pragma: no cover — guarded by _DIST_TABLE
            raise UnsupportedPlatform(f"unsupported archive type: {suffix!r}")
    except (OSError, tarfile.TarError, zipfile.BadZipFile) as exc:
        raise InstallFailed(
            f"failed to extract {archive}: {exc}",
            step="extract_node",
            cause=exc,
        ) from exc

    if len(roots) != 1:
        raise InstallFailed(
            f"expected single top-level dir in {archive}, got {sorted(roots)!r}",
            step="extract_node",
        )
    return target_dir / next(iter(roots))


def _windows() -> bool:
    return platform.system() == "Windows"


def _node_bin_paths(extracted_dir: Path) -> tuple[Path, Path, Path]:
    """Return ``(node, npm, bin_dir)`` based on platform."""
    if _windows():
        bin_dir = extracted_dir
        return (bin_dir / "node.exe", bin_dir / "npm.cmd", bin_dir)
    bin_dir = extracted_dir / "bin"
    return (bin_dir / "node", bin_dir / "npm", bin_dir)


# ---------------------------------------------------------------------------
def install_node_portable(
    *,
    version: str = DEFAULT_NODE_VERSION,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    force: bool = False,
) -> NodeInstall:
    """Download + extract a portable Node distribution. Idempotent."""
    fname, url, _ext = _resolve_archive(version)
    install_root = cache_dir / f"node-v{version}"
    extracted_marker = install_root / ".extracted"

    if extracted_marker.exists() and not force:
        # Pick the single top-level extracted dir.
        candidates = [
            p for p in install_root.iterdir()
            if p.is_dir() and p.name.startswith("node-v")
        ]
        if candidates:
            extracted = candidates[0]
            node, npm, bin_dir = _node_bin_paths(extracted)
            return NodeInstall(install_root, node, npm, bin_dir)

    install_root.mkdir(parents=True, exist_ok=True)
    archive_path = install_root / fname
    if not archive_path.exists() or force:
        _download(url, archive_path)
    extracted = _extract(archive_path, install_root)
    extracted_marker.write_text(version, encoding="utf-8")

    node, npm, bin_dir = _node_bin_paths(extracted)
    if not node.exists() or not npm.exists():
        raise InstallFailed(
            f"after extract, node/npm not found at expected paths: "
            f"{node}, {npm}",
            step="locate_node_after_extract",
        )
    return NodeInstall(
        install_dir=install_root,
        node_bin=node,
        npm_bin=npm,
        bin_dir=bin_dir,
    )


def _run_blocking(
    args: Sequence[str],
    *,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
    timeout: float = 600.0,
    step: str,
) -> str:
    """``subprocess.run`` wrapper that raises :class:`InstallFailed` on non-zero."""
    try:
        completed = subprocess.run(  # noqa: S603 — caller-controlled args
            list(args),
            capture_output=True,
            text=True,
            env=env,
            cwd=str(cwd) if cwd else None,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise InstallFailed(
            f"{step}: spawn failed: {exc}", step=step, cause=exc
        ) from exc
    if completed.returncode != 0:
        raise InstallFailed(
            f"{step}: exit={completed.returncode}\n"
            f"stdout=\n{completed.stdout}\nstderr=\n{completed.stderr}",
            step=step,
        )
    return completed.stdout or ""


def install_claude_global(
    *,
    npm_bin: Path,
    node_bin: Path,
    prefix_dir: Path,
    package: str = CLAUDE_NPM_PACKAGE,
    extra_env: dict[str, str] | None = None,
) -> ClaudeInstall:
    """Run ``npm install -g --prefix=<prefix_dir> <package>``.

    ``prefix_dir`` is the local prefix; ``claude`` ends up at
    ``<prefix_dir>/bin/claude`` (Linux/Mac) or ``<prefix_dir>/claude.cmd``
    (Windows).
    """
    prefix_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    # Make sure the freshly-installed node is on PATH for npm's child node spawns.
    env["PATH"] = os.pathsep.join([str(node_bin.parent), env.get("PATH", "")])
    env["npm_config_prefix"] = str(prefix_dir)

    _run_blocking(
        [str(npm_bin), "install", "-g", "--prefix", str(prefix_dir), package],
        env=env,
        step="npm_install_claude",
    )

    if _windows():
        bin_dir = prefix_dir
        candidates = [bin_dir / "claude.cmd", bin_dir / "claude.exe", bin_dir / "claude"]
    else:
        bin_dir = prefix_dir / "bin"
        candidates = [bin_dir / "claude"]

    for cand in candidates:
        if cand.exists():
            return ClaudeInstall(
                prefix_dir=prefix_dir,
                claude_bin=cand,
                bin_dir=bin_dir,
            )
    raise InstallFailed(
        f"npm reported success but no claude binary found in {bin_dir}",
        step="locate_claude_after_install",
    )
