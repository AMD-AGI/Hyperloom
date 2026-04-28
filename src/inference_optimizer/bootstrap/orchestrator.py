"""High-level ``ensure_claude_cli`` orchestrator.

Single entry point used by the CLI / Conductor:

    report = ensure_claude_cli(auto_install=True)

Behaviour matrix
================

================== ============== ================ ==================================
node present?      claude present? auto_install     outcome
================== ============== ================ ==================================
yes (>=18)         yes             any              ProbeResult only; no install
yes (>=18)         no              False            ``MissingDependency``
yes (>=18)         no              True             install claude only
yes (<18) or no    no              False            ``MissingDependency``
yes (<18) or no    no              True             install node + claude
yes                yes             True (force=...) re-install if requested
================== ============== ================ ==================================

The function modifies ``os.environ['PATH']`` so subsequent calls (e.g. the
ClaudeBackend launching the SDK) see the freshly-installed binaries.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .errors import InstallFailed, MissingDependency
from .install import (
    DEFAULT_CACHE_DIR,
    DEFAULT_NODE_VERSION,
    ClaudeInstall,
    NodeInstall,
    install_claude_global,
    install_node_portable,
)
from .probe import (
    NODE_MIN_VERSION,
    ProbeResult,
    parse_version,
    probe_environment,
    _run as _probe_run,
)


@dataclass
class InstallReport:
    """Outcome of ``ensure_claude_cli``."""

    probe_before: ProbeResult
    probe_after: ProbeResult
    cache_dir: Path
    node_install: NodeInstall | None = None
    claude_install: ClaudeInstall | None = None
    extra_path_dirs: tuple[Path, ...] = field(default_factory=tuple)
    notes: list[str] = field(default_factory=list)

    @property
    def installed_node(self) -> bool:
        return self.node_install is not None

    @property
    def installed_claude(self) -> bool:
        return self.claude_install is not None

    def summary(self) -> str:
        lines = ["Bootstrap report:"]
        lines.append(
            f"  node:   path={self.probe_after.node_path} "
            f"version={self.probe_after.node_version}"
        )
        lines.append(
            f"  claude: path={self.probe_after.claude_path} "
            f"version={self.probe_after.claude_version}"
        )
        lines.append(
            f"  installed_node={self.installed_node} "
            f"installed_claude={self.installed_claude} "
            f"cache_dir={self.cache_dir}"
        )
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
def _missing_message(probe: ProbeResult, *, cache_dir: Path) -> str:
    fixes: list[str] = []
    if not probe.has_node or not probe.node_is_recent_enough:
        fixes.append(
            "  node missing or <18; run with --auto-install or pre-install:\n"
            "    Linux/Mac: curl -fsSL https://nodejs.org/dist/v"
            f"{DEFAULT_NODE_VERSION}/node-v{DEFAULT_NODE_VERSION}-linux-x64.tar.xz | "
            f"tar xJ -C ~ && export PATH=~/node-v{DEFAULT_NODE_VERSION}-linux-x64/bin:$PATH\n"
            "    Windows: choco install nodejs-lts  (or download from nodejs.org)"
        )
    if not probe.has_claude:
        fixes.append(
            "  claude CLI missing; run with --auto-install or pre-install:\n"
            "    npm install -g @anthropic-ai/claude-code"
        )
    return (
        "Claude CLI bootstrap failed: required binaries missing.\n"
        + "\n".join(fixes) +
        f"\n\nWith --auto-install we will place everything under: {cache_dir}"
    )


def _prepend_path(extra_dirs: tuple[Path, ...]) -> None:
    """Mutate ``os.environ['PATH']`` so child processes inherit our overlay."""
    if not extra_dirs:
        return
    current = os.environ.get("PATH", "")
    overlay = os.pathsep.join(str(d) for d in extra_dirs)
    if overlay in current:
        return
    os.environ["PATH"] = overlay + os.pathsep + current


def ensure_claude_cli(
    *,
    auto_install: bool = False,
    cache_dir: Path | None = None,
    node_version: str = DEFAULT_NODE_VERSION,
    force_reinstall: bool = False,
) -> InstallReport:
    """Make sure ``node>=18`` and ``claude`` are available; install if asked."""
    cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    pre = probe_environment()
    extras: list[Path] = []
    node_install: NodeInstall | None = None
    claude_install: ClaudeInstall | None = None
    notes: list[str] = []

    needs_node = (
        not pre.has_node
        or not pre.node_is_recent_enough
        or force_reinstall
    )
    needs_claude = not pre.has_claude or force_reinstall

    if not (needs_node or needs_claude):
        return InstallReport(
            probe_before=pre,
            probe_after=pre,
            cache_dir=cache_dir,
            notes=["already satisfied"],
        )

    if not auto_install:
        raise MissingDependency(
            _missing_message(pre, cache_dir=cache_dir),
            missing=tuple(
                k for k, on in (("node", needs_node), ("claude", needs_claude)) if on
            ),
        )

    # ------------------------------------------------------------------
    # 1. Node
    # ------------------------------------------------------------------
    if needs_node:
        node_install = install_node_portable(
            version=node_version, cache_dir=cache_dir, force=force_reinstall,
        )
        extras.append(node_install.bin_dir)
        notes.append(f"installed node {node_version} at {node_install.install_dir}")
    else:
        # Use the system node for npm install. We still might need to know
        # where npm is to invoke it.
        if pre.npm_path is None:
            raise InstallFailed(
                "system node found but npm not on PATH; cannot install claude CLI",
                step="resolve_npm",
            )
        node_install = NodeInstall(
            install_dir=pre.node_path.parent if pre.node_path else cache_dir,
            node_bin=pre.node_path or Path("node"),
            npm_bin=pre.npm_path,
            bin_dir=pre.node_path.parent if pre.node_path else cache_dir,
        )

    # ------------------------------------------------------------------
    # 2. Claude
    # ------------------------------------------------------------------
    if needs_claude:
        prefix_dir = cache_dir / "npm-prefix"
        claude_install = install_claude_global(
            npm_bin=node_install.npm_bin,
            node_bin=node_install.node_bin,
            prefix_dir=prefix_dir,
        )
        extras.append(claude_install.bin_dir)
        notes.append(f"installed claude CLI at {claude_install.claude_bin}")

    extras_t = tuple(extras)
    _prepend_path(extras_t)

    post = probe_environment(extra_dirs=extras_t)
    if not post.has_node or not post.node_is_recent_enough or not post.has_claude:
        raise InstallFailed(
            f"post-install probe failed: {post}",
            step="post_probe",
        )

    return InstallReport(
        probe_before=pre,
        probe_after=post,
        cache_dir=cache_dir,
        node_install=node_install if needs_node else None,
        claude_install=claude_install,
        extra_path_dirs=extras_t,
        notes=notes,
    )


# Re-export internal helper (needed by tests / orchestrator integration).
__all__ = ["InstallReport", "ensure_claude_cli", "_probe_run"]
