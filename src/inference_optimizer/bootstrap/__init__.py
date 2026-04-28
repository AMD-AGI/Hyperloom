"""Bootstrap — environment readiness for the inference optimizer.

Public API:
    ensure_claude_cli(auto_install: bool = False, ...) -> InstallReport

Detects whether ``node`` and ``claude`` (the Claude Code CLI) are usable;
optionally installs them into a per-user cache directory (no sudo).

The Python ``claude-agent-sdk`` package does **not** bundle the CLI for the
Python platform (only the TypeScript SDK does), so any production deploy
that calls into the SDK needs the CLI present on PATH.

See :mod:`inference_optimizer.bootstrap.probe` for detection and
:mod:`inference_optimizer.bootstrap.install` for the install paths.
"""
from __future__ import annotations

from .errors import (
    BootstrapError,
    InstallFailed,
    MissingDependency,
    UnsupportedPlatform,
)
from .install import (
    DEFAULT_CACHE_DIR,
    DEFAULT_NODE_VERSION,
    install_claude_global,
    install_node_portable,
)
from .orchestrator import InstallReport, ensure_claude_cli
from .probe import (
    NODE_MIN_VERSION,
    ProbeResult,
    find_claude,
    find_node,
    parse_version,
    probe_environment,
)

__all__ = [
    "BootstrapError",
    "DEFAULT_CACHE_DIR",
    "DEFAULT_NODE_VERSION",
    "InstallFailed",
    "InstallReport",
    "MissingDependency",
    "NODE_MIN_VERSION",
    "ProbeResult",
    "UnsupportedPlatform",
    "ensure_claude_cli",
    "find_claude",
    "find_node",
    "install_claude_global",
    "install_node_portable",
    "parse_version",
    "probe_environment",
]
