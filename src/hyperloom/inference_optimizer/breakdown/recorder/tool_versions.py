"""Author-time recording of external tool versions into ``metadata.versions``.

Which build of tracelens, GEAK, forge or a CLI agent produced a session's
results is recorded when the tool is first used, so the exported provenance is
what the run resolved rather than a re-derivation at export time. Probing is
best-effort and cached per (tool, root).

Each tool owns one row in the ``versions`` item stream, keyed by its name, and
the assembler folds the stream into ``metadata.versions.tools``. The rows
cannot go into the ``metadata`` singleton directly: assembly keeps only the
newest file per producer, so these writes would be dropped whole by the
Coordinator's own metadata write, which is reissued on every state save.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .recorder import recorder_for
from .trace import trace_skip

log = logging.getLogger(__name__)

SECTION = "versions"
PRODUCER_KERNEL_AGENT = "kernel-agent"

_TOOL_META_CACHE: dict[str, dict[str, Any]] = {}

# Per-tool "authoritative version" recipe. ``root_env`` holds the install root
# (used for the commit probe and git-based version strategies). ``version``
# picks how the human version is derived:
#   * "git_describe" -> ``git describe --tags --always --dirty`` of the root
#   * "git_short"    -> ``git rev-parse --short HEAD`` of the root (== commit)
#   * ("cmd", argv)  -> first line of ``argv --version`` style CLI output
#   * ("dist", names)-> importlib.metadata version of the first matching dist
_TOOL_PROVENANCE: dict[str, dict[str, Any]] = {
    "tracelens": {"root_env": "TRACELENS_ROOT", "version": "git_describe"},
    # bypass and forge ship inside this distribution: no checkout to
    # ``git rev-parse``, so their version is Hyperloom's own. Both keys stay
    # even so -- downstream provenance JSON reads them by name.
    "bypass": {"root_env": "", "version": ("dist", ("hyperloom-inference_optimizer",))},
    "geak": {"root_env": "GEAK_ROOT", "version": "git_short"},
    "forge": {"root_env": "", "version": ("dist", ("hyperloom-inference_optimizer",))},
    "claude": {"root_env": "", "version": ("cmd", ("claude", "--version"))},
    "codex": {"root_env": "", "version": ("cmd", ("codex", "--version"))},
    "inferencex": {"root_env": "INFERENCEX_PATH", "version": "git_short"},
    "kernel_agent": {"root_env": "HYPERLOOM_KERNEL_AGENT_ROOT", "version": "git_short"},
}


def _run_first_line(argv: list[str]) -> str:
    """Trimmed first output line of ``argv``, capped at 120 chars; ``""`` on any failure."""
    import subprocess  # local: keep module import cost off the common path

    try:
        out = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except Exception:  # noqa: BLE001
        return ""
    if out.returncode != 0:
        return ""
    text = (out.stdout or "").strip() or (out.stderr or "").strip()
    return text.splitlines()[0].strip()[:120] if text else ""


def _git_short_commit(root: Path) -> str:
    """Best-effort ``git rev-parse --short HEAD`` for ``root`` (never raises)."""
    return _run_first_line(
        ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
    )


def _git_describe(root: Path) -> str:
    """Best-effort ``git describe --tags --always --dirty`` (never raises)."""
    return _run_first_line(
        ["git", "-C", str(root), "describe", "--tags", "--always", "--dirty"],
    )


def _dist_version(names: tuple[str, ...]) -> str:
    """First resolvable ``importlib.metadata`` version among ``names`` ("" if none)."""
    try:
        from importlib.metadata import version as _dist_ver
    except Exception:  # noqa: BLE001
        return ""
    for name in names:
        try:
            v = str(_dist_ver(name) or "").strip()
        except Exception:  # noqa: BLE001
            continue
        # Reject a stale 0.0.0 masquerade.
        if v and v != "0.0.0":
            return v
    return ""


def _probe_tool_version(strategy: Any, root_dir: str) -> str:
    """Resolve a tool's human version per its ``_TOOL_PROVENANCE`` strategy."""
    try:
        if strategy == "git_describe":
            return _git_describe(Path(root_dir)) if root_dir else ""
        if strategy == "git_short":
            return _git_short_commit(Path(root_dir)) if root_dir else ""
        if isinstance(strategy, tuple) and len(strategy) == 2:
            kind, arg = strategy
            if kind == "cmd":
                return _run_first_line(list(arg))
            if kind == "dist":
                return _dist_version(tuple(arg))
    except Exception:  # noqa: BLE001
        return ""
    return ""


def _tool_metadata(
    tool: str,
    *,
    root: str | None = None,
    root_env: str | None = None,
    version: str | None = None,
) -> dict[str, Any]:
    """Resolve ``{tool, root_dir, commit, version}`` for an external tool.

    Root resolution: explicit ``root`` > caller ``root_env`` > the tool's
    registered ``root_env``. ``commit`` is a cached ``git rev-parse`` of the
    root, and ``version`` is the caller's value, else a cached ``_TOOL_PROVENANCE``
    probe. Best-effort: never raises into the optimizer.
    """
    import os

    key = str(tool or "").lower()
    hint = _TOOL_PROVENANCE.get(key, {})
    root_dir = str(
        root or os.environ.get(root_env or "", "") or os.environ.get(str(hint.get("root_env") or ""), "")
    ).strip()
    cache_key = f"{key}:{root_dir}"
    cached = _TOOL_META_CACHE.get(cache_key)
    if cached is None:
        commit = ""
        if root_dir:
            try:
                if Path(root_dir).is_dir():
                    commit = _git_short_commit(Path(root_dir))
            except Exception:  # noqa: BLE001
                commit = ""
        probed = _probe_tool_version(hint.get("version"), root_dir) if hint else ""
        cached = {
            "tool": tool,
            "root_dir": root_dir,
            "commit": commit,
            "_probed_version": probed,
        }
        _TOOL_META_CACHE[cache_key] = cached
    meta = {
        "tool": cached["tool"],
        "root_dir": cached["root_dir"],
        "commit": cached["commit"],
    }
    meta["version"] = str(version or "") or str(cached.get("_probed_version") or "")
    return meta


def record_tool_version(
    session_dir: Path | str | None,
    *,
    tool: str,
    root: str | None = None,
    root_env: str | None = None,
    version: str | None = None,
    producer: str = PRODUCER_KERNEL_AGENT,
) -> None:
    """Record one external tool's resolved provenance under ``metadata.versions.tools``.

    Idempotent per tool: the row is keyed by the tool name, so re-recording the
    same tool overwrites its own row and leaves the other tools alone. A falsy
    ``session_dir`` or ``tool`` is a no-op.
    """
    name = str(tool or "").strip().lower()
    if not session_dir or not name:
        trace_skip(reason="no session_dir" if not session_dir else "no tool", section=SECTION)
        return
    try:
        meta = _tool_metadata(name, root=root, root_env=root_env, version=version)
        recorder_for(session_dir, producer=producer).record_item(SECTION, meta, key=name)
    except Exception as exc:  # noqa: BLE001
        log.debug("record_tool_version failed for %s", name, exc_info=True)
        trace_skip(reason="writer raised", section=SECTION, entity=name, error=exc)


__all__ = ["record_tool_version"]
