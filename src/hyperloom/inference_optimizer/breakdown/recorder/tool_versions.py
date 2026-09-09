"""Author-time recording of external tool versions into ``metadata.versions``.

Which build of tracelens, GEAK, forge or a CLI agent produced a session's
results is a static property of the run: probed once when the tool is first
used and unchanged thereafter. Recording it at author time keeps the exported
provenance to what the run actually resolved rather than a re-derivation from
whatever the environment looks like at export time.

Each tool owns one row in the ``versions`` item stream, keyed by its name, and
the assembler folds the stream into ``metadata.versions.tools``. The rows
cannot be written into the ``metadata`` singleton directly: a singleton is one
file per producer and assembly keeps only the newest, so these writes would be
dropped whole by the Coordinator's own metadata write, which is reissued on
every state save and is therefore always the newer of the two.

Probing is best-effort and cached per (tool, root): a tool that cannot be
resolved contributes an empty version rather than blocking the caller.
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
    # The bypass trace reader ships inside this distribution, like forge below:
    # there is no checkout to ``git rev-parse``, so its version is Hyperloom's.
    # Without this entry a bypass run mints an all-empty ``versions["bypass"]``.
    "bypass": {"root_env": "", "version": ("dist", ("hyperloom-inference_optimizer",))},
    # The whole-pipeline GEAK e2e optimizer. Its checkout lives under $GEAK_ROOT
    # and its version is that repo's git SHA.
    "geak": {"root_env": "GEAK_ROOT", "version": "git_short"},
    # forge (the Kernel-Forge autonomous loop) ships inside this distribution,
    # so there is no checkout to ``git rev-parse``: its version is Hyperloom's.
    # The "forge" key stays -- downstream provenance JSON reads it by name.
    "forge": {"root_env": "", "version": ("dist", ("hyperloom-inference_optimizer",))},
    "claude": {"root_env": "", "version": ("cmd", ("claude", "--version"))},
    "codex": {"root_env": "", "version": ("cmd", ("codex", "--version"))},
    "inferencex": {"root_env": "INFERENCEX_PATH", "version": "git_short"},
    "kernel_agent": {"root_env": "HYPERLOOM_KERNEL_AGENT_ROOT", "version": "git_short"},
}


def _run_first_line(argv: list[str]) -> str:
    """Run ``argv`` and return the trimmed first output line (never raises).

    Args:
        argv (list[str]): the command argv to run.

    Returns:
        str: the trimmed first line of output (capped at 120 chars), or ``""``
            on failure / non-zero exit.
    """
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
    """Best-effort ``git rev-parse --short HEAD`` for ``root`` (never raises).

    Args:
        root (Path): the repo root to inspect.

    Returns:
        str: the short commit hash, or ``""`` when it cannot be resolved.
    """
    return _run_first_line(
        ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
    )


def _git_describe(root: Path) -> str:
    """Best-effort ``git describe --tags --always --dirty`` (never raises).

    Args:
        root (Path): the repo root to inspect.

    Returns:
        str: the ``git describe`` output, or ``""`` when it cannot be resolved.
    """
    return _run_first_line(
        ["git", "-C", str(root), "describe", "--tags", "--always", "--dirty"],
    )


def _dist_version(names: tuple[str, ...]) -> str:
    """First resolvable ``importlib.metadata`` version among ``names`` ("" if none).

    Args:
        names (tuple[str, ...]): candidate distribution names to resolve in
            order.

    Returns:
        str: the first resolvable distribution version (rejecting a stale
            ``0.0.0``), or ``""`` when none resolve.
    """
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
    """Resolve a tool's human version per its provenance ``strategy``.

    Args:
        strategy (Any): the provenance strategy (``"git_describe"`` /
            ``"git_short"`` / a ``("cmd", argv)`` or ``("dist", names)`` tuple).
        root_dir (str): the tool install root for git-based strategies.

    Returns:
        str: the resolved version string, or ``""`` when it cannot be derived.
    """
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
    root. ``version`` is the caller-supplied value, else a cached per-tool probe
    following ``_TOOL_PROVENANCE``. Best-effort: never raises into the optimizer.

    Args:
        tool (str): the external tool name (keys into ``_TOOL_PROVENANCE``).
        root (str | None): an explicit install root, highest precedence.
        root_env (str | None): a caller-supplied env var naming the root.
        version (str | None): a caller-supplied version, preferred over the
            probe.

    Returns:
        dict[str, Any]: the resolved ``{tool, root_dir, commit, version}``
            metadata.
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
    same tool overwrites its own row and leaves the other tools alone.

    Args:
        session_dir (Path | str | None): the session directory; a falsy value
            is a no-op.
        tool (str): the external tool name; keys the recorded entry.
        root (str | None): an explicit install root, highest precedence.
        root_env (str | None): a caller-supplied env var naming the root.
        version (str | None): a caller-supplied version, preferred over the
            probe.
        producer (str): the breakdown producer label.
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
