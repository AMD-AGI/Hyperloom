# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The TraceLens checkout a run analyses with: where it lives, and the gateway and budget it is allowed."""

from __future__ import annotations

import os
import subprocess
import uuid
from pathlib import Path

from ._io_utils import append_log


ARCH_BENCHMARK_TIMEOUT_ENV = "TRACELENS_ARCH_BENCHMARK_TIMEOUT_SEC"

ARCH_BENCHMARK_TIMEOUT_FLOOR_S = 600


def _is_safe_litellm_gateway() -> bool:
    """True when the Claude SDK targets a strict LiteLLM-style gateway (#574).

    Detected via the SDK's ``ANTHROPIC_BASE_URL`` / ``OPENAI_BASE_URL`` host;
    other backends are left alone. A deployment fronting the gateway on a
    hostname with no protocol marker names it in
    ``HYPERLOOM_STRICT_GATEWAY_MARKERS``.
    """
    base_url = (os.environ.get("ANTHROPIC_BASE_URL", "") or os.environ.get("OPENAI_BASE_URL", "")).lower()
    # Generic protocol markers by default (no operator/brand strings shipped);
    # a specific deployment can add its own gateway host substrings via
    # HYPERLOOM_STRICT_GATEWAY_MARKERS (comma-separated).
    markers = tuple(
        m.strip().lower()
        for m in os.environ.get("HYPERLOOM_STRICT_GATEWAY_MARKERS", "litellm,llm-proxy").split(",")
        if m.strip()
    )
    return any(m in base_url for m in markers)


def _resolve_tracelens_model() -> str:
    """Resolve ``ANTHROPIC_MODEL``, else ``CLAUDE_MODEL``, to the gateway's dash-form id (#574).

    Only the strict SAFE/LiteLLM gateway rejects the image's dot form
    (``Claude-Opus-4.7``); for it, map to dash (``claude-opus-4-7``). Other
    backends keep the raw id; both unset yields ``""`` (SDK default).

    Returns:
        The model id to pass to the SDK, normalized only for SAFE/LiteLLM.
    """
    raw = os.environ.get("ANTHROPIC_MODEL", "").strip() or os.environ.get("CLAUDE_MODEL", "").strip()
    if not raw:
        return ""
    if not _is_safe_litellm_gateway():
        return raw
    return raw.lower().replace(".", "-")


def _resolve_arch_benchmark_timeout_s() -> int:
    """Return the GPU arch microbenchmark timeout in seconds (floor 600s).

    Configured via ``TRACELENS_ARCH_BENCHMARK_TIMEOUT_SEC``. Empty, non-numeric,
    or out-of-range values fall back to the 600s floor rather than crashing the
    pipeline with a ``ValueError`` before the microbenchmark runs.

    Returns:
        The arch microbenchmark timeout in seconds (at least the 600s floor).
    """
    raw = os.environ.get(ARCH_BENCHMARK_TIMEOUT_ENV, "").strip()
    if not raw:
        return ARCH_BENCHMARK_TIMEOUT_FLOOR_S
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return ARCH_BENCHMARK_TIMEOUT_FLOOR_S
    return max(ARCH_BENCHMARK_TIMEOUT_FLOOR_S, value)


def run_command(
    cmd: list[str],
    *,
    cwd: Path | None,
    log_path: Path,
    timeout_s: int,
    env: dict[str, str] | None = None,
) -> int:
    """Run a subprocess, tee its output to a log, and return the exit code.

    Args:
        cmd: Command and arguments to execute.
        cwd: Working directory, or ``None`` to inherit the current one.
        log_path: Log file the command line and output are appended to.
        timeout_s: Subprocess timeout in seconds.
        env: Optional environment for the child process.

    Returns:
        The process return code.
    """
    append_log(log_path, f"$ {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_s,
    )
    append_log(log_path, proc.stdout or "")
    append_log(log_path, f"[exit_code] {proc.returncode}")
    return proc.returncode


# Kept in sync with install.sh (TRACELENS_REPO / TRACELENS_REF). Overridable via env.
_TRACELENS_REPO_DEFAULT = "https://github.com/AMD-AGI/TraceLens.git"

_TRACELENS_REF_DEFAULT = "2a77507a29d3a2d10853bfa4316a28a70956e2bb"


def _default_tracelens_root() -> Path:
    """Installer-managed default checkout path (mirrors install.sh /
    hyperloom.inference_optimizer.session.paths.deps_cache_root)."""
    from hyperloom.inference_optimizer.session.paths import deps_cache_root

    return deps_cache_root() / "TraceLens"


def _is_default_tracelens_root(tl_root: Path) -> bool:
    """True when tl_root is the installer-managed default (not an operator
    override). Only the default path is self-healed; an explicit
    --tracelens-root / TRACELENS_ROOT is operator-maintained and fails fast."""
    try:
        return Path(tl_root).resolve() == _default_tracelens_root().resolve()
    except OSError:
        return False


def _tracelens_checkout_complete(tl_root: Path) -> bool:
    """A checkout is usable when it has TraceLens metadata or the skill tree.

    A real checkout normally has ``.git`` (which may be a file in worktrees),
    but tests and copied source bundles can be valid without VCS metadata. The
    TraceLens skill file is the runtime contract this wrapper needs before pip
    install and analysis, so accept that layout too.
    """
    if (tl_root / ".git").exists():
        return True
    return (tl_root / "TraceLens/Agent/Analysis/skills/analysis-orchestrator/SKILL.md").exists()


def _rmtree_quiet(path: Path) -> None:
    """Best-effort recursive delete; never raises (used on cleanup paths)."""
    import shutil

    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        return
    except OSError:
        return


def _ensure_tracelens_checkout(tl_root: Path, *, log_path: Path) -> None:
    """Idempotently rebuild the TraceLens checkout if it vanished mid-run.

    The pod-local checkout can disappear when a concurrent install re-clones it.
    Take the shared pod-local ``.install.lock``, double-check under the lock, then
    clone into a temp sibling and atomically rename into place so a partial clone
    is never observed. Keep in lockstep with install.sh (ensure_tracelens).
    """
    tl_root = Path(tl_root)
    if _tracelens_checkout_complete(tl_root):
        return
    repo = os.environ.get("TRACELENS_REPO") or _TRACELENS_REPO_DEFAULT
    ref = os.environ.get("TRACELENS_REF") or _TRACELENS_REF_DEFAULT
    # .install.lock lives at the open-source root, the same path both installers lock.
    lock_dir = tl_root.parent
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / ".install.lock"
    append_log(log_path, f"TraceLens root missing/incomplete; self-healing checkout at {tl_root}")
    import fcntl

    with lock_path.open("w") as lock_fh:
        try:
            fcntl.flock(lock_fh, fcntl.LOCK_EX)
        except OSError as exc:
            # Lock unavailable: proceed unlocked (the atomic rename still avoids a torn checkout).
            append_log(log_path, f"TraceLens self-heal: flock failed ({exc}); proceeding without lock")
        # Re-check completeness under the lock (a concurrent healer may have finished).
        if _tracelens_checkout_complete(tl_root):
            append_log(log_path, "TraceLens checkout completed by a concurrent healer; reusing")
            return
        # A stale/partial tree blocks the atomic rename below; move it aside first.
        if tl_root.exists():
            stale = tl_root.parent / f".{tl_root.name}.stale.{uuid.uuid4().hex[:8]}"
            os.replace(tl_root, stale)
            _rmtree_quiet(stale)
        tmp_dir = tl_root.parent / f".{tl_root.name}.heal.{uuid.uuid4().hex[:8]}"
        try:
            rc = run_command(
                ["git", "clone", "--depth", "1", repo, str(tmp_dir)],
                cwd=None,
                log_path=log_path,
                timeout_s=600,
            )
            if rc != 0:
                raise FileNotFoundError(
                    f"TraceLens root not found and self-heal clone failed (repo={repo}); tried to rebuild at {tl_root}"
                )
            # Pin to the requested SHA; a failed fetch/checkout must not ship the
            # clone's default HEAD (an unpinned tree).
            if ref and ref != "HEAD":
                rc = run_command(
                    ["git", "-C", str(tmp_dir), "fetch", "--depth", "1", "origin", ref],
                    cwd=None,
                    log_path=log_path,
                    timeout_s=600,
                )
                if rc == 0:
                    rc = run_command(
                        ["git", "-C", str(tmp_dir), "checkout", "-q", "FETCH_HEAD"],
                        cwd=None,
                        log_path=log_path,
                        timeout_s=120,
                    )
                if rc != 0:
                    raise FileNotFoundError(
                        f"TraceLens self-heal could not pin ref={ref} (repo={repo}); "
                        f"refusing to install an unpinned checkout at {tl_root}"
                    )
            os.replace(tmp_dir, tl_root)
        except BaseException:
            _rmtree_quiet(tmp_dir)
            raise
        append_log(log_path, f"TraceLens checkout self-healed at {tl_root}")
