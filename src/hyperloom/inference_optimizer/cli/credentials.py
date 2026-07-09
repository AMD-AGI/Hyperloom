# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""CLI entry — ``optimize`` subcommand wiring Claude+Codex backends, executors, objective, and Coordinator.run().

Env vars consumed: MODEL_PATH, OPENAI_BASE_URL + SAFE_API_KEY, ROCR_VISIBLE_DEVICES,
CLAUDE_MODEL, CODEX_MODEL, USER_DATA_PATH.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

# Hard model allowlist (_CLAUDE_ALLOWED_MODELS): orchestration MUST resolve to Opus 4-7 (preferred)
# or 4-6 (fallback) before Coordinator boots; other models drifted behaviour measurably (operator 2026-05-09).
_CLAUDE_PREFERRED_MODEL = "claude-opus-4-7"

_CLAUDE_FALLBACK_MODEL = "claude-opus-4-6"

_CLAUDE_ALLOWED_MODELS = (_CLAUDE_PREFERRED_MODEL, _CLAUDE_FALLBACK_MODEL)

# Catalog probe retry contract: gateway is documented-flaky. Sleep N seconds before attempt i+1;
# len(_CATALOG_RETRY_DELAYS_SEC) is the retry count after the initial attempt.
_CATALOG_RETRY_DELAYS_SEC = (1.0, 3.0, 5.0)

# Critic-agent skill root resolution. Env wins; else the in-tree
# ``src/hyperloom/agents/critic/`` package (tree-reform.MD P2.5: critic-agent
# was promoted from the sibling ``critic-agent/`` checkout into the
# ``hyperloom`` src-layout namespace; the CLI module is now package-qualified
# as ``hyperloom.agents.critic.runtime.cli`` instead of bare ``runtime.cli``).
_CRITIC_AGENT_ROOT_ENV = "CRITIC_AGENT_ROOT"

def _resolve_critic_agent_root() -> Path | None:
    """Return the critic-agent skill root (``$CRITIC_AGENT_ROOT`` else the in-tree package), or ``None``.

    Returns:
        Path | None: The validated critic-agent root, or ``None`` when no
            candidate contains ``runtime/cli.py``.
    """
    override = os.environ.get(_CRITIC_AGENT_ROOT_ENV, "").strip()
    if override:
        p = Path(override).expanduser()
        return p if (p / "runtime" / "cli.py").is_file() else None
    from ..session.paths import PACKAGE_ROOT

    candidate = PACKAGE_ROOT.parent / "agents" / "critic"
    return candidate if (candidate / "runtime" / "cli.py").is_file() else None

def _validate_critic_agent_runtime(root: Path) -> None:
    """Fail fast (SystemExit) if ``python -m hyperloom.agents.critic.runtime.cli --help`` doesn't work.

    Args:
        root (Path): The critic-agent skill root to validate.

    Raises:
        SystemExit: With code 2 when the runtime cannot start or exits
            non-zero.
    """
    cmd = [sys.executable, "-m", "hyperloom.agents.critic.runtime.cli", "--help"]
    # The probe cost is dominated by Python import time, which can spike on a
    # loaded / shared pod (heavy transitive imports over a busy filesystem).
    # Keep a safe default but allow operators to widen it via env so a slow-but-
    # healthy runtime is not misdiagnosed as broken.
    try:
        _probe_timeout = float(os.environ.get("CRITIC_AGENT_PROBE_TIMEOUT_SEC", "90"))
    except (TypeError, ValueError):
        _probe_timeout = 90.0
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=_probe_timeout,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        print(
            f"ERROR: critic-agent runtime sanity check failed: {exc!r}\n"
            f"  cwd={root}\n"
            f"  cmd={' '.join(cmd)}\n"
            f"Either fix CRITIC_AGENT_ROOT, check the "
            f"src/hyperloom/agents/critic/ install, or pass --critic-mock to "
            f"bypass critic-agent.",
            file=sys.stderr,
        )
        sys.exit(2)
    if proc.returncode != 0:
        print(
            f"ERROR: hyperloom.agents.critic.runtime.cli --help exited rc={proc.returncode}\n"
            f"  cwd={root}\n"
            f"  stderr={proc.stderr.strip()[:500]}",
            file=sys.stderr,
        )
        sys.exit(2)

# Robustness-agent runtime location resolution; mirrors critic-agent helpers
# above. tree-reform.MD P2.5: robustness-agent was promoted from a sibling
# ``robustness-agent/`` checkout into the in-tree
# ``src/hyperloom/agents/robustness/`` package (module ``robustness_agent``
# -> package-qualified ``hyperloom.agents.robustness``).
_ROBUSTNESS_AGENT_ROOT_ENV = "ROBUSTNESS_AGENT_ROOT"

def _resolve_robustness_agent_root() -> Path | None:
    """Return robustness-agent skill root (``$ROBUSTNESS_AGENT_ROOT`` else the in-tree package), or ``None``.

    Returns:
        Path | None: The validated robustness-agent root, or ``None`` when no
            candidate contains the expected ``runtime/cli.py`` module.
    """
    override = os.environ.get(_ROBUSTNESS_AGENT_ROOT_ENV, "").strip()
    if override:
        p = Path(override).expanduser()
        return p if (p / "runtime" / "cli.py").is_file() else None
    from ..session.paths import PACKAGE_ROOT

    candidate = PACKAGE_ROOT.parent / "agents" / "robustness"
    cli_module = candidate / "runtime" / "cli.py"
    return candidate if cli_module.is_file() else None

def _validate_robustness_agent_runtime(root: Path) -> None:
    """Fail fast if ``python -m hyperloom.agents.robustness.runtime.cli --help`` doesn't work.

    Runs the runtime's ``--help`` with ``cwd=root``. Any launch failure or
    non-zero exit prints an operator-facing message and aborts.

    Args:
        root (Path): The robustness-agent skill root to validate.

    Raises:
        SystemExit: With code 2 when the runtime cannot start or exits non-zero.
    """
    cmd = [sys.executable, "-m", "hyperloom.agents.robustness.runtime.cli", "--help"]
    # See _validate_critic_agent_runtime: probe cost is import-bound and can
    # spike on a loaded / shared pod. Widen the default and allow an env
    # override so a slow-but-healthy runtime is not misdiagnosed as broken.
    try:
        _probe_timeout = float(os.environ.get("ROBUSTNESS_AGENT_PROBE_TIMEOUT_SEC", "90"))
    except (TypeError, ValueError):
        _probe_timeout = 90.0
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=_probe_timeout,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        print(
            f"ERROR: robustness-agent runtime sanity check failed: {exc!r}\n"
            f"  cwd={root}\n"
            f"  cmd={' '.join(cmd)}\n"
            f"Either fix ROBUSTNESS_AGENT_ROOT, check the "
            f"src/hyperloom/agents/robustness/ install, or pass "
            f"--robustness-mock to bypass.",
            file=sys.stderr,
        )
        sys.exit(2)
    if proc.returncode != 0:
        print(
            f"ERROR: hyperloom.agents.robustness.runtime.cli --help exited rc={proc.returncode}\n"
            f"  cwd={root}\n"
            f"  stderr={proc.stderr.strip()[:500]}",
            file=sys.stderr,
        )
        sys.exit(2)

# Legacy local auth-proxy endpoint. The component was removed; any leftover
# URL pinned at this host:port is stale and must be force-rewritten to the
# upstream gateway even when an operator value is otherwise preserved (#521).
_STALE_PROXY_HOSTPORT = "127.0.0.1:4002"

def _is_stale_proxy_url(url: str | None) -> bool:
    """Return True for a leftover legacy auth-proxy URL (``127.0.0.1:4002``).

    Args:
        url (str | None): The URL to test.

    Returns:
        bool: ``True`` when the URL pins the stale legacy proxy host:port.
    """
    return _STALE_PROXY_HOSTPORT in str(url or "")

# Matches the ``base_url:`` line in the GEAK litellm yaml (two-space indent
# written by kernel-agent/scripts/install.sh, but tolerant of any indent).
_GEAK_BASE_URL_RE = re.compile(r"(?m)^([ \t]*base_url[ \t]*:[ \t]*).*$")

def _sync_geak_config_base_url(geak_config_path: str, base_url: str) -> bool:
    """Rewrite ``base_url:`` in the GEAK litellm config to match ``base_url`` (#521).

    GEAK reads its endpoint from ``--config $GEAK_CONFIG`` — a yaml written
    once at install time — not from ``$GEAK_BASE_URL`` at runtime. So when an
    operator points ``GEAK_BASE_URL`` at a reachable endpoint (e.g. a
    host-local reverse tunnel) AFTER install, the env override alone is not
    enough: the stale yaml still sends GEAK at the unreachable gateway and the
    KERNEL_AGENT phase burns budget on connection-error retries. Syncing the yaml in
    place closes that gap so the Kernel-agent actually dials the operator's
    endpoint.

    Best-effort: returns ``False`` (never raises) when the path is empty, the
    file is missing/unreadable/unwritable, it has no ``base_url:`` line, or it
    is already in sync. Returns ``True`` only when a rewrite was applied.

    Args:
        geak_config_path (str): Path to the GEAK litellm yaml config.
        base_url (str): The endpoint to write into the ``base_url:`` line.

    Returns:
        bool: ``True`` when a rewrite was applied, ``False`` otherwise.
    """
    if not geak_config_path or not base_url:
        return False
    path = Path(geak_config_path)
    try:
        if not path.is_file():
            return False
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    match = _GEAK_BASE_URL_RE.search(text)
    if match is None:
        return False
    current = match.group(0)[len(match.group(1)) :].strip()
    if current == base_url:
        return False
    # Use a function replacement so a URL containing regex backreference
    # characters (e.g. ``\g``) cannot corrupt the rewrite.
    new_text = _GEAK_BASE_URL_RE.sub(
        lambda m: m.group(1) + base_url,
        text,
        count=1,
    )
    try:
        path.write_text(new_text, encoding="utf-8")
    except OSError:
        return False
    return True

def _derive_anthropic_base_url(openai_base_url: str) -> str:
    """Derive ``ANTHROPIC_BASE_URL`` from ``OPENAI_BASE_URL`` by stripping a trailing ``/v1`` (SDK re-appends it).

    Args:
        openai_base_url (str): The ``OPENAI_BASE_URL`` value.

    Returns:
        str: The derived Anthropic base URL with a trailing ``/v1`` removed.
    """
    from urllib.parse import urlparse, urlunparse

    parsed = urlparse(openai_base_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[: -len("/v1")]
    return urlunparse(parsed._replace(path=path))

def _resolve_llm_endpoints() -> tuple[str, str]:
    """Resolve ``(anthropic_base_url, openai_base_url)`` for split entrypoints.

    Each side keeps an explicit operator value; a missing side falls back to
    the other so the legacy single-gateway setup (only ``OPENAI_BASE_URL``)
    keeps working and both Claude and Codex still reach an endpoint. Known
    stale auth-proxy leftovers (127.0.0.1:4002) are treated as unset so they
    are re-derived rather than preserved.
    """
    openai_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    anthropic_url = os.environ.get("ANTHROPIC_BASE_URL", "").strip()
    if _is_stale_proxy_url(openai_url):
        openai_url = ""
    if _is_stale_proxy_url(anthropic_url):
        anthropic_url = ""

    if anthropic_url and openai_url:
        # Both explicitly configured: respect each as-is (true dual entry).
        return anthropic_url, openai_url
    if openai_url and not anthropic_url:
        # Single OpenAI-style gateway: derive the Anthropic base from it.
        return _derive_anthropic_base_url(openai_url), openai_url
    if anthropic_url and not openai_url:
        # Anthropic-only entry: let the OpenAI/Codex side reuse the same URL.
        return anthropic_url, anthropic_url
    return "", ""

def _reset_claude_config_to_upstream(primary_api_key: str, anthropic_base_url: str) -> None:
    """Point ``~/.claude/config.json`` ``customApiUrl`` at the upstream gateway (stale 127.0.0.1:4002 would fail).

    Args:
        primary_api_key (str): The Claude CLI primary API key to write; blank
            leaves any existing key untouched. Callers should pass the
            Anthropic-side key (explicit ANTHROPIC_API_KEY wins, SAFE_API_KEY
            is the fallback) so a split-entrypoint deploy authenticates Claude
            with its own key rather than the shared gateway key.
        anthropic_base_url (str): The upstream gateway URL; blank is a no-op.
    """
    import json as _json

    if not anthropic_base_url:
        return
    claude_config_path = Path.home() / ".claude" / "config.json"
    config_data: dict = {}
    if claude_config_path.exists():
        try:
            config_data = _json.loads(claude_config_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            config_data = {}
        current_url = config_data.get("customApiUrl", "")
        if current_url == anthropic_base_url:
            print("Preflight: ~/.claude/config.json already points at upstream")
            return

    config_data.setdefault("theme", "dark")
    config_data.setdefault("hasCompletedOnboarding", True)
    if primary_api_key:
        config_data["primaryApiKey"] = primary_api_key
    elif "primaryApiKey" not in config_data:
        config_data["primaryApiKey"] = ""
    config_data["customApiUrl"] = anthropic_base_url
    claude_config_path.parent.mkdir(parents=True, exist_ok=True)
    claude_config_path.write_text(
        _json.dumps(config_data, indent=2) + "\n",
        encoding="utf-8",
    )
    claude_config_path.chmod(0o600)
    print(f"Preflight: updated ~/.claude/config.json customApiUrl -> {anthropic_base_url}")

def _validate_credentials() -> None:
    """Fail fast when no usable LLM endpoint/key is configured.

    Accepts either the legacy single-gateway pair (``SAFE_API_KEY`` +
    ``OPENAI_BASE_URL``) or the split Anthropic/OpenAI entrypoints: at least
    one base URL (``OPENAI_BASE_URL`` / ``ANTHROPIC_BASE_URL``) and at least
    one key (``SAFE_API_KEY`` / ``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY`` /
    ``ANTHROPIC_AUTH_TOKEN``).
    """
    has_url = bool(os.environ.get("OPENAI_BASE_URL") or os.environ.get("ANTHROPIC_BASE_URL"))
    has_key = bool(
        os.environ.get("SAFE_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    )
    if has_url and has_key:
        return

    missing: list[str] = []
    if not has_url:
        missing.append("a base URL (OPENAI_BASE_URL or ANTHROPIC_BASE_URL)")
    if not has_key:
        missing.append("an API key (SAFE_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY)")
    repo_root = os.environ.get("REPO_ROOT") or os.getcwd()
    env_file = Path(repo_root) / ".env"
    env_status = "present" if env_file.exists() else "not found"
    print(
        "\nERROR: Missing required credential(s): "
        f"{', '.join(missing)}\n\n"
        "Tried loading from:\n"
        "  - shell environment\n"
        f"  - $REPO_ROOT/.env  ({env_status}: {env_file})\n\n"
        "Configure ONE of:\n"
        "  1. Single gateway (AMD / LiteLLM-style):\n"
        "       export SAFE_API_KEY=sk-xxxxx\n"
        "       export OPENAI_BASE_URL=https://gateway.example.com/v1\n"
        "  2. Split entrypoints (native Anthropic + OpenAI):\n"
        "       export ANTHROPIC_BASE_URL=https://api.anthropic.com  ANTHROPIC_API_KEY=sk-ant-xxx\n"
        "       export OPENAI_BASE_URL=https://api.openai.com/v1      OPENAI_API_KEY=sk-xxx",
        file=sys.stderr,
    )
    sys.exit(2)
