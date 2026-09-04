# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI entry — ``optimize`` subcommand wiring Claude+Codex backends, executors, objective, and Coordinator.run().

Env vars consumed: MODEL_PATH, OPENAI_BASE_URL / ANTHROPIC_BASE_URL +
OPENAI_API_KEY / ANTHROPIC_API_KEY, ROCR_VISIBLE_DEVICES,
CLAUDE_MODEL, CODEX_MODEL, USER_DATA_PATH.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from pathlib import Path

from hyperloom.common.hermes_runtime import resolve_hermes_executable
from hyperloom.common.llm_config import (
    ANTHROPIC_SYNTHESIZABLE_KEY_ENVS,
    CLAUDE_OAUTH_TOKEN_ENV,
    anthropic_synthesizable_key,
    has_anthropic_credential,
)

log = logging.getLogger(__name__)

_OFFICIAL_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
_OFFICIAL_OPENAI_BASE_URL = "https://api.openai.com/v1"


def _explicit_hermes_profile_is_configured() -> bool:
    """Return whether explicit Hermes orchestration has a resolvable profile."""

    if os.environ.get("HYPERLOOM_AGENT_BACKEND", "").strip().lower() != "hermes":
        return False
    profile = os.environ.get("HYPERLOOM_HERMES_PROFILE", "").strip()
    if not profile:
        return False
    if not resolve_hermes_executable():
        return False
    root = Path(os.environ.get("HERMES_HOME", "").strip() or (Path.home() / ".hermes"))
    config = root / "config.yaml" if profile == "default" else root / "profiles" / profile / "config.yaml"
    return config.is_file()


# AMD Claude allowlist, ordered best-first: on a catalog miss preflight walks
# this tuple and takes the first id the gateway actually serves, so the order
# is the fallback ladder. Enforced as a hard gate only under
# INFERENCE_OPTIMIZER_ALLOW_CUSTOM_ORCH_MODEL=0; by default the gateway catalog
# probe is the gate, and custom ids outside this tuple fail outright rather
# than degrade.
_CLAUDE_PREFERRED_MODEL = "claude-opus-5"

_CLAUDE_ALLOWED_MODELS = (
    _CLAUDE_PREFERRED_MODEL,
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
)

# Codex-side counterpart, also ordered best-first. This is a fallback ladder
# only, never a gate: the Codex smoke test stays WARN-only, so an id outside
# this tuple is left untouched and merely reported. It exists because the
# default Codex model is as new as the default Claude one, and without a ladder
# a gateway that lags behind would only fail on the first Codex turn.
_CODEX_PREFERRED_MODEL = "gpt-5.6-sol"

_CODEX_FALLBACK_MODELS = (
    _CODEX_PREFERRED_MODEL,
    "gpt-5.5",
    "gpt-5.4",
)

# Catalog probe retry delays: sleep N seconds before attempt i+1; the length is
# the retry count after the initial attempt.
_CATALOG_RETRY_DELAYS_SEC = (1.0, 3.0, 5.0)

# Critic-agent skill root resolution. Env wins; else the in-tree package.
_CRITIC_AGENT_ROOT_ENV = "CRITIC_AGENT_ROOT"


def _resolve_agent_root(agent: str) -> Path | None:
    """Return an agent skill root (``$<AGENT>_AGENT_ROOT`` else the in-tree package), or ``None``.

    Args:
        agent (str): Agent package name under ``hyperloom/agents`` (e.g.
            ``"critic"``), which also names the ``<AGENT>_AGENT_ROOT`` env.

    Returns:
        Path | None: The validated agent root, or ``None`` when no candidate
            contains ``runtime/cli.py``.
    """
    override = os.environ.get(f"{agent.upper()}_AGENT_ROOT", "").strip()
    if override:
        p = Path(override).expanduser()
        return p if (p / "runtime" / "cli.py").is_file() else None
    from ..session.paths import PACKAGE_ROOT

    candidate = PACKAGE_ROOT.parent / "agents" / agent
    return candidate if (candidate / "runtime" / "cli.py").is_file() else None


def _validate_agent_runtime(root: Path, *, agent: str) -> None:
    """Fail fast (SystemExit) if ``python -m hyperloom.agents.<agent>.runtime.cli --help`` doesn't work.

    Shared by the critic and robustness preflights: both probe their runtime
    module's ``--help`` with ``cwd=root`` and abort on any launch failure or
    non-zero exit, differing only in which agent package and env/flag names
    appear in the operator-facing message.

    Args:
        root (Path): The agent skill root to validate.
        agent (str): Agent package name under ``hyperloom.agents`` (e.g.
            ``"critic"``), which also names the ``<AGENT>_AGENT_*`` env vars
            and the ``--<agent>-mock`` bypass flag.

    Raises:
        SystemExit: With code 2 when the runtime cannot start or exits
            non-zero.
    """
    module = f"hyperloom.agents.{agent}.runtime.cli"
    cmd = [sys.executable, "-m", module, "--help"]
    # Probe cost is import-bound and can spike on a loaded pod; allow an env
    # override so a slow-but-healthy runtime is not misdiagnosed as broken.
    try:
        _probe_timeout = float(os.environ.get(f"{agent.upper()}_AGENT_PROBE_TIMEOUT_SEC", "90"))
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
            f"ERROR: {agent}-agent runtime sanity check failed: {exc!r}\n"
            f"  cwd={root}\n"
            f"  cmd={' '.join(cmd)}\n"
            f"Either fix {agent.upper()}_AGENT_ROOT, check the "
            f"src/hyperloom/agents/{agent}/ install, or pass --{agent}-mock to "
            f"bypass {agent}-agent.",
            file=sys.stderr,
        )
        sys.exit(2)
    if proc.returncode != 0:
        print(
            f"ERROR: {module} --help exited rc={proc.returncode}\n  cwd={root}\n  stderr={proc.stderr.strip()[:500]}",
            file=sys.stderr,
        )
        sys.exit(2)


# Robustness-agent runtime location resolution; mirrors the critic-agent env.
_ROBUSTNESS_AGENT_ROOT_ENV = "ROBUSTNESS_AGENT_ROOT"


# Matches the ``base_url:`` line in a legacy / explicitly supplied GEAK litellm yaml.
_GEAK_BASE_URL_RE = re.compile(r"(?m)^([ \t]*base_url[ \t]*:[ \t]*).*$")


def _sync_geak_config_base_url(geak_config_path: str, base_url: str) -> bool:
    """Rewrite ``base_url:`` in the GEAK litellm config to match ``base_url``.

    Legacy GEAK invocations can pass ``--config $GEAK_CONFIG`` with an endpoint
    embedded in yaml. When an operator points ``GEAK_BASE_URL`` at a reachable
    endpoint, sync the yaml in place so that config does not keep dialing a
    stale gateway.

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
    # Function replacement so a URL with regex backreference chars can't corrupt it.
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


def _has_claude_oauth_token() -> bool:
    """True when a ``claude setup-token`` subscription credential is exported."""
    return bool(os.environ.get(CLAUDE_OAUTH_TOKEN_ENV, "").strip())


def _has_explicit_anthropic_key() -> bool:
    """True for any credential form in ``ANTHROPIC_CREDENTIAL_ENV_ORDER``."""
    return has_anthropic_credential()


def _has_explicit_openai_key() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def _is_stale_proxy_url(value: str | None) -> bool:
    """Return true for the retired local llm-proxy endpoint.

    The old installer wrote ``127.0.0.1:4002`` as a local proxy default. Modern
    operator tunnels may also be loopback URLs, so only that legacy port is
    treated as stale and force-rewritten by preflight.
    """
    if not value:
        return False
    from urllib.parse import urlparse

    parsed = urlparse(str(value).strip())
    if parsed.hostname not in {"127.0.0.1", "localhost"}:
        return False
    try:
        return parsed.port == 4002
    except ValueError:
        return False


def _resolve_llm_endpoints() -> tuple[str, str]:
    """Resolve ``(anthropic_base_url, openai_base_url)`` for split entrypoints.

    Each side resolves from its own configuration only: its explicit
    ``*_BASE_URL``, or the official SDK endpoint implied by that side's own key,
    or empty. An empty side disables the features that speak its protocol; the
    other provider's endpoint is never substituted.
    """
    openai_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    anthropic_url = os.environ.get("ANTHROPIC_BASE_URL", "").strip()

    if not anthropic_url and _has_explicit_anthropic_key():
        anthropic_url = _OFFICIAL_ANTHROPIC_BASE_URL
    if not openai_url and _has_explicit_openai_key():
        openai_url = _OFFICIAL_OPENAI_BASE_URL
    return anthropic_url, openai_url


def _reset_claude_config_to_upstream(primary_api_key: str, anthropic_base_url: str) -> None:
    """Point ``~/.claude/config.json`` ``customApiUrl`` at the upstream gateway.

    Args:
        primary_api_key (str): The Claude CLI primary API key to write; blank
            leaves any existing key untouched. Callers should pass the
            Anthropic-side key (``ANTHROPIC_API_KEY``) so a split-entrypoint
            deploy authenticates Claude with its own key rather than the shared
            gateway key. A subscription OAuth token is rejected here, and with
            no other key the file is left untouched.
        anthropic_base_url (str): The upstream gateway URL; blank is a no-op.
    """
    import json as _json

    if not anthropic_base_url:
        return
    oauth_token = os.environ.get(CLAUDE_OAUTH_TOKEN_ENV, "").strip()
    if oauth_token and primary_api_key.strip() == oauth_token:
        # primaryApiKey is an API-credits credential; persisting the subscription
        # token here would move the run off the Max/Pro plan onto API billing.
        # The variable name is spelled out rather than interpolated from
        # CLAUDE_OAUTH_TOKEN_ENV: a token-named constant reads as a secret to
        # static analysis even when only its name reaches the message.
        print(
            "Preflight: refusing to write CLAUDE_CODE_OAUTH_TOKEN into "
            "~/.claude/config.json primaryApiKey (subscription credential)"
        )
        primary_api_key = ""
    if oauth_token and not anthropic_synthesizable_key():
        # Subscription mode: the token is only valid against Anthropic itself,
        # so writing customApiUrl would point the CLI away from the endpoint
        # that accepts it. The test is "is this run on the subscription", not
        # "is primary_api_key empty" -- the caller passes ANTHROPIC_API_KEY, so
        # a host authenticating through ANTHROPIC_AUTH_TOKEN also arrives with
        # an empty key while genuinely needing the gateway URL written.
        print("Preflight: subscription token in use; ~/.claude/config.json left alone")
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


def _reject_cross_provider_pairing() -> None:
    """Fail fast unless the credentials form one of the three legal shapes.

    Legal: the Anthropic side alone, the OpenAI side alone, or both sides. A side
    is either fully absent or carries both its own base URL and its own key.
    Rejected: a base URL whose only key belongs to the other provider, and a key
    whose only endpoint would come from the other provider.

    Reads the *raw* environment, before :func:`_resolve_llm_endpoints` fills in
    any implied endpoint. A key that implies its own official endpoint may omit
    its ``*_BASE_URL``, so official-key pairings across both sides are legal.
    """
    openai_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    anthropic_url = os.environ.get("ANTHROPIC_BASE_URL", "").strip()
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    anthropic_key = has_anthropic_credential()
    # A subscription OAuth token only validates against Anthropic itself, so it
    # implies the official endpoint and completes the side without a base URL.
    if anthropic_url:
        anthropic_endpoint = anthropic_url
    elif _has_claude_oauth_token():
        anthropic_endpoint = _OFFICIAL_ANTHROPIC_BASE_URL
    else:
        anthropic_endpoint = ""
    offender = ""
    if openai_url and not openai_key and anthropic_key:
        offender = "OPENAI_BASE_URL is set without an OPENAI_API_KEY, while an Anthropic-side key is configured"
    elif anthropic_url and not anthropic_key and openai_key:
        offender = "ANTHROPIC_BASE_URL is set without an Anthropic-side key, while an OPENAI_API_KEY is configured"
    elif openai_url and anthropic_key and not anthropic_endpoint:
        offender = (
            "an Anthropic-side key is configured without ANTHROPIC_BASE_URL, "
            "while the OpenAI side points at OPENAI_BASE_URL"
        )
    elif anthropic_url and openai_key and not openai_url:
        # Only an explicit ANTHROPIC_BASE_URL signals a gateway-shaped deploy
        # whose OPENAI_API_KEY is likely a gateway key missing its own URL.
        offender = (
            "OPENAI_API_KEY is configured without OPENAI_BASE_URL, while the "
            "Anthropic side points at ANTHROPIC_BASE_URL"
        )
    if not offender:
        return
    print(
        f"\nERROR: Conflicting LLM credentials: {offender}.\n\n"
        "Hyperloom never borrows one provider's key or endpoint for the other. "
        "Configure exactly ONE of:\n"
        "  1. Anthropic side only:\n"
        "       export ANTHROPIC_BASE_URL=...  ANTHROPIC_API_KEY=...\n"
        "  2. OpenAI side only:\n"
        "       export OPENAI_BASE_URL=...     OPENAI_API_KEY=...\n"
        "  3. Both sides, each with its own base URL and key.\n"
        "Leaving a side unset simply disables the features that speak its "
        "protocol.\n\n"
        "Values are read from the shell AND from the repo .env, so a variable "
        "you did not export yourself may come from there; remove the stale "
        "entry or complete that side.",
        file=sys.stderr,
    )
    sys.exit(2)


def _warn_on_shadowed_oauth_token() -> None:
    """Warn when an API key will silently outrank the subscription token.

    The Claude CLI prefers an API key over the OAuth token, so this combination
    bills API credits even though the operator configured a subscription.
    """
    if not _has_claude_oauth_token():
        return
    shadowing = [name for name in ANTHROPIC_SYNTHESIZABLE_KEY_ENVS if os.environ.get(name, "").strip()]
    if not shadowing:
        return
    names = " and ".join(shadowing)
    print(
        f"Preflight: WARNING — CLAUDE_CODE_OAUTH_TOKEN is set alongside {names}; "
        "the Claude CLI prefers the API key, so this run bills API credits rather "
        f"than your subscription. Unset {names} to use the subscription.",
        file=sys.stderr,
    )


def _warn_on_oauth_against_a_foreign_endpoint() -> None:
    """Warn when a subscription token is pointed at a non-Anthropic endpoint.

    The token authenticates against Anthropic itself and nothing else, so this
    pairing cannot succeed. It is worth its own message because the failure is
    not the operator's first concern here: reaching a third-party gateway means
    the subscription credential is put on the wire to a host that was never
    meant to see it.
    """
    if not _has_claude_oauth_token() or anthropic_synthesizable_key():
        return
    base_url = os.environ.get("ANTHROPIC_BASE_URL", "").strip()
    if not base_url or base_url.rstrip("/") == _OFFICIAL_ANTHROPIC_BASE_URL:
        return
    print(
        "Preflight: WARNING — CLAUDE_CODE_OAUTH_TOKEN is set with "
        f"ANTHROPIC_BASE_URL={base_url}. A subscription token is only valid against "
        f"{_OFFICIAL_ANTHROPIC_BASE_URL}, so this run cannot authenticate, and the "
        "token would be sent to that endpoint. Unset ANTHROPIC_BASE_URL to use the "
        "subscription, or supply an API key that endpoint accepts.",
        file=sys.stderr,
    )


def _warn_on_oauth_widened_provider_shape() -> None:
    """Warn when a subscription token turns an OpenAI-only deploy dual-sided.

    The token is a full Anthropic side, so its mere presence in the shell moves
    orchestration off the gateway and onto the subscription — surprising for an
    operator who only configured the OpenAI side.
    """
    if not _has_claude_oauth_token():
        return
    if anthropic_synthesizable_key() or os.environ.get("DEEPSEEK_API_KEY", "").strip():
        return
    if os.environ.get("ANTHROPIC_BASE_URL", "").strip():
        return
    if not (os.environ.get("OPENAI_BASE_URL", "").strip() and os.environ.get("OPENAI_API_KEY", "").strip()):
        return
    print(
        "Preflight: WARNING — CLAUDE_CODE_OAUTH_TOKEN is set alongside a fully "
        "configured OpenAI side, so orchestration runs on the Claude subscription "
        "rather than OPENAI_BASE_URL. Unset CLAUDE_CODE_OAUTH_TOKEN for an "
        "OpenAI-only run.",
        file=sys.stderr,
    )


def _validate_credentials() -> None:
    """Fail fast when no usable LLM endpoint/key is configured.

    Requires a base URL (``OPENAI_BASE_URL`` / ``ANTHROPIC_BASE_URL``, or an
    official provider credential that implies its default endpoint) plus at
    least one key (``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY`` /
    ``ANTHROPIC_AUTH_TOKEN`` / ``CLAUDE_CODE_OAUTH_TOKEN``). Split
    Anthropic/OpenAI entrypoints and single gateways (same key under both
    provider env names) are both accepted.
    """
    _reject_cross_provider_pairing()
    _warn_on_shadowed_oauth_token()
    _warn_on_oauth_against_a_foreign_endpoint()
    _warn_on_oauth_widened_provider_shape()
    if _explicit_hermes_profile_is_configured():
        return
    anthropic_url, openai_url = _resolve_llm_endpoints()
    has_anthropic_side = has_anthropic_credential()
    has_key = bool(os.environ.get("OPENAI_API_KEY") or has_anthropic_side)
    has_usable_endpoint = bool(
        (anthropic_url and has_anthropic_side) or (openai_url and os.environ.get("OPENAI_API_KEY"))
    )
    if has_usable_endpoint and has_key:
        return

    missing: list[str] = []
    if not has_usable_endpoint:
        missing.append("a usable endpoint/key pair")
    if not has_key:
        missing.append(
            "an API key (OPENAI_API_KEY / ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / CLAUDE_CODE_OAUTH_TOKEN)"
        )
    repo_root = os.environ.get("REPO_ROOT") or os.getcwd()
    env_file = Path(repo_root) / ".env"
    env_status = "present" if env_file.exists() else "not found"
    print(
        "\nERROR: Missing required credential(s): "
        f"{', '.join(missing)}\n\n"
        "Tried loading from:\n"
        "  - shell environment\n"
        f"  - $REPO_ROOT/.env  ({env_status}: {env_file})\n\n"
        "Each provider side needs BOTH its own base URL and its own key; a side\n"
        "you leave unset just disables the features that speak its protocol.\n"
        "Configure ONE of:\n"
        "  1. OpenAI side only (Codex / GEAK; Claude-side features disabled):\n"
        "       export OPENAI_BASE_URL=https://gateway.example.com/v1  OPENAI_API_KEY=ak-your-key\n"
        "  2. Anthropic side only (Claude; Codex / GEAK disabled):\n"
        "       export ANTHROPIC_BASE_URL=https://api.anthropic.com  ANTHROPIC_API_KEY=sk-ant-xxx\n"
        "  3. Both sides, each with its own base URL and key. One gateway serving\n"
        "     both providers is this shape: point both URLs at it and set both\n"
        "     keys, even when the key value is the same.\n"
        "     Official provider keys may omit the matching *_BASE_URL.\n"
        "     A dual-protocol gateway such as DeepSeek is exactly this shape:\n"
        "       export ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic  ANTHROPIC_API_KEY=sk-xxx\n"
        "       export OPENAI_BASE_URL=https://api.deepseek.com/v1            OPENAI_API_KEY=sk-xxx\n"
        "     A gateway serving only its own models also needs the model ids;\n"
        "     known hosts default themselves, otherwise set CLAUDE_MODEL (the\n"
        "     Anthropic side) and CODEX_MODEL (the OpenAI side) yourself.\n"
        "  4. Claude Max/Pro subscription (no API credits; run `claude setup-token`):\n"
        "       export CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-xxx\n"
        "       # leave ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN unset: either one\n"
        "       # switches the Claude CLI off subscription mode.\n"
        "  5. Explicit Hermes profile (profile owns provider/auth):\n"
        "       export HYPERLOOM_AGENT_BACKEND=hermes\n"
        "       export HYPERLOOM_HERMES_PROFILE=<profile-name>",
        file=sys.stderr,
    )
    sys.exit(2)
