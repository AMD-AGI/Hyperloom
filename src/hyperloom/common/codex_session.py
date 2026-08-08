# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One-shot Codex Agent SDK turns for Hyperloom's OpenAI-side runners.

Hyperloom never issues bare LLM API calls: every interaction runs inside an
agent runtime. This module is the Codex half of that contract. It wraps
``openai_codex`` so callers inherit the SDK's shell/file tools, sandbox, turn
management and usage accounting instead of hand-rolling a tool-calling loop.

The SDK plumbing follows ``kernel_agents.agent_backends.codex.CodexBackend``,
but that class cannot be reused: its workspace guard requires the session cwd
to be a git worktree and enforces KernelForge's benchmark-file protection.
Hyperloom's Codex sessions run against plain output directories, so only the
patterns are shared.

Codex's ``read-only`` and ``workspace-write`` presets rely on bubblewrap.
Hyperloom defaults to ``workspace-write`` and performs a real bubblewrap
capability probe before starting the SDK, so a binary that exists but cannot
create the required namespace fails closed. ``bypass`` is available only when
the operator selects it with :data:`CODEX_SANDBOX_MODE_ENV` *and* confirms that
an external sandbox is already enforcing isolation with
:data:`CODEX_EXTERNAL_SANDBOX_ENV`.

Provider credentials and headers remain environment-backed. Config overrides
contain variable names only, because the SDK forwards every override through
the app-server command line. Each run also gets a private ``CODEX_HOME`` under
the configured runtime/output area; it is removed after the SDK client closes.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from hyperloom.common.llm_config import LLMConfigError, parse_custom_headers, resolve_openai_client_config

# Name Codex records the gateway under in its own TOML config. Only stability
# matters: the thread's ``model_provider`` refers back to this key.
CODEX_PROVIDER_NAME = "hyperloom"

_CLIENT_NAME = "hyperloom"
_CLIENT_TITLE = "Hyperloom"

# OpenAI-side API key names in the established Codex precedence order. Codex is
# handed the winning variable's NAME, so the secret never reaches app-server
# argv.
_API_KEY_ENV_FALLBACKS: tuple[str, ...] = ("OPENAI_API_KEY", "LLM_GATEWAY_KEY")

# TOML bare-key charset. Header names outside it would need a quoted key, which
# the Codex ``-c key=value`` override parser does not accept.
_TOML_BARE_KEY_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_EXACT_ENV_REF_RE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")
_PRIVATE_HEADER_ENV_PREFIX = "HYPERLOOM_CODEX_HTTP_HEADER_"

# Sandbox preset selector. Full access is a double opt-in because it delegates
# all containment to an external boundary.
CODEX_SANDBOX_MODE_ENV = "HYPERLOOM_CODEX_SANDBOX_MODE"
CODEX_EXTERNAL_SANDBOX_ENV = "HYPERLOOM_CODEX_EXTERNAL_SANDBOX"
DEFAULT_CODEX_SANDBOX_MODE = "workspace-write"
# Ordered so the error raised for an unknown mode lists them predictably.
CODEX_SANDBOX_MODES: tuple[str, ...] = ("bypass", "workspace-write", "read-only")

# Private per-run Codex state is created below this directory when configured.
HYPERLOOM_RUNTIME_DIR_ENV = "HYPERLOOM_RUNTIME_DIR"

# The probe mirrors the mount/user-namespace operations Codex's bubblewrap
# sandbox needs. A short timeout prevents a broken helper from blocking startup.
_BWRAP_PROBE_TIMEOUT_SEC = 10.0
_BWRAP_PROBE_CACHE: dict[tuple[Any, ...], bool] = {}
_BWRAP_PROBE_CACHE_LOCK = threading.Lock()

# Grace period for tearing a timed-out turn down before giving up on it.
_INTERRUPT_TIMEOUT_SEC = 5.0


class CodexSessionError(RuntimeError):
    """Raised when a Codex SDK turn cannot start or does not complete."""


class CodexSessionUnavailableError(CodexSessionError):
    """Raised when the Codex SDK is missing or its configuration is unusable."""


class CodexSessionTimeoutError(CodexSessionError):
    """Raised when a Codex turn outlived its timeout and was interrupted."""


@dataclass(frozen=True)
class CodexSessionResult:
    """Normalized outcome of one Codex SDK turn.

    Attributes:
        text: The agent's final response.
        items: One JSON-ready mapping per typed SDK thread item, in order.
        usage: Token accounting for the turn (empty when the SDK reported none).
        thread_id: The SDK thread handle.
        error: The in-band SDK error message, or ``""`` when the turn was clean.
            A provider-side failure can complete the turn without an answer, so
            callers must treat a non-empty value as a failed run.
    """

    text: str = ""
    items: tuple[dict[str, Any], ...] = ()
    usage: dict[str, int] = field(default_factory=dict)
    thread_id: str = ""
    error: str = ""


@dataclass(frozen=True)
class CodexProviderConfig:
    """Secret-safe provider settings resolved for one Codex child process.

    ``overrides`` is safe to place on the app-server command line: credentials
    and custom-header values are represented only by environment variable
    names. ``env_additions`` carries any private variables generated for
    literal header values and is an immutable tuple so resolution never mutates
    the caller's mapping invisibly.
    """

    overrides: tuple[str, ...]
    env_additions: tuple[tuple[str, str], ...] = field(default=(), repr=False)


def _effective_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Overlay caller values on the process environment exactly once."""
    effective = os.environ.copy()
    if env is not None:
        effective.update(env)
    return effective


def load_codex_sdk() -> Any:
    """Import ``openai_codex`` lazily and return the module.

    Returns:
        Any: The ``openai_codex`` module.

    Raises:
        CodexSessionUnavailableError: If the Codex SDK is not installed.
    """
    try:
        import openai_codex  # type: ignore[import-not-found]
    except ImportError as exc:
        raise CodexSessionUnavailableError(
            "openai_codex is not installed; install the Codex SDK "
            "(pip install 'hyperloom-inference_optimizer[llm]') before running a Codex session"
        ) from exc
    return openai_codex


def _toml_string(value: str) -> str:
    """Encode a TOML basic string for one Codex ``-c key=value`` override."""
    return json.dumps(value)


def api_key_env_name(
    *,
    api_key_env: str = "OPENAI_API_KEY",
    env: dict[str, str] | None = None,
) -> str:
    """Return the NAME of the env var holding the OpenAI-side API key.

    Mirrors the precedence in
    :func:`hyperloom.common.llm_config.resolve_openai_client_config` so Codex
    reads the same credential as Hyperloom's other OpenAI-side callers, while
    only the variable name crosses into Codex's configuration.

    Args:
        api_key_env (str): Preferred env var name to check first.
        env (dict[str, str] | None): Values to overlay on ``os.environ``.

    Returns:
        str: The name of the first env var in precedence order that is set.

    Raises:
        CodexSessionUnavailableError: If none of the candidates is set.
    """
    return _api_key_env_name(api_key_env=api_key_env, source=_effective_env(env))


def _api_key_env_name(*, api_key_env: str, source: Mapping[str, str]) -> str:
    """Resolve the API key variable name from an already-effective mapping."""
    candidates = list(dict.fromkeys([api_key_env, *_API_KEY_ENV_FALLBACKS]))
    for name in candidates:
        if (source.get(name) or "").strip():
            return name
    raise CodexSessionUnavailableError(f"none of {' / '.join(candidates)} is set in env; Codex cannot authenticate")


def _unexpanded_custom_headers(raw: str | None) -> dict[str, str]:
    """Parse custom headers without resolving ``${VAR}`` expressions."""
    if not raw:
        return {}
    text = raw.strip()
    if not text:
        return {}
    if text.startswith("{"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return {str(key).strip(): str(value).strip() for key, value in parsed.items() if str(key).strip()}

    headers: dict[str, str] = {}
    for line in raw.splitlines():
        name, separator, value = line.partition(":")
        if separator and name.strip():
            headers[name.strip()] = value.strip()
    return headers


def _selected_custom_header_source(
    source: Mapping[str, str],
    *,
    base_url_env: str,
) -> str | None:
    """Return the raw header setting selected by OpenAI config resolution."""
    openai_raw = source.get("OPENAI_CUSTOM_HEADERS")
    if parse_custom_headers(openai_raw, env=source):
        return openai_raw

    explicit_base_url = (source.get(base_url_env) or "").strip() or (source.get("OPENAI_BASE_URL") or "").strip()
    derived_base_url = (source.get("ANTHROPIC_BASE_URL") or "").strip()
    if not explicit_base_url and derived_base_url:
        return source.get("ANTHROPIC_CUSTOM_HEADERS")
    return openai_raw


def _private_header_env_name(
    index: int,
    *,
    source: Mapping[str, str],
    additions: Mapping[str, str],
) -> str:
    """Return a generated child-only variable name that cannot overwrite input."""
    base = f"{_PRIVATE_HEADER_ENV_PREFIX}{index}"
    candidate = base
    suffix = 1
    while candidate in source or candidate in additions:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def resolve_codex_provider_config(
    *,
    api_key_env: str = "OPENAI_API_KEY",
    base_url_env: str = "OPENAI_BASE_URL",
    env: dict[str, str] | None = None,
) -> CodexProviderConfig:
    """Resolve env-backed ``model_providers`` settings for the Codex gateway.

    The installed Codex CLI's ``env_key`` and ``env_http_headers`` fields map
    provider credentials and headers to environment variable names. Resolved
    values therefore never enter ``config_overrides``, which the Python SDK
    forwards as app-server argv. An exact ``${VAR}`` header uses that existing
    name; a literal or composite value is copied into a generated child-only
    variable returned in :attr:`CodexProviderConfig.env_additions`.

    Args:
        api_key_env (str): Preferred API-key env var name.
        base_url_env (str): Preferred base-URL env var name.
        env (dict[str, str] | None): Values to overlay on ``os.environ``.

    Returns:
        CodexProviderConfig: Command-line-safe overrides plus child env
        additions.

    Raises:
        CodexSessionUnavailableError: If the credential or the base URL is
            missing, or a gateway header name cannot be expressed as a Codex
            config key.
    """
    return _resolve_codex_provider_config(
        api_key_env=api_key_env,
        base_url_env=base_url_env,
        source=_effective_env(env),
    )


def _resolve_codex_provider_config(
    *,
    api_key_env: str,
    base_url_env: str,
    source: Mapping[str, str],
) -> CodexProviderConfig:
    """Resolve provider config from one already-effective environment."""
    try:
        config = resolve_openai_client_config(
            api_key_env=api_key_env,
            base_url_env=base_url_env,
            env=dict(source),
        )
    except LLMConfigError as exc:
        raise CodexSessionUnavailableError(f"Codex gateway credential is missing: {exc}") from exc
    if not config.base_url:
        raise CodexSessionUnavailableError(
            f"{base_url_env} is not set; Codex needs an explicit OpenAI-compatible gateway base URL"
        )

    key_env = _api_key_env_name(api_key_env=api_key_env, source=source)
    provider = CODEX_PROVIDER_NAME
    overrides = [
        f"model_provider={_toml_string(provider)}",
        f"model_providers.{provider}.name={_toml_string(provider)}",
        f"model_providers.{provider}.base_url={_toml_string(config.base_url)}",
        f"model_providers.{provider}.wire_api={_toml_string('responses')}",
        f"model_providers.{provider}.env_key={_toml_string(key_env)}",
    ]
    raw_headers = _unexpanded_custom_headers(_selected_custom_header_source(source, base_url_env=base_url_env))
    env_additions: dict[str, str] = {}
    for index, (header, value) in enumerate(config.default_headers.items()):
        if not _TOML_BARE_KEY_RE.match(header):
            raise CodexSessionUnavailableError(f"gateway header name {header!r} is not a valid Codex config key")
        raw_value = raw_headers.get(header, "")
        env_reference = _EXACT_ENV_REF_RE.fullmatch(raw_value)
        referenced_name = env_reference.group(1) if env_reference is not None else ""
        if referenced_name and source.get(referenced_name) == value:
            header_env_name = referenced_name
        else:
            header_env_name = _private_header_env_name(index, source=source, additions=env_additions)
            env_additions[header_env_name] = value
        overrides.append(f"model_providers.{provider}.env_http_headers.{header}={_toml_string(header_env_name)}")
    return CodexProviderConfig(overrides=tuple(overrides), env_additions=tuple(env_additions.items()))


def codex_provider_overrides(
    *,
    api_key_env: str = "OPENAI_API_KEY",
    base_url_env: str = "OPENAI_BASE_URL",
    env: dict[str, str] | None = None,
) -> tuple[str, ...]:
    """Return provider overrides for compatibility with existing callers.

    New process-launching integrations should use
    :func:`resolve_codex_provider_config` and apply its ``env_additions``.
    """
    return resolve_codex_provider_config(
        api_key_env=api_key_env,
        base_url_env=base_url_env,
        env=env,
    ).overrides


def _writable_root_overrides(writable_roots: Sequence[Path]) -> tuple[str, ...]:
    """Widen the ``workspace_write`` sandbox to the given roots.

    The session cwd is already writable under that preset, so only the extra
    roots are declared. Codex reads the table only when that preset is active,
    so the override is emitted for every sandbox mode and simply goes unread
    under the others.
    """
    if not writable_roots:
        return ()
    roots = [str(Path(root).resolve()) for root in writable_roots]
    return (f"sandbox_workspace_write.writable_roots={json.dumps(roots)}",)


def _validated_sandbox_mode(mode: str) -> str:
    """Return ``mode`` when it names a known preset family, else fail loudly."""
    if mode not in CODEX_SANDBOX_MODES:
        raise CodexSessionUnavailableError(
            f"unknown Codex sandbox mode {mode!r}; set {CODEX_SANDBOX_MODE_ENV} to one of "
            f"{' / '.join(CODEX_SANDBOX_MODES)}"
        )
    return mode


def resolve_codex_sandbox_mode(*, sandbox_mode: str = "", env: dict[str, str] | None = None) -> str:
    """Resolve which Codex sandbox preset family a session may use.

    ``bypass`` is deliberately different from the contained modes: the
    deployment must set both ``HYPERLOOM_CODEX_SANDBOX_MODE=bypass`` and
    ``HYPERLOOM_CODEX_EXTERNAL_SANDBOX=1``. A caller argument may narrow the
    deployment policy but cannot replace either operator-owned bypass opt-in.

    Args:
        sandbox_mode (str): Mode stated by the caller; outranks the
            environment. Blank defers to :data:`CODEX_SANDBOX_MODE_ENV`.
        env (dict[str, str] | None): Values to overlay on ``os.environ``.

    Returns:
        str: One of :data:`CODEX_SANDBOX_MODES`, defaulting to
            :data:`DEFAULT_CODEX_SANDBOX_MODE`.

    Raises:
        CodexSessionUnavailableError: If the resolved value names no known mode
            or ``bypass`` lacks either required opt-in.
    """
    return _resolve_codex_sandbox_mode(
        sandbox_mode=sandbox_mode,
        source=_effective_env(env),
    )


def _resolve_codex_sandbox_mode(*, sandbox_mode: str, source: Mapping[str, str]) -> str:
    """Resolve sandbox policy from one already-effective environment."""
    stated = sandbox_mode.strip().lower()
    configured = (source.get(CODEX_SANDBOX_MODE_ENV) or "").strip().lower()
    resolved = _validated_sandbox_mode(stated or configured or DEFAULT_CODEX_SANDBOX_MODE)
    if resolved != "bypass":
        return resolved
    if configured != "bypass":
        raise CodexSessionUnavailableError(
            f"Codex sandbox bypass requires {CODEX_SANDBOX_MODE_ENV}=bypass in the effective environment"
        )
    if (source.get(CODEX_EXTERNAL_SANDBOX_ENV) or "").strip() != "1":
        raise CodexSessionUnavailableError(
            f"Codex sandbox bypass requires {CODEX_EXTERNAL_SANDBOX_ENV}=1 to confirm external isolation"
        )
    return resolved


def _bwrap_probe_cache_key(bwrap: str, source: Mapping[str, str]) -> tuple[Any, ...]:
    """Identify the executable, credentials and namespaces relevant to bwrap."""
    path = Path(bwrap).resolve()
    stat_result = path.stat()

    def _namespace_inode(name: str) -> int:
        try:
            return Path(f"/proc/self/ns/{name}").stat().st_ino
        except OSError:
            return 0

    def _sysctl_value(pathname: str) -> str:
        try:
            return Path(pathname).read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    return (
        str(path),
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_mtime_ns,
        stat_result.st_size,
        os.getuid(),
        os.geteuid(),
        os.getgid(),
        os.getegid(),
        _namespace_inode("user"),
        _namespace_inode("mnt"),
        _sysctl_value("/proc/sys/kernel/unprivileged_userns_clone"),
        _sysctl_value("/proc/sys/user/max_user_namespaces"),
        source.get("PATH", ""),
        source.get("LD_LIBRARY_PATH", ""),
        source.get("LD_PRELOAD", ""),
    )


def _run_bwrap_probe(
    bwrap: str,
    *,
    source: Mapping[str, str],
    runner: Callable[..., Any],
) -> bool:
    """Execute the namespace and root bind that Codex requires."""
    command = [
        bwrap,
        "--unshare-user",
        "--unshare-net",
        "--ro-bind",
        "/",
        "/",
        "/bin/true",
    ]
    try:
        completed = runner(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=_BWRAP_PROBE_TIMEOUT_SEC,
            check=False,
            env=dict(source),
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return getattr(completed, "returncode", 1) == 0


def probe_codex_sandbox_capability(
    *,
    env: dict[str, str] | None = None,
    bwrap_resolver: Callable[..., str | None] = shutil.which,
    runner: Callable[..., Any] = subprocess.run,
    use_cache: bool = True,
) -> bool:
    """Return whether bubblewrap can create the sandbox Codex needs.

    Resolving the binary is not enough in restricted containers: the executable
    can exist while user/mount namespace setup fails. This probe executes a
    read-only root bind inside a new user namespace. Default probes are cached
    by executable identity, process credentials, namespace identity, relevant
    kernel controls and dynamic-loader environment. Resolver and runner
    injection keep the behavior hermetic in unit tests.
    """
    return _probe_codex_sandbox_capability(
        source=_effective_env(env),
        bwrap_resolver=bwrap_resolver,
        runner=runner,
        use_cache=use_cache,
    )


def _probe_codex_sandbox_capability(
    *,
    source: Mapping[str, str],
    bwrap_resolver: Callable[..., str | None],
    runner: Callable[..., Any],
    use_cache: bool,
) -> bool:
    """Probe bubblewrap from one already-effective environment."""
    try:
        bwrap = bwrap_resolver("bwrap", path=source.get("PATH"))
    except OSError:
        return False
    if not bwrap:
        return False

    should_cache = use_cache and bwrap_resolver is shutil.which and runner is subprocess.run
    if not should_cache:
        return _run_bwrap_probe(bwrap, source=source, runner=runner)
    try:
        cache_key = _bwrap_probe_cache_key(bwrap, source)
    except OSError:
        return False
    with _BWRAP_PROBE_CACHE_LOCK:
        if cache_key not in _BWRAP_PROBE_CACHE:
            _BWRAP_PROBE_CACHE[cache_key] = _run_bwrap_probe(
                bwrap,
                source=source,
                runner=runner,
            )
        return _BWRAP_PROBE_CACHE[cache_key]


def codex_sandbox(sdk: Any, *, writable_roots: Sequence[Path], sandbox_mode: str) -> Any:
    """Map a sandbox mode and the requested write scope onto a Codex preset.

    A confirmed ``bypass`` always means ``Sandbox.full_access`` because the
    external sandbox is authoritative in that mode. The contained modes retain
    least-privilege semantics: no writable roots yields ``read_only``, and an
    explicit ``read-only`` mode is a ceiling a declared root cannot raise.

    Args:
        sdk (Any): The loaded ``openai_codex`` module.
        writable_roots (Sequence[Path]): Extra roots the session may write.
        sandbox_mode (str): A mode already resolved by
            :func:`resolve_codex_sandbox_mode`.

    Returns:
        Any: The ``Sandbox`` preset to run the session under.

    Raises:
        CodexSessionUnavailableError: If ``sandbox_mode`` names no known mode.
    """
    mode = _validated_sandbox_mode(sandbox_mode)
    if mode == "bypass":
        return sdk.Sandbox.full_access
    if not writable_roots or mode == "read-only":
        return sdk.Sandbox.read_only
    return sdk.Sandbox.workspace_write


def _path_is_within(path: Path, root: Path) -> bool:
    """Return whether ``path`` is ``root`` or one of its descendants."""
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _path_is_in_source_tree(path: Path, source: Mapping[str, str]) -> bool:
    """Reject CODEX_HOME parents inside an operator/shared source checkout."""
    resolved = path.resolve()
    repo_root = (source.get("REPO_ROOT") or "").strip()
    if repo_root and _path_is_within(resolved, Path(repo_root).resolve()):
        return True
    for candidate in (resolved, *resolved.parents):
        if (candidate / ".git").exists():
            return True
    return False


def _codex_home_parent(
    *,
    cwd: Path,
    writable_roots: Sequence[Path],
    source: Mapping[str, str],
) -> Path:
    """Choose a writable runtime/output parent outside source checkouts."""
    configured = (source.get(HYPERLOOM_RUNTIME_DIR_ENV) or "").strip()
    if configured:
        parent = Path(configured).resolve()
        if _path_is_in_source_tree(parent, source):
            raise CodexSessionUnavailableError(
                f"{HYPERLOOM_RUNTIME_DIR_ENV} points inside a source checkout; "
                "Codex state requires a private runtime/output location"
            )
        try:
            parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        except OSError as exc:
            raise CodexSessionUnavailableError(
                f"cannot create {HYPERLOOM_RUNTIME_DIR_ENV} directory {parent}: {exc}"
            ) from exc
        if not parent.is_dir():
            raise CodexSessionUnavailableError(f"{HYPERLOOM_RUNTIME_DIR_ENV} is not a directory: {parent}")
        return parent

    for root in writable_roots:
        candidate = Path(root).resolve()
        if candidate.is_dir() and not _path_is_in_source_tree(candidate, source):
            return candidate

    fallback = Path(cwd).resolve()
    if not fallback.is_dir():
        raise CodexSessionUnavailableError(f"Codex cwd is not a directory: {fallback}")
    if _path_is_in_source_tree(fallback, source):
        raise CodexSessionUnavailableError(
            f"no safe CODEX_HOME parent is available outside the source checkout; "
            f"set {HYPERLOOM_RUNTIME_DIR_ENV} to a private runtime directory"
        )
    return fallback


@contextlib.contextmanager
def _private_codex_home(
    *,
    cwd: Path,
    writable_roots: Sequence[Path],
    source: Mapping[str, str],
) -> Iterator[Path]:
    """Create and clean up one private mode-0700 Codex state directory."""
    parent = _codex_home_parent(cwd=cwd, writable_roots=writable_roots, source=source)
    try:
        temporary = tempfile.TemporaryDirectory(prefix=".hyperloom-codex-home-", dir=parent)
    except OSError as exc:
        raise CodexSessionUnavailableError(f"cannot create private CODEX_HOME under {parent}: {exc}") from exc
    codex_home = Path(temporary.name)
    try:
        codex_home.chmod(0o700)
    except OSError as exc:
        temporary.cleanup()
        raise CodexSessionUnavailableError(f"cannot secure private CODEX_HOME {codex_home}: {exc}") from exc
    try:
        yield codex_home
    finally:
        temporary.cleanup()


def _item_dict(item: Any) -> dict[str, Any]:
    """Convert one typed SDK thread item into a JSON-ready mapping."""
    root = getattr(item, "root", item)
    if hasattr(root, "model_dump"):
        dumped = root.model_dump(by_alias=True, mode="json")
        return dumped if isinstance(dumped, dict) else {}
    return dict(root) if isinstance(root, dict) else {}


def normalize_codex_items(result: Any) -> tuple[dict[str, Any], ...]:
    """Normalize the typed thread items of one turn into JSON-ready mappings.

    Args:
        result (Any): A ``TurnResult`` (or a stand-in exposing ``items``).

    Returns:
        tuple[dict[str, Any], ...]: One mapping per item, in stream order.
    """
    return tuple(_item_dict(item) for item in (getattr(result, "items", None) or ()))


def codex_item_type(item: dict[str, Any]) -> str:
    """Return an item's SDK type folded to a separator-insensitive key.

    The SDK emits camelCase (``commandExecution``); folding to ``commandexecution``
    keeps callers independent of that spelling.
    """
    return str(item.get("type") or "").replace("_", "").lower()


def describe_codex_item(item: dict[str, Any]) -> str:
    """Summarize one normalized thread item as a single log line.

    Args:
        item (dict[str, Any]): A mapping from :func:`normalize_codex_items`.

    Returns:
        str: A one-line summary, or ``""`` for items with nothing to report.
    """
    kind = codex_item_type(item)
    if kind == "agentmessage":
        return str(item.get("text") or "").strip()
    if kind == "commandexecution":
        command = str(item.get("command") or "").strip()
        if not command:
            return ""
        exit_code = item.get("exitCode")
        suffix = "" if exit_code is None else f" (exit {exit_code})"
        return f"$ {command}{suffix}"
    if kind == "filechange":
        paths = codex_file_changes(item)
        return f"wrote {', '.join(paths)}" if paths else ""
    return ""


def codex_file_changes(item: dict[str, Any]) -> tuple[str, ...]:
    """Return the paths a ``fileChange`` item reports, in order."""
    changes = item.get("changes")
    if not isinstance(changes, (list, tuple)):
        return ()
    return tuple(
        change["path"] for change in changes if isinstance(change, dict) and isinstance(change.get("path"), str)
    )


def _usage_int(payload: dict[str, Any], key: str) -> int:
    """Read one non-negative token count, treating unusable values as 0.

    Token accounting is diagnostic, so a malformed count must not abort a run
    that otherwise produced its artifacts.
    """
    value = payload.get(key)
    if isinstance(value, bool):
        return 0
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
    return parsed if parsed >= 0 else 0


def normalize_codex_usage(usage: Any) -> dict[str, int]:
    """Normalize ``ThreadTokenUsage`` into Hyperloom's transcript usage fields.

    Only the ``last`` breakdown is read: it covers the turn that just ran,
    whereas ``total`` accumulates across the whole thread. Reasoning tokens are
    reported separately because on a reasoning model they dominate the output
    budget and are invisible in the response text.

    Args:
        usage (Any): The SDK usage object, or ``None``.

    Returns:
        dict[str, int]: Token counts, or ``{}`` when the SDK reported none.
    """
    if usage is None:
        return {}
    breakdown = usage.get("last", usage) if isinstance(usage, dict) else getattr(usage, "last", usage)
    if hasattr(breakdown, "model_dump"):
        breakdown = breakdown.model_dump()
    if not isinstance(breakdown, dict):
        return {}
    return {
        "input_tokens": _usage_int(breakdown, "input_tokens"),
        "output_tokens": _usage_int(breakdown, "output_tokens"),
        "cache_read_input_tokens": _usage_int(breakdown, "cached_input_tokens"),
        "reasoning_output_tokens": _usage_int(breakdown, "reasoning_output_tokens"),
    }


def _turn_error_message(result: Any) -> str:
    """Extract the in-band SDK error message from a completed turn."""
    error = getattr(result, "error", None)
    if error is None:
        return ""
    return str(getattr(error, "message", None) or error)


def normalize_codex_result(result: Any, thread_id: str) -> CodexSessionResult:
    """Normalize one completed SDK turn into a :class:`CodexSessionResult`.

    Args:
        result (Any): The SDK ``TurnResult``.
        thread_id (str): The thread handle the turn ran on.

    Returns:
        CodexSessionResult: The normalized turn outcome.
    """
    return CodexSessionResult(
        text=str(getattr(result, "final_response", "") or "").strip(),
        items=normalize_codex_items(result),
        usage=normalize_codex_usage(getattr(result, "usage", None)),
        thread_id=thread_id,
        error=_turn_error_message(result),
    )


async def run_codex_turn(
    *,
    prompt: str,
    developer_instructions: str,
    cwd: Path,
    model: str,
    timeout_sec: float,
    writable_roots: Sequence[Path] = (),
    sandbox_mode: str = "",
    api_key_env: str = "OPENAI_API_KEY",
    base_url_env: str = "OPENAI_BASE_URL",
    codex_bin: str = "",
    env: dict[str, str] | None = None,
) -> CodexSessionResult:
    """Run one non-interactive Codex turn and normalize its typed result.

    The turn runs with ``ApprovalMode.deny_all`` (no approval prompt can block
    an unattended run). Thread and turn receive the exact same resolved sandbox
    preset. Contained presets first pass a real bubblewrap capability probe;
    failure never falls back to ``full_access``. ``CODEX_HOME`` is a private
    mode-0700 directory below ``HYPERLOOM_RUNTIME_DIR``, the first safe writable
    root, or a run-local ``cwd`` fallback, and is removed after the SDK closes.

    Args:
        prompt (str): The user prompt for the turn.
        developer_instructions (str): System-level instructions for the thread.
        cwd (Path): Working directory for the thread; writable under the
            ``workspace_write`` and ``full_access`` sandboxes.
        model (str): The Codex model id.
        timeout_sec (float): Wall-clock budget for the turn. On expiry the turn
            is interrupted and :class:`CodexSessionTimeoutError` is raised.
        writable_roots (Sequence[Path]): Extra roots the session may write.
            Empty keeps contained sessions read-only; confirmed ``bypass``
            remains full access because external isolation is authoritative.
        sandbox_mode (str): One of :data:`CODEX_SANDBOX_MODES`. Blank defers to
            :data:`CODEX_SANDBOX_MODE_ENV`, then to
            :data:`DEFAULT_CODEX_SANDBOX_MODE`.
        api_key_env (str): Preferred API-key env var name.
        base_url_env (str): Preferred base-URL env var name.
        codex_bin (str): Optional Codex runtime path; the SDK resolves its own
            when empty.
        env (dict[str, str] | None): Values overlaid on ``os.environ``. The
            resulting mapping drives policy, credentials, config and child
            process launch.

    Returns:
        CodexSessionResult: The normalized turn outcome.

    Raises:
        CodexSessionUnavailableError: If the SDK is missing, or its gateway
            config or sandbox mode is unusable.
        CodexSessionTimeoutError: If the turn outlived ``timeout_sec``.
        CodexSessionError: If the SDK failed for any other reason.
    """
    effective_env = _effective_env(env)
    resolved_sandbox_mode = _resolve_codex_sandbox_mode(
        sandbox_mode=sandbox_mode,
        source=effective_env,
    )
    if resolved_sandbox_mode != "bypass" and not _probe_codex_sandbox_capability(
        source=effective_env,
        bwrap_resolver=shutil.which,
        runner=subprocess.run,
        use_cache=True,
    ):
        raise CodexSessionUnavailableError(
            f"Codex sandbox mode {resolved_sandbox_mode!r} requires a working bubblewrap sandbox, "
            "but the capability probe failed; refusing to fall back to bypass"
        )

    provider_config = _resolve_codex_provider_config(
        api_key_env=api_key_env,
        base_url_env=base_url_env,
        source=effective_env,
    )
    sdk = load_codex_sdk()
    sandbox = codex_sandbox(
        sdk,
        writable_roots=writable_roots,
        sandbox_mode=resolved_sandbox_mode,
    )
    config_overrides = (
        "features.memories=false",
        *provider_config.overrides,
        *_writable_root_overrides(writable_roots),
    )
    thread_id = ""
    turn_task: asyncio.Task[Any] | None = None

    with _private_codex_home(
        cwd=cwd,
        writable_roots=writable_roots,
        source=effective_env,
    ) as codex_home:
        child_env = effective_env.copy()
        child_env.update(provider_config.env_additions)
        child_env["CODEX_HOME"] = str(codex_home)
        config = sdk.CodexConfig(
            codex_bin=codex_bin or None,
            config_overrides=config_overrides,
            cwd=str(cwd),
            env=child_env,
            client_name=_CLIENT_NAME,
            client_title=_CLIENT_TITLE,
        )
        try:
            async with sdk.AsyncCodex(config) as client:
                thread = await client.thread_start(
                    approval_mode=sdk.ApprovalMode.deny_all,
                    cwd=str(cwd),
                    developer_instructions=developer_instructions,
                    model=model,
                    model_provider=CODEX_PROVIDER_NAME,
                    sandbox=sandbox,
                )
                thread_id = str(getattr(thread, "id", "") or "")
                turn_handle = await thread.turn(
                    prompt,
                    approval_mode=sdk.ApprovalMode.deny_all,
                    cwd=str(cwd),
                    model=model,
                    sandbox=sandbox,
                )
                turn_task = asyncio.create_task(turn_handle.run())
                completed, _pending = await asyncio.wait({turn_task}, timeout=timeout_sec)
                if not completed:
                    # Teardown of an already-failed turn: the timeout below is
                    # the reported failure, so interrupt errors add no signal.
                    with contextlib.suppress(Exception):
                        await asyncio.wait_for(turn_handle.interrupt(), timeout=_INTERRUPT_TIMEOUT_SEC)
                    with contextlib.suppress(Exception):
                        await asyncio.wait_for(asyncio.shield(turn_task), timeout=_INTERRUPT_TIMEOUT_SEC)
                    raise CodexSessionTimeoutError(f"Codex turn timed out after {timeout_sec:g}s")
                sdk_result = turn_task.result()
        except CodexSessionError:
            raise
        except Exception as exc:
            raise CodexSessionError(f"Codex SDK turn failed: {exc}") from exc
        finally:
            if turn_task is not None and not turn_task.done():
                turn_task.cancel()
                # Let the cancellation settle. The turn's own outcome is already
                # decided, so the task's result is discarded rather than raised;
                # a cancellation of *this* task still propagates.
                await asyncio.gather(turn_task, return_exceptions=True)

    return normalize_codex_result(sdk_result, thread_id)


__all__ = [
    "CODEX_EXTERNAL_SANDBOX_ENV",
    "CODEX_PROVIDER_NAME",
    "CODEX_SANDBOX_MODES",
    "CODEX_SANDBOX_MODE_ENV",
    "CodexProviderConfig",
    "CodexSessionError",
    "CodexSessionResult",
    "CodexSessionTimeoutError",
    "CodexSessionUnavailableError",
    "DEFAULT_CODEX_SANDBOX_MODE",
    "HYPERLOOM_RUNTIME_DIR_ENV",
    "api_key_env_name",
    "codex_file_changes",
    "codex_item_type",
    "codex_provider_overrides",
    "codex_sandbox",
    "describe_codex_item",
    "load_codex_sdk",
    "normalize_codex_items",
    "normalize_codex_result",
    "normalize_codex_usage",
    "probe_codex_sandbox_capability",
    "resolve_codex_provider_config",
    "resolve_codex_sandbox_mode",
    "run_codex_turn",
]
