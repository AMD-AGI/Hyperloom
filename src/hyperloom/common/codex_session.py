# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Codex Agent SDK sessions for Hyperloom's OpenAI-side runners."""

from __future__ import annotations

import asyncio
import contextlib
import errno
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from hyperloom_common.coerce import to_int
from hyperloom_common.llm.llm_attribution import inject_env as inject_attribution_env
from hyperloom_common.llm.llm_config import LLMConfigError, resolve_openai_client_config
from hyperloom_common.llm.llm_headers import parse_unexpanded_headers
from hyperloom_common.llm.token_usage import uncached_input_tokens

# Name Codex records the gateway under in its own TOML config.
CODEX_PROVIDER_NAME = "hyperloom"

_CLIENT_NAME = "hyperloom"
_CLIENT_TITLE = "Hyperloom"

# TOML bare-key charset.
_TOML_BARE_KEY_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_EXACT_ENV_REF_RE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")
_PRIVATE_HEADER_ENV_PREFIX = "HYPERLOOM_CODEX_HTTP_HEADER_"

# Sandbox preset selector.
CODEX_SANDBOX_MODE_ENV = "HYPERLOOM_CODEX_SANDBOX_MODE"
DEFAULT_CODEX_SANDBOX_MODE = "workspace-write"
# Ordered so the error raised for an unknown mode lists them predictably.
CODEX_SANDBOX_MODES: tuple[str, ...] = ("bypass", "workspace-write", "read-only")

# Private per-run Codex state is created below this directory when configured.
HYPERLOOM_RUNTIME_DIR_ENV = "HYPERLOOM_RUNTIME_DIR"

# The probe mirrors the mount/user-namespace operations Codex's bubblewrap sandbox needs.
_BWRAP_PROBE_TIMEOUT_SEC = 10.0
_BWRAP_PROBE_CACHE: dict[tuple[Any, ...], bool] = {}
_BWRAP_PROBE_CACHE_LOCK = threading.Lock()

# Grace period for tearing a timed-out turn down before giving up on it.
_INTERRUPT_TIMEOUT_SEC = 5.0

# AsyncCodex can finish closing before a short-lived helper has released or stopped writing CODEX_HOME.
_CODEX_HOME_CLEANUP_TIMEOUT_SEC = 1.0
_CODEX_HOME_CLEANUP_GRACE_SEC = 0.1
_CODEX_HOME_CLEANUP_SETTLE_SEC = 0.05
_CODEX_HOME_CLEANUP_INITIAL_BACKOFF_SEC = 0.02
_CODEX_HOME_CLEANUP_MAX_BACKOFF_SEC = 0.2
_CODEX_HOME_TRANSIENT_ERRNOS = frozenset({errno.ENOTEMPTY, errno.EBUSY})


class CodexSessionError(RuntimeError):
    """Raised when a Codex SDK turn cannot start or does not complete."""


class CodexSessionUnavailableError(CodexSessionError):
    """Raised when the Codex SDK is missing or its configuration is unusable."""


class CodexSessionTimeoutError(CodexSessionError):
    """Raised when a Codex turn outlived its timeout and was interrupted."""


class CodexHomeCleanupError(CodexSessionError):
    """Raised when a private CODEX_HOME cannot be removed within the bound."""

    def __init__(self, path: Path, status: str) -> None:
        super().__init__(f"CODEX_HOME cleanup failed for {path}: {status}")


@dataclass(frozen=True)
class CodexSessionResult:
    """Normalized outcome of one Codex SDK turn."""

    text: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    error: str = ""


@dataclass(frozen=True)
class CodexProviderConfig:
    """Secret-safe provider settings resolved for one Codex child process."""

    overrides: tuple[str, ...]
    env_additions: tuple[tuple[str, str], ...] = field(default=(), repr=False)


def _effective_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Overlay caller values on the process environment exactly once."""
    effective = os.environ.copy()
    if env is not None:
        effective.update(env)
    return effective


def load_codex_sdk() -> Any:
    """Import ``openai_codex`` lazily and return the module."""
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


def resolve_codex_provider_config(*, env: dict[str, str] | None = None) -> CodexProviderConfig:
    """Resolve env-backed ``model_providers`` settings for the Codex gateway."""
    return _resolve_codex_provider_config(_effective_env(env))


def _resolve_codex_provider_config(source: Mapping[str, str]) -> CodexProviderConfig:
    """Resolve provider config from one already-effective environment."""
    try:
        config = resolve_openai_client_config(env=dict(source))
    except LLMConfigError as exc:
        raise CodexSessionUnavailableError(f"Codex gateway credential is missing: {exc}") from exc
    if not config.base_url:
        raise CodexSessionUnavailableError(
            "OPENAI_BASE_URL is not set; Codex needs an explicit OpenAI-compatible gateway base URL"
        )

    if not (source.get("OPENAI_API_KEY") or "").strip():
        raise CodexSessionUnavailableError("OPENAI_API_KEY is not set in env; Codex cannot authenticate")
    provider = CODEX_PROVIDER_NAME
    overrides = [
        f"model_provider={_toml_string(provider)}",
        f"model_providers.{provider}.name={_toml_string(provider)}",
        f"model_providers.{provider}.base_url={_toml_string(config.base_url)}",
        f"model_providers.{provider}.wire_api={_toml_string('responses')}",
        f"model_providers.{provider}.env_key={_toml_string('OPENAI_API_KEY')}",
    ]
    raw_headers = parse_unexpanded_headers(source.get(config.headers_env))
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


def _writable_root_overrides(writable_roots: Sequence[Path]) -> tuple[str, ...]:
    """Widen the ``workspace_write`` sandbox to the given roots."""
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
    """Resolve which Codex sandbox preset family a session may use."""
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


def _run_bwrap_probe(bwrap: str, *, source: Mapping[str, str]) -> bool:
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
        completed = subprocess.run(
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
    return completed.returncode == 0


def probe_codex_sandbox_capability(*, env: dict[str, str] | None = None) -> bool:
    """Return whether bubblewrap can create the sandbox Codex needs."""
    return _probe_codex_sandbox_capability(source=_effective_env(env))


def _probe_codex_sandbox_capability(*, source: Mapping[str, str]) -> bool:
    """Probe bubblewrap from one already-effective environment."""
    bwrap = shutil.which("bwrap", path=source.get("PATH"))
    if not bwrap:
        return False
    try:
        cache_key = _bwrap_probe_cache_key(bwrap, source)
    except OSError:
        return False
    with _BWRAP_PROBE_CACHE_LOCK:
        if cache_key not in _BWRAP_PROBE_CACHE:
            _BWRAP_PROBE_CACHE[cache_key] = _run_bwrap_probe(bwrap, source=source)
        return _BWRAP_PROBE_CACHE[cache_key]


def codex_sandbox(sdk: Any, *, writable_roots: Sequence[Path], sandbox_mode: str) -> Any:
    """Map a sandbox mode and the requested write scope onto a Codex preset."""
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


def _cleanup_codex_home(temporary: tempfile.TemporaryDirectory[str]) -> None:
    """Remove exactly one generated CODEX_HOME despite transient late writers."""
    path = Path(temporary.name)
    timeout = max(0.0, _CODEX_HOME_CLEANUP_TIMEOUT_SEC)
    grace = max(0.0, _CODEX_HOME_CLEANUP_GRACE_SEC)
    settle = max(0.0, _CODEX_HOME_CLEANUP_SETTLE_SEC)
    initial_backoff = max(0.001, _CODEX_HOME_CLEANUP_INITIAL_BACKOFF_SEC)
    maximum_backoff = max(initial_backoff, _CODEX_HOME_CLEANUP_MAX_BACKOFF_SEC)
    deadline = time.monotonic() + timeout

    def _sleep_within_deadline(duration: float) -> None:
        remaining = max(0.0, deadline - time.monotonic())
        delay = min(max(0.0, duration), remaining)
        if delay:
            time.sleep(delay)

    _sleep_within_deadline(grace)
    backoff = initial_backoff
    last_status = "still-present"
    while True:
        try:
            temporary.cleanup()
        except FileNotFoundError:
            last_status = "already-removed"
        except OSError as exc:
            if exc.errno not in _CODEX_HOME_TRANSIENT_ERRNOS:
                error_number = exc.errno if exc.errno is not None else "unknown"
                raise CodexHomeCleanupError(path, f"non-transient-error-errno-{error_number}") from None
            last_status = "directory-not-empty" if exc.errno == errno.ENOTEMPTY else "resource-busy"
        else:
            last_status = "removed"

        if not path.exists():
            _sleep_within_deadline(settle)
            if not path.exists():
                return
            last_status = "recreated-by-late-writer"

        if time.monotonic() >= deadline:
            raise CodexHomeCleanupError(path, f"{last_status}-after-{timeout:g}s")
        _sleep_within_deadline(backoff)
        backoff = min(maximum_backoff, backoff * 2)


@contextlib.contextmanager
def _private_codex_home(
    *,
    cwd: Path,
    writable_roots: Sequence[Path],
    source: Mapping[str, str],
) -> Iterator[Path]:
    """Create and deterministically clean one private mode-0700 state directory."""
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
        _cleanup_codex_home(temporary)


def _usage_int(payload: dict[str, Any], key: str) -> int:
    """Read one non-negative token count, treating unusable values as 0."""
    return max(0, to_int(payload.get(key), 0))


def normalize_codex_usage(usage: Any) -> dict[str, int]:
    """Normalize ``ThreadTokenUsage`` into Hyperloom's token usage fields."""
    if usage is None:
        return {}
    breakdown = usage.get("last", usage) if isinstance(usage, dict) else getattr(usage, "last", usage)
    if hasattr(breakdown, "model_dump"):
        breakdown = breakdown.model_dump()
    if not isinstance(breakdown, dict):
        return {}
    cached = _usage_int(breakdown, "cached_input_tokens")
    return {
        "input_tokens": uncached_input_tokens(_usage_int(breakdown, "input_tokens"), cached),
        "output_tokens": _usage_int(breakdown, "output_tokens"),
        "cache_read_input_tokens": cached,
        "reasoning_output_tokens": _usage_int(breakdown, "reasoning_output_tokens"),
    }


def _turn_error_message(result: Any) -> str:
    """Extract the in-band SDK error message from a completed turn."""
    error = getattr(result, "error", None)
    if error is None:
        return ""
    return str(getattr(error, "message", None) or error)


def normalize_codex_result(result: Any) -> CodexSessionResult:
    """Normalize one completed SDK turn into a :class:`CodexSessionResult`."""
    return CodexSessionResult(
        text=str(getattr(result, "final_response", "") or "").strip(),
        usage=normalize_codex_usage(getattr(result, "usage", None)),
        error=_turn_error_message(result),
    )


@dataclass(frozen=True)
class CodexSessionConfig:
    """The runtime one Codex session runs in: workspace, model and the roots it may write."""

    cwd: Path
    model: str
    writable_roots: Sequence[Path] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "cwd", Path(self.cwd))
        object.__setattr__(self, "writable_roots", tuple(self.writable_roots))


@dataclass(frozen=True)
class CodexTurnRequest:
    """One prompt sent on a Codex thread, its wall-clock budget and an optional enforced output schema."""

    prompt: str
    timeout_sec: float
    output_schema: dict[str, Any] | None = None


class CodexSession:
    """One Codex Agent SDK runtime held open across many turns.

    ``developer_instructions`` scope the conversation thread and may change between threads.
    """

    def __init__(
        self,
        config: CodexSessionConfig,
        *,
        developer_instructions: str = "",
        component: str = "",
        operation: str = "",
    ) -> None:
        self.config = config
        self.developer_instructions = developer_instructions
        self.component = component
        self.operation = operation
        self._stack: contextlib.AsyncExitStack | None = None
        self._sdk: Any | None = None
        self._sandbox: Any = None
        self._client: Any = None
        self._thread: Any = None

    async def __aenter__(self) -> "CodexSession":
        """Start the session and return it."""
        await self.start()
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        """Close the session."""
        await self.aclose()

    async def start(self) -> None:
        """Resolve policy and credentials, then open the SDK client."""
        if self._stack is not None:
            return
        effective_env = _effective_env()
        # Tag before provider resolution so the header is mapped into ``env_http_headers`` with the gateway's own.
        if self.component:
            inject_attribution_env(
                effective_env,
                component=self.component,
                operation=self.operation,
            )
        resolved_sandbox_mode = _resolve_codex_sandbox_mode(sandbox_mode="", source=effective_env)
        if resolved_sandbox_mode != "bypass" and not _probe_codex_sandbox_capability(source=effective_env):
            raise CodexSessionUnavailableError(
                f"Codex sandbox mode {resolved_sandbox_mode!r} requires a working bubblewrap sandbox, "
                "but the capability probe failed; refusing to fall back to bypass"
            )

        provider_config = _resolve_codex_provider_config(effective_env)
        sdk = load_codex_sdk()
        sandbox = codex_sandbox(
            sdk,
            writable_roots=self.config.writable_roots,
            sandbox_mode=resolved_sandbox_mode,
        )
        config_overrides = (
            "features.memories=false",
            *provider_config.overrides,
            *_writable_root_overrides(self.config.writable_roots),
        )
        # The client is entered inside the CODEX_HOME context so unwinding closes the client first and only then
        # removes the state it writes.
        stack = contextlib.AsyncExitStack()
        try:
            codex_home = stack.enter_context(
                _private_codex_home(
                    cwd=self.config.cwd,
                    writable_roots=self.config.writable_roots,
                    source=effective_env,
                )
            )
            child_env = effective_env.copy()
            child_env.update(provider_config.env_additions)
            child_env["CODEX_HOME"] = str(codex_home)
            config = sdk.CodexConfig(
                config_overrides=config_overrides,
                cwd=str(self.config.cwd),
                env=child_env,
                client_name=_CLIENT_NAME,
                client_title=_CLIENT_TITLE,
            )
            client = await stack.enter_async_context(sdk.AsyncCodex(config))
        except CodexSessionError:
            await stack.aclose()
            raise
        except Exception as exc:
            await stack.aclose()
            raise CodexSessionError(f"Codex SDK session failed to start: {exc}") from exc
        self._stack = stack
        self._sdk = sdk
        self._sandbox = sandbox
        self._client = client

    def reset_thread(self) -> None:
        """Drop the open conversation; the next turn opens a fresh thread."""
        self._thread = None

    async def aclose(self) -> None:
        """Close the SDK client and remove the private state directory."""
        stack, self._stack = self._stack, None
        self._sdk = None
        self._sandbox = None
        self._client = None
        self.reset_thread()
        if stack is not None:
            await stack.aclose()

    async def _open_thread(self) -> Any:
        """Return the open conversation, opening one on first use."""
        if self._client is None or self._sdk is None:
            raise CodexSessionError("Codex session is not started; call start() first")
        if self._thread is None:
            self._thread = await self._client.thread_start(
                approval_mode=self._sdk.ApprovalMode.deny_all,
                cwd=str(self.config.cwd),
                developer_instructions=self.developer_instructions,
                model=self.config.model,
                model_provider=CODEX_PROVIDER_NAME,
                sandbox=self._sandbox,
            )
        return self._thread

    async def turn(self, request: CodexTurnRequest) -> CodexSessionResult:
        """Run one turn on this session's conversation."""
        # Both, not just the client: ``start()`` sets them together and ``aclose()`` clears them together, and
        # ``_open_thread`` reaches for the SDK as well.
        if self._client is None or self._sdk is None:
            raise CodexSessionError("Codex session is not started; call start() before turn()")
        turn_task: asyncio.Task[Any] | None = None
        try:
            thread = await self._open_thread()
            turn_handle = await thread.turn(
                request.prompt,
                approval_mode=self._sdk.ApprovalMode.deny_all,
                cwd=str(self.config.cwd),
                model=self.config.model,
                output_schema=request.output_schema,
                sandbox=self._sandbox,
            )
            turn_task = asyncio.create_task(turn_handle.run())
            completed, _pending = await asyncio.wait({turn_task}, timeout=request.timeout_sec)
            if not completed:
                # Teardown of an already-failed turn: the timeout below is the reported failure, so interrupt errors
                # add no signal.
                with contextlib.suppress(self._sdk.CodexError, OSError, asyncio.TimeoutError):
                    await asyncio.wait_for(turn_handle.interrupt(), timeout=_INTERRUPT_TIMEOUT_SEC)
                with contextlib.suppress(self._sdk.CodexError, RuntimeError, asyncio.TimeoutError):
                    await asyncio.wait_for(asyncio.shield(turn_task), timeout=_INTERRUPT_TIMEOUT_SEC)
                raise CodexSessionTimeoutError(f"Codex turn timed out after {request.timeout_sec:g}s")
            sdk_result = turn_task.result()
        except CodexSessionError:
            raise
        except Exception as exc:
            raise CodexSessionError(f"Codex SDK turn failed: {exc}") from exc
        finally:
            if turn_task is not None and not turn_task.done():
                turn_task.cancel()
                # Let the cancellation settle.
                await asyncio.gather(turn_task, return_exceptions=True)
        return normalize_codex_result(sdk_result)


async def run_codex_turn(
    *,
    config: CodexSessionConfig,
    request: CodexTurnRequest,
    developer_instructions: str,
    component: str = "",
    operation: str = "",
) -> CodexSessionResult:
    """Run one non-interactive Codex turn in a session of its own."""
    session = CodexSession(
        config,
        developer_instructions=developer_instructions,
        component=component,
        operation=operation,
    )
    await session.start()
    try:
        return await session.turn(request)
    finally:
        await session.aclose()


__all__ = [
    "CODEX_PROVIDER_NAME",
    "CODEX_SANDBOX_MODES",
    "CODEX_SANDBOX_MODE_ENV",
    "CodexHomeCleanupError",
    "CodexProviderConfig",
    "CodexSession",
    "CodexSessionConfig",
    "CodexSessionError",
    "CodexSessionResult",
    "CodexSessionTimeoutError",
    "CodexTurnRequest",
    "CodexSessionUnavailableError",
    "DEFAULT_CODEX_SANDBOX_MODE",
    "HYPERLOOM_RUNTIME_DIR_ENV",
    "codex_sandbox",
    "load_codex_sdk",
    "normalize_codex_result",
    "normalize_codex_usage",
    "probe_codex_sandbox_capability",
    "resolve_codex_provider_config",
    "resolve_codex_sandbox_mode",
    "run_codex_turn",
]
