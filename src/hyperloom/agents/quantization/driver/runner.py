"""One-attempt agent-runtime driver for the quantization-agent.

Claude retains its SDK injection seams (``sdk_query_factory`` /
``sdk_options_cls``); Codex and Hermes are CLI transports. Every provider
receives the same rendered ``SKILL.md`` prompt and stores runtime errors on the
result rather than raising. Hermes starts from the writable workspace while the
read-only Quark root remains pinned in the prompt.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

from hyperloom.common.env_safety import is_secret_shaped_env_name, redact_secret_values
from hyperloom.common.hermes_runtime import hermes_external_sandbox_enabled, resolve_hermes_executable
from hyperloom.common.llm_attribution import sdk_env_overlay


DEFAULT_MODEL = "claude-opus-5"
DEFAULT_CODEX_MODEL = "gpt-5.6-sol"
DEFAULT_HERMES_MODEL = "gpt-5.6-sol"
SUPPORTED_PROVIDERS = ("claude", "codex", "hermes")
CODEX_HOME_ENV = "HYPERLOOM_CODEX_HOME"
DEFAULT_ALLOWED_TOOLS = ["Read", "Write", "Edit", "Bash"]
DEFAULT_MAX_TURNS = 240  # Quark workflow has 4 STOPs + validator + eval

SKILL_RELATIVE_PATH = "SKILL.md"
QUARK_PY310_COMPAT_DIR = ".hyperloom_quark_py310_compat"
QUARK_PY310_SITE_CUSTOMIZE = """\
import datetime as _datetime
import typing as _typing

from typing_extensions import Self as _Self

_typing.Self = _Self
_datetime.UTC = _datetime.timezone.utc
"""


@dataclass
class AttemptResult:
    """Low-level output of one SDK session.

    The classifier consumes ``workspace`` + ``sdk_error`` + ``last_phase``;
    ``raw_text`` is kept for debugging / logging only.
    """

    workspace: Path
    sdk_error: str = ""
    raw_text: str = ""
    chunks: list[str] = field(default_factory=list)


def _import_sdk() -> tuple[Any, Any]:
    """Import the Claude Agent SDK and return its query primitives.

    Returns:
        A ``(query, ClaudeAgentOptions)`` tuple from ``claude_agent_sdk``.

    Raises:
        RuntimeError: If the SDK is not installed or is missing the required
            ``query`` / ``ClaudeAgentOptions`` attributes.
    """
    try:
        import claude_agent_sdk as sdk  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised via injection seams in tests
        raise RuntimeError("claude_agent_sdk not installed; run the quantization-agent installer") from exc
    if not (hasattr(sdk, "query") and hasattr(sdk, "ClaudeAgentOptions")):
        raise RuntimeError("claude_agent_sdk missing query / ClaudeAgentOptions")
    return sdk.query, sdk.ClaudeAgentOptions


def _iter_message_text(message: Any) -> Iterable[str]:
    """Yield the non-empty text fragments of a Claude Agent SDK message."""
    from hyperloom.common.claude_oneshot import message_text  # noqa: PLC0415

    yield from (fragment for fragment in message_text(message) if fragment)


def resolve_skill_path(package_root: Path | None = None) -> Path:
    """Return the on-disk path of the quantization agent's ``SKILL.md``.

    Resolution is centralized here so callers don't hardcode the layout.

    Args:
        package_root: Override for the package root; defaults to the
            parent of this module's directory.

    Returns:
        The path to ``SKILL.md`` under the package root.
    """
    # SKILL.md lives one level up from this module, at the package root.
    root = package_root if package_root is not None else Path(__file__).resolve().parent.parent
    return root / SKILL_RELATIVE_PATH


def _prepare_quark_py310_compat(workspace: Path) -> Path:
    """Create a workspace-local Python 3.10 compatibility shim for Quark 0.12.

    Quark 0.12 uses Python 3.11 symbols (``typing.Self`` and ``datetime.UTC``);
    inject them via ``sitecustomize`` without modifying the Quark checkout.
    """
    compat_dir = workspace / QUARK_PY310_COMPAT_DIR
    compat_dir.mkdir(parents=True, exist_ok=True)
    (compat_dir / "sitecustomize.py").write_text(QUARK_PY310_SITE_CUSTOMIZE, encoding="utf-8")
    return compat_dir


def _prepend_pythonpath(path: Path, current: str | None) -> str:
    prefix = str(path)
    return prefix if not current else prefix + os.pathsep + current


def _quark_py310_compat_env(workspace: Path, base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Return child-process env exposing Quark's Python 3.10 shim."""

    env = dict(os.environ if base_env is None else base_env)
    compat_dir = _prepare_quark_py310_compat(workspace)
    env["PYTHONPATH"] = _prepend_pythonpath(compat_dir, env.get("PYTHONPATH"))
    env["PIP_IGNORE_REQUIRES_PYTHON"] = "1"
    return env


async def _run_cli_attempt(
    *,
    provider: str,
    prompt: str,
    workspace: Path,
    quark_root: Path,
    model: str | None,
    log: Callable[[str], None] | None,
) -> AttemptResult:
    """Run the same skill prompt through Codex or Hermes Agent."""

    env = _quark_py310_compat_env(workspace)
    env.update(sdk_env_overlay(component="quantization", operation="quantize_attempt"))
    if provider == "codex":
        executable = shutil.which("codex")
        selected_model = model or env.get("CODEX_MODEL") or DEFAULT_CODEX_MODEL
        configured_home = str(env.get(CODEX_HOME_ENV) or "").strip()
        if configured_home:
            home_path = Path(configured_home).expanduser()
            if not home_path.is_absolute():
                return AttemptResult(workspace=workspace, sdk_error="native OAuth CODEX_HOME must be absolute")
            if home_path.is_symlink():
                return AttemptResult(workspace=workspace, sdk_error="native OAuth CODEX_HOME cannot be a symlink")
            if not home_path.is_dir():
                return AttemptResult(workspace=workspace, sdk_error="native OAuth CODEX_HOME must already exist")
            env["CODEX_HOME"] = str(home_path.resolve())
            for name in (
                "OPENAI_BASE_URL",
                "OPENAI_API_KEY",
                "OPENAI_CUSTOM_HEADERS",
                "CODEX_API_KEY",
            ):
                env.pop(name, None)
        argv = [
            executable or "codex",
            "exec",
            "--sandbox",
            "workspace-write",
            "--ephemeral",
            "--ignore-rules",
            "-m",
            selected_model,
        ]
        stdin = prompt
    else:
        executable = resolve_hermes_executable()
        if not hermes_external_sandbox_enabled(env):
            return AttemptResult(
                workspace=workspace,
                sdk_error=(
                    "Hermes quantization requires a verifiable outer container; "
                    "set HYPERLOOM_HERMES_EXTERNAL_SANDBOX=1 only inside that boundary"
                ),
            )
        selected_model = model or env.get("HYPERLOOM_HERMES_MODEL") or DEFAULT_HERMES_MODEL
        profile = env.get("HYPERLOOM_HERMES_PROFILE", "hyperloomfaithful")
        inference_provider = env.get("HYPERLOOM_HERMES_PROVIDER", "openai-codex")
        argv = [
            executable or "hermes",
            "--profile",
            profile,
            "--provider",
            inference_provider,
            "--model",
            selected_model,
            "--safe-mode",
            "--toolsets",
            "terminal,file",
            "-z",
            prompt,
        ]
        stdin = None

    if executable is None:
        return AttemptResult(workspace=workspace, sdk_error=f"{provider} executable not found")

    def _invoke() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            argv,
            cwd=workspace,
            env=env,
            input=stdin,
            capture_output=True,
            text=True,
            check=False,
        )

    if log:
        log(f"quantization-agent {provider} runner: workspace={workspace} quark_root={quark_root}")
    try:
        completed = await asyncio.to_thread(_invoke)
    except Exception as exc:  # noqa: BLE001
        return AttemptResult(workspace=workspace, sdk_error=f"{type(exc).__name__}: {exc}")

    def _redact_diagnostic(value: str) -> str:
        redacted = redact_secret_values(value or "")
        for key, secret in env.items():
            if secret and is_secret_shaped_env_name(key):
                redacted = redacted.replace(secret, "[REDACTED]")
        return redacted

    raw_text = _redact_diagnostic(completed.stdout or "")
    stderr_text = _redact_diagnostic(completed.stderr or "")
    if log:
        for line in raw_text.splitlines():
            log(f"[{provider}] {line[:1000]}")
        for line in stderr_text.splitlines():
            log(f"[{provider}] {line[:1000]}")
    sdk_error = ""
    if completed.returncode != 0:
        sdk_error = f"{provider} exited rc={completed.returncode}: {stderr_text.strip()[-1000:]}"
    return AttemptResult(
        workspace=workspace,
        sdk_error=sdk_error,
        raw_text=raw_text,
        chunks=[raw_text] if raw_text else [],
    )


def build_attempt_prompt(
    *,
    user_prompt: str,
    skill_path: Path,
    workspace: Path,
    quark_root: Path,
    attempt_number: int,
    acceptable_eval_gap: float | None,
    interactive: bool | None,
    previous_outcome: str | None,
    fix_hypothesis_path: Path | None,
) -> str:
    """Assemble the prompt handed to the SDK for one attempt.

    Pins the run context (workspace / quark_root / attempt / threshold /
    interactivity) and embeds the verbatim user prompt. Retry attempts also
    reference the prior outcome ID and the fix-hypothesis file so the LLM can
    target the diagnosed cause.

    Args:
        user_prompt: The verbatim user instruction to embed.
        skill_path: Path to ``SKILL.md`` (the runtime contract).
        workspace: Directory where the attempt writes artifacts.
        quark_root: Read-only Quark project root.
        attempt_number: 1-based attempt index.
        acceptable_eval_gap: Caller-supplied eval-gap threshold, if any.
        interactive: Interactivity mode (``None`` = auto).
        previous_outcome: Prior attempt's outcome ID, for retry context.
        fix_hypothesis_path: Path to the prior fix-hypothesis file, if any.

    Returns:
        The fully-rendered prompt string.
    """

    interactive_str = (
        "auto (use stdin if a tty is attached)"
        if interactive is None
        else ("on (always relay checkpoints to operator)" if interactive else "off (batch / non-interactive)")
    )
    threshold_str = (
        f"{acceptable_eval_gap:.4f} (caller-supplied)"
        if acceptable_eval_gap is not None
        else "see SKILL.md §Eval (caller did not override; resolve from eval_gap_threshold.txt or default 0.03)"
    )
    retry_block = ""
    if attempt_number > 1 and previous_outcome:
        hint = (
            f"\n- Fix hypothesis from prior attempt: {fix_hypothesis_path}" if fix_hypothesis_path is not None else ""
        )
        retry_block = (
            f"\n\n## Retry context\nThis is attempt #{attempt_number}. The previous "
            f"attempt ended with outcome `{previous_outcome}`. Diagnose and apply the "
            f"fix you wrote in `fix_hypothesis_attempt_{attempt_number}.md` before "
            f"re-running quark-torch-ptq.{hint}"
        )

    return f"""You are the Hyperloom quantization-agent.

Read and follow the FULL runtime contract in this skill file:
{skill_path}

## Run context (passed in via prompt; SKILL.md tells you what to do with these)
- Workspace (write all your artifacts here): {workspace}
- Quark project root (READ-ONLY; never edit files under this path): {quark_root}
- Attempt number: {attempt_number}
- Acceptable eval gap: {threshold_str}
- Interactive mode: {interactive_str}{retry_block}

## User prompt (verbatim)
{user_prompt}

Begin the workflow now. Do not ask the user clarifying questions unless the
SKILL.md retry/checkpoint protocol explicitly requires it.
"""


async def run_one_attempt(
    *,
    user_prompt: str,
    workspace: Path,
    quark_root: Path,
    attempt_number: int = 1,
    acceptable_eval_gap: float | None = None,
    interactive: bool | None = None,
    previous_outcome: str | None = None,
    provider: str = "claude",
    skill_path: Path | None = None,
    model: str | None = None,
    max_turns: int = DEFAULT_MAX_TURNS,
    allowed_tools: list[str] | None = None,
    sdk_query_factory: Callable[..., Any] | None = None,
    sdk_options_cls: Any | None = None,
    log: Callable[[str], None] | None = None,
) -> AttemptResult:
    """Run one SDK session driving SKILL.md.

    Errors raised by the SDK (rate limits, max turns, network) are captured
    and returned via ``AttemptResult.sdk_error`` rather than propagated, so
    the retry loop can read the workspace state — which often contains valid
    artifacts even when the SDK aborted late.

    Args:
        user_prompt: The verbatim user instruction.
        workspace: Directory for attempt artifacts (created if needed).
        quark_root: Read-only Quark project root.
        attempt_number: 1-based attempt index.
        acceptable_eval_gap: Caller-supplied eval-gap threshold, if any.
        interactive: Interactivity mode (``None`` = auto).
        previous_outcome: Prior attempt's outcome ID, for retry context.
        skill_path: Override for the ``SKILL.md`` path.
        model: Optional model identifier.
        max_turns: Maximum SDK turns for the session.
        allowed_tools: Optional explicit tool allowlist.
        sdk_query_factory: Override for the SDK query callable (testing).
        sdk_options_cls: Override for the SDK options class (testing).
        log: Optional line-logging callback.

    Returns:
        The :class:`AttemptResult` for the session.
    """
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    quark_root = Path(quark_root)

    skill_path = skill_path or resolve_skill_path()
    if not skill_path.is_file():
        raise FileNotFoundError(f"SKILL.md not found at {skill_path}")

    fix_hypothesis_path: Path | None = None
    if attempt_number > 1:
        candidate = workspace / f"fix_hypothesis_attempt_{attempt_number}.md"
        fix_hypothesis_path = candidate if candidate.is_file() else None

    prompt = build_attempt_prompt(
        user_prompt=user_prompt,
        skill_path=skill_path,
        workspace=workspace,
        quark_root=quark_root,
        attempt_number=attempt_number,
        acceptable_eval_gap=acceptable_eval_gap,
        interactive=interactive,
        previous_outcome=previous_outcome,
        fix_hypothesis_path=fix_hypothesis_path,
    )

    provider = provider.strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"unsupported quantization-agent provider: {provider}")
    if provider != "claude":
        return await _run_cli_attempt(
            provider=provider,
            prompt=prompt,
            workspace=workspace,
            quark_root=quark_root,
            model=model,
            log=log,
        )

    if sdk_query_factory is None or sdk_options_cls is None:
        query, options_cls = _import_sdk()
        sdk_query_factory = sdk_query_factory or query
        sdk_options_cls = sdk_options_cls or options_cls

    system_prompt = (
        "You are the Hyperloom quantization-agent. Drive the Quark workflow per "
        "SKILL.md. Never modify files under quark_root. Treat artifact presence "
        "in workspace as the source of truth; do not lie about success."
    )
    kwargs: dict[str, Any] = {
        "max_turns": max_turns,
        "system_prompt": system_prompt,
        "allowed_tools": DEFAULT_ALLOWED_TOOLS if allowed_tools is None else allowed_tools,
        "stderr": (lambda line: log(f"[claude-sdk] {line.rstrip()}")) if log else None,
        "env": _quark_py310_compat_env(workspace),
    }
    kwargs["env"].update(sdk_env_overlay(component="quantization", operation="quantize_attempt"))
    if model:
        kwargs["model"] = model
    kwargs["cwd"] = str(quark_root)

    try:
        options = sdk_options_cls(**kwargs)
    except TypeError:
        # Older SDK builds may not support cwd; prompt + SKILL.md use absolute
        # paths so retrying without cwd is safe.
        kwargs.pop("cwd", None)
        try:
            options = sdk_options_cls(**kwargs)
        except TypeError as env_exc:
            # The Quark py310 shim must be passed to SDK-spawned tools without
            # mutating process-global os.environ across async awaits. If this
            # SDK predates the env option, fail clearly instead of silently
            # running Quark 0.12 in an incompatible Python 3.10 environment.
            raise RuntimeError(
                "claude_agent_sdk.ClaudeAgentOptions does not support env; "
                "upgrade claude-agent-sdk so Hyperloom can pass the Quark "
                "Python 3.10 compatibility shim to SDK subprocesses"
            ) from env_exc

    chunks: list[str] = []
    sdk_error = ""

    if log:
        log(f"quantization-agent SDK runner: workspace={workspace} quark_root={quark_root} attempt={attempt_number}")

    try:
        async for message in sdk_query_factory(prompt=prompt, options=options):
            for text in _iter_message_text(message):
                chunks.append(text)
                if log:
                    log(f"[claude-sdk] {text[:1000]}")
    except Exception as exc:  # noqa: BLE001
        # Capture but don't raise: valid artifacts may exist before the abort.
        sdk_error = f"{type(exc).__name__}: {exc}"
        if log:
            log(f"[claude-sdk] WARNING: {sdk_error}")

    return AttemptResult(
        workspace=workspace,
        sdk_error=sdk_error,
        raw_text="\n".join(chunks),
        chunks=chunks,
    )


# Injection seam used in tests and by driver/retry.py.
RunOneAttemptFn = Callable[..., Awaitable[AttemptResult]]


__all__ = [
    "AttemptResult",
    "DEFAULT_ALLOWED_TOOLS",
    "DEFAULT_MAX_TURNS",
    "DEFAULT_MODEL",
    "DEFAULT_CODEX_MODEL",
    "DEFAULT_HERMES_MODEL",
    "CODEX_HOME_ENV",
    "SUPPORTED_PROVIDERS",
    "RunOneAttemptFn",
    "build_attempt_prompt",
    "resolve_skill_path",
    "run_one_attempt",
]
