"""One-attempt SDK driver for the quantization-agent."""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

from hyperloom.common.llm_attribution import sdk_env_overlay


DEFAULT_MODEL = "claude-opus-5"
DEFAULT_ALLOWED_TOOLS = ["Read", "Write", "Edit", "Bash"]
DEFAULT_MAX_TURNS = 240  # Quark workflow has 4 STOPs + validator + eval

SKILL_RELATIVE_PATH = "SKILL.md"
QUARK_PY310_SITE_CUSTOMIZE = """\
import datetime as _datetime
import typing as _typing

from typing_extensions import Self as _Self

_typing.Self = _Self
_datetime.UTC = _datetime.timezone.utc
"""


@dataclass
class AttemptResult:
    """Low-level output of one SDK session."""

    workspace: Path
    sdk_error: str = ""
    raw_text: str = ""
    chunks: list[str] = field(default_factory=list)


def _import_sdk() -> tuple[Any, Any]:
    """Import the Claude Agent SDK and return its query primitives."""
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
    """Return the on-disk path of the quantization agent's ``SKILL.md``."""
    # SKILL.md lives one level up from this module, at the package root.
    root = package_root if package_root is not None else Path(__file__).resolve().parent.parent
    return root / SKILL_RELATIVE_PATH


def _cleanup_quark_py310_compat(compat_dir: Path) -> None:
    """Remove the process-level compatibility shim."""
    try:
        compat_dir.chmod(0o755)
        shutil.rmtree(compat_dir)
    except OSError:
        # Interpreter-exit cleanup; a leftover temp shim must not raise.
        pass


@lru_cache(maxsize=1)
def _prepare_quark_py310_compat() -> Path:
    """Create a workspace-external Python 3.10 compatibility shim for Quark 0.12."""
    compat_dir = Path(tempfile.mkdtemp(prefix="hyperloom_quark_py310_"))
    sitecustomize = compat_dir / "sitecustomize.py"
    sitecustomize.write_text(QUARK_PY310_SITE_CUSTOMIZE, encoding="utf-8")
    sitecustomize.chmod(0o444)
    compat_dir.chmod(0o555)
    atexit.register(_cleanup_quark_py310_compat, compat_dir)
    return compat_dir


def _prepend_pythonpath(path: Path, current: str | None) -> str:
    prefix = str(path)
    return prefix if not current else prefix + os.pathsep + current


def _quark_py310_compat_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Return child-process env exposing Quark's Python 3.10 shim."""

    env = dict(os.environ if base_env is None else base_env)
    compat_dir = _prepare_quark_py310_compat()
    env["PYTHONPATH"] = _prepend_pythonpath(compat_dir, env.get("PYTHONPATH"))
    env["PIP_IGNORE_REQUIRES_PYTHON"] = "1"
    return env


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
    """Assemble the prompt handed to the SDK for one attempt."""

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
- Workspace (WRITE only here; quantized_model_dir must resolve inside this path after Path.resolve(); see SKILL.md §1.1): {workspace}
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
    skill_path: Path | None = None,
    model: str | None = None,
    max_turns: int = DEFAULT_MAX_TURNS,
    allowed_tools: list[str] | None = None,
    sdk_query_factory: Callable[..., Any] | None = None,
    sdk_options_cls: Any | None = None,
    log: Callable[[str], None] | None = None,
) -> AttemptResult:
    """Run one SDK session driving SKILL.md."""
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
        "env": _quark_py310_compat_env(),
    }
    kwargs["env"].update(sdk_env_overlay(component="quantization", operation="quantize_attempt"))
    if model:
        kwargs["model"] = model
    kwargs["cwd"] = str(workspace)

    try:
        options = sdk_options_cls(**kwargs)
    except TypeError:
        # Older SDK builds may not support cwd; prompt + SKILL.md use absolute paths so retrying without cwd is safe.
        kwargs.pop("cwd", None)
        try:
            options = sdk_options_cls(**kwargs)
        except TypeError as env_exc:
            # The Quark py310 shim must be passed to SDK-spawned tools without mutating process-global os.environ
            # across async awaits.
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
    "RunOneAttemptFn",
    "build_attempt_prompt",
    "resolve_skill_path",
    "run_one_attempt",
]
