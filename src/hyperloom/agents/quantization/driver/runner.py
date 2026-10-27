"""One-attempt SDK driver for the quantization-agent."""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
import traceback
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

from hyperloom.common import llm_config
from hyperloom.agents.quantization.driver.result_collector import CollectedArtifacts, collect_artifacts
from hyperloom.common.deepseek_task import DeepSeekTaskRequest, OwedArtifact, run_deepseek_task
from hyperloom.common.llm_attribution import sdk_env_overlay
from hyperloom.common.llm_config import DEFAULT_CLAUDE_MODEL


DEFAULT_MODEL = DEFAULT_CLAUDE_MODEL
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


@dataclass(frozen=True)
class AttemptContext:
    """One attempt as the agent is told about it: the request, where it works, and how the last attempt ended."""

    user_prompt: str
    workspace: Path
    quark_root: Path
    attempt_number: int = 1
    acceptable_eval_gap: float | None = None
    interactive: bool | None = None
    previous_outcome: str | None = None


@dataclass(frozen=True)
class ClaudeAgentSdk:
    """The Claude Agent SDK entry points one attempt drives."""

    query: Callable[..., Any]
    options_cls: Any


@dataclass
class AttemptResult:
    """Low-level output of one SDK session."""

    workspace: Path
    sdk_error: str = ""
    raw_text: str = ""


def _import_sdk() -> ClaudeAgentSdk:
    """Import the Claude Agent SDK and return its query primitives."""
    try:
        import claude_agent_sdk as sdk  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised via injection seams in tests
        raise RuntimeError("claude_agent_sdk not installed; run the quantization-agent installer") from exc
    if not (hasattr(sdk, "query") and hasattr(sdk, "ClaudeAgentOptions")):
        raise RuntimeError("claude_agent_sdk missing query / ClaudeAgentOptions")
    return ClaudeAgentSdk(query=sdk.query, options_cls=sdk.ClaudeAgentOptions)


def _iter_message_text(message: Any) -> Iterable[str]:
    """Yield the non-empty text fragments of a Claude Agent SDK message."""
    from hyperloom.common.claude_oneshot import message_text

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


def build_attempt_prompt(attempt: AttemptContext, *, skill_path: Path, fix_hypothesis_path: Path | None) -> str:
    """Assemble the prompt handed to the SDK for one attempt."""

    interactive = attempt.interactive
    acceptable_eval_gap = attempt.acceptable_eval_gap
    attempt_number = attempt.attempt_number
    previous_outcome = attempt.previous_outcome
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
- Workspace (WRITE only here; quantized_model_dir must resolve inside this path after Path.resolve(); see SKILL.md §1.1): {attempt.workspace}
- Quark project root (READ-ONLY; never edit files under this path): {attempt.quark_root}
- Attempt number: {attempt_number}
- Acceptable eval gap: {threshold_str}
- Interactive mode: {interactive_str}{retry_block}

## User prompt (verbatim)
{attempt.user_prompt}

Begin the workflow now. Do not ask the user clarifying questions unless the
SKILL.md retry/checkpoint protocol explicitly requires it.
"""


_ATTEMPT_SYSTEM_PROMPT = (
    "You are the Hyperloom quantization-agent. Drive the Quark workflow per "
    "SKILL.md. Never modify files under quark_root. Treat artifact presence "
    "in workspace as the source of truth; do not lie about success."
)


def _attempt_child_env() -> dict[str, str]:
    """The Quark py310 shim plus the attribution tags, for the tools an attempt's session spawns."""
    env = _quark_py310_compat_env()
    env.update(sdk_env_overlay(component="quantization", operation="quantize_attempt"))
    return env


def attempt_ended(artifacts: CollectedArtifacts) -> bool:
    """Whether the attempt has an end the classifier can judge: blocked, an eval report, or why eval was skipped.

    ``last_phase.txt`` is written when a phase starts, so reaching eval is not an end. The owed-end check and the retry
    loop both read this, so a session is continued exactly when its attempt would otherwise be judged unfinished.
    """
    return (
        artifacts.blocked_reason is not None
        or artifacts.eval_report_present
        or artifacts.eval_skipped_reason is not None
    )


def _attempt_finished(workspace: Path) -> bool:
    """:func:`attempt_ended` on the files ``workspace`` holds now."""
    return attempt_ended(collect_artifacts(workspace))


def _owed_attempt_end(workspace: Path) -> OwedArtifact:
    """What an attempt owes: a ``blocked.md``, or the workflow carried through evaluation."""
    return OwedArtifact(
        done=lambda: _attempt_finished(workspace),
        continuation_prompt=(
            "The quantization workflow is not finished: there is no blocked.md, eval_report.json or "
            "eval_skipped.txt. A tool call written out as text in a reply is not executed. Continue from the phase "
            "last_phase.txt names, calling tools directly, and write blocked.md if you cannot go on."
        ),
    )


async def _run_attempt_on_deepseek(
    attempt: AttemptContext,
    prompt: str,
    *,
    log: Callable[[str], None] | None,
) -> AttemptResult:
    """Run one attempt's session on the DeepSeek Harness."""
    workspace = Path(attempt.workspace)
    if log:
        log(
            f"quantization-agent DeepSeek runner: workspace={workspace} "
            f"quark_root={attempt.quark_root} attempt={attempt.attempt_number}"
        )
    request = DeepSeekTaskRequest(
        prompt=prompt,
        system_prompt=_ATTEMPT_SYSTEM_PROMPT,
        cwd=workspace,
        writable=True,
        child_env=_attempt_child_env(),
        log=log,
    )
    # An interactive attempt stops at its checkpoints for an answer nobody gives a headless session.
    owed = None if attempt.interactive else _owed_attempt_end(workspace)
    result = await run_deepseek_task(request, owed)
    return AttemptResult(workspace=workspace, sdk_error=result.error, raw_text=result.text)


async def run_one_attempt(
    attempt: AttemptContext,
    *,
    skill_path: Path | None = None,
    model: str | None = None,
    sdk: ClaudeAgentSdk | None = None,
    log: Callable[[str], None] | None = None,
) -> AttemptResult:
    """Run one SDK session driving SKILL.md."""
    workspace = Path(attempt.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    quark_root = Path(attempt.quark_root)
    attempt_number = attempt.attempt_number

    skill_path = skill_path or resolve_skill_path()
    if not skill_path.is_file():
        raise FileNotFoundError(f"SKILL.md not found at {skill_path}")

    fix_hypothesis_path: Path | None = None
    if attempt_number > 1:
        candidate = workspace / f"fix_hypothesis_attempt_{attempt_number}.md"
        fix_hypothesis_path = candidate if candidate.is_file() else None

    prompt = build_attempt_prompt(attempt, skill_path=skill_path, fix_hypothesis_path=fix_hypothesis_path)

    if sdk is None and llm_config.resolve_agent_backend() == llm_config.AGENT_BACKEND_DEEPSEEK:
        return await _run_attempt_on_deepseek(attempt, prompt, log=log)

    sdk = sdk or _import_sdk()

    kwargs: dict[str, Any] = {
        "max_turns": DEFAULT_MAX_TURNS,
        "system_prompt": _ATTEMPT_SYSTEM_PROMPT,
        "allowed_tools": DEFAULT_ALLOWED_TOOLS,
        "stderr": (lambda line: log(f"[claude-sdk] {line.rstrip()}")) if log else None,
        "env": _attempt_child_env(),
    }
    if model:
        kwargs["model"] = model
    kwargs["cwd"] = str(workspace)

    try:
        options = sdk.options_cls(**kwargs)
    except TypeError:
        # Older SDK builds may not support cwd; prompt + SKILL.md use absolute paths so retrying without cwd is safe.
        kwargs.pop("cwd", None)
        try:
            options = sdk.options_cls(**kwargs)
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
        async for message in sdk.query(prompt=prompt, options=options):
            for text in _iter_message_text(message):
                chunks.append(text)
                if log:
                    log(f"[claude-sdk] {text[:1000]}")
    except Exception as exc:  # noqa: BLE001 - isolation boundary: one SDK quantization attempt
        # Capture but don't raise: valid artifacts may exist before the abort.
        sdk_error = f"{type(exc).__name__}: {exc}"
        if log:
            log(f"[claude-sdk] WARNING: {sdk_error}\n{traceback.format_exc()}")

    return AttemptResult(
        workspace=workspace,
        sdk_error=sdk_error,
        raw_text="\n".join(chunks),
    )


# Injection seam used in tests and by driver/retry.py.
RunOneAttemptFn = Callable[..., Awaitable[AttemptResult]]


__all__ = [
    "AttemptContext",
    "AttemptResult",
    "ClaudeAgentSdk",
    "DEFAULT_ALLOWED_TOOLS",
    "DEFAULT_MAX_TURNS",
    "DEFAULT_MODEL",
    "RunOneAttemptFn",
    "build_attempt_prompt",
    "resolve_skill_path",
    "run_one_attempt",
]
