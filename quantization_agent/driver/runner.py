"""One-attempt SDK driver for the quantization-agent.

Mirrors :mod:`kernel-agent.tools.tracelens_skill_runner`:

* keyword-only public API,
* lazy ``_import_sdk`` with explicit ``sdk_query_factory`` / ``sdk_options_cls``
  injection seams so tests don't need the SDK installed,
* ``cwd = quark_root`` with graceful fallback for older SDK builds,
* SDK errors are stored on the result rather than raised — the classifier in
  :mod:`driver.assessment` decides whether to fail the attempt by reading the
  workspace state alongside ``sdk_error``,
* a single ``log`` callable so the caller can route output to its logger.

The agent leans entirely on ``SKILL.md`` as the runtime contract: this module
just plumbs ``(workspace, quark_root, attempt_n, threshold, interactive,
user_prompt)`` into a templated prompt and hands it to the SDK.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable


DEFAULT_MODEL = "claude-opus-4-7"
DEFAULT_ALLOWED_TOOLS = ["Read", "Write", "Edit", "Bash"]
DEFAULT_MAX_TURNS = 240  # ~ Quark workflow has 4 STOPs + validator + eval; generous

SKILL_RELATIVE_PATH = "SKILL.md"


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
    try:
        import claude_agent_sdk as sdk  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised via injection seams in tests
        raise RuntimeError(
            "claude_agent_sdk not installed; run the quantization-agent installer"
        ) from exc
    if not (hasattr(sdk, "query") and hasattr(sdk, "ClaudeAgentOptions")):
        raise RuntimeError("claude_agent_sdk missing query / ClaudeAgentOptions")
    return sdk.query, sdk.ClaudeAgentOptions


def _iter_message_text(message: Any) -> Iterable[str]:
    for block in list(getattr(message, "content", None) or []):
        text = getattr(block, "text", None)
        if isinstance(text, str) and text:
            yield text
        elif isinstance(block, dict) and isinstance(block.get("text"), str):
            yield block["text"]
    result_text = getattr(message, "result", None)
    if isinstance(result_text, str) and result_text:
        yield result_text


def resolve_skill_path(package_root: Path | None = None) -> Path:
    """Return the on-disk path of ``quantization_agent/SKILL.md``.

    The SKILL is the runtime contract loaded into every attempt; resolution
    is centralized here so callers (including the CLI smoke test) don't
    hardcode the layout.
    """

    # __file__ is .../quantization_agent/driver/runner.py — SKILL.md lives one
    # level up at the package root (a runtime contract, not a driver detail).
    root = package_root if package_root is not None else Path(__file__).resolve().parent.parent
    return root / SKILL_RELATIVE_PATH


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

    The runtime contract is in ``SKILL.md``; this prompt just pins the run
    context (workspace / quark_root / attempt / threshold / interactivity)
    and embeds the verbatim user prompt. Retry attempts also reference the
    prior outcome ID + the fix-hypothesis file SKILL.md wrote at the end of
    the previous attempt, so the LLM can target the diagnosed cause rather
    than re-running the same plan blindly.
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
            f"\n- Fix hypothesis from prior attempt: {fix_hypothesis_path}"
            if fix_hypothesis_path is not None
            else ""
        )
        retry_block = (
            f"\n\n## Retry context\nThis is attempt #{attempt_number}. The previous "
            f"attempt ended with outcome `{previous_outcome}`. Diagnose and apply the "
            f"fix you wrote in `fix_hypothesis_attempt_{attempt_number}.md` before "
            f"re-running quark-ptq.{hint}"
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
        "allowed_tools": allowed_tools or DEFAULT_ALLOWED_TOOLS,
        "stderr": (lambda line: log(f"[claude-sdk] {line.rstrip()}")) if log else None,
    }
    if model:
        kwargs["model"] = model
    kwargs["cwd"] = str(quark_root)

    try:
        options = sdk_options_cls(**kwargs)
    except TypeError:
        # Older SDK builds may not support cwd; prompt + SKILL.md use absolute
        # paths so retrying without cwd is safe.
        kwargs.pop("cwd", None)
        options = sdk_options_cls(**kwargs)

    chunks: list[str] = []
    sdk_error = ""

    if log:
        log(
            f"quantization-agent SDK runner: workspace={workspace} "
            f"quark_root={quark_root} attempt={attempt_number}"
        )

    try:
        async for message in sdk_query_factory(prompt=prompt, options=options):
            for text in _iter_message_text(message):
                chunks.append(text)
                if log:
                    log(f"[claude-sdk] {text[:1000]}")
    except Exception as exc:  # noqa: BLE001
        # Per kernel-agent precedent: capture the error but don't raise —
        # SKILL.md may have produced valid artifacts before the SDK aborted
        # (e.g. "max turns reached" after validate phase finished).
        sdk_error = f"{type(exc).__name__}: {exc}"
        if log:
            log(f"[claude-sdk] WARNING: {sdk_error}")

    return AttemptResult(
        workspace=workspace,
        sdk_error=sdk_error,
        raw_text="\n".join(chunks),
        chunks=chunks,
    )


# Type alias for the injection seam used in tests + by driver/retry.py.
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
