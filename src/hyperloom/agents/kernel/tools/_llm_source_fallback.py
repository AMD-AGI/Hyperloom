###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Last-resort semantic pick of a kernel's source file, from a fixed shortlist.

This is the final tier of source resolution, behind the curated dictionary, the
trace-derived launcher, and the name grep. It exists for models or frameworks
where all three come up empty -- not for any case seen so far.

Two properties keep it from becoming another source of nondeterminism, which
matters because an unresolved-launcher sentinel produced by an LLM is what broke
this pipeline in the first place:

* **Selection, never generation.** The model receives a shortlist gathered by a
  relaxed grep and may only return one of those exact strings. A path it invents
  is rejected outright.
* **Last in line.** It sees a candidate only after the curated dictionary, the
  trace-derived launcher and the grep have all come up empty, and only when the
  kernel is worth at least 5% of GPU time.

Every accepted answer is stamped ``source_resolution_method="llm_fallback"`` so
it can be audited apart from deterministic resolutions.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
from typing import Any, Callable
from urllib.parse import urlsplit

try:
    from hyperloom.common import kernel_source_contract as _KSC
except ImportError:  # pragma: no cover - standalone invocation
    _KSC = None  # type: ignore[assignment]

# An answer below this confidence is discarded: a coin-flip pick would send a
# backend at the wrong file and burn a whole optimization attempt.
_MIN_CONFIDENCE = 0.7

# Head of each shortlisted file shown to the model; enough to tell a kernel
# definition from a test or a dispatch shim.
_PREVIEW_LINES = 40
_PREVIEW_CHARS = 2000

# Shipping those heads sends repository source to the model provider. That is an
# operator's data-egress decision, not a default this tier may assume, so the
# preview is off unless explicitly authorised. Without it the tiers still see
# the candidate paths, which carry most of the signal.
_PREVIEW_ENV = "HYPERLOOM_LLM_SOURCE_PREVIEW"
_PROVIDER_ENV = "HYPERLOOM_LLM_SOURCE_PROVIDER"
_MODEL_ENV = "HYPERLOOM_LLM_SOURCE_MODEL"

_PROVIDER_CLAUDE = "claude_agent_sdk"
_PROVIDER_OPENAI = "openai_compatible"
_PROVIDER_ALIASES = {
    "anthropic": _PROVIDER_CLAUDE,
    "claude": _PROVIDER_CLAUDE,
    "claude-agent-sdk": _PROVIDER_CLAUDE,
    "claude_agent_sdk": _PROVIDER_CLAUDE,
    "openai": _PROVIDER_OPENAI,
    "openai-compatible": _PROVIDER_OPENAI,
    "openai_compatible": _PROVIDER_OPENAI,
}

# Claude Code must behave as a plain completion client in this tier. Explicitly
# denying every built-in tool prevents a source-resolution request from reading
# anything beyond the prompt assembled under the egress policy above.
_CLAUDE_DISALLOWED_TOOLS = (
    "Bash",
    "BashOutput",
    "KillShell",
    "Read",
    "Write",
    "Edit",
    "NotebookEdit",
    "Glob",
    "Grep",
    "Agent",
    "Task",
    "TaskOutput",
    "TaskStop",
    "WebFetch",
    "WebSearch",
    "TodoWrite",
    "AskUserQuestion",
    "EnterPlanMode",
    "ExitPlanMode",
    "Skill",
    "SlashCommand",
)

_DEFAULT_TIMEOUT_SEC = 60.0

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

_SYSTEM_PROMPT = (
    "You identify which source file implements a GPU kernel. "
    "You are given the kernel symbol and a shortlist of candidate files. "
    "Choose the file that DEFINES the kernel body. Reject files that merely "
    "call it, test it, or dispatch to it, and reject a CPU implementation when "
    "the kernel runs on GPU. "
    'Answer with JSON only: {"source_file": "<one of the candidates, verbatim>", '
    '"confidence": <0..1>, "reason": "<one sentence>"}. '
    'If none of the candidates defines the kernel, return "source_file": "".'
)


def _default_claude_model() -> str:
    """The project-wide default Claude model, or "" when it cannot be read.

    Deliberately without a hardcoded fallback id: a pinned default here would
    go stale the moment the project moves (it already outlived claude-opus-4-8),
    and a tier quietly running an older model than the rest of the pipeline is
    worse than one that declines to run. An empty result is reported as a
    resolution failure by the caller rather than guessing.
    """
    try:
        from hyperloom.orchestrator.roles.agent_role import (  # noqa: PLC0415
            DEFAULT_CLAUDE_MODEL,
        )
    except Exception:  # noqa: BLE001 - tools also run outside the package
        return ""
    return str(DEFAULT_CLAUDE_MODEL or "")


def _resolve_provider(provider: str = "") -> str:
    """Resolve the explicit provider or infer it from canonical credential shape.

    The role-specific override wins. Otherwise Anthropic-only deployments use
    the native Claude path, while any configured OpenAI side uses the OpenAI
    path. The latter preserves the project's OpenAI default for dual-configured
    single-shot roles.
    """
    raw = str(provider or os.environ.get(_PROVIDER_ENV) or "").strip().lower()
    supported = "claude_agent_sdk, openai_compatible"
    if raw:
        resolved = _PROVIDER_ALIASES.get(raw)
        if resolved:
            return resolved
        raise RuntimeError(f"unsupported {_PROVIDER_ENV}={raw!r}; choose one of: {supported}")

    from hyperloom.common import llm_config  # noqa: PLC0415 - keep standalone import-light

    if llm_config.is_anthropic_only():
        return _PROVIDER_CLAUDE
    if llm_config.has_openai_side():
        return _PROVIDER_OPENAI
    raise RuntimeError(
        f"{_PROVIDER_ENV} is not set and no provider credentials are configured; choose one of: {supported}"
    )


def _resolve_model(model: str = "", provider: str = "") -> str:
    """Resolve a model without borrowing another provider's model setting."""
    explicit = str(model or os.environ.get(_MODEL_ENV) or "").strip()
    if explicit:
        return explicit
    if provider == _PROVIDER_OPENAI:
        return str(os.environ.get("OPENAI_MODEL") or os.environ.get("CODEX_MODEL") or "").strip()
    return str(os.environ.get("CLAUDE_MODEL") or _default_claude_model()).strip()


def _endpoint_host(provider: str) -> str:
    """Return a credential-free endpoint identifier for artifact auditing."""
    if provider == _PROVIDER_OPENAI:
        raw = str(os.environ.get("OPENAI_BASE_URL") or "").strip()
        default = "api.openai.com"
    else:
        raw = str(os.environ.get("ANTHROPIC_BASE_URL") or os.environ.get("DEEPSEEK_BASE_URL") or "").strip()
        default = "provider-default"
    if not raw:
        return default
    return urlsplit(raw).hostname or "configured"


def llm_source_provider_configured() -> bool:
    """Whether source resolution has an explicit or inferred provider.

    Lets a caller skip the work it would only do in order to build a request --
    the shortlist grep walks every framework root -- when neither the role
    override nor a canonical credential side can select a provider.
    """
    try:
        _resolve_provider()
    except RuntimeError:
        return False
    return True


def llm_source_audit(*, provider: str = "", model: str = "") -> dict[str, Any]:
    """Return non-secret provider metadata suitable for an audit artifact."""
    try:
        selected_provider = _resolve_provider(provider)
    except RuntimeError:
        return {
            "provider": "unconfigured",
            "model": str(model or os.environ.get(_MODEL_ENV) or "").strip(),
            "endpoint_host": "",
            "source_preview_authorised": source_preview_authorised(),
        }
    return {
        "provider": selected_provider,
        "model": _resolve_model(model, selected_provider),
        "endpoint_host": _endpoint_host(selected_provider),
        "source_preview_authorised": source_preview_authorised(),
    }


def source_preview_authorised() -> bool:
    """Whether the operator opted into sending file heads to the model."""
    return str(os.environ.get(_PREVIEW_ENV, "")).strip().lower() in ("1", "true", "yes", "on")


def _preview(path: str) -> str:
    """First lines of ``path``, for telling an implementation from a shim."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            head = "".join(next(fh, "") for _ in range(_PREVIEW_LINES))
    except OSError:
        return "<unreadable>"
    return head[:_PREVIEW_CHARS]


def _canonical_candidates(
    candidates: list[str],
    framework_roots: tuple[str, ...],
    *,
    require_roots: bool = True,
) -> dict[str, str]:
    """Map candidates to canonical targets after filesystem validation."""
    canonical: dict[str, str] = {}
    for path in candidates:
        if require_roots:
            target = _KSC.canonical_source_path(path, framework_roots) if _KSC else ""
        else:
            bare = _KSC.strip_line_suffix(path) if _KSC else path
            target = os.path.realpath(bare) if bare and os.path.isfile(bare) else ""
        if target:
            canonical[path] = target
    return canonical


def _build_prompt_with_targets(
    kernel_name: str,
    candidates: list[str],
    context_block: str = "",
    with_preview: bool | None = None,
    *,
    framework_roots: tuple[str, ...] = (),
) -> tuple[str, dict[str, str]]:
    """Render the prompt and return its prevalidated canonical targets."""
    if with_preview is None:
        with_preview = source_preview_authorised()
    canonical_paths = _canonical_candidates(
        candidates,
        framework_roots,
        require_roots=bool(framework_roots),
    )
    preview_paths = canonical_paths if framework_roots else {}
    parts = []
    if context_block:
        parts.append(context_block + "\n")
    parts += [f"Kernel symbol: {kernel_name}", "", "Candidates:"]
    for index, path in enumerate(candidates, 1):
        canonical = preview_paths.get(path, "")
        if with_preview and canonical:
            parts.append(f"\n[{index}] {path}\n```\n{_preview(canonical)}\n```")
        else:
            parts.append(f"\n[{index}] {path}")
    return "\n".join(parts), canonical_paths


def _build_prompt(
    kernel_name: str,
    candidates: list[str],
    context_block: str = "",
    with_preview: bool | None = None,
    *,
    framework_roots: tuple[str, ...] = (),
) -> str:
    """Render the shortlist, with validated previews when authorised."""
    prompt, _ = _build_prompt_with_targets(
        kernel_name,
        candidates,
        context_block,
        with_preview,
        framework_roots=framework_roots,
    )
    return prompt


def _parse_answer(text: str) -> tuple[bool, str, float, str]:
    """Extract ``(parsed, source_file, confidence, reason)`` from a model reply.

    ``parsed`` separates "the reply was unreadable" from "the model answered that
    no candidate fits". Collapsing the two would report a malformed reply as a
    considered verdict and send triage the wrong way.
    """
    if not isinstance(text, str):
        return False, "", 0.0, "reply is not text"
    match = _JSON_BLOCK_RE.search(text or "")
    if not match:
        return False, "", 0.0, "no JSON object in reply"
    try:
        payload = json.loads(match.group(0))
    except (TypeError, ValueError):
        return False, "", 0.0, "unparseable JSON"
    if not isinstance(payload, dict):
        return False, "", 0.0, "JSON payload is not an object"
    try:
        confidence = float(payload.get("confidence") or 0.0)
    except (TypeError, ValueError):
        return False, "", 0.0, "confidence is not numeric"
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        return False, "", 0.0, "confidence must be finite and in [0, 1]"
    return (
        True,
        str(payload.get("source_file") or "").strip(),
        confidence,
        str(payload.get("reason") or "").strip(),
    )


def _validate(
    picked: str,
    candidates: list[str],
    canonical_paths: dict[str, str],
    framework_roots: tuple[str, ...],
) -> tuple[str, str]:
    """Return the prevalidated canonical pick and any rejection reason."""
    if not picked:
        return "", "model reported no candidate defines the kernel"
    if picked not in candidates:
        # The whole point of a shortlist is that the answer comes from it.
        return "", f"path is not one of the candidates: {picked!r}"
    canonical = canonical_paths.get(picked, "")
    if canonical:
        return canonical, ""
    if framework_roots:
        return "", f"path does not exist or is outside every framework root: {picked!r}"
    return "", f"path does not exist: {picked!r}"


def _safe_exception_label(exc: BaseException) -> str:
    """Return a stable exception type and optional non-secret error code."""
    label = type(exc).__name__
    for attribute in ("status_code", "code", "errno"):
        try:
            value = getattr(exc, attribute, None)
        except Exception:  # noqa: BLE001 - hostile exception properties stay private
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return f"{label} ({attribute}={value})"
        if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", value):
            return f"{label} ({attribute}={value})"
    return label


def _complete_openai(prompt: str, model: str, timeout_sec: float) -> str:
    """Run one completion against the configured OpenAI-compatible endpoint."""
    from hyperloom.common import llm_config  # noqa: PLC0415 - optional dependency

    return llm_config.chat_completion(
        llm_config.get_openai_client(),
        component="kernel_agent",
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        timeout=timeout_sec,
    ).text


def _complete_claude_sdk(prompt: str, model: str, timeout_sec: float) -> str:
    """Run one tool-free completion through the native Claude Agent SDK."""
    try:
        import claude_agent_sdk as sdk  # type: ignore[import-not-found]  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("claude_agent_sdk is not installed") from exc
    if not (hasattr(sdk, "query") and hasattr(sdk, "ClaudeAgentOptions")):
        raise RuntimeError("claude_agent_sdk missing query / ClaudeAgentOptions")

    from hyperloom.common.claude_oneshot import message_text  # noqa: PLC0415
    from hyperloom.common.llm_config import claude_sdk_env_options  # noqa: PLC0415

    kwargs: dict[str, Any] = dict(claude_sdk_env_options(model=model, component="kernel_agent"))
    kwargs.update(
        {
            "model": model,
            "system_prompt": _SYSTEM_PROMPT,
            "tools": [],
            "setting_sources": [],
            "skills": [],
            "strict_mcp_config": True,
            "mcp_servers": {},
            "plugins": [],
            "max_turns": 1,
            "allowed_tools": [],
            "disallowed_tools": list(_CLAUDE_DISALLOWED_TOOLS),
        }
    )
    options = sdk.ClaudeAgentOptions(**kwargs)

    async def _drive() -> str:
        """Collect the final result, falling back to streamed text blocks."""
        final = ""
        chunks: list[str] = []
        async for message in sdk.query(prompt=prompt, options=options):
            result = getattr(message, "result", None)
            if isinstance(result, str) and result.strip():
                final = result
                continue
            chunks.extend(message_text(message))
        return final.strip() or "".join(chunks).strip()

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(asyncio.wait_for(_drive(), timeout=max(0.1, float(timeout_sec))))
    raise RuntimeError("claude_agent_sdk completion cannot run inside an active event loop")


def _complete(prompt: str, model: str, timeout_sec: float) -> str:
    """Route one completion through the selected native provider."""
    provider = _resolve_provider()
    if provider == _PROVIDER_CLAUDE:
        return _complete_claude_sdk(prompt, model, timeout_sec)
    return _complete_openai(prompt, model, timeout_sec)


def select_source_via_llm(
    kernel_name: str,
    candidates: list[str],
    *,
    framework_roots: tuple[str, ...] = (),
    model: str = "",
    timeout_sec: float = _DEFAULT_TIMEOUT_SEC,
    context_block: str = "",
    log: Callable[[str], None] | None = None,
    complete: Callable[[str, str, float], str] | None = None,
    errors: list[str] | None = None,
) -> tuple[str, float, str]:
    """Pick the file that defines ``kernel_name`` from ``candidates``.

    Args:
        kernel_name: The kernel symbol being resolved.
        candidates: Shortlist of on-disk paths from the relaxed grep. An empty
            list short-circuits: with nothing to choose from there is nothing to
            ask, and inventing a path is not allowed.
        framework_roots: Accepted path roots; a pick outside them is rejected.
        model: Chat model; defaults to ``$HYPERLOOM_LLM_SOURCE_MODEL``, then
            the selected provider's model setting.
        timeout_sec: Per-call ceiling. There is no retry -- a failure here is
            advisory and the candidate simply stays unresolved.
        log: Optional ``callable(str)`` for diagnostics.
        complete: Injection point for the completion call (tests).
        errors: Optional output list for configuration, transport and parsing
            failures. A valid model refusal does not append to it.

    Returns:
        ``(source_file, confidence, reason)``; ``source_file`` is ``""`` on any
        failure, including a low-confidence answer.
    """

    def _say(message: str) -> None:
        if callable(log):
            log(f"llm_source_fallback: {message}")

    if not kernel_name or not candidates:
        return "", 0.0, "no candidates to choose from"

    shortlist = [str(c) for c in candidates if str(c).strip()]
    caller = complete or _complete
    try:
        provider = _resolve_provider() if complete is None else ""
        chosen_model = _resolve_model(model, provider)
    except RuntimeError as exc:
        detail = _safe_exception_label(exc)
        _say(f"configuration failed: {detail}")
        reason = f"llm configuration failed: {detail}"
        if errors is not None:
            errors.append(reason)
        return "", 0.0, reason
    if not chosen_model:
        _say(f"no model configured; set ${_MODEL_ENV}")
        reason = "no model configured"
        if errors is not None:
            errors.append(reason)
        return "", 0.0, reason
    try:
        prompt, canonical_paths = _build_prompt_with_targets(
            kernel_name,
            shortlist,
            context_block,
            framework_roots=framework_roots,
        )
        reply = caller(
            prompt,
            chosen_model,
            timeout_sec,
        )
    except Exception as exc:  # noqa: BLE001 - advisory tier, never fatal
        detail = _safe_exception_label(exc)
        _say(f"call failed: {detail}")
        reason = f"llm call failed: {detail}"
        if errors is not None:
            errors.append(reason)
        return "", 0.0, reason

    parsed, picked, confidence, reason = _parse_answer(reply)
    if not parsed:
        _say(f"rejected: {reason}")
        if errors is not None:
            errors.append(reason)
        return "", 0.0, reason
    canonical, why = _validate(picked, shortlist, canonical_paths, framework_roots)
    if not canonical:
        _say(f"rejected: {why}")
        return "", confidence, why
    if confidence < _MIN_CONFIDENCE:
        _say(f"rejected: confidence {confidence:.2f} < {_MIN_CONFIDENCE}")
        return "", confidence, f"confidence {confidence:.2f} below {_MIN_CONFIDENCE}"

    _say(f"accepted {canonical} (confidence={confidence:.2f})")
    return canonical, confidence, reason


__all__ = [
    "llm_source_audit",
    "llm_source_provider_configured",
    "select_source_via_llm",
]
