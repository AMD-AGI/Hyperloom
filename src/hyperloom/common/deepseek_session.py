# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""DeepSeekSession — one live DeepSeek Harness runtime and the thread held on it."""

from __future__ import annotations

import asyncio
import contextlib
import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hyperloom.common.llm_config import LLMConfigError, resolve_openai_client_config
from hyperloom.common.reasoning_effort import DEFAULT_REASONING_EFFORT, harness_reasoning_effort

#: The route the harness mounts for DeepSeek models in its own default plugin
#: composition. Naming any other provider needs a composition that mounts an
#: adapter owning it, which initialization refuses to guess at.
DEEPSEEK_PROVIDER_ROUTE = "deepseek-official"

#: Handshake budget. Initialization spawns a Node runtime and mounts the plugin
#: composition, so it is slower than any later turn on the same runtime.
_INITIALIZE_TIMEOUT_SEC = 120.0

#: Harness token fields, in their wire spelling, mapped onto the shared names.
#: ``totalTokens`` is deliberately absent: the harness reports it as a running
#: total for the whole turn, so summing it across steps would count the same
#: prefix once per step.
_USAGE_WIRE_NAMES: dict[str, str] = {
    "inputTokens": "input_tokens",
    "outputTokens": "output_tokens",
    "cacheReadTokens": "cache_read_input_tokens",
}


class DeepSeekSessionError(RuntimeError):
    """Base error for a DeepSeek Harness session."""


class DeepSeekSessionUnavailableError(DeepSeekSessionError):
    """Raised when the SDK or its configuration cannot serve any session."""


class DeepSeekSessionTimeoutError(DeepSeekSessionError):
    """Raised when one turn outlives its budget."""


@dataclass(frozen=True)
class DeepSeekSessionResult:
    """Normalized outcome of one DeepSeek Harness turn."""

    text: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    session_id: str = ""
    error: str = ""


def _load_harness_sdk() -> Any:
    """Import the harness SDK, naming the extra that provides it when absent."""
    try:
        import deepseek_harness
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise DeepSeekSessionUnavailableError(
            "the DeepSeek Harness SDK is not installed: install the 'deepseek' extra"
        ) from exc
    return deepseek_harness


def usage_from_events(events: list[dict[str, Any]]) -> dict[str, int]:
    """Sum the per-step token counts the harness reports for one turn."""
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    seen = False
    for event in events:
        if not isinstance(event, dict) or event.get("type") != "assistant/message":
            continue
        data = event.get("data")
        usage = data.get("usage") if isinstance(data, dict) else None
        if not isinstance(usage, dict):
            continue
        seen = True
        for wire, name in _USAGE_WIRE_NAMES.items():
            value = usage.get(wire)
            if isinstance(value, bool):
                continue
            with contextlib.suppress(TypeError, ValueError):
                parsed = int(value)
                if parsed >= 0:
                    totals[name] += parsed
    return totals if seen else {}


def turn_end_failure(events: list[dict[str, Any]]) -> str:
    """Return the message of the last failing ``turn/end``, if it carries one."""
    for event in reversed(events):
        if not isinstance(event, dict) or event.get("type") != "turn/end":
            continue
        data = event.get("data")
        reason = data.get("reason") if isinstance(data, dict) else None
        error = reason.get("error") if isinstance(reason, dict) else None
        if isinstance(error, dict):
            code = str(error.get("code") or "").strip()
            message = str(error.get("message") or "").strip()
            return f"{code}: {message}" if code and message else message or code
        break
    return ""


def normalize_run_result(result: Any, session_id: str) -> DeepSeekSessionResult:
    """Normalize one completed harness activity interval into a session result.

    ``finish_reason`` is the kind of the last ``turn/end`` in the interval: the
    harness reports a provider failure or a spent output cap there while the run
    itself completes, so an unexamined result reads as a turn that simply had
    nothing to say.
    """
    events = [event for event in (getattr(result, "events", ()) or ()) if isinstance(event, dict)]
    text = str(getattr(result, "final_response", "") or "").strip()
    resolved_id = str(getattr(result, "session_id", "") or session_id)

    error = ""
    finish_reason = str(getattr(result, "finish_reason", "") or "")
    if finish_reason == "max-tokens":
        error = "the turn stopped on the output token cap before it finished its reply"
    elif finish_reason and finish_reason != "completed":
        error = turn_end_failure(events) or f"the turn ended with turn/end {finish_reason!r}"

    return DeepSeekSessionResult(
        text=text,
        usage=usage_from_events(events),
        session_id=resolved_id,
        error=error,
    )


@dataclass
class DeepSeekSession:
    """One harness runtime plus the durable conversation currently held on it.

    The runtime is held for the session's lifetime rather than rebuilt per turn
    because the SDK cannot adopt a persisted conversation from a second process:
    a later resume against an id this process already ran is answered with
    ``session "<id>" already exists``. Keeping one live runtime is what lets
    :meth:`turn` continue the conversation instead of starting a fresh one that
    has never seen the work it is reasoning about.
    """

    cwd: Path
    model: str
    #: Carried on the first message of each conversation. The harness owns its
    #: system prompt through its plugin composition and its SDK exposes no
    #: per-session field for one, so there is nowhere else to put these.
    instructions: str = ""
    dsh_home: Path | None = None
    reasoning_effort: str = ""
    permission_mode: str = "read-only"
    env: Mapping[str, str] | None = None
    component: str = "orchestration"
    operation: str = "orchestrate_turn"
    request_timeout_sec: float = 300.0

    _harness: Any = field(default=None, init=False, repr=False)
    _session_id: str = field(default="", init=False, repr=False)
    _turn_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        """Resolve the workspace and the harness home before any turn runs."""
        self.cwd = Path(self.cwd).expanduser().resolve()
        home = Path(self.dsh_home).expanduser() if self.dsh_home else Path.home() / ".cache" / "hyperloom" / "dsh_home"
        self.dsh_home = home.resolve()

    def _child_environment(self) -> dict[str, str]:
        """Build the environment the harness runtime subprocess is spawned with.

        The harness's DeepSeek adapter authenticates from ``DEEPSEEK_*`` only,
        while this repository configures the gateway on the OpenAI-compatible
        line, so the gateway is written onto the adapter's names here, for the
        subprocess alone.
        """
        env = os.environ.copy()
        if self.env:
            env.update(self.env)
        try:
            gateway = resolve_openai_client_config(env=env)
        except LLMConfigError as exc:
            raise DeepSeekSessionUnavailableError(
                f"the DeepSeek Harness speaks the OpenAI-compatible protocol and needs "
                f"OPENAI_BASE_URL and OPENAI_API_KEY: {exc}"
            ) from exc
        if not gateway.base_url or not gateway.api_key:
            raise DeepSeekSessionUnavailableError(
                "the DeepSeek Harness speaks the OpenAI-compatible protocol and needs "
                "both OPENAI_BASE_URL and OPENAI_API_KEY"
            )
        env["DEEPSEEK_BASE_URL"] = gateway.base_url
        env["DEEPSEEK_API_KEY"] = gateway.api_key
        env["DSH_PERMISSION_MODE"] = self.permission_mode
        # These outrank -C and cwd, so a caller that set them to reach one
        # repository would answer every git command the session runs anywhere.
        env.pop("GIT_DIR", None)
        env.pop("GIT_WORK_TREE", None)
        return env

    async def start(self) -> None:
        """Spawn the harness runtime and mount its plugin composition."""
        if self._harness is not None:
            return
        sdk = _load_harness_sdk()
        assert self.dsh_home is not None  # set in __post_init__
        try:
            self.dsh_home.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise DeepSeekSessionUnavailableError(f"cannot prepare the harness home {self.dsh_home}: {exc}") from exc
        workspace = str(self.cwd)
        child_env = self._child_environment()
        try:
            self._harness = await asyncio.to_thread(
                sdk.DeepSeekHarness,
                dsh_home=str(self.dsh_home),
                cwd=workspace,
                # The harness reads its sandbox root from the subprocess working
                # directory, so a runtime rooted anywhere else would confine a
                # read-only session to the wrong tree.
                runtime_cwd=workspace,
                provider=DEEPSEEK_PROVIDER_ROUTE,
                model=self.model,
                # Initialization refuses an empty or unsupported effort, so an
                # unset one takes the shared default and an unsupported one is
                # rounded up to a level the harness has.
                reasoning_effort=harness_reasoning_effort(self.reasoning_effort or DEFAULT_REASONING_EFFORT),
                env=child_env,
                initialize_timeout_seconds=_INITIALIZE_TIMEOUT_SEC,
                request_timeout_seconds=self.request_timeout_sec,
            )
        except Exception as exc:
            self._harness = None
            raise DeepSeekSessionUnavailableError(f"the DeepSeek Harness runtime failed to start: {exc}") from exc

    def reset_conversation(self) -> None:
        """Drop the held conversation so the next turn opens a fresh one.

        Re-scoping instructions needs this: they ride the first message of a
        conversation, so a running one cannot be told about a new set.
        """
        self._session_id = ""

    async def turn(self, prompt: str, *, timeout_sec: float) -> DeepSeekSessionResult:
        """Run one turn on the held conversation and normalize its outcome."""
        if self._harness is None:
            # A timed-out turn closes the runtime, so a turn after it starts the next one.
            await self.start()
        async with self._turn_lock:
            opening = not self._session_id
            if opening:
                self._session_id = str(uuid.uuid4())
            body = f"{self.instructions}\n\n{prompt}" if opening and self.instructions else prompt
            session_id = self._session_id
            run = asyncio.to_thread(self._harness.run, body, session_id=session_id)
            try:
                sdk_result = await asyncio.wait_for(run, timeout=timeout_sec)
            except asyncio.TimeoutError as exc:
                # The SDK has no per-turn cancel: abandoning the await leaves the turn running, and spending, on
                # a runtime the next turn would share. Closing the runtime is the only way to stop it; the next
                # turn starts a new one and a new conversation.
                await self.aclose()
                raise DeepSeekSessionTimeoutError(f"the harness turn outlived its {timeout_sec}s budget") from exc
            except Exception as exc:
                raise DeepSeekSessionError(f"the harness turn failed: {exc}") from exc
            result = normalize_run_result(sdk_result, session_id)
            self._session_id = result.session_id or session_id
            return result

    async def aclose(self) -> None:
        """Release the runtime: SDK client, child process, harness home handle."""
        harness, self._harness = self._harness, None
        self._session_id = ""
        if harness is None:
            return
        closer = getattr(harness, "close", None) or getattr(harness, "shutdown", None)
        if closer is None:
            return
        with contextlib.suppress(Exception):
            await asyncio.to_thread(closer)


__all__ = [
    "DEEPSEEK_PROVIDER_ROUTE",
    "DeepSeekSession",
    "DeepSeekSessionError",
    "DeepSeekSessionResult",
    "DeepSeekSessionTimeoutError",
    "DeepSeekSessionUnavailableError",
    "normalize_run_result",
    "turn_end_failure",
    "usage_from_events",
]
