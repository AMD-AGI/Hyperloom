# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CodexBackend — a reactor role driven by the Codex Agent SDK."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from hyperloom.common.codex_session import (
    CodexSession,
    CodexSessionError,
    CodexSessionUnavailableError,
)
from hyperloom.common.env_safety import redact_secret_values
from hyperloom.inference_optimizer.protocol.intent import (
    IntentType,
    IntentValidationError,
    NoIntentEmitted,
    validate_envelope,
)
from ..prompts.transport import TRANSPORT_STRUCTURED_OUTPUT
from ..trace.llm_trace import new_call_id
from .agent_role import DEFAULT_CODEX_MODEL
from .base import (
    BackendError,
    BackendTurnResult,
    LLMCallFailed,
    RetryPolicy,
    derive_turn_budget_sec,
    parse_call_timeout_env,
    retry_with_backoff,
    safe_int,
)
from .mcp_emit_intent import (
    build_intent_envelope_schema,
    constraints_sentence,
    payload_contract,
)

# Prepended to a turn that runs without the enforced schema.
_PER_TURN_SCHEMA_OVERRIDE = (
    "THIS TURN ONLY: ignore the OUTPUT FORMAT block in your instructions. "
    "Do not emit an intent envelope. Reply exactly as the message below asks."
)


def build_output_instructions(allowed_intents: Iterable[IntentType]) -> str:
    """Render the transport contract for one role's intent set."""
    from hyperloom.inference_optimizer.protocol.intent import IntentType as _IT

    contract = payload_contract(allowed_intents)
    constraints = constraints_sentence(allowed_intents)
    constraints_line = f"\n-{constraints}" if constraints else ""
    always_emit_line = (
        "- ALWAYS emit at least one intent."
        if _IT.SEND_MESSAGE in set(allowed_intents)
        else "- ALWAYS emit exactly one intent; the schema requires it."
    )
    return f"""
==== OUTPUT FORMAT (REQUIRED) ====
Your final message MUST be exactly one JSON object matching the enforced
output schema — no prose, no code fences, nothing around it:

{{"intents": [{{"intent_type": "...", "payload": "..."}}]}}

- `payload` is a STRING holding a serialized JSON object. The schema cannot
  express a free-form object, so serialize the payload and escape it.
- Required keys per intent_type: {contract}.{constraints_line}
- Emit several intents by adding entries to `intents`.
- Put only NEW information in payload bodies; do not restate context already
  in SharedState, your inbox, or analysis.md. Keep length proportional to
  substance.
{always_emit_line}
==== END OUTPUT FORMAT ====
""".strip()


def decode_intent_envelope(text: str) -> dict[str, Any]:
    """Decode a schema-enforced reply into a :func:`validate_envelope` input."""
    try:
        envelope = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise NoIntentEmitted(
            f"codex reply did not honour the enforced output schema (chars={len(text or '')})"
        ) from exc
    if not isinstance(envelope, dict) or not isinstance(envelope.get("intents"), list):
        raise NoIntentEmitted("codex reply is valid JSON but carries no 'intents' list")
    # An empty list satisfies validate_envelope, so without this the tick is recorded as a success that did nothing
    # and the backend's error streak is reset.
    if not envelope["intents"]:
        raise NoIntentEmitted("codex reply carried an empty 'intents' list")
    decoded: list[Any] = []
    for index, item in enumerate(envelope["intents"]):
        if not isinstance(item, dict):
            raise NoIntentEmitted(f"codex intents[{index}] is not an object")
        payload = item.get("payload")
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except ValueError as exc:
                raise NoIntentEmitted(f"codex intents[{index}] payload is not decodable JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise NoIntentEmitted(f"codex intents[{index}] payload did not decode to an object")
        decoded.append({"intent_type": item.get("intent_type"), "payload": payload})
    return {"intents": decoded}


@dataclass
class CodexBackend:
    """Production Codex reactor backend. Implements :class:`Backend`."""

    allowed_intents: frozenset[IntentType]
    model: str = DEFAULT_CODEX_MODEL
    cwd: Path = field(default_factory=Path.cwd)
    writable_roots: tuple[Path, ...] = ()
    sandbox_mode: str = ""
    codex_bin: str = ""
    # An agent turn carries a tool loop, so it needs the orchestration budget,
    # not a completion's 120s.
    call_timeout_s: float = field(
        default_factory=lambda: parse_call_timeout_env(
            "INFERENCE_OPTIMIZER_CODEX_CALL_TIMEOUT_SEC",
            default=300.0,
        )
    )
    env: dict[str, str] | None = None
    # Bounded transient-failure retry/backoff, on the same INFERENCE_OPTIMIZER_LLM_RETRY_* knobs the Claude path
    # reads.
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy.from_env)

    name: str = "codex"
    calls: list[dict[str, Any]] = field(default_factory=list)

    # Which prompt modules describe a surface this backend actually has. Read
    # by the prompt builder, which cannot infer it from the role: the
    # orchestration role is Claude on paper and Codex in an OpenAI-only run.
    transport = TRANSPORT_STRUCTURED_OUTPUT

    _session: CodexSession | None = field(default=None, init=False, repr=False)
    # Developer instructions the open thread was started with.
    _thread_instructions: str = field(default="", init=False, repr=False)

    @property
    def turn_budget_sec(self) -> float:
        """float: Wall-clock ceiling for one ``run()`` call, retries included."""
        return derive_turn_budget_sec(lambda _n: self.call_timeout_s, self.retry_policy)

    def __post_init__(self) -> None:
        """Normalize and secure the session-private runtime root."""
        if not self.allowed_intents:
            raise BackendError("CodexBackend requires the emitting role's allowed_intents")
        self.cwd = Path(self.cwd).expanduser().resolve()
        try:
            self.cwd.mkdir(parents=True, mode=0o700, exist_ok=True)
            self.cwd.chmod(0o700)
        except OSError as exc:
            raise BackendError(f"cannot prepare Codex backend cwd {self.cwd}: {exc}") from exc
        roots = tuple(Path(root).expanduser().resolve() for root in self.writable_roots)
        if self.cwd not in roots:
            roots = (self.cwd, *roots)
        self.writable_roots = roots

    # ------------------------------------------------------------------
    async def run(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        tools: list[str] | None = None,
        max_turns: int = 1,
        allow_no_intent: bool = False,
    ) -> BackendTurnResult:
        """Run one Codex Agent SDK turn and parse its enforced intent envelope."""
        output_schema = None if allow_no_intent else build_intent_envelope_schema(self.allowed_intents)
        # Dropping the schema is not enough on its own: the thread's developer instructions carry the OUTPUT FORMAT
        # block for the life of the thread and cannot be scoped out for one turn, so a checkpoint turn asking for a
        # different JSON shape would be answered with an intent envelope.
        turn_prompt = f"{_PER_TURN_SCHEMA_OVERRIDE}\n\n{prompt}" if allow_no_intent else prompt

        async def _one_attempt() -> Any:
            """Acquire the session and run one turn under the retry policy."""
            session = await self._session_for(system_prompt)
            return await session.turn(
                turn_prompt,
                timeout_sec=self.call_timeout_s,
                output_schema=output_schema,
            )

        def _note_retry(attempt: int, exc: BaseException, delay: float) -> None:
            """Record a transient-failure retry warning into the call log."""
            self.calls.append(
                {"warn": f"codex SDK transient failure (attempt {attempt}): {exc!r}; retrying in {delay:.2f}s"}
            )

        try:
            sdk_result = await retry_with_backoff(
                _one_attempt,
                policy=self.retry_policy,
                retry_on=(CodexSessionError,),
                on_retry=_note_retry,
            )
        except CodexSessionError as exc:
            raise LLMCallFailed(f"Codex Agent SDK turn failed: {redact_secret_values(str(exc))}") from exc
        if sdk_result.error:
            raise LLMCallFailed("Codex Agent SDK turn failed: " + redact_secret_values(sdk_result.error))

        usage = dict(sdk_result.usage or {})
        input_tokens = safe_int(usage.get("input_tokens"))
        output_tokens = safe_int(usage.get("output_tokens"))
        cache_read_tokens = safe_int(usage.get("cache_read_input_tokens"))
        reasoning_tokens = safe_int(usage.get("reasoning_output_tokens"))
        self.calls.append(
            {
                "model": self.model,
                "prompt_chars": len(prompt),
                "reply_chars": len(sdk_result.text),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_read_input_tokens": cache_read_tokens,
                "reasoning_output_tokens": reasoning_tokens,
                "thread_id": sdk_result.thread_id,
            }
        )
        metadata: dict[str, Any] = {
            "model": self.model,
            "thread_id": sdk_result.thread_id,
            # Pairs this turn's token row with its conversation row.
            "call_id": new_call_id(),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": cache_read_tokens,
            "reasoning_output_tokens": reasoning_tokens,
            # Codex reports per-turn counts, so the input side already is this request's context size — the figure the
            # checkpoint policy compares against the model's window.
            "context_tokens_peak": input_tokens,
            # Stated by Codex per turn, and better than any table this side keeps: the compaction trigger is a
            # fraction of it.
            "model_context_window": safe_int(usage.get("model_context_window")),
            # Full conversation text for conversations.jsonl, handed up for the caller to persist.
            "prompt": prompt,
            "response": sdk_result.text,
        }

        if allow_no_intent:
            return BackendTurnResult(intents=[], raw_text=sdk_result.text, metadata=metadata)
        envelope = decode_intent_envelope(sdk_result.text)
        try:
            intents = validate_envelope(envelope)
        except IntentValidationError as exc:
            raise NoIntentEmitted(f"codex envelope invalid: {exc}") from exc
        return BackendTurnResult(intents=intents, raw_text=sdk_result.text, metadata=metadata)

    def _instructions_for(self, system_prompt: str | None) -> str:
        """Thread-level instructions implied by one system prompt."""
        return "\n\n".join(
            part for part in ((system_prompt or "").strip(), build_output_instructions(self.allowed_intents)) if part
        )

    async def aclose(self) -> None:
        """Release the held session: SDK client, child process, ``CODEX_HOME``."""
        session, self._session = self._session, None
        if session is not None:
            await session.aclose()

    async def _session_for(self, system_prompt: str | None) -> CodexSession:
        """Return the held session, opening it (or re-scoping it) as needed."""
        instructions = self._instructions_for(system_prompt)
        if self._session is None:
            self._session = CodexSession(
                cwd=self.cwd,
                model=self.model,
                developer_instructions=instructions,
                writable_roots=self.writable_roots,
                sandbox_mode=self.sandbox_mode,
                codex_bin=self.codex_bin,
                env=self.env,
                component="orchestration",
                operation="orchestrate_turn",
            )
            try:
                await self._session.start()
            except CodexSessionUnavailableError as exc:
                # A config or sandbox fault, identical on every attempt.
                self._session = None
                raise BackendError(redact_secret_values(str(exc))) from exc
            except CodexSessionError:
                # Left as it is so the retry wrapper around the attempt can see it.
                self._session = None
                raise
        elif instructions != self._thread_instructions:
            self._session.developer_instructions = instructions
            self._session.reset_thread()
        self._thread_instructions = instructions
        return self._session


__all__ = ["CodexBackend", "build_output_instructions", "decode_intent_envelope"]
