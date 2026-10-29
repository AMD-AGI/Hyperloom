# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""DeepSeekBackend — a reactor role driven by the DeepSeek Harness SDK."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from common.deepseek_session import (
    DEEPSEEK_RUNTIME_MAPPING,
    DeepSeekSession,
    DeepSeekSessionError,
    DeepSeekSessionUnavailableError,
)
from hyperloom.common.env_safety import redact_secret_values
from common.llm_config import DEFAULT_DEEPSEEK_HARNESS_MODEL
from hyperloom.inference_optimizer.protocol.intent import (
    IntentType,
    IntentValidationError,
    NoIntentEmitted,
    validate_envelope,
)

from ..prompts.transport import TRANSPORT_STRUCTURED_OUTPUT
from hyperloom.inference_optimizer.trace.llm_trace import new_call_id
from hyperloom.inference_optimizer.trace.trajectory_trace import current_context
from common.coerce import to_int
from .base import (
    BackendError,
    BackendTurnResult,
    LLMCallFailed,
    RetryPolicy,
    parse_call_timeout_env,
    retry_with_backoff,
)

from .mcp_emit_intent import constraints_sentence, payload_contract

# Restated every turn. The contract rides the first message of a conversation
# and the harness has no schema to hold the reply to it, so a long conversation
# would otherwise drift back to prose.
_ENVELOPE_REMINDER = (
    "Reply with exactly one JSON object matching the OUTPUT FORMAT block in your "
    "instructions. No prose and no code fence around it. Its shape is "
    '{"intents": [{"intent_type": "<type>", "payload": {<JSON object>}}]}; '
    "each intent carries exactly the keys `intent_type` and `payload`, and "
    "`payload` is an inline JSON object, not a string."
)


def build_output_instructions(allowed_intents: Iterable[IntentType]) -> str:
    """Render the reply contract for one role's intent set.

    The Codex contract serializes ``payload`` into a string because its
    enforced schema cannot express a free-form object. Nothing enforces a
    schema here, so the payload is written inline rather than hand-escaped.
    """
    contract = payload_contract(allowed_intents)
    constraints = constraints_sentence(allowed_intents)
    constraints_line = f"\n-{constraints}" if constraints else ""
    always_emit_line = (
        "- ALWAYS emit at least one intent."
        if IntentType.SEND_MESSAGE in set(allowed_intents)
        else "- ALWAYS emit exactly one intent."
    )
    return f"""
==== OUTPUT FORMAT (REQUIRED) ====
Your final message MUST be exactly one JSON object -- no prose, no code fences,
nothing around it:

{{"intents": [{{"intent_type": "...", "payload": {{...}}}}]}}

- `payload` is a JSON object written inline. Do not serialize it into a string.
- Required keys per intent_type: {contract}.{constraints_line}
- Emit several intents by adding entries to `intents`.
- Put only NEW information in payload bodies; do not restate context already
  in SharedState, your inbox, or analysis.md. Keep length proportional to
  substance.
{always_emit_line}
==== END OUTPUT FORMAT ====
""".strip()


_REJECTED_REPLY = "Your previous reply was rejected and none of its intents were applied: {error}. Reply again."


def _strip_code_fence(text: str) -> str:
    """Return the body of a fenced block, or ``text`` when it is not fenced."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    without_open = stripped.split("\n", 1)
    if len(without_open) != 2:
        return stripped
    body = without_open[1]
    closing = body.rfind("```")
    return (body[:closing] if closing != -1 else body).strip()


def _balanced_object_at(text: str, start: int) -> str:
    """Return the balanced ``{...}`` run that opens at ``start``, else ``""``."""
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return ""


def _json_objects(text: str) -> list[Any]:
    """Every balanced ``{...}`` run in ``text`` that decodes as JSON, in order.

    The harness enforces no output schema, so a reply arrives as reasoning prose
    followed by the envelope, and the prose itself quotes the contract
    (``delegate:{action_name}``) and sample payloads. Taking the first balanced
    run handed those fragments to the decoder and dropped the envelope behind
    them, so every decodable run is collected and the caller picks the envelope.
    """
    found: list[Any] = []
    start = text.find("{")
    while start != -1:
        run = _balanced_object_at(text, start)
        if run:
            try:
                found.append(json.loads(run))
            except ValueError:
                pass
            else:
                start = text.find("{", start + len(run))
                continue
        start = text.find("{", start + 1)
    return found


def decode_intent_envelope(text: str) -> dict[str, Any]:
    """Decode a harness reply into a :func:`validate_envelope` input."""
    candidate = _strip_code_fence(text or "")
    try:
        envelope = json.loads(candidate)
    except (TypeError, ValueError):
        objects = _json_objects(candidate)
        if not objects:
            raise NoIntentEmitted(f"deepseek reply carried no JSON object (chars={len(text or '')})") from None
        # The contract puts the envelope last, after any reasoning that quotes a payload shape.
        envelopes = [obj for obj in objects if isinstance(obj, dict) and isinstance(obj.get("intents"), list)]
        envelope = envelopes[-1] if envelopes else objects[-1]
    if not isinstance(envelope, dict) or not isinstance(envelope.get("intents"), list):
        raise NoIntentEmitted("deepseek reply is valid JSON but carries no 'intents' list")
    # An empty list satisfies validate_envelope, so without this the tick is
    # recorded as a success that did nothing and the error streak is reset.
    if not envelope["intents"]:
        raise NoIntentEmitted("deepseek reply carried an empty 'intents' list")
    decoded: list[Any] = []
    for index, item in enumerate(envelope["intents"]):
        if not isinstance(item, dict):
            raise NoIntentEmitted(f"deepseek intents[{index}] is not an object")
        if "payload" not in item:
            keys = ", ".join(sorted(str(key) for key in item))
            raise NoIntentEmitted(f"deepseek intents[{index}] has no 'payload' key (keys: {keys})")
        payload = item.get("payload")
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except ValueError as exc:
                raise NoIntentEmitted(f"deepseek intents[{index}] payload is not decodable JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise NoIntentEmitted(f"deepseek intents[{index}] payload did not decode to an object")
        decoded.append({"intent_type": item.get("intent_type"), "payload": payload})
    return {"intents": decoded}


@dataclass
class DeepSeekBackend:
    """Production DeepSeek Harness reactor backend. Implements :class:`Backend`."""

    allowed_intents: frozenset[IntentType]
    model: str = DEFAULT_DEEPSEEK_HARNESS_MODEL
    cwd: Path = field(default_factory=Path.cwd)
    dsh_home: Path | None = None
    reasoning_effort: str = ""
    permission_mode: str = "read-only"
    # An agent turn carries a tool loop, so it needs the orchestration budget,
    # not a completion's 120s.
    call_timeout_s: float = field(
        default_factory=lambda: parse_call_timeout_env(
            "INFERENCE_OPTIMIZER_DEEPSEEK_CALL_TIMEOUT_SEC",
            default=300.0,
        )
    )
    env: dict[str, str] | None = None
    # Bounded transient-failure retry/backoff, on the same
    # INFERENCE_OPTIMIZER_LLM_RETRY_* knobs the Claude and Codex paths read.
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy.from_env)

    name: str = "deepseek"
    calls: list[dict[str, Any]] = field(default_factory=list)

    # The harness mounts no MCP server a reactor role could hand tools through,
    # so intents come back in the reply body like they do on the Codex route.
    transport = TRANSPORT_STRUCTURED_OUTPUT

    _session: DeepSeekSession | None = field(default=None, init=False, repr=False)
    # Instructions the open conversation was started with.
    _conversation_instructions: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        """Normalize and secure the session-private runtime root."""
        if not self.allowed_intents:
            raise BackendError("DeepSeekBackend requires the emitting role's allowed_intents")
        self.cwd = Path(self.cwd).expanduser().resolve()
        try:
            self.cwd.mkdir(parents=True, mode=0o700, exist_ok=True)
            self.cwd.chmod(0o700)
        except OSError as exc:
            raise BackendError(f"cannot prepare DeepSeek backend cwd {self.cwd}: {exc}") from exc

    # ------------------------------------------------------------------
    async def run(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        tools: list[str] | None = None,
        max_turns: int = 1,
    ) -> BackendTurnResult:
        """Run one DeepSeek Harness turn and parse the intent envelope it replied with."""
        # The harness owns its tool set and its own turn loop.
        del tools, max_turns
        turn_prompt = f"{prompt}\n\n{_ENVELOPE_REMINDER} {self._legal_intent_types()}"

        sdk_result = await self._drive_turn(system_prompt, turn_prompt, prompt_chars=len(prompt))
        usage = dict(sdk_result.usage or {})
        rejection: NoIntentEmitted | None = None
        intents: list[Any] = []
        try:
            intents = self._parse(sdk_result.text)
        except NoIntentEmitted as exc:
            rejection = exc
        if rejection is not None:
            # A schema-backed transport hands the validation error back to the model within the turn.
            correction = (
                f"{_REJECTED_REPLY.format(error=rejection)}\n\n{_ENVELOPE_REMINDER} {self._legal_intent_types()}"
            )
            sdk_result = await self._drive_turn(system_prompt, correction, prompt_chars=len(correction))
            for key, value in (sdk_result.usage or {}).items():
                if key == "context_tokens_peak":
                    usage[key] = max(to_int(usage.get(key), 0), to_int(value, 0))
                else:
                    usage[key] = to_int(usage.get(key), 0) + to_int(value, 0)

        input_tokens = to_int(usage.get("input_tokens"), 0)
        output_tokens = to_int(usage.get("output_tokens"), 0)
        cache_read_tokens = to_int(usage.get("cache_read_input_tokens"), 0)
        metadata: dict[str, Any] = {
            "model": self.model,
            "session_id": sdk_result.session_id,
            # Pairs this turn's token row with its conversation row. A caller that opened an ``llm.call``
            # trajectory span owns the id.
            "call_id": current_context().call_id or new_call_id(),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": cache_read_tokens,
            # The largest single request of the turn -- the figure the checkpoint
            # policy compares against the model's window.
            "context_tokens_peak": to_int(usage.get("context_tokens_peak"), 0),
            # Full conversation text for conversations.jsonl, handed up for the
            # caller to persist.
            "prompt": prompt,
            "response": sdk_result.text,
        }

        if rejection is not None:
            intents = self._parse(sdk_result.text)
        return BackendTurnResult(intents=intents, raw_text=sdk_result.text, metadata=metadata)

    async def _drive_turn(self, system_prompt: str | None, turn_prompt: str, *, prompt_chars: int) -> Any:
        """Run one harness turn under the transient-failure retry policy and log its usage."""

        async def _one_attempt() -> Any:
            session = await self._session_for(system_prompt)
            return await session.turn(turn_prompt, timeout_sec=self.call_timeout_s)

        def _note_retry(attempt: int, exc: BaseException, delay: float) -> None:
            self.calls.append(
                {"warn": f"deepseek SDK transient failure (attempt {attempt}): {exc!r}; retrying in {delay:.2f}s"}
            )

        try:
            sdk_result = await retry_with_backoff(
                _one_attempt,
                policy=self.retry_policy,
                retry_on=(DeepSeekSessionError,),
                on_retry=_note_retry,
            )
        except DeepSeekSessionError as exc:
            raise LLMCallFailed(f"DeepSeek Harness turn failed: {redact_secret_values(str(exc))}") from exc
        if sdk_result.error:
            raise LLMCallFailed("DeepSeek Harness turn failed: " + redact_secret_values(sdk_result.error))
        usage = dict(sdk_result.usage or {})
        self.calls.append(
            {
                "model": self.model,
                "prompt_chars": prompt_chars,
                "reply_chars": len(sdk_result.text),
                "input_tokens": to_int(usage.get("input_tokens"), 0),
                "output_tokens": to_int(usage.get("output_tokens"), 0),
                "cache_read_input_tokens": to_int(usage.get("cache_read_input_tokens"), 0),
                "session_id": sdk_result.session_id,
            }
        )
        return sdk_result

    @staticmethod
    def _parse(text: str) -> list[Any]:
        envelope = decode_intent_envelope(text)
        try:
            return validate_envelope(envelope)
        except IntentValidationError as exc:
            raise NoIntentEmitted(f"deepseek envelope invalid: {exc}") from exc

    def _legal_intent_types(self) -> str:
        """The one line that stands in for the schema enum this route lacks.

        The shared contract writes each intent as ``name:{required keys}`` and
        leaves the value set to the enforced schema. Nothing enforces one here,
        and a live run read that notation as licence to qualify the name
        (``send_message:kernel``), so the legal values are spelled out.
        """
        legal = ", ".join(sorted(intent.value for intent in self.allowed_intents))
        return f"`intent_type` MUST be exactly one of: {legal}. Never add a suffix or qualifier to it."

    def _instructions_for(self, system_prompt: str | None) -> str:
        """Conversation-level instructions implied by one system prompt."""
        return "\n\n".join(
            part
            for part in (
                DEEPSEEK_RUNTIME_MAPPING.strip(),
                (system_prompt or "").strip(),
                build_output_instructions(self.allowed_intents),
                self._legal_intent_types(),
            )
            if part
        )

    async def aclose(self) -> None:
        """Release the held session: SDK client, child process, harness home."""
        session, self._session = self._session, None
        if session is not None:
            await session.aclose()

    async def _session_for(self, system_prompt: str | None) -> DeepSeekSession:
        """Return the held session, opening it (or re-scoping it) as needed."""
        instructions = self._instructions_for(system_prompt)
        if self._session is None:
            self._session = DeepSeekSession(
                cwd=self.cwd,
                model=self.model,
                instructions=instructions,
                dsh_home=self.dsh_home,
                reasoning_effort=self.reasoning_effort,
                permission_mode=self.permission_mode,
                env=self.env,
                request_timeout_sec=self.call_timeout_s,
            )
            try:
                await self._session.start()
            except DeepSeekSessionUnavailableError as exc:
                # A config or sandbox fault, identical on every attempt.
                self._session = None
                raise BackendError(redact_secret_values(str(exc))) from exc
            except DeepSeekSessionError:
                # Left as it is so the retry wrapper around the attempt can see it.
                self._session = None
                raise
        elif instructions != self._conversation_instructions:
            self._session.instructions = instructions
            self._session.reset_conversation()
        self._conversation_instructions = instructions
        return self._session


__all__ = ["DeepSeekBackend", "decode_intent_envelope"]
