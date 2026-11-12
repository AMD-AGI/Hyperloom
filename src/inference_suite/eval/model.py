# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The lm-eval model an eval scores through: chat completions against the served model.

Imported only inside the eval process, so it builds on lm-eval, the standard library and the eval request alone. It
reads its probe thresholds, probe file and model directory from the request file. Over lm-eval's
``LocalChatCompletion`` it:

- takes a reasoning model's ``reasoning_content`` when ``content`` is empty, and sends chat history without lm-eval's
  ``type`` field, as InferenceX's lm-eval patches do;
- stops generation on every terminator the model's own metadata declares;
- counts responses cut at the token limit and, once the share reaches the probe's ratio after its minimum samples,
  answers every remaining generation empty instead of sending it, on both lm-eval's concurrent and sequential paths:
  a model that does not terminate scores near 0 instead of running the eval out;
- reports a request that failed before any response as itself, not as lm-eval's ``UnboundLocalError``
  (EleutherAI/lm-evaluation-harness#3293, still present at the pinned commit).
"""

from __future__ import annotations

import asyncio
import atexit
import json
import os
import sys
from typing import Any

from lm_eval.models.api_models import JsonChatStr
from lm_eval.models.openai_completions import LocalChatCompletion

from inference_suite.eval.run import read_request

#: Stop strings the chat completions API takes at most.
MAX_STOP_STRINGS = 4


def _read_json(path: str) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _token_text(value: object) -> str | None:
    """A special token's text: tokenizer configs write it bare or as an ``AddedToken`` mapping."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("content"), str):
        return value["content"]
    return None


def _is_token_id(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def model_terminators(model_dir: str | None) -> tuple[list[str], list[int]]:
    """The stop strings and stop token ids ``model_dir``'s metadata declares.

    ``generation_config.json`` may list several EOS ids (Qwen3 declares two) where ``eos_token`` names one, and
    lm-eval's concurrent path sends neither, so the model's own list is read here. ``tokenizer_config.json``'s
    ``eos_token`` is the fallback, which gives a string and no id.
    """
    if not model_dir or not os.path.isdir(model_dir):
        return [], []
    tokenizer = _read_json(os.path.join(model_dir, "tokenizer_config.json"))
    generation = _read_json(os.path.join(model_dir, "generation_config.json"))
    decoder = tokenizer.get("added_tokens_decoder")
    decoder = decoder if isinstance(decoder, dict) else {}
    declared = generation.get("eos_token_id")
    declared = [declared] if _is_token_id(declared) else declared if isinstance(declared, list) else []
    ids: list[int] = []
    stops: list[str] = []
    for token_id in declared:
        if not _is_token_id(token_id) or token_id in ids:
            continue
        ids.append(token_id)
        text = _token_text(decoder.get(str(token_id)))
        if text and text not in stops:
            stops.append(text)
    text = _token_text(tokenizer.get("eos_token"))
    if text and text not in stops:
        stops.append(text)
    return stops, ids


def generations(output: object) -> list[str]:
    """The texts of one chat completions response, by choice index; ``[""]`` for a response that has none."""
    choices = output.get("choices") if isinstance(output, dict) else None
    if not isinstance(choices, list):
        return [""]
    texts = [""] * len(choices)
    for choice in choices:
        index = choice.get("index", 0) if isinstance(choice, dict) else None
        if not _is_token_id(index) or not 0 <= index < len(texts):
            return [""]
        message = choice.get("message")
        message = message if isinstance(message, dict) else {}
        content = message.get("content")
        texts[index] = content if content not in (None, "", []) else message.get("reasoning_content") or ""
    return texts


class TruncationProbe:
    """Counts generations and those cut at the token limit, and trips once the cut share reaches ``length_ratio``."""

    def __init__(self, *, enabled: bool, min_samples: int, length_ratio: float) -> None:
        self.enabled = enabled
        self.min_samples = min_samples
        self.length_ratio = length_ratio
        self.samples_seen = 0
        self.cut = 0
        self.tripped = False

    @property
    def length_ratio_observed(self) -> float | None:
        return self.cut / self.samples_seen if self.samples_seen else None

    def observe(self, outputs: list[object]) -> bool:
        """Count ``outputs``; true when this batch trips the probe."""
        for output in outputs:
            choices = output.get("choices") if isinstance(output, dict) else None
            for choice in choices if isinstance(choices, list) else []:
                self.samples_seen += 1
                if isinstance(choice, dict) and choice.get("finish_reason") == "length":
                    self.cut += 1
        if self.tripped or not self.enabled or self.samples_seen < self.min_samples:
            return False
        self.tripped = self.cut / self.samples_seen >= self.length_ratio
        return self.tripped

    def record(self) -> dict[str, object]:
        """The outcome the report's ``eval.probe`` carries."""
        return {
            "tripped": self.tripped,
            "samples_seen": self.samples_seen,
            "length_ratio_observed": self.length_ratio_observed,
        }


def bounded_payload(payload: dict[str, Any], stops: list[str], stop_ids: list[int]) -> dict[str, Any]:
    """``payload`` stopping on the model's terminators too.

    The task's own stop strings keep their place ahead of the model's: its answer extraction depends on them, and the
    API keeps only the first :data:`MAX_STOP_STRINGS`. ``stop_token_ids`` has no such limit.
    """
    if stop_ids:
        payload["stop_token_ids"] = [*stop_ids, *(i for i in payload.get("stop_token_ids") or [] if i not in stop_ids)]
    if stops:
        merged = [item for item in payload.get("stop") or [] if item is not None]
        merged += [item for item in stops if item not in merged]
        payload["stop"] = merged[:MAX_STOP_STRINGS]
    return payload


class SuiteChatCompletion(LocalChatCompletion):
    """``local-chat-completions`` with the suite's probe and terminators; see the module docstring."""

    def __init__(self, *, request: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        settings = read_request(request)
        self._probe = TruncationProbe(
            enabled=settings.probe_enabled,
            min_samples=settings.probe_min_samples,
            length_ratio=settings.probe_length_ratio,
        )
        self._probe_file = settings.probe_file
        self._stops, self._stop_ids = model_terminators(settings.model_dir)
        # asyncio.run gives each batch a fresh loop, and a semaphore binds to the first loop that awaits it.
        self._gate: tuple[asyncio.AbstractEventLoop, asyncio.Semaphore] | None = None
        atexit.register(self._write_probe)

    def _write_probe(self) -> None:
        os.makedirs(os.path.dirname(self._probe_file) or ".", exist_ok=True)
        with open(self._probe_file, "w", encoding="utf-8") as fh:
            json.dump(self._probe.record(), fh, sort_keys=True)

    def _loop_gate(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        if self._gate is None or self._gate[0] is not loop:
            self._gate = (loop, asyncio.Semaphore(max(1, self._concurrent)))
        return self._gate[1]

    def parse_generations(self, outputs: Any, **kwargs: Any) -> list[str]:  # type: ignore[override]
        batch = outputs if isinstance(outputs, list) else [outputs]
        if self._probe.observe(batch):
            print("INFERENCE_SUITE_EVAL_PROBE_TRIPPED " + json.dumps(self._probe.record()), file=sys.stderr, flush=True)
            self._write_probe()
        return [text for output in batch for text in generations(output)]

    def apply_chat_template(self, chat_history: list[dict[str, str]], add_generation_prompt: bool = True) -> Any:
        if self.tokenizer_backend == "huggingface" and self.tokenized_requests:
            return self.tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt,
            )
        if self.tokenizer_backend == "remote" and self.tokenized_requests:
            return chat_history
        return JsonChatStr(json.dumps(list(chat_history), ensure_ascii=False))

    def _create_payload(self, messages: Any, generate: bool = False, **kwargs: Any) -> dict[str, Any]:
        payload = super()._create_payload(messages, generate=generate, **kwargs)
        return bounded_payload(payload, self._stops, self._stop_ids) if generate else payload

    def model_call(self, messages: Any, *, generate: bool = True, **kwargs: Any) -> Any:
        # lm-eval's sequential path (num_concurrent <= 1) parses this response itself; a response with no choices
        # parses to one empty answer per request, chat completions sending one request per batch.
        if self._probe.tripped and generate:
            return {}
        return super().model_call(messages, generate=generate, **kwargs)

    async def amodel_call(self, session: Any, sem: Any, messages: Any, **kwargs: Any) -> Any:
        # lm-eval starts every request at once and the base builds a payload before taking its semaphore, so
        # requests wait here instead, and the probe is checked once each gets its turn.
        async with self._loop_gate():
            if self._probe.tripped and kwargs.get("generate", True):
                answers = [""] * len(messages)
                for answer, cache_key in zip(answers, kwargs.get("cache_keys") or []):
                    self.cache_hook.add_partial("generate_until", cache_key, answer)
                return answers
            try:
                return await super().amodel_call(session, sem, messages, **kwargs)
            except UnboundLocalError as exc:
                raise RuntimeError(
                    "an eval request failed before any response was parsed, and lm-eval's error handler raised "
                    "UnboundLocalError over the original error (EleutherAI/lm-evaluation-harness#3293); the usual "
                    "cause is a refused connection to the served endpoint"
                ) from exc


__all__ = [
    "MAX_STOP_STRINGS",
    "SuiteChatCompletion",
    "TruncationProbe",
    "bounded_payload",
    "generations",
    "model_terminators",
]
