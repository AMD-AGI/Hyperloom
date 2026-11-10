# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""An eval's definition, fixed on the host before the eval process starts, and the request that process runs.

Two scores compare only under one definition digest. The eval runs in the context the server serves: the
framework's context flag in the launch's final argv, the last one winning, else the model's native context. Every
generation may take what that context leaves after :data:`PROMPT_ROOM_TOKENS`, as InferenceX's ``run_lm_eval``
budgets it, and never more than ``max_tokens_cap``; a launch whose context is unknown is refused
(``eval_context_too_short``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from inference_suite.argv import argv_flag_occurrences, canonical_flag_name, last_alias_value
from inference_suite.contracts.base_launch import DIGEST_PREFIX
from inference_suite.contracts.config import SuiteConfig
from inference_suite.contracts.launch import EvalInvocation, Launch
from inference_suite.contracts.layout import RunLayout
from inference_suite.contracts.report import EvalDefinition, FailedStep
from inference_suite.eval.run import EVAL_ENTRY, EvalRequest
from inference_suite.frameworks import Framework, framework_facts
from inference_suite.runtime.server import server_url

#: Tokens left for the prompt, as InferenceX's ``run_lm_eval`` leaves them.
PROMPT_ROOM_TOKENS = 4096
#: ``config.json`` keys holding a model's native context, in the order InferenceX reads them.
_NATIVE_CONTEXT_KEYS = ("max_position_embeddings", "max_sequence_length", "seq_length", "n_positions")
#: Where a model ships its chat template, newest convention first.
_CHAT_TEMPLATE_FILES = ("chat_template.jinja", "chat_template.json")
_MODEL_CLASS_FILE = Path(__file__).with_name("model.py")
#: The directory holding the ``inference_suite`` package this module was imported from.
_SOURCE_ROOT = Path(__file__).resolve().parents[2]
_UTF8 = "utf-8"


@dataclass(frozen=True)
class StagedEval:
    """What staging fixed for the eval: the task file it scores and the lm-eval version installed."""

    task_yaml: Path
    lm_eval_version: str


@dataclass(frozen=True)
class EvalRefusal:
    """Why the eval cannot run on this launch."""

    step: FailedStep
    detail: str


@dataclass(frozen=True)
class EvalPlan:
    """The definition an eval's score belongs to, and the request the eval process runs."""

    definition: EvalDefinition
    request: EvalRequest

    @property
    def digest(self) -> str:
        return definition_digest(self.definition)

    def invocation(self, python: str, request_file: Path) -> EvalInvocation:
        """The eval process run by ``python`` on the request written to ``request_file``."""
        return EvalInvocation((python, "-I", "-c", EVAL_ENTRY, str(_SOURCE_ROOT), str(request_file)), self.digest)

    def write(self, layout: RunLayout, request_file: Path) -> None:
        """Write the definition into the run_dir's eval directory and the request to ``request_file``."""
        layout.eval_dir.mkdir(parents=True, exist_ok=True)
        layout.eval_definition.write_text(_canonical(self.definition.to_document()) + "\n", encoding=_UTF8)
        request_file.parent.mkdir(parents=True, exist_ok=True)
        request_file.write_text(_canonical(self.request.to_document()) + "\n", encoding=_UTF8)


def _canonical(document: object) -> str:
    return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def definition_digest(definition: EvalDefinition) -> str:
    """The identity of ``definition``: scores with different digests do not compare."""
    return DIGEST_PREFIX + _sha256(_canonical(definition.to_document()).encode())


def _read_json(path: Path) -> dict[str, object]:
    try:
        data = json.loads(path.read_text(encoding=_UTF8))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _native_context(model_dir: Path | None) -> int | None:
    if model_dir is None:
        return None
    config = _read_json(model_dir / "config.json")
    text_config = config.get("text_config")
    for section in (config, text_config if isinstance(text_config, dict) else {}):
        for key in _NATIVE_CONTEXT_KEYS:
            value = section.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
    return None


def served_context(launch: Launch, model_dir: Path | None) -> int | None:
    """The context ``launch`` serves: its last context flag, else the model's native one; ``None`` when unknown."""
    server_flags = framework_facts(Framework(launch.base.framework)).server_flags
    if server_flags is None:
        return None
    flag = frozenset({canonical_flag_name(server_flags.context_length)})
    value = last_alias_value(argv_flag_occurrences(launch.argv), flag)
    if value is not None:
        return int(value) if value.isdigit() and int(value) > 0 else None
    return _native_context(model_dir)


def chat_template_sha256(model_dir: Path | None) -> str | None:
    """The sha256 of the chat template ``model_dir`` ships, or ``None`` when it ships none."""
    if model_dir is None:
        return None
    for name in _CHAT_TEMPLATE_FILES:
        path = model_dir / name
        if path.is_file():
            return _sha256(path.read_bytes())
    template = _read_json(model_dir / "tokenizer_config.json").get("chat_template")
    return None if template is None else _sha256(_canonical(template).encode())


def plan_eval(config: SuiteConfig, launch: Launch, staged: StagedEval, layout: RunLayout) -> EvalPlan | EvalRefusal:
    """The eval ``config`` asks of the server ``launch`` runs, or why that server cannot serve it."""
    server, settings = config.server, config.eval
    model_path = Path(server.model_path)
    model_dir = model_path if model_path.is_dir() else None
    context = served_context(launch, model_dir)
    if context is None:
        detail = "the launch serves an unknown context, so no completion budget fits it"
        return EvalRefusal(FailedStep.EVAL_CONTEXT_TOO_SHORT, detail)
    max_tokens = min(settings.max_tokens_cap, _completion_room(context))
    definition = EvalDefinition(
        task_yaml_sha256=_sha256(staged.task_yaml.read_bytes()),
        lm_eval_version=staged.lm_eval_version,
        model_class_sha256=_sha256(_MODEL_CLASS_FILE.read_bytes()),
        chat_template_sha256=chat_template_sha256(model_dir),
        concurrency=settings.concurrency,
        max_length=context,
        max_tokens_cap=max_tokens,
        limit=settings.limit,
        num_fewshot=settings.num_fewshot,
        golden_applied=launch.forced_acceptance,
        probe=settings.probe,
    )
    request = EvalRequest(
        task_yaml=str(staged.task_yaml),
        base_url=f"{server_url(server)}/v1/chat/completions",
        served_model_name=server.served_model_name,
        output_dir=str(layout.eval_dir),
        probe_file=str(layout.eval_probe),
        concurrency=settings.concurrency,
        max_length=context,
        max_tokens=max_tokens,
        num_fewshot=settings.num_fewshot,
        probe_enabled=settings.probe.enabled,
        probe_min_samples=settings.probe.min_samples,
        probe_length_ratio=settings.probe.length_ratio,
        limit=settings.limit,
        model_dir=None if model_dir is None else str(model_dir),
    )
    return EvalPlan(definition, request)


def _completion_room(context: int) -> int:
    """InferenceX's ``run_lm_eval`` output budget for a served ``context``."""
    return context - PROMPT_ROOM_TOKENS if context > PROMPT_ROOM_TOKENS else context // 2


__all__ = [
    "PROMPT_ROOM_TOKENS",
    "EvalPlan",
    "EvalRefusal",
    "StagedEval",
    "chat_template_sha256",
    "definition_digest",
    "plan_eval",
    "served_context",
]
