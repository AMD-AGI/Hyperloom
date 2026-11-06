# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The eval process: ``python -I -m inference_suite.eval.run <request.json>`` scores one task against a served model.

It runs in the staged lm-eval environment, so at import it needs only the standard library; lm-eval is imported once it runs, and
imports :mod:`inference_suite.eval.model` itself. The request holds every input, and nothing is read from
the environment.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path

EVAL_REQUEST_SCHEMA = "inference-suite-eval-request/1"
EVAL_MODULE = "inference_suite.eval.run"
#: The lm-eval model name the eval runs under, and the class lm-eval imports for it once asked.
MODEL_NAME = "inference-suite-chat"
MODEL_CLASS = "inference_suite.eval.model:SuiteChatCompletion"
#: Per-request timeout and retries, as InferenceX's ``run_lm_eval`` sets them.
REQUEST_TIMEOUT_SEC = 1800
MAX_RETRIES = 5
#: lm-eval splits ``--model_args`` on it, so no value may hold one.
_MODEL_ARGS_SEPARATOR = ","


class EvalRequestError(ValueError):
    """An eval request that cannot be run as written."""


@dataclass(frozen=True)
class EvalRequest:
    """One eval run: where to send requests, what to score, and the probe's thresholds.

    ``max_tokens`` is the completion budget of every generation; ``model_dir`` is the served model's local directory,
    read for the terminators its metadata declares, or ``None`` when the model is not local.
    """

    task_yaml: str
    base_url: str
    served_model_name: str
    output_dir: str
    probe_file: str
    concurrency: int
    max_length: int
    max_tokens: int
    num_fewshot: int
    probe_enabled: bool
    probe_min_samples: int
    probe_length_ratio: float
    limit: int | None = None
    model_dir: str | None = None

    def to_document(self) -> dict[str, object]:
        return {"schema": EVAL_REQUEST_SCHEMA, **asdict(self)}

    @classmethod
    def from_document(cls, document: object) -> EvalRequest:
        if not isinstance(document, Mapping):
            raise EvalRequestError(f"an eval request is a JSON object, got {type(document).__name__}")
        if document.get("schema") != EVAL_REQUEST_SCHEMA:
            raise EvalRequestError(f"the eval request schema is {document.get('schema')!r}, not {EVAL_REQUEST_SCHEMA}")
        names = {field.name for field in fields(cls)}
        unknown = sorted(set(document) - names - {"schema"})
        if unknown:
            raise EvalRequestError(f"unknown eval request keys: {', '.join(unknown)}")
        try:
            return cls(**{name: document[name] for name in names if name in document})
        except TypeError as exc:
            raise EvalRequestError(f"an incomplete eval request: {exc}") from exc


def read_request(path: str) -> EvalRequest:
    """The eval request written to ``path``."""
    return EvalRequest.from_document(json.loads(Path(path).read_text(encoding="utf-8")))


def model_args(request: EvalRequest, request_file: str) -> str:
    """The ``--model_args`` of ``request``: lm-eval's own settings, then ``request_file``, which the suite's model
    class reads its probe and model directory from."""
    values: dict[str, object] = {
        "model": request.served_model_name,
        "base_url": request.base_url,
        "num_concurrent": request.concurrency,
        "max_retries": MAX_RETRIES,
        "timeout": REQUEST_TIMEOUT_SEC,
        "tokenized_requests": False,
        "max_length": request.max_length,
        "request": request_file,
    }
    for name, value in values.items():
        if _MODEL_ARGS_SEPARATOR in str(value):
            raise EvalRequestError(f"the eval model arg {name}={value!r} holds a ',' lm-eval would split it on")
    return _MODEL_ARGS_SEPARATOR.join(f"{name}={value}" for name, value in values.items())


def lm_eval_argv(request: EvalRequest, request_file: str) -> list[str]:
    """The lm-eval ``run`` arguments of ``request``, read from ``request_file``: greedy chat completions with the
    request's completion budget."""
    argv = [
        "run",
        "--model",
        MODEL_NAME,
        "--apply_chat_template",
        "--tasks",
        request.task_yaml,
        "--num_fewshot",
        str(request.num_fewshot),
        "--output_path",
        request.output_dir,
        "--log_samples",
        "--model_args",
        model_args(request, request_file),
        "--gen_kwargs",
        f"max_tokens={request.max_tokens},temperature=0,top_p=1",
    ]
    if request.limit is not None:
        argv += ["--limit", str(request.limit)]
    return argv


def main(argv: Sequence[str] | None = None) -> None:
    """Run the eval the request file names; lm-eval exits with its own status."""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        raise SystemExit(f"usage: python -I -m {EVAL_MODULE} <request.json>")
    request = read_request(args[0])

    from lm_eval.__main__ import cli_evaluate
    from lm_eval.api.registry import model_registry

    model_registry.register(MODEL_NAME, target=MODEL_CLASS)
    # lm-eval's CLI parses sys.argv and offers no argv parameter.
    sys.argv = ["lm-eval", *lm_eval_argv(request, args[0])]
    cli_evaluate()


__all__ = [
    "EVAL_MODULE",
    "EVAL_REQUEST_SCHEMA",
    "MAX_RETRIES",
    "MODEL_CLASS",
    "MODEL_NAME",
    "REQUEST_TIMEOUT_SEC",
    "EvalRequest",
    "EvalRequestError",
    "lm_eval_argv",
    "main",
    "model_args",
    "read_request",
]


if __name__ == "__main__":
    main()
