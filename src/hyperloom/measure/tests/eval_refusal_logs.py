# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Logs of a test whose accuracy evaluation was refused, for the tests that classify it.

The server is started with ``--context-length 2304`` while the point asks the evaluation for ``max_length=8192`` with
4096 completion tokens, so every request is answered ``400``. The lines are in the words of SGLang and the evaluation
client. What the log does next is the shape these tests need: the client prints the refusal once per request, then one
traceback per request that gave up (:data:`CASCADE_BLOCKS` of them here), so the refusal ends far before the end of the
file.
"""

from __future__ import annotations

#: The client's report of a refused request, with the server's message.
REFUSAL = (
    "2000-01-01:00:00:00 WARNING  [models.api_models:521] API request failed! Status code: 400, Response text: "
    '{"object":"error","message":"max_completion_tokens is too large: 4096.This model supports at most 2304 '
    'completion tokens.","type":"BadRequestError","param":null,"code":400}. Retrying...'
)
EVAL_HEAD = (
    "2000-01-01:00:00:00 INFO     [models.api_models:733] Tokenized requests are disabled. Context + generation length "
    "is not checked.\n"
    "\n"
    "Requesting API:   0%|          | 0/1319 [00:00<?, ?it/s]"
    'HYPERLOOM_EVAL_BOUNDS max_tokens=4096 stop=["</s>", "<|im_end|>"] stop_token_ids=null\n'
    f"{REFUSAL}\n"
    "2000-01-01:00:00:00 INFO     [models.api_models:593] Retry attempt 1\n"
)
#: What the evaluation shim leaves for each request that gave up: the harness error handler raised over the 400.
CASCADE = """\
Task exception was never retrieved
future: <Task finished name='Task-2' coro=<tqdm_asyncio.gather.<locals>.wrap_awaitable() done, defined at /opt/venv/lib/python3.10/site-packages/tqdm/asyncio.py:75> exception=RuntimeError('lm_eval request failed before a response was parsed; the harness error handler then raised UnboundLocalError over the original error (EleutherAI/lm-evaluation-harness#3293). The usual cause is a refused connection to the served endpoint.')>
Traceback (most recent call last):
  File "/opt/venv/lib/python3.10/site-packages/tqdm/asyncio.py", line 76, in wrap_awaitable
    return i, await f
  File "/opt/venv/lib/python3.10/site-packages/tenacity/asyncio/__init__.py", line 232, in async_wrapped
    return await copy(fn, *args, **kwargs)  # type: ignore[type-var]
  File "/opt/venv/lib/python3.10/site-packages/tenacity/asyncio/__init__.py", line 126, in __call__
    do = await self.iter(retry_state=retry_state)
  File "/opt/venv/lib/python3.10/site-packages/tenacity/asyncio/__init__.py", line 171, in iter
    result = await action(retry_state)
  File "/opt/venv/lib/python3.10/site-packages/tenacity/_utils.py", line 130, in inner
    return call(*args, **kwargs)
  File "/opt/venv/lib/python3.10/site-packages/tenacity/__init__.py", line 482, in exc_check
    retry_exc.reraise()
  File "/opt/venv/lib/python3.10/site-packages/tenacity/__init__.py", line 216, in reraise
    raise self.last_attempt.result()
  File "/usr/lib/python3.10/concurrent/futures/_base.py", line 451, in result
    return self.__get_result()
  File "/usr/lib/python3.10/concurrent/futures/_base.py", line 403, in __get_result
    raise self._exception
  File "/opt/venv/lib/python3.10/site-packages/tenacity/asyncio/__init__.py", line 130, in __call__
    result = await fn(*args, **kwargs)
  File "/tmp/tmp.XXXXXXXXXX/sitecustomize.py", line 277, in _hl_probe_amodel_call
    return await _hl_prev_amodel_call(self, session, sem, messages, **kwargs)
  File "/tmp/tmp.XXXXXXXXXX/sitecustomize.py", line 150, in _hl_amodel_call
    raise RuntimeError(
RuntimeError: lm_eval request failed before a response was parsed; the harness error handler then raised UnboundLocalError over the original error (EleutherAI/lm-evaluation-harness#3293). The usual cause is a refused connection to the served endpoint.
"""
CASCADE_BLOCKS = 100
EVAL_TAIL = (
    "+ local eval_exit=1\n"
    "+ set +x\n"
    "ERROR: run_eval failed with exit code 1\n"
    "[magpie_bench_remote_compat] accuracy artifact: /work/out/variant_00_warmup/benchmark_sglang/accuracy_report.json\n"
)
SERVER_REFUSAL = (
    '[2000-01-01 00:00:00] INFO:     127.0.0.1:40000 - "POST /v1/chat/completions HTTP/1.1" 400 Bad Request\n'
)
SERVER_DECODE = (
    "[2000-01-01 00:01:00] Decode batch, #running-req: 64, #token: 100000, token usage: 0.04, cuda graph: True, "
    "gen throughput (token/s): 1000.00, #queue-req: 0\n"
)


def benchmark_stderr() -> str:
    """The benchmark's stderr as a refused evaluation leaves it: the refusal, then the cascade, then the exit line."""
    return EVAL_HEAD + CASCADE * CASCADE_BLOCKS + EVAL_TAIL


def server_log() -> str:
    """The server's log: its refusals, then the decode lines of a second round served on the same server."""
    return SERVER_REFUSAL * 40 + SERVER_DECODE * 1000
