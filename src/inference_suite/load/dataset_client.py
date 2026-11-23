# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The fixed load's client for a dataset of real prompts, run as ``python <file>`` in place of InferenceX's.

Each prompt is one user message sent to the server's ``/v1/chat/completions``, so the server renders it with the
model's own chat template, streamed at temperature 0 with ``max_tokens`` the workload's OSL and EOS honoured: a real
prompt's answer ends where the model ends it. A pass sends ``--num-prompts`` requests from the first
``--max-prompts`` lines, shuffled by ``--seed`` and cycled, so two arms on one seed send identical requests.

Token counts are the server's ``usage``: a request the server reports none for fails, as this client has no
tokenizer. The saved result carries the keys the fixed load reads from InferenceX's result, computed as
InferenceX's ``benchmark_serving`` computes them.

It imports nothing outside the standard library: it runs under whatever interpreter runs InferenceX's client.
"""

from __future__ import annotations

import argparse
import http.client
import json
import math
import random
import sys
import threading
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

ENDPOINT = "/v1/chat/completions"
#: InferenceX's request-failure gate: a pass with more failed requests than this share exits non-zero.
MAX_FAILURE_RATE = 0.05
#: How long one read of a response may block; InferenceX bounds a whole request by six hours.
READ_TIMEOUT_SEC = 6 * 3600
PERCENTILE_METRICS = ("ttft", "tpot", "e2el")
#: The delta fields a streamed chunk carries generated text in: the answer, and the reasoning a server splits off it.
_TEXT_FIELDS = ("content", "reasoning_content", "reasoning")
_ENCODING = "utf-8"


class DatasetError(ValueError):
    """The dataset file holds no prompt this client can send."""


def read_prompts(path: Path, max_prompts: int | None) -> list[str]:
    """The prompts of the first ``max_prompts`` lines of the JSONL file at ``path`` (all when ``None``): each line's
    ``prompt`` string, else the first of its ``turns``."""
    prompts: list[str] = []
    for number, line in enumerate(path.read_text(encoding=_ENCODING).splitlines(), start=1):
        if max_prompts is not None and len(prompts) == max_prompts:
            break
        if not line.strip():
            continue
        row = json.loads(line)
        prompt = row.get("prompt") if isinstance(row, dict) else None
        turns = row.get("turns") if isinstance(row, dict) else None
        if prompt is None and isinstance(turns, list) and turns:
            prompt = turns[0]
        if not isinstance(prompt, str) or not prompt.strip():
            raise DatasetError(f"{path}:{number} holds no 'prompt' string and no 'turns' list of strings")
        prompts.append(prompt)
    if not prompts:
        raise DatasetError(f"{path} holds no prompt")
    return prompts


def request_order(pool: list[str], num_prompts: int, seed: int) -> list[str]:
    """``num_prompts`` prompts of ``pool``: each cycle through it shuffled by one generator seeded with ``seed``."""
    rng = random.Random(seed)
    order: list[str] = []
    while len(order) < num_prompts:
        cycle = list(pool)
        rng.shuffle(cycle)
        order += cycle
    return order[:num_prompts]


@dataclass
class RequestOutput:
    """One request as InferenceX's ``RequestFuncOutput`` records it."""

    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0
    prompt_len: int = 0
    output_tokens: int = 0
    generated_text: str = ""
    error: str = ""


@dataclass(frozen=True)
class Target:
    """Where requests go and what they ask for."""

    host: str
    port: int
    served_model_name: str
    max_tokens: int


def _chunk_text(chunk: dict) -> str:
    choices = chunk.get("choices") or []
    delta = (choices[0].get("delta") or {}) if choices else {}
    return "".join(value for name in _TEXT_FIELDS if isinstance(value := delta.get(name), str))


def _read_stream(response: http.client.HTTPResponse, output: RequestOutput, start: float) -> None:
    last = start
    usage: dict = {}
    for raw in response:
        line = raw.decode(_ENCODING).strip()
        if not line.startswith("data:"):
            continue
        payload = line.removeprefix("data:").strip()
        if payload == "[DONE]":
            break
        chunk = json.loads(payload)
        if chunk.get("error"):
            output.error = json.dumps(chunk["error"])
            return
        if text := _chunk_text(chunk):
            last = time.perf_counter()
            if not output.generated_text:
                output.ttft = last - start
            output.generated_text += text
        usage = chunk.get("usage") or usage
    output.latency = last - start
    if not output.generated_text:
        output.error = "the server streamed no generated text"
    elif not isinstance(usage.get("completion_tokens"), int) or not isinstance(usage.get("prompt_tokens"), int):
        output.error = f"the server reported no token usage: {usage!r}"
    else:
        output.output_tokens, output.prompt_len = usage["completion_tokens"], usage["prompt_tokens"]
        output.success = True


def send(target: Target, prompt: str) -> RequestOutput:
    """Stream one chat request for ``prompt`` and time it as InferenceX times a request."""
    body = {
        "model": target.served_model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": target.max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    output = RequestOutput()
    connection = http.client.HTTPConnection(target.host, target.port, timeout=READ_TIMEOUT_SEC)
    start = time.perf_counter()
    try:
        connection.request("POST", ENDPOINT, json.dumps(body), {"Content-Type": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            output.error = f"HTTP {response.status}: {response.read().decode(_ENCODING, 'replace')[:500]}"
        else:
            _read_stream(response, output, start)
    except (OSError, http.client.HTTPException, ValueError) as exc:  # one request's failure, recorded in its output
        output.error = f"{type(exc).__name__}: {exc}"
    finally:
        connection.close()
    return output


@dataclass(frozen=True)
class Pass:
    """What one pass sent: each request's output in prompt order, its duration and its start and end as Unix times."""

    outputs: list[RequestOutput]
    duration: float
    start_unix: float
    end_unix: float


def run_requests(target: Target, prompts: list[str], concurrency: int) -> Pass:
    """Send ``prompts`` in order, at most ``concurrency`` at once."""
    outputs = [RequestOutput() for _ in prompts]
    lock = threading.Lock()
    pending = iter(range(len(prompts)))

    def worker() -> None:
        while True:
            with lock:
                index = next(pending, None)
            if index is None:
                return
            outputs[index] = send(target, prompts[index])
            state = "ok" if outputs[index].success else f"failed: {outputs[index].error}"
            print(f"request {index + 1}/{len(prompts)} {state}", flush=True)

    workers = [threading.Thread(target=worker) for _ in range(min(concurrency, len(prompts)))]
    start, start_unix = time.perf_counter(), time.time()
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join()
    return Pass(outputs, time.perf_counter() - start, start_unix, time.time())


def percentile(values: list[float], p: float) -> float:
    """``numpy.percentile`` with its default linear interpolation."""
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p / 100
    low, high = math.floor(rank), math.ceil(rank)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def _stats(name: str, seconds: list[float], percentiles: list[float]) -> dict[str, float]:
    values = [value * 1000 for value in seconds] or [0.0]
    mean = sum(values) / len(values)
    stats = {
        f"mean_{name}_ms": mean,
        f"median_{name}_ms": percentile(values, 50),
    }
    for p in percentiles:
        stats[f"p{int(p) if int(p) == p else p}_{name}_ms"] = percentile(values, p)
    return stats


def outcome(requested: int, completed: int) -> dict[str, object]:
    """InferenceX's request-count verdict."""
    failed = requested - completed
    return {
        "status": "failed" if failed / requested > MAX_FAILURE_RATE else "passed",
        "requested": requested,
        "completed": completed,
        "failed": failed,
        "max_failure_rate": MAX_FAILURE_RATE,
    }


def summarize(outputs: list[RequestOutput], duration: float, percentiles: list[float]) -> dict[str, object]:
    """The metrics of InferenceX's ``calculate_metrics`` a fixed round reads, keyed as its result saves them."""
    done = [out for out in outputs if out.success]
    total_input, total_output = sum(out.prompt_len for out in done), sum(out.output_tokens for out in done)
    tpots = [(out.latency - out.ttft) / (out.output_tokens - 1) for out in done if out.output_tokens > 1]
    result: dict[str, object] = {
        "completed": len(done),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "request_throughput": len(done) / duration,
        "output_throughput": total_output / duration,
        "total_token_throughput": (total_input + total_output) / duration,
    }
    latencies = {
        "ttft": [out.ttft for out in done],
        "tpot": tpots,
        "e2el": [out.latency for out in done],
    }
    for name in PERCENTILE_METRICS:
        result.update(_stats(name, latencies[name], percentiles))
    return result


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--num-prompts", type=int, required=True)
    parser.add_argument("--max-concurrency", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--metric-percentiles", required=True)
    parser.add_argument("--result", type=Path, default=None)
    return parser.parse_args(argv)


def _target(args: argparse.Namespace) -> Target:
    url = urllib.parse.urlsplit(args.base_url)
    if url.scheme != "http" or not url.hostname:
        raise SystemExit(f"--base-url must be http://<host>[:<port>], got {args.base_url!r}")
    return Target(url.hostname, url.port or 80, args.served_model_name, args.max_tokens)


def _result(args: argparse.Namespace, sent: Pass, metrics: dict[str, object], verdict: dict[str, object]) -> dict:
    return {
        "num_prompts": args.num_prompts,
        "duration": sent.duration,
        "benchmark_start_time_unix": sent.start_unix,
        "benchmark_end_time_unix": sent.end_unix,
        **metrics,
        "benchmark_outcome": verdict,
    }


def main(argv: list[str]) -> int:
    args = _parse(argv)
    target = _target(args)
    prompts = request_order(read_prompts(args.dataset, args.max_prompts), args.num_prompts, args.seed)
    sent = run_requests(target, prompts, args.max_concurrency)
    metrics = summarize(sent.outputs, sent.duration, [float(p) for p in args.metric_percentiles.split(",")])
    verdict = outcome(args.num_prompts, int(metrics["completed"]))
    if args.result is not None:
        args.result.write_text(json.dumps(_result(args, sent, metrics, verdict)), encoding=_ENCODING)
    if verdict["status"] == "failed":
        print(f"FAIL: {verdict['failed']} of {verdict['requested']} requests failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
