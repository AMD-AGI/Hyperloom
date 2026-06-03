#!/usr/bin/env python3
"""Run a batch of OpenAI-compatible LLM gateway requests.

Edit ``build_tasks()`` to return the real list of chat messages. Results are
returned in input order and can also be written as JSONL with ``--output``.

API keys are intentionally read from environment variables instead of being
hard-coded in this file:

  export LLM_PROXY_KEY_GZ=ak-...
  export LLM_PROXY_KEY_FSY=ak-...
  export LLM_PROXY_KEY_BYK=ak-...
  export LLM_PROXY_KEY_LSS=ak-...
  export LLM_PROXY_KEY_ZXF=ak-...
  export LLM_PROXY_KEY_HCJ=ak-...
"""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI


DEFAULT_BASE_URL = "https://core42.primus-safe.amd.com/api/v1/llm-proxy/v1"
DEFAULT_MODEL = "claude-opus-4-7"
DEFAULT_KEY_ENV_NAMES = (
    ("gz", "LLM_PROXY_KEY_GZ"),
    ("fsy", "LLM_PROXY_KEY_FSY"),
    ("byk", "LLM_PROXY_KEY_BYK"),
    ("lss", "LLM_PROXY_KEY_LSS"),
    ("zxf", "LLM_PROXY_KEY_ZXF"),
    ("hcj", "LLM_PROXY_KEY_HCJ"),
)


@dataclass(frozen=True)
class BatchConfig:
    base_url: str
    model: str
    max_concurrency: int
    max_retries: int
    backoff_cap_s: float
    max_tokens: int
    timeout_s: float
    verify_tls: bool


class GlobalCooldownGate:
    """Shared 429 cooldown gate for all worker threads."""

    def __init__(self) -> None:
        self._until = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        while True:
            with self._lock:
                wait_s = self._until - time.time()
            if wait_s <= 0:
                return
            time.sleep(min(wait_s, 5.0))

    def trip(self, seconds: float) -> None:
        seconds = max(0.0, seconds)
        with self._lock:
            self._until = max(self._until, time.time() + seconds)


def build_tasks() -> list[list[dict[str, str]]]:
    """Replace this with the real task builder.

    Each item must be a normal OpenAI chat ``messages`` list.
    """

    return [
        [{"role": "user", "content": f"task {i}: say hi"}]
        for i in range(300)
    ]


def _load_keys() -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    for name, env_name in DEFAULT_KEY_ENV_NAMES:
        value = (os.environ.get(env_name) or "").strip()
        if value:
            keys.append((name, value))
    if keys:
        return keys

    raw = (os.environ.get("LLM_PROXY_KEYS") or "").strip()
    if raw:
        for idx, item in enumerate(raw.split(","), start=1):
            key = item.strip()
            if key:
                keys.append((f"key{idx}", key))
    return keys


def _retry_after_seconds(exc: APIStatusError, attempt: int, cap_s: float) -> float:
    try:
        header = exc.response.headers.get("retry-after")
        delay = float(header) if header is not None else None
    except (TypeError, ValueError):
        delay = None
    if delay is None:
        delay = min(2 ** attempt, cap_s) + random.random()
    return min(max(delay, 0.0), cap_s)


def _network_backoff_seconds(attempt: int, cap_s: float) -> float:
    return min(2 ** attempt, cap_s) + random.random()


def _call_one(
    idx: int,
    messages: list[dict[str, str]],
    clients: list[tuple[str, OpenAI]],
    sem: threading.Semaphore,
    gate: GlobalCooldownGate,
    cfg: BatchConfig,
) -> str:
    name, client = clients[idx % len(clients)]
    last_err: BaseException | None = None

    for attempt in range(cfg.max_retries):
        gate.wait()
        with sem:
            try:
                resp = client.chat.completions.create(
                    model=cfg.model,
                    messages=messages,
                    max_tokens=cfg.max_tokens,
                )
                return resp.choices[0].message.content or ""
            except APIStatusError as exc:
                last_err = exc
                if exc.status_code != 429:
                    raise
                delay_s = _retry_after_seconds(exc, attempt, cfg.backoff_cap_s)
                gate.trip(delay_s)
            except (APIConnectionError, APITimeoutError) as exc:
                last_err = exc
                delay_s = _network_backoff_seconds(attempt, cfg.backoff_cap_s)
        time.sleep(delay_s)

    raise RuntimeError(
        f"task {idx} (key={name}) failed after {cfg.max_retries} retries: {last_err}"
    )


def run_batch(tasks: list[list[dict[str, str]]], cfg: BatchConfig) -> list[str | None]:
    keys = _load_keys()
    if not keys:
        names = ", ".join(env for _, env in DEFAULT_KEY_ENV_NAMES)
        raise RuntimeError(
            f"no API keys configured; set one or more of {names}, "
            "or set comma-separated LLM_PROXY_KEYS"
        )

    http_client = httpx.Client(verify=cfg.verify_tls, timeout=cfg.timeout_s)
    clients = [
        (name, OpenAI(api_key=key, base_url=cfg.base_url, http_client=http_client))
        for name, key in keys
    ]
    sem = threading.Semaphore(max(1, cfg.max_concurrency))
    gate = GlobalCooldownGate()

    results: list[str | None] = [None] * len(tasks)
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=max(1, cfg.max_concurrency)) as pool:
        futs = {
            pool.submit(_call_one, i, messages, clients, sem, gate, cfg): i
            for i, messages in enumerate(tasks)
        }
        for fut in as_completed(futs):
            idx = futs[fut]
            try:
                results[idx] = fut.result()
                ok += 1
            except Exception as exc:  # noqa: BLE001 - batch runner should continue.
                results[idx] = None
                fail += 1
                print(f"[FAIL] task {idx}: {exc}", flush=True)

    print(f"\ndone: OK={ok}  FAIL={fail}  total={len(tasks)}", flush=True)
    http_client.close()
    return results


def _write_jsonl(path: Path, tasks: list[Any], results: list[str | None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for idx, (messages, result) in enumerate(zip(tasks, results)):
            f.write(json.dumps({
                "idx": idx,
                "messages": messages,
                "ok": result is not None,
                "result": result,
            }, ensure_ascii=False) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("LLM_PROXY_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=os.environ.get("LLM_PROXY_MODEL", DEFAULT_MODEL))
    parser.add_argument("--max-concurrency", type=int, default=int(os.environ.get("LLM_PROXY_CONCURRENCY", "15")))
    parser.add_argument("--max-retries", type=int, default=int(os.environ.get("LLM_PROXY_MAX_RETRIES", "10")))
    parser.add_argument("--backoff-cap-s", type=float, default=float(os.environ.get("LLM_PROXY_BACKOFF_CAP_S", "90")))
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("LLM_PROXY_MAX_TOKENS", "1024")))
    parser.add_argument("--timeout-s", type=float, default=float(os.environ.get("LLM_PROXY_TIMEOUT_S", "120")))
    parser.add_argument("--verify-tls", action="store_true", help="Enable TLS verification. Default keeps compatibility with the current gateway.")
    parser.add_argument("--output", default="", help="Optional JSONL output path.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    cfg = BatchConfig(
        base_url=args.base_url,
        model=args.model,
        max_concurrency=args.max_concurrency,
        max_retries=args.max_retries,
        backoff_cap_s=args.backoff_cap_s,
        max_tokens=args.max_tokens,
        timeout_s=args.timeout_s,
        verify_tls=args.verify_tls,
    )
    tasks = build_tasks()
    results = run_batch(tasks, cfg)
    if args.output:
        _write_jsonl(Path(args.output), tasks, results)
    return 0 if all(item is not None for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
