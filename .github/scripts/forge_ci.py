#!/usr/bin/env python3
"""Kernel Arena forge-run helper for the ``forge-kernel-bench`` CI workflow.

This is the runtime driver behind the forge regression suite: it triggers a
long-running (24h) *forge* agent run for one benchmark on the Kernel Arena
controller and, optionally, polls it to a terminal state and reports the
resulting speedup / score.

Design notes
------------
* **Standard library only** (``urllib``/``json``/``ssl``). The self-hosted
  project1 runner needs no ``pip install`` to execute this.
* **Auth**: SaFE API key (``ak-...``) sent as ``Authorization: Bearer <key>``.
  The key's user must own the benchmark (or be a system-admin), otherwise the
  controller rejects ``POST /v1/runs`` with 403. Supplied via ``KA_API_KEY``.
* **Network**: the controller API is only reachable from inside the project1
  network (higress ingress ``project1.tw325.primus-safe.amd.com``); this is why
  the workflow job runs on a project1 self-hosted runner. Base URL via
  ``KA_API_BASE``.

Sub-commands
------------
``matrix``
    Parse the ``KA_BENCHMARK_IDS`` secret into a GitHub Actions ``matrix`` value
    so the number/identity of kernels is driven entirely by the secret (add or
    remove a benchmark_id there and the fan-out follows).

``run``
    Trigger one forge run (``agent_template=forge``) and optionally poll it to
    completion, emitting ``run_id`` / ``status`` / ``speedup`` / ``total_score``
    as step outputs and a Markdown row in the job summary.
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

# Run states that will never change again (see controller src/types.ts RunStatus).
TERMINAL_STATES = {"done", "failed", "timed_out", "stopped"}
# The only terminal state we treat as a passing run.
SUCCESS_STATE = "done"


# --------------------------------------------------------------------------- #
# GitHub Actions output helpers
# --------------------------------------------------------------------------- #
def _gh_output(pairs: Dict[str, str]) -> None:
    """Append ``key=value`` step outputs (no-op when not on a runner)."""
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as fh:
        for key, value in pairs.items():
            fh.write(f"{key}={value}\n")


def _gh_summary(markdown: str) -> None:
    """Append a block to the job summary (no-op when not on a runner)."""
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(markdown.rstrip() + "\n")


def _annotate(level: str, message: str) -> None:
    """Emit a GitHub Actions ``::error``/``::warning`` annotation + plain line."""
    print(f"::{level}::{message}")
    print(f"[{level.upper()}] {message}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
def _http(
    method: str,
    url: str,
    api_key: str,
    *,
    body: Optional[dict] = None,
    insecure: bool = False,
    timeout: float = 30.0,
) -> Tuple[int, Any]:
    """Perform a JSON HTTP request; return ``(status_code, parsed_body_or_text)``."""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    ctx = ssl.create_default_context()
    if insecure:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            raw = resp.read().decode("utf-8", "replace")
            return resp.status, _maybe_json(raw)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        return exc.code, _maybe_json(raw)


def _maybe_json(raw: str) -> Any:
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return raw


# --------------------------------------------------------------------------- #
# Response parsing (tolerant of flat vs nested shapes)
# --------------------------------------------------------------------------- #
def _dig(obj: Any, *keys: str) -> Any:
    """Return the first non-None value found for any of ``keys``, searching the
    top level and a nested ``run``/``score`` object."""
    if not isinstance(obj, dict):
        return None
    scopes = [obj]
    for nested in ("run", "score", "detail"):
        if isinstance(obj.get(nested), dict):
            scopes.append(obj[nested])
    for scope in scopes:
        for key in keys:
            if scope.get(key) is not None:
                return scope[key]
    return None


def _extract_status(detail: Any) -> Optional[str]:
    status = _dig(detail, "status")
    return str(status) if status is not None else None


def _extract_metrics(detail: Any) -> Dict[str, Optional[float]]:
    return {
        "speedup": _num(_dig(detail, "speedup", "speedup_ratio")),
        "total_score": _num(_dig(detail, "total_score")),
        "authoring_status": _dig(detail, "authoring_status"),
    }


def _num(value: Any) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (ValueError, TypeError):
        return None


# --------------------------------------------------------------------------- #
# matrix sub-command
# --------------------------------------------------------------------------- #
def parse_benchmarks(raw: str) -> List[Dict[str, str]]:
    """Parse ``KA_BENCHMARK_IDS`` into ``[{name, benchmark_id}, ...]``.

    Accepted formats (whichever is most convenient to store as a secret):
      * JSON array : ``[{"name":"softmax_kernel","benchmark_id":"kb_..."}, ...]``
      * JSON object: ``{"softmax_kernel":"kb_...", "rmsnorm_kernel":"kb_..."}``
      * CSV        : ``softmax_kernel=kb_...,rmsnorm_kernel=kb_...`` (name=id pairs,
                     comma/newline separated). A bare ``kb_...`` with no ``name=``
                     is kept with its id as the name.
    """
    raw = (raw or "").strip()
    if not raw:
        return []
    items: List[Dict[str, str]] = []
    if raw[0] in "[{":
        doc = json.loads(raw)
        if isinstance(doc, dict):
            items = [{"name": k, "benchmark_id": v} for k, v in doc.items()]
        elif isinstance(doc, list):
            for entry in doc:
                if isinstance(entry, str):
                    items.append({"name": entry, "benchmark_id": entry})
                elif isinstance(entry, dict):
                    bid = entry.get("benchmark_id") or entry.get("id")
                    if bid:
                        items.append({"name": entry.get("name") or bid, "benchmark_id": bid})
    else:
        for token in raw.replace("\n", ",").split(","):
            token = token.strip()
            if not token:
                continue
            if "=" in token:
                name, bid = token.split("=", 1)
                items.append({"name": name.strip(), "benchmark_id": bid.strip()})
            else:
                items.append({"name": token, "benchmark_id": token})
    return [it for it in items if it.get("benchmark_id")]


def cmd_matrix(args: argparse.Namespace) -> int:
    raw = os.environ.get("KA_BENCHMARK_IDS", "")
    try:
        benchmarks = parse_benchmarks(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        _annotate("error", f"KA_BENCHMARK_IDS is not valid: {exc}")
        return 1
    if not benchmarks:
        _annotate("error", "KA_BENCHMARK_IDS is empty - set it to the flydsl benchmark ids")
        return 1
    matrix = {"include": benchmarks}
    _gh_output({"matrix": json.dumps(matrix)})
    print(f"Resolved {len(benchmarks)} benchmark(s) from KA_BENCHMARK_IDS:")
    for it in benchmarks:
        print(f"  - {it['name']}: {it['benchmark_id']}")
    return 0


# --------------------------------------------------------------------------- #
# run sub-command
# --------------------------------------------------------------------------- #
def trigger_run(args: argparse.Namespace) -> Optional[str]:
    body: Dict[str, Any] = {
        "benchmark_id": args.benchmark_id,
        "agent_template": args.agent_template,
        "model": args.model,
        "gpu_model": args.gpu_model,
        "max_iterations": args.max_iterations,
        "timeout_seconds": args.timeout_seconds,
    }
    if args.aka_ref:
        body["aka_ref"] = args.aka_ref
    url = f"{args.api_base.rstrip('/')}/v1/runs"
    code, payload = _http("POST", url, args.api_key, body=body, insecure=args.insecure)
    if code not in (200, 201, 202):
        _annotate("error", f"POST /v1/runs -> HTTP {code}: {payload}")
        return None
    run_id = _dig(payload, "run_id") or (payload.get("run_id") if isinstance(payload, dict) else None)
    if not run_id:
        _annotate("error", f"POST /v1/runs succeeded ({code}) but no run_id in response: {payload}")
        return None
    print(
        f"Triggered forge run for '{args.name}': run_id={run_id} "
        f"(model={args.model}, max_iterations={args.max_iterations}, "
        f"timeout_seconds={args.timeout_seconds})"
    )
    return str(run_id)


def poll_run(args: argparse.Namespace, run_id: str) -> Tuple[str, Dict[str, Optional[float]]]:
    """Poll ``/v1/runs/<id>`` until terminal or ``--poll-timeout`` elapses."""
    url = f"{args.api_base.rstrip('/')}/v1/runs/{run_id}"
    deadline = time.monotonic() + args.poll_timeout
    status = "unknown"
    metrics: Dict[str, Optional[float]] = {"speedup": None, "total_score": None, "authoring_status": None}
    last_status: Optional[str] = None
    while True:
        code, payload = _http("GET", url, args.api_key, insecure=args.insecure)
        if code == 200:
            status = _extract_status(payload) or "unknown"
            metrics = _extract_metrics(payload)
            if status != last_status:
                print(f"[{_now()}] run {run_id} status={status}")
                last_status = status
            if status in TERMINAL_STATES:
                return status, metrics
        else:
            _annotate("warning", f"GET /v1/runs/{run_id} -> HTTP {code}: {payload}")
        if time.monotonic() >= deadline:
            _annotate(
                "warning",
                f"poll timeout after {args.poll_timeout}s; run {run_id} still '{status}' "
                f"(the run keeps executing on the controller)",
            )
            return status, metrics
        time.sleep(args.poll_interval)


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())


def cmd_run(args: argparse.Namespace) -> int:
    if not args.api_base or not args.api_key:
        _annotate("error", "KA_API_BASE and KA_API_KEY must be set (repo/environment secrets)")
        return 1

    run_id = trigger_run(args)
    if not run_id:
        return 1
    _gh_output({"run_id": run_id})

    if not args.poll:
        _gh_summary(f"| `{args.name}` | {run_id} | triggered (not polled) | n/a | n/a |")
        print("Polling disabled (--no-poll); run continues asynchronously on the controller.")
        return 0

    status, metrics = poll_run(args, run_id)
    speedup = metrics.get("speedup")
    total = metrics.get("total_score")
    authoring = metrics.get("authoring_status")
    _gh_output(
        {
            "status": status,
            "speedup": "" if speedup is None else f"{speedup:.4f}",
            "total_score": "" if total is None else f"{total:.4f}",
        }
    )
    speedup_str = "n/a" if speedup is None else f"{speedup:.2f}x"
    total_str = "n/a" if total is None else f"{total:.2f}"
    _gh_summary(f"| `{args.name}` | {run_id} | {status} | {speedup_str} | {total_str} |")

    ok = status == SUCCESS_STATE
    print(
        f"Run {run_id} finished: status={status} speedup={speedup_str} "
        f"total_score={total_str} authoring_status={authoring}"
    )
    if not ok and args.fail_on_run_failure:
        _annotate("error", f"forge run for '{args.name}' ended in non-success state '{status}'")
        return 1
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Kernel Arena forge-run CI helper")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("matrix", help="Emit a GitHub Actions matrix from KA_BENCHMARK_IDS")

    r = sub.add_parser("run", help="Trigger (and optionally poll) one forge run")
    r.add_argument("--benchmark-id", required=True)
    r.add_argument("--name", default="", help="Human label for logs/summary")
    r.add_argument("--api-base", default=os.environ.get("KA_API_BASE", ""))
    r.add_argument("--api-key", default=os.environ.get("KA_API_KEY", ""))
    r.add_argument("--agent-template", default="forge")
    r.add_argument("--model", default="claude-opus-4-8")
    r.add_argument("--gpu-model", default="MI325X")
    r.add_argument("--max-iterations", type=int, default=720)
    r.add_argument("--timeout-seconds", type=int, default=86400)
    r.add_argument("--aka-ref", default="", help="AKA git ref for kernel defs (empty = controller default)")
    r.add_argument("--poll", dest="poll", action="store_true", default=True)
    r.add_argument("--no-poll", dest="poll", action="store_false")
    r.add_argument("--poll-interval", type=float, default=300.0, help="seconds between polls")
    r.add_argument("--poll-timeout", type=float, default=93600.0, help="max seconds to poll (~26h)")
    r.add_argument("--fail-on-run-failure", dest="fail_on_run_failure", action="store_true", default=True)
    r.add_argument("--no-fail-on-run-failure", dest="fail_on_run_failure", action="store_false")
    r.add_argument("--insecure", action="store_true", help="skip TLS verification (self-signed ingress)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "matrix":
        return cmd_matrix(args)
    if args.command == "run":
        return cmd_run(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
