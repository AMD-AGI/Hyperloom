#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Prove a fresh Hyperloom environment can start useful work."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import socket
import ssl
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from hyperloom.common import llm_config
from hyperloom.common.env_safety import redact_secret_values
from hyperloom.orchestrator.roles.agent_role import (
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_CODEX_MODEL,
    default_role_registry,
)
from hyperloom.orchestrator.roles.base import RetryPolicy


@dataclass(frozen=True)
class CheckResult:
    """One bounded cold-start check."""

    name: str
    status: str
    duration_ms: int
    detail: str = ""


def _result(name: str, started: float, *, status: str, detail: str = "") -> CheckResult:
    return CheckResult(
        name=name,
        status=status,
        duration_ms=int((time.perf_counter() - started) * 1000),
        detail=redact_secret_values(detail.strip())[:1000],
    )


def _run_process(name: str, command: list[str], *, timeout: float, cwd: Path | None = None) -> CheckResult:
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return _result(name, started, status="failed", detail=f"{type(exc).__name__}: {exc}")
    output = "\n".join(part.strip() for part in (proc.stdout, proc.stderr) if part.strip())
    tail = "\n".join(output.splitlines()[-12:])
    return _result(
        name,
        started,
        status="passed" if proc.returncode == 0 else "failed",
        detail=f"exit_code={proc.returncode}" + (f"\n{tail}" if tail else ""),
    )


def _check_dotenv(repo_root: Path) -> CheckResult:
    started = time.perf_counter()
    path = repo_root / ".env"
    if not path.is_file():
        return _result(
            "dotenv", started, status="skipped", detail=".env is absent; process environment is authoritative"
        )
    placeholders: list[str] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        low = value.strip().strip("\"'").lower()
        if any(marker in low for marker in ("<your-", "<please_fill_in>", "your-api-key", "your-token-here")):
            placeholders.append(key.removeprefix("export ").strip())
    if placeholders:
        return _result(
            "dotenv",
            started,
            status="failed",
            detail="active placeholder values: " + ", ".join(sorted(placeholders)),
        )
    return _result("dotenv", started, status="passed", detail=str(path))


def _check_framework(framework: str) -> CheckResult:
    started = time.perf_counter()
    if framework not in {"vllm", "sglang", "atom"}:
        return _result(
            "framework_import",
            started,
            status="skipped",
            detail=f"{framework} has no standard import probe",
        )
    try:
        module = importlib.import_module(framework)
    except Exception as exc:
        return _result("framework_import", started, status="failed", detail=f"{type(exc).__name__}: {exc}")
    version = str(getattr(module, "__version__", "") or "unknown")
    return _result("framework_import", started, status="passed", detail=f"{framework}={version}")


def _gateway_url() -> str:
    if llm_config.is_openai_only():
        return (os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").strip()
    if llm_config.has_anthropic_side():
        return (os.environ.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com").strip()
    return ""


def _check_gateway_tls(timeout: float) -> CheckResult:
    started = time.perf_counter()
    url = _gateway_url()
    parsed = urlsplit(url)
    host = parsed.hostname
    if not host:
        if any(
            llm_config.is_truthy(os.environ.get(name)) for name in ("CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX")
        ):
            return _result("gateway_tls", started, status="skipped", detail="managed Anthropic gateway")
        return _result("gateway_tls", started, status="failed", detail="no configured LLM endpoint")
    port = parsed.port or (80 if parsed.scheme == "http" else 443)
    try:
        with socket.create_connection((host, port), timeout=timeout) as raw:
            if parsed.scheme == "https":
                context = ssl.create_default_context()
                with context.wrap_socket(raw, server_hostname=host):
                    pass
            elif parsed.scheme != "http":
                raise ValueError(f"unsupported LLM endpoint scheme: {parsed.scheme or 'missing'}")
    except (OSError, ssl.SSLError, ValueError) as exc:
        hint = ""
        if "CERTIFICATE_VERIFY_FAILED" in str(exc).upper() or "CERTIFICATE VERIFICATION FAILED" in str(exc).upper():
            hint = "; install the gateway CA and set NODE_EXTRA_CA_CERTS for the Node-based agent CLI"
        return _result("gateway_tls", started, status="failed", detail=f"{type(exc).__name__}: {exc}{hint}")
    protocol = "trusted TLS" if parsed.scheme == "https" else "HTTP"
    return _result("gateway_tls", started, status="passed", detail=f"{protocol} endpoint: {host}:{port}")


def _check_experience_kb(require_experience_kb: bool) -> CheckResult:
    started = time.perf_counter()
    from hyperloom.inference_optimizer.experience_v1 import enabled, validate_experience_config

    if not enabled():
        status = "failed" if require_experience_kb else "skipped"
        return _result("experience_kb", started, status=status, detail="HYPERLOOM_KB_URL is not configured")
    try:
        validate_experience_config()
        from hyperloom_kb import experience_kb_from_env

        configured = experience_kb_from_env()
        if not configured.enabled:
            raise RuntimeError("configured Experience KB is disabled")
        remote_client = getattr(configured, "client", None)
        if remote_client is not None:
            health = remote_client.health()
            if str(health.get("status") or "") != "ok":
                raise RuntimeError("Experience KB health check did not return ok")
    except Exception as exc:
        return _result("experience_kb", started, status="failed", detail=f"{type(exc).__name__}: {exc}")
    detail = (
        "Experience KB bootstrap and health check succeeded"
        if getattr(configured, "client", None) is not None
        else "collector bootstrap succeeded"
    )
    return _result("experience_kb", started, status="passed", detail=detail)


async def _check_llm_round_trip(
    *,
    claude_model: str,
    codex_model: str,
    timeout: float,
) -> CheckResult:
    started = time.perf_counter()
    prompt = "Reply with exactly HYPERLOOM_COLD_START_OK."
    backend: Any | None = None
    temporary: tempfile.TemporaryDirectory[str] | None = None
    try:
        if llm_config.is_openai_only():
            from hyperloom.orchestrator.roles.codex import CodexBackend

            temporary = tempfile.TemporaryDirectory(prefix="hyperloom-cold-start-codex-")
            backend = CodexBackend(
                allowed_intents=default_role_registry()["orchestration"].allowed_intents,
                model=codex_model,
                cwd=Path(temporary.name),
                call_timeout_s=timeout,
                retry_policy=RetryPolicy(max_attempts=1),
            )
            result = await asyncio.wait_for(
                backend.run(prompt, system_prompt="Hyperloom cold-start transport check.", allow_no_intent=True),
                timeout=timeout + 5,
            )
        else:
            from hyperloom.orchestrator.roles.claude import ClaudeBackend

            backend = ClaudeBackend(
                model=claude_model,
                raw_completion=True,
                call_timeout_s=timeout,
                retry_policy=RetryPolicy(max_attempts=1),
            )
            result = await asyncio.wait_for(backend.run(prompt, tools=[], max_turns=1), timeout=timeout + 5)
        if not result.raw_text.strip():
            raise RuntimeError("agent backend returned an empty response")
    except Exception as exc:
        return _result("llm_round_trip", started, status="failed", detail=f"{type(exc).__name__}: {exc}")
    finally:
        closer = getattr(backend, "aclose", None)
        if callable(closer):
            await closer()
        if temporary is not None:
            temporary.cleanup()
    return _result("llm_round_trip", started, status="passed", detail="production orchestration transport responded")


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


async def _run(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    repo_root = args.repo_root.expanduser().resolve()
    checks = [
        _check_dotenv(repo_root),
        _run_process(
            "install_check",
            ["bash", str(repo_root / "src/hyperloom/inference_optimizer/assets/install.sh"), "--check-only"],
            timeout=args.install_timeout_sec,
            cwd=repo_root,
        ),
        _run_process(
            "launcher_preflight",
            [
                sys.executable,
                str(repo_root / "src/hyperloom/inference_optimizer/tools/preflight_optimizer.py"),
                str(args.model),
            ],
            timeout=args.check_timeout_sec,
            cwd=repo_root,
        ),
        _check_framework(args.framework),
        _check_experience_kb(args.require_experience_kb),
        _check_gateway_tls(args.check_timeout_sec),
        await _check_llm_round_trip(
            claude_model=args.claude_model,
            codex_model=args.codex_model,
            timeout=args.llm_timeout_sec,
        ),
    ]
    ok = all(item.status in {"passed", "skipped"} for item in checks)
    report = {
        "schema_version": "hyperloom.cold-start-check.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ok": ok,
        "model": str(args.model),
        "framework": args.framework,
        "checks": [asdict(item) for item in checks],
    }
    return (0 if ok else 2), report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--framework", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-experience-kb", action="store_true")
    parser.add_argument("--claude-model", default=os.environ.get("CLAUDE_MODEL") or DEFAULT_CLAUDE_MODEL)
    parser.add_argument("--codex-model", default=os.environ.get("CODEX_MODEL") or DEFAULT_CODEX_MODEL)
    parser.add_argument("--check-timeout-sec", type=float, default=30.0)
    parser.add_argument("--llm-timeout-sec", type=float, default=60.0)
    parser.add_argument("--install-timeout-sec", type=float, default=300.0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    code, report = asyncio.run(_run(args))
    if args.output is not None:
        _write_report(args.output.expanduser().resolve(), report)
    for check in report["checks"]:
        print(f"{check['name']}: {check['status']} ({check['duration_ms']} ms)")
        if check["detail"] and check["status"] != "passed":
            print(f"  {check['detail']}")
    print(f"cold_start_ready={str(report['ok']).lower()}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
