#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Normalize and publish a Hyperloom result directory.

This is the one-shot end-of-run helper used by Web/skill flows:

  result artifacts -> normalized_results.ndjson -> results service

Failures are non-fatal by default so result publishing never invalidates an
optimization run. Pass --strict when a caller wants publish failures to fail.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from artifact_normalizer import normalize_task_result, write_single_result  # noqa: E402
from publish_results import publish  # noqa: E402

DEFAULT_SERVICE_URL = "http://hyperloom-results-service.primus-claw-dev.svc.cluster.local"


def _default_model(task_dir: Path) -> str:
    env_model = os.environ.get("MODEL_NAME") or os.environ.get("MODEL")
    if env_model:
        return env_model.rstrip("/").split("/")[-1]
    run_context = task_dir / "results" / "run_context.env"
    if run_context.exists():
        for line in run_context.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("MODEL="):
                return line.split("=", 1)[1].rstrip("/").split("/")[-1]
    return "unknown"


def _task_id(model: str) -> str:
    explicit = (
        os.environ.get("HYPERLOOM_TASK_ID")
        or os.environ.get("SAFE_TASK_ID")
        or os.environ.get("CLAW_SESSION_ID")
    )
    if explicit:
        return explicit
    safe_model = "".join(ch.lower() if ch.isalnum() else "-" for ch in model).strip("-")
    return f"web-{safe_model or 'task'}-{int(time.time())}"


def _host_reachable(url: str) -> tuple[bool, str]:
    host = url.split("://", 1)[-1].split("/", 1)[0].split(":", 1)[0]
    try:
        socket.gethostbyname(host)
        return True, ""
    except OSError as e:
        return False, str(e)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", default=os.environ.get("HYPERLOOM_RESULT_DIR") or os.environ.get("USER_DATA_PATH") or "/workspace/hyperloom")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--display-name", default="")
    parser.add_argument("--task-id", default="")
    parser.add_argument(
        "--url",
        default=os.environ.get("HYPERLOOM_RESULTS_SERVICE_URL", DEFAULT_SERVICE_URL),
        help="Results service base URL",
    )
    parser.add_argument("--token", default=os.environ.get("HYPERLOOM_RESULTS_SERVICE_TOKEN", ""))
    parser.add_argument("--timeout", type=int, default=int(os.environ.get("HYPERLOOM_RESULTS_TIMEOUT", "20")))
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when publish fails")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    task_dir = Path(args.task_dir)
    out_dir = Path(args.out_dir) if args.out_dir else task_dir / "normalized"
    model = args.model or _default_model(task_dir)
    task_id = args.task_id or _task_id(model)
    display_name = args.display_name or task_id

    result: dict[str, Any] = {
        "task_id": task_id,
        "model": model,
        "display_name": display_name,
        "published": False,
        "normalized_dir": str(out_dir),
    }

    try:
        normalized = normalize_task_result(
            task_dir,
            {
                "task_id": task_id,
                "model": model,
                "display_name": display_name,
                "status": "imported",
                "final_status": "Succeeded",
            },
            {"source": "hyperloom-auto-publish"},
        )
        write_single_result(normalized, out_dir)
        result["normalizer_exit_code"] = 0
        result["warnings"] = normalized.get("warnings") or []

        reachable, reason = _host_reachable(args.url)
        if not reachable:
            result["publish_error"] = f"results service host not resolvable: {reason}"
            print(json.dumps(result, indent=2))
            return 2 if args.strict else 0

        response = publish([normalized], args.url, args.token, args.timeout)
        result["published"] = True
        result["publish_response"] = response
        print(json.dumps(result, indent=2))
        return 0
    except Exception as e:
        result["error"] = str(e)
        print(json.dumps(result, indent=2))
        return 1 if args.strict else 0


if __name__ == "__main__":
    sys.exit(main())
