# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A run that never got a GPU is reported as no capacity, not as a failure of the change.

Both e2e scripts run against a stub `dispatron-ci` that reports a timeout and a stub
`curl` that records the commit statuses posted.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = {
    "ci-e2e": _ROOT / ".github" / "scripts" / "ci-e2e-dispatch.sh",
    "forge": _ROOT / ".github" / "scripts" / "forge-ci-e2e-dispatch.sh",
}

_STUB_CLI = """#!/usr/bin/env bash
printf '%s\\n' "$@" > "$STUB_DIR/cli-args"
printf '{"event":"submitted","uid":"u-1"}\\n' >> "$DISPATCH_EVENTS_FILE"
printf '{"event":"terminal","uid":"u-1","result":"timeout","phase":"Queued",'\\
'"timed_out_in":"%s","explanation":"stub"}\\n' "$STUB_TIMED_OUT_IN" >> "$DISPATCH_EVENTS_FILE"
exit 1
"""

_STUB_CURL = """#!/usr/bin/env bash
while [ $# -gt 0 ]; do
  case "$1" in
    -d|--data) printf '%s' "$2" | jq -c . >> "$STUB_DIR/statuses"; shift 2 ;;
    *) shift ;;
  esac
done
printf '201'
"""

pytestmark = pytest.mark.skipif(shutil.which("jq") is None, reason="the e2e scripts need jq")


def _stub(directory: Path, name: str, body: str) -> None:
    path = directory / name
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _run(script: str, timed_out_in: str, tmp_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """The CLI's arguments and the statuses posted, for a run that timed out."""
    _stub(tmp_path, "dispatron-ci", _STUB_CLI)
    _stub(tmp_path, "curl", _STUB_CURL)
    env = {
        **os.environ,
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        "STUB_DIR": str(tmp_path),
        "STUB_TIMED_OUT_IN": timed_out_in,
        "DISPATRON_BASE_URL": "http://dispatron.test",
        "HEAD_REF": "feature",
        "HEAD_SHA": "0123456789abcdef0123456789abcdef01234567",
        "GH_STATUS_TOKEN": "t",
        "GH_STATUS_REPO": "org/repo",
        "GH_STATUS_SHA": "0123456789abcdef0123456789abcdef01234567",
        "RUNNER_TEMP": str(tmp_path),
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"),
        "QUEUE_TIMEOUT_S": "14400",
    }
    env.pop("PR_NUMBER", None)
    env.pop("GITHUB_OUTPUT", None)
    subprocess.run(["bash", str(_SCRIPTS[script])], cwd=_ROOT, env=env, check=False, timeout=60)
    args = (tmp_path / "cli-args").read_text(encoding="utf-8").splitlines()
    statuses = [json.loads(line) for line in (tmp_path / "statuses").read_text(encoding="utf-8").splitlines()]
    return args, statuses


@pytest.mark.parametrize("script", sorted(_SCRIPTS))
def test_a_run_that_never_got_a_gpu_is_an_error_saying_nothing_ran(script: str, tmp_path: Path) -> None:
    args, statuses = _run(script, "queue", tmp_path)
    final = statuses[-1]
    assert final["state"] == "error"
    assert "no GPU within 240m; nothing ran" in final["description"]
    assert "--queue-timeout" in args and "--poll-max" not in args


@pytest.mark.parametrize("script", sorted(_SCRIPTS))
def test_a_run_that_started_and_overran_is_still_a_failure(script: str, tmp_path: Path) -> None:
    _, statuses = _run(script, "run", tmp_path)
    final = statuses[-1]
    assert final["state"] == "failure"
    assert final["description"].startswith("timeout;")


def test_the_hyperloom_run_clock_covers_the_agent_setting_up(tmp_path: Path) -> None:
    """The run is seen running once its sandbox is ready, and the agent sets up before
    the optimizer's MAX_HOURS starts counting."""
    args, _ = _run("ci-e2e", "run", tmp_path)
    run_timeout = float(args[args.index("--run-timeout") + 1])
    assert run_timeout >= 0.5 * 3600 + 3600
