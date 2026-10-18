# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The real runner flow with only the GPU measurement stubbed, started as a local sandbox template's command.

Everything the runner does around a test is the shipped code: reading the sheet, checking it, preparing the tools, the
rewritten YAML, the rebuilt framework, the inventory, the signed manifest and ``status.json``. Only the call that would
boot an inference server is replaced by :class:`StubBackend`, which writes one report and records what the server would
have been given, so the test can read it back.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from hyperloom.contracts.measurement import SandboxFacts
from hyperloom.measure.runner import run as runner_flow
from hyperloom.measure.socket import RoundOutcome, TestCall, TestOutcome

PROBE = "import os, sys, yaml; print(os.path.exists(os.path.join(os.path.dirname(yaml.__file__), '_hl_marker.py')))"


class StubBackend:
    """Stands in for the local measurement backend: no server, one report, and a record of the point it was given."""

    def run_test(self, call: TestCall) -> TestOutcome:
        workspace = Path(call.output_dir) / "benchmark_stub_20261014_000000"
        workspace.mkdir(parents=True)
        report = {"success": True, "throughput": {"output_throughput": 1234.5, "completed_requests": 8}}
        (workspace / "benchmark_report.json").write_text(json.dumps(report), encoding="utf-8")
        point = yaml.safe_load(Path(call.points[0]).read_text(encoding="utf-8"))
        (Path(call.output_dir) / "point_seen.json").write_text(json.dumps(point), encoding="utf-8")
        envs = (point.get("benchmark") or {}).get("envs") or {}
        probe = subprocess.run(
            [sys.executable, "-c", PROBE],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(envs.get("PYTHONPATH") or "")},
            check=False,
        )
        (Path(call.output_dir) / "import_probe.txt").write_text(probe.stdout.strip(), encoding="utf-8")
        sandbox = SandboxFacts(sandbox_id="stub", backend="local")
        rounds = (RoundOutcome(0, "measure", workspace, 0),)
        return TestOutcome(0, "[]", "", rounds, sandbox, request_id=call.task_id)


def main() -> int:
    root = Path(os.environ["HYPERLOOM_SANDBOX_ROOT"])
    runner_flow.LocalMeasurementBackend = StubBackend  # type: ignore[assignment,misc]
    return runner_flow.run(root / "in", root / "out", scratch=root / "work" / "scratch", spec_wait_sec=60)


if __name__ == "__main__":
    sys.exit(main())
