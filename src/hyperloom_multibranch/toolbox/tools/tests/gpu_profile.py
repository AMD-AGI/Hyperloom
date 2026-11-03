# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GPU check of the profile tool: one real profiling round through the tool runner, result read from the inbox push.

Run with ``HL_PROFILE_TEST=1`` on a card the caller owns, with ``HL_PROFILE_MODEL`` a model directory and
``HL_PROFILE_INFERENCEX`` an InferenceX checkout.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from hyperloom_multibranch.toolbox.tools import JobStatus, ToolJobs, ToolRequest, profile_tool
from hyperloom_multibranch.toolbox.tools.tests.sinks import RecordingSink

MODEL = os.environ.get("HL_PROFILE_MODEL", "")
INFERENCEX = os.environ.get("HL_PROFILE_INFERENCEX", "")
pytestmark = pytest.mark.skipif(
    os.environ.get("HL_PROFILE_TEST") != "1" or not MODEL or not INFERENCEX,
    reason="GPU run; set HL_PROFILE_TEST=1, HL_PROFILE_MODEL and HL_PROFILE_INFERENCEX",
)


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("HYPERLOOM_BENCHMARK_BACKEND", "bypass")
    monkeypatch.setenv("INFERENCEX_PATH", INFERENCEX)
    monkeypatch.delenv("MAGPIE_PATH", raising=False)
    monkeypatch.setenv("FRAMEWORK", "sglang")
    monkeypatch.setenv("TP", "1")
    monkeypatch.setenv("MODEL_PATH", MODEL)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_LEAK_ROOTS", str(tmp_path / "leak"))
    monkeypatch.setenv("INFERENCE_OPTIMIZER_BASELINE_SERVER_READY_SEC", "900")


async def test_profile_round_reaches_the_inbox(tmp_path: Path) -> None:
    out = Path(os.environ.get("HL_PROFILE_OUT", str(tmp_path / "out")))
    sink = RecordingSink()
    runner = ToolJobs([profile_tool()], sink, out / "results", push_delay_seconds=0)
    params = {
        "model_path": MODEL,
        "extra_envs": {"CONC": "4", "ISL": "256", "OSL": "128", "NUM_PROMPTS": "16", "NUM_WARMUPS": "2"},
    }
    job_id = runner.submit(ToolRequest("profile", "s1", params))
    status = await runner.wait(job_id)
    (pointer,) = sink.pointers
    record = json.loads(Path(pointer.location).read_text())
    print(
        "W11PROFILE "
        + json.dumps({"status": status.value, "summary": pointer.summary, "keys": sorted(record["result"] or {})})
    )
    assert status is JobStatus.SUCCEEDED, pointer.summary
    assert record["result"]["trace_files"]
    assert Path(record["result"]["main_trace_path"]).is_file()
