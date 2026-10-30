# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Two sequential accuracy tests whose aiter config lists the image's table and another for one variable (W81b).

aiter merges the tables a variable lists and writes the de-duplicated rows back into each source file; listing the
image's own table made that a write into the image, which every later test then measured. Each test here lists the
image's ``bf16_tuned_gemm.csv`` and a table that holds one of its shapes again with half the time, so any aiter process
that reads the variable rewrites its source. The tests assert what must hold either way: the image's table is the
same file (row count and sha256) after both, and each test's signed ``framework_base.json`` says it began on the
session's base. What the write-back did is printed (``PROBE``): whether the server read the variable at all (a server
that does not has nothing to rewrite) and which files in the test's private copies were rewritten.

Needs what ``test_toolbox_kinds_gpu`` needs (``HL_W13_MODEL``, ``HL_W13_INFERENCEX``, ``HL_W13_OUT``, ``HL_W13_CARDS``)
and the image's aiter tables at ``/sgl-workspace``; anywhere else the module is skipped.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from hyperloom.contracts.measurement import Purpose, TestKind
from hyperloom.contracts.recipe import ConfigDelta, RecipeDelta
from hyperloom.measure import local
from hyperloom.measure.framework.base import DEFAULT_CONFIGS, FRAMEWORK_BASE_FILE, framework_envs
from hyperloom.measure.local import LocalMeasurementBackend
from hyperloom.measure.socket import TestCall
from hyperloom.orchestrator.actions.executors import VariantResult
from hyperloom_multibranch.toolbox.tests.helpers import req
from hyperloom_multibranch.toolbox.tests.test_toolbox_kinds_gpu import (  # noqa: F401  (workload_environment is a fixture)
    _probe,
    _run_root,
    _toolbox,
    pytestmark,
    workload_environment,
)

VARIABLE = "AITER_CONFIG_GEMM_BF16"
SHARED = Path(DEFAULT_CONFIGS) / "bf16_tuned_gemm.csv"
SERVER_SAW = ("duplicate shape", "Updated files", "tuned config")
pytestmark = [
    pytestmark,
    pytest.mark.skipif(not SHARED.is_file(), reason=f"needs the image's aiter table {SHARED}"),
]


def _fingerprint(path: Path) -> tuple[int, str]:
    data = path.read_bytes()
    return len(data.splitlines()), hashlib.sha256(data).hexdigest()


def _table_repeating_a_shape(path: Path, row: int) -> Path:
    """A table of one row: the shared table's ``row``-th shape again, with half its time."""
    with SHARED.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    header, chosen = rows[0], rows[1 + row]
    chosen[header.index("us")] = str(float(chosen[header.index("us")]) / 2)
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows([header, chosen])
    return path


def _landed(session: Path, request_id: str) -> Path:
    (found,) = sorted((session / "tests" / request_id).rglob(FRAMEWORK_BASE_FILE))[-1:]
    return found.parent


async def test_aiter_merging_the_images_table_with_another_leaves_the_images_table_as_it_was() -> None:
    root = _run_root("w81b")
    before = _fingerprint(SHARED)
    toolbox, _, _ = _toolbox(root, "w81b")
    probes = []
    try:
        await toolbox.start()
        for index in range(2):
            other = _table_repeating_a_shape(root / f"other-{index}.csv", row=index)
            delta = RecipeDelta(config=ConfigDelta(envs={VARIABLE: f"{SHARED}:{other}"}))
            request = req(kind=TestKind.ACCURACY, purpose=Purpose.EXPLORE, accuracy=True, change=delta)
            request_id = await toolbox.submit_test(request)
            await asyncio.wait_for(toolbox.drain(), 3300)
            landed = _landed(root / "session", request_id)
            evidence = json.loads((landed / FRAMEWORK_BASE_FILE).read_text())
            logs = "\n".join(p.read_text(errors="replace") for p in sorted(landed.glob("server*.log")))
            probes.append(evidence)
            _probe(
                f"test-{index}",
                request_id=request_id,
                isolation=evidence["isolation"],
                matches_session=evidence["matches_session"],
                framework_base_changed=evidence["framework_base_changed"],
                redirected=evidence["redirected"],
                private_rewritten=evidence["private_rewritten"],
                point_tables=evidence["point_tables"],
                server_log_lines=[line for line in logs.splitlines() if any(s in line for s in SERVER_SAW)][:8],
                shared_now=_fingerprint(SHARED),
            )
    finally:
        await toolbox.close()
    _probe("shared-table", before=before, after=_fingerprint(SHARED))
    assert _fingerprint(SHARED) == before, "the image's table is the file it was"
    for evidence in probes:
        assert evidence["isolation"] == "private_copies" and evidence["matches_session"] is True
        assert evidence["framework_base_changed"] is False and evidence["restored"] == {"start": [], "end": []}
        assert evidence["shared_sha256_start"] == evidence["shared_sha256_end"] == evidence["session_sha256"]
        assert VARIABLE in evidence["point_tables"], "the test's own table list was copied"


# What a server's python does the first time aiter's bf16 GEMM is used: read the variable, merge, write back.
_AITER_CALL = """
import aiter
try:
    import aiter.tuned_gemm
    print('MERGED_WITHOUT_DUPLICATES')
except RuntimeError as exc:
    print('AITER_SAID ' + str(exc).replace(chr(10), ' | '))
"""


class _AiterAsServer:
    """``run_grid`` as a real aiter process: it gets the environment a server would and runs aiter's own merge."""

    def __init__(self) -> None:
        self.said: list[str] = []

    async def __call__(self, **kwargs: Any) -> list[VariantResult]:
        config = yaml.safe_load(Path(kwargs["recipe"].base_yaml_path).read_text())
        env = {**os.environ, **{str(k): str(v) for k, v in config["benchmark"]["envs"].items()}}
        done = subprocess.run([sys.executable, "-c", _AITER_CALL], env=env, capture_output=True, text=True, timeout=600)
        self.said.append(done.stdout + done.stderr)
        workspace = Path(kwargs["output_root"]) / "aiter"
        workspace.mkdir(parents=True, exist_ok=True)
        fields = {"extra_server_args": "", "extra_envs": {}, "status": "succeeded", "returncode": 0}
        return [VariantResult(name="test", workspace=str(workspace), **fields)]


def _aiter_test(root: Path, name: str, table: Path) -> Path:
    point = root / f"{name}.yaml"
    envs = {"TP": 1, "CONC": 4, "RUN_EVAL": "false", VARIABLE: f"{SHARED}:{table}"}
    point.write_text(yaml.safe_dump({"benchmark": {"framework": "sglang", "envs": envs}}), encoding="utf-8")
    call = TestCall(
        points=(point,),
        output_dir=root / "out" / name,
        timeout_sec=900,
        cwd=str(root),
        magpie_python="",
        task_id=name,
    )
    LocalMeasurementBackend().run_test(call)
    return call.output_dir / FRAMEWORK_BASE_FILE


@pytest.mark.skipif(importlib.util.find_spec("aiter") is None, reason="needs aiter importable here")
def test_a_real_aiter_merge_of_the_images_table_with_another_rewrites_only_the_tests_private_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The write-back itself, with aiter's own code and the image's table: where it lands, and what the next test reads."""
    for name, value in framework_envs(tmp_path / "framework").items():
        monkeypatch.setenv(name, value)
    server = _AiterAsServer()
    monkeypatch.setattr(local, "run_grid", server)
    before = _fingerprint(SHARED)
    evidence = [
        json.loads(_aiter_test(tmp_path, name, _table_repeating_a_shape(tmp_path / f"{name}.csv", row=0)).read_text())
        for name in ("first", "second")
    ]
    _probe(
        "aiter-direct",
        said=[text[:600] + " ... " + text[-300:] for text in server.said],
        private_rewritten=[e["private_rewritten"] for e in evidence],
        shared_before=before,
        shared_after=_fingerprint(SHARED),
    )
    assert _fingerprint(SHARED) == before, "the image's table is the file it was"
    rows = before[0] - 1
    for said, base in zip(server.said, evidence, strict=True):
        written = re.search(rf"Updated files: \|\s+(\S+): {rows} -> {rows - 1} rows", said)
        assert written, "aiter's write-back ran, on the table as the session found it"
        assert "hl-framework-" in written.group(1) and not written.group(1).startswith("/sgl-workspace"), written.group(
            1
        )
        assert base["matches_session"] is True and base["framework_base_changed"] is False
        assert any(name.endswith("bf16_tuned_gemm.csv") for name in base["private_rewritten"])
