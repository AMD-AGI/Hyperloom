# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Two tests that run at once on one node do not share aiter's merged tuned tables.

Newer aiter writes the merged view of a variable that lists several tables to ``tempfile.gettempdir()/aiter_configs``.
The runner gives each test's server a temporary directory of its own (``hyperloom.measure.runner.own_tmp``), unless the
environment it inherits already names one. Each test runs in a sandbox of its own, whose runner is a process of its own,
and a local sandbox does not pass the toolbox's ``TMPDIR`` on: so two sandboxes measuring at once on one node, under a
toolbox that has a ``TMPDIR``, each merge into a directory of their own.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.measure.runner.own_tmp import OWN_TMP_PREFIX
from hyperloom.sandbox.tests.test_sandbox_local import _create, _read_when_present, _runner, make  # noqa: F401

#: What a server does with aiter's merge: write its name to ``gettempdir()/aiter_configs/merged.csv`` (only below a
#: directory of the test's own or the toolbox's, never into the machine's ``/tmp``), say where, wait until told to go on,
#: and say what it reads back.
_SERVER = """\
import json, os, pathlib, sys, tempfile, time
name, shared, root = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
own = pathlib.Path(tempfile.gettempdir())
merged = own / "aiter_configs" / "merged.csv"
if own.name.startswith(sys.argv[4]) or own == shared:
    merged.parent.mkdir(exist_ok=True)
    merged.write_text(name)
(root / "out" / "written").write_text(str(own))
while not (root / "in" / "go").exists():
    time.sleep(0.05)
report = {"tmp": str(own), "read": merged.read_text() if merged.is_file() else None}
(root / "out" / "report.tmp").write_text(json.dumps(report))
os.rename(root / "out" / "report.tmp", root / "out" / "report.json")
"""

#: The runner, standing in: one test's server started under ``own_tmpdir``, as ``runner.run`` starts it.
_RUNNER = """\
import os, pathlib, subprocess, sys, time
import yaml
from hyperloom.measure.runner.own_tmp import OWN_TMP_PREFIX, own_tmpdir
from hyperloom.measure.socket import TestCall
root = pathlib.Path(os.environ["HYPERLOOM_SANDBOX_ROOT"])
work = root / "work"
point = work / "point.yaml"
point.write_text(yaml.safe_dump({{"benchmark": {{"envs": {{}}}}}}))
call = TestCall(points=(point,), output_dir=work, timeout_sec=1, cwd=str(work), magpie_python="", task_id="t")
with own_tmpdir(call, os.environ):
    name = os.environ["E2B_SANDBOX_ID"]
    subprocess.run([sys.executable, "-c", {server!r}, name, {shared!r}, str(root), OWN_TMP_PREFIX], check=True)
time.sleep(600)
"""


def test_two_tests_at_once_on_one_node_merge_aiter_tables_into_directories_of_their_own(
    make, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared = tmp_path / "toolbox-tmp"
    shared.mkdir()
    monkeypatch.setenv("TMPDIR", str(shared))  # the toolbox's own, which no test may merge into
    code = _RUNNER.format(server=_SERVER, shared=str(shared))
    provider = make([_runner(code)])
    first, second = _create(provider, "runner", "measurement"), _create(provider, "runner", "measurement")

    written = [_read_when_present(sandbox, "/out/written").decode() for sandbox in (first, second)]
    for sandbox in (first, second):  # both have merged before either reads back
        sandbox.write_file("/in/go", b"")
    reports = [json.loads(_read_when_present(sandbox, "/out/report.json")) for sandbox in (first, second)]

    assert [report["tmp"] for report in reports] == written
    assert all(Path(tmp).name.startswith(OWN_TMP_PREFIX) for tmp in written), written
    assert written[0] != written[1], "each test's server has a temporary directory of its own"
    assert [report["read"] for report in reports] == [first.sandbox_id, second.sandbox_id], "neither is overwritten"
    assert not (shared / "aiter_configs").exists(), "nothing is merged below the toolbox's TMPDIR"
