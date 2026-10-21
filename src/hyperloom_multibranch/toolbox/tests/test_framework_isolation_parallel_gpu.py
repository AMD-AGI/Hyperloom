# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Eight tests at once on an eight-card node, through the real toolbox (W86).

The real aiter first: the start-up hook gives a process that imports it a merge directory of its own, and what aiter
keeps in ``/tmp/aiter_configs`` is not written. Then nine ``explore`` tests (TP1, one card each) with distinct small
deltas, four of them giving ``AITER_CONFIG_GEMM_BF16`` a table of their own; the ninth waits for a card, and one of the
first eight is withdrawn while it runs. What must hold: eight disjoint cards and eight distinct ports with no
``port_in_use``, every test's ``framework_base.json`` on the session's base with a merge directory below its own
sandbox, the shared tables and ``/tmp/aiter_configs`` as they were, the withdrawn test's card taken by the ninth test
within seconds, and every other test a valid measurement. What was measured is printed (``PROBE``).

Needs what ``test_toolbox_kinds_gpu`` needs, with ``HL_W13_CARDS`` naming eight cards, and the image's aiter tables at
``/sgl-workspace``; anywhere else the module is skipped.
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from hyperloom.contracts.measurement import TestResult, Validity
from hyperloom.contracts.recipe import ConfigDelta, RecipeDelta
from hyperloom.measure.framework_base import DEFAULT_CONFIGS, FRAMEWORK_BASE_FILE, load_snapshot, tree_files
from hyperloom.measure.framework_merge import PRIVATE, MergeView
from hyperloom_multibranch.contracts.toolbox import CancelOutcome
from hyperloom_multibranch.toolbox.tests.helpers import req
from hyperloom_multibranch.toolbox.tests.test_framework_isolation_gpu import _landed
from hyperloom_multibranch.toolbox.tests.test_toolbox_kinds_gpu import (  # noqa: F401  (workload_environment is a fixture)
    CARDS,
    _probe,
    _run_root,
    _toolbox,
    pytestmark,
    workload_environment,
)

VARIABLE = "AITER_CONFIG_GEMM_BF16"
#: Without it SGLang never imports ``aiter.tuned_gemm``, the module that reads the table, and a Qwen3-8B server merges none.
USE_AITER = "SGLANG_USE_AITER"
SHARED = Path(DEFAULT_CONFIGS) / "bf16_tuned_gemm.csv"
AITER_MERGE_DIR = Path("/tmp/aiter_configs")
TESTS = 9
WITHDRAWN = 5
OWN_TABLE = (0, 1, 2, 3)
WITHDRAW_AFTER_SEC = 90
NEXT_TEST_WITHIN_SEC = 20
AITER_CORE = Path(DEFAULT_CONFIGS).parent / "jit" / "core.py"


def _installed_aiter_names_the_variable() -> bool:
    """Whether the installed aiter has the ``AITER_CONFIG_GEMM_BF16_FILE`` property this module reads the table by."""
    return AITER_CORE.is_file() and f"def {VARIABLE}_FILE" in AITER_CORE.read_text(encoding="utf-8")


pytestmark = [
    pytestmark,
    pytest.mark.skipif(not SHARED.is_file(), reason=f"needs the image's aiter table {SHARED}"),
    pytest.mark.skipif(not _installed_aiter_names_the_variable(), reason=f"the installed aiter has no {VARIABLE}_FILE"),
]


def _table_with_a_shape_of_its_own(path: Path, index: int) -> Path:
    """A one-row table in the shared table's format: its first row with an ``M`` no model uses."""
    with SHARED.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    header, row = rows[0], rows[1]
    row[header.index("M")] = str(100_003 + index)
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows([header, row])
    return path


def _request(index: int, root: Path):
    envs = {USE_AITER: "1"}
    if index in OWN_TABLE:
        own = _table_with_a_shape_of_its_own(root / f"own-{index}.csv", index)
        envs[VARIABLE] = f"{SHARED}{os.pathsep}{own}"
    delta = RecipeDelta(config=ConfigDelta(server_args=f"--max-running-requests {64 + 16 * index}", envs=envs))
    return req(f"b{index}", change=delta)


def test_the_hook_gives_the_real_aiter_a_merge_directory_of_its_own(tmp_path: Path) -> None:
    before = tree_files(AITER_MERGE_DIR)
    own = _table_with_a_shape_of_its_own(tmp_path / "own.csv", 0)
    copy = shutil.copyfile(SHARED, tmp_path / "bf16_tuned_gemm.csv")
    view = MergeView.make(tmp_path / "private")
    environment = {
        **os.environ,
        VARIABLE: f"{copy}{os.pathsep}{own}",
        **view.environment(os.environ.get("PYTHONPATH", "")),
    }
    code = f"from aiter.jit.core import AITER_CONFIGS; print(AITER_CONFIGS.{VARIABLE}_FILE)"

    done = subprocess.run([sys.executable, "-c", code], env=environment, capture_output=True, text=True, timeout=600)

    assert done.returncode == 0, done.stderr[-2000:]
    merged = Path(done.stdout.strip().splitlines()[-1])
    report = view.report()
    _probe("real-aiter", merged=str(merged), report=report, rows=len(merged.read_text().splitlines()))
    assert merged == view.directory / "bf16_tuned_gemm.csv" and merged.is_file()
    assert len(merged.read_text().splitlines()) == len(Path(copy).read_text().splitlines()) + 1, (
        "shared rows and its own"
    )
    assert tree_files(AITER_MERGE_DIR) == before, "what aiter keeps in /tmp/aiter_configs is as it was"
    assert report["merge_isolation"] == PRIVATE and report["merge_patched_processes"] == 1


async def _started(toolbox, request_ids: list[str]) -> float:
    """Seconds from now to the moment all of the first eight run and the one to be withdrawn has run long enough."""
    began = time.monotonic()
    while True:
        progress = [await toolbox.progress(rid) for rid in request_ids[:8]]
        running = [p is not None and p.state == "running" for p in progress]
        mine = progress[WITHDRAWN]
        if all(running) and mine is not None and (mine.running_seconds or 0) >= WITHDRAW_AFTER_SEC:
            return time.monotonic() - began
        await asyncio.sleep(2)


async def _withdraw_and_watch(toolbox, request_ids: list[str]) -> dict[str, float]:
    """Withdraw one running test; say how long until the withdrawal returned and the ninth test had a sandbox."""
    waiting = request_ids[8]
    assert (await toolbox.progress(waiting)).state == "queued", "the ninth test waits for a card"
    asked = time.monotonic()
    outcome = await toolbox.cancel(request_ids[WITHDRAWN])
    returned = time.monotonic()
    assert outcome is CancelOutcome.WITHDRAWN
    while (record := toolbox.backend.journal.latest(waiting)) is None or not record.sandbox_id:
        assert time.monotonic() - asked < 120, "the ninth test never got a sandbox"
        await asyncio.sleep(0.1)
    return {"withdraw_returned_sec": returned - asked, "next_test_has_a_sandbox_sec": time.monotonic() - asked}


def _facts(root: Path, result: TestResult) -> dict[str, object]:
    landed = _landed(root / "session", result.request_id)
    base = json.loads((landed / FRAMEWORK_BASE_FILE).read_text())
    (config_file,) = sorted(landed.rglob("config.yaml"))[:1]
    config = yaml.safe_load(config_file.read_text())["benchmark"]["envs"]
    logs = "\n".join(p.read_text(errors="replace") for p in sorted(landed.rglob("server*.log")))
    return {
        "base": base,
        "port": int(config["PORT"]),
        "card_env": str(config["ROCR_VISIBLE_DEVICES"]),
        "port_in_use": bool(re.search(r"address already in use|port_in_use", logs, re.IGNORECASE)),
        "wall_sec": round(result.finished_at - result.started_at, 1),
    }


async def test_eight_tests_at_once_use_disjoint_cards_and_ports_and_read_their_own_tables() -> None:
    assert len(CARDS) == 8, f"HL_W13_CARDS must name eight cards, it is {CARDS}"
    root = _run_root("w86")
    tables_before, merged_before = tree_files(Path(DEFAULT_CONFIGS)), tree_files(AITER_MERGE_DIR)
    toolbox, _, _ = _toolbox(root, "w86", timeout_sec=3000)
    results: dict[int, TestResult] = {}
    try:
        await toolbox.start()
        began = time.monotonic()
        ids = [await toolbox.submit_test(_request(index, root)) for index in range(TESTS)]
        submitted = time.monotonic() - began
        all_running_after = await _started(toolbox, ids)
        withdrawal = await _withdraw_and_watch(toolbox, ids)
        await asyncio.wait_for(toolbox.drain(), 3300)
        for index in range(TESTS):
            if index != WITHDRAWN:
                results[index] = await toolbox.outcome(ids[index])
    finally:
        await toolbox.close()
    wall = time.monotonic() - began
    facts = {index: _facts(root, result) for index, result in results.items()}
    snapshot = load_snapshot(root / "session" / "framework")
    merged_name = f"{snapshot.variables[VARIABLE].tuned}.csv"
    cards = {index: result.sandbox.gpu_ids for index, result in results.items()}
    _probe(
        "parallel",
        wall_sec=round(wall, 1),
        submitted_sec=round(submitted, 1),
        all_running_after_sec=round(all_running_after, 1),
        withdrawal=withdrawal,
        cards=cards,
        ports={index: f["port"] for index, f in facts.items()},
        card_env={index: f["card_env"] for index, f in facts.items()},
        wall_by_test={index: f["wall_sec"] for index, f in facts.items()},
        validity={index: result.validity.value for index, result in results.items()},
        merge={
            index: {k: f["base"][k] for k in ("merge_isolation", "merge_dir", "merge_patched_processes")}
            for index, f in facts.items()
        },
        merged_gemm_bf16={index: f["base"]["merge_files"].get(merged_name) for index, f in facts.items()},
        merged_names=sorted({name for f in facts.values() for name in f["base"]["merge_files"]}),
        shared_tables_unchanged=tree_files(Path(DEFAULT_CONFIGS)) == tables_before,
        aiter_merge_dir_unchanged=tree_files(AITER_MERGE_DIR) == merged_before,
    )

    assert all(result.validity is Validity.SUCCEEDED for result in results.values()), {
        index: (result.validity, result.failure_class, result.failure_reason) for index, result in results.items()
    }
    assert all(r.rounds and r.rounds[-1].metrics.get("output_throughput", 0) > 0 for r in results.values())
    first_seven = [cards[index][0] for index in results if index < 8]
    assert all(len(card) == 1 for card in cards.values()) and len(set(first_seven)) == 7, "disjoint cards"
    (freed,) = set(CARDS) - set(first_seven)
    assert cards[8] == (freed,), "the ninth test took the card of the withdrawn one"
    assert withdrawal["next_test_has_a_sandbox_sec"] < NEXT_TEST_WITHIN_SEC, withdrawal
    assert len({f["port"] for f in facts.values()}) == len(facts), "distinct ports"
    assert not [index for index, f in facts.items() if f["port_in_use"]]
    assert all(f["card_env"] == str(cards[index][0]) for index, f in facts.items())
    _assert_isolated(facts, results, merged_name)
    assert tree_files(Path(DEFAULT_CONFIGS)) == tables_before, "the shared tables are as they were"
    assert tree_files(AITER_MERGE_DIR) == merged_before, "nothing reached /tmp/aiter_configs"


def _assert_isolated(facts: dict[int, dict[str, object]], results: dict[int, TestResult], merged_name: str) -> None:
    for index, fact in facts.items():
        base = fact["base"]
        assert base["isolation"] == "private_copies" and base["matches_session"] is True
        assert base["framework_base_changed"] is False and base["restored"] == {"start": [], "end": []}
        assert base["shared_sha256_start"] == base["shared_sha256_end"] == base["session_sha256"]
        assert base["merge_isolation"] == PRIVATE and base["merge_patched_processes"] >= 1, base["merge_reason"]
        assert base["merge_chain_failure"] is None
        assert f"/sandboxes/{results[index].sandbox.sandbox_id}/" in base["merge_dir"], "below its own sandbox"
        assert (VARIABLE in base["point_tables"]) == (index in OWN_TABLE)
    assert len({fact["base"]["merge_dir"] for fact in facts.values()}) == len(facts), "one merge directory each"
    own = {facts[index]["base"]["merge_files"].get(merged_name) for index in OWN_TABLE}
    default = {fact["base"]["merge_files"].get(merged_name) for i, fact in facts.items() if i not in OWN_TABLE}
    assert None not in own | default and len(own) == len(OWN_TABLE), "each table of its own, its own merged view"
    assert len(default) == 1 and not default & own, "the default tests share nothing with the others' merges"
