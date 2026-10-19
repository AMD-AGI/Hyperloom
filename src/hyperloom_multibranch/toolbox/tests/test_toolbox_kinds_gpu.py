# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The real toolbox on a GPU machine: the accuracy test, the concurrency sweep, a patched recipe and a profile (W13).

Each test submits a ``TestRequest`` to a toolbox built by ``build_local_toolbox`` whose planner is the real
``HostedPlanner`` (compile, points, surface check), and reads the result back from the notebook. Needs SGLang, a model
directory and an InferenceX checkout; set ``HL_W13_MODEL``, ``HL_W13_INFERENCEX`` and ``HL_W13_OUT`` (an empty
directory), and optionally ``HL_W13_CARDS`` (default ``6,7``). Anywhere else the module is skipped. The profile test
also needs a TraceLens checkout (``TRACELENS_ROOT``), the agent runtime credentials the TraceLens skill route runs on, and
``HL_W13_CARDS`` to be exactly two cards. ``HL_W13_OUT`` must be on a filesystem that accepts ``:`` in file names.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import Purpose, TestKind, TestResult, Validity
from hyperloom.contracts.recipe import ComponentChange, RecipeDelta
from hyperloom.measure.compile import (
    COMPONENT_LAYOUTS,
    EnvironmentCompiler,
    LocalBlobStore,
    locate_installed_package,
)
from hyperloom.measure.hosted_spec import SpecContext
from hyperloom.inference_optimizer.session import paths as session_paths
from hyperloom.measure.points import default_canonical
from hyperloom.measure.profile_traces import PROFILE_DIR
from hyperloom.measure.spec import DEFAULT_MAX_TOTAL_BYTES
from hyperloom.measure.tests._recipes import IMAGE, git_patch, put_patch, recipe_with
from hyperloom.orchestrator.actions.executors._workload_envs import default_baseline_config
from hyperloom_multibranch.contracts.notebook import Role, Writer
from hyperloom_multibranch.contracts.notebook.records import ToolResultPointer
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.toolbox.local_hosted import (
    LocalHostedToolbox,
    LocalOptions,
    LocalSession,
    build_local_toolbox,
)
from hyperloom_multibranch.toolbox.planner import HostedPlanner
from hyperloom_multibranch.toolbox.toolbox import AcceptanceSetup
from hyperloom_multibranch.toolbox.tests.helpers import FakeRepo, req
from hyperloom_multibranch.toolbox.tools.op_shares import DEGRADED_NOTICE

MODEL = os.environ.get("HL_W13_MODEL", "")
INFERENCEX = os.environ.get("HL_W13_INFERENCEX", "")
OUT = os.environ.get("HL_W13_OUT", "")
CARDS = [int(card) for card in os.environ.get("HL_W13_CARDS", "6,7").split(",")]
USES_REAL_GPUS = True  # keep the host's amd-smi and rocm-smi: see hyperloom.measure.tests.gpu_free_host
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("sglang") is None or not Path(MODEL).is_dir() or not Path(INFERENCEX, ".git").exists(),
    reason="needs SGLang, HL_W13_MODEL and HL_W13_INFERENCEX (a git checkout)",
)

LEVELS = (4, 16, 64)
EVAL_LIMIT = "60"
TARGET = "launch_server.py"
READY_LINE = "The server is fired up and ready to roll"
FLEXIBLE = "exact_match,flexible-extract"
_WORKLOAD = {"FRAMEWORK": "sglang", "ISL": "256", "OSL": "64", "INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP": "1"}
_CLEARED = ("CONC", "RUN_EVAL", "TP", "EP", "PORT", "MAX_MODEL_LEN", "PRECISION", "RANDOM_RANGE_RATIO")


@pytest.fixture(autouse=True)
def workload_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The process environment the canonical point and the request points are rendered from."""
    for name in _CLEARED:
        monkeypatch.delenv(name, raising=False)
    for name, value in {**_WORKLOAD, "MAGPIE_EVAL_LIMIT": EVAL_LIMIT, "USER_DATA_PATH": str(tmp_path)}.items():
        monkeypatch.setenv(name, value)


def _run_root(name: str) -> Path:
    root = Path(OUT) / name
    root.mkdir(parents=True, exist_ok=False)
    return root


def _toolbox(root: Path, name: str, tp: int = 1) -> tuple[LocalHostedToolbox, LocalNotebook, LocalBlobStore]:
    session = root / "session"
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=INFERENCEX, capture_output=True, text=True, check=True
    ).stdout.strip()
    blobs = LocalBlobStore(session / "hosted" / "blobs")
    compiler = EnvironmentCompiler(blobs, IMAGE)
    canonical = default_canonical(
        root / "canonical", primary_concurrency=LEVELS[0], config_path=default_baseline_config(), model_path=MODEL
    )
    notebook = LocalNotebook(root / "notebook", Writer(Role.TOOLBOX, "toolbox"))
    spec = SpecContext(
        image_ref="local",
        image_digest=IMAGE,
        gpu_type="MI355X",
        magpie_commit="a" * 40,
        inferencex_commit=commit,
        blobs=blobs,
        process_environ={},
        platform_env={},
        benchmark_backend="bypass",
        blob_root=str(blobs.root),
    )
    toolbox = build_local_toolbox(
        LocalSession(session, name, CARDS, tp=tp),
        notebook,
        AcceptanceSetup(FakeRepo(recipe_with()), HostedPlanner(compiler, canonical, root / "points"), IMAGE),
        spec,
        options=LocalOptions(
            runner_envs={"INFERENCEX_REPO": INFERENCEX, "INFERENCE_OPTIMIZER_BASELINE_SERVER_READY_SEC": "900"},
            timeout_sec=2400,
        ),
    )
    return toolbox, notebook, blobs


async def _submit(toolbox: LocalHostedToolbox, notebook: LocalNotebook, request, root: Path):
    await toolbox.start()
    request_id = await toolbox.submit_test(request)
    await asyncio.wait_for(toolbox.drain(), 3300)
    entry = toolbox._log.get(request_id)
    assert entry is not None and entry.outcome is not None and "result" in entry.outcome, entry and entry.outcome
    result = TestResult.from_document(entry.outcome["result"])
    record = await notebook.record(request_id)
    assert record.result is not None and record.result.manifest_sha256 == result.manifest_sha256
    assert record.result.accuracy == result.accuracy and record.result.rounds == result.rounds
    journal = toolbox.backend.journal.latest(request_id)
    assert journal.extra["verified"] is True and result.validity is Validity.SUCCEEDED
    return request_id, result, root / "session" / result.output_dir


def _probe(label: str, **fields: object) -> None:
    print(f"PROBE {label} " + json.dumps(fields, default=str), flush=True)


async def test_an_accuracy_test_puts_the_measured_score_in_the_notebook_result() -> None:
    root = _run_root("accuracy")
    toolbox, notebook, *_ = _toolbox(root, "w13-accuracy")
    try:
        request = req(kind=TestKind.ACCURACY, purpose=Purpose.BASELINE, accuracy=True, change=RecipeDelta())
        request_id, result, landed = await _submit(toolbox, notebook, request, root)
        rounds = json.loads((landed / "rounds.json").read_text())
        _probe(
            "accuracy",
            request_id=request_id,
            accuracy=[value.to_document() for value in result.accuracy],
            rounds_file=rounds,
            result_files=sorted(str(p.relative_to(landed)) for p in landed.rglob("results*.json")),
        )
        assert result.accuracy, "an accuracy test with no accuracy value"
        for value in result.accuracy:
            assert re.fullmatch(r"lm_eval=\S+;\S+=\S+", value.version), f"version {value.version!r}"
        scores = {value.metric: value.score for value in result.accuracy}
        assert 0.0 < scores[FLEXIBLE] <= 1.0
        assert any(r["phase"] == "accuracy" and r["accuracy"] for r in rounds)
    finally:
        await toolbox.close()


async def test_a_concurrency_sweep_is_one_server_and_one_round_per_level() -> None:
    root = _run_root("sweep")
    toolbox, notebook, *_ = _toolbox(root, "w13-sweep")
    try:
        request = req(purpose=Purpose.BASELINE, change=RecipeDelta(), concurrency=LEVELS)
        request_id, result, landed = await _submit(toolbox, notebook, request, root)
        measured = [r for r in result.rounds if r.phase == "measure"]
        logs = sorted(landed.rglob("server*.log"))
        ready = {str(p.relative_to(landed)): p.read_text(errors="replace").count(READY_LINE) for p in logs}
        pid_files = sorted(str(p.relative_to(landed)) for p in landed.rglob("*.pid"))
        _probe(
            "sweep",
            request_id=request_id,
            levels=[r.concurrency for r in measured],
            throughput=[r.metrics.get("output_throughput") for r in measured],
            server_ready_lines=ready,
            pid_files=pid_files,
        )
        assert sorted(r.concurrency for r in measured) == sorted(LEVELS)
        assert sum(ready.values()) == 1, f"one server start for the arm, saw {ready}"
    finally:
        await toolbox.close()


def _marker_patch(workdir: Path, marker: Path) -> tuple[bytes, str]:
    """The patch that makes ``launch_server.py`` log its own path, and the text of the patched file."""
    layout = COMPONENT_LAYOUTS["sglang"]
    original = (locate_installed_package(layout).path / TARGET).read_text()
    code = f"with open({str(marker)!r}, 'a', encoding='utf-8') as _hl_f:\n    _hl_f.write(__file__ + '\\n')\n"
    key = f"{layout.upstream_prefix}/{TARGET}"
    return git_patch(workdir, {key: original}, {key: code + original}), code + original


async def test_a_patched_recipe_goes_through_the_planner_and_is_measured_on_the_rebuilt_framework() -> None:
    root = _run_root("patched")
    toolbox, notebook, blobs = _toolbox(root, "w13-patched")
    try:
        marker = root / "marker.txt"
        diff, patched = _marker_patch(root / "patch-repo", marker)
        patch = put_patch(blobs, "w13-marker.patch", diff)
        delta = RecipeDelta(components=(ComponentChange("sglang", add_patches=(patch,)),))
        request_id, result, landed = await _submit(toolbox, notebook, req(change=delta), root)
        inventory = json.loads(next(landed.rglob("farm_inventory.json")).read_text())
        imported = set(marker.read_text().split())
        _probe(
            "patched",
            request_id=request_id,
            spec_fingerprint=result.spec_fingerprint,
            imported=sorted(imported),
            inventory=inventory,
            throughput=[r.metrics.get("output_throughput") for r in result.rounds],
        )
        (farm_file,) = imported
        assert "/farm/site/sglang/" + TARGET in farm_file
        # the sandbox, and with it the rebuilt file, is gone by now: the signed inventory carries the file's hash
        assert toolbox.backend.journal.latest(request_id).sandbox_id in farm_file
        (root_entry,) = inventory["roots"].values()
        assert root_entry["upserts"][TARGET] == hashlib.sha256(patched.encode()).hexdigest()
    finally:
        await toolbox.close()


PROFILE_TRACE = "*.trace.json.gz"  # SGLang writes <ts>-TP-<r>-<DECODE|EXTEND>.trace.json.gz and a merged one
TRACELENS_JOB_SEC = 1800
MIN_TOTAL_SHARE = 50.0  # percent of the GPU time the (at most 50) kept ops must add up to
#: The per-op shares come from the ``ops_summary.csv`` TraceLens' skill route writes (the route runs an agent runtime,
#: which needs its credentials); the ``bypass`` route writes none. ``HL_W13_ANALYSIS_ROUTE=bypass`` therefore shows empty shares.
ANALYSIS_ROUTE = os.environ.get("HL_W13_ANALYSIS_ROUTE", "agent")


def _tree_bytes(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def _require_tracelens() -> None:
    """The analysis must run the real TraceLens: a missing checkout fails the test, it never skips it."""
    root = session_paths.tracelens_root()
    assert (root / ".git").exists(), f"no TraceLens git checkout at {root}: set TRACELENS_ROOT to one"


class _Pointers:
    """Remembers every tool-result pointer the toolbox hands to the notebook, then lets the notebook take it."""

    def __init__(self, notebook: LocalNotebook) -> None:
        self.seen: list[ToolResultPointer] = []
        self._deliver = notebook.notify_tool_result
        notebook.notify_tool_result = self.notify_tool_result  # type: ignore[method-assign]

    async def notify_tool_result(self, pointer: ToolResultPointer) -> None:
        self.seen.append(pointer)
        await self._deliver(pointer)


def _assert_profile_landed(session: Path, result: TestResult) -> list[Path]:
    """The profile's location holds real torch traces and the landed bundle is under the spec's byte cap."""
    location = session / result.profile_location
    assert result.profile_location and location.is_dir(), f"profile_location {result.profile_location!r}"
    traces = sorted(location.rglob(PROFILE_TRACE))
    assert traces and all(trace.stat().st_size > 0 for trace in traces), f"no {PROFILE_TRACE} in {location}"
    assert location.name == PROFILE_DIR and location.is_relative_to(session / result.output_dir)
    landed, archived = _tree_bytes(session / result.output_dir), _tree_bytes(session / "archive")
    assert 0 < landed < DEFAULT_MAX_TOTAL_BYTES and archived < DEFAULT_MAX_TOTAL_BYTES, (landed, archived)
    return traces


async def test_a_profile_on_two_cards_is_analysed_by_the_real_tracelens_into_op_shares(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert len(CARDS) == 2, f"the profile test runs on exactly two cards, HL_W13_CARDS is {CARDS}"
    _require_tracelens()
    monkeypatch.setenv("TP", "2")
    root = _run_root("profile")
    toolbox, notebook, _ = _toolbox(root, "w45-profile", tp=2)
    pointers = _Pointers(notebook)
    try:
        request = req(kind=TestKind.PROFILE, purpose=Purpose.PROFILE, change=RecipeDelta())
        request_id, result, _ = await _submit(toolbox, notebook, request, root)
        traces = _assert_profile_landed(root / "session", result)
        job = await toolbox.submit_analysis(
            request.branch_id, "tracelens", {"analysis_route": ANALYSIS_ROUTE, "framework": "sglang"}, request_id
        )
        status = await asyncio.wait_for(toolbox._tools.wait(job), TRACELENS_JOB_SEC)  # type: ignore[union-attr]
        (pointer,) = [p for p in pointers.seen if p.job_id == job]
        _probe(
            "profile",
            request_id=request_id,
            traces=[trace.name for trace in traces],
            profile_location=result.profile_location,
            route=ANALYSIS_ROUTE,
            job_status=str(status),
            summary=pointer.summary,
            top_op_shares=sorted(pointer.op_shares.items(), key=lambda item: -item[1])[:5],
        )
        assert DEGRADED_NOTICE not in pointer.summary, f"TraceLens report is degraded: {pointer.summary}"
        assert pointer.op_shares, f"TraceLens left no op shares: {pointer.summary}"
        assert pointer.commit_id == request.parent_commit and pointer.profile_request_id == request_id
        assert all(0.0 < share <= 100.0 for share in pointer.op_shares.values())
        # the complete per-op GPU time table, not the few optimisation tasks: the top ops cover most of the time
        assert sum(pointer.op_shares.values()) >= MIN_TOTAL_SHARE, pointer.op_shares
        print("TOP5_OP_SHARES", sorted(pointer.op_shares.items(), key=lambda item: -item[1])[:5])
        shares = await notebook.op_shares(request.parent_commit)
        assert shares is not None and shares.job_id == job and dict(shares.shares) == dict(pointer.op_shares)
    finally:
        await toolbox.close()
