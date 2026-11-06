# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract of the fixed and scriptable loads: every client knob explicit, both seeds recorded, results kept."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import time
from dataclasses import replace
from pathlib import Path

import pytest

from inference_suite.contracts import (
    Deadlines,
    EvalConfig,
    EvalProbe,
    FailedStep,
    FixedWorkload,
    LifecycleMode,
    Lifecycle,
    ProfileConfig,
    RunLayout,
    RunPaths,
    ScriptableWorkload,
    SuiteConfig,
    WorkloadKind,
)
from inference_suite.contracts.launch import ClientInvocation
from inference_suite.load.fixed import ClientPass, ClientTool, FixedRound, client_argv, fixed_metrics, run_fixed
from inference_suite.load.scriptable import (
    ShippedScript,
    run_scriptable,
    script_env,
    script_path,
)
from inference_suite.runtime.phases import Phase, PhaseClock
from inference_suite.runtime.server import Server
from inference_suite.runtime.watch import RoundGate
from inference_suite.tests.test_config_contract import _FIXED, _config

_DEADLINES = Deadlines(boot_sec=60, round_sec=60, eval_sec=60, silence_sec=60)
_SHA = "d" * 64
_FAKE_CLIENT = textwrap.dedent(
    """
    import json, os, sys
    argv = sys.argv[1:]
    with open(os.environ["FAKE_CALLS"], "a") as calls:
        calls.write(json.dumps(argv) + "\\n")
    value = lambda flag: argv[argv.index(flag) + 1]
    if "--save-result" in argv:
        result = {
            "num_prompts": int(value("--num-prompts")),
            "completed": int(value("--num-prompts")) - 1,
            "duration": 12.5,
            "output_throughput": 900.0,
            "mean_ttft_ms": 40.0,
            "median_ttft_ms": 38.0,
            "p99_tpot_ms": 9.0,
            "benchmark_outcome": {"failed": 1},
        }
        with open(os.path.join(value("--result-dir"), value("--result-filename")), "w") as out:
            json.dump(result, out)
    sys.exit(int(os.environ.get("FAKE_EXIT", "0")))
    """
)


def _round_config(tmp_path: Path, workload: WorkloadKind, **sections) -> SuiteConfig:
    probe = EvalProbe(enabled=False, min_samples=1, length_ratio=0.5)
    return replace(
        _config(),
        workload=workload,
        agentx=None,
        eval=EvalConfig(enabled=False, num_fewshot=0, concurrency=1, probe=probe),
        profile=ProfileConfig(enabled=False),
        lifecycle=Lifecycle(mode=LifecycleMode.OWN),
        deadlines=_DEADLINES,
        paths=RunPaths(run_dir=str(tmp_path / "run"), tool_cache=str(tmp_path / "tools"), tool_manifest_sha256=_SHA),
        **sections,
    )


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ClientTool:
    package = tmp_path / "ix" / "infx" / "bench_serving"
    package.mkdir(parents=True)
    for init in (package.parent / "__init__.py", package / "__init__.py"):
        init.write_text("")
    (package / "benchmark_serving.py").write_text(_FAKE_CLIENT)
    monkeypatch.setenv("FAKE_CALLS", str(tmp_path / "calls.jsonl"))
    return ClientTool(python=sys.executable, package_root=str(tmp_path / "ix"))


def _fixed_round(tmp_path: Path, tool: ClientTool, workload: FixedWorkload = _FIXED) -> tuple[FixedRound, PhaseClock]:
    config = _round_config(tmp_path, WorkloadKind.FIXED, fixed=workload)
    config = replace(config, server=replace(config.server, env_inherit=("FAKE_CALLS", "FAKE_EXIT")))
    clock = PhaseClock()
    server = Server("http://127.0.0.1:30000", True, None, lambda: True)
    return FixedRound(config, tool, server, RoundGate(_DEADLINES, lambda: "", time.monotonic()), clock), clock


def _calls(tmp_path: Path) -> list[list[str]]:
    return [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]


def _flag(argv: list[str] | tuple[str, ...], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def test_client_argv_passes_every_load_knob_explicitly(tmp_path: Path):
    tool = ClientTool(python="/py", package_root="/ix")
    server = _config().server
    argv = client_argv(
        tool, "http://127.0.0.1:30000", server, _FIXED, ClientPass(seed=7, num_prompts=640, result=tmp_path / "r.json")
    )
    assert argv[:3] == ("/py", "-m", "infx.bench_serving.benchmark_serving")
    expected = {
        "--seed": "7",
        "--num-prompts": "640",
        "--num-warmups": "0",
        "--tokenizer-mode": "auto",
        "--random-range-ratio": "0.8",
        "--random-input-len": "1024",
        "--random-output-len": "1024",
        "--random-prefix-len": "0",
        "--random-num-workers": "8",
        "--max-concurrency": "64",
        "--dataset-name": "random",
        "--request-rate": "inf",
        "--model": "/models/GLM-5.3",
        "--served-model-name": "glm",
        "--base-url": "http://127.0.0.1:30000",
        "--metric-percentiles": "90,99",
        "--result-dir": str(tmp_path),
        "--result-filename": "r.json",
    }
    assert {flag: _flag(argv, flag) for flag in expected} == expected
    assert "--trust-remote-code" in argv and "--save-result" in argv and "--ignore-eos" in argv
    assert "--use-chat-template" not in argv
    quiet = client_argv(
        tool, "http://127.0.0.1:30000", server, replace(_FIXED, trust_remote_code=False), ClientPass(0, 1, None)
    )
    assert "--trust-remote-code" not in quiet and "--save-result" not in quiet


def test_fixed_runs_the_warmup_on_its_own_seed_then_the_measured_pass_and_records_both_seeds(tmp_path, client):
    fixed_round, clock = _fixed_round(tmp_path, client)
    outcome = run_fixed(fixed_round)

    warmup, measured = _calls(tmp_path)
    assert (_flag(warmup, "--seed"), _flag(warmup, "--num-prompts"), "--save-result" in warmup) == ("1", "128", False)
    assert (_flag(measured, "--seed"), _flag(measured, "--num-prompts"), _flag(measured, "--num-warmups")) == (
        "0",
        "640",
        "0",
    )
    assert outcome.stop is None and outcome.returncode == 0
    assert outcome.invocation.argv[3:] == tuple(measured)
    assert (outcome.invocation.seed, outcome.invocation.warmup_seed) == (0, 1)
    document = outcome.invocation.to_document()
    assert ClientInvocation.from_document(document) == outcome.invocation
    assert (document["seed"], document["warmup_seed"], document["env"]) == (0, 1, {"PYTHONPATH": client.package_root})

    phases = [mark.phase for mark in clock.marks]
    assert phases == [Phase.WARMUP, Phase.MEASURED, Phase.FINISHED]
    metrics = outcome.metrics.to_document()
    assert (metrics["requested_requests"], metrics["completed_requests"], metrics["failed_requests"]) == (640, 639, 1)
    assert (metrics["ttft_mean_ms"], metrics["ttft_p50_ms"], metrics["tpot_p99_ms"]) == (40.0, 38.0, 9.0)
    assert metrics["input_throughput"] is None and metrics["e2el_p99_ms"] is None


def test_fixed_without_warmups_runs_one_pass(tmp_path, client):
    fixed_round, clock = _fixed_round(tmp_path, client, replace(_FIXED, num_warmups=0))
    assert run_fixed(fixed_round).stop is None
    assert len(_calls(tmp_path)) == 1
    assert [mark.phase for mark in clock.marks] == [Phase.MEASURED, Phase.FINISHED]


def test_a_client_failing_after_saving_keeps_its_metrics_and_fails_the_round(tmp_path, client, monkeypatch):
    monkeypatch.setenv("FAKE_EXIT", "1")
    fixed_round, _ = _fixed_round(tmp_path, client, replace(_FIXED, num_warmups=0))
    outcome = run_fixed(fixed_round)
    assert outcome.stop.step is FailedStep.CLIENT_FAILED and outcome.returncode == 1
    assert outcome.metrics is not None


def test_a_failing_warmup_skips_the_measured_pass(tmp_path, client, monkeypatch):
    monkeypatch.setenv("FAKE_EXIT", "3")
    fixed_round, _ = _fixed_round(tmp_path, client)
    outcome = run_fixed(fixed_round)
    assert outcome.stop.step is FailedStep.CLIENT_FAILED and outcome.metrics is None
    assert len(_calls(tmp_path)) == 1


def test_fixed_metrics_refuse_a_mistyped_value_and_leave_absent_ones_null():
    with pytest.raises(ValueError):
        fixed_metrics({"completed": "many"})
    assert set(fixed_metrics({}).to_document().values()) == {None}


def _scriptable_config(tmp_path: Path, workload: ScriptableWorkload) -> SuiteConfig:
    return _round_config(tmp_path, WorkloadKind.SCRIPTABLE, scriptable=workload)


def test_the_xdit_script_ships_and_parses():
    assert script_path("xdit") == ShippedScript.XDIT.path
    subprocess.run(["bash", "-n", str(ShippedScript.XDIT.path)], check=True)


def test_a_script_is_a_shipped_name_or_an_existing_absolute_path(tmp_path: Path):
    own = tmp_path / "bench.sh"
    own.write_text("exit 0\n")
    assert script_path(str(own)) == own
    assert script_path("bench.sh") is None
    assert script_path(str(tmp_path / "missing.sh")) is None


def test_script_env_takes_params_and_keeps_run_scoped_names_for_the_round(tmp_path: Path):
    workload = ScriptableWorkload("xdit", {"XDIT_USE_TORCH_COMPILE": True, "XDIT_HEIGHT": 768}, "/refs/cat.png")
    config = _scriptable_config(tmp_path, workload)
    env = script_env(workload, config.server, RunLayout(Path(config.paths.run_dir)))
    assert env == {
        "XDIT_USE_TORCH_COMPILE": "1",
        "XDIT_HEIGHT": "768",
        "RESULT_DIR": str(tmp_path / "run" / "client"),
        "RESULT_FILENAME": "inferencex_result",
        "MODEL": "/models/GLM-5.3",
        "TP": "8",
        "ROCR_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        "QUALITY_REFERENCE": "/refs/cat.png",
    }


def _run_script(tmp_path: Path, body: str):
    script = tmp_path / "bench.sh"
    script.write_text(body)
    clock = PhaseClock()
    config = _scriptable_config(tmp_path, ScriptableWorkload(str(script), {"STEPS": 3}))
    return run_scriptable(config, RoundGate(_DEADLINES, lambda: "", time.monotonic()), clock), clock


def test_scriptable_keeps_the_result_the_script_writes(tmp_path: Path):
    body = 'echo "{\\"img_per_s\\": 2.5, \\"steps\\": $STEPS}" > "$RESULT_DIR/$RESULT_FILENAME.json"\n'
    outcome, clock = _run_script(tmp_path, body)
    assert outcome.stop is None and outcome.returncode == 0
    assert outcome.result.to_document() == {
        "result_file": "client/inferencex_result.json",
        "result": {"img_per_s": 2.5, "steps": 3},
    }
    assert [mark.phase for mark in clock.marks] == [Phase.MEASURED, Phase.FINISHED]


@pytest.mark.parametrize(
    ("body", "step"),
    [("exit 4\n", FailedStep.CLIENT_FAILED), ("exit 0\n", FailedStep.CLIENT_OUTPUT_MISSING)],
)
def test_scriptable_fails_on_a_bad_exit_or_no_result(tmp_path: Path, body: str, step: FailedStep):
    outcome, _ = _run_script(tmp_path, body)
    assert outcome.stop.step is step and outcome.result is None


def test_scriptable_refuses_a_script_it_cannot_find(tmp_path: Path):
    config = _scriptable_config(tmp_path, ScriptableWorkload("nonesuch"))
    outcome = run_scriptable(config, RoundGate(_DEADLINES, lambda: "", time.monotonic()), PhaseClock())
    assert outcome.invocation is None and outcome.stop.step is FailedStep.CONFIG_INVALID
