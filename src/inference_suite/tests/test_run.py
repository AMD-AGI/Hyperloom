# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract of a round: one round per call into an empty run_dir, which does not change once the call returns, GPU
samples from the measured window alone, KV samples tagged by phase, the checks that refuse a round before its server starts, a reuse round on
the server a boot round left, and the CLI's ``run`` and ``stop`` round trips."""

from __future__ import annotations

import csv
import functools
import hashlib
import json
import socket
import subprocess
import sys
import textwrap
import time
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from hyperloom_common import gpu_telemetry
from hyperloom_common.tests.test_gpu_telemetry import _row
from inference_suite import cli, parser_check, run
from inference_suite.contracts import (
    BaseRef,
    Deadlines,
    FailedStep,
    FixedWorkload,
    GpuTelemetry,
    Lifecycle,
    LifecycleMode,
    ProfileConfig,
    Redacted,
    RunLayout,
    RunPaths,
    RunStatus,
    SuiteConfig,
    WorkloadKind,
)
from inference_suite.contracts.base_launch import Workload
from inference_suite.contracts.launch import Launch, LayerSource
from inference_suite.contracts.report import Report
from inference_suite.eval.definition import EvalPlan
from inference_suite.framework_probe import FrameworkInstall
from inference_suite.parser_check import FlagRefusal, RefusedFlag
from inference_suite.pins import INFERENCEX_TOOL, PINNED_TOOLS
from inference_suite.runtime import server as runtime_server
from inference_suite.runtime import watch
from inference_suite.runtime.phases import Phase
from inference_suite.staging import GSM8K_TASK, StagingReport
from inference_suite.tests.test_base_launch_contract import _base
from inference_suite.tests.test_config_contract import _SHA, _config, _eval

pytestmark = pytest.mark.skipif(not Path("/proc/net/tcp").exists(), reason="needs Linux procfs")

_SECRET = "hf-secret"
_SERVER = textwrap.dedent(
    """
    import http.server, sys
    port = int(sys.argv[sys.argv.index("--port") + 1])
    print("server starting on", port, flush=True)

    class Health(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = b"sglang:token_usage 0.25\\nsglang:max_total_num_tokens 4096\\n" if self.path == "/metrics" else b""
            self.send_response(200 if self.path in ("/health", "/metrics") else 404)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    httpd = http.server.HTTPServer(("127.0.0.1", port), Health)
    print("The server is fired up and ready to roll!", flush=True)
    httpd.serve_forever()
    """
)
_CLIENT = textwrap.dedent(
    """
    import argparse, json, os, sys, time
    parser = argparse.ArgumentParser()
    for name in ("--seed", "--num-prompts", "--result-dir", "--result-filename"):
        parser.add_argument(name)
    parser.add_argument("--save-result", action="store_true")
    args, _ = parser.parse_known_args()
    with open(os.environ["FAKE_CLIENT_CALLS"], "a") as calls:
        calls.write(json.dumps({"seed": args.seed, "num_prompts": args.num_prompts}) + "\\n")
    for _ in range(6):
        print("request done", flush=True)
        time.sleep(0.2)
    if args.save_result:
        result = {"output_throughput": 812.5, "completed": int(args.num_prompts), "num_prompts": int(args.num_prompts),
                  "duration": 1.2, "benchmark_outcome": {"failed": 0}}
        with open(os.path.join(args.result_dir, args.result_filename), "w") as fh:
            json.dump(result, fh)
    """
)
_EVAL = textwrap.dedent(
    """
    import json, os, sys, time
    request = json.load(open(sys.argv[1]))
    time.sleep(0.5)
    out = os.path.join(request["output_dir"], "served")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "results_2026-11-06T00-00-00.json"), "w") as fh:
        json.dump({"results": {"gsm8k": {"alias": "gsm8k", "exact_match,strict-match": 0.95}},
                   "versions": {"gsm8k": 3}}, fh)
    with open(request["probe_file"], "w") as fh:
        json.dump({"tripped": False, "samples_seen": 200, "length_ratio_observed": 0.01}, fh)
    """
)
_FIXED = FixedWorkload(
    concurrency=4,
    isl=128,
    osl=64,
    random_range_ratio=1.0,
    num_prompts=8,
    num_warmups=2,
    seed=0,
    warmup_seed=1,
    tokenizer_mode="auto",
    trust_remote_code=False,
)


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    monkeypatch.setattr(watch, "POLL_SEC", 0.05)
    monkeypatch.setattr(runtime_server, "POLL_SEC", 0.05)
    monkeypatch.setattr(gpu_telemetry, "_read_node_power", lambda: None)
    monkeypatch.setattr(gpu_telemetry, "MIN_INTERVAL_SEC", 0.01)


class _AmdSmi:
    """Answers every sample with eight loaded cards; a ``--violation`` read finds no counters."""

    def __call__(self, *metrics: str) -> object:
        if "--violation" in metrics:
            return {}
        return {"gpu_data": [_row(gpu, 700.0, 200_000.0) for gpu in range(8)]}


class _Rig:
    """A fake framework server, the staged fake client and eval, and the config of a fixed round on them."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.tmp = tmp_path
        self.calls = tmp_path / "client_calls.jsonl"
        monkeypatch.setenv("FAKE_CLIENT_CALLS", str(self.calls))
        monkeypatch.setenv("HF_TOKEN", _SECRET)
        self.tools = tmp_path / "tools"
        root = self.tools / INFERENCEX_TOOL / PINNED_TOOLS[INFERENCEX_TOOL]
        client = root / "infx" / "bench_serving"
        client.mkdir(parents=True)
        for package in (root / "infx", client):
            (package / "__init__.py").write_text("")
        (client / "benchmark_serving.py").write_text(_CLIENT)
        (root / GSM8K_TASK).parent.mkdir(parents=True)
        (root / GSM8K_TASK).write_text("task: gsm8k\n")
        server_script = tmp_path / "fake_server.py"
        server_script.write_text(_SERVER)
        self.eval_script = tmp_path / "fake_eval.py"
        self.eval_script.write_text(_EVAL)
        self.base = _base(
            query=replace(_base().query, workload=Workload.FIXED, isl=_FIXED.isl, osl=_FIXED.osl),
            golden_al=None,
            entrypoint=(sys.executable, str(server_script)),
            args=("--disable-radix-cache",),
            env={"SGLANG_USE_AITER": "1", "HF_TOKEN": Redacted(hashlib.sha256(_SECRET.encode()).hexdigest())},
        )
        self.base_path = tmp_path / "base_launch.json"
        self.base_path.write_text(json.dumps(self.base.to_document()))
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        self.installed = "0.5.23"
        self.refusal: FlagRefusal | None = None
        monkeypatch.setattr(run, "probe_install", self._probe)
        monkeypatch.setattr(parser_check, "check", lambda *_args: self.refusal)
        monkeypatch.setattr(run, "_lm_eval_version", lambda: "0.4.9")
        monkeypatch.setattr(run, "GpuPowerRecorder", functools.partial(gpu_telemetry.GpuPowerRecorder, query=_AmdSmi()))
        planned = EvalPlan.invocation
        monkeypatch.setattr(
            EvalPlan,
            "invocation",
            lambda plan, python, request: replace(
                planned(plan, python, request), argv=(python, str(self.eval_script), str(request))
            ),
        )
        self.staging = StagingReport(_SHA, dict(PINNED_TOOLS), ())
        monkeypatch.setattr(run.staging, "verify", lambda *_args: self.staging)

    def _probe(self, python, framework, env):
        assert env["HF_TOKEN"] == _SECRET
        return FrameworkInstall(python, "3.12.3", "/site-packages/sglang/__init__.py", self.installed)

    def config(self, run_dir: str, **changes) -> SuiteConfig:
        base = _config()
        server = replace(
            base.server,
            base=BaseRef(str(self.base_path), self.base.digest()),
            port=self.port,
            source_files={},
            absent_source_files=(),
            env_inherit=("FAKE_CLIENT_CALLS", "HF_TOKEN"),
        )
        config = replace(
            base,
            server=server,
            workload=WorkloadKind.FIXED,
            agentx=None,
            fixed=_FIXED,
            profile=ProfileConfig(enabled=False),
            eval=_eval(enabled=False),
            telemetry=GpuTelemetry(enabled=True, interval_sec=0.05),
            deadlines=Deadlines(boot_sec=60, round_sec=120, eval_sec=60, silence_sec=60),
            paths=RunPaths(str(self.tmp / run_dir), str(self.tools), _SHA),
        )
        return replace(config, **changes)


@pytest.fixture
def rig(tmp_path, monkeypatch):
    return _Rig(tmp_path, monkeypatch)


def _snapshot(root: Path) -> dict[str, tuple[int, int]]:
    return {str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns) for path in root.rglob("*")}


def _marks(layout: RunLayout) -> dict[str, float]:
    return {mark["phase"]: mark["ts"] for mark in json.loads(layout.phase_marks.read_text())["phase_marks"]}


def test_a_fixed_round_with_eval_reports_what_it_measured_and_leaves_its_run_dir_settled(rig):
    config = rig.config("r1", eval=_eval(limit=None))

    result = run.run_round(config)

    report, layout = result.report, result.layout
    assert report.run_status is RunStatus.COMPLETED, report.failure_detail
    assert Report.from_document(json.loads(layout.report.read_text())) == report
    launch = Launch.from_document(json.loads(layout.launch.read_text()))
    assert (report.launch_digest, report.base_digest) == (launch.digest(), rig.base.digest())
    assert _SECRET not in layout.launch.read_text()
    assert SuiteConfig.from_document(yaml.safe_load(layout.config.read_text())) == config
    assert report.fixed is not None and report.fixed.values["output_throughput"] == 812.5
    assert report.codes.client == 0 and report.codes.eval == 0 and report.concurrency == _FIXED.concurrency
    assert [json.loads(line) for line in rig.calls.read_text().splitlines()] == [
        {"seed": "1", "num_prompts": "2"},
        {"seed": "0", "num_prompts": "8"},
    ]
    assert report.eval is not None and report.eval.scores == {"exact_match,strict-match": 0.95}
    assert report.eval.definition_digest == launch.eval.definition_digest
    assert report.eval.results_file == "eval/served/results_2026-11-06T00-00-00.json"
    assert report.server is not None and not report.server.reused and report.server.log_range is None
    assert not layout.server_pid.exists()

    settled = _snapshot(layout.root)
    time.sleep(0.5)
    assert _snapshot(layout.root) == settled


def test_gpu_samples_cover_the_measured_window_alone(rig):
    result = run.run_round(rig.config("r1", eval=_eval(limit=None)))

    layout = result.layout
    marks = _marks(layout)
    assert list(marks) == [
        Phase.SPAWN.value,
        Phase.READY.value,
        Phase.WARMUP.value,
        Phase.MEASURED.value,
        Phase.FINISHED.value,
        Phase.EVAL.value,
        Phase.EVAL_END.value,
        Phase.TEARDOWN.value,
    ]
    with layout.gpu_samples.open() as samples:
        stamps = [float(row["unix"]) for row in csv.DictReader(samples)]
    assert stamps
    assert all(marks[Phase.MEASURED.value] - 0.01 <= unix <= marks[Phase.FINISHED.value] + 0.01 for unix in stamps)
    blocks = result.report.gpu_monitor
    assert [block.gpu for block in blocks] == list(range(8)) and all(block.sample_count for block in blocks)
    assert json.loads(layout.gpu_power.read_text())["samples"] == len(stamps) // 8


def test_kv_samples_cover_the_ready_server_and_carry_their_phase(rig):
    config = rig.config("r1", eval=_eval(limit=None))
    result = run.run_round(config)

    launch = Launch.from_document(json.loads(result.layout.launch.read_text()))
    base_url = runtime_server.server_url(config.server)
    assert launch.client is not None and launch.client.argv[launch.client.argv.index("--base-url") + 1] == base_url
    kv = json.loads(result.layout.kv_metrics.read_text())
    assert kv["url"] == base_url + "/metrics" and kv["available"] is True and not kv["aborted"]
    phases = {row["phase"] for row in kv["samples"]}
    assert {Phase.WARMUP.value, Phase.MEASURED.value, Phase.FINISHED.value, Phase.EVAL.value} <= phases
    assert all(row["capacity_tokens"] == 4096 for row in kv["samples"])


def test_the_refusal_says_whether_the_host_holds_a_redacted_value(rig, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "another")
    differs = run.run_round(rig.config("r1")).report.failure_detail
    monkeypatch.delenv("HF_TOKEN")
    absent = run.run_round(rig.config("r2")).report.failure_detail

    assert differs is not None and "not the one the base was resolved with" in differs
    assert absent is not None and "holds no value" in absent


def test_a_round_writes_only_into_an_empty_run_dir(rig):
    config = rig.config("r1", telemetry=GpuTelemetry(enabled=False, interval_sec=1))
    rig.staging = StagingReport(_SHA, {}, ("inferencex at its pin (not staged)",))
    run.run_round(config)

    with pytest.raises(run.RunDirError):
        run.run_round(config)


@pytest.mark.parametrize(
    ("change", "step"),
    [
        ("staging", FailedStep.STAGING_MISSING),
        ("digest", FailedStep.BASE_DIGEST_MISMATCH),
        ("source_files", FailedStep.SOURCE_FILES_CHANGED),
        ("too_old", FailedStep.FRAMEWORK_VERSION_TOO_OLD),
        ("unversioned", FailedStep.FRAMEWORK_VERSION_UNKNOWN),
        ("redacted", FailedStep.CONFIG_INVALID),
        ("profile", FailedStep.CONFIG_INVALID),
    ],
)
def test_a_round_refused_before_its_server_starts_writes_a_failed_report(rig, monkeypatch, tmp_path, change, step):
    config = rig.config("r1")
    if change == "staging":
        rig.staging = StagingReport(_SHA, {}, ("inferencex at its pin (not staged)",))
    elif change == "digest":
        config = replace(config, server=replace(config.server, base=BaseRef(str(rig.base_path), "sha256:" + "0" * 64)))
    elif change == "source_files":
        patch = tmp_path / "patch.py"
        patch.write_text("changed\n")
        config = replace(config, server=replace(config.server, source_files={str(patch): _SHA}))
    elif change == "too_old":
        rig.installed = "0.5.20"
    elif change == "unversioned":
        rig.installed = None
    elif change == "redacted":
        monkeypatch.setenv("HF_TOKEN", "another")
    else:
        window = ProfileConfig(enabled=True, num_prompts=8, delay_iters=2, max_iters=4)
        config = replace(config, profile=window, lifecycle=Lifecycle(LifecycleMode.OWN))

    report = run.run_round(config).report

    assert (report.run_status, report.failed_step) == (RunStatus.FAILED, step)
    layout = RunLayout(Path(config.paths.run_dir))
    assert not layout.server_log.exists() and report.server is None
    assert Report.from_document(json.loads(layout.report.read_text())) == report


def test_a_flag_the_parser_refuses_names_the_candidate_layer(rig):
    rig.refusal = FlagRefusal((RefusedFlag("--cuda-graph-max-bs", LayerSource.CANDIDATE),), "unrecognized arguments")

    report = run.run_round(rig.config("r1")).report

    assert report.failed_step is FailedStep.FLAG_UNKNOWN
    assert report.failure_detail == "the candidate layer: unrecognized arguments"


def test_a_cancelled_round_stops_and_reaps_its_server(rig):
    report = run.run_round(rig.config("r1"), cancel=lambda: "the operator cancelled").report

    assert (report.run_status, report.failed_step) == (RunStatus.STOPPED, FailedStep.CANCELLED)
    assert report.failure_detail == "the operator cancelled"
    assert not RunLayout(rig.tmp / "r1").server_pid.exists()


def test_a_reuse_round_runs_on_the_server_a_boot_round_left_and_the_cli_stops_it(rig):
    pid_dir = rig.tmp / "pids"
    boot = rig.config(
        "boot", workload=WorkloadKind.NONE, fixed=None, lifecycle=Lifecycle(LifecycleMode.BOOT, str(pid_dir))
    )
    booted = run.run_round(boot).report
    assert booted.run_status is RunStatus.COMPLETED, booted.failure_detail

    reuse = rig.config("reuse", lifecycle=Lifecycle(LifecycleMode.REUSE, str(pid_dir)))
    reused = run.run_round(reuse).report

    assert reused.run_status is RunStatus.COMPLETED, reused.failure_detail
    assert reused.server is not None and reused.server.reused and reused.server.log_range is not None
    assert RunLayout(rig.tmp / "reuse").read_ready_stamp().boot_sec == 0
    stopped = subprocess.run(
        [sys.executable, "-m", "inference_suite", "stop", "--pid-dir", str(pid_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert stopped.returncode == cli.EXIT_OK, stopped.stderr
    assert json.loads(stopped.stdout)["reaped"] and not json.loads(stopped.stdout)["survivors"]
    assert not (pid_dir / "server.pid").exists()


def test_cli_run_round_trips_the_config_and_exits_by_the_round_status(rig, tmp_path, capsys):
    config = rig.config("ignored", telemetry=GpuTelemetry(enabled=False, interval_sec=1))
    rig.staging = StagingReport(_SHA, {}, ("inferencex at its pin (not staged)",))
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(config.to_document()))
    run_dir = tmp_path / "cli-run"

    assert cli.main(["run", "--config", str(config_file), "--run-dir", str(run_dir)]) == cli.EXIT_NOT_COMPLETED

    layout = RunLayout(run_dir)
    assert capsys.readouterr().out.strip() == str(layout.report)
    written = SuiteConfig.from_document(yaml.safe_load(layout.config.read_text()))
    assert written == replace(config, paths=replace(config.paths, run_dir=str(run_dir)))
    assert Report.from_document(json.loads(layout.report.read_text())).failed_step is FailedStep.STAGING_MISSING
    assert cli.main(["run", "--config", str(config_file), "--run-dir", str(run_dir)]) == cli.EXIT_REFUSED
    config_file.write_text("schema: inference-suite-config/0\n")
    assert cli.main(["run", "--config", str(config_file), "--run-dir", str(tmp_path / "x")]) == cli.EXIT_REFUSED


def test_cli_stop_on_a_pid_dir_without_a_server_reaps_nothing(tmp_path, capsys):
    assert cli.main(["stop", "--pid-dir", str(tmp_path)]) == cli.EXIT_OK
    assert json.loads(capsys.readouterr().out) == {"reaped": [], "survivors": []}
