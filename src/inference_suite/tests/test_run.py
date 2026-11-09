# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract of a round: one round per call into an empty run_dir, which does not change once the call returns, GPU
samples from the measured window alone, KV samples tagged by phase, a fixed profile round's window around its
measured pass, an AgentX profile round's window on the replay's phase clock, an MLPerf round's harness against the
round's server, the checks that refuse a round before its
server starts, a reuse round on the server a boot round left, and the CLI's ``run`` and ``stop`` round trips."""

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
    AgentxWorkload,
    BaseRef,
    Deadlines,
    FailedStep,
    FixedWorkload,
    GpuTelemetry,
    LaunchModel,
    LaunchQuery,
    Lifecycle,
    LifecycleMode,
    ProfileConfig,
    Redacted,
    RunDirError,
    RunLayout,
    RunPaths,
    RunStatus,
    ScriptableWorkload,
    SuiteConfig,
    Telemetry,
    Topology,
    WorkloadKind,
)
from inference_suite.contracts.base_launch import LaunchSource, SourceKind, Workload
from inference_suite.contracts.launch import Launch, LayerSource, VersionCheck
from inference_suite.contracts.report import Report
from inference_suite.eval.definition import EvalPlan
from inference_suite.framework_probe import FrameworkInstall, FrameworkProbeError
from inference_suite.load.scriptable import scriptable_base
from inference_suite.parser_check import FlagRefusal, RefusedFlag
from inference_suite.pins import INFERENCEX_TOOL, PINNED_TOOLS
from inference_suite.recipe.image_version import NO_RECIPE_IMAGE
from inference_suite.runtime import server as runtime_server
from inference_suite.runtime import watch
from inference_suite.runtime.phases import Phase
from inference_suite.staging import AGENTX_PYTHON, GSM8K_TASK, StagingReport
from inference_suite.tests.test_base_launch_contract import _base
from inference_suite.tests.test_config_contract import _AGENTX, _MLPERF, _SHA, _config, _eval

pytestmark = pytest.mark.skipif(not Path("/proc/net/tcp").exists(), reason="needs Linux procfs")

_SECRET = "hf-secret"
_SERVER = textwrap.dedent(
    """
    import http.server, json, os, sys
    port = int(sys.argv[sys.argv.index("--port") + 1])
    print("server starting on", port, flush=True)
    metrics = b"sglang:token_usage 0.25\\nsglang:max_total_num_tokens 4096\\nsglang:num_running_reqs 8\\n"

    class Health(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = metrics if self.path == "/metrics" else b""
            self.send_response(200 if self.path in ("/health", "/metrics") else 404)
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"null")
            with open(os.environ["FAKE_CLIENT_CALLS"], "a") as calls:
                calls.write(json.dumps({"path": self.path, "body": body}) + "\\n")
            if self.path == "/stop_profile":
                trace_dir = os.environ["SGLANG_TORCH_PROFILER_DIR"]
                with open(os.path.join(trace_dir, "rank0.trace.json.gz"), "w") as trace:
                    trace.write("trace")
            refused = os.environ.get("FAKE_PROFILE_REFUSED") == "1"
            known = self.path in ("/start_profile", "/stop_profile")
            self.send_response(500 if refused else 200 if known else 404)
            self.end_headers()

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
_HARNESS = textwrap.dedent(
    """
    import http.server, json, os, sys, threading, time
    argv = sys.argv[1:]
    value = lambda flag: argv[argv.index(flag) + 1]
    if "agentx_cli" not in argv:
        result = {"num_requests_successful": 2,
                  "request_accounting": {"records_total": 4, "records_warmup_dropped": 2, "records_error_dropped": 0,
                                         "records_dropped_total": 2},
                  "request_metrics": {"qps": {"mean": 3.0}}}
        with open(os.path.join(os.environ["RESULT_DIR"], os.environ["RESULT_FILENAME"] + ".json"), "w") as fh:
            json.dump(result, fh)
        sys.exit(0)
    with open(os.environ["FAKE_CLIENT_CALLS"], "a") as calls:
        calls.write(json.dumps(argv) + "\\n")
    begun = time.time_ns()
    phases = {"warmup": {"start_ns": begun}}

    class Progress(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"phases": phases}).encode())

        def log_message(self, *args):
            pass

    api = http.server.HTTPServer((value("--api-host"), int(value("--api-port"))), Progress)
    threading.Thread(target=api.serve_forever, daemon=True).start()
    time.sleep(0.4)
    phases["profiling"] = {"start_ns": time.time_ns()}
    for _ in range(8):
        print("request done", flush=True)
        time.sleep(0.2)
    phases["profiling"]["requests_end_ns"] = time.time_ns()
    time.sleep(0.2)
    artifacts = value("--output-artifact-dir")
    os.makedirs(artifacts, exist_ok=True)
    with open(os.path.join(artifacts, "profile_export.jsonl"), "w") as export:
        for phase in ("warmup", "warmup", "profiling", "profiling"):
            export.write(json.dumps({"metadata": {"benchmark_phase": phase}, "metrics": {}}) + "\\n")
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
        harness = tmp_path / "fake_harness.py"
        harness.write_text(_HARNESS)
        harness_python = root / AGENTX_PYTHON
        harness_python.parent.mkdir(parents=True)
        harness_python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{harness}" "$@"\n')
        harness_python.chmod(0o755)
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
        self.probe_cwds: list[Path] = []
        monkeypatch.setattr(run, "probe_install", self._probe)
        monkeypatch.setattr(parser_check, "check", self._check)
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
        self.busy_gpus: str | None = None
        monkeypatch.setattr(runtime_server, "unavailable_gpus", lambda gpus: self.busy_gpus)

    def _probe(self, python, framework, env, *, cwd):
        assert env["HF_TOKEN"] == _SECRET
        self.probe_cwds.append(cwd)
        return FrameworkInstall(python, "3.12.3", "/site-packages/sglang/__init__.py", self.installed)

    def _check(self, launch, python, env, *, cwd):
        self.probe_cwds.append(cwd)
        return self.refusal

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
            telemetry=Telemetry(GpuTelemetry(enabled=True, interval_sec=0.05), kv=True),
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
    assert rig.probe_cwds == [layout.root, layout.root]

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


def _user_file_base(rig) -> None:
    rig.base = replace(
        rig.base,
        source=LaunchSource(kind=SourceKind.USER_FILE, path="launch.sh", files={"launch.sh": _SHA}),
        image=NO_RECIPE_IMAGE,
    )
    rig.base_path.write_text(json.dumps(rig.base.to_document()))


@pytest.mark.parametrize("kv", [True, False])
def test_a_user_file_sglang_base_serves_metrics_exactly_when_the_round_scrapes_them(rig, kv):
    _user_file_base(rig)
    config = rig.config("r1", telemetry=replace(rig.config("r1").telemetry, kv=kv))
    result = run.run_round(config)

    assert result.report.run_status is RunStatus.COMPLETED, result.report.failure_detail
    argv = Launch.from_document(json.loads(result.layout.launch.read_text())).argv
    assert ("--enable-metrics" in argv) is kv
    assert result.layout.kv_metrics.exists() is kv


_FIXED_WINDOW = ProfileConfig(
    enabled=True, start_body={"merge_profiles": True}, num_prompts=16, delay_iters=2, max_iters=4
)


def test_a_fixed_profile_round_opens_its_window_after_the_warmup_and_closes_it_after_the_measured_pass(rig):
    result = run.run_round(rig.config("r1", profile=_FIXED_WINDOW))

    report, layout = result.report, result.layout
    assert report.run_status is RunStatus.COMPLETED, report.failure_detail
    assert [json.loads(line) for line in rig.calls.read_text().splitlines()] == [
        {"seed": "1", "num_prompts": "2"},
        {"path": "/start_profile", "body": {"merge_profiles": True, "start_step": 2, "num_steps": 4}},
        {"seed": "0", "num_prompts": "16"},
        {"path": "/stop_profile", "body": {}},
    ]
    launch = Launch.from_document(json.loads(layout.launch.read_text()))
    assert launch.client is not None and launch.client.argv[launch.client.argv.index("--num-prompts") + 1] == "16"
    marks = _marks(layout)
    assert marks[Phase.WARMUP.value] <= marks[Phase.PROFILE_OPEN.value] <= marks[Phase.MEASURED.value]
    assert marks[Phase.FINISHED.value] <= marks[Phase.PROFILE_CLOSE.value] <= marks[Phase.TEARDOWN.value]
    profile = report.profile
    assert profile is not None and profile.trace_dirs == ("profile/torch_trace",) and profile.has_step_traces
    capture = json.loads(layout.profile_capture.read_text())
    assert (capture["window_open_unix"], capture["window_close_unix"]) == (
        profile.window_open_unix,
        profile.window_close_unix,
    )
    assert capture["error"] is None and profile.running_requests_in_window is None


def test_a_fixed_profile_window_the_server_refuses_leaves_no_trace_and_the_round_completes(rig, monkeypatch):
    monkeypatch.setenv("FAKE_PROFILE_REFUSED", "1")
    config = rig.config("r1", profile=_FIXED_WINDOW)
    config = replace(
        config, server=replace(config.server, env_inherit=(*config.server.env_inherit, "FAKE_PROFILE_REFUSED"))
    )

    result = run.run_round(config)

    report, layout = result.report, result.layout
    assert report.run_status is RunStatus.COMPLETED, report.failure_detail
    assert [json.loads(line).get("path") for line in rig.calls.read_text().splitlines()] == [
        None,
        "/start_profile",
        None,
    ]
    assert Phase.PROFILE_OPEN.value not in _marks(layout)
    assert report.profile is not None and not report.profile.has_step_traces and report.profile.trace_dirs == ()
    capture = json.loads(layout.profile_capture.read_text())
    assert capture["window_open_unix"] is None and "/start_profile" in capture["error"]


def _agentx_config(rig, run_id: str, workload: AgentxWorkload, profile: ProfileConfig) -> SuiteConfig:
    agentx_base = replace(rig.base, query=replace(rig.base.query, workload=Workload.AGENTX, isl=None, osl=None))
    rig.base_path.write_text(json.dumps(agentx_base.to_document()))
    config = rig.config(run_id)
    return replace(
        config,
        server=replace(config.server, base=BaseRef(str(rig.base_path), agentx_base.digest())),
        workload=WorkloadKind.AGENTX,
        fixed=None,
        agentx=workload,
        profile=profile,
    )


def test_an_agentx_profile_round_replays_and_opens_its_window_from_the_first_profiling_request(rig):
    window = ProfileConfig(enabled=True, agentx_open_after_sec=0.2, agentx_duration_sec=0.4)
    config = _agentx_config(rig, "r1", replace(_AGENTX, warmup_seed=_AGENTX.seed), window)

    result = run.run_round(config)

    report, layout = result.report, result.layout
    assert report.run_status is RunStatus.COMPLETED, report.failure_detail
    calls = [json.loads(line) for line in rig.calls.read_text().splitlines()]
    (replay,) = [call for call in calls if isinstance(call, list)]
    assert [call["path"] for call in calls if isinstance(call, dict)] == ["/start_profile", "/stop_profile"]
    launch = Launch.from_document(json.loads(layout.launch.read_text()))
    assert launch.client is not None and list(launch.client.argv[1:]) == replay
    assert launch.client.seed == launch.client.warmup_seed == _AGENTX.seed
    assert report.agentx is not None and report.agentx.counts["successful"] == 2 and report.codes.client == 0
    marks = _marks(layout)
    assert marks[Phase.WARMUP.value] < marks[Phase.MEASURED.value] < marks[Phase.FINISHED.value]
    capture = json.loads(layout.profile_capture.read_text())
    assert capture["window_open_unix"] - capture["profiling_start_unix"] >= 0.2
    assert capture["running_requests_in_window"] == 8 and capture["error"] is None
    profile = report.profile
    assert profile is not None and profile.has_step_traces and profile.running_requests_in_window == 8
    assert profile.window_open_unix == capture["window_open_unix"]


def test_an_agentx_window_longer_than_the_replay_closes_when_the_profiling_requests_end(rig):
    window = ProfileConfig(enabled=True, agentx_open_after_sec=0.2, agentx_duration_sec=600)
    config = _agentx_config(rig, "r1", replace(_AGENTX, warmup_seed=_AGENTX.seed), window)

    result = run.run_round(config)

    assert result.report.run_status is RunStatus.COMPLETED, result.report.failure_detail
    finished = _marks(result.layout)[Phase.FINISHED.value]
    capture = json.loads(result.layout.profile_capture.read_text())
    assert finished <= capture["window_close_unix"] < finished + 0.15, "closed at the next poll, not after the tail"


@pytest.mark.parametrize(
    ("changes", "match"),
    [({"warmup_seed": _AGENTX.seed + 1}, "warmup_seed"), ({"warmup_requests_per_lane": 0}, "warmup request")],
)
def test_an_agentx_workload_the_harness_cannot_honour_is_refused_before_the_server_starts(rig, changes, match):
    config = _agentx_config(
        rig, "r1", replace(_AGENTX, **{"warmup_seed": _AGENTX.seed, **changes}), ProfileConfig(enabled=False)
    )

    result = run.run_round(config)

    assert result.report.failed_step is FailedStep.CONFIG_INVALID and match in result.report.failure_detail
    assert not result.layout.launch.exists() and not rig.calls.exists()


_RUN_AGENTIC = """#!/usr/bin/env bash
printf '%s\\n' "$PWD $* $PORT $AGENTIC_CONCURRENCY $AGENTIC_NUM_TRAJECTORIES" >> "$FAKE_CLIENT_CALLS"
for _ in 1 2 3 4; do echo "trajectory done"; sleep 0.2; done
mkdir -p "$RESULTS_DIR/run"
cp "$FAKE_MLPERF_OUTPUT"/* "$RESULTS_DIR/run/"
exit "${FAKE_MLPERF_EXIT:-0}"
"""


def _mlperf_config(rig, run_id: str, monkeypatch) -> SuiteConfig:
    endpoints = rig.tmp / "endpoints"
    script = endpoints / "utility" / "run_agentic.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(_RUN_AGENTIC)
    (rig.tmp / "tokenizer").mkdir(exist_ok=True)
    (rig.tmp / "agentic_combined_v6.jsonl").write_text("")
    monkeypatch.setenv("FAKE_MLPERF_OUTPUT", str(Path(__file__).parent / "fixtures" / "mlperf_agentic_smoke"))
    workload = replace(
        _MLPERF,
        endpoints_dir=str(endpoints),
        tokenizer=str(rig.tmp / "tokenizer"),
        dataset=str(rig.tmp / "agentic_combined_v6.jsonl"),
    )
    config = rig.config(run_id, workload=WorkloadKind.MLPERF, fixed=None, mlperf=workload)
    inherit = (*config.server.env_inherit, "FAKE_MLPERF_OUTPUT", "FAKE_MLPERF_EXIT")
    return replace(config, server=replace(config.server, env_inherit=inherit))


def test_an_mlperf_round_runs_the_harness_against_its_own_server_and_maps_its_result(rig, monkeypatch):
    result = run.run_round(_mlperf_config(rig, "r1", monkeypatch))

    report, layout = result.report, result.layout
    assert report.run_status is RunStatus.COMPLETED, report.failure_detail
    (call,) = rig.calls.read_text().splitlines()
    assert call == f"{rig.tmp / 'endpoints'} smoke_test glm mi355x perf {rig.port} 16 150"
    assert report.mlperf is not None and report.codes.client == 0 and report.concurrency == 16
    assert report.mlperf.result["completed"] == 172
    assert json.loads(layout.inferencex_result.read_text())["accuracy_score"] == pytest.approx(0.7178)
    marks = _marks(layout)
    assert marks[Phase.MEASURED.value] < marks[Phase.FINISHED.value]
    assert not layout.server_pid.exists(), "the round owns its server and takes it down"


def test_a_failing_mlperf_harness_fails_the_round_as_its_client(rig, monkeypatch):
    monkeypatch.setenv("FAKE_MLPERF_EXIT", "2")

    report = run.run_round(_mlperf_config(rig, "r1", monkeypatch)).report

    assert report.failed_step is FailedStep.CLIENT_FAILED and report.codes.client == 2
    assert report.mlperf is None and report.run_status is RunStatus.FAILED


def test_an_mlperf_harness_that_is_not_there_is_refused_before_the_server_starts(rig, monkeypatch):
    config = _mlperf_config(rig, "r1", monkeypatch)
    config = replace(config, mlperf=replace(config.mlperf, endpoints_dir=str(rig.tmp / "nowhere")))

    result = run.run_round(config)

    assert result.report.failed_step is FailedStep.CONFIG_INVALID
    assert "run_agentic.sh" in result.report.failure_detail
    assert not result.layout.launch.exists() and not rig.calls.exists()


def test_a_round_whose_gpus_another_process_holds_is_refused_before_its_server_starts(rig):
    rig.busy_gpus = "GPU 3: held by pid 4242 (python3)"

    result = run.run_round(rig.config("r1"))

    report = result.report
    assert report.run_status is RunStatus.FAILED and report.failed_step is FailedStep.GPUS_UNAVAILABLE
    assert report.failure_detail == "GPU 3: held by pid 4242 (python3)"
    assert result.layout.launch.exists() and not result.layout.server_log.exists() and not rig.calls.exists()


def test_the_refusal_says_whether_the_host_holds_a_redacted_value(rig, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "another")
    differs = run.run_round(rig.config("r1")).report.failure_detail
    monkeypatch.delenv("HF_TOKEN")
    absent = run.run_round(rig.config("r2")).report.failure_detail

    assert differs is not None and "not the one the base was resolved with" in differs
    assert absent is not None and "holds no value" in absent


def test_a_round_writes_only_into_an_empty_run_dir(rig):
    config = rig.config("r1", telemetry=Telemetry(GpuTelemetry(enabled=False, interval_sec=1), kv=True))
    rig.staging = StagingReport(_SHA, {}, ("inferencex at its pin (not staged)",))
    run.run_round(config)

    with pytest.raises(RunDirError):
        run.run_round(config)


def _crashed_parser(*_args, **_kwargs):
    raise FrameworkProbeError("the probe exited 1 with no answer: KeyError: 'libamd_smi.so'")


@pytest.mark.parametrize(
    ("change", "step"),
    [
        ("staging", FailedStep.STAGING_MISSING),
        ("digest", FailedStep.BASE_DIGEST_MISMATCH),
        ("source_files", FailedStep.SOURCE_FILES_CHANGED),
        ("too_old", FailedStep.FRAMEWORK_VERSION_TOO_OLD),
        ("unversioned", FailedStep.FRAMEWORK_VERSION_UNKNOWN),
        ("parser_crashed", FailedStep.FRAMEWORK_VERSION_UNKNOWN),
        ("redacted", FailedStep.CONFIG_INVALID),
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
    elif change == "parser_crashed":
        monkeypatch.setattr(parser_check, "check", _crashed_parser)
    else:
        monkeypatch.setenv("HF_TOKEN", "another")

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
    config = rig.config("ignored", telemetry=Telemetry(GpuTelemetry(enabled=False, interval_sec=1), kv=True))
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


def test_an_external_round_measures_a_server_it_neither_starts_nor_holds_to_the_local_install(rig):
    pid_dir = rig.tmp / "pids"
    boot = rig.config(
        "boot", workload=WorkloadKind.NONE, fixed=None, lifecycle=Lifecycle(LifecycleMode.BOOT, str(pid_dir))
    )
    assert run.run_round(boot).report.run_status is RunStatus.COMPLETED
    rig.installed = "0.0.1"
    rig.refusal = FlagRefusal((RefusedFlag("--disable-radix-cache", LayerSource.BASE),), "unrecognized arguments")

    result = run.run_round(rig.config("external", lifecycle=Lifecycle(LifecycleMode.EXTERNAL)))

    report, layout = result.report, result.layout
    assert report.run_status is RunStatus.COMPLETED, report.failure_detail
    assert report.fixed is not None and report.server is not None and report.server.reused
    assert report.server.log_range is None and not layout.server_log.exists()
    launch = Launch.from_document(json.loads(layout.launch.read_text()))
    assert launch.framework.version_check is VersionCheck.NOT_HELD
    assert (pid_dir / "server.pid").exists()
    assert not runtime_server.stop(pid_dir).survivors


def _scriptable_round(rig, run_id: str, body: str, *, profile: bool = False) -> SuiteConfig:
    script = rig.tmp / f"{run_id}.sh"
    script.write_text(body)
    query = LaunchQuery("FLUX.1-dev", "custom", "mi355x", Workload.SCRIPTABLE, (1,))
    base = scriptable_base(query, str(script), LaunchModel("FLUX.1-dev"), Topology(tp=1))
    base_path = rig.tmp / f"{run_id}_base.json"
    base_path.write_text(json.dumps(base.to_document()))
    config = rig.config(run_id, workload=WorkloadKind.SCRIPTABLE, fixed=None, scriptable=ScriptableWorkload({"N": 2}))
    server = replace(config.server, base=BaseRef(str(base_path), base.digest()))
    return replace(config, server=server, profile=ProfileConfig(enabled=profile))


_TRACING_SCRIPT = """
mkdir -p "$VLLM_TORCH_PROFILER_DIR" && echo trace > "$VLLM_TORCH_PROFILER_DIR/rank0.json"
echo "{\\"output_throughput\\": $N, \\"profile\\": \\"$PROFILE\\"}" > "$RESULT_DIR/$RESULT_FILENAME.json"
"""


def test_a_scriptable_profile_round_runs_its_base_script_and_keeps_the_trace_it_wrote(rig):
    result = run.run_round(_scriptable_round(rig, "s1", _TRACING_SCRIPT, profile=True))

    report = result.report
    assert report.run_status is RunStatus.COMPLETED, report.failure_detail
    assert report.scriptable is not None and report.scriptable.result == {"output_throughput": 2, "profile": "1"}
    assert report.profile is not None and report.profile.has_step_traces
    assert report.profile.window_open_unix <= report.profile.window_close_unix
    assert report.server is None and report.launch_digest is None and not result.layout.launch.exists()


def test_a_scriptable_round_on_a_serving_base_is_refused(rig):
    config = rig.config("s2", workload=WorkloadKind.SCRIPTABLE, fixed=None, scriptable=ScriptableWorkload())

    report = run.run_round(config).report

    assert report.failed_step is FailedStep.CONFIG_INVALID and "needs a scriptable base" in report.failure_detail
