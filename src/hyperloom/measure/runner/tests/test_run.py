# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The runner's main flow with the measurement backend, the tool checkouts and the GPU queries stubbed."""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest
import yaml

from hyperloom.measure import signing
from hyperloom.measure.bundle import BundleExpectation, verify_bundle
from hyperloom.measure.manifest import ResultManifest
from hyperloom.measure.runner import run as runner
from hyperloom.measure.returncodes import SERVER_BOOT_TIMEOUT_RC
from hyperloom.measure.runner.hardware import EccCounts, GpuFacts
from hyperloom.measure.socket import RoundOutcome, TestCall, TestOutcome
from hyperloom.measure.spec import MeasurementSpec
from hyperloom.measure.tests.bundle_factory import REQUEST_ID, spec_document
from hyperloom.contracts.measurement import SandboxFacts

POINT = {"benchmark": {"model": "/m", "framework": "sglang", "envs": {"TP": 1, "CONC": 4, "RUN_EVAL": "false"}}}


def _sheet(**changes) -> dict:
    document = spec_document()
    text = yaml.safe_dump(POINT)
    import hashlib

    document["config"] = [{"yaml": text, "sha256": hashlib.sha256(text.encode()).hexdigest()}]
    document["attachments"] = []
    document["environment"] = {"roots": [], "runtime": None, "blob_root": "/b"}
    document["model_hashes"] = {}
    document["unset_envs"] = ["STALE_PIN"]
    document["procedure"].update(
        {"benchmark_backend": "bypass", "rounds": {"warmup": 1, "measure": 1}, "deadline_unix": 0}
    )
    document.update(changes)
    return document


class _Backend:
    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.calls: list[TestCall] = []

    def __call__(self):
        return self

    def run_test(self, call: TestCall) -> TestOutcome:
        self.calls.append(call)
        return self.behaviour(call)


def _ok(call: TestCall) -> TestOutcome:
    (call.output_dir / "server.log").write_text("log")
    rounds = (RoundOutcome(0, "measure", call.output_dir, 0),)
    return TestOutcome(0, "[]", "", rounds, SandboxFacts(sandbox_id="x"), request_id=call.task_id)


@pytest.fixture
def key():
    return signing.generate_private_key()


@pytest.fixture
def env(tmp_path, key, monkeypatch):
    key_file = tmp_path / "key.pem"
    key_file.write_bytes(signing.private_key_pem(key))
    environ = {
        "HYPERLOOM_SIGNING_KEY_FILE": str(key_file),
        "NODE_NAME": "node-1",
        "E2B_SANDBOX_ID": "sbx-1",
        "ROCR_VISIBLE_DEVICES": "0",
        "MASTER_ADDR": "x",
        "PET_RANK": "0",
        "NCCL_IB_SL": "9",
    }
    monkeypatch.setattr(runner, "ensure_inferencex", lambda commit, cache: tmp_path / "ix")
    monkeypatch.setattr(runner.hardware, "read_gpu_facts", lambda visible: None)
    monkeypatch.setattr(runner.hardware, "ecc_counts", lambda visible: None)
    return environ


def _go(tmp_path, env, sheet: dict | bytes | None, backend=None, **kwargs) -> int:
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    in_dir.mkdir(exist_ok=True)
    if sheet is not None:
        raw = sheet if isinstance(sheet, bytes) else json.dumps(sheet).encode()
        (in_dir / "spec.json").write_bytes(raw)
        (in_dir / "READY").write_text("")
    return runner.run(in_dir, out_dir, scratch=tmp_path / "scratch", environ=env, spec_wait_sec=0.05, **kwargs)


def _status(tmp_path) -> dict:
    return json.loads((tmp_path / "out" / "status.json").read_text())


def test_no_sheet_ends_with_no_spec_and_exit_zero(tmp_path, env):
    assert _go(tmp_path, env, None) == 0
    assert _status(tmp_path)["error_class"] == "no_spec"


def test_a_sheet_that_does_not_parse_is_spec_rejected(tmp_path, env):
    assert _go(tmp_path, env, b"{not json") == 3
    assert _status(tmp_path)["error_class"] == "spec_rejected"
    assert not (tmp_path / "out" / "manifest.json").exists()


def test_a_missing_signing_key_stops_the_runner_at_start(tmp_path, env):
    del env["HYPERLOOM_SIGNING_KEY_FILE"]

    assert _go(tmp_path, env, _sheet()) == 2
    assert _status(tmp_path)["error_class"] == "runner_start_failed"


def test_a_successful_test_leaves_a_signed_manifest_and_status_last(tmp_path, env, key, monkeypatch):
    backend = _Backend(_ok)
    monkeypatch.setattr(runner, "LocalMeasurementBackend", backend)

    assert _go(tmp_path, env, _sheet()) == 0

    out = tmp_path / "out"
    raw = (out / "manifest.json").read_bytes()
    staging = tmp_path / "staging"  # what the reader assembles: the result files with the manifest and signature
    shutil.copytree(out / "files", staging)
    shutil.copy(out / "manifest.json", staging)
    shutil.copy(out / "manifest.sig", staging)
    verify_bundle(
        staging,
        BundleExpectation(
            public_key_pem=signing.public_key_pem(key),
            request_id=REQUEST_ID,
            spec_fingerprint=ResultManifest.from_bytes(raw).spec_fingerprint,
            spec_sha256=None,
            sandbox_id="sbx-1",
        ),
    )
    assert signing.verify(raw, (out / "manifest.sig").read_text(), signing.public_key_pem(key))
    manifest = ResultManifest.from_bytes(raw)
    assert manifest.request_id == REQUEST_ID and manifest.outcome.returncode == 0
    assert sorted(f.path for f in manifest.files) == ["farm_inventory.json", "rounds.json", "server.log"]
    assert json.loads((out / "files" / "rounds.json").read_text()) == [
        {"point": 0, "phase": "measure", "dir": ".", "returncode": 0}
    ]
    assert manifest.runner.sandbox_id == "sbx-1" and manifest.runner.node_name == "node-1"
    assert manifest.platform_env_changed == ("NCCL_IB_SL",)
    assert _status(tmp_path) == {"state": "succeeded", "error_class": None, "message": ""}


def test_the_call_is_built_from_the_sheet_and_the_environment_is_the_sandboxes(tmp_path, env, monkeypatch):
    backend = _Backend(_ok)
    monkeypatch.setattr(runner, "LocalMeasurementBackend", backend)

    _go(tmp_path, env, _sheet())

    (call,) = backend.calls
    assert call.task_id == REQUEST_ID and call.prep == "grid" and call.kind == "test" and not call.accuracy
    assert call.rounds.warmup_rounds == 1
    assert call.unset_envs == ("STALE_PIN",) and dict(call.env_overrides) == {"AITER_LOG_LEVEL": "WARNING"}
    assert call.session_facts.inferencex_path == str(tmp_path / "ix")
    rewritten = yaml.safe_load(call.points[0].read_text())["benchmark"]
    assert rewritten["inferencex_path"] == str(tmp_path / "ix")
    assert rewritten["envs"]["ROCR_VISIBLE_DEVICES"] == "0"
    assert env["NCCL_IB_SL"] == "3" and "MASTER_ADDR" not in env and "PET_RANK" not in env
    assert "HYPERLOOM_SIGNING_KEY_FILE" not in env and env["HYPERLOOM_BENCHMARK_BACKEND"] == "bypass"


def test_a_failed_test_is_a_signed_result_with_its_failure_class(tmp_path, env, monkeypatch):
    def failing(call):
        return TestOutcome(1, "", "CUDA out of memory: HIP out of memory", (), SandboxFacts(sandbox_id="x"))

    monkeypatch.setattr(runner, "LocalMeasurementBackend", _Backend(failing))

    assert _go(tmp_path, env, _sheet()) == 0

    outcome = ResultManifest.from_bytes((tmp_path / "out" / "manifest.json").read_bytes()).outcome
    assert outcome.returncode == 1 and (outcome.failure_class or outcome.error_class)
    assert _status(tmp_path)["state"] == "succeeded"


def test_the_cause_the_backend_read_in_the_logs_is_what_the_signed_result_records(tmp_path, env, monkeypatch):
    reason = (
        "serve_flag: launch_server.py: error: ambiguous option: --chunked-prefill could match --chunked-prefill-size"
    )

    def refused_flag(call):
        return TestOutcome(
            1, "[]", "", (), SandboxFacts(sandbox_id="x"), failure_class="candidate", failure_reason=reason
        )

    monkeypatch.setattr(runner, "LocalMeasurementBackend", _Backend(refused_flag))

    assert _go(tmp_path, env, _sheet()) == 0

    outcome = ResultManifest.from_bytes((tmp_path / "out" / "manifest.json").read_bytes()).outcome
    assert (outcome.failure_class, outcome.failure_reason, outcome.error_class) == ("candidate", reason, None)


def test_a_boot_timeout_is_recorded_as_the_candidates_failure_with_its_return_code(tmp_path, env, monkeypatch):
    reason = "server_boot_timeout: the server was not ready within 1800s of the 3600s a round may take"

    def slow_boot(call):
        return TestOutcome(
            SERVER_BOOT_TIMEOUT_RC,
            "[]",
            "",
            (),
            SandboxFacts(sandbox_id="x"),
            failure_class="candidate",
            failure_reason=reason,
        )

    monkeypatch.setattr(runner, "LocalMeasurementBackend", _Backend(slow_boot))

    assert _go(tmp_path, env, _sheet()) == 0

    outcome = ResultManifest.from_bytes((tmp_path / "out" / "manifest.json").read_bytes()).outcome
    assert (outcome.returncode, outcome.failure_class, outcome.failure_reason) == (
        SERVER_BOOT_TIMEOUT_RC,
        "candidate",
        reason,
    )


def test_a_timeout_is_recorded_as_timed_out(tmp_path, env, monkeypatch):
    def slow(call):
        raise subprocess.TimeoutExpired("t", 1)

    monkeypatch.setattr(runner, "LocalMeasurementBackend", _Backend(slow))

    _go(tmp_path, env, _sheet())

    outcome = ResultManifest.from_bytes((tmp_path / "out" / "manifest.json").read_bytes()).outcome
    assert outcome.timed_out and outcome.failure_class == "unclear" and outcome.failure_reason == "timeout"


def test_a_call_the_local_backend_refuses_is_spec_rejected(tmp_path, env, monkeypatch):
    def refuse(call):
        raise runner.TestCallRefused("RUN_EVAL contradicts the call")

    monkeypatch.setattr(runner, "LocalMeasurementBackend", _Backend(refuse))

    assert _go(tmp_path, env, _sheet()) == 3
    assert "RUN_EVAL contradicts" in _status(tmp_path)["message"]


def test_hardware_that_differs_from_the_sheet_is_a_preflight_mismatch(tmp_path, env, monkeypatch):
    monkeypatch.setattr(
        runner.hardware, "read_gpu_facts", lambda visible: [GpuFacts(0, "AMD Instinct MI300X", 750.0, None)]
    )
    backend = _Backend(_ok)
    monkeypatch.setattr(runner, "LocalMeasurementBackend", backend)

    assert _go(tmp_path, env, _sheet()) == 4
    assert _status(tmp_path)["error_class"] == "preflight_mismatch" and backend.calls == []


def test_new_uncorrectable_ecc_errors_are_a_node_fault_and_no_result(tmp_path, env, monkeypatch):
    counts = iter([{0: EccCounts(0, 0)}, {0: EccCounts(0, 2)}])
    monkeypatch.setattr(runner.hardware, "ecc_counts", lambda visible: next(counts))
    monkeypatch.setattr(runner, "LocalMeasurementBackend", _Backend(_ok))

    assert _go(tmp_path, env, _sheet()) == 5
    assert _status(tmp_path)["error_class"] == "node_fault"
    assert not (tmp_path / "out" / "manifest.json").exists()


def test_a_symlink_in_the_output_is_refused_and_nothing_is_signed(tmp_path, env, monkeypatch):
    def planted(call):
        (call.output_dir / "link").symlink_to("/etc/passwd")
        return _ok(call)

    monkeypatch.setattr(runner, "LocalMeasurementBackend", _Backend(planted))

    assert _go(tmp_path, env, _sheet()) == 1
    assert _status(tmp_path)["error_class"] == "unsafe_output"
    assert not (tmp_path / "out" / "manifest.json").exists()


def test_the_replay_clients_installation_in_an_agentx_workspace_is_not_part_of_the_signed_output(
    tmp_path, env, monkeypatch
):
    def agentx(call):
        workspace = call.output_dir / "variant_00_test" / "benchmark_sglang_1"
        (workspace / ".agentx-runtime" / "venv" / "bin").mkdir(parents=True)
        (workspace / ".agentx-runtime" / "venv" / "bin" / "python").symlink_to("/usr/bin/python3")
        (workspace / "benchmark_report.json").write_text("{}")
        return _ok(call)

    monkeypatch.setattr(runner, "LocalMeasurementBackend", _Backend(agentx))

    assert _go(tmp_path, env, _sheet()) == 0

    files = sorted(f.path for f in ResultManifest.from_bytes((tmp_path / "out" / "manifest.json").read_bytes()).files)
    assert "variant_00_test/benchmark_sglang_1/benchmark_report.json" in files
    assert not any(".agentx-runtime" in path for path in files)
    assert not list((tmp_path / "out" / "files").rglob(".agentx-runtime"))


def test_a_link_where_the_replay_clients_installation_belongs_is_still_refused(tmp_path, env, monkeypatch):
    def planted(call):
        (call.output_dir / ".agentx-runtime").symlink_to("/etc")
        return _ok(call)

    monkeypatch.setattr(runner, "LocalMeasurementBackend", _Backend(planted))

    assert _go(tmp_path, env, _sheet()) == 1
    assert _status(tmp_path)["error_class"] == "unsafe_output"


def test_the_sheet_validates_as_the_writer_produces_it():
    MeasurementSpec.from_document(_sheet())


def _repro_sheet(tmp_path, kind: str) -> dict:
    from hyperloom.measure.tests.repro_fixtures import case_attachment

    class _Blobs:  # the sandbox's /in/blobs
        def put(self, data: bytes) -> str:
            import hashlib

            name = hashlib.sha256(data).hexdigest()
            (tmp_path / "in" / "blobs").mkdir(parents=True, exist_ok=True)
            (tmp_path / "in" / "blobs" / name).write_bytes(data)
            return name

    case = case_attachment(_Blobs(), "exit 0\n")
    attachment = {"id": case.id, "kind": "dir", "blob": f"sha256:{case.sha256}", "manifest_sha256": "1" * 64}
    return _sheet(kind=kind, attachments=[attachment])


def test_a_repro_call_carries_its_attachments_already_unpacked_under_the_work_directory(tmp_path, env, monkeypatch):
    seen: list[bool] = []

    def observing(call: TestCall) -> TestOutcome:
        seen.append((tmp_path / "scratch" / "test" / "attach" / "case-1" / "repro.sh").is_file())
        return _ok(call)

    backend = _Backend(observing)
    monkeypatch.setattr(runner, "LocalMeasurementBackend", backend)
    (tmp_path / "in").mkdir()

    _go(tmp_path, env, _repro_sheet(tmp_path, "repro"))

    (call,) = backend.calls
    assert call.kind == "repro" and [(a.id, a.kind) for a in call.attachments] == [("case-1", "dir")]
    assert call.cwd == str(tmp_path / "scratch" / "test") and seen == [True]


def test_only_a_repro_call_is_told_which_attachments_are_its_case(tmp_path, env, monkeypatch):
    backend = _Backend(_ok)
    monkeypatch.setattr(runner, "LocalMeasurementBackend", backend)
    (tmp_path / "in").mkdir()

    _go(tmp_path, env, _repro_sheet(tmp_path, "test"))

    assert backend.calls[0].attachments == ()


def test_the_signing_key_file_is_gone_before_the_test_runs_and_the_result_is_still_signed(
    tmp_path, env, key, monkeypatch
):
    key_file = tmp_path / "key.pem"
    seen: list[bool] = []

    def observing(call: TestCall) -> TestOutcome:
        seen.append(key_file.exists())
        return _ok(call)

    monkeypatch.setattr(runner, "LocalMeasurementBackend", _Backend(observing))

    assert _go(tmp_path, env, _sheet()) == 0

    assert seen == [False], "what the test runs must not be able to read the key"
    out = tmp_path / "out"
    assert signing.verify(
        (out / "manifest.json").read_bytes(), (out / "manifest.sig").read_text(), signing.public_key_pem(key)
    )


def test_a_signing_key_file_that_cannot_be_removed_stops_the_runner_at_start(tmp_path, env, monkeypatch):
    backend = _Backend(_ok)
    monkeypatch.setattr(runner, "LocalMeasurementBackend", backend)

    def refuse(self, missing_ok=False):
        raise PermissionError("read-only")

    monkeypatch.setattr(runner.Path, "unlink", refuse)

    assert _go(tmp_path, env, _sheet()) != 0

    assert _status(tmp_path)["error_class"] == "runner_start_failed" and backend.calls == []
