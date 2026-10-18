# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A stand-in for the measurement runner, started as a local sandbox template's command.

It speaks the runner's file protocol (``/in/READY``, ``/in/spec.json``, ``/out/progress.json``, ``/out/files``,
``/out/manifest.json``, ``/out/manifest.sig``, ``/out/status.json``) and signs with the key the sandbox mounted, but it
measures nothing: what it writes is chosen by ``fake_mode`` and friends in the YAML of the sheet it receives.

``fake_mode``: ``ok`` (default), ``fail_test`` (non-zero return code with a known failure signature), ``fail_unknown``
(non-zero, no signature), ``timed_out``, ``spec_rejected`` and ``node_fault`` (``status.json`` failures), ``tamper``
(a listed file changes after signing), ``wrong_key`` (signed with another key), ``other_sandbox``, ``other_request``,
``other_spec`` (the manifest names another sandbox, request or sheet), ``extra_file`` (a file the manifest does not
list), ``no_manifest`` and ``hang`` (never finishes). ``fake_delay`` seconds pass before the work starts;
``fake_throughput`` and ``fake_completed`` set the report. ``fake_accuracy`` adds an accuracy round to the rounds file
with that score (``none``: a round that holds no score). ``fake_trace: true`` adds the torch-profiler trace a profile
test leaves (``profile/torch_trace/...``); ``fake_report: false`` leaves out the benchmark report, as a profiled round
that printed no readable measurement.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from hyperloom.measure import signing
from hyperloom.measure.manifest import RESULT_SCHEMA, encode_manifest
from hyperloom.measure.spec import MeasurementSpec, fingerprint, spec_sha256

REPORT_PATH = "benchmark_fake_20261014_000000/benchmark_report.json"
TRACE_PATH = "profile/torch_trace/rank0.pt.trace.json.gz"


def _atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_name(f".{path.name}.tmp")
    scratch.write_bytes(data)
    os.replace(scratch, path)


def _runner_facts(key: Any, sandbox_id: str, mode: str) -> dict[str, Any]:
    public_pem = signing.public_key_pem(key)
    return {
        "sandbox_id": "sbx-other" if mode == "other_sandbox" else sandbox_id,
        "signing_key_fingerprint": signing.public_key_fingerprint(public_pem),
        "node_name": os.environ.get("NODE_NAME"),
        "image_digest": "sha256:" + "11" * 32,
        "gpu_model": "FakeGPU",
        "power_cap_w": 500,
        "partition_mode": "SPX/NPS1",
        "hyperloom_version": _version(),
        "magpie_commit": "e" * 40,
        "inferencex_commit": "f" * 40,
        "gpu_ids": [int(i) for i in os.environ.get("ROCR_VISIBLE_DEVICES", "").split(",") if i.strip().isdigit()],
        "ecc_delta": {"correctable": 0, "uncorrectable": 0},
    }


def _version() -> str:
    from importlib import metadata

    try:
        return metadata.version("hyperloom")
    except metadata.PackageNotFoundError:
        return "0"


def _write_files(out: Path, config: dict[str, Any], mode: str) -> dict[str, bytes]:
    report = {
        "success": mode != "fail_test",
        "throughput": {
            "output_throughput": float(config.get("fake_throughput", 5000.0)),
            "completed_requests": int(config.get("fake_completed", 64)),
            "num_prompts": int(config.get("fake_requested", 64)),
        },
    }
    files = {"server.log": b"server up\nserver down\n"}
    if config.get("fake_report", True):
        files[REPORT_PATH] = json.dumps(report).encode()
    if config.get("fake_trace"):
        files[TRACE_PATH] = b"fake trace bytes"
    if config.get("fake_warmup") and REPORT_PATH in files:
        files["warmup_round/benchmark_fake_20261014_000000/benchmark_report.json"] = files[REPORT_PATH]
    if mode == "fail_test":
        files["server.log"] = b"ModuleNotFoundError: No module named 'sglang_kernel'\n"
    if mode == "fail_unknown":
        files["server.log"] = b"something odd happened\n"
    rounds = [{"point": 0, "phase": "measure", "dir": REPORT_PATH.rsplit("/", 1)[0], "returncode": 0}]
    if config.get("fake_inventory"):
        files["farm_inventory.json"] = json.dumps(config["fake_inventory"]).encode()
    if "fake_accuracy" in config:
        score = config["fake_accuracy"]
        values = [] if score is None else [{"task": "gsm8k", "metric": "exact_match,flexible-extract", "score": score}]
        if "fake_accuracy_version" in config:
            values = [{**value, "version": config["fake_accuracy_version"]} for value in values]
        rounds.append({**rounds[0], "phase": "accuracy", "accuracy": values})
    if config.get("fake_rounds") or "fake_accuracy" in config:
        files["rounds.json"] = json.dumps(rounds).encode()
    for relative, data in files.items():
        _atomic(out / "files" / relative, data)
    return files


def _manifest(spec: MeasurementSpec, raw: bytes, key: Any, files: dict[str, bytes], mode: str) -> dict[str, Any]:
    failing = mode in ("fail_test", "fail_unknown", "timed_out")
    document = {
        "schema": RESULT_SCHEMA,
        "request_id": "other-request" if mode == "other_request" else spec.request_id,
        "spec_fingerprint": ("0" * 16) if mode == "other_spec" else fingerprint(spec, hyperloom_version=_version()),
        "spec_sha256": spec_sha256(raw),
        "surface_sha256": None,
        "runner": _runner_facts(key, os.environ["E2B_SANDBOX_ID"], mode),
        "outcome": {
            "returncode": 1 if failing else 0,
            "timed_out": mode == "timed_out",
            "error_class": None,
            "failure_class": "unclear" if mode in ("fail_unknown", "timed_out") else None,
            "failure_reason": (
                "no_known_signature" if mode == "fail_unknown" else "timeout" if mode == "timed_out" else None
            ),
            "stdout_tail": "ran",
            "stderr_tail": "ModuleNotFoundError: x" if mode == "fail_test" else "",
        },
        "platform_env_changed": [],
        "timing": {
            "runner_started_unix": 1.0,
            "spec_received_unix": 2.0,
            "server_ready_unix": 3.0,
            "finished_unix": 4.0,
        },
        "files": [
            {"path": path, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for path, data in sorted(files.items())
        ],
    }
    if mode == "fail_test":
        document["outcome"]["failure_class"] = "candidate"
        document["outcome"]["failure_reason"] = "import_error"
    return document


def _publish(out: Path, spec: MeasurementSpec, raw: bytes, config: dict[str, Any], mode: str) -> None:
    key = signing.load_private_key(Path(os.environ["HYPERLOOM_SIGNING_KEY_FILE"]))
    files = _write_files(out, config, mode)
    manifest = _manifest(spec, raw, key, files, mode)
    manifest_raw = encode_manifest(manifest)
    signer = signing.generate_private_key() if mode == "wrong_key" else key
    if mode == "tamper":
        original = bytearray((out / "files" / "server.log").read_bytes())
        original[0] ^= 0x01
        (out / "files" / "server.log").write_bytes(bytes(original))
    if mode == "extra_file":
        _atomic(out / "files" / "extra.txt", b"not listed")
    if mode != "no_manifest":
        _atomic(out / "manifest.json", manifest_raw)
        _atomic(out / "manifest.sig", signing.sign(manifest_raw, signer).encode("ascii"))
    _atomic(out / "status.json", json.dumps({"state": "succeeded", "error_class": None, "message": ""}).encode())


def main() -> int:
    root = Path(os.environ["HYPERLOOM_SANDBOX_ROOT"])
    inbox, out = root / "in", root / "out"
    _atomic(out / "progress.json", json.dumps({"phase": "waiting_spec", "last_output_unix": 0.0}).encode())
    while not (inbox / "READY").exists():
        time.sleep(0.02)
    raw = (inbox / "spec.json").read_bytes()
    spec = MeasurementSpec.from_bytes(raw)
    config = yaml.safe_load(spec.config[0].yaml) or {}
    mode = str(config.get("fake_mode", "ok"))
    _atomic(out / "progress.json", json.dumps({"phase": "running", "last_output_unix": time.time()}).encode())
    time.sleep(float(config.get("fake_delay", 0)))
    if mode == "hang":
        time.sleep(3600)
    if mode in ("spec_rejected", "node_fault"):
        state = {"state": "failed", "error_class": mode, "message": f"fake {mode}"}
        _atomic(out / "files" / "server.log", b"partial log\n")
        _atomic(out / "status.json", json.dumps(state).encode())
        return 0
    _publish(out, spec, raw, config, mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
