# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build signed result bundles and instruction sheets for the measurement tests."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom.measure import signing
from hyperloom.measure.bundle import BundleExpectation
from hyperloom.measure.manifest import RESULT_SCHEMA, encode_manifest

REQUEST_ID = "6c0e0000-0000-5000-8000-000000000001"
SPEC_FINGERPRINT = "a1b2c3d4e5f60718"
SPEC_SHA256 = "ab" * 32
SANDBOX_ID = "sbx_01JTEST"
DEFAULT_FILES = {
    "benchmark_sglang_20261005_101500/benchmark_report.json": b'{"output_throughput": 5123.4}\n',
    "server.log": b"server up\nserver down\n" * 50,
}


@dataclass
class Bundle:
    staging: Path
    key: Any
    public_pem: str
    document: dict[str, Any]

    def write_manifest(self, document: dict[str, Any] | None = None, *, key: Any = None) -> bytes:
        """Write ``manifest.json`` and a signature over exactly those bytes (by ``key``, default the sandbox key)."""
        raw = encode_manifest(document if document is not None else self.document)
        (self.staging / "manifest.json").write_bytes(raw)
        (self.staging / "manifest.sig").write_text(signing.sign(raw, key or self.key), encoding="ascii")
        return raw


def manifest_document(files: dict[str, bytes], public_pem: str, **overrides: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "request_id": REQUEST_ID,
        "spec_fingerprint": SPEC_FINGERPRINT,
        "spec_sha256": SPEC_SHA256,
        "surface_sha256": "cd" * 32,
        "runner": {
            "sandbox_id": SANDBOX_ID,
            "signing_key_fingerprint": signing.public_key_fingerprint(public_pem),
            "node_name": "node-17",
            "image_digest": "sha256:" + "11" * 32,
            "gpu_model": "MI355X",
            "power_cap_w": 1400,
            "partition_mode": "SPX/NPS1",
            "hyperloom_version": "1.1.3+gtest",
            "magpie_commit": "e" * 40,
            "inferencex_commit": "f" * 40,
            "gpu_ids": [0, 1],
            "ecc_delta": {"correctable": 0, "uncorrectable": 0},
        },
        "outcome": {
            "returncode": 0,
            "timed_out": False,
            "error_class": None,
            "failure_class": None,
            "failure_reason": None,
            "stdout_tail": "done",
            "stderr_tail": "",
        },
        "platform_env_changed": ["MASTER_ADDR"],
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
    document.update(copy.deepcopy(overrides))
    return document


def make_bundle(staging: Path, files: dict[str, bytes] | None = None, *, key: Any = None, **overrides: Any) -> Bundle:
    """A bundle that verifies: files on disk, manifest listing them, signature by the sandbox key."""
    files = dict(DEFAULT_FILES if files is None else files)
    key = key or signing.generate_private_key()
    public_pem = signing.public_key_pem(key)
    staging.mkdir(parents=True, exist_ok=True)
    for relative, data in files.items():
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    bundle = Bundle(staging, key, public_pem, manifest_document(files, public_pem, **overrides))
    bundle.write_manifest()
    return bundle


def verify_args(bundle: Bundle) -> BundleExpectation:
    return BundleExpectation(
        public_key_pem=bundle.public_pem,
        request_id=REQUEST_ID,
        spec_fingerprint=SPEC_FINGERPRINT,
        spec_sha256=SPEC_SHA256,
        sandbox_id=SANDBOX_ID,
    )


def flip_one_byte(path: Path, offset: int = 0) -> None:
    data = bytearray(path.read_bytes())
    data[offset] ^= 0x01
    path.write_bytes(bytes(data))


def point_yaml(*, conc: int, port: int = 8000) -> str:
    return (
        "benchmark:\n  model: m\n  envs:\n"
        f'    CONC: "{conc}"\n    NUM_PROMPTS: "{conc * 10}"\n    RUN_EVAL: "false"\n'
        f"server:\n  pid_dir: /shared/u1/sess-a/pids\n  port: {port}\ninferencex_path: /shared/u1/sess-a/ix\n"
    )


def point_document(yaml_text: str) -> dict[str, Any]:
    return {"yaml": yaml_text, "sha256": hashlib.sha256(yaml_text.encode()).hexdigest()}


def spec_document() -> dict[str, Any]:
    yaml_text = (
        "model: m\nserver:\n  pid_dir: /shared/u1/sess-a/pids\n  port: 8000\ninferencex_path: /shared/u1/sess-a/ix\n"
    )
    return {
        "schema": "hyperloom.measurement-spec/v1",
        "request_id": REQUEST_ID,
        "kind": "test",
        "meta": {"purpose": "explore", "priority": 2, "branch_id": "b1", "parent_commit": "c0ffee", "retest_of": None},
        "hardware": {"gpu_type": "MI355X", "gpu_count": 1, "power_cap_w": 1400, "partition_mode": "SPX/NPS1"},
        "image": {"ref": "registry/fw:1", "digest": "sha256:" + "22" * 32},
        "config": [point_document(yaml_text)],
        "attachments": [{"id": "csv1", "kind": "file", "blob": "sha256:" + "33" * 32, "manifest_sha256": "44" * 32}],
        "process_env": {"AITER_LOG_LEVEL": "WARNING"},
        "platform_env": {"NCCL_IB_SL": "3"},
        "unset_envs": ["PYTHONPATH"],
        "tools": {"magpie": {"commit": "a" * 40}, "inferencex": {"commit": "b" * 40}},
        "environment": {
            "roots": [
                {
                    "id": "sglang",
                    "kind": "package",
                    "import_name": "sglang",
                    "dist": {"name": "sglang", "version": "0.5.4"},
                    "entries": [
                        {"path": "srt/a.py", "op": "upsert", "blob": "sha256:" + "55" * 32, "mode": "0644"},
                        {"path": "srt/old.py", "op": "delete"},
                    ],
                }
            ],
            "runtime": {"snapshot": "/shared/u1/sess-a/hosted/runtimes/rt-1", "manifest_sha256": "66" * 32},
            "blob_root": "/shared/u1/sess-a/hosted/blobs",
        },
        "model_hashes": {"config.json": "77" * 32},
        "procedure": {
            "prep": "grid",
            "benchmark_backend": "magpie",
            "rounds": {"warmup": 1, "measure": 3},
            "accuracy": False,
            "timeout_sec": 3600,
            "silence_timeout_sec": 600,
            "deadline_unix": 1790000000.0,
        },
        "outputs": {"result_dir": "results", "max_total_bytes": 4294967296},
    }


def spec_json(document: dict[str, Any]) -> bytes:
    return json.dumps(document).encode()
