# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build signed result bundles and instruction sheets for the measurement tests."""

from __future__ import annotations

import copy
import functools
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom.measure import signing
from hyperloom.measure.bundle import BundleExpectation
from hyperloom.measure.manifest import RESULT_SCHEMA, encode_manifest
from hyperloom.measure.runner.tools import INFERENCE_SUITE_DIR
from inference_suite import LaunchQuery, SourceSnapshot, resolve_base, staging_key
from inference_suite.contracts import BaseLaunch, NormalisedManifest, PackageIdentity, Workload
from inference_suite.pins import INFERENCEX_CLIENT_COMMIT, INFERENCEX_TOOL, LM_EVAL_VERSION, LM_EVAL_WHEEL_SHA256
from inference_suite.recipe.resolve import UnreadSource
from inference_suite.client_env import BUILD_FILE, CLIENT_DIR, STAGED_LOCK
from inference_suite.staging import INFERENCEX_REQUIRED, MANIFEST_FILE, StagedTool, StagingManifest, tool_root

REQUEST_ID = "6c0e0000-0000-5000-8000-000000000001"
SPEC_FINGERPRINT = "a1b2c3d4e5f60718"
SPEC_SHA256 = "ab" * 32
SANDBOX_ID = "sbx_01JTEST"
DEFAULT_FILES = {
    "benchmark_sglang_20000101_000000/benchmark_report.json": b'{"output_throughput": 5000.0}\n',
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
            "inference_suite_staging": "e" * 64,
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


#: What :func:`stage_into` stages below the staged InferenceX root, by relative path: the files a load runs, and a
#: client environment whose build record, lock and one installed distribution give :data:`STAGING`.
_HARNESS_SHA256 = "c" * 64
_STAGED_FILES = {
    **dict.fromkeys(INFERENCEX_REQUIRED, b"staged\n"),
    BUILD_FILE: json.dumps({"version": "3.12.15", "sha256": "b" * 64, "bundled": ["pip"]}).encode(),
    STAGED_LOCK: f"agentx-harness==1.0.13 --hash=sha256:{_HARNESS_SHA256}\n".encode(),
    f"{CLIENT_DIR}/lib/python3.12/site-packages/agentx_harness-1.0.13.dist-info/RECORD": b"agentx_cli.py,,\n",
    f"{CLIENT_DIR}/lib/python3.12/site-packages/pip-26.2.2.dist-info/RECORD": b"pip/__init__.py,,\n",
}
STAGING_MANIFEST = StagingManifest(
    {
        INFERENCEX_TOOL: StagedTool(
            INFERENCEX_CLIENT_COMMIT, {path: hashlib.sha256(data).hexdigest() for path, data in _STAGED_FILES.items()}
        )
    }
).to_bytes()
#: The normalised manifest of that staging with lm-eval at its pin, which is how a sheet names it and how both sides
#: compare theirs.
STAGING = NormalisedManifest(
    INFERENCEX_CLIENT_COMMIT,
    "3.12.15",
    "b" * 64,
    (
        PackageIdentity("agentx-harness", "1.0.13", _HARNESS_SHA256),
        PackageIdentity("lm-eval", LM_EVAL_VERSION, LM_EVAL_WHEEL_SHA256),
    ),
)


def stage_into(runner_cache: Path) -> Path:
    """Lay a staging that verifies, with :data:`STAGING_MANIFEST`, where a runner whose cache is ``runner_cache``
    finds the published inference_suite staging of its pins; returns that tool cache."""
    tools = runner_cache / INFERENCE_SUITE_DIR / staging_key()
    for path, data in _STAGED_FILES.items():
        staged = tool_root(tools, INFERENCEX_TOOL, INFERENCEX_CLIENT_COMMIT) / path
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_bytes(data)
    (tools / MANIFEST_FILE).write_bytes(STAGING_MANIFEST)
    return tools


def lm_eval_at_its_pin(directory: Path) -> Path:
    """A directory that, first on ``sys.path``, makes this interpreter report lm-eval at its pin, as a runner's does;
    CI installs no lm-eval."""
    info = directory / f"lm_eval-{LM_EVAL_VERSION}.dist-info"
    info.mkdir(parents=True, exist_ok=True)
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: lm_eval\nVersion: {LM_EVAL_VERSION}\n")
    return directory


@functools.lru_cache(maxsize=None)
def base_launch() -> BaseLaunch:
    """A frozen base resolved offline from the default table, as a session without InferenceX access freezes one."""
    snapshot = SourceSnapshot(None, 1_793_000_000, unread=UnreadSource("InferenceX", "offline"))
    query = LaunchQuery(
        "meta-llama/Llama-3.1-8B-Instruct", "sglang", "mi355x", Workload.FIXED, (64,), isl=1024, osl=1024, tp=1
    )
    base = resolve_base(query, snapshot, gpus=8)
    assert isinstance(base, BaseLaunch), base
    return base


def spec_document() -> dict[str, Any]:
    """A sheet: the frozen base carried whole and the inference_suite staging as its normalised manifest."""
    yaml_text = (
        "model: m\nserver:\n  pid_dir: /shared/u1/sess-a/pids\n  port: 8000\ninferencex_path: /shared/u1/sess-a/ix\n"
    )
    return {
        "schema": "hyperloom.measurement-spec/v2",
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
        "tools": {
            "inference_suite": {"staging": STAGING.sha256(), "normalised": STAGING.to_document()},
            "inferencex": {"commit": "b" * 40},
        },
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
            "rounds": {"warmup": 1, "measure": 3},
            "accuracy": False,
            "timeout_sec": 3600,
            "silence_timeout_sec": 600,
            "deadline_unix": 1790000000.0,
            "base_launch": base_launch().to_document(),
        },
        "outputs": {"result_dir": "results", "max_total_bytes": 4294967296},
    }


def spec_json(document: dict[str, Any]) -> bytes:
    return json.dumps(document).encode()
