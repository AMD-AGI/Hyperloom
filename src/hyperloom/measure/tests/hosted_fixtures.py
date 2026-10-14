# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A hosted backend over the local sandbox provider and the fake runner, for the hosted tests."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom.measure.archive import DirResultArchive
from hyperloom.measure.compile import LocalBlobStore
from hyperloom.measure.hosted import HostedConfig, HostedMeasurementBackend
from hyperloom.measure.hosted_spec import SpecContext
from hyperloom.measure.socket import TestCall
from hyperloom.sandbox.local import LocalSandboxProvider, LocalTemplate

FAKE_RUNNER = (sys.executable, "-m", "hyperloom.measure.tests.fake_runner")
COMMIT = "a" * 40
HOSTNAME = "node-a"


def runner_template(name: str = "runner-1", *, gpus: int = 1, signing: bool = True) -> LocalTemplate:
    return LocalTemplate(name, "measurement", command=FAKE_RUNNER, gpus=gpus, sealed=True, signing=signing)


def point_yaml(directory: Path, name: str = "point.yaml", *, tp: int = 1, **fake: Any) -> Path:
    """A minimal materialised point; ``fake`` keys steer the fake runner."""
    lines = ["benchmark:", "  envs:", f"    TP: '{tp}'", "    RUN_EVAL: 'false'"]
    lines.extend(f"{key}: {value}" for key, value in fake.items())
    path = directory / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def make_call(tmp: Path, task_id: str = "req-1", *, point: Path | None = None, **overrides: Any) -> TestCall:
    arguments: dict[str, Any] = {
        "points": (point or point_yaml(tmp),),
        "output_dir": tmp / "out" / task_id,
        "timeout_sec": 60,
        "cwd": str(tmp),
        "magpie_python": sys.executable,
        "task_id": task_id,
    }
    arguments.update(overrides)
    return TestCall(**arguments)


@dataclass
class Rig:
    provider: LocalSandboxProvider
    backend: HostedMeasurementBackend
    config: HostedConfig
    archive: DirResultArchive
    tmp: Path

    def record_archive(self, request_id: str) -> str:
        return str(self.backend.journal.latest(request_id).extra["archive"])

    def another_backend(self, **overrides: Any) -> HostedMeasurementBackend:
        """A second backend over the same session directory and provider: what a restarted application looks like."""
        config = HostedConfig(**{**self.config.__dict__, **overrides})
        backend = HostedMeasurementBackend(self.provider, config, sleep=lambda _: None)
        backend.start()
        return backend


def spec_context(tmp: Path, **overrides: Any) -> SpecContext:
    arguments: dict[str, Any] = {
        "image_ref": "registry.invalid/fw@sha256:" + "22" * 32,
        "image_digest": "sha256:" + "22" * 32,
        "gpu_type": "FakeGPU",
        "magpie_commit": COMMIT,
        "inferencex_commit": COMMIT,
        "blobs": LocalBlobStore(tmp / "session" / "hosted" / "blobs"),
        "process_environ": {},
        "platform_env": {},
        "blob_root": str(tmp / "session" / "hosted" / "blobs"),
    }
    arguments.update(overrides)
    return SpecContext(**arguments)


def make_rig(
    tmp: Path,
    *,
    templates: list[LocalTemplate] | None = None,
    gpus: int = 8,
    spec: Mapping[str, Any] | None = None,
    **config: Any,
) -> Rig:
    provider = LocalSandboxProvider(
        templates or [runner_template()], gpus=range(gpus), base_dir=tmp / "sandboxes", hostname=HOSTNAME
    )
    archive = DirResultArchive(tmp / "archive")
    arguments: dict[str, Any] = {
        "session_dir": tmp / "session",
        "run_id": "run-1",
        "spec": spec_context(tmp, **(spec or {})),
        "runner_templates": {1: "runner-1"},
        "archive": archive,
        "poll_interval_sec": 0.05,
        "runner_start_timeout_sec": 5,
        "staging_root": tmp / "staging",
    }
    arguments.update(config)
    hosted = HostedConfig(**arguments)
    backend = HostedMeasurementBackend(provider, hosted)
    backend.start()
    return Rig(provider, backend, hosted, archive, tmp)
