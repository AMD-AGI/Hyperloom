# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The run_dir layout: every file a round leaves, by name, and the ``server_ready_at`` stamp's format.

One round writes one run_dir and leaves it unchanged once the round ends. Readers locate files through
:class:`RunLayout` rather than spelling names or globbing. ``gpu_metrics.json`` is not in the run_dir: the host
derives it from ``report.json`` into its own round directory, and :data:`HOST_DERIVED` names it.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

CONFIG_FILE = "config.yaml"
LAUNCH_FILE = "launch.json"
REPORT_FILE = "report.json"
SERVER_LOG = "server.log"
READY_STAMP = "server_ready_at"
SERVER_PID = "server.pid"
PHASE_MARKS = "phase_marks.json"
KV_METRICS = "kv_metrics.json"
AGENTX_TIMELINE = "agentx_timeline.jsonl"
GPU_SAMPLES = "gpu_samples.csv"
GPU_POWER = "gpu_power.json"
GPU_METRICS = "gpu_metrics.json"
#: Files a host writes from the run_dir into its own round directory; never inside a run_dir.
HOST_DERIVED = (GPU_METRICS,)

CLIENT_DIR = "client"
EVAL_DIR = "eval"
PROFILE_DIR = "profile"
TORCH_TRACE_DIR = "torch_trace"
CAPTURE_TRACES_DIR = "capture_traces"
GRAPH_CAPTURE_DIR = "graph_capture_profile"
#: Where TraceLens-patched servers write the CUDA-graph capture sidecars of boot: never a serving step.
CAPTURE_TRACE_DIRS = (CAPTURE_TRACES_DIR, GRAPH_CAPTURE_DIR)
#: The profiler's trace subdirectories, kept apart rather than flattened so a reader tells a step from a sidecar.
PROFILE_TRACE_DIRS = (TORCH_TRACE_DIR, *CAPTURE_TRACE_DIRS)
_UTF8 = "utf-8"


@dataclass(frozen=True)
class ReadyStamp:
    """When the server first reported ready, and how long its boot took (0 for a server the round reused).

    The boot is a duration on the spawning process's monotonic clock, so it means the same on a host whose wall
    clock disagrees; the wall-clock instant only says which round the stamp belongs to.
    """

    ready_unix: float
    boot_sec: float

    def __post_init__(self) -> None:
        for name in ("ready_unix", "boot_sec"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"the ready stamp's {name} must be a finite number, got {value!r}")
        if self.ready_unix <= 0 or self.boot_sec < 0:
            raise ValueError(f"a ready stamp needs ready_unix > 0 and boot_sec >= 0, got {self!r}")


def format_ready_stamp(stamp: ReadyStamp) -> str:
    return f"{stamp.ready_unix:.3f} {stamp.boot_sec:.3f}\n"


def parse_ready_stamp(text: str) -> ReadyStamp | None:
    """The stamp ``text`` records, or ``None`` when it records none: readers poll while the server boots."""

    fields = text.split()
    if len(fields) != 2:
        return None
    try:
        return ReadyStamp(float(fields[0]), float(fields[1]))
    except ValueError:
        return None


class RunDirError(ValueError):
    """The run_dir already holds something: one round writes one run_dir, from empty."""


@dataclass(frozen=True)
class RunLayout:
    """Where each file of the run_dir at ``root`` lives."""

    root: Path

    @property
    def framework_cwd(self) -> Path:
        """Where the serving interpreter starts, for the server and for every probe of its install.

        ``python -c`` and ``python -m`` put the current directory first on the import path. A host's own directory can
        hold a framework checkout without an ``__init__.py`` (the sglang image starts in ``/sgl-workspace``, beside
        ``sglang/``), which then imports as an empty namespace package ahead of the editable install.
        """
        return self.root

    @property
    def config(self) -> Path:
        return self.root / CONFIG_FILE

    @property
    def launch(self) -> Path:
        return self.root / LAUNCH_FILE

    @property
    def report(self) -> Path:
        return self.root / REPORT_FILE

    @property
    def server_log(self) -> Path:
        return self.root / SERVER_LOG

    @property
    def ready_stamp(self) -> Path:
        return self.root / READY_STAMP

    @property
    def server_pid(self) -> Path:
        """An own round's pidfile; ``boot`` and ``reuse`` keep theirs in the lifecycle's ``pid_dir``."""
        return self.root / SERVER_PID

    @property
    def phase_marks(self) -> Path:
        return self.root / PHASE_MARKS

    @property
    def kv_metrics(self) -> Path:
        return self.root / KV_METRICS

    @property
    def agentx_timeline(self) -> Path:
        return self.root / AGENTX_TIMELINE

    @property
    def gpu_samples(self) -> Path:
        return self.root / GPU_SAMPLES

    @property
    def gpu_power(self) -> Path:
        return self.root / GPU_POWER

    @property
    def client_dir(self) -> Path:
        return self.root / CLIENT_DIR

    @property
    def client_stdout(self) -> Path:
        return self.client_dir / "stdout.log"

    @property
    def client_stderr(self) -> Path:
        return self.client_dir / "stderr.log"

    @property
    def agentx_client_log(self) -> Path:
        return self.client_dir / "agentx_client.log"

    @property
    def inferencex_result(self) -> Path:
        return self.client_dir / "inferencex_result.json"

    @property
    def aiperf_artifacts(self) -> Path:
        return self.client_dir / "aiperf_artifacts"

    @property
    def aiperf_request_export(self) -> Path:
        return self.aiperf_artifacts / "profile_export.jsonl"

    @property
    def aiperf_log(self) -> Path:
        return self.aiperf_artifacts / "logs" / "aiperf.log"

    @property
    def mlperf_artifacts(self) -> Path:
        return self.client_dir / "mlperf_artifacts"

    @property
    def eval_dir(self) -> Path:
        """lm-eval names its ``results_<time>.json`` itself; readers search below this directory for it."""
        return self.root / EVAL_DIR

    @property
    def eval_stdout(self) -> Path:
        return self.eval_dir / "stdout.log"

    @property
    def eval_stderr(self) -> Path:
        return self.eval_dir / "stderr.log"

    @property
    def eval_probe(self) -> Path:
        return self.eval_dir / "hyperloom_eval_probe.json"

    @property
    def eval_definition(self) -> Path:
        return self.eval_dir / "definition.json"

    @property
    def profile_dir(self) -> Path:
        return self.root / PROFILE_DIR

    @property
    def profile_capture(self) -> Path:
        return self.profile_dir / "capture.json"

    @property
    def profile_torch_trace(self) -> Path:
        return self.profile_dir / TORCH_TRACE_DIR

    def profile_trace_dirs(self) -> tuple[Path, ...]:
        return tuple(self.profile_dir / name for name in PROFILE_TRACE_DIRS)

    def write_ready_stamp(self, stamp: ReadyStamp) -> None:
        """Replace the stamp whole, so a reader polling during the write never sees half of it."""

        staging = self.ready_stamp.with_name(f".{READY_STAMP}.{os.getpid()}")
        staging.write_text(format_ready_stamp(stamp), encoding=_UTF8)
        os.replace(staging, self.ready_stamp)

    def read_ready_stamp(self) -> ReadyStamp | None:
        """The stamp in this run_dir, or ``None`` before the server has reported ready."""

        try:
            text = self.ready_stamp.read_text(encoding=_UTF8)
        except FileNotFoundError:
            return None
        return parse_ready_stamp(text)

    def clear_ready_stamp(self) -> None:
        self.ready_stamp.unlink(missing_ok=True)

    def claim(self) -> None:
        """Create the run_dir for one round.

        Raises:
            RunDirError: The run_dir already holds something.
        """
        if self.root.exists() and any(self.root.iterdir()):
            raise RunDirError(f"the run_dir {self.root} is not empty")
        self.root.mkdir(parents=True, exist_ok=True)


__all__ = [
    "AGENTX_TIMELINE",
    "CAPTURE_TRACES_DIR",
    "CAPTURE_TRACE_DIRS",
    "CLIENT_DIR",
    "CONFIG_FILE",
    "EVAL_DIR",
    "GPU_METRICS",
    "GRAPH_CAPTURE_DIR",
    "GPU_POWER",
    "GPU_SAMPLES",
    "HOST_DERIVED",
    "KV_METRICS",
    "LAUNCH_FILE",
    "PHASE_MARKS",
    "PROFILE_DIR",
    "PROFILE_TRACE_DIRS",
    "READY_STAMP",
    "REPORT_FILE",
    "SERVER_LOG",
    "SERVER_PID",
    "TORCH_TRACE_DIR",
    "ReadyStamp",
    "RunDirError",
    "RunLayout",
    "format_ready_stamp",
    "parse_ready_stamp",
]
