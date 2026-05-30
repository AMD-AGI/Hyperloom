"""Session configuration schema.

Defines what the user provides (benchmark script, accuracy script, model path)
and what the system infers (framework, GPU type, execution mode).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ExecutionMode(Enum):
    LOCAL = "local"
    CLUSTER = "cluster"
    AUTO = "auto"


class OutputFormat(Enum):
    JSON = "json"
    REGEX = "regex"
    LAST_LINE = "last_line"


@dataclass
class BenchmarkConfig:
    """Configuration for the benchmark plugin."""

    script: str = ""
    framework: str = ""  # "sglang", "vllm", "inferencex", or "" for custom
    output_format: str = "json"
    throughput_key: str = "throughput"
    latency_key: str = "latency_mean_ms"
    timeout: int = 7200
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class AccuracyConfig:
    """Configuration for the accuracy eval plugin."""

    script: str = ""
    output_format: str = "json"
    score_key: str = "score"
    threshold: float = 0.0
    timeout: int = 7200
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class SessionConfig:
    """Top-level session configuration."""

    model_path: str = ""
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    accuracy: AccuracyConfig = field(default_factory=AccuracyConfig)

    mode: ExecutionMode = ExecutionMode.AUTO
    target_gain: float = 10.0  # percent improvement target
    max_hours: float = 4.0
    gpu_type: str = ""  # auto-detected if empty
    gpus: str = ""  # e.g., "0,1,2,3,4,5,6,7"
    tp: int = 0  # tensor parallelism (0 = auto)
    port: int = 8000  # serving port

    launch_script: str = ""  # script to launch the serving framework
    session_dir: str = ""
    agent_model: str = "claude-sonnet-4-6"

    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_path:
            self.model_path = os.environ.get("MODEL_PATH", "")
        if not self.session_dir:
            self.session_dir = os.environ.get(
                "HYPERLOOM_SESSION_DIR", "sessions"
            )
        if not self.gpu_type:
            from .gpu import detect_gpu
            spec = detect_gpu()
            if spec:
                self.gpu_type = spec.name

    def validate(self) -> list[str]:
        """Return list of validation errors (empty = valid)."""
        errors = []
        if not self.model_path:
            errors.append("--model is required (or set MODEL_PATH env var)")
        if not self.benchmark.script and not self.benchmark.framework:
            errors.append(
                "--benchmark <script> or --framework <name> is required"
            )
        if self.benchmark.script and not Path(self.benchmark.script).exists():
            errors.append(
                f"Benchmark script not found: {self.benchmark.script}"
            )
        if self.accuracy.script and not Path(self.accuracy.script).exists():
            errors.append(
                f"Accuracy script not found: {self.accuracy.script}"
            )
        return errors
