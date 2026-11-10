# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Preflight the independent MLPerf agentic harness."""

from pathlib import Path
import shutil
from typing import Mapping


class AgentXPreflightError(RuntimeError):
    """An unusable MLPerf harness."""


def _check_mlperf_framework(env: Mapping[str, str]) -> None:
    """Raise unless this framework can serve the agentic client.

    The client speaks OpenAI chat-completions to ``$PORT`` and delegates the server to Magpie's
    ``{framework}_{gpu}.sh``, so every registered *serving* framework qualifies -- sglang, vllm and atom all expose
    that surface. A scriptable framework has no endpoint to drive and is refused by kind rather than by name, so a
    framework added to the registry later does not silently inherit a guard written before it existed.
    """
    from .. import framework_registry

    framework = str(env.get("FRAMEWORK") or "").strip().lower()
    if not framework:
        return
    if not framework_registry.is_supported(framework):
        raise AgentXPreflightError(
            f"HYPERLOOM_AGENTIC_BACKEND=mlperf: {framework!r} is not a registered framework "
            f"(known: {', '.join(framework_registry.names())})."
        )
    if framework_registry.is_scriptable(framework):
        raise AgentXPreflightError(
            f"HYPERLOOM_AGENTIC_BACKEND=mlperf drives an HTTP endpoint; refusing scriptable framework {framework!r}."
        )


def check_mlperf_harness(env: Mapping[str, str]) -> None:
    """Raise :class:`AgentXPreflightError` unless the MLPerf agentic harness can run."""
    root = Path(str(env.get("MLPERF_ENDPOINTS_DIR") or "/opt/mlperf-endpoints"))
    script = root / "utility" / "run_agentic.sh"
    if not script.is_file():
        raise AgentXPreflightError(
            f"HYPERLOOM_AGENTIC_BACKEND=mlperf but {script} is missing. "
            "Mount mlperf-endpoints and set MLPERF_ENDPOINTS_DIR."
        )
    dataset = str(env.get("AGENTIC_DATASET_PATH") or "").strip()
    if not dataset or not Path(dataset).exists():
        raise AgentXPreflightError(
            "HYPERLOOM_AGENTIC_BACKEND=mlperf requires AGENTIC_DATASET_PATH to "
            f"point at agentic_combined_v6.jsonl (got {dataset!r})."
        )
    tokenizer = str(env.get("MLPERF_TOKENIZER_DIR") or "").strip()
    if not tokenizer or not Path(tokenizer).is_dir():
        raise AgentXPreflightError(
            f"HYPERLOOM_AGENTIC_BACKEND=mlperf requires MLPERF_TOKENIZER_DIR (got {tokenizer!r})."
        )
    if not shutil.which("inference-endpoint", path=env.get("PATH")) and not shutil.which("uv", path=env.get("PATH")):
        raise AgentXPreflightError("HYPERLOOM_AGENTIC_BACKEND=mlperf needs inference-endpoint or uv on PATH.")
    _check_mlperf_framework(env)
    model = str(env.get("MODEL") or env.get("MODEL_PATH") or "").strip().lower()
    if model and "kimi" not in model:
        raise AgentXPreflightError(f"HYPERLOOM_AGENTIC_BACKEND=mlperf only measures Kimi-K3; refusing model {model!r}.")
