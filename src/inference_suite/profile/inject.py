# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Turn the torch profiler on in a profile round's server, and keep its three trace directories apart.

SGLang reads the trace dir from its environment and opens the window on ``/start_profile`` with the body the client
sends. vLLM takes the dir and the window as ``--profiler-config.*`` flags, and ATOM (fixed load only) takes the dir as
a flag. The layer composes last, so the dir it names beats any placeholder an earlier layer carries.

TraceLens-patched servers write their CUDA-graph capture sidecars into ``capture_traces`` and
``graph_capture_profile`` below the trace dir; :func:`settle_traces` lifts them beside ``torch_trace``, so a reader
tells a step trace from a sidecar by its directory.
"""

from __future__ import annotations

from pathlib import Path

from inference_suite.contracts.config import LaunchDelta, ProfileConfig, SuiteConfig
from inference_suite.contracts.launch import LaunchError
from inference_suite.contracts.layout import PROFILE_TRACE_DIRS, RunLayout
from inference_suite.frameworks import Framework

TORCH_TRACE_DIR, CAPTURE_TRACES_DIR, GRAPH_CAPTURE_DIR = PROFILE_TRACE_DIRS
SGLANG_PROFILER_DIR_ENV = "SGLANG_TORCH_PROFILER_DIR"
ATOM_PROFILER_DIR_FLAG = "--torch-profiler-dir"
VLLM_PROFILER_CONFIG = "--profiler-config."
#: The ``/start_profile`` body keys that bound SGLang's capture window, in engine forward steps.
SGLANG_WINDOW_KEYS = ("start_step", "num_steps")


def torch_trace_dir(layout: RunLayout) -> Path:
    return layout.profile_dir / TORCH_TRACE_DIR


def _vllm_args(profile: ProfileConfig, trace_dir: str) -> tuple[str, ...]:
    settings: list[tuple[str, object]] = [("profiler", "torch"), ("torch_profiler_dir", trace_dir)]
    if profile.delay_iters is not None:
        settings += [("delay_iterations", profile.delay_iters), ("max_iterations", profile.max_iters)]
    return tuple(token for name, value in settings for token in (VLLM_PROFILER_CONFIG + name, str(value)))


def profiler_layer(config: SuiteConfig, framework: Framework) -> LaunchDelta | None:
    """The layer that turns the profiler on in ``framework``'s server; ``None`` when the round profiles nothing.

    Raises:
        LaunchError: When ``framework`` runs no server to profile.
    """
    if not config.profile.enabled:
        return None
    trace_dir = str(torch_trace_dir(RunLayout(Path(config.paths.run_dir))).absolute())
    if framework is Framework.SGLANG:
        return LaunchDelta(env_set={SGLANG_PROFILER_DIR_ENV: trace_dir})
    if framework is Framework.VLLM:
        return LaunchDelta(args_add=_vllm_args(config.profile, trace_dir))
    if framework is Framework.ATOM:
        return LaunchDelta(args_add=(ATOM_PROFILER_DIR_FLAG, trace_dir))
    raise LaunchError(f"{framework.value} runs no server to profile")


def start_body(profile: ProfileConfig, framework: Framework) -> dict[str, object]:
    """The ``/start_profile`` body the client sends: SGLang's carries the step window, the others take it as given."""
    body = dict(profile.start_body or {})
    if framework is Framework.SGLANG and profile.delay_iters is not None:
        body.update(zip(SGLANG_WINDOW_KEYS, (profile.delay_iters, profile.max_iters)))
    return body


def make_trace_dirs(layout: RunLayout) -> None:
    for directory in layout.profile_trace_dirs():
        directory.mkdir(parents=True, exist_ok=True)


def settle_traces(layout: RunLayout) -> tuple[str, ...]:
    """Lift the capture sidecars out of ``torch_trace`` beside it; return the trace dirs that hold a file.

    Runs once the server has stopped writing, after :func:`make_trace_dirs`. The paths are relative to the run_dir.
    """
    for name in (CAPTURE_TRACES_DIR, GRAPH_CAPTURE_DIR):
        nested = torch_trace_dir(layout) / name
        if nested.is_dir():
            # rmdir refuses a non-empty target, so two captures are never merged into one directory.
            (layout.profile_dir / name).rmdir()
            nested.rename(layout.profile_dir / name)
    return tuple(
        directory.relative_to(layout.root).as_posix()
        for directory in layout.profile_trace_dirs()
        if any(path.is_file() for path in directory.rglob("*"))
    )


__all__ = [
    "CAPTURE_TRACES_DIR",
    "GRAPH_CAPTURE_DIR",
    "SGLANG_PROFILER_DIR_ENV",
    "TORCH_TRACE_DIR",
    "make_trace_dirs",
    "profiler_layer",
    "settle_traces",
    "start_body",
    "torch_trace_dir",
]
