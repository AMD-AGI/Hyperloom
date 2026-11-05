# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The InferenceX client of a ``profile`` test, made to honour the capture window the point asks for.

A point's ``PROFILE_EXTRA_BODY`` is the ``/start_profile`` body of its capture window (it starts after the warmup and
spans enough steps to hold prefill chunks as well as decode steps), and its ``NUM_PROMPTS`` is what keeps the run going
long enough to reach that window. A stock InferenceX checkout honours neither: its client sends a built-in body that
captures one step per stage, and its profile mode sends one request per client slot. The single-lineage profile patches
the checkout it runs; a test on a runner's clean checkout must do the same, or it profiles a one-step window whose
CUDA-graph replays the ROCm profiler leaves without kernels, and the analysis of that trace is degraded.

The patches are the old flow's own (``_inferencex_patcher``); this module applies them to the one checkout the point
names and reads the result back, because the patcher reports success when any checkout it can reach was patched.
"""

from __future__ import annotations

from pathlib import Path

from hyperloom.orchestrator.actions.executors import (
    benchmark_serving_path_in,
    ensure_benchmark_lib_patched,
    ensure_benchmark_serving_patched,
)

__all__ = ["ProfileClientError", "patch_profile_client"]

_ENCODING = "utf-8"
#: What each patched file holds once its patch is in place: the library takes the prompt count from ``NUM_PROMPTS``, the
#: client takes the ``/start_profile`` body from ``PROFILE_EXTRA_BODY``.
_NUM_PROMPTS_SENTINEL = "${NUM_PROMPTS:-$max_concurrency}"
_BODY_SENTINEL = "PROFILE_EXTRA_BODY"


class ProfileClientError(Exception):
    """The checkout a profile test runs cannot be made to honour the point's capture window."""


def _holds(path: Path, needle: str) -> bool:
    try:
        return needle in path.read_text(encoding=_ENCODING, errors="replace")
    except OSError:
        return False


def patch_profile_client(root: Path | str) -> None:
    """Make the InferenceX checkout at ``root`` take the capture window and prompt count from the point's environment.

    Idempotent, and inert for every run that does not profile: both files read the variables only in profile mode.
    Raises :class:`ProfileClientError` naming the files that still lack the patch, when upstream's text has moved from
    the lines the patches anchor on.
    """
    base = Path(root)
    ensure_benchmark_lib_patched(base)
    ensure_benchmark_serving_patched(base)
    library = base / "benchmarks" / "benchmark_lib.sh"
    client = benchmark_serving_path_in(base)
    missing = [
        str(path.relative_to(base))
        for path, needle in ((library, _NUM_PROMPTS_SENTINEL), (client, _BODY_SENTINEL))
        if not _holds(path, needle)
    ]
    if missing:
        raise ProfileClientError(
            f"the InferenceX checkout cannot honour a profile point's NUM_PROMPTS and PROFILE_EXTRA_BODY: "
            f"{', '.join(missing)} lack the patch"
        )
