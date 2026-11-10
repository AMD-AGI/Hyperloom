# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a GPU test needs from the machine, and the base launch its session runs on, frozen as a multibranch session
freezes its own."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from hyperloom_multibranch.scheduling.session_base import freeze_session_base
from hyperloom_multibranch.scheduling.settings import InferenceXShape, KernelOptimizer, Workload
from inference_suite.contracts import BaseLaunch


def gpu_inputs_missing(model: str, inferencex: str, out: str) -> bool:
    """Whether SGLang, the model directory, the InferenceX checkout or the output directory a GPU test's variables name is
    missing. An unset variable is missing: ``Path("")`` is the working directory, which from the repository root has a
    ``.git`` and would take the test's output."""
    if importlib.util.find_spec("sglang") is None or not all(value.strip() for value in (model, inferencex, out)):
        return True
    return not Path(model).is_dir() or not Path(inferencex, ".git").exists()


def freeze_gpu_session_base(session: Path, model: str, gpu_type: str, *, tp: int, conc: int) -> BaseLaunch:
    """Freeze the base launch of an InferenceX session serving ``model`` at ``tp`` and ``conc`` in ``session``. The
    framework and the request shape are the ``FRAMEWORK``, ``ISL`` and ``OSL`` the test's canonical point is rendered
    from; the board is lowercased as the multibranch command line lowercases ``--gpu-type``."""

    shape = InferenceXShape(int(os.environ["ISL"]), int(os.environ["OSL"]))
    workload = Workload(
        model, os.environ["FRAMEWORK"], tp, conc, "bf16", gpu_type.lower(), KernelOptimizer.FORGE, shape
    )
    return freeze_session_base(session, workload)
