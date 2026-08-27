# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Dense FP8 per-token/per-tensor GEMM tuner via aiter's gemm_a8w8_tune.py."""

from __future__ import annotations

import logging

from .base import BaseTuner, TuneResult
from ._aiter_dense_common import run_aiter_dense_tuner, validate_dense_tuner_inputs
from ..utils import TUNER_ENV_VARS

log = logging.getLogger(__name__)


class A8W8Tuner(BaseTuner):
    """Tune dense FP8 per-token/per-tensor GEMM kernels."""

    name = "a8w8"
    env_var = TUNER_ENV_VARS["a8w8"]

    def validate(self) -> str | None:
        return validate_dense_tuner_inputs(self.ctx, "a8w8", script_label="a8w8")

    def run(self) -> TuneResult:
        return run_aiter_dense_tuner(
            tuner_name=self.name,
            script_key="a8w8",
            env_var=self.env_var,
            ctx=self.ctx,
            work_dir=self.work_dir,
        )
