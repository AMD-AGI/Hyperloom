# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Dense FP4 blockscale GEMM tuner via aiter's gemm_a4w4_blockscale_tune.py."""

from __future__ import annotations

import logging

from .base import BaseTuner, TuneResult
from ._aiter_dense_common import DenseTunerSpec, run_aiter_dense_tuner, validate_dense_tuner_inputs
from ..utils import TUNER_ENV_VARS

log = logging.getLogger(__name__)


class A4W4BlockscaleTuner(BaseTuner):
    """Tune dense FP4 blockscale GEMM kernels."""

    name = "a4w4_blockscale"
    env_var = TUNER_ENV_VARS["a4w4_blockscale"]

    def validate(self) -> str | None:
        return validate_dense_tuner_inputs(self.ctx, "a4w4_blockscale", script_label="a4w4 blockscale")

    def run(self) -> TuneResult:
        return run_aiter_dense_tuner(
            DenseTunerSpec(self.name, "a4w4_blockscale", self.env_var),
            self.ctx,
            self.work_dir,
        )
