# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Dense FP8 bpreshuffle GEMM tuner via aiter's gemm_a8w8_bpreshuffle_tune.py."""

from __future__ import annotations

import logging

from .base import BaseTuner, TuneResult
from ._aiter_dense_common import run_aiter_dense_tuner, validate_dense_tuner_inputs
from ..utils import TUNER_ENV_VARS

log = logging.getLogger(__name__)


class A8W8BpreshuffleTuner(BaseTuner):
    """Tune dense FP8 bpreshuffle GEMM kernels."""

    name = "a8w8_bpreshuffle"
    env_var = TUNER_ENV_VARS["a8w8_bpreshuffle"]

    def validate(self) -> str | None:
        return validate_dense_tuner_inputs(self.ctx, "a8w8_bpreshuffle", script_label="bpreshuffle")

    def run(self) -> TuneResult:
        return run_aiter_dense_tuner(
            tuner_name=self.name,
            script_key="a8w8_bpreshuffle",
            env_var=self.env_var,
            ctx=self.ctx,
            work_dir=self.work_dir,
        )
