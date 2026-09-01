# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Dense FP8 blockscale GEMM tuner via aiter's gemm_a8w8_blockscale_tune.py."""

from __future__ import annotations

import logging

from .base import BaseTuner, TuneResult
from ._aiter_dense_common import (
    SPLITK_TRIAL_SCRIPT_KEY,
    run_aiter_dense_tuner,
    validate_dense_tuner_inputs,
)
from ..utils import TUNER_ENV_VARS

log = logging.getLogger(__name__)


class A8W8BlockscaleTuner(BaseTuner):
    """Tune dense FP8 blockscale GEMM kernels."""

    name = "a8w8_blockscale"
    env_var = TUNER_ENV_VARS["a8w8_blockscale"]

    def validate(self) -> str | None:
        return validate_dense_tuner_inputs(self.ctx, "a8w8_blockscale", script_label="blockscale")

    def run(self) -> TuneResult:
        return run_aiter_dense_tuner(
            tuner_name=self.name,
            script_key=SPLITK_TRIAL_SCRIPT_KEY,
            env_var=self.env_var,
            ctx=self.ctx,
            work_dir=self.work_dir,
            # --splitK enables aiter's split-K search. Without it the tuner sets
            # maxsplitK=0 and never evaluates split-K>0 (see
            # gemm_a8w8_blockscale_tune.py: `maxsplitK = compute_gemm_SplitK(...)
            # if args.splitK else 0`). split-K>0 is the fastest config for
            # small-M (decode) GEMMs and carries the measured e2e throughput
            # gain; omitting the flag silently loses it.
            extra_args=["--libtype", "all", "--splitK"],
        )
