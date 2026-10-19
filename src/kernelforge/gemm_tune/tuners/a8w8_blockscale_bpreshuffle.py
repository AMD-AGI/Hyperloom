# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Dense FP8 blockscale+bpreshuffle GEMM tuner for MI355X (gfx950)."""

from __future__ import annotations

import logging

from .base import BaseTuner, TuneResult
from ._aiter_dense_common import DenseTunerSpec, run_aiter_dense_tuner, validate_dense_tuner_inputs
from ..utils import TUNER_ENV_VARS

log = logging.getLogger(__name__)


class A8W8BlockscaleBpreshuffleTuner(BaseTuner):
    """Tune dense FP8 blockscale+bpreshuffle GEMM kernels (MI355X)."""

    name = "a8w8_blockscale_bpreshuffle"
    env_var = TUNER_ENV_VARS["a8w8_blockscale_bpreshuffle"]

    def validate(self) -> str | None:
        return validate_dense_tuner_inputs(
            self.ctx,
            "a8w8_blockscale_bpreshuffle",
            script_label="blockscale_bpreshuffle",
        )

    def run(self) -> TuneResult:
        return run_aiter_dense_tuner(
            DenseTunerSpec(
                self.name, "a8w8_blockscale_bpreshuffle", self.env_var, ("--libtype", "all", "--preshuffle")
            ),
            self.ctx,
            self.work_dir,
        )
