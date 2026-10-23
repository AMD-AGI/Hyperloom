# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Deliver a finished forge-loop campaign: one reviewed, checked commit in place of its kept rounds."""

from kernelforge.finalize.delivery import (
    FINALIZE_JSON,
    FINALIZE_MD,
    HISTORY_BRANCH,
    DeliveryOptions,
    FinalizeError,
    finalize_campaign,
)

__all__ = ["FINALIZE_JSON", "FINALIZE_MD", "HISTORY_BRANCH", "DeliveryOptions", "FinalizeError", "finalize_campaign"]
