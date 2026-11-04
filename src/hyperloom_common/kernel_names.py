# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Kernel-name tokens that spell an op but name a model."""

from __future__ import annotations

import re

# ``gemma`` as a whole token, optionally versioned (``gemma3``), is the model family and spells ``gemm``;
# CK and aiter GEMM types continue it with a letter or digit (``GemmA8W8``, ``BlockGemmASmem``).
_GEMMA_MODEL = re.compile(r"(?<![a-z0-9])gemma\d*(?![a-z0-9])")


def without_model_names(lowered: str) -> str:
    """``lowered`` with model-family tokens removed, so an op marker matches only the op."""
    return _GEMMA_MODEL.sub("", lowered)


__all__ = ["without_model_names"]
