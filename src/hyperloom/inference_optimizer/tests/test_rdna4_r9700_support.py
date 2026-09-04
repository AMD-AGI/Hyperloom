# SPDX-FileCopyrightText: 2026 The-Monk
# SPDX-License-Identifier: MIT

"""RDNA4 (gfx1201 / Radeon AI PRO R9700) support.

Upstream maps consumer arches to ``None``, so a gfx1201 box resolves to no
runner and no dispatch identity. These lock the port's behaviour, including
the parts that must stay NEGATIVE: gfx1200 is deliberately unmapped (same ISA
family, never measured here) and mxfp4 must stay unavailable, because the MX
scaled-converts are gfx950/gfx1250-gated silicon rather than a missing kernel.
"""

from __future__ import annotations

import pytest

from hyperloom.common.gpu_identity import AMD_GPU_DISPATCH_IDENTITIES, gfx_arch_for_gpu_type
from hyperloom.inference_optimizer.gpu_types import (
    _AMD_GPU_TYPES,
    _GFX_TO_RUNNER,
    _PRODUCT_TAGS,
    _gpu_runner_type,
    amd_gpu_dispatch_identity,
)
from hyperloom.orchestrator.phases.quantization_schemes import (
    SchemeNotSupportedError,
    supported_schemes,
    validate_scheme,
)


def test_r9700_has_a_dispatch_identity():
    # 64 CUs read from the board (amd-smi NUM_COMPUTE_UNITS), not a spec sheet.
    assert AMD_GPU_DISPATCH_IDENTITIES["r9700"] == ("gfx1201", 64)
    assert gfx_arch_for_gpu_type("r9700") == "gfx1201"


def test_r9700_is_an_accepted_board_and_resolves_explicitly():
    assert "r9700" in _AMD_GPU_TYPES
    assert amd_gpu_dispatch_identity("r9700") == ("gfx1201", 64)


def test_gfx1201_selects_the_r9700_runner():
    assert _GFX_TO_RUNNER["gfx1201"] == "r9700"
    # Not aliased onto an Instinct runner the way mi325x/mi308x fold to mi300x.
    assert _gpu_runner_type("r9700") == "r9700"


def test_r9700_product_tag_is_derived_for_autodetect():
    # rocm-smi reports "AMD Radeon AI PRO R9700"; autodetect substring-matches
    # the derived tag, so the board is detectable without a bespoke branch.
    assert "R9700" in _PRODUCT_TAGS


def test_gfx1200_is_deliberately_unmapped():
    # Navi 44 is the same ISA family but has not been measured on this port.
    # A runner label is a claim of support; absence here is the honest state.
    assert "gfx1200" not in _GFX_TO_RUNNER


def test_mxfp4_is_unavailable_on_rdna4():
    schemes = supported_schemes("r9700")
    assert "fp8" in schemes and "ptpc_fp8" in schemes
    assert "mxfp4" not in schemes and "mxfp4_fp8" not in schemes


def test_mxfp4_request_on_rdna4_fails_loudly():
    with pytest.raises(SchemeNotSupportedError):
        validate_scheme("mxfp4", "r9700")


def test_fp8_is_accepted_on_rdna4():
    validate_scheme("fp8", "r9700")  # must not raise
