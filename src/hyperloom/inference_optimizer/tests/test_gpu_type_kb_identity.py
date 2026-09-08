# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A Radeon card must key the recipe KB on its own identity, not ``unknown_gpu``.

The probe recognised only MI product tags, so a Radeon run resolved
``gpu_type=''`` and landed on the shared ``unknown_gpu`` KB key -- one namespace
for every unrecognised card, so a recipe learned on a Strix Halo iGPU was read
back on unrelated hardware.

The MI product-tag path, the torch ``gcnArchName`` fallback and the undetectable
case are covered by ``test_coverage_margin3_unit.py``'s
``test_gpu_type_autodetect_rocm_and_torch_fallback``; not repeated here.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from hyperloom.inference_optimizer import gpu_types
from hyperloom.inference_optimizer.recipe_snapshot_constants import (
    kb_hardware_slug,
    recipe_canonical_id,
)

# Verbatim `rocm-smi --showproductname` from a Ryzen AI Max+ 395 (Strix Halo).
# The Card Series has no "AMD " prefix on this build; others print "AMD Radeon
# 8060S Graphics", which is why the needle starts at "RADEON".
STRIX_HALO_ROCM_SMI = """\
============================ ROCm System Management Interface ============================
====================================== Product Info ======================================
GPU[0]\t\t: Card Series: \t\tRadeon 8060S Graphics
GPU[0]\t\t: Card Model: \t\t0x1586
GPU[0]\t\t: Card Vendor: \t\tAdvanced Micro Devices, Inc. [AMD/ATI]
GPU[0]\t\t: Card SKU: \t\tSTRXLGEN
GPU[0]\t\t: Subsystem ID: \t0x0124
GPU[0]\t\t: Device Rev: \t\t0xc1
GPU[0]\t\t: Node ID: \t\t1
GPU[0]\t\t: GUID: \t\t11948
GPU[0]\t\t: GFX Version: \t\tgfx1151
==========================================================================================
================================== End of ROCm SMI Log ===================================
"""


@pytest.fixture
def stub_rocm_smi(monkeypatch):
    """Return a callable that pins ``rocm-smi`` stdout for the probe."""

    def _install(stdout: str):
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *_a, **_k: SimpleNamespace(stdout=stdout, stderr="", returncode=0),
        )

    return _install


@pytest.mark.parametrize("card_series", ["Radeon 8060S Graphics", "AMD Radeon 8060S Graphics"])
def test_probe_reads_radeon8060s_from_the_product_name(stub_rocm_smi, card_series):
    """Both spellings are in the wild, so the needle must not anchor on "AMD"."""
    stub_rocm_smi(STRIX_HALO_ROCM_SMI.replace("Radeon 8060S Graphics", card_series))
    assert gpu_types._autodetect_gpu_type() == "radeon8060s"


def test_radeon8060s_is_the_kb_key_not_unknown_gpu():
    """The canonical_id hardware segment carries the real card."""
    hw = kb_hardware_slug("radeon8060s", nodes=1, gpus_per_node=1)
    assert hw == "radeon8060s"
    canonical_id = recipe_canonical_id(
        model="gemma4",
        hardware=hw,
        framework_name="vllm",
        framework_version="0.11.0",
        precision="bf16",
    )
    assert ":radeon8060s:" in canonical_id
    assert "unknown_gpu" not in canonical_id


def test_radeon8060s_names_its_own_magpie_script():
    """runner_type is the gpu_type verbatim, so Magpie pins vllm_radeon8060s.sh."""
    assert gpu_types._gpu_runner_type("radeon8060s") == "radeon8060s"


def test_radeon8060s_dispatches_as_gfx1151():
    """Read from rocminfo on a Ryzen AI Max+ 395; AITER CSVs are matched on it."""
    assert gpu_types.amd_gpu_dispatch_identity("radeon8060s") == ("gfx1151", 40)
    assert gpu_types._resolve_amd_gpu_type("radeon8060s") == "radeon8060s"


def test_product_tags_stay_derived_from_the_identities_table():
    """The needle is the gpu_type uppercased; the probe drops spaces to match."""
    assert "RADEON8060S" in gpu_types._PRODUCT_TAGS
    assert {t.lower() for t in gpu_types._PRODUCT_TAGS} == set(gpu_types._AMD_GPU_TYPES)


def test_radeon8060s_does_not_reach_the_cdna3_fast_paths():
    """Being a known AMD board is not the same as being gfx942.

    The sglang FP8-per-token env and the CK block-scale switch gate on
    ``_GFX942_GPU_TYPES``, not on the identities table, so widening the latter
    must not pull an RDNA board into either.
    """
    from hyperloom.orchestrator.actions.executors._workload_envs import _GFX942_GPU_TYPES

    assert "radeon8060s" not in _GFX942_GPU_TYPES
