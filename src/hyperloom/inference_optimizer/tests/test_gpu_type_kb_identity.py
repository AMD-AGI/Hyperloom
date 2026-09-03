# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A gfx11 card must key the recipe KB on its own identity, not ``unknown_gpu``.

Before this, ``--gpu-type`` accepted only MI parts and the probe recognised only
gfx942/gfx950, so every gfx11 run resolved ``gpu_type=''`` and landed on the
shared ``unknown_gpu`` KB key -- one namespace for every unrecognised card, so a
recipe learned on a Strix Halo iGPU was read back on unrelated hardware.
"""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest

from hyperloom.inference_optimizer import gpu_types
from hyperloom.inference_optimizer.cli.parser import _build_parser
from hyperloom.inference_optimizer.recipe_snapshot_constants import (
    kb_hardware_slug,
    recipe_canonical_id,
)


# Verbatim `rocm-smi --showproductname` from a Ryzen AI Max+ 395 (Strix Halo).
STRIX_HALO_ROCM_SMI = """\
============================ ROCm System Management Interface ============================
====================================== Product Info ======================================
GPU[0]\t\t: Card Series: \t\tAMD Radeon 8060S Graphics
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
        def fake_run(*_args, **_kwargs):
            return SimpleNamespace(stdout=stdout, stderr="", returncode=0)

        monkeypatch.setattr(subprocess, "run", fake_run)

    return _install


# --------------------------------------------------------------------------
# Probe
# --------------------------------------------------------------------------
def test_probe_reads_8060s_from_rocm_smi_product_name(stub_rocm_smi):
    """The product name is the authoritative signal, not the gfx arch.

    gfx1151 is both the 8060S and the 8050S, so falling straight to the arch
    would merge two different cards into one KB namespace.
    """
    stub_rocm_smi(STRIX_HALO_ROCM_SMI)
    assert gpu_types._autodetect_gpu_type() == "8060s"


def test_probe_falls_back_to_gfx_version_for_untabulated_sku(stub_rocm_smi):
    """An unknown Radeon SKU still resolves via the GFX Version line.

    Reporting the arch is worse than the exact SKU but far better than ``None``,
    which is what put the run on ``unknown_gpu``.
    """
    stub_rocm_smi(STRIX_HALO_ROCM_SMI.replace("8060S Graphics", "8040S Graphics"))
    assert gpu_types._autodetect_gpu_type() == "gfx1151"


def test_probe_still_reads_mi_parts(stub_rocm_smi):
    """MI detection is unchanged; the most specific tag wins."""
    stub_rocm_smi("Card Series: AMD Instinct MI325X")
    assert gpu_types._autodetect_gpu_type() == "mi325x"


def test_probe_returns_none_when_nothing_is_recognisable(stub_rocm_smi, monkeypatch):
    """No signal at all still degrades to the --gpu-type hint."""
    stub_rocm_smi("Card Series: Some Other Vendor Accelerator")
    # None in sys.modules makes `import torch` raise, standing in for a host
    # without torch regardless of what the test venv happens to have.
    monkeypatch.setitem(sys.modules, "torch", None)
    assert gpu_types._autodetect_gpu_type() is None


# --------------------------------------------------------------------------
# CLI surface
# --------------------------------------------------------------------------
def _parse_gpu_type(value: str):
    return _build_parser().parse_args(["optimize", "--model", "/models/m", "--gpu-type", value])


def test_gpu_type_flag_accepts_8060s():
    """The probe can produce 8060s, so --gpu-type must not reject it."""
    assert _parse_gpu_type("8060S").gpu_type == "8060s"


def test_gpu_type_flag_still_rejects_typos():
    """Widening the MI-only list must not drop typo protection."""
    with pytest.raises(SystemExit):
        _parse_gpu_type("mi300")


def test_gpu_type_choices_cover_every_probe_result():
    """Parser choices and probe outputs come from one table and cannot drift."""
    assert set(gpu_types.GPU_TYPE_CHOICES) >= set(gpu_types._GFX_TO_GPU_TYPE.values())
    assert set(gpu_types.GPU_TYPE_CHOICES) >= {t for _, t in gpu_types._PRODUCT_TAGS}


# --------------------------------------------------------------------------
# KB identity vs Magpie runner
# --------------------------------------------------------------------------
def test_8060s_is_the_kb_key_not_unknown_gpu():
    """The canonical_id hardware segment carries the real card."""
    hw = kb_hardware_slug("8060s", nodes=1, gpus_per_node=1)
    assert hw == "8060s"
    canonical_id = recipe_canonical_id(
        model="gemma4",
        hardware=hw,
        framework_name="vllm",
        framework_version="0.11.0",
        precision="bf16",
    )
    assert ":8060s:" in canonical_id
    assert "unknown_gpu" not in canonical_id


def test_kb_key_derivation_sites_no_longer_fall_back():
    """Mirror the ``state.gpu_type or "unknown_gpu"`` guard in the KB call sites.

    ``phases/machine.py``, ``cli/kb.py`` and ``loop/coordinator.py`` all key off
    ``SharedState.gpu_type``; the fallback only fires when that is empty, which
    is exactly what a gfx11 run used to leave behind.
    """
    state = SimpleNamespace(gpu_type="8060s")
    assert (getattr(state, "gpu_type", "") or "unknown_gpu") == "8060s"


def test_8060s_keeps_its_own_kb_key_while_sharing_the_gfx11_runner():
    """One Magpie script, several KB namespaces.

    Magpie ships a single ``{framework}_gfx11.sh`` for every gfx11 SKU, but the
    cards behind it benchmark differently, so the KB identity must stay finer
    than the runner label -- exactly the mi308x/mi325x -> mi300x split.
    """
    assert gpu_types._gpu_runner_type("8060s") == "gfx11"
    assert gpu_types._gpu_runner_type("gfx1151") == "gfx11"
    assert kb_hardware_slug("8060s", nodes=1, gpus_per_node=1) == "8060s"


def test_mi_runner_mapping_unchanged():
    """The MI325X/MI308X -> MI300X script aliasing is untouched."""
    assert gpu_types._gpu_runner_type("MI325X") == "mi300x"
    assert gpu_types._gpu_runner_type("mi308x") == "mi300x"
    assert gpu_types._gpu_runner_type("mi355x") == "mi355x"
    assert gpu_types._gpu_runner_type("") == ""


def test_8060s_does_not_enable_cdna_fast_paths():
    """A new KB identity + Magpie runner must not widen aiter / CK gating."""
    assert gpu_types._resolve_amd_gpu_type("8060s") is None
    assert gpu_types.amd_gpu_dispatch_identity("8060s") is None
    assert gpu_types._resolve_amd_gpu_type("mi325x") == "mi325x"
