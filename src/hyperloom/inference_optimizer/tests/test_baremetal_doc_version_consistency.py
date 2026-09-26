# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Guard: install_baremetal.sh defaults stay in sync with docs/compatibility.rst."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
INSTALLER = REPO_ROOT / "src/hyperloom/inference_optimizer/assets/install_baremetal.sh"
COMPAT = REPO_ROOT / "docs/compatibility.rst"


def _default(var: str, text: str) -> str:
    m = re.search(r'%s="\$\{%s:-([^}]+)\}"' % (var, var), text)
    assert m, "could not find default for %s in install_baremetal.sh" % var
    return m.group(1)


def _sglang_extra_for_rocm72(text: str) -> str:
    """The wheel extra the installer derives for a ROCm 7.2 torch stack, which
    is the row docs/compatibility.rst describes."""
    m = re.search(r'7\.2\*\)\s*echo "([^"]+)"', text)
    assert m, "could not find the ROCm 7.2 SGLang wheel extra in install_baremetal.sh"
    return m.group(1)


def test_baremetal_defaults_match_compat_doc():
    sh = INSTALLER.read_text(encoding="utf-8")
    doc = COMPAT.read_text(encoding="utf-8")

    vllm_version = _default("VLLM_VERSION", sh)  # e.g. 0.29.0
    vllm_variant = _default("VLLM_ROCM_VARIANT", sh)  # e.g. rocm723
    sglang_ref = _default("SGLANG_REF", sh)  # e.g. v0.5.17
    sglang_rocm_extra = _sglang_extra_for_rocm72(sh)  # e.g. rocm724

    # compatibility.rst documents e.g. "v0.29.0 (rocm723)" and the pip spec "vllm==0.29.0+rocm723"; keep both in
    # lockstep with the script defaults.
    assert "v%s (%s)" % (vllm_version, vllm_variant) in doc, (
        "docs/compatibility.rst must document vLLM 'v%s (%s)' to match "
        "install_baremetal.sh defaults" % (vllm_version, vllm_variant)
    )
    assert "vllm==%s+%s" % (vllm_version, vllm_variant) in doc, (
        "docs/compatibility.rst pip spec must be 'vllm==%s+%s'" % (vllm_version, vllm_variant)
    )

    vllm_source_ref = _default("VLLM_SOURCE_REF", sh)
    assert vllm_source_ref[:12] in doc, (
        "docs/compatibility.rst must name the pinned vLLM source commit %s" % vllm_source_ref[:12]
    )

    assert not sglang_ref.startswith("v"), "SGLANG_REF is expected to pin a commit SHA (see docs/compatibility.rst)"
    assert sglang_ref[:12] in doc, "docs/compatibility.rst must name the pinned SGLang commit %s" % sglang_ref[:12]

    sglang_pretend = _default("SGLANG_PRETEND_VERSION", sh)
    assert sglang_pretend == "0.5.20", (
        "SGLANG_PRETEND_VERSION must name the release the pinned commit fits; got %s" % sglang_pretend
    )
    assert 'SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG="$SGLANG_PRETEND_VERSION"' in sh, (
        "install_baremetal.sh must export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG from SGLANG_PRETEND_VERSION"
    )
    assert "%s (rocm10)" % sglang_pretend in doc, (
        "docs/compatibility.rst must document SGLang '%s (rocm10)' for the validated docker stack" % sglang_pretend
    )
    assert "SGLANG_ROCM_EXTRA=%s" % sglang_rocm_extra in doc, (
        "docs/compatibility.rst must document SGLANG_ROCM_EXTRA=%s for ROCm 7.2.x bare-metal overrides"
        % sglang_rocm_extra
    )
