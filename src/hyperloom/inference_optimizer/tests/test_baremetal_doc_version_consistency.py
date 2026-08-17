# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Guard: install_baremetal.sh defaults stay in sync with docs/compatibility.rst.

Cheap, no-network consistency check. It does NOT verify that the
(version, variant) tuple actually exists on wheels.vllm.ai -- it only ensures
the script defaults and the documented recommendation do not silently drift
apart (the failure mode behind the 0.24.0 rocm722->rocm723 regression).
"""
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


def test_baremetal_defaults_match_compat_doc():
    sh = INSTALLER.read_text(encoding="utf-8")
    doc = COMPAT.read_text(encoding="utf-8")

    vllm_version = _default("VLLM_VERSION", sh)        # e.g. 0.27.1
    vllm_variant = _default("VLLM_ROCM_VARIANT", sh)   # e.g. rocm723
    sglang_ref = _default("SGLANG_REF", sh)            # e.g. v0.5.17

    # compatibility.rst documents e.g. "v0.27.1 (rocm723)" and the pip spec
    # "vllm==0.27.1+rocm723"; keep both in lockstep with the script defaults.
    assert "v%s (%s)" % (vllm_version, vllm_variant) in doc, (
        "docs/compatibility.rst must document vLLM 'v%s (%s)' to match "
        "install_baremetal.sh defaults" % (vllm_version, vllm_variant)
    )
    assert "vllm==%s+%s" % (vllm_version, vllm_variant) in doc, (
        "docs/compatibility.rst pip spec must be 'vllm==%s+%s'"
        % (vllm_version, vllm_variant)
    )

    sglang_version = sglang_ref.lstrip("v")
    assert "v%s (rocm" % sglang_version in doc, (
        "docs/compatibility.rst must document SGLang v%s" % sglang_version
    )
