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


def test_baremetal_defaults_match_compat_doc():
    sh = INSTALLER.read_text(encoding="utf-8")
    doc = COMPAT.read_text(encoding="utf-8")

    vllm_version = _default("VLLM_VERSION", sh)  # e.g. 0.28.0
    vllm_variant = _default("VLLM_ROCM_VARIANT", sh)  # e.g. rocm723
    sglang_ref = _default("SGLANG_REF", sh)  # e.g. v0.5.17
    sglang_rocm_extra = _default("SGLANG_ROCM_EXTRA", sh)  # e.g. rocm724

    # compatibility.rst documents e.g. "v0.28.0 (rocm723)" and the pip spec "vllm==0.28.0+rocm723"; keep both in
    # lockstep with the script defaults.
    assert "v%s (%s)" % (vllm_version, vllm_variant) in doc, (
        "docs/compatibility.rst must document vLLM 'v%s (%s)' to match "
        "install_baremetal.sh defaults" % (vllm_version, vllm_variant)
    )
    assert "vllm==%s+%s" % (vllm_version, vllm_variant) in doc, (
        "docs/compatibility.rst pip spec must be 'vllm==%s+%s'" % (vllm_version, vllm_variant)
    )

    # SGLANG_REF is a commit, not a tag: upstream dropped a field the TraceLens
    # annotation patches need between this commit and v0.5.18. The doc must name
    # the exact ref so moving the pin cannot leave the matrix behind.
    assert not sglang_ref.startswith("v"), (
        "SGLANG_REF is expected to pin a commit; a tag reintroduces the patch "
        "mismatch this pin exists to avoid (see docs/compatibility.rst)"
    )
    assert sglang_ref[:12] in doc, "docs/compatibility.rst must name the pinned SGLang commit %s" % sglang_ref[:12]

    # An untagged commit gives setuptools_scm nothing to derive from, so the build
    # falls back to 0.0.0.* and the patch sets are refused on the version gate.
    # The declared version travels with the pin and must name the patch set.
    sglang_pretend = _default("SGLANG_PRETEND_VERSION", sh)
    assert sglang_pretend == "0.5.18", (
        "SGLANG_PRETEND_VERSION must name the patch set the pinned commit fits; got %s" % sglang_pretend
    )
    assert 'SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG="$SGLANG_PRETEND_VERSION"' in sh, (
        "install_baremetal.sh must export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG from SGLANG_PRETEND_VERSION"
    )
    assert "0.5.18 (%s)" % sglang_rocm_extra in doc, (
        "docs/compatibility.rst must document SGLang '0.5.18 (%s)' to match "
        "install_baremetal.sh defaults" % sglang_rocm_extra
    )
    assert "SGLANG_ROCM_EXTRA=%s" % sglang_rocm_extra in doc, (
        "docs/compatibility.rst must document SGLANG_ROCM_EXTRA=%s" % sglang_rocm_extra
    )
