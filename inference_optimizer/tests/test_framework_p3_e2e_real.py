"""Opt-in real-vllm/sglang E2E test for the 5th Framework role (P3 PR-I).

This test runs the FULL framework_optimize -> framework_integrate
loop against a real vllm/sglang source tree. It is skipped by default
because:

* AST scan needs the framework source mounted (VLLM_SOURCE_ROOT or
  SGLANG_SOURCE_ROOT)
* framework_integrate needs a running server + Magpie benchmark + GPU
* The full loop takes ~17 min p50 / 60 min p75 per design §13.4

To run::

    pytest -m real_gpu inference_optimizer/tests/test_framework_p3_e2e_real.py

CI marks ``real_gpu`` skip; nightly runs unmark it on a dedicated
GPU node.
"""

from __future__ import annotations

import os

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("FRAMEWORK_AGENT_E2E_REAL", "0") != "1",
    reason="opt-in real-GPU E2E; set FRAMEWORK_AGENT_E2E_REAL=1 to enable",
)


@pytest.mark.real_gpu
def test_framework_optimize_real_vllm_min_10_flags():
    """Real vllm tree must surface >=10 flags through the AST scanner."""
    from framework_agent.agent.ast_scanner import scan_framework_args
    from framework_agent.agent.source_resolver import resolve_framework_sources

    resolved = resolve_framework_sources(("vllm",))
    assert "vllm" in resolved, (
        "VLLM_SOURCE_ROOT not set or container path missing"
    )
    result = scan_framework_args("vllm", resolved["vllm"])
    cli_flags = [f for f in result.flags if f.surface == "cli"]
    assert len(cli_flags) >= 10, (
        f"expected >=10 vllm CLI flags, got {len(cli_flags)}; "
        f"mode={result.mode} parse_failures={result.parse_failures}"
    )


@pytest.mark.real_gpu
def test_framework_optimize_real_sglang_min_10_flags():
    from framework_agent.agent.ast_scanner import scan_framework_args
    from framework_agent.agent.source_resolver import resolve_framework_sources

    resolved = resolve_framework_sources(("sglang",))
    assert "sglang" in resolved
    result = scan_framework_args("sglang", resolved["sglang"])
    cli_flags = [f for f in result.flags if f.surface == "cli"]
    assert len(cli_flags) >= 10, (
        f"expected >=10 sglang CLI flags, got {len(cli_flags)}"
    )


@pytest.mark.real_gpu
def test_framework_integrate_e2e_keep_path_real_server():
    """Full apply -> server restart -> Magpie -> accuracy gate -> KEEP.

    This is the design §13.4 verification: a single framework patch
    lands a >=3% throughput gain with <=1% accuracy drop on real
    vllm/sglang. The test is gated behind FRAMEWORK_AGENT_E2E_REAL=1
    AND a pre-cooked toy patch path (so we don't depend on the LLM
    proposer for the verification).

    Caller is responsible for setting:
      * FRAMEWORK_E2E_PATCH_PATH=/path/to/proposal.diff
      * FRAMEWORK_E2E_BASELINE_TPUT, _BASELINE_ACCURACY (floats)
    """
    pytest.skip(
        "Full P3 E2E orchestration not implemented in PR-I; requires "
        "real server lifecycle + Magpie wiring (cli.py hooks land "
        "alongside this test in a followup once the hooks are wired)."
    )
