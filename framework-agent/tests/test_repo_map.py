"""Cover the canonical repo-URL mapping that framework-agent owns.

Lives here (not in inference_optimizer/tests) so the test runs even when
inference_optimizer isn't on PYTHONPATH — the whole point of moving
``repo_url_for_framework`` to ``framework_agent.repo_map`` was so the
standalone ``fa`` CLI doesn't reverse-import the orchestrator side.
"""

from __future__ import annotations

from framework_agent.repo_map import repo_url_for_framework


def test_repo_url_for_framework_known():
    assert repo_url_for_framework("sglang") == (
        "https://github.com/sgl-project/sglang.git"
    )
    assert repo_url_for_framework("vllm") == (
        "https://github.com/ROCm/vllm.git"
    )


def test_repo_url_for_framework_lowercases_and_strips():
    assert repo_url_for_framework("SGLang") == (
        "https://github.com/sgl-project/sglang.git"
    )
    assert repo_url_for_framework("  vllm  ") == (
        "https://github.com/ROCm/vllm.git"
    )


def test_repo_url_for_framework_unknown_returns_empty():
    assert repo_url_for_framework("rust-burn") == ""
    assert repo_url_for_framework("") == ""
    assert repo_url_for_framework("   ") == ""
