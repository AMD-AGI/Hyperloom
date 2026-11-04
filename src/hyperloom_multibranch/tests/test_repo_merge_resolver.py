# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The integrator's resolver on a repository merge that split a conflicting patch by file."""

import json

from hyperloom.contracts.recipe import apply
from hyperloom_multibranch.contracts.llm import CompletionReply
from hyperloom_multibranch.contracts.llm_prices import TokenCounts
from hyperloom_multibranch.scheduling.integrator import LlmPatchResolver
from hyperloom_multibranch.tests.test_repo_merge import _built, _config, _on, _wide_onto


async def test_the_resolver_is_asked_about_the_overlapping_files_alone(tmp_path):
    repo, installed, head, candidate, merge = await _wide_onto(tmp_path)
    asked: list[str] = []
    rewrite = _on(_config(a="11", g="70"), _config(a="10", g="70"))

    async def completion(_system: str, prompt: str) -> CompletionReply:
        asked.append(prompt)
        return CompletionReply(json.dumps({"patch": rewrite}), TokenCounts(1, 1), None)

    resolved = await LlmPatchResolver(completion, repo, "model").resolve(merge, candidate)

    (to_rewrite,) = [prompt.split("Why it does not apply")[0] for prompt in asked]
    assert "config.py" in to_rewrite and "scheduler.py" not in to_rewrite
    built = _built(apply(await repo.read_recipe(head), resolved.delta).component("vllm"), installed, tmp_path)
    assert built == {"config.py": "".join(_config(a="11", g="70")), "scheduler.py": "policy = 'lpm'\n"}
