# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Excerpts of the stock InferenceX checkout a runner holds, as the profile tests need them.

``LIBRARY`` is the profile block of ``benchmarks/benchmark_lib.sh`` and ``CLIENT`` the ``/start_profile`` request of
``utils/bench_serving/benchmark_serving.py``, both copied unchanged from the commit the sessions pin.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

LIBRARY = """\
run_benchmark_serving() {
    local num_prompts=""
    local max_concurrency="$1"

    # Profiling support: when PROFILE=1, ensure profiler dir exists, add --profile flag,
    # and cap num_prompts to keep traces small.
    local profile_flag=()
    if [[ "${PROFILE:-}" == "1" ]]; then
        local _prof_dir="${SGLANG_TORCH_PROFILER_DIR:-${VLLM_TORCH_PROFILER_DIR:-}}"
        if [[ -n "$_prof_dir" ]]; then
            mkdir -p "$_prof_dir"
        fi
        profile_flag+=(--profile)
        num_prompts="$max_concurrency"
    fi

    echo "$num_prompts"
}
"""

CLIENT = """\
async def benchmark(profile, model_id, model_name, test_prompt, base_url, test_prompt_len, test_output_len,
                    logprobs, best_of, test_mm_content, ignore_eos, request_func):
    if profile:
        print("Starting profiler...")
        profile_input = RequestFuncInput(model=model_id,
                                         model_name=model_name,
                                         prompt=test_prompt,
                                         api_url=base_url + "/start_profile",
                                         prompt_len=test_prompt_len,
                                         output_len=test_output_len,
                                         extra_body={"num_steps": 1, "merge_profiles": True, "profile_by_stage": True},
                                         logprobs=logprobs,
                                         best_of=best_of,
                                         multi_modal_content=test_mm_content,
                                         ignore_eos=ignore_eos)
        profile_output = await request_func(request_func_input=profile_input)
        if profile_output.success:
            print("Profiler started")
        return profile_input.extra_body
"""

LIBRARY_PATH = Path("benchmarks") / "benchmark_lib.sh"
CLIENT_PATH = Path("utils") / "bench_serving" / "benchmark_serving.py"


def stock_checkout(root: Path, *, library: str = LIBRARY, client: str = CLIENT) -> Path:
    """An InferenceX tree holding the two files as upstream ships them."""
    for relative, text in ((LIBRARY_PATH, library), (CLIENT_PATH, client)):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def stock_repo(path: Path) -> str:
    """A git repository of the stock tree, the way a runner fetches it by hash; returns its commit."""
    stock_checkout(path)
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    for args in (["init", "-q"], ["add", "."], ["commit", "-q", "-m", "x"]):
        subprocess.run(["git", *args], cwd=path, check=True, env={**env, "PATH": "/usr/bin:/bin"})
    done = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True)
    return done.stdout.strip()
