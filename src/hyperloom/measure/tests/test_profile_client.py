# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The InferenceX checkout of a profile test is patched so the point's capture window and prompt count take effect."""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hyperloom.measure.profile_client import ProfileClientError, patch_profile_client
from hyperloom.measure.tests._inferencex import CLIENT, CLIENT_PATH, LIBRARY, LIBRARY_PATH, stock_checkout

#: The capture window a rendered profile point asks for, as the profile configuration carries it.
WINDOW = {
    "detailed_annotations": True,
    "start_step": 6080,
    "num_steps": 128,
    "merge_profiles": False,
    "profile_by_stage": False,
}
#: What a stock client sends whatever the point says: one step per stage, the stage traces merged.
BUILT_IN = {"num_steps": 1, "merge_profiles": True, "profile_by_stage": True}


def _sent_body(client_source: str, monkeypatch: pytest.MonkeyPatch, window: dict[str, Any] | None) -> dict[str, Any]:
    """Run the ``/start_profile`` request of ``client_source`` with the process environment a test sets; return its body."""
    if window is None:
        monkeypatch.delenv("PROFILE_EXTRA_BODY", raising=False)
    else:
        monkeypatch.setenv("PROFILE_EXTRA_BODY", json.dumps(window))
    sent: list[Any] = []

    async def request_func(request_func_input: Any) -> Any:
        sent.append(request_func_input)
        return SimpleNamespace(success=True)

    namespace: dict[str, Any] = {"RequestFuncInput": lambda **fields: SimpleNamespace(**fields)}
    exec(compile(client_source, "benchmark_serving.py", "exec"), namespace)
    args = dict(profile=True, model_id="m", model_name="m", test_prompt="p", base_url="http://x", test_prompt_len=1)
    args.update(test_output_len=1, logprobs=None, best_of=1, test_mm_content=None, ignore_eos=True)
    body = asyncio.run(namespace["benchmark"](request_func=request_func, **args))
    assert sent and body == sent[0].extra_body
    return body


def _prompt_count(library: str, tmp_path: Path, environment: dict[str, str]) -> str:
    """Run the profile block of ``library`` in bash at concurrency 64; return the prompt count it ends up with."""
    script = tmp_path / "lib.sh"
    script.write_text(library + "run_benchmark_serving 64\n", encoding="utf-8")
    done = subprocess.run(
        ["bash", str(script)], capture_output=True, text=True, check=True, env={"PATH": "/usr/bin:/bin", **environment}
    )
    return done.stdout.strip()


def test_a_stock_client_ignores_the_points_window_and_a_patched_one_sends_it(tmp_path, monkeypatch):
    root = stock_checkout(tmp_path / "ix")
    client = root / CLIENT_PATH

    assert _sent_body(client.read_text(), monkeypatch, WINDOW) == BUILT_IN, "the stock client drops the window"
    patch_profile_client(root)

    assert _sent_body(client.read_text(), monkeypatch, WINDOW) == WINDOW
    assert _sent_body(client.read_text(), monkeypatch, None) == BUILT_IN, "without a window the client is as before"


def test_a_stock_library_sends_one_request_per_slot_and_a_patched_one_takes_the_points_prompt_count(tmp_path):
    root = stock_checkout(tmp_path / "ix")
    profiling = {"PROFILE": "1", "NUM_PROMPTS": "776"}

    assert _prompt_count((root / LIBRARY_PATH).read_text(), tmp_path, profiling) == "64"
    patch_profile_client(root)
    library = (root / LIBRARY_PATH).read_text()

    assert _prompt_count(library, tmp_path, profiling) == "776"
    assert _prompt_count(library, tmp_path, {"PROFILE": "1"}) == "64", "a point without a count keeps the old cap"
    assert _prompt_count(library, tmp_path, {"NUM_PROMPTS": "776"}) == "", "outside profile mode nothing is read"


def test_patching_twice_changes_nothing_the_second_time(tmp_path):
    root = stock_checkout(tmp_path / "ix")
    patch_profile_client(root)
    once = {path: (root / path).read_text() for path in (LIBRARY_PATH, CLIENT_PATH)}

    patch_profile_client(root)

    assert {path: (root / path).read_text() for path in once} == once


@pytest.mark.parametrize(
    ("changed", "lacking"),
    [
        (
            {"library": LIBRARY.replace('num_prompts="$max_concurrency"', 'num_prompts="${max_concurrency}"')},
            "benchmark_lib.sh",
        ),
        ({"client": CLIENT.replace('"merge_profiles": True', '"merge_profiles": False')}, "benchmark_serving.py"),
    ],
)
def test_a_checkout_whose_lines_moved_is_refused_naming_the_file_that_lacks_the_patch(tmp_path, changed, lacking):
    root = stock_checkout(tmp_path / "ix", **changed)

    with pytest.raises(ProfileClientError, match=rf"{lacking} lack the patch"):
        patch_profile_client(root)


def test_a_checkout_without_the_client_is_refused(tmp_path):
    root = stock_checkout(tmp_path / "ix")
    (root / CLIENT_PATH).unlink()

    with pytest.raises(ProfileClientError, match="benchmark_serving.py lack the patch"):
        patch_profile_client(root)
