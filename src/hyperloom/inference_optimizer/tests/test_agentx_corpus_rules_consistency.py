# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The corpus a model family replays is decided in two places; they must agree.

``aiperf_client.sh`` selects the corpus at runtime, mirroring upstream's
model-family whitelist. ``_workload_envs.build_agentx_workload_spec`` reports the
same decision to GEAK as ``canonical_corpus``, and the client stamps a workload
as non-canonical when the corpus it ran differs from that value. So a family
added to one side and not the other does not fail loudly -- it produces a run
that reports itself as a deviation from a corpus nobody chose, or a handoff that
names a corpus the client never loaded.

These tests execute the shell's own functions rather than restating them, so the
guard cannot rot into a copy of the thing it is checking. They also cover the
normalization, which is where the two implementations previously diverged: the
shell drops ``._-`` while the Python side used to drop every non-alphanumeric.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.executors._workload_envs import (
    AGENTX_CORPUS_256K,
    AGENTX_CORPUS_FULL,
    AGENTX_FULL_CONTEXT_FAMILIES,
    _agentx_default_corpus,
    _agentx_model_family,
)

_CLIENT = Path(__file__).resolve().parents[1] / "assets" / "agentx" / "aiperf_client.sh"

# Real identities, separator variants, and names that must NOT match a family.
_MODELS = [
    "/models/Kimi-K3",
    "Kimi-K3",
    "kimi_k3",
    "kimi.k3",
    "KIMIK3",
    "/models/DeepSeek-V4",
    "deepseek_v4-base",
    "/models/DSv4",
    "GLM-5.2",
    "glm52",
    "MiniMax-M3",
    "minimax.m3",
    "/models/gpt-oss-120b",
    "Qwen3-32B",
    "Llama-4-405B",
    "/a/b/c/Mixtral-8x22B",
    "kimi",
    "k3-kimi",
    "",
]


def _extract_shell_func(text: str, name: str) -> str:
    """Pull one ``name() { ... }`` block out of the client script."""
    match = re.search(rf"^{re.escape(name)}\(\) \{{$", text, re.MULTILINE)
    assert match, f"{name}() not found in {_CLIENT}; the guard needs updating"
    lines = text[match.start() :].splitlines()
    for idx, line in enumerate(lines):
        if idx and line == "}":
            return "\n".join(lines[: idx + 1])
    raise AssertionError(f"unterminated {name}() in {_CLIENT}")


def _shell_defaults(models: list[str]) -> list[str]:
    """Run the client's own ``_default_loader`` over *models*."""
    text = _CLIENT.read_text(encoding="utf-8")
    script = "\n".join(
        [
            "set -eu",
            _extract_shell_func(text, "_model_family"),
            _extract_shell_func(text, "_default_loader"),
            'for m in "$@"; do _default_loader "$m"; printf "\\n"; done',
        ]
    )
    out = subprocess.run(
        ["bash", "-c", script, "bash", *models],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return out.stdout.splitlines()


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash to run the client's own functions")
def test_python_and_client_choose_the_same_corpus_for_every_model():
    assert _shell_defaults(_MODELS) == [_agentx_default_corpus(m) for m in _MODELS]


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash to run the client's own functions")
def test_the_two_normalizations_agree_on_separator_variants():
    """``tr -d '._-'`` vs a Python strip: the case that used to differ."""
    text = _CLIENT.read_text(encoding="utf-8")
    script = "\n".join(
        [
            "set -eu",
            _extract_shell_func(text, "_model_family"),
            'for m in "$@"; do _model_family "$m"; printf "\\n"; done',
        ]
    )
    names = ["Kimi-K3", "kimi_k3", "kimi.k3", "/models/Kimi-K3", "GLM-5.2", "a-b_c.d"]
    out = subprocess.run(
        ["bash", "-c", script, "bash", *names],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert out.stdout.splitlines() == [_agentx_model_family(n) for n in names]


def test_the_whitelist_still_names_the_families_it_is_meant_to():
    """A family silently dropped from the tuple would fall back to 256k unnoticed."""
    for family in AGENTX_FULL_CONTEXT_FAMILIES:
        assert _agentx_default_corpus(f"/models/{family}-instruct") == AGENTX_CORPUS_FULL
    assert _agentx_default_corpus("/models/gpt-oss-120b") == AGENTX_CORPUS_256K


def test_preflight_accepts_exactly_the_corpora_the_rules_can_produce():
    """``preflight`` allowlists the client's self-selected corpora; keep them in step."""
    from hyperloom.inference_optimizer.agentx.preflight import _DEFAULT_CORPORA

    assert set(_DEFAULT_CORPORA) == {AGENTX_CORPUS_FULL, AGENTX_CORPUS_256K}
