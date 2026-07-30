"""Tests for KB framework resolution + its fault tolerance (soft slug input)."""

from __future__ import annotations

import sys
from pathlib import Path

_BACKENDS_DIR = Path(__file__).resolve().parent.parent / "tools" / "backends"
sys.path.insert(0, str(_BACKENDS_DIR))
import forge_submit  # noqa: E402


def test_resolve_framework_from_explicit_candidate_field():
    assert forge_submit._resolve_framework({"framework": "vLLM"}) == "vllm"
    assert forge_submit._resolve_framework({"backend": "sglang"}) == "sglang"


def test_resolve_framework_ignores_language_backends():
    # 'triton' is a backend LANGUAGE, not a framework — must not become framework.
    assert forge_submit._resolve_framework({"backend": "triton"}) == ""
    assert forge_submit._resolve_framework({"framework": "triton"}) == ""


def test_resolve_framework_from_kernel_path_when_candidate_silent():
    fw = forge_submit._resolve_framework(
        {}, "/ws/worktree/vllm/model_executor/layers/fused_moe/x.py")
    assert fw == "vllm"


def test_resolve_framework_returns_empty_when_unknown():
    # Unresolvable -> "" so the caller OMITS --framework and forge-loop infers.
    # Fault tolerance: never raises, never guesses a wrong framework.
    assert forge_submit._resolve_framework({}, "/tmp/scratch/kernel.py") == ""
    assert forge_submit._resolve_framework(None, "") == ""
    assert forge_submit._resolve_framework({"framework": None, "backend": None}) == ""


def test_resolve_framework_explicit_beats_path():
    # An explicit recognized framework wins over a (possibly flattened) path.
    fw = forge_submit._resolve_framework({"framework": "sglang"}, "/x/vllm/y/k.py")
    assert fw == "sglang"
